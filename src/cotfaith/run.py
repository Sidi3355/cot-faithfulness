"""Runs one config on the CPU engine and writes its results directory.

For every test example in scope, the runner builds the prompts for the base
example, its N1 rerun, each intervention and each N2 twin; generates (greedy,
or K samples); parses every output; and, for thinking runs, derives every
budget from the one uncapped generation (PLAN.md section 4.4).

One parquet file is written per example (rows/<example_id>.parquet), so an
interrupted run resumes where it stopped. Metrics are computed later from these
rows (pilot/scripts/analyze.py); nothing here depends on a metric definition.
"""

import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import pickle
import subprocess
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch

from cotfaith import budget, config as config_lib, datasets, interventions, parse, prompts, tokens
from cotfaith.hf_engine import HFEngine, Output, Sampling, label_token_index

FORCED_ANSWER_TOKENS = 64
# Forced answers per generate call (and per checkpoint) in thinking runs.
FORCED_CHUNK = 8
ROW_TYPES = ("base", "n1", "real", "n2")


@dataclasses.dataclass(frozen=True)
class Item:
  key: str
  row_type: str
  row: pd.Series
  intervention_idx: int | None = None
  inserted_word: str | None = None
  modified_key: str | None = None


def sha256(path: str | pathlib.Path) -> str:
  return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def examples_in_scope(cfg: config_lib.RunConfig, test_df: pd.DataFrame) -> pd.DataFrame:
  """The paper's fixed evaluation order; the pilot takes its first n_examples."""
  ordered = test_df.sample(frac=1, random_state=0)
  return ordered.iloc[: cfg.n_examples] if cfg.pilot else ordered


def example_items(data_idx, test_row: pd.Series, ints: pd.DataFrame, twins: pd.DataFrame,
                  controls: config_lib.Controls) -> list[Item]:
  items = [Item("base", "base", test_row)]
  if controls.n1:
    items.append(Item("n1", "n1", test_row))
  for kind, frame in (("real", ints), ("n2", twins if controls.n2 else None)):
    if frame is None:
      continue
    for _, r in frame.iterrows():
      items.append(Item(
          f"{kind}{int(r['intervention_idx'])}", kind, r, int(r["intervention_idx"]),
          r["inserted_word"], r["modified_sentence_key"],
      ))
  return items


def _sampling(cfg: config_lib.RunConfig) -> Sampling:
  d = cfg.decoding
  if d.mode == "greedy":
    return Sampling(greedy=True, repetition_penalty=d.repetition_penalty)
  return Sampling(greedy=False, temperature=d.temperature, top_p=d.top_p, top_k=d.top_k,
                  min_p=d.min_p, repetition_penalty=d.repetition_penalty)


class Runner:
  def __init__(self, cfg: config_lib.RunConfig, data_dir: str, results_root: str, threads: int = 4,
               batch_size: int = 16, log: Callable[[str], None] = print, dtype=None):
    if cfg.engine != "hf_cpu":
      raise ValueError("this runner implements the hf_cpu engine")
    self.cfg, self.log, self.batch_size = cfg, log, batch_size
    self.spec = datasets.get_spec(cfg.dataset)
    splits = datasets.load_splits(cfg.dataset, data_dir)
    self.train, self.test = splits["train"], splits["test"]
    if not cfg.smoke and sha256(cfg.interventions_path) != cfg.interventions_sha256:
      raise ValueError("intervention file does not match interventions_sha256")
    self.ints = pd.read_parquet(cfg.interventions_path)
    self.twins = interventions.n2_twins(self.ints, self.test)
    self.out_dir = cfg.results_dir(results_root)
    (self.out_dir / "rows").mkdir(parents=True, exist_ok=True)
    self.engine = HFEngine(cfg.model, threads=threads, dtype=dtype or getattr(torch, cfg.dtype))
    self.tok = self.engine.tokenizer
    self.label_tokens = {
        t: self.spec.class_labels.index(l)
        for t, l in tokens.prefix_tokens_for_labels(self.tok, self.spec.class_labels).items()
    }
    self.sampling = _sampling(cfg)
    self.thinking_flag = cfg.thinking if self.engine.info.hybrid_thinking else None
    if cfg.thinking:
      self.think_start, self.think_end = tokens.think_token_ids(self.tok)
      self.early_exit_ids = self.engine.encode(budget.EARLY_EXIT)
    self._write_meta()

  # Bookkeeping ---------------------------------------------------------------

  def _write_meta(self):
    (self.out_dir / config_lib.CONFIG_FILE).write_text(config_lib.dump(self.cfg))
    meta_path = self.out_dir / config_lib.META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("started", time.strftime("%Y-%m-%dT%H:%M:%S"))
    meta.update(
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        engine="hf_cpu", dtype=str(self.engine.model.dtype).replace("torch.", ""),
        versions={p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "pandas")},
        model_revision=self.engine.info.revision,
        interventions_sha256=sha256(self.cfg.interventions_path),
        n_label_tokens=len(self.label_tokens),
        label_scores="processed (after default processors)" if self.sampling.greedy else "raw logits",
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

  def _row_path(self, data_idx) -> pathlib.Path:
    return self.out_dir / "rows" / f"{data_idx}.parquet"

  # Main loop -----------------------------------------------------------------

  def run(self, max_examples: int | None = None) -> None:
    examples = examples_in_scope(self.cfg, self.test)
    done = 0
    for data_idx, test_row in examples.iterrows():
      if self._row_path(data_idx).exists():
        continue
      if max_examples is not None and done >= max_examples:
        break
      start = time.time()
      rows = self.run_example(data_idx, test_row)
      frame = pd.DataFrame(rows)
      tmp = self._row_path(data_idx).with_suffix(".tmp")
      frame.to_parquet(tmp)
      os.replace(tmp, self._row_path(data_idx))
      self._partial_path(data_idx).unlink(missing_ok=True)
      secs = time.time() - start
      with open(self.out_dir / "timing.csv", "a") as f:
        f.write(f"{data_idx},{len(frame)},{secs:.1f}\n")
      done += 1
      self.log(f"{self.out_dir.name} example {data_idx}: {len(frame)} rows in {secs:.0f}s")

  def run_example(self, data_idx, test_row) -> list[dict]:
    ints = self.ints[self.ints["data_idx"] == data_idx]
    twins = self.twins[self.twins["data_idx"] == data_idx]
    items = example_items(data_idx, test_row, ints, twins, self.cfg.controls)
    prompt_ids = [
        self.engine.encode_chat(
            prompts.build_prompt(self.spec, self.train, data_idx, it.row, self.cfg.template,
                                 self.cfg.demos, self.cfg.length),
            self.thinking_flag,
        )
        for it in items
    ]
    seed = self.cfg.seed * 100_003 + int(data_idx)
    common = dict(example_id=int(data_idx), gold_label=self.spec.true_label(test_row))
    if self.cfg.thinking:
      return self._thinking_example(data_idx, items, prompt_ids, common, seed)
    outs = self.engine.generate(
        prompt_ids, n=self.cfg.decoding.n, sampling=self.sampling, max_new_tokens=self.cfg.max_new_tokens,
        label_tokens=self.label_tokens, n_labels=len(self.spec.class_labels), seed=seed,
        batch_size=self.batch_size, chunk=64,
    )
    rows = []
    for it, p_ids, samples in zip(items, prompt_ids, outs):
      for s, o in enumerate(samples):
        rows.append(dict(common, **self._item_fields(it), sample_idx=s, budget="none",
                         n_prompt_tokens=len(p_ids), **self._answer_fields(o.token_ids, o, 0), forced=False,
                         finished=o.finished, n_new_tokens=len(o.token_ids), trace=None,
                         trace_n_tokens=None, terminated=None))
    return rows

  # Output parsing ------------------------------------------------------------

  @staticmethod
  def _item_fields(it: Item) -> dict:
    return dict(item=it.key, row_type=it.row_type, intervention_idx=it.intervention_idx,
                inserted_word=it.inserted_word, modified_key=it.modified_key)

  def _answer_fields(self, answer_ids: list[int], o: Output, step_offset: int) -> dict:
    """Parse the answer tokens; label probabilities at the label token."""
    text = self.tok.decode(answer_ids, skip_special_tokens=True)
    ans = parse.parse_answer(self.spec, self.cfg.template, text)
    probs, idx = None, None
    if ans.label is not None:
      idx = label_token_index(self.tok, answer_ids, text, ans.label_start)
      if idx is not None and step_offset + idx < len(o.label_probs):
        probs = o.label_probs[step_offset + idx].astype(float).tolist()
    return dict(
        answer_text=text, label=ans.label, explanation=ans.explanation, parse_error=ans.error,
        label_probs=json.dumps(probs) if probs is not None else None,
        label_mass=float(np.sum(probs)) if probs is not None else None,
    )

  # Thinking budgets ----------------------------------------------------------

  def _partial_path(self, data_idx) -> pathlib.Path:
    return self.out_dir / "rows" / f"{data_idx}.partial.pkl"

  def _thinking_example(self, data_idx, items, prompt_ids, common, seed) -> list[dict]:
    """Thinking runs take long per example, so progress is checkpointed after
    every group of traces and every chunk of forced answers. Seeds depend only on
    the group or chunk index, so a resumed example samples as an uninterrupted one."""
    n, cap = self.cfg.decoding.n, self.cfg.thinking_cap
    budgets = [*self.cfg.budgets, cap]
    path = self._partial_path(data_idx)
    state = pickle.loads(path.read_bytes()) if path.exists() else {"outs": {}, "forced": {}}

    def save():
      tmp = path.with_suffix(".pkltmp")
      tmp.write_bytes(pickle.dumps(state))
      os.replace(tmp, path)

    gen = dict(sampling=self.sampling, label_tokens=self.label_tokens, n_labels=len(self.spec.class_labels))
    per_call = max(1, math.ceil(self.batch_size / n))
    for g, start in enumerate(range(0, len(items), per_call)):
      group = list(range(start, min(start + per_call, len(items))))
      if all((i, 0) in state["outs"] for i in group):
        continue
      res = self.engine.generate([prompt_ids[i] for i in group], n=n, max_new_tokens=self.cfg.max_new_tokens,
                                 seed=seed + 1000 * g, batch_size=len(group) * n, chunk=256, **gen)
      for i, samples in zip(group, res):
        for s, o in enumerate(samples):
          state["outs"][i, s] = o
      save()
      self.log(f"  example {data_idx}: traces for {min(start + per_call, len(items))}/{len(items)} items")
    outs = [[state["outs"][i, s] for s in range(n)] for i in range(len(items))]

    plans, forced_prompts, forced_keys = {}, [], []
    for i, samples in enumerate(outs):
      for s, o in enumerate(samples):
        span = budget.thinking_span(o.token_ids, self.think_start, self.think_end)
        plans[i, s] = span
        for b in budgets:
          if span.n_tokens > b or not span.terminated:
            keep = min(b, span.n_tokens)
            forced_prompts.append(list(prompt_ids[i]) + o.token_ids[: span.start + keep] + self.early_exit_ids)
            forced_keys.append((i, s, b))
    for c, start in enumerate(range(0, len(forced_prompts), FORCED_CHUNK)):
      keys = forced_keys[start:start + FORCED_CHUNK]
      if all(k in state["forced"] for k in keys):
        continue
      res = self.engine.generate(forced_prompts[start:start + FORCED_CHUNK], n=1,
                                 max_new_tokens=FORCED_ANSWER_TOKENS, seed=seed + 7 + 1000 * c,
                                 batch_size=FORCED_CHUNK, chunk=FORCED_ANSWER_TOKENS, **gen)
      for k, r in zip(keys, res):
        state["forced"][k] = r[0]
      save()
    forced = state["forced"]
    return self._thinking_rows(items, prompt_ids, outs, plans, forced, common)

  def _thinking_rows(self, items, prompt_ids, outs, plans, forced, common) -> list[dict]:
    cap = self.cfg.thinking_cap
    budgets = [*self.cfg.budgets, cap]
    rows = []
    for i, samples in enumerate(outs):
      for s, o in enumerate(samples):
        span = plans[i, s]
        for b in budgets:
          name = "uncapped" if b == cap else str(b)
          keep = min(b, span.n_tokens)
          trace = self.tok.decode(o.token_ids[span.start: span.start + keep], skip_special_tokens=True)
          if (i, s, b) in forced:
            f = forced[i, s, b]
            answer = self._answer_fields(f.token_ids, f, 0)
            extra = dict(forced=True, finished=f.finished, n_new_tokens=len(f.token_ids))
          else:
            answer_ids = o.token_ids[span.end + 1:]
            answer = self._answer_fields(answer_ids, o, span.end + 1)
            extra = dict(forced=False, finished=o.finished, n_new_tokens=len(answer_ids))
          rows.append(dict(
              common, **self._item_fields(items[i]), sample_idx=s, budget=name,
              n_prompt_tokens=len(prompt_ids[i]), **answer, **extra, trace=trace, trace_n_tokens=keep,
              terminated=span.terminated and span.n_tokens <= cap,
          ))
    return rows


def load_rows(results_dir: str | pathlib.Path) -> pd.DataFrame:
  files = sorted(pathlib.Path(results_dir, "rows").glob("*.parquet"))
  return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
