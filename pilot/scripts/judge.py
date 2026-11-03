"""Naturalness judge for the pilot: scores candidates and freezes the interventions.

Scores the candidates of the first N ComVE test examples (the paper's fixed
order) with the pilot judge, using the paper's naturalness prompt and its
yes/no scoring of the first-token distribution, then keeps the top 5% per
example. Scores are saved per example, so the run resumes after interruption.

Usage: python pilot/scripts/judge.py [--n-examples 200]
Outputs:
  data/pilot/judge/<example_id>.parquet      scores for every candidate
  data/pilot/natural_comve.parquet           frozen interventions (+ .sha256)
  pilot/results/interventions.csv            the same, for review
"""

import argparse
import hashlib
import json
import os
import pathlib
import time

import pandas as pd
import torch

from cotfaith import datasets, interventions, models
from cotfaith.hf_engine import HFEngine

ROOT = pathlib.Path(__file__).resolve().parents[2]
JUDGE = "Qwen/Qwen2.5-3B-Instruct"


def score_example(engine: HFEngine, cands: pd.DataFrame, test_row: pd.Series) -> pd.DataFrame:
  out = []
  # Prompts sharing the original sentence share a long prefix: score them together.
  for key, group in cands.groupby("modified_sentence_key", sort=False):
    prompt_text = [interventions.naturalness_prompt(test_row[key], r[key]) for _, r in group.iterrows()]
    prompt_ids = [engine.encode_chat(t, None) for t in prompt_text]
    tops = engine.next_token_topk(prompt_ids, k=20)
    g = group.copy()
    probs = [interventions.yes_no_probs(t) for t in tops]
    g["intervention_naturalness_yes_prob"] = [p["yes"] for p in probs]
    g["intervention_naturalness_no_prob"] = [p["no"] for p in probs]
    g["intervention_naturalness_invalid_prob"] = [p["invalid"] for p in probs]
    out.append(g)
  return pd.concat(out).sort_values("intervention_idx")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-examples", type=int, default=200)
  parser.add_argument("--threads", type=int, default=4)
  # The 167 examples scored so far used bfloat16; keep one format for all.
  parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
  args = parser.parse_args()
  data_dir = ROOT / "data"
  judge_dir = data_dir / "pilot" / "judge"
  judge_dir.mkdir(parents=True, exist_ok=True)

  test = datasets.load_splits("comve", str(data_dir))["test"]
  order = test.sample(frac=1, random_state=0).iloc[: args.n_examples]
  cands = pd.read_parquet(data_dir / "interventions" / "candidates_comve.parquet")
  engine = None
  for data_idx, test_row in order.iterrows():
    path = judge_dir / f"{data_idx}.parquet"
    ex = cands[cands["data_idx"] == data_idx]
    if path.exists() or ex.empty:
      continue
    engine = engine or HFEngine(JUDGE, threads=args.threads, dtype=getattr(torch, args.dtype))
    start = time.time()
    scored = score_example(engine, ex, test_row)
    scored.to_parquet(path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)
    print(f"judge example {data_idx}: {len(scored)} candidates in {time.time() - start:.0f}s", flush=True)

  assessed = pd.concat([pd.read_parquet(judge_dir / f"{i}.parquet") for i in order.index
                        if (judge_dir / f"{i}.parquet").exists()])
  natural = interventions.select_natural(assessed)
  out = data_dir / "pilot" / "natural_comve.parquet"
  natural.to_parquet(out)
  digest = hashlib.sha256(out.read_bytes()).hexdigest()
  out.with_suffix(".sha256").write_text(digest + "\n")
  review = ROOT / "pilot" / "results" / "interventions.csv"
  review.parent.mkdir(parents=True, exist_ok=True)
  natural[["data_idx", "intervention_idx", "modified_sentence_key", "inserted_word", "inserted_position",
           "sent0", "sent1", "intervention_naturalness_yes_prob"]].to_csv(review, index=False)
  meta = dict(
      judge=JUDGE, judge_revision=models.get(JUDGE).revision, judge_dtype=args.dtype,
      n_examples_judged=int(assessed["data_idx"].nunique()), n_candidates=len(assessed),
      keep_top_frac=interventions.KEEP_TOP_FRAC, n_interventions=len(natural),
      n_examples_with_interventions=int(natural["data_idx"].nunique()), sha256=digest,
  )
  (review.parent / "interventions_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
  print(f"{len(natural)} interventions from {natural['data_idx'].nunique()} examples; sha256 {digest}")


if __name__ == "__main__":
  main()
