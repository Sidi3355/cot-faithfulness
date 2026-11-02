"""Runs pilot/scripts/analyze.py end to end on synthetic runner rows."""

import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import upstream
from cotfaith import config, datasets

pytestmark = pytest.mark.data
SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "pilot" / "scripts"


def _load(name):
  spec = importlib.util.spec_from_file_location(f"pilot_{name}", SCRIPTS / f"{name}.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def _fake_rows(cfg: config.RunConfig, test: pd.DataFrame, ints: pd.DataFrame, rng) -> pd.DataFrame:
  examples = test.sample(frac=1, random_state=0).index[: cfg.n_examples]
  budgets = [str(b) for b in cfg.budgets] + ["uncapped"] if cfg.thinking else ["none"]
  rows = []
  for ex in examples:
    items = [("base", "base", None, None)]
    if cfg.controls.n1:
      items.append(("n1", "n1", None, None))
    for _, r in ints[ints.data_idx == ex].iterrows():
      items.append((f"real{r.intervention_idx}", "real", r.intervention_idx, r.inserted_word))
      if cfg.controls.n2:
        items.append((f"n2{r.intervention_idx}", "n2", r.intervention_idx, r.inserted_word))
    for key, kind, idx, word in items:
      for s in range(cfg.decoding.n):
        for b in budgets:
          p1 = float(rng.uniform(0.05, 0.95))
          label = "1" if rng.random() < p1 else "0"
          mention = word is not None and rng.random() < (0.6 if label == "1" else 0.3)
          text = f"Because the sentence says {word}." if mention else "Because it is odd."
          rows.append(dict(
              example_id=int(ex), gold_label=str(test.loc[ex, "false_idx"]), item=key, row_type=kind,
              intervention_idx=idx, inserted_word=word, modified_key="sent0" if idx is not None else None,
              sample_idx=s, budget=b, n_prompt_tokens=600, answer_text=f"FALSE SENTENCE: {label}", label=label,
              explanation=None if cfg.thinking else text, parse_error=None,
              label_probs=json.dumps([1 - p1, p1]), label_mass=1.0, forced=b != "uncapped" and cfg.thinking,
              finished=True, n_new_tokens=30, trace=text if cfg.thinking else None,
              trace_n_tokens=min(int(b), 200) if b.isdigit() else 200, terminated=True,
          ))
  return pd.DataFrame(rows)


def test_analysis_end_to_end(tmp_path, monkeypatch, splits):
  test = splits("comve")["test"]
  # Synthetic interventions: 11 per example for the first 40 examples.
  order = test.sample(frac=1, random_state=0).index[:40]
  ints = pd.DataFrame([
      dict(test.loc[ex].to_dict(), data_idx=ex, intervention_idx=k, modified_sentence_key="sent0",
           inserted_word=f"word{k}", inserted_position=0, token_char_offset=0)
      for ex in order for k in range(11)
  ])
  ints_path = tmp_path / "natural_comve.parquet"
  ints.to_parquet(ints_path)
  ints_path.with_suffix(".sha256").write_text(hashlib.sha256(ints_path.read_bytes()).hexdigest() + "\n")

  make_configs = _load("make_configs")
  monkeypatch.setattr(make_configs, "INTERVENTIONS", ints_path)
  analyze = _load("analyze")
  for name, value in (("RAW", tmp_path / "raw"), ("OUT", tmp_path / "out"), ("CONFIGS", tmp_path / "configs"),
                      ("INTERVENTIONS", ints_path)):
    monkeypatch.setattr(analyze, name, value)
  (tmp_path / "configs").mkdir()
  rng = np.random.default_rng(0)
  for name, cfg in make_configs.configs(r_examples=20).items():
    (tmp_path / "configs" / f"{name}.yaml").write_text(config.dump(cfg))
    rows_dir = cfg.results_dir(tmp_path / "raw") / "rows"
    rows_dir.mkdir(parents=True)
    frame = _fake_rows(cfg, test, ints, rng)
    for ex, g in frame.groupby("example_id"):
      g.to_parquet(rows_dir / f"{ex}.parquet")

  data = analyze.Data()
  analyze.prepilot(data)
  analyze.arm_r(data)
  analyze.arm_c(data)
  out = tmp_path / "out"
  prepilot = pd.read_csv(out / "R" / "prepilot_sample_size.csv")
  assert prepilot["chosen_N"].iloc[0] in analyze.N_CANDIDATES
  cells = pd.read_csv(out / "R" / "cells.csv")
  assert len(cells) == 15 and cells["phi_cct"].notna().all()
  assert (cells["phi_cct_lo"] <= cells["phi_cct"]).all() and (cells["phi_cct"] <= cells["phi_cct_hi"]).all()
  for f in ("f_auroc.csv", "F1_verbosity.csv", "F2_scale.csv", "F3_phi_vs_cct.csv", "paper_comparison.csv",
            "tpr_fpr_by_length.png", "metrics_by_size.png", "phi_vs_cct.png"):
    assert (out / "R" / f).exists(), f
  c2 = pd.read_csv(out / "C" / "C2_cells_by_budget.csv")
  assert list(c2["budget"].astype(str)) == ["128", "256", "512", "1024", "uncapped"]
  assert "excess_flip_rate" in c2
  for f in ("C1_cells.csv", "C1_f_auroc.csv", "C2_f_auroc.csv", "C2_dummy_explainers.csv", "C2_copy_rates.csv",
            "C2_failure_sensitivity.csv", "C3A_cells.csv", "C3A_f_auroc.csv", "C3R_reference.csv",
            "trace_vs_explanation_curve_test.csv", "trace_vs_explanation_regression.csv",
            "tpr_fpr_explanation_vs_trace.png", "phi_vs_accuracy_trace_vs_explanation.png"):
    assert (out / "C" / f).exists(), f
  dummies = pd.read_csv(out / "C" / "C2_dummy_explainers.csv")
  copier = dummies[dummies.explainer.str.startswith("copier")]
  assert (copier.tpr_ct == 1).all() and copier.phi_cct.isna().all()
