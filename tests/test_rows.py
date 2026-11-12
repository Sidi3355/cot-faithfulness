import json

import numpy as np
import pandas as pd
import pytest

from cotfaith import datasets
from cotfaith import rows as rows_lib

LABELS = ("0", "1")


def _row(ex, row_type, label, probs, sample=0, budget="none", word=None, idx=None, text=None, gold="0"):
  return dict(example_id=ex, row_type=row_type, label=label, sample_idx=sample, budget=budget,
              label_probs=json.dumps(probs) if probs is not None else None, inserted_word=word,
              intervention_idx=idx, explanation=text, gold_label=gold)


def test_greedy_flips_and_tvd():
  df = pd.DataFrame([
      _row(1, "base", "0", [0.8, 0.2]),
      _row(1, "n1", "0", [0.8, 0.2]),
      _row(1, "real", "1", [0.3, 0.7], word="red", idx=0, text="It is red."),
      _row(1, "real", None, None, word="old", idx=1, text=None),
      _row(2, "base", "1", [0.4, 0.6], gold="0"),
      _row(2, "real", "1", [0.4, 0.6], word="big", idx=0, text="nothing"),
  ])
  out = rows_lib.add_flips(df, LABELS)
  assert out.loc[1, "I"] == 0.0  # N1 matches the base.
  assert out.loc[2, "I"] == 1.0 and np.isnan(out.loc[3, "I"]) and out.loc[5, "I"] == 0.0
  assert out.loc[2, "tvd"] == pytest.approx(0.5)
  assert out.loc[0, "correct"] == 1.0 and out.loc[4, "correct"] == 0.0
  assert np.isnan(out.loc[2, "correct"])


def test_sampled_flips_use_modal_base_per_budget():
  rows = []
  for b, base_labels in (("128", ["0", "0", "1"]), ("uncapped", ["1", "1", "0"])):
    for s, l in enumerate(base_labels):
      rows.append(_row(1, "base", l, [0.6, 0.4] if l == "0" else [0.3, 0.7], sample=s, budget=b))
    rows.append(_row(1, "real", "1", [0.2, 0.8], budget=b, word="x", idx=0, text="t"))
  out = rows_lib.add_flips(pd.DataFrame(rows), LABELS)
  real = out[out.row_type == "real"].set_index("budget")
  assert real.loc["128", "I"] == 1.0  # Modal base "0".
  assert real.loc["uncapped", "I"] == 0.0  # Modal base "1".
  mean_base = np.mean([[0.6, 0.4], [0.6, 0.4], [0.3, 0.7]], axis=0)
  assert real.loc["128", "tvd"] == pytest.approx(0.5 * np.abs(np.array([0.2, 0.8]) - mean_base).sum())


def test_mentions_and_inputs():
  spec = datasets.COMVE
  ints = pd.DataFrame([{"data_idx": 1, "intervention_idx": 0, "sent0": "a red car", "sent1": "a car flies"}])
  twins = pd.DataFrame([{"data_idx": 1, "intervention_idx": 0, "sent0": "a  car", "sent1": "a car flies"}])
  df = pd.DataFrame([
      _row(1, "base", "0", [0.9, 0.1]),
      _row(1, "real", "1", [0.1, 0.9], word="red", idx=0, text="SENTENCE 0: a red car is fine; red is odd"),
      _row(1, "n2", "0", [0.9, 0.1], word="red", idx=0, text="nothing"),
  ])
  inputs = rows_lib.input_texts(df, spec, ints, twins)
  assert inputs.iloc[0] is None and "red" in inputs.iloc[1] and "red" not in inputs.iloc[2]
  out = rows_lib.add_mentions(df, "explanation", inputs)
  assert out.loc[1, "E_paper"] == 1.0 and out.loc[1, "E_novel"] == 1.0
  assert out.loc[2, "E_paper"] == 0.0
  assert np.isnan(out.loc[0, "E_paper"])
