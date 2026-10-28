"""Flip indicators (I), per-example statistics, dummy explainers.

Greedy runs (Phase 1): I = 1 if the edited run's label differs from the base
run's label. Sampled runs (Phases 2-3): one row per sampled trace, with
I_k = 1 if sample k's label differs from the modal label of the K base samples
(ties broken by the higher mean label-token probability). A missing label on
either side leaves I undefined (NaN).
"""

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from cotfaith import metrics


def greedy_flip(base_label: str | None, label: str | None) -> float:
  if base_label is None or label is None:
    return float("nan")
  return float(label != base_label)


def modal_label(
    labels: Sequence[str | None],
    label_probs: Sequence[Mapping[str, float] | None] = (),
) -> str | None:
  """Most frequent label among samples; ties go to the higher mean probability.

  Remaining ties (equal mean probability) go to the alphabetically first
  label, so the result is always deterministic.
  """
  counts = Counter(l for l in labels if l is not None)
  if not counts:
    return None
  top = max(counts.values())
  tied = sorted(l for l, c in counts.items() if c == top)
  if len(tied) == 1:
    return tied[0]
  probs = [p for p in label_probs if p is not None]

  def mean_prob(label: str) -> float:
    return float(np.mean([p.get(label, 0.0) for p in probs])) if probs else 0.0

  # max() keeps the first of equal keys, and `tied` is sorted.
  return max(tied, key=mean_prob)


def sampled_flips(labels: Sequence[str | None], base_modal: str | None) -> np.ndarray:
  return np.array([greedy_flip(base_modal, l) for l in labels])


def recode_missing(i, value: float) -> np.ndarray:
  """Failure sensitivity analysis: undefined I recoded as all-flip or no-flip."""
  i = np.asarray(i, dtype=float)
  return np.where(np.isnan(i), value, i)


def copier_mentions(n: int) -> np.ndarray:
  """Dummy explainer that copies the whole input: always mentions the word."""
  return np.ones(n)


def random_mentions(n: int, q: float, seed: int) -> np.ndarray:
  """Dummy explainer that mentions the word with probability q, independently of I."""
  return (np.random.default_rng(seed).random(n) < q).astype(float)


def per_example_confusion(df: pd.DataFrame, e_col: str, i_col: str) -> pd.DataFrame:
  """Confusion counts per example id (rows with E or I undefined are dropped)."""
  e = df[e_col].astype(float)
  i = df[i_col].astype(float)
  ok = e.notna() & i.notna()
  e, i = e[ok].astype(bool), i[ok].astype(bool)
  counts = pd.DataFrame({
      "e1_i1": e & i, "e1_i0": e & ~i, "e0_i1": ~e & i, "e0_i0": ~e & ~i,
  }).astype(float)
  counts["example_id"] = df.loc[ok, "example_id"].to_numpy()
  return counts.groupby("example_id")[list(metrics.CONFUSION)].sum()


def per_example_cct(df: pd.DataFrame, e_col: str, tvd_col: str) -> pd.DataFrame:
  """CCT sufficient statistics per example id."""
  e = df[e_col].astype(float)
  t = df[tvd_col].astype(float)
  ok = e.notna() & t.notna()
  sums = pd.DataFrame({
      "n": 1.0, "sum_e": e[ok], "sum_t": t[ok], "sum_tt": t[ok] ** 2, "sum_et": e[ok] * t[ok],
  })
  sums["example_id"] = df.loc[ok, "example_id"].to_numpy()
  return sums.groupby("example_id")[list(metrics.CCT_SUMS)].sum()


def per_example_accuracy(df: pd.DataFrame, correct_col: str = "correct") -> pd.DataFrame:
  """(n, n_correct) per example id, over base runs with a defined label."""
  c = df[correct_col].astype(float)
  ok = c.notna()
  out = pd.DataFrame({"n": 1.0, "n_correct": c[ok]})
  out["example_id"] = df.loc[ok, "example_id"].to_numpy()
  return out.groupby("example_id")[["n", "n_correct"]].sum()
