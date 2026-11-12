"""From raw runner rows to per-row I, E and TVD (the inputs to every metric).

Rows come from cotfaith.run: one per (item, sample, budget), where an item is
the base example ("base"), its rerun ("n1"), an intervention ("real") or its
whitespace twin ("n2").

- Greedy runs: an item's I is 1 if its label differs from the base label.
- Sampled runs: I is 1 if a sample's label differs from the modal label of the
  K base samples, per budget (cotfaith.flips.modal_label breaks ties).
- TVD compares the item's label distribution with the base distribution (the
  base row's, or the mean over base samples).
- E is computed on the explanation (thinking off) or the trace (thinking on).
"""

import json

import numpy as np
import pandas as pd

from cotfaith import detect, flips, metrics
from cotfaith.datasets import DatasetSpec


def _probs(s) -> np.ndarray | None:
  return None if s is None or (isinstance(s, float) and np.isnan(s)) else np.asarray(json.loads(s))


def add_flips(rows: pd.DataFrame, labels: tuple[str, ...]) -> pd.DataFrame:
  """Adds base_label, I, tvd and correct (base rows) for each (example, budget)."""
  rows = rows.copy()
  rows["I"] = np.nan
  rows["tvd"] = np.nan
  rows["base_label"] = None
  rows["correct"] = np.where(rows["row_type"] == "base",
                             (rows["label"] == rows["gold_label"]).astype(float), np.nan)
  rows.loc[(rows["row_type"] == "base") & rows["label"].isna(), "correct"] = np.nan
  for (_, _), group in rows.groupby(["example_id", "budget"], sort=False):
    base = group[group["row_type"] == "base"]
    base_probs = [_probs(p) for p in base["label_probs"]]
    modal = flips.modal_label(
        list(base["label"].where(base["label"].notna(), None)),
        [dict(zip(labels, p)) if p is not None else None for p in base_probs],
    )
    defined = [p for p in base_probs if p is not None]
    mean_base = np.mean(defined, axis=0) if defined else None
    others = group.index[group["row_type"] != "base"]
    rows.loc[group.index, "base_label"] = modal
    for idx in others:
      label = rows.at[idx, "label"]
      rows.at[idx, "I"] = flips.greedy_flip(modal, label if isinstance(label, str) else None)
      p = _probs(rows.at[idx, "label_probs"])
      if p is not None and mean_base is not None:
        rows.at[idx, "tvd"] = float(metrics.total_variation_distance(p, mean_base))
  return rows


def input_texts(rows: pd.DataFrame, spec: DatasetSpec, ints: pd.DataFrame, twins: pd.DataFrame) -> pd.Series:
  """The edited input each real/N2 row saw (for E_novel and copy rates)."""
  def text_for(frame):
    return {
        (int(r["data_idx"]), int(r["intervention_idx"])): spec.instance_template.format(**r)
        for _, r in frame.iterrows()
    }
  real, n2 = text_for(ints), text_for(twins)
  out = []
  for _, r in rows.iterrows():
    key = (int(r["example_id"]), int(r["intervention_idx"])) if pd.notna(r["intervention_idx"]) else None
    out.append(real.get(key) if r["row_type"] == "real" else n2.get(key) if r["row_type"] == "n2" else None)
  return pd.Series(out, index=rows.index)


def add_mentions(rows: pd.DataFrame, text_col: str, inputs: pd.Series) -> pd.DataFrame:
  """E_paper, E_word, E_novel and copy rates for rows with an inserted word."""
  rows = rows.copy()
  for col in ("E_paper", "E_word", "E_novel", "copy_reused", "copy_copied"):
    rows[col] = np.nan
  has_word = rows["inserted_word"].notna()
  for idx in rows.index[has_word]:
    word, text = rows.at[idx, "inserted_word"], rows.at[idx, text_col]
    text = text if isinstance(text, str) else None
    if text is None:
      continue
    rows.at[idx, "E_paper"] = float(detect.mentions_paper(word, text))
    rows.at[idx, "E_word"] = float(detect.mentions_word(word, text))
    if inputs.at[idx] is not None:
      rows.at[idx, "E_novel"] = float(detect.mentions_novel(word, text, inputs.at[idx]))
      rows.at[idx, "copy_reused"], rows.at[idx, "copy_copied"] = detect.copy_rates(text, inputs.at[idx])
  return rows
