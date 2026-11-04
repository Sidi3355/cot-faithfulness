"""Tables, statistics and figures for the pilot (no interpretation).

Usage:
  python pilot/scripts/analyze.py prepilot   # arm R sample-size table (after the 20-example pre-pilot)
  python pilot/scripts/analyze.py r          # arm R tables and figures
  python pilot/scripts/analyze.py c          # arm C tables and figures
Writes CSV and PNG files to pilot/results/{R,C}/.
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cotfaith import analysis, bootstrap, config, datasets, flips, interventions, metrics, models, run  # noqa: E402
from cotfaith import rows as rows_lib  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "pilot" / "raw"  # Live runner output (synced to pilot/results/raw).
OUT = ROOT / "pilot" / "results"
CONFIGS = ROOT / "pilot" / "configs"
INTERVENTIONS = ROOT / "pilot" / "interventions" / "natural_comve.parquet"
SPEC = datasets.COMVE
LENGTHS = ("very_concise", "concise", "empty", "comprehensive", "very_comprehensive")
R_LENGTHS = ("very_concise", "empty", "very_comprehensive")  # Arm R uses three lengths.
BUDGETS = ("128", "256", "512", "1024", "uncapped")
R_MODELS = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct")
CELL_METRICS = ("accuracy", "flip_rate", "mention_rate", "tpr_ct", "fpr", "phi_cct", "cct")
N_CANDIDATES = (50, 100, 150)  # 167 examples were judged (see pilot/results/interventions_meta.json).
TARGET_HALF_WIDTH = 0.05


# Loading -----------------------------------------------------------------------


class Data:
  def __init__(self):
    splits = datasets.load_splits("comve", str(ROOT / "data"))
    self.test = splits["test"]
    self.ints = pd.read_parquet(INTERVENTIONS)
    self.twins = interventions.n2_twins(self.ints, self.test)

  def load(self, cfg: config.RunConfig, n_examples: int | None = None) -> pd.DataFrame:
    rows = run.load_rows(cfg.results_dir(RAW))
    if rows.empty:
      return rows
    if n_examples is not None:
      keep = self.test.sample(frac=1, random_state=0).index[:n_examples]
      rows = rows[rows["example_id"].isin(keep)]
    rows = rows_lib.add_flips(rows, SPEC.class_labels)
    text_col = "trace" if cfg.thinking else "explanation"
    return rows_lib.add_mentions(rows, text_col, rows_lib.input_texts(rows, SPEC, self.ints, self.twins))


def cfg_named(name: str) -> config.RunConfig:
  return config.load(CONFIGS / f"{name}.yaml")


# Statistics --------------------------------------------------------------------


def stats_block(rows: pd.DataFrame, boot: bootstrap.ClusterBootstrap, e_col: str = "E_paper",
                row_type: str = "real") -> np.ndarray:
  """Per-example [confusion(4), cct sums(5), n_base, n_correct] for one cell."""
  sub = rows[rows["row_type"] == row_type]
  conf = boot.align(flips.per_example_confusion(sub, e_col, "I"))
  cct = boot.align(flips.per_example_cct(sub, e_col, "tvd"))
  acc = boot.align(flips.per_example_accuracy(rows[rows["row_type"] == "base"], "correct"))
  return np.concatenate([conf, cct, acc], axis=1)


def cell_metrics(s: np.ndarray) -> np.ndarray:
  """Metrics (last axis = CELL_METRICS) from summed stats_block statistics."""
  conf, cct, acc = s[..., :4], s[..., 4:9], s[..., 9:11]
  return np.stack([
      metrics.accuracy(acc), metrics.flip_rate(conf), metrics.mention_rate(conf), metrics.tpr(conf),
      metrics.fpr(conf), metrics.phi(conf), metrics.cct(cct),
  ], axis=-1)


def summarize(name: str, block: np.ndarray, boot: bootstrap.ClusterBootstrap) -> tuple[dict, np.ndarray]:
  point = cell_metrics(block.sum(axis=0))
  dist = boot.distribution(block, cell_metrics)
  row = {"cell": name, "n_examples": int((block[:, 9] > 0).sum()), "n_rows": int(block[:, :4].sum()),
         "n_flips": int(block[:, 0].sum() + block[:, 2].sum())}
  for k, m in enumerate(CELL_METRICS):
    lo, hi = bootstrap.percentile_ci(dist[:, k])
    row.update({m: point[k], f"{m}_lo": lo, f"{m}_hi": hi})
  return row, dist


def f_auroc_row(name: str, blocks: list[np.ndarray], boot: bootstrap.ClusterBootstrap) -> tuple[dict, np.ndarray]:
  stacked = np.stack([b[:, :4] for b in blocks], axis=1)  # (examples, settings, 4)
  fn = lambda s: metrics.f_auroc_batch(metrics.fpr(s), metrics.tpr(s))
  point = metrics.f_auroc(metrics.fpr(stacked.sum(0)), metrics.tpr(stacked.sum(0)))
  dist = boot.distribution(stacked, fn)
  lo, hi = bootstrap.percentile_ci(dist)
  return {"cell": name, "f_auroc": point, "f_auroc_lo": lo, "f_auroc_hi": hi}, dist


def paired(name: str, a: np.ndarray, b: np.ndarray, point_a: float, point_b: float) -> dict:
  d = a - b
  lo, hi = bootstrap.percentile_ci(d)
  return {"comparison": name, "difference": point_a - point_b, "lo": lo, "hi": hi,
          "p_value": bootstrap.paired_p_value(d)}


def write(df: pd.DataFrame, path: pathlib.Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(path, index=False, float_format="%.4f")
  print("wrote", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def universe(n: int) -> bootstrap.ClusterBootstrap:
  test = datasets.load_splits("comve", str(ROOT / "data"))["test"]
  ids = test.sample(frac=1, random_state=0).index[:n]
  return bootstrap.ClusterBootstrap.for_examples(list(ids))


def short(model: str) -> str:
  return model.split("/")[1]


# Arm R -------------------------------------------------------------------------


def prepilot(data: Data) -> None:
  boot = universe(20)
  rows = []
  for model in R_MODELS:
    cfg = cfg_named(f"R_{short(model)}_empty")
    block = stats_block(data.load(cfg, n_examples=20), boot)
    summary, dist = summarize(short(model), block, boot)
    lo, hi = summary["phi_cct_lo"], summary["phi_cct_hi"]
    half = (hi - lo) / 2
    row = {"model": short(model), "n_examples": 20, "n_interventions": summary["n_rows"],
           "n_flips": summary["n_flips"], "flip_rate": summary["flip_rate"], "phi_cct": summary["phi_cct"],
           "phi_cct_ci_half_width": half}
    for n in N_CANDIDATES:
      row[f"projected_half_width_N{n}"] = half * np.sqrt(20 / n)
    rows.append(row)
  table = pd.DataFrame(rows)
  ok = [n for n in N_CANDIDATES if (table[f"projected_half_width_N{n}"] <= TARGET_HALF_WIDTH).all()]
  chosen = ok[0] if ok else max(N_CANDIDATES)
  table["chosen_N"] = chosen
  write(table, OUT / "R" / "prepilot_sample_size.csv")
  print("chosen N:", chosen)


def arm_r(data: Data) -> None:
  n = cfg_named(f"R_{short(R_MODELS[0])}_empty").n_examples
  boot = universe(n)
  cells, dists, blocks = [], {}, {}
  for model in R_MODELS:
    for length in R_LENGTHS:
      cfg = cfg_named(f"R_{short(model)}_{length}")
      rows = data.load(cfg)
      if rows.empty:
        continue
      block = stats_block(rows, boot)
      summary, dist = summarize(f"{short(model)} | {length}", block, boot)
      base = rows.loc[rows.row_type == "base"]
      summary.update(model=short(model), length=length,
                     n1_flip_rate=float(rows.loc[rows.row_type == "n1", "I"].mean()),
                     parse_failure_rate=float(rows["label"].isna().mean()),
                     base_parse_failure_rate=float(base["label"].isna().mean()),
                     # Position bias: share of parsed base predictions that pick
                     # option 1, against the share of examples whose gold is 1.
                     base_pred_label1_rate=float((base["label"].dropna() == "1").mean()),
                     gold_label1_rate=float((base["gold_label"] == "1").mean()))
      cells.append(summary)
      dists[model, length], blocks[model, length] = dist, block
  cells_df = pd.DataFrame(cells)
  write(cells_df, OUT / "R" / "cells.csv")

  # F-AUROC per model across the five lengths.
  fa_rows, fa_dists = [], {}
  for model in R_MODELS:
    if all((model, l) in blocks for l in R_LENGTHS):
      row, dist = f_auroc_row(short(model), [blocks[model, l] for l in R_LENGTHS], boot)
      fa_rows.append(row)
      fa_dists[model] = dist
  write(pd.DataFrame(fa_rows), OUT / "R" / "f_auroc.csv")

  # F1: very comprehensive minus very concise, TPR and FPR, per model.
  idx = {m: k for k, m in enumerate(CELL_METRICS)}
  point = {(m, l): cell_metrics(blocks[m, l].sum(0)) for (m, l) in blocks}
  f1 = []
  for metric in ("tpr_ct", "fpr"):
    rows_m = []
    for model in R_MODELS:
      a, b = (model, "very_comprehensive"), (model, "very_concise")
      if a in dists and b in dists:
        k = idx[metric]
        rows_m.append(dict(paired(f"{short(model)}: {metric} very_comprehensive - very_concise",
                                  dists[a][:, k], dists[b][:, k], point[a][k], point[b][k]), metric=metric))
    for r, adj in zip(rows_m, bootstrap.holm([r["p_value"] for r in rows_m])):
      r["p_holm"] = adj
    f1.extend(rows_m)
  write(pd.DataFrame(f1), OUT / "R" / "F1_verbosity.csv")

  # F2: larger minus smaller model, at the no-instruction length, and F-AUROC.
  pairs = [(R_MODELS[1], R_MODELS[0]), (R_MODELS[2], R_MODELS[1]), (R_MODELS[2], R_MODELS[0])]
  f2 = []
  for metric in ("phi_cct", "tpr_ct", "accuracy"):
    rows_m = []
    for big, small in pairs:
      a, b = (big, "empty"), (small, "empty")
      if a in dists and b in dists:
        k = idx[metric]
        rows_m.append(dict(paired(f"{metric}: {short(big)} - {short(small)}", dists[a][:, k], dists[b][:, k],
                                  point[a][k], point[b][k]), metric=metric))
    for r, adj in zip(rows_m, bootstrap.holm([r["p_value"] for r in rows_m])):
      r["p_holm"] = adj
    f2.extend(rows_m)
  rows_m = []
  fa_point = {short(r["cell"]) if "/" in r["cell"] else r["cell"]: r["f_auroc"] for r in fa_rows}
  for big, small in pairs:
    if big in fa_dists and small in fa_dists:
      rows_m.append(dict(paired(f"f_auroc: {short(big)} - {short(small)}", fa_dists[big], fa_dists[small],
                                fa_point[short(big)], fa_point[short(small)]), metric="f_auroc"))
  for r, adj in zip(rows_m, bootstrap.holm([r["p_value"] for r in rows_m])):
    r["p_holm"] = adj
  f2.extend(rows_m)
  write(pd.DataFrame(f2), OUT / "R" / "F2_scale.csv")

  # F3: phi-CCT against CCT across all cells.
  valid = cells_df.dropna(subset=["phi_cct", "cct"])
  r2 = float(np.corrcoef(valid["phi_cct"], valid["cct"])[0, 1] ** 2) if len(valid) > 2 else float("nan")
  write(pd.DataFrame([{"n_cells": len(valid), "r_squared": r2,
                       "pearson_r": float(np.corrcoef(valid["phi_cct"], valid["cct"])[0, 1])}]),
        OUT / "R" / "F3_phi_vs_cct.csv")

  # Comparison with the paper (paper columns filled in once the paper text is available).
  comp = []
  for model in R_MODELS:
    for metric in ("phi_cct", "tpr_ct", "cct", "accuracy"):
      sel = cells_df[(cells_df.model == short(model)) & (cells_df.length == "empty")]
      if not sel.empty:
        r = sel.iloc[0]
        comp.append({"model": short(model), "metric": metric, "ours": r[metric], "ours_lo": r[f"{metric}_lo"],
                     "ours_hi": r[f"{metric}_hi"], "paper": None, "paper_within_our_ci": None})
  write(pd.DataFrame(comp), OUT / "R" / "paper_comparison.csv")

  # Figures.
  fig, axes = plt.subplots(1, len(R_MODELS), figsize=(4 * len(R_MODELS), 4), sharey=True)
  for ax, model in zip(np.atleast_1d(axes), R_MODELS):
    sel = cells_df[cells_df.model == short(model)].set_index("length").reindex(R_LENGTHS)
    _tpr_fpr_plot(ax, sel["fpr"], sel["tpr_ct"], R_LENGTHS, short(model))
  fig.tight_layout(); fig.savefig(OUT / "R" / "tpr_fpr_by_length.png", dpi=150); plt.close(fig)

  fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
  sizes = [models.get(m).params for m in R_MODELS]
  for ax, metric in zip(axes, ("phi_cct", "tpr_ct", "accuracy")):
    sel = cells_df[cells_df.length == "empty"].set_index("model").reindex([short(m) for m in R_MODELS])
    ax.errorbar(sizes, sel[metric], yerr=[sel[metric] - sel[f"{metric}_lo"], sel[f"{metric}_hi"] - sel[metric]],
                marker="o", capsize=3)
    ax.set_xscale("log"); ax.set_xlabel("parameters"); ax.set_title(f"{metric} (no length instruction)")
  fig.tight_layout(); fig.savefig(OUT / "R" / "metrics_by_size.png", dpi=150); plt.close(fig)

  fig, ax = plt.subplots(figsize=(4, 4))
  ax.scatter(valid["phi_cct"], valid["cct"])
  ax.set_xlabel("phi-CCT"); ax.set_ylabel("CCT"); ax.set_title(f"R² = {r2:.3f} ({len(valid)} cells)")
  fig.tight_layout(); fig.savefig(OUT / "R" / "phi_vs_cct.png", dpi=150); plt.close(fig)


def _tpr_fpr_plot(ax, fpr, tpr, labels, title):
  ax.plot([0, 1], [0, 1], color="0.8", lw=1)
  ax.scatter(fpr, tpr)
  for x, y, l in zip(fpr, tpr, labels):
    if pd.notna(x) and pd.notna(y):
      ax.annotate(str(l), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
  pts = sorted({(0.0, 0.0), (1.0, 1.0)} | {(x, y) for x, y in zip(fpr, tpr) if pd.notna(x) and pd.notna(y)})
  hull = []
  for p in pts:
    while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) >= 0:
      hull.pop()
    hull.append(p)
  ax.plot(*zip(*hull), color="C1", lw=1)
  ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR (CT)"); ax.set_title(title)


# Arm C -------------------------------------------------------------------------


def arm_c(data: Data) -> None:
  boot = universe(cfg_named("C2_thinking").n_examples)
  out = OUT / "C"

  # C1: greedy explanation baseline, 20 conditions.
  c1, c1_blocks = [], {}
  for template in ("pe", "ep"):
    for demos in ("with_explanations", "labels_only"):
      for length in LENGTHS:
        cfg = cfg_named(f"C1_{template}_{demos}_{length}")
        rows = data.load(cfg)
        if rows.empty:
          continue
        block = stats_block(rows, boot)
        summary, _ = summarize(f"{template} | {demos} | {length}", block, boot)
        summary.update(template=template, demos=demos, length=length, **_floors(rows, boot),
                       parse_failure_rate=float(rows["label"].isna().mean()))
        c1.append(summary)
        c1_blocks[template, demos, length] = block
  c1_df = pd.DataFrame(c1)
  write(c1_df, out / "C1_cells.csv")
  fa = []
  for template in ("pe", "ep"):
    for demos in ("with_explanations", "labels_only"):
      if all((template, demos, l) in c1_blocks for l in LENGTHS):
        fa.append(f_auroc_row(f"{template} | {demos}", [c1_blocks[template, demos, l] for l in LENGTHS], boot)[0])
  write(pd.DataFrame(fa), out / "C1_f_auroc.csv")

  # C2 / cell B: thinking traces, per budget.
  c2_rows = data.load(cfg_named("C2_thinking"))
  c2, c2_blocks, dummies, copies, failures = [], {}, [], [], []
  for b in BUDGETS:
    rows = c2_rows[c2_rows["budget"] == b]
    if rows.empty:
      continue
    block = stats_block(rows, boot)
    summary, _ = summarize(f"budget {b}", block, boot)
    real = rows[rows.row_type == "real"]
    summary.update(budget=b, **_floors(rows, boot), forced_answer_rate=float(rows["forced"].mean()),
                   mean_trace_tokens=float(rows["trace_n_tokens"].mean()))
    for alt in ("E_word", "E_novel"):
      alt_summary, _ = summarize("", stats_block(rows, boot, e_col=alt), boot)
      summary.update({f"{alt}_tpr": alt_summary["tpr_ct"], f"{alt}_fpr": alt_summary["fpr"],
                      f"{alt}_phi": alt_summary["phi_cct"]})
    c2.append(summary)
    c2_blocks[b] = block
    dummies.extend(_dummies(real, b))
    copies.append({"budget": b, "copy_reused_mean": float(real["copy_reused"].mean()),
                   "copy_copied_mean": float(real["copy_copied"].mean()),
                   "E_paper_minus_E_novel": float((real["E_paper"] - real["E_novel"]).mean())})
    failures.append(_failures(rows, b, boot))
  write(pd.DataFrame(c2), out / "C2_cells_by_budget.csv")
  if all(b in c2_blocks for b in BUDGETS):
    write(pd.DataFrame([f_auroc_row("thinking traces (budgets)", [c2_blocks[b] for b in BUDGETS], boot)[0]]),
          out / "C2_f_auroc.csv")
  write(pd.DataFrame(dummies), out / "C2_dummy_explainers.csv")
  write(pd.DataFrame(copies), out / "C2_copy_rates.csv")
  write(pd.DataFrame(failures), out / "C2_failure_sensitivity.csv")

  # C3: cell A (explanations, thinking off, sampled) and cell R (reference).
  c3a, c3a_blocks, c3a_rows_by_len = [], {}, {}
  for length in LENGTHS:
    rows = data.load(cfg_named(f"C3A_{length}"))
    if rows.empty:
      continue
    block = stats_block(rows, boot)
    summary, _ = summarize(f"cell A | {length}", block, boot)
    summary.update(length=length, **_floors(rows, boot), parse_failure_rate=float(rows["label"].isna().mean()))
    c3a.append(summary)
    c3a_blocks[length] = block
    c3a_rows_by_len[length] = rows
  write(pd.DataFrame(c3a), out / "C3A_cells.csv")
  if all(l in c3a_blocks for l in LENGTHS):
    write(pd.DataFrame([f_auroc_row("cell A (lengths)", [c3a_blocks[l] for l in LENGTHS], boot)[0]]),
          out / "C3A_f_auroc.csv")
  r_rows = data.load(cfg_named("C3R"))
  if not r_rows.empty:
    acc = boot.align(flips.per_example_accuracy(r_rows[r_rows.row_type == "base"], "correct"))
    real = r_rows[r_rows.row_type == "real"]
    write(pd.DataFrame([{"cell": "R (thinking off, label only)", "accuracy": metrics.accuracy(acc.sum(0)),
                         "flip_rate": float(real["I"].mean()), **_floors(r_rows, boot)}]), out / "C3R_reference.csv")

  # Trace (cell B) vs explanation (cell A) at matched accuracy.
  if c3a_blocks and c2_blocks:
    off = np.stack([_point_stats(c3a_blocks[l]) for l in LENGTHS if l in c3a_blocks], axis=1)
    on = np.stack([_point_stats(c2_blocks[b]) for b in BUDGETS if b in c2_blocks], axis=1)
    res = []
    for iso in (False, True):
      t = analysis.curve_test(off, on, boot, isotonic=iso)
      res.append({"fit": "isotonic" if iso else "ols", "delta": t.estimate, "lo": t.ci[0], "hi": t.ci[1],
                  "p_value": t.p_value, "undefined_fraction": t.undefined_fraction})
    write(pd.DataFrame(res), out / "trace_vs_explanation_curve_test.csv")
    reg_rows = []
    if "empty" in c3a_rows_by_len:
      a = c3a_rows_by_len["empty"]
      b = c2_rows[c2_rows.budget == "uncapped"]
      for frame, mode in ((a, 0), (b, 1)):
        real = frame[frame.row_type == "real"].copy()
        base_ok = frame[frame.row_type == "base"].groupby("example_id")["correct"].mean()
        real["mode"] = mode
        real["base_correct"] = (real["example_id"].map(base_ok) >= 0.5).astype(float)
        real["E"] = real["E_paper"]
        reg_rows.append(real[["E", "I", "mode", "base_correct", "example_id"]])
      data_reg = pd.concat(reg_rows).dropna()
      g = analysis.interaction_gee(data_reg.assign(log10_params=0.0), formula="E ~ I * mode + base_correct")
      write(pd.DataFrame([{"coefficient": "I:mode", "estimate": g.coef, "lo": g.ci[0], "hi": g.ci[1],
                           "p_value": g.p_value, "n_rows": g.n_rows, "n_clusters": g.n_clusters}]),
            out / "trace_vs_explanation_regression.csv")
    _figures_c(c3a, c2, out)


def _point_stats(block: np.ndarray) -> np.ndarray:
  """Per-example [confusion(4), n_base, n_correct] for the curve test."""
  return np.concatenate([block[:, :4], block[:, 9:11]], axis=1)


def _floors(rows: pd.DataFrame, boot: bootstrap.ClusterBootstrap) -> dict:
  """Flip rates of real edits, N1 and N2, and the paired excess flip rate (real - N2)."""
  out = {}
  for t in ("real", "n1", "n2"):
    sub = rows[rows.row_type == t]
    out[f"{t}_flip_rate"] = float(sub["I"].mean()) if len(sub) else float("nan")
  real, n2 = rows[rows.row_type == "real"], rows[rows.row_type == "n2"]
  if len(n2):
    per = pd.DataFrame({
        "real_n": real.groupby("example_id")["I"].count(), "real_f": real.groupby("example_id")["I"].sum(),
        "n2_n": n2.groupby("example_id")["I"].count(), "n2_f": n2.groupby("example_id")["I"].sum(),
    }).fillna(0)
    stats = boot.align(per)
    fn = lambda s: metrics.ratio(s[..., 1], s[..., 0]) - metrics.ratio(s[..., 3], s[..., 2])
    dist = boot.distribution(stats, fn)
    lo, hi = bootstrap.percentile_ci(dist)
    out.update(excess_flip_rate=float(fn(stats.sum(0))), excess_lo=lo, excess_hi=hi,
               n2_spontaneous_mention_rate=float(n2["E_paper"].mean()))
  return out


def _dummies(real: pd.DataFrame, budget: str) -> list[dict]:
  i = real["I"].to_numpy(float)
  q = float(np.nanmean(real["E_paper"]))
  out = []
  for name, e in (("copier (always mentions)", flips.copier_mentions(len(i))),
                  (f"random (q = {q:.3f})", flips.random_mentions(len(i), q, seed=int(budget) if budget.isdigit() else 0))):
    c = metrics.confusion_counts(e, i)
    out.append({"budget": budget, "explainer": name, "tpr_ct": float(metrics.tpr(c)), "fpr": float(metrics.fpr(c)),
                "phi_cct": float(metrics.phi(c)), "f_auroc_single_point": metrics.f_auroc([metrics.fpr(c)], [metrics.tpr(c)])})
  return out


def _failures(rows: pd.DataFrame, budget: str, boot) -> dict:
  real = rows[rows.row_type == "real"]
  out = {"budget": budget, "parse_failure_rate": float(rows["label"].isna().mean()),
         "unterminated_rate": float((~rows["terminated"].astype(bool)).mean())}
  for value in (1.0, 0.0):
    r = real.copy()
    r["I"] = flips.recode_missing(r["I"], value)
    c = metrics.confusion_counts(r["E_paper"], r["I"])
    out[f"phi_cct_failures_as_{int(value)}"] = float(metrics.phi(c))
    out[f"tpr_ct_failures_as_{int(value)}"] = float(metrics.tpr(c))
  return out


def _figures_c(c3a, c2, out):
  a, b = pd.DataFrame(c3a), pd.DataFrame(c2)
  fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
  if not a.empty:
    _tpr_fpr_plot(axes[0], a["fpr"], a["tpr_ct"], a["length"], "explanation (thinking off), by length")
  if not b.empty:
    _tpr_fpr_plot(axes[1], b["fpr"], b["tpr_ct"], b["budget"], "trace (thinking on), by budget")
  fig.tight_layout(); fig.savefig(out / "tpr_fpr_explanation_vs_trace.png", dpi=150); plt.close(fig)
  fig, ax = plt.subplots(figsize=(5, 4))
  for frame, label, key in ((a, "explanation (lengths)", "length"), (b, "trace (budgets)", "budget")):
    if frame.empty:
      continue
    ax.errorbar(frame["accuracy"], frame["phi_cct"],
                yerr=[frame["phi_cct"] - frame["phi_cct_lo"], frame["phi_cct_hi"] - frame["phi_cct"]],
                fmt="o", capsize=3, label=label)
    for _, r in frame.iterrows():
      ax.annotate(str(r[key]), (r["accuracy"], r["phi_cct"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
  ax.set_xlabel("base accuracy"); ax.set_ylabel("phi-CCT"); ax.legend(fontsize=8)
  fig.tight_layout(); fig.savefig(out / "phi_vs_accuracy_trace_vs_explanation.png", dpi=150); plt.close(fig)
  if not b.empty:
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for m in ("tpr_ct", "fpr", "phi_cct"):
      ax.plot(b["budget"], b[m], marker="o", label=m)
    ax.set_xlabel("thinking budget"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "trace_metrics_by_budget.png", dpi=150); plt.close(fig)


def run_log() -> None:
  rows = []
  for meta_path in sorted(RAW.rglob("meta.json")):
    d = meta_path.parent
    meta = json.loads(meta_path.read_text())
    timing = pd.read_csv(d / "timing.csv", names=["example_id", "n_rows", "seconds"]) if (d / "timing.csv").exists() else None
    rows.append({"results_dir": str(d.relative_to(RAW)), "model_revision": meta.get("model_revision"),
                 "git_commit": meta.get("git_commit"), "started": meta.get("started"),
                 "torch": meta["versions"]["torch"], "transformers": meta["versions"]["transformers"],
                 "interventions_sha256": meta.get("interventions_sha256"),
                 "n_examples_done": 0 if timing is None else len(timing),
                 "n_rows": 0 if timing is None else int(timing["n_rows"].sum()),
                 "hours": 0 if timing is None else round(timing["seconds"].sum() / 3600, 2)})
  write(pd.DataFrame(rows), OUT / "run_log.csv")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("what", choices=["prepilot", "r", "c", "log"])
  args = parser.parse_args()
  data = Data()
  {"prepilot": prepilot, "r": arm_r, "c": arm_c}.get(args.what, lambda d: None)(data)
  run_log()


if __name__ == "__main__":
  main()
