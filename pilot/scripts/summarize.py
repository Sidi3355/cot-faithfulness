"""Writes pilot/results/SUMMARY.md: the key numbers from the result CSVs on one page.

Usage: python pilot/scripts/summarize.py   (after analyze.py r and c)
No interpretation: every number is copied from a CSV in pilot/results/.
"""

import pandas as pd

A = None  # The analyze module: output folder (A.OUT) and folder names. Set by write_summary.


def read(folder: str, name: str) -> pd.DataFrame:
  path = A.OUT / folder / name
  return pd.read_csv(path) if path.exists() else pd.DataFrame()


def f2(x) -> str:
  return "–" if pd.isna(x) else f"{round(x, 2) + 0.0:.2f}"  # + 0.0 avoids "-0.00".


_NAMES = [("phi_cct", "φ-CCT"), ("tpr_ct", "CT"), ("f_auroc", "F-AUROC"), ("fpr", "FPR"), ("cct", "CCT"),
          ("very_comprehensive", "very comprehensive"), ("very_concise", "very concise"),
          ("with_explanations", "with explanations"), ("labels_only", "labels only"), ("empty", "none")]


def pretty(s) -> str:
  """Column and condition names as plain words."""
  s = str(s)
  for a, b in _NAMES:
    s = s.replace(a, b)
  return s


def ci(row, col: str) -> str:
  """Value with its 95% interval, e.g. '0.36 [0.27, 0.46]'."""
  if pd.isna(row[col]):
    return "–"
  return f"{f2(row[col])} [{f2(row[f'{col}_lo'])}, {f2(row[f'{col}_hi'])}]"


def pct(x) -> str:
  return "–" if pd.isna(x) else f"{100 * x:.0f}%"


def rng(s: pd.Series, fmt=f2) -> str:
  lo, hi = s.min(), s.max()
  return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)} to {fmt(hi)}"


def count_rng(s: pd.Series) -> str:
  return rng(s, fmt=lambda x: str(int(x)))


def p_value(x) -> str:
  if pd.isna(x):
    return "–"
  return "< 0.001" if x < 0.001 else f"{x:.3f}"


def table(header: list[str], rows: list[list[str]]) -> str:
  lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
  lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
  return "\n".join(lines) + "\n"


def model_name(s: str) -> str:
  return s.replace("Qwen2.5-", "").replace("-Instruct", "")


def arm_r() -> str:
  out = ["## Arm R: replication (Qwen2.5-Instruct, 150 questions, greedy, predict then explain)\n"]
  cells = read(A.R_DIR, "cells.csv")
  if cells.empty:
    return ""
  out.append("**Per model and length** (CT = share of answer-changing words mentioned; FPR = share of "
             "other words mentioned; φ-CCT: 0 = unrelated, 1 = perfect)\n")
  out.append(table(
      ["Model", "Length", "Questions", "Accuracy", "Flip rate", "CT", "FPR", "φ-CCT", "CCT"],
      [[model_name(r["model"]), pretty(r["length"]), r["n_examples"], f2(r["accuracy"]), f2(r["flip_rate"]),
        ci(r, "tpr_ct"), ci(r, "fpr"), ci(r, "phi_cct"), ci(r, "cct")] for _, r in cells.iterrows()]))
  fa = read(A.R_DIR, "f_auroc.csv")
  out.append("\n**F-AUROC** (over the 3 lengths; 0.5 = chance, 1 = perfect): "
             + "; ".join(f"{model_name(r['cell'])} {ci(r, 'f_auroc')}" for _, r in fa.iterrows()) + "\n")
  for name, title in (("F1_verbosity.csv", "**F1, verbosity** (very comprehensive − very concise)"),
                      ("F2_scale.csv", "**F2, scale** (larger − smaller model)")):
    t = read(A.R_DIR, name)
    out.append(f"\n{title}\n")
    out.append(table(["Comparison", "Difference [95% CI]", "p (Holm)"],
                     [[pretty(model_name(r["comparison"])), f"{round(r['difference'], 2) + 0.0:+.2f} [{f2(r['lo'])}, {f2(r['hi'])}]",
                       p_value(r["p_holm"])] for _, r in t.iterrows()]))
  f3 = read(A.R_DIR, "F3_phi_vs_cct.csv")
  if not f3.empty:
    r = f3.iloc[0]
    out.append(f"\n**F3, φ-CCT vs CCT** across the {int(r['n_cells'])} cells: R² = {f2(r['r_squared'])} "
               f"(Pearson r = {f2(r['pearson_r'])}). Paper: R² = 0.92.\n")
  comp = read(A.R_DIR, "paper_comparison.csv")
  if not comp.empty:
    out.append("\n**Compared with the paper** (ComVE; ours = range over our 3 lengths; paper = read off its "
               "figures, ±0.02)\n")
    out.append(table(["Model", "Metric", "Ours", "Paper", "Paper source"],
                     [[model_name(r["model"]), pretty(r["metric"]), f"{f2(r['ours_min'])} to {f2(r['ours_max'])}",
                       f2(r["paper_min"]) if r["paper_min"] == r["paper_max"]
                       else f"{f2(r['paper_min'])} to {f2(r['paper_max'])}", r["paper_source"]]
                      for _, r in comp.iterrows()]))
  out.append("\n**Other checks**\n")
  out.append(table(
      ["Model", "Picks option 1", "Option 1 correct", "Unreadable original answers", "Rerun (N1) flip rate"],
      [[model_name(m), rng(g["base_pred_label1_rate"], pct), rng(g["gold_label1_rate"], pct),
        rng(g["base_parse_failure_rate"], pct), rng(g["n1_flip_rate"], pct)]
       for m, g in cells.groupby("model", sort=False)]))
  return "\n".join(out)


def arm_c() -> str:
  out = ["## Arm C: chain of thought (Qwen3-0.6B, 40 questions)\n"]
  c1 = read(A.C1_DIR, "C1_cells.csv")
  if not c1.empty:
    out.append("### C1: written explanations, thinking off (greedy)\n")
    out.append("Ranges are over the 5 length instructions; F-AUROC combines them.\n")
    fa = read(A.C1_DIR, "C1_f_auroc.csv").set_index("cell")
    rows = []
    for (template, demos), g in c1.groupby(["template", "demos"], sort=False):
      key = f"{template} | {demos}"
      rows.append([template.upper(), pretty(demos), ci(fa.loc[key], "f_auroc") if key in fa.index else "–",
                   rng(g["accuracy"]), rng(g["tpr_ct"]), rng(g["fpr"]), rng(g["phi_cct"]),
                   count_rng(g["n_flips"]), count_rng(g["n_examples"]),
                   rng(g["parse_failure_rate"], pct)])
    out.append(table(["Order", "Examples", "F-AUROC", "Accuracy", "CT", "FPR", "φ-CCT", "Flips",
                      "Questions", "Unreadable"], rows))
    out.append("\n<details><summary>All 20 C1 conditions</summary>\n")
    out.append(table(["Condition", "Questions", "Accuracy", "Flips", "CT", "FPR", "φ-CCT", "Excess flip"],
                     [[f'{r["template"].upper()}, {pretty(r["demos"])}, {pretty(r["length"])}', r["n_examples"],
                       f2(r["accuracy"]), r["n_flips"], ci(r, "tpr_ct"), ci(r, "fpr"),
                       ci(r, "phi_cct"), f2(r["excess_flip_rate"])] for _, r in c1.iterrows()]))
    out.append("</details>\n")

  c2 = read(A.C2_DIR, "C2_cells_by_budget.csv")
  if not c2.empty:
    out.append("### C2: thinking traces (thinking on, 3 samples per prompt)\n")
    out.append(table(["Budget", "Accuracy", "CT", "FPR", "φ-CCT", "Forced answer", "Trace tokens"],
                     [[r["budget"], f2(r["accuracy"]), ci(r, "tpr_ct"), ci(r, "fpr"), ci(r, "phi_cct"),
                       pct(r["forced_answer_rate"]), f"{r['mean_trace_tokens']:.0f}"] for _, r in c2.iterrows()]))
    fa = read(A.C2_DIR, "C2_f_auroc.csv")
    if not fa.empty:
      out.append(f"\n**F-AUROC** over the 5 budgets: {ci(fa.iloc[0], 'f_auroc')}\n")
    out.append("\n**Noise vs real flips** (N1 = same question again; N2 = extra space only)\n")
    out.append(table(["Budget", "Real flip rate", "N1 flip rate", "N2 flip rate", "Excess over noise"],
                     [[r["budget"], f2(r["real_flip_rate"]), f2(r["n1_flip_rate"]), f2(r["n2_flip_rate"]),
                       f"{f2(r['excess_flip_rate'])} [{f2(r['excess_lo'])}, {f2(r['excess_hi'])}]"]
                      for _, r in c2.iterrows()]))
    copies = read(A.C2_DIR, "C2_copy_rates.csv").set_index("budget")
    dummies = read(A.C2_DIR, "C2_dummy_explainers.csv")
    fails = read(A.C2_DIR, "C2_failure_sensitivity.csv").set_index("budget")
    rand = dummies[dummies["explainer"].str.startswith("random")].set_index("budget")
    out.append("\n**Checks**: stricter mention detectors (E_word: whole words; E_novel: also ignores text "
               "copied from the question), copying, a random-mention dummy, unreadable answers\n")
    rows = []
    for _, r in c2.iterrows():
      b = r["budget"]
      rows.append([b, f2(r["phi_cct"]), f2(r["E_word_phi"]), f2(r["E_novel_phi"]),
                   f"{f2(r['E_novel_tpr'])} / {f2(r['E_novel_fpr'])}",
                   pct(copies.loc[b, "copy_reused_mean"]) if b in copies.index else "–",
                   f2(rand.loc[b, "phi_cct"]) if b in rand.index else "–",
                   pct(fails.loc[b, "parse_failure_rate"]) if b in fails.index else "–",
                   f"{f2(fails.loc[b, 'phi_cct_failures_as_1'])} / {f2(fails.loc[b, 'phi_cct_failures_as_0'])}"
                   if b in fails.index else "–"])
    out.append(table(["Budget", "φ-CCT", "φ-CCT (E_word)", "φ-CCT (E_novel)", "CT / FPR (E_novel)",
                      "Question words reused", "φ-CCT, random dummy", "Unreadable",
                      "φ-CCT if unreadable = flip / no flip"], rows))

  c3a = read(A.C3_DIR, "C3A_cells.csv")
  if not c3a.empty:
    out.append("\n### C3: traces vs written explanations at the same accuracy\n")
    out.append("**C3A**: thinking off, explain then predict, answers-only examples, sampled like C2\n")
    out.append(table(["Length", "Questions", "Accuracy", "CT", "FPR", "φ-CCT", "Real flip", "N1 flip",
                      "Unreadable"],
                     [[pretty(r["length"]), r["n_examples"], f2(r["accuracy"]), ci(r, "tpr_ct"), ci(r, "fpr"),
                       ci(r, "phi_cct"), f2(r["real_flip_rate"]), f2(r["n1_flip_rate"]),
                       pct(r["parse_failure_rate"])] for _, r in c3a.iterrows()]))
    fa = read(A.C3_DIR, "C3A_f_auroc.csv")
    if not fa.empty:
      out.append(f"\n**C3A F-AUROC** over the 5 lengths: {ci(fa.iloc[0], 'f_auroc')}\n")
    c3r = read(A.C3_DIR, "C3R_reference.csv")
    if not c3r.empty:
      r = c3r.iloc[0]
      out.append(f"\n**C3R** (answer only, no explanation): accuracy {f2(r['accuracy'])}, flip rate "
                 f"{f2(r['real_flip_rate'])}, N1 flip rate {f2(r['n1_flip_rate'])}, excess "
                 f"{f2(r['excess_flip_rate'])} [{f2(r['excess_lo'])}, {f2(r['excess_hi'])}]\n")
    curve = read(A.C3_DIR, "trace_vs_explanation_curve_test.csv")
    if not curve.empty:
      out.append("\n**Curve test** (thinking points' mean distance from the explanations' φ-CCT-vs-accuracy "
                 "line; blank estimate = fewer than 3 thinking points inside the explanations' accuracy range)\n")
      out.append(table(["Fit", "Estimate", "95% CI", "p", "Resamples where undefined"],
                       [[r["fit"], f2(r["delta"]), f"[{f2(r['lo'])}, {f2(r['hi'])}]", p_value(r["p_value"]),
                         pct(r["undefined_fraction"])] for _, r in curve.iterrows()]))
    reg = read(A.C3_DIR, "trace_vs_explanation_regression.csv")
    if not reg.empty:
      r = reg.iloc[0]
      out.append(f"\n**Regression** (does mentioning track flipping differently for traces than for "
                 f"explanations? flipped × thinking coefficient, log-odds): {round(r['estimate'], 2) + 0.0:+.2f} "
                 f"[{f2(r['lo'])}, {f2(r['hi'])}], p = {p_value(r['p_value'])}; {int(r['n_rows'])} cases, "
                 f"{int(r['n_clusters'])} questions\n")
  return "\n".join(out)


def write_summary(analyze_module) -> None:
  global A
  A = analyze_module
  text = "\n".join([
      "# Pilot results: summary\n",
      "The key numbers from every table, on one page. No interpretation. Brackets are 95% intervals "
      "(10,000 resamples of questions). What each run and test is: [NOTES.md](NOTES.md). Every table "
      "and column: [README.md](README.md). Real outputs: [EXAMPLES.md](EXAMPLES.md). Generated by "
      "`pilot/scripts/summarize.py` from the CSVs.\n",
      arm_r(),
      arm_c(),
  ])
  path = A.OUT / "SUMMARY.md"
  path.write_text(text)
  print("wrote", path.relative_to(A.ROOT) if path.is_relative_to(A.ROOT) else path)


def main() -> None:
  import analyze  # pilot/scripts is on sys.path when this file is run as a script.
  write_summary(analyze)


if __name__ == "__main__":
  main()
