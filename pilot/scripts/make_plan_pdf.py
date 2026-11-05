"""Builds pilot/PILOT_PLAN.pdf. Needs reportlab.

Usage: python pilot/scripts/make_plan_pdf.py
"""

import pathlib

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "PILOT_PLAN.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=2)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.3, leading=12.2, alignment=TA_LEFT)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.3, leading=10.5)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.2, leading=10.2)
CELL_B = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")


def p(text, style=BODY):
  return Paragraph(text, style)


def bullets(items, style=BODY):
  return ListFlowable(
      [ListItem(p(i, style), leftIndent=10, value="•") for i in items],
      bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8,
  )


def table(rows, widths, header=True):
  data = [[p(c, CELL_B if (header and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
  t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if header else 0)
  t.setStyle(TableStyle([
      ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
      ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3") if header else colors.white),
      ("VALIGN", (0, 0), (-1, -1), "TOP"),
      ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
      ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
  ]))
  return t


story = [
    p("Pilot plan", H1),
    p("Counterfactual faithfulness of explanations and chain-of-thought traces: a small-scale "
      "replication of Siegel et al. (arXiv 2503.13445) and a pilot of the chain-of-thought "
      "extension. CPU only, on the Claude Code cloud machine. Companion to PLAN.md, which "
      "describes the full study.", SMALL),
    Spacer(1, 4),

    p("1. Purpose", H2),
    bullets([
        "<b>Arm R, replication:</b> reproduce three of the paper's findings on three of the paper's "
        "own models, at small scale.",
        "<b>Arm C, chain-of-thought pilot:</b> run the full design of the new experiment once, "
        "end to end, on one small model. The goal is to debug the pipeline, produce every table "
        "and figure, and measure the quantities that size the full study (trace lengths, flip "
        "rates, noise floors).",
        "<b>Output:</b> tables, statistics and figures in one folder, <font face='Courier'>pilot/results/</font>. "
        "There is no written interpretation; conclusions are yours.",
    ]),

    p("2. Shared setup (both arms)", H2),
    table([
        ["Item", "Setting"],
        ["Dataset", "ComVE test set (999 examples). Examples are taken in the paper's fixed "
         "shuffle order, so 'the first N examples' is reproducible."],
        ["Task", "Two sentences; which one violates common sense? Label 0 or 1."],
        ["Interventions", "The paper's procedure: a random WordNet adjective before a noun, or "
         "adverb before a verb. 220 candidates per example (already generated). A judge scores "
         "each candidate; the top 5% per example are kept, which is 11 per example. Result: 1,837 "
         "interventions from 167 examples."],
        ["Judge", "Qwen2.5-3B-Instruct (paper: Qwen2.5-72B). It scores the probability that its "
         "first answer token is 'Yes' to the paper's naturalness question. It ran once, on the "
         "first 167 examples (36,740 candidates), in bf16. It stopped at 167 (planned: 200) when a "
         "restart moved the session to a CPU without bf16 hardware, to keep one number format for "
         "the judge. Like the paper's judge, it is also one of the subject models."],
        ["Prompts", "The paper's prompts, verbatim (tested byte-identical): task description, "
         "10 training demos (the same for an example and all its edits), instruction, test example."],
        ["Engine", "Hugging Face transformers on CPU, float32 (the paper also used transformers, in fp16 "
         "on GPU). float32 because a restart can move the session to a different CPU: it is fast on "
         "CPUs with and without bf16 hardware, and gives identical numbers on both. Models are "
         "pinned to exact Hugging Face commits."],
        ["Label probabilities", "Full softmax at the label token, exactly as the paper computes "
         "them. They give the original CCT."],
        ["Mention (E)", "The paper's rule (primary). Two secondary detectors are recorded as well: "
         "whole-word match, and whole-word match ignoring text copied from the input."],
        ["Statistics", "95% CIs from 10,000 bootstrap resamples of whole examples. Comparisons "
         "between conditions use paired differences on identical interventions. Holm correction "
         "within each family of tests."],
    ], [32, 138]),

    p("3. Arm R: small-scale replication", H2),
    table([
        ["Item", "Setting"],
        ["Models", "Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct (all evaluated "
         "in the paper; 7B and larger don't fit in this machine's 15 GB RAM)."],
        ["Conditions", "Predict-then-explain, demos with human explanations, 3 of the paper's 5 length "
         "instructions: very concise, none, very comprehensive (chosen after the pre-pilot to fit "
         "the CPU). 3 conditions per model."],
        ["Decoding", "Greedy, max 500 new tokens (the paper's settings)."],
        ["Per example", "Base run + its 11 interventions. Also one rerun of the base prompt (N1) as "
         "a determinism check, which should give zero flips."],
        ["Sample size", "Pre-pilot on the first 20 examples (no-length-instruction condition) to "
         "measure flip rates. Then N = the smallest of 50 / 100 / 150 whose projected 95% "
         "CI half-width for phi-CCT is at most 0.05 for every model (projected = pre-pilot "
         "half-width x sqrt(20/N)); 150 if none (167 examples were judged). The pre-pilot "
         "examples are included in the N."],
    ], [32, 138]),
    Spacer(1, 3),
    p("Findings tested", H3),
    table([
        ["ID", "Paper finding", "What is computed"],
        ["F1", "Verbosity tradeoff: asking for longer explanations raises TPR and FPR together "
         "(models move along a frontier).",
         "TPR and FPR at each length; paired difference very comprehensive minus very concise, "
         "per model. TPR/FPR plot with the convex hull, and F-AUROC per model."],
        ["F2", "Larger models are more faithful.",
         "phi-CCT, CT and F-AUROC per model; paired differences 1.5B-0.5B, 3B-1.5B, 3B-0.5B "
         "(Holm within each metric)."],
        ["F3", "phi-CCT tracks the original CCT (paper: R-squared about 0.92).",
         "phi-CCT against CCT across the 9 model x length cells; R-squared."],
    ], [10, 72, 88]),
    Spacer(1, 3),
    p("Comparison with the paper's reported values for these three models needs the paper "
      "text (arxiv.org is blocked in this session). The comparison table is created with "
      "those columns left empty until it's available.", SMALL),

    p("4. Arm C: chain-of-thought pilot", H2),
    table([
        ["Item", "Setting"],
        ["Model", "Qwen3-0.6B (hybrid thinking: the same weights with thinking on or off)."],
        ["Examples", "The first 40 examples (a subset of Arm R's), each with all 11 interventions: "
         "about 440 interventions."],
        ["Controls", "On every example and intervention. N1: rerun of the base prompt. N2: the same "
         "position with only an extra space inserted (no new word)."],
        ["Sampling", "K = 3 samples per prompt (full study: 5); temperature 0.6, top-p 0.95, "
         "top-k 20, the same in both thinking modes. One row per sample; a flip means the "
         "sample's label differs from the majority label of the 3 base samples."],
    ], [32, 138]),
    Spacer(1, 3),
    table([
        ["Part", "Thinking", "Prompt", "Verbosity settings", "Decoding", "E read from"],
        ["C1 explanation baseline", "off", "PE and EP; demos with explanations and labels-only",
         "5 lengths (20 conditions)", "greedy", "explanation"],
        ["C2 thinking traces", "on", "predict-only, labels-only demos (no explanation asked)",
         "budgets 128, 256, 512, 1024, uncapped (pilot cap 4,096 tokens)", "K = 3", "trace"],
        ["C3 cell A", "off", "EP, labels-only demos", "5 lengths", "K = 3", "explanation"],
        ["C3 cell B", "on", "= C2 (reused)", "5 budgets", "K = 3", "trace"],
        ["C3 cell R", "off", "predict-only, labels-only demos", "-", "K = 3", "none (reference)"],
    ], [26, 17, 41, 35, 18, 33]),
    Spacer(1, 3),
    bullets([
        "<b>Budgets:</b> each trace is generated once. For a budget B, a longer trace is cut "
        "after B thinking tokens and closed with Qwen's own stop phrase (checked against the Qwen3 "
        "technical report), then a fresh answer is sampled.",
        "<b>Parsing:</b> upstream's answer patterns. Explain-then-predict answers (C1 EP, C3 cell "
        "A) that they reject are retried with a pattern allowing a multi-line explanation and a "
        "full stop after the label; answers upstream accepts parse identically.",
        "<b>Computed:</b> all metrics and CIs, as in Arm R. Also: flip rates for real edits vs N1 "
        "and N2 (the noise floor); the dummy explainers (always mention; mention at random); copy "
        "rates (how much of the trace restates the input); and the parse-failure and "
        "unfinished-trace rates, with metrics recomputed counting failures as all-flip and as "
        "no-flip.",
        "<b>Trace vs explanation:</b> phi-CCT against accuracy for the 5 thinking-off points (cell "
        "A) and the 5 thinking-on points (cell B), the curve test (mean distance of the thinking-on "
        "points from the thinking-off line), and a regression of E on I x thinking mode. With one "
        "model these are thin, and serve to exercise the analysis.",
    ], SMALL),

    p("5. Order of work and estimated time", H2),
    table([
        ["Step", "What", "Est. hours"],
        ["0", "Build the CPU runner; check it against the paper's code on the same prompts; tests", "-"],
        ["1", "Download weights for the 4 models; judge the examples; freeze interventions (done: 167 examples)", "done"],
        ["2", "Arm R pre-pilot (20 examples x 3 models) and the sample-size rule", "≈ 1"],
        ["3", "Arm R main run (150 examples x 3 models x 3 lengths)", "≈ 2.5 days"],
        ["4", "Arm C: C1 (20 conditions)", "≈ 5"],
        ["5", "Arm C: C2 thinking traces (about 2,900 traces)", "≈ 13"],
        ["6", "Arm C: C3 cells A and R", "≈ 4"],
        ["7", "Tables, statistics and figures into pilot/results/", "< 1"],
    ], [12, 128, 30]),
    p("Times are scaled from benchmarks on this machine (pilot/benchmarks/). Runs are split into "
      "resumable chunks, and finished results are pushed to the repository as they complete, "
      "because the machine is reclaimed after inactivity.", SMALL),

    p("6. Output folder: pilot/results/", H2),
    table([
        ["Path", "Contents"],
        ["interventions.csv", "Every kept intervention: example, sentence, inserted word, position, judge score."],
        ["R/*.csv, R/*.png", "Per model x length: accuracy, flip rate, mention rate, TPR (CT), FPR, "
         "phi-CCT, CCT with CIs; F-AUROC; F1/F2 paired differences; F3 R-squared; pre-pilot sample "
         "size table; empty paper-comparison table; figures."],
        ["C/*.csv, C/*.png", "The same metrics for C1, C2 (per budget) and C3; noise floors; dummy "
         "explainers; copy rates; failure sensitivity; curve test; regression; figures."],
        ["raw/", "One row per model output (prompt id, completion, trace, label, probabilities, E, I), "
         "so every number can be recomputed without rerunning models."],
        ["run_log.csv", "Every run: config, model revision, package versions, timings, failure counts."],
    ], [32, 138]),

    p("7. Compared with the full plan", H2),
    table([
        ["", "Full plan (EC2, GPU)", "This pilot (CPU)"],
        ["Models", "5 Qwen3 sizes + 2 Qwen2.5 anchors", "R: Qwen2.5 0.5B / 1.5B / 3B. C: Qwen3-0.6B"],
        ["Datasets", "e-SNLI, ECQA, ComVE", "ComVE"],
        ["Examples", "All test examples", "R: N (50, 100 or 150). C: 40"],
        ["Judge", "Qwen2.5-32B-AWQ", "Qwen2.5-3B-Instruct"],
        ["Engine", "vLLM, bf16", "transformers on CPU, float32"],
        ["Samples per input", "5", "3"],
        ["Thinking cap", "8,192 tokens", "4,096 tokens"],
        ["Base vs instruction-tuned", "-", "Not included"],
        ["Claims", "Pre-registered tests", "R: small-scale replication of F1-F3 (wide CIs). C: none; "
         "it is a feasibility and sizing pilot"],
    ], [34, 62, 74]),
]

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm,
    bottomMargin=16 * mm, title="Pilot plan", author="cot-faithfulness",
)
doc.build(story)
print(OUT)
