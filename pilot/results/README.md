# Pilot results

All pilot output: tables, figures and every raw model response. There is no write-up. For what
each run and test is, read **[NOTES.md](NOTES.md)** first.

```
results/
├── NOTES.md                         what was run and what each test measures
├── interventions/                   the inserted words used by every run
├── R_replication/                   arm R: Qwen2.5-0.5B/1.5B/3B-Instruct, 150 questions
├── C_chain_of_thought/              arm C: Qwen3-0.6B, 40 questions
│   ├── C1_explanations/             thinking off, written explanations (20 conditions)
│   ├── C2_thinking_traces/          thinking on, trace = explanation (5 budgets)
│   └── C3_trace_vs_explanation/     explanations sampled like C2, and the trace-vs-explanation tests
├── logs/                            run log and Kaggle logs
└── raw/                             every model response, one file per question
```

All CSVs open as tables on GitHub. Columns ending in `_lo` / `_hi` are the 95% interval of the
column before them.

## interventions/

| File | Contents |
|---|---|
| `interventions.csv` | The 1,837 inserted words: question id, which sentence was edited, the word, the edited sentences, and the judge's naturalness score. |
| `interventions_meta.json` | Judge model, number of questions judged (167), candidates (36,740), share kept (5%), file checksum. |

## R_replication/

| File | Contents |
|---|---|
| `cells.csv` | **Main table.** One row per model × length: accuracy, flip rate, CT, FPR, φ-CCT, CCT, noise and parse-failure rates, answer-position bias. |
| `f_auroc.csv` | F-AUROC per model, over its 3 lengths. |
| `F1_verbosity.csv` | Very comprehensive minus very concise, per model, for CT and FPR. |
| `F2_scale.csv` | Differences between model sizes for φ-CCT, CT, accuracy and F-AUROC. |
| `F3_phi_vs_cct.csv` | How closely φ-CCT follows CCT across the 9 cells (R², Pearson r). |
| `paper_comparison.csv` | Our range over the 3 lengths next to values read off the paper's ComVE figures. |
| `prepilot_sample_size.csv` | The 20-question pre-pilot used to choose 150 questions. |
| `tpr_fpr_by_length.png` | CT against FPR for each model, one point per length. |
| `metrics_by_size.png` | φ-CCT, CT and accuracy by model size (no length instruction), with intervals. |
| `phi_vs_cct.png` | φ-CCT against CCT, one point per cell. |

## C_chain_of_thought/

**C1_explanations/**

| File | Contents |
|---|---|
| `C1_cells.csv` | **Main table.** One row per order × example type × length (20 rows): the same metrics as arm R, plus N1/N2 noise. |
| `C1_f_auroc.csv` | F-AUROC per order × example type, over its 5 lengths. |

**C2_thinking_traces/**

| File | Contents |
|---|---|
| `C2_cells_by_budget.csv` | **Main table.** One row per budget: the same metrics plus noise, forced-answer rate, trace length, and the E_word / E_novel versions of CT, FPR and φ-CCT. |
| `C2_f_auroc.csv` | F-AUROC over the 5 budgets. |
| `C2_dummy_explainers.csv` | The same metrics for an "always mention" and a "mention at random" explainer. |
| `C2_copy_rates.csv` | How much of each trace repeats the question. |
| `C2_failure_sensitivity.csv` | Parse-failure and unfinished-trace rates, and CT/φ-CCT with failures counted as flips or as no flips. |
| `trace_metrics_by_budget.png` | CT, FPR and φ-CCT by budget. |

**C3_trace_vs_explanation/**

| File | Contents |
|---|---|
| `C3A_cells.csv` | **Main table.** Explanations sampled like C2, one row per length. |
| `C3A_f_auroc.csv` | F-AUROC over the 5 lengths. |
| `C3R_reference.csv` | Answer only, no explanation: accuracy, flip rate, noise. |
| `trace_vs_explanation_curve_test.csv` | Curve test, with a straight-line fit (`ols`) and a non-decreasing fit (`isotonic`). `delta` is blank when fewer than 3 thinking points fall in the explanations' accuracy range. |
| `trace_vs_explanation_regression.csv` | The flipped × thinking-mode coefficient (log-odds), with interval and p-value. |
| `tpr_fpr_explanation_vs_trace.png` | CT against FPR: explanations by length next to traces by budget. |
| `phi_vs_accuracy_trace_vs_explanation.png` | φ-CCT against accuracy for both, with intervals. |

## logs/

| File | Contents |
|---|---|
| `run_log.csv` | One row per run folder: model revision, code commit, library versions, questions done, hours. |
| `kaggle_c2.log`, `kaggle_c13.log` | Console output of the arm C Kaggle sessions. |

## Columns in the main tables

| Column | Meaning |
|---|---|
| `cell` | The condition (model, length, order, example type or budget). |
| `n_examples` | Questions with a readable original answer (the others are left out). |
| `n_rows` | Inserted-word cases counted. |
| `n_flips` | Of those, how many changed the answer. |
| `accuracy` | Share of original answers that are correct. |
| `flip_rate`, `real_flip_rate` | Share of inserted words that changed the answer. |
| `mention_rate` | Share of inserted words the explanation mentions, flip or not. |
| `tpr_ct` | CT: share of answer-changing words that are mentioned. |
| `fpr` | Share of non-changing words that are mentioned. |
| `phi_cct` | Correlation between changed and mentioned. |
| `cct` | The older probability-based version of φ-CCT. |
| `n1_flip_rate` | Answer changes when the same question is asked again (noise). |
| `n2_flip_rate` | Answer changes after adding only a space (noise). |
| `excess_flip_rate` | `real_flip_rate` minus the noise. |
| `n2_spontaneous_mention_rate` | How often the word is mentioned even though it was never inserted. |
| `parse_failure_rate` | Share of responses whose answer could not be read. |
| `base_parse_failure_rate` | The same, for the original questions only. |
| `base_pred_label1_rate`, `gold_label1_rate` | Share of original answers that pick option 1, and share where option 1 is correct. |
| `forced_answer_rate` | Share of traces cut by the budget, so the answer was forced. |
| `mean_trace_tokens` | Average trace length after the cut. |
| `E_word_*`, `E_novel_*` | CT (`tpr`), FPR and φ-CCT with the stricter mention detectors (see NOTES.md). |
| `copy_reused_mean` | Share of the question's distinct words that reappear in the trace. |
| `copy_copied_mean` | Share of the trace that sits inside 3-word phrases copied from the question. |
| `E_paper_minus_E_novel` | Share of cases that count as mentions only because of copied text. |

## raw/

Every model response, as Parquet files (open with `pandas.read_parquet`):

```
raw/<phase>/<model>/comve/<order>-<examples>-<length>-<think|nothink>-<greedy1|sample3>[-tolerant]/rows/<question id>.parquet
```

- `phase1`: arm R and C1. `phase2`: C2. `phase3`: C3A (`ep-…-sample3-tolerant`) and C3R
  (`po-…-sample3`).
- `pe` / `ep` / `po`: predict then explain / explain then predict / predict only.
- Folders ending `-tolerant` use the looser answer reading. The `ep` folders without it are the
  earlier runs, kept but not used.
- Each folder also has `config.yaml` (exact settings), `meta.json` (versions, GPU) and
  `timing.csv`.

Main columns:
- `row_type`: `base` (original), `real` (inserted word), `n1`, `n2`.
- `inserted_word`.
- `sample_idx`: 0–2 in sampled runs.
- `budget`: C2 only.
- `answer_text`: the model's output (after the thinking, if any).
- `trace`: the thinking text.
- `label`: the answer that was read.
- `explanation`.
- `parse_error`.
- `label_probs`.
- `forced`, `terminated`.

To regenerate every table and figure from `raw/` (needs the ComVE data: `DATASETS=comve scripts/fetch_data.sh`):

```bash
.venv/bin/python pilot/scripts/analyze.py r
```

```bash
.venv/bin/python pilot/scripts/analyze.py c
```
