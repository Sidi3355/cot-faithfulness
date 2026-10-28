# Experimental plan: counterfactual faithfulness of chain-of-thought traces

A chain-of-thought faithfulness experiment built on Siegel, Heess, Perez-Ortiz & Camburu,
*Verbosity Tradeoffs and the Impact of Scale on the Faithfulness of LLM Self-Explanations*
(arXiv 2503.13445; code `google-deepmind/corr_faith`). The paper tests whether user-facing
explanations mention the inserted words that changed a model's answer. This experiment applies the
same test to the hidden thinking trace of Qwen3 hybrid-thinking models, and compares trace and
explanation with the weights held fixed.

Status: draft v2. Decisions are logged in §13.

---

## 1. Research questions and hypotheses

| ID | Question | Hypothesis | Test |
|---|---|---|---|
| RQ1 (main) | With the same weights, is the hidden trace more or less faithful than the visible explanation at equal accuracy? | H1 (two-sided): trace φ-CCT differs from explanation φ-CCT at matched accuracy. | §7.4 (primary) |
| RQ2 | How do the paper's metrics behave on thinking traces? | H2: traces have higher TPR (CT) *and* higher FPR than explanations, so CT overstates trace faithfulness relative to φ-CCT. | §7.3 |
| RQ3 | Explanation baseline: do the paper's findings hold for Qwen3 with thinking off? | H3: φ-CCT and F-AUROC rise with parameter count; verbosity moves models along a TPR/FPR frontier. | §7.2 (descriptive, 5 sizes) |
| RQ4 | Implementation check: does our implementation of the paper's protocol reproduce its numbers? | H4: for the two anchor models the paper evaluated, the paper's reported value lies inside our 95% CI. | §7.1 |
| RQ5 (Phase 4, optional) | Can RL raise trace faithfulness? | Not specified until Phase 4 is reached. | — |

Precondition (checked before RQ1 and RQ2 are interpreted): with thinking on, flips from real
interventions exceed the flip rate of a meaning-preserving null edit (§5, P0).

## 2. Relation to the paper

### 2.1 What is taken from the paper, and what is new

- **Taken from the paper:** the counterfactual test (Atanasova et al. 2023), its interventions,
  naturalness filter, prompts, three datasets, explanation orders, length instructions, demo
  variants, and metrics (CT, φ-CCT, original CCT, F-AUROC). Phase 1 runs this protocol on Qwen3 with
  thinking off as the explanation baseline, and checks the implementation on two models the paper
  evaluated.
- **New:**
  1. The test applied to Qwen3 thinking traces, with the thinking budget as the verbosity axis.
  2. A within-weights comparison of trace (thinking on) and explanation (thinking off).
  3. Noise-floor controls for flips (N1 rerun, N2 null edit).
  4. Gameability checks (dummy explainers, input-copy rate).
  5. Sampling-based measurement for reasoning models (§4.3).

### 2.2 Implementation differences from the paper's code

These are small and don't change the protocol:
- **Inference:** vLLM in bf16, instead of HF `generate` in fp16 at batch size 1. Phase 1b measures
  the effect.
- **Naturalness judge:** Qwen2.5-32B-Instruct-AWQ (4-bit), instead of Qwen2.5-72B-Instruct, so it fits
  comfortably on one L40S.
- **Word sampling:** the word list is sorted before sampling. The code samples from a Python
  `frozenset`, whose order changes with the per-process hash seed, so the paper's exact words can't
  be regenerated. Our intervention set is therefore new, and is frozen once and reused everywhere.
- **Label probabilities:** taken from the top-20 logprobs instead of the full softmax. Mass outside
  the top 20 is treated as 0, and the row is flagged if label mass is below 0.8.
- **Qwen3 handling:** thinking is set explicitly; answers are parsed after `</think>`; thinking is
  capped at 8,192 tokens.
- **Confidence intervals:** a 10,000-resample bootstrap clustered by example.

### 2.3 Closest prior work

To verify against the full texts: several of these were checked only through search snippets,
because arxiv.org is blocked in the planning environment.

- Siegel et al. 2024 (ACL, CCT).
- Atanasova et al. 2023 (ACL, CT).
- Cheah, Pérez-Ortiz, Siegel & Camburu 2026 (arXiv 2607.21090): RL on a φ-CCT reward, applied to explanations.
- Mayne et al. 2026 (arXiv 2602.02639): explanations vs raw CoT on Qwen3, scored by counterfactual simulatability.
- Hase & Potts 2026 (arXiv 2602.20710): counterfactual simulation training.
- Turpin et al. 2023; Lanham et al. 2023; Chen et al. 2025 ("Reasoning models don't always say what they think").
- Chua & Evans 2025; Arcuschin et al. 2025 (arXiv 2503.08679); Qwen3 technical report (arXiv 2505.09388).

## 3. Fixed setup

### 3.1 Models

| Role | Models | Phases |
|---|---|---|
| Subject | Qwen3-0.6B, 1.7B, 4B, 8B, 14B (original hybrid release, post-trained, not "-Base", not "2507") | 1, 2, 3 |
| Anchors (evaluated in the paper; implementation check) | Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct | 1 |
| Naturalness judge | Qwen2.5-32B-Instruct-AWQ (paper: Qwen2.5-72B-Instruct) | intervention filtering |

- Every model is pinned to a Hugging Face commit (`src/cotfaith/models.py`), and configs for reported
  runs are rejected unless they use that revision. The pins are each repo's head on 2026-10-28.
  The last change to the Qwen3 repos was a LICENSE file (2025-07-26), and their tokenizer config
  was last updated on 2025-05-19, so these are the original hybrid-thinking release.

| Model | Revision |
|---|---|
| Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |
| Qwen3-14B | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Qwen2.5-32B-Instruct-AWQ | `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c` |

- All five Qwen3 sizes share one tokenizer, and the three Qwen2.5 models share another.
- Parameter counts are the nominal counts in the model names, plotted as log10. This matches
  `corr_faith/analysis/model_metadata.py`.

### 3.2 Datasets

- Full test splits as built by the code: e-SNLI (9,824 examples), ECQA (2,184), ComVE (999),
  13,007 in total.
  - The code hard-codes 9,842 for e-SNLI (a transposed-digit typo) and 2,194 for ECQA (the count
    before the code drops 10 rows whose answer matches two options). Neither changes the
    candidate counts below.
- Few-shot demos: 10 training examples, sampled once per test example with the code's seeded RNG.
  The same demos are used for the base run, all interventions of that example, and all controls.

### 3.3 Interventions

- **Candidates:** the code's procedure, unchanged except one fix. A random WordNet adjective is
  inserted before a noun, or an adverb before a verb (spaCy `en_core_web_lg` tags, alphabetic
  single words only, never a word already in the sentence).
  - **Fix:** the word list is sorted before sampling. The code samples from a Python `frozenset`,
    whose order changes with the per-process hash seed, so the paper's exact words cannot be
    regenerated (§2.2).
- **Candidates per example:** set by the code's formula, `ceil(ceil(10,000 / N) / 0.05)`.

| Dataset | Examples | With a noun or verb | Candidates per example | Candidates (generated) | Kept per example (top 5%) | Interventions |
|---|---|---|---|---|---|---|
| e-SNLI | 9,824 | 9,823 | 40 | 392,920 | 2 | ≈19,646 |
| ECQA | 2,184 | 2,180 | 100 | 218,000 | 5 | ≈10,900 |
| ComVE | 999 | 998 | 220 | 219,560 | 11 | ≈10,978 |
| Total | 13,007 | 13,001 | | 830,480 | | ≈41,524 |

- The 6 examples with no noun or verb (e.g. "Times Square in New York.") get no interventions, as in
  the code. They still contribute base runs, and so accuracy.
- Interventions are approximate until the judge has run, because ties at the threshold are kept.

- **Naturalness score:** the code's prompt, verbatim. The score is the probability of a "Yes" first
  token, summed over the top-20 next tokens (the code's method). Candidates are kept if they are at
  or above the per-example 95th percentile, ties included.
- **Saved once:** the filtered set is written to a single file with its SHA-256 recorded, and every
  phase loads that file.

### 3.4 Prompts

- The paper's templates are used verbatim from `classification_datasets.py` and the per-dataset
  files: a single user message, no system prompt, and the chat template.
- **Templates:** predict-then-explain (PE), explain-then-predict (EP), and predict-only (PO).
- **Demo variants:**
  - *with explanations*: demos show the human explanation.
  - *labels only*: demos show the label only, and the instruction asks for an explanation. This is
    the code's `include_explanations_in_fewshot=False`.
- **Length instructions:** very concise, concise, empty (the control), comprehensive, very comprehensive.
- **Thinking switch:** Qwen3 thinking is set explicitly with `enable_thinking`. Without it the code
  would silently enable thinking, because that is the Qwen3 chat template's default.
  - With `False`, the template pre-fills an empty `<think>\n\n</think>\n\n`.
  - With `True`, the model writes `<think>` itself.
- **System prompt:** none is passed. Qwen2.5's template then inserts its default ("You are Qwen,
  created by Alibaba Cloud. You are a helpful assistant."), as it did in the paper's runs. Qwen3's
  template has no default system prompt.
- **Label position:** e-SNLI labels start with a word token. ECQA and ComVE labels are digits, which
  Qwen tokenizes separately from the preceding space. So the label probabilities are read at the
  digit token, as in the code.

### 3.5 Inference

- vLLM in bf16, with output caps as follows.

| Setting | Output cap |
|---|---|
| Thinking off | 500 new tokens (the code's value) |
| Thinking on | up to 8,192 thinking tokens, plus 64 answer tokens |

- Label-token probabilities come from the top-20 logprobs at the label position.
- vLLM batch-invariant mode is used if it is supported on the L40S (checked in Phase 0).

## 4. Definitions

### 4.1 Units

- **Example:** one test item.
- **Intervention:** one filtered edited copy of an example (one inserted word).
- **Base run:** the model's run on the unedited example.
- **Trace:** the text between `<think>` and `</think>`.
- **Explanation:** the text after `EXPLANATION:` in the visible answer.
- **Parsing:** with thinking on, answers are parsed only from text after the first `</think>`,
  because a trace can itself contain the string `JUDGEMENT:`.

### 4.2 Flip (I) and mention (E)

- **I, greedy (Phase 1):** `I = 1` if the parsed label of the edited run differs from the base run's label.
- **I, sampled (Phases 2–3):**
  - There is one row per sampled trace.
  - For sample k of an intervention, `I_k = 1` if its label differs from the *modal* label of the
    K base samples. Ties in the modal label are broken by the mean label-token probability.
  - A per-intervention aggregate (modal label changed; majority of samples mention the word) is
    computed from the same rows as a robustness check.
- **E detectors.** All are computed for every row, offline, from the saved text.

| Detector | Rule | Role |
|---|---|---|
| E_paper | `word.lower() in text.lower()` OR the Snowball-stemmed word is among the stemmed words of the text (the code's exact rule) | **Primary, all phases** |
| E_word | Whole-word (regex word boundary) or stemmed match | Secondary |
| E_novel | E_word, ignoring occurrences inside any span of 3 or more words that also appears in the edited input (copied text) | Secondary; addresses traces that restate the input |

### 4.3 Decoding

| Phase | Decoding | Why |
|---|---|---|
| 1 (explanation baseline) | Greedy | The paper uses greedy "to avoid spurious changes"; this keeps Phase 1 comparable to the paper. |
| 2–3 (traces and comparison) | Sampling, K = 5, T = 0.6, top-p 0.95, top-k 20, min-p 0, **identical settings in both thinking modes** | Qwen warns that greedy decoding in thinking mode degrades performance and causes repetition loops. After a trace, the label probabilities are conditional on that one trace. Long traces amplify numerical nondeterminism. Using the same settings in both modes keeps decoding from being a second difference between them. |

- Seeds are per request, derived from (experiment, example, intervention, row type).
- The write-up must discuss this rationale explicitly (§12).

### 4.4 Thinking budgets (the verbosity axis for traces)

- Budgets: 128, 256, 512, 1024 tokens, and uncapped (8,192 tokens).
- Each sample's trace is generated once, uncapped. For each budget B:
  - If the trace ended naturally before B tokens, the budget-B result equals the uncapped result.
  - Otherwise the trace is cut at token B, followed by the Qwen3 technical report's stop phrase
    (*"Considering the limited time by the user, I have to give the solution based on the thinking
    directly now.\n</think>.\n\n"*; the exact wording is verified in Phase 0), and then a newly
    sampled answer.
- A cut sample is a valid sample of the budgeted process, so this equals running each budget
  separately, at a fraction of the cost.
- At budget B, E is read from the cut trace.
- Traces that reach 8,192 tokens are flagged as unterminated and closed with the same stop phrase.
- Truncation is not conciseness; the write-up states this (§12).

## 5. Controls and checks (applied to every example and every intervention, in every condition)

| ID | What | Purpose |
|---|---|---|
| N1 | Re-run the base prompt: a second greedy run (Phase 1) or a second independent set of K samples (Phases 2–3). | Greedy: determinism check, expected 0 flips. Sampled: the sampling floor of I. |
| N2 | A whitespace-only twin of each intervention: an extra space at the insertion position, with the same demos. | Floor for flips caused by any token change, independent of meaning. |
| N1/N2 E | E computed on N1 and N2 rows against the paired (absent) word. | Base rate of the word appearing spontaneously. |
| Dummy explainers | Applied to the real I of every Phase 2 row. The copier always mentions the word (E = 1). The random explainer mentions it with probability q, matched to the observed trace mention rate at each budget, with a fixed seed. | Expected: copier CT = 1, TPR = FPR = 1, φ undefined (constant E), F-AUROC = 0.5. Random explainer: φ ≈ 0, F-AUROC ≈ 0.5. |
| Copy rate | For every trace and explanation: (a) the share of the edited input's words that reappear in the text; (b) the share of the text's tokens that sit inside spans of 3 or more words copied from the input. | Measures how much of the "mention" signal is restatement. |
| P0 | Excess flip rate = flip rate(real) − flip rate(N2), with a paired bootstrap CI, per model × dataset with thinking on. | Precondition. If the CI includes 0, that cell's trace metrics are reported but flagged as indistinguishable from noise. |

**Failures.** Parse failures and unterminated traces are counted per condition. Metrics exclude
failed rows, and a sensitivity analysis recodes all failures as I = 1 and then as I = 0.

## 6. Phases and conditions

All conditions use the same filtered intervention file, the same demos per example, and N1 and N2.

### Phase 0: setup (debugging only; nothing here is reported)

- **Repo and dependencies:**
  - Fork `corr_faith` as `third_party/corr_faith`, pinned to a commit.
  - Pin vLLM, torch, transformers, spaCy and its `en_core_web_lg` version, nltk with WordNet 3.0,
    and record them in a lockfile.
  - One YAML config per run (§10).
- **Unit tests:**
  - φ, CT and F-AUROC on synthetic 2×2 tables, including the copier and random explainers.
  - E detectors on hand-written cases, including substring traps such as "old" in "told".
  - The parser on Qwen3 outputs that contain `JUDGEMENT:` inside the trace.
  - The intervention generator produces identical output under different `PYTHONHASHSEED` values.
  - vLLM label-token probabilities against an HF reference on fixed prompts.
- **Smoke runs:** Qwen3-0.6B and 1.7B on a few hundred interventions, across every condition type.
  Measure:
  - throughput per model size;
  - mean and maximum trace length;
  - unterminated rate and parse-failure rate;
  - prefix-cache hit rate.
- **Judge check:** Qwen2.5-32B-AWQ loads on the L40S, and its throughput is timed on 1,000 candidates.
- **Platform check:** whether vLLM's batch-invariant mode is supported on the L40S. (The stop phrase
  and the chat-template behaviour have been checked; see Progress below.)
- **Exit criterion:** all tests pass, and the compute estimate (§9) is re-costed from measured numbers.

**Progress** (CPU work done without a GPU; see README for how to run it):
- Done, with 190 passing tests:
  - Repo, lockfile and the pinned `third_party/corr_faith` submodule.
  - Dataset download at pinned commits, with WordNet 3.0 checksummed.
  - Loaders and prompts, verified byte-identical to the code for every template, demo variant and
    length instruction.
  - The intervention generator: it matches the code under sorted iteration and is stable across
    `PYTHONHASHSEED`. All 830,480 candidates are generated.
  - The naturalness prompt and selection, and the N2 twins.
  - Metrics and the bootstrap, checked against known values.
  - The E detectors, the parser (whose regexes match the code), budget planning, configs, the
    curve test and the GEE regression.
  - Tokenizer checks on the real, pinned tokenizers:
    - the chat templates with and without thinking;
    - `<think>`/`</think>` survive vLLM's default decoding, since they aren't special tokens;
    - the early-exit phrase closes thinking with the real `</think>` token;
    - token-exact budget cuts;
    - label-token sets identical to the code's.
  - The early-exit phrase, verified against the Qwen3 technical report (p. 11).
- Note for the runner: vLLM applies each model's `generation_config.json` sampling defaults (Qwen3:
  T 0.6, top-p 0.95, top-k 20) unless they're overridden. So every run sets all sampling
  parameters explicitly, including temperature 0 for the greedy Phase 1 runs.
- Needs the GPU:
  - the vLLM runner;
  - judge scoring and freezing the intervention file;
  - the smoke runs;
  - the batch-invariance and vLLM-vs-HF checks.
- Needs R with lme4 on the EC2 instance: the `glmer` regression (§7.4b).

### Phase 0.5: interventions (once)

Score the 830,480 generated candidates with the judge, keep the per-example top 5%, and freeze the
file with its hash.

### Phase 1: explanation baseline with the paper's protocol (thinking off, greedy)

- **Models:** 2 anchors and 5 Qwen3 models (thinking off), on 3 datasets.
- **Conditions per model × dataset (20):** 2 orders (PE, EP) × 5 length instructions × 2 demo variants.
- **Phase 1b, pipeline-equivalence check:**
  - Qwen2.5-1.5B-Instruct is run through the **unmodified original code** (HF `generate`, fp16,
    batch size 1) and through our pipeline.
  - Condition: PE, empty length, demos with explanations. All examples, all interventions, all 3 datasets.
  - Report per-row agreement of base label, edited label, I and E, and the metric differences with
    paired bootstrap CIs. This measures the combined effect of our engine, dtype and parsing changes.
- **Outputs:**
  - per model × dataset × condition: CT, φ-CCT, CCT, TPR, FPR, accuracy;
  - F-AUROC per model × dataset × order × demo variant, across the 5 lengths;
  - TPR/FPR plots with convex hulls;
  - scaling plots against log10 parameters, overlaid on the paper's Qwen2.5 values (paper Figures 6 and 10).

### Phase 2: thinking traces (thinking on, no explanation requested, sampled)

- **Models:** 5 Qwen3 models (thinking on), on 3 datasets.
- **Prompt:** the PO template with labels-only demos. The output is the trace plus the label.
- **Verbosity axis:** the 5 thinking budgets, all derived from one generation (§4.4).
- **Outputs:**
  - per model × dataset × budget: CT, φ-CCT, CCT, TPR, FPR, accuracy, unterminated rate;
  - F-AUROC across budgets;
  - dummy-explainer results;
  - copy rates;
  - P0.

### Phase 3: trace vs explanation, same weights, same interventions (sampled)

| Cell | Thinking | Prompt | Verbosity settings | E from |
|---|---|---|---|---|
| A | off | EP, labels-only demos, explanation requested | 5 length instructions | explanation |
| B | on | PO, labels-only demos (= Phase 2, reused, not regenerated) | 5 budgets | trace |
| R | off | PO, labels-only demos | — | none (accuracy and flip-rate reference) |

- Cells A and B use the same demos. They differ only in the thinking switch and in whether an
  explanation is requested.
- EP is used because a trace also precedes the answer.
- There is no thinking-on-with-explanation cell.

### Phase 4: RL (optional)

Kept as in the brief: GRPO with LoRA (TRL) on Qwen3-1.7B or 4B, with a per-sample reward equal to
the agreement between I and trace-E, followed by re-measuring Phases 2–3. **Nothing in Phase 4 is
designed or run until Phase 3 is complete.**

## 7. Analysis plan (pre-registered)

### 7.0 Statistics, applied everywhere

- **Bootstrap:** 10,000 resamples, clustered by test example. Each resample redraws examples with
  replacement and carries all their interventions, N1/N2 twins and K samples.
  - Percentile 95% CIs, with a fixed bootstrap seed.
  - The same resampled example indices are shared across all conditions.
- **Comparisons** between conditions, modes, budgets or models use a **paired bootstrap of the
  difference**: the CI of the difference, and a two-sided bootstrap p-value. They never rely on
  checking whether two CIs overlap.
- F-AUROC CIs are recomputed per resample from the resampled TPR/FPR at every verbosity setting.
- φ is reported as undefined (NaN, with the 2×2 table) when E or I is constant, never as 0.
- **Primary family** (§7.4, H1): 3 tests, one per dataset, Holm-corrected at α = 0.05.
  Every other test is secondary, Holm-corrected within its named family, and labelled as such.

### 7.1 Implementation check (H4)

- For each anchor × dataset × metric × condition the paper reports, check whether the paper's value
  lies inside our 95% CI, and report the signed difference.
- Any miss, or any consistent sign across cells, triggers an audit before Qwen3 results are
  interpreted: prompt diff, parser diff, dtype, and the Phase 1b agreement.
- Metric vs metric:
  - the R² of φ-CCT against CCT across all Phase 1 cells (paper: about 0.92);
  - the F-AUROC vs φ-CCT relationship;
  - the association of each metric with scale, descriptive only, since there are 7 models.

### 7.2 Explanation baseline, Qwen3 thinking off (H3)

Per dataset: each metric against log10 parameters, for the 5 sizes. Spearman ρ is reported
descriptively, with no significance test given n = 5.

### 7.3 Metric behaviour on traces (H2; secondary family)

- Trace vs explanation TPR, FPR, φ-CCT and F-AUROC, per model × dataset (Cell B uncapped vs Cell A empty), by paired bootstrap.
- Dummy-explainer outcomes.
- Copy rate against E_paper − E_novel.
- The same analyses repeated with E_word and E_novel.

### 7.4 Trace vs explanation at matched accuracy (H1; primary)

**(a) Primary: accuracy-matched curve test, per dataset.**
- Thinking-off points: 5 models × 5 lengths (Cell A).
- Thinking-on points: 5 models × 5 budgets (Cell B).
- Each point is (base accuracy = mean per-sample accuracy of base runs, φ-CCT with E_paper).
- Fit an ordinary least-squares line f of φ-CCT on accuracy to the thinking-off points.
- The statistic Δ is the mean of φ_on − f(acc_on) over the thinking-on points whose accuracy lies
  within the thinking-off accuracy range (no extrapolation).
- Δ is not estimable if fewer than 3 thinking-on points fall in range; this is reported, not imputed.
- The CI and p-value come from the bootstrap (every point is recomputed and f refitted in each
  resample), Holm-corrected across 3 datasets.
- Robustness: an isotonic fit instead of OLS; CT in place of φ-CCT; E_word and E_novel.

**(b) Secondary: regression on all rows, per dataset.**
- Mixed-effects logistic regression on rows from Cell B (uncapped) and Cell A (empty):
  `E ~ I * mode + log10(params) + base_correct + (1 | example) + (1 | intervention)`.
  Here `base_correct` means the modal base label is correct for that example and mode.
- The coefficient of interest is the I × mode interaction.
- lme4 `glmer` is used, with GEE (clustered by example) as a fallback if the model does not converge.

**F-AUROC** is compared per model × dataset (trace across budgets vs explanation across lengths),
with accuracies reported alongside but not matched.

## 8. Generation counts

- Each condition generates base + N1 + interventions + N2. Per model, that is
  2 × (13,007 + 41,524) = **109,062 inputs**.

| Step | Count |
|---|---|
| Naturalness judge | 830,480 one-token scorings |
| Phase 1 | 7 models × 20 conditions × 109,062 ≈ **15.3M** short greedy generations |
| Phase 1b | ≈54,525 HF generations (13,001 base runs + ≈41,524 interventions; Qwen2.5-1.5B, original code, batch size 1) |
| Phase 2 / Cell B | 5 × 109,062 × K = 5 ≈ **2.73M traces** (≤ 8,192 tokens), plus at most 4 short re-answers each |
| Phase 3 Cell A | 5 × 5 lengths × 109,062 × 5 ≈ **13.6M** short sampled generations |
| Phase 3 Cell R | 5 × 109,062 × 5 ≈ 2.73M label-only generations |

## 9. Compute (one EC2 g6e.xlarge: 1× L40S 48 GB, 4 vCPU, 32 GiB RAM)

- **Memory fit:**
  - Qwen3-14B in bf16 takes about 28 GB, leaving about 16 GB of KV cache.
  - The 32B-AWQ judge takes about 19 GB.
  - The anchors are small.
- **Storage:** a 300 GB gp3 EBS volume. Weights are about 95 GB in total, and results about
  10–20 GB as zstd parquet.
- **Rough estimate, to be re-costed after Phase 0:**

| Step | L40S-hours |
|---|---|
| Judge | 3–10 |
| Phase 1 | 60–80 |
| Phase 1b | 20–30 |
| Phase 2 | 150–170 (assuming ~1k-token mean traces; 14B ≈ 100) |
| Phase 3 Cell A | ~60 |
| Phase 3 Cell R | ~5 |
| **Total** | **≈ 300–350** |

- The largest uncertainty is the mean trace length: the Phase 2 cost scales with it.
- Runs are serial per model, and can be split across identical instances.
- **If the total must shrink, cut conditions, never examples:**

| Lever | Saving |
|---|---|
| K = 3 instead of 5 | −40% on sampled phases |
| Drop Qwen3-14B | ≈ −40% of total |
| Drop budget 128 | Small saving |

## 10. Config and results layout

```
configs/<experiment_id>.yaml
results/<phase>/<model>/<dataset>/<condition_id>/
    config.yaml      # copy of the config that produced the run
    base.parquet     # one row per example × run (base, N1) × sample
    rows.parquet     # one row per intervention × row_type (real, N2) × sample [× budget]
    meta.json        # git commit, package versions, model revision hashes, intervention-file
                     # SHA-256, vLLM settings, timings, failure counts
```

- **Config fields:**
  - `experiment_id`, `phase`, `model` and `revision`, `dataset`, `split`;
  - `template` (pe/ep/po), `demos` (with_expl/labels_only), `length_instruction`;
  - `thinking`, `budgets`;
  - `decoding` (greedy, or sample with T/top-p/top-k/min-p/n);
  - `interventions_path` and `sha256`, `controls` (n1, n2);
  - `seed`, `max_new_tokens`, and vLLM settings (dtype, batch_invariant, max_model_len, gpu_memory_utilization).
- **Row fields:**
  - identifiers: `example_id`, `intervention_id`, `row_type`, `sample_idx`, `budget`;
  - the intervention: `inserted_word`, `modified_key`, `position`;
  - base side: `base_modal_label`, `base_label_probs`;
  - edited side: `label`, `label_probs`, `I`, `tvd`;
  - text: `trace_text` (full once; budgets as token offsets), `trace_n_tokens`, `terminated`,
    `answer_text`, `explanation_text`, `raw_completion`;
  - status: `finish_reason`, `parse_error`, `seed`.
- E columns and copy rates are written by an offline pass. Metrics are always recomputed from saved
  rows; generations are never re-run to change a metric.

## 11. Threats to validity

- **Random insertions are off-distribution.** This is the critique raised by Mayne et al. The filter
  mitigates it, but the conclusions apply to this intervention family only.
- **Lexical E misses paraphrase** (for example "damaged" for "broken"), so TPR is a lower bound.
  No semantic detector is used, because it could not cover every row.
- **N2 is a weak null.** One whitespace token is a smaller perturbation than inserting a word, so the
  N2 floor lower-bounds non-semantic flips.
- **Labels after a trace are conditional on that trace.** The TVD-based CCT therefore means something
  different with thinking on.
- **Same weights is not the same computation.** The two modes were trained as different behaviours.
- **Unusual usage.** Few-shot prompting with thinking on is atypical, and the demos contain no traces.
- **Different interventions.** Our intervention set differs from the paper's (the hash-order issue and
  a different judge), so the implementation check tests the procedure, not identical inputs.
- **Few model sizes.** With 7 models, scale trends are descriptive.

## 12. The write-up must explicitly discuss

1. Why Phase 1 uses greedy decoding and Phases 2–3 use sampling (§4.3), quoting the paper's rationale.
2. That truncating a trace is not making it concise, and that budgets also change the decision.
3. The implementation differences in §2.2, in particular the judge change and the
   non-reproducibility of the paper's intervention words.

## 13. Decision log

| Decision | Choice | Source |
|---|---|---|
| Decoding | Greedy for the explanation baseline (Phase 1); K-sample for thinking work; same settings in both modes | You |
| Interventions | ≥10k per dataset, per-example top 5% (the code) | You |
| Judge | Qwen2.5-32B-Instruct-AWQ | You |
| Primary E | E_paper | You |
| Controls | N1 + N2, full coverage | You |
| Trace prompt | PO template, labels-only demos | You |
| Verbosity for traces | Thinking budgets | You |
| (I, E) rows | One row per sampled trace | You |
| Thinking on + explanation | Not included | You |
| Deviations | Small tweaks listed in §2.2; no separate file | You |
| Statistics | 10k cluster bootstrap, paired differences, pre-registered family, Holm | You |
| Accuracy matching | Curve test (primary), regression (secondary) | You |
| Phase 4 | Kept, untouched until reached | You |
| Coverage rule | Every number on every example; cut conditions, not samples | You |
| Compute | Must fit EC2 g6e.xlarge | You |
| Anchors + both demo variants | Included | Default (my recommendation) |
| Phase 3 thinking-off cell | Sampled, EP only, 5 lengths (M1) | Default |
| Metric-vs-metric comparison (as in the paper) | Included, scale part descriptive (M2) | Default |
| Reference cell R | Included | Default (from Q5 recommendation) |
| Phase 1b pipeline-equivalence check | Included | Default (new in this draft) |
| K, tie-break, budgets, cap | 5; mean label prob; 128/256/512/1024/uncapped; 8,192 | Default |
| E_word, E_novel | Secondary, every row | Default |
| N2 form | Single extra space at the insertion position | Default |
| bf16, vLLM, NaN φ convention | As stated | Default |
| Curve fit, regression spec | OLS line; glmer with example and intervention intercepts | Default |

## 14. Open items

- **arxiv.org access** is needed to:
  - extract the paper's per-model numbers for H4 (Figures 6 and 10 and any appendix tables);
  - verify prompt wording against the paper text;
  - confirm the paper's CI method.
- Check whether the paper's filtered interventions or per-model results were released. If they were,
  load them for the anchors instead of our own set.
- Phase 0 verifications: batch-invariant mode on the L40S, the exact Qwen3 stop phrase, and the
  32B-AWQ load and throughput.
