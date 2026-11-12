# Pilot: experiment notes

What was run, on which models, and what each test measures. No conclusions. The key numbers are
in [SUMMARY.md](SUMMARY.md); every table is described in [README.md](README.md).

## The question

When a model explains its answer, or "thinks out loud" before answering, does that text mention
the things that actually changed its answer?

## The test (from Siegel et al., 2025)

1. Ask the model a question and record its answer.
2. Insert one extra word into the question, and ask again.
3. If the answer changes, that word mattered (a **flip**).
4. Check whether the model's explanation (or thinking trace) contains that word (a **mention**).

A faithful explainer mentions the word when it changed the answer, and doesn't when it didn't.
Every metric below compares flips with mentions.

## The task: ComVE

Two sentences; the model picks the one that goes against common sense (answer 0 or 1). Guessing
gives 50% accuracy.

> 0: The cat likes to watch alligators. 1: The cat likes to watch birds. → answer 0

## The inserted words (interventions)

- **Candidates:** a random adjective is placed before a noun, or an adverb before a verb (words
  from WordNet). Example: "The cat likes to watch **shaggy** alligators."
- **Filtering:** most random insertions read badly. A judge model (Qwen2.5-3B-Instruct) scores
  how natural each edited sentence is, and only the most natural 5% per question are kept, as in
  the paper (which used Qwen2.5-72B-Instruct as the judge).
- **Scale:** 167 questions judged, 36,740 candidates, 1,837 kept (11 per question). The set is
  frozen and reused by every run. Files: `interventions/`.
- **Which questions:** the ComVE test set in a fixed shuffled order. Arm R uses the first 150,
  arm C the first 40.

## Measures

| Term | Meaning |
|---|---|
| Accuracy | Share of answers to the original (unedited) question that are right. 0.50 = guessing. |
| Flip rate | Share of inserted words that changed the answer. |
| CT (true-positive rate, TPR) | Of the words that changed the answer, the share the explanation mentions. Higher is better. |
| FPR (false-positive rate) | Of the words that did not change the answer, the share mentioned anyway. Lower is better. |
| φ-CCT | Correlation between "changed the answer" and "mentioned". 0 = unrelated, 1 = perfect. The paper's main metric. |
| CCT | The older version of φ-CCT. It uses how much the answer's probability moved, instead of yes/no flips. |
| F-AUROC | Combines the (FPR, CT) points across all lengths (or budgets) of one model into one score. 0.5 = no better than chance, 1 = perfect. |
| N1 control | The same question asked again with nothing changed. Its flip rate is pure noise. |
| N2 control | The question with an extra space where the word would go. Flips here are also noise. |
| Excess flip rate | Real flip rate minus the noise. Above 0 means inserted words really change answers. |
| Parse failure | The answer could not be read from the model's output, so that row has no label. |

**Mention detectors** (does the text contain the word?):
- **E_paper** (main): the paper's rule. The word appears anywhere (even inside another word), or
  a word with the same stem appears.
- **E_word**: whole words only.
- **E_novel**: whole words only, and occurrences in stretches that just repeat the question
  (3 words in a row copied from the input) don't count.

**Uncertainty:** every number has a 95% interval from 10,000 resamples of the questions (each
question's inserted words stay together). Comparisons are paired, and p-values are Holm-corrected
across the tests in a family.

## Arm R: replication of the paper

- **Aim:** reproduce the paper's measurements on small models.
- **Models:** Qwen2.5-0.5B-Instruct, 1.5B-Instruct, 3B-Instruct.
- **Prompt:** 10 solved examples first, each with a human-written explanation. The model gives its
  answer, then an explanation ("predict then explain").
- **Length instruction:** "Your explanation should be very concise", no instruction, or "very
  comprehensive" (3 of the paper's 5).
- **Decoding:** greedy (always the most likely next word), with each model's default repetition
  penalty, as in the paper's code.
- **Questions:** 150. A 20-question pre-pilot projected that no allowed size (50, 100, 150)
  reached the target interval width of ±0.05 for φ-CCT, so the largest, 150, was used
  (`prepilot_sample_size.csv`).
- **Per question:** the original, 11 versions with an inserted word, and one N1 rerun.
- **Tests:**
  - **F1, verbosity:** do CT and FPR change between "very concise" and "very comprehensive"?
  - **F2, scale:** do the metrics differ between model sizes?
  - **F3:** how closely does φ-CCT follow CCT across the 9 model × length cells? (Paper:
    R² = 0.92.)
  - **Paper comparison:** our numbers next to values read off the paper's ComVE figures.
- **Run:** CPU, float32 (the paper used float16 on GPUs).

## Arm C: chain of thought (Qwen3-0.6B)

- **Model:** Qwen3-0.6B. It can answer with **thinking on** (it writes a reasoning trace before
  answering) or **thinking off**.
- **Questions:** 40, with the same 11 inserted words per question, plus N1 and N2 controls.
- **Run:** Kaggle, 2 T4 GPUs, float32.

### C1: explanations, thinking off

- 20 conditions:
  - **Order:** predict then explain, or explain then predict.
  - **Examples:** with human explanations, or answers only.
  - **Length:** the paper's 5 instructions (very concise, concise, none, comprehensive, very
    comprehensive).
- This is the paper's setup for instruction-tuned models, so it is the explanation baseline for
  this model.
- Greedy decoding.
- **Measured:** the arm R metrics, per condition, plus F-AUROC per order × example type.

### C2: thinking traces

- **Prompt:** thinking on, answer only. No explanation is requested; the **thinking trace is
  treated as the explanation**.
- **Budgets:** each trace is generated once, up to 4,096 tokens. For a budget of 128, 256, 512 or
  1,024 tokens, a longer trace is cut there and closed with Qwen's own stop phrase ("Considering
  the limited time by the user, I have to give the solution based on the thinking directly
  now."). A fresh answer is then generated. "Uncapped" uses the full trace.
- **Sampling:** 3 samples per prompt (temperature 0.6, top-p 0.95, top-k 20, Qwen's recommended
  thinking settings). The answer is the most common of the 3; ties go to the more confident one.
  About 2,900 traces in total.
- **Measured, per budget:**
  - the arm R metrics;
  - noise floors (N1, N2, excess flip rate);
  - share of answers that had to be forced by the cut;
  - average trace length;
  - all three mention detectors.
- **Dummy explainers,** as a reference: one that always mentions the word, and one that mentions
  it at random at the same overall rate.
- **Copy rates:** how much of each trace repeats words and phrases from the question.
- **Failure sensitivity:** metrics recomputed counting unparseable answers as all flips, then as
  no flips.

### C3: trace vs explanation at the same accuracy

- **C3A:** thinking off, explain then predict, answers-only examples, 5 length instructions.
  Sampled exactly as C2 (3 samples, same settings), so the main difference from C2 is a written
  explanation instead of a thinking trace.
- **C3R:** thinking off, answer only, no explanation. Reference accuracy and flip rate.
- **Curve test:**
  - Fit a line through the 5 explanation points (φ-CCT against accuracy).
  - Measure how far the 5 thinking points (the C2 budgets) sit above or below it.
  - Only thinking points inside the explanations' accuracy range count, and at least 3 are
    needed.
- **Regression:** whether "mentioned" depends on "flipped" differently for traces than for
  explanations. This compares C3A with no length instruction against uncapped C2, and controls
  for whether the original answer was right.

## Differences from the paper

- One dataset (ComVE), 150 or 40 questions; the paper used all 999 ComVE questions and about
  20 inserted words each.
- The judge is Qwen2.5-3B-Instruct, not 72B.
- The word list is sorted before random sampling. The paper's sampling order can't be
  reproduced, so the inserted words are new ones, not the paper's.
- float32 instead of float16.
- Arm R uses 3 of the 5 length instructions.
- Sampled runs use 3 samples per prompt; the main study plans 5.
- The thinking cap is 4,096 tokens; the main study plans 8,192.
- **Answer reading:** explain-then-predict answers in arm C that the paper's pattern can't read
  are retried with a looser pattern. That pattern allows a multi-line explanation or a full stop
  after the answer; answers the paper's pattern reads are unchanged.

## What happened during the runs

- **Kaggle session ended mid-run:** a new session resumed from the results already pushed; no
  work was lost.
- **GPU out of memory:** this hit "C3A very concise" partway through. Each condition now runs in
  its own process, and the condition was rerun.
- **Unreadable explain-then-predict answers:**
  - With answers-only examples, the paper's answer pattern failed on 36–79% of C3A answers.
  - The looser pattern was added, and all explain-then-predict runs in arm C were rerun with it.
  - The earlier results are kept in `raw/` (the folders without `-tolerant` in their name) but
    are not used in the tables.
  - Still unreadable afterwards: up to 43% in C1 explain-then-predict with answers-only examples,
    mostly answers that repeat "EXPLANATION:" until the 500-token limit.
- **Answer-position bias in arm R:** Qwen2.5-0.5B picks option 1 21% of the time and 1.5B 80%,
  against 45% in the correct answers (columns `base_pred_label1_rate`, `gold_label1_rate` in
  `R_replication/cells.csv`).
- **Curve test:** it has no point estimate (blank `delta`), because only one thinking budget
  (128) falls inside the explanations' accuracy range. Its interval comes from the resamples
  where it is defined (`undefined_fraction` says how many were not).
