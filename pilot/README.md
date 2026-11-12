# Pilot

**The pilot that is being run is option A, described in [PILOT_PLAN.pdf](PILOT_PLAN.pdf).** It has
two arms: a small-scale replication on Qwen2.5 0.5B, 1.5B and 3B, and the chain-of-thought pilot on
Qwen3-0.6B. Results (tables, statistics and figures only) are in [`results/`](results/README.md);
[`results/NOTES.md`](results/NOTES.md) explains each run and test.

How it runs:

```bash
.venv/bin/python pilot/scripts/judge.py --n-examples 200
```

```bash
.venv/bin/python pilot/scripts/make_configs.py --r-examples 20
```

```bash
.venv/bin/python pilot/scripts/run_pilot.py pilot/configs/R_*_empty.yaml
```

```bash
.venv/bin/python pilot/scripts/analyze.py prepilot
```

The pre-pilot fixes N. Then regenerate the configs with `--r-examples N`, run every config in
`pilot/configs/`, and analyse:

```bash
.venv/bin/python pilot/scripts/analyze.py r
```

```bash
.venv/bin/python pilot/scripts/analyze.py c
```

The rest of this page is the background comparison of options that led to that choice.

---

A smaller version of the experiment in [PLAN.md](../PLAN.md), runnable before the EC2 instance
exists. The **design stays the same**: prompts, interventions, controls, decoding, metrics,
statistics and figures. Only the **scale** shrinks: models, datasets, examples and samples.

## What stays identical

- The paper's prompts (verified byte-identical) and the intervention procedure.
- The controls on every example and intervention:
  - N1, rerunning the base prompt;
  - N2, a whitespace-only twin of each intervention.
- Decoding: greedy for the explanation baseline (Phase 1), and sampling with the same settings
  in both thinking modes for Phases 2–3.
- Thinking budgets of 128, 256, 512 and 1024 tokens, plus uncapped.
- Every metric: CT, TPR/FPR, φ-CCT, CCT, F-AUROC, and the three mention detectors.
- The example-clustered bootstrap, paired differences, the curve test and the regression.
- Every figure: TPR/FPR with convex hulls, metrics against size, budgets, and trace vs
  explanation against accuracy.
- The config files and the results layout.

## Models

| Model | Why | Used in |
|---|---|---|
| Qwen3-0.6B | Smallest study model; the pilot's workhorse | All phases |
| Qwen3-1.7B | Second size, the minimum for a size trend | Phase 1 on CPU; all phases on GPU |
| Qwen2.5-1.5B-Instruct | Evaluated in the paper, so it checks our implementation | Phase 1 |
| Judge: Qwen2.5-3B-Instruct (CPU) or Qwen2.5-14B/32B-AWQ (GPU) | The 32B judge doesn't fit on a CPU with 15 GB RAM | Choosing interventions |

## Dataset: ComVE only

- It has the shortest prompts: about 550–720 tokens, against 600–870 for e-SNLI and 800–1,660
  for ECQA.
- It has the smallest test set (999 examples) and 11 interventions per example.
- Its labels are binary, so a majority vote over 3 or 5 samples can never tie.

## Option A: this cloud machine (CPU), end-to-end pilot

Measured on this machine (4 CPU cores with AMX-BF16, 15 GB RAM, no GPU; numbers in
[`benchmarks/cpu_2026-11-02.json`](benchmarks/cpu_2026-11-02.json)):

| | Qwen3-0.6B | Qwen3-1.7B |
|---|---|---|
| Process a 739-token prompt | 1.6 s | 2.6 s |
| Same, reusing the 597-token demo prefix | 0.4 s | 0.6 s |
| Generation speed, batch of 16 | ≈ 70 tokens/s | ≈ 30 tokens/s |
| Real thinking traces | 624–1,530 tokens, 1–2 min each | 439 tokens, 1 min |

At these speeds a full-coverage thinking run is impossible here. ComVE alone would need about
120M generated tokens, which is roughly 20 days for 0.6B. So option A uses a **fixed subset**:

- **Examples:** the first 40 ComVE test examples in the paper's fixed shuffle order, with
  **every** intervention, N1 and N2 for each. That's about 440 interventions and 960 prompts per
  condition.
- **Samples:** K = 3 instead of 5.
- **Judge:** Qwen2.5-3B-Instruct, scoring those 40 examples' 8,800 candidates.

| Step | Generations | Estimated hours |
|---|---|---|
| Judge | 8,800 one-token scores | ≈ 1 |
| Phase 1, Qwen3-0.6B: 2 orders × 5 lengths × 2 demo variants | 19,200 | ≈ 5 |
| Phase 2, Qwen3-0.6B: thinking, all budgets from one generation | 2,880 traces | ≈ 13 |
| Phase 3, Qwen3-0.6B: cell A (5 lengths) + cell R | 17,280 | ≈ 4 |
| **Qwen3-0.6B, all phases** | | **≈ 23** |
| Optional: Phase 1 for Qwen3-1.7B | 19,200 | ≈ 12 |
| Optional: Phase 1 for the Qwen2.5-1.5B anchor | 19,200 | ≈ 11 |

**What it's for:**
- debugging the runner on real outputs;
- producing every table and figure once;
- measuring the numbers the full run depends on: trace lengths, flip rates, mention rates, the
  N1/N2 noise floors, and parse-failure and unterminated rates.

Those numbers feed a power analysis and a firm compute estimate for EC2.

**What it can't do:** support any claim. With about 440 interventions, the confidence intervals
are wide. It also breaks the every-example rule on purpose, so its results are never reported.

**Practical note:** this machine is reclaimed after a period of inactivity. Runs are split into
resumable chunks, and the (small) results are pushed to the branch as they finish.

## Option B: free cloud GPU (Kaggle or Colab), full-coverage pilot

- **Scope:** the full ComVE test set (999 examples, about 10,978 interventions) and the full
  design (K = 5, every budget, N1 and N2) for Qwen3-0.6B and Qwen3-1.7B.
- **This respects the every-example rule.** Its results are real, reportable findings for those
  two models on ComVE.
- **Hardware:** Kaggle gives about 30 free GPU-hours a week on a T4 (16 GB), or two T4s.
  Colab's free tier gives a T4 with less predictable limits.
- **Rough estimate (unmeasured):** about 120M generated tokens per model for Phase 2. That's
  roughly 10–15 GPU-hours for 0.6B and 20–30 for 1.7B, so 2–3 weeks of Kaggle quota.
- **Compromises:**
  - fp16 instead of bf16, since T4s lack bf16. That's what the paper's code used.
  - A smaller judge: Qwen2.5-14B-AWQ on one T4, or the 32B-AWQ split across two T4s if vLLM
    supports that there.
  - Sessions are capped at about 12 hours, so the runner must resume.
  - vLLM support on T4s needs checking in the first session; the fallback is Hugging Face on the
    GPU, which is slower.

## Option C: your own PC

This depends on the hardware:
- **NVIDIA GPU with 8 GB or more** (e.g. RTX 3060 12 GB or newer), on Linux or WSL2: like
  option B. RTX 30-series and newer support bf16.
- **Apple Silicon Mac:** vLLM doesn't run there, so it would need MLX or llama.cpp. That's a
  different engine, and the most numerically different option.
- **CPU only:** like option A.

## Compromises compared with the full plan

| | Full plan (EC2) | A: CPU here | B: free GPU |
|---|---|---|---|
| Models | 5 Qwen3 + 2 anchors | Qwen3-0.6B (+ 1.7B and anchor, Phase 1 only) | Qwen3-0.6B, 1.7B (+ anchor) |
| Datasets | 3 | ComVE | ComVE |
| Examples | All | 40, fixed | All 999 |
| Samples per input (K) | 5 | 3 | 5 |
| Naturalness judge | Qwen2.5-32B-AWQ | Qwen2.5-3B | Qwen2.5-14B-AWQ or 32B-AWQ |
| Engine, number format | vLLM, bf16 | Hugging Face on CPU, bf16 | vLLM, fp16 |
| Size trend | 5 sizes | none | 2 points |
| Reportable | Yes | No (pilot only) | Yes, for these cells |

Every option shares one limit: one dataset and at most two sizes. Neither can say anything about
how results scale with model size or carry across datasets.

## Contents

- `benchmarks/`: measured throughput on this machine.
- `scripts/bench_cpu.py`: the benchmark. It needs `torch`, `transformers` and `accelerate`, plus
  the weights in `data/hf`.
