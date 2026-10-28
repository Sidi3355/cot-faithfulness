# Deviations

Every departure from the paper (Siegel et al., arXiv 2503.13445), from its code
(`google-deepmind/corr_faith`), or from the original project brief. Planned deviations are
listed now; new ones are appended as they occur, with date and affected runs.

## From the paper / code

| ID | Paper / code | This project | Reason | Affects |
|---|---|---|---|---|
| D1 | HF `transformers` `generate`, batch size 1 | vLLM | Throughput; Phase 1b measures the combined effect of D1, D2 and D5 | All phases |
| D2 | fp16 (bf16 only for Gemma-2-27B) | bf16 | Qwen3 is released in bf16; avoids fp16 overflow | All phases |
| D3 | Naturalness judge Qwen2.5-72B-Instruct | Qwen2.5-32B-Instruct-AWQ (4-bit) | 72B is a tight fit on one L40S; decided by the project lead | Intervention set |
| D4 | Inserted word drawn from `tuple(frozenset - set)`, so the order depends on the per-process string hash seed (`PYTHONHASHSEED` not set in the Dockerfile) | Word list sorted before sampling | Makes interventions deterministic. Side effect: the paper's exact words cannot be regenerated, so our set differs from the paper's even before D3 | Intervention set |
| D5 | Label probabilities from the full softmax at the label position | Sum over the top-20 logprobs returned by vLLM | vLLM API; mass outside the top 20 is treated as 0 and flagged if label mass < 0.8 | CCT / TVD |
| D6 | No Qwen3 handling (the chat template would silently enable thinking; the parser would read through the trace; 500-token cap) | `enable_thinking` set explicitly; answers parsed after `</think>`; thinking cap 8,192 | Qwen3 support | Phases 1–3 (Qwen3) |
| D7 | No rerun or null-edit control | N1 and N2 on every example and intervention | Noise floor for flips (extension) | All phases |
| D8 | Greedy decoding throughout | Greedy in Phase 1; K = 5 sampling in Phases 2–3 | Qwen3 thinking mode should not be decoded greedily; see PLAN.md §4.3 (extension) | Phases 2–3 |
| D9 | CI method for metrics: to be confirmed from the paper text | 10,000-resample bootstrap clustered by example; paired differences | Rigour | All reported CIs |

## From the original brief

| ID | Brief | This project | Reason |
|---|---|---|---|
| B1 | 20,000 interventions per dataset | ≥10,000 per dataset with the per-example top 5% (≈19.7k / 11.0k / 11.0k) | Matches what the code actually runs |
| B2 | Few-shot examples sampled fresh per query | Sampled once per test example, shared with its interventions | Matches the code |
| B3 | Phase 3 includes thinking on + explanation, and trace-E vs explanation-E agreement | Dropped | Decided by the project lead |
| B4 | 100 bootstrap resamples | 10,000 | 100 gives unstable 95% bounds; the bootstrap runs on saved results, so the extra resamples are free |
| B5 | Greedy decoding everywhere | See D8 | See D8 |
| B6 | Phase 1 has two orders × five lengths | Adds both demo variants (with explanations / labels only), as in the paper's sweep | Matches the paper |
| B7 | Qwen3 models only | Adds Qwen2.5-1.5B-Instruct and 7B-Instruct as replication anchors, plus the Phase 1b equivalence check | The paper has no Qwen3 results to compare against |
| B8 | Dummy check: "φ-CCT gives the copier near 0" | φ is undefined for the copier (constant E) and is reported as such | Correctness |
