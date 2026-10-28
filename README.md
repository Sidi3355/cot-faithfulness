# cot-faithfulness

Counterfactual faithfulness of Qwen3 chain-of-thought traces, built on Siegel et al. (arXiv 2503.13445).

- [PLAN.md](PLAN.md): experimental plan

## Setup (CPU parts)

Requires [uv](https://docs.astral.sh/uv/) and git. The upstream paper code is a pinned submodule.

```bash
git submodule update --init
```

```bash
uv sync --extra interventions
```

```bash
PYTHON=.venv/bin/python scripts/fetch_data.sh
```

On the GPU instance, add `--extra gpu` to `uv sync` to install vLLM.

## Tests

```bash
.venv/bin/pytest -q
```

Tests marked `data`, `nlp` or `upstream` are skipped if the datasets, the spaCy model and WordNet, or
the submodule are missing.

## Intervention candidates

```bash
.venv/bin/python scripts/make_candidates.py
```

This writes `data/interventions/candidates_<dataset>.parquet` and `candidates_meta.json` (hashes and
package versions). The naturalness judge then scores these on the GPU.

## Layout

| Path | Contents |
|---|---|
| `src/cotfaith/datasets.py`, `prompts.py` | Loaders and prompts, verified identical to the upstream code |
| `src/cotfaith/interventions.py` | Candidate generation, naturalness prompt and selection, N2 twins |
| `src/cotfaith/parse.py`, `budget.py` | Answer and trace parsing, thinking-budget cuts |
| `src/cotfaith/detect.py` | Mention detectors (E_paper, E_word, E_novel) and copy rates |
| `src/cotfaith/flips.py`, `metrics.py` | Flip indicators, CT, φ-CCT, CCT, F-AUROC |
| `src/cotfaith/bootstrap.py`, `analysis.py` | Clustered bootstrap, curve test, regression |
| `src/cotfaith/config.py`, `configs/` | One validated YAML config per run |
| `third_party/corr_faith` | Upstream code at commit c6e862b (Apache-2.0) |
