# cot-faithfulness

Counterfactual faithfulness of Qwen3 chain-of-thought traces, built on Siegel et al. (arXiv 2503.13445).

- [PLAN.md](PLAN.md): experimental plan
- [pilot/](pilot/README.md): smaller-scale pilot that runs before the EC2 instance exists
- [pilot/results/](pilot/results/README.md): pilot tables, figures and raw outputs, with [notes](pilot/results/NOTES.md) on each run

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

```bash
.venv/bin/python scripts/fetch_tokenizers.py
```

The last step downloads tokenizer and config files (no weights) at the pinned revisions into
`data/hf`. It needs `huggingface.co` and Hugging Face's download hosts (`*.hf.co`) to be reachable.

On the GPU instance, add `--extra gpu` to `uv sync` to install vLLM.

## Tests

```bash
.venv/bin/pytest -q
```

Tests marked `data`, `nlp`, `upstream` or `hf` are skipped if the datasets, the spaCy model and
WordNet, the submodule, or the tokenizers are missing.

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
| `src/cotfaith/models.py`, `tokens.py` | Pinned model revisions; chat templates, think tokens, label tokens |
| `src/cotfaith/config.py`, `configs/` | One validated YAML config per run |
| `third_party/corr_faith` | Upstream code at commit c6e862b (Apache-2.0) |
