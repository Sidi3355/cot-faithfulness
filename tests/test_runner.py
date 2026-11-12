"""End-to-end runner checks on real (small) models. Slow: marked `weights`."""

import json

import pandas as pd
import pytest
import torch

import upstream
from cotfaith import config, datasets, interventions, models, prompts, run

pytestmark = [pytest.mark.weights, pytest.mark.data]

R_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
C_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def smoke_interventions(tmp_path_factory, splits):
  """First 4 candidates of the first 2 examples (paper order), as a stand-in file."""
  test = splits("comve")["test"].sample(frac=1, random_state=0).iloc[:2]
  cands = pd.read_parquet(upstream.DATA_DIR / "interventions" / "candidates_comve.parquet")
  sel = cands[cands["data_idx"].isin(test.index) & (cands["intervention_idx"] < 4)]
  path = tmp_path_factory.mktemp("ints") / "smoke.parquet"
  sel.to_parquet(path)
  return str(path), list(test.index)


def _cfg(path, **kw):
  base = dict(
      experiment_id="test", phase="phase1", model=R_MODEL, model_revision=models.get(R_MODEL).revision,
      dataset="comve", template="pe", demos="with_explanations",
      decoding={"mode": "greedy", "repetition_penalty": models.get(R_MODEL).default_repetition_penalty},
      interventions_path=path, interventions_sha256="unfrozen", engine="hf_cpu", pilot=True, n_examples=1,
      smoke=True,
  )
  base.update(kw)
  return config.RunConfig(**base)


@pytest.mark.upstream
def test_greedy_rows_match_upstream_predictions(smoke_interventions, splits, tmp_path):
  path, ids = smoke_interventions
  cfg = _cfg(path)
  runner = run.Runner(cfg, str(upstream.DATA_DIR), str(tmp_path), dtype=torch.float32, log=lambda m: None)
  runner.run()
  rows = run.load_rows(cfg.results_dir(tmp_path))
  assert set(rows.row_type) == {"base", "n1", "real", "n2"}
  assert len(rows) == 2 + 2 * 4

  data = splits("comve")
  ds = upstream.module("experiments.dataset_specific").dataset_from_string("comve")
  pipe = upstream.pipeline_for(runner.engine.model, runner.engine.tokenizer)
  ints = pd.read_parquet(path)
  for _, r in rows[rows.row_type.isin(["base", "real"])].iterrows():
    item = data["test"].loc[r.example_id] if r.row_type == "base" else ints[
        (ints.data_idx == r.example_id) & (ints.intervention_idx == r.intervention_idx)].iloc[0]
    text = prompts.build_prompt(datasets.COMVE, data["train"], r.example_id, item,
                                prompts.Template.PREDICT_THEN_EXPLAIN, prompts.Demos.WITH_EXPLANATIONS)
    cd = upstream.module("experiments.dataset_specific.classification_datasets")
    theirs = ds.predict_huggingface(pipe, text, prompt_template=cd.PromptTemplate.PREDICT_THEN_EXPLAIN,
                                    is_instruction_tuned=True)
    assert r.label == theirs.pred_label
    assert r.explanation == theirs.explanation
    ours = json.loads(r.label_probs)
    for k, label in enumerate(datasets.COMVE.class_labels):
      assert ours[k] == pytest.approx(theirs.token_probs[label], abs=1e-4)


def test_n1_matches_base_in_fp32(smoke_interventions, tmp_path):
  path, _ = smoke_interventions
  cfg = _cfg(path)
  run.Runner(cfg, str(upstream.DATA_DIR), str(tmp_path), dtype=torch.float32, log=lambda m: None).run()
  rows = run.load_rows(cfg.results_dir(tmp_path)).set_index("row_type")
  assert rows.loc["base", "answer_text"] == rows.loc["n1", "answer_text"]


def test_thinking_rows_and_resume(smoke_interventions, tmp_path):
  path, ids = smoke_interventions
  cfg = _cfg(
      path, phase="phase2", model=C_MODEL, model_revision=models.get(C_MODEL).revision, template="po",
      demos="labels_only", thinking=True, budgets=[16, 32], thinking_cap=48, max_new_tokens=112,
      decoding=dict(config.SAMPLED.model_dump(), n=3), controls={"n1": True, "n2": False},
  )
  logs = []
  runner = run.Runner(cfg, str(upstream.DATA_DIR), str(tmp_path), log=logs.append)
  runner.run()
  rows = run.load_rows(cfg.results_dir(tmp_path))
  # (base, n1, 4 interventions) x 3 samples x (2 budgets + uncapped).
  assert len(rows) == 6 * 3 * 3
  assert set(rows.budget) == {"16", "32", "uncapped"}
  b16 = rows[rows.budget == "16"]
  assert (b16.trace_n_tokens <= 16).all()
  assert b16.forced.all() or (b16.trace_n_tokens < 16).any()
  # Cut traces are prefixes of the uncapped trace for the same sample.
  merged = b16.merge(rows[rows.budget == "uncapped"], on=["item", "sample_idx"], suffixes=("", "_u"))
  assert all(u.startswith(c) for c, u in zip(merged.trace, merged.trace_u))
  # Resume: nothing left to do.
  logs.clear()
  run.Runner(cfg, str(upstream.DATA_DIR), str(tmp_path), log=logs.append).run()
  assert logs == []


def test_interrupted_thinking_example_resumes_identically(smoke_interventions, tmp_path, monkeypatch):
  path, _ = smoke_interventions
  cfg = _cfg(
      path, phase="phase2", model=C_MODEL, model_revision=models.get(C_MODEL).revision, template="po",
      demos="labels_only", thinking=True, budgets=[16], thinking_cap=32, max_new_tokens=96,
      decoding=dict(config.SAMPLED.model_dump(), n=3), controls={"n1": False, "n2": False},
  )
  full = run.Runner(cfg, str(upstream.DATA_DIR), str(tmp_path / "a"), dtype=torch.float32,
                    batch_size=4, log=lambda m: None)
  full.run()
  expected = run.load_rows(cfg.results_dir(tmp_path / "a"))

  runner = run.Runner(cfg, str(upstream.DATA_DIR), str(tmp_path / "b"), dtype=torch.float32,
                      batch_size=4, log=lambda m: None)
  real_generate, calls = runner.engine.generate, []

  def flaky(*args, **kwargs):  # Dies on the second call, after one checkpoint.
    calls.append(1)
    if len(calls) == 2:
      raise RuntimeError("simulated restart")
    return real_generate(*args, **kwargs)

  monkeypatch.setattr(runner.engine, "generate", flaky)
  with pytest.raises(RuntimeError):
    runner.run()
  assert list((cfg.results_dir(tmp_path / "b") / "rows").glob("*.partial.pkl"))
  monkeypatch.setattr(runner.engine, "generate", real_generate)
  runner.run()
  resumed = run.load_rows(cfg.results_dir(tmp_path / "b"))
  assert not list((cfg.results_dir(tmp_path / "b") / "rows").glob("*.partial.pkl"))
  cols = ["item", "sample_idx", "budget", "trace", "answer_text", "label"]
  pd.testing.assert_frame_equal(expected[cols], resumed[cols])
