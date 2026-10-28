import pathlib

import pydantic
import pytest

from cotfaith import config, models
from cotfaith.prompts import Demos, Template

EXAMPLES = sorted((pathlib.Path(__file__).parents[1] / "configs").rglob("*.yaml"))

BASE = dict(
    experiment_id="x",
    phase="phase1",
    model="Qwen/Qwen3-4B",
    model_revision=models.get("Qwen/Qwen3-4B").revision,
    dataset="esnli",
    template="pe",
    demos="with_explanations",
    decoding={"mode": "greedy"},
    interventions_path="data/interventions/natural_esnli.parquet",
    interventions_sha256="b" * 64,
)
THINKING = dict(
    BASE, phase="phase2", template="po", demos="labels_only", thinking=True,
    budgets=[128, 256, 512, 1024], decoding=config.SAMPLED.model_dump(), max_new_tokens=8256,
)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_configs_validate(path):
  config.load(path)


def test_valid_configs():
  config.RunConfig(**BASE)
  config.RunConfig(**THINKING)
  anchor = models.get("Qwen/Qwen2.5-7B-Instruct")
  config.RunConfig(**dict(BASE, model=anchor.name, model_revision=anchor.revision))
  config.RunConfig(**dict(  # Phase 3 cell R.
      BASE, phase="phase3", template="po", demos="labels_only", decoding=config.SAMPLED.model_dump()
  ))


@pytest.mark.parametrize("changes", [
    dict(model_revision="main"),  # Unpinned outside smoke runs.
    dict(model_revision="a" * 40),  # Not the pinned revision.
    dict(model="Qwen/Qwen3-4B-Instruct-2507"),  # Not in the study.
    dict(model="Qwen/Qwen2.5-32B-Instruct-AWQ", model_revision=models.get("Qwen/Qwen2.5-32B-Instruct-AWQ").revision),
    dict(interventions_sha256="unfrozen"),
    dict(template="po", demos="with_explanations"),
    dict(template="po", demos="labels_only", length="concise"),
    dict(thinking=True, budgets=[128]),  # Thinking in phase 1.
    dict(decoding={"mode": "greedy", "temperature": 0.5}),
    dict(unknown_field=1),
])
def test_invalid_phase1_configs(changes):
  with pytest.raises(pydantic.ValidationError):
    config.RunConfig(**dict(BASE, **changes))


@pytest.mark.parametrize("changes", [
    dict(budgets=[]),
    dict(budgets=[128, 9000]),
    dict(template="ep"),  # Thinking with an explanation is not a condition.
    dict(model="Qwen/Qwen2.5-7B-Instruct", model_revision=models.get("Qwen/Qwen2.5-7B-Instruct").revision),
    dict(decoding={"mode": "sample", "temperature": 1.0, "n": 5}),  # Not the shared settings.
    dict(max_new_tokens=4096),
])
def test_invalid_thinking_configs(changes):
  with pytest.raises(pydantic.ValidationError):
    config.RunConfig(**dict(THINKING, **changes))


def test_smoke_runs_may_be_unpinned():
  config.RunConfig(**dict(BASE, model_revision="main", interventions_sha256="unfrozen", smoke=True))


def test_condition_id_and_results_dir():
  cfg = config.RunConfig(**THINKING)
  assert cfg.template is Template.PREDICT_ONLY and cfg.demos is Demos.LABELS_ONLY
  assert cfg.condition_id == "po-labels_only-empty-think-sample5"
  assert cfg.results_dir("results") == pathlib.Path(
      "results/phase2/Qwen__Qwen3-4B/esnli/po-labels_only-empty-think-sample5"
  )


def test_dump_round_trip(tmp_path):
  cfg = config.RunConfig(**THINKING)
  path = tmp_path / "c.yaml"
  path.write_text(config.dump(cfg))
  assert config.load(path) == cfg
