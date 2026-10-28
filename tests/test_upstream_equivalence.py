"""Our loaders and prompts must match the upstream corr_faith code exactly."""

import itertools

import pandas as pd
import pytest

import upstream
from cotfaith import datasets, prompts

pytestmark = [pytest.mark.data, pytest.mark.upstream]

N_EXAMPLES = 40  # Per dataset and prompt setting; prompt construction is cheap.

_LENGTH_ENUM = {
    "empty": "EMPTY",
    "very_concise": "VERY_CONCISE",
    "concise": "CONCISE",
    "comprehensive": "COMPREHENSIVE",
    "very_comprehensive": "VERY_COMPREHENSIVE",
}
_TEMPLATE_NAME = {
    prompts.Template.PREDICT_THEN_EXPLAIN: "predict_then_explain",
    prompts.Template.EXPLAIN_THEN_PREDICT: "explain_then_predict",
    prompts.Template.PREDICT_ONLY: "predict_only",
}


def upstream_dataset(name: str):
  return upstream.module("experiments.dataset_specific").dataset_from_string(name)


def upstream_prompt(ds, train_df, row, data_idx, template, include_expl, length):
  """Mirrors the prompt assembly in upstream evaluate_faithfulness.main."""
  cd = upstream.module("experiments.dataset_specific.classification_datasets")
  ci = upstream.module("experiments.counterfactual_interventions")
  prompt_template = _TEMPLATE_NAME[template]
  fewshot_template = prompt_template if include_expl else "predict_only"
  base = cd.FEWSHOT_EXAMPLE_SEP.join([
      ds.fewshot_prompt_prefix(include_explanation=fewshot_template != "predict_only"),
      ds.get_fewshot_prompt_examples(
          rng=ci.get_rng(data_idx, ds), train_df=train_df, n_shots=10,
          prompt_template=fewshot_template,
      ),
  ])
  instructions = (
      cd.ExplanationLengthInstructions[_LENGTH_ENUM[length]].value
      + cd.ExplanationFaithfulnessInstructions.EMPTY.value
  )
  end = ds.get_prompt_end_for_new_example(
      row=row, prompt_template=prompt_template, is_instruction_tuned=True,
      include_explanations_in_fewshot=include_expl, explanation_instructions=instructions,
  )
  return cd.FEWSHOT_EXAMPLE_SEP.join([base, end])


@pytest.mark.parametrize("name", datasets.DATASET_NAMES)
def test_splits_match_upstream(name, splits):
  ours = splits(name)
  theirs = upstream_dataset(name).load_data_splits(base_data_dir=str(upstream.DATA_DIR))
  for split in ("train", "dev", "test"):
    pd.testing.assert_frame_equal(ours[split], theirs[split])


@pytest.mark.parametrize("name", datasets.DATASET_NAMES)
def test_spec_matches_upstream(name):
  spec, ds = datasets.get_spec(name), upstream_dataset(name)
  assert spec.unique_id == ds.dataset_unique_id
  assert spec.class_labels == tuple(ds.class_labels)
  assert spec.keys_to_intervene_on == tuple(ds.keys_to_intervene_on)
  assert spec.instance_template == ds.problem_instance_template
  assert spec.class_template == ds.problem_class_template
  assert spec.explanation_template == ds.problem_explanation_template
  assert spec.start_of_next_example == ds.start_of_next_example_str
  for include in (True, False):
    assert spec.describe_example(include) == ds.describe_example(include)


SETTINGS = [
    (t, d, length)
    for t, d in itertools.product(prompts.Template, prompts.Demos)
    if not (t is prompts.Template.PREDICT_ONLY and d is prompts.Demos.WITH_EXPLANATIONS)
    for length in (["empty"] if t is prompts.Template.PREDICT_ONLY else prompts.LENGTH_INSTRUCTIONS)
]


@pytest.mark.parametrize("name", datasets.DATASET_NAMES)
@pytest.mark.parametrize("template,demos,length", SETTINGS, ids=str)
def test_prompts_match_upstream(name, template, demos, length, splits):
  spec, ds = datasets.get_spec(name), upstream_dataset(name)
  data = splits(name)
  # Same evaluation order as upstream: a fixed shuffle of the test set.
  test_df = data["test"].sample(frac=1, random_state=0).iloc[:N_EXAMPLES]
  include_expl = prompts.include_explanations_in_fewshot(template, demos)
  for data_idx, row in test_df.iterrows():
    ours = prompts.build_prompt(spec, data["train"], data_idx, row, template, demos, length)
    theirs = upstream_prompt(ds, data["train"], row, data_idx, template, include_expl, length)
    assert ours == theirs
