import json
import os
import subprocess
import sys

import pandas as pd
import pytest

import upstream
from cotfaith import datasets, interventions

N_EXAMPLES = 30  # Per dataset; upstream with sorted iteration is slow.


def test_candidates_per_example():
  assert interventions.candidates_per_example("esnli") == 40
  assert interventions.candidates_per_example("ecqa") == 100
  assert interventions.candidates_per_example("comve") == 220
  # Upstream's hard-coded sizes and the true sizes give the same counts.
  assert interventions.candidates_per_example("esnli", 9824) == 40
  assert interventions.candidates_per_example("ecqa", 2184) == 100


def test_select_natural_keeps_top_fraction_per_example():
  scores = list(range(40)) + list(range(100))
  df = pd.DataFrame({
      "data_idx": [0] * 40 + [1] * 100,
      "intervention_naturalness_yes_prob": scores,
  })
  kept = interventions.select_natural(df)
  assert kept.groupby("data_idx").size().to_dict() == {0: 2, 1: 5}
  assert kept[kept.data_idx == 0].intervention_naturalness_yes_prob.tolist() == [38, 39]


def test_select_natural_keeps_ties():
  df = pd.DataFrame({"data_idx": [0] * 40, "intervention_naturalness_yes_prob": [1.0] * 40})
  assert len(interventions.select_natural(df)) == 40


def test_yes_no_probs():
  probs = interventions.yes_no_probs([
      ("Yes", 0.7), ("yes", 0.1), ("No", 0.1), (".YesNo", 0.05), ("The", 0.05),
  ])
  assert probs == pytest.approx(dict(yes=0.8, no=0.1, invalid=0.1))


def test_n2_sentence_adds_one_space():
  assert interventions.n2_sentence("A man plays.", 2) == "A  man plays."
  assert interventions.n2_sentence("Dogs bark.", 0) == " Dogs bark."


@pytest.mark.upstream
def test_naturalness_prompt_matches_upstream():
  theirs = upstream.module("experiments.intervention_naturalness")
  assert interventions.NATURALNESS_PROMPT_TEMPLATE == theirs.INTERVENTION_NATURALNESS_PROMPT_TEMPLATE


class _SortedFrozenset(frozenset):
  """frozenset that iterates in sorted order, including after subtraction."""

  def __sub__(self, other):
    return _SortedFrozenset(frozenset.__sub__(self, other))

  def __iter__(self):
    return iter(sorted(frozenset.__iter__(self)))


def _test_rows(splits, name, n=N_EXAMPLES):
  return splits(name)["test"].sample(frac=1, random_state=0).iloc[:n]


@pytest.mark.data
@pytest.mark.nlp
@pytest.mark.upstream
@pytest.mark.parametrize("name", datasets.DATASET_NAMES)
def test_matches_upstream_with_sorted_iteration(name, splits, monkeypatch):
  """Our only change from upstream is the order of the word pools."""
  ci = upstream.module("experiments.counterfactual_interventions")
  original_words = ci.classical_nlp.get_wordnet_words_by_part_of_speech

  def sorted_words(**kwargs):
    return {pos: _SortedFrozenset(words) for pos, words in original_words(**kwargs).items()}

  monkeypatch.setattr(ci.classical_nlp, "get_wordnet_words_by_part_of_speech", sorted_words)
  spec = datasets.get_spec(name)
  ds = upstream.module("experiments.dataset_specific").dataset_from_string(name)
  nlp = interventions.spacy_nlp()
  n = interventions.candidates_per_example(name)
  for data_idx, row in _test_rows(splits, name).iterrows():
    sentences = row[list(spec.keys_to_intervene_on)].to_dict()
    ours = interventions.produce_random_interventions(
        datasets.example_rng(data_idx, spec), nlp, sentences, n
    )
    theirs = ci.produce_random_interventions(ci.get_rng(data_idx, ds), nlp, sentences, n)
    assert len(ours) == len(theirs) == n
    for a, b in zip(ours, theirs):
      assert {k: a[k] for k in b} == b


_SUBPROCESS_SCRIPT = """
import json, sys
from cotfaith import datasets, interventions
out = {}
for name in datasets.DATASET_NAMES:
    spec = datasets.get_spec(name)
    test = datasets.load_splits(name, sys.argv[1])["test"].iloc[:10]
    df = interventions.generate_candidates(spec, test, interventions.candidates_per_example(name))
    out[name] = df[["data_idx", "inserted_word", "inserted_position", "modified_sentence_key"]].values.tolist()
print(json.dumps(out, default=str))
"""


@pytest.mark.data
@pytest.mark.nlp
def test_deterministic_across_hash_seeds():
  outputs = []
  for seed in ("1", "2"):
    env = dict(os.environ, PYTHONHASHSEED=seed)
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, str(upstream.DATA_DIR)],
        env=env, capture_output=True, text=True, check=True,
    )
    outputs.append(json.loads(result.stdout))
  assert outputs[0] == outputs[1]


@pytest.mark.data
@pytest.mark.nlp
def test_n2_twins_change_only_whitespace(splits):
  spec = datasets.get_spec("esnli")
  test = splits("esnli")["test"].iloc[:20]
  candidates = interventions.generate_candidates(spec, test, 40)
  twins = interventions.n2_twins(candidates, test)
  for _, twin in twins.iterrows():
    key = twin["modified_sentence_key"]
    original = test.loc[twin["data_idx"], key]
    assert twin[key] != original
    assert twin[key].split() == original.split()
    assert len(twin[key]) == len(original) + 1
