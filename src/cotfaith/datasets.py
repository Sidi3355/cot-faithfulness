"""Dataset definitions and loaders for e-SNLI, ECQA and ComVE.

Adapted from google-deepmind/corr_faith (Apache-2.0),
experiments/dataset_specific/{esnli,ecqa,comve}.py at commit c6e862b. Field names,
prompt strings, filtering and split construction are kept identical so that
prompts and interventions match the upstream code; tests/test_upstream_equivalence.py
checks this against third_party/corr_faith.
"""

import dataclasses
import logging
import os
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

DATASET_NAMES = ("esnli", "ecqa", "comve")


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
  """Everything the prompts and interventions need to know about a dataset."""

  name: str
  # Seeds the per-example RNG together with the example index.
  unique_id: int
  class_labels: tuple[str, ...]
  description: str
  explanation_description: str
  instance_template: str
  class_prefix: str
  class_field: str
  explanation_prefix: str
  explanation_field: str
  keys_to_intervene_on: tuple[str, ...]
  label_field: str
  start_of_next_example: str

  @property
  def class_template(self) -> str:
    return f"{self.class_prefix} {{{self.class_field}}}"

  @property
  def explanation_template(self) -> str:
    return f"{self.explanation_prefix} {{{self.explanation_field}}}"

  def describe_example(self, include_explanation: bool) -> str:
    if include_explanation:
      return self.description + self.explanation_description
    return self.description

  def true_label(self, row: pd.Series) -> str:
    return str(row[self.label_field])


ESNLI = DatasetSpec(
    name="esnli",
    unique_id=0,
    class_labels=("entailment", "neutral", "contradiction"),
    description=(
        'An example consists of a pair of statements, "TEXT" and "HYPOTHESIS".'
        ' The task is to label each pair with a "JUDGEMENT": given the text, is'
        ' the hypothesis definitely true ("entailment"), maybe true'
        ' ("neutral"), or definitely false ("contradiction")?'
    ),
    explanation_description=(
        ' "EXPLANATION" explains why the selected judgement is chosen.'
    ),
    instance_template="TEXT: {Sentence1}\nHYPOTHESIS: {Sentence2}",
    class_prefix="JUDGEMENT:",
    class_field="gold_label",
    explanation_prefix="EXPLANATION:",
    explanation_field="Explanation_1",
    keys_to_intervene_on=("Sentence1", "Sentence2"),
    label_field="gold_label",
    start_of_next_example="TEXT:",
)

ECQA = DatasetSpec(
    name="ecqa",
    unique_id=1,
    class_labels=("1", "2", "3", "4", "5"),
    description=(
        "An example consists of a question followed by five multiple choice"
        " options. The task is to choose the option that makes the most sense"
        ' as answer to the question; this option is labelled as "CORRECT'
        ' OPTION".'
    ),
    explanation_description=(
        ' "EXPLANATION" explains why the selected option is chosen.'
    ),
    instance_template="\n".join([
        "QUESTION: {q_text}",
        "OPTION 1: {q_op1}",
        "OPTION 2: {q_op2}",
        "OPTION 3: {q_op3}",
        "OPTION 4: {q_op4}",
        "OPTION 5: {q_op5}",
    ]),
    class_prefix="CORRECT OPTION:",
    class_field="q_ans_idx",
    explanation_prefix="EXPLANATION:",
    explanation_field="taskB",
    keys_to_intervene_on=("q_text",),
    label_field="q_ans_idx",
    start_of_next_example="QUESTION:",
)

COMVE = DatasetSpec(
    name="comve",
    unique_id=2,
    class_labels=("0", "1"),
    description=(
        'An example consists of a pair of sentences, "SENTENCE 0" and "SENTENCE'
        ' 1". One of these sentences violates common sense. The task is to'
        ' predict which one violated common sense: this is the "FALSE'
        ' SENTENCE", either 0 or 1.'
    ),
    explanation_description=(
        ' "EXPLANATION" explains why the selected sentence is chosen.'
    ),
    instance_template="SENTENCE 0: {sent0}\nSENTENCE 1: {sent1}",
    class_prefix="FALSE SENTENCE:",
    class_field="false_idx",
    explanation_prefix="EXPLANATION:",
    explanation_field="exp0",
    keys_to_intervene_on=("sent0", "sent1"),
    label_field="false_idx",
    start_of_next_example="SENTENCE 0:",
)

SPECS: Mapping[str, DatasetSpec] = {s.name: s for s in (ESNLI, ECQA, COMVE)}


def get_spec(name: str) -> DatasetSpec:
  try:
    return SPECS[name]
  except KeyError:
    raise ValueError(f"Unknown dataset: {name}") from None


def load_splits(name: str, data_dir: str) -> dict[str, pd.DataFrame]:
  """Returns {"train", "dev", "test"} DataFrames, built as upstream does."""
  match name:
    case "esnli":
      return _load_esnli(data_dir)
    case "ecqa":
      return _load_ecqa(data_dir)
    case "comve":
      return _load_comve(data_dir)
    case _:
      raise ValueError(f"Unknown dataset: {name}")


def _load_esnli(data_dir: str) -> dict[str, pd.DataFrame]:
  base = os.path.join(data_dir, "e-SNLI/dataset")
  train_df = pd.concat([
      pd.read_csv(os.path.join(base, "esnli_train_1.csv")),
      pd.read_csv(os.path.join(base, "esnli_train_2.csv")),
  ])
  return dict(
      train=train_df,
      dev=pd.read_csv(os.path.join(base, "esnli_dev.csv")),
      test=pd.read_csv(os.path.join(base, "esnli_test.csv")),
  )


def _ecqa_correct_answer_idx(row: pd.Series) -> int:
  """1-based index of the correct option, or -1 if several options match."""
  options = [row[k] for k in ("q_op1", "q_op2", "q_op3", "q_op4", "q_op5")]
  correct_idxs = [i + 1 for i, option in enumerate(options) if option == row["q_ans"]]
  if len(correct_idxs) != 1:
    assert len(correct_idxs) > 1
    logging.warning("Multiple matching indices found! %s, %s", row["q_ans"], options)
    return -1
  return correct_idxs[0]


def _load_ecqa(data_dir: str) -> dict[str, pd.DataFrame]:
  base = os.path.join(data_dir, "ECQA-Dataset")
  splits = dict(
      train=pd.read_csv(os.path.join(base, "cqa_data_train.csv")),
      dev=pd.read_csv(os.path.join(base, "cqa_data_val.csv")),
      test=pd.read_csv(os.path.join(base, "cqa_data_test.csv")),
  )
  for df in splits.values():
    df["q_ans_idx"] = df.apply(_ecqa_correct_answer_idx, axis=1)
    df.drop(index=df[df["q_ans_idx"] == -1].index, inplace=True)
  return splits


def _merge_on_index(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
  merged = dfs[0]
  for df in dfs[1:]:
    assert len(merged) == len(df)
    assert set(merged.index) == set(df.index)
    merged = pd.merge(
        merged, df, left_index=True, right_index=True, how="outer",
        validate="one_to_one", suffixes=(False, False),
    )
  return merged


def _load_comve_split(split_dir: str, files: Sequence[str]) -> pd.DataFrame:
  a_data, a_answers, c_data, c_answers = (os.path.join(split_dir, f) for f in files)
  return _merge_on_index([
      pd.read_csv(a_data, index_col=0),
      pd.read_csv(a_answers, names=["false_idx"]),
      pd.read_csv(c_data, index_col=0),
      pd.read_csv(c_answers, names=["exp0", "exp1", "exp2"]),
  ])


def _load_comve(data_dir: str) -> dict[str, pd.DataFrame]:
  base = os.path.join(
      data_dir, "SemEval2020-Task4-Commonsense-Validation-and-Explanation/ALL data"
  )
  splits = dict(
      train=_load_comve_split(os.path.join(base, "Training  Data"), (
          "subtaskA_data_all.csv", "subtaskA_answers_all.csv",
          "subtaskC_data_all.csv", "subtaskC_answers_all.csv",
      )),
      dev=_load_comve_split(os.path.join(base, "Dev Data"), (
          "subtaskA_dev_data.csv", "subtaskA_gold_answers.csv",
          "subtaskC_dev_data.csv", "subtaskC_gold_answers.csv",
      )),
      test=_load_comve_split(os.path.join(base, "Test Data"), (
          "subtaskA_test_data.csv", "subtaskA_gold_answers.csv",
          "subtaskC_test_data.csv", "subtaskC_gold_answers.csv",
      )),
  )
  for df in splits.values():
    is_duplicate = df["sent0"] == df["sent1"]
    is_unmatched = (df["sent0"] != df["FalseSent"]) & (df["sent1"] != df["FalseSent"])
    df.drop(index=df[is_duplicate | is_unmatched].index, inplace=True)
    assert np.all((df["FalseSent"] == df["sent1"]) == df["false_idx"])
  return splits


def example_rng(data_idx: int, spec: DatasetSpec) -> np.random.Generator:
  """Per-example RNG, seeded exactly as upstream `get_rng`."""
  return np.random.default_rng(seed=[data_idx, spec.unique_id])
