import dataclasses

import pandas as pd
import pytest

import upstream
from cotfaith import budget, datasets, parse
from cotfaith.prompts import Template

ESNLI = datasets.get_spec("esnli")

COMPLETIONS = [
    (Template.PREDICT_THEN_EXPLAIN, "JUDGEMENT: entailment\nEXPLANATION: Playing on stage is performing."),
    (Template.PREDICT_THEN_EXPLAIN, "Sure!\n**JUDGEMENT:** Neutral\n**EXPLANATION:** Not stated."),
    (Template.PREDICT_THEN_EXPLAIN, "JUDGEMENT: entailment\nEXPLANATION: First line.\nSecond line."),
    (Template.PREDICT_THEN_EXPLAIN, "JUDGEMENT: maybe\nEXPLANATION: Unknown label."),
    (Template.PREDICT_THEN_EXPLAIN, "I cannot answer that."),
    (Template.EXPLAIN_THEN_PREDICT, "EXPLANATION: A man performing.\nJUDGEMENT: entailment"),
    (Template.EXPLAIN_THEN_PREDICT, "EXPLANATION: Para one.\n\nPara two.\nJUDGEMENT: contradiction"),
    (Template.PREDICT_ONLY, "JUDGEMENT: contradiction"),
    (Template.PREDICT_ONLY, "\n\nJUDGEMENT: neutral\n"),
]

_UPSTREAM_TEMPLATE = {
    Template.PREDICT_THEN_EXPLAIN: "predict_then_explain",
    Template.EXPLAIN_THEN_PREDICT: "explain_then_predict",
    Template.PREDICT_ONLY: "predict_only",
}


@dataclasses.dataclass
class _Response:
  text: str
  n_prompt_tokens: int = 0
  n_completion_tokens: int = 0


class _FakeClient:
  def __init__(self, text):
    self.text = text

  def generate(self, prompt, max_new_tokens, do_sample):
    return _Response(self.text)


@pytest.mark.upstream
@pytest.mark.parametrize("template", list(Template))
def test_regex_matches_upstream(template):
  ds = upstream.module("experiments.dataset_specific").dataset_from_string("esnli")
  cd = upstream.module("experiments.dataset_specific.classification_datasets")
  theirs = ds._get_completion_regex(cd.PromptTemplate(_UPSTREAM_TEMPLATE[template]), True)
  assert parse.completion_regex(ESNLI, template).pattern == theirs.pattern


@pytest.mark.upstream
@pytest.mark.parametrize("template,text", COMPLETIONS)
def test_parse_matches_upstream(template, text):
  ds = upstream.module("experiments.dataset_specific").dataset_from_string("esnli")
  cd = upstream.module("experiments.dataset_specific.classification_datasets")
  theirs = ds.predict_api(
      model_client=_FakeClient(text), prompt="", prompt_template=cd.PromptTemplate(_UPSTREAM_TEMPLATE[template]),
  )
  ours = parse.parse_answer(ESNLI, template, text)
  their_label = None if theirs.pred_label is pd.NA else theirs.pred_label
  assert ours.label == their_label
  if ours.error != "no_match":
    their_expl = theirs.explanation if template is not Template.PREDICT_ONLY else None
    assert ours.explanation == their_expl


def test_explanation_is_first_line_only():
  ans = parse.parse_answer(ESNLI, Template.PREDICT_THEN_EXPLAIN, COMPLETIONS[2][1])
  assert ans.label == "entailment" and ans.explanation == " First line."


def test_split_thinking_and_parse_answer_only():
  text = (
      "<think>\nThe text says JUDGEMENT: contradiction would be wrong.\n</think>\n\n"
      "JUDGEMENT: entailment"
  )
  parsed = parse.parse_completion(ESNLI, Template.PREDICT_ONLY, text, thinking=True)
  assert parsed.label == "entailment"
  assert "JUDGEMENT: contradiction" in parsed.trace
  assert parsed.terminated


def test_unterminated_trace():
  parsed = parse.parse_completion(ESNLI, Template.PREDICT_ONLY, "<think>\nStill thinking", thinking=True)
  assert not parsed.terminated and parsed.label is None and parsed.error == "unterminated"


def test_thinking_off_parses_whole_text():
  parsed = parse.parse_completion(ESNLI, Template.PREDICT_ONLY, "JUDGEMENT: neutral", thinking=False)
  assert parsed.trace is None and parsed.label == "neutral"


START, END = 1, 2


def test_thinking_span():
  assert budget.thinking_span([START, 10, 11, 12, END, 20], START, END) == budget.ThinkingSpan(1, 4, True)
  assert budget.thinking_span([10, 11, END], START, END) == budget.ThinkingSpan(0, 2, True)
  assert budget.thinking_span([START, 10, 11], START, END) == budget.ThinkingSpan(1, 3, False)


def test_plan_reuses_short_traces_and_cuts_long_ones():
  span = budget.ThinkingSpan(1, 1 + 300, True)
  plans = {p.budget: p for p in budget.plan(span)}
  assert plans[128] == budget.BudgetPlan(128, 128, True)
  assert plans[256] == budget.BudgetPlan(256, 256, True)
  assert plans[512] == budget.BudgetPlan(512, 300, False)
  assert plans[None] == budget.BudgetPlan(None, 300, False)


def test_plan_unterminated_trace_needs_forced_answer_everywhere():
  span = budget.ThinkingSpan(1, 1 + budget.UNCAPPED_LIMIT, False)
  assert all(p.needs_forced_answer for p in budget.plan(span))


def test_forced_answer_prompt_and_trace_at():
  decode = lambda ids: " ".join(f"t{i}" for i in ids)
  generated = [START] + list(range(100, 110)) + [END]
  span = budget.thinking_span(generated, START, END)
  prompt = budget.forced_answer_prompt("P|", generated, span, 3, decode)
  assert prompt == "P|" + decode([START, 100, 101, 102]) + budget.EARLY_EXIT
  assert budget.trace_at(generated, span, 3, decode) == "t100 t101 t102"
  assert budget.EARLY_EXIT.endswith("</think>.\n\n")
