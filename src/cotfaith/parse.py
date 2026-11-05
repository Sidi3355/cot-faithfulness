"""Parsing model completions into (trace, label, explanation).

The answer regexes are upstream's instruction-tuned patterns, verbatim
(google-deepmind/corr_faith, Apache-2.0,
experiments/dataset_specific/classification_datasets.py at c6e862b). As
upstream, an explanation is the rest of the line after "EXPLANATION:".

With thinking on, only the text after the first `</think>` is parsed, because a
trace can itself contain strings like "JUDGEMENT:".

Pilot arm C's explain-then-predict runs use `tolerant=True`: an answer that
upstream's pattern rejects is retried with one that lets the explanation span
several lines and allows a full stop or bold markers after the label. Answers
upstream's pattern accepts parse exactly as upstream.
"""

import dataclasses
import functools
import re

from cotfaith.datasets import DatasetSpec
from cotfaith.prompts import Template

THINK_START = "<think>"
THINK_END = "</think>"

_SPACE_EXCEPT_NEWLINE = r"[^\S\r\n]"
# Instruction-tuned models sometimes start with "Sure! ..." or repeat the input.
ALLOWED_PREFIX = r"(?:.*\n)*" + f"{_SPACE_EXCEPT_NEWLINE}?"


def _maybe_bold(s: str) -> str:
  return r"(?:\*\*)?" + s + r"(?:\*\*)?"


@functools.cache
def completion_regex(spec: DatasetSpec, template: Template) -> re.Pattern[str]:
  match template:
    case Template.PREDICT_THEN_EXPLAIN:
      pattern = (
          ALLOWED_PREFIX + _maybe_bold(spec.class_prefix) + r" ?(?P<pred_label>.*)\s+"
          + _maybe_bold(spec.explanation_prefix) + r"(?P<explanation>.*)"
      )
    case Template.EXPLAIN_THEN_PREDICT:
      pattern = (
          ALLOWED_PREFIX + _maybe_bold(spec.explanation_prefix) + r"(?P<explanation>.*)\s+"
          + _maybe_bold(spec.class_prefix) + r" ?(?P<pred_label>.*)"
      )
    case Template.PREDICT_ONLY:
      pattern = ALLOWED_PREFIX + spec.class_prefix + " (?P<pred_label>.*)"
  return re.compile(pattern)


@functools.cache
def tolerant_regex(spec: DatasetSpec) -> re.Pattern[str]:
  """Explain-then-predict fallback: multi-line explanation, then the label alone on its line."""
  labels = "|".join(re.escape(l) for l in sorted(spec.class_labels, key=len, reverse=True))
  line_end = f"{_SPACE_EXCEPT_NEWLINE}*" + r"(?:\*\*)?[.!]?(?:\*\*)?" + f"{_SPACE_EXCEPT_NEWLINE}*" + r"(?:\n|$)"
  return re.compile(
      ALLOWED_PREFIX + _maybe_bold(spec.explanation_prefix) + r"(?P<explanation>[\s\S]*?)\s+"
      + _maybe_bold(spec.class_prefix) + r" ?(?P<pred_label>(?i:" + labels + r"))(?=" + line_end + ")"
  )


@dataclasses.dataclass(frozen=True)
class Answer:
  label: str | None
  explanation: str | None
  error: str | None
  # Character offset in `text` where the label group starts (upstream reads the
  # label probabilities at the token containing it).
  label_start: int | None = None


def parse_answer(spec: DatasetSpec, template: Template, text: str, tolerant: bool = False) -> Answer:
  if tolerant and template is not Template.EXPLAIN_THEN_PREDICT:
    raise ValueError("the tolerant parser is defined for explain-then-predict only")
  match = completion_regex(spec, template).match(text)
  if match is None:
    upstream = Answer(None, None, "no_match")
  else:
    label = match.group("pred_label").lower().strip()
    explanation = None if template is Template.PREDICT_ONLY else match.group("explanation")
    if label in spec.class_labels:
      return Answer(label, explanation, None, match.start("pred_label"))
    upstream = Answer(None, explanation, "unknown_label")
  if tolerant and (match := tolerant_regex(spec).match(text)) is not None:
    return Answer(match.group("pred_label").lower(), match.group("explanation"), None,
                  match.start("pred_label"))
  return upstream


@dataclasses.dataclass(frozen=True)
class Completion:
  trace: str | None  # None when thinking is off.
  answer: str  # Text after </think>, or the whole completion.
  terminated: bool  # False if the trace never reached </think>.


def split_thinking(text: str, thinking: bool) -> Completion:
  if not thinking:
    return Completion(None, text, True)
  body = text.lstrip()
  if body.startswith(THINK_START):
    body = body[len(THINK_START):]
  end = body.find(THINK_END)
  if end < 0:
    return Completion(body, "", False)
  return Completion(body[:end], body[end + len(THINK_END):], True)


@dataclasses.dataclass(frozen=True)
class Parsed:
  trace: str | None
  terminated: bool
  label: str | None
  explanation: str | None
  error: str | None


def parse_completion(spec: DatasetSpec, template: Template, text: str, thinking: bool) -> Parsed:
  completion = split_thinking(text, thinking)
  if not completion.terminated:
    return Parsed(completion.trace, False, None, None, "unterminated")
  answer = parse_answer(spec, template, completion.answer)
  return Parsed(completion.trace, True, answer.label, answer.explanation, answer.error)
