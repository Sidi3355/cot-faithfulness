"""Thinking budgets: the verbosity axis for traces (PLAN.md section 4.4).

Each sample's trace is generated once, uncapped (up to UNCAPPED_LIMIT thinking
tokens). For a budget B, a trace that ended before B tokens is reused as is;
a longer trace is cut after B thinking tokens and closed with Qwen's early-exit
phrase, and a fresh answer is sampled from that point. A cut sample is a valid
sample of the budgeted process, so this equals running each budget separately.
"""

import dataclasses
from collections.abc import Callable, Sequence

BUDGETS = (128, 256, 512, 1024)
UNCAPPED_LIMIT = 8192

# Qwen3 technical report (arXiv 2505.09388), "thinking budget". Verify the exact
# wording against the report in Phase 0 before any reported run.
EARLY_EXIT = (
    "\n\nConsidering the limited time by the user, I have to give the solution"
    " based on the thinking directly now.\n</think>.\n\n"
)


@dataclasses.dataclass(frozen=True)
class ThinkingSpan:
  """Token indices of the thinking content within the generated ids."""

  start: int  # First token after <think> (0 if the model didn't emit it).
  end: int  # Index of </think>, or len(generated) if never emitted.
  terminated: bool

  @property
  def n_tokens(self) -> int:
    return self.end - self.start


def thinking_span(generated: Sequence[int], think_start_id: int, think_end_id: int) -> ThinkingSpan:
  start = 0
  if generated and generated[0] == think_start_id:
    start = 1
  try:
    end = list(generated).index(think_end_id, start)
  except ValueError:
    return ThinkingSpan(start, len(generated), False)
  return ThinkingSpan(start, end, True)


@dataclasses.dataclass(frozen=True)
class BudgetPlan:
  budget: int | None  # None = uncapped.
  n_thinking_tokens: int  # Thinking tokens the answer at this budget sees.
  needs_forced_answer: bool  # True if the trace is cut here and must be closed.


def plan(span: ThinkingSpan, budgets: Sequence[int] = BUDGETS) -> list[BudgetPlan]:
  """What each budget (plus uncapped) reuses or regenerates for one sample."""
  plans = [
      BudgetPlan(b, min(b, span.n_tokens), span.n_tokens > b or not span.terminated)
      for b in budgets
  ]
  plans.append(BudgetPlan(None, span.n_tokens, not span.terminated))
  return plans


def forced_answer_prompt(
    prompt: str,
    generated: Sequence[int],
    span: ThinkingSpan,
    n_thinking_tokens: int,
    decode: Callable[[Sequence[int]], str],
) -> str:
  """Prompt from which to sample the answer after a cut trace."""
  kept = list(generated[: span.start + n_thinking_tokens])
  return prompt + decode(kept) + EARLY_EXIT


def trace_at(
    generated: Sequence[int],
    span: ThinkingSpan,
    n_thinking_tokens: int,
    decode: Callable[[Sequence[int]], str],
) -> str:
  """Trace text the model had produced after n_thinking_tokens tokens."""
  return decode(list(generated[span.start: span.start + n_thinking_tokens]))
