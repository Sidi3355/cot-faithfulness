"""Tokenizer-level helpers: chat templates, think tokens, label tokens.

`prefix_tokens_for_labels` is ported verbatim from google-deepmind/corr_faith
(Apache-2.0), experiments/models/tokenization.py at c6e862b: a token counts
towards a label if its decoded text (lowercased, leading whitespace removed) is
a prefix of exactly one label.
"""

import functools
from collections.abc import Mapping, Sequence

import numpy as np

from cotfaith import models

DEFAULT_CACHE_DIR = "data/hf"
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "config.json", "generation_config.json",
)
THINK_START = "<think>"
THINK_END = "</think>"
# Upstream logs a warning when the label tokens hold less than this much mass.
MIN_LABEL_MASS = 0.8


@functools.cache
def load_tokenizer(name: str, cache_dir: str = DEFAULT_CACHE_DIR):
  """Tokenizer at the pinned revision, from the local cache only."""
  from transformers import AutoTokenizer

  info = models.get(name)
  return AutoTokenizer.from_pretrained(
      info.name, revision=info.revision, cache_dir=cache_dir, local_files_only=True
  )


def chat_prompt(tokenizer, user_message: str, thinking: bool | None) -> str:
  """The prompt text for one user message, ending with the assistant turn.

  `thinking` sets Qwen3's `enable_thinking`; None leaves the template default
  (thinking on for Qwen3), which is what the upstream code did. Models without
  a thinking switch ignore it.
  """
  kwargs = {} if thinking is None else {"enable_thinking": thinking}
  return tokenizer.apply_chat_template(
      [{"role": "user", "content": user_message}],
      tokenize=False, add_generation_prompt=True, **kwargs,
  )


def think_token_ids(tokenizer) -> tuple[int, int]:
  ids = tokenizer.convert_tokens_to_ids([THINK_START, THINK_END])
  if tokenizer.unk_token_id in ids or None in ids:
    raise ValueError("tokenizer has no think tokens")
  return ids[0], ids[1]


def prefix_tokens_for_labels(tokenizer, label_options: Sequence[str]) -> dict[int, str]:
  """All tokens that could be the first token of exactly one label."""
  for label in label_options:
    if label != label.lower().lstrip():
      raise ValueError(f"Label {label} must be all lowercase and stripped of leading whitespace.")
  token_to_label = {}
  for token in tokenizer.vocab.values():
    stripped = tokenizer.decode([token], skip_special_tokens=True).lower().lstrip()
    possible = [label for label in label_options if label.startswith(stripped)]
    if len(possible) == 1:
      token_to_label[token] = possible[0]
  for label in label_options:
    if label not in token_to_label.values():
      raise ValueError(f"No unambiguous token found for label {label}.")
  return token_to_label


def label_probs(
    top_logprobs: Mapping[int, float],
    token_to_label: Mapping[int, str],
    label_options: Sequence[str],
) -> tuple[dict[str, float], float]:
  """Label distribution at the label position from the top-k logprobs.

  Returns ({label: summed probability}, total label mass). Tokens outside the
  top k count as 0 (PLAN.md section 2.2); callers flag rows whose mass is
  below MIN_LABEL_MASS.
  """
  probs = {label: 0.0 for label in label_options}
  for token, logprob in top_logprobs.items():
    label = token_to_label.get(token)
    if label is not None:
      probs[label] += float(np.exp(logprob))
  return probs, sum(probs.values())
