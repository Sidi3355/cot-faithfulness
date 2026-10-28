"""Mention detectors (E) and input-copy rates.

E_paper is upstream's rule (experiments/counterfactual_interventions.py and
experiments/scripts/evaluate_faithfulness.py in google-deepmind/corr_faith,
Apache-2.0): a case-insensitive substring match OR a Snowball-stem match
against whitespace-split words. E_word and E_novel are the secondary detectors
defined in PLAN.md section 4.2.
"""

import functools
import re
import string

from nltk.stem import snowball

COPY_NGRAM = 3
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


@functools.cache
def _stemmer() -> snowball.SnowballStemmer:
  return snowball.SnowballStemmer("english")


def tokens(text: str) -> list[str]:
  """Lowercase word tokens, used by E_word, E_novel and the copy rates."""
  return _TOKEN.findall(text.lower())


def mentions_paper(word: str, text: str | None) -> bool | None:
  """Upstream's E: substring match, or stemmed whole-word match."""
  if text is None:
    return None
  stem = _stemmer().stem
  if word.lower() in text.lower():
    return True
  stemmed_words = {stem(w.rstrip(string.punctuation)) for w in text.split()}
  return stem(word).rstrip(string.punctuation) in stemmed_words


def _occurrences(word: str, text_tokens: list[str]) -> list[int]:
  stem = _stemmer().stem
  word, word_stem = word.lower(), stem(word.lower())
  return [j for j, t in enumerate(text_tokens) if t == word or stem(t) == word_stem]


def mentions_word(word: str, text: str | None) -> bool | None:
  """Whole-word or stemmed match; no substring hits ("old" in "told")."""
  if text is None:
    return None
  return bool(_occurrences(word, tokens(text)))


def _ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
  return {tuple(toks[k:k + n]) for k in range(len(toks) - n + 1)}


def mentions_novel(word: str, text: str | None, input_text: str) -> bool | None:
  """E_word, ignoring occurrences inside text copied from the input.

  An occurrence is copied if some COPY_NGRAM-word window of the text that
  contains it also appears in the (edited) input.
  """
  if text is None:
    return None
  text_tokens = tokens(text)
  input_ngrams = _ngrams(tokens(input_text), COPY_NGRAM)
  n = COPY_NGRAM
  for j in _occurrences(word, text_tokens):
    windows = (
        tuple(text_tokens[s:s + n])
        for s in range(max(0, j - n + 1), min(j, len(text_tokens) - n) + 1)
    )
    if not any(w in input_ngrams for w in windows):
      return True
  return False


def copy_rates(text: str | None, input_text: str) -> tuple[float, float]:
  """(share of input word types reused in the text,
      share of text tokens inside COPY_NGRAM-grams copied from the input)."""
  if text is None:
    return float("nan"), float("nan")
  text_tokens, input_tokens = tokens(text), tokens(input_text)
  input_types = set(input_tokens)
  reused = len(input_types & set(text_tokens)) / len(input_types) if input_types else float("nan")
  if not text_tokens:
    return reused, float("nan")
  input_ngrams = _ngrams(input_tokens, COPY_NGRAM)
  copied = [False] * len(text_tokens)
  for s in range(len(text_tokens) - COPY_NGRAM + 1):
    if tuple(text_tokens[s:s + COPY_NGRAM]) in input_ngrams:
      copied[s:s + COPY_NGRAM] = [True] * COPY_NGRAM
  return reused, sum(copied) / len(text_tokens)
