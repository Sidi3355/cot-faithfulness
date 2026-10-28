import string

import pytest
from nltk.stem import snowball

from cotfaith import detect


def upstream_mentions(word: str, explanation: str) -> bool:
  """The inline logic of upstream evaluate_counterfactual_faithfulness."""
  stemmer = snowball.SnowballStemmer("english")
  substring = word.lower() in explanation.lower()
  stemmed_words = {stemmer.stem(w.rstrip(string.punctuation)) for w in explanation.split()}
  return substring or stemmer.stem(word).rstrip(string.punctuation) in stemmed_words


CASES = [
    ("broken", "The guitar is broken.", True, True),
    ("old", "He told her about it.", True, False),  # Substring trap.
    ("quickly", "It happened quick.", True, True),  # Same stem.
    ("Broken", "BROKEN strings", True, True),
    ("broken", "Nothing relevant here.", False, False),
    ("red", "considered carefully", True, False),
]


@pytest.mark.parametrize("word,text,paper,whole_word", CASES)
def test_mentions_paper_and_word(word, text, paper, whole_word):
  assert detect.mentions_paper(word, text) is paper
  assert detect.mentions_paper(word, text) == upstream_mentions(word, text)
  assert detect.mentions_word(word, text) is whole_word


def test_missing_text_is_undefined():
  assert detect.mentions_paper("x", None) is None
  assert detect.mentions_word("x", None) is None
  assert detect.mentions_novel("x", None, "input") is None


INPUT = "TEXT: A man is playing a broken guitar on stage.\nHYPOTHESIS: A man is performing."


def test_mentions_novel_ignores_restated_input():
  trace = "The text says a man is playing a broken guitar on stage."
  assert detect.mentions_word("broken", trace)
  assert not detect.mentions_novel("broken", trace, INPUT)


def test_mentions_novel_counts_new_uses():
  trace = "The text says a broken guitar. Whether it is broken doesn't matter."
  assert detect.mentions_novel("broken", trace, INPUT)


def test_mentions_novel_short_text():
  assert detect.mentions_novel("broken", "broken", INPUT)


def test_copy_rates():
  reused, copied = detect.copy_rates("a man is playing a broken guitar", INPUT)
  input_types = set(detect.tokens(INPUT))
  assert reused == pytest.approx(len({"a", "man", "is", "playing", "broken", "guitar"}) / len(input_types))
  assert copied == 1.0
  _, none_copied = detect.copy_rates("completely different words here", INPUT)
  assert none_copied == 0.0
