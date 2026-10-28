import math

import pytest

from cotfaith import models, tokens


class _FakeTokenizer:
  """Minimal stand-in: token id -> decoded text."""

  def __init__(self, pieces):
    self.vocab = {piece: i for i, piece in enumerate(pieces)}
    self._pieces = pieces

  def decode(self, ids, skip_special_tokens=False):
    return "".join(self._pieces[i] for i in ids)


def test_prefix_tokens_for_labels():
  tok = _FakeTokenizer(["ent", " Neutral", "con", " ", "n", "c", "x", "entailment"])
  mapping = tokens.prefix_tokens_for_labels(tok, ("entailment", "neutral", "contradiction"))
  # " " decodes to "" after stripping, which prefixes every label: excluded.
  assert mapping == {0: "entailment", 1: "neutral", 2: "contradiction", 4: "neutral",
                     5: "contradiction", 7: "entailment"}


def test_prefix_tokens_require_every_label():
  with pytest.raises(ValueError):
    tokens.prefix_tokens_for_labels(_FakeTokenizer(["ent"]), ("entailment", "neutral"))
  with pytest.raises(ValueError):
    tokens.prefix_tokens_for_labels(_FakeTokenizer(["e"]), ("Entailment",))


def test_label_probs_from_top_logprobs():
  token_to_label = {0: "entailment", 4: "neutral", 5: "contradiction", 7: "entailment"}
  top = {0: math.log(0.5), 7: math.log(0.2), 4: math.log(0.1), 99: math.log(0.15)}
  probs, mass = tokens.label_probs(top, token_to_label, ("entailment", "neutral", "contradiction"))
  assert probs == pytest.approx({"entailment": 0.7, "neutral": 0.1, "contradiction": 0.0})
  assert mass == pytest.approx(0.8)


def test_model_registry():
  subjects = [m for m in models.MODELS.values() if m.role == "subject"]
  assert [m.params for m in subjects] == [0.6e9, 1.7e9, 4e9, 8e9, 14e9]
  assert all(m.hybrid_thinking for m in subjects)
  assert all(len(m.revision) == 40 for m in models.MODELS.values())
  with pytest.raises(ValueError):
    models.get("Qwen/Qwen3-4B-Instruct-2507")
