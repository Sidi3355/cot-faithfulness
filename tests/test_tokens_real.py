"""Tokenizer behaviour on the real, pinned Qwen tokenizers (scripts/fetch_tokenizers.py)."""

import pytest

import upstream
from cotfaith import budget, datasets, models, parse, tokens
from cotfaith.prompts import Template

pytestmark = pytest.mark.hf

QWEN3 = [m.name for m in models.MODELS.values() if m.hybrid_thinking]
QWEN25 = [m.name for m in models.MODELS.values() if not m.hybrid_thinking]


@pytest.mark.parametrize("name", QWEN3)
def test_qwen3_thinking_switch(name):
  tok = tokens.load_tokenizer(name)
  base = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
  # The template's default is thinking on: the model writes <think> itself.
  assert tokens.chat_prompt(tok, "Hi", None) == base
  assert tokens.chat_prompt(tok, "Hi", True) == base
  assert tokens.chat_prompt(tok, "Hi", False) == base + "<think>\n\n</think>\n\n"


@pytest.mark.parametrize("name", QWEN25)
def test_qwen25_template_has_default_system_prompt_and_no_thinking(name):
  tok = tokens.load_tokenizer(name)
  prompt = tokens.chat_prompt(tok, "Hi", None)
  assert prompt.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.")
  assert tokens.chat_prompt(tok, "Hi", False) == prompt


def test_think_tokens_survive_decoding():
  tok = tokens.load_tokenizer("Qwen/Qwen3-0.6B")
  start, end = tokens.think_token_ids(tok)
  text = "<think>\nThe hypothesis says JUDGEMENT: contradiction.\n</think>\n\nJUDGEMENT: neutral"
  ids = tok.encode(text, add_special_tokens=False)
  assert ids[0] == start and end in ids
  # vLLM decodes with skip_special_tokens=True by default.
  decoded = tok.decode(ids, skip_special_tokens=True)
  assert decoded == text
  parsed = parse.parse_completion(datasets.ESNLI, Template.PREDICT_ONLY, decoded, thinking=True)
  assert parsed.label == "neutral"


def test_early_exit_closes_thinking_with_the_real_token():
  tok = tokens.load_tokenizer("Qwen/Qwen3-0.6B")
  _, end = tokens.think_token_ids(tok)
  assert tok.encode(budget.EARLY_EXIT, add_special_tokens=False).count(end) == 1


def test_budget_cut_on_real_tokens():
  tok = tokens.load_tokenizer("Qwen/Qwen3-0.6B")
  start, end = tokens.think_token_ids(tok)
  thinking = " ".join(f"step{i}" for i in range(400))
  ids = tok.encode(f"<think>\n{thinking}\n</think>\n\nJUDGEMENT: neutral", add_special_tokens=False)
  span = budget.thinking_span(ids, start, end)
  assert span.terminated and span.n_tokens > 256
  decode = lambda x: tok.decode(x, skip_special_tokens=False)
  cut = budget.trace_at(ids, span, 128, decode)
  assert len(tok.encode(cut, add_special_tokens=False)) == 128
  assert ("\n" + thinking).startswith(cut)
  prompt = budget.forced_answer_prompt("P", ids, span, 128, decode)
  assert prompt == "P<think>" + cut + budget.EARLY_EXIT


def test_all_qwen3_sizes_share_one_tokenizer():
  vocabs = {name: tokens.load_tokenizer(name).get_vocab() for name in QWEN3}
  first = vocabs[QWEN3[0]]
  assert all(v == first for v in vocabs.values())


@pytest.mark.upstream
@pytest.mark.parametrize("family", [QWEN3[0], QWEN25[0]])
@pytest.mark.parametrize("dataset", datasets.DATASET_NAMES)
def test_label_tokens_match_upstream(family, dataset):
  tok = tokens.load_tokenizer(family)
  labels = datasets.get_spec(dataset).class_labels
  theirs = upstream.module("experiments.models.tokenization").get_prefix_tokens_for_labels(tok, labels)
  ours = tokens.prefix_tokens_for_labels(tok, labels)
  assert ours == theirs
  # The label as it follows its prefix (e.g. " entailment") starts with a mapped
  # token. Digits are tokenized separately from the preceding space, so for
  # ECQA and ComVE the label token comes after a bare space token.
  for label in labels:
    ids = tok.encode(" " + label, add_special_tokens=False)
    if tok.decode(ids[:1]).strip() == "":
      ids = ids[1:]
    assert ours.get(ids[0]) == label
