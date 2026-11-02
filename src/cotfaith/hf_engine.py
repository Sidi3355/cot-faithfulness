"""Batched generation with Hugging Face transformers on CPU (the pilot engine).

The prompts passed to one `generate` call are an example's prompts (base, N1,
interventions, N2), which share a long prefix (chat header, demos,
instructions). That prefix is run through the model once and its KV cache is
reused for every row. Each row's remaining tokens are left-padded *after* the
shared prefix and masked, so positions stay contiguous.

Generation runs in chunks; after each chunk, rows that have emitted EOS are
dropped from the batch and their cache rows discarded, so a few long outputs
don't keep the whole batch running.

At every step a recorder stores the label distribution, summing the softmax
probabilities of every token that can start each label (as upstream does), so
the distribution at the label position can be read off after parsing:
- greedy runs record the scores after the model's default processors (for
  Qwen2.5, the repetition penalty from its generation_config), which is exactly
  what upstream's `output_scores` gave;
- sampled runs record the raw logits; temperature/top-k/top-p are applied
  after the recorder.
"""

import copy
import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, LogitsProcessor, LogitsProcessorList
from transformers.generation.logits_process import (
    MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
)

from cotfaith import models, tokens


@dataclasses.dataclass(frozen=True)
class Sampling:
  greedy: bool
  temperature: float = 1.0
  top_p: float = 1.0
  top_k: int = 0
  min_p: float = 0.0
  repetition_penalty: float = 1.0


@dataclasses.dataclass
class Output:
  token_ids: list[int]  # Generated tokens, without padding; includes EOS if emitted.
  label_probs: np.ndarray  # (len(token_ids), n_labels): label distribution at each step.
  finished: bool  # True if the model emitted EOS.


class _LabelRecorder(LogitsProcessor):
  """Stores the per-step label distribution; leaves the scores unchanged."""

  def __init__(self, token_ids: torch.Tensor, label_index: torch.Tensor, n_labels: int):
    self.token_ids, self.label_index, self.n_labels = token_ids, label_index, n_labels
    self.steps: list[torch.Tensor] = []

  def __call__(self, input_ids, scores):
    probs = torch.softmax(scores.float(), dim=-1)[:, self.token_ids]
    out = torch.zeros(scores.shape[0], self.n_labels, dtype=torch.float32)
    self.steps.append(out.index_add_(1, self.label_index, probs))
    return scores


def common_prefix_len(seqs: Sequence[Sequence[int]]) -> int:
  first = seqs[0]
  n = min(len(s) for s in seqs)
  for i in range(n):
    if any(s[i] != first[i] for s in seqs):
      return i
  return n


class HFEngine:
  def __init__(self, model_name: str, dtype: torch.dtype = torch.bfloat16, threads: int = 4,
               cache_dir: str = tokens.DEFAULT_CACHE_DIR):
    info = models.get(model_name)
    torch.set_num_threads(threads)
    self.info = info
    self.tokenizer = AutoTokenizer.from_pretrained(
        info.name, revision=info.revision, cache_dir=cache_dir, local_files_only=True)
    self.model = AutoModelForCausalLM.from_pretrained(
        info.name, revision=info.revision, cache_dir=cache_dir, local_files_only=True, dtype=dtype).eval()
    eos = self.model.generation_config.eos_token_id
    self.eos_ids = set(eos if isinstance(eos, list) else [eos])
    self.pad_id = self.model.generation_config.pad_token_id
    if self.pad_id is None:
      self.pad_id = self.tokenizer.pad_token_id
    # The model's own default, applied by upstream's greedy generate calls.
    self.default_repetition_penalty = float(self.model.generation_config.repetition_penalty or 1.0)

  def encode(self, text: str) -> list[int]:
    return self.tokenizer(text, add_special_tokens=False).input_ids

  def encode_chat(self, user_message: str, thinking: bool | None) -> list[int]:
    return self.encode(tokens.chat_prompt(self.tokenizer, user_message, thinking))

  # Prefix cache -------------------------------------------------------------

  @torch.inference_mode()
  def _prefix(self, prompts: Sequence[Sequence[int]]):
    p = min(common_prefix_len(prompts), min(len(x) for x in prompts) - 1)
    cache = DynamicCache()
    if p > 0:
      self.model(torch.tensor([list(prompts[0][:p])]), past_key_values=cache, use_cache=True)
    return p, cache

  def _batch_inputs(self, prompts, p, cache, rows):
    suffixes = [list(prompts[i][p:]) for i in rows]
    width = max(len(s) for s in suffixes)
    ids = [list(prompts[rows[0]][:p]) + [self.pad_id] * (width - len(s)) + s for s in suffixes]
    mask = [[1] * p + [0] * (width - len(s)) + [1] * len(s) for s in suffixes]
    batch_cache = copy.deepcopy(cache)
    if p > 0:
      batch_cache.batch_repeat_interleave(len(rows))
    return torch.tensor(ids), torch.tensor(mask), batch_cache

  # Next-token distribution (naturalness judge) -------------------------------

  @torch.inference_mode()
  def next_token_topk(self, prompts: Sequence[Sequence[int]], k: int = 20, batch_size: int = 32,
                      repetition_penalty: float | None = None) -> list[list[tuple[str, float]]]:
    """Top-k (raw vocab token, probability) of the first generated token.

    As upstream's judge, the model's default repetition penalty is applied to
    the scores before the softmax.
    """
    penalty = self.default_repetition_penalty if repetition_penalty is None else repetition_penalty
    processor = RepetitionPenaltyLogitsProcessor(penalty) if penalty != 1.0 else None
    p, cache = self._prefix(prompts)
    out = []
    for start in range(0, len(prompts), batch_size):
      rows = list(range(start, min(start + batch_size, len(prompts))))
      ids, mask, batch_cache = self._batch_inputs(prompts, p, cache, rows)
      position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
      logits = self.model(
          ids[:, p:], attention_mask=mask, position_ids=position_ids[:, p:],
          past_key_values=batch_cache, use_cache=True,
      ).logits[:, -1, :].float()
      if processor is not None:
        logits = processor(ids, logits)
      probs = torch.softmax(logits, dim=-1)
      top = torch.topk(probs, k, dim=-1)
      for r in range(len(rows)):
        ids_k = top.indices[r].tolist()
        out.append(list(zip(self.tokenizer.convert_ids_to_tokens(ids_k), top.values[r].tolist())))
    return out

  # Generation ----------------------------------------------------------------

  def _processors(self, sampling: Sampling, recorder: _LabelRecorder):
    if sampling.greedy:
      return LogitsProcessorList([recorder])
    procs = [recorder]
    if sampling.repetition_penalty != 1.0:
      procs.insert(0, RepetitionPenaltyLogitsProcessor(sampling.repetition_penalty))
    if sampling.temperature != 1.0:
      procs.append(TemperatureLogitsWarper(sampling.temperature))
    if sampling.top_k:
      procs.append(TopKLogitsWarper(sampling.top_k))
    if sampling.top_p < 1.0:
      procs.append(TopPLogitsWarper(sampling.top_p))
    if sampling.min_p > 0:
      procs.append(MinPLogitsWarper(sampling.min_p))
    return LogitsProcessorList(procs)

  @torch.inference_mode()
  def generate(
      self,
      prompts: Sequence[Sequence[int]],
      n: int,
      sampling: Sampling,
      max_new_tokens: int,
      label_tokens: Mapping[int, int],
      n_labels: int,
      seed: int = 0,
      batch_size: int = 16,
      chunk: int = 64,
  ) -> list[list[Output]]:
    """n outputs per prompt. `label_tokens` maps token id -> label index."""
    ids_t = torch.tensor(sorted(label_tokens))
    lab_t = torch.tensor([label_tokens[t] for t in sorted(label_tokens)])
    p, cache = self._prefix(prompts)
    rows = [(i, s) for i in range(len(prompts)) for s in range(n)]
    results: list[list[Output | None]] = [[None] * n for _ in prompts]
    for start in range(0, len(rows), batch_size):
      batch_rows = rows[start:start + batch_size]
      torch.manual_seed(seed * 1_000_003 + start)
      ids, mask, batch_cache = self._batch_inputs(prompts, p, cache, [i for i, _ in batch_rows])
      outs = self._generate_chunked(ids, mask, batch_cache, sampling, max_new_tokens,
                                    _LabelRecorder(ids_t, lab_t, n_labels), chunk)
      for (i, s), o in zip(batch_rows, outs):
        results[i][s] = o
    return results  # type: ignore[return-value]

  def _generate_chunked(self, ids, mask, cache, sampling, max_new_tokens, recorder, chunk):
    b = ids.shape[0]
    toks: list[list[int]] = [[] for _ in range(b)]
    probs: list[list[torch.Tensor]] = [[] for _ in range(b)]
    finished = [False] * b
    active = list(range(b))
    remaining = max_new_tokens
    gen_kwargs = dict(
        do_sample=not sampling.greedy, pad_token_id=self.pad_id, eos_token_id=sorted(self.eos_ids),
        return_dict_in_generate=True, use_cache=True,
    )
    if sampling.greedy:
      gen_kwargs["repetition_penalty"] = sampling.repetition_penalty
    else:  # Neutralise generation_config defaults; our processors do the work.
      gen_kwargs.update(temperature=1.0, top_k=0, top_p=1.0, min_p=None, repetition_penalty=1.0)
    while active and remaining > 0:
      step = min(chunk, remaining)
      recorder.steps = []
      out = self.model.generate(
          input_ids=ids, attention_mask=mask, past_key_values=cache, max_new_tokens=step,
          logits_processor=self._processors(sampling, recorder), **gen_kwargs,
      )
      new = out.sequences[:, ids.shape[1]:]
      step_probs = torch.stack(recorder.steps, dim=1)
      keep = []
      for j, r in enumerate(active):
        row = new[j].tolist()
        cut = len(row)
        for k, t in enumerate(row):
          if t in self.eos_ids:
            cut, finished[r] = k + 1, True
            break
        toks[r].extend(row[:cut])
        probs[r].append(step_probs[j, :cut])
        if not finished[r]:
          keep.append(j)
      remaining -= new.shape[1]
      if not keep or new.shape[1] == 0:
        break
      keep_t = torch.tensor(keep)
      ids = out.sequences[keep_t]
      mask = torch.cat([mask[keep_t], torch.ones(len(keep), new.shape[1], dtype=mask.dtype)], dim=1)
      cache = out.past_key_values
      cache.batch_select_indices(keep_t)
      active = [active[j] for j in keep]
    return [
        Output(toks[r], torch.cat(probs[r]).numpy() if probs[r] else np.zeros((0, recorder.n_labels)), finished[r])
        for r in range(b)
    ]


def label_token_index(tokenizer, token_ids: Sequence[int], text: str, char_start: int) -> int | None:
  """Index of the generated token containing text[char_start], as upstream finds it.

  `text` must be tokenizer.decode(token_ids, skip_special_tokens=True).
  """
  enc = tokenizer(text, add_special_tokens=False)
  ids = list(token_ids)
  if enc.input_ids == ids or enc.input_ids == ids[:-1]:
    return enc.char_to_token(char_start)
  # Fall back to decoding prefixes, as upstream's decode_with_spans.
  for k in range(len(ids)):
    if len(tokenizer.decode(ids[: k + 1], skip_special_tokens=True)) > char_start:
      return k
  return None
