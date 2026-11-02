"""CPU throughput benchmark used to size the pilot (pilot/README.md).

Needs torch, transformers and accelerate on top of the project environment, and
the model weights in data/hf. Usage:
  python pilot/scripts/bench_cpu.py Qwen/Qwen3-0.6B [n_thinking_traces] > out.json
"""

import copy
import json
import pathlib
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cotfaith import datasets, models, parse, prompts, tokens  # noqa: E402

torch.set_num_threads(4)
NAME = sys.argv[1]
N_THINK = int(sys.argv[2]) if len(sys.argv) > 2 else 0
info = models.get(NAME)
tok = AutoTokenizer.from_pretrained(NAME, revision=info.revision, cache_dir=str(ROOT / "data/hf"), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    NAME, revision=info.revision, cache_dir=str(ROOT / "data/hf"),
    local_files_only=True, dtype=torch.bfloat16,
).eval()

spec = datasets.COMVE
splits = datasets.load_splits("comve", str(ROOT / "data"))
test = splits["test"].sample(frac=1, random_state=0)
out = {"model": NAME}


def chat(row, idx, template, demos, thinking):
  return tokens.chat_prompt(tok, prompts.build_prompt(spec, splits["train"], idx, row, template, demos), thinking)


PE, WITH = prompts.Template.PREDICT_THEN_EXPLAIN, prompts.Demos.WITH_EXPLANATIONS
idx, row = next(test.iterrows())
text = chat(row, idx, PE, WITH, False)
ids = tok(text, return_tensors="pt").input_ids
out["prompt_tokens"] = ids.shape[1]

with torch.inference_mode():
  model(ids[:, :32])  # Warm up.
  t = time.perf_counter(); model(ids); out["prefill_s_batch1"] = time.perf_counter() - t

  # Prefix reuse: demos are shared by an example's base run and all its edits.
  split = text.rfind("Now I'm going to show you")
  prefix_ids = tok(text[:split], return_tensors="pt").input_ids
  t = time.perf_counter(); cache = model(prefix_ids, use_cache=True).past_key_values
  out["prefix_tokens"] = prefix_ids.shape[1]
  suffix_ids = ids[:, prefix_ids.shape[1]:]
  t = time.perf_counter(); model(suffix_ids, past_key_values=copy.deepcopy(cache), use_cache=True)
  out["suffix_tokens"] = suffix_ids.shape[1]
  out["suffix_prefill_s"] = time.perf_counter() - t

  # Decode throughput, greedy, batch 1 and 16 (same prompt, so no padding).
  for bs in (1, 16):
    batch = ids.repeat(bs, 1)
    t = time.perf_counter()
    gen = model.generate(batch, max_new_tokens=64, min_new_tokens=64, do_sample=False)
    out[f"gen64_s_batch{bs}"] = time.perf_counter() - t
  out["sample_completion"] = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)

  # Real thinking traces (Phase 2 prompt), sampled with the plan's settings.
  traces = []
  for k, (i, r) in enumerate(test.iloc[1:1 + N_THINK].iterrows()):
    p = tok(chat(r, i, prompts.Template.PREDICT_ONLY, prompts.Demos.LABELS_ONLY, True), return_tensors="pt").input_ids
    t = time.perf_counter()
    g = model.generate(p, max_new_tokens=3000, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
    secs = time.perf_counter() - t
    completion = tok.decode(g[0, p.shape[1]:], skip_special_tokens=True)
    parsed = parse.parse_completion(spec, prompts.Template.PREDICT_ONLY, completion, thinking=True)
    traces.append(dict(
        new_tokens=int(g.shape[1] - p.shape[1]), seconds=round(secs, 1), terminated=parsed.terminated,
        label=parsed.label, gold=spec.true_label(r),
        trace_start=(parsed.trace or "")[:300],
    ))
  out["traces"] = traces

print(json.dumps(out, indent=1))
