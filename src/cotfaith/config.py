"""One YAML config per run, validated against the plan's design.

A config fixes everything that produced a results directory: model and its
revision, dataset, prompt, thinking switch and budgets, decoding, the frozen
intervention file, controls, seed and inference settings. The results layout
(PLAN.md section 10) is derived from it.
"""

import pathlib
import re
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from cotfaith import budget, models, prompts
from cotfaith.datasets import DATASET_NAMES

Phase = Literal["phase1", "phase1b", "phase2", "phase3"]

# Files inside a results directory.
CONFIG_FILE = "config.yaml"
BASE_FILE = "base.parquet"
ROWS_FILE = "rows.parquet"
META_FILE = "meta.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")


class _Strict(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)


class Decoding(_Strict):
  mode: Literal["greedy", "sample"]
  temperature: float = 0.0
  top_p: float = 1.0
  top_k: int = -1
  min_p: float = 0.0
  n: int = 1
  repetition_penalty: float = 1.0

  @model_validator(mode="after")
  def _check(self):
    if self.mode == "greedy" and (self.temperature != 0.0 or self.n != 1):
      raise ValueError("greedy decoding uses temperature 0 and n = 1")
    if self.mode == "sample" and (self.temperature <= 0 or self.n < 1):
      raise ValueError("sampling needs temperature > 0 and n >= 1")
    return self


# Phases 2-3 (PLAN.md section 4.3): identical settings in both thinking modes.
SAMPLED = Decoding(mode="sample", temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, n=5)
GREEDY = Decoding(mode="greedy")


class Controls(_Strict):
  n1: bool = True  # Rerun of the base prompt.
  n2: bool = True  # Whitespace-only twin of every intervention.


class Vllm(_Strict):
  dtype: str = "bfloat16"
  batch_invariant: bool = True
  max_model_len: int = 16384
  gpu_memory_utilization: float = 0.90


class RunConfig(_Strict):
  experiment_id: str
  phase: Phase
  model: str
  # Must equal the pinned revision in cotfaith.models, unless this is a smoke run.
  model_revision: str
  dataset: Literal[DATASET_NAMES]
  template: prompts.Template
  demos: prompts.Demos
  length: Literal[tuple(prompts.LENGTH_INSTRUCTIONS)] = "empty"
  thinking: bool = False
  budgets: tuple[int, ...] = ()
  decoding: Decoding
  interventions_path: str
  interventions_sha256: str
  controls: Controls = Controls()
  seed: int = 0
  max_new_tokens: int = 500
  vllm: Vllm = Vllm()
  engine: Literal["vllm", "hf_cpu"] = "vllm"
  # Number format for hf_cpu runs (vLLM uses vllm.dtype). The CPU pilot uses
  # float32: fast on CPUs without bf16 hardware and identical across hosts.
  dtype: Literal["bfloat16", "float32"] = "bfloat16"
  # Thinking tokens after which a trace counts as unfinished ("uncapped" budget).
  thinking_cap: int = budget.UNCAPPED_LIMIT
  # CPU pilot (pilot/): runs on the first n_examples test examples; may use
  # K = 3 and a lower thinking cap. Pilot runs are never part of the main study.
  pilot: bool = False
  n_examples: int | None = None
  # Smoke runs are for debugging only and are never reported.
  smoke: bool = False
  # Answer parsing (cotfaith.parse): "tolerant" retries answers upstream's
  # pattern rejects (pilot arm C, explain-then-predict only).
  parser: Literal["upstream", "tolerant"] = "upstream"

  @property
  def is_qwen3(self) -> bool:
    return models.get(self.model).hybrid_thinking

  @model_validator(mode="after")
  def _check(self):
    info = models.get(self.model)  # Raises for models outside the study.
    if info.role == "judge":
      raise ValueError("the naturalness judge is not a subject model")
    if info.role == "pilot" and not self.pilot:
      raise ValueError(f"{self.model} is only used in the pilot")
    if self.pilot != (self.n_examples is not None):
      raise ValueError("pilot runs set n_examples; main-study runs use every example")
    if not self.pilot and self.thinking_cap != budget.UNCAPPED_LIMIT:
      raise ValueError(f"main-study runs use a thinking cap of {budget.UNCAPPED_LIMIT}")
    if not 0 < self.thinking_cap <= budget.UNCAPPED_LIMIT:
      raise ValueError(f"thinking_cap must lie in (0, {budget.UNCAPPED_LIMIT}]")
    prompts.include_explanations_in_fewshot(self.template, self.demos)  # Raises if invalid.
    if self.parser == "tolerant" and (self.template is not prompts.Template.EXPLAIN_THEN_PREDICT
                                      or not self.pilot):
      raise ValueError("the tolerant parser is for pilot explain-then-predict runs")
    if self.template is prompts.Template.PREDICT_ONLY and self.length != "empty":
      raise ValueError("length instructions need a template with an explanation")
    if self.thinking and not self.is_qwen3:
      raise ValueError("thinking mode is only defined for Qwen3 hybrid models")
    if self.thinking and self.template is not prompts.Template.PREDICT_ONLY:
      raise ValueError("thinking runs request no explanation (predict-only template)")
    if self.thinking != bool(self.budgets):
      raise ValueError("budgets are required with thinking and not allowed without it")
    if any(b <= 0 or b >= self.thinking_cap for b in self.budgets):
      raise ValueError(f"budgets must lie in (0, thinking_cap = {self.thinking_cap})")
    if not self.smoke and self.model_revision != info.revision:
      raise ValueError(f"reported runs need the pinned revision {info.revision}")
    if not self.smoke and not _SHA256.fullmatch(self.interventions_sha256):
      raise ValueError("reported runs need the frozen intervention file's SHA-256")
    if self.phase in ("phase1", "phase1b"):
      if self.decoding.mode != "greedy" or self.thinking:
        raise ValueError("phase 1 runs are greedy with thinking off")
      if self.decoding.repetition_penalty != info.default_repetition_penalty:
        raise ValueError(
            "phase 1 follows the paper's code, which applied the model's default repetition"
            f" penalty ({info.default_repetition_penalty} for {self.model})"
        )
    else:
      allowed_n = (3, SAMPLED.n) if self.pilot else (SAMPLED.n,)
      if self.decoding.model_copy(update={"n": SAMPLED.n}) != SAMPLED or self.decoding.n not in allowed_n:
        raise ValueError("phases 2-3 use the shared sampling settings (config.SAMPLED; pilot: n = 3)")
      if not self.is_qwen3:
        raise ValueError("phases 2-3 use the Qwen3 models")
    if self.phase == "phase2" and not self.thinking:
      raise ValueError("phase 2 runs have thinking on")
    if self.thinking and self.max_new_tokens < self.thinking_cap:
      raise ValueError(f"thinking runs need max_new_tokens >= thinking_cap ({self.thinking_cap})")
    return self

  @property
  def condition_id(self) -> str:
    parts = [self.template.value, self.demos.value, self.length]
    parts.append("think" if self.thinking else "nothink")
    parts.append(f"{self.decoding.mode}{self.decoding.n}")
    if self.parser != "upstream":
      parts.append(self.parser)
    return "-".join(parts)

  def results_dir(self, root: str | pathlib.Path) -> pathlib.Path:
    model_slug = self.model.replace("/", "__")
    return pathlib.Path(root) / self.phase / model_slug / self.dataset / self.condition_id


def load(path: str | pathlib.Path) -> RunConfig:
  return RunConfig.model_validate(yaml.safe_load(pathlib.Path(path).read_text()))


def dump(config: RunConfig) -> str:
  return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
