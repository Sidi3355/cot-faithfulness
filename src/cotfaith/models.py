"""The models in the study, pinned to Hugging Face commits.

Revisions are each repo's head as of 2026-10-28. For the Qwen3 repos, the last
change is a LICENSE file (2025-07-26), and the tokenizer config was last updated
on 2025-05-19, so these are the original hybrid-thinking release.
"""

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class ModelInfo:
  name: str
  revision: str
  role: Literal["subject", "anchor", "judge"]
  # Nominal parameter count from the model name, as the paper's code uses.
  params: float
  hybrid_thinking: bool


MODELS = {
    m.name: m
    for m in (
        ModelInfo("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "subject", 0.6e9, True),
        ModelInfo("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "subject", 1.7e9, True),
        ModelInfo("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "subject", 4e9, True),
        ModelInfo("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "subject", 8e9, True),
        ModelInfo("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18", "subject", 14e9, True),
        ModelInfo("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "anchor", 1.5e9, False),
        ModelInfo("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28", "anchor", 7e9, False),
        ModelInfo("Qwen/Qwen2.5-32B-Instruct-AWQ", "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c", "judge", 32e9, False),
    )
}


def get(name: str) -> ModelInfo:
  try:
    return MODELS[name]
  except KeyError:
    raise ValueError(f"Unknown model: {name}") from None
