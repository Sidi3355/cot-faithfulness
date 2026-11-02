"""Writes the pilot's run configs (pilot/PILOT_PLAN.pdf) into pilot/configs/.

Usage: python pilot/scripts/make_configs.py --r-examples N
  --r-examples   examples for arm R (20 for the pre-pilot, then the chosen N)
"""

import argparse
import pathlib

from cotfaith import config, models, prompts

ROOT = pathlib.Path(__file__).resolve().parents[2]
INTERVENTIONS = ROOT / "data" / "pilot" / "natural_comve.parquet"
R_MODELS = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct")
C_MODEL = "Qwen/Qwen3-0.6B"
C_EXAMPLES = 40
LENGTHS = ("very_concise", "concise", "empty", "comprehensive", "very_comprehensive")
BUDGETS = (128, 256, 512, 1024)
THINKING_CAP = 4096
K = 3


def _base(model: str, n_examples: int, **kw) -> dict:
  info = models.get(model)
  return dict(
      model=model, model_revision=info.revision, dataset="comve",
      interventions_path=str(INTERVENTIONS.relative_to(ROOT) if INTERVENTIONS.is_relative_to(ROOT) else INTERVENTIONS),
      interventions_sha256=INTERVENTIONS.with_suffix(".sha256").read_text().strip(),
      engine="hf_cpu", pilot=True, n_examples=n_examples, **kw,
  )


def configs(r_examples: int) -> dict[str, config.RunConfig]:
  out = {}
  sampled = dict(config.SAMPLED.model_dump(), n=K)
  for model in R_MODELS:
    rep = models.get(model).default_repetition_penalty
    for length in LENGTHS:
      name = f"R_{model.split('/')[1]}_{length}"
      out[name] = config.RunConfig(**_base(
          model, r_examples, experiment_id=name, phase="phase1", template="pe", demos="with_explanations",
          length=length, decoding={"mode": "greedy", "repetition_penalty": rep},
          controls={"n1": True, "n2": False},
      ))
  for template in ("pe", "ep"):
    for demos in ("with_explanations", "labels_only"):
      for length in LENGTHS:
        name = f"C1_{template}_{demos}_{length}"
        out[name] = config.RunConfig(**_base(
            C_MODEL, C_EXAMPLES, experiment_id=name, phase="phase1", template=template, demos=demos,
            length=length, decoding={"mode": "greedy"},
        ))
  out["C2_thinking"] = config.RunConfig(**_base(
      C_MODEL, C_EXAMPLES, experiment_id="C2_thinking", phase="phase2", template="po", demos="labels_only",
      thinking=True, budgets=BUDGETS, thinking_cap=THINKING_CAP, max_new_tokens=THINKING_CAP + 64,
      decoding=sampled,
  ))
  for length in LENGTHS:
    name = f"C3A_{length}"
    out[name] = config.RunConfig(**_base(
        C_MODEL, C_EXAMPLES, experiment_id=name, phase="phase3", template="ep", demos="labels_only",
        length=length, decoding=sampled,
    ))
  out["C3R"] = config.RunConfig(**_base(
      C_MODEL, C_EXAMPLES, experiment_id="C3R", phase="phase3", template="po", demos="labels_only",
      decoding=sampled,
  ))
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--r-examples", type=int, required=True)
  args = parser.parse_args()
  out_dir = ROOT / "pilot" / "configs"
  out_dir.mkdir(parents=True, exist_ok=True)
  for name, cfg in configs(args.r_examples).items():
    (out_dir / f"{name}.yaml").write_text(config.dump(cfg))
  print(f"wrote {len(configs(args.r_examples))} configs to {out_dir}")


if __name__ == "__main__":
  main()
