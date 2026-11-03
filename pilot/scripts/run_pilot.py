"""Runs pilot configs in order, resuming whatever is unfinished.

Usage: python pilot/scripts/run_pilot.py pilot/configs/R_*empty.yaml [...]
Raw rows go to data/pilot/raw/<phase>/<model>/comve/<condition>/rows/ (untracked);
pilot/scripts/sync_results.sh copies them into pilot/results/raw/ and commits.
"""

import argparse
import pathlib
import time

from cotfaith import config, models, run

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data" / "pilot" / "raw"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("configs", nargs="+")
  parser.add_argument("--threads", type=int, default=4)
  parser.add_argument("--batch-size", type=int, default=16)
  args = parser.parse_args()
  for path in args.configs:
    cfg = config.load(path)
    start = time.time()
    print(f"== {pathlib.Path(path).name}", flush=True)
    # Memory on a 15 GB machine in float32: 3B weights are ~12.4 GB, and a thinking
    # row's KV cache can reach ~4,700 tokens x 230 KB (Qwen3-0.6B) = ~1 GB.
    batch_size = args.batch_size
    if models.get(cfg.model).params >= 3e9 or cfg.thinking:
      batch_size = min(batch_size, 8)
    runner = run.Runner(cfg, str(ROOT / "data"), str(RESULTS), threads=args.threads,
                        batch_size=batch_size, log=lambda m: print(m, flush=True))
    runner.run()
    del runner
    print(f"== done {pathlib.Path(path).name} in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
  main()
