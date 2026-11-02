"""Runs pilot configs in order, resuming whatever is unfinished.

Usage: python pilot/scripts/run_pilot.py pilot/configs/R_*empty.yaml [...]
Raw rows go to pilot/results/raw/<phase>/<model>/comve/<condition>/rows/.
"""

import argparse
import pathlib
import time

from cotfaith import config, run

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "pilot" / "results" / "raw"


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
    runner = run.Runner(cfg, str(ROOT / "data"), str(RESULTS), threads=args.threads,
                        batch_size=args.batch_size, log=lambda m: print(m, flush=True))
    runner.run()
    del runner
    print(f"== done {pathlib.Path(path).name} in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
  main()
