"""Generates every intervention candidate for the three test sets (CPU only).

Writes <out_dir>/candidates_<dataset>.parquet and candidates_meta.json. The
naturalness judge (GPU) scores these afterwards; the top 5% per example become
the frozen intervention set.

Usage: python scripts/make_candidates.py [--data-dir data] [--out-dir data/interventions]
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import time

from cotfaith import datasets, interventions


def sha256(path: pathlib.Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--data-dir", default="data")
  parser.add_argument("--out-dir", default="data/interventions")
  args = parser.parse_args()
  os.environ.setdefault("NLTK_DATA", os.path.join(args.data_dir, "nltk_data"))
  out_dir = pathlib.Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  meta = dict(
      git_commit=subprocess.run(
          ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
      ).stdout.strip(),
      versions={
          pkg: importlib.metadata.version(pkg)
          for pkg in ("spacy", "en-core-web-lg", "nltk", "numpy", "pandas")
      },
      datasets={},
  )
  for name in datasets.DATASET_NAMES:
    start = time.time()
    spec = datasets.get_spec(name)
    test_df = datasets.load_splits(name, args.data_dir)["test"]
    n = interventions.candidates_per_example(name)
    candidates = interventions.generate_candidates(spec, test_df, n)
    path = out_dir / f"candidates_{name}.parquet"
    candidates.to_parquet(path)
    meta["datasets"][name] = dict(
        n_test_examples=len(test_df),
        candidates_per_example=n,
        n_candidates=len(candidates),
        n_examples_without_candidates=int(len(test_df) - candidates["data_idx"].nunique()),
        sha256=sha256(path),
        seconds=round(time.time() - start, 1),
    )
    print(name, meta["datasets"][name], flush=True)
  (out_dir / "candidates_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
  main()
