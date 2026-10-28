"""Downloads tokenizer and config files (no weights) for every pinned model.

Usage: python scripts/fetch_tokenizers.py [--cache-dir data/hf]
"""

import argparse

from huggingface_hub import snapshot_download

from cotfaith import models, tokens


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--cache-dir", default=tokens.DEFAULT_CACHE_DIR)
  args = parser.parse_args()
  for info in models.MODELS.values():
    path = snapshot_download(
        info.name, revision=info.revision, cache_dir=args.cache_dir,
        allow_patterns=list(tokens.TOKENIZER_FILES),
    )
    print(info.name, path, flush=True)


if __name__ == "__main__":
  main()
