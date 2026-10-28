import functools
import os

import pytest

import upstream
from cotfaith import datasets, models

os.environ.setdefault("NLTK_DATA", str(upstream.DATA_DIR / "nltk_data"))


def _have_data() -> bool:
  return (upstream.DATA_DIR / "e-SNLI" / "dataset" / "esnli_test.csv").is_file()


def _have_nlp() -> bool:
  try:
    import spacy.util
  except ImportError:
    return False
  return spacy.util.is_package("en_core_web_lg") and (
      upstream.DATA_DIR / "nltk_data" / "corpora" / "wordnet"
  ).is_dir()


def _have_tokenizers() -> bool:
  return all(
      list((upstream.DATA_DIR / "hf").glob(f"models--{m.name.replace('/', '--')}/snapshots/{m.revision}/tokenizer.json"))
      for m in models.MODELS.values()
  )


def pytest_collection_modifyitems(config, items):
  checks = {
      "data": (_have_data(), "datasets missing: run scripts/fetch_data.sh"),
      "nlp": (_have_nlp(), "spaCy en_core_web_lg or WordNet missing"),
      "upstream": (upstream.available(), "third_party/corr_faith submodule missing"),
      "hf": (_have_tokenizers(), "tokenizers missing: run scripts/fetch_tokenizers.py"),
  }
  for item in items:
    for marker, (ok, reason) in checks.items():
      if marker in item.keywords and not ok:
        item.add_marker(pytest.mark.skip(reason=reason))


@functools.cache
def _splits(name: str):
  return datasets.load_splits(name, str(upstream.DATA_DIR))


@pytest.fixture(scope="session")
def splits():
  return _splits
