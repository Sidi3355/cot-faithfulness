"""Imports the upstream corr_faith code (third_party/) for equivalence tests.

Upstream imports torch, transformers and several API clients at module level,
even in modules whose prompt and intervention logic doesn't use them. Those
that aren't installed are replaced with inert stand-ins, so the pure-Python
parts can be imported. Installed ones (e.g. transformers, for tokenizers) are
imported for real first, so a stand-in never shadows them.
"""

import functools
import importlib
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM_SRC = ROOT / "third_party" / "corr_faith" / "src"
DATA_DIR = ROOT / "data"

_STUBBED = (
    "torch", "transformers", "accelerate", "anthropic", "openai",
    "google.genai", "google.auth", "google.auth.exceptions", "google.cloud",
    "google.cloud.storage", "google.cloud.storage.transfer_manager",
)


class _StubModule(types.ModuleType):
  """Module whose every attribute is a fresh dummy class."""

  def __getattr__(self, name: str):
    if name.startswith("__"):
      raise AttributeError(name)
    cls = type(name, (), {})
    setattr(self, name, cls)
    return cls


@functools.cache
def _install_stubs() -> None:
  # Import every installed module before stubbing anything, so that e.g.
  # transformers doesn't mistake a torch stand-in for an installed torch.
  missing = []
  for name in _STUBBED:
    try:
      importlib.import_module(name)
    except ImportError:
      missing.append(name)
  for name in missing:
    if name in sys.modules:
      continue
    stub = _StubModule(name)
    stub.__path__ = []  # So that submodules of the stub can be imported.
    sys.modules[name] = stub
    parent_name, _, child = name.rpartition(".")
    if parent_name:
      if parent_name not in sys.modules:
        try:
          importlib.import_module(parent_name)
        except ImportError:
          parent = _StubModule(parent_name)
          parent.__path__ = []
          sys.modules[parent_name] = parent
      setattr(sys.modules[parent_name], child, stub)
  if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))


def available() -> bool:
  return (UPSTREAM_SRC / "corr_faith").is_dir()


def module(name: str) -> types.ModuleType:
  """Imports `corr_faith.<name>` with stand-ins for the heavy dependencies."""
  _install_stubs()
  return importlib.import_module(f"corr_faith.{name}")


class V4ChatTemplate:
  """Tokenizer proxy giving upstream the transformers-4 behaviour it was written
  for: apply_chat_template(..., return_tensors="pt") returns the id tensor (v5
  returns a dict unless return_dict=False). Everything else is delegated."""

  def __init__(self, tokenizer):
    self._tokenizer = tokenizer

  def apply_chat_template(self, *args, **kwargs):
    kwargs.setdefault("return_dict", False)
    return self._tokenizer.apply_chat_template(*args, **kwargs)

  def __call__(self, *args, **kwargs):  # Special methods bypass __getattr__.
    return self._tokenizer(*args, **kwargs)

  def __getattr__(self, name):
    return getattr(self._tokenizer, name)


def pipeline_for(model, tokenizer):
  """The parts of a transformers text-generation pipeline that upstream uses."""
  import torch
  return types.SimpleNamespace(model=model, tokenizer=V4ChatTemplate(tokenizer), device=torch.device("cpu"))
