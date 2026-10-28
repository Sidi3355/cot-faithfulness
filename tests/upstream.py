"""Imports the upstream corr_faith code (third_party/) for equivalence tests.

Upstream imports torch, transformers and several API clients at module level,
even in modules whose prompt and intervention logic doesn't use them. Those
modules are replaced with inert stand-ins, so the pure-Python parts can be
imported without installing them.
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
  for name in _STUBBED:
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
