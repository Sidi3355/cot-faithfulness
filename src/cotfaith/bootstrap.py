"""Cluster bootstrap over test examples, with paired comparisons and Holm.

Each resample draws the dataset's examples with replacement and carries every
row of a drawn example (its interventions, N1/N2 twins and samples). Instead
of materialising resampled rows, each resample is a vector of per-example
multiplicities; a metric's resampled value is computed from multiplicity-
weighted sums of per-example sufficient statistics.

The multiplicities depend only on (n_examples, seed), so every condition
evaluated with the same ClusterBootstrap sees identical resamples. That makes
differences between conditions paired.
"""

import dataclasses
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd

N_RESAMPLES = 10_000


@dataclasses.dataclass(frozen=True)
class ClusterBootstrap:
  """Resamples over a fixed, ordered universe of example ids."""

  example_ids: tuple
  n_resamples: int = N_RESAMPLES
  seed: int = 0
  chunk_size: int = 500

  @classmethod
  def for_examples(cls, example_ids: Sequence, **kwargs) -> "ClusterBootstrap":
    return cls(tuple(sorted(example_ids)), **kwargs)

  def weight_chunks(self) -> Iterator[np.ndarray]:
    """(chunk, n_examples) multiplicity matrices; deterministic for a seed."""
    n = len(self.example_ids)
    rng = np.random.default_rng(self.seed)
    pvals = np.full(n, 1.0 / n)
    done = 0
    while done < self.n_resamples:
      size = min(self.chunk_size, self.n_resamples - done)
      yield rng.multinomial(n, pvals, size=size).astype(np.float64)
      done += size

  def align(self, stats: pd.DataFrame) -> np.ndarray:
    """Per-example statistics indexed by example id -> (n_examples, k) array.

    Examples without rows (e.g. every row failed to parse) contribute zeros.
    """
    extra = set(stats.index) - set(self.example_ids)
    if extra:
      raise ValueError(f"{len(extra)} example ids are outside the bootstrap universe")
    return stats.reindex(list(self.example_ids), fill_value=0).to_numpy(dtype=float)

  def distribution(
      self, per_example: np.ndarray, metric: Callable[[np.ndarray], np.ndarray]
  ) -> np.ndarray:
    """Resampled metric values, shape (n_resamples,).

    `per_example` is (n_examples, k) (or (n_examples, s, k) for s settings);
    `metric` maps summed statistics of shape (n_resamples, ...) to values.
    """
    per_example = np.asarray(per_example, dtype=float)
    flat = per_example.reshape(per_example.shape[0], -1)
    out = []
    for w in self.weight_chunks():
      sums = (w @ flat).reshape((w.shape[0],) + per_example.shape[1:])
      out.append(np.asarray(metric(sums)))
    return np.concatenate(out)


def percentile_ci(dist: np.ndarray, level: float = 0.95) -> tuple[float, float]:
  """Percentile interval over resamples where the metric is defined."""
  dist = np.asarray(dist, dtype=float)
  dist = dist[~np.isnan(dist)]
  if dist.size == 0:
    return float("nan"), float("nan")
  alpha = (1 - level) / 2
  lo, hi = np.quantile(dist, [alpha, 1 - alpha])
  return float(lo), float(hi)


def undefined_fraction(dist: np.ndarray) -> float:
  return float(np.mean(np.isnan(np.asarray(dist, dtype=float))))


def paired_p_value(diff_dist: np.ndarray) -> float:
  """Two-sided bootstrap p-value for H0: difference = 0."""
  d = np.asarray(diff_dist, dtype=float)
  d = d[~np.isnan(d)]
  if d.size == 0:
    return float("nan")
  tail = min(np.sum(d <= 0), np.sum(d >= 0))
  return float(min(1.0, 2 * (tail + 1) / (d.size + 1)))


def holm(p_values: Sequence[float]) -> np.ndarray:
  """Holm-adjusted p-values (family-wise error control), in input order."""
  p = np.asarray(p_values, dtype=float)
  order = np.argsort(p)
  m = len(p)
  adjusted = np.empty(m)
  running = 0.0
  for rank, idx in enumerate(order):
    running = max(running, (m - rank) * p[idx])
    adjusted[idx] = min(1.0, running)
  return adjusted
