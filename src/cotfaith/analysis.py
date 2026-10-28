"""Pre-registered analyses for trace vs explanation (PLAN.md section 7.4).

(a) Primary: accuracy-matched curve test. Fit phi-CCT against base accuracy on
the thinking-off points (models x lengths), then average the thinking-on points'
residuals from that fit, using only thinking-on points whose accuracy lies
within the thinking-off range. Bootstrap by recomputing every point and
refitting in each resample.

(b) Secondary: logistic regression of E on I x mode with covariates. The plan
specifies lme4 glmer with example and intervention intercepts; this module
implements the GEE fallback (clustered by example).
"""

import dataclasses

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from cotfaith import bootstrap, metrics

MIN_POINTS_IN_RANGE = 3
# Per-example statistics for one point (model x verbosity setting):
# the four confusion counts, then (n base runs, n correct base runs).
POINT_STATS = metrics.CONFUSION + ("n", "n_correct")


def _ols_fit(x: np.ndarray, y: np.ndarray):
  """Row-wise OLS of y on x, ignoring NaNs. Returns (intercept, slope, lo, hi)."""
  ok = ~(np.isnan(x) | np.isnan(y))
  n = ok.sum(axis=-1)
  xs, ys = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
  with np.errstate(invalid="ignore", divide="ignore"):
    xm, ym = xs.sum(-1) / n, ys.sum(-1) / n
    dx = np.where(ok, x - xm[..., None], 0.0)
    dy = np.where(ok, y - ym[..., None], 0.0)
    sxx = (dx * dx).sum(-1)
    slope = np.where(sxx > 0, (dx * dy).sum(-1) / np.where(sxx > 0, sxx, 1), np.nan)
  intercept = ym - slope * xm
  lo = np.where(ok, x, np.inf).min(-1)
  hi = np.where(ok, x, -np.inf).max(-1)
  return intercept, slope, lo, hi


def _mean_in_range(residual, on_acc, lo, hi):
  in_range = ~np.isnan(residual) & (on_acc >= lo[..., None]) & (on_acc <= hi[..., None])
  count = in_range.sum(-1)
  total = np.where(in_range, residual, 0.0).sum(-1)
  with np.errstate(invalid="ignore", divide="ignore"):
    return np.where(count >= MIN_POINTS_IN_RANGE, total / count, np.nan)


def curve_delta(off_acc, off_phi, on_acc, on_phi) -> np.ndarray:
  """Mean residual of thinking-on points from the thinking-off OLS line.

  Inputs are (..., n_off) and (..., n_on); leading axes (e.g. resamples) are
  vectorised. NaN if fewer than MIN_POINTS_IN_RANGE on-points are in range.
  """
  off_acc, off_phi, on_acc, on_phi = (np.asarray(a, dtype=float) for a in (off_acc, off_phi, on_acc, on_phi))
  intercept, slope, lo, hi = _ols_fit(off_acc, off_phi)
  residual = on_phi - (intercept[..., None] + slope[..., None] * on_acc)
  return _mean_in_range(residual, on_acc, lo, hi)


def curve_delta_isotonic(off_acc, off_phi, on_acc, on_phi) -> np.ndarray:
  """Robustness variant: non-decreasing isotonic fit instead of OLS."""
  off_acc, off_phi, on_acc, on_phi = (
      np.atleast_2d(np.asarray(a, dtype=float)) for a in (off_acc, off_phi, on_acc, on_phi)
  )
  out = np.full(off_acc.shape[0], np.nan)
  for r in range(off_acc.shape[0]):
    ok = ~(np.isnan(off_acc[r]) | np.isnan(off_phi[r]))
    if ok.sum() < 2:
      continue
    fit = IsotonicRegression(increasing=True, out_of_bounds="nan")
    fit.fit(off_acc[r, ok], off_phi[r, ok])
    on_ok = ~np.isnan(on_acc[r])
    pred = np.full(on_acc.shape[1], np.nan)
    pred[on_ok] = fit.predict(on_acc[r, on_ok])
    lo, hi = off_acc[r, ok].min(), off_acc[r, ok].max()
    out[r] = _mean_in_range(on_phi[r] - pred, on_acc[r], np.array(lo), np.array(hi))
  return out


def _points(sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """(accuracy, phi) per point from summed POINT_STATS (last axis)."""
  return metrics.accuracy(sums[..., 4:6]), metrics.phi(sums[..., :4])


@dataclasses.dataclass(frozen=True)
class TestResult:
  estimate: float
  ci: tuple[float, float]
  p_value: float
  undefined_fraction: float


def curve_test(off: np.ndarray, on: np.ndarray, boot: bootstrap.ClusterBootstrap, isotonic: bool = False) -> TestResult:
  """Curve test from per-example POINT_STATS arrays.

  `off` is (n_examples, n_off_points, 6) and `on` is (n_examples, n_on_points, 6),
  aligned to `boot.example_ids`.
  """
  n_off = off.shape[1]
  stats = np.concatenate([off, on], axis=1)
  delta = curve_delta_isotonic if isotonic else curve_delta

  def metric(sums):
    acc, phi = _points(sums)
    return delta(acc[..., :n_off], phi[..., :n_off], acc[..., n_off:], phi[..., n_off:])

  estimate = float(np.ravel(metric(stats.sum(axis=0)[None]))[0])
  dist = boot.distribution(stats, metric)
  return TestResult(
      estimate=estimate,
      ci=bootstrap.percentile_ci(dist),
      p_value=bootstrap.paired_p_value(dist),
      undefined_fraction=bootstrap.undefined_fraction(dist),
  )


@dataclasses.dataclass(frozen=True)
class InteractionResult:
  coef: float
  ci: tuple[float, float]
  p_value: float
  n_rows: int
  n_clusters: int


INTERACTION_FORMULA = "E ~ I * mode + log10_params + base_correct"


def interaction_gee(df: pd.DataFrame) -> InteractionResult:
  """GEE logistic regression; returns the I x mode coefficient.

  Expects columns E, I (0/1), mode (0 = thinking off, 1 = thinking on),
  log10_params, base_correct (0/1) and example_id. Rows with any missing
  value are dropped.
  """
  import statsmodels.api as sm
  import statsmodels.formula.api as smf

  cols = ["E", "I", "mode", "log10_params", "base_correct", "example_id"]
  data = df[cols].dropna()
  model = smf.gee(
      INTERACTION_FORMULA, groups="example_id", data=data,
      family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
  )
  fit = model.fit()
  lo, hi = fit.conf_int().loc["I:mode"]
  return InteractionResult(
      coef=float(fit.params["I:mode"]),
      ci=(float(lo), float(hi)),
      p_value=float(fit.pvalues["I:mode"]),
      n_rows=len(data),
      n_clusters=int(data["example_id"].nunique()),
  )
