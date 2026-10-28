import numpy as np
import pandas as pd
import pytest

from cotfaith import analysis, bootstrap

OFF_ACC = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
OFF_PHI = 0.5 * OFF_ACC


def test_curve_delta_recovers_a_constant_shift():
  on_acc = np.array([0.55, 0.65, 0.75, 0.85])
  assert analysis.curve_delta(OFF_ACC, OFF_PHI, on_acc, 0.5 * on_acc + 0.1) == pytest.approx(0.1)


def test_curve_delta_ignores_points_outside_the_range():
  on_acc = np.array([0.55, 0.65, 0.75, 0.95, 0.30])
  on_phi = np.array([0.5 * 0.55, 0.5 * 0.65, 0.5 * 0.75, 9.0, -9.0])
  assert analysis.curve_delta(OFF_ACC, OFF_PHI, on_acc, on_phi) == pytest.approx(0.0)


def test_curve_delta_needs_three_points_in_range():
  on_acc = np.array([0.55, 0.65, 0.99])
  assert np.isnan(analysis.curve_delta(OFF_ACC, OFF_PHI, on_acc, 0.5 * on_acc))


def test_curve_delta_skips_undefined_points_and_vectorises():
  off_phi = np.stack([OFF_PHI, np.where(OFF_ACC == 0.7, np.nan, OFF_PHI)])
  off_acc = np.stack([OFF_ACC, OFF_ACC])
  on_acc = np.tile([0.55, 0.65, 0.75, 0.85], (2, 1))
  on_phi = 0.5 * on_acc + np.array([[0.1], [0.2]])
  np.testing.assert_allclose(analysis.curve_delta(off_acc, off_phi, on_acc, on_phi), [0.1, 0.2])


def test_isotonic_variant_agrees_on_monotone_data():
  on_acc = np.array([0.6, 0.7, 0.8])
  assert analysis.curve_delta_isotonic(OFF_ACC, OFF_PHI, on_acc, 0.5 * on_acc + 0.05)[0] == pytest.approx(0.05)


def _point_stats(rng, n_examples, accuracy, tpr, fpr, flip_rate=0.3, rows=6):
  """Per-example POINT_STATS for one point with given true rates."""
  i = rng.random((n_examples, rows)) < flip_rate
  e = np.where(i, rng.random(i.shape) < tpr, rng.random(i.shape) < fpr)
  correct = rng.random((n_examples, 2)) < accuracy
  return np.stack([
      (e & i).sum(1), (e & ~i).sum(1), (~e & i).sum(1), (~e & ~i).sum(1),
      np.full(n_examples, 2), correct.sum(1),
  ], axis=1).astype(float)


def _curve_data(rng, n_examples, on_tpr_shift):
  accs = [0.55, 0.65, 0.75, 0.85, 0.9]
  # Off points: TPR (and so phi) rises with accuracy; on points add a shift.
  off = np.stack([_point_stats(rng, n_examples, a, tpr=a - 0.2, fpr=0.2) for a in accs], axis=1)
  on = np.stack([
      _point_stats(rng, n_examples, a, tpr=a - 0.2 + on_tpr_shift, fpr=0.2) for a in accs[:4]
  ], axis=1)
  return off, on


def test_curve_test_detects_a_real_shift():
  rng = np.random.default_rng(0)
  off, on = _curve_data(rng, 1500, on_tpr_shift=0.15)
  boot = bootstrap.ClusterBootstrap.for_examples(range(1500), n_resamples=400)
  result = analysis.curve_test(off, on, boot)
  assert result.estimate > 0
  assert result.ci[0] > 0
  assert result.p_value < 0.01


def test_curve_test_null_case():
  rng = np.random.default_rng(1)
  off, on = _curve_data(rng, 1500, on_tpr_shift=0.0)
  boot = bootstrap.ClusterBootstrap.for_examples(range(1500), n_resamples=400)
  result = analysis.curve_test(off, on, boot)
  assert result.ci[0] < 0 < result.ci[1]
  assert result.p_value > 0.05


def test_curve_test_isotonic_runs():
  rng = np.random.default_rng(2)
  off, on = _curve_data(rng, 300, on_tpr_shift=0.1)
  boot = bootstrap.ClusterBootstrap.for_examples(range(300), n_resamples=50)
  result = analysis.curve_test(off, on, boot, isotonic=True)
  assert np.isfinite(result.estimate)


def test_interaction_gee_recovers_coefficient():
  rng = np.random.default_rng(3)
  n_examples, rows = 3000, 4
  example = np.repeat(np.arange(n_examples), rows)
  mode = rng.integers(0, 2, size=len(example))
  flip = rng.integers(0, 2, size=len(example))
  base_correct = rng.integers(0, 2, size=len(example))
  log10_params = rng.choice(np.log10([0.6e9, 1.7e9, 4e9, 8e9, 14e9]), size=len(example))
  example_effect = rng.normal(0, 0.5, n_examples)[example]
  logit = -1.0 + 0.8 * flip + 0.5 * mode + 1.0 * flip * mode + 0.1 * log10_params + example_effect
  e = rng.random(len(example)) < 1 / (1 + np.exp(-logit))
  df = pd.DataFrame(dict(
      E=e.astype(float), I=flip, mode=mode, log10_params=log10_params,
      base_correct=base_correct, example_id=example,
  ))
  result = analysis.interaction_gee(df)
  # GEE estimates the population-averaged effect: the conditional 1.0,
  # attenuated to about 0.96 by the example-level variation (sd 0.5).
  assert result.ci[0] < 0.96 < result.ci[1]
  assert result.coef == pytest.approx(0.96, abs=0.1)
  assert result.p_value < 1e-6
  assert result.n_clusters == n_examples
