import math

import numpy as np
import pandas as pd
import pytest

from cotfaith import bootstrap, flips, metrics


def counts(a, b, c, d):
  return np.array([a, b, c, d], dtype=float)


def test_confusion_counts_drop_undefined_rows():
  e = [1, 1, 0, 0, 1, np.nan]
  i = [1, 0, 1, 0, np.nan, 1]
  np.testing.assert_array_equal(metrics.confusion_counts(e, i), [1, 1, 1, 1])


def test_tpr_fpr_ct():
  c = counts(30, 10, 20, 40)
  assert metrics.tpr(c) == pytest.approx(30 / 50)
  assert metrics.ct(c) == metrics.tpr(c)
  assert metrics.fpr(c) == pytest.approx(10 / 50)
  assert metrics.flip_rate(c) == pytest.approx(0.5)
  assert metrics.mention_rate(c) == pytest.approx(0.4)


def test_phi_equals_pearson_correlation():
  rng = np.random.default_rng(0)
  i = rng.random(500) < 0.3
  e = np.where(i, rng.random(500) < 0.7, rng.random(500) < 0.2)
  c = metrics.confusion_counts(e, i)
  assert metrics.phi(c) == pytest.approx(np.corrcoef(e, i)[0, 1])


def test_phi_known_value():
  # (30*40 - 10*20) / sqrt(40*60*50*50)
  assert metrics.phi(counts(30, 10, 20, 40)) == pytest.approx(1000 / math.sqrt(40 * 60 * 50 * 50))


def test_phi_undefined_when_constant():
  assert np.isnan(metrics.phi(counts(10, 5, 0, 0)))  # E always 1
  assert np.isnan(metrics.phi(counts(0, 7, 0, 3)))  # I always 0


def test_metrics_vectorise_over_resamples():
  batch = np.stack([counts(30, 10, 20, 40), counts(5, 5, 5, 5), counts(1, 0, 0, 0)])
  np.testing.assert_allclose(metrics.tpr(batch), [0.6, 0.5, 1.0])
  assert np.isnan(metrics.phi(batch)[2])


def test_cct_equals_pearson_with_tvd():
  rng = np.random.default_rng(1)
  t = rng.random(300)
  e = (rng.random(300) < t).astype(float)
  assert metrics.cct(metrics.cct_sums(e, t)) == pytest.approx(np.corrcoef(e, t)[0, 1])


def test_total_variation_distance():
  assert metrics.total_variation_distance([0.7, 0.2, 0.1], [0.1, 0.2, 0.7]) == pytest.approx(0.6)


@pytest.mark.parametrize("fprs,tprs,expected", [
    ([0.2], [0.6], 0.2 * 0.6 / 2 + 0.8 * (0.6 + 1) / 2),  # One point.
    ([0.0], [1.0], 1.0),  # Perfect explainer.
    ([0.3, 0.6], [0.3, 0.6], 0.5),  # Points on the diagonal.
    ([0.5], [0.2], 0.5),  # Below the diagonal: the hull ignores it.
    # (0.2, 0.3) is dominated by (0.2, 0.6); (0.5, 0.85) is on the hull.
    ([0.2, 0.2, 0.5], [0.3, 0.6, 0.85], 0.2 * 0.6 / 2 + 0.3 * 1.45 / 2 + 0.5 * 1.85 / 2),
    ([0.1, np.nan], [0.5, 0.9], 0.1 * 0.5 / 2 + 0.9 * 1.5 / 2),  # Undefined setting skipped.
])
def test_f_auroc(fprs, tprs, expected):
  assert metrics.f_auroc(fprs, tprs) == pytest.approx(expected)


def test_f_auroc_drops_dominated_points():
  # (0.4, 0.5) lies under the segment from (0.2, 0.6) to (1, 1).
  assert metrics.f_auroc([0.2, 0.4], [0.6, 0.5]) == pytest.approx(metrics.f_auroc([0.2], [0.6]))


def test_f_auroc_undefined_without_points():
  assert np.isnan(metrics.f_auroc([np.nan], [np.nan]))


def test_dummy_copier():
  i = (np.random.default_rng(2).random(1000) < 0.2).astype(float)
  c = metrics.confusion_counts(flips.copier_mentions(len(i)), i)
  assert metrics.ct(c) == 1.0
  assert metrics.fpr(c) == 1.0
  assert np.isnan(metrics.phi(c))
  assert metrics.f_auroc([metrics.fpr(c)], [metrics.tpr(c)]) == pytest.approx(0.5)


def test_dummy_random_explainer():
  i = (np.random.default_rng(3).random(200_000) < 0.2).astype(float)
  c = metrics.confusion_counts(flips.random_mentions(len(i), q=0.3, seed=4), i)
  assert metrics.phi(c) == pytest.approx(0.0, abs=0.01)
  assert metrics.tpr(c) == pytest.approx(0.3, abs=0.01)
  assert metrics.f_auroc([metrics.fpr(c)], [metrics.tpr(c)]) == pytest.approx(0.5, abs=0.01)


def test_greedy_flip():
  assert flips.greedy_flip("a", "b") == 1.0
  assert flips.greedy_flip("a", "a") == 0.0
  assert np.isnan(flips.greedy_flip(None, "a"))


def test_modal_label_majority_and_tie_breaks():
  assert flips.modal_label(["a", "b", "a", None, "c"]) == "a"
  probs = [{"a": 0.4, "b": 0.6}, {"a": 0.45, "b": 0.55}]
  assert flips.modal_label(["a", "b", "a", "b"], probs) == "b"  # Higher mean prob.
  assert flips.modal_label(["b", "a"], [{"a": 0.5, "b": 0.5}]) == "a"  # Then alphabetical.
  assert flips.modal_label([None, None]) is None


def test_recode_missing():
  np.testing.assert_array_equal(flips.recode_missing([1, np.nan, 0], 1.0), [1, 1, 0])


def _rows(n_examples=200, rows_per_example=5, seed=0):
  rng = np.random.default_rng(seed)
  ex = np.repeat(np.arange(n_examples), rows_per_example)
  i = rng.random(len(ex)) < 0.3
  e = np.where(i, rng.random(len(ex)) < 0.6, rng.random(len(ex)) < 0.2)
  return pd.DataFrame({"example_id": ex, "E": e.astype(float), "I": i.astype(float)})


def test_per_example_confusion_sums_to_total():
  df = _rows()
  per_ex = flips.per_example_confusion(df, "E", "I")
  np.testing.assert_array_equal(per_ex.sum().to_numpy(), metrics.confusion_counts(df.E, df.I))


def test_bootstrap_is_deterministic_and_paired():
  df = _rows()
  boot = bootstrap.ClusterBootstrap.for_examples(df.example_id.unique(), n_resamples=300)
  stats = boot.align(flips.per_example_confusion(df, "E", "I"))
  a = boot.distribution(stats, metrics.phi)
  b = boot.distribution(stats, metrics.phi)
  np.testing.assert_array_equal(a, b)
  assert bootstrap.paired_p_value(a - b) == 1.0


def test_bootstrap_ci_covers_point_estimate_and_matches_iid_resampling():
  df = _rows(n_examples=400)
  ids = df.example_id.unique()
  boot = bootstrap.ClusterBootstrap.for_examples(ids, n_resamples=2000, seed=5)
  stats = boot.align(flips.per_example_confusion(df, "E", "I"))
  dist = boot.distribution(stats, metrics.phi)
  point = metrics.phi(stats.sum(axis=0))
  lo, hi = bootstrap.percentile_ci(dist)
  assert lo < point < hi
  # Direct resampling of examples gives the same spread.
  rng = np.random.default_rng(6)
  by_ex = {k: g for k, g in df.groupby("example_id")}
  direct = []
  for _ in range(500):
    sample = pd.concat([by_ex[k] for k in rng.choice(ids, size=len(ids))])
    direct.append(metrics.phi(metrics.confusion_counts(sample.E, sample.I)))
  assert np.std(dist) == pytest.approx(np.std(direct), rel=0.15)


def test_bootstrap_align_fills_missing_examples_with_zeros():
  boot = bootstrap.ClusterBootstrap.for_examples([3, 1, 2], n_resamples=10)
  stats = pd.DataFrame({"x": [5.0]}, index=[2])
  np.testing.assert_array_equal(boot.align(stats), [[0.0], [5.0], [0.0]])
  with pytest.raises(ValueError):
    boot.align(pd.DataFrame({"x": [1.0]}, index=[9]))


def test_bootstrap_detects_a_real_difference():
  df = _rows(n_examples=400, seed=7)
  noisy = df.copy()
  noisy["E"] = (np.random.default_rng(8).random(len(df)) < 0.3).astype(float)
  boot = bootstrap.ClusterBootstrap.for_examples(df.example_id.unique(), n_resamples=1000)
  a = boot.distribution(boot.align(flips.per_example_confusion(df, "E", "I")), metrics.phi)
  b = boot.distribution(boot.align(flips.per_example_confusion(noisy, "E", "I")), metrics.phi)
  assert bootstrap.paired_p_value(a - b) < 0.01


def test_settings_axis_for_f_auroc():
  """Per-example stats with a settings axis: (examples, settings, 4)."""
  frames = [_rows(seed=s) for s in range(3)]
  boot = bootstrap.ClusterBootstrap.for_examples(frames[0].example_id.unique(), n_resamples=50)
  stats = np.stack([boot.align(flips.per_example_confusion(f, "E", "I")) for f in frames], axis=1)
  dist = boot.distribution(stats, lambda s: metrics.f_auroc_batch(metrics.fpr(s), metrics.tpr(s)))
  assert dist.shape == (50,)
  assert np.all((dist >= 0.5) & (dist <= 1.0))


def test_holm():
  np.testing.assert_allclose(bootstrap.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
  np.testing.assert_allclose(bootstrap.holm([0.5, 0.9]), [1.0, 1.0])
