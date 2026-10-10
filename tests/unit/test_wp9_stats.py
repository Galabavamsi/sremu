"""``sremu.analysis.stats`` against SciPy references (WP9 acceptance: stats unit tests).

* batch means: the t interval of ``scipy.stats.t.interval`` on the batch means;
* paired bootstrap: ``scipy.stats.bootstrap(paired=True)`` with an identical keyed
  generator (same resampled indices), percentile and BCa;
* Clopper-Pearson: ``scipy.stats.binomtest(...).proportion_ci(method="exact")``;
* Spearman: rho and p-value of ``scipy.stats.spearmanr``; the Fieller interval from its
  closed form;
* error metrics: hand-computed values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from sremu import rng as sremu_rng
from sremu.analysis.stats import (
    batch_means,
    clopper_pearson,
    error_metrics,
    mean_difference,
    paired_bootstrap,
    rmse_difference,
    spearman,
)

pytestmark = pytest.mark.unit


def _gen(name: str, seed: int = 0) -> np.random.Generator:
    return sremu_rng.generator(seed, "trial", f"test_stats/{name}")


def test_batch_means_matches_scipy_t_interval():
    x = _gen("bm").standard_normal(1003) + 2.0
    res = batch_means(x, 20, confidence=0.95)
    assert res.batch_size == 50 and res.n_dropped == 3 and res.n_batches == 20
    bm = x[:1000].reshape(20, 50).mean(axis=1)
    np.testing.assert_allclose(res.batch_values, bm, rtol=1e-15)
    lo, hi = stats.t.interval(0.95, 19, loc=bm.mean(), scale=stats.sem(bm))
    assert res.mean == pytest.approx(float(np.mean(x[:1000])), rel=1e-14)
    assert res.ci_low == pytest.approx(lo, rel=1e-13) and res.ci_high == pytest.approx(
        hi, rel=1e-13
    )
    assert res.se == pytest.approx(stats.sem(bm), rel=1e-13)
    with pytest.raises(ValueError):
        batch_means(x[:10], 20)
    with pytest.raises(ValueError):
        batch_means(x, 1)


@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_paired_bootstrap_matches_scipy_with_the_same_generator(method):
    g = _gen("pb-data")
    obs = g.standard_normal(12)
    e0 = obs + g.normal(0.5, 1.0, 12)  # errors of a worse model
    e2 = obs * 0.3 + g.normal(0.0, 0.4, 12)
    mine = paired_bootstrap(
        [e0, e2], rmse_difference, rng=_gen("pb", 1), n_resamples=5000, method=method
    )
    ref = stats.bootstrap(
        (e0, e2),
        rmse_difference,
        paired=True,
        vectorized=True,
        n_resamples=5000,
        method="BCa" if method == "bca" else "percentile",
        rng=_gen("pb", 1),
    )
    np.testing.assert_allclose(mine.distribution, ref.bootstrap_distribution, rtol=1e-13)
    assert mine.ci_low == pytest.approx(ref.confidence_interval.low, rel=1e-12)
    assert mine.ci_high == pytest.approx(ref.confidence_interval.high, rel=1e-12)
    assert mine.standard_error == pytest.approx(ref.standard_error, rel=1e-12)
    assert mine.estimate == pytest.approx(float(rmse_difference(e0, e2)), rel=1e-15)


def test_paired_bootstrap_mean_difference_and_zero_exclusion():
    g = _gen("pb2")
    a = g.normal(1.0, 0.2, 40)
    b = a - 0.5 + g.normal(0.0, 0.05, 40)
    res = paired_bootstrap([a, b], mean_difference, rng=_gen("pb2r"), n_resamples=2000)
    assert res.excludes_zero and res.ci_low < 0.5 < res.ci_high
    with pytest.raises(ValueError):
        paired_bootstrap([a, b[:-1]], mean_difference, rng=_gen("x"))


@pytest.mark.parametrize(("k", "n"), [(0, 20), (1, 20), (7, 20), (20, 20), (13, 2000), (0, 1)])
@pytest.mark.parametrize("conf", [0.95, 0.99])
def test_clopper_pearson_matches_scipy_exact(k, n, conf):
    lo, hi = clopper_pearson(k, n, conf)
    ref = stats.binomtest(k, n).proportion_ci(confidence_level=conf, method="exact")
    assert lo == pytest.approx(ref.low, rel=1e-12, abs=1e-15)
    assert hi == pytest.approx(ref.high, rel=1e-12, abs=1e-15)


def test_clopper_pearson_rejects_bad_counts():
    for k, n in ((-1, 5), (6, 5), (0, 0)):
        with pytest.raises(ValueError):
            clopper_pearson(k, n)


def test_spearman_matches_scipy_and_fieller_closed_form():
    g = _gen("sp")
    x = g.standard_normal(30)
    y = 0.6 * x + 0.8 * g.standard_normal(30)
    y[3] = y[7]  # a tie
    res = spearman(x, y)
    ref = stats.spearmanr(x, y)
    assert res.rho == pytest.approx(ref.statistic, rel=1e-13)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    z, h = math.atanh(res.rho), stats.norm.ppf(0.975) * math.sqrt(1.06 / 27)
    assert res.ci_low == pytest.approx(math.tanh(z - h), rel=1e-14)
    assert res.ci_high == pytest.approx(math.tanh(z + h), rel=1e-14)
    assert res.excludes_zero
    boot = spearman(x, y, ci_method="bootstrap", rng=_gen("spb"), n_resamples=2000)
    ref_b = stats.bootstrap(
        (x, y),
        lambda a, b, axis: (
            stats.spearmanr(a, b, axis=axis).statistic
            if a.ndim == 1
            else np.array([stats.spearmanr(a[i], b[i]).statistic for i in range(a.shape[0])])
        ),
        paired=True,
        vectorized=True,
        n_resamples=2000,
        method="percentile",
        rng=_gen("spb"),
    )
    assert boot.ci_low == pytest.approx(ref_b.confidence_interval.low, rel=1e-12)
    assert boot.ci_high == pytest.approx(ref_b.confidence_interval.high, rel=1e-12)
    with pytest.raises(ValueError):
        spearman(x, y, ci_method="bootstrap")  # needs a keyed generator


def test_spearman_perfect_monotone_and_constant_input():
    x = np.arange(10.0)
    res = spearman(x, x**3)
    assert res.rho == 1.0 and res.p_value == 0.0
    with pytest.raises(ValueError):
        spearman(x, np.ones(10))


def test_error_metrics_hand_computed():
    est = np.array([1.0, 2.0, 4.0, -1.0])
    ref = np.array([0.0, 2.5, 3.0, 0.0])
    m = error_metrics(est, ref)  # e = [1, -0.5, 1, -1]
    assert m.n == 4
    assert m.bias == pytest.approx(0.125, rel=1e-15)
    assert m.rmse == pytest.approx(math.sqrt(3.25 / 4), rel=1e-15)
    assert m.mae == pytest.approx(0.875, rel=1e-15)
    assert m.max_abs == 1.0
    assert m.sd == pytest.approx(float(np.std([1, -0.5, 1, -1], ddof=1)), rel=1e-15)
    assert set(m.as_dict()) == {"n", "bias", "sd", "rmse", "mae", "max_abs"}
    assert math.isnan(error_metrics([1.0], [0.0]).sd)
    with pytest.raises(ValueError):
        error_metrics([1.0, np.nan], [0.0, 0.0])
