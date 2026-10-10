"""Statistics for the studies (DESIGN.md sections 10.9, 11; WP9).

* :func:`batch_means`: non-overlapping batch means with a Student-t confidence interval
  (E-A "20 non-overlapping batches", W3, the WCL letter's reference [7]).
* :func:`paired_bootstrap`: paired bootstrap of any statistic of paired samples (V-LAB C2:
  "paired-bootstrap 95% CI of Delta RMSE, 10^4 resamples over test cells"), percentile or
  BCa interval. The resampling is the one of ``scipy.stats.bootstrap(paired=True)``: with
  the same generator both draw the same indices.
* :func:`clopper_pearson`: exact binomial interval (ST-04, E-B2 PER, E-F BER).
* :func:`spearman`: Spearman rank correlation with p-value and confidence interval (Fieller
  et al. 1957 standard error on Fisher's z, or a paired bootstrap).
* :func:`error_metrics`: bias, SD, RMSE, MAE and max |e| of ``estimate - reference``.

Randomness only through a caller-supplied keyed generator (``sremu.rng.generator``, e.g.
purpose ``trial``); there is no default generator (section 6.1).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import special, stats

__all__ = [
    "BatchMeansResult",
    "BootstrapResult",
    "ErrorMetrics",
    "SpearmanResult",
    "batch_means",
    "clopper_pearson",
    "error_metrics",
    "mean_difference",
    "paired_bootstrap",
    "rankdata",
    "rmse",
    "rmse_difference",
    "spearman",
]


def _check_conf(confidence: float) -> float:
    c = float(confidence)
    if not 0.0 < c < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    return c


def _vec(x: object, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


# =========================================================================== batch means


@dataclass(frozen=True)
class BatchMeansResult:
    """Non-overlapping batch means: ``mean`` of the batch means (= mean of the samples
    used), its standard error ``se`` (SD of the batch means / sqrt(k)), the two-sided
    Student-t interval ``(ci_low, ci_high)`` with ``half_width`` and ``k - 1`` degrees of
    freedom, ``n_batches`` k, ``batch_size`` b and ``n_dropped`` trailing samples."""

    mean: float
    se: float
    half_width: float
    ci_low: float
    ci_high: float
    n_batches: int
    batch_size: int
    n_dropped: int
    confidence: float
    batch_values: np.ndarray


def batch_means(
    x: Sequence[float] | np.ndarray, n_batches: int = 20, *, confidence: float = 0.95
) -> BatchMeansResult:
    """Non-overlapping batch means of a (correlated) series ``x``.

    The first ``k b`` samples form ``k = n_batches`` consecutive batches of
    ``b = len(x) // k`` samples; the ``len(x) - k b`` trailing samples are dropped
    (reported). The interval is ``mean +- t_{1-alpha/2, k-1} s_k/sqrt(k)``.
    """
    a = _vec(x, "x")
    k = int(n_batches)
    if k < 2:
        raise ValueError("n_batches must be >= 2")
    b = a.size // k
    if b < 1:
        raise ValueError(f"need at least {k} samples for {k} batches, got {a.size}")
    conf = _check_conf(confidence)
    used = a[: k * b].reshape(k, b)
    bm = used.mean(axis=1)
    mean = float(bm.mean())
    se = float(bm.std(ddof=1) / math.sqrt(k))
    hw = float(stats.t.ppf(0.5 + conf / 2.0, k - 1) * se)
    return BatchMeansResult(
        mean=mean,
        se=se,
        half_width=hw,
        ci_low=mean - hw,
        ci_high=mean + hw,
        n_batches=k,
        batch_size=b,
        n_dropped=int(a.size - k * b),
        confidence=conf,
        batch_values=bm,
    )


# =========================================================================== bootstrap


@dataclass(frozen=True)
class BootstrapResult:
    """Paired bootstrap: ``estimate`` (statistic on the data), interval
    ``(ci_low, ci_high)``, ``standard_error`` (SD of the bootstrap distribution, ddof 1),
    the ``distribution`` (n_resamples,), ``method`` and ``confidence``."""

    estimate: float
    ci_low: float
    ci_high: float
    standard_error: float
    distribution: np.ndarray
    method: str
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval excludes 0."""
        return bool(self.ci_low > 0.0 or self.ci_high < 0.0)


Statistic = Callable[..., np.ndarray]
"""``statistic(*samples, axis=-1)``: vectorized over leading axes, reduces the last."""


def paired_bootstrap(
    samples: Sequence[Sequence[float] | np.ndarray],
    statistic: Statistic,
    *,
    rng: np.random.Generator,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    method: Literal["percentile", "bca"] = "percentile",
) -> BootstrapResult:
    """Paired bootstrap confidence interval of ``statistic(*samples)``.

    Observations (positions along the samples) are resampled jointly: resample r uses
    indices ``rng.integers(0, n, size=(n_resamples, n))[r]`` for every sample (the draw of
    ``scipy.stats.bootstrap(..., paired=True, rng=rng)``). ``method="percentile"`` gives the
    ``alpha/2`` and ``1 - alpha/2`` quantiles (linear interpolation) of the bootstrap
    distribution; ``"bca"`` the bias-corrected and accelerated interval (Efron and
    Tibshirani 1993, Eq. 14.10 with the jackknife acceleration), as scipy computes it.
    """
    xs = [_vec(s, f"samples[{i}]") for i, s in enumerate(samples)]
    if not xs:
        raise ValueError("need at least one sample")
    n = xs[0].size
    if n < 2 or any(s.size != n for s in xs):
        raise ValueError("paired samples must have the same length >= 2")
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be >= 1")
    if method not in ("percentile", "bca"):
        raise ValueError(f"unknown method {method!r}")
    conf = _check_conf(confidence)
    est = float(statistic(*xs, axis=-1))
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    dist = np.asarray(statistic(*[s[idx] for s in xs], axis=-1), dtype=np.float64)
    alpha = (1.0 - conf) / 2.0
    if method == "percentile":
        q = np.array([alpha, 1.0 - alpha])
    else:
        b = dist.size
        frac = (np.count_nonzero(dist < est) + np.count_nonzero(dist <= est)) / (2.0 * b)
        z0 = float(special.ndtri(frac))
        keep = ~np.eye(n, dtype=bool)
        jack = np.asarray(
            statistic(*[np.broadcast_to(s, (n, n))[keep].reshape(n, n - 1) for s in xs], axis=-1),
            dtype=np.float64,
        )
        u = (n - 1) * (jack.mean() - jack)
        a_hat = float((np.sum(u**3) / n**3) / (6.0 * (np.sum(u**2) / n**2) ** 1.5))
        za = float(special.ndtri(alpha))
        q = np.array(
            [
                special.ndtr(z0 + (z0 + za) / (1.0 - a_hat * (z0 + za))),
                special.ndtr(z0 + (z0 - za) / (1.0 - a_hat * (z0 - za))),
            ]
        )
    lo, hi = np.quantile(dist, q)
    return BootstrapResult(
        estimate=est,
        ci_low=float(lo),
        ci_high=float(hi),
        standard_error=float(np.std(dist, ddof=1)) if dist.size > 1 else math.nan,
        distribution=dist,
        method=method,
        confidence=conf,
    )


def rmse(e: np.ndarray, axis: int = -1) -> np.ndarray:
    """Root mean square of ``e`` along ``axis``."""
    return np.sqrt(np.mean(np.square(e), axis=axis))


def rmse_difference(e_a: np.ndarray, e_b: np.ndarray, axis: int = -1) -> np.ndarray:
    """``RMSE(e_a) - RMSE(e_b)`` (V-LAB Delta RMSE = RMSE(M0) - RMSE(M2) on test cells)."""
    return rmse(e_a, axis) - rmse(e_b, axis)


def mean_difference(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
    """``mean(a - b)`` along ``axis``."""
    return np.mean(np.asarray(a) - np.asarray(b), axis=axis)


# =========================================================================== binomial


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided interval for a binomial proportion ``k/n``:
    ``Beta(alpha/2; k, n-k+1)`` and ``Beta(1-alpha/2; k+1, n-k)`` quantiles, with the
    bounds 0 (k = 0) and 1 (k = n)."""
    k, n = int(k), int(n)
    if n < 1 or not 0 <= k <= n:
        raise ValueError(f"need n >= 1 and 0 <= k <= n, got k={k}, n={n}")
    alpha = 1.0 - _check_conf(confidence)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


# =========================================================================== rank correlation


def rankdata(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Average ranks (1-based; ties share the mean rank) along ``axis``."""
    return np.asarray(stats.rankdata(x, method="average", axis=axis), dtype=np.float64)


def _pearson_last(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    da = a - a.mean(axis=-1, keepdims=True)
    db = b - b.mean(axis=-1, keepdims=True)
    num = np.sum(da * db, axis=-1)
    den = np.sqrt(np.sum(da * da, axis=-1) * np.sum(db * db, axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(num / den, -1.0, 1.0)


def _spearman_stat(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    return _pearson_last(rankdata(x, axis=axis), rankdata(y, axis=axis))


@dataclass(frozen=True)
class SpearmanResult:
    """Spearman ``rho``, two-sided ``p_value`` (Student t with n-2 degrees of freedom, as
    ``scipy.stats.spearmanr``), interval ``(ci_low, ci_high)``, ``n`` and ``ci_method``."""

    rho: float
    p_value: float
    ci_low: float
    ci_high: float
    n: int
    ci_method: str
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval excludes 0."""
        return bool(self.ci_low > 0.0 or self.ci_high < 0.0)


def spearman(
    x: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    ci_method: Literal["fieller", "bootstrap"] = "fieller",
    rng: np.random.Generator | None = None,
    n_resamples: int = 10_000,
) -> SpearmanResult:
    """Spearman rank correlation of paired samples with a confidence interval.

    ``rho`` is Pearson's correlation of the average ranks. ``ci_method="fieller"``:
    ``tanh(atanh(rho) +- z_{1-alpha/2} sqrt(1.06/(n-3)))`` (Fieller, Hartley and Pearson
    1957; needs n >= 4). ``"bootstrap"``: paired percentile bootstrap of rho (needs ``rng``).
    """
    a, b = _vec(x, "x"), _vec(y, "y")
    if a.size != b.size or a.size < 3:
        raise ValueError("x and y must have the same length >= 3")
    conf = _check_conf(confidence)
    n = a.size
    rho = float(_spearman_stat(a, b))
    if not math.isfinite(rho):
        raise ValueError("Spearman rho is undefined (a sample is constant)")
    dof = n - 2
    if abs(rho) >= 1.0:
        p = 0.0
    else:
        t = rho * math.sqrt(dof / ((1.0 - rho) * (1.0 + rho)))
        p = float(2.0 * stats.t.sf(abs(t), dof))
    if ci_method == "fieller":
        if n < 4:
            raise ValueError("the Fieller interval needs n >= 4")
        z = math.atanh(max(min(rho, 1.0 - 1e-15), -1.0 + 1e-15))
        h = float(stats.norm.ppf(0.5 + conf / 2.0)) * math.sqrt(1.06 / (n - 3))
        lo, hi = math.tanh(z - h), math.tanh(z + h)
    elif ci_method == "bootstrap":
        if rng is None:
            raise ValueError("ci_method='bootstrap' needs a keyed rng")
        res = paired_bootstrap(
            [a, b], _spearman_stat, rng=rng, n_resamples=n_resamples, confidence=conf
        )
        lo, hi = res.ci_low, res.ci_high
    else:
        raise ValueError(f"unknown ci_method {ci_method!r}")
    return SpearmanResult(
        rho=rho, p_value=p, ci_low=lo, ci_high=hi, n=n, ci_method=ci_method, confidence=conf
    )


# =========================================================================== error metrics


@dataclass(frozen=True)
class ErrorMetrics:
    """Errors ``e = estimate - reference``: ``n``, ``bias`` (mean e), ``sd`` (ddof 1; NaN
    for n = 1), ``rmse`` (sqrt mean e^2), ``mae`` (mean |e|), ``max_abs`` (max |e|)."""

    n: int
    bias: float
    sd: float
    rmse: float
    mae: float
    max_abs: float

    def as_dict(self) -> dict[str, float | int]:
        """Plain dict."""
        return {
            "n": self.n,
            "bias": self.bias,
            "sd": self.sd,
            "rmse": self.rmse,
            "mae": self.mae,
            "max_abs": self.max_abs,
        }


def error_metrics(
    estimate: Sequence[float] | np.ndarray, reference: Sequence[float] | np.ndarray
) -> ErrorMetrics:
    """Bias, SD, RMSE, MAE and max |e| of ``e = estimate - reference`` (same units).

    For V-TANG, ``estimate`` = measured and ``reference`` = model gives the reported
    "measurements lie +0.98 dB above the model" sign convention.
    """
    a, b = _vec(estimate, "estimate"), _vec(reference, "reference")
    if a.size != b.size or a.size < 1:
        raise ValueError("estimate and reference must have the same length >= 1")
    e = a - b
    return ErrorMetrics(
        n=int(e.size),
        bias=float(np.mean(e)),
        sd=float(np.std(e, ddof=1)) if e.size > 1 else math.nan,
        rmse=float(np.sqrt(np.mean(e * e))),
        mae=float(np.mean(np.abs(e))),
        max_abs=float(np.max(np.abs(e))),
    )
