"""Closed-form mobility, Doppler, interpolation-budget and statistics oracles.

DESIGN.md sections 4.4 (error budget), 5.10 (mobility), 5.11 (SoS diffuse, ST-02), 10.5
(ST-01/ST-04/ST-05) and 10.2 (PH-17, PH-21). Independent of ``sremu.physics``,
``sremu.channel`` and ``sremu.impair``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import j0
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from sremu.units import C0

__all__ = [
    "ber_bpsk",
    "block_constant_alias_hz",
    "block_constant_jump_rad",
    "block_constant_mean_doppler_hz",
    "chord_phase_error_rad",
    "clopper_pearson",
    "corr_5sigma",
    "doppler_hz",
    "jakes_acf",
    "ks_critical",
    "ks_statistic",
    "radial_speed_mps",
    "rayleigh_afd_s",
    "rayleigh_cdf",
    "rayleigh_lcr_hz",
    "rwp1d_cdf",
    "rwp1d_leg_sd",
    "rwp1d_mean_leg",
    "rwp1d_pdf",
    "t_curv_s",
    "tile_side_max_m",
]


# --------------------------------------------------------------------------- Doppler


def radial_speed_mps(p_node: ArrayLike, v_node: ArrayLike, p_ref: ArrayLike) -> float:
    """Approach speed of a moving node toward a fixed point (positive when approaching)."""
    p = np.asarray(p_node, dtype=float)
    v = np.asarray(v_node, dtype=float)
    r = np.asarray(p_ref, dtype=float)
    u = (r - p) / np.linalg.norm(r - p)
    return float(v @ u)


def doppler_hz(v_approach_mps: float, fc_hz: float) -> float:
    """Doppler shift ``f_D = v*fc/C0``, positive on approach (section 3.1)."""
    return v_approach_mps * fc_hz / C0


# --------------------------------------------------------------------------- block_constant teeth


def _wrap_rad(x: float) -> float:
    """Principal value in (-pi, pi]."""
    y = math.remainder(x, 2.0 * math.pi)
    return math.pi if y == -math.pi else y


def block_constant_jump_rad(fc_hz: float, v_approach_mps: float, block_s: Fraction) -> float:
    """Carrier-phase jump of the v2 staircase at each block boundary: ``k v T`` (unwrapped).

    v2 holds the geometry constant over each block of ``T`` s, so the LOS phase ``-k d(t)``
    stays flat inside a block and jumps by ``k v T`` at its end (4.401 rad in T9).
    """
    return 2.0 * math.pi * fc_hz / C0 * v_approach_mps * float(block_s)


def block_constant_alias_hz(fc_hz: float, v_approach_mps: float, block_s: Fraction) -> float:
    """Long-run apparent frequency of the staircase: ``wrap(k v T) / (2 pi T)``.

    -149.77 Hz for the T9 case (3.5 GHz, 30 m/s, 2 ms blocks): NOT the PH-17 teeth value,
    which is the mean over the finite T9 window (:func:`block_constant_mean_doppler_hz`).
    """
    jump = _wrap_rad(block_constant_jump_rad(fc_hz, v_approach_mps, block_s))
    return jump / (2.0 * math.pi * float(block_s))


def block_constant_mean_doppler_hz(
    fc_hz: float, v_approach_mps: float, block_s: Fraction, fs_hz: Fraction, n_samples: int
) -> float:
    """PH-17 teeth (D7): mean instantaneous frequency of the v2 staircase over a sample window.

    Samples ``n = 0 .. n_samples-1`` at ``fs_hz``; the instantaneous frequency is the wrapped
    phase step between neighbouring samples times ``fs/(2 pi)``. Inside a block the step is 0;
    each of the ``floor((n_samples - 1)/(fs T))`` block boundaries crossed by the window
    contributes ``wrap(k v T)`` (exact integer count with ``Fraction`` rates). The T9 window
    (``pm_checks.py``: 0.2 s at 10 kS/s, 2 ms blocks, 30 m/s, 3.5 GHz) gives 99 boundaries
    over 1999 steps: -148.33 Hz (the long-run alias is -149.77 Hz, see
    :func:`block_constant_alias_hz`).
    """
    spb = Fraction(fs_hz) * Fraction(block_s)  # samples per block
    if spb < 1:
        raise ValueError("block shorter than one sample")
    boundaries = math.floor(Fraction(n_samples - 1) / spb)
    jump = _wrap_rad(block_constant_jump_rad(fc_hz, v_approach_mps, block_s))
    return boundaries * jump * float(fs_hz) / (2.0 * math.pi * (n_samples - 1))


# --------------------------------------------------------------------------- error budget (4.4)


def chord_phase_error_rad(fc_hz: float, v_perp_mps: float, r_min_m: float, t_u_s: float) -> float:
    """Worst chord (linear-delay) phase error over one snapshot interval: ``k r'' T^2 / 8``.

    ``r'' <= v_perp^2 / r_min`` (section 4.4); the inverse of :func:`t_curv_s`.
    """
    k = 2.0 * math.pi * fc_hz / C0
    return k * (v_perp_mps**2 / r_min_m) * t_u_s**2 / 8.0


def t_curv_s(eps_rel: float, r_min_m: float, fc_hz: float, v_perp_mps: float) -> float:
    """Chord-curvature snapshot interval ``T_curv = sqrt(8 eps r_min / (k v_perp^2))`` (4.4).

    From ``k * rddot * T^2 / 8 <= eps`` with ``rddot <= v_perp^2/r_min``.
    """
    k = 2.0 * math.pi * fc_hz / C0
    return math.sqrt(8.0 * eps_rel * r_min_m / (k * v_perp_mps**2))


def tile_side_max_m(
    eps_rel: float, fc_hz: float, r_min_m: float, v_rel_mps: float, t_u_s: float
) -> float:
    """Largest tile side ``D_t = sqrt(8 eps) lambda R_min / (2 pi v_rel T_u)`` (section 4.4)."""
    lam = C0 / fc_hz
    return math.sqrt(8.0 * eps_rel) * lam * r_min_m / (2.0 * math.pi * v_rel_mps * t_u_s)


# --------------------------------------------------------------------------- RWP (ST-01)


def rwp1d_pdf(x: ArrayLike, l_m: float) -> NDArray[np.float64]:
    """Stationary position pdf of 1-D random waypoint on [0, l] (no pause): 6x/l^2 - 6x^2/l^3."""
    xv = np.asarray(x, dtype=float)
    return np.where((xv >= 0) & (xv <= l_m), 6.0 * xv / l_m**2 - 6.0 * xv**2 / l_m**3, 0.0)


def rwp1d_cdf(x: ArrayLike, l_m: float) -> NDArray[np.float64]:
    """Stationary position CDF of 1-D random waypoint: ``3x^2/l^2 - 2x^3/l^3`` (ST-01)."""
    xv = np.clip(np.asarray(x, dtype=float), 0.0, l_m)
    return 3.0 * xv**2 / l_m**2 - 2.0 * xv**3 / l_m**3


def rwp1d_mean_leg(l_m: float) -> float:
    """Mean leg length ``E|U1 - U2| = l/3`` for independent uniform waypoints."""
    return l_m / 3.0


def rwp1d_leg_sd(l_m: float) -> float:
    """Standard deviation of the leg length ``|U1 - U2|``: ``l*sqrt(1/18)``."""
    return l_m * math.sqrt(1.0 / 18.0)


# --------------------------------------------------------------------------- fading (ST-02)


def jakes_acf(lag_s: ArrayLike, fd_hz: float) -> NDArray[np.float64]:
    """Normalized Clarke/Jakes autocorrelation ``J0(2 pi f_D tau)``."""
    return j0(2.0 * math.pi * fd_hz * np.asarray(lag_s, dtype=float))


def rayleigh_cdf(x: ArrayLike) -> NDArray[np.float64]:
    """CDF of the RMS-normalized Rayleigh envelope: ``1 - exp(-x^2)``."""
    xv = np.asarray(x, dtype=float)
    return 1.0 - np.exp(-xv * xv)


def rayleigh_lcr_hz(rho: ArrayLike, fd_hz: float) -> NDArray[np.float64]:
    """Level-crossing rate ``sqrt(2 pi) f_D rho exp(-rho^2)`` (rho = level / RMS)."""
    r = np.asarray(rho, dtype=float)
    return math.sqrt(2.0 * math.pi) * fd_hz * r * np.exp(-r * r)


def rayleigh_afd_s(rho: ArrayLike, fd_hz: float) -> NDArray[np.float64]:
    """Average fade duration ``(exp(rho^2) - 1) / (rho f_D sqrt(2 pi))``."""
    r = np.asarray(rho, dtype=float)
    return (np.exp(r * r) - 1.0) / (r * fd_hz * math.sqrt(2.0 * math.pi))


# --------------------------------------------------------------------------- statistics


def ks_statistic(samples: ArrayLike, cdf: Callable[[NDArray], ArrayLike]) -> float:
    """One-sample Kolmogorov-Smirnov statistic ``sup |F_n - F|``."""
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    f = np.asarray(cdf(x), dtype=float)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - f), np.max(f - (i - 1) / n)))


def ks_critical(n: int, alpha: float = 0.01) -> float:
    """Asymptotic one-sample KS critical value ``c(alpha)/sqrt(n)``.

    ``c = sqrt(-ln(alpha/2)/2)`` (1.628 at alpha = 0.01).
    """
    return math.sqrt(-math.log(alpha / 2.0) / 2.0) / math.sqrt(n)


def corr_5sigma(n: int) -> float:
    """5-sigma bound on the sample correlation magnitude of independent complex noise: 5/sqrt(n)."""
    return 5.0 / math.sqrt(n)


def ber_bpsk(ebn0_db: ArrayLike) -> NDArray[np.float64]:
    """Coherent BPSK (and Gray QPSK per bit) bit error rate ``Q(sqrt(2 Eb/N0))``."""
    g = 10.0 ** (np.asarray(ebn0_db, dtype=float) / 10.0)
    return norm.sf(np.sqrt(2.0 * g))


def clopper_pearson(k: int, n: int, confidence: float = 0.99) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval for k successes in n trials."""
    a = 1.0 - confidence
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1.0 - a / 2.0, k + 1, n - k))
    return lo, hi
