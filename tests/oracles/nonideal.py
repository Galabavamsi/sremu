"""Closed-form non-ideality and optimum oracles (DESIGN.md sections 5.5, 5.6 and 5.13).

* Large-N alphabet loss ``eta`` (PH-13): uniform b-bit closed form, binary closed form and a
  quadrature for any finite alphabet (unequal amplitudes allowed).
* Abeywickrama phase-dependent amplitude model and its loss (PH-14).
* Binary m/d decomposition (PH-25), efficiency and dead-cell losses (PH-15).
* Incidence-only reciprocal extension (PH-28).
* Exhaustive finite-alphabet optimum and an independent critical-angle reference (PH-24/29).

Independent of ``sremu.physics`` (sections 2.3, 10.1).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

__all__ = [
    "abeywickrama_beta",
    "abeywickrama_eta_db",
    "alphabet_eta",
    "alphabet_eta_db",
    "binary_eta",
    "binary_md",
    "bruteforce_optimum",
    "critical_angle_optimum",
    "dead_cell_loss_db",
    "efficiency_amplitude",
    "half_difference_extension",
    "ideal_bbit",
    "quantize_argmax",
    "uniform_bbit_eta",
]


def ideal_bbit(b: int) -> NDArray[np.complex128]:
    """Ideal b-bit alphabet ``exp(j 2 pi s / 2^b)``, s = 0..2^b - 1."""
    s = np.arange(2**b)
    return np.exp(2j * math.pi * s / 2**b)


def uniform_bbit_eta(b: int) -> float:
    """Large-N power loss of ideal b-bit phase quantization: ``(2^b/pi * sin(pi/2^b))^2``."""
    m = 2**b
    return (m / math.pi * math.sin(math.pi / m)) ** 2


def binary_eta(delta_rad: float) -> float:
    """Large-N loss of an equal-amplitude binary alphabet with phase separation Delta.

    ``eta = (2 sin(Delta'/2)/pi)^2`` with ``Delta' = min(Delta, 2 pi - Delta)`` (section 5.6).
    """
    d = math.fmod(abs(delta_rad), 2.0 * math.pi)
    d = min(d, 2.0 * math.pi - d)
    return (2.0 * math.sin(d / 2.0) / math.pi) ** 2


def alphabet_eta(alphabet: ArrayLike, reference_mag: float = 1.0) -> float:
    """Large-N loss of an arbitrary finite alphabet, by exact arc integrals (PH-13).

    With uniform channel phases and the per-element best choice, each element contributes
    ``max_s Re{Gamma_s exp(j phi)}`` along the beam direction, so
    ``eta = (E_phi[max_s Re{Gamma_s e^{j phi}}] / reference_mag)^2`` relative to an ideal
    continuous phase shifter of amplitude ``reference_mag``.

    The integrand is piecewise sinusoidal: between consecutive switching angles (where two
    entries tie) one entry ``g_s`` wins, and its arc integral is exact,
    ``int Re{g_s e^{j phi}} dphi = Re{g_s (e^{j phi_hi} - e^{j phi_lo}) / j}``.
    """
    g = np.asarray(alphabet, dtype=complex).ravel()
    # breakpoints: where two alphabet entries tie, Re{(g_a - g_b) e^{j phi}} = 0
    br = [-math.pi, math.pi]
    for a in range(g.size):
        for b in range(a + 1, g.size):
            dg = g[a] - g[b]
            if dg != 0:
                base = -math.atan2(dg.imag, dg.real) + math.pi / 2.0
                br.extend(math.remainder(t, 2.0 * math.pi) for t in (base, base + math.pi))
    br = sorted(set(br))
    total = 0.0
    for lo, hi in itertools.pairwise(br):
        if hi > lo:
            mid = 0.5 * (lo + hi)
            s = int(np.argmax(np.real(g * np.exp(1j * mid))))
            total += float(np.real(g[s] * (np.exp(1j * hi) - np.exp(1j * lo)) / 1j))
    mean = total / (2.0 * math.pi)
    return (mean / reference_mag) ** 2


def alphabet_eta_db(alphabet: ArrayLike, reference_mag: float = 1.0) -> float:
    """``10*log10(alphabet_eta)`` in dB."""
    return 10.0 * math.log10(alphabet_eta(alphabet, reference_mag))


def abeywickrama_beta(theta: ArrayLike, beta_min: float, phi: float, alpha: float) -> NDArray:
    """Abeywickrama amplitude.

    ``beta(theta) = (1 - beta_min)((sin(theta - phi) + 1)/2)^alpha + beta_min``.
    """
    t = np.asarray(theta, dtype=float)
    return (1.0 - beta_min) * ((np.sin(t - phi) + 1.0) / 2.0) ** alpha + beta_min


def abeywickrama_eta_db(beta_min: float, alpha: float, phi: float = 0.43 * math.pi) -> float:
    """Large-N loss of the Abeywickrama model with continuous phase control (PH-14).

    The phase is set per element to co-phase the channel, so with uniform channel phases the
    required phase theta is uniform and the coherent amplitude is ``mean_theta beta(theta)``;
    ``eta = 20*log10(mean beta)``. Independent of ``phi`` (integral over a full period).
    """
    m = quad(
        lambda t: float(abeywickrama_beta(t, beta_min, phi, alpha)), -math.pi, math.pi, epsabs=1e-14
    )[0]
    return 20.0 * math.log10(m / (2.0 * math.pi))


def binary_md(a: complex, b: complex) -> tuple[complex, complex]:
    """Binary decomposition ``Gamma = m + d*s``, s in {+1 (state a), -1 (state b)} (PH-25)."""
    return (a + b) / 2.0, (a - b) / 2.0


def efficiency_amplitude(eta_eff: float) -> float:
    """Amplitude scale ``sqrt(eta_eff)`` applied by the efficiency step (section 5.6, PH-15)."""
    return math.sqrt(eta_eff)


def dead_cell_loss_db(dead_fraction: float) -> float:
    """Expected coherent loss of a focused large panel with absorbing dead cells: 20 log10(1-p).

    Every live cell adds an equal co-phased amplitude, so the sum scales with the live
    fraction (PH-15: 10 % dead -> -0.915 dB).
    """
    return 20.0 * math.log10(1.0 - dead_fraction)


def half_difference_extension(g: Callable[[float], complex]) -> Callable[[float, float], complex]:
    """Incidence-only table extension (PH-28).

    ``Gamma(theta_i, theta_s) := g(|theta_i - theta_s|/2)``.
    """

    def gamma(theta_i: float, theta_s: float) -> complex:
        return g(abs(theta_i - theta_s) / 2.0)

    return gamma


def quantize_argmax(alphabet: ArrayLike, phi_rad: ArrayLike) -> NDArray[np.int64]:
    """Quantizer of section 5.6: ``s = argmax_s Re{Gamma(s) exp(-j phi)}`` per element (PH-30)."""
    g = np.asarray(alphabet, dtype=complex).ravel()
    p = np.asarray(phi_rad, dtype=float).ravel()
    return np.argmax(np.real(g[:, None] * np.exp(-1j * p[None, :])), axis=0).astype(np.int64)


# --------------------------------------------------------------------------- optimum


def bruteforce_optimum(
    z0: complex, Z: ArrayLike, chunk: int = 1 << 16
) -> tuple[float, NDArray[np.int64]]:
    """Exhaustive ``max_s |z0 + sum_n Z[s_n, n]|`` over all S^N configurations (PH-24/29).

    Returns (value, s*). Vectorized in chunks of configurations (base-S digits).
    """
    Zm = np.asarray(Z, dtype=complex)
    S, N = Zm.shape
    total = S**N
    best_v = -1.0
    best_i = 0
    powers = S ** np.arange(N, dtype=np.int64)
    for start in range(0, total, chunk):
        idx = np.arange(start, min(total, start + chunk), dtype=np.int64)
        digits = (idx[:, None] // powers[None, :]) % S  # (C, N)
        vals = np.abs(z0 + Zm[digits, np.arange(N)[None, :]].sum(axis=1))
        j = int(np.argmax(vals))
        if vals[j] > best_v:
            best_v = float(vals[j])
            best_i = int(idx[j])
    s = (best_i // powers) % S
    return best_v, s.astype(np.int64)


def critical_angle_optimum(z0: complex, Z: ArrayLike) -> tuple[float, NDArray[np.int64]]:
    """Reference O(S^2 N * S^2 N) optimum by evaluating every critical angle (section 5.13).

    For a direction theta the best choice is ``s_n(theta) = argmax_s Re{Z[s,n] e^{-j theta}}``;
    it changes only at ``arg(Z_a - Z_b) +- pi/2``. Every arc between consecutive critical
    angles is probed at its midpoint (and the angle itself), which covers all candidates.
    Written independently of any sweep implementation; for tests only.
    """
    Zm = np.asarray(Z, dtype=complex)
    S, N = Zm.shape
    crit = []
    for a in range(S):
        for b in range(a + 1, S):
            d = Zm[a] - Zm[b]
            nz = d != 0
            ang = np.angle(d[nz])
            crit.append(ang + math.pi / 2.0)
            crit.append(ang - math.pi / 2.0)
    c = np.sort(np.mod(np.concatenate(crit) if crit else np.zeros(1), 2.0 * math.pi))
    mids = (
        c + np.roll(c, -1) + np.where(np.arange(c.size) == c.size - 1, 2.0 * math.pi, 0.0)
    ) / 2.0
    probes = np.concatenate([c, mids])
    best_v = -1.0
    best_s = np.zeros(N, dtype=np.int64)
    cols = np.arange(N)
    for th in probes:
        s = np.argmax(np.real(Zm * np.exp(-1j * th)), axis=0)
        v = abs(z0 + Zm[s, cols].sum())
        if v > best_v:
            best_v = float(v)
            best_s = s.astype(np.int64)
    return best_v, best_s
