"""Closed-form RF oracles: free space, two-ray, Fresnel, knife edge, noise, CFO, CRB, FD filters.

Independent of ``sremu.physics``, ``sremu.channel`` and ``sremu.impair`` (DESIGN.md sections
2.3, 10.1, 14.3). Ported from ``research/scratch/pm_lib.py``, ``pm_checks.py``,
``design_checks/pv_checks.py`` and ``design_checks/fd_check.py`` and written from the
equations of DESIGN.md sections 3.1, 5.2, 5.7, 5.9, 5.12 and 6.3.

Conventions (section 3.1):

* A coefficient ``h`` is dimensionless and Friis-normalized: ``P_R = P_T*|h|**2``.
* Every complex coefficient returned here INCLUDES the propagation phase
  ``exp(-1j*k*path_length)`` (``k = 2*pi*fc/C0``), i.e. it is the full complex-baseband gain of
  the path at the carrier. (sremu's ``PathBundle.amp`` excludes it; a comparison must
  multiply by ``exp(-2j*pi*fc*delay_s)``.)
* Speed of light: ``sremu.units.C0`` only.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import fresnel

from sremu.units import C0, K_B, T0

__all__ = [
    "KT0_DBM_PER_HZ",
    "brewster_grazing_angle_rad",
    "cfo_rotate",
    "complex_permittivity",
    "crb_frequency_std_hz",
    "db20",
    "fd_inband_error_db",
    "fresnel_gamma_h",
    "fresnel_gamma_v",
    "friis_h",
    "image_theory_h",
    "itu_j_db",
    "kaiser_sinc_taps",
    "knife_edge_F",
    "knife_edge_loss_db",
    "knife_edge_nu",
    "lagrange4_taps",
    "legacy_truncation_cfo_slope",
    "thermal_noise_dbm",
    "thermal_noise_w",
    "two_ray_h",
    "v2_tick_boundary_phase_step_rad",
    "wavelength",
    "wavenumber",
]

KT0_DBM_PER_HZ: float = 10.0 * math.log10(K_B * T0) + 30.0
"""Thermal noise density k_B*T0 in dBm/Hz (-173.975 dBm/Hz)."""


def wavelength(fc_hz: float) -> float:
    """Free-space wavelength C0/fc in m."""
    return C0 / fc_hz


def wavenumber(fc_hz: float) -> float:
    """Free-space wavenumber 2*pi*fc/C0 in rad/m."""
    return 2.0 * math.pi * fc_hz / C0


def db20(h: ArrayLike) -> NDArray[np.float64]:
    """Amplitude ratio |h| in dB: 20*log10|h|."""
    return 20.0 * np.log10(np.abs(np.asarray(h)))


# --------------------------------------------------------------------------- free space


def friis_h(d_m: ArrayLike, fc_hz: float, g_t: float = 1.0, g_r: float = 1.0) -> NDArray:
    """LOS coefficient sqrt(G_T G_R)*lambda/(4*pi*d)*exp(-j*k*d) (section 5.2, PH-01)."""
    lam = wavelength(fc_hz)
    d = np.asarray(d_m, dtype=float)
    return math.sqrt(g_t * g_r) * lam / (4.0 * math.pi * d) * np.exp(-1j * wavenumber(fc_hz) * d)


def image_theory_h(d1_m: float, d2_m: float, fc_hz: float) -> complex:
    """Specular reflection from an infinite perfect mirror (image theory, PH-05).

    ``lambda/(4*pi*(d1+d2))*exp(-j*k*(d1+d2))`` with Gamma = +1 (the RIS-kernel convention:
    a uniformly reflecting plate with Gamma = 1 reproduces the image of the source).
    """
    L = d1_m + d2_m
    return complex(wavelength(fc_hz) / (4.0 * math.pi * L) * np.exp(-1j * wavenumber(fc_hz) * L))


def two_ray_h(d_m: float, ht_m: float, hr_m: float, fc_hz: float, gamma: complex = -1.0) -> complex:
    """Flat-earth two-ray coefficient (section 5.7, PH-10, G8).

    ``lambda/(4*pi)*(exp(-j*k*d0)/d0 + gamma*exp(-j*k*d1)/d1)`` with the direct length
    ``d0 = hypot(d, ht-hr)`` and the image length ``d1 = hypot(d, ht+hr)``; isotropic antennas.
    """
    k = wavenumber(fc_hz)
    lam = wavelength(fc_hz)
    d0 = math.hypot(d_m, ht_m - hr_m)
    d1 = math.hypot(d_m, ht_m + hr_m)
    # the reflected ray relative to the direct ray: path difference computed without cancellation
    dd = 4.0 * ht_m * hr_m / (d0 + d1)
    return complex(
        lam
        / (4.0 * math.pi)
        * np.exp(-1j * k * d0)
        * (1.0 / d0 + gamma * np.exp(-1j * k * dd) / d1)
    )


# --------------------------------------------------------------------------- ground (Fresnel)


def complex_permittivity(eps_r: float, sigma_s_per_m: float, fc_hz: float) -> complex:
    """``eps_c = eps_r - j*60*sigma*lambda`` (section 5.7)."""
    return complex(eps_r, -60.0 * sigma_s_per_m * wavelength(fc_hz))


def _root(eps_c: complex, psi: ArrayLike) -> NDArray:
    c = np.cos(np.asarray(psi, dtype=float))
    return np.sqrt(eps_c - c * c + 0j)


def fresnel_gamma_h(psi_rad: ArrayLike, eps_c: complex) -> NDArray:
    """Horizontal-polarization Fresnel coefficient at grazing angle psi (section 5.7).

    ``(sin psi - sqrt(eps_c - cos^2 psi)) / (sin psi + sqrt(eps_c - cos^2 psi))``.
    """
    s = np.sin(np.asarray(psi_rad, dtype=float))
    r = _root(eps_c, psi_rad)
    return (s - r) / (s + r)


def fresnel_gamma_v(psi_rad: ArrayLike, eps_c: complex) -> NDArray:
    """Vertical-polarization Fresnel coefficient at grazing angle psi (section 5.7).

    ``(eps_c sin psi - sqrt(eps_c - cos^2 psi)) / (eps_c sin psi + sqrt(eps_c - cos^2 psi))``.
    """
    s = np.sin(np.asarray(psi_rad, dtype=float))
    r = _root(eps_c, psi_rad)
    return (eps_c * s - r) / (eps_c * s + r)


def brewster_grazing_angle_rad(eps_r: float) -> float:
    """Grazing Brewster angle of a lossless dielectric: ``asin(1/sqrt(eps_r + 1))``."""
    return math.asin(1.0 / math.sqrt(eps_r + 1.0))


# --------------------------------------------------------------------------- knife edge


def knife_edge_nu(h_m: float, d1_m: float, d2_m: float, fc_hz: float) -> float:
    """Fresnel-Kirchhoff parameter ``nu = h*sqrt(2*(d1+d2)/(lambda*d1*d2))`` (section 5.9)."""
    lam = wavelength(fc_hz)
    return h_m * math.sqrt(2.0 * (d1_m + d2_m) / (lam * d1_m * d2_m))


def knife_edge_F(nu: ArrayLike) -> NDArray:
    """Complex knife-edge field factor ``F(nu) = ((1+j)/2)*int_nu^inf exp(-j*pi*t^2/2) dt``.

    Uses ``int_0^inf exp(-j*pi*t^2/2) dt = (1-j)/2`` and scipy's Fresnel integrals
    ``S(nu), C(nu)``: the tail integral is ``(1/2 - C) - j*(1/2 - S)``.
    """
    s, c = fresnel(np.asarray(nu, dtype=float))
    tail = (0.5 - c) - 1j * (0.5 - s)
    return (1.0 + 1.0j) / 2.0 * tail


def knife_edge_loss_db(nu: ArrayLike) -> NDArray[np.float64]:
    """Exact diffraction loss ``-20*log10|F(nu)|`` in dB (positive = loss)."""
    return -db20(knife_edge_F(nu))


def itu_j_db(nu: ArrayLike) -> NDArray[np.float64]:
    """ITU-R P.526 approximation ``J(nu) = 6.9 + 20*log10(sqrt((nu-0.1)^2+1) + nu - 0.1)``.

    Valid for nu > -0.78 (oracle only, section 5.9).
    """
    v = np.asarray(nu, dtype=float)
    return 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)


# --------------------------------------------------------------------------- noise and CFO


def thermal_noise_w(f_s_hz: float, noise_figure_db: float = 0.0) -> float:
    """Per-receiver thermal noise power ``k_B*T0*F*f_s`` in W (section 5.12, PH-21, G5)."""
    return K_B * T0 * 10.0 ** (noise_figure_db / 10.0) * f_s_hz


def thermal_noise_dbm(f_s_hz: float, noise_figure_db: float = 0.0) -> float:
    """Per-receiver thermal noise power in dBm (-113.975 dBm at 1 MS/s, F = 0 dB)."""
    return 10.0 * math.log10(thermal_noise_w(f_s_hz, noise_figure_db)) + 30.0


def cfo_rotate(x: ArrayLike, df_hz: int | Fraction, f_s: int | Fraction, n0: int = 0) -> NDArray:
    """Apply a carrier frequency offset: ``y[n] = x[n]*exp(+j*2*pi*df*n/f_s)``, n absolute.

    The phase is reduced exactly (``frac(df*n/f_s)`` in rational arithmetic, section 6.3), so
    the result does not depend on how the stream is chunked. ``df_hz`` and ``f_s`` must be
    exact (int or Fraction). This is the ``comm.PhaseFrequencyOffset`` convention (G4) and the
    ``lo_offset_hz = f_c,tx - f_c,rx`` convention of the render contract.
    """
    q = Fraction(df_hz) / Fraction(f_s)
    xv = np.asarray(x, dtype=complex)
    n = np.arange(n0, n0 + xv.shape[0], dtype=np.int64)
    num, den = q.numerator, q.denominator
    # exact fractional cycles: ((num*n) mod den)/den, computed in Python ints for safety
    frac_cycles = np.array([((num * int(k)) % den) / den for k in n], dtype=float)
    return xv * np.exp(2j * math.pi * frac_cycles)


def crb_frequency_std_hz(f_s_hz: float, snr_linear: float, n_samples: int) -> float:
    """Cramer-Rao bound (std) for the frequency of one complex tone in AWGN (Rife-Boorstyn).

    ``var(f) >= 6*f_s^2 / ((2*pi)^2 * SNR * N * (N^2 - 1))`` with SNR the per-sample SNR
    (unknown amplitude, phase and frequency). Used by V-CFO-4 / ST-05.
    """
    n = float(n_samples)
    return f_s_hz / (2.0 * math.pi) * math.sqrt(6.0 / (snr_linear * n * (n * n - 1.0)))


def legacy_truncation_cfo_slope(f_s: int | Fraction, tau_s: Fraction) -> Fraction:
    """Slope of the estimated-vs-true CFO under v1's ``int(f_s*tau)`` truncation (V-CFO-2).

    v1 advances the CFO phase by a full ``tau`` per tick but emits only ``int(f_s*tau)``
    samples, so the apparent frequency is scaled by ``f_s*tau/int(f_s*tau)`` (11.76/11).
    """
    ft = Fraction(f_s) * Fraction(tau_s)
    return ft / math.floor(ft)


def v2_tick_boundary_phase_step_rad(df_hz: float, f_s: int | Fraction, tau_s: Fraction) -> float:
    """Phase step at a tick boundary under the v2 per-tick CFO phase (RT-09 teeth).

    v2 restarts the CFO phase at ``2*pi*df*m*tau`` for tick m but each tick carries
    ``n = int(f_s*tau)`` samples, so between the last sample of tick m and the first of tick
    m+1 the phase advances by ``2*pi*df*(tau - (n-1)/f_s)`` instead of ``2*pi*df/f_s``.
    """
    n = math.floor(Fraction(f_s) * Fraction(tau_s))
    return 2.0 * math.pi * df_hz * float(Fraction(tau_s) - Fraction(n - 1) / Fraction(f_s))


# --------------------------------------------------------------------------- FD filters (PH-20)


def lagrange4_taps(mu: float) -> tuple[NDArray[np.float64], float]:
    """4-tap cubic Lagrange fractional-delay filter for delay ``D = 1 + mu`` (taps 0..3).

    Ported from ``design_checks/fd_check.py``; the rejected default of the input documents
    (teeth of PH-20).
    """
    D = 1.0 + mu
    h = []
    for k in range(4):
        p = 1.0
        for m in range(4):
            if m != k:
                p *= (D - m) / (k - m)
        h.append(p)
    return np.asarray(h), D


def kaiser_sinc_taps(n_taps: int, beta: float, mu: float) -> tuple[NDArray[np.float64], float]:
    """Kaiser-windowed sinc FD filter with the window evaluated at the fractional offset.

    Delay ``D = (L-1)//2 + mu``; taps ``sinc(n-D) * I0(beta*sqrt(1-t^2))/I0(beta)`` with
    ``t = 2*(n-D)/L`` (zero for |t| > 1). Ported from ``design_checks/fd_check.py``.
    """
    n = np.arange(n_taps)
    D = (n_taps - 1) // 2 + mu
    t = 2.0 * (n - D) / n_taps
    w = np.where(
        np.abs(t) <= 1.0, np.i0(beta * np.sqrt(np.clip(1.0 - t * t, 0.0, 1.0))) / np.i0(beta), 0.0
    )
    return np.sinc(n - D) * w, D


def fd_inband_error_db(
    h: ArrayLike, delay_samples: float, band: float, n_freq: int = 2001
) -> float:
    """Worst in-band error ``max |H(f) - exp(-j*2*pi*f*D)|`` over ``|f| <= band*f_s``, in dB."""
    taps = np.asarray(h, dtype=float)
    f = np.linspace(-band, band, n_freq)
    n = np.arange(taps.shape[0])
    H = (taps[None, :] * np.exp(-2j * math.pi * f[:, None] * n[None, :])).sum(axis=1)
    E = np.exp(-2j * math.pi * f * delay_samples)
    err = float(np.max(np.abs(H - E)))
    return 20.0 * math.log10(err) if err > 0.0 else -math.inf
