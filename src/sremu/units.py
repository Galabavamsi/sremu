"""Physical constants and dB helpers (DESIGN.md sections 3.1 and 5.12).

Rules (normative, section 3.1):

* ``C0`` is the only speed of light in the code base. No other literal for c may appear
  anywhere; ``tests/unit/test_layering.py`` enforces this with a token scan.
* Internal quantities are SI. dB and dBm appear only at I/O boundaries, through the helpers
  below.

Power conventions: a "power ratio" is dimensionless (|h|^2), an "amplitude ratio" is |h|.
``dbm_to_w``/``w_to_dbm`` convert absolute power in watts.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "C0",
    "DBM_PER_HZ_THERMAL",
    "K_B",
    "T0",
    "amp_ratio_to_db",
    "db_to_amp_ratio",
    "db_to_power_ratio",
    "dbm_to_w",
    "noise_power_w",
    "power_ratio_to_db",
    "thermal_noise_dbm",
    "w_to_dbm",
    "wavelength_m",
    "wavenumber_rad_per_m",
]

C0: Final[float] = 299_792_458.0
"""Speed of light in vacuum, m/s (exact SI value). The only c in sremu."""

K_B: Final[float] = 1.380649e-23
"""Boltzmann constant, J/K (exact SI 2019 value)."""

T0: Final[float] = 290.0
"""Reference noise temperature, K (IEEE standard T0)."""

DBM_PER_HZ_THERMAL: Final[float] = 10.0 * math.log10(K_B * T0 * 1e3)
"""Thermal noise density k_B*T0 in dBm/Hz (about -173.975 dBm/Hz)."""


def db_to_power_ratio[T: (float, NDArray[np.float64])](x_db: T) -> T:
    """Convert dB to a dimensionless power ratio: ``10**(x_db/10)``."""
    return 10.0 ** (x_db / 10.0)


def power_ratio_to_db[T: (float, NDArray[np.float64])](x: T) -> T:
    """Convert a dimensionless power ratio to dB: ``10*log10(x)``; 0 maps to -inf."""
    with np.errstate(divide="ignore"):
        out = 10.0 * np.log10(x)
    return float(out) if isinstance(x, float | int) else out


def db_to_amp_ratio[T: (float, NDArray[np.float64])](x_db: T) -> T:
    """Convert dB to an amplitude (field) ratio: ``10**(x_db/20)``."""
    return 10.0 ** (x_db / 20.0)


def amp_ratio_to_db[T: (float, NDArray[np.float64])](x: T) -> T:
    """Convert an amplitude (field) ratio |h| to dB: ``20*log10(x)``; 0 maps to -inf."""
    with np.errstate(divide="ignore"):
        out = 20.0 * np.log10(x)
    return float(out) if isinstance(x, float | int) else out


def dbm_to_w[T: (float, NDArray[np.float64])](p_dbm: T) -> T:
    """Convert absolute power from dBm to W: ``10**((p_dbm-30)/10)``."""
    return 10.0 ** ((p_dbm - 30.0) / 10.0)


def w_to_dbm[T: (float, NDArray[np.float64])](p_w: T) -> T:
    """Convert absolute power from W to dBm: ``10*log10(p_w) + 30``; 0 W maps to -inf."""
    with np.errstate(divide="ignore"):
        out = 10.0 * np.log10(p_w) + 30.0
    return float(out) if isinstance(p_w, float | int) else out


def wavelength_m(fc_hz: float) -> float:
    """Free-space wavelength in m for a carrier ``fc_hz`` in Hz: ``C0/fc_hz``."""
    if not fc_hz > 0.0:
        raise ValueError(f"carrier frequency must be > 0 Hz, got {fc_hz!r}")
    return C0 / fc_hz


def wavenumber_rad_per_m(fc_hz: float) -> float:
    """Free-space wavenumber k = 2*pi*fc/C0 in rad/m."""
    return 2.0 * math.pi / wavelength_m(fc_hz)


def noise_power_w(f_s: float, noise_figure_db: float = 0.0) -> float:
    """Thermal noise power per receiver, W: ``k_B * T0 * F * f_s`` (section 5.12).

    ``f_s`` is the RX sample rate in S/s (the complex baseband noise bandwidth) and
    ``noise_figure_db`` is the receiver noise figure F in dB.
    """
    if not f_s > 0:
        raise ValueError(f"sample rate must be > 0 S/s, got {f_s!r}")
    return K_B * T0 * db_to_power_ratio(float(noise_figure_db)) * float(f_s)


def thermal_noise_dbm(f_s: float, noise_figure_db: float = 0.0) -> float:
    """Thermal noise power per receiver in dBm (e.g. -113.975 dBm at 1 MS/s, F = 0 dB)."""
    return float(w_to_dbm(noise_power_w(f_s, noise_figure_db)))
