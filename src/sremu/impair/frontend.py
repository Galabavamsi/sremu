"""Device front end: full scale, gain, optional ADC clip, sc16 wire format (DESIGN.md 3.3, 6.3).

* **TX**: ``x = d * sqrt(P_FS,tx) * 10**(G_tx/20)`` with ``P_FS,tx`` from ``full_scale_dbm``
  (default 0 dBm) and ``gain_db`` (default 0). ``|x|**2`` is in W at the antenna port.
* **RX**: ``d = y * 10**(G_rx/20) / sqrt(P_FS,rx)``; by default ``10*log10|d|**2`` reads
  directly in dBm. ``adc_clip=True`` clips I and Q to +-1.
* **sc16**: ``round(d*32767)`` (round half to even) saturated to [-32768, 32767],
  interleaved I/Q; :func:`from_sc16` divides by 32767.

Chains (section 6.3): RX = sum of links -> diffuse -> thermal noise -> RX LO -> gain/full
scale -> optional clip -> deliver; TX = full scale/gain -> TX LO -> ring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sremu.errors import ConfigurationError
from sremu.units import db_to_amp_ratio, dbm_to_w

__all__ = ["SC16_SCALE", "RxFrontEnd", "TxFrontEnd", "from_sc16", "full_scale_w", "to_sc16"]

SC16_SCALE: Final[float] = 32767.0
"""sc16 code of digital full scale (|I| or |Q| = 1)."""


def full_scale_w(full_scale_dbm: float) -> float:
    """Full-scale power ``P_FS`` in W from ``full_scale_dbm``."""
    p = float(dbm_to_w(float(full_scale_dbm)))
    if not (p > 0.0 and math.isfinite(p)):
        raise ConfigurationError(f"full_scale_dbm must be finite, got {full_scale_dbm!r}")
    return p


@dataclass(frozen=True)
class TxFrontEnd:
    """TX scaling: digital samples (full scale 1) to watts-referenced baseband."""

    full_scale_dbm: float = 0.0
    gain_db: float = 0.0

    @property
    def scale(self) -> float:
        """``sqrt(P_FS,tx) * 10**(G_tx/20)`` in sqrt(W)."""
        return math.sqrt(full_scale_w(self.full_scale_dbm)) * float(
            db_to_amp_ratio(float(self.gain_db))
        )

    def apply(self, d: NDArray[np.complexfloating]) -> NDArray[np.complex64]:
        """``x = d * scale`` (complex64)."""
        return (np.asarray(d).astype(np.complex128) * self.scale).astype(np.complex64)


@dataclass(frozen=True)
class RxFrontEnd:
    """RX scaling: watts-referenced baseband to digital samples (full scale 1)."""

    full_scale_dbm: float = 0.0
    gain_db: float = 0.0
    adc_clip: bool = False

    @property
    def scale(self) -> float:
        """``10**(G_rx/20) / sqrt(P_FS,rx)`` in 1/sqrt(W)."""
        return float(db_to_amp_ratio(float(self.gain_db))) / math.sqrt(
            full_scale_w(self.full_scale_dbm)
        )

    def apply(self, y: NDArray[np.complexfloating]) -> NDArray[np.complex64]:
        """``d = y * scale``, optionally clipped to +-1 per component (complex64)."""
        d = np.asarray(y).astype(np.complex128) * self.scale
        if self.adc_clip:
            out = np.empty(d.shape[0], dtype=np.complex128)
            out.real = np.clip(d.real, -1.0, 1.0)
            out.imag = np.clip(d.imag, -1.0, 1.0)
            d = out
        return d.astype(np.complex64)


def to_sc16(d: NDArray[np.complexfloating]) -> NDArray[np.int16]:
    """Interleaved sc16 (``(2n,)`` int16) with saturation; NaN maps to 0."""
    z = np.asarray(d).astype(np.complex128)
    iq = np.empty(2 * z.shape[0], dtype=np.float64)
    iq[0::2] = z.real
    iq[1::2] = z.imag
    iq = np.nan_to_num(iq, nan=0.0, posinf=np.inf, neginf=-np.inf)
    codes = np.clip(np.rint(iq * SC16_SCALE), -32768.0, 32767.0)
    return codes.astype(np.int16)


def from_sc16(iq: NDArray[np.integer]) -> NDArray[np.complex64]:
    """complex64 from interleaved sc16 (``(2n,)`` or ``(n, 2)``), divided by 32767."""
    a = np.asarray(iq).reshape(-1)
    if a.shape[0] % 2:
        raise ConfigurationError("sc16 data needs an even number of int16 values")
    z = (a[0::2].astype(np.float64) + 1j * a[1::2].astype(np.float64)) / SC16_SCALE
    return z.astype(np.complex64)
