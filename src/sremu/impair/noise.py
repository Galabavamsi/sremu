"""Thermal noise per receiver (DESIGN.md sections 3.4, 5.12, 6.1; D20).

Module-level API (render contract, D20): ``noise(rx_key, k0, n, sigma)`` returns complex64
samples for absolute RX indices [k0, k0 + n). ``sigma`` is the TOTAL complex standard
deviation: ``sigma**2 = E|w|**2`` (so I and Q each have variance ``sigma**2/2``). For thermal
noise ``sigma**2 = k_B*T0*F*f_s`` W per receiver (:func:`thermal_sigma`; -113.975 dBm at
1 MS/s with F = 0 dB). ``rx_key`` is the pair of uint32 Philox key words of the receiver's
noise purpose, ``rng.derive_key(master_seed, "noise", <rx_stream>)``. The default backend is
``numba_cpu``; ``backend="numpy_ref"`` selects the NumPy twin.

The variates come from :mod:`sremu.rng` (scheme ``philox4x32-10/blake2b-v1``):
``rx_key = derive_key(master_seed, "noise", <rx_stream>)``, counter ``(k_lo, k_hi, 0, 0)``
with k the absolute sample index (lane 0), Box-Muller from the first two output words:
``u1 = (r0 + 1)*2**-32``, ``u2 = r1*2**-32``,
``w = sigma*sqrt(-ln u1)*(cos 2 pi u2 + j sin 2 pi u2)``. Every sample is a pure function of
(key, absolute index): chunking, skipping and thread count cannot change a bit (I2a, I4, D-6).
``noise_numba`` and ``noise_numpy`` share the bitwise-identical uint32 Philox stream (I5); the
normals agree to a few ulps (math-library ``log``/``cos``/``sin``).
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sremu import rng
from sremu.contracts import BackendName
from sremu.errors import ConfigurationError
from sremu.units import noise_power_w

__all__ = [
    "NOISE_LANE",
    "noise",
    "noise_numba",
    "noise_numpy",
    "normals_numba",
    "normals_numpy",
    "thermal_sigma",
]

NOISE_LANE: Final[int] = 0
"""Philox counter lane of thermal noise."""


def thermal_sigma(f_s: float, noise_figure_db: float = 0.0) -> float:
    """Complex standard deviation ``sqrt(k_B*T0*F*f_s)`` in sqrt(W) (section 5.12)."""
    return math.sqrt(noise_power_w(float(f_s), float(noise_figure_db)))


def noise(
    rx_key: tuple[int, int],
    k0: int,
    n: int,
    sigma: float,
    *,
    backend: BackendName = "numba_cpu",
) -> NDArray[np.complex64]:
    """Complex64 AWGN for absolute RX samples [k0, k0 + n) with ``E|w|**2 = sigma**2`` (D20).

    A pure function of ``(rx_key, absolute index)``: chunking, skipping and the thread count
    cannot change a bit (I2a, I4). ``backend`` selects ``numba_cpu`` (default) or the
    ``numpy_ref`` twin (same Philox words; normals equal to a few ulps, I5).
    """
    if backend == "numba_cpu":
        return noise_numba(rx_key, k0, n, sigma)
    if backend == "numpy_ref":
        return noise_numpy(rx_key, k0, n, sigma)
    raise ConfigurationError(f"unknown backend {backend!r} (numpy_ref | numba_cpu)")


def noise_numba(rx_key: tuple[int, int], k0: int, n: int, sigma: float) -> NDArray[np.complex64]:
    """Complex64 AWGN for absolute samples [k0, k0 + n) (numba, ``prange`` over samples)."""
    out = rng.complex_normal(rx_key, k0, n, sigma, sub=NOISE_LANE, backend="numba_cpu")
    return np.asarray(out, dtype=np.complex64)


def noise_numpy(rx_key: tuple[int, int], k0: int, n: int, sigma: float) -> NDArray[np.complex64]:
    """NumPy twin of :func:`noise_numba` (same Philox words, same Box-Muller)."""
    out = rng.complex_normal(rx_key, k0, n, sigma, sub=NOISE_LANE, backend="numpy_ref")
    return np.asarray(out, dtype=np.complex64)


def normals_numba(key: tuple[int, int], lane: int, k0: int, n: int) -> NDArray[np.float64]:
    """Standard normals for counters ``(k_lo, k_hi, lane, 0)`` (numba)."""
    return rng.standard_normal(key, k0, n, sub=lane, backend="numba_cpu")


def normals_numpy(key: tuple[int, int], lane: int, k0: int, n: int) -> NDArray[np.float64]:
    """Standard normals for counters ``(k_lo, k_hi, lane, 0)`` (NumPy twin)."""
    return rng.standard_normal(key, k0, n, sub=lane, backend="numpy_ref")
