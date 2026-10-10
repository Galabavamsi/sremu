"""Legacy (non-physical) kernels for teeth tests only (DESIGN.md section 5.14; WP2).

Every item raises ``NonPhysicalError`` unless called with ``allow_nonphysical=True``.

Items (formula per cell; teeth: the legacy item must FAIL the physical guard):

* ``legacy_v1``: v1 ``total_nlos_gain``, whole-panel area in every cell, ``q = 0.285``,
  ``abs(cos)``, v1's own speed of light ``units.C_LEGACY_V1 = 3e8`` (D4). Teeth:
  +24.39 +- 0.05 dB vs K1 in the v1 default geometry.
* ``legacy_v2``: ``(lambda/4pi)^2 sqrt(cos cos)/(r_i r_s)`` (no area, no j). Teeth:
  +12.04 dB per 4x refinement.
* ``legacy_wcl_eq2``: WCL letter Eq. (2)/(3), ``(lambda/4) A (cos cos)^q/(r_i r_s)``. Teeth:
  power ratio to K1 ``(pi lambda)^2 (cos cos)^(2q-1)``.
* ``ellingson_qgamma``: Ellingson Eq. (10), ``G_e = gamma_q cos^(2q)`` (area-free).
* ``no_j``: K1 without the Huygens ``j``. Teeth: phase -90 deg vs image theory.
* ``block_constant_delay``: v2 staircase geometry (delay held per block). Teeth: the
  -148.3 Hz alias instead of +350.2 Hz.

Speed of light (D4): ``legacy_v1`` reproduces v1 exactly, including v1's rounded
``c = 3e8``, through ``units.C_LEGACY_V1``. This module is the only user of that constant
(``tests/unit/test_layering.py`` enforces it); every other kernel here, and K1, uses
``units.C0``. With ``C0`` instead, the v1 default-geometry gap would be 24.48 dB.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from sremu.errors import NonPhysicalError
from sremu.physics.antennas import Antenna
from sremu.physics.ris_kernel import Obliquity, k1_channel
from sremu.units import C0, C_LEGACY_V1

__all__ = [
    "LEGACY_ITEMS",
    "WCL_Q",
    "block_constant_delay",
    "ellingson_qgamma",
    "legacy_v1",
    "legacy_v2",
    "legacy_wcl_eq2",
    "no_j",
]

WCL_Q = 0.285
"""The cos exponent q of v1 and of the WCL letter."""

LEGACY_ITEMS = (
    "legacy_v1",
    "legacy_v2",
    "legacy_wcl_eq2",
    "ellingson_qgamma",
    "no_j",
    "block_constant_delay",
)


def _guard(allow: bool, name: str) -> None:
    if not allow:
        raise NonPhysicalError(
            f"{name} is a legacy non-physical kernel for teeth tests; "
            "call it with allow_nonphysical=True"
        )


def _geom(tx: np.ndarray, rx: np.ndarray, pts: np.ndarray, n: np.ndarray):  # type: ignore[no-untyped-def]
    vi = np.asarray(tx, np.float64)[None, :] - pts
    vs = np.asarray(rx, np.float64)[None, :] - pts
    ri = np.sqrt(np.sum(vi * vi, axis=1))
    rs = np.sqrt(np.sum(vs * vs, axis=1))
    return ri, rs, (vi @ n) / ri, (vs @ n) / rs


def legacy_v1(
    tx: Sequence[float],
    rx: Sequence[float],
    points: np.ndarray,
    panel_area_m2: float,
    normal: Sequence[float],
    fc_hz: float,
    bn: np.ndarray | None = None,
    *,
    allow_nonphysical: bool = False,
) -> complex:
    """v1 ``total_nlos_gain`` transcription: ``p = (q A_panel c)^2/(4 pi f)^2 * pi
    |ci|^(2q) |cs|^(2q)/(ri^2 rs^2)``, summed as ``bn sqrt(p) exp(-j 2 pi f (ri+rs)/c)`` with
    the WHOLE panel area in every cell and ``abs(cos)`` (back side admitted). ``c`` is v1's
    ``units.C_LEGACY_V1 = 3e8`` in both the amplitude and the phase (D4)."""
    _guard(allow_nonphysical, "legacy_v1")
    c_v1 = C_LEGACY_V1
    pts = np.asarray(points, np.float64)
    ri, rs, ci, cs = _geom(np.asarray(tx), np.asarray(rx), pts, np.asarray(normal, np.float64))
    q = WCL_Q
    p = (q * panel_area_m2 * c_v1) ** 2 / (4 * math.pi * fc_hz) ** 2
    p = p * math.pi * np.abs(ci) ** (2 * q) * np.abs(cs) ** (2 * q) / (ri**2 * rs**2)
    b = np.exp(1j * math.pi / 4) * np.ones_like(ri) if bn is None else np.asarray(bn)
    return complex(np.sum(b * np.sqrt(p) * np.exp(-1j * 2 * math.pi * fc_hz * (ri + rs) / c_v1)))


def legacy_v2(
    tx: Sequence[float],
    rx: Sequence[float],
    points: np.ndarray,
    normal: Sequence[float],
    fc_hz: float,
    gamma: np.ndarray | complex = 1.0,
    *,
    allow_nonphysical: bool = False,
) -> complex:
    """v2 ``engine.py``: ``(lambda/4 pi)^2 sqrt(ci cs)/(ri rs) Gamma exp(-jk(ri+rs))``,
    area-free."""
    _guard(allow_nonphysical, "legacy_v2")
    pts = np.asarray(points, np.float64)
    ri, rs, ci, cs = _geom(np.asarray(tx), np.asarray(rx), pts, np.asarray(normal, np.float64))
    lam = C0 / fc_hz
    k = 2 * math.pi / lam
    vis = (ci > 0) & (cs > 0)
    amp = (lam / (4 * math.pi)) ** 2 * np.sqrt(np.where(vis, ci * cs, 0.0)) / (ri * rs)
    g = np.broadcast_to(np.asarray(gamma, np.complex128), ri.shape)
    return complex(np.sum(g * amp * np.exp(-1j * k * (ri + rs))))


def legacy_wcl_eq2(
    tx: Sequence[float],
    rx: Sequence[float],
    points: np.ndarray,
    cell_area_m2: float,
    normal: Sequence[float],
    fc_hz: float,
    gamma: np.ndarray | complex = 1.0,
    q: float = WCL_Q,
    *,
    allow_nonphysical: bool = False,
) -> complex:
    """WCL letter Eq. (2)/(3) per-element amplitude ``(lambda/4) A (ci cs)^q / (ri rs)``.

    Its power ratio to K1 (isotropic antennas, default obliquity) is
    ``(pi lambda)^2 (ci cs)^(2q-1)`` per element (lambda in metres: the letter's expression
    is not dimensionless).
    """
    _guard(allow_nonphysical, "legacy_wcl_eq2")
    pts = np.asarray(points, np.float64)
    ri, rs, ci, cs = _geom(np.asarray(tx), np.asarray(rx), pts, np.asarray(normal, np.float64))
    lam = C0 / fc_hz
    k = 2 * math.pi / lam
    vis = (ci > 0) & (cs > 0)
    amp = (lam / 4.0) * cell_area_m2 * np.where(vis, ci * cs, 0.0) ** q / (ri * rs)
    g = np.broadcast_to(np.asarray(gamma, np.complex128), ri.shape)
    return complex(np.sum(g * amp * np.exp(-1j * k * (ri + rs))))


def ellingson_qgamma(
    tx: Sequence[float],
    rx: Sequence[float],
    points: np.ndarray,
    normal: Sequence[float],
    fc_hz: float,
    gamma: np.ndarray | complex = 1.0,
    q: float = WCL_Q,
    gamma_q: float | None = None,
    *,
    allow_nonphysical: bool = False,
) -> complex:
    """Ellingson Eq. (10) with ``G_e(psi) = gamma_q cos^(2q) psi`` (``gamma_q = 2(2q+1)``):
    ``(lambda/4 pi)^2 sqrt(G_e(psi_i) G_e(psi_s)) / (ri rs)`` (area-free element gain)."""
    _guard(allow_nonphysical, "ellingson_qgamma")
    pts = np.asarray(points, np.float64)
    ri, rs, ci, cs = _geom(np.asarray(tx), np.asarray(rx), pts, np.asarray(normal, np.float64))
    lam = C0 / fc_hz
    k = 2 * math.pi / lam
    gq = 2 * (2 * q + 1) if gamma_q is None else gamma_q
    vis = (ci > 0) & (cs > 0)
    ge_i = gq * np.clip(ci, 0, None) ** (2 * q)
    ge_s = gq * np.clip(cs, 0, None) ** (2 * q)
    amp = np.where(vis, (lam / (4 * math.pi)) ** 2 * np.sqrt(ge_i * ge_s) / (ri * rs), 0.0)
    g = np.broadcast_to(np.asarray(gamma, np.complex128), ri.shape)
    return complex(np.sum(g * amp * np.exp(-1j * k * (ri + rs))))


def no_j(
    tx: Sequence[float],
    rx: Sequence[float],
    points: np.ndarray,
    areas: np.ndarray | float,
    normal: Sequence[float],
    fc_hz: float,
    gamma: np.ndarray | complex = 1.0,
    *,
    ant_tx: Antenna | None = None,
    ant_rx: Antenna | None = None,
    obliquity: Obliquity = "sqrt",
    allow_nonphysical: bool = False,
) -> complex:
    """K1 without the Huygens ``j`` (phase off by -90 deg relative to image theory)."""
    _guard(allow_nonphysical, "no_j")
    return k1_channel(
        tx,
        rx,
        points,
        areas,
        normal,
        fc_hz,
        gamma,
        ant_tx=ant_tx,
        ant_rx=ant_rx,
        obliquity=obliquity,
        huygens_j=False,
    )


def block_constant_delay(
    tau_fn: Callable[[np.ndarray], np.ndarray],
    t_s: np.ndarray,
    block_s: float,
    *,
    allow_nonphysical: bool = False,
) -> np.ndarray:
    """v2 staircase geometry: ``tau(floor(t / T_b) T_b)`` (the -148.3 Hz alias teeth)."""
    _guard(allow_nonphysical, "block_constant_delay")
    t = np.asarray(t_s, dtype=np.float64)
    return np.asarray(tau_fn(np.floor(t / block_s) * block_s), dtype=np.float64)
