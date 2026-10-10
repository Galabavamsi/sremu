"""Independent closed forms for the WP2 physics tests (no ``sremu.physics`` imports).

Ported from the oracle sources (``research/scratch/pm_lib.py`` and ``pm_checks.py``) as
plain formulas; only ``sremu.units.C0`` is shared (section 3.1: the only c).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import exp1

from sremu.units import C0


def wavelength(fc_hz: float) -> float:
    return C0 / fc_hz


def db20(x: complex | float) -> float:
    return 20.0 * math.log10(abs(x))


def make_panel(lu: float, lv: float, nu: int, nv: int, z: float = 0.0) -> tuple[np.ndarray, float]:
    """Cell centres of a panel in the z = ``z`` plane (normal +z, u = x, v = y), ordered
    ``n = iv * nu + iu``, and the pitch area."""
    du, dv = lu / nu, lv / nv
    iu = (np.arange(nu) - (nu - 1) / 2) * du
    iv = (np.arange(nv) - (nv - 1) / 2) * dv
    gv, gu = np.meshgrid(iv, iu, indexing="ij")
    pos = np.stack([gu.ravel(), gv.ravel(), np.full(gu.size, z)], axis=1)
    return pos, du * dv


def k1_point_sum(
    tx, rx, pos, area, fc_hz, gamma=1.0, *, obliquity="sqrt", gt=None, gr=None, with_j=True
) -> complex:
    """K1 for point cells on the z = 0 plane with normal +z, written out term by term."""
    k = 2 * math.pi / wavelength(fc_hz)
    vi = np.asarray(tx, float)[None, :] - pos
    vs = np.asarray(rx, float)[None, :] - pos
    ri = np.linalg.norm(vi, axis=1)
    rs = np.linalg.norm(vs, axis=1)
    ci = vi[:, 2] / ri
    cs = vs[:, 2] / rs
    vis = (ci > 0) & (cs > 0)
    if obliquity == "sqrt":
        o = np.sqrt(np.clip(ci, 0, None) * np.clip(cs, 0, None))
    else:
        o = 0.5 * (ci + cs)
    g_t = np.ones_like(ri) if gt is None else gt
    g_r = np.ones_like(ri) if gr is None else gr
    a = np.sqrt(g_t * g_r) * area * o / (4 * math.pi * ri * rs) * vis
    g = np.broadcast_to(np.asarray(gamma, complex), ri.shape)
    j = 1j if with_j else 1.0
    return complex(np.sum(j * g * a * np.exp(-1j * k * (ri + rs))))


def disk_kirchhoff_onaxis(d: float, radius: float, fc_hz: float) -> complex:
    """Exact Kirchhoff-obliquity monostatic on-axis response of a Gamma = 1 disk (radius R,
    Tx = Rx at height d): ``(j/4pi) 2 pi d [E2(j2kd)/d - E2(j2k r_max)/r_max]``,
    ``E2(z) = exp(-z) - z E1(z)``."""
    k = 2 * math.pi / wavelength(fc_hz)
    rmax = math.hypot(d, radius)

    def e2(z: complex) -> complex:
        return complex(np.exp(-z) - z * exp1(z))

    integral = e2(2j * k * d) / d - e2(2j * k * rmax) / rmax
    return 1j / (4 * math.pi) * 2 * math.pi * d * integral


def rect_solid_angle_onaxis(a: float, b: float, d: float) -> float:
    """Solid angle of a 2a x 2b rectangle seen on its axis at distance d."""
    return 4 * math.atan(a * b / (d * math.sqrt(a * a + b * b + d * d)))


def subset_sums(z0: complex, z: np.ndarray) -> np.ndarray:
    """All ``z0 + sum_n Z[s_n, n]`` over s in {0..S-1}^N (brute force by doubling)."""
    sums = np.array([complex(z0)])
    for n in range(z.shape[1]):
        sums = (sums[None, :] + z[:, n][:, None]).ravel()
    return sums
