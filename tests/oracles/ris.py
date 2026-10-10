"""Closed-form RIS oracles: panel geometry, the K1 kernel in its three published forms,
far-field closed forms, image theory, disk and plate references, focusing, legacy kernels.

Independent of ``sremu.physics`` (DESIGN.md sections 2.3, 10.1, 14.3). Written from the
equations of DESIGN.md section 5.3 and ported from ``research/scratch/pm_lib.py``.

K1 (section 5.3), summed over quadrature points q (one per physical cell for
``cell_model="point"``)::

    h = sum_q sqrt(G_T(r_i^) G_R(-r_s^)) * Gamma_n(q) * A_q * o(theta_i, theta_s)
              / (4 pi r_i r_s) * j * exp(-j k (r_i + r_s)) * 1[cos theta_i > 0, cos theta_s > 0]

with ``r_i^`` the unit vector Tx -> cell and ``r_s^`` the unit vector cell -> Rx. All complex
coefficients returned here INCLUDE the propagation phase (see ``tests/oracles/rf.py``).

Antenna patterns are callables ``g(u) -> power gain`` of an (M, 3) array of unit vectors
pointing from the antenna toward the far point (:func:`isotropic`, :func:`cos_power`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import exp1

from sremu.units import C0
from tests.oracles.rf import wavelength, wavenumber

__all__ = [
    "AntennaPattern",
    "Panel",
    "cos_power",
    "disk_kirchhoff_onaxis_h",
    "ellingson_area_gain_h",
    "ellingson_qgamma_h",
    "etsi_cell_gain_h",
    "etsi_farfield_focus_h_abs",
    "focus_phases",
    "isotropic",
    "k1_delays_s",
    "k1_frequency_response",
    "k1_h",
    "k1_terms",
    "legacy_v2_h",
    "los_h",
    "make_panel",
    "make_panel_extent",
    "plate_monostatic_h_abs",
    "po_plate_bound_h_abs",
    "rect_solid_angle_onaxis",
    "subdivide",
    "table_angles_rad",
    "tang_eq8_h_abs",
    "tang_eq12_h",
    "wcl_eq2_amplitude",
    "wcl_eq2_over_k1_power_ratio",
]

AntennaPattern = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def isotropic(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Isotropic power gain 1 in every direction."""
    return np.ones(np.asarray(u).shape[:-1])


def cos_power(alpha: float, boresight: ArrayLike) -> AntennaPattern:
    """``CosPower(alpha)``: power gain ``2*(alpha+1)*max(cos psi, 0)**alpha`` (section 5.2).

    ``psi`` is the angle between the direction and the unit ``boresight`` vector; the peak
    gain ``G_0 = 2(alpha+1)`` normalizes the pattern over the front hemisphere.
    """
    b = np.asarray(boresight, dtype=float)
    b = b / np.linalg.norm(b)
    g0 = 2.0 * (alpha + 1.0)

    def g(u: NDArray[np.float64]) -> NDArray[np.float64]:
        c = np.asarray(u) @ b
        return g0 * np.clip(c, 0.0, None) ** alpha

    return g


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Panel:
    """Quadrature points of a flat panel.

    ``cells``: (N, 3) positions in m. ``area``: (N,) pitch area per point, m^2.
    ``normal``/``u``/``v``: unit vectors (``v = normal x u``). ``iu``/``iv``: integer grid
    indices of each physical cell (cell order ``n = iv*n_u + iu``). ``parent``: (N,) index of
    the physical cell that carries the state (identity for physical-cell panels).
    """

    cells: NDArray[np.float64]
    area: NDArray[np.float64]
    normal: NDArray[np.float64]
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    iu: NDArray[np.int64]
    iv: NDArray[np.int64]
    parent: NDArray[np.int64]

    @property
    def n(self) -> int:
        """Number of quadrature points."""
        return int(self.cells.shape[0])


def _frame(normal: ArrayLike, u_axis: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    u = np.asarray(u_axis, dtype=float)
    u = u - (u @ n) * n
    u = u / np.linalg.norm(u)
    return n, u, np.cross(n, u)


def make_panel(
    center: ArrayLike,
    normal: ArrayLike,
    u_axis: ArrayLike,
    n_u: int,
    n_v: int,
    pitch_u: float,
    pitch_v: float,
    tiles: tuple[int, int] = (1, 1),
    gap_m: float = 0.0,
) -> Panel:
    """Physical cells of a panel of ``n_u x n_v`` cells, cell order ``n = iv*n_u + iu``.

    In-plane coordinates of cell (iu, iv), centred on ``center``::

        u = (iu - (n_u - 1)/2)*pitch_u + (tu - (T_u - 1)/2)*gap_m,  tu = iu // (n_u/T_u)

    and likewise for v: ``tiles = (T_u, T_v)`` physical tiles separated by an extra inert
    ``gap_m`` (so the cell spacing across a tile boundary is ``pitch + gap``; section 5.3).
    The quadrature area of every cell is its pitch area ``pitch_u*pitch_v``.
    """
    t_u, t_v = tiles
    if n_u % t_u or n_v % t_v:
        raise ValueError("cells per axis must be a multiple of the tile count")
    n, u, v = _frame(normal, u_axis)
    iu_g, iv_g = np.meshgrid(np.arange(n_u), np.arange(n_v))  # (n_v, n_u): row iv, column iu
    iu = iu_g.ravel().astype(np.int64)
    iv = iv_g.ravel().astype(np.int64)
    tu = iu // (n_u // t_u)
    tv = iv // (n_v // t_v)
    cu = (iu - (n_u - 1) / 2.0) * pitch_u + (tu - (t_u - 1) / 2.0) * gap_m
    cv = (iv - (n_v - 1) / 2.0) * pitch_v + (tv - (t_v - 1) / 2.0) * gap_m
    c = np.asarray(center, dtype=float)
    cells = c[None, :] + cu[:, None] * u[None, :] + cv[:, None] * v[None, :]
    area = np.full(cells.shape[0], pitch_u * pitch_v)
    return Panel(cells, area, n, u, v, iu, iv, np.arange(cells.shape[0], dtype=np.int64))


def make_panel_extent(
    center: ArrayLike,
    normal: ArrayLike,
    u_axis: ArrayLike,
    L_u: float,
    L_v: float,
    n_u: int,
    n_v: int,
) -> Panel:
    """Uniform panel of ``n_u x n_v`` cells covering ``L_u x L_v`` m (``pm_lib.make_panel``)."""
    return make_panel(center, normal, u_axis, n_u, n_v, L_u / n_u, L_v / n_v)


def subdivide(panel: Panel, pitch_u: float, pitch_v: float, max_sub_m: float) -> Panel:
    """``po_patch`` quadrature: split each cell into ``ceil(p_u/D) x ceil(p_v/D)`` sub-cells.

    Sub-cells share their parent's state (``parent``) and carry the sub-pitch area, so
    ``sum_{q in n} A_q = A_n`` (section 5.4).
    """
    m_u = math.ceil(pitch_u / max_sub_m - 1e-12)
    m_v = math.ceil(pitch_v / max_sub_m - 1e-12)
    su = (np.arange(m_u) - (m_u - 1) / 2.0) * (pitch_u / m_u)
    sv = (np.arange(m_v) - (m_v - 1) / 2.0) * (pitch_v / m_v)
    gu, gv = np.meshgrid(su, sv)
    off = gu.ravel()[:, None] * panel.u[None, :] + gv.ravel()[:, None] * panel.v[None, :]
    cells = (panel.cells[:, None, :] + off[None, :, :]).reshape(-1, 3)
    k = m_u * m_v
    area = np.repeat(panel.area / k, k)
    parent = np.repeat(panel.parent, k)
    return Panel(
        cells,
        area,
        panel.normal,
        panel.u,
        panel.v,
        np.repeat(panel.iu, k),
        np.repeat(panel.iv, k),
        parent,
    )


# --------------------------------------------------------------------------- K1


@dataclass(frozen=True)
class _Geom:
    r_i: NDArray[np.float64]
    r_s: NDArray[np.float64]
    cos_i: NDArray[np.float64]
    cos_s: NDArray[np.float64]
    u_tx: NDArray[np.float64]  # unit Tx -> cell
    u_rx: NDArray[np.float64]  # unit Rx -> cell


def _geometry(tx: ArrayLike, rx: ArrayLike, panel: Panel) -> _Geom:
    t = np.asarray(tx, dtype=float)
    r = np.asarray(rx, dtype=float)
    vi = t[None, :] - panel.cells  # cell -> Tx
    vs = r[None, :] - panel.cells  # cell -> Rx
    r_i = np.sqrt(np.einsum("ij,ij->i", vi, vi))
    r_s = np.sqrt(np.einsum("ij,ij->i", vs, vs))
    cos_i = (vi @ panel.normal) / r_i
    cos_s = (vs @ panel.normal) / r_s
    return _Geom(r_i, r_s, cos_i, cos_s, -vi / r_i[:, None], -vs / r_s[:, None])


def k1_terms(
    tx: ArrayLike,
    rx: ArrayLike,
    panel: Panel,
    fc_hz: float,
    gamma: ArrayLike = 1.0,
    *,
    obliquity: str = "geo",
    huygens_j: bool = True,
    g_t: AntennaPattern = isotropic,
    g_r: AntennaPattern = isotropic,
) -> NDArray[np.complex128]:
    """Per-quadrature-point K1 terms (summing them gives :func:`k1_h`).

    ``gamma``: scalar or (N_phys,) complex per PHYSICAL cell (indexed through
    ``panel.parent``). ``obliquity``: ``"geo"`` = sqrt(cos_i cos_s) (default, Tang) or
    ``"kirchhoff"`` = (cos_i + cos_s)/2. Back-side terms are exactly 0 (PH-09).
    """
    g = _geometry(tx, rx, panel)
    k = wavenumber(fc_hz)
    vis = (g.cos_i > 0.0) & (g.cos_s > 0.0)
    ci = np.where(vis, g.cos_i, 0.0)
    cs = np.where(vis, g.cos_s, 0.0)
    if obliquity == "geo":
        o = np.sqrt(ci * cs)
    elif obliquity == "kirchhoff":
        o = 0.5 * (ci + cs)
    else:
        raise ValueError(obliquity)
    gam = np.asarray(gamma, dtype=complex)
    gam_q = gam[panel.parent] if gam.ndim else np.full(panel.n, complex(gam))
    ant = np.sqrt(g_t(g.u_tx) * g_r(g.u_rx))
    jf = 1j if huygens_j else 1.0
    amp = ant * gam_q * panel.area * o / (4.0 * math.pi * g.r_i * g.r_s)
    return np.where(vis, jf * amp * np.exp(-1j * k * (g.r_i + g.r_s)), 0.0 + 0.0j)


def k1_h(
    tx: ArrayLike, rx: ArrayLike, panel: Panel, fc_hz: float, gamma: ArrayLike = 1.0, **kw
) -> complex:
    """K1 RIS-path coefficient (section 5.3), summed in f64 (see :func:`k1_terms`)."""
    return complex(np.sum(k1_terms(tx, rx, panel, fc_hz, gamma, **kw)))


def table_angles_rad(
    tx: ArrayLike, rx: ArrayLike, panel: Panel
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Signed response-table angles ``(theta_i, theta_s)`` per quadrature point (section 3.1).

    Measured from the normal in the (u, normal) plane after projecting out the v component
    (section 5.5 off-plane rule), signed by the u component; ``theta_i`` points from the cell
    toward the Tx and ``theta_s`` from the cell toward the Rx, so specular is
    ``theta_s = -theta_i``. Back-side directions fall outside [-pi/2, pi/2].
    """
    t = np.asarray(tx, dtype=float)
    r = np.asarray(rx, dtype=float)
    vi = t[None, :] - panel.cells
    vs = r[None, :] - panel.cells
    th_i = np.arctan2(vi @ panel.u, vi @ panel.normal)
    th_s = np.arctan2(vs @ panel.u, vs @ panel.normal)
    return th_i, th_s


def k1_delays_s(tx: ArrayLike, rx: ArrayLike, panel: Panel) -> NDArray[np.float64]:
    """Per-quadrature-point propagation delay ``(r_i + r_s)/C0`` in s (f64)."""
    g = _geometry(tx, rx, panel)
    return (g.r_i + g.r_s) / C0


def k1_frequency_response(
    tx: ArrayLike,
    rx: ArrayLike,
    panel: Panel,
    fc_hz: float,
    f_offset_hz: ArrayLike,
    gamma: ArrayLike = 1.0,
    **kw,
) -> NDArray[np.complex128]:
    """Exact wideband K1 response ``H(f) = sum_q a_q exp(-j 2 pi (f_c + f) tau_q)``.

    ``a_q`` are the K1 amplitudes at the carrier (frequency-flat Gamma and cell model, as in
    section 5.3) and ``tau_q = (r_iq + r_sq)/C0``; ``f`` is the baseband offset from ``f_c``.
    At ``f = 0`` this equals :func:`k1_h`. Reference for the wideband/tile rendering checks
    (section 4.5) and for ``pm_checks.py`` T13.
    """
    terms = k1_terms(tx, rx, panel, fc_hz, gamma, **kw)  # a_q exp(-j 2 pi f_c tau_q)
    tau = k1_delays_s(tx, rx, panel)
    f = np.asarray(f_offset_hz, dtype=float)
    return np.exp(-2j * math.pi * f[..., None] * tau) @ terms


def los_h(
    tx: ArrayLike,
    rx: ArrayLike,
    fc_hz: float,
    g_t: AntennaPattern = isotropic,
    g_r: AntennaPattern = isotropic,
) -> complex:
    """LOS coefficient ``sqrt(G_T(r^) G_R(-r^)) lambda/(4 pi d) exp(-jkd)`` (section 5.2)."""
    t = np.asarray(tx, dtype=float)
    r = np.asarray(rx, dtype=float)
    d = float(np.linalg.norm(r - t))
    u = ((r - t) / d)[None, :]
    gain = float(np.sqrt(g_t(u) * g_r(-u))[0])
    return complex(
        gain * wavelength(fc_hz) / (4.0 * math.pi * d) * np.exp(-1j * wavenumber(fc_hz) * d)
    )


# The three published forms of the same model (PH-02). Each is written in its own paper's
# variables so that an algebra slip in one of them shows up as a mismatch.


def tang_eq12_h(
    tx: ArrayLike,
    rx: ArrayLike,
    panel: Panel,
    fc_hz: float,
    gamma: ArrayLike = 1.0,
    g_t: AntennaPattern = isotropic,
    g_r: AntennaPattern = isotropic,
) -> complex:
    """Tang et al. 2022 Eq. (12), summed over cells, times the Huygens j of K1.

    Per cell: ``sqrt(G_t G_r) * dx*dy * sqrt(F) * |Gamma| / (4 pi r_t r_r) *
    exp(-j(k(r_t + r_r) - arg Gamma))`` with the cell power pattern
    ``F = cos(theta_t) cos(theta_r)``.
    """
    g = _geometry(tx, rx, panel)
    k = wavenumber(fc_hz)
    gam = np.asarray(gamma, dtype=complex)
    gam_q = gam[panel.parent] if gam.ndim else np.full(panel.n, complex(gam))
    F = np.clip(g.cos_i, 0.0, None) * np.clip(g.cos_s, 0.0, None)
    gtgr = g_t(g.u_tx) * g_r(g.u_rx)
    s = (
        np.sqrt(gtgr)
        * panel.area
        * np.sqrt(F)
        * np.abs(gam_q)
        / (4.0 * math.pi * g.r_i * g.r_s)
        * np.exp(-1j * (k * (g.r_i + g.r_s) - np.angle(gam_q)))
    )
    return complex(1j * np.sum(s))


def etsi_cell_gain_h(
    tx: ArrayLike,
    rx: ArrayLike,
    panel: Panel,
    fc_hz: float,
    gamma: ArrayLike = 1.0,
    g_t: AntennaPattern = isotropic,
    g_r: AntennaPattern = isotropic,
) -> complex:
    """ETSI GR RIS 003 V1.2.1 section 6.3.1.2 two-hop form with the recommended cell gain.

    Each cell is a receive-then-reradiate element with gain pattern
    ``G_cell(theta) = (4 pi dx dy / lambda^2) cos(theta)``; per cell
    ``(lambda/(4 pi r_i)) sqrt(G_t G_cell(theta_i)) * Gamma * sqrt(G_cell(theta_s) G_r)
    (lambda/(4 pi r_s)) exp(-jk(r_i + r_s))``, times the Huygens j.
    """
    g = _geometry(tx, rx, panel)
    lam = wavelength(fc_hz)
    k = wavenumber(fc_hz)
    gam = np.asarray(gamma, dtype=complex)
    gam_q = gam[panel.parent] if gam.ndim else np.full(panel.n, complex(gam))
    g_cell0 = 4.0 * math.pi * panel.area / lam**2
    gci = g_cell0 * np.clip(g.cos_i, 0.0, None)
    gcs = g_cell0 * np.clip(g.cos_s, 0.0, None)
    hop1 = lam / (4.0 * math.pi * g.r_i) * np.sqrt(g_t(g.u_tx) * gci)
    hop2 = lam / (4.0 * math.pi * g.r_s) * np.sqrt(gcs * g_r(g.u_rx))
    return complex(1j * np.sum(hop1 * gam_q * hop2 * np.exp(-1j * k * (g.r_i + g.r_s))))


def ellingson_area_gain_h(
    tx: ArrayLike,
    rx: ArrayLike,
    panel: Panel,
    fc_hz: float,
    gamma: ArrayLike = 1.0,
    g_t: AntennaPattern = isotropic,
    g_r: AntennaPattern = isotropic,
) -> complex:
    """Ellingson (arXiv:1912.06759) Eq. (10) with the area-tied element gain.

    Per element power ``P_T G_T G_R (lambda/4 pi)^4 G_e(psi_i) G_e(psi_s) / (r_i^2 r_s^2)``
    with ``G_e(psi) = (4 pi A_n / lambda^2) cos psi``; the field is its square root with the
    phase ``exp(-jk(r_i + r_s)) * Gamma``, times the Huygens j.
    """
    g = _geometry(tx, rx, panel)
    lam = wavelength(fc_hz)
    k = wavenumber(fc_hz)
    gam = np.asarray(gamma, dtype=complex)
    gam_q = gam[panel.parent] if gam.ndim else np.full(panel.n, complex(gam))
    ge_i = 4.0 * math.pi * panel.area / lam**2 * np.clip(g.cos_i, 0.0, None)
    ge_s = 4.0 * math.pi * panel.area / lam**2 * np.clip(g.cos_s, 0.0, None)
    p = (
        g_t(g.u_tx)
        * g_r(g.u_rx)
        * (lam / (4.0 * math.pi)) ** 4
        * ge_i
        * ge_s
        / (g.r_i**2 * g.r_s**2)
    )
    return complex(1j * np.sum(np.sqrt(p) * gam_q * np.exp(-1j * k * (g.r_i + g.r_s))))


def focus_phases(
    tx: ArrayLike, rx: ArrayLike, panel: Panel, fc_hz: float, huygens_j: bool = True
) -> NDArray:
    """Per-cell Gamma that co-phases every K1 term: ``exp(+jk(r_i + r_s)) * (-j)``."""
    g = _geometry(tx, rx, panel)
    ph = np.exp(1j * wavenumber(fc_hz) * (g.r_i + g.r_s))
    return ph * (-1j if huygens_j else 1.0)


# --------------------------------------------------------------------------- closed forms


def tang_eq8_h_abs(
    n_cells: int,
    cell_area_m2: float,
    d1_m: float,
    d2_m: float,
    theta_t_rad: float,
    theta_r_rad: float,
    gamma_abs: float = 1.0,
    g_t: float = 1.0,
    g_r: float = 1.0,
) -> float:
    """Tang 2022 Eq. (8) far-field beamforming closed form (PH-03), as an amplitude.

    ``PL = 16 pi^2 (d1 d2)^2 / (G_t G_r (N A)^2 cos(theta_t) cos(theta_r) |Gamma|^2)``;
    returns ``|h| = 1/sqrt(PL)``.
    """
    pl = (16.0 * math.pi**2 * (d1_m * d2_m) ** 2) / (
        g_t
        * g_r
        * (n_cells * cell_area_m2) ** 2
        * math.cos(theta_t_rad)
        * math.cos(theta_r_rad)
        * gamma_abs**2
    )
    return 1.0 / math.sqrt(pl)


def etsi_farfield_focus_h_abs(
    n_cells: int, dx_m: float, dy_m: float, d_t_m: float, d_r_m: float, a: float = 1.0
) -> float:
    """ETSI GR RIS 003 far-field path gain at the focus: ``N dx dy A / (4 pi d_t d_r)`` (G7).

    With unit antenna gains and unit cell pattern (PL = (4 pi d_t d_r/(N dx dy A))^2).
    """
    return n_cells * dx_m * dy_m * a / (4.0 * math.pi * d_t_m * d_r_m)


def po_plate_bound_h_abs(area_m2: float, d_t_m: float, d_r_m: float) -> float:
    """Physical-optics bound of a perfectly phased flat plate: ``A/(4 pi d_t d_r)`` (G7)."""
    return area_m2 / (4.0 * math.pi * d_t_m * d_r_m)


def plate_monostatic_h_abs(area_m2: float, d_m: float) -> float:
    """Monostatic far-field plate at normal incidence: ``A/(4 pi d^2)`` (PH-06).

    Equal to the radar equation with ``sigma = 4 pi A^2/lambda^2``.
    """
    return area_m2 / (4.0 * math.pi * d_m * d_m)


def rect_solid_angle_onaxis(a_m: float, b_m: float, d_m: float) -> float:
    """Solid angle of a ``2a x 2b`` rectangle seen from distance d on its axis."""
    return 4.0 * math.atan(a_m * b_m / (d_m * math.sqrt(a_m * a_m + b_m * b_m + d_m * d_m)))


def disk_kirchhoff_onaxis_h(d_m: float, R_m: float, fc_hz: float) -> complex:
    """Exact Kirchhoff monostatic on-axis response of a uniformly reflecting disk (PH-05).

    Obliquity ``(cos_i + cos_s)/2 = d/r``, Tx = Rx at height d over a disk of radius R:
    ``h = (j/(4 pi)) * 2 pi d * [E2(j 2kd)/d - E2(j 2k r_max)/r_max]`` with
    ``E2(z) = exp(-z) - z E1(z)`` and ``r_max = hypot(d, R)``. Includes the Huygens j.
    """
    k = wavenumber(fc_hz)
    r_max = math.hypot(d_m, R_m)

    def e2(z: complex) -> complex:
        return complex(np.exp(-z) - z * exp1(z))

    integral = e2(2j * k * d_m) / d_m - e2(2j * k * r_max) / r_max
    return 1j / (4.0 * math.pi) * 2.0 * math.pi * d_m * integral


# --------------------------------------------------------------------------- legacy (teeth)


def legacy_v2_h(
    tx: ArrayLike, rx: ArrayLike, panel: Panel, fc_hz: float, gamma: ArrayLike = 1.0
) -> complex:
    """v2 per-element kernel ``(lambda/4 pi)^2 sqrt(cos_i cos_s)/(r_i r_s)`` (no area, no j).

    Teeth of PH-04: it gains +12.04 dB per 4x refinement of a fixed aperture.
    """
    g = _geometry(tx, rx, panel)
    lam = wavelength(fc_hz)
    k = wavenumber(fc_hz)
    gam = np.asarray(gamma, dtype=complex)
    gam_q = gam[panel.parent] if gam.ndim else np.full(panel.n, complex(gam))
    vis = (g.cos_i > 0) & (g.cos_s > 0)
    o = np.sqrt(np.clip(g.cos_i, 0, None) * np.clip(g.cos_s, 0, None))
    t = gam_q * (lam / (4 * math.pi)) ** 2 * o / (g.r_i * g.r_s) * np.exp(-1j * k * (g.r_i + g.r_s))
    return complex(np.sum(np.where(vis, t, 0.0)))


def ellingson_qgamma_h(
    tx: ArrayLike,
    rx: ArrayLike,
    panel: Panel,
    fc_hz: float,
    gamma: ArrayLike = 1.0,
    q: float = 0.285,
    gamma_q: float | None = None,
) -> complex:
    """Ellingson Eq. (10) with the area-free element gain ``G_e = gamma_q cos^(2q)`` (legacy).

    Per element ``(lambda/4 pi)^2 sqrt(G_e(theta_i) G_e(theta_s)) / (r_i r_s)``, no j, with
    ``gamma_q = 2(2q + 1)`` by default (``pm_lib.ris_h(model="ellingson")``; the
    ``ellingson_qgamma`` row of section 5.14). Not area-tied, so it does not converge under
    aperture refinement.
    """
    g = _geometry(tx, rx, panel)
    lam = wavelength(fc_hz)
    k = wavenumber(fc_hz)
    gq = 2.0 * (2.0 * q + 1.0) if gamma_q is None else gamma_q
    gam = np.asarray(gamma, dtype=complex)
    gam_q = gam[panel.parent] if gam.ndim else np.full(panel.n, complex(gam))
    vis = (g.cos_i > 0) & (g.cos_s > 0)
    ge_i = gq * np.clip(g.cos_i, 0, None) ** (2 * q)
    ge_s = gq * np.clip(g.cos_s, 0, None) ** (2 * q)
    t = (
        gam_q
        * (lam / (4 * math.pi)) ** 2
        * np.sqrt(ge_i * ge_s)
        / (g.r_i * g.r_s)
        * np.exp(-1j * k * (g.r_i + g.r_s))
    )
    return complex(np.sum(np.where(vis, t, 0.0)))


def wcl_eq2_amplitude(
    lam_m: float, w_m: float, h_m: float, d_t_m: ArrayLike, d_r_m: ArrayLike, q: float = 0.285
) -> NDArray:
    """Per-element amplitude of the WCL letter's Eq. (2)/(3) (``P_T = G = eps = 1``).

    ``sqrt((lambda/4)^2 (w^2)^2 h^{4q} / (d_t d_r)^{2+2q})`` (ported from pv_checks.py).
    """
    dt = np.asarray(d_t_m, dtype=float)
    dr = np.asarray(d_r_m, dtype=float)
    return np.sqrt(
        (lam_m / 4.0) ** 2 * (w_m * w_m) ** 2 * h_m ** (4 * q) / (dt * dr) ** (2 + 2 * q)
    )


def wcl_eq2_over_k1_power_ratio(
    lam_m: float, cos_i: ArrayLike, cos_s: ArrayLike, q: float = 0.285
) -> NDArray:
    """Per-element power ratio of the WCL Eq. (2) kernel to K1.

    ``(pi lambda)^2 (cos_i cos_s)^{2q-1}``.

    Used by the V-WCL W1 teeth (``legacy_wcl_eq2``); lambda in metres, as in the letter.
    """
    c = np.asarray(cos_i, dtype=float) * np.asarray(cos_s, dtype=float)
    return (math.pi * lam_m) ** 2 * c ** (2.0 * q - 1.0)
