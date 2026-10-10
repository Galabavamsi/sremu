"""RIS kernel K1, the only physical RIS kernel (DESIGN.md section 5.3; WP2).

.. math::

   h_{RIS} = \\sum_q \\sqrt{G_T(\\hat r^i_q) G_R(-\\hat r^s_q)}\\, \\Gamma_{n(q)}(s, f,
   \\theta_{iq}, \\theta_{sq})\\, \\frac{A_q\\, o(\\theta_{iq}, \\theta_{sq})}
   {4\\pi r_{iq} r_{sq}}\\, j\\, e^{-jk(r_{iq} + r_{sq})}\\,
   \\mathbb 1[\\cos\\theta_{iq} > 0, \\cos\\theta_{sq} > 0]

* ``A_q``: pitch area of the quadrature cell (gaps included); sum over a cell's sub-cells is
  the cell's pitch area.
* ``o``: obliquity, default ``sqrt(cos theta_i cos theta_s)`` (``"sqrt"``, Tang 2022's
  measured cos pattern); option ``(cos theta_i + cos theta_s)/2`` (``"kirchhoff"``);
  ``"none"`` (o = 1) only for tables with ``includes_cell_pattern=True``.
* ``j``: Huygens / physical-optics factor, always on (``huygens_j=False`` exists only for
  the ``no_j`` legacy teeth kernel in :mod:`sremu.physics.legacy`).
* Visibility: back-side incidence or reflection contributes exactly 0.

Two backends compute the per-quadrature-point terms and the per-cell accumulation:
``numpy_ref`` (float64 reference) and ``numba_cpu`` (parallel over rows, sequential and
fixed-order over quadrature points, no fastmath). Both compute distances as
``sqrt((dx*dx + dy*dy) + dz*dz)`` and cosines as ``((dx*nx + dy*ny) + dz*nz)/r`` so that
they agree to <= 1e-12 relative.

Phases: the terms carry ``exp(-j k (r_i + r_s - L_ref))``; ``L_ref`` (m) is the path length
through the cell's tile reference point (``L_ref = C0 * tau_tile``), so the coefficient
phase is RELATIVE to the tile reference as the contract requires. ``L_ref = 0`` gives the
absolute phase used by the closed-form oracles.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numba
import numpy as np

from sremu.errors import ConfigurationError, GeometryError
from sremu.physics.antennas import Antenna, Isotropic
from sremu.units import C0

__all__ = [
    "BACKENDS",
    "OBLIQUITIES",
    "Backend",
    "Obliquity",
    "accumulate_cells",
    "energy_budget",
    "farfield_focused_amplitude",
    "fraunhofer_distance_m",
    "k1_channel",
    "k1_terms",
    "obliquity_code",
    "tile_sums",
]

Obliquity = Literal["sqrt", "kirchhoff", "none"]
Backend = Literal["numpy_ref", "numba_cpu"]
OBLIQUITIES: tuple[str, ...] = ("sqrt", "kirchhoff", "none")
BACKENDS: tuple[str, ...] = ("numpy_ref", "numba_cpu")
_FOUR_PI = 4.0 * math.pi


def obliquity_code(obliquity: str) -> int:
    """0 = sqrt(cos cos), 1 = Kirchhoff (cos + cos)/2, 2 = none."""
    try:
        return OBLIQUITIES.index(obliquity)
    except ValueError:
        raise ConfigurationError(f"unknown obliquity {obliquity!r}; use {OBLIQUITIES}") from None


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ConfigurationError(f"unknown backend {backend!r}; use {BACKENDS}")


# --------------------------------------------------------------------------- numpy reference


def _terms_numpy(
    txs: np.ndarray,
    rxs: np.ndarray,
    pts: np.ndarray,
    areas: np.ndarray,
    nrm: np.ndarray,
    k: np.ndarray,
    ref_len: np.ndarray,
    gt: np.ndarray,
    gr: np.ndarray,
    mode: int,
    huygens: bool,
) -> np.ndarray:
    """(R, Q) complex128 K1 terms without Gamma (float64 reference)."""
    px, py, pz = pts[:, 0][None, :], pts[:, 1][None, :], pts[:, 2][None, :]
    nx, ny, nz = float(nrm[0]), float(nrm[1]), float(nrm[2])
    dxi = txs[:, 0:1] - px
    dyi = txs[:, 1:2] - py
    dzi = txs[:, 2:3] - pz
    dxs = rxs[:, 0:1] - px
    dys = rxs[:, 1:2] - py
    dzs = rxs[:, 2:3] - pz
    ri = np.sqrt((dxi * dxi + dyi * dyi) + dzi * dzi)
    rs = np.sqrt((dxs * dxs + dys * dys) + dzs * dzs)
    ci = ((dxi * nx + dyi * ny) + dzi * nz) / ri
    cs = ((dxs * nx + dys * ny) + dzs * nz) / rs
    vis = (ci > 0.0) & (cs > 0.0)
    if mode == 0:
        o = np.sqrt(np.where(vis, ci * cs, 0.0))
    elif mode == 1:
        o = 0.5 * (ci + cs)
    else:
        o = np.ones_like(ci)
    amp = np.sqrt(gt * gr) * areas[None, :] * o / (_FOUR_PI * ri * rs)
    amp = np.where(vis, amp, 0.0)
    phase = k[:, None] * ((ri + rs) - ref_len)
    c, s = np.cos(phase), np.sin(phase)
    if huygens:  # j * exp(-j phase) = sin(phase) + j cos(phase)
        return (amp * s) + 1j * (amp * c)
    return (amp * c) - 1j * (amp * s)


def _accumulate_numpy(
    terms: np.ndarray, parent: np.ndarray, n_cells: int, gamma: np.ndarray | None
) -> np.ndarray:
    """(R, S, N) per-cell sums ``sum_q gamma[r, s, q] * terms[r, q]`` (sequential in q)."""
    r_count, _q = terms.shape
    if gamma is None:
        out = np.empty((r_count, 1, n_cells), dtype=np.complex128)
        for r in range(r_count):
            out[r, 0] = np.bincount(parent, terms[r].real, n_cells) + 1j * np.bincount(
                parent, terms[r].imag, n_cells
            )
        return out
    s_count = gamma.shape[1]
    out = np.empty((r_count, s_count, n_cells), dtype=np.complex128)
    for r in range(r_count):
        prod = gamma[r] * terms[r][None, :]
        for s in range(s_count):
            out[r, s] = np.bincount(parent, prod[s].real, n_cells) + 1j * np.bincount(
                parent, prod[s].imag, n_cells
            )
    return out


# --------------------------------------------------------------------------- numba twin


@numba.njit(cache=True, nogil=True, fastmath=False)
def _k1_row(
    out, r, txs, rxs, pts, areas, nrm, k, ref_len, gt, gr, mode, huygens
):  # pragma: no cover
    """K1 terms of row ``r`` (sequential over quadrature points)."""
    q_count = pts.shape[0]
    nx = nrm[0]
    ny = nrm[1]
    nz = nrm[2]
    four_pi = 4.0 * math.pi
    tx0 = txs[r, 0]
    tx1 = txs[r, 1]
    tx2 = txs[r, 2]
    rx0 = rxs[r, 0]
    rx1 = rxs[r, 1]
    rx2 = rxs[r, 2]
    kr = k[r]
    for q in range(q_count):
        dxi = tx0 - pts[q, 0]
        dyi = tx1 - pts[q, 1]
        dzi = tx2 - pts[q, 2]
        dxs = rx0 - pts[q, 0]
        dys = rx1 - pts[q, 1]
        dzs = rx2 - pts[q, 2]
        ri = math.sqrt((dxi * dxi + dyi * dyi) + dzi * dzi)
        rs = math.sqrt((dxs * dxs + dys * dys) + dzs * dzs)
        ci = ((dxi * nx + dyi * ny) + dzi * nz) / ri
        cs = ((dxs * nx + dys * ny) + dzs * nz) / rs
        if not (ci > 0.0 and cs > 0.0):
            continue
        if mode == 0:
            o = math.sqrt(ci * cs)
        elif mode == 1:
            o = 0.5 * (ci + cs)
        else:
            o = 1.0
        amp = math.sqrt(gt[r, q] * gr[r, q]) * areas[q] * o / (four_pi * ri * rs)
        phase = kr * ((ri + rs) - ref_len[r, q])
        c = math.cos(phase)
        s = math.sin(phase)
        if huygens:
            out[r, q] = complex(amp * s, amp * c)
        else:
            out[r, q] = complex(amp * c, -(amp * s))


@numba.njit(cache=True, nogil=True, fastmath=False)
def _terms_numba_serial(
    txs, rxs, pts, areas, nrm, k, ref_len, gt, gr, mode, huygens
):  # pragma: no cover
    out = np.zeros((txs.shape[0], pts.shape[0]), dtype=np.complex128)
    for r in range(txs.shape[0]):
        _k1_row(out, r, txs, rxs, pts, areas, nrm, k, ref_len, gt, gr, mode, huygens)
    return out


@numba.njit(cache=True, parallel=True, nogil=True, fastmath=False)
def _terms_numba_parallel(
    txs, rxs, pts, areas, nrm, k, ref_len, gt, gr, mode, huygens
):  # pragma: no cover
    out = np.zeros((txs.shape[0], pts.shape[0]), dtype=np.complex128)
    for r in numba.prange(txs.shape[0]):
        _k1_row(out, r, txs, rxs, pts, areas, nrm, k, ref_len, gt, gr, mode, huygens)
    return out


@numba.njit(cache=True, nogil=True, fastmath=False)
def _acc_row(out, r, terms, parent, gamma):  # pragma: no cover
    """Per-cell sums of row ``r``; ``gamma`` (R, S, Q) or an empty (0, 0, 0) array."""
    q_count = terms.shape[1]
    if gamma.shape[0] == 0:
        for q in range(q_count):
            out[r, 0, parent[q]] += terms[r, q]
    else:
        for s in range(gamma.shape[1]):
            for q in range(q_count):
                out[r, s, parent[q]] += gamma[r, s, q] * terms[r, q]


@numba.njit(cache=True, nogil=True, fastmath=False)
def _accumulate_numba_serial(terms, parent, n_cells, gamma, s_count):  # pragma: no cover
    out = np.zeros((terms.shape[0], s_count, n_cells), dtype=np.complex128)
    for r in range(terms.shape[0]):
        _acc_row(out, r, terms, parent, gamma)
    return out


@numba.njit(cache=True, parallel=True, nogil=True, fastmath=False)
def _accumulate_numba_parallel(terms, parent, n_cells, gamma, s_count):  # pragma: no cover
    out = np.zeros((terms.shape[0], s_count, n_cells), dtype=np.complex128)
    for r in numba.prange(terms.shape[0]):
        _acc_row(out, r, terms, parent, gamma)
    return out


PARALLEL_MIN_WORK = 1 << 18
"""Row-parallel numba drivers run only when R >= 4 and R x Q >= this many terms; smaller
problems use the serial driver (identical results: rows are independent, and each row is
summed sequentially in a fixed order)."""


def _use_parallel(r_count: int, q_count: int) -> bool:
    return r_count >= 4 and r_count * q_count >= PARALLEL_MIN_WORK


@numba.njit(cache=True, nogil=True, fastmath=False)
def _tiles_separable_numba(c, g, tile, n_tiles):  # pragma: no cover
    r_count, n = c.shape
    out = np.zeros((r_count, n_tiles), dtype=np.complex128)
    for r in range(r_count):
        for i in range(n):
            out[r, tile[i]] += c[r, i] * g[i]
    return out


@numba.njit(cache=True, nogil=True, fastmath=False)
def _tiles_b_numba(b, state, tile, n_tiles):  # pragma: no cover
    r_count, _s, n = b.shape
    out = np.zeros((r_count, n_tiles), dtype=np.complex128)
    for r in range(r_count):
        for i in range(n):
            s_i = state[i]
            if s_i >= 0:
                out[r, tile[i]] += b[r, s_i, i]
    return out


def tile_sums(
    tile: np.ndarray,
    n_tiles: int,
    *,
    c: np.ndarray | None = None,
    g: np.ndarray | None = None,
    b: np.ndarray | None = None,
    state: np.ndarray | None = None,
    backend: Backend = "numpy_ref",
) -> np.ndarray:
    """Configuration GEMV (D-5): per-row tile amplitudes (R, K), fixed summation order.

    Separable form: ``out[r, k] = sum_{n in k} c[r, n] g[n]`` (``g`` per cell, dead cells
    already mapped to ``gamma_dead``). B form: ``out[r, k] = sum_{n in k, s_n >= 0}
    b[r, s_n, n]``. Cells are summed sequentially in index order per row on both backends
    (numba loop; numpy flat ``bincount``), never in parallel over the summation index.
    """
    _check_backend(backend)
    t = np.ascontiguousarray(tile, dtype=np.int64)
    if c is not None:
        assert g is not None
        cc = np.ascontiguousarray(c, dtype=np.complex128)
        gg = np.ascontiguousarray(g, dtype=np.complex128)
        if backend == "numba_cpu":
            return _tiles_separable_numba(cc, gg, t, int(n_tiles))
        w = cc * gg[None, :]
    else:
        assert b is not None
        assert state is not None
        bb = np.ascontiguousarray(b, dtype=np.complex128)
        st = np.ascontiguousarray(state, dtype=np.int64)
        if backend == "numba_cpu":
            return _tiles_b_numba(bb, st, t, int(n_tiles))
        alive = st >= 0
        w = np.where(alive[None, :], bb[:, np.where(alive, st, 0), np.arange(st.size)], 0.0)
    r_count = w.shape[0]
    idx = (np.arange(r_count)[:, None] * n_tiles + t[None, :]).ravel()
    re = np.bincount(idx, w.real.ravel(), r_count * n_tiles)
    im = np.bincount(idx, w.imag.ravel(), r_count * n_tiles)
    return (re + 1j * im).reshape(r_count, n_tiles)


# --------------------------------------------------------------------------- public API


def k1_terms(
    txs: np.ndarray,
    rxs: np.ndarray,
    points: np.ndarray,
    areas: np.ndarray,
    normal: np.ndarray,
    fc_hz: float | np.ndarray,
    *,
    ref_len_m: np.ndarray | float = 0.0,
    gain_tx: np.ndarray | None = None,
    gain_rx: np.ndarray | None = None,
    obliquity: Obliquity = "sqrt",
    huygens_j: bool = True,
    backend: Backend = "numpy_ref",
) -> np.ndarray:
    """K1 terms (R, Q) complex128 for R (Tx, Rx) rows and Q quadrature points, without Gamma.

    ``txs``/``rxs``: (R, 3) or (3,) positions, m. ``points`` (Q, 3), ``areas`` (Q,) or scalar m^2,
    ``normal`` (3,) unit panel normal. ``fc_hz``: scalar or (R,) carrier. ``ref_len_m``:
    reference path length (R, Q), (Q,) or scalar, m. ``gain_tx``/``gain_rx``: (R, Q) linear
    antenna power gains toward the points (default isotropic).
    """
    _check_backend(backend)
    txs = np.atleast_2d(np.asarray(txs, dtype=np.float64))
    rxs = np.atleast_2d(np.asarray(rxs, dtype=np.float64))
    pts = np.ascontiguousarray(points, dtype=np.float64)
    nrm = np.ascontiguousarray(normal, dtype=np.float64)
    r_count, q_count = txs.shape[0], pts.shape[0]
    a = np.ascontiguousarray(np.broadcast_to(np.asarray(areas, dtype=np.float64), (q_count,)))
    if rxs.shape != txs.shape or txs.shape[1] != 3 or pts.shape != (q_count, 3):
        raise ConfigurationError("k1_terms: txs/rxs must be (R, 3) and points (Q, 3)")
    if a.shape != (q_count,):
        raise ConfigurationError("k1_terms: areas must be (Q,)")
    k = np.broadcast_to(2.0 * math.pi * np.asarray(fc_hz, dtype=np.float64) / C0, (r_count,))
    k = np.ascontiguousarray(k)
    ref = np.ascontiguousarray(
        np.broadcast_to(np.asarray(ref_len_m, np.float64), (r_count, q_count))
    )
    ones = np.ones((r_count, q_count))
    gt = ones if gain_tx is None else np.ascontiguousarray(np.broadcast_to(gain_tx, ones.shape))
    gr = ones if gain_rx is None else np.ascontiguousarray(np.broadcast_to(gain_rx, ones.shape))
    mode = obliquity_code(obliquity)
    if backend == "numba_cpu":
        drv = _terms_numba_parallel if _use_parallel(r_count, q_count) else _terms_numba_serial
        return drv(txs, rxs, pts, a, nrm, k, ref, gt, gr, mode, bool(huygens_j))
    return _terms_numpy(txs, rxs, pts, a, nrm, k, ref, gt, gr, mode, bool(huygens_j))


def accumulate_cells(
    terms: np.ndarray,
    parent: np.ndarray,
    n_cells: int,
    gamma_q: np.ndarray | None = None,
    *,
    backend: Backend = "numpy_ref",
) -> np.ndarray:
    """Per-cell sums (R, S, N): ``sum_{q in n} gamma_q[r, s, q] * terms[r, q]``.

    ``gamma_q`` is (R, S, Q) complex128, or None for the geometric factor (S = 1). Quadrature
    points are summed sequentially in index order (fixed summation order, D-3).
    """
    _check_backend(backend)
    t = np.ascontiguousarray(terms, dtype=np.complex128)
    par = np.ascontiguousarray(parent, dtype=np.int64)
    if par.shape != (t.shape[1],):
        raise ConfigurationError("accumulate_cells: parent must be (Q,)")
    if par.size and (int(par.min()) < 0 or int(par.max()) >= n_cells):
        raise ConfigurationError("accumulate_cells: parent indices out of range")
    if gamma_q is not None:
        g = np.ascontiguousarray(gamma_q, dtype=np.complex128)
        if g.ndim != 3 or g.shape[0] != t.shape[0] or g.shape[2] != t.shape[1]:
            raise ConfigurationError("accumulate_cells: gamma_q must be (R, S, Q)")
    else:
        g = None
    if backend == "numba_cpu":
        drv = (
            _accumulate_numba_parallel
            if _use_parallel(t.shape[0], t.shape[1])
            else _accumulate_numba_serial
        )
        if g is None:
            return drv(t, par, int(n_cells), np.zeros((0, 0, 0), np.complex128), 1)
        return drv(t, par, int(n_cells), g, g.shape[1])
    return _accumulate_numpy(t, par, int(n_cells), g)


def _gains(ant: Antenna | None, src: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    if ant is None or isinstance(ant, Isotropic):
        return None
    d = pts - src[None, :]
    r = np.sqrt((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2])
    return np.asarray(ant.gain(d / r[:, None]), dtype=np.float64)[None, :]


def k1_channel(
    tx: Sequence[float] | np.ndarray,
    rx: Sequence[float] | np.ndarray,
    points: np.ndarray,
    areas: np.ndarray | float,
    normal: Sequence[float] | np.ndarray,
    fc_hz: float,
    gamma: np.ndarray | complex = 1.0,
    *,
    ant_tx: Antenna | None = None,
    ant_rx: Antenna | None = None,
    obliquity: Obliquity = "sqrt",
    huygens_j: bool = True,
    backend: Backend = "numpy_ref",
) -> complex:
    """Total K1 coefficient (absolute phase, ``L_ref = 0``) for per-point ``gamma`` (Q,).

    Convenience form for closed-form checks: every point is its own cell.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    q = pts.shape[0]
    a = np.broadcast_to(np.asarray(areas, dtype=np.float64), (q,)).copy()
    tx_ = np.asarray(tx, dtype=np.float64)
    rx_ = np.asarray(rx, dtype=np.float64)
    terms = k1_terms(
        tx_,
        rx_,
        pts,
        a,
        np.asarray(normal, dtype=np.float64),
        fc_hz,
        gain_tx=_gains(ant_tx, tx_, pts),
        gain_rx=_gains(ant_rx, rx_, pts),
        obliquity=obliquity,
        huygens_j=huygens_j,
        backend=backend,
    )[0]
    g = np.broadcast_to(np.asarray(gamma, dtype=np.complex128), (q,))
    return complex(np.sum(g * terms))


# --------------------------------------------------------------------------- far field


def fraunhofer_distance_m(diagonal_m: float, fc_hz: float) -> float:
    """Fraunhofer distance ``2 D^2 / lambda`` (m)."""
    return 2.0 * diagonal_m**2 / (C0 / fc_hz)


def farfield_focused_amplitude(
    n_cells: int,
    cell_area_m2: float,
    d1_m: float,
    d2_m: float,
    theta_t_rad: float,
    theta_r_rad: float,
    fc_hz: float,
    *,
    panel_diagonal_m: float,
    gamma_mag: float = 1.0,
    g_t: float = 1.0,
    g_r: float = 1.0,
) -> float:
    """Far-field closed form of a focused panel (Tang 2022 Eq. 8), an accelerator only.

    ``|h| = sqrt(G_T G_R) |Gamma| N A sqrt(cos theta_t cos theta_r) / (4 pi d1 d2)``.
    Raises ``GeometryError`` unless both distances exceed the Fraunhofer distance
    ``2 D^2 / lambda`` (section 5.3; guarded by PH-03).
    """
    lim = fraunhofer_distance_m(panel_diagonal_m, fc_hz)
    if min(d1_m, d2_m) < lim:
        raise GeometryError(
            f"far-field closed form needs d >= 2D^2/lambda = {lim:.3g} m (got {d1_m}, {d2_m})"
        )
    ct, cr = math.cos(theta_t_rad), math.cos(theta_r_rad)
    if ct <= 0.0 or cr <= 0.0:
        return 0.0
    return (
        math.sqrt(g_t * g_r)
        * gamma_mag
        * n_cells
        * cell_area_m2
        * math.sqrt(ct * cr)
        / (_FOUR_PI * d1_m * d2_m)
    )


# --------------------------------------------------------------------------- energy diagnostic


def _sinc_like(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hemisphere integrals I0 = int e dOmega, I1 = int cos e dOmega, I2 = int cos^2 e dOmega
    of ``e = exp(j k rhat.d)`` for in-plane ``d`` with ``x = k |d|``."""
    from scipy.special import j1

    small = x < 1e-3
    xs = np.where(small, 1.0, x)
    i0 = np.where(small, 2 * np.pi * (1 - x * x / 6), 2 * np.pi * np.sin(xs) / xs)
    i1 = np.where(small, np.pi * (1 - x * x / 8), 2 * np.pi * j1(xs) / xs)
    i2 = np.where(
        small,
        2 * np.pi * (1.0 / 3.0 - x * x / 30.0),
        2 * np.pi * (np.sin(xs) - xs * np.cos(xs)) / xs**3,
    )
    return i0, i1, i2


def energy_budget(
    tx: Sequence[float] | np.ndarray,
    points: np.ndarray,
    areas: np.ndarray | float,
    normal: Sequence[float] | np.ndarray,
    fc_hz: float,
    gamma: np.ndarray | complex = 1.0,
    *,
    ant_tx: Antenna | None = None,
    obliquity: Obliquity = "sqrt",
) -> tuple[Any, float]:
    """Scattered and incident power of a panel, both relative to P_T (PH-23).

    Incident: ``sum_q G_T A_q cos(theta_iq) / (4 pi r_iq^2)`` over front-lit points.
    Scattered: total far-zone power radiated by the K1 sources into the front hemisphere,
    evaluated in closed form from the hemisphere integrals of the obliquity-weighted plane
    waves (``2 pi J1(x)/x`` for the default obliquity; ``sin x / x`` and
    ``(sin x - x cos x)/x^3`` terms for Kirchhoff). Points must be coplanar (one panel) and
    Gamma is taken as independent of the scattering direction. ``gamma`` is a scalar, (Q,)
    (returns a float) or (M, Q) for M configurations (returns an (M,) array); the O(Q^2)
    kernel is built once per call.
    """
    pts = np.asarray(points, dtype=np.float64)
    q = pts.shape[0]
    a = np.broadcast_to(np.asarray(areas, dtype=np.float64), (q,))
    n = np.asarray(normal, dtype=np.float64)
    t = np.asarray(tx, dtype=np.float64)
    d = t[None, :] - pts
    ri = np.sqrt(np.sum(d * d, axis=1))
    ci = (d @ n) / ri
    lit = ci > 0.0
    gt = np.ones(q) if ant_tx is None else np.asarray(ant_tx.gain(-d / ri[:, None]))
    g_in = np.asarray(gamma, dtype=np.complex128)
    multi = g_in.ndim == 2
    g = g_in if multi else np.broadcast_to(g_in, (q,))[None, :]
    k = 2 * math.pi * fc_hz / C0
    lam = C0 / fc_hz
    p_inc = float(np.sum(np.where(lit, gt * a * ci / (_FOUR_PI * ri * ri), 0.0)))
    base = np.where(lit, np.sqrt(gt) * a * np.exp(-1j * k * ri) / ri, 0.0)[None, :] * g
    diff = pts[:, None, :] - pts[None, :, :]
    x = k * np.sqrt(np.sum(diff * diff, axis=2))
    i0, i1, i2 = _sinc_like(x)
    c = np.where(lit, ci, 0.0)
    code = obliquity_code(obliquity)
    if code == 0:
        w = base * np.sqrt(c)[None, :]
        kern = i1
    elif code == 1:
        w = base
        kern = 0.25 * (np.outer(c, c) * i0 + (c[:, None] + c[None, :]) * i1 + i2)
    else:
        w = base
        kern = i0
    quad = np.real(np.einsum("mq,mq->m", w, np.conj(w) @ kern.T))
    p_scat = quad / (_FOUR_PI * lam * lam)
    return (p_scat if multi else float(p_scat[0])), p_inc
