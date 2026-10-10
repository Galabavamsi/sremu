"""``numba_cpu`` render backend (DESIGN.md sections 4.3 D-2..D-5, 4.4, 7.1).

Same signal model and prepared inputs as ``render_numpy`` (shared ``prepare_segment``), with:

* **Rotators.** The carrier phase of each path and the link LO offset are advanced by a
  complex128 recursion ``z <- z*r`` with ``r = exp(-j 2 pi dc*inv_len)`` (exact up to
  rounding, because the chord phase is linear in n). The recursion restarts from a
  closed-form anchor at every **absolute 64-sample block** boundary and at the snapshot
  interval's first sample. Page sizes are powers of two >= 64, so every page start is an
  anchor (D-2). A segment that starts inside a block replays the recursion from the block
  anchor, so results are bitwise independent of segmentation; ``LinkJob.rot_state`` is
  therefore not needed and is ignored.
* **Parallelism** over 64-sample blocks with ``prange`` (never over the summation index,
  D-3); per-block scratch rows are private to their block, so the output does not depend on
  the thread count.
* **No fastmath** (D-4). Sums run in fixed order: links in job order, paths by index (D-3).
* **FD**: the same polyphase rows, linear tap interpolation and tap summation order as
  ``numpy_ref`` (four partial sums over m mod 4, combined ``((0+1)+(2+3))``). When a group's
  envelope delay is constant over the interval (``g_dd == 0``, static scenes) the taps are
  computed once per block; the values are identical to the per-sample formula.
* **Caches.** The flattened static arrays of a link set are memoized by the identity of the
  memoized link statics (``render_numpy.link_static``); per segment only the TX windows and
  the D-1 grid enter the kernel call.

I5 contract with ``numpy_ref``: <= 1e-9 relative on the complex128 accumulator.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Final

import numba as nb
import numpy as np
from numpy.typing import NDArray

from sremu.contracts import BackendName, SegmentJob
from sremu.errors import ConfigurationError

from .render_numpy import LinkStatic, PreparedSegment, prepare_segment
from .render_numpy import clear_caches as _clear_numpy_caches

__all__ = ["BLOCK", "NumbaCpuBackend", "clear_caches", "noise", "render", "render_prepared"]

BLOCK: Final[int] = 64
"""Rotator anchor block (absolute, aligned); divides every page size."""

_SPLITTER = 134217729.0


@nb.njit(inline="always", nogil=True, cache=True)
def _frac_cycles(n: np.int64, q_hi: float, q_lo: float) -> float:  # pragma: no cover - numba
    nf = np.float64(n)
    p = nf * q_hi
    t = _SPLITTER * nf
    ahi = t - (t - nf)
    alo = nf - ahi
    tq = _SPLITTER * q_hi
    bhi = tq - (tq - q_hi)
    blo = q_hi - bhi
    e = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo
    r = p - np.rint(p)
    c = r + (e + nf * q_lo)
    return c - np.rint(c)


@nb.njit(inline="always", nogil=True, cache=True)
def _profile(code: np.int64, x: float, dur: float) -> float:  # pragma: no cover - numba
    if x < 0.0:
        return 0.0
    if code == 0:
        return 1.0
    if code == 1:
        v = x / dur
        return 1.0 if v > 1.0 else v
    return -np.expm1(-x / dur)


@nb.njit(inline="always", nogil=True, cache=True)
def _taps_into(hb, mu, nph, tables, tab_off, ntap):  # pragma: no cover - numba
    """Linearly interpolated polyphase taps at fractional delay ``mu`` into ``hb``."""
    jf = mu * nph
    j = np.int64(np.floor(jf))
    if j > nph - 1:
        j = nph - 1
    alpha = jf - j
    r0 = tab_off + j * ntap
    r1 = r0 + ntap
    for m in range(ntap):
        t0 = tables[r0 + m]
        hb[m] = t0 + alpha * (tables[r1 + m] - t0)


@nb.njit(inline="always", nogil=True, cache=True)
def _fd_dot_taps(xr, xi, base, hb, ntap):  # pragma: no cover - numba
    """sum_m hb[m] x[base - m] as four partial sums over m mod 4, ((0+1)+(2+3))."""
    a0r = 0.0
    a1r = 0.0
    a2r = 0.0
    a3r = 0.0
    a0i = 0.0
    a1i = 0.0
    a2i = 0.0
    a3i = 0.0
    m = 0
    while m + 4 <= ntap:
        a0r = a0r + hb[m] * xr[base - m]
        a0i = a0i + hb[m] * xi[base - m]
        a1r = a1r + hb[m + 1] * xr[base - m - 1]
        a1i = a1i + hb[m + 1] * xi[base - m - 1]
        a2r = a2r + hb[m + 2] * xr[base - m - 2]
        a2i = a2i + hb[m + 2] * xi[base - m - 2]
        a3r = a3r + hb[m + 3] * xr[base - m - 3]
        a3i = a3i + hb[m + 3] * xi[base - m - 3]
        m += 4
    while m < ntap:  # at most three remaining taps: m mod 4 = 0, 1, 2
        q4 = m & 3
        if q4 == 0:
            a0r = a0r + hb[m] * xr[base - m]
            a0i = a0i + hb[m] * xi[base - m]
        elif q4 == 1:
            a1r = a1r + hb[m] * xr[base - m]
            a1i = a1i + hb[m] * xi[base - m]
        else:
            a2r = a2r + hb[m] * xr[base - m]
            a2i = a2i + hb[m] * xi[base - m]
        m += 1
    return complex((a0r + a1r) + (a2r + a3r), (a0i + a1i) + (a2i + a3i))


@nb.njit(inline="always", nogil=True, cache=True)
def _fd_dot(xr, xi, base, tables, r0, r1, alpha, ntap):  # pragma: no cover - numba
    """As :func:`_fd_dot_taps` with the taps interpolated inline (same values, same order)."""
    a0r = 0.0
    a1r = 0.0
    a2r = 0.0
    a3r = 0.0
    a0i = 0.0
    a1i = 0.0
    a2i = 0.0
    a3i = 0.0
    m = 0
    while m + 4 <= ntap:
        t0 = tables[r0 + m]
        h0 = t0 + alpha * (tables[r1 + m] - t0)
        t0 = tables[r0 + m + 1]
        h1 = t0 + alpha * (tables[r1 + m + 1] - t0)
        t0 = tables[r0 + m + 2]
        h2 = t0 + alpha * (tables[r1 + m + 2] - t0)
        t0 = tables[r0 + m + 3]
        h3 = t0 + alpha * (tables[r1 + m + 3] - t0)
        a0r = a0r + h0 * xr[base - m]
        a0i = a0i + h0 * xi[base - m]
        a1r = a1r + h1 * xr[base - m - 1]
        a1i = a1i + h1 * xi[base - m - 1]
        a2r = a2r + h2 * xr[base - m - 2]
        a2i = a2i + h2 * xi[base - m - 2]
        a3r = a3r + h3 * xr[base - m - 3]
        a3i = a3i + h3 * xi[base - m - 3]
        m += 4
    while m < ntap:
        t0 = tables[r0 + m]
        h = t0 + alpha * (tables[r1 + m] - t0)
        q4 = m & 3
        if q4 == 0:
            a0r = a0r + h * xr[base - m]
            a0i = a0i + h * xi[base - m]
        elif q4 == 1:
            a1r = a1r + h * xr[base - m]
            a1i = a1i + h * xi[base - m]
        else:
            a2r = a2r + h * xr[base - m]
            a2i = a2i + h * xi[base - m]
        m += 1
    return complex((a0r + a1r) + (a2r + a3r), (a0i + a1i) + (a2i + a3i))


@nb.njit(parallel=True, nogil=True, cache=True)
def _kernel(  # pragma: no cover - numba
    out,
    k0,
    k1,
    n_first,
    n_a_int,
    n_a_frac,
    inv_len,
    # links
    l_poff,
    l_goff,
    l_toff,
    l_xoff,
    l_xlo,
    l_haslo,
    l_qhi,
    l_qlo,
    l_rlo,
    xr,
    xi,
    # groups
    g_dint,
    g_dfrac,
    g_dd,
    g_taps,
    g_center,
    g_integer,
    g_tab_off,
    g_nph,
    tables,
    # paths
    p_ca,
    p_dc,
    p_aa,
    p_da,
    p_g,
    p_r,
    # terms
    t_k,
    t_code,
    t_dur,
    t_aoff,
    t_dAa,
    t_ddA,
):
    n_links = l_poff.shape[0] - 1
    n_paths = p_ca.shape[0]
    n_groups = g_dint.shape[0]
    n_terms = t_k.shape[0]
    first_blk = k0 // BLOCK
    n_blk = (k1 - 1) // BLOCK - first_blk + 1
    z_all = np.empty((n_blk, max(n_paths, 1)), dtype=np.complex128)
    zl_all = np.empty((n_blk, max(n_links, 1)), dtype=np.complex128)
    e_all = np.empty((n_blk, max(n_groups, 1)), dtype=np.complex128)
    gv_all = np.empty((n_blk, max(n_terms, 1)), dtype=np.float64)
    l_max = 1
    for g in range(n_groups):
        if g_taps[g] > l_max:
            l_max = g_taps[g]
    h_all = np.empty((n_blk, max(n_groups, 1), l_max), dtype=np.float64)
    s_all = np.empty((n_blk, max(n_groups, 1)), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for bi in nb.prange(n_blk):
        z = z_all[bi]
        zl = zl_all[bi]
        env = e_all[bi]
        gv = gv_all[bi]
        s_last = s_all[bi]
        for g in range(n_groups):
            s_last[g] = np.nan  # forces a tap computation at the first sample of the block
        blk = first_blk + bi
        s0 = blk * BLOCK
        if s0 < n_first:
            s0 = n_first
        e_end = (blk + 1) * BLOCK
        if e_end > k1:
            e_end = k1
        # closed-form anchors at s0
        w0 = (np.float64(s0 - n_a_int) - n_a_frac) * inv_len
        for p in range(n_paths):
            c = p_ca[p] + w0 * p_dc[p]
            c = c - np.rint(c)
            ang = two_pi * c
            z[p] = complex(np.cos(ang), -np.sin(ang))
        for li in range(n_links):
            if l_haslo[li]:
                c = _frac_cycles(np.int64(s0), l_qhi[li], l_qlo[li])
                ang = two_pi * c
                zl[li] = complex(np.cos(ang), np.sin(ang))
        for n in range(s0, e_end):
            if n >= k0:
                w = (np.float64(n - n_a_int) - n_a_frac) * inv_len
                for t in range(n_terms):
                    gv[t] = _profile(t_code[t], np.float64(n - t_k[t]), t_dur[t])
                for li in range(n_links):
                    pa = l_poff[li]
                    pb = l_poff[li + 1]
                    if pb == pa:
                        continue
                    xo = l_xoff[li]
                    xlo = l_xlo[li]
                    for g in range(l_goff[li], l_goff[li + 1]):
                        s = g_dfrac[g] + w * g_dd[g]
                        if g_integer[g]:
                            i = g_dint[g] + np.int64(np.floor(s + 0.5))
                            idx = xo + (n - i) - xlo
                            env[g] = complex(xr[idx], xi[idx])
                        else:
                            fi = np.floor(s)
                            i = g_dint[g] + np.int64(fi)
                            ntap = g_taps[g]
                            base = xo + (n - i + g_center[g]) - xlo
                            if g_dd[g] == 0.0:
                                # constant envelope delay over the interval: s, hence the taps,
                                # are the same at every sample; compute them once per block
                                hb = h_all[bi, g]
                                if s != s_last[g]:
                                    _taps_into(hb, s - fi, g_nph[g], tables, g_tab_off[g], ntap)
                                    s_last[g] = s
                                env[g] = _fd_dot_taps(xr, xi, base, hb, ntap)
                            else:
                                mu = s - fi
                                nph = g_nph[g]
                                jf = mu * nph
                                j = np.int64(np.floor(jf))
                                if j > nph - 1:
                                    j = nph - 1
                                r0 = g_tab_off[g] + j * ntap
                                env[g] = _fd_dot(xr, xi, base, tables, r0, r0 + ntap, jf - j, ntap)
                    acc = 0j
                    ta = l_toff[li]
                    tb = l_toff[li + 1]
                    for p in range(pa, pb):
                        a = p_aa[p] + w * p_da[p]
                        for t in range(ta, tb):
                            gval = gv[t]
                            if gval != 0.0:
                                q = t_aoff[t] + (p - pa)
                                a = a + gval * (t_dAa[q] + w * t_ddA[q])
                        acc = acc + (a * z[p]) * env[p_g[p]]
                    if l_haslo[li]:
                        acc = acc * zl[li]
                    out[n - k0] += acc
            for p in range(n_paths):
                z[p] = z[p] * p_r[p]
            for li in range(n_links):
                if l_haslo[li]:
                    zl[li] = zl[li] * l_rlo[li]


_FLAT_CACHE_MAX = 256
_flat_cache: OrderedDict[tuple[LinkStatic, ...], tuple] = OrderedDict()
_flat_lock = threading.Lock()  # render may be called from several engine threads (nogil)


def _flatten_static(statics: tuple[LinkStatic, ...]) -> tuple:
    """Concatenate the per-link static arrays (memoized by the identity of the statics,
    which the cache entry keeps alive)."""
    with _flat_lock:
        hit = _flat_cache.get(statics)
        if hit is not None:
            _flat_cache.move_to_end(statics)
            return hit
    nl = len(statics)
    l_poff = np.zeros(nl + 1, dtype=np.int64)
    l_goff = np.zeros(nl + 1, dtype=np.int64)
    l_toff = np.zeros(nl + 1, dtype=np.int64)
    for i, st in enumerate(statics):
        l_poff[i + 1] = l_poff[i] + st.n_paths
        l_goff[i + 1] = l_goff[i] + st.n_groups
        l_toff[i + 1] = l_toff[i] + st.t_k.shape[0]
    l_haslo = np.array([st.has_lo for st in statics], dtype=np.bool_)
    l_qhi = np.array([st.q_hi for st in statics], dtype=np.float64)
    l_qlo = np.array([st.q_lo for st in statics], dtype=np.float64)
    l_rlo = np.array([st.r_lo for st in statics], dtype=np.complex128)

    tables: list[np.ndarray] = []
    tab_index: dict[int, tuple[int, int, int]] = {}
    tab_len = 0
    g_rows: list[tuple[int, float, float, int, int, bool, int, int]] = []
    for st in statics:
        if st.table is not None:
            key = id(st.table)
            if key not in tab_index:
                tab_index[key] = (tab_len, st.table.shape[1], st.table.shape[0] - 1)
                tables.append(st.table.reshape(-1))
                tab_len += st.table.size
            off, taps, nph = tab_index[key]
        else:
            off, taps, nph = 0, 1, 1
        for g in range(st.n_groups):
            g_rows.append(
                (
                    int(st.g_dint[g]),
                    float(st.g_dfrac[g]),
                    float(st.g_dd[g]),
                    taps,
                    st.center,
                    st.table is None,
                    off,
                    nph,
                )
            )
    tab = np.concatenate(tables) if tables else np.zeros(1, dtype=np.float64)

    def col(i: int, dtype: type[np.generic]) -> np.ndarray:
        return np.array([r[i] for r in g_rows], dtype=dtype)

    def cat(attr: str, dtype: type[np.generic]) -> np.ndarray:
        parts = [getattr(st, attr) for st in statics]
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    if nl:
        p_g = np.concatenate([st.p_g + l_goff[i] for i, st in enumerate(statics)])
    else:
        p_g = np.zeros(0, np.int64)
    t_aoff_parts: list[int] = []
    dAa_parts: list[np.ndarray] = []
    ddA_parts: list[np.ndarray] = []
    off = 0
    for st in statics:
        for t in range(st.t_k.shape[0]):
            t_aoff_parts.append(off)
            dAa_parts.append(st.t_dAa[t])
            ddA_parts.append(st.t_ddA[t])
            off += st.n_paths
    links_f = (l_poff, l_goff, l_toff, l_haslo, l_qhi, l_qlo, l_rlo)
    groups_f = (
        col(0, np.int64),
        col(1, np.float64),
        col(2, np.float64),
        col(3, np.int64),
        col(4, np.int64),
        col(5, np.bool_),
        col(6, np.int64),
        col(7, np.int64),
        tab,
    )
    paths_f = (
        cat("p_ca", np.float64),
        cat("p_dc", np.float64),
        cat("p_aa", np.complex128),
        cat("p_da", np.complex128),
        p_g.astype(np.int64),
        cat("p_r", np.complex128),
    )
    terms_f = (
        cat("t_k", np.int64),
        cat("t_code", np.int64),
        cat("t_dur", np.float64),
        np.array(t_aoff_parts, dtype=np.int64),
        np.concatenate(dAa_parts) if dAa_parts else np.zeros(0, np.complex128),
        np.concatenate(ddA_parts) if ddA_parts else np.zeros(0, np.complex128),
    )
    flat = (links_f, groups_f, paths_f, terms_f)
    with _flat_lock:
        _flat_cache[statics] = flat
        if len(_flat_cache) > _FLAT_CACHE_MAX:
            _flat_cache.popitem(last=False)
    return flat


def clear_caches() -> None:
    """Drop the memoized flattenings and the shared link statics."""
    with _flat_lock:
        _flat_cache.clear()
    _clear_numpy_caches()


def render_prepared(seg: PreparedSegment, out: NDArray[np.complex128]) -> None:
    """numba evaluation of a prepared segment (adds into ``out``)."""
    if seg.n == 0 or not seg.links:
        return
    links_f, groups_f, paths_f, terms_f = _flatten_static(tuple(pl.static for pl in seg.links))
    l_poff, l_goff, l_toff, l_haslo, l_qhi, l_qlo, l_rlo = links_f
    if len(seg.links) == 1:
        x = seg.links[0].x
        l_xoff = np.array([0, x.shape[0]], dtype=np.int64)
    else:
        x = np.concatenate([pl.x for pl in seg.links])
        l_xoff = np.zeros(len(seg.links) + 1, dtype=np.int64)
        l_xoff[1:] = np.cumsum([pl.x.shape[0] for pl in seg.links])
    l_xlo = np.array([pl.x_lo for pl in seg.links], dtype=np.int64)
    g = seg.grid
    _kernel(
        out,
        np.int64(seg.k0),
        np.int64(seg.k1),
        np.int64(g.n_first),
        np.int64(g.n_a_int),
        float(g.n_a_frac),
        float(g.inv_len),
        l_poff,
        l_goff,
        l_toff,
        l_xoff,
        l_xlo,
        l_haslo,
        l_qhi,
        l_qlo,
        l_rlo,
        x.real.astype(np.float64),
        x.imag.astype(np.float64),
        *groups_f,
        *paths_f,
        *terms_f,
    )


def render(job: SegmentJob, out: NDArray[np.complex128]) -> None:
    """``numba_cpu`` render: add the segment's received signal into ``out`` (section 3.4)."""
    if not isinstance(out, np.ndarray) or out.dtype != np.complex128 or out.ndim != 1:
        raise ConfigurationError("out must be a 1-D complex128 accumulator")
    if out.shape[0] != int(job.k1) - int(job.k0):
        raise ConfigurationError(
            f"out has length {out.shape[0]}, expected {int(job.k1) - int(job.k0)}"
        )
    if not out.flags.c_contiguous:
        raise ConfigurationError("out must be C-contiguous")
    render_prepared(prepare_segment(job), out)


def noise(rx_key: tuple[int, int], k0: int, n: int, sigma: float) -> NDArray[np.complex64]:
    """``numba_cpu`` AWGN; see :func:`sremu.impair.noise.noise_numba`."""
    from sremu.impair.noise import noise_numba

    return noise_numba(rx_key, k0, n, sigma)


class NumbaCpuBackend:
    """The ``numba_cpu`` backend (``contracts.RenderBackend``)."""

    name: BackendName = "numba_cpu"

    def render(self, job: SegmentJob, out: NDArray[np.complex128]) -> None:
        """See :func:`render`."""
        render(job, out)

    def noise(
        self, rx_key: tuple[int, int], k0: int, n: int, sigma: float
    ) -> NDArray[np.complex64]:
        """See :func:`noise`."""
        return noise(rx_key, k0, n, sigma)
