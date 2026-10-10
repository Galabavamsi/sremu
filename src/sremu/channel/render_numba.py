"""``numba_cpu`` render backend (DESIGN.md sections 3.4, 4.3 D-2..D-5, 4.4, 4.5, 7.1).

Module-level API (render contract, D20):

* ``render(job, out)`` ADDS the received signal of RX samples [k0, k1) of ``job`` into the
  complex128 accumulator ``out`` (length ``k1 - k0``, C-contiguous). It is a pure function of
  its inputs; the signal model is the one documented in :mod:`sremu.channel.render_numpy`
  (chord delays with exact carrier phase, complex-linear amplitudes, transition terms,
  polyphase FD envelopes per group, and the link LO offset rotating by
  ``exp(+j 2 pi lo_offset_hz n / f_s)`` at the RX rate from absolute sample index 0).
* ``noise(rx_key, k0, n, sigma)`` is :func:`sremu.impair.noise.noise` on this backend
  (``sigma**2 = E|w|**2``, the total complex variance).

Same prepared inputs as ``render_numpy`` (shared ``prepare_segment``), evaluated as follows:

* **Blocks.** RX samples are processed in absolute 64-sample blocks ``[64 b, 64 (b + 1))``
  intersected with the segment. Page sizes are powers of two >= 64, so every page start is a
  block start.
* **Rotators (D-2).** The carrier phase of each path is linear in n inside a snapshot
  interval (chord delay), so ``e^{-j 2 pi c(n)} = z(s0)*r**(n - s0)`` with the step rotator
  ``r = exp(-j 2 pi dc*inv_len)``. ``z(s0)`` is the closed-form anchor at the block start
  ``s0`` (or the interval's first sample when that lies inside the block) and the step powers
  ``r**i`` (i < 64) are evaluated once per link static in closed form
  (``exp(-j 2 pi frac(i*step))``); likewise for the LO rotator with its absolute-index anchor
  (``frac(df*s0/f_s)`` in double-double). No recursion runs across samples, so the samples of
  a block are independent (no dependency chain) and a segment that starts inside a block uses
  the same anchor and powers as an aligned one: results are bitwise independent of
  segmentation. ``LinkJob.rot_state`` is therefore not needed and is ignored (it is
  optional, D11).
* **FD envelopes** are computed per block and group before the path sums, with the same
  polyphase rows, linear tap interpolation ``h = T[j] + alpha*(T[j+1] - T[j])`` and tap
  summation order as ``numpy_ref`` (four partial sums over m mod 4, each in increasing m,
  combined ``((0+1)+(2+3))``):

  - the envelope delay is linear in n and every rounding step of ``floor``, the table row and
    the integer delay is monotone, so when both block ends agree on them the whole block
    does (decided from the two ends, not per sample);
  - then (always for a constant envelope delay, and for slowly moving delays) the block
    uses one table row pair and a contiguous TX window: the tap index is the outer loop and
    the sample index the inner one, which vectorizes; every sample's partial sums still
    receive the same products in the same order, so the values are bitwise those of the
    per-sample formula;
  - otherwise each sample is evaluated on its own (same formula). Integer-only delays copy
    the TX window directly when the block has one offset.
* **Paths** in fixed order (D-3): links in job order, paths by index; per sample
  ``acc += (A_p(n)*z_p(n))*E_g(p)(n)``, times the LO rotator, added to ``out``.
* **Parallelism** over blocks with ``prange`` (never over a summation index, D-3) when the
  estimated work ``(k1 - k0)*units`` reaches :data:`PARALLEL_MIN_WORK`; smaller segments run
  the same block function serially, which avoids the parallel-region launch (about 10 us on
  the development laptop) on small pages. The parallel driver splits the blocks into
  ``numba.get_num_threads()`` contiguous runs, one ``prange`` iteration and one private
  scratch slot each. Every block starts from its own anchors and both drivers execute
  identical floating-point operations, so the output does not depend on the thread count,
  the split or the driver (I2a). The renderer never changes the thread count: changing it
  between calls re-sizes the OpenMP team, which cost more than it saved; a run should fix it
  once.
* **No fastmath** (D-4).
* **Caches.** Each memoized ``LinkStatic`` carries its packed kernel arrays (built once);
  link sets with several links are concatenated once per tuple of statics (LRU). Polyphase
  tables live in one module-level bank: each distinct table is copied into it once per
  process, so a new snapshot interval copies no table. TX windows are read as complex64
  directly (no per-segment float64 copies).

I5 contract with ``numpy_ref``: <= 1e-9 relative on the complex128 accumulator.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numba as nb
import numpy as np
from numpy.typing import NDArray

from sremu.contracts import BackendName, SegmentJob
from sremu.errors import ConfigurationError

from .render_numpy import LinkStatic, PreparedSegment, prepare_segment
from .render_numpy import clear_caches as _clear_numpy_caches

__all__ = [
    "BLOCK",
    "PARALLEL_MIN_WORK",
    "NumbaCpuBackend",
    "bank_size",
    "clear_caches",
    "noise",
    "render",
    "render_prepared",
]

BLOCK: Final[int] = 64
"""Rotator anchor block (absolute, aligned); divides every page size."""

PARALLEL_MIN_WORK: Final[int] = 1 << 16
"""[ENG] Segments whose work estimate ``(k1 - k0)*units`` is below this run the block loop
serially (``units`` = FD taps over all groups + 4*(paths*(1 + transition terms)) per sample);
about 20 us of single-thread work on the development laptop, twice the parallel launch cost."""

_SPLITTER = 134217729.0


# --------------------------------------------------------------------------- kernel pieces


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
def _fd_dot(xc, base, bank, r0, r1, alpha, ntap):  # pragma: no cover - numba
    """``sum_m h[m] x[base - m]`` with ``h[m] = T[r0+m] + alpha*(T[r1+m] - T[r0+m])``, as four
    partial sums over m mod 4 (each in increasing m) combined ``((0+1)+(2+3))``."""
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
        t0 = bank[r0 + m]
        h0 = t0 + alpha * (bank[r1 + m] - t0)
        t0 = bank[r0 + m + 1]
        h1 = t0 + alpha * (bank[r1 + m + 1] - t0)
        t0 = bank[r0 + m + 2]
        h2 = t0 + alpha * (bank[r1 + m + 2] - t0)
        t0 = bank[r0 + m + 3]
        h3 = t0 + alpha * (bank[r1 + m + 3] - t0)
        v0 = xc[base - m]
        v1 = xc[base - m - 1]
        v2 = xc[base - m - 2]
        v3 = xc[base - m - 3]
        a0r = a0r + h0 * np.float64(v0.real)
        a0i = a0i + h0 * np.float64(v0.imag)
        a1r = a1r + h1 * np.float64(v1.real)
        a1i = a1i + h1 * np.float64(v1.imag)
        a2r = a2r + h2 * np.float64(v2.real)
        a2i = a2i + h2 * np.float64(v2.imag)
        a3r = a3r + h3 * np.float64(v3.real)
        a3i = a3i + h3 * np.float64(v3.imag)
        m += 4
    while m < ntap:  # at most three remaining taps: m mod 4 = 0, 1, 2
        t0 = bank[r0 + m]
        h = t0 + alpha * (bank[r1 + m] - t0)
        v = xc[base - m]
        q4 = m & 3
        if q4 == 0:
            a0r = a0r + h * np.float64(v.real)
            a0i = a0i + h * np.float64(v.imag)
        elif q4 == 1:
            a1r = a1r + h * np.float64(v.real)
            a1i = a1i + h * np.float64(v.imag)
        else:
            a2r = a2r + h * np.float64(v.real)
            a2i = a2i + h * np.float64(v.imag)
        m += 1
    return complex((a0r + a1r) + (a2r + a3r), (a0i + a1i) + (a2i + a3i))


@nb.njit(nogil=True, cache=True)
def _envelope_block(  # pragma: no cover - numba
    er, ei, g, lo, cnt, wv, xo, xlo, xc, G_i, G_f, bank, acc, hb, al
):
    """FD envelope of group ``g`` for RX samples [lo, lo + cnt) into ``er``/``ei``.

    ``wv[i]`` is the D-1 weight of sample ``lo + i``; ``acc``, ``hb`` and ``al`` are scratch.
    """
    dint = G_i[g, 0]
    ntap = G_i[g, 1]
    center = G_i[g, 2]
    tab_off = G_i[g, 4]
    nph = G_i[g, 5]
    dfrac = G_f[g, 0]
    dd = G_f[g, 1]
    # The delay s(i) = dfrac + wv[i]*dd is monotone in i (w(n) is monotone in n and every
    # rounding step is monotone), and so are floor(s), the table row and the integer delay.
    # Equal values at the two block ends therefore hold for every sample of the block.
    s_f = dfrac + wv[0] * dd
    s_l = dfrac + wv[cnt - 1] * dd
    if G_i[g, 3]:  # integer-only delay: x[n - floor(d + 1/2)]
        ii0 = dint + np.int64(np.floor(s_f + 0.5))
        if ii0 == dint + np.int64(np.floor(s_l + 0.5)):  # one TX offset for the block
            o = xo + (lo - ii0) - xlo
            for i in range(cnt):
                v = xc[o + i]
                er[i] = np.float64(v.real)
                ei[i] = np.float64(v.imag)
            return
        for i in range(cnt):
            s = dfrac + wv[i] * dd
            ii = dint + np.int64(np.floor(s + 0.5))
            v = xc[xo + (lo + i - ii) - xlo]
            er[i] = np.float64(v.real)
            ei[i] = np.float64(v.imag)
        return
    fi0 = np.floor(s_f)
    jf0 = (s_f - fi0) * nph
    j0 = np.int64(np.floor(jf0))
    if j0 > nph - 1:
        j0 = nph - 1
    uniform = False
    if fi0 == np.floor(s_l):
        jl = np.int64(np.floor((s_l - fi0) * nph))
        if jl > nph - 1:
            jl = nph - 1
        uniform = jl == j0
    if not uniform:  # a table-row or integer-delay change inside the block: per sample
        for i in range(cnt):
            s = dfrac + wv[i] * dd
            fi = np.floor(s)
            ii = dint + np.int64(fi)
            jf = (s - fi) * nph
            j = np.int64(np.floor(jf))
            if j > nph - 1:
                j = nph - 1
            r0 = tab_off + j * ntap
            v = _fd_dot(xc, xo + (lo + i - ii + center) - xlo, bank, r0, r0 + ntap, jf - j, ntap)
            er[i] = v.real
            ei[i] = v.imag
        return
    # one table row pair and a contiguous TX window over the block: tap index outer. Per
    # sample, alpha = (s - floor(s))*nph - j with the block's floor(s) and j: the same
    # operations as the per-sample formula.
    r0 = tab_off + j0 * ntap
    r1 = r0 + ntap
    b0 = xo + (lo - (dint + np.int64(fi0)) + center) - xlo
    for q in range(8):
        row = acc[q]
        for i in range(cnt):
            row[i] = 0.0
    if dd == 0.0:  # constant envelope delay: one tap set for the block
        a = jf0 - j0
        for m in range(ntap):
            t0 = bank[r0 + m]
            hb[m] = t0 + a * (bank[r1 + m] - t0)
        for m in range(ntap):
            h = hb[m]
            q2 = 2 * (m & 3)
            ar = acc[q2]
            ai = acc[q2 + 1]
            xs = xc[b0 - m : b0 - m + cnt]
            for i in range(cnt):
                v = xs[i]
                ar[i] = ar[i] + h * np.float64(v.real)
                ai[i] = ai[i] + h * np.float64(v.imag)
    else:
        for i in range(cnt):
            al[i] = ((dfrac + wv[i] * dd) - fi0) * nph - j0
        for m in range(ntap):
            t0 = bank[r0 + m]
            dm = bank[r1 + m] - t0
            q2 = 2 * (m & 3)
            ar = acc[q2]
            ai = acc[q2 + 1]
            xs = xc[b0 - m : b0 - m + cnt]
            for i in range(cnt):
                h = t0 + al[i] * dm
                v = xs[i]
                ar[i] = ar[i] + h * np.float64(v.real)
                ai[i] = ai[i] + h * np.float64(v.imag)
    a0r = acc[0]
    a0i = acc[1]
    a1r = acc[2]
    a1i = acc[3]
    a2r = acc[4]
    a2i = acc[5]
    a3r = acc[6]
    a3i = acc[7]
    for i in range(cnt):
        er[i] = (a0r[i] + a1r[i]) + (a2r[i] + a3r[i])
        ei[i] = (a0i[i] + a1i[i]) + (a2i[i] + a3i[i])


@nb.njit(nogil=True, cache=True)
def _render_block(  # pragma: no cover - numba
    slot,
    blk,
    out,
    k0,
    k1,
    n_first,
    n_a_int,
    n_a_frac,
    inv_len,
    L_i,
    L_f,
    L_R,
    X_i,
    xc,
    G_i,
    G_f,
    bank,
    P_f,
    P_c,
    P_R,
    P_g,
    T_i,
    T_f,
    T_c,
    S,
):
    """Add RX samples ``[64 blk, 64 (blk + 1)) & [k0, k1)`` into ``out`` (scratch row ``slot``).

    Per link: FD envelopes of its groups, then the paths in index order into a per-sample
    accumulator, then the LO rotator and the addition into ``out``. Each rotator is its
    closed-form anchor at the block start ``s0`` times the step power ``R[i] = r**i`` of the
    sample's offset from ``s0`` (no recursion across samples). Per sample the operations and
    their order are those of the per-sample formula (D-3).
    """
    S_w, S_gv, S_env, S_acc, S_hb, S_al, S_lk = S
    n_links = L_i.shape[0]
    n_terms = T_i.shape[0]
    s0 = blk * BLOCK
    if s0 < n_first:
        s0 = n_first
    e_end = (blk + 1) * BLOCK
    if e_end > k1:
        e_end = k1
    lo = s0 if s0 > k0 else k0
    cnt = e_end - lo
    if cnt <= 0:
        return
    wv = S_w[slot]
    for i in range(cnt):
        wv[i] = (np.float64(lo + i - n_a_int) - n_a_frac) * inv_len
    gvb = S_gv[slot]
    for t in range(n_terms):
        for i in range(cnt):
            gvb[t, i] = _profile(T_i[t, 1], np.float64(lo + i - T_i[t, 0]), T_f[t])
    env = S_env[slot]
    accl = S_lk[slot]
    two_pi = 2.0 * np.pi
    w0 = (np.float64(s0 - n_a_int) - n_a_frac) * inv_len
    d0 = lo - s0  # offset of the first rendered sample from the anchor
    o0 = lo - k0
    for li in range(n_links):
        pa = L_i[li, 0]
        pb = L_i[li, 1]
        if pb == pa:
            continue
        # 1) FD envelopes of the link's groups
        xo = X_i[li, 0]
        xlo = X_i[li, 1]
        for g in range(L_i[li, 2], L_i[li, 3]):
            _envelope_block(
                env[g, 0],
                env[g, 1],
                g,
                lo,
                cnt,
                wv,
                xo,
                xlo,
                xc,
                G_i,
                G_f,
                bank,
                S_acc[slot],
                S_hb[slot],
                S_al[slot],
            )
        # 2) paths in index order; rotator = anchor (closed form at s0) * step power
        ta = L_i[li, 4]
        tb = L_i[li, 5]
        for i in range(cnt):
            accl[i] = 0j
        for p in range(pa, pb):
            aa = P_c[p, 0]
            da = P_c[p, 1]
            rp = P_R[p, d0 : d0 + cnt]
            g = P_g[p]
            er = env[g, 0]
            ei = env[g, 1]
            c = P_f[p, 0] + w0 * P_f[p, 1]
            c = c - np.rint(c)
            ang = two_pi * c
            z0 = complex(np.cos(ang), -np.sin(ang))
            if ta == tb:
                for i in range(cnt):
                    a = aa + wv[i] * da
                    accl[i] = accl[i] + (a * (z0 * rp[i])) * complex(er[i], ei[i])
            else:
                for i in range(cnt):
                    w = wv[i]
                    a = aa + w * da
                    for t in range(ta, tb):
                        gval = gvb[t, i]
                        if gval != 0.0:
                            q = T_i[t, 2] + (p - pa)
                            a = a + gval * (T_c[q, 0] + w * T_c[q, 1])
                    accl[i] = accl[i] + (a * (z0 * rp[i])) * complex(er[i], ei[i])
        # 3) LO rotator (absolute index, anchored at s0) and the addition into out
        if L_i[li, 6]:
            rl = L_R[li, d0 : d0 + cnt]
            c = _frac_cycles(np.int64(s0), L_f[li, 0], L_f[li, 1])
            ang = two_pi * c
            zl0 = complex(np.cos(ang), np.sin(ang))
            for i in range(cnt):
                out[o0 + i] += accl[i] * (zl0 * rl[i])
        else:
            for i in range(cnt):
                out[o0 + i] += accl[i]


@nb.njit(nogil=True, cache=True)
def _scratch(n_slots, n_terms, n_groups, l_max):  # pragma: no cover - numba
    return (
        np.empty((n_slots, BLOCK), dtype=np.float64),
        np.empty((n_slots, max(n_terms, 1), BLOCK), dtype=np.float64),
        np.empty((n_slots, max(n_groups, 1), 2, BLOCK), dtype=np.float64),
        np.empty((n_slots, 8, BLOCK), dtype=np.float64),
        np.empty((n_slots, max(l_max, 1)), dtype=np.float64),
        np.empty((n_slots, BLOCK), dtype=np.float64),
        np.empty((n_slots, BLOCK), dtype=np.complex128),
    )


@nb.njit(parallel=True, nogil=True, cache=True)
def _kernel_par(  # pragma: no cover - numba
    out,
    k0,
    k1,
    n_first,
    n_a_int,
    n_a_frac,
    inv_len,
    L_i,
    L_f,
    L_R,
    X_i,
    xc,
    G_i,
    G_f,
    bank,
    P_f,
    P_c,
    P_R,
    P_g,
    T_i,
    T_f,
    T_c,
    l_max,
    n_chunks,
):
    first_blk = k0 // BLOCK
    n_blk = (k1 - 1) // BLOCK - first_blk + 1
    if n_chunks > n_blk:
        n_chunks = n_blk
    S = _scratch(n_chunks, T_i.shape[0], G_i.shape[0], l_max)
    for c in nb.prange(n_chunks):  # contiguous runs of blocks, one scratch slot per run
        for blk in range(
            first_blk + (c * n_blk) // n_chunks, first_blk + ((c + 1) * n_blk) // n_chunks
        ):
            _render_block(
                c,
                blk,
                out,
                k0,
                k1,
                n_first,
                n_a_int,
                n_a_frac,
                inv_len,
                L_i,
                L_f,
                L_R,
                X_i,
                xc,
                G_i,
                G_f,
                bank,
                P_f,
                P_c,
                P_R,
                P_g,
                T_i,
                T_f,
                T_c,
                S,
            )


@nb.njit(nogil=True, cache=True)
def _kernel_ser(  # pragma: no cover - numba
    out,
    k0,
    k1,
    n_first,
    n_a_int,
    n_a_frac,
    inv_len,
    L_i,
    L_f,
    L_R,
    X_i,
    xc,
    G_i,
    G_f,
    bank,
    P_f,
    P_c,
    P_R,
    P_g,
    T_i,
    T_f,
    T_c,
    l_max,
):
    first_blk = k0 // BLOCK
    n_blk = (k1 - 1) // BLOCK - first_blk + 1
    S = _scratch(1, T_i.shape[0], G_i.shape[0], l_max)
    for bi in range(n_blk):
        _render_block(
            0,
            first_blk + bi,
            out,
            k0,
            k1,
            n_first,
            n_a_int,
            n_a_frac,
            inv_len,
            L_i,
            L_f,
            L_R,
            X_i,
            xc,
            G_i,
            G_f,
            bank,
            P_f,
            P_c,
            P_R,
            P_g,
            T_i,
            T_f,
            T_c,
            S,
        )


# --------------------------------------------------------------------------- table bank

_bank_lock = threading.Lock()
_bank: NDArray[np.float64] = np.zeros(1, dtype=np.float64)  # entry 0: dummy (integer groups)
_bank_index: dict[tuple[int, int, bytes], int] = {}
_bank_by_id: dict[int, tuple[NDArray[np.float64], int]] = {}


def _bank_offset(table: NDArray[np.float64]) -> int:
    """Offset of ``table`` (flattened) in the module table bank, adding it once if new.

    The bank only grows (a new table appends a copy), so offsets never move and the current
    bank always contains every table registered before. Tables are found by identity first
    (``fd.polyphase_table`` returns one shared read-only object per filter) and otherwise by
    shape and content, so a table is never copied per snapshot interval.
    """
    global _bank
    hit = _bank_by_id.get(id(table))
    if hit is not None and hit[0] is table:
        return hit[1]
    key = (int(table.shape[0]), int(table.shape[1]), table.tobytes())
    with _bank_lock:
        off = _bank_index.get(key)
        if off is None:
            off = int(_bank.shape[0])
            _bank = np.concatenate([_bank, np.ascontiguousarray(table, np.float64).reshape(-1)])
            _bank_index[key] = off
        _bank_by_id[id(table)] = (table, off)  # keeps the table alive, so the id stays unique
        return off


def bank_size() -> int:
    """Number of float64 entries in the table bank (diagnostic)."""
    return int(_bank.shape[0])


# --------------------------------------------------------------------------- packing


@dataclass(frozen=True, eq=False)
class _Pack:
    """Kernel arrays of one link (local offsets) or of a link set (global offsets).

    ``L_i`` (links, 7) int64: path range p0, p1; group range g0, g1; term range t0, t1;
    has_lo. ``L_f`` (links, 2): q_hi, q_lo. ``L_R`` (links, BLOCK): LO rotator step powers
    ``exp(+j 2 pi frac(i*q))``. ``G_i`` (groups, 6) int64: dint, taps, center, integer, bank
    offset, n_phases. ``G_f`` (groups, 2): dfrac, dd. ``P_f`` (paths, 2): ca, dc. ``P_c``
    (paths, 2): aa, da. ``P_R`` (paths, BLOCK): path rotator step powers
    ``exp(-j 2 pi frac(i*step))``. ``P_g`` (paths,): global group. ``T_i`` (terms, 3) int64:
    k_start, profile code, amplitude row offset. ``T_f`` (terms,): duration. ``T_c`` (term
    rows, 2): dA_a, dA_b - dA_a. ``units``: work per sample (see :data:`PARALLEL_MIN_WORK`).
    """

    L_i: NDArray[np.int64]
    L_f: NDArray[np.float64]
    L_R: NDArray[np.complex128]
    G_i: NDArray[np.int64]
    G_f: NDArray[np.float64]
    P_f: NDArray[np.float64]
    P_c: NDArray[np.complex128]
    P_R: NDArray[np.complex128]
    P_g: NDArray[np.int64]
    T_i: NDArray[np.int64]
    T_f: NDArray[np.float64]
    T_c: NDArray[np.complex128]
    units: int
    l_max: int

    @property
    def n_paths(self) -> int:
        """Total number of paths."""
        return int(self.P_f.shape[0])


_BLOCK_IDX: Final[NDArray[np.float64]] = np.arange(BLOCK, dtype=np.float64)


def _step_powers(step: NDArray[np.float64], sign: float) -> NDArray[np.complex128]:
    """Rotator step powers ``exp(sign*j*2*pi*frac(i*step))`` for i = 0..BLOCK-1, one row per
    step (``step`` in cycles per sample). Row i is ``r**i`` for the step rotator
    ``r = exp(sign*j*2*pi*step)``, evaluated in closed form."""
    x = np.multiply.outer(np.asarray(step, dtype=np.float64), _BLOCK_IDX)
    x -= np.rint(x)
    ang = (2.0 * np.pi) * x
    out = np.empty(x.shape, dtype=np.complex128)
    out.real = np.cos(ang)
    out.imag = sign * np.sin(ang)
    return out


def _link_pack(st: LinkStatic) -> _Pack:
    """Packed kernel arrays of one link, built once and kept on the memoized static."""
    pk = st.nb_pack
    if isinstance(pk, _Pack):
        return pk
    np_, ng, nt = st.n_paths, st.n_groups, int(st.t_k.shape[0])
    off = _bank_offset(st.table) if st.table is not None else 0
    nph = st.table.shape[0] - 1 if st.table is not None else 1
    L_i = np.array([[0, np_, 0, ng, 0, nt, int(st.has_lo)]], dtype=np.int64)
    L_f = np.array([[st.q_hi, st.q_lo]], dtype=np.float64)
    L_R = _step_powers(np.array([st.lo_step]), 1.0)
    G_i = np.empty((ng, 6), dtype=np.int64)
    G_i[:, 0] = st.g_dint
    G_i[:, 1] = st.taps
    G_i[:, 2] = st.center
    G_i[:, 3] = int(st.table is None)
    G_i[:, 4] = off
    G_i[:, 5] = nph
    G_f = np.empty((ng, 2), dtype=np.float64)
    G_f[:, 0] = st.g_dfrac
    G_f[:, 1] = st.g_dd
    P_f = np.empty((np_, 2), dtype=np.float64)
    P_f[:, 0] = st.p_ca
    P_f[:, 1] = st.p_dc
    P_c = np.empty((np_, 2), dtype=np.complex128)
    P_c[:, 0] = st.p_aa
    P_c[:, 1] = st.p_da
    T_i = np.empty((nt, 3), dtype=np.int64)
    T_i[:, 0] = st.t_k
    T_i[:, 1] = st.t_code
    T_i[:, 2] = np.arange(nt, dtype=np.int64) * np_
    T_c = np.empty((nt * np_, 2), dtype=np.complex128)
    T_c[:, 0] = st.t_dAa.reshape(-1)
    T_c[:, 1] = st.t_ddA.reshape(-1)
    units = (st.taps * ng + 4 * np_ * (1 + nt)) if np_ else 0
    pk = _Pack(
        L_i,
        L_f,
        L_R,
        G_i,
        G_f,
        P_f,
        P_c,
        _step_powers(st.p_step, -1.0),
        st.p_g.astype(np.int64),
        T_i,
        st.t_dur.astype(np.float64),
        T_c,
        int(units),
        int(st.taps),
    )
    st.nb_pack = pk
    return pk


_SET_CACHE_MAX = 256
_set_cache: OrderedDict[tuple[LinkStatic, ...], _Pack] = OrderedDict()
_set_lock = threading.Lock()  # render may be called from several engine threads (nogil)


def _set_pack(statics: tuple[LinkStatic, ...]) -> _Pack:
    """Kernel arrays of a link set: the link's own pack for one link, else the concatenation
    with global offsets (memoized by the identity of the statics, which the entry keeps
    alive)."""
    if len(statics) == 1:
        return _link_pack(statics[0])
    with _set_lock:
        hit = _set_cache.get(statics)
        if hit is not None:
            _set_cache.move_to_end(statics)
            return hit
    packs = [_link_pack(st) for st in statics]
    L_i = np.concatenate([p.L_i for p in packs])
    P_g = []
    T_i = []
    po = go = to = ro = 0
    for i, p in enumerate(packs):
        L_i[i, 0:2] += po
        L_i[i, 2:4] += go
        L_i[i, 4:6] += to
        P_g.append(p.P_g + go)
        ti = p.T_i.copy()
        ti[:, 2] += ro
        T_i.append(ti)
        po += p.P_f.shape[0]
        go += p.G_i.shape[0]
        to += p.T_i.shape[0]
        ro += p.T_c.shape[0]
    pk = _Pack(
        L_i,
        np.concatenate([p.L_f for p in packs]),
        np.concatenate([p.L_R for p in packs]),
        np.concatenate([p.G_i for p in packs]),
        np.concatenate([p.G_f for p in packs]),
        np.concatenate([p.P_f for p in packs]),
        np.concatenate([p.P_c for p in packs]),
        np.concatenate([p.P_R for p in packs]),
        np.concatenate(P_g),
        np.concatenate(T_i),
        np.concatenate([p.T_f for p in packs]),
        np.concatenate([p.T_c for p in packs]),
        sum(p.units for p in packs),
        max(p.l_max for p in packs),
    )
    with _set_lock:
        _set_cache[statics] = pk
        if len(_set_cache) > _SET_CACHE_MAX:
            _set_cache.popitem(last=False)
    return pk


def clear_caches() -> None:
    """Drop the memoized link-set packs and the shared link statics (with their packs).

    The table bank is kept: it holds one copy per distinct FD table and offsets into it must
    stay valid for any static still referenced elsewhere.
    """
    with _set_lock:
        _set_cache.clear()
    _clear_numpy_caches()


# --------------------------------------------------------------------------- public API


def render_prepared(
    seg: PreparedSegment,
    out: NDArray[np.complex128],
    *,
    parallel: bool | None = None,
    n_chunks: int | None = None,
) -> None:
    """numba evaluation of a prepared segment (adds into ``out``).

    ``parallel``: None chooses the driver from the work estimate (:data:`PARALLEL_MIN_WORK`);
    True/False force the ``prange`` or the serial driver. ``n_chunks`` overrides the number of
    contiguous block runs of the parallel driver (default ``numba.get_num_threads()``). Every
    choice executes the same arithmetic per sample, so the output is bitwise the same.
    """
    if seg.n == 0 or not seg.links:
        return
    pk = _set_pack(tuple(pl.static for pl in seg.links))
    if pk.n_paths == 0:
        return
    nl = len(seg.links)
    if nl == 1:
        pl = seg.links[0]
        xc = pl.x
        X_i = np.array([[0, pl.x_lo]], dtype=np.int64)
    else:
        xc = np.concatenate([pl.x for pl in seg.links])
        X_i = np.empty((nl, 2), dtype=np.int64)
        o = 0
        for i, pl in enumerate(seg.links):
            X_i[i, 0] = o
            X_i[i, 1] = pl.x_lo
            o += pl.x.shape[0]
    bank = _bank  # read after packing: contains every table the pack refers to
    work = seg.n * pk.units
    if parallel is None:
        parallel = work >= PARALLEL_MIN_WORK
    if not parallel:
        _call(_kernel_ser, seg, out, pk, X_i, xc, bank)
        return
    if n_chunks is None:
        n_chunks = nb.get_num_threads()
    if n_chunks < 1:
        raise ConfigurationError(f"n_chunks must be >= 1, got {n_chunks}")
    _call(_kernel_par, seg, out, pk, X_i, xc, bank, int(n_chunks))


def _call(
    kern: Callable[..., None],
    seg: PreparedSegment,
    out: NDArray[np.complex128],
    pk: _Pack,
    X_i: NDArray[np.int64],
    xc: NDArray[np.complex64],
    bank: NDArray[np.float64],
    *chunks: int,
) -> None:
    g = seg.grid
    kern(
        out,
        seg.k0,
        seg.k1,
        g.n_first,
        g.n_a_int,
        g.n_a_frac,
        g.inv_len,
        pk.L_i,
        pk.L_f,
        pk.L_R,
        X_i,
        xc,
        pk.G_i,
        pk.G_f,
        bank,
        pk.P_f,
        pk.P_c,
        pk.P_R,
        pk.P_g,
        pk.T_i,
        pk.T_f,
        pk.T_c,
        pk.l_max,
        *chunks,
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
    """``numba_cpu`` AWGN, ``E|w|**2 = sigma**2``; see :func:`sremu.impair.noise.noise`."""
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
