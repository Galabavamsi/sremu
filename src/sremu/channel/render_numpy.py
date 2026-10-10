"""``numpy_ref`` render backend and the segment preparation shared with ``numba_cpu``.

Render contract (DESIGN.md section 3.4): ``render(job, out)`` ADDS the received signal of
RX samples [k0, k1) into the complex128 accumulator ``out``. For each link (scenario order)
and RX sample n, with ``w(n)``, chord delays and carrier cycles from D-1 (``interp.py``):

    y(n) = e^{+j 2 pi frac(df*n/f_s)} * sum_p A_p(n) e^{-j 2 pi c_p(n)} E_g(p)(n)

* ``A_p(n) = a_a + w(a_b - a_a) + sum_t g_t(n - k_t)(dA_a + w(dA_b - dA_a))`` (complex-linear
  amplitude plus transition terms, section 4.7);
* ``c_p(n) = frac(f_c tau_a) + w*f_c(tau_b - tau_a)`` (exact carrier phase of the chord delay);
* ``E_g(n)`` is the FD-filtered TX history at the group's envelope delay
  ``d_g(n) = tau_g(n)*f_s`` with ``tau_g = (min + max)/2`` of the group's delays at each end
  (narrowband links have one group): ``E = sum_m h_mu[m] x[n - i + c - m]`` with
  ``i = floor(d)``, ``mu = d - i`` and the polyphase table of ``fd.py``, summed as four
  partial sums over m mod 4 combined ``((0+1)+(2+3))``; integer-only delay reads
  ``x[n - floor(d + 1/2)]``;
* ``df = f_c,tx - f_c,rx`` (``lo_offset_hz``) as an exact pure-function rotator.

Summation order (D-3): links in job order, paths by index; the per-link sum is added to
``out``. ``numpy_ref`` evaluates every sample in closed form with direct ``cos``/``sin``
(complex128); it is the oracle for ``numba_cpu``.

Preparation is split into a per-link **static** part (everything that depends only on the
link's arrays and the snapshot interval: anchors, group delays, FD table, steps) and a
per-segment part (the TX window). The static part is memoized under a content key (array
bytes, dtypes and shapes, scalars and the exact interval), so a cache hit can never return
data for different inputs, including arrays mutated in place.

Guards: delays must be finite and >= 0; ``t_b >= t_a``; the segment must lie inside the
snapshot interval ([ceil(t_a f_s), ceil(t_b f_s)), unless the interval is degenerate,
t_b == t_a: a static, event-only interval with w = 0).
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import BackendName, FDSpec, LinkJob, SegmentJob
from sremu.errors import ConfigurationError

from .fd import fd_center, table_for
from .interp import (
    IntervalGrid,
    double_double,
    frac_cycles_of_index,
    frac_of_product,
    group_envelope_delays,
    interval_grid,
    split_product,
    weights,
)
from .transitions import profile_code, profile_values

__all__ = [
    "LinkStatic",
    "NumpyRefBackend",
    "PreparedLink",
    "PreparedSegment",
    "clear_caches",
    "get_backend",
    "link_static",
    "noise",
    "prepare_segment",
    "render",
    "render_prepared",
]

TWO_PI: Final[float] = 2.0 * math.pi
_WINDOW_MARGIN: Final[int] = 2
_PROFILE_NAMES: Final[tuple[str, ...]] = ("step", "linear", "first_order")
_STATIC_CACHE_MAX: Final[int] = 512


@dataclass(eq=False)
class LinkStatic:
    """Everything about one link that does not depend on the segment bounds."""

    link: str
    has_lo: bool
    q_hi: float  # LO offset in cycles per sample, double-double
    q_lo: float
    r_lo: complex  # exp(+j 2 pi q): LO rotator step (numba)
    # groups
    g_dint: NDArray[np.int64]
    g_dfrac: NDArray[np.float64]
    g_dd: NDArray[np.float64]
    fd: FDSpec
    table: NDArray[np.float64] | None  # (n_phases + 1, L) or None for integer delay
    taps: int
    center: int
    # paths
    p_ca: NDArray[np.float64]  # frac(f_c tau_a), cycles in [-1/2, 1/2]
    p_dc: NDArray[np.float64]  # f_c (tau_b - tau_a), cycles over the interval
    p_aa: NDArray[np.complex128]
    p_da: NDArray[np.complex128]
    p_g: NDArray[np.int64]  # local group index
    p_r: NDArray[np.complex128]  # exp(-j 2 pi dc*inv_len): path rotator step (numba)
    # transition terms
    t_k: NDArray[np.int64]
    t_code: NDArray[np.int64]
    t_dur: NDArray[np.float64]
    t_dAa: NDArray[np.complex128]  # (T, P)
    t_ddA: NDArray[np.complex128]  # (T, P)

    @property
    def n_paths(self) -> int:
        """Number of paths P."""
        return int(self.p_ca.shape[0])

    @property
    def n_groups(self) -> int:
        """Number of FD groups."""
        return int(self.g_dint.shape[0])

    def window(self, grid: IntervalGrid, k0: int, k1: int) -> tuple[int, int]:
        """TX index range [lo, hi] (inclusive) needed for RX samples [k0, k1)."""
        ends = np.array([k0, k1 - 1], dtype=np.int64)
        w_ends = weights(grid, ends)
        s = self.g_dfrac[:, None] + w_ends[None, :] * self.g_dd[:, None]
        f = np.floor(s + 0.5) if self.table is None else np.floor(s)
        i_ends = self.g_dint[:, None] + f.astype(np.int64)
        lo = k0 - int(i_ends.max()) - (self.taps - 1 - self.center) - _WINDOW_MARGIN
        hi = (k1 - 1) - int(i_ends.min()) + self.center + _WINDOW_MARGIN
        return lo, hi


@dataclass(eq=False)
class PreparedLink:
    """A link's static part plus its TX window for one segment."""

    static: LinkStatic
    x: NDArray[np.complex64]  # TX window, absolute indices [x_lo, x_lo + len)
    x_lo: int


@dataclass(eq=False)
class PreparedSegment:
    """A validated segment: the D-1 grid and its prepared links."""

    k0: int
    k1: int
    grid: IntervalGrid
    links: list[PreparedLink]

    @property
    def n(self) -> int:
        """Number of samples k1 - k0."""
        return self.k1 - self.k0


def _check_1d(name: str, arr: object, dtype: type[np.generic], n: int | None) -> NDArray:
    if not isinstance(arr, np.ndarray) or arr.ndim != 1:
        raise ConfigurationError(f"{name} must be a 1-D numpy array")
    if arr.dtype != np.dtype(dtype):
        raise ConfigurationError(f"{name} must have dtype {np.dtype(dtype)}, got {arr.dtype}")
    if n is not None and arr.shape[0] != n:
        raise ConfigurationError(f"{name} has length {arr.shape[0]}, expected {n}")
    return arr


def _arr_key(a: object) -> tuple[object, ...]:
    if isinstance(a, np.ndarray):
        return (a.dtype.str, a.shape, a.tobytes())
    return ("obj", repr(a))


def _static_key(lj: LinkJob, t_a: Fraction, t_b: Fraction, fs: Fraction) -> tuple[object, ...]:
    trans = tuple(
        (
            int(t.k_start),
            str(t.profile),
            float(t.duration_samples),
            _arr_key(t.dA_a),
            _arr_key(t.dA_b),
        )
        for t in lj.trans
    )
    return (
        str(lj.link),
        float(lj.fc_hz),
        float(lj.lo_offset_hz),
        lj.fd,
        _arr_key(lj.amp_a),
        _arr_key(lj.amp_b),
        _arr_key(lj.tau_a),
        _arr_key(lj.tau_b),
        _arr_key(lj.group),
        trans,
        t_a,
        t_b,
        fs,
    )


_static_cache: OrderedDict[tuple[object, ...], LinkStatic] = OrderedDict()
_static_lock = threading.Lock()  # render may be called from several engine threads (nogil)


def clear_caches() -> None:
    """Drop the memoized link statics and interval grids (``render_numba.clear_caches``
    also drops its flattenings)."""
    with _static_lock:
        _static_cache.clear()
    _grid.cache_clear()


@lru_cache(maxsize=256)
def _grid(t_a: Fraction, t_b: Fraction, fs: Fraction) -> IntervalGrid:
    return interval_grid(t_a, t_b, fs)


def _build_static(lj: LinkJob, grid: IntervalGrid, fs: Fraction) -> LinkStatic:
    name = f"link {lj.link}"
    aa = _check_1d(f"{name}: amp_a", lj.amp_a, np.complex128, None)
    p = aa.shape[0]
    ab = _check_1d(f"{name}: amp_b", lj.amp_b, np.complex128, p)
    ta = _check_1d(f"{name}: tau_a", lj.tau_a, np.float64, p)
    tb = _check_1d(f"{name}: tau_b", lj.tau_b, np.float64, p)
    grp = lj.group
    if not isinstance(grp, np.ndarray) or grp.shape != (p,) or grp.dtype.kind not in "iu":
        raise ConfigurationError(f"{name}: group must be a ({p},) integer array")
    if p and not (np.all(np.isfinite(ta)) and np.all(np.isfinite(tb))):
        raise ConfigurationError(f"{name}: delays must be finite")
    if p and (float(ta.min()) < 0.0 or float(tb.min()) < 0.0):
        raise ConfigurationError(f"{name}: delays must be >= 0 s (causal paths only)")
    fd = lj.fd
    integer = fd.filter_id == "integer"
    table = None if integer else table_for(fd)
    taps = 1 if integer else int(fd.taps)
    c = 0 if integer else fd_center(taps)
    fc = float(lj.fc_hz)
    fs_f = float(fs)

    if p:
        labels, env_a = group_envelope_delays(ta, grp)
        _, env_b = group_envelope_delays(tb, grp)
        p_g = np.searchsorted(labels, grp.astype(np.int64)).astype(np.int64)
    else:
        env_a = env_b = np.zeros(0)
        p_g = np.zeros(0, np.int64)
    g_dint, g_dfrac = split_product(env_a, fs_f)

    lo_hz = float(lj.lo_offset_hz)
    if not math.isfinite(lo_hz):
        raise ConfigurationError(f"{name}: lo_offset_hz must be finite")
    q_hi, q_lo = double_double(Fraction(lo_hz) / fs) if lo_hz != 0.0 else (0.0, 0.0)
    qred = (q_hi - round(q_hi)) + q_lo
    r_lo = complex(math.cos(TWO_PI * qred), math.sin(TWO_PI * qred))

    nt = len(lj.trans)
    t_k = np.zeros(nt, dtype=np.int64)
    t_code = np.zeros(nt, dtype=np.int64)
    t_dur = np.zeros(nt, dtype=np.float64)
    t_dAa = np.zeros((nt, p), dtype=np.complex128)
    t_ddA = np.zeros((nt, p), dtype=np.complex128)
    for i, tt in enumerate(lj.trans):
        da_ = _check_1d(f"{name}: trans[{i}].dA_a", tt.dA_a, np.complex128, p)
        db_ = _check_1d(f"{name}: trans[{i}].dA_b", tt.dA_b, np.complex128, p)
        dur = float(tt.duration_samples)
        if not math.isfinite(dur):
            raise ConfigurationError(f"{name}: trans[{i}] duration must be finite")
        t_k[i] = int(tt.k_start)
        t_code[i] = profile_code(tt.profile, dur)
        t_dur[i] = dur
        t_dAa[i] = da_
        t_ddA[i] = db_ - da_

    p_dc = (fc * (tb - ta)).astype(np.float64)
    step = p_dc * grid.inv_len
    step = step - np.rint(step)
    p_r = np.empty(p, dtype=np.complex128)
    p_r.real = np.cos(TWO_PI * step)
    p_r.imag = -np.sin(TWO_PI * step)
    return LinkStatic(
        link=lj.link,
        has_lo=lo_hz != 0.0,
        q_hi=float(q_hi),
        q_lo=float(q_lo),
        r_lo=r_lo,
        g_dint=np.atleast_1d(g_dint).astype(np.int64),
        g_dfrac=np.atleast_1d(g_dfrac).astype(np.float64),
        g_dd=np.atleast_1d(fs_f * (env_b - env_a)).astype(np.float64),
        fd=fd,
        table=table,
        taps=taps,
        center=c,
        p_ca=np.atleast_1d(frac_of_product(fc, ta)).astype(np.float64) if p else np.zeros(0),
        p_dc=p_dc,
        p_aa=aa.copy(),
        p_da=ab - aa,
        p_g=p_g,
        p_r=p_r,
        t_k=t_k,
        t_code=t_code,
        t_dur=t_dur,
        t_dAa=t_dAa,
        t_ddA=t_ddA,
    )


def link_static(lj: LinkJob, t_a: Fraction, t_b: Fraction, fs: Fraction) -> LinkStatic:
    """Memoized static preparation of one link for the interval [t_a, t_b)."""
    key = _static_key(lj, t_a, t_b, fs)
    with _static_lock:
        hit = _static_cache.get(key)
        if hit is not None:
            _static_cache.move_to_end(key)
            return hit
    st = _build_static(lj, _grid(t_a, t_b, fs), fs)
    with _static_lock:
        _static_cache[key] = st
        if len(_static_cache) > _STATIC_CACHE_MAX:
            _static_cache.popitem(last=False)
    return st


def prepare_segment(job: SegmentJob) -> PreparedSegment:
    """Validate a segment job and compute everything both backends share (anchors, windows).

    Both backends consume the same prepared arrays, so their per-sample inputs are
    identical (I5).
    """
    k0, k1 = int(job.k0), int(job.k1)
    if k1 < k0:
        raise ConfigurationError(f"segment end {k1} < start {k0}")
    fs = Fraction(job.f_s)
    if fs <= 0:
        raise ConfigurationError(f"sample rate must be > 0, got {fs}")
    t_a, t_b = Fraction(job.t_a), Fraction(job.t_b)
    if t_b < t_a:
        raise ConfigurationError(
            f"snapshot interval end t_b = {t_b} precedes its start t_a = {t_a} "
            "(t_b == t_a is a static, event-only interval)"
        )
    grid = _grid(t_a, t_b, fs)
    if k1 > k0:
        if k0 < grid.n_first:
            raise ConfigurationError(
                f"segment [{k0}, {k1}) starts before its snapshot interval (first sample "
                f"{grid.n_first})"
            )
        if grid.n_end is not None and k1 > grid.n_end:
            raise ConfigurationError(
                f"segment [{k0}, {k1}) ends after its snapshot interval (end {grid.n_end})"
            )
    links = []
    for lj in job.links:
        st = link_static(lj, t_a, t_b, fs)
        if st.n_paths and k1 > k0:
            lo, hi = st.window(grid, k0, k1)
            x = np.ascontiguousarray(lj.ring.read(lo, hi - lo + 1), dtype=np.complex64)
        else:
            lo, x = k0, np.zeros(0, dtype=np.complex64)
        links.append(PreparedLink(st, x, int(lo)))
    return PreparedSegment(k0, k1, grid, links)


def _envelope(
    st: LinkStatic,
    x: NDArray[np.complex64],
    x_lo: int,
    g: int,
    n: NDArray[np.int64],
    w: NDArray[np.float64],
) -> NDArray[np.complex128]:
    s = st.g_dfrac[g] + w * st.g_dd[g]
    if st.table is None:
        i = st.g_dint[g] + np.floor(s + 0.5).astype(np.int64)
        return x[(n - i) - x_lo].astype(np.complex128)
    fi = np.floor(s)
    mu = s - fi
    i = st.g_dint[g] + fi.astype(np.int64)
    nph = st.table.shape[0] - 1
    jf = mu * nph
    j = np.minimum(np.floor(jf).astype(np.int64), nph - 1)
    alpha = jf - j
    base = (n - i + st.center) - x_lo
    xr = x.real.astype(np.float64)
    xi = x.imag.astype(np.float64)
    # four partial sums over m mod 4 (each in increasing m), combined ((0+1)+(2+3))
    acc_re = [np.zeros(n.shape[0], dtype=np.float64) for _ in range(4)]
    acc_im = [np.zeros(n.shape[0], dtype=np.float64) for _ in range(4)]
    tab = st.table
    for m in range(tab.shape[1]):
        t0 = tab[j, m]
        h = t0 + alpha * (tab[j + 1, m] - t0)
        idx = base - m
        acc_re[m & 3] = acc_re[m & 3] + h * xr[idx]
        acc_im[m & 3] = acc_im[m & 3] + h * xi[idx]
    out = np.empty(n.shape[0], dtype=np.complex128)
    out.real = (acc_re[0] + acc_re[1]) + (acc_re[2] + acc_re[3])
    out.imag = (acc_im[0] + acc_im[1]) + (acc_im[2] + acc_im[3])
    return out


def _expj(cycles: NDArray[np.float64], sign: float) -> NDArray[np.complex128]:
    ang = TWO_PI * cycles
    out = np.empty(cycles.shape[0], dtype=np.complex128)
    out.real = np.cos(ang)
    out.imag = sign * np.sin(ang)
    return out


def render_prepared(seg: PreparedSegment, out: NDArray[np.complex128]) -> None:
    """Reference evaluation of a prepared segment (adds into ``out``)."""
    if seg.n == 0:
        return
    n = np.arange(seg.k0, seg.k1, dtype=np.int64)
    w = weights(seg.grid, n)
    for pl in seg.links:
        st = pl.static
        if st.n_paths == 0:
            continue
        env = [_envelope(st, pl.x, pl.x_lo, g, n, w) for g in range(st.n_groups)]
        gv = [
            profile_values(
                _PROFILE_NAMES[int(st.t_code[t])],
                (n - st.t_k[t]).astype(np.float64),
                float(st.t_dur[t]),
            )
            for t in range(st.t_k.shape[0])
        ]
        y = np.zeros(seg.n, dtype=np.complex128)
        for p in range(st.n_paths):
            a = st.p_aa[p] + w * st.p_da[p]
            for t in range(len(gv)):
                a = a + gv[t] * (st.t_dAa[t, p] + w * st.t_ddA[t, p])
            c = st.p_ca[p] + w * st.p_dc[p]
            c = c - np.rint(c)
            y = y + (a * _expj(c, -1.0)) * env[int(st.p_g[p])]
        if st.has_lo:
            y = y * _expj(frac_cycles_of_index(n, st.q_hi, st.q_lo), 1.0)
        out += y


def _check_out(job: SegmentJob, out: NDArray[np.complex128]) -> None:
    if not isinstance(out, np.ndarray) or out.dtype != np.complex128 or out.ndim != 1:
        raise ConfigurationError("out must be a 1-D complex128 accumulator")
    if out.shape[0] != int(job.k1) - int(job.k0):
        raise ConfigurationError(
            f"out has length {out.shape[0]}, expected {int(job.k1) - int(job.k0)}"
        )


def render(job: SegmentJob, out: NDArray[np.complex128]) -> None:
    """``numpy_ref`` render: add the segment's received signal into ``out`` (section 3.4)."""
    _check_out(job, out)
    render_prepared(prepare_segment(job), out)


def noise(rx_key: tuple[int, int], k0: int, n: int, sigma: float) -> NDArray[np.complex64]:
    """``numpy_ref`` AWGN; see :func:`sremu.impair.noise.noise_numpy`."""
    from sremu.impair.noise import noise_numpy

    return noise_numpy(rx_key, k0, n, sigma)


class NumpyRefBackend:
    """The ``numpy_ref`` backend (``contracts.RenderBackend``)."""

    name: BackendName = "numpy_ref"

    def render(self, job: SegmentJob, out: NDArray[np.complex128]) -> None:
        """See :func:`render`."""
        render(job, out)

    def noise(
        self, rx_key: tuple[int, int], k0: int, n: int, sigma: float
    ) -> NDArray[np.complex64]:
        """See :func:`noise`."""
        return noise(rx_key, k0, n, sigma)


def get_backend(name: BackendName) -> NumpyRefBackend:
    """Backend registry: ``"numpy_ref"`` or ``"numba_cpu"`` (imports numba lazily)."""
    if name == "numpy_ref":
        return NumpyRefBackend()
    if name == "numba_cpu":
        from .render_numba import NumbaCpuBackend

        return NumbaCpuBackend()  # type: ignore[return-value]
    raise ConfigurationError(f"unknown backend {name!r} (numpy_ref | numba_cpu)")
