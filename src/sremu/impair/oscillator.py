"""Local-oscillator impairments: CFO and Wiener phase noise (DESIGN.md section 6.3).

**CFO** (class (a), MUST). ``phi(k) = 2*pi*frac(df*k/f_s)`` for absolute sample index k.
``df/f_s`` is converted exactly (``Fraction(df)/f_s``) into a double-double, and
``frac(k*q)`` uses an error-free product, so the phase is exact to about 1e-16 cycles for any
k < 2**53. The phasor is ``anchor(block) * table[m]`` with ``block = k // 1024``,
``m = k % 1024``: one closed-form anchor per **absolute** 1024-sample block times a
1024-entry table, hence a pure function of k (chunking cannot change a bit; RT-09, G4).

Sign convention: :func:`cfo_phasor` returns ``exp(+j*phi(k))``. A TX LO that is ``df`` Hz
high multiplies TX baseband by ``exp(+j phi)``; an RX LO ``df`` Hz high multiplies RX
baseband by ``exp(-j phi)``, i.e. use ``cfo_phasor(-df, ...)`` (:func:`apply_lo`).

**Wiener phase noise** (class (b), SHOULD). Linewidth beta (Hz): ``Var phi(t) = 2*pi*beta*t``,
independent of f_s (fixes v2 P4). Block increments ``B_j ~ N(0, 2*pi*beta*M/f_s)`` keyed by
the absolute block index j (Philox lane 1) give the block-start phases ``Phi_j`` as a
sequential prefix sum (cached at checkpoints, so any query order yields the same bits). Inside
block j a Brownian bridge from per-sample keyed variates ``e_k ~ N(0, 2*pi*beta/f_s)``
(Philox lane 2, counter = absolute sample index) interpolates:
``phi(jM + m) = Phi_j + (m/M)(Phi_{j+1} - Phi_j) + S_m - (m/M) S_M`` with
``S_m = sum_{i<m} e_{jM+i}``. The block is a fixed M = 1024 samples on the absolute grid
(not the RX page), so changing ``page_target_s`` cannot change the realization [ENG].
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from sremu.channel.interp import double_double, frac_cycles_of_index
from sremu.errors import ConfigurationError
from sremu.timebase import parse_exact

from .noise import normals_numba, normals_numpy

__all__ = [
    "CFO_BLOCK",
    "PN_BLOCK",
    "PN_LANE_BLOCK",
    "PN_LANE_BRIDGE",
    "WienerPhaseNoise",
    "apply_cfo",
    "apply_lo",
    "cfo_cycles",
    "cfo_phasor",
    "cycles_per_sample",
]

CFO_BLOCK: Final[int] = 1024
"""CFO anchor block and table length (absolute grid)."""

PN_BLOCK: Final[int] = 1024
"""Wiener phase-noise block M (absolute grid) [ENG]."""

PN_LANE_BLOCK: Final[int] = 1
"""Philox lane of the block increments (counter = block index)."""

PN_LANE_BRIDGE: Final[int] = 2
"""Philox lane of the per-sample bridge variates (counter = sample index)."""

_CHECKPOINT: Final[int] = 256  # blocks between cached block-phase checkpoints


def cycles_per_sample(df_hz: float, f_s: Fraction | int | str) -> tuple[float, float]:
    """``df/f_s`` cycles per sample as an exact double-double ``(hi, lo)``."""
    fs = parse_exact(f_s, name="f_s")
    if fs <= 0:
        raise ConfigurationError(f"sample rate must be > 0, got {fs}")
    df = float(df_hz)
    if not math.isfinite(df):
        raise ConfigurationError(f"frequency offset must be finite, got {df_hz!r}")
    return double_double(Fraction(df) / fs)


def _indices(k0: int, n: int) -> NDArray[np.int64]:
    if int(n) < 0:
        raise ConfigurationError(f"length must be >= 0, got {n}")
    if abs(int(k0)) + int(n) >= 1 << 53:
        raise ConfigurationError("sample index beyond 2**53")
    return np.arange(int(k0), int(k0) + int(n), dtype=np.int64)


def cfo_cycles(df_hz: float, f_s: Fraction | int | str, k0: int, n: int) -> NDArray[np.float64]:
    """``frac(df*k/f_s)`` in [-1/2, 1/2] cycles for k in [k0, k0 + n) (closed form)."""
    q_hi, q_lo = cycles_per_sample(df_hz, f_s)
    return frac_cycles_of_index(_indices(k0, n), q_hi, q_lo)


@lru_cache(maxsize=64)
def _cfo_table(q_hi: float, q_lo: float) -> NDArray[np.complex128]:
    c = frac_cycles_of_index(np.arange(CFO_BLOCK, dtype=np.int64), q_hi, q_lo)
    t = np.exp(2j * np.pi * c)
    t.setflags(write=False)
    return t


def cfo_phasor(df_hz: float, f_s: Fraction | int | str, k0: int, n: int) -> NDArray[np.complex128]:
    """``exp(+j*2*pi*frac(df*k/f_s))`` for k in [k0, k0 + n): absolute anchor x table."""
    k = _indices(k0, n)
    if n == 0:
        return np.zeros(0, dtype=np.complex128)
    q_hi, q_lo = cycles_per_sample(df_hz, f_s)
    if q_hi == 0.0 and q_lo == 0.0:
        return np.ones(int(n), dtype=np.complex128)
    blk = np.floor_divide(k, CFO_BLOCK)
    m = k - blk * CFO_BLOCK
    ub, inv = np.unique(blk, return_inverse=True)
    anchors = np.exp(2j * np.pi * frac_cycles_of_index(ub * CFO_BLOCK, q_hi, q_lo))
    return anchors[inv] * _cfo_table(q_hi, q_lo)[m]


def apply_cfo(
    x: NDArray[np.complexfloating], df_hz: float, f_s: Fraction | int | str, k0: int
) -> NDArray[np.complex64]:
    """``x * exp(+j*phi(k))`` with x starting at absolute index ``k0`` (complex64 out)."""
    xx = np.asarray(x)
    return (xx.astype(np.complex128) * cfo_phasor(df_hz, f_s, k0, xx.shape[0])).astype(np.complex64)


class WienerPhaseNoise:
    """Wiener phase noise of one LO (see the module docstring).

    ``key``: the two uint32 Philox key words of purpose ``pn/<node>.<lo>``.
    ``linewidth_hz``: beta >= 0 (Hz). ``f_s``: the stream rate (exact). ``backend`` selects
    the normal generator (both give bitwise-identical uint32 streams).
    """

    def __init__(
        self,
        key: tuple[int, int],
        linewidth_hz: float,
        f_s: Fraction | int | str,
        *,
        block: int = PN_BLOCK,
        backend: Literal["numpy_ref", "numba_cpu"] = "numba_cpu",
    ) -> None:
        if not (linewidth_hz >= 0.0 and math.isfinite(linewidth_hz)):
            raise ConfigurationError(f"linewidth must be finite and >= 0 Hz, got {linewidth_hz!r}")
        if block < 1 or block & (block - 1):
            raise ConfigurationError(f"PN block must be a power of two, got {block}")
        self.key = (int(key[0]), int(key[1]))
        self.linewidth_hz = float(linewidth_hz)
        self.f_s = parse_exact(f_s, name="f_s")
        self.block = int(block)
        self._normals = normals_numba if backend == "numba_cpu" else normals_numpy
        var_step = 2.0 * math.pi * self.linewidth_hz / float(self.f_s)
        self.sigma_step = math.sqrt(var_step)
        self.sigma_block = math.sqrt(var_step * self.block)
        self._ckpt: list[float] = [0.0]  # Phi at blocks 0, C, 2C, ...

    def _increments(self, j0: int, count: int) -> NDArray[np.float64]:
        return self.sigma_block * self._normals(self.key, PN_LANE_BLOCK, j0, count)

    def _span(self, j0: int, count: int) -> NDArray[np.float64]:
        """Phi at blocks j0 .. j0 + count (inclusive), j0 a checkpoint index."""
        base = self._ckpt[j0 // _CHECKPOINT]
        seq = np.empty(count + 1, dtype=np.float64)
        seq[0] = base
        seq[1:] = self._increments(j0, count)
        return np.cumsum(seq)

    def _ensure(self, ci: int) -> None:
        while len(self._ckpt) <= ci:
            i = len(self._ckpt) - 1
            self._ckpt.append(float(self._span(i * _CHECKPOINT, _CHECKPOINT)[-1]))

    def block_phases(self, j_first: int, j_last: int) -> NDArray[np.float64]:
        """``Phi_j`` for j in [j_first, j_last] (inclusive), rad."""
        if j_first < 0 or j_last < j_first:
            raise ConfigurationError("block range must satisfy 0 <= first <= last")
        out = np.empty(j_last - j_first + 1, dtype=np.float64)
        j = j_first
        while j <= j_last:
            ci = j // _CHECKPOINT
            self._ensure(ci)
            stop = min(j_last, (ci + 1) * _CHECKPOINT)
            # a prefix of the sequential cumsum: bitwise equal to the full-span values
            span = self._span(ci * _CHECKPOINT, stop - ci * _CHECKPOINT)
            out[j - j_first : stop - j_first + 1] = span[
                j - ci * _CHECKPOINT : stop - ci * _CHECKPOINT + 1
            ]
            j = stop + 1
        return out

    def phase(self, k0: int, n: int) -> NDArray[np.float64]:
        """Phase phi(k) in rad for absolute samples [k0, k0 + n) (pure in k)."""
        if int(k0) < 0 or int(n) < 0:
            raise ConfigurationError("phase noise needs k0 >= 0 and n >= 0")
        n = int(n)
        if n == 0 or self.linewidth_hz == 0.0:
            return np.zeros(n, dtype=np.float64)
        m_blk = self.block
        b0 = int(k0) // m_blk
        b1 = (int(k0) + n - 1) // m_blk
        nb_ = b1 - b0 + 1
        phi = self.block_phases(b0, b1 + 1)
        e = self.sigma_step * self._normals(self.key, PN_LANE_BRIDGE, b0 * m_blk, nb_ * m_blk)
        e = e.reshape(nb_, m_blk)
        s = np.zeros((nb_, m_blk + 1), dtype=np.float64)
        s[:, 1:] = np.cumsum(e, axis=1)
        frac = np.arange(m_blk, dtype=np.float64) / m_blk
        full = (
            phi[:-1, None]
            + frac[None, :] * (phi[1:, None] - phi[:-1, None])
            + (s[:, :-1] - frac[None, :] * s[:, -1:])
        ).reshape(-1)
        off = int(k0) - b0 * m_blk
        return full[off : off + n].copy()

    def phasor(self, k0: int, n: int) -> NDArray[np.complex128]:
        """``exp(+j*phi(k))`` for absolute samples [k0, k0 + n)."""
        ph = self.phase(k0, n)
        return np.cos(ph) + 1j * np.sin(ph)


def apply_lo(
    x: NDArray[np.complexfloating],
    k0: int,
    f_s: Fraction | int | str,
    *,
    cfo_hz: float = 0.0,
    pn: WienerPhaseNoise | None = None,
    direction: Literal["tx", "rx"] = "tx",
) -> NDArray[np.complex64]:
    """Apply an LO's CFO and phase noise to samples starting at absolute index ``k0``.

    A TX LO multiplies by ``exp(+j(phi_cfo + phi_pn))``, an RX LO by the conjugate.
    """
    xx = np.asarray(x).astype(np.complex128)
    n = xx.shape[0]
    rot = cfo_phasor(cfo_hz, f_s, k0, n) if cfo_hz != 0.0 else np.ones(n, dtype=np.complex128)
    if pn is not None:
        rot = rot * pn.phasor(k0, n)
    if direction == "rx":
        rot = np.conj(rot)
    elif direction != "tx":
        raise ConfigurationError(f"direction must be 'tx' or 'rx', got {direction!r}")
    return (xx * rot).astype(np.complex64)
