"""TX history ring (DESIGN.md section 4.5; implements ``contracts.TxRing``).

One ring per TX stream holds the TX history after TX impairments (section 6.3 chain:
full scale/gain -> TX LO -> ring). Absolute TX sample ``k`` lives at ``buffer[k & mask]``.

Semantics:

* The capacity ``H`` is a power of two, ``H = 2**ceil(log2(tau_max*f_s + L_FD + 2P + 1024))``
  (:func:`ring_capacity`).
* ``committed_through()`` is the first absolute index not yet committed. Writes append at
  or after it; a gap is zero-filled (silence before a timed burst), and
  :meth:`TxRingBuffer.idle_until` commits silence without data (``idle_until``/EOB, 4.8).
* ``read(k0, n)`` returns complex64. Indices before 0 (before the stream started) and at or
  after ``committed_through()`` (not yet sent) read as 0, as the contract requires.
  Indices that were committed but have already been overwritten (below ``oldest()``) raise
  ``IndexError``: the data is lost and silently returning zeros would break I6.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import StreamId
from sremu.errors import ConfigurationError
from sremu.timebase import parse_exact

__all__ = ["RING_SLACK", "TxRingBuffer", "ring_capacity"]

RING_SLACK = 1024
"""Extra samples in the ring capacity rule (section 4.5)."""


def ring_capacity(tau_max_s: float, f_s: Fraction | int | str, fd_taps: int, page: int) -> int:
    """Ring length ``H = 2**ceil(log2(tau_max*f_s + L_FD + 2P + 1024))`` (section 4.5).

    ``tau_max_s``: largest propagation delay of any path from this TX, s. ``fd_taps``: FD
    filter length ``L_FD`` (1 for integer delay). ``page``: RX page size ``P`` in samples.
    """
    fs = parse_exact(f_s, name="f_s")
    if fs <= 0:
        raise ConfigurationError(f"sample rate must be > 0, got {fs}")
    if not (tau_max_s >= 0.0 and math.isfinite(tau_max_s)):
        raise ConfigurationError(f"tau_max_s must be finite and >= 0, got {tau_max_s!r}")
    need = tau_max_s * float(fs) + int(fd_taps) + 2 * int(page) + RING_SLACK
    return 1 << max(0, math.ceil(math.log2(need)))


class TxRingBuffer:
    """Circular complex64 TX history of one TX stream (``contracts.TxRing``).

    ``stream``: TX stream id. ``f_s``: TX rate, S/s (exact). ``capacity``: H (power of two).
    """

    def __init__(self, stream: StreamId, f_s: Fraction | int | str, capacity: int) -> None:
        if capacity < 1 or capacity & (capacity - 1):
            raise ConfigurationError(f"ring capacity must be a power of two, got {capacity}")
        self.stream: StreamId = stream
        self.f_s: Fraction = parse_exact(f_s, name="f_s")
        self.capacity: int = int(capacity)
        self.mask: int = self.capacity - 1
        self.buffer: NDArray[np.complex64] = np.zeros(self.capacity, dtype=np.complex64)
        self._head = 0  # first absolute index not yet committed

    # ------------------------------------------------------------------ state
    def committed_through(self) -> int:
        """First absolute index NOT yet committed."""
        return self._head

    def oldest(self) -> int:
        """Smallest absolute index still retained (indices below it were overwritten)."""
        return max(0, self._head - self.capacity)

    # ------------------------------------------------------------------ writing
    def _store(self, k0: int, data: NDArray[np.complex64]) -> None:
        n = data.shape[0]
        if n == 0:
            return
        if n > self.capacity:  # only the newest H samples survive
            k0 += n - self.capacity
            data = data[n - self.capacity :]
            n = self.capacity
        start = k0 & self.mask
        first = min(n, self.capacity - start)
        self.buffer[start : start + first] = data[:first]
        if first < n:
            self.buffer[: n - first] = data[first:]

    def _zero_fill(self, k_from: int, k_to: int) -> None:
        if k_to <= k_from:
            return
        if k_to - k_from >= self.capacity:
            self.buffer[:] = 0
            return
        self._store(k_from, np.zeros(k_to - k_from, dtype=np.complex64))

    def write(self, k0: int, samples: NDArray[np.complexfloating]) -> None:
        """Write samples at absolute indices [k0, k0 + len) (cast to complex64).

        ``k0`` must be >= ``committed_through()``: committed samples are never rewritten
        (lockstep determinism). A gap before ``k0`` is committed as silence.
        """
        k0 = int(k0)
        if k0 < self._head:
            raise ValueError(
                f"ring {self.stream}: write at {k0} < committed_through {self._head} "
                "(committed samples are immutable)"
            )
        data = np.ascontiguousarray(samples, dtype=np.complex64).reshape(-1)
        self._zero_fill(self._head, k0)
        self._store(k0, data)
        self._head = k0 + data.shape[0]

    def idle_until(self, k: int) -> None:
        """Commit silence through absolute index ``k`` (exclusive); no-op if already there."""
        k = int(k)
        if k > self._head:
            self._zero_fill(self._head, k)
            self._head = k

    # ------------------------------------------------------------------ reading
    def read(
        self, k0: int, n: int, out: NDArray[np.complex64] | None = None
    ) -> NDArray[np.complex64]:
        """Read absolute indices [k0, k0 + n) as complex64 (see the module semantics)."""
        k0 = int(k0)
        n = int(n)
        if n < 0:
            raise ValueError(f"read length must be >= 0, got {n}")
        if out is None:
            out = np.zeros(n, dtype=np.complex64)
        else:
            if out.shape != (n,) or out.dtype != np.complex64:
                raise ValueError(f"out must be a ({n},) complex64 array")
            out[:] = 0
        lo = max(k0, 0)
        hi = min(k0 + n, self._head)
        if hi <= lo:
            return out
        if lo < self.oldest():
            raise IndexError(
                f"ring {self.stream}: samples [{lo}, {min(hi, self.oldest())}) were overwritten "
                f"(capacity {self.capacity}, committed through {self._head})"
            )
        start = lo & self.mask
        cnt = hi - lo
        first = min(cnt, self.capacity - start)
        o = lo - k0
        out[o : o + first] = self.buffer[start : start + first]
        if first < cnt:
            out[o + first : o + cnt] = self.buffer[: cnt - first]
        return out
