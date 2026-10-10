"""``LegacyTruncationAdapter``: the v1 time base, for the V-CFO teeth (DESIGN.md 5.14, 10.10).

v1 delivered ``int(f_s*tau)`` samples per engine tick while its clock advanced by a full
``tau`` (audit_v1 B6). At 5880 S/s and tau = 2 ms that is 11 samples per 11.76 samples of
time, so a receiver that assumes the nominal rate over-estimates every CFO by
``11.76/11 = 1.06909`` (ANTS Fig. 2; V-CFO-2 slope 1.0691 +- 0.0005). The same
truncation drops the tail: ``int(n/(f_s*tau))`` ticks of 11 samples deliver 2233 of 2389.

The adapter re-times samples produced by the sremu engine (a CFO-free channel output) onto
the v1 clock: delivered sample ``j`` lies in tick ``c = j // spt`` at position
``i = j % spt`` and carries the CFO phase of time ``c*tau + i*dt`` (``dt = 1/f_s`` by
default; v1 itself used ``1/6000`` s inside a tick, ``intra_tick_dt_s``). It is the one
legacy adapter shared by the in-process (V-CFO-2) and two-process (V-CFO-3) studies.

Every entry point raises ``NonPhysicalError`` unless ``allow_nonphysical=True``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from sremu.errors import ConfigurationError, NonPhysicalError
from sremu.timebase import parse_exact

__all__ = ["LegacyTruncationAdapter"]


class LegacyTruncationAdapter:
    """v1 per-tick truncation (see the module docstring).

    ``rate``: the stream rate (S/s; exact or float, as v1 computed it in floats).
    ``tau_s``: the v1 tick (default 2 ms). ``intra_tick_dt_s``: time step between samples
    inside a tick (default ``1/rate``).
    """

    def __init__(
        self,
        rate: Fraction | int | str | float,
        tau_s: Fraction | int | str = Fraction(1, 500),
        *,
        intra_tick_dt_s: float | None = None,
        allow_nonphysical: bool = False,
    ) -> None:
        if not allow_nonphysical:
            raise NonPhysicalError(
                "LegacyTruncationAdapter reproduces a v1 bug; pass allow_nonphysical=True "
                "(teeth and V-CFO studies only, section 5.14)"
            )
        self.rate = float(rate)
        if not self.rate > 0:
            raise ConfigurationError(f"rate must be > 0, got {rate!r}")
        self.tau_s = float(parse_exact(tau_s, name="tau_s"))
        self.samples_per_tick = int(self.rate * self.tau_s)  # v1: int(SR*tau)
        if self.samples_per_tick < 1:
            raise ConfigurationError("tau shorter than one sample")
        self.dt_s = 1.0 / self.rate if intra_tick_dt_s is None else float(intra_tick_dt_s)

    @property
    def apparent_scale(self) -> float:
        """CFO scale seen by a receiver assuming the nominal rate: ``tau*f_s/spt``."""
        return self.tau_s * self.rate / self.samples_per_tick

    def ticks(self, n_requested: int) -> int:
        """v1 ticks for a request of ``n`` samples: ``int(n/(f_s*tau))`` (floor)."""
        return int(n_requested / (self.rate * self.tau_s))

    def delivered(self, n_requested: int) -> int:
        """Samples v1 delivers for a request of ``n``: ``ticks*spt`` (2233 of 2389)."""
        return self.ticks(n_requested) * self.samples_per_tick

    def tick_times(self, n: int) -> NDArray[np.float64]:
        """v1 clock time (s) of delivered samples ``0..n-1``: ``c*tau + i*dt``."""
        j = np.arange(int(n), dtype=np.int64)
        c = j // self.samples_per_tick
        i = j % self.samples_per_tick
        return c.astype(np.float64) * self.tau_s + i.astype(np.float64) * self.dt_s

    def deliver(
        self, y: NDArray[np.complexfloating], cfo_hz: float, *, n_requested: int | None = None
    ) -> NDArray[np.complex64]:
        """What v1 hands the receiver: the first ``delivered(n)`` engine samples, rotated by
        ``exp(+j 2 pi cfo t_v1(j))`` on the v1 clock (``cfo_hz`` = frequency offset of the
        received baseband)."""
        x = np.asarray(y)
        n = x.shape[0] if n_requested is None else int(n_requested)
        m = min(self.delivered(n), x.shape[0])
        t = self.tick_times(m)
        rot = np.exp(2j * np.pi * float(cfo_hz) * t)
        return (x[:m].astype(np.complex128) * rot).astype(np.complex64)
