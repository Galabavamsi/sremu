"""Paced mode: wall-clock release, hybrid sleep and deadlines (DESIGN.md section 4.10).

* **Release.** Page ``j`` of a rate group (RX samples ending at ``k1``) is released at
  ``R_j = t0 + k1/f + D_lag`` (wall time; ``D_lag`` default 1 ms of TX input slack).
* **Sleep.** Hybrid: ``sleep(R - now - 1.5 ms)``, then spin. Windows ``time.sleep(0.5 ms)``
  overshoots by about 0.5-0.9 ms [MEASURED], so paced numbers for the paper come from
  Linux. The spin yields the CPU and the GIL on every iteration (``os.sched_yield`` on
  POSIX, ``time.sleep(0)`` = ``Sleep(0)`` on Windows) until :data:`YIELD_GUARD_NS` before
  the release and spins plainly only for that last stretch. A plain spin holds the GIL, so
  a served engine's control and transport threads got it only when a page's compute
  released it or after the 5 ms switch interval; at 0.5 ms pages the paced thread never
  sleeps, it spins between every two pages. (Not ``time.sleep(0)`` on Linux: there it is a
  ``clock_nanosleep`` that returns tens of microseconds late, which would delay releases.)
* **Deadline** = ``R_j + slack``; misses are counted at slack 0, 1 and 5 ms by default
  (5 ms is ACHEM's threshold).
* Late TX data, RX overflow and recorder drops are counted by the emulator
  (``late_samples``, ``underflow_slip_samples``, ``overflow_pages``, ``rec_dropped``).

Wall time is allowed here and in ``metrics/`` only (I7); it never enters data-plane
arithmetic. The clock is injectable (:class:`WallClock`) so tests can drive paced mode
with a deterministic fake clock (RT-18).
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Final, Protocol

from sremu.metrics.rt import DeadlineStats
from sremu.timebase import parse_exact

__all__ = ["SPIN_NS", "YIELD_GUARD_NS", "FakeClock", "Pacer", "SystemClock", "WallClock"]

SPIN_NS: Final[int] = 1_500_000
"""Spin for the last 1.5 ms before a release (hybrid sleep)."""
YIELD_GUARD_NS: Final[int] = 100_000
"""The spin yields the GIL until 0.1 ms before a release, then spins plainly (release
precision: a thread that took the GIL during a yield may hold it for a while)."""


def _sleep0() -> None:
    time.sleep(0)


_yield: Callable[[], None] = getattr(os, "sched_yield", None) or _sleep0
"""Give up the CPU and the GIL once (``os.sched_yield`` on POSIX, ``Sleep(0)`` on Windows)."""


class WallClock(Protocol):
    """Monotonic wall clock in integer nanoseconds."""

    def now_ns(self) -> int:
        """Current monotonic time, ns."""
        ...

    def sleep_until(self, t_ns: int, spin_ns: int = SPIN_NS) -> None:
        """Return at or after ``t_ns`` (hybrid sleep then spin)."""
        ...

    def sleep(self, dt_s: float) -> None:
        """Sleep ``dt_s`` seconds."""
        ...


class SystemClock:
    """``time.perf_counter_ns`` with the hybrid sleep of section 4.10."""

    def now_ns(self) -> int:
        """Current monotonic time, ns."""
        return time.perf_counter_ns()

    def sleep_until(self, t_ns: int, spin_ns: int = SPIN_NS) -> None:
        """``sleep(t - now - spin)``, then spin until ``t``: yielding the GIL until
        :data:`YIELD_GUARD_NS` before ``t``, plainly after that."""
        rem = t_ns - time.perf_counter_ns()
        if rem > spin_ns:
            time.sleep((rem - spin_ns) * 1e-9)
        t_yield = t_ns - YIELD_GUARD_NS
        while time.perf_counter_ns() < t_yield:
            _yield()  # other threads (control requests, transport) run meanwhile
        while time.perf_counter_ns() < t_ns:
            pass

    def sleep(self, dt_s: float) -> None:
        """Plain sleep."""
        if dt_s > 0:
            time.sleep(dt_s)


class FakeClock:
    """Deterministic clock for tests: time advances only through sleeps and
    :meth:`advance` (injected load)."""

    def __init__(self, t0_ns: int = 0) -> None:
        self.t_ns = int(t0_ns)

    def now_ns(self) -> int:
        """Current fake time, ns."""
        return self.t_ns

    def sleep_until(self, t_ns: int, spin_ns: int = SPIN_NS) -> None:
        """Jump to ``t_ns`` when it is in the future."""
        self.t_ns = max(self.t_ns, int(t_ns))

    def sleep(self, dt_s: float) -> None:
        """Advance by ``dt_s`` seconds."""
        self.t_ns += max(0, round(dt_s * 1e9))

    def advance(self, dt_ns: int) -> None:
        """Advance by ``dt_ns`` (simulated compute load)."""
        self.t_ns += int(dt_ns)


class Pacer:
    """Release schedule and deadline accounting of paced mode."""

    def __init__(
        self,
        clock: WallClock,
        *,
        d_lag_s: Fraction | int | str = Fraction(1, 1000),
        slacks_s: Sequence[Fraction | int | str] = (0, Fraction(1, 1000), Fraction(5, 1000)),
        spin_ns: int = SPIN_NS,
    ) -> None:
        self.clock = clock
        self.d_lag_ns = round(parse_exact(d_lag_s, name="d_lag_s") * 10**9)
        self.slacks_ns = tuple(round(parse_exact(s, name="slack") * 10**9) for s in slacks_s)
        self.spin_ns = int(spin_ns)
        self.t0_ns: int | None = None
        self.stats = DeadlineStats(self.slacks_ns)

    def start(self) -> int:
        """Set ``t0`` to now; returns it."""
        self.t0_ns = self.clock.now_ns()
        return self.t0_ns

    def release_ns(self, k1: int, f: Fraction) -> int:
        """Release time of the page ending at RX sample ``k1`` at rate ``f``."""
        if self.t0_ns is None:
            raise RuntimeError("Pacer.start() must be called first")
        return self.t0_ns + round(Fraction(k1) / f * 10**9) + self.d_lag_ns

    def wait_until(self, r_ns: int) -> None:
        """Hybrid sleep until ``r_ns``."""
        self.clock.sleep_until(r_ns, self.spin_ns)

    def page_done(
        self,
        release_ns: int,
        delivered_ns: int,
        *,
        compute_ns: int | None = None,
        page_ns: int | None = None,
        inputs_ns: int | None = None,
    ) -> list[int]:
        """Account one delivered page; returns the slacks (ns) whose deadline it missed."""
        self.stats.page(release_ns, delivered_ns, inputs_ns, compute_ns, page_ns)
        return [s for s in self.slacks_ns if delivered_ns > release_ns + s]
