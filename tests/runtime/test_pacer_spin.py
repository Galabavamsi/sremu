"""Paced mode: the hybrid sleep's spin yields the GIL (DESIGN.md section 4.10).

At 0.5 ms pages the paced engine thread never sleeps between pages; it spins. A spin that
holds the GIL leaves a served engine's control thread (requests) and transport threads
waiting for the interpreter's switch interval; the spin of ``SystemClock.sleep_until``
yields the GIL until ``YIELD_GUARD_NS`` before the release and still never returns early.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time

import pytest

from sremu.engine.pacer import YIELD_GUARD_NS, SystemClock

pytestmark = pytest.mark.runtime


def test_spin_lets_another_thread_take_the_gil_promptly() -> None:
    """With a 50 ms switch interval a GIL-holding spin delays a thread that wakes from a
    2 ms sleep by tens of ms; the yielding spin by about a millisecond at most."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(0.05)
    late_ns: list[int] = []

    def sleeper() -> None:
        for _ in range(5):
            t = time.perf_counter_ns()
            time.sleep(0.002)  # GIL released; needs it back to go on
            late_ns.append(time.perf_counter_ns() - t - 2_000_000)

    try:
        th = threading.Thread(target=sleeper, daemon=True)
        th.start()
        # the whole wait is spin (spin_ns larger than the wait), longer than the sleeper
        SystemClock().sleep_until(time.perf_counter_ns() + 400_000_000, spin_ns=10**12)
        th.join(10.0)
    finally:
        sys.setswitchinterval(old)
    assert len(late_ns) == 5
    assert statistics.median(late_ns) < 15_000_000, late_ns


@pytest.mark.parametrize("dt_ns", [0, 50_000, YIELD_GUARD_NS + 50_000, 3_000_000])
def test_sleep_until_never_returns_early(dt_ns: int) -> None:
    clock = SystemClock()
    for _ in range(20):
        t = time.perf_counter_ns() + dt_ns
        clock.sleep_until(t)
        assert time.perf_counter_ns() >= t
