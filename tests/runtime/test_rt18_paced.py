"""RT-18: paced-mode counters (DESIGN.md sections 4.10, 10.3, 12.1).

A deterministic :class:`~sremu.engine.pacer.FakeClock` drives the pacer: compute takes no
fake time except an injected load, so every count is exact.

* injected sleep load -> the deadline-miss count at slack 0 equals the injected count (and
  is 0 at 1 and 5 ms slack);
* a TX burst in the past -> async ``TIME_ERROR`` and ``late_samples``; an untimed stream
  without data -> zeros, ``UNDERFLOW`` and ``underflow_slip_samples``;
* an unread RX stream -> pages dropped (``overflow_pages``) and the next read after the gap
  returns ``OVERFLOW`` with the post-gap ``time_spec``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import StreamCmd, StreamSpec, TxMetadata
from sremu.engine.pacer import FakeClock, Pacer
from sremu.engine.streams import ToneSource
from sremu.errors import AsyncCode, ErrorCode
from sremu.timebase import TimeSpec
from tests.runtime._engine_support import lab_like_fake, make_engine, waveform

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
P = 512


def _paced(clock: FakeClock, **kw: object):
    return make_engine(
        lab_like_fake(), backend="numpy_ref", mode="paced", coupling="barrier", clock=clock, **kw
    )


def test_rt18_injected_load_miss_count_equals_injected_count() -> None:
    clock = FakeClock()
    eng = _paced(clock, traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)}, always_active=True)
    loaded = {3, 8, 21}
    seen: set[int] = set()

    def load(rx: str, k0: int, k1: int) -> None:
        j = k0 // P
        if rx == "rx1.rx0" and j in loaded and j not in seen:
            seen.add(j)
            clock.advance(300_000)  # 0.3 ms of "compute" (> slack 0, < 1 ms)

    eng.on_page(load)
    s = eng.run_paced(Fraction(30 * P) / F)
    assert seen == loaded
    assert s.paced is not None
    assert s.paced["pages"] == 30
    assert s.paced["misses"] == {"0ms": 3, "1ms": 0, "5ms": 0}
    assert s.counters["deadline_misses"] == 3
    lat = s.paced["latency_us"]
    assert lat["max"] == pytest.approx(300.0)
    assert lat["count"] == 30


def test_rt18_release_times_follow_t0_k1_over_f_plus_dlag() -> None:
    clock = FakeClock(1_000_000_000)
    p = Pacer(clock, d_lag_s="1e-3")
    t0 = p.start()
    assert p.release_ns(512, F) == t0 + 512_000 + 1_000_000
    p.wait_until(p.release_ns(512, F))
    assert clock.now_ns() == t0 + 1_512_000


def test_rt18_past_burst_time_error_and_untimed_underflow() -> None:
    clock = FakeClock()
    eng = _paced(clock, bw=8.0e5, always_active=True)
    htx = eng.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
    sent = {"late": 0}

    def inject(rx: str, k0: int, k1: int) -> None:
        if rx == "rx1.rx0" and k0 == 10 * P and not sent["late"]:
            sent["late"] = 1
            got = eng.submit_tx(
                htx, waveform(40), TxMetadata(True, TimeSpec.from_ticks(1000, F), True, True)
            )
            assert got == 0

    eng.on_page(inject)
    s = eng.run_paced(Fraction(20 * P) / F)
    assert s.counters["late_samples"] == 40
    # no client data at all: every needed TX sample was zero-filled at release
    assert s.counters["underflow_slip_samples"] == eng.tx["ap.tx0"].ring.committed_through()
    assert s.counters["underflow_slip_samples"] >= 20 * P
    codes = []
    while (ev := eng.recv_async(htx, 0.0)) is not None:
        codes.append((ev.event_code, ev.sample_index, ev.n_samples))
    assert (AsyncCode.TIME_ERROR, 1000, 40) in codes
    under = [c for c in codes if c[0] == AsyncCode.UNDERFLOW]
    assert sum(c[2] for c in under) == s.counters["underflow_slip_samples"]


def test_rt18_unread_rx_overflow_with_time_spec_jump() -> None:
    clock = FakeClock()
    eng = _paced(clock, traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)}, rx_queue_pages=2)
    h = eng.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
    eng.stream_cmd(h, StreamCmd("start_cont", stream_now=True))
    s = eng.run_paced(Fraction(10 * P) / F)
    assert s.counters["overflow_pages"] == 8
    d0, m0 = eng.read_rx(h, P, 0.0)
    d1, m1 = eng.read_rx(h, P, 0.0)
    assert (d0.size, d1.size) == (P, P)
    assert m0.time_spec == TimeSpec.from_ticks(0, F)
    assert m1.time_spec == TimeSpec.from_ticks(P, F)
    d2, m2 = eng.read_rx(h, P, 0.0)
    assert d2.size == 0
    assert m2.error_code == ErrorCode.OVERFLOW
    assert m2.time_spec == TimeSpec.from_ticks(10 * P, F)  # post-gap time: the jump
    assert m2.time_spec.to_ticks(F) - m1.time_spec.to_ticks(F) - P == 8 * P


def test_rt18_paced_without_misses_equals_lockstep_bitwise() -> None:
    """I2a: paced without misses gives the same RX as lockstep."""

    def run(mode: str) -> np.ndarray:
        clock = FakeClock()
        eng = make_engine(
            lab_like_fake(moving=True),
            backend="numba_cpu",
            mode=mode,
            noise=True,
            render_tiles="auto",
            clock=clock,
            always_active=True,
            traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        )
        cap = eng.capture_accumulator("rx1.rx0")
        if mode == "paced":
            s = eng.run_paced(Fraction(20 * P) / F)
            assert s.paced is not None
            assert s.paced["misses"]["0ms"] == 0
        else:
            eng.run(Fraction(20 * P) / F)
        return np.concatenate([y for _, y in cap])

    a, b = run("lockstep"), run("paced")
    assert np.array_equal(a.view(np.float64), b.view(np.float64))


def test_paced_two_rate_groups_with_barrier_controller_matches_lockstep() -> None:
    """paced + barrier (in-process controller) over two rate groups: no livelock at the
    barrier limits, lateness 0, and (without misses) the same RX as lockstep."""
    from tests.runtime._engine_support import ScriptedController, alternate_script
    from tests.runtime.test_rt08_rt17_barriers import _two_rate_scene

    ra, rb = Fraction(10**6), Fraction(768_000)

    def run(mode: str) -> dict[str, np.ndarray]:
        from tests.runtime._engine_support import captured

        eng = make_engine(
            _two_rate_scene(ra, rb),
            backend="numpy_ref",
            mode=mode,
            clock=FakeClock(),
            always_active=True,
            traffic={"ta.tx0": ToneSource(ra, 0.0, 1.0), "tb.tx0": ToneSource(rb, 0.0, 1.0)},
        )
        n = eng.commands.specs["ris0"].n_controls
        eng.register_controller(
            ScriptedController("alt", "1/1000", "2/10000", alternate_script("ris0", n))
        )
        caps = {r: eng.capture_accumulator(r) for r in ("ra.rx0", "rb.rx0")}
        if mode == "paced":
            eng.run_paced(Fraction(5, 1000))
        else:
            eng.run(Fraction(5, 1000))
        recs = [r for r in eng.records if r.src == "alt"]
        assert len(recs) == 5
        assert all(v == 0 for r in recs for v in r.lateness_samples.values())
        return {
            "ra.rx0": captured(caps["ra.rx0"], 0, 5000),
            "rb.rx0": captured(caps["rb.rx0"], 0, 3840),
        }

    a, b = run("lockstep"), run("paced")
    for r in a:
        assert np.array_equal(a[r].view(np.float64), b[r].view(np.float64))
