"""RT-10 skip invariance, RT-13 TX commitment and stalls, RT-15 (I7) wall-clock independence.

DESIGN.md sections 4.3 (D-6), 4.8, 4.2 (I7), 10.3.
"""

from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import StreamCmd, StreamSpec, TxMetadata
from sremu.engine.pacer import FakeClock
from sremu.engine.streams import ToneSource
from sremu.errors import AsyncCode, ErrorCode, NondeterminismWarning, SremuTimeout
from sremu.timebase import TimeSpec
from tests.runtime._engine_support import (
    Collect,
    captured,
    lab_like_fake,
    make_engine,
    waveform,
)

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
CFO = {"ap.tx0": -55.5, "rx1.rx0": 310.0}


# --------------------------------------------------------------------------- RT-10


def _skip_engine(backend: str, **kw: object):
    return make_engine(
        lab_like_fake(moving=True, horizon="1"),
        backend=backend,
        noise=True,
        bw=8.0e5,
        cfo=CFO,
        render_tiles="auto",
        traffic={"ap.tx0": ToneSource(F, 2.5e3, 0.5)},
        seed=5,
        **kw,
    )


def test_rt10_skip_invariance_inactive_then_active_bitwise(backend_name: str) -> None:
    """RX inactive for 0.3 s (no consumer: pages skipped, D-6), then active, equals the
    always-active run on the active part."""
    k_on = 300_000
    n = 20_000
    eng = _skip_engine(backend_name)
    h = eng.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
    eng.stream_cmd(
        h, StreamCmd("num_done", num_samps=n, stream_now=False, time_spec=TimeSpec(0, 0.3))
    )
    got = []
    while sum(x.size for x in got) < n:
        d, md = eng.read_rx(h, 4096, 0.0)
        assert md.error_code in (ErrorCode.NONE, ErrorCode.TIMEOUT)
        got.append(d)
    y_skip = np.concatenate(got)
    assert eng.groups[0].cursor >= k_on
    ref = _skip_engine(backend_name, always_active=True)
    sink = Collect()
    ref.add_sink(sink)
    ref.run(Fraction(k_on + n) / F)
    y_ref = sink.rx_array("rx1.rx0", k_on, k_on + n)
    assert np.array_equal(y_skip.view(np.uint64), y_ref.view(np.uint64))


def test_rt10_skipped_pages_are_not_computed() -> None:
    """Teeth for RT-10: the skipping run really skipped (no RX samples computed before
    0.3 s), so the equality above is not trivial."""
    eng = _skip_engine("numpy_ref")
    computed: list[int] = []
    eng.on_page(lambda rx, k0, k1: computed.append(k0))
    h = eng.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
    eng.stream_cmd(
        h, StreamCmd("num_done", num_samps=1000, stream_now=False, time_spec=TimeSpec(0, 0.3))
    )
    while True:
        d, _md = eng.read_rx(h, 1000, 0.0)
        if d.size:
            break
    assert computed
    assert min(computed) >= 300_000 - 512  # only pages holding wanted samples were computed
    assert eng.groups[0].cursor >= 300_000


# --------------------------------------------------------------------------- RT-13


def _client_engine(**kw: object):
    eng = make_engine(lab_like_fake(), backend="numpy_ref", bw=8.0e5, **kw)
    htx = eng.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
    hrx = eng.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
    eng.stream_cmd(hrx, StreamCmd("start_cont", stream_now=True))
    return eng, htx, hrx


def _drain(eng, h, n_max: int = 10**6) -> int:
    total = 0
    while total < n_max:
        d, md = eng.read_rx(h, 100_000, 0.0)
        if d.size == 0:
            assert md.error_code == ErrorCode.TIMEOUT
            break
        total += d.size
    return total


def test_rt13_commitment_by_samples_idle_until_and_eob() -> None:
    """RX [k0, k1) needs TX committed through k1 + h_FD - floor(tau_min f) (32 taps at
    B = 0.8 f_s: h_FD = 16; tau_min f < 1 here)."""
    eng, htx, hrx = _client_engine()
    x = waveform(5000)
    assert eng.submit_tx(htx, x, TxMetadata(True, TimeSpec(0, 0), True, False)) == 5000
    got = _drain(eng, hrx)
    assert got == 5000 - 16  # samples covering K: exactly the causal limit of section 4.5
    # idle_until(K) commits silence through K
    eng.submit_tx(
        htx,
        np.zeros(0, np.complex64),
        TxMetadata(True, TimeSpec.from_ticks(9000, F), idle_until=True),
    )
    got += _drain(eng, hrx)
    assert got == 9000 - 16
    # EOB: silent until the next timed burst -> computable indefinitely (bounded here by
    # the drain); a later timed burst in the future is still accepted
    eng.submit_tx(htx, x[:100], TxMetadata(True, TimeSpec.from_ticks(9000, F), True, True))
    got += _drain(eng, hrx, 20_000)
    assert got >= 20_000
    ack = eng.recv_async(htx, 0.0)
    assert ack is not None
    assert ack.event_code == AsyncCode.BURST_ACK


def test_rt13_late_timed_burst_time_error_counted() -> None:
    eng, htx, hrx = _client_engine()
    eng.submit_tx(htx, waveform(3000), TxMetadata(True, TimeSpec(0, 0), True, True))
    _drain(eng, hrx, 2000)
    late = eng.submit_tx(
        htx, waveform(50), TxMetadata(True, TimeSpec.from_ticks(100, F), True, True)
    )
    assert late == 0
    assert eng.counters["late_samples"] == 50
    ev = eng.recv_async(htx, 0.0)
    while ev is not None and ev.event_code != AsyncCode.TIME_ERROR:
        ev = eng.recv_async(htx, 0.0)
    assert ev is not None
    assert ev.sample_index == 100
    assert ev.n_samples == 50


def test_rt13_stall_report_then_wait_raises_without_external_inputs() -> None:
    eng = make_engine(
        lab_like_fake(), backend="numpy_ref", bw=8.0e5, unopened_tx="wait", always_active=True
    )
    with pytest.raises(SremuTimeout) as ei:
        eng.run(Fraction(1, 1000), stall_timeout_s=0.05)
    assert len(eng.stall_reports) == 1
    rep = eng.stall_reports[0].to_dict()
    g = rep["groups"][0]
    assert g["cursor"] == 0
    assert "ap.tx0" in g["waiting_on_tx"]
    assert "ap.tx0" in str(ei.value)


def test_rt13_stall_zero_fill_counted_and_flagged() -> None:
    sink = Collect()
    eng = make_engine(
        lab_like_fake(),
        backend="numpy_ref",
        bw=8.0e5,
        unopened_tx="wait",
        always_active=True,
        on_stall="zero_fill",
        sinks=[sink],
    )
    with pytest.warns(NondeterminismWarning):
        s = eng.run(Fraction(1, 1000), stall_timeout_s=0.02)
    assert s.counters["stall_zero_samples"] > 0
    # exactly the TX samples committed by the zero fill: through the last page's need
    assert s.counters["stall_zero_samples"] == eng.tx["ap.tx0"].ring.committed_through()
    assert s.nondeterministic_inputs
    assert s.stalls >= 1
    assert len(sink.stalls) == s.stalls
    assert eng.groups[0].cursor == 1000


# --------------------------------------------------------------------------- RT-15 (I7)


class _GarbageClock(FakeClock):
    """A wall clock that jumps erratically (I7: it must not change the data plane)."""

    def __init__(self) -> None:
        super().__init__(123_456_789)
        self.calls = 0

    def now_ns(self) -> int:
        self.calls += 1
        self.t_ns += (self.calls * 7_919_003) % 104_729_000
        return self.t_ns


def test_rt15_monkeypatched_clock_identical_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run once with the system clock and once with an erratic injected clock while
    ``time.perf_counter_ns``/``time.monotonic`` are monkeypatched: bitwise identical RX."""

    def run(clock: object) -> np.ndarray:
        eng = _skip_engine("numba_cpu", always_active=True, clock=clock)
        cap = eng.capture_accumulator("rx1.rx0")
        eng.run(Fraction(30, 1000))
        return captured(cap, 0, 30_000)

    a = run(None)
    weird = iter(range(10**12, 10**13, 977_777))
    monkeypatch.setattr(time, "perf_counter_ns", lambda: next(weird))
    monkeypatch.setattr(time, "monotonic", lambda: 42.0)
    clk = _GarbageClock()
    b = run(clk)
    assert clk.calls > 10
    assert np.array_equal(a.view(np.float64), b.view(np.float64))
