"""IO-06: API status codes TIMEOUT, OVERFLOW, LATE_COMMAND, TIME_ERROR, SEQ_ERROR (DESIGN 8.1).

TIMEOUT, LATE_COMMAND and TIME_ERROR come from the engine's normative timing rules
(``FakeEngine``: R3 mapping, frontier, lockstep commitment). OVERFLOW (paced RX ring full,
section 4.10) and SEQ_ERROR (frame-sequence gap, section 8.3) are produced by the paced
engine and the transport; here a scripted engine emits them exactly as specified and the
test checks that the API surfaces them unchanged, with UHD semantics: an error met after
samples were copied is queued for the next ``recv``, and ``OVERFLOW`` carries the post-gap
``time_spec``.
"""

from __future__ import annotations

import importlib.util
import time
from collections import deque
from fractions import Fraction

import numpy as np
import pytest

from sremu.api import (
    AsyncCode,
    AsyncMetadata,
    Device,
    ErrorCode,
    RxMetadata,
    StreamArgs,
    StreamCmd,
    TimeSpec,
    TxMetadata,
)
from sremu.contracts import CommandReceipt, StreamHandle, StreamSpec
from sremu.errors import ConfigurationError
from tests.io import _wp7_support as sup

pytestmark = pytest.mark.io

FC = 3.5e9
F = Fraction(1_000_000)


def _dev(**kw):
    return Device.from_engine(sup.make_engine(), **kw)


def _tone(n, k0=0):
    k = np.arange(k0, k0 + n)
    return (np.exp(2j * np.pi * k / 64) * (1 + k % 3)).astype(np.complex64)


def _burst(at_k):
    return TxMetadata(
        has_time_spec=True,
        time_spec=TimeSpec.from_ticks(at_k, F),
        start_of_burst=True,
        end_of_burst=True,
    )


def _read(rx, n, chunk=97):
    out = np.zeros(n, np.complex64)
    got = 0
    md = RxMetadata()
    while got < n:
        k = rx.recv(out[got : got + min(chunk, n - got)], md, timeout=0.0)
        assert k > 0, md.error_code
        got += k
    return out


# --------------------------------------------------------------------------- TIMEOUT


def test_io06_timeout_without_stream_command():
    dev = _dev()
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    md = RxMetadata(error_code=ErrorCode.OVERFLOW)  # stale value must be reset
    assert rx.recv(np.zeros(10, np.complex64), md, timeout=0.0) == 0
    assert md.error_code == ErrorCode.TIMEOUT
    assert md.has_time_spec is False


def test_io06_timeout_while_linked_tx_is_uncommitted():
    dev = _dev()
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=100))
    md = RxMetadata()
    t0 = time.monotonic()
    assert rx.recv(np.zeros(100, np.complex64), md, timeout=0.2) == 0
    waited = time.monotonic() - t0
    assert md.error_code == ErrorCode.TIMEOUT
    assert 0.15 <= waited < 5.0  # waited about the timeout, then gave up
    # once the TX commits, the very same request delivers, starting at sample 0
    tx.send(_tone(100), _burst(0))
    y = np.zeros(100, np.complex64)
    assert rx.recv(y, md, timeout=0.0) == 100
    assert md.error_code == ErrorCode.NONE
    assert md.first_sample == 0
    assert md.end_of_burst is True
    np.testing.assert_array_equal(y.view(np.uint32), _tone(100).view(np.uint32))


# --------------------------------------------------------------------------- LATE_COMMAND


def test_io06_late_stream_command():
    dev = _dev()
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    tx.send(_tone(1000), _burst(0))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=300))
    _read(rx, 300)
    rx.issue_stream_cmd(
        StreamCmd("num_done", num_samps=10, stream_now=False, time_spec=TimeSpec.from_ticks(50, F))
    )
    md = RxMetadata()
    assert rx.recv(np.zeros(10, np.complex64), md, timeout=0.0) == 0
    assert md.error_code == ErrorCode.LATE_COMMAND
    assert rx.recv(np.zeros(10, np.complex64), md, timeout=0.0) == 0
    assert md.error_code == ErrorCode.TIMEOUT  # nothing was started


def test_io06_late_ris_command_reject_and_apply_on_arrival():
    dev = _dev()
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    x = _tone(2000)
    tx.send(x, _burst(0))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=2000))
    y0 = _read(rx, 700)
    ones = np.ones(sup.N_EL, np.int16)
    # reject: LATE_COMMAND, nothing applied
    r = dev.ris("ris0").set_config(states=ones, time_spec=TimeSpec(0, 1e-4), late_policy="reject")
    assert isinstance(r, CommandReceipt)
    assert (r.ok, r.status, r.error) == (False, "rejected", ErrorCode.LATE_COMMAND)
    # apply_on_arrival: moved to the frontier (sample 700), lateness logged exactly
    r2 = dev.ris("ris0").set_config(states=ones, time_spec=TimeSpec(0, 2e-4))
    assert (r2.ok, r2.status, r2.k_req) == (True, "late_applied", {"mu.rx0": 200})
    y1 = _read(rx, 1300)
    g_on = sup.gain({"ris0": _cfg(ones)})  # complex128, as the engine applies it
    np.testing.assert_array_equal(y0, x[:700])  # initial state 0: gain exactly 1
    want = (x[700:].astype(np.complex128) * g_on).astype(np.complex64)
    np.testing.assert_array_equal(y1, want)
    recs = {r.cmd_id: r for r in dev.engine.records}
    rej, late = recs[r.cmd_id], recs[r2.cmd_id]
    assert (rej.status, rej.t_applied) == ("rejected", None)
    assert late.status == "late_applied"
    assert late.k_applied == {"mu.rx0": 700}
    assert late.lateness_samples == {"mu.rx0": 500}
    assert late.t_applied == Fraction(700, 1_000_000)


def _cfg(states):
    from sremu.contracts import PanelConfig

    return PanelConfig("ris0", np.asarray(states, np.int16))


# --------------------------------------------------------------------------- TIME_ERROR


def test_io06_time_error_for_late_tx_burst():
    dev = _dev()
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    tx.send(_tone(500), TxMetadata(has_time_spec=True, time_spec=TimeSpec(0, 0.0)))
    ack = tx.recv_async_msg(timeout=0.0)
    assert ack is None  # no EOB yet
    tx.send(np.zeros(0, np.complex64), TxMetadata(end_of_burst=True))
    ack = tx.recv_async_msg(timeout=0.0)
    assert ack is not None
    assert (ack.event_code, ack.sample_index) == (AsyncCode.BURST_ACK, 500)
    rx.issue_stream_cmd(StreamCmd("start_cont"))
    _read(rx, 400)
    late = _tone(37)
    n = tx.send(late, _burst(123))  # sample 123 is already computed
    assert n == 0
    msg = tx.recv_async_msg(timeout=0.0)
    assert msg == AsyncMetadata(
        AsyncCode.TIME_ERROR, "ap.tx0", 0, True, TimeSpec.from_ticks(123, F), 123, 37
    )
    assert dev.engine.counters["late_samples"] == 37
    assert tx.recv_async_msg(timeout=0.0) is None


# --------------------------------------------------------------------------- scripted engine


class _Scripted:
    """Minimal EngineProtocol double that replays a script of RX chunks and TX events."""

    def __init__(self, rx_script, async_script=()):
        self.rx_script = deque(rx_script)
        self.async_script = deque(async_script)
        self.reads: list[int] = []

    def open_stream(self, spec: StreamSpec) -> StreamHandle:
        return StreamHandle(1 if spec.direction == "tx" else 2, spec, 512)

    def submit_tx(self, h, samples, md):
        return int(samples.size)

    def stream_cmd(self, h, cmd):
        pass

    def read_rx(self, h, n, timeout_s):
        self.reads.append(n)
        if not self.rx_script:
            return np.zeros(0, np.complex64), RxMetadata(error_code=ErrorCode.TIMEOUT)
        k0, n_samps, code = self.rx_script[0]
        if code != ErrorCode.NONE and n_samps == 0:
            self.rx_script.popleft()
            md = RxMetadata(
                has_time_spec=True,
                time_spec=TimeSpec.from_ticks(k0, F),
                error_code=code,
                first_sample=k0,
            )
            return np.zeros(0, np.complex64), md
        take = min(n, n_samps)
        if take == n_samps:
            self.rx_script.popleft()
        else:
            self.rx_script[0] = (k0 + take, n_samps - take, code)
        md = RxMetadata(
            has_time_spec=True,
            time_spec=TimeSpec.from_ticks(k0, F),
            error_code=code,
            first_sample=k0,
        )
        return _tone(take, k0), md

    def recv_async(self, h, timeout_s):
        return self.async_script.popleft() if self.async_script else None

    def schedule(self, req):  # pragma: no cover - not used
        raise NotImplementedError

    def register_controller(self, ctrl):  # pragma: no cover - not used
        raise NotImplementedError

    def time_now(self):
        return Fraction(0)

    def caps(self):
        return {}


def test_io06_overflow_is_queued_after_partial_and_carries_post_gap_time():
    # paced engine: samples [0,100), then the RX ring overflowed and samples [100,300) were
    # dropped; the overflow report carries the post-gap time (sample 300), then [300,450).
    eng = _Scripted(
        [(0, 100, ErrorCode.NONE), (300, 0, ErrorCode.OVERFLOW), (300, 150, ErrorCode.NONE)]
    )
    dev = Device(eng)
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    buf = np.zeros(256, np.complex64)
    md = RxMetadata()
    assert rx.recv(buf, md, timeout=0.0) == 100  # partial, error queued (UHD)
    assert md.error_code == ErrorCode.NONE
    assert md.time_spec == TimeSpec(0, 0.0)
    np.testing.assert_array_equal(buf[:100], _tone(100))
    assert rx.recv(buf, md, timeout=0.0) == 0
    assert md.error_code == ErrorCode.OVERFLOW
    assert md.time_spec == TimeSpec.from_ticks(300, F)
    assert md.time_spec.exact() == Fraction(3, 10_000)
    assert rx.recv(buf, md, timeout=0.0) == 150
    assert (md.error_code, md.first_sample) == (ErrorCode.NONE, 300)
    np.testing.assert_array_equal(buf[:150], _tone(150, 300))


def test_io06_overflow_first_is_returned_immediately():
    eng = _Scripted([(1000, 0, ErrorCode.OVERFLOW), (1000, 10, ErrorCode.NONE)])
    dev = Device(eng)
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    md = RxMetadata()
    assert rx.recv(np.zeros(64, np.complex64), md, timeout=0.0) == 0
    assert (md.error_code, md.time_spec) == (ErrorCode.OVERFLOW, TimeSpec(0, 1e-3))


def test_io06_recv_never_spans_a_gap():
    eng = _Scripted([(0, 50, ErrorCode.NONE), (80, 50, ErrorCode.NONE)])
    dev = Device(eng)
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    buf = np.zeros(200, np.complex64)
    md = RxMetadata()
    assert rx.recv(buf, md, timeout=0.0) == 50
    assert md.first_sample == 0
    assert rx.recv(buf, md, timeout=0.0) == 50
    assert md.first_sample == 80
    assert md.time_spec == TimeSpec.from_ticks(80, F)


def test_io06_seq_error_async_message_surfaces_exactly():
    msg = AsyncMetadata(AsyncCode.SEQ_ERROR, "ap.tx0", 0, True, TimeSpec.from_ticks(4096, F), 4096)
    under = AsyncMetadata(AsyncCode.UNDERFLOW, "ap.tx0", 0, True, TimeSpec(0, 0.01), 10_000, 64)
    eng = _Scripted([], [msg, under])
    dev = Device(eng)
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    assert tx.recv_async_msg(timeout=0.0) == msg
    got = tx.recv_async_msg(timeout=0.0)
    assert got == under
    assert got.event_code == AsyncCode.UNDERFLOW
    assert tx.recv_async_msg(timeout=0.0) is None


# --------------------------------------------------------------------------- other codes


def test_io06_unknown_target_and_not_permitted_receipts():
    dev = _dev()
    dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    r = dev.ris("nope").set_config(states=[1, 0], time_spec=TimeSpec(0, 1e-3))
    assert (r.ok, r.error) == (False, ErrorCode.UNKNOWN_TARGET)
    r = dev.node("mu").set_position([1.0, 2.0, 0.0], time_spec=TimeSpec(0, 1e-3))
    assert (r.ok, r.error) == (False, ErrorCode.NOT_PERMITTED)  # FakeEngine models no geometry


def test_io06_nchan_above_one_is_rejected():
    with pytest.raises(ConfigurationError, match="nchan"):
        StreamArgs("ap.tx0", FC, "1e6", nchan=2)
    dev = _dev()
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    with pytest.raises(ConfigurationError, match="nchan must be 1"):
        tx.send(np.zeros((2, 8), np.complex64))


def test_io06_held_chunk_is_split_across_small_buffers():
    """Samples held back at a gap are delivered in order, split to the caller's buffers,
    each call carrying the exact time_spec of its first sample."""
    eng = _Scripted([(0, 60, ErrorCode.NONE), (100, 40, ErrorCode.NONE)])
    dev = Device(eng)
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    md = RxMetadata()
    big = np.zeros(100, np.complex64)
    assert rx.recv(big, md, timeout=0.0) == 60  # stops at the gap; [100, 140) is held
    got = []
    for _ in range(4):
        small = np.zeros(10, np.complex64)
        assert rx.recv(small, md, timeout=0.0) == 10
        assert md.time_spec == TimeSpec.from_ticks(100 + 10 * len(got), F)
        assert md.first_sample == 100 + 10 * len(got)
        got.append(small)
    np.testing.assert_array_equal(np.concatenate(got), _tone(40, 100))
    assert rx.recv(big, md, timeout=0.0) == 0
    assert md.error_code == ErrorCode.TIMEOUT


@pytest.mark.skipif(
    importlib.util.find_spec("sremu.engine.emulator") is None,
    reason="WP6 engine (sremu.engine.emulator) not available yet",
)
def test_io06_codes_through_the_wp6_engine(repo_root):
    """TIMEOUT, TIME_ERROR and LATE_COMMAND reach the API unchanged from the real engine."""
    f = Fraction(5880)
    fc = 2399996950.0
    dev = Device.inproc(repo_root / "examples" / "bpsk_txrx" / "bpsk.yaml", backend="numpy_ref")
    tx = dev.get_tx_stream(StreamArgs("node_1.tx0", fc, "5880"))
    rx = dev.get_rx_stream(StreamArgs("node_2.rx0", fc, "5880"))
    md = RxMetadata()
    assert rx.recv(np.zeros(10, np.complex64), md, timeout=0.0) == 0
    assert md.error_code == ErrorCode.TIMEOUT  # no stream command
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=500))
    assert rx.recv(np.zeros(10, np.complex64), md, timeout=0.0) == 0
    assert md.error_code == ErrorCode.TIMEOUT  # linked TX not committed (lockstep)
    assert tx.send(np.ones(1000, np.complex64), TxMetadata(True, TimeSpec(0, 0.0), True, True))
    buf = np.zeros(500, np.complex64)
    assert rx.recv(buf, md, timeout=0.0) == 500
    assert (md.error_code, md.first_sample, md.end_of_burst) == (ErrorCode.NONE, 0, True)
    assert (
        tx.send(np.ones(7, np.complex64), TxMetadata(True, TimeSpec.from_ticks(10, f), True, True))
        == 0
    )
    ack = tx.recv_async_msg(timeout=0.0)
    assert (ack.event_code, ack.sample_index) == (AsyncCode.BURST_ACK, 1000)
    err = tx.recv_async_msg(timeout=0.0)
    assert err == AsyncMetadata(
        AsyncCode.TIME_ERROR, "node_1.tx0", 0, True, TimeSpec.from_ticks(10, f), 10, 7
    )
    assert err.time_spec.exact() == Fraction(1, 588)
    assert dev.engine.counters["late_samples"] == 7
    rx.issue_stream_cmd(
        StreamCmd("num_done", num_samps=5, stream_now=False, time_spec=TimeSpec.from_ticks(3, f))
    )
    assert rx.recv(np.zeros(5, np.complex64), md, timeout=0.0) == 0
    assert md.error_code == ErrorCode.LATE_COMMAND
    dev.close()


@pytest.mark.skipif(
    importlib.util.find_spec("sremu.engine.emulator") is None,
    reason="WP6 engine (sremu.engine.emulator) not available yet",
)
def test_io06_overflow_through_the_wp6_paced_engine(repo_root):
    """OVERFLOW from the real engine (section 4.10): paced mode with a 2-page RX queue and a
    client that does not read. A deterministic FakeClock drives the pacer. quickstart.yaml
    (1 MS/s, page 512, in-process tone on ap.tx0) is paced for 10 pages: pages 0-1 stay
    queued, pages 2-9 are dropped (``overflow_pages`` = 8). Through the API the first recv
    returns the 1024 queued samples (the error is queued, UHD), the next returns 0 samples
    with OVERFLOW and the post-gap time_spec ``from_ticks(10*512)`` = 16/3125 s."""
    from sremu.engine.pacer import FakeClock

    dev = Device.inproc(
        repo_root / "scenarios" / "quickstart.yaml",
        mode="paced",
        backend="numpy_ref",
        clock=FakeClock(),
        rx_queue_pages=2,
    )
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
    rx.issue_stream_cmd(StreamCmd("start_cont"))
    summary = dev.engine.run_paced(Fraction(10 * 512) / F)
    assert summary.counters["overflow_pages"] == 8
    buf = np.zeros(4096, np.complex64)
    md = RxMetadata()
    assert rx.recv(buf, md, timeout=0.0) == 1024
    assert (md.error_code, md.first_sample, md.time_spec) == (ErrorCode.NONE, 0, TimeSpec(0, 0.0))
    assert rx.recv(buf, md, timeout=0.0) == 0
    assert md.error_code == ErrorCode.OVERFLOW
    assert md.time_spec == TimeSpec.from_ticks(5120, F)
    assert md.time_spec.exact() == Fraction(16, 3125)
    dev.close()
