"""Python API behaviour (DESIGN.md 8.1): streams, metadata, commands, recording, threads.

Oracles: the exact time base (R3, R5), the UHD recv contract (one contiguous run per call,
``time_spec`` of the first sample), the sc16 scaling of :mod:`sremu.api.streams`, the
command-time rules of :mod:`sremu.api.device` and the FakeEngine identity channel.
"""

from __future__ import annotations

import importlib.util
import threading
from fractions import Fraction

import numpy as np
import pytest

from sremu.api import (
    Device,
    Linear,
    RxMetadata,
    StreamArgs,
    StreamCmd,
    TimeSpec,
    TxMetadata,
)
from sremu.api.metadata import FirstOrder, Step, table_iii, table_iii_markdown
from sremu.api.streams import SC16_SCALE, complex64_to_sc16
from sremu.contracts import TransitionSpec
from sremu.errors import ConfigurationError, ErrorCode, NotPermitted, StreamError
from sremu.record.eventlog import read_events
from sremu.timebase import from_ticks
from tests.io import _wp7_support as sup

pytestmark = pytest.mark.io

FC = 3.5e9
F = Fraction(1_000_000)


def _x(n, k0=0):
    k = np.arange(k0, k0 + n, dtype=np.float64)
    return (np.cos(k / 7.0) + 1j * np.sin(k / 11.0)).astype(np.complex64)


def _burst(k):
    return TxMetadata(
        has_time_spec=True,
        time_spec=TimeSpec.from_ticks(k, F),
        start_of_burst=True,
        end_of_burst=True,
    )


def test_api_facade_exports_device_and_streamargs():
    import sremu
    import sremu.api.device
    import sremu.api.streams

    assert sremu.Device is sremu.api.device.Device
    assert sremu.StreamArgs is sremu.api.streams.StreamArgs


def test_api_streamargs_rates_are_exact():
    assert StreamArgs("a.tx0", FC, 5880.0).rate_sps == Fraction(5880)
    assert StreamArgs("a.tx0", FC, "23.04e6").rate_sps == Fraction(23_040_000)
    assert StreamArgs("a.tx0", FC, 30.72e6).rate_sps == Fraction(30_720_000)
    assert StreamArgs("a.tx0", FC, "1/3").rate_sps == Fraction(1, 3)
    with pytest.raises(ConfigurationError):
        StreamArgs("not-a-stream-id", FC, "1e6")
    with pytest.raises(ConfigurationError):
        StreamArgs("a.tx0", FC, "1e6", cpu="sc8")


@pytest.mark.parametrize("chunks", [[1], [7], [1000], [512], [3, 250, 1, 999, 64]])
def test_api_recv_chunking_is_invisible(chunks):
    """Output and per-call time_spec are independent of the client buffer sizes (I2a)."""
    dev = Device.from_engine(sup.make_engine())
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    x = _x(3000)
    tx.send(x, _burst(0))
    dev.ris("ris0").set_config(states=np.arange(8) % 2, time_spec=TimeSpec(0, 1.234e-3))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=3000))
    out = np.zeros(3000, np.complex64)
    got, i = 0, 0
    md = RxMetadata()
    while got < 3000:
        n = rx.recv(out[got : got + chunks[i % len(chunks)]], md, timeout=0.0)
        assert n > 0
        assert md.first_sample == got
        assert md.time_spec == TimeSpec.from_ticks(got, F)
        assert md.has_time_spec
        assert md.start_of_burst == (got == 0)
        got += n
        i += 1
    assert md.end_of_burst
    ref = Device.from_engine(sup.make_engine())
    ref.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6")).send(x, _burst(0))
    ref.ris("ris0").set_config(states=np.arange(8) % 2, time_spec=TimeSpec(0, 1.234e-3))
    rrx = ref.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    rrx.issue_stream_cmd(StreamCmd("num_done", num_samps=3000))
    whole = np.zeros(3000, np.complex64)
    assert rrx.recv(whole, RxMetadata(), timeout=0.0) == 3000
    np.testing.assert_array_equal(out.view(np.uint32), whole.view(np.uint32))
    # the switch at RIS time 1.234 ms lands at to_ticks = 1234 (tau = 0 in the fake)
    g = sup.gain({"ris0": _cfg(np.arange(8) % 2)})
    np.testing.assert_array_equal(whole[:1234], x[:1234])
    want = (x[1234:].astype(np.complex128) * g).astype(np.complex64)
    np.testing.assert_array_equal(whole[1234:], want)


def _cfg(states):
    from sremu.contracts import PanelConfig

    return PanelConfig("ris0", np.asarray(states, np.int16))


def test_api_sc16_tx_and_rx_conversion_saturates():
    dev = Device.from_engine(sup.make_engine())
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6", cpu="sc16"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6", cpu="sc16"))
    iq = np.array([[32767, -32768], [16384, 0], [-1, 1], [0, 32767]], np.int16)
    tx.send(iq, _burst(0))
    # the engine saw (I + jQ)/32767 in complex64
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=4))
    out = np.zeros((4, 2), np.int16)
    assert rx.recv(out, RxMetadata(), timeout=0.0) == 4
    back = (iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)) / np.float32(SC16_SCALE)
    np.testing.assert_array_equal(out, complex64_to_sc16(back.astype(np.complex64)))
    np.testing.assert_array_equal(out, iq)  # sc16 -> fc32 -> sc16 is the identity
    # values beyond full scale saturate (section 3.3)
    big = np.array([2.0 - 3.0j, -1.5 + 0.0j, 0.5 + 0.5j], np.complex64)
    np.testing.assert_array_equal(
        complex64_to_sc16(big), [[32767, -32768], [-32768, 0], [16384, 16384]]
    )
    with pytest.raises(TypeError):
        tx.send(np.zeros(4, np.complex64))  # sc16 stream needs int16 (n, 2)


def test_api_command_time_and_time_forms():
    dev = Device.from_engine(sup.make_engine())
    dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    s = np.zeros(8, np.int16)
    dev.set_command_time(TimeSpec(0, 8e-3))
    assert dev.get_command_time() == TimeSpec.from_exact(Fraction(1, 125))
    r = dev.ris("ris0").set_config(states=s)
    assert r.t_req == Fraction(1, 125)
    assert r.k_req == {"mu.rx0": 8000}
    r = dev.ris("ris0").set_config(states=s, time_spec="1/3000")  # explicit wins
    assert (r.t_req, r.k_req) == (Fraction(1, 3000), {"mu.rx0": 333})
    dev.clear_command_time()
    r = dev.ris("ris0").set_config(states=s)  # asap: the frontier time (0)
    assert r.t_req == 0
    r = dev.ris("ris0").set_config(states=s, at_sample=("mu.rx0", 4321))
    assert r.t_req == from_ticks(4321, F)
    r = dev.ris("ris0").set_config(states=s, time_spec=0.5e-3)  # float via repr (R5)
    assert r.k_req == {"mu.rx0": 500}
    with pytest.raises(ConfigurationError, match="time_spec or at_sample"):
        dev.ris("ris0").set_config(states=s, time_spec=1, at_sample=("mu.rx0", 1))
    with pytest.raises(ConfigurationError, match="integer state"):
        dev.ris("ris0").set_config(states=[0.5] * 8, time_spec=1)


def test_api_transition_helpers_and_recorded_request(tmp_path):
    assert Step() == TransitionSpec("step")
    assert Linear("2e-5") == TransitionSpec("linear", Fraction(1, 50_000))
    assert FirstOrder("1e-6") == TransitionSpec("first_order", Fraction(1, 1_000_000))
    assert Linear() == TransitionSpec("linear", None)
    dev = Device.from_engine(sup.make_engine(), record=tmp_path)
    dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    r = dev.ris("ris0").set_config(
        states=np.ones(8, np.int16), time_spec=TimeSpec(0, 5e-3), transition=Linear("2e-5")
    )
    dev.close()
    (ev,) = read_events(tmp_path)
    assert ev.record.seq == r.seq
    assert ev.record.transition == {"model": "linear", "duration_s": "1/50000", "classes": "single"}
    assert ev.request.transition == TransitionSpec("linear", Fraction(1, 50_000))
    assert ev.request.at.t == Fraction(1, 200)


def test_api_marker_and_set_enabled_records(tmp_path):
    dev = Device.from_engine(sup.make_engine(), record=tmp_path, src="lab")
    dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    m = dev.marker("start", time_spec=0, note="hello")
    e = dev.ris("ris0").set_enabled(False, time_spec=TimeSpec(0, 1e-3))
    dev.close()
    evs = {x.record.cmd_id: x for x in read_events(tmp_path)}
    assert m.cmd_id == "lab-1"
    assert evs["lab-1"].record.op == "rec.marker"
    assert evs["lab-1"].request.payload == {"label": "start", "note": "hello"}
    assert evs[e.cmd_id].record.counterfactual is True
    assert evs[e.cmd_id].request.enabled is False


def test_api_blocking_recv_waits_for_a_tx_thread():
    dev = Device.from_engine(sup.make_engine())
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=2000))
    x = _x(2000)
    start = threading.Event()

    def producer():
        start.wait(5)
        for k in range(0, 2000, 500):
            md = TxMetadata(has_time_spec=k == 0, time_spec=TimeSpec(0, 0.0))
            md.start_of_burst = k == 0
            md.end_of_burst = k == 1500
            tx.send(x[k : k + 500], md)

    th = threading.Thread(target=producer)
    th.start()
    out = np.zeros(2000, np.complex64)
    got = 0
    md = RxMetadata()
    start.set()
    while got < 2000:
        n = rx.recv(out[got:], md, timeout=5.0)
        assert n > 0, md.error_code
        got += n
    th.join(5)
    np.testing.assert_array_equal(out.view(np.uint32), x.view(np.uint32))


def test_api_recv_timeout_is_not_shortened_by_other_threads():
    """``timeout`` is the wall time recv may block (UHD): wake-ups caused by another
    thread's calls (here a command every millisecond) must not end the wait early. The
    linked TX never sends, so the only correct outcome is TIMEOUT after ``timeout`` (the
    defect this guards against returned after 0.02-0.13 s)."""
    import time

    dev = Device.from_engine(sup.make_engine())
    dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))  # linked, never commits
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=10))
    stop = threading.Event()
    calls = []

    def chatter():
        while not stop.is_set():
            calls.append(dev.marker("tick", time_spec=1))
            time.sleep(0.001)

    th = threading.Thread(target=chatter)
    th.start()
    md = RxMetadata()
    t0 = time.monotonic()
    try:
        n = rx.recv(np.zeros(10, np.complex64), md, timeout=0.5)
    finally:
        waited = time.monotonic() - t0
        stop.set()
        th.join(5)
    assert (n, md.error_code) == (0, ErrorCode.TIMEOUT)
    assert len(calls) > 20  # the waiting thread was woken many times
    assert 0.48 <= waited < 5.0  # 20 ms margin for the Windows timer tick (15.6 ms)
    # recv_async_msg uses the same deadline
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    t0 = time.monotonic()
    assert tx.recv_async_msg(timeout=0.2) is None
    assert time.monotonic() - t0 >= 0.18


def test_api_transition_durations_accept_r5_floats():
    """API floats are exact through ``Fraction(repr(x))`` (R5), also for transitions."""
    assert Linear(2e-5) == Linear("2e-5") == TransitionSpec("linear", Fraction(1, 50_000))
    assert FirstOrder(1e-6) == TransitionSpec("first_order", Fraction(1, 1_000_000))
    with pytest.raises(ValueError, match="non-finite"):
        Linear(float("inf"))


def test_api_get_stream_twice_returns_the_same_streamer():
    dev = Device.from_engine(sup.make_engine())
    a = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    assert dev.get_tx_stream(StreamArgs("ap.tx0", FC, 1e6)) is a
    with pytest.raises(ConfigurationError, match="already open"):
        dev.get_tx_stream(StreamArgs("ap.tx0", FC, "2e6"))


def test_api_closed_device_refuses_calls(tmp_path):
    dev = Device.from_engine(sup.make_engine(), record=tmp_path)
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    dev.close()
    dev.close()  # idempotent
    with pytest.raises(StreamError, match="closed"):
        tx.send(_x(4), _burst(0))
    with pytest.raises(StreamError, match="closed"):
        dev.ris("ris0").set_config(states=np.zeros(8), time_spec=0)


def test_api_device_recording_needs_tx_head_for_untimed_burst_start(tmp_path):
    eng = sup.make_engine(with_head=False)
    dev = Device.from_engine(eng, record=tmp_path)
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    with pytest.raises(NotPermitted, match="tx_head"):
        tx.send(_x(10), TxMetadata(start_of_burst=True))
    assert eng.counters["late_samples"] == 0
    # timed bursts and in-burst continuations are placed by the API itself (R3)
    assert tx.send(_x(10), TxMetadata(has_time_spec=True, time_spec=TimeSpec(0, 0.0))) == 10
    assert tx.send(_x(5, 10), TxMetadata(end_of_burst=True)) == 5
    dev.close()
    from sremu.record.sigmf import read_recording

    r = read_recording(tmp_path / "iq" / "ap.tx0.sigmf-meta")
    assert [(c.global_index, c.length) for c in r.captures] == [(0, 15)]
    np.testing.assert_array_equal(r.data, _x(15))


@pytest.mark.parametrize("with_head", [False, True])
def test_api_idle_until_does_not_open_a_burst_for_recording(tmp_path, with_head):
    """``idle_until`` is a commitment promise (section 4.8), not a burst: an untimed burst
    start after it is placed by the engine. Without ``tx_head`` the Device cannot know
    where, so it refuses to record it (NotPermitted) instead of guessing the idle index;
    with ``tx_head`` it records at the engine's index."""
    run = tmp_path / f"h{with_head}"
    dev = Device.from_engine(sup.make_engine(with_head=with_head), record=run)
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    assert tx.send(_x(100), _burst(0)) == 100
    tx.idle_until(TimeSpec.from_ticks(500, F))
    md = TxMetadata(start_of_burst=True, end_of_burst=True)
    if with_head:
        assert tx.send(_x(10), md) == 10
        k0 = dev.engine.tx_head(tx.handle) - 10
    else:
        with pytest.raises(NotPermitted, match="tx_head"):
            tx.send(_x(10), md)
    dev.close()
    from sremu.record.sigmf import read_recording

    r = read_recording(run / "iq" / "ap.tx0.sigmf-meta")
    caps = [(c.global_index, c.length) for c in r.captures]
    if with_head:
        assert caps == ([(0, 110)] if k0 == 100 else [(0, 100), (k0, 10)])
    else:
        assert caps == [(0, 100)]


def test_api_untimed_burst_recorded_at_engine_index(tmp_path):
    dev = Device.from_engine(sup.make_engine(), record=tmp_path)
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    tx.send(_x(100), _burst(0))
    rx.issue_stream_cmd(StreamCmd("start_cont"))
    out = np.zeros(300, np.complex64)
    assert rx.recv(out, RxMetadata(), timeout=0.0) == 300  # silence after the EOB
    tx.send(_x(40), TxMetadata(start_of_burst=True, end_of_burst=True))  # untimed: frontier
    dev.close()
    from sremu.record.sigmf import read_recording

    r = read_recording(tmp_path / "iq" / "ap.tx0.sigmf-meta")
    assert [(c.global_index, c.length) for c in r.captures] == [(0, 100), (300, 40)]


def test_api_table_iii_rows():
    rows = table_iii()
    assert len(rows) == 11
    uhd = " ".join(r.uhd for r in rows)
    for name in (
        "tx_streamer::send",
        "rx_streamer::recv",
        "tx_metadata_t",
        "rx_metadata_t",
        "async_metadata_t",
        "stream_cmd_t",
        "time_spec_t",
        "set_command_time",
        "clear_command_time",
    ):
        assert name in uhd
    assert rows[0].sremu.startswith("``TxStreamer.send")
    assert "[UNVERIFIED]" in rows[0].srsran_zmq
    md = table_iii_markdown().splitlines()
    assert md[0] == "| sremu | UHD | srsRAN ZMQ device args | GNU Radio ZMQ blocks |"
    assert len(md) == 2 + 11


@pytest.mark.skipif(
    importlib.util.find_spec("sremu.engine.emulator") is not None, reason="WP6 engine present"
)
def test_api_inproc_reports_missing_engine(repo_root):
    with pytest.raises(NotImplementedError, match="WP6"):
        Device.inproc(repo_root / "scenarios" / "quickstart.yaml")


@pytest.mark.skipif(
    importlib.util.find_spec("sremu.transport.client") is not None, reason="WP8 present"
)
def test_api_connect_reports_missing_transport():
    with pytest.raises(NotImplementedError, match="WP8"):
        Device.connect("tcp://127.0.0.1:5600")
