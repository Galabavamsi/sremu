"""Stream and server lifecycle over the control socket (DESIGN.md sections 4.8, 4.9, 8.3, 8.5;
I6). Added by the WP8 review.

* ``stream.close`` of a TX stream cannot overtake sample frames still in flight on the
  (separate) data-plane connection: ``RemoteEngine.close_stream`` sends ``END_OF_STREAM``
  in-band and waits until TX ingress took every frame. Oracle: the receiver's samples are
  bitwise equal to an in-process run that submits the same samples and then
  ``end_of_stream``; nothing is counted as a bad packet. Frames sent to the closed stream
  id afterwards are dropped and counted (``BAD_PACKET``, reason ``stream_id``).
* ``stream.close`` of an RX stream stops it and closes its egress.
* An orderly ``sim.shutdown`` while the engine waits at a remote barrier is not an engine
  failure (``sremu serve`` exits 0); a real ``ack_timeout_s`` expiry still is.
* A socket client cannot use a barrier controller's name as ``src`` (the engine exempts
  barrier controllers' late commands from the ``nondeterministic_inputs`` flag).
* ``sim.pause`` is refused once a paced run is going (it has no pause point).
* PUB telemetry with non-finite floats stays valid against ``telemetry.schema.json``.
* An oversize TX part (above the largest valid frame, below the 128 MiB ZMQ backstop) is
  counted as a bad packet.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import zmq

from sremu.contracts import StreamCmd, StreamSpec, TxMetadata
from sremu.engine.streams import ToneSource
from sremu.errors import ErrorCode
from sremu.timebase import TimeSpec
from sremu.transport.client import ClientSession, RemoteEngine
from sremu.transport.framing import Flag, FrameHeader, make_frame
from tests.io._wp8_support import bits, serve, served, wait_until
from tests.runtime._engine_support import lab_like_fake, make_engine, waveform

pytestmark = pytest.mark.io

F = Fraction(10**6)
N_TX = 2000
N_RX = 3000
CHUNK = 5  # many small frames: ingestion takes long enough for a close to overtake them


def _engine(**kw: object):
    return make_engine(lab_like_fake(), **kw)


def _inprocess_reference(x: np.ndarray) -> np.ndarray:
    eng = _engine()
    htx = eng.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
    hrx = eng.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
    eng.stream_cmd(hrx, StreamCmd("num_done", N_RX, False, TimeSpec(0, 0)))
    assert eng.submit_tx(htx, x, TxMetadata(True, TimeSpec(0, 0), True, False)) == x.size
    eng.submit_tx(htx, np.zeros(0, np.complex64), TxMetadata(end_of_stream=True))
    parts = []
    while sum(p.size for p in parts) < N_RX:
        y, md = eng.read_rx(hrx, N_RX, 0.0)
        assert y.size, md.error_code
        parts.append(y)
    return np.concatenate(parts)


def test_i6_stream_close_tx_is_ordered_after_in_flight_frames() -> None:
    x = waveform(N_TX, seed=31)
    eng = _engine(unopened_tx="wait")
    with served(eng, start_paused=True) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]), name="closer")
        try:
            hrx = cli.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
            cli.stream_cmd(hrx, StreamCmd("num_done", N_RX, False, TimeSpec(0, 0)))
            htx = cli.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
            for a in range(0, N_TX, CHUNK):
                md = TxMetadata(a == 0, TimeSpec(0, 0), a == 0, False)  # one open burst
                assert cli.submit_tx(htx, x[a : a + CHUNK], md) == CHUNK
            cli.close_stream(htx)  # at once: frames are still in flight
            assert eng.tx_head(htx) == N_TX  # every sample reached the engine first
            assert srv.counters["bad_packets"] == 0
            assert srv.counters["frames_in"] == N_TX // CHUNK + 1  # + the in-band EOS
            cli.resume()
            out = []
            got = 0
            while got < N_RX:
                y, md = cli.read_rx(hrx, N_RX, 10.0)
                assert md.error_code in (ErrorCode.NONE, ErrorCode.TIMEOUT), md.error_code
                if y.size:
                    assert md.first_sample == got
                    out.append(y)
                    got += y.size
            y = np.concatenate(out)
            # frames to the closed stream id are dropped and counted (I6)
            push = cli.ctx.socket(zmq.PUSH)
            push.setsockopt(zmq.LINGER, 0)
            push.connect(str(srv.endpoints["tx"]))
            hdr = FrameHeader(int(Flag.HAS_TIME | Flag.SOB), htx.id, 99, N_RX, 4)
            push.send_multipart(make_frame(hdr, np.ones(4, np.complex64)))
            wait_until(lambda: srv.counters["bad_packets"] == 1, 10.0, "the stray frame")
            assert srv.counters.reasons_snapshot() == {"stream_id": 1}
            push.close(0)
            # RX close: the stream stops and its egress closes
            cli.close_stream(hrx)
            assert not eng.rx["rx1.rx0"].streaming
            assert hrx.id not in srv._egress
        finally:
            cli.close()
    assert np.array_equal(bits(y), bits(_inprocess_reference(x)))
    assert float(np.mean(np.abs(y[:N_TX]) ** 2)) > 0.0


def test_rt08_client_cannot_borrow_a_barrier_controllers_src() -> None:
    """The engine trusts a barrier controller's commands (lookahead bound; a late one does
    not set ``nondeterministic_inputs``). A socket client naming an in-process controller,
    or another connection's remote controller, as ``src`` is refused (NOT_PERMITTED), so
    a late client command is always flagged in lockstep."""
    from tests.runtime._engine_support import ScriptedController, alternate_script

    eng = _engine(traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)})
    n = eng.commands.specs["ris0"].n_controls
    eng.register_controller(
        ScriptedController("alt", "1/1000", "2/10000", alternate_script("ris0", n))
    )
    with served(eng, start_paused=True) as srv:
        a = ClientSession(str(srv.endpoints["control"]), name="a")
        b = ClientSession(str(srv.endpoints["control"]), name="b")
        try:
            b.call("ctrl.register", name="remote1", period_s="1/1000", min_latency_s="1e-4")
            marker = {"op": "rec.marker", "label": "m"}
            for src in ("alt", "remote1"):
                r = a.request({**marker, "src": src})
                assert r["ok"] is False and r["error"]["code"] == "NOT_PERMITTED", r
            hello_alias = ClientSession(str(srv.endpoints["control"]), name="alt")
            r = hello_alias.request(marker)  # default src = the client name
            assert r["error"]["code"] == "NOT_PERMITTED", r
            hello_alias.close()
            assert a.request({**marker, "src": "a"})["ok"]  # its own name is fine
            assert b.request({**marker, "src": "remote1"})["ok"]  # the owner may
        finally:
            a.close()
            b.close()


def test_i6_orderly_shutdown_at_a_remote_barrier_is_not_an_engine_error() -> None:
    eng = _engine(traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)})
    srv = serve(eng)
    s = ClientSession(str(srv.endpoints["control"]), name="c")
    try:
        s.call("ctrl.register", name="c1", period_s="1/1000", min_latency_s="1e-4")
        wait_until(lambda: srv.controllers["c1"].pending == 0, 30.0, "the first barrier")
        s.call("sim.shutdown")
        assert srv.wait(10.0)
    finally:
        srv.stop()
        s.close()
    assert srv.engine_error is None


def test_i6_ack_timeout_is_still_an_engine_error() -> None:
    eng = _engine(traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)})
    srv = serve(eng, ack_timeout_s=0.3)
    s = ClientSession(str(srv.endpoints["control"]), name="c")
    try:
        s.call("ctrl.register", name="c1", period_s="1/1000", min_latency_s="1e-4")
        wait_until(lambda: srv.engine_error is not None, 30.0, "the ack timeout")
        assert "did not ack" in str(srv.engine_error)
        assert s.call("stats.get")["engine_error"] is not None
    finally:
        srv.stop()
        s.close()


def test_rt18_paced_run_refuses_pause_once_running() -> None:
    """A paced run follows the wall clock and has no pause point: ``sim.pause`` is accepted
    while the server still holds the start (``start_paused``) and refused (NOT_PERMITTED)
    once the run is going, instead of replying ``paused: true`` while time runs on."""
    eng = _engine(
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        mode="paced",
        coupling="async",
        duration_s=Fraction(1, 10),
    )
    with served(eng, start_paused=True) as srv:
        s = ClientSession(str(srv.endpoints["control"]), name="p")
        try:
            assert s.call("sim.pause")["paused"] is True  # still held at t = 0
            s.call("sim.resume")
            r = s.request({"op": "sim.pause"})
            assert r["ok"] is False and r["error"]["code"] == "NOT_PERMITTED", r
            assert not srv.paused
        finally:
            s.close()


def test_io01_telemetry_with_non_finite_floats_stays_schema_valid() -> None:
    """A metrics window can carry ``rtf = inf`` (no wall time elapsed on a coarse clock):
    PUB telemetry sends non-finite floats as strings and still validates against
    ``telemetry.schema.json`` (it used to be replaced by a body without ``window``)."""
    import json
    import math

    from sremu.transport.messages import validator

    eng = _engine()
    with served(eng, start_paused=True) as srv:
        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        try:
            sub.setsockopt(zmq.SUBSCRIBE, b"metrics.window")
            sub.connect(str(srv.endpoints["pub"]))
            body = {"window": {"rtf": math.inf, "wall_s": 0.0, "x": [math.nan, -math.inf]}}
            msg = None
            for _ in range(100):  # PUB/SUB joins asynchronously: publish until one arrives
                srv.publish("metrics.window", body)
                if sub.poll(50):
                    msg = sub.recv_multipart()
                    break
            assert msg is not None and msg[0] == b"metrics.window"
            d = json.loads(msg[1])
            assert d["window"] == {"rtf": "Infinity", "wall_s": 0.0, "x": ["NaN", "-Infinity"]}
            assert validator("telemetry.schema.json").is_valid(d)
        finally:
            sub.close(0)
            ctx.term()


def test_io01_oversize_tx_part_is_counted_below_the_backstop() -> None:
    """A TX frame whose payload exceeds the largest valid one (2**22 cf32 samples, 32 MiB)
    is still received and counted as a bad packet (``payload_size``) below the 128 MiB
    ZMQ backstop; later valid frames are taken as usual."""
    from sremu.contracts import MAX_FRAME_SAMPLES
    from sremu.transport.zmq_native import TX_HARD_PART_LIMIT

    assert TX_HARD_PART_LIMIT == 4 * MAX_FRAME_SAMPLES * 8
    eng = _engine(unopened_tx="wait")
    with served(eng) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
            push = cli.ctx.socket(zmq.PUSH)
            push.setsockopt(zmq.LINGER, 0)
            push.connect(str(srv.endpoints["tx"]))
            hdr = FrameHeader(int(Flag.HAS_TIME | Flag.SOB), h.id, 0, 0, MAX_FRAME_SAMPLES)
            push.send_multipart([hdr.pack(), bytes(MAX_FRAME_SAMPLES * 8 + 8)])
            good = FrameHeader(int(Flag.HAS_TIME | Flag.SOB), h.id, 0, 0, 10)
            push.send_multipart(make_frame(good, np.ones(10, np.complex64)))
            wait_until(lambda: srv.counters["frames_in"] == 1, 60.0, "the valid frame")
            assert srv.counters.reasons_snapshot() == {"payload_size": 1}
            assert eng.tx_head(h) == 10
            push.close(0)
        finally:
            cli.close()
