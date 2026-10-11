"""Data-plane flow control (DESIGN.md sections 4.10, 8.3; I6).

* Lockstep TX backpressure: a client that sends far ahead of its receivers is held at the
  ZMQ high-water mark (TX ingress stops reading beyond ``tx_window_samples``); nothing is
  dropped, and everything is accepted once the receivers consume.
* Lockstep RX backpressure: a client that stops reading stops the engine (bounded
  frontier); reading again resumes it without loss.
* Paced RX overflow: a client too slow for the paced engine loses pages, counted (I6), and
  its next read returns ``OVERFLOW`` with the post-gap ``time_spec`` (``OVERFLOW_BEFORE``),
  after which samples continue contiguously.
* A gap in a TX stream's frame counter is counted (``seq_errors``) and reported to the
  stream's owner as ``SEQ_ERROR``.
* ``IDLE_UNTIL`` and ``END_OF_STREAM`` frames commit a TX stream as silence (section 4.8).
"""

from __future__ import annotations

import threading
import time
from fractions import Fraction

import numpy as np
import pytest
import zmq

from sremu.contracts import StreamCmd, StreamSpec, TxMetadata
from sremu.engine.streams import ToneSource
from sremu.errors import ErrorCode
from sremu.timebase import TimeSpec
from sremu.transport.client import RemoteEngine
from tests.io._wp8_support import served, tcp_buffer_bytes
from tests.runtime._engine_support import lab_like_fake, make_engine

pytestmark = pytest.mark.io

F = Fraction(10**6)


def test_i6_lockstep_tx_backpressure_holds_the_client_without_loss() -> None:
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    with served(eng, tx_window_samples=50_000, tx_rcvhwm=4) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            htx = cli.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
            assert cli._tx is not None
            chunk = np.ones(10_000, np.complex64)
            total = 300
            sent = [0]

            def sender() -> None:
                for i in range(total):
                    md = TxMetadata(i == 0, TimeSpec(0, 0), i == 0, i == total - 1)
                    cli.submit_tx(htx, chunk, md)
                    sent[0] += 1

            th = threading.Thread(target=sender, daemon=True)
            th.start()
            time.sleep(1.0)
            held = sent[0]
            assert th.is_alive() and held < total  # blocked at the high-water mark
            ingested = eng.tx_head(htx)
            assert 50_000 < ingested <= 50_000 + 10_000 * 40  # window + bounded queues
            # a receiver consumes: the engine advances, ingress resumes, nothing is lost
            hrx = cli.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
            cli.stream_cmd(hrx, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
            got = 0
            t0 = time.monotonic()
            while eng.tx_head(htx) < total * 10_000:  # keep consuming until all TX is in
                y, md = cli.read_rx(hrx, 1 << 16, 1.0)
                assert md.error_code in (ErrorCode.NONE, ErrorCode.TIMEOUT)
                assert y.size == 0 or md.first_sample == got  # no RX sample lost either
                got += y.size
                assert time.monotonic() - t0 < 120, "TX ingestion did not resume"
            th.join(60)
            assert not th.is_alive() and sent[0] == total
            assert srv.counters["bad_packets"] == 0 and eng.counters["late_samples"] == 0
            assert srv.counters["frames_in"] == total
        finally:
            cli.close()


def _held_frontier(eng, stream: str, *, still_s: float = 1.0, deadline_s: float = 120.0) -> int:
    """The RX frontier once the engine is held by its non-reading client: the stream's RX
    queue is full (the lockstep backpressure condition) and the frontier has not moved for
    ``still_s`` while it stayed full. How long the engine, the egress pump, ZMQ and the OS
    TCP buffers take to fill depends on the machine and its load (CI: 2 vCPUs shared by 8
    xdist workers), so the test waits for that state instead of sampling the frontier at
    fixed times; an engine that never stops for its reader fails at ``deadline_s``."""
    st = eng.rx[stream]
    t_end = time.monotonic() + deadline_s
    last: int | None = None
    since = time.monotonic()
    while True:
        f = int(eng.frontier()[stream])
        now = time.monotonic()
        if f != last or st.queued < st.queue_samples:
            last, since = f, now
        elif now - since >= still_s:
            return f
        assert now < t_end, f"the engine did not stop for its reader (frontier {f})"
        time.sleep(0.02)


def test_i6_lockstep_rx_backpressure_bounds_the_frontier() -> None:
    eng = make_engine(lab_like_fake(), traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)})
    # server egress HWM: 2 pages, rounded up to whole frames (one frame of a lockstep batch)
    with served(eng, rx_sndhwm=2) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
            cli.stream_cmd(h, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
            f_held = _held_frontier(eng, "rx1.rx0")  # the engine waits for the reader
            # bounded by the configured buffering: the engine's RX queue (64 pages), the ZMQ
            # high-water marks (in frames; lockstep batches several pages per frame and
            # sizes both marks in frames of that length) and the OS TCP buffers in between
            eg, rx = srv._egress[h.id], cli._rx[h.id]
            assert eg.frame_samples == rx.frame_samples >= h.page_size
            queue = eng.rx["rx1.rx0"].queue_samples
            frames = eg.sndhwm + rx.rcvhwm + 2
            bound = queue + frames * eg.frame_samples + tcp_buffer_bytes() // 8
            assert queue <= f_held <= bound
            # reading again resumes the engine; every sample arrives, in order, no loss
            expect = 0
            while expect < f_held + queue:
                y, md = cli.read_rx(h, 4096, 30.0)
                assert md.error_code == ErrorCode.NONE and md.first_sample == expect
                expect += y.size
            assert eng.frontier()["rx1.rx0"] > f_held
            assert eng.counters["overflow_pages"] == 0
            assert srv.counters["rx_egress_dropped_samples"] == 0
        finally:
            cli.close()


def test_i6_lockstep_rx_backpressure_slow_small_reads_keep_the_client_bounded() -> None:
    """A reader that keeps calling ``recv`` but takes one sample at a time must still hold
    the engine back: the client takes frames off its socket only as far as each read needs
    them (at most one frame buffered), so the rest stays queued in ZMQ at the high-water
    mark instead of piling up in client memory (found by the WP8 review: every read used
    to drain the whole socket). A frame is several pages in lockstep (RX batching)."""
    eng = make_engine(lab_like_fake(), traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)})
    with served(eng, rx_sndhwm=2) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
            cli.stream_cmd(h, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
            rx = cli._rx[h.id]
            got = 0
            peak = 0
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.5 or got < 5000:
                y, md = cli.read_rx(h, 1, 5.0)
                assert md.error_code == ErrorCode.NONE and md.first_sample == got
                got += y.size
                peak = max(peak, rx.queued)
            assert peak < rx.frame_samples  # one RX frame minus the sample just read
            assert eng.frontier()["rx1.rx0"] - got < 250_000  # the engine was held back
        finally:
            cli.close()


def test_rt18_paced_rx_overflow_is_counted_and_reported() -> None:
    eng = make_engine(
        lab_like_fake(),
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        mode="paced",
        coupling="async",
        duration_s=Fraction(4, 10),
        rx_queue_pages=4,
    )
    with served(eng, start_paused=True, rx_sndhwm=1) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
            cli._rx[h.id].sock.setsockopt(zmq.RCVHWM, 1)  # a client with little buffering
            cli.stream_cmd(h, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
            cli.resume()
            time.sleep(0.25)  # a slow client: the paced engine does not wait for it
            overflow = None
            expect = None
            n = 0
            while not srv.is_done() or n < 50:
                y, md = cli.read_rx(h, 1 << 16, 0.5)
                if md.error_code == ErrorCode.OVERFLOW:
                    overflow = md
                    expect = md.first_sample
                    continue
                if y.size:
                    if expect is not None:
                        assert md.first_sample == expect  # contiguous after the gap
                        expect += y.size
                    n += 1
                elif srv.is_done():
                    break
            assert overflow is not None
            assert overflow.time_spec == TimeSpec.from_ticks(overflow.first_sample, F)
            lost = eng.counters["overflow_pages"] + srv.counters["rx_egress_dropped_samples"]
            assert lost > 0
        finally:
            cli.close()


def test_i6_seq_gap_counted_and_reported_as_seq_error() -> None:
    """A gap in a TX stream's frame counter counts ``seq_errors`` and sends ``SEQ_ERROR``
    to the stream's owner; the frame itself is still used (UHD semantics)."""
    from sremu.transport.framing import Flag, FrameHeader, make_frame

    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    with served(eng) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
            push = cli.ctx.socket(zmq.PUSH)
            push.setsockopt(zmq.LINGER, 0)
            push.connect(str(srv.endpoints["tx"]))
            x = np.ones(10, np.complex64)
            for seq, first, flags in (
                (0, 0, Flag.HAS_TIME | Flag.SOB),
                (1, 0, 0),
                (7, 0, Flag.EOB),
            ):
                push.send_multipart(make_frame(FrameHeader(int(flags), h.id, seq, first, 10), x))
            events = []
            for _ in range(200):
                ev = cli.recv_async(h, 0.05)
                if ev is not None:
                    events.append(ev.event_code.name)
                if "BURST_ACK" in events:
                    break
            assert events == ["SEQ_ERROR", "BURST_ACK"]
            assert srv.counters["seq_errors"] == 1 and srv.counters["bad_packets"] == 0
            assert eng.tx_head(h) == 30  # the frame after the gap was used
            push.close(0)
        finally:
            cli.close()


def test_rt13_idle_until_and_end_of_stream_frames_commit_the_tx() -> None:
    """``idle_until`` (frame flag IDLE_UNTIL) and ``end_of_stream`` (END_OF_STREAM) commit
    a client TX stream as silence, so lockstep receivers advance without TX samples."""
    from sremu.api import Device, RxMetadata, StreamArgs

    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    with (
        served(eng, start_paused=True) as srv,
        Device.connect(str(srv.endpoints["control"])) as dev,
    ):
        rx = dev.get_rx_stream(StreamArgs(stream="rx1.rx0", fc_hz=3.5e9, rate_sps="1e6"))
        rx.issue_stream_cmd(StreamCmd("num_done", 6000, False, TimeSpec(0, 0)))
        tx = dev.get_tx_stream(StreamArgs(stream="ap.tx0", fc_hz=3.5e9, rate_sps="1e6"))
        dev.engine.resume()  # type: ignore[attr-defined]
        out = np.zeros(6000, np.complex64)
        md = RxMetadata()
        assert rx.recv(out, md, timeout=0.5) == 0  # nothing committed yet: the engine waits
        tx.idle_until(TimeSpec.from_ticks(4000, F))
        got = 0
        while got < 3000:
            got += rx.recv(out[got:], md, timeout=5.0)
        assert eng.frontier()["rx1.rx0"] < 4000 + 1024  # not beyond the promise (+ a page)
        tx.end_of_stream()
        while got < 6000:
            got += rx.recv(out[got:], md, timeout=5.0)
        assert md.end_of_burst and got == 6000
        assert not np.any(out)  # noise is off and the transmitter was silent
