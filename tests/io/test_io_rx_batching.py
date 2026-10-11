"""Lockstep RX egress batching (DESIGN.md section 8.3; served data plane).

In lockstep the RX egress pump sends ``RX_BATCH_PAGES`` pages per frame: it sleeps in
``Emulator.wait_rx`` until a full frame is queued, and the engine wakes it earlier when it
is about to wait for an outside input (``mark_blocked``: no TX data, a barrier, a pause, the
end of the run). Paced mode keeps one page per frame.

* The samples, their indices and the burst flags a client reads are the same for every
  frame length (one page, the default batch, ``2**22``) and equal to the in-process engine;
  frames never cross an end of burst or a gap; batching cuts the frame count.
* Closed loop: a client that sends TX only up to the RX it has received plus a lead (one
  page, a batch minus a page, a batch, four batches) completes with ``rx_flush_s`` = 60 s,
  so the engine-blocked flush, not the wall timeout, delivers each partial batch.
* A pause marks the engine blocked; paced streams and an explicit one-page frame do not
  batch; both high-water marks are sized in frames of the batch; a server stop ends a pump's
  long flush wait at once.
* ``Emulator.wait_rx``: woken at ``n`` queued samples, and by the blocked signal only when
  samples are queued; with two rate groups, also when its own group waits for TX while the
  other group keeps the engine busy (the engine as a whole never blocks then).

Scenes are SIM geometry on ``FakePropagation`` (no physics result).
"""

from __future__ import annotations

import threading
import time
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from sremu.contracts import StreamCmd, StreamSpec, TxMetadata
from sremu.engine.streams import ToneSource
from sremu.errors import ErrorCode
from sremu.testing.fake_propagation import FakeNode, FakePropagation
from sremu.testing.scenes import stream
from sremu.timebase import TimeSpec
from sremu.transport.client import RemoteEngine
from sremu.transport.control_server import RX_BATCH_PAGES
from tests.io._wp8_support import bits, serve, served, wait_until
from tests.runtime._engine_support import client_run, lab_like_fake, make_engine, waveform

pytestmark = pytest.mark.io

F = Fraction(10**6)
FC = 3.5e9
PAGE = 512  # 1 MS/s, default page target
BATCH = RX_BATCH_PAGES * PAGE


def _reference(x: np.ndarray, n: int) -> np.ndarray:
    eng = make_engine(lab_like_fake())
    try:
        return client_run(eng, "ap.tx0", "rx1.rx0", x, n, send_chunk=1000, read_chunk=4096)
    finally:
        eng.close()


def _send(cli: RemoteEngine, h: Any, x: np.ndarray, a: int, b: int, chunk: int = 4096) -> None:
    while a < b:
        m = min(chunk, b - a)
        md = TxMetadata(a == 0, TimeSpec(0, 0), a == 0, a + m >= x.shape[0])
        assert cli.submit_tx(h, x[a : a + m], md) == m
        a += m


def _read(cli: RemoteEngine, h: Any, n: int, first: int, timeout_s: float = 20.0) -> tuple:
    """``n`` samples from ``first`` on: (samples, reads that started / ended a burst)."""
    parts, got, sob, eob = [], 0, [], []
    while got < n:
        y, md = cli.read_rx(h, min(4096, n - got), timeout_s)
        assert md.error_code == ErrorCode.NONE, md.error_code
        assert md.first_sample == first + got  # contiguous, in order
        if md.start_of_burst:
            sob.append(first + got)
        got += y.size
        if md.end_of_burst:
            eob.append(first + got)
        parts.append(y)
    return np.concatenate(parts), sob, eob


@pytest.mark.parametrize("frame", [PAGE, None, 1 << 22])
def test_same_samples_and_flags_for_every_frame_length(frame: int | None) -> None:
    n1, gap, n2 = 10_000, 5_000, 9_000
    k2 = n1 + gap
    x = waveform(k2 + n2 + 1024, seed=3)
    ref = _reference(x, k2 + n2)
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    kw: dict[str, Any] = {} if frame is None else {"rx_frame_samples": frame}
    with served(eng, start_paused=True, **kw) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            htx = cli.open_stream(StreamSpec("ap.tx0", "tx", FC, F))
            hrx = cli.open_stream(StreamSpec("rx1.rx0", "rx", FC, F))
            eg = srv._egress[hrx.id]
            assert eg.batching == (frame != PAGE)  # frames longer than a page batch
            assert eg.frame_samples == cli._rx[hrx.id].frame_samples == (frame or BATCH)
            cli.stream_cmd(hrx, StreamCmd("num_done", n1, False, TimeSpec(0, 0)))
            _send(cli, htx, x, 0, x.shape[0])
            cli.resume()
            y1, sob1, eob1 = _read(cli, hrx, n1, 0)
            # second burst after a gap (the server holds virtual time meanwhile: no RX
            # stream streams, so the start lies ahead of the frontier)
            assert max(cli.frontier().values()) <= k2
            cli.stream_cmd(hrx, StreamCmd("num_done", n2, False, TimeSpec.from_ticks(k2, F)))
            y2, sob2, eob2 = _read(cli, hrx, n2, k2)
            frames = srv.counters["frames_out"]
        finally:
            cli.close()
    assert np.array_equal(bits(y1), bits(ref[:n1]))
    assert np.array_equal(bits(y2), bits(ref[k2:]))
    assert (sob1, eob1, sob2, eob2) == ([0], [n1], [k2], [k2 + n2])  # a frame never crosses
    pages = -(-n1 // PAGE) + -(-n2 // PAGE)
    if frame == PAGE:
        assert frames >= pages
    else:
        assert frames < pages / 2  # several pages per frame


@pytest.mark.parametrize("lead", [PAGE, BATCH - PAGE, BATCH, 4 * BATCH])
def test_closed_loop_client_is_flushed_by_the_blocked_engine(lead: int) -> None:
    """TX only up to the RX received + ``lead``: with a 60 s flush timeout, any batch that
    waited for its timer would fail the 20 s reads."""
    n = 40 * PAGE + 100
    x = waveform(n + 1024, seed=5)
    ref = _reference(x, n)
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    with served(eng, start_paused=True, rx_flush_s=60.0) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            htx = cli.open_stream(StreamSpec("ap.tx0", "tx", FC, F))
            hrx = cli.open_stream(StreamSpec("rx1.rx0", "rx", FC, F))
            assert srv._egress[hrx.id].batching
            cli.stream_cmd(hrx, StreamCmd("num_done", n, False, TimeSpec(0, 0)))
            sent = min(lead, x.shape[0])
            _send(cli, htx, x, 0, sent)
            cli.resume()
            t0 = time.monotonic()
            parts, got = [], 0
            while got < n:
                y, md = cli.read_rx(hrx, 4096, 20.0)
                assert md.error_code == ErrorCode.NONE, (md.error_code, got)
                assert md.first_sample == got
                parts.append(y)
                got += y.size
                nxt = min(got + lead, x.shape[0])
                _send(cli, htx, x, sent, nxt)
                sent = max(sent, nxt)
            assert time.monotonic() - t0 < 30.0
        finally:
            cli.close()
    assert np.array_equal(bits(np.concatenate(parts)), bits(ref))


def test_pause_flushes_a_partial_batch_and_marks_the_engine_blocked() -> None:
    """A reader keeps up with the engine; after ``sim.pause`` every computed sample reaches
    it within seconds although the flush timer is 60 s (the last batch is partial unless
    the pause fell on a frame boundary)."""
    eng = make_engine(lab_like_fake(), traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)})
    with served(eng, rx_sndhwm=64, rx_flush_s=60.0) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        stop = threading.Event()
        got = [0]
        th: threading.Thread | None = None

        def reader() -> None:  # the RX socket is used by this thread only
            while not stop.is_set():
                y, md = cli.read_rx(h, 4096, 0.2)
                assert md.error_code in (ErrorCode.NONE, ErrorCode.TIMEOUT)
                assert not y.size or md.first_sample == got[0]
                got[0] += y.size

        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", FC, F))
            eg, rx = srv._egress[h.id], cli._rx[h.id]
            # 64 pages in flight at each end, in frames of a batch
            assert (eg.sndhwm, rx.rcvhwm) == (64 // RX_BATCH_PAGES, 64 // RX_BATCH_PAGES)
            th = threading.Thread(target=reader, daemon=True)
            cli.stream_cmd(h, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
            th.start()
            wait_until(lambda: got[0] > 5 * BATCH, 30.0, "RX flowing")
            cli.pause()
            wait_until(lambda: eng._blocked, 10.0, "the paused engine marked blocked")
            time.sleep(0.2)  # the step running at the pause has ended
            f = int(eng.frontier()["rx1.rx0"])
            wait_until(lambda: got[0] >= f, 10.0, f"every computed sample read ({f})")
            assert got[0] == f
        finally:
            stop.set()
            if th is not None and th.is_alive():
                th.join(5.0)
            cli.close()


def test_server_stop_ends_a_long_flush_wait_promptly() -> None:
    """An egress pump waiting for a batch (paused engine, empty queue, 60 s flush timer)
    sees ``stop()`` within a wait slice; it used to sit out the egress join timeout (5 s)
    and outlive its closed socket."""
    eng = make_engine(lab_like_fake(), traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)})
    srv = serve(eng, start_paused=True, rx_flush_s=60.0)
    try:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", FC, F))
            eg = srv._egress[h.id]
            assert eg.batching
            cli.stream_cmd(h, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
            time.sleep(0.3)  # the pump waits for a batch that cannot come
        finally:
            cli.close()
        t0 = time.monotonic()
        srv.stop()
        assert time.monotonic() - t0 < 2.0
        assert not eg.thread.is_alive()
    finally:
        srv.stop()


def test_end_of_stream_follows_the_end_of_the_run_promptly() -> None:
    """A finite run that ends while the pump waits for a batch (``start_cont``: the stream
    never stops by itself, the queue is empty): ``END_OF_STREAM`` follows within a wait
    slice, not after the 60 s flush timer (``exit_when_done`` waits for it)."""
    eng = make_engine(
        lab_like_fake(), traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)}, duration_s=Fraction(1, 20)
    )
    with served(eng, start_paused=True, rx_flush_s=60.0) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", FC, F))
            cli.stream_cmd(h, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
            cli.resume()
            rx = cli._rx[h.id]
            got = 0
            t_last = time.monotonic()
            while not rx.ended:
                y, md = cli.read_rx(h, 4096, 0.05)
                assert md.error_code in (ErrorCode.NONE, ErrorCode.TIMEOUT)
                if y.size:
                    got, t_last = got + y.size, time.monotonic()
                assert time.monotonic() - t_last < 10.0, "no END_OF_STREAM"
            assert got == 50_000
            assert time.monotonic() - t_last < 2.0
        finally:
            cli.close()


def test_paced_streams_keep_one_page_per_frame() -> None:
    eng = make_engine(
        lab_like_fake(),
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        mode="paced",
        coupling="async",
        duration_s=Fraction(1, 10),
    )
    with served(eng, start_paused=True) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", FC, F))
            eg = srv._egress[h.id]
            assert not eg.batching and eg.frame_samples == h.page_size
            assert (eg.sndhwm, cli._rx[h.id].rcvhwm) == (64, 64)
        finally:
            cli.close()


def test_wait_rx_wakes_on_a_full_batch_and_on_blocked_with_samples_only() -> None:
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    htx = eng.open_stream(StreamSpec("ap.tx0", "tx", FC, F))
    hrx = eng.open_stream(StreamSpec("rx1.rx0", "rx", FC, F))
    st = eng.rx["rx1.rx0"]
    eng.stream_cmd(hrx, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
    x = waveform(3 * BATCH, seed=9)
    eng.start()
    try:
        # (a) enough TX for several batches: woken once n samples are queued
        eng.submit_tx(htx, x[: 2 * BATCH], TxMetadata(True, TimeSpec(0, 0), True, False))
        t0 = time.monotonic()
        eng.wait_rx(hrx, BATCH, 30.0)
        assert time.monotonic() - t0 < 10.0 and st.queued >= BATCH
        y, _ = eng.read_rx(hrx, 1 << 20, 0.0)
        got = y.size
        while got < 2 * BATCH - 2 * PAGE:  # drain what the TX allows
            eng.wait_rx(hrx, BATCH, 5.0)
            got += eng.read_rx(hrx, 1 << 20, 0.0)[0].size
        # ... to the last sample: a fruitless step after the flag is cleared means the
        # engine computed all of it (the served loop steps every 5 ms while it waits). A
        # leftover would satisfy (b) at once and could be followed, under load, by the
        # engine computing (b)'s samples during (c)'s wait.
        with eng._lock:
            eng._blocked = False
        wait_until(lambda: eng._blocked, 10.0, "blocked on TX")
        got += eng.read_rx(hrx, 1 << 20, 0.0)[0].size
        assert st.queued == 0
        # (b) TX for a part of a batch: the engine computes it and blocks on TX; the
        # blocked signal wakes the waiter long before its 30 s
        eng.submit_tx(htx, x[2 * BATCH : 2 * BATCH + 1500], TxMetadata())
        t0 = time.monotonic()
        eng.wait_rx(hrx, 10 * BATCH, 30.0)
        assert time.monotonic() - t0 < 10.0
        assert 0 < st.queued < BATCH
        got += eng.read_rx(hrx, 1 << 20, 0.0)[0].size
        # (c) blocked with nothing queued: no wake-up, the wait runs to its timeout
        wait_until(lambda: eng._blocked and st.queued == 0, 10.0, "blocked, empty queue")
        t0 = time.monotonic()
        eng.wait_rx(hrx, BATCH, 0.3)
        assert time.monotonic() - t0 >= 0.25
    finally:
        eng.close()


def _two_group_model() -> FakePropagation:
    """``lab_like_fake``'s TX and rx1 (no panel) plus ``rx2.rx0`` at 0.5 MS/s on 3.6 GHz: a
    second rate group with no TX at its rate (no link: the bands do not overlap), so it
    never waits for TX."""
    return FakePropagation(
        [
            FakeNode("ap", (1.732050808, -1.0, 0.0), streams=(stream("ap.tx0", "tx", FC, F),)),
            FakeNode(
                "rx1", (1.159110992, 0.310582854, 0.0), streams=(stream("rx1.rx0", "rx", FC, F),)
            ),
            FakeNode("rx2", (0.98, 0.69, 0.0), streams=(stream("rx2.rx0", "rx", 3.6e9, F / 2),)),
        ],
        horizon_s="2",
    )


def test_wait_rx_wakes_when_its_group_waits_for_tx_while_another_group_runs() -> None:
    """rx1's group waits for TX while rx2's group (no client: its pages are skipped) always
    advances, so no step is fruitless and the engine-wide blocked signal never comes. The
    waiting group's pump must still get its partial batch at once, or a closed-loop client
    on rx1 waits for the flush timer (``rx_flush_s``) on every round trip."""
    eng = make_engine(_two_group_model(), unopened_tx="wait")
    assert len(eng.groups) == 2
    htx = eng.open_stream(StreamSpec("ap.tx0", "tx", FC, F))
    hrx = eng.open_stream(StreamSpec("rx1.rx0", "rx", FC, F))
    eng.stream_cmd(hrx, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
    eng.start()
    try:
        eng.submit_tx(htx, waveform(1500, seed=2), TxMetadata(True, TimeSpec(0, 0), True, False))
        t0 = time.monotonic()
        assert eng.wait_rx(hrx, BATCH, 6.0)  # woken, not timed out
        assert time.monotonic() - t0 < 3.0
        assert 0 < eng.rx["rx1.rx0"].queued < BATCH
        assert eng.groups[1].cursor > 0 and not eng._blocked  # rx2's group kept going
    finally:
        eng.close()
