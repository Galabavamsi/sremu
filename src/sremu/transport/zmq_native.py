"""Native framed ZMQ data plane (DESIGN.md section 8.3).

**Sockets.** TX ingress is one ``PULL`` bound at ``tcp://127.0.0.1:5602``; clients
``PUSH``. RX egress is one ``PUSH`` per opened RX stream, bound to an ephemeral localhost
port that ``stream.open`` returns; clients ``PULL``. Every message is a two-part frame
``[header, payload]`` (:mod:`sremu.transport.framing`). PUB/SUB never carries IQ.

**Server side.**

* :class:`TxIngress` validates each frame (a bad one is dropped and counted as
  ``bad_packets`` with its reason; the engine never sees it and never raises), checks the
  per-stream ``seq`` (a gap counts ``seq_errors`` and sends ``SEQ_ERROR`` to the stream's
  owner) and hands the samples to ``EngineProtocol.submit_tx`` with the frame's timing:
  ``HAS_TIME`` -> a timed burst at ``first_sample``, ``IDLE_UNTIL`` -> commit silence through
  ``first_sample``, ``SOB``/``EOB``/``END_OF_STREAM`` -> burst flags.
  **Backpressure (lockstep):** while the engine is not waiting for any TX input and an open
  TX stream is more than ``window_samples`` ahead of its receivers' frontier, the ingress
  stops reading; ZMQ queues fill to their high-water mark and the clients' ``PUSH`` blocks.
* :class:`RxEgress` reads computed samples from the engine and pushes RX frames
  (``HAS_TIME`` always, ``SOB``/``EOB`` from the stream, ``OVERFLOW_BEFORE`` after a gap,
  ``LATE`` for a stream command in the past, ``END_OF_STREAM`` once the run is over and
  drained). In lockstep the send blocks at the high-water mark (the engine's RX queue then
  fills and the engine stops: backpressure). In paced mode the send is ``NOBLOCK``; on
  ``EAGAIN`` the frame is dropped, counted (``rx_egress_dropped_samples``) and the next frame
  carries ``OVERFLOW_BEFORE``.
  **Batching (lockstep).** With frames longer than a page, the pump sleeps in the engine's
  ``wait_rx`` until a full frame is queued and sends it as one message: one wake-up and
  one send per frame instead of per page, which is most of the served cost in one
  interpreter (each wake-up is a GIL hand-off with the engine thread). It sends a shorter
  frame at once when the engine cannot progress (no TX data, a barrier, a pause, the end of
  the run), so a client that times its TX from its RX never waits on a half-filled frame,
  and after ``flush_s`` wall seconds otherwise. Frames carry the same samples, indices and
  flags in the same order; an end of burst or a gap still ends a frame. Paced mode keeps
  one frame per page (a batch would add its length to the delivery latency).
  **High-water marks** count messages: both ends size theirs in frames so that about the
  same number of samples is in flight whatever the frame length (:func:`hwm_frames`), and
  the lockstep run-ahead of the engine over a slow reader stays bounded as with page-sized
  frames.

**Client side.** :class:`TxSender` turns ``submit_tx`` calls into frames (splitting at
``2**22`` samples); :class:`RxReceiver` turns frames back into ``read_rx`` results with the
in-process semantics (one contiguous run per read, never across an end of burst or a gap;
``OVERFLOW`` with the post-gap ``time_spec``; ``LATE_COMMAND``). It takes frames off its
socket only as far as the current read needs them, so a slow reader leaves the rest queued
in ZMQ and the lockstep backpressure reaches the engine.

**Arrival wake-ups.** A caller that polls ``read_rx`` with timeout 0 and waits in between
(``sremu.Device`` does, under its own lock) learns of a new frame only at its next poll.
:class:`ReadyWatcher` closes that gap without touching any socket: its thread
``select()``s on the sockets' ``ZMQ_FD`` handles, which libzmq signals when a message
reaches a socket whose reader last found it empty, and then calls its listeners
(``Device.notify``). Socket ownership, read semantics and backpressure are unchanged.

``wall_ns`` (the sender's ``time.perf_counter_ns``: monotonic and system-wide, the clock of
the engine's latency stamps; ``time.monotonic`` ticks at 15.6 ms on Windows) is carried for
latency metrics only; it never enters data-plane arithmetic (I7).
"""

from __future__ import annotations

import contextlib
import select
import socket
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import zmq
from numpy.typing import NDArray

from sremu.contracts import (
    COUNTER_NAMES,
    MAX_FRAME_SAMPLES,
    AsyncMetadata,
    EngineProtocol,
    RxMetadata,
    StreamHandle,
    TxMetadata,
)
from sremu.errors import AsyncCode, ErrorCode, SremuError
from sremu.timebase import TimeSpec, to_ticks
from sremu.transport.framing import (
    BadFrame,
    Dtype,
    Flag,
    FrameHeader,
    make_frame,
    parse_frame,
)

__all__ = [
    "RX_FLUSH_S",
    "RX_HWM_PAGES",
    "ReadyWatcher",
    "RxEgress",
    "RxReceiver",
    "TransportCounters",
    "TxIngress",
    "TxSender",
    "hwm_frames",
]

POLL_MS: Final[int] = 20
"""Poll granularity of the transport threads, ms (wall)."""
RX_HWM_PAGES: Final[int] = 64
"""Default RX high-water mark of each end (server egress, client receiver), in pages."""
RX_FLUSH_S: Final[float] = 0.01
"""Default wall time a batching egress pump waits for a full frame before it sends what is
queued (the engine-blocked flush normally comes first)."""
WATCH_IDLE_S: Final[float] = 1.0
"""Select timeout of the :class:`ReadyWatcher` thread (liveness only; every change of its
handle set wakes it at once)."""
TX_HARD_PART_LIMIT: Final[int] = 4 * MAX_FRAME_SAMPLES * 8
"""TX ingress parts above 128 MiB are dropped by ZMQ (peer disconnected) without a count."""


def hwm_frames(hwm_pages: int, page: int, frame_samples: int) -> int:
    """High-water mark in messages that holds about ``hwm_pages`` pages of ``page`` samples
    in frames of ``frame_samples`` (at least 1)."""
    return max(1, -(-int(hwm_pages) * int(page) // max(1, int(frame_samples))))


class TransportCounters:
    """Thread-safe I6 counters of the transport (canonical names plus transport-specific
    ones: ``too_large_messages``, ``rx_egress_dropped_samples``, ``frames_in``,
    ``frames_out``, ``internal_errors``) and the reasons of bad packets."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: dict[str, int] = dict.fromkeys(COUNTER_NAMES, 0)
        self._c.update(
            too_large_messages=0,
            rx_egress_dropped_samples=0,
            frames_in=0,
            frames_out=0,
            internal_errors=0,
        )
        self.reasons: Counter[str] = Counter()

    def add(self, name: str, n: int = 1) -> None:
        """Increase ``name`` by ``n``."""
        with self._lock:
            self._c[name] = self._c.get(name, 0) + int(n)

    def bad(self, reason: str) -> None:
        """Count one bad packet with its reason."""
        with self._lock:
            self._c["bad_packets"] += 1
            self.reasons[reason] += 1

    def __getitem__(self, name: str) -> int:
        with self._lock:
            return self._c.get(name, 0)

    def snapshot(self) -> dict[str, int]:
        """A copy of every counter."""
        with self._lock:
            return dict(self._c)

    def reasons_snapshot(self) -> dict[str, int]:
        """A copy of the bad-packet reasons."""
        with self._lock:
            return dict(self.reasons)


def _now_ns() -> int:
    return time.perf_counter_ns()


def tx_metadata(h: FrameHeader, rate: Any) -> TxMetadata:
    """``TxMetadata`` of a TX frame header at stream rate ``rate``."""
    if h.has(Flag.IDLE_UNTIL):
        return TxMetadata(
            has_time_spec=True, time_spec=TimeSpec.from_ticks(h.first_sample, rate), idle_until=True
        )
    timed = h.has(Flag.HAS_TIME)
    return TxMetadata(
        has_time_spec=timed,
        time_spec=TimeSpec.from_ticks(h.first_sample, rate) if timed else TimeSpec(),
        start_of_burst=h.has(Flag.SOB),
        end_of_burst=h.has(Flag.EOB),
        end_of_stream=h.has(Flag.END_OF_STREAM),
    )


# =========================================================================== server: TX


class TxIngress:
    """The TX ingress ``PULL`` socket and its thread (see the module docstring).

    ``lookup(stream_id)`` returns the open TX ``StreamHandle`` (None for unknown ids);
    ``on_async(handle, md)`` delivers SEQ_ERROR to the stream's owner; ``throttle()``
    returns True while reading should pause (lockstep backpressure).
    """

    def __init__(
        self,
        ctx: zmq.Context[Any],
        endpoint: str,
        engine: EngineProtocol,
        lookup: Callable[[int], StreamHandle | None],
        counters: TransportCounters,
        *,
        on_async: Callable[[StreamHandle, AsyncMetadata], None] | None = None,
        throttle: Callable[[], bool] | None = None,
        rcvhwm: int = 64,
    ) -> None:
        self.engine = engine
        self.lookup = lookup
        self.counters = counters
        self.on_async = on_async
        self.throttle = throttle
        self.sock = ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVHWM, int(rcvhwm))
        # memory backstop (section 8.10), as on the control socket: no valid part exceeds a
        # full cf32 payload (32 MiB); parts up to 4x that are received and counted as bad
        # packets (payload_size), larger ones make ZMQ disconnect the peer (never allocated,
        # not counted)
        self.sock.setsockopt(zmq.MAXMSGSIZE, TX_HARD_PART_LIMIT)
        self.sock.bind(endpoint)
        self.endpoint: str = self.sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self._seq: dict[int, int] = {}
        self._stop = threading.Event()
        self.frames = 0
        self.samples = 0
        self.last_latency_ns: int | None = None
        self.thread = threading.Thread(target=self._run, name="sremu-tx-ingress", daemon=True)

    def start(self) -> None:
        """Start the ingress thread."""
        self.thread.start()

    def stop(self) -> None:
        """Stop the thread and close the socket."""
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(5.0)
        self.sock.close(0)

    def forget(self, stream_id: int) -> None:
        """Drop the sequence state of a closed stream."""
        self._seq.pop(int(stream_id), None)

    def next_seq(self, stream_id: int) -> int | None:
        """``seq`` expected next on ``stream_id`` (the number of frames ingested so far
        when the sender had no gap); None before its first frame."""
        return self._seq.get(int(stream_id))

    def _run(self) -> None:
        sock = self.sock
        while not self._stop.is_set():
            if self.throttle is not None and self.throttle():
                self._stop.wait(POLL_MS / 1000.0 / 4)
                continue
            if not sock.poll(POLL_MS, zmq.POLLIN):
                continue
            try:
                parts = sock.recv_multipart(zmq.NOBLOCK, copy=False)
            except zmq.Again:
                continue
            except zmq.ZMQError:  # pragma: no cover - socket closed under us
                break
            try:
                self.handle(parts)
            except Exception:  # last line of defence: the ingress thread must not die (I6)
                self.counters.bad("internal")
                self.counters.add("internal_errors")

    def handle(self, parts: list[Any]) -> bool:
        """Validate and submit one received message; False when it was dropped."""
        try:
            fr = parse_frame(parts)
        except BadFrame as exc:
            self.counters.bad(exc.reason)
            return False
        hdr = fr.header
        h = self.lookup(hdr.stream_id)
        if h is None or h.spec.direction != "tx":
            self.counters.bad("stream_id")
            return False
        exp = self._seq.get(hdr.stream_id)
        if exp is not None and hdr.seq != exp:
            self.counters.add("seq_errors")
            if self.on_async is not None:
                k = hdr.first_sample if hdr.has(Flag.HAS_TIME) else None
                self.on_async(
                    h,
                    AsyncMetadata(
                        AsyncCode.SEQ_ERROR,
                        h.spec.stream,
                        0,
                        k is not None,
                        TimeSpec.from_ticks(k, h.spec.rate_sps) if k is not None else TimeSpec(),
                        k,
                        0,
                    ),
                )
        self._seq[hdr.stream_id] = (hdr.seq + 1) & 0xFFFFFFFF
        md = tx_metadata(hdr, h.spec.rate_sps)
        try:
            self.engine.submit_tx(h, fr.samples, md)
        except SremuError:
            self.counters.bad("rejected")
            return False
        self.counters.add("frames_in")
        self.frames += 1
        self.samples += hdr.nsamps
        if hdr.wall_ns:
            self.last_latency_ns = _now_ns() - hdr.wall_ns
        return True


# =========================================================================== server: RX


class RxEgress:
    """One RX stream's egress ``PUSH`` socket and pump thread (see the module docstring).

    ``frame_samples``: samples per frame (at most). ``paced``: NOBLOCK sends, no batching.
    ``sndhwm``: the socket's high-water mark in messages. ``is_done()``: whether the run has
    ended (then an empty read ends the stream). ``frontier()``: the stream's first
    uncomputed sample (for ``LATE`` frames). ``publish(topic, body)``: telemetry.
    ``flush_s``: longest wait for a full frame when batching (lockstep, frames longer than a
    page, an engine with ``wait_rx``).
    """

    def __init__(
        self,
        ctx: zmq.Context[Any],
        bind: str,
        engine: EngineProtocol,
        handle: StreamHandle,
        counters: TransportCounters,
        *,
        frame_samples: int,
        paced: bool = False,
        sndhwm: int = RX_HWM_PAGES,
        is_done: Callable[[], bool] | None = None,
        frontier: Callable[[], int] | None = None,
        publish: Callable[[str, dict[str, Any]], None] | None = None,
        read_timeout_s: float = 0.05,
        flush_s: float = RX_FLUSH_S,
    ) -> None:
        self.engine = engine
        self.handle = handle
        self.counters = counters
        self.frame_samples = max(1, min(int(frame_samples), MAX_FRAME_SAMPLES))
        self.paced = paced
        self.is_done = is_done or (lambda: False)
        self.frontier = frontier or (lambda: 0)
        self.publish = publish
        self.read_timeout_s = float(read_timeout_s)
        self.flush_s = float(flush_s)
        wait_rx = getattr(engine, "wait_rx", None)
        batch = not paced and self.frame_samples > handle.page_size and callable(wait_rx)
        self._wait_rx: Callable[[StreamHandle, int, float], bool | None] | None = (
            wait_rx if batch else None
        )
        self.error: BaseException | None = None
        self.sndhwm = int(sndhwm)
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SNDHWM, self.sndhwm)
        self.sock.bind(bind)
        self.endpoint: str = self.sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self.seq = 0
        self.sent_samples = 0
        self.ended = False
        self._overflow_before = False
        self._stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name=f"sremu-rx-{handle.spec.stream}", daemon=True
        )

    def start(self) -> None:
        """Start the pump thread."""
        self.thread.start()

    def stop(self) -> None:
        """Stop the pump and close the socket."""
        self._stop.set()
        if self.thread.is_alive():
            self.thread.join(5.0)
        self.sock.close(0)

    @property
    def batching(self) -> bool:
        """Whether the pump waits for full frames (see the module docstring)."""
        return self._wait_rx is not None

    def _send(self, parts: list[bytes], give_up_s: float | None = None) -> bool:
        t0 = time.perf_counter()
        while not self._stop.is_set():
            try:
                self.sock.send_multipart(parts, zmq.NOBLOCK, copy=False)
                return True
            except zmq.Again:
                if self.paced:
                    return False
                if give_up_s is not None and time.perf_counter() - t0 >= give_up_s:
                    return False
                self.sock.poll(POLL_MS, zmq.POLLOUT)
            except zmq.ZMQError:  # pragma: no cover - socket closed under us
                return False
        return False

    def _frame(
        self,
        flags: int,
        first: int,
        y: NDArray[np.complex64] | None,
        give_up_s: float | None = None,
    ) -> bool:
        n = 0 if y is None else int(y.shape[0])
        hdr = FrameHeader(
            int(flags), self.handle.id, self.seq, int(first), n, Dtype.CF32_LE, 1, _now_ns()
        )
        ok = self._send(make_frame(hdr, y), give_up_s)
        if ok:
            self.seq = (self.seq + 1) & 0xFFFFFFFF
            self.counters.add("frames_out")
        return ok

    def _run(self) -> None:
        try:
            self._pump()
        except Exception as exc:  # the pump must not die silently (I6): count and end
            self.error = exc
            self.counters.add("internal_errors")
            self.ended = True

    def _pump(self) -> None:
        h = self.handle
        wait_rx = self._wait_rx
        while not self._stop.is_set():
            done = self.is_done()
            t0 = time.perf_counter()
            if wait_rx is not None and not done:  # batching: one wake-up per full frame
                # in slices of at most read_timeout_s, so that stop() or the end of the run
                # ends a long flush_s wait (neither wakes the engine's waiter; after the end
                # no batch can fill, and END_OF_STREAM must not wait for the flush timer)
                t_flush = t0 + self.flush_s
                while True:
                    left = min(self.read_timeout_s, t_flush - time.perf_counter())
                    if wait_rx(h, self.frame_samples, left) is not False:
                        break  # a full frame, or a read that cannot wait for more
                    if self._stop.is_set() or time.perf_counter() >= t_flush or self.is_done():
                        break
            # batching reads at once (also after the end of the run: nothing more comes)
            timeout_s = 0.0 if wait_rx is not None else self.read_timeout_s
            y, md = self.engine.read_rx(h, self.frame_samples, timeout_s)
            code = md.error_code
            if not y.size and code == ErrorCode.TIMEOUT and time.perf_counter() - t0 < 0.002:
                # the engine did not wait (e.g. the stream is not streaming): do not spin
                self._stop.wait(min(0.01, self.read_timeout_s))
            if code == ErrorCode.OVERFLOW:
                self._overflow_before = True
                if self.publish is not None:
                    self.publish(
                        "stream.overflow",
                        {"stream": h.spec.stream, "first_sample": md.first_sample},
                    )
                continue
            if code == ErrorCode.LATE_COMMAND:
                self._frame(Flag.LATE | Flag.HAS_TIME, self.frontier(), None)
                continue
            if not y.size:
                if done:  # best effort: the client may already have disconnected
                    self._frame(Flag.END_OF_STREAM | Flag.HAS_TIME, self.frontier(), None, 1.0)
                    self.ended = True
                    return
                continue
            first = md.first_sample
            if first is None:
                first = to_ticks(md.time_spec, h.spec.rate_sps)
            flags = Flag.HAS_TIME
            if md.start_of_burst:
                flags |= Flag.SOB
            if md.end_of_burst:
                flags |= Flag.EOB
            if self._overflow_before:
                flags |= Flag.OVERFLOW_BEFORE
            if self._frame(flags, first, y):
                self._overflow_before = False
                self.sent_samples += int(y.shape[0])
            elif not self._stop.is_set():  # paced: the client is too slow
                self.counters.add("rx_egress_dropped_samples", int(y.shape[0]))
                self._overflow_before = True
                if self.publish is not None:
                    self.publish(
                        "stream.overflow", {"stream": h.spec.stream, "first_sample": first}
                    )


# =========================================================================== client


class ReadyWatcher:
    """Calls listeners when a watched client socket may have become readable.

    ``add(fd)`` watches a socket's ``ZMQ_FD`` handle (read with ``getsockopt(zmq.FD)`` by
    the socket's owner thread) and returns the socket's *arm* function. libzmq makes that
    handle readable when a command reaches the socket's idle mailbox, and a message that
    arrives on a pipe the reader last found empty is such a command (``ZMQ_FD`` is
    edge-triggered; zmq_getsockopt(3)). So whenever a reader polled, found nothing and
    went to wait, the next arrival fires the handle.

    The thread only waits on the handles (``poll``, or ``select`` on Windows) and never
    calls into libzmq, so every socket stays owned by its reader thread (ZMQ sockets are
    not thread safe). A handle that fired is disarmed until its owner calls the arm
    function after its next socket operation (which also consumes the signal), so a
    socket nobody reads never makes the thread spin. It stays armed when its owner
    operated on the socket while the wait ran: that arm call came before the disarm, and
    disarming would lose the wake-up of a read that has just found the socket empty.
    Listeners run in the watcher thread, outside its lock; a spurious call (a command
    without a message, e.g. a connection event) costs the woken caller one retry. A
    listener that raises is counted (``internal_errors``) and the watcher goes on.

    ``remove(fd)`` and ``stop()`` return only once the thread no longer waits on the
    handle(s), so the caller may close the socket right after.
    """

    def __init__(self, counters: TransportCounters | None = None) -> None:
        self.counters = counters
        self._cv = threading.Condition()
        self._armed: dict[int, bool] = {}
        self._gen: dict[int, int] = {}  # arm calls per handle (socket operations)
        self._listeners: list[Callable[[], None]] = []
        self._r, self._w = socket.socketpair()
        self._r.setblocking(False)
        self._w.setblocking(False)
        self._stopping = False
        self._in_select = False
        self._iter = 0
        self.wakeups = 0  # handles that fired (diagnostics)
        self.thread = threading.Thread(target=self._run, name="sremu-client-wake", daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ registration
    def listen(self, fn: Callable[[], None]) -> None:
        """Call ``fn()`` whenever a watched handle fires (idempotent per ``fn``)."""
        with self._cv:
            if fn not in self._listeners:
                self._listeners.append(fn)

    def unlisten(self, fn: Callable[[], None]) -> None:
        """Stop calling ``fn``."""
        with self._cv:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def add(self, fd: int) -> Callable[[], None]:
        """Watch handle ``fd`` (armed) and return its arm function."""
        fd = int(fd)
        with self._cv:
            self._armed[fd] = True
            self._gen[fd] = 0
            self._poke()
        return lambda: self.arm(fd)

    def arm(self, fd: int) -> None:
        """The owner just operated on the socket: watch ``fd`` (again)."""
        with self._cv:
            if fd not in self._armed:
                return
            self._gen[fd] += 1
            if not self._armed[fd]:
                self._armed[fd] = True
                self._poke()

    def remove(self, fd: int) -> None:
        """Stop watching ``fd``; returns once no ``select`` includes it any more."""
        with self._cv:
            self._gen.pop(int(fd), None)
            if self._armed.pop(int(fd), None) is not None:
                self._leave_select()

    def stop(self) -> None:
        """Stop the thread (idempotent); returns once it selects no more."""
        with self._cv:
            if self._stopping:
                return
            self._stopping = True
            self._leave_select()
        # the thread may still be inside a listener (e.g. waiting for a lock our caller
        # holds); it exits at its next check without touching a handle again
        for s in (self._r, self._w):
            with contextlib.suppress(OSError):
                s.close()

    # ------------------------------------------------------------------ thread
    def _poke(self) -> None:
        with contextlib.suppress(OSError):  # BlockingIOError: a poke is pending already
            self._w.send(b"\0")

    def _leave_select(self) -> None:
        """(lock held) Wait until a ``select`` begun with the old handle set has returned."""
        if not self._in_select or threading.current_thread() is self.thread:
            return
        it0 = self._iter
        self._poke()
        self._cv.wait_for(lambda: self._iter != it0 or not self.thread.is_alive(), 2.0)

    @staticmethod
    def _wait(fds: list[int], timeout_s: float) -> tuple[list[int], list[int]]:
        """Readable and invalid handles among ``fds`` (``poll`` on POSIX: no FD_SETSIZE
        limit on handle values; ``select`` on Windows, where libzmq's handles are
        sockets)."""
        if sys.platform != "win32":
            p = select.poll()
            for fd in fds:
                p.register(fd, select.POLLIN)
            events = p.poll(timeout_s * 1000.0)
            bad_mask = select.POLLNVAL | select.POLLERR
            return [fd for fd, ev in events if not ev & bad_mask], [
                fd for fd, ev in events if ev & bad_mask
            ]
        try:
            ready, _, _ = select.select(fds, [], [], timeout_s)
            return list(ready), []
        except (OSError, ValueError):  # a handle closed without remove(): find it
            bad = []
            for fd in fds:
                try:
                    select.select([fd], [], [], 0)
                except (OSError, ValueError):
                    bad.append(fd)
            if not bad:
                raise
            return [], bad

    def _run(self) -> None:
        r = self._r
        rfd = r.fileno()
        while True:
            with self._cv:
                if self._stopping:
                    return
                gen0 = {fd: self._gen[fd] for fd, on in self._armed.items() if on}
                fds = list(gen0)
                self._in_select = True
            try:
                ready, bad = self._wait([rfd, *fds], WATCH_IDLE_S)
            except (OSError, ValueError):  # unexplained failure: stop (callers poll)
                ready, bad = [], [rfd]
            fired = [fd for fd in ready if fd != rfd]
            broken = rfd in bad  # the wake-up pair itself failed: give up (callers poll)
            bad = [fd for fd in bad if fd != rfd]
            if rfd in ready:
                with contextlib.suppress(OSError):
                    while r.recv(4096):
                        pass
            with self._cv:
                self._in_select = False
                self._iter += 1
                for fd in fired:
                    if fd in self._armed and self._gen[fd] == gen0[fd]:
                        self._armed[fd] = False  # until the owner's next socket operation
                for fd in bad:
                    self._armed.pop(fd, None)
                    self._gen.pop(fd, None)
                if broken:
                    self._stopping = True
                listeners = list(self._listeners) if fired else []
                self.wakeups += len(fired)
                self._cv.notify_all()
            if (bad or broken) and self.counters is not None:
                self.counters.add("internal_errors")
            for fn in listeners:
                try:
                    fn()
                except Exception:  # a listener must not end the wake-ups (I6)
                    if self.counters is not None:
                        self.counters.add("internal_errors")


class TxSender:
    """Client side of TX ingress: one ``PUSH`` socket for every TX stream of a client.

    ``send`` blocks while the server applies backpressure (high-water mark).
    """

    def __init__(self, ctx: zmq.Context[Any], endpoint: str, *, sndhwm: int = 64) -> None:
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.LINGER, 2000)
        self.sock.setsockopt(zmq.SNDHWM, int(sndhwm))
        self.sock.connect(endpoint)
        self.endpoint = endpoint
        self._seq: dict[int, int] = {}
        self._lock = threading.Lock()

    def _push(self, hdr: FrameHeader, x: NDArray[np.complex64] | None) -> None:
        self.sock.send_multipart(make_frame(hdr, x), copy=False)

    def next_seq(self, stream_id: int) -> int:
        """``seq`` of the next frame on ``stream_id`` (frames sent so far, modulo 2**32)."""
        with self._lock:
            return self._seq.get(int(stream_id), 0)

    def send(self, h: StreamHandle, samples: NDArray[np.complexfloating], md: TxMetadata) -> int:
        """Frame and send one ``submit_tx`` call; returns the number of samples sent."""
        x = np.ascontiguousarray(np.asarray(samples).reshape(-1), dtype=np.complex64)
        rate = h.spec.rate_sps
        with self._lock:
            seq = self._seq.get(h.id, 0)
            if md.idle_until:
                hdr = FrameHeader(
                    Flag.IDLE_UNTIL, h.id, seq, to_ticks(md.time_spec, rate), 0, wall_ns=_now_ns()
                )
                self._push(hdr, None)
                self._seq[h.id] = (seq + 1) & 0xFFFFFFFF
                return 0
            n = int(x.shape[0])
            starts = list(range(0, n, MAX_FRAME_SAMPLES)) or [0]
            for j, a in enumerate(starts):
                part = x[a : a + MAX_FRAME_SAMPLES]
                flags = 0
                first = 0
                if j == 0:
                    if md.has_time_spec:
                        flags |= Flag.HAS_TIME
                        first = to_ticks(md.time_spec, rate)
                    if md.start_of_burst:
                        flags |= Flag.SOB
                if j == len(starts) - 1:
                    if md.end_of_burst:
                        flags |= Flag.EOB
                    if md.end_of_stream:
                        flags |= Flag.END_OF_STREAM
                hdr = FrameHeader(
                    int(flags), h.id, seq, first, int(part.shape[0]), wall_ns=_now_ns()
                )
                self._push(hdr, part)
                seq = (seq + 1) & 0xFFFFFFFF
            self._seq[h.id] = seq
            return n

    def close(self) -> None:
        """Close the socket (pending frames get up to 2 s to leave)."""
        self.sock.close()


@dataclass(eq=False)
class _Chunk:
    k0: int
    data: NDArray[np.complex64] | None  # None: an error marker
    sob: bool = False
    eob: bool = False
    error: ErrorCode = ErrorCode.NONE
    k_after: int = 0


class RxReceiver:
    """Client side of one RX stream's egress: a ``PULL`` socket and the read semantics of
    the in-process engine (see the module docstring).

    ``fd``: the socket's ``ZMQ_FD`` handle for a :class:`ReadyWatcher`; ``on_read`` (the
    watcher's arm function, when watched) is called after every read's socket access.
    ``frame_samples``: the server's RX frame length (``stream.open``); the default
    ``rcvhwm`` holds :data:`RX_HWM_PAGES` pages in frames of that length."""

    def __init__(
        self,
        ctx: zmq.Context[Any],
        endpoint: str,
        handle: StreamHandle,
        *,
        rcvhwm: int | None = None,
        counters: TransportCounters | None = None,
        frame_samples: int | None = None,
    ) -> None:
        self.handle = handle
        self.frame_samples = int(frame_samples or handle.page_size)
        self.rcvhwm = int(
            rcvhwm
            if rcvhwm is not None
            else hwm_frames(RX_HWM_PAGES, handle.page_size, self.frame_samples)
        )
        self.sock = ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVHWM, self.rcvhwm)
        self.sock.connect(endpoint)
        self.endpoint = endpoint
        self.fd = int(self.sock.getsockopt(zmq.FD))  # read here, by the owner thread
        self.on_read: Callable[[], None] | None = None
        self.counters = counters or TransportCounters()
        self.queue: deque[_Chunk] = deque()
        self.queued = 0  # samples held in ``queue`` (data chunks only)
        self.ended = False
        self._seq: int | None = None
        self._out_of_sequence = False

    def _ingest(self, parts: list[Any]) -> None:
        try:
            fr = parse_frame(parts)
        except BadFrame as exc:
            self.counters.bad(exc.reason)
            return
        hdr = fr.header
        if hdr.stream_id != self.handle.id:
            self.counters.bad("stream_id")
            return
        if self._seq is not None and hdr.seq != self._seq:
            self.counters.add("seq_errors")
            self._out_of_sequence = True
        self._seq = (hdr.seq + 1) & 0xFFFFFFFF
        k = int(hdr.first_sample)
        if hdr.has(Flag.OVERFLOW_BEFORE):
            self.queue.append(_Chunk(k, None, error=ErrorCode.OVERFLOW, k_after=k))
        if hdr.has(Flag.LATE):
            self.queue.append(_Chunk(k, None, error=ErrorCode.LATE_COMMAND))
        if hdr.nsamps:
            self.queue.append(_Chunk(k, fr.samples, hdr.has(Flag.SOB), hdr.has(Flag.EOB)))
            self.queued += int(hdr.nsamps)
        if hdr.has(Flag.END_OF_STREAM):
            self.ended = True

    def _drain(self, n: int, timeout_s: float) -> None:
        """Move frames from the socket into ``queue`` until it holds ``n`` samples (or the
        socket is empty). Never more: frames left in the socket keep the server's egress at
        its high-water mark, so a slow reader still applies lockstep backpressure and the
        client's memory stays bounded by one read plus one frame."""
        wait_ms = max(0, round(float(timeout_s) * 1000))
        if not self.queue and not self.ended and not self.sock.poll(wait_ms, zmq.POLLIN):
            return
        while self.queued < max(1, int(n)):
            try:
                parts = self.sock.recv_multipart(zmq.NOBLOCK, copy=False)
            except zmq.Again:
                return
            self._ingest(parts)

    def read(self, n: int, timeout_s: float) -> tuple[NDArray[np.complex64], RxMetadata]:
        """Up to ``n`` samples of one contiguous run (``TIMEOUT`` when none arrived within
        ``timeout_s`` wall seconds)."""
        self._drain(n, timeout_s)
        if self.on_read is not None:
            self.on_read()  # re-arm the wake-up: this read consumed any pending signal
        md = RxMetadata()
        empty = np.zeros(0, dtype=np.complex64)
        rate = self.handle.spec.rate_sps
        if self._out_of_sequence:
            md.out_of_sequence = True
            self._out_of_sequence = False
        if not self.queue:
            md.error_code = ErrorCode.TIMEOUT
            return empty, md
        head = self.queue[0]
        if head.data is None:
            self.queue.popleft()
            md.error_code = head.error
            if head.error == ErrorCode.OVERFLOW:
                md.has_time_spec = True
                md.time_spec = TimeSpec.from_ticks(head.k_after, rate)
                md.first_sample = head.k_after
            return empty, md
        parts: list[NDArray[np.complex64]] = []
        got = 0
        first = head.k0
        expect = first
        md.start_of_burst = head.sob
        while self.queue and got < n:
            e = self.queue[0]
            if e.data is None or e.k0 != expect:
                break
            take = min(n - got, int(e.data.shape[0]))
            parts.append(e.data[:take])
            got += take
            expect += take
            if take == e.data.shape[0]:
                self.queue.popleft()
                if e.eob:
                    md.end_of_burst = True
                    break
            else:
                e.data = e.data[take:]
                e.k0 += take
                e.sob = False
                break
        self.queued -= got
        md.has_time_spec = True
        md.time_spec = TimeSpec.from_ticks(first, rate)
        md.first_sample = first
        out = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return np.ascontiguousarray(out, dtype=np.complex64), md

    def close(self) -> None:
        """Close the socket."""
        self.sock.close(0)
