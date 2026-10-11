"""Client side of the control socket and the native data plane (DESIGN.md sections 8.1-8.5).

* :class:`ClientSession`: one ``DEALER`` connection to the control ``ROUTER``. Requests are
  answered in order; unsolicited messages (``ctrl.barrier``, ``stream.async``) that arrive
  while a reply is awaited are queued for :meth:`ClientSession.next_event` and the
  per-stream async queues. Server messages are trusted (they may exceed the 1 MiB request
  cap, e.g. large observations), up to ``max_message_bytes``.
* :class:`RemoteEngine`: an :class:`~sremu.contracts.EngineProtocol` backed by a served
  emulator (``sremu serve``), so ``sremu.Device.connect("tcp://127.0.0.1:5600")`` behaves
  like an in-process device: ``submit_tx`` frames samples to the TX ingress (blocking while
  the server applies lockstep backpressure), ``read_rx`` reads the stream's RX egress,
  ``schedule`` sends commands and returns the engine's receipt, ``register_controller``
  runs an in-process ``Controller`` object as a remote controller in a background thread
  (:mod:`sremu.control.remote`).

  Differences from the in-process engine (documented, not hidden): a late timed TX burst
  is accepted locally and reported asynchronously (``TIME_ERROR`` through
  ``recv_async``), because the server decides lateness; ``read_rx`` waits up to its
  ``timeout_s`` for frames.

  **Wake-ups.** ``sremu.Device`` polls ``read_rx``/``recv_async`` with timeout 0 under its
  lock and waits on its condition between polls; nothing in the client process signals
  that condition when a frame arrives from the server, so without help every blocking
  ``recv`` that has to wait for the server sleeps a full ``DEFAULT_POLL_S`` (25 ms: the
  measured ~26 ms two-process round trip). :meth:`RemoteEngine.add_wake_listener`
  registers a callable (``Device.notify``) that a :class:`ReadyWatcher` thread calls as
  soon as an RX frame or an unsolicited control message (``stream.async``) may have
  arrived; the poll stays as the fallback.

:func:`connect_engine` is what ``Device.connect`` calls.
"""

from __future__ import annotations

import itertools
import re
import threading
from collections import deque
from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any, Final

import numpy as np
import zmq
from numpy.typing import NDArray

from sremu.contracts import (
    CONTRACT_VERSION,
    DEFAULT_CONTROL_ENDPOINT,
    AsyncMetadata,
    CommandReceipt,
    CommandRequest,
    Controller,
    RemoteControllerProxy,
    RisCapabilities,
    RxMetadata,
    StreamCmd,
    StreamHandle,
    StreamSpec,
    TxMetadata,
)
from sremu.errors import AsyncCode, ErrorCode, SremuError, SremuTimeout, StreamError
from sremu.timebase import TimeSpec, fraction_to_str, parse_exact
from sremu.transport.messages import (
    SCHEMA_VERSION,
    caps_from_json,
    command_to_request,
    decode_json,
    encode_json,
    receipt_from_reply,
)
from sremu.transport.zmq_native import ReadyWatcher, RxReceiver, TransportCounters, TxSender

__all__ = ["ClientSession", "RemoteEngine", "RemoteError", "connect_engine"]

CLIENT_MAX_MESSAGE_BYTES: Final[int] = 64 << 20
"""Largest server message a client accepts (server messages are trusted)."""
_ID_SAFE: Final = re.compile(r"[^A-Za-z0-9_.:\-]")


class RemoteError(SremuError):
    """An error reply; ``code`` is the reply's ``ErrorCode``."""

    def __init__(self, code: ErrorCode, msg: str) -> None:
        super().__init__(f"{code.name}: {msg}")
        self.code = code  # type: ignore[misc]
        self.msg = msg


def _safe_name(name: str) -> str:
    s = _ID_SAFE.sub("_", name)[:40]
    return s or "client"


class ClientSession:
    """A ``DEALER`` connection to a served emulator's control socket.

    ``name``: client name (``hello``; default ``src`` of commands and prefix of message
    ids). ``timeout_s``: default wall-clock wait for a reply (``SremuTimeout`` after it).
    ``hello``: send ``hello`` with this ``CONTRACT_VERSION`` on connect (``info`` keeps
    the reply).
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_CONTROL_ENDPOINT,
        *,
        name: str = "client",
        timeout_s: float = 10.0,
        ctx: zmq.Context[Any] | None = None,
        hello: bool = True,
        max_message_bytes: int = CLIENT_MAX_MESSAGE_BYTES,
    ) -> None:
        self.endpoint = endpoint
        self.name = _safe_name(name)
        self.timeout_s = float(timeout_s)
        self._own_ctx = ctx is None
        self.ctx: zmq.Context[Any] = ctx if ctx is not None else zmq.Context()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.LINGER, 200)  # let a last request leave; never hang on close
        self.sock.setsockopt(zmq.MAXMSGSIZE, int(max_message_bytes))
        self.sock.connect(endpoint)
        self.fd = int(self.sock.getsockopt(zmq.FD))  # for a ReadyWatcher (owner thread)
        self.on_read: Callable[[], None] | None = None  # the watcher's arm function
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self.events: deque[dict[str, Any]] = deque()
        self.async_events: dict[int, deque[AsyncMetadata]] = {}
        self.closed = False
        self.info: dict[str, Any] = {}
        if hello:
            self.info = self.call("hello", client=self.name, contract_version=CONTRACT_VERSION)

    # ------------------------------------------------------------------ plumbing
    def next_id(self) -> str:
        """A fresh message id (``<name>-<n>``)."""
        return f"{self.name}-{next(self._ids)}"

    def _dispatch(self, msg: Any) -> None:
        if not isinstance(msg, dict):
            return
        op = msg.get("op")
        if op == "stream.async":
            sid = int(msg.get("stream_id", -1))
            k = msg.get("sample_index")
            t = msg.get("t")
            md = AsyncMetadata(
                AsyncCode[str(msg["event_code"])],
                str(msg.get("stream", "")),
                0,
                t is not None,
                TimeSpec.from_exact(parse_exact(t)) if t is not None else TimeSpec(),
                None if k is None else int(k),
                int(msg.get("n_samples", 0)),
            )
            self.async_events.setdefault(sid, deque()).append(md)
        else:
            self.events.append(msg)

    def _recv(self, timeout_s: float) -> Any | None:
        try:
            if not self.sock.poll(max(0, round(timeout_s * 1000)), zmq.POLLIN):
                return None
            raw = self.sock.recv()
        finally:
            if self.on_read is not None:
                self.on_read()  # re-arm the wake-up (this access consumed any signal)
        try:
            return decode_json(raw)
        except (UnicodeDecodeError, ValueError):
            return None

    def send_raw(self, payload: bytes) -> None:
        """Send one raw message (tests; no reply is awaited)."""
        with self._lock:
            self.sock.send(payload)

    def request(self, msg: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        """Send an envelope (``v`` and ``id`` are filled in when missing) and return its
        reply (``SremuTimeout`` when none arrives in time)."""
        body = dict(msg)
        body.setdefault("v", SCHEMA_VERSION)
        body.setdefault("id", self.next_id())
        mid = body["id"]
        wait = self.timeout_s if timeout_s is None else float(timeout_s)
        with self._lock:
            if self.closed:
                raise StreamError("the session is closed")
            self.sock.send(encode_json(body))
            deadline = None if wait <= 0 else wait
            slice_s = 0.05
            waited = 0.0
            while True:
                reply = self._recv(slice_s)
                if reply is None:
                    waited += slice_s
                    if deadline is not None and waited >= deadline:
                        raise SremuTimeout(f"no reply to {body.get('op')} {mid!r} in {wait} s")
                    continue
                if isinstance(reply, dict) and "ok" in reply and reply.get("id") in (mid, None):
                    if reply.get("id") is None and reply.get("ok"):
                        continue
                    return reply
                self._dispatch(reply)

    def call(self, op: str, *, timeout_s: float | None = None, **fields: Any) -> dict[str, Any]:
        """``request`` and raise :class:`RemoteError` for an error reply."""
        reply = self.request({"op": op, **fields}, timeout_s=timeout_s)
        if not reply.get("ok"):
            err = dict(reply.get("error") or {})
            code = ErrorCode.__members__.get(str(err.get("code")), ErrorCode.NOT_PERMITTED)
            raise RemoteError(code, str(err.get("msg", "")))
        return reply

    def pump(self, timeout_s: float = 0.0) -> None:
        """Receive pending unsolicited messages (waiting up to ``timeout_s`` for one)."""
        with self._lock:
            msg = self._recv(timeout_s)
            while msg is not None:
                self._dispatch(msg)
                msg = self._recv(0.0)

    def next_event(self, op: str | None = None, timeout_s: float = 0.0) -> dict[str, Any] | None:
        """The next unsolicited message (with ``op`` when given), waiting up to
        ``timeout_s`` wall seconds; None when there is none."""
        waited = 0.0
        while True:
            for i, ev in enumerate(self.events):
                if op is None or ev.get("op") == op:
                    del self.events[i]
                    return ev
            if waited >= timeout_s:
                return None
            step = min(0.05, max(0.0, timeout_s - waited))
            self.pump(step)
            waited += step if step > 0 else 0.05

    def close(self) -> None:
        """Close the socket (and the context when the session created it)."""
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self.sock.close()
            if self._own_ctx:
                self.ctx.term()


class RemoteEngine:
    """``EngineProtocol`` over a served emulator (see the module docstring).

    ``endpoint``: control endpoint. ``name``: client name and default command source.
    ``timeout_s``: reply timeout of control requests (wall seconds).
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_CONTROL_ENDPOINT,
        *,
        name: str = "api",
        timeout_s: float = 10.0,
    ) -> None:
        self.ctx: zmq.Context[Any] = zmq.Context()
        self.session = ClientSession(endpoint, name=name, timeout_s=timeout_s, ctx=self.ctx)
        self.info = self.session.info
        self.counters = TransportCounters()
        self._tx: TxSender | None = None
        self._rx: dict[int, RxReceiver] = {}
        self._handles: dict[int, StreamHandle] = {}
        self._caps: dict[str, RisCapabilities] | None = None
        self._runners: list[Any] = []
        self._watcher: ReadyWatcher | None = None
        self._watch_lock = threading.Lock()
        self.closed = False

    # ------------------------------------------------------------------ wake-ups
    def add_wake_listener(self, fn: Callable[[], None]) -> None:
        """Call ``fn()`` (from a background thread) whenever an RX frame of an open RX
        stream or an unsolicited control message (``stream.async``) may have arrived.

        ``sremu.Device`` registers its ``notify``, so a blocked ``recv`` or
        ``recv_async_msg`` retries as soon as the server's data is here instead of at its
        next poll. Calls can be spurious (connection events); ``fn`` must be cheap and must
        not call back into this engine (it runs in the watcher thread)."""
        with self._watch_lock:
            if self.closed:
                raise StreamError("the engine is closed")
            if self._watcher is None:
                w = ReadyWatcher(self.counters)
                self.session.on_read = w.add(self.session.fd)
                for rx in self._rx.values():
                    rx.on_read = w.add(rx.fd)
                self._watcher = w
            self._watcher.listen(fn)

    def remove_wake_listener(self, fn: Callable[[], None]) -> None:
        """Stop calling ``fn`` (the watcher thread keeps running until :meth:`close`)."""
        with self._watch_lock:
            if self._watcher is not None:
                self._watcher.unlisten(fn)

    # ------------------------------------------------------------------ streams
    def open_stream(self, spec: StreamSpec) -> StreamHandle:
        """``stream.open``; RX streams connect their egress socket."""
        r = self._call(
            "stream.open",
            stream=spec.stream,
            direction=spec.direction,
            fc_hz=float(spec.fc_hz),
            rate_sps=fraction_to_str(spec.rate_sps),
            cpu=spec.cpu,
            nchan=spec.nchan,
        )
        h = StreamHandle(int(r["stream_id"]), spec, int(r["page_size"]))
        self._handles[h.id] = h
        if spec.direction == "tx":
            if self._tx is None:
                self._tx = TxSender(self.ctx, str(r["endpoint"]))
        else:
            # the server's RX frame length sizes the receive high-water mark (in frames)
            fs = r.get("frame_samples")
            rx = RxReceiver(
                self.ctx,
                str(r["endpoint"]),
                h,
                counters=self.counters,
                frame_samples=int(fs) if fs else None,
            )
            with self._watch_lock:
                if self._watcher is not None:
                    rx.on_read = self._watcher.add(rx.fd)
                self._rx[h.id] = rx
        return h

    def _handle(self, h: StreamHandle, direction: str) -> StreamHandle:
        if self._handles.get(h.id) is None:
            raise StreamError(f"{h.spec.stream!r} is not open")
        if h.spec.direction != direction:
            raise StreamError(f"{h.spec.stream!r} is not a {direction.upper()} stream")
        return h

    def submit_tx(
        self, h: StreamHandle, samples: NDArray[np.complexfloating], md: TxMetadata
    ) -> int:
        """Frame and send TX samples (blocks under lockstep backpressure)."""
        self._handle(h, "tx")
        assert self._tx is not None
        return self._tx.send(h, samples, md)

    def stream_cmd(self, h: StreamHandle, cmd: StreamCmd) -> None:
        """``stream.cmd``."""
        self._handle(h, "rx")
        fields: dict[str, Any] = {
            "stream_id": h.id,
            "mode": cmd.mode,
            "num_samps": int(cmd.num_samps),
            "stream_now": bool(cmd.stream_now),
        }
        if not cmd.stream_now:
            fields["t"] = fraction_to_str(cmd.time_spec.exact())
        self._call("stream.cmd", **fields)

    def read_rx(
        self, h: StreamHandle, n: int, timeout_s: float
    ) -> tuple[NDArray[np.complex64], RxMetadata]:
        """Up to ``n`` RX samples from the stream's egress (``TIMEOUT`` when none)."""
        self._handle(h, "rx")
        return self._rx[h.id].read(int(n), float(timeout_s))

    def recv_async(self, h: StreamHandle, timeout_s: float) -> AsyncMetadata | None:
        """Next asynchronous TX event of ``h`` from the server, or None."""
        self._handle(h, "tx")
        q = self.session.async_events.get(h.id)
        if not q:
            self.session.pump(max(0.0, float(timeout_s)))
            q = self.session.async_events.get(h.id)
        return q.popleft() if q else None

    def close_stream(self, h: StreamHandle, *, timeout_s: float | None = None) -> None:
        """``stream.close``. A TX stream first sends ``END_OF_STREAM`` in-band (ordered after
        every sample frame on the data plane) and waits until TX ingress has taken all of
        its frames (``SremuTimeout`` after ``timeout_s``, default the session timeout), so
        the close cannot overtake samples still in flight."""
        if h.spec.direction == "tx" and self._tx is not None and h.id in self._handles:
            self._tx.send(h, np.zeros(0, np.complex64), TxMetadata(end_of_stream=True))
            target = self._tx.next_seq(h.id)
            wait = self.session.timeout_s if timeout_s is None else float(timeout_s)
            waited = 0.0
            while True:
                seen = self._call("stats.get").get("tx_next_seq", {}).get(str(h.id))
                if seen is not None and int(seen) == target:
                    break
                if waited >= wait:
                    raise SremuTimeout(
                        f"TX ingress took {seen} of {target} frames of {h.spec.stream!r}"
                    )
                self.session.pump(0.01)  # keeps async events flowing while waiting
                waited += 0.01
        self._call("stream.close", stream_id=h.id)
        with self._watch_lock:
            rx = self._rx.pop(h.id, None)
            if rx is not None and self._watcher is not None:
                self._watcher.remove(rx.fd)  # before the socket (and its handle) closes
        if rx is not None:
            rx.close()
        self._handles.pop(h.id, None)

    # ------------------------------------------------------------------ commands
    def schedule(self, req: CommandRequest) -> CommandReceipt:
        """Send a command; the reply is the engine's receipt."""
        reply = self.session.request(command_to_request(req))
        return receipt_from_reply(reply)

    def register_controller(self, ctrl: Controller | RemoteControllerProxy) -> None:
        """Run ``ctrl`` (an in-process ``Controller`` object) as a remote controller in a
        background thread with its own connection (:mod:`sremu.control.remote`)."""
        from sremu.control.remote import RemoteControllerRunner

        if not isinstance(ctrl, Controller):
            raise StreamError("register_controller over a session needs a Controller object")
        runner = RemoteControllerRunner(ctrl, self.session.endpoint)
        runner.start()
        self._runners.append(runner)

    # ------------------------------------------------------------------ queries
    def time_now(self) -> Fraction:
        """Virtual time through which every RX stream has been computed."""
        return parse_exact(self._call("time.now")["t"])

    def frontier(self) -> dict[str, int]:
        """First uncomputed sample of every RX stream."""
        return {str(k): int(v) for k, v in self._call("time.now")["frontier"].items()}

    def caps(self) -> Mapping[str, RisCapabilities]:
        """Panel capabilities (cached after the first call)."""
        if self._caps is None:
            self._caps = caps_from_json(self._call("caps.get")["caps"])
        return self._caps

    def stats(self) -> dict[str, Any]:
        """``stats.get`` (engine and transport counters)."""
        return self._call("stats.get")

    def pause(self) -> None:
        """``sim.pause``."""
        self._call("sim.pause")

    def resume(self) -> None:
        """``sim.resume`` (start virtual time of a server started paused)."""
        self._call("sim.resume")

    def shutdown(self) -> None:
        """``sim.shutdown``: ask the server to stop."""
        self._call("sim.shutdown")

    def _call(self, op: str, **fields: Any) -> dict[str, Any]:
        return self.session.call(op, **fields)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Stop controller threads and close every socket (idempotent)."""
        if self.closed:
            return
        with self._watch_lock:
            self.closed = True
            if self._watcher is not None:
                self._watcher.stop()  # selects on no handle once this returns
        for r in self._runners:
            r.stop()
        for rx in self._rx.values():
            rx.close()
        if self._tx is not None:
            self._tx.close()
        self.session.close()
        self.ctx.term()


def connect_engine(endpoint: str = DEFAULT_CONTROL_ENDPOINT, **kw: Any) -> RemoteEngine:
    """A :class:`RemoteEngine` connected to a served emulator (``Device.connect``)."""
    return RemoteEngine(endpoint, **kw)
