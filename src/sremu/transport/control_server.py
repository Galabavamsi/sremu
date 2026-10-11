"""The control socket and the served emulator (DESIGN.md sections 4.8, 4.9, 8.3-8.5, 8.10).

:class:`ControlServer` serves any :class:`~sremu.contracts.EngineProtocol` (the WP6
``Emulator`` or a test double) to other processes:

* **Control** ``ROUTER`` at ``tcp://127.0.0.1:5600``: JSON requests (schema version 1,
  validated before dispatch by :func:`sremu.transport.messages.parse_request`), one reply
  per request. Registered remote controllers use ``DEALER`` and receive ``ctrl.barrier``
  on the same connection (never on PUB, which may drop).
* **Telemetry** ``PUB`` at ``:5601``: ``ris.applied``, ``cmd.late``, ``cmd.rejected``,
  ``stream.underflow``, ``stream.overflow``, ``metrics.window``, ``sim.stall``.
* **Data plane** (:mod:`sremu.transport.zmq_native`): TX ingress ``PULL`` at ``:5602``,
  one RX egress ``PUSH`` per opened RX stream; raw REQ/REP peers
  (:mod:`sremu.transport.rawreqrep`) when configured.

**Security (8.10).** Every socket binds to a loopback address; any other bind address
(``tcp://*``, ``0.0.0.0``, a LAN address, an interface name) raises ``NotPermitted`` unless
``allow_remote=True`` (``--allow-remote``). There is no authentication. Requests larger than
1 MiB or carrying more than 65,536 elements get ``TOO_LARGE``; malformed or schema-invalid
requests get ``SCHEMA_ERROR``; no request field is a filesystem path (trajectory ``trace``
cannot be sent); commands above ``max_command_rate_hz`` per client (token bucket, burst
``max(1, rate*rate_burst_s)``) are refused with ``NOT_PERMITTED`` and counted as
``rate_limited_commands``. Every refusal is counted (I6) and nothing a client sends can
raise inside the server.

**Remote controllers (4.9).** ``ctrl.register`` creates an engine-side
:class:`RemoteProxy` (``RemoteControllerProxy``) registered with the engine; scenario
controllers of kind ``external`` are registered before the run starts, so their barriers
start at ``t_b = 0`` and the remote side only attaches by name (parameters must match).
At each barrier the engine sends ``ctrl.barrier {t_b, k_b, m, observation}`` and waits; the
controller sends zero or more ``ris.set_config`` / ``ris.set_enabled`` (``src`` = its name)
and then ``ctrl.ack {t_b}``. Requests of one connection are handled in order, so every
command precedes its ack. A barrier that waits longer than ``stall_timeout_s`` publishes
one ``sim.stall``; the wait ends with the ack, ``ctrl.unregister`` or shutdown, or after
``ack_timeout_s`` (when set; the engine then raises ``SremuTimeout`` and the server reports
the engine error). A registered controller's name can be used as ``src`` only from the
connection that registered it, and an in-process barrier controller's name never (the engine
trusts barrier controllers' commands: lookahead bound instead of the nondeterministic-input
flag).

**Run control.** ``start_paused=True`` (``sremu serve`` default) holds virtual time at 0
until ``sim.resume``, so clients can open streams and issue timed stream commands first;
TX streams that no client has opened yet are not committed (``unopened_tx="wait"``, the
served default), so the engine waits for their transmitters instead of computing silence.
``sim.shutdown`` stops the server after its reply.

**Engine driver.** With the WP6 emulator in lockstep the server uses ``Emulator.start()``
(its served loop, RX backpressure and stall reports) and wraps the instance's ``step`` to
implement pause, to hold virtual time while nothing consumes it (no RX stream streaming,
no barrier controller, no RX recording; advancing would only skip pages and make later
timed RX starts late; with ``exit_when_done`` it never holds once a consumer has attached,
so the run reaches its end and the server exits) and to capture an engine exception
(reported as ``sim.stall`` with an ``error`` and in ``stats.get``) instead of losing it in
a thread. Paced mode runs ``run_paced`` in a driver thread. Engines without a ``start``
method (e.g. the synchronous ``FakeEngine``) compute on demand.
"""

from __future__ import annotations

import contextlib
import functools
import ipaddress
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Final
from urllib.parse import urlsplit

import numpy as np
import zmq

from sremu import __version__ as SREMU_VERSION
from sremu.contracts import (
    CONTRACT_VERSION,
    DEFAULT_BIND_HOST,
    DEFAULT_CONTROL_ENDPOINT,
    DEFAULT_MAX_COMMAND_RATE_HZ,
    DEFAULT_PUB_ENDPOINT,
    DEFAULT_TX_INGRESS_ENDPOINT,
    MAX_ELEMENTS_PER_COMMAND,
    MAX_FRAME_SAMPLES,
    MAX_MESSAGE_BYTES,
    AsyncMetadata,
    CommandRecord,
    EngineProtocol,
    LatencyModel,
    Observation,
    ObservationSpec,
    PanelConfig,
    StreamCmd,
    StreamHandle,
    StreamSpec,
    config_hash,
)
from sremu.engine.emulator import EngineSink
from sremu.errors import ErrorCode, NotPermitted, SremuError
from sremu.timebase import TimeSpec, fraction_to_str, to_ticks
from sremu.transport.messages import (
    SCHEMA_VERSION,
    ProtocolError,
    b64_of,
    caps_to_json,
    command_from_request,
    encode_json,
    error_reply,
    exact_from_json,
    num_to_json,
    observation_to_json,
    parse_request,
    receipt_reply,
)
from sremu.transport.zmq_native import (
    RX_FLUSH_S,
    RX_HWM_PAGES,
    RxEgress,
    TransportCounters,
    TxIngress,
    hwm_frames,
)

__all__ = [
    "COMMAND_OPS",
    "HARD_MESSAGE_LIMIT",
    "OPS",
    "RX_BATCH_PAGES",
    "ControlServer",
    "RemoteProxy",
    "ServerConfig",
    "check_bind_endpoint",
    "endpoints_summary",
    "is_loopback_endpoint",
    "serve_engine",
]

OPS: Final[tuple[str, ...]] = (
    "hello",
    "caps.get",
    "time.now",
    "ris.set_config",
    "ris.set_codebook",
    "ris.get_state",
    "ris.set_enabled",
    "node.set_position",
    "node.set_trajectory",
    "stream.open",
    "stream.close",
    "stream.cmd",
    "ctrl.register",
    "ctrl.ack",
    "ctrl.unregister",
    "rec.marker",
    "stats.get",
    "sim.pause",
    "sim.resume",
    "sim.shutdown",
)
"""Every control operation (section 8.5 table)."""
COMMAND_OPS: Final[frozenset[str]] = frozenset(
    {"ris.set_config", "ris.set_enabled", "node.set_position", "node.set_trajectory", "rec.marker"}
)
"""Operations that become engine commands (rate limited)."""

_LOOPBACK_NAMES: Final = frozenset({"localhost"})
HARD_MESSAGE_LIMIT: Final[int] = 16 << 20
"""Messages above 16 MiB are dropped by ZMQ (connection closed) without a reply."""
RX_BATCH_PAGES: Final[int] = 8
"""Default RX frame length in lockstep, in pages (paced: one page, since a batch would add
its length to the delivery latency). One frame per page made the egress pump wake, take the
engine lock and send once per page, each a GIL hand-off with the engine thread; 8 pages per
frame cut the frames 8x (served B2 base point, end-to-end RTF x1.28 on a Windows laptop and
x1.40 under WSL2, paired runs) while frames stay small (32 KiB at 1 MS/s, 512 KiB at
23.04 MS/s with 8192-sample pages)."""


# =========================================================================== endpoints


def is_loopback_endpoint(endpoint: str) -> bool:
    """Whether binding ``endpoint`` exposes nothing beyond this host (loopback TCP,
    ``ipc://`` or ``inproc://``)."""
    scheme, _, rest = endpoint.partition("://")
    if scheme in ("inproc", "ipc"):
        return True
    if scheme != "tcp" or not rest:
        return False
    host = urlsplit(f"tcp://{rest}").hostname
    if host is None:
        return False
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # '*', interface names, DNS names


def check_bind_endpoint(endpoint: str, allow_remote: bool) -> None:
    """Refuse a non-loopback bind unless ``allow_remote`` (section 8.10, IO-04)."""
    if not allow_remote and not is_loopback_endpoint(endpoint):
        raise NotPermitted(
            f"refusing to bind {endpoint!r}: only loopback addresses without --allow-remote "
            "(there is no authentication; see SECURITY.md)"
        )


def _bind_host(endpoint: str) -> str:
    """Host part for the ephemeral RX egress binds (loopback for non-TCP endpoints)."""
    scheme, _, rest = endpoint.partition("://")
    if scheme != "tcp":
        return DEFAULT_BIND_HOST
    host = urlsplit(f"tcp://{rest}").hostname or DEFAULT_BIND_HOST
    return f"[{host}]" if ":" in host else host


# =========================================================================== config


@dataclass(frozen=True)
class ServerConfig:
    """Settings of a :class:`ControlServer` (defaults: section 8.5).

    ``pub_endpoint`` None disables telemetry. Ports ``0`` or ``*`` bind ephemeral ports
    (read them from :attr:`ControlServer.endpoints`). ``tx_window_samples``: how far an
    open TX stream may run ahead of its receivers before TX ingress pauses (lockstep
    backpressure). ``rx_frame_samples``: samples per RX frame at most; default (None) the
    stream's page size in paced mode and :data:`RX_BATCH_PAGES` pages in lockstep, where
    the egress pump batches pages into one frame (:mod:`sremu.transport.zmq_native`:
    flushed early when the engine blocks, after ``rx_flush_s`` otherwise; the same samples
    and flags). ``rx_sndhwm``: RX egress high-water mark in pages (converted to frames, so
    about the same samples are in flight for any frame length). ``ack_timeout_s``: None
    waits for remote barrier acks indefinitely (publishing
    ``sim.stall``); a number ends the wait with an engine error. ``exit_when_done``: stop
    the server once the run has ended and every RX egress has sent ``END_OF_STREAM``; once
    a consumer has attached (RX stream, raw peer, remote controller) the run goes on to its
    end even after every consumer has left.
    """

    control_endpoint: str = DEFAULT_CONTROL_ENDPOINT
    pub_endpoint: str | None = DEFAULT_PUB_ENDPOINT
    tx_endpoint: str = DEFAULT_TX_INGRESS_ENDPOINT
    allow_remote: bool = False
    max_message_bytes: int = MAX_MESSAGE_BYTES
    max_command_rate_hz: float = DEFAULT_MAX_COMMAND_RATE_HZ
    rate_burst_s: float = 0.01
    start_paused: bool = False
    ack_timeout_s: float | None = None
    tx_window_samples: int = 1 << 21
    rx_frame_samples: int | None = None
    rx_flush_s: float = RX_FLUSH_S
    rx_sndhwm: int = RX_HWM_PAGES
    tx_rcvhwm: int = 64
    exit_when_done: bool = False
    raw_peers: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.max_message_bytes <= MAX_MESSAGE_BYTES:
            raise NotPermitted(f"max_message_bytes must be in (0, {MAX_MESSAGE_BYTES}]")
        if not self.max_command_rate_hz > 0:
            raise NotPermitted("max_command_rate_hz must be > 0")


def _ephemeral(endpoint: str) -> str:
    return (
        endpoint[:-2] + ":*"
        if endpoint.startswith("tcp://") and endpoint.endswith(":0")
        else (endpoint)
    )


# =========================================================================== helpers


class _TokenBucket:
    """Per-client command rate limit (wall clock; section 8.5)."""

    def __init__(self, rate_hz: float, burst: int) -> None:
        self.rate = float(rate_hz)
        self.capacity = float(max(1, burst))
        self.tokens = self.capacity
        self.t = time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


@dataclass(eq=False)
class _Session:
    ident: bytes
    delimited: bool
    name: str = "client"
    bucket: _TokenBucket | None = None
    controllers: set[str] = field(default_factory=set)
    streams: set[int] = field(default_factory=set)


class _PubSink(EngineSink):
    """Engine sink publishing records, metrics windows and stall reports (never blocks).

    It does not override ``on_rx``/``on_tx``, so the engine keeps skipping pages nobody
    consumes (D-6)."""

    def __init__(self, server: ControlServer) -> None:
        self.server = server

    def on_record(self, rec: CommandRecord, cfg: PanelConfig | None) -> None:
        body = {"record": rec.to_dict()}
        if rec.status == "rejected":
            self.server.publish("cmd.rejected", body)
            return
        self.server.publish("ris.applied", body)
        if rec.status == "late_applied":
            self.server.publish("cmd.late", body)

    def on_metrics(self, window: Mapping[str, Any]) -> None:
        self.server.publish("metrics.window", {"window": dict(window)})

    def on_stall(self, report: Any) -> None:
        to_dict = getattr(report, "to_dict", None)
        self.server.publish("sim.stall", {"report": to_dict() if callable(to_dict) else {}})


# =========================================================================== remote proxy


class RemoteProxy:
    """Engine-side stand-in for a remote controller (``RemoteControllerProxy``, 8.5).

    ``send_barrier`` runs in the engine thread and only queues the ``ctrl.barrier`` message
    for the controller's connection (or keeps it until the controller attaches);
    ``wait_ack`` blocks the engine (whose lock is released meanwhile) until ``ctrl.ack
    {t_b}`` (see the module docstring for stall reports, timeouts and release).
    """

    def __init__(
        self,
        server: ControlServer,
        name: str,
        period_s: Fraction,
        min_latency_s: Fraction,
        spec: ObservationSpec,
        latency: LatencyModel,
        *,
        preregistered: bool = False,
    ) -> None:
        self.server = server
        self.name = name
        self.period_s = Fraction(period_s)
        self.min_latency_s = Fraction(min_latency_s)
        self.spec = spec
        self.latency = latency
        self.preregistered = preregistered
        self.ident: bytes | None = None
        self.delimited = False
        self._cv = threading.Condition()
        self._pending: Fraction | None = None
        self._outbox: bytes | None = None
        self._acked: Fraction | None = None
        self._released = False
        self.barriers_sent = 0
        self.acks = 0

    # ------------------------------------------------------------------ protocol
    def send_barrier(self, obs: Observation) -> None:
        """Queue ``ctrl.barrier`` for this controller (engine thread)."""
        m = int(obs.t_b / self.period_s) if self.period_s else 0
        payload = encode_json(
            {
                "v": SCHEMA_VERSION,
                "op": "ctrl.barrier",
                "controller": self.name,
                "t_b": fraction_to_str(obs.t_b),
                "k_b": {r: int(k) for r, k in obs.k_b.items()},  # section 4.9 {t_b, k_b, ...}
                "m": m,
                "observation": observation_to_json(obs),
            }
        )
        with self._cv:
            self._pending = obs.t_b
            self._acked = None
            self._outbox = payload
            self.barriers_sent += 1
            ident, delim = self.ident, self.delimited
        if ident is not None:
            self.server.route(ident, delim, payload)

    def wait_ack(self, t_b: Fraction, timeout_s: float) -> bool:
        """Block until ``ctrl.ack {t_b}`` (True), release (True), shutdown (False) or
        ``ack_timeout_s`` after the controller attached (False). ``timeout_s`` (the engine's
        ``stall_timeout_s``) sets when one ``sim.stall`` report is published."""
        stall_s = float(timeout_s) if timeout_s and timeout_s > 0 else 2.0
        ack_timeout = self.server.config.ack_timeout_s
        t0 = time.monotonic()
        t_att: float | None = None
        stalled = False
        with self._cv:
            while True:
                if self._released or self._acked == t_b:
                    self._pending = None
                    self._outbox = None
                    return True
                if self.server.stopping:
                    return False
                now = time.monotonic()
                if self.ident is not None and t_att is None:
                    t_att = now
                if ack_timeout is not None and t_att is not None and now - t_att >= ack_timeout:
                    return False
                if not stalled and now - t0 >= stall_s:
                    stalled = True
                    self.server.publish(
                        "sim.stall",
                        {
                            "report": {
                                "pending_barrier": {
                                    "controller": self.name,
                                    "t_b": fraction_to_str(t_b),
                                    "attached": self.ident is not None,
                                    "waited_s": now - t0,
                                }
                            }
                        },
                    )
                self._cv.wait(0.1)

    # ------------------------------------------------------------------ server side
    def attach(self, ident: bytes, delimited: bool) -> None:
        """Bind the controller's connection; a queued barrier is delivered now."""
        with self._cv:
            self.ident = ident
            self.delimited = delimited
            out = self._outbox if self._pending is not None else None
        if out is not None:
            self.server.route(ident, delimited, out)

    def ack(self, t_b: Fraction) -> bool:
        """Record ``ctrl.ack {t_b}``; False when no barrier at ``t_b`` is pending."""
        with self._cv:
            if self._pending is None or self._pending != t_b:
                return False
            self._acked = t_b
            self.acks += 1
            self._cv.notify_all()
            return True

    def release(self) -> None:
        """End any wait (unregister or shutdown)."""
        with self._cv:
            self._released = True
            self._cv.notify_all()

    def wake(self) -> None:
        """Wake a waiting engine (e.g. on shutdown)."""
        with self._cv:
            self._cv.notify_all()

    @property
    def pending(self) -> Fraction | None:
        """``t_b`` of the barrier waiting for its ack, or None."""
        with self._cv:
            return self._pending


def _json_safe(x: Any) -> Any:
    """``x`` with every non-finite float replaced by its string form (``num_to_json``)."""
    if isinstance(x, float):
        return num_to_json(x)
    if isinstance(x, Mapping):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_safe(v) for v in x]
    return x


def _spec_from_json(d: Mapping[str, Any] | None) -> ObservationSpec:
    d = dict(d or {})
    kw: dict[str, Any] = {}
    if "fields" in d:
        kw["fields"] = tuple(d["fields"])
    for k in ("window_s", "position_delay_s"):
        if k in d:
            kw[k] = exact_from_json(d[k], k)
    if "position_noise_m" in d:
        kw["position_noise_m"] = float(d["position_noise_m"])
    return ObservationSpec(**kw)


def _latency_from_json(d: Mapping[str, Any] | None, l_min: Fraction) -> LatencyModel:
    if d is None:
        return LatencyModel(l_min)
    return LatencyModel(exact_from_json(d["fixed_s"], "latency.fixed_s"), d.get("jitter"))


def _same_controller(
    p: RemoteProxy, period: Fraction, lmin: Fraction, lat: LatencyModel, spec: ObservationSpec
) -> bool:
    return (
        p.period_s == period
        and p.min_latency_s == lmin
        and p.latency.fixed_s == lat.fixed_s
        and (p.latency.jitter or None) == (lat.jitter or None)
        and tuple(p.spec.fields) == tuple(spec.fields)
        and p.spec.window_s == spec.window_s
        and p.spec.position_delay_s == spec.position_delay_s
        and p.spec.position_noise_m == spec.position_noise_m
    )


# =========================================================================== server


class ControlServer:
    """Serve ``engine`` over the control socket and the native data plane (module
    docstring). ``scenario``: the scenario mapping (``model_dump``) when the engine was
    built from one; it provides external controllers and the external-node check of
    ``node.*`` commands (default: ``engine.scenario`` when present).
    ``builtin_factory(spec)``: builds the scenario's ``builtin`` controllers, registered
    in scenario order with the external ones (``sremu serve`` passes WP9's
    ``build_controller``). ``overrides``: :class:`ServerConfig` fields."""

    def __init__(
        self,
        engine: EngineProtocol,
        config: ServerConfig | None = None,
        *,
        scenario: Mapping[str, Any] | None = None,
        builtin_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        **overrides: Any,
    ) -> None:
        cfg = config or ServerConfig()
        if overrides:
            cfg = ServerConfig(**{**cfg.__dict__, **overrides})
        self.config = cfg
        self.engine = engine
        # t_wall_recv_ns is compared with the engine's t_wall_computed/visible_ns, so it is
        # read from the engine's own wall clock (SystemClock: perf_counter_ns; a test's
        # FakeClock). time.monotonic_ns is a different clock: GetTickCount64 on Windows,
        # 15.6 ms ticks and an offset of tens of ms from perf_counter.
        now_ns = getattr(getattr(engine, "clock", None), "now_ns", None)
        self._now_ns: Callable[[], int] = now_ns if callable(now_ns) else time.perf_counter_ns
        self.scenario = scenario if scenario is not None else getattr(engine, "scenario", None)
        self.builtin_factory = builtin_factory
        self.counters = TransportCounters()
        self.ctx: zmq.Context[Any] | None = None
        self.endpoints: dict[str, str | None] = {}
        self.stopping = False
        self.engine_error: BaseException | None = None
        self._out: deque[tuple[str, bytes, bool, bytes]] = deque()
        self._out_lock = threading.Lock()
        self._wake_lock = threading.Lock()
        self._wake_push: zmq.Socket[Any] | None = None
        self._sessions: dict[bytes, _Session] = {}
        self._handles: dict[int, StreamHandle] = {}
        self._owner: dict[int, bytes] = {}
        self._egress: dict[int, RxEgress] = {}
        self._handles_lock = threading.Lock()
        self.controllers: dict[str, RemoteProxy] = {}
        self._paused = threading.Event()
        self._resume_cv = threading.Condition()
        self._was_paused = False
        self._shutdown = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._driver: threading.Thread | None = None
        self._ingress: TxIngress | None = None
        self._raw: list[Any] = []
        # sticky: some consumer (RX egress, raw peer, remote controller) attached once;
        # with exit_when_done the run then goes on to its end (see _engine_idle)
        self._consumer_attached = False
        self._started = False
        self._router: Any = None
        self._pub: Any = None
        self._wake_pull: Any = None
        self.requests = 0
        if cfg.start_paused:
            self._paused.set()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> ControlServer:
        """Bind every socket, register scenario controllers, start the threads. On any
        failure everything created so far is closed again before the error propagates."""
        if self._started:
            raise NotPermitted("the server is already started")
        cfg = self.config
        for ep in (cfg.control_endpoint, cfg.pub_endpoint, cfg.tx_endpoint):
            if ep is not None:
                check_bind_endpoint(ep, cfg.allow_remote)
        for peer in cfg.raw_peers:
            rx_port = getattr(peer, "rx_port", None)
            if rx_port:
                check_bind_endpoint(rx_port, cfg.allow_remote)
        if cfg.raw_peers and getattr(getattr(self.engine, "cfg", None), "mode", "") == "paced":
            raise NotPermitted("raw REQ/REP peers need lockstep (timing is implicit in counts)")
        self.ctx = zmq.Context()
        try:
            self._start_sockets_and_threads()
        except BaseException:
            self.stop()
            raise
        self._started = True
        return self

    def _start_sockets_and_threads(self) -> None:
        cfg = self.config
        assert self.ctx is not None
        ctx = self.ctx
        self._router = ctx.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, 0)
        # transport backstop: larger messages drop the connection; up to it they get a
        # counted TOO_LARGE reply (the 1 MiB cap is enforced by parse_request)
        self._router.setsockopt(zmq.MAXMSGSIZE, HARD_MESSAGE_LIMIT)
        self._router.bind(_ephemeral(cfg.control_endpoint))
        if cfg.pub_endpoint is not None:
            self._pub = ctx.socket(zmq.PUB)
            self._pub.setsockopt(zmq.LINGER, 0)
            self._pub.bind(_ephemeral(cfg.pub_endpoint))
        wake_ep = f"inproc://sremu-wake-{id(self)}"
        self._wake_pull = ctx.socket(zmq.PULL)
        self._wake_pull.bind(wake_ep)
        self._wake_push = ctx.socket(zmq.PUSH)
        self._wake_push.setsockopt(zmq.SNDHWM, 1)
        self._wake_push.connect(wake_ep)
        self._ingress = TxIngress(
            ctx,
            _ephemeral(cfg.tx_endpoint),
            self.engine,
            self._lookup,
            self.counters,
            on_async=self._send_async,
            throttle=self._tx_throttle,
            rcvhwm=cfg.tx_rcvhwm,
        )
        self.endpoints = {
            "control": self._router.getsockopt_string(zmq.LAST_ENDPOINT),
            "pub": None if self._pub is None else self._pub.getsockopt_string(zmq.LAST_ENDPOINT),
            "tx": self._ingress.endpoint,
        }
        self._rx_host = _bind_host(self.endpoints["control"] or DEFAULT_CONTROL_ENDPOINT)
        add_sink = getattr(self.engine, "add_sink", None)
        if callable(add_sink):
            add_sink(_PubSink(self))
        self._register_scenario_controllers()
        if cfg.raw_peers:
            from sremu.transport.rawreqrep import RawPeer

            for pc in cfg.raw_peers:
                peer = RawPeer(self, pc)
                peer.open()  # streams exist before the engine computes anything
                self._raw.append(peer)
                self._consumer_attached = True
        self._ingress.start()
        self._thread = threading.Thread(target=self._loop, name="sremu-control", daemon=True)
        self._thread.start()
        self._start_engine()
        for peer in self._raw:
            peer.start()

    def stop(self) -> None:
        """Stop every thread, the engine and the sockets (idempotent)."""
        if self._stopped.is_set():
            return
        self.stopping = True
        self._shutdown.set()
        for p in list(self.controllers.values()):
            p.wake()
        with self._resume_cv:
            self._resume_cv.notify_all()
        for peer in self._raw:
            peer.stop()
        for eg in list(self._egress.values()):
            eg.stop()
        if self._ingress is not None:
            self._ingress.stop()
        if self._driver is not None:  # paced: end the run where it is, then join
            self._end_paced_run()
            self._driver.join(5.0)
        stop = getattr(self.engine, "stop", None)
        if callable(stop):
            stop()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(5.0)
        with self._wake_lock:
            for s in ("_router", "_pub", "_wake_pull", "_wake_push"):
                sock = getattr(self, s, None)
                if sock is not None:
                    sock.close(0)
                    setattr(self, s, None)
        if self.ctx is not None:
            self.ctx.term()
        self._stopped.set()

    def wait(self, timeout_s: float | None = None) -> bool:
        """Block until a shutdown was requested (``sim.shutdown``, ``exit_when_done``) or
        ``timeout_s``; True when it was."""
        return self._shutdown.wait(timeout_s)

    def serve_forever(self) -> None:
        """Start (if needed), wait for shutdown, stop."""
        if not self._started:
            self.start()
        try:
            while not self._shutdown.wait(0.2):
                pass
        finally:
            self.stop()

    def __enter__(self) -> ControlServer:
        if not self._started:
            self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ engine driver
    @property
    def raw_peers(self) -> tuple[Any, ...]:
        """The started raw REQ/REP peers (:class:`sremu.transport.rawreqrep.RawPeer`)."""
        return tuple(self._raw)

    @property
    def paused(self) -> bool:
        """Whether virtual time is held (``sim.pause`` or ``start_paused``)."""
        return self._paused.is_set()

    def pause(self) -> None:
        """Hold virtual time after the current step."""
        self._paused.set()

    def resume(self) -> None:
        """Release virtual time."""
        self._paused.clear()
        with self._resume_cv:
            self._resume_cv.notify_all()

    def _wait_resume(self) -> bool:
        """Block while paused; True when the caller may step."""
        if not self._paused.is_set():
            return True
        with self._resume_cv:
            while self._paused.is_set() and not self.stopping:
                self._resume_cv.wait(0.05)
        return not self.stopping

    def _start_engine(self) -> None:
        eng: Any = self.engine
        start = getattr(eng, "start", None)
        if not callable(start):
            return
        cfg = getattr(eng, "cfg", None)
        if getattr(cfg, "mode", "lockstep") == "paced":
            self._driver = threading.Thread(
                target=self._paced_main, name="sremu-paced", daemon=True
            )
            self._driver.start()
            return
        orig = eng.step
        mark_blocked = getattr(eng, "mark_blocked", None)

        def step(*, zero_fill: bool = False) -> bool:
            if self._paused.is_set():
                if callable(mark_blocked):
                    mark_blocked()  # batching RX egress sends what is queued now
                ok = self._wait_resume()
                return ok and self.engine_error is None  # 'progress': restart stall timers
            if self.stopping or self.engine_error is not None:
                if self.engine_error is not None:
                    self._shutdown.wait(0.05)
                return False
            if self._engine_idle():
                return False  # nobody consumes RX: hold virtual time (cheap 5 ms waits)
            try:
                progressed = bool(orig(zero_fill=zero_fill))
            except Exception as exc:  # report, never lose it in the engine thread
                self._engine_failed(exc)
                return False
            time.sleep(0)  # let request and ingress threads take the engine lock
            return progressed

        eng.step = step
        start()

    def _engine_idle(self) -> bool:
        """Served lockstep holds virtual time while nothing consumes it: no RX stream is
        streaming or tapped (``capture_accumulator``), no barrier controller is registered,
        no sink records RX and the engine is not ``always_active``. Advancing would only
        skip pages (D-6) and make a later timed RX start late; the engine resumes as soon
        as a client streams.

        With ``exit_when_done`` the run proceeds to its end once a consumer has attached (a
        client opened an RX stream, a raw peer is configured, or a remote controller
        registered), and keeps going after that consumer closed its stream or unregistered:
        holding virtual time with nobody left to resume it would never end the run, so the
        server would never exit (a remote controller that unregistered before the last
        barrier used to leave ``sremu serve --exit-when-done`` waiting forever)."""
        eng: Any = self.engine
        rx = getattr(eng, "rx", None)
        cfg = getattr(eng, "cfg", None)
        if not isinstance(rx, Mapping) or cfg is None or getattr(cfg, "always_active", True):
            return False
        if self.config.exit_when_done and (self._consumer_attached or self._egress or self._raw):
            return False  # run on to the end of the run, then exit
        if any(
            getattr(st, "streaming", True) or getattr(st, "capture", None) is not None
            for st in rx.values()
        ):
            return False  # a client streams, or an accumulator tap consumes every page
        if getattr(getattr(eng, "barriers", None), "slots", True):
            return False
        sinks = getattr(eng, "sinks", ())
        return not any(type(s).on_rx is not EngineSink.on_rx for s in sinks)

    def _paced_main(self) -> None:
        eng: Any = self.engine
        # collect and freeze the heap while the run is still held: done in run_paced it would
        # delay the run's t0 by the collection (~25-55 ms) after sim.resume, and a client that
        # estimates virtual time from its resume would run that far ahead of the pacer
        freeze = getattr(eng, "freeze_heap", None)
        if callable(freeze):
            freeze()
        if not self._wait_resume():
            # stopped before the resume: no run_paced will release the freeze
            thaw = getattr(eng, "_thaw_heap", None)
            if callable(thaw):
                thaw()
            return
        # read_rx must wait for the paced loop instead of stepping the scheduler itself,
        # as with the engine's own served thread (Emulator.start); CCR WP8-2 asks WP6 for a
        # public served paced mode
        eng._stop_evt = threading.Event()
        eng._thread = threading.current_thread()
        try:
            eng.run_paced()
        except Exception as exc:
            self._engine_failed(exc)

    def _end_paced_run(self) -> None:
        """Make ``run_paced`` return: every rate group ends at its cursor."""
        eng: Any = self.engine
        lock = getattr(eng, "_lock", None)
        with lock if lock is not None else contextlib.nullcontext():
            for g in getattr(eng, "groups", ()):
                if g.end is None or g.end > g.cursor:
                    g.end = g.cursor

    def _engine_failed(self, exc: BaseException) -> None:
        if self.stopping:
            # an orderly stop unwinds a barrier wait (wait_ack -> False -> SremuTimeout):
            # that is the shutdown itself, not an engine failure
            return
        if self.engine_error is None:
            self.engine_error = exc
            self.publish(
                "sim.stall",
                {"report": {"error": f"{type(exc).__name__}: {exc}", "policy": "stopped"}},
            )

    def is_done(self) -> bool:
        """Whether the run has ended (every rate group reached its end) or failed."""
        if self.engine_error is not None or self.stopping:
            return True
        groups = getattr(self.engine, "groups", None)
        if not groups:
            return False
        return all(g.end is not None and g.cursor >= g.end for g in groups)

    def _frontier_of(self, h: StreamHandle) -> int:
        fr = getattr(self.engine, "frontier", None)
        if callable(fr):
            d = fr()
            if h.spec.stream in d:
                return int(d[h.spec.stream])
        return to_ticks(self.engine.time_now(), h.spec.rate_sps)

    def _tx_throttle(self) -> bool:
        """Lockstep backpressure on TX ingress (see :mod:`zmq_native`)."""
        eng: Any = self.engine
        groups = getattr(eng, "groups", None)
        tx_head = getattr(eng, "tx_head", None)
        if not groups or not callable(tx_head) or getattr(eng.cfg, "mode", "") != "lockstep":
            return False
        if any(g.waiting_on for g in groups):
            return False
        window = self.config.tx_window_samples
        with self._handles_lock:
            txs = [h for h in self._handles.values() if h.spec.direction == "tx"]
        for h in txs:
            rate = h.spec.rate_sps
            cur = max((g.cursor for g in groups if g.rate == rate), default=None)
            if cur is None:
                continue
            try:
                if int(tx_head(h)) - cur > window:
                    return True
            except SremuError:
                continue
        return False

    # ------------------------------------------------------------------ outbound
    def _wake(self) -> None:
        with self._wake_lock:
            sock = self._wake_push
            if sock is None or self.stopping:
                return
            with contextlib.suppress(zmq.ZMQError):
                sock.send(b"", zmq.NOBLOCK)

    def route(self, ident: bytes, delimited: bool, payload: bytes) -> None:
        """Queue a message for one connection (any thread)."""
        with self._out_lock:
            self._out.append(("route", ident, delimited, payload))
        self._wake()

    def publish(self, topic: str, body: Mapping[str, Any]) -> None:
        """Queue a PUB telemetry message (any thread; dropped when PUB is disabled)."""
        if self.config.pub_endpoint is None:
            return
        try:
            # non-finite floats (e.g. a metrics window's rtf = inf when no wall time passed
            # on a coarse clock) travel as "Infinity"/"-Infinity"/"NaN", as in observations
            payload = encode_json({"v": SCHEMA_VERSION, "topic": topic, **_json_safe(body)})
        except (TypeError, ValueError):  # not JSON at all: drop (PUB may drop), but count
            self.counters.add("internal_errors")
            return
        with self._out_lock:
            self._out.append(("pub", topic.encode("ascii"), False, payload))
        self._wake()

    def _flush(self) -> None:
        while True:
            with self._out_lock:
                if not self._out:
                    return
                kind, a, delim, payload = self._out.popleft()
            try:
                if kind == "route":
                    self._router.send_multipart(
                        [a, b"", payload] if delim else [a, payload], zmq.NOBLOCK
                    )
                elif self._pub is not None:
                    self._pub.send_multipart([a, payload], zmq.NOBLOCK)
            except zmq.ZMQError:
                pass  # unroutable peer (gone) or PUB queue full: telemetry may drop

    def _send_async(self, h: StreamHandle, md: AsyncMetadata) -> None:
        owner = self._owner.get(h.id)
        body = {
            "v": SCHEMA_VERSION,
            "op": "stream.async",
            "stream_id": h.id,
            "stream": h.spec.stream,
            "event_code": md.event_code.name,
            "sample_index": md.sample_index,
            "n_samples": int(md.n_samples),
            "t": fraction_to_str(md.time_spec.exact()) if md.has_time_spec else None,
        }
        if md.event_code.name == "UNDERFLOW":
            self.publish(
                "stream.underflow",
                {
                    "stream": h.spec.stream,
                    "sample_index": md.sample_index,
                    "n_samples": md.n_samples,
                },
            )
        if owner is not None:
            s = self._sessions.get(owner)
            self.route(owner, bool(s and s.delimited), encode_json(body))

    def _forward_async(self) -> None:
        recv_async = getattr(self.engine, "recv_async", None)
        if not callable(recv_async):
            return
        with self._handles_lock:
            txs = [h for h in self._handles.values() if h.spec.direction == "tx"]
        for h in txs:
            for _ in range(64):
                try:
                    md = recv_async(h, 0.0)
                except SremuError:
                    break
                if md is None:
                    break
                self._send_async(h, md)

    # ------------------------------------------------------------------ main loop
    def _loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        poller.register(self._wake_pull, zmq.POLLIN)
        while not self.stopping:
            try:
                events = dict(poller.poll(20))
            except zmq.ZMQError:  # pragma: no cover - context terminated
                break
            if self._wake_pull in events:
                while True:
                    try:
                        self._wake_pull.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
            if self._router in events:
                for _ in range(256):
                    try:
                        frames = self._router.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        self._on_message(frames)
                    except Exception:  # pragma: no cover - last line of defence
                        self.counters.add("internal_errors")
            self._forward_async()
            self._flush()
            if self._shutdown.is_set():
                break
            if (
                self.config.exit_when_done
                and self.is_done()
                and all(e.ended for e in self._egress.values())
            ):
                self._shutdown.set()
                break
        self._flush()

    # ------------------------------------------------------------------ requests
    def _on_message(self, frames: list[bytes]) -> None:
        ident = frames[0]
        if len(frames) == 2:
            delimited, payload = False, frames[1]
        elif len(frames) == 3 and frames[1] == b"":
            delimited, payload = True, frames[2]
        else:
            self.counters.add("schema_errors")
            self._reply(
                ident,
                False,
                error_reply(None, ErrorCode.SCHEMA_ERROR, "a request is one message part"),
            )
            return
        self.requests += 1
        s = self._sessions.get(ident)
        if s is None:
            s = _Session(ident, delimited)
            self._sessions[ident] = s
        s.delimited = delimited
        t_recv = self._now_ns()  # the engine's clock (see __init__)
        try:
            msg = parse_request(payload, max_bytes=self.config.max_message_bytes)
        except ProtocolError as exc:
            self._count_refusal(exc.code)
            self._reply(ident, delimited, error_reply(exc.msg_id, exc.code, exc.msg))
            return
        mid = msg["id"]
        try:
            reply = self._dispatch(s, msg, t_recv)
        except ProtocolError as exc:
            self._count_refusal(exc.code)
            reply = error_reply(mid, exc.code, exc.msg)
        except SremuError as exc:
            code = exc.code if isinstance(exc.code, ErrorCode) else ErrorCode.NOT_PERMITTED
            reply = error_reply(mid, code, str(exc))
        except Exception as exc:  # never let a request take the server down
            self.counters.add("internal_errors")
            reply = error_reply(mid, ErrorCode.NOT_PERMITTED, f"{type(exc).__name__}: {exc}")
        if reply is not None:
            self._reply(ident, delimited, reply)
        if msg["op"] == "sim.shutdown":
            self._flush()
            self._shutdown.set()

    def _count_refusal(self, code: ErrorCode) -> None:
        if code == ErrorCode.SCHEMA_ERROR:
            self.counters.add("schema_errors")
        elif code == ErrorCode.TOO_LARGE:
            self.counters.add("too_large_messages")

    def _reply(self, ident: bytes, delimited: bool, body: Mapping[str, Any]) -> None:
        try:
            payload = encode_json(body)
        except (TypeError, ValueError) as exc:  # a reply that JSON cannot carry
            self.counters.add("internal_errors")
            payload = encode_json(
                error_reply(body.get("id"), ErrorCode.NOT_PERMITTED, f"unencodable reply: {exc}")
            )
        with contextlib.suppress(zmq.ZMQError):  # an unroutable (gone) peer
            self._router.send_multipart(
                [ident, b"", payload] if delimited else [ident, payload], zmq.NOBLOCK
            )

    def _ok(self, msg: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
        return {"v": SCHEMA_VERSION, "id": msg["id"], "ok": True, **fields}

    def _dispatch(self, s: _Session, msg: dict[str, Any], t_recv: int) -> dict[str, Any] | None:
        op = msg["op"]
        if op in COMMAND_OPS:
            return self._command(s, msg, t_recv)
        handler: Callable[[_Session, dict[str, Any]], dict[str, Any]] = getattr(
            self, "_op_" + op.replace(".", "_")
        )
        return handler(s, msg)

    # ------------------------------------------------------------------ ops
    def _op_hello(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        cv = msg.get("contract_version")
        if cv is not None and cv.split(".")[0] != CONTRACT_VERSION.split(".")[0]:
            raise ProtocolError(
                ErrorCode.NOT_PERMITTED,
                f"incompatible contract_version {cv!r}; this emulator speaks {CONTRACT_VERSION}",
            )
        if "client" in msg:
            s.name = str(msg["client"])
        eng: Any = self.engine
        cfg = getattr(eng, "cfg", None)
        streams = {}
        for sid, c in dict(getattr(eng, "configs", {}) or {}).items():
            streams[sid] = {
                "direction": c.direction,
                "fc_hz": float(c.fc_hz),
                "rate_sps": fraction_to_str(c.rate),
            }
        return self._ok(
            msg,
            server="sremu",
            sremu_version=SREMU_VERSION,
            contract_version=CONTRACT_VERSION,
            schema_version=SCHEMA_VERSION,
            ops=list(OPS),
            mode=getattr(cfg, "mode", "lockstep"),
            coupling=getattr(cfg, "coupling", "barrier"),
            paused=self.paused,
            endpoints=dict(self.endpoints),
            limits={
                "max_message_bytes": self.config.max_message_bytes,
                "max_elements_per_command": MAX_ELEMENTS_PER_COMMAND,
                "max_command_rate_hz": self.config.max_command_rate_hz,
                "max_frame_samples": MAX_FRAME_SAMPLES,
            },
            streams=streams,
        )

    def _op_caps_get(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        caps = self.engine.caps()
        p = msg.get("panel")
        if p is not None:
            if p not in caps:
                raise ProtocolError(ErrorCode.UNKNOWN_TARGET, f"unknown panel {p!r}")
            caps = {p: caps[p]}
        return self._ok(msg, caps=caps_to_json(caps))

    def _op_time_now(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        fr = getattr(self.engine, "frontier", None)
        return self._ok(
            msg,
            t=fraction_to_str(self.engine.time_now()),
            frontier={k: int(v) for k, v in (fr() if callable(fr) else {}).items()},
            paused=self.paused,
            done=self.is_done(),
        )

    def _command(self, s: _Session, msg: dict[str, Any], t_recv: int) -> dict[str, Any]:
        if s.bucket is None:
            rate = self.config.max_command_rate_hz
            s.bucket = _TokenBucket(rate, math.ceil(rate * self.config.rate_burst_s))
        if not s.bucket.take():
            self.counters.add("rate_limited_commands")
            self.publish(
                "cmd.rejected",
                {"id": msg["id"], "error": {"code": "NOT_PERMITTED", "msg": "rate limit"}},
            )
            raise ProtocolError(
                ErrorCode.NOT_PERMITTED,
                f"command rate above {self.config.max_command_rate_hz} Hz for this client",
            )
        src = str(msg.get("src") or s.name)
        proxy = self.controllers.get(src)
        if proxy is not None and proxy.ident != s.ident:
            raise ProtocolError(
                ErrorCode.NOT_PERMITTED,
                f"src {src!r} is a remote controller registered by another connection",
            )
        if proxy is None and self._is_barrier_controller(src):
            # the engine trusts a barrier controller's commands (lookahead bound, no
            # nondeterministic-input flag when late): a client must not borrow that name
            raise ProtocolError(
                ErrorCode.NOT_PERMITTED, f"src {src!r} is an in-process barrier controller"
            )
        op = msg["op"]
        if op.startswith("node."):
            self._check_node(msg)
        req = command_from_request(msg, src=src, t_wall_recv_ns=t_recv)
        receipt = self.engine.schedule(req)
        return receipt_reply(receipt)

    def _is_barrier_controller(self, name: str) -> bool:
        slot = getattr(getattr(self.engine, "barriers", None), "slot", None)
        return bool(callable(slot) and slot(name) is not None)

    def _check_node(self, msg: Mapping[str, Any]) -> None:
        """node.* commands need an external-trajectory node (scenario, section 9.2) and a
        trajectory that the scenario schema accepts; refused before reaching the engine."""
        node = msg["node"]
        sc = self.scenario
        if sc is not None:
            nodes = {n.get("id"): n for n in sc.get("nodes") or []}
            if node not in nodes:
                raise ProtocolError(ErrorCode.UNKNOWN_TARGET, f"unknown node {node!r}")
            traj = nodes[node].get("trajectory") or {}
            model = traj.get("model") if isinstance(traj, Mapping) else None
            if model != "external":
                raise ProtocolError(
                    ErrorCode.NOT_PERMITTED,
                    f"node {node!r} does not have an external trajectory (model {model!r})",
                )
        if msg["op"] == "node.set_trajectory":
            from pydantic import TypeAdapter, ValidationError

            from sremu.scenario.models import Trajectory

            try:
                TypeAdapter(Trajectory).validate_python(dict(msg["trajectory"]))
            except ValidationError as exc:
                raise ProtocolError(
                    ErrorCode.SCHEMA_ERROR, f"invalid trajectory: {exc.errors()[0]['msg']}"
                ) from None

    def _op_ris_set_codebook(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        from sremu.transport.messages import _int16_b64

        eng: Any = self.engine
        specs = getattr(getattr(eng, "commands", None), "specs", None)
        p = msg["panel"]
        if specs is None:
            raise ProtocolError(ErrorCode.NOT_PERMITTED, "this engine has no codebook support")
        if p not in specs:
            raise ProtocolError(ErrorCode.UNKNOWN_TARGET, f"unknown panel {p!r}")
        k, n = (int(v) for v in msg["shape"])
        arr = _int16_b64(msg["codebook_b64"], "codebook_b64")
        if arr.size != k * n:
            raise ProtocolError(
                ErrorCode.SCHEMA_ERROR, f"codebook has {arr.size} entries != {k}x{n}"
            )
        n_ctrl = int(specs[p].n_controls)
        if n != n_ctrl:
            raise ProtocolError(ErrorCode.SCHEMA_ERROR, f"codebook rows need {n_ctrl} controls")
        name = str(msg.get("name") or "default")
        specs[p].codebooks[name] = arr.reshape(k, n).copy()
        return self._ok(msg, panel=p, name=name, shape=[k, n])

    def _op_ris_get_state(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        eng: Any = self.engine
        p = msg["panel"]
        if p not in self.engine.caps():
            raise ProtocolError(ErrorCode.UNKNOWN_TARGET, f"unknown panel {p!r}")
        t = self.engine.time_now()
        state_fn = getattr(eng, "panel_state", None)
        if not callable(state_fn):
            raise ProtocolError(ErrorCode.NOT_PERMITTED, "this engine cannot report panel states")
        cfg: PanelConfig = state_fn(p, t)
        recs = getattr(eng, "records", ())
        recs = recs() if callable(recs) else recs
        pending = [r.to_dict() for r in recs if r.panel == p and r.status == "scheduled"]
        return self._ok(
            msg,
            panel=p,
            t=fraction_to_str(t),
            states_b64=b64_of(cfg.state, "<i2"),
            dtype="i2",
            enabled=bool(cfg.enabled),
            config_hash=config_hash(cfg.state, bool(cfg.enabled)),
            pending=pending,
        )

    def _lookup(self, sid: int) -> StreamHandle | None:
        with self._handles_lock:
            return self._handles.get(int(sid))

    def rx_frame_for(self, page_size: int) -> int:
        """Samples per RX frame (at most) of a stream with ``page_size``: the configured
        ``rx_frame_samples``, else one page in paced mode and :data:`RX_BATCH_PAGES` pages
        in lockstep."""
        if self.config.rx_frame_samples:
            return min(int(self.config.rx_frame_samples), MAX_FRAME_SAMPLES)
        if getattr(getattr(self.engine, "cfg", None), "mode", "lockstep") == "paced":
            return int(page_size)
        return min(RX_BATCH_PAGES * int(page_size), MAX_FRAME_SAMPLES)

    def _op_stream_open(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        rate = exact_from_json(msg["rate_sps"], "rate_sps")
        spec = StreamSpec(
            msg["stream"],
            msg["direction"],
            float(msg["fc_hz"]),
            rate,
            msg.get("cpu", "fc32"),
            int(msg.get("nchan", 1)),
        )
        h = self.engine.open_stream(spec)
        with self._handles_lock:
            self._handles[h.id] = h
            self._owner[h.id] = s.ident
        s.streams.add(h.id)
        extra: dict[str, Any] = {}
        if spec.direction == "tx":
            endpoint = self.endpoints["tx"]
        else:
            assert self.ctx is not None
            cfg = getattr(self.engine, "cfg", None)
            paced = getattr(cfg, "mode", "lockstep") == "paced"
            fs = self.rx_frame_for(h.page_size)
            eg = RxEgress(
                self.ctx,
                f"tcp://{self._rx_host}:*",
                self.engine,
                h,
                self.counters,
                frame_samples=fs,
                paced=paced,
                sndhwm=hwm_frames(self.config.rx_sndhwm, h.page_size, fs),
                is_done=self.is_done,
                frontier=functools.partial(self._frontier_of, h),
                publish=self.publish,
                flush_s=self.config.rx_flush_s,
            )
            self._egress[h.id] = eg
            self._consumer_attached = True
            eg.start()
            endpoint = eg.endpoint
            extra["frame_samples"] = eg.frame_samples  # sizes the client's RCVHWM
        return self._ok(
            msg,
            stream_id=h.id,
            stream=spec.stream,
            direction=spec.direction,
            page_size=h.page_size,
            rate_sps=fraction_to_str(spec.rate_sps),
            fc_hz=spec.fc_hz,
            endpoint=endpoint,
            **extra,
        )

    def _owned(self, s: _Session, sid: int) -> StreamHandle:
        h = self._lookup(sid)
        if h is None:
            raise ProtocolError(ErrorCode.UNKNOWN_TARGET, f"unknown stream_id {sid}")
        if self._owner.get(h.id) != s.ident:
            raise ProtocolError(ErrorCode.NOT_PERMITTED, "stream opened by another connection")
        return h

    def _op_stream_close(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        """TX: end of stream (committed silence from the stream's head) and the stream id
        is forgotten; RX: ``stop_cont`` and its egress socket closes. The data plane is a
        separate connection, so frames still in flight to a closed TX stream would be
        counted as bad packets (``stream_id``): ``RemoteEngine.close_stream`` therefore
        sends ``END_OF_STREAM`` in-band and waits until TX ingress has taken every frame
        (``stats.get`` ``tx_next_seq``) before it sends ``stream.close``."""
        from sremu.contracts import TxMetadata

        h = self._owned(s, int(msg["stream_id"]))
        if h.spec.direction == "tx":
            self.engine.submit_tx(h, np.zeros(0, np.complex64), TxMetadata(end_of_stream=True))
            if self._ingress is not None:
                self._ingress.forget(h.id)
        else:
            self.engine.stream_cmd(h, StreamCmd("stop_cont"))
            eg = self._egress.pop(h.id, None)
            if eg is not None:
                eg.stop()
        with self._handles_lock:
            self._handles.pop(h.id, None)
            self._owner.pop(h.id, None)
        s.streams.discard(h.id)
        return self._ok(msg, stream_id=h.id)

    def _op_stream_cmd(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        h = self._owned(s, int(msg["stream_id"]))
        if h.spec.direction != "rx":
            raise ProtocolError(ErrorCode.NOT_PERMITTED, "stream.cmd needs an RX stream")
        t = msg.get("t")
        cmd = StreamCmd(
            msg["mode"],
            int(msg.get("num_samps", 0)),
            bool(msg.get("stream_now", t is None)),
            TimeSpec.from_exact(exact_from_json(t, "t")) if t is not None else TimeSpec(),
        )
        self.engine.stream_cmd(h, cmd)
        return self._ok(msg, stream_id=h.id)

    # ------------------------------------------------------------------ controllers
    def _register_scenario_controllers(self) -> None:
        """Register the scenario's controllers in scenario order (barrier ties follow
        registration order): ``external`` ones as :class:`RemoteProxy`, ``builtin`` ones
        through ``builtin_factory`` (when given)."""
        sc = self.scenario
        if not sc:
            return
        for c in sc.get("controllers") or []:
            if c.get("kind") != "external":
                if self.builtin_factory is not None:
                    self.engine.register_controller(self.builtin_factory(c))
                continue
            lat = dict(c.get("latency") or {})
            obs = dict(c.get("observation") or {})
            spec = ObservationSpec(
                tuple(obs.get("fields", ("rx_power_dbm", "ris_state"))),
                obs.get("window_s", Fraction(1, 1000)),
                obs.get("position_delay_s", Fraction(0)),
                float(obs.get("position_noise_m", 0.0)),
            )
            proxy = RemoteProxy(
                self,
                str(c["name"]),
                Fraction(c["period_s"]),
                Fraction(c["min_latency_s"]),
                spec,
                LatencyModel(lat.get("fixed_s", Fraction(0)), lat.get("jitter")),
                preregistered=True,
            )
            self.engine.register_controller(proxy)
            self.controllers[proxy.name] = proxy

    def _estimate_observation_bytes(self, spec: ObservationSpec) -> int:
        eng: Any = self.engine
        total = 4096
        configs = dict(getattr(eng, "configs", {}) or {})
        rx = [c for c in configs.values() if c.direction == "rx"]
        total += 256 * len(rx)
        if "iq_window" in spec.fields:
            total += sum(to_ticks(spec.window_s, c.rate) * 8 * 4 // 3 + 64 for c in rx)
        if "csi_oracle" in spec.fields:
            caps = self.engine.caps()
            model = getattr(eng, "model", None)
            n_links = len(model.links()) if model is not None else len(rx)
            for c in caps.values():
                total += n_links * len(c.states) * c.n_elements * 16 * 4 // 3
        return total

    def _op_ctrl_register(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        name = str(msg["name"])
        period = exact_from_json(msg["period_s"], "period_s")
        lmin = exact_from_json(msg["min_latency_s"], "min_latency_s")
        if period <= 0:
            raise ProtocolError(ErrorCode.SCHEMA_ERROR, "period_s must be > 0")
        try:
            lat = _latency_from_json(msg.get("latency"), lmin)
            spec = _spec_from_json(msg.get("observation"))
        except SremuError as exc:
            raise ProtocolError(ErrorCode.SCHEMA_ERROR, str(exc)) from None
        if self._estimate_observation_bytes(spec) > self.config.max_message_bytes:
            raise ProtocolError(
                ErrorCode.TOO_LARGE,
                "the requested observation does not fit in one 1 MiB message",
            )
        proxy = self.controllers.get(name)
        if proxy is not None:
            if not proxy.preregistered or proxy.ident is not None:
                raise ProtocolError(ErrorCode.NOT_PERMITTED, f"controller {name!r} is registered")
            if not _same_controller(proxy, period, lmin, lat, spec):
                raise ProtocolError(
                    ErrorCode.NOT_PERMITTED,
                    f"controller {name!r} differs from the scenario's external controller "
                    f"(period_s {proxy.period_s}, min_latency_s {proxy.min_latency_s}, "
                    f"latency {proxy.latency}, observation {proxy.spec})",
                )
        else:
            proxy = RemoteProxy(self, name, period, lmin, spec, lat)
            self.engine.register_controller(proxy)
            self.controllers[name] = proxy
        s.controllers.add(name)
        proxy.attach(s.ident, s.delimited)
        self._consumer_attached = True
        eng: Any = self.engine
        cfg = getattr(eng, "cfg", None)
        return self._ok(
            msg,
            name=name,
            seed=int(getattr(cfg, "seed", 0)),
            late_policy=getattr(cfg, "late_policy", "apply_on_arrival"),
            mode=getattr(cfg, "mode", "lockstep"),
            coupling=getattr(cfg, "coupling", "barrier"),
            preregistered=proxy.preregistered,
            caps=caps_to_json(self.engine.caps()),
        )

    def _controller_of(self, s: _Session, name: str) -> RemoteProxy:
        p = self.controllers.get(name)
        if p is None:
            raise ProtocolError(ErrorCode.UNKNOWN_TARGET, f"unknown controller {name!r}")
        if p.ident != s.ident:
            raise ProtocolError(ErrorCode.NOT_PERMITTED, f"controller {name!r} is not yours")
        return p

    def _op_ctrl_ack(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        p = self._controller_of(s, str(msg["controller"]))
        t_b = exact_from_json(msg["t_b"], "t_b")
        if not p.ack(t_b):
            raise ProtocolError(ErrorCode.NOT_PERMITTED, f"no barrier pending at t_b = {t_b}")
        return self._ok(msg, controller=p.name, t_b=fraction_to_str(t_b))

    def _op_ctrl_unregister(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        p = self._controller_of(s, str(msg["controller"]))
        p.release()
        unregister = getattr(self.engine, "unregister_controller", None)
        if callable(unregister):
            unregister(p.name)
        self.controllers.pop(p.name, None)
        s.controllers.discard(p.name)
        return self._ok(msg, controller=p.name)

    # ------------------------------------------------------------------ misc
    def _op_stats_get(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        eng: Any = self.engine
        counters = dict(getattr(eng, "counters", {}) or {})
        recs = getattr(eng, "records", ())
        recs = recs() if callable(recs) else recs
        return self._ok(
            msg,
            counters={k: int(v) for k, v in counters.items()},
            transport=self.counters.snapshot(),
            bad_packet_reasons=self.counters.reasons_snapshot(),
            nondeterministic_inputs=bool(getattr(eng, "nondeterministic_inputs", False)),
            records=len(recs),
            requests=self.requests,
            t=fraction_to_str(self.engine.time_now()),
            paused=self.paused,
            done=self.is_done(),
            engine_error=None
            if self.engine_error is None
            else f"{type(self.engine_error).__name__}: {self.engine_error}",
            controllers={
                n: {"attached": p.ident is not None, "barriers": p.barriers_sent, "acks": p.acks}
                for n, p in self.controllers.items()
            },
            raw_peers={p.config.name: p.stats() for p in self._raw},
            tx_next_seq=self._tx_next_seq(),
        )

    def _tx_next_seq(self) -> dict[str, int | None]:
        """Per open TX stream id: the frame ``seq`` TX ingress expects next (how far the
        data plane has been ingested; ``RemoteEngine.close_stream`` waits on it)."""
        with self._handles_lock:
            sids = [sid for sid, h in self._handles.items() if h.spec.direction == "tx"]
        ing = self._ingress
        return {str(sid): None if ing is None else ing.next_seq(sid) for sid in sids}

    def _op_sim_pause(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        if self._driver is not None and not self._paused.is_set():
            # a running paced run follows the wall clock (run_paced has no pause point):
            # refuse instead of replying paused=True while virtual time keeps running
            raise ProtocolError(ErrorCode.NOT_PERMITTED, "a running paced run cannot pause")
        self.pause()
        return self._ok(msg, paused=True)

    def _op_sim_resume(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        self.resume()
        return self._ok(msg, paused=False)

    def _op_sim_shutdown(self, s: _Session, msg: dict[str, Any]) -> dict[str, Any]:
        return self._ok(msg, shutdown=True)


def serve_engine(engine: EngineProtocol, **config: Any) -> ControlServer:
    """Start a :class:`ControlServer` for ``engine`` with ``ServerConfig`` fields."""
    return ControlServer(engine, ServerConfig(**config)).start()


def endpoints_summary(server: ControlServer) -> dict[str, Any]:
    """JSON-ready endpoints of a started server (printed by ``sremu serve``)."""
    return {"endpoints": dict(server.endpoints), "paused": server.paused}
