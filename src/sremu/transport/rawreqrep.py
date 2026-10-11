"""Raw REQ/REP mode for srsRAN- and GNU Radio-style peers (DESIGN.md section 8.4; IO-08).

Per peer (device args ``tx_port``, ``rx_port``, ``base_srate``) the emulator acts like the
GNU Radio broker of the srsRAN ZMQ tutorials:

* it **connects a REQ** to the peer's ``tx_port`` (the peer's REP) and pulls TX samples;
* it **binds a REP** at the peer's ``rx_port`` and answers the peer's RX requests with
  raw samples.

There is no header: timing is implicit in sample counts (the peer's TX stream and RX stream
both start at absolute sample 0 and are contiguous), so the mode is **lockstep only**.
Wording: "srsRAN-style REQ/REP raw-IQ mode", never "wire-compatible".

**Tested peers.** Stub peers of both formats (IO-08) and, for ``gnuradio_count``, stock GNU
Radio 3.10.9.2 (Ubuntu 24.04 package; gr-zeromq REP Sink / REQ Source with a gr-digital
DQPSK modem, ``examples/gnuradio_zmq``, ``tests/integration/test_gnuradio_raw.py``; Linux
only). No srsRAN binary has been run. srsRAN 4G pads its TX stream only up to the end of the
RX block it is about to read, while lockstep computes RX samples only once the reaching TX is
committed one fractional-delay look-ahead further, so an srsRAN pair would stall at its
first subframe without a per-peer TX delay of at least that look-ahead and partial RX
replies (not implemented).

**Ending a run.** ``exit_when_done`` (``sremu serve --exit-when-done``) needs every raw peer
to read its RX stream to the end of the run: the engine cannot finish while a raw consumer
stops reading (RX backpressure), so the server would never exit. When the run has ended,
:meth:`RawPeer.stop` first lets the peer read what the engine already computed for it (the
engine may end up to ``rx_queue_pages`` pages ahead of a slow peer); it gives up once the
peer has sent no request for ``DRAIN_IDLE_S`` or after ``DRAIN_MAX_S``.

**TX request size.** A ``gnuradio_count`` TX request asks for ``request_items`` samples
(default :func:`default_request_items`: 8 pages, at least 4096 and at most 65,536). The REP
Sink answers ``min(available, count)``, so a larger request never waits for more samples; it
only saves round trips: with one page per request a free-running GNU Radio transmitter
halved the served lockstep throughput at 1 MS/s (one cross-thread hand-off per 0.5 ms page).

**Request formats, pinned from upstream source** (``request_format`` per peer):

* ``"srsran_dummy"`` -- srsRAN 4G ``lib/src/phy/rf/rf_zmq_imp_rx.c``
  (``rf_zmq_async_rx_thread``): the RX side is a REQ that sends one byte
  (``uint8_t dummy = 0xFF; zmq_send(sock, &dummy, sizeof(dummy), 0)``) and then receives up
  to ``ZMQ_MAX_BUFFER_SIZE`` bytes (``NSAMPLES2NBYTES(3072000)``, ``rf_zmq_imp_trx.h``) of
  raw baseband into its ring buffer. ``rf_zmq_imp_tx.c`` (``_rf_zmq_tx_baseband``): the TX
  side is a bound REP that reads the request into a one-byte buffer, ignores it and replies
  with ``nsamples`` samples of whatever its PHY produced. Samples are ``fc32`` by default
  (``tx_format``/``rx_format`` ``sc16``: int16 with full scale ``INT16_MAX``).
* ``"gnuradio_count"`` -- GNU Radio 3.10 ``gr-zeromq/lib/req_source_impl.cc``: the REQ
  sends a 4-byte message holding ``uint32_t req_len = noutput_items - done`` in host byte
  order (little-endian on the x86/ARM hosts supported here) and accepts a reply of any
  whole number of items (``pass_tags=false``: no header). ``rep_sink_impl.cc``: a REP that
  reads a request of at least 4 bytes as that ``uint32`` item count and replies with
  ``min(available, req)`` items (all available items when the request is shorter).

The emulator's REP is strict: in ``srsran_dummy`` mode a request must be the single byte
``0xFF``; in ``gnuradio_count`` mode exactly 4 bytes. Anything else is counted as a bad
packet (reason ``raw_request``) and answered with an empty reply (the REQ/REP state
machine needs a reply). Replies carry exactly the requested count (``gnuradio_count``) or
``reply_items`` samples (``srsran_dummy``; at most 3,072,000), blocking until the engine has
computed them (fewer only at the end of the run), so the peer receives every RX sample
exactly once and in order. TX replies
must be whole samples (else ``raw_size`` bad packet, dropped); every TX sample received is
placed contiguously from index 0.

Sources checked on 10 Oct 2026: srsRAN_4G ``master`` (rf_zmq_imp_rx.c, rf_zmq_imp_tx.c,
rf_zmq_imp_trx.h; identical to the copies in ``research/scratch/src_refs/``; rf_zmq_imp.c:
``rx_opts.socket_type = ZMQ_REQ``, ``tx_opts.socket_type = ZMQ_REP`` by default, device args
``base_srate``, ``tx_port``, ``rx_port``, ``tx_freq``, ``rx_freq``, ``tx_format``,
``rx_format``) and GNU Radio ``maint-3.10`` and ``main`` (req_source_impl.cc,
rep_sink_impl.cc). IO-08 tests stub peers that implement exactly these behaviours; the GNU
Radio format was confirmed against GNU Radio 3.10.9.2 itself (see "Tested peers").
"""

from __future__ import annotations

import struct
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
import zmq
from numpy.typing import NDArray

from sremu.contracts import StreamCmd, StreamHandle, StreamSpec, TxMetadata
from sremu.errors import ConfigurationError, ErrorCode, NotPermitted, SremuError
from sremu.timebase import TimeSpec, parse_exact
from sremu.transport.framing import Dtype, decode_payload, encode_payload

if TYPE_CHECKING:
    from sremu.transport.control_server import ControlServer

__all__ = [
    "DRAIN_IDLE_S",
    "DRAIN_MAX_S",
    "GNURADIO_COUNT",
    "GNURADIO_REQUEST_MAX_ITEMS",
    "GNURADIO_REQUEST_MIN_ITEMS",
    "GNURADIO_REQUEST_PAGES",
    "REQUEST_FORMATS",
    "SRSRAN_DUMMY",
    "SRSRAN_DUMMY_REQUEST",
    "SRSRAN_MAX_REPLY_SAMPLES",
    "RawPeer",
    "RawPeerConfig",
    "RawProtocolError",
    "default_request_items",
    "encode_request",
    "parse_request",
]

RequestFormat = Literal["srsran_dummy", "gnuradio_count"]
SRSRAN_DUMMY: Final = "srsran_dummy"
GNURADIO_COUNT: Final = "gnuradio_count"
REQUEST_FORMATS: Final[tuple[str, ...]] = (SRSRAN_DUMMY, GNURADIO_COUNT)
SRSRAN_DUMMY_REQUEST: Final[bytes] = b"\xff"
"""srsRAN 4G RX request: one byte 0xFF (rf_zmq_imp_rx.c)."""
SRSRAN_MAX_REPLY_SAMPLES: Final[int] = 3_072_000
"""srsRAN 4G ZMQ_MAX_BUFFER_SIZE in samples (rf_zmq_imp_trx.h: 10 subframes at 20 MHz)."""
GNURADIO_REQUEST_PAGES: Final[int] = 8
"""Default ``gnuradio_count`` TX request: this many pages of the TX stream ..."""
GNURADIO_REQUEST_MIN_ITEMS: Final[int] = 4096
"""... but at least this many samples ..."""
GNURADIO_REQUEST_MAX_ITEMS: Final[int] = 65_536
"""... and at most this many (the largest request measured, 8 pages at 23.04 MS/s)."""
DRAIN_IDLE_S: Final[float] = 1.0
"""At the end of a run, stop waiting for a raw peer that sent no RX request for this long."""
DRAIN_MAX_S: Final[float] = 30.0
"""At the end of a run, stop waiting for a raw peer after this long in any case."""
_U32: Final = struct.Struct("<I")


def default_request_items(fmt: str, page_size: int) -> int:
    """Default items per TX request: for ``gnuradio_count`` :data:`GNURADIO_REQUEST_PAGES`
    pages, clipped to [:data:`GNURADIO_REQUEST_MIN_ITEMS`,
    :data:`GNURADIO_REQUEST_MAX_ITEMS`]; for ``srsran_dummy`` (the peer ignores the request's
    content) the page size."""
    if fmt == GNURADIO_COUNT:
        return min(
            GNURADIO_REQUEST_MAX_ITEMS,
            max(GNURADIO_REQUEST_MIN_ITEMS, GNURADIO_REQUEST_PAGES * int(page_size)),
        )
    return int(page_size)


class RawProtocolError(ValueError):
    """A raw request or reply that does not follow the peer's pinned format."""

    code = ErrorCode.BAD_PACKET

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def encode_request(fmt: str, n_items: int) -> bytes:
    """The request the emulator's REQ sends to a peer's TX REP in format ``fmt``."""
    if fmt == SRSRAN_DUMMY:
        return SRSRAN_DUMMY_REQUEST
    if fmt == GNURADIO_COUNT:
        return _U32.pack(int(n_items))
    raise ConfigurationError(f"unknown request_format {fmt!r}")


def parse_request(fmt: str, msg: bytes) -> int | None:
    """Items requested by a peer's RX REQ (None: srsRAN, 'whatever is available').
    Raises :class:`RawProtocolError` when ``msg`` is not in the pinned format."""
    if fmt == SRSRAN_DUMMY:
        if msg != SRSRAN_DUMMY_REQUEST:
            raise RawProtocolError("raw_request", f"srsran_dummy expects b'\\xff', got {msg[:8]!r}")
        return None
    if fmt == GNURADIO_COUNT:
        if len(msg) != 4:
            raise RawProtocolError("raw_request", f"gnuradio_count expects 4 bytes, got {len(msg)}")
        return int(_U32.unpack(msg)[0])
    raise ConfigurationError(f"unknown request_format {fmt!r}")


def _bindable(endpoint: str) -> str:
    """``tcp://host:0`` -> ``tcp://host:*`` (ephemeral port; read it from ``rx_endpoint``)."""
    if endpoint.startswith("tcp://") and endpoint.endswith(":0"):
        return endpoint[:-2] + ":*"
    return endpoint


def _connectable(endpoint: str) -> str:
    """A peer's bind endpoint (``tcp://*:2000``) as a connect endpoint (loopback)."""
    for wild in ("tcp://*:", "tcp://0.0.0.0:"):
        if endpoint.startswith(wild):
            return "tcp://127.0.0.1:" + endpoint[len(wild) :]
    return endpoint


@dataclass(frozen=True)
class RawPeerConfig:
    """One raw REQ/REP peer.

    ``request_format``: ``srsran_dummy`` or ``gnuradio_count`` (module docstring).
    ``tx_port``: the peer's TX REP endpoint (the emulator connects; ``tcp://*:N`` means
    loopback port N). ``rx_port``: the endpoint where the emulator binds the REP that the
    peer's RX REQ connects to. ``tx_stream`` / ``rx_stream``: the scenario streams fed by /
    delivered to the peer. ``sample_format``: ``fc32`` or ``sc16`` (full scale 32767).
    ``base_srate``: optional exact rate that must equal the streams' rate (no resampling).
    ``request_items``: items asked per TX request in ``gnuradio_count`` mode (default:
    :func:`default_request_items`, 8 pages of the TX stream within [4096, 65536]).
    ``reply_items``: samples per RX reply in ``srsran_dummy`` mode (default: the page size;
    at most 3,072,000).
    """

    name: str
    request_format: RequestFormat
    tx_port: str | None = None
    rx_port: str | None = None
    tx_stream: str | None = None
    rx_stream: str | None = None
    sample_format: Literal["fc32", "sc16"] = "fc32"
    base_srate: str | int | Fraction | None = None
    request_items: int | None = None
    reply_items: int | None = None

    def __post_init__(self) -> None:
        if self.request_format not in REQUEST_FORMATS:
            raise ConfigurationError(f"unknown request_format {self.request_format!r}")
        if self.sample_format not in ("fc32", "sc16"):
            raise ConfigurationError(f"unknown sample_format {self.sample_format!r}")
        if (self.tx_port is None) != (self.tx_stream is None):
            raise ConfigurationError("give tx_port and tx_stream together")
        if (self.rx_port is None) != (self.rx_stream is None):
            raise ConfigurationError("give rx_port and rx_stream together")
        if self.tx_port is None and self.rx_port is None:
            raise ConfigurationError("a raw peer needs tx_port or rx_port")
        for v in (self.request_items, self.reply_items):
            if v is not None and not 0 < int(v) <= SRSRAN_MAX_REPLY_SAMPLES:
                raise ConfigurationError(f"item counts must be in (0, {SRSRAN_MAX_REPLY_SAMPLES}]")

    @property
    def dtype(self) -> int:
        """Framing dtype of the raw samples."""
        return Dtype.CF32_LE if self.sample_format == "fc32" else Dtype.SC16_LE

    @property
    def bytes_per_sample(self) -> int:
        """8 (fc32) or 4 (sc16)."""
        return 8 if self.sample_format == "fc32" else 4

    @classmethod
    def from_device_args(
        cls, name: str, request_format: RequestFormat, args: str, **kw: Any
    ) -> RawPeerConfig:
        """Build from srsRAN-style device args (``"tx_port=tcp://*:2000,rx_port=
        tcp://localhost:2001,base_srate=23.04e6"``) as the peer sees them; ``kw`` gives
        the streams and the remaining fields."""
        d: dict[str, str] = {}
        for item in args.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                d[k.strip()] = v.strip()
        rx = d.get("rx_port")
        if rx is not None:
            rx = rx.replace("://localhost:", "://127.0.0.1:")
        return cls(
            name,
            request_format,
            tx_port=_connectable(d["tx_port"]) if "tx_port" in d else None,
            rx_port=rx,
            base_srate=d.get("base_srate"),
            **kw,
        )


class RawPeer:
    """The TX puller and RX server threads of one raw peer (module docstring)."""

    def __init__(self, server: ControlServer, config: RawPeerConfig) -> None:
        self.server = server
        self.config = config
        eng: Any = server.engine
        cfg = getattr(eng, "cfg", None)
        if getattr(cfg, "mode", "lockstep") != "lockstep":
            raise NotPermitted("raw REQ/REP peers need lockstep (timing is implicit in counts)")
        self.engine = eng
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._sockets: list[zmq.Socket[Any]] = []
        self.tx_samples = 0
        self.rx_samples = 0
        self.tx_requests = 0
        self.rx_requests = 0
        self.bad = 0
        self.tx_handle: StreamHandle | None = None
        self.rx_handle: StreamHandle | None = None
        self.rx_endpoint: str | None = None
        self.request_items: int | None = None
        self._rx_busy = False  # between an RX request and its reply
        self._rx_last_request = time.monotonic()

    # ------------------------------------------------------------------ setup
    def _spec(self, stream: str, direction: str) -> StreamSpec:
        configs: Mapping[str, Any] = getattr(self.engine, "configs", {}) or {}
        c = configs.get(stream)
        if c is None:
            raise ConfigurationError(f"raw peer {self.config.name}: unknown stream {stream!r}")
        if self.config.base_srate is not None and parse_exact(self.config.base_srate) != c.rate:
            raise ConfigurationError(
                f"raw peer {self.config.name}: base_srate {self.config.base_srate} != stream rate "
                f"{c.rate} (no resampling in release 1)"
            )
        return StreamSpec(stream, direction, float(c.fc_hz), c.rate)  # type: ignore[arg-type]

    def open(self) -> None:
        """Open the streams (RX from sample 0) and the sockets; call before the engine
        runs, so nothing is computed before the peer's streams exist."""
        ctx = self.server.ctx
        assert ctx is not None
        c = self.config
        if c.rx_port is not None and c.rx_stream is not None:
            self.rx_handle = self.engine.open_stream(self._spec(c.rx_stream, "rx"))
            self.engine.stream_cmd(
                self.rx_handle, StreamCmd("start_cont", stream_now=False, time_spec=TimeSpec(0, 0))
            )
            rep = ctx.socket(zmq.REP)
            self._sockets.append(rep)
            rep.setsockopt(zmq.LINGER, 0)
            rep.bind(_bindable(c.rx_port))
            self.rx_endpoint = rep.getsockopt_string(zmq.LAST_ENDPOINT)
            self._threads.append(
                threading.Thread(
                    target=self._rx_main, args=(rep,), daemon=True, name=f"sremu-raw-rx-{c.name}"
                )
            )
        if c.tx_port is not None and c.tx_stream is not None:
            self.tx_handle = self.engine.open_stream(self._spec(c.tx_stream, "tx"))
            self.request_items = int(
                c.request_items or default_request_items(c.request_format, self.tx_handle.page_size)
            )
            req = ctx.socket(zmq.REQ)
            self._sockets.append(req)
            req.setsockopt(zmq.LINGER, 0)
            req.connect(c.tx_port)
            self._threads.append(
                threading.Thread(
                    target=self._tx_main, args=(req,), daemon=True, name=f"sremu-raw-tx-{c.name}"
                )
            )

    def start(self) -> None:
        """Start the threads (after :meth:`open` and after the engine started)."""
        for th in self._threads:
            th.start()

    def stop(self) -> None:
        """Stop both threads and close the sockets (also when the threads never ran).
        When the run has ended, the peer may first read the RX samples the engine already
        computed for it (module docstring, "Ending a run")."""
        if any(th.is_alive() for th in self._threads) and self._run_ended():
            self._drain()
        self._stop.set()
        for th in self._threads:
            if th.is_alive():
                th.join(5.0)
        for sock in self._sockets:
            sock.close(0)

    def _run_ended(self) -> bool:
        """Every rate group reached its end and the engine did not fail."""
        groups = getattr(self.engine, "groups", None)
        if not groups or getattr(self.server, "engine_error", None) is not None:
            return False
        return all(g.end is not None and g.cursor >= g.end for g in groups)

    def _rx_queued(self) -> int:
        """RX samples computed for the peer and not yet read (0 when unknown)."""
        h = self.rx_handle
        rx = getattr(self.engine, "rx", None)
        if h is None or not isinstance(rx, Mapping):
            return 0
        return int(getattr(rx.get(h.spec.stream), "queued", 0) or 0)

    def _drain(self) -> None:
        """Wait until the peer has read every computed RX sample, has been idle for
        :data:`DRAIN_IDLE_S`, or :data:`DRAIN_MAX_S` passed."""
        if self.rx_handle is None:
            return
        t0 = time.monotonic()
        while time.monotonic() - t0 < DRAIN_MAX_S:
            if not self._rx_busy:
                if self._rx_queued() == 0:
                    return
                if time.monotonic() - self._rx_last_request > DRAIN_IDLE_S:
                    return
            time.sleep(0.002)

    def stats(self) -> dict[str, Any]:
        """Counts of this peer (``request_items``: the TX request size in use)."""
        return {
            "tx_samples": self.tx_samples,
            "rx_samples": self.rx_samples,
            "tx_requests": self.tx_requests,
            "rx_requests": self.rx_requests,
            "bad": self.bad,
            "rx_endpoint": self.rx_endpoint,
            "request_items": self.request_items,
        }

    def _bad(self, reason: str) -> None:
        self.bad += 1
        self.server.counters.bad(reason)

    # ------------------------------------------------------------------ TX
    def _need_tx(self, h: StreamHandle) -> bool:
        groups = getattr(self.engine, "groups", None)
        tx_head = getattr(self.engine, "tx_head", None)
        if not groups or not callable(tx_head):
            return True
        if any(h.spec.stream in g.waiting_on for g in groups):
            return True
        cur = max((g.cursor for g in groups if g.rate == h.spec.rate_sps), default=0)
        return int(tx_head(h)) - cur < self.server.config.tx_window_samples

    def _tx_main(self, req: zmq.Socket[Any]) -> None:
        c = self.config
        h = self.tx_handle
        assert h is not None
        n_ask = int(self.request_items or default_request_items(c.request_format, h.page_size))
        first = True
        try:
            while not self._stop.is_set():
                if not self._need_tx(h):
                    self._stop.wait(0.002)
                    continue
                req.send(encode_request(c.request_format, n_ask))
                self.tx_requests += 1
                while not self._stop.is_set() and not req.poll(50, zmq.POLLIN):
                    pass
                if self._stop.is_set():
                    return
                msg = req.recv()
                if len(msg) % c.bytes_per_sample:
                    self._bad("raw_size")
                    continue
                n = len(msg) // c.bytes_per_sample
                if n == 0:
                    self._stop.wait(0.002)
                    continue
                x = decode_payload(c.dtype, msg, n)
                md = TxMetadata(has_time_spec=first, time_spec=TimeSpec(0, 0), start_of_burst=first)
                try:
                    self.engine.submit_tx(h, x, md)
                except SremuError:
                    self._bad("rejected")
                    continue
                first = False
                self.tx_samples += n
        finally:
            req.close(0)

    # ------------------------------------------------------------------ RX
    def _collect(self, want: int) -> NDArray[np.complex64]:
        """Exactly ``want`` RX samples (fewer only when the run ended or on stop)."""
        h = self.rx_handle
        assert h is not None
        parts: list[NDArray[np.complex64]] = []
        got = 0
        while got < want and not self._stop.is_set():
            y, md = self.engine.read_rx(h, want - got, 0.05)
            if y.size:
                parts.append(y)
                got += int(y.shape[0])
                continue
            if md.error_code not in (ErrorCode.NONE, ErrorCode.TIMEOUT):
                self._bad("raw_rx_" + md.error_code.name.lower())
            if self.server.is_done():
                break
        if not parts:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def _rx_main(self, rep: zmq.Socket[Any]) -> None:
        c = self.config
        h = self.rx_handle
        assert h is not None
        default = int(c.reply_items or h.page_size)
        try:
            while not self._stop.is_set():
                if not rep.poll(50, zmq.POLLIN):
                    continue
                msg = rep.recv()
                self._rx_busy = True
                self._rx_last_request = time.monotonic()
                self.rx_requests += 1
                try:
                    want = parse_request(c.request_format, msg)
                except RawProtocolError as exc:
                    self._bad(exc.reason)
                    rep.send(b"")
                    self._rx_busy = False
                    continue
                n = default if want is None else min(int(want), SRSRAN_MAX_REPLY_SAMPLES)
                y = self._collect(n) if n > 0 else np.zeros(0, np.complex64)
                self.rx_samples += int(y.shape[0])
                rep.send(encode_payload(y, c.dtype))
                self._rx_busy = False
        finally:
            rep.close(0)
