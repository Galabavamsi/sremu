"""The sremu ``Device`` (DESIGN.md section 8.1; UHD-shaped, not UHD-pretending).

```python
dev = sremu.Device.inproc("scenarios/lab_like_A.yaml", mode="lockstep", backend="numba_cpu",
                          record="runs/r001")
# or: dev = sremu.Device.connect("tcp://127.0.0.1:5600")
tx = dev.get_tx_stream(StreamArgs(stream="ap.tx0", fc_hz=3.5e9, rate_sps="1e6", cpu="fc32"))
rx = dev.get_rx_stream(StreamArgs(stream="rx1.rx0", fc_hz=3.5e9, rate_sps="1e6", cpu="fc32"))
n = tx.send(buf, TxMetadata(has_time_spec=True, time_spec=TimeSpec(0, 1e-3),
                            start_of_burst=True, end_of_burst=True), timeout=0.1)
rx.issue_stream_cmd(StreamCmd(mode="num_done", num_samps=10_000, stream_now=False,
                              time_spec=TimeSpec(0, 1e-3)))
n = rx.recv(out, md := RxMetadata(), timeout=1.0)
dev.ris("ris0").set_config(states=s, time_spec=TimeSpec(0, 5e-3), transition=Linear("2e-5"))
dev.set_command_time(TimeSpec(0, 8e-3)); dev.node("rx1").set_position([1.2, 0.3, 0.0])
dev.clear_command_time()
```

A ``Device`` wraps any :class:`~sremu.contracts.EngineProtocol`: the WP6 engine
(:meth:`Device.inproc`), a remote session (:meth:`Device.connect`, WP8) or a test double
(:meth:`Device.from_engine`). Engine calls are serialized by one lock, so TX and RX may run
in different threads of one process (see :mod:`sremu.api.streams` for blocking).

**Commands** (``dev.ris(p)``, ``dev.node(n)``, ``dev.marker``) return the engine's
:class:`~sremu.contracts.CommandReceipt`; timing errors are codes, never exceptions
(``receipt.error`` is e.g. ``LATE_COMMAND`` with ``late_policy="reject"``,
``UNKNOWN_TARGET``, ``LOOKAHEAD_VIOLATION``). A command without ``time_spec`` uses the
command time (``set_command_time``) if set, else ``"asap"``.

**Recording** (``record=DIR``) writes a run directory (section 3.6): ``events.jsonl``,
SigMF recordings of every stream and ``run.sigmf-collection``
(:class:`sremu.record.sigmf.RunRecorder`); :meth:`Device.inproc` also writes
``manifest.json``, ``scenario.resolved.yaml`` (WP1 ``ManifestRecorder``, completed from the
engine's ``manifest_info()``) and ``metrics.jsonl``. With the WP6 engine
(:meth:`Device.inproc`) the engine feeds the TX/RX samples and metrics windows through its
sink interface (:class:`sremu.record.sigmf.EngineRecorderSink`). Otherwise
(:meth:`Device.from_engine`) the Device writes them from the API calls it sees: accepted TX
samples at their absolute indices and delivered RX samples. The index of accepted TX
samples comes from the engine's optional ``tx_head(h)`` method (contract change request
WP7-1) when it has one; otherwise the API places timed bursts and in-burst continuations
itself (R3) and refuses to record an untimed burst start (``NotPermitted``), whose index
only the engine knows. In both cases the Device keeps the command records current from
the engine's
optional ``command_log()`` (``(record, request)`` pairs, WP7-1) or ``records`` and attaches
the requests it issued itself, so ``sremu replay`` can re-inject them. For a command issued
inside the engine (an in-process or remote controller) the event log keeps the applied
configuration instead: from the WP6 sink's final-record callback, or from the engine's
optional ``panel_state(panel, t)`` once the command is applied.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Final, TypeVar

import numpy as np

from sremu.api.streams import RxStreamer, StreamArgs, TxStreamer
from sremu.contracts import (
    DEFAULT_CONTROL_ENDPOINT,
    CommandReceipt,
    CommandRecord,
    CommandRequest,
    CommandTime,
    Controller,
    EngineProtocol,
    LatePolicy,
    PanelId,
    RemoteControllerProxy,
    RisCapabilities,
    StreamCmd,
    StreamId,
    TransitionSpec,
    TxMetadata,
)
from sremu.errors import ConfigurationError, NotPermitted, StreamError
from sremu.record.sigmf import EngineRecorderSink, RecordSink, RunRecorder
from sremu.timebase import TimeSpec, fraction_to_str, to_exact

__all__ = [
    "DEFAULT_POLL_S",
    "Device",
    "NodeHandle",
    "RisHandle",
    "engine_command_log",
    "scenario_with_overrides",
]

DEFAULT_POLL_S: Final[float] = 0.025
"""Wait granularity of blocking calls, wall seconds."""

_T = TypeVar("_T")
TimeArg = TimeSpec | Fraction | int | str | float


def _engine_log(engine: object) -> tuple[Sequence[Any], bool]:
    """``(entries, paired)``: the engine's ``command_log()`` (``(record, request)`` pairs,
    ``paired`` True) or ``records`` (records only); an empty tuple when it has neither."""
    log = getattr(engine, "command_log", None)
    if callable(log):
        return log(), True
    recs = getattr(engine, "records", None)
    if recs is None:
        return (), False
    if callable(recs):
        recs = recs()
    return recs, False


def engine_command_log(engine: object) -> list[tuple[CommandRecord, CommandRequest | None]]:
    """Every command record of ``engine`` with its request when the engine provides one.

    Uses the optional ``command_log()`` (pairs) or ``records`` (records only; a property or
    a method); engines with neither yield an empty list.
    """
    entries, paired = _engine_log(engine)
    if paired:
        return [(r, q) for r, q in entries]
    return [(r, None) for r in entries]


_FINAL: Final = frozenset({"applied", "late_applied", "rejected", "superseded"})


def _exact_time(t: TimeArg) -> Fraction:
    if isinstance(t, TimeSpec):
        return t.exact()
    return to_exact(t, name="time_spec")


def _states_int16(states: object) -> np.ndarray:
    arr = np.asarray(states)
    if arr.ndim != 1:
        raise ConfigurationError(f"states must be 1-D, got shape {arr.shape}")
    integral = arr.dtype.kind in "iu" or (
        arr.dtype.kind == "f" and bool(np.all(np.isfinite(arr) & (arr == np.round(arr))))
    )
    if not integral:
        raise ConfigurationError("states must be integer state indices")
    if arr.size and (int(arr.min()) < -1 or int(arr.max()) > np.iinfo(np.int16).max):
        raise ConfigurationError("states must lie in [-1, 32767]")
    return np.ascontiguousarray(arr, dtype=np.int16)


class RisHandle:
    """Commands for one RIS panel (``dev.ris(panel)``)."""

    def __init__(self, device: Device, panel: PanelId) -> None:
        self._dev = device
        self.panel = panel

    @property
    def caps(self) -> RisCapabilities:
        """The panel's capabilities (``UnknownTarget`` style ``KeyError`` if absent)."""
        return self._dev.caps()[self.panel]

    def set_config(
        self,
        states: object | None = None,
        *,
        phases_rad: object | None = None,
        quantizer: str = "max_real",
        codebook_index: int | None = None,
        elements: str | Sequence[int] = "all",
        time_spec: TimeArg | None = None,
        at_sample: tuple[StreamId, int] | None = None,
        transition: TransitionSpec | None = None,
        late_policy: LatePolicy = "apply_on_arrival",
        cmd_id: str | None = None,
    ) -> CommandReceipt:
        """Timed ``ris.set_config``: exactly one of ``states`` ((N_ctrl,) state indices),
        ``phases_rad`` (quantized by the panel) or ``codebook_index``."""
        return self._dev.schedule(
            CommandRequest(
                cmd_id=cmd_id or self._dev._next_cmd_id(),
                op="ris.set_config",
                src=self._dev.src,
                target=self.panel,
                at=self._dev._when(time_spec, at_sample),
                states=None if states is None else _states_int16(states),
                phases_rad=None
                if phases_rad is None
                else np.ascontiguousarray(phases_rad, dtype=np.float64),
                quantizer=quantizer,
                codebook_index=codebook_index,
                elements=elements if isinstance(elements, str) else tuple(elements),
                transition=transition,
                late_policy=late_policy,
            )
        )

    def set_enabled(
        self,
        enabled: bool,
        *,
        time_spec: TimeArg | None = None,
        at_sample: tuple[StreamId, int] | None = None,
        late_policy: LatePolicy = "apply_on_arrival",
        cmd_id: str | None = None,
    ) -> CommandReceipt:
        """Counterfactual ``ris.set_enabled`` (RIS absent/present; logged as such)."""
        return self._dev.schedule(
            CommandRequest(
                cmd_id=cmd_id or self._dev._next_cmd_id(),
                op="ris.set_enabled",
                src=self._dev.src,
                target=self.panel,
                at=self._dev._when(time_spec, at_sample),
                enabled=bool(enabled),
                late_policy=late_policy,
            )
        )


class NodeHandle:
    """Timed geometry commands for one node (``dev.node(node)``)."""

    def __init__(self, device: Device, node: str) -> None:
        self._dev = device
        self.node = node

    def set_position(
        self,
        position_m: Sequence[float],
        *,
        time_spec: TimeArg | None = None,
        cmd_id: str | None = None,
    ) -> CommandReceipt:
        """``node.set_position``: hold ``position_m`` (3,) m from the command time on."""
        return self._dev.schedule(
            CommandRequest(
                cmd_id=cmd_id or self._dev._next_cmd_id(),
                op="node.set_position",
                src=self._dev.src,
                target=self.node,
                at=self._dev._when(time_spec, None),
                position_m=np.asarray(position_m, dtype=np.float64).reshape(-1),
            )
        )

    def set_trajectory(
        self,
        trajectory: Mapping[str, Any],
        *,
        time_spec: TimeArg | None = None,
        cmd_id: str | None = None,
    ) -> CommandReceipt:
        """``node.set_trajectory`` with a trajectory spec in scenario-schema form."""
        return self._dev.schedule(
            CommandRequest(
                cmd_id=cmd_id or self._dev._next_cmd_id(),
                op="node.set_trajectory",
                src=self._dev.src,
                target=self.node,
                at=self._dev._when(time_spec, None),
                payload=dict(trajectory),
            )
        )


class Device:
    """An emulated radio environment seen through a UHD-shaped API (module docstring).

    ``engine``: any ``EngineProtocol``. ``recorder``: a :class:`RecordSink`. The Device opens
    every stream in it and keeps its command records current (attaching the requests it
    issued); it also writes the TX/RX samples unless ``engine_samples`` says the engine
    feeds them (``Device.inproc``). ``manifest``: an optional WP1 ``ManifestRecorder``
    completed from the engine's ``manifest_info()`` (when present) and finished on
    :meth:`close`. ``src``: the source name of commands issued through this Device
    (``CommandRecord.src``).
    """

    def __init__(
        self,
        engine: EngineProtocol,
        *,
        recorder: RecordSink | None = None,
        engine_samples: bool = False,
        manifest: Any = None,
        src: str = "api",
        poll_s: float = DEFAULT_POLL_S,
        owns_engine: bool = False,
    ) -> None:
        self._engine = engine
        self._rec = recorder
        self._feed = recorder is not None and not engine_samples
        self._manifest = manifest
        self.src = src
        self._poll_s = float(poll_s)
        self._owns_engine = owns_engine
        self._cond = threading.Condition(threading.RLock())
        self._tx: dict[StreamId, TxStreamer] = {}
        self._rx: dict[StreamId, RxStreamer] = {}
        self._args: dict[StreamId, StreamArgs] = {}
        self._cmd_n = 0
        self._command_time: Fraction | None = None
        self._req_by_seq: dict[int, CommandRequest] = {}
        self._sync_seen = 0
        self._sync_pending: list[int] = []
        self.closed = False

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_engine(
        cls,
        engine: EngineProtocol,
        *,
        record: str | Path | None = None,
        iq: bool = True,
        epoch_datetime: Any = None,
        src: str = "api",
        poll_s: float = DEFAULT_POLL_S,
    ) -> Device:
        """Wrap an engine; ``record=DIR`` records the run at the API boundary."""
        rec = RunRecorder(record, iq=iq, epoch_datetime=epoch_datetime) if record else None
        return cls(engine, recorder=rec, src=src, poll_s=poll_s)

    @classmethod
    def inproc(
        cls,
        scenario: Any,
        *,
        mode: str | None = None,
        backend: str | None = None,
        record: str | Path | None = None,
        iq: bool = True,
        epoch_datetime: Any = None,
        src: str = "api",
        **engine_kw: Any,
    ) -> Device:
        """In-process WP6 emulator for a scenario file (or a ``ResolvedScenario``).

        ``mode`` and ``backend`` override the scenario's ``run.mode`` and
        ``numerics.backend``; ``engine_kw`` are further engine overrides (``EngineConfig``
        fields, ``traffic``, ``clock``). Overrides that are scenario fields (``run.*``,
        ``numerics.*``, ``seed``; ``traffic={}`` clears the scenario's generators) are
        written into the resolved scenario the run uses, so ``scenario.resolved.yaml`` and
        the manifest describe the run as executed. With ``record=DIR`` the run directory
        gets ``manifest.json``, ``scenario.resolved.yaml``, ``events.jsonl``,
        ``metrics.jsonl`` and (``iq``) SigMF recordings; the engine feeds the samples and
        metrics through its sink interface. Engine overrides that are not scenario fields
        (e.g. ``noise=False``) go to ``engine_overrides.json`` and a manifest assumption,
        and ``sremu replay`` applies them; one that is not plain data (a traffic source
        object) is named in the manifest assumptions as not replayable. ``epoch_datetime``
        (optional, timezone-aware) only sets the SigMF ``core:datetime`` of the captures
        (R2).
        """
        from sremu.record.replay import write_engine_overrides
        from sremu.scenario.load import ResolvedScenario, load_scenario

        resolved = scenario if isinstance(scenario, ResolvedScenario) else load_scenario(scenario)
        emu = _wp6_module()
        overrides = dict(engine_kw)
        if mode is not None:
            overrides["mode"] = mode
        if backend is not None:
            overrides["backend"] = backend
        resolved, rest = scenario_with_overrides(resolved, overrides)
        # build (and so validate) the engine before anything is written to the run directory;
        # the recorder sink is attached before any sample exists
        engine = emu.Emulator.from_scenario(resolved, **rest)
        rec = mrec = None
        if record:
            from sremu.record.manifest import ManifestRecorder

            run_dir = Path(record)
            mrec = ManifestRecorder(resolved, run_dir)
            rec = RunRecorder(run_dir, iq=iq, epoch_datetime=epoch_datetime)
            engine.add_sink(EngineRecorderSink(rec))
            saved, unsaved = write_engine_overrides(run_dir, rest)
            if saved:
                mrec.add_assumption(
                    "engine overrides outside the scenario (engine_overrides.json, applied "
                    "by sremu replay): " + ", ".join(f"{k}={v!r}" for k, v in saved.items())
                )
            for k in unsaved:
                mrec.add_assumption(
                    f"engine override {k!r} ({type(rest[k]).__name__}) is not plain data and "
                    "is not recorded; sremu replay cannot reproduce it"
                )
        return cls(
            engine,
            recorder=rec,
            engine_samples=rec is not None,
            manifest=mrec,
            src=src,
            owns_engine=True,
        )

    @classmethod
    def connect(cls, endpoint: str = DEFAULT_CONTROL_ENDPOINT, **kw: Any) -> Device:
        """Session with a served emulator (``sremu run --serve``). Needs WP8:
        ``sremu.transport.client.connect_engine(endpoint, **kw) -> EngineProtocol``."""
        try:
            client = importlib.import_module("sremu.transport.client")
        except ModuleNotFoundError as exc:
            if exc.name in ("sremu.transport.client", "sremu.transport"):
                raise NotImplementedError(
                    "Device.connect needs the WP8 transport (sremu.transport.client), "
                    "which is not available yet"
                ) from exc
            raise
        return cls(client.connect_engine(endpoint, **kw), owns_engine=True)

    # ------------------------------------------------------------------ plumbing
    @property
    def engine(self) -> EngineProtocol:
        """The wrapped engine."""
        return self._engine

    @property
    def recorder(self) -> RecordSink | None:
        """The run recorder, if recording."""
        return self._rec

    def _check_open(self) -> None:
        if self.closed:
            raise StreamError("the Device is closed")

    def _retry(self, fn: Callable[[], _T], ready: Callable[[_T], bool], timeout: float) -> _T:
        """Call ``fn`` under the lock until ``ready``; between calls wait for progress from
        other threads (woken by their calls, else every ``poll_s``) until ``timeout`` wall
        seconds have passed.

        The deadline is a ``threading.Timer`` (this module reads no clock; I7), so wake-ups
        by other threads' calls do not shorten the wait: ``fn`` is retried until the timer
        fires, then once more."""
        with self._cond:
            res = fn()
            if ready(res) or not timeout > 0:
                return res
            expired = threading.Event()

            def expire() -> None:
                expired.set()
                with self._cond:
                    self._cond.notify_all()

            timer = threading.Timer(float(timeout), expire)
            timer.daemon = True
            timer.start()
            try:
                while True:
                    self._cond.wait(self._poll_s)
                    last = expired.is_set()  # read before fn: one attempt after the deadline
                    res = fn()
                    if ready(res) or last:
                        return res
            finally:
                timer.cancel()

    def notify(self) -> None:
        """Wake blocked calls (after inputs arrived from outside this Device)."""
        with self._cond:
            self._cond.notify_all()

    def _next_cmd_id(self) -> str:
        with self._cond:
            self._cmd_n += 1
            return f"{self.src}-{self._cmd_n}"

    def _when(
        self, time_spec: TimeArg | None, at_sample: tuple[StreamId, int] | None
    ) -> CommandTime:
        if at_sample is not None:
            if time_spec is not None:
                raise ConfigurationError("give time_spec or at_sample, not both")
            return CommandTime.at_sample(at_sample[0], at_sample[1])
        if time_spec is not None:
            return CommandTime.at(_exact_time(time_spec))
        if self._command_time is not None:
            return CommandTime.at(self._command_time)
        return CommandTime.asap()

    # ------------------------------------------------------------------ streams
    def _open(self, args: StreamArgs, direction: str) -> TxStreamer | RxStreamer:
        self._check_open()
        if not isinstance(args, StreamArgs):
            raise TypeError("expected StreamArgs")
        with self._cond:
            table: dict[StreamId, Any] = self._tx if direction == "tx" else self._rx
            if args.stream in table:
                if self._args[args.stream] != args:
                    raise ConfigurationError(
                        f"stream {args.stream!r} is already open with {self._args[args.stream]}"
                    )
                return table[args.stream]
            spec = args.spec("tx" if direction == "tx" else "rx")
            h = self._engine.open_stream(spec)
            if self._rec is not None:
                self._rec.open_stream(spec)
            s = TxStreamer(self, h) if direction == "tx" else RxStreamer(self, h)
            table[args.stream] = s
            self._args[args.stream] = args
            self._cond.notify_all()
            return s

    def get_tx_stream(self, args: StreamArgs) -> TxStreamer:
        """Open (or return the already open) TX stream ``args.stream``."""
        s = self._open(args, "tx")
        assert isinstance(s, TxStreamer)
        return s

    def get_rx_stream(self, args: StreamArgs) -> RxStreamer:
        """Open (or return the already open) RX stream ``args.stream``."""
        s = self._open(args, "rx")
        assert isinstance(s, RxStreamer)
        return s

    def _send(self, tx: TxStreamer, x: np.ndarray, md: TxMetadata) -> int:
        self._check_open()
        with self._cond:
            k0 = tx._start_index(md)
            recording = self._feed and self._rec is not None and x.size > 0
            tx_head = getattr(self._engine, "tx_head", None)
            if recording and k0 is None and not callable(tx_head):
                raise NotPermitted(
                    "recording an untimed TX burst start needs an engine with tx_head() "
                    "(CCR WP7-1); send a timed burst instead"
                )
            n = int(self._engine.submit_tx(tx.handle, x, md))
            if n and callable(tx_head):
                # the engine's placement is authoritative (e.g. a paced continuation that
                # slipped to the frontier after an underflow)
                k0 = int(tx_head(tx.handle)) - n
            if recording and n:
                assert self._rec is not None
                assert k0 is not None
                self._rec.write(tx.stream, k0, x[:n])
            tx._after_send(k0, int(x.size), n, md)
            self._cond.notify_all()
            return n

    def _stream_cmd(self, rx: RxStreamer, cmd: StreamCmd) -> None:
        self._check_open()
        with self._cond:
            self._engine.stream_cmd(rx.handle, cmd)
            self._cond.notify_all()

    def _record_rx(self, rx: RxStreamer, k0: int, y: np.ndarray) -> None:
        if self._feed and self._rec is not None:
            self._rec.write(rx.stream, k0, y)

    # ------------------------------------------------------------------ commands
    def ris(self, panel: PanelId) -> RisHandle:
        """Command handle of RIS ``panel``."""
        return RisHandle(self, panel)

    def node(self, node: str) -> NodeHandle:
        """Command handle of ``node``."""
        return NodeHandle(self, node)

    def set_command_time(self, time_spec: TimeArg) -> None:
        """Time later commands (without their own ``time_spec``) at ``time_spec``."""
        self._command_time = _exact_time(time_spec)

    def clear_command_time(self) -> None:
        """Back to ``"asap"`` for commands without a ``time_spec``."""
        self._command_time = None

    def get_command_time(self) -> TimeSpec | None:
        """The current command time, or None."""
        return None if self._command_time is None else TimeSpec.from_exact(self._command_time)

    def marker(
        self, label: str, *, time_spec: TimeArg | None = None, **fields: Any
    ) -> CommandReceipt:
        """``rec.marker``: an annotation in the recording (``payload = {label, **fields}``)."""
        return self.schedule(
            CommandRequest(
                cmd_id=self._next_cmd_id(),
                op="rec.marker",
                src=self.src,
                target="rec",
                at=self._when(time_spec, None),
                payload={"label": label, **fields},
            )
        )

    def schedule(self, req: CommandRequest) -> CommandReceipt:
        """Hand a command to the engine and return its receipt (codes, never raises for
        timing errors)."""
        self._check_open()
        with self._cond:
            receipt = self._engine.schedule(req)
            if receipt.seq is not None:
                self._req_by_seq[int(receipt.seq)] = req
            else:
                self._match_rejected(req)
            self._sync_records()
            self._cond.notify_all()
            return receipt

    def _match_rejected(self, req: CommandRequest) -> None:
        cand = [
            int(r.seq)
            for r, _ in engine_command_log(self._engine)
            if r.cmd_id == req.cmd_id and r.status == "rejected" and r.seq not in self._req_by_seq
        ]
        if cand:
            self._req_by_seq[max(cand)] = req

    def _sync_records(self) -> None:
        """Feed new or changed command records, with the requests this Device issued, to
        the recorder (the event log keeps one line per change). Records already in a final
        state (applied, late_applied, rejected, superseded) are not looked at again."""
        if self._rec is None:
            return
        entries, paired = _engine_log(self._engine)
        todo = self._sync_pending + list(range(self._sync_seen, len(entries)))
        self._sync_seen = len(entries)
        pending = []
        panel_state = getattr(self._engine, "panel_state", None)
        for i in todo:
            rec, req = entries[i] if paired else (entries[i], None)
            if req is None:
                req = self._req_by_seq.get(rec.seq)
            applied = None
            if (
                req is None
                and callable(panel_state)
                and rec.status in ("applied", "late_applied")
                and rec.t_applied is not None
                and rec.op.startswith("ris.")
            ):
                # a command issued inside the engine (e.g. an in-process controller): keep
                # the configuration after the switch as the replay fallback
                applied = panel_state(rec.panel, rec.t_applied)
            self._rec.on_command(rec, req, applied)
            if rec.status not in _FINAL:
                pending.append(i)
        self._sync_pending = pending

    def register_controller(self, ctrl: Controller | RemoteControllerProxy) -> None:
        """Register a barrier controller with the engine."""
        self._check_open()
        with self._cond:
            self._engine.register_controller(ctrl)

    # ------------------------------------------------------------------ queries
    def time_now(self) -> TimeSpec:
        """Virtual time through which every RX stream has been computed."""
        with self._cond:
            return TimeSpec.from_exact(self._engine.time_now())

    get_time_now = time_now

    def caps(self) -> Mapping[PanelId, RisCapabilities]:
        """Capabilities of every panel."""
        with self._cond:
            return self._engine.caps()

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Finish the recording (meta files, collection, manifest); idempotent."""
        if self.closed:
            return
        with self._cond:
            if self._owns_engine:  # stop the engine first: no sink callback after this
                close = getattr(self._engine, "close", None)
                if callable(close):
                    close()
            self._sync_records()
            if self._rec is not None:
                self._rec.close()
            if self._manifest is not None:
                _complete_manifest(self._manifest, self._engine)
                self._manifest.finish()
            self.closed = True
            self._cond.notify_all()

    def __enter__(self) -> Device:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


_SCENARIO_FIELD: Final[dict[str, tuple[str, ...]]] = {
    "backend": ("numerics", "backend"),
    "page_target_s": ("numerics", "page_target_s"),
    "verify_interp": ("numerics", "verify_interp"),
    "interp_rel_error": ("numerics", "interp_rel_error"),
    "t_u_max_s": ("numerics", "t_u_max_s"),
    "t_u_s": ("numerics", "t_u_s"),
    "fd_error_target_db": ("numerics", "fd_error_target_db"),
    "narrowband_phase_rad": ("numerics", "narrowband_phase_rad"),
    "deterministic": ("numerics", "deterministic"),
    "mode": ("run", "mode"),
    "coupling": ("run", "coupling"),
    "duration_s": ("run", "duration_s"),
    "late_policy": ("run", "late_policy"),
    "on_stall": ("run", "on_stall"),
    "stall_timeout_s": ("run", "stall_timeout_s"),
    "d_lag_s": ("run", "paced", "d_lag_s"),
    "slacks_s": ("run", "paced", "slacks_s"),
    "seed": ("seed",),
}
"""Engine overrides that are scenario fields (section 9.2), with their scenario path."""


def _scenario_json(v: Any) -> Any:
    if isinstance(v, Fraction):
        return fraction_to_str(v)
    if isinstance(v, list | tuple):
        return [_scenario_json(x) for x in v]
    return v


def scenario_with_overrides(
    resolved: Any, overrides: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Split engine ``overrides`` for a ``ResolvedScenario``: those that are scenario fields
    (:data:`_SCENARIO_FIELD`; ``traffic={}`` clears the scenario's generators) are written
    into a re-validated copy of the scenario, which is returned with the remaining
    overrides. Without such overrides the scenario is returned unchanged."""
    import dataclasses

    from sremu.scenario.models import Scenario

    mapped = {k: v for k, v in overrides.items() if k in _SCENARIO_FIELD}
    rest = {k: v for k, v in overrides.items() if k not in _SCENARIO_FIELD}
    traffic = rest.get("traffic")
    clear_traffic = "traffic" in rest and isinstance(traffic, Mapping) and not traffic
    if clear_traffic:
        del rest["traffic"]
    if not mapped and not clear_traffic:
        return resolved, rest
    d = resolved.scenario.model_dump(mode="json")
    for key, value in mapped.items():
        *parents, leaf = _SCENARIO_FIELD[key]
        obj = d
        for p in parents:
            if not isinstance(obj.get(p), dict):
                obj[p] = {}
            obj = obj[p]
        obj[leaf] = _scenario_json(value)
    if clear_traffic:
        d["traffic"] = []
    try:
        sc = Scenario.model_validate(d)
    except ValueError as exc:
        raise ConfigurationError(
            f"engine overrides {sorted(mapped)} are not valid scenario values: {exc}"
        ) from None
    return dataclasses.replace(resolved, scenario=sc), rest


def _wp6_module() -> Any:
    """The WP6 module ``sremu.engine.emulator``, or ``NotImplementedError``."""
    try:
        return importlib.import_module("sremu.engine.emulator")
    except ModuleNotFoundError as exc:
        if exc.name in ("sremu.engine.emulator", "sremu.engine"):
            raise NotImplementedError(
                "Device.inproc needs the WP6 engine (sremu.engine.emulator), "
                "which is not available yet"
            ) from exc
        raise


def _complete_manifest(mrec: Any, engine: object) -> None:
    """Report the engine's run facts into a WP1 ``ManifestRecorder`` (section 3.6)."""
    info_fn = getattr(engine, "manifest_info", None)
    info: Mapping[str, Any] = info_fn() if callable(info_fn) else {}
    counters = info.get("counters", getattr(engine, "counters", None))
    if isinstance(counters, Mapping):
        mrec.add_counters({k: int(v) for k, v in counters.items()})
    if info.get("nondeterministic_inputs", getattr(engine, "nondeterministic_inputs", False)):
        mrec.mark_nondeterministic()
    if info.get("oracle_csi_used", getattr(engine, "oracle_csi_used", False)):
        mrec.mark_oracle_csi_used()
    for stream, size in dict(info.get("page_size", {})).items():
        mrec.set_page_size(stream, int(size))
    vmax = info.get("verify_interp_max_rel")
    if vmax is not None:
        mrec.update_verify_interp(float(vmax))
    pol = dict(info.get("snapshot_policy", {}))
    t_u = pol.get("t_u_s_used", pol.get("t_u_s"))
    tiles = {str(p): int(np.prod(v)) for p, v in dict(pol.get("tiles", {})).items()}
    bound = pol.get("total_bound_rel")
    for link, dec in dict(info.get("link_decisions", {})).items():
        d = dict(dec)
        mrec.set_link(
            link,
            {
                "narrowband": d.pop("narrowband", None),
                "fd_filter": d.pop("fd_filter", None),
                "t_u_s": t_u,
                "tiles": tiles,
                "interp_error_bound": None if bound is None else float(bound),
                "details": d,
            },
        )
