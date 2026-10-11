"""The emulator: ``EngineProtocol`` over the lockstep/paced scheduler (DESIGN.md sections 4, 7).

:class:`Emulator` wires a ``PropagationModel`` (WP2 ``ScenePropagation`` or any model of
the contract, e.g. ``FakePropagation``), the render backends and impairments of WP5, the
command semantics of :mod:`.commands`, the barriers of :mod:`.barriers` and the pacer of
:mod:`.pacer`.

* **Backends (7.1).** ``numba_cpu`` (default) or ``numpy_ref`` for rendering and noise; the
  choice is fixed per emulator (mixing would break I5/I2a).
* **Rate groups (4.3).** RX streams with equal rate and page size share a cursor; pages are
  ``P_r = clamp(2**round(log2(f_r*page_target_s)), 64, 16384)`` on the absolute grid.
* **Lockstep (4.8).** :meth:`Emulator.step` advances every group by at most one page portion
  (skipping inactive pages, D-6), runs due barriers and updates command statuses. In-process
  API reads pull the scheduler synchronously; :meth:`Emulator.run` drives a headless run
  with stall detection (``stall_timeout_s`` wall; ``on_stall: wait | zero_fill``).
* **Paced (4.10).** :meth:`Emulator.run_paced` releases each page at
  ``R_j = t0 + k1/f + D_lag``, zero-fills missing TX data (``UNDERFLOW``,
  ``underflow_slip_samples``) and counts deadline misses per slack.
* **I7.** Wall time is read only for metrics, stall timeouts and pacing (``self.clock``);
  it never enters data-plane arithmetic, so a monkeypatched clock cannot change the output
  (RT-15).
* **Geometry events** (``node.set_position``/``node.set_trajectory``) are applied to the
  model before any sample of the snapshot interval they split is computed (on a grid:
  up to the end of the grid interval holding the furthest page end), so on-time events are
  deterministic for every page size; pending RIS commands at or after the event are
  re-mapped (I3). An event acts on the samples ``n >= ceil(t f)`` (geometry at the sample
  instants), which its record logs. An event earlier than one already applied is late
  (moved to that time; the model takes events in time order).

Records (one :class:`~sremu.contracts.CommandRecord` per command), RX/TX samples and
metrics windows go to optional sinks (:class:`EngineSink`), so the recording layer (WP7) and
the CLI can persist them without the engine importing them.
"""

from __future__ import annotations

import bisect
import math
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sremu.channel.fd import choose_fd
from sremu.channel.interp import SnapshotPolicy, choose_policy_for_model
from sremu.channel.ring import ring_capacity
from sremu.contracts import (
    AsyncMetadata,
    BackendName,
    CommandReceipt,
    CommandRecord,
    CommandRequest,
    Controller,
    Coupling,
    EventPriority,
    EventReference,
    LatePolicy,
    Observation,
    PanelConfig,
    PanelId,
    PropagationModel,
    RemoteControllerProxy,
    RisCapabilities,
    RunMode,
    RxMetadata,
    SegmentJob,
    StreamCmd,
    StreamHandle,
    StreamId,
    StreamSpec,
    TransitionSpec,
    TxMetadata,
)
from sremu.errors import (
    AsyncCode,
    ConfigurationError,
    ErrorCode,
    NondeterminismWarning,
    NonPhysicalError,
    SremuTimeout,
    StreamError,
    UnknownTarget,
)
from sremu.metrics.rt import Counters, RtfMeter
from sremu.timebase import (
    DEFAULT_PAGE_TARGET_S,
    ceil_ticks,
    fraction_to_str,
    from_ticks,
    page_size,
    parse_exact,
    to_ticks,
)

from .barriers import BarrierManager, ControllerSlot, action_request, csi_view
from .commands import CommandManager, PanelSpec, RxPort, panel_specs_from_model
from .events import EventHeap
from .pacer import Pacer, SystemClock, WallClock
from .scheduler import JobBuilder, RateGroup, SnapshotManager, Step, advance_group
from .streams import (
    RxStream,
    StreamConfig,
    TrafficSource,
    TxStream,
    stream_configs_from_model,
)

__all__ = ["Emulator", "EngineConfig", "EngineSink", "RunSummary", "StallReport"]


# =========================================================================== configuration


@dataclass(frozen=True)
class EngineConfig:
    """Engine settings (scenario ``numerics`` and ``run``, section 9.2).

    Exact quantities (``page_target_s``, ``duration_s``, ``t_u_max_s``, ``t_u_s``,
    ``d_lag_s``, ``slacks_s``) accept Fractions, ints or exact strings.
    ``render_tiles``: ``"auto"`` applies the section 4.4 tile policy to the model,
    ``"model"`` keeps the model's render tiles (the snapshot period still follows the
    policy). ``always_active``: compute every RX page (headless runs, recording);
    otherwise pages no consumer needs are skipped (D-6). ``unopened_tx``: an unopened TX
    stream without a traffic source is silent or never committed.
    ``external_inputs``: another thread or process may still supply inputs (stalls then
    wait instead of raising).
    """

    backend: BackendName = "numba_cpu"
    page_target_s: Fraction = Fraction(1, 2000)
    mode: RunMode = "lockstep"
    coupling: Coupling = "barrier"
    duration_s: Fraction | None = None
    late_policy: LatePolicy = "apply_on_arrival"
    ris_event_reference: EventReference = "ris"
    noise: bool = True
    verify_interp: bool = True
    interp_rel_error: float = 1e-3
    t_u_max_s: Fraction = Fraction(1, 100)
    t_u_s: Fraction | None = None
    fd_error_target_db: float = -60.0
    narrowband_phase_rad: float = 0.05
    render_tiles: Literal["auto", "model"] = "auto"
    on_stall: Literal["wait", "zero_fill"] = "wait"
    stall_timeout_s: float = 2.0
    d_lag_s: Fraction = Fraction(1, 1000)
    slacks_s: tuple[Fraction, ...] = (Fraction(0), Fraction(1, 1000), Fraction(5, 1000))
    rx_queue_pages: int = 64
    seed: int = 0
    deterministic: bool = True
    unopened_tx: Literal["silent", "wait"] = "silent"
    always_active: bool = False
    external_inputs: bool = False

    def __post_init__(self) -> None:
        for name in ("page_target_s", "t_u_max_s", "d_lag_s"):
            object.__setattr__(self, name, parse_exact(getattr(self, name), name=name))
        for name in ("duration_s", "t_u_s"):
            v = getattr(self, name)
            if v is not None:
                object.__setattr__(self, name, parse_exact(v, name=name))
        object.__setattr__(
            self, "slacks_s", tuple(parse_exact(s, name="slack") for s in self.slacks_s)
        )
        if self.backend not in ("numpy_ref", "numba_cpu"):
            raise ConfigurationError(f"unknown backend {self.backend!r} (numpy_ref | numba_cpu)")
        if self.mode not in ("lockstep", "paced"):
            raise ConfigurationError(f"unknown mode {self.mode!r}")
        if self.mode == "lockstep" and self.coupling == "async":
            raise ConfigurationError(
                "lockstep + async coupling is rejected (section 4.9 coupling matrix)"
            )

    @classmethod
    def from_scenario(cls, scenario: Mapping[str, Any], **overrides: Any) -> EngineConfig:
        """Settings of a scenario mapping (``model_dump`` of the pydantic model)."""
        num = dict(scenario.get("numerics") or {})
        run = dict(scenario.get("run") or {})
        paced = dict(run.get("paced") or {})
        kw: dict[str, Any] = {
            "backend": num.get("backend", "numba_cpu"),
            "page_target_s": num.get("page_target_s", DEFAULT_PAGE_TARGET_S),
            "verify_interp": bool(num.get("verify_interp", True)),
            "interp_rel_error": float(num.get("interp_rel_error", 1e-3)),
            "t_u_max_s": num.get("t_u_max_s", Fraction(1, 100)),
            "t_u_s": num.get("t_u_s"),
            "fd_error_target_db": float(num.get("fd_error_target_db", -60.0)),
            "narrowband_phase_rad": float(num.get("narrowband_phase_rad", 0.05)),
            "deterministic": bool(num.get("deterministic", True)),
            "mode": run.get("mode", "lockstep"),
            "coupling": run.get("coupling", "barrier"),
            "duration_s": run.get("duration_s"),
            "late_policy": run.get("late_policy", "apply_on_arrival"),
            "on_stall": run.get("on_stall", "wait"),
            "stall_timeout_s": float(run.get("stall_timeout_s", 2.0)),
            "d_lag_s": paced.get("d_lag_s", Fraction(1, 1000)),
            "slacks_s": tuple(paced.get("slacks_s", ("0", "1e-3", "5e-3"))),
            "seed": int(scenario.get("seed", 0)),
        }
        for k in ("page_target_s", "t_u_max_s", "t_u_s", "duration_s", "d_lag_s"):
            v = kw.get(k)
            if isinstance(v, float):  # pydantic dumps exact fields as Fraction; never floats
                raise ConfigurationError(f"{k} must be exact, got float {v!r}")
        kw.update(overrides)
        return cls(**kw)


class EngineSink:
    """Optional callbacks for recording and telemetry (override what you need)."""

    def on_record(self, rec: CommandRecord, cfg: PanelConfig | None) -> None:
        """A command record became final (applied, late_applied, rejected, superseded)."""

    def on_rx(self, stream: StreamId, k0: int, d: NDArray[np.complex64]) -> None:
        """RX samples [k0, k0 + n) were computed (digital, complex64)."""

    def on_tx(self, stream: StreamId, k0: int, x: NDArray[np.complex64]) -> None:
        """A client submitted TX samples at [k0, k0 + n) (digital, complex64)."""

    def on_metrics(self, window: Mapping[str, Any]) -> None:
        """A 100 ms metrics window closed."""

    def on_stall(self, report: StallReport) -> None:
        """A stall (deadlock) report was written."""


@dataclass(frozen=True)
class StallReport:
    """Deadlock report (section 4.8): which RX waits on which TX, pending barriers."""

    wall_s: float
    groups: tuple[dict[str, Any], ...]
    barriers: tuple[dict[str, Any], ...]
    policy: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict."""
        return {
            "wall_s": self.wall_s,
            "groups": list(self.groups),
            "barriers": list(self.barriers),
            "policy": self.policy,
        }


@dataclass
class RunSummary:
    """What a run produced: virtual time per stream, counters, RTF and flags."""

    samples: dict[StreamId, int]
    counters: dict[str, int]
    rtf: dict[str, Any]
    nondeterministic_inputs: bool
    oracle_csi_used: bool
    stalls: int
    verify_interp_max_rel: float | None
    paced: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# =========================================================================== emulator


def _render_fn(backend: BackendName) -> Callable[[SegmentJob, NDArray[np.complex128]], None]:
    if backend == "numpy_ref":
        from sremu.channel import render_numpy

        return render_numpy.render
    from sremu.channel import render_numba

    return render_numba.render


class Emulator:
    """In-process emulator implementing ``EngineProtocol`` (see the module docstring).

    ``model``: the propagation model. ``streams``: per-stream engine settings (default:
    every model stream with default front end, no LO offsets). ``panels``: panel specs
    (default: from the model). ``traffic``: in-process TX generators. ``sinks``: recording
    and telemetry callbacks. ``clock``: wall clock for metrics, stalls and pacing.
    """

    def __init__(
        self,
        model: PropagationModel,
        *,
        streams: Mapping[StreamId, StreamConfig] | None = None,
        panels: Mapping[PanelId, PanelSpec] | None = None,
        config: EngineConfig | None = None,
        traffic: Mapping[StreamId, TrafficSource] | None = None,
        sinks: Sequence[EngineSink] = (),
        clock: WallClock | None = None,
        legacy_block_s: Fraction | None = None,
        allow_nonphysical: bool = False,
    ) -> None:
        if legacy_block_s is not None and not allow_nonphysical:
            raise NonPhysicalError(
                "legacy block-constant geometry needs allow_nonphysical=True "
                "(use sremu.legacy.block_constant.block_constant_emulator)"
            )
        block_s = legacy_block_s
        cfg = config or EngineConfig()
        self.cfg = cfg
        self.clock: WallClock = clock or SystemClock()
        self.sinks: list[EngineSink] = list(sinks)
        self._sinks_want_rx = self._sinks_record_rx()
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._kicks = 0  # input notifications (see _kick)
        self.configs: dict[StreamId, StreamConfig] = dict(
            streams if streams is not None else stream_configs_from_model(model)
        )
        rx_cfgs = [c for c in self.configs.values() if c.direction == "rx"]
        tx_cfgs = [c for c in self.configs.values() if c.direction == "tx"]
        max_rate = max((c.rate for c in rx_cfgs), default=Fraction(1))
        # ---------------------------------------------------------- snapshot policy (4.4)
        self.policy: SnapshotPolicy = choose_policy_for_model(
            model,
            max_rate,
            eps=cfg.interp_rel_error,
            t_u_max_s=cfg.t_u_max_s,
            t_u_scenario_s=cfg.t_u_s,
        )
        if cfg.render_tiles == "auto" and self.policy.tiles:
            tiles = {p: (tp.tiles_u, tp.tiles_v) for p, tp in self.policy.tiles.items()}
            model = model.with_render_tiles(tiles)
        self.model = model
        t_u = self.policy.t_u_s
        if t_u is None and cfg.t_u_s is not None:
            t_u = cfg.t_u_s
        self.t_u: Fraction | None = t_u
        snap0 = model.snapshot(Fraction(0))
        n_tiles = {pid: rc.n_tiles for pid, rc in snap0.ris.items()}
        # ---------------------------------------------------------- commands
        specs = dict(panels) if panels is not None else panel_specs_from_model(model)
        self.rx_ports = [RxPort(c.stream, c.rate) for c in rx_cfgs]
        static = self.t_u is None and block_s is None
        self.commands = CommandManager(
            specs,
            self.rx_ports,
            model.ris_delay_s,
            n_tiles,
            reference=cfg.ris_event_reference,
            static_delays=static,
        )
        # ---------------------------------------------------------- streams
        traffic = dict(traffic or {})
        for sid in traffic:
            if sid not in self.configs or self.configs[sid].direction != "tx":
                raise ConfigurationError(f"traffic source for unknown TX stream {sid!r}")
        pages = {c.rate: page_size(c.rate, cfg.page_target_s) for c in self.configs.values()}
        structure = model.combine(snap0, {})
        self.tx: dict[StreamId, TxStream] = {}
        for c in tx_cfgs:
            tau_max = max(
                (
                    float(lp.delay_s.max())
                    for lk, lp in structure.items()
                    if lk.startswith(c.stream + "->") and lp.n_paths
                ),
                default=0.0,
            )
            fd = choose_fd(c.bandwidth_hz, c.rate, cfg.fd_error_target_db)
            cap = ring_capacity(2.0 * tau_max, c.rate, fd.taps, pages[c.rate])
            self.tx[c.stream] = TxStream(
                c,
                cap,
                seed=cfg.seed,
                backend=cfg.backend,
                source=traffic.get(c.stream),
                unopened=cfg.unopened_tx,
            )
        self.rx: dict[StreamId, RxStream] = {}
        for c in rx_cfgs:
            p = pages[c.rate]
            self.rx[c.stream] = RxStream(
                c,
                seed=cfg.seed,
                backend=cfg.backend,
                noise_on=cfg.noise,
                history=max(4 * p, 4096),
                queue_samples=cfg.rx_queue_pages * p,
            )
        # ---------------------------------------------------------- groups
        by: dict[tuple[Fraction, int], RateGroup] = {}
        for c in rx_cfgs:
            key = (c.rate, pages[c.rate])
            g = by.setdefault(key, RateGroup(c.rate, pages[c.rate], []))
            g.rx.append(c.stream)
        self.groups: list[RateGroup] = list(by.values())
        for g in self.groups:
            g.tx = [c.stream for c in tx_cfgs if c.rate == g.rate]
            if cfg.duration_s is not None:
                g.end = ceil_ticks(cfg.duration_s, g.rate)
        self._group_of_rx = {r: g for g in self.groups for r in g.rx}
        # ---------------------------------------------------------- scheduler pieces
        self.snaps = SnapshotManager(model, t_u=self.t_u, block_s=block_s)
        self._render = _render_fn(cfg.backend)
        self.builder = JobBuilder(
            model,
            self.snaps,
            self.commands,
            self.tx,
            self.configs,
            render=self._render,
            fd_target_db=cfg.fd_error_target_db,
            narrowband_eps=cfg.narrowband_phase_rad,
            verify=cfg.verify_interp and block_s is None,
        )
        self.barriers = BarrierManager(cfg.seed, mode=cfg.mode, coupling=cfg.coupling)
        self.events = EventHeap()
        self.counters = Counters()
        self.nondeterministic_inputs = False
        self.oracle_csi_used = False
        self.stall_reports: list[StallReport] = []
        self._handles: dict[int, StreamHandle] = {}
        self._by_name: dict[StreamId, int] = {}
        self._next_id = 1
        self._produced: dict[StreamId, int] = dict.fromkeys(self.rx, 0)
        self.rtf = RtfMeter({r: self.configs[r].rate for r in self.rx}) if self.rx else None
        self._node_records: list[CommandRecord] = []
        self._node_late: set[int] = set()  # ids of node records moved by a late policy
        # time of the latest geometry event applied to the model (events are applied ahead
        # of the frontier, up to the end of the snapshot interval they fall in)
        self._geo_applied: Fraction | None = None
        self._geo_applied_node: dict[str, Fraction] = {}  # the same, per node
        self._req_of: dict[int, CommandRequest | None] = {}
        self._block_s = block_s
        self._on_page: list[Callable[[StreamId, int, int], None]] = []
        self._metrics_window = 0
        self._metrics_counters = dict(self.counters)
        self._pacer: Pacer | None = None
        self._paced_active = False
        self._closed = False

    # ================================================================== construction
    @classmethod
    def from_scenario(
        cls,
        scenario: Any,
        *,
        base_dir: str | Path | None = None,
        sinks: Sequence[EngineSink] = (),
        clock: WallClock | None = None,
        traffic: Mapping[StreamId, TrafficSource] | None = None,
        **overrides: Any,
    ) -> Emulator:
        """Build from a scenario: a path, a ``ResolvedScenario``, the pydantic model or a
        mapping with the section 9.2 field names. ``overrides`` replace
        :class:`EngineConfig` fields; ``traffic`` replaces the scenario's generators."""
        from sremu.physics.propagation import from_scenario as model_from_scenario

        from .streams import stream_configs_from_scenario, traffic_from_scenario

        if isinstance(scenario, str | Path):
            from sremu.scenario.load import load_scenario

            scenario = load_scenario(scenario)
        if hasattr(scenario, "scenario") and hasattr(scenario, "base_dir"):
            base_dir = scenario.base_dir if base_dir is None else base_dir
            scenario = scenario.scenario
        d = (
            scenario.model_dump(mode="python")
            if hasattr(scenario, "model_dump")
            else dict(scenario)
        )
        bdir = None if base_dir is None else Path(base_dir)
        cfg = EngineConfig.from_scenario(d, **overrides)
        model = model_from_scenario(d, base_dir=bdir, backend=cfg.backend)
        configs = stream_configs_from_scenario(d)
        src = (
            dict(traffic)
            if traffic is not None
            else traffic_from_scenario(d, configs, base_dir=bdir)
        )
        eng = cls(model, streams=configs, config=cfg, traffic=src, sinks=sinks, clock=clock)
        eng.scenario = d
        return eng

    scenario: Mapping[str, Any] | None = None

    # ================================================================== helpers
    def frontier(self) -> dict[StreamId, int]:
        """First uncomputed sample of every RX stream (its group's cursor)."""
        return {r: g.cursor for g in self.groups for r in g.rx}

    def _frontier_time_max(self) -> Fraction:
        return max((from_ticks(g.cursor, g.rate) for g in self.groups), default=Fraction(0))

    def _emit_records(self, recs: Sequence[CommandRecord]) -> None:
        if not recs:
            return
        now = self.clock.now_ns()
        for r in recs:
            if r.status in ("applied", "late_applied"):
                r.t_wall_computed_ns = now
                if r.t_wall_visible_ns is None:
                    r.t_wall_visible_ns = now
            cfg = None
            book = self.commands.books.get(r.panel)
            if (
                book is not None
                and r.t_applied is not None
                and r.status
                in (
                    "applied",
                    "late_applied",
                )
            ):
                cfg = book.config_at(r.t_applied)  # the configuration right after the switch
            for s in self.sinks:
                s.on_record(r, cfg)

    def add_sink(self, sink: EngineSink) -> None:
        """Attach a recording/telemetry sink."""
        with self._lock:
            self.sinks.append(sink)
            self._sinks_want_rx = self._sinks_record_rx()

    def _sinks_record_rx(self) -> bool:
        """Whether a sink records RX samples (then every RX page is computed)."""
        return any(type(s).on_rx is not EngineSink.on_rx for s in self.sinks)

    def on_page(self, fn: Callable[[StreamId, int, int], None]) -> None:
        """Call ``fn(rx, k0, k1)`` after each computed RX range (tests, injected load)."""
        self._on_page.append(fn)

    def capture_accumulator(
        self, rx: StreamId, enabled: bool = True
    ) -> list[tuple[int, NDArray[np.complex128]]]:
        """Capture the complex128 RX accumulator (after noise and RX LO, before the
        complex64 front-end cast) of ``rx``; returns the list that collects ``(k0, y)``."""
        with self._lock:
            st = self._rx_state(rx)
            st.capture = [] if enabled else None
            return st.capture if st.capture is not None else []

    def _rx_state(self, rx: StreamId) -> RxStream:
        try:
            return self.rx[rx]
        except KeyError:
            raise UnknownTarget(f"unknown RX stream {rx!r}") from None

    # ================================================================== scheduling core
    def _active(self, rx: StreamId, k0: int, k1: int) -> bool:
        st = self.rx[rx]
        if self.cfg.always_active or st.capture is not None or self.barriers.slots:
            return True
        if self._sinks_want_rx:
            return True
        return st.wants(k0, k1)

    def _post(self, rx: StreamId, k0: int, acc: NDArray[np.complex128]) -> None:
        st = self.rx[rx]
        d = st.process(k0, acc)
        st.deliver(k0, d, drop_when_full=self._paced_active)
        for s in self.sinks:
            s.on_rx(rx, k0, d)
        self._produced[rx] += int(d.shape[0])
        for fn in self._on_page:
            fn(rx, k0, k0 + int(d.shape[0]))

    def _apply_events(self) -> None:
        """Apply the geometry events that this step's pages may need.

        The horizon is the furthest page end of any group; on a snapshot grid it is raised
        to the end of the grid interval holding that page end, so an event inside a grid
        interval splits it before any of its samples is rendered (an on-time event then
        gives the same output for every page size, I2a/I2b)."""
        if not len(self.events):
            return
        t_max = max(
            (from_ticks((g.cursor // g.page + 1) * g.page, g.rate) for g in self.groups),
            default=Fraction(0),
        )
        if self.snaps.mode == "grid" and self.t_u is not None:
            t_max = math.ceil(t_max / self.t_u) * self.t_u
        for ev in self.events.pop_due(t_max):
            self._apply_geometry(ev.t, ev.kind, ev.payload)

    def _apply_geometry(self, t: Fraction, kind: str, payload: Any) -> None:
        node = payload["node"]
        if self.snaps.mode == "grid":
            # the interval this event splits starts at the grid point or at the latest
            # earlier geometry event; it is deterministic only if no sample of it was computed
            iv_lo = math.floor(t / self.t_u) * self.t_u if self.t_u else Fraction(0)
            i = bisect.bisect_right(self.snaps.events, t)
            if i:
                iv_lo = max(iv_lo, self.snaps.events[i - 1])
            for g in self.groups:
                if g.cursor > ceil_ticks(iv_lo, g.rate):
                    self._flag_nondeterministic(
                        f"geometry event at t={t} arrived after its snapshot interval started"
                    )
        if kind == "node.set_position":
            pos = payload["position_m"]
            self.snaps.apply_event(t, lambda: self.model.set_node_position(node, t, pos))
        else:
            traj = payload["trajectory"]
            self.snaps.apply_event(t, lambda: self.model.set_node_trajectory(node, t, traj))
        self._geo_applied = t if self._geo_applied is None else max(self._geo_applied, t)
        prev = self._geo_applied_node.get(node)
        self._geo_applied_node[node] = t if prev is None else max(prev, t)
        self.builder.clear()
        # tau_RIS->r may change from t on: re-map pending RIS commands (I3)
        self.commands.geometry_changed(t)

    def _flag_nondeterministic(self, why: str) -> None:
        if not self.nondeterministic_inputs:
            warnings.warn(f"nondeterministic inputs: {why}", NondeterminismWarning, stacklevel=3)
        self.nondeterministic_inputs = True

    def step(self, *, zero_fill: bool = False) -> bool:
        """One lockstep iteration (section 4.8): every group advances by at most one page
        portion, then due barriers run. Returns True when anything progressed."""
        with self._lock:
            if self._closed:
                return False
            self._apply_events()
            progress = False
            for g in self.groups:
                if self._backpressured(g):
                    continue
                lim = self.barriers.limit(g.rate)
                st: Step = advance_group(
                    g,
                    self.builder,
                    self.tx,
                    limit=lim,
                    active=self._active,
                    post=self._post,
                    zero_fill=zero_fill,
                )
                if st.zero_filled:
                    self.counters.add("stall_zero_samples", sum(st.zero_filled.values()))
                    self._flag_nondeterministic("stall zero_fill")
                progress |= st.progress
            if self.barriers.slots and self.groups:
                s = self.barriers.due([(g.rate, g.cursor, g.end) for g in self.groups])
                if s is not None and s.t_next < self._t_end():
                    self._run_barrier(s)
                    progress = True
            if self.commands.has_pending or self._node_records:
                fr = self.frontier()
                final = self.commands.update_statuses(fr)
                final += self._finalize_node_records(fr)
                self._emit_records(final)
            if progress:
                self._advance_metrics()
                self._cond.notify_all()
            return progress

    def _t_end(self) -> Fraction | float:
        """Latest end time of any group (infinity when a group has no end)."""
        ends = [Fraction(g.end) / g.rate if g.end is not None else None for g in self.groups]
        if not ends or any(e is None for e in ends):
            return float("inf")
        return max(e for e in ends if e is not None)

    def _backpressured(self, g: RateGroup) -> bool:
        """Lockstep with an engine thread: do not compute ahead of a client whose RX queue
        is full (backpressure; paced mode drops pages instead, section 4.10)."""
        if self._thread is None or self.cfg.mode != "lockstep":
            return False
        for r in g.rx:
            st = self.rx[r]
            if st.streaming and st.queue_samples is not None and st.queued >= st.queue_samples:
                return True
        return False

    def _paced_blocked(self, g: RateGroup) -> bool:
        """Ended, or waiting at the next barrier's frontier limit (paced selection)."""
        if g.end is not None and g.cursor >= g.end:
            return True
        lim = self.barriers.limit(g.rate)
        return lim is not None and g.cursor >= lim

    def _all_ended(self) -> bool:
        return all(g.end is not None and g.cursor >= g.end for g in self.groups)

    def _finalize_node_records(self, fr: Mapping[StreamId, int]) -> list[CommandRecord]:
        out = []
        keep = []
        for r in self._node_records:
            if all(int(fr.get(s, 0)) > k for s, k in r.k_applied.items()):
                late = any(v != 0 for v in r.lateness_samples.values()) or id(r) in self._node_late
                self._node_late.discard(id(r))
                r.status = "late_applied" if late else "applied"
                out.append(r)
            else:
                keep.append(r)
        self._node_records = keep
        return out

    def _advance_metrics(self) -> None:
        if self.rtf is None:
            return
        now = self.clock.now_ns()
        if not self.rtf.started:
            self.rtf.start(now)
        produced = self._produced
        self._produced = dict.fromkeys(produced, 0)
        self.rtf.advance(produced, now)
        while len(self.rtf.windows) > self._metrics_window:
            t0, wall, rtf = self.rtf.windows[self._metrics_window]
            delta = self.counters.delta(self._metrics_counters)
            self._metrics_counters = dict(self.counters)
            win = {
                "t0_s": fraction_to_str(t0),
                "t1_s": fraction_to_str(t0 + self.rtf.window_s),
                "wall_s": wall,
                "rtf": rtf,
                "counters": delta,
            }
            for s in self.sinks:
                s.on_metrics(win)
            self._metrics_window += 1

    # ================================================================== barriers
    def _observation(self, s: ControllerSlot, t_b: Fraction) -> Observation:
        fields_ = set(s.spec.fields)
        k_b: dict[StreamId, int] = {}
        power: dict[StreamId, float] = {}
        iq: dict[StreamId, NDArray[np.complex64]] = {}
        for r, st in self.rx.items():
            f = st.cfg.rate
            kb = to_ticks(t_b, f)
            kw = max(0, to_ticks(t_b - s.spec.window_s, f))
            y = st.history.read(kw, kb - kw)
            k_b[r] = kb
            p = st.power_w(y)
            power[r] = 10.0 * math.log10(p / 1e-3) if p > 0 else (-math.inf if p == 0 else math.nan)
            if "iq_window" in fields_:
                iq[r] = y
        positions = None
        if "positions" in fields_:
            node_pos = getattr(self.model, "node_position", None)
            if node_pos is None:
                raise ConfigurationError("this model cannot report node positions")
            nodes = sorted({c.node for c in self.configs.values()})
            t_pos = t_b - s.spec.position_delay_s
            noise = self.barriers.position_noise(s, nodes)
            positions = {
                n: np.asarray(node_pos(n, max(t_pos, Fraction(0))), np.float64) + noise[n]
                for n in nodes
            }
        csi = None
        if "csi_oracle" in fields_:
            self.oracle_csi_used = True
            snap = self.model.snapshot(t_b)
            csi = {
                lk: csi_view(self.model, snap, lk, self.model.link_fc_hz(lk))
                for lk in self.model.links()
            }
        return Observation(
            t_b=t_b,
            k_b=k_b,
            rx_power_dbm=power,
            positions=positions,
            csi_oracle=csi,
            iq_window=iq if "iq_window" in fields_ else None,
            ris_state=self.commands.snapshot_states(t_b),
            pending=self.commands.pending_receipts(t_b),
        )

    def _run_barrier(self, s: ControllerSlot) -> None:
        t_b = s.t_next
        s.t_b_last = t_b
        obs = self._observation(s, t_b)
        if s.remote:
            proxy = s.ctrl
            assert isinstance(proxy, RemoteControllerProxy)
            proxy.send_barrier(obs)
            # release the (re-entrant) lock completely so the remote side can schedule
            # its commands from another thread, as Condition.wait does
            saved = self._lock._release_save()  # type: ignore[attr-defined]
            try:
                ok = proxy.wait_ack(t_b, self.cfg.stall_timeout_s)
            finally:
                self._lock._acquire_restore(saved)  # type: ignore[attr-defined]
            if not ok:
                raise SremuTimeout(f"controller {s.name!r} did not ack barrier t_b = {t_b}")
        else:
            ctrl = s.ctrl
            assert isinstance(ctrl, Controller)
            actions = list(ctrl.decide(obs))
            for i, a in enumerate(actions):
                at = self.barriers.action_time(s, t_b, i, a)
                req = action_request(s.name, s.m, i, a, at, self.cfg.late_policy)
                receipt = self._schedule_locked(req, internal=True)
                if not receipt.ok:
                    if receipt.error == ErrorCode.UNKNOWN_TARGET:
                        raise UnknownTarget(receipt.msg)
                    raise ConfigurationError(
                        f"{s.name}: action rejected: {receipt.error.name}: {receipt.msg}"
                    )
        s.m += 1

    def register_controller(self, ctrl: Controller | RemoteControllerProxy) -> None:
        """Register a barrier controller (in-process ``Controller`` or remote proxy)."""
        with self._lock:
            slot = self.barriers.register(ctrl, self._frontier_time_max(), self.commands.caps())
            # the observation window [to_ticks(t_b - window), to_ticks(t_b)) is read when the
            # frontier is at most to_ticks(t_b + L_min): keep that much RX history
            for st in self.rx.values():
                f = st.cfg.rate
                need = math.ceil((slot.l_min + slot.spec.window_s) * f) + 2
                st.history.grow(need + page_size(f, self.cfg.page_target_s))

    def unregister_controller(self, name: str) -> None:
        """Remove a registered controller."""
        with self._lock:
            self.barriers.unregister(name)
            self._kick()

    def _kick(self) -> None:
        """(lock held) An input changed (TX, stream command, command, RX consumed): wake
        the waiters and count it, so a loop that is between a fruitless ``step`` and its
        wait steps again at once instead of sleeping through the input (lost wake-up).
        Wall-clock only: what a step computes does not depend on it."""
        self._kicks += 1
        self._cond.notify_all()

    # ================================================================== commands
    def schedule(self, req: CommandRequest) -> CommandReceipt:
        """Schedule a timed command (sections 4.6, 8.5); never raises on bad input."""
        with self._lock:
            receipt = self._schedule_locked(req)
            self._kick()
            return receipt

    def _schedule_locked(
        self,
        req: CommandRequest,
        *,
        physical: PanelConfig | None = None,
        internal: bool = False,
    ) -> CommandReceipt:
        n0 = len(self.commands.records)
        receipt = self._schedule_inner(req, physical=physical)
        for rec in self.commands.records[n0:]:  # pair new records with their request
            self._req_of[id(rec)] = None if (physical is not None or internal) else req
        return receipt

    def _schedule_inner(
        self, req: CommandRequest, *, physical: PanelConfig | None = None
    ) -> CommandReceipt:
        fr = self.frontier()
        if req.op in ("ris.set_config", "ris.set_enabled"):
            bound = self.barriers.lookahead_bound(req.src)
            out = self.commands.schedule(
                req, fr, late_policy=req.late_policy, lookahead_bound=bound, physical=physical
            )
            if out.late:
                self.counters.add("late_commands")
                if self.cfg.mode == "lockstep" and self.barriers.slot(req.src) is None:
                    self._flag_nondeterministic(f"late command {req.cmd_id!r} from {req.src!r}")
            if not out.receipt.ok:
                self.counters.add("rejected_commands")
            self._emit_records(out.final)
            return out.receipt
        if req.op in ("node.set_position", "node.set_trajectory"):
            return self._schedule_node(req, fr)
        if req.op == "rec.marker":
            return self._schedule_marker(req)
        out = self.commands.reject(req, ErrorCode.NOT_PERMITTED, f"unsupported op {req.op!r}", None)
        self.counters.add("rejected_commands")
        self._emit_records(out.final)
        return out.receipt

    def _rx_ticks(self, t: Fraction) -> dict[StreamId, int]:
        return {r: to_ticks(t, st.cfg.rate) for r, st in self.rx.items()}

    def _rx_first_samples(self, t: Fraction) -> dict[StreamId, int]:
        """First RX sample at or after ``t`` (``ceil(t*f)``) per RX: where a geometry event
        at ``t`` first shows. Geometry is a function of continuous time evaluated at the
        sample instants ``n/f`` and snapshot intervals hold the samples
        ``[ceil(t_a f), ceil(t_b f))`` (WP5 render contract), so this, not ``to_ticks``,
        is the sample the event acts on."""
        return {r: ceil_ticks(t, st.cfg.rate) for r, st in self.rx.items()}

    def _schedule_node(self, req: CommandRequest, fr: Mapping[StreamId, int]) -> CommandReceipt:
        nodes = {c.node for c in self.configs.values()}
        if req.target not in nodes:
            out = self.commands.reject(
                req, ErrorCode.UNKNOWN_TARGET, f"unknown node {req.target!r}", None
            )
            self.counters.add("rejected_commands")
            self._emit_records(out.final)
            return out.receipt
        if req.at.kind == "t":
            assert req.at.t is not None
            t_req = req.at.t
        elif req.at.kind == "sample":
            assert req.at.stream is not None
            assert req.at.index is not None
            if req.at.stream not in self.configs:
                out = self.commands.reject(req, ErrorCode.UNKNOWN_TARGET, "unknown stream", None)
                self.counters.add("rejected_commands")
                self._emit_records(out.final)
                return out.receipt
            t_req = from_ticks(req.at.index, self.configs[req.at.stream].rate)
        else:
            t_req = self._frontier_time_max()
        k_req = self._rx_first_samples(t_req)
        if req.op == "node.set_trajectory" and self.snaps.mode == "static":
            out = self.commands.reject(
                req,
                ErrorCode.NOT_PERMITTED,
                "node.set_trajectory needs a snapshot grid: set numerics.t_u_s for scenes "
                "whose external nodes move",
                t_req,
                k_req,
            )
            self.counters.add("rejected_commands")
            self._emit_records(out.final)
            return out.receipt
        behind_frontier = any(k_req[r] < fr.get(r, 0) for r in k_req)
        # the model takes a node's events in time order, and on a snapshot grid the
        # snapshots after an applied event (its left limit included) already hold the
        # geometry before it: an event earlier than one already applied (ahead, with its
        # snapshot interval) cannot act before it (static scenes: same node only)
        floor_t = (
            self._geo_applied
            if self.snaps.mode == "grid"
            else self._geo_applied_node.get(req.target)
        )
        behind_applied = floor_t is not None and t_req < floor_t
        late = behind_frontier or behind_applied
        t_app = t_req
        if late:
            self.counters.add("late_commands")
            if req.late_policy == "reject":
                out = self.commands.reject(
                    req, ErrorCode.LATE_COMMAND, "command is late", t_req, k_req
                )
                self.counters.add("rejected_commands")
                self._emit_records(out.final)
                return out.receipt
            if behind_frontier:
                t_app = max(from_ticks(fr.get(r, 0), st.cfg.rate) for r, st in self.rx.items())
            if floor_t is not None:
                t_app = max(t_app, floor_t)
            if self.cfg.mode == "lockstep" and self.barriers.slot(req.src) is None:
                self._flag_nondeterministic(f"late node command {req.cmd_id!r}")
        payload: dict[str, Any] = {"node": req.target}
        if req.op == "node.set_position":
            assert req.position_m is not None
            payload["position_m"] = np.asarray(req.position_m, np.float64).copy()
        else:
            payload["trajectory"] = dict(req.payload.get("trajectory", req.payload))
        seq = self.commands.next_seq()
        k_app = self._rx_first_samples(t_app)
        self.events.push(t_app, EventPriority.GEOMETRY, seq, req.op, payload)
        rec = self.commands._record(
            req.cmd_id,
            seq,
            req.src,
            req.op,
            req.target,
            t_req,
            t_app,
            k_req,
            k_app,
            "scheduled",
            "",
            None,
            False,
            req.t_wall_recv_ns,
        )
        self.commands.records.append(rec)
        self._node_records.append(rec)
        if late:
            self._node_late.add(id(rec))
        return CommandReceipt(
            req.cmd_id,
            True,
            "late_applied" if late else "scheduled",
            seq=seq,
            t_req=t_req,
            k_req=k_req,
            t_wall_recv_ns=req.t_wall_recv_ns,
        )

    def _schedule_marker(self, req: CommandRequest) -> CommandReceipt:
        t = req.at.t if req.at.kind == "t" and req.at.t is not None else self._frontier_time_max()
        k = self._rx_ticks(t)
        seq = self.commands.next_seq()
        rec = self.commands._record(
            req.cmd_id,
            seq,
            req.src,
            req.op,
            req.target,
            t,
            t,
            k,
            dict(k),
            "applied",
            "",
            None,
            False,
            req.t_wall_recv_ns,
        )
        self.commands.records.append(rec)
        self._emit_records([rec])
        return CommandReceipt(req.cmd_id, True, "scheduled", seq=seq, t_req=t, k_req=k)

    def schedule_physical(
        self,
        panel: PanelId,
        t: Fraction,
        cfg: PanelConfig,
        *,
        transition: TransitionSpec | Mapping[str, Any] | None = None,
        cmd_id: str = "replay",
        src: str = "replay",
    ) -> CommandReceipt:
        """Replay form (section 4.12): apply the recorded physical configuration of a
        command at its recorded ``t_applied`` as an on-time command."""
        from sremu.contracts import CommandTime

        if isinstance(transition, Mapping):
            transition = TransitionSpec(
                transition.get("model", "step"),
                None
                if transition.get("duration_s") is None
                else parse_exact(transition["duration_s"]),
                transition.get("classes", "single"),
            )
        n = self.commands.specs[panel].n_controls
        with self._lock:
            book = self.commands.books[panel]
            receipts = []
            cur = book.config_at(t)
            if bool(cur.enabled) != bool(cfg.enabled):
                receipts.append(
                    self._schedule_locked(
                        CommandRequest(
                            cmd_id + ":en",
                            "ris.set_enabled",
                            src,
                            panel,
                            CommandTime.at(t),
                            enabled=bool(cfg.enabled),
                        )
                    )
                )
            req = CommandRequest(
                cmd_id,
                "ris.set_config",
                src,
                panel,
                CommandTime.at(t),
                states=np.zeros(n, np.int16),
                transition=transition,
            )
            receipts.append(self._schedule_locked(req, physical=cfg))
            return receipts[-1]

    # ================================================================== streams
    def open_stream(self, spec: StreamSpec) -> StreamHandle:
        """Attach a client to a scenario stream (rate and carrier must match)."""
        with self._lock:
            c = self.configs.get(spec.stream)
            if c is None:
                raise UnknownTarget(f"unknown stream {spec.stream!r}")
            if spec.direction != c.direction:
                raise ConfigurationError(f"{spec.stream} is a {c.direction} stream")
            if Fraction(spec.rate_sps) != c.rate:
                raise ConfigurationError(
                    f"{spec.stream}: rate {spec.rate_sps} != scenario rate {c.rate} (no resampler)"
                )
            if float(spec.fc_hz) != float(c.fc_hz):
                raise ConfigurationError(
                    f"{spec.stream}: carrier {spec.fc_hz} != scenario carrier {c.fc_hz}"
                )
            if spec.stream in self._by_name:
                raise ConfigurationError(f"stream {spec.stream!r} is already open")
            h = StreamHandle(self._next_id, spec, page_size(c.rate, self.cfg.page_target_s))
            self._next_id += 1
            self._handles[h.id] = h
            self._by_name[spec.stream] = h.id
            if c.direction == "tx":
                self.tx[c.stream].open()
            else:
                self.rx[c.stream].opened = True
            return h

    def _handle(self, h: StreamHandle, direction: str) -> StreamId:
        if self._handles.get(h.id) is not h and self._handles.get(h.id) != h:
            raise StreamError(f"{h.spec.stream!r} is not open")
        if h.spec.direction != direction:
            raise StreamError(f"{h.spec.stream!r} is not a {direction.upper()} stream")
        return h.spec.stream

    def _tx_frontier(self, tx: StreamId) -> int:
        rate = self.configs[tx].rate
        cur = [g.cursor for g in self.groups if g.rate == rate]
        return max(cur, default=0)

    def submit_tx(
        self, h: StreamHandle, samples: NDArray[np.complexfloating], md: TxMetadata
    ) -> int:
        """Queue TX samples (complex64 digital, full scale 1); late timed bursts are dropped
        with async ``TIME_ERROR`` and counted in ``late_samples`` (I6)."""
        with self._lock:
            sid = self._handle(h, "tx")
            st = self.tx[sid]
            x = np.ascontiguousarray(samples, dtype=np.complex64).reshape(-1)
            if st.source is not None and x.size:
                # the ring follows the in-process generator; client samples would be
                # dropped silently (I6)
                raise StreamError(
                    f"{sid} is driven by an in-process traffic generator; clear the "
                    "scenario's traffic for this stream to send from a client"
                )
            acc, late = st.submit(x, md, self._tx_frontier(sid))
            if late:
                self.counters.add("late_samples", late)
            elif acc:
                k0 = st.data_end - acc
                for s in self.sinks:
                    s.on_tx(sid, k0, x)
            self._kick()
            return acc

    def stream_cmd(self, h: StreamHandle, cmd: StreamCmd) -> None:
        """Issue an RX stream command (UHD modes)."""
        with self._lock:
            sid = self._handle(h, "rx")
            self.rx[sid].stream_cmd(cmd, self._group_of_rx[sid].cursor)
            self._kick()

    def read_rx(
        self, h: StreamHandle, n: int, timeout_s: float
    ) -> tuple[NDArray[np.complex64], RxMetadata]:
        """Read up to ``n`` RX samples. In-process (no engine thread) the scheduler is
        pulled until enough samples exist or no progress is possible (TIMEOUT when none);
        with a running engine thread it waits up to ``timeout_s`` wall seconds."""
        with self._lock:
            sid = self._handle(h, "rx")
            st = self.rx[sid]
            if self._thread is None:
                while st.queued < n and st.streaming and st.pending_error == ErrorCode.NONE:
                    if not self.step():
                        break
            elif st.queued == 0 and st.pending_error == ErrorCode.NONE:
                self._cond.wait_for(
                    lambda: st.queued > 0 or not st.streaming or self._thread is None,
                    max(0.0, float(timeout_s)),
                )
            out = st.read(n)
            if self._thread is not None:
                self._kick()  # queue space freed: a backpressured group may advance
            return out

    def recv_async(self, h: StreamHandle, timeout_s: float) -> AsyncMetadata | None:
        """Next asynchronous TX event of ``h`` (BURST_ACK, TIME_ERROR, UNDERFLOW), or None."""
        with self._lock:
            sid = self._handle(h, "tx")
            q = self.tx[sid].asyncq
            return q.popleft() if q else None

    def time_now(self) -> Fraction:
        """Min over RX streams of ``frontier_r/f_r`` (s); 0 without RX streams."""
        with self._lock:
            return min((from_ticks(g.cursor, g.rate) for g in self.groups), default=Fraction(0))

    def caps(self) -> Mapping[PanelId, RisCapabilities]:
        """Capabilities of every panel."""
        return self.commands.caps()

    @property
    def records(self) -> tuple[CommandRecord, ...]:
        """Every command record so far, in arrival order."""
        return tuple(self.commands.records)

    def command_log(self) -> list[tuple[CommandRecord, CommandRequest | None]]:
        """Every command record with the client request that produced it (arrival order).
        The request is None for commands issued inside the engine (in-process controller
        actions) and for physical replays (:meth:`schedule_physical`); their applied
        configuration comes with the final record (``EngineSink.on_record``) or from
        :meth:`panel_state`."""
        with self._lock:
            return [(r, self._req_of.get(id(r))) for r in self.commands.records]

    def panel_state(self, panel: PanelId, t: Fraction) -> PanelConfig:
        """Configuration of ``panel`` in effect at RIS time ``t`` (latest command target and
        enable flag); ``UnknownTarget`` for unknown panels."""
        with self._lock:
            book = self.commands.books.get(panel)
            if book is None:
                raise UnknownTarget(f"unknown panel {panel!r}")
            return book.config_at(parse_exact(t, name="t"))

    def tx_head(self, h: StreamHandle) -> int:
        """First absolute TX index after the samples accepted so far (an untimed burst of
        ``n`` accepted samples started at ``tx_head(h) - n``)."""
        with self._lock:
            sid = self._handle(h, "tx")
            return int(self.tx[sid].data_end)

    # ================================================================== runs
    _thread: threading.Thread | None = None
    _stop_evt: threading.Event | None = None
    _kicks: int = 0

    def start(self) -> None:
        """Run the lockstep loop in a background thread (served mode: clients in other
        threads or processes feed TX and read RX; ``read_rx`` then blocks up to its
        timeout). A stall (no progress for ``stall_timeout_s``) writes one deadlock report
        per episode and, with ``on_stall="zero_fill"``, commits the missing TX samples as
        zeros. Stop with :meth:`stop`."""
        with self._lock:
            if self._thread is not None:
                raise ConfigurationError("the engine thread is already running")
            if self.cfg.mode != "lockstep":
                raise ConfigurationError("start() serves lockstep; use run_paced() for paced")
            self._stop_evt = threading.Event()
            self._thread = threading.Thread(
                target=self._serve_loop, name="sremu-engine", daemon=True
            )
            self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop the engine thread (idempotent); waiting readers wake up."""
        th = self._thread
        if th is None:
            return
        assert self._stop_evt is not None
        self._stop_evt.set()
        with self._lock:
            self._cond.notify_all()
        th.join(timeout_s)
        with self._lock:
            self._thread = None
            self._cond.notify_all()

    def _serve_loop(self) -> None:
        assert self._stop_evt is not None
        stop = self._stop_evt
        last = self.clock.now_ns()
        stalled = False
        while not stop.is_set():
            if self.groups and self._all_ended():
                break
            kicks = self._kicks
            if self.step():
                last = self.clock.now_ns()
                stalled = False
                continue
            with self._lock:
                if stop.is_set():
                    break
                if self._kicks == kicks:  # else an input arrived after the step began
                    self._cond.wait(0.005)
            waiting = any(g.waiting_on for g in self.groups)
            overdue = (self.clock.now_ns() - last) * 1e-9 >= self.cfg.stall_timeout_s
            if waiting and not stalled and overdue:
                stalled = True
                with self._lock:
                    self._stall()
                if self.cfg.on_stall == "zero_fill":
                    self.step(zero_fill=True)
                    last = self.clock.now_ns()
                    stalled = False

    def stall_report(self) -> StallReport:
        """Which RX waits on which TX and which barriers are pending (section 4.8)."""
        groups = []
        for g in self.groups:
            waiting = {}
            for t in g.waiting_on:
                waiting[t] = {
                    "committed_through": self.tx[t].committed(),
                    "ring_head": self.tx[t].ring.committed_through(),
                }
            groups.append(
                {
                    "rate_sps": f"{g.rate.numerator}/{g.rate.denominator}",
                    "rx": list(g.rx),
                    "cursor": g.cursor,
                    "end": g.end,
                    "waiting_on_tx": waiting,
                }
            )
        bars = [
            {
                "controller": s.name,
                "t_b": f"{s.t_next.numerator}/{s.t_next.denominator}",
                "remote": s.remote,
                "limits": {
                    f"{g.rate.numerator}/{g.rate.denominator}": s.limit(g.rate) for g in self.groups
                },
            }
            for s in self.barriers.slots
        ]
        return StallReport(self.cfg.stall_timeout_s, tuple(groups), tuple(bars), self.cfg.on_stall)

    def _stall(self) -> StallReport:
        rep = self.stall_report()
        self.stall_reports.append(rep)
        for s in self.sinks:
            s.on_stall(rep)
        return rep

    def run(
        self,
        until_s: Fraction | int | str | None = None,
        *,
        stall_timeout_s: float | None = None,
    ) -> RunSummary:
        """Headless lockstep run until every RX reached ``until_s`` (default: the configured
        duration). On a stall (no progress for ``stall_timeout_s`` wall seconds) a deadlock
        report is written; ``on_stall="zero_fill"`` then commits the missing TX samples as
        zeros (``stall_zero_samples``, ``nondeterministic_inputs``); ``"wait"`` keeps
        waiting when ``external_inputs`` is set and raises ``SremuTimeout`` otherwise."""
        if until_s is not None:
            u = parse_exact(until_s, name="until_s")
            with self._lock:
                for g in self.groups:
                    g.end = ceil_ticks(u, g.rate)
        if any(g.end is None for g in self.groups):
            raise ConfigurationError("run() needs until_s or a configured duration_s")
        timeout = self.cfg.stall_timeout_s if stall_timeout_s is None else float(stall_timeout_s)
        if self.rtf is not None and not self.rtf.started:
            self.rtf.start(self.clock.now_ns())
        last = self.clock.now_ns()
        while not self._all_ended():
            kicks = self._kicks
            if self.step():
                last = self.clock.now_ns()
                continue
            with self._lock:
                if self._kicks == kicks:  # else an input arrived after the step began
                    self._cond.wait(min(0.01, max(timeout, 0.0)) if self.cfg.external_inputs else 0)
            now = self.clock.now_ns()
            if (now - last) * 1e-9 < timeout and self.cfg.external_inputs:
                continue
            if (now - last) * 1e-9 < timeout:
                self.clock.sleep(min(0.001, timeout))
                if (self.clock.now_ns() - last) * 1e-9 < timeout:
                    continue
            rep = self._stall()
            if self.cfg.on_stall == "zero_fill":
                self.step(zero_fill=True)
                last = self.clock.now_ns()
                continue
            if not self.cfg.external_inputs:
                raise SremuTimeout(
                    f"lockstep stall (no external inputs can arrive): {rep.to_dict()}"
                )
            last = self.clock.now_ns()
        return self.summary()

    def run_paced(self, until_s: Fraction | int | str | None = None) -> RunSummary:
        """Paced run (section 4.10): pages released at ``t0 + k1/f + D_lag`` (wall);
        missing TX samples at release are zeros (``UNDERFLOW``), a full RX queue drops the
        page (``overflow_pages``); deadline misses are counted per slack."""
        if until_s is not None:
            u = parse_exact(until_s, name="until_s")
            for g in self.groups:
                g.end = ceil_ticks(u, g.rate)
        if any(g.end is None for g in self.groups):
            raise ConfigurationError("run_paced() needs until_s or a configured duration_s")
        pacer = Pacer(self.clock, d_lag_s=self.cfg.d_lag_s, slacks_s=self.cfg.slacks_s)
        self._pacer = pacer
        pacer.start()
        if self.rtf is not None and not self.rtf.started:
            self.rtf.start(self.clock.now_ns())
        self._paced_active = True
        try:
            while not self._all_ended():
                ready = [g for g in self.groups if not self._paced_blocked(g)]
                if not ready:  # every open group waits at a barrier limit: run it now
                    with self._lock:
                        s = self.barriers.due([(x.rate, x.cursor, x.end) for x in self.groups])
                        if s is None or s.t_next >= self._t_end():  # pragma: no cover
                            raise ConfigurationError("paced run: no group can advance")
                        self._run_barrier(s)
                    continue
                g = min(
                    ready,
                    key=lambda g: pacer.release_ns(
                        min((g.cursor // g.page + 1) * g.page, g.end or 0), g.rate
                    ),
                )
                k1 = (g.cursor // g.page + 1) * g.page
                if g.end is not None:
                    k1 = min(k1, g.end)
                r_ns = pacer.release_ns(k1, g.rate)
                pacer.wait_until(r_ns)
                t0 = self.clock.now_ns()
                with self._lock:
                    lim = self.barriers.limit(g.rate)
                    self._apply_events()
                    st = advance_group(
                        g,
                        self.builder,
                        self.tx,
                        limit=lim,
                        active=self._active,
                        post=self._post,
                        zero_fill=True,
                    )
                    if st.zero_filled:
                        for t, z in st.zero_filled.items():
                            self.counters.add("underflow_slip_samples", z)
                            tx = self.tx[t]
                            tx.slip = True
                            tx._async(AsyncCode.UNDERFLOW, tx.ring.committed_through() - z, z)
                    if self.barriers.slots:
                        s = self.barriers.due([(x.rate, x.cursor, x.end) for x in self.groups])
                        if s is not None and s.t_next < self._t_end():
                            self._run_barrier(s)
                    fr = self.frontier()
                    self._emit_records(
                        self.commands.update_statuses(fr) + self._finalize_node_records(fr)
                    )
                    self._advance_metrics()
                    self._cond.notify_all()
                t1 = self.clock.now_ns()
                if st.progress and not st.skipped:
                    misses = pacer.page_done(
                        r_ns,
                        t1,
                        compute_ns=t1 - t0,
                        page_ns=round(Fraction(st.k1 - st.k0) / g.rate * 10**9),
                    )
                    if pacer.slacks_ns and min(pacer.slacks_ns) in misses:
                        self.counters.add("deadline_misses")  # at the smallest slack
            self.counters["overflow_pages"] = sum(r.overflow_pages for r in self.rx.values())
        finally:
            self._paced_active = False
        return self.summary()

    def reset_metrics(self) -> None:
        """Restart the RTF measurement now (e.g. after a warm-up, section 12.3)."""
        with self._lock:
            if self.rtf is None:
                return
            self._advance_metrics()
            self.rtf.restart(self.clock.now_ns(), {r: g.cursor for g in self.groups for r in g.rx})
            self._metrics_window = 0

    def summary(self) -> RunSummary:
        """Counters, samples per stream, RTF and flags of the run so far."""
        with self._lock:
            self._advance_metrics()
            samples = {r: g.cursor for g in self.groups for r in g.rx}
            vmax = max(self.builder.verify_max.values(), default=None)
            rep = self.rtf.report().to_dict() if self.rtf is not None else {}
            self.counters["overflow_pages"] = sum(r.overflow_pages for r in self.rx.values())
            return RunSummary(
                samples=samples,
                counters=dict(self.counters),
                rtf=rep,
                nondeterministic_inputs=self.nondeterministic_inputs,
                oracle_csi_used=self.oracle_csi_used,
                stalls=len(self.stall_reports),
                verify_interp_max_rel=vmax,
                paced=None if self._pacer is None else self._pacer.stats.to_dict(),
            )

    def manifest_info(self) -> dict[str, Any]:
        """Run facts for the manifest (section 3.6): mode, backend, page sizes, per-link
        decisions, snapshot policy, ``verify_interp_max_rel``, counters and flags."""
        pol = self.policy.to_manifest()
        pol["t_u_s_used"] = (
            None if self.t_u is None else f"{self.t_u.numerator}/{self.t_u.denominator}"
        )
        vmax = max(self.builder.verify_max.values(), default=None)
        return {
            "mode": self.cfg.mode,
            "coupling": self.cfg.coupling,
            "backend": self.cfg.backend,
            "deterministic": self.cfg.deterministic,
            "page_size": {r: g.page for g in self.groups for r in g.rx},
            "link_decisions": dict(self.builder.decisions),
            "snapshot_policy": pol,
            "verify_interp_max_rel": vmax,
            "verify_interp_flagged": None
            if vmax is None
            else bool(vmax > 2 * self.cfg.interp_rel_error),
            "counters": dict(self.counters),
            "nondeterministic_inputs": self.nondeterministic_inputs,
            "oracle_csi_used": self.oracle_csi_used,
            "ris_event_reference": self.cfg.ris_event_reference,
        }

    def close(self) -> None:
        """Stop serving; further steps do nothing."""
        self.stop()
        with self._lock:
            self._closed = True
            self._cond.notify_all()
