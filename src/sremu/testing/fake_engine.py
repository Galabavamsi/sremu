"""Identity-channel ``EngineProtocol`` with the exact time base (DESIGN.md section 14.3).

``FakeEngine`` lets the API, transport and controller work packages test against the
engine contract before the real engine (WP6) exists. It implements every
:class:`sremu.contracts.EngineProtocol` method with the normative timing rules:

* **Channel.** RX stream r receives ``y_r[k] = g_r(k) * sum_tx x_tx[k]`` over the TX streams
  linked to it (equal rate, ``|fc_tx - fc_rx| < f_s/2``, different nodes). There is no
  delay, no noise and no LO rotation. ``g_r(k)`` is 1 unless a ``gain_fn`` maps the panel
  configurations in effect at sample k to a complex gain, which makes RIS switches visible.
* **Time base.** Timed bursts start at ``to_ticks(time_spec, f)`` (R3); RX metadata carry
  the exact ``TimeSpec`` of the first sample; ``time_now()`` is the minimum RX frontier time.
* **Lockstep commitment** (section 4.8). A TX stream is committed through K when its
  samples cover K, after EOB (silence until the next timed burst), or after ``idle_until``.
  An RX read returns only samples whose linked TX streams are committed; with nothing
  available it returns no samples and ``ErrorCode.TIMEOUT`` (it never blocks).
* **Commands** (section 4.6) with tau_RIS->r = 0: ``k_r = to_ticks(t_sw, f_r)`` on every RX;
  late commands follow ``apply_on_arrival`` (one physical time
  ``t_sw' = max_r frontier_r/f_r``) or ``reject`` (``LATE_COMMAND``); same-time commands on
  one panel: the later ``seq`` wins and the earlier one becomes ``superseded`` (it never
  takes effect). Element-subset commands compose in (t, seq) order, independent of their
  arrival order. Every command, accepted or rejected, produces a :class:`CommandRecord`
  (rejections get a seq too). Transitions are logged but applied as steps.
* **Barriers** (section 4.9). A registered controller with period T_c gets barriers at
  ``t_b = m*T_c``; no RX is delivered past ``to_ticks(t_b + L_min, f_r)`` before the barrier
  ran; the observation uses samples ``< to_ticks(t_b, f_r)``; actions default to
  ``t_b + max(latency.fixed_s, L_min)`` (jitter is not drawn) and actions earlier than
  ``t_b + L_min`` raise ``LookaheadViolation``. A timed command from a barrier-registered
  (e.g. remote) source earlier than ``t_b + L_min`` of its latest barrier is rejected with
  ``LOOKAHEAD_VIOLATION`` (RT-17 semantics) instead of being applied late.

Timeouts are ignored (the fake is synchronous) and no wall clock is read (I7).
"""

from __future__ import annotations

import hashlib
import itertools
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from sremu.contracts import (
    COUNTER_NAMES,
    Action,
    AsyncMetadata,
    CommandReceipt,
    CommandRecord,
    CommandRequest,
    CommandStatus,
    CommandTime,
    Controller,
    LinkId,
    Observation,
    PanelConfig,
    PanelId,
    RemoteControllerProxy,
    RisCapabilities,
    RxMetadata,
    StreamCmd,
    StreamHandle,
    StreamId,
    StreamSpec,
    TxMetadata,
    check_link,
    link_id,
)
from sremu.errors import (
    AsyncCode,
    ConfigurationError,
    ErrorCode,
    LookaheadViolation,
    SremuTimeout,
    StreamError,
    UnknownTarget,
)
from sremu.timebase import (
    DEFAULT_PAGE_TARGET_S,
    TimeSpec,
    ceil_ticks,
    from_ticks,
    page_size,
    parse_exact,
    to_ticks,
)

__all__ = ["FakeEngine", "GainFn"]

GainFn = Callable[[Mapping[PanelId, PanelConfig]], complex]
"""Maps the panel configurations in effect at a sample to a complex channel gain."""


@dataclass
class _Tx:
    handle: StreamHandle
    chunks: list[tuple[int, np.ndarray]] = field(default_factory=list)
    head: int = 0  # first index after the last written sample
    in_burst: bool = False
    silent: bool = False  # after EOB: silence until the next timed burst
    idle_until: int = 0
    asyncq: deque[AsyncMetadata] = field(default_factory=deque)

    def committed(self) -> int | None:
        """First uncommitted index, or None when committed indefinitely (silence)."""
        return None if self.silent else max(self.head, self.idle_until)


@dataclass
class _Rx:
    handle: StreamHandle
    frontier: int = 0  # first sample not yet delivered (computed)
    streaming: bool = False
    start: int | None = None  # absolute index where the current burst starts
    remaining: int | None = None  # samples left for num_done / num_more
    end_of_burst_at_end: bool = True
    burst_started: bool = False
    pending_error: ErrorCode = ErrorCode.NONE


@dataclass(frozen=True)
class _Update:
    """What one command changes: the enabled flag, or the states of some (or all) cells."""

    panel: PanelId
    enabled: bool | None = None
    idx: np.ndarray | None = None  # element indices; None = all elements
    states: np.ndarray | None = None

    def apply(self, cfg: PanelConfig) -> PanelConfig:
        if self.enabled is not None:
            return PanelConfig(self.panel, cfg.state.copy(), self.enabled)
        assert self.states is not None
        state = cfg.state.copy()
        if self.idx is None:
            state[:] = self.states
        else:
            state[self.idx] = self.states
        return PanelConfig(self.panel, state, cfg.enabled)


@dataclass
class _Event:
    t: Fraction  # switch time at the RIS (= at every RX, tau = 0)
    seq: int
    update: _Update
    record: CommandRecord


@dataclass
class _Ctrl:
    ctrl: Controller | RemoteControllerProxy
    m: int  # index of the next barrier
    t_b_last: Fraction | None = None  # most recent barrier handed to the controller


class FakeEngine:
    """Identity-channel engine with exact timing; see the module docstring.

    ``panels`` gives the capabilities of each panel; its initial state is all zeros (or
    ``initial_states[panel]``). ``gain_fn`` makes configurations visible in RX output.
    ``barrier_timeout_s`` is passed to ``RemoteControllerProxy.wait_ack``.
    """

    def __init__(
        self,
        panels: Mapping[PanelId, RisCapabilities] | None = None,
        *,
        gain_fn: GainFn | None = None,
        initial_states: Mapping[PanelId, np.ndarray] | None = None,
        page_target_s: str | int | Fraction = DEFAULT_PAGE_TARGET_S,
        barrier_timeout_s: float = 2.0,
        retain_s: str | int | Fraction = 1,
    ) -> None:
        self._caps: dict[PanelId, RisCapabilities] = dict(panels or {})
        self._gain_fn = gain_fn
        self._page_target_s = page_target_s
        self._barrier_timeout_s = barrier_timeout_s
        self._retain_s = parse_exact(retain_s, name="retain_s")
        self._initial: dict[PanelId, PanelConfig] = {}
        for pid, cap in self._caps.items():
            st = (initial_states or {}).get(pid)
            st = (
                np.zeros(cap.n_elements, dtype=np.int16) if st is None else np.asarray(st, np.int16)
            )
            self._initial[pid] = PanelConfig(pid, st.copy())
        self._tx: dict[int, _Tx] = {}
        self._rx: dict[int, _Rx] = {}
        self._by_name: dict[StreamId, int] = {}
        self._next_id = 1
        self._seq = 0
        self._events: dict[PanelId, list[_Event]] = {pid: [] for pid in self._caps}
        self._records: list[CommandRecord] = []
        self._ctrls: list[_Ctrl] = []
        self._ctrl_names: set[str] = set()
        self.counters: dict[str, int] = dict.fromkeys(COUNTER_NAMES, 0)
        self.nondeterministic_inputs = False

    # ================================================================== streams
    def open_stream(self, spec: StreamSpec) -> StreamHandle:
        """Open a TX or RX stream; raises ``RateMismatchError`` for overlapping bands."""
        if spec.stream in self._by_name:
            raise ConfigurationError(f"stream {spec.stream!r} is already open")
        others = [t.handle.spec for t in self._tx.values()]
        others += [r.handle.spec for r in self._rx.values()]
        for o in others:
            if o.direction != spec.direction and o.node != spec.node:
                tx, rx = (spec, o) if spec.direction == "tx" else (o, spec)
                check_link(tx.fc_hz, tx.rate_sps, rx.fc_hz, rx.rate_sps)
        h = StreamHandle(self._next_id, spec, page_size(spec.rate_sps, self._page_target_s))
        self._next_id += 1
        self._by_name[spec.stream] = h.id
        if spec.direction == "tx":
            self._tx[h.id] = _Tx(h)
        else:
            self._rx[h.id] = _Rx(h)
        return h

    def links(self) -> tuple[LinkId, ...]:
        """Links between open streams (TX order, then RX order)."""
        out = []
        for tx in self._tx.values():
            for rx in self._rx.values():
                if self._linked(tx, rx):
                    out.append(link_id(tx.handle.spec.stream, rx.handle.spec.stream))
        return tuple(out)

    @staticmethod
    def _linked(tx: _Tx, rx: _Rx) -> bool:
        a, b = tx.handle.spec, rx.handle.spec
        if a.node == b.node:
            return False
        return check_link(a.fc_hz, a.rate_sps, b.fc_hz, b.rate_sps)

    def _get_tx(self, h: StreamHandle) -> _Tx:
        if h.id not in self._tx:
            raise StreamError(f"{h.spec.stream!r} is not an open TX stream")
        return self._tx[h.id]

    def _get_rx(self, h: StreamHandle) -> _Rx:
        if h.id not in self._rx:
            raise StreamError(f"{h.spec.stream!r} is not an open RX stream")
        return self._rx[h.id]

    def _rx_frontier_max(self, tx: _Tx) -> int:
        fr = [rx.frontier for rx in self._rx.values() if self._linked(tx, rx)]
        return max(fr) if fr else 0

    def submit_tx(self, h: StreamHandle, samples: np.ndarray, md: TxMetadata) -> int:
        """Queue TX samples; late timed bursts are dropped with async ``TIME_ERROR``."""
        tx = self._get_tx(h)
        f = h.spec.rate_sps
        x = np.ascontiguousarray(samples, dtype=np.complex64).reshape(-1)
        if md.idle_until:
            if x.size:
                raise StreamError("idle_until frames carry no samples")
            tx.idle_until = max(tx.idle_until, to_ticks(md.time_spec, f))
            return 0
        if md.has_time_spec:
            k0 = to_ticks(md.time_spec, f)
        elif tx.in_burst:
            k0 = tx.head
        else:
            k0 = max(tx.head, self._rx_frontier_max(tx))
        if k0 < tx.head or k0 < self._rx_frontier_max(tx):
            self.counters["late_samples"] += int(x.size)
            tx.asyncq.append(
                AsyncMetadata(
                    AsyncCode.TIME_ERROR,
                    h.spec.stream,
                    0,
                    True,
                    TimeSpec.from_ticks(k0, f),
                    k0,
                    int(x.size),
                )
            )
            return 0
        if x.size:
            tx.chunks.append((k0, x.copy()))
        tx.head = k0 + int(x.size)
        tx.in_burst = not (md.end_of_burst or md.end_of_stream)
        tx.silent = md.end_of_burst or md.end_of_stream
        if md.end_of_burst:
            tx.asyncq.append(
                AsyncMetadata(
                    AsyncCode.BURST_ACK,
                    h.spec.stream,
                    0,
                    True,
                    TimeSpec.from_ticks(tx.head, f),
                    tx.head,
                    0,
                )
            )
        return int(x.size)

    def stream_cmd(self, h: StreamHandle, cmd: StreamCmd) -> None:
        """Start or stop RX streaming (UHD stream modes)."""
        rx = self._get_rx(h)
        if cmd.mode == "stop_cont":
            rx.streaming = False
            return
        start = rx.frontier if cmd.stream_now else to_ticks(cmd.time_spec, h.spec.rate_sps)
        if start < rx.frontier:
            rx.pending_error = ErrorCode.LATE_COMMAND
            rx.streaming = False
            return
        rx.streaming = True
        rx.start = start
        rx.burst_started = False
        rx.remaining = None if cmd.mode == "start_cont" else int(cmd.num_samps)
        rx.end_of_burst_at_end = cmd.mode == "num_done"

    def recv_async(self, h: StreamHandle, timeout_s: float) -> AsyncMetadata | None:
        """Next async TX event of ``h`` (BURST_ACK, TIME_ERROR), or None."""
        tx = self._get_tx(h)
        return tx.asyncq.popleft() if tx.asyncq else None

    # ================================================================== RX output
    def _tx_samples(self, tx: _Tx, a: int, b: int) -> np.ndarray:
        out = np.zeros(b - a, dtype=np.complex128)
        for k0, x in tx.chunks:
            lo, hi = max(a, k0), min(b, k0 + x.size)
            if lo < hi:
                out[lo - a : hi - a] += x[lo - k0 : hi - k0]
        return out

    def _available(self, rx: _Rx) -> int | None:
        """First RX index not computable yet (None: unbounded)."""
        c = [
            k
            for tx in self._tx.values()
            if self._linked(tx, rx) and (k := tx.committed()) is not None
        ]
        return min(c) if c else None

    def _active_events(self) -> list[_Event]:
        """Events that take effect, in (t, seq) order (superseded ones never do)."""
        evs = [
            e
            for lst in self._events.values()
            for e in lst
            if e.record.status not in ("rejected", "superseded")
        ]
        evs.sort(key=lambda e: (e.t, e.seq))
        return evs

    def _configs_at_index(self, k: int, f: Fraction) -> dict[PanelId, PanelConfig]:
        """Configurations in effect at RX sample ``k`` at rate ``f``: every update with
        ``to_ticks(t, f) <= k`` folded in (t, seq) order, so element subsets compose by
        time, not by arrival."""
        cfg = dict(self._initial)
        for e in self._active_events():
            if to_ticks(e.t, f) <= k:
                cfg[e.update.panel] = e.update.apply(cfg[e.update.panel])
        return cfg

    def rx_output(self, stream: StreamId, k0: int, k1: int) -> np.ndarray:
        """Pure RX output (complex64) on absolute samples [k0, k1) of ``stream``.

        Does not check commitment and does not advance any frontier.
        """
        rx = self._rx[self._by_name[stream]]
        f = rx.handle.spec.rate_sps
        y = np.zeros(k1 - k0, dtype=np.complex128)
        for tx in self._tx.values():
            if self._linked(tx, rx):
                y += self._tx_samples(tx, k0, k1)
        if self._gain_fn is not None and k1 > k0:
            cuts = sorted(
                {k0, k1} | {k for e in self._active_events() if k0 < (k := to_ticks(e.t, f)) < k1}
            )
            for a, b in itertools.pairwise(cuts):
                y[a - k0 : b - k0] *= complex(self._gain_fn(self._configs_at_index(a, f)))
        return y.astype(np.complex64)

    def read_rx(self, h: StreamHandle, n: int, timeout_s: float) -> tuple[np.ndarray, RxMetadata]:
        """Return up to ``n`` available RX samples (never blocks; TIMEOUT when none)."""
        rx = self._get_rx(h)
        f = h.spec.rate_sps
        md = RxMetadata()
        empty = np.zeros(0, dtype=np.complex64)
        if rx.pending_error != ErrorCode.NONE:
            md.error_code, rx.pending_error = rx.pending_error, ErrorCode.NONE
            return empty, md
        if not rx.streaming or n <= 0:
            md.error_code = ErrorCode.TIMEOUT
            return empty, md
        assert rx.start is not None
        first = max(rx.frontier, rx.start)
        while True:
            limit = self._barrier_limit(rx)
            if limit is None or limit > first or not self._run_due_barrier(rx, first):
                break
        bounds = [first + n, self._available(rx), self._barrier_limit(rx)]
        if rx.remaining is not None:
            bounds.append(first + rx.remaining)
        k1 = min(b for b in bounds if b is not None)
        if k1 <= first:
            md.error_code = ErrorCode.TIMEOUT
            return empty, md
        y = self.rx_output(h.spec.stream, first, k1)
        rx.frontier = k1
        md.has_time_spec = True
        md.time_spec = TimeSpec.from_ticks(first, f)
        md.first_sample = first
        md.start_of_burst = not rx.burst_started
        rx.burst_started = True
        if rx.remaining is not None:
            rx.remaining -= k1 - first
            if rx.remaining == 0:
                rx.streaming = False
                md.end_of_burst = rx.end_of_burst_at_end
        self._update_statuses()
        self._prune()
        return y, md

    def _prune(self) -> None:
        """Drop TX chunks older than the retention horizon behind every linked RX frontier."""
        horizon = self._retain_s + max(
            (Fraction(c.ctrl.spec.window_s) + Fraction(c.ctrl.min_latency_s) for c in self._ctrls),
            default=Fraction(0),
        )
        for tx in self._tx.values():
            fr = [rx.frontier for rx in self._rx.values() if self._linked(tx, rx)]
            if fr:
                lo = min(fr) - ceil_ticks(horizon, tx.handle.spec.rate_sps)
                tx.chunks = [(k0, x) for k0, x in tx.chunks if k0 + x.size > lo]

    # ================================================================== time / caps
    def time_now(self) -> Fraction:
        """min over RX streams of frontier_r / f_r (s); 0 when no RX stream is open."""
        if not self._rx:
            return Fraction(0)
        return min(from_ticks(rx.frontier, rx.handle.spec.rate_sps) for rx in self._rx.values())

    def caps(self) -> Mapping[PanelId, RisCapabilities]:
        """Capabilities of every panel."""
        return dict(self._caps)

    @property
    def records(self) -> tuple[CommandRecord, ...]:
        """Every command record so far, in arrival order."""
        return tuple(self._records)

    def panel_state(self, panel: PanelId, t: Fraction) -> PanelConfig:
        """Configuration of ``panel`` in effect at RIS time ``t`` (s)."""
        cfg = self._initial[panel]
        for e in self._active_events():
            if e.update.panel == panel and e.t <= t:
                cfg = e.update.apply(cfg)
        return cfg

    # ================================================================== commands
    def _frontier_time(self) -> Fraction:
        if not self._rx:
            return Fraction(0)
        return max(from_ticks(rx.frontier, rx.handle.spec.rate_sps) for rx in self._rx.values())

    def _requested_time(self, req: CommandRequest) -> Fraction | None:
        """Requested switch time at the RIS (tau = 0 in the fake); None for an unknown
        ``sample`` stream."""
        if req.at.kind == "t":
            assert req.at.t is not None
            return req.at.t
        if req.at.kind == "sample":
            assert req.at.stream is not None
            assert req.at.index is not None
            if req.at.stream not in self._by_name:
                return None
            return from_ticks(req.at.index, self._spec(req.at.stream).rate_sps)
        return self._frontier_time()

    def _reject(
        self,
        req: CommandRequest,
        code: ErrorCode,
        msg: str,
        t_req: Fraction | None = None,
        k_req: dict[StreamId, int] | None = None,
        config_hash: str = "",
    ) -> CommandReceipt:
        """Count, log (a ``rejected`` CommandRecord with a seq, I6) and refuse a command."""
        self.counters["rejected_commands"] += 1
        self._seq += 1
        self._records.append(
            self._record(
                req,
                self._seq,
                self._frontier_time() if t_req is None else t_req,
                None,
                {} if k_req is None else k_req,
                {},
                "rejected",
                config_hash,
            )
        )
        return CommandReceipt(
            req.cmd_id, False, "rejected", error=code, msg=msg, t_wall_recv_ns=req.t_wall_recv_ns
        )

    def schedule(self, req: CommandRequest) -> CommandReceipt:
        """Schedule a timed command (``ris.set_config`` with ``states``, ``ris.set_enabled``,
        ``rec.marker``); other ops are rejected with ``NOT_PERMITTED``. Every command,
        accepted or rejected, produces a :class:`CommandRecord` (section 4.6, I6)."""
        if req.op == "rec.marker":
            return self._schedule_marker(req)
        t_req = self._requested_time(req)
        if req.op not in ("ris.set_config", "ris.set_enabled"):
            return self._reject(
                req, ErrorCode.NOT_PERMITTED, f"{req.op} is not modelled by FakeEngine", t_req
            )
        if req.target not in self._caps:
            return self._reject(
                req, ErrorCode.UNKNOWN_TARGET, f"unknown panel {req.target!r}", t_req
            )
        if t_req is None:
            return self._reject(req, ErrorCode.UNKNOWN_TARGET, f"unknown stream {req.at.stream!r}")
        rxs = list(self._rx.values())
        k_req = {rx.handle.spec.stream: to_ticks(t_req, rx.handle.spec.rate_sps) for rx in rxs}
        bound = self._lookahead_bound(req)
        if bound is not None and t_req < bound:
            return self._reject(
                req,
                ErrorCode.LOOKAHEAD_VIOLATION,
                f"{req.src}: at = {t_req} < t_b + L_min = {bound}",
                t_req,
                k_req,
            )
        try:
            upd = self._make_update(req)
        except ConfigurationError as exc:
            return self._reject(req, ErrorCode.SCHEMA_ERROR, str(exc), t_req, k_req)
        late = any(k_req[rx.handle.spec.stream] < rx.frontier for rx in rxs)
        t_app = t_req
        if late:
            self.counters["late_commands"] += 1
            if req.src not in self._ctrl_names:
                self.nondeterministic_inputs = True
            if req.late_policy == "reject":
                wanted = upd.apply(self.panel_state(req.target, t_req))
                return self._reject(
                    req,
                    ErrorCode.LATE_COMMAND,
                    "command is late",
                    t_req,
                    k_req,
                    wanted.config_hash(),
                )
            t_app = max(from_ticks(rx.frontier, rx.handle.spec.rate_sps) for rx in rxs)
        k_app = {rx.handle.spec.stream: to_ticks(t_app, rx.handle.spec.rate_sps) for rx in rxs}
        self._seq += 1
        for e in self._events[req.target]:
            if e.t == t_app and e.record.status == "scheduled":
                e.record.status = "superseded"
        # configuration right after this command (all earlier-time updates known now)
        cfg = upd.apply(self.panel_state(req.target, t_app))
        rec = self._record(
            req, self._seq, t_req, t_app, k_req, k_app, "scheduled", cfg.config_hash()
        )
        self._events[req.target].append(_Event(t_app, self._seq, upd, rec))
        self._records.append(rec)
        self._update_statuses()
        return CommandReceipt(
            req.cmd_id,
            True,
            "late_applied" if late else "scheduled",
            seq=self._seq,
            t_req=t_req,
            k_req=k_req,
            t_wall_recv_ns=req.t_wall_recv_ns,
        )

    def _schedule_marker(self, req: CommandRequest) -> CommandReceipt:
        self._seq += 1
        t = req.at.t if req.at.kind == "t" and req.at.t is not None else self._frontier_time()
        k = {
            rx.handle.spec.stream: to_ticks(t, rx.handle.spec.rate_sps) for rx in self._rx.values()
        }
        rec = CommandRecord(
            req.cmd_id,
            self._seq,
            req.src,
            req.op,
            req.target,
            t,
            t,
            k,
            dict(k),
            dict.fromkeys(k, 0),
            req.t_wall_recv_ns,
            None,
            None,
            "applied",
            "",
            None,
            False,
        )
        self._records.append(rec)
        return CommandReceipt(
            req.cmd_id,
            True,
            "scheduled",
            seq=self._seq,
            t_req=t,
            k_req=k,
            t_wall_recv_ns=req.t_wall_recv_ns,
        )

    def _spec(self, stream: StreamId) -> StreamSpec:
        i = self._by_name[stream]
        s = self._tx.get(i) or self._rx.get(i)
        assert s is not None
        return s.handle.spec

    def _make_update(self, req: CommandRequest) -> _Update:
        """Validate a ``ris.*`` command against the panel and turn it into an update."""
        if req.op == "ris.set_enabled":
            assert req.enabled is not None
            return _Update(req.target, enabled=bool(req.enabled))
        if req.states is None:
            raise ConfigurationError("FakeEngine supports only 'states' configurations")
        n = self._caps[req.target].n_elements
        if req.elements == "all":
            if req.states.shape != (n,):
                raise ConfigurationError(f"states must have shape ({n},)")
            return _Update(req.target, states=req.states.copy())
        if isinstance(req.elements, tuple):
            idx = np.asarray(req.elements, dtype=np.int64)
            if idx.shape != req.states.shape or (idx.size and (idx.min() < 0 or idx.max() >= n)):
                raise ConfigurationError("element indices do not match states or are out of range")
            return _Update(req.target, idx=idx, states=req.states.copy())
        raise ConfigurationError("FakeEngine does not model element groups")

    @staticmethod
    def _record(
        req: CommandRequest,
        seq: int,
        t_req: Fraction,
        t_app: Fraction | None,
        k_req: dict[StreamId, int],
        k_app: dict[StreamId, int],
        status: CommandStatus,
        config_hash: str,
    ) -> CommandRecord:
        tr = req.transition
        trans = (
            None
            if tr is None
            else {
                "model": tr.model,
                "duration_s": None if tr.duration_s is None else str(tr.duration_s),
                "classes": tr.classes,
            }
        )
        return CommandRecord(
            cmd_id=req.cmd_id,
            seq=seq,
            src=req.src,
            op=req.op,
            panel=req.target,
            t_req=t_req,
            t_applied=t_app,
            k_req=dict(k_req),
            k_applied=dict(k_app),
            lateness_samples={s: k_app[s] - k_req[s] for s in k_app},
            t_wall_recv_ns=req.t_wall_recv_ns,
            t_wall_computed_ns=None,
            t_wall_visible_ns=None,
            status=status,
            config_hash=config_hash,
            transition=trans,
            counterfactual=req.counterfactual,
        )

    def _update_statuses(self) -> None:
        fr = {rx.handle.spec.stream: rx.frontier for rx in self._rx.values()}
        for lst in self._events.values():
            for e in lst:
                r = e.record
                if r.status == "scheduled" and all(
                    fr.get(s, 0) > k for s, k in r.k_applied.items()
                ):
                    late = any(v != 0 for v in r.lateness_samples.values())
                    r.status = "late_applied" if late else "applied"

    # ================================================================== controllers
    def register_controller(self, ctrl: Controller | RemoteControllerProxy) -> None:
        """Register a barrier controller; in-process controllers get ``setup`` now."""
        if ctrl.name in self._ctrl_names:
            raise ConfigurationError(f"controller {ctrl.name!r} is already registered")
        period = Fraction(ctrl.period_s)
        if period <= 0:
            raise ConfigurationError("controller period_s must be > 0")
        now = self._frontier_time()
        m0 = -((-now.numerator * period.denominator) // (now.denominator * period.numerator))
        if isinstance(ctrl, Controller):
            digest = hashlib.blake2b(f"ctrl/{ctrl.name}".encode(), digest_size=8).digest()
            key = int.from_bytes(digest, "little")
            ctrl.setup(self.caps(), np.random.Generator(np.random.Philox(key=key)))
        self._ctrls.append(_Ctrl(ctrl, int(m0)))
        self._ctrl_names.add(ctrl.name)

    def _lookahead_bound(self, req: CommandRequest) -> Fraction | None:
        """``t_b + L_min`` of the latest barrier of the barrier-registered source ``req.src``
        (section 4.9); None for unregistered sources, before the first barrier, or for
        ``asap`` commands."""
        if req.at.kind == "asap":
            return None
        for c in self._ctrls:
            if c.ctrl.name == req.src and c.t_b_last is not None:
                return c.t_b_last + Fraction(c.ctrl.min_latency_s)
        return None

    def _barrier_limit(self, rx: _Rx) -> int | None:
        """RX index ``to_ticks(t_b + L_min)`` of the next pending barrier (None: no barrier)."""
        f = rx.handle.spec.rate_sps
        lims = [
            to_ticks(c.m * Fraction(c.ctrl.period_s) + Fraction(c.ctrl.min_latency_s), f)
            for c in self._ctrls
        ]
        return min(lims) if lims else None

    def _run_due_barrier(self, rx: _Rx, first: int) -> bool:
        """Run the earliest barrier blocking ``rx`` at ``first``; False if inputs are missing."""
        f = rx.handle.spec.rate_sps
        due = [
            c
            for c in self._ctrls
            if to_ticks(c.m * Fraction(c.ctrl.period_s) + Fraction(c.ctrl.min_latency_s), f)
            <= first
        ]
        if not due:
            return False
        c = min(due, key=lambda c: c.m * Fraction(c.ctrl.period_s))
        t_b = c.m * Fraction(c.ctrl.period_s)
        for r in self._rx.values():
            avail = self._available(r)
            if avail is not None and avail < to_ticks(t_b, r.handle.spec.rate_sps):
                return False
        c.t_b_last = t_b
        obs = self._observation(c.ctrl, t_b)
        lmin = Fraction(c.ctrl.min_latency_s)
        ctrl = c.ctrl
        if isinstance(ctrl, Controller):
            for i, a in enumerate(ctrl.decide(obs)):
                self._schedule_action(ctrl, c.m, i, a, t_b, lmin)
        else:
            ctrl.send_barrier(obs)
            if not ctrl.wait_ack(t_b, self._barrier_timeout_s):
                raise SremuTimeout(f"controller {ctrl.name!r} did not ack barrier t_b = {t_b}")
        c.m += 1
        return True

    def _schedule_action(
        self, ctrl: Controller, m: int, i: int, a: Action, t_b: Fraction, lmin: Fraction
    ) -> None:
        name = ctrl.name
        at = a.at_s if a.at_s is not None else t_b + max(Fraction(ctrl.latency.fixed_s), lmin)
        if at < t_b + lmin:
            raise LookaheadViolation(f"{name}: action at {at} < t_b + L_min = {t_b + lmin}")
        if a.enabled is not None:
            req = CommandRequest(
                f"{name}:{m}:{i}",
                "ris.set_enabled",
                name,
                a.panel,
                CommandTime.at(at),
                enabled=a.enabled,
            )
        else:
            req = CommandRequest(
                f"{name}:{m}:{i}",
                "ris.set_config",
                name,
                a.panel,
                CommandTime.at(at),
                states=a.states,
                phases_rad=a.phases_rad,
                codebook_index=a.codebook_index,
                transition=a.transition,
            )
        receipt = self.schedule(req)
        if receipt.ok:
            return
        if receipt.error == ErrorCode.UNKNOWN_TARGET:
            raise UnknownTarget(receipt.msg)
        raise ConfigurationError(f"{name}: action rejected: {receipt.error.name}: {receipt.msg}")

    def _observation(self, ctrl: Controller | RemoteControllerProxy, t_b: Fraction) -> Observation:
        k_b: dict[StreamId, int] = {}
        power: dict[StreamId, float] = {}
        iq: dict[StreamId, np.ndarray] = {}
        for rx in self._rx.values():
            s = rx.handle.spec
            kb = to_ticks(t_b, s.rate_sps)
            kw = max(0, to_ticks(t_b - ctrl.spec.window_s, s.rate_sps))
            y = self.rx_output(s.stream, kw, kb)
            k_b[s.stream] = kb
            p = float(np.mean(np.abs(y.astype(np.complex128)) ** 2)) if y.size else math.nan
            power[s.stream] = 10.0 * math.log10(p) if p > 0 else (-math.inf if p == 0 else math.nan)
            iq[s.stream] = y
        states = {pid: self.panel_state(pid, t_b).state.copy() for pid in self._caps}
        return Observation(
            t_b=t_b,
            k_b=k_b,
            rx_power_dbm=power,
            positions=None,
            csi_oracle=None,
            iq_window=iq if "iq_window" in ctrl.spec.fields else None,
            ris_state=states,
            pending=(),
        )
