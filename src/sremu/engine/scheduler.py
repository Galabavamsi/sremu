"""Lockstep scheduler: snapshot grid, segments and render jobs (DESIGN.md 4.3-4.5, 4.8).

No wall clock here (I7); stall detection and pacing live in ``emulator.py``/``pacer.py``.

**Snapshot grid (4.4).** Geometry is evaluated at ``t_k = k*T_u`` aligned to absolute time
(``T_u`` from the section 4.4 policy, :func:`sremu.channel.interp.choose_policy_for_model`).
Between two snapshots every path is rendered with its chord delay, exact carrier phase and
complex-linear amplitude (D-1). Static scenes (``T_u`` None) use snapshots only at geometry
events: each interval is rendered as a degenerate interval ``t_a = t_b`` at its start.
Geometry events (``node.set_position``/``node.set_trajectory``) split the grid; the interval
ending at an event uses the snapshot evaluated just before the event was applied (the left
limit).

**Segments (4.3).** A page of a rate group is cut, per RX, at snapshot-interval
boundaries and at every per-RX, per-tile switch sample of every panel command
(:mod:`sremu.engine.commands`); barrier limits and input availability cut it further.
Within a piece the render job is a pure function of (interval, per-tile epochs, command
data), so the output is bitwise independent of where pages, barriers or client chunks cut
the piece (I2a). Sums follow D-3 (links in scenario order, paths by index).

**Jobs.** For each RX, link and piece the :class:`JobBuilder` assembles a ``LinkJob``:
non-tile paths (LOS, mechanisms, structural gap paths with the panel-centre enable flag)
from one ``combine``; every RIS tile path from the epoch its tile is in (the latest command
that reached that tile at that RX): a completed or step epoch renders the amplitudes of its
final mixture; a running non-step transition renders its start mixture plus one
``TransitionTerm`` per transition class and distinct tile switch sample (section 4.7).
Disabled tiles (counterfactual ``ris.set_enabled``) render zero.

**Diffuse multipath (5.11, 6.3).** When the model carries WP3's additive SoS mechanism,
each link adds, per tap, ``sqrt(P_tap)*g(t_n)*x(t_n - tau_tap)``: the envelope is rendered
like a path without carrier phase (amplitude and delay chord between snapshots, the link's
FD filter and LO offset) and multiplied by the unit-power SoS gain ``g`` evaluated at the
absolute sample times (``physics.diffuse.sos_gain_samples``), so it stays a pure function
of the absolute index. It never multiplies the deterministic channel.

**Causality (4.5).** RX samples [k0, k1) need TX committed through
``k1 + h_FD - floor(tau_min*f_s)``; a group advances only as far as its linked TX streams
are committed (partial pages are allowed: segmentation by availability).

**Caches** (performance only; every cached value is a pure function of its key, so a hit
is bitwise the value a fresh evaluation would give; snapshots are keyed by ``(t, left)``,
which identifies them only between geometry events, and the engine clears these caches at
every applied event). Jobs are keyed by (RX, interval,
timeline signature); the signature holds the epoch identities ``(seq, version)`` per tile,
the switch samples of running transitions and the enable flags, not the book version, so a
command that switches later does not invalidate the current piece's jobs. ``combine()`` is
memoized per (snapshot, configuration): the plan, the non-tile amplitudes and an epoch at
the same configuration share one evaluation per snapshot (the plan reads only the path
structure, which every configuration shares). FD specs and link layouts are memoized on
their inputs.
"""

from __future__ import annotations

import bisect
import math
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sremu.channel.fd import choose_fd
from sremu.channel.interp import plan_groups, should_verify, verify_interp
from sremu.contracts import (
    FDSpec,
    LinkId,
    LinkJob,
    LinkPaths,
    PanelConfig,
    PanelId,
    PropagationModel,
    SegmentJob,
    Snapshot,
    StreamId,
    TransitionTerm,
)
from sremu.errors import ConfigurationError
from sremu.timebase import ceil_ticks

from .commands import CommandManager, EpochInfo, mixture_amplitudes
from .streams import StreamConfig, TxStream

__all__ = [
    "Interval",
    "JobBuilder",
    "LinkPlanInfo",
    "RateGroup",
    "SnapshotManager",
    "Step",
    "advance_group",
]

_LOOKBACK_MARGIN = 2  # extra TX history behind the oldest FD tap (render window margin)


# =========================================================================== snapshots


@dataclass(frozen=True)
class Interval:
    """A snapshot interval: ``[t_lo, t_hi)`` in time (``t_hi`` None: open-ended) and the
    render endpoints ``t_a``/``t_b`` (equal for degenerate intervals) with their snapshots.
    ``index`` is the global grid index used by ``verify_interp`` (None when static)."""

    key: tuple[Any, ...]
    t_lo: Fraction
    t_hi: Fraction | None
    t_a: Fraction
    t_b: Fraction
    snap_a: Snapshot
    snap_b: Snapshot
    index: int | None

    @property
    def degenerate(self) -> bool:
        """True when rendered with ``t_a == t_b`` (w = 0)."""
        return self.t_a == self.t_b

    def first(self, f: Fraction) -> int:
        """First RX sample of the interval at rate ``f``."""
        return ceil_ticks(self.t_lo, f)

    def end(self, f: Fraction) -> int | None:
        """First RX sample after the interval at rate ``f`` (None: open-ended)."""
        return None if self.t_hi is None else ceil_ticks(self.t_hi, f)


class SnapshotManager:
    """Snapshot grid and cache (see the module docstring).

    ``t_u``: snapshot period (None: static scene, event-only snapshots). ``block_s``:
    legacy staircase mode (degenerate intervals of ``block_s``; only through
    :mod:`sremu.legacy.block_constant`).
    """

    def __init__(
        self,
        model: PropagationModel,
        *,
        t_u: Fraction | None,
        block_s: Fraction | None = None,
        cache_size: int = 16,
    ) -> None:
        self.model = model
        self.t_u = t_u
        self.block_s = block_s
        self.events: list[Fraction] = []
        self._cache: OrderedDict[tuple[Fraction, bool], Snapshot] = OrderedDict()
        self._left: dict[Fraction, Snapshot] = {}
        self._intervals: OrderedDict[tuple[Any, ...], Interval] = OrderedDict()
        self._cache_size = int(cache_size)
        self.n_snapshots = 0
        self._last: dict[Fraction, tuple[int, int | None, Interval]] = {}
        self._one: tuple[Fraction, int, int | None, Interval] | None = None

    @property
    def mode(self) -> str:
        """``static``, ``grid`` or ``block``."""
        if self.block_s is not None:
            return "block"
        return "static" if self.t_u is None else "grid"

    def snapshot(self, t: Fraction, *, left: bool = False) -> Snapshot:
        """Snapshot at exact time ``t`` (``left``: the limit just before a geometry event)."""
        if left and t in self._left:
            return self._left[t]
        key = (t, False)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        snap = self.model.snapshot(t)
        self.n_snapshots += 1
        self._cache[key] = snap
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return snap

    def apply_event(self, t: Fraction, fn: Callable[[], None]) -> None:
        """Apply a geometry event at ``t``: keep the left-limit snapshot, mutate the model
        via ``fn``, drop cached snapshots and intervals at or after ``t``. A second event at
        the same ``t`` keeps the first left limit (the geometry before every event at t)."""
        if (self.mode != "static" or t != 0) and t not in self._left:
            self._left[t] = self.snapshot(t)
        fn()
        self._last.clear()
        self._one = None
        for key in [k for k in self._cache if k[0] >= t]:
            del self._cache[key]
        for key in [k for k, iv in self._intervals.items() if iv.t_hi is None or iv.t_hi > t]:
            del self._intervals[key]
        bisect.insort(self.events, t)

    def interval_for(self, n: int, f: Fraction) -> Interval:
        """The snapshot interval holding RX sample ``n`` at rate ``f``."""
        one = self._one
        if one is not None and one[0] is f and one[1] <= n and (one[2] is None or n < one[2]):
            return one[3]  # identity fast path: groups pass the same rate object
        last = self._last.get(f)
        if last is not None and last[0] <= n and (last[1] is None or n < last[1]):
            self._one = (f, last[0], last[1], last[2])
            return last[2]
        iv = self._interval_for(n, f)
        self._last[f] = (iv.first(f), iv.end(f), iv)
        self._one = (f, iv.first(f), iv.end(f), iv)
        return iv

    def _interval_for(self, n: int, f: Fraction) -> Interval:
        t = Fraction(n) / f
        if self.block_s is not None:
            m = math.floor(t / self.block_s)
            key: tuple[Any, ...] = ("b", m)
            hit = self._intervals.get(key)
            if hit is None:
                ta = m * self.block_s
                s = self.snapshot(ta)
                hit = Interval(key, ta, ta + self.block_s, ta, ta, s, s, m)
                self._remember(key, hit)
            return hit
        i = bisect.bisect_right(self.events, t)
        ev_lo = self.events[i - 1] if i > 0 else Fraction(0)
        ev_hi = self.events[i] if i < len(self.events) else None
        if self.t_u is None:
            key = ("s", ev_lo)
            hit = self._intervals.get(key)
            if hit is None:
                s = self.snapshot(ev_lo)
                hit = Interval(key, ev_lo, ev_hi, ev_lo, ev_lo, s, s, None)
                self._remember(key, hit)
            return hit
        m = math.floor(t / self.t_u)
        g_lo, g_hi = m * self.t_u, (m + 1) * self.t_u
        lo = max(g_lo, ev_lo)
        hi = g_hi if ev_hi is None else min(g_hi, ev_hi)
        key = ("g", lo, hi)
        hit = self._intervals.get(key)
        if hit is None:
            sa = self.snapshot(lo)
            sb = self.snapshot(hi, left=hi in self._left)
            hit = Interval(key, lo, hi, lo, hi, sa, sb, m)
            self._remember(key, hit)
        return hit

    def _remember(self, key: tuple[Any, ...], iv: Interval) -> None:
        self._intervals[key] = iv
        if len(self._intervals) > self._cache_size:
            self._intervals.popitem(last=False)


# =========================================================================== plans


@dataclass(frozen=True, eq=False)
class LinkLayout:
    """Path bookkeeping of one link: non-tile path indices and, per panel, the indices of
    its RIS tile paths and their tile numbers."""

    n_paths: int
    nontile: NDArray[np.int64]
    panels: tuple[tuple[PanelId, NDArray[np.int64], NDArray[np.int64]], ...]


def link_layout(snap: Snapshot, lp: LinkPaths) -> LinkLayout:
    """Decode ``LinkPaths.tile`` keys: offsets accumulate ``n_rows*n_tiles`` in snapshot
    panel order; tile ``= (key - offset) % n_tiles`` (the D10 flat key)."""
    keys = lp.tile.astype(np.int64)
    nontile = np.flatnonzero(keys < 0)
    panels = []
    off = 0
    for pid, rc in snap.ris.items():
        size = rc.n_rows * rc.n_tiles
        sel = np.flatnonzero((keys >= off) & (keys < off + size))
        if sel.size:
            panels.append(
                (pid, sel.astype(np.int64), ((keys[sel] - off) % rc.n_tiles).astype(np.int64))
            )
        off += size
    return LinkLayout(int(lp.n_paths), nontile.astype(np.int64), tuple(panels))


@dataclass(frozen=True, eq=False)
class LinkPlanInfo:
    """Per (interval, link): delays, FD groups, filter, narrowband flag and causality data."""

    link: LinkId
    tau_a: NDArray[np.float64]
    tau_b: NDArray[np.float64]
    group: NDArray[np.int16]
    fd: FDSpec
    narrowband: bool
    spread_s: float
    tau_min_s: float
    tau_max_s: float
    layout: LinkLayout


@dataclass(frozen=True)
class RxLink:
    """A link into an RX stream: its TX stream, carrier and LO offset."""

    link: LinkId
    tx: StreamId
    fc_hz: float
    lo_offset_hz: float
    bandwidth_hz: float


# =========================================================================== job builder


class JobBuilder:
    """Builds and caches ``LinkJob`` lists per (RX, interval, per-tile epoch signature)."""

    def __init__(
        self,
        model: PropagationModel,
        snaps: SnapshotManager,
        commands: CommandManager,
        tx: Mapping[StreamId, TxStream],
        configs: Mapping[StreamId, StreamConfig],
        *,
        render: Callable[[SegmentJob, NDArray[np.complex128]], None],
        fd_target_db: float = -60.0,
        narrowband_eps: float = 0.05,
        verify: bool = False,
        cache_size: int = 256,
    ) -> None:
        self.model = model
        self.snaps = snaps
        self.commands = commands
        self.tx = tx
        self.configs = configs
        self.render_fn = render
        self.fd_target_db = float(fd_target_db)
        self.nb_eps = float(narrowband_eps)
        self.verify = bool(verify)
        self.links_by_rx: dict[StreamId, list[RxLink]] = {}
        for lk in model.links():
            tx_s, rx_s = lk.split("->")
            if tx_s not in configs or rx_s not in configs:
                continue
            fc = float(model.link_fc_hz(lk))
            lo = float(configs[tx_s].fc_hz) - float(configs[rx_s].fc_hz)
            self.links_by_rx.setdefault(rx_s, []).append(
                RxLink(lk, tx_s, fc, lo, configs[tx_s].bandwidth_hz)
            )
        self._plans: OrderedDict[tuple[Any, ...], dict[LinkId, LinkPlanInfo]] = OrderedDict()
        self._jobs: OrderedDict[tuple[Any, ...], list[LinkJob]] = OrderedDict()
        self._amps: OrderedDict[tuple[Any, ...], dict[LinkId, NDArray[np.complex128]]] = (
            OrderedDict()
        )
        # combine() results per (snapshot key, configuration bytes): plans() of consecutive
        # grid intervals share their common snapshot, and the non-tile amplitudes and an
        # epoch at the same configuration share one evaluation (combine is a pure function
        # of the snapshot and the configuration, so a hit is bitwise the same result)
        self._comb: OrderedDict[tuple[Any, ...], Mapping[LinkId, LinkPaths]] = OrderedDict()
        self._initial_flags = tuple(bool(b.spec.initial.enabled) for b in commands.books.values())
        self._fd: dict[tuple[float, Fraction], FDSpec] = {}
        self._layouts: dict[tuple[Any, ...], LinkLayout] = {}
        self._splits: dict[StreamId, tuple[tuple[int, ...], NDArray[np.int64]]] = {}
        self._needs: OrderedDict[tuple[Any, ...], dict[StreamId, tuple[int, int]]] = OrderedDict()
        self._sigs: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        self._sig_last: dict[StreamId, tuple[tuple[int, ...], int, int, tuple[Any, ...]]] = {}
        self._needs_last: (
            tuple[Interval, tuple[StreamId, ...], dict[StreamId, tuple[int, int]]] | None
        ) = None
        self._cache_size = int(cache_size)
        self.verify_max: dict[LinkId, float] = {}
        self._verified: set[int] = set()
        self.decisions: dict[LinkId, dict[str, Any]] = {}
        # additive diffuse multipath (section 5.11): WP3's mechanism, when the model has one
        self.diffuse = next(
            (
                m
                for m in getattr(model, "mechanisms", ())
                if callable(getattr(m, "terms", None)) and callable(getattr(m, "link_terms", None))
            ),
            None,
        )

    # ------------------------------------------------------------------ caches
    def clear(self) -> None:
        """Drop cached jobs and amplitudes (after a ring reallocation or model change)."""
        self._jobs.clear()
        self._amps.clear()
        self._comb.clear()
        self._plans.clear()
        self._needs.clear()
        self._sigs.clear()
        self._needs_last = None

    def _lru_put(
        self, cache: OrderedDict[Any, Any], key: Any, value: Any, size: int | None = None
    ) -> None:
        cache[key] = value
        if len(cache) > (size or self._cache_size):
            cache.popitem(last=False)

    # ------------------------------------------------------------------ combine
    def _combine(
        self, snap: Snapshot, skey: tuple[Any, ...], cfg: Mapping[PanelId, PanelConfig]
    ) -> Mapping[LinkId, LinkPaths]:
        """``model.combine(snap, cfg)``, memoized per (snapshot key, configuration)."""
        key = (
            skey,
            tuple((pid, c.state.tobytes(), bool(c.enabled)) for pid, c in cfg.items()),
        )
        hit = self._comb.get(key)
        if hit is None:
            hit = self.model.combine(snap, cfg)
            self._lru_put(self._comb, key, hit, 64)
        return hit

    # ------------------------------------------------------------------ plans
    def plans(self, iv: Interval) -> dict[LinkId, LinkPlanInfo]:
        """Delays, FD plan and layout of every link for an interval (config independent)."""
        hit = self._plans.get(iv.key)
        if hit is not None:
            return hit
        # only the path structure is read here (delays, path count, tile keys), and it does
        # not depend on the configuration: a disabled or unconfigured panel keeps its paths
        # with zero amplitude, which jobs() relies on when it assembles amplitudes from
        # several configurations into one path-indexed array. The non-tile configuration at
        # the initial enable flags is the one _nontile() evaluates for the same snapshot,
        # so the plan shares that combine() instead of running one of its own.
        cfg = self._nontile_cfg(self._initial_flags)
        pa = self._combine(iv.snap_a, self._snap_key(iv.snap_a, iv, "a"), cfg)
        pb = (
            pa
            if iv.snap_b is iv.snap_a
            else self._combine(iv.snap_b, self._snap_key(iv.snap_b, iv, "b"), cfg)
        )
        out: dict[LinkId, LinkPlanInfo] = {}
        for links in self.links_by_rx.values():
            for rl in links:
                a, b = pa[rl.link], pb[rl.link]
                if a.n_paths != b.n_paths:
                    raise ConfigurationError(f"link {rl.link}: path count changes in an interval")
                group, nb = plan_groups(a.delay_s, b.delay_s, rl.bandwidth_hz, self.nb_eps)
                fkey = (rl.bandwidth_hz, self.configs[rl.tx].rate)
                fd = self._fd.get(fkey)
                if fd is None:  # a pure function of (bandwidth, rate, target)
                    fd = choose_fd(rl.bandwidth_hz, self.configs[rl.tx].rate, self.fd_target_db)
                    self._fd[fkey] = fd
                spread = float(max(np.ptp(a.delay_s), np.ptp(b.delay_s))) if a.n_paths else 0.0
                tmin = float(min(a.delay_s.min(), b.delay_s.min())) if a.n_paths else 0.0
                tmax = float(max(a.delay_s.max(), b.delay_s.max())) if a.n_paths else 0.0
                info = LinkPlanInfo(
                    rl.link,
                    np.ascontiguousarray(a.delay_s, np.float64),
                    np.ascontiguousarray(b.delay_s, np.float64),
                    group,
                    fd,
                    nb,
                    spread,
                    tmin,
                    tmax,
                    self._layout(iv.snap_a, a),
                )
                out[rl.link] = info
                if rl.link not in self.decisions:
                    self.decisions[rl.link] = {
                        "narrowband": bool(nb),
                        "fd_filter": fd.filter_id,
                        "delay_spread_s": spread,
                        "n_groups": int(np.unique(group).size) if group.size else 0,
                    }
        self._lru_put(self._plans, iv.key, out, 32)
        if self.verify and iv.index is not None and not iv.degenerate:
            self._verify(iv)
        return out

    def _verify(self, iv: Interval) -> None:
        """``verify_interp`` on every 64th grid interval (section 4.4)."""
        assert iv.index is not None
        if iv.index in self._verified or not should_verify(iv.index):
            return
        self._verified.add(iv.index)
        cfg = {pid: b.config_at(iv.t_a) for pid, b in self.commands.books.items()}
        fcs = {rl.link: rl.fc_hz for links in self.links_by_rx.values() for rl in links}
        errs = verify_interp(
            self.model,
            cfg,
            iv.t_a,
            iv.t_b,
            fcs,
            snap_a=iv.snap_a,
            snap_b=iv.snap_b,
            links=list(fcs),
        )
        for lk, e in errs.items():
            self.verify_max[lk] = max(self.verify_max.get(lk, 0.0), float(e))

    # ------------------------------------------------------------------ amplitudes
    def _snap_key(self, snap: Snapshot, iv: Interval, side: str) -> tuple[Any, ...]:
        t = iv.t_a if side == "a" else iv.t_b
        left = (
            side == "b" and iv.t_hi is not None and iv.t_b in self.snaps._left and not iv.degenerate
        )
        return (t, left)

    def _layout(self, snap: Snapshot, lp: LinkPaths) -> LinkLayout:
        """:func:`link_layout`, memoized on its inputs (tile keys and panel structure)."""
        key = (
            lp.tile.tobytes(),
            int(lp.n_paths),
            tuple((pid, rc.n_rows, rc.n_tiles) for pid, rc in snap.ris.items()),
        )
        hit = self._layouts.get(key)
        if hit is None:
            hit = link_layout(snap, lp)
            if len(self._layouts) > 1024:
                self._layouts.clear()
            self._layouts[key] = hit
        return hit

    def _nontile_cfg(self, flags: tuple[bool, ...]) -> dict[PanelId, PanelConfig]:
        """Every panel at its initial states with the given (panel-centre) enable flags."""
        return {
            pid: PanelConfig(pid, b.spec.initial.state, flag)
            for (pid, b), flag in zip(self.commands.books.items(), flags, strict=True)
        }

    def _nontile(
        self, snap: Snapshot, skey: tuple[Any, ...], flags: tuple[bool, ...]
    ) -> dict[LinkId, NDArray[np.complex128]]:
        key = ("nt", skey, flags)
        hit = self._amps.get(key)
        if hit is None:
            cfg = self._nontile_cfg(flags)
            hit = {lk: lp.amp for lk, lp in self._combine(snap, skey, cfg).items()}
            self._lru_put(self._amps, key, hit)
        return hit

    def _epoch_amp(
        self, snap: Snapshot, skey: tuple[Any, ...], pid: PanelId, ep: EpochInfo, which: Any
    ) -> dict[LinkId, NDArray[np.complex128]]:
        key = ("ep", skey, pid, ep.key, which)
        hit = self._amps.get(key)
        if hit is None:
            book = self.commands.books[pid]

            def fn(
                states: NDArray[np.int16], snap: Snapshot = snap, pid: PanelId = pid
            ) -> Mapping[LinkId, NDArray[np.complex128]]:
                # panel_combine_fn through the memo: only ``pid`` configured, enabled
                cfg = {pid: PanelConfig(pid, np.asarray(states, np.int16), True)}
                return {lk: lp.amp for lk, lp in self._combine(snap, skey, cfg).items()}

            if which == "after":
                mix = ep.after()
            elif which == "start":
                mix = ep.start
            else:
                c = int(which[1])
                mix = ep.start.apply([ep.classes[c]], ep.target, [1.0])
            hit = mixture_amplitudes(mix, book.spec.n_states, fn)
            self._lru_put(self._amps, key, hit)
        return hit

    # ------------------------------------------------------------------ signatures
    def _versions(self) -> tuple[int, ...]:
        return tuple(b.version for b in self.commands.books.values())

    def split_points(self, rx: StreamId) -> NDArray[np.int64]:
        """Every command switch sample of every panel at ``rx`` (sorted, unique)."""
        vers = self._versions()
        hit = self._splits.get(rx)
        if hit is not None and hit[0] == vers:
            return hit[1]
        books = list(self.commands.books.values())
        if len(books) == 1:
            sp = books[0].view(rx).splits  # already sorted and unique
        elif books:
            sp = np.unique(np.concatenate([b.view(rx).splits for b in books]))
        else:
            sp = np.zeros(0, np.int64)
        self._splits[rx] = (vers, sp)
        return sp

    def signature(self, rx: StreamId, a: int) -> tuple[Any, ...]:
        """Timeline signature of ``rx`` at sample ``a`` (see :meth:`_signature`).

        Constant between consecutive switch samples, so it is cached per (versions, split
        index); the last one per RX is kept with its sample range ``[lo, hi)``."""
        vers = self._versions()
        last = self._sig_last.get(rx)
        if last is not None and last[0] == vers and last[1] <= a < last[2]:
            return last[3]
        splits = self.split_points(rx)
        n = int(splits.shape[0])
        i = int(np.searchsorted(splits, a, side="right")) if n else 0
        lo = int(splits[i - 1]) if i > 0 else -(1 << 62)
        hi = int(splits[i]) if i < n else 1 << 62
        key = (rx, vers, i)
        sig_t = self._sigs.get(key)
        if sig_t is None:
            if len(self._sigs) > 4096:
                self._sigs.clear()
            sig_t = self._signature(rx, a)
            self._sigs[key] = sig_t
        self._sig_last[rx] = (vers, lo, hi, sig_t)
        return sig_t

    def _signature(self, rx: StreamId, a: int) -> tuple[Any, ...]:
        """Everything :meth:`jobs` reads from the command timeline at sample ``a``: per panel
        and tile the identity ``(seq, version)`` of the epoch in force (a version changes
        whenever an epoch's start, classes or target change), the switch samples of every
        running non-step epoch at this RX (its transition terms start there), and the
        per-tile and panel-centre enable flags. It does not contain the book version, so
        inserting a command that switches later (or pruning old ones) keeps the cached jobs
        of the current piece valid."""
        sig = []
        for pid, book in self.commands.books.items():
            v = book.view(rx)
            ep = v.epochs_at(a)
            en, enc = v.enabled_at(a)
            eps = v.cfg_epochs
            idx = ep.tolist()
            keys = tuple(eps[e].key for e in idx)
            trans = tuple(
                (eps[e].key, v.cfg_k[e - 1].tobytes())
                for e in sorted(set(idx))
                if not eps[e].is_step
            )
            sig.append((pid, keys, trans, en.tobytes(), enc))
        return tuple(sig)

    # ------------------------------------------------------------------ jobs
    def jobs(self, rx: StreamId, f: Fraction, iv: Interval, a: int) -> list[LinkJob]:
        """``LinkJob`` list of ``rx`` for the piece starting at sample ``a`` of ``iv``."""
        sig = self.signature(rx, a)
        key = (rx, iv.key, sig)
        hit = self._jobs.get(key)
        if hit is not None:
            self._jobs.move_to_end(key)
            return hit
        plans = self.plans(iv)
        books = self.commands.books
        views = {pid: books[pid].view(rx) for pid in books}
        ska = self._snap_key(iv.snap_a, iv, "a")
        skb = self._snap_key(iv.snap_b, iv, "b")
        centre_flags = tuple(bool(s[4]) for s in sig)
        nt_a = self._nontile(iv.snap_a, ska, centre_flags)
        nt_b = nt_a if iv.degenerate else self._nontile(iv.snap_b, skb, centre_flags)
        # per panel (the same for every link of this RX): the epoch of each tile at ``a``
        # and, per distinct epoch, the tiles that are in it and enabled
        tile_groups: dict[PanelId, list[tuple[int, NDArray[np.bool_]]]] = {}
        for pid, v in views.items():
            eps_tiles = v.epochs_at(a)
            en_tiles, _enc = v.enabled_at(a)
            tile_groups[pid] = [(int(e), (eps_tiles == e) & en_tiles) for e in np.unique(eps_tiles)]
        out: list[LinkJob] = []
        for rl in self.links_by_rx.get(rx, []):
            pl = plans[rl.link]
            lay = pl.layout
            amp_a = np.zeros(lay.n_paths, dtype=np.complex128)
            amp_b = np.zeros(lay.n_paths, dtype=np.complex128)
            amp_a[lay.nontile] = nt_a[rl.link][lay.nontile]
            amp_b[lay.nontile] = nt_b[rl.link][lay.nontile]
            terms: list[TransitionTerm] = []
            for pid, idx, tiles in lay.panels:
                v = views[pid]
                for e, on in tile_groups[pid]:
                    sel = on[tiles]
                    if not sel.any():
                        continue
                    pidx = idx[sel]
                    ep = v.cfg_epochs[int(e)]
                    if ep.is_step:
                        ea = self._epoch_amp(iv.snap_a, ska, pid, ep, "after")
                        eb = (
                            ea
                            if iv.degenerate
                            else self._epoch_amp(iv.snap_b, skb, pid, ep, "after")
                        )
                        amp_a[pidx] = ea[rl.link][pidx]
                        amp_b[pidx] = eb[rl.link][pidx]
                        continue
                    sa = self._epoch_amp(iv.snap_a, ska, pid, ep, "start")
                    sb = sa if iv.degenerate else self._epoch_amp(iv.snap_b, skb, pid, ep, "start")
                    amp_a[pidx] = sa[rl.link][pidx]
                    amp_b[pidx] = sb[rl.link][pidx]
                    ks_tile = v.cfg_k[int(e) - 1]  # (K,) switch samples of this epoch
                    path_k = ks_tile[tiles[sel]]
                    dur = float(ep.duration_s * f) if ep.duration_s else 0.0
                    for c in range(len(ep.classes)):
                        ca = self._epoch_amp(iv.snap_a, ska, pid, ep, ("class", c))
                        cb = (
                            ca
                            if iv.degenerate
                            else self._epoch_amp(iv.snap_b, skb, pid, ep, ("class", c))
                        )
                        da_full = ca[rl.link] - sa[rl.link]
                        db_full = cb[rl.link] - sb[rl.link]
                        for kk in np.unique(path_k):
                            m = pidx[path_k == kk]
                            da = np.zeros(lay.n_paths, dtype=np.complex128)
                            db = np.zeros(lay.n_paths, dtype=np.complex128)
                            da[m] = da_full[m]
                            db[m] = db_full[m]
                            terms.append(TransitionTerm(int(kk), ep.profile, dur, da, db))
            out.append(
                LinkJob(
                    link=rl.link,
                    ring=self.tx[rl.tx].ring,
                    fc_hz=rl.fc_hz,
                    lo_offset_hz=rl.lo_offset_hz,
                    amp_a=amp_a,
                    amp_b=amp_b,
                    tau_a=pl.tau_a,
                    tau_b=pl.tau_b,
                    group=pl.group,
                    fd=pl.fd,
                    trans=terms,
                )
            )
        self._lru_put(self._jobs, key, out)
        return out

    # ------------------------------------------------------------------ rendering
    def render_range(
        self, rx: StreamId, f: Fraction, k0: int, k1: int, acc: NDArray[np.complex128]
    ) -> int:
        """Render RX samples [k0, k1) of ``rx`` into ``acc`` (adds); returns the piece count."""
        pos = k0
        pieces = 0
        splits = self.split_points(rx)
        while pos < k1:
            iv = self.snaps.interval_for(pos, f)
            end = k1
            e_iv = iv.end(f)
            if e_iv is not None:
                end = min(end, e_iv)
            if splits.size:
                i = int(np.searchsorted(splits, pos, side="right"))
                if i < splits.size:
                    end = min(end, int(splits[i]))
            if end <= pos:  # pragma: no cover - defensive
                raise ConfigurationError(f"{rx}: empty piece at {pos}")
            links = self.jobs(rx, f, iv, pos)
            if links:
                job = SegmentJob(rx, f, pos, end, iv.t_a, iv.t_b, links)
                self.render_fn(job, acc[pos - k0 : end - k0])
            if self.diffuse is not None:
                self._render_diffuse(rx, f, iv, pos, end, acc[pos - k0 : end - k0])
            pieces += 1
            pos = end
        return pieces

    # ------------------------------------------------------------------ diffuse
    def _diffuse_terms(self, snap: Snapshot, skey: tuple[Any, ...]) -> dict[LinkId, Any]:
        key = ("diffuse", skey)
        hit = self._amps.get(key)
        if hit is None:
            assert self.diffuse is not None
            hit = dict(self.diffuse.terms(self.model, snap))
            self._lru_put(self._amps, key, hit)
        return hit

    def diffuse_jobs(self, rx: StreamId, iv: Interval) -> list[tuple[LinkJob, Any]]:
        """Per link and tap of ``rx``: the envelope job of the diffuse term (amplitude
        ``sqrt(P_tap)`` and envelope delay chord between the snapshots, no carrier phase,
        the link's FD filter and LO offset) and its term (for the SoS gain)."""
        key = ("djobs", rx, iv.key)
        hit = self._jobs.get(key)
        if hit is not None:
            return hit  # type: ignore[return-value]
        plans = self.plans(iv)
        ta = self._diffuse_terms(iv.snap_a, self._snap_key(iv.snap_a, iv, "a"))
        tb = (
            ta
            if iv.degenerate
            else self._diffuse_terms(iv.snap_b, self._snap_key(iv.snap_b, iv, "b"))
        )
        out: list[tuple[LinkJob, Any]] = []
        for rl in self.links_by_rx.get(rx, []):
            for da, db in zip(ta.get(rl.link, ()), tb.get(rl.link, ()), strict=True):
                lj = LinkJob(
                    link=f"{rl.link}#diffuse{int(da.tap)}",
                    ring=self.tx[rl.tx].ring,
                    fc_hz=0.0,  # the SoS phase is the term's only carrier phase
                    lo_offset_hz=rl.lo_offset_hz,
                    amp_a=np.array([da.amp], dtype=np.complex128),
                    amp_b=np.array([db.amp], dtype=np.complex128),
                    tau_a=np.array([da.delay_s], dtype=np.float64),
                    tau_b=np.array([db.delay_s], dtype=np.float64),
                    group=np.zeros(1, dtype=np.int16),
                    fd=plans[rl.link].fd,
                )
                out.append((lj, da))
        self._lru_put(self._jobs, key, out)
        return out

    def _render_diffuse(
        self, rx: StreamId, f: Fraction, iv: Interval, a: int, b: int, out: NDArray[np.complex128]
    ) -> None:
        from sremu.physics.diffuse import sos_gain_samples

        for lj, term in self.diffuse_jobs(rx, iv):
            env = np.zeros(b - a, dtype=np.complex128)
            self.render_fn(SegmentJob(rx, f, a, b, iv.t_a, iv.t_b, [lj]), env)
            g = sos_gain_samples(
                term.key, a, b - a, f, term.max_doppler_hz, term.n_sinusoids, tap=term.tap
            )
            out += env * g

    # ------------------------------------------------------------------ causality
    def tx_needs(
        self, rx_list: Sequence[StreamId], f: Fraction, k0: int, k1: int
    ) -> dict[StreamId, tuple[int, int]]:
        """Per TX stream: ``(lookahead, lookback)`` in samples for RX samples [k0, k1).

        ``lookahead = h_FD - floor(tau_min*f)`` (TX needed through ``k1 + lookahead``,
        section 4.5); ``lookback = ceil(tau_max*f) + L_FD + margin`` (history behind k0).
        """
        rxs = tuple(rx_list)
        iv0 = self.snaps.interval_for(k0, f)
        e0 = iv0.end(f)
        if e0 is None or e0 >= k1:  # one interval: cached per (interval, receivers)
            last = self._needs_last
            if last is not None and last[0] is iv0 and last[1] == rxs:
                return last[2]
            res = self._iv_needs(iv0, rxs, f)
            self._needs_last = (iv0, rxs, res)
            return res
        out: dict[StreamId, tuple[int, int]] = {}
        pos = k0
        while pos < k1:
            iv = self.snaps.interval_for(pos, f)
            for t, (la, lb) in self._iv_needs(iv, rxs, f).items():
                prev = out.get(t)
                out[t] = (la, lb) if prev is None else (max(prev[0], la), max(prev[1], lb))
            e = iv.end(f)
            if e is None or e >= k1:
                break
            pos = e
        return out

    def _iv_needs(
        self, iv: Interval, rxs: tuple[StreamId, ...], f: Fraction
    ) -> dict[StreamId, tuple[int, int]]:
        key = (iv.key, rxs, f)
        hit = self._needs.get(key)
        if hit is not None:
            return hit
        plans = self.plans(iv)
        out: dict[StreamId, tuple[int, int]] = {}
        for rx in rxs:
            for rl in self.links_by_rx.get(rx, []):
                pl = plans[rl.link]
                # section 4.5: TX committed through k1 + h_FD - floor(tau_min*f) suffices
                # (the renderer's window margin reads unused samples, which read as 0)
                la = pl.fd.lookahead - math.floor(pl.tau_min_s * float(f))
                lb = math.ceil(pl.tau_max_s * float(f)) + pl.fd.taps + _LOOKBACK_MARGIN + 1
                prev = out.get(rl.tx)
                out[rl.tx] = (la, lb) if prev is None else (max(prev[0], la), max(prev[1], lb))
            if self.diffuse is not None:
                # diffuse taps arrive at tau_LOS + tap delay: they can need more TX history
                # than the deterministic paths (never more lookahead: tau_LOS >= tau_min)
                for lj, _term in self.diffuse_jobs(rx, iv):
                    tx_s = lj.link.split("->")[0]
                    tmin = float(min(lj.tau_a.min(), lj.tau_b.min()))
                    tmax = float(max(lj.tau_a.max(), lj.tau_b.max()))
                    la = lj.fd.lookahead - math.floor(tmin * float(f))
                    lb = math.ceil(tmax * float(f)) + lj.fd.taps + _LOOKBACK_MARGIN + 1
                    prev = out.get(tx_s)
                    out[tx_s] = (la, lb) if prev is None else (max(prev[0], la), max(prev[1], lb))
        self._lru_put(self._needs, key, out, 64)
        return out


# =========================================================================== groups


@dataclass
class RateGroup:
    """RX streams with equal rate and page size; one cursor (first uncomputed sample)."""

    rate: Fraction
    page: int
    rx: list[StreamId]
    tx: list[StreamId] = field(default_factory=list)
    cursor: int = 0
    end: int | None = None
    waiting_on: tuple[StreamId, ...] = ()


@dataclass
class Step:
    """Outcome of :func:`advance_group`: progress flag, range, skip flag and wait reason."""

    progress: bool
    k0: int = 0
    k1: int = 0
    skipped: bool = False
    reason: str | None = None
    waiting_on: tuple[StreamId, ...] = ()
    zero_filled: dict[StreamId, int] = field(default_factory=dict)


def advance_group(
    g: RateGroup,
    builder: JobBuilder,
    tx: Mapping[StreamId, TxStream],
    *,
    limit: int | None,
    active: Callable[[StreamId, int, int], bool],
    post: Callable[[StreamId, int, NDArray[np.complex128]], None],
    zero_fill: bool = False,
) -> Step:
    """Advance one rate group by at most one page portion (section 4.8 loop body).

    ``limit``: barrier frontier limit (exclusive) or None. ``active(rx, k0, k1)``: whether an
    RX must be computed on [k0, k1) (otherwise the page is skipped, D-6). ``post(rx, k0,
    acc)``: RX chain and delivery. With ``zero_fill`` missing TX samples are committed as
    zeros (stall policy, paced underflow) and counted in ``Step.zero_filled``.
    """
    k0 = g.cursor
    k1 = (k0 // g.page + 1) * g.page
    if limit is not None:
        k1 = min(k1, limit)
    if g.end is not None:
        k1 = min(k1, g.end)
    if k1 <= k0:
        return Step(False, k0, k0, reason="barrier" if limit is not None and limit <= k0 else "end")
    act = [r for r in g.rx if active(r, k0, k1)]
    if not act:
        for t in g.tx:
            tx[t].drop_before(k1 - tx[t].ring.capacity // 2)
        g.cursor = k1
        return Step(True, k0, k1, skipped=True)
    needs = builder.tx_needs(act, g.rate, k0, k1)
    avail = k1
    waiting: list[StreamId] = []
    for t, (la, _lb) in needs.items():
        com = tx[t].committed()
        if com is None:
            continue
        if com < k1 + la:
            avail = min(avail, com - la)
            waiting.append(t)
    zeros: dict[StreamId, int] = {}
    if zero_fill:
        # stall/underflow policy: whatever is missing at this point is silence, so the
        # whole page portion is computed now (paced mode accounts one page per release)
        avail = k1
    elif avail <= k0:
        g.waiting_on = tuple(waiting)
        return Step(False, k0, k0, reason="tx", waiting_on=tuple(waiting))
    k1 = avail
    g.waiting_on = ()
    rebuilt = False
    for t, (la, lb) in needs.items():
        st = tx[t]
        span = (k1 + la) - (k0 - lb) + 1024
        if span > st.ring.capacity:
            st.ensure_capacity(span)
            rebuilt = True
        z = st.write_through(k1 + la, zero_fill=zero_fill)
        if z:
            zeros[t] = z
    if rebuilt:
        builder.clear()
    for r in act:
        acc = np.zeros(k1 - k0, dtype=np.complex128)
        builder.render_range(r, g.rate, k0, k1, acc)
        post(r, k0, acc)
    g.cursor = k1
    return Step(True, k0, k1, zero_filled=zeros)
