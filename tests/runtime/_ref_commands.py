"""Reference (brute-force) command timeline for equivalence tests (not collected).

:class:`RefPanelBook` and :class:`RefManager` keep the straightforward implementation of
:mod:`sremu.engine.commands` that the incremental one replaced: every insert re-sorts the
book and recomposes every active config command, every query scans the active list, every
view is rebuilt from scratch, and every status update scans all scheduled commands. The
code of each overridden method is the pre-change engine code; ``schedule()``,
``switch_samples()`` and the request-time logic are inherited from the engine, so this
reference checks the book and status bookkeeping, not ``schedule()`` itself.
``test_rt_perf_commands.py`` drives both implementations with the same requests and
frontiers and requires identical state.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

import numpy as np

from sremu.contracts import CommandRecord, PanelConfig, StreamId
from sremu.engine.commands import (
    CommandManager,
    EpochInfo,
    Mixture,
    PanelBook,
    PanelCommand,
    RxView,
    _classes,
    _same_mixture,
)


class RefPanelBook(PanelBook):
    """Pre-change panel book: full re-sort and recompose per insert, scans per query."""

    def active_config(self) -> list[PanelCommand]:
        return [c for c in self.config if not c.superseded]

    def active_enabled(self) -> list[PanelCommand]:
        return [c for c in self.enabled if not c.superseded]

    def latest_config_at(self, t: Fraction) -> PanelCommand | None:
        act = [c for c in self.active_config() if c.t <= t]
        return act[-1] if act else None

    def next_time_after(self, t: Fraction) -> Fraction | None:
        ts = [c.t for c in self.active_config() + self.active_enabled() if c.t > t]
        return min(ts) if ts else None

    def insert(self, cmd: PanelCommand) -> list[PanelCommand]:
        lst = self.config if cmd.kind == "config" else self.enabled
        sup = [c for c in lst if c.t == cmd.t and not c.superseded and c.seq < cmd.seq]
        for c in sup:
            c.superseded = True
            c.record.status = "superseded"
        lst.append(cmd)
        lst.sort(key=lambda c: (c.t, c.seq))
        self._recompose()
        return sup

    def switch_samples_changed(self) -> None:
        self.version += 1

    def _recompose(self) -> None:
        spec = self.spec
        prev: PanelCommand | None = self._head
        if prev is None:
            controls = spec.initial_controls.astype(np.int16).copy()
            target = spec.initial.state.astype(np.int16).copy()
        else:
            assert prev.controls is not None
            assert prev.target is not None
            controls = prev.controls.copy()
            target = prev.target.copy()
        mix = Mixture(target.copy())
        for c in self.active_config():
            if prev is not None:
                mix = prev.frozen(c.t - prev.t)
            if c.physical is not None:
                new_target = np.asarray(c.physical, np.int16).copy()
            else:
                assert c.update_vals is not None
                if c.update_idx is None:
                    controls = c.update_vals.astype(np.int16).copy()
                else:
                    controls = controls.copy()
                    controls[c.update_idx] = c.update_vals
                new_target = spec.expand(controls)
            masks = _classes(mix, target, new_target, spec.n_states, c.transition.classes)
            changed = (
                c.target is None
                or not np.array_equal(c.target, new_target)
                or c.start is None
                or not _same_mixture(c.start, mix)
                or len(c.classes) != len(masks)
                or any(not np.array_equal(a, b) for a, b in zip(c.classes, masks, strict=True))
            )
            c.controls = controls.copy()
            c.last = target.copy()
            if changed:
                self._ver_counter += 1
                c.version = self._ver_counter
                c.target = new_target
                c.start = mix
                c.classes = masks
            target = new_target
            prev = c
        self.version += 1

    def epochs(self) -> tuple[EpochInfo, ...]:
        out = [self._initial_epoch]
        for c in self.active_config():
            assert c.start is not None
            assert c.target is not None
            out.append(
                EpochInfo(
                    (c.seq, c.version),
                    c.start,
                    tuple(c.classes),
                    c.target,
                    c.profile,
                    c.duration_s,
                    c.is_step or not c.classes,
                )
            )
        return tuple(out)

    def view(self, rx: StreamId) -> RxView:
        hit = self._views.get(rx)
        if hit is not None and hit[0] == self.version:
            return hit[1]
        cfg = self.active_config()
        en = self.active_enabled()
        k = self.n_tiles
        cfg_k = (
            np.stack([c.k_tiles[rx] for c in cfg]).astype(np.int64)
            if cfg
            else np.zeros((0, k), np.int64)
        )
        en_k = (
            np.stack([c.k_tiles[rx] for c in en]).astype(np.int64)
            if en
            else np.zeros((0, k), np.int64)
        )
        en_kc = np.array([c.k_centre[rx] for c in en], dtype=np.int64)
        en_vals = np.array([bool(c.enabled) for c in en], dtype=bool)
        splits = np.unique(np.concatenate([cfg_k.ravel(), en_k.ravel(), en_kc]))
        v = RxView(
            cfg_k, self.epochs(), en_k, en_kc, en_vals, bool(self.spec.initial.enabled), splits
        )
        self._views[rx] = (self.version, v)
        return v

    def config_at(self, t: Fraction) -> PanelConfig:
        st = self.spec.initial.state if self._head is None else self._head.target
        for c in self.active_config():
            if c.t <= t:
                assert c.target is not None
                st = c.target
        en = self._head_enabled
        for c in self.active_enabled():
            if c.t <= t:
                en = bool(c.enabled)
        assert st is not None
        return PanelConfig(self.spec.panel, np.asarray(st, np.int16).copy(), en)

    def pending_after(self, t: Fraction) -> list[PanelCommand]:
        return [c for c in self.active_config() + self.active_enabled() if c.t > t]

    def prune(self, frontier: Mapping[StreamId, int], keep: int = 256) -> None:
        if len(self.config) <= keep:
            return
        act = self.active_config()
        if len(act) <= keep:
            return
        cut = 0
        for i in range(len(act) - 1):
            nxt = act[i + 1]
            if all(int(nxt.k_tiles[r].max()) < f for r, f in frontier.items() if r in nxt.k_tiles):
                cut = i + 1
            else:
                break
        if cut < 1:
            return
        head = act[cut - 1]
        drop = {id(c) for c in act[:cut]}
        self.config = [
            c for c in self.config if id(c) not in drop and not (c.superseded and c.t <= head.t)
        ]
        self._head = head
        self._initial_epoch = EpochInfo(
            (-2, head.seq),
            head.after(self.spec.n_states),
            (),
            head.target.copy(),  # type: ignore[union-attr]
            "step",
            None,
            True,
        )
        self._recompose()


class RefManager(CommandManager):
    """Pre-change status bookkeeping: one list of scheduled commands, scanned per update."""

    def __init__(self, *a: object, **kw: object) -> None:
        super().__init__(*a, **kw)  # type: ignore[arg-type]
        self.books = {
            pid: RefPanelBook(b.spec, b.n_tiles)  # type: ignore[misc]
            for pid, b in self.books.items()
        }
        self._scheduled: list[PanelCommand] = []

    @property
    def has_pending(self) -> bool:
        return bool(self._scheduled)

    def _track(self, c: PanelCommand) -> None:
        self._scheduled.append(c)

    def _untrack(self, seq: int) -> None:  # superseded commands stay until the next sweep
        pass

    def _tau_exact(self, panel: str, rx: StreamId, t: Fraction, tile: int | None) -> Fraction:
        return Fraction(self.tau(panel, rx, t, tile))

    def geometry_changed(self, t: Fraction) -> None:
        import bisect

        bisect.insort(self._geometry_events, t)
        self._tau_cache.clear()
        for pid, book in self.books.items():
            touched = False
            for c in book.config + book.enabled:
                if c.t < t or c.superseded or c.record.status != "scheduled":
                    continue
                ks, kc = self.switch_samples(pid, c.t)
                c.k_tiles, c.k_centre = ks, kc
                rec = c.record
                rec.k_applied = dict(kc)
                if c.t_req >= t:
                    rec.k_req = dict(self.switch_samples(pid, c.t_req)[1])
                rec.lateness_samples = {
                    s: rec.k_applied[s] - rec.k_req[s] for s in rec.k_applied if s in rec.k_req
                }
                touched = True
            if touched:
                book.version += 1

    def update_statuses(self, frontier: Mapping[StreamId, int]) -> list[CommandRecord]:
        out: list[CommandRecord] = []
        if not self._scheduled:
            return out
        keep: list[PanelCommand] = []
        for c in self._scheduled:
            r = c.record
            if r.status != "scheduled":
                continue
            if all(int(frontier.get(s, 0)) > int(k.max()) for s, k in c.k_tiles.items()):
                late = any(v != 0 for v in r.lateness_samples.values()) or c.late
                r.status = "late_applied" if late else "applied"
                out.append(r)
            else:
                keep.append(c)
        self._scheduled = keep
        if out:
            for book in self.books.values():
                book.prune(frontier)
        return out
