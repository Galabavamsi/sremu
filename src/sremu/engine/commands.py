"""RIS command semantics and transition composition (DESIGN.md sections 4.6, 4.7, 5.6).

No wall clock here (I7). Normative rules implemented:

* **Switch time at the RIS.** A command carries ``t_sw``, the switching time AT THE RIS
  (exact). RX ``r`` sees it on the paths of render tile ``j`` at
  ``k_{r,j} = to_ticks(t_sw + tau_RIS->r,j(t_sw), f_r)`` (I3), with tau from the panel
  centre when the panel has one render tile and from tile ``j``'s reference point when it
  is tiled (``PropagationModel.ris_delay_s(panel, rx, t, tile)``). With
  ``ris_event_reference="rx"`` the switch is referenced at the receivers instead:
  ``k_r = to_ticks(t_sw, f_r)``. The record's ``k_req``/``k_applied`` use the panel centre.
  tau is evaluated with the geometry in force at ``t_sw``: when a timed geometry event at
  ``t_e`` is applied to the model, every pending command with ``t_sw >= t_e`` is re-mapped
  (:meth:`CommandManager.geometry_changed`).
* **Time forms.** ``{"t": ...}``; ``{"sample": {stream, index}}`` converted to
  ``t = from_ticks(index) - tau_RIS->stream`` (tau at the panel centre at
  ``from_ticks(index)``); ``"asap"``: ``t_req`` = the earliest physically consistent time at
  arrival, ``max_{r,j}(frontier_r/f_r - tau_{r,j})`` (never late).
* **Late commands.** ``frontier_r`` is the first uncomputed sample of RX r. A command is late
  if some RX has ``min_j k_{r,j} < frontier_r``. ``apply_on_arrival`` moves it to the
  earliest single physically consistent time ``t' = max_{r,j}(frontier_r/f_r - tau_{r,j})``
  and re-maps every RX with R3 (tau re-evaluated at ``t'``; ``t'`` is raised until every
  ``k >= frontier``). ``reject`` refuses it with ``LATE_COMMAND`` and applies nothing.
* **Ties.** Every command is kept (no single slot). Two config commands with the same
  ``t`` on the same panel: the later ``seq`` wins and the earlier gets ``superseded`` (it
  never takes effect); likewise for two ``ris.set_enabled`` commands. A config command and
  an enable command at the same instant both apply.
* **Element subsets** compose in ``(t, seq)`` order at the control level (``N_ctrl``), then
  go through granularity expansion and the failure map (section 5.6 steps 1-3).
* **Counterfactual** ``ris.set_enabled`` is a separate per-panel timeline (RIS absent /
  present) with the same per-RX, per-tile switch samples; its records carry
  ``counterfactual = True``. It multiplies the panel's paths and never changes the element
  states.
* **Transitions (4.7)** are composed per element in RIS time. The state of the panel is a
  :class:`Mixture`: per element a weight vector over the states, kept as a pure base plus
  layers ``row <- (1 - g)*row + g*onehot(target)``. Command ``i`` starts from the mixture at
  its own switch time (``start_i``); its moving elements (rows that are not already
  ``onehot(target_i)``) form transition classes (``single`` or ``by_edge`` with
  ``(last commanded, target)`` pairs) that follow the profile ``g_c`` (step, linear over
  ``T_c``, first order ``1 - exp(-t/T_c)``). A later command ``i+1`` starts from the
  instantaneous per-element Gamma ``start_i + g_c(t_{i+1} - t_i)*(onehot(target_i) -
  start_i)``, with ``g`` evaluated at the exact RIS-time offset, so the frozen state is the
  same for every receiver and tile [ENG; the per-tile renderer then starts the new
  transition at each tile's own switch sample]. The channel of a mixture follows by
  linearity of ``combine`` in Gamma (:func:`mixture_amplitudes`).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import (
    CommandReceipt,
    CommandRecord,
    CommandRequest,
    CommandStatus,
    EventReference,
    LinkId,
    PanelConfig,
    PanelId,
    PropagationModel,
    RisCapabilities,
    Snapshot,
    StreamId,
    TransitionModel,
    TransitionSpec,
    config_hash,
)
from sremu.errors import ConfigurationError, ErrorCode
from sremu.timebase import from_ticks, parse_exact, to_ticks

__all__ = [
    "CommandManager",
    "DelayFn",
    "EpochInfo",
    "Layer",
    "Mixture",
    "PanelBook",
    "PanelCommand",
    "PanelSpec",
    "RxPort",
    "RxView",
    "ScheduleOutcome",
    "mixture_amplitudes",
    "panel_specs_from_model",
    "profile_value",
    "transition_record",
]

DelayFn = Callable[[PanelId, StreamId, Fraction, int | None], float]
"""``ris_delay_s(panel, rx, t, tile)`` of the propagation model, s."""

_MAX_REMAP_ITER = 64


# =========================================================================== small types


@dataclass(frozen=True)
class RxPort:
    """An RX stream as the command logic sees it: id and exact rate (S/s)."""

    stream: StreamId
    rate: Fraction


def profile_value(profile: str, dt_s: Fraction, duration_s: Fraction | None) -> float:
    """Transition profile ``g(dt)`` at an exact RIS-time offset ``dt`` (s), 0 for dt < 0.

    ``step`` -> 1; ``linear`` -> ``min(dt/T, 1)``; ``first_order`` -> ``1 - exp(-dt/T)``.
    A missing or non-positive duration is a step.
    """
    if dt_s < 0:
        return 0.0
    if profile == "step" or duration_s is None or duration_s <= 0:
        return 1.0
    x = float(dt_s / duration_s)
    if profile == "linear":
        return 1.0 if x >= 1.0 else x
    if profile == "first_order":
        return float(-math.expm1(-x))
    raise ConfigurationError(f"unknown transition profile {profile!r}")


def transition_record(spec: TransitionSpec | None) -> dict[str, Any] | None:
    """``CommandRecord.transition`` form of a transition spec."""
    if spec is None:
        return None
    return {
        "model": spec.model,
        "duration_s": None
        if spec.duration_s is None
        else f"{spec.duration_s.numerator}/{spec.duration_s.denominator}",
        "classes": spec.classes,
    }


# =========================================================================== panel spec


@dataclass(eq=False)
class PanelSpec:
    """What the engine needs about one panel (section 5.6 steps 1-3 and capabilities).

    ``ctrl_of_cell`` (N,) maps physical cells to controls; ``initial_controls`` (N_ctrl,)
    int16 are the pre-failure control states; ``initial`` is the physical initial
    configuration (failures applied). ``failures`` maps a physical (N,) state vector to the
    one after stuck-at and dead handling. ``quantize`` maps control phases (rad) to states
    (None: phases unsupported). ``codebooks`` are named (K, N_ctrl) int16 arrays.
    ``transition`` is the panel default; its ``duration_s`` None means
    ``reconfiguration_time_s``.
    """

    panel: PanelId
    n_cells: int
    n_states: int
    ctrl_of_cell: NDArray[np.int64]
    initial_controls: NDArray[np.int16]
    initial: PanelConfig
    caps: RisCapabilities
    failures: Callable[[NDArray[np.int16]], NDArray[np.int16]]
    transition: TransitionSpec = field(default_factory=TransitionSpec)
    reconfiguration_time_s: Fraction | None = None
    control_names: tuple[str, ...] = ()
    quantize: Callable[[NDArray[np.float64]], NDArray[np.int16]] | None = None
    codebooks: dict[str, NDArray[np.int16]] = field(default_factory=dict)

    @property
    def n_controls(self) -> int:
        """N_ctrl."""
        return int(self.initial_controls.shape[0])

    def expand(self, controls: NDArray[np.int16]) -> NDArray[np.int16]:
        """Control states (N_ctrl,) -> physical (N,) states, failures applied."""
        phys = np.asarray(controls, dtype=np.int16)[self.ctrl_of_cell]
        return np.asarray(self.failures(phys), dtype=np.int16)

    def resolve_elements(self, elements: str | tuple[int, ...]) -> NDArray[np.int64] | None:
        """Control indices addressed by ``elements`` (None = all controls).

        ``"all"``; a tuple of control indices; or a group name: a control name of the
        panel's granularity (``e5``, ``col3``, ``tile2``, a ``groups`` name), or
        ``tile<k>``/``col<k>`` meaning the controls of the cells of physical tile/column k.
        """
        if isinstance(elements, str):
            if elements == "all":
                return None
            if elements in self.control_names:
                return np.array([self.control_names.index(elements)], dtype=np.int64)
            for prefix in ("tile", "col"):
                if elements.startswith(prefix) and elements[len(prefix) :].isdigit():
                    cells = self._cells_of(prefix, int(elements[len(prefix) :]))
                    if cells is not None:
                        return np.unique(self.ctrl_of_cell[cells])
            raise ConfigurationError(f"panel {self.panel}: unknown element group {elements!r}")
        idx = np.asarray(elements, dtype=np.int64)
        if idx.ndim != 1 or (idx.size and (idx.min() < 0 or idx.max() >= self.n_controls)):
            raise ConfigurationError(
                f"panel {self.panel}: element indices outside [0, {self.n_controls})"
            )
        if np.unique(idx).size != idx.size:
            raise ConfigurationError(f"panel {self.panel}: repeated element indices")
        return idx

    _layout: tuple[int, int, int, int] | None = None  # n_u, n_v, n_tiles_u, n_tiles_v

    def _cells_of(self, kind: str, k: int) -> NDArray[np.int64] | None:
        if self._layout is None:
            return None
        n_u, n_v, tu, tv = self._layout
        cells = np.arange(n_u * n_v)
        iu, iv = cells % n_u, cells // n_u
        if kind == "col":
            sel = iu == k
        else:
            tile = (iv // (n_v // tv)) * tu + iu // (n_u // tu)
            sel = tile == k
        return cells[sel] if sel.any() else None


def _caps_fallback(
    pid: PanelId, n_cells: int, n_states: int, layout: tuple[int, int]
) -> RisCapabilities:
    return RisCapabilities(
        n_elements=n_cells,
        layout=layout,
        control_granularity="element",
        states=tuple(f"s{i}" for i in range(n_states)),
        gamma_source=f"analytic:{pid}",
        operating_band_hz=(0.0, math.inf),
        max_bandwidth_hz=math.inf,
    )


def panel_specs_from_model(model: PropagationModel) -> dict[PanelId, PanelSpec]:
    """Panel specs of a propagation model, in snapshot (scenario) order.

    The physics model (``ScenePropagation``) provides the non-ideality pipeline, initial
    configuration, capabilities and transition default; any other model gets element
    granularity, no failures, all-zero initial states and step transitions.
    """
    snap = model.snapshot(Fraction(0))
    out: dict[PanelId, PanelSpec] = {}
    for pid, rc in snap.ris.items():
        geo = model.panel_geometry(pid)
        layout = (geo.n_u, geo.n_v, geo.n_tiles_u, geo.n_tiles_v)
        has_pipeline = all(
            hasattr(model, a) for a in ("pipeline", "initial_config", "capabilities", "panel")
        )
        if has_pipeline:
            from sremu.physics.nonideal import control_map, quantize_phases

            pipe = model.pipeline(pid)  # type: ignore[attr-defined]
            pm = model.panel(pid)  # type: ignore[attr-defined]
            cfg0: PanelConfig = model.initial_config(pid)  # type: ignore[attr-defined]
            caps: RisCapabilities = model.capabilities(pid)  # type: ignore[attr-defined]
            ctrl = np.asarray(pipe.ctrl_of_cell, dtype=np.int64)
            pre = (
                np.zeros(rc.n_cells, np.int16)
                if pm.initial_state is None
                else np.asarray(pm.initial_state, np.int16)
            )
            first = np.zeros(int(ctrl.max()) + 1, dtype=np.int64)
            for c in range(first.shape[0]):
                first[c] = int(np.flatnonzero(ctrl == c)[0])
            init_ctrl = pre[first].astype(np.int16)
            _, names = control_map(
                pm.granularity,
                geo.n_u,
                geo.n_v,
                tiles=(geo.n_tiles_u, geo.n_tiles_v),
                groups=pm.groups,
            )
            tr = dict(pm.transition or {"model": "step"})
            gref = np.asarray(pipe.gamma_ref)

            def _quant(ph: NDArray[np.float64], g: NDArray[Any] = gref) -> NDArray[np.int16]:
                return np.asarray(quantize_phases(np.asarray(ph, np.float64), g), np.int16)

            spec = PanelSpec(
                panel=pid,
                n_cells=rc.n_cells,
                n_states=rc.n_states,
                ctrl_of_cell=ctrl,
                initial_controls=init_ctrl,
                initial=cfg0,
                caps=caps,
                failures=pipe.failures.apply,
                transition=TransitionSpec(
                    tr.get("model", "step"),
                    None if tr.get("duration_s") is None else parse_exact(tr["duration_s"]),
                    tr.get("classes", "single"),
                ),
                reconfiguration_time_s=caps.reconfiguration_time_s,
                control_names=tuple(names),
                quantize=_quant,
                codebooks=pipe.codebooks,
            )
        else:
            n = rc.n_cells
            spec = PanelSpec(
                panel=pid,
                n_cells=n,
                n_states=rc.n_states,
                ctrl_of_cell=np.arange(n, dtype=np.int64),
                initial_controls=np.zeros(n, np.int16),
                initial=PanelConfig(pid, np.zeros(n, np.int16), True),
                caps=_caps_fallback(pid, n, rc.n_states, (geo.n_u, geo.n_v)),
                failures=lambda s: np.asarray(s, np.int16).copy(),
                control_names=tuple(f"e{k}" for k in range(n)),
            )
        spec._layout = layout
        out[pid] = spec
    return out


# =========================================================================== mixtures


@dataclass(frozen=True, eq=False)
class Layer:
    """``row <- (1 - g)*row + g*onehot(target)`` on the masked rows (0 < g < 1)."""

    mask: NDArray[np.bool_]
    target: NDArray[np.int16]
    g: float


@dataclass(frozen=True, eq=False)
class Mixture:
    """Per-element state weights: a pure base (int16, -1 = dead) plus fractional layers."""

    base: NDArray[np.int16]
    layers: tuple[Layer, ...] = ()

    @property
    def pure(self) -> bool:
        """True when every element is in exactly one state."""
        return not self.layers

    def touched(self) -> NDArray[np.bool_]:
        """Rows touched by a layer."""
        u = np.zeros(self.base.shape[0], dtype=bool)
        for la in self.layers:
            u |= la.mask
        return u

    def dense(self, rows: NDArray[np.int64], n_states: int) -> NDArray[np.float64]:
        """Weight rows (len(rows), S) of the given elements."""
        w = np.zeros((rows.shape[0], n_states), dtype=np.float64)
        b = self.base[rows].astype(np.int64)
        ok = b >= 0
        w[np.flatnonzero(ok), b[ok]] = 1.0
        for la in self.layers:
            sel = la.mask[rows]
            if not sel.any():
                continue
            i = np.flatnonzero(sel)
            w[i] *= 1.0 - la.g
            w[i, la.target[rows[i]].astype(np.int64)] += la.g
        return w

    def moving(self, target: NDArray[np.int16], n_states: int) -> NDArray[np.bool_]:
        """Elements whose weights are not ``onehot(target)`` (dead cells never move)."""
        mv = (self.base != target) & (target >= 0)
        if self.layers:
            rows = np.flatnonzero(self.touched() & (target >= 0))
            if rows.size:
                w = self.dense(rows, n_states)
                t = target[rows].astype(np.int64)
                one = np.zeros_like(w)
                one[np.arange(rows.size), t] = 1.0
                mv[rows] = np.any(w != one, axis=1)
        return mv

    def apply(
        self, masks: Sequence[NDArray[np.bool_]], target: NDArray[np.int16], gs: Sequence[float]
    ) -> Mixture:
        """Move class ``c`` (mask) toward ``target`` by ``g_c`` (1: fully, 0: not at all)."""
        base = self.base.copy()
        layers = list(self.layers)
        for mask, g in zip(masks, gs, strict=True):
            if g <= 0.0 or not mask.any():
                continue
            if g >= 1.0:
                base[mask] = target[mask]
                kept = []
                for la in layers:
                    m2 = la.mask & ~mask
                    if m2.any():
                        kept.append(Layer(m2, la.target, la.g))
                layers = kept
            else:
                layers.append(Layer(mask.copy(), target.copy(), float(g)))
        return Mixture(base, tuple(layers))


CombineFn = Callable[[NDArray[np.int16]], Mapping[LinkId, NDArray[np.complex128]]]
"""Per-link amplitudes (all paths) with one panel at the given pure physical states."""


def mixture_amplitudes(
    mix: Mixture, n_states: int, combine: CombineFn
) -> dict[LinkId, NDArray[np.complex128]]:
    """Channel amplitudes of a mixture, by linearity of ``combine`` in Gamma.

    With the reference ``R`` (per element the state of largest weight, ties to the lowest
    index) and groups ``G`` of touched elements with identical weight rows,
    ``h(W) = h(R) + sum_G sum_{s != R_G} w_{G,s} (h(R with G <- s) - h(R))``. A pure
    mixture returns ``combine(base)`` exactly.
    """
    if mix.pure:
        return {lk: np.asarray(a, np.complex128).copy() for lk, a in combine(mix.base).items()}
    rows = np.flatnonzero(mix.touched() & (mix.base >= 0))
    w = mix.dense(rows, n_states)
    ref = mix.base.copy()
    ref[rows] = np.argmax(w, axis=1).astype(np.int16)  # ties -> lowest state
    h_ref = {lk: np.asarray(a, np.complex128) for lk, a in combine(ref).items()}
    out = {lk: a.copy() for lk, a in h_ref.items()}
    uniq, inv = np.unique(w, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    for gi in range(uniq.shape[0]):
        grp = rows[inv == gi]
        row = uniq[gi]
        r_state = int(np.argmax(row))
        for s in range(n_states):
            wt = float(row[s])
            if s == r_state or wt == 0.0:
                continue
            cfg = ref.copy()
            cfg[grp] = s
            for lk, a in combine(cfg).items():
                out[lk] = out[lk] + wt * (np.asarray(a, np.complex128) - h_ref[lk])
    return out


# =========================================================================== commands


@dataclass(eq=False)
class PanelCommand:
    """One accepted panel command (config or enable) with its derived timeline data."""

    seq: int
    cmd_id: str
    src: str
    op: str
    t_req: Fraction
    t: Fraction
    kind: Literal["config", "enabled"]
    record: CommandRecord
    update_idx: NDArray[np.int64] | None = None
    update_vals: NDArray[np.int16] | None = None
    physical: NDArray[np.int16] | None = None
    enabled: bool | None = None
    transition: TransitionSpec = field(default_factory=TransitionSpec)
    duration_s: Fraction | None = None
    late: bool = False
    superseded: bool = False
    # derived by PanelBook._recompose (config commands)
    version: int = 0
    controls: NDArray[np.int16] | None = None
    target: NDArray[np.int16] | None = None
    start: Mixture | None = None
    last: NDArray[np.int16] | None = None
    classes: list[NDArray[np.bool_]] = field(default_factory=list)
    # per RX: (K,) tile switch samples and the panel-centre sample
    k_tiles: dict[StreamId, NDArray[np.int64]] = field(default_factory=dict)
    k_centre: dict[StreamId, int] = field(default_factory=dict)

    @property
    def profile(self) -> TransitionModel:
        """Transition profile of this command."""
        return self.transition.model

    @property
    def is_step(self) -> bool:
        """True for a step (or a zero-length) transition."""
        return self.transition.model == "step" or not self.duration_s

    def g_at(self, dt_s: Fraction) -> float:
        """Profile value at RIS-time offset ``dt_s`` (s) after the switch."""
        return profile_value(self.transition.model, dt_s, self.duration_s)

    def after(self, n_states: int) -> Mixture:
        """Mixture once this command's transition has completed."""
        assert self.start is not None
        assert self.target is not None
        return self.start.apply(self.classes, self.target, [1.0] * len(self.classes))

    def frozen(self, dt_s: Fraction) -> Mixture:
        """Mixture ``dt_s`` (s, RIS time) after this command's switch."""
        assert self.start is not None
        assert self.target is not None
        g = self.g_at(dt_s)
        return self.start.apply(self.classes, self.target, [g] * len(self.classes))


@dataclass(frozen=True, eq=False)
class EpochInfo:
    """What the renderer needs about one panel epoch (the latest command seen by a tile).

    ``key``: hashable identity ``(seq, version)`` (``(-1, 0)`` for the initial epoch).
    ``start``: mixture at the switch. ``classes``: moving masks. ``target``: physical target.
    ``profile``/``duration_s``: transition (step epochs render ``after`` directly).
    """

    key: tuple[int, int]
    start: Mixture
    classes: tuple[NDArray[np.bool_], ...]
    target: NDArray[np.int16]
    profile: TransitionModel
    duration_s: Fraction | None
    is_step: bool

    def after(self) -> Mixture:
        """Mixture once the transition has completed."""
        return self.start.apply(self.classes, self.target, [1.0] * len(self.classes))


@dataclass(frozen=True, eq=False)
class RxView:
    """Per-RX switch-sample arrays of one panel (valid for one book version).

    ``cfg_k`` (n_cfg, K) int64 and ``cfg_epochs`` (n_cfg + 1 epochs, index 0 = before the
    first listed command); ``en_k`` (n_en, K), ``en_kc`` (n_en,), ``en_vals`` (n_en,) bool;
    ``splits``: every switch sample of this panel at this RX, sorted and unique.
    """

    cfg_k: NDArray[np.int64]
    cfg_epochs: tuple[EpochInfo, ...]
    en_k: NDArray[np.int64]
    en_kc: NDArray[np.int64]
    en_vals: NDArray[np.bool_]
    en_initial: bool
    splits: NDArray[np.int64]

    def epochs_at(self, a: int) -> NDArray[np.int64]:
        """(K,) epoch index of each tile at RX sample ``a``."""
        if self.cfg_k.shape[0] == 0:
            return np.zeros(self.cfg_k.shape[1], dtype=np.int64)
        return np.count_nonzero(self.cfg_k <= a, axis=0).astype(np.int64)

    def enabled_at(self, a: int) -> tuple[NDArray[np.bool_], bool]:
        """(K,) enabled flag per tile and the panel-centre flag at RX sample ``a``."""
        k = self.cfg_k.shape[1]
        if self.en_k.shape[0] == 0:
            return np.full(k, self.en_initial), self.en_initial
        cnt = np.count_nonzero(self.en_k <= a, axis=0)
        vals = np.concatenate([[self.en_initial], self.en_vals])
        cc = int(np.count_nonzero(self.en_kc <= a))
        return vals[cnt].astype(bool), bool(vals[cc])


class PanelBook:
    """Command timeline of one panel (see the module docstring)."""

    def __init__(self, spec: PanelSpec, n_tiles: int) -> None:
        self.spec = spec
        self.n_tiles = int(n_tiles)
        self.config: list[PanelCommand] = []
        self.enabled: list[PanelCommand] = []
        self.version = 0
        self._views: dict[StreamId, tuple[int, RxView]] = {}
        init = spec.initial.state.astype(np.int16)
        self._initial_epoch = EpochInfo(
            (-1, 0), Mixture(init.copy()), (), init.copy(), "step", None, True
        )
        # the last pruned config command stands in for everything before the list
        self._head: PanelCommand | None = None
        self._head_enabled: bool = bool(spec.initial.enabled)
        self._ver_counter = 0

    # ------------------------------------------------------------------ timeline
    def active_config(self) -> list[PanelCommand]:
        """Non-superseded config commands in ``(t, seq)`` order."""
        return [c for c in self.config if not c.superseded]

    def active_enabled(self) -> list[PanelCommand]:
        """Non-superseded enable commands in ``(t, seq)`` order."""
        return [c for c in self.enabled if not c.superseded]

    def insert(self, cmd: PanelCommand) -> list[PanelCommand]:
        """Insert an accepted command; returns the commands it supersedes."""
        lst = self.config if cmd.kind == "config" else self.enabled
        sup = [c for c in lst if c.t == cmd.t and not c.superseded and c.seq < cmd.seq]
        for c in sup:
            c.superseded = True
            c.record.status = "superseded"
        lst.append(cmd)
        lst.sort(key=lambda c: (c.t, c.seq))
        self._recompose()
        return sup

    def _recompose(self) -> None:
        """Recompute targets, start mixtures and classes of the config timeline."""
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

    # ------------------------------------------------------------------ queries
    def epochs(self) -> tuple[EpochInfo, ...]:
        """Epoch 0 (initial) followed by one epoch per active config command."""
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
        """Per-RX switch-sample arrays (cached per book version)."""
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
        """Configuration in effect at RIS time ``t``: the target of the latest config command
        with ``t_i <= t`` (dominant states) and the enable flag at ``t``."""
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

    def config_after(self, cmd: PanelCommand) -> PanelConfig:
        """Configuration right after ``cmd`` (its target and the enable flag at its time)."""
        if cmd.kind == "config":
            assert cmd.target is not None
            st = cmd.target
            en = bool(self.config_at(cmd.t).enabled)
        else:
            st = self.config_at(cmd.t).state
            en = bool(cmd.enabled)
        return PanelConfig(self.spec.panel, np.asarray(st, np.int16).copy(), en)

    def pending_after(self, t: Fraction) -> list[PanelCommand]:
        """Active commands with ``t_i > t``."""
        return [c for c in self.active_config() + self.active_enabled() if c.t > t]

    def prune(self, frontier: Mapping[StreamId, int], keep: int = 256) -> None:
        """Drop config commands that no RX can use again (keeps memory bounded).

        Command i is dropped when command i+1 is entirely before every RX frontier (every
        tile of every RX has switched to i+1 or later), so no sample >= the frontier is
        rendered with epoch i. Only runs when more than ``keep`` commands are listed.
        """
        if len(self.config) <= keep:
            return
        act = self.active_config()
        if len(act) <= keep:
            return
        cut = 0  # commands act[:cut] are never rendered again
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
        # superseded commands at or before the head's time are final and never rendered
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


def _same_mixture(a: Mixture, b: Mixture) -> bool:
    if not np.array_equal(a.base, b.base) or len(a.layers) != len(b.layers):
        return False
    return all(
        np.array_equal(x.mask, y.mask) and np.array_equal(x.target, y.target) and x.g == y.g
        for x, y in zip(a.layers, b.layers, strict=True)
    )


def _classes(
    start: Mixture,
    last: NDArray[np.int16],
    target: NDArray[np.int16],
    n_states: int,
    classes: str,
) -> list[NDArray[np.bool_]]:
    mv = start.moving(target, n_states)
    if not mv.any():
        return []
    if classes == "single":
        return [mv]
    if classes != "by_edge":
        raise ConfigurationError(f"unknown transition classes {classes!r}")
    pairs = sorted({(int(x), int(y)) for x, y in zip(last[mv], target[mv], strict=True)})
    return [mv & (last == x) & (target == y) for x, y in pairs]


# =========================================================================== manager


@dataclass
class ScheduleOutcome:
    """Result of :meth:`CommandManager.schedule`: the receipt, the command (None when
    rejected), whether it was late, and records that became final."""

    receipt: CommandReceipt
    command: PanelCommand | None
    late: bool
    final: list[CommandRecord]


class CommandManager:
    """Panel books, sequence numbers, switch-sample mapping and command records.

    ``rx_ports``: every RX stream (id, rate). ``delay_fn``: ``ris_delay_s``. ``n_tiles``:
    render tiles K per panel. ``static_delays``: tau does not depend on t (static scene),
    so it is cached per (panel, rx, tile).
    """

    def __init__(
        self,
        specs: Mapping[PanelId, PanelSpec],
        rx_ports: Sequence[RxPort],
        delay_fn: DelayFn,
        n_tiles: Mapping[PanelId, int],
        *,
        reference: EventReference = "ris",
        static_delays: bool = False,
    ) -> None:
        if reference not in ("ris", "rx"):
            raise ConfigurationError(
                f"ris_event_reference must be 'ris' or 'rx', got {reference!r}"
            )
        self.specs = dict(specs)
        self.rx_ports = tuple(rx_ports)
        self.rates = {p.stream: p.rate for p in self.rx_ports}
        self.delay_fn = delay_fn
        self.reference: EventReference = reference
        self.static_delays = bool(static_delays)
        self.books = {
            pid: PanelBook(sp, int(n_tiles.get(pid, 1))) for pid, sp in self.specs.items()
        }
        self.records: list[CommandRecord] = []
        self._pending_final: list[CommandRecord] = []
        self._seq = 0
        self._scheduled: list[PanelCommand] = []
        self._tau_cache: dict[tuple[PanelId, StreamId, int | None, Fraction], float] = {}
        # times of the geometry events applied to the model (static scenes: tau is constant
        # between two of them, so the cache is keyed by the latest event at or before t)
        self._geometry_events: list[Fraction] = []
        self.counts = {"late_commands": 0, "rejected_commands": 0}

    # ------------------------------------------------------------------ basics
    def next_seq(self) -> int:
        """Next engine-wide sequence number (monotone)."""
        self._seq += 1
        return self._seq

    @property
    def seq(self) -> int:
        """Last assigned sequence number."""
        return self._seq

    @property
    def has_pending(self) -> bool:
        """Whether some accepted command is not yet applied on every RX."""
        return bool(self._scheduled)

    def caps(self) -> dict[PanelId, RisCapabilities]:
        """Capabilities of every panel."""
        return {pid: sp.caps for pid, sp in self.specs.items()}

    def tau(self, panel: PanelId, rx: StreamId, t: Fraction, tile: int | None) -> float:
        """tau_RIS->rx in s (panel centre for ``tile`` None) at RIS time ``t``.

        Static scenes cache it per geometry epoch: nodes only move through timed
        ``node.set_position`` events, so tau is constant from one applied event to the next
        and is evaluated at the latest event time ``<= t`` (0 before the first)."""
        if self.static_delays:
            i = bisect.bisect_right(self._geometry_events, t)
            t_ref = self._geometry_events[i - 1] if i else Fraction(0)
            key = (panel, rx, tile, t_ref)
            hit = self._tau_cache.get(key)
            if hit is None:
                hit = float(self.delay_fn(panel, rx, t_ref, tile))
                self._tau_cache[key] = hit
            return hit
        return float(self.delay_fn(panel, rx, t, tile))

    def geometry_changed(self, t: Fraction) -> None:
        """A timed geometry event at ``t`` was applied to the propagation model, so
        ``tau_RIS->r`` may change from ``t`` on (I3 evaluates it at the switch time with the
        geometry in force then). Re-map every command switching at or after ``t`` that is
        not final yet; its record's ``k_applied`` (and ``k_req`` when ``t_req >= t``) and
        lateness follow. Such a command has not been rendered on any RX: the event is
        applied before any sample at or after ``t`` is computed."""
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
                book.version += 1  # invalidates the per-RX views and the renderer's caches

    def switch_samples(
        self, panel: PanelId, t: Fraction
    ) -> tuple[dict[StreamId, NDArray[np.int64]], dict[StreamId, int]]:
        """Per RX: (K,) tile switch samples and the panel-centre sample of a switch at ``t``."""
        book = self.books[panel]
        k_count = book.n_tiles
        tiles: dict[StreamId, NDArray[np.int64]] = {}
        centre: dict[StreamId, int] = {}
        for p in self.rx_ports:
            if self.reference == "rx":
                k = to_ticks(t, p.rate)
                tiles[p.stream] = np.full(k_count, k, dtype=np.int64)
                centre[p.stream] = k
                continue
            kc = to_ticks(t + Fraction(self.tau(panel, p.stream, t, None)), p.rate)
            if k_count == 1:
                tiles[p.stream] = np.array([kc], dtype=np.int64)
            else:
                tiles[p.stream] = np.array(
                    [
                        to_ticks(t + Fraction(self.tau(panel, p.stream, t, j)), p.rate)
                        for j in range(k_count)
                    ],
                    dtype=np.int64,
                )
            centre[p.stream] = kc
        return tiles, centre

    def earliest_time(
        self, panel: PanelId, frontier: Mapping[StreamId, int], t_ref: Fraction
    ) -> Fraction:
        """Earliest single physically consistent switch time not late on any RX:
        ``max_{r,j}(frontier_r/f_r - tau_{r,j})`` (tau at ``t_ref``), then raised until
        every switch sample is ``>= frontier_r`` (tau re-evaluated at the new time)."""
        k_count = self.books[panel].n_tiles
        tiles = [None] if k_count == 1 else list(range(k_count))
        cands = [Fraction(0)]
        for p in self.rx_ports:
            f0 = from_ticks(int(frontier.get(p.stream, 0)), p.rate)
            for j in tiles:
                tau = 0.0 if self.reference == "rx" else self.tau(panel, p.stream, t_ref, j)
                cands.append(f0 - Fraction(tau))
        t = max(cands)
        for _ in range(_MAX_REMAP_ITER):
            ks, _kc = self.switch_samples(panel, t)
            deficit = Fraction(0)
            for p in self.rx_ports:
                fr = int(frontier.get(p.stream, 0))
                kmin = int(ks[p.stream].min())
                if kmin < fr:
                    deficit = max(deficit, from_ticks(fr - kmin, p.rate))
            if deficit == 0:
                return t
            t += deficit
        raise ConfigurationError(
            f"panel {panel}: could not find a non-late switch time"
        )  # pragma: no cover

    def is_late(
        self, ks: Mapping[StreamId, NDArray[np.int64]], frontier: Mapping[StreamId, int]
    ) -> bool:
        """Whether some RX has a tile switch sample before its frontier."""
        return any(int(ks[r].min()) < int(frontier.get(r, 0)) for r in ks)

    # ------------------------------------------------------------------ schedule
    def _record(
        self,
        req_id: str,
        seq: int,
        src: str,
        op: str,
        target: str,
        t_req: Fraction,
        t_app: Fraction | None,
        k_req: dict[StreamId, int],
        k_app: dict[StreamId, int],
        status: CommandStatus,
        chash: str,
        transition: TransitionSpec | None,
        counterfactual: bool,
        t_wall_recv_ns: int | None,
    ) -> CommandRecord:
        return CommandRecord(
            cmd_id=req_id,
            seq=seq,
            src=src,
            op=op,
            panel=target,
            t_req=t_req,
            t_applied=t_app,
            k_req=dict(k_req),
            k_applied=dict(k_app),
            lateness_samples={s: k_app[s] - k_req[s] for s in k_app if s in k_req},
            t_wall_recv_ns=t_wall_recv_ns,
            t_wall_computed_ns=None,
            t_wall_visible_ns=None,
            status=status,
            config_hash=chash,
            transition=transition_record(transition),
            counterfactual=counterfactual,
        )

    def reject(
        self,
        req: CommandRequest,
        code: ErrorCode,
        msg: str,
        t_req: Fraction | None,
        k_req: dict[StreamId, int] | None = None,
        chash: str = "",
        frontier_time: Fraction = Fraction(0),
    ) -> ScheduleOutcome:
        """Count, log (a ``rejected`` record with a seq, I6) and refuse a command."""
        self.counts["rejected_commands"] += 1
        seq = self.next_seq()
        rec = self._record(
            req.cmd_id,
            seq,
            req.src,
            req.op,
            req.target,
            frontier_time if t_req is None else t_req,
            None,
            k_req or {},
            {},
            "rejected",
            chash,
            req.transition,
            req.counterfactual,
            req.t_wall_recv_ns,
        )
        self.records.append(rec)
        receipt = CommandReceipt(
            req.cmd_id, False, "rejected", error=code, msg=msg, t_wall_recv_ns=req.t_wall_recv_ns
        )
        return ScheduleOutcome(receipt, None, False, [rec])

    def requested_time(
        self, req: CommandRequest, frontier: Mapping[StreamId, int]
    ) -> Fraction | None:
        """``t_req`` of a ris.* request (None: unknown ``sample`` stream)."""
        at = req.at
        if at.kind == "t":
            assert at.t is not None
            return at.t
        if at.kind == "sample":
            assert at.stream is not None
            assert at.index is not None
            if at.stream not in self.rates:
                return None
            t0 = from_ticks(at.index, self.rates[at.stream])
            if self.reference == "rx":
                return t0
            return t0 - Fraction(self.tau(req.target, at.stream, t0, None))
        tf = max(
            (from_ticks(int(frontier.get(p.stream, 0)), p.rate) for p in self.rx_ports),
            default=Fraction(0),
        )
        return self.earliest_time(req.target, frontier, tf)

    def _payload(
        self, req: CommandRequest, spec: PanelSpec
    ) -> tuple[NDArray[np.int64] | None, NDArray[np.int16]]:
        """Control indices and int16 control states of a ris.set_config request."""
        idx = spec.resolve_elements(req.elements)
        n = spec.n_controls if idx is None else int(idx.shape[0])
        if req.states is not None:
            vals = np.asarray(req.states, dtype=np.int16)
        elif req.phases_rad is not None:
            if spec.quantize is None:
                raise ConfigurationError(
                    f"panel {spec.panel}: phases are not supported (no quantizer)"
                )
            vals = np.asarray(spec.quantize(np.asarray(req.phases_rad, np.float64)), np.int16)
        else:
            assert req.codebook_index is not None
            book = spec.codebooks.get("default")
            if book is None:
                raise ConfigurationError(f"panel {spec.panel}: no codebook registered")
            if not 0 <= int(req.codebook_index) < book.shape[0]:
                raise ConfigurationError(
                    f"codebook index {req.codebook_index} outside [0, {book.shape[0]})"
                )
            row = np.asarray(book[int(req.codebook_index)], np.int16)
            vals = row if idx is None else row[idx]
        if vals.shape != (n,):
            raise ConfigurationError(
                f"panel {spec.panel}: {vals.shape[0] if vals.ndim == 1 else vals.shape} values "
                f"for {n} controls"
            )
        if vals.size and (int(vals.min()) < 0 or int(vals.max()) >= spec.n_states):
            raise ConfigurationError(
                f"panel {spec.panel}: state values outside [0, {spec.n_states})"
            )
        return idx, vals

    def schedule(
        self,
        req: CommandRequest,
        frontier: Mapping[StreamId, int],
        *,
        late_policy: str | None = None,
        lookahead_bound: Fraction | None = None,
        physical: PanelConfig | None = None,
    ) -> ScheduleOutcome:
        """Schedule a ``ris.set_config`` or ``ris.set_enabled`` request (section 4.6).

        ``frontier``: first uncomputed sample per RX. ``lookahead_bound``: ``t_b + L_min``
        of the latest barrier when the source is barrier-registered (earlier timed
        commands are rejected with ``LOOKAHEAD_VIOLATION``). ``physical``: replay form, the
        applied physical configuration given directly (bypasses the pipeline).
        """
        policy = late_policy or req.late_policy
        f_time = max(
            (from_ticks(int(frontier.get(p.stream, 0)), p.rate) for p in self.rx_ports),
            default=Fraction(0),
        )
        if req.op not in ("ris.set_config", "ris.set_enabled"):
            return self.reject(
                req,
                ErrorCode.NOT_PERMITTED,
                f"{req.op} is not a panel command",
                None,
                frontier_time=f_time,
            )
        spec = self.specs.get(req.target)
        if spec is None:
            return self.reject(
                req,
                ErrorCode.UNKNOWN_TARGET,
                f"unknown panel {req.target!r}",
                None,
                frontier_time=f_time,
            )
        if req.at.kind == "sample" and req.at.stream not in self.rates:
            return self.reject(
                req,
                ErrorCode.UNKNOWN_TARGET,
                f"unknown RX stream {req.at.stream!r}",
                None,
                frontier_time=f_time,
            )
        t_req = self.requested_time(req, frontier)
        assert t_req is not None
        ks_req, kc_req = self.switch_samples(req.target, t_req)
        if lookahead_bound is not None and req.at.kind != "asap" and t_req < lookahead_bound:
            return self.reject(
                req,
                ErrorCode.LOOKAHEAD_VIOLATION,
                f"{req.src}: at = {t_req} < t_b + L_min = {lookahead_bound}",
                t_req,
                kc_req,
            )
        kind: Literal["config", "enabled"] = "enabled" if req.op == "ris.set_enabled" else "config"
        idx: NDArray[np.int64] | None = None
        vals: NDArray[np.int16] | None = None
        phys: NDArray[np.int16] | None = None
        trans = req.transition or spec.transition
        dur = trans.duration_s if trans.duration_s is not None else spec.reconfiguration_time_s
        if kind == "config":
            try:
                if physical is not None:
                    phys = np.asarray(physical.state, np.int16)
                    if phys.shape != (spec.n_cells,):
                        raise ConfigurationError(f"physical state must have {spec.n_cells} cells")
                else:
                    idx, vals = self._payload(req, spec)
                if trans.model != "step" and not dur:
                    raise ConfigurationError(
                        f"panel {spec.panel}: a {trans.model} transition needs a duration "
                        "(no duration_s and no reconfiguration_time_s)"
                    )
            except ConfigurationError as exc:
                return self.reject(req, ErrorCode.SCHEMA_ERROR, str(exc), t_req, kc_req)
        late = self.is_late(ks_req, frontier) and req.at.kind != "asap"
        t_app = t_req
        if late:
            self.counts["late_commands"] += 1
            if policy == "reject":
                book = self.books[req.target]
                want = book.config_at(t_req)
                if kind == "config" and vals is not None:
                    act = [c for c in book.active_config() if c.t <= t_req]
                    last_ctrl = act[-1].controls if act else None
                    ctrl = spec.initial_controls.copy() if last_ctrl is None else last_ctrl.copy()
                    if idx is None:
                        ctrl = vals.copy()
                    else:
                        ctrl[idx] = vals
                    chash = config_hash(spec.expand(ctrl), want.enabled)
                elif kind == "config" and phys is not None:
                    chash = config_hash(phys, want.enabled)
                else:
                    chash = config_hash(want.state, bool(req.enabled))
                out_rej = self.reject(
                    req, ErrorCode.LATE_COMMAND, "command is late", t_req, kc_req, chash
                )
                out_rej.late = True
                return out_rej
            t_app = self.earliest_time(req.target, frontier, t_req)
        ks, kc = self.switch_samples(req.target, t_app)
        seq = self.next_seq()
        rec = self._record(
            req.cmd_id,
            seq,
            req.src,
            req.op,
            req.target,
            t_req,
            t_app,
            kc_req,
            kc,
            "scheduled",
            "",
            req.transition if kind == "config" else None,
            req.counterfactual,
            req.t_wall_recv_ns,
        )
        cmd = PanelCommand(
            seq=seq,
            cmd_id=req.cmd_id,
            src=req.src,
            op=req.op,
            t_req=t_req,
            t=t_app,
            kind=kind,
            record=rec,
            update_idx=idx,
            update_vals=vals,
            physical=phys,
            enabled=None if req.enabled is None else bool(req.enabled),
            transition=trans if kind == "config" else TransitionSpec(),
            duration_s=dur if kind == "config" else None,
            late=late,
            k_tiles=ks,
            k_centre=kc,
        )
        book = self.books[req.target]
        sup = book.insert(cmd)
        self._scheduled.append(cmd)
        rec.config_hash = book.config_after(cmd).config_hash()
        self.records.append(rec)
        final = [c.record for c in sup]
        receipt = CommandReceipt(
            req.cmd_id,
            True,
            "late_applied" if late else "scheduled",
            seq=seq,
            t_req=t_req,
            k_req=kc_req,
            t_wall_recv_ns=req.t_wall_recv_ns,
        )
        return ScheduleOutcome(receipt, cmd, late, final)

    # ------------------------------------------------------------------ statuses
    def update_statuses(self, frontier: Mapping[StreamId, int]) -> list[CommandRecord]:
        """Mark scheduled commands applied once every RX frontier passed every tile switch
        sample; returns the records that became final."""
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

    def pending_receipts(self, t: Fraction) -> tuple[CommandReceipt, ...]:
        """Receipts of active commands scheduled after RIS time ``t`` (observations)."""
        out = []
        for book in self.books.values():
            for c in sorted(book.pending_after(t), key=lambda c: (c.t, c.seq)):
                out.append(
                    CommandReceipt(
                        c.cmd_id,
                        True,
                        "scheduled",
                        seq=c.seq,
                        t_req=c.t_req,
                        k_req=dict(c.k_centre),
                    )
                )
        return tuple(out)

    def snapshot_states(self, t: Fraction) -> dict[PanelId, NDArray[np.int16]]:
        """Per panel (N,) int16 state in effect at RIS time ``t``."""
        return {pid: b.config_at(t).state.copy() for pid, b in self.books.items()}


# =========================================================================== epoch amplitudes


def panel_combine_fn(model: PropagationModel, snap: Snapshot, panel: PanelId) -> CombineFn:
    """``combine`` with only ``panel`` configured (enabled) at the given physical states."""

    def fn(states: NDArray[np.int16]) -> Mapping[LinkId, NDArray[np.complex128]]:
        cfg = PanelConfig(panel, np.asarray(states, np.int16), True)
        return {lk: lp.amp for lk, lp in model.combine(snap, {panel: cfg}).items()}

    return fn
