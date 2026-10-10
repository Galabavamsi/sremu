"""Controller helpers (DESIGN.md sections 4.9, 6.1, 8.7; WP9).

What every in-process controller needs, independent of any particular strategy:

* :class:`BaseController`: the attributes of the :class:`sremu.contracts.Controller`
  protocol (``name``, ``period_s``, ``min_latency_s``, ``spec``, ``latency``) with exact-time
  validation, and a ``setup`` that keeps the panel capabilities and the keyed generator.
* Keyed randomness (section 6.1, D14): :func:`controller_rng` is the generator of purpose
  ``ctrl/<name>``, ``rng.generator(seed, "ctrl", name)``; :func:`setup_controller` hands it
  to a controller. Controllers never build generators of their own, so their decisions are
  a pure function of (seed, name, observations).
* Channel views for position-driven controllers. A controller sees positions, not the
  channel; it evaluates the channel on **its own model of the scene** (section 8.7):

  - :class:`SnapshotChannel`: the scene's ``PropagationModel`` evaluated at the time the
    reported positions refer to (``t_b - position_delay_s``). Positions are analytic
    functions of time, so this is the channel at the reported positions when they carry no
    noise. It is oracle geometry (the controller knows the scene exactly).
  - :class:`GridChannel`: channels precomputed at grid positions of one moving node (e.g.
    a 1 cm grid along the corridor of E-A); the nearest grid point to the reported position
    is used. Works with noisy positions.
  - :class:`FunctionChannel`: any user function.

  Each returns a :class:`sremu.contracts.CsiView` per link: ``h_direct`` and per panel the
  (S, N) cascade, carrier phase included, so ``h(cfg) = h_direct + sum_n Z[s_n, n]``.
* :func:`csi_from_snapshot` and :func:`narrowband_coefficient`: the narrowband channel of a
  link from any ``PropagationModel`` (contract types only), the same quantities the
  batch path evaluates (section 4.11).
* Control granularity: :func:`aggregate_controls` sums the cascade over the cells of each
  control (``Z_ctrl[s, g] = sum_{n: g(n) = g} Z[s, n]``); :func:`expand_controls` maps
  control states back to cells.
* :func:`make_observation` builds an :class:`sremu.contracts.Observation` (tests, the
  gym-style environment).

Units: SI; exact times are ``Fraction`` seconds (floats are rejected, section 3.1).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sremu import rng as _rng
from sremu.contracts import (
    Action,
    CommandReceipt,
    Controller,
    CsiView,
    LatencyModel,
    LinkId,
    NodeId,
    Observation,
    ObservationSpec,
    PanelConfig,
    PanelId,
    PropagationModel,
    RisCapabilities,
    Snapshot,
    StreamId,
    split_link_id,
    split_stream_id,
)
from sremu.errors import ConfigurationError, UnknownTarget
from sremu.timebase import parse_exact

__all__ = [
    "BaseController",
    "ChannelModel",
    "FunctionChannel",
    "GridChannel",
    "SnapshotChannel",
    "aggregate_controls",
    "controller_rng",
    "csi_from_snapshot",
    "expand_controls",
    "link_nodes",
    "make_observation",
    "narrowband_coefficient",
    "position_time",
    "setup_controller",
    "wrap_phase",
]

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- keyed RNG


def controller_rng(seed: int, name: str) -> np.random.Generator:
    """The keyed generator of controller ``name``: purpose ``ctrl``, name ``name``
    (``sremu.rng.generator(seed, "ctrl", name)``, sections 6.1-6.2, D14)."""
    return _rng.generator(seed, "ctrl", name)


def setup_controller(
    ctrl: Controller, caps: Mapping[PanelId, RisCapabilities], seed: int
) -> np.random.Generator:
    """Call ``ctrl.setup(caps, controller_rng(seed, ctrl.name))`` and return the generator."""
    g = controller_rng(seed, ctrl.name)
    ctrl.setup(caps, g)
    return g


# --------------------------------------------------------------------------- small helpers


def wrap_phase(x: np.ndarray | float) -> np.ndarray:
    """Wrap phases (rad) to ``(-pi, pi]``."""
    y = np.mod(np.asarray(x, dtype=np.float64) + math.pi, _TWO_PI) - math.pi
    return np.where(y == -math.pi, math.pi, y)


def link_nodes(link: LinkId) -> tuple[NodeId, NodeId]:
    """(TX node, RX node) of a link id ``<tx node>.<s>-><rx node>.<s>``."""
    tx, rx = split_link_id(link)
    return split_stream_id(tx)[0], split_stream_id(rx)[0]


def position_time(obs: Observation, spec: ObservationSpec) -> Fraction:
    """Exact time the reported positions refer to: ``max(0, t_b - position_delay_s)``."""
    return max(Fraction(0), Fraction(obs.t_b) - Fraction(spec.position_delay_s))


def aggregate_controls(
    z: np.ndarray, control_of_cell: np.ndarray | None, n_controls: int
) -> np.ndarray:
    """Cascade per control: ``Z_ctrl[s, g] = sum_{n: control_of_cell[n] = g} Z[s, n]``.

    ``z`` is (S, N) complex. ``control_of_cell`` is (N,) int in [0, n_controls) or None for
    element granularity (then N must equal ``n_controls`` and ``z`` is returned as is).
    """
    zz = np.asarray(z, dtype=np.complex128)
    if zz.ndim != 2:
        raise ConfigurationError(f"cascade must be (S, N), got shape {zz.shape}")
    if control_of_cell is None:
        if zz.shape[1] != n_controls:
            raise ConfigurationError(
                f"the panel has {zz.shape[1]} cells but {n_controls} controls; give "
                "control_of_cell (granularity other than 'element')"
            )
        return zz
    m = np.asarray(control_of_cell, dtype=np.int64)
    if m.shape != (zz.shape[1],):
        raise ConfigurationError(f"control_of_cell must be ({zz.shape[1]},), got {m.shape}")
    if m.size and (int(m.min()) < 0 or int(m.max()) >= n_controls):
        raise ConfigurationError(f"control_of_cell values must lie in [0, {n_controls})")
    out = np.zeros((zz.shape[0], n_controls), dtype=np.complex128)
    for s in range(zz.shape[0]):  # fixed summation order (cells in index order)
        np.add.at(out[s], m, zz[s])
    return out


def expand_controls(states: np.ndarray, control_of_cell: np.ndarray | None) -> np.ndarray:
    """Per-cell states (N,) int16 from per-control states (``states[control_of_cell]``)."""
    s = np.asarray(states, dtype=np.int16)
    if control_of_cell is None:
        return s.copy()
    return s[np.asarray(control_of_cell, dtype=np.int64)].astype(np.int16)


# --------------------------------------------------------------------------- channel views


def _cycles(fc_hz: float, delay_s: np.ndarray) -> np.ndarray:
    return np.mod(fc_hz * np.asarray(delay_s, dtype=np.float64), 1.0)


def csi_from_snapshot(model: PropagationModel, snap: Snapshot, link: LinkId) -> CsiView:
    """Narrowband CSI of one link from any ``PropagationModel`` snapshot (contract types only).

    ``h_direct = sum_p amp_p exp(-j 2 pi f_c tau_p)`` over the link's ``PathBundle`` (all
    non-configurable paths, structural gap reflections included); per panel the (S, N)
    cascade ``Z[s, n] = sum_rows B[r, s, n] exp(-j 2 pi f_c tau_tile(r, n))`` over every row
    (hop variant) of the link (``B[r, s, n] = gamma[s] C[r, n]`` for separable tables).
    The carrier phase uses cycles reduced modulo 1 (D-1). Dead cells (state -1) contribute
    nothing in :meth:`CsiView.evaluate`.
    """
    if link not in snap.paths:
        raise UnknownTarget(f"unknown link {link!r}")
    fc = float(model.link_fc_hz(link))
    pb = snap.paths[link]
    h0 = complex(np.sum(pb.amp * np.exp(-2j * math.pi * _cycles(fc, pb.delay_s))))
    cascade: dict[PanelId, np.ndarray] = {}
    for pid, rc in snap.ris.items():
        acc = np.zeros((rc.n_states, rc.n_cells), dtype=np.complex128)
        for r, (rlk, _variant) in enumerate(rc.rows):
            if rlk != link:
                continue
            ph = np.exp(-2j * math.pi * _cycles(fc, rc.tile_delay_s[r][rc.tile_of_cell]))
            if rc.B is not None:
                acc += rc.B[r] * ph[None, :]
            else:
                assert rc.C is not None
                assert rc.gamma is not None
                acc += rc.gamma[:, None] * (rc.C[r] * ph)[None, :]
        cascade[pid] = acc
    return CsiView(link, snap.t, fc, h0, cascade)


def narrowband_coefficient(
    model: PropagationModel,
    snap: Snapshot,
    cfg: Mapping[PanelId, PanelConfig],
    link: LinkId,
) -> complex:
    """``h = sum_p amp_p exp(-j 2 pi f_c tau_p)`` over ``model.combine(snap, cfg)[link]``
    (the coefficient the batch path uses; ``P_R = P_T |h|^2``)."""
    paths = model.combine(snap, cfg)
    if link not in paths:
        raise UnknownTarget(f"unknown link {link!r}")
    lp = paths[link]
    fc = float(model.link_fc_hz(link))
    return complex(np.sum(lp.amp * np.exp(-2j * math.pi * _cycles(fc, lp.delay_s))))


@runtime_checkable
class ChannelModel(Protocol):
    """A controller's model of the channel at reported positions (section 8.7)."""

    def csi(
        self, link: LinkId, positions: Mapping[str, np.ndarray] | None, t_s: Fraction
    ) -> CsiView:
        """CSI of ``link`` for the reported ``positions`` (node -> (3,) m), which refer to
        the exact time ``t_s`` (s)."""
        ...


class SnapshotChannel:
    """The scene model evaluated at the time the reported positions refer to.

    ``csi(link, positions, t_s)`` is :func:`csi_from_snapshot` of ``model.snapshot(t_s)``;
    ``positions`` are not used (they are a function of ``t_s`` in the model). This is
    oracle geometry: the controller knows the scene exactly. The last snapshot is cached.
    """

    def __init__(self, model: PropagationModel) -> None:
        self.model = model
        self._t: Fraction | None = None
        self._snap: Snapshot | None = None

    def snapshot(self, t_s: Fraction) -> Snapshot:
        """``model.snapshot(t_s)``, cached for repeated calls at the same time."""
        t = parse_exact(t_s, name="t_s")
        if self._t != t or self._snap is None:
            self._snap = self.model.snapshot(t)
            self._t = t
        return self._snap

    def csi(
        self, link: LinkId, positions: Mapping[str, np.ndarray] | None, t_s: Fraction
    ) -> CsiView:
        """CSI of ``link`` at time ``t_s`` (s)."""
        return csi_from_snapshot(self.model, self.snapshot(t_s), link)


class FunctionChannel:
    """Wrap a function ``fn(link, positions, t_s) -> CsiView`` as a :class:`ChannelModel`."""

    def __init__(self, fn: Callable[[LinkId, Mapping[str, np.ndarray] | None, Fraction], CsiView]):
        self.fn = fn

    def csi(
        self, link: LinkId, positions: Mapping[str, np.ndarray] | None, t_s: Fraction
    ) -> CsiView:
        """``fn(link, positions, t_s)``."""
        return self.fn(link, positions, t_s)


class GridChannel:
    """Channels precomputed at grid positions of one moving node.

    ``points`` (G, 3) m are the node's grid positions and ``views[i]`` maps each link to its
    CSI at ``points[i]``. :meth:`index` returns the nearest grid point (Euclidean; the lowest
    index on ties) to ``positions[node]``; :meth:`csi` returns that point's CSI. Build it
    from a scene model with :meth:`from_model`.
    """

    def __init__(
        self,
        node: NodeId,
        points: np.ndarray,
        views: Sequence[Mapping[LinkId, CsiView]],
    ) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 1:
            raise ConfigurationError(f"grid points must be (G, 3) with G >= 1, got {pts.shape}")
        if len(views) != pts.shape[0]:
            raise ConfigurationError(f"{len(views)} views for {pts.shape[0]} grid points")
        self.node = node
        self.points = pts
        self.views = [dict(v) for v in views]

    @classmethod
    def from_model(
        cls,
        model: PropagationModel,
        node: NodeId,
        times_s: Sequence[Fraction | int | str],
        *,
        links: Sequence[LinkId] | None = None,
        points: np.ndarray | None = None,
    ) -> GridChannel:
        """Sample ``model`` at exact times ``times_s`` while ``node`` moves over the grid.

        The grid point of time ``t`` is ``points[i]`` when given, else
        ``model.node_position(node, t)`` (the physics and fake models provide it).
        ``links`` defaults to every link of the model.
        """
        ts = [parse_exact(t, name="times_s") for t in times_s]
        lks = tuple(model.links()) if links is None else tuple(links)
        if points is None:
            pos_fn = getattr(model, "node_position", None)
            if pos_fn is None:
                raise ConfigurationError(
                    "the model has no node_position(); pass the grid points explicitly"
                )
            pts = np.stack([np.asarray(pos_fn(node, t), dtype=np.float64) for t in ts])
        else:
            pts = np.asarray(points, dtype=np.float64)
        views = []
        for t in ts:
            snap = model.snapshot(t)
            views.append({lk: csi_from_snapshot(model, snap, lk) for lk in lks})
        return cls(node, pts, views)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def index(self, positions: Mapping[str, np.ndarray] | None) -> int:
        """Index of the grid point nearest to ``positions[node]``."""
        if positions is None or self.node not in positions:
            raise ConfigurationError(
                f"GridChannel needs the position of node {self.node!r} in the observation"
            )
        p = np.asarray(positions[self.node], dtype=np.float64)
        d = self.points - p[None, :]
        return int(np.argmin((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]))

    def csi(
        self, link: LinkId, positions: Mapping[str, np.ndarray] | None, t_s: Fraction
    ) -> CsiView:
        """CSI of ``link`` at the grid point nearest to the reported position."""
        view = self.views[self.index(positions)]
        if link not in view:
            raise UnknownTarget(f"link {link!r} is not in the grid channel")
        return view[link]


# --------------------------------------------------------------------------- observations


def make_observation(
    t_b: Fraction | int | str,
    *,
    ris_state: Mapping[PanelId, np.ndarray],
    rx_power_dbm: Mapping[StreamId, float] | None = None,
    positions: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    csi_oracle: Mapping[LinkId, CsiView] | None = None,
    iq_window: Mapping[StreamId, np.ndarray] | None = None,
    k_b: Mapping[StreamId, int] | None = None,
    pending: Sequence[CommandReceipt] = (),
) -> Observation:
    """An :class:`Observation` at barrier time ``t_b`` (s) for tests and the environment."""
    pos = (
        None
        if positions is None
        else {k: np.asarray(v, dtype=np.float64).reshape(3) for k, v in positions.items()}
    )
    return Observation(
        t_b=parse_exact(t_b, name="t_b"),
        k_b=dict(k_b or {}),
        rx_power_dbm=dict(rx_power_dbm or {}),
        positions=pos,
        csi_oracle=None if csi_oracle is None else dict(csi_oracle),
        iq_window=None if iq_window is None else dict(iq_window),
        ris_state={p: np.asarray(s, dtype=np.int16) for p, s in ris_state.items()},
        pending=tuple(pending),
    )


# --------------------------------------------------------------------------- base class


class BaseController:
    """Common attributes and setup of the built-in controllers (contract ``Controller``).

    * ``name``: controller id (also the RNG name of purpose ``ctrl``).
    * ``period_s``: barrier period T_c, s (exact, > 0).
    * ``min_latency_s``: declared lookahead L_min, s (exact, >= 0).
    * ``latency``: :class:`LatencyModel`; default ``fixed_s = min_latency_s``. Its
      ``fixed_s`` must be >= ``min_latency_s`` (section 4.9).
    * ``spec``: :class:`ObservationSpec`; the controller's :attr:`required_fields` are
      always added to its fields.

    ``setup(caps, rng)`` keeps the capabilities and the keyed generator (purpose
    ``ctrl/<name>``) and calls :meth:`reset`. Subclasses implement :meth:`decide`.
    """

    required_fields: tuple[str, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        period_s: Fraction | int | str = Fraction(1, 1000),
        min_latency_s: Fraction | int | str = 0,
        latency: LatencyModel | None = None,
        spec: ObservationSpec | None = None,
    ) -> None:
        if not isinstance(name, str) or not name or "\x1f" in name:
            raise ConfigurationError(f"invalid controller name {name!r}")
        self.name = name
        self.period_s = parse_exact(period_s, name="period_s")
        if self.period_s <= 0:
            raise ConfigurationError("period_s must be > 0")
        self.min_latency_s = parse_exact(min_latency_s, name="min_latency_s")
        if self.min_latency_s < 0:
            raise ConfigurationError("min_latency_s must be >= 0")
        self.latency = latency if latency is not None else LatencyModel(fixed_s=self.min_latency_s)
        if Fraction(self.latency.fixed_s) < self.min_latency_s:
            raise ConfigurationError(
                f"latency.fixed_s = {self.latency.fixed_s} s < min_latency_s = "
                f"{self.min_latency_s} s (actions would violate the lookahead, 4.9)"
            )
        base = spec if spec is not None else ObservationSpec()
        fields = tuple(base.fields) + tuple(f for f in self.required_fields if f not in base.fields)
        self.spec = replace(base, fields=fields)
        self.caps: dict[PanelId, RisCapabilities] = {}
        self.rng: np.random.Generator | None = None

    # ---- protocol
    def setup(self, caps: Mapping[PanelId, RisCapabilities], rng: np.random.Generator) -> None:
        """Keep the capabilities and the keyed generator; then :meth:`reset`."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("setup needs a keyed numpy Generator (sremu.rng.generator)")
        self.caps = dict(caps)
        self.rng = rng
        self.reset()

    def reset(self) -> None:
        """Reset per-run state (called by :meth:`setup`)."""

    def decide(self, obs: Observation) -> Sequence[Action]:  # pragma: no cover - abstract
        """Actions for the barrier at ``obs.t_b``."""
        raise NotImplementedError

    # ---- helpers for subclasses
    def _rng(self) -> np.random.Generator:
        if self.rng is None:
            raise ConfigurationError(f"controller {self.name!r}: setup() was not called")
        return self.rng

    def _panel(self, panel: PanelId | None) -> PanelId:
        """The configured panel, or the only panel in the capabilities."""
        if panel is not None:
            if self.caps and panel not in self.caps:
                raise UnknownTarget(f"controller {self.name!r}: unknown panel {panel!r}")
            return panel
        if len(self.caps) != 1:
            raise ConfigurationError(
                f"controller {self.name!r}: give panel= (capabilities list {sorted(self.caps)})"
            )
        return next(iter(self.caps))

    def _n_controls(self, panel: PanelId) -> int:
        if panel not in self.caps:
            raise ConfigurationError(f"controller {self.name!r}: setup() has no panel {panel!r}")
        cap = self.caps[panel]
        return int(cap.n_controls if cap.n_controls is not None else cap.n_elements)

    def describe(self) -> dict[str, Any]:
        """JSON-ready description (name, class, period, latency, observation fields)."""
        return {
            "name": self.name,
            "impl": type(self).__name__,
            "period_s": str(self.period_s),
            "min_latency_s": str(self.min_latency_s),
            "latency_fixed_s": str(self.latency.fixed_s),
            "latency_jitter": self.latency.jitter,
            "fields": list(self.spec.fields),
            "position_delay_s": str(self.spec.position_delay_s),
        }
