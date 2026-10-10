"""``PropagationModel`` implementation: LOS + K1 RIS rows (DESIGN.md sections 3.4, 5.1-5.6; WP2).

:class:`ScenePropagation` implements :class:`sremu.contracts.PropagationModel`:

* ``links()``: TX streams (node order) x RX streams (node order), different nodes, equal
  rates and ``|fc_tx - fc_rx| < f_s/2`` (contract ``check_link``).
* ``snapshot(t)``: a pure function of (scenario, t). Positions come from analytic
  trajectories at the exact time ``t``. Per link a :class:`PathBundle` (LOS with explicit
  occluder/partition factors, plus mechanism paths from WP3); per panel a
  :class:`RISCoefficients` with one row per (link, hop variant) and
  ``B[r, s, n] = sum_{q in n} Gamma_s(f, theta_iq, theta_sq) * K1_q`` whose phase is
  relative to the cell's tile reference (``exp(-jk(r_i + r_s - L_tile))``), or the
  separable ``C[r, n]`` with ``gamma[s]`` when the table is angle-independent and every row
  sees the same alphabet. Efficiency is already applied to Gamma.
* ``combine(snap, cfg)``: linear in Gamma; one RIS_TILE path per (panel, row, tile) with
  amplitude ``g_k = sum_{n in k} B[r, s_n, n]`` and delay ``tau_tile``. The path set is fixed
  (disabled or unconfigured panels keep zero-amplitude paths), so paths stay aligned
  across configurations (transitions, interpolation).
* ``ris_delay_s(panel, rx, t)``: ``|p_rx(t) - c_panel| / C0``.
* ``motion_bounds()``: speeds and closest approaches over ``[0, horizon_s]`` (exact for
  piecewise-linear trajectories).

**Mechanism hook (WP3 integration point).** ``mechanisms`` are objects with any of:
``los_factor(ctx) -> complex`` (multiplies the LOS ``V``: knife edge ``F(nu)``),
``link_paths(ctx) -> list[(amp, delay_s, PathKind)]`` (ground image, walls; amplitudes
exclude the propagation phase) and ``ris_hops(ctx, panel) -> list[RisHopSource]`` (ground
image expansion on RIS hops, variants 1-3). ``ctx`` is a :class:`LinkContext`.

Inter-tile gaps are inert by default. A structural ``gap_gamma`` adds one fixed
(non-configurable) path per (panel, link) with kind ``RIS_TILE`` in the ``PathBundle``
(tile key -1 in ``combine``); its delay is the path through the panel centre.
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sremu.contracts import (
    CsiView,
    Granularity,
    LinkId,
    LinkPaths,
    MotionBounds,
    NodeId,
    PanelConfig,
    PanelId,
    PathBundle,
    PathKind,
    RisCapabilities,
    RISCoefficients,
    Snapshot,
    StreamId,
    StreamSpec,
    check_link,
    link_id,
    split_stream_id,
    stream_id,
)
from sremu.errors import ConfigurationError, GeometryError, UnknownTarget, ValidityWarning
from sremu.physics.antennas import Antenna, Isotropic, antenna_from_spec
from sremu.physics.geometry import (
    PanelGrid,
    Pose,
    Trajectory,
    angles_to_node,
    keyed_generator,
    point_rectangle_distance,
    trajectory_from_spec,
)
from sremu.physics.los import Occluder, los_path, los_visibility, occluders_from_spec
from sremu.physics.nonideal import CouplingHook, FailureMap, PanelPipeline, control_map
from sremu.physics.quadrature import (
    LinkSample,
    Quadrature,
    RichardsonResult,
    build_quadrature,
    gap_quadrature,
    resolve_cell_model,
    richardson_check,
)
from sremu.physics.response import ResponseTable, load_table
from sremu.physics.ris_kernel import (
    Backend,
    Obliquity,
    accumulate_cells,
    k1_terms,
    tile_sums,
)
from sremu.timebase import parse_exact
from sremu.units import C0

__all__ = [
    "LinkContext",
    "NodeModel",
    "Numerics",
    "PanelModel",
    "RisHopSource",
    "ScenePropagation",
    "from_scenario",
]

_OUT_OF_PLANE_TOL_DEG = 1.0


@dataclass(frozen=True)
class Numerics:
    """Physics numerics (scenario ``numerics``, section 9.2)."""

    point_max_pitch_wavelengths: float = 0.5
    subcell_max_wavelengths: float = 0.2
    richardson_tol_db: float = 0.05
    richardson_max_halvings: int = 2


@dataclass(eq=False)
class NodeModel:
    """A node: trajectory, antenna and streams (ids ``<node>.<stream>``)."""

    id: NodeId
    trajectory: Trajectory
    antenna: Antenna = field(default_factory=Isotropic)
    streams: tuple[StreamSpec, ...] = ()
    occupied_bw_hz: Mapping[StreamId, float | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.streams = tuple(self.streams)
        for s in self.streams:
            if s.node != self.id:
                raise ConfigurationError(f"stream {s.stream!r} does not belong to node {self.id!r}")


@dataclass(eq=False)
class PanelModel:
    """A RIS panel: geometry, response table (efficiency applied) and pipeline settings."""

    id: PanelId
    grid: PanelGrid
    table: ResponseTable
    cell_model: Literal["auto", "point", "po_patch"] = "auto"
    obliquity: Obliquity = "sqrt"
    gamma_dead: complex = 0j
    gap_gamma: complex | None = None
    render_tiles: tuple[int, int] = (1, 1)
    granularity: Granularity = "element"
    groups: Mapping[str, Sequence[int]] | None = None
    failures: FailureMap | None = None
    initial_state: np.ndarray | None = None
    enabled: bool = True
    transition: Mapping[str, Any] = field(default_factory=lambda: {"model": "step"})
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    out_of_plane: Literal["project", "error"] = "project"
    coupling: CouplingHook | None = None


@dataclass(frozen=True, eq=False)
class LinkContext:
    """What a mechanism sees for one link at one snapshot."""

    link: LinkId
    t: Fraction
    tx_node: NodeId
    rx_node: NodeId
    p_tx: np.ndarray
    p_rx: np.ndarray
    fc_hz: float
    ant_tx: Antenna
    ant_rx: Antenna


@dataclass(frozen=True, eq=False)
class RisHopSource:
    """An image-expanded RIS row (ground on RIS hops, WP3): hop ``variant`` in {1, 2, 3}.

    ``p_src``/``p_dst``: positions the incident/outgoing spherical waves appear to come
    from / go to (Tx or its image, Rx or its image). ``weight(points) -> (Q,)`` complex
    ground factors per quadrature point. ``gain_tx``/``gain_rx(points) -> (Q,)`` power gains
    of the real antennas along the image paths (default: toward ``points`` from
    ``p_src``/``p_dst``).
    """

    variant: int
    p_src: np.ndarray
    p_dst: np.ndarray
    weight: Callable[[np.ndarray], np.ndarray]
    gain_tx: Callable[[np.ndarray], np.ndarray] | None = None
    gain_rx: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass(eq=False)
class _PanelState:
    model: PanelModel
    resolved: Literal["point", "po_patch"]
    quad: Quadrature
    richardson: RichardsonResult | None
    tile_of_cell: np.ndarray
    tile_ref: np.ndarray
    q_tile: np.ndarray
    gap_points: np.ndarray
    gap_areas: np.ndarray
    obliquity: Obliquity
    pipeline: PanelPipeline
    wavelength_m: float


def _dist(points: np.ndarray, p: np.ndarray) -> np.ndarray:
    d = points - p[None, :]
    return np.sqrt((d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2])


def _dist_rows(points: np.ndarray, srcs: np.ndarray) -> np.ndarray:
    """(R, K) distances from R sources to K points (same rounding as :func:`_dist`)."""
    d = points[None, :, :] - srcs[:, None, :]
    return np.sqrt((d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) + d[..., 2] * d[..., 2])


def _stack_gains(gains: Sequence[np.ndarray | None], q: int) -> np.ndarray | None:
    """(R, Q) gain array, or None when every row is isotropic."""
    if all(g is None for g in gains):
        return None
    return np.stack([np.ones(q) if g is None else np.asarray(g, np.float64) for g in gains])


def _gain_toward(ant: Antenna, src: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if isinstance(ant, Isotropic):
        return np.ones(pts.shape[0])
    d = pts - src[None, :]
    return np.asarray(ant.gain(d / _dist(pts, src)[:, None]), dtype=np.float64)


def _complex(x: Any, name: str) -> complex:
    if x is None:
        return 0j
    if isinstance(x, list | tuple) and len(x) == 2:
        return complex(float(x[0]), float(x[1]))
    try:
        return complex(str(x).replace(" ", "")) if isinstance(x, str) else complex(x)
    except ValueError as exc:
        raise ConfigurationError(f"{name}: cannot read {x!r} as a complex number") from exc


def _closest_approach(dp0: np.ndarray, dp1: np.ndarray) -> float:
    """min over s in [0, 1] of |dp0 + s (dp1 - dp0)|."""
    dv = dp1 - dp0
    vv = float(dv @ dv)
    s = 0.0 if vv == 0.0 else min(max(-float(dp0 @ dv) / vv, 0.0), 1.0)
    return float(np.linalg.norm(dp0 + s * dv))


class ScenePropagation:
    """LOS + K1 RIS propagation model (contract ``PropagationModel``).

    ``backend``: ``numpy_ref`` (float64 reference) or ``numba_cpu`` for the K1 kernel.
    ``horizon_s``: run duration used by :meth:`motion_bounds`. ``seed``: master seed
    (keyed purposes ``failure/<panel>``, ``mobility/<node>``).
    """

    def __init__(
        self,
        nodes: Sequence[NodeModel],
        panels: Sequence[PanelModel] = (),
        *,
        occluders: Iterable[Occluder] = (),
        mechanisms: Iterable[Any] = (),
        backend: Backend = "numpy_ref",
        horizon_s: str | int | Fraction = 10,
        numerics: Numerics | None = None,
        seed: int = 0,
    ) -> None:
        if backend not in ("numpy_ref", "numba_cpu"):
            raise ConfigurationError(f"unknown backend {backend!r}")
        self.backend: Backend = backend
        self.numerics = numerics or Numerics()
        self.seed = int(seed)
        self.horizon_s = float(parse_exact(horizon_s, name="horizon_s"))
        self.occluders = tuple(occluders)
        self.mechanisms = tuple(mechanisms)
        self._nodes: dict[NodeId, NodeModel] = {}
        self._streams: dict[StreamId, StreamSpec] = {}
        for nd in nodes:
            if nd.id in self._nodes:
                raise ConfigurationError(f"duplicate node id {nd.id!r}")
            self._nodes[nd.id] = nd
            for s in nd.streams:
                if s.stream in self._streams:
                    raise ConfigurationError(f"duplicate stream id {s.stream!r}")
                self._streams[s.stream] = s
        txs = [s for s in self._streams.values() if s.direction == "tx"]
        rxs = [s for s in self._streams.values() if s.direction == "rx"]
        self._links: dict[LinkId, tuple[StreamSpec, StreamSpec]] = {}
        for tx in txs:
            for rx in rxs:
                if tx.node != rx.node and check_link(tx.fc_hz, tx.rate_sps, rx.fc_hz, rx.rate_sps):
                    self._links[link_id(tx.stream, rx.stream)] = (tx, rx)
        self._panels: dict[PanelId, _PanelState] = {}
        self._warned: set[tuple[str, str]] = set()
        for p in panels:
            if p.id in self._panels:
                raise ConfigurationError(f"duplicate panel id {p.id!r}")
            self._panels[p.id] = self._prepare_panel(p)

    # ------------------------------------------------------------------ preparation
    def _occupied_bw(self, tx: StreamSpec) -> float:
        nd = self._nodes[tx.node]
        bw = nd.occupied_bw_hz.get(tx.stream)
        return float(bw) if bw is not None else 0.8 * float(tx.rate_sps)

    def _prepare_panel(self, p: PanelModel) -> _PanelState:
        fcs = sorted({tx.fc_hz for tx, _ in self._links.values()})
        if not fcs:
            fcs = sorted({s.fc_hz for s in self._streams.values()}) or [1.0]
        lam = C0 / max(fcs)
        for tx, _rx in self._links.values():
            p.table.check_bandwidth(self._occupied_bw(tx), tx.stream)
        if p.coupling is not None and not p.table.angle_independent:
            raise ConfigurationError(
                f"panel {p.id}: the coupling hook works on separable tables only (section 5.6 "
                "step 6); this table depends on angle"
            )
        if p.gamma_dead != 0 and not p.table.angle_independent:  # fail at build, not at t
            raise ConfigurationError(
                f"panel {p.id}: a structural gamma_dead needs a separable table in release 1 "
                "(the B form carries no geometric factor for dead cells)"
            )
        resolved = resolve_cell_model(
            p.cell_model,
            p.grid.pitch_u_m,
            p.grid.pitch_v_m,
            lam,
            point_max_pitch_wavelengths=self.numerics.point_max_pitch_wavelengths,
            includes_cell_pattern=p.table.includes_cell_pattern,
        )
        obliq: Obliquity = "none" if p.table.includes_cell_pattern else p.obliquity
        rich: RichardsonResult | None = None
        level = 0
        if resolved == "po_patch":
            samples = []
            for lk, (tx, rx) in self._links.items():
                samples.append(
                    LinkSample(
                        lk,
                        self._nodes[tx.node].trajectory.position(Fraction(0)),
                        self._nodes[rx.node].trajectory.position(Fraction(0)),
                        tx.fc_hz,
                        self._nodes[tx.node].antenna,
                        self._nodes[rx.node].antenna,
                    )
                )
            rich = richardson_check(
                p.grid,
                lam,
                samples,
                subcell_max_wavelengths=self.numerics.subcell_max_wavelengths,
                tol_db=self.numerics.richardson_tol_db,
                max_halvings=self.numerics.richardson_max_halvings,
                obliquity=obliq,
                backend=self.backend,
            )
            level = rich.level
        quad = build_quadrature(
            p.grid,
            resolved,
            lam,
            subcell_max_wavelengths=self.numerics.subcell_max_wavelengths,
            level=level,
        )
        ku, kv = p.render_tiles
        tile = p.grid.render_tile_of_cell(ku, kv)
        centres = p.grid.cell_centres()
        tref = np.stack([centres[tile == k].mean(axis=0) for k in range(ku * kv)])
        if p.gap_gamma is not None:
            gp, ga = gap_quadrature(
                p.grid, lam, subcell_max_wavelengths=self.numerics.subcell_max_wavelengths
            )
        else:
            gp, ga = np.zeros((0, 3)), np.zeros(0)
        ctrl, _names = control_map(
            p.granularity,
            p.grid.n_u,
            p.grid.n_v,
            tiles=(p.grid.n_tiles_u, p.grid.n_tiles_v),
            groups=p.groups,
        )
        gref = self._gamma_reference(p.table, fcs[0])
        pipe = PanelPipeline(p.id, ctrl, gref, p.failures or FailureMap.healthy(p.grid.n_cells))
        if p.initial_state is not None and np.asarray(p.initial_state).shape != (p.grid.n_cells,):
            raise ConfigurationError(
                f"panel {p.id}: initial state must have {p.grid.n_cells} cells"
            )
        return _PanelState(
            p, resolved, quad, rich, tile, tref, tile[quad.parent], gp, ga, obliq, pipe, lam
        )

    @staticmethod
    def _gamma_reference(table: ResponseTable, fc_hz: float) -> np.ndarray:
        """Quantizer reference: the table at normal incidence and reflection (0, 0)."""
        if table.angle_independent:
            return table.gamma_at(fc_hz)
        return table.query(fc_hz, 0.0, 0.0)[:, 0]

    # ------------------------------------------------------------------ accessors
    @property
    def nodes(self) -> Mapping[NodeId, NodeModel]:
        """Nodes by id."""
        return dict(self._nodes)

    @property
    def streams(self) -> Mapping[StreamId, StreamSpec]:
        """Streams by id."""
        return dict(self._streams)

    def panel_ids(self) -> tuple[PanelId, ...]:
        """Panels in scenario order."""
        return tuple(self._panels)

    def panel(self, pid: PanelId) -> PanelModel:
        """The panel model."""
        return self._state(pid).model

    def _state(self, pid: PanelId) -> _PanelState:
        try:
            return self._panels[pid]
        except KeyError:
            raise UnknownTarget(f"unknown panel {pid!r}") from None

    def quadrature(self, pid: PanelId) -> Quadrature:
        """The panel's quadrature (after the Richardson check)."""
        return self._state(pid).quad

    def cell_model(self, pid: PanelId) -> Literal["point", "po_patch"]:
        """Resolved cell model of a panel."""
        return self._state(pid).resolved

    def richardson(self, pid: PanelId) -> RichardsonResult | None:
        """Richardson result (None for ``point``)."""
        return self._state(pid).richardson

    def pipeline(self, pid: PanelId) -> PanelPipeline:
        """Non-ideality pipeline (steps 1-3) of a panel."""
        return self._state(pid).pipeline

    def link_streams(self, link: LinkId) -> tuple[StreamSpec, StreamSpec]:
        """(tx, rx) stream specs of a link."""
        try:
            return self._links[link]
        except KeyError:
            raise UnknownTarget(f"unknown link {link!r}") from None

    def node_position(self, node: NodeId, t: Fraction | int | str) -> np.ndarray:
        """Position (3,) of ``node`` at exact time ``t``, m."""
        if node not in self._nodes:
            raise UnknownTarget(f"unknown node {node!r}")
        return self._nodes[node].trajectory.position(parse_exact(t, name="t"))

    def set_node_position(self, node: NodeId, t: Fraction, position_m: Sequence[float]) -> None:
        """Apply a timed ``node.set_position`` event (node must use an external trajectory)."""
        traj = self._nodes[node].trajectory
        if not hasattr(traj, "set_position"):
            raise ConfigurationError(f"node {node!r} does not have an external trajectory")
        traj.set_position(t, position_m)

    def set_node_trajectory(self, node: NodeId, t: Fraction, traj: Trajectory) -> None:
        """Apply a timed ``node.set_trajectory`` event (external trajectory nodes only)."""
        cur = self._nodes[node].trajectory
        if not hasattr(cur, "set_trajectory"):
            raise ConfigurationError(f"node {node!r} does not have an external trajectory")
        cur.set_trajectory(t, traj)

    def initial_config(self, pid: PanelId) -> PanelConfig:
        """The panel's initial configuration (scenario ``initial``), failures applied."""
        st = self._state(pid)
        m = st.model
        s = (
            np.zeros(m.grid.n_cells, dtype=np.int16)
            if m.initial_state is None
            else (np.asarray(m.initial_state, dtype=np.int16))
        )
        return PanelConfig(pid, st.pipeline.failures.apply(s), m.enabled)

    def with_render_tiles(self, tiles: Mapping[PanelId, tuple[int, int]]) -> ScenePropagation:
        """A copy using render (Doppler) tiles ``(tiles_u, tiles_v)`` per panel (section 4.4)."""
        new = copy.copy(self)
        new._panels = dict(self._panels)
        for pid, (ku, kv) in tiles.items():
            st = copy.copy(self._state(pid))
            model = copy.copy(st.model)
            model.render_tiles = (int(ku), int(kv))
            tile = model.grid.render_tile_of_cell(int(ku), int(kv))
            centres = model.grid.cell_centres()
            st.model = model
            st.tile_of_cell = tile
            st.tile_ref = np.stack([centres[tile == k].mean(axis=0) for k in range(ku * kv)])
            st.q_tile = tile[st.quad.parent]
            new._panels[pid] = st
        return new

    def capabilities(self, pid: PanelId) -> RisCapabilities:
        """``RisCapabilities`` of a panel (GS RIS 008 names, section 8.5)."""
        st = self._state(pid)
        m = st.model
        t = m.table
        src = (
            "measured:" + (t.provenance.raw_sha256 or t.sha256())
            if t.provenance.source_kind in ("measured", "digitized")
            else f"analytic:{t.name or 'table'}"
        )
        caps = dict(m.capabilities)
        rt = caps.get("reconfiguration_time_s", t.reconfiguration_time_s)
        return RisCapabilities(
            n_elements=m.grid.n_cells,
            layout=(m.grid.n_u, m.grid.n_v),
            control_granularity=m.granularity,
            states=t.labels,
            gamma_source=src,
            operating_band_hz=t.operating_band_hz(),
            max_bandwidth_hz=t.validity_bw_hz if t.validity_bw_hz is not None else math.inf,
            reconfiguration_time_s=None
            if rt is None
            else parse_exact(
                rt if not isinstance(rt, float) else str(rt), name="reconfiguration_time_s"
            ),
            max_command_rate_hz=float(caps.get("max_command_rate_hz", 1e4)),
            static_power_w=caps.get("static_power_w"),
            n_controls=st.pipeline.n_controls,
        )

    # ------------------------------------------------------------------ contract
    def links(self) -> tuple[LinkId, ...]:
        """All links in scenario order."""
        return tuple(self._links)

    def _context(self, lk: LinkId, t: Fraction, pos: Mapping[NodeId, np.ndarray]) -> LinkContext:
        tx, rx = self._links[lk]
        return LinkContext(
            lk,
            t,
            tx.node,
            rx.node,
            pos[tx.node],
            pos[rx.node],
            tx.fc_hz,
            self._nodes[tx.node].antenna,
            self._nodes[rx.node].antenna,
        )

    def snapshot(self, t: Fraction | int | str) -> Snapshot:
        """Geometry at exact time ``t`` (s): a pure function of (scenario, t)."""
        tq = parse_exact(t, name="t")
        pos = {nid: nd.trajectory.position(tq) for nid, nd in self._nodes.items()}
        ctxs = {lk: self._context(lk, tq, pos) for lk in self._links}
        paths: dict[LinkId, PathBundle] = {}
        diag: dict[str, float] = {}
        gap_paths = self._gap_paths(ctxs)
        for lk, ctx in ctxs.items():
            v = (
                complex(los_visibility(ctx.p_tx, ctx.p_rx, self.occluders))
                if self.occluders
                else 1.0
            )
            for mech in self.mechanisms:
                if hasattr(mech, "los_factor"):
                    v *= complex(mech.los_factor(ctx))
            amp, tau = los_path(ctx.p_tx, ctx.p_rx, ctx.fc_hz, ctx.ant_tx, ctx.ant_rx, v)
            amps, taus, kinds = [amp], [tau], [int(PathKind.LOS)]
            for mech in self.mechanisms:
                if hasattr(mech, "link_paths"):
                    for a, d, k in mech.link_paths(ctx):
                        amps.append(complex(a))
                        taus.append(float(d))
                        kinds.append(int(k))
            for a, d in gap_paths.get(lk, []):
                amps.append(a)
                taus.append(d)
                kinds.append(int(PathKind.RIS_TILE))
            paths[lk] = PathBundle(
                lk,
                tq,
                np.asarray(amps, dtype=np.complex128),
                np.asarray(taus, dtype=np.float64),
                np.asarray(kinds, dtype=np.int8),
            )
        ris = {
            pid: self._panel_coefficients(st, tq, ctxs, diag) for pid, st in self._panels.items()
        }
        return Snapshot(t=tq, paths=paths, ris=ris, diag=diag)

    def _gap_paths(
        self, ctxs: Mapping[LinkId, LinkContext]
    ) -> dict[LinkId, list[tuple[complex, float]]]:
        out: dict[LinkId, list[tuple[complex, float]]] = {}
        for st in self._panels.values():
            g = st.model.gap_gamma
            if g is None or st.gap_points.shape[0] == 0:
                continue
            c = st.model.grid.pose.center_m
            for lk, ctx in ctxs.items():
                lref = float(_dist(c[None, :], ctx.p_tx)[0] + _dist(c[None, :], ctx.p_rx)[0])
                terms = k1_terms(
                    ctx.p_tx,
                    ctx.p_rx,
                    st.gap_points,
                    st.gap_areas,
                    st.model.grid.pose.normal,
                    ctx.fc_hz,
                    ref_len_m=lref,
                    gain_tx=_gain_toward(ctx.ant_tx, ctx.p_tx, st.gap_points)[None, :],
                    gain_rx=_gain_toward(ctx.ant_rx, ctx.p_rx, st.gap_points)[None, :],
                    obliquity=st.obliquity,
                    backend=self.backend,
                )[0]
                out.setdefault(lk, []).append((complex(g) * complex(np.sum(terms)), lref / C0))
        return out

    def _guard_distance(self, pid: str, node: str, r: np.ndarray, lam: float) -> None:
        rmin = float(r.min())
        if rmin < lam:
            raise GeometryError(
                f"panel {pid}: node {node} is {rmin:.4g} m from a cell, below one wavelength "
                f"({lam:.4g} m); the point-scatterer kernel is invalid there"
            )
        if rmin < 3 * lam and (pid, node) not in self._warned:
            self._warned.add((pid, node))
            warnings.warn(
                f"panel {pid}: node {node} is {rmin:.4g} m (< 3 lambda) from a cell; the "
                "point-scatterer approximation is marginal",
                ValidityWarning,
                stacklevel=3,
            )

    def _panel_coefficients(
        self,
        st: _PanelState,
        t: Fraction,
        ctxs: Mapping[LinkId, LinkContext],
        diag: dict[str, float],
    ) -> RISCoefficients:
        m = st.model
        pose = m.grid.pose
        pts = st.quad.points
        rows: list[tuple[LinkId, int]] = []
        srcs, dsts, fcs, gts, grs, weights = [], [], [], [], [], []
        for lk, ctx in ctxs.items():
            hops: list[RisHopSource | None] = [None]
            for mech in self.mechanisms:
                if hasattr(mech, "ris_hops"):
                    hops.extend(mech.ris_hops(ctx, m))
            for hop in hops:
                if hop is None:
                    src, dst, var = ctx.p_tx, ctx.p_rx, 0
                    gt = (
                        None
                        if isinstance(ctx.ant_tx, Isotropic)
                        else _gain_toward(ctx.ant_tx, src, pts)
                    )
                    gr = (
                        None
                        if isinstance(ctx.ant_rx, Isotropic)
                        else _gain_toward(ctx.ant_rx, dst, pts)
                    )
                    w = None
                else:
                    src, dst, var = np.asarray(hop.p_src), np.asarray(hop.p_dst), int(hop.variant)
                    gt = (
                        hop.gain_tx(pts)
                        if hop.gain_tx is not None
                        else _gain_toward(ctx.ant_tx, src, pts)
                    )
                    gr = (
                        hop.gain_rx(pts)
                        if hop.gain_rx is not None
                        else _gain_toward(ctx.ant_rx, dst, pts)
                    )
                    w = np.asarray(hop.weight(pts), dtype=np.complex128)
                rows.append((lk, var))
                srcs.append(src)
                dsts.append(dst)
                fcs.append(ctx.fc_hz)
                gts.append(gt)
                grs.append(gr)
                weights.append(w)
        r_count, n, k_count = len(rows), m.grid.n_cells, st.tile_ref.shape[0]
        tile = st.tile_of_cell.astype(np.int32)
        if r_count == 0:  # a panel without links: empty but well-formed coefficients
            if m.table.angle_independent:
                return RISCoefficients(
                    m.id,
                    t,
                    (),
                    tile,
                    np.zeros((0, k_count)),
                    None,
                    np.zeros((0, n), np.complex128),
                    m.table.gamma[:, 0, 0, 0].copy(),
                    complex(m.gamma_dead),
                )
            return RISCoefficients(
                m.id,
                t,
                (),
                tile,
                np.zeros((0, k_count)),
                np.zeros((0, m.table.n_states, n), np.complex128),
                None,
                None,
                complex(m.gamma_dead),
            )
        s_arr = np.asarray(srcs, dtype=np.float64).reshape(r_count, 3)
        d_arr = np.asarray(dsts, dtype=np.float64).reshape(r_count, 3)
        lref = _dist_rows(st.tile_ref, s_arr) + _dist_rows(st.tile_ref, d_arr)  # (R, K)
        # distance guards (r < lambda: error, r < 3 lambda: warning). The distance to the
        # panel rectangle bounds the distance to every quadrature point from below, so the
        # exact per-point check runs only for nodes within 3 lambda of the rectangle.
        rect_s = point_rectangle_distance(s_arr, m.grid)
        rect_d = point_rectangle_distance(d_arr, m.grid)
        lam_r = C0 / np.asarray(fcs)
        for rr in np.flatnonzero((rect_s < 3 * lam_r) | (rect_d < 3 * lam_r)):
            lk = rows[int(rr)][0]
            self._guard_distance(m.id, self._links[lk][0].node, _dist(pts, s_arr[rr]), lam_r[rr])
            self._guard_distance(m.id, self._links[lk][1].node, _dist(pts, d_arr[rr]), lam_r[rr])
        max_off = 0.0
        angle_table = not m.table.angle_independent
        th_i = np.empty((r_count, pts.shape[0])) if angle_table else None
        th_s = np.empty((r_count, pts.shape[0])) if angle_table else None
        if angle_table:
            assert th_i is not None
            assert th_s is not None
            for r in range(r_count):
                ai = angles_to_node(pts, s_arr[r], pose)
                as_ = angles_to_node(pts, d_arr[r], pose)
                vis = (ai["cos"] > 0) & (as_["cos"] > 0)
                if np.any(vis):
                    off = np.maximum(ai["off_plane_rad"][vis], as_["off_plane_rad"][vis])
                    max_off = max(max_off, float(np.degrees(np.max(off))))
                # back-side points contribute 0; clamp their angles into the table domain
                th_i[r] = np.where(vis, np.degrees(ai["theta_rad"]), 0.0)
                th_s[r] = np.where(vis, np.degrees(as_["theta_rad"]), 0.0)
        if angle_table and m.out_of_plane == "error" and max_off > _OUT_OF_PLANE_TOL_DEG:
            raise GeometryError(
                f"panel {m.id}: off-plane angle {max_off:.2f} deg > {_OUT_OF_PLANE_TOL_DEG} deg "
                "with out_of_plane='error'"
            )
        ref_len = lref[:, st.q_tile]
        terms = k1_terms(
            s_arr,
            d_arr,
            pts,
            st.quad.areas,
            pose.normal,
            np.asarray(fcs),
            ref_len_m=ref_len,
            gain_tx=_stack_gains(gts, pts.shape[0]),
            gain_rx=_stack_gains(grs, pts.shape[0]),
            obliquity=st.obliquity,
            backend=self.backend,
        )
        for r, w in enumerate(weights):
            if w is not None:
                terms[r] = terms[r] * w
        diag[f"max_off_plane_deg/{m.id}"] = max_off
        diag[f"quadrature_points/{m.id}"] = float(pts.shape[0])
        diag[f"richardson_db/{m.id}"] = 0.0 if st.richardson is None else st.richardson.estimate_db
        tile_delay = lref / C0
        if m.table.angle_independent:
            gam = [m.table.gamma_at(f) for f in fcs]
            if all(np.array_equal(gam[0], g) for g in gam[1:]):
                c = accumulate_cells(terms, st.quad.parent, n, backend=self.backend)[:, 0, :]
                return RISCoefficients(
                    m.id,
                    t,
                    tuple(rows),
                    tile,
                    tile_delay,
                    None,
                    np.ascontiguousarray(c),
                    np.asarray(gam[0], dtype=np.complex128),
                    complex(m.gamma_dead),
                )
            gq = np.stack([np.repeat(g[:, None], pts.shape[0], axis=1) for g in gam])
        else:
            assert th_i is not None
            assert th_s is not None
            gq = np.stack([m.table.query(fcs[r], th_i[r], th_s[r]) for r in range(r_count)])
        if m.gamma_dead != 0:
            raise ConfigurationError(
                f"panel {m.id}: a structural gamma_dead needs a separable table in release 1 "
                "(the B form carries no geometric factor for dead cells)"
            )
        b = accumulate_cells(terms, st.quad.parent, n, gq, backend=self.backend)
        return RISCoefficients(
            m.id, t, tuple(rows), tile, tile_delay, b, None, None, complex(m.gamma_dead)
        )

    def tile_amplitudes(
        self, snap: Snapshot, cfg: Mapping[PanelId, PanelConfig]
    ) -> dict[PanelId, np.ndarray | None]:
        """Per panel, the configured tile amplitudes (R, K) (the configuration GEMV, D-5), or
        None for a disabled or unconfigured panel. Linear in Gamma."""
        out: dict[PanelId, np.ndarray | None] = {}
        for pid, rc in snap.ris.items():
            conf = cfg.get(pid)
            if conf is None or not conf.enabled:
                out[pid] = None
                continue
            s = np.asarray(conf.state)
            if s.shape != (rc.n_cells,):
                raise ConfigurationError(f"panel {pid}: state shape {s.shape} != ({rc.n_cells},)")
            if s.size and (int(s.max()) >= rc.n_states or int(s.min()) < -1):
                raise ConfigurationError(f"panel {pid}: state outside [-1, {rc.n_states})")
            hook = self._panels[pid].model.coupling if pid in self._panels else None
            if rc.separable:
                assert rc.C is not None
                assert rc.gamma is not None
                alive = s >= 0
                g = np.where(alive, rc.gamma[np.where(alive, s, 0)], rc.gamma_dead)
                if hook is not None:
                    g = g + np.asarray(hook.delta(s), dtype=np.complex128)
                out[pid] = tile_sums(rc.tile_of_cell, rc.n_tiles, c=rc.C, g=g, backend=self.backend)
            else:
                if hook is not None:  # section 5.6 step 6: never drop a configured hook silently
                    raise ConfigurationError(
                        f"panel {pid}: the coupling hook needs separable coefficients (C, gamma); "
                        "this snapshot carries the B form (angle-dependent or per-carrier table)"
                    )
                assert rc.B is not None
                out[pid] = tile_sums(
                    rc.tile_of_cell, rc.n_tiles, b=rc.B, state=s, backend=self.backend
                )
        return out

    def combine(
        self, snap: Snapshot, cfg: Mapping[PanelId, PanelConfig]
    ) -> Mapping[LinkId, LinkPaths]:
        """Per-link paths with the configurations applied (linear in Gamma).

        Order: the link's non-RIS paths, then for each panel (scenario order) and each of the
        link's rows (row order) one RIS_TILE path per tile. A panel that is disabled or has
        no configuration keeps its tile paths with zero amplitude; so does its structural
        gap path (``gap_gamma``), because a disabled panel is absent as a whole (RQ-12).
        Tile keys are the flat (panel, row, tile) index ``offset_panel + r K + k``.
        """
        amps_by_panel = self.tile_amplitudes(snap, cfg)
        gap_pids = [
            pid
            for pid, st in self._panels.items()
            if st.model.gap_gamma is not None and st.gap_points.shape[0] > 0
        ]
        gap_off = [
            i
            for i, pid in enumerate(gap_pids)
            if pid not in cfg or not cfg[pid].enabled  # absent panel: no gap reflection either
        ]
        rows_of: dict[PanelId, dict[LinkId, list[int]]] = {}
        offsets: dict[PanelId, int] = {}
        offset = 0
        for pid, rc in snap.ris.items():
            per: dict[LinkId, list[int]] = {}
            for r, (rlk, _v) in enumerate(rc.rows):
                per.setdefault(rlk, []).append(r)
            rows_of[pid] = per
            offsets[pid] = offset
            offset += rc.n_rows * rc.n_tiles
        out: dict[LinkId, LinkPaths] = {}
        for lk, pb in snap.paths.items():
            base = pb.amp
            if gap_off:  # gap paths are the last len(gap_pids) entries, in panel order
                n0 = pb.n_paths - len(gap_pids)
                if n0 < 0 or np.any(pb.kind[n0:] != PathKind.RIS_TILE):
                    raise ConfigurationError(f"link {lk}: snapshot does not match this model")
                base = pb.amp.copy()
                base[[n0 + i for i in gap_off]] = 0.0
            amps = [base]
            delays = [pb.delay_s]
            kinds = [pb.kind]
            tiles = [np.full(pb.n_paths, -1, dtype=np.int32)]
            for pid, rc in snap.ris.items():
                rows = rows_of[pid].get(lk)
                if not rows:
                    continue
                k_count = rc.n_tiles
                ta = amps_by_panel[pid]
                ridx = np.asarray(rows, dtype=np.int64)
                if ta is None:
                    amps.append(np.zeros(ridx.size * k_count, dtype=np.complex128))
                else:
                    amps.append(ta[ridx].ravel())
                delays.append(rc.tile_delay_s[ridx].ravel())
                kinds.append(np.full(ridx.size * k_count, PathKind.RIS_TILE, dtype=np.int8))
                keys = offsets[pid] + ridx[:, None] * k_count + np.arange(k_count)[None, :]
                tiles.append(keys.ravel().astype(np.int32))
            out[lk] = LinkPaths(
                lk,
                snap.t,
                np.concatenate(amps),
                np.concatenate(delays),
                np.concatenate(kinds),
                np.concatenate(tiles),
            )
        return out

    def ris_delay_s(self, panel: PanelId, rx: StreamId, t: Fraction) -> float:
        """tau_RIS->rx at the panel centre, s."""
        st = self._state(panel)
        node, _ = split_stream_id(rx)
        if rx not in self._streams:
            raise UnknownTarget(f"unknown stream {rx!r}")
        p = self.node_position(node, t)
        return float(np.linalg.norm(p - st.model.grid.pose.center_m)) / C0

    def motion_bounds(self) -> MotionBounds:
        """Speeds and closest approaches over ``[0, horizon_s]`` (section 4.4 policy inputs)."""
        h = self.horizon_s
        v_node = {nid: nd.trajectory.max_speed_mps(h) for nid, nd in self._nodes.items()}
        grid_t: dict[NodeId, list[float]] = {
            nid: nd.trajectory.breakpoints(h) for nid, nd in self._nodes.items()
        }

        def times(*nids: NodeId) -> np.ndarray:
            ts = sorted(set().union(*(grid_t[x] for x in nids)) | {0.0, h})
            refined = []
            for a, b in itertools.pairwise(ts):
                refined.extend(np.linspace(a, b, 17)[:-1].tolist())
            refined.append(ts[-1])
            return np.asarray(refined)

        def pos(nid: NodeId, tt: np.ndarray) -> np.ndarray:
            traj = self._nodes[nid].trajectory
            return np.stack([traj.position(Fraction(float(x))) for x in tt])

        v_rel: dict[LinkId, float] = {}
        r_min: dict[LinkId, float] = {}
        for lk, (tx, rx) in self._links.items():
            a, b = tx.node, rx.node
            v_rel[lk] = v_node[a] + v_node[b]
            if v_rel[lk] == 0.0:
                r_min[lk] = float(
                    np.linalg.norm(self.node_position(b, 0) - self.node_position(a, 0))
                )
                continue
            tt = times(a, b)
            dp = pos(b, tt) - pos(a, tt)
            r_min[lk] = min(_closest_approach(dp[i], dp[i + 1]) for i in range(len(tt) - 1))
        linked = sorted({s.node for pair in self._links.values() for s in pair})
        p_r: dict[PanelId, float] = {}
        p_v: dict[PanelId, float] = {}
        for pid, st in self._panels.items():
            best = math.inf
            for nid in linked:
                tt = times(nid) if v_node[nid] > 0 else np.array([0.0])
                d = point_rectangle_distance(pos(nid, tt), st.model.grid)
                dt = float(np.max(np.diff(tt))) if tt.size > 1 else 0.0
                best = min(best, max(float(d.min()) - v_node[nid] * dt / 2, 0.0))
            p_r[pid] = best
            p_v[pid] = max((v_node[x] for x in linked), default=0.0)
        return MotionBounds(
            static=all(nd.trajectory.is_static for nd in self._nodes.values()),
            v_rel_max_mps=v_rel,
            v_perp_max_mps=dict(v_rel),
            r_min_m=r_min,
            panel_r_min_m=p_r,
            panel_v_rel_max_mps=p_v,
        )

    # ------------------------------------------------------------------ helpers for users
    def evaluate(self, snap: Snapshot, cfg: Mapping[PanelId, PanelConfig]) -> dict[LinkId, complex]:
        """Narrowband channel per link: ``sum_p amp_p exp(-j 2 pi f_c tau_p)``."""
        out = {}
        for lk, lp in self.combine(snap, cfg).items():
            fc = self._links[lk][0].fc_hz
            cyc = np.mod(fc * lp.delay_s, 1.0)
            out[lk] = complex(np.sum(lp.amp * np.exp(-2j * np.pi * cyc)))
        return out

    def csi_view(self, snap: Snapshot, link: LinkId) -> CsiView:
        """Oracle CSI of one link (``h_direct`` and per-panel (S, N) cascades, carrier phase
        included); use is to be flagged as oracle by the caller."""
        fc = self.link_streams(link)[0].fc_hz
        pb = snap.paths[link]
        h0 = complex(np.sum(pb.amp * np.exp(-2j * np.pi * np.mod(fc * pb.delay_s, 1.0))))
        cascade: dict[PanelId, np.ndarray] = {}
        for pid, rc in snap.ris.items():
            acc = np.zeros((rc.n_states, rc.n_cells), dtype=np.complex128)
            for r, (rlk, _v) in enumerate(rc.rows):
                if rlk != link:
                    continue
                ph = np.exp(-2j * np.pi * np.mod(fc * rc.tile_delay_s[r][rc.tile_of_cell], 1.0))
                if rc.separable:
                    assert rc.C is not None
                    assert rc.gamma is not None
                    acc += rc.gamma[:, None] * (rc.C[r] * ph)[None, :]
                else:
                    assert rc.B is not None
                    acc += rc.B[r] * ph[None, :]
            cascade[pid] = acc
        return CsiView(link, snap.t, fc, h0, cascade)

    def diagnostics(self) -> dict[str, Any]:
        """Physics diagnostics for ``sremu validate`` (section 9.5) and manifests.

        Per panel: resolved cell model, quadrature size and sub-cells, Richardson level and
        estimate (dB), and ``tau_RIS->r`` at t = 0 in samples of every RX stream. Per link:
        carrier, wavelength, ``f_D,max = v_rel,max / lambda`` (Hz) and ``r_min`` (m) over
        ``[0, horizon_s]``.
        """
        mb = self.motion_bounds()
        panels: dict[str, Any] = {}
        for pid, st in self._panels.items():
            rich = st.richardson
            panels[pid] = {
                "cell_model": st.resolved,
                "quadrature_points": st.quad.n_points,
                "subcells": [st.quad.m_u, st.quad.m_v],
                "subcell_size_m": st.quad.delta_m,
                "richardson_level": None if rich is None else rich.level,
                "richardson_db": None if rich is None else rich.estimate_db,
                "tau_ris_rx_samples": {
                    sid: self.ris_delay_s(pid, sid, Fraction(0)) * float(s.rate_sps)
                    for sid, s in self._streams.items()
                    if s.direction == "rx"
                },
            }
        links: dict[str, Any] = {}
        for lk, (tx, _rx) in self._links.items():
            lam = C0 / tx.fc_hz
            links[lk] = {
                "fc_hz": tx.fc_hz,
                "lambda_m": lam,
                "f_d_max_hz": mb.v_rel_max_mps[lk] / lam,
                "r_min_m": mb.r_min_m[lk],
            }
        return {"panels": panels, "links": links, "static": mb.static}

    # ------------------------------------------------------------------ scenario
    @classmethod
    def from_scenario(
        cls,
        scenario: Any,
        *,
        base_dir: str | Path | None = None,
        backend: Backend | None = None,
        render_tiles: Mapping[PanelId, tuple[int, int]] | None = None,
        mechanisms: Iterable[Any] = (),
        horizon_s: str | int | Fraction | None = None,
    ) -> ScenePropagation:
        """Build from a scenario (section 9.2): a pydantic model (``model_dump``), a mapping
        with the frozen field names, or the path of a YAML/JSON/TOML scenario file (relative
        files then resolve against its directory). Ground, walls, knife edges and diffuse
        fading need WP3 mechanisms passed through ``mechanisms``.
        """
        if isinstance(scenario, str | Path):
            path = Path(scenario)
            d = _read_scenario_file(path)
            base_dir = path.parent if base_dir is None else base_dir
        elif hasattr(scenario, "model_dump"):
            d = scenario.model_dump(mode="python")
        else:
            d = dict(scenario)
        bdir = None if base_dir is None else Path(base_dir)
        seed = int(d.get("seed", 0))
        num = dict(d.get("numerics") or {})
        numerics = Numerics(
            point_max_pitch_wavelengths=float(num.get("point_max_pitch_wavelengths", 0.5)),
            subcell_max_wavelengths=float(num.get("subcell_max_wavelengths", 0.2)),
            richardson_tol_db=float(num.get("richardson_tol_db", 0.05)),
        )
        be = backend or num.get("backend") or "numpy_ref"
        env = dict(d.get("environment") or {})
        mechs = tuple(mechanisms)
        needs_wp3 = [k for k in ("ground", "walls", "edges", "diffuse") if env.get(k)]
        if needs_wp3 and not mechs:
            raise ConfigurationError(
                f"environment {needs_wp3} needs the WP3 mechanisms (physics.ground/walls/"
                "diffraction/diffuse); pass them through mechanisms="
            )
        occ = occluders_from_spec(env.get("occluders"))
        nodes = []
        for nd in d.get("nodes") or []:
            nid = str(nd["id"])
            traj = trajectory_from_spec(
                nd.get("trajectory"), nd.get("position_m"), node=nid, seed=seed, base_dir=bdir
            )
            ant = antenna_from_spec(nd.get("antenna"), base_dir=bdir)
            specs, bws = [], {}
            for s in nd.get("streams") or []:
                sid = stream_id(nid, str(s["name"]))
                rate = s["rate_sps"]
                specs.append(
                    StreamSpec(
                        sid,
                        s["direction"],
                        float(s["fc_hz"]),
                        parse_exact(rate, name=f"{sid}.rate_sps"),
                    )
                )
                bw = s.get("occupied_bw_hz")
                bws[sid] = None if bw is None else float(bw)
            nodes.append(NodeModel(nid, traj, ant, tuple(specs), bws))
        panels = []
        rt = dict(render_tiles or {})
        for p in d.get("ris") or []:
            pid = str(p["id"])
            pose_d = p["pose"]
            pose = Pose(
                np.asarray(pose_d["center_m"], float),
                np.asarray(pose_d["normal"], float),
                np.asarray(pose_d["u_axis"], float),
            )
            g = p["grid"]
            tl = dict(p.get("tiles") or {})
            grid = PanelGrid(
                pose,
                int(g["n_u"]),
                int(g["n_v"]),
                float(g["pitch_u_m"]),
                float(g["pitch_v_m"]),
                int(tl.get("n_tiles_u", 1)),
                int(tl.get("n_tiles_v", 1)),
                float(tl.get("gap_m", 0.0)),
            )
            st = dict(p["states"])
            table = load_table(st, base_dir=bdir).with_efficiency(
                float(st.get("efficiency", 1.0)),
                override=bool(st.get("override_efficiency", False)),
            )
            fl = dict(p.get("failures") or {})
            fmap = FailureMap.from_rates(
                grid.n_cells,
                float(fl.get("dead_rate", 0.0)),
                float(fl.get("stuck_rate", 0.0)),
                keyed_generator(seed, "failure", pid),
                n_states=table.n_states,
            )
            ctrl = dict(p.get("control") or {})
            init = dict(p.get("initial") or {"fill": 0})
            if "states" in init and init["states"] is not None:
                ctrl_map, _ = control_map(
                    ctrl.get("granularity", "element"),
                    grid.n_u,
                    grid.n_v,
                    tiles=(grid.n_tiles_u, grid.n_tiles_v),
                    groups=ctrl.get("groups"),
                )
                init_state = np.asarray(init["states"], dtype=np.int16)[ctrl_map]
            else:
                init_state = np.full(grid.n_cells, int(init.get("fill", 0)), dtype=np.int16)
            panels.append(
                PanelModel(
                    id=pid,
                    grid=grid,
                    table=table,
                    cell_model=g.get("cell_model", "auto"),
                    gamma_dead=_complex(fl.get("gamma_dead", 0), f"{pid}.failures.gamma_dead"),
                    gap_gamma=None
                    if tl.get("gap_gamma") is None
                    else _complex(tl["gap_gamma"], f"{pid}.tiles.gap_gamma"),
                    render_tiles=tuple(rt.get(pid, (1, 1))),  # type: ignore[arg-type]
                    granularity=ctrl.get("granularity", "element"),
                    groups=ctrl.get("groups"),
                    failures=fmap,
                    initial_state=init_state,
                    enabled=bool(p.get("enabled", True)),
                    transition=dict(p.get("transition") or {"model": "step"}),
                    capabilities=dict(p.get("capabilities") or {}),
                )
            )
        run = dict(d.get("run") or {})
        hz = horizon_s if horizon_s is not None else run.get("duration_s", 10)
        return cls(
            nodes,
            panels,
            occluders=occ,
            mechanisms=mechs,
            backend=be,
            horizon_s=hz if not isinstance(hz, float) else str(hz),
            numerics=numerics,
            seed=seed,
        )


def _read_scenario_file(path: Path) -> dict[str, Any]:
    try:  # WP1 reader: YAML 1.2 core scalars, JSON and TOML
        from sremu.scenario.load import read_document
    except ImportError:
        import yaml

        out = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        out = read_document(path)
    if not isinstance(out, Mapping):
        raise ConfigurationError(f"{path}: a scenario must be a mapping")
    return dict(out)


def from_scenario(scenario: Any, **kwargs: Any) -> ScenePropagation:
    """Module-level factory (section 2.2): :meth:`ScenePropagation.from_scenario`."""
    return ScenePropagation.from_scenario(scenario, **kwargs)
