"""Analytic LOS-only ``PropagationModel`` for parallel development (DESIGN.md section 14.3).

``FakePropagation`` implements :class:`sremu.contracts.PropagationModel` for free-space
line-of-sight links between nodes that move with constant velocity:

* path amplitude ``lambda/(4*pi*d)`` (isotropic antennas, Friis-normalized, real positive;
  the propagation phase ``exp(-j*2*pi*f_c*tau)`` is excluded as the contract requires);
* absolute delay ``tau = d/C0`` with ``d = |p_rx(t) - p_tx(t)|`` evaluated in float64 at the
  exact snapshot time; positions are analytic functions of continuous time.

Optional ``FakePanel`` objects add a **simplified** point-cell RIS so runtime work packages
can exercise RIS rows, tiles and timed switching before the physics core lands. The cell
coefficient is K1 with isotropic antennas, obliquity 1 and no visibility test:
``A * j/(4*pi*r_i*r_s) * exp(-j*k*(r_i + r_s - L_tile))`` with ``A`` the cell pitch area and
``L_tile = C0*tau_tile`` the Tx -> tile reference -> Rx length. It is NOT a physics model
and must never be used for physics results.

Contract 1.1 (D11): ``link_fc_hz``, ``panel_geometry`` (D17 conventions; point cells, so
Q = N), ``ris_delay_s(..., tile=k)`` from the tile reference point (the centroid of the
tile's cells, the point used in ``tile_delay_s``), ``with_render_tiles`` (a copy with other
panel tile grids) and timed node events. ``set_node_position`` / ``set_node_trajectory``
need a node declared ``external=True`` (like the physics model's external trajectories) and
events in time order; positions are then piecewise constant-velocity: a position event holds
the position, a trajectory event follows ``start + v*(t - t_event)`` for any object with
``velocity_mps`` and ``start_m`` or ``position_m`` (``FakeNode``, or the physics ``Static``
and ``ConstantVelocity`` trajectories). Motion bounds stay exact for that piecewise motion.

Self-links (TX and RX on the same node) are not modelled.
"""

from __future__ import annotations

import bisect
import copy
import dataclasses
import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from sremu.contracts import (
    LinkId,
    LinkPaths,
    MotionBounds,
    NodeId,
    PanelConfig,
    PanelGeometry,
    PanelId,
    PathBundle,
    PathKind,
    PropagationModel,
    RISCoefficients,
    Snapshot,
    StreamId,
    StreamSpec,
    check_link,
    link_id,
    split_stream_id,
)
from sremu.errors import ConfigurationError, GeometryError, UnknownTarget
from sremu.timebase import parse_exact
from sremu.units import C0

__all__ = ["FakeNode", "FakePanel", "FakePropagation"]

Vec3 = tuple[float, float, float]


def _vec3(v: Iterable[float], name: str) -> Vec3:
    t = tuple(float(x) for x in v)
    if len(t) != 3 or not all(math.isfinite(x) for x in t):
        raise ConfigurationError(f"{name} must be 3 finite floats, got {v!r}")
    return (t[0], t[1], t[2])


@dataclass(frozen=True)
class FakeNode:
    """A node at ``position_m`` (m) at t = 0 moving with constant ``velocity_mps`` (m/s).

    ``external=True`` lets timed ``node.set_position`` / ``node.set_trajectory`` events move
    the node (``FakePropagation.set_node_position``); :meth:`position` is always the initial
    motion.
    """

    id: NodeId
    position_m: Vec3
    velocity_mps: Vec3 = (0.0, 0.0, 0.0)
    streams: tuple[StreamSpec, ...] = ()
    external: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", _vec3(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_mps", _vec3(self.velocity_mps, "velocity_mps"))
        object.__setattr__(self, "streams", tuple(self.streams))
        for s in self.streams:
            if s.node != self.id:
                raise ConfigurationError(f"stream {s.stream!r} does not belong to node {self.id!r}")

    def position(self, t: Fraction) -> np.ndarray:
        """Position (3,) float64 in m at exact time ``t`` (s): ``p0 + v*t``."""
        return np.asarray(self.position_m) + np.asarray(self.velocity_mps) * float(t)

    @property
    def moving(self) -> bool:
        """True when the velocity is non-zero."""
        return any(v != 0.0 for v in self.velocity_mps)


@dataclass(frozen=True)
class FakePanel:
    """A flat panel of ``n_u x n_v`` point cells (simplified RIS; see the module docstring).

    ``center_m``: panel centre, m. ``normal``/``u_axis``: unit normal and in-plane axis
    (``v = normal x u``). ``pitch_u_m``/``pitch_v_m``: cell pitch, m. ``gamma``: separable
    state alphabet. ``tiles_u``/``tiles_v``: tile grid. Cell index ``n = iv*n_u + iu``.
    """

    id: PanelId
    center_m: Vec3
    normal: Vec3
    u_axis: Vec3
    n_u: int
    n_v: int
    pitch_u_m: float
    pitch_v_m: float
    gamma: tuple[complex, ...] = (1 + 0j, -1 + 0j)
    tiles_u: int = 1
    tiles_v: int = 1
    gamma_dead: complex = 0j
    _cells: np.ndarray = field(init=False, repr=False, compare=False)
    _tile_of_cell: np.ndarray = field(init=False, repr=False, compare=False)
    _tile_ref: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        c = np.asarray(_vec3(self.center_m, "center_m"))
        n = np.asarray(_vec3(self.normal, "normal"))
        u = np.asarray(_vec3(self.u_axis, "u_axis"))
        if abs(np.linalg.norm(n) - 1) > 1e-12 or abs(np.linalg.norm(u) - 1) > 1e-12:
            raise ConfigurationError("normal and u_axis must be unit vectors")
        if abs(float(n @ u)) > 1e-12:
            raise ConfigurationError("u_axis must be orthogonal to normal")
        if self.n_u < 1 or self.n_v < 1 or self.tiles_u < 1 or self.tiles_v < 1:
            raise ConfigurationError("panel and tile grid sizes must be >= 1")
        if self.tiles_u > self.n_u or self.tiles_v > self.n_v:
            raise ConfigurationError("more tiles than cells along an axis")
        v = np.cross(n, u)
        iu = np.arange(self.n_u)
        iv = np.arange(self.n_v)
        gu, gv = np.meshgrid(iu, iv)  # shape (n_v, n_u): row iv, column iu
        ou = (gu.ravel() - (self.n_u - 1) / 2) * self.pitch_u_m
        ov = (gv.ravel() - (self.n_v - 1) / 2) * self.pitch_v_m
        cells = c[None, :] + ou[:, None] * u[None, :] + ov[:, None] * v[None, :]
        tu = (gu.ravel() * self.tiles_u) // self.n_u
        tv = (gv.ravel() * self.tiles_v) // self.n_v
        tile = (tv * self.tiles_u + tu).astype(np.int32)
        k = self.tiles_u * self.tiles_v
        ref = np.stack([cells[tile == j].mean(axis=0) for j in range(k)])
        object.__setattr__(self, "gamma", tuple(complex(g) for g in self.gamma))
        object.__setattr__(self, "_cells", cells)
        object.__setattr__(self, "_tile_of_cell", tile)
        object.__setattr__(self, "_tile_ref", ref)

    @property
    def n_cells(self) -> int:
        """Number of physical cells N."""
        return self.n_u * self.n_v

    @property
    def n_tiles(self) -> int:
        """Number of tiles K."""
        return self.tiles_u * self.tiles_v

    def cell_centres(self) -> np.ndarray:
        """(N, 3) float64 cell centres, m."""
        return self._cells.copy()

    def tile_of_cell(self) -> np.ndarray:
        """(N,) int32 tile index of each cell."""
        return self._tile_of_cell.copy()

    def tile_reference_points(self) -> np.ndarray:
        """(K, 3) float64 tile reference points (centroids of the tiles' cells), m."""
        return self._tile_ref.copy()


def _min_distance(dp: np.ndarray, dv: np.ndarray, horizon_s: float) -> float:
    """min over t in [0, horizon] of |dp + dv*t| (closest approach, m)."""
    vv = float(dv @ dv)
    t = 0.0 if vv == 0.0 else min(max(-float(dp @ dv) / vv, 0.0), horizon_s)
    return float(np.linalg.norm(dp + dv * t))


@dataclass(frozen=True, eq=False)
class _Segment:
    """Constant-velocity motion from exact time ``t0`` on: ``p0 + v*(t - t0)``."""

    t0: Fraction
    p0: np.ndarray
    v: np.ndarray

    def position(self, t: Fraction) -> np.ndarray:
        return self.p0 + self.v * float(t - self.t0)


def _segment_at(segs: Sequence[_Segment], t: Fraction) -> _Segment:
    i = bisect.bisect_right([sg.t0 for sg in segs], t) - 1
    return segs[max(i, 0)]


def _constant_velocity(traj: Any) -> tuple[np.ndarray, np.ndarray]:
    """(start (3,), velocity (3,)) of a constant-velocity trajectory object."""
    start = getattr(traj, "start_m", None)
    if start is None:
        start = getattr(traj, "position_m", None)
    if start is None:
        raise TypeError(
            "FakePropagation supports constant-velocity trajectories only: objects with "
            "'start_m' or 'position_m' and optional 'velocity_mps'"
        )
    p0 = np.asarray(_vec3(start, "trajectory start"))
    v = np.asarray(_vec3(getattr(traj, "velocity_mps", (0.0, 0.0, 0.0)), "velocity_mps"))
    return p0, v


class FakePropagation:
    """LOS-only analytic ``PropagationModel`` with constant-velocity nodes (WP0 fake).

    ``nodes`` define positions, velocities and streams; links follow the contract rule
    (equal rates and ``|fc_tx - fc_rx| < f_s/2``) in scenario order (TX streams in node
    order, then RX streams in node order). ``panels`` add simplified RIS rows (one row per
    link, hop variant 0). ``horizon_s`` bounds the run for :meth:`motion_bounds`.
    """

    def __init__(
        self,
        nodes: Sequence[FakeNode],
        panels: Sequence[FakePanel] = (),
        *,
        horizon_s: str | int | Fraction = 10,
    ) -> None:
        self._nodes: dict[NodeId, FakeNode] = {}
        for nd in nodes:
            if nd.id in self._nodes:
                raise ConfigurationError(f"duplicate node id {nd.id!r}")
            self._nodes[nd.id] = nd
        self._panels: dict[PanelId, FakePanel] = {}
        for p in panels:
            if p.id in self._panels:
                raise ConfigurationError(f"duplicate panel id {p.id!r}")
            self._panels[p.id] = p
        self._streams: dict[StreamId, StreamSpec] = {}
        for nd in nodes:
            for s in nd.streams:
                if s.stream in self._streams:
                    raise ConfigurationError(f"duplicate stream id {s.stream!r}")
                self._streams[s.stream] = s
        self._horizon_q = parse_exact(horizon_s, name="horizon_s")
        self._horizon_s = float(self._horizon_q)
        self._segments: dict[NodeId, list[_Segment]] = {
            nd.id: [_Segment(Fraction(0), np.asarray(nd.position_m), np.asarray(nd.velocity_mps))]
            for nd in nodes
        }
        txs = [s for s in self._streams.values() if s.direction == "tx"]
        rxs = [s for s in self._streams.values() if s.direction == "rx"]
        self._links: dict[LinkId, tuple[StreamSpec, StreamSpec]] = {}
        for tx in txs:
            for rx in rxs:
                if tx.node == rx.node:
                    continue
                if check_link(tx.fc_hz, tx.rate_sps, rx.fc_hz, rx.rate_sps):
                    self._links[link_id(tx.stream, rx.stream)] = (tx, rx)

    # ------------------------------------------------------------------ helpers
    @property
    def nodes(self) -> Mapping[NodeId, FakeNode]:
        """Nodes by id."""
        return dict(self._nodes)

    @property
    def panels(self) -> Mapping[PanelId, FakePanel]:
        """Panels by id."""
        return dict(self._panels)

    @property
    def streams(self) -> Mapping[StreamId, StreamSpec]:
        """Stream specs by id."""
        return dict(self._streams)

    def link_streams(self, link: LinkId) -> tuple[StreamSpec, StreamSpec]:
        """(tx, rx) stream specs of a link."""
        try:
            return self._links[link]
        except KeyError:
            raise UnknownTarget(f"unknown link {link!r}") from None

    def node_position(self, node: NodeId, t: Fraction | int | str) -> np.ndarray:
        """Position (3,) of ``node`` at exact time ``t`` (s), m, timed node events included."""
        if node not in self._segments:
            raise UnknownTarget(f"unknown node {node!r}")
        tq = parse_exact(t, name="t")
        return _segment_at(self._segments[node], tq).position(tq)

    def los_distance_m(self, link: LinkId, t: Fraction) -> float:
        """LOS distance of ``link`` at time ``t``, m."""
        tx, rx = self._links[link]
        d = float(np.linalg.norm(self.node_position(rx.node, t) - self.node_position(tx.node, t)))
        if not d > 0.0:
            raise GeometryError(f"link {link}: TX and RX are co-located at t = {t}")
        return d

    # ------------------------------------------------------------------ contract
    def links(self) -> tuple[LinkId, ...]:
        """All links in scenario order."""
        return tuple(self._links)

    def link_fc_hz(self, link: LinkId) -> float:
        """Carrier of the link's TX stream, Hz."""
        return float(self.link_streams(link)[0].fc_hz)

    def _panel(self, panel: PanelId) -> FakePanel:
        try:
            return self._panels[panel]
        except KeyError:
            raise UnknownTarget(f"unknown panel {panel!r}") from None

    def panel_geometry(self, panel: PanelId) -> PanelGeometry:
        """D17 geometry of a panel (point cells: Q = N; no physical tiles or gaps)."""
        p = self._panel(panel)
        return PanelGeometry.from_cells(
            p.id, p.center_m, p.normal, p.u_axis, p.n_u, p.n_v, p.pitch_u_m, p.pitch_v_m
        )

    def with_render_tiles(self, tiles: Mapping[PanelId, tuple[int, int]]) -> FakePropagation:
        """A copy whose listed panels use render tiles ``(tiles_u, tiles_v)``; the receiver is
        unchanged. The copy shares the node event history (as the physics model does)."""
        new = copy.copy(self)
        new._panels = dict(self._panels)
        for pid, (ku, kv) in tiles.items():
            new._panels[pid] = dataclasses.replace(
                self._panel(pid), tiles_u=int(ku), tiles_v=int(kv)
            )
        return new

    def _event(self, node: NodeId, t: Fraction | int | str, p0: np.ndarray, v: np.ndarray) -> None:
        if node not in self._nodes:
            raise UnknownTarget(f"unknown node {node!r}")
        if not self._nodes[node].external:
            raise ConfigurationError(f"node {node!r} does not have an external trajectory")
        tq = parse_exact(t, name="event time")
        segs = self._segments[node]
        if tq < segs[-1].t0:
            raise ConfigurationError("external trajectory events must be in time order")
        segs.append(_Segment(tq, p0, v))

    def set_node_position(
        self, node: NodeId, t: Fraction | int | str, position_m: Sequence[float]
    ) -> None:
        """Timed ``node.set_position``: hold ``position_m`` (m) from exact time ``t`` (s) on."""
        self._event(node, t, np.asarray(_vec3(position_m, "position_m")), np.zeros(3))

    def set_node_trajectory(self, node: NodeId, t: Fraction | int | str, traj: Any) -> None:
        """Timed ``node.set_trajectory``: from ``t`` on, ``start + v*(t' - t)`` (constant
        velocity; see the module docstring for the accepted objects)."""
        p0, v = _constant_velocity(traj)
        self._event(node, t, p0, v)

    def snapshot(self, t: Fraction) -> Snapshot:
        """LOS paths of every link and simplified RIS coefficients at time ``t`` (s)."""
        tq = parse_exact(t, name="t")
        paths: dict[LinkId, PathBundle] = {}
        for lk, (tx, _rx) in self._links.items():
            d = self.los_distance_m(lk, tq)
            lam = C0 / tx.fc_hz
            paths[lk] = PathBundle(
                link=lk,
                t=tq,
                amp=np.array([lam / (4.0 * math.pi * d)], dtype=np.complex128),
                delay_s=np.array([d / C0], dtype=np.float64),
                kind=np.array([PathKind.LOS], dtype=np.int8),
            )
        ris = {pid: self._panel_coefficients(p, tq) for pid, p in self._panels.items()}
        return Snapshot(t=tq, paths=paths, ris=ris, diag={})

    def _panel_coefficients(self, p: FakePanel, t: Fraction) -> RISCoefficients:
        links = list(self._links)
        r_count, n, k_count = len(links), p.n_cells, p.n_tiles
        cells = p._cells
        tile = p._tile_of_cell
        ref = p._tile_ref
        area = p.pitch_u_m * p.pitch_v_m
        c = np.zeros((r_count, n), dtype=np.complex128)
        tile_delay = np.zeros((r_count, k_count), dtype=np.float64)
        for r, lk in enumerate(links):
            tx, rx = self._links[lk]
            ptx = self.node_position(tx.node, t)
            prx = self.node_position(rx.node, t)
            ri = np.linalg.norm(cells - ptx[None, :], axis=1)
            rs = np.linalg.norm(cells - prx[None, :], axis=1)
            l_ref = np.linalg.norm(ref - ptx[None, :], axis=1) + np.linalg.norm(
                ref - prx[None, :], axis=1
            )
            tile_delay[r] = l_ref / C0
            k = 2.0 * math.pi * tx.fc_hz / C0
            c[r] = (
                area * 1j / (4.0 * math.pi * ri * rs) * np.exp(-1j * k * ((ri + rs) - l_ref[tile]))
            )
        return RISCoefficients(
            panel=p.id,
            t=t,
            rows=tuple((lk, 0) for lk in links),
            tile_of_cell=tile.copy(),
            tile_delay_s=tile_delay,
            B=None,
            C=c,
            gamma=np.asarray(p.gamma, dtype=np.complex128),
            gamma_dead=p.gamma_dead,
        )

    def combine(
        self, snap: Snapshot, cfg: Mapping[PanelId, PanelConfig]
    ) -> Mapping[LinkId, LinkPaths]:
        """Paths per link with configurations applied (linear in Gamma).

        The path set is fixed: LOS first, then one RIS_TILE path per (panel, row, tile),
        panels in scenario order. A disabled panel (or a missing configuration) keeps its
        paths with zero amplitude, so path arrays stay aligned across configurations.
        """
        out: dict[LinkId, LinkPaths] = {}
        row_of: dict[PanelId, dict[LinkId, int]] = {
            pid: {lk: r for r, (lk, _v) in enumerate(rc.rows)} for pid, rc in snap.ris.items()
        }
        for lk, pb in snap.paths.items():
            amps = [pb.amp]
            delays = [pb.delay_s]
            kinds = [pb.kind]
            tiles = [np.full(pb.n_paths, -1, dtype=np.int32)]
            offset = 0
            for pid, rc in snap.ris.items():
                k_count = rc.n_tiles
                r = row_of[pid][lk]
                conf = cfg.get(pid)
                tile_amp = np.zeros(k_count, dtype=np.complex128)
                if conf is not None and conf.enabled:
                    if conf.state.shape != (rc.n_cells,):
                        raise ConfigurationError(
                            f"panel {pid}: state shape {conf.state.shape} != ({rc.n_cells},)"
                        )
                    assert rc.C is not None
                    assert rc.gamma is not None
                    s = conf.state.astype(np.int64)
                    if s.size and int(s.max()) >= rc.n_states:
                        raise ConfigurationError(f"panel {pid}: state index out of range")
                    g = np.where(s >= 0, rc.gamma[np.clip(s, 0, None)], rc.gamma_dead)
                    np.add.at(tile_amp, rc.tile_of_cell, rc.C[r] * g)
                amps.append(tile_amp)
                delays.append(rc.tile_delay_s[r].copy())
                kinds.append(np.full(k_count, PathKind.RIS_TILE, dtype=np.int8))
                tiles.append((offset + r * k_count + np.arange(k_count)).astype(np.int32))
                offset += rc.n_rows * k_count
            out[lk] = LinkPaths(
                link=lk,
                t=snap.t,
                amp=np.concatenate(amps),
                delay_s=np.concatenate(delays),
                kind=np.concatenate(kinds),
                tile=np.concatenate(tiles),
            )
        return out

    def ris_delay_s(
        self, panel: PanelId, rx: StreamId, t: Fraction, tile: int | None = None
    ) -> float:
        """tau_RIS->rx in s from the panel centre, or from tile ``tile``'s reference point
        (the tile centroid used in ``tile_delay_s``)."""
        p = self._panel(panel)
        node, _ = split_stream_id(rx)
        if node not in self._nodes:
            raise UnknownTarget(f"unknown stream {rx!r}")
        pos = self.node_position(node, parse_exact(t))
        if tile is None:
            return float(np.linalg.norm(pos - np.asarray(p.center_m))) / C0
        if isinstance(tile, bool) or int(tile) != tile or not 0 <= int(tile) < p.n_tiles:
            raise ConfigurationError(f"panel {panel!r}: tile {tile!r} outside [0, {p.n_tiles})")
        # the same expression as the tile reference delay of _panel_coefficients
        return float(np.linalg.norm(p._tile_ref - pos[None, :], axis=1)[int(tile)]) / C0

    def _breaks(self, *nodes: NodeId) -> list[Fraction]:
        h = self._horizon_q
        pts = {Fraction(0), h}
        for nd in nodes:
            pts.update(sg.t0 for sg in self._segments[nd] if 0 < sg.t0 < h)
        return sorted(pts)

    def _closest(self, a: NodeId | None, b: NodeId, point: np.ndarray | None = None) -> float:
        """Exact closest approach over [0, horizon] of node ``b`` to node ``a`` (or to the
        fixed ``point`` when ``a`` is None) for piecewise constant-velocity motion, m."""
        ts = self._breaks(b) if a is None else self._breaks(a, b)
        best = math.inf
        for s0, s1 in itertools.pairwise(ts):
            sb = _segment_at(self._segments[b], s0)
            if a is None:
                assert point is not None
                dp, dv = sb.position(s0) - point, sb.v
            else:
                sa = _segment_at(self._segments[a], s0)
                dp, dv = sb.position(s0) - sa.position(s0), sb.v - sa.v
            best = min(best, _min_distance(dp, dv, float(s1 - s0)))
        return best

    def _max_speed(self, node: NodeId, other: NodeId | None = None) -> float:
        """Largest |v_node - v_other| (or |v_node|) over [0, horizon], m/s."""
        ts = self._breaks(node) if other is None else self._breaks(node, other)
        best = 0.0
        for s0 in ts[:-1]:
            v = _segment_at(self._segments[node], s0).v
            if other is not None:
                v = v - _segment_at(self._segments[other], s0).v
            best = max(best, float(np.linalg.norm(v)))
        return best

    def motion_bounds(self) -> MotionBounds:
        """Exact bounds over [0, horizon_s] for piecewise constant-velocity motion (v_perp is
        bounded by v_rel). ``static`` is True when no node ever has a non-zero velocity
        (position events are discrete events, handled by event snapshots)."""
        v_rel: dict[LinkId, float] = {}
        r_min: dict[LinkId, float] = {}
        for lk, (tx, rx) in self._links.items():
            v_rel[lk] = self._max_speed(rx.node, tx.node)
            r_min[lk] = self._closest(tx.node, rx.node)
        p_r: dict[PanelId, float] = {}
        p_v: dict[PanelId, float] = {}
        linked = sorted({s.node for pair in self._links.values() for s in pair})
        for pid, p in self._panels.items():
            c = np.asarray(p.center_m)
            dists = [self._closest(None, nd, c) for nd in linked]
            speeds = [self._max_speed(nd) for nd in linked]
            p_r[pid] = min(dists) if dists else math.inf
            p_v[pid] = max(speeds) if speeds else 0.0
        moving = any(bool(np.any(sg.v != 0.0)) for segs in self._segments.values() for sg in segs)
        return MotionBounds(
            static=not moving,
            v_rel_max_mps=v_rel,
            v_perp_max_mps=dict(v_rel),
            r_min_m=r_min,
            panel_r_min_m=p_r,
            panel_v_rel_max_mps=p_v,
        )


def _conforms(model: FakePropagation) -> PropagationModel:
    """Static check (mypy): ``FakePropagation`` structurally satisfies the 1.1 protocol."""
    return model
