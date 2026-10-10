"""Geometry: poses, RIS panel grids, angles, visibility, segment-polygon test, trajectories.

DESIGN.md sections 3.1, 5.3, 5.4 and 5.10 (WP2).

Conventions (section 3.1, normative):

* Right-handed frame, metres, z up. A panel pose is (centre ``c``, unit normal ``n``, unit
  in-plane axis ``u``) with ``v = n x u``.
* Table angles ``theta`` lie in [-90 deg, 90 deg]: measured from ``n`` in the (u, n) plane and
  signed by the ``u`` component. ``theta_i`` points from the cell toward the Tx, ``theta_s``
  from the cell toward the Rx; specular reflection is ``theta_s = -theta_i``. Off-plane
  directions are projected onto the (u, n) plane; the off-plane angle is reported.
* Cell index of an ``n_u x n_v`` panel: ``n = iv * n_u + iu`` (row ``iv``, column ``iu``).

Trajectories are analytic functions of continuous time (section 5.10): positions never
depend on any update grid. Times may be exact ``Fraction`` (preferred) or floats.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sremu import rng as _rng
from sremu.errors import ConfigurationError, GeometryError
from sremu.timebase import parse_exact, to_exact

__all__ = [
    "ConstantVelocity",
    "ExternalTrajectory",
    "PanelGrid",
    "Pose",
    "RandomWaypoint1D",
    "RandomWaypoint2D",
    "Static",
    "TraceTrajectory",
    "Trajectory",
    "Waypoints",
    "angles_to_node",
    "keyed_generator",
    "point_rectangle_distance",
    "segment_hits_polygon",
    "segments_hit_polygon",
    "trajectory_from_spec",
    "unit",
    "vec3",
]

_ORTHO_TOL = 1e-9


def vec3(v: Iterable[float] | np.ndarray, name: str = "vector") -> np.ndarray:
    """A finite (3,) float64 vector (raises ``ConfigurationError`` otherwise)."""
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ConfigurationError(f"{name} must be 3 finite numbers, got {v!r}")
    return a


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize vectors along ``axis`` (zero vectors raise ``GeometryError``)."""
    a = np.asarray(v, dtype=np.float64)
    n = np.sqrt(np.sum(a * a, axis=axis, keepdims=True))
    if np.any(n == 0.0):
        raise GeometryError("cannot normalize a zero-length vector")
    return a / n


def keyed_generator(seed: int, purpose: str, name: str, counter: int = 0) -> np.random.Generator:
    """Keyed Philox generator of section 6.1 (no global RNG): ``sremu.rng.generator``.

    ``purpose`` must be registered in ``sremu.rng.PURPOSES`` (``mobility``, ``failure``,
    ``trial``, ...); ``counter`` selects an independent stream (second 64-bit counter word),
    so e.g. random-waypoint leg ``i`` is a pure function of ``(seed, purpose, name, i)``.
    """
    return _rng.generator(int(seed), purpose, name, counter=int(counter))


# --------------------------------------------------------------------------- pose


@dataclass(frozen=True, eq=False)
class Pose:
    """Panel pose: centre ``center_m`` (m), unit ``normal`` and unit in-plane ``u_axis``."""

    center_m: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    _v: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        c = vec3(self.center_m, "center_m")
        n = vec3(self.normal, "normal")
        u = vec3(self.u_axis, "u_axis")
        nn, nu = float(np.linalg.norm(n)), float(np.linalg.norm(u))
        if nn == 0.0 or nu == 0.0:
            raise GeometryError("normal and u_axis must be non-zero")
        n = n / nn
        u = u / nu
        if abs(float(n @ u)) > _ORTHO_TOL:
            raise GeometryError(f"u_axis must be orthogonal to normal (n.u = {float(n @ u):.3g})")
        object.__setattr__(self, "center_m", c)
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "u_axis", u)
        object.__setattr__(self, "_v", np.cross(n, u))

    @property
    def v_axis(self) -> np.ndarray:
        """``v = n x u``."""
        return self._v.copy()

    def to_world(self, u: np.ndarray, v: np.ndarray, w: np.ndarray | float = 0.0) -> np.ndarray:
        """World points (..., 3) from local coordinates along (u, v, n), m."""
        u = np.asarray(u, dtype=np.float64)[..., None]
        v = np.asarray(v, dtype=np.float64)[..., None]
        w = np.asarray(w, dtype=np.float64)[..., None]
        return self.center_m + u * self.u_axis + v * self.v_axis + w * self.normal

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Local coordinates (..., 3) = (u, v, n) of world points, m."""
        d = np.asarray(points, dtype=np.float64) - self.center_m
        return np.stack([d @ self.u_axis, d @ self._v, d @ self.normal], axis=-1)


def angles_to_node(points: np.ndarray, node: np.ndarray, pose: Pose) -> dict[str, np.ndarray]:
    """Distances and table angles from panel points toward a node (section 3.1).

    Returns ``r`` (Q,) m, ``cos`` (Q,) = cosine of the angle between ``n`` and the direction
    point -> node (the visibility/obliquity cosine), ``theta_rad`` (Q,) signed in-plane angle
    ``atan2(d.u, d.n)`` (the off-plane component is projected out) and ``off_plane_rad`` (Q,)
    ``asin(|d.v|/r)``. Distances use ``sqrt((dx*dx + dy*dy) + dz*dz)`` so that the numpy
    and numba kernels round identically.
    """
    p = np.asarray(points, dtype=np.float64)
    d = np.asarray(node, dtype=np.float64)[None, :] - p
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    r = np.sqrt((dx * dx + dy * dy) + dz * dz)
    if np.any(r == 0.0):
        raise GeometryError("a node coincides with a panel point")
    dn = d @ pose.normal
    du = d @ pose.u_axis
    dv = d @ pose.v_axis
    return {
        "r": r,
        "cos": dn / r,
        "theta_rad": np.arctan2(du, dn),
        "off_plane_rad": np.arcsin(np.clip(np.abs(dv) / r, 0.0, 1.0)),
    }


# --------------------------------------------------------------------------- panel grid


@dataclass(frozen=True, eq=False)
class PanelGrid:
    """Physical cell layout of a planar RIS panel (sections 5.3, 9.2).

    ``n_u x n_v`` cells of pitch ``(pitch_u_m, pitch_v_m)``; the cell area is the pitch area.
    Optional physical tiles ``n_tiles_u x n_tiles_v`` (each must divide the cell count) are
    separated by explicit gaps of ``gap_m``: the gaps carry no cells (inert by default).
    """

    pose: Pose
    n_u: int
    n_v: int
    pitch_u_m: float
    pitch_v_m: float
    n_tiles_u: int = 1
    n_tiles_v: int = 1
    gap_m: float = 0.0
    _centres: np.ndarray = field(init=False, repr=False)
    _local: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("n_u", "n_v", "n_tiles_u", "n_tiles_v"):
            if int(getattr(self, name)) < 1:
                raise ConfigurationError(f"PanelGrid.{name} must be >= 1")
        if not (self.pitch_u_m > 0.0 and self.pitch_v_m > 0.0):
            raise ConfigurationError("cell pitches must be > 0 m")
        if self.gap_m < 0.0:
            raise ConfigurationError("gap_m must be >= 0")
        if self.n_u % self.n_tiles_u or self.n_v % self.n_tiles_v:
            raise ConfigurationError("physical tiles must divide the cell grid evenly")
        iu = np.arange(self.n_u)
        iv = np.arange(self.n_v)
        tu = iu // (self.n_u // self.n_tiles_u)
        tv = iv // (self.n_v // self.n_tiles_v)
        width_u = self.n_u * self.pitch_u_m + (self.n_tiles_u - 1) * self.gap_m
        width_v = self.n_v * self.pitch_v_m + (self.n_tiles_v - 1) * self.gap_m
        ou = (iu + 0.5) * self.pitch_u_m + tu * self.gap_m - width_u / 2.0
        ov = (iv + 0.5) * self.pitch_v_m + tv * self.gap_m - width_v / 2.0
        gv, gu = np.meshgrid(ov, ou, indexing="ij")  # (n_v, n_u): n = iv*n_u + iu
        local = np.stack([gu.ravel(), gv.ravel()], axis=1)
        object.__setattr__(self, "_local", local)
        object.__setattr__(self, "_centres", self.pose.to_world(local[:, 0], local[:, 1]))

    @property
    def n_cells(self) -> int:
        """Number of physical cells N."""
        return self.n_u * self.n_v

    @property
    def cell_area_m2(self) -> float:
        """Pitch area of one cell, m^2."""
        return self.pitch_u_m * self.pitch_v_m

    @property
    def size_m(self) -> tuple[float, float]:
        """Overall panel extent (u, v) including gaps, m."""
        return (
            self.n_u * self.pitch_u_m + (self.n_tiles_u - 1) * self.gap_m,
            self.n_v * self.pitch_v_m + (self.n_tiles_v - 1) * self.gap_m,
        )

    @property
    def diagonal_m(self) -> float:
        """Panel diagonal D (m), used for the Fraunhofer distance 2 D^2 / lambda."""
        su, sv = self.size_m
        return math.hypot(su, sv)

    def cell_centres(self) -> np.ndarray:
        """(N, 3) float64 cell centres, m (copy)."""
        return self._centres.copy()

    def cell_local(self) -> np.ndarray:
        """(N, 2) float64 cell centres in panel (u, v) coordinates, m (copy)."""
        return self._local.copy()

    def cell_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """(iu, iv) int arrays of every cell, in cell order."""
        n = np.arange(self.n_cells)
        return n % self.n_u, n // self.n_u

    def physical_tile_of_cell(self) -> np.ndarray:
        """(N,) int32 physical tile index ``tv * n_tiles_u + tu``."""
        iu, iv = self.cell_indices()
        tu = iu // (self.n_u // self.n_tiles_u)
        tv = iv // (self.n_v // self.n_tiles_v)
        return (tv * self.n_tiles_u + tu).astype(np.int32)

    def render_tile_of_cell(self, tiles_u: int = 1, tiles_v: int = 1) -> np.ndarray:
        """(N,) int32 render (Doppler) tile ``tv * tiles_u + tu``, ``tu = iu*tiles_u // n_u``."""
        if not (1 <= tiles_u <= self.n_u and 1 <= tiles_v <= self.n_v):
            raise ConfigurationError(
                f"render tiles ({tiles_u}, {tiles_v}) must lie in [1, ({self.n_u}, {self.n_v})]"
            )
        iu, iv = self.cell_indices()
        tu = (iu * tiles_u) // self.n_u
        tv = (iv * tiles_v) // self.n_v
        return (tv * tiles_u + tu).astype(np.int32)

    def gap_strips(self) -> list[tuple[float, float, float, float]]:
        """Gap regions as local rectangles ``(u0, u1, v0, v1)`` (m), non-overlapping."""
        if self.gap_m == 0.0 or (self.n_tiles_u == 1 and self.n_tiles_v == 1):
            return []
        wu, wv = self.size_m
        tile_u = (self.n_u // self.n_tiles_u) * self.pitch_u_m
        tile_v = (self.n_v // self.n_tiles_v) * self.pitch_v_m
        out: list[tuple[float, float, float, float]] = []
        for k in range(1, self.n_tiles_u):  # full-height strips between tile columns
            u0 = -wu / 2 + k * tile_u + (k - 1) * self.gap_m
            out.append((u0, u0 + self.gap_m, -wv / 2, wv / 2))
        for k in range(1, self.n_tiles_v):  # horizontal strips; crossings belong to the above
            v0 = -wv / 2 + k * tile_v + (k - 1) * self.gap_m
            for j in range(self.n_tiles_u):
                u0 = -wu / 2 + j * (tile_u + self.gap_m)
                out.append((u0, u0 + tile_u, v0, v0 + self.gap_m))
        return out

    def contains_local(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Whether local (u, v) points lie on the panel's outer rectangle."""
        wu, wv = self.size_m
        return (np.abs(u) <= wu / 2) & (np.abs(v) <= wv / 2)


def point_rectangle_distance(points: np.ndarray, grid: PanelGrid) -> np.ndarray:
    """Distance (m) from points (..., 3) to the panel's outer rectangle."""
    loc = grid.pose.to_local(points)
    wu, wv = grid.size_m
    du = np.maximum(np.abs(loc[..., 0]) - wu / 2, 0.0)
    dv = np.maximum(np.abs(loc[..., 1]) - wv / 2, 0.0)
    return np.sqrt(du * du + dv * dv + loc[..., 2] ** 2)


# --------------------------------------------------------------------------- segment-polygon


def _polygon_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
        raise ConfigurationError("a polygon needs at least 3 vertices of 3 coordinates")
    # Newell normal (robust for planar polygons in any orientation)
    nx = ny = nz = 0.0
    m = v.shape[0]
    for i in range(m):
        a, b = v[i], v[(i + 1) % m]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    n = np.array([nx, ny, nz])
    nn = float(np.linalg.norm(n))
    if nn == 0.0:
        raise ConfigurationError("degenerate polygon (zero area)")
    n = n / nn
    origin = v.mean(axis=0)
    e1 = v[1] - v[0]
    e1 = e1 - (e1 @ n) * n
    e1 = e1 / float(np.linalg.norm(e1))
    e2 = np.cross(n, e1)
    if np.max(np.abs((v - origin) @ n)) > 1e-9 * max(1.0, float(np.ptp(v))):
        raise ConfigurationError("polygon vertices are not coplanar")
    return origin, n, e1, e2


def _point_in_polygon_2d(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Crossing-number test (points on the boundary count as inside)."""
    inside = np.zeros(px.shape, dtype=bool)
    on_edge = np.zeros(px.shape, dtype=bool)
    m = poly.shape[0]
    for i in range(m):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % m]
        # boundary: collinear and within the segment's bounding box
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        scale = max(abs(x2 - x1), abs(y2 - y1), 1e-300)
        on_seg = (
            (np.abs(cross) <= 1e-12 * scale * scale)
            & (px >= min(x1, x2) - 1e-12)
            & (px <= max(x1, x2) + 1e-12)
            & (py >= min(y1, y2) - 1e-12)
            & (py <= max(y1, y2) + 1e-12)
        )
        on_edge |= on_seg
        cond = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (px < xint)
    return inside | on_edge


def segments_hit_polygon(a: np.ndarray, b: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Whether open segments a -> b (..., 3) cross a planar polygon (section 5.9).

    A segment hits the polygon when it crosses the polygon's plane strictly between its
    endpoints (parameter in (0, 1)) at a point inside or on the polygon boundary. Segments
    lying in the plane, or touching it only at an endpoint, do not hit.
    """
    origin, n, e1, e2 = _polygon_frame(vertices)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    da = (a - origin) @ n
    db = (b - origin) @ n
    denom = da - db
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(denom != 0.0, da / denom, np.nan)
    crosses = (np.sign(da) * np.sign(db) < 0) & (s > 0.0) & (s < 1.0)
    p = a + s[..., None] * (b - a)
    rel = p - origin
    px, py = rel @ e1, rel @ e2
    poly = np.stack([(np.asarray(vertices) - origin) @ e1, (np.asarray(vertices) - origin) @ e2], 1)
    inside = _point_in_polygon_2d(np.nan_to_num(px), np.nan_to_num(py), poly)
    return crosses & inside


def segment_hits_polygon(a: Sequence[float], b: Sequence[float], vertices: np.ndarray) -> bool:
    """Scalar form of :func:`segments_hit_polygon`."""
    return bool(segments_hit_polygon(np.asarray(a)[None], np.asarray(b)[None], vertices)[0])


# --------------------------------------------------------------------------- trajectories

TimeLike = Fraction | int | float


def _as_float_time(t: TimeLike) -> float:
    return float(t)


@runtime_checkable
class Trajectory(Protocol):
    """An analytic position function of continuous time (section 5.10)."""

    def position(self, t: TimeLike) -> np.ndarray:
        """Position (3,) float64 at time ``t`` (s), m."""
        ...

    def positions(self, t: np.ndarray) -> np.ndarray:
        """Positions (n, 3) at float times ``t`` (s), m."""
        ...

    def max_speed_mps(self, horizon_s: float) -> float:
        """Upper bound of the speed over [0, horizon_s], m/s."""
        ...

    def breakpoints(self, horizon_s: float) -> list[float]:
        """Times in [0, horizon_s] where the velocity may be discontinuous, s."""
        ...

    @property
    def is_static(self) -> bool:
        """True when the position never changes."""
        ...


class _VectorizedMixin:
    def positions(self, t: np.ndarray) -> np.ndarray:
        return np.stack([self.position(float(x)) for x in np.asarray(t, dtype=np.float64)])  # type: ignore[attr-defined]


@dataclass(frozen=True, eq=False)
class Static(_VectorizedMixin):
    """Fixed position ``position_m``."""

    position_m: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", vec3(self.position_m, "position_m"))

    def position(self, t: TimeLike) -> np.ndarray:
        return self.position_m.copy()

    def positions(self, t: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.position_m, (np.asarray(t).size, 3)).copy()

    def max_speed_mps(self, horizon_s: float) -> float:
        return 0.0

    def breakpoints(self, horizon_s: float) -> list[float]:
        return [0.0, float(horizon_s)]

    @property
    def is_static(self) -> bool:
        return True


@dataclass(frozen=True, eq=False)
class ConstantVelocity(_VectorizedMixin):
    """``p(t) = start_m + velocity_mps * t``."""

    start_m: np.ndarray
    velocity_mps: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_m", vec3(self.start_m, "start_m"))
        object.__setattr__(self, "velocity_mps", vec3(self.velocity_mps, "velocity_mps"))

    def position(self, t: TimeLike) -> np.ndarray:
        return self.start_m + self.velocity_mps * _as_float_time(t)

    def positions(self, t: np.ndarray) -> np.ndarray:
        tt = np.asarray(t, dtype=np.float64)[:, None]
        return self.start_m[None, :] + self.velocity_mps[None, :] * tt

    def max_speed_mps(self, horizon_s: float) -> float:
        return float(np.linalg.norm(self.velocity_mps))

    def breakpoints(self, horizon_s: float) -> list[float]:
        return [0.0, float(horizon_s)]

    @property
    def is_static(self) -> bool:
        return not np.any(self.velocity_mps)


@dataclass(frozen=True, eq=False)
class Waypoints(_VectorizedMixin):
    """Piecewise-linear motion through ``positions_m`` at exact times ``t_s`` (increasing).

    Before the first time the node rests at the first waypoint, after the last at the last.
    """

    t_s: tuple[Fraction, ...]
    positions_m: np.ndarray

    def __post_init__(self) -> None:
        ts = tuple(parse_exact(x, name="waypoint t_s") for x in self.t_s)
        p = np.asarray(self.positions_m, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3 or p.shape[0] != len(ts) or len(ts) < 1:
            raise ConfigurationError("waypoints need matching t_s and positions_m (K, 3)")
        if any(b <= a for a, b in itertools.pairwise(ts)):
            raise ConfigurationError("waypoint times must be strictly increasing")
        if not np.all(np.isfinite(p)):
            raise ConfigurationError("waypoint positions must be finite")
        object.__setattr__(self, "t_s", ts)
        object.__setattr__(self, "positions_m", p)

    def position(self, t: TimeLike) -> np.ndarray:
        ts = self.t_s
        tq = t if isinstance(t, Fraction | int) else Fraction(t)
        if tq <= ts[0]:
            return self.positions_m[0].copy()
        if tq >= ts[-1]:
            return self.positions_m[-1].copy()
        i = bisect.bisect_right(ts, tq) - 1
        w = float((tq - ts[i]) / (ts[i + 1] - ts[i]))  # exact fraction, then rounded once
        return self.positions_m[i] + w * (self.positions_m[i + 1] - self.positions_m[i])

    def max_speed_mps(self, horizon_s: float) -> float:
        best = 0.0
        for i in range(len(self.t_s) - 1):
            d = float(np.linalg.norm(self.positions_m[i + 1] - self.positions_m[i]))
            best = max(best, d / float(self.t_s[i + 1] - self.t_s[i]))
        return best

    def breakpoints(self, horizon_s: float) -> list[float]:
        pts = [0.0, float(horizon_s)] + [float(x) for x in self.t_s if 0 <= x <= horizon_s]
        return sorted(set(pts))

    @property
    def is_static(self) -> bool:
        return bool(np.all(self.positions_m == self.positions_m[0]))


class _RandomWaypointBase(_VectorizedMixin):
    """Bettstetter random waypoint: legs drawn per leg index from a keyed generator.

    Waypoint ``w_i`` is drawn from ``leg_rng(i)`` (uniform over the domain); leg ``i`` goes
    from ``w_i`` to ``w_{i+1}`` at constant ``speed_mps``, then pauses ``pause_s``. Leg end
    times are accumulated in a fixed order, so positions are a pure function of time.
    """

    speed_mps: float
    pause_s: float

    def __init__(
        self, leg_rng: Callable[[int], np.random.Generator], speed_mps: float, pause_s: TimeLike = 0
    ) -> None:
        if not speed_mps > 0.0:
            raise ConfigurationError("random waypoint speed_mps must be > 0")
        self._leg_rng = leg_rng
        self.speed_mps = float(speed_mps)
        self.pause_s = float(pause_s)
        if self.pause_s < 0.0:
            raise ConfigurationError("pause_s must be >= 0")
        self._wp: list[np.ndarray] = [self._draw(0)]
        self._t_end: list[float] = []  # time at which leg i (incl. pause) ends
        self._t_move_end: list[float] = []

    def _draw(self, i: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _extend_to(self, t: float) -> None:
        while not self._t_end or self._t_end[-1] <= t:
            i = len(self._t_end)
            self._wp.append(self._draw(i + 1))
            d = float(np.linalg.norm(self._wp[i + 1] - self._wp[i]))
            t0 = self._t_end[-1] if self._t_end else 0.0
            tm = t0 + d / self.speed_mps
            self._t_move_end.append(tm)
            self._t_end.append(tm + self.pause_s)

    def position(self, t: TimeLike) -> np.ndarray:
        tf = _as_float_time(t)
        if tf < 0.0:
            return self._wp[0].copy()
        self._extend_to(tf)
        i = bisect.bisect_right(self._t_end, tf)
        t0 = self._t_end[i - 1] if i > 0 else 0.0
        tm = self._t_move_end[i]
        a, b = self._wp[i], self._wp[i + 1]
        if tf >= tm or tm == t0:
            return b.copy()
        return a + ((tf - t0) / (tm - t0)) * (b - a)

    def leg_starts(self, horizon_s: float) -> list[float]:
        """Leg start times in [0, horizon_s] (s)."""
        self._extend_to(horizon_s)
        return [0.0] + [x for x in self._t_end if x <= horizon_s]

    def waypoint(self, i: int) -> np.ndarray:
        """Waypoint ``w_i`` (3,), m."""
        while len(self._wp) <= i:
            self._extend_to(self._t_end[-1] if self._t_end else 0.0)
        return self._wp[i].copy()

    def max_speed_mps(self, horizon_s: float) -> float:
        return self.speed_mps

    def breakpoints(self, horizon_s: float) -> list[float]:
        self._extend_to(horizon_s)
        pts = {0.0, float(horizon_s)}
        pts.update(x for x in self._t_end if x <= horizon_s)
        pts.update(x for x in self._t_move_end if x <= horizon_s)
        return sorted(pts)

    @property
    def is_static(self) -> bool:
        return False


class RandomWaypoint1D(_RandomWaypointBase):
    """1-D random waypoint on the segment ``p0_m -> p1_m`` (ST-01 model, section 5.10)."""

    def __init__(
        self,
        p0_m: Sequence[float],
        p1_m: Sequence[float],
        speed_mps: float,
        leg_rng: Callable[[int], np.random.Generator],
        pause_s: TimeLike = 0,
    ) -> None:
        self.p0_m = vec3(p0_m, "p0_m")
        self.p1_m = vec3(p1_m, "p1_m")
        if np.all(self.p0_m == self.p1_m):
            raise ConfigurationError("rwp_1d needs p0_m != p1_m")
        super().__init__(leg_rng, speed_mps, pause_s)

    def _draw(self, i: int) -> np.ndarray:
        u = float(self._leg_rng(i).random())
        return self.p0_m + u * (self.p1_m - self.p0_m)

    @property
    def length_m(self) -> float:
        """Segment length l, m."""
        return float(np.linalg.norm(self.p1_m - self.p0_m))


class RandomWaypoint2D(_RandomWaypointBase):
    """2-D random waypoint, uniform in the rectangle ``[x0, x1] x [y0, y1]`` at height ``z_m``."""

    def __init__(
        self,
        bounds_m: Sequence[Sequence[float]],
        z_m: float,
        speed_mps: float,
        leg_rng: Callable[[int], np.random.Generator],
        pause_s: TimeLike = 0,
    ) -> None:
        (x0, y0), (x1, y1) = bounds_m
        if not (x1 > x0 and y1 > y0):
            raise ConfigurationError("rwp_2d bounds must be [[x0, y0], [x1, y1]] with x1>x0, y1>y0")
        self.lo = np.array([x0, y0], dtype=np.float64)
        self.hi = np.array([x1, y1], dtype=np.float64)
        self.z_m = float(z_m)
        super().__init__(leg_rng, speed_mps, pause_s)

    def _draw(self, i: int) -> np.ndarray:
        u = self._leg_rng(i).random(2)
        xy = self.lo + u * (self.hi - self.lo)
        return np.array([xy[0], xy[1], self.z_m])


class TraceTrajectory(_VectorizedMixin):
    """Recorded trace ``(t, x, y, z)`` interpolated by a C^2 cubic spline; no extrapolation."""

    def __init__(self, t_s: Sequence[float], xyz_m: np.ndarray) -> None:
        from scipy.interpolate import CubicSpline

        t = np.asarray(t_s, dtype=np.float64)
        p = np.asarray(xyz_m, dtype=np.float64)
        if t.ndim != 1 or p.shape != (t.size, 3) or t.size < 2:
            raise ConfigurationError("trace needs (K,) times and (K, 3) positions, K >= 2")
        if np.any(np.diff(t) <= 0):
            raise ConfigurationError("trace times must be strictly increasing")
        self.t_s = t
        self.xyz_m = p
        self._spline = CubicSpline(t, p, axis=0, bc_type="not-a-knot", extrapolate=False)
        self._dspline = self._spline.derivative()

    @classmethod
    def from_csv(cls, path: str | Path) -> TraceTrajectory:
        """Load a CSV with header ``t,x,y,z`` (s, m)."""
        data: Any = np.genfromtxt(Path(path), delimiter=",", names=True, dtype=np.float64)
        try:
            return cls(data["t"], np.stack([data["x"], data["y"], data["z"]], axis=1))
        except ValueError as exc:
            raise ConfigurationError(f"trace {path}: expected columns t,x,y,z") from exc

    def _check(self, t: np.ndarray) -> None:
        if np.any(t < self.t_s[0]) or np.any(t > self.t_s[-1]):
            raise GeometryError(
                f"trace evaluated outside [{self.t_s[0]}, {self.t_s[-1]}] s (no extrapolation)"
            )

    def position(self, t: TimeLike) -> np.ndarray:
        tt = np.array([_as_float_time(t)])
        self._check(tt)
        return np.asarray(self._spline(tt)[0], dtype=np.float64)

    def positions(self, t: np.ndarray) -> np.ndarray:
        tt = np.asarray(t, dtype=np.float64)
        self._check(tt)
        return np.asarray(self._spline(tt), dtype=np.float64)

    def max_speed_mps(self, horizon_s: float) -> float:
        hi = min(float(horizon_s), float(self.t_s[-1]))
        tt = np.linspace(self.t_s[0], hi, max(2, 20 * self.t_s.size))
        v = np.linalg.norm(self._dspline(tt), axis=1)
        return float(v.max()) * 1.05  # 5 % margin for the sampled maximum

    def breakpoints(self, horizon_s: float) -> list[float]:
        return sorted({0.0, float(horizon_s), *[float(x) for x in self.t_s if x <= horizon_s]})

    @property
    def is_static(self) -> bool:
        return bool(np.all(self.xyz_m == self.xyz_m[0]))


class ExternalTrajectory(_VectorizedMixin):
    """Positions set by timed ``node.set_position`` / ``node.set_trajectory`` events.

    Before the first event the ``initial`` trajectory applies. ``set_position(t, p)`` holds
    ``p`` from ``t`` on; ``set_trajectory(t, traj)`` follows ``traj`` evaluated at the time
    since the event (``traj.position(t - t_event)``). Events must arrive in time order.
    """

    def __init__(self, initial: Trajectory) -> None:
        self._segments: list[tuple[Fraction, Trajectory, bool]] = [(Fraction(0), initial, False)]

    def _append(self, t: TimeLike, traj: Trajectory, relative: bool) -> None:
        tq = to_exact(t, name="event time")
        if tq < self._segments[-1][0]:
            raise ConfigurationError("external trajectory events must be in time order")
        self._segments.append((tq, traj, relative))

    def set_position(self, t: TimeLike, position_m: Sequence[float]) -> None:
        """Hold ``position_m`` from time ``t`` on."""
        self._append(t, Static(np.asarray(position_m, dtype=np.float64)), False)

    def set_trajectory(self, t: TimeLike, traj: Trajectory) -> None:
        """Follow ``traj`` (time measured from the event) from time ``t`` on."""
        self._append(t, traj, True)

    def position(self, t: TimeLike) -> np.ndarray:
        tq = t if isinstance(t, Fraction | int) else Fraction(t)
        starts = [s[0] for s in self._segments]
        i = max(0, bisect.bisect_right(starts, tq) - 1)
        t0, traj, rel = self._segments[i]
        return traj.position(tq - t0 if rel else tq)

    def max_speed_mps(self, horizon_s: float) -> float:
        return max(s[1].max_speed_mps(horizon_s) for s in self._segments)

    def breakpoints(self, horizon_s: float) -> list[float]:
        pts = {0.0, float(horizon_s)}
        for t0, traj, rel in self._segments:
            pts.add(float(t0))
            off = float(t0) if rel else 0.0
            pts.update(off + x for x in traj.breakpoints(horizon_s) if off + x <= horizon_s)
        return sorted(x for x in pts if 0.0 <= x <= horizon_s)

    @property
    def is_static(self) -> bool:
        return all(s[1].is_static for s in self._segments) and len(self._segments) == 1


def trajectory_from_spec(
    spec: dict[str, Any] | None,
    position_m: Sequence[float] | None = None,
    *,
    node: str = "",
    seed: int = 0,
    base_dir: Path | None = None,
) -> Trajectory:
    """Build a trajectory from the scenario form (section 9.2).

    ``position_m`` alone gives :class:`Static`. ``spec["model"]`` selects
    ``static`` (``position_m``), ``waypoints`` (``t_s`` exact strings, ``positions_m``),
    ``constant_velocity`` (``start_m``, ``velocity_mps``), ``rwp_1d`` (``p0_m``, ``p1_m``,
    ``speed_mps``, ``pause_s``), ``rwp_2d`` (``bounds_m``, ``z_m``, ``speed_mps``,
    ``pause_s``), ``trace`` (``file``) or ``external`` (``initial_m``). Random waypoint legs
    use the keyed purpose ``mobility/<node>`` (section 6.2).
    """
    if spec is None:
        if position_m is None:
            raise ConfigurationError(f"node {node!r}: give position_m or trajectory")
        return Static(np.asarray(position_m, dtype=np.float64))
    model = spec.get("model")
    p = {k: v for k, v in spec.items() if k != "model"}

    def leg_rng(i: int) -> np.random.Generator:
        return keyed_generator(seed, "mobility", node, counter=i)

    if model == "static":
        return Static(np.asarray(p.get("position_m", position_m), dtype=np.float64))
    if model == "waypoints":
        return Waypoints(tuple(p["t_s"]), np.asarray(p["positions_m"], dtype=np.float64))
    if model == "constant_velocity":
        start = p.get("start_m", position_m)
        return ConstantVelocity(
            np.asarray(start, dtype=np.float64), np.asarray(p["velocity_mps"], dtype=np.float64)
        )
    if model == "rwp_1d":
        return RandomWaypoint1D(
            p["p0_m"],
            p["p1_m"],
            float(p["speed_mps"]),
            leg_rng,
            parse_exact(p.get("pause_s", "0"), name="pause_s"),
        )
    if model == "rwp_2d":
        return RandomWaypoint2D(
            p["bounds_m"],
            float(p.get("z_m", 0.0)),
            float(p["speed_mps"]),
            leg_rng,
            parse_exact(p.get("pause_s", "0"), name="pause_s"),
        )
    if model == "trace":
        path = Path(p["file"])
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        return TraceTrajectory.from_csv(path)
    if model == "external":
        init = p.get("initial_m", position_m)
        if init is None:
            raise ConfigurationError(f"node {node!r}: external trajectory needs initial_m")
        return ExternalTrajectory(Static(np.asarray(init, dtype=np.float64)))
    raise ConfigurationError(f"node {node!r}: unknown trajectory model {model!r}")
