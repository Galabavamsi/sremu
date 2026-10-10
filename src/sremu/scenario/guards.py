"""Scenario guards and derived quantities (DESIGN.md section 9.5; also 3.2, 4.5, 4.9, 5.3-5.6).

:func:`check_scenario` runs every scenario-level guard on a resolved scenario and returns a
:class:`GuardReport` (all errors and warnings, each with a stable ``code``), so
``sremu validate`` can print everything at once. ``report.raise_for_errors()`` raises the
first error's exception class with every error in the message.

**Errors:** unique ids; references resolve (traffic streams, state libraries, files);
exact rates (schema); RATE_MISMATCH links; any RIS cell closer than one wavelength to a
node; a malformed trace file (TRACE_FILE) or a trace that does not cover [0, duration]
(TRACE_COVERAGE: traces are not extrapolated, section 5.10); occupied band > 0.8 f_s
with the default FD target; table checks that need the
scenario (passivity, ``assume_mag`` for ``mag: null`` states, efficiency on measured or
digitized tables, frequency domain, validity bandwidth, initial states); ``t_u_s`` above
the automatic value (when the caller supplies it; the automatic T_u policy is WP5's
``channel/interp.py``); incidence-only tables without ``angle_mapping`` (schema);
coupling-mode matrix (4.9).

**Warnings:** a cell closer than 3 wavelengths to a node; an undeclared ``occupied_bw_hz``;
``cell_model: point`` forced with pitch > lambda/2; absolute paths longer than 1 km
(float64 reference note); recorder disk estimate above 10 GB.

Geometry here is plain vector arithmetic on the scenario (cell centres, node reachable
sets over [0, duration]); the propagation physics lives in ``sremu.physics`` (WP2), which
this layer may not import. A trace is the C2 cubic spline of section 5.10 (scipy
``CubicSpline``, not-a-knot, no extrapolation; the definition physics evaluates), sampled
densely with a bounded chord error (:func:`trace_geometry`), so CELL_TOO_CLOSE, CELL_NEAR
and LONG_PATH see the spline, not a polyline through the rows; distances are made
conservative by that bound.
:func:`derived_quantities` computes the numbers ``sremu validate`` prints; the per-link
narrowband estimate and T_curv bound are informative, the engine's policies (WP5) decide.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import check_link
from sremu.errors import (
    BandwidthExceeded,
    ConfigurationError,
    FDBandwidthExceeded,
    GeometryError,
    OutOfDomain,
    PassivityError,
    RateMismatchError,
    SchemaError,
    TableValidationError,
    ValidityWarning,
)
from sremu.scenario.load import ResolvedScenario
from sremu.scenario.models import (
    ConstantVelocityTrajectory,
    ExternalTrajectory,
    Node,
    RisPanel,
    Rwp1dTrajectory,
    Rwp2dTrajectory,
    Scenario,
    StateLibrary,
    StaticTrajectory,
    TraceTrajectory,
    WaypointsTrajectory,
    exact_to_str,
)
from sremu.timebase import page_size
from sremu.units import C0, thermal_noise_dbm

__all__ = [
    "DISK_WARN_BYTES",
    "LONG_PATH_M",
    "TRACE_CHORD_TOL_M",
    "GuardIssue",
    "GuardReport",
    "NodeGeometry",
    "cell_centres",
    "check_scenario",
    "derived_quantities",
    "fd_choice",
    "links",
    "max_pair_distance",
    "node_geometry",
    "panel_extent_m",
    "read_trace",
    "set_distance_lower_bound",
    "trace_geometry",
]

LONG_PATH_M: Final[float] = 1000.0
"""Paths longer than this get the float64-reference warning (section 9.5)."""
DISK_WARN_BYTES: Final[float] = 10e9
"""Recorder disk estimate above which a warning is issued (section 9.5)."""
_FD_MAX_FRACTION: Final[float] = 0.8
_DEFAULT_FD_TARGET_DB: Final[float] = -60.0
_INTEGER_DELAY_PHASE: Final[float] = 0.05  # pi*B/(2 f_s) <= 0.05 (section 4.5)
_MAX_SEPARABLE_FREE_STATES: Final[int] = 16  # S > 16 needs a separable table (3.5)


# =========================================================================== report


@dataclass(frozen=True)
class GuardIssue:
    """One guard finding. ``exc`` is the exception class raised for an error."""

    severity: Literal["error", "warning"]
    code: str
    where: str
    message: str
    exc: type[Exception] = ConfigurationError

    def __str__(self) -> str:
        return f"[{self.code}] {self.where}: {self.message}"


@dataclass
class GuardReport:
    """All guard findings of one scenario plus its derived quantities."""

    issues: list[GuardIssue] = field(default_factory=list)
    derived: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[GuardIssue]:
        """Error findings."""
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[GuardIssue]:
        """Warning findings."""
        return [i for i in self.issues if i.severity == "warning"]

    def codes(self, severity: str | None = None) -> set[str]:
        """Codes of the findings (optionally of one severity)."""
        return {i.code for i in self.issues if severity is None or i.severity == severity}

    def error(self, code: str, where: str, msg: str, exc: type[Exception]) -> None:
        """Record an error."""
        self.issues.append(GuardIssue("error", code, where, msg, exc))

    def warn(self, code: str, where: str, msg: str) -> None:
        """Record a warning."""
        self.issues.append(GuardIssue("warning", code, where, msg, ValidityWarning))

    def raise_for_errors(self) -> None:
        """Raise the first error's exception class, listing every error; no-op if none."""
        errs = self.errors
        if errs:
            text = f"{len(errs)} scenario error(s):\n" + "\n".join(f"  {e}" for e in errs)
            raise errs[0].exc(text)

    def emit_warnings(self) -> None:
        """Emit each warning as a ``ValidityWarning``."""
        for w in self.warnings:
            warnings.warn(str(w), ValidityWarning, stacklevel=2)


# =========================================================================== geometry


def _unit(v: Any) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def panel_axes(p: RisPanel) -> tuple[NDArray[np.float64], ...]:
    """(n, u, v) unit vectors: u is orthogonalized against n, v = n x u (section 3.1)."""
    n = _unit(p.pose.normal)
    u = np.asarray(p.pose.u_axis, dtype=np.float64)
    u = _unit(u - np.dot(u, n) * n)
    return n, u, np.cross(n, u)


def _offsets(n: int, pitch: float, n_tiles: int, gap: float) -> NDArray[np.float64]:
    i = np.arange(n, dtype=np.float64)
    per_tile = n // n_tiles
    tile = np.arange(n) // per_tile
    return (i - (n - 1) / 2.0) * pitch + (tile - (n_tiles - 1) / 2.0) * gap


def cell_centres(p: RisPanel) -> NDArray[np.float64]:
    """(N, 3) cell centres in m, index ``n = iv*n_u + iu``; tile gaps are explicit."""
    _n_hat, u_hat, v_hat = panel_axes(p)
    tu = p.tiles.n_tiles_u if p.tiles else 1
    tv = p.tiles.n_tiles_v if p.tiles else 1
    gap = p.tiles.gap_m if p.tiles else 0.0
    ou = _offsets(p.grid.n_u, p.grid.pitch_u_m, tu, gap)
    ov = _offsets(p.grid.n_v, p.grid.pitch_v_m, tv, gap)
    gu, gv = np.meshgrid(ou, ov)  # (n_v, n_u): row iv, column iu
    c = np.asarray(p.pose.center_m, dtype=np.float64)
    return c + gu.reshape(-1, 1) * u_hat + gv.reshape(-1, 1) * v_hat


def panel_extent_m(p: RisPanel) -> tuple[float, float]:
    """Physical side lengths (u, v) in m including tile gaps."""
    tu = p.tiles.n_tiles_u if p.tiles else 1
    tv = p.tiles.n_tiles_v if p.tiles else 1
    gap = p.tiles.gap_m if p.tiles else 0.0
    return (
        p.grid.n_u * p.grid.pitch_u_m + (tu - 1) * gap,
        p.grid.n_v * p.grid.pitch_v_m + (tv - 1) * gap,
    )


_PAIR_BLOCK: Final[int] = 1 << 18  # (segment, query) pairs per vectorized block


def _segment_distances(
    q: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """(S, M) distances from query points ``q`` (M, 3) to segments ``[a_s, b_s]`` (S, 3)."""
    ab = b - a
    den = np.einsum("ij,ij->i", ab, ab)
    aq = q[None, :, :] - a[:, None, :]
    num = np.einsum("smk,sk->sm", aq, ab)
    safe = np.where(den > 0.0, den, 1.0)
    t = np.where(den[:, None] > 0.0, np.clip(num / safe[:, None], 0.0, 1.0), 0.0)
    diff = aq - t[:, :, None] * ab[:, None, :]
    out: NDArray[np.float64] = np.sqrt(np.einsum("smk,smk->sm", diff, diff))
    return out


@dataclass(frozen=True)
class NodeGeometry:
    """Reachable positions of a node over [0, duration].

    ``kind``: ``"points"`` (a polyline through ``pts``; one point for a static node) or
    ``"rect"`` (axis-aligned rectangle ``pts = [[x0, y0, z], [x1, y1, z]]``).
    ``initial``: position at t = 0 (None when random, e.g. random waypoint, or unknown).
    ``speed_max_mps``: maximum speed, or an upper bound on it (None when unknown, e.g.
    external).
    ``tol_m``: bound on the distance between the true path and the polyline through
    ``pts`` (0 for exact polylines; the chord bound of a sampled trace spline). Distances
    computed from ``pts`` are within ``tol_m`` of the true ones.
    ``defined_s``: ``(t0, t1)`` interval on which the trajectory is defined (a trace: no
    extrapolation), or None when it is defined for all t.
    """

    kind: Literal["points", "rect"]
    pts: NDArray[np.float64]
    initial: NDArray[np.float64] | None
    speed_max_mps: float | None
    tol_m: float = 0.0
    defined_s: tuple[float, float] | None = None

    def vertices(self) -> NDArray[np.float64]:
        """Points whose convex hull contains the reachable set (for maximum distances)."""
        if self.kind == "points":
            return self.pts
        (x0, y0, z), (x1, y1, _) = self.pts
        return np.array([[x0, y0, z], [x1, y0, z], [x0, y1, z], [x1, y1, z]], dtype=np.float64)

    def _rect_distance(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        (x0, y0, z), (x1, y1, _) = self.pts
        dx = np.clip(q[:, 0], x0, x1) - q[:, 0]
        dy = np.clip(q[:, 1], y0, y1) - q[:, 1]
        out: NDArray[np.float64] = np.sqrt(dx * dx + dy * dy + (q[:, 2] - z) ** 2)
        return out

    def min_distance(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Minimum distance in m from each query point (M, 3) to the reachable set."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        if self.kind == "rect":
            return self._rect_distance(q)
        pts = self.pts
        if len(pts) == 1:
            out: NDArray[np.float64] = np.linalg.norm(q - pts[0], axis=1)
            return out
        a, b = pts[:-1], pts[1:]
        best = np.full(len(q), np.inf)
        step = max(1, _PAIR_BLOCK // max(1, len(q)))
        for i in range(0, len(a), step):
            d = _segment_distances(q, a[i : i + step], b[i : i + step])
            best = np.minimum(best, d.min(axis=0))
        return best

    def min_distance_to(self, q: NDArray[np.float64]) -> float:
        """``min(self.min_distance(q))``: the smallest distance in m between the query
        points (M, 3) and the reachable set.

        For long polylines (sampled traces) the segments are visited in order of a lower
        bound (distance to the centroid of ``q`` minus the radius of ``q``) and the search
        stops once no remaining segment can be closer, so the result is exact but cheap.
        """
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        if self.kind == "rect" or len(self.pts) <= 2:
            return float(self.min_distance(q).min())
        a, b = self.pts[:-1], self.pts[1:]
        c = q.mean(axis=0)
        r = float(np.max(np.linalg.norm(q - c, axis=1)))
        lb = _segment_distances(c[None, :], a, b)[:, 0] - r
        order = np.argsort(lb, kind="stable")
        step = max(1, _PAIR_BLOCK // len(q))
        best = math.inf
        for i in range(0, len(order), step):
            idx = order[i : i + step]
            if lb[idx[0]] >= best:
                break
            best = min(best, float(_segment_distances(q, a[idx], b[idx]).min()))
        return best

    def sample_cover(self, cover_m: float) -> tuple[NDArray[np.float64], float]:
        """Points ``S`` on the polyline with every polyline point within ``c <= cover_m`` of
        some point of ``S``: ``(S, c)``. Each segment is split into equal parts of length at
        most ``2 cover_m`` (``c`` is half the longest part). Polylines only."""
        if self.kind != "points":
            raise ValueError("sample_cover is defined for polylines")
        pts = self.pts
        if len(pts) == 1:
            return pts, 0.0
        a, ab = pts[:-1], np.diff(pts, axis=0)
        seg = np.linalg.norm(ab, axis=1)
        k = np.maximum(1, np.ceil(seg / (2.0 * cover_m))).astype(np.int64)
        idx = np.repeat(np.arange(len(k)), k)
        start = np.repeat(np.cumsum(k) - k, k)
        frac = (np.arange(int(k.sum())) - start) / np.repeat(k, k)
        s = np.vstack([a[idx] + frac[:, None] * ab[idx], pts[-1:]])
        return s, float(np.max(seg / k)) / 2.0

    def path_length_m(self) -> float:
        """Length of the polyline (0 for a rectangle)."""
        if self.kind != "points" or len(self.pts) == 1:
            return 0.0
        return float(np.linalg.norm(np.diff(self.pts, axis=0), axis=1).sum())


_COVER_ABS_M: Final[float] = 1e-3
_COVER_REL: Final[float] = 1.0 / 4000.0
_EXACT_PAIRS: Final[int] = 20_000_000


def set_distance_lower_bound(ga: NodeGeometry, gb: NodeGeometry) -> float:
    """Lower bound (m) on ``min |x - y|`` over ``x`` in ``ga``'s and ``y`` in ``gb``'s
    reachable sets (the polylines and rectangles themselves; ``tol_m`` is not applied).

    Exact when either set is a single point (point-to-set distance) or both are rectangles
    (closed form). Otherwise the polyline ``A`` is replaced by samples ``S`` that cover it
    within ``c`` (:meth:`NodeGeometry.sample_cover`, ``c <= max(1 mm, length/4000)``) and
    ``min_s d(s, B) - c`` is returned: for every ``x`` there is an ``s`` with
    ``|x - s| <= c``, so ``d(x, B) >= d(s, B) - c``. ``d(s, B)`` is exact (point to
    rectangle or to polyline segments); when both polylines are long it is replaced by the
    nearest sample of ``B``'s own cover (k-d tree), minus that cover too. The result is never
    above the true distance and is below it by at most the cover radii used.
    """
    if ga.kind == "points" and len(ga.pts) == 1:
        return gb.min_distance_to(ga.pts)
    if gb.kind == "points" and len(gb.pts) == 1:
        return ga.min_distance_to(gb.pts)
    if ga.kind == "rect" and gb.kind == "rect":
        (ax0, ay0, az), (ax1, ay1, _) = ga.pts
        (bx0, by0, bz), (bx1, by1, _) = gb.pts
        gx = max(0.0, bx0 - ax1, ax0 - bx1)
        gy = max(0.0, by0 - ay1, ay0 - by1)
        return math.sqrt(gx * gx + gy * gy + (az - bz) ** 2)
    if ga.kind == "rect":
        ga, gb = gb, ga  # ga is a polyline from here on
    sa, ca = ga.sample_cover(max(_COVER_ABS_M, _COVER_REL * ga.path_length_m()))
    if gb.kind == "rect" or len(sa) * (len(gb.pts) - 1) <= _EXACT_PAIRS:
        return max(gb.min_distance_to(sa) - ca, 0.0)
    from scipy.spatial import cKDTree

    sb, cb = gb.sample_cover(max(_COVER_ABS_M, _COVER_REL * gb.path_length_m()))
    d = float(np.min(cKDTree(sb).query(sa, k=1)[0]))
    return max(d - ca - cb, 0.0)


# --------------------------------------------------------------------------- traces

TRACE_CHORD_TOL_M: Final[float] = 1e-6
"""Target bound (m) on the distance between a trace's C2 spline and the polyline through
the guard's samples of it (section 5.10)."""
_TRACE_MIN_PER_PIECE: Final[int] = 4
_TRACE_MAX_PER_PIECE: Final[int] = 4096
_TRACE_COLUMNS: Final[tuple[str, ...]] = ("t", "x", "y", "z")


def read_trace(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read a trace CSV: ``(t_s (K,), xyz_m (K, 3))``.

    Format (section 5.10, "CSV (t, x, y, z)"): the first non-blank line is the header (an
    optional leading ``#`` is allowed); it names the columns ``t, x, y, z`` (s, m) exactly
    (case-sensitive, surrounding spaces ignored), each once. Columns are found by name;
    other columns are ignored, but every data row has as many fields as the header. Later
    blank lines and lines starting with ``#`` are skipped. At least two rows, finite values
    and strictly increasing times are required (``SchemaError`` otherwise).

    The format is deliberately no wider than what the engine's trace reader
    (``sremu.physics.geometry.TraceTrajectory.from_csv``) accepts, so a trace that passes
    the guards also loads in the engine (unit-suffixed names such as ``t_s`` are rejected
    for that reason).
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    i0 = next((i for i, line in enumerate(lines) if line.strip()), None)
    if i0 is None:
        raise SchemaError(f"{p}: empty trace file")
    names = [x.strip() for x in lines[i0].strip().lstrip("#").split(",")]
    cols: list[int] = []
    for key in _TRACE_COLUMNS:
        hits = [j for j, nm in enumerate(names) if nm == key]
        if len(hits) != 1:
            raise SchemaError(
                f"{p}: the header must name each of the columns t, x, y, z (s, m) exactly "
                f"once; column {key!r} found {len(hits)} times in {lines[i0].strip()!r}"
            )
        cols.append(hits[0])
    rows: list[list[float]] = []
    for i in range(i0 + 1, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != len(names):
            raise SchemaError(
                f"{p}: trace row {i + 1} has {len(parts)} fields, the header {len(names)}: {line!r}"
            )
        try:
            vals = [float(parts[j]) for j in cols]
        except ValueError:
            raise SchemaError(f"{p}: bad trace row {i + 1}: {line!r}") from None
        if not all(math.isfinite(v) for v in vals):
            raise SchemaError(f"{p}: non-finite value in trace row {i + 1}: {line!r}")
        rows.append(vals)
    if len(rows) < 2:
        raise SchemaError(f"{p}: a trace needs >= 2 rows of t, x, y, z")
    arr = np.asarray(rows, dtype=np.float64)
    if np.any(np.diff(arr[:, 0]) <= 0):
        raise SchemaError(f"{p}: trace times must be strictly increasing")
    return arr[:, 0].copy(), arr[:, 1:].copy()


def trace_geometry(
    t_s: NDArray[np.float64], xyz_m: NDArray[np.float64], duration_s: Fraction | float
) -> NodeGeometry:
    """Reachable set of a trace over [0, duration] intersected with [t_0, t_last] (5.10).

    The trajectory is the C2 cubic spline through the rows, ``scipy.interpolate.CubicSpline
    (t, xyz, axis=0, bc_type="not-a-knot", extrapolate=False)`` (the definition the physics
    layer evaluates). It is sampled densely: on each spline piece ``[a, b]`` of the window
    (pieces split at 0 and the duration), with ``A = max |p''|`` over the piece (``p''`` is
    linear there, so ``A`` is its larger end value), ``n = ceil((b - a) sqrt(A/(8 tol)))``
    samples (clipped to [4, 4096]) bound the distance between the spline and the polyline
    through the samples by ``((b - a)/n)**2 A/8`` (linear-interpolation error bound);
    ``tol_m`` is the largest such bound (``tol`` = :data:`TRACE_CHORD_TOL_M`).
    ``speed_max_mps`` is ``max |p'|`` over the samples plus ``max (h/2) A`` (a rigorous upper
    bound, ``h`` the sample step). When the window is empty (the trace does not overlap the
    run) the whole trace is described; the coverage guard reports it.
    """
    from scipy.interpolate import CubicSpline

    t = np.asarray(t_s, dtype=np.float64)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    spl = CubicSpline(t, xyz, axis=0, bc_type="not-a-knot", extrapolate=False)
    span = (float(t[0]), float(t[-1]))
    d = float(duration_s)
    lo, hi = max(span[0], 0.0), min(span[1], d)
    if lo > hi:
        lo, hi = span
    bps = np.unique(np.concatenate([[lo, hi], t[(t > lo) & (t < hi)]]))
    init_ok = lo == 0.0
    if len(bps) == 1:
        pts = np.asarray(spl(bps), dtype=np.float64)
        v1 = float(np.linalg.norm(np.asarray(spl.derivative(1)(bps))[0]))
        return NodeGeometry("points", pts, pts[0] if init_ok else None, v1, 0.0, span)
    acc = np.linalg.norm(np.asarray(spl.derivative(2)(bps), dtype=np.float64), axis=1)
    a_piece = np.maximum(acc[:-1], acc[1:])
    width = np.diff(bps)
    n_f = np.ceil(width * np.sqrt(a_piece / (8.0 * TRACE_CHORD_TOL_M)))
    n = np.clip(n_f, _TRACE_MIN_PER_PIECE, _TRACE_MAX_PER_PIECE).astype(np.int64)
    ts = np.concatenate(
        [np.linspace(bps[i], bps[i + 1], int(n[i]), endpoint=False) for i in range(len(n))]
        + [bps[-1:]]
    )
    pts = np.asarray(spl(ts), dtype=np.float64)
    h = width / n
    tol = float(np.max(h * h * a_piece / 8.0))
    vel = np.linalg.norm(np.asarray(spl.derivative(1)(ts), dtype=np.float64), axis=1)
    v_bound = float(vel.max()) + float(np.max(0.5 * h * a_piece))
    return NodeGeometry("points", pts, pts[0] if init_ok else None, v_bound, tol, span)


def node_geometry(node: Node, duration_s: Fraction, base_dir: Path | None) -> NodeGeometry:
    """Reachable set of ``node`` over [0, duration_s] (see :class:`NodeGeometry`).

    A trace is read with :func:`read_trace` (``SchemaError`` on a malformed file) and
    sampled with :func:`trace_geometry`.
    """
    d = float(duration_s)
    if node.position_m is not None:
        p = np.asarray([node.position_m], dtype=np.float64)
        return NodeGeometry("points", p, p[0], 0.0)
    tr = node.trajectory
    if isinstance(tr, StaticTrajectory):
        p = np.asarray([tr.position_m], dtype=np.float64)
        return NodeGeometry("points", p, p[0], 0.0)
    if isinstance(tr, ExternalTrajectory):
        p = np.asarray([tr.initial_m], dtype=np.float64)
        return NodeGeometry("points", p, p[0], None)
    if isinstance(tr, ConstantVelocityTrajectory):
        s = np.asarray(tr.start_m, dtype=np.float64)
        v = np.asarray(tr.velocity_mps, dtype=np.float64)
        return NodeGeometry("points", np.vstack([s, s + v * d]), s, float(np.linalg.norm(v)))
    if isinstance(tr, WaypointsTrajectory):
        ts = np.array([float(t) for t in tr.t_s])
        ps = np.array(tr.positions_m, dtype=np.float64)
        knots = np.unique(np.concatenate([[0.0, d], ts[(ts > 0) & (ts < d)]]))
        pts = np.column_stack([np.interp(knots, ts, ps[:, k]) for k in range(3)])
        speeds = np.linalg.norm(np.diff(ps, axis=0), axis=1) / np.diff(ts)
        return NodeGeometry("points", pts, pts[0], float(speeds.max()) if speeds.size else 0.0)
    if isinstance(tr, Rwp1dTrajectory):
        pts = np.asarray([tr.p0_m, tr.p1_m], dtype=np.float64)
        return NodeGeometry("points", pts, None, tr.speed_mps)
    if isinstance(tr, Rwp2dTrajectory):
        (x0, y0), (x1, y1) = tr.bounds_m
        pts = np.asarray([[x0, y0, tr.z_m], [x1, y1, tr.z_m]], dtype=np.float64)
        return NodeGeometry("rect", pts, None, tr.speed_mps)
    if isinstance(tr, TraceTrajectory):
        path = Path(tr.file)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        t, xyz = read_trace(path)
        return trace_geometry(t, xyz, duration_s)
    raise AssertionError(f"unhandled trajectory {tr!r}")


def _extreme_points(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """A subset of ``p`` containing its convex-hull vertices (all of ``p`` when small or
    degenerate); maximum distances to ``p`` are attained on it."""
    if len(p) <= 64:
        return p
    from scipy.spatial import ConvexHull, QhullError

    try:
        hull = ConvexHull(p, qhull_options="QJ")  # joggle: planar and collinear sets work
    except (QhullError, ValueError):
        return p
    return p[np.unique(hull.vertices)]


def max_pair_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """``max |a_i - b_j|`` in m over two point sets (convex-hull reduction, then blocks)."""
    a = _extreme_points(np.atleast_2d(np.asarray(a, dtype=np.float64)))
    b = _extreme_points(np.atleast_2d(np.asarray(b, dtype=np.float64)))
    best = 0.0
    step = max(1, _PAIR_BLOCK // len(b))
    for i in range(0, len(a), step):
        diff = a[i : i + step, None, :] - b[None, :, :]
        best = max(best, float(np.sqrt(np.max(np.einsum("ijk,ijk->ij", diff, diff)))))
    return best


# =========================================================================== links / FD


def links(sc: Scenario) -> tuple[list[tuple[str, str]], list[str]]:
    """``(links, rate_mismatch_messages)``: (tx_stream, rx_stream) pairs on different nodes
    that form a link (equal rates, ``|dfc| < f_s/2``, section 3.2), in scenario order."""
    out: list[tuple[str, str]] = []
    bad: list[str] = []
    streams = sc.streams()
    for tx_id, (tx_node, tx) in streams.items():
        if tx.direction != "tx":
            continue
        for rx_id, (rx_node, rx) in streams.items():
            if rx.direction != "rx" or rx_node.id == tx_node.id:
                continue
            try:
                if check_link(tx.fc_hz, tx.rate_sps, rx.fc_hz, rx.rate_sps):
                    out.append((tx_id, rx_id))
            except RateMismatchError as exc:
                bad.append(f"{tx_id} -> {rx_id}: {exc}")
    return out, bad


def fd_choice(b_hz: float, fs: Fraction) -> dict[str, Any]:
    """Fractional-delay filter row for occupied band ``b_hz`` at rate ``fs`` (section 4.5
    table; [PILOT] error values). Informative: the renderer (WP5) decides."""
    frac = b_hz / float(fs)
    integer_ok = math.pi * b_hz / (2.0 * float(fs)) <= _INTEGER_DELAY_PHASE
    row: dict[str, Any]
    if frac <= 0.2:
        row = {"taps": 8, "beta": 8, "lookahead": 4, "pilot_error_db": -71.4}
    elif frac <= 0.6:
        row = {"taps": 16, "beta": 8, "lookahead": 8, "pilot_error_db": -75.5}
    elif frac <= _FD_MAX_FRACTION:
        row = {"taps": 32, "beta": 8, "lookahead": 16, "pilot_error_db": -76.5}
    else:
        row = {"taps": None, "beta": None, "lookahead": None, "pilot_error_db": None}
    return {"b_over_fs": frac, "integer_delay_allowed": integer_ok, **row}


# =========================================================================== guards


def _ids(sc: Scenario, rep: GuardReport) -> None:
    seen: dict[str, str] = {}
    objs = (
        [("nodes", n.id) for n in sc.nodes]
        + [("ris", p.id) for p in sc.ris]
        + [("environment.walls", w.id) for w in sc.environment.walls]
        + [("environment.occluders", o.id) for o in sc.environment.occluders]
        + [("environment.edges", e.id) for e in sc.environment.edges]
    )
    for where, oid in objs:
        if oid in seen:
            rep.error(
                "DUPLICATE_ID",
                where,
                f"id {oid!r} is already used in {seen[oid]} (ids are unique per scenario)",
                SchemaError,
            )
        else:
            seen[oid] = where
    for n in sc.nodes:
        names = [s.name for s in n.streams]
        for dup in sorted({x for x in names if names.count(x) > 1}):
            rep.error(
                "DUPLICATE_ID", f"nodes[{n.id}].streams", f"stream {dup!r} repeated", SchemaError
            )
    names = [c.name for c in sc.controllers]
    for dup in sorted({x for x in names if names.count(x) > 1}):
        rep.error("DUPLICATE_ID", "controllers", f"controller {dup!r} repeated", SchemaError)


def _refs(sc: Scenario, rep: GuardReport) -> None:
    streams = sc.streams()
    fed: dict[str, int] = {}
    for i, t in enumerate(sc.traffic):
        where = f"traffic[{i}]"
        if t.stream not in streams:
            rep.error("UNRESOLVED_REF", where, f"stream {t.stream!r} does not exist", SchemaError)
        elif streams[t.stream][1].direction != "tx":
            rep.error(
                "UNRESOLVED_REF", where, f"stream {t.stream!r} is not a TX stream", SchemaError
            )
        fed[t.stream] = fed.get(t.stream, 0) + 1
    for sid, k in fed.items():
        if k > 1:
            rep.error(
                "UNRESOLVED_REF", "traffic", f"stream {sid!r} has {k} generators", SchemaError
            )


def _coupling(sc: Scenario, rep: GuardReport) -> None:
    # lockstep + async is rejected by the schema; paced + barrier needs in-process ctrls.
    if sc.run.mode == "paced" and sc.run.coupling == "barrier":
        for c in sc.controllers:
            if c.kind == "external":
                rep.error(
                    "COUPLING",
                    f"controllers[{c.name}]",
                    "paced mode with barrier coupling allows in-process (builtin) controllers "
                    "only (section 4.9)",
                    ConfigurationError,
                )


def _bandwidth(sc: Scenario, rep: GuardReport) -> None:
    for sid, (_n, s) in sc.streams().items():
        if s.direction != "tx" or s.occupied_bw_hz is None:
            continue
        fs = float(s.rate_sps)
        if s.occupied_bw_hz > fs:
            rep.error(
                "FD_BANDWIDTH",
                sid,
                f"occupied_bw_hz {s.occupied_bw_hz:g} exceeds the sample rate {fs:g}",
                FDBandwidthExceeded,
            )
        elif s.occupied_bw_hz > _FD_MAX_FRACTION * fs:
            if sc.numerics.fd_error_target_db <= _DEFAULT_FD_TARGET_DB:
                rep.error(
                    "FD_BANDWIDTH",
                    sid,
                    f"occupied_bw_hz {s.occupied_bw_hz:g} > 0.8 f_s = {0.8 * fs:g} Hz: no FD "
                    f"filter meets fd_error_target_db = {sc.numerics.fd_error_target_db:g} dB "
                    "(48 taps reach only -57.1 dB at +/-0.45 f_s, section 4.5)",
                    FDBandwidthExceeded,
                )
            else:
                rep.warn(
                    "FD_BANDWIDTH_RELAXED",
                    sid,
                    f"occupied_bw_hz > 0.8 f_s with a relaxed fd_error_target_db "
                    f"({sc.numerics.fd_error_target_db:g} dB); the renderer decides",
                )


def _lambda_m(fc_hz: float) -> float:
    return C0 / fc_hz


def _distances(sc: Scenario, geo: Mapping[str, NodeGeometry], rep: GuardReport) -> None:
    """CELL_TOO_CLOSE / CELL_NEAR on the lower bound ``dmin - tol_m`` of the distance between
    the node's reachable set and the cell centres (conservative for sampled traces)."""
    for p in sc.ris:
        cells = cell_centres(p)
        for n in sc.nodes:
            g = geo.get(n.id)
            if g is None:  # malformed trace, reported as TRACE_FILE
                continue
            lam = max(_lambda_m(s.fc_hz) for s in n.streams)
            dmin = g.min_distance_to(cells) - g.tol_m
            where = f"ris[{p.id}] / nodes[{n.id}]"
            if dmin < lam:
                rep.error(
                    "CELL_TOO_CLOSE",
                    where,
                    f"a cell is {dmin:.4g} m from the node, below one wavelength "
                    f"({lam:.4g} m); the point-scatterer kernel is invalid (section 5.3)",
                    GeometryError,
                )
            elif dmin < 3.0 * lam:
                rep.warn(
                    "CELL_NEAR",
                    where,
                    f"a cell is {dmin:.4g} m from the node, below 3 wavelengths "
                    f"({3 * lam:.4g} m; point-scatterer approximation, section 5.3)",
                )


def _long_paths(
    sc: Scenario, geo: Mapping[str, NodeGeometry], lks: list[tuple[str, str]], rep: GuardReport
) -> None:
    """LONG_PATH on the upper bound ``max + tol_m`` of LOS and RIS-hop path lengths."""
    streams = sc.streams()
    worst = 0.0
    where = ""
    for tx_id, rx_id in lks:
        gt, gr = geo.get(streams[tx_id][0].id), geo.get(streams[rx_id][0].id)
        if gt is None or gr is None:
            continue
        a, b = gt.vertices(), gr.vertices()
        tol = gt.tol_m + gr.tol_m
        d_los = max_pair_distance(a, b) + tol
        if d_los > worst:
            worst, where = d_los, f"{tx_id}->{rx_id} (LOS)"
        for p in sc.ris:
            c = np.asarray(p.pose.center_m, dtype=np.float64)
            hop = (
                float(np.max(np.linalg.norm(a - c, axis=1)))
                + float(np.max(np.linalg.norm(b - c, axis=1)))
                + tol
            )
            if hop > worst:
                worst, where = hop, f"{tx_id}->{rx_id} (via {p.id})"
    if worst > LONG_PATH_M:
        rep.warn(
            "LONG_PATH",
            where,
            f"absolute path length up to {worst:.4g} m > 1 km: geometry stays float64 "
            "on the reference path (section 9.5)",
        )


def _traces(sc: Scenario, geo: Mapping[str, NodeGeometry], rep: GuardReport) -> None:
    """TRACE_COVERAGE: a trace must cover [0, duration] (cubic spline, no extrapolation,
    section 5.10). Compared in float64, as the physics layer evaluates trace times."""
    d = float(sc.run.duration_s)
    for n in sc.nodes:
        g = geo.get(n.id)
        if g is None or g.defined_s is None:
            continue
        t0, t1 = g.defined_s
        if t0 > 0.0 or t1 < d:
            rep.error(
                "TRACE_COVERAGE",
                f"nodes[{n.id}].trajectory",
                f"the trace covers [{t0:.9g}, {t1:.9g}] s but the run needs positions on "
                f"[0, {d:.9g}] s; traces are not extrapolated (section 5.10)",
                GeometryError,
            )


def _disk(sc: Scenario, rep: GuardReport) -> float:
    total = 0.0
    if sc.record.iq:
        for _sid, (_n, s) in sc.streams().items():
            total += float(s.rate_sps * sc.run.duration_s) * 8.0  # complex64
        if total > DISK_WARN_BYTES:
            rep.warn(
                "RECORDER_DISK",
                "record.iq",
                f"IQ recording needs about {total / 1e9:.3g} GB (> 10 GB)",
            )
    return total


_LN10_20: Final[float] = math.log(10.0) / 20.0


def _passive_tol(lib: StateLibrary, sigma_mag_db: float | None) -> float:
    """Passivity margin above |Gamma| = 1 (section 5.5): 1e-9 for analytic, full-wave,
    circuit or assumed tables; ``3 sigma_mag`` for measured or digitized ones, with the
    linear magnitude uncertainty ``sigma_mag = (ln10/20) sigma_mag_db`` at |Gamma| = 1 (the
    same dB-to-linear conversion as the section 5.5 reciprocity tolerance, and the bound
    ``physics.response.ResponseTable`` enforces at load)."""
    if lib.provenance.source_kind in ("measured", "digitized"):
        sig = lib.provenance.sigma_mag_db if sigma_mag_db is None else sigma_mag_db
        return 3.0 * _LN10_20 * sig  # literally 1 + 3 sigma_mag: no floor when sigma = 0
    return 1e-9


def _tables(
    sc: Scenario, res: ResolvedScenario, lks: list[tuple[str, str]], rep: GuardReport
) -> None:
    streams = sc.streams()
    linked_tx = list(dict.fromkeys(t for t, _ in lks))
    for p in sc.ris:
        where = f"ris[{p.id}].states"
        if p.id in res.library_errors:
            rep.error("STATES", where, res.library_errors[p.id], SchemaError)
            continue
        lib = res.libraries.get(p.id)
        if lib is None:
            rep.error("STATES", where, "state library not resolved", SchemaError)
            continue
        for s in lib.states:
            for g in s.gamma:
                assert g.mag is not None  # assume_mag applied at resolution
                if g.mag > 1.0 + _passive_tol(lib, g.sigma_mag_db):
                    rep.error(
                        "TABLE_PASSIVITY",
                        where,
                        f"state {s.label}: |Gamma| = {g.mag:g} > 1 (passivity)",
                        PassivityError,
                    )
        if (
            p.states.efficiency != 1.0
            and lib.provenance.source_kind in ("measured", "digitized")
            and not p.states.override_efficiency
        ):
            rep.error(
                "EFFICIENCY_REFUSED",
                where,
                f"efficiency {p.states.efficiency:g} on a {lib.provenance.source_kind} table "
                "(the measurement already contains the loss); set override_efficiency: true "
                "to force it (section 5.6)",
                TableValidationError,
            )
        if lib.n_states > _MAX_SEPARABLE_FREE_STATES and not lib.is_separable():
            rep.error(
                "TABLE_STATES",
                where,
                f"{lib.n_states} states need a separable (angle-independent) table",
                TableValidationError,
            )
        s_max = lib.n_states
        init = [p.initial.fill] if p.initial.fill is not None else list(p.initial.states or [])
        bad = sorted({v for v in init if v >= s_max})
        if bad:
            rep.error(
                "INITIAL_STATE",
                f"ris[{p.id}].initial",
                f"states {bad} outside [0, {s_max}) of library {lib.name}",
                SchemaError,
            )
        # Disabled panels are checked too: ``enabled`` is only the initial state (a timed
        # ``ris.set_enabled`` event may switch the panel on, section 4.6), and the physics
        # model validates every panel's table at load.
        for tx_id in linked_tx:
            tx = streams[tx_id][1]
            assert tx.occupied_bw_hz is not None  # resolved
            if lib.validity_bw_hz is not None and tx.occupied_bw_hz > lib.validity_bw_hz:
                rep.error(
                    "TABLE_BANDWIDTH",
                    f"{where} / {tx_id}",
                    f"occupied band {tx.occupied_bw_hz:g} Hz exceeds the table's "
                    f"validity_bw_hz {lib.validity_bw_hz:g} Hz (release 1 has no Gamma FIR)",
                    BandwidthExceeded,
                )
            if lib.frequency_hz == "any":
                continue
            f = list(lib.frequency_hz)
            if len(f) == 1:
                half = (lib.validity_bw_hz or 0.0) / 2.0
                ok = abs(tx.fc_hz - f[0]) <= half
                dom = f"{f[0]:g} Hz +/- validity_bw_hz/2"
            else:
                ok = f[0] <= tx.fc_hz <= f[-1]
                dom = f"[{f[0]:g}, {f[-1]:g}] Hz"
            if not ok:
                rep.error(
                    "TABLE_DOMAIN",
                    f"{where} / {tx_id}",
                    f"carrier {tx.fc_hz:g} Hz is outside library {lib.name} ({dom})",
                    OutOfDomain,
                )


def _pitch(sc: Scenario, lks: list[tuple[str, str]], rep: GuardReport) -> None:
    streams = sc.streams()
    fcs = [streams[t][1].fc_hz for t, _ in lks] or [s.fc_hz for _, s in streams.values()]
    lam_min = _lambda_m(max(fcs))
    limit = sc.numerics.point_max_pitch_wavelengths * lam_min
    for p in sc.ris:
        pitch = max(p.grid.pitch_u_m, p.grid.pitch_v_m)
        if p.grid.cell_model == "point" and pitch > limit:
            rep.warn(
                "POINT_PITCH",
                f"ris[{p.id}].grid",
                f"cell_model 'point' forced with pitch {pitch:g} m > "
                f"{sc.numerics.point_max_pitch_wavelengths:g} lambda = {limit:.4g} m "
                "(section 5.4)",
            )


def _t_u(sc: Scenario, auto_t_u_s: Fraction | None, rep: GuardReport) -> None:
    if auto_t_u_s is None or sc.numerics.t_u_s is None:
        return
    if sc.numerics.t_u_s > auto_t_u_s:
        rep.error(
            "T_U_ABOVE_AUTO",
            "numerics.t_u_s",
            f"t_u_s = {exact_to_str(sc.numerics.t_u_s)} s is above the automatic "
            f"T_u = {exact_to_str(auto_t_u_s)} s; it may only lower it (section 4.4)",
            ConfigurationError,
        )


def check_scenario(
    res: ResolvedScenario, *, auto_t_u_s: Fraction | None = None, derived: bool = True
) -> GuardReport:
    """Run every scenario-level guard on a resolved scenario (section 9.5).

    ``auto_t_u_s``: the automatic snapshot interval from the T_u policy (WP5), if known;
    ``numerics.t_u_s`` above it is an error. ``derived`` also fills ``report.derived``.
    """
    sc = res.scenario
    rep = GuardReport()
    _ids(sc, rep)
    _refs(sc, rep)
    _coupling(sc, rep)
    lks, mismatch = links(sc)
    for msg in mismatch:
        rep.error("RATE_MISMATCH", "nodes", msg, RateMismatchError)
    if not lks and not mismatch:
        rep.warn("NO_LINKS", "nodes", "no TX/RX stream pair forms a link (section 3.2)")
    for sid in res.defaulted_bw:
        bw = sc.streams()[sid][1].occupied_bw_hz
        rep.warn(
            "OCCUPIED_BW_UNDECLARED",
            sid,
            f"occupied_bw_hz not declared; the default 0.8*f_s = {bw:g} Hz is used (4.5)",
        )
    _bandwidth(sc, rep)
    geo: dict[str, NodeGeometry] = {}
    for n in sc.nodes:
        try:
            geo[n.id] = node_geometry(n, sc.run.duration_s, res.base_dir)
        except SchemaError as exc:  # malformed trace file
            rep.error("TRACE_FILE", f"nodes[{n.id}].trajectory", str(exc), SchemaError)
    _traces(sc, geo, rep)
    _distances(sc, geo, rep)
    _long_paths(sc, geo, lks, rep)
    _disk(sc, rep)
    _tables(sc, res, lks, rep)
    _pitch(sc, lks, rep)
    _t_u(sc, auto_t_u_s, rep)
    if derived and not rep.errors:
        rep.derived = derived_quantities(res, geo=geo, lks=lks)
    return rep


# =========================================================================== derived


def _t_curv_s(eps: float, r_min: float, fc_hz: float, v: float) -> float:
    """``sqrt(8 eps r_min/(k v**2))`` (section 4.4) for the chord budget ``eps``; inf for
    v = 0. The caller passes ``eps = eps_rel/2`` (lead decision D1)."""
    if v <= 0.0:
        return math.inf
    k = 2.0 * math.pi * fc_hz / C0
    return math.sqrt(8.0 * eps * r_min / (k * v * v))


def derived_quantities(
    res: ResolvedScenario,
    *,
    geo: Mapping[str, NodeGeometry] | None = None,
    lks: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Numbers printed by ``sremu validate`` (section 9.5).

    Per stream: wavelength, page size (RX). Per panel: size, cell count, Fraunhofer
    distance ``2 D**2/lambda`` (D = longest side) per linked carrier. Per link: maximum
    Doppler bound ``(v_tx + v_rx)/lambda``, the narrowband metric ``pi*B*dtau`` from the
    t = 0 geometry (LOS plus every visible RIS cell of every panel, disabled ones included
    because ``ris.set_enabled`` may switch them on; walls, ground and occluders ignored),
    the FD row, and a conservative T_curv bound ``sqrt(8 (eps_rel/2) r_min/(k v**2))`` (D1:
    the chord rule gets half of eps_rel) with ``v = v_tx + v_rx`` (>= v_perp) and
    ``r_min`` a lower bound on the smallest node-node and node-cell distance over the run:
    the node-node term is :func:`set_distance_lower_bound` of the two reachable sets, and
    both nodes' trace sampling tolerances are subtracted. Per RX: noise floor and
    ``tau_RIS->r`` in samples per panel.
    """
    sc = res.scenario
    if geo is None:
        geo = {n.id: node_geometry(n, sc.run.duration_s, res.base_dir) for n in sc.nodes}
    if lks is None:
        lks, _ = links(sc)
    streams = sc.streams()
    out: dict[str, Any] = {"streams": {}, "panels": {}, "links": {}, "receivers": {}}
    for sid, (_n, s) in streams.items():
        row: dict[str, Any] = {"lambda_m": _lambda_m(s.fc_hz), "rate_sps": exact_to_str(s.rate_sps)}
        if s.direction == "rx":
            row["page_size"] = page_size(s.rate_sps, sc.numerics.page_target_s)
        out["streams"][sid] = row
    linked_fc = sorted({streams[t][1].fc_hz for t, _ in lks})
    cells = {p.id: cell_centres(p) for p in sc.ris}
    for p in sc.ris:
        su, sv = panel_extent_m(p)
        dmax = max(su, sv)
        out["panels"][p.id] = {
            "n_cells": p.n_cells,
            "n_controls": p.n_controls,
            "size_m": [su, sv],
            "fraunhofer_m": {f"{fc:g}": 2.0 * dmax * dmax / _lambda_m(fc) for fc in linked_fc},
            "library": res.libraries[p.id].name if p.id in res.libraries else None,
        }
    for tx_id, rx_id in lks:
        tx_node, tx = streams[tx_id]
        rx_node = streams[rx_id][0]
        gt, gr = geo[tx_node.id], geo[rx_node.id]
        lam = _lambda_m(tx.fc_hz)
        v = None
        if gt.speed_max_mps is not None and gr.speed_max_mps is not None:
            v = gt.speed_max_mps + gr.speed_max_mps
        row = {
            "fd_max_hz_bound": None if v is None else v / lam,
            "fd": fd_choice(float(tx.occupied_bw_hz or 0.0), tx.rate_sps),
        }
        if gt.initial is not None and gr.initial is not None:
            a, b = gt.initial, gr.initial
            delays = [float(np.linalg.norm(a - b)) / C0]
            for p in sc.ris:
                n_hat = panel_axes(p)[0]
                c = cells[p.id]
                vis = ((a - c) @ n_hat > 0) & ((b - c) @ n_hat > 0)
                if np.any(vis):
                    r = np.linalg.norm(a - c[vis], axis=1) + np.linalg.norm(b - c[vis], axis=1)
                    delays.extend((r / C0).tolist())
            env = sc.environment.diffuse
            if env is not None and env.taps:
                delays.append(delays[0] + max(t.delay_s for t in env.taps))
            dtau = max(delays) - min(delays)
            metric = math.pi * float(tx.occupied_bw_hz or 0.0) * dtau
            row.update(
                delay_spread_s=dtau,
                narrowband_metric_rad=metric,
                narrowband=metric <= sc.numerics.narrowband_phase_rad,
            )
        if v:
            tol = gt.tol_m + gr.tol_m
            dists = [set_distance_lower_bound(gt, gr)]
            for p in sc.ris:
                dists += [gt.min_distance_to(cells[p.id]), gr.min_distance_to(cells[p.id])]
            r_min = max(min(dists) - tol, 1e-12)
            # D1: the chord (curvature) rule gets half of the shared budget eps_rel
            t_c = _t_curv_s(sc.numerics.interp_rel_error / 2.0, r_min, tx.fc_hz, v)
            row["r_min_m"] = r_min
            row["t_curv_s_bound"] = t_c
            row["t_u0_s_estimate"] = min(t_c, float(sc.numerics.t_u_max_s))
        out["links"][f"{tx_id}->{rx_id}"] = row
    for sid, (node, s) in streams.items():
        if s.direction != "rx":
            continue
        g = geo[node.id]
        row = {"noise_floor_dbm": thermal_noise_dbm(float(s.rate_sps), s.noise_figure_db)}
        if g.initial is not None:
            row["tau_ris_samples"] = {
                p.id: float(np.linalg.norm(np.asarray(p.pose.center_m) - g.initial))
                / C0
                * float(s.rate_sps)
                for p in sc.ris
            }
        out["receivers"][sid] = row
    out["recorder_bytes"] = _disk(sc, GuardReport())
    return out
