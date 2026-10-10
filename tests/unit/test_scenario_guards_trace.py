"""Trace trajectories in the scenario guards (DESIGN.md 5.10, 9.5).

A trace is the C2 cubic spline through its rows with no extrapolation (section 5.10). The
guards sample that spline (not the polyline through the rows) for CELL_TOO_CLOSE /
CELL_NEAR / LONG_PATH, and TRACE_COVERAGE requires the trace to cover [0, duration].

Oracles:

* **Closed-form polynomials.** The not-a-knot cubic spline through >= 4 samples of a
  polynomial of degree <= 3 is that polynomial (a cubic satisfies every not-a-knot
  condition and the interpolant is unique), so the guarded trajectory is known exactly
  and its distances, extrema and speeds are evaluated here in closed form, without scipy.
* **The physics trace.** For a general trace, the guard's distances must equal brute-force
  distances of ``sremu.physics.geometry.TraceTrajectory`` (the engine's evaluator) on a
  dense grid, to within the guard's stated chord tolerance.
* **Brute force** for the distance helpers (all pairs).
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sremu.errors import GeometryError, SchemaError, ValidityWarning
from sremu.scenario.guards import (
    TRACE_CHORD_TOL_M,
    GuardReport,
    NodeGeometry,
    cell_centres,
    check_scenario,
    max_pair_distance,
    node_geometry,
    read_trace,
    set_distance_lower_bound,
    trace_geometry,
)
from sremu.scenario.load import dump_yaml, load_scenario, parse_scenario, read_document
from sremu.scenario.load import resolve_scenario as _resolve
from sremu.units import C0

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SCEN = REPO / "scenarios"
LAB_DOC = read_document(SCEN / "lab_like_A.yaml")
QUICK_DOC = read_document(SCEN / "quickstart.yaml")
LAM = C0 / 3.5e9


def write_trace(path: Path, t: Any, xyz: Any, header: str = "t,x,y,z") -> str:
    """Write a trace CSV (repr floats, so the file holds the exact float64 values)."""
    rows = [header] + [
        ",".join(repr(float(v)) for v in (ti, *p)) for ti, p in zip(t, xyz, strict=True)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path.name


def with_trace(doc: dict[str, Any], node: int, file: str) -> dict[str, Any]:
    d = copy.deepcopy(doc)
    d["nodes"][node].pop("position_m", None)
    d["nodes"][node]["trajectory"] = {"model": "trace", "file": file, "data_class": "SIM-sremu"}
    return d


def report(doc: dict[str, Any], base: Path) -> GuardReport:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        res = _resolve(parse_scenario(doc), base, warn=False, strict=False)
    return check_scenario(res)


# --------------------------------------------------------------------------- CELL_TOO_CLOSE


def _dip_trace(c: float, y: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows at t = 0, 2, 4, 6 of x(t) = c + 0.3 (t-1)^2 + 0.01 (t-1)^3 (y, z fixed).

    On [0, 5], x(t) - c = (t-1)^2 (0.3 + 0.01 (t-1)) >= 0 with equality only at t = 1, so the
    closest approach to the panel plane x = 0 is exactly ``c`` at t = 1 -- between rows,
    whose own x values are all >= c + 0.29."""
    t = np.array([0.0, 2.0, 4.0, 6.0])
    x = c + 0.3 * (t - 1) ** 2 + 0.01 * (t - 1) ** 3
    return t, np.column_stack([x, np.full(4, y), np.full(4, z)])


@pytest.mark.parametrize(
    ("c_over_lambda", "code", "severity"),
    [(0.99, "CELL_TOO_CLOSE", "error"), (1.01, "CELL_NEAR", "warning"), (3.01, None, None)],
)
def test_s9_5_trace_spline_dip_between_rows_is_guarded(tmp_path, c_over_lambda, code, severity):
    d = copy.deepcopy(LAB_DOC)  # duration 5 s; panel at the origin, normal +x
    corner = cell_centres(parse_scenario(d).ris[0])[0]
    c = c_over_lambda * LAM
    t, xyz = _dip_trace(c, float(corner[1]), float(corner[2]))
    # teeth: the polyline through the rows never comes within 3 lambda of any cell
    tt = np.linspace(0.0, 5.0, 50_001)
    lin = np.column_stack([np.interp(tt, t, xyz[:, k]) for k in range(3)])
    cells = cell_centres(parse_scenario(d).ris[0])
    lin_min = min(np.linalg.norm(cells - p, axis=1).min() for p in lin[::50])
    assert lin_min > 3 * LAM
    # closed-form oracle: the spline is the cubic, closest approach c to the corner cell
    xc = c + 0.3 * (tt - 1) ** 2 + 0.01 * (tt - 1) ** 3
    assert xc.min() == pytest.approx(c, abs=1e-12)
    d = with_trace(d, 2, write_trace(tmp_path / "dip.csv", t, xyz))
    rep = report(d, tmp_path)
    cell_codes = {"CELL_TOO_CLOSE", "CELL_NEAR"} & rep.codes()
    if code is None:
        assert not cell_codes, rep.issues
        assert not rep.errors
    else:
        assert cell_codes == {code}, rep.issues
        assert code in rep.codes(severity)
        issue = next(i for i in rep.issues if i.code == code)
        assert issue.where == "ris[ris0] / nodes[rx2]"
        reported = float(issue.message.split(" m from")[0].split()[-1])
        assert reported == pytest.approx(c, rel=1e-3)  # the message prints 4 digits
    if code == "CELL_TOO_CLOSE":
        with pytest.raises(GeometryError):
            rep.raise_for_errors()


def test_s9_5_trace_distance_is_conservative_by_the_chord_bound(tmp_path):
    """node_geometry of the dip trace: polyline distance minus tol_m is a lower bound of the
    exact closest approach c, and within tol_m + 1e-12 of it."""
    sc = parse_scenario(LAB_DOC)
    corner = cell_centres(sc.ris[0])[0]
    c = 0.5
    t, xyz = _dip_trace(c, float(corner[1]), float(corner[2]))
    g = trace_geometry(t, xyz, Fraction(5))
    assert 0.0 < g.tol_m <= TRACE_CHORD_TOL_M
    dmin = g.min_distance_to(cells := cell_centres(sc.ris[0]))
    assert dmin - g.tol_m <= c + 1e-12
    assert abs(dmin - c) <= g.tol_m + 1e-12
    assert dmin == pytest.approx(float(g.min_distance(cells).min()), abs=1e-15)
    assert g.defined_s == (0.0, 6.0)
    np.testing.assert_allclose(g.initial, [c + 0.29, corner[1], corner[2]], rtol=0, atol=1e-12)
    # the samples cover exactly the run window [0, 5]
    x5 = c + 0.3 * 16 + 0.01 * 64
    np.testing.assert_allclose(g.pts[-1], [x5, corner[1], corner[2]], rtol=0, atol=1e-12)


def test_s5_10_trace_speed_bound_closed_form():
    """max |p'| of x(t) = c + 0.3 (t-1)^2 + 0.01 (t-1)^3 on [0, 5] is x'(5) = 2.88 m/s; the
    guard's bound is >= the true maximum and tight to 0.1 %."""
    t, xyz = _dip_trace(0.5, 0.0, 0.0)
    g = trace_geometry(t, xyz, Fraction(5))
    assert g.speed_max_mps is not None
    assert 2.88 <= g.speed_max_mps <= 2.88 * (1 + 1e-3)


# --------------------------------------------------------------------------- LONG_PATH


def _bulge_rows(amp: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows at t = 0, 0.2, 1.8, 2 of x(t) = 600 + amp (1 - (t-1)^2), y = 300, z = 0: the
    rows reach x <= 600 + 0.36 amp, the spline (the quadratic itself) x = 600 + amp."""
    t = np.array([0.0, 0.2, 1.8, 2.0])
    x = 600.0 + amp * (1.0 - (t - 1.0) ** 2)
    return t, np.column_stack([x, np.full(4, 300.0), np.zeros(4)])


@pytest.mark.parametrize("amp", [300.0, 500.0])
def test_s9_5_trace_spline_overshoot_long_path(tmp_path, amp):
    d = copy.deepcopy(QUICK_DOC)
    d["run"]["duration_s"] = "2"
    t, xyz = _bulge_rows(amp)
    d = with_trace(d, 1, write_trace(tmp_path / "bulge.csv", t, xyz))
    sc = parse_scenario(d)
    ap = np.array(sc.nodes[0].position_m)
    centre = np.array(sc.ris[0].pose.center_m, dtype=float)
    # closed-form oracle over the run: LOS and RIS-hop lengths of the quadratic path
    tt = np.linspace(0.0, 2.0, 200_001)
    path = np.column_stack(
        [600.0 + amp * (1.0 - (tt - 1.0) ** 2), np.full(tt.size, 300.0), np.zeros(tt.size)]
    )
    worst = max(
        np.linalg.norm(path - ap, axis=1).max(),
        np.linalg.norm(ap - centre) + np.linalg.norm(path - centre, axis=1).max(),
    )
    rows_worst = max(
        np.linalg.norm(xyz - ap, axis=1).max(),
        np.linalg.norm(ap - centre) + np.linalg.norm(xyz - centre, axis=1).max(),
    )
    assert rows_worst < 1000.0  # teeth: the rows alone never exceed 1 km
    rep = report(d, tmp_path)
    assert ("LONG_PATH" in rep.codes("warning")) is bool(worst > 1000.0)
    assert (amp == 500.0) is bool(worst > 1000.0)  # the two cases straddle the threshold
    if amp == 500.0:
        msg = next(i.message for i in rep.issues if i.code == "LONG_PATH")
        assert float(msg.split("up to ")[1].split(" m")[0]) == pytest.approx(worst, rel=1e-3)


# --------------------------------------------------------------------------- coverage


@pytest.mark.parametrize(
    ("t_rows", "duration", "covered"),
    [
        ((0.0, 5.0), "5", True),  # exactly [0, duration]
        ((-1.0, 0.0, 7.0), "5", True),
        ((0.0, 0.3), "0.3", True),  # float64 comparison: float(3/10) == 0.3
        ((0.5, 6.0), "5", False),  # starts late
        ((0.0, 4.9), "5", False),  # ends early
        ((1e-9, 5.0), "5", False),
        ((6.0, 7.0), "5", False),  # no overlap at all
        ((-3.0, -1.0), "5", False),
    ],
)
def test_s5_10_trace_must_cover_the_run(tmp_path, t_rows, duration, covered):
    d = copy.deepcopy(LAB_DOC)
    d["run"]["duration_s"] = duration
    pos = np.array(d["nodes"][2]["position_m"], dtype=float)
    xyz = np.tile(pos, (len(t_rows), 1))
    d = with_trace(d, 2, write_trace(tmp_path / "cov.csv", t_rows, xyz))
    rep = report(d, tmp_path)
    assert ("TRACE_COVERAGE" in rep.codes("error")) is (not covered), rep.issues
    if covered:
        assert not rep.errors
    else:
        assert rep.codes("error") == {"TRACE_COVERAGE"}
        with pytest.raises(GeometryError, match="not extrapolated"):
            rep.raise_for_errors()
        f = tmp_path / "s.yaml"
        f.write_text(dump_yaml(d), encoding="utf-8")
        with pytest.raises(GeometryError, match="TRACE_COVERAGE"):
            load_scenario(f, warn=False)


# --------------------------------------------------------------------------- file format


TRACE_FORMAT_CASES = [
    ("t,x,y,z\n0,1,2,3\n5,1,2,4\n", True),
    ("# t, x, y, z\n0,1,2,3\n# comment\n\n5,1,2,4\n", True),
    (" t , x , y , z \n0, 1, 2, 3\n5,1,2,4\n", True),  # spaces around names and values
    ("\n\nt,x,y,z,speed_mps\n0,1,2,3,9\n5,1,2,4,9\n", True),  # extra column ignored
    ("z,y,x,t\n3,2,1,0\n4,2,1,5\n", True),  # columns found by name
    ("t_s,x_m,y_m,z_m\n0,1,2,3\n5,1,2,4\n", False),  # 5.10 names are t, x, y, z
    ("T,X,Y,Z\n0,1,2,3\n5,1,2,4\n", False),  # case-sensitive
    ("0,1,2,3\n5,1,2,4\n", False),  # no header
    ("t,x,y\n0,1,2\n5,1,2\n", False),  # missing column
    ("t,t,x,y,z\n0,0,1,2,3\n5,5,1,2,4\n", False),  # ambiguous time column
    ("t,x,y,z\n0,1,2,3\n", False),  # one row
    ("t,x,y,z\n0,1,2,3\n5,1,nan,4\n", False),  # non-finite
    ("t,x,y,z\n0,1,2,3\n0,1,2,4\n", False),  # not strictly increasing
    ("t,x,y,z\n0,1,2,3\n5,1,two,4\n", False),  # bad number
    ("t,x,y,z\n0,1,2,3\n5,1,2\n", False),  # short row
    ("t,x,y,z\n0,1,2,3\n5,1,2,4,7\n", False),  # long row
    ("t,x,y,z\n0,1,2,3\n5,1,2,\n", False),  # empty field
    ("", False),
]


@pytest.mark.parametrize(("text", "ok"), TRACE_FORMAT_CASES)
def test_s5_10_trace_file_format(tmp_path, text, ok):
    f = tmp_path / "tr.csv"
    f.write_text(text, encoding="utf-8")
    if ok:
        t, xyz = read_trace(f)
        np.testing.assert_array_equal(t, [0.0, 5.0])
        np.testing.assert_array_equal(xyz, [[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]])
    else:
        with pytest.raises(SchemaError):
            read_trace(f)


@pytest.mark.parametrize("text", [text for text, ok in TRACE_FORMAT_CASES if ok])
def test_s5_10_guard_trace_format_is_no_wider_than_the_engine_reader(tmp_path, text):
    """Every trace the guard accepts loads in the engine's reader
    (``physics.geometry.TraceTrajectory.from_csv``) with the same rows, so ``sremu validate``
    never passes a trace that the run then rejects. (The guard may be stricter.)"""
    from sremu.physics.geometry import TraceTrajectory

    f = tmp_path / "tr.csv"
    f.write_text(text, encoding="utf-8")
    t, xyz = read_trace(f)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a reader warning would be a disagreement too
        phys = TraceTrajectory.from_csv(f)
    np.testing.assert_array_equal(phys.t_s, t)
    np.testing.assert_array_equal(phys.xyz_m, xyz)


def test_s9_5_malformed_trace_is_a_guard_error_not_a_crash(tmp_path):
    (tmp_path / "bad.csv").write_text("0,1,2,3\n5,1,2,4\n", encoding="utf-8")  # no header
    d = with_trace(copy.deepcopy(LAB_DOC), 2, "bad.csv")
    rep = report(d, tmp_path)
    assert rep.codes("error") == {"TRACE_FILE"}
    issue = rep.errors[0]
    assert issue.where == "nodes[rx2].trajectory" and "header" in issue.message
    with pytest.raises(SchemaError):
        rep.raise_for_errors()


# --------------------------------------------------------------------------- vs physics


def test_s5_10_guard_spline_is_the_physics_trace_spline(tmp_path):
    """The guard samples the same C2 spline the engine evaluates: brute-force distances and
    speeds of ``physics.geometry.TraceTrajectory`` on a dense grid agree with the guard's
    within its chord tolerance (rows irregular in time, non-polynomial motion)."""
    from sremu.physics.geometry import TraceTrajectory

    t = np.array([-0.3, 0.0, 0.17, 0.5, 0.71, 1.2, 1.35, 1.9, 2.4, 2.6])
    xyz = np.column_stack(
        [0.4 + 0.3 * np.sin(3.1 * t), -0.2 + 0.25 * np.cos(2.3 * t) * t, 0.05 * t**2]
    )
    f = tmp_path / "tr.csv"
    write_trace(f, t, xyz)
    phys = TraceTrajectory.from_csv(f)
    duration = Fraction(2)
    rt, rxyz = read_trace(f)
    g = trace_geometry(rt, rxyz, duration)
    assert g.tol_m <= TRACE_CHORD_TOL_M
    h = 1e-5  # grid step, s; speeds < 2 m/s, so the grid minimum is exact to ~1e-10 m
    tt = (np.arange(200_003) - 1) * h  # [-h, 2 + h]: central differences cover [0, 2]
    p = phys.positions(tt)
    sc = parse_scenario(LAB_DOC)
    for q in (cell_centres(sc.ris[0]), np.array([[0.4, -0.2, 0.0]]), np.array([[3.0, 2, 1.0]])):
        brute = min(float(np.linalg.norm(p[1:-1] - qi, axis=1).min()) for qi in q)
        assert abs(g.min_distance_to(q) - brute) <= g.tol_m + 1e-9
    np.testing.assert_allclose(g.initial, phys.position(Fraction(0)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(g.pts[-1], phys.position(duration), rtol=0, atol=1e-12)
    # speed oracle: central differences of the physics positions (error O(h^2 |p'''|)). The
    # guard's bound max|p'(samples)| + max (h_s/2) A has slack <= sqrt(2 tol A_max), since
    # its sample step obeys h_s^2 A/8 <= tol on every piece.
    v = float(np.linalg.norm((p[2:] - p[:-2]) / (2 * h), axis=1).max())
    a_max = float(np.linalg.norm((p[2:] - 2 * p[1:-1] + p[:-2]) / h**2, axis=1).max())
    slack = math.sqrt(2 * TRACE_CHORD_TOL_M * a_max)
    assert g.speed_max_mps is not None
    assert v - 1e-6 <= g.speed_max_mps <= v + slack * (1 + 1e-3) + 1e-6


def test_s5_10_node_geometry_reads_the_trace_relative_to_the_scenario(tmp_path):
    t, xyz = _dip_trace(0.5, 0.1, 0.2)
    write_trace(tmp_path / "rel.csv", t, xyz, header="# t, x, y, z")
    d = with_trace(copy.deepcopy(LAB_DOC), 2, "rel.csv")
    node = parse_scenario(d).nodes[2]
    g = node_geometry(node, Fraction(5), tmp_path)
    assert g.defined_s == (0.0, 6.0) and g.kind == "points" and len(g.pts) > 100


# --------------------------------------------------------------------------- distance helpers


def _brute_point_segment(q: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Per-query minimum distance to a polyline, one query and one segment at a time."""
    out = np.empty(len(q))
    for i, x in enumerate(q):
        best = math.inf
        for a, b in itertools.pairwise(pts):
            ab = b - a
            den = float(ab @ ab)
            s = 0.0 if den == 0.0 else min(1.0, max(0.0, float((x - a) @ ab) / den))
            best = min(best, float(np.linalg.norm(x - (a + s * ab))))
        out[i] = best
    return out


def test_min_distance_helpers_match_brute_force(rng):
    gen = rng  # keyed per-test generator (conftest)
    pts = np.cumsum(gen.normal(size=(400, 3)), axis=0)
    pts[57] = pts[56]  # a zero-length segment
    g = NodeGeometry("points", pts, pts[0], 1.0)
    for q in (gen.normal(size=(40, 3)) * 10, pts[::37] + 1e-3, np.array([[1e3, 0.0, 0.0]])):
        brute = _brute_point_segment(q, pts)
        np.testing.assert_allclose(g.min_distance(q), brute, rtol=1e-12, atol=1e-12)
        assert g.min_distance_to(q) == pytest.approx(brute.min(), rel=1e-12, abs=1e-12)
    single = NodeGeometry("points", pts[:1], pts[0], 0.0)
    assert single.min_distance_to(pts[1:3]) == pytest.approx(
        np.linalg.norm(pts[1:3] - pts[0], axis=1).min(), rel=1e-15
    )


@pytest.mark.parametrize("shape", ["3d", "plane", "line", "small"])
def test_max_pair_distance_matches_brute_force(rng, shape):
    gen = rng
    a, b = gen.normal(size=(300, 3)), gen.normal(size=(200, 3)) + 5.0
    if shape == "plane":
        a[:, 2], b[:, 2] = 0.3, -1.0
    elif shape == "line":
        a = gen.random((300, 1)) * np.array([1.0, 2.0, 3.0])
        b = gen.random((200, 1)) * np.array([-1.0, 0.0, 1.0]) + 2.0
    elif shape == "small":
        a, b = a[:3], b[:1]
    brute = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)).max()
    assert max_pair_distance(a, b) == pytest.approx(brute, rel=1e-12)


def _seg_seg(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Exact distance between segments [p0, p1] and [q0, q1] (independent oracle).

    |p(s) - q(t)|^2 is a convex quadratic on [0, 1]^2: its minimum is the unconstrained
    stationary point when that lies inside, else on an edge, where it is a point-to-segment
    distance of one of the four end points."""

    def pt_seg(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        ab = b - a
        den = float(ab @ ab)
        s = 0.0 if den == 0.0 else min(1.0, max(0.0, float((x - a) @ ab) / den))
        return float(np.linalg.norm(x - (a + s * ab)))

    best = min(pt_seg(p0, q0, q1), pt_seg(p1, q0, q1), pt_seg(q0, p0, p1), pt_seg(q1, p0, p1))
    u, v, w = p1 - p0, q1 - q0, p0 - q0
    a, b, c, d, e = u @ u, u @ v, v @ v, u @ w, v @ w
    det = a * c - b * b
    if det > 1e-14 * a * c:
        s, t = (b * e - c * d) / det, (a * e - b * d) / det
        if 0.0 < s < 1.0 and 0.0 < t < 1.0:
            best = min(best, float(np.linalg.norm(w + s * u - t * v)))
    return best


def _polyline_distance(pa: np.ndarray, pb: np.ndarray) -> float:
    return min(
        _seg_seg(a0, a1, b0, b1)
        for a0, a1 in itertools.pairwise(pa)
        for b0, b1 in itertools.pairwise(pb)
    )


def test_sample_cover_covers_the_polyline(rng):
    pts = np.cumsum(rng.normal(size=(30, 3)), axis=0)
    pts[7] = pts[6]  # a zero-length segment
    g = NodeGeometry("points", pts, pts[0], 1.0)
    u = np.linspace(0.0, 1.0, 2001)[:, None]
    for cover in (0.5, 0.05):
        s, c = g.sample_cover(cover)
        assert 0.0 < c <= cover
        # every vertex is a sample, and every sample lies on the polyline
        assert all(np.linalg.norm(s - p, axis=1).min() == 0.0 for p in pts)
        assert float(g.min_distance(s).max()) <= 1e-12
        # every polyline point is within c of a sample (dense check, segment by segment)
        worst = 0.0
        for a, b in itertools.pairwise(pts):
            dense = a + u * (b - a)
            near = np.linalg.norm(dense[:, None, :] - s[None, :, :], axis=2).min(axis=1)
            worst = max(worst, float(near.max()))
        assert worst <= c + 1e-12
    s1, c1 = NodeGeometry("points", pts[:1], pts[0], 0.0).sample_cover(0.1)
    assert c1 == 0.0 and np.array_equal(s1, pts[:1])
    with pytest.raises(ValueError):
        NodeGeometry("rect", np.array([[0.0, 0, 0], [1, 1, 0]]), None, 1.0).sample_cover(0.1)


def _cover_bound(g: NodeGeometry) -> float:
    """The documented cover radius bound of set_distance_lower_bound: max(1 mm, L/4000)."""
    return max(1e-3, g.path_length_m() / 4000.0)


def test_set_distance_lower_bound_point_and_rect_cases_are_exact(rng):
    pts = np.cumsum(rng.normal(size=(50, 3)), axis=0)
    line = NodeGeometry("points", pts, pts[0], 1.0)
    q = np.array([[0.3, -2.0, 1.0]])
    point = NodeGeometry("points", q, q[0], 0.0)
    exact = float(_brute_point_segment(q, pts)[0])
    assert set_distance_lower_bound(point, line) == pytest.approx(exact, rel=1e-12)
    assert set_distance_lower_bound(line, point) == pytest.approx(exact, rel=1e-12)
    # rectangles: the nearest points are grid nodes, so a brute-force grid is exact
    ra = NodeGeometry("rect", np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]), None, 1.0)
    cases = [
        ((2.0, 1.5, 3.0, 2.0, 0.5), math.sqrt(1.5)),  # corner (1,1,0) to corner (2,1.5,0.5)
        ((0.5, -1.0, 3.0, 0.5, 0.7), 0.7),  # overlapping footprints: height difference
        ((-3.0, 0.25, -1.0, 0.75, 0.0), 1.0),  # edge to edge
    ]
    for (x0, y0, x1, y1, z), true in cases:
        rb = NodeGeometry("rect", np.array([[x0, y0, z], [x1, y1, z]]), None, 1.0)
        xa, ya = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        xb, yb = np.meshgrid(np.linspace(x0, x1, 9), np.linspace(y0, y1, 9))
        a3 = np.column_stack([xa.ravel(), ya.ravel(), np.zeros(xa.size)])
        b3 = np.column_stack([xb.ravel(), yb.ravel(), np.full(xb.size, z)])
        brute = float(np.sqrt(((a3[:, None] - b3[None]) ** 2).sum(-1)).min())
        assert brute == pytest.approx(true, abs=1e-12)
        assert set_distance_lower_bound(ra, rb) == pytest.approx(true, abs=1e-12)
        assert set_distance_lower_bound(rb, ra) == pytest.approx(true, abs=1e-12)


def test_set_distance_lower_bound_rect_and_polyline():
    """Segment (-1, 0.5, 1) -> (2, 0.5, 0.3) over the rectangle [0,1]^2 at z = 0. With
    s in [0, 1]: x = -1 + 3s, z = 1 - 0.7s. Over x <= 1 the distance is >= z(2/3) = 0.533;
    for x > 1 it is sqrt((3s - 2)^2 + (1 - 0.7s)^2), minimal at s* = 13.4/18.98 (zero
    derivative; s* in (2/3, 1)), which is the global minimum."""
    seg = NodeGeometry("points", np.array([[-1.0, 0.5, 1.0], [2.0, 0.5, 0.3]]), None, 1.0)
    rect = NodeGeometry("rect", np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]), None, 1.0)
    s = 13.4 / 18.98
    true = math.hypot(3 * s - 2, 1 - 0.7 * s)
    assert true < 1 - 0.7 * (2 / 3)
    for lb in (set_distance_lower_bound(seg, rect), set_distance_lower_bound(rect, seg)):
        assert true - _cover_bound(seg) <= lb <= true


@pytest.mark.parametrize("n_seg", [1, 7, 6000])
def test_set_distance_lower_bound_skew_polylines(n_seg):
    """Two straight paths split into n_seg segments: A along x (y = z = 0, x in [0, 10]),
    B along z at x = 5.0001, y = 0.3 (z in [-5, 5]). The exact distance is 0.3 (closest
    points (5.0001, 0, 0) and (5.0001, 0.3, 0), on no vertex). n_seg = 6000 takes the k-d
    tree branch (6001 samples x 6000 segments > 2e7 pairs)."""
    u = np.linspace(0.0, 1.0, n_seg + 1)[:, None]
    pa = np.array([0.0, 0.0, 0.0]) + u * np.array([10.0, 0.0, 0.0])
    pb = np.array([5.0001, 0.3, -5.0]) + u * np.array([0.0, 0.0, 10.0])
    ga = NodeGeometry("points", pa, pa[0], 1.0)
    gb = NodeGeometry("points", pb, pb[0], 1.0)
    true = 0.3
    if n_seg <= 7:
        assert _polyline_distance(pa, pb) == pytest.approx(true, abs=1e-12)
    slack = _cover_bound(ga) + _cover_bound(gb)
    for lb in (set_distance_lower_bound(ga, gb), set_distance_lower_bound(gb, ga)):
        assert true - slack <= lb <= true


def test_set_distance_lower_bound_random_polylines(rng):
    for _ in range(5):
        pa = np.cumsum(rng.normal(size=(6, 3)), axis=0)
        pb = np.cumsum(rng.normal(size=(5, 3)), axis=0) + rng.normal(size=3) * 2
        ga = NodeGeometry("points", pa, pa[0], 1.0)
        gb = NodeGeometry("points", pb, pb[0], 1.0)
        true = _polyline_distance(pa, pb)
        lb = set_distance_lower_bound(ga, gb)
        assert true - _cover_bound(ga) - 1e-12 <= lb <= true + 1e-12


def test_derived_r_min_is_a_lower_bound_for_two_moving_nodes():
    """T_curv's r_min (D1 formula) must not exceed the true node-node distance. The AP
    moves along y at x = 3 (z = 0), the user along z at x = 3.2, y = 0.013 (z from -1.007
    to 0.993): the paths are 0.2 m apart at (3, 0.013, 0) / (3.2, 0.013, 0). Even samples
    (65 per path) miss both points and overestimate the distance (teeth)."""
    pa0, pa1 = np.array([3.0, -1.0, 0.0]), np.array([3.0, 1.0, 0.0])
    pb0, pb1 = np.array([3.2, 0.013, -1.007]), np.array([3.2, 0.013, 0.993])
    d = copy.deepcopy(QUICK_DOC)
    for i, (p0, p1) in enumerate([(pa0, pa1), (pb0, pb1)]):
        d["nodes"][i].pop("position_m")
        d["nodes"][i]["trajectory"] = {
            "model": "waypoints",
            "t_s": ["0", "0.1"],
            "positions_m": [p0.tolist(), p1.tolist()],
        }
    res = _resolve(parse_scenario(d), SCEN, warn=False, strict=False)
    rep = check_scenario(res)
    assert not rep.errors, rep.issues
    row = rep.derived["links"]["ap.tx0->mu.rx0"]
    true = 0.2
    assert _seg_seg(pa0, pa1, pb0, pb1) == pytest.approx(true, abs=1e-12)
    cells = cell_centres(res.scenario.ris[0])
    assert np.linalg.norm(cells - pa0, axis=1).min() > 2.5  # the node-node term is r_min
    assert true - 2e-3 <= row["r_min_m"] <= true
    v = 20.0 + 20.0  # each path is 2 m long, covered in 0.1 s
    k = 2 * math.pi * 3.5e9 / C0
    expect = math.sqrt(8 * (1e-3 / 2) * row["r_min_m"] / (k * v**2))
    assert row["t_curv_s_bound"] == pytest.approx(expect, rel=1e-12)
    # teeth: 65 evenly spaced samples per path (spacing 2/64 m) overestimate the distance
    g = np.linspace(0.0, 1.0, 65)[:, None]
    sa, sb = pa0 + g * (pa1 - pa0), pb0 + g * (pb1 - pb0)
    vertex_only = min(
        float(_brute_point_segment(sa, np.vstack([pb0, pb1])).min()),
        float(_brute_point_segment(sb, np.vstack([pa0, pa1])).min()),
    )
    assert vertex_only > true + 1e-4
