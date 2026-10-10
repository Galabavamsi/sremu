"""Geometry and analytic trajectories (sections 3.1, 5.2, 5.9 segment-polygon, 5.10)."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu import rng as sremu_rng
from sremu.errors import ConfigurationError, GeometryError
from sremu.physics.antennas import CosPower, Isotropic, Tabulated, antenna_from_spec
from sremu.physics.geometry import (
    ConstantVelocity,
    ExternalTrajectory,
    PanelGrid,
    Pose,
    RandomWaypoint1D,
    Static,
    TraceTrajectory,
    Waypoints,
    angles_to_node,
    segment_hits_polygon,
    trajectory_from_spec,
)

pytestmark = pytest.mark.physics


def test_pose_and_signed_table_angles():
    pose = Pose([1.0, 2.0, 0.0], [0, 0, 2.0], [3.0, 0, 0])  # normalized on construction
    assert np.allclose(pose.v_axis, [0, 1, 0])
    with pytest.raises(GeometryError):
        Pose([0, 0, 0], [0, 0, 1], [1, 0, 1])  # not orthogonal
    pts = np.array([[1.0, 2.0, 0.0]])
    # node in the (u, n) plane at +30 deg toward +u: theta = +30 deg, no off-plane angle
    node = np.array([1.0 + math.sin(math.radians(30)), 2.0, math.cos(math.radians(30))])
    a = angles_to_node(pts, node, pose)
    assert math.degrees(a["theta_rad"][0]) == pytest.approx(30.0, abs=1e-12)
    assert a["off_plane_rad"][0] == pytest.approx(0.0, abs=1e-15)
    assert a["cos"][0] == pytest.approx(math.cos(math.radians(30)), abs=1e-15)
    # specular partner: theta_s = -theta_i
    node2 = np.array([1.0 - math.sin(math.radians(30)), 2.0, math.cos(math.radians(30))])
    assert math.degrees(angles_to_node(pts, node2, pose)["theta_rad"][0]) == pytest.approx(-30.0)
    # off-plane component is projected out and reported
    node3 = np.array([1.0, 2.0 + 1.0, 1.0])
    a3 = angles_to_node(pts, node3, pose)
    assert a3["theta_rad"][0] == pytest.approx(0.0, abs=1e-15)
    assert math.degrees(a3["off_plane_rad"][0]) == pytest.approx(45.0, abs=1e-12)


def test_panel_grid_cells_tiles_and_gaps():
    pose = Pose([0, 0, 0], [0, 0, 1], [1, 0, 0])
    g = PanelGrid(pose, 6, 2, 0.1, 0.05, n_tiles_u=3, gap_m=0.002)
    loc = g.cell_local()
    # n = iv * n_u + iu; u spacing jumps by the gap at tile borders
    assert np.allclose(np.diff(loc[:6, 0]), [0.1, 0.102, 0.1, 0.102, 0.1])
    assert g.size_m == pytest.approx((0.6 + 0.004, 0.1))
    assert loc[:, 0].mean() == pytest.approx(0.0, abs=1e-15)
    assert g.physical_tile_of_cell().tolist() == [0, 0, 1, 1, 2, 2] * 2
    assert g.render_tile_of_cell(2, 2).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    strips = g.gap_strips()
    assert len(strips) == 2 and all(abs((s[1] - s[0]) - 0.002) < 1e-15 for s in strips)
    with pytest.raises(ConfigurationError):
        PanelGrid(pose, 5, 2, 0.1, 0.1, n_tiles_u=2)


@pytest.mark.parametrize(
    ("a", "b", "hit"),
    [
        ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), True),  # through the centre
        ((0.9, 0.9, -1.0), (0.9, 0.9, 1.0), True),  # near a corner, inside
        ((1.1, 0.0, -1.0), (1.1, 0.0, 1.0), False),  # outside the square
        ((0.0, 0.0, 0.5), (0.0, 0.0, 2.0), False),  # does not reach the plane
        ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0), False),  # touches only at an endpoint
        ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), False),  # lies in the plane
        ((1.0, 0.0, -1.0), (1.0, 0.0, 1.0), True),  # on the boundary counts as a hit
    ],
)
def test_segment_polygon(a, b, hit):  # type: ignore[no-untyped-def]
    square = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    assert segment_hits_polygon(a, b, square) is hit


def test_trajectories_are_analytic_functions_of_time():
    w = Waypoints(("0", "1/2", "3"), np.array([[0, 0, 0], [1, 0, 0], [1, 5, 0]], float))
    assert np.array_equal(w.position(Fraction(1, 4)), [0.5, 0, 0])
    assert np.allclose(w.position(Fraction(7, 4)), [1, 2.5, 0], atol=1e-15)
    assert np.array_equal(w.position(10), [1, 5, 0])
    assert w.max_speed_mps(3) == pytest.approx(2.0)
    cv = ConstantVelocity(np.array([1.0, 0, 0]), np.array([0, -2.0, 0]))
    assert np.array_equal(cv.position(Fraction(3)), [1, -6, 0])
    assert np.allclose(cv.positions(np.array([0.0, 0.5])), [[1, 0, 0], [1, -1, 0]])
    assert Static(np.ones(3)).is_static and not cv.is_static


def test_rwp_1d_is_pure_in_time_and_keyed():
    def legs(i: int) -> np.random.Generator:
        return sremu_rng.generator(7, "mobility", "mu", counter=i)  # D14

    a = RandomWaypoint1D([0, 0, 0], [10, 0, 0], 1.0, legs)
    b = RandomWaypoint1D([0, 0, 0], [10, 0, 0], 1.0, legs)
    times = np.linspace(0, 200, 4001)
    pa = np.stack([a.position(float(t)) for t in times])
    pb = np.stack([b.position(float(t)) for t in times[::-1]])[::-1]  # reverse query order
    assert np.array_equal(pa, pb)
    step = np.linalg.norm(np.diff(pa, axis=0), axis=1)
    assert np.all(step <= 1.0 * (times[1] - times[0]) + 1e-12)  # no jumps faster than v
    assert np.all((pa[:, 0] >= 0) & (pa[:, 0] <= 10)) and np.all(pa[:, 1:] == 0)
    other = RandomWaypoint1D(
        [0, 0, 0], [10, 0, 0], 1.0, lambda i: sremu_rng.generator(8, "mobility", "mu", counter=i)
    )
    assert not np.array_equal(other.position(50.0), a.position(50.0))
    # waypoint i depends only on (seed, purpose, name, i)
    assert a.waypoint(3)[0] == pytest.approx(
        10 * sremu_rng.generator(7, "mobility", "mu", counter=3).random()
    )


def test_trace_spline_reproduces_cubic_and_refuses_extrapolation(tmp_path: Path):
    t = np.linspace(0, 2, 9)
    xyz = np.stack([t**3 - t, 2 * t**2, np.full_like(t, 1.5)], axis=1)
    f = tmp_path / "trace.csv"
    np.savetxt(f, np.column_stack([t, xyz]), delimiter=",", header="t,x,y,z", comments="")
    tr = TraceTrajectory.from_csv(f)
    tq = 1.37
    assert np.allclose(tr.position(tq), [tq**3 - tq, 2 * tq**2, 1.5], atol=1e-12)
    with pytest.raises(GeometryError):
        tr.position(2.5)


def test_external_trajectory_events_and_spec_builder():
    ext = trajectory_from_spec({"model": "external", "initial_m": [0, 0, 0]}, node="n")
    assert isinstance(ext, ExternalTrajectory)
    ext.set_position(Fraction(1), [1.0, 2.0, 3.0])
    ext.set_trajectory(Fraction(2), ConstantVelocity(np.array([5.0, 0, 0]), np.array([1.0, 0, 0])))
    assert np.array_equal(ext.position(Fraction(1, 2)), [0, 0, 0])
    assert np.array_equal(ext.position(Fraction(3, 2)), [1, 2, 3])
    assert np.array_equal(ext.position(Fraction(5, 2)), [5.5, 0, 0])  # time since the event
    with pytest.raises(ConfigurationError):
        ext.set_position(Fraction(1), [0, 0, 0])  # out of order
    wp = trajectory_from_spec(
        {"model": "waypoints", "t_s": ["0", "1"], "positions_m": [[0, 0, 0], [2, 0, 0]]}
    )
    assert np.array_equal(wp.position(Fraction(1, 2)), [1, 0, 0])
    with pytest.raises(ConfigurationError):
        trajectory_from_spec({"model": "gauss_markov"})


def test_antenna_patterns(tmp_path: Path):
    cp = CosPower(3.0, np.array([0.0, 0.0, 2.0]))
    d = np.array([[0, 0, 1.0], [math.sin(0.3), 0, math.cos(0.3)], [1.0, 0, 0], [0, 0, -1.0]])
    assert np.allclose(cp.gain(d), [8.0, 8.0 * math.cos(0.3) ** 3, 0.0, 0.0])
    assert Isotropic().gain(d).tolist() == [1.0] * 4
    f = tmp_path / "pat.csv"
    rows = ["theta_deg,phi_deg,gain_dbi"] + [
        f"{th},{ph},{10 - 0.1 * th}" for th in (0, 30, 60, 90, 180) for ph in (0, 90, 180, 270)
    ]
    f.write_text("\n".join(rows) + "\n")
    tab = Tabulated.from_csv(
        f,
        {"source_kind": "assumed", "description": "t", "reference": "t"},
        boresight=np.array([0, 0, 1.0]),
        ref_axis=np.array([1.0, 0, 0]),
    )
    g = tab.gain(
        np.array([[0, 0, 1.0], [math.sin(math.radians(45)), 0, math.cos(math.radians(45))]])
    )
    assert np.allclose(10 * np.log10(g), [10.0, 10 - 4.5])
    spec = antenna_from_spec({"pattern": {"cos_power": {"alpha": 1}}, "boresight": [1, 0, 0]})
    assert isinstance(spec, CosPower) and spec.peak_gain == 4.0
