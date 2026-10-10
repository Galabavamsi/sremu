"""PH-11: single knife edge F(nu) (DESIGN.md sections 5.2, 5.9, 10.2).

Spec rows (oracle values from ``tests/oracles/tables.py``): |F(0)| = 0.5 (1e-12); the
ITU-R P.526 J(nu) (oracle only) deviates from the exact loss by at most 0.123 dB on
[-0.78, 10] (<= 0.15 dB); continuity at nu = +-1e-9 (<= 1e-6 dB); nu = 10 gives 32.95 dB
(+-0.01 dB). Further checks use independent forms: numerical quadrature of the Fresnel
integral, the large-nu asymptote (phase consistency) and hand geometry for nu.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import quad

from sremu.contracts import PathKind
from sremu.errors import ConfigurationError, ValidityWarning
from sremu.physics.diffraction import (
    KnifeEdge,
    KnifeEdgeMechanism,
    edges_from_spec,
    knife_edge_F,
    knife_edge_loss_db,
    knife_edge_nu,
    mechanism_from_scenario,
)
from sremu.physics.los import Occluder
from sremu.units import C0
from tests.oracles import rf
from tests.oracles.tables import ORACLE
from tests.physics._wp3_scenes import by_kind, link, two_node

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC


def _check(key: str, value: float) -> None:
    o = ORACLE[key]
    assert abs(value - o.value) <= o.tol, (key, value, o.value, o.tol)


def test_ph11_f0_is_one_half():
    _check("PH11_F0_abs", abs(complex(knife_edge_F(0.0))))
    assert abs(complex(knife_edge_F(0.0)) - 0.5) <= 1e-15  # F(0) = ((1+j)/2)((1-j)/2) = 1/2


def test_ph11_itu_j_is_within_0p15_db_of_exact():
    nu = np.linspace(-0.78, 10.0, 20001)
    dev = float(np.max(np.abs(rf.itu_j_db(nu) - knife_edge_loss_db(nu))))
    _check("PH11_J_max_dev_db", dev)
    assert dev <= 0.15


def test_ph11_continuity_and_nu10_asymptote():
    _check("PH11_continuity_db", float(knife_edge_loss_db(1e-9) - knife_edge_loss_db(-1e-9)))
    _check("PH11_nu10_db", float(knife_edge_loss_db(10.0)))


@pytest.mark.parametrize("nu", [-3.0, -0.78, -0.2, 0.0, 0.4, 1.0, 2.5, 5.0])
def test_ph11_matches_numerical_fresnel_integral(nu: float):
    """F = ((1+j)/2) [ (1-j)/2 - int_0^nu exp(-j pi t^2/2) dt ] with the finite integral by
    adaptive quadrature (independent of scipy.special.fresnel)."""
    re = quad(
        lambda t: math.cos(math.pi * t * t / 2), 0.0, nu, epsabs=1e-14, epsrel=1e-13, limit=400
    )[0]
    im = quad(
        lambda t: -math.sin(math.pi * t * t / 2), 0.0, nu, epsabs=1e-14, epsrel=1e-13, limit=400
    )[0]
    ref = (1 + 1j) / 2 * ((1 - 1j) / 2 - complex(re, im))
    got = complex(knife_edge_F(nu))
    assert abs(got - ref) <= 1e-11
    assert abs(got - complex(rf.knife_edge_F(nu))) <= 1e-15


@pytest.mark.parametrize("nu", [5.0, 10.0, 20.0, 40.0])
def test_ph11_phase_consistent_with_the_extra_path(nu: float):
    """Large nu: F ~ ((1+j)/2) exp(-j pi nu^2/2)/(j pi nu), i.e. |F| = 1/(pi nu sqrt 2) and
    phase -pi nu^2/2 - pi/4 = -k Delta - pi/4 with Delta = nu^2 lambda/4 the excess path, so
    h_LOS F at the LOS delay is a ray of length d + Delta (section 5.9). The next asymptotic
    term is O(1/(pi nu^2))."""
    f = complex(knife_edge_F(nu))
    assert abs(abs(f) * math.pi * nu * math.sqrt(2) - 1) <= 2.0 / (math.pi * nu * nu)
    want = -math.pi * nu * nu / 2 - math.pi / 4
    err = (np.angle(f) - want + math.pi) % (2 * math.pi) - math.pi
    assert abs(err) <= 2.0 / (math.pi * nu * nu)


def test_ph11_vectorized_equals_scalar():
    nu = np.linspace(-5, 15, 101)
    vec = knife_edge_F(nu)
    assert isinstance(vec, np.ndarray) and vec.dtype == np.complex128
    assert all(vec[i] == complex(knife_edge_F(float(x))) for i, x in enumerate(nu))


# --------------------------------------------------------------------------- geometry


def _edge(x0: float, z: float, half: float = 3.0, eid: str = "e") -> KnifeEdge:
    return KnifeEdge(eid, np.array([x0, -half, z]), np.array([x0, half, z]))


def test_ph11_nu_from_hand_geometry_horizontal_los():
    """Horizontal LOS at z = 1.5 under an edge at z = 1.62, x = 1.4: h = 0.12 m, d1, d2 from
    the crossing; nu by the section 5.9 formula."""
    a, b = np.array([-2.0, 0.3, 1.5]), np.array([4.0, 0.3, 1.5])
    cr = _edge(1.4, 1.62).crossing(a, b)
    assert cr is not None
    assert cr.h_m == pytest.approx(0.12, rel=1e-12)
    assert cr.d1_m == pytest.approx(3.4, rel=1e-14) and cr.d2_m == pytest.approx(2.6, rel=1e-14)
    nu = 0.12 * math.sqrt(2 * (3.4 + 2.6) / (LAM * 3.4 * 2.6))
    assert cr.nu(FC) == pytest.approx(nu, rel=1e-12)
    assert knife_edge_nu(0.12, 3.4, 2.6, FC) == pytest.approx(
        rf.knife_edge_nu(0.12, 3.4, 2.6, FC), rel=1e-15
    )


def test_ph11_nu_for_a_slanted_los_uses_the_perpendicular_clearance():
    """Slanted LOS: h is the distance from the edge line to the LOS line (common
    perpendicular), signed positive when obstructed; checked with a cross-product form."""
    a, b = np.array([-2.0, 0.0, 2.4]), np.array([5.0, 0.0, 0.9])
    edge = _edge(1.0, 1.95)
    cr = edge.crossing(a, b)
    assert cr is not None
    k = (b - a) / np.linalg.norm(b - a)
    q = np.array([1.0, 0.0, 1.95])
    dist = float(np.linalg.norm(np.cross(q - a, k)))  # point-to-line distance (edge _|_ LOS)
    z_los = 2.4 + (1.0 + 2.0) / 7.0 * (0.9 - 2.4)
    assert 1.95 > z_los  # obstructed
    assert cr.h_m == pytest.approx(dist, rel=1e-12)
    # below the LOS: same magnitude formula, negative sign
    cr2 = _edge(1.0, 1.0).crossing(a, b)
    assert cr2 is not None and cr2.h_m < 0
    assert abs(cr2.h_m) == pytest.approx(
        float(np.linalg.norm(np.cross(np.array([1.0, 0, 1.0]) - a, k))), rel=1e-12
    )


def test_ph11_edge_applicability():
    a, b = np.array([-2.0, 0.0, 1.5]), np.array([4.0, 0.0, 1.5])
    assert _edge(1.0, 1.6, half=0.5).crossing(a, b) is not None
    # LOS passes beside the edge's extent (y = 0.8 > half-length 0.5)
    dy = np.array([0.0, 0.8, 0.0])
    assert _edge(1.0, 1.6, half=0.5).crossing(a + dy, b + dy) is None
    # both nodes on the same side of the screen plane
    assert _edge(5.0, 1.6).crossing(a, b) is None
    # vertical edges are rejected (no "below")
    with pytest.raises(ConfigurationError, match="vertical"):
        KnifeEdge("v", np.array([0.0, 0, 0]), np.array([0.0, 0, 2]))
    with pytest.raises(ConfigurationError):
        KnifeEdge("z", np.array([0.0, 0, 1]), np.array([0.0, 0, 1]))


def test_ph11_reciprocal_nu():
    a, b = np.array([-2.0, 0.4, 2.1]), np.array([4.5, -0.6, 1.2])
    e = KnifeEdge("e", np.array([0.5, -3.0, 1.7]), np.array([1.5, 3.0, 1.9]))
    c1, c2 = e.crossing(a, b), e.crossing(b, a)
    assert c1 is not None and c2 is not None
    assert c1.h_m == pytest.approx(c2.h_m, rel=1e-14)
    assert c1.d1_m == pytest.approx(c2.d2_m, rel=1e-14) and c1.d2_m == pytest.approx(
        c2.d1_m, rel=1e-14
    )
    assert c1.nu(FC) == pytest.approx(c2.nu(FC), rel=1e-13)


def test_ph11_los_factor_in_the_scene():
    """The LOS amplitude is the Friis term times F(nu) (rel 1e-14); kind and delay unchanged."""
    a, b = np.array([-2.0, 0.3, 1.5]), np.array([4.0, 0.3, 1.5])
    mech = KnifeEdgeMechanism((_edge(1.4, 1.62),))
    m0 = two_node(a, b, FC)
    m1 = two_node(a, b, FC, mechanisms=[mech])
    p0 = m0.snapshot(Fraction(0)).paths[link()]
    p1 = m1.snapshot(Fraction(0)).paths[link()]
    nu = 0.12 * math.sqrt(2 * 6.0 / (LAM * 3.4 * 2.6))
    assert p1.kind.tolist() == [PathKind.LOS]
    assert p1.delay_s[0] == p0.delay_s[0]
    want = p0.amp[0] * complex(rf.knife_edge_F(nu))
    assert abs(p1.amp[0] - want) <= 1e-14 * abs(want)


def test_ph11_partition_and_edge_multiply():
    a, b = np.array([-2.0, 0.3, 1.5]), np.array([4.0, 0.3, 1.5])
    part = Occluder(
        "p", np.array([[0.0, -1, 0], [0.0, 1, 0], [0.0, 1, 3], [0.0, -1, 3]]), "partition", 6.0
    )
    mech = KnifeEdgeMechanism((_edge(1.4, 1.62),))
    m = two_node(a, b, FC, mechanisms=[mech], occluders=[part])
    m0 = two_node(a, b, FC)
    f = mech.factor(a, b, FC)
    want = m0.snapshot(0).paths[link()].amp[0] * 10 ** (-6 / 20) * f
    got = m.snapshot(0).paths[link()].amp[0]
    assert abs(got - want) <= 1e-15 * abs(want)


def test_ph11_one_edge_per_path_keeps_the_dominant_and_warns():
    a, b = np.array([-2.0, 0.3, 1.5]), np.array([4.0, 0.3, 1.5])
    low, high = _edge(0.5, 1.52, eid="low"), _edge(2.0, 1.7, eid="high")
    mech = KnifeEdgeMechanism((low, high))
    with pytest.warns(ValidityWarning, match="dominant"):
        dom = mech.dominant(a, b, FC, "l1")
    assert dom is not None and dom[0].id == "high"
    nu_high = high.crossing(a, b).nu(FC)  # type: ignore[union-attr]
    assert dom[1] == nu_high
    assert mech.factor(a, b, FC, "l1") == complex(knife_edge_F(nu_high))  # warned once per link


def test_ph11_from_scenario_spec():
    spec = [{"id": "e1", "p0_m": (1.4, -3.0, 1.62), "p1_m": (1.4, 3.0, 1.62)}]
    mech = mechanism_from_scenario(spec, {})
    assert isinstance(mech, KnifeEdgeMechanism) and [e.id for e in mech.edges] == ["e1"]
    assert mechanism_from_scenario([], {}) is None
    with pytest.raises(ConfigurationError, match="unique"):
        KnifeEdgeMechanism(edges_from_spec(spec + spec))
    a, b = np.array([-2.0, 0.3, 1.5]), np.array([4.0, 0.3, 1.5])
    v = by_kind(two_node(a, b, FC, mechanisms=[mech]), link())
    assert (
        abs(abs(v[-1]) / abs(by_kind(two_node(a, b, FC), link())[-1]) - abs(mech.factor(a, b, FC)))
        <= 1e-14
    )


# --------------------------------------------------------------------------- paths with legs
# Section 9.2: a knife edge "applies to the paths it intersects"; section 5.9: one edge per
# path. The ground and wall image paths Tx -> S -> Rx are diffracted by the dominant edge
# crossing either leg (h from that leg's line, d1/d2 along the unfolded path).


def test_ph11_two_point_path_is_exactly_the_los_rule():
    a, b = np.array([-2.0, 0.4, 2.1]), np.array([4.5, -0.6, 1.2])
    e = KnifeEdge("e", np.array([0.5, -3.0, 1.7]), np.array([1.5, 3.0, 1.9]))
    (c,) = e.path_crossings(np.stack([a, b]))
    ref = e.crossing(a, b)
    assert ref is not None
    assert (c.h_m, c.d1_m, c.d2_m) == (ref.h_m, ref.d1_m, ref.d2_m)
    assert np.array_equal(c.point_m, ref.point_m)
    mech = KnifeEdgeMechanism((e,))
    assert mech.path_factor(np.stack([a, b]), FC) == mech.factor(a, b, FC)


def test_ph11_edge_diffracts_the_ground_leg_it_crosses():
    """Tx (0,0,2), Rx (10,0,2), ground z = 0 (specular point (5,0,0)); edge along y at
    x = 2, z = 1.5. The LOS passes 0.5 m above the edge (nu < 0); the Tx leg crosses x = 2 at
    z = 1.2, 0.3 m below the edge: h = 0.3 cos(atan(2/5)), d1 = 0.4 sqrt(29), d2 = 1.6
    sqrt(29) (hand geometry). Ground path = lambda/(4 pi d_img) Gamma_V(psi) F(nu), all from
    the oracle module (rel 1e-13); reciprocal; the LOS is diffracted by its own nu."""
    from sremu.physics.ground import GroundMechanism, GroundPlane

    fc = 3.5e9
    lam = C0 / fc
    edge = KnifeEdge("e", np.array([2.0, -3.0, 1.5]), np.array([2.0, 3.0, 1.5]))
    plane = GroundPlane(15.0, 0.01, "V")
    mech = GroundMechanism(plane, edges=(edge,))
    a, b = np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])
    amp, tau = mech.ground_path(a, b, fc)
    r29 = math.sqrt(29.0)
    h = 0.3 * 5.0 / r29
    d1, d2 = 0.4 * r29, 1.6 * r29
    nu = h * math.sqrt(2 * (d1 + d2) / (lam * d1 * d2))
    d_img = math.sqrt(116.0)
    psi = math.asin(4.0 / d_img)
    gam = complex(rf.fresnel_gamma_v(psi, rf.complex_permittivity(15.0, 0.01, fc)))
    want = lam / (4 * math.pi * d_img) * gam * complex(rf.knife_edge_F(nu))
    assert 0.5 < nu < 1.5  # a real obstruction of the leg, |F| well below 1
    assert abs(amp - want) <= 1e-13 * abs(want)
    assert tau == pytest.approx(d_img / C0, rel=1e-15)  # the delay is the image path's
    back = mech.ground_path(b, a, fc)
    assert abs(back[0] - amp) <= 1e-13 * abs(amp) and back[1] == tau
    # in a scene: the LOS (0.5 m clearance) carries F(nu_LOS) with nu_LOS < 0
    m = two_node(a, b, fc, mechanisms=[mech, KnifeEdgeMechanism((edge,))])
    pb = m.snapshot(Fraction(0)).paths[link()]
    nu_los = -0.5 * math.sqrt(2 * 10.0 / (lam * 2.0 * 8.0))
    want_los = lam / (4 * math.pi * 10.0) * complex(rf.knife_edge_F(nu_los))
    assert abs(pb.amp[0] - want_los) <= 1e-13 * abs(want_los)
    assert abs(pb.amp[1] - want) <= 1e-13 * abs(want)


def test_ph11_edge_diffracts_the_wall_leg_it_crosses():
    """Vertical wall y = 4, Tx (0,0,1.5), Rx (6,1,1.5): specular point (24/7, 4, 1.5).
    Edge along x at y = 2, z = 1.6 over x in [0, 4]: the Tx leg crosses y = 2 at its midpoint
    0.1 m below the edge (h = 0.1); the Rx leg crosses y = 2 at x = 36/7 > 4 (beside the
    edge) and the LOS never reaches y = 2. Wall path = lambda/(4 pi d_img) Gamma_H(psi)
    F(nu) ("V" link on a vertical wall is TE), hand geometry and oracle forms (1e-13)."""
    from sremu.physics.walls import Wall, WallsMechanism

    wall = Wall(
        "w", np.array([[-20.0, 4, 0], [20.0, 4, 0], [20.0, 4, 6], [-20.0, 4, 6]]), 4.5, 0.02
    )
    edge = KnifeEdge("e", np.array([0.0, 2.0, 1.6]), np.array([4.0, 2.0, 1.6]))
    a, b = np.array([0.0, 0.0, 1.5]), np.array([6.0, 1.0, 1.5])
    assert edge.crossing(a, b) is None
    leg = math.hypot(24.0 / 7.0, 4.0)
    d_img = math.sqrt(85.0)
    d1, d2 = leg / 2, d_img - leg / 2
    nu = 0.1 * math.sqrt(2 * (d1 + d2) / (LAM * d1 * d2))
    psi = math.asin(7.0 / d_img)
    gam = complex(rf.fresnel_gamma_h(psi, rf.complex_permittivity(4.5, 0.02, FC)))
    want = LAM / (4 * math.pi * d_img) * gam * complex(rf.knife_edge_F(nu))
    mech = WallsMechanism((wall,), "V", edges=(edge,))
    got = mech.wall_path(wall, a, b, FC)
    assert got is not None
    assert 0.0 < nu and abs(complex(rf.knife_edge_F(nu))) < 0.5
    assert abs(got[0] - want) <= 1e-13 * abs(want)
    assert got[1] == pytest.approx(d_img / C0, rel=1e-15)
    back = mech.wall_path(wall, b, a, FC)
    assert back is not None and abs(back[0] - got[0]) <= 1e-13 * abs(got[0])
    # without the edge it is the plain wall path
    plain = WallsMechanism((wall,), "V").wall_path(wall, a, b, FC)
    assert plain is not None
    assert abs(got[0] - plain[0] * complex(rf.knife_edge_F(nu))) <= 1e-13 * abs(got[0])


def test_ph11_reversed_path_and_dominant_crossing_over_legs():
    """A path crossing the same screen on both legs (Tx and Rx on one side, the reflection
    point on the other) has two crossings: the dominant one (largest nu) is kept with a
    warning, and reversing the path gives the same nu with d1 and d2 swapped (reciprocity)."""
    edge = KnifeEdge("e", np.array([2.0, -5.0, 1.0]), np.array([2.0, 5.0, 1.0]))
    path = np.array([[0.0, 0.0, 1.4], [5.0, 0.3, 0.2], [0.5, 1.0, 0.6]])
    fwd = edge.path_crossings(path)
    rev = edge.path_crossings(path[::-1])
    assert len(fwd) == 2 and len(rev) == 2
    total = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    for c, r in zip(fwd, rev[::-1], strict=True):
        assert c.d1_m + c.d2_m == pytest.approx(total, rel=1e-14)
        assert c.h_m == pytest.approx(r.h_m, rel=1e-13)
        assert c.d1_m == pytest.approx(r.d2_m, rel=1e-13)
        assert c.nu(FC) == pytest.approx(r.nu(FC), rel=1e-12)
    mech = KnifeEdgeMechanism((edge,))
    with pytest.warns(ValidityWarning, match="dominant"):
        dom = mech.dominant_on_path(path, FC, "p")
    assert dom is not None and dom[1] == max(c.nu(FC) for c in fwd)
    assert mech.path_factor(path, FC, "p") == complex(knife_edge_F(dom[1]))  # warned once
    with pytest.raises(ConfigurationError):
        edge.path_crossings(path[:1])


def _hill_node(nid: str, pos: list[float], direction: str) -> dict:  # type: ignore[type-arg]
    s: dict = {"name": f"{direction}0", "direction": direction, "fc_hz": 1e8, "rate_sps": "1e6"}  # type: ignore[type-arg]
    if direction == "tx":
        s["occupied_bw_hz"] = 1e5
    return {"id": nid, "position_m": pos, "antenna": {"pattern": "isotropic"}, "streams": [s]}


def test_ph11_ground_path_no_longer_bypasses_a_hill(tmp_path):
    """E-D-like scene through from_scenario (D18): 100 MHz, Tx 30 m, Rx 1.5 m at 10 km, a
    hill crest at 20 m halfway. The edge reaches the ground path (both nu > 0, the ground
    leg's nu larger), so the ground path is weaker than the diffracted LOS instead of
    passing through the hill unattenuated (WP3 review finding)."""
    from sremu.physics.ground import GroundMechanism, GroundPlane
    from sremu.physics.propagation import from_scenario

    a, b = [0.0, 0.0, 30.0], [10_000.0, 0.0, 1.5]
    p0, p1 = [5000.0, -500.0, 20.0], [5000.0, 500.0, 20.0]
    doc = {
        "schema_version": "1.0",
        "name": "hill",
        "seed": 5,
        "nodes": [_hill_node("a", a, "tx"), _hill_node("b", b, "rx")],
        "environment": {
            "ground": {"eps_r": 15.0, "sigma_s_per_m": 5e-3, "polarization": "H"},
            "edges": [{"id": "hill", "p0_m": p0, "p1_m": p1}],
        },
        "run": {
            "mode": "lockstep",
            "coupling": "barrier",
            "duration_s": "1",
            "late_policy": "apply_on_arrival",
        },
    }
    pb = from_scenario(doc, base_dir=tmp_path).snapshot(Fraction(0)).paths[link()]
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.GROUND]
    plane = GroundPlane(15.0, 5e-3, "H")
    free_ground = GroundMechanism(plane).ground_path(a, b, 1e8)[0]
    edge = KnifeEdge("hill", np.array(p0), np.array(p1))
    s = plane.specular_point(np.array(a), np.array(b))
    (cr,) = edge.path_crossings(np.stack([np.array(a), s, np.array(b)]))
    los_cr = edge.crossing(a, b)
    assert los_cr is not None and 0 < los_cr.nu(1e8) < cr.nu(1e8)
    want = free_ground * complex(rf.knife_edge_F(cr.nu(1e8)))
    assert abs(pb.amp[1] - want) <= 1e-13 * abs(want)
    assert abs(pb.amp[1]) < abs(pb.amp[0])
