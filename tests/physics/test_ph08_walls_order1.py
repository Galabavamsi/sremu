"""Order-1 image walls (DESIGN.md section 5.8; supports PH-08).

Expected values: image geometry by hand, the section 5.7 Fresnel formulas from
``tests/oracles/rf.py``, and the ground image path (a horizontal wall must reproduce it).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PathKind
from sremu.errors import ConfigurationError, OutOfDomain
from sremu.physics.antennas import CosPower
from sremu.physics.ground import GroundMechanism, GroundPlane, complex_permittivity
from sremu.physics.propagation import from_scenario
from sremu.physics.walls import (
    P2040_CITATION,
    P2040_TABLE3,
    Wall,
    WallsMechanism,
    copol_gamma,
    mechanism_from_scenario,
    p2040_material,
    wall_bounce,
)
from sremu.units import C0
from tests.oracles import rf
from tests.physics._wp3_scenes import link, two_node

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC


def _vwall(y: float, x0: float = -20.0, x1: float = 20.0, z1: float = 6.0, **kw) -> Wall:  # type: ignore[no-untyped-def]
    """Vertical wall in the plane y = const."""
    v = np.array([[x0, y, 0.0], [x1, y, 0.0], [x1, y, z1], [x0, y, z1]])
    return Wall(kw.pop("id", "w"), v, kw.pop("eps_r", 5.0), kw.pop("sigma", 0.05), **kw)


@pytest.mark.parametrize("pol", ["H", "V"])
def test_walls_horizontal_wall_reproduces_the_ground_path(pol: str):
    """A large horizontal wall at z = 0 is the ground: same amplitude (rel 1e-13) and delay,
    with directive antennas (the co-polar form reduces to Gamma_H / Gamma_V)."""
    floor = Wall(
        "floor",
        np.array([[-1e3, -1e3, 0.0], [1e3, -1e3, 0.0], [1e3, 1e3, 0.0], [-1e3, 1e3, 0.0]]),
        12.0,
        0.03,
    )
    a, b = np.array([0.3, -1.0, 2.2]), np.array([7.5, 2.0, 1.1])
    ga, gb = CosPower(2.0, np.array([1.0, 0.3, -0.4])), CosPower(1.0, np.array([-1.0, -0.2, -0.1]))
    w = WallsMechanism((floor,), pol).wall_path(floor, a, b, FC, ga, gb)  # type: ignore[arg-type]
    g = GroundMechanism(GroundPlane(12.0, 0.03, pol, 0.0)).ground_path(a, b, FC, ga, gb)  # type: ignore[arg-type]
    assert w is not None
    assert abs(w[0] - g[0]) <= 1e-13 * abs(g[0])
    assert w[1] == pytest.approx(g[1], rel=1e-15)


@pytest.mark.parametrize(("pol", "oracle"), [("V", rf.fresnel_gamma_h), ("H", rf.fresnel_gamma_v)])
def test_walls_vertical_wall_horizontal_rays(pol: str, oracle):  # type: ignore[no-untyped-def]
    """Vertical wall, rays in a horizontal plane: a "V" link is TE (Gamma_H formula), an
    "H" link is TM (Gamma_V formula), at the grazing angle to the wall plane."""
    wall = _vwall(4.0, eps_r=4.5, sigma=0.02)
    a, b = np.array([0.0, 0.0, 1.5]), np.array([6.0, 1.0, 1.5])
    path = WallsMechanism((wall,), pol).wall_path(wall, a, b, FC)  # type: ignore[arg-type]
    assert path is not None
    a_img = np.array([0.0, 8.0, 1.5])
    d = float(np.linalg.norm(b - a_img))
    psi = math.asin((4.0 + 3.0) / d)
    gam = complex(oracle(psi, rf.complex_permittivity(4.5, 0.02, FC)))
    want = LAM / (4 * math.pi * d) * gam
    assert abs(path[0] - want) <= 1e-13 * abs(want)
    assert path[1] == pytest.approx(d / C0, rel=1e-15)


def test_walls_pec_vertical_wall_signs():
    wall = _vwall(4.0, eps_r=1.0, sigma=math.inf)
    a, b = np.array([0.0, 0.0, 1.5]), np.array([6.0, 1.0, 1.5])
    d = float(np.linalg.norm(b - np.array([0.0, 8.0, 1.5])))
    v = WallsMechanism((wall,), "V").wall_path(wall, a, b, FC)
    h = WallsMechanism((wall,), "H").wall_path(wall, a, b, FC)
    assert v is not None and h is not None
    assert abs(v[0] / (LAM / (4 * math.pi * d)) + 1) <= 1e-14  # E tangential: -1
    assert abs(h[0] / (LAM / (4 * math.pi * d)) - 1) <= 1e-14  # E (nearly) normal: +1


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("pol", ["H", "V"])
def test_walls_copol_gamma_is_reciprocal(pol: str, rng: np.random.Generator):
    """Swapping Tx and Rx (incident direction -k_r) leaves the co-polar coefficient
    unchanged, for oblique walls and lossy materials (1e-14)."""
    for _ in range(200):
        n = _random_unit(rng)
        k = _random_unit(rng)
        if abs(k @ n) < 1e-3:
            continue
        kr = k - 2 * (k @ n) * n
        ec = complex_permittivity(float(rng.uniform(1.5, 40)), float(10 ** rng.uniform(-4, 0)), FC)
        g1 = copol_gamma(k, n, ec, pol)  # type: ignore[arg-type]
        g2 = copol_gamma(-kr, n, ec, pol)  # type: ignore[arg-type]
        assert abs(g1 - g2) <= 1e-14 * max(1.0, abs(g1))
        assert abs(g1) <= 1.0 + 1e-12  # passive


def test_walls_copol_gamma_normal_incidence_and_continuity():
    """At normal incidence the value is Gamma_H(pi/2) (p_i . p_r) whatever the transverse
    basis; tilting slightly in either plane is continuous (TE for horizontal tilt, TM with
    the basis flip for vertical tilt)."""
    ec = complex_permittivity(6.0, 0.01, FC)
    n = np.array([0.0, 1.0, 0.0])
    g0 = copol_gamma(-n, n, ec, "V")
    gh90 = complex(rf.fresnel_gamma_h(math.pi / 2, ec))
    assert abs(g0 - gh90) <= 1e-15
    assert abs(complex(rf.fresnel_gamma_v(math.pi / 2, ec)) + gh90) <= 1e-15  # convention
    for tilt in (np.array([1e-7, 0.0, 0.0]), np.array([0.0, 0.0, 1e-7])):
        k = -n + tilt
        k = k / np.linalg.norm(k)
        for pol in ("H", "V"):
            assert abs(copol_gamma(k, n, ec, pol) - copol_gamma(-n, n, ec, pol)) <= 1e-9  # type: ignore[arg-type]


def test_walls_validity_rules():
    wall = _vwall(4.0, x0=0.0, x1=4.0, z1=3.0)
    a = np.array([0.0, 0.0, 1.5])
    assert wall_bounce(wall, a, np.array([6.0, 0.0, 1.5])) is not None  # specular x = 3: inside
    assert wall_bounce(wall, a, np.array([10.0, 0.0, 1.5])) is None  # specular x = 5: outside
    assert (
        wall_bounce(wall, a, np.array([8.0, 0.0, 1.5])) is not None
    )  # specular x = 4: on the edge
    assert wall_bounce(wall, a, np.array([3.0, 6.0, 1.5])) is None  # opposite sides
    assert wall_bounce(wall, a, np.array([2.0, 4.0, 1.5])) is None  # Rx in the plane
    behind = wall_bounce(wall, np.array([0.0, 8.0, 1.5]), np.array([6.0, 8.0, 1.5]))  # far face
    assert behind is not None and behind.specular_m.tolist() == [3.0, 4.0, 1.5]


def test_walls_roughness_factor():
    smooth = _vwall(4.0, eps_r=5.0, sigma=0.02)
    rough = _vwall(4.0, eps_r=5.0, sigma=0.02, roughness_m=0.004)
    a, b = np.array([0.0, 0.0, 1.5]), np.array([6.0, 1.0, 1.5])
    p0 = WallsMechanism((smooth,), "V").wall_path(smooth, a, b, FC)
    p1 = WallsMechanism((rough,), "V").wall_path(rough, a, b, FC)
    assert p0 is not None and p1 is not None
    d = float(np.linalg.norm(b - np.array([0.0, 8.0, 1.5])))
    sin_psi = 7.0 / d
    rho = math.exp(-2 * (2 * math.pi / LAM * 0.004 * sin_psi) ** 2)
    assert abs(p1[0] / p0[0] - rho) <= 1e-14
    assert 0.0 < rho < 1.0


def test_walls_antenna_gains_toward_the_specular_point():
    wall = _vwall(4.0)
    a, b = np.array([0.0, 0.0, 1.5]), np.array([6.0, 1.0, 1.5])
    s = np.array([24.0 / 7.0, 4.0, 1.5])  # image (0, 8) -> (6, 1) crosses y = 4 at x = 6*4/7
    ga, gb = CosPower(3.0, np.array([0.2, 1.0, 0.0])), CosPower(2.0, np.array([-1.0, 0.5, 0.0]))
    p_iso = WallsMechanism((wall,), "V").wall_path(wall, a, b, FC)
    p_dir = WallsMechanism((wall,), "V").wall_path(wall, a, b, FC, ga, gb)
    assert p_iso is not None and p_dir is not None
    ua, ub = (s - a) / np.linalg.norm(s - a), (s - b) / np.linalg.norm(s - b)
    g = math.sqrt(float(ga.gain(ua[None])[0]) * float(gb.gain(ub[None])[0]))
    assert abs(p_dir[0] - g * p_iso[0]) <= 1e-14 * abs(p_dir[0])


def test_walls_in_scene_order_and_reciprocity():
    w1, w2, w3 = _vwall(-3.0, id="w1"), _vwall(4.0, x0=50, x1=60, id="w2"), _vwall(5.0, id="w3")
    mech = WallsMechanism((w1, w2, w3), "V")
    a, b = np.array([0.0, 0.0, 1.5]), np.array([6.0, 1.0, 1.2])
    pb = two_node(a, b, FC, mechanisms=[mech]).snapshot(Fraction(0)).paths[link()]
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.WALL, PathKind.WALL]  # w2 misses
    assert pb.amp[1] == mech.wall_path(w1, a, b, FC)[0]  # type: ignore[index]
    assert pb.amp[2] == mech.wall_path(w3, a, b, FC)[0]  # type: ignore[index]
    pr = two_node(b, a, FC, mechanisms=[mech], tx="b", rx="a").snapshot(0).paths[link("b", "a")]
    assert np.allclose(pr.amp, pb.amp, rtol=1e-14, atol=0) and np.array_equal(
        pr.delay_s, pb.delay_s
    )


def test_walls_validation():
    with pytest.raises(ConfigurationError, match="coplanar"):
        Wall("w", np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0.3], [0, 1, 0]]), 5.0, 0.0)
    with pytest.raises(ConfigurationError, match="degenerate"):
        Wall("w", np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]), 5.0, 0.0)
    with pytest.raises(ConfigurationError):
        Wall("w", np.zeros((2, 3)), 5.0, 0.0)
    with pytest.raises(ConfigurationError):
        _vwall(1.0, eps_r=-1.0)
    with pytest.raises(ConfigurationError, match="unique"):
        WallsMechanism((_vwall(1.0), _vwall(2.0)))


def test_walls_from_scenario_polarization_and_assumption():
    walls = [
        {
            "id": "w1",
            "vertices_m": [(-5, 3, 0), (5, 3, 0), (5, 3, 3), (-5, 3, 3)],
            "eps_r": 5.0,
            "sigma_s_per_m": 0.1,
        }
    ]
    m = mechanism_from_scenario(walls, {"environment": {"walls": walls}})
    assert isinstance(m, WallsMechanism) and m.polarization == "V" and m.assumptions
    m2 = mechanism_from_scenario(
        walls,
        {
            "environment": {
                "walls": walls,
                "ground": {"eps_r": 15, "sigma_s_per_m": 0.01, "polarization": "H"},
            }
        },
    )
    assert isinstance(m2, WallsMechanism) and m2.polarization == "H" and not m2.assumptions
    assert mechanism_from_scenario([], {}) is None


def test_walls_attached_by_from_scenario(tmp_path):
    doc = {
        "schema_version": "1.0",
        "name": "walls",
        "seed": 2,
        "nodes": [
            {
                "id": "a",
                "position_m": [0.0, 0.0, 1.5],
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": "tx0",
                        "direction": "tx",
                        "fc_hz": FC,
                        "rate_sps": "1e6",
                        "occupied_bw_hz": 1e5,
                    }
                ],
            },
            {
                "id": "b",
                "position_m": [6.0, 1.0, 1.5],
                "antenna": {"pattern": "isotropic"},
                "streams": [{"name": "rx0", "direction": "rx", "fc_hz": FC, "rate_sps": "1e6"}],
            },
        ],
        "environment": {
            "walls": [
                {
                    "id": "w1",
                    "vertices_m": [[-20, 4, 0], [20, 4, 0], [20, 4, 6], [-20, 4, 6]],
                    "eps_r": 5.0,
                    "sigma_s_per_m": 0.05,
                },
                {
                    "id": "w2",
                    "vertices_m": [[-20, -3, 0], [20, -3, 0], [20, -3, 6], [-20, -3, 6]],
                    "eps_r": 3.0,
                    "sigma_s_per_m": 0.0,
                    "roughness_m": 0.002,
                },
            ]
        },
        "run": {
            "mode": "lockstep",
            "coupling": "barrier",
            "duration_s": "1",
            "late_policy": "apply_on_arrival",
        },
    }
    model = from_scenario(doc, base_dir=tmp_path)
    pb = model.snapshot(0).paths[link()]
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.WALL, PathKind.WALL]
    mech = next(x for x in model.mechanisms if isinstance(x, WallsMechanism))
    assert [w.id for w in mech.walls] == ["w1", "w2"] and mech.walls[1].roughness_m == 0.002


def test_p2040_table3_transcription_and_lookup():
    """Rec. ITU-R P.2040-3 Table 3 (cited): eta' = a f^b, sigma = c f^d, f in GHz."""
    assert "P.2040-3" in P2040_CITATION and "Table 3" in P2040_CITATION
    eps, sig = p2040_material("concrete", 3.5e9)
    assert eps == 5.24 and sig == pytest.approx(0.0462 * 3.5**0.7822, rel=1e-15)
    eps, sig = p2040_material("Medium dry ground", 5e9)
    assert eps == pytest.approx(15.0 * 5.0**-0.1, rel=1e-15) and sig == pytest.approx(
        0.035 * 5.0**1.63, rel=1e-15
    )
    assert p2040_material("glass", 300e9) == pytest.approx((5.79, 0.0004 * 300.0**1.658), rel=1e-15)
    assert p2040_material("metal", 28e9) == (1.0, 1.0e7)
    with pytest.raises(OutOfDomain):
        p2040_material("brick", 60e9)
    with pytest.raises(OutOfDomain):
        p2040_material("glass", 150e9)  # between the two glass rows
    with pytest.raises(ConfigurationError, match="unknown"):
        p2040_material("adamantium", 1e9)
    for r in P2040_TABLE3:
        assert r.a > 0 and r.c >= 0 and r.f_min_ghz < r.f_max_ghz
