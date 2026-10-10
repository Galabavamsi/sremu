"""PH-10: flat ground, Fresnel coefficients and the image path (DESIGN.md sections 5.7, 10.2).

Spec rows (oracle values from ``tests/oracles/tables.py``):

* two-ray, 52 km, h = 200/1.5 m, 100 MHz, Gamma = -1: 139.10 dB (+-0.01 dB);
* Gamma_H,V(psi = 1e-4) -> -1 for eps_r 15, sigma 1e-3 S/m (|Gamma + 1| <= 1e-3);
* lossless eps_r = 4: |Gamma_V| = 0 at psi_B = 26.565 deg (<= 1e-12);
* PEC: Gamma_H = -1, Gamma_V = +1 (exact).

The other expected values are independent closed forms (``tests/oracles/rf.py``,
``tests/oracles/ris.py``) or hand geometry.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from sremu.contracts import PathKind
from sremu.errors import ConfigurationError, GeometryError
from sremu.physics.antennas import CosPower
from sremu.physics.ground import (
    GroundMechanism,
    GroundPlane,
    complex_permittivity,
    fresnel_gamma,
    fresnel_gamma_h,
    fresnel_gamma_sin,
    fresnel_gamma_v,
    mechanism_from_scenario,
)
from sremu.physics.los import Occluder
from sremu.physics.propagation import from_scenario
from sremu.units import C0
from tests.oracles import rf
from tests.oracles._data import pm_checks
from tests.oracles.ris import cos_power
from tests.oracles.tables import ORACLE
from tests.physics._wp3_scenes import by_kind, link, two_node

pytestmark = pytest.mark.physics

FC = 100e6
PEC_H = GroundPlane(15.0, math.inf, "H", 0.0)


def _two_ray_model(d_m: float, ht: float, hr: float, plane: GroundPlane = PEC_H):
    return two_node([0.0, 0.0, ht], [d_m, 0.0, hr], FC, mechanisms=[GroundMechanism(plane)])


def test_ph10_two_ray_52km_h200_h1p5_is_139p10_db():
    v = by_kind(_two_ray_model(52_000.0, 200.0, 1.5), link())
    pl = -20 * math.log10(abs(v[-1]))
    o = ORACLE["PH10_two_ray_52km_h1p5_db"]
    assert abs(pl - o.value) <= o.tol
    assert set(v) == {-1, int(PathKind.LOS), int(PathKind.GROUND)}


def test_ph10_two_ray_table_matches_closed_form_and_scratch():
    """48/52/56 km x h_r = 1.5/10 m: equal to the closed-form two-ray coefficient (oracle,
    Gamma = -1) to 1e-7 relative, and to the scratch table (rounded to 0.01 dB)."""
    rows = pm_checks()["T15_FM_direct_path_loss"]
    assert len(rows) == 6
    for row in rows:
        d, hr = row["d_km"] * 1e3, row["hr_m"]
        h = by_kind(_two_ray_model(d, 200.0, hr), link())[-1]
        ref = rf.two_ray_h(d, 200.0, hr, FC, -1.0)
        assert abs(h - ref) <= 1e-7 * abs(ref)
        assert abs(-20 * math.log10(abs(h)) - row["two_ray_flat_dB"]) <= 0.005 + 1e-9


@pytest.mark.parametrize("fc", [100e6, 3.5e9])
def test_ph10_grazing_limit_is_minus_one(fc: float):
    ec = complex_permittivity(15.0, 1e-3, fc)
    tol = ORACLE["PH10_grazing_gamma_plus1_abs"].tol
    assert abs(complex(fresnel_gamma_h(1e-4, ec)) + 1) <= tol
    assert abs(complex(fresnel_gamma_v(1e-4, ec)) + 1) <= tol
    # exactly grazing: both are -1 (to rounding) for any finite eps_c != 1
    assert abs(complex(fresnel_gamma_h(0.0, ec)) + 1) <= 1e-15
    assert abs(complex(fresnel_gamma_v(0.0, ec)) + 1) <= 1e-15


def test_ph10_brewster_lossless_eps4():
    eps = 4.0 + 0j
    psi_b = math.atan(1.0 / math.sqrt(4.0))  # tan psi_B = 1/sqrt(eps_r): grazing Brewster
    o = ORACLE["PH10_brewster_deg"]
    assert abs(math.degrees(psi_b) - o.value) <= o.tol
    assert abs(complex(fresnel_gamma_v(psi_b, eps))) <= 1e-12
    # and it is the minimum of |Gamma_V| found numerically (no knowledge of the formula)
    res = minimize_scalar(
        lambda p: abs(complex(fresnel_gamma_v(p, eps))),
        bounds=(0.1, 1.2),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert abs(math.degrees(res.x) - o.value) <= o.tol
    # Gamma_H has no zero: it stays below -1/3 (lossless eps_r = 4 at normal incidence)
    psi = np.linspace(0.0, math.pi / 2, 1001)
    assert np.all(np.abs(fresnel_gamma_h(psi, eps)) >= (2 - 1) / (2 + 1) - 1e-12)


def test_ph10_pec_is_exact():
    ec = complex_permittivity(15.0, math.inf, 3.5e9)
    psi = np.array([0.0, 1e-6, 0.3, 1.2, math.pi / 2])
    assert np.all(fresnel_gamma_h(psi, ec) == -1)
    assert np.all(fresnel_gamma_v(psi, ec) == 1)
    assert complex_permittivity(math.inf, 0.0, 1e9) == complex(math.inf, 0)
    assert PEC_H.is_pec and not GroundPlane(15.0, 1e9, "H").is_pec


def test_ph10_fresnel_matches_the_closed_form(rng: np.random.Generator):
    """Random materials and angles: equal to the section 5.7 formulas written out in the
    oracle module (1e-13 relative)."""
    for _ in range(50):
        eps_r = float(rng.uniform(1.5, 80.0))
        sigma = float(10 ** rng.uniform(-4, 1))
        fc = float(10 ** rng.uniform(7, 10.5))
        psi = rng.uniform(1e-5, math.pi / 2, 16)
        ec = complex_permittivity(eps_r, sigma, fc)
        assert ec == rf.complex_permittivity(eps_r, sigma, fc)
        for mine, ref in (
            (fresnel_gamma_h, rf.fresnel_gamma_h),
            (fresnel_gamma_v, rf.fresnel_gamma_v),
        ):
            a, b = mine(psi, ec), ref(psi, ec)
            assert np.max(np.abs(a - b)) <= 1e-13 * max(1.0, float(np.max(np.abs(b))))
        # the sin form equals the angle form
        assert (
            np.max(np.abs(fresnel_gamma_sin(np.sin(psi), ec, "V") - fresnel_gamma(psi, ec, "V")))
            == 0
        )


@pytest.mark.parametrize("eps_r", [0.3, 0.5, 0.9])
def test_ph10_lossless_below_unity_uses_the_decaying_branch(eps_r: float):
    """Lossless eps_r < 1 beyond the critical angle (sin^2 psi < 1 - eps_r): the root is
    -j q with q = sqrt(1 - eps_r - sin^2 psi) (decaying transmitted wave for exp(+j omega t)),
    so Gamma_H = (s + j q)/(s - j q) and Gamma_V = (eps s + j q)/(eps s - j q), |Gamma| = 1,
    equal to the sigma -> 0+ limit (WP3 review: the +0j radicand picked the conjugate)."""
    s = np.linspace(0.01, math.sqrt(1.0 - eps_r) - 1e-3, 37)
    q = np.sqrt(1.0 - eps_r - s * s)
    ec = complex_permittivity(eps_r, 0.0, 1e9)
    want_h = (s + 1j * q) / (s - 1j * q)
    want_v = (eps_r * s + 1j * q) / (eps_r * s - 1j * q)
    assert np.max(np.abs(fresnel_gamma_sin(s, ec, "H") - want_h)) <= 1e-14
    assert np.max(np.abs(fresnel_gamma_sin(s, ec, "V") - want_v)) <= 1e-14
    assert np.max(np.abs(np.abs(fresnel_gamma_sin(s, ec, "H")) - 1.0)) <= 1e-14
    # continuity in sigma at 0+ (dGamma/dsigma ~ 1/q^2 grows near the critical angle; the
    # wrong branch differs by O(1))
    tiny = complex_permittivity(eps_r, 1e-18, 1e9)
    for pol in ("H", "V"):
        a = fresnel_gamma_sin(s, ec, pol)  # type: ignore[arg-type]
        b = fresnel_gamma_sin(s, tiny, pol)  # type: ignore[arg-type]
        assert np.max(np.abs(a - b)) <= 1e-12
    # a complex eps_c given with +0j (lossless) means the same material
    assert np.array_equal(
        fresnel_gamma_sin(s, complex(eps_r, 0.0), "H"), fresnel_gamma_sin(s, ec, "H")
    )


def test_ph10_free_space_ground_reflects_nothing():
    """eps_c = 1 (no interface): Gamma = 0 exactly for both polarizations, including at
    grazing (the root is formed as sqrt((eps_c - 1) + sin^2 psi), section 5.7 identity)."""
    psi = np.linspace(0.0, math.pi / 2, 101)
    s = np.linspace(0.0, 1.0, 101)
    for pol in ("H", "V"):
        assert np.all(fresnel_gamma(psi, 1.0 + 0j, pol) == 0)  # type: ignore[arg-type]
        assert np.all(fresnel_gamma_sin(s, 1.0 + 0j, pol) == 0)  # type: ignore[arg-type]


def test_ph10_ground_path_amplitude_delay_and_gains():
    """Directive antennas: amplitude = sqrt(G_T G_R) lambda/(4 pi d_img) Gamma(psi) with gains
    toward the specular point; delay d_img/C0 (independent closed form)."""
    fc = 3.5e9
    plane = GroundPlane(5.0, 0.02, "V", 0.0)
    a, b = np.array([0.2, -0.4, 2.0]), np.array([6.1, 1.3, 1.1])
    ant_a = CosPower(2.0, np.array([1.0, 0.2, -0.3]))
    ant_b = CosPower(4.0, np.array([-1.0, -0.1, -0.2]))
    m = two_node(a, b, fc, mechanisms=[GroundMechanism(plane)], ant_tx=ant_a, ant_rx=ant_b)
    pb = m.snapshot(Fraction(0)).paths[link()]
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.GROUND]
    a_img, b_img = a * [1, 1, -1], b * [1, 1, -1]
    d_img = float(np.linalg.norm(b - a_img))
    psi = math.asin((a[2] + b[2]) / d_img)
    gam = complex(rf.fresnel_gamma_v(psi, rf.complex_permittivity(5.0, 0.02, fc)))
    u_a = (b_img - a) / np.linalg.norm(b_img - a)
    u_b = (a_img - b) / np.linalg.norm(a_img - b)
    g = math.sqrt(
        float(cos_power(2.0, [1.0, 0.2, -0.3])(u_a[None])[0])
        * float(cos_power(4.0, [-1.0, -0.1, -0.2])(u_b[None])[0])
    )
    expected = g * (C0 / fc) / (4 * math.pi * d_img) * gam
    assert abs(pb.amp[1] - expected) <= 1e-13 * abs(expected)
    assert pb.delay_s[1] == pytest.approx(d_img / C0, rel=1e-15)
    # the direct path is untouched by the ground mechanism
    m0 = two_node(a, b, fc, ant_tx=ant_a, ant_rx=ant_b)
    assert m0.snapshot(Fraction(0)).paths[link()].amp[0] == pb.amp[0]


def test_ph10_ground_height_is_a_translation():
    """Raising the ground and every node by the same offset changes nothing (1e-15)."""
    fc = 2.4e9
    a, b = np.array([0.0, 0.0, 3.0]), np.array([40.0, 5.0, 1.5])
    off = np.array([0.0, 0.0, 12.5])
    p0 = GroundMechanism(GroundPlane(15.0, 0.005, "H", 0.0)).ground_path(a, b, fc)
    p1 = GroundMechanism(GroundPlane(15.0, 0.005, "H", 12.5)).ground_path(a + off, b + off, fc)
    assert abs(p0[0] - p1[0]) <= 1e-15 * abs(p0[0])
    assert abs(p0[1] - p1[1]) <= 1e-15 * p0[1]


def test_ph10_reciprocal_ground_path():
    fc = 3.5e9
    mech = GroundMechanism(GroundPlane(7.0, 0.1, "V", -0.5))
    a, b = np.array([1.0, 2.0, 1.7]), np.array([-4.0, 9.0, 0.2])
    ga, gb = CosPower(1.5, np.array([-1.0, 1.0, 0.0])), CosPower(3.0, np.array([1.0, -1.0, -0.5]))
    p_ab = mech.ground_path(a, b, fc, ga, gb)
    p_ba = mech.ground_path(b, a, fc, gb, ga)
    assert abs(p_ab[0] - p_ba[0]) <= 1e-15 * abs(p_ab[0])
    assert p_ab[1] == p_ba[1]


def test_ph10_node_below_ground_raises():
    m = two_node(
        [0, 0, -0.1], [10, 0, 1.5], FC, mechanisms=[GroundMechanism(GroundPlane(15, 1e-3, "H"))]
    )
    with pytest.raises(GeometryError, match="below the ground"):
        m.snapshot(Fraction(0))


def test_ph10_occluders_act_on_both_ground_legs():
    """A partition crossing only the Tx -> ground leg scales the ground path by exactly
    10^(-L/20) and leaves the LOS unchanged; an opaque one zeroes it."""
    fc = 3.5e9
    a, b = np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])
    # specular point at (5, 0, 0); the Tx leg crosses x = 2 at z = 1.2, the LOS at z = 2
    verts = np.array([[2.0, -1.0, 0.5], [2.0, 1.0, 0.5], [2.0, 1.0, 1.6], [2.0, -1.0, 1.6]])
    plane = GroundPlane(15.0, 0.01, "V")
    free = GroundMechanism(plane).ground_path(a, b, fc)
    for loss_db in (3.0, 17.5):
        part = Occluder("p", verts, "partition", loss_db)
        got = GroundMechanism(plane, (part,)).ground_path(a, b, fc)
        assert got[0] == free[0] * 10 ** (-loss_db / 20)
        assert got[1] == free[1]
        m = two_node(a, b, fc, mechanisms=[GroundMechanism(plane, (part,))], occluders=[part])
        assert (
            m.snapshot(Fraction(0)).paths[link()].amp[0]
            == two_node(a, b, fc).snapshot(0).paths[link()].amp[0]
        )
    opaque = Occluder("o", verts, "opaque")
    assert GroundMechanism(plane, (opaque,)).ground_path(a, b, fc)[0] == 0


def test_ph10_from_scenario_attaches_the_ground(tmp_path):
    """D18: ``environment.ground`` builds the same mechanism through from_scenario."""
    doc = {
        "schema_version": "1.0",
        "name": "ph10",
        "seed": 1,
        "nodes": [
            {
                "id": "a",
                "position_m": [0.0, 0.0, 30.0],
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": "tx0",
                        "direction": "tx",
                        "fc_hz": 9e8,
                        "rate_sps": "1e6",
                        "occupied_bw_hz": 1e5,
                    }
                ],
            },
            {
                "id": "b",
                "position_m": [800.0, 50.0, 1.5],
                "antenna": {"pattern": "isotropic"},
                "streams": [{"name": "rx0", "direction": "rx", "fc_hz": 9e8, "rate_sps": "1e6"}],
            },
        ],
        "environment": {
            "ground": {"eps_r": 15.0, "sigma_s_per_m": 0.005, "polarization": "V", "height_m": 0.0}
        },
        "run": {
            "mode": "lockstep",
            "coupling": "barrier",
            "duration_s": "1",
            "late_policy": "apply_on_arrival",
        },
    }
    m = from_scenario(doc, base_dir=tmp_path)
    mechs = [x for x in m.mechanisms if isinstance(x, GroundMechanism)]
    assert len(mechs) == 1 and mechs[0].plane == GroundPlane(15.0, 0.005, "V", 0.0)
    pb = m.snapshot(0).paths["a.tx0->b.rx0"]
    direct = GroundMechanism(GroundPlane(15.0, 0.005, "V")).ground_path(
        [0, 0, 30.0], [800.0, 50.0, 1.5], 9e8
    )
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.GROUND]
    assert pb.amp[1] == direct[0] and pb.delay_s[1] == direct[1]
    assert mechanism_from_scenario(None, doc) is None
    with pytest.raises(ConfigurationError):
        mechanism_from_scenario({"eps_r": 15.0}, doc)
