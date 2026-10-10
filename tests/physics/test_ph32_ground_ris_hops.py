"""PH-32: ground image expansion on RIS hops (DESIGN.md sections 3.4, 5.7, 10.2).

Spec rows: ``Gamma_g = 0`` reduces exactly to the free-space RIS term; the four hop
variants are reciprocal (rel <= 1e-12). In addition, each hop variant equals an independent
K1 sum over the image source/receiver with the Fresnel weight of every specular point and
the real antennas' gains toward the specular points (``tests/oracles``).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PanelConfig, PathKind
from sremu.errors import GeometryError
from sremu.physics.antennas import CosPower, Isotropic
from sremu.physics.geometry import PanelGrid, Pose
from sremu.physics.ground import GroundMechanism, GroundPlane
from sremu.physics.propagation import PanelModel
from sremu.physics.response import builtin
from sremu.units import C0
from tests.oracles import rf
from tests.oracles.ris import cos_power, k1_terms, make_panel
from tests.physics._wp3_scenes import by_kind, link, row_values, two_node

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
A = np.array([-2.0, -0.5, 1.6])
B = np.array([2.5, 0.4, 1.1])
CENTER = [0.0, 3.0, 1.5]
NORMAL = [0.0, -1.0, 0.0]
U_AXIS = [1.0, 0.0, 0.0]
NU, NV = 8, 6
ALPHA_A, BORE_A = 1.5, np.array([0.6, 1.0, -0.1])
ALPHA_B, BORE_B = 3.0, np.array([-0.7, 1.0, 0.1])
LOSSY_V = GroundPlane(15.0, 0.005, "V", 0.0)
MIRROR = np.array([1.0, 1.0, -1.0])


def _panel(center=CENTER) -> PanelModel:  # type: ignore[no-untyped-def]
    grid = PanelGrid(Pose(center, NORMAL, U_AXIS), NU, NV, LAM / 2, LAM / 2)
    return PanelModel("ris0", grid, builtin("ideal_bbit(2)"), cell_model="point")


def _scene(plane: GroundPlane | None, *, swap: bool = False, iso: bool = False):  # type: ignore[no-untyped-def]
    ant_a = Isotropic() if iso else CosPower(ALPHA_A, BORE_A)
    ant_b = Isotropic() if iso else CosPower(ALPHA_B, BORE_B)
    mechs = [] if plane is None else [GroundMechanism(plane)]
    if not swap:
        return two_node(
            A,
            B,
            FC,
            mechanisms=mechs,
            panels=[_panel()],
            ant_tx=ant_a,
            ant_rx=ant_b,
            tx="a",
            rx="b",
        )
    return two_node(
        B, A, FC, mechanisms=mechs, panels=[_panel()], ant_tx=ant_b, ant_rx=ant_a, tx="b", rx="a"
    )


def _cfg(rng: np.random.Generator) -> dict[str, PanelConfig]:
    return {"ris0": PanelConfig("ris0", rng.integers(0, 4, NU * NV).astype(np.int16))}


def test_ph32_rows_carry_the_four_hop_variants():
    m = _scene(LOSSY_V)
    rc = m.snapshot(Fraction(0)).ris["ris0"]
    assert rc.rows == tuple((link(), v) for v in range(4))
    assert rc.tile_delay_s.shape == (4, 1)
    # unfolded tile delays: Tx or its image -> tile reference -> Rx or its image
    c = np.asarray(CENTER)  # one render tile: the reference point is the centroid = centre
    srcs = (A, A * MIRROR, A, A * MIRROR)
    dsts = (B, B, B * MIRROR, B * MIRROR)
    for r, (s, d) in enumerate(zip(srcs, dsts, strict=True)):
        ref = (np.linalg.norm(c - s) + np.linalg.norm(c - d)) / C0
        assert rc.tile_delay_s[r, 0] == pytest.approx(ref, rel=1e-14)


@pytest.mark.parametrize("pol", ["H", "V"])
def test_ph32_gamma_zero_reduces_exactly_to_free_space(pol: str, rng: np.random.Generator):
    """eps_r = 1, sigma = 0: Gamma_g = 0, so the RIS term (and the total) equals the
    free-space model's (rel <= 1e-12); the hop rows and the ground path are exactly 0."""
    cfg = _cfg(rng)
    free = by_kind(_scene(None), link(), cfg)
    m = _scene(GroundPlane(1.0, 0.0, pol, 0.0))  # type: ignore[arg-type]
    got = by_kind(m, link(), cfg)
    ris_f, ris_g = free[int(PathKind.RIS_TILE)], got[int(PathKind.RIS_TILE)]
    assert abs(ris_g - ris_f) <= 1e-12 * abs(ris_f)
    assert abs(got[-1] - free[-1]) <= 1e-12 * abs(free[-1])
    assert got[int(PathKind.GROUND)] == 0
    rows = row_values(m, "ris0", link(), cfg)
    assert rows[1] == 0 and rows[2] == 0 and rows[3] == 0


@pytest.mark.parametrize("plane", [LOSSY_V, GroundPlane(4.0, 0.0, "H", -0.3)])
def test_ph32_four_hop_variants_are_reciprocal(plane: GroundPlane, rng: np.random.Generator):
    """Tx <-> Rx (with their antennas): variant 1 <-> 2, variants 0 and 3 map to themselves,
    each row equal to rel <= 1e-12; so are the ground path and the total."""
    cfg = _cfg(rng)
    ab = row_values(_scene(plane), "ris0", link("a", "b"), cfg)
    ba = row_values(_scene(plane, swap=True), "ris0", link("b", "a"), cfg)
    assert set(ab) == set(ba) == {0, 1, 2, 3}
    for v, w in ((0, 0), (1, 2), (2, 1), (3, 3)):
        assert abs(ab[v]) > 0
        assert abs(ab[v] - ba[w]) <= 1e-12 * abs(ab[v])
    t_ab = by_kind(_scene(plane), link("a", "b"), cfg)
    t_ba = by_kind(_scene(plane, swap=True), link("b", "a"), cfg)
    for k in (int(PathKind.LOS), int(PathKind.GROUND), int(PathKind.RIS_TILE), -1):
        assert abs(t_ab[k] - t_ba[k]) <= 1e-12 * abs(t_ab[k])


def test_ph32_each_variant_matches_the_image_oracle(rng: np.random.Generator):
    """Independent K1 sums (tests/oracles/ris.py) over the image source/receiver, weighted by
    the oracle Fresnel coefficient of each specular point, with the real antennas' gains
    toward the specular points (gain of the mirrored direction)."""
    cfg = _cfg(rng)
    rows = row_values(_scene(LOSSY_V), "ris0", link(), cfg)
    p = make_panel(CENTER, NORMAL, U_AXIS, NU, NV, LAM / 2, LAM / 2)
    gam = np.exp(2j * math.pi * cfg["ris0"].state / 4)
    ec = rf.complex_permittivity(LOSSY_V.eps_r, LOSSY_V.sigma_s_per_m, FC)
    ga, gb = cos_power(ALPHA_A, BORE_A), cos_power(ALPHA_B, BORE_B)

    def mirrored(g):  # type: ignore[no-untyped-def]
        return lambda u: g(np.asarray(u) * MIRROR)

    def weight(node: np.ndarray) -> np.ndarray:
        img = node * MIRROR
        d = np.linalg.norm(p.cells - img[None, :], axis=1)
        psi = np.arcsin((node[2] + p.cells[:, 2]) / d)
        return np.asarray(rf.fresnel_gamma_v(psi, ec))

    wa, wb = weight(A), weight(B)
    expected = {
        0: np.sum(k1_terms(A, B, p, FC, gam, g_t=ga, g_r=gb)),
        1: np.sum(wa * k1_terms(A * MIRROR, B, p, FC, gam, g_t=mirrored(ga), g_r=gb)),
        2: np.sum(wb * k1_terms(A, B * MIRROR, p, FC, gam, g_t=ga, g_r=mirrored(gb))),
        3: np.sum(
            wa
            * wb
            * k1_terms(A * MIRROR, B * MIRROR, p, FC, gam, g_t=mirrored(ga), g_r=mirrored(gb))
        ),
    }
    for v in range(4):
        assert abs(rows[v] - expected[v]) <= 1e-12 * abs(expected[v]), v


def test_ph32_pec_ground_is_image_theory():
    """Isotropic antennas, Gamma_H = -1 exactly (PEC, pol H): variant 1 equals minus the
    direct row of a scene whose Tx sits at the image position (bitwise-equal kernel input)."""
    cfg = {"ris0": PanelConfig("ris0", np.arange(NU * NV, dtype=np.int16) % 4)}
    m = _scene(GroundPlane(15.0, math.inf, "H"), iso=True)
    rows = row_values(m, "ris0", link(), cfg)
    img = two_node(A * MIRROR, B, FC, panels=[_panel()])
    direct_img = row_values(img, "ris0", link(), cfg)[0]
    assert abs(rows[1] + direct_img) <= 1e-15 * abs(direct_img)


def test_ph32_panel_below_ground_raises():
    m = two_node(A, B, FC, mechanisms=[GroundMechanism(LOSSY_V)], panels=[_panel([0.0, 3.0, 0.05])])
    with pytest.raises(GeometryError, match="below the ground"):
        m.snapshot(Fraction(0))


def test_ph32_ris_hops_can_be_disabled_for_diagnostics():
    m = two_node(
        A, B, FC, mechanisms=[GroundMechanism(LOSSY_V, ris_hops_enabled=False)], panels=[_panel()]
    )
    snap = m.snapshot(Fraction(0))
    assert snap.ris["ris0"].rows == ((link(), 0),)
    assert snap.paths[link()].kind.tolist() == [PathKind.LOS, PathKind.GROUND]
