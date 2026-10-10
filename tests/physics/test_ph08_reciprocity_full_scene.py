"""PH-08: reciprocity of the full scene (DESIGN.md sections 5.7-5.9, 10.2).

Spec row: LOS + ground + 2 walls (order 1) + knife edge + RIS with a symmetric angle table,
including the ground hop variants: swapping the roles of the two nodes (with their
antennas) leaves the channel unchanged, rel <= 1e-12; an asymmetric analytic table raises.

Every mechanism is checked to be active in the scene (non-zero ground, two wall paths, an
applied knife edge with |F| != 1 on the LOS and on the ground and wall paths whose legs it
crosses, four RIS hop rows), so the identity is not vacuous.

The 1e-12 criterion is applied to the total channel, to the RIS sum, to every RIS hop row
and to every individual deterministic path. The per-kind sums of several paths are compared
relative to the kind's summed path magnitudes (their rounding scale): the two wall terms
partly cancel, and relative to the cancelled sum the ~1e-13 phase rounding of each term
(``mod(f_c tau, 1)`` at ~120 cycles, delays equal to 1 ulp) is amplified.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PanelConfig, PathKind
from sremu.errors import ReciprocityError
from sremu.physics.antennas import CosPower
from sremu.physics.diffraction import KnifeEdge, KnifeEdgeMechanism
from sremu.physics.geometry import PanelGrid, Pose
from sremu.physics.ground import GroundMechanism, GroundPlane
from sremu.physics.propagation import PanelModel
from sremu.physics.response import ResponseTable
from sremu.physics.walls import Wall, WallsMechanism
from sremu.units import C0
from tests.physics._wp3_scenes import analytic_provenance, by_kind, link, row_values, two_node

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
A = np.array([-3.0, -0.5, 1.6])
B = np.array([3.0, 0.8, 1.2])
THETA = np.arange(-90.0, 90.0 + 1e-9, 5.0)


def _table(symmetric: bool = True) -> ResponseTable:
    ti, ts = np.meshgrid(np.radians(THETA), np.radians(THETA), indexing="ij")
    if symmetric:
        shape = 0.25 * (ti + ts) + 0.1 * np.cos(ti - ts)
    else:
        shape = 0.25 * ti + 0.05 * ts  # analytic and NOT reciprocal
    g0 = 0.9 * np.exp(1j * (0.3 + shape))
    g1 = 0.7 * np.exp(1j * (0.3 + 2.8 + shape))
    gamma = np.stack([g0, g1])[:, None]  # (S=2, F=1, T, T)
    return ResponseTable(
        ["s0", "s1"], [FC], THETA, gamma, analytic_provenance("PH-08 table"), name="ph08"
    )


E1 = KnifeEdge("e1", np.array([0.4, -3.0, 1.45]), np.array([0.4, 3.0, 1.45]))


def _mechanisms(image_path_edges: bool = True) -> list:  # type: ignore[type-arg]
    """As ``from_scenario`` builds them: the knife edge acts on the LOS and on the legs of
    the ground and wall image paths (section 9.2, "applies to the paths it intersects")."""
    legs = (E1,) if image_path_edges else ()
    ground = GroundMechanism(GroundPlane(15.0, 0.005, "V", 0.0), edges=legs)
    w1 = Wall("w1", np.array([[-8.0, -4, 0], [8.0, -4, 0], [8.0, -4, 4], [-8.0, -4, 4]]), 5.0, 0.05)
    w2 = Wall(
        "w2",
        np.array([[-8.0, 5, 0], [8.0, 5, 0], [8.0, 5, 4], [-8.0, 5, 4]]),
        3.0,
        0.01,
        roughness_m=0.002,
    )
    walls = WallsMechanism((w1, w2), "V", edges=legs)
    edge = KnifeEdgeMechanism((E1,))
    return [ground, walls, edge]


def _panel() -> PanelModel:
    grid = PanelGrid(
        Pose([0.0, 3.0, 1.5], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]), 8, 6, LAM / 2, LAM / 2
    )
    return PanelModel("ris0", grid, _table(), cell_model="point")


def _scene(swap: bool):  # type: ignore[no-untyped-def]
    ant_a = CosPower(1.5, np.array([1.0, 0.0, 0.0]))
    ant_b = CosPower(3.0, np.array([-1.0, 0.0, 0.0]))
    if not swap:
        return two_node(
            A,
            B,
            FC,
            mechanisms=_mechanisms(),
            panels=[_panel()],
            ant_tx=ant_a,
            ant_rx=ant_b,
            tx="a",
            rx="b",
        )
    return two_node(
        B,
        A,
        FC,
        mechanisms=_mechanisms(),
        panels=[_panel()],
        ant_tx=ant_b,
        ant_rx=ant_a,
        tx="b",
        rx="a",
    )


def _path_terms(model, lk: str) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[no-untyped-def]
    """Per deterministic (non-RIS) path of ``lk``: the coefficient at the carrier and kind."""
    pb = model.snapshot(Fraction(0)).paths[lk]
    fc = model.link_fc_hz(lk)
    return pb.amp * np.exp(-2j * math.pi * np.mod(fc * pb.delay_s, 1.0)), pb.kind


def test_ph08_full_scene_is_reciprocal(rng: np.random.Generator):
    ab_model, ba_model = _scene(False), _scene(True)
    # every deterministic path on its own (same order both ways: LOS, ground, w1, w2)
    t_ab, k_ab = _path_terms(ab_model, link("a", "b"))
    t_ba, k_ba = _path_terms(ba_model, link("b", "a"))
    assert np.array_equal(k_ab, k_ba) and k_ab.size == 4
    assert np.all(np.abs(t_ab - t_ba) <= 1e-12 * np.abs(t_ab))
    scale = {int(k): float(np.sum(np.abs(t_ab[k_ab == k]))) for k in np.unique(k_ab)}
    for _ in range(5):
        cfg = {"ris0": PanelConfig("ris0", rng.integers(0, 2, 48).astype(np.int16))}
        ab = by_kind(ab_model, link("a", "b"), cfg)
        ba = by_kind(ba_model, link("b", "a"), cfg)
        assert (
            set(ab)
            == set(ba)
            == {
                -1,
                int(PathKind.LOS),
                int(PathKind.GROUND),
                int(PathKind.WALL),
                int(PathKind.RIS_TILE),
            }
        )
        for k, v in ab.items():
            assert abs(v) > 0, k
            ref = abs(v) if k in (-1, int(PathKind.RIS_TILE)) else scale[k]
            assert abs(v - ba[k]) <= 1e-12 * ref, k
        # each ground hop variant on its own (1 <-> 2)
        rab = row_values(ab_model, "ris0", link("a", "b"), cfg)
        rba = row_values(ba_model, "ris0", link("b", "a"), cfg)
        for v, w in ((0, 0), (1, 2), (2, 1), (3, 3)):
            assert abs(rab[v] - rba[w]) <= 1e-12 * abs(rab[v])


def test_ph08_every_mechanism_is_active():
    m = _scene(False)
    snap = m.snapshot(Fraction(0))
    pb = snap.paths[link("a", "b")]
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.GROUND, PathKind.WALL, PathKind.WALL]
    edge = m.mechanisms[2]
    f = edge.factor(A, B, FC)
    assert 0.3 < abs(f) < 0.9  # a real obstruction (nu ~ 0.3)
    los_free = two_node(
        A,
        B,
        FC,
        ant_tx=CosPower(1.5, np.array([1.0, 0, 0])),
        ant_rx=CosPower(3.0, np.array([-1.0, 0, 0])),
    )
    assert abs(pb.amp[0] - los_free.snapshot(0).paths[link("a", "b")].amp[0] * f) <= 1e-15 * abs(
        pb.amp[0]
    )
    # the edge also diffracts image paths whose legs cross it (ground and wall w1 here)
    plain = two_node(
        A,
        B,
        FC,
        mechanisms=_mechanisms(image_path_edges=False),
        ant_tx=CosPower(1.5, np.array([1.0, 0, 0])),
        ant_rx=CosPower(3.0, np.array([-1.0, 0, 0])),
    )
    ratio = np.abs(pb.amp[1:] / plain.snapshot(0).paths[link("a", "b")].amp[1:])
    assert ratio[0] < 0.5 and ratio[1] < 0.9 and np.all(ratio > 0)
    assert snap.ris["ris0"].rows == tuple((link("a", "b"), v) for v in range(4))
    assert snap.ris["ris0"].B is not None  # angle table: the B form
    assert snap.diag["max_off_plane_deg/ris0"] > 0.0


def test_ph08_asymmetric_analytic_table_raises():
    with pytest.raises(ReciprocityError):
        _table(symmetric=False)


def test_ph08_rows_follow_the_symmetric_angle_table():
    """The angle table is really exercised: replacing it by its theta-independent
    (normal-incidence) value changes the RIS term by more than 1 %."""
    m = _scene(False)
    cfg = {"ris0": PanelConfig("ris0", np.zeros(48, dtype=np.int16))}
    ris = by_kind(m, link("a", "b"), cfg)[int(PathKind.RIS_TILE)]
    flat = _table().gamma[:, :, THETA.size // 2, THETA.size // 2][:, :, None, None]
    tflat = ResponseTable(["s0", "s1"], [FC], [0.0], flat, analytic_provenance())
    grid = PanelGrid(
        Pose([0.0, 3.0, 1.5], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]), 8, 6, LAM / 2, LAM / 2
    )
    m2 = two_node(
        A,
        B,
        FC,
        mechanisms=_mechanisms(),
        panels=[PanelModel("ris0", grid, tflat, cell_model="point")],
        ant_tx=CosPower(1.5, np.array([1.0, 0.0, 0.0])),
        ant_rx=CosPower(3.0, np.array([-1.0, 0.0, 0.0])),
    )
    ris2 = by_kind(m2, link("a", "b"), cfg)[int(PathKind.RIS_TILE)]
    assert abs(ris - ris2) > 0.01 * abs(ris)
    assert math.isfinite(abs(ris))
