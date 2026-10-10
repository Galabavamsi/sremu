"""PH-27: occluders and partitions (DESIGN.md sections 5.2, 5.9, 10.2).

Spec row: segment-polygon hit/miss cases; a partition of L dB gives exactly -L dB on the
LOS; an opaque occluder gives 0 (exact). The occluder code is WP2's (``physics/los.py``,
``physics/geometry.py``); WP3 owns this test and applies the same rule to the legs of the
ground and wall image paths (also checked here).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PathKind
from sremu.errors import ConfigurationError
from sremu.physics.geometry import segment_hits_polygon, segments_hit_polygon
from sremu.physics.ground import GroundMechanism, GroundPlane
from sremu.physics.los import Occluder, los_visibility, occluders_from_spec
from sremu.physics.propagation import from_scenario
from sremu.physics.walls import Wall, WallsMechanism
from tests.physics._wp3_scenes import link, two_node

pytestmark = pytest.mark.physics

FC = 3.5e9
# unit square in the plane x = 1, spanning y, z in [0, 1]
SQUARE = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
# L-shaped (non-convex) polygon in the plane z = 0: the notch is [1, 2] x [1, 2]
L_SHAPE = np.array([[0.0, 0, 0], [2.0, 0, 0], [2.0, 1, 0], [1.0, 1, 0], [1.0, 2, 0], [0.0, 2, 0]])


@pytest.mark.parametrize(
    ("a", "b", "poly", "hit"),
    [
        ((0, 0.5, 0.5), (2, 0.5, 0.5), SQUARE, True),  # through the interior
        ((0, 0.5, 0.5), (2, 0.6, 0.4), SQUARE, True),  # oblique, interior
        ((0, 1.5, 0.5), (2, 1.5, 0.5), SQUARE, False),  # crosses the plane outside
        ((0, 0.5, 0.5), (0.9, 0.5, 0.5), SQUARE, False),  # stops short of the plane
        ((0, 0.5, 0.5), (1.0, 0.5, 0.5), SQUARE, False),  # touches the plane at its end
        ((1.0, 0.2, 0.2), (1.0, 0.8, 0.8), SQUARE, False),  # lies in the plane
        ((0, 1.0, 0.5), (2, 1.0, 0.5), SQUARE, True),  # through an edge (boundary counts)
        ((0, 0.0, 0.0), (2, 0.0, 0.0), SQUARE, True),  # through a vertex
        ((0, 0.5, 0.5), (2, 0.5, 1.5), SQUARE, True),  # crosses at z = 1 (on the edge)
        ((0, 0.5, 0.5), (2, 0.5, 1.6), SQUARE, False),  # crosses at z = 1.05: outside
        ((0.5, 0.5, -1), (0.5, 0.5, 1), L_SHAPE, True),  # inside the L
        ((1.5, 1.5, -1), (1.5, 1.5, 1), L_SHAPE, False),  # in the notch of the L
        ((1.5, 0.5, 1), (1.5, 0.5, -1), L_SHAPE, True),  # reversed direction, inside
    ],
)
def test_ph27_segment_polygon_cases(a, b, poly, hit):  # type: ignore[no-untyped-def]
    assert segment_hits_polygon(a, b, poly) is hit
    assert segment_hits_polygon(b, a, poly) is hit  # symmetric in the segment ends
    vec = segments_hit_polygon(np.array([a, b], float), np.array([b, a], float), poly)
    assert vec.tolist() == [hit, hit]


def _los_amp(occ) -> complex:  # type: ignore[no-untyped-def]
    m = two_node([0.0, 0.5, 0.5], [3.0, 0.4, 0.6], FC, occluders=occ)
    pb = m.snapshot(Fraction(0)).paths[link()]
    assert pb.kind.tolist() == [PathKind.LOS]
    return complex(pb.amp[0])


@pytest.mark.parametrize("loss_db", [0.0, 3.0, 10.0, 20.0, 37.25])
def test_ph27_partition_is_exactly_minus_l_db_on_the_los(loss_db: float):
    free = _los_amp([])
    got = _los_amp([Occluder("p", SQUARE, "partition", loss_db)])
    assert got == free * 10 ** (-loss_db / 20)  # bitwise: one rounding of the same product
    assert got.imag == 0.0 and free.imag == 0.0  # no phase change [ENG]
    assert abs(20 * math.log10(abs(got / free)) + loss_db) <= 1e-12


def test_ph27_two_partitions_add_in_db_and_opaque_is_zero():
    free = _los_amp([])
    p1 = Occluder("p1", SQUARE, "partition", 6.0)
    p2 = Occluder("p2", SQUARE + np.array([1.0, 0.0, 0.0]), "partition", 4.0)
    got = _los_amp([p1, p2])
    assert abs(20 * math.log10(abs(got / free)) + 10.0) <= 1e-12
    assert _los_amp([Occluder("o", SQUARE, "opaque")]) == 0
    assert _los_amp([Occluder("o", SQUARE + np.array([0.0, 5.0, 0.0]), "opaque")]) == free  # misses


def test_ph27_visibility_factor_vectorized():
    occ = (Occluder("p", SQUARE, "partition", 20.0),)
    a = np.array([[0.0, 0.5, 0.5], [0.0, 1.5, 0.5], [0.0, 0.5, 0.5]])
    b = np.array([[2.0, 0.5, 0.5], [2.0, 1.5, 0.5], [0.5, 0.5, 0.5]])
    assert los_visibility(a, b, occ).tolist() == [0.1, 1.0, 1.0]


def test_ph27_occluder_validation():
    with pytest.raises(ConfigurationError):
        Occluder("p", SQUARE, "partition", None)
    with pytest.raises(ConfigurationError):
        occluders_from_spec([{"id": "a", "vertices_m": SQUARE}, {"id": "a", "vertices_m": SQUARE}])
    with pytest.raises(ConfigurationError, match="coplanar"):
        segment_hits_polygon(
            (0, 0, 0), (1, 1, 1), np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0.5], [0, 1, 0]])
        )


def test_ph27_scenario_names_never_imply_blockage(tmp_path):
    """Only explicit occluders block: a wall in the LOS reflects but does not attenuate it."""
    wall = Wall("w", SQUARE * [1, 4, 4] - [0, 1, 1], 5.0, 0.01)  # big wall across the LOS
    m = two_node([0.0, 0.5, 0.5], [3.0, 0.4, 0.6], FC, mechanisms=[WallsMechanism((wall,))])
    pb = m.snapshot(Fraction(0)).paths[link()]
    assert pb.amp[0] == _los_amp([])
    assert pb.kind.tolist() == [PathKind.LOS]  # the nodes are on opposite sides: no image path


def test_ph27_occluders_apply_to_the_legs_of_image_paths():
    """WP3: the same factors act on the Tx -> specular and specular -> Rx legs of the ground
    and wall paths (exact products)."""
    a, b = np.array([0.0, 0.0, 2.0]), np.array([10.0, 0.0, 2.0])
    # crosses only the Rx leg of the ground path (specular point (5, 0, 0)): x = 8, z in [0.3, 1.4]
    part = Occluder(
        "p",
        np.array([[8.0, -1, 0.3], [8.0, 1, 0.3], [8.0, 1, 1.4], [8.0, -1, 1.4]]),
        "partition",
        9.0,
    )
    plane = GroundPlane(10.0, 0.01, "V")
    free = GroundMechanism(plane).ground_path(a, b, FC)[0]
    assert GroundMechanism(plane, (part,)).ground_path(a, b, FC)[0] == free * 10 ** (-9 / 20)
    # a vertical wall at y = 3; its specular point is (5, 3, 2); an opaque plate on the Tx leg
    wall = Wall("w", np.array([[0.0, 3, 0], [10.0, 3, 0], [10.0, 3, 4], [0.0, 3, 4]]), 5.0, 0.0)
    plate = Occluder(
        "o",
        np.array([[2.5, 1.0, 1.5], [2.5, 2.0, 1.5], [2.5, 2.0, 2.5], [2.5, 1.0, 2.5]]),
        "opaque",
    )
    clear = WallsMechanism((wall,)).wall_path(wall, a, b, FC)
    blocked = WallsMechanism((wall,), occluders=(plate,)).wall_path(wall, a, b, FC)
    assert clear is not None and blocked is not None
    assert clear[0] != 0 and blocked[0] == 0 and blocked[1] == clear[1]
    assert los_visibility(a, b, (plate,)) == 1.0  # the plate is off the LOS


def test_ph27_scenario_occluders_reach_the_wp3_mechanisms(tmp_path):
    doc = {
        "schema_version": "1.0",
        "name": "ph27",
        "seed": 3,
        "nodes": [
            {
                "id": "a",
                "position_m": [0.0, 0.0, 2.0],
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
                "position_m": [10.0, 0.0, 2.0],
                "antenna": {"pattern": "isotropic"},
                "streams": [{"name": "rx0", "direction": "rx", "fc_hz": FC, "rate_sps": "1e6"}],
            },
        ],
        "environment": {
            "ground": {"eps_r": 10.0, "sigma_s_per_m": 0.01, "polarization": "V"},
            "occluders": [
                {
                    "id": "p",
                    "kind": "partition",
                    "transmission_loss_db": 9.0,
                    "vertices_m": [[8.0, -1, 0.3], [8.0, 1, 0.3], [8.0, 1, 1.4], [8.0, -1, 1.4]],
                }
            ],
        },
        "run": {
            "mode": "lockstep",
            "coupling": "barrier",
            "duration_s": "1",
            "late_policy": "apply_on_arrival",
        },
    }
    pb = from_scenario(doc, base_dir=tmp_path).snapshot(0).paths[link()]
    plane = GroundPlane(10.0, 0.01, "V")
    want = GroundMechanism(plane).ground_path([0, 0, 2.0], [10.0, 0, 2.0], FC)[0] * 10 ** (-9 / 20)
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.GROUND]
    assert pb.amp[1] == want
    assert pb.amp[0] == two_node([0, 0, 2.0], [10.0, 0, 2.0], FC).snapshot(0).paths[link()].amp[0]
