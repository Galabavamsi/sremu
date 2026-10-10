"""D18: ``from_scenario`` attaches every WP3 mechanism a scenario enables (DESIGN.md 2.2, 9.2).

A section 9.2 scenario with ground, two walls, a partition, a knife edge and diffuse
multipath, validated by the WP1 pydantic model, builds a model whose snapshot equals the
mechanisms built directly from the same entries.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from sremu.contracts import PathKind
from sremu.physics.diffraction import KnifeEdgeMechanism
from sremu.physics.diffuse import DiffuseMechanism, find_diffuse
from sremu.physics.ground import GroundMechanism, GroundPlane
from sremu.physics.los import Occluder
from sremu.physics.propagation import ScenePropagation, from_scenario
from sremu.physics.walls import Wall, WallsMechanism
from sremu.scenario.models import Scenario
from tests.physics._wp3_scenes import link, two_node

pytestmark = pytest.mark.physics

FC = 3.5e9
A = [-3.0, -0.5, 1.6]
B = [3.0, 0.8, 1.2]
W1 = [[-8.0, -4.0, 0.0], [8.0, -4.0, 0.0], [8.0, -4.0, 4.0], [-8.0, -4.0, 4.0]]
W2 = [[-8.0, 5.0, 0.0], [8.0, 5.0, 0.0], [8.0, 5.0, 4.0], [-8.0, 5.0, 4.0]]
PART = [[-1.0, -2.0, 0.0], [-1.0, 2.0, 0.0], [-1.0, 2.0, 3.0], [-1.0, -2.0, 3.0]]


def _doc() -> dict:  # type: ignore[type-arg]
    return {
        "schema_version": "1.0",
        "name": "wp3_full_env",
        "seed": 31337,
        "nodes": [
            {
                "id": "a",
                "position_m": A,
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
                "position_m": B,
                "antenna": {"pattern": "isotropic"},
                "streams": [{"name": "rx0", "direction": "rx", "fc_hz": FC, "rate_sps": "1e6"}],
            },
        ],
        "environment": {
            "ground": {"eps_r": 15.0, "sigma_s_per_m": 0.005, "polarization": "H", "height_m": 0.0},
            "walls": [
                {"id": "w1", "vertices_m": W1, "eps_r": 5.0, "sigma_s_per_m": 0.05},
                {
                    "id": "w2",
                    "vertices_m": W2,
                    "eps_r": 3.0,
                    "sigma_s_per_m": 0.01,
                    "roughness_m": 0.002,
                },
            ],
            "occluders": [
                {"id": "p1", "vertices_m": PART, "kind": "partition", "transmission_loss_db": 8.0}
            ],
            "edges": [{"id": "e1", "p0_m": [0.4, -3.0, 1.45], "p1_m": [0.4, 3.0, 1.45]}],
            "diffuse": {"k_factor_db": 9.0, "max_doppler_hz": 4.0},
        },
        "run": {
            "mode": "lockstep",
            "coupling": "barrier",
            "duration_s": "1",
            "late_policy": "apply_on_arrival",
        },
    }


def _direct(*, image_path_edges: bool = True) -> ScenePropagation:
    from sremu.physics.diffraction import KnifeEdge

    part = Occluder("p1", np.array(PART), "partition", 8.0)
    plane = GroundPlane(15.0, 0.005, "H", 0.0)
    e1 = KnifeEdge("e1", np.array([0.4, -3.0, 1.45]), np.array([0.4, 3.0, 1.45]))
    legs = (e1,) if image_path_edges else ()
    walls = WallsMechanism(
        (
            Wall("w1", np.array(W1), 5.0, 0.05),
            Wall("w2", np.array(W2), 3.0, 0.01, roughness_m=0.002),
        ),
        "H",
        (part,),
        edges=legs,
    )
    edge = KnifeEdgeMechanism((e1,))
    return two_node(
        A,
        B,
        FC,
        mechanisms=[GroundMechanism(plane, (part,), edges=legs), walls, edge],
        occluders=[part],
    )


@pytest.mark.parametrize("via", ["pydantic", "mapping", "json_file"])
def test_d18_full_environment_matches_direct_mechanisms(via: str, tmp_path):  # type: ignore[no-untyped-def]
    doc = _doc()
    if via == "pydantic":
        model = from_scenario(Scenario.model_validate(doc))
    elif via == "mapping":
        model = from_scenario(doc, base_dir=tmp_path)
    else:
        path = tmp_path / "s.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        model = from_scenario(path)
    kinds = [type(m) for m in model.mechanisms]
    assert kinds == [GroundMechanism, WallsMechanism, KnifeEdgeMechanism, DiffuseMechanism]
    pb = model.snapshot(0).paths[link()]
    ref = _direct().snapshot(0).paths[link()]
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.GROUND, PathKind.WALL, PathKind.WALL]
    assert np.array_equal(pb.kind, ref.kind) and np.array_equal(pb.delay_s, ref.delay_s)
    assert np.max(np.abs(pb.amp - ref.amp)) <= 1e-15 * np.max(np.abs(ref.amp))
    # the LOS carries the partition and the knife edge: |V| = 10^(-8/20) |F(nu)| < 10^(-8/20)
    free = two_node(A, B, FC).snapshot(0).paths[link()].amp[0]
    assert abs(pb.amp[0]) < abs(free) * 10 ** (-8 / 20)
    # section 9.2: the edge "applies to the paths it intersects": the ground path (whose Tx
    # leg crosses the screen near the ground) and wall w1's path are diffracted too, w2's
    # path is not (its legs pass beside the edge's extent)
    undiff = _direct(image_path_edges=False).snapshot(0).paths[link()]
    ratio = np.abs(pb.amp / undiff.amp)
    assert ratio[1] < 0.1 and 0.0 < ratio[2] < 1.0 - 1e-3 and ratio[3] == 1.0
    mech = find_diffuse(model)
    assert mech is not None and mech.seed == 31337
    terms = mech.terms(model, model.snapshot(0))[link()]
    p_ref = float(np.sum(np.abs(pb.amp) ** 2))
    assert terms[0].amp == pytest.approx(math.sqrt(p_ref / 10**0.9), rel=1e-14)


def test_d18_describe_is_json_ready():
    model = from_scenario(_doc())
    out = [m.describe() for m in model.mechanisms]
    text = json.dumps(out)
    assert [d["mechanism"] for d in out] == ["ground", "walls", "knife_edge", "diffuse_sos"]
    assert "P.2040" not in text  # materials are given directly; no unverified table used
