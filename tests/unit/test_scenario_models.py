"""Scenario model rules (DESIGN.md 3.1, 4.9, 9.1, 9.2).

Each test mutates one field of a valid scenario and checks that exactly the intended rule
fires (SchemaError with the rule's message), or that a valid variant passes with the exact
parsed value. Field names are the frozen names of section 9.2.
"""

from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from sremu.errors import SchemaError
from sremu.scenario.load import parse_scenario, read_document
from sremu.scenario.models import FloatNotExactError, RisPanel

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
LAB_DOC = read_document(REPO / "scenarios" / "lab_like_A.yaml")


def lab() -> dict[str, Any]:
    return copy.deepcopy(LAB_DOC)


def bad(doc: dict[str, Any], match: str) -> None:
    with pytest.raises(SchemaError, match=match):
        parse_scenario(doc)


def test_frozen_section_9_2_top_level_fields():
    from sremu.scenario.models import Scenario

    assert list(Scenario.model_fields) == [
        "schema_version", "name", "seed", "description", "numerics", "environment", "nodes",
        "ris", "controllers", "run", "record", "traffic",
    ]  # fmt: skip
    sc = parse_scenario(lab())
    n = sc.numerics
    assert (n.page_target_s, n.interp_rel_error, n.t_u_max_s, n.t_u_s) == (
        Fraction(1, 2000),
        1e-3,
        Fraction(1, 100),
        None,
    )
    assert (
        n.fd_error_target_db,
        n.narrowband_phase_rad,
        n.point_max_pitch_wavelengths,
        n.subcell_max_wavelengths,
        n.richardson_tol_db,
    ) == (-60.0, 0.05, 0.5, 0.2, 0.05)
    assert (n.deterministic, n.backend, n.verify_interp) == (True, "numba_cpu", True)
    r = sc.run
    assert (r.mode, r.coupling, r.late_policy, r.on_stall, r.stall_timeout_s) == (
        "lockstep",
        "barrier",
        "apply_on_arrival",
        "wait",
        2.0,
    )
    assert r.paced.d_lag_s == Fraction(1, 1000)
    assert sc.record.model_dump() == {
        "iq": True,
        "events": True,
        "metrics": True,
        "snapshots": False,
    }
    s = sc.nodes[1].streams[0]
    assert (s.full_scale_dbm, s.gain_db, s.noise_figure_db, s.adc_clip) == (0.0, 0.0, 7.0, False)
    assert (s.lo.cfo_hz, s.lo.linewidth_hz) == (0.0, 0.0)
    p = sc.ris[0]
    assert p.failures is None and p.transition.classes == "single"
    assert p.capabilities.max_command_rate_hz == 1e4
    c = sc.controllers[0]
    assert c.observation.fields == ["rx_power_dbm", "ris_state"]
    assert c.observation.window_s == Fraction(1, 1000)


def test_exact_quantities():
    d = lab()
    d["run"]["duration_s"] = "1/3"
    d["nodes"][0]["streams"][0]["rate_sps"] = 1000000  # ints are exact
    for s in d["nodes"][1:]:
        s["streams"][0]["rate_sps"] = "1e6"
    sc = parse_scenario(d)
    assert sc.run.duration_s == Fraction(1, 3)
    assert sc.nodes[0].streams[0].rate_sps == Fraction(1_000_000)
    for field, value in (("duration_s", 0.5), ("duration_s", True)):
        d = lab()
        d["run"][field] = value
        with pytest.raises(SchemaError):
            parse_scenario(d)
    d = lab()
    d["run"]["duration_s"] = 0.5
    with pytest.raises(FloatNotExactError, match=r"'0.5'"):
        parse_scenario(d)
    for text, msg in (("0", "> 0"), ("-1e-3", "> 0"), ("5 s", "not an exact")):
        d = lab()
        d["run"]["duration_s"] = text
        bad(d, msg)
    d = lab()
    d["controllers"][0]["min_latency_s"] = "-1e-4"
    bad(d, ">= 0")


def test_strict_scalar_types():
    for path, value in [
        (("seed",), "20261010"),
        (("seed",), 1.0),
        (("seed",), -1),
        (("numerics", "deterministic"), "yes"),
        (("numerics", "deterministic"), 1),
        (("nodes", 0, "streams", 0, "fc_hz"), "3.5e9"),
        (("nodes", 0, "streams", 0, "fc_hz"), True),
        (("nodes", 0, "streams", 0, "fc_hz"), 0.0),
        (("nodes", 0, "streams", 0, "fc_hz"), float("inf")),
        (("ris", 0, "grid", "n_u"), 24.0),
        (("nodes", 0, "id"), "a.b"),
        (("nodes", 0, "id"), ""),
        (("name",), "has space"),
    ]:
        d = lab()
        target = d
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
        with pytest.raises(SchemaError):
            parse_scenario(d)


def test_extra_keys_forbidden_everywhere():
    for where in (
        lambda d: d,
        lambda d: d["nodes"][0],
        lambda d: d["ris"][0]["grid"],
        lambda d: d["run"],
        lambda d: (
            d["nodes"][0]["streams"][0]["lo"]
            if "lo" in d["nodes"][0]["streams"][0]
            else d["controllers"][0]["latency"]
        ),
    ):
        d = lab()
        where(d)["unexpected_m"] = 1
        bad(d, "Extra inputs are not permitted")


def test_streams_direction_rules():
    d = lab()
    d["nodes"][0]["streams"][0]["noise_figure_db"] = 5
    bad(d, "RX streams only")
    d = lab()
    d["nodes"][0]["streams"][0]["adc_clip"] = True
    bad(d, "RX streams only")
    d = lab()
    d["nodes"][1]["streams"][0]["occupied_bw_hz"] = 1e5
    bad(d, "TX streams only")


def test_node_position_xor_trajectory_and_trajectories():
    d = lab()
    d["nodes"][1]["trajectory"] = {"model": "static", "position_m": [1, 0, 0]}
    bad(d, "exactly one of position_m or trajectory")
    d = lab()
    del d["nodes"][1]["position_m"]
    bad(d, "exactly one of position_m or trajectory")
    good = [
        {"model": "static", "position_m": [1, 0, 0]},
        {"model": "constant_velocity", "start_m": [1, 0, 0], "velocity_mps": [0, 1.5, 0]},
        {"model": "waypoints", "t_s": ["0", "2.5"], "positions_m": [[1, 0, 0], [1, 1, 0]]},
        {"model": "waypoints", "t_s": [0], "positions_m": [[1, 0, 0]]},
        {"model": "rwp_1d", "p0_m": [1, -1, 0], "p1_m": [1, 1, 0], "speed_mps": 1.0},
        {
            "model": "rwp_2d",
            "bounds_m": [[1, -1], [2, 1]],
            "z_m": 0,
            "speed_mps": 1.0,
            "pause_s": "0.5",
        },
        {"model": "external", "initial_m": [1, 0, 0]},
    ]
    for tr in good:
        d = lab()
        del d["nodes"][1]["position_m"]
        d["nodes"][1]["trajectory"] = tr
        sc = parse_scenario(d)
        assert sc.nodes[1].trajectory.model == tr["model"]
    for tr, msg in [
        (
            {"model": "waypoints", "t_s": ["1", "1"], "positions_m": [[0, 0, 0], [1, 0, 0]]},
            "strictly increasing",
        ),
        (
            {"model": "waypoints", "t_s": ["1", "2"], "positions_m": [[0, 0, 0]]},
            "as many t_s as positions_m",
        ),
        (
            {"model": "waypoints", "t_s": ["1", 2.5], "positions_m": [[0, 0, 0], [1, 0, 0]]},
            "floats are rejected",
        ),
        (
            {"model": "rwp_1d", "p0_m": [1, 0, 0], "p1_m": [1, 0, 0], "speed_mps": 1.0},
            "p0_m != p1_m",
        ),
        (
            {
                "model": "rwp_2d",
                "bounds_m": [[2, 0], [1, 1]],
                "z_m": 0,
                "speed_mps": 1.0,
            },
            "x0 < x1",
        ),
        ({"model": "teleport", "position_m": [0, 0, 0]}, "does not match any of the expected tags"),
        ({"model": "trace", "file": "x.csv"}, "data_class"),
    ]:
        d = lab()
        del d["nodes"][1]["position_m"]
        d["nodes"][1]["trajectory"] = tr
        bad(d, msg)


def test_antenna_rules():
    d = lab()
    d["nodes"][0]["antenna"] = {"pattern": {"cos_power": {"alpha": 3}}}
    bad(d, "nonzero boresight")
    d["nodes"][0]["antenna"]["boresight"] = [-1, 0, 0]
    assert parse_scenario(d).nodes[0].antenna.pattern.cos_power.alpha == 3.0
    d = lab()
    d["nodes"][0]["antenna"]["boresight"] = [1, 0, 0]
    bad(d, "isotropic antenna takes no boresight")


def test_panel_rules():
    d = lab()
    d["ris"][0]["pose"]["u_axis"] = [0.1, 1, 0]
    bad(d, "orthogonal")
    d = lab()
    d["ris"][0]["tiles"]["n_tiles_u"] = 5  # 24 cells are not divisible by 5
    bad(d, "multiples")
    d = lab()
    d["ris"][0]["control"] = {"granularity": "tile"}
    assert parse_scenario(d).ris[0].n_controls == 3
    d["ris"][0]["control"] = {"granularity": "column"}
    assert parse_scenario(d).ris[0].n_controls == 24
    d = lab()
    del d["ris"][0]["tiles"]
    d["ris"][0]["control"] = {"granularity": "tile"}
    bad(d, "needs a tiles layout")
    d = lab()
    d["ris"][0]["control"] = {
        "granularity": "groups",
        "groups": [list(range(60)), list(range(60, 120))],
    }
    p: RisPanel = parse_scenario(d).ris[0]
    assert p.n_cells == 120 and p.n_controls == 2
    d["ris"][0]["control"]["groups"] = [list(range(60)), list(range(59, 120))]
    bad(d, "partition")
    d = lab()
    d["ris"][0]["initial"] = {"states": [0, 1]}
    bad(d, "120 controls")
    d["ris"][0]["initial"] = {"fill": 0, "states": [0] * 120}
    bad(d, "exactly one of initial.fill")
    d = lab()
    d["ris"][0]["transition"] = {"model": "linear"}
    bad(d, "needs transition.duration_s or capabilities.reconfiguration_time_s")
    d["ris"][0]["capabilities"] = {"reconfiguration_time_s": "2e-5"}
    assert parse_scenario(d).ris[0].capabilities.reconfiguration_time_s == Fraction(1, 50000)
    d = lab()
    d["ris"][0]["states"] = {"library": "on_off", "file": "x.yaml"}
    bad(d, "exactly one of states.library or states.file")
    d = lab()
    d["ris"][0]["states"]["efficiency"] = 0.0
    bad(d, "greater than 0")
    d = lab()
    d["ris"][0]["failures"] = {"dead_rate": 0.6, "stuck_rate": 0.6}
    bad(d, "<= 1")
    d["ris"][0]["failures"] = {"gamma_dead": [0.9, 0.9]}
    bad(d, "passive")
    d["ris"][0]["failures"] = {"dead_rate": 0.1, "gamma_dead": [-1, 0]}
    assert parse_scenario(d).ris[0].failures.gamma_dead == -1 + 0j
    d["ris"][0]["tiles"]["gap_gamma"] = -1
    assert parse_scenario(d).ris[0].tiles.gap_gamma == complex(-1, 0)
    d["ris"][0]["tiles"]["gap_gamma"] = [0.8, 0.61]  # |0.8 + 0.61j| = 1.006 > 1
    bad(d, "gap_gamma must be passive")


def test_controller_and_run_rules():
    d = lab()
    d["controllers"][0]["latency"]["fixed_s"] = "1e-4"  # below min_latency_s 2e-4
    bad(d, "lookahead")
    d = lab()
    d["controllers"][0]["latency"]["jitter"] = "gauss:1e-3"
    bad(d, "uniform:a,b")
    d = lab()
    d["controllers"][0]["latency"]["jitter"] = "uniform:2e-3,1e-3"
    bad(d, "0 <= a <= b")
    d = lab()
    del d["controllers"][0]["impl"]
    bad(d, "needs impl")
    d = lab()
    d["controllers"][0]["kind"] = "external"
    bad(d, "takes no impl")
    d = lab()
    d["controllers"][0]["observation"] = {"fields": ["rx_power_dbm", "telepathy"]}
    bad(d, "rx_power_dbm")
    d = lab()
    d["run"]["coupling"] = "async"
    bad(d, "lockstep' with coupling 'async' is rejected")
    d["run"]["mode"] = "paced"
    assert parse_scenario(d).run.coupling == "async"
    d = lab()
    d["numerics"]["t_u_s"] = "2e-2"
    bad(d, "exceeds t_u_max_s")
    d["numerics"]["t_u_s"] = "1e-3"
    assert parse_scenario(d).numerics.t_u_s == Fraction(1, 1000)


def test_environment_rules():
    d = lab()
    d["environment"] = {
        "ground": {"eps_r": 15, "sigma_s_per_m": 1e-3, "polarization": "H"},
        "walls": [
            {
                "id": "w1",
                "vertices_m": [[3, -1, -1], [3, 1, -1], [3, 1, 1]],
                "eps_r": 5.3,
                "sigma_s_per_m": 0.03,
            }
        ],
        "occluders": [
            {
                "id": "o1",
                "vertices_m": [[1, -0.5, -1], [1, 0.5, -1], [1, 0, 1]],
                "kind": "partition",
                "transmission_loss_db": 20,
            }
        ],
        "edges": [{"id": "e1", "p0_m": [0, 0, 2], "p1_m": [1, 0, 2]}],
        "diffuse": {"k_factor_db": 10, "max_doppler_hz": 5},
    }
    env = parse_scenario(d).environment
    assert env.ground.height_m == 0.0 and env.diffuse.n_sinusoids == 48
    d["environment"]["occluders"][0]["kind"] = "opaque"
    bad(d, "takes no transmission_loss_db")
    del d["environment"]["occluders"][0]["transmission_loss_db"]
    parse_scenario(d)
    d["environment"]["diffuse"] = {"k_factor_db": 10, "max_doppler_hz": 5, "from_motion": True}
    bad(d, "exactly one of max_doppler_hz or from_motion")
    d["environment"]["diffuse"] = {"k_factor_db": 10, "from_motion": True, "n_sinusoids": 16}
    bad(d, "greater than or equal to 48")
    d["environment"]["diffuse"] = {"k_factor_db": 10, "from_motion": True}
    d["environment"]["walls"][0]["vertices_m"] = [[0, 0, 0], [1, 0, 0]]
    bad(d, "at least 3")


def test_convenience_views():
    sc = parse_scenario(lab())
    assert list(sc.streams()) == ["ap.tx0", "rx1.rx0", "rx2.rx0"]
    assert sc.node("rx2").id == "rx2" and sc.panel("ris0").n_cells == 120
    with pytest.raises(KeyError):
        sc.node("nope")
