"""Scenario guards (DESIGN.md 9.5) and derived quantities.

Each guard is triggered by a minimal mutation of a valid scenario; the test checks the
guard code, its severity and the exception class ``raise_for_errors`` raises, and the
boundary on the passing side. Geometry oracles (distances, cell positions, Fraunhofer
distance, noise floor, page size) are closed forms evaluated here.
"""

from __future__ import annotations

import copy
import math
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sremu.errors import (
    BandwidthExceeded,
    ConfigurationError,
    FDBandwidthExceeded,
    GeometryError,
    OutOfDomain,
    PassivityError,
    RateMismatchError,
    SchemaError,
    TableValidationError,
    ValidityWarning,
)
from sremu.scenario.guards import (
    GuardReport,
    cell_centres,
    check_scenario,
    fd_choice,
    links,
    node_geometry,
    panel_extent_m,
)
from sremu.scenario.load import (
    dump_yaml,
    load_scenario,
    parse_scenario,
    read_document,
    resolve_scenario,
)
from sremu.units import C0

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SCEN = REPO / "scenarios"
LAB_DOC = read_document(SCEN / "lab_like_A.yaml")
QUICK_DOC = read_document(SCEN / "quickstart.yaml")
LAM = C0 / 3.5e9


def lab() -> dict[str, Any]:
    return copy.deepcopy(LAB_DOC)


def quick() -> dict[str, Any]:
    return copy.deepcopy(QUICK_DOC)


def report(doc: dict[str, Any], base: Path = SCEN, **kw: Any) -> GuardReport:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        res = resolve_scenario(parse_scenario(doc), base, warn=False, strict=False)
    return check_scenario(res, **kw)


def assert_error(rep: GuardReport, code: str, exc: type[Exception]) -> None:
    assert code in rep.codes("error"), rep.issues
    with pytest.raises(exc):
        rep.raise_for_errors()
    first = next(i for i in rep.errors if i.code == code)
    assert issubclass(first.exc, exc)


# --------------------------------------------------------------------------- shipped files


@pytest.mark.parametrize("name", ["quickstart.yaml", "lab_like_A.yaml"])
def test_shipped_scenarios_pass_every_guard_without_warnings(name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ValidityWarning)
        res = load_scenario(SCEN / name)
    rep = check_scenario(res)
    assert rep.issues == []


def test_lab_like_A_geometry_is_the_E_B_geometry():
    sc = parse_scenario(lab())
    expect = {"ap": (2.0, -30.0), "rx1": (1.2, 15.0), "rx2": (1.2, 35.0)}
    for n in sc.nodes:
        x, y, z = n.position_m
        r, ang = expect[n.id]
        assert math.hypot(x, y) == pytest.approx(r, abs=1e-9)
        assert math.degrees(math.atan2(y, x)) == pytest.approx(ang, abs=1e-7)
        assert z == 0.0
    p = sc.ris[0]
    assert (p.grid.n_u, p.grid.n_v, p.tiles.n_tiles_u, p.tiles.n_tiles_v) == (24, 5, 3, 1)
    assert (p.grid.pitch_u_m, p.tiles.gap_m) == (0.0428, 0.002)
    assert p.grid.pitch_u_m < LAM / 2  # auto picks "point" (5.4)
    assert sc.nodes[0].streams[0].occupied_bw_hz == 8.0e5
    assert sc.nodes[1].streams[0].noise_figure_db == 7.0


# --------------------------------------------------------------------------- geometry


def test_cell_centres_with_tile_gaps():
    p = parse_scenario(lab()).ris[0]
    c = cell_centres(p)
    assert c.shape == (120, 3)
    pitch, gap = 0.0428, 0.002
    # cell (iu, iv) -> index iv*24 + iu; u axis is +y, v = n x u = +z
    iu = np.arange(24)
    tile = iu // 8
    y_expect = (iu - 11.5) * pitch + (tile - 1) * gap
    np.testing.assert_allclose(c[:24, 1], y_expect, atol=1e-15)
    np.testing.assert_allclose(c[::24, 2], (np.arange(5) - 2) * pitch, atol=1e-15)
    assert np.all(c[:, 0] == 0.0)
    assert c[8, 1] - c[7, 1] == pytest.approx(pitch + gap, abs=1e-15)  # across a tile gap
    assert c[7, 1] - c[6, 1] == pytest.approx(pitch, abs=1e-15)
    su, sv = panel_extent_m(p)
    assert su == pytest.approx(24 * pitch + 2 * gap, abs=1e-15)
    assert sv == pytest.approx(5 * pitch, abs=1e-15)


def test_node_geometry_reachable_sets():
    d = quick()
    d["nodes"][1].pop("position_m")
    d["nodes"][1]["trajectory"] = {
        "model": "constant_velocity",
        "start_m": [1, -3, 0],
        "velocity_mps": [0, 2, 0],
    }
    sc = parse_scenario(d)
    g = node_geometry(sc.nodes[1], Fraction(2), None)  # from (1,-3,0) to (1,1,0)
    assert g.speed_max_mps == 2.0
    np.testing.assert_allclose(g.min_distance(np.array([[0.0, 0.0, 0.0]])), [1.0])
    np.testing.assert_allclose(g.min_distance(np.array([[1.0, 3.0, 0.0]])), [2.0])
    d["nodes"][1]["trajectory"] = {
        "model": "waypoints",
        "t_s": ["1", "3"],
        "positions_m": [[2, 0, 0], [2, 4, 0]],
    }
    g = node_geometry(parse_scenario(d).nodes[1], Fraction(2), None)
    np.testing.assert_allclose(g.pts, [[2, 0, 0], [2, 0, 0], [2, 2, 0]])  # held, then half
    assert g.speed_max_mps == pytest.approx(2.0)
    d["nodes"][1]["trajectory"] = {
        "model": "rwp_2d",
        "bounds_m": [[1, -1], [3, 1]],
        "z_m": 0.5,
        "speed_mps": 1,
    }
    g = node_geometry(parse_scenario(d).nodes[1], Fraction(2), None)
    np.testing.assert_allclose(
        g.min_distance(np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 0.0]])), [1.0, 0.5]
    )
    assert g.initial is None


# --------------------------------------------------------------------------- errors


def test_duplicate_ids():
    d = lab()
    d["nodes"][2]["id"] = "rx1"
    assert_error(report(d), "DUPLICATE_ID", SchemaError)
    d = lab()
    d["ris"][0]["id"] = "ap"  # a panel may not reuse a node id
    d["controllers"] = []
    assert_error(report(d), "DUPLICATE_ID", SchemaError)
    d = lab()
    d["nodes"][0]["streams"].append(dict(d["nodes"][0]["streams"][0]))
    assert_error(report(d), "DUPLICATE_ID", SchemaError)
    d = lab()
    d["controllers"].append(dict(d["controllers"][0]))
    assert_error(report(d), "DUPLICATE_ID", SchemaError)


def test_unresolved_references():
    d = lab()
    d["traffic"][0]["stream"] = "ap.tx9"
    assert_error(report(d), "UNRESOLVED_REF", SchemaError)
    d = lab()
    d["traffic"][0]["stream"] = "rx1.rx0"  # not a TX stream
    assert_error(report(d), "UNRESOLVED_REF", SchemaError)
    d = lab()
    d["ris"][0]["states"] = {"library": "no_such_library"}
    assert_error(report(d), "STATES", SchemaError)
    d = lab()
    d["ris"][0]["states"] = {"file": "missing_states.yaml"}
    assert_error(report(d), "STATES", SchemaError)
    d = lab()
    d["ris"][0]["states"]["assume_mag"] = {"sigma1": 1.0}
    assert_error(report(d), "STATES", SchemaError)


def test_rate_mismatch_and_links():
    d = lab()
    d["nodes"][2]["streams"][0]["rate_sps"] = "2e6"  # same carrier, overlapping band
    assert_error(report(d), "RATE_MISMATCH", RateMismatchError)
    d = lab()
    d["nodes"][2]["streams"][0]["rate_sps"] = "2e6"
    d["nodes"][2]["streams"][0]["fc_hz"] = 3.5e9 + 1.5e6  # |dfc| = (f1 + f2)/2: no overlap
    lks, bad = links(parse_scenario(d))
    assert lks == [("ap.tx0", "rx1.rx0")] and bad == []
    d["nodes"][2]["streams"][0]["fc_hz"] = 3.5e9 + 1.5e6 - 1.0
    assert links(parse_scenario(d))[1]
    d = lab()
    d["nodes"][2]["streams"][0]["fc_hz"] = 3.5e9 + 0.5e6  # equal rates: |dfc| < f_s/2 needed
    assert links(parse_scenario(d))[0] == [("ap.tx0", "rx1.rx0")]
    d["nodes"][2]["streams"][0]["fc_hz"] = 3.5e9 + 0.5e6 - 1.0
    assert len(links(parse_scenario(d))[0]) == 2


def test_cell_closer_than_one_wavelength_is_an_error():
    # node on the panel normal through the corner cell; nearest cell centre at distance x
    p = parse_scenario(lab()).ris[0]
    corner = cell_centres(p)[0]
    for x, code, sev in (
        (0.99 * LAM, "CELL_TOO_CLOSE", "error"),
        (1.01 * LAM, "CELL_NEAR", "warning"),
        (2.99 * LAM, "CELL_NEAR", "warning"),
        (3.01 * LAM, None, None),
    ):
        d = lab()
        d["nodes"][2]["position_m"] = [float(x), float(corner[1]), float(corner[2])]
        rep = report(d)
        if code is None:
            assert not ({"CELL_TOO_CLOSE", "CELL_NEAR"} & rep.codes()), x
        else:
            assert code in rep.codes(sev), (x, rep.issues)
    d = lab()
    d["nodes"][2]["position_m"] = [0.05, 0.0, 0.0]
    assert_error(report(d), "CELL_TOO_CLOSE", GeometryError)


def test_moving_node_passing_the_panel_is_caught():
    d = lab()
    d["nodes"][2].pop("position_m")
    # passes 0.02 m in front of the panel centre at t = 1 s; duration 5 s
    d["nodes"][2]["trajectory"] = {
        "model": "constant_velocity",
        "start_m": [0.02, -1.0, 0.0],
        "velocity_mps": [0.0, 1.0, 0.0],
    }
    assert_error(report(d), "CELL_TOO_CLOSE", GeometryError)
    d["run"]["duration_s"] = "0.3"  # stops at y = -0.7 m, 0.205 m from the edge cell
    rep = report(d)
    assert "CELL_TOO_CLOSE" not in rep.codes()
    assert "CELL_NEAR" in rep.codes("warning")  # 0.205 m < 3 lambda = 0.257 m


def test_fd_bandwidth_limit():
    d = lab()
    d["nodes"][0]["streams"][0]["occupied_bw_hz"] = 0.8e6  # exactly 0.8 f_s: allowed
    assert "FD_BANDWIDTH" not in report(d).codes()
    d["nodes"][0]["streams"][0]["occupied_bw_hz"] = 0.80001e6
    assert_error(report(d), "FD_BANDWIDTH", FDBandwidthExceeded)
    d["numerics"]["fd_error_target_db"] = -50
    rep = report(d)
    assert "FD_BANDWIDTH_RELAXED" in rep.codes("warning") and not rep.errors
    d["nodes"][0]["streams"][0]["occupied_bw_hz"] = 1.2e6  # above f_s: always an error
    assert_error(report(d), "FD_BANDWIDTH", FDBandwidthExceeded)


def test_table_bandwidth_and_domain():
    # lab table validity 20 MHz: 23.04 MS/s with B = 0.8 f_s = 18.432 MHz passes ...
    d = lab()
    for n in d["nodes"]:
        n["streams"][0]["rate_sps"] = "23.04e6"
    d["nodes"][0]["streams"][0]["occupied_bw_hz"] = 0.8 * 23.04e6
    assert not report(d).errors
    # ... 30.72 MS/s with B = 24.576 MHz exceeds it
    for n in d["nodes"]:
        n["streams"][0]["rate_sps"] = "30.72e6"
    d["nodes"][0]["streams"][0]["occupied_bw_hz"] = 0.8 * 30.72e6
    assert_error(report(d), "TABLE_BANDWIDTH", BandwidthExceeded)
    d = lab()
    for n in d["nodes"]:
        n["streams"][0]["fc_hz"] = 2.4e9
    assert_error(report(d), "TABLE_DOMAIN", OutOfDomain)
    d = lab()
    for n in d["nodes"]:
        n["streams"][0]["fc_hz"] = 3.5e9 + 9.9e6  # inside f0 +/- validity/2 = +/- 10 MHz
    assert "TABLE_DOMAIN" not in report(d).codes()
    d["ris"][0]["states"] = {"library": "tang2022_ris1_27ghz"}
    assert_error(report(d), "TABLE_DOMAIN", OutOfDomain)
    d["ris"][0]["states"] = {"library": "ideal_bbit(b=2)"}  # frequency independent
    assert not report(d).errors


def _write_lib(tmp_path: Path, **over: Any) -> str:
    base = read_document(REPO / "data" / "states" / "ideal_antipodal.yaml")
    base.update(over)
    (tmp_path / "custom.yaml").write_text(dump_yaml(base), encoding="utf-8")
    return "custom.yaml"


def test_passivity_efficiency_initial_state(tmp_path):
    d = quick()
    states = [
        {"id": 0, "label": "a", "gamma": [{"f_hz": None, "mag": 1.0 + 2e-9, "phase_deg": 0}]},
        {"id": 1, "label": "b", "gamma": [{"f_hz": None, "mag": 1.0, "phase_deg": 180}]},
    ]
    d["ris"][0]["states"] = {"file": _write_lib(tmp_path, states=states)}
    assert_error(report(d, tmp_path), "TABLE_PASSIVITY", PassivityError)
    states[0]["gamma"][0]["mag"] = 1.0 + 0.5e-9  # within the 1e-9 analytic tolerance
    d["ris"][0]["states"] = {"file": _write_lib(tmp_path, states=states)}
    assert not report(d, tmp_path).errors
    # measured tables: |Gamma| <= 1 + 3 sigma_mag, sigma_mag = (ln10/20) sigma_mag_db (5.5)
    prov = read_document(REPO / "data" / "states" / "ideal_antipodal.yaml")["provenance"]
    prov = {**prov, "source_kind": "measured", "sigma_mag_db": 0.5}
    bound = 1.0 + 3.0 * (math.log(10.0) / 20.0) * 0.5  # 1.17269...
    assert bound == pytest.approx(1.1726938819745534, rel=1e-15)
    states[0]["gamma"][0]["mag"] = bound - 1e-6
    d["ris"][0]["states"] = {"file": _write_lib(tmp_path, states=states, provenance=prov)}
    assert not report(d, tmp_path).errors
    states[0]["gamma"][0]["mag"] = bound + 1e-6
    d["ris"][0]["states"] = {"file": _write_lib(tmp_path, states=states, provenance=prov)}
    assert_error(report(d, tmp_path), "TABLE_PASSIVITY", PassivityError)
    # a value between the linear bound and 10**(3 sigma_db/20) = 1.1885 must fail too (the
    # bound the physics ResponseTable enforces at load; review fix)
    states[0]["gamma"][0]["mag"] = 1.18
    d["ris"][0]["states"] = {"file": _write_lib(tmp_path, states=states, provenance=prov)}
    assert_error(report(d, tmp_path), "TABLE_PASSIVITY", PassivityError)
    # efficiency refused on measured tables unless overridden; fine on analytic ones
    d = quick()
    for n in d["nodes"]:
        n["streams"][0]["fc_hz"] = 2.7e10
    d["ris"][0]["states"] = {"library": "tang2022_ris1_27ghz", "efficiency": 0.5}
    d["ris"][0]["grid"].update(pitch_u_m=0.005, pitch_v_m=0.005)
    assert_error(report(d), "EFFICIENCY_REFUSED", TableValidationError)
    d["ris"][0]["states"]["override_efficiency"] = True
    assert not report(d).errors
    d = quick()
    d["ris"][0]["states"]["efficiency"] = 0.5
    assert not report(d).errors
    d["ris"][0]["initial"] = {"fill": 2}  # binary alphabet
    assert_error(report(d), "INITIAL_STATE", SchemaError)


@pytest.mark.parametrize(
    ("source_kind", "sigma_db"),
    [("analytic", 0.0), ("assumed", 2.0), ("measured", 0.0), ("measured", 0.5), ("digitized", 1.0)],
)
@pytest.mark.parametrize("side", [-1, +1])
def test_passivity_bound_agrees_with_the_physics_response_table(
    tmp_path, source_kind, sigma_db, side
):
    """The guard and physics.response.ResponseTable apply the same section 5.5 bound, so
    ``sremu validate`` never passes a table the engine then refuses (or the reverse)."""
    from sremu.physics.response import ResponseTable

    uncertain = source_kind in ("measured", "digitized")
    margin = 3.0 * (math.log(10.0) / 20.0) * sigma_db if uncertain else 1e-9
    mag = 1.0 + margin + side * max(1e-6 * margin, 2e-10)
    base = read_document(REPO / "data" / "states" / "ideal_antipodal.yaml")
    prov = {**base["provenance"], "source_kind": source_kind, "sigma_mag_db": sigma_db}
    states = [
        {"id": 0, "label": "a", "gamma": [{"f_hz": None, "mag": mag, "phase_deg": 0}]},
        {"id": 1, "label": "b", "gamma": [{"f_hz": None, "mag": 1.0, "phase_deg": 180}]},
    ]
    d = quick()
    d["ris"][0]["states"] = {"file": _write_lib(tmp_path, states=states, provenance=prov)}
    guard_fails = "TABLE_PASSIVITY" in report(d, tmp_path).codes("error")
    assert guard_fails is (side > 0)
    try:
        ResponseTable.from_state_library(tmp_path / "custom.yaml")
        physics_fails = False
    except PassivityError:
        physics_fails = True
    assert physics_fails is guard_fails


def test_disabled_panel_tables_are_still_checked():
    """``enabled: false`` is only the initial state (a ``ris.set_enabled`` event can switch
    the panel on), so its table must fit the linked bands as well."""
    d = lab()
    for n in d["nodes"]:
        n["streams"][0]["rate_sps"] = "30.72e6"
    d["nodes"][0]["streams"][0]["occupied_bw_hz"] = 0.8 * 30.72e6  # > validity 20 MHz
    d["ris"][0]["enabled"] = False
    assert_error(report(d), "TABLE_BANDWIDTH", BandwidthExceeded)
    d = lab()
    for n in d["nodes"]:
        n["streams"][0]["fc_hz"] = 2.4e9
    d["ris"][0]["enabled"] = False
    assert_error(report(d), "TABLE_DOMAIN", OutOfDomain)


def test_narrowband_estimate_counts_disabled_panels():
    """The t = 0 delay spread includes the paths of a disabled panel (it may be switched on
    during the run); oracle: LOS and every cell path, recomputed here."""
    d = lab()
    d["ris"][0]["enabled"] = False
    res = resolve_scenario(parse_scenario(d), SCEN, warn=False)
    link = check_scenario(res).derived["links"]["ap.tx0->rx2.rx0"]
    sc = res.scenario
    a, b = np.array(sc.nodes[0].position_m), np.array(sc.nodes[2].position_m)
    c = cell_centres(sc.ris[0])
    paths = np.concatenate(
        [[np.linalg.norm(a - b)], np.linalg.norm(a - c, axis=1) + np.linalg.norm(b - c, axis=1)]
    )
    assert link["delay_spread_s"] == pytest.approx((paths.max() - paths.min()) / C0, rel=1e-12)
    assert link["delay_spread_s"] > 0.0


def test_many_states_need_a_separable_table(tmp_path):
    grid = [-30.0, 30.0]
    states = [
        {
            "id": s,
            "label": f"s{s}",
            "gamma": [
                {
                    "f_hz": None,
                    "mag": 0.5,
                    "phase_deg": 10.0 * s,
                    "theta_i_deg": a,
                    "theta_s_deg": b,
                }
                for a in grid
                for b in grid
            ],
        }
        for s in range(17)
    ]
    d = quick()
    d["ris"][0]["states"] = {
        "file": _write_lib(tmp_path, states=states, angle_grid={"theta_deg": grid})
    }
    assert_error(report(d, tmp_path), "TABLE_STATES", TableValidationError)


def test_coupling_matrix():
    d = lab()
    d["run"]["mode"] = "paced"
    d["controllers"][0]["kind"] = "external"
    del d["controllers"][0]["impl"]
    assert_error(report(d), "COUPLING", ConfigurationError)
    d["run"]["coupling"] = "async"
    assert "COUPLING" not in report(d).codes()
    d = lab()
    d["run"]["mode"] = "paced"
    assert not report(d).errors  # paced + barrier with an in-process controller


def test_t_u_may_only_lower_the_automatic_value():
    d = quick()
    d["numerics"] = {"t_u_s": "4e-3"}
    assert_error(report(d, auto_t_u_s=Fraction(3, 1000)), "T_U_ABOVE_AUTO", ConfigurationError)
    assert not report(d, auto_t_u_s=Fraction(4, 1000)).errors
    assert not report(d).errors  # without the WP5 value only t_u_max_s is checked (schema)


# --------------------------------------------------------------------------- warnings


def test_warnings():
    d = quick()
    del d["nodes"][0]["streams"][0]["occupied_bw_hz"]
    rep = report(d)
    assert "OCCUPIED_BW_UNDECLARED" in rep.codes("warning") and not rep.errors
    d = quick()
    d["ris"][0]["grid"].update(pitch_u_m=0.05, cell_model="point")  # > lambda/2 = 0.0428 m
    assert "POINT_PITCH" in report(d).codes("warning")
    d["ris"][0]["grid"]["cell_model"] = "auto"
    assert "POINT_PITCH" not in report(d).codes()
    d = quick()
    d["nodes"][1]["position_m"] = [600.0, 300.0, 0.0]  # RIS hop 5 + 670.8 m; LOS 668 m
    assert "LONG_PATH" not in report(d).codes()
    d["nodes"][1]["position_m"] = [900.0, 500.0, 0.0]  # LOS 1027 m, RIS hop 1034.6 m
    assert "LONG_PATH" in report(d).codes("warning")
    d = quick()
    d["record"] = {"iq": True}
    d["run"]["duration_s"] = "625"  # 2 streams x 1e6 S/s x 625 s x 8 B = 10.0 GB exactly
    assert "RECORDER_DISK" not in report(d).codes()
    d["run"]["duration_s"] = "626"
    assert "RECORDER_DISK" in report(d).codes("warning")
    d = quick()
    d["nodes"][1]["streams"][0]["fc_hz"] = 2.4e9
    assert "NO_LINKS" in report(d).codes("warning")
    rep = report(quick())
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        GuardReport(issues=report(d).issues).emit_warnings()
    assert any(issubclass(x.category, ValidityWarning) for x in w)
    assert rep.issues == []


# --------------------------------------------------------------------------- derived


def test_derived_quantities_lab():
    res = load_scenario(SCEN / "lab_like_A.yaml")
    der = check_scenario(res).derived
    assert der["streams"]["rx1.rx0"]["page_size"] == 512  # 1 MS/s (4.3)
    assert der["streams"]["ap.tx0"]["lambda_m"] == pytest.approx(C0 / 3.5e9, rel=1e-15)
    nf = der["receivers"]["rx1.rx0"]["noise_floor_dbm"]
    assert nf == pytest.approx(-113.975 + 7.0, abs=5e-4)  # -113.975 dBm at 1 MS/s, F = 0 dB
    tau = der["receivers"]["rx2.rx0"]["tau_ris_samples"]["ris0"]
    assert tau == pytest.approx(1.2 / C0 * 1e6, rel=1e-9)
    d_side = 24 * 0.0428 + 2 * 0.002
    assert der["panels"]["ris0"]["fraunhofer_m"]["3.5e+09"] == pytest.approx(
        2 * d_side**2 / LAM, rel=1e-12
    )
    link = der["links"]["ap.tx0->rx1.rx0"]
    assert link["fd_max_hz_bound"] == 0.0  # static scene
    assert link["fd"]["taps"] == 32  # B/f_s = 0.8 (4.5 table)
    assert link["narrowband"] is True  # pi B dtau << 0.05 (4.5 worked decision: narrowband)
    # delay spread oracle: LOS and every visible cell path at t = 0
    sc = res.scenario
    a = np.array(sc.nodes[0].position_m)
    b = np.array(sc.nodes[1].position_m)
    c = cell_centres(sc.ris[0])
    paths = np.concatenate(
        [[np.linalg.norm(a - b)], np.linalg.norm(a - c, axis=1) + np.linalg.norm(b - c, axis=1)]
    )
    assert link["delay_spread_s"] == pytest.approx((paths.max() - paths.min()) / C0, rel=1e-12)
    assert der["recorder_bytes"] == 3 * 1e6 * 5 * 8


def test_derived_doppler_bound_and_t_curv():
    d = quick()
    d["nodes"][1].pop("position_m")
    d["nodes"][1]["trajectory"] = {
        "model": "constant_velocity",
        "start_m": [4.7, 1.7, 0.0],
        "velocity_mps": [0.0, 1.5, 0.0],
    }
    res = resolve_scenario(parse_scenario(d), SCEN, warn=False)
    row = check_scenario(res).derived["links"]["ap.tx0->mu.rx0"]
    assert row["fd_max_hz_bound"] == pytest.approx(1.5 * 3.5e9 / C0, rel=1e-12)  # 17.51 Hz
    assert row["fd_max_hz_bound"] == pytest.approx(17.51, abs=0.005)
    # r_min oracle, independent of NodeGeometry: brute force over a dense sampling of the
    # user's straight path over [0, 0.1 s] (AP-user, AP-cells and user-cells distances).
    sc = res.scenario
    ap = np.array(sc.nodes[0].position_m)
    t = np.linspace(0.0, 0.1, 200_001)[:, None]
    mu = np.array([4.7, 1.7, 0.0]) + t * np.array([0.0, 1.5, 0.0])
    cells = cell_centres(sc.ris[0])
    d_ap_mu = np.linalg.norm(mu - ap, axis=1).min()
    d_ap_cells = np.linalg.norm(cells - ap, axis=1).min()
    d_mu_cells = min(np.linalg.norm(cells - m, axis=1).min() for m in mu[::1000])
    r_min = min(d_ap_mu, d_ap_cells, d_mu_cells)
    assert row["r_min_m"] == pytest.approx(r_min, abs=1e-9)
    k = 2 * math.pi * 3.5e9 / C0
    expect = math.sqrt(8 * 1e-3 * r_min / (k * 1.5**2))
    assert row["t_curv_s_bound"] == pytest.approx(expect, rel=1e-8)
    assert row["t_u0_s_estimate"] == pytest.approx(min(expect, 0.01), rel=1e-8)


def test_fd_choice_table():
    assert fd_choice(0.2e6, Fraction(10**6))["taps"] == 8
    assert fd_choice(0.2000001e6, Fraction(10**6))["taps"] == 16
    assert fd_choice(0.6e6, Fraction(10**6))["taps"] == 16
    assert fd_choice(0.8e6, Fraction(10**6))["taps"] == 32
    assert fd_choice(0.81e6, Fraction(10**6))["taps"] is None
    lim = 0.1 / math.pi  # pi B / (2 f_s) <= 0.05  <=>  B <= 0.0318 f_s
    assert fd_choice(lim * 1e6 * 0.999, Fraction(10**6))["integer_delay_allowed"] is True
    assert fd_choice(lim * 1e6 * 1.001, Fraction(10**6))["integer_delay_allowed"] is False
