"""E-B on the lab arc: scenario file, geometry, blocked variant, multi-panel design and the
sweep specs of ``python -m experiments.eb`` (DESIGN.md section 11, E-B; D24 naming)."""

from __future__ import annotations

import json
import math
from fractions import Fraction

import numpy as np
import pytest
from experiments import eb
from experiments.common import PAPER
from validation.ants_fig4b import prior

from sremu.contracts import PathKind
from sremu.control.controller import csi_from_snapshot
from sremu.physics.optimize import exact_alphabet_optimum_reference
from sremu.physics.propagation import from_scenario
from sremu.scenario.guards import check_scenario
from sremu.scenario.load import parse_scenario, read_document, resolve_scenario

ARC_REF = "scenarios/studies/lab_arc.yaml"
R_M, SPAN_DEG, CENTRE_DEG, H_M, PITCH_M = 1.95, 46.0, 22.5, 1.0, 0.0428


def _nominal_prior_draw() -> prior.PriorDraw:
    return prior.PriorDraw(
        name="nominal",
        geometry_model=prior.ARC,
        origin="O_T",
        theta0="arc",
        radius_m=R_M,
        span_deg=SPAN_DEG,
        centre_deg=CENTRE_DEG,
        d_m=math.nan,
        a_deg=math.nan,
        gap_m=math.nan,
        pitch="square_42p8mm",
        cell_model="auto",
        gamma_mag=(1.0, 1.0),
        gamma_phase_offset_deg=(0.0, 0.0),
        antenna="isotropic",
        point_dxy_m=np.zeros((30, 2)),
        plane_dz_m=0.0,
        height_m=H_M,
        polarization="V",
    )


def test_lab_arc_scenario_validates() -> None:
    doc = read_document(eb.ARC)
    sc = parse_scenario(doc, label="lab_arc")
    res = resolve_scenario(sc, eb.ARC.parent, warn=False)
    rep = check_scenario(res)
    rep.raise_for_errors()
    assert sc.name == "lab_arc"
    assert [p.id for p in sc.ris] == prior.panel_ids(6)


def test_lab_arc_geometry_is_the_nominal_arc() -> None:
    """Tile poses = the arc branch of validation/ants_fig4b/prior.py at the nominal values,
    and the task's closed form: phi_k = 22.5 + (k + 0.5 - 3) 46/6 deg, R = 1.95 m, normals at
    the Tx, 5-cell side horizontal and tangential, 8-cell side vertical."""
    doc = read_document(eb.ARC)
    tx, c, n, u = prior._tile_poses(
        _nominal_prior_draw(), {"prior": {"tiles": {"n_tiles": 6}}}, H_M, 5 * PITCH_M
    )
    proto = json.loads((prior.ANTENNA_DIR.parent / "protocol.json").read_text(encoding="utf-8"))
    by_pos = proto["prior"]["tiles"]["states_by_position"]
    assert by_pos == [0, 0, 1, 1, 0, 0]  # s1 s1 s2 s2 s1 s1
    nodes = {nd["id"]: np.asarray(nd["position_m"], dtype=np.float64) for nd in doc["nodes"]}
    np.testing.assert_array_equal(nodes["ap"], tx)
    for rid, th in (("rx1", 15.0), ("rx2", 35.0)):
        want = [1.2 * math.cos(math.radians(th)), 1.2 * math.sin(math.radians(th)), H_M]
        np.testing.assert_allclose(nodes[rid], want, rtol=0, atol=1e-9)
    assert len(doc["ris"]) == 6
    for k, p in enumerate(doc["ris"]):
        cen = np.asarray(p["pose"]["center_m"], dtype=np.float64)
        nrm = np.asarray(p["pose"]["normal"], dtype=np.float64)
        uax = np.asarray(p["pose"]["u_axis"], dtype=np.float64)
        np.testing.assert_allclose(cen, c[k], rtol=0, atol=1e-9)
        np.testing.assert_allclose(nrm, n[k], rtol=0, atol=1e-12)
        np.testing.assert_allclose(uax, u[k], rtol=0, atol=1e-12)
        phi = math.radians(CENTRE_DEG + (k + 0.5 - 3) * SPAN_DEG / 6)
        np.testing.assert_allclose(
            cen, [R_M * math.cos(phi), R_M * math.sin(phi), H_M], rtol=0, atol=1e-9
        )
        to_tx = (tx - cen) / np.linalg.norm(tx - cen)
        np.testing.assert_allclose(nrm, to_tx, rtol=0, atol=1e-9)  # faces the Tx
        np.testing.assert_allclose(np.cross(nrm, uax), [0.0, 0.0, 1.0], atol=1e-12)  # v = +z
        assert abs(uax[2]) == 0.0
        assert p["grid"]["n_u"] == 5 and p["grid"]["n_v"] == 8  # 5 along u (horizontal)
        assert p["grid"]["pitch_u_m"] == PITCH_M and p["grid"]["pitch_v_m"] == PITCH_M
        assert p["tiles"] == {"n_tiles_u": 1, "n_tiles_v": 1, "gap_m": 0.0}
        assert p["initial"] == {"fill": by_pos[k]}
        assert p["states"] == {
            "library": "lab_binary_3p5ghz",
            "assume_mag": {"sigma1": 1.0, "sigma2": 1.0},
        }
        assert p["control"] == {"granularity": "element"}
    for nd in doc["nodes"]:
        assert nd["antenna"] == {"pattern": "isotropic"}
        for s in nd["streams"]:
            assert s["fc_hz"] == 3.5e9


def test_lab_arc_alphabet_phases() -> None:
    """sigma1 = 49.64 deg, sigma2 = -153.77 deg, |Gamma| = 1 (assumed) on every tile."""
    model = from_scenario(read_document(eb.ARC), backend="numpy_ref")
    snap = model.snapshot(Fraction(0))
    assert sorted(snap.ris) == prior.panel_ids(6)
    for rc in snap.ris.values():
        assert rc.gamma is not None
        np.testing.assert_allclose(np.abs(rc.gamma), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.angle(rc.gamma, deg=True), [49.64, -153.77], atol=1e-9)


def test_scenario_refs_and_labels() -> None:
    assert eb.scenario_ref(None) is None
    assert eb.scenario_ref("scenarios/lab_like_A.yaml") is None
    assert eb.scenario_ref(eb.ARC) == ARC_REF
    assert eb.scenario_ref(ARC_REF) == ARC_REF
    assert eb.geometry_label(None) == "lab-like SIM geometry (not the lab's)"
    assert eb.geometry_label(ARC_REF) == "modelled on the lab arrangement (SIM, nominal geometry)"


def test_default_scenario_mapping_unchanged() -> None:
    """The default geometry's E-B scenario is built exactly as before (reproducibility)."""
    d = eb.lab_scenario("lab", "blocked")
    assert d["name"] == "eb_lab_blocked"
    assert (
        d["description"] == "E-B lab-like SIM geometry (not the lab's), alphabet lab, LOS blocked"
    )
    assert d["environment"] == {
        "occluders": [{"id": "los_blocker", "kind": "opaque", "vertices_m": eb.BLOCKER}]
    }
    assert eb.BLOCKER == [
        [1.5, -0.6, -0.5],
        [1.5, -0.35, -0.5],
        [1.5, -0.35, 0.5],
        [1.5, -0.6, 0.5],
    ]
    assert [p["id"] for p in d["ris"]] == [eb.PANEL]
    d2 = eb.lab_scenario("ideal1", "present", initial_states=[1] * 120)
    assert d2["ris"][0]["initial"] == {"states": [1] * 120}
    assert "environment" not in d2


def test_split_states() -> None:
    s = np.arange(10) % 2
    parts = eb.split_states(s, ["a", "b"], [4, 6])
    assert list(parts) == ["a", "b"]
    np.testing.assert_array_equal(np.concatenate([parts["a"], parts["b"]]), s)
    assert parts["a"].dtype == np.int16
    with pytest.raises(ValueError):
        eb.split_states(s, ["a", "b"], [4, 5])


def test_arc_blocked_variant_removes_only_los() -> None:
    """The plate crosses both LOS segments and no cell -> RX segment; in the model it zeroes
    both LOS paths and leaves every RIS coefficient bitwise unchanged (occluders act on the
    LOS only, section 5.2)."""
    dp = eb.lab_scenario("lab", "present", scenario=ARC_REF)
    db = eb.lab_scenario("lab", "blocked", scenario=ARC_REF)
    (occ,) = db["environment"]["occluders"]
    assert occ["id"] == "los_blocker" and occ["kind"] == "opaque"
    assert occ["vertices_m"] == eb.los_blocker(dp)
    cr = eb.blocker_crossings(db, occ["vertices_m"])
    assert cr["los_crossed"] == {eb.RX1: True, eb.RX2: True}
    assert cr["cell_rx_segments_crossed"] == {eb.RX1: 0, eb.RX2: 0}
    assert cr["n_cells"] == 240
    assert 0 < cr["tx_cell_segments_crossed"] < 240  # documented idealization
    eb.resolve(db, eb.ARC.parent)
    sp = from_scenario(dp, backend="numpy_ref").snapshot(Fraction(0))
    sb = from_scenario(db, backend="numpy_ref").snapshot(Fraction(0))
    for lk in eb.LINKS.values():
        kinds_p, kinds_b = sp.paths[lk].kind, sb.paths[lk].kind
        np.testing.assert_array_equal(kinds_p, kinds_b)
        los = np.nonzero(kinds_b == int(PathKind.LOS))[0]
        assert los.size == 1
        assert sp.paths[lk].amp[los[0]] != 0 and sb.paths[lk].amp[los[0]] == 0
    for pid, rc in sp.ris.items():
        rb = sb.ris[pid]
        for a, b in ((rc.C, rb.C), (rc.B, rb.B)):
            if a is None:
                assert b is None
            else:
                np.testing.assert_array_equal(a, b)


def _seg_hits_convex(a: np.ndarray, b: np.ndarray, poly: np.ndarray) -> bool:
    """Independent segment / planar convex polygon test (plane crossing + same-side edges)."""
    n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
    da, db_ = float(np.dot(a - poly[0], n)), float(np.dot(b - poly[0], n))
    if da * db_ >= 0:
        return False
    x = a + da / (da - db_) * (b - a)
    s = [float(np.dot(np.cross(poly[(i + 1) % 4] - poly[i], x - poly[i]), n)) for i in range(4)]
    return all(v > 0 for v in s) or all(v < 0 for v in s)


def test_arc_blocker_crossings_independent() -> None:
    """blocker_crossings against cell centres from the closed-form arc (D17 layout) and an
    independent segment/polygon test; the plate is vertical, across both LOS segments at
    half the along-bisector LOS length, and a 1 cm strip in its place would miss every
    Tx -> cell ray (so the crossings come from the plate's physical size, not the arc)."""
    db = eb.lab_scenario("lab", "blocked", scenario=ARC_REF)
    poly = np.asarray(db["environment"]["occluders"][0]["vertices_m"], dtype=np.float64)
    assert np.allclose(poly[:, 2], [0.5, 0.5, 1.5, 1.5])  # 1 m tall, centred on h
    tx = np.array([0.0, 0.0, H_M])
    rxs = {
        eb.RX1: np.array([1.2 * math.cos(math.radians(15)), 1.2 * math.sin(math.radians(15)), H_M]),
        eb.RX2: np.array([1.2 * math.cos(math.radians(35)), 1.2 * math.sin(math.radians(35)), H_M]),
    }
    cells = []
    for k in range(6):
        phi = math.radians(CENTRE_DEG + (k + 0.5 - 3) * SPAN_DEG / 6)
        c = np.array([R_M * math.cos(phi), R_M * math.sin(phi), H_M])
        u = np.array([math.sin(phi), -math.cos(phi), 0.0])
        v = np.array([0.0, 0.0, 1.0])
        for iv in range(8):
            for iu in range(5):
                cells.append(c + (iu - 2) * PITCH_M * u + (iv - 3.5) * PITCH_M * v)
    want_tx = sum(_seg_hits_convex(tx, q, poly) for q in cells)
    want_rx = {r: sum(_seg_hits_convex(q, p, poly) for q in cells) for r, p in rxs.items()}
    cr = eb.blocker_crossings(db, poly.tolist())
    assert cr["n_cells"] == len(cells) == 240
    assert cr["tx_cell_segments_crossed"] == want_tx
    assert cr["cell_rx_segments_crossed"] == want_rx == {eb.RX1: 0, eb.RX2: 0}
    assert cr["los_crossed"] == {r: _seg_hits_convex(tx, p, poly) for r, p in rxs.items()}
    assert all(cr["los_crossed"].values())
    strip = poly.copy()
    strip[:, 2] = [H_M - 0.005, H_M - 0.005, H_M + 0.005, H_M + 0.005]
    assert all(_seg_hits_convex(tx, p, strip) for p in rxs.values())
    assert sum(_seg_hits_convex(tx, q, strip) for q in cells) == 0
    assert eb.blocked_assumption({"blocker": cr}) is not None
    assert f"{want_tx}/240 Tx->cell" in str(eb.blocked_assumption({"blocker": cr}))
    assert eb.blocked_assumption(eb.design("lab", "blocked").diag) is None  # default geometry


@pytest.mark.parametrize("alphabet", ["lab", "ideal1"])
def test_arc_design_joint_optimum_and_calibration(alphabet: str) -> None:
    d25 = eb.design(alphabet, "present", scenario=ARC_REF)
    d15 = eb.design(alphabet, "present", 15.0, scenario=ARC_REF)
    assert d25.panels == tuple(prior.panel_ids(6)) and d25.sizes == (40,) * 6
    assert d25.s1.shape == (240,) and d25.s2.shape == (240,)
    assert d25.scenario == ARC_REF
    assert abs(d25.es_n0_db[eb.RX1] - 25.0) < 1e-9
    assert abs(d15.es_n0_db[eb.RX1] - 15.0) < 1e-9
    assert abs((d25.tx_dbm - d15.tx_dbm) - 10.0) < 1e-9
    np.testing.assert_array_equal(d25.s1, d15.s1)
    # the joint optimum over all 240 cells equals the reference exact algorithm
    doc = eb.lab_scenario(alphabet, "present", scenario=ARC_REF)
    model = from_scenario(doc, backend="numpy_ref")
    snap = model.snapshot(Fraction(0))
    for rx, s, key in ((eb.RX1, d25.s1, "s1"), (eb.RX2, d25.s2, "s2")):
        v = csi_from_snapshot(model, snap, eb.LINKS[rx])
        z = np.concatenate([v.cascade[p] for p in d25.panels], axis=1)
        val_ref, _ = exact_alphabet_optimum_reference(v.h_direct, z)
        assert abs(abs(d25.h[rx][key]) - val_ref) <= 1e-12 * val_ref
        assert abs(v.evaluate(d25.split(s)) - d25.h[rx][key]) == 0.0
    # better at RX1 than the lab arrangement s1 s1 s2 s2 s1 s1
    v1 = csi_from_snapshot(model, snap, eb.LINKS[eb.RX1])
    arr = {
        p: np.full(40, st, dtype=np.int16)
        for p, st in zip(d25.panels, [0, 0, 1, 1, 0, 0], strict=True)
    }
    assert abs(d25.h[eb.RX1]["s1"]) >= abs(v1.evaluate(arr))
    assert len(d25.changed_panels()) >= 1


def test_arc_sweep_specs() -> None:
    specs = eb.sweep_specs(ARC_REF)
    by = {s.name: s for s in specs}
    assert list(by) == ["eb1", "eb2_lab", "eb2_ideal1", "eb2_lab_es15", "eb2_ideal1_es15"]
    counts = {k: len(s.points()) for k, s in by.items()}
    assert counts == {
        "eb1": 72,
        "eb2_lab": 150,
        "eb2_ideal1": 75,
        "eb2_lab_es15": 75,
        "eb2_ideal1_es15": 75,
    }
    for s in specs:
        assert s.seeds == (20261010,)
        for pt in s.points():
            assert pt.params["scenario"] == ARC_REF
            assert "legacy" not in json.dumps(pt.params)  # D24 naming
            if s.name.startswith("eb2"):
                assert pt.params["n_packets"] == 2000
                assert pt.params["semantics"] in eb.SEMANTICS
    for name in ("eb2_lab_es15", "eb2_ideal1_es15"):
        pts = by[name].points()
        assert {p.params["es_n0_db"] for p in pts} == {15.0}
        assert {p.params["los"] for p in pts} == {"present"}
        # same E-B2 design as the 25 dB LOS-present slice
        alph = name.split("_")[1]
        ref = [p for p in by[f"eb2_{alph}"].points() if p.params.get("los", "present") == "present"]
        strip = ("es_n0_db",)
        assert [{k: v for k, v in p.params.items() if k not in strip} for p in pts] == [
            {**p.params, "los": "present"} for p in ref
        ]
    for name in ("eb1", "eb2_lab", "eb2_ideal1"):
        assert all("es_n0_db" not in p.params for p in by[name].points())
    smoke = eb.sweep_specs(ARC_REF, smoke=True)
    assert [s.name for s in smoke] == ["eb1_smoke", "eb2_smoke"]
    assert [len(s.points()) for s in smoke] == [1, 1]
    assert smoke[1].points()[0].params["n_packets"] == 50


@pytest.mark.skipif(not (PAPER / "sweeps" / "eb1.yaml").is_file(), reason="paper repo absent")
def test_default_sweep_specs_match_paper_specs() -> None:
    """Without --scenario the driver runs the original design (paper/sweeps/eb*.yaml), with
    the paper-facing name 'tick-based' for the former 'legacy' semantics."""
    from sremu.experiments.sweep import load_spec

    by = {s.name: s for s in eb.sweep_specs(None)}
    for name in ("eb1", "eb2_lab", "eb2_ideal1"):
        old = load_spec(PAPER / "sweeps" / f"{name}.yaml").points()
        new = by[name].points()
        mapped = [
            {
                k: ("tick-based" if (k == "semantics" and v == "legacy") else v)
                for k, v in p.params.items()
            }
            for p in old
        ]
        assert [p.params for p in new] == mapped
        assert [p.seed for p in new] == [p.seed for p in old]


def test_semantics_names() -> None:
    assert eb.canonical_semantics("legacy") == "tick-based"
    assert eb.canonical_semantics("tick-based") == "tick-based"
    with pytest.raises(ValueError):
        eb.canonical_semantics("next_tick")
    d = eb.analytic_per("tick-based", Fraction(1, 2000), Fraction(1, 1000), Fraction(1, 2000))
    assert d["p_switch_in_packet"] is None
    assert eb.packet_start("tick-based", 500, 500) == 500


def test_cli_rejects_missing_scenario(tmp_path) -> None:
    with pytest.raises(SystemExit) as ei:
        eb.main(["--scenario", str(tmp_path / "nope.yaml"), "--out", str(tmp_path), "--smoke"])
    assert ei.value.code == 2
