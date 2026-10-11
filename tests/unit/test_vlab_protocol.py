"""V-LAB protocol files, prior and prediction code (DESIGN.md section 10.9; WP10; protocol v2).

No measured data are used: only protocol.json, sremu physics and keyed synthetic draws.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from validation.ants_fig4b import predict as pr
from validation.ants_fig4b import prior as pp
from validation.ants_fig4b.protocol import (
    PROTOCOL_DIR,
    cell_layout,
    load_protocol,
    prediction_spec_sha256,
    prior_variants,
    ring_split,
    sensitivity_run,
    split_masks,
    variant_protocol,
)

from sremu.errors import ValidityWarning
from sremu.physics.antennas import Tabulated
from sremu.physics.propagation import from_scenario
from sremu.physics.response import load_table
from sremu.physics.walls import p2040_material
from sremu.units import C0

pytestmark = pytest.mark.unit

LAM = C0 / 3.5e9


@pytest.fixture(scope="module")
def proto() -> dict:
    return load_protocol()


@pytest.fixture(scope="module")
def sor(proto: dict) -> dict:
    return variant_protocol(proto, "S-OR")


# --------------------------------------------------------------------------- protocol files


def test_design_numbers_in_protocol_json(proto: dict) -> None:
    """The numbers DESIGN 10.9 fixes, as machine-readable values (unchanged in v2)."""
    d = proto["data"]
    assert proto["protocol_version"] == "2.0"
    assert d["theta_deg"] == [0, 5, 15, 25, 35, 45]
    assert d["r_m"] == [0.6, 0.9, 1.2, 1.5, 1.8]
    assert d["n_cells"] == 30
    assert d["sigma_dig_db"]["disclosed_range"] == [0.045, 0.22]
    assert d["colorbar_pinned_dbm"] == [-10.06, -33.2]
    assert proto["split"]["primary"] == {
        "train_rings_m": [0.6, 1.2, 1.8],
        "test_rings_m": [0.9, 1.5],
    }
    assert proto["primary_analysis"]["K"] == 2000
    assert proto["secondary_analysis"]["sigma_tot_db"] == 2.0
    assert proto["uncertainty_budget"]["sigma_rep_db"] == 1.0
    assert proto["uncertainty_budget"]["sigma_rep_sensitivity_db"] == [0.5, 2.0]
    assert proto["bootstrap"]["n_resamples_primary"] == 10_000
    assert proto["bootstrap"]["n_resamples_decoy"] == 2000
    assert proto["power_analysis"]["n_decoys"] == 500
    assert proto["power_analysis"]["n_bootstrap"] == 2000
    assert proto["power_analysis"]["underpowered_if_c2_rate_below"] == 0.5
    c1, c2 = proto["decision_rules"]["C1"], proto["decision_rules"]["C2"]
    assert (c1["median_abs_bias_max_db"], c1["rmse_margin_vs_m0n_db"]) == (1.5, 0.5)
    assert c2["median_delta_rmse_min_db"] == 1.0
    assert c2["ci_excludes_zero_min_fraction"] == 0.8
    assert c2["spearman_rho_min"] == 0.3
    pri = proto["prior"]
    assert pri["gamma"]["mag"] == [0.6, 1.0] and pri["gamma"]["phase_offset_deg"] == [-15.0, 15.0]
    assert pri["cell_model"] == {"values": ["point", "po_patch"], "weights": [0.5, 0.5]}
    assert pri["floor"]["antenna_height_m"] == [0.8, 1.5]
    assert proto["naming"]["label"] == "consistency check under geometric uncertainty"


def test_v2_prior_encodes_the_author_facts_exactly(proto: dict) -> None:
    """Author facts 1-7 (2026-10-11) and the lead's near-field decision, number by number."""
    pri = proto["prior"]
    g = pri["geometry"]
    assert g["model"] == "arc_facing_tx" and g["polar_origin"] == "O_T"  # fact 1
    assert g["radius_author_m"] == [1.9, 2.0]  # fact 3 as stated
    assert g["radius_m"] == [1.88, 2.05]  # the allowed robustness margin (PROTOCOL 4.1)
    assert g["span_deg"] == [40.0, 52.0] and g["centre_nominal_deg"] == 22.5  # fact 5
    assert g["centre_offset_deg"] == [-3.0, 3.0]
    assert g["draw_order"] == ["radius_m", "span_deg", "centre_offset_deg"]
    tl = pri["tiles"]
    assert tl["n_tiles"] == 6 and tl["states_by_position"] == [0, 0, 1, 1, 0, 0]  # fact 2
    assert (tl["n_u"], tl["n_v"]) == (5, 8)  # fact 4: 5 cells horizontal (u), 8 vertical
    assert pri["pitch"]["pitch_u_m"] == [0.0428, 0.02598]
    assert pri["pitch"]["pitch_v_m"] == [0.0428, 0.038]
    ant = pri["antenna"]  # fact 6
    assert ant["values"] == ["isotropic", "short_dipole_z"] and ant["weights"] == [0.5, 0.5]
    assert ant["peak_gain_linear"] == [None, 1.5] and ant["axis"] == [0.0, 0.0, 1.0]
    assert pri["point_error"] == {"xy_half_width_m": 0.02, "plane_height_half_width_m": 0.02}
    assert pri["floor"]["polarization"] == {"values": ["V"], "weights": [1.0]}
    fe = pri["feasibility"]
    assert fe["min_node_tile_distance_wavelengths"] == 1.0
    assert fe["los_must_not_cross_tiles"] is False and fe["tiles_must_not_overlap"] is True
    nf = proto["models"]["near_field"]
    assert nf["point_vs_patch"]["n_draws"] == 200
    assert "full-wave" in nf["rigorous_reference"]
    assert any("coupling" in x for x in nf["not_modelled"])
    assert proto["uncertainty_budget"]["sigma_pos"]["z_half_width_m"] == 0.02
    facts = proto["author_facts"]
    assert facts["recorded_before_unblinding"] is True and len(facts["facts"]) == 7
    assert "2026-10-11" in facts["source"] and "sketch" in facts["source"]


def test_revision_history(proto: dict) -> None:
    v1, v2 = proto["revision_history"]
    assert (v1["version"], v1["commit"], v1["date"]) == ("1.0", "b03e52d", "2026-10-10")
    assert v1["power"]["c2_detection_count"] == "0/500" and v1["power"]["underpowered"] is True
    assert (v2["version"], v2["date"]) == ("2.0", "2026-10-11")
    assert "never opened" in v2["blinding"]


def test_protocol_md_carries_the_numbers(proto: dict) -> None:
    md = " ".join((PROTOCOL_DIR / "PROTOCOL.md").read_text(encoding="utf-8").split())
    for needle in (
        "## Revision history",
        "b03e52d",
        "2026-10-11",
        "0/500",
        "hand sketch",
        "K = 2000",
        "sigma_tot = 2.0 dB",
        "10^4 resamples",
        "2000 bootstrap resamples per decoy",
        "500 decoys",
        "below 0.5",
        '"underpowered"',
        "consistency check under geometric uncertainty",
        "{0.6, 1.2, 1.8} m (18 cells)",
        "{0.9, 1.5} m (12 cells)",
        "Leave-one-ring-out (5 folds",
        "Leave-one-wedge-out (6 folds",
        "<= 1.5 dB",
        "+ 0.5 dB",
        ">= 1.0 dB",
        ">= 80 % of draws",
        "rho >= 0.3",
        "(35 deg, 1.5 m)",
        "O_T, the Tx (P = 1",
        "U[1.88, 2.05] m",
        "U[1.90, 2.00] m",
        "U[40, 52] deg",
        "22.5 deg + U[-3, 3] deg",
        "[s1 s1 s2 s2 s1 s1]",
        "U[-2, 2] cm",
        "U[0.6, 1.0]",
        "U[-15, 15] deg",
        "U[0.8, 1.5] m",
        "U[1, 3] m",
        "U[-45, 0] deg",
        "U[0, 5] mm",
        "1.0 dB assumed; sensitivity 0.5 and 2.0 dB",
        "**S-OR**",
        "**S-NF**",
        "element-wise near-field summation",
        "edge of the element-wise model's validity",
        "MoM/FDTD",
        "POWER_ANALYSIS_RESULT:BEGIN",
        "POWER_ANALYSIS_RESULT:END",
        "Diagnostic, not part of the protocol",
    ):
        assert needle in md, needle


def test_reference_values_match_sremu(proto: dict) -> None:
    pri = proto["prior"]
    t = load_table({"library": pri["gamma"]["table"], "assume_mag": {"sigma1": 1, "sigma2": 1}})
    np.testing.assert_allclose(
        np.degrees(np.angle(t.gamma_at(pri["fc_hz"]))), pri["gamma"]["nominal_phase_deg"], atol=1e-9
    )
    er, sg = p2040_material(pri["floor"]["material"], pri["fc_hz"])
    assert er == pri["floor"]["eps_r"]
    assert math.isclose(sg, pri["floor"]["sigma_s_per_m"], rel_tol=1e-12)
    assert math.isclose(
        float(np.mean(proto["data"]["disclosed_descriptive_statistics"]["ring_means_dbm"])),
        -18.318,
        abs_tol=1e-9,
    )


def test_cells_and_splits(proto: dict) -> None:
    lay = cell_layout(proto)
    assert lay.n == 30
    assert lay.index(35.0, 1.5) == 6 * 3 + 4
    assert (lay.theta_deg[7], lay.r_m[7]) == (5.0, 0.9)
    sm = split_masks(proto, lay)
    tr, te = sm["primary"]
    assert tr.sum() == 18 and te.sum() == 12 and not np.any(tr & te)
    assert set(np.unique(lay.r_m[te])) == {0.9, 1.5}
    folds = [k for k in sm if k != "primary"]
    assert len([k for k in folds if k.startswith("loro")]) == 5
    assert len([k for k in folds if k.startswith("lowo")]) == 6
    for k in folds:
        ftr, fte = sm[k]
        assert np.all(ftr ^ fte)
    snf = sensitivity_run(proto, "S-NF")["split"]
    tr2, te2 = ring_split(proto, snf["train_rings_m"], snf["test_rings_m"], lay)
    assert tr2.sum() == 12 and np.array_equal(te2, te)
    assert not np.any(tr2 & (lay.r_m == 1.8))
    with pytest.raises(ValueError, match="disjoint"):
        ring_split(proto, [0.6, 0.9], [0.9], lay)


def test_prediction_spec_ignores_the_power_result(proto: dict) -> None:
    other = json.loads(json.dumps(proto))
    other["power_analysis_result"] = {"c2_detection_rate": 0.123}
    other["sensitivity_runs"][-1]["description"] += " (edited)"
    assert prediction_spec_sha256(other) == prediction_spec_sha256(proto)
    other["prior"]["geometry"]["radius_m"] = [1.9, 2.0]
    assert prediction_spec_sha256(other) != prediction_spec_sha256(proto)


def test_s_or_variant_protocol(proto: dict, sor: dict) -> None:
    assert prior_variants(proto) == ("S-OR",)
    spec = sensitivity_run(proto, "S-OR")
    g = sor["prior"]["geometry"]
    assert g == spec["prior_overrides"]["geometry"]
    assert (g["model"], g["polar_origin"]) == ("flat_panel_v1", "O_R")
    assert g["theta0_direction"] == {"values": ["normal", "plane"], "weights": [0.5, 0.5]}
    assert (g["d_m"], g["a_deg"], g["tile_gap_m"]) == ([1.0, 3.0], [-45.0, 0.0], [0.0, 0.005])
    assert sor["prior"]["feasibility"]["los_must_not_cross_tiles"] is True
    for key in ("tiles", "pitch", "cell_model", "gamma", "antenna", "point_error", "floor"):
        assert sor["prior"][key] == proto["prior"][key], key
    assert sor["rng"]["prior_draw"]["name"] == "vlab_prior_sor/{k}"
    assert sor["variant"] == "S-OR" and "power_analysis_result" not in sor
    assert prediction_spec_sha256(sor) != prediction_spec_sha256(proto)
    assert variant_protocol(proto, None) == json.loads(json.dumps(proto))
    with pytest.raises(KeyError, match="not an alternative-prior"):
        variant_protocol(proto, "S1")


# --------------------------------------------------------------------------- antennas


def test_antenna_tables_are_generated(proto: dict) -> None:
    for name in ("short_dipole_z",):
        path = pp.antenna_table_path(name, proto)
        assert path.read_text(encoding="utf-8") == pp.antenna_table_text(name, proto), (
            f"{path} differs from its generator; run prior.write_antenna_tables()"
        )


def test_tabulated_dipole_is_a_vertical_short_dipole(proto: dict) -> None:
    spec = pp._antenna_spec("short_dipole_z", proto)
    tab = spec["pattern"]["tabulated"]
    ant = Tabulated.from_csv(
        tab["file"], tab["provenance"], np.asarray(spec["boresight"]), np.array(tab["ref_axis"])
    )
    el = np.radians(np.linspace(-80.0, 80.0, 161))
    for az in (0.0, 1.0, 2.5):  # azimuth-omnidirectional
        dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
        np.testing.assert_allclose(ant.gain(dirs), 1.5 * np.cos(el) ** 2, rtol=2e-3)
    assert ant.gain(np.array([[0.0, 0.0, 1.0]]))[0] == pytest.approx(1.5e-4, rel=1e-6)
    assert pp._antenna_spec("isotropic", proto) == {"pattern": "isotropic"}


# --------------------------------------------------------------------------- prior


def test_prior_draws_are_keyed_and_within_the_v2_bounds(proto: dict) -> None:
    a = pp.draw_prior(proto, purpose="trial", name="vlab_prior/3")
    b = pp.draw_prior(proto, purpose="trial", name="vlab_prior/3")
    c = pp.draw_prior(proto, purpose="trial", name="vlab_prior/4")
    assert a.as_record() == b.as_record()
    np.testing.assert_array_equal(a.point_dxy_m, b.point_dxy_m)
    assert a.as_record() != c.as_record()
    for k in range(16):
        d = pp.draw_prior(proto, purpose="trial", name=f"vlab_prior/{k}")
        ok, why = pp.feasibility(pp.geometry(d, proto), proto)
        assert ok, why
        assert (d.geometry_model, d.origin, d.theta0, d.polarization) == (
            "arc_facing_tx",
            "O_T",
            "arc",
            "V",
        )
        assert 1.88 <= d.radius_m <= 2.05 and 40.0 <= d.span_deg <= 52.0
        assert 19.5 <= d.centre_deg <= 25.5
        assert all(math.isnan(x) for x in (d.d_m, d.a_deg, d.gap_m))
        assert np.max(np.abs(d.point_dxy_m)) <= 0.02 and abs(d.plane_dz_m) <= 0.02
        assert 0.8 <= d.height_m <= 1.5
        assert all(0.6 <= m <= 1.0 for m in d.gamma_mag)
        assert all(-15.0 <= p <= 15.0 for p in d.gamma_phase_offset_deg)
        assert d.antenna in ("isotropic", "short_dipole_z")


def _arc(proto: dict, **kw: object) -> pp.PriorDraw:
    base = pp.draw_prior(proto, purpose="trial", name="vlab_prior/0")
    fields = {
        "radius_m": 1.95,
        "span_deg": 46.0,
        "centre_deg": 22.5,
        "pitch": "square_42p8mm",
        "cell_model": "point",
        "antenna": "isotropic",
        "point_dxy_m": np.zeros((30, 2)),
        "plane_dz_m": 0.0,
        "height_m": 1.2,
    }
    fields.update(kw)
    return replace(base, **fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(("radius", "span", "centre"), [(1.95, 46.0, 22.5), (1.9, 52.0, 19.5)])
def test_arc_geometry_is_the_author_sketch(
    proto: dict, radius: float, span: float, centre: float
) -> None:
    """Fact 2/4/5: tile k at (R cos phi_k, R sin phi_k, h), phi_k = centre + (k + 0.5 - 3) S/6,
    facing the Tx, 5 cells along the tangential horizontal u, 8 along +z."""
    dr = _arc(proto, radius_m=radius, span_deg=span, centre_deg=centre)
    g = pp.geometry(dr, proto)
    phi = np.radians(centre + (np.arange(6) + 0.5 - 3.0) * span / 6.0)
    e = np.stack([np.cos(phi), np.sin(phi), np.zeros(6)], axis=1)
    np.testing.assert_allclose(g.tx_m, [0.0, 0.0, 1.2])
    np.testing.assert_allclose(g.tile_center_m, radius * e + [0.0, 0.0, 1.2], atol=1e-12)
    np.testing.assert_allclose(g.tile_normal, -e, atol=1e-12)
    np.testing.assert_allclose(g.tile_phi_deg, np.degrees(phi), atol=1e-9)
    np.testing.assert_allclose(np.sum(g.tile_u_axis * e, axis=1), 0.0, atol=1e-12)  # tangential
    np.testing.assert_allclose(g.tile_u_axis[:, 2], 0.0)  # horizontal
    np.testing.assert_allclose(np.cross(g.tile_normal, g.tile_u_axis), np.tile([0, 0, 1], (6, 1)))
    to_tx = g.tx_m[None, :] - g.tile_center_m
    np.testing.assert_allclose(np.sum(to_tx * g.tile_normal, axis=1), radius)  # faces the Tx
    assert list(g.tile_state) == [0, 0, 1, 1, 0, 0] and np.all(np.diff(g.tile_phi_deg) > 0)
    assert (g.n_u, g.n_v, g.tile_size_m) == (5, 8, pytest.approx((0.214, 0.3424)))
    gi = pp.geometry(_arc(proto, pitch="indicon_38x25p98mm"), proto)
    assert gi.tile_size_m == pytest.approx((0.1299, 0.304))
    lay = cell_layout(proto)
    th = np.radians(lay.theta_deg)
    np.testing.assert_allclose(g.points_m[:, 0], lay.r_m * np.cos(th))
    np.testing.assert_allclose(g.points_m[:, 1], lay.r_m * np.sin(th))
    sc = pp.scenario_dict(dr, proto, ground=False)
    assert [r["id"] for r in sc["ris"]] == pp.panel_ids(6)
    for k, r in enumerate(sc["ris"]):
        assert (r["grid"]["n_u"], r["grid"]["n_v"]) == (5, 8)
        assert r["initial"]["states"] == [int(g.tile_state[k])] * 40
        np.testing.assert_allclose(r["pose"]["normal"], g.tile_normal[k])


def test_arc_feasibility_rules(proto: dict) -> None:
    # every point in front of every tile, never behind; LOS cannot reach a tile
    g = pp.geometry(_arc(proto), proto)
    assert pp.feasibility(g, proto) == (True, "ok")
    side = np.einsum("tpj,tj->tp", g.points_m[None] - g.tile_center_m[:, None], g.tile_normal)
    assert np.all(side > 0)
    # worst case for the (unapplied) LOS rule: smallest R, points pushed outward by 2 cm
    out = np.full((30, 2), 0.02)  # largest outward error for theta in [0, 45] deg
    worst = pp.geometry(_arc(proto, radius_m=1.88, point_dxy_m=out), proto)
    a = np.broadcast_to(worst.tx_m, worst.points_m.shape)
    for corners in pp.tile_corners(worst):
        assert not np.any(pp.segments_hit_polygon(a, worst.points_m, corners))
    assert np.max(np.linalg.norm(worst.points_m[:, :2], axis=1)) < 1.83
    # that worst case is closer than one wavelength to a tile: near_tile
    assert pp.feasibility(worst, proto) == (False, "near_tile")
    assert float(np.min(pp.node_tile_distance(worst, worst.points_m))) < LAM
    # a gap just above one wavelength is feasible
    ok = pp.geometry(_arc(proto, radius_m=1.8 + LAM + 1e-3), proto)
    assert pp.feasibility(ok, proto) == (True, "ok")
    # tiles never overlap within the prior bounds; an artificially short arc overlaps
    tight = pp.geometry(_arc(proto, radius_m=1.88, span_deg=40.0), proto)
    assert not pp._tiles_overlap(tight)
    assert pp.feasibility(pp.geometry(_arc(proto, radius_m=2.05, span_deg=30.0), proto), proto) == (
        False,
        "tiles_overlap",
    )


def test_s_or_flat_panel_rows(sor: dict) -> None:
    """The v1 O_R rows (sensitivity run S-OR) with the six v2 tiles in one flat row."""
    base = pp.draw_prior(sor, purpose="trial", name="vlab_prior_sor/0")
    assert base.geometry_model == "flat_panel_v1" and base.origin == "O_R"
    assert 1.0 <= base.d_m <= 3.0 and -45.0 <= base.a_deg <= 0.0 and 0 <= base.gap_m <= 0.005
    d = replace(
        base,
        theta0="normal",
        d_m=2.0,
        a_deg=-30.0,
        gap_m=0.002,
        pitch="square_42p8mm",
        point_dxy_m=np.zeros((30, 2)),
        plane_dz_m=0.0,
        height_m=1.2,
    )
    g = pp.geometry(d, sor)
    a = math.radians(-30.0)
    np.testing.assert_allclose(g.tx_m, [2 * math.cos(a), 2 * math.sin(a), 1.2])
    np.testing.assert_allclose(g.tile_normal, np.tile([1.0, 0.0, 0.0], (6, 1)))
    np.testing.assert_allclose(np.mean(g.tile_center_m, axis=0), [0.0, 0.0, 1.2], atol=1e-12)
    step = np.diff(g.tile_center_m, axis=0)
    np.testing.assert_allclose(np.linalg.norm(step, axis=1), 5 * 0.0428 + 0.002)
    np.testing.assert_allclose(np.cross(g.tile_normal[0], g.tile_u_axis[0]), [0, 0, 1])
    assert pp.feasibility(g, sor) == (True, "ok")
    gp = pp.geometry(replace(d, theta0="plane"), sor)
    np.testing.assert_allclose(gp.tile_normal[0], [0.0, 1.0, 0.0])
    assert gp.tx_m[0] < 0 < gp.tx_m[1]
    assert pp.feasibility(gp, sor) == (False, "near_tile")  # theta = 0 points in the plane
    # a Tx behind the panel: LOS through a tile (the v1 rule, applied in S-OR only)
    behind = replace(d, d_m=1.0, a_deg=180.0, pitch="indicon_38x25p98mm")
    assert pp.feasibility(pp.geometry(behind, sor), sor) == (False, "los_through_tile")


def test_s_or_plane_needs_the_indicon_cell(proto: dict, sor: dict) -> None:
    """The implicit S-OR conditioning recorded in PROTOCOL.md section 11: under O_R/plane the
    theta = 0 points lie in the panel plane and the 1.284 m flat row of 42.8 mm-cell tiles
    always passes within one wavelength of them, so only INDICON O_R/plane draws are feasible."""
    base = pp.draw_prior(sor, purpose="trial", name="vlab_prior_sor/0")
    g = np.random.default_rng(7)
    for _ in range(50):
        d = replace(
            base,
            theta0="plane",
            pitch="square_42p8mm",
            d_m=float(g.uniform(1.0, 3.0)),
            a_deg=float(g.uniform(-45.0, 0.0)),
            gap_m=float(g.uniform(0.0, 0.005)),
            point_dxy_m=g.uniform(-0.02, 0.02, size=(30, 2)),
            plane_dz_m=float(g.uniform(-0.02, 0.02)),
        )
        assert pp.feasibility(pp.geometry(d, sor), sor) == (False, "near_tile")
    ok = replace(
        base,
        theta0="plane",
        pitch="indicon_38x25p98mm",
        d_m=2.0,
        a_deg=-30.0,
        gap_m=0.005,
        point_dxy_m=np.zeros((30, 2)),
        plane_dz_m=0.0,
    )
    assert pp.feasibility(pp.geometry(ok, sor), sor) == (True, "ok")
    rec = sensitivity_run(proto, "S-OR")["tensor_result"]["accepted_pitch_counts"]
    assert rec["O_R/plane"]["square_42p8mm"] == 0
    assert sum(rec["O_R/plane"].values()) + sum(rec["O_R/normal"].values()) == 2000


# --------------------------------------------------------------------------- predictions


@pytest.mark.parametrize(
    ("cell_model", "antenna", "mags"),
    [("point", "short_dipole_z", (1.0, 1.0)), ("po_patch", "isotropic", (0.7, 0.9))],
)
def test_posthoc_gamma_equals_sremu_evaluate(
    proto: dict, cell_model: str, antenna: str, mags: tuple[float, float]
) -> None:
    """The drawn Gamma applied to the separable coefficients of the six tile panels
    reproduces ScenePropagation.evaluate (assume_mag = the drawn magnitudes) for M0, M2, M3."""
    d = _arc(
        proto,
        cell_model=cell_model,
        antenna=antenna,
        pitch="indicon_38x25p98mm",
        gamma_mag=mags,
        gamma_phase_offset_deg=(0.0, 0.0),
    )
    res = pr.evaluate_draw(d, proto, backend="numpy_ref")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        for ground, col in ((True, 3), (False, 2)):
            sc = pp.scenario_dict(d, proto, ground=ground, backend="numpy_ref")
            for r in sc["ris"]:
                r["states"]["assume_mag"] = {"sigma1": mags[0], "sigma2": mags[1]}
            m = from_scenario(sc)
            if cell_model == "po_patch":
                assert m.quadrature("t0").n_points > 40  # sub-cells, not points
            s = m.snapshot(0)
            links = pr._rx_links(m)
            ev = m.evaluate(s, {pid: m.initial_config(pid) for pid in m.panel_ids()})
            ref = 10 * np.log10(np.abs(np.array([ev[lk] for lk in links])) ** 2)
            np.testing.assert_allclose(res.power_db[col], ref, rtol=0, atol=1e-9)
            if not ground:
                ev0 = m.evaluate(s, {})
                ref0 = 10 * np.log10(np.abs(np.array([ev0[lk] for lk in links])) ** 2)
                np.testing.assert_allclose(res.power_db[0], ref0, rtol=0, atol=1e-9)


def test_gamma_draw_and_mirror(proto: dict) -> None:
    d = _arc(proto, gamma_mag=(0.7, 0.9), gamma_phase_offset_deg=(10.0, -5.0))
    tab = load_table(
        {"library": "lab_binary_3p5ghz", "assume_mag": {"sigma1": 1, "sigma2": 1}}
    ).gamma_at(3.5e9)
    g = pr._gamma(d, tab)
    np.testing.assert_allclose(np.abs(g), [0.7, 0.9])
    np.testing.assert_allclose(np.degrees(np.angle(g)), [49.64 + 10.0, -153.77 - 5.0], atol=1e-9)
    r = pr.evaluate_draw(d, proto)
    assert r.power_db.shape == (4, 30) and np.all(np.isfinite(r.power_db))
    geo = pp.geometry(d, proto)
    np.testing.assert_allclose(r.dist_m, np.linalg.norm(geo.points_m - geo.tx_m, axis=1))
    # M1 (every tile sigma1) differs from M2 ([s1 s1 s2 s2 s1 s1]) only through the middle tiles
    assert not np.allclose(r.power_db[1], r.power_db[2])
    r2 = pr.evaluate_draw(d, proto, backend="numpy_ref")  # numba and numpy backends agree
    np.testing.assert_allclose(r.power_db, r2.power_db, rtol=0, atol=1e-9)


def test_nearfield_warning_is_the_only_one_suppressed(proto: dict) -> None:
    """models.near_field: sremu's 3-lambda ValidityWarning is expected at the 1.8 m ring; it
    alone is suppressed and counted; the minimum point-to-cell distance is recorded."""
    d = _arc(proto, radius_m=1.92)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        m = from_scenario(pp.scenario_dict(d, proto, ground=False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.snapshot(0)
    msgs = [str(w.message) for w in caught if issubclass(w.category, ValidityWarning)]
    assert msgs and all(pr.NEARFIELD_WARNING.search(x) for x in msgs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing escapes evaluate_draw
        r = pr.evaluate_draw(d, proto)
    assert r.n_nearfield_warnings == len(msgs) > 0 and r.n_other_warnings == 0
    geo = pp.geometry(d, proto)
    cc = geo.cell_centres()
    want = np.min(np.linalg.norm(geo.points_m[:, None] - cc[None], axis=2))
    assert r.min_point_cell_centre_m == pytest.approx(want)
    assert LAM <= r.min_point_tile_m <= r.min_point_cell_centre_m < 3 * LAM
    assert r.frac_points_behind_tiles == 0.0
    assert not [w for w in caught if not issubclass(w.category, ValidityWarning)]


def test_paired_point_vs_patch(proto: dict) -> None:
    d = _arc(proto, cell_model="point", pitch="indicon_38x25p98mm")
    assert pr.alternative_cell_model(d) == "po_patch"
    alt = pr.evaluate_alternative(d, proto)
    ref = pr.evaluate_draw(replace(d, cell_model="po_patch"), proto)
    np.testing.assert_allclose(alt, ref.power_db, rtol=0, atol=1e-12)
    assert ref.richardson_level >= 0 and ref.converged_subcell_m < 0.2 * LAM + 1e-12


def test_tensor_is_deterministic_and_cached(proto: dict, tmp_path: Path) -> None:
    a = pr.compute_tensor(proto, K=3, jobs=1, n_sigma_pos=1, n_alt=2)
    b = pr.compute_tensor(proto, K=3, jobs=1, n_sigma_pos=1, n_alt=2)
    assert pr.content_sha256(a["arrays"]) == pr.content_sha256(b["arrays"])
    for key in ("pred_db", "dist_m", "sigma_pos_db", "point_dxy_m", "pred_alt_cm_db"):
        np.testing.assert_array_equal(a["arrays"][key], b["arrays"][key])
    arr = a["arrays"]
    assert arr["pred_db"].shape == (3, 30, 4) and arr["pred_alt_cm_db"].shape == (2, 30, 4)
    assert np.all(arr["sigma_pos_db"] >= 0) and np.all(np.isfinite(arr["sigma_pos_db"]))
    assert np.all((arr["radius_m"] >= 1.88) & (arr["radius_m"] <= 2.05))
    assert np.all(arr["min_point_tile_m"] >= LAM) and np.all(arr["n_other_warnings"] == 0)
    assert a["codes"]["origin"] == ["O_T"] and a["codes"]["theta0"] == ["arc"]
    pvp = pr.point_vs_patch_spread(arr, a["codes"], proto)
    assert pvp["n_draws"] == 2 and set(pvp["rings_m"]) == {"0.6", "0.9", "1.2", "1.5", "1.8"}
    summ = pr._summary(a, proto)
    assert summ["other_validity_warnings_total"] == 0 and summ["K"] == 3
    meta = pr.save_cache(tmp_path / "t.npz", a, proto)
    t = pr.load_cache(tmp_path / "t.npz", protocol=proto)
    assert t.K == 3 and t.meta["content_sha256"] == meta["content_sha256"]
    np.testing.assert_array_equal(t.model("M2"), a["arrays"]["pred_db"][:, :, 2])
    assert re.fullmatch(r"[0-9a-f]{64}", meta["file_sha256"])
    other = json.loads(json.dumps(proto))
    other["prior"]["geometry"]["span_deg"] = [40.0, 50.0]
    with pytest.raises(ValueError, match="prediction spec"):
        pr.load_cache(tmp_path / "t.npz", protocol=other)
    with pytest.raises(ValueError, match="prediction spec"):
        pr.load_cache(tmp_path / "t.npz", protocol=variant_protocol(proto, "S-OR"))
    # tampering with an array is detected
    with np.load(tmp_path / "t.npz") as z:
        arrs = {k: z[k] for k in z.files}
    arrs["pred_db"] = arrs["pred_db"] + 1e-9
    np.savez(tmp_path / "bad.npz", **arrs)
    with pytest.raises(ValueError, match="content hash"):
        pr.load_cache(tmp_path / "bad.npz")


def test_s_or_tensor(sor: dict) -> None:
    a = pr.compute_tensor(sor, K=2, jobs=1, n_sigma_pos=0, n_alt=0)
    assert a["codes"]["geometry_model"] == ["flat_panel_v1"] and a["codes"]["origin"] == ["O_R"]
    assert np.all(np.isnan(a["arrays"]["radius_m"])) and np.all(a["arrays"]["d_m"] >= 1.0)
    assert np.all(np.isfinite(a["arrays"]["pred_db"]))
