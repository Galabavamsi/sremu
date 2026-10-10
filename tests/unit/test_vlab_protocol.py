"""V-LAB protocol files, prior and prediction code (DESIGN.md section 10.9; WP10).

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
    split_masks,
)

from sremu.physics.antennas import CosPower, Tabulated, cos_power_alpha_for_gain_dbi
from sremu.physics.propagation import from_scenario
from sremu.physics.response import load_table
from sremu.physics.walls import p2040_material
from sremu.units import C0

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def proto() -> dict:
    return load_protocol()


# --------------------------------------------------------------------------- protocol files


def test_design_numbers_in_protocol_json(proto: dict) -> None:
    """The numbers DESIGN 10.9 fixes, as machine-readable values."""
    d = proto["data"]
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
    assert pri["polar_origin"] == {"values": ["O_R", "O_T"], "weights": [0.7, 0.3]}
    assert pri["placement"]["d_m"] == [1.0, 3.0] and pri["placement"]["a_deg"] == [-45.0, 0.0]
    assert pri["gamma"]["mag"] == [0.6, 1.0] and pri["gamma"]["phase_offset_deg"] == [-15.0, 15.0]
    assert pri["pitch"]["pitch_u_m"] == [0.0428, 0.038]
    assert pri["pitch"]["pitch_v_m"] == [0.0428, 0.02598]
    assert pri["tile_gap_m"] == [0.0, 0.005]
    assert pri["floor"]["antenna_height_m"] == [0.8, 1.5]
    assert proto["naming"]["label"] == "consistency check under geometric uncertainty"


def test_protocol_md_carries_the_numbers(proto: dict) -> None:
    md = " ".join((PROTOCOL_DIR / "PROTOCOL.md").read_text(encoding="utf-8").split())
    for needle in (
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
        "P = 0.7",
        "U[1, 3] m",
        "U[-45, 0] deg",
        "U[0.6, 1.0]",
        "U[-15, 15] deg",
        "U[0, 5] mm",
        "U[0.8, 1.5] m",
        "1.0 dB assumed; sensitivity 0.5 and 2.0 dB",
        "POWER_ANALYSIS_RESULT:BEGIN",
        "POWER_ANALYSIS_RESULT:END",
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
        tr, te = sm[k]
        assert np.all(tr ^ te)


def test_prediction_spec_ignores_the_power_result(proto: dict) -> None:
    other = json.loads(json.dumps(proto))
    other["power_analysis_result"] = {"c2_detection_rate": 0.123}
    assert prediction_spec_sha256(other) == prediction_spec_sha256(proto)
    other["prior"]["placement"]["d_m"] = [1.0, 3.5]
    assert prediction_spec_sha256(other) != prediction_spec_sha256(proto)


# --------------------------------------------------------------------------- antennas


def test_antenna_tables_are_generated(proto: dict) -> None:
    for name in ("patch_6dbi", "patch_9dbi"):
        path = pp.antenna_table_path(name, proto)
        assert path.read_text(encoding="utf-8") == pp.antenna_table_text(name, proto), (
            f"{path} differs from its generator; run prior.write_antenna_tables()"
        )


@pytest.mark.parametrize("g0", [6.0, 9.0])
def test_tabulated_patch_matches_cos_power_above_the_floor(proto: dict, g0: float) -> None:
    name = f"patch_{g0:g}dbi"
    tab = Tabulated.from_csv(
        pp.antenna_table_path(name, proto),
        {"source_kind": "assumed", "description": "t", "reference": "t"},
    )
    cp = CosPower(cos_power_alpha_for_gain_dbi(g0))
    psi = np.radians(np.linspace(0.0, 179.0, 359))
    dirs = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], axis=1)
    g_tab = 10 * np.log10(tab.gain(dirs))
    with np.errstate(divide="ignore"):
        g_cp = 10 * np.log10(cp.gain(dirs))
    floor = g0 - proto["prior"]["antenna"]["front_to_back_db"]
    above = g_cp > floor + 0.5
    assert np.max(np.abs(g_tab[above] - g_cp[above])) < 0.01
    assert np.all(g_tab >= floor - 1e-9)
    assert math.isclose(g_tab[0], g0, abs_tol=1e-6)


# --------------------------------------------------------------------------- prior


def test_prior_draws_are_keyed_and_feasible(proto: dict) -> None:
    a = pp.draw_prior(proto, purpose="trial", name="vlab_prior/3")
    b = pp.draw_prior(proto, purpose="trial", name="vlab_prior/3")
    c = pp.draw_prior(proto, purpose="trial", name="vlab_prior/4")
    assert a.as_record() == b.as_record()
    np.testing.assert_array_equal(a.point_dxy_m, b.point_dxy_m)
    assert a.as_record() != c.as_record()
    for k in range(12):
        d = pp.draw_prior(proto, purpose="trial", name=f"vlab_prior/{k}")
        ok, why = pp.feasibility(pp.geometry(d, proto), proto)
        assert ok, why
        assert np.max(np.abs(d.point_dxy_m)) <= 0.02
        assert abs(d.plane_dz_m) <= 0.05 and 0.8 <= d.height_m <= 1.5
        assert 1.0 <= d.d_m <= 3.0 and -45.0 <= d.a_deg <= 0.0
        assert all(0.6 <= m <= 1.0 for m in d.gamma_mag)


def _draw(proto: dict, **kw: object) -> pp.PriorDraw:
    base = pp.draw_prior(proto, purpose="trial", name="vlab_prior/0")
    fields = {
        "origin": "O_R",
        "theta0": "normal",
        "d_m": 2.0,
        "a_deg": -30.0,
        "layout": "A",
        "pitch": "square_42p8mm",
        "gap_m": 0.002,
        "cell_model": "point",
        "antenna": "isotropic",
        "point_dxy_m": np.zeros((30, 2)),
        "plane_dz_m": 0.0,
        "height_m": 1.2,
    }
    fields.update(kw)
    return replace(base, **fields)  # type: ignore[arg-type]


def test_frames(proto: dict) -> None:
    lay = cell_layout(proto)
    g = pp.geometry(_draw(proto), proto)
    np.testing.assert_allclose(g.ris_center_m, [0, 0, 1.2])
    np.testing.assert_allclose(g.normal, [1, 0, 0])
    np.testing.assert_allclose(
        g.tx_m, [2 * math.cos(math.radians(-30)), 2 * math.sin(math.radians(-30)), 1.2]
    )
    th = np.radians(lay.theta_deg)
    np.testing.assert_allclose(g.points_m[:, 0], lay.r_m * np.cos(th))
    np.testing.assert_allclose(g.points_m[:, 1], lay.r_m * np.sin(th))
    np.testing.assert_allclose(np.cross(g.normal, g.u_axis), [0, 0, 1])
    g = pp.geometry(_draw(proto, theta0="plane"), proto)
    np.testing.assert_allclose(g.normal, [0, 1, 0])
    assert g.tx_m[0] < 0 < g.tx_m[1]  # opposite side of the normal from the points
    g = pp.geometry(_draw(proto, origin="O_T"), proto)
    np.testing.assert_allclose(g.tx_m, [0, 0, 1.2])
    np.testing.assert_allclose(g.normal, [-1, 0, 0])
    assert (g.tx_m - g.ris_center_m) @ g.normal > 0  # the panel faces the Tx
    g = pp.geometry(_draw(proto, origin="O_T", theta0="plane"), proto)
    np.testing.assert_allclose(g.normal, [0, 1, 0])
    assert (g.tx_m - g.ris_center_m) @ g.normal > 0


@pytest.mark.parametrize(
    ("layout", "shape", "cols"),
    [("A", (24, 5), [0, 1, 0]), ("B", (48, 5), [0, 0, 1, 1, 0, 0]), ("C", (24, 10), [0, 1, 0])],
)
def test_layout_states(proto: dict, layout: str, shape: tuple[int, int], cols: list[int]) -> None:
    g = pp.geometry(_draw(proto, layout=layout), proto)
    assert (g.n_u, g.n_v) == shape
    st = g.cell_state.reshape(g.n_v, g.n_u)
    for tu, s in enumerate(cols):
        assert np.all(st[:, 8 * tu : 8 * tu + 8] == s)


def test_feasibility_rules(proto: dict) -> None:
    # O_R / plane with the wide layout B: points at theta = 0 lie inside the panel plane
    ok, why = pp.feasibility(pp.geometry(_draw(proto, theta0="plane", layout="B"), proto), proto)
    assert (ok, why) == (False, "near_panel")
    # O_T / normal with the RIS between the Tx and far points: LOS through the panel
    g = pp.geometry(_draw(proto, origin="O_T", d_m=1.0, a_deg=-5.0, layout="C"), proto)
    ok, why = pp.feasibility(g, proto)
    assert not ok and why in ("los_through_panel", "near_panel")
    assert pp.feasibility(pp.geometry(_draw(proto), proto), proto) == (True, "ok")
    assert C0 / proto["prior"]["fc_hz"] == pytest.approx(0.0856549880, abs=1e-9)


# --------------------------------------------------------------------------- predictions


@pytest.mark.parametrize("antenna", ["isotropic", "patch_9dbi"])
def test_posthoc_gamma_equals_sremu_evaluate(proto: dict, antenna: str) -> None:
    """The drawn Gamma applied to the snapshot's separable coefficients reproduces
    ScenePropagation.evaluate (nominal Gamma, |Gamma| = 1) for M0, M2 and M3."""
    d = _draw(
        proto,
        antenna=antenna,
        layout="C",
        gamma_mag=(1.0, 1.0),
        gamma_phase_offset_deg=(0.0, 0.0),
        polarization="H",
    )
    res = pr.evaluate_draw(d, proto, backend="numpy_ref")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ground, col in ((True, 3), (False, 2)):
            m = from_scenario(pp.scenario_dict(d, proto, ground=ground, backend="numpy_ref"))
            s = m.snapshot(0)
            links = pr._rx_links(m)
            ev = m.evaluate(s, {"ris": m.initial_config("ris")})
            ref = 10 * np.log10(np.abs(np.array([ev[lk] for lk in links])) ** 2)
            np.testing.assert_allclose(res.power_db[col], ref, rtol=0, atol=1e-9)
            if not ground:
                ev0 = m.evaluate(s, {})
                ref0 = 10 * np.log10(np.abs(np.array([ev0[lk] for lk in links])) ** 2)
                np.testing.assert_allclose(res.power_db[0], ref0, rtol=0, atol=1e-9)


def test_posthoc_gamma_po_patch_unequal_magnitudes(proto: dict) -> None:
    """Reviewer addition: the post-hoc Gamma path also holds for po_patch quadrature and
    unequal |Gamma| (sremu evaluate with assume_mag = the drawn magnitudes)."""
    mags = (0.7, 0.9)
    d = _draw(
        proto,
        cell_model="po_patch",
        pitch="indicon_38x25p98mm",
        layout="A",
        gamma_mag=mags,
        gamma_phase_offset_deg=(0.0, 0.0),
        polarization="V",
    )
    res = pr.evaluate_draw(d, proto)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ground, col in ((True, 3), (False, 2)):
            sc = pp.scenario_dict(d, proto, ground=ground)
            sc["ris"][0]["states"]["assume_mag"] = {"sigma1": mags[0], "sigma2": mags[1]}
            m = from_scenario(sc)
            assert m.quadrature("ris").n_points > 120  # sub-cells, not points
            s = m.snapshot(0)
            links = pr._rx_links(m)
            ev = m.evaluate(s, {"ris": m.initial_config("ris")})
            ref = 10 * np.log10(np.abs(np.array([ev[lk] for lk in links])) ** 2)
            np.testing.assert_allclose(res.power_db[col], ref, rtol=0, atol=1e-9)


def test_gamma_draw_and_mirror(proto: dict) -> None:
    d = _draw(proto, gamma_mag=(0.7, 0.9), gamma_phase_offset_deg=(10.0, -5.0))
    tab = load_table(
        {"library": "lab_binary_3p5ghz", "assume_mag": {"sigma1": 1, "sigma2": 1}}
    ).gamma_at(3.5e9)
    g = pr._gamma(d, tab)
    np.testing.assert_allclose(np.abs(g), [0.7, 0.9])
    np.testing.assert_allclose(np.degrees(np.angle(g)), [49.64 + 10.0, -153.77 - 5.0], atol=1e-9)
    r = pr.evaluate_draw(d, proto)
    assert r.power_db.shape == (4, 30) and np.all(np.isfinite(r.power_db))
    assert np.allclose(
        r.dist_m,
        np.linalg.norm(pp.geometry(d, proto).points_m - pp.geometry(d, proto).tx_m, axis=1),
    )
    # numba and numpy backends agree
    r2 = pr.evaluate_draw(d, proto, backend="numpy_ref")
    np.testing.assert_allclose(r.power_db, r2.power_db, rtol=0, atol=1e-9)


def test_tensor_is_deterministic_and_cached(proto: dict, tmp_path: Path) -> None:
    a = pr.compute_tensor(proto, K=3, jobs=1, n_sigma_pos=1)
    b = pr.compute_tensor(proto, K=3, jobs=1, n_sigma_pos=1)
    assert pr.content_sha256(a["arrays"]) == pr.content_sha256(b["arrays"])
    for key in ("pred_db", "dist_m", "sigma_pos_db", "point_dxy_m"):
        np.testing.assert_array_equal(a["arrays"][key], b["arrays"][key])
    assert a["arrays"]["pred_db"].shape == (3, 30, 4)
    assert np.all(a["arrays"]["sigma_pos_db"] >= 0) and np.all(
        np.isfinite(a["arrays"]["sigma_pos_db"])
    )
    meta = pr.save_cache(tmp_path / "t.npz", a, proto)
    t = pr.load_cache(tmp_path / "t.npz", protocol=proto)
    assert t.K == 3 and t.meta["content_sha256"] == meta["content_sha256"]
    np.testing.assert_array_equal(t.model("M2"), a["arrays"]["pred_db"][:, :, 2])
    assert re.fullmatch(r"[0-9a-f]{64}", meta["file_sha256"])
    other = json.loads(json.dumps(proto))
    other["prior"]["tile_gap_m"] = [0.0, 0.004]
    with pytest.raises(ValueError, match="prediction spec"):
        pr.load_cache(tmp_path / "t.npz", protocol=other)
    # tampering with an array is detected
    with np.load(tmp_path / "t.npz") as z:
        arrs = {k: z[k] for k in z.files}
    arrs["pred_db"] = arrs["pred_db"] + 1e-9
    np.savez(tmp_path / "bad.npz", **arrs)
    with pytest.raises(ValueError, match="content hash"):
        pr.load_cache(tmp_path / "bad.npz")
