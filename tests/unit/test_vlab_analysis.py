"""V-LAB protocol pipeline: fits, bootstrap, Spearman, decisions, decoys (WP10).

Synthetic tensors and keyed synthetic datasets only; no measured data.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from validation.ants_fig4b import analysis as an
from validation.ants_fig4b import decoy as dc
from validation.ants_fig4b.predict import MODELS, TensorCache
from validation.ants_fig4b.protocol import cell_layout, load_protocol, split_masks

from sremu import rng as sremu_rng
from sremu.analysis import stats as sstats

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def proto() -> dict:
    return load_protocol()


def _tensor(pred: np.ndarray, dist: np.ndarray) -> TensorCache:
    k = pred.shape[0]
    codes = {"origin": ["O_R", "O_T"]}
    return TensorCache(
        arrays={"pred_db": pred, "dist_m": dist, "origin_idx": np.arange(k) % 2},
        meta={"codes": codes},
        path=Path("synthetic.npz"),
    )


def _structured(proto: dict, k: int, *, ris_db: float, seed: str) -> tuple[TensorCache, np.ndarray]:
    """Synthetic tensor: M0 = smooth distance law; M1/M2/M3 add a cell pattern of +-ris_db;
    the 'truth' is M2 of draw 0 (so M2 explains the structure when the draws agree)."""
    lay = cell_layout(proto)
    g = sremu_rng.generator(1, "trial", f"vlab_test/{seed}")
    dist = np.tile(lay.r_m, (k, 1)) * (1 + 0.01 * g.standard_normal((k, 30)))
    m0 = -20 * np.log10(dist)
    pattern = ris_db * np.sin(np.radians(4 * lay.theta_deg) + lay.r_m * 3.0)
    pred = np.empty((k, 30, 4))
    pred[:, :, 0] = m0
    pred[:, :, 1] = m0 + 0.2 * pattern
    pred[:, :, 2] = m0 + pattern + 0.05 * g.standard_normal((k, 30))
    pred[:, :, 3] = pred[:, :, 2] + 0.1
    return _tensor(pred, dist), pred[0, :, 2].copy()


def test_offset_and_log_distance_fits(proto: dict) -> None:
    t, _ = _structured(proto, 5, ris_db=3.0, seed="fit")
    lay = cell_layout(proto)
    tr, _te = split_masks(proto, lay)["primary"]
    g = sremu_rng.generator(2, "trial", "vlab_test/y")
    y = g.normal(-20, 3, 30)
    yhat, par = an.fitted_predictions(t, y, tr)
    for m in MODELS:
        np.testing.assert_allclose(np.mean(y[tr][None, :] - yhat[m][:, tr], axis=1), 0, atol=1e-12)
    for k in range(5):
        x = -10 * np.log10(t.arrays["dist_m"][k, tr])
        coef, *_ = np.linalg.lstsq(np.stack([np.ones_like(x), x], 1), y[tr], rcond=None)
        np.testing.assert_allclose([par["C_M0n"][k], par["n_M0n"][k]], coef, rtol=1e-10)


def test_vectorized_bootstrap_equals_sremu_paired_bootstrap(proto: dict) -> None:
    lay = cell_layout(proto)
    _tr, te = split_masks(proto, lay)["primary"]
    g = sremu_rng.generator(3, "trial", "vlab_test/e")
    errs = {m: g.normal(0, 2 + i, (7, 30)) for i, m in enumerate(("M0", "M1", "M2"))}
    n = int(te.sum())
    idx = sremu_rng.generator(4, "trial", "vlab_test/b").integers(0, n, size=(500, n))
    ci = an.bootstrap_delta_ci(
        errs, [("M0", "M2"), ("M1", "M2")], te, idx, chunk_bytes=2 * 500 * n * 8
    )
    for k in range(7):
        for a, b in (("M0", "M2"), ("M1", "M2")):
            ref = sstats.paired_bootstrap(
                [errs[a][k, te], errs[b][k, te]],
                sstats.rmse_difference,
                rng=sremu_rng.generator(4, "trial", "vlab_test/b"),
                n_resamples=500,
            )
            assert ci[(a, b)][0][k] == pytest.approx(ref.ci_low, abs=1e-12)
            assert ci[(a, b)][1][k] == pytest.approx(ref.ci_high, abs=1e-12)


def test_ring_removed(proto: dict) -> None:
    lay = cell_layout(proto)
    _tr, te = split_masks(proto, lay)["primary"]
    v = np.arange(30.0) ** 1.5
    rr = an.ring_removed(v, te, lay.ring)
    assert rr.shape == (12,)
    for r in np.unique(lay.ring[te]):
        assert abs(rr[lay.ring[te] == r].mean()) < 1e-12


def test_decisions_detect_a_clear_ris_structure(proto: dict) -> None:
    t, truth = _structured(proto, 40, ris_db=4.0, seed="clear")
    g = sremu_rng.generator(5, "trial", "vlab_test/noise")
    ds = an.Dataset(
        truth - 15 + 0.3 * g.standard_normal(30), np.full(30, 0.07), "synthetic", "SIM-sremu"
    )
    res = dc.analyze_dataset(
        t, ds, proto, n_boot=400, rng=sremu_rng.generator(6, "trial", "vlab_test/bt")
    )
    d = res["primary"]["decisions"]
    assert d["C2"]["holds"], d["C2"]
    assert d["C3"]["holds"], d["C3"]
    assert d["C2"]["spearman_rho"] > 0.8
    assert res["secondary"]["delta_rmse"]["M0-M2"] > 1.0
    assert res["secondary"]["models"]["M2"]["ess"] >= 1.0


def test_decisions_null_when_the_map_has_no_ris_structure(proto: dict) -> None:
    t, _ = _structured(proto, 40, ris_db=4.0, seed="null")
    lay = cell_layout(proto)
    g = sremu_rng.generator(7, "trial", "vlab_test/noise")
    y = -15 - 20 * np.log10(lay.r_m) + 0.3 * g.standard_normal(30)
    ds = an.Dataset(y, np.full(30, 0.07), "synthetic LOS", "SIM-sremu")
    res = dc.analyze_dataset(
        t, ds, proto, n_boot=400, rng=sremu_rng.generator(8, "trial", "vlab_test/bt")
    )
    d = res["primary"]["decisions"]
    assert not d["C2"]["holds"]
    assert d["C2"]["median_delta_rmse_db"] < 0
    assert d["C1"]["median_rmse_m0n_db"] < 1.0


def test_c1_logic(proto: dict) -> None:
    t, truth = _structured(proto, 10, ris_db=1.0, seed="c1")
    ds = an.Dataset(truth - 15, np.full(30, 0.07), "synthetic", "SIM-sremu")
    res = dc.analyze_dataset(
        t, ds, proto, n_boot=100, rng=sremu_rng.generator(9, "trial", "vlab_test/bt")
    )
    c1 = res["primary"]["decisions"]["C1"]
    assert c1["bias_ok"] == (c1["median_abs_bias_db"] <= 1.5)
    assert c1["rmse_ok"] == (c1["median_rmse_m2_db"] <= c1["median_rmse_m0n_db"] + 0.5)
    assert c1["holds"] == (c1["bias_ok"] and c1["rmse_ok"])


def test_folds_budget_breakdown(proto: dict) -> None:
    t, truth = _structured(proto, 12, ris_db=3.0, seed="folds")
    t.arrays["sigma_pos_db"] = np.full((3, 30), 0.2)  # type: ignore[index]
    ds = an.Dataset(truth - 15, np.full(30, 0.07), "synthetic", "SIM-sremu")
    folds = an.run_folds(t, ds, proto)
    assert len(folds) == 11
    assert folds["loro_r0.9"]["n_test"] == 6 and folds["lowo_t35"]["n_test"] == 5
    lay = cell_layout(proto)
    tr, te = split_masks(proto, lay)["primary"]
    prim = an.run_split(t, ds, proto, train=tr, test=te, boot_idx=None, layout=lay)
    bud = an.uncertainty_budget(t, ds, prim, proto, test=te)
    assert set(bud["coverage_test_cells"]) == {"sigma_rep_1", "sigma_rep_0.5", "sigma_rep_2"}
    assert bud["median_over_cells_db"]["sigma_pos"] == pytest.approx(0.2)
    br = an.breakdown(t, prim)
    assert br["origin"]["O_R"]["n_draws"] == 6


def test_pinned_cells(proto: dict) -> None:
    y = np.linspace(-30.0, -12.0, 30)
    y[4], y[17] = -10.06, -33.2
    p = an.pinned_cells(y, proto)
    assert p["cells"] == [4, 17] and not any(x["fallback_extreme"] for x in p["detail"])
    y2 = np.linspace(-30.0, -12.0, 30)
    p2 = an.pinned_cells(y2, proto)
    assert p2["cells"] == [0, 29] and all(x["fallback_extreme"] for x in p2["detail"])


# --------------------------------------------------------------------------- decoys


def test_decoy_noise_is_keyed(proto: dict) -> None:
    s1, z1, w1 = dc.decoy_noise(3, proto)
    s2, z2, w2 = dc.decoy_noise(3, proto)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(z1, z2)
    np.testing.assert_array_equal(w1, w2)
    assert np.all((s1 >= 0.045) & (s1 <= 0.22))
    s3, _, _ = dc.decoy_noise(4, proto)
    assert not np.array_equal(s1, s3)


def test_decoy_dataset_construction(proto: dict) -> None:
    truth = np.tile(np.linspace(-50, -40, 30)[:, None], (1, 4))
    truth[:, 2] += 3.0
    ds, c = dc.decoy_dataset(2, truth, proto, sigma_rep_db=0.0)
    s_dig, _z1, z2 = dc.decoy_noise(2, proto)
    np.testing.assert_allclose(ds.y_dbm, truth[:, 2] + c + s_dig * z2)
    assert math.isclose(float(np.mean(truth[:, 2] + c)), -18.318, abs_tol=1e-9)
    ds1, _ = dc.decoy_dataset(2, truth, proto, sigma_rep_db=1.0)
    ds2, _ = dc.decoy_dataset(2, truth, proto, sigma_rep_db=2.0)
    np.testing.assert_allclose(ds2.y_dbm - ds.y_dbm, 2 * (ds1.y_dbm - ds.y_dbm))  # common z
    ds0, _ = dc.decoy_dataset(2, truth, proto, sigma_rep_db=0.0, truth_model="M0")
    np.testing.assert_allclose(ds0.y_dbm - ds.y_dbm, 0.0, atol=1e-12)  # same mean shift
    assert ds.data_class == "SIM-sremu"


def test_synthetic_decoy_uses_a_fresh_keyed_draw(proto: dict) -> None:
    d = dc.synthetic_decoy(1, protocol=proto)
    d2 = dc.synthetic_decoy(1, protocol=proto)
    np.testing.assert_array_equal(d.dataset.y_dbm, d2.dataset.y_dbm)
    assert d.truth_draw is not None and d.truth_draw.name == "vlab/1"
    assert d.truth_model == "M2" and d.sigma_rep_db == 1.0
    assert np.all(np.isfinite(d.dataset.y_dbm))
    assert math.isclose(float(np.mean(d.truth_pred_db[:, 2] + d.offset_dbm)), -18.318, abs_tol=1e-9)


def _row(i: int, tm: str, *, c2: bool, sp: bool) -> dict:
    return {
        "decoy": i,
        "truth_model": tm,
        "sigma_rep_db": 1.0,
        "C1": False,
        "C2": c2,
        "C3": False,
        "c2_median_ok": c2,
        "c2_frac_ok": c2,
        "spearman_ok": sp,
        "c2_median_delta_db": 0.0,
        "c2_frac_ci_excl0": 0.0,
        "spearman_rho": 0.0,
        "c3_median_delta_db": 0.0,
        "median_rmse_m2_db": 1.0,
        "median_rmse_m0_db": 1.0,
        "secondary_delta_m0_m2_db": 0.0,
        "secondary_rmse_m2_db": 1.0,
    }


def test_power_block_flags_a_nonspecific_spearman_condition(proto: dict) -> None:
    """Reviewer addition: when the Spearman sub-condition holds at least as often with the
    RIS-free truth as with the M2 truth, the generated protocol block says so, and it states
    that C1 does not test absolute power."""
    rows = [_row(i, "M2", c2=False, sp=i < 2) for i in range(10)]
    rows += [_row(i, "M0", c2=False, sp=i < 5) for i in range(10)]
    s = dc.summarize_power(rows, proto)
    meta = {"content_sha256": "0" * 64, "prediction_spec_sha256": "1" * 64, "K": 10}
    run_info = {
        "t_utc": "t",
        "run_dir": "r",
        "git_commit": "c",
        "git_dirty": False,
        "truth_sha256": "2" * 64,
        "n_boot": 10,
    }
    _pr, md = dc._protocol_result(s, meta, run_info)
    assert "holds for 0.500 of the decoys whose truth is M0" in md
    assert "does not indicate RIS-specific structure" in md
    assert "does not test the absolute received power" in md
    rows2 = [_row(i, "M2", c2=False, sp=i < 6) for i in range(10)]
    rows2 += [_row(i, "M0", c2=False, sp=i < 1) for i in range(10)]
    _pr2, md2 = dc._protocol_result(dc.summarize_power(rows2, proto), meta, run_info)
    assert "does not indicate RIS-specific structure" not in md2


def test_report_states_the_real_lock_state(proto: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reviewer addition: report.md must not call an unlocked protocol 'locked'."""
    from validation.ants_fig4b import protocol_lock as pl
    from validation.ants_fig4b import run

    t, truth = _structured(proto, 8, ris_db=3.0, seed="report")
    ds = an.Dataset(truth - 15, np.full(30, 0.07), "synthetic", "SIM-sremu")
    res = run.analyze(t, ds, proto, n_boot=50)
    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", "")
    res["lock"] = run.lock_state()
    assert res["lock"]["locked"] is False
    assert "protocol NOT locked" in run._report_md(res, dry=True)
    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", pl.protocol_digest())
    res["lock"] = run.lock_state()
    assert res["lock"]["locked"] is True
    assert "protocol locked" in run._report_md(res, dry=False)


def test_power_summary_and_underpowered_rule(proto: dict) -> None:
    rows = []
    for i in range(10):
        for tm, hit in (("M2", i < 4), ("M0", False)):
            rows.append(
                {
                    "decoy": i,
                    "truth_model": tm,
                    "sigma_rep_db": 1.0,
                    "C1": False,
                    "C2": hit,
                    "C3": False,
                    "c2_median_ok": hit,
                    "c2_frac_ok": hit,
                    "spearman_ok": hit,
                    "c2_median_delta_db": 0.0,
                    "c2_frac_ci_excl0": 0.0,
                    "spearman_rho": 0.0,
                    "c3_median_delta_db": 0.0,
                    "median_rmse_m2_db": 1.0,
                    "median_rmse_m0_db": 1.0,
                    "secondary_delta_m0_m2_db": 0.0,
                    "secondary_rmse_m2_db": 1.0,
                }
            )
    s = dc.summarize_power(rows, proto)
    assert s["c2_detection_rate"] == 0.4 and s["underpowered"] is True
    lo, hi = s["c2_detection_ci95"]
    assert (lo, hi) == pytest.approx(sstats.clopper_pearson(4, 10))
    assert next(iter(s["groups"])).startswith("M2/")
    rows2 = [{**r, "C2": r["truth_model"] == "M2"} for r in rows]
    assert dc.summarize_power(rows2, proto)["underpowered"] is False
