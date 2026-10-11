"""WP10 validation studies V-TANG, V-CFO, V-WCL and V-FM end to end at smoke sizes
(validation_smoke tier, DESIGN.md sections 2.6, 10.8, 10.10-10.12).

Each test runs the study script's ``study()`` into ``tmp_path`` and checks the run
directory contract (manifest that validates, ``meta.json`` with data classes and the
sha256 of every output, the CSV tables) plus the acceptance rows that are meaningful at
smoke sizes. The full studies (10^4 movement periods, 200 trials per point, ...) are run by
``python validation/<study>/v<study>.py``; their numbers are in ``validation/*/report.md``.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from sremu.record.manifest import read_manifest

pytestmark = [pytest.mark.validation_smoke, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]


def _load(sub: str, name: str) -> ModuleType:
    """Import a study script by its top-level name (as the sweep runner resolves it)."""
    for p in (REPO / "validation", REPO / "validation" / sub):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module(name)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rows(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _run_dir_contract(run_dir: Path, csvs: list[str]) -> dict[str, Any]:
    man = read_manifest(run_dir)  # schema-validated manifest (section 3.6)
    assert (run_dir / "scenario.resolved.yaml").is_file()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["sremu_commit"] == man.git_commit
    assert meta["sremu_dirty"] == man.git_dirty
    assert meta["data_classes"]
    assert {i.data_class for i in man.inputs} <= {
        "MEAS",
        "DIG",
        "REPORTED",
        "ASSUMED",
        "SIM-sremu",
        "SIM-legacy-v1",
        "SIM-legacy-v2",
        "ANALYTIC",
        "MATLAB",
    }
    for rel, sha in meta["outputs"].items():
        assert _sha(run_dir / rel) == sha, rel
    for name in csvs:
        assert name in meta["outputs"], name
        assert _rows(run_dir / name), name
    return meta


def test_vtang_smoke(tmp_path: Path) -> None:
    vt = _load("tang", "vtang")
    rd = vt.study(root=tmp_path)
    meta = _run_dir_contract(rd, ["vtang_points.csv"])
    acc = meta["acceptance"]
    assert acc["V-TANG-1_max_abs_le_0.05dB"] is True
    assert acc["V-TANG-2_rmse_regression_1.05_pm_0.05dB"] is True
    assert acc["cell_model_auto_resolved"] == ["point"]
    assert len(_rows(rd / "vtang_points.csv")) == 36
    assert {"DIG", "REPORTED"} <= {i["data_class"] for i in meta["inputs"]}
    num = meta["numbers"]
    assert "n = 36" in num["wording"]
    # "within X dB" is a bound: X is the maximum rounded up to 0.01 dB
    assert num["vtang1_bound_for_wording_db"] >= num["vtang1_max_abs_db"]
    assert num["vtang1_bound_for_wording_db"] - num["vtang1_max_abs_db"] < 0.01
    assert f"within {num['vtang1_bound_for_wording_db']:.2f} dB" in num["wording"]
    # the wording's bias is the exact mean over the two-decimal digitized values, rounded
    # half-even (on this data set it is the tie 197/200 dB, so float rounding would decide
    # it by representation noise)
    pts = _rows(rd / "vtang_points.csv")
    exact = sum(
        (Fraction(r["measured_dbm"]) - Fraction(r["published_eq9_dbm"]) for r in pts),
        Fraction(0),
    ) / len(pts)
    assert num["vtang2_measured_minus_published_bias_exact"] == str(exact)
    q = round(exact * 100)  # Fraction rounding: exact, ties to even
    bias_txt = f"{'+' if q >= 0 else '-'}{abs(q) // 100}.{abs(q) % 100:02d}"
    assert f"lie {bias_txt} dB" in num["wording"]
    text = vt.report(rd)  # report.md renders from meta.json (not written for tmp roots)
    assert "V-TANG-1" in text and "not** presented as sremu accuracy" in text
    assert f"> {num['wording']}" in text
    assert not (rd.parent / "report.md").exists()


def test_vcfo_smoke(tmp_path: Path) -> None:
    vc = _load("cfo", "vcfo")
    # the CRB of DESIGN 10.10 [DERIVED] (N = 2000, 5880 S/s), recomputed
    for snr, ref in ((0.0, 0.0256), (10.0, 0.0081), (20.0, 0.0026)):
        assert round(vc.crb_std_hz(snr), 4) == ref
    rd = vc.study(jobs=1, fast=True, root=tmp_path)
    meta = _run_dir_contract(
        rd, ["vcfo_sweep_noise_free.csv", "vcfo4_points.csv", "vcfo4_rmse_vs_crb.csv"]
    )
    acc = meta["acceptance"]
    assert acc["V-CFO-1_engine_ls_and_lag1_le_1e-6_hz"] is True
    assert acc["V-CFO-2_legacy_slope_1.0691_pm_0.0005"] is True
    nf = meta["numbers"]["noise_free"]
    assert abs(nf["engine_ls"]["slope"] - 1.0) < 1e-9
    assert abs(nf["dig_vs_nominal"]["slope"] - 1.0681) < 5e-4  # the DIG re-analysis
    assert len(_rows(rd / "vcfo4_points.csv")) == 21
    for snr in ("10", "20"):  # 28 trials per SNR at smoke size: a loose sanity band only
        d = meta["numbers"]["vcfo4"]["by_snr"][snr]
        assert 0.3 < d["periodogram_rmse_over_crb_pooled"] < 3.0
        assert abs(d["snr_hat_db_mean"] - float(snr)) < 1.0
        # the lag-1 estimator follows its small-error theory, far above the CRB
        assert 0.5 < d["lag1_rmse_over_theory_pooled"] < 2.0
        assert d["lag1_rmse_over_crb_pooled"] > 10.0
    assert all(r["crb_matches_design_4dp"] for r in meta["numbers"]["crb"])
    # D24: per-series paper use; the DIG points of the earlier simulator are internal only
    assert "internal only" in meta["paper_use"]["dig_*"]
    nc = meta["numbers"]["noise_check"]
    assert abs(nc["variance_over_kT0Ffs"] - 1.0) < 5 * nc["variance_rel_se"] + 0.02
    text = vc.report(rd)
    assert "V-CFO-4" in text
    # D24: no before/after framing in the report's title; internal series marked as such
    assert "before/after" not in text.splitlines()[0].lower()
    assert "internal only" in text


def test_vcfo_lag1_theory_matches_simulation() -> None:
    """The [DERIVED] lag-1 small-error standard deviation against a direct Monte Carlo of a
    complex tone in complex AWGN (no engine), N = 2000, 20 dB."""
    vc = _load("cfo", "vcfo")
    rng = np.random.default_rng(12345)
    n, snr = 2000, 100.0
    k = np.arange(n)
    est = []
    for _ in range(400):
        w = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * math.sqrt(1 / (2 * snr))
        y = np.exp(2j * np.pi * 0.01 * k) + w
        est.append(np.angle(np.sum(y[1:] * np.conj(y[:-1]))))
    sd_hz = float(np.std(est, ddof=1)) * float(vc.FS) / (2 * np.pi)
    assert sd_hz == pytest.approx(vc.lag1_std_hz(20.0, n), rel=0.12)


def test_vwcl_oracle_matches_engine_k1() -> None:
    """The independent corridor closed form equals sremu's K1 cascade and LOS (1e-12)."""
    vw = _load("wcl", "vwcl")
    for h in (1.0, 6.0):
        sc = vw.corridor_scenario(h, duration_s="10")
        model = vw.build_model(sc)
        for t in (Fraction(0), Fraction(37, 10), Fraction(9)):
            v = model.csi_view(model.snapshot(t), vw.LINK)
            x = model.node_position("mu", t)[0]
            h0, s = vw.oracle_channel(np.array([x]), h, vw.D0_M)
            assert abs(v.h_direct - h0[0]) <= 1e-12 * abs(h0[0])
            z = v.cascade[vw.PANEL]
            assert np.all(z[0] == 0)
            assert np.max(np.abs(z[1] - s[0])) <= 1e-12 * np.max(np.abs(s[0]))
            from sremu.physics.optimize import exact_alphabet_optimum

            a = vw.exact_binary(h0[0], s[0])[0]
            b = exact_alphabet_optimum(v.h_direct, z)[0]
            assert abs(a - b) <= 1e-12 * b


def test_vwcl_smoke(tmp_path: Path) -> None:
    vw = _load("wcl", "vwcl")
    rd = vw.study(["w1", "w3", "w5"], jobs=1, profile="smoke", root=tmp_path)
    meta = _run_dir_contract(
        rd,
        [
            "w1_position_histogram.csv",
            "w3_coefficient.csv",
            "w3_conditional.csv",
            "w3_iq.csv",
            "w5.csv",
            "w6.csv",
        ],
    )
    w1 = meta["numbers"]["W1"]
    assert w1["teeth_pass"] is True and w1["teeth_eq2_ratio_max_rel_dev"] <= 1e-9
    assert "internal only" in meta["paper_use"]["W1 teeth_eq2_*"]  # D24
    assert w1["max_jump_pass"] is True
    iq = _rows(rd / "w3_iq.csv")
    assert iq and all(r["config_mismatches"] == "0" for r in iq)
    assert all(abs(float(r["delta_iq_vs_coef_mid_db"])) < 0.01 for r in iq)
    for r in _rows(rd / "w5.csv"):
        assert r["controller"] == "ExactOptimum"
        assert abs(float(r["primary_delta_db"])) <= 0.02
        assert float(r["decisions_identical_fraction"]) == 1.0
        assert r["lateness_max"] == "0"
    for r in _rows(rd / "w6_run_batch.csv"):  # run_batch == the piecewise coefficient path
        assert abs(float(r["batch_vs_piecewise_rel"])) <= 1e-12
    w3 = _rows(rd / "w3_coefficient.csv")
    assert w3 and all(math.isfinite(float(r["analytic_mean_power_dbm"])) for r in w3)
    # same-trajectory W3: the independent replica reproduces the emulator's batch path
    for r in _rows(rd / "w3_conditional.csv"):
        assert float(r["oracle_replica_max_rel_diff"]) <= 1e-9
        assert float(r["los_replica_max_rel_diff"]) <= 1e-9
        assert float(r["forms_differ_db"]) >= 0.0  # mean power >= power of the mean field
    # W6: midpoint references are part of the table and of the Doppler-isolated verdict
    grids = {r["grid"] for r in _rows(rd / "w6.csv")}
    assert {"1ms", "1ms_midpoint_oracle", "5ms_midpoint"} <= grids
    assert "w6_doppler_isolated_pass" in meta["acceptance"]
    # re-aggregation from the saved per-point outputs reproduces the tables and acceptance
    before = {n: (rd / n).read_bytes() for n in ("w3_coefficient.csv", "w5.csv", "w6.csv")}
    acc0 = dict(meta["acceptance"])
    vw.reaggregate(rd)
    meta2 = _run_dir_contract(rd, ["w3_coefficient.csv", "w5.csv"])
    assert {n: (rd / n).read_bytes() for n in before} == before
    assert meta2["acceptance"] == acc0 and meta2["reaggregated_utc"]
    text = vw.report(rd)
    assert "W5" in text and "documented deviation" in text.lower()
    # D24: the internal kernel-ratio teeth test is marked internal; no 'legacy' heading
    assert not any(ln.startswith("#") and "legacy" in ln.lower() for ln in text.splitlines())
    assert "internal, not for the paper" in text


def test_vwcl_resume(tmp_path: Path) -> None:
    """``--resume`` continues a run directory: completed points are skipped, the start-of-run
    patch and manifest are kept under new names and recorded in meta.json."""
    vw = _load("wcl", "vwcl")
    rd = vw.study(["w5"], jobs=1, profile="smoke", root=tmp_path)
    had_patch = (rd / "dirty.patch").exists()
    rd2 = vw.study(["w5"], jobs=1, profile="smoke", resume_dir=rd)
    assert rd2 == rd
    meta = _run_dir_contract(rd, ["w5.csv"])
    assert meta["sweep"]["n_resumed"] == meta["sweep"]["n_points"] > 0
    names = {r["file"] for r in meta["resumed_from"]}
    assert "manifest.start1.json" in names
    assert ("dirty.start1.patch" in names) == had_patch


def test_vfm_smoke(tmp_path: Path) -> None:
    vf = _load("fm", "vfm")
    rd = vf.study(root=tmp_path)
    meta = _run_dir_contract(
        rd, ["vfm_direct.csv", "vfm_ris.csv", "vfm_ed_map.csv", "vfm1_regression.csv"]
    )
    acc = meta["acceptance"]
    assert acc["V-FM-1_ris_with_ground_pm_0.1dB_d2_1_2km"] is True
    assert acc["PH-10_two_ray_pec_139.10_122.63_pm_0.01dB"] is True
    assert meta["label"] == "model-based dimensioning, not validated"
    ed = _rows(rd / "vfm_ed_map.csv")
    exp = [r for r in ed if "experimental" in r["baseline"]]
    assert exp  # the P.526 row is present and labelled
    assert {r["baseline"] for r in ed} == {
        "two_ray_knife_edge",
        "p526_plus_knife_edge_experimental",
        "four_ray_knife_edge_sensitivity",
        "free_space_knife_edge_reference",
    }
    assert {r["ground_on_ris_hops"] for r in ed} == {"true", "false"}
    assert {float(r["d2_m"]) for r in ed} == {500.0, 1000.0, 2000.0}
    assert {float(r["h_r_m"]) for r in ed} == {1.5, 10.0}
    # A_req is the coherent-aperture scaling A0 |h_dir| / |h_RIS(A0)|, row by row
    for r in ed[:: max(1, len(ed) // 50)]:
        a_req = vf.A0 * 10 ** ((float(r["pl_ris_a0_db"]) - float(r["pl_direct_db"])) / 20)
        assert float(r["a_req_equal_power_m2"]) == pytest.approx(a_req, rel=1e-12)
    text = vf.report(rd)
    assert "not validated" in text
    assert "Validity of the A_req scaling" in text


def test_vfm_four_ray_geometry_consistent_with_engine_edge() -> None:
    """The [ENG] four-ray sensitivity row uses the same ridge as sremu's knife edge: its
    TX -> RX ray has the engine's LOS nu (the ridge height is solved on the engine's own
    crossing), every image ray is more obstructed, and the sum is the sum of its rays."""
    vf = _load("fm", "vfm")
    for h_r in (1.5, 10.0):
        for nu in (0.0, 1.0, 3.0):
            _spec, z = vf.ridge_edge(h_r, nu, 2000.0)
            tot, rays = vf.four_ray_knife_edge(h_r, z, 2000.0, detail=True)
            by = {(a, b): (n_, c) for a, b, n_, c in rays}
            assert by[("tx", "rx")][0] == pytest.approx(nu, abs=1e-4)
            for key in (("tx", "rx_image"), ("tx_image", "rx"), ("tx_image", "rx_image")):
                assert by[key][0] > by[("tx", "rx")][0]
            assert tot == pytest.approx(sum(c for *_x, c in rays), rel=1e-12)
