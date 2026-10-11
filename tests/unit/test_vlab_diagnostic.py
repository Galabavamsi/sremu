"""V-LAB power diagnostic (diagnostic, not part of the protocol; PROTOCOL.md section 13.2).

Synthetic arrays and keyed prior draws only; no measured data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from validation.ants_fig4b import diagnostic as dg
from validation.ants_fig4b import prior as pp
from validation.ants_fig4b.predict import TensorCache
from validation.ants_fig4b.protocol import PROTOCOL_DIR, PROTOCOL_JSON, PROTOCOL_MD, load_protocol

from sremu import rng as sremu_rng

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def proto() -> dict:
    return load_protocol()


def _arrays(k: int, seed: str) -> dict[str, np.ndarray]:
    g = sremu_rng.generator(1, "trial", f"vlab_test/diag/{seed}")
    return {
        "pred_db": g.normal(size=(k, 30, 4)),
        "radius_m": g.uniform(1.88, 2.05, k),
        "centre_deg": g.uniform(19.5, 25.5, k),
        "span_deg": g.uniform(40, 52, k),
        "pitch_idx": g.integers(0, 2, k),
        "antenna_idx": g.integers(0, 2, k),
        "gamma_mag": g.uniform(0.6, 1.0, (k, 2)),
        "gamma_phase_offset_deg": g.uniform(-15, 15, (k, 2)),
        "sigma_pos_db": np.ones((3, 30)),
    }


def test_subset_masks_condition_on_the_truth() -> None:
    t, tr = _arrays(400, "t"), _arrays(5, "tr")
    for i in range(5):
        m = dg.SUBSET_FACTS["R_pm10mm"][1](t, tr, i)
        np.testing.assert_array_equal(m, np.abs(t["radius_m"] - tr["radius_m"][i]) <= 0.01)
        m = dg.SUBSET_FACTS["pitch"][1](t, tr, i)
        assert np.all(t["pitch_idx"][m] == tr["pitch_idx"][i])
        m = dg.SUBSET_FACTS["arc_angles"][1](t, tr, i)
        assert np.all(np.abs(t["centre_deg"][m] - tr["centre_deg"][i]) <= 0.5)
        assert np.all(np.abs(t["span_deg"][m] - tr["span_deg"][i]) <= 1.0)
        m = dg.SUBSET_FACTS["gamma"][1](t, tr, i)
        assert np.all(np.abs(t["gamma_mag"][m] - tr["gamma_mag"][i]) <= 0.1)
        assert np.all(np.abs(t["gamma_phase_offset_deg"][m] - tr["gamma_phase_offset_deg"][i]) <= 5)


def test_subset_tensor_slices_per_draw_arrays() -> None:
    a = _arrays(10, "s")
    t = TensorCache(arrays=a, meta={"codes": {}}, path=Path("x.npz"))
    m = np.arange(10) % 3 == 0
    s = dg.subset_tensor(t, m)
    assert s.K == 4 and "sigma_pos_db" not in s.arrays
    np.testing.assert_array_equal(s.arrays["radius_m"], a["radius_m"][m])


def test_restricted_protocols(proto: dict) -> None:
    p5 = dg.restricted_protocol(proto, "points_pm5mm", K=7, n_decoys=3)
    assert p5["prior"]["point_error"] == {
        "xy_half_width_m": 0.005,
        "plane_height_half_width_m": 0.005,
    }
    assert p5["prior"]["geometry"] == proto["prior"]["geometry"]
    assert p5["rng"]["prior_draw"]["name"] == "vlab_diag_points_pm5mm/{k}"
    assert p5["rng"]["decoy_truth"]["name"] == "vlab_diag_points_pm5mm/{i}"
    assert p5["primary_analysis"]["K"] == 7 and p5["power_analysis"]["n_decoys"] == 3
    assert (
        "power_analysis_result" not in p5
        and proto["prior"]["point_error"]["xy_half_width_m"] == 0.02
    )
    d = pp.draw_prior(p5, purpose="trial", name="vlab_diag_points_pm5mm/0")
    assert np.max(np.abs(d.point_dxy_m)) <= 0.005 and abs(d.plane_dz_m) <= 0.005
    rp = dg.restricted_protocol(proto, "R_pm2p5mm_points_pm5mm", K=7, n_decoys=3)
    assert rp["prior"]["geometry"]["radius_m"] == pytest.approx([1.9475, 1.9525])
    gk = dg.restricted_protocol(proto, "geometry_known", K=7, n_decoys=3)
    d = pp.draw_prior(gk, purpose="trial", name="vlab_diag_geometry_known/0")
    assert (d.radius_m, d.span_deg, d.centre_deg, d.pitch) == (1.95, 46.0, 22.5, "square_42p8mm")
    assert np.all(d.point_dxy_m == 0.0) and d.plane_dz_m == 0.0


def _rows(fact: str, n: int, k2: int, k0: int) -> list[dict]:
    out = []
    for tm, k in (("M2", k2), ("M0", k0)):
        for i in range(n):
            out.append(
                {
                    "fact": fact,
                    "decoy": i,
                    "truth_model": tm,
                    "sigma_rep_db": 1.0,
                    "n_draws": 50,
                    "C1": False,
                    "C2": i < k,
                    "C3": False,
                    "c2_median_ok": i < k,
                    "c2_frac_ok": i < k,
                    "spearman_ok": True,
                    "c2_median_delta_db": 0.1 * i,
                }
            )
    return out


def test_summary_and_block(proto: dict, tmp_path: Path) -> None:
    rows = _rows("R_pm10mm", 10, 2, 0) + _rows("points_pm1mm", 10, 4, 1)
    rows.append({"fact": "R_pm10mm", "decoy": 99, "truth_model": "M2", "n_draws": 3})
    s = dg.summarize(rows, ["R_pm10mm", "points_pm1mm"])
    assert s["R_pm10mm"]["M2"]["C2_rate"] == 0.2 and s["R_pm10mm"]["M2"]["n_skipped"] == 1
    assert s["points_pm1mm"]["M0"]["C2_rate"] == 0.1
    base = {
        tm: {
            "C2_rate": 0.0,
            "C2_ci95": [0.0, 0.01],
            "median_delta_rmse_db": -0.5,
            "median_n_draws": 2000.0,
        }
        for tm in ("M2", "M0")
    }
    run = {
        "t_utc": "t",
        "run_dir": "r",
        "git_commit": "c",
        "git_dirty": True,
        "settings": {
            "sigma_rep_db": 1.0,
            "n_boot": 10,
            "n_decoys_subset": 10,
            "K_restricted": 5,
            "n_decoys_restricted": 10,
            "min_draws": 20,
        },
    }
    rec, md = dg.protocol_block(s, base, run)
    assert rec["label"] == dg.LABEL and rec["best_single_fact"] == "points_pm1mm"
    assert dg.LABEL in md and "No single fact reaches the 0.5 power threshold." in md
    d = tmp_path / "proto"
    d.mkdir()
    for f in (PROTOCOL_MD, PROTOCOL_JSON):
        shutil.copyfile(PROTOCOL_DIR / f, d / f)
    dg.write_protocol_block(rec, md, d)
    assert (
        "best_single_fact"
        in json.loads((d / PROTOCOL_JSON).read_text(encoding="utf-8"))[
            "diagnostic_not_part_of_protocol"
        ]
    )
    text = (d / PROTOCOL_MD).read_text(encoding="utf-8")
    assert md in text and text.count(dg.BLOCK_BEGIN) == 1
    assert dg.rows_for_csv(rows)[-1]["C2"] == ""


def test_ties_at_zero_and_row_roundtrip(tmp_path: Path) -> None:
    rows = _rows("R_pm10mm", 6, 0, 0) + _rows("points_pm1mm", 6, 0, 0)
    for r in rows:
        if r["fact"] == "points_pm1mm":
            r["c2_median_delta_db"] += 1.0
    rows += _rows("geometry_known", 6, 4, 0)
    s = dg.summarize(rows, ["R_pm10mm", "points_pm1mm", "geometry_known"])
    base = {
        tm: {
            "C2_rate": 0.0,
            "C2_ci95": [0.0, 0.1],
            "median_delta_rmse_db": -1.0,
            "median_n_draws": 9,
        }
        for tm in ("M2", "M0")
    }
    settings = {
        "sigma_rep_db": 1.0,
        "n_boot": 1,
        "n_decoys_subset": 6,
        "K_restricted": 5,
        "n_decoys_restricted": 6,
        "min_draws": 1,
        "merged_runs": ["prev"],
    }
    run = {
        "t_utc": "t",
        "run_dir": "r",
        "git_commit": "c",
        "git_dirty": False,
        "settings": settings,
    }
    rec, md = dg.protocol_block(s, base, run)
    assert rec["best_single_fact"] is None
    assert rec["closest_single_fact_by_delta_rmse"] == "points_pm1mm"
    assert "No single fact raises the C2 detection rate above 0" in md
    assert "Combinations that reach 0.5" in md and "(rows of `prev` merged)" in md
    from validation.ants_fig4b import runlog

    class _Rec:
        run_dir = tmp_path

    runlog.RunRecorder.write_csv(_Rec(), "rows.csv", dg.rows_for_csv(rows))  # type: ignore[arg-type]
    back = dg.read_rows(tmp_path / "rows.csv")
    assert dg.summarize(back, ["geometry_known"]) == dg.summarize(rows, ["geometry_known"])
