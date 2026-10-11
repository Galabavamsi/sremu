"""V-LAB end-to-end smoke (validation_smoke tier; WP10; protocol v2): a tiny tensor (two
worker processes, equal to the serial result), a two-decoy power analysis, a tiny S-OR tensor
and the run.py decoy dry run, each in its own run directory with a manifest. No measured data
are used.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from validation.ants_fig4b import decoy, predict, protocol_lock, run
from validation.ants_fig4b.protocol import load_protocol, variant_protocol

from sremu.record.manifest import read_manifest

pytestmark = [pytest.mark.validation_smoke, pytest.mark.slow]


def _one(root: Path, suffix: str) -> Path:
    hits = sorted(root.glob(f"*_{suffix}"))
    assert len(hits) == 1, hits
    return hits[0]


def test_vlab_pipeline_smoke(tmp_path: Path) -> None:
    proto = load_protocol()
    serial = predict.compute_tensor(proto, K=4, jobs=1, n_sigma_pos=1, n_alt=4)
    out = tmp_path / "runs"
    assert (
        predict.main(
            [
                "--out",
                str(out),
                "--K",
                "4",
                "--jobs",
                "2",
                "--n-sigma-pos",
                "1",
                "--n-alt",
                "4",
                "--bg-seconds",
                "0",
            ]
        )
        == 0
    )
    tdir = _one(out, "vlab_tensor")
    tpath = tdir / "pred_tensor.npz"
    t = predict.load_cache(tpath, protocol=proto)
    assert t.meta["content_sha256"] == predict.content_sha256(serial["arrays"])  # jobs-invariant
    man = read_manifest(tdir)
    assert man.seed == proto["seed"] and man.scenario_name == "vlab/vlab_tensor"
    meta = json.loads((tdir / "meta.json").read_text(encoding="utf-8"))
    # "locked" mirrors the committed lock constant: False in the blinded phase, True after
    # the lead's lock commit 101ee87.
    assert meta["locked"] is bool(protocol_lock.PREREGISTERED_SHA256)
    assert "SIM-sremu" in meta["data_classes"]

    assert (
        decoy.main(
            [
                "--tensor",
                str(tpath),
                "--out",
                str(out),
                "--n",
                "2",
                "--n-boot",
                "50",
                "--bg-seconds",
                "0",
            ]
        )
        == 0
    )
    pdir = _one(out, "vlab_power")
    with (pdir / "decoys.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 * 2 * 3  # decoys x truth models x sigma_rep
    assert all("snf_C2" in r for r in rows)  # S-NF analysed on every decoy (information)
    summ = json.loads((pdir / "summary.json").read_text(encoding="utf-8"))
    assert 0.0 <= summ["c2_detection_rate"] <= 1.0
    assert summ["protocol_result"]["tensor_content_sha256"] == t.meta["content_sha256"]
    assert "POWER" not in summ["protocol_result_md"]  # the block body only

    assert (
        predict.main(
            [
                "--out",
                str(out),
                "--K",
                "3",
                "--n-sigma-pos",
                "0",
                "--n-alt",
                "0",
                "--variant",
                "S-OR",
                "--bg-seconds",
                "0",
            ]
        )
        == 0
    )
    sor_path = _one(out, "vlab_tensor_S-OR") / "pred_tensor.npz"
    tsor = predict.load_cache(sor_path, protocol=variant_protocol(proto, "S-OR"))
    assert tsor.meta["variant"] == "S-OR" and tsor.meta["codes"]["origin"] == ["O_R"]

    assert (
        run.main(
            [
                "--tensor",
                str(tpath),
                "--tensor-sor",
                str(sor_path),
                "--out",
                str(out),
                "--dry-run-decoy",
                "0",
                "--n-boot",
                "100",
            ]
        )
        == 0
    )
    rdir = _one(out, "vlab_dryrun_decoy")
    for f in (
        "F4.csv",
        "F4_delta_rmse.csv",
        "F4.meta.json",
        "table_v.csv",
        "table_v.json",
        "results.json",
        "report.md",
        "manifest.json",
        "meta.json",
    ):
        assert (rdir / f).is_file(), f
    f4 = json.loads((rdir / "F4.meta.json").read_text(encoding="utf-8"))
    assert f4["dry_run_decoy"] is True and "DIG" not in f4["data_classes"]
    res = json.loads((rdir / "results.json").read_text(encoding="utf-8"))
    assert set(res["primary"]["decisions"]) == {"C1", "C2", "C3"}
    assert set(res["sensitivity"]) == {"S1", "S2", "S-NF", "S-OR"}
    assert res["sensitivity"]["S1"]["result"]["n_test"] == 11
    snf = res["sensitivity"]["S-NF"]
    assert (snf["result"]["n_train"], snf["result"]["n_test"]) == (12, 12)
    assert snf["point_vs_patch"]["n_draws"] == 4
    assert res["sensitivity"]["S-OR"]["run"] is True
    assert res["tensor_sor_mismatch_allowed"] is True  # the tiny tensor is not the recorded one
    assert set(res["sensitivity"]["S-OR"]["result"]["decisions"]) == {"C1", "C2", "C3"}
    assert summ["protocol_result"]["specificity"]
    with (rdir / "F4.csv").open(encoding="utf-8") as fh:
        f4rows = list(csv.DictReader(fh))
    assert len(f4rows) == 30
    assert np.isfinite([float(r["pred_m2_prior_median_dbm"]) for r in f4rows]).all()
    assert "DECOY DRY RUN" in (rdir / "report.md").read_text(encoding="utf-8")
