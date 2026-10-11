"""Tools of the pinned-commit regeneration (experiments/regenerate_paper_data.sh): the
comparison rules of ``experiments/compare_regen.py`` for the E-B arc, validation and latency
scopes, the E-B arc and E-F latency collectors of ``experiments/finalize.py`` and the V-LAB
integrity check ``experiments/vlab_check.py``. Synthetic trees in ``tmp_path``; no git
process, no experiment run.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
from experiments import compare_regen as cr
from experiments import finalize, install_regen, vlab_check
from experiments.common import code_hashes as common_code_hashes

REPO = Path(__file__).resolve().parents[2]
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write_json(p: Path, obj: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _write_csv(p: Path, rows: list[dict[str, object]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# =========================================================================== compare_regen


def test_tolerance_rules() -> None:
    tol = cr.Tol(1e-9, 1e-12, 1e-9, 1e-7)
    assert tol.ok("evm_db", -20.0, -20.0 + 5e-10) and not tol.ok("evm_db", -20.0, -20.0 + 2e-9)
    # V-CFO residuals near zero: absolute 1e-7 Hz, not relative
    assert tol.ok("max_abs_err_hz", 3.1e-10, 2.0e-10)
    assert not tol.ok("max_abs_err_hz", 3.1e-10, 3.0e-7)
    # without atol_hz (every other scope) the same residual change fails
    assert not tol.with_hz(None).ok("max_abs_err_hz", 3.1e-10, 2.0e-10)
    assert tol.ok("tx_dbm", -78.0, -78.0)


def test_index_csv_keyed_by_point_and_run_fields_skipped(tmp_path: Path) -> None:
    ref = [
        {
            "point_id": "p00000",
            "status": "ok",
            "wall_s": "1.0",
            "result.wall_s": "0.5",
            "result.bit_errors": "3",
            "result.evm_db": "-20.0",
        },
        {
            "point_id": "p00001",
            "status": "ok",
            "wall_s": "2.0",
            "result.wall_s": "0.7",
            "result.bit_errors": "0",
            "result.evm_db": "-25.0",
        },
    ]
    new = [dict(r) for r in reversed(ref)]  # other order, other times
    for r in new:
        r["wall_s"], r["result.wall_s"] = "9.9", "9.9"
    _write_csv(tmp_path / "ref.csv", ref)
    _write_csv(tmp_path / "new.csv", new)
    skip = cr.HOST_FIELDS | cr.RUN_FIELDS
    rel = "E-B-arc/sweeps/eb1/index.csv"
    fr = cr.compare_csv(
        rel, tmp_path / "new.csv", tmp_path / "ref.csv", cr.Tol(1e-9, 1e-12, 1e-9), False, skip
    )
    assert not fr.failed(False) and fr.rows_compared == 2
    assert {"wall_s", "result.wall_s"} <= fr.not_compared
    new[0]["result.bit_errors"] = "4"
    _write_csv(tmp_path / "new.csv", new)
    fr = cr.compare_csv(
        rel, tmp_path / "new.csv", tmp_path / "ref.csv", cr.Tol(1e-9, 1e-12, 1e-9), False, skip
    )
    assert fr.failed(False) and "result.bit_errors" in fr.exact_mismatches[0]


def test_arc_meta_and_root_latency_note(tmp_path: Path) -> None:
    ref = {
        "experiment": "E-B-arc",
        "data_classes": ["SIM-sremu"],
        "assumptions": ["a"],
        "run": {"run_id": "x"},
        "host": {"instance_type": "c7i.16xlarge"},
        "sremu": {"commit": "5be649a"},
        "files": {},
        "written_utc": "t",
    }
    new = {
        **ref,
        "run": {"run_id": "y"},
        "host": {"instance_type": "c7i.48xlarge"},
        "sremu": {"commit": COMMIT},
    }
    _write_json(tmp_path / "r.json", ref)
    _write_json(tmp_path / "n.json", new)
    tol = cr.Tol(1e-9, 1e-12, 1e-9)
    fr = cr.compare_meta("E-B-arc/meta.json", tmp_path / "n.json", tmp_path / "r.json", tol, False)
    assert not fr.failed(False)
    # root meta.json: the E-F latency note and directory are reported apart, not compared
    ref = {
        "experiment": "WP11-results",
        "assumptions": [finalize.EXCLUDED["E-F/latency"], "b"],
        "experiments": ["E-A"],
    }
    new = {
        "experiment": "WP11-results",
        "assumptions": [finalize.LATENCY_NOTE, "b"],
        "experiments": ["E-A", "E-F/latency"],
    }
    _write_json(tmp_path / "r.json", ref)
    _write_json(tmp_path / "n.json", new)
    assert not cr.compare_meta(
        "meta.json", tmp_path / "n.json", tmp_path / "r.json", tol, False
    ).failed(False)
    assert cr.in_scope("E-B-arc/E-B1/eb1_points.csv") and not cr.in_scope("E-B-arc/host/lscpu.txt")


def test_validation_runs_compared_without_provenance_files(tmp_path: Path) -> None:
    repo = tmp_path / "sremu"
    ref = repo / "validation" / "tang" / "results" / "20261010T232551Z_vtang"
    meta = {
        "study": "V-TANG",
        "sremu_commit": "9dca79f",
        "sremu_dirty": True,
        "wall_s": 13.0,
        "acceptance": {"V-TANG-1": True},
        "numbers": {"vtang1_max_abs_db": 0.012, "wording": "lie +0.99 dB"},
    }
    _write_json(ref / "meta.json", meta)
    (ref / "dirty.patch").write_text("patch", encoding="utf-8")
    _write_json(ref / "manifest.json", {"git_commit": "9dca79f"})
    _write_csv(ref / "vtang_points.csv", [{"panel": "a", "engine_dbm": "-60.5"}])
    new = tmp_path / "job" / "validation" / "tang" / "20261012T101010Z_vtang"
    _write_json(
        new / "meta.json",
        {
            **meta,
            "sremu_commit": COMMIT,
            "sremu_dirty": False,
            "wall_s": 9.0,
            "git_state_source": "env",
            "numbers": {"vtang1_max_abs_db": 0.012, "wording": "lie +0.98 dB"},  # rendered text
        },
    )
    _write_json(new / "manifest.json", {"git_commit": COMMIT})
    _write_csv(new / "vtang_points.csv", [{"panel": "a", "engine_dbm": "-60.5"}])
    runs = cr.find_validation_runs(tmp_path / "job" / "validation")
    assert runs == {"V-TANG": new}
    refs = {
        "V-TANG": "validation/tang/results/20261010T232551Z_vtang",
        "V-CFO": "validation/cfo/results/none",
        "V-FM": "validation/fm/results/none",
    }
    files, missing, extra = cr.compare_validation(
        runs, refs, repo, cr.Tol(1e-9, 1e-12, 1e-9), False
    )
    assert set(files) == {"V-TANG/meta.json", "V-TANG/vtang_points.csv"}
    assert not any(f.failed(False) for f in files.values())
    assert "wording" in files["V-TANG/meta.json"].not_compared
    assert missing == ["V-CFO/ (no regenerated run)", "V-FM/ (no regenerated run)"] and not extra
    prov = cr.provenance(tmp_path / "empty", runs)
    assert prov["per_experiment"]["V-TANG"]["git_commit"] == COMMIT


def test_validation_refs_read_from_gen_results(tmp_path: Path) -> None:
    (tmp_path / "gen_results.py").write_text(
        'TANG_RUN = "validation/tang/results/X_vtang"\nCFO_RUN = "validation/cfo/results/Y"\n',
        encoding="utf-8",
    )
    refs, _src = cr.validation_refs(tmp_path)
    assert refs["V-TANG"].endswith("X_vtang") and refs["V-CFO"].endswith("/Y")
    assert refs["V-FM"] == cr.VALIDATION_FALLBACK["V-FM"]
    assert cr.validation_refs(tmp_path / "absent")[1].startswith("fallback")


def test_git_head_from_files(tmp_path: Path) -> None:
    g = tmp_path / "r" / ".git"
    (g / "refs" / "heads").mkdir(parents=True)
    (g / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    (g / "refs" / "heads" / "master").write_text(COMMIT + "\n", encoding="utf-8")
    assert cr.git_head(tmp_path / "r") == COMMIT
    (g / "refs" / "heads" / "master").unlink()
    (g / "packed-refs").write_text(f"# pack-refs\n{COMMIT} refs/heads/master\n", encoding="utf-8")
    assert cr.git_head(tmp_path / "r") == COMMIT
    assert cr.git_head(tmp_path / "none") is None


def test_code_hashes_match_the_experiment_record() -> None:
    a, b = cr.code_hashes(REPO), common_code_hashes()
    assert a["src_sremu_tree_sha256"] == b["src_sremu_tree_sha256"]
    assert a["files"] == {k: v for k, v in b["files"].items() if not k.startswith("paper/")}


def test_latency_check(tmp_path: Path) -> None:
    d = tmp_path / "E-F" / "latency"
    _write_json(d / "meta.json", {"data_classes": ["MEAS"]})
    _write_json(d / "manifest.json", {"cpu_model": "Xeon", "aws_instance": {"type": "c7i"}})
    res = cr.latency_check(tmp_path)
    assert res is not None and res["ok"] and res["values_compared"] is False
    _write_json(d / "meta.json", {"data_classes": ["SIM-sremu"]})
    assert not cr.latency_check(tmp_path)["ok"]
    assert cr.latency_check(tmp_path / "none") is None


# =========================================================================== finalize


def _fake_arc_run(root: Path, *, other_commit: bool = False) -> Path:
    src = root / "eb_arc"
    gs = {
        "commit": COMMIT,
        "dirty": False,
        "dirty_patch_sha256": None,
        "code": {"files": {finalize.ARC_SCENARIO: "ab" * 32}, "src_sremu_tree_sha256": "cd"},
    }
    _write_json(src / "git_state.json", gs)
    specs = [{"name": n, "spec_sha256": "0" * 64, "n_points": 1} for n in finalize.ARC_SPECS]
    _write_json(
        src / "eb_plan.json",
        {
            "scenario_path": finalize.ARC_SCENARIO,
            "specs": specs,
            "geometry_label": "arc label",
            "argv": ["eb.py"],
        },
    )
    _write_json(
        src / "eb_summary.json",
        {
            "ok": True,
            "n_failed": 0,
            "va5": {"pass": True},
            "t_start_utc": "a",
            "t_end_utc": "b",
            "wall_s": 1.0,
            "specs": {n: {"n_points": 1} for n in finalize.ARC_SPECS},
        },
    )
    for exp, f in (("E-B1", "eb1_points.csv"), ("E-B2", "eb2_per.csv")):
        _write_csv(src / "results" / exp / f, [{"x": "1"}])
        _write_json(src / "results" / exp / "meta.json", {"assumptions": ["arc label"]})
    for i, n in enumerate(finalize.ARC_SPECS):
        _write_json(src / n / "sweep.json", {"name": n, "jobs": 64})
        _write_csv(src / n / "index.csv", [{"point_id": "p00000", "status": "ok"}])
        commit = "f" * 40 if (other_commit and i == 2) else COMMIT
        _write_json(
            src / n / "p00000" / "manifest.json",
            {
                "git_commit": commit,
                "git_dirty": False,
                "platform": "Linux",
                "cpu_model": "Xeon",
                "logical_cpus": 192,
                "aws_instance": {"type": "c7i.48xlarge", "id": "i-1"},
                "versions": {"python": "3.12"},
                "seed": 20261010,
            },
        )
    return src


def test_collect_eb_arc_layout_and_meta(tmp_path: Path) -> None:
    src = _fake_arc_run(tmp_path)
    res = tmp_path / "paper_results"
    dst = finalize.collect_eb_arc(src, res)
    assert dst == res / "E-B-arc"
    for rel in (
        "E-B1/eb1_points.csv",
        "E-B2/eb2_per.csv",
        "eb_plan.json",
        "eb_summary.json",
        "git_state.json",
        "run_manifests.json",
        "meta.json",
        "sweeps/eb2_lab/index.csv",
        "sweeps/eb2_lab/sweep.json",
    ):
        assert (dst / rel).is_file(), rel
    assert not (dst / "host").exists()
    meta = json.loads((dst / "meta.json").read_text(encoding="utf-8"))
    assert meta["sremu"]["commit"] == COMMIT and meta["sremu"]["dirty"] is False
    assert meta["host"]["instance_type"] == "c7i.48xlarge" and meta["host"]["jobs"] == [64]
    assert meta["geometry_label"] == "arc label"
    assert meta["description"] == finalize.ARC_TEXT["description"]
    rm = json.loads((dst / "run_manifests.json").read_text(encoding="utf-8"))
    assert rm["n_manifests"] == 5 and rm["shared"]["git_commit"] == COMMIT
    assert not finalize.d24_violations(dst)


def test_collect_eb_arc_refuses_mixed_git_states(tmp_path: Path) -> None:
    src = _fake_arc_run(tmp_path, other_commit=True)
    with pytest.raises(RuntimeError, match="manifest differs"):
        finalize.collect_eb_arc(src, tmp_path / "paper_results")
    assert not (tmp_path / "paper_results" / "E-B-arc").exists()


def test_collect_latency_is_meas_only(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    lat = {
        "inproc": {
            "send": {"p50_us": 1.0},
            "recv": {"p50_us": 2.0},
            "sustained_msps": 50.0,
            "buffer": 1000,
        },
        "zmq_two_process": {
            "round_trip": {"p50_us": 90.0},
            "sustained_msps": 20.0,
            "sustained_buffer": 10000,
        },
    }
    _write_json(runs / "ef_latency" / "latency.json", lat)
    _write_json(runs / "ef_latency" / "manifest.json", {"cpu_model": "Xeon", "aws_instance": None})
    _write_json(runs / "ef_latency" / "meta.json", {"data_classes": ["MEAS"]})
    res = tmp_path / "res"
    finalize.collect_latency(runs, res)
    s = finalize.latency_summary(res)
    assert s["data_class"] == "MEAS" and s["host"]["cpu_model"] == "Xeon"
    assert s["zmq_round_trip"]["p50_us"] == 90.0
    _write_json(runs / "ef_latency" / "meta.json", {"data_classes": ["SIM-sremu"]})
    with pytest.raises(RuntimeError, match="MEAS"):
        finalize.collect_latency(runs, res)


def test_d24_allows_lscpu_vulnerability_names(tmp_path: Path) -> None:
    (tmp_path / "lscpu.txt").write_text(
        "Vulnerability Spectre v1:  Mitigation\nVulnerability Spectre v2:  Mitigation\n",
        encoding="utf-8",
    )
    assert finalize.d24_violations(tmp_path) == []
    (tmp_path / "x.txt").write_text("compared with v1\n", encoding="utf-8")
    assert len(finalize.d24_violations(tmp_path)) == 1


# =========================================================================== V-LAB check


def test_vlab_check_passes_on_this_tree_and_detects_a_change(tmp_path: Path) -> None:
    res = vlab_check.check(REPO)
    assert res["ok"], res["summary"]
    copy = tmp_path / "sremu"
    shutil.copytree(
        REPO / vlab_check.VLAB_DIR,
        copy / vlab_check.VLAB_DIR,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    f = copy / vlab_check.VLAB_DIR / "analysis.py"
    f.write_bytes(f.read_bytes() + b"\n# changed\n")
    res = vlab_check.check(copy)
    assert not res["ok"]
    assert res["files_differing_from_7a63287"] == [f"{vlab_check.VLAB_DIR}/analysis.py"]
    assert res["digest_equals_lock"]  # the protocol files themselves are unchanged


# =========================================================================== install_regen


def test_install_preconditions_need_a_full_pinned_passing_report() -> None:
    rep = {
        "mode": "full",
        "new_is_smoke_tree": False,
        "provenance": {"new": {"pinned": True}},
        "code_check": {"performed": True, "ok": True},
        "missing_files": [],
        "failed_files": [],
        "vlab_check": {"ok": True, "summary": "ok"},
    }
    assert install_regen.preconditions(rep, False) == []
    # a missing V-LAB record refuses, as a failed one does
    assert install_regen.preconditions({**rep, "vlab_check": None}, False)
    assert install_regen.preconditions({**rep, "vlab_check": {"ok": False}}, False)
    # differences refuse unless the lead accepts them
    diff = {**rep, "failed_files": ["E-A/gains.csv"]}
    assert install_regen.preconditions(diff, False)
    assert install_regen.preconditions(diff, True) == []
    assert install_regen.preconditions({**rep, "mode": "smoke"}, True)
