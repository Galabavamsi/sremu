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


# =========================================================================== E-E and junit
# (experiments/regen_checks.py, the E-E and junit scopes of compare_regen.py and
# install_regen.py, tools/tasks.py paper-junit --git-state)


def _ee_export(root: Path, *, commit: str = COMMIT, rep2_decisions: str = "d1") -> Path:
    """A minimal E-E export (layout of experiments/ee_gnuradio.py export)."""
    ex = root / "E-E"
    sha = {
        "static_focus": {
            "rep1": {"rx_iq": "a", "costas": "c", "decisions": "d"},
            "rep2": {"rx_iq": "a", "costas": "c", "decisions": "d"},
        },
        "switching": {
            "rep1": {"rx_iq": "b", "costas": "e", "decisions": "d1"},
            "rep2": {"rx_iq": "b", "costas": "e", "decisions": rep2_decisions},
        },
    }
    checks: dict[str, object] = dict.fromkeys(cr.EE_REQUIRED_CHECKS, True)
    checks.update({"problems": [], "identical_per_run": dict.fromkeys(sha, True)})
    tput = {
        "rate_sps": 1e6,
        "duration_s": 2.0,
        "sremu_rtf_median": 1.5,
        "gr_rtf_steady": 1.4,
        "sremu_rtf_windows": 19,
        "ser": 0.0,
        "ok": True,
    }
    _write_json(
        ex / "summary.json",
        {
            "experiment": "E-E",
            "repetitions": ["rep1", "rep2"],
            "runs_per_repetition": 2,
            "checks": checks,
            "static": [{"config": "focus", "symbols": 70000, "symbol_errors": 0, "ls_snr_db": 2.5}],
            "sha256": sha,
            "throughput": [tput],
            "gnuradio_version": "3.10.9.2",
            "host": {"cpu_model": "Xeon", "aws_instance": {"type": "c7i.48xlarge"}},
            "git": {"commit": commit, "dirty": False},
            "note": "n",
        },
    )
    _write_json(
        ex / "meta.json",
        {
            "experiment": "E-E",
            "data_classes": ["SIM-sremu"],
            "code": {"src_sremu_tree_sha256": "t"},
            "inputs": [{"path": "runs/rep1/x/run.json", "sha256": "x", "data_class": "SIM-sremu"}],
            "git": {"commit": commit},
            "root": "/opt/job/out/ee",
        },
    )
    for rep in ("rep1", "rep2"):
        _write_json(ex / "runs" / rep / "static_focus" / "manifest.json", {"git_commit": None})
    return ex


def test_regen_checks_ee_export(tmp_path: Path) -> None:
    from experiments import regen_checks as rc

    gs = {"commit": COMMIT, "dirty": False, "code": {"src_sremu_tree_sha256": "t"}}
    ex = _ee_export(tmp_path / "new")
    rep = rc.ee_check(ex, gs, reps=2, rates=[1e6], d24=False)
    assert rep["ok"], rep["problems"]
    # the served emulator records null in a bundle without .git: reported, not required
    assert rep["run_manifest_git_states"] == {"None|None": 2}
    assert rep["run_manifests_record_job_commit"] is False
    assert not rc.ee_check(ex, gs, reps=2, rates=[1e6, 1.92e6], d24=False)["ok"]
    assert not rc.ee_check(ex, {**gs, "commit": "f" * 40}, reps=2, rates=[1e6], d24=False)["ok"]
    # repetitions that differ in one hash fail; equality with a reference is only reported
    bad = _ee_export(tmp_path / "bad", rep2_decisions="other")
    rep = rc.ee_check(bad, gs, reps=2, rates=[1e6], ref=ex, d24=False)
    assert not rep["ok"]
    assert rep["identical_across_repetitions"]["switching"] is False
    eq = rep["vs_reference"]["hashes_equal_per_run"]["static_focus"]
    assert eq == {"costas": True, "decisions": True, "rx_iq": True}


def test_compare_ee_scope_skips_meas_and_compares_values(tmp_path: Path) -> None:
    ref = tmp_path / "ref"
    _ee_export(ref, commit="e" * 40)
    new = tmp_path / "job" / "paper_results"
    ex = _ee_export(new)
    s = json.loads((ex / "summary.json").read_text(encoding="utf-8"))
    s["throughput"][0].update({"sremu_rtf_median": 0.7, "gr_rtf_steady": 0.6})  # MEAS
    s["sha256"]["static_focus"]["rep1"]["rx_iq"] = "z"  # hashes: reported apart
    s["sha256"]["static_focus"]["rep2"]["rx_iq"] = "z"
    s["host"] = {"cpu_model": "other"}
    _write_json(ex / "summary.json", s)
    ev_new = ex / "runs" / "rep1" / "static_focus" / "events.jsonl"
    ev_new.write_text('{"k_applied": 5, "t_wall_recv_ns": 1}\n', encoding="utf-8")
    ev_ref = ref / "E-E" / "runs" / "rep1" / "static_focus" / "events.jsonl"
    ev_ref.write_text('{"k_applied": 5, "t_wall_recv_ns": 9}\n', encoding="utf-8")
    tol = cr.Tol(1e-9, 1e-12, 1e-9)
    files, missing, extra = cr.compare_ee(new, ref, tol, False)
    assert not missing
    assert not extra
    failed = {k: f.exact_mismatches for k, f in files.items() if f.failed(False)}
    assert not failed
    assert {"sremu_rtf_median", "sha256"} <= files["E-E/summary.json"].not_compared
    # a changed decision count or event sample is a difference
    s["static"][0]["symbol_errors"] = 3
    _write_json(ex / "summary.json", s)
    ev_new.write_text('{"k_applied": 6, "t_wall_recv_ns": 1}\n', encoding="utf-8")
    files, _m, _e = cr.compare_ee(new, ref, tol, False)
    assert files["E-E/summary.json"].failed(False)
    assert files["E-E/runs/rep1/static_focus/events.jsonl"].failed(False)
    prov = cr.provenance(new)
    assert prov["per_experiment"]["E-E"]["git_commit"] == COMMIT
    assert prov["per_experiment"]["E-E"]["run_manifest_git_commits"] == {"None": 2}


def _junit_xml(p: Path, cases: list[tuple[str, str, str]]) -> Path:
    inner = {"failed": "<failure message='x'/>", "skipped": "<skipped message='s'/>"}
    body = "".join(
        f'<testcase classname="{cls}" name="{name}">{inner.get(status, "")}</testcase>'
        for cls, name, status in cases
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="pytest">'
        + body
        + "</testsuite></testsuites>",
        encoding="utf-8",
    )
    return p


GR_RAW = "tests.integration.test_gnuradio_raw"
VA1_CASE = ("tests.validation.test_va1_record", "test_va1_record_meets_the_tolerance", "ok")
GOOD_CASES = [
    VA1_CASE,
    (GR_RAW, "test_gnuradio_dqpsk_switching_run_through_raw_mode", "ok"),
    (GR_RAW, "test_gnuradio_static_runs_are_bitwise_reproducible", "ok"),
    ("tests.integration.test_gnuradio_analysis", "test_tx_symbols_are_pinned", "ok"),
    ("tests.io.test_io08_raw_exit_when_done", "test_io08_exit", "ok"),
]


def test_regen_checks_junit(tmp_path: Path) -> None:
    from experiments import regen_checks as rc

    rep = rc.junit_check(_junit_xml(tmp_path / "j.xml", GOOD_CASES))
    assert rep["ok"], rep["problems"]
    assert {"C-RAW", "C-VA1"} <= set(rep["claims_junit_green"])
    assert "C-EE" in rep["claims_not_junit_green"]  # V-CFO-3, IO-02 not in this junit
    # a skipped Table IV id (gen_results.py refuses it) and a skipped GNU Radio test fail
    io05 = ("tests.io.test_io05_sigmf", "test_io05_sigmf_package_validator", "skipped")
    cases = [VA1_CASE, (GR_RAW, "test_gnuradio_x", "skipped"), *GOOD_CASES[3:], io05]
    rep = rc.junit_check(_junit_xml(tmp_path / "k.xml", cases))
    assert not rep["ok"]
    assert rep["skipped_with_table4_id"] == [f"{io05[0]}::{io05[1]}"]
    assert rep["required_tests"]["tests/integration/test_gnuradio_raw.py"] == "skipped"
    # without GNU Radio (Windows) the raw tests may skip; smoke: the VA-1 test may fail
    cases = [
        (c, n, "failed" if c == VA1_CASE[0] else ("skipped" if c == GR_RAW else s))
        for c, n, s in GOOD_CASES
    ]
    xml = _junit_xml(tmp_path / "s.xml", cases)
    assert rc.junit_check(xml, mode="smoke", expect_gnuradio=False)["ok"]
    assert not rc.junit_check(xml, mode="full", expect_gnuradio=False)["ok"]
    assert rc.spec_id_pattern("VA-1").search("tests.validation.test_va1_record::test_va1_x")
    assert not rc.spec_id_pattern("PH-02").search("tests.physics.test_ph20::test_x")


def test_regen_checks_host_label() -> None:
    from experiments import regen_checks as rc

    assert rc.host_label({"aws_instance": {"type": "c7i.48xlarge"}}) == "AWS c7i.48xlarge"
    wsl = "Linux-6.18.40.1-microsoft-standard-WSL2-x86_64-with-glibc2.39"
    assert rc.host_label({"platform": wsl}) == "laptop (WSL2)"
    assert rc.host_label({"platform": "Windows-11-10.0.26200-SP0"}) == "laptop (Windows)"
    assert rc.host_label({}) == "n/r"


def test_tasks_paper_junit_reads_a_git_state_file(tmp_path: Path, monkeypatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("sremu_tasks_regen", REPO / "tools" / "tasks.py")
    assert spec is not None
    assert spec.loader is not None
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], budget_s: float | None = None) -> int:
        calls.append(cmd)
        return 0

    def no_git(repo: Path) -> tuple[str, bool]:
        raise AssertionError("git must not run with --git-state")

    monkeypatch.setattr(tasks, "_run", fake_run)
    monkeypatch.setattr(tasks, "_git_state", no_git)
    gs = tmp_path / "git_state.json"
    _write_json(gs, {"commit": COMMIT, "dirty": False, "code": {}})
    xml = tmp_path / "junit" / "sremu_paper.xml"
    argv = ["paper-junit", str(xml), "--git-state", str(gs), "--host", "AWS c7i.48xlarge", "-x"]
    assert tasks.main(argv) == 0
    (cmd,) = calls
    assert "-x" in cmd
    assert f"--junitxml={xml}" in cmd
    assert "--host" not in cmd
    meta = json.loads(xml.with_name("sremu_paper.meta.json").read_text(encoding="utf-8"))
    assert meta["sremu_commit"] == COMMIT
    assert meta["sremu_dirty"] is False
    assert meta["host"] == "AWS c7i.48xlarge"
    assert "no git process" in meta["git_state_source"]
    _write_json(gs, {"commit": "abc", "dirty": False})  # not a full commit id: refused
    assert tasks.main(["paper-junit", str(xml), "--git-state", str(gs)]) == 1


def _install_trees(tmp_path: Path) -> tuple[Path, Path]:
    """A job output and a paper tree that install_regen.plan accepts."""
    job, paper = tmp_path / "job", tmp_path / "paper"
    new = job / "paper_results"
    for d in (*install_regen.RESULT_DIRS, install_regen.EE_DIR):
        (new / d).mkdir(parents=True)
    for f in install_regen.ROOT_FILES:
        (new / f).write_text("{}", encoding="utf-8")
    for n in (install_regen.JUNIT_NAME, install_regen.JUNIT_META):
        _write_json(job / "junit" / n, {})
    old = {
        st: f"validation/{sub}/results/OLD_{sub}"
        for st, (sub, _n, _c) in install_regen.STUDIES.items()
    }
    gr = paper / "tools" / "gen_results.py"
    gr.parent.mkdir(parents=True)
    gr.write_text(
        "".join(f'{c} = "{old[st]}"\n' for st, (_s, _n, c) in install_regen.STUDIES.items()),
        encoding="utf-8",
    )
    for rel, want in install_regen.READERS.items():
        if rel != "tools/gen_results.py":
            (paper / rel).parent.mkdir(parents=True, exist_ok=True)
            text = "".join(f"{old[st]}\n" * n for st, n in want.items())
            (paper / rel).write_text(text, encoding="utf-8")
    for sub, name, _c in install_regen.STUDIES.values():
        (job / "validation" / sub / f"20261012T000000Z_{name}").mkdir(parents=True)
    (paper / "generated" / "junit").mkdir(parents=True)
    return job, paper


def test_install_with_ee_and_junit(tmp_path: Path) -> None:
    rep = {
        "mode": "full",
        "new_is_smoke_tree": False,
        "provenance": {"new": {"pinned": True}},
        "code_check": {"performed": True, "ok": True, "job_commit": COMMIT},
        "missing_files": [],
        "failed_files": [],
        "vlab_check": {"ok": True},
        "ee": {
            "performed": True,
            "check": {"ok": True},
            "job_check": {"ok": True},
            "missing_files": [],
            "failed_files": ["E-E/summary.json"],
        },
        "junit": {
            "performed": True,
            "ok": True,
            "sidecar": {"sremu_commit": COMMIT, "sremu_dirty": False},
        },
    }
    assert install_regen.preconditions(rep, False) == []  # E-E and junit are opt-in
    assert install_regen.preconditions(rep, False, with_ee=True)  # E-E differs: review first
    assert install_regen.preconditions(rep, True, with_ee=True, with_junit=True) == []
    dirty = {**rep, "junit": {**rep["junit"], "sidecar": {"sremu_commit": COMMIT}}}
    assert install_regen.preconditions(dirty, True, with_junit=True)
    assert install_regen.preconditions({**rep, "ee": {"performed": False}}, True, with_ee=True)
    # plan: the job's junit replaces generated/junit/sremu_paper.xml; another *.xml refuses
    job, paper = _install_trees(tmp_path)
    p = install_regen.plan(job, paper, tmp_path / "sremu", False, with_ee=True, with_junit=True)
    dsts = {d.relative_to(paper).as_posix() for _s, d in p["copies"]}
    assert {"sweeps/results/E-E", "generated/junit/sremu_paper.xml"} <= dsts
    assert "generated/junit/sremu_paper.meta.json" in dsts
    (paper / "generated" / "junit" / "windows.xml").write_text("<x/>", encoding="utf-8")
    with pytest.raises(SystemExit, match="other junit files"):
        install_regen.plan(job, paper, tmp_path / "sremu", False, with_junit=True)


def test_regen_checks_gnuradio_installs_only_what_is_missing(tmp_path: Path, monkeypatch) -> None:
    import argparse

    from experiments import regen_checks as rc

    calls: list[list[str]] = []
    monkeypatch.setattr(rc, "apt_install", lambda pk: calls.append(list(pk)) or [{"rc": 0}])
    monkeypatch.setattr(rc, "dpkg_versions", lambda patterns=(): {})
    monkeypatch.setattr(rc.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rc.os, "geteuid", lambda: 0, raising=False)
    present = {"available": True, "gnuradio_version": "3.10.9.2"}
    monkeypatch.setattr(rc, "gr_probe", lambda py: dict(present))
    which = {"apt-get": "/usr/bin/apt-get", "git": None}
    monkeypatch.setattr(rc.shutil, "which", lambda n: which.get(n))
    a = argparse.Namespace(
        gr_python="/usr/bin/python3", install="auto", out=str(tmp_path / "g.json")
    )
    assert rc.cmd_gnuradio(a) == 0
    assert calls == [["git"]]  # GNU Radio present: only git (root, missing)
    rec = json.loads((tmp_path / "g.json").read_text(encoding="utf-8"))
    assert rec["installed_by_job"] is False and rec["apt_packages"] == ["git"]
    calls.clear()
    which["git"] = "/usr/bin/git"
    probes = iter([{"available": False, "reason": "no module"}, dict(present)])
    monkeypatch.setattr(rc, "gr_probe", lambda py: next(probes))
    assert rc.cmd_gnuradio(a) == 0
    assert calls == [["gnuradio"]]
    calls.clear()
    monkeypatch.setattr(rc, "gr_probe", lambda py: {"available": False, "reason": "absent"})
    a.install = "never"
    assert rc.cmd_gnuradio(a) == 1
    assert calls == []
