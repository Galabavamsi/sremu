"""WP11 benchmark suites: plumbing and quick suite paths.

These check that the benchmark code measures what it claims (non-zero RIS path assertion,
plan and drop bookkeeping that accounts for every point of the section 12.2 product, series
pruning, manifests with data classes and pinning, the Windows paced label, fits recovering
known exponents, the B6 scope table); they never assert a timing. Tests that start engine
worker processes are marked ``slow`` and ``integration`` (nightly tier, not CI).
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bench
from bench import _common, _scenes, b1_kernels, b2_lockstep, b3_paced, b6_crosstool

REPO = Path(__file__).resolve().parents[2]
SLOW = [pytest.mark.integration, pytest.mark.slow]
PRODUCT = 8 * 5 * 4 * 4 * 6


def test_split_links_and_scenarios_validate() -> None:
    from sremu.scenario.load import parse_scenario

    for links in (1, 2, 4, 8, 128):
        n_tx, n_rx = _scenes.split_links(links)
        assert n_tx * n_rx == links and n_tx <= n_rx
    with pytest.raises(ValueError):
        _scenes.split_links(3)
    for feat in _scenes.FEATURES:
        sc = _scenes.bench_scenario(links=4, n_cells=40, n_ris=2, rate_sps="1e6", features=feat)
        parse_scenario(sc)
        rx = [n for n in sc["nodes"] if n["id"].startswith("rx")]
        assert ("trajectory" in rx[0]) == (_scenes.feature_index(feat) >= 3)
        assert ("lo" in rx[0]["streams"][0]) == (_scenes.feature_index(feat) >= 2)


def test_ris_path_check_detects_a_zero_ris_term() -> None:
    from bench._engine_point import _resolve

    from sremu.physics.propagation import from_scenario

    sc = _scenes.bench_scenario(links=2, n_cells=40, n_ris=1, rate_sps="1e6")
    model = from_scenario(_resolve(sc).scenario.model_dump(mode="python"))
    chk = _scenes.ris_path_check(model)
    assert chk["n_panels"] == 1 and math.isfinite(chk["min_ris_minus_los_db"])
    # every receiver behind the panel: the RIS term is exactly zero -> must raise
    for n in sc["nodes"]:
        if n["id"].startswith("rx"):
            n["position_m"][0] = -5.0
    behind = from_scenario(_resolve(sc).scenario.model_dump(mode="python"))
    with pytest.raises(AssertionError, match="exactly zero"):
        _scenes.ris_path_check(behind)


# =========================================================================== B2 plans


def test_b2_slices_partition_the_full_product() -> None:
    pts = b2_lockstep.slice_points(b2_lockstep.SLICES, max_threads=16)
    keys = {(p["L"], p["N"], p["R"], p["fs"], p["F"], p["T"]) for p in pts}
    assert len(keys) == len(pts)
    product = [p for p in pts if p["T"] is None]
    dropped = b2_lockstep.dropped_points(pts)
    assert len(product) + len(dropped) == PRODUCT
    assert {d["reason"] for d in dropped} == {"not in a slice"}
    base = [p for p in product if (p["N"], p["R"], p["fs"], p["F"]) == (240, 1, "1e6", "F1")]
    assert {p["L"] for p in base} == set(_scenes.LINKS)
    assert all(set(p["slices"]) >= {"S1", "S2", "S3", "S4"} for p in base)
    s6 = [p for p in pts if p["T"] is not None]
    assert {p["T"] for p in s6} == {1, 2, 4, 8} and {p["L"] for p in s6} == {1, 8, 64}
    assert all(p["slices"] == ["S6"] and p["N"] == 240 for p in s6)
    assert b2_lockstep.point_id(s6[0]).endswith(f"_T{s6[0]['T']}")
    t32 = {p["T"] for p in b2_lockstep.slice_points(["S6"], max_threads=32) if p["T"]}
    assert t32 == {1, 2, 4, 8, 16}


def test_b2_server_plan_accounts_for_every_product_point() -> None:
    pts = b2_lockstep.server_points(16, host_cpus=192)
    product = [p for p in pts if p["T"] is None]
    dropped = b2_lockstep.dropped_points(pts, "server")
    assert len(product) + len(dropped) == PRODUCT
    sel = {(p["L"], p["N"], p["R"], p["fs"], p["F"]) for p in product}
    assert not sel.intersection((d["L"], d["N"], d["R"], d["fs"], d["F"]) for d in dropped)
    assert "not in the plan" not in {d["reason"] for d in dropped}
    # every slice point is in tier 0; tiers grow in plan order
    assert {p["tier"] for p in pts} == {0, 1, 2, 3, 4}
    assert all(p["tier"] == 0 for p in pts if set(p["slices"]) & set(b2_lockstep.SLICES))
    # pruning rules: R=0 only at N=240; F1 grid only at the static N; anchors carry F0..F5
    assert all(p["N"] == 240 for p in product if p["R"] == 0)
    f3 = [p for p in product if p["F"] == "F3"]
    assert {p["N"] for p in f3} == set(_scenes.CELLS) and {p["fs"] for p in f3} == set(
        _scenes.RATES_SPS
    )
    for n, r, fs in b2_lockstep.ANCHORS:
        assert {p["F"] for p in product if (p["N"], p["R"], p["fs"]) == (n, r, fs)} == set(
            _scenes.FEATURES
        )
    # thread scaling: S6 below the slot, S7 above it (exclusive), none above the host
    s7 = [p for p in pts if "S7" in p["slices"]]
    assert {p["T"] for p in s7} == {32, 48, 96} and all(p["tier"] == 4 for p in s7)
    series = b2_lockstep.make_series(pts, 16)
    assert sum(s.exclusive for s in series) == 3
    assert [s.tier for s in series] == sorted(s.tier for s in series)
    assert all([p["L"] for p in s.points] == sorted(p["L"] for p in s.points) for s in series)
    assert not [p for p in b2_lockstep.server_points(16, host_cpus=40) if p["T"] in (48, 96)]


def _fake_result(p: dict[str, Any], rtf: float, floor: float | None) -> dict[str, Any]:
    status = "below_floor" if floor is not None and rtf < floor else "ok"
    return {
        "status": status,
        "rtf_floor": floor,
        "scenario_sha256": "0" * 64,
        "resolved": None,
        "facts": {
            "page": 500,
            "t_u_s": None,
            "fd_filter": "integer",
            "fd_taps": 1,
            "ris_tile_paths": p["L"] * p["R"],
            "rq": p["L"] * p["N"],
        },
        "decisions": {},
        "ris_check": {"n_panels": p["R"], "min_ris_minus_los_db": -10.0 if p["R"] else None},
        "rtf_median": None if status != "ok" else rtf,
        "rtf_reps": [] if status != "ok" else [rtf] * 3,
        "rtf_min_worst": None if status != "ok" else rtf * 0.8,
        "aggregate_msps_median": 1.0,
        "reps": 0 if status != "ok" else 3,
        "rep_virtual_s": 0.3,
        "rep_pages": 600,
        "pilot_rtf": rtf,
        "pilot_virtual_s": 0.1,
        "build_s": 0.1,
        "verify_interp_max_rel": None,
        "counters": {"late_samples": 0, "overflow_pages": 0},
        "nondeterministic_inputs": False,
        "recorded_bytes": 0,
    }


def _rtf_model(p: dict[str, Any]) -> float:
    """Synthetic RTF: 4/L, mobility 10x and transport 40x slower."""
    return 4.0 / p["L"] / {"F3": 10.0, "F5": 40.0}.get(p["F"], 1.0)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _check_accounting(run_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Every planned point is run or dropped (never both, never twice), and every point of
    the 3840-point product appears exactly once in points.csv or dropped.csv."""
    rows = _read_csv(run_dir / "points.csv")
    dropped = _read_csv(run_dir / "dropped.csv")
    planned = set(json.loads((run_dir / "spec.json").read_text())["points"])
    run_ids = [r["point_id"] for r in rows]

    def pid(d: dict[str, str]) -> str:
        t = int(d["T"]) if d.get("T") else None
        q = {"L": int(d["L"]), "N": int(d["N"]), "R": int(d["R"]), "fs": d["fs"], "F": d["F"]}
        return b2_lockstep.point_id({**q, "T": t})

    drop_ids = [pid(d) for d in dropped]
    assert len(run_ids) == len(set(run_ids)) and len(drop_ids) == len(set(drop_ids))
    assert set(run_ids).isdisjoint(drop_ids)
    assert planned <= set(run_ids) | set(drop_ids), planned - set(run_ids) - set(drop_ids)
    product = {
        b2_lockstep.point_id({"L": L, "N": N, "R": R, "fs": fs, "F": F, "T": None})
        for L, N, R, fs, F in itertools.product(
            _scenes.LINKS, _scenes.CELLS, _scenes.RIS_COUNTS, _scenes.RATES_SPS, _scenes.FEATURES
        )
    }
    accounted = {x for x in set(run_ids) | set(drop_ids) if not re.search(r"_T\d+$", x)}
    assert accounted == product
    return rows, dropped


def test_b2_sequential_series_pruning_and_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_measure(p: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        calls.append(b2_lockstep.point_id(p))
        return _fake_result(p, _rtf_model(p), settings["rtf_floor"])

    monkeypatch.setattr(b2_lockstep, "measure_point", fake_measure)
    monkeypatch.setattr(b2_lockstep, "jit_warmup", lambda features="F3": None)
    monkeypatch.setattr(_common.BenchRun, "save_scenario", lambda self, pid, res: "0" * 64)
    rc = b2_lockstep.main(
        ["--quick", "--rtf-floor", "0.5", "--out", str(tmp_path), "--point-quiet-window-s", "0"]
    )
    assert rc == 0
    (run_dir,) = list(tmp_path.iterdir())
    rows, dropped = _check_accounting(run_dir)
    # F5 L=1: rtf 0.1 < floor -> below_floor (no RTF), its L=4 point dropped by pruning
    f5 = [r for r in rows if r["features"] == "F5"]
    assert [(r["L"], r["status"]) for r in f5] == [("1", "below_floor")]
    assert f5[0]["rtf_median"] == "" and float(f5[0]["pilot_rtf"]) == pytest.approx(0.1)
    stop = [d for d in dropped if d["reason"].startswith("series stopped at L=1")]
    assert [(d["L"], d["F"]) for d in stop] == [("4", "F5")]
    # F3 L=1 (S5, a one-point series): rtf 0.4 < floor as well
    assert [r["status"] for r in rows if r["features"] == "F3"] == ["below_floor"]
    assert len(calls) == len(rows)
    summ = json.loads((run_dir / "summary.json").read_text())
    assert summ["points_below_floor"] == 2 and summ["points_failed"] == 0
    assert summ["base_point_rtf"] == pytest.approx(4.0)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert "affinity" in meta["pinning"] and "NUMBA_NUM_THREADS" in meta["pinning"]["env"]


def test_b2_parallel_scheduler_prunes_and_checks_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parallel scheduler with fake workers: every series runs on some slot in
    increasing L, pruning stops a series, the isolation check reruns reference points."""
    import threading

    lock = threading.Lock()
    order: list[tuple[int, str]] = []

    def fake_run_point(
        self: b2_lockstep.ParallelRunner,
        p: dict[str, Any],
        cpus: list[int],
        slot: int,
        tag: str = "",
    ) -> dict[str, Any]:
        with lock:
            order.append((slot, b2_lockstep.point_id(p) + tag))
        r = _fake_result(p, _rtf_model(p), self.settings["rtf_floor"])
        row = b2_lockstep._row(p, r, None, 0.01, len(cpus))
        row.update({"slot": slot, "slot_cpus": len(cpus), "worker_wall_s": 0.01})
        row.update({"concurrent_points_at_start": 0, "concurrent_points_at_end": 0})
        return row

    monkeypatch.setattr(b2_lockstep.ParallelRunner, "run_point", fake_run_point)
    # on Linux the orchestrator would pin itself (this test process) to the reserved core
    from bench import _cpus

    pins: list[list[int]] = []
    monkeypatch.setattr(
        _cpus, "pin_self", lambda cpus: pins.append(list(cpus)) or {"applied": False}
    )
    rc = b2_lockstep.main(
        [
            "--plan",
            "server",
            "--quick",
            "--workers",
            "3",
            "--cpus-per-point",
            "1",
            "--rtf-floor",
            "0.5",
            "--isolation-check",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    (run_dir,) = list(tmp_path.iterdir())
    rows, dropped = _check_accounting(run_dir)
    assert {r["slot"] for r in rows} <= {"0", "1", "2"}
    assert order[0][1].endswith("_jitwarm")
    # within a series, points start in increasing L
    seen_l: dict[str, list[int]] = {}
    for _, pid in order:
        if pid.endswith(("_jitwarm", "_iso")):
            continue
        L, rest = pid.split("_", 1)
        seen_l.setdefault(rest, []).append(int(L[1:]))
    assert all(v == sorted(v) for v in seen_l.values())
    # the quick plan holds one isolation reference point (L=1 base)
    iso = _read_csv(run_dir / "isolation.csv")
    assert [r["point_id"] for r in iso] == ["L1_N240_R1_fs1e6_F1"]
    assert float(iso[0]["ratio_isolated_over_parallel"]) == 1.0
    stop = [d for d in dropped if d["reason"].startswith("series stopped at L=1")]
    assert ("4", "F5") in {(d["L"], d["F"]) for d in stop}
    slots = json.loads((run_dir / "slots.json").read_text())
    assert len(slots["slots"]) == 3
    if slots["pinned"]:  # Linux: the orchestrator moved itself onto the reserved core
        assert pins and pins[0] == bench.parse_cpu_list(slots["reserved"])
    summ = json.loads((run_dir / "summary.json").read_text())
    assert summ["isolation_ratio"]["n"] == 1 and summ["plan"] == "server"


@pytest.mark.integration
@pytest.mark.slow
def test_b2_worker_process_measures_one_point(tmp_path: Path) -> None:
    """One real worker process (unpinned off Linux): JIT warm-up, measurement, JSON out."""
    import subprocess

    job = tmp_path / "job.json"
    p = {"L": 1, "N": 40, "R": 1, "fs": "1e5", "F": "F1", "T": None, "slices": ["t"]}
    settings = {
        "reps": 1,
        "target_pages": 20,
        "rep_wall_target_s": 0.05,
        "min_virtual_s": "0.2",
        "warmup_s": "0.02",
        "point_wall_cap_s": 30.0,
        "rtf_floor": None,
        "warm_wall_max_s": 5.0,
        "pilot_wall_max_s": 2.0,
    }
    job.write_text(json.dumps({"point": p, "settings": settings}))
    out = tmp_path / "out.json"
    cmd = [
        sys.executable,
        "-m",
        "bench.b2_lockstep",
        "--worker-point",
        str(job),
        "--worker-out",
        str(out),
        "--worker-scen-dir",
        str(tmp_path / "scen"),
        "--threads",
        "2",
    ]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert res["row"]["status"] == "ok" and res["row"]["rtf_median"] > 0
    assert res["numba_threads"] == 2 and (tmp_path / "scen" / "L1_N40_R1_fs1e5_F1.yaml").exists()


# =========================================================================== shared plumbing


def test_interruptible_run_writes_partial_results(tmp_path: Path) -> None:
    run = _common.BenchRun("intr", {"x": 2}, out_root=tmp_path, quiet_s=0.0, load_period_s=0.05)
    rows = []
    with run.interruptible():
        rows.append({"i": 0})
        raise KeyboardInterrupt
    assert run.interrupted and "partial" in run.interrupted
    run.write_csv("points.csv", rows)
    run.finish({"points": len(rows)})
    summ = json.loads((run.dir / "summary.json").read_text())
    meta = json.loads((run.dir / "meta.json").read_text())
    assert summ["points"] == 1 and summ["interrupted"] == run.interrupted
    assert meta["interrupted"] == run.interrupted and (run.dir / "manifest.json").exists()
    ok = _common.BenchRun("intr", {"x": 3}, out_root=tmp_path, quiet_s=0.0, load_period_s=0.05)
    with ok.interruptible():
        pass
    ok.finish({})
    assert json.loads((ok.dir / "summary.json").read_text())["interrupted"] is None


def test_load_monitor_gate_and_window() -> None:
    lm = _common.LoadMonitor(period_s=0.05).start()
    gate = lm.wait_quiet(max_busy=1.0, window_s=0.1, timeout_s=0.0, tag="gate")
    assert gate["met"] in (True, False) and gate["waited_s"] >= 0.1
    i0 = lm.mark()
    sum(range(200_000))
    w = lm.window_stats(i0)
    lm.stop()
    assert w["n"] >= 1 and (w["mean"] is None or 0.0 <= w["mean"] <= 1.0)
    assert any(s["tag"] == "gate" for s in lm.samples)


def test_cpu_list_threads_and_client_pinning(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    import bench

    assert bench.parse_cpu_list("0-1,4, 6-7") == [0, 1, 4, 6, 7]
    with pytest.raises(ValueError):
        bench.parse_cpu_list("3-1")
    monkeypatch.delenv("SREMU_CLIENT_CPUS", raising=False)
    assert bench.pin_client_cpus() is None
    if hasattr(os, "sched_setaffinity"):
        cur = sorted(os.sched_getaffinity(0))
        monkeypatch.setenv("SREMU_CLIENT_CPUS", str(cur[0]))
        try:
            assert bench.pin_client_cpus() == [cur[0]]
        finally:
            os.sched_setaffinity(0, cur)
    import numba

    n0 = numba.get_num_threads()
    try:
        assert bench.set_threads(1) == 1 and numba.get_num_threads() == 1
    finally:
        numba.set_num_threads(n0)
    assert bench.set_threads(None) is None
    with pytest.raises(ValueError):
        bench.set_threads(0)


def test_lockstep_point_measures_records_and_floors(tmp_path: Path) -> None:
    from bench._engine_point import measure_lockstep

    sc = _scenes.bench_scenario(links=2, n_cells=40, n_ris=1, rate_sps="1e5", features="F1")
    r = measure_lockstep(
        sc, features="F1", reps=2, target_pages=20, rep_wall_target_s=0.05, min_virtual_s="0.2"
    )
    assert r["status"] == "ok" and r["reps"] == 2 and r["rtf_median"] > 0
    assert r["rep_virtual_s"] >= 0.2 and r["pilot_virtual_s"] > 0
    assert r["facts"]["n_links"] == 2 and r["facts"]["n_panels"] == 1
    assert r["counters"]["late_samples"] == 0
    floored = measure_lockstep(
        sc, features="F1", reps=2, target_pages=20, min_virtual_s="0.2", rtf_floor=1e9
    )
    assert floored["status"] == "below_floor" and floored["reps"] == 0
    assert floored["rtf_median"] is None and floored["pilot_rtf"] > 0
    run = _common.BenchRun(
        "test",
        {"x": 1},
        out_root=tmp_path,
        quiet_s=0.0,
        load_period_s=0.05,
        data_classes={"points.csv": "SIM-sremu"},
        paced=True,
    )
    sha = run.save_scenario("p0", r["resolved"])
    run.write_csv("points.csv", [{"a": 1, "rtf": r["rtf_median"], "d": {"k": 1}}])
    run.finish({"ok": True})
    from sremu.record.manifest import read_manifest

    m = read_manifest(run.dir)
    assert m.scenario_name == "bench-test"
    if (REPO / ".git").exists():  # a job bundle (git archive) has no .git
        assert m.git_commit is not None
    assert m.scenario_sha256 == _common.canonical_sha256({"x": 1})
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["data_classes"]["points.csv"] == "SIM-sremu"
    assert meta["windows_paced_not_representative"] == (sys.platform == "win32")
    if sys.platform == "win32":
        assert "Windows, not representative" in meta["platform_label"]
    assert (run.dir / "scenarios" / "p0.yaml").exists() and len(sha) == 64


def test_advance_limited_respects_the_wall_limit() -> None:
    from fractions import Fraction

    from bench._engine_point import _advance_limited

    class Eng:
        def __init__(self) -> None:
            self.calls: list[Fraction] = []

        def run(self, until_s: Fraction) -> None:
            self.calls.append(until_s)

    e = Eng()
    page = Fraction(1, 1000)
    t = _advance_limited(e, Fraction(0), Fraction(1), page, wall_max_s=10.0)
    assert t == 1 and e.calls[0] == 4 * page and e.calls[1] == 12 * page  # chunks 4, 8, ...
    e2 = Eng()
    t2 = _advance_limited(e2, Fraction(0), Fraction(10**6), page, wall_max_s=0.0)
    assert t2 == 4 * page and len(e2.calls) == 1  # stops after the first chunk


# =========================================================================== B1, B3, B6


def test_b1_fits_recover_known_exponents() -> None:
    rows = [
        {"kernel": "snapshot_k1", "backend": "numba_cpu", "RQ": rq, "t_median_s": 1e-5 + 3e-8 * rq}
        for rq in (40, 1e3, 1e4, 1e5, 1e6, 1e7)
    ]
    fit = b1_kernels.fit_snapshot(rows)["numba_cpu"]
    assert abs(fit["exponent_alpha_vs_RQ"] - 1.0) < 0.05
    assert abs(fit["b_ns_per_element"] - 30.0) < 0.5
    rend = [
        {
            "kernel": "render",
            "backend": "numba_cpu",
            "groups": "one",
            "L": L,
            "P": P,
            "T_FD": T,
            "S": S,
            "t_median_s": 1e-9 * L * max(T, 1) * S * P**0.5,
        }
        for L in (1, 4, 16)
        for P in (1, 4)
        for T in (0, 8, 32)
        for S in (512, 8192)
    ]
    rf = b1_kernels.fit_render(rend)["numba_cpu"]
    assert abs(rf["exp_L"] - 1) < 1e-6 and abs(rf["exp_T_eff"] - 1) < 1e-6
    assert abs(rf["exp_S"] - 1) < 1e-6 and abs(rf["exp_P"] - 0.5) < 1e-6
    eng = [
        {
            "T_page_s": 5e-5 + 2e-9 * xs + 1e-9 * xf + 3e-9 * xr + 4e-9 * xp,
            "x_snap": xs,
            "x_fd": xf,
            "x_rx": xr,
            "x_paths": xp,
        }
        for xs, xf, xr, xp in [
            (0, 512, 512, 1024),
            (1e4, 4096, 1024, 2048),
            (0, 8e4, 4096, 9e3),
            (5e5, 512, 512, 4096),
            (2e5, 1.6e4, 2048, 4e4),
            (0, 256, 64, 128),
            (1e3, 2e3, 512, 600),
        ]
    ]
    ef = b1_kernels.fit_engine(eng)
    assert ef["rel_rms"] < 1e-6 and abs(ef["d_ns_per_rx_sample"] - 3.0) < 1e-3
    # the min-based variants read their own timing columns
    lo = [{**r, "t_min_s": 0.5 * r["t_median_s"]} for r in rows]
    fit_lo = b1_kernels.fit_snapshot(lo, "t_min_s")["numba_cpu"]
    assert abs(fit_lo["b_ns_per_element"] - 15.0) < 0.5
    best = [{**r, "T_page_best_s": 0.5 * r["T_page_s"]} for r in eng]
    eb = b1_kernels.fit_engine(best, "T_page_best_s")
    assert abs(eb["d_ns_per_rx_sample"] - 1.5) < 1e-3 and "rel_err_T_page_best_s" in best[0]


def test_b3_plans_and_dropped_series() -> None:
    srv = b3_paced.plan_series("server", [], [], 240)
    assert srv[0] == ("1e6", "5e-4", 240, 1, "F1") and len(srv) == len(set(srv)) == 9
    assert len(b3_paced.dropped_series(srv)) == 4 * 3 * 5 * 4 * 6 - 9
    base = b3_paced.plan_series("base", ["1e6"], ["5e-4", "1e-3"], 240)
    assert base == [("1e6", "5e-4", 240, 1, "F1"), ("1e6", "1e-3", 240, 1, "F1")]
    assert b3_paced.series_key(srv[0]) == "fs1e6_page5e-4_N240_R1_F1"


def test_b6_context_table_scopes_and_agreement() -> None:
    rows = [
        {"tool": "sremu", "method": "sremu_numba_cpu_snapshot+combine", "N": 256,
         "t_median_s": 1e-4, "h_abs_db": -60.0},
        {"tool": "MATLAB", "method": "matlab_hand_loop", "N": 256, "t_median_s": 2e-3,
         "h_abs_db": -60.001},
        {"tool": "MATLAB", "method": "matlab_risreflector_step", "N": 256, "t_median_s": 5e-5,
         "h_abs_db": None},
    ]  # fmt: skip
    table = b6_crosstool.context_table(rows, "host")
    by = {r["method"]: r for r in table if r["N"] == 256}
    assert by["matlab_hand_loop"]["h_minus_sremu_db"] == pytest.approx(-0.001)
    assert by["matlab_risreflector_step"]["same_model_as_sremu"] is False
    assert by["matlab_risreflector_step"]["h_minus_sremu_db"] is None
    assert "far-field" in by["matlab_risreflector_step"]["scope"]
    assert by["sremu_numba_cpu_snapshot+combine"]["t_per_cell_ns"] == pytest.approx(
        1e-4 / 256 * 1e9
    )
    lit = [r for r in table if r["data_class"] == "REPORTED"]
    assert len(lit) == len(b6_crosstool.LITERATURE) and all(r["N"] is None for r in lit)
    with pytest.raises(ValueError):
        b6_crosstool.scene([100, 200])
    assert (REPO / "bench" / "b6_matlab.m").exists()


def test_every_bench_module_imports() -> None:
    import importlib

    mods = [
        "bench." + ".".join(p.relative_to(REPO / "bench").with_suffix("").parts)
        for p in sorted((REPO / "bench").rglob("*.py"))
        if p.name != "__init__.py"
    ]
    assert "bench.aws.run_job" in mods and "bench.b4_control" in mods
    for m in mods:
        assert importlib.import_module(m) is not None


def test_time_calls_and_percentiles() -> None:
    st = _common.time_calls(lambda: np.sum(np.arange(100)), min_time_s=0.01, min_calls=5)
    assert st["n"] >= 5 and 0 < st["min"] <= st["median"] <= st["p90"]
    p = _common.percentiles([1, 2, 3, 4, 100], (50, 99))
    assert p["p50"] == 3 and p["p99"] == 100 and p["max"] == 100


def test_bundle_commit_from_the_launch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job bundle has no .git: the manifest takes the commit from the launch command."""
    monkeypatch.delenv("SREMU_GIT_COMMIT", raising=False)
    monkeypatch.delenv("SREMU_GIT_DIRTY", raising=False)
    assert _common.bundle_commit() == (None, None, None)
    monkeypatch.setenv("SREMU_GIT_COMMIT", "f518ed1")
    assert _common.bundle_commit()[:2] == ("f518ed1", None)
    monkeypatch.setenv("SREMU_GIT_DIRTY", "1")
    commit, dirty, note = _common.bundle_commit()
    assert (commit, dirty) == ("f518ed1", True) and note and "SREMU_GIT_COMMIT" in note
    monkeypatch.setenv("SREMU_GIT_DIRTY", "0")
    assert _common.bundle_commit()[1] is False


def test_bench_cli_suites_and_rejections() -> None:
    from typer.testing import CliRunner

    from sremu import cli
    from sremu.commands import bench as bench_cmd

    assert set(bench_cmd.SUITES) == {"b1", "b2", "b3", "b4", "b6", "eb3"}
    for bad in ("b0", "b5"):
        assert CliRunner().invoke(cli.app, ["bench", "--suite", bad]).exit_code == 2


def test_bench_sources_carry_no_prior_version_naming() -> None:
    """D24: benchmark outputs and sources never name an earlier simulator (v1/v2/legacy)."""
    pat = re.compile(r"\bv[12]\b|legacy|\bANTS\b|historical", re.IGNORECASE)
    files = sorted((REPO / "bench").rglob("*.py")) + sorted((REPO / "bench").rglob("*.m"))
    files += [REPO / "bench" / "README.md", REPO / "bench" / "aws" / "job.sh"]
    files += [REPO / "src" / "sremu" / "commands" / "bench.py"]
    hits = [
        f"{f.relative_to(REPO)}:{i}: {ln.strip()}"
        for f in files
        for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
        if pat.search(ln)
    ]
    assert not hits, "\n".join(hits)


def test_summarize_picks_latest_complete_run_per_name(tmp_path: Path) -> None:
    from bench import summarize

    for name, complete in (
        ("20261010T000000Z_b2", True),
        ("20261010T010000Z_b2", True),
        ("20261010T020000Z_b2", False),
        ("20261010T000000Z_b4_lockstep", True),
        ("20261010T000000Z_other", True),
    ):
        d = tmp_path / name
        d.mkdir()
        if complete:
            (d / "manifest.json").write_text('{"git_commit": "abc"}', encoding="utf-8")
            (d / "meta.json").write_text("{}", encoding="utf-8")
            (d / "summary.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    res = summarize.summarize(tmp_path)
    assert set(res) == {"b2", "b4_lockstep"}
    assert res["b2"]["n_runs"] == 2
    assert res["b2"]["latest"]["summary"]["name"] == "20261010T010000Z_b2"


def test_b1_extended_engine_fit_recovers_dispatch_terms() -> None:
    """The diagnostic fit adds per-link-page and per-snapshot terms to the section 7.4 form
    and recovers them from exact data; the primary fit cannot (it has no such terms)."""
    pts = []
    grid = itertools.product((1, 4, 16), (40, 4096), (500, 2000), ("1e6", "1e5"), (None, 2e-3))
    for L, n, page, fs, t_u in grid:
        x = {"x_snap": 0.0 if t_u is None else float(n * L) * page / (float(fs) * t_u)}
        x.update({"x_fd": 8.0 * L * page, "x_rx": 2.0 * page, "x_paths": 4.0 * L * page})
        ns = 0.0 if t_u is None else page / (float(fs) * t_u)
        t = 2e-5 + 1e-9 * x["x_snap"] + 2e-10 * x["x_fd"] + 3e-9 * x["x_rx"] + 4e-9 * x["x_paths"]
        t += 2.5e-5 * L + 1e-3 * ns + 5e-4 * L * ns
        pts.append({**x, "L": L, "page": page, "f_s": fs, "t_u_s": t_u, "T_page_s": t})
    ext = b1_kernels.fit_engine_extended(pts)
    assert ext["rel_rms"] < 1e-6 and ext["n_coefficients"] == 8
    assert ext["f_us_per_link_page"] == pytest.approx(25.0, rel=1e-4)
    assert ext["g_us_per_snapshot"] == pytest.approx(1000.0, rel=1e-4)
    assert ext["h_us_per_link_snapshot"] == pytest.approx(500.0, rel=1e-4)
    assert b1_kernels.fit_engine(pts)["rel_rms"] > 0.01
    assert b1_kernels.fit_engine_extended(pts[:5]) == {}


def test_b2_summary_counts_ris_check_failures_and_stray_threads() -> None:
    p = {"L": 1, "N": 240, "R": 1, "fs": "1e6", "F": "F1", "T": None, "slices": ["S1"]}
    ok = b2_lockstep._row(p, _fake_result(p, 2.0, None), None, 1.0, 16)
    ok["all_threads_in_slot"] = True
    q = {**p, "L": 2}
    bad = b2_lockstep._row(q, None, "AssertionError: link a->b: RIS term is exactly zero", 1.0)
    s = b2_lockstep._summary([ok, bad], [p, q], [], [], "slices", 2.0)
    assert s["ris_check_failures"] == [b2_lockstep.point_id(q)]
    assert s["all_ris_paths_nonzero"] is False and s["points_failed"] == 1
    stray = {**ok, "all_threads_in_slot": False}
    s2 = b2_lockstep._summary([stray], [p], [], [], "slices", 1.0)
    assert s2["all_ris_paths_nonzero"] is True and s2["ris_check_failures"] == []
    assert s2["points_with_threads_outside_slot"] == [b2_lockstep.point_id(p)]


def test_b3_records_a_failed_point_and_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bench._engine_point as ep

    def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(ep, "measure_paced", boom)
    rc = b3_paced.main(["--quick", "--out", str(tmp_path)])
    assert rc == 1
    (run_dir,) = list(tmp_path.iterdir())
    summ = json.loads((run_dir / "summary.json").read_text())
    assert summ["points_failed"] == 1 and "engine exploded" in summ["failed"][0]["error"]
    assert (run_dir / "manifest.json").exists() and (run_dir / "failed.csv").exists()
    dropped = _read_csv(run_dir / "dropped.csv")
    assert [(d["L"], d["reason"]) for d in dropped] == [("4", "L=1 failed")]


def test_b4_lockstep_check_rejects_vacuous_zero_lateness() -> None:
    from bench import b4_control

    good = {
        "variant": "inproc",
        "rate_hz": 1000,
        "commands": 250,
        "expected_commands": 250,
        "status": {"applied": 250},
        "lateness_samples": {"n": 500.0},
        "lateness_nonzero": 0,
        "lateness_max": 0,
        "n_rx": 2,
    }
    assert b4_control.lockstep_problems(good) == []
    assert b4_control.lockstep_problems({**good, "variant": "none", "commands": 0}) == []
    none_cmds = {**good, "commands": 0, "status": {}, "lateness_samples": {}}
    assert b4_control.lockstep_problems(none_cmds)  # no command: zero lateness proves nothing
    assert b4_control.lockstep_problems({**good, "status": {"applied": 240, "rejected": 10}})
    assert b4_control.lockstep_problems({**good, "lateness_samples": {"n": 250.0}})
    assert b4_control.lockstep_problems({**good, "lateness_nonzero": 3, "lateness_max": 7})
    assert b4_control.lockstep_problems({**good, "variant": "none", "commands": 5})


def test_async_control_client_stops_at_its_wall_deadline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When replies take longer than 1/rate (a lagging engine), the B4 client stops at its
    wall deadline and reports the rate it achieved, instead of sending rate x duration
    commands long after the paced run ended."""
    import argparse
    import time
    from fractions import Fraction
    from types import SimpleNamespace

    import sremu.transport.client as client_mod
    from bench import _ctl_client

    class SlowEngine:
        session = SimpleNamespace(call=lambda *a, **k: None)

        def time_now(self) -> Fraction:
            return Fraction(0)

        def schedule(self, req: Any) -> Any:
            time.sleep(0.02)  # 20 ms per reply: at most ~50 commands/s
            return SimpleNamespace(ok=True, status="scheduled", error=None)

        def close(self) -> None:
            pass

    monkeypatch.setattr(client_mod, "connect_engine", lambda *a, **k: SlowEngine())
    a = argparse.Namespace(
        endpoint="tcp://x", rate_hz="1000", duration_s="0.3", mode="asap", lead_s="2e-3",
        panel="ris0", n_controls="4", seed="1", name="t", start_delay_s="0", resume=False,
    )  # fmt: skip
    t0 = time.perf_counter()
    assert _ctl_client._async(a) == 0
    assert time.perf_counter() - t0 < 2.0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["target"] == 300 and 5 <= out["sent"] <= 20
    assert out["achieved_rate_hz"] < 100 and out["status"] == {"scheduled": out["sent"]}
