"""B3: paced sustainable links and page-latency CDFs (DESIGN.md sections 4.10, 12.1-12.3, F8).

Usage::

    # laptop / custom: rates x page targets at one cell count
    python -m bench.b3_paced [--rates 1e6] [--page-targets 5e-4,1e-3,2.5e-4] [--n-cells 240]
                             [--links 1,2,4,...] [--reps 3] [--warmup-s 5] [--measure-s 60]
                             [--budget-min 85] [--quick] [--out DIR]
    # quiet Linux host (c7i.8xlarge): the server series plan
    python -m bench.b3_paced --plan server --budget-min 150 --out $SREMU_JOB_OUT/runs

Scene: the B2 benchmark scene (``bench._scenes.bench_scenario``) with L links, headless paced
run (in-process TX generators; RX pages delivered to the engine's RX history; no transport).
A **series** fixes (f_s, page target, N, R, F); L is doubled from 1 while the point is
sustainable and the first unsustainable L ends the series. Series run in plan order, so a
budget cut drops the last ones (``dropped.csv``).

* plan ``base`` (default): f_s in ``--rates`` x page target in ``--page-targets`` at
  ``--n-cells``, R = 1, F1; the 0.5 ms page target first.
* plan ``server``: :data:`SERVER_SERIES` (base point at 0.5/1/0.25 ms pages, then 5.76 and
  23.04 MS/s, N = 4096, mobility F3, four panels, 0.1 MS/s). Of the section 12.2 grid
  (N x R x f_s x F x page target), every series not in the plan is listed in
  ``dropped_series.csv``.

Per repetition (section 12.3): ``--warmup-s`` warm-up, then ``--measure-s`` measured (the
pacer's deadline statistics, the RTF meter and the counters restart at the end of the
warm-up); 3 repetitions per point. **Sustainable** (section 12.1): miss rate < 1e-3 at
1 ms slack and zero ``late_samples`` and ``overflow_pages`` over the measured window; a
point is sustainable when every repetition is. Reported: miss rate at slack 0/1/5 ms, page
latency p50/p99/p99.9/max with the full log-linear histogram (``latency_hist.json``, for
CCDF plots), utilization, counters, and the largest sustainable L per series with the median
of medians and the worst p99 (section 12.3).

Paced numbers for the paper come only from Linux; on Windows every output is labelled
"Windows, not representative". ``summary.json`` states whether the run met the section 12.3
protocol (warm-up >= 5 s, >= 60 s and >= 1e4 pages per repetition, >= 3 repetitions); each
row carries the background busy fraction sampled during its repetition and the run's CPU
affinity is in ``meta.json``.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Any

import bench
from bench._common import BenchRun

SUITE = "b3"
SERVER_SERIES: tuple[tuple[str, str, int, int, str], ...] = (
    ("1e6", "5e-4", 240, 1, "F1"),
    ("1e6", "1e-3", 240, 1, "F1"),
    ("1e6", "2.5e-4", 240, 1, "F1"),
    ("5.76e6", "5e-4", 240, 1, "F1"),
    ("23.04e6", "1e-3", 240, 1, "F1"),
    ("1e6", "5e-4", 4096, 1, "F1"),
    ("1e6", "5e-4", 240, 1, "F3"),
    ("1e6", "5e-4", 240, 4, "F1"),
    ("1e5", "1e-3", 240, 1, "F1"),
)
"""(f_s, page target, N, R, F) of the server plan, in priority order."""
PAGE_TARGETS: tuple[str, ...] = ("2.5e-4", "5e-4", "1e-3")


def series_key(s: tuple[str, str, int, int, str]) -> str:
    """Stable id of a series."""
    fs, pt, n, r, f = s
    return f"fs{fs}_page{pt}_N{n}_R{r}_{f}"


def plan_series(
    plan: str, rates: list[str], targets: list[str], n_cells: int
) -> list[tuple[str, str, int, int, str]]:
    """Series of a plan (see the module docstring)."""
    if plan == "server":
        return list(SERVER_SERIES)
    return [(fs, pt, n_cells, 1, "F1") for fs in rates for pt in targets]


def dropped_series(chosen: list[tuple[str, str, int, int, str]]) -> list[dict[str, Any]]:
    """Series of the section 12.2 grid (N x R x f_s x F x page target) not in the plan."""
    from bench._scenes import CELLS, FEATURES, RATES_SPS, RIS_COUNTS

    have = set(chosen)
    out = []
    for fs, pt, n, r, f in itertools.product(RATES_SPS, PAGE_TARGETS, CELLS, RIS_COUNTS, FEATURES):
        if (fs, pt, n, r, f) not in have:
            out.append(
                {
                    "f_s": fs,
                    "page_target_s": pt,
                    "N": n,
                    "R": r,
                    "F": f,
                    "reason": "not in the plan",
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    """Run B3 and write its run directory."""
    bench.deterministic_env()
    ap = argparse.ArgumentParser(prog="bench.b3_paced", description=__doc__.splitlines()[0])
    ap.add_argument("--plan", choices=("base", "server"), default="base")
    ap.add_argument("--rates", default="1e6")
    ap.add_argument("--page-targets", default="5e-4,1e-3,2.5e-4")
    ap.add_argument("--links", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--n-cells", type=int, default=240)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup-s", default="5")
    ap.add_argument("--measure-s", default="60")
    ap.add_argument("--budget-min", type=float, default=85.0)
    ap.add_argument("--quick", action="store_true", help="1 rep, 1 s warm-up, 3 s measured")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wait-quiet-min", type=float, default=0.0, help="wait for an idle host first")
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: all)")
    a = ap.parse_args(argv)
    bench.set_threads(a.threads)
    from bench._engine_point import measure_paced
    from bench._scenes import bench_scenario

    if a.quick:
        a.reps, a.warmup_s, a.measure_s = 1, "1", "3"
        a.links = "1,4"
        a.page_targets = "5e-4"
        a.plan = "base"
    rates = [r for r in a.rates.split(",") if r]
    targets = [p for p in a.page_targets.split(",") if p]
    links = [int(x) for x in a.links.split(",") if x]
    series = plan_series(a.plan, rates, targets, a.n_cells)
    spec = {
        "plan": a.plan,
        "threads": a.threads,
        "series": [series_key(s) for s in series],
        "links": links,
        "reps": a.reps,
        "warmup_s": a.warmup_s,
        "measure_s": a.measure_s,
        "budget_min": a.budget_min,
        "sustainable": (
            "miss rate < 1e-3 at 1 ms slack, late_samples == 0, overflow_pages == 0, every rep"
        ),
    }
    run = BenchRun(
        SUITE,
        spec,
        out_root=a.out,
        tag=a.tag or ("quick" if a.quick else (a.plan if a.plan != "base" else None)),
        mode="paced",
        coupling="barrier",
        paced=True,
        caveats=[
            "Headless paced runs: in-process TX generators, no transport, RX delivered to the "
            "engine's RX history; client and transport costs are not included.",
            "Page latency = delivery - release (inputs are always complete in-process).",
            "L is doubled while sustainable; the first unsustainable L ends a series.",
        ],
        quiet_s=1.0 if a.quick else 3.0,
        wait_quiet_s=a.wait_quiet_min * 60.0,
    )
    t_start = time.perf_counter()
    budget = a.budget_min * 60.0
    per_rep = float(Fraction(a.warmup_s) + Fraction(a.measure_s)) + 10.0
    rows: list[dict[str, Any]] = []
    hists: dict[str, Any] = {}
    dropped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    with run.interruptible():
        for s in series:
            fs, pt, n_cells, n_ris, feat = s
            key = series_key(s)
            last_ok = None
            for L in links:
                if time.perf_counter() - t_start + per_rep * a.reps > budget:
                    dropped.append({"series": key, "L": L, "reason": "budget"})
                    continue
                reps = []
                for rep in range(a.reps):
                    sc = bench_scenario(
                        links=L,
                        n_cells=n_cells,
                        n_ris=n_ris,
                        rate_sps=fs,
                        features=feat,
                        page_target_s=pt,
                        mode="paced",
                        duration_s="1",
                        seed=rep,
                    )
                    i_load = run.load.mark()
                    pid = f"{key}_L{L}_rep{rep}"
                    try:
                        r = measure_paced(
                            sc, features=feat, warmup_s=a.warmup_s, measure_s=a.measure_s
                        )
                    except Exception as exc:  # recorded; the series ends, the suite goes on
                        traceback.print_exc()
                        failed.append(
                            {
                                "point_id": pid,
                                "series": key,
                                "L": L,
                                "rep": rep,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        break
                    load = run.load.window_stats(i_load)
                    hists[pid] = r.pop("latency_hist_ns")
                    sha = run.save_scenario(pid, r.pop("resolved"))
                    lat = r["latency_us"] or {}
                    row = {
                        "point_id": pid,
                        "series": key,
                        "rate_sps": fs,
                        "page_target_s": pt,
                        "N": n_cells,
                        "n_ris": n_ris,
                        "features": feat,
                        "L": L,
                        "rep": rep,
                        "page": r["facts"]["page"],
                        "pages": r["pages"],
                        "measured_s": r["measured_s"],
                        "warmup_restart_ok": r["warmup_restart_ok"],
                        "miss_rate_0ms": r["miss_rate"].get("0ms"),
                        "miss_rate_1ms": r["miss_rate"].get("1ms"),
                        "miss_rate_5ms": r["miss_rate"].get("5ms"),
                        "lat_p50_us": lat.get("p50"),
                        "lat_p99_us": lat.get("p99"),
                        "lat_p999_us": lat.get("p99.9"),
                        "lat_max_us": lat.get("max"),
                        "util_p50": (r["utilization"] or {}).get("p50"),
                        "util_p99": (r["utilization"] or {}).get("p99"),
                        "late_samples": r["counters"].get("late_samples"),
                        "overflow_pages": r["counters"].get("overflow_pages"),
                        "underflow_slip_samples": r["counters"].get("underflow_slip_samples"),
                        "rtf": r["rtf"],
                        "sustainable": r["sustainable"],
                        "min_ris_minus_los_db": r["ris_check"]["min_ris_minus_los_db"],
                        "wall_s": r["wall_s"],
                        "bg_busy_mean": load["mean"],
                        "bg_busy_max": load["max"],
                        "scenario_sha256": sha,
                        # numba threading layer and the render driver calls so far (this
                        # process; paced engines use the serial driver below 2**18 work)
                        "threading_layer": (r.get("threading") or {}).get("layer"),
                        "render_calls": ((r.get("threading") or {}).get("render") or {}).get(
                            "calls"
                        ),
                    }
                    rows.append(row)
                    reps.append(row)
                    run.write_csv("points.csv", rows)
                    print(
                        f"{pid}: miss1ms={row['miss_rate_1ms']} p99={row['lat_p99_us']} us "
                        f"sustainable={row['sustainable']}",
                        flush=True,
                    )
                    if not row["sustainable"]:
                        break  # the point is unsustainable: no further repetitions needed
                ok = len(reps) == a.reps and all(x["sustainable"] for x in reps)
                if failed and failed[-1]["series"] == key and failed[-1]["L"] == L:
                    for L2 in links[links.index(L) + 1 :]:
                        dropped.append({"series": key, "L": L2, "reason": f"L={L} failed"})
                    break
                if ok:
                    last_ok = {
                        "L": L,
                        "median_of_median_latency_us": statistics.median(
                            x["lat_p50_us"] for x in reps if x["lat_p50_us"] is not None
                        ),
                        "worst_p99_latency_us": max(
                            x["lat_p99_us"] for x in reps if x["lat_p99_us"] is not None
                        ),
                        "worst_miss_rate_1ms": max(x["miss_rate_1ms"] for x in reps),
                        "min_pages_per_rep": min(x["pages"] or 0 for x in reps),
                        "reps": len(reps),
                    }
                else:
                    for L2 in links[links.index(L) + 1 :]:
                        dropped.append({"series": key, "L": L2, "reason": f"L={L} not sustainable"})
                    break
            best[key] = last_ok
    run.write_json("latency_hist.json", hists)
    if dropped:
        run.write_csv("dropped.csv", dropped)
    run.write_csv("dropped_series.csv", dropped_series(series))
    if failed:
        run.write_csv("failed.csv", failed)
    pages = [r["pages"] or 0 for r in rows]
    warm_f, meas_f = float(Fraction(a.warmup_s)), float(Fraction(a.measure_s))
    summary = {
        "suite": SUITE,
        "plan": a.plan,
        "protocol_12_3": {
            "warmup_s": warm_f,
            "measure_s": meas_f,
            "reps": a.reps,
            "min_pages_per_rep": min(pages) if pages else None,
            "meets_12_3": bool(
                warm_f >= 5 and meas_f >= 60 and a.reps >= 3 and pages and min(pages) >= 10_000
            ),
        },
        "largest_sustainable": best,
        "points": len(rows),
        "points_failed": len(failed),
        "failed": failed,
        "dropped": len(dropped),
        "wall_s": time.perf_counter() - t_start,
    }
    run.finish(summary)
    print(f"B3 written to {run.dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
