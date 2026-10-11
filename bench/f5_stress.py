"""F5 stress check: the B2 F5 base point measured many times by concurrent workers.

Confirms that the served F5 measurement (:func:`bench._engine_point.measure_zmq`: lockstep
engine + TX and RX client processes over native ZMQ) completes under concurrency with no
stall, and compares the end-to-end RTF with the RX client's own figure (pre-fix metric) and
the engine-only figure on every run. Usage::

    python -m bench.f5_stress --runs 100 --concurrency 10 [--cpus-per-worker 16]
        [--f5-clients separate|slot] [--links 1 --cells 240 --ris 1 --fs 1e6]
        [--reps 1 --rep-wall-s 2] [--rx-pause-s 0] [--out DIR]

``--concurrency`` worker processes run at the same time, each measuring its share of the
``--runs`` points one after the other (one ``measure_zmq`` call per run: pilot plus
``--reps`` repetitions, ``--point-cap-s`` wall cap). On Linux with ``--cpus-per-worker`` the
workers are pinned like the B2 workers (:func:`bench._cpus.plan_slots`: whole cores of one
NUMA node per worker after ``--reserve-cores``; numba threads = slot size) and their client
processes go to the free cores (``--f5-clients separate``,
:func:`bench.b2_lockstep.client_cpu_sets`) or to the worker's slot (``slot``, the B2
placement before the fix). Elsewhere (Windows) the workers are unpinned with ``--threads``
numba threads each (default: CPUs / concurrency), which oversubscribes the host as soon as
the client processes run. ``--rx-pause-s`` makes every RX client sleep once before its
warm-up completes (a starved client; the old client-only figure inflates, the end-to-end
figure must not).

A run fails when ``measure_zmq`` raises (e.g. an RX ``recv`` timeout, a stall, a client
killed at its deadline); a worker still running ``--timeout-min`` after the start is killed
and its unfinished runs count as stalled. Writes ``runs.jsonl`` (one record per run),
``summary.json`` and ``workers/`` into ``--out`` (default ``runs/bench/<UTC>_f5stress``);
exits 1 when any run failed or stalled.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import bench
from bench._common import REPO

RUN_KEYS = (
    "status",
    "rtf_median",
    "rtf_rx_client_median",
    "rtf_server_median",
    "pilot_rtf",
    "pilot_rtf_rx_client",
    "pilot_rtf_server",
    "reps",
    "reps_truncated",
    "reps_skipped_cap",
    "rep_virtual_s",
    "point_wall_s",
    "clock_shared",
    "client_affinity",
    "client_cpus_requested",
    "counters",
)


def _stats(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    return {
        "n": float(len(xs)),
        "min": min(xs),
        "median": statistics.median(xs),
        "max": max(xs),
    }


# =========================================================================== worker


def worker_main(a: argparse.Namespace) -> int:
    """Worker mode: pin, warm the JIT, measure ``--n-runs`` F5 points, one JSON line each."""
    from bench._cpus import pin_self

    cpus = bench.parse_cpu_list(a.cpus) if a.cpus else None
    pin = pin_self(cpus) if cpus else {"applied": False, "note": "unpinned"}
    if a.client_cpus:
        os.environ["SREMU_CLIENT_CPUS"] = a.client_cpus
    elif cpus:
        os.environ["SREMU_CLIENT_CPUS"] = a.cpus
    bench.set_threads(a.threads or (len(cpus) if cpus else None))
    import numba

    from bench._engine_point import measure_zmq
    from bench._scenes import bench_scenario

    out = Path(a.worker_out)
    sc = bench_scenario(links=a.links, n_cells=a.cells, n_ris=a.ris, rate_sps=a.fs, features="F5")
    inject = {"rx_pause_s": a.rx_pause_s} if a.rx_pause_s > 0 else None
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        head = {
            "worker": a.worker_index,
            "pid": os.getpid(),
            "pin": pin,
            "numba_threads": numba.get_num_threads(),
            "client_cpus": os.environ.get("SREMU_CLIENT_CPUS"),
        }
        fh.write(json.dumps({"kind": "worker", **head}) + "\n")
        fh.flush()
        # JIT warm-up: load (or compile) the kernels of the served engine (not recorded)
        warm_sc = bench_scenario(links=1, n_cells=40, n_ris=1, rate_sps="1e5", features="F5")
        try:
            measure_zmq(warm_sc, reps=0, warmup_s="0.02", min_virtual_s="0.06", point_wall_cap_s=60)
        except Exception as exc:  # the runs below report the problem again if it persists
            fh.write(json.dumps({"kind": "warmup_error", "error": repr(exc)[-2000:]}) + "\n")
            traceback.print_exc()
        for i in range(a.n_runs):
            t0 = time.perf_counter()
            rec: dict[str, Any] = {"kind": "run", "worker": a.worker_index, "i": i}
            try:
                r = measure_zmq(
                    sc,
                    reps=a.reps,
                    warmup_s="0.05",
                    target_pages=10_000,
                    min_virtual_s=a.min_virtual_s,
                    rep_wall_target_s=a.rep_wall_s,
                    point_wall_cap_s=a.point_cap_s,
                    rtf_floor=None,
                    inject=inject,
                )
                rec.update({k: r.get(k) for k in RUN_KEYS})
                rec["runs"] = [
                    {
                        k: x.get(k)
                        for k in (
                            "run",
                            "rtf",
                            "rtf_rx_client",
                            "rtf_server",
                            "virtual_s",
                            "truncated_by_wall",
                            "max_gap_s",
                            "run_wall_s",
                            "clock_ok",
                        )
                    }
                    for x in r["f5_runs"]
                ]
            except Exception as exc:  # recorded: a failed run is the finding
                rec.update(status="failed", error=f"{type(exc).__name__}: {exc}"[-3000:])
                traceback.print_exc()
            rec["wall_s"] = time.perf_counter() - t0
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
    return 0


# =========================================================================== orchestrator


def _plan(a: argparse.Namespace) -> tuple[list[list[int] | None], list[str | None], str]:
    """Per worker: its CPUs (None: unpinned) and its client CPUs; and the client mode."""
    from bench import _cpus

    k = a.concurrency
    if not (a.cpus_per_worker and hasattr(os, "sched_setaffinity")):
        return [None] * k, [None] * k, "unpinned"
    from bench.b2_lockstep import client_cpu_sets

    cores = _cpus.read_topology(cpus=_cpus.all_cpus())
    slots, reserved = _cpus.plan_slots(
        cores, a.cpus_per_worker, reserve_cores=a.reserve_cores, max_slots=k
    )
    if len(slots) < k:
        raise SystemExit(
            f"only {len(slots)} slots of {a.cpus_per_worker} CPUs fit; lower --concurrency "
            "or --cpus-per-worker"
        )
    if reserved:
        _cpus.pin_self(reserved)
    if a.f5_clients == "separate":
        sets = client_cpu_sets(cores, slots, reserved)
        cli = [_cpus.format_cpu_list(c) if c else None for c in sets]
        mode = "separate" if all(cli) else "separate (slot where no free CPU)"
    else:
        cli = [None] * k
        mode = "slot"
    return [list(s) for s in slots], cli, mode


def main(argv: list[str] | None = None) -> int:
    """Run the stress check and write its directory."""
    bench.deterministic_env()
    ap = argparse.ArgumentParser(prog="bench.f5_stress", description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=40, help="F5 points measured in total")
    ap.add_argument("--concurrency", type=int, default=10, help="concurrent worker processes")
    ap.add_argument("--links", type=int, default=1)
    ap.add_argument("--cells", type=int, default=240)
    ap.add_argument("--ris", type=int, default=1)
    ap.add_argument("--fs", default="1e6")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--rep-wall-s", type=float, default=2.0)
    ap.add_argument("--min-virtual-s", default="0.3")
    ap.add_argument("--point-cap-s", type=float, default=150.0)
    ap.add_argument("--rx-pause-s", type=float, default=0.0, help="RX client pause (starved)")
    ap.add_argument("--cpus-per-worker", type=int, default=None, help="pin workers (Linux)")
    ap.add_argument("--reserve-cores", type=int, default=1)
    ap.add_argument("--f5-clients", choices=("separate", "slot"), default="separate")
    ap.add_argument("--threads", type=int, default=None, help="numba threads per worker")
    ap.add_argument("--timeout-min", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--n-runs", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--cpus", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--client-cpus", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker_out:
        return worker_main(a)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = a.out or (REPO / "runs" / "bench" / f"{stamp}_f5stress")
    out.mkdir(parents=True, exist_ok=True)
    (out / "workers").mkdir(exist_ok=True)
    cpus, cli, mode = _plan(a)
    threads = a.threads
    if threads is None and cpus[0] is None:
        threads = max(1, (os.cpu_count() or 1) // a.concurrency)
    shares = [
        a.runs // a.concurrency + (1 if i < a.runs % a.concurrency else 0)
        for i in range(a.concurrency)
    ]
    from bench._cpus import format_cpu_list

    procs: list[tuple[int, subprocess.Popen[bytes], Path, Any]] = []
    t0 = time.perf_counter()
    for i, n in enumerate(shares):
        if n == 0:
            continue
        wout = out / "workers" / f"w{i:02d}.jsonl"
        cmd = [
            sys.executable,
            "-m",
            "bench.f5_stress",
            "--worker-out",
            str(wout),
            "--worker-index",
            str(i),
            "--n-runs",
            str(n),
            "--links",
            str(a.links),
            "--cells",
            str(a.cells),
            "--ris",
            str(a.ris),
            "--fs",
            a.fs,
            "--reps",
            str(a.reps),
            "--rep-wall-s",
            str(a.rep_wall_s),
            "--min-virtual-s",
            a.min_virtual_s,
            "--point-cap-s",
            str(a.point_cap_s),
            "--rx-pause-s",
            str(a.rx_pause_s),
        ]
        c = cpus[i]
        if c is not None:
            cmd += ["--cpus", format_cpu_list(c)]
        if cli[i]:
            cmd += ["--client-cpus", str(cli[i])]
        if threads:
            cmd += ["--threads", str(threads)]
        env = dict(os.environ)
        env["NUMBA_NUM_THREADS"] = str(threads or (len(c) if c else os.cpu_count() or 1))
        log = (out / "workers" / f"w{i:02d}.log").open("wb")
        procs.append(
            (i, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=log), wout, log)
        )
    print(
        f"[f5_stress] {a.runs} runs, {len(procs)} workers, clients {mode}, "
        f"threads {threads or 'slot'}, out {out}",
        flush=True,
    )
    deadline = t0 + a.timeout_min * 60.0
    killed: list[int] = []
    for i, p, _, log in procs:
        try:
            p.wait(timeout=max(1.0, deadline - time.perf_counter()))
        except subprocess.TimeoutExpired:
            killed.append(i)
            p.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=30)
        log.close()
    wall = time.perf_counter() - t0
    runs: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    for _i, p, wout, _ in procs:
        lines = wout.read_text(encoding="utf-8").splitlines() if wout.exists() else []
        recs = [json.loads(x) for x in lines if x.strip()]
        workers += [{**r, "rc": p.returncode} for r in recs if r.get("kind") == "worker"]
        runs += [r for r in recs if r.get("kind") == "run"]
    with (out / "runs.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in runs:
            fh.write(json.dumps(r, default=str) + "\n")
    ok = [r for r in runs if r.get("status") == "ok"]
    failed = [r for r in runs if r.get("status") != "ok"]
    stalled = a.runs - len(runs)  # never finished (worker killed or crashed)
    e2e = [r["rtf_median"] for r in ok if r.get("rtf_median")]
    cli_rtf = [r["rtf_rx_client_median"] for r in ok if r.get("rtf_rx_client_median")]
    srv = [r["rtf_server_median"] for r in ok if r.get("rtf_server_median")]

    def ratios(pilot: bool) -> list[float]:  # client-only over end-to-end, per served run
        return [
            x["rtf_rx_client"] / x["rtf"]
            for r in ok
            for x in r.get("runs", [])
            if x.get("rtf") and x.get("rtf_rx_client") and (x.get("run") == "pilot") == pilot
        ]

    ratio, ratio_p = ratios(False), ratios(True)
    over = [r["point_wall_s"] for r in ok if r.get("point_wall_s", 0) > a.point_cap_s]
    summary = {
        "runs_requested": a.runs,
        "runs_finished": len(runs),
        "runs_ok": len(ok),
        "runs_failed": len(failed),
        "runs_stalled_or_lost": stalled,
        "workers": len(procs),
        "workers_killed_at_timeout": killed,
        "worker_exit_codes": [p.returncode for _, p, _, _ in procs],
        "concurrency": a.concurrency,
        "point": {"L": a.links, "N": a.cells, "R": a.ris, "fs": a.fs},
        "settings": {
            "reps": a.reps,
            "rep_wall_s": a.rep_wall_s,
            "min_virtual_s": a.min_virtual_s,
            "point_cap_s": a.point_cap_s,
            "rx_pause_s": a.rx_pause_s,
        },
        "pinning": {
            "mode": mode,
            "worker_cpus": [None if c is None else format_cpu_list(c) for c in cpus],
            "client_cpus": cli,
            "threads": threads,
        },
        "rtf_end_to_end": _stats(e2e),
        "rtf_rx_client": _stats(cli_rtf),
        "rtf_server": _stats(srv),
        "ratio_rx_client_over_end_to_end_reps": _stats(ratio),
        "ratio_rx_client_over_end_to_end_pilots": _stats(ratio_p),
        "reps_with_ratio_gt_1.25": sum(1 for x in ratio if x > 1.25),
        "pilots_with_ratio_gt_1.25": sum(1 for x in ratio_p if x > 1.25),
        "reps_truncated": sum(int(r.get("reps_truncated") or 0) for r in ok),
        "points_over_cap": len(over),
        "clock_shared_all": all(r.get("clock_shared") for r in ok) if ok else None,
        "bad_packets_or_seq_errors": sum(
            int((r.get("counters") or {}).get(k, 0))
            for r in ok
            for k in ("bad_packets", "seq_errors")
        ),
        "errors": sorted({str(r.get("error"))[:300] for r in failed}),
        "wall_s": wall,
        "python": sys.version,
        "platform": sys.platform,
        "worker_records": workers,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in list(summary)[:12]}, default=str), flush=True)
    print(
        f"[f5_stress] e2e {summary['rtf_end_to_end']} rx_client {summary['rtf_rx_client']} "
        f"ratio reps {summary['ratio_rx_client_over_end_to_end_reps']} "
        f"pilots {summary['ratio_rx_client_over_end_to_end_pilots']}",
        flush=True,
    )
    return 0 if not failed and stalled == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
