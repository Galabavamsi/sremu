"""B2: lockstep RTF sweep (DESIGN.md sections 12.1-12.3, F7).

Usage::

    # laptop: one-factor slices, sequential, in-process
    python -m bench.b2_lockstep [--quick] [--slices S1,S2,...] [--budget-min 150] [--out DIR]
    # large Linux host (c7i.48xlarge): server plan, one pinned worker process per point
    python -m bench.b2_lockstep --plan server --workers auto --cpus-per-point 16 \\
        --budget-min 150 --out $SREMU_JOB_OUT

The full product L x N x RIS count x f_s x feature set (8 x 5 x 4 x 4 x 6 = 3840 points;
section 12.2) is pruned by a plan; every product point that is not run is listed in
``dropped.csv`` with its reason. L = links, N = cells per panel, R = panels, F = feature set
(:data:`bench._scenes.FEATURE_TEXT`); the base point is N = 240, R = 1, f_s = 1 MS/s, F1.

**Plan ``slices``** (laptop default): one-factor slices through the base point:

* **S1** L x N at the base (F7a: RTF vs links for each N);
* **S2** L x F0..F5 at the base (feature-set cost; F5 = ZMQ transport);
* **S3** L x R in {0, 1, 2, 4} at the base;
* **S4** L x f_s in {0.1, 1, 5.76, 23.04} MS/s at the base;
* **S5** N x L in {1, 8, 64} with mobility (F3), where the snapshot cost grows with N;
* **S6** numba threads T in {1, 2, 4, 8, 16} (below the default count) x L in {1, 8, 64}.

**Plan ``server``** (tiers, run in this order; a budget cut drops the later tiers first):

0. the slices S1-S6 (S6: T below the per-point CPU count);
1. **G-F3**: L x N x R in {1, 2, 4} x f_s with mobility (F3), plus R = 0 at N = 240;
2. **G-F1**: L x N in {240, 16384} x R in {1, 2, 4} x f_s, static (F1), plus R = 0;
3. **G-anchor**: L x F0..F5 at the anchors (N, R, f_s) in :data:`ANCHORS`;
4. **S7** (exclusive host): T in {32, 48, 96} x L in {8, 64, 128} at the base, each point
   alone on the host, pinned to whole cores of the first NUMA node(s).

Pruning rules of the server plan (each a ``dropped.csv`` reason): with R = 0 the cell count
has no effect (no panel), so only N = 240 runs; in a static scene (F0-F2) N enters only the
one-time snapshot, so F1 runs N in {240, 16384} (S1 has every N at 1 MS/s); the feature
sets other than F1/F3 run only at the base and the anchors.

**Series pruning** (both plans): points are grouped in series (same N, R, f_s, F, threads)
and run in increasing L. When a point's pilot RTF is below ``--rtf-floor`` (status
``below_floor``), fails or times out, the larger-L points of its series are not run (reason
``series stopped at L=...``): RTF falls with L, so they would only be slower.

**Parallel workers** (``--workers N|auto``, Linux): the host is cut into CPU slots of
``--cpus-per-point`` logical CPUs, each made of whole physical cores inside one NUMA node,
after reserving ``--reserve-cores`` cores of node 0 for the OS and this orchestrator
(:func:`bench._cpus.plan_slots`). Each slot runs one series at a time, one worker process
per point (``python -m bench.b2_lockstep --worker-point ...``), which pins every one of its
threads to its slot (``sched_setaffinity`` per thread, so the OpenBLAS pool started by
``import numpy`` moves too) before numba starts and runs ``--cpus-per-point`` numba threads
(S6 points: T threads in the same slot); F5 client processes share the slot. Each row
records its slot, the affinity read back, the CPU masks of all worker threads after the
point (``all_threads_in_slot``) and how many other points ran concurrently.
Concurrent points share the socket's L3 cache, memory bandwidth and power budget, so an
**isolation check** reruns ``--isolation-check`` reference points alone on one slot after the
sweep and reports ``rtf_isolated / rtf_parallel`` (``isolation.csv``). On Windows the
workers run unpinned (recorded).

Per point (:func:`bench._engine_point.measure_lockstep`): a wall-limited warm-up (>= 50 ms of
virtual time), a wall-limited untimed pilot, then up to 3 repetitions of up to 1e4 pages
each, shortened so one repetition takes about ``--rep-wall-s`` of wall time and never below
0.3 s of virtual time (two complete 100 ms RTF windows). This deviates from the paced
protocol of section 12.3 (>= 60 s, >= 1e4 pages): the table states the virtual time and
pages of every repetition. Every scene with panels asserts a non-zero RIS term on every link
(:func:`bench._scenes.ris_path_check`); the per-point RIS-minus-LOS power is recorded.

**Background load** (sequential mode): before each point the host is sampled for
``--point-quiet-window-s``; with ``--point-quiet-wait-s`` > 0 the suite waits (up to that long)
until the system busy fraction is at most ``--point-quiet-busy``. Each row records the gate
result and the background busy fraction during the point.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import bench
from bench._common import REPO, BenchRun
from bench._scenes import CELLS, FEATURE_TEXT, FEATURES, LINKS, RATES_SPS, RIS_COUNTS, split_links

SUITE = "b2"
BASE = {"N": 240, "R": 1, "fs": "1e6", "F": "F1"}
SLICES = ("S1", "S2", "S3", "S4", "S5", "S6")
PLANS = ("slices", "server")
THREADS_S6 = (1, 2, 4, 8, 16)
THREADS_S7 = (32, 48, 96)
LINKS_S7 = (8, 64, 128)
ANCHORS: tuple[tuple[int, int, str], ...] = (
    (240, 1, "5.76e6"),
    (4096, 1, "1e6"),
    (240, 2, "23.04e6"),
)
"""(N, R, f_s) at which the server plan sweeps every feature set F0..F5 (besides the base)."""
STATIC_N = (240, 16384)
"""Cell counts of the static (F1) server grid."""
ISOLATION_PREF: tuple[tuple[int, int, int, str, str], ...] = (
    (1, 240, 1, "1e6", "F1"),
    (8, 240, 1, "1e6", "F1"),
    (64, 240, 1, "1e6", "F1"),
    (8, 4096, 1, "1e6", "F3"),
    (8, 240, 1, "23.04e6", "F1"),
    (8, 240, 1, "1e6", "F5"),
    (128, 240, 1, "1e6", "F1"),
    (64, 16384, 4, "1e6", "F3"),
)
"""Reference points (L, N, R, f_s, F) of the isolation check, in order of preference."""


# =========================================================================== plans


Point = dict[str, Any]


def _adder(pts: dict[tuple[Any, ...], Point]) -> Callable[..., None]:
    def add(
        sl: str, L: int, N: int, R: int, fs: str, F: str, T: int | None = None, tier: int = 0
    ) -> None:
        key = (L, N, R, fs, F, T)
        if key not in pts:
            pts[key] = {
                "L": L,
                "N": N,
                "R": R,
                "fs": fs,
                "F": F,
                "T": T,
                "slices": [sl],
                "tier": tier,
            }
        else:
            p = pts[key]
            if sl not in p["slices"]:
                p["slices"].append(sl)
            p["tier"] = min(p["tier"], tier)

    return add


def slice_points(
    slices: Sequence[str], quick: bool = False, max_threads: int | None = None
) -> list[Point]:
    """Ordered, de-duplicated points of the requested slices (each lists its slices).
    S6 uses the thread counts of :data:`THREADS_S6` below ``max_threads`` (the default
    count; those runs are the S1-S4 base points)."""
    links = (1, 4) if quick else LINKS
    pts: dict[tuple[Any, ...], Point] = {}
    add = _adder(pts)
    b = BASE
    for sl in slices:
        if sl == "S1":
            for N, L in itertools.product((40, 240) if quick else CELLS, links):
                add(sl, L, N, b["R"], b["fs"], b["F"])
        elif sl == "S2":
            for F, L in itertools.product(("F0", "F1", "F4", "F5") if quick else FEATURES, links):
                add(sl, L, b["N"], b["R"], b["fs"], F)
        elif sl == "S3":
            for R, L in itertools.product((0, 2) if quick else RIS_COUNTS, links):
                add(sl, L, b["N"], R, b["fs"], b["F"])
        elif sl == "S4":
            for fs, L in itertools.product(("1e5", "1e6") if quick else RATES_SPS, links):
                add(sl, L, b["N"], b["R"], fs, b["F"])
        elif sl == "S5":
            for N, L in itertools.product((40,) if quick else CELLS, (1,) if quick else (1, 8, 64)):
                add(sl, L, N, b["R"], b["fs"], "F3")
        elif sl == "S6":
            ts = [t for t in THREADS_S6 if max_threads is None or t < max_threads]
            for T, L in itertools.product(ts[:2] if quick else ts, (1, 4) if quick else (1, 8, 64)):
                add(sl, L, b["N"], b["R"], b["fs"], b["F"], T)
        else:
            raise ValueError(f"unknown slice {sl!r}")
    return list(pts.values())


def server_points(slot_cpus: int, host_cpus: int | None = None, quick: bool = False) -> list[Point]:
    """Points of the server plan (see the module docstring), tier by tier. ``slot_cpus`` is
    the per-point CPU count (S6 uses T below it); S7 uses T in :data:`THREADS_S7` with
    ``slot_cpus < T <= host_cpus``."""
    pts: dict[tuple[Any, ...], Point] = {}
    for p in slice_points(SLICES, quick, max_threads=slot_cpus):
        pts[(p["L"], p["N"], p["R"], p["fs"], p["F"], p["T"])] = p
    add = _adder(pts)
    links = (1, 2) if quick else LINKS
    cells = (40, 240) if quick else CELLS
    rates = ("1e5", "1e6") if quick else RATES_SPS
    ris = (1,) if quick else (1, 2, 4)
    static_n = (240,) if quick else STATIC_N
    anchors = ((40, 1, "1e5"),) if quick else ANCHORS
    feats = ("F0", "F4") if quick else FEATURES
    for fs, L in itertools.product(rates, links):  # tier 1: mobility grid
        add("G-F3", L, 240, 0, fs, "F3", tier=1)
        for N, R in itertools.product(cells, ris):
            add("G-F3", L, N, R, fs, "F3", tier=1)
    for fs, L in itertools.product(rates, links):  # tier 2: static grid
        add("G-F1", L, 240, 0, fs, "F1", tier=2)
        for N, R in itertools.product(static_n, ris):
            add("G-F1", L, N, R, fs, "F1", tier=2)
    for (N, R, fs), F, L in itertools.product(anchors, feats, links):  # tier 3
        add("G-anchor", L, N, R, fs, F, tier=3)
    if host_cpus is not None and not quick:  # tier 4: exclusive large-thread points
        for T, L in itertools.product(THREADS_S7, LINKS_S7):
            if slot_cpus < T <= host_cpus:
                add("S7", L, BASE["N"], BASE["R"], BASE["fs"], BASE["F"], T, tier=4)
    return list(pts.values())


def point_id(p: Point) -> str:
    """Stable id of a point."""
    t = f"_T{p['T']}" if p.get("T") else ""
    return f"L{p['L']}_N{p['N']}_R{p['R']}_fs{p['fs']}_{p['F']}{t}"


def _drop_reason(plan: str, N: int, R: int, fs: str, F: str) -> str:
    if plan != "server":
        return "not in a slice"
    if R == 0 and BASE["N"] != N:
        return "R=0: no panel, so N has no effect (N=240 run)"
    if F not in ("F1", "F3") and (N, R, fs) not in ANCHORS:
        return "feature set run only at the base and the anchors"
    if F == "F1" and N not in STATIC_N:
        return "static scene: N enters only the one-time snapshot (F1 runs N=240,16384; S1 all N)"
    return "not in the plan"


def dropped_points(selected: Sequence[Point], plan: str = "slices") -> list[dict[str, Any]]:
    """Points of the full product that the plan does not select, with the reason; points
    with explicit thread counts (S6, S7) are extra and do not cover product points."""
    chosen = {(p["L"], p["N"], p["R"], p["fs"], p["F"]) for p in selected if not p.get("T")}
    out = []
    for L, N, R, fs, F in itertools.product(LINKS, CELLS, RIS_COUNTS, RATES_SPS, FEATURES):
        if (L, N, R, fs, F) not in chosen:
            out.append(
                {
                    "L": L,
                    "N": N,
                    "R": R,
                    "fs": fs,
                    "F": F,
                    "reason": _drop_reason(plan, N, R, fs, F),
                }
            )
    return out


@dataclass
class Series:
    """Points with the same (N, R, f_s, F, T), in increasing L."""

    key: tuple[Any, ...]
    points: list[Point]
    tier: int
    order: int
    exclusive: bool = False
    stopped: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)


def make_series(points: Sequence[Point], slot_cpus: int | None = None) -> list[Series]:
    """Group points into series (first-appearance order within each tier); a series whose
    thread count exceeds ``slot_cpus`` is ``exclusive`` (runs alone on the host)."""
    groups: dict[tuple[Any, ...], Series] = {}
    for i, p in enumerate(points):
        key = (p["N"], p["R"], p["fs"], p["F"], p.get("T"))
        s = groups.get(key)
        if s is None:
            excl = bool(slot_cpus and p.get("T") and p["T"] > slot_cpus)
            s = groups[key] = Series(key, [], p.get("tier", 0), i, excl)
        s.points.append(p)
        s.tier = min(s.tier, p.get("tier", 0))
    for s in groups.values():
        s.points.sort(key=lambda q: q["L"])
    return sorted(groups.values(), key=lambda s: (s.tier, s.order))


# =========================================================================== rows


def _row(
    p: Point,
    r: dict[str, Any] | None,
    err: str | None,
    wall: float,
    default_threads: int | None = None,
) -> dict[str, Any]:
    n_tx, n_rx = split_links(p["L"])
    status = "failed" if err is not None else (r or {}).get("status", "ok")
    row: dict[str, Any] = {
        "point_id": point_id(p),
        "slices": "+".join(p["slices"]),
        "tier": p.get("tier", 0),
        "L": p["L"],
        "n_tx": n_tx,
        "n_rx": n_rx,
        "N": p["N"],
        "n_ris": p["R"],
        "f_s": p["fs"],
        "features": p["F"],
        "threads": p.get("T") or default_threads,
        "threads_explicit": bool(p.get("T")),
        "status": status,
        "error": err,
        "point_wall_s": wall,
    }
    if r is None:
        return row
    f = r["facts"]
    dec = r.get("decisions") or {}
    chk = r["ris_check"]
    row.update(
        {
            "page": f["page"],
            "t_u_s": f["t_u_s"],
            "fd_filter": f["fd_filter"],
            "fd_taps": f["fd_taps"],
            "mean_fd_groups": dec.get("mean_fd_groups"),
            "narrowband_fraction": dec.get("narrowband_fraction"),
            "ris_tile_paths": f["ris_tile_paths"],
            "rq": f["rq"],
            "min_ris_minus_los_db": chk["min_ris_minus_los_db"],
            "ris_path_nonzero": bool(
                chk["n_panels"] == 0 or chk["min_ris_minus_los_db"] is not None
            ),
            "rtf_median": r["rtf_median"],
            "rtf_reps": r["rtf_reps"],
            "rtf_min_worst": r["rtf_min_worst"],
            "aggregate_msps_median": r["aggregate_msps_median"],
            "reps": r["reps"],
            "rep_virtual_s": r["rep_virtual_s"],
            "rep_pages": r["rep_pages"],
            "pilot_rtf": r["pilot_rtf"],
            "pilot_virtual_s": r.get("pilot_virtual_s"),
            "rtf_floor": r.get("rtf_floor"),
            "build_s": r["build_s"],
            "verify_interp_max_rel": r["verify_interp_max_rel"],
            "late_samples": r["counters"].get("late_samples"),
            "overflow_pages": r["counters"].get("overflow_pages"),
            "stall_zero_samples": r["counters"].get("stall_zero_samples"),
            "bad_packets": r["counters"].get("bad_packets"),
            "seq_errors": r["counters"].get("seq_errors"),
            "nondeterministic_inputs": r["nondeterministic_inputs"],
            "recorded_bytes": r["recorded_bytes"],
            "rtf_source": r.get("rtf_source", "engine RtfMeter (min over RX streams)"),
            "scenario_sha256": r["scenario_sha256"],
        }
    )
    return row


def _settings(a: argparse.Namespace) -> dict[str, Any]:
    quick = bool(a.quick)
    return {
        "reps": 3,
        "target_pages": 200 if quick else 10_000,
        "rep_wall_target_s": 0.5 if quick else a.rep_wall_s,
        "min_virtual_s": "0.3",
        "warmup_s": "0.05",
        "point_wall_cap_s": a.point_cap_s,
        "rtf_floor": a.rtf_floor if a.rtf_floor > 0 else None,
        "warm_wall_max_s": 5.0 if quick else 20.0,
        "pilot_wall_max_s": 3.0 if quick else 10.0,
    }


def measure_point(p: Point, settings: dict[str, Any]) -> dict[str, Any]:
    """Measure one point in this process (:func:`bench._engine_point.measure_lockstep`)."""
    from bench._engine_point import measure_lockstep
    from bench._scenes import bench_scenario

    sc = bench_scenario(
        links=p["L"], n_cells=p["N"], n_ris=p["R"], rate_sps=p["fs"], features=p["F"]
    )
    return measure_lockstep(
        sc,
        features=p["F"],
        reps=int(settings["reps"]),
        target_pages=int(settings["target_pages"]),
        rep_wall_target_s=float(settings["rep_wall_target_s"]),
        min_virtual_s=str(settings["min_virtual_s"]),
        warmup_s=str(settings["warmup_s"]),
        point_wall_cap_s=float(settings["point_wall_cap_s"]),
        rtf_floor=settings.get("rtf_floor"),
        warm_wall_max_s=float(settings["warm_wall_max_s"]),
        pilot_wall_max_s=float(settings["pilot_wall_max_s"]),
    )


def jit_warmup(features: str = "F3") -> None:
    """Load (or compile) the numba kernels once in this process (not recorded)."""
    from bench._engine_point import measure_lockstep
    from bench._scenes import bench_scenario

    measure_lockstep(
        bench_scenario(links=1, n_cells=40, n_ris=1, rate_sps="1e6", features=features),
        features=features,
        reps=1,
        target_pages=50,
        rep_wall_target_s=0.2,
        min_virtual_s="0.1",
    )


def _stop_reason(row: dict[str, Any]) -> str | None:
    st = row["status"]
    if st == "ok":
        return None
    if st == "interrupted":
        return "interrupted"
    if st == "below_floor":
        return (
            f"series stopped at L={row['L']}: pilot RTF {row.get('pilot_rtf')} < floor "
            f"{row.get('rtf_floor')}"
        )
    return f"series stopped at L={row['L']}: {st}"


# =========================================================================== worker process


def worker_main(a: argparse.Namespace) -> int:
    """Worker mode: measure one point pinned to ``--cpus`` and write ``--worker-out``."""
    from bench._cpus import pin_self

    cpus = bench.parse_cpu_list(a.cpus) if a.cpus else None
    pin = pin_self(cpus) if cpus else {"applied": False, "note": "unpinned"}
    if cpus:
        os.environ["SREMU_CLIENT_CPUS"] = a.cpus
    threads = a.threads or (len(cpus) if cpus else None)
    bench.set_threads(threads)
    import numba

    job = json.loads(Path(a.worker_point).read_text(encoding="utf-8"))
    p, settings = job["point"], job["settings"]
    if p.get("T"):
        numba.set_num_threads(int(p["T"]))
    out: dict[str, Any] = {"pin": pin, "numba_threads": numba.get_num_threads(), "pid": os.getpid()}
    t0 = time.perf_counter()
    r = None
    err = None
    try:
        jit_warmup("F3")
        t0 = time.perf_counter()
        r = measure_point(p, settings)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    wall = time.perf_counter() - t0
    row = _row(p, r, err, wall, numba.get_num_threads())
    # every thread of the worker (numba pool, OpenBLAS pool, recorder) after the point
    masks = bench.thread_affinities()
    out["thread_affinities_end"] = masks
    if cpus and masks is not None:
        row["all_threads_in_slot"] = all(set(bench.parse_cpu_list(m)) <= set(cpus) for m in masks)
    if r is not None:
        res = r["resolved"]
        scen = Path(a.worker_scen_dir) / f"{point_id(p)}.yaml"
        scen.parent.mkdir(parents=True, exist_ok=True)
        scen.write_text(res.to_yaml(), encoding="utf-8", newline="\n")
        row["scenario_sha256"] = str(res.sha256())
        out["inputs"] = [
            {"path": i.path, "sha256": i.sha256, "data_class": i.data_class} for i in res.inputs
        ]
        out["assumptions"] = list(res.assumptions)
    out["row"] = row
    from bench._common import write_json

    write_json(Path(a.worker_out), out)
    return 0


# =========================================================================== parallel runner


def _kill_tree(p: subprocess.Popen[Any]) -> None:
    """Kill a worker and its children (F5 clients)."""
    if p.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, timeout=30
            )
        else:
            os.killpg(p.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        p.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        p.wait(timeout=30)


class ParallelRunner:
    """Runs series on CPU slots, one pinned worker process per point (module docstring)."""

    def __init__(
        self,
        run: BenchRun,
        slots: list[list[int]],
        settings: dict[str, Any],
        *,
        pinned: bool,
        budget_s: float,
        t_start: float,
        point_timeout_s: float,
    ) -> None:
        self.run = run
        self.slots = slots
        self.settings = settings
        self.pinned = pinned
        self.budget_s = budget_s
        self.t_start = t_start
        self.point_timeout_s = point_timeout_s
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []
        self.dropped: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self.running: dict[int, subprocess.Popen[Any]] = {}
        self.jobs_dir = run.path("workers/.keep").parent

    def _over_budget(self) -> bool:
        return time.perf_counter() - self.t_start > self.budget_s

    def run_point(self, p: Point, cpus: list[int], slot: int, tag: str = "") -> dict[str, Any]:
        """Run one point in a worker process pinned to ``cpus``; returns its row."""
        pid = point_id(p) + tag
        job = self.jobs_dir / f"{pid}.job.json"
        out = self.jobs_dir / f"{pid}.out.json"
        log = self.jobs_dir / f"{pid}.log"
        job.write_text(json.dumps({"point": p, "settings": self.settings}), encoding="utf-8")
        threads = len(cpus)
        cmd = [
            sys.executable,
            "-m",
            "bench.b2_lockstep",
            "--worker-point",
            str(job),
            "--worker-out",
            str(out),
            "--worker-scen-dir",
            str(self.run.path("scenarios/.keep").parent),
            "--threads",
            str(threads),
        ]
        if self.pinned:
            from bench._cpus import format_cpu_list

            cmd += ["--cpus", format_cpu_list(cpus)]
        env = dict(os.environ)
        env["NUMBA_NUM_THREADS"] = str(threads)
        kw: dict[str, Any] = {}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        t0 = time.perf_counter()
        with self.lock:
            conc0 = len(self.running)
        with log.open("wb") as fh:
            proc = subprocess.Popen(
                cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT, **kw
            )
            with self.lock:
                self.running[slot] = proc
            err = None
            try:
                rc = proc.wait(timeout=self.point_timeout_s)
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                rc = None
                err = f"timeout after {self.point_timeout_s:.0f} s (worker killed)"
            except KeyboardInterrupt:  # main-thread points (JIT warm-up, exclusive, isolation)
                _kill_tree(proc)  # never leave a pinned worker running after the suite
                raise
            finally:
                with self.lock:
                    self.running.pop(slot, None)
                    conc1 = len(self.running)
        wall = time.perf_counter() - t0
        if self.stop.is_set() and err is None and (rc != 0 or not out.exists()):
            err = "interrupted"
        if err is None and (rc != 0 or not out.exists()):
            tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
            err = f"worker exit {rc}: {tail}"
        if err is not None:
            row = _row(p, None, err, wall, p.get("T") or threads)
            if err.startswith("timeout"):
                row["status"] = "timeout"
            elif err == "interrupted":
                row["status"] = "interrupted"
        else:
            res = json.loads(out.read_text(encoding="utf-8"))
            row = res["row"]
            for i in res.get("inputs", []):
                self.run.add_input(i["path"], i["sha256"], i["data_class"])
            for t in res.get("assumptions", []):
                self.run.add_assumption(t)
            row["pinned"] = bool(res["pin"].get("applied"))
            row["affinity"] = res["pin"].get("affinity")
            row["worker_numba_threads"] = res.get("numba_threads")
            row["thread_affinities_end"] = res.get("thread_affinities_end")
        row.update(
            {
                "slot": slot,
                "slot_cpus": len(cpus),
                "worker_wall_s": wall,
                "concurrent_points_at_start": conc0,
                "concurrent_points_at_end": conc1,
            }
        )
        return row

    def _series_loop(self, slot: int, queue: list[Series]) -> None:
        cpus = self.slots[slot]
        while not self.stop.is_set():
            with self.lock:
                if not queue:
                    return
                s = queue.pop(0)
            self.run_series(s, cpus, slot)

    def run_series(self, s: Series, cpus: list[int], slot: int) -> None:
        """Run a series in increasing L with series pruning, budget and stop checks."""
        for i, p in enumerate(s.points):
            reason = None
            if self.stop.is_set():
                reason = "interrupted"
            elif self._over_budget():
                reason = "budget"
            if reason is not None:
                with self.lock:
                    self.dropped.extend({**_drop_fields(q), "reason": reason} for q in s.points[i:])
                return
            row = self.run_point(p, cpus, slot)
            with self.lock:
                self.rows.append(row)
                s.rows.append(row)
                self.run.write_csv("points.csv", self.rows)
            print(
                f"[slot {slot:2d}] {row['point_id']:36s} {row['status']:11s} "
                f"rtf={row.get('rtf_median')} pilot={row.get('pilot_rtf')} "
                f"{row['worker_wall_s']:.1f}s conc={row['concurrent_points_at_start']}",
                flush=True,
            )
            why = _stop_reason(row)
            if why is not None:
                s.stopped = why
                with self.lock:
                    self.dropped.extend(
                        {**_drop_fields(q), "reason": why} for q in s.points[i + 1 :]
                    )
                return

    def run_parallel(self, series: list[Series]) -> None:
        """Run the non-exclusive series on every slot concurrently."""
        queue = list(series)
        threads = [
            threading.Thread(target=self._series_loop, args=(k, queue), daemon=True)
            for k in range(len(self.slots))
        ]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(0.5)
        except KeyboardInterrupt:
            self.halt()
            for t in threads:
                t.join(60)
            with self.lock:
                self.dropped.extend(
                    {**_drop_fields(q), "reason": "interrupted"} for s in queue for q in s.points
                )
                queue.clear()
            raise

    def halt(self) -> None:
        """Stop scheduling and kill every running worker."""
        self.stop.set()
        with self.lock:
            procs = list(self.running.values())
        for proc in procs:
            _kill_tree(proc)


def _drop_fields(p: Point) -> dict[str, Any]:
    return {"L": p["L"], "N": p["N"], "R": p["R"], "fs": p["fs"], "F": p["F"], "T": p.get("T")}


def mark_missing(
    series: Sequence[Series],
    rows: Sequence[dict[str, Any]],
    dropped: list[dict[str, Any]],
    reason: str,
) -> None:
    """Append ``reason`` for every planned point that has neither a row nor a drop entry."""
    seen = {r["point_id"] for r in rows} | {
        point_id({**d, "slices": []}) for d in dropped if "L" in d
    }
    for s in series:
        for q in s.points:
            if point_id(q) not in seen:
                dropped.append({**_drop_fields(q), "reason": reason})


def exclusive_cpus(cores: Sequence[Any], n: int) -> list[int]:
    """Whole cores in node-major order until at least ``n`` CPUs (exclusive-phase points)."""
    out: list[int] = []
    for k in cores:
        if len(out) >= n:
            break
        out.extend(k.cpus)
    return sorted(out)


def isolation_points(rows: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Up to ``k`` ok rows matching :data:`ISOLATION_PREF` (default thread count)."""
    out = []
    for L, N, R, fs, F in ISOLATION_PREF:
        for r in rows:
            if (
                r["status"] == "ok"
                and not r.get("threads_explicit")
                and (r["L"], r["N"], r["n_ris"], r["f_s"], r["features"]) == (L, N, R, fs, F)
            ):
                out.append(r)
                break
        if len(out) >= k:
            break
    return out


# =========================================================================== main


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="bench.b2_lockstep", description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="tiny grid, short runs (smoke test)")
    ap.add_argument("--plan", choices=PLANS, default="slices")
    ap.add_argument("--slices", default=",".join(SLICES), help="plan 'slices' only")
    ap.add_argument("--budget-min", type=float, default=150.0, help="overall wall budget")
    ap.add_argument("--point-cap-s", type=float, default=150.0, help="wall cap per point")
    ap.add_argument("--rep-wall-s", type=float, default=6.0, help="target wall per repetition")
    ap.add_argument(
        "--rtf-floor",
        type=float,
        default=0.005,
        help="stop a series when a point's pilot RTF is below this (0 = never)",
    )
    ap.add_argument("--workers", default="1", help="parallel worker slots: N or 'auto'")
    ap.add_argument("--cpus-per-point", type=int, default=16, help="CPUs per worker slot")
    ap.add_argument("--reserve-cores", type=int, default=1, help="cores kept for the OS")
    ap.add_argument(
        "--isolation-check",
        type=int,
        default=None,
        help="reference points rerun alone after a parallel sweep (default 6; 0 = none)",
    )
    ap.add_argument("--point-timeout-s", type=float, default=None, help="kill a worker after")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wait-quiet-min", type=float, default=0.0, help="wait for an idle host first")
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: all)")
    ap.add_argument(
        "--point-quiet-window-s", type=float, default=1.0, help="load sample before each point"
    )
    ap.add_argument(
        "--point-quiet-wait-s", type=float, default=0.0, help="max wait for a quiet host per point"
    )
    ap.add_argument("--point-quiet-busy", type=float, default=0.30, help="quiet threshold")
    # worker mode (internal)
    ap.add_argument("--worker-point", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--worker-scen-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--cpus", default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run B2 and write its run directory."""
    bench.deterministic_env()
    a = _parse(argv)
    if a.worker_point:
        return worker_main(a)
    from bench import _cpus

    parallel = a.workers == "auto" or int(a.workers) > 1
    pinned = parallel and hasattr(os, "sched_setaffinity")
    # a parallel planner sees the whole host, whatever its own affinity (it may have been
    # started on the OS core); it then moves itself onto the reserved cores
    cores = _cpus.read_topology(cpus=_cpus.all_cpus()) if pinned else _cpus.read_topology()
    slots: list[list[int]] = []
    reserved: list[int] = []
    orchestrator_pin: dict[str, Any] | None = None
    if parallel:
        if pinned:
            slots, reserved = _cpus.plan_slots(
                cores, a.cpus_per_point, reserve_cores=a.reserve_cores
            )
            if reserved:
                orchestrator_pin = _cpus.pin_self(reserved)
        else:  # no pinning (Windows): nominal slots for bookkeeping only
            n = max(1, (os.cpu_count() or 1) // max(1, a.cpus_per_point))
            slots = [
                list(range(k * a.cpus_per_point, (k + 1) * a.cpus_per_point)) for k in range(n)
            ]
        if a.workers != "auto":
            slots = slots[: int(a.workers)]
        default_threads = len(slots[0])
    else:
        bench.set_threads(a.threads)
        import numba

        default_threads = int(numba.get_num_threads())
    host_cpus = sum(len(k.cpus) for k in cores)
    if a.plan == "server":
        points = server_points(default_threads, host_cpus if parallel else None, a.quick)
    else:
        slices = [s.strip() for s in a.slices.split(",") if s.strip()]
        points = slice_points(slices, a.quick, max_threads=default_threads)
    settings = _settings(a)
    iso_k = (6 if parallel else 0) if a.isolation_check is None else a.isolation_check
    timeout_s = a.point_timeout_s or max(600.0, 3 * a.point_cap_s + 120.0)
    spec = {
        "plan": a.plan,
        "slices": a.slices if a.plan == "slices" else None,
        "base": BASE,
        "anchors": [list(x) for x in ANCHORS] if a.plan == "server" else None,
        "quick": a.quick,
        "points": [point_id(p) for p in points],
        "features": FEATURE_TEXT,
        "settings": settings,
        "budget_min": a.budget_min,
        "numba_threads_per_point": default_threads,
        "parallel": {
            "workers": len(slots) if parallel else 1,
            "cpus_per_point": a.cpus_per_point if parallel else None,
            "reserve_cores": a.reserve_cores if parallel else None,
            "pinned": pinned,
            "point_timeout_s": timeout_s if parallel else None,
            "isolation_check": iso_k,
        },
        "point_quiet": None
        if parallel
        else {
            "window_s": a.point_quiet_window_s,
            "wait_s": a.point_quiet_wait_s,
            "max_busy": a.point_quiet_busy,
        },
    }
    caveats = [
        "Lockstep RTF: virtual time (min over RX streams) / wall time; RTF_min over "
        "complete 100 ms virtual windows (section 12.1).",
        "Repetitions are shorter than the paced protocol of section 12.3 (>= 60 s, "
        ">= 1e4 pages); every point states its virtual time and pages.",
        "The full product is pruned by the plan and by series pruning (see dropped.csv).",
        "SIM benchmark geometry; LOS and RIS paths only (no ground, walls or diffuse).",
        "F5 RTF is measured by the RX client process (includes transport and clients).",
        "below_floor rows carry only a pilot RTF (sizing estimate), not a measured RTF.",
    ]
    if parallel:
        caveats += [
            f"{len(slots)} points ran concurrently, each pinned to {default_threads} CPUs of "
            "whole cores in one NUMA node; concurrent points share L3, memory bandwidth and "
            "the power budget (see isolation.csv for the same points run alone).",
            "Background load in meta.json includes the other concurrent points.",
        ]
        if not pinned:
            caveats.append("Workers were NOT pinned (no sched_setaffinity on this platform).")
    run = BenchRun(
        SUITE,
        spec,
        out_root=a.out,
        tag=a.tag or ("quick" if a.quick else (a.plan if a.plan != "slices" else None)),
        quiet_s=1.0 if a.quick else 3.0,
        caveats=caveats,
        wait_quiet_s=a.wait_quiet_min * 60.0,
    )
    run.set_manifest(threads=default_threads)
    if parallel:
        run.write_json(
            "slots.json",
            {
                "slots": [_cpus.format_cpu_list(s) for s in slots],
                "reserved": _cpus.format_cpu_list(reserved),
                "pinned": pinned,
                "orchestrator_pin": orchestrator_pin,
                "cores": [{"node": k.node, "cpus": list(k.cpus)} for k in cores],
            },
        )
    t_start = time.perf_counter()
    series = make_series(points, default_threads if parallel else None)
    if parallel:
        rows, extra_dropped, iso_rows = _run_parallel(
            run, series, slots, cores, settings, pinned, a, t_start, timeout_s, iso_k
        )
    else:
        rows, extra_dropped = _run_sequential(run, series, settings, default_threads, a, t_start)
        iso_rows = []
    run.write_csv("points.csv", rows)
    run.write_csv("dropped.csv", dropped_points(points, a.plan) + extra_dropped)
    if iso_rows:
        run.write_csv("isolation.csv", iso_rows)
    summary = _summary(rows, points, extra_dropped, iso_rows, a.plan, time.perf_counter() - t_start)
    run.finish(summary)
    print(f"B2 written to {run.dir}")
    return 0 if not summary["points_failed"] else 1


def _run_sequential(
    run: BenchRun,
    series: list[Series],
    settings: dict[str, Any],
    default_threads: int,
    a: argparse.Namespace,
    t_start: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import numba

    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    budget_s = a.budget_min * 60.0
    todo = list(series)
    with run.interruptible():
        jit_warmup("F3")
        while todo:
            s = todo.pop(0)
            for i, p in enumerate(s.points):
                if time.perf_counter() - t_start > budget_s:
                    dropped.extend({**_drop_fields(q), "reason": "budget"} for q in s.points[i:])
                    break
                gate = (
                    run.load.wait_quiet(
                        a.point_quiet_busy,
                        a.point_quiet_window_s,
                        a.point_quiet_wait_s,
                        "point-gate",
                    )
                    if a.point_quiet_window_s > 0
                    else None
                )
                numba.set_num_threads(p.get("T") or default_threads)
                i_load = run.load.mark()
                t0 = time.perf_counter()
                r = None
                err = None
                try:
                    r = measure_point(p, settings)
                except Exception as exc:  # recorded, never silently dropped
                    err = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                wall = time.perf_counter() - t0
                numba.set_num_threads(default_threads)
                load = run.load.window_stats(i_load)
                row = _row(p, r, err, wall, default_threads)
                row.update(
                    {
                        "bg_busy_before": None if gate is None else gate["last_busy_fraction"],
                        "gate_waited_s": None if gate is None else gate["waited_s"],
                        "gate_met": None if gate is None else gate["met"],
                        "bg_busy_mean": load["mean"],
                        "bg_busy_max": load["max"],
                    }
                )
                if r is not None:
                    row["scenario_sha256"] = run.save_scenario(point_id(p), r["resolved"])
                rows.append(row)
                run.write_csv("points.csv", rows)
                print(
                    f"[{len(rows)}] {row['point_id']:36s} {row['status']:11s} "
                    f"rtf={row.get('rtf_median')} min={row.get('rtf_min_worst')} "
                    f"reps={row.get('reps')} pages={row.get('rep_pages')} {wall:.1f}s",
                    flush=True,
                )
                why = _stop_reason(row)
                if why is not None:
                    dropped.extend({**_drop_fields(q), "reason": why} for q in s.points[i + 1 :])
                    break
    if run.interrupted:
        mark_missing(series, rows, dropped, "interrupted")
    return rows, dropped


def _run_parallel(
    run: BenchRun,
    series: list[Series],
    slots: list[list[int]],
    cores: Sequence[Any],
    settings: dict[str, Any],
    pinned: bool,
    a: argparse.Namespace,
    t_start: float,
    timeout_s: float,
    iso_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pr = ParallelRunner(
        run,
        slots,
        settings,
        pinned=pinned,
        budget_s=a.budget_min * 60.0,
        t_start=t_start,
        point_timeout_s=timeout_s,
    )
    shared = [s for s in series if not s.exclusive]
    excl = [s for s in series if s.exclusive]
    iso_rows: list[dict[str, Any]] = []
    with run.interruptible():
        # populate the numba cache once before many processes load it
        warm = {"L": 1, "N": 40, "R": 1, "fs": "1e6", "F": "F3", "T": None, "slices": ["warm"]}
        w = pr.run_point(warm, slots[0], 0, tag="_jitwarm")
        print(f"[b2] JIT warm-up worker: {w['status']} {w['worker_wall_s']:.1f}s", flush=True)
        pr.run_parallel(shared)
        for s in excl:  # alone on the host, whole cores in node-major order
            cpus = exclusive_cpus(cores, int(s.points[0]["T"])) if pinned else slots[0]
            pr.run_series(s, cpus, 0)
        if iso_k > 0 and not pr.stop.is_set():
            for ref in isolation_points(pr.rows, iso_k):
                if time.perf_counter() - t_start > a.budget_min * 90.0:
                    break  # 50 % over the budget: skip the remaining checks
                p = {
                    "L": ref["L"],
                    "N": ref["N"],
                    "R": ref["n_ris"],
                    "fs": ref["f_s"],
                    "F": ref["features"],
                    "T": None,
                    "slices": ["isolation"],
                }
                iso = pr.run_point(p, slots[0], 0, tag="_iso")
                iso_rows.append(
                    {
                        "point_id": ref["point_id"],
                        "rtf_parallel": ref.get("rtf_median"),
                        "rtf_isolated": iso.get("rtf_median"),
                        "ratio_isolated_over_parallel": (
                            iso["rtf_median"] / ref["rtf_median"]
                            if iso.get("rtf_median") and ref.get("rtf_median")
                            else None
                        ),
                        "rtf_min_parallel": ref.get("rtf_min_worst"),
                        "rtf_min_isolated": iso.get("rtf_min_worst"),
                        "concurrent_points_parallel": ref.get("concurrent_points_at_start"),
                        "status_isolated": iso["status"],
                        "slot_cpus": len(slots[0]),
                    }
                )
                print(f"[b2] isolation {ref['point_id']}: {iso_rows[-1]}", flush=True)
    if run.interrupted:
        pr.halt()
        mark_missing(series, pr.rows, pr.dropped, "interrupted")
    return pr.rows, pr.dropped, iso_rows


def _summary(
    rows: list[dict[str, Any]],
    points: list[Point],
    extra_dropped: list[dict[str, Any]],
    iso_rows: list[dict[str, Any]],
    plan: str,
    wall_s: float,
) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] in ("failed", "timeout")]
    interrupted = sum(1 for r in rows if r["status"] == "interrupted")
    reasons: dict[str, int] = {}
    for d in extra_dropped:
        key = d["reason"].split(":")[0] if d["reason"].startswith("series stopped") else d["reason"]
        reasons[key] = reasons.get(key, 0) + 1
    ratios = [
        r["ratio_isolated_over_parallel"] for r in iso_rows if r.get("ratio_isolated_over_parallel")
    ]
    return {
        "suite": SUITE,
        "plan": plan,
        "points_planned": len(points),
        "points_run": len(rows),
        "points_ok": len(ok),
        "points_below_floor": sum(1 for r in rows if r["status"] == "below_floor"),
        "points_failed": len(failed),
        "points_interrupted": interrupted,
        "failed_point_ids": [r["point_id"] for r in failed],
        "points_not_in_plan": len(dropped_points(points, plan)),
        "runtime_dropped": reasons,
        "wall_s": wall_s,
        "base_point_rtf": next(
            (
                r["rtf_median"]
                for r in ok
                if (r["L"], r["N"], r["n_ris"], r["f_s"], r["features"]) == (1, 240, 1, "1e6", "F1")
                and not r["threads_explicit"]
            ),
            None,
        ),
        "thread_scaling_rtf": {
            f"L{r['L']}_T{r['threads']}": r["rtf_median"]
            for r in ok
            if (r["N"], r["n_ris"], r["f_s"], r["features"]) == (240, 1, "1e6", "F1")
            and r["L"] in (1, 8, 64, 128)
        },
        "max_L_rtf_ge_1": {
            f"N{n}": max(
                (
                    r["L"]
                    for r in ok
                    if "S1" in r["slices"] and r["N"] == n and (r["rtf_median"] or 0) >= 1.0
                ),
                default=None,
            )
            for n in CELLS
        },
        # the RIS check raises inside the point, so a zero RIS term shows up as a failed row
        "ris_check_failures": [
            r["point_id"]
            for r in rows
            if r["status"] == "failed"
            and any(s in str(r.get("error")) for s in ("RIS term", "no RIS path"))
        ],
        "all_ris_paths_nonzero": all(
            r.get("ris_path_nonzero", True) for r in rows if r["status"] in ("ok", "below_floor")
        )
        and not any(
            r["status"] == "failed"
            and any(s in str(r.get("error")) for s in ("RIS term", "no RIS path"))
            for r in rows
        ),
        "points_with_threads_outside_slot": [
            r["point_id"] for r in rows if r.get("all_threads_in_slot") is False
        ],
        "lateness_or_loss_nonzero": [
            r["point_id"]
            for r in ok
            if (r.get("late_samples") or 0) or (r.get("overflow_pages") or 0)
        ],
        "isolation_ratio": {
            "n": len(ratios),
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
        },
    }


if __name__ == "__main__":
    sys.exit(main())
