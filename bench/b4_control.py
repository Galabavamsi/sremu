"""B4: control loop timing (DESIGN.md sections 4.9, 12.1, 12.2).

Usage::

    python -m bench.b4_control --part lockstep [--quick] [--out DIR]
    python -m bench.b4_control --part paced [--rates 10,100,1000,10000] [--reps 3]
                               [--warmup-s 5] [--measure-s 60] [--out DIR]

Scene: the B2 base scene with ``--links`` L links (default 2: 1 TX, 2 RX), N = 240, one
panel, 1 MS/s, F1. Two links keep the paced engine well inside its real-time budget, so the
control-loop numbers are not dominated by a lagging engine; every paced repetition states
whether it was **sustainable** (section 12.1: miss rate < 1e-3 at 1 ms slack, zero
``late_samples`` and ``overflow_pages`` after the warm-up).

**Lockstep part** (laptop): command rate r in {10, 100, 1000, 10000} Hz.

* ``inproc``: WP9's ``Urap(p=0.5)`` registered in-process as a barrier controller with
  period 1/r and L_min = 0.1 ms (latency L_min + U[0, L_min]);
* ``remote``: the same controller in a separate process over the control socket
  (section 8.5 barrier protocol; :mod:`bench._ctl_client` ``barrier``);
* ``none``: the same scene without a controller (reference RTF).

Reported: commands, statuses, ``lateness_samples`` on every RX (section 4.9: must be 0;
the suite fails otherwise, and also when the zero is vacuous: ``lockstep_check_ok`` needs
>= 90 % of the expected commands, all applied, one lateness value per command and RX),
RTF with and without the loop, and the lockstep wall-clock
"visible latency" ``t_wall_visible - t_wall_recv`` (how long the engine took to reach the
switch sample; not a real-time latency).

**Paced part** (Linux job ``b4`` on a quiet c7i.8xlarge; on Windows labelled "Windows, not
representative"): the engine runs paced with async coupling, served over the control
socket; an external process (:mod:`bench._ctl_client` ``async``) sends one
``ris.set_config`` every 1/r wall seconds, timed (lead 2 ms) or ``asap``. Per repetition:
``--warmup-s`` warm-up, then ``--measure-s`` measured (commands issued during the warm-up
are excluded). Reported: lateness distribution (samples) per RX, command visible latency
``t_wall_visible - t_wall_recv`` percentiles, statuses, client round-trip percentiles,
deadline misses and page latency of the paced run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Any

import bench
from bench._common import REPO, BenchRun, percentiles

SUITE = "b4"
L_MIN = Fraction(1, 10_000)
RATES = (10, 100, 1000, 10_000)
SCENE_LINKS = 2


def _scene(
    duration_s: Fraction, mode: str = "lockstep", coupling: str = "barrier"
) -> dict[str, Any]:
    from bench._scenes import bench_scenario

    return bench_scenario(
        links=SCENE_LINKS,
        n_cells=240,
        n_ris=1,
        rate_sps="1e6",
        features="F1",
        duration_s=str(duration_s),
        mode=mode,
        coupling=coupling,
        name="b4_control",
    )


def _record_stats(recs: list[Any], src: str, t_from: Fraction | None = None) -> dict[str, Any]:
    mine = [
        r
        for r in recs
        if r.src == src and (t_from is None or (r.t_req is not None and r.t_req >= t_from))
    ]
    status: dict[str, int] = {}
    late: list[int] = []
    late_by_rx: dict[str, list[int]] = {}
    vis: list[float] = []
    for r in mine:
        status[r.status] = status.get(r.status, 0) + 1
        for rx, v in (r.lateness_samples or {}).items():
            late.append(int(v))
            late_by_rx.setdefault(rx, []).append(int(v))
        if r.t_wall_visible_ns is not None and r.t_wall_recv_ns is not None:
            vis.append((r.t_wall_visible_ns - r.t_wall_recv_ns) * 1e-3)
    return {
        "commands": len(mine),
        "status": status,
        "lateness_samples": percentiles(late, (50, 99, 99.9)) if late else {},
        "lateness_nonzero": sum(1 for v in late if v != 0),
        "lateness_max": max(late) if late else None,
        "lateness_by_rx_max": {k: max(v) for k, v in late_by_rx.items()},
        "visible_latency_us": percentiles(vis, (50, 90, 99, 99.9)) if vis else {},
    }


# =========================================================================== lockstep


def lockstep_point(rate_hz: int, variant: str, duration: Fraction) -> dict[str, Any]:
    """One lockstep run: ``variant`` in none | inproc | remote."""
    from bench._engine_point import _resolve, build_engine
    from bench._scenes import ris_path_check
    from sremu.contracts import LatencyModel
    from sremu.control.builtin import Urap

    res = _resolve(_scene(duration))
    eng = build_engine(res, "F1")
    chk = ris_path_check(eng.model)
    n_ctl = eng.commands.specs["ris0"].n_controls
    out: dict[str, Any] = {"rate_hz": rate_hz, "variant": variant, "virtual_s": float(duration)}
    t0 = time.perf_counter()
    if variant == "remote":
        from sremu.transport.control_server import ControlServer, ServerConfig

        srv = ControlServer(
            eng,
            ServerConfig(
                control_endpoint="tcp://127.0.0.1:0",
                pub_endpoint=None,
                tx_endpoint="tcp://127.0.0.1:0",
                start_paused=True,
                max_command_rate_hz=1e5,
            ),
        ).start()
        try:
            cmd = [
                sys.executable,
                "-m",
                "bench._ctl_client",
                "barrier",
                "--endpoint",
                str(srv.endpoints["control"]),
                "--rate-hz",
                str(rate_hz),
                "--l-min",
                str(L_MIN),
                "--idle-s",
                "3",
            ]
            p = subprocess.Popen(
                cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            t_run = time.perf_counter()
            while not srv.is_done():
                if p.poll() is not None and not srv.is_done():
                    break
                time.sleep(0.01)
            wall = time.perf_counter() - t_run
            o, e = p.communicate(timeout=120)
            if srv.engine_error is not None:
                raise RuntimeError(f"engine error: {srv.engine_error}")
            out["client"] = (
                json.loads(o.strip().splitlines()[-1]) if o.strip() else {"stderr": e[-2000:]}
            )
        finally:
            srv.stop()
        s = eng.summary()
        # the engine's RTF meter starts at the first computed page (after sim.resume); the
        # outer wall time also contains the client's start-up while the server is paused
        out["wall_s_incl_client_start"] = wall
        out["wall_s"] = s.rtf.get("wall_s")
        out["rtf"] = s.rtf.get("rtf")
        out["rtf_engine_meter"] = s.rtf.get("rtf")
        out["rtf_min"] = s.rtf.get("rtf_min")
    else:
        if variant == "inproc":
            eng.register_controller(
                Urap(
                    0.5,
                    name="b4",
                    period_s=Fraction(1, rate_hz),
                    min_latency_s=L_MIN,
                    latency=LatencyModel(L_MIN, f"uniform:0,{float(L_MIN)!r}"),
                )
            )
        s = eng.run(until_s=duration)
        out["wall_s"] = time.perf_counter() - t0
        out["rtf"] = s.rtf.get("rtf")
        out["rtf_engine_meter"] = s.rtf.get("rtf")
        out["rtf_min"] = s.rtf.get("rtf_min")
    out.update(_record_stats(list(eng.records), "b4"))
    out["n_controls"] = n_ctl
    out["n_rx"] = len(eng.rx)
    out["expected_commands"] = 0 if variant == "none" else int(rate_hz * duration)
    out["nondeterministic_inputs"] = eng.nondeterministic_inputs
    out["min_ris_minus_los_db"] = chk["min_ris_minus_los_db"]
    out["scenario_sha256"] = res.sha256()
    eng.close()
    return out


def lockstep_problems(r: dict[str, Any]) -> list[str]:
    """Why a lockstep row does not demonstrate zero lateness (empty: it does).

    Zero lateness is only evidence when commands were really issued and applied: a
    controller row needs at least 90 % of its expected commands (one per period), every
    command ``applied``, one lateness value per command and RX, and all of them 0; the
    reference row (``none``) must have no command at all."""
    out: list[str] = []
    n = int(r.get("commands") or 0)
    if r["variant"] == "none":
        if n:
            out.append(f"{n} commands in the reference run")
        return out
    exp = int(r.get("expected_commands") or 0)
    if n < max(1, int(0.9 * exp)):
        out.append(f"{n} commands, expected about {exp}")
    other = {k: v for k, v in (r.get("status") or {}).items() if k != "applied"}
    if other:
        out.append(f"statuses other than applied: {other}")
    n_late = int((r.get("lateness_samples") or {}).get("n", 0))
    if n_late != n * int(r.get("n_rx") or 0):
        out.append(f"{n_late} lateness values for {n} commands x {r.get('n_rx')} RX")
    if r.get("lateness_nonzero"):
        out.append(f"{r['lateness_nonzero']} non-zero lateness values (max {r['lateness_max']})")
    return out


def run_lockstep(run: BenchRun, quick: bool) -> dict[str, Any]:
    from bench._engine_point import _resolve, prewarm

    prewarm(_resolve(_scene(Fraction(1))), "F1")  # numba kernels hot before any timing
    rows = []
    rates = (10, 1000) if quick else RATES
    with run.interruptible():
        for rate in rates:
            n_cmd = 50 if quick else 200
            duration = max(Fraction(1, 4) if quick else Fraction(1), Fraction(n_cmd, rate))
            for variant in ("none", "inproc", "remote"):
                r = lockstep_point(rate, variant, duration)
                rows.append(r)
                print(
                    f"lockstep {variant:6s} {rate:6d} Hz: rtf={r['rtf']:.3f} "
                    f"commands={r['commands']} late_nonzero={r['lateness_nonzero']} "
                    f"status={r['status']}",
                    flush=True,
                )
    flat = []
    for r in rows:
        flat.append(
            {
                k: (json.dumps(v) if isinstance(v, dict) else v)
                for k, v in r.items()
                if k not in ("client",)
            }
        )
    run.write_csv("lockstep.csv", flat)
    run.write_json("lockstep.json", rows)
    bad = [r for r in rows if r["variant"] != "none" and r["lateness_nonzero"]]
    problems = {
        f"{r['variant']}@{r['rate_hz']}Hz": lockstep_problems(r)
        for r in rows
        if lockstep_problems(r)
    }
    return {
        "lockstep_points": len(rows),
        "lockstep_all_lateness_zero": not bad,
        "lockstep_check_ok": bool(rows) and not problems and not run.interrupted,
        "lockstep_check_problems": problems,
        "lockstep_commands_total": sum(r["commands"] for r in rows),
        "lockstep_rtf": {f"{r['variant']}@{r['rate_hz']}Hz": r["rtf"] for r in rows},
    }


# =========================================================================== paced


def paced_point(
    rate_hz: int, mode: str, warmup_s: Fraction, measure_s: Fraction, rep: int
) -> dict[str, Any]:
    """One paced + async repetition with an external command process."""
    from bench._engine_point import _resolve, _restart_paced_stats, build_engine, prewarm
    from bench._scenes import ris_path_check
    from sremu.transport.control_server import ControlServer, ServerConfig

    total = warmup_s + measure_s
    res = _resolve(_scene(total, mode="paced", coupling="async"))
    prewarm(res, "F1")
    eng = build_engine(res, "F1")
    chk = ris_path_check(eng.model)
    n_ctl = eng.commands.specs["ris0"].n_controls
    g = eng.groups[0]
    state: dict[str, Any] = {}
    eng.on_page(_restart_paced_stats(eng, int(warmup_s * g.rate), state))
    srv = ControlServer(
        eng,
        ServerConfig(
            control_endpoint="tcp://127.0.0.1:0",
            pub_endpoint=None,
            tx_endpoint="tcp://127.0.0.1:0",
            start_paused=True,
            max_command_rate_hz=1e5,
        ),
    ).start()
    p: subprocess.Popen[str] | None = None
    try:
        cmd = [
            sys.executable,
            "-m",
            "bench._ctl_client",
            "async",
            "--endpoint",
            str(srv.endpoints["control"]),
            "--rate-hz",
            str(rate_hz),
            "--duration-s",
            str(float(total) - 0.5),
            "--mode",
            mode,
            "--lead-s",
            "2e-3",
            "--n-controls",
            str(n_ctl),
            "--seed",
            str(rep + 1),
            "--resume",
        ]
        p = subprocess.Popen(
            cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        t0 = time.perf_counter()
        while not srv.is_done():
            time.sleep(0.05)
            if time.perf_counter() - t0 > float(total) * 10 + 300:
                raise RuntimeError("paced run did not finish")
        wall = time.perf_counter() - t0
        o, e = p.communicate(timeout=120)
        client = json.loads(o.strip().splitlines()[-1]) if o.strip() else {"stderr": e[-2000:]}
        if srv.engine_error is not None:
            raise RuntimeError(f"engine error: {srv.engine_error}")
    except BaseException:
        if p is not None and p.poll() is None:
            p.kill()
        srv.stop()
        eng.close()  # a failed or interrupted repetition must not leave its engine behind
        raise
    srv.stop()
    s = eng.summary()
    paced = s.paced or {}
    c0 = state.get("counters0", {})
    counters = {k: int(v) - int(c0.get(k, 0)) for k, v in s.counters.items()}
    miss_1ms = (paced.get("miss_rate") or {}).get("1ms")
    sustainable = bool(
        miss_1ms is not None
        and miss_1ms < 1e-3
        and counters.get("late_samples", 0) == 0
        and counters.get("overflow_pages", 0) == 0
    )
    stats = _record_stats(list(eng.records), "b4async", t_from=warmup_s)
    eng.close()
    return {
        "rate_hz": rate_hz,
        "mode": mode,
        "rep": rep,
        "warmup_s": float(warmup_s),
        "measure_s": float(measure_s),
        "warmup_restart_ok": bool(state.get("done")),
        "wall_s": wall,
        **stats,
        "client": client,
        "pages": paced.get("pages"),
        "miss_rate": paced.get("miss_rate"),
        "page_latency_us": paced.get("latency_us"),
        "utilization": paced.get("utilization"),
        "counters": counters,
        "sustainable": sustainable,
        # commands whose request time lies in the measured virtual window, per virtual
        # second; far below rate_hz when the engine lags (its virtual time crawls while the
        # client's wall-clock schedule runs out)
        "commands_per_virtual_s": stats["commands"] / float(measure_s),
        "engine_bound": stats["commands"] < 0.5 * rate_hz * float(measure_s),
        "n_controls": n_ctl,
        "min_ris_minus_los_db": chk["min_ris_minus_los_db"],
        "scenario_sha256": res.sha256(),
    }


def run_paced(
    run: BenchRun,
    rates: list[int],
    reps: int,
    warm: Fraction,
    meas: Fraction,
    modes: list[str],
    budget_s: float,
) -> dict[str, Any]:
    rows = []
    dropped = []
    failed: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    virt = float(warm + meas)
    slow = 1.2  # wall / virtual of a repetition; raised to what the runs show (a lagging
    # engine stretches a paced run far beyond its virtual length)
    with run.interruptible():
        for rate in rates:
            for mode in modes:
                for rep in range(reps):
                    if time.perf_counter() - t_start + virt * slow > budget_s:
                        dropped.append(
                            {"rate_hz": rate, "mode": mode, "rep": rep, "reason": "budget"}
                        )
                        continue
                    try:
                        r = paced_point(rate, mode, warm, meas, rep)
                    except Exception as exc:  # recorded; the suite goes on
                        traceback.print_exc()
                        failed.append(
                            {
                                "rate_hz": rate,
                                "mode": mode,
                                "rep": rep,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue
                    slow = max(slow, 1.1 * float(r["wall_s"]) / virt)
                    rows.append(r)
                    print(
                        f"paced {mode:5s} {rate:6d} Hz rep {rep}: commands={r['commands']} "
                        f"late p99={r['lateness_samples'].get('p99')} max={r['lateness_max']} "
                        f"vis p99={r['visible_latency_us'].get('p99')} us miss={r['miss_rate']} "
                        f"sustainable={r['sustainable']}",
                        flush=True,
                    )
                    run.write_json("paced.json", rows)
    flat = [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in rows]
    run.write_csv("paced.csv", flat)
    if dropped:
        run.write_csv("paced_dropped.csv", dropped)
    if failed:
        run.write_csv("paced_failed.csv", failed)
    by: dict[str, Any] = {}
    for rate in rates:
        for mode in modes:
            rs = [r for r in rows if r["rate_hz"] == rate and r["mode"] == mode]
            if not rs:
                continue
            p99 = [r["lateness_samples"].get("p99") for r in rs if r["lateness_samples"]]
            vis = [r["visible_latency_us"].get("p99") for r in rs if r["visible_latency_us"]]
            ach = [
                r["client"]["achieved_rate_hz"]
                for r in rs
                if isinstance(r.get("client"), dict) and r["client"].get("achieved_rate_hz")
            ]
            by[f"{mode}@{rate}Hz"] = {
                "reps": len(rs),
                # the client waits for each reply (REQ/REP): a slow engine caps its rate
                "achieved_rate_hz_median": statistics.median(ach) if ach else None,
                "wall_over_virtual_max": max(float(r["wall_s"]) / virt for r in rs),
                "lateness_p99_samples_median": statistics.median(p99) if p99 else None,
                "lateness_max_samples_worst": max((r["lateness_max"] or 0) for r in rs),
                "visible_latency_p99_us_worst": max(vis) if vis else None,
                "late_applied_total": sum(r["status"].get("late_applied", 0) for r in rs),
                "sustainable_reps": sum(1 for r in rs if r["sustainable"]),
                "commands_total": sum(r["commands"] for r in rs),
                "commands_per_virtual_s_median": statistics.median(
                    r["commands_per_virtual_s"] for r in rs
                ),
                "engine_bound_reps": sum(1 for r in rs if r["engine_bound"]),
            }
    pages = [int(r["pages"] or 0) for r in rows]
    w, m = float(warm), float(meas)
    return {
        "paced": by,
        "paced_all_reps_sustainable": bool(rows) and all(r["sustainable"] for r in rows),
        "paced_dropped": len(dropped),
        "paced_failed": failed,
        "protocol_12_3": {
            "warmup_s": w,
            "measure_s": m,
            "reps": reps,
            "min_pages_per_rep": min(pages) if pages else None,
            "meets_12_3": bool(w >= 5 and m >= 60 and reps >= 3 and pages and min(pages) >= 10_000),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run B4 and write its run directory."""
    global SCENE_LINKS
    bench.deterministic_env()
    ap = argparse.ArgumentParser(prog="bench.b4_control", description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("lockstep", "paced"), default="lockstep")
    ap.add_argument("--links", type=int, default=SCENE_LINKS, help="scene links (power of 2)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rates", default=",".join(str(r) for r in RATES))
    ap.add_argument("--modes", default="timed,asap")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup-s", default="5")
    ap.add_argument("--measure-s", default="60")
    ap.add_argument("--budget-min", type=float, default=85.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wait-quiet-min", type=float, default=0.0, help="wait for an idle host first")
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: all)")
    a = ap.parse_args(argv)
    bench.set_threads(a.threads)
    SCENE_LINKS = int(a.links)
    paced = a.part == "paced"
    rates = [int(x) for x in a.rates.split(",") if x]
    modes = [m for m in a.modes.split(",") if m]
    spec = {
        "threads": a.threads,
        "part": a.part,
        "quick": a.quick,
        "scene": f"B2 base, L={a.links}, N=240, 1 panel, 1 MS/s, F1",
        "links": a.links,
        "controller": "Urap(p=0.5), L_min=1e-4 s, latency L_min+U[0,L_min] (lockstep)",
        "rates_hz": rates if paced else list((10, 1000) if a.quick else RATES),
        "modes": modes if paced else ["none", "inproc", "remote"],
        "reps": a.reps if paced else 1,
        "warmup_s": a.warmup_s if paced else None,
        "measure_s": a.measure_s if paced else None,
    }
    caveats = [
        "Lockstep 'visible latency' is wall time until the engine computed the switch "
        "sample; it is not a real-time latency.",
    ]
    if paced:
        caveats.append("Paced numbers are meaningful only on the stated Linux host.")
    run = BenchRun(
        SUITE,
        spec,
        out_root=a.out,
        tag=a.tag or (a.part + ("_quick" if a.quick else "")),
        mode="paced" if paced else "lockstep",
        coupling="async" if paced else "barrier",
        paced=paced,
        caveats=caveats,
        quiet_s=1.0 if a.quick else 3.0,
        wait_quiet_s=a.wait_quiet_min * 60.0,
    )
    if paced:
        summary = run_paced(
            run,
            rates,
            a.reps,
            Fraction(a.warmup_s),
            Fraction(a.measure_s),
            modes,
            a.budget_min * 60.0,
        )
    else:
        summary = run_lockstep(run, a.quick)
    summary["suite"] = SUITE
    summary["part"] = a.part
    run.finish(summary)
    print(f"B4 written to {run.dir}")
    if not paced and not summary["lockstep_check_ok"]:
        return 1
    if paced and summary["paced_failed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
