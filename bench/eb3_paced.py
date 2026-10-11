"""E-B3 timing demonstration: a real external TDMA controller over ZMQ, paced + async
(DESIGN.md section 11 E-B3, SHOULD; Linux job ``eb3`` of ``bench/aws/run_job.py``).

Usage::

    python -m bench.eb3_paced [--leads 5e-4,2e-3,5e-3] [--reps 3] [--warmup-s 5]
                              [--measure-s 60] [--budget-min 85] [--quick] [--out DIR]

Scene: ``scenarios/lab_like_A.yaml`` (E-B geometry, SIM "lab-like"), its in-process QPSK
packet traffic (64 + 512 symbols at 4 samples/symbol, 0.5 ms guard: packet period 2804
samples at 1 MS/s), run **paced with async coupling**, scenario controllers removed. The
controller is a separate OS process (:mod:`bench._ctl_client` ``tdma``): for every guard m
it sends, ``lead`` seconds of wall time before the guard centre, a timed
``ris.set_config`` at the guard centre with the configuration of packet m+1's target
(alternating RX1/RX2; the exact finite-alphabet optimum of each RX's link at t = 0,
:func:`sremu.physics.optimize.exact_alphabet_optimum` on ``csi_view``). Late commands follow
``apply_on_arrival``.

Measured over the window after the warm-up: the lateness distribution (samples) of every
command on each RX, statuses, the command visible latency, the **switch-after-packet-start
rate** (packets m+1 whose target RX applied the switch of guard m only at or after the
packet's first sample, i.e. the packet began with the previous configuration) and the
**switch-inside-packet rate** (the subset whose switch sample lies inside the packet body),
each with a Clopper-Pearson 95% interval. These are timing outcomes, not a PER: PER needs
demodulation; the run records the command timing so a PER can be computed offline by
replay (section 4.12) with the E-B receiver.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import bench
from bench._common import REPO, BenchRun, percentiles

SUITE = "eb3"
SCENARIO = REPO / "scenarios" / "lab_like_A.yaml"
RXS = ("rx1.rx0", "rx2.rx0")


def _scenario(total_s: Fraction) -> dict[str, Any]:
    from sremu.scenario.load import read_document

    d = read_document(SCENARIO)
    d["controllers"] = []
    d["run"] = {
        "mode": "paced",
        "coupling": "async",
        "duration_s": str(total_s),
        "late_policy": "apply_on_arrival",
    }
    d["record"] = {"iq": False, "events": True, "metrics": True}
    d["name"] = "eb3_paced_controller"
    return d


def _packet_geometry(d: dict[str, Any]) -> dict[str, int]:
    """Packet body, guard and period in samples of the QPSK TX stream (and its rate)."""
    from sremu.timebase import to_ticks

    t = next(x for x in d["traffic"] if x["generator"] == "qpsk_packets")
    node, sname = str(t["stream"]).split(".")
    st = next(s for n in d["nodes"] if n["id"] == node for s in n["streams"] if s["name"] == sname)
    rate = Fraction(str(st["rate_sps"]))
    p = t["params"]
    sps, pre, pay = int(p["sps"]), int(p["preamble_syms"]), int(p["payload_syms"])
    guard = to_ticks(Fraction(str(p["guard_s"])), rate)  # as traffic_from_scenario
    body = (pre + pay) * sps
    return {"body": body, "guard": guard, "period": body + guard, "rate_sps": int(rate)}


def _optimum_configs(eng: Any) -> list[list[int]]:
    from sremu.physics.optimize import exact_alphabet_optimum

    snap = eng.model.snapshot(Fraction(0))
    out = []
    for rx in RXS:
        cv = eng.model.csi_view(snap, f"ap.tx0->{rx}")
        _, s = exact_alphabet_optimum(cv.h_direct, cv.cascade["ris0"])
        out.append([int(x) for x in s])
    return out


def run_once(
    lead_s: str, warm: Fraction, meas: Fraction, rep: int, run_dir: Path
) -> dict[str, Any]:
    from bench._engine_point import _resolve, _restart_paced_stats, build_engine, prewarm
    from sremu.analysis.stats import clopper_pearson
    from sremu.transport.control_server import ControlServer, ServerConfig

    total = warm + meas
    d = _scenario(total)
    geo = _packet_geometry(d)
    res = _resolve(d)
    prewarm(res, "F1")
    eng = build_engine(res, "F1")
    configs = _optimum_configs(eng)
    cfg_path = run_dir / "eb3_configs.json"
    cfg_path.write_text(json.dumps(configs), encoding="utf-8")
    g = eng.groups[0]
    state: dict[str, Any] = {}
    eng.on_page(_restart_paced_stats(eng, int(warm * g.rate), state))
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
    rate = Fraction(geo["rate_sps"])
    period_s = Fraction(geo["period"]) / rate
    centre_s = Fraction(geo["body"] + geo["guard"] // 2) / rate
    try:
        cmd = [
            sys.executable,
            "-m",
            "bench._ctl_client",
            "tdma",
            "--endpoint",
            str(srv.endpoints["control"]),
            "--configs",
            str(cfg_path),
            "--period-s",
            str(period_s),
            "--guard-centre-s",
            str(centre_s),
            "--lead-s",
            lead_s,
            "--duration-s",
            str(float(total) - 0.2),
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
        o, e = p.communicate(timeout=120)
        client = json.loads(o.strip().splitlines()[-1]) if o.strip() else {"stderr": e[-2000:]}
        if srv.engine_error is not None:
            raise RuntimeError(f"engine error: {srv.engine_error}")
    finally:
        srv.stop()
    s = eng.summary()
    recs = [r for r in eng.records if r.src == "eb3" and r.t_req is not None and r.t_req >= warm]
    eng.close()
    per, body = geo["period"], geo["body"]
    late: dict[str, list[int]] = {rx: [] for rx in RXS}
    inside = 0
    after_start = 0
    counted = 0
    status: dict[str, int] = {}
    vis = []
    compact = []
    for r in recs:
        status[r.status] = status.get(r.status, 0) + 1
        m = int(r.cmd_id.split(":")[1])
        target = RXS[(m + 1) % 2]
        for rx in RXS:
            late[rx].append(int(r.lateness_samples.get(rx, 0)))
        k = r.k_applied.get(target)
        if k is not None:
            counted += 1
            start_next = (m + 1) * per  # first sample of packet m + 1 (its target's packet)
            if k >= start_next:
                after_start += 1  # packet m + 1 began with the previous configuration
                if k < start_next + body:
                    inside += 1  # ... and the switch lands inside its body
        if r.t_wall_visible_ns is not None and r.t_wall_recv_ns is not None:
            vis.append((r.t_wall_visible_ns - r.t_wall_recv_ns) * 1e-3)
        compact.append(
            [m, r.k_req.get(RXS[0]), *(int(r.lateness_samples.get(x, 0)) for x in RXS), r.status]
        )
    lo, hi = clopper_pearson(inside, counted) if counted else (None, None)
    lo2, hi2 = clopper_pearson(after_start, counted) if counted else (None, None)
    paced = s.paced or {}
    return {
        "lead_s": lead_s,
        "rep": rep,
        "commands": len(recs),
        "status": status,
        "lateness_samples": {rx: percentiles(v, (50, 99, 99.9)) for rx, v in late.items()},
        "lateness_nonzero": {rx: sum(1 for x in v if x) for rx, v in late.items()},
        "visible_latency_us": percentiles(vis, (50, 99, 99.9)),
        "switch_inside_packet": inside,
        "packets_counted": counted,
        "switch_inside_packet_rate": inside / counted if counted else None,
        "switch_inside_packet_ci95": [lo, hi],
        "switch_after_packet_start": after_start,
        "switch_after_packet_start_rate": after_start / counted if counted else None,
        "switch_after_packet_start_ci95": [lo2, hi2],
        "client": client,
        "pages": paced.get("pages"),
        "miss_rate": paced.get("miss_rate"),
        "page_latency_us": paced.get("latency_us"),
        "counters": dict(s.counters),
        "warmup_restart_ok": bool(state.get("done")),
        "packet_geometry": geo,
        "scenario_sha256": res.sha256(),
        "_compact": compact,
        "_resolved": res,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the E-B3 timing demonstration and write its run directory."""
    bench.deterministic_env()
    ap = argparse.ArgumentParser(prog="bench.eb3_paced", description=__doc__.splitlines()[0])
    ap.add_argument("--leads", default="5e-4,2e-3,5e-3")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup-s", default="5")
    ap.add_argument("--measure-s", default="60")
    ap.add_argument("--budget-min", type=float, default=85.0)
    ap.add_argument("--quick", action="store_true", help="1 lead, 1 rep, 1 s + 3 s")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wait-quiet-min", type=float, default=0.0, help="wait for an idle host first")
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: all)")
    ap.add_argument(
        "--omp-wait-policy",
        choices=("passive", "active", "env"),
        default="passive",
        help="OMP_WAIT_POLICY for numba's OpenMP layer (env: leave the environment)",
    )
    a = ap.parse_args(argv)
    # served paced engine: idle OpenMP workers sleep at once (see bench.omp_wait_policy)
    wait_policy = bench.omp_wait_policy(None if a.omp_wait_policy == "env" else a.omp_wait_policy)
    bench.set_threads(a.threads)
    if a.quick:
        a.leads, a.reps, a.warmup_s, a.measure_s = "2e-3", 1, "1", "3"
    leads = [x for x in a.leads.split(",") if x]
    warm, meas = Fraction(a.warmup_s), Fraction(a.measure_s)
    spec = {
        "threads": a.threads,
        "omp_wait_policy": wait_policy,
        "scenario": "scenarios/lab_like_A.yaml (controllers removed; paced + async)",
        "leads_s": leads,
        "reps": a.reps,
        "warmup_s": a.warmup_s,
        "measure_s": a.measure_s,
        "controller": "bench._ctl_client tdma (separate process, timed at guard centres)",
    }
    run = BenchRun(
        SUITE,
        spec,
        out_root=a.out,
        tag=a.tag or ("quick" if a.quick else None),
        mode="paced",
        coupling="async",
        paced=True,
        caveats=[
            "Timing demonstration: switch-inside-packet rate is not a PER.",
            "SIM lab-like geometry (not the lab's); exact optima at t = 0 for RX1/RX2.",
        ],
        quiet_s=1.0 if a.quick else 3.0,
        wait_quiet_s=a.wait_quiet_min * 60.0,
    )
    t0 = time.perf_counter()
    rows = []
    dropped = []
    with run.interruptible():
        for lead in leads:
            for rep in range(a.reps):
                if time.perf_counter() - t0 + float(warm + meas) + 30 > a.budget_min * 60:
                    dropped.append({"lead_s": lead, "rep": rep, "reason": "budget"})
                    continue
                r = run_once(lead, warm, meas, rep, run.dir)
                compact = r.pop("_compact")
                res = r.pop("_resolved")
                run.save_scenario(f"lead{lead}_rep{rep}", res)
                if rep == 0:
                    run.write_csv(
                        f"commands_lead{lead}_rep0.csv",
                        [
                            dict(
                                zip(
                                    ("m", "k_req_rx1", "late_rx1", "late_rx2", "status"),
                                    c,
                                    strict=True,
                                )
                            )
                            for c in compact
                        ],
                    )
                rows.append(r)
                run.write_json("eb3.json", rows)
                print(
                    f"lead {lead} rep {rep}: commands={r['commands']} "
                    f"inside={r['switch_inside_packet']}"
                    f"/{r['packets_counted']} late_rx1={r['lateness_samples']['rx1.rx0']}",
                    flush=True,
                )
    if dropped:
        run.write_csv("dropped.csv", dropped)
    summary = {
        "suite": SUITE,
        "by_lead": {
            lead: {
                "reps": sum(1 for r in rows if r["lead_s"] == lead),
                "switch_inside_packet_rates": [
                    r["switch_inside_packet_rate"] for r in rows if r["lead_s"] == lead
                ],
                "late_applied": sum(
                    r["status"].get("late_applied", 0) for r in rows if r["lead_s"] == lead
                ),
                "commands": sum(r["commands"] for r in rows if r["lead_s"] == lead),
            }
            for lead in leads
        },
        "dropped": len(dropped),
        "protocol_12_3": {
            "warmup_s": float(warm),
            "measure_s": float(meas),
            "reps": a.reps,
            "min_pages_per_rep": min((int(r["pages"] or 0) for r in rows), default=None),
            "meets_12_3": bool(
                warm >= 5
                and meas >= 60
                and a.reps >= 3
                and rows
                and min(int(r["pages"] or 0) for r in rows) >= 10_000
            ),
        },
    }
    run.finish(summary)
    print(f"E-B3 written to {run.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
