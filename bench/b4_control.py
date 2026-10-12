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
switch sample; not a real-time latency). A command whose switch sample lies at or after the
end of the run stays ``scheduled`` (the run stops before the engine computes that sample; at
10 kHz the last barrier's command, at ``t_b + L_min + U[0, L_min]`` after the barrier one
period before the end, always does): it is counted in ``scheduled_after_end`` and is not a
failure of the check.

``--recompute RUN_DIR`` rebuilds ``summary.json`` of a lockstep run offline from its
``lockstep.json`` (no measurement) into ``summary_recomputed.json``, keeping the original
summary's verdict next to the recomputed one.

**Paced part** (Linux job ``b4`` on a quiet c7i.8xlarge; on Windows labelled "Windows, not
representative"): the engine runs paced with async coupling, served over the control
socket; an external process (:mod:`bench._ctl_client` ``async``) sends one
``ris.set_config`` every 1/r wall seconds, timed (``at`` = its estimate of virtual time now
plus a lead) or ``asap``. Per repetition: ``--warmup-s`` warm-up, then ``--measure-s``
measured (commands issued during the warm-up are excluded). Reported: lateness distribution
(samples) per RX, command visible latency ``t_wall_visible - t_wall_recv`` percentiles,
statuses, client round-trip percentiles, deadline misses and page latency of the paced run.
The paced part sets ``OMP_WAIT_POLICY=PASSIVE`` (``--omp-wait-policy``; an explicit
environment setting wins): its served pages render serially
(``render_numba.PACED_PARALLEL_MIN_WORK``, ``Emulator.set_served``)
and the few parallel regions (noise blocks) left the OpenMP workers spinning for milliseconds
beside the control and transport threads (WSL2, 1 ms pages: compute p50 261 vs 203 us, round
trip p99 12.8 vs 4.6 ms, miss rate at 5 ms 1.5 % vs 0.05 %).
Paced plans (``--plan``):

* ``classic`` (default; the first B4 run): 0.5 ms pages (512 samples), ``--rates`` x
  ``--modes``, timed with lead ``--lead-ms`` (default 2 ms).
* ``served-1ms`` (the AWS job ``b4``): 1 ms pages (1024 samples) and a lead sweep. A timed
  command lands on time only if its lead exceeds the command's latency (send to schedule,
  p99 4.6-5.2 ms in the first B4 run) plus the client's error in estimating virtual time and
  the page that holds the switch, so the first run's 2 ms lead made late commands expected.
  Instead of one hand-picked lead, the plan measures the late fraction per lead: timed at
  2, 4, 6, 8 and 12 ms and asap at 100 and 10 commands/s, then timed (8 ms) and asap at
  1000 commands/s (where the engine is command-bound). Summary keys name the lead
  (``timed_lead8ms@100Hz``) and give ``late_fraction``.

**Validity of a paced repetition.** The timed client sets ``at = tv0 + (now - wsync) +
lead``: it assumes that virtual time follows wall time. When the engine lags (a backlog of
pages), a command arrives ``lead + lag`` ahead of the page that holds its switch, so
``late_applied = 0`` is vacuous and ``t_wall_visible - t_wall_recv`` grows with the lag
instead of measuring command latency. A repetition is **backlogged** (``backlogged``, with
``backlog_reasons``) when its page latency p50 exceeds one page or its miss rate at 5 ms slack
exceeds 1 % (:func:`backlog_reasons`). A backlogged repetition's ``visible_latency_us`` is
empty (the distribution is kept as ``visible_minus_recv_us``, never as a latency) and it is
left out of the per-key latency, late and exact fractions (``backlogged_reps`` counts it).

**Timed-command metrics** per repetition (:func:`timed_metrics`, commands of the measured
window; ``t0``, ``D_lag`` from the paced run's pacer):

* ``late_fraction``: ``late_applied`` / commands; ``exact_fraction``: applied with lateness 0
  on every RX (on the scheduled sample) / commands;
* ``send_to_schedule_rtt_us``: the client's round trip of ``ris.set_config`` (send to receipt);
* ``virtual_margin_us`` (timed): ``t_req - (t_wall_recv - t0 - D_lag)``, how far ahead of the
  pacer's wall schedule (the virtual time whose page is released at the receipt) the switch
  was requested; it does not depend on how far the engine lags;
* ``apply_delay_us``: ``t_wall_visible - (t0 + k1/f + D_lag)``, wall time from the nominal
  release of the page holding the applied switch (it ends at sample ``k1``) to the record
  being emitted after that page: compute time on time, the backlog when the engine lags.
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
CLASSIC_PAGE_S = "5e-4"
CLASSIC_LEAD_S = "2e-3"
PACED_PLANS: dict[str, dict[str, Any]] = {
    "served-1ms": {
        "page_target_s": "1e-3",
        # (rate Hz, mode, lead s), in priority order (a budget cut drops the tail)
        "combos": [
            *[
                (r, m, ld)
                for r in (100, 10)
                for m, ld in [("timed", f"{x}e-3") for x in (2, 4, 6, 8, 12)] + [("asap", None)]
            ],
            (1000, "timed", "8e-3"),
            (1000, "asap", None),
        ],
    },
}
"""Named paced plans besides ``classic`` (see the module docstring)."""


def _scene(
    duration_s: Fraction,
    mode: str = "lockstep",
    coupling: str = "barrier",
    page_target_s: str = CLASSIC_PAGE_S,
) -> dict[str, Any]:
    from bench._scenes import bench_scenario

    return bench_scenario(
        links=SCENE_LINKS,
        n_cells=240,
        n_ris=1,
        rate_sps="1e6",
        features="F1",
        duration_s=str(duration_s),
        page_target_s=page_target_s,
        mode=mode,
        coupling=coupling,
        name="b4_control",
    )


def combo_key(rate_hz: int, mode: str, lead_s: str | None, with_lead: bool) -> str:
    """Summary key of a paced combination: ``asap@100Hz``, ``timed@100Hz`` (classic, one
    lead) or ``timed_lead8ms@100Hz``."""
    if mode != "timed" or not with_lead or lead_s is None:
        return f"{mode}@{rate_hz}Hz"
    return f"timed_lead{float(Fraction(lead_s)) * 1e3:g}ms@{rate_hz}Hz"


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


BACKLOG_MISS_5MS = 0.01
"""A paced repetition whose miss rate at 5 ms slack exceeds this is backlogged."""


def backlog_reasons(
    page_latency_us: dict[str, Any] | None, miss_rate: dict[str, Any] | None, page_s: float
) -> list[str]:
    """Why a paced repetition is backlogged (empty: it is not): page latency p50 above one
    page (``page_s``) or miss rate at 5 ms slack above :data:`BACKLOG_MISS_5MS`. Without
    paced statistics the repetition cannot be validated and counts as backlogged."""
    p50 = (page_latency_us or {}).get("p50")
    m5 = (miss_rate or {}).get("5ms")
    if p50 is None and m5 is None:
        return ["no paced page statistics"]
    out = []
    if p50 is not None and float(p50) > page_s * 1e6:
        out.append(f"page latency p50 {float(p50):.0f} us > one page ({page_s * 1e6:.0f} us)")
    if m5 is not None and float(m5) > BACKLOG_MISS_5MS:
        out.append(f"miss rate at 5 ms {float(m5):.3g} > {BACKLOG_MISS_5MS:g}")
    return out


def timed_metrics(
    recs: list[Any],
    *,
    page: int,
    rate: Fraction,
    t0_ns: int,
    d_lag_ns: int,
    end: int | None = None,
    timed: bool = True,
) -> dict[str, Any]:
    """Command metrics of a paced repetition (``recs``: the records of its measured window;
    ``page``, ``rate``, ``end``: the RX rate group; ``t0_ns``, ``d_lag_ns``: its pacer; see
    the module docstring): ``commands``, ``late_fraction``, ``exact_fraction``,
    ``virtual_margin_us`` (timed commands only) and ``apply_delay_us`` percentiles."""
    n = len(recs)
    late = sum(1 for r in recs if r.status == "late_applied")
    exact = sum(
        1
        for r in recs
        if r.status == "applied"
        and r.lateness_samples
        and all(int(v) == 0 for v in r.lateness_samples.values())
    )
    margin: list[float] = []
    delay: list[float] = []
    for r in recs:
        if timed and r.t_req is not None and r.t_wall_recv_ns is not None:
            t_virt_ns = r.t_wall_recv_ns - t0_ns - d_lag_ns  # pacer schedule at the receipt
            margin.append((float(Fraction(r.t_req) * 10**9) - t_virt_ns) * 1e-3)
        if r.status in ("applied", "late_applied") and r.k_applied and r.t_wall_visible_ns:
            k1 = (min(int(k) for k in r.k_applied.values()) // page + 1) * page
            if end is not None:
                k1 = min(k1, int(end))
            release_ns = t0_ns + round(Fraction(k1) / rate * 10**9) + d_lag_ns
            delay.append((r.t_wall_visible_ns - release_ns) * 1e-3)
    return {
        "commands": n,
        "late_fraction": late / n if n else None,
        "exact_fraction": exact / n if n else None,
        "virtual_margin_us": percentiles(margin, (1, 50, 99)) if margin else {},
        "apply_delay_us": percentiles(delay, (50, 90, 99)) if delay else {},
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
    recs = list(eng.records)
    out.update(_record_stats(recs, "b4"))
    ends = {rx: g.end for g in eng.groups for rx in g.rx}
    out["scheduled_after_end"] = scheduled_after_end([r for r in recs if r.src == "b4"], ends)
    out["n_controls"] = n_ctl
    out["n_rx"] = len(eng.rx)
    out["expected_commands"] = 0 if variant == "none" else int(rate_hz * duration)
    out["nondeterministic_inputs"] = eng.nondeterministic_inputs
    out["min_ris_minus_los_db"] = chk["min_ris_minus_los_db"]
    out["scenario_sha256"] = res.sha256()
    eng.close()
    return out


def scheduled_after_end(recs: list[Any], ends: dict[str, int | None]) -> int:
    """Commands still ``scheduled`` because their switch sample on some RX is at or after
    that RX's end of run (``ends``: first sample the run does not compute, per RX). The run
    stopped before the engine reached the switch, so they cannot have been applied."""
    n = 0
    for r in recs:
        if r.status != "scheduled":
            continue
        ks = r.k_applied or r.k_req or {}
        if any(ends.get(rx) is not None and int(k) >= int(ends[rx]) for rx, k in ks.items()):
            n += 1
    return n


def after_end_bound(rate_hz: float, virtual_s: float | Fraction, l_min: Fraction = L_MIN) -> int:
    """Upper bound on the commands of a lockstep row whose switch can lie at or after the end
    of the run, from the controller's latency model alone (rows written before
    ``scheduled_after_end`` was recorded): barriers at ``t_b = m / rate`` before the end with
    ``t_b + 2 L_min >= end`` (latency ``L_min + U[0, L_min]``)."""
    if not rate_hz or not virtual_s:
        return 0
    rate = Fraction(rate_hz).limit_denominator(10**6)
    end = Fraction(virtual_s).limit_denominator(10**9)
    lo = end - 2 * l_min
    return sum(1 for m in range(int(end * rate) + 1) if lo <= m / rate < end)


def lockstep_problems(r: dict[str, Any]) -> list[str]:
    """Why a lockstep row does not demonstrate zero lateness (empty: it does).

    Zero lateness is only evidence when commands were really issued and applied: a
    controller row needs at least 90 % of its expected commands (one per period), every
    command ``applied`` except those whose switch lies at or after the end of the run (they
    stay ``scheduled``: ``scheduled_after_end``, or for older rows at most
    :func:`after_end_bound` of them), one lateness value per command and RX, and all of them
    0; the reference row (``none``) must have no command at all."""
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
    if "scheduled_after_end" in r:
        allowed = int(r["scheduled_after_end"] or 0)
    else:
        allowed = after_end_bound(r.get("rate_hz") or 0, r.get("virtual_s") or 0)
    if other.get("scheduled", 0) <= allowed:
        other.pop("scheduled", None)
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
    return lockstep_summary(rows, interrupted=bool(run.interrupted))


def lockstep_summary(rows: list[dict[str, Any]], interrupted: bool = False) -> dict[str, Any]:
    """Summary of the lockstep rows (the ``lockstep.json`` records), with the check."""
    bad = [r for r in rows if r["variant"] != "none" and r["lateness_nonzero"]]
    problems = {
        f"{r['variant']}@{r['rate_hz']}Hz": lockstep_problems(r)
        for r in rows
        if lockstep_problems(r)
    }
    return {
        "lockstep_points": len(rows),
        "lockstep_all_lateness_zero": not bad,
        "lockstep_check_ok": bool(rows) and not problems and not interrupted,
        "lockstep_check_problems": problems,
        "lockstep_commands_total": sum(r["commands"] for r in rows),
        # rows with commands left 'scheduled': how many, how many lie after the end of the
        # run (recorded; None for older rows) and the bound the check used for older rows
        "lockstep_scheduled_at_end": {
            f"{r['variant']}@{r['rate_hz']}Hz": {
                "scheduled": (r.get("status") or {}).get("scheduled", 0),
                "after_end": r.get("scheduled_after_end"),
                "after_end_bound": after_end_bound(r.get("rate_hz") or 0, r.get("virtual_s") or 0),
            }
            for r in rows
            if (r.get("status") or {}).get("scheduled", 0)
        },
        "lockstep_rtf": {f"{r['variant']}@{r['rate_hz']}Hz": r["rtf"] for r in rows},
    }


def recompute_lockstep(run_dir: Path, out: Path | None = None) -> dict[str, Any]:
    """Offline: rebuild a lockstep run's summary from its ``lockstep.json`` (no measurement)
    and write it to ``out`` (default ``run_dir/summary_recomputed.json``). The original
    summary's check fields are kept under ``original``."""
    rows = json.loads((run_dir / "lockstep.json").read_text(encoding="utf-8"))
    orig_path = run_dir / "summary.json"
    orig = json.loads(orig_path.read_text(encoding="utf-8")) if orig_path.exists() else {}
    summ = lockstep_summary(rows, interrupted=bool(orig.get("interrupted")))
    summ.update({"suite": SUITE, "part": "lockstep", "interrupted": orig.get("interrupted")})
    summ["recomputed"] = {
        "from": "lockstep.json (unchanged)",
        "by": "bench.b4_control.lockstep_summary",
        "rule": (
            "a command still 'scheduled' at the end of the run is not a failure when its "
            "switch lies at or after the end of the run (rows without scheduled_after_end: "
            "at most after_end_bound() such commands, from the L_min + U[0, L_min] latency)"
        ),
        "original": {
            k: orig.get(k)
            for k in ("lockstep_check_ok", "lockstep_check_problems", "lockstep_all_lateness_zero")
        },
    }
    dst = out or run_dir / "summary_recomputed.json"
    with dst.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(summ, indent=2) + "\n")
    return summ


# =========================================================================== paced


def paced_point(
    rate_hz: int,
    mode: str,
    warmup_s: Fraction,
    measure_s: Fraction,
    rep: int,
    *,
    lead_s: str = CLASSIC_LEAD_S,
    page_target_s: str = CLASSIC_PAGE_S,
) -> dict[str, Any]:
    """One paced + async repetition with an external command process (``lead_s``: lead of
    timed commands; ``page_target_s``: the scene's page target)."""
    from bench._engine_point import (
        _resolve,
        _restart_paced_stats,
        build_engine,
        prewarm,
        threading_record,
    )
    from bench._scenes import ris_path_check
    from sremu.transport.control_server import ControlServer, ServerConfig

    total = warmup_s + measure_s
    res = _resolve(_scene(total, mode="paced", coupling="async", page_target_s=page_target_s))
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
            str(lead_s),
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
        th_rec: dict[str, Any] | None = None
        while not srv.is_done():
            time.sleep(0.05)
            if th_rec is None and state.get("done"):  # in the measured window: pools alive
                th_rec = threading_record()
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
    recs = list(eng.records)
    stats = _record_stats(recs, "b4async", t_from=warmup_s)
    pacer = eng._pacer
    mine = [r for r in recs if r.src == "b4async" and r.t_req is not None and r.t_req >= warmup_s]
    tm: dict[str, Any] = {}
    if pacer is not None and pacer.t0_ns is not None:
        tm = timed_metrics(
            mine,
            page=int(g.page),
            rate=g.rate,
            t0_ns=int(pacer.t0_ns),
            d_lag_ns=int(pacer.d_lag_ns),
            end=g.end,
            timed=mode == "timed",
        )
    page_s = float(Fraction(int(g.page)) / g.rate)
    reasons = backlog_reasons(paced.get("latency_us"), paced.get("miss_rate"), page_s)
    vis = stats.pop("visible_latency_us")
    eng.close()
    return {
        "rate_hz": rate_hz,
        "mode": mode,
        "lead_s": float(Fraction(lead_s)) if mode == "timed" else None,
        "page_samples": int(g.page),
        "page_s": page_s,
        "page_target_s": page_target_s,
        "rep": rep,
        "warmup_s": float(warmup_s),
        "measure_s": float(measure_s),
        "warmup_restart_ok": bool(state.get("done")),
        "wall_s": wall,
        **stats,
        # a backlogged repetition's visible - recv is lead + lag, not a command latency
        "backlogged": bool(reasons),
        "backlog_reasons": reasons,
        "visible_latency_us": {} if reasons else vis,
        "visible_minus_recv_us": vis,
        "late_fraction": tm.get("late_fraction"),
        "exact_fraction": tm.get("exact_fraction"),
        "virtual_margin_us": tm.get("virtual_margin_us", {}),
        "apply_delay_us": tm.get("apply_delay_us", {}),
        "send_to_schedule_rtt_us": client.get("rtt_us"),
        "pacer_t0_ns": None if pacer is None else pacer.t0_ns,
        "d_lag_ns": None if pacer is None else pacer.d_lag_ns,
        "threading": th_rec,
        "render_min_work": eng.render_parallel_min_work,  # served paced: 2**18
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


BACKLOG_RULE = (
    "a repetition is backlogged when its page latency p50 exceeds one page or its miss rate "
    f"at 5 ms slack exceeds {BACKLOG_MISS_5MS:g}; its visible - recv time (lead + lag for "
    "timed commands) is not reported as latency and it is left out of the per-key latency, "
    "late and exact fractions"
)


def is_backlogged(r: dict[str, Any]) -> bool:
    """A paced row's ``backlogged`` flag; rows written before it existed are judged from their
    page latency, miss rate and page length (:func:`backlog_reasons`; B4 scene at 1 MS/s)."""
    if "backlogged" in r:
        return bool(r["backlogged"])
    page_s = r.get("page_s") or int(r.get("page_samples") or 0) / 1e6
    return bool(backlog_reasons(r.get("page_latency_us"), r.get("miss_rate"), float(page_s)))


def _med(xs: list[Any]) -> float | None:
    ys = [float(x) for x in xs if x is not None]
    return statistics.median(ys) if ys else None


def paced_by_key(
    rows: list[dict[str, Any]],
    combos: list[tuple[int, str, str | None]],
    keys_with_lead: bool,
    virt: float,
) -> dict[str, Any]:
    """Per-combination summary of the paced rows. Latency, lateness and the late and exact
    fractions come from the repetitions that are not backlogged (:func:`is_backlogged`;
    ``backlogged_reps`` counts the others); the apply delay, virtual margin and round trip
    are wall-clock figures that stay meaningful in a backlog and use every repetition."""
    by: dict[str, Any] = {}
    for rate, mode, lead in combos:
        key = combo_key(rate, mode, lead, keys_with_lead)
        if key in by:
            continue
        rs = [r for r in rows if r["key"] == key]
        if not rs:
            continue
        ok = [r for r in rs if not is_backlogged(r)]
        p99 = [r["lateness_samples"].get("p99") for r in ok if r["lateness_samples"]]
        vis = [r["visible_latency_us"].get("p99") for r in ok if r.get("visible_latency_us")]
        ach = [
            r["client"]["achieved_rate_hz"]
            for r in rs
            if isinstance(r.get("client"), dict) and r["client"].get("achieved_rate_hz")
        ]
        late = sum(r["status"].get("late_applied", 0) for r in ok)
        n_ok = sum(r["commands"] for r in ok)
        exact = [(r.get("exact_fraction"), r["commands"]) for r in ok]
        n_exact = sum(n for f, n in exact if f is not None)
        rtt = [((r.get("send_to_schedule_rtt_us") or {}).get("p99")) for r in rs]
        by[key] = {
            "rate_hz": rate,
            "mode": mode,
            "lead_s": rs[0]["lead_s"],
            "page_samples": rs[0]["page_samples"],
            "reps": len(rs),
            "backlogged_reps": len(rs) - len(ok),
            # the client waits for each reply (REQ/REP): a slow engine caps its rate
            "achieved_rate_hz_median": statistics.median(ach) if ach else None,
            "wall_over_virtual_max": max(float(r["wall_s"]) / virt for r in rs),
            "lateness_p99_samples_median": statistics.median(p99) if p99 else None,
            "lateness_max_samples_worst": max(((r["lateness_max"] or 0) for r in ok), default=None),
            "visible_latency_p99_us_worst": max(vis) if vis else None,
            "late_applied_total": late,
            # applied late / commands in the measured windows, over the valid repetitions
            "late_fraction": late / n_ok if n_ok else None,
            # applied on the scheduled sample (lateness 0 on every RX) / commands, likewise
            "exact_fraction": sum(f * n for f, n in exact if f is not None) / n_exact
            if n_exact
            else None,
            "apply_delay_p50_us_median": _med(
                [(r.get("apply_delay_us") or {}).get("p50") for r in rs]
            ),
            "apply_delay_p99_us_worst": max(
                (x for r in rs if (x := (r.get("apply_delay_us") or {}).get("p99")) is not None),
                default=None,
            ),
            "virtual_margin_p50_us_median": _med(
                [(r.get("virtual_margin_us") or {}).get("p50") for r in rs]
            ),
            "virtual_margin_p1_us_min": min(
                (x for r in rs if (x := (r.get("virtual_margin_us") or {}).get("p1")) is not None),
                default=None,
            ),
            "send_to_schedule_rtt_p99_us_worst": max(
                (x for x in rtt if x is not None), default=None
            ),
            "sustainable_reps": sum(1 for r in rs if r["sustainable"]),
            "commands_total": sum(r["commands"] for r in rs),
            "commands_valid": n_ok,
            "commands_per_virtual_s_median": statistics.median(
                r["commands_per_virtual_s"] for r in rs
            ),
            "engine_bound_reps": sum(1 for r in rs if r["engine_bound"]),
        }
    return by


def paced_combos(
    plan: str, rates: list[int], modes: list[str], leads_s: list[str]
) -> tuple[list[tuple[int, str, str | None]], str]:
    """``(combos, page_target_s)`` of a paced plan: ``classic`` crosses ``rates`` with
    ``modes`` (timed once per lead), a named plan is fixed (:data:`PACED_PLANS`)."""
    if plan != "classic":
        p = PACED_PLANS[plan]
        return list(p["combos"]), str(p["page_target_s"])
    combos: list[tuple[int, str, str | None]] = []
    for rate in rates:
        for mode in modes:
            for lead in leads_s if mode == "timed" else [None]:
                combos.append((rate, mode, lead))
    return combos, CLASSIC_PAGE_S


def run_paced(
    run: BenchRun,
    combos: list[tuple[int, str, str | None]],
    reps: int,
    warm: Fraction,
    meas: Fraction,
    budget_s: float,
    *,
    page_target_s: str = CLASSIC_PAGE_S,
    keys_with_lead: bool = False,
) -> dict[str, Any]:
    rows = []
    dropped = []
    failed: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    virt = float(warm + meas)
    slow = 1.2  # wall / virtual of a repetition; raised to what the runs show (a lagging
    # engine stretches a paced run far beyond its virtual length)
    with run.interruptible():
        for rate, mode, lead in combos:
            lead_s = lead or CLASSIC_LEAD_S  # asap ignores it
            key = combo_key(rate, mode, lead, keys_with_lead)
            for rep in range(reps):
                if time.perf_counter() - t_start + virt * slow > budget_s:
                    dropped.append(
                        {"key": key, "rate_hz": rate, "mode": mode, "rep": rep, "reason": "budget"}
                    )
                    continue
                try:
                    r = paced_point(
                        rate, mode, warm, meas, rep, lead_s=lead_s, page_target_s=page_target_s
                    )
                except Exception as exc:  # recorded; the suite goes on
                    traceback.print_exc()
                    failed.append(
                        {
                            "key": key,
                            "rate_hz": rate,
                            "mode": mode,
                            "rep": rep,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                r["key"] = key
                slow = max(slow, 1.1 * float(r["wall_s"]) / virt)
                rows.append(r)
                vis = (
                    f"BACKLOGGED ({'; '.join(r['backlog_reasons'])}): no latency"
                    if r["backlogged"]
                    else f"vis p99={r['visible_latency_us'].get('p99')} us"
                )
                print(
                    f"paced {key} rep {rep}: commands={r['commands']} "
                    f"late p99={r['lateness_samples'].get('p99')} max={r['lateness_max']} "
                    f"{vis} apply p99={r['apply_delay_us'].get('p99')} us "
                    f"miss={r['miss_rate']} sustainable={r['sustainable']}",
                    flush=True,
                )
                run.write_json("paced.json", rows)
    flat = [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in rows]
    run.write_csv("paced.csv", flat)
    if dropped:
        run.write_csv("paced_dropped.csv", dropped)
    if failed:
        run.write_csv("paced_failed.csv", failed)
    by = paced_by_key(rows, combos, keys_with_lead, virt)
    pages = [int(r["pages"] or 0) for r in rows]
    w, m = float(warm), float(meas)
    return {
        "paced": by,
        "page_target_s": page_target_s,
        "paced_all_reps_sustainable": bool(rows) and all(r["sustainable"] for r in rows),
        "paced_dropped": len(dropped),
        "paced_failed": failed,
        "paced_backlogged_reps": sum(1 for r in rows if is_backlogged(r)),
        "backlog_rule": BACKLOG_RULE,
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
    ap.add_argument(
        "--plan",
        choices=("classic", *PACED_PLANS),
        default="classic",
        help="paced plan; a named plan fixes pages, rates, modes and leads",
    )
    ap.add_argument("--lead-ms", default="2", help="classic plan: timed leads, ms (list)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup-s", default="5")
    ap.add_argument("--measure-s", default="60")
    ap.add_argument("--budget-min", type=float, default=85.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wait-quiet-min", type=float, default=0.0, help="wait for an idle host first")
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: all)")
    ap.add_argument(
        "--omp-wait-policy",
        choices=("passive", "active", "env"),
        default="passive",
        help="paced part: OMP_WAIT_POLICY for numba's OpenMP layer (env: leave the environment)",
    )
    ap.add_argument(
        "--recompute",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="offline: rebuild a lockstep run's summary from its lockstep.json",
    )
    ap.add_argument("--recompute-out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.recompute is not None:
        summ = recompute_lockstep(a.recompute, a.recompute_out)
        print(f"lockstep_check_ok={summ['lockstep_check_ok']}")
        return 0 if summ["lockstep_check_ok"] else 1
    paced = a.part == "paced"
    # served paced engine: idle OpenMP workers sleep at once instead of spinning beside the
    # control and transport threads (before numba is imported; an explicit setting wins)
    wait_policy = bench.omp_wait_policy(
        None if a.omp_wait_policy == "env" or not paced else a.omp_wait_policy
    )
    bench.set_threads(a.threads)
    SCENE_LINKS = int(a.links)
    rates = [int(x) for x in a.rates.split(",") if x]
    modes = [m for m in a.modes.split(",") if m]
    leads = [str(Fraction(x) / 1000) for x in a.lead_ms.split(",") if x]
    combos, page_target = paced_combos(a.plan, rates, modes, leads)
    keys_with_lead = a.plan != "classic" or len(leads) > 1
    spec = {
        "threads": a.threads,
        "part": a.part,
        "quick": a.quick,
        "scene": f"B2 base, L={a.links}, N=240, 1 panel, 1 MS/s, F1",
        "links": a.links,
        "controller": "Urap(p=0.5), L_min=1e-4 s, latency L_min+U[0,L_min] (lockstep)",
        "rates_hz": sorted({c[0] for c in combos})
        if paced
        else list((10, 1000) if a.quick else RATES),
        "modes": sorted({c[1] for c in combos}) if paced else ["none", "inproc", "remote"],
        "reps": a.reps if paced else 1,
        "warmup_s": a.warmup_s if paced else None,
        "measure_s": a.measure_s if paced else None,
    }
    if paced:
        spec.update(
            omp_wait_policy=wait_policy,
            plan=a.plan,
            page_target_s=page_target,
            combos=[{"rate_hz": r, "mode": m, "lead_s": ld} for r, m, ld in combos],
        )
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
            combos,
            a.reps,
            Fraction(a.warmup_s),
            Fraction(a.measure_s),
            a.budget_min * 60.0,
            page_target_s=page_target,
            keys_with_lead=keys_with_lead,
        )
        summary["plan"] = a.plan
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
