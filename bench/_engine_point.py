"""Lockstep and paced measurement of one benchmark scenario (DESIGN.md sections 12.1-12.3).

:func:`measure_lockstep` builds an :class:`sremu.engine.emulator.Emulator` from a scenario
mapping (:func:`bench._scenes.bench_scenario`), checks the non-zero RIS path, warms up, runs
one untimed pilot to size the repetitions, then ``reps`` measured repetitions. RTF is the
engine's own section 12.1 meter (min over RX streams of samples/f_s divided by wall time;
``RTF_min`` over complete 100 ms virtual windows), restarted (``Emulator.reset_metrics``)
before each repetition.

The warm-up and the pilot are **wall-limited** (``warm_wall_max_s``, ``pilot_wall_max_s``):
they advance in growing chunks of pages and stop at their virtual length or their wall
limit, whichever comes first, so a very slow point cannot spend minutes before its first
measurement. With ``rtf_floor`` set, a point whose pilot RTF is below the floor is not
measured further: it returns ``status = "below_floor"`` with the pilot RTF (a sizing
estimate, never reported as a measured RTF), and the B2 sweep drops the larger-L points of
that series (logged in ``dropped.csv``).

Feature sets F0-F4 run headless in-process (``always_active``: every RX page is computed);
F4 attaches the run-directory recorder of ``sremu run`` (``sremu.commands.run.make_recorder``:
SigMF IQ of every RX, events, metrics) writing to a temporary directory that is deleted
afterwards (its size is reported). F5 serves the engine over the control socket and native
ZMQ data plane (``sremu.transport.control_server``) to two client processes
(:mod:`bench._zmq_client`): one sends every TX stream, one reads every RX stream; the RTF of
F5 is measured by the RX client (virtual time read / wall time after its warm-up samples).

:func:`measure_paced` runs ``Emulator.run_paced`` headless: 5 s warm-up then the measured
window (section 12.3), with the pacer's deadline statistics restarted at the end of the
warm-up (see :func:`_restart_paced_stats`).
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from ._common import REPO

__all__ = ["build_engine", "measure_lockstep", "measure_paced", "prewarm", "snapshot_facts"]


def _resolve(scenario: Mapping[str, Any]) -> Any:
    from sremu.scenario.load import parse_scenario, resolve_scenario

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return resolve_scenario(parse_scenario(dict(scenario)), warn=False)


def build_engine(resolved: Any, features: str, **overrides: Any) -> Any:
    """Emulator of a resolved scenario for feature set ``features`` (F0 = noise off)."""
    from sremu.engine.emulator import Emulator

    kw: dict[str, Any] = {"always_active": True, "noise": features != "F0"}
    kw.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Emulator.from_scenario(resolved, **kw)


def prewarm(resolved: Any, features: str = "F1", pages: int = 64) -> None:
    """JIT pre-warm: run ``pages`` pages of a throwaway lockstep engine of the same scenario
    (numba compiles or loads its kernels once per process), so compilation cannot leave a
    backlog in a paced run that follows."""
    pre = build_engine(resolved, features, mode="lockstep", coupling="barrier")
    g = pre.groups[0]
    pre.run(until_s=Fraction(pages * g.page) / g.rate)
    pre.close()


def snapshot_facts(eng: Any) -> dict[str, Any]:
    """Structural facts of an engine: links, RX streams, panels, rows, quadrature points,
    tiles, RIS tile paths per link, snapshot period, page size, FD filter of each link."""
    snap = eng.model.snapshot(Fraction(0))
    rows = sum(rc.n_rows for rc in snap.ris.values())
    q = 0
    for pid in snap.ris:
        q = max(q, int(eng.model.quadrature(pid).points.shape[0]))
    rq = sum(
        rc.n_rows * int(eng.model.quadrature(pid).points.shape[0]) for pid, rc in snap.ris.items()
    )
    tiles = sum(rc.n_rows * rc.n_tiles for rc in snap.ris.values())
    n_cells = sum(rc.n_cells for rc in snap.ris.values())
    links = list(eng.model.links())
    from sremu.channel.fd import choose_fd

    fds = {}
    for c in eng.configs.values():
        if c.direction == "tx":
            fds[c.stream] = choose_fd(c.bandwidth_hz, c.rate, eng.cfg.fd_error_target_db)
    taps = max((f.taps for f in fds.values()), default=1)
    return {
        "n_links": len(links),
        "n_rx": len(eng.rx),
        "n_tx": len(eng.tx),
        "n_panels": len(snap.ris),
        "ris_rows": rows,
        "quad_points_per_panel": q,
        "rq": rq,
        "cells_total": n_cells,
        "ris_tile_paths": tiles,
        "t_u_s": None if eng.t_u is None else float(eng.t_u),
        "t_u_exact": None if eng.t_u is None else f"{eng.t_u.numerator}/{eng.t_u.denominator}",
        "page": eng.groups[0].page if eng.groups else None,
        "fd_taps": taps,
        "fd_filter": next(iter(fds.values())).filter_id if fds else None,
    }


def _run_rep(eng: Any, t_start: Fraction, dur: Fraction) -> dict[str, Any]:
    eng.reset_metrics()
    w0 = time.perf_counter()
    s = eng.run(until_s=t_start + dur)
    wall = time.perf_counter() - w0
    r = dict(s.rtf)
    return {
        "virtual_s": float(dur),
        "wall_s_outer": wall,
        "rtf": r.get("rtf"),
        "rtf_min": r.get("rtf_min"),
        "aggregate_sps": r.get("aggregate_sps"),
        "n_windows": r.get("n_windows"),
        "counters": dict(s.counters),
        "verify_interp_max_rel": s.verify_interp_max_rel,
        "nondeterministic_inputs": s.nondeterministic_inputs,
    }


def _advance_limited(
    eng: Any, t: Fraction, t_end: Fraction, page: Fraction, wall_max_s: float
) -> Fraction:
    """Run the engine from virtual time ``t`` towards ``t_end`` in chunks of pages (4, 8,
    16, ... pages, doubling while a chunk takes < 0.5 s of wall time) and stop at ``t_end``
    or once ``wall_max_s`` of wall time has elapsed. Returns the virtual time reached."""
    w0 = time.perf_counter()
    chunk = 4
    while t < t_end:
        nxt = min(t_end, t + chunk * page)
        c0 = time.perf_counter()
        eng.run(until_s=nxt)
        t = nxt
        now = time.perf_counter()
        if now - w0 >= wall_max_s:
            break
        if now - c0 < 0.5:
            chunk *= 2
    return t


def _pilot(
    eng: Any, t: Fraction, dur: Fraction, page: Fraction, wall_max_s: float
) -> tuple[dict[str, Any], Fraction]:
    """Untimed pilot repetition (sizing only): RTF over at most ``dur`` of virtual time or
    ``wall_max_s`` of wall time. Returns the pilot record and the virtual time reached."""
    eng.reset_metrics()
    w0 = time.perf_counter()
    t1 = _advance_limited(eng, t, t + dur, page, wall_max_s)
    wall = time.perf_counter() - w0
    s = eng.summary()
    return {
        "virtual_s": float(t1 - t),
        "wall_s_outer": wall,
        "rtf": s.rtf.get("rtf"),
        "truncated_by_wall": t1 < t + dur,
    }, t1


def _page_dur(eng: Any) -> Fraction:
    g = eng.groups[0]
    return Fraction(g.page) / g.rate


def _round_pages(t: Fraction, page: Fraction) -> Fraction:
    return max(1, math.ceil(t / page)) * page


def measure_lockstep(
    scenario: Mapping[str, Any],
    *,
    features: str,
    reps: int = 3,
    warmup_s: str | Fraction = "0.05",
    target_pages: int = 10_000,
    min_virtual_s: str | Fraction = "0.3",
    rep_wall_target_s: float = 6.0,
    point_wall_cap_s: float = 240.0,
    record_root: Path | None = None,
    rtf_floor: float | None = None,
    warm_wall_max_s: float = 20.0,
    pilot_wall_max_s: float = 10.0,
) -> dict[str, Any]:
    """Measure one lockstep point; returns the per-repetition and aggregate results.

    Repetition length: ``target_pages`` pages, reduced so one repetition takes about
    ``rep_wall_target_s`` (pilot RTF), never below ``min_virtual_s`` (two complete 100 ms
    windows); the number of repetitions is reduced (to 1) when ``point_wall_cap_s`` would
    be exceeded. The result says how many pages and seconds each repetition really had.
    ``status`` is ``"ok"``, or ``"below_floor"`` (pilot RTF < ``rtf_floor``: no
    repetitions, ``rtf_median`` is None and ``pilot_rtf`` holds the sizing estimate).
    """
    from bench._scenes import ris_path_check

    if features == "F5":
        return measure_zmq(
            scenario,
            reps=reps,
            warmup_s=warmup_s,
            target_pages=target_pages,
            min_virtual_s=min_virtual_s,
            rep_wall_target_s=rep_wall_target_s,
            point_wall_cap_s=point_wall_cap_s,
            rtf_floor=rtf_floor,
        )
    t_build = time.perf_counter()
    res = _resolve(scenario)
    eng = build_engine(res, features)
    build_s = time.perf_counter() - t_build
    rec_dir = None
    close_rec = None
    rec_bytes = 0
    try:
        if features == "F4":
            from sremu.commands.run import make_recorder

            rec_dir = Path(tempfile.mkdtemp(prefix="sremu_b2_rec_", dir=record_root))
            sink, close_rec = make_recorder(rec_dir, eng, iq=True, events=True, metrics=True)
            eng.add_sink(sink)
        facts = snapshot_facts(eng)
        chk = ris_path_check(eng.model)
        page = _page_dur(eng)
        w0 = time.perf_counter()
        warm = _advance_limited(
            eng, Fraction(0), _round_pages(Fraction(warmup_s), page), page, warm_wall_max_s
        )
        warm_wall = time.perf_counter() - w0
        t = warm
        pilot_dur = _round_pages(Fraction(min_virtual_s) / 3, page)
        pilot, t = _pilot(eng, t, pilot_dur, page, pilot_wall_max_s)
        rtf_p = pilot["rtf"] or 1e-9
        below = rtf_floor is not None and rtf_p < rtf_floor
        dur_pages = _round_pages(page * target_pages, page)
        dur_wall = _round_pages(Fraction(rtf_p * rep_wall_target_s).limit_denominator(10**6), page)
        dur = max(min(dur_pages, dur_wall), _round_pages(Fraction(min_virtual_s), page))
        est_rep_wall = float(dur) / rtf_p
        n_reps = 0 if below else reps
        while n_reps > 1 and est_rep_wall * n_reps > point_wall_cap_s:
            n_reps -= 1
        out_reps = []
        for _ in range(n_reps):
            r = _run_rep(eng, t, dur)
            t += dur
            out_reps.append(r)
        decisions = _decisions(eng)
        if close_rec is not None:
            close_rec()
            close_rec = None
            assert rec_dir is not None
            rec_bytes = sum(f.stat().st_size for f in rec_dir.rglob("*") if f.is_file())
    finally:
        if close_rec is not None:
            close_rec()
        eng.close()
        if rec_dir is not None:
            shutil.rmtree(rec_dir, ignore_errors=True)
    rtfs = [r["rtf"] for r in out_reps if r["rtf"] is not None]
    mins = [r["rtf_min"] for r in out_reps if r["rtf_min"] is not None]
    agg = [r["aggregate_sps"] for r in out_reps if r["aggregate_sps"] is not None]
    counters: dict[str, int] = {}
    for r in out_reps:
        for k, v in r["counters"].items():
            counters[k] = max(counters.get(k, 0), int(v))
    pages_per_rep = int(dur / page)
    vmax = [r["verify_interp_max_rel"] for r in out_reps if r["verify_interp_max_rel"] is not None]
    return {
        "status": "below_floor" if below else "ok",
        "rtf_floor": rtf_floor,
        "scenario_sha256": res.sha256(),
        "resolved": res,
        "features": features,
        "build_s": build_s,
        "warmup_virtual_s": float(warm),
        "warmup_wall_s": warm_wall,
        "pilot_rtf": pilot["rtf"],
        "pilot_virtual_s": pilot["virtual_s"],
        "pilot_truncated_by_wall": pilot["truncated_by_wall"],
        "reps": len(out_reps),
        "rep_virtual_s": float(dur),
        "rep_pages": pages_per_rep,
        "rtf_median": statistics.median(rtfs) if rtfs else None,
        "rtf_reps": rtfs,
        "rtf_min_worst": min(mins) if mins else None,
        "aggregate_msps_median": statistics.median(agg) / 1e6 if agg else None,
        "counters": counters,
        "verify_interp_max_rel": max(vmax) if vmax else None,
        "nondeterministic_inputs": any(r["nondeterministic_inputs"] for r in out_reps),
        "recorded_bytes": rec_bytes,
        "ris_check": chk,
        "facts": facts,
        "decisions": decisions,
    }


def _decisions(eng: Any) -> dict[str, Any]:
    """Summary of the engine's per-link render decisions (narrowband, FD filter, groups)."""
    info = eng.manifest_info()
    dec = info.get("link_decisions", {}) or {}
    groups = [int(v.get("n_groups", 1)) for v in dec.values()]
    nb = [bool(v.get("narrowband")) for v in dec.values()]
    fds = sorted({str(v.get("fd_filter")) for v in dec.values()})
    pol = info.get("snapshot_policy", {}) or {}
    return {
        "n_links_decided": len(dec),
        "mean_fd_groups": statistics.fmean(groups) if groups else None,
        "max_fd_groups": max(groups) if groups else None,
        "narrowband_fraction": statistics.fmean(nb) if nb else None,
        "fd_filters": fds,
        "t_u_s_used": pol.get("t_u_s_used"),
        "tiles": pol.get("tiles"),
        "interp_bound_rel": pol.get("total_bound_rel"),
    }


# =========================================================================== F5 (ZMQ)


def measure_zmq(
    scenario: Mapping[str, Any],
    *,
    reps: int = 3,
    warmup_s: str | Fraction = "0.05",
    target_pages: int = 10_000,
    min_virtual_s: str | Fraction = "0.3",
    rep_wall_target_s: float = 6.0,
    point_wall_cap_s: float = 240.0,
    rtf_floor: float | None = None,
) -> dict[str, Any]:
    """F5: lockstep served over native ZMQ to a TX client process and an RX client process.

    Each repetition is a fresh served engine (the clients send/read ``warmup + dur``
    samples per stream; the RX client times only the samples after the warm-up). The
    recorder of F4 stays attached (cumulative feature sets)."""
    from bench._scenes import ris_path_check
    from sremu.commands.run import make_recorder
    from sremu.transport.control_server import ControlServer, ServerConfig

    res = _resolve(scenario)
    d = res.scenario.model_dump(mode="python")
    probe = build_engine(res, "F4", traffic={})
    facts = snapshot_facts(probe)
    chk = ris_path_check(probe.model)
    page = _page_dur(probe)
    rate = probe.groups[0].rate
    probe.close()
    tx_ids = [s for s, c in probe.configs.items() if c.direction == "tx"]
    rx_ids = [s for s, c in probe.configs.items() if c.direction == "rx"]
    fc = float(next(iter(probe.configs.values())).fc_hz)
    warm = _round_pages(Fraction(warmup_s), page)

    def one(dur: Fraction) -> dict[str, Any]:
        n_total = int((warm + dur) * rate)
        n_warm = int(warm * rate)
        rec_dir = Path(tempfile.mkdtemp(prefix="sremu_b2_zmq_"))
        eng = build_engine(res, "F4", traffic={}, always_active=False, unopened_tx="wait")
        sink, close_rec = make_recorder(rec_dir, eng, iq=True, events=True, metrics=True)
        eng.add_sink(sink)
        cfg = ServerConfig(
            control_endpoint="tcp://127.0.0.1:0",
            pub_endpoint=None,
            tx_endpoint="tcp://127.0.0.1:0",
            start_paused=True,
        )
        srv = ControlServer(eng, cfg).start()
        try:
            ep = str(srv.endpoints["control"])
            common = [
                "--endpoint",
                ep,
                "--fc",
                repr(fc),
                "--rate",
                str(d["nodes"][0]["streams"][0]["rate_sps"]),
            ]
            tx_cmd = [
                sys.executable,
                "-m",
                "bench._zmq_client",
                "tx",
                *common,
                "--streams",
                ",".join(tx_ids),
                "--n",
                str(n_total),
                "--tone",
                repr(float(rate) / 10.0),
            ]
            rx_cmd = [
                sys.executable,
                "-m",
                "bench._zmq_client",
                "rx",
                *common,
                "--streams",
                ",".join(rx_ids),
                "--n",
                str(n_total),
                "--warm",
                str(n_warm),
            ]
            procs = [
                subprocess.Popen(
                    c, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                for c in (rx_cmd, tx_cmd)
            ]
            c_role = {id(p): r for p, r in zip(procs, ("rx", "tx"), strict=True)}
            outs = []
            for p in procs:
                o, e = p.communicate(timeout=max(600.0, point_wall_cap_s * 4))
                if p.returncode != 0:
                    role = c_role[id(p)]
                    raise RuntimeError(f"ZMQ client failed ({role}): {e.strip()[-2000:]}")
                outs.append(o)
            rx_res = json.loads(outs[0].strip().splitlines()[-1])
            counters = dict(srv.counters.snapshot())
        finally:
            srv.stop()
            close_rec()
            eng.close()
            shutil.rmtree(rec_dir, ignore_errors=True)
        virt = float(dur)
        wall = rx_res["wall_after_warm_s"]
        return {
            "virtual_s": virt,
            "rtf": virt / wall if wall > 0 else None,
            "wall_s": wall,
            "aggregate_sps": rx_res["samples_after_warm_total"] / wall if wall > 0 else None,
            "transport_counters": counters,
        }

    pilot_dur = _round_pages(Fraction(min_virtual_s) / 3, page)
    pilot = one(pilot_dur)
    rtf_p = pilot["rtf"] or 1e-9
    below = rtf_floor is not None and rtf_p < rtf_floor
    dur_pages = _round_pages(page * target_pages, page)
    dur_wall = _round_pages(Fraction(rtf_p * rep_wall_target_s).limit_denominator(10**6), page)
    dur = max(min(dur_pages, dur_wall), _round_pages(Fraction(min_virtual_s), page))
    n_reps = 0 if below else reps
    while n_reps > 1 and float(dur) / rtf_p * n_reps > point_wall_cap_s:
        n_reps -= 1
    out_reps = [one(dur) for _ in range(n_reps)]
    rtfs = [r["rtf"] for r in out_reps if r["rtf"] is not None]
    agg = [r["aggregate_sps"] for r in out_reps if r["aggregate_sps"] is not None]
    bad = {
        k: max(int(r["transport_counters"].get(k, 0)) for r in out_reps)
        for k in ("bad_packets", "seq_errors")
    }
    return {
        "status": "below_floor" if below else "ok",
        "rtf_floor": rtf_floor,
        "scenario_sha256": res.sha256(),
        "resolved": res,
        "features": "F5",
        "build_s": None,
        "warmup_virtual_s": float(warm),
        "warmup_wall_s": None,
        "pilot_rtf": pilot["rtf"],
        "pilot_virtual_s": pilot["virtual_s"],
        "pilot_truncated_by_wall": False,
        "reps": len(out_reps),
        "rep_virtual_s": float(dur),
        "rep_pages": int(dur / page),
        "rtf_median": statistics.median(rtfs) if rtfs else None,
        "rtf_reps": rtfs,
        "rtf_min_worst": None,
        "aggregate_msps_median": statistics.median(agg) / 1e6 if agg else None,
        "counters": bad,
        "verify_interp_max_rel": None,
        "nondeterministic_inputs": None,
        "recorded_bytes": None,
        "ris_check": chk,
        "facts": facts,
        "rtf_source": "RX client process: virtual time read after warm-up / wall time",
    }


# =========================================================================== paced


def _restart_paced_stats(eng: Any, at_sample: int, state: dict[str, Any]) -> Any:
    """``on_page`` hook restarting the pacer's deadline statistics, the RTF meter and the
    counter baseline once every RX stream has passed ``at_sample`` (end of the warm-up).

    Uses ``Emulator._pacer`` (the paced run's :class:`~sremu.engine.pacer.Pacer`, created by
    ``run_paced``); the hook fails loudly if that attribute disappears."""
    from sremu.metrics.rt import DeadlineStats

    def hook(rx: str, k0: int, k1: int) -> None:
        if state.get("done"):
            return
        state.setdefault("passed", set())
        if k1 >= at_sample:
            state["passed"].add(rx)
        if len(state["passed"]) < len(eng.rx):
            return
        pacer = eng._pacer
        if pacer is None:  # pragma: no cover - engine internals changed
            raise RuntimeError("Emulator._pacer missing: cannot restart paced statistics")
        pacer.stats = DeadlineStats(pacer.slacks_ns)
        c0 = dict(eng.counters)
        # the engine folds the RX queues' overflow counts into its counters only at the end
        # of a paced run: take the live value as the baseline
        c0["overflow_pages"] = sum(int(r.overflow_pages) for r in eng.rx.values())
        state["counters0"] = c0
        state["wall0_ns"] = eng.clock.now_ns()
        state["done"] = True
        eng.reset_metrics()

    return hook


def measure_paced(
    scenario: Mapping[str, Any],
    *,
    features: str = "F1",
    warmup_s: str = "5",
    measure_s: str = "60",
) -> dict[str, Any]:
    """One paced repetition (section 12.3): warm-up then ``measure_s`` measured seconds.

    Headless (in-process TX generators; RX pages delivered to the engine's RX history; no
    transport). Sustainable (section 12.1) = miss rate < 1e-3 at 1 ms slack and zero
    ``late_samples``/``overflow_pages`` over the measured window."""
    from bench._scenes import ris_path_check

    res = _resolve(scenario)
    prewarm(res, features)
    eng = build_engine(res, features)
    facts = snapshot_facts(eng)
    chk = ris_path_check(eng.model)
    g = eng.groups[0]
    warm = Fraction(warmup_s)
    total = warm + Fraction(measure_s)
    state: dict[str, Any] = {}
    eng.on_page(_restart_paced_stats(eng, math.ceil(warm * g.rate), state))
    w0 = time.perf_counter()
    try:
        s = eng.run_paced(until_s=total)
        pacer = eng._pacer
        hist = None if pacer is None else pacer.stats.latency.to_dict()
        decisions = _decisions(eng)
    finally:
        eng.close()
    wall = time.perf_counter() - w0
    paced = s.paced or {}
    c0 = state.get("counters0", {})
    counters = {k: int(v) - int(c0.get(k, 0)) for k, v in s.counters.items()}
    mr = paced.get("miss_rate", {})
    miss_1ms = mr.get("1ms")
    sustainable = (
        miss_1ms is not None
        and miss_1ms < 1e-3
        and counters.get("late_samples", 0) == 0
        and counters.get("overflow_pages", 0) == 0
    )
    return {
        "scenario_sha256": res.sha256(),
        "resolved": res,
        "features": features,
        "warmup_s": float(warm),
        "measured_s": float(Fraction(measure_s)),
        "warmup_restart_ok": bool(state.get("done")),
        "wall_s": wall,
        "pages": paced.get("pages"),
        "misses": paced.get("misses"),
        "miss_rate": mr,
        "latency_us": paced.get("latency_us"),
        "utilization": paced.get("utilization"),
        "counters": counters,
        "rtf": s.rtf.get("rtf"),
        "rtf_min": s.rtf.get("rtf_min"),
        "sustainable": sustainable,
        "latency_hist_ns": hist,
        "ris_check": chk,
        "facts": facts,
        "decisions": decisions,
    }
