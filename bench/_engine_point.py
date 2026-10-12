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
(:mod:`bench._zmq_client`): one sends every TX stream, one reads every RX stream. The F5 RTF
is end-to-end: virtual time after the warm-up delivered to the RX client over the wall time
from the served engine's warm-up boundary (server clock) to the RX client's last sample
(client clock); see :func:`measure_zmq` for why a starved or buffering client cannot
inflate it, and for the client-only and server-only figures recorded beside it.

:func:`measure_paced` runs ``Emulator.run_paced`` headless: 5 s warm-up then the measured
window (section 12.3), with the pacer's deadline statistics restarted at the end of the
warm-up (see :func:`_restart_paced_stats`).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from ._common import REPO

__all__ = [
    "F5_METRIC",
    "ServerMarks",
    "build_engine",
    "f5_rates",
    "measure_lockstep",
    "measure_paced",
    "measure_zmq",
    "prewarm",
    "run_in_thread",
    "snapshot_facts",
    "threading_record",
]


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


def run_in_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)`` in a throwaway thread (joined; its exception re-raised).

    With numba's OpenMP threading layer every thread that runs a parallel region keeps its
    own worker pool while it lives. JIT warm-ups therefore run here: a pool left in the main
    thread would compete for the CPUs with the pool of a served engine thread (two pools of
    ``NUMBA_NUM_THREADS`` on one CPU set; libgomp then lets idle workers sleep at once, so
    every parallel region of the engine waits for its workers to wake: ~870 us instead of
    ~200 us of compute per 1 ms page in the B4 scene). The throwaway thread's pool ends with
    it, so the thread that runs the engine owns the process's only pool."""
    box: dict[str, Any] = {}

    def body() -> None:
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as exc:  # re-raised in the caller
            box["error"] = exc

    th = threading.Thread(target=body, name="bench-warmup", daemon=True)
    th.start()
    th.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def prewarm(resolved: Any, features: str = "F1", pages: int = 64) -> None:
    """JIT pre-warm: run ``pages`` pages of a throwaway lockstep engine of the same scenario
    (numba compiles or loads its kernels once per process), so compilation cannot leave a
    backlog in a paced run that follows. Runs in a throwaway thread (:func:`run_in_thread`):
    the main thread starts no OpenMP pool."""

    def warm() -> None:
        pre = build_engine(resolved, features, mode="lockstep", coupling="barrier")
        g = pre.groups[0]
        pre.run(until_s=Fraction(pages * g.page) / g.rate)
        pre.close()

    run_in_thread(warm)


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
            warm_wall_max_s=warm_wall_max_s,
            pilot_wall_max_s=pilot_wall_max_s,
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
        "threading": threading_record(),
    }


def threading_record() -> dict[str, Any]:
    """:func:`sremu.sysinfo.threading_info` of this process now (numba threading layer, OS
    threads, render driver calls and the threads that ran the parallel driver)."""
    from sremu.sysinfo import threading_info

    return threading_info()


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

F5_METRIC = (
    "end-to-end: virtual time after the warm-up delivered to the RX client process (min over "
    "RX streams) / wall time from the served engine's warm-up boundary (server clock) to the "
    "RX client's last sample (client clock); perf_counter_ns, system-wide"
)
"""The F5 RTF definition (``rtf_source`` of every F5 point)."""
CLIENT_GRACE_S = 90.0
"""Wall time a client process may take beyond its own deadline (start, imports, connect,
teardown) before the benchmark kills it and fails the point."""
STALL_IDLE_S = 30.0
"""A repetition whose RX client read nothing during the last ``STALL_IDLE_S`` before its
wall deadline fails as a stall (never reported as a truncated measurement)."""
TX_CHUNK_SAMPLES = 4096
"""Samples per ``send`` of the F5 TX client (recorded in every F5 point as
``tx_chunk_samples``; one TX frame per call)."""


class ServerMarks:
    """``Emulator.on_page`` hook of a served F5 engine (runs on the engine thread).

    ``t_warm_ns``: ``time.perf_counter_ns()`` (benchmark process) at which every RX stream
    has computed and queued its first ``n_warm`` samples, i.e. the end of the last warm-up
    page. Lockstep computes pages in order on that thread, so no sample after the warm-up
    is computed, let alone delivered, before ``t_warm_ns``. ``progress()`` gives the samples
    computed on every RX stream and the wall time of the last computed page."""

    def __init__(self, rx_ids: Sequence[str], n_warm: int) -> None:
        self.n_warm = int(n_warm)
        self._ids = frozenset(rx_ids)
        self._warm: set[str] = set()
        self.k: dict[str, int] = dict.fromkeys(rx_ids, 0)
        self.t_ns: dict[str, int] = {}
        self.t_warm_ns: int | None = None

    def __call__(self, rx: str, k0: int, k1: int) -> None:
        if rx not in self._ids:
            return
        now = time.perf_counter_ns()
        if k1 > self.k[rx]:
            self.k[rx] = k1
            self.t_ns[rx] = now
        if self.t_warm_ns is None and k1 >= self.n_warm:
            self._warm.add(rx)
            if len(self._warm) == len(self._ids):
                self.t_warm_ns = now

    def progress(self) -> tuple[int, int | None]:
        """(samples computed on every RX stream, wall time of the last computed page)."""
        return min(self.k.values(), default=0), max(self.t_ns.values(), default=None)


def f5_rates(
    client: Mapping[str, Any],
    server: Mapping[str, Any],
    *,
    n_warm: int,
    rate: Fraction | float,
    t_launch_ns: int,
    t_done_ns: int,
) -> dict[str, Any]:
    """The rates of one served F5 run (see :func:`measure_zmq`).

    ``client``: the RX client's JSON line (:mod:`bench._zmq_client`); ``server``:
    ``{"t_warm_ns", "k_min", "t_last_ns"}`` from :class:`ServerMarks`; ``t_launch_ns`` /
    ``t_done_ns``: the benchmark process's clock just before it started the clients and
    just after the RX client exited. Returns:

    * ``rtf`` (end-to-end, the F5 figure): ``v / (t_end_client - t_warm_server)`` with ``v``
      the virtual time after the warm-up that reached the RX client (min over streams);
    * ``rtf_rx_client`` (diagnostic, the pre-fix F5 figure): ``v`` over the client's own
      span from reading its warm-up samples to its last sample;
    * ``rtf_server`` (diagnostic): virtual time computed after the warm-up over the engine's
      span from the warm-up boundary to its last computed page;
    * ``rtf_whole_run`` (sizing fallback): everything delivered, warm-up included, over the
      client's span from the resume to its last sample;
    * ``clock_ok``: the client's timestamps are ordered with the benchmark's own
      (``t_launch <= t_start <= t_resume <= t_end <= t_done`` and ``t_warm_server <=
      t_end``). If not (process clocks not comparable), ``rtf`` falls back to the smaller of
      the server-only and client-only figures (``rtf_basis`` says so)."""
    rate_f = float(rate)
    got_min = int(client["got_min"])
    v = max(0, got_min - int(n_warm)) / rate_f
    t_end = int(client["t_end_ns"])
    t_cw = client.get("t_warm_ns")
    t_sw = server.get("t_warm_ns")
    t_sl = server.get("t_last_ns")

    def ratio(virt: float, wall_s: float | None) -> float | None:
        return virt / wall_s if wall_s is not None and wall_s > 0 and virt > 0 else None

    wall_c = None if t_cw is None else (t_end - int(t_cw)) * 1e-9
    v_s = max(0, int(server.get("k_min", 0)) - int(n_warm)) / rate_f
    wall_s = None if t_sw is None or t_sl is None else (int(t_sl) - int(t_sw)) * 1e-9
    wall_w = (t_end - int(client["t_resume_ns"])) * 1e-9
    clock_ok = bool(
        t_launch_ns <= int(client["t_start_ns"]) <= int(client["t_resume_ns"]) <= t_end <= t_done_ns
        and (t_sw is None or v <= 0 or int(t_sw) <= t_end)
    )
    rtf_c = ratio(v, wall_c)
    rtf_s = ratio(v_s, wall_s)
    if clock_ok:
        wall_e = None if t_sw is None else (t_end - int(t_sw)) * 1e-9
        rtf_e = ratio(v, wall_e)
        basis = "end-to-end"
    else:
        cands = [x for x in (rtf_s, rtf_c) if x is not None]
        rtf_e = min(cands) if cands else None
        wall_e = v / rtf_e if rtf_e else None
        basis = "min(server, rx client): process clocks not comparable"
    return {
        "virtual_s": v,
        "rtf": rtf_e,
        "wall_s": wall_e,
        "rtf_basis": basis,
        "rtf_rx_client": rtf_c,
        "wall_rx_client_s": wall_c,
        "rtf_server": rtf_s,
        "wall_server_s": wall_s,
        "virtual_server_s": v_s,
        "rtf_whole_run": ratio(got_min / rate_f, wall_w),
        "aggregate_sps": (
            float(client.get("samples_after_warm_total", 0)) / wall_e if wall_e else None
        ),
        "clock_ok": clock_ok,
    }


def _served_state(srv: Any, eng: Any, n_total: int) -> str:
    """Where a served run stood when a client failed (for the error message): per rate
    group the cursor, end and TX streams waited on, RX samples delivered, engine error."""
    groups = "; ".join(
        f"cursor {g.cursor} end {g.end} waiting_on {list(g.waiting_on)}" for g in eng.groups
    )
    rx = {r: int(st.delivered) for r, st in eng.rx.items()}
    return (
        f"server: {groups}; streamed {n_total}/stream; rx delivered {rx}; "
        f"paused {srv.paused}; engine_error {srv.engine_error!r}"
    )


def _last_json(text: str) -> dict[str, Any]:
    lines = text.strip().splitlines()
    if not lines:
        raise RuntimeError("ZMQ client printed no result line")
    out: dict[str, Any] = json.loads(lines[-1])
    return out


def _median(xs: Sequence[float | None]) -> float | None:
    ys = [x for x in xs if x is not None]
    return statistics.median(ys) if ys else None


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
    warm_wall_max_s: float = 20.0,
    pilot_wall_max_s: float = 10.0,
    inject: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """F5: lockstep served over native ZMQ to a TX client process and an RX client process.

    **Metric.** F5 measures the sustained end-to-end rate of the served engine with real
    client processes: the virtual time after the warm-up that reached the RX client (min
    over RX streams) divided by the wall time over which those samples were both computed by
    the server and delivered to the RX client. That span starts at the served engine's
    warm-up boundary, timestamped on the server (:class:`ServerMarks`: every RX stream has
    computed its first ``n_warm`` samples; lockstep computes no later sample before it), and
    ends when the RX client read its last sample, timestamped on the client. Both are
    ``time.perf_counter_ns()``, which is system-wide (Linux ``CLOCK_MONOTONIC``, Windows
    ``QueryPerformanceCounter`` since Python 3.10), and every run checks the client's
    timestamps against the benchmark's own launch and completion times (``clock_shared``).

    The span contains every post-warm-up computation and delivery, so a client cannot shrink
    it: a client starved before its warm-up completes, whose samples the server keeps
    computing into the loopback/ZMQ buffers, reads them later in a burst, which can only end
    the span later. A starvation that delays the last sample lowers the figure (the samples
    did reach the application that late); one that does not leaves it unchanged. The pre-fix
    F5 figure, which timed only the RX client's own post-warm-up reads (from the moment it
    had *read* its warm-up samples), is kept as ``rtf_rx_client`` (diagnostic): it agrees
    with ``rtf`` for an unstarved client and over-states it when the server ran ahead into
    the buffers while the client was starved (50x in a WSL2 test with a 3 s pause). The
    engine-only figure ``rtf_server`` (warm-up boundary to the last computed page, server
    clock) is recorded too; ``rtf <= min(rtf_server, rtf_rx_client)`` up to clock jitter.

    **Runs.** Each repetition (and the pilot) is a fresh served engine whose run lasts
    exactly ``warmup + dur`` (the clients send/read that many samples per stream), whatever
    the scenario's ``run.duration_s``. The recorder of F4 stays attached (cumulative feature
    sets). The pilot (``min_virtual_s / 3`` after the warm-up, at most ``warm_wall_max_s +
    pilot_wall_max_s`` of wall time) is measured with the same end-to-end metric (falling
    back to ``rtf_whole_run`` when its warm-up did not complete within that limit) and sizes
    ``dur`` as in :func:`measure_lockstep`. With ``rtf_floor`` set and the pilot below it the
    point is ``below_floor`` (no repetitions).

    **Wall cap.** ``point_wall_cap_s`` bounds the whole point, pilot included. A repetition
    is estimated as the pilot's wall time plus its extra virtual time at the pilot rate; the
    repetition count is reduced (then ``dur``, down to ``min_virtual_s``) to fit the cap.
    Every run also has a hard deadline: the RX client stops reading ``wall_max`` seconds
    after the resume (``min(remaining cap, max(2 x estimate, estimate + 5 s))``) and the run
    is then measured over what it delivered (``truncated_by_wall``); a repetition that would
    start with less than its estimate left is skipped (``reps_skipped_cap``); a client still
    running ``CLIENT_GRACE_S`` after its deadline is killed and fails the point, as does a
    truncated run whose RX client was idle for ``STALL_IDLE_S`` before its deadline (stall)
    or whose TX client failed. ``counters`` are the transport's worst ``bad_packets`` and
    ``seq_errors`` over the pilot and the repetitions.

    **Client CPUs.** The clients pin themselves to ``$SREMU_CLIENT_CPUS`` when it is set
    (:func:`bench.pin_client_cpus`; B2 sets it per slot) and report the affinity they ran
    with (``client_affinity``).

    ``inject`` (tests, stress): ``rx_pause_s`` / ``rx_pause_at`` (the RX client sleeps once
    after every stream read ``rx_pause_at`` samples; default 0, before its warm-up ends),
    ``server_page_sleep_s`` (the served engine sleeps after each computed page: a known
    engine cost), ``rep_server_page_sleep_s`` (the same in the repetitions only: a pilot
    that over-states the rate)."""
    from bench._scenes import ris_path_check
    from sremu.commands.run import make_recorder
    from sremu.transport.control_server import ControlServer, ServerConfig

    t_point0 = time.perf_counter()

    def left() -> float:
        return point_wall_cap_s - (time.perf_counter() - t_point0)

    inj = dict(inject or {})
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
    n_warm = int(warm * rate)

    def one(dur: Fraction, wall_max_s: float, page_sleep_s: float) -> dict[str, Any]:
        w_one = time.perf_counter()
        n_total = int((warm + dur) * rate)
        rec_dir = Path(tempfile.mkdtemp(prefix="sremu_b2_zmq_"))
        # the served run must last as long as the clients stream: with the scenario's
        # run.duration_s (1 s in the bench scenes) a pilot RTF above ~0.16 sized dur past
        # the end of the run, the server ended the RX streams there and the RX client
        # waited its whole recv timeout for samples that are never computed (B2 run
        # 20261011T084048Z; tests/bench/test_bench_zmq_point.py)
        eng = build_engine(
            res,
            "F4",
            traffic={},
            always_active=False,
            unopened_tx="wait",
            duration_s=warm + dur,
        )
        marks = ServerMarks(rx_ids, n_warm)
        eng.on_page(marks)
        if page_sleep_s > 0:
            first = rx_ids[0]

            def _load(rx: str, k0: int, k1: int) -> None:
                if rx == first:
                    time.sleep(page_sleep_s)

            eng.on_page(_load)
        sink, close_rec = make_recorder(rec_dir, eng, iq=True, events=True, metrics=True)
        eng.add_sink(sink)
        cfg = ServerConfig(
            control_endpoint="tcp://127.0.0.1:0",
            pub_endpoint=None,
            tx_endpoint="tcp://127.0.0.1:0",
            start_paused=True,
        )
        srv = ControlServer(eng, cfg).start()
        # RX frame length the served streams use (lockstep batches several pages per frame)
        frame_for = getattr(srv, "rx_frame_for", None)
        rx_frame = int(frame_for(eng.groups[0].page)) if callable(frame_for) else None
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
                "--chunk",
                str(TX_CHUNK_SAMPLES),
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
                "--wall-max-s",
                f"{wall_max_s:.3f}",
            ]
            if inj.get("rx_pause_s"):
                rx_cmd += [
                    "--pause-at",
                    str(int(inj.get("rx_pause_at", 0))),
                    "--pause-s",
                    repr(float(inj["rx_pause_s"])),
                ]
            t_launch = time.perf_counter_ns()
            rx_p, tx_p = (
                subprocess.Popen(
                    c, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                for c in (rx_cmd, tx_cmd)
            )
            tx_res: dict[str, Any] | None = None
            try:
                try:
                    o, e = rx_p.communicate(timeout=wall_max_s + CLIENT_GRACE_S)
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"ZMQ RX client still running {wall_max_s + CLIENT_GRACE_S:.0f} s after "
                        f"its launch (deadline {wall_max_s:.0f} s after the resume) "
                        f"[{_served_state(srv, eng, n_total)}]"
                    ) from None
                t_done = time.perf_counter_ns()

                def tx_failure() -> str | None:
                    if tx_p.poll() in (None, 0):
                        return None
                    eq = tx_p.communicate()[1].strip()[-2000:]
                    return f"tx client exit {tx_p.returncode}: {eq}"

                if rx_p.returncode != 0:
                    msg = f"ZMQ client failed (rx): {e.strip()[-2000:]}"
                    why = tx_failure()  # e.g. an RX timeout caused by a failed TX client
                    if why:
                        msg += f" [{why}]"
                    raise RuntimeError(f"{msg} [{_served_state(srv, eng, n_total)}]")
                rx_res = _last_json(o)
                if rx_res["truncated"]:
                    why = tx_failure()
                    if why:
                        raise RuntimeError(
                            f"ZMQ client failed (tx) before the RX deadline: {why} "
                            f"[{_served_state(srv, eng, n_total)}]"
                        )
                    if float(rx_res["idle_at_stop_s"]) > STALL_IDLE_S:
                        raise RuntimeError(
                            f"ZMQ RX client read nothing during the last "
                            f"{rx_res['idle_at_stop_s']:.1f} s before its deadline (stall; "
                            f"got {rx_res['got']} of {n_total}) "
                            f"[{_served_state(srv, eng, n_total)}]"
                        )
                    # the served run ends early: the TX client is killed below
                else:
                    try:
                        o2, e2 = tx_p.communicate(timeout=CLIENT_GRACE_S)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(
                            f"ZMQ TX client still running {CLIENT_GRACE_S:.0f} s after the RX "
                            f"client completed [{_served_state(srv, eng, n_total)}]"
                        ) from None
                    if tx_p.returncode != 0:
                        raise RuntimeError(
                            f"ZMQ client failed (tx): {e2.strip()[-2000:]} "
                            f"[{_served_state(srv, eng, n_total)}]"
                        )
                    tx_res = _last_json(o2)
            finally:  # never leave a client behind (e.g. blocked on a dead server)
                for p in (rx_p, tx_p):
                    if p.poll() is None:
                        p.kill()
                        p.communicate()
            counters = dict(srv.counters.snapshot())
        finally:
            srv.stop()
            close_rec()
            eng.close()
            shutil.rmtree(rec_dir, ignore_errors=True)
        k_min, t_last = marks.progress()
        rates = f5_rates(
            rx_res,
            {"t_warm_ns": marks.t_warm_ns, "k_min": k_min, "t_last_ns": t_last},
            n_warm=n_warm,
            rate=rate,
            t_launch_ns=t_launch,
            t_done_ns=t_done,
        )
        return {
            **rates,
            "virtual_s_planned": float(dur),
            "delivered_virtual_s": int(rx_res["got_min"]) / float(rate),
            "truncated_by_wall": bool(rx_res["truncated"]),
            "wall_max_s": wall_max_s,
            "run_wall_s": time.perf_counter() - w_one,
            "max_gap_s": rx_res.get("max_gap_s"),
            "rx_pause_s": rx_res.get("pause_s"),
            "rx_frame_samples": rx_frame,
            "transport_counters": counters,
            "client_affinity": {
                "rx": rx_res.get("affinity"),
                "tx": None if tx_res is None else tx_res.get("affinity"),
            },
            "timestamps_ns": {
                "launch": t_launch,
                "server_warm": marks.t_warm_ns,
                "server_last_page": t_last,
                "client_start": rx_res.get("t_start_ns"),
                "client_resume": rx_res.get("t_resume_ns"),
                "client_warm": rx_res.get("t_warm_ns"),
                "client_end": rx_res.get("t_end_ns"),
                "done": t_done,
            },
        }

    sleep_all = float(inj.get("server_page_sleep_s", 0.0))
    sleep_reps = float(inj.get("rep_server_page_sleep_s", 0.0)) or sleep_all
    pilot_dur = _round_pages(Fraction(min_virtual_s) / 3, page)
    pilot_max = max(1.0, min(warm_wall_max_s + pilot_wall_max_s, left()))
    pilot = one(pilot_dur, pilot_max, sleep_all)
    pilot_basis = pilot["rtf_basis"]
    rtf_p = pilot["rtf"]
    if rtf_p is None:  # warm-up not completed within the pilot's wall limit
        rtf_p = pilot["rtf_whole_run"]
        pilot_basis = "whole run (warm-up not completed within the pilot's wall limit)"
    rtf_p = rtf_p or 1e-9
    below = rtf_floor is not None and rtf_p < rtf_floor
    min_dur = _round_pages(Fraction(min_virtual_s), page)
    dur_pages = _round_pages(page * target_pages, page)
    dur_wall = _round_pages(Fraction(rtf_p * rep_wall_target_s).limit_denominator(10**6), page)
    dur = max(min(dur_pages, dur_wall), min_dur)

    def estimate(dur_: Fraction) -> float:
        # the pilot's wall time (start-up, warm-up, its own samples, teardown) plus the
        # virtual time a repetition has beyond what the pilot delivered, at the pilot rate
        extra = float(warm + dur_) - pilot["delivered_virtual_s"]
        return float(pilot["run_wall_s"]) + max(0.0, extra) / rtf_p

    est = estimate(dur)
    n_reps = 0 if below else reps
    while n_reps > 1 and est * n_reps > left():
        n_reps -= 1
    if n_reps == 1 and est > left() and dur > min_dur:  # shrink dur so one repetition fits
        room = (left() - float(pilot["run_wall_s"])) * rtf_p + pilot["delivered_virtual_s"]
        fit = Fraction(max(0.0, room - float(warm))).limit_denominator(10**6)
        dur = max(min_dur, min(dur, math.floor(fit / page) * page))
        est = estimate(dur)
    out_reps: list[dict[str, Any]] = []
    skipped = 0
    for i in range(n_reps):
        rem = left()
        if rem < est and (out_reps or rem < 5.0):  # it could not complete within the cap
            skipped = n_reps - i
            break
        out_reps.append(one(dur, max(1.0, min(rem, max(2.0 * est, est + 5.0))), sleep_reps))
    runs = (pilot, *out_reps)
    # over the pilot too: a below-floor point has no repetitions (max() of nothing raised)
    bad = {
        k: max(int(r["transport_counters"].get(k, 0)) for r in runs)
        for k in ("bad_packets", "seq_errors")
    }
    rtfs = [r["rtf"] for r in out_reps if r["rtf"] is not None]
    agg = [r["aggregate_sps"] for r in out_reps if r["aggregate_sps"] is not None]
    return {
        "status": "below_floor" if below else "ok",
        "rtf_floor": rtf_floor,
        "scenario_sha256": res.sha256(),
        "resolved": res,
        "features": "F5",
        "build_s": None,
        "warmup_virtual_s": float(warm),
        "warmup_wall_s": None,
        "pilot_rtf": rtf_p if pilot["rtf"] is not None or pilot["rtf_whole_run"] else None,
        "pilot_rtf_basis": pilot_basis,
        "pilot_rtf_rx_client": pilot["rtf_rx_client"],
        "pilot_rtf_server": pilot["rtf_server"],
        "pilot_virtual_s": pilot["virtual_s"],
        "pilot_wall_s": pilot["run_wall_s"],
        "pilot_truncated_by_wall": pilot["truncated_by_wall"],
        "reps": len(out_reps),
        "reps_skipped_cap": skipped,
        "reps_truncated": sum(1 for r in out_reps if r["truncated_by_wall"]),
        "est_rep_wall_s": est,
        "rep_virtual_s": float(dur),
        "rep_virtual_s_measured": [r["virtual_s"] for r in out_reps],
        "rep_pages": int(dur / page),
        "rtf_median": statistics.median(rtfs) if rtfs else None,
        "rtf_reps": rtfs,
        "rtf_rx_client_median": _median([r["rtf_rx_client"] for r in out_reps]),
        "rtf_rx_client_reps": [r["rtf_rx_client"] for r in out_reps],
        "rtf_server_median": _median([r["rtf_server"] for r in out_reps]),
        "rtf_server_reps": [r["rtf_server"] for r in out_reps],
        "rtf_min_worst": None,
        "aggregate_msps_median": statistics.median(agg) / 1e6 if agg else None,
        "counters": bad,
        "verify_interp_max_rel": None,
        "nondeterministic_inputs": None,
        "recorded_bytes": None,
        "ris_check": chk,
        "facts": facts,
        "point_wall_s": time.perf_counter() - t_point0,
        "point_wall_cap_s": point_wall_cap_s,
        "clock": time.get_clock_info("perf_counter").implementation,
        "clock_shared": all(r["clock_ok"] for r in runs),
        "client_cpus_requested": os.environ.get("SREMU_CLIENT_CPUS") or None,
        "client_affinity": (out_reps[0] if out_reps else pilot)["client_affinity"],
        "rx_frame_samples": pilot["rx_frame_samples"],
        "tx_chunk_samples": TX_CHUNK_SAMPLES,
        "rtf_source": F5_METRIC,
        "f5_runs": [{"run": "pilot", **pilot}]
        + [{"run": f"rep{i}", **r} for i, r in enumerate(out_reps)],
        "threading": threading_record(),
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
    transport), so the engine keeps the lockstep render driver threshold
    (``render_min_work``: ``Emulator.render_parallel_min_work``). Sustainable (section 12.1)
    = miss rate < 1e-3 at 1 ms slack and zero ``late_samples``/``overflow_pages`` over the
    measured window."""
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
        threading_rec = threading_record()
        min_work = eng.render_parallel_min_work  # headless paced: the lockstep threshold
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
        "threading": threading_rec,
        "render_min_work": min_work,
    }
