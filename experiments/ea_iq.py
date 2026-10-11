"""E-A IQ consistency subset (VA-1/W5; DESIGN.md sections 4.11, 11 E-A). A consistency
check of the batch path, not a finding.

1 GHz, h = 2 m, LOS partition 20 dB, tau_SUI in {0.1, 1} s, {ExactOptimum, Urap}, 1000 s,
10 kS/s tone. Three OS processes per run:

* ``serve``: the emulator (lockstep, barrier coupling) served over the control socket and
  the native ZMQ data plane (``sremu.transport.control_server.ControlServer``), with the
  scenario's tone generator on ``ap.tx0`` and the controller declared ``external``; noise
  on or off (``EngineConfig.noise``);
* ``controller``: the controller (``DelayedController(ExactOptimum)`` with the 1 cm
  ``GridChannel``, or ``Urap(0.2)``) through ``sremu.control.remote.run_remote``;
* ``rxapp``: ``Device.connect`` + ``mu.rx0`` stream from t = 0, mean power over every
  10 ms window (100 samples), written to CSV.

The batch reference is ``sremu.engine.batch.run_batch`` on the same scenario (5 ms grid,
pieces weighted by 10 kS/s sample counts), same controller class and seed. Acceptance:
|IQ - batch| <= 0.02 dB with noise off; with noise on, the batch value inside the 95 %
batch-means interval of the IQ window powers (20 batches).

    python -m experiments.ea_iq run --out runs/wp11/ea_iq        # all 8 runs + aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ea  # noqa: E402
from experiments.common import (  # noqa: E402
    RunRecord,
    db10,
    export_git_info,
    quiet_warnings,
    read_csv,
    write_csv,
    write_json,
    write_meta,
)

FS = 10_000
WINDOW = 100  # 10 ms
DURATION_S = int(__import__("os").environ.get("SREMU_EAIQ_DURATION_S", "1000"))  # smoke tests only
POINTS = [("exact", "0.1"), ("exact", "1"), ("urap", "0.1"), ("urap", "1")]


def make_controller(kind: str, tau: str, cache_dir: Path) -> Any:
    """The controller object (same class and parameters as the batch reference)."""
    from sremu.contracts import LatencyModel

    if kind == "exact":
        geom = ea.build_geometry(1e9, 2, 20, cache_dir)
        ea.set_table_dir(geom, cache_dir)
        return ea.DelayedController(ea.make_inner("exact", geom), Fraction(tau))
    return ea.Urap(
        ea.URAP_P,
        panel=ea.PANEL,
        name="urap",
        period_s=Fraction(tau),
        min_latency_s=ea.LAT_S,
        latency=LatencyModel(ea.LAT_S),
    )


def scenario_for(kind: str, tau: str, cache_dir: Path) -> dict[str, Any]:
    """The corridor scenario with the controller declared ``external`` (fields from the
    controller object's own registration message, so the remote side matches)."""
    from sremu.control.remote import register_message

    d = ea.corridor_scenario(1e9, 2, 20, duration_s=str(DURATION_S))
    reg = register_message(make_controller(kind, tau, cache_dir))
    d["controllers"] = [
        {
            "name": reg["name"],
            "kind": "external",
            "period_s": reg["period_s"],
            "min_latency_s": reg["min_latency_s"],
            "latency": {"fixed_s": reg["latency"]["fixed_s"]},
            "observation": {
                "fields": reg["observation"]["fields"],
                "window_s": reg["observation"]["window_s"],
                "position_delay_s": reg["observation"]["position_delay_s"],
                "position_noise_m": reg["observation"]["position_noise_m"],
            },
        }
    ]
    d["name"] = f"ea_iq_{kind}_tau{tau.replace('.', 'p')}"
    return d


# =========================================================================== processes


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve the scenario (paused until sim.resume; exits when the run is done)."""
    from sremu.control.builtin import build_controller
    from sremu.engine.emulator import Emulator
    from sremu.scenario.load import load_scenario
    from sremu.transport.control_server import ControlServer, ServerConfig

    warnings.simplefilter("ignore")
    res = load_scenario(args.scenario)
    d = res.scenario.model_dump(mode="python")
    eng = Emulator.from_scenario(
        res, unopened_tx="wait", external_inputs=True, noise=args.noise == "on"
    )
    cfg = ServerConfig(
        control_endpoint="tcp://127.0.0.1:0",
        pub_endpoint=None,
        tx_endpoint="tcp://127.0.0.1:0",
        start_paused=True,
        exit_when_done=True,
    )
    server = ControlServer(
        eng, cfg, scenario=d, builtin_factory=lambda s: build_controller(s, model=eng.model)
    )
    server.start()
    print("SREMU-SERVE " + json.dumps({"endpoints": server.endpoints}), flush=True)
    t0 = time.perf_counter()
    try:
        while not server.wait(0.2):
            pass
    finally:
        server.stop()
    summ = eng.summary()
    info = eng.manifest_info()
    write_json(
        args.summary,
        {
            "engine_error": None if server.engine_error is None else str(server.engine_error),
            "wall_s": time.perf_counter() - t0,
            "samples": summ.samples,
            "counters": summ.counters,
            "rtf": summ.rtf,
            "nondeterministic_inputs": summ.nondeterministic_inputs,
            "verify_interp_max_rel": summ.verify_interp_max_rel,
            "manifest_info": info,
            "transport_counters": dict(server.counters.snapshot()),
            "n_records": len(eng.records),
            "late_records": sum(1 for r in eng.records if r.status == "late_applied"),
        },
    )
    return 0 if server.engine_error is None else 1


def cmd_controller(args: argparse.Namespace) -> int:
    """Run the controller remotely until the server stops sending barriers."""
    from sremu.control.remote import run_remote

    warnings.simplefilter("ignore")
    ctrl = make_controller(args.kind, args.tau, Path(args.cache))
    stats = run_remote(ctrl, args.endpoint, idle_timeout_s=30.0)
    print(json.dumps({"barriers": stats.barriers, "commands": stats.commands}), flush=True)
    sys.stderr.flush()
    os._exit(0)  # the server is gone: do not wait for ZMQ context termination (linger)


def cmd_rxapp(args: argparse.Namespace) -> int:
    """External RX application: mean power of every 10 ms window from t = 0."""
    from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd

    warnings.simplefilter("ignore")
    total = int(args.total)
    dev = Device.connect(args.endpoint)
    win = np.empty(total // WINDOW)
    buf = np.zeros(10_000, dtype=np.complex64)
    got_total = 0
    first_k = None
    t0 = time.perf_counter()
    with dev:
        rx = dev.get_rx_stream(StreamArgs("mu.rx0", 1e9, "1e4"))
        rx.issue_stream_cmd(StreamCmd(mode="num_done", num_samps=total, stream_now=False))
        print("RX-READY", flush=True)
        acc = np.zeros(0, dtype=np.complex64)
        while got_total < total:
            md = RxMetadata()
            n = rx.recv(buf, md, timeout=120.0)
            if n <= 0:
                raise RuntimeError(f"recv returned {n}, error {md.error_code}")
            if first_k is None:
                first_k = md.time_spec
            acc = np.concatenate([acc, buf[:n]])
            nw = acc.size // WINDOW
            if nw:
                seg = acc[: nw * WINDOW].astype(np.complex128).reshape(nw, WINDOW)
                i0 = got_total // WINDOW
                win[i0 : i0 + nw] = np.mean(np.abs(seg) ** 2, axis=1)
                got_total += nw * WINDOW
                acc = acc[nw * WINDOW :]
    rows = [{"window": i, "t_s": i * WINDOW / FS, "p_mw": float(win[i])} for i in range(win.size)]
    write_csv(args.out, rows)
    print(
        json.dumps(
            {
                "samples": got_total,
                "first_time_spec": str(first_k),
                "wall_s": time.perf_counter() - t0,
            }
        ),
        flush=True,
    )
    return 0


def _spawn(argv: list[str], log: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "experiments.ea_iq", *argv],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=open(log, "w"),
        text=True,
    )


def run_point(kind: str, tau: str, noise: str, out_root: Path, cache: Path) -> dict[str, Any]:
    """One IQ run (three processes) and its run directory."""
    quiet_warnings()
    out = out_root / f"{kind}_tau{tau}_noise{noise}"
    out.mkdir(parents=True, exist_ok=True)
    d = scenario_for(kind, tau, cache)
    res = ea.resolve(d)
    scen_path = out / "scenario.yaml"
    scen_path.write_text(res.to_yaml(), encoding="utf-8", newline="\n")
    run = RunRecord(
        out, ea.SEED, resolved=res, mode="lockstep", coupling="barrier", patch_root=out_root
    )
    srv = _spawn(
        ["serve", str(scen_path), "--noise", noise, "--summary", str(out / "server_summary.json")],
        out / "server.log",
    )
    assert srv.stdout is not None
    line = srv.stdout.readline()
    if not line.startswith("SREMU-SERVE "):
        srv.kill()
        raise RuntimeError(f"server did not start: {line!r}")
    ep = json.loads(line.split(" ", 1)[1])["endpoints"]["control"]
    rxp = _spawn(
        [
            "rxapp",
            "--endpoint",
            ep,
            "--total",
            str(DURATION_S * FS),
            "--out",
            str(out / "rx_windows.csv"),
        ],
        out / "rxapp.log",
    )
    assert rxp.stdout is not None
    if not rxp.stdout.readline().startswith("RX-READY"):
        raise RuntimeError("rx app did not start")
    ctl = _spawn(
        ["controller", "--endpoint", ep, "--kind", kind, "--tau", tau, "--cache", str(cache)],
        out / "controller.log",
    )
    time.sleep(3.0)  # let the controller attach before virtual time starts
    from sremu.transport.client import connect_engine

    t0 = time.perf_counter()
    eng = connect_engine(ep)
    eng.resume()
    eng.close()
    rx_out = rxp.stdout.read()
    rxp.wait()
    srv.wait(timeout=600)
    try:
        ctl_out, _ = ctl.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        ctl.kill()
        ctl_out = "(controller did not exit within 120 s after the run; killed)"
    wall = time.perf_counter() - t0
    run.sample_load()
    win = np.array([float(r["p_mw"]) for r in read_csv(out / "rx_windows.csv")])
    p_iq_mw = float(win.mean())
    from sremu.analysis.stats import batch_means

    bm = batch_means(win, 20)
    srv_sum = json.loads((out / "server_summary.json").read_text(encoding="utf-8"))
    res_doc = {
        "kind": kind,
        "tau_sui_s": tau,
        "noise": noise,
        "duration_s": DURATION_S,
        "p_iq_dbm": db10(p_iq_mw),
        "p_iq_ci_dbm": [db10(bm.ci_low), db10(bm.ci_high)],
        "n_windows": int(win.size),
        "wall_s": wall,
        "server": srv_sum,
        "controller_stdout": ctl_out.strip(),
        "rxapp_stdout": rx_out.strip(),
    }
    write_json(out / "result.json", res_doc)
    mi = srv_sum["manifest_info"]
    run.page_size = mi["page_size"]
    run.verify_interp_max_rel = mi["verify_interp_max_rel"]
    run.counters = {k: v for k, v in srv_sum["counters"].items() if isinstance(v, int)}
    run.nondeterministic_inputs = bool(srv_sum["nondeterministic_inputs"])
    run.oracle_csi_used = kind == "exact"
    run.add_assumption(
        f"noise {noise}; external controller process (barrier mode); external RX app"
    )
    write_meta(
        out,
        experiment="E-A-IQ",
        data_classes=["SIM-sremu"],
        description=f"E-A IQ consistency run: {kind}, tau_SUI {tau} s, noise {noise}, 1000 s",
        assumptions=run.assumptions,
        inputs=run.inputs,
        files={"rx_windows.csv": "RX power per 10 ms window (mW)", "result.json": "summary"},
    )
    run.finish()
    return res_doc


def batch_reference(kind: str, tau: str, out_root: Path, cache: Path) -> dict[str, Any]:
    """``run_batch`` over [0, 1000) s (5 ms grid, 10 kS/s sample weights)."""
    from sremu.engine.batch import run_batch

    quiet_warnings()
    out = out_root / f"batch_{kind}_tau{tau}"
    out.mkdir(parents=True, exist_ok=True)
    d = ea.corridor_scenario(1e9, 2, 20, duration_s=str(DURATION_S))
    res = ea.resolve(d)
    run = RunRecord(
        out, ea.SEED, resolved=res, mode="batch", coupling="barrier", patch_root=out_root
    )
    model = ea.from_scenario(d, backend="numpy_ref")
    ctrl = make_controller(kind, tau, cache)
    t0 = time.perf_counter()
    rb = run_batch(model, DURATION_S, dt_s=ea.DT_S, rate=FS, controllers=[ctrl], seed=ea.SEED)
    wall = time.perf_counter() - t0
    p = rb.mean_rx_power_w[ea.RX]
    geom = ea.build_geometry(1e9, 2, 20, cache)
    ea.set_table_dir(geom, cache)
    traj = ea.build_trajectory(d, 33_334.0, cache)
    n_cells = int(Fraction(DURATION_S) / ea.DT_S)
    vec = {
        rule: float(ea.run_controller(kind, geom, traj, tau, n_cells, rule=rule).p_cell.mean())
        for rule in ("mid", "left")
    }
    doc = {
        "kind": kind,
        "tau_sui_s": tau,
        "p_batch_dbm": db10(p / 1e-3),
        "pieces": len(rb.t),
        "wall_s": wall,
        "n_commands": sum(1 for r in rb.records if r.status in ("applied", "late_applied")),
        "p_vectorized_mid_dbm": db10(vec["mid"] / 1e-3),
        "p_vectorized_left_dbm": db10(vec["left"] / 1e-3),
    }
    write_json(out / "result.json", doc)
    run.add_assumption("run_batch: 5 ms grid, pieces weighted by 10 kS/s RX sample counts")
    write_meta(
        out,
        experiment="E-A-IQ",
        data_classes=["SIM-sremu"],
        description=f"E-A batch reference for the IQ subset: {kind}, tau {tau} s, 1000 s",
        assumptions=run.assumptions,
        inputs=run.inputs,
        files={"result.json": "summary"},
    )
    run.finish()
    return doc


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    export_git_info(out)
    cache = Path(args.cache).resolve()
    ea.prebuild(1e9, 2, 20, cache)
    rows = []
    for kind, tau in POINTS:
        if args.only and f"{kind}:{tau}" != args.only:
            continue
        bp = out / f"batch_{kind}_tau{tau}" / "result.json"
        if bp.exists() and (bp.parent / "manifest.json").exists():
            b = json.loads(bp.read_text(encoding="utf-8"))  # reuse a finished batch reference
        else:
            b = batch_reference(kind, tau, out, cache)
        print(json.dumps(b), flush=True)
        for noise in ("off", "on"):
            r = run_point(kind, tau, noise, out, cache)
            ref = b["p_vectorized_mid_dbm"]  # the E-A evaluator (midpoint rule, amendment 1)
            diff = r["p_iq_dbm"] - ref
            lo, hi = r["p_iq_ci_dbm"]
            row = {
                "controller": kind,
                "tau_sui_s": tau,
                "noise": noise,
                "p_vectorized_mid_dbm": ref,
                "p_vectorized_left_dbm": b["p_vectorized_left_dbm"],
                "p_run_batch_left5ms_dbm": b["p_batch_dbm"],
                "p_iq_dbm": r["p_iq_dbm"],
                "diff_db": diff,
                "diff_run_batch_db": r["p_iq_dbm"] - b["p_batch_dbm"],
                "iq_ci_low_dbm": lo,
                "iq_ci_high_dbm": hi,
                "batch_in_iq_ci": bool(lo <= ref <= hi),
                "accept": bool(abs(diff) <= 0.02) if noise == "off" else bool(lo <= ref <= hi),
                "iq_wall_s": r["wall_s"],
                "batch_wall_s": b["wall_s"],
                "late_records": r["server"]["late_records"],
                "nondeterministic_inputs": r["server"]["nondeterministic_inputs"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_csv(out / "va1.csv", rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("scenario")
    s.add_argument("--noise", choices=("on", "off"), default="off")
    s.add_argument("--summary", required=True)
    c = sub.add_parser("controller")
    c.add_argument("--endpoint", required=True)
    c.add_argument("--kind", choices=("exact", "urap"), required=True)
    c.add_argument("--tau", required=True)
    c.add_argument("--cache", required=True)
    r = sub.add_parser("rxapp")
    r.add_argument("--endpoint", required=True)
    r.add_argument("--total", required=True)
    r.add_argument("--out", required=True)
    g = sub.add_parser("run")
    g.add_argument("--out", default=str(REPO / "runs" / "wp11" / "ea_iq"))
    g.add_argument("--cache", default=str(REPO / "runs" / "wp11" / "_ea_cache"))
    g.add_argument("--only", default=None, help="kind:tau, e.g. exact:0.1")
    a = ap.parse_args(argv)
    return {"serve": cmd_serve, "controller": cmd_controller, "rxapp": cmd_rxapp, "run": cmd_run}[
        a.cmd
    ](a)


if __name__ == "__main__":
    sys.exit(main())


def aggregate(out_root: str | Path, results_dir: str | Path) -> list[dict[str, Any]]:
    """VA-1 table from the run directories under ``out_root`` (one row per IQ run)."""
    root, out = Path(out_root), Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = RunRecord(out, ea.SEED, mode="analysis", patch_root=out)
    rows = []
    for kind, tau in POINTS:
        bpath = root / f"batch_{kind}_tau{tau}" / "result.json"
        if not bpath.exists():
            continue
        b = json.loads(bpath.read_text(encoding="utf-8"))
        for noise in ("off", "on"):
            rp = root / f"{kind}_tau{tau}_noise{noise}" / "result.json"
            if not rp.exists():
                continue
            r = json.loads(rp.read_text(encoding="utf-8"))
            lo, hi = r["p_iq_ci_dbm"]
            ref = b["p_vectorized_mid_dbm"]
            diff = r["p_iq_dbm"] - ref
            rows.append(
                {
                    "controller": kind,
                    "tau_sui_s": tau,
                    "noise": noise,
                    "duration_s": r["duration_s"],
                    "p_iq_dbm": r["p_iq_dbm"],
                    "iq_ci_low_dbm": lo,
                    "iq_ci_high_dbm": hi,
                    "p_vectorized_mid_dbm": ref,
                    "p_vectorized_left_dbm": b["p_vectorized_left_dbm"],
                    "p_run_batch_left5ms_dbm": b["p_batch_dbm"],
                    "diff_iq_minus_vectorized_mid_db": diff,
                    "diff_iq_minus_run_batch_db": r["p_iq_dbm"] - b["p_batch_dbm"],
                    "accept": bool(abs(diff) <= 0.02) if noise == "off" else bool(lo <= ref <= hi),
                    "iq_wall_s": r["wall_s"],
                    "rtf": r["server"]["rtf"].get("rtf"),
                    "late_records": r["server"]["late_records"],
                    "nondeterministic_inputs": r["server"]["nondeterministic_inputs"],
                    "verify_interp_max_rel": r["server"]["verify_interp_max_rel"],
                    "barriers_commands": r["controller_stdout"],
                }
            )
    write_csv(out / "va1.csv", rows)
    write_meta(
        out,
        experiment="E-A-IQ",
        data_classes=["SIM-sremu"],
        description="VA-1/W5 IQ consistency subset: served IQ runs (external controller process in "
        "barrier mode, external RX app, 10 ms windows) vs the batch evaluators over 1000 s",
        assumptions=[
            "acceptance (noise off): |IQ - vectorized midpoint evaluator| <= 0.02 dB",
            "acceptance (noise on): vectorized value inside the IQ 95 % batch-means interval (20 batches)",
            "run_batch (left-point rule, 5 ms grid) reported for comparison",
        ],
        files={"va1.csv": "one row per (controller, tau_SUI, noise)"},
        extra={"out_root": str(root)},
    )
    rec.finish()
    return rows
