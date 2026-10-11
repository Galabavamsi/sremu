"""Stock GNU Radio through sremu's raw REQ/REP mode: link quality vs RIS configuration.

    python examples/gnuradio_zmq/run_ris_ber.py --out RUNS [--configs all] [--switching]
        [--gr-python /usr/bin/python3] [--duration-s 0.3] [--rate 1e6] [--tag r1]

Linux / WSL2 only (GNU Radio from Ubuntu 24.04's ``gnuradio`` package; see README.md). This
script runs in sremu's interpreter (numpy >= 2.2); GNU Radio runs in its own (``--gr-python``
or ``$SREMU_GR_PYTHON``, default ``/usr/bin/python3`` with numpy 1.26), as a subprocess.

Every run is fresh: one ``sremu.commands.serve`` process (lockstep, paused, ``--exit-when-done``,
``--out``) with the modem as one raw peer in ``gnuradio_count`` format, one new
``gr_dqpsk_modem.py`` process, and a control client (this script) that schedules the RIS
configuration over the control socket before it sends ``sim.resume``:

* static runs (``--configs``): one RIS configuration from t = 0 for ``--duration-s``
  (``ris_configs.STATIC``: focusing, focusing with a fraction of cells inverted, all cells
  in state 0, random);
* the switching run (``--switching``): ``uniform`` on [0, T), ``focus`` on [T, 2T), ``random``
  on [2T, 3T), ``focus`` again from 3T (T = ``--segment-s``), each switch scheduled at an exact
  RIS time before virtual time starts.

The modem reads the whole run (``--exit-when-done`` needs the raw peer to read to the end).
Each run directory holds the scene, the logs, GNU Radio's files (``gr_*``), the emulator's run
directory (``rec/``: manifest, events with every applied command, metrics, the RX stream as
SigMF), ``run.json`` (what this script did) and ``analysis.json`` (``analyze_ris_ber.py``).
Virtual time is driven by the sample exchange (lockstep): wall-clock speed is reported, not
required. Exit code 0 when every run completed with every counter zero.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analyze_ris_ber import analyze_run, summary_line  # noqa: E402
from ris_configs import STATIC, Configs  # noqa: E402

SCENE = HERE / "gr_ris_ber.yaml"
MODEM = HERE / "gr_dqpsk_modem.py"
TX_STREAM, RX_STREAM = "gr.tx0", "grrx.rx0"
BASE_RATE = 1e6
SWITCHING = ("uniform", "focus", "random", "focus")


def free_port() -> int:
    """An unused loopback TCP port (bound and released; fine for one host and one user)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def scene_for(rate: float, duration_s: str, record_iq: bool) -> dict[str, Any]:
    """``gr_ris_ber.yaml`` at ``rate``: TX full scale + 10 log10(rate / 1 MS/s) and occupied
    bandwidth 0.34 x rate, so the SNR per symbol does not change with the rate."""
    d = yaml.safe_load(SCENE.read_text(encoding="utf-8"))
    scale = rate / BASE_RATE
    for nd in d["nodes"]:
        for st in nd["streams"]:
            st["rate_sps"] = int(rate)  # an exact integer (a YAML float is rejected)
            if st["direction"] == "tx":
                fs = float(st["full_scale_dbm"]) + 10 * math.log10(scale)
                st["full_scale_dbm"] = round(fs, 6)
                st["occupied_bw_hz"] = 0.34 * rate
    d["run"]["duration_s"] = duration_s
    d["record"]["iq"] = bool(record_iq)
    return d


def _gr_env() -> dict[str, str]:
    """Environment of the GNU Radio process: without the sremu virtualenv's paths."""
    env = {
        k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")
    }
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _wait_line(proc: subprocess.Popen[str], log: Path, tag: str, timeout_s: float) -> str:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        for ln in text.splitlines():
            if ln.startswith(tag):
                return ln
        if proc.poll() is not None:
            raise RuntimeError(f"the emulator exited ({proc.returncode}):\n{text}")
        time.sleep(0.05)
    raise RuntimeError(f"no {tag} line after {timeout_s} s")


def _config_message(states: np.ndarray, at: str) -> dict[str, Any]:
    return {
        "op": "ris.set_config",
        "panel": "ris0",
        "at": {"t": at},
        "config": {
            "states_b64": base64.b64encode(states.astype("<i2").tobytes()).decode(),
            "dtype": "i2",
        },
    }


def one_run(
    run_dir: Path,
    *,
    kind: str,
    configs: list[str],
    duration_s: str,
    rate: float = BASE_RATE,
    segment_s: str | None = None,
    gr_python: str = "/usr/bin/python3",
    request_items: int | None = None,
    record_iq: bool = True,
    timeout_s: float = 900.0,
    analyze: bool = True,
) -> dict[str, Any]:
    """One fresh served run with one fresh GNU Radio modem (module docstring)."""
    from sremu.transport.client import ClientSession

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    scene = scene_for(rate, duration_s, record_iq)
    scene_path = run_dir / "scene.yaml"
    scene_path.write_text(yaml.safe_dump(scene, sort_keys=False), encoding="utf-8", newline="\n")
    cfgs = Configs(scene)
    n_run = int(Fraction(duration_s) * Fraction(rate).limit_denominator(1000))
    ports = {k: free_port() for k in ("control", "tx", "gr_tx", "raw_rx")}
    peer = (
        f"name=gr,format=gnuradio_count,tx_port=tcp://127.0.0.1:{ports['gr_tx']},"
        f"rx_port=tcp://127.0.0.1:{ports['raw_rx']},tx_stream={TX_STREAM},"
        f"rx_stream={RX_STREAM}"
    )
    if request_items:
        peer += f",request_items={int(request_items)}"
    serve_cmd = [
        sys.executable,
        "-m",
        "sremu.commands.serve",
        str(scene_path),
        "--serve",
        f"tcp://127.0.0.1:{ports['control']}",
        "--pub",
        "none",
        "--tx",
        f"tcp://127.0.0.1:{ports['tx']}",
        "--out",
        str(run_dir / "rec"),
        "--exit-when-done",
        "--raw-peer",
        peer,
    ]
    gr_cmd = [
        gr_python,
        str(MODEM),
        "--tx",
        f"tcp://127.0.0.1:{ports['gr_tx']}",
        "--rx",
        f"tcp://127.0.0.1:{ports['raw_rx']}",
        "--n",
        str(n_run),
        "--out",
        str(run_dir / "gr"),
        "--timeout",
        str(timeout_s),
    ]
    meta: dict[str, Any] = {
        "kind": kind,
        "configs": configs,
        "duration_s": duration_s,
        "segment_s": segment_s,
        "rate_sps": rate,
        "n_run": n_run,
        "tx_stream": TX_STREAM,
        "rx_stream": RX_STREAM,
        "serve_cmd": serve_cmd,
        "gr_cmd": gr_cmd,
        "request_items": request_items,
        "af_pred_db": {c: cfgs.array_factor_db(c) for c in set(configs)},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "t_start_unix": time.time(),
    }
    t0 = time.monotonic()
    with (
        (run_dir / "serve.log").open("w", encoding="utf-8") as slog,
        (run_dir / "gr.log").open("w", encoding="utf-8") as glog,
    ):
        srv = subprocess.Popen(serve_cmd, stdout=slog, stderr=subprocess.STDOUT, text=True)
        gr_proc: subprocess.Popen[str] | None = None
        try:
            _wait_line(srv, run_dir / "serve.log", "SREMU-SERVE", 120.0)
            gr_proc = subprocess.Popen(
                gr_cmd,
                stdout=glog,
                stderr=subprocess.STDOUT,
                text=True,
                env=_gr_env(),
                cwd=str(run_dir),
            )
            s = ClientSession(f"tcp://127.0.0.1:{ports['control']}", name="ris-schedule")
            try:
                caps = s.call("caps.get", panel="ris0")["caps"]["ris0"]
                n_ctl = int(caps.get("n_controls") or caps["n_elements"])
                assert n_ctl == cfgs.n_cells, "the scene's panel differs from ris_configs"
                plan = (
                    [("0", configs[0])]
                    if kind == "static"
                    else [
                        (str(Decimal(segment_s or "0") * i), c)
                        for i, c in enumerate(configs)
                        if i > 0
                    ]
                )
                if kind == "switching" and configs[0] != "uniform":
                    plan.insert(0, ("0", configs[0]))
                sent = []
                for at, name in plan:
                    r = s.request(_config_message(cfgs.states(name), at))
                    sent.append({"config": name, "at": at, "reply": r})
                    if not r.get("ok"):
                        raise RuntimeError(f"ris.set_config refused: {r}")
                meta["commands_sent"] = sent
                s.call("sim.resume")
                meta["t_resume_wall_s"] = time.monotonic() - t0
            finally:
                s.close()
            gr_rc = gr_proc.wait(timeout=timeout_s + 60)
            meta["gr_exit"] = gr_rc
            meta["t_gr_done_wall_s"] = time.monotonic() - t0
            srv_rc = srv.wait(timeout=120)
            meta["serve_exit"] = srv_rc
        finally:
            for p in (gr_proc, srv):
                if p is not None and p.poll() is None:
                    p.kill()
                    p.wait()
                    meta.setdefault("killed", []).append("serve" if p is srv else "gr")
    meta["wall_s"] = time.monotonic() - t0
    (run_dir / "run.json").write_text(
        json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    if not analyze:
        return {"run": run_dir.name, "meta": meta}
    a = analyze_run(run_dir)
    a["ok"] = bool(
        a["complete"]
        and not a.get("counters_nonzero")
        and meta.get("gr_exit") == 0
        and meta.get("serve_exit") == 0
        and "killed" not in meta
        and a.get("rx_equals_emulator_record", True)
        and all(c["status"] == "applied" and c["lateness_samples"] == 0 for c in a["commands"])
    )
    (run_dir / "analysis.json").write_text(
        json.dumps(a, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return a


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="root of the run directories")
    ap.add_argument(
        "--configs",
        default="all",
        help="static configurations: 'all', 'none' or a comma list (focus, "
        "focusq:0.35, uniform, random)",
    )
    ap.add_argument("--switching", action="store_true", help="also the in-run switching run")
    ap.add_argument("--duration-s", default="0.3", help="static run length (exact string)")
    ap.add_argument("--segment-s", default="0.4", help="switching run: time per configuration")
    ap.add_argument("--rate", type=float, default=BASE_RATE)
    ap.add_argument("--request-items", type=int, default=None)
    ap.add_argument("--gr-python", default=os.environ.get("SREMU_GR_PYTHON", "/usr/bin/python3"))
    ap.add_argument("--tag", default="", help="suffix of the run directory names")
    ap.add_argument(
        "--no-iq-record",
        action="store_true",
        help="do not record the RX stream in the emulator (throughput runs)",
    )
    a = ap.parse_args(argv)
    if platform.system() != "Linux":
        print("run_ris_ber.py needs Linux or WSL2 with GNU Radio (see README.md)", file=sys.stderr)
        return 3
    names = (
        list(STATIC)
        if a.configs == "all"
        else ([] if a.configs == "none" else [c.strip() for c in a.configs.split(",") if c.strip()])
    )
    sfx = f"_{a.tag}" if a.tag else ""
    jobs: list[tuple[str, dict[str, Any]]] = []
    for c in names:
        jobs.append(
            (
                f"static_{c.replace(':', '')}{sfx}",
                {"kind": "static", "configs": [c], "duration_s": a.duration_s},
            )
        )
    if a.switching:
        dur = str(Decimal(a.segment_s) * len(SWITCHING) + Decimal("0.05"))
        jobs.append(
            (
                f"switching{sfx}",
                {
                    "kind": "switching",
                    "configs": list(SWITCHING),
                    "duration_s": dur,
                    "segment_s": a.segment_s,
                },
            )
        )
    ok = True
    for name, kw in jobs:
        res = one_run(
            a.out / name,
            rate=a.rate,
            gr_python=a.gr_python,
            request_items=a.request_items,
            record_iq=not a.no_iq_record,
            **kw,
        )
        ok &= bool(res["ok"])
        print(("OK   " if res["ok"] else "FAIL ") + summary_line(res), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
