"""Run the laptop benchmark campaign in order (B1, B2, B4 lockstep, B6, B3 Windows).

Usage::

    python -m bench.run_laptop [--wait-quiet-min 120] [--only b1,b2,...] [--out runs/bench]

Each suite runs in its own process (``python -m bench.<suite>``), first waiting up to
``--wait-quiet-min`` minutes for an idle host (system busy fraction <= 20 % over 5 s;
``bench._common.LoadMonitor.wait_quiet``); if the host does not become idle the suite runs
anyway and its ``meta.json`` records the gate result and the background load. B1 also waits
(up to 15 min) before each kernel section and B2 (up to 10 min) before each point for a
busy fraction <= 30 %, and both record the background load per section or point: numba's
OpenMP kernels slow down by orders of magnitude when other processes occupy cores. The B3 run
here is the laptop paced reference, always labelled "Windows, not representative" on
Windows; paced numbers for the paper come from the AWS jobs. ``campaign.json`` (in
``--out``) lists every suite's command, exit code, wall time and run directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from bench._common import REPO, RUNS_ROOT

SUITES: dict[str, list[str]] = {
    "b1": ["bench.b1_kernels", "--section-quiet-wait-s", "900"],
    "b2": ["bench.b2_lockstep", "--budget-min", "150", "--point-quiet-wait-s", "600"],
    "b4": ["bench.b4_control", "--part", "lockstep"],
    "b6": ["bench.b6_crosstool"],
    "b3w": [
        "bench.b3_paced",
        "--links",
        "1,2,4,8,16",
        "--page-targets",
        "5e-4",
        "--reps",
        "3",
        "--warmup-s",
        "5",
        "--measure-s",
        "60",
        "--budget-min",
        "30",
        "--tag",
        "windows",
    ],
}


def main(argv: list[str] | None = None) -> int:
    """Run the campaign; returns the number of failed suites."""
    import bench

    bench.deterministic_env()
    ap = argparse.ArgumentParser(prog="bench.run_laptop", description=__doc__.splitlines()[0])
    ap.add_argument("--wait-quiet-min", type=float, default=120.0)
    ap.add_argument("--only", default=",".join(SUITES))
    ap.add_argument("--out", type=Path, default=RUNS_ROOT)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, object]] = []
    failed = 0
    for name in [s for s in a.only.split(",") if s]:
        cmd = [
            sys.executable,
            "-m",
            *SUITES[name],
            "--out",
            str(a.out),
            "--wait-quiet-min",
            f"{a.wait_quiet_min:g}",
        ]
        before = {p.name for p in a.out.iterdir()}
        t0 = time.time()
        print(f"[campaign] {name}: {' '.join(cmd)}", flush=True)
        with (a.out / f"campaign_{name}.log").open("ab") as fh:
            rc = subprocess.call(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        new = sorted({p.name for p in a.out.iterdir() if p.is_dir()} - before)
        log.append(
            {"suite": name, "cmd": cmd, "rc": rc, "wall_s": time.time() - t0, "run_dirs": new}
        )
        failed += rc != 0
        (a.out / "campaign.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        print(f"[campaign] {name}: rc={rc} dirs={new}", flush=True)
    return failed


if __name__ == "__main__":
    sys.exit(main())
