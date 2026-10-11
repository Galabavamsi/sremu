#!/usr/bin/env python3
"""Run one Linux benchmark job of the sremu paper (DESIGN.md sections 12.3, 12.4; WP11).

The lead's AWS runner (``research/aws/runner.py``) owns the transport: it boots Ubuntu
24.04, unpacks the bundle (``git archive`` of the repository plus extra paths), creates the
venv, sets ``SREMU_JOB_OUT``, runs one shell command from the bundle root, uploads
``SREMU_JOB_OUT`` and powers off. A benchmark job is therefore just::

    bash bench/aws/job.sh JOB [--deadline-min M] [--dry]

which quiets the host and calls this script. Usage::

    python bench/aws/run_job.py JOB --out DIR [--deadline-min M] [--dry]

Jobs (``--deadline-min``: wall budget of the whole job; a suite still running at its
deadline gets SIGINT, writes its partial run directory and the job ends, so the runner can
still upload):

* ``b1``: B1 kernels + section 7.4 cost-model fit, pinned to one CPU slot
  (``--cpus-per-slot`` CPUs of whole cores in NUMA node 0, after core 0), the rest idle.
* ``b2``: B2 server plan, one pinned worker per point on every slot of the host
  (``bench.b2_lockstep --plan server --workers auto``); B2 plans the whole host and then
  pins its orchestrator to core 0 itself (no taskset).
* ``b1b2``: ``b1`` then ``b2`` (sized for c7i.48xlarge, 192 vCPUs).
* ``b3``: B3 paced server plan, sequential, engine pinned to one slot, client processes on
  the next ``--client-cores`` cores (sized for a quiet c7i.8xlarge).
* ``b4``: B4 lockstep part (lateness must be 0) then the paced + async part, pinned as b3.
* ``b3b4``: ``b3`` then ``b4``.
* ``eb3``: the E-B3 timing demonstration (external TDMA controller), pinned as b3.
* ``smoke``: quick versions of B1, B2 (server plan, 2 workers), B3 and B4 (minutes).

Writes into ``--out``: ``sysinfo.json`` (uname, lscpu, governor, affinity, CPU plan,
``pip freeze``), ``status.json`` (every step's command, CPUs, budget, exit code, wall time),
``<step>.log`` and the suites' run directories under ``runs/``. Standard library only (plus
:mod:`bench._cpus`, which is standard library too).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bench._cpus import Core, format_cpu_list, plan_slots, read_topology  # noqa: E402

JOBS = ("b1", "b2", "b1b2", "b3", "b4", "b3b4", "eb3", "smoke")
SIGINT_GRACE_S = 90.0
DEFAULT_BUDGET_MIN = 150.0


@dataclass
class Step:
    """One suite invocation: argv (without pinning; ``{budget}`` is replaced by the wall
    budget in minutes at launch), the CPUs to pin it to and extra environment."""

    name: str
    argv: list[str]
    cpus: list[int] | None = None
    env: dict[str, str] = field(default_factory=dict)
    reserve_min: float = 0.0
    """Minutes this step must leave for the steps after it."""
    tail_min: float = 8.0
    """Minutes the suite may run past its ``{budget}`` (in-flight points, final phases)."""


def cpu_plan(
    cpus_per_slot: int, client_cores: int, cores: list[Core] | None = None
) -> dict[str, list[int]]:
    """Engine slot (whole cores of node 0 after core 0), client CPUs (the next
    ``client_cores`` free cores of that node) and the reserved OS CPUs. On a host too small
    for ``cpus_per_slot`` the engine gets every non-reserved core of node 0."""
    cores = read_topology() if cores is None else cores
    try:
        slots, reserved = plan_slots(cores, cpus_per_slot, reserve_cores=1, max_slots=1)
        engine = slots[0]
    except ValueError:
        node0 = [k for k in cores if k.node == cores[0].node]
        reserved = list(node0[0].cpus) if len(node0) > 1 else []
        engine = sorted(c for k in node0[1 if reserved else 0 :] for c in k.cpus)
    used = set(engine) | set(reserved)
    node = next(k.node for k in cores if engine[0] in k.cpus)
    free = [k for k in cores if k.node == node and not used.intersection(k.cpus)]
    clients = sorted(c for k in free[:client_cores] for c in k.cpus)
    return {"engine": engine, "clients": clients, "reserved": sorted(reserved)}


def steps(
    job: str, out: Path, py: str, plan: dict[str, list[int]], cpus_per_slot: int
) -> list[Step]:
    """The steps of a job."""
    runs = str(out / "runs")
    eng, cli = plan["engine"], plan["clients"]
    env = {
        "NUMBA_NUM_THREADS": str(len(eng)),
        "SREMU_CPUS": format_cpu_list(eng),
        "SREMU_CLIENT_CPUS": format_cpu_list(cli),
    }
    m = [py, "-m"]
    b1 = Step("b1", [*m, "bench.b1_kernels", "--tag", f"slot{len(eng)}", "--out", runs], eng, env)
    b2 = Step(
        "b2",
        [
            *m,
            "bench.b2_lockstep",
            "--plan",
            "server",
            "--workers",
            "auto",
            "--cpus-per-point",
            str(cpus_per_slot),
            "--budget-min",
            "{budget}",
            "--out",
            runs,
        ],
        None,  # not tasksetted: B2 plans the whole host, then pins itself to core 0
        {},
        tail_min=25.0,
    )
    b3 = Step(
        "b3",
        [*m, "bench.b3_paced", "--plan", "server", "--budget-min", "{budget}", "--out", runs],
        eng,
        env,
    )
    b4l = Step(
        "b4_lockstep", [*m, "bench.b4_control", "--part", "lockstep", "--out", runs], eng, env
    )
    b4p = Step(
        "b4_paced",
        [*m, "bench.b4_control", "--part", "paced", "--budget-min", "{budget}", "--out", runs],
        eng,
        env,
        tail_min=5.0,
    )
    if job == "eb3":
        eb3 = [*m, "bench.eb3_paced", "--budget-min", "{budget}", "--out", runs]
        return [Step("eb3", eb3, eng, env)]
    if job == "b1":
        return [b1]
    if job == "b2":
        return [b2]
    if job == "b1b2":
        b1.reserve_min = 60.0
        return [b1, b2]
    if job == "b3":
        return [b3]
    # B4 paced: 24 repetitions of 65 s virtual; while a configuration command costs
    # milliseconds of engine time, a repetition at >= 100 commands/s runs about 2.4x its
    # virtual length (measured), so the paced part needs about an hour
    if job == "b4":
        b4l.reserve_min = 60.0
        return [b4l, b4p]
    if job == "b3b4":
        b3.reserve_min = 70.0
        b4l.reserve_min = 60.0
        return [b3, b4l, b4p]
    if job == "smoke":
        return [
            Step("b1_smoke", [*b1.argv, "--quick", "--skip-engine"], eng, env, 12.0),
            Step(
                "b2_smoke",
                [
                    *m,
                    "bench.b2_lockstep",
                    "--plan",
                    "server",
                    "--quick",
                    "--workers",
                    "2",
                    "--cpus-per-point",
                    "4",
                    "--isolation-check",
                    "1",
                    "--out",
                    runs,
                ],
                None,
                {},
                8.0,
            ),
            Step("b3_smoke", [*m, "bench.b3_paced", "--quick", "--out", runs], eng, env, 4.0),
            Step(
                "b4_smoke",
                [
                    *m,
                    "bench.b4_control",
                    "--part",
                    "paced",
                    "--rates",
                    "100",
                    "--modes",
                    "timed",
                    "--reps",
                    "1",
                    "--warmup-s",
                    "2",
                    "--measure-s",
                    "5",
                    "--tag",
                    "paced_smoke",
                    "--out",
                    runs,
                ],
                eng,
                env,
            ),
        ]
    raise SystemExit(f"unknown job {job!r}; one of {', '.join(JOBS)}")


def pinned_argv(step: Step) -> list[str]:
    """``taskset -c CPUS argv`` on Linux when taskset exists, else argv unchanged."""
    if step.cpus and sys.platform.startswith("linux") and shutil.which("taskset"):
        return ["taskset", "-c", format_cpu_list(step.cpus), *step.argv]
    return list(step.argv)


def budget_and_timeout(
    step: Step, is_last: bool, left_s: float | None, grace_s: float
) -> tuple[float, float | None]:
    """``(budget_min, timeout_s)`` of a step launched with ``left_s`` seconds before the
    job deadline (None: no deadline)."""
    if left_s is None:
        return DEFAULT_BUDGET_MIN, None
    reserve_s = 0.0 if is_last else step.reserve_min * 60.0
    budget = max(5.0, (left_s - reserve_s) / 60.0 - step.tail_min)
    timeout = max(60.0, left_s - reserve_s - grace_s)
    return budget, timeout


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def sysinfo(py: str, plan: dict[str, list[int]]) -> dict[str, object]:
    """Host facts for the result directory."""
    info: dict[str, object] = {
        "uname": platform.uname()._asdict(),
        "python_dispatch": sys.version,
        "nproc": os.cpu_count(),
        "affinity": format_cpu_list(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "governor_cpu0": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        or "unavailable (no cpufreq exposed)",
        "dmi_product": _read("/sys/devices/virtual/dmi/id/product_name"),
        "dmi_asset_tag": _read("/sys/devices/virtual/dmi/id/board_asset_tag"),
        "loadavg": _read("/proc/loadavg"),
        "cpu_plan": {k: format_cpu_list(v) for k, v in plan.items()},
        "env": {
            k: os.environ.get(k)
            for k in (
                "NUMBA_DISABLE_INTEL_SVML",
                "SREMU_GOVERNOR",
                "SREMU_JOB_OUT",
                "SREMU_GIT_COMMIT",
                "SREMU_GIT_DIRTY",
            )
        },
    }
    for key, cmd in (("lscpu", ["lscpu"]), ("numactl", ["numactl", "--hardware"])):
        try:
            info[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            info[key] = None
    try:
        info["pip_freeze"] = subprocess.run(
            [py, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        info["pip_freeze"] = None
    return info


def run_child(
    cmd: list[str], log: Path, env: dict[str, str], timeout_s: float | None, grace_s: float
) -> tuple[int, str | None]:
    """Run one suite (own process group) with output appended to ``log``. At its timeout
    (``"deadline"``) or on our own SIGINT/SIGTERM (``"signal"``) send SIGINT to the group
    (the suite writes its partial run directory), wait ``grace_s``, then kill. Returns
    ``(rc, interrupted_by)``."""
    kw: dict[str, Any] = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    with log.open("ab") as fh:
        p = subprocess.Popen(
            cmd, cwd=SRC, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, **env}, **kw
        )
        why = None
        try:
            return p.wait(timeout=timeout_s), None
        except subprocess.TimeoutExpired:
            why = "deadline"
        except KeyboardInterrupt:
            why = "signal"
        _interrupt(p)
        try:
            return p.wait(timeout=grace_s), why
        except subprocess.TimeoutExpired:
            p.kill()
            return p.wait(), why


def _interrupt(p: subprocess.Popen[bytes]) -> None:
    try:
        if sys.platform == "win32":
            p.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(p.pid, signal.SIGINT)
    except OSError:
        pass


def _sigterm(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("job", choices=JOBS)
    ap.add_argument("--out", type=Path, default=os.environ.get("SREMU_JOB_OUT"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--deadline-min", type=float, default=None, help="wall budget of the job")
    ap.add_argument("--cpus-per-slot", type=int, default=16, help="engine / per-point CPUs")
    ap.add_argument("--client-cores", type=int, default=2, help="cores for client processes")
    ap.add_argument("--grace-s", type=float, default=SIGINT_GRACE_S)
    ap.add_argument("--dry", action="store_true", help="print the steps only")
    a = ap.parse_args(argv)
    if a.out is None:
        ap.error("--out is required (or set SREMU_JOB_OUT)")
    plan = cpu_plan(a.cpus_per_slot, a.client_cores)
    todo = steps(a.job, Path(a.out), a.python, plan, a.cpus_per_slot)
    if a.dry:
        print(json.dumps([{**asdict(s), "pinned_argv": pinned_argv(s)} for s in todo], indent=1))
        return 0
    signal.signal(signal.SIGTERM, _sigterm)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sysinfo.json").write_text(json.dumps(sysinfo(a.python, plan), indent=1), "utf-8")
    t0 = time.time()
    deadline = None if a.deadline_min is None else t0 + a.deadline_min * 60.0
    results: list[dict[str, object]] = []
    status: dict[str, object] = {
        "job": a.job,
        "deadline_min": a.deadline_min,
        "cpu_plan": {k: format_cpu_list(v) for k, v in plan.items()},
        "results": results,
    }
    rc_all = 0
    for i, st in enumerate(todo):
        left_s = None if deadline is None else deadline - time.time()
        if left_s is not None and left_s < 120.0:
            results.append({"step": st.name, "skipped": "deadline reached"})
            rc_all = rc_all or 124
            continue
        budget, timeout = budget_and_timeout(st, i + 1 == len(todo), left_s, a.grace_s)
        argv = [x.replace("{budget}", f"{budget:.0f}") for x in pinned_argv(st)]
        log = out / f"{st.name}.log"
        print(f"[run_job] {st.name}: {' '.join(argv)} (timeout {timeout})", flush=True)
        t = time.time()
        rc, why = run_child(argv, log, st.env, timeout, a.grace_s)
        results.append(
            {
                "step": st.name,
                "cmd": argv,
                "cpus": None if st.cpus is None else format_cpu_list(st.cpus),
                "env": st.env,
                "budget_min": budget if "{budget}" in st.argv else None,
                "timeout_s": timeout,
                "rc": rc,
                "wall_s": time.time() - t,
                "log": log.name,
                "interrupted_by": why,
            }
        )
        rc_all = rc_all or rc
        (out / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
        if why == "signal":
            break
    status["wall_s"] = time.time() - t0
    status["rc"] = rc_all
    (out / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    print(f"[run_job] done rc={rc_all} wall={status['wall_s']:.0f}s", flush=True)
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
