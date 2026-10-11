"""WP11 Linux job layer: job.sh, run_job.py and the CPU topology/slot planner (DESIGN 12.3-12.4).

Transport (bundle, upload, termination) belongs to the lead's runner; these tests check
what the bench side owns: the shell entry point (syntax and argument forwarding, with a
real bash only; the WSL launcher on Windows is never used), the job steps and their budget
arithmetic, pinning commands, deadline handling and the slot plan on c7i-like topologies
(fake ``/sys`` trees). No test needs Linux, root or AWS.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from bench.aws import run_job as rj

import bench
from bench import _cpus

REPO = Path(__file__).resolve().parents[2]
JOB_SH = REPO / "bench" / "aws" / "job.sh"


# =========================================================================== bash helpers


def _usable_bash() -> str | None:
    """A bash that runs scripts natively: Git Bash on Windows (never the System32 or
    WindowsApps ``bash.exe`` WSL launchers, which need a WSL distribution), ``bash`` on
    PATH elsewhere. Returns None when there is none or it does not answer a probe."""
    cands: list[str] = []
    if sys.platform == "win32":
        git = shutil.which("git")
        if git:
            root = Path(git).resolve().parents[1]
            cands += [str(root / "bin" / "bash.exe"), str(root / "usr" / "bin" / "bash.exe")]
        for env in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
            base = os.environ.get(env)
            if base:
                cands.append(str(Path(base) / "Git" / "bin" / "bash.exe"))
        found = shutil.which("bash")
        if found and not any(part in found.lower() for part in ("\\system32\\", "\\windowsapps\\")):
            cands.append(found)
    else:
        found = shutil.which("bash")
        if found:
            cands.append(found)
    for c in cands:
        if not Path(c).is_file():
            continue
        try:
            r = subprocess.run([c, "-c", "echo ok"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0 and r.stdout.strip() == "ok":
            return c
    return None


BASH = _usable_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no native bash (WSL launcher not used)")


@needs_bash
def test_user_data_shell_syntax() -> None:
    """The job entry point parses (``bash -n``); the script is fed on stdin so the test
    works with Git Bash, Linux bash and macOS bash alike."""
    assert BASH is not None
    r = subprocess.run(
        [BASH, "-n", "-s"], input=JOB_SH.read_bytes(), capture_output=True, timeout=60
    )
    assert r.returncode == 0, r.stderr.decode(errors="replace")


@needs_bash
def test_job_sh_forwards_job_and_options(tmp_path: Path) -> None:
    assert BASH is not None
    out = tmp_path / "out"
    env = {
        **os.environ,
        "SREMU_JOB_OUT": str(out),
        "SREMU_PYTHON": "echo",  # exec echo ... instead of python
        "SREMU_SKIP_HOST_PREP": "1",
    }
    r = subprocess.run(
        [BASH, JOB_SH.as_posix(), "b3b4", "--deadline-min", "200"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    last = r.stdout.strip().splitlines()[-1]
    assert last.split()[:2] == ["bench/aws/run_job.py", "b3b4"]
    assert "--deadline-min 200" in last and "--out" in last
    prep = (out / "job_host_prep.txt").read_text()
    assert "SREMU_SKIP_HOST_PREP=1" in prep and "governor=" in prep
    # no job, or no output directory: refuse
    bad = subprocess.run([BASH, JOB_SH.as_posix()], capture_output=True, timeout=60, env=env)
    assert bad.returncode != 0
    env.pop("SREMU_JOB_OUT")
    bad = subprocess.run([BASH, JOB_SH.as_posix(), "b1"], capture_output=True, timeout=60, env=env)
    assert bad.returncode != 0


def test_job_sh_has_no_transport_or_prior_version_code() -> None:
    text = JOB_SH.read_text(encoding="utf-8")
    assert "curl" not in text and "poweroff" not in text and "shutdown" not in text
    assert "v1" not in text and "legacy" not in text.lower()
    assert 'exec "$PY" bench/aws/run_job.py "$JOB" --out "$OUT"' in text


# =========================================================================== topology


def _fake_sys(root: Path, n_cores: int, nodes: int, smt: int = 2) -> Path:
    """A ``/sys/devices/system`` tree: CPU c and c + n_cores are siblings (Linux/AWS
    numbering); cores split evenly over ``nodes``."""
    per = n_cores // nodes
    for nd in range(nodes):
        cores = range(nd * per, (nd + 1) * per)
        cpus = [c + s * n_cores for s in range(smt) for c in cores]
        d = root / "node" / f"node{nd}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(_cpus.format_cpu_list(cpus) + "\n")
    for c in range(n_cores):
        sib = ",".join(str(c + s * n_cores) for s in range(smt))
        for s in range(smt):
            t = root / "cpu" / f"cpu{c + s * n_cores}" / "topology"
            t.mkdir(parents=True)
            (t / "thread_siblings_list").write_text(sib + "\n")
    (root / "cpu" / "online").write_text(f"0-{smt * n_cores - 1}\n")
    return root


def test_cpu_list_format_roundtrip() -> None:
    for cpus in ([0, 1, 2, 5, 96, 97], [3], [], list(range(48)) + list(range(96, 144))):
        text = _cpus.format_cpu_list(cpus)
        assert (bench.parse_cpu_list(text) if text else []) == sorted(cpus)
    assert _cpus.format_cpu_list([5, 1, 2, 3]) == "1-3,5"


def test_slots_on_c7i_48xlarge_topology(tmp_path: Path) -> None:
    root = _fake_sys(tmp_path / "sys", n_cores=96, nodes=2)
    assert _cpus.all_cpus(root) == list(range(192))  # the host, not this process's affinity
    assert _cpus.all_cpus(tmp_path / "none") == list(range(os.cpu_count() or 1))
    cores = _cpus.read_topology(root, cpus=_cpus.all_cpus(root))
    assert len(cores) == 96 and all(len(k.cpus) == 2 for k in cores)
    assert cores[0].cpus == (0, 96) and cores[48].node == 1 and cores[48].cpus == (48, 144)
    slots, reserved = _cpus.plan_slots(cores, 16, reserve_cores=1)
    assert reserved == [0, 96]
    assert len(slots) == 11  # node 0: 47 cores -> 5 slots; node 1: 48 cores -> 6 slots
    seen: set[int] = set()
    node_of = {c: k.node for k in cores for c in k.cpus}
    for s in slots:
        assert len(s) == 16 and not seen.intersection(s) and not set(reserved).intersection(s)
        assert len({node_of[c] for c in s}) == 1  # one NUMA node per slot
        assert all((c + 96) % 192 in s for c in s)  # whole cores (both SMT siblings)
        seen.update(s)
    assert _cpus.plan_slots(cores, 16, max_slots=2)[0] == slots[:2]
    from bench.b2_lockstep import exclusive_cpus

    assert exclusive_cpus(cores, 96) == list(range(48)) + list(range(96, 144))  # node 0
    assert len(exclusive_cpus(cores, 32)) == 32
    with pytest.raises(ValueError):
        _cpus.plan_slots(cores[:4], 16)


def test_cpu_plan_on_c7i_8xlarge_and_small_hosts(tmp_path: Path) -> None:
    root = _fake_sys(tmp_path / "sys", n_cores=16, nodes=1)
    cores = _cpus.read_topology(root, cpus=range(32))
    plan = rj.cpu_plan(16, 2, cores)
    assert plan["reserved"] == [0, 16]
    assert plan["engine"] == list(range(1, 9)) + list(range(17, 25))
    assert plan["clients"] == [9, 10, 25, 26]
    # a host without topology files: every CPU is a core; too small for 16 -> all but core 0
    flat = _cpus.read_topology(tmp_path / "none", cpus=range(8))
    small = rj.cpu_plan(16, 2, flat)
    assert small["reserved"] == [0] and small["engine"] == list(range(1, 8))
    assert small["clients"] == []


# =========================================================================== run_job


def _plan() -> dict[str, list[int]]:
    return {"engine": list(range(1, 17)), "clients": [17, 18], "reserved": [0]}


def test_run_job_steps_cover_every_job(tmp_path: Path) -> None:
    for job in rj.JOBS:
        sts = rj.steps(job, tmp_path, "py", _plan(), 16)
        assert sts, job
        for st in sts:
            assert st.argv[:2] == ["py", "-m"] and st.argv[2].startswith("bench.")
            assert st.argv[st.argv.index("--out") + 1] == str(tmp_path / "runs")
            assert "b5" not in st.argv[2] and "v1" not in " ".join(st.argv)
    b1b2 = rj.steps("b1b2", tmp_path, "py", _plan(), 16)
    assert [s.name for s in b1b2] == ["b1", "b2"]
    b2 = b1b2[1].argv
    assert b2[b2.index("--plan") + 1] == "server" and b2[b2.index("--workers") + 1] == "auto"
    assert "{budget}" in b2 and b1b2[1].cpus is None  # B2 plans the host, pins itself
    assert b1b2[0].cpus == list(range(1, 17)) and b1b2[0].env["NUMBA_NUM_THREADS"] == "16"
    b3b4 = rj.steps("b3b4", tmp_path, "py", _plan(), 16)
    assert [s.name for s in b3b4] == ["b3", "b4_lockstep", "b4_paced"]
    assert all(s.env["SREMU_CLIENT_CPUS"] == "17-18" for s in b3b4)
    with pytest.raises(SystemExit):
        rj.steps("b5", tmp_path, "py", _plan(), 16)


def test_budget_and_timeout_arithmetic() -> None:
    st = rj.Step("b1", ["x"], reserve_min=60.0, tail_min=8.0)
    assert rj.budget_and_timeout(st, False, None, 90.0) == (rj.DEFAULT_BUDGET_MIN, None)
    budget, timeout = rj.budget_and_timeout(st, False, 200 * 60.0, 90.0)
    assert budget == pytest.approx(200 - 60 - 8) and timeout == pytest.approx(140 * 60 - 90)
    last = rj.Step("b2", ["x"], reserve_min=60.0, tail_min=25.0)
    budget, timeout = rj.budget_and_timeout(last, True, 120 * 60.0, 90.0)
    assert budget == pytest.approx(95) and timeout == pytest.approx(120 * 60 - 90)
    assert rj.budget_and_timeout(last, True, 60.0, 90.0) == (5.0, 60.0)


def test_pinned_argv_uses_taskset_only_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    st = rj.Step("b3", ["py", "-m", "bench.b3_paced"], cpus=[1, 2, 3, 17, 18, 19])
    monkeypatch.setattr(rj.sys, "platform", "linux")
    monkeypatch.setattr(rj.shutil, "which", lambda name: "/usr/bin/taskset")
    assert rj.pinned_argv(st) == ["taskset", "-c", "1-3,17-19", "py", "-m", "bench.b3_paced"]
    monkeypatch.setattr(rj.shutil, "which", lambda name: None)
    assert rj.pinned_argv(st) == st.argv
    monkeypatch.setattr(rj.sys, "platform", "win32")
    monkeypatch.setattr(rj.shutil, "which", lambda name: "taskset")
    assert rj.pinned_argv(st) == st.argv
    assert rj.pinned_argv(rj.Step("x", ["a"], cpus=None)) == ["a"]


def test_run_child_exit_codes_and_deadline(tmp_path: Path) -> None:
    log = tmp_path / "c.log"
    rc, why = rj.run_child([sys.executable, "-c", "print('hi')"], log, {}, None, 5.0)
    assert rc == 0 and why is None and "hi" in log.read_text()
    rc, why = rj.run_child(
        [sys.executable, "-c", "import sys; sys.exit(3)"], log, {"X_TEST": "1"}, 30.0, 5.0
    )
    assert rc == 3 and why is None
    t0 = time.perf_counter()
    rc, why = rj.run_child(
        [sys.executable, "-c", "import time; time.sleep(60)"], log, {}, 1.0, 10.0
    )
    assert why == "deadline" and rc != 0 and time.perf_counter() - t0 < 30.0


def test_run_job_dry_lists_pinned_steps(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert rj.main(["smoke", "--out", str(tmp_path), "--dry"]) == 0
    steps = json.loads(capsys.readouterr().out)
    assert [s["name"] for s in steps] == ["b1_smoke", "b2_smoke", "b3_smoke", "b4_smoke"]
    assert all("--quick" in s["argv"] or "paced_smoke" in s["argv"] for s in steps)
    assert not (tmp_path / "status.json").exists()  # dry: nothing run, nothing written


def test_aws_dir_holds_only_the_job_layer() -> None:
    names = {p.name for p in (REPO / "bench" / "aws").iterdir() if p.name != "__pycache__"}
    assert names == {"__init__.py", "job.sh", "run_job.py"}


def test_set_process_affinity_moves_threads_started_before_the_pin() -> None:
    """``sched_setaffinity(0, ...)`` alone leaves earlier threads (e.g. the OpenBLAS pool
    started by ``import numpy``) on the old CPUs; the bench helper moves every thread."""
    import threading

    if not hasattr(os, "sched_setaffinity") or not os.path.isdir(bench.TASK_DIR):
        assert bench.thread_affinities() is None or isinstance(bench.thread_affinities(), list)
        pytest.skip("no per-thread affinity on this platform")
    orig = sorted(os.sched_getaffinity(0))
    if len(orig) < 2:
        pytest.skip("needs two CPUs")
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, daemon=True)
    t.start()
    try:
        os.sched_setaffinity(0, [orig[0]])  # the old behaviour: only this thread moves
        assert len(bench.thread_affinities() or []) >= 2
        n = bench.set_process_affinity([orig[0]])
        assert n >= 2 and bench.thread_affinities() == [str(orig[0])]
        rec = _cpus.pin_self([orig[-1]])
        assert rec["applied"] and rec["thread_affinities"] == [str(orig[-1])]
    finally:
        bench.set_process_affinity(orig)
        stop.set()
        t.join(5)
