"""sremu benchmarks B1-B4, B6 and the Linux job scripts (DESIGN.md section 12; WP11).

Each suite is a script ``bench/bN_*.py`` with a ``main(argv)`` entry point; ``sremu bench
--suite bN`` (``sremu.commands.bench``) or ``python -m bench.bN_<name>`` runs it. Every
suite writes one run directory (``runs/bench/<UTC>_<suite>[_<tag>]/``) with
``manifest.json``, CSV/JSON data, ``meta.json`` (data classes, platform label, caveats) and
``summary.json``; see :mod:`bench._common`.
"""

from __future__ import annotations

import os
import sys
import threading


def deterministic_env() -> bool:
    """Apply the deterministic-mode environment of the CLI (D-4, D15) when numba is not
    imported yet: ``NUMBA_DISABLE_INTEL_SVML=1`` (unless explicitly ``0``) and the
    ``-nosvml`` numba cache directory. Called at the start of every suite's ``main``;
    importing the package changes nothing. Returns whether the environment was applied."""
    if "numba" in sys.modules:
        return False
    os.environ["NUMBA_DISABLE_INTEL_SVML"] = os.environ.get("NUMBA_DISABLE_INTEL_SVML") or "1"
    if os.environ["NUMBA_DISABLE_INTEL_SVML"] not in ("", "0"):
        from sremu.sysinfo import nosvml_cache_dir

        os.environ["NUMBA_CACHE_DIR"] = nosvml_cache_dir(os.environ.get("NUMBA_CACHE_DIR"))
    return True


def omp_wait_policy(policy: str | None) -> str | None:
    """Set ``OMP_WAIT_POLICY`` for numba's OpenMP threading layer before numba is imported
    (the runtime reads it when it starts): ``passive`` puts idle OpenMP workers to sleep at
    once instead of letting them spin for milliseconds after every parallel region. A served
    paced engine runs few, short parallel regions (paced pages render serially below
    ``render_numba.PACED_PARALLEL_MIN_WORK``); spinning workers then compete for the engine's
    CPUs with the control and transport threads. An explicit environment setting wins, and
    nothing is set once numba is imported. Returns the policy in effect (None: runtime
    default)."""
    if policy and "OMP_WAIT_POLICY" not in os.environ and "numba" not in sys.modules:
        os.environ["OMP_WAIT_POLICY"] = policy.upper()
    return os.environ.get("OMP_WAIT_POLICY")


def set_threads(n: int | None) -> int | None:
    """Cap numba's thread count for a suite (``--threads``): before numba is imported this
    sets ``NUMBA_NUM_THREADS`` (inherited by every thread and child process); afterwards it
    calls ``numba.set_num_threads`` (thread-local: the calling thread only). ``None`` leaves
    numba's default. Returns ``n``."""
    if n is None:
        return None
    if int(n) < 1:
        raise ValueError(f"threads must be >= 1, got {n}")
    if "numba" not in sys.modules:
        os.environ["NUMBA_NUM_THREADS"] = str(int(n))
    else:
        import numba

        numba.set_num_threads(int(n))
    return int(n)


def parse_cpu_list(spec: str) -> list[int]:
    """``"0-1,4"`` -> ``[0, 1, 4]`` (Linux cpulist syntax)."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"bad CPU range {part!r}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


TASK_DIR = "/proc/self/task"


def set_process_affinity(cpus: list[int]) -> int:
    """Pin EVERY thread of this process to ``cpus`` (Linux) and return the number of threads
    moved.

    ``os.sched_setaffinity(0, ...)`` moves only the calling thread: threads that already
    exist keep their old mask (e.g. the OpenBLAS pool that ``import numpy`` starts would stay
    on the parent's CPUs). This sets the main thread and then every task listed in
    ``/proc/self/task``; threads started later inherit the mask. Returns 0 where there is
    no ``os.sched_setaffinity`` (Windows)."""
    setaff = getattr(os, "sched_setaffinity", None)
    if setaff is None:
        return 0
    setaff(0, cpus)
    n = 1
    me = threading.get_native_id()
    try:
        tids = [int(t) for t in os.listdir(TASK_DIR)]
    except (OSError, ValueError):
        return n
    for tid in tids:
        if tid == me:
            continue
        try:
            setaff(tid, cpus)
            n += 1
        except OSError:  # the thread exited meanwhile
            continue
    return n


def thread_affinities() -> list[str] | None:
    """Distinct CPU masks (``Cpus_allowed_list``) of all threads of this process, sorted
    (Linux ``/proc/self/task``); None where unavailable. One entry means every thread is
    confined to the same CPUs."""
    try:
        tids = os.listdir(TASK_DIR)
    except OSError:
        return None
    masks: set[str] = set()
    for t in tids:
        try:
            with open(f"{TASK_DIR}/{t}/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("Cpus_allowed_list:"):
                        masks.add(line.split(":", 1)[1].strip())
                        break
        except OSError:
            continue
    return sorted(masks)


def pin_client_cpus(var: str = "SREMU_CLIENT_CPUS") -> list[int] | None:
    """Move a benchmark client process (ZMQ data clients, controller processes) to the CPUs
    in ``$SREMU_CLIENT_CPUS`` (Linux only; set by ``bench/aws/run_job.py`` to cores left out
    of the engine's ``taskset`` set, and by the B2 workers to their own slot), so a busy
    client never preempts a numba worker of the engine. Every thread of the process is moved
    (:func:`set_process_affinity`). Returns the CPU list applied, or None (variable unset or
    empty, or no ``os.sched_setaffinity``)."""
    spec = os.environ.get(var, "").strip()
    if not spec or not hasattr(os, "sched_setaffinity"):
        return None
    cpus = parse_cpu_list(spec)
    set_process_affinity(cpus)
    return cpus
