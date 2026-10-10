"""Task runner (DESIGN.md section 2.6; no GNU make on Windows).

Usage::

    python tools/tasks.py ci [pytest args...]          # CI tier, <= 8 min
    python tools/tasks.py nightly [pytest args...]     # statistical/integration/validation_smoke
    python tools/tasks.py goldens [--regenerate]       # MATLAB goldens (WP4) + Python side
    python tools/tasks.py bench [sremu bench args...]  # benchmarks (WP11)
    python tools/tasks.py paper-data [args...]         # release tier via the paper pipeline (WP12)
    python tools/tasks.py lint                         # ruff check, ruff format --check, mypy

Every task runs with the interpreter that runs this script and sets
``NUMBA_DISABLE_INTEL_SVML=1`` (deterministic mode, D-4) unless already set. A task whose
implementation does not exist yet exits with code 3.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOT_IMPLEMENTED = 3
PYTEST_NO_TESTS = 5  # pytest exit code when nothing is collected
CI_MARKERS = "unit or physics or runtime or io or golden"
NIGHTLY_MARKERS = "statistical or integration or validation_smoke"  # DESIGN.md 2.6
BUDGET_S = {"ci": 8 * 60, "nightly": 30 * 60}


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")
    return env


def _run(cmd: list[str], budget_s: float | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    t0 = time.monotonic()
    rc = subprocess.call(cmd, cwd=REPO, env=_env())
    elapsed = time.monotonic() - t0
    print(f"[tasks] exit {rc} after {elapsed:.1f} s", flush=True)
    if budget_s is not None and elapsed > budget_s:
        print(f"[tasks] WARNING: over the {budget_s / 60:.0f} min budget", flush=True)
    return rc


def _xdist_args() -> list[str]:
    try:
        import xdist  # noqa: F401
    except ImportError:
        return []
    return ["-n", "auto"]


def _pytest(markers: str, extra: list[str], budget_s: float | None = None) -> int:
    cmd = [sys.executable, "-m", "pytest", "-m", markers, *_xdist_args(), *extra]
    rc = _run(cmd, budget_s)
    if rc == PYTEST_NO_TESTS:
        print(f"[tasks] no tests selected by -m {markers!r} yet; treated as success", flush=True)
        return 0
    return rc


def _not_implemented(task: str, owner: str, what: str) -> int:
    print(f"[tasks] {task}: not yet implemented (owner {owner}; expected {what})", file=sys.stderr)
    return NOT_IMPLEMENTED


def task_ci(extra: list[str]) -> int:
    return _pytest(CI_MARKERS, extra, BUDGET_S["ci"])


def task_nightly(extra: list[str]) -> int:
    return _pytest(NIGHTLY_MARKERS, extra, BUDGET_S["nightly"])


def task_goldens(extra: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tasks.py goldens")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run the MATLAB R2026a scripts before the Python-side checks.",
    )
    args, rest = parser.parse_known_args(extra)
    if args.regenerate:
        script = REPO / "tools" / "matlab" / "run_goldens.ps1"
        if not script.exists():
            return _not_implemented("goldens", "WP4", "tools/matlab/run_goldens.ps1")
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if shell is None:
            print("[tasks] goldens: PowerShell not found", file=sys.stderr)
            return 1
        rc = _run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)])
        if rc:
            return rc
    return _pytest("golden", rest)


def task_bench(extra: list[str]) -> int:
    return _run([sys.executable, "-m", "sremu.cli", "bench", *extra])


def task_paper_data(extra: list[str]) -> int:
    paper = Path(os.environ.get("SREMU_PAPER_DIR", REPO.parent / "paper"))
    script = paper / "tools" / "build_paper.py"
    if not script.exists():
        return _not_implemented("paper-data", "WP12", f"{script}")
    os.environ.setdefault("PROFILE", "fast")
    return _run([sys.executable, str(script), *extra])


def task_lint(extra: list[str]) -> int:
    rc = _run([sys.executable, "-m", "ruff", "check", "src", "tests", "tools"])
    rc |= _run([sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "tools"])
    rc |= _run([sys.executable, "-m", "mypy", *extra])
    return rc


TASKS = {
    "ci": task_ci,
    "nightly": task_nightly,
    "goldens": task_goldens,
    "bench": task_bench,
    "paper-data": task_paper_data,
    "lint": task_lint,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in TASKS:
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    task, extra = argv[0], argv[1:]
    if extra[:1] == ["--"]:
        extra = extra[1:]
    return TASKS[task](extra)


if __name__ == "__main__":
    sys.exit(main())
