"""Task runner (DESIGN.md section 2.6; no GNU make on Windows).

Usage::

    python tools/tasks.py ci [pytest args...]          # CI tier, <= 8 min
    python tools/tasks.py nightly [pytest args...]     # statistical/integration/validation_smoke
    python tools/tasks.py paper-junit PATH.xml [pytest args...]  # CI + nightly -> junit + meta
    python tools/tasks.py goldens [--regenerate]       # MATLAB goldens (WP4) + Python side
    python tools/tasks.py bench [sremu bench args...]  # benchmarks (WP11)
    python tools/tasks.py paper-data [args...]         # release tier via the paper pipeline (WP12)
    python tools/tasks.py lint                         # ruff check, ruff format --check, mypy

Every task runs with the interpreter that runs this script and sets
``NUMBA_DISABLE_INTEL_SVML=1`` (deterministic mode, D-4) unless already set. A task whose
implementation does not exist yet exits with code 3. pytest tasks use ``-n 8`` xdist workers
when pytest-xdist is installed (D21: a 16-worker run crashed) and pytest's built-in
``faulthandler_timeout`` (``HANG_DUMP_S``): a test that hangs dumps every thread's stack.

``paper-junit`` (D21) runs the CI and nightly markers with ``--junitxml=PATH`` and writes the
sidecar ``<stem>.meta.json`` next to it: ``{"sremu_commit": <git rev-parse HEAD>,
"sremu_dirty": <git status --porcelain is non-empty>}``, both read before the tests run.
These two read-only git calls are the only git use of this script.
"""

from __future__ import annotations

import argparse
import json
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
# Record checks of regenerated results that the paper quotes carry the validation_smoke marker
# (tests/validation, e.g. test_va1_record.py: VA-1 read from the paper repository at
# $SREMU_PAPER_DIR or ../paper, skipped when the record is absent), so the paper's junit
# carries their spec ids. validation_smoke is part of NIGHTLY_MARKERS and so of PAPER_MARKERS.
PAPER_RECORD_MARKERS = "validation_smoke"
PAPER_MARKERS = f"({CI_MARKERS}) or ({NIGHTLY_MARKERS})"  # D21: what the paper's junit pins
assert PAPER_RECORD_MARKERS in NIGHTLY_MARKERS  # the VA-1 record check stays in the paper junit
BUDGET_S = {"ci": 8 * 60, "nightly": 30 * 60}
XDIST_WORKERS = 8  # D21
# pytest's built-in faulthandler_timeout: a test running longer than this dumps the stack of
# every thread to stderr (CI log) and keeps running, so a hang is diagnosable instead of the
# CI job being cancelled without a trace (GitHub Actions run 38066711708, windows-latest).
# No single test of either tier should come near it.
HANG_DUMP_S = 300


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
    return ["-n", str(XDIST_WORKERS)]


def _hang_dump_args(extra: list[str]) -> list[str]:
    """``-o faulthandler_timeout=HANG_DUMP_S`` unless the caller sets it."""
    if any("faulthandler_timeout" in a for a in extra):
        return []
    return ["-o", f"faulthandler_timeout={HANG_DUMP_S}"]


def _pytest(markers: str, extra: list[str], budget_s: float | None = None) -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        markers,
        *_xdist_args(),
        *_hang_dump_args(extra),
        *extra,
    ]
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


def _git_state(repo: Path = REPO) -> tuple[str, bool]:
    """``(HEAD commit, dirty)`` of ``repo`` from two read-only git calls (paper-junit only)."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git not found on PATH")

    def call(*args: str) -> str:
        res = subprocess.run(
            [git, "-C", str(repo), *args], capture_output=True, text=True, check=False
        )
        if res.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
        return res.stdout

    commit = call("rev-parse", "HEAD").strip()
    dirty = bool(call("status", "--porcelain").strip())
    return commit, dirty


def meta_path(junit: Path) -> Path:
    """Sidecar of a junit file: ``<stem>.meta.json`` in the same directory."""
    return junit.with_name(f"{junit.stem}.meta.json")


def task_paper_junit(extra: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="tasks.py paper-junit",
        description="CI + nightly tiers with --junitxml and a {sremu_commit, sremu_dirty} sidecar.",
    )
    parser.add_argument("junit", type=Path, help="junit XML file to write")
    args, rest = parser.parse_known_args(extra)
    junit = args.junit if args.junit.is_absolute() else Path.cwd() / args.junit
    try:
        commit, dirty = _git_state(REPO)
    except RuntimeError as exc:
        print(f"[tasks] paper-junit: cannot read the sremu commit: {exc}", file=sys.stderr)
        return 1
    junit.parent.mkdir(parents=True, exist_ok=True)
    rc = _pytest(
        PAPER_MARKERS,
        [f"--junitxml={junit}", *rest],
        BUDGET_S["ci"] + BUDGET_S["nightly"],
    )
    meta = {"sremu_commit": commit, "sremu_dirty": dirty}
    meta_path(junit).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[tasks] wrote {junit} and {meta_path(junit)} (commit {commit}, dirty {dirty})")
    return rc


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
    "paper-junit": task_paper_junit,
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
