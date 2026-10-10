"""CI hardening: task runner hang dumps and the GitHub Actions triggers (DESIGN.md 2.6).

* ``tools/tasks.py`` arms pytest's ``faulthandler_timeout`` for the pytest tiers, so a hung
  test dumps every thread's stack instead of the CI job being cancelled without a trace;
  a caller's own setting wins.
* ``ci.yml``: a push or pull request runs ubuntu only; Windows runs on a manual dispatch
  (private-repo minutes). ``nightly.yml`` has no schedule while the repository is private.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"


def _load_tasks() -> Any:
    spec = importlib.util.spec_from_file_location("sremu_tasks_ci", REPO / "tools" / "tasks.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _workflow(name: str) -> dict[str, Any]:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    # YAML 1.1 reads the bare key 'on' as True
    if True in data:
        data["on"] = data.pop(True)
    return data


@pytest.mark.parametrize("task", ["ci", "nightly"])
def test_pytest_tiers_arm_the_faulthandler_hang_dump(task: str, monkeypatch) -> None:
    tasks = _load_tasks()
    calls: list[list[str]] = []
    monkeypatch.setattr(tasks, "_run", lambda cmd, budget_s=None: calls.append(cmd) or 0)
    assert tasks.main([task, "-x"]) == 0
    (cmd,) = calls
    i = cmd.index("-o")
    assert cmd[i + 1] == f"faulthandler_timeout={tasks.HANG_DUMP_S}"
    assert 0 < tasks.HANG_DUMP_S <= tasks.BUDGET_S[task]
    assert cmd[-1] == "-x"


def test_callers_faulthandler_timeout_wins(monkeypatch) -> None:
    tasks = _load_tasks()
    calls: list[list[str]] = []
    monkeypatch.setattr(tasks, "_run", lambda cmd, budget_s=None: calls.append(cmd) or 0)
    assert tasks.main(["ci", "-o", "faulthandler_timeout=5"]) == 0
    (cmd,) = calls
    assert [a for a in cmd if "faulthandler_timeout" in a] == ["faulthandler_timeout=5"]


def test_ci_push_runs_ubuntu_only_and_windows_on_dispatch() -> None:
    wf = _workflow("ci.yml")
    on = wf["on"]
    assert set(on) == {"push", "pull_request", "workflow_dispatch"}
    choice = on["workflow_dispatch"]["inputs"]["os"]
    assert set(choice["options"]) == {"windows-latest", "ubuntu-latest", "both"}
    assert choice["default"] == "windows-latest"
    job = wf["jobs"]["ci"]
    expr = job["strategy"]["matrix"]["os"]
    # push / pull_request: ubuntu only; the windows runner appears only for a dispatch
    assert "github.event_name != 'workflow_dispatch' && '[\"ubuntu-latest\"]'" in expr
    assert "windows" not in expr.split("||")[0]
    assert "windows-latest" in str(job["timeout-minutes"])
    steps = [s.get("run", "") for s in job["steps"]]
    assert any(s.startswith("python tools/tasks.py ci") for s in steps)
    assert any(s.startswith("python tools/tasks.py lint") for s in steps)


def test_nightly_schedule_is_off_while_private() -> None:
    wf = _workflow("nightly.yml")
    assert set(wf["on"]) == {"workflow_dispatch"}
    steps = [s.get("run", "") for s in wf["jobs"]["nightly"]["steps"]]
    assert any(s.startswith("python tools/tasks.py nightly") for s in steps)
