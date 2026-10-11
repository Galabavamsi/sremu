"""Packaging and tier configuration against DESIGN.md sections 2.1, 2.2, 2.4 and 2.6.

Expected values are transcribed from the spec text, not read back from the files.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))

# section 2.4 "Core" list, verbatim
SPEC_CORE_DEPS = {
    "numpy>=2.2",
    "scipy>=1.14",
    "numba>=0.62",
    "pydantic>=2.12",
    "pyyaml>=6",
    "pyzmq>=27",
    "typer>=0.12",
    "matplotlib>=3.9",
}
# section 2.4 "Extras": extra -> package names
SPEC_EXTRAS = {
    "dev": {"pytest", "hypothesis", "ruff", "mypy"},
    "ml": {"gymnasium"},
    "sigmf": {"sigmf"},
    "parquet": {"pyarrow"},
    "gpu": {"torch"},
}
# section 2.6 tier contents
SPEC_CI = {"unit", "physics", "runtime", "io", "golden"}
SPEC_NIGHTLY = {"statistical", "integration", "validation_smoke"}


def _load_tasks():
    spec = importlib.util.spec_from_file_location("sremu_tasks", REPO / "tools" / "tasks.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _names(expr: str) -> set[str]:
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    return words - {"or", "and", "not"}


def _pkg(req: str) -> str:
    return re.split(r"[<>=!~\[; ]", req, maxsplit=1)[0].lower()


def test_build_system_and_metadata_section_2_1_and_2_4():
    bs = PYPROJECT["build-system"]
    assert bs["build-backend"] == "setuptools.build_meta"
    assert "setuptools>=70" in bs["requires"]
    proj = PYPROJECT["project"]
    assert proj["name"] == "sremu"
    assert proj["license"] == {"text": "LicenseRef-Pending"}
    assert proj["requires-python"] == ">=3.12"
    assert proj["scripts"]["sremu"] == "sremu.cli:app"
    assert (REPO / "LICENSE-PENDING.md").is_file()
    assert not (REPO / "LICENSE").exists()  # licence is an open ADMIN gate (H-2)


def test_core_dependencies_equal_section_2_4():
    assert set(PYPROJECT["project"]["dependencies"]) == SPEC_CORE_DEPS


def test_extras_cover_section_2_4():
    extras = PYPROJECT["project"]["optional-dependencies"]
    for name, pkgs in SPEC_EXTRAS.items():
        assert pkgs <= {_pkg(r) for r in extras[name]}, name


def test_tier_marker_expressions_match_section_2_6_and_are_registered():
    """``--strict-markers`` does not check ``-m`` expressions: a typo there silently selects
    nothing, so the selections are checked against the registered markers here."""
    tasks = _load_tasks()
    registered = {
        m.split(":", 1)[0].strip() for m in PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]
    }
    assert _names(tasks.CI_MARKERS) == SPEC_CI
    assert _names(tasks.NIGHTLY_MARKERS) == SPEC_NIGHTLY
    assert SPEC_CI | SPEC_NIGHTLY <= registered
    assert tasks.BUDGET_S == {"ci": 8 * 60, "nightly": 30 * 60}


def test_conftest_tier_markers_are_registered():
    from tests import conftest

    registered = {
        m.split(":", 1)[0].strip() for m in PYPROJECT["tool"]["pytest"]["ini_options"]["markers"]
    }
    assert set(conftest.TIER_MARKERS) <= registered
    assert set(conftest._DIR_TIER.values()) <= registered


def test_gitattributes_lf_for_csv_and_json_section_2_2():
    rules = [
        (ln.split()[0], ln.split()[1:])
        for ln in (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    order = {pattern: i for i, (pattern, _) in enumerate(rules)}
    lines = dict(rules)
    for pattern in ("*.csv", "*.json"):
        assert {"text", "eol=lf"} <= set(lines[pattern]), pattern
    # hash-pinned data stays byte-exact: its "-text" rules come later, so they win (gitattributes
    # applies the last matching line per attribute)
    for pinned in ("data/validation/**", "tests/golden/**"):
        assert lines[pinned] == ["-text"], pinned
        assert order[pinned] > max(order["*.csv"], order["*.json"]), pinned


def test_readme_states_independence_d24():
    """D24 (fresh-attempt framing) replaces the lineage statement of section 2.1."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "sremu is an independent new implementation" in text
    assert "third generation" not in text.lower()
    assert "## Lineage" not in text and "v1-legacy" not in text and "v2-legacy" not in text


def test_d13_offline_install_documented():
    for doc in ("README.md", "CONTRIBUTING.md"):
        text = (REPO / doc).read_text(encoding="utf-8")
        assert "uv pip install --offline --no-deps -e ." in text, doc


def test_d21_pytest_tasks_use_8_workers():
    tasks = _load_tasks()
    assert tasks.XDIST_WORKERS == 8
    if importlib.util.find_spec("xdist") is not None:
        assert tasks._xdist_args() == ["-n", "8"]


class _Recorder:
    def __init__(self, rc: int = 0) -> None:
        self.rc = rc
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], budget_s: float | None = None) -> int:
        self.calls.append(list(cmd))
        return self.rc


def test_d21_paper_junit_runs_ci_and_nightly_and_writes_the_sidecar(tmp_path, monkeypatch):
    tasks = _load_tasks()
    rec = _Recorder(rc=0)
    monkeypatch.setattr(tasks, "_run", rec)
    monkeypatch.setattr(tasks, "_git_state", lambda repo: ("0123abcd" * 5, True))
    junit = tmp_path / "out" / "sremu-junit.xml"
    assert tasks.main(["paper-junit", str(junit), "-x"]) == 0
    (cmd,) = rec.calls
    assert cmd[1:3] == ["-m", "pytest"]
    i = cmd.index("-m", 3)
    assert _names(cmd[i + 1]) == SPEC_CI | SPEC_NIGHTLY
    assert f"--junitxml={junit}" in cmd and "-x" in cmd
    meta = json.loads((tmp_path / "out" / "sremu-junit.meta.json").read_text(encoding="utf-8"))
    assert meta == {"sremu_commit": "0123abcd" * 5, "sremu_dirty": True}
    assert tasks.meta_path(Path("a/b/junit.xml")) == Path("a/b/junit.meta.json")


def test_d21_paper_junit_relative_path_failures_and_git_errors(tmp_path, monkeypatch):
    tasks = _load_tasks()
    monkeypatch.chdir(tmp_path)
    rec = _Recorder(rc=1)  # failing tests: the exit code propagates, the sidecar is written
    monkeypatch.setattr(tasks, "_run", rec)
    monkeypatch.setattr(tasks, "_git_state", lambda repo: ("f" * 40, False))
    assert tasks.main(["paper-junit", "j.xml"]) == 1
    assert f"--junitxml={tmp_path / 'j.xml'}" in rec.calls[0]
    meta = json.loads((tmp_path / "j.meta.json").read_text(encoding="utf-8"))
    assert meta == {"sremu_commit": "f" * 40, "sremu_dirty": False}

    def no_git(repo):
        raise RuntimeError("not a git repository")

    rec2 = _Recorder()
    monkeypatch.setattr(tasks, "_run", rec2)
    monkeypatch.setattr(tasks, "_git_state", no_git)
    assert tasks.main(["paper-junit", "k.xml"]) == 1
    assert rec2.calls == [] and not (tmp_path / "k.meta.json").exists()


def test_d21_git_state_uses_read_only_commands(monkeypatch):
    """The two git calls (no git is run here): rev-parse HEAD and status --porcelain."""
    tasks = _load_tasks()
    seen: list[list[str]] = []
    outputs = {"rev-parse": "abc\n", "status": " M src/sremu/units.py\n"}

    def fake_run(args, **kw):
        seen.append(list(args))
        return subprocess.CompletedProcess(args, 0, outputs[args[3]], "")

    monkeypatch.setattr(tasks.shutil, "which", lambda name: "git")
    monkeypatch.setattr(tasks.subprocess, "run", fake_run)
    assert tasks._git_state(Path("R")) == ("abc", True)
    assert [a[3:] for a in seen] == [["rev-parse", "HEAD"], ["status", "--porcelain"]]
    assert all(a[:3] == ["git", "-C", "R"] for a in seen)
    outputs["status"] = ""
    assert tasks._git_state(Path("R")) == ("abc", False)
    monkeypatch.setattr(tasks.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="git not found"):
        tasks._git_state(Path("R"))


def test_lazy_facade_resolves_foundation_names_and_reports_missing_owners():
    import sremu
    import sremu.contracts
    import sremu.errors
    import sremu.timebase

    assert sremu.TimeSpec is sremu.timebase.TimeSpec
    assert sremu.Rate is sremu.timebase.Rate
    assert sremu.TxMetadata is sremu.contracts.TxMetadata
    assert sremu.ErrorCode is sremu.errors.ErrorCode
    assert sremu.CONTRACT_VERSION == "1.1"
    assert {"Device", "StreamArgs", "TimeSpec"} <= set(dir(sremu))
    if importlib.util.find_spec("sremu.api.device") is None:  # until WP7 lands
        with pytest.raises(AttributeError, match="WP7"):
            _ = sremu.Device
    with pytest.raises(AttributeError):
        _ = sremu.no_such_name
