"""Import isolation of the oracles (DESIGN.md sections 2.3, 10.1, 14.3) and data provenance.

The static AST check lives in ``tests/unit/test_layering.py::test_oracle_isolation``. This
module adds the runtime check: importing every oracle module in a fresh interpreter loads no
module of ``sremu.physics``, ``sremu.channel`` or ``sremu.impair`` (directly or through a
transitive import).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.oracles._data import DATA, provenance, sha256_lf

pytestmark = pytest.mark.unit

ORACLE_DIR = Path(__file__).resolve().parent
REPO = ORACLE_DIR.parents[1]
FORBIDDEN = ("sremu.physics", "sremu.channel", "sremu.impair")


def _oracle_modules() -> list[str]:
    return sorted(
        f"tests.oracles.{p.stem}"
        for p in ORACLE_DIR.glob("*.py")
        if not p.name.startswith("test_") and p.name != "__init__.py"
    )


def test_oracle_modules_listed():
    mods = _oracle_modules()
    for expected in ("rf", "ris", "nonideal", "mobility", "p526", "tables", "_data"):
        assert f"tests.oracles.{expected}" in mods


def test_oracle_import_isolation_runtime():
    code = (
        "import importlib, json, sys\n"
        f"mods = {_oracle_modules()!r}\n"
        "for m in mods: importlib.import_module(m)\n"
        f"bad = sorted(m for m in sys.modules if m.startswith({FORBIDDEN!r}))\n"
        "print(json.dumps(bad))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    loaded = json.loads(out.stdout.strip().splitlines()[-1])
    assert loaded == [], f"oracles pulled in {loaded}"


def test_oracle_isolation_runtime_check_has_teeth():
    """The same probe flags a module that does load a forbidden package."""
    code = (
        "import sys, types, json\n"
        "sys.modules['sremu.physics.fake_probe'] = types.ModuleType('sremu.physics.fake_probe')\n"
        f"print(json.dumps(sorted(m for m in sys.modules if m.startswith({FORBIDDEN!r}))))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(out.stdout.strip()) == ["sremu.physics.fake_probe"]


@pytest.mark.parametrize("name", ["physics_model_checks.json", "tang2022_fig10_vector.json"])
def test_oracle_data_sha256_matches_provenance(name):
    entry = provenance()["files"][name]
    assert sha256_lf(DATA / name) == entry["sha256"]
    assert entry["data_class"] in {"ANALYTIC", "DIG"}
    for field in ("source", "source_sha256", "generator", "generator_sha256", "caveats"):
        assert entry[field]
