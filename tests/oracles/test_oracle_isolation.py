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


# --------------------------------------------------------------------------- D4 legacy constant

LEGACY_C = "C_LEGACY_V1"
LEGACY_C_ALLOWED = {"ris": {"<import>", "legacy_v1_h"}}  # D4: only the legacy_v1 oracle


def legacy_c_scopes(source: str) -> set[str]:
    """Where ``source`` references ``units.C_LEGACY_V1``: ``"<import>"`` for an import of the
    name, the top-level function name for a use inside a function (defaults included), and
    ``"<module>"`` for any other module-level use."""
    import ast

    tree = ast.parse(source)
    names = {LEGACY_C} | {  # import aliases of the constant count as the constant
        a.asname
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        for a in n.names
        if a.name == LEGACY_C and a.asname
    }

    def uses(node: ast.AST) -> bool:
        return any(
            (isinstance(n, ast.Name) and n.id in names)
            or (isinstance(n, ast.Attribute) and n.attr == LEGACY_C)
            for n in ast.walk(node)
        )

    scopes: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and any(a.name == LEGACY_C for a in node.names):
            scopes.add("<import>")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and uses(node):
            scopes.add(node.name)
        elif uses(node):
            scopes.add("<module>")
    return scopes


def test_d4_legacy_c_used_only_by_the_legacy_v1_oracle():
    """D4: v1's ``c = 3e8`` (``units.C_LEGACY_V1``) enters the oracles only through
    :func:`tests.oracles.ris.legacy_v1_h`; every other oracle uses ``units.C0``."""
    found = {
        p.stem: s
        for p in sorted(ORACLE_DIR.glob("*.py"))
        if not p.name.startswith("test_") and (s := legacy_c_scopes(p.read_text(encoding="utf-8")))
    }
    assert found == LEGACY_C_ALLOWED, found
    import inspect

    from sremu.units import C0, C_LEGACY_V1
    from tests.oracles.ris import legacy_v1_h

    assert inspect.signature(legacy_v1_h).parameters["c_m_per_s"].default == C_LEGACY_V1
    assert C_LEGACY_V1 != C0


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "from sremu.units import C_LEGACY_V1\ndef k1(c=C_LEGACY_V1):\n    return c\n",
            {"<import>", "k1"},
        ),
        ("from sremu import units\ndef f():\n    return units.C_LEGACY_V1\n", {"f"}),
        ("from sremu.units import C_LEGACY_V1 as c\nC = c\n", {"<import>", "<module>"}),
        ("from sremu import units\nC = units.C_LEGACY_V1\n", {"<module>"}),
        ("from sremu.units import C0\ndef f(c=C0):\n    return c\n", set()),
    ],
)
def test_d4_legacy_c_scope_finder_has_teeth(src, expected):
    assert legacy_c_scopes(src) == expected


@pytest.mark.parametrize("name", ["physics_model_checks.json", "tang2022_fig10_vector.json"])
def test_oracle_data_sha256_matches_provenance(name):
    entry = provenance()["files"][name]
    assert sha256_lf(DATA / name) == entry["sha256"]
    assert entry["data_class"] in {"ANALYTIC", "DIG"}
    for field in ("source", "source_sha256", "generator", "generator_sha256", "caveats"):
        assert entry[field]
