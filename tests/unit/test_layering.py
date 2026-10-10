"""Static architecture rules (DESIGN.md sections 2.3, 3.1, 6.1; RT-15 static part).

* Layering: a module may import only from its own layer or lower layers.
* Oracle isolation: ``tests/oracles`` must not import ``sremu.physics``, ``sremu.channel``
  or ``sremu.impair``.
* I7 (static part of RT-15): no module under ``physics/``, ``channel/``, ``impair/``,
  ``engine/scheduler.py``, ``engine/events.py`` or ``engine/commands.py`` may import
  ``time``, ``datetime`` or ``perf_counter``.
* Speed of light: no literal 299792458 / 3e8 / 2.998e8 (any spelling) outside ``units.py``.
* No global RNG: no ``np.random.seed``, legacy ``np.random.<dist>``/``RandomState``,
  stdlib ``random`` or spawn-order-dependent ``.spawn()`` in any Python tree (section 6.1).

Every checker has a "teeth" test proving it flags a known violation.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PKG = SRC / "sremu"

# --------------------------------------------------------------------------- layer map (2.3)
LAYER_OF: dict[str, int] = {
    "units": 0,
    "timebase": 0,
    "errors": 0,
    "contracts": 0,
    "rng": 1,
    "scenario": 1,
    "physics": 2,
    "channel": 3,
    "impair": 3,
    "engine": 4,
    "metrics": 4,
    "legacy": 4,
    "api": 5,
    "record": 5,
    "transport": 5,
    "control": 5,
    "ml": 5,
    "compat": 5,
    "analysis": 6,
    "experiments": 6,
    "cli": 7,
    "commands": 7,
}
TESTING = "testing"  # may import layer 0 only; production code must never import it
FACADE_NAMES = {"__version__"}  # names production modules may import from the top package

I7_PREFIXES = ("sremu.physics", "sremu.channel", "sremu.impair")
I7_MODULES = ("sremu.engine.scheduler", "sremu.engine.events", "sremu.engine.commands")
WALLCLOCK_MODULES = {"time", "datetime"}
WALLCLOCK_NAMES = {"perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns", "time_ns"}

ORACLE_FORBIDDEN = ("sremu.physics", "sremu.channel", "sremu.impair")

RNG_ALLOWED = {
    "Generator",
    "Philox",
    "SeedSequence",
    "BitGenerator",
    "PCG64",
    "PCG64DXSM",
    "MT19937",
    "SFC64",
    "default_rng",
}

C_LITERAL_RE = re.compile(
    r"^(?:299792458(?:\.0*)?|2\.99[78]\d*e\+?0*8|3(?:\.0*)?e\+?0*8"
    r"|0?\.299792458e\+?0*9|299792\.458(?:e\+?0*3)?)$",
    re.I,
)
C_SCAN_DIRS = ("src", "tests", "tools", "bench", "validation", "examples", "scenarios")
# "There is no global RNG anywhere" (section 6.1): every Python tree, not only src/
RNG_SCAN_DIRS = ("src", "tests", "tools", "bench", "validation", "examples")


@dataclass(frozen=True)
class Imp:
    """One import edge: the importing module, the imported module and the source line."""

    src: str
    target: str
    line: int
    names: tuple[str, ...] = ()


def module_name(path: Path, root: Path = SRC) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def is_package(path: Path) -> bool:
    return path.name == "__init__.py"


def iter_imports(source: str, modname: str, package: bool) -> Iterator[Imp]:
    """Every import in ``source`` (including nested and ``TYPE_CHECKING`` blocks), resolved
    to absolute module names; ``importlib.import_module("x")``/``__import__("x")`` with a
    constant string are included."""
    tree = ast.parse(source)
    pkg_parts = modname.split(".") if package else modname.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield Imp(modname, alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            names = tuple(a.name for a in node.names)
            yield Imp(modname, target, node.lineno, names)
        elif isinstance(node, ast.Call) and node.args:
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            arg = node.args[0]
            if (
                fname in ("import_module", "__import__")
                and isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
            ):
                yield Imp(modname, arg.value, node.lineno)


def _sremu_component(target: str) -> str | None:
    parts = target.split(".")
    if parts[0] != "sremu":
        return None
    return parts[1] if len(parts) > 1 else ""


def layering_violations(source: str, modname: str, package: bool) -> list[str]:
    """Layering errors for one module of the ``sremu`` package."""
    comp = modname.split(".")[1] if "." in modname else ""
    errors: list[str] = []
    if comp == "":
        src_layer = 0  # sremu/__init__.py: module-level imports must stay at the bottom
    elif comp == TESTING:
        src_layer = 0
    elif comp in LAYER_OF:
        src_layer = LAYER_OF[comp]
    else:
        return [f"{modname}: unclassified module; add it to the layer map (CCR)"]
    for imp in iter_imports(source, modname, package):
        tcomp = _sremu_component(imp.target)
        if tcomp is None:
            continue
        targets: list[str] = []
        if tcomp == "":  # from sremu import X
            for name in imp.names:
                if name in FACADE_NAMES:
                    continue
                if (PKG / name).is_dir() or (PKG / f"{name}.py").is_file() or name in LAYER_OF:
                    targets.append(name)
                else:
                    errors.append(
                        f"{modname}:{imp.line}: 'from sremu import {name}' uses the lazy facade;"
                        " import from the defining module"
                    )
        else:
            targets.append(tcomp)
        for t in targets:
            if t == TESTING:
                if comp != TESTING:
                    errors.append(f"{modname}:{imp.line}: production code imports sremu.testing")
                continue
            if t not in LAYER_OF:
                errors.append(f"{modname}:{imp.line}: imports unclassified sremu.{t}")
                continue
            if LAYER_OF[t] > src_layer:
                errors.append(
                    f"{modname}:{imp.line}: layer {src_layer} imports sremu.{t} "
                    f"(layer {LAYER_OF[t]})"
                )
    return errors


def wallclock_violations(source: str, modname: str, package: bool) -> list[str]:
    errors = []
    for imp in iter_imports(source, modname, package):
        root = imp.target.split(".")[0]
        if root in WALLCLOCK_MODULES:
            errors.append(f"{modname}:{imp.line}: imports {imp.target} (I7)")
        bad = set(imp.names) & WALLCLOCK_NAMES
        if bad:
            errors.append(f"{modname}:{imp.line}: imports {sorted(bad)} (I7)")
    return errors


def oracle_violations(source: str, modname: str, package: bool) -> list[str]:
    errors = []
    for imp in iter_imports(source, modname, package):
        targets = [imp.target]
        if imp.target == "sremu":
            targets = [f"sremu.{n}" for n in imp.names]
        for t in targets:
            if any(t == f or t.startswith(f + ".") for f in ORACLE_FORBIDDEN):
                errors.append(f"{modname}:{imp.line}: oracle imports {t}")
    return errors


def c_literal_violations(source: str, label: str) -> list[str]:
    errors = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.NUMBER:
            text = tok.string.replace("_", "")
            if C_LITERAL_RE.match(text):
                errors.append(f"{label}:{tok.start[0]}: speed-of-light literal {tok.string!r}")
    return errors


def rng_violations(source: str, label: str) -> list[str]:
    """Global/legacy RNG use: numpy legacy API, ``np.random.seed`` and stdlib ``random``."""
    tree = ast.parse(source)
    np_aliases: set[str] = set()
    npr_aliases: set[str] = set()
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "numpy":
                    np_aliases.add(a.asname or "numpy")
                elif a.name == "numpy.random":
                    if a.asname:
                        npr_aliases.add(a.asname)
                    else:
                        np_aliases.add("numpy")
                elif a.name == "random":
                    errors.append(f"{label}:{node.lineno}: stdlib 'random' (global RNG)")
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            if node.module == "numpy":
                for a in node.names:
                    if a.name == "random":
                        npr_aliases.add(a.asname or "random")
            elif node.module == "numpy.random":
                for a in node.names:
                    if a.name not in RNG_ALLOWED:
                        errors.append(f"{label}:{node.lineno}: legacy numpy.random.{a.name}")
            elif node.module == "random":
                errors.append(f"{label}:{node.lineno}: stdlib 'random' (global RNG)")
    for node in ast.walk(tree):
        # SeedSequence.spawn / Generator.spawn / BitGenerator.spawn depend on spawn order
        # (section 6.1: "SeedSequence.spawn is not used")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "spawn"
        ):
            errors.append(f"{label}:{node.lineno}: .spawn() is spawn-order dependent (6.1)")
        if not isinstance(node, ast.Attribute):
            continue
        v = node.value
        via_np = (
            isinstance(v, ast.Attribute)
            and v.attr == "random"
            and isinstance(v.value, ast.Name)
            and v.value.id in np_aliases
        )
        via_npr = isinstance(v, ast.Name) and v.id in npr_aliases
        if (via_np or via_npr) and node.attr not in RNG_ALLOWED:
            errors.append(f"{label}:{node.lineno}: legacy/global numpy.random.{node.attr}")
    return errors


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if ".venv" not in p.parts)


def _src_modules() -> list[tuple[Path, str]]:
    return [(p, module_name(p)) for p in _py_files(PKG)]


# --------------------------------------------------------------------------- tests


def test_layering_src_respects_layers():
    errors = []
    for path, mod in _src_modules():
        errors += layering_violations(path.read_text(encoding="utf-8"), mod, is_package(path))
    assert not errors, "\n".join(errors)


def test_layering_teeth_detects_upward_and_testing_imports():
    src = "from sremu.engine import scheduler\nfrom ..transport import client\n"
    errs = layering_violations(src, "sremu.physics.los", package=False)
    assert len(errs) == 2
    assert "layer 2 imports sremu.engine (layer 4)" in errs[0]
    assert "sremu.transport" in errs[1]
    errs = layering_violations("import sremu.testing.fake_engine\n", "sremu.engine.x", False)
    assert errs and "sremu.testing" in errs[0]
    errs = layering_violations("from sremu import Device\n", "sremu.record.report", False)
    assert errs and "lazy facade" in errs[0]
    errs = layering_violations("import os\n", "sremu.mystery", False)
    assert errs and "unclassified" in errs[0]
    # lower or equal layers are fine, including relative and TYPE_CHECKING imports
    ok = (
        "from typing import TYPE_CHECKING\nfrom sremu.units import C0\nfrom . import geometry\n"
        "if TYPE_CHECKING:\n    from sremu.scenario.models import Scenario\n"
    )
    assert layering_violations(ok, "sremu.physics.los", False) == []
    # but TYPE_CHECKING-only upward imports still count
    bad = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from sremu.api import device\n"
    assert layering_violations(bad, "sremu.engine.emulator", False)


def test_layering_testing_package_imports_foundation_only():
    for path, mod in _src_modules():
        if mod.startswith("sremu.testing"):
            for imp in iter_imports(path.read_text(encoding="utf-8"), mod, is_package(path)):
                comp = _sremu_component(imp.target)
                if comp not in (None, "", TESTING):
                    assert LAYER_OF[comp] == 0, f"{mod}:{imp.line} imports sremu.{comp}"


def test_layering_relative_import_resolution():
    imps = list(
        iter_imports("from ..channel import ring\nfrom . import x\n", "sremu.engine.a", False)
    )
    assert [i.target for i in imps] == ["sremu.channel", "sremu.engine"]
    imps = list(iter_imports("from .ring import R\n", "sremu.channel", True))
    assert imps[0].target == "sremu.channel.ring"


def test_rt15_static_no_wallclock_in_data_plane():
    errors = []
    for path, mod in _src_modules():
        if mod.startswith(I7_PREFIXES) or mod in I7_MODULES:
            errors += wallclock_violations(path.read_text(encoding="utf-8"), mod, is_package(path))
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "src",
    [
        "import time\n",
        "import datetime as dt\n",
        "from time import perf_counter\n",
        "from datetime import datetime\n",
        "def f():\n    import time\n    return time.time()\n",
        "import importlib\nt = importlib.import_module('time')\n",
        "from time import monotonic_ns\n",
    ],
)
def test_rt15_teeth_wallclock_imports_are_flagged(src):
    assert wallclock_violations(src, "sremu.physics.los", False)


def test_fakes_import_no_wallclock():
    """The WP0 fakes stand in for the data plane (FakeEngine claims "no wall clock is read")."""
    errors = []
    for path, mod in _src_modules():
        if mod.startswith("sremu.testing"):
            errors += wallclock_violations(path.read_text(encoding="utf-8"), mod, is_package(path))
    assert not errors, "\n".join(errors)


def test_rt15_scope_includes_listed_modules():
    assert "sremu.engine.scheduler" in I7_MODULES
    assert "sremu.engine.events" in I7_MODULES
    assert "sremu.engine.commands" in I7_MODULES
    assert "sremu.engine.pacer" not in I7_MODULES  # wall clock is allowed in the pacer


def test_oracle_isolation():
    root = REPO / "tests" / "oracles"
    errors = []
    for path in _py_files(root) if root.exists() else []:
        mod = module_name(path, REPO)
        errors += oracle_violations(path.read_text(encoding="utf-8"), mod, is_package(path))
    assert not errors, "\n".join(errors)


def test_oracle_isolation_teeth():
    for src in (
        "from sremu.physics.ris_kernel import k1\n",
        "import sremu.channel.render_numpy\n",
        "from sremu import impair\n",
    ):
        assert oracle_violations(src, "tests.oracles.k1", False)
    assert oracle_violations("from sremu.units import C0\n", "tests.oracles.k1", False) == []


def test_speed_of_light_literal_only_in_units():
    errors = []
    this = Path(__file__).resolve()
    for d in C_SCAN_DIRS:
        root = REPO / d
        if not root.exists():
            continue
        for path in _py_files(root):
            if path.resolve() in (this, (PKG / "units.py").resolve()):
                continue
            errors += c_literal_violations(
                path.read_text(encoding="utf-8"), str(path.relative_to(REPO))
            )
    assert not errors, "\n".join(errors)
    # units.py defines it exactly once
    units_hits = c_literal_violations((PKG / "units.py").read_text(encoding="utf-8"), "units.py")
    assert len(units_hits) == 1


@pytest.mark.parametrize(
    "src",
    [
        "c = 299792458\n",
        "c = 299_792_458.0\n",
        "c = 3e8\n",
        "c = 3.0E+08\n",
        "c = 2.998e8\n",
        "c = 2.99792458e8\n",
        "c = 2.9979e8\n",
        "c = 2.99792e+08\n",
        "c = 0.299792458e9\n",
        "c_kmps = 299792.458\n",
        "c = 299792.458e3\n",
    ],
)
def test_speed_of_light_teeth(src):
    assert c_literal_violations(src, "x.py")


def test_speed_of_light_ignores_other_numbers_and_text():
    src = (
        "f = 300e6\nx = 3e9\ns = '3e8 in a string'\n# 299792458 in a comment\ny = 2.998\n"
        "z = 2.99e9\nw = 29.9792458\nf_vhf = 2.99e8\ng = 2.995e8\n"
    )
    assert c_literal_violations(src, "x.py") == []


def test_no_global_rng_anywhere():
    errors = []
    for d in RNG_SCAN_DIRS:
        root = REPO / d
        if not root.exists():
            continue
        for path in _py_files(root):
            errors += rng_violations(path.read_text(encoding="utf-8"), str(path.relative_to(REPO)))
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "src",
    [
        "import numpy as np\nnp.random.seed(1)\n",
        "import numpy as np\nx = np.random.rand(3)\n",
        "import numpy\nx = numpy.random.normal()\n",
        "import numpy.random as npr\nnpr.shuffle(a)\n",
        "from numpy import random\nrandom.randint(3)\n",
        "from numpy.random import randn\n",
        "import numpy as np\nr = np.random.RandomState(0)\n",
        "import random\n",
        "from random import random\n",
        "import numpy as np\nkids = np.random.SeedSequence(5).spawn(4)\n",
        "import numpy as np\ng = np.random.Generator(np.random.Philox(1))\nh = g.spawn(2)\n",
    ],
)
def test_no_global_rng_teeth(src):
    assert rng_violations(src, "x.py")


def test_no_global_rng_allows_keyed_generators():
    src = (
        "import numpy as np\n"
        "g = np.random.Generator(np.random.Philox(key=5))\n"
        "from numpy.random import Generator, Philox\n"
    )
    assert rng_violations(src, "x.py") == []


def test_every_section_2_2_subpackage_exists():
    for sub in (
        "scenario",
        "physics",
        "physics/experimental",
        "channel",
        "impair",
        "engine",
        "metrics",
        "legacy",
        "api",
        "record",
        "transport",
        "control",
        "ml",
        "analysis",
        "experiments",
        "testing",
        "commands",
    ):
        assert (PKG / sub / "__init__.py").is_file(), sub
    # rng.py and compat.py are modules (section 2.2), so no package may shadow them
    assert not (PKG / "rng").exists()
    assert not (PKG / "compat").exists()
