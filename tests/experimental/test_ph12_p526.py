"""PH-12 (EXPERIMENTAL, not in any gate): ITU-R P.526 smooth earth (DESIGN.md 5.9, 10.2).

Regression against the scratch values of ``research/scratch/pm_checks.py`` (T15, copied to
``tests/oracles/data/physics_model_checks.json``): excess loss 37.14 dB at 52 km,
h = 200/1.5 m and 20.28 dB at 200/10 m, 100 MHz (+-0.01 dB). Also: the module is labelled
EXPERIMENTAL and is never imported by the core.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sremu.physics.experimental import p526
from tests.oracles import p526 as p526_oracle
from tests.oracles._data import pm_checks
from tests.oracles.tables import ORACLE

pytestmark = pytest.mark.experimental

SRC = Path(__file__).resolve().parents[2] / "src" / "sremu"
EXPERIMENTAL_PKG = "sremu.physics.experimental"


@pytest.mark.parametrize(
    ("key", "hr"), [("PH12_52km_h1p5_excess_db", 1.5), ("PH12_52km_h10_excess_db", 10.0)]
)
def test_ph12_regression_values(key: str, hr: float):
    exc, branch = p526.p526_smooth(52.0, 200.0, hr, 100.0)
    o = ORACLE[key]
    assert abs(exc - o.value) <= o.tol
    assert branch == "interpolated"


@pytest.mark.parametrize(("hr", "want_db"), [(1.5, 37.14), (10.0, 20.28)])
def test_ph12_implementation_matches_the_pinned_oracle_literals(hr: float, want_db: float):
    """Implementation test of PH-12 with the spec's oracle values written out (DESIGN 10.2):
    sremu.physics.experimental.p526 at 52 km, h1 = 200 m, 100 MHz must give 37.14 dB
    (h2 = 1.5 m) and 20.28 dB (h2 = 10 m) within 0.01 dB. The literals also catch a silent
    edit of the oracle table that ``test_ph12_regression_values`` reads."""
    exc, branch = p526.p526_smooth(52.0, 200.0, hr, 100.0)
    assert abs(exc - want_db) <= 0.01
    assert branch == "interpolated"


def test_ph12_full_scratch_table():
    """All six T15 rows (48/52/56 km x 1.5/10 m): excess loss within 0.01 dB of the scratch
    values (rounded there to 0.01 dB) and the same branch."""
    rows = pm_checks()["T15_FM_direct_path_loss"]
    assert len(rows) == 6
    for row in rows:
        exc, branch = p526.p526_smooth(row["d_km"], 200.0, row["hr_m"], 100.0)
        assert abs(exc - row["P526_smooth_excess_dB"]) <= 0.01
        assert branch == row["P526_mode"]


@pytest.mark.parametrize("pol", ["H", "V"])
def test_ph12_port_equals_the_independent_oracle_port(pol: str):
    """Code-to-code with WP4's port of the same scratch function over all branches."""
    for d_km in (5.0, 20.0, 48.0, 52.0, 60.0, 70.0, 90.0):
        for h2 in (1.5, 10.0, 34.9, 120.0):
            got = p526.p526_smooth(d_km, 200.0, h2, 100.0, pol=pol)
            ref = p526_oracle.p526_smooth_excess_db(d_km, 200.0, h2, 100.0, pol=pol)
            assert got[1] == ref[1]
            assert abs(got[0] - ref[0]) <= 1e-12 * max(1.0, abs(ref[0]))
    assert p526.radio_horizon_km(200.0, 1.5) == p526_oracle.radio_horizon_km(200.0, 1.5)


def test_ph12_is_labelled_experimental():
    assert p526.EXPERIMENTAL is True
    assert (p526.__doc__ or "").startswith("EXPERIMENTAL")
    assert "EXPERIMENTAL" in (p526.p526_smooth.__doc__ or "")


_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w, ]+)|import\s+([\w., ]+))", re.MULTILINE
)


def _imports(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text)
    except SyntaxError:  # a file being edited elsewhere: fall back to a line scan
        out_re: set[str] = set()
        for frm, names, plain in _IMPORT_RE.findall(text):
            if frm:
                out_re.add(frm)
                out_re.update(f"{frm}.{n.strip()}" for n in names.split(","))
            else:
                out_re.update(n.strip().split(" ")[0] for n in plain.split(","))
        return out_re
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def test_ph12_core_never_imports_the_experimental_package():
    """Static: no module outside ``physics/experimental`` imports it (section 5.9)."""
    exp_dir = SRC / "physics" / "experimental"
    offenders = []
    for path in SRC.rglob("*.py"):
        if exp_dir in path.parents:
            continue
        if any(
            m == EXPERIMENTAL_PKG or m.startswith(EXPERIMENTAL_PKG + ".") for m in _imports(path)
        ):
            offenders.append(str(path.relative_to(SRC)))
    assert offenders == []


@pytest.mark.parametrize(
    "src",
    [
        "from sremu.physics.experimental.p526 import p526_smooth\n",
        "import sremu.physics.experimental.p526 as p\n",
        "from sremu.physics import experimental\n",
        "from sremu.physics.experimental import p526\nx = (\n",  # unparsable: line scan
    ],
)
def test_ph12_static_check_has_teeth(tmp_path: Path, src: str):
    bad = tmp_path / "bad.py"
    bad.write_text(src, encoding="utf-8")
    found = _imports(bad)
    assert any(m == EXPERIMENTAL_PKG or m.startswith(EXPERIMENTAL_PKG + ".") for m in found)


def test_ph12_importing_the_core_does_not_load_it():
    """Dynamic: building a full model (ground, walls, edges, diffuse) leaves the
    experimental package unloaded."""
    code = (
        "import sys\n"
        "import sremu, sremu.physics, sremu.physics.propagation\n"
        "import sremu.physics.ground, sremu.physics.walls\n"
        "import sremu.physics.diffraction, sremu.physics.diffuse\n"
        "print(any(m.startswith('sremu.physics.experimental') for m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
