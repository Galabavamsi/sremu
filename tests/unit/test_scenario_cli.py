"""``sremu validate`` and ``sremu schema`` through the lazy CLI registry (DESIGN.md 2.5, 9.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sremu.cli import app
from sremu.record.manifest import manifest_json_schema
from sremu.scenario.load import dump_yaml, read_document
from sremu.scenario.schema_export import (
    scenario_json_schema,
    schema_text,
    state_library_json_schema,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "scenarios" / "lab_like_A.yaml"
runner = CliRunner()


def test_validate_valid_scenario_prints_derived_quantities():
    r = runner.invoke(app, ["validate", str(LAB)])
    assert r.exit_code == 0, r.output
    out = r.output
    assert out.startswith("VALID")
    assert "noise floor -106.975 dBm" in out  # -173.975 dBm/Hz + 60 dB + NF 7 dB
    assert "page 512" in out  # 1 MS/s (4.3)
    assert "assumption: ris ris0: |Gamma| of state 'sigma1'" in out
    assert "FD 32 taps" in out and "narrowband" in out


def test_validate_json_output():
    r = runner.invoke(app, ["validate", "--json", str(LAB)])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["valid"] is True and data["errors"] == []
    assert data["derived"]["streams"]["rx1.rx0"]["page_size"] == 512


def test_validate_exit_codes(tmp_path):
    doc = read_document(LAB)
    doc["nodes"][2]["position_m"] = [0.05, 0.0, 0.0]  # closer than one wavelength to a cell
    f = tmp_path / "bad_guard.yaml"
    f.write_text(dump_yaml(doc), encoding="utf-8")
    r = runner.invoke(app, ["validate", str(f)])
    assert r.exit_code == 1
    assert "CELL_TOO_CLOSE" in r.output
    doc["run"]["duration_s"] = 5.0  # float for an exact quantity: schema error
    g = tmp_path / "bad_schema.yaml"
    g.write_text(dump_yaml(doc), encoding="utf-8")
    r = runner.invoke(app, ["validate", str(g)])
    assert r.exit_code == 2
    assert "floats are rejected" in r.output
    r = runner.invoke(app, ["validate", str(tmp_path / "missing.yaml")])
    assert r.exit_code == 2


def _angle_table(phase_deg) -> dict:
    """A 2-state analytic angle table on theta = -30, 0, 30 deg, |Gamma| = 0.5."""
    base = read_document(REPO / "data" / "states" / "ideal_antipodal.yaml")
    grid = [-30.0, 0.0, 30.0]
    base.update(
        name="angle_test",
        angle_grid={"theta_deg": grid},
        states=[
            {
                "id": s,
                "label": f"s{s}",
                "gamma": [
                    {
                        "f_hz": None,
                        "mag": 0.5,
                        "phase_deg": 180.0 * s + phase_deg(a, b),
                        "theta_i_deg": a,
                        "theta_s_deg": b,
                    }
                    for a in grid
                    for b in grid
                ],
            }
            for s in range(2)
        ],
    )
    return base


@pytest.mark.parametrize(
    ("phase", "code", "exc_name"),
    [
        (lambda a, b: 5.0 * (abs(a) + abs(b)) / 30.0, None, None),  # symmetric, fine grid
        (lambda a, b: 5.0 * (a - b) / 30.0, "TABLE_VALIDATOR", "ReciprocityError"),
        (lambda a, b: 60.0 * (abs(a) + abs(b)) / 30.0, "TABLE_VALIDATOR", "GridTooCoarse"),
    ],
    ids=["reciprocal", "asymmetric", "coarse"],
)
def test_validate_runs_the_physics_table_validators(tmp_path, phase, code, exc_name):
    """Section 9.5 'table validators (5.5)': reciprocity (tol 1e-9 for sigma = 0 analytic
    tables) and grid density (adjacent |dGamma|/max|Gamma| <= 0.35) come from the physics
    ResponseTable. Oracles: the asymmetric table differs from its transpose by
    |0.5 (e^{j10deg} - e^{-j10deg})| = sin(10 deg) = 0.17 >> 1e-9; the coarse one has adjacent
    steps of |0.5 (e^{j60deg} - 1)|/0.5 = 1 > 0.35; the fine one 2 sin(2.5 deg) = 0.087."""
    (tmp_path / "angle.yaml").write_text(dump_yaml(_angle_table(phase)), encoding="utf-8")
    doc = read_document(REPO / "scenarios" / "quickstart.yaml")
    doc["ris"][0]["states"] = {"file": "angle.yaml"}
    f = tmp_path / "s.yaml"
    f.write_text(dump_yaml(doc), encoding="utf-8")
    r = runner.invoke(app, ["validate", "--json", str(f)])
    data = json.loads(r.output)
    if code is None:
        assert r.exit_code == 0, r.output
        assert data["valid"] is True and data["errors"] == []
    else:
        assert r.exit_code == 1, r.output
        errs = [e for e in data["errors"] if e["code"] == code]
        assert errs and errs[0]["exc"] == exc_name, data["errors"]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("scenario", scenario_json_schema),
        ("states", state_library_json_schema),
        ("manifest", manifest_json_schema),
    ],
)
def test_schema_command(kind, expected, tmp_path):
    r = runner.invoke(app, ["schema", kind])
    assert r.exit_code == 0, r.output
    assert r.output == schema_text(expected())
    out = tmp_path / f"{kind}.json"
    r = runner.invoke(app, ["schema", kind, "--out", str(out)])
    assert r.exit_code == 0 and json.loads(out.read_text(encoding="utf-8")) == expected()


def test_schema_command_control_and_unknown():
    import sremu

    shipped = Path(sremu.__file__).parent / "transport" / "schema"
    r = runner.invoke(app, ["schema", "control"])
    if shipped.is_dir() and any(shipped.glob("*.json")):
        assert r.exit_code == 0
    else:
        assert r.exit_code == 3  # WP8 has not shipped the control schemas yet
    r = runner.invoke(app, ["schema", "nope"])
    assert r.exit_code == 2
