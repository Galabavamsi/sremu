"""``sremu states import|check|list`` (WP2 CLI module, lazily registered by WP0)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from sremu.commands.states import app
from sremu.physics.response import ResponseTable

pytestmark = pytest.mark.unit
runner = CliRunner()

PROV = {
    "source_kind": "measured",
    "description": "cli test",
    "reference": "test",
    "reference_plane_offset_m": 0.0,
    "polarization": "co",
    "includes_cell_pattern": False,
    "sigma_mag_db": 0.1,
    "sigma_phase_deg": 1.0,
    "raw_sha256": None,
    "license": "test",
    "created": {"date": "2026-10-10"},
}


def test_states_check_builtin_and_missing_assume_mag():
    r = runner.invoke(
        app,
        [
            "check",
            "lab_binary_3p5ghz",
            "--assume-mag",
            "sigma1=1",
            "--assume-mag",
            "sigma2=1",
            "--json",
        ],
    )
    assert r.exit_code == 0, r.output
    info = json.loads(r.output)
    assert info["states"] == ["sigma1", "sigma2"]
    assert info["md_db"]["d_db"] == pytest.approx(-0.18, abs=0.01)
    r = runner.invoke(app, ["check", "lab_binary_3p5ghz"])
    assert r.exit_code == 1
    assert runner.invoke(app, ["list"]).exit_code == 0


def test_states_import_csv_round_trip(tmp_path: Path):
    path = tmp_path / "tab.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["state", "f_hz", "theta_i_deg", "theta_s_deg", "mag_db", "phase_deg"])
        for s, ph in ((0, 30.0), (1, -150.0)):
            w.writerow([s, 3.5e9, "", "", -0.3, ph])
    (tmp_path / "tab.yaml").write_text(yaml.safe_dump({"provenance": PROV, "validity_bw_hz": 1e7}))
    out = tmp_path / "lib.yaml"
    r = runner.invoke(app, ["import", str(path), "--out", str(out)])
    assert r.exit_code == 0, r.output
    t = ResponseTable.from_state_library(out)
    src = ResponseTable.from_csv(path, tmp_path / "tab.yaml")
    assert t.labels == src.labels
    assert abs(t.gamma - src.gamma).max() <= 1e-12
    r = runner.invoke(app, ["check", str(out)])
    assert r.exit_code == 0, r.output


def test_states_import_rejects_invalid_table(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("state,f_hz,theta_i_deg,theta_s_deg,mag_db,phase_deg\n0,3.5e9,,,3.0,0\n")
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump({"provenance": PROV}))
    r = runner.invoke(app, ["import", str(path)])
    assert r.exit_code == 1  # |Gamma| = +3 dB fails passivity


def test_states_import_matlab_scurve_and_unknown_symmetrize(tmp_path: Path):
    """``--format matlab``: lossless S-curve rows (tan_delta = 0) become unit-amplitude states
    exp(j phase), linear in f (closed form); an unknown --symmetrize value is refused."""
    import numpy as np

    path = tmp_path / "unitcell_fullwave_cli.csv"
    rows = ["f_hz,tan_delta,patch_L_mm,gamma_phase_deg,gamma_abs"]
    rows += [f"{f},0,{L},{120 - 20 * L + f / 1e9},1.0" for f in (3.4e9, 3.6e9) for L in (1, 4)]
    rows.append("3.4e9,0.02,1,7.0,0.4")  # lossy row: excluded
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "scurve.yaml"
    r = runner.invoke(app, ["import", str(path), "--format", "matlab", "--out", str(out)])
    assert r.exit_code == 0, r.output
    t = ResponseTable.from_state_library(out)
    assert t.labels == ("L1mm", "L4mm") and t.provenance.source_kind == "fullwave"
    want = 0.5 * (
        np.exp(1j * np.radians([100 + 3.4, 40 + 3.4]))
        + np.exp(1j * np.radians([100 + 3.6, 40 + 3.6]))
    )
    assert np.max(np.abs(t.gamma_at(3.5e9) - want)) <= 1e-12
    r = runner.invoke(app, ["import", str(path), "--format", "matlab", "--symmetrize", "max"])
    assert r.exit_code == 1
