"""``sremu verify`` (WP4, DESIGN.md section 2.5): the CLI runs through the lazy registry, its
closed forms agree with the independent oracles in ``tests/oracles``, and FAIL is visible in
the exit status."""

from __future__ import annotations

import json
import math

import pytest
from typer.testing import CliRunner

from sremu import cli
from sremu.commands import verify
from tests.oracles import mobility as mob
from tests.oracles import nonideal as ni
from tests.oracles import rf
from tests.oracles.tables import ORACLE

pytestmark = pytest.mark.unit


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _by_quantity(results: list[dict]) -> dict[str, dict]:
    return {r["quantity"]: r for r in results}


def test_verify_quick_json_via_registry(runner):
    res = runner.invoke(cli.app, ["verify", "--quick", "--json"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out and all(r["group"] == "oracle" and r["status"] == "PASS" for r in out)
    ids = {r["spec_id"] for r in out}
    assert {
        "PH-01",
        "PH-03",
        "PH-06",
        "PH-07",
        "PH-10",
        "PH-11",
        "PH-13",
        "PH-14",
        "PH-17",
        "PH-21",
        "PH-25",
    } <= ids


def test_verify_closed_forms_agree_with_independent_oracles():
    """verify duplicates a few closed forms (the wheel has no tests/); they equal the oracles."""
    q = _by_quantity([r.__dict__ for r in verify.oracle_checks()])
    assert abs(q["Friis 3.5 GHz, 10 m"]["value"] - float(rf.db20(rf.friis_h(10.0, 3.5e9)))) <= 1e-12
    two_ray = -float(rf.db20(rf.two_ray_h(52e3, 200.0, 1.5, 100e6)))
    assert abs(q["two-ray 52 km, 200/1.5 m, 100 MHz"]["value"] - two_ray) <= 1e-9
    assert (
        abs(q["knife-edge loss at nu = 10"]["value"] - float(rf.knife_edge_loss_db(10.0))) <= 1e-12
    )
    assert (
        abs(q["eta(2) uniform 2-bit"]["value"] - 10 * math.log10(ni.uniform_bbit_eta(2))) <= 1e-12
    )
    assert abs(q["Abeywickrama (0.2, 1.6)"]["value"] - ni.abeywickrama_eta_db(0.2, 1.6)) <= 1e-9
    assert abs(q["LOS Doppler 30 m/s, 3.5 GHz"]["value"] - mob.doppler_hz(30.0, 3.5e9)) <= 1e-12
    assert abs(q["noise floor 1 MS/s, F = 0 dB"]["value"] - rf.thermal_noise_dbm(1e6)) <= 1e-12
    crb = rf.crb_frequency_std_hz(5880.0, 10.0, 2000)
    assert abs(q["CRB std, 10 dB, N = 2000"]["value"] - crb) <= 1e-15
    # expected values are the spec oracle values
    assert q["Friis 3.5 GHz, 10 m"]["expected"] == ORACLE["PH01_friis_3p5GHz_10m_db"].value
    assert (
        q["two-ray 52 km, 200/1.5 m, 100 MHz"]["expected"]
        == ORACLE["PH10_two_ray_52km_h1p5_db"].value
    )


def test_verify_full_never_fails_and_reports_implementation(runner):
    res = runner.invoke(cli.app, ["verify", "--json"])
    out = json.loads(res.output)
    impl = [r for r in out if r["group"] == "implementation"]
    assert {r["spec_id"] for r in impl} == {"PH-01", "PH-04", "PH-21"}
    assert all(r["status"] in {"PASS", "SKIP"} for r in impl), impl
    assert res.exit_code == 0


def test_verify_fail_sets_exit_status_and_text_table(runner, monkeypatch):
    bad = verify._judge("oracle", "PH-01", "teeth", -60.0, -63.329, 5e-4, "dB")
    assert bad.status == "FAIL"
    monkeypatch.setattr(verify, "oracle_checks", lambda: [bad])
    res = runner.invoke(cli.app, ["verify", "--quick"])
    assert res.exit_code == 1
    assert "FAIL" in res.output and "1 checks: 0 pass, 1 fail, 0 skipped" in res.output


def test_verify_nan_is_a_failure():
    assert verify._judge("oracle", "X", "nan", float("nan"), 0.0, 1.0, "1").status == "FAIL"
