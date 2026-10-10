"""``sremu run`` and ``sremu batch`` (WP6 command modules, lazily registered in ``cli.py``)."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from sremu.cli import load_command
from sremu.commands.batch import command as batch_command
from sremu.commands.run import command as run_command
from sremu.contracts import CommandRecord
from sremu.record.manifest import read_manifest
from sremu.scenario.load import dump_yaml, read_document

pytestmark = pytest.mark.unit
runner = CliRunner()


def _app(fn: object):
    import typer

    app = typer.Typer()
    app.command()(fn)  # type: ignore[arg-type]
    return app


def _scenario(tmp_path: Path, repo_root: Path, **edit: object) -> Path:
    d = read_document(repo_root / "scenarios" / "quickstart.yaml")
    d["run"] = {"duration_s": "0.02"}
    for k, v in edit.items():
        d[k] = v
    p = tmp_path / "s.yaml"
    p.write_text(dump_yaml(d), encoding="utf-8")  # YAML 1.2 core scalars (exact strings)
    return p


def test_cli_registry_resolves_run_and_batch() -> None:
    assert load_command("run") is not None
    assert load_command("batch") is not None


def test_cli_run_writes_run_directory(tmp_path: Path, repo_root: Path) -> None:
    sc = _scenario(tmp_path, repo_root, record={"iq": True, "events": True, "metrics": True})
    args = [str(sc), "--out", str(tmp_path / "runs"), "--backend", "numpy_ref", "--json"]
    r = runner.invoke(_app(run_command), args)
    assert r.exit_code == 0, r.output
    summary = json.loads(r.output[r.output.index("{") :])
    run_dir = Path(summary["run_dir"])
    assert summary["samples"] == {"mu.rx0": 20_000}
    m = read_manifest(run_dir)
    assert m.backend == "numpy_ref"  # the override is part of the resolved scenario
    assert m.page_size == {"mu.rx0": 512}
    assert "ap.tx0->mu.rx0" in m.links
    assert m.links["ap.tx0->mu.rx0"].narrowband is True
    resolved = read_document(run_dir / "scenario.resolved.yaml")
    assert resolved["numerics"]["backend"] == "numpy_ref"
    for name in ("scenario.resolved.yaml", "events.jsonl", "summary.json"):
        assert (run_dir / name).is_file(), name
    if importlib.util.find_spec("sremu.record.sigmf") is not None:
        from sremu.record.sigmf import list_recordings

        recs = {x.stream: x for x in list_recordings(run_dir)}
        assert recs["mu.rx0"].data.shape == (20_000,)
    else:
        iq = np.fromfile(run_dir / "iq" / "mu.rx0.sigmf-data", dtype="<c8")
        assert iq.shape == (20_000,)


def test_cli_run_events_and_applied_configs_for_replay(tmp_path: Path, repo_root: Path) -> None:
    """events.jsonl carries one CommandRecord per command; applied_configs.jsonl carries the
    applied physical state (states_b64, int16 LE) replay needs."""
    from fractions import Fraction

    from sremu.commands.run import RunDirSink
    from sremu.contracts import CommandTime
    from sremu.engine.emulator import Emulator
    from sremu.engine.streams import ToneSource
    from tests.runtime._engine_support import config_request

    eng = Emulator.from_scenario(
        repo_root / "scenarios" / "quickstart.yaml",
        backend="numpy_ref",
        always_active=True,
        traffic={"ap.tx0": ToneSource(Fraction(10**6))},
    )
    sink = RunDirSink(tmp_path, iq=False, events=True, metrics=True, rates={}, fcs={})
    eng.add_sink(sink)
    n = eng.commands.specs["ris0"].n_controls
    st = np.zeros(n, np.int16)
    st[::5] = 1
    eng.schedule(config_request("c1", "ris0", st, CommandTime.at(Fraction(1, 1000))))
    eng.run(Fraction(2, 1000))
    sink.close()
    recs = [
        CommandRecord.from_dict(json.loads(x))
        for x in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [r.cmd_id for r in recs] == ["c1"]
    assert recs[0].status == "applied"
    cfg = json.loads((tmp_path / "applied_configs.jsonl").read_text().splitlines()[0])
    state = np.frombuffer(base64.b64decode(cfg["states_b64"]), dtype="<i2")
    assert np.array_equal(state, st)
    assert cfg["config_hash"] == recs[0].config_hash


def test_cli_run_serve_needs_wp8(tmp_path: Path, repo_root: Path) -> None:
    if importlib.util.find_spec("sremu.transport.control_server") is not None:
        pytest.skip("WP8 transport is present")
    sc = _scenario(tmp_path, repo_root)
    r = runner.invoke(_app(run_command), [str(sc), "--serve", "tcp://127.0.0.1:5600"])
    assert r.exit_code == 3


def test_cli_run_builtin_controllers_need_wp9(tmp_path: Path, repo_root: Path) -> None:
    if importlib.util.find_spec("sremu.control.builtin") is not None:
        pytest.skip("WP9 controllers are present")
    sc = repo_root / "scenarios" / "lab_like_A.yaml"
    r = runner.invoke(_app(run_command), [str(sc), "--out", str(tmp_path / "runs")])
    assert r.exit_code == 3
    assert "WP9" in r.output


def test_cli_batch_summary_json(tmp_path: Path, repo_root: Path) -> None:
    sc = _scenario(tmp_path, repo_root)
    out = tmp_path / "b.json"
    r = runner.invoke(_app(batch_command), [str(sc), "--json", "--rate", "1e6", "--out", str(out)])
    assert r.exit_code == 0, r.output
    s = json.loads(r.output)
    assert s["weights"] == "samples"
    assert s["pieces"] == 1
    h2 = s["mean_h2_db"]["ap.tx0->mu.rx0"]
    p = s["mean_rx_power_dbm"]["mu.rx0"]
    assert p == pytest.approx(h2, abs=1e-9)  # unit digital power at 0 dBm full scale
    table = json.loads(out.read_text(encoding="utf-8"))
    assert table["w"] == [20_000.0]
