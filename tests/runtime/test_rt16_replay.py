"""RT-16 (engine part): recorded TX + applied commands reproduce RX bitwise (section 4.12).

The recording layer (WP7, ``sremu replay RUN_DIR``) persists the TX streams and
``events.jsonl``; the engine provides what replay needs: every applied command's record
with the applied physical configuration (``EngineSink.on_record``), TX submissions
(``EngineSink.on_tx``) and :meth:`Emulator.schedule_physical`, which re-injects a command at
its recorded ``t_applied`` as an on-time command. This test runs the original (a barrier
controller with jittered latency, a late unregistered command moved by
``apply_on_arrival``, a transition, noise and CFO) and the replay in one process.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.contracts import CommandTime, LatencyModel, StreamSpec, TransitionSpec, TxMetadata
from sremu.errors import NondeterminismWarning
from sremu.timebase import TimeSpec
from tests.runtime._engine_support import (
    Collect,
    ScriptedController,
    alternate_script,
    config_request,
    lab_like_fake,
    make_engine,
    waveform,
)

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
N = 9000
CFO = {"ap.tx0": 77.0, "rx2.rx0": -12.25}


def _engine(sink: Collect):
    return make_engine(
        lab_like_fake(moving=True),
        backend="numba_cpu",
        noise=True,
        bw=8.0e5,
        cfo=CFO,
        render_tiles="auto",
        always_active=True,
        sinks=[sink],
        seed=3,
    )


def test_rt16_replay_recorded_tx_and_commands_bitwise() -> None:
    rec = Collect()
    eng = _engine(rec)
    n = eng.commands.specs["ris0"].n_controls
    htx = eng.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
    x = waveform(N + 200)
    for k in range(0, x.size, 1500):  # timed chunks, EOB at the end
        part = x[k : k + 1500]
        eng.submit_tx(
            htx, part, TxMetadata(True, TimeSpec.from_ticks(k, F), k == 0, k + 1500 >= x.size)
        )
    ctrl = ScriptedController(
        "alt",
        "1/1000",
        "1/10000",
        alternate_script("ris0", n),
        latency=LatencyModel(Fraction(1, 10_000), "uniform:0,4e-4"),
    )
    eng.register_controller(ctrl)
    eng.run(Fraction(2500, 10**6))
    with pytest.warns(NondeterminismWarning):
        r = eng.schedule(
            config_request(
                "late",
                "ris0",
                np.zeros(n, np.int16),
                CommandTime.at(Fraction(1, 1000)),
                transition=TransitionSpec("linear", Fraction(5, 100_000)),
            )
        )
    assert r.status == "late_applied"
    eng.run(Fraction(N) / F)
    applied = [
        (rr, cfg)
        for rr, cfg in zip(rec.records, rec.configs, strict=True)
        if rr.status in ("applied", "late_applied") and rr.op.startswith("ris.")
    ]
    assert len(applied) >= 8
    assert any(rr.status == "late_applied" for rr, _ in applied)

    # ------------------------------------------------------------------ replay
    rep = Collect()
    eng2 = _engine(rep)
    htx2 = eng2.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
    chunks = rec.tx["ap.tx0"]
    for i, (k0, d) in enumerate(chunks):
        eng2.submit_tx(
            htx2, d, TxMetadata(True, TimeSpec.from_ticks(k0, F), i == 0, i == len(chunks) - 1)
        )
    for rr, cfg in sorted(applied, key=lambda p: (p[0].t_applied, p[0].seq)):
        assert cfg is not None
        out = eng2.schedule_physical(
            rr.panel, rr.t_applied, cfg, transition=rr.transition, cmd_id=rr.cmd_id
        )
        assert out.ok
        assert out.status == "scheduled"  # on time
    eng2.run(Fraction(N) / F)
    assert not eng2.nondeterministic_inputs
    for r in ("rx1.rx0", "rx2.rx0"):
        a = rec.rx_array(r, 0, N)
        b = rep.rx_array(r, 0, N)
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), r
    # the replayed commands land on the recorded samples
    by_id = {rr.cmd_id: rr for rr in eng2.records if rr.op == "ris.set_config"}
    for rr, _ in applied:
        if rr.op == "ris.set_config":
            assert by_id[rr.cmd_id].k_applied == rr.k_applied


def test_rt16_cli_run_then_wp7_replay_bitwise(tmp_path: Path, repo_root: Path) -> None:
    """RT-16 end to end: ``sremu run`` (WP6) records a run with a WP9 built-in controller
    (keyed jittered latency); WP7's ``replay`` re-injects TX and commands at their recorded
    ``t_applied`` and the RX is bitwise equal."""
    pytest.importorskip("sremu.record.replay")
    pytest.importorskip("sremu.control.builtin")
    import json

    import typer
    from typer.testing import CliRunner

    from sremu.commands.run import command
    from sremu.record.replay import replay
    from sremu.scenario.load import dump_yaml, read_document

    d = read_document(repo_root / "scenarios" / "quickstart.yaml")
    d["run"] = {"duration_s": "0.02"}
    d["record"] = {"iq": True, "events": True, "metrics": True}
    d["controllers"] = [
        {
            "name": "urap",
            "kind": "builtin",
            "impl": "Urap",
            "period_s": "1e-3",
            "min_latency_s": "1e-4",
            "latency": {"fixed_s": "1e-4", "jitter": "uniform:0,3e-4"},
            "params": {"p": 0.5},
        }
    ]
    sc = tmp_path / "s.yaml"
    sc.write_text(dump_yaml(d), encoding="utf-8")
    app = typer.Typer()
    app.command()(command)
    r = CliRunner().invoke(app, [str(sc), "--out", str(tmp_path / "runs"), "--json"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.output[r.output.index("{") :])
    assert out["records"] == 20
    res = replay(out["run_dir"])
    assert res.ok, res.to_dict()
    assert res.n_commands == 20
    assert res.command_mismatches == []
    assert res.streams["mu.rx0"].bitwise_equal
    assert res.streams["mu.rx0"].n_read == 20_000
