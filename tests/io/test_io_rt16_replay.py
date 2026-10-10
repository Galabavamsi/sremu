"""RT-16: replay of recorded TX + events.jsonl reproduces RX bitwise (DESIGN.md 4.12, 10.3).

A recorded session exercises every recording path: two TX streams (timed bursts, an
untimed in-burst continuation, an untimed burst start placed by the engine, gaps), two RX
streams (one with a timed start and a gap), and RIS commands that are on time, element
subsets, superseded, late (apply_on_arrival), rejected (late, reject) and counterfactual
(``set_enabled``), plus a marker. The replay must reproduce every recorded RX sample bit
for bit and every command at its recorded ``k_applied``. Teeth: a dropped command, a
shifted ``t_applied`` and a corrupted TX sample each give exactly the mismatches predicted
by the command semantics of section 4.6.

The engine is ``FakeEngine`` (identity channel times an element-dependent RIS gain). With
the WP6 engine (when ``sremu.engine.emulator`` exists) the replay is checked for a
scenario-generated TX, client TX (timed and untimed sends), in-process controller commands
(with and without their requests), ``Device.inproc`` overrides and a paced run with a late
command, each with teeth.
"""

from __future__ import annotations

import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TimeSpec, TxMetadata
from sremu.cli import build_app
from sremu.errors import ErrorCode
from sremu.record.eventlog import read_events
from sremu.record.replay import ReplayError, replay
from sremu.record.sigmf import read_recording, validate_collection
from tests.io import _wp7_support as sup

pytestmark = pytest.mark.runtime

FC = 3.5e9
F = Fraction(1_000_000)
FACTORY = "tests.io._wp7_support:replay_engine"


def _ts(k):
    return TimeSpec.from_ticks(k, F)


def _data(rng, n):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def _read(rx, n, chunks=(1, 7, 1000, 333)):
    out = np.zeros(n, np.complex64)
    got, i = 0, 0
    md = RxMetadata()
    while got < n:
        k = rx.recv(out[got : got + chunks[i % len(chunks)]], md, timeout=0.0)
        assert k > 0, md.error_code
        got += k
        i += 1
    return out


def _session(run: Path, rng) -> dict[str, object]:
    """Record the session described in the module docstring; returns the cmd ids."""
    dev = Device.from_engine(sup.make_engine(), record=run)
    ap = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    bs = dev.get_tx_stream(StreamArgs("bs.tx0", FC, "1e6"))
    mu = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    ue = dev.get_rx_stream(StreamArgs("ue.rx0", FC, "1e6"))
    ris = dev.ris("ris0")
    ids = {"m1": dev.marker("start", time_spec=0).cmd_id}
    ids["c1"] = ris.set_config(states=[1, 0, 1, 0, 1, 0, 1, 0], time_spec=_ts(1500)).cmd_id
    ids["c2"] = ris.set_config(states=[1, 1], elements=(0, 1), time_spec=_ts(2200)).cmd_id
    ids["c3"] = ris.set_config(states=np.ones(8), time_spec=_ts(4000)).cmd_id
    ids["c4"] = ris.set_config(states=np.zeros(8), time_spec=_ts(4000)).cmd_id  # wins at t
    ids["c5"] = ris.set_enabled(False, time_spec=_ts(4800)).cmd_id
    ids["c6"] = ris.set_enabled(True, time_spec=_ts(5200)).cmd_id
    # ap: [0, 3000) timed, [3000, 4000) untimed in-burst, [4000, 4500) untimed + EOB
    ap.send(_data(rng, 3000), TxMetadata(has_time_spec=True, time_spec=_ts(0)))
    ap.send(_data(rng, 1000), TxMetadata())
    ap.send(_data(rng, 500), TxMetadata(end_of_burst=True))
    # bs: [1000, 3000) timed burst
    bs.send(_data(rng, 2000), TxMetadata(True, _ts(1000), True, True))
    mu.issue_stream_cmd(StreamCmd("num_done", num_samps=6000))
    _read(mu, 3500)
    ue.issue_stream_cmd(StreamCmd("num_done", num_samps=2800, stream_now=False, time_spec=_ts(200)))
    _read(ue, 2800)
    s7 = np.array([0, 1, 1, 0, 0, 1, 1, 0], np.int16)
    r7 = ris.set_config(states=s7, time_spec=_ts(1000))  # late: moved to sample 3500
    r8 = ris.set_config(states=np.ones(8), time_spec=_ts(500), late_policy="reject")
    assert (r7.status, r8.status, r8.error) == ("late_applied", "rejected", ErrorCode.LATE_COMMAND)
    ids["c7"], ids["c8"] = r7.cmd_id, r8.cmd_id
    # bs: untimed burst start, placed by the engine at the frontier (3500)
    bs.send(_data(rng, 300), TxMetadata(start_of_burst=True, end_of_burst=True))
    bs.send(_data(rng, 1000), TxMetadata(True, _ts(5000), True, True))
    _read(mu, 2500)
    ue.issue_stream_cmd(
        StreamCmd("num_done", num_samps=2500, stream_now=False, time_spec=_ts(3500))
    )
    _read(ue, 2500)
    dev.close()
    return ids


@pytest.fixture
def recorded(tmp_path, rng):
    run = tmp_path / "run"
    ids = _session(run, rng)
    return run, ids


def test_rt16_recording_shape(recorded):
    run, ids = recorded
    assert validate_collection(run) == []
    caps = {
        s: [
            (c.global_index, c.length)
            for c in read_recording(run / "iq" / f"{s}.sigmf-meta").captures
        ]
        for s in ("ap.tx0", "bs.tx0", "mu.rx0", "ue.rx0")
    }
    assert caps == {
        "ap.tx0": [(0, 4500)],
        "bs.tx0": [(1000, 2000), (3500, 300), (5000, 1000)],
        "mu.rx0": [(0, 6000)],
        "ue.rx0": [(200, 2800), (3500, 2500)],
    }
    final = {e.record.cmd_id: e.record for e in read_events(run)}
    assert final[ids["c3"]].status == "superseded"
    assert final[ids["c7"]].status == "late_applied"
    assert final[ids["c7"]].k_applied == {"mu.rx0": 3500, "ue.rx0": 3500}
    assert final[ids["c7"]].lateness_samples == {"mu.rx0": 2500, "ue.rx0": 2500}
    assert final[ids["c8"]].status == "rejected"
    assert all(final[ids[c]].status == "applied" for c in ("c1", "c2", "c4", "c5", "c6"))


def test_rt16_replay_is_bitwise(recorded, tmp_path):
    run, ids = recorded
    res = replay(run, engine_factory=sup.replay_engine, out=tmp_path / "again")
    assert res.ok, res.to_dict()
    assert {s: (c.n_samples, c.n_read, c.n_mismatch) for s, c in res.streams.items()} == {
        "mu.rx0": (6000, 6000, 0),
        "ue.rx0": (5300, 5300, 0),
    }
    assert all(c.bitwise_equal and c.max_abs_diff == 0.0 for c in res.streams.values())
    assert res.command_mismatches == []
    # every command except the rejected one is re-injected (superseded ones again lose)
    final = read_events(run)
    assert res.n_commands == len(final) - 1
    assert [st for _, st in res.skipped] == ["rejected"]
    # the replayed run is itself a valid recording with identical RX
    assert validate_collection(tmp_path / "again") == []
    for s in ("mu.rx0", "ue.rx0"):
        a = read_recording(run / "iq" / f"{s}.sigmf-meta")
        b = read_recording(tmp_path / "again" / "iq" / f"{s}.sigmf-meta")
        (cb,) = b.captures  # the replay reads each RX span in one piece (gap included)
        for c in a.captures:
            seg = b.data[c.global_index - cb.global_index : c.global_end - cb.global_index]
            want = a.data[c.sample_start : c.sample_start + c.length]
            np.testing.assert_array_equal(seg.view(np.uint32), want.view(np.uint32))
    again = {e.record.cmd_id: e.record for e in read_events(tmp_path / "again")}
    assert again[ids["c3"]].status == "superseded"
    assert again[ids["c7"]].status == "applied"  # re-injected on time
    assert again[ids["c7"]].k_applied == {"mu.rx0": 3500, "ue.rx0": 3500}


def _rewrite_events(run: Path, fn) -> None:
    path = run / "events.jsonl"
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    out = [y for x in lines if (y := fn(x)) is not None]
    path.write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")


def test_rt16_teeth_dropped_command(recorded):
    run, ids = recorded
    _rewrite_events(run, lambda x: None if x["cmd_id"] == ids["c1"] else x)
    res = replay(run, engine_factory=sup.replay_engine)
    assert not res.ok
    # without c1 the configuration differs on [1500, 3500) (c7 rewrites every cell at 3500)
    assert res.streams["mu.rx0"].n_mismatch == 2000
    assert res.streams["ue.rx0"].n_mismatch == 1500  # recorded part [1500, 3000)
    assert res.command_mismatches == []


def test_rt16_teeth_shifted_t_applied(recorded):
    run, ids = recorded

    def shift(x):
        if x["cmd_id"] == ids["c4"]:
            x["t_applied"] = "4001/1000000"
        return x

    _rewrite_events(run, shift)
    res = replay(run, engine_factory=sup.replay_engine)
    assert not res.ok
    # c3 (all ones) now holds for exactly one sample, 4000, before c4 at 4001
    assert res.streams["mu.rx0"].n_mismatch == 1
    assert res.streams["ue.rx0"].n_mismatch == 1
    assert any("k_applied 4001 != recorded 4000" in m for m in res.command_mismatches)


def test_rt16_teeth_corrupted_tx_sample(recorded):
    run, _ = recorded
    data = run / "iq" / "ap.tx0.sigmf-data"
    raw = bytearray(data.read_bytes())
    raw[100 * 8] ^= 0x01  # flip one bit of I at absolute sample 100
    data.write_bytes(bytes(raw))
    res = replay(run, engine_factory=sup.replay_engine)
    assert res.streams["mu.rx0"].n_mismatch == 1  # mu recorded sample 100
    assert res.streams["ue.rx0"].n_mismatch == 0  # ue starts at 200
    assert not res.ok


def test_rt16_missing_request_payload_is_an_error(recorded):
    run, ids = recorded

    def strip(x):
        if x["cmd_id"] == ids["c2"]:
            x["request"] = None
        return x

    _rewrite_events(run, strip)
    with pytest.raises(ReplayError, match="no request"):
        replay(run, engine_factory=sup.replay_engine)


def test_rt16_cli_exit_codes(recorded, tmp_path):
    run, _ = recorded
    app = build_app()
    runner = CliRunner()
    ok = runner.invoke(app, ["replay", str(run), "--engine", FACTORY])
    assert ok.exit_code == 0, ok.output
    assert "mu.rx0: 6000/6000 samples, 0 differ" in ok.output
    assert "replay: OK" in ok.output
    js = runner.invoke(app, ["replay", str(run), "--engine", FACTORY, "--json"])
    assert js.exit_code == 0
    assert json.loads(js.output)["ok"] is True
    data = run / "iq" / "bs.tx0.sigmf-data"
    raw = bytearray(data.read_bytes())
    raw[-1] ^= 0x80
    data.write_bytes(bytes(raw))
    bad = runner.invoke(app, ["replay", str(run), "--engine", FACTORY])
    assert bad.exit_code == 1
    assert "replay: MISMATCH" in bad.output
    empty = tmp_path / "empty"
    empty.mkdir()
    assert runner.invoke(app, ["replay", str(empty), "--engine", FACTORY]).exit_code == 2


needs_wp6 = pytest.mark.skipif(
    importlib.util.find_spec("sremu.engine.emulator") is None,
    reason="WP6 engine (sremu.engine.emulator) not available yet",
)
N_WP6 = 20_000
K_SW = 7_000


@pytest.fixture(scope="module")
def wp6_run(tmp_path_factory):
    """quickstart.yaml through Device.inproc (WP6 engine, numba_cpu, thermal noise) with a
    RIS switch to all-ones at 7 ms, recorded. ap.tx0 is driven by the scenario's in-process
    tone generator (not opened by the client), so the replay regenerates it."""
    import shutil

    from sremu.scenario.load import load_scenario

    root = Path(__file__).resolve().parents[2]
    run = tmp_path_factory.mktemp("wp6") / "qs"
    dev = Device.inproc(load_scenario(root / "scenarios" / "quickstart.yaml"), record=run)
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
    r = dev.ris("ris0").set_config(states=np.ones(256, np.int16), time_spec=TimeSpec(0, 7e-3))
    assert r.k_req == {"mu.rx0": K_SW}  # tau_RIS->rx of 5 m is far below one sample at 1 MS/s
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=N_WP6))
    _read(rx, N_WP6)
    dev.close()
    yield run
    shutil.rmtree(run, ignore_errors=True)


@needs_wp6
def test_rt16_replay_bitwise_with_wp6_engine(wp6_run):
    assert validate_collection(wp6_run) == []
    (ev,) = read_events(wp6_run)
    assert (ev.record.status, ev.record.k_applied) == ("applied", {"mu.rx0": K_SW})
    man = json.loads((wp6_run / "manifest.json").read_text(encoding="utf-8"))
    assert (man["scenario_name"], man["mode"], man["backend"]) == (
        "quickstart",
        "lockstep",
        "numba_cpu",
    )
    assert man["page_size"] == {"mu.rx0": 512}
    assert "ap.tx0->mu.rx0" in man["links"]
    res = replay(wp6_run)  # default factory: WP6 engine from scenario.resolved.yaml
    assert res.ok, res.to_dict()
    (cmp,) = res.streams.values()
    assert cmp.n_samples >= N_WP6
    assert cmp.n_mismatch == 0
    # the CLI uses the same default factory; the report shows the manifest
    out = CliRunner().invoke(build_app(), ["replay", str(wp6_run)])
    assert out.exit_code == 0, out.output
    assert "replay: OK" in out.output
    from sremu.record.report import render_report

    html = render_report(wp6_run)
    assert "<th>scenario_name</th><td>quickstart</td>" in html
    assert "<th>backend</th><td>numba_cpu</td>" in html


@needs_wp6
def test_rt16_teeth_wp6_dropped_command(wp6_run, tmp_path):
    import shutil

    run = tmp_path / "copy"
    shutil.copytree(wp6_run, run)
    _rewrite_events(run, lambda x: None)
    res = replay(run)
    (cmp,) = res.streams.values()
    # every recorded sample from the switch on differs; none before it
    assert cmp.n_mismatch == cmp.n_samples - K_SW
    assert not res.ok


def _controller_session(run: Path, rng, n: int = 5000) -> None:
    """A FakeEngine run whose RIS commands all come from an in-process barrier controller
    (so the recorder never sees their requests)."""
    dev = Device.from_engine(sup.make_engine(), record=run)
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    dev.register_controller(sup.ToggleController())
    tx.send(_data(rng, n), TxMetadata(True, _ts(0), True, True))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=n))
    _read(rx, n)
    dev.close()


def test_rt16_replay_of_controller_commands_uses_the_applied_config(tmp_path, rng):
    run = tmp_path / "ctrl"
    _controller_session(run, rng)
    evs = read_events(run)
    applied = [e for e in evs if e.record.status == "applied"]
    # barriers at 0, 1, ..., 4 ms; actions 0.1 ms later
    assert [e.record.k_applied["mu.rx0"] for e in applied] == [100, 1100, 2100, 3100, 4100]
    assert all(e.request is None and e.applied is not None for e in applied)
    assert [int(e.applied.state[0]) for e in applied] == [0, 1, 0, 1, 0]
    res = replay(run, engine_factory=sup.replay_engine)
    assert res.ok, res.to_dict()
    assert res.n_commands == 5
    # teeth: without the applied configuration the commands cannot be re-injected
    _rewrite_events(run, lambda x: {k: v for k, v in x.items() if k != "applied_config"})
    with pytest.raises(ReplayError, match="no applied configuration"):
        replay(run, engine_factory=sup.replay_engine)


@needs_wp6
def test_rt16_replay_of_controller_commands_with_wp6_engine(tmp_path):
    """Commands of an in-process controller (issued inside the engine) replay bitwise:
    from the exact requests when the engine reports them (``command_log``, CCR WP7-1), and
    from the applied configuration the WP6 sink reports when they are missing."""
    from sremu.scenario.load import load_scenario

    root = Path(__file__).resolve().parents[2]
    run = tmp_path / "ctrl6"
    n = 8000
    dev = Device.inproc(load_scenario(root / "scenarios" / "quickstart.yaml"), record=run)
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
    dev.register_controller(sup.ToggleController(n_el=256, period_s="2e-3"))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=n))
    _read(rx, n)
    dev.close()
    applied = [e for e in read_events(run) if e.record.status in ("applied", "late_applied")]
    # barriers at 0, 2, 4, 6 ms, actions 0.1 ms later (tau far below one sample)
    assert [e.record.k_applied["mu.rx0"] for e in applied] == [100, 2100, 4100, 6100]
    assert [int(e.applied.state.sum()) for e in applied] == [0, 256, 0, 256]
    assert {e.record.src for e in applied} == {"toggle"}  # issued by the controller (8.5 src)
    res = replay(run)
    assert res.ok, res.to_dict()
    (cmp,) = res.streams.values()
    assert cmp.n_samples >= n
    assert res.n_commands >= 4
    # without the requests, the applied configurations alone still replay bitwise
    _rewrite_events(run, lambda x: {**x, "request": None})
    assert all(e.request is None for e in read_events(run))
    res = replay(run)
    assert res.ok, res.to_dict()
    # teeth: drop the toggle to all-ones at 2.1 ms; recorded samples [2100, 4100) differ
    _rewrite_events(run, lambda x: None if x["k_applied"].get("mu.rx0") == 2100 else x)
    res = replay(run)
    assert res.streams["mu.rx0"].n_mismatch == 2000
    assert not res.ok


def _client_tx_scenario(tmp_path: Path) -> Path:
    """quickstart.yaml without its in-process tone generator: ap.tx0 carries client TX."""
    root = Path(__file__).resolve().parents[2]
    text = (root / "scenarios" / "quickstart.yaml").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("traffic:")]
    assert len(lines) == len(text.splitlines()) - 1
    path = tmp_path / "qs_client_tx.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@needs_wp6
def test_rt16_replay_reinjects_client_tx_with_wp6_engine(tmp_path, rng):
    """RT-16 through the WP6 engine with the TX samples coming from the client (timed burst
    start, untimed in-burst continuation, untimed EOB): the recorded ap.tx0 stream is
    re-injected and the replay reproduces every recorded RX sample bit for bit. Teeth: one
    flipped bit of TX sample 5000 changes exactly one RX sample, because a 10 kHz occupied
    band at 1 MS/s uses integer-only delay (B <= 0.0318 f_s, section 4.5)."""
    run = tmp_path / "run"
    dev = Device.inproc(_client_tx_scenario(tmp_path), record=run)
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", 3.5e9, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
    x = _data(rng, 9000) * np.complex64(0.1)
    tx.send(x[:3000], TxMetadata(True, _ts(0), True, False))
    tx.send(x[3000:6000], TxMetadata())
    tx.send(x[6000:], TxMetadata(end_of_burst=True))
    r = dev.ris("ris0").set_config(states=np.ones(256, np.int16), time_spec=_ts(4000))
    assert r.k_req == {"mu.rx0": 4000}
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=9000))
    y = _read(rx, 9000)
    dev.close()
    t = read_recording(run / "iq" / "ap.tx0.sigmf-meta")
    assert [(c.global_index, c.length) for c in t.captures] == [(0, 9000)]
    np.testing.assert_array_equal(t.data.view(np.uint32), x.view(np.uint32))
    rr = read_recording(run / "iq" / "mu.rx0.sigmf-meta")
    np.testing.assert_array_equal(rr.data[:9000].view(np.uint32), y.view(np.uint32))
    res = replay(run)
    assert res.ok, res.to_dict()
    assert res.streams["mu.rx0"].n_samples == rr.data.size
    data = run / "iq" / "ap.tx0.sigmf-data"
    raw = bytearray(data.read_bytes())
    raw[5000 * 8] ^= 0x10
    data.write_bytes(bytes(raw))
    res = replay(run)
    assert res.streams["mu.rx0"].n_mismatch == 1
    assert not res.ok


@needs_wp6
def test_rt16_inproc_overrides_are_recorded_and_replayed(tmp_path):
    """Device.inproc overrides describe the run in its directory: scenario fields
    (backend, page_target_s) go into scenario.resolved.yaml and the manifest; an engine
    option outside the scenario (noise=False) goes to engine_overrides.json and a manifest
    assumption, and the default replay applies it; ``epoch_datetime`` reaches the SigMF
    ``core:datetime`` (R2). Teeth: without engine_overrides.json the replay adds thermal
    noise, so every recorded sample differs."""
    import yaml

    from sremu.record.replay import ENGINE_OVERRIDES_NAME

    root = Path(__file__).resolve().parents[2]
    run = tmp_path / "ov"
    dev = Device.inproc(
        root / "scenarios" / "quickstart.yaml",
        backend="numpy_ref",
        page_target_s="1/4000",
        noise=False,
        record=run,
        epoch_datetime="2026-10-10T12:00:00Z",
    )
    assert (dev.engine.cfg.backend, dev.engine.cfg.noise) == ("numpy_ref", False)
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
    assert rx.handle.page_size == 256  # 2^round(log2(1e6/4000))
    dev.ris("ris0").set_config(states=np.ones(256, np.int16), time_spec=_ts(1000))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=3000))
    _read(rx, 3000)
    dev.close()
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert (man["backend"], man["mode"], man["page_size"]) == (
        "numpy_ref",
        "lockstep",
        {"mu.rx0": 256},
    )
    assert any("noise=False" in a and ENGINE_OVERRIDES_NAME in a for a in man["assumptions"])
    sc = yaml.safe_load((run / "scenario.resolved.yaml").read_text(encoding="utf-8"))
    assert sc["numerics"]["backend"] == "numpy_ref"
    assert str(sc["numerics"]["page_target_s"]) in ("1/4000", "0.00025")
    rx_meta = read_recording(run / "iq" / "mu.rx0.sigmf-meta").meta
    assert rx_meta["captures"][0]["core:global_index"] == 0
    assert rx_meta["captures"][0]["core:datetime"] == "2026-10-10T12:00:00.000000000Z"  # R2
    ov = json.loads((run / ENGINE_OVERRIDES_NAME).read_text(encoding="utf-8"))
    assert ov == {"format": 1, "overrides": {"noise": False}}
    res = replay(run)
    assert res.ok, res.to_dict()
    (run / ENGINE_OVERRIDES_NAME).unlink()
    res = replay(run)
    (cmp,) = res.streams.values()
    assert cmp.n_mismatch == cmp.n_samples > 0


@needs_wp6
def test_rt16_paced_run_replays_offline_with_wp6_engine(tmp_path, rng):
    """Section 4.12: a paced run is reproduced offline (lockstep replay). Paced mode with a
    deterministic FakeClock and async coupling; client TX; an on-time command at sample
    1000; then, after 4096 samples were computed, a command for sample 2000 that
    ``apply_on_arrival`` moves to the frontier (k_applied 4096, lateness 2096). The replay
    re-injects it at its recorded ``t_applied`` and is bitwise. Teeth: re-injected at its
    requested time instead, exactly the recorded samples [2000, 4096) differ."""
    from sremu.engine.pacer import FakeClock

    run = tmp_path / "paced"
    root = Path(__file__).resolve().parents[2]
    dev = Device.inproc(
        root / "scenarios" / "quickstart.yaml",
        mode="paced",
        coupling="async",
        clock=FakeClock(),
        traffic={},
        record=run,
    )
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", 3.5e9, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
    n = 9216  # 18 pages of 512
    tx.send(_data(rng, n) * np.complex64(0.1), TxMetadata(True, _ts(0), True, True))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=n))
    on_time = dev.ris("ris0").set_config(states=np.ones(256, np.int16), time_spec=_ts(1000))
    dev.engine.run_paced(Fraction(4096) / F)
    late = dev.ris("ris0").set_config(states=np.zeros(256, np.int16), time_spec=_ts(2000))
    assert late.status == "late_applied"
    dev.engine.run_paced(Fraction(n) / F)
    _read(rx, n)
    dev.close()
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert (man["mode"], man["coupling"]) == ("paced", "async")
    final = {e.record.cmd_id: e.record for e in read_events(run)}
    assert final[on_time.cmd_id].k_applied == {"mu.rx0": 1000}
    assert final[late.cmd_id].k_applied == {"mu.rx0": 4096}
    assert final[late.cmd_id].lateness_samples == {"mu.rx0": 2096}
    res = replay(run)
    assert res.ok, res.to_dict()
    assert res.streams["mu.rx0"].n_samples >= n

    def at_request(x):
        if x["cmd_id"] == late.cmd_id:
            x["t_applied"] = x["t_req"]
        return x

    _rewrite_events(run, at_request)
    res = replay(run)
    assert res.streams["mu.rx0"].n_mismatch == 4096 - 2000
    assert not res.ok


def test_rt16_engine_overrides_file_roundtrip_and_staleness(tmp_path):
    """engine_overrides.json: plain values (exact ones as "p/q") are written and read back,
    non-plain values are reported unsaved, clock/sinks are ignored, an empty set removes a
    stale file of an earlier run, and an unknown format is refused."""
    from sremu.record.replay import (
        ENGINE_OVERRIDES_NAME,
        read_engine_overrides,
        write_engine_overrides,
    )

    saved, unsaved = write_engine_overrides(
        tmp_path,
        {
            "noise": False,
            "render_tiles": "model",
            "rx_queue_pages": 3,
            "x_s": Fraction(1, 3),
            "traffic": {"ap.tx0": object()},
            "clock": object(),
        },
    )
    assert saved == {"noise": False, "render_tiles": "model", "rx_queue_pages": 3, "x_s": "1/3"}
    assert unsaved == ["traffic"]
    assert read_engine_overrides(tmp_path) == saved
    assert write_engine_overrides(tmp_path, {"clock": object()}) == ({}, [])
    assert not (tmp_path / ENGINE_OVERRIDES_NAME).exists()
    assert read_engine_overrides(tmp_path) == {}
    (tmp_path / ENGINE_OVERRIDES_NAME).write_text('{"format": 2}', encoding="utf-8")
    with pytest.raises(ReplayError, match="format"):
        read_engine_overrides(tmp_path)


@needs_wp6
def test_rt16_physical_replay_for_column_control_and_dead_cells(tmp_path):
    """Commands logged without their request (as ``sremu run`` logs controller commands)
    carry the applied per-cell configuration. On a panel with column control (16 controls
    for 256 cells) and dead cells (state -1) that configuration is not a valid
    ``ris.set_config`` payload, so the replay applies it through the engine's
    ``schedule_physical`` (section 4.12) and is bitwise. Teeth: dropping the toggle to
    all-ones at 2.1 ms changes exactly the recorded samples [2100, 4100)."""
    root = Path(__file__).resolve().parents[2]
    text = (root / "scenarios" / "quickstart.yaml").read_text(encoding="utf-8")
    extra = "    control: {granularity: column}\n    failures: {dead_rate: 0.2}\n"
    text = text.replace("    initial: {fill: 0}\n", "    initial: {fill: 0}\n" + extra)
    assert extra in text
    scen = tmp_path / "qs_column_dead.yaml"
    scen.write_text(text, encoding="utf-8")
    run = tmp_path / "phys"
    dev = Device.inproc(scen, record=run)
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
    dev.register_controller(sup.ToggleController(n_el=16, period_s="2e-3"))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=8000))
    _read(rx, 8000)
    dev.close()
    _rewrite_events(run, lambda x: {**x, "request": None})
    applied = [e for e in read_events(run) if e.record.status == "applied"]
    assert [e.record.k_applied["mu.rx0"] for e in applied] == [100, 2100, 4100, 6100]
    assert all(e.request is None and e.applied.state.shape == (256,) for e in applied)
    assert all(int(e.applied.state.min()) == -1 for e in applied)  # dead cells present
    res = replay(run)
    assert res.ok, res.to_dict()
    assert res.command_mismatches == []
    _rewrite_events(run, lambda x: None if x["k_applied"].get("mu.rx0") == 2100 else x)
    res = replay(run)
    assert res.streams["mu.rx0"].n_mismatch == 2000
    assert not res.ok
