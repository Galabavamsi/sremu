"""examples/quickstart runs end to end (WP7 deliverable; DESIGN.md 8.1).

Oracles: with the identity-channel FakeEngine and the example's toy gain (1 with every
cell in state 0, 0.5 otherwise) a unit tone reads 0 dB before the switch and
``20*log10(0.5) = -6.021`` dB after it, the switch lands at sample 50000 (RIS time 50 ms at
1 MS/s), and the recorded run renders and validates.
"""

from __future__ import annotations

import importlib.util
import math
import re
import runpy
from pathlib import Path

import pytest

from sremu.record.eventlog import read_events
from sremu.record.report import render_report
from sremu.record.sigmf import validate_collection

pytestmark = pytest.mark.io

QS = Path(__file__).resolve().parents[2] / "examples" / "quickstart" / "quickstart.py"


def _powers(out: str) -> tuple[float, float]:
    m = re.search(r"mean power before (-?[\d.]+) dB, after (-?[\d.]+) dB", out)
    assert m, out
    return float(m.group(1)), float(m.group(2))


def test_quickstart_fake_engine_with_recording(tmp_path, capsys):
    mod = runpy.run_path(str(QS))
    run = tmp_path / "qs"
    assert mod["main"](["--engine", "fake", "--record", str(run)]) == 0
    out = capsys.readouterr().out
    assert "received 100000 samples; switch at sample 50000" in out
    before, after = _powers(out)
    assert before == 0.0
    assert after == round(20 * math.log10(0.5), 3)
    assert validate_collection(run) == []
    (ev,) = read_events(run)
    assert (ev.record.status, ev.record.k_applied) == ("applied", {"mu.rx0": 50000})
    html = render_report(run)
    assert "<script" not in html.lower()
    assert ev.record.cmd_id in html


@pytest.mark.skipif(
    importlib.util.find_spec("sremu.engine.emulator") is None,
    reason="WP6 engine (sremu.engine.emulator) not available yet",
)
def test_quickstart_wp6_engine(tmp_path, capsys):
    """Through the WP6 engine the printed powers equal ``10 log10(|h|^2 + sigma^2)`` before
    and after the switch, with ``h = sum_p a_p exp(-j 2 pi f_c tau_p)`` of the scenario's
    propagation model (all cells in state 0, then 1) for the unit-amplitude 1 kHz tone
    (narrowband: 2 pi * 1 kHz * tau ~ 1e-4 rad) and ``sigma^2 = k_B T_0 F f_s`` (NF 7 dB,
    in mW: 0 dBm full scale, section 3.3). Tolerance 0.002 dB: 0.0005 dB print rounding plus
    the sampling error of a 50 000-sample mean at an SNR of about 49 dB (~1e-4 dB). The
    recorded run replays bitwise, with the client's tone re-injected (``traffic={}``)."""
    import numpy as np

    from sremu.contracts import PanelConfig
    from sremu.physics.propagation import from_scenario
    from sremu.record.replay import replay
    from sremu.record.sigmf import read_recording
    from sremu.scenario.load import load_scenario
    from sremu.units import K_B, T0

    mod = runpy.run_path(str(QS))
    run = tmp_path / "qs6"
    assert mod["main"](["--engine", "inproc", "--record", str(run)]) == 0
    out = capsys.readouterr().out
    assert "(inproc engine)" in out
    assert "received 100000 samples; switch at sample 50000" in out
    before, after = _powers(out)
    sc = load_scenario(QS.parents[2] / "scenarios" / "quickstart.yaml")
    model = from_scenario(sc.scenario.model_dump(mode="python"), base_dir=sc.base_dir)
    snap = model.snapshot(0)

    def h(state: int) -> complex:
        cfg = {"ris0": PanelConfig("ris0", np.full(256, state, np.int16))}
        lp = model.combine(snap, cfg)["ap.tx0->mu.rx0"]
        return complex(np.sum(lp.amp * np.exp(-2j * np.pi * 3.5e9 * lp.delay_s)))

    sigma2_mw = K_B * T0 * 10 ** (7 / 10) * 1e6 / 1e-3
    want = [10 * math.log10(abs(h(s)) ** 2 + sigma2_mw) for s in (0, 1)]
    assert abs(before - want[0]) <= 0.002, (before, want)
    assert abs(after - want[1]) <= 0.002, (after, want)
    assert abs((before - after) - (want[0] - want[1])) <= 0.002
    assert (run / "manifest.json").is_file()
    assert validate_collection(run) == []
    tx = read_recording(run / "iq" / "ap.tx0.sigmf-meta")
    assert [(c.global_index, c.length) for c in tx.captures] == [(0, 100_000)]
    res = replay(run)
    assert res.ok, res.to_dict()
