"""VA-4: with tau_SUI -> 0 and an in-process controller, the IQ engine equals the batch path
at 1e-9 (DESIGN.md section 4.11 agreement contract).

WP2 physics model (scenarios/lab_like_A.yaml, static), noise off, DC tone with integer delay
(a CW tone: B <= 0.0318 f_s), a barrier controller with zero latency that alternates the
configuration every 1 ms. IQ: time average of |y|^2 / |x_W|^2 over every RX sample (the
complex128 accumulator before the complex64 cast). Batch: ``run_batch`` with the same model,
controller semantics and RX sample-count weights.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.contracts import LatencyModel
from sremu.engine.batch import run_batch
from sremu.engine.emulator import Emulator
from sremu.engine.streams import ToneSource, stream_configs_from_scenario
from sremu.physics.propagation import from_scenario
from sremu.scenario.load import load_scenario
from tests.runtime._engine_support import ScriptedController, alternate_script, captured

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
N = 20_000
XW2 = abs(complex(np.complex64(np.sqrt(1e-3)))) ** 2


def _scenario() -> tuple[dict, Path]:
    res = load_scenario(
        Path(__file__).resolve().parents[2] / "scenarios" / "lab_like_A.yaml", warn=False
    )
    d = res.scenario.model_dump(mode="python")
    d["nodes"][0]["streams"][0]["occupied_bw_hz"] = 1.0e3  # CW tone: integer-only delay
    d["controllers"] = []
    d["traffic"] = []
    assert res.base_dir is not None
    return d, res.base_dir


def _ctrl(n: int) -> ScriptedController:
    return ScriptedController(
        "alt", "1/1000", "0", alternate_script("ris0", n), latency=LatencyModel(Fraction(0))
    )


def test_va4_tau_sui_zero_iq_equals_batch_1e_9() -> None:
    d, base = _scenario()
    eng = Emulator.from_scenario(
        d,
        base_dir=base,
        backend="numba_cpu",
        noise=False,
        always_active=True,
        traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)},
    )
    n = eng.commands.specs["ris0"].n_controls
    eng.register_controller(_ctrl(n))
    caps = {r: eng.capture_accumulator(r) for r in ("rx1.rx0", "rx2.rx0")}
    eng.run(Fraction(N) / F)
    assert all(v == 0 for r in eng.records for v in r.lateness_samples.values())
    assert sum(r.status == "applied" for r in eng.records) == 20
    iq = {r: float(np.mean(np.abs(captured(c, 0, N)) ** 2)) / XW2 for r, c in caps.items()}

    model = from_scenario(d, base_dir=base)
    res = run_batch(
        model,
        Fraction(N) / F,
        rate=F,
        controllers=[_ctrl(n)],
        streams=stream_configs_from_scenario(d),
    )
    assert res.weights == "samples"
    assert float(res.w.sum()) == N
    for r in ("rx1.rx0", "rx2.rx0"):
        b = res.mean_h2[f"ap.tx0->{r}"]
        assert abs(iq[r] - b) <= 1e-9 * b, (r, iq[r], b)
    # teeth: the two configurations differ, so the controller's timing matters
    assert np.ptp(res.h2["ap.tx0->rx1.rx0"]) > 0.05 * res.mean_h2["ap.tx0->rx1.rx0"]
    # tau_RIS->r is reported (neglected by the RIS-time convention of the batch path)
    assert 0 < res.tau_ris_s["ris0/rx1.rx0"] < 1e-8


def test_va4_batch_static_without_commands_is_one_piece() -> None:
    d, base = _scenario()
    model = from_scenario(d, base_dir=base)
    res = run_batch(model, Fraction(1, 100), streams=stream_configs_from_scenario(d))
    assert len(res.t) == 1  # static scene, nothing changes: one evaluation
    assert res.weights == "time"
    p = res.mean_rx_power_dbm("rx1.rx0")
    assert np.isfinite(p)
    assert p == pytest.approx(10 * np.log10(res.mean_h2["ap.tx0->rx1.rx0"]), abs=1e-9)
