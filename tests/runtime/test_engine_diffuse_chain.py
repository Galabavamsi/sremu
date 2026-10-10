"""The diffuse step of the RX chain (DESIGN.md sections 5.11, 6.3) through the engine.

WP2 physics model of ``scenarios/quickstart.yaml`` with ``environment.diffuse`` enabled
(WP3 mechanism), noise off, a DC tone with integer delay: the difference between the runs
with and without diffuse is exactly ``sum_tap sqrt(P_tap) g_tap(t_n) x_W`` with ``g`` the
WP3 sum of sinusoids at the absolute sample times (additive, never a multiplier of the
deterministic channel), and it is chunk invariant.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.engine.emulator import Emulator
from sremu.engine.streams import ToneSource
from sremu.scenario.load import load_scenario
from tests.runtime._engine_support import captured

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
N = 4000
XW = complex(np.complex64(np.sqrt(1e-3)))


def _run(diffuse: bool) -> tuple[np.ndarray, Emulator]:
    pytest.importorskip("sremu.physics.diffuse")
    res = load_scenario(
        Path(__file__).resolve().parents[2] / "scenarios" / "quickstart.yaml", warn=False
    )
    d = res.scenario.model_dump(mode="python")
    d["nodes"][0]["streams"][0]["occupied_bw_hz"] = 1.0e3  # CW: integer-only delay
    d["traffic"] = []
    if diffuse:
        d["environment"]["diffuse"] = {
            "k_factor_db": 3.0,
            "max_doppler_hz": 50.0,
            "n_sinusoids": 48,
            "taps": [
                {"delay_s": 0.0, "rel_power_db": 0.0},
                {"delay_s": 2.0e-6, "rel_power_db": -3.0},
            ],
        }
    eng = Emulator.from_scenario(
        d,
        base_dir=res.base_dir,
        backend="numpy_ref",
        noise=False,
        always_active=True,
        traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)},
    )
    cap = eng.capture_accumulator("mu.rx0")
    eng.run(Fraction(N) / F)
    return captured(cap, 0, N), eng


def test_engine_diffuse_is_additive_sos_term() -> None:
    y1, eng = _run(True)
    y0, _ = _run(False)
    assert eng.builder.diffuse is not None
    snap = eng.model.snapshot(Fraction(0))
    terms = eng.builder.diffuse.terms(eng.model, snap)["ap.tx0->mu.rx0"]
    assert len(terms) == 2
    expect = np.zeros(N, dtype=np.complex128)
    for t in terms:
        d = round(t.delay_s * float(F))  # integer delay; the DC tone starts at sample 0
        g = t.gain_samples(0, N, F)
        expect[d:] += g[d:] * XW
    diff = y1 - y0
    err = np.max(np.abs(diff - expect)) / np.max(np.abs(expect))
    assert err <= 1e-9
    # teeth: the term is time varying (Doppler 50 Hz over 4 ms) and not negligible
    assert np.ptp(np.abs(diff[100:])) > 0.05 * np.max(np.abs(diff))


def test_engine_diffuse_numba_matches_numpy() -> None:
    pytest.importorskip("sremu.physics.diffuse")
    res = load_scenario(
        Path(__file__).resolve().parents[2] / "scenarios" / "quickstart.yaml", warn=False
    )
    out = []
    for backend in ("numpy_ref", "numba_cpu"):
        d = res.scenario.model_dump(mode="python")
        d["environment"]["diffuse"] = {"k_factor_db": 0.0, "max_doppler_hz": 20.0}
        eng = Emulator.from_scenario(
            d,
            base_dir=res.base_dir,
            backend=backend,
            always_active=True,
            traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        )
        cap = eng.capture_accumulator("mu.rx0")
        eng.run(Fraction(2000) / F)
        out.append(captured(cap, 0, 2000))
    assert np.max(np.abs(out[0] - out[1])) <= 1e-9 * np.max(np.abs(out[0]))
