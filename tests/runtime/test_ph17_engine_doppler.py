"""PH-17 through the engine, and its teeth via the legacy ``block_constant`` Doppler.

T9 geometry of ``pm_checks.py`` (``sremu.testing.scenes.doppler_t9``): 3.5 GHz, TX at the
origin, RX from (50, 0, 0) approaching at 30 m/s, 10 kS/s, DC tone (integer delay).

* Engine (auto T_u, chord delay + exact phase): phase equals ``-2 pi f_c d(t)/c`` to 1e-6 rad
  over 1 s and the instantaneous frequency is 350.242 +- 0.001 Hz (DESIGN.md 10.2 PH-17).
* Teeth (D7): ``block_constant`` staircase geometry over 2 ms blocks, measured over the T9
  window (0.2 s at 10 kS/s): mean instantaneous frequency -148.3 +- 1 Hz, failing PH-17.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.engine.streams import ToneSource
from sremu.errors import NonPhysicalError
from sremu.legacy.block_constant import block_constant_emulator
from sremu.testing.scenes import doppler_t9
from sremu.units import C0
from tests.oracles.tables import ORACLE, PH17_T9_BLOCK_S, PH17_T9_FS, PH17_T9_WINDOW_S
from tests.runtime._engine_support import captured, make_engine, stream_cfgs

pytestmark = pytest.mark.runtime

FC = 3.5e9
FS = Fraction(10_000)


def _inst_freq(y: np.ndarray, fs: float) -> np.ndarray:
    return np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * np.pi)


def test_ph17_engine_los_doppler_350p242_and_phase(backend_name: str) -> None:
    model = doppler_t9()
    eng = make_engine(
        model,
        backend=backend_name,
        traffic={"tx.tx0": ToneSource(FS, 0.0, 1.0)},
        always_active=True,
        render_tiles="auto",
    )
    assert eng.t_u is not None
    cap = eng.capture_accumulator("rx.rx0")
    n = 10_000
    eng.run(Fraction(1))
    y = captured(cap, 0, n)
    t = np.arange(n) / 1e4
    d = 50.0 - 30.0 * t
    cyc = FC * d / C0
    oracle = np.exp(-2j * np.pi * (cyc - np.rint(cyc)))
    assert np.max(np.abs(np.angle(y * np.conj(oracle)))) <= 1e-6
    fi = _inst_freq(y, 1e4)
    f_d = 30.0 * FC / C0
    ref = ORACLE["PH17_los_fd_hz"]
    assert f_d == pytest.approx(ref.value, abs=ref.tol)
    assert np.max(np.abs(fi - f_d)) <= ref.tol
    vi = eng.manifest_info()["verify_interp_max_rel"]
    assert vi is not None
    assert vi <= 2e-3  # within 2 eps_rel (section 4.4)


def test_ph17_teeth_engine_block_constant_minus148hz() -> None:
    model = doppler_t9()
    with pytest.raises(NonPhysicalError):
        block_constant_emulator(model, PH17_T9_BLOCK_S)
    eng = block_constant_emulator(
        model,
        PH17_T9_BLOCK_S,
        allow_nonphysical=True,
        streams=stream_cfgs(model),
        traffic={"tx.tx0": ToneSource(FS, 0.0, 1.0)},
    )
    cap = eng.capture_accumulator("rx.rx0")
    n = int(PH17_T9_WINDOW_S * PH17_T9_FS)
    eng.run(PH17_T9_WINDOW_S)
    y = captured(cap, 0, n)
    f_mean = float(_inst_freq(y, float(PH17_T9_FS)).mean())
    teeth = ORACLE["PH17_block_constant_hz"]
    assert f_mean == pytest.approx(teeth.value, abs=teeth.tol)
    assert abs(f_mean - 350.242) > 0.001  # fails the PH-17 check
