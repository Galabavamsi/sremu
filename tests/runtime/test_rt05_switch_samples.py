"""RT-05 (I3): a RIS switch at RIS time t_sw takes effect on RX r at
``k_r = to_ticks(t_sw + tau_RIS->r(t_sw), f_r)``, per render tile when tiled. Exact.
DESIGN.md sections 4.2 (I3), 4.6, 10.3.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.contracts import CommandTime, PanelConfig, PathKind
from sremu.engine.streams import ToneSource
from sremu.testing.scenes import near_far_rx
from sremu.timebase import to_ticks
from tests.runtime._engine_support import captured, config_request, make_engine

pytestmark = pytest.mark.runtime

F = Fraction(23_040_000)


def _run(model, t_sw: Fraction | None, n: int, backend: str = "numpy_ref"):
    eng = make_engine(
        model,
        backend=backend,
        traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)},
        always_active=True,
        verify_interp=False,
    )
    caps = {r: eng.capture_accumulator(r) for r in ("rx1.rx0", "rx2.rx0")}
    if t_sw is not None:
        nc = eng.commands.specs["ris0"].n_controls
        rec = eng.schedule(
            config_request("sw", "ris0", np.ones(nc, np.int16), CommandTime.at(t_sw))
        )
        assert rec.ok
    eng.run(Fraction(n) / F)
    return eng, {r: captured(c, 0, n) for r, c in caps.items()}


def _first_diff(a: np.ndarray, b: np.ndarray) -> int:
    d = np.flatnonzero(a != b)
    assert d.size, "the switch changed nothing"
    return int(d[0])


def test_rt05_far_rx_2km_switches_154_samples_after_near_rx() -> None:
    """23.04 MS/s, far RX at 2 km: first affected sample on every RX equals
    to_ticks(t_sw + tau) (about 154 samples after the near RX)."""
    model = near_far_rx()
    t_sw = Fraction(1, 10_000)  # 100 us
    n = 3000
    eng, on = _run(model, t_sw, n)
    _, off = _run(model, None, n)
    ks = {}
    for r in ("rx1.rx0", "rx2.rx0"):
        taus = [model.ris_delay_s("ris0", r, t_sw, j) for j in range(3)]
        expect = min(to_ticks(t_sw + Fraction(tau), F) for tau in taus)
        ks[r] = _first_diff(on[r], off[r])
        assert ks[r] == expect
        rec = eng.records[0]
        assert rec.k_applied[r] == to_ticks(t_sw + Fraction(model.ris_delay_s("ris0", r, t_sw)), F)
        assert rec.lateness_samples[r] == 0
    assert ks["rx1.rx0"] == 2304  # t_sw*f = 2304 exactly; tau_near*f = 0.09 samples
    gap = ks["rx2.rx0"] - ks["rx1.rx0"]
    assert gap in (153, 154)  # 2 km at 23.04 MS/s is 153.7 samples
    assert eng.records[0].status == "applied"


def test_rt05_per_tile_switch_samples_and_amplitudes() -> None:
    """t_sw chosen so the 3 render tiles round to different samples at rx1: the centre
    tile switches one sample before the outer tiles; the RX value in between equals the
    per-path mixture computed from combine() of the two configurations."""
    model = near_far_rx()
    taus = [model.ris_delay_s("ris0", "rx1.rx0", Fraction(0), j) for j in range(3)]
    lo, hi = min(taus), max(taus)
    assert hi - lo > 0  # outer tiles are farther than the centre tile
    K = 4000
    t_sw = (Fraction(K) + Fraction(1, 2)) / F - Fraction((lo + hi) / 2)
    kj = [to_ticks(t_sw + Fraction(tau), F) for tau in taus]
    assert sorted(set(kj)) == [K, K + 1]
    n = 4200
    _eng, on = _run(model, t_sw, n)
    y = on["rx1.rx0"]
    # expected values: the TX ring holds x_W = sqrt(1 mW) as complex64 (section 3.5), integer
    # delay
    snap = model.snapshot(Fraction(0))
    nc = 120
    old = model.combine(snap, {"ris0": PanelConfig("ris0", np.zeros(nc, np.int16))})[
        "ap.tx0->rx1.rx0"
    ]
    new = model.combine(snap, {"ris0": PanelConfig("ris0", np.ones(nc, np.int16))})[
        "ap.tx0->rx1.rx0"
    ]
    fc = 3.5e9
    ph = np.exp(-2j * np.pi * np.mod(fc * old.delay_s, 1.0))
    xw = complex(np.complex64(np.sqrt(1e-3)))
    tile_of_path = np.where(old.kind == PathKind.RIS_TILE, old.tile % 3, -1)
    for k, switched in (
        (K - 1, set()),
        (K, {j for j in range(3) if kj[j] == K}),
        (K + 1, {0, 1, 2}),
    ):
        amp = np.where(np.isin(tile_of_path, list(switched)), new.amp, old.amp)
        expect = complex(np.sum(amp * ph)) * xw
        assert abs(y[k] - expect) <= 1e-12 * abs(expect), (k, y[k], expect)


def test_rt05_real_model_every_rx_lab_like() -> None:
    """WP2 physics model (scenarios/lab_like_A.yaml, 1 MS/s): the record's k_applied is
    to_ticks(t_sw + tau_RIS->r) on both receivers and the output changes there first."""
    from sremu.engine.emulator import Emulator

    f = Fraction(10**6)
    outs = {}
    for with_cmd in (True, False):
        eng = Emulator.from_scenario(
            Path(__file__).resolve().parents[2] / "scenarios" / "lab_like_A.yaml",
            backend="numpy_ref",
            noise=False,
            always_active=True,
            traffic={"ap.tx0": ToneSource(f, 0.0, 1.0)},
        )
        caps = {r: eng.capture_accumulator(r) for r in ("rx1.rx0", "rx2.rx0")}
        t_sw = Fraction(1234567, 10**9)
        if with_cmd:
            nc = eng.commands.specs["ris0"].n_controls
            assert eng.schedule(
                config_request("sw", "ris0", np.ones(nc, np.int16), CommandTime.at(t_sw))
            ).ok
        eng.run(Fraction(3, 1000))
        outs[with_cmd] = ({r: captured(c, 0, 3000) for r, c in caps.items()}, eng)
    eng = outs[True][1]
    rec = eng.records[0]
    for r in ("rx1.rx0", "rx2.rx0"):
        tau = eng.model.ris_delay_s("ris0", r, t_sw)
        k = to_ticks(t_sw + Fraction(tau), f)
        assert rec.k_applied[r] == k
        assert _first_diff(outs[True][0][r], outs[False][0][r]) == k
