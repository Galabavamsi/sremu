"""Transitions composed by the engine (DESIGN.md 4.7; the engine side of RT-12).

Static SIM scene, DC tone, noise off, integer delay, so each RX sample is
``x_W * sum_p a_p(n) exp(-j 2 pi f_c tau_p)`` and the oracle is built from ``combine()`` of
pure configurations only (not from the engine's job builder):

* a linear 20 us ramp from all-0 to all-1 starting at k1 follows
  ``h_old + min((n - k1)/20, 1) (h_new - h_old)`` to 1e-9;
* a re-command at RIS time t1 + 8 us (g = 0.4) toward all-0 over 10 us starts from the
  instantaneous Gamma ``Gamma_old + 0.4 (Gamma_new - Gamma_old)`` (RIS-time freeze) and ramps
  from there;
* ``by_edge`` classes with a first-order profile compose by linearity.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import CommandTime, PanelConfig, TransitionSpec
from sremu.engine.streams import ToneSource
from sremu.timebase import to_ticks
from tests.runtime._engine_support import captured, config_request, lab_like_fake, make_engine

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
LINK = "ap.tx0->rx1.rx0"


def _h(model, states: np.ndarray) -> complex:
    snap = model.snapshot(Fraction(0))
    lp = model.combine(snap, {"ris0": PanelConfig("ris0", states.astype(np.int16))})[LINK]
    return complex(np.sum(lp.amp * np.exp(-2j * np.pi * np.mod(3.5e9 * lp.delay_s, 1.0))))


def _engine(backend: str):
    model = lab_like_fake(tiles=(1, 1))
    eng = make_engine(
        model, backend=backend, traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)}, always_active=True
    )
    return model, eng, eng.capture_accumulator("rx1.rx0")


XW = complex(np.complex64(np.sqrt(1e-3)))  # TX ring value of a full-scale DC tone


def test_rt12_engine_linear_ramp_and_recommand_from_instantaneous_gamma(backend_name: str) -> None:
    model, eng, cap = _engine(backend_name)
    n = eng.commands.specs["ris0"].n_controls
    zeros, ones = np.zeros(n, np.int16), np.ones(n, np.int16)
    t1 = Fraction(1, 1000)
    t2 = t1 + Fraction(8, 10**6)
    eng.schedule(
        config_request(
            "a",
            "ris0",
            ones,
            CommandTime.at(t1),
            transition=TransitionSpec("linear", Fraction(20, 10**6)),
        )
    )
    eng.schedule(
        config_request(
            "b",
            "ris0",
            zeros,
            CommandTime.at(t2),
            transition=TransitionSpec("linear", Fraction(10, 10**6)),
        )
    )
    eng.run(Fraction(2, 1000))
    y = captured(cap, 0, 2000) / XW
    h0, h1 = _h(model, zeros), _h(model, ones)
    tau = model.ris_delay_s("ris0", "rx1.rx0", t1)
    k1 = to_ticks(t1 + Fraction(tau), F)
    k2 = to_ticks(t2 + Fraction(model.ris_delay_s("ris0", "rx1.rx0", t2)), F)
    assert (k1, k2) == (1000, 1008)
    expect = np.full(2000, h0, dtype=np.complex128)
    for i in range(k1, k2):
        expect[i] = h0 + min((i - k1) / 20.0, 1.0) * (h1 - h0)
    h_start = h0 + 0.4 * (h1 - h0)  # instantaneous Gamma at RIS time t2 (g1 = 8/20)
    for i in range(k2, 2000):
        expect[i] = h_start + min((i - k2) / 10.0, 1.0) * (h0 - h_start)
    err = np.abs(y - expect) / np.abs(expect)
    assert err.max() <= 1e-9, (int(err.argmax()), float(err.max()))
    # the ramp is visible (teeth): mid-ramp differs from both endpoints by > 10 %
    assert abs(y[k1 + 5] - h0) > 0.1 * abs(h1 - h0)
    assert abs(y[k1 + 5] - h1) > 0.1 * abs(h1 - h0)
    recs = {r.cmd_id: r for r in eng.records}
    assert recs["a"].transition == {"model": "linear", "duration_s": "1/50000", "classes": "single"}


def test_rt12_engine_by_edge_first_order_linearity() -> None:
    """Two edge classes (0->1 and 1->0) with a first-order profile: each class moves with
    1 - exp(-x/tau); the result equals the per-class closed form."""
    model, eng, cap = _engine("numpy_ref")
    n = eng.commands.specs["ris0"].n_controls
    start = np.zeros(n, np.int16)
    start[::2] = 1
    target = 1 - start
    eng.schedule(config_request("s", "ris0", start, CommandTime.at(Fraction(0))))
    t1 = Fraction(5, 10**4)
    eng.schedule(
        config_request(
            "e",
            "ris0",
            target,
            CommandTime.at(t1),
            transition=TransitionSpec("first_order", Fraction(7, 10**6), "by_edge"),
        )
    )
    eng.run(Fraction(1, 1000))
    y = captured(cap, 0, 1000) / XW
    h_s, h_t = _h(model, start), _h(model, target)
    k1 = to_ticks(t1 + Fraction(model.ris_delay_s("ris0", "rx1.rx0", t1)), F)
    m = np.arange(1000) - k1
    g = np.where(m >= 0, -np.expm1(-np.maximum(m, 0) / 7.0), 0.0)
    expect = h_s + g * (h_t - h_s)  # both classes share the profile: linear in Gamma
    err = np.abs(y[1:] - expect[1:]) / np.abs(expect[1:])
    assert err.max() <= 1e-9
