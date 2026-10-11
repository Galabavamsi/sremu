"""VA-5 (DESIGN.md section 11, E-B1): sample exactness of an intra-frame RIS switch.

A QPSK packet (64 + 512 symbols, 4 samples/symbol, 1 MS/s) goes to RX1 of the lab-like
geometry; the RIS switches from RX1's to RX2's exact optimum at 0.25 T_p after the packet
start (step transition, physical noise). The first RX1 sample that differs from the
no-switch run (same keyed noise) must equal ``to_ticks(t_sw + tau_RIS->RX1)`` and the
engine's ``k_applied``; every earlier sample (so every symbol wholly before the switch) is
bitwise identical. A linear transition starts with g(0) = 0, so its first differing sample is
``k_applied + 1``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from experiments import eb

from sremu.contracts import CommandRequest, CommandTime
from sremu.engine.streams import QpskPacketSource
from sremu.timebase import to_ticks

pytestmark = pytest.mark.runtime

START = 3000


@pytest.fixture(scope="module")
def setup():
    dsg = eb.design("lab", "present")
    n = START + eb.BODY + 600
    scen = eb.lab_scenario(
        "lab",
        "present",
        full_scale_dbm=dsg.tx_dbm,
        initial_states=dsg.s1,
        duration_s=Fraction(n) / eb.FS,
    )
    res = eb.resolve(scen)
    src = QpskPacketSource(
        eb.TX, int(res.scenario.seed), eb.SPS, eb.N_PRE, eb.N_PAY, guard=10**9, start=START
    )
    ref, _, _ = eb.run_iq(res, src, [], n)
    return dsg, res, src, n, ref


@pytest.mark.parametrize(("transition", "offset"), [("step", 0), ("linear20us", 1)])
def test_va5_first_affected_sample(setup, transition: str, offset: int) -> None:
    dsg, res, src, n, ref = setup
    t_sw = Fraction(START) / eb.FS + Fraction(1, 4) * eb.T_P
    req = CommandRequest(
        "va5",
        "ris.set_config",
        "va5",
        eb.PANEL,
        CommandTime.at(t_sw),
        states=dsg.s2,
        transition=eb.TRANSITIONS[transition],
    )
    sink, eng, _ = eb.run_iq(res, src, [req], n)
    (rec,) = [r for r in eng.records if r.cmd_id == "va5"]
    tau = float(eng.model.ris_delay_s(eb.PANEL, eb.RX1, t_sw))
    k_expect = to_ticks(t_sw + Fraction(repr(tau)), eb.FS)
    assert rec.k_applied[eb.RX1] == k_expect
    y, y0 = sink.buf[eb.RX1], ref.buf[eb.RX1]
    diff = np.nonzero(y.view(np.uint64) != y0.view(np.uint64))[0]
    assert diff.size > 0
    assert int(diff[0]) == k_expect + offset
    assert np.array_equal(y[: diff[0]].view(np.uint64), y0[: diff[0]].view(np.uint64))
    # the switch changes RX1's channel (not a no-op)
    assert abs(dsg.h[eb.RX1]["s2"] - dsg.h[eb.RX1]["s1"]) > 0.1 * abs(dsg.h[eb.RX1]["s1"])
