"""FakeEngine property test: barrier semantics (section 4.9) for random rational rates.

The expected switch sample is the R3 closed form ``floor((m*T_c + L_min)*f + 1/2)``,
computed here without ``sremu.timebase``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import (
    Action,
    LatencyModel,
    ObservationSpec,
    RisCapabilities,
    StreamCmd,
    StreamSpec,
    TxMetadata,
)
from sremu.errors import ErrorCode
from sremu.testing import FakeEngine

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

pytestmark = pytest.mark.unit

CAP = RisCapabilities(
    n_elements=4,
    layout=(4, 1),
    control_granularity="element",
    states=("s0", "s1", "s2"),
    gamma_source="analytic:test",
    operating_band_hz=(0.5e9, 4.0e9),
    max_bandwidth_hz=2e7,
)


def gain(cfgs) -> complex:
    cfg = cfgs["ris0"]
    return complex(1 + int(cfg.state[0])) if cfg.enabled else 0.5 + 0j


def states(first: int) -> np.ndarray:
    return np.array([first, 0, 0, 0], dtype=np.int16)


class _Toggler:
    """Barrier controller that toggles element 0 at t_b + L_min (latency = L_min)."""

    name = "toggler"

    def __init__(self, period_s: Fraction, lmin_s: Fraction) -> None:
        self.period_s = period_s
        self.min_latency_s = lmin_s
        self.spec = ObservationSpec(window_s=Fraction(1, 10_000))
        self.latency = LatencyModel(fixed_s=lmin_s)
        self.n = 0

    def setup(self, caps, rng) -> None:
        pass

    def decide(self, obs):
        self.n += 1
        return [Action("ris0", states=states(self.n % 2))]


@settings(max_examples=40, deadline=None)
@given(
    fa=st.builds(Fraction, st.integers(2000, 20_000), st.integers(1, 7)),
    fb=st.builds(Fraction, st.integers(2000, 20_000), st.integers(1, 7)),
    period=st.builds(Fraction, st.integers(1, 50), st.integers(1000, 5000)),
    lmin=st.builds(Fraction, st.integers(0, 40), st.integers(1000, 9000)),
    seed=st.integers(0, 2**32 - 1),
)
def test_fake_barrier_lateness_zero_for_random_rational_rates(fa, fb, period, lmin, seed):
    """Section 4.9 semantics on the fake (the property WP6 must meet as I8/RT-08): with a
    barrier-registered controller acting at t_b + L_min, every RX gets the switch of barrier
    m at exactly to_ticks(m*T_c + L_min, f_r), lateness is 0, and chunking does not matter."""
    eng = FakeEngine({"ris0": CAP}, gain_fn=gain)
    pairs = []
    for tag, f, fc in (("a", fa, 1.0e9), ("b", fb, 2.0e9)):
        tx = eng.open_stream(StreamSpec(f"tx{tag}.tx0", "tx", fc, f))
        rx = eng.open_stream(StreamSpec(f"rx{tag}.rx0", "rx", fc, f))
        pairs.append((tx, rx, f))
    ctrl = _Toggler(period, lmin)
    eng.register_controller(ctrl)
    dur = min(10 * period, Fraction(1, 10))
    totals = {}
    for tx, rx, f in pairs:
        n = -((-dur.numerator * f.numerator) // (dur.denominator * f.denominator))  # ceil
        totals[rx.id] = n
        eng.submit_tx(tx, np.ones(n + 8, np.complex64), TxMetadata(end_of_burst=True))
        eng.stream_cmd(rx, StreamCmd("start_cont"))
    gen = np.random.Generator(np.random.Philox(key=seed))
    got = {rx.id: [] for _, rx, _ in pairs}
    count_ = dict.fromkeys(got, 0)
    while any(count_[i] < totals[i] for i in got):
        _, rx, _ = pairs[int(gen.integers(0, 2))]
        left = totals[rx.id] - count_[rx.id]
        if left == 0:
            continue
        y, md = eng.read_rx(rx, min(left, int(gen.integers(1, 400))), 0.0)
        assert md.error_code == ErrorCode.NONE
        got[rx.id].append(y)
        count_[rx.id] += y.size
    for _, rx, f in pairs:
        y = np.concatenate(got[rx.id])
        k = np.arange(y.size)
        g = np.ones(y.size)
        m = 0
        while True:
            ks = math.floor((m * period + lmin) * f + Fraction(1, 2))  # R3 closed form
            if ks >= y.size:
                break
            g[k >= ks] = 1 + (m + 1) % 2
            m += 1
        np.testing.assert_array_equal(y, g.astype(np.complex64))
    names = {rx.spec.stream: totals[rx.id] for _, rx, _ in pairs}
    # barrier m runs once some RX reads from its limit on, i.e. iff a limit lies inside
    # that RX's read range (reads stop exactly at limits)
    n_barriers = 0
    while any(
        math.floor((n_barriers * period + lmin) * f + Fraction(1, 2)) < totals[rx.id]
        for _, rx, f in pairs
    ):
        n_barriers += 1
    assert len(eng.records) == n_barriers == ctrl.n
    for r in eng.records:
        done = all(k < names[s] for s, k in r.k_applied.items())
        assert r.status == ("applied" if done else "scheduled"), r
    assert all(v == 0 for r in eng.records for v in r.lateness_samples.values())
    assert not eng.nondeterministic_inputs
