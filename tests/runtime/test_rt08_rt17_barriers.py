"""RT-08 (I8) barrier determinism and RT-17 lookahead violation (DESIGN.md 4.9, 10.3).

RT-08: in-process and remote barrier controllers sleeping 0 vs 50 ms give bitwise-identical
RX; property test: lateness == 0 on every RX for random rational rates, periods and L_min.
RT-17: an action earlier than ``t_b + L_min`` raises ``LOOKAHEAD_VIOLATION`` (in-process)
or is rejected with it (barrier-registered source over the control path).
"""

from __future__ import annotations

import threading
import time
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sremu.contracts import (
    Action,
    CommandTime,
    LatencyModel,
    Observation,
    ObservationSpec,
)
from sremu.engine.streams import ToneSource
from sremu.errors import ErrorCode, LookaheadViolation
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import stream
from sremu.timebase import to_ticks
from tests.runtime._engine_support import (
    ScriptedController,
    alternate_script,
    captured,
    config_request,
    lab_like_fake,
    make_engine,
)

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
RXS = ("rx1.rx0", "rx2.rx0")


def _engine(**kw: object):
    eng = make_engine(
        lab_like_fake(moving=True),
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        noise=True,
        always_active=True,
        render_tiles="auto",
        **kw,
    )
    caps = {r: eng.capture_accumulator(r) for r in RXS}
    return eng, caps


def _n(eng) -> int:
    return eng.commands.specs["ris0"].n_controls


def _jitter_latency() -> LatencyModel:
    return LatencyModel(Fraction(2, 10_000), "uniform:0,3e-4")


def _run_inprocess(sleep_s: float, backend: str, n: int = 8000) -> dict[str, np.ndarray]:
    eng, caps = _engine(backend=backend)
    ctrl = ScriptedController(
        "alt",
        "1/1000",
        "2/10000",
        alternate_script("ris0", _n(eng)),
        latency=_jitter_latency(),
        sleep_s=sleep_s,
    )
    eng.register_controller(ctrl)
    eng.run(Fraction(n) / F)
    recs = [r for r in eng.records if r.src == "alt"]
    assert len(recs) >= 7
    assert all(r.status == "applied" for r in recs)
    assert all(v == 0 for r in recs for v in r.lateness_samples.values())
    return {r: captured(c, 0, n) for r, c in caps.items()}


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_rt08_inprocess_controller_sleep_0_vs_50ms_bitwise(backend: str) -> None:
    a = _run_inprocess(0.0, backend)
    b = _run_inprocess(0.05, backend, n=8000)
    for r in RXS:
        assert np.array_equal(a[r].view(np.uint64), b[r].view(np.uint64))


class _ThreadedRemote:
    """A remote-controller stand-in: the 'remote side' runs in its own thread, sleeps,
    sends its commands through ``Emulator.schedule`` and then acks (section 8.5 protocol)."""

    def __init__(self, eng, name: str, n: int, sleep_s: float) -> None:
        self.eng = eng
        self.name = name
        self.period_s = Fraction(1, 1000)
        self.min_latency_s = Fraction(2, 10_000)
        self.spec = ObservationSpec(("rx_power_dbm",), Fraction(1, 1000))
        self.latency = LatencyModel(Fraction(2, 10_000))
        self.n = n
        self.sleep_s = sleep_s
        self._acked: dict[Fraction, threading.Event] = {}
        self.m = 0

    def send_barrier(self, obs: Observation) -> None:
        ev = threading.Event()
        self._acked[obs.t_b] = ev
        m = self.m
        self.m += 1

        def remote() -> None:
            time.sleep(self.sleep_s)
            s = np.full(self.n, m % 2, dtype=np.int16)
            at = obs.t_b + Fraction(3, 10_000)
            r = self.eng.schedule(
                config_request(f"{self.name}:{m}", "ris0", s, CommandTime.at(at), src=self.name)
            )
            assert r.ok, r
            ev.set()

        threading.Thread(target=remote, daemon=True).start()

    def wait_ack(self, t_b: Fraction, timeout_s: float) -> bool:
        return self._acked[t_b].wait(timeout_s)


def _run_remote(sleep_s: float, n: int = 6000) -> dict[str, np.ndarray]:
    eng, caps = _engine(backend="numpy_ref")
    eng.register_controller(_ThreadedRemote(eng, "rc", _n(eng), sleep_s))
    eng.run(Fraction(n) / F)
    recs = [r for r in eng.records if r.src == "rc"]
    assert len(recs) >= 5
    assert all(r.status == "applied" for r in recs)
    assert all(v == 0 for r in recs for v in r.lateness_samples.values())
    assert not eng.nondeterministic_inputs
    return {r: captured(c, 0, n) for r, c in caps.items()}


def test_rt08_remote_controller_sleep_0_vs_50ms_bitwise() -> None:
    a = _run_remote(0.0)
    b = _run_remote(0.05)
    for r in RXS:
        assert np.array_equal(a[r].view(np.uint64), b[r].view(np.uint64))


def test_rt08_observation_uses_samples_before_t_b_only() -> None:
    """The observation window ends at k_b = to_ticks(t_b): changing what happens after t_b
    (a command at t_b + L_min) cannot change the observation of that barrier."""
    eng, _caps = _engine(backend="numpy_ref")
    ctrl = ScriptedController(
        "obs", "1/1000", "2/10000", lambda o, m: [], fields=("rx_power_dbm", "iq_window")
    )
    eng.register_controller(ctrl)
    eng.run(Fraction(5, 1000))
    for o in ctrl.observations:
        for r in RXS:
            assert o.k_b[r] == to_ticks(o.t_b, F)
            assert o.iq_window is not None
            assert o.iq_window[r].shape[0] == min(o.k_b[r], 1000)


def _two_rate_scene(rate_a: Fraction, rate_b: Fraction) -> FakePropagation:
    nodes = [
        FakeNode("ta", (1.0, -1.0, 0.0), streams=(stream("ta.tx0", "tx", 1.0e9, rate_a),)),
        FakeNode("ra", (1.5, 0.5, 0.0), streams=(stream("ra.rx0", "rx", 1.0e9, rate_a),)),
        FakeNode("tb", (1.0, -1.2, 0.0), streams=(stream("tb.tx0", "tx", 3.0e9, rate_b),)),
        FakeNode("rb", (300.0, 7.0, 0.0), streams=(stream("rb.rx0", "rx", 3.0e9, rate_b),)),
    ]
    panel = FakePanel("ris0", (0, 0, 0), (1, 0, 0), (0, 1, 0), 4, 4, 0.05, 0.05)
    return FakePropagation(nodes, [panel], horizon_s=1)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    na=st.integers(3000, 30000),
    nb=st.integers(3000, 30000),
    per_num=st.integers(1, 40),
    per_den=st.integers(9000, 30000),
    lmin_num=st.integers(0, 15),
    lat_extra=st.integers(0, 9),
)
def test_rt08_property_lateness_zero_random_rational_rates(
    na: int, nb: int, per_num: int, per_den: int, lmin_num: int, lat_extra: int
) -> None:
    """I8: lockstep, barrier-registered controller, every action at t >= t_b + L_min:
    lateness == 0 on every RX for random rational rates (two rate groups), periods, L_min
    and latencies."""
    rate_a, rate_b = Fraction(na, 3), Fraction(nb, 7)
    model = _two_rate_scene(rate_a, rate_b)
    eng = make_engine(
        model,
        traffic={"ta.tx0": ToneSource(rate_a, 0.0, 1.0), "tb.tx0": ToneSource(rate_b, 0.0, 1.0)},
        always_active=True,
        verify_interp=False,
    )
    period = Fraction(per_num, per_den)
    lmin = Fraction(lmin_num, 7919)
    n = eng.commands.specs["ris0"].n_controls

    def script(obs: Observation, m: int) -> list[Action]:
        return [Action("ris0", states=np.full(n, m % 2, dtype=np.int16))]

    ctrl = ScriptedController(
        "p", period, lmin, script, latency=LatencyModel(lmin + Fraction(lat_extra, 104729))
    )
    eng.register_controller(ctrl)
    eng.run(min(Fraction(1, 100), 12 * period))
    recs = [r for r in eng.records if r.src == "p"]
    assert recs
    for r in recs:
        assert r.status in ("applied", "scheduled")
        assert all(v == 0 for v in r.lateness_samples.values()), r


# --------------------------------------------------------------------------- RT-17


def test_rt17_inprocess_action_before_lookahead_raises() -> None:
    eng, _ = _engine(backend="numpy_ref")
    n = _n(eng)

    def bad(obs: Observation, m: int) -> list[Action]:
        return [Action("ris0", states=np.ones(n, np.int16), at_s=obs.t_b + Fraction(1, 10_000))]

    eng.register_controller(ScriptedController("bad", "1/1000", "2/10000", bad))
    with pytest.raises(LookaheadViolation) as ei:
        eng.run(Fraction(3, 1000))
    assert ei.value.code == ErrorCode.LOOKAHEAD_VIOLATION


def test_rt17_registered_source_timed_command_rejected() -> None:
    """A barrier-registered source (e.g. a remote controller) that sends a timed command
    earlier than t_b + L_min of its latest barrier gets LOOKAHEAD_VIOLATION."""
    eng, _ = _engine(backend="numpy_ref")
    ctrl = ScriptedController("ctl", "1/1000", "2/10000", lambda o, m: [])
    eng.register_controller(ctrl)
    eng.run(Fraction(22, 10_000))  # barriers at 0, 1 and 2 ms ran; frontier = 2200
    assert ctrl.observations[-1].t_b == Fraction(2, 1000)
    assert eng.frontier() == dict.fromkeys(RXS, 2200)
    r = eng.schedule(
        config_request(
            "x", "ris0", np.ones(_n(eng), np.int16), CommandTime.at(Fraction(21, 10_000)), src="ctl"
        )
    )
    assert not r.ok
    assert r.error == ErrorCode.LOOKAHEAD_VIOLATION
    assert eng.records[-1].status == "rejected"
    ok = eng.schedule(
        config_request(
            "y", "ris0", np.ones(_n(eng), np.int16), CommandTime.at(Fraction(22, 10_000)), src="ctl"
        )
    )
    assert ok.ok  # exactly t_b + L_min is allowed
