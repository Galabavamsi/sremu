"""FakeEngine: exact time base, lockstep commitment and command / barrier semantics.

Expected sample indices are hand-computed closed forms of R3 (nearest, ties up) and the
section 4.6 / 4.9 rules with tau_RIS->r = 0.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import (
    Action,
    CommandRequest,
    CommandTime,
    LatencyModel,
    ObservationSpec,
    PanelConfig,
    RisCapabilities,
    StreamCmd,
    StreamSpec,
    TxMetadata,
)
from sremu.errors import (
    AsyncCode,
    ErrorCode,
    LookaheadViolation,
    RateMismatchError,
    SremuTimeout,
)
from sremu.testing import FakeEngine
from sremu.timebase import TimeSpec

pytestmark = pytest.mark.unit

CAP = RisCapabilities(
    n_elements=4,
    layout=(4, 1),
    control_granularity="element",
    states=("s0", "s1", "s2"),
    gamma_source="analytic:test",
    operating_band_hz=(3.0e9, 4.0e9),
    max_bandwidth_hz=2e7,
)


def gain(cfgs) -> complex:
    cfg = cfgs["ris0"]
    return complex(1 + int(cfg.state[0])) if cfg.enabled else 0.5 + 0j


def spec(sid: str, direction: str, rate: str = "1e6", fc: float = 3.5e9) -> StreamSpec:
    return StreamSpec(sid, direction, fc, rate)  # type: ignore[arg-type]


def states(first: int) -> np.ndarray:
    return np.array([first, 0, 0, 0], dtype=np.int16)


def read_all(eng: FakeEngine, h, total: int, chunk: int):
    out, mds = [], []
    got = 0
    while got < total:
        y, md = eng.read_rx(h, min(chunk, total - got), 0.0)
        assert md.error_code == ErrorCode.NONE, md
        assert y.dtype == np.complex64
        out.append(y)
        mds.append(md)
        got += y.size
    return np.concatenate(out), mds


def engine_with_pair(rate: str = "1e6", fc: float = 3.5e9, tag: str = "a", eng=None):
    eng = eng or FakeEngine({"ris0": CAP}, gain_fn=gain)
    tx = eng.open_stream(spec(f"tx{tag}.tx0", "tx", rate, fc))
    rx = eng.open_stream(spec(f"rx{tag}.rx0", "rx", rate, fc))
    return eng, tx, rx


# --------------------------------------------------------------------------- data plane


def test_timed_burst_lands_at_to_ticks_and_metadata_is_exact():
    eng, tx, rx = engine_with_pair()
    x = (np.arange(100) + 1j).astype(np.complex64)
    md = TxMetadata(
        has_time_spec=True, time_spec=TimeSpec(0, 1e-3), start_of_burst=True, end_of_burst=True
    )
    assert eng.submit_tx(tx, x, md) == 100
    eng.stream_cmd(rx, StreamCmd("num_done", 2000, stream_now=False, time_spec=TimeSpec(0, 0.0)))
    y, mds = read_all(eng, rx, 2000, 700)
    assert y.size == 2000
    np.testing.assert_array_equal(y[1000:1100], x)  # to_ticks(1e-3, 1e6) = 1000
    assert not y[:1000].any() and not y[1100:].any()
    firsts = [m.first_sample for m in mds]
    assert firsts == [0, 700, 1400]
    assert [m.time_spec for m in mds] == [TimeSpec.from_ticks(k, 10**6) for k in firsts]
    assert mds[0].start_of_burst and not mds[1].start_of_burst
    assert mds[-1].end_of_burst and not mds[0].end_of_burst
    ack = eng.recv_async(tx, 0.0)
    assert ack is not None and ack.event_code == AsyncCode.BURST_ACK and ack.sample_index == 1100
    assert eng.recv_async(tx, 0.0) is None


def test_lockstep_commitment_without_eob_and_idle_until():
    eng, tx, rx = engine_with_pair()
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    eng.submit_tx(tx, np.ones(300, np.complex64), TxMetadata(start_of_burst=True))
    y, md = eng.read_rx(rx, 1000, 0.0)
    assert y.size == 300 and md.first_sample == 0  # TX committed only through 300
    y, md = eng.read_rx(rx, 1000, 0.0)
    assert y.size == 0 and md.error_code == ErrorCode.TIMEOUT
    eng.submit_tx(
        tx,
        np.zeros(0, np.complex64),
        TxMetadata(has_time_spec=True, time_spec=TimeSpec(0, 1e-3), idle_until=True),
    )
    y, md = eng.read_rx(rx, 1000, 0.0)
    assert (y.size, md.first_sample) == (700, 300)
    assert not y.any()
    assert eng.read_rx(rx, 10, 0.0)[1].error_code == ErrorCode.TIMEOUT


def test_late_timed_burst_is_dropped_with_time_error():
    eng, tx, rx = engine_with_pair()
    eng.submit_tx(
        tx,
        np.zeros(0, np.complex64),
        TxMetadata(has_time_spec=True, time_spec=TimeSpec(0, 1e-3), idle_until=True),
    )
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    assert eng.read_rx(rx, 500, 0.0)[0].size == 500
    late = TxMetadata(has_time_spec=True, time_spec=TimeSpec(0, 0.2e-3), end_of_burst=True)
    assert eng.submit_tx(tx, np.ones(50, np.complex64), late) == 0
    assert eng.counters["late_samples"] == 50
    ev = eng.recv_async(tx, 0.0)
    assert ev is not None
    assert (ev.event_code, ev.sample_index, ev.n_samples) == (AsyncCode.TIME_ERROR, 200, 50)


def test_untimed_burst_after_eob_starts_at_the_frontier():
    eng, tx, rx = engine_with_pair()
    eng.submit_tx(tx, np.ones(10, np.complex64), TxMetadata(start_of_burst=True, end_of_burst=True))
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    assert eng.read_rx(rx, 400, 0.0)[0].size == 400  # silence after EOB is committed
    eng.submit_tx(
        tx, 2 * np.ones(5, np.complex64), TxMetadata(start_of_burst=True, end_of_burst=True)
    )
    y, md = eng.read_rx(rx, 10, 0.0)
    assert md.first_sample == 400
    np.testing.assert_array_equal(y[:5], 2 * np.ones(5, np.complex64))


def test_multi_rate_time_now_and_rate_mismatch():
    eng = FakeEngine()
    a = eng.open_stream(spec("rxa.rx0", "rx", "5880", 1.0e9))
    b = eng.open_stream(spec("rxb.rx0", "rx", "23.04e6", 2.0e9))
    eng.stream_cmd(a, StreamCmd("start_cont"))
    eng.stream_cmd(b, StreamCmd("start_cont"))
    assert eng.read_rx(a, 37, 0.0)[0].size == 37
    _, md = eng.read_rx(b, 100_000, 0.0)
    assert md.time_spec == TimeSpec(0, 0.0)
    assert eng.time_now() == min(Fraction(37, 5880), Fraction(100_000, 23_040_000))
    with pytest.raises(RateMismatchError):
        eng.open_stream(spec("txc.tx0", "tx", "1e6", 2.0e9))  # overlaps rxb at another rate


def test_stream_cmd_late_start_and_stop():
    eng, _, rx = engine_with_pair()
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    eng.read_rx(rx, 0, 0.0)
    eng2 = FakeEngine()
    r = eng2.open_stream(spec("x.rx0", "rx"))
    eng2.stream_cmd(r, StreamCmd("start_cont"))
    assert eng2.read_rx(r, 100, 0.0)[0].size == 100
    eng2.stream_cmd(r, StreamCmd("num_done", 5, stream_now=False, time_spec=TimeSpec(0, 1e-5)))
    assert eng2.read_rx(r, 5, 0.0)[1].error_code == ErrorCode.LATE_COMMAND
    eng2.stream_cmd(r, StreamCmd("stop_cont"))
    assert eng2.read_rx(r, 5, 0.0)[1].error_code == ErrorCode.TIMEOUT


# --------------------------------------------------------------------------- commands


def two_rate_engine():
    eng, txa, rxa = engine_with_pair("1e6", 3.5e9, "a")
    _, txb, rxb = engine_with_pair("5880", 1.0e9, "b", eng=eng)
    for tx, n in ((txa, 20_000), (txb, 118)):
        eng.submit_tx(
            tx, np.ones(n, np.complex64), TxMetadata(start_of_burst=True, end_of_burst=True)
        )
    for rx in (rxa, rxb):
        eng.stream_cmd(rx, StreamCmd("start_cont"))
    return eng, rxa, rxb


def test_command_lands_at_to_ticks_on_every_rx():
    eng, rxa, rxb = two_rate_engine()
    req = CommandRequest(
        "c-1",
        "ris.set_config",
        "client",
        "ris0",
        CommandTime.at(Fraction(1, 300)),
        states=states(1),
    )
    rc = eng.schedule(req)
    # 1/300 s: 3333.33 samples at 1 MS/s -> 3333; 19.6 samples at 5880 S/s -> 20
    assert rc.ok and rc.status == "scheduled"
    assert rc.k_req == {"rxa.rx0": 3333, "rxb.rx0": 20}
    ya, _ = read_all(eng, rxa, 10_000, 997)
    yb, _ = read_all(eng, rxb, 59, 7)
    np.testing.assert_array_equal(ya[:3333], 1)
    np.testing.assert_array_equal(ya[3333:], 2)
    np.testing.assert_array_equal(yb[:20], 1)
    np.testing.assert_array_equal(yb[20:], 2)
    (rec,) = eng.records
    assert rec.status == "applied"
    assert rec.lateness_samples == {"rxa.rx0": 0, "rxb.rx0": 0}
    assert rec.config_hash == PanelConfig("ris0", states(1)).config_hash()


def test_late_command_apply_on_arrival_uses_one_physical_time():
    eng, rxa, rxb = two_rate_engine()
    read_all(eng, rxa, 1000, 1000)
    read_all(eng, rxb, 10, 10)
    rc = eng.schedule(
        CommandRequest(
            "late", "ris.set_config", "client", "ris0", CommandTime.at("5e-4"), states=states(1)
        )
    )
    assert rc.ok and rc.status == "late_applied"
    rec = eng.records[-1]
    # t_sw' = max(1000/1e6, 10/5880) = 10/5880 s; 10/5880 * 1e6 = 1700.68 -> 1701
    assert rec.t_applied == Fraction(10, 5880)
    assert rec.k_req == {"rxa.rx0": 500, "rxb.rx0": 3}  # 0.5e-3 * 5880 = 2.94 -> 3
    assert rec.k_applied == {"rxa.rx0": 1701, "rxb.rx0": 10}
    assert rec.lateness_samples == {"rxa.rx0": 1201, "rxb.rx0": 7}
    assert eng.counters["late_commands"] == 1
    assert eng.nondeterministic_inputs  # late command from an unregistered source
    ya, _ = read_all(eng, rxa, 1000, 1000)
    np.testing.assert_array_equal(ya[:701], 1)
    np.testing.assert_array_equal(ya[701:], 2)


def test_late_command_reject_applies_nothing():
    eng, rxa, _ = two_rate_engine()
    read_all(eng, rxa, 1000, 1000)
    rc = eng.schedule(
        CommandRequest(
            "r",
            "ris.set_config",
            "client",
            "ris0",
            CommandTime.at("5e-4"),
            states=states(1),
            late_policy="reject",
        )
    )
    assert (rc.ok, rc.status, rc.error) == (False, "rejected", ErrorCode.LATE_COMMAND)
    assert eng.records[-1].status == "rejected"
    ya, _ = read_all(eng, rxa, 1000, 1000)
    np.testing.assert_array_equal(ya, 1)


def test_same_time_tie_later_seq_wins_and_future_commands_all_apply():
    eng, rxa, rxb = two_rate_engine()
    at = CommandTime.at("2e-3")
    eng.schedule(CommandRequest("first", "ris.set_config", "c", "ris0", at, states=states(1)))
    eng.schedule(CommandRequest("second", "ris.set_config", "c", "ris0", at, states=states(2)))
    eng.schedule(
        CommandRequest(
            "third", "ris.set_config", "c", "ris0", CommandTime.at("3e-3"), states=states(1)
        )
    )
    eng.schedule(
        CommandRequest(
            "fourth",
            "ris.set_config",
            "c",
            "ris0",
            CommandTime.at_sample("rxa.rx0", 4000),
            states=states(0),
        )
    )
    ya, _ = read_all(eng, rxa, 5000, 1234)
    np.testing.assert_array_equal(ya[:2000], 1)
    np.testing.assert_array_equal(ya[2000:3000], 3)
    np.testing.assert_array_equal(ya[3000:4000], 2)
    np.testing.assert_array_equal(ya[4000:], 1)
    status = {r.cmd_id: r.status for r in eng.records}
    assert status["second"] == "scheduled"  # rxb has not computed past its switch yet
    read_all(eng, rxb, 59, 59)
    status = {r.cmd_id: r.status for r in eng.records}
    assert status["first"] == "superseded"
    assert status["second"] == status["third"] == status["fourth"] == "applied"
    seqs = [r.seq for r in eng.records]
    assert seqs == sorted(seqs) and len(set(seqs)) == 4


def test_counterfactual_enable_and_rejections():
    eng, rxa, _ = two_rate_engine()
    eng.schedule(
        CommandRequest("off", "ris.set_enabled", "c", "ris0", CommandTime.at("1e-3"), enabled=False)
    )
    ya, _ = read_all(eng, rxa, 2000, 2000)
    np.testing.assert_array_equal(ya[1000:], 0.5)
    assert eng.records[-1].counterfactual
    cases = [
        (
            CommandRequest(
                "u", "ris.set_config", "c", "risX", CommandTime.asap(), states=states(1)
            ),
            ErrorCode.UNKNOWN_TARGET,
        ),
        (
            CommandRequest(
                "p", "node.set_position", "c", "rxa", CommandTime.asap(), position_m=np.zeros(3)
            ),
            ErrorCode.NOT_PERMITTED,
        ),
        (
            CommandRequest(
                "k", "ris.set_config", "c", "ris0", CommandTime.asap(), codebook_index=2
            ),
            ErrorCode.SCHEMA_ERROR,
        ),
    ]
    for req, code in cases:
        rc = eng.schedule(req)
        assert (rc.ok, rc.error) == (False, code)
    m = eng.schedule(CommandRequest("m", "rec.marker", "c", "run", CommandTime.at("3e-3")))
    assert m.ok and eng.records[-1].op == "rec.marker"


# --------------------------------------------------------------------------- barriers


class Toggle:
    name = "toggle"
    period_s = Fraction(1, 1000)
    min_latency_s = Fraction(1, 5000)
    spec = ObservationSpec(window_s=Fraction(1, 2000))
    latency = LatencyModel(fixed_s=Fraction(1, 5000))

    def __init__(self, early: bool = False) -> None:
        self.obs: list = []
        self.early = early
        self.caps = None
        self.rng = None

    def setup(self, caps, rng) -> None:
        self.caps, self.rng = caps, rng

    def decide(self, obs):
        self.obs.append(obs)
        at = obs.t_b + Fraction(1, 10_000) if self.early else None
        return [Action("ris0", states=states(len(self.obs) % 2), at_s=at)]


def test_barrier_controller_is_deterministic_with_zero_lateness():
    eng, tx, rx = engine_with_pair()
    ctrl = Toggle()
    eng.register_controller(ctrl)
    assert ctrl.caps == {"ris0": CAP} and isinstance(ctrl.rng, np.random.Generator)
    eng.submit_tx(tx, np.full(6000, 0.1, np.complex64), TxMetadata(end_of_burst=True))
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    y, _ = read_all(eng, rx, 5000, 333)
    # barrier m at t_b = m ms; its action switches at t_b + 0.2 ms = sample 1000 m + 200
    g = np.ones(5000)
    for m in range(5):
        g[1000 * m + 200 :] = 1 + (m + 1) % 2
    np.testing.assert_array_equal(y, (0.1 * g).astype(np.complex64))
    assert [o.t_b for o in ctrl.obs] == [Fraction(m, 1000) for m in range(5)]
    assert [o.k_b["rxa.rx0"] for o in ctrl.obs] == [0, 1000, 2000, 3000, 4000]
    assert math.isnan(ctrl.obs[0].rx_power_dbm["rxa.rx0"])  # empty window before t = 0
    # window [k_b - 500, k_b): gain 2 at m = 1 -> 10 log10(0.04); gain 1 at m = 2 -> -20 dBm
    assert ctrl.obs[1].rx_power_dbm["rxa.rx0"] == pytest.approx(10 * math.log10(0.04), abs=1e-6)
    assert ctrl.obs[2].rx_power_dbm["rxa.rx0"] == pytest.approx(-20.0, abs=1e-6)
    assert ctrl.obs[1].ris_state["ris0"][0] == 1
    assert all(sum(r.lateness_samples.values()) == 0 for r in eng.records)
    assert not eng.nondeterministic_inputs


def test_barrier_lookahead_violation_raises():
    eng, tx, rx = engine_with_pair()
    eng.register_controller(Toggle(early=True))
    eng.submit_tx(tx, np.ones(3000, np.complex64), TxMetadata(end_of_burst=True))
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    with pytest.raises(LookaheadViolation):
        read_all(eng, rx, 3000, 3000)


class Proxy:
    name = "remote"
    period_s = Fraction(1, 1000)
    min_latency_s = Fraction(1, 5000)
    spec = ObservationSpec()
    latency = LatencyModel()

    def __init__(self, eng: FakeEngine, ack: bool = True) -> None:
        self.eng, self.ack, self.barriers = eng, ack, []

    def send_barrier(self, obs) -> None:
        self.barriers.append(obs.t_b)

    def wait_ack(self, t_b, timeout_s) -> bool:
        # the remote controller sends ris.set_config before ctrl.ack
        self.eng.schedule(
            CommandRequest(
                f"r{len(self.barriers)}",
                "ris.set_config",
                self.name,
                "ris0",
                CommandTime.at(t_b + self.min_latency_s),
                states=states(len(self.barriers) % 2),
            )
        )
        return self.ack


def test_remote_proxy_barrier_protocol():
    eng, tx, rx = engine_with_pair()
    proxy = Proxy(eng)
    eng.register_controller(proxy)
    eng.submit_tx(tx, np.ones(3000, np.complex64), TxMetadata(end_of_burst=True))
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    y, _ = read_all(eng, rx, 2500, 450)
    assert proxy.barriers == [Fraction(0), Fraction(1, 1000), Fraction(2, 1000)]
    assert all(sum(r.lateness_samples.values()) == 0 for r in eng.records)
    np.testing.assert_array_equal(y[200:1200], 2)
    np.testing.assert_array_equal(y[1200:2200], 1)
    assert not eng.nondeterministic_inputs


def test_remote_proxy_without_ack_times_out():
    eng, tx, rx = engine_with_pair()
    eng.register_controller(Proxy(eng, ack=False))
    eng.submit_tx(tx, np.ones(500, np.complex64), TxMetadata(end_of_burst=True))
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    eng.read_rx(rx, 200, 0.0)  # up to the first limit (sample 200)
    with pytest.raises(SremuTimeout):
        eng.read_rx(rx, 10, 0.0)


# --------------------------------------------------------------------------- review additions


def _subset(cmd_id: str, at: str, idx: tuple[int, ...], vals: list[int]) -> CommandRequest:
    return CommandRequest(
        cmd_id,
        "ris.set_config",
        "c",
        "ris0",
        CommandTime.at(at),
        states=np.array(vals, dtype=np.int16),
        elements=idx,
    )


def test_element_subsets_compose_in_time_order_not_arrival_order():
    """A command for a later time that arrives first must not hide the element change of an
    earlier-time command that arrives second (state at t = fold of all updates with t' <= t
    in (t', seq) order)."""

    def gain2(cfgs) -> complex:
        st = cfgs["ris0"].state
        return complex(1 + int(st[0]) + 10 * int(st[1]))

    eng = FakeEngine({"ris0": CAP}, gain_fn=gain2)
    tx = eng.open_stream(spec("txa.tx0", "tx"))
    rx = eng.open_stream(spec("rxa.rx0", "rx"))
    eng.submit_tx(tx, np.ones(4000, np.complex64), TxMetadata(end_of_burst=True))
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    assert eng.schedule(_subset("A", "2e-3", (0,), [1])).ok  # arrives first, later time
    assert eng.schedule(_subset("B", "1e-3", (1,), [2])).ok  # arrives second, earlier time
    np.testing.assert_array_equal(eng.panel_state("ris0", Fraction(3, 1000)).state, [1, 2, 0, 0])
    np.testing.assert_array_equal(eng.panel_state("ris0", Fraction(3, 2000)).state, [0, 2, 0, 0])
    y, _ = read_all(eng, rx, 3000, 512)
    np.testing.assert_array_equal(y[:1000], 1)  # initial state
    np.testing.assert_array_equal(y[1000:2000], 21)  # B: st[1] = 2
    np.testing.assert_array_equal(y[2000:], 22)  # A on top of B
    assert {r.cmd_id: r.status for r in eng.records} == {"A": "applied", "B": "applied"}


def test_same_time_superseded_subset_never_takes_effect():
    eng = FakeEngine({"ris0": CAP})
    eng.open_stream(spec("rxa.rx0", "rx"))
    eng.schedule(_subset("early", "1e-3", (0,), [2]))
    eng.schedule(_subset("late", "1e-3", (1,), [1]))
    np.testing.assert_array_equal(eng.panel_state("ris0", Fraction(1, 1000)).state, [0, 1, 0, 0])
    status = {r.cmd_id: r.status for r in eng.records}
    assert status["early"] == "superseded"
    rec = next(r for r in eng.records if r.cmd_id == "late")
    assert rec.config_hash == PanelConfig("ris0", np.array([0, 1, 0, 0], np.int16)).config_hash()


def test_every_rejected_command_is_counted_and_logged_with_a_seq():
    """Section 4.6 / I6: every command produces a CommandRecord, rejections included."""
    eng, rxa, _ = two_rate_engine()
    read_all(eng, rxa, 1000, 1000)
    reqs = [
        CommandRequest("u", "ris.set_config", "c", "risX", CommandTime.asap(), states=states(1)),
        CommandRequest(
            "p", "node.set_position", "c", "rxa", CommandTime.asap(), position_m=np.zeros(3)
        ),
        CommandRequest("k", "ris.set_config", "c", "ris0", CommandTime.asap(), codebook_index=2),
        CommandRequest(
            "s",
            "ris.set_config",
            "c",
            "ris0",
            CommandTime.at_sample("nobody.rx0", 5),
            states=states(1),
        ),
        CommandRequest(
            "late",
            "ris.set_config",
            "c",
            "ris0",
            CommandTime.at("5e-4"),
            states=states(2),
            late_policy="reject",
        ),
    ]
    codes = [eng.schedule(r).error for r in reqs]
    assert codes == [
        ErrorCode.UNKNOWN_TARGET,
        ErrorCode.NOT_PERMITTED,
        ErrorCode.SCHEMA_ERROR,
        ErrorCode.UNKNOWN_TARGET,
        ErrorCode.LATE_COMMAND,
    ]
    recs = eng.records
    assert [r.cmd_id for r in recs] == ["u", "p", "k", "s", "late"]
    assert all(r.status == "rejected" and r.t_applied is None and r.k_applied == {} for r in recs)
    assert [r.seq for r in recs] == [1, 2, 3, 4, 5]
    assert eng.counters["rejected_commands"] == 5
    assert eng.counters["late_commands"] == 1
    late = recs[-1]
    assert late.t_req == Fraction(1, 2000)
    assert late.k_req == {"rxa.rx0": 500, "rxb.rx0": 3}  # R3 at 1 MS/s and 5880 S/s
    assert late.config_hash == PanelConfig("ris0", states(2)).config_hash()
    y, _ = read_all(eng, rxa, 1000, 1000)
    np.testing.assert_array_equal(y, 1)  # nothing was applied


class EarlyProxy(Proxy):
    """Remote controller that answers barrier t_b with a command at t_b + L_min/2."""

    def wait_ack(self, t_b, timeout_s) -> bool:
        self.receipt = self.eng.schedule(
            CommandRequest(
                f"e{len(self.barriers)}",
                "ris.set_config",
                self.name,
                "ris0",
                CommandTime.at(t_b + self.min_latency_s / 2),
                states=states(1),
            )
        )
        return True


def test_remote_lookahead_violation_is_rejected_not_applied_late():
    """Section 4.9 / RT-17 semantics: a barrier-registered source acting before t_b + L_min
    gets LOOKAHEAD_VIOLATION; it must not be silently moved (late) with lateness > 0."""
    eng, tx, rx = engine_with_pair()
    proxy = EarlyProxy(eng)
    eng.register_controller(proxy)
    eng.submit_tx(tx, np.ones(1000, np.complex64), TxMetadata(end_of_burst=True))
    eng.stream_cmd(rx, StreamCmd("start_cont"))
    y, _ = read_all(eng, rx, 600, 600)
    assert (proxy.receipt.ok, proxy.receipt.error) == (False, ErrorCode.LOOKAHEAD_VIOLATION)
    assert all(r.status == "rejected" for r in eng.records)
    assert eng.counters["late_commands"] == 0
    np.testing.assert_array_equal(y, 1)  # nothing applied
