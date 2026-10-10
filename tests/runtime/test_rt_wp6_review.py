"""WP6 review regressions (DESIGN.md sections 4.2-4.6, 4.9-4.11, 10.3).

Each test pins a defect found in the WP6 review and its fix:

* I3 (RT-05) with timed geometry events: a RIS switch after ``node.set_position`` uses
  ``tau_RIS->r`` of the geometry in force at ``t_sw`` (static scenes cached tau at t = 0;
  pending commands kept the tau of their scheduling time);
* I2b (RT-04) and the ``nondeterministic_inputs`` flag with an on-time geometry event on a
  snapshot grid (it was applied only once its page started, after its snapshot interval);
* geometry events: logged sample = first affected sample; equal-time events keep the
  geometry before them as the left limit; an event behind one already applied is moved
  (late), not a crash;
* RT-08 observations: an observation window longer than the default RX history;
* section 4.11 batch path: a non-step transition in a static scene follows its profile;
* RT-18 paced counters: ``deadline_misses`` at the smallest configured slack; a page with
  partial TX data is one release, not two;
* I6: client samples sent to a generator-driven TX stream are refused, not dropped.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import (
    CommandRequest,
    CommandTime,
    PanelConfig,
    StreamSpec,
    TransitionSpec,
    TxMetadata,
)
from sremu.engine.batch import run_batch
from sremu.engine.pacer import FakeClock
from sremu.engine.streams import ToneSource
from sremu.errors import NondeterminismWarning, StreamError
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import stream
from sremu.timebase import TimeSpec, ceil_ticks, to_ticks
from tests.runtime._engine_support import (
    Collect,
    ScriptedController,
    captured,
    config_request,
    lab_like_fake,
    make_engine,
    waveform,
)

pytestmark = pytest.mark.runtime

FC = 3.5e9
F1 = Fraction(10**6)


def _move(node: str, t: Fraction, x_m: float, cmd_id: str = "mv") -> CommandRequest:
    return CommandRequest(
        cmd_id,
        "node.set_position",
        "test",
        node,
        CommandTime.at(t),
        position_m=np.array([x_m, 0.0, 0.0]),
    )


# --------------------------------------------------------------------------- I3 + geometry


def _static_ris_scene(rate: Fraction) -> FakePropagation:
    """Static scene: AP 2 m from a panel, external RX ``mu`` 1.2 m in front of it."""
    nodes = [
        FakeNode("ap", (1.7320508, -1.0, 0.0), streams=(stream("ap.tx0", "tx", FC, rate),)),
        FakeNode("mu", (1.2, 0.0, 0.0), streams=(stream("mu.rx0", "rx", FC, rate),), external=True),
    ]
    panel = FakePanel(
        "ris0", (0, 0, 0), (1, 0, 0), (0, 1, 0), 8, 8, 0.0428, 0.0428, gamma=(1 + 0j, -1 + 0j)
    )
    return FakePropagation(nodes, [panel], horizon_s=1)


@pytest.mark.parametrize("command_first", [True, False], ids=["cmd_first", "move_first"])
def test_rt05_i3_switch_after_rx_moved_by_set_position(command_first: bool) -> None:
    """23.04 MS/s static scene: ``mu`` jumps from 1.2 m to 2 km at 100 us, a RIS switch at
    200 us must land on ``to_ticks(t_sw + tau_new)`` (about 154 samples later than with the
    old geometry), whatever the order in which the two commands were scheduled."""
    f = Fraction(23_040_000)
    n = 8000
    t_sw = Fraction(2, 10_000)

    def run(with_cmd: bool):
        eng = make_engine(
            _static_ris_scene(f),
            traffic={"ap.tx0": ToneSource(f, 0.0, 1.0)},
            always_active=True,
            verify_interp=False,
        )
        cap = eng.capture_accumulator("mu.rx0")
        nc = eng.commands.specs["ris0"].n_controls
        cmd = config_request("sw", "ris0", np.ones(nc, np.int16), CommandTime.at(t_sw))
        if with_cmd and command_first:
            assert eng.schedule(cmd).ok
        assert eng.schedule(_move("mu", Fraction(1, 10_000), 2000.0)).ok
        if with_cmd and not command_first:
            assert eng.schedule(cmd).ok
        eng.run(Fraction(n) / f)
        return eng, captured(cap, 0, n)

    eng, y = run(True)
    _, y0 = run(False)
    tau_new = eng.model.ris_delay_s("ris0", "mu.rx0", t_sw)
    assert tau_new * float(f) > 150.0  # the RX really is 2 km away at t_sw
    k = to_ticks(t_sw + Fraction(tau_new), f)
    first = int(np.flatnonzero(y != y0)[0])
    assert first == k
    rec = next(r for r in eng.records if r.cmd_id == "sw")
    assert rec.k_applied == {"mu.rx0": k}
    assert rec.k_req == {"mu.rx0": k}
    assert rec.lateness_samples == {"mu.rx0": 0}
    assert rec.status == "applied"
    assert not eng.nondeterministic_inputs


def _grid_scene() -> FakePropagation:
    """Moving AP (snapshot grid) and an external, initially static RX ``mu`` at 10 m."""
    nodes = [
        FakeNode(
            "ap", (0.0, 0.0, 0.0), (0.0, 20.0, 0.0), streams=(stream("ap.tx0", "tx", FC, F1),)
        ),
        FakeNode("mu", (10.0, 0.0, 0.0), streams=(stream("mu.rx0", "rx", FC, F1),), external=True),
    ]
    return FakePropagation(nodes, horizon_s=1)


def _grid_run(page_target: str, events: list[CommandRequest], n: int = 6000):
    eng = make_engine(
        _grid_scene(),
        traffic={"ap.tx0": ToneSource(F1, 0.0, 1.0)},
        always_active=True,
        page_target_s=page_target,
        verify_interp=False,
    )
    assert eng.t_u is not None and eng.t_u > Fraction(1, 1000)  # grid intervals > 1 page
    cap = eng.capture_accumulator("mu.rx0")
    for ev in events:
        assert eng.schedule(ev).ok
    with warnings.catch_warnings():
        warnings.simplefilter("error", NondeterminismWarning)
        eng.run(Fraction(n) / F1)
    return eng, captured(cap, 0, n)


def test_rt04_i2b_on_time_geometry_event_inside_grid_interval_page_invariant() -> None:
    """An on-time ``node.set_position`` scheduled before the run, inside a snapshot interval
    (T_u = 1.167 ms, event at 1.9 ms): not flagged nondeterministic and the complex128
    output is the same for pages of 256, 512 and 1024 samples (I2b, <= 1e-9)."""
    t_e = Fraction(19, 10_000)
    outs = {}
    for target in ("2.5e-4", "5e-4", "1e-3"):
        eng, y = _grid_run(target, [_move("mu", t_e, 12.0)])
        assert not eng.nondeterministic_inputs
        outs[target] = y
    ref = outs["5e-4"]
    assert abs(abs(ref[5000]) / abs(ref[0]) - 10.0 / 12.0) < 0.02  # the move is visible
    for y in outs.values():
        assert np.max(np.abs(y - ref)) <= 1e-9 * np.max(np.abs(ref))


def test_geometry_events_at_the_same_time_keep_the_left_limit() -> None:
    """Two ``node.set_position`` at the same t (12 m, then 14 m: the later wins) give the
    same output as the single 14 m event; the interval before t ends on the geometry
    before both events (bitwise)."""
    t_e = Fraction(19, 10_000)
    _, two = _grid_run("5e-4", [_move("mu", t_e, 12.0, "a"), _move("mu", t_e, 14.0, "b")])
    _, one = _grid_run("5e-4", [_move("mu", t_e, 14.0, "b")])
    assert np.array_equal(two.view(np.float64), one.view(np.float64))


def test_geometry_event_logged_at_its_first_affected_sample() -> None:
    """``frac(t_e f) = 0.3``: the move shows from ``ceil(t_e f)`` (geometry at the sample
    instants n/f), and the record logs that sample (I6)."""
    nodes = [
        FakeNode("ap", (0.0, 0.0, 0.0), streams=(stream("ap.tx0", "tx", FC, F1),)),
        FakeNode("mu", (10.0, 0.0, 0.0), streams=(stream("mu.rx0", "rx", FC, F1),), external=True),
    ]
    eng = make_engine(
        FakePropagation(nodes, horizon_s=1),
        traffic={"ap.tx0": ToneSource(F1, 0.0, 1.0)},
        always_active=True,
    )
    cap = eng.capture_accumulator("mu.rx0")
    t_e = Fraction(15003, 10**7)
    assert eng.schedule(_move("mu", t_e, 20.0)).ok
    eng.run(Fraction(3, 1000))
    y = captured(cap, 0, 3000)
    first = int(np.flatnonzero(y != y[0])[0])
    assert first == ceil_ticks(t_e, F1) == 1501
    assert to_ticks(t_e, F1) == 1500  # R3 rounding would log one sample early
    assert eng.records[0].k_applied == {"mu.rx0": 1501}
    assert eng.records[0].status == "applied"


def test_geometry_event_behind_an_applied_event_is_moved_late() -> None:
    """An event already applied ahead (with its snapshot interval) fixes the geometry up to
    its time; a later-arriving event for an earlier, not yet computed time is moved to that
    time (late_applied, counted, nondeterministic), instead of breaking the model's event
    order."""
    eng = make_engine(
        _grid_scene(),
        traffic={"ap.tx0": ToneSource(F1, 0.0, 1.0)},
        always_active=True,
        verify_interp=False,
    )
    t_a = Fraction(21, 10_000)
    assert eng.schedule(_move("mu", t_a, 12.0, "a")).ok
    eng.run(Fraction(1800, 10**6))  # frontier 1800; the event at 2.1 ms is applied ahead
    assert eng.frontier() == {"mu.rx0": 1800}
    assert eng._geo_applied == t_a
    with pytest.warns(NondeterminismWarning):
        r = eng.schedule(_move("mu", Fraction(19, 10_000), 14.0, "b"))
    assert r.ok
    assert r.status == "late_applied"
    eng.run(Fraction(3, 1000))
    rec = {x.cmd_id: x for x in eng.records}
    assert rec["b"].t_applied == t_a
    assert rec["b"].status == "late_applied"
    assert eng.counters["late_commands"] == 1
    assert eng.nondeterministic_inputs


# --------------------------------------------------------------------------- RT-08 window


def test_rt08_observation_window_longer_than_default_history() -> None:
    """A 10 ms observation window at 1 MS/s (longer than the default 4096-sample RX
    history): ``rx_power_dbm`` and ``iq_window`` cover the whole window
    [to_ticks(t_b - W), to_ticks(t_b))."""
    sink = Collect()
    eng = make_engine(
        lab_like_fake(),
        traffic={"ap.tx0": ToneSource(F1, 0.0, 1.0)},
        sinks=[sink],
        always_active=True,
    )
    ctrl = ScriptedController(
        "obs",
        "1/100",
        "1/1000",
        lambda o, m: [],
        window_s="1e-2",
        fields=("rx_power_dbm", "iq_window"),
    )
    eng.register_controller(ctrl)
    eng.run(Fraction(35, 1000))
    full = [o for o in ctrl.observations if o.t_b >= Fraction(1, 100)]
    assert len(full) == 3
    for o in full:
        for r in ("rx1.rx0", "rx2.rx0"):
            kb = o.k_b[r]
            d = sink.rx_array(r, kb - 10_000, kb)
            assert o.iq_window is not None
            assert np.array_equal(o.iq_window[r].view(np.uint64), d.view(np.uint64))
            # full scale 0 dBm, gain 0 dB: |d|**2 reads in mW (section 3.3)
            p_mw = float(np.mean(np.abs(d.astype(np.complex128)) ** 2))
            assert o.rx_power_dbm[r] == pytest.approx(10 * math.log10(p_mw), abs=1e-9)


# --------------------------------------------------------------------------- batch (4.11)


def _h(model: FakePropagation, states: np.ndarray, link: str) -> complex:
    snap = model.snapshot(Fraction(0))
    lp = model.combine(snap, {"ris0": PanelConfig("ris0", states.astype(np.int16))})[link]
    return complex(np.sum(lp.amp * np.exp(-2j * np.pi * np.mod(FC * lp.delay_s, 1.0))))


@pytest.mark.parametrize("profile", ["linear", "first_order"])
def test_batch_static_scene_transition_follows_its_profile(profile: str) -> None:
    """Static scene, one 20 us transition at 1 ms over 10 ms: every piece value is
    ``|h0 + g(t_i - t_sw)(h1 - h0)|**2`` at its start ``t_i`` (closed form, RIS time); once
    the profile has settled the pieces carry the new configuration, and the time average is
    within the left-point bound of the exact integral."""
    model = lab_like_fake(tiles=(1, 1))
    link = "ap.tx0->rx1.rx0"
    n = 120
    T = Fraction(20, 10**6)
    t_sw = Fraction(1, 1000)
    dur = Fraction(10, 1000)
    req = config_request(
        "c",
        "ris0",
        np.ones(n, np.int16),
        CommandTime.at(t_sw),
        transition=TransitionSpec(profile, T),
    )
    res = run_batch(model, dur, commands=[req])
    h0 = _h(model, np.zeros(n, np.int16), link)
    h1 = _h(model, np.ones(n, np.int16), link)

    def g(dt: Fraction) -> float:
        if dt < 0:
            return 0.0
        x = float(dt / T)
        return min(x, 1.0) if profile == "linear" else -math.expm1(-x)

    for ti, v in zip(res.t, res.h2[link], strict=True):
        expect = abs(h0 + g(ti - t_sw) * (h1 - h0)) ** 2
        assert v == pytest.approx(expect, rel=1e-12, abs=0.0)
    assert res.t[-1] < dur
    assert res.h2[link][-1] == pytest.approx(abs(h1) ** 2, rel=1e-12)  # settled: new config
    # time average vs the exact integral of |h(t)|^2 (dense midpoint rule, 2e5 points)
    s = (np.arange(200_000) + 0.5) / 200_000 * float(dur)
    dt = s - float(t_sw)
    gg = np.where(
        dt < 0,
        0.0,
        np.minimum(dt / float(T), 1.0)
        if profile == "linear"
        else -np.expm1(-np.maximum(dt, 0) / float(T)),
    )
    exact = float(np.mean(np.abs(h0 + gg * (h1 - h0)) ** 2))
    # left-point error <= step * total variation / duration; TV <= 2 max |h|^2 here
    step = float(T) / 64 if profile == "linear" else float(T) / 8
    tv = 2 * max(abs(h0), abs(h1), abs(h0 + 0.5 * (h1 - h0))) ** 2
    assert abs(res.mean_h2[link] - exact) <= step * tv / float(dur) + 1e-6 * exact
    # teeth: the commanded configuration dominates the average (it did not before the fix)
    assert abs(res.mean_h2[link] - abs(h1) ** 2) < abs(res.mean_h2[link] - abs(h0) ** 2)


# --------------------------------------------------------------------------- RT-18 paced


def test_rt18_deadline_misses_counter_at_smallest_slack() -> None:
    """Slacks (0.1 ms, 1 ms) without 0: an injected 0.3 ms load on 2 pages counts 2
    ``deadline_misses`` (misses at the smallest slack)."""
    clock = FakeClock()
    eng = make_engine(
        lab_like_fake(),
        backend="numpy_ref",
        mode="paced",
        clock=clock,
        traffic={"ap.tx0": ToneSource(F1, 0.0, 1.0)},
        always_active=True,
        slacks_s=("1e-4", "1e-3"),
    )
    loaded = {2, 5}
    seen: set[int] = set()

    def load(rx: str, k0: int, k1: int) -> None:
        j = k0 // 512
        if rx == "rx1.rx0" and j in loaded and j not in seen:
            seen.add(j)
            clock.advance(300_000)

    eng.on_page(load)
    s = eng.run_paced(Fraction(10 * 512) / F1)
    assert s.paced is not None
    assert s.paced["misses"] == {"0.1ms": 2, "1ms": 0}
    assert s.counters["deadline_misses"] == 2


def test_rt18_paced_page_with_partial_tx_data_is_one_release() -> None:
    """Paced, client TX covering only part of a page: the missing samples are zeros
    (UNDERFLOW, counted) and the page is one release, so the deadline statistics count
    exactly one page per page."""
    clock = FakeClock()
    eng = make_engine(
        lab_like_fake(),
        backend="numpy_ref",
        mode="paced",
        bw=8.0e5,
        clock=clock,
        always_active=True,
    )
    h = eng.open_stream(StreamSpec("ap.tx0", "tx", FC, F1))
    assert eng.submit_tx(h, waveform(700), TxMetadata(True, TimeSpec(0, 0), True, False)) == 700
    s = eng.run_paced(Fraction(4 * 512) / F1)
    assert s.paced is not None
    assert s.paced["pages"] == 4
    committed = eng.tx["ap.tx0"].ring.committed_through()
    assert s.counters["underflow_slip_samples"] == committed - 700


# --------------------------------------------------------------------------- I6


def test_i6_client_samples_to_generator_driven_tx_are_refused() -> None:
    eng = make_engine(lab_like_fake(), traffic={"ap.tx0": ToneSource(F1, 0.0, 1.0)})
    h = eng.open_stream(StreamSpec("ap.tx0", "tx", FC, F1))
    with pytest.raises(StreamError):
        eng.submit_tx(h, waveform(10), TxMetadata())


# --------------------------------------------------------------------------- diffuse (5.11)


def test_diffuse_long_tap_delay_sizes_the_tx_history() -> None:
    """A diffuse tap 5 ms after the LOS (5000 samples at 1 MS/s, beyond the default TX ring
    of 4096 samples): the ring grows for it and the tap term is the WP3 SoS gain times the
    TX sample 5000 samples back (it raised an overwritten-history error before)."""
    pytest.importorskip("sremu.physics.diffuse")
    from pathlib import Path

    from sremu.engine.emulator import Emulator
    from sremu.scenario.load import load_scenario

    res = load_scenario(
        Path(__file__).resolve().parents[2] / "scenarios" / "quickstart.yaml", warn=False
    )
    n = 8000

    def run(diffuse: bool) -> tuple[np.ndarray, Emulator]:
        d = res.scenario.model_dump(mode="python")
        d["nodes"][0]["streams"][0]["occupied_bw_hz"] = 1.0e3  # CW: integer-only delay
        d["traffic"] = []
        if diffuse:
            d["environment"]["diffuse"] = {
                "k_factor_db": 3.0,
                "max_doppler_hz": 50.0,
                "taps": [{"delay_s": 5.0e-3, "rel_power_db": 0.0}],
            }
        eng = Emulator.from_scenario(
            d,
            base_dir=res.base_dir,
            backend="numpy_ref",
            noise=False,
            always_active=True,
            traffic={"ap.tx0": ToneSource(F1, 0.0, 1.0)},
        )
        assert eng.tx["ap.tx0"].ring.capacity < 5000
        cap = eng.capture_accumulator("mu.rx0")
        eng.run(Fraction(n) / F1)
        return captured(cap, 0, n), eng

    y1, eng = run(True)
    y0, _ = run(False)
    (term,) = eng.builder.diffuse.terms(eng.model, eng.model.snapshot(Fraction(0)))[
        "ap.tx0->mu.rx0"
    ]
    k = round(term.delay_s * float(F1))
    assert k == 5000
    expect = np.zeros(n, dtype=np.complex128)
    expect[k:] = term.gain_samples(0, n, F1)[k:] * complex(np.complex64(np.sqrt(1e-3)))
    diff = y1 - y0
    assert np.max(np.abs(diff - expect)) <= 1e-9 * np.max(np.abs(expect))
    assert eng.tx["ap.tx0"].ring.capacity > 5000


def test_history_ring_grow_keeps_retained_samples() -> None:
    """``HistoryRing.grow`` (controller registration mid-run) keeps what was retained and
    then holds the larger window."""
    from sremu.engine.streams import HistoryRing

    h = HistoryRing(64)
    h.write(0, np.arange(100, dtype=np.complex64))  # retains [36, 100)
    h.grow(256)
    assert h.capacity == 256
    out = h.read(0, 100)
    assert np.all(out[:36] == 0)  # lost before the growth: still reads as 0
    assert out[36:].real.tolist() == list(range(36, 100))
    h.write(100, np.arange(100, 300, dtype=np.complex64))
    assert h.read(44, 256).real.tolist() == list(range(44, 300))


def test_geometry_event_after_an_applied_one_in_the_same_grid_interval_is_on_time() -> None:
    """Grid interval [1.167, 2.334) ms: event A at 1.5 ms is applied ahead; event B at
    2.0 ms arrives while the frontier is at 1.4 ms. The interval B splits, [1.5 ms, ...),
    has no computed sample, so B is on time: not flagged, and the output equals the run
    with both events scheduled before the start (bitwise)."""
    t_a, t_b = Fraction(15, 10_000), Fraction(2, 1000)
    eng = make_engine(
        _grid_scene(),
        traffic={"ap.tx0": ToneSource(F1, 0.0, 1.0)},
        always_active=True,
        verify_interp=False,
    )
    assert eng.t_u == Fraction(1167, 10**6)
    cap = eng.capture_accumulator("mu.rx0")
    assert eng.schedule(_move("mu", t_a, 12.0, "a")).ok
    eng.run(Fraction(1400, 10**6))
    assert eng._geo_applied == t_a  # applied ahead with its snapshot interval
    with warnings.catch_warnings():
        warnings.simplefilter("error", NondeterminismWarning)
        r = eng.schedule(_move("mu", t_b, 14.0, "b"))
        assert r.ok
        assert r.status == "scheduled"
        eng.run(Fraction(4, 1000))
    assert not eng.nondeterministic_inputs
    _, ref = _grid_run("5e-4", [_move("mu", t_a, 12.0, "a"), _move("mu", t_b, 14.0, "b")], 4000)
    assert np.array_equal(captured(cap, 0, 4000).view(np.float64), ref.view(np.float64))


def _static_two_external() -> FakePropagation:
    nodes = [
        FakeNode("ap", (0.0, 0.0, 0.0), streams=(stream("ap.tx0", "tx", FC, F1),), external=True),
        FakeNode("mu", (10.0, 0.0, 0.0), streams=(stream("mu.rx0", "rx", FC, F1),), external=True),
    ]
    return FakePropagation(nodes, horizon_s=1)


def test_static_scene_event_of_another_node_before_an_applied_one_is_on_time() -> None:
    """Static scene (event-only snapshots): ``mu``'s event at 1.4 ms is applied with its
    page; ``ap``'s event at 1.2 ms then arrives with the frontier at 1.1 ms. Different node,
    no snapshot computed past 1.1 ms: on time and bitwise equal to scheduling both up front.
    The same node earlier than its applied event is late (moved to 1.4 ms)."""
    t_a, t_b = Fraction(14, 10_000), Fraction(12, 10_000)

    def make():
        eng = make_engine(
            _static_two_external(), traffic={"ap.tx0": ToneSource(F1, 0.0, 1.0)}, always_active=True
        )
        assert eng.t_u is None
        return eng, eng.capture_accumulator("mu.rx0")

    eng, cap = make()
    assert eng.schedule(_move("mu", t_a, 12.0, "a")).ok
    eng.run(Fraction(1100, 10**6))
    assert eng._geo_applied_node == {"mu": t_a}
    with warnings.catch_warnings():
        warnings.simplefilter("error", NondeterminismWarning)
        r = eng.schedule(_move("ap", t_b, 1.0, "b"))
    assert r.ok
    assert r.status == "scheduled"
    with pytest.warns(NondeterminismWarning):
        late = eng.schedule(_move("mu", t_b, 15.0, "c"))
    assert late.status == "late_applied"
    eng.run(Fraction(3, 1000))
    assert {x.cmd_id: x.t_applied for x in eng.records} == {"a": t_a, "b": t_b, "c": t_a}
    ref_eng, ref_cap = make()
    for ev in (
        _move("mu", t_a, 12.0, "a"),
        _move("ap", t_b, 1.0, "b"),
        _move("mu", t_a, 15.0, "c"),
    ):
        assert ref_eng.schedule(ev).ok
    ref_eng.run(Fraction(3, 1000))
    assert np.array_equal(
        captured(cap, 0, 3000).view(np.float64), captured(ref_cap, 0, 3000).view(np.float64)
    )
