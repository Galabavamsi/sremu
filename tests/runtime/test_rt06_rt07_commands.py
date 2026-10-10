"""RT-06 (I6) late commands and RT-07 same-time ties (DESIGN.md sections 4.6, 10.3, 10.7).

RT-06: ``apply_on_arrival`` moves a late command to ONE physical switch time
``t' = max_{r,j}(frontier_r/f_r - tau_{r,j})`` and re-maps every RX with R3, logging the
exact lateness; ``reject`` returns ``LATE_COMMAND`` and applies nothing.
RT-07: for two commands with the same t on the same panel the later seq wins, both are
logged and the earlier is ``superseded``; two future commands both apply. Teeth:
``LegacyNextTickAdapter`` (v2 single slot) loses the first of two commands.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import CommandTime
from sremu.engine.streams import ToneSource
from sremu.errors import ErrorCode, NondeterminismWarning, NonPhysicalError
from sremu.legacy.next_tick import LegacyNextTickAdapter
from sremu.timebase import from_ticks, to_ticks
from tests.runtime._engine_support import captured, config_request, lab_like_fake, make_engine

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
RXS = ("rx1.rx0", "rx2.rx0")


def _engine(**kw: object):
    eng = make_engine(
        lab_like_fake(far_m=2000.0),
        traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)},
        always_active=True,
        verify_interp=False,
        **kw,
    )
    caps = {r: eng.capture_accumulator(r) for r in RXS}
    return eng, caps


def _ones(eng) -> np.ndarray:
    return np.ones(eng.commands.specs["ris0"].n_controls, np.int16)


def _baseline(n: int) -> dict[str, np.ndarray]:
    eng, caps = _engine()
    eng.run(Fraction(n) / F)
    return {r: captured(c, 0, n) for r, c in caps.items()}


# --------------------------------------------------------------------------- RT-06


def test_rt06_apply_on_arrival_one_physical_time_exact_lateness() -> None:
    n = 6000
    base = _baseline(n)
    eng, caps = _engine()
    eng.run(Fraction(2, 1000))  # frontier 2000 on both RX
    fr = eng.frontier()
    assert fr == {"rx1.rx0": 2000, "rx2.rx0": 2000}
    t_req = Fraction(1500, 10**6)
    with pytest.warns(NondeterminismWarning):  # late command from an unregistered source
        receipt = eng.schedule(config_request("late", "ris0", _ones(eng), CommandTime.at(t_req)))
    assert receipt.ok
    assert receipt.status == "late_applied"
    model = eng.model
    taus = {r: [model.ris_delay_s("ris0", r, t_req, j) for j in range(3)] for r in RXS}
    t_exp = max(from_ticks(fr[r], F) - Fraction(tau) for r in RXS for tau in taus[r])
    rec = eng.records[0]
    assert rec.t_req == t_req
    assert rec.t_applied == t_exp  # one physical switch time for every RX
    eng.run(Fraction(n) / F)
    out = {r: captured(c, 0, n) for r, c in caps.items()}
    for r in RXS:
        tau_c = model.ris_delay_s("ris0", r, t_exp)
        k_app = to_ticks(rec.t_applied + Fraction(tau_c), F)
        k_req = to_ticks(t_req + Fraction(model.ris_delay_s("ris0", r, t_req)), F)
        assert rec.k_applied[r] == k_app
        assert rec.k_req[r] == k_req
        assert rec.lateness_samples[r] == k_app - k_req
        assert k_app >= fr[r]
        tile_k = min(
            to_ticks(t_exp + Fraction(model.ris_delay_s("ris0", r, t_exp, j)), F) for j in range(3)
        )
        assert tile_k >= fr[r]
        d = np.flatnonzero(out[r] != base[r])
        assert int(d[0]) == tile_k  # first affected sample on this RX
        assert np.array_equal(out[r][:tile_k], base[r][:tile_k])
    # the binding RX switches exactly at its frontier; the far RX about 6.7 samples later
    assert rec.k_applied["rx1.rx0"] == 2000
    assert rec.k_applied["rx2.rx0"] - 2000 in (6, 7)
    assert rec.status == "late_applied"
    assert eng.counters["late_commands"] == 1
    assert eng.counters["rejected_commands"] == 0
    assert eng.nondeterministic_inputs  # late command from an unregistered source


def test_rt06_reject_applies_nothing() -> None:
    n = 4000
    base = _baseline(n)
    eng, caps = _engine()
    eng.run(Fraction(2, 1000))
    with pytest.warns(NondeterminismWarning):
        receipt = eng.schedule(
            config_request(
                "late", "ris0", _ones(eng), CommandTime.at(Fraction(1, 1000)), late_policy="reject"
            )
        )
    assert not receipt.ok
    assert receipt.error == ErrorCode.LATE_COMMAND
    assert receipt.status == "rejected"
    eng.run(Fraction(n) / F)
    out = {r: captured(c, 0, n) for r, c in caps.items()}
    for r in RXS:
        assert np.array_equal(out[r], base[r])
    rec = eng.records[0]
    assert rec.status == "rejected"
    assert rec.t_applied is None
    assert rec.seq == 1
    assert eng.counters["late_commands"] == 1
    assert eng.counters["rejected_commands"] == 1


def test_rt06_on_time_command_is_not_late() -> None:
    eng, _caps = _engine()
    eng.run(Fraction(2, 1000))
    receipt = eng.schedule(
        config_request("ok", "ris0", _ones(eng), CommandTime.at(Fraction(2, 1000)))
    )
    assert receipt.ok
    assert receipt.status == "scheduled"
    eng.run(Fraction(3, 1000))
    rec = eng.records[0]
    assert rec.status == "applied"
    assert all(v == 0 for v in rec.lateness_samples.values())
    assert eng.counters["late_commands"] == 0
    assert not eng.nondeterministic_inputs


# --------------------------------------------------------------------------- RT-07


def test_rt07_same_time_later_seq_wins_earlier_superseded() -> None:
    n = 5000
    t = Fraction(2, 1000)
    eng_a, caps_a = _engine()
    ones = _ones(eng_a)
    half = ones.copy()
    half[: half.size // 2] = 0
    ra = eng_a.schedule(config_request("first", "ris0", ones, CommandTime.at(t)))
    rb = eng_a.schedule(config_request("second", "ris0", half, CommandTime.at(t)))
    assert ra.ok
    assert rb.ok
    assert rb.seq > ra.seq
    eng_a.run(Fraction(n) / F)
    by_id = {r.cmd_id: r for r in eng_a.records}
    assert by_id["first"].status == "superseded"
    assert by_id["second"].status == "applied"
    # identical to a run that only ever saw the later command
    eng_b, caps_b = _engine()
    eng_b.schedule(config_request("second", "ris0", half, CommandTime.at(t)))
    eng_b.run(Fraction(n) / F)
    for r in RXS:
        assert np.array_equal(captured(caps_a[r], 0, n), captured(caps_b[r], 0, n))


def test_rt07_two_future_commands_both_apply() -> None:
    n = 6000
    eng, caps = _engine()
    ones = _ones(eng)
    eng.schedule(config_request("a", "ris0", ones, CommandTime.at(Fraction(2, 1000))))
    eng.schedule(config_request("b", "ris0", 0 * ones, CommandTime.at(Fraction(4, 1000))))
    eng.run(Fraction(n) / F)
    assert [r.status for r in eng.records] == ["applied", "applied"]
    y = captured(caps["rx1.rx0"], 0, n)
    base = _baseline(n)["rx1.rx0"]
    assert np.array_equal(y[:2000], base[:2000])
    assert not np.array_equal(y[2100:3900], base[2100:3900])  # first command visible
    assert np.array_equal(y[4100:], base[4100:])  # second command restores the initial state


def test_rt07_teeth_legacy_next_tick_loses_the_first_command() -> None:
    """v2 semantics: two commands submitted within one 2 ms tick -> only the second is
    applied (at the next tick); the native engine applies both (previous test)."""
    with pytest.raises(NonPhysicalError):
        LegacyNextTickAdapter(_engine()[0])
    n = 6000
    eng, caps = _engine()
    eng.run(Fraction(1, 1000))  # frontier at 1 ms: next v2 tick is 2 ms
    legacy = LegacyNextTickAdapter(eng, Fraction(2, 1000), allow_nonphysical=True)
    ones = _ones(eng)
    legacy.schedule(config_request("a", "ris0", ones, CommandTime.at(Fraction(11, 10000))))
    legacy.schedule(config_request("b", "ris0", 0 * ones, CommandTime.at(Fraction(15, 10000))))
    eng.run(Fraction(n) / F)
    status = {r.cmd_id: r.status for r in eng.records}
    assert status == {"a": "superseded", "b": "applied"}
    assert legacy.lost == 1
    assert eng.records[1].t_applied == Fraction(2, 1000)
    # the first command never took effect: output equals the no-command baseline
    assert np.array_equal(captured(caps["rx1.rx0"], 0, n), _baseline(n)["rx1.rx0"])
