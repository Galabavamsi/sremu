"""WP11 experiment drivers (DESIGN.md section 11): fast checks of the helpers in
``experiments/`` (no long simulation; the studies themselves run from ``paper/sweeps``).

* E-A scenarios: the corridor geometry of every sweep point validates (schema, guards) and
  has N = floor(l / (lambda/4)) cells (133 at 1 GHz, 320 at 2.4 GHz).
* tau_half interpolation and censoring; the pre-registered decision helper.
* ``FastProbeBandit`` makes bitwise the same decisions as ``ProbeBandit`` (O(K) bookkeeping
  instead of the parent's O(history) rescan).
* E-B packet schedule and the [DERIVED] timing expectations.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from experiments import ea, eb

from sremu import rng as _rng
from sremu.units import C0
from sremu.contracts import LatencyModel, RisCapabilities
from sremu.control.builtin import ProbeBandit
from sremu.control.controller import make_observation, setup_controller

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("fc", "n_expected"), [(1e9, 133), (2.4e9, 320)])
def test_ea_cell_count(fc: float, n_expected: int) -> None:
    n, k = ea.n_cells(fc)
    assert n == n_expected
    assert k == pytest.approx(C0 / fc / 4, rel=1e-15)


@pytest.mark.parametrize(("fc", "h", "loss"), sorted(set(ea.geometries())))
def test_ea_corridor_scenarios_validate(fc: float, h: float, loss: float) -> None:
    d = ea.corridor_scenario(fc, h, loss)
    res = ea.resolve(d)
    sc = res.scenario
    assert sc.ris[0].grid.n_u == ea.n_cells(fc)[0]
    assert sc.ris[0].pose.center_m[1] == h
    occ = sc.environment.occluders
    assert (len(occ) == 1 and occ[0].transmission_loss_db == loss) if loss else not occ
    surv = ea.resolve(ea.corridor_scenario(fc, h, loss, survey=True))
    assert surv.scenario.nodes[1].trajectory.model == "constant_velocity"


def test_ea_spec_points_cover_the_design() -> None:
    pts = ea.spec_points()
    geos = ea.geometries()
    assert len(geos) == 12
    assert len(pts) == 12 * (3 + 4 * len(ea.TAUS))
    assert ea.TAUS == ("0.01", "0.03", "0.1", "0.3", "1", "3", "10")
    assert ea.N_BATCHES * ea.STEPS_PER_BATCH * ea.DT_S == Fraction(333333, 10)


def test_tau_half_interpolation_and_censoring() -> None:
    taus = [0.01, 0.1, 1.0, 10.0]
    t, flag = ea.tau_half(taus, [0.9, 0.7, 0.3, 0.1])
    assert flag == "ok"
    assert math.log10(t) == pytest.approx(-1 + 0.5, abs=1e-12)  # midway in log10 tau
    assert ea.tau_half(taus, [0.4, 0.3, 0.2, 0.1]) == (0.01, "left")
    assert ea.tau_half(taus, [0.9, 0.8, 0.7, 0.6]) == (10.0, "right")


def test_decision_rule() -> None:
    assert ea._decide((0.2, 0.8), 0.1, 1.0) == "supported"
    assert ea._decide((1.2, 3.0), 0.1, 1.0) == "not supported"
    assert ea._decide((0.05, 0.5), 0.1, 1.0) == "inconclusive"


def test_nearest_coarse_matches_argmin_ties_low() -> None:
    pts = np.arange(1001) / 100
    g = _rng.generator(1, "trial", "wp11/nearest")
    x = np.concatenate([g.uniform(0, 10, 5000), [0.0, 10.0, 0.015, 9.995]])
    got = ea.nearest_coarse(x, 1001)
    ref = np.array([int(np.argmin((pts - v) ** 2)) for v in x])
    assert np.array_equal(got, ref)


def _caps(n: int) -> dict:
    return {
        "ris0": RisCapabilities(n, (n, 1), "element", ("0", "1"), "analytic:test", (1e9, 1e9), 1e6)
    }


def test_fast_probe_bandit_equals_probe_bandit() -> None:
    n, k = 8, 16
    g = _rng.generator(7, "trial", "wp11/bandit_codebook")
    cb = g.integers(0, 2, size=(k, n)).astype(np.int16)
    ctrls = []
    for cls in (ProbeBandit, ea.FastProbeBandit):
        c = cls(
            cb,
            "2e-2",
            c=1.0,
            panel="ris0",
            rx="mu.rx0",
            min_latency_s="1e-3",
            latency=LatencyModel(Fraction(1, 1000)),
        )
        setup_controller(c, _caps(n), 3)
        ctrls.append(c)
    gp = _rng.generator(7, "trial", "wp11/bandit_powers")
    arm_power = gp.uniform(-70, -50, k)
    seqs: list[list[int]] = [[], []]
    last = [None, None]
    for m in range(1500):
        for j, c in enumerate(ctrls):
            p = math.nan if last[j] is None else float(arm_power[last[j]] + gp.normal(0, 3) * 0)
            obs = make_observation(
                Fraction(m, 50), ris_state={"ris0": np.zeros(n)}, rx_power_dbm={"mu.rx0": p}
            )
            (a,) = c.decide(obs)
            arm = int(np.nonzero((cb == a.states).all(axis=1))[0][0])
            seqs[j].append(arm)
            last[j] = arm
        assert np.array_equal(ctrls[0].means(), ctrls[1].means(), equal_nan=True)
    assert seqs[0] == seqs[1]


def test_eb_packet_schedule() -> None:
    # timed: guard centres at m*slot + lead; asap/legacy: guard starts at m*slot
    assert eb.packet_start("timed", 500, 500) == 750
    assert eb.packet_start("asap", 500, 500) == 500
    assert eb.BODY == 2304 and eb.T_P == Fraction(2304, 10**6)


def test_eb_derived_expectations() -> None:
    d = eb.analytic_per("timed", Fraction(1, 2000), Fraction(0), Fraction(1, 2000))
    assert d["p_switch_in_packet"] == 0.0
    d = eb.analytic_per("asap", Fraction(1, 2000), Fraction(1, 1000), Fraction(0))
    assert d["p_switch_in_packet"] == pytest.approx(0.7)  # P(0.2 ms + U[0, 1 ms] > 0.5 ms)
    d = eb.analytic_per("asap", Fraction(1, 10000), Fraction(1, 1000), Fraction(0))
    assert d["p_switch_in_packet"] == 1.0
