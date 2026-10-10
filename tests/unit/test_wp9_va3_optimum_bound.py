"""VA-3: at N = 20 the exhaustive optimum is >= every controller at every sampled x
(product_paper.md section 5.1 "Oracles", WP9 acceptance in DESIGN.md section 14.1).

Scene: the WCL letter's Fig. 6 geometry (l = 3 m, h = 2 m, N = 20 cells of lambda/4 at
1 GHz, AP at the origin of the user line, MU moving along the line at 1 m/s), with the real
K1 physics (``ScenePropagation``) and two binary alphabets: the letter's ``on_off`` (OFF
absorbing) and ``ideal_antipodal`` (OFF not absorbing, so every configuration changes
every cell's contribution).

Oracle: at each of 29 sampled positions x = 0.1 ... 2.9 m the optimum is found by
exhaustion over all 2^20 configurations of the CSI (``base + sum of a subset of
d_n = Z[1, n] - Z[0, n]``). Every built-in controller of section 8.7 then decides at that
position, and the power of its decision is evaluated independently of the controller code
through the physics ``combine()`` narrowband sum. Accept: P_ctrl <= P_o (1 + 1e-12) for
every decision, and ExactOptimum equal to P_o to 1e-12 (relative, in |h|).

The GA (300, 0.01, 150) ratio to the optimum (W4's GA row) is reported as a junit
property, not gated.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PanelConfig, StreamSpec
from sremu.control.builtin import (
    CodebookSweep,
    DelayedController,
    ExactOptimum,
    Fixed,
    GeneticAlgorithm,
    GreedyQuantized,
    IdealDesigned,
    ProbeBandit,
    RisAbsent,
    TdmaAlternate,
    Urap,
    WclHeuristic,
)
from sremu.control.controller import (
    SnapshotChannel,
    controller_rng,
    csi_from_snapshot,
    make_observation,
    narrowband_coefficient,
    setup_controller,
)
from sremu.physics.geometry import ConstantVelocity, PanelGrid, Pose, Static
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation
from sremu.physics.response import builtin
from sremu.units import C0

pytestmark = pytest.mark.unit

FC = 1.0e9
K = C0 / FC / 4  # element width = length = lambda/4
N = 20
H_M = 2.0
X0 = Fraction(1, 10)  # MU abscissa at t = 0, m (speed 1 m/s: x = x0 + t)
DX = Fraction(1, 10)
N_POS = 29  # x = 0.1 ... 2.9 m
LINK = "ap.tx0->mu.rx0"
RX = "mu.rx0"
REL = 1e-12


def _scene(table: str) -> ScenePropagation:
    rate = Fraction(10_000)
    ap = NodeModel("ap", Static(np.zeros(3)), streams=(StreamSpec("ap.tx0", "tx", FC, rate),))
    mu = NodeModel(
        "mu",
        ConstantVelocity(np.array([float(X0), 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        streams=(StreamSpec(RX, "rx", FC, rate),),
    )
    pose = Pose(np.array([N * K / 2, H_M, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([1.0, 0, 0]))
    panel = PanelModel("ris0", PanelGrid(pose, N, 1, K, K), builtin(table))
    return ScenePropagation([ap, mu], [panel], horizon_s=3)


def _subset_sums(d: np.ndarray) -> np.ndarray:
    """All 2^N subset sums of (N,) ``d``; bit n of the index selects d[n]."""
    lo = d.size // 2

    def sums(v: np.ndarray) -> np.ndarray:
        idx = np.arange(1 << v.size)
        bits = (idx[:, None] >> np.arange(v.size)[None, :]) & 1
        return bits @ v

    return (sums(d[:lo])[:, None] + sums(d[lo:])[None, :]).reshape(-1)


def _brute_force_power(model: ScenePropagation, t: Fraction) -> float:
    """max over all 2^N binary configurations of |h|^2 (exhaustion on the CSI)."""
    v = csi_from_snapshot(model, model.snapshot(t), LINK)
    z = v.cascade["ris0"]
    assert z.shape == (2, N)
    base = v.h_direct + complex(np.sum(z[0]))
    return float(np.max(np.abs(base + _subset_sums(z[1] - z[0])))) ** 2


def _power(model: ScenePropagation, t: Fraction, states: np.ndarray, enabled: bool = True) -> float:
    """|h|^2 of a configuration through the physics combine() (not the controller code)."""
    cfg = {"ris0": PanelConfig("ris0", np.asarray(states, np.int16), enabled)}
    return abs(narrowband_coefficient(model, model.snapshot(t), cfg, LINK)) ** 2


def _positions(model: ScenePropagation, t: Fraction) -> dict[str, np.ndarray]:
    return {n: model.node_position(n, t) for n in ("ap", "mu")}


@pytest.fixture(scope="module", params=["on_off", "ideal_antipodal"])
def case(request):
    model = _scene(request.param)
    times = [i * DX for i in range(N_POS)]
    p_opt = [_brute_force_power(model, t) for t in times]
    return request.param, model, times, p_opt


def _controllers(model: ScenePropagation, table: str, times: list[Fraction]) -> list:
    ch = SnapshotChannel(model)
    cells = model.panel_geometry("ris0").cell_centres_m()
    gamma = builtin(table).gamma_at(FC)
    common = {"link": LINK, "channel": ch, "skip_unchanged": False}
    probe = [i * N_POS // 4 for i in range(4)]  # codebook: exact optima at 4 positions
    cb = np.stack([_opt_states(model, times[i]) for i in probe])
    rand_fixed = controller_rng(99, "va3/fixed").integers(0, 2, N).astype(np.int16)
    return [
        ExactOptimum(**common),
        WclHeuristic(**common, name="wcl_phasor"),
        WclHeuristic(**common, phase_source="path_length", cell_centres_m=cells, name="wcl_path"),
        GreedyQuantized(**common),
        IdealDesigned(gamma, **common),
        GeneticAlgorithm(300, 0.01, 150, **common),
        DelayedController(ExactOptimum(**common, name="delayed_opt"), "1/10", "3/10"),
        Urap(0.2, period_s=DX),
        Fixed(np.ones(N, np.int16), name="fixed_on"),
        Fixed(rand_fixed, name="fixed_random"),
        RisAbsent(),
        CodebookSweep(cb, period_s=DX),
        ProbeBandit(cb, "1/10"),
        TdmaAlternate([cb[1], cb[3]], slot_s=DX),
    ]


def _opt_states(model: ScenePropagation, t: Fraction) -> np.ndarray:
    v = csi_from_snapshot(model, model.snapshot(t), LINK)
    z = v.cascade["ris0"]
    base = v.h_direct + complex(np.sum(z[0]))
    i = int(np.argmax(np.abs(base + _subset_sums(z[1] - z[0]))))
    lo = N // 2  # flat index = (subset of d[:lo]) * 2^(N - lo) + (subset of d[lo:])
    a, b = i >> (N - lo), i & ((1 << (N - lo)) - 1)
    bits = np.concatenate([(a >> np.arange(lo)) & 1, (b >> np.arange(N - lo)) & 1])
    return bits.astype(np.int16)


def test_va3_exhaustive_optimum_bounds_every_controller_at_every_position(case, record_property):
    """VA-3: P_o (2^20 exhaustion) >= P of every built-in controller's decision at each of
    29 positions; ExactOptimum equals P_o (1e-12)."""
    table, model, times, p_opt = case
    ctrls = _controllers(model, table, times)
    caps = {"ris0": model.capabilities("ris0")}
    for c in ctrls:
        setup_controller(c, caps, seed=3)
    names = [c.name for c in ctrls]
    assert len(set(names)) == len(names)
    ratios: dict[str, list[float]] = {n: [] for n in names}
    last_power: dict[str, float | None] = dict.fromkeys(names)
    applied: dict[str, tuple[np.ndarray, bool]] = {n: (np.zeros(N, np.int16), True) for n in names}
    n_decisions = 0
    for i, t in enumerate(times):
        p_o = p_opt[i]
        for c in ctrls:
            draws = 16 if isinstance(c, Urap | CodebookSweep | ProbeBandit) else 1
            for j in range(draws):
                t_b = t + Fraction(j, 1000)  # extra barriers inside the position's slot
                lp = last_power[c.name]
                obs = make_observation(
                    t_b,
                    ris_state={"ris0": applied[c.name][0]},
                    positions=_positions(model, t),
                    # P_T = 0 dBm, so P_R = |h|^2 mW
                    rx_power_dbm=None if lp is None else {RX: 10.0 * np.log10(lp)},
                )
                acts = list(c.decide(obs))
                if not acts:  # Fixed / RisAbsent send once; the configuration stays applied
                    states, enabled = applied[c.name]
                else:
                    (a,) = acts
                    assert a.panel == "ris0"
                    if a.enabled is not None:
                        states, enabled = applied[c.name][0], bool(a.enabled)
                    else:
                        assert a.states is not None and a.states.shape == (N,)
                        states, enabled = a.states.copy(), True
                applied[c.name] = (states, enabled)
                # the power at this position, evaluated through the physics combine()
                p = _power(model, t, states, enabled)
                last_power[c.name] = p
                assert p <= p_o * (1.0 + REL), (table, c.name, float(t + X0), p / p_o)
                ratios[c.name].append(p / p_o)
                n_decisions += 1
        # the controller of the exact optimum reaches it
        assert abs(ratios["exact_optimum"][-1] - 1.0) <= 2 * REL, (table, float(t + X0))
    assert n_decisions == N_POS * (11 + 3 * 16)
    for n, r in ratios.items():
        record_property(f"VA3_{table}_{n}_P_over_Po_min", float(np.min(r)))
        record_property(f"VA3_{table}_{n}_P_over_Po_mean", float(np.mean(r)))
    record_property(f"W4_{table}_GA_P_over_Po_mean", float(np.mean(ratios["genetic_algorithm"])))
    # the comparison has teeth: the non-optimal controllers do fall short somewhere
    assert min(ratios["urap"]) < 0.99 and min(ratios["fixed_random"]) < 0.99
    assert min(ratios["delayed_opt"]) < 1.0 - 1e-6  # stale positions lose power


def test_va3_bound_has_teeth_a_config_above_the_claimed_optimum_is_caught(case):
    """Teeth: the exhaustive optimum is attained (the bound is tight) and any reduction of
    P_o by 1e-9 would flag the ExactOptimum decision, so the VA-3 comparison is not vacuous."""
    _table, model, times, p_opt = case
    for i in (0, N_POS // 2, N_POS - 1):
        s = _opt_states(model, times[i])
        p = _power(model, times[i], s)
        assert abs(p - p_opt[i]) <= 2 * REL * p_opt[i]
        assert not p <= p_opt[i] * (1.0 - 1e-9) * (1.0 + REL)
