"""WP9 controller helpers and built-in controllers (DESIGN.md sections 5.13, 8.7).

Oracles: exhaustive search over every configuration (ExactOptimum, IdealDesigned, the WCL
candidate search), explicit per-element loops (GreedyQuantized, control aggregation),
the ``combine()`` narrowband sum and the physics model's own ``csi_view`` (CSI helpers),
binomial bounds (Urap) and an independent UCB1 re-implementation (ProbeBandit).
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import (
    Controller,
    LatencyModel,
    ObservationSpec,
    PanelConfig,
    RisCapabilities,
    StreamSpec,
)
from sremu.control.builtin import (
    BUILTINS,
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
    build_controller,
    ga_optimize,
    optimum_codebook,
    wcl_activation,
    wcl_candidates,
    wcl_heuristic,
)
from sremu.control.controller import (
    BaseController,
    FunctionChannel,
    GridChannel,
    SnapshotChannel,
    aggregate_controls,
    controller_rng,
    csi_from_snapshot,
    expand_controls,
    make_observation,
    narrowband_coefficient,
    position_time,
    setup_controller,
    wrap_phase,
)
from sremu.errors import ConfigurationError
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation

pytestmark = pytest.mark.unit

LINK = "ap.tx0->mu.rx0"
GAMMA3 = (0.9 * np.exp(1j * 2.88), 0.7 + 0j, 0.5 * np.exp(-1j * 1.2))  # unequal amplitudes


def _stream(sid: str, d: str) -> StreamSpec:
    return StreamSpec(sid, d, 3.5e9, Fraction(10**6))  # type: ignore[arg-type]


def scene(
    *,
    n_u: int = 3,
    n_v: int = 2,
    gamma=GAMMA3,
    moving: bool = False,
    two_panels: bool = False,
) -> FakePropagation:
    """TX at the origin, RX 4 m away (optionally moving), one or two small panels."""
    nodes = [
        FakeNode("ap", (0.0, 0.0, 0.0), streams=(_stream("ap.tx0", "tx"),)),
        FakeNode(
            "mu",
            (4.0, 0.5, 0.0),
            (0.0, 1.0, 0.0) if moving else (0.0, 0.0, 0.0),
            streams=(_stream("mu.rx0", "rx"),),
        ),
    ]
    p = 0.0428
    panels = [
        FakePanel("ris0", (2.0, 2.0, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), n_u, n_v, p, p, gamma)
    ]
    if two_panels:
        panels.append(
            FakePanel("ris1", (2.0, -2.0, 0.3), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), 2, 2, p, p, gamma)
        )
    return FakePropagation(nodes, panels, horizon_s=10)


def caps(model, panels=("ris0",), n_controls=None) -> dict[str, RisCapabilities]:
    out = {}
    for pid in panels:
        geo = model.panel_geometry(pid)
        n_states = len(model.panels[pid].gamma)
        out[pid] = RisCapabilities(
            n_elements=geo.n_cells,
            layout=(geo.n_u, geo.n_v),
            control_granularity="element" if n_controls is None else "groups",
            states=tuple(f"s{i}" for i in range(n_states)),
            gamma_source="analytic:test",
            operating_band_hz=(3e9, 4e9),
            max_bandwidth_hz=1e7,
            n_controls=n_controls,
        )
    return out


def brute_force(view, panel, others=None) -> tuple[float, tuple[int, ...]]:
    """max |h| over every configuration of ``panel`` (others fixed), by exhaustion."""
    z = view.cascade[panel]
    best, arg = -1.0, ()
    for cfg in itertools.product(range(z.shape[0]), repeat=z.shape[1]):
        states = {panel: np.asarray(cfg, np.int16), **(others or {})}
        v = abs(view.evaluate(states))
        if v > best:
            best, arg = v, cfg
    return best, arg


def obs_at(t, model=None, *, positions=None, csi=None, ris_state=None, rx_power=None):
    rs = ris_state or {"ris0": np.zeros(6, np.int16)}
    return make_observation(
        t, ris_state=rs, positions=positions, csi_oracle=csi, rx_power_dbm=rx_power
    )


# =========================================================================== helpers


def test_csi_from_snapshot_matches_combine_for_random_configs(rng):
    model = scene(two_panels=True)
    snap = model.snapshot(Fraction(1, 3))
    view = csi_from_snapshot(model, snap, LINK)
    assert set(view.cascade) == {"ris0", "ris1"}
    for _ in range(50):
        s0 = rng.integers(-1, 3, 6).astype(np.int16)  # -1 = dead (gamma_dead = 0)
        s1 = rng.integers(-1, 3, 4).astype(np.int16)
        cfg = {"ris0": PanelConfig("ris0", s0), "ris1": PanelConfig("ris1", s1)}
        ref = narrowband_coefficient(model, snap, cfg, LINK)
        got = view.evaluate({"ris0": s0, "ris1": s1})
        assert abs(got - ref) <= 1e-13 * abs(ref)


def test_csi_from_snapshot_equals_physics_csi_view(repo_root):
    from sremu.physics.propagation import from_scenario

    model = from_scenario(repo_root / "scenarios" / "quickstart.yaml")
    snap = model.snapshot(0)
    lk = model.links()[0]
    mine = csi_from_snapshot(model, snap, lk)
    ref = model.csi_view(snap, lk)
    assert mine.h_direct == ref.h_direct
    np.testing.assert_array_equal(mine.cascade["ris0"], ref.cascade["ris0"])
    assert mine.fc_hz == ref.fc_hz == model.link_fc_hz(lk)


def test_aggregate_and_expand_controls_against_loops(rng):
    z = rng.standard_normal((3, 8)) + 1j * rng.standard_normal((3, 8))
    coc = np.array([0, 0, 1, 2, 2, 2, 1, 0])
    agg = aggregate_controls(z, coc, 3)
    for s in range(3):
        for g in range(3):
            ref = sum(z[s, n] for n in range(8) if coc[n] == g)
            assert abs(agg[s, g] - ref) <= 1e-15 * 8
    assert aggregate_controls(z, None, 8) is not None
    np.testing.assert_array_equal(aggregate_controls(z, None, 8), z)
    with pytest.raises(ConfigurationError):
        aggregate_controls(z, None, 3)
    with pytest.raises(ConfigurationError):
        aggregate_controls(z, np.array([0, 1, 2, 3, 0, 0, 0, 0]), 3)
    np.testing.assert_array_equal(
        expand_controls(np.array([5, 6, 7]), coc), [5, 5, 6, 7, 7, 7, 6, 5]
    )


def test_wrap_phase_range_and_values():
    x = np.array([0.0, math.pi, -math.pi, 3 * math.pi, 2 * math.pi + 0.5, -0.5])
    w = wrap_phase(x)
    assert np.all((w > -math.pi) & (w <= math.pi))
    np.testing.assert_allclose(w, [0.0, math.pi, math.pi, math.pi, 0.5, -0.5], atol=1e-12)


def test_grid_channel_from_model_and_nearest_lookup():
    model = scene(moving=True)  # mu moves along +y at 1 m/s from (4, 0.5, 0)
    times = [Fraction(i, 10) for i in range(11)]
    ch = GridChannel.from_model(model, "mu", times)
    np.testing.assert_allclose(ch.points[:, 1], 0.5 + np.arange(11) / 10, atol=1e-15)
    assert ch.index({"mu": np.array([4.0, 0.5 + 0.34, 0.0])}) == 3
    assert ch.index({"mu": np.array([4.0, 0.5 + 0.36, 0.0])}) == 4
    v = ch.csi(LINK, {"mu": ch.points[7]}, Fraction(0))
    ref = csi_from_snapshot(model, model.snapshot(times[7]), LINK)
    assert v.h_direct == ref.h_direct
    np.testing.assert_array_equal(v.cascade["ris0"], ref.cascade["ris0"])
    with pytest.raises(ConfigurationError):
        ch.index({"ap": np.zeros(3)})


def test_snapshot_channel_uses_the_position_time():
    model = scene(moving=True)
    ch = SnapshotChannel(model)
    spec = ObservationSpec(fields=("positions",), position_delay_s="1/4")
    obs = obs_at(Fraction(1, 2))
    t_pos = position_time(obs, spec)
    assert t_pos == Fraction(1, 4)
    assert position_time(obs_at(Fraction(1, 10)), spec) == 0
    v = ch.csi(LINK, None, t_pos)
    ref = csi_from_snapshot(model, model.snapshot(Fraction(1, 4)), LINK)
    assert v.h_direct == ref.h_direct


def test_base_controller_validation_and_protocol():
    with pytest.raises(TypeError):
        BaseController("x", period_s=0.001)  # floats are rejected (section 3.1)
    with pytest.raises(ConfigurationError):
        BaseController("x", min_latency_s="1e-3", latency=LatencyModel(fixed_s="5e-4"))
    with pytest.raises(ConfigurationError):
        BaseController("x", period_s=0)
    c = Urap(0.3, min_latency_s="1e-4")
    assert c.latency.fixed_s == Fraction(1, 10_000)
    ex = ExactOptimum(link=LINK)
    assert "positions" in ex.spec.fields and "csi_oracle" in ex.spec.fields
    model = scene()
    for ctrl in (
        Fixed([0] * 6),
        RisAbsent(),
        Urap(),
        ExactOptimum(),
        GreedyQuantized(),
        WclHeuristic(),
        IdealDesigned((1, -1)),
        GeneticAlgorithm(),
        CodebookSweep(4),
        ProbeBandit(4),
        TdmaAlternate([[0] * 6, [1] * 6], slot_s="1e-3"),
        DelayedController(ExactOptimum(channel=SnapshotChannel(model)), "1e-1"),
    ):
        assert isinstance(ctrl, Controller), type(ctrl).__name__
    with pytest.raises(ConfigurationError):
        Urap().decide(obs_at(0))  # setup() not called


# =========================================================================== simple controllers


def test_fixed_and_ris_absent_send_once():
    model = scene()
    f = Fixed([2, 1, 0, 0, 1, 2])
    setup_controller(f, caps(model), seed=1)
    (a,) = f.decide(obs_at(0))
    np.testing.assert_array_equal(a.states, [2, 1, 0, 0, 1, 2])
    assert a.panel == "ris0" and a.states.dtype == np.int16
    assert f.decide(obs_at(Fraction(1, 1000))) == []
    with pytest.raises(ConfigurationError):
        setup_controller(Fixed([0, 1]), caps(model), seed=1)  # wrong length
    r = RisAbsent()
    setup_controller(r, caps(model), seed=1)
    (b,) = r.decide(obs_at(0))
    assert b.enabled is False and b.states is None
    assert r.decide(obs_at(1)) == []


def test_urap_activation_fraction_is_binomial():
    model = scene(n_u=12, n_v=10)
    for p in (0.0, 0.2, 1.0):
        u = Urap(p)
        setup_controller(u, caps(model), seed=5)
        draws = np.stack([u.decide(obs_at(0))[0].states for _ in range(200)])
        assert set(np.unique(draws)) <= {0, 1}
        n = draws.size
        frac = draws.mean()
        assert abs(frac - p) <= 5 * math.sqrt(p * (1 - p) / n) + 1e-12
    with pytest.raises(ConfigurationError):
        Urap(1.5)


# =========================================================================== position-driven


def test_exact_optimum_equals_exhaustive_search(rng):
    model = scene()
    ch = SnapshotChannel(model)
    ctrl = ExactOptimum(link=LINK, channel=ch, skip_unchanged=False)
    setup_controller(ctrl, caps(model), seed=1)
    for t in (Fraction(0), Fraction(1, 7)):
        (a,) = ctrl.decide(obs_at(t))
        view = csi_from_snapshot(model, model.snapshot(t), LINK)
        best, _ = brute_force(view, "ris0")
        got = abs(view.evaluate({"ris0": a.states}))
        assert abs(got - best) <= 1e-12 * best
        assert abs(ctrl.last["abs_h"] - best) <= 1e-12 * best


def test_exact_optimum_keeps_other_panels_at_their_observed_state():
    model = scene(two_panels=True)
    view = csi_from_snapshot(model, model.snapshot(0), LINK)
    other = np.array([2, 0, 1, -1], np.int16)
    ctrl = ExactOptimum(link=LINK, panel="ris0", channel=SnapshotChannel(model))
    setup_controller(ctrl, caps(model, ("ris0", "ris1")), seed=1)
    obs = obs_at(0, ris_state={"ris0": np.zeros(6, np.int16), "ris1": other})
    (a,) = ctrl.decide(obs)
    best, _ = brute_force(view, "ris0", {"ris1": other})
    got = abs(view.evaluate({"ris0": a.states, "ris1": other}))
    assert abs(got - best) <= 1e-12 * best


def test_exact_optimum_with_control_groups_and_caching():
    model = scene()
    coc = np.array([0, 0, 1, 1, 2, 2])
    times = [Fraction(i, 5) for i in range(3)]
    grid = GridChannel.from_model(scene(moving=True), "mu", times)
    ctrl = ExactOptimum(link=LINK, channel=grid, control_of_cell=coc)
    setup_controller(ctrl, caps(model, n_controls=3), seed=1)
    pos = {"mu": grid.points[1]}
    (a,) = ctrl.decide(obs_at(0, positions=pos))
    assert a.states.shape == (3,)
    view = grid.views[1][LINK]
    best = max(
        abs(view.evaluate({"ris0": expand_controls(np.asarray(c), coc)}))
        for c in itertools.product(range(3), repeat=3)
    )
    assert abs(abs(view.evaluate({"ris0": expand_controls(a.states, coc)})) - best) <= 1e-12 * best
    assert ctrl.decide(obs_at(Fraction(1, 1000), positions=pos)) == []  # unchanged: skipped
    assert ctrl.last["grid_index"] == 1 and 1 in ctrl._cache


def test_snapshot_channel_with_position_noise_is_rejected_at_setup():
    """A SnapshotChannel ignores the reported positions, so a noisy-position spec would be
    silently noise-free; setup must refuse it (GridChannel uses the noisy positions)."""
    model = scene(moving=True)
    noisy = ObservationSpec(fields=("positions",), position_noise_m=0.05)
    for ctrl in (
        ExactOptimum(link=LINK, channel=SnapshotChannel(model), spec=noisy),
        WclHeuristic(link=LINK, channel=SnapshotChannel(model), spec=noisy),
        DelayedController(
            GreedyQuantized(link=LINK, channel=SnapshotChannel(model), spec=noisy), "1/10"
        ),
    ):
        with pytest.raises(ConfigurationError, match="position_noise_m"):
            setup_controller(ctrl, caps(model), seed=1)
    grid = GridChannel.from_model(model, "mu", [Fraction(i, 10) for i in range(3)])
    ok = ExactOptimum(link=LINK, channel=grid, spec=noisy)
    setup_controller(ok, caps(model), seed=1)  # uses the reported (noisy) positions
    (a,) = ok.decide(obs_at(0, positions={"mu": grid.points[2] + 0.01}))
    best, _ = brute_force(grid.views[2][LINK], "ris0")
    assert abs(abs(grid.views[2][LINK].evaluate({"ris0": a.states})) - best) <= 1e-12 * best


def test_exact_optimum_from_oracle_csi_without_a_channel():
    model = scene()
    view = csi_from_snapshot(model, model.snapshot(0), LINK)
    ctrl = ExactOptimum()
    setup_controller(ctrl, caps(model), seed=1)
    (a,) = ctrl.decide(obs_at(0, csi={LINK: view}))
    best, _ = brute_force(view, "ris0")
    assert abs(abs(view.evaluate({"ris0": a.states})) - best) <= 1e-12 * best
    with pytest.raises(ConfigurationError):
        ctrl.decide(obs_at(0))  # neither a channel nor csi_oracle


def test_greedy_quantized_is_per_element_projection_argmax():
    model = scene()
    view = csi_from_snapshot(model, model.snapshot(0), LINK)
    ctrl = GreedyQuantized(link=LINK, channel=FunctionChannel(lambda lk, p, t: view))
    setup_controller(ctrl, caps(model), seed=1)
    (a,) = ctrl.decide(obs_at(0))
    z = view.cascade["ris0"]
    ref_phase = math.atan2(view.h_direct.imag, view.h_direct.real)
    for n in range(z.shape[1]):
        proj = [
            (z[s, n] * complex(math.cos(ref_phase), -math.sin(ref_phase))).real for s in range(3)
        ]
        assert a.states[n] == int(np.argmax(proj))


def test_ideal_designed_deploys_the_antipodal_design_indices():
    gam_true = np.array([np.exp(1j * np.deg2rad(49.64)), 0.7 * np.exp(1j * np.deg2rad(-153.77))])
    model = scene(gamma=tuple(gam_true))
    view = csi_from_snapshot(model, model.snapshot(0), LINK)
    ctrl = IdealDesigned(gam_true, link=LINK, channel=SnapshotChannel(model))
    setup_controller(ctrl, caps(model), seed=1)
    (a,) = ctrl.decide(obs_at(0))
    c = view.cascade["ris0"][0] / gam_true[0]  # Gamma-free factor
    best, arg = -1.0, None
    for cfg in itertools.product((0, 1), repeat=6):
        v = abs(view.h_direct + sum((1 if s == 0 else -1) * c[n] for n, s in enumerate(cfg)))
        if v > best:
            best, arg = v, cfg
    np.testing.assert_array_equal(a.states, arg)
    # with the true alphabet antipodal, IdealDesigned equals the exact optimum
    model2 = scene(gamma=(1 + 0j, -1 + 0j))
    i2 = IdealDesigned((1, -1), link=LINK, channel=SnapshotChannel(model2))
    e2 = ExactOptimum(link=LINK, channel=SnapshotChannel(model2))
    for c_ in (i2, e2):
        setup_controller(c_, caps(model2), seed=1)
    v2 = csi_from_snapshot(model2, model2.snapshot(0), LINK)
    h_i = abs(v2.evaluate({"ris0": i2.decide(obs_at(0))[0].states}))
    h_e = abs(v2.evaluate({"ris0": e2.decide(obs_at(0))[0].states}))
    assert abs(h_i - h_e) <= 1e-12 * h_e


# --------------------------------------------------------------------------- WCL heuristic


def test_wcl_candidates_default_grid_is_the_letter_count():
    cands = wcl_candidates(4, 3)
    assert len(cands) == 4 * 3 * 4 // 2 == 24
    thetas = sorted({round(t, 12) for t, _ in cands})
    np.testing.assert_allclose(thetas, [math.pi / 8, math.pi / 4, 3 * math.pi / 8, math.pi / 2])
    fr = [p / t for t, p in cands[:6]]
    np.testing.assert_allclose(fr, [1 / 2, 1 / 3, 2 / 3, 1 / 4, 2 / 4, 3 / 4])
    assert len(wcl_candidates(thetas_rad=[1.0], psi_fracs=[0.5])) == 1
    with pytest.raises(ConfigurationError):
        wcl_candidates(thetas_rad=[math.pi])


def test_wcl_activation_set_hand_computed():
    delta = np.array([0.0, 0.1, -0.1, 0.7, -0.7, 0.8, -0.8, math.pi, 2 * math.pi + 0.05])
    m = wcl_activation(delta, math.pi / 2, math.pi / 4)  # arc (-pi/4, pi/4) around phi(x)
    np.testing.assert_array_equal(m, [1, 1, 1, 1, 1, 0, 0, 0, 1])
    m = wcl_activation(delta, math.pi / 2, math.pi / 8)  # arc (-pi/8, 3 pi/8)
    np.testing.assert_array_equal(m, [1, 1, 1, 1, 0, 1, 0, 0, 1])


def test_wcl_heuristic_is_the_best_candidate_by_exhaustion(rng):
    n = 30
    z1 = 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    z = np.stack([np.zeros(n, complex), z1])
    z0 = 0.3 * np.exp(1j * 0.4)
    cands = wcl_candidates(4, 3)
    val, s, best = wcl_heuristic(z0, z, cands)
    delta = wrap_phase(np.angle(z0) - np.angle(z1))
    powers = [abs(z0 + z1[wcl_activation(delta, t, p)].sum()) for t, p in cands]
    assert abs(val - max(powers)) <= 1e-15 * val
    assert best == cands[int(np.argmax(powers))]
    np.testing.assert_array_equal(s, wcl_activation(delta, *best).astype(np.int16))


def test_wcl_path_length_phases_equal_phasor_phases_for_a_delay_only_model():
    """With the letter's phasors a e^{-j phi} (no K1 factor j) the two phase sources agree."""
    from sremu.contracts import CsiView
    from sremu.units import C0

    fc = 1e9
    k = C0 / fc / 4
    cells = np.stack([(np.arange(20) + 0.5) * k, np.full(20, 2.0), np.zeros(20)], axis=1)
    ap, mu = np.zeros(3), np.array([1.3, 0.0, 0.0])
    dn = np.linalg.norm(cells - ap, axis=1) + np.linalg.norm(cells - mu, axis=1)
    kk = 2 * math.pi * fc / C0
    z1 = 0.01 / dn * np.exp(-1j * kk * dn)
    z0 = 0.3 / 1.3 * np.exp(-1j * kk * 1.3)
    view = CsiView(LINK, Fraction(0), fc, z0, {"ris0": np.stack([np.zeros(20, complex), z1])})
    ch = FunctionChannel(lambda lk, p, t: view)
    geo_caps = {
        "ris0": RisCapabilities(
            20, (20, 1), "element", ("off", "on"), "analytic:on_off", (fc, fc), 1e6
        )
    }
    pos = {"ap": ap, "mu": mu}
    out = []
    for src in ("phasor", "path_length"):
        h = WclHeuristic(link=LINK, channel=ch, phase_source=src, cell_centres_m=cells)
        setup_controller(h, geo_caps, seed=0)
        out.append(
            (
                h.decide(obs_at(0, positions=pos, ris_state={"ris0": np.zeros(20)}))[0].states,
                h.last["theta"],
                h.last["psi"],
            )
        )
    np.testing.assert_array_equal(out[0][0], out[1][0])
    assert out[0][1:] == out[1][1:]


# --------------------------------------------------------------------------- GA


def test_ga_is_bounded_by_the_optimum_and_reaches_it_on_a_small_problem(rng):
    n = 10
    z1 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    z = np.stack([np.zeros(n, complex), z1])
    z0 = 1.0 + 0.5j
    opt = max(abs(z0 + z1[np.asarray(c, bool)].sum()) for c in itertools.product((0, 1), repeat=n))
    val, s = ga_optimize(z0, z, controller_rng(3, "ga"), pop=60, mut=0.05, iters=60)
    assert val <= opt * (1 + 1e-12)
    assert abs(val - abs(z0 + z1[s.astype(bool)].sum())) <= 1e-12 * opt
    assert val >= 0.999 * opt  # 2^10 configurations: the GA should find the optimum


# =========================================================================== codebooks, bandit


def test_codebook_sweep_cycles_and_credits_the_previous_entry():
    model = scene()
    cb = np.array([[0] * 6, [1] * 6, [2] * 6])
    c = CodebookSweep(cb)
    setup_controller(c, caps(model), seed=1)
    played = []
    for m in range(7):
        (a,) = c.decide(obs_at(Fraction(m, 1000), rx_power={"mu.rx0": float(-10 * (m % 3))}))
        played.append(int(a.states[0]))
    assert played == [0, 1, 2, 0, 1, 2, 0]
    # power observed at barrier m belongs to the entry played at m - 1
    # entry 0 (m = 0, 3) is credited at m = 1, 4 (-10 dBm); entry 2 (m = 2, 5) at m = 3, 6
    assert c.powers_mw[0] == pytest.approx([0.1, 0.1], rel=1e-15)
    assert c.powers_mw[1] == pytest.approx([0.01, 0.01], rel=1e-15)
    assert c.powers_mw[2] == pytest.approx([1.0, 1.0], rel=1e-15)
    assert c.best_index() == 2
    with pytest.raises(ConfigurationError):
        setup_controller(CodebookSweep(np.zeros((2, 4), int)), caps(model), seed=1)


def _ucb1_reference(rewards_of_arm, k: int, n_rounds: int, c: float) -> list[int]:
    """Independent UCB1 with rewards normalized by the largest reward seen so far."""
    counts = [0] * k
    raw: list[tuple[int, float]] = []
    plays = []
    last = None
    for _ in range(n_rounds):
        if last is not None:
            counts[last] += 1
            raw.append((last, rewards_of_arm(last)))
        if 0 in counts:
            arm = counts.index(0)
        else:
            top = max(r for _, r in raw)
            sums = [0.0] * k
            for a, r in raw:
                sums[a] += r / top
            n = sum(counts)
            idx = [
                sums[i] / counts[i] + c * math.sqrt(2 * math.log(n) / counts[i]) for i in range(k)
            ]
            arm = max(range(k), key=lambda i: (idx[i], -i))
        plays.append(arm)
        last = arm
    return plays


def test_probe_bandit_matches_an_independent_ucb1():
    model = scene()
    k = 5
    p_mw = [1.0, 3.0, 2.0, 0.5, 2.9]
    bandit = ProbeBandit(k, "2e-2")
    assert bandit.period_s == Fraction(1, 50) and bandit.spec.window_s == Fraction(1, 100)
    setup_controller(bandit, caps(model), seed=9)
    cb = bandit.codebook.copy()
    arm_of = {tuple(r): i for i, r in enumerate(cb)}
    assert len(arm_of) == k  # distinct random entries for this seed
    plays = []
    last = None
    for m in range(60):
        rx = {} if last is None else {"mu.rx0": 10 * math.log10(p_mw[last])}
        (a,) = bandit.decide(obs_at(Fraction(m, 50), rx_power=rx))
        last = arm_of[tuple(a.states)]
        plays.append(last)
    assert plays == _ucb1_reference(lambda a: p_mw[a], k, 60, 1.0)
    assert plays[:k] == list(range(k))
    assert np.bincount(plays[-30:], minlength=k).argmax() == 1


# =========================================================================== TDMA, delay


def test_tdma_alternate_targets_and_switch_times():
    model = scene()
    cfgs = [[0] * 6, [1] * 6, [2] * 6]
    timed = TdmaAlternate(cfgs, slot_s="1e-3", lead_s="2e-4", min_latency_s="1e-4")
    asap = TdmaAlternate(cfgs, slot_s="1e-3", timing="asap", name="t2")
    for c in (timed, asap):
        setup_controller(c, caps(model), seed=1)
    for m in range(5):
        t_b = Fraction(m, 1000)
        (a,) = timed.decide(obs_at(t_b))
        assert int(a.states[0]) == m % 3 and a.at_s == t_b + Fraction(1, 5000)
        assert timed.guard_centre_s(m) == a.at_s
        (b,) = asap.decide(obs_at(t_b))
        assert b.at_s is None and int(b.states[0]) == m % 3
    with pytest.raises(ConfigurationError):
        TdmaAlternate(cfgs, slot_s="1e-3", lead_s="1e-5", min_latency_s="1e-4")


def test_delayed_controller_decides_on_old_positions():
    model = scene(moving=True)
    inner = ExactOptimum(link=LINK, channel=SnapshotChannel(model), skip_unchanged=False)
    d = DelayedController(inner, "1/10", "3/10")
    assert d.period_s == Fraction(1, 10) and d.name == inner.name
    assert d.spec.position_delay_s == Fraction(3, 10) and inner.spec is d.spec
    assert "positions" in d.spec.fields
    setup_controller(d, caps(model), seed=1)
    for t in (Fraction(0), Fraction(1, 5), Fraction(9, 10)):
        (a,) = d.decide(obs_at(t))
        t_pos = max(Fraction(0), t - Fraction(3, 10))
        view = csi_from_snapshot(model, model.snapshot(t_pos), LINK)
        best, _ = brute_force(view, "ris0")
        assert abs(abs(view.evaluate({"ris0": a.states})) - best) <= 1e-12 * best
    with pytest.raises(ConfigurationError):
        DelayedController(inner, 0)


# =========================================================================== factory


def test_build_controller_from_mappings_and_scenario_models():
    from sremu.scenario.models import ControllerModel

    model = scene()
    spec = {
        "name": "u1",
        "kind": "builtin",
        "impl": "Urap",
        "period_s": "1e-2",
        "min_latency_s": "1e-3",
        "latency": {"fixed_s": "1e-3", "jitter": "uniform:0,1e-3"},
        "observation": {"fields": ["rx_power_dbm"], "window_s": "5e-3"},
        "params": {"p": 0.3},
    }
    for s in (spec, ControllerModel.model_validate(spec)):
        c = build_controller(s)
        assert isinstance(c, Urap) and c.p == 0.3 and c.name == "u1"
        assert c.period_s == Fraction(1, 100) and c.min_latency_s == Fraction(1, 1000)
        assert c.latency.jitter == "uniform:0,1e-3" and c.spec.window_s == Fraction(1, 200)
    ex = build_controller(
        {"name": "opt", "impl": "ExactOptimum", "period_s": "1e-1", "params": {"link": LINK}},
        model=model,
    )
    assert isinstance(ex.channel, SnapshotChannel) and ex.channel.model is model
    dl = build_controller(
        {
            "name": "dly",
            "impl": "DelayedController",
            "period_s": "1e-2",
            "params": {
                "inner": {"impl": "WclHeuristic", "params": {"link": LINK}},
                "tau_sui_s": "1/2",
                "report_delay_s": "1e-3",
            },
        },
        model=model,
    )
    assert isinstance(dl, DelayedController) and isinstance(dl.inner, WclHeuristic)
    assert dl.period_s == Fraction(1, 2) and dl.name == "dly"
    pb = build_controller(
        {"name": "pb", "impl": "ProbeBandit", "period_s": "1/50", "params": {"codebook": 4}}
    )
    assert pb.period_s == Fraction(1, 50)
    td = build_controller(
        {
            "name": "td",
            "impl": "TdmaAlternate",
            "period_s": "1e-3",
            "params": {"configs": [[0] * 6, [1] * 6], "timing": "asap"},
        }
    )
    assert td.period_s == Fraction(1, 1000)
    with pytest.raises(ConfigurationError):
        build_controller({"name": "x", "impl": "Nope"})
    assert set(BUILTINS) >= {
        "Fixed",
        "RisAbsent",
        "Urap",
        "WclHeuristic",
        "ExactOptimum",
        "DelayedController",
        "TdmaAlternate",
        "ProbeBandit",
        "IdealDesigned",
    }


def test_optimum_codebook_rows_are_exact_optima():
    model = scene(moving=True)
    views = [csi_from_snapshot(model, model.snapshot(Fraction(i, 4)), LINK) for i in range(3)]
    cb = optimum_codebook(views, "ris0")
    assert cb.shape == (3, 6) and cb.dtype == np.int16
    for v, row in zip(views, cb, strict=True):
        best, _ = brute_force(v, "ris0")
        assert abs(abs(v.evaluate({"ris0": row})) - best) <= 1e-12 * best
