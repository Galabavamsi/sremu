"""WP5: geometric Doppler, tiles/interpolation, invariance and the T_u/tile policy.

PH-17 (T9 geometry of ``pm_checks.py``), PH-18 (auto T_u and tiles vs brute force per
element per sample, plus ``verify_interp``), PH-19 (T_u and chunk invariance) on
``FakePropagation``. Oracles: closed-form path lengths of constant-velocity nodes, the
per-element K1-like sum of ``FakePanel`` (``_render_support.ris_brute``) and the section 4.4
error-budget formulas.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numba
import numpy as np
import pytest

from sremu.channel.interp import (
    EPS_REL,
    PanelGeometry,
    choose_policy,
    interval_grid,
    residual_bound_rel,
    t_curv_s,
    tile_side_max_m,
    verify_interp,
    weights,
)
from sremu.channel.render_numpy import get_backend
from sremu.contracts import FDSpec, LinkJob, MotionBounds, PanelConfig, PathKind, SegmentJob
from sremu.testing.scenes import doppler_t9
from sremu.units import C0

from ._render_support import (
    RIS_KINDS,
    SpanSpec,
    const_ring,
    focus_states,
    lab_walk,
    render_span,
    ris_brute,
    vehicle,
    wcl_corridor,
)

pytestmark = pytest.mark.runtime

FC_T9 = 3.5e9
LINK_T9 = "tx.tx0->rx.rx0"


def _inst_freq(y: np.ndarray, fs: float) -> np.ndarray:
    return np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * np.pi)


# --------------------------------------------------------------------------- PH-17


def test_ph17_los_doppler_and_phase_over_one_second(backend_name: str) -> None:
    """Radial LOS at 30 m/s, 3.5 GHz: phase equals -2 pi f_c d(t)/c to 1e-6 rad over 1 s and
    the instantaneous frequency equals v f_c/c = 350.242 Hz to 0.001 Hz."""
    model = doppler_t9()
    fs = Fraction(10_000)
    pol = choose_policy(model.motion_bounds(), [], {LINK_T9: FC_T9}, fs)
    assert pol.t_u_s is not None
    n = 10_000
    ring = const_ring("tx.tx0", fs, n + 64)
    spec = SpanSpec(model, {}, LINK_T9, FC_T9, ring, fs, pol.t_u_s)
    y = render_span(spec, 0, n, backend_name)
    t = np.arange(n) / 1e4
    d = 50.0 - 30.0 * t  # exact constant-velocity path length (m)
    cyc = FC_T9 * d / C0
    oracle = np.exp(-2j * np.pi * (cyc - np.rint(cyc)))
    phase_err = np.abs(np.angle(y * np.conj(oracle)))
    assert phase_err.max() <= 1e-6
    f_d = 30.0 * FC_T9 / C0
    assert f_d == pytest.approx(350.242, abs=0.001)  # PH-17 oracle value
    fi = _inst_freq(y, 1e4)
    assert np.max(np.abs(fi - f_d)) <= 0.001
    assert abs(fi.mean() - 350.242) <= 0.001


def test_ph17_teeth_block_constant_geometry_aliases_to_minus148hz() -> None:
    """v2 staircase (geometry frozen per 2 ms tick, T9 at 10 kS/s for 0.2 s): the same
    frequency measurement reports about -148.3 Hz instead of +350.2 Hz."""
    fs = Fraction(10_000)
    ring = const_ring("tx.tx0", fs, 2100)
    be = get_backend("numpy_ref")
    y = np.zeros(2000, np.complex128)
    for b in range(100):  # 2 ms blocks = 20 samples, degenerate (frozen) intervals
        tau = np.array([(50.0 - 30.0 * b * 0.002) / C0])
        lj = LinkJob(
            LINK_T9,
            ring,
            FC_T9,
            0.0,
            np.ones(1, np.complex128),
            np.ones(1, np.complex128),
            tau,
            tau.copy(),
            np.zeros(1, np.int16),
            FDSpec.integer(),
        )
        job = SegmentJob(
            "rx.rx0", fs, 20 * b, 20 * b + 20, Fraction(b, 500), Fraction(b, 500), [lj]
        )
        be.render(job, y[20 * b : 20 * b + 20])
    f_mean = float(_inst_freq(y, 1e4).mean())
    assert f_mean == pytest.approx(-148.3, abs=1.0)
    assert abs(f_mean - 350.242) > 0.001  # fails the PH-17 check


def test_ph17_ris_path_doppler_at_t0_and_chord_phase_bound(backend_name: str) -> None:
    """T9 RIS path (panel at (25, 10, 0), tile reference at its centre): rendered like a path,
    f_D(t0) = 325.192 Hz +- 0.01 Hz, and the phase error stays within k*rdd*T_u**2/8 + 1e-6."""
    model = doppler_t9(with_panel=True, tiles=1)
    fs = Fraction(10_000)
    t_u = Fraction(1, 1000)  # the scenario may only lower T_u (auto is about 1.56 ms)
    geo = PanelGeometry("ris0", 0.5, 0.5, 12, 12, FC_T9, 1)
    pol = choose_policy(model.motion_bounds(), [geo], {LINK_T9: FC_T9}, fs)
    assert pol.t_u_s is not None and t_u <= pol.t_u_s
    n = 10_000
    ring = const_ring("tx.tx0", fs, n + 64)
    be = get_backend(backend_name)  # type: ignore[arg-type]
    y = np.zeros(n, np.complex128)
    for m in range(n // 10):
        ta, tb = m * t_u, (m + 1) * t_u
        da = model.snapshot(ta).ris["ris0"].tile_delay_s[:, 0]
        db = model.snapshot(tb).ris["ris0"].tile_delay_s[:, 0]
        one = np.ones(1, np.complex128)
        lj = LinkJob(
            LINK_T9,
            ring,
            FC_T9,
            0.0,
            one,
            one.copy(),
            da.copy(),
            db.copy(),
            np.zeros(1, np.int16),
            FDSpec.integer(),
        )
        be.render(
            SegmentJob("rx.rx0", fs, 10 * m, 10 * m + 10, ta, tb, [lj]), y[10 * m : 10 * m + 10]
        )
    c = np.array([25.0, 10.0, 0.0])
    t = np.arange(n) / 1e4
    rx = np.stack([50.0 - 30.0 * t, np.zeros(n), np.zeros(n)], axis=1)
    length = np.linalg.norm(c) + np.linalg.norm(rx - c, axis=1)
    cyc = FC_T9 * length / C0
    exact = np.exp(-2j * np.pi * (cyc - np.rint(cyc)))
    k = 2 * np.pi * FC_T9 / C0
    rdd_max = 30.0**2 / 10.0  # v**2 / closest approach to the panel centre (10 m)
    bound = k * rdd_max * float(t_u) ** 2 / 8 + 1e-6
    assert np.max(np.abs(np.angle(y * np.conj(exact)))) <= bound
    # f_D(t0): the chord is exact at the knots; fit the knot phases and differentiate at 0
    knots = np.arange(0, 31) * 10
    ph = np.unwrap(np.angle(y[knots]))
    coef = np.polyfit(t[knots], ph, 4)
    f0 = np.polyder(coef)[-1] / (2 * np.pi)
    u = (c - np.array([50.0, 0.0, 0.0])) / np.linalg.norm(c - np.array([50.0, 0.0, 0.0]))
    f_closed = float(u @ np.array([-30.0, 0.0, 0.0])) * FC_T9 / C0
    assert f_closed == pytest.approx(325.192, abs=0.001)  # PH-17 oracle value
    assert f0 == pytest.approx(f_closed, abs=0.01)


# --------------------------------------------------------------------------- PH-18


@pytest.mark.parametrize("scene_fn", [lab_walk, vehicle, wcl_corridor], ids=lambda f: f.__name__)
@pytest.mark.parametrize("policy_rate", ["render", "1e6"])
def test_ph18_auto_tiles_vs_brute_force_per_element(scene_fn, policy_rate: str) -> None:
    """Auto (T_u, tiles) renderer vs per-element per-sample brute force: |dh|/|h| <= 1e-3;
    verify_interp reports the same maximum to 10%."""
    sc = scene_fn()
    m1 = sc.build(1, 1)
    rate = sc.f_s if policy_rate == "render" else Fraction(10**6)
    pol = choose_policy(m1.motion_bounds(), [sc.geometry(m1)], {sc.link: sc.fc_hz}, rate)
    assert pol.feasible and pol.t_u_s is not None
    tp = pol.tiles[sc.panel]
    model = sc.build(tp.tiles_u, tp.tiles_v)
    st = focus_states(model, sc.panel, sc.link, sc.t_focus_s, sc.fc_hz)
    cfg = {sc.panel: PanelConfig(sc.panel, st)}
    m_a = math.floor(Fraction(sc.window_s[0]).limit_denominator(10**6) / pol.t_u_s)
    m_b = math.ceil(Fraction(sc.window_s[1]).limit_denominator(10**6) / pol.t_u_s)
    k_lo = math.ceil(m_a * pol.t_u_s * sc.f_s)
    k_hi = math.ceil(m_b * pol.t_u_s * sc.f_s)
    ring = const_ring("tx.tx0", sc.f_s, k_hi + 64)
    y = render_span(
        SpanSpec(model, cfg, sc.link, sc.fc_hz, ring, sc.f_s, pol.t_u_s, kinds=RIS_KINDS),
        k_lo,
        k_hi,
    )
    t = np.arange(k_lo, k_hi) / float(sc.f_s)
    gam = np.asarray(model.panels[sc.panel].gamma)[st.astype(np.int64)]
    h = ris_brute(model, sc.panel, sc.link, gam, t, sc.fc_hz)
    rel = np.abs(y - h) / np.abs(h)
    assert rel.max() <= EPS_REL
    vi = [
        verify_interp(
            model, cfg, m * pol.t_u_s, (m + 1) * pol.t_u_s, {sc.link: sc.fc_hz}, kinds=RIS_KINDS
        )[sc.link]
        for m in range(m_a, m_b)
    ]
    assert max(vi) == pytest.approx(rel.max(), rel=0.10)


_PH18_BUDGET_SPLIT = (
    "open issue (WP5 review): the section 4.4 policy gives the full eps_rel to the chord "
    "(curvature) bound and up to eps_rel to the tile residual bound, so their sum can exceed "
    "the PH-18 limit. Vehicle at 2e4 S/s picks T_u = 778 us, 2x2 tiles: chord alone 0.997e-3 "
    "(9x9 tiles), with the tile residual 1.004e-3 / 1.0055e-3 at focus 0.06 / 0.07 s. Needs a "
    "spec decision (e.g. split eps between the two rules); the tolerance is not weakened."
)


@pytest.mark.parametrize(
    "t_focus",
    [
        0.03,
        0.04,
        0.05,
        pytest.param(0.06, marks=pytest.mark.xfail(strict=True, reason=_PH18_BUDGET_SPLIT)),
        pytest.param(0.07, marks=pytest.mark.xfail(strict=True, reason=_PH18_BUDGET_SPLIT)),
    ],
)
def test_ph18_vehicle_focus_sweep_vs_brute_force(t_focus: float) -> None:
    """(review) PH-18 vehicle scene at the render rate, panel focused at other times on the
    pass (window +-15 ms around the focus): |dh|/|h| <= 1e-3 per element per sample."""
    sc = vehicle()
    m1 = sc.build(1, 1)
    pol = choose_policy(m1.motion_bounds(), [sc.geometry(m1)], {sc.link: sc.fc_hz}, sc.f_s)
    assert pol.feasible and pol.t_u_s is not None
    tp = pol.tiles[sc.panel]
    model = sc.build(tp.tiles_u, tp.tiles_v)
    st = focus_states(model, sc.panel, sc.link, t_focus, sc.fc_hz)
    cfg = {sc.panel: PanelConfig(sc.panel, st)}
    lo = Fraction(t_focus).limit_denominator(10**6) - Fraction(15, 1000)
    hi = Fraction(t_focus).limit_denominator(10**6) + Fraction(15, 1000)
    m_a, m_b = math.floor(lo / pol.t_u_s), math.ceil(hi / pol.t_u_s)
    k_lo = math.ceil(m_a * pol.t_u_s * sc.f_s)
    k_hi = math.ceil(m_b * pol.t_u_s * sc.f_s)
    ring = const_ring("tx.tx0", sc.f_s, k_hi + 64)
    y = render_span(
        SpanSpec(model, cfg, sc.link, sc.fc_hz, ring, sc.f_s, pol.t_u_s, kinds=RIS_KINDS),
        k_lo,
        k_hi,
    )
    t = np.arange(k_lo, k_hi) / float(sc.f_s)
    gam = np.asarray(model.panels[sc.panel].gamma)[st.astype(np.int64)]
    h = ris_brute(model, sc.panel, sc.link, gam, t, sc.fc_hz)
    assert float(np.max(np.abs(y - h) / np.abs(h))) <= EPS_REL


def test_ph18_verify_interp_whole_channel_and_static_scene() -> None:
    sc = lab_walk()
    model = sc.build(4, 4)
    st = focus_states(model, sc.panel, sc.link, sc.t_focus_s, sc.fc_hz)
    cfg = {sc.panel: PanelConfig(sc.panel, st)}
    t_a, t_b = Fraction(2, 10), Fraction(2, 10) + Fraction(1, 1000)
    whole = verify_interp(model, cfg, t_a, t_b, {sc.link: sc.fc_hz})[sc.link]
    ris = verify_interp(model, cfg, t_a, t_b, {sc.link: sc.fc_hz}, kinds=RIS_KINDS)[sc.link]
    assert 0.0 < whole and 0.0 < ris
    # a static interval (t_a == t_b) interpolates exactly
    assert verify_interp(model, cfg, t_a, t_a, {sc.link: sc.fc_hz})[sc.link] == 0.0


# --------------------------------------------------------------------------- PH-19


def test_ph19_t_u_invariance_within_2e_minus_3() -> None:
    sc = lab_walk()
    m1 = sc.build(1, 1)
    pol = choose_policy(m1.motion_bounds(), [sc.geometry(m1)], {sc.link: sc.fc_hz}, sc.f_s)
    tp = pol.tiles[sc.panel]
    model = sc.build(tp.tiles_u, tp.tiles_v)
    cfg = {sc.panel: PanelConfig(sc.panel, focus_states(model, sc.panel, sc.link, 0.2, sc.fc_hz))}
    k_lo, k_hi = 3600, 4400
    ring = const_ring("tx.tx0", sc.f_s, k_hi + 64)
    outs = []
    for div in (1, 2, 4):
        spec = SpanSpec(model, cfg, sc.link, sc.fc_hz, ring, sc.f_s, pol.t_u_s / div)
        outs.append(render_span(spec, k_lo, k_hi))
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.max(np.abs(outs[i] - outs[j]) / np.abs(outs[0])) <= 2 * EPS_REL
    # positions (and hence snapshot paths) at a common time are bitwise identical
    t_common = 9 * pol.t_u_s
    p1 = model.combine(model.snapshot(t_common), cfg)[sc.link]
    model.snapshot(Fraction(1, 7))  # evaluation order must not matter (pure in t)
    p2 = model.combine(model.snapshot(t_common), cfg)[sc.link]
    assert np.array_equal(p1.amp, p2.amp) and np.array_equal(p1.delay_s, p2.delay_s)


@pytest.mark.parametrize("chunk_ms", ["1/4", "1", "37/10"])
def test_ph19_client_chunks_bitwise(backend_name: str, chunk_ms: str) -> None:
    """Splitting segments at 0.25 / 1 / 3.7 ms chunk edges leaves the output bitwise equal."""
    sc = vehicle()
    model = sc.build(2, 2)
    cfg = {
        sc.panel: PanelConfig(
            sc.panel, focus_states(model, sc.panel, sc.link, sc.t_focus_s, sc.fc_hz)
        )
    }
    fs = sc.f_s
    k_lo, k_hi = 500, 1500
    ring = const_ring("tx.tx0", fs, k_hi + 64)
    spec = SpanSpec(model, cfg, sc.link, sc.fc_hz, ring, fs, Fraction(778, 10**6))
    ref = render_span(spec, k_lo, k_hi, backend_name)
    chunk = Fraction(chunk_ms) / 1000 * fs
    edges = [math.ceil(i * chunk) for i in range(1, math.ceil(k_hi / chunk) + 1)]
    got = render_span(spec, k_lo, k_hi, backend_name, chunk_edges=edges)
    assert np.array_equal(ref, got)


def test_i2a_numba_thread_count_bitwise() -> None:
    sc = lab_walk()
    model = sc.build(4, 4)
    cfg = {sc.panel: PanelConfig(sc.panel, focus_states(model, sc.panel, sc.link, 0.2, sc.fc_hz))}
    ring = const_ring("tx.tx0", sc.f_s, 5000)
    spec = SpanSpec(model, cfg, sc.link, sc.fc_hz, ring, sc.f_s, Fraction(1167, 10**6))
    prev = numba.get_num_threads()
    try:
        outs = []
        for nt in sorted(
            {1, min(4, numba.config.NUMBA_NUM_THREADS), numba.config.NUMBA_NUM_THREADS}
        ):
            numba.set_num_threads(nt)
            outs.append(render_span(spec, 3700, 4300, "numba_cpu"))
    finally:
        numba.set_num_threads(prev)
    assert all(np.array_equal(outs[0], o) for o in outs[1:])


# --------------------------------------------------------------------------- policy


def _bounds(v: float, r: float, static: bool = False) -> MotionBounds:
    return MotionBounds(
        static=static,
        v_rel_max_mps={"a.tx0->b.rx0": v},
        v_perp_max_mps={"a.tx0->b.rx0": v},
        r_min_m={"a.tx0->b.rx0": r},
        panel_r_min_m={"ris0": r},
        panel_v_rel_max_mps={"ris0": v},
    )


def test_ph18_policy_error_budget_and_minimum_cost() -> None:
    fc = 3.5e9
    geo = PanelGeometry("ris0", 1.08, 1.08, 25, 25, fc, 1)
    pol = choose_policy(_bounds(1.5, 0.45), [geo], {"a.tx0->b.rx0": fc}, Fraction(10**6))
    k = 2 * math.pi * fc / C0
    tc = math.sqrt(8 * EPS_REL * 0.45 / (k * 1.5**2))  # closed form
    assert pol.t_curv_s == pytest.approx(tc, rel=1e-12)
    assert len(pol.candidates) == 5
    t0 = pol.candidates[0].t_u_s
    assert t0 <= Fraction(tc) and Fraction(tc) - t0 < Fraction(1, 10**6)  # quantized down to 1 us
    for c in pol.candidates:
        assert c.t_u_s == t0 / 2**c.m
        tp = c.tiles["ris0"]
        dt = math.sqrt(8 * EPS_REL) * (C0 / fc) * 0.45 / (2 * math.pi * 1.5 * float(c.t_u_s))
        assert tp.tile_side_max_m == pytest.approx(dt, rel=1e-12)
        assert tp.tiles_u == min(math.ceil(1.08 / dt), 25)
        if tp.tiles_u > 1:  # minimal: one tile fewer would violate the side bound
            assert 1.08 / (tp.tiles_u - 1) > dt
        # a tile of side D_t exactly meets the residual budget dphi**2/8 = eps
        assert residual_bound_rel(dt, 0.45, 1.5, fc, float(c.t_u_s)) == pytest.approx(
            EPS_REL, rel=1e-12
        )
    best = min((c for c in pol.candidates if c.feasible), key=lambda c: c.cost)
    assert pol.m == best.m and pol.cost == best.cost
    assert pol.curvature_bound_rel <= EPS_REL and pol.residual_bound_rel <= EPS_REL
    assert t_curv_s(0.45, 0.0, fc) == math.inf and tile_side_max_m(0.45, 0.0, fc, 1e-3) == math.inf


def test_ph18_policy_cap_scenario_lowering_and_static() -> None:
    fc = 1e9
    geo = PanelGeometry("ris0", 10.0, 0.15, 67, 1, fc, 1)
    slow = choose_policy(_bounds(0.01, 2.0), [geo], {"a.tx0->b.rx0": fc}, 5000)
    assert slow.candidates[0].t_u_s == Fraction(1, 100)  # 10 ms cap
    lower = choose_policy(
        _bounds(0.01, 2.0), [geo], {"a.tx0->b.rx0": fc}, 5000, t_u_scenario_s=Fraction(1, 1000)
    )
    assert lower.candidates[0].t_u_s == Fraction(1, 1000)
    higher = choose_policy(
        _bounds(0.01, 2.0), [geo], {"a.tx0->b.rx0": fc}, 5000, t_u_scenario_s=Fraction(1, 10)
    )
    assert higher.candidates[0].t_u_s == Fraction(1, 100)  # may only lower
    static = choose_policy(_bounds(0.0, 2.0, static=True), [geo], {"a.tx0->b.rx0": fc}, 5000)
    assert static.t_u_s is None and static.tiles["ris0"].n_tiles == 1
    assert static.to_manifest()["t_u_s"] is None


def test_ph18_policy_never_rounds_t_u_above_the_curvature_bound() -> None:
    """(review) T_u0 = min(T_curv, 10 ms) must not exceed T_curv, also when T_curv < 1 us
    (the [ENG] 1 us quantum is refined instead of rounding up)."""
    fc = 3.5e9
    k = 2 * math.pi * fc / C0
    for v, r in ((1.5, 0.45), (30.0, 5.0), (2.0e4, 1.0), (1.0e6, 0.5)):
        pol = choose_policy(_bounds(v, r), [], {"a.tx0->b.rx0": fc}, Fraction(10**6))
        tc = math.sqrt(8 * EPS_REL * r / (k * v * v))
        t0 = pol.candidates[0].t_u_s
        assert 0 < t0 <= Fraction(tc)
        # quantized down by less than one 1 us step (or 0.1 % below 1 us)
        assert tc - float(t0) < (1e-6 if tc >= 1e-6 else 1e-3 * tc)
        assert k * (v * v / r) * float(t0) ** 2 / 8 <= EPS_REL


def test_d1_weights_closed_form_and_degenerate_interval() -> None:
    g = interval_grid(Fraction(1, 3), Fraction(1, 3) + Fraction(7, 1000), Fraction(23_040_000))
    assert g.n_a_int == 7_680_000 and g.n_a_frac == 0.0 and g.n_first == 7_680_000
    assert g.n_end == 7_680_000 + 161_280
    n = np.array([7_680_000, 7_680_000 + 80_640, 7_680_000 + 161_279], dtype=np.int64)
    np.testing.assert_allclose(weights(g, n), [0.0, 0.5, 161_279 / 161_280], rtol=0, atol=1e-15)
    g2 = interval_grid(Fraction(1, 7), Fraction(2, 7), Fraction(5880))
    assert g2.n_a_int == 840 and g2.n_a_frac == 0.0
    g3 = interval_grid(Fraction(1, 3000), Fraction(2, 3000), Fraction(10_000))
    assert g3.n_a_int == 3 and g3.n_a_frac == pytest.approx(1 / 3, abs=1e-16) and g3.n_first == 4
    gd = interval_grid(Fraction(1, 2), Fraction(1, 2), Fraction(1000))
    assert gd.degenerate and np.all(weights(gd, np.arange(500, 600)) == 0.0)


def test_render_validation_errors(backend_name: str) -> None:
    from sremu.errors import ConfigurationError

    be = get_backend(backend_name)  # type: ignore[arg-type]
    fs = Fraction(1000)
    ring = const_ring("a.tx0", fs, 100)
    one = np.ones(1, np.complex128)
    good = LinkJob(
        "a.tx0->b.rx0",
        ring,
        1e9,
        0.0,
        one,
        one.copy(),
        np.array([1e-6]),
        np.array([1e-6]),
        np.zeros(1, np.int16),
        FDSpec.integer(),
    )
    job = SegmentJob("b.rx0", fs, 10, 20, Fraction(1, 100), Fraction(2, 100), [good])
    with pytest.raises(ConfigurationError, match="complex128"):
        be.render(job, np.zeros(10, np.complex64))
    with pytest.raises(ConfigurationError, match="length"):
        be.render(job, np.zeros(9, np.complex128))
    late = SegmentJob("b.rx0", fs, 15, 25, Fraction(1, 100), Fraction(2, 100), [good])
    with pytest.raises(ConfigurationError, match="ends after"):
        be.render(late, np.zeros(10, np.complex128))
    early = SegmentJob("b.rx0", fs, 5, 15, Fraction(1, 100), Fraction(2, 100), [good])
    with pytest.raises(ConfigurationError, match="starts before"):
        be.render(early, np.zeros(10, np.complex128))
    bad = LinkJob(
        "a.tx0->b.rx0",
        ring,
        1e9,
        0.0,
        one,
        one.copy(),
        np.array([-1e-6]),
        np.array([1e-6]),
        np.zeros(1, np.int16),
        FDSpec.integer(),
    )
    with pytest.raises(ConfigurationError, match=">= 0"):
        be.render(
            SegmentJob("b.rx0", fs, 10, 20, Fraction(1, 100), Fraction(2, 100), [bad]),
            np.zeros(10, np.complex128),
        )
    # (review) a reversed snapshot interval is a caller bug, not a static interval
    with pytest.raises(ConfigurationError, match="precedes"):
        be.render(
            SegmentJob("b.rx0", fs, 10, 20, Fraction(2, 100), Fraction(1, 100), [good]),
            np.zeros(10, np.complex128),
        )
    assert PathKind.RIS_TILE == 4
