"""WP5: transition math (RT-12) through the renderer.

Oracles: the per-element channel ``h(n) = LOS(n) + sum_cells Gamma_n(n) c_n(n)`` with the
element Gamma following the closed-form profiles (``_render_support.ris_brute``), and the
closed-form LOS ``lambda/(4 pi d) exp(-jkd)``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.channel.interp import choose_policy
from sremu.channel.render_numpy import get_backend
from sremu.channel.transitions import (
    build_terms,
    class_deltas,
    freeze_terms,
    profile_values,
    recommand_classes,
    recommand_deltas,
    terms_from_deltas,
    transition_classes,
)
from sremu.contracts import LinkJob, PanelConfig, SegmentJob, TransitionSpec
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import stream
from sremu.timebase import to_ticks
from sremu.units import C0

from ._render_support import SpanSpec, _scene, const_ring, render_span, ris_brute

pytestmark = pytest.mark.runtime

FC = 3.5e9
FS = Fraction(23_040_000)
LINK = "ap.tx0->mu.rx0"


def _static_scene() -> FakePropagation:
    lam = C0 / FC
    nodes = [
        FakeNode("ap", (2.0, -1.0, 0.2), streams=(stream("ap.tx0", "tx", FC, "23.04e6"),)),
        FakeNode("mu", (1.5, 0.5, -0.1), streams=(stream("mu.rx0", "rx", FC, "23.04e6"),)),
    ]
    panel = FakePanel(
        "ris0",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        8,
        8,
        lam / 2,
        lam / 2,
        gamma=(1 + 0j, -1 + 0j),
        tiles_u=2,
        tiles_v=2,
    )
    return FakePropagation(nodes, [panel], horizon_s=1)


def _los(model: FakePropagation, t: np.ndarray) -> np.ndarray:
    tx, rx = model.link_streams(LINK)
    a, b = model.nodes[tx.node], model.nodes[rx.node]
    pa = np.asarray(a.position_m)[None, :] + np.asarray(a.velocity_mps)[None, :] * t[:, None]
    pb = np.asarray(b.position_m)[None, :] + np.asarray(b.velocity_mps)[None, :] * t[:, None]
    d = np.linalg.norm(pb - pa, axis=1)
    lam = C0 / FC
    return lam / (4 * math.pi * d) * np.exp(-2j * math.pi * FC * d / C0)


def _static_job(
    model: FakePropagation, cfg: dict, ring, k0: int, k1: int, terms: list
) -> SegmentJob:
    snap = model.snapshot(Fraction(0))
    lp = model.combine(snap, cfg)[LINK]
    from sremu.contracts import FDSpec

    lj = LinkJob(
        LINK,
        ring,
        FC,
        0.0,
        lp.amp,
        lp.amp.copy(),
        lp.delay_s,
        lp.delay_s.copy(),
        np.zeros(lp.n_paths, np.int16),
        FDSpec.integer(),
        list(terms),
    )
    return SegmentJob("mu.rx0", FS, k0, k1, Fraction(0), Fraction(0), [lj])


def test_rt12_profiles_closed_form() -> None:
    x = np.array([-1.0, 0.0, 115.2, 230.4, 460.8, 1000.0])
    np.testing.assert_array_equal(profile_values("step", x, 0.0), [0, 1, 1, 1, 1, 1])
    np.testing.assert_allclose(
        profile_values("linear", x, 460.8), [0, 0, 0.25, 0.5, 1, 1], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        profile_values("first_order", x, 100.0),
        [0, 0, 1 - math.exp(-1.152), 1 - math.exp(-2.304), 1 - math.exp(-4.608), 1 - math.exp(-10)],
        rtol=1e-15,
        atol=0,
    )
    # a non-positive duration is a step
    np.testing.assert_array_equal(profile_values("linear", x, 0.0), profile_values("step", x, 0.0))


def test_rt12_classes_single_and_by_edge() -> None:
    a = np.array([0, 1, 0, 1, 1], np.int16)
    b = np.array([1, 1, 0, 0, 0], np.int16)
    single = transition_classes(a, b, "single")
    assert [lbl for lbl, _ in single] == ["all"] and single[0][1].tolist() == [
        True,
        False,
        False,
        True,
        True,
    ]
    edge = transition_classes(a, b, "by_edge")
    assert [lbl for lbl, _ in edge] == ["0->1", "1->0"]
    assert edge[0][1].tolist() == [True, False, False, False, False]
    assert edge[1][1].tolist() == [False, False, False, True, True]
    assert transition_classes(a, a) == []
    first = np.array([True, False, False, False, False])
    new2 = np.array([1, 0, 0, 0, 1], np.int16)
    rc = recommand_classes(a, b, [(first, 0.5)], new2)
    # element 0 still moving (target stays 1); 1: 1 -> 0; 2 stays 0; 3: 1 -> 0; 4 stays 1
    assert rc[0][1].tolist() == [True, True, False, True, False]
    # once the first transition has finished, element 0 has settled at 1 = new2
    assert recommand_classes(a, b, [(first, 1.0)], new2)[0][1].tolist() == [
        False,
        True,
        False,
        True,
        False,
    ]
    assert [lbl for lbl, _ in recommand_classes(a, b, [(first, 0.5)], new2, "by_edge")] == [
        "1->0",
        "1->1",
    ]


def test_rt12_linear_20us_ramp_matches_analytic(backend_name: str) -> None:
    """Static scene, all-0 -> random binary over a linear 20 us ramp: |h| (and h) equal the
    analytic h_old + g(n - k)(h_new - h_old) to 1e-9."""
    model = _static_scene()
    n_cells = 64
    g = np.random.Generator(np.random.Philox(key=12))
    old = np.zeros(n_cells, np.int16)
    new = g.integers(0, 2, n_cells).astype(np.int16)
    cfg = {"ris0": PanelConfig("ris0", old)}
    snap = model.snapshot(Fraction(0))
    k_s = 1000
    terms = build_terms(
        model, snap, snap, cfg, "ris0", new, TransitionSpec("linear", "2e-5"), FS, {LINK: k_s}
    )[LINK]
    assert len(terms) == 1 and terms[0].duration_samples == pytest.approx(460.8, abs=1e-12)
    ring = const_ring("ap.tx0", FS, 3000)
    y = np.zeros(1500, np.complex128)
    get_backend(backend_name).render(_static_job(model, cfg, ring, 500, 2000, terms), y)  # type: ignore[arg-type]
    n = np.arange(500, 2000)
    t = np.zeros(n.shape[0])
    gam = np.asarray(model.panels["ris0"].gamma)
    ramp = np.clip((n - k_s) / 460.8, 0.0, 1.0)
    h = _los(model, t) + ris_brute(
        model,
        "ris0",
        LINK,
        lambda idx: gam[old][None, :] + ramp[idx][:, None] * (gam[new] - gam[old])[None, :],
        t,
        FC,
    )
    assert np.max(np.abs(np.abs(y) - np.abs(h)) / np.abs(h)) <= 1e-9
    assert np.max(np.abs(y - h) / np.abs(h)) <= 1e-9
    assert abs(y[0] - h[0]) / abs(h[0]) <= 1e-9 and abs(y[-1] - h[-1]) / abs(h[-1]) <= 1e-9


def test_rt12_recommand_mid_transition_restarts_from_instantaneous_gamma(backend_name: str) -> None:
    model = _static_scene()
    n_cells = 64
    g = np.random.Generator(np.random.Philox(key=13))
    old = g.integers(0, 2, n_cells).astype(np.int16)
    s_a = g.integers(0, 2, n_cells).astype(np.int16)
    s_b = g.integers(0, 2, n_cells).astype(np.int16)
    cfg = {"ris0": PanelConfig("ris0", old)}
    snap = model.snapshot(Fraction(0))
    k1, k2, dur = 1000, 1200, 460.8
    spec = TransitionSpec("linear", "2e-5")
    terms1 = build_terms(model, snap, snap, cfg, "ris0", s_a, spec, FS, {LINK: k1})[LINK]
    mask1 = transition_classes(old, s_a)[0][1]
    g1 = float(profile_values("linear", float(k2 - k1), dur))
    assert 0.0 < g1 < 1.0
    cls2 = recommand_classes(old, s_a, [(mask1, g1)], s_b)
    masks2 = [m for _, m in cls2]
    d2 = recommand_deltas(model, snap, cfg, "ris0", s_a, [(mask1, g1)], s_b, masks2)
    terms2 = freeze_terms(terms1, k2)
    for d in d2:
        terms2 += terms_from_deltas(d[LINK], d[LINK], "linear", dur, k2)
    ring = const_ring("ap.tx0", FS, 3000)
    be = get_backend(backend_name)  # type: ignore[arg-type]
    y = np.zeros(1500, np.complex128)
    be.render(_static_job(model, cfg, ring, 500, k2, terms1), y[: k2 - 500])
    be.render(_static_job(model, cfg, ring, k2, 2000, terms2), y[k2 - 500 :])
    gam = np.asarray(model.panels["ris0"].gamma)
    n = np.arange(500, 2000)
    r1 = np.clip((n - k1) / dur, 0, 1)
    r2 = np.clip((n - k2) / dur, 0, 1)
    g_old, g_a, g_b = gam[old], gam[s_a], gam[s_b]
    g_inst = g_old + g1 * mask1 * (g_a - g_old)

    def gamma_t(idx: np.ndarray) -> np.ndarray:
        before = g_old[None, :] + (r1[idx][:, None] * mask1[None, :]) * (g_a - g_old)[None, :]
        after = g_inst[None, :] + r2[idx][:, None] * (g_b - g_inst)[None, :]
        return np.where((n[idx] < k2)[:, None], before, after)

    t = np.zeros(n.shape[0])
    h = _los(model, t) + ris_brute(model, "ris0", LINK, gamma_t, t, FC)
    assert np.max(np.abs(y - h) / np.abs(h)) <= 1e-9
    # continuity at the re-command sample and arrival at the new configuration
    assert abs(y[k2 - 500] - y[k2 - 501]) < 0.05 * abs(y[k2 - 500])
    cfg_b = {"ris0": PanelConfig("ris0", s_b)}
    h_b = model.combine(snap, cfg_b)[LINK]
    y_b = np.zeros(1, np.complex128)
    be.render(_static_job(model, cfg_b, ring, 1999, 2000, []), y_b)
    assert abs(y[-1] - y_b[0]) <= 1e-9 * abs(y_b[0]) and h_b.n_paths > 1


def test_rt12_per_tile_switch_samples() -> None:
    """Paths of different tiles may see the switch at different RX samples (I3 per tile)."""
    model = _static_scene()
    old = np.zeros(64, np.int16)
    new = np.ones(64, np.int16)
    cfg = {"ris0": PanelConfig("ris0", old)}
    snap = model.snapshot(Fraction(0))
    d = class_deltas(model, snap, cfg, "ris0", new, [new != old])[0][LINK]
    ks = np.array([1000, 1000, 1003, 1003, 1007], np.int64)
    terms = terms_from_deltas(d, d, "step", 0.0, ks)
    assert [t.k_start for t in terms] == [1000, 1003, 1007]
    ring = const_ring("ap.tx0", FS, 3000)
    y = np.zeros(20, np.complex128)
    get_backend("numpy_ref").render(_static_job(model, cfg, ring, 995, 1015, terms), y)
    lp_old = model.combine(snap, cfg)[LINK]
    ph = np.exp(-2j * np.pi * FC * lp_old.delay_s)
    for i, n in enumerate(range(995, 1015)):
        amp = lp_old.amp + np.where(ks <= n, d, 0)
        assert abs(y[i] - np.sum(amp * ph)) <= 1e-12 * abs(y[i])


def test_rt12_per_class_deltas_under_mobility_vs_brute_force() -> None:
    """Lab-walk geometry with a binary panel: rising (0->1) linear 2 ms and falling (1->0)
    first-order 0.7 ms classes, deltas interpolated per tile; vs per-element per-sample
    brute force <= 1e-3."""
    fc = 3.5e9
    fs = Fraction(20_000)

    def build(tu: int, tv: int) -> FakePropagation:
        return _scene(fc, "2e4", (1.5, -1.0, 0.3), (0.45, -0.3, 0.0), (0.0, 1.5, 0.0),
                      25, 25, 1.08 / 25, "4/10", (tu, tv), bits=1)  # fmt: skip

    link = "tx.tx0->rx.rx0"
    m1 = build(1, 1)
    from ._render_support import MobileScene

    geo = MobileScene("lab1", fc, build, link, "ris0", 0.2, (0.19, 0.21), fs).geometry(m1)
    pol = choose_policy(m1.motion_bounds(), [geo], {link: fc}, fs)
    tp = pol.tiles["ris0"]
    model = build(tp.tiles_u, tp.tiles_v)
    g = np.random.Generator(np.random.Philox(key=14))
    old = g.integers(0, 2, 625).astype(np.int16)
    new = g.integers(0, 2, 625).astype(np.int16)
    cfg = {"ris0": PanelConfig("ris0", old)}
    t_sw = Fraction(195, 1000)
    k_s = to_ticks(t_sw + Fraction(model.ris_delay_s("ris0", "rx.rx0", t_sw)), fs)
    profiles = {"0->1": ("linear", Fraction(2, 1000)), "1->0": ("first_order", Fraction(7, 10000))}
    spec = TransitionSpec("step", None, "by_edge")

    def trans_fn(m, pa, pb):
        sa = model.snapshot(m * pol.t_u_s)
        sb = model.snapshot((m + 1) * pol.t_u_s)
        return build_terms(
            model, sa, sb, cfg, "ris0", new, spec, fs, {link: k_s}, class_profiles=profiles
        )[link]

    k_lo, k_hi = 3800, 4200
    ring = const_ring("tx.tx0", fs, k_hi + 64)
    y = render_span(
        SpanSpec(model, cfg, link, fc, ring, fs, pol.t_u_s, trans_fn=trans_fn), k_lo, k_hi
    )
    n = np.arange(k_lo, k_hi)
    t = n / float(fs)
    gam = np.asarray(model.panels["ris0"].gamma)
    rise = profile_values("linear", (n - k_s).astype(float), 40.0)
    fall = profile_values("first_order", (n - k_s).astype(float), 14.0)
    up = (old == 0) & (new == 1)
    dn = (old == 1) & (new == 0)

    def gamma_t(idx: np.ndarray) -> np.ndarray:
        gt = np.repeat(gam[old][None, :], idx.shape[0], axis=0)
        gt += rise[idx][:, None] * (up * (gam[new] - gam[old]))[None, :]
        gt += fall[idx][:, None] * (dn * (gam[new] - gam[old]))[None, :]
        return gt

    h = _los_moving(model, link, t, fc) + ris_brute(model, "ris0", link, gamma_t, t, fc)
    assert np.max(np.abs(y - h) / np.abs(h)) <= 1e-3


def _los_moving(model: FakePropagation, link: str, t: np.ndarray, fc: float) -> np.ndarray:
    tx, rx = model.link_streams(link)
    a, b = model.nodes[tx.node], model.nodes[rx.node]
    pa = np.asarray(a.position_m)[None, :] + np.asarray(a.velocity_mps)[None, :] * t[:, None]
    pb = np.asarray(b.position_m)[None, :] + np.asarray(b.velocity_mps)[None, :] * t[:, None]
    d = np.linalg.norm(pb - pa, axis=1)
    cyc = fc * d / C0
    return (C0 / fc) / (4 * math.pi * d) * np.exp(-2j * math.pi * (cyc - np.rint(cyc)))
