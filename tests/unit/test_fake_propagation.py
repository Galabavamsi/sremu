"""FakePropagation against closed forms (amplitude, delay, Doppler, RIS phase convention)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PanelConfig, PathKind
from sremu.testing import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import doppler_t9, multi_rate_streams, near_far_rx, two_node_static
from sremu.timebase import Rate, to_ticks
from sremu.units import C0, amp_ratio_to_db

pytestmark = pytest.mark.unit


def test_static_los_amplitude_and_delay_friis_oracle_values():
    for fc, d, expected_db in ((3.5e9, 10.0, -63.329), (100e6, 52e3, -106.768)):
        prop = two_node_static(d_m=d, fc_hz=fc)
        (link,) = prop.links()
        pb = prop.snapshot(Fraction(0)).paths[link]
        lam = C0 / fc
        assert pb.amp.dtype == np.complex128
        assert pb.amp[0].imag == 0.0
        assert pb.amp[0].real == pytest.approx(lam / (4 * math.pi * d), rel=1e-12)
        assert amp_ratio_to_db(abs(pb.amp[0])) == pytest.approx(expected_db, abs=5e-4)
        assert pb.delay_s[0] == pytest.approx(d / C0, rel=1e-15)
        assert pb.kind[0] == PathKind.LOS


def test_moving_rx_delay_and_doppler_t9():
    prop = doppler_t9()
    (link,) = prop.links()
    for t in (Fraction(0), Fraction(1, 10), Fraction(1, 3), Fraction(1)):
        tau = prop.snapshot(t).paths[link].delay_s[0]
        assert tau == pytest.approx((50.0 - 30.0 * float(t)) / C0, rel=1e-14)
    t0, t1 = Fraction(0), Fraction(1, 1000)
    d_tau = prop.snapshot(t1).paths[link].delay_s[0] - prop.snapshot(t0).paths[link].delay_s[0]
    f_d = -3.5e9 * d_tau / float(t1 - t0)
    assert f_d == pytest.approx(350.242, abs=1e-3)  # section 4.4: 3.5 GHz at 30 m/s


def test_t9_ris_tile_delay_doppler_matches_oracle_325_192_hz():
    """T9 RIS geometry (pm_checks.py; physics_model_checks.json T9_RIS_path_doppler_expected
    _Hz_at_t0 = 325.192): a single tile references the panel centre (25, 10, 0), so the tile
    delay rate gives the RIS-path Doppler of the approaching RX at t = 0 (PH-17 value)."""
    prop = doppler_t9(with_panel=True, tiles=1)
    assert prop.snapshot(Fraction(0)).ris["ris0"].rows == ((prop.links()[0], 0),)
    dt = Fraction(1, 10**6)
    tau0 = prop.snapshot(Fraction(0)).ris["ris0"].tile_delay_s[0, 0]
    tau1 = prop.snapshot(dt).ris["ris0"].tile_delay_s[0, 0]
    f_d = -3.5e9 * (tau1 - tau0) / float(dt)
    assert f_d == pytest.approx(325.192, abs=0.01)
    # the delay is TX -> centre -> RX in metres / C0
    assert tau0 * C0 == pytest.approx(math.hypot(25.0, 10.0) * 2, rel=1e-14)
    assert prop.ris_delay_s("ris0", "rx.rx0", Fraction(0)) * C0 == pytest.approx(
        math.hypot(25.0, 10.0), rel=1e-14
    )


def test_motion_bounds_constant_velocity():
    mb = doppler_t9().motion_bounds()
    (link,) = mb.v_rel_max_mps
    assert not mb.static
    assert mb.v_rel_max_mps[link] == pytest.approx(30.0)
    assert mb.r_min_m[link] == pytest.approx(20.0)  # 50 m -> 20 m over the 1 s horizon
    assert two_node_static().motion_bounds().static


def test_links_follow_rate_and_band_rules():
    assert multi_rate_streams().links() == (
        "txa.tx0->rxa.rx0",
        "txb.tx0->rxb.rx0",
        "txc.tx0->rxc.rx0",
    )


def test_ris_delay_and_far_rx_switch_offset_near_far_geometry():
    prop = near_far_rx()
    f = Rate.parse("23.04e6")
    tau_far = prop.ris_delay_s("ris0", "rx2.rx0", Fraction(0))
    tau_near = prop.ris_delay_s("ris0", "rx1.rx0", Fraction(0))
    assert tau_far == pytest.approx(2000.0 / C0, rel=1e-15)
    assert tau_far * float(f) == pytest.approx(153.7, abs=0.05)  # section 4.6 [DERIVED]
    t_sw = Fraction(1, 200)
    k_near = to_ticks(t_sw + Fraction(tau_near), f)
    k_far = to_ticks(t_sw + Fraction(tau_far), f)
    assert k_far - k_near == 154


def _direct_ris_sum(
    prop: FakePropagation, panel: FakePanel, link: str, states, t
) -> tuple[complex, float]:
    """Independent per-cell sum with ABSOLUTE phase: A j/(4 pi r_i r_s) exp(-jk(r_i + r_s)) G.

    Returns the sum and the sum of term magnitudes (the conditioning scale).
    """
    tx, rx = prop.link_streams(link)
    ptx = prop.node_position(tx.node, t)
    prx = prop.node_position(rx.node, t)
    n = np.asarray(panel.normal)
    u = np.asarray(panel.u_axis)
    v = np.cross(n, u)
    k = 2 * math.pi * tx.fc_hz / C0
    total = 0j
    scale = 0.0
    for iv in range(panel.n_v):
        for iu in range(panel.n_u):
            idx = iv * panel.n_u + iu
            p = (
                np.asarray(panel.center_m)
                + (iu - (panel.n_u - 1) / 2) * panel.pitch_u_m * u
                + (iv - (panel.n_v - 1) / 2) * panel.pitch_v_m * v
            )
            ri = float(np.linalg.norm(p - ptx))
            rs = float(np.linalg.norm(p - prx))
            g = panel.gamma[states[idx]] if states[idx] >= 0 else panel.gamma_dead
            term = (
                panel.pitch_u_m
                * panel.pitch_v_m
                * 1j
                / (4 * math.pi * ri * rs)
                * np.exp(-1j * k * (ri + rs))
                * g
            )
            total += term
            scale += abs(term)
    return complex(total), scale


@pytest.mark.parametrize("tiles", [1, 3])
def test_ris_tiles_reproduce_direct_sum_with_absolute_phase(tiles):
    prop = doppler_t9(with_panel=True, tiles=tiles)
    panel = prop.panels["ris0"]
    (link,) = prop.links()
    fc = 3.5e9
    gen = np.random.Generator(np.random.Philox(key=tiles))
    for t in (Fraction(0), Fraction(1, 7)):
        snap = prop.snapshot(t)
        states = gen.integers(-1, 2, size=panel.n_cells).astype(np.int16)
        lp = prop.combine(snap, {"ris0": PanelConfig("ris0", states)})[link]
        assert lp.n_paths == 1 + tiles * tiles
        ris = lp.kind == PathKind.RIS_TILE
        assert ris.sum() == tiles * tiles
        assert list(lp.tile[~ris]) == [-1]
        # renderer convention: h = sum amp * exp(-j 2 pi fc tau)
        h_ris = complex(np.sum(lp.amp[ris] * np.exp(-2j * math.pi * fc * lp.delay_s[ris])))
        expected, scale = _direct_ris_sum(prop, panel, link, states, t)
        # random states make the sum partly cancel; float64 phases of ~5e3 rad bound the
        # error relative to the sum of term magnitudes, not to |sum|
        assert abs(h_ris - expected) <= 1e-12 * scale


def test_combine_is_linear_in_gamma_and_disabled_is_zero():
    prop = doppler_t9(with_panel=True, tiles=2)
    (link,) = prop.links()
    snap = prop.snapshot(Fraction(0))
    n = prop.panels["ris0"].n_cells
    a = prop.combine(snap, {"ris0": PanelConfig("ris0", np.zeros(n, np.int16))})[link]
    b = prop.combine(snap, {"ris0": PanelConfig("ris0", np.ones(n, np.int16))})[link]
    off = prop.combine(snap, {"ris0": PanelConfig("ris0", np.zeros(n, np.int16), False)})[link]
    ris = a.kind == PathKind.RIS_TILE
    np.testing.assert_array_equal(b.amp[ris], -a.amp[ris])  # gamma (1, -1)
    np.testing.assert_array_equal(a.amp[~ris], b.amp[~ris])
    assert np.all(off.amp[ris] == 0)
    np.testing.assert_array_equal(off.delay_s, a.delay_s)  # path set stays aligned


def test_geometry_errors():
    nodes = [FakeNode("a", (0, 0, 0), streams=()), FakeNode("a", (1, 0, 0))]
    with pytest.raises(ValueError, match="duplicate"):
        FakePropagation(nodes)
    with pytest.raises(ValueError, match="unit vectors"):
        FakePanel("p", (0, 0, 0), (2, 0, 0), (0, 1, 0), 2, 2, 0.1, 0.1)
