"""Oracle self-consistency: equivalent forms, symmetries and independent numerical checks.

These tests prove the oracles against each other and against brute-force numerics, so that a
mistake in one closed form cannot pass silently into the PH/RT/ST/G checks of other WPs.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

from tests.oracles import mobility as mob
from tests.oracles import nonideal as ni
from tests.oracles import rf
from tests.oracles.ris import (
    cos_power,
    ellingson_area_gain_h,
    etsi_cell_gain_h,
    focus_phases,
    isotropic,
    k1_h,
    k1_terms,
    los_h,
    make_panel,
    subdivide,
    tang_eq12_h,
)

pytestmark = pytest.mark.physics


def _random_scene(rng: np.random.Generator):
    n_u, n_v = int(rng.integers(3, 12)), int(rng.integers(2, 9))
    fc = float(rng.uniform(1e9, 30e9))
    lam = rf.wavelength(fc)
    p = make_panel(
        rng.normal(size=3),
        [0.0, 0.0, 1.0],
        [math.cos(rng.uniform(0, 2 * math.pi)), math.sin(rng.uniform(0, 2 * math.pi)), 0.0],
        n_u,
        n_v,
        lam * rng.uniform(0.1, 0.5),
        lam * rng.uniform(0.1, 0.5),
    )
    c = p.cells.mean(axis=0)
    tx = c + np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.5, 4)])
    rx = c + np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.5, 4)])
    gam = rng.uniform(0.3, 1.0, p.n) * np.exp(2j * np.pi * rng.random(p.n))
    return p, fc, tx, rx, gam


def test_ph02_three_published_forms_equal_k1(rng):
    """K1 = Tang Eq. (12) = ETSI (G = 4 pi dx dy/lambda^2) = Ellingson with area gain.

    Relative tolerance 1e-12 (PH-02).
    """
    for _ in range(25):
        p, fc, tx, rx, gam = _random_scene(rng)
        c = p.cells.mean(axis=0)
        g_t = cos_power(float(rng.uniform(0.5, 6)), c - tx)
        g_r = cos_power(float(rng.uniform(0.5, 6)), c - rx)
        h = k1_h(tx, rx, p, fc, gam, g_t=g_t, g_r=g_r)
        for form in (tang_eq12_h, etsi_cell_gain_h, ellingson_area_gain_h):
            assert abs(form(tx, rx, p, fc, gam, g_t, g_r) - h) <= 1e-12 * abs(h), form.__name__


def test_ph08_k1_reciprocity_with_reciprocal_gamma(rng):
    """Swapping Tx and Rx (and their antennas) leaves h unchanged (rel 1e-12)."""
    for _ in range(25):
        p, fc, tx, rx, gam = _random_scene(rng)
        c = p.cells.mean(axis=0)
        ga, gb = cos_power(2.0, c - tx), cos_power(5.0, c - rx)
        h1 = k1_h(tx, rx, p, fc, gam, g_t=ga, g_r=gb)
        h2 = k1_h(rx, tx, p, fc, gam, g_t=gb, g_r=ga)
        assert abs(h1 - h2) <= 1e-12 * abs(h1)
        assert abs(los_h(tx, rx, fc, ga, gb) - los_h(rx, tx, fc, gb, ga)) <= 1e-15


def test_ph28_table_angle_convention_signed_by_u_and_specular():
    """Section 3.1 table angles: signed by the u component, specular is theta_s = -theta_i,
    and the off-plane (v) component is projected out (section 5.5)."""
    from tests.oracles.ris import table_angles_rad

    cell = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 1, 1, 0.01, 0.01)
    th = math.radians(30)
    tx = np.array([-2 * math.sin(th), 0.0, 2 * math.cos(th)])
    rx = np.array([3 * math.sin(th), 0.0, 3 * math.cos(th)])
    ti, ts = table_angles_rad(tx, rx, cell)
    assert abs(ti[0] - (-th)) <= 1e-15 and abs(ts[0] - th) <= 1e-15
    assert abs(ts[0] + ti[0]) <= 1e-15  # specular
    ti2, _ = table_angles_rad([-1.0, 5.0, 1.0], rx, cell)  # off-plane +5 m along v
    assert abs(ti2[0] - math.radians(-45.0)) <= 1e-15
    # rotating u by 180 deg flips every sign
    flipped = make_panel([0, 0, 0], [0, 0, 1], [-1, 0, 0], 1, 1, 0.01, 0.01)
    fi, fs = table_angles_rad(tx, rx, flipped)
    assert fi[0] == -ti[0] and fs[0] == -ts[0]


def test_ph08_k1_reciprocity_with_symmetric_angle_table(rng):
    """K1 with an angle-dependent table Gamma(theta_i, theta_s) that is symmetric is reciprocal
    (rel 1e-12); the same law applied to theta_i only (non-reciprocal table) is not."""
    from tests.oracles.ris import table_angles_rad

    def sym(a, b):
        return 0.8 * np.exp(1j * (0.4 + 1.3 * (a + b) ** 2 + 0.2 * np.cos(a - b)))

    def inc_only(a, b):
        return 0.8 * np.exp(1j * 1.3 * a)

    for _ in range(10):
        p, fc, tx, rx, _ = _random_scene(rng)
        ti, ts = table_angles_rad(tx, rx, p)
        h1 = k1_h(tx, rx, p, fc, sym(ti, ts))
        h2 = k1_h(rx, tx, p, fc, sym(ts, ti))
        assert abs(h1 - h2) <= 1e-12 * abs(h1)
        # angles swap exactly under Tx <-> Rx
        si, ss = table_angles_rad(rx, tx, p)
        assert np.array_equal(si, ts) and np.array_equal(ss, ti)
    a1 = k1_h(tx, rx, p, fc, inc_only(ti, ts))
    a2 = k1_h(rx, tx, p, fc, inc_only(ts, ti))
    assert abs(a1 - a2) > 1e-6 * abs(a1)  # teeth: an incidence-only law breaks reciprocity


def test_ph09_visibility_masks_per_cell():
    """A point beside the panel plane sees only the cells in front of it (cos > 0)."""
    p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 10, 1, 0.1, 0.1)
    t = k1_terms([0.0, 0.0, 1.0], [0.3, 0.0, 0.0], p, 3.5e9)  # Rx in the plane: cos_s = 0 for all
    assert np.all(t == 0)


def test_w1_wcl_eq2_ratio_matches_per_element_k1_in_the_corridor(rng):
    """V-WCL W1: on the letter's corridor (RIS line parallel to the user line at distance h,
    cos theta = h/d), the Eq. (2)/(3) element amplitude over the one-cell K1 amplitude squared
    equals ``(pi lambda)^2 (cos_i cos_s)^(2q-1)`` (rel 1e-12, well inside W1's 1e-9)."""
    from tests.oracles.ris import wcl_eq2_amplitude, wcl_eq2_over_k1_power_ratio

    for _ in range(50):
        fc = float(rng.choice([1e9, 2.4e9, 3.5e9]))
        lam = rf.wavelength(fc)
        w, h = lam / 4, float(rng.uniform(1.0, 6.0))
        y_cell = float(rng.uniform(0.0, 10.0))
        x_t, x_r = float(rng.uniform(0.0, 10.0)), float(rng.uniform(0.0, 10.0))
        # cell on the RIS line y = h facing the user line y = 0 (normal -y)
        cell = make_panel([y_cell, h, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], 1, 1, w, w)
        tx, rx = np.array([x_t, 0.0, 0.0]), np.array([x_r, 0.0, 0.0])
        d_t, d_r = math.hypot(x_t - y_cell, h), math.hypot(x_r - y_cell, h)
        k1 = abs(k1_h(tx, rx, cell, fc))
        letter = float(wcl_eq2_amplitude(lam, w, h, d_t, d_r))
        ratio = float(wcl_eq2_over_k1_power_ratio(lam, h / d_t, h / d_r))
        assert abs((letter / k1) ** 2 / ratio - 1.0) <= 1e-12


def test_ph05_k1_huygens_j_is_exactly_a_quarter_turn(rng):
    p, fc, tx, rx, gam = _random_scene(rng)
    assert abs(k1_h(tx, rx, p, fc, gam) - 1j * k1_h(tx, rx, p, fc, gam, huygens_j=False)) <= 1e-18


def test_ph26_subdivide_preserves_area_and_parent():
    p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 3, 2, 0.05, 0.03, tiles=(3, 1), gap_m=0.004)
    q = subdivide(p, 0.05, 0.03, 0.011)
    assert q.n == p.n * 5 * 3
    assert np.allclose(np.bincount(q.parent, weights=q.area), p.area, rtol=1e-14)
    # sub-cell centroid of every parent equals the parent centre
    for n in range(p.n):
        assert np.allclose(q.cells[q.parent == n].mean(axis=0), p.cells[n], atol=1e-15)


def test_g1_panel_tile_gap_geometry():
    """Across a tile boundary the cell spacing is pitch + gap; inside a tile it is the pitch."""
    p = make_panel(
        [0, 0, 0], [1, 0, 0], [0, 1, 0], 24, 5, 0.0448, 0.0448, tiles=(3, 1), gap_m=0.002
    )
    y = p.cells[p.iv == 0, 1]
    dy = np.diff(y)
    assert np.allclose(dy[[7, 15]], 0.0448 + 0.002, atol=1e-15)
    assert np.allclose(np.delete(dy, [7, 15]), 0.0448, atol=1e-15)
    assert abs(y.min() + y.max()) <= 1e-15  # centred


# --------------------------------------------------------------------------- optimum (PH-24/29/30)


def test_ph24_binary_optimum_bruteforce_vs_critical_angle(rng):
    for _ in range(20):
        n = 12
        z = rng.normal(size=n) + 1j * rng.normal(size=n)
        Z = np.stack([np.zeros(n, complex), z])
        z0 = complex(rng.normal(), rng.normal())
        vb, sb = ni.bruteforce_optimum(z0, Z)
        vc, _sc = ni.critical_angle_optimum(z0, Z)
        assert abs(vb - vc) <= 1e-12 * vb
        assert abs(abs(z0 + Z[sb, np.arange(n)].sum()) - vb) <= 1e-12 * vb


def test_ph29_general_alphabet_optimum_bruteforce_vs_critical_angle(rng):
    for _ in range(10):
        S, n = 4, 7
        Z = (rng.uniform(0.2, 1.0, (S, n)) * np.exp(2j * np.pi * rng.random((S, n)))) * np.exp(
            2j * np.pi * rng.random(n)
        )[None, :]
        z0 = complex(rng.normal(), rng.normal()) * 0.5
        vb, _ = ni.bruteforce_optimum(z0, Z)
        vc, _ = ni.critical_angle_optimum(z0, Z)
        assert abs(vb - vc) <= 1e-12 * vb


def test_ph30_quantizer_equals_per_element_bruteforce(rng):
    tang = np.array([0.9 * np.exp(1j * np.radians(165)), 0.7 + 0j])
    phi = rng.uniform(-np.pi, np.pi, 2000)
    s = ni.quantize_argmax(tang, phi)
    proj = np.real(tang[:, None] * np.exp(-1j * phi)[None, :])
    assert np.array_equal(s, np.argmax(proj, axis=0))
    assert np.all(proj[s, np.arange(phi.size)] >= proj.max(axis=0))


def test_ph24_continuous_optimum_closed_form(rng):
    """With unit-amplitude continuous phases the optimum is |z0| + sum|c_n|."""
    c = rng.normal(size=9) + 1j * rng.normal(size=9)
    z0 = complex(0.3, -1.1)
    phases = -np.angle(c) + np.angle(z0)
    assert abs(abs(z0 + np.sum(c * np.exp(1j * phases))) - (abs(z0) + np.abs(c).sum())) <= 1e-12


def test_ph28_half_difference_extension_symmetric_and_specular():
    g = lambda th: 0.8 * np.exp(1j * (0.3 + 2 * th))  # noqa: E731 - incidence-only law
    G = ni.half_difference_extension(g)
    grid = np.radians(np.arange(-90, 91, 7.5))
    for a in grid:
        for b in grid:
            assert G(a, b) == G(b, a)
        assert G(a, -a) == g(abs(a))  # specular theta_s = -theta_i


def test_ph13_binary_eta_equals_quadrature_for_equal_amplitude_pairs():
    for d in np.radians([10.0, 90.0, 150.0, 180.0, 203.41, 300.0]):
        assert abs(ni.binary_eta(d) - ni.alphabet_eta([1.0, np.exp(1j * d)])) <= 1e-12


# --------------------------------------------------------------------------- CFO / CRB


def test_g4_rt09_cfo_rotate_is_chunk_invariant_and_matches_float_formula(rng):
    x = rng.normal(size=4000) + 1j * rng.normal(size=4000)
    y = rf.cfo_rotate(x, 1375, 5880)
    parts, k = [], 0
    for size in (1, 7, 333, 1000, 659, 2000):
        parts.append(rf.cfo_rotate(x[k : k + size], 1375, 5880, n0=k))
        k += size
    assert np.array_equal(np.concatenate(parts), y)
    n = np.arange(x.size)
    ref = x * np.exp(2j * np.pi * 1375 * n / 5880)
    assert np.max(np.abs(ref - y)) <= 1e-10 * np.sqrt(np.mean(np.abs(y) ** 2))
    neg = rf.cfo_rotate(x, Fraction(-500), 5880)
    assert np.allclose(neg, np.conj(rf.cfo_rotate(np.conj(x), 500, 5880)), rtol=0, atol=1e-13)


def test_st05_crb_equals_inverse_fisher_information():
    """Rife-Boorstyn CRB vs the numerically inverted Fisher matrix (amplitude, phase, frequency)."""
    fs, n, snr = 5880.0, 2000, 10.0
    a, sigma2 = 1.0, 1.0 / snr
    t = np.arange(n) / fs
    # derivatives of s = a exp(j(2 pi f t + phi)) w.r.t. (a, phi, f) at phi = 0
    s = a * np.exp(1j * 2 * np.pi * 300.0 * t)
    d = np.stack([s / a, 1j * s, 1j * 2 * np.pi * t * s])
    fim = 2.0 / sigma2 * np.real(np.conj(d) @ d.T)
    crb_f = math.sqrt(np.linalg.inv(fim)[2, 2])
    assert abs(crb_f - rf.crb_frequency_std_hz(fs, snr, n)) <= 1e-6 * crb_f


# --------------------------------------------------------------------------- noise


def test_ph21_thermal_noise_density_and_figure():
    assert abs(rf.KT0_DBM_PER_HZ - (-173.975)) <= 5e-4
    assert abs(rf.thermal_noise_dbm(1e6, 7.0) - (rf.thermal_noise_dbm(1e6) + 7.0)) <= 1e-12
    assert (
        abs(rf.thermal_noise_dbm(23.04e6) - (rf.KT0_DBM_PER_HZ + 10 * math.log10(23.04e6))) <= 1e-12
    )


# --------------------------------------------------------------------------- budget (4.4)


def test_ph18_t_curv_matches_numerical_chord_error():
    """At T = T_curv the chord error of a straight pass at distance r_min is eps_rel/k."""
    fc, v, r_min, eps = 3.5e9, 1.5, 0.45, 1e-3
    T = mob.t_curv_s(eps, r_min, fc, v)
    k = rf.wavenumber(fc)
    t = np.linspace(-T / 2, T / 2, 20001)
    r = np.hypot(r_min, v * t)
    chord = r[0] + (r[-1] - r[0]) * (t - t[0]) / (t[-1] - t[0])
    err = float(np.max(np.abs(chord - r)))
    assert abs(k * err - eps) <= 1e-3 * eps


def test_ph18_tile_side_gives_the_residual_phase_budget():
    fc, r_min, v, t_u, eps = 3.5e9, 0.45, 1.5, 2e-3, 1e-3
    d_t = mob.tile_side_max_m(eps, fc, r_min, v, t_u)
    dphi = 2 * math.pi * v * d_t * t_u / (rf.wavelength(fc) * r_min)
    assert abs(dphi - math.sqrt(8 * eps)) <= 1e-12
    assert abs((dphi**2 / 8) - eps) <= 1e-15  # complex-linear midpoint dip


# --------------------------------------------------------------------------- RWP / fading / stats


def test_st01_rwp_pdf_cdf_and_leg_moments():
    ell = 10.0
    assert abs(quad(lambda x: float(mob.rwp1d_pdf(x, ell)), 0, ell)[0] - 1.0) <= 1e-12
    for x in (0.0, 1.3, 5.0, 9.9, 10.0):
        assert (
            abs(
                quad(lambda u: float(mob.rwp1d_pdf(u, ell)), 0, x)[0] - float(mob.rwp1d_cdf(x, ell))
            )
            <= 1e-12
        )
    # E|U1 - U2| and its SD for independent uniforms on [0, ell]
    m1 = quad(lambda u: quad(lambda w: abs(u - w), 0, ell, points=[u])[0], 0, ell)[0] / ell**2
    m2 = quad(lambda u: quad(lambda w: (u - w) ** 2, 0, ell)[0], 0, ell)[0] / ell**2
    assert abs(m1 - mob.rwp1d_mean_leg(ell)) <= 1e-9
    assert abs(math.sqrt(m2 - m1**2) - mob.rwp1d_leg_sd(ell)) <= 1e-9
    # the stationary pdf is the length-biased midpoint density of legs (Bettstetter)
    x = 3.7
    lb = quad(lambda u: quad(lambda w: abs(u - w), 0, ell, points=[u])[0], 0, ell)[0]
    ind = quad(
        lambda u: quad(
            lambda w: 1.0 if min(u, w) <= x <= max(u, w) else 0.0, 0, ell, points=[u, x]
        )[0],
        0,
        ell,
        points=[x],
    )[0]
    assert abs(ind / lb - float(mob.rwp1d_pdf(x, ell))) <= 1e-6


def test_st02_rayleigh_lcr_afd_identity():
    """LCR * AFD = P(envelope < rho) = 1 - exp(-rho^2) for every level."""
    rho = np.logspace(-2, 0.5, 30)
    lhs = mob.rayleigh_lcr_hz(rho, 50.0) * mob.rayleigh_afd_s(rho, 50.0)
    assert np.allclose(lhs, mob.rayleigh_cdf(rho), rtol=1e-12)
    assert float(mob.jakes_acf(0.0, 50.0)) == 1.0


def test_st01_ks_statistic_matches_scipy(rng):
    x = rng.random(5000)
    assert (
        abs(mob.ks_statistic(x, lambda v: np.clip(v, 0, 1)) - stats.kstest(x, "uniform").statistic)
        <= 1e-15
    )
    assert abs(mob.ks_critical(10_000, 0.01) - stats.kstwo.ppf(0.99, 10_000)) <= 2e-4


def test_st04_ber_and_clopper_pearson():
    assert abs(float(mob.ber_bpsk(0.0)) - 0.078649603525143) <= 1e-12
    assert abs(float(mob.ber_bpsk(10.0)) - 3.872108215522035e-06) <= 1e-15
    lo, hi = mob.clopper_pearson(7, 100, 0.95)
    ref = stats.binomtest(7, 100).proportion_ci(0.95, method="exact")
    assert abs(lo - ref.low) <= 1e-12 and abs(hi - ref.high) <= 1e-12


def test_ph03_focus_phases_cophase_every_term():
    p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 8, 8, 0.02, 0.02)
    tx, rx = np.array([-0.5, 0.2, 1.5]), np.array([0.7, -0.1, 1.2])
    t = k1_terms(tx, rx, p, 3.5e9, focus_phases(tx, rx, p, 3.5e9))
    assert np.max(np.abs(np.angle(t))) <= 1e-12
    assert isotropic(np.zeros((4, 3))).shape == (4,)
