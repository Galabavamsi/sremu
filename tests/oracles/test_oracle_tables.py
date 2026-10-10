"""Recompute the spec oracle values (``tests/oracles/tables.py``) from the closed forms.

Each test names its spec ID. The spec value and tolerance come from ``ORACLE``; the computed
value comes only from this package's oracles (no sremu.physics/channel/impair).
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.units import C0
from tests.oracles import mobility as mob
from tests.oracles import nonideal as ni
from tests.oracles import p526, rf
from tests.oracles.ris import (
    disk_kirchhoff_onaxis_h,
    focus_phases,
    k1_h,
    k1_terms,
    legacy_v1_h,
    legacy_v2_h,
    local_subaperture_factor,
    make_panel,
    make_panel_extent,
    ph04_geometry,
    plate_k1_integral_h,
    po_patch_h,
    rect_solid_angle_onaxis,
    richardson_po_patch,
    subaperture_factor,
    subdivide,
    tang_eq8_h_abs,
    v1_default_scene,
)
from tests.oracles.tables import (
    ORACLE,
    PH17_T9_BLOCK_S,
    PH17_T9_FC_HZ,
    PH17_T9_FS,
    PH17_T9_RX0_M,
    PH17_T9_SPEED_MPS,
    PH17_T9_WINDOW_S,
    V_CFO_FS,
    V_CFO_TAU,
)

pytestmark = pytest.mark.physics


def check(key: str, value: float) -> None:
    o = ORACLE[key]
    assert abs(value - o.value) <= o.tol + 1e-12, (
        f"{o.spec_id} {o.what}: {value!r} vs {o.value} +- {o.tol}"
    )


def db(h: complex | np.ndarray) -> float:
    return float(rf.db20(h))


# --------------------------------------------------------------------------- PH-01


def test_ph01_friis_values():
    check("PH01_friis_3p5GHz_10m_db", db(rf.friis_h(10.0, 3.5e9)))
    check("PH01_friis_100MHz_52km_db", db(rf.friis_h(52000.0, 100e6)))


# --------------------------------------------------------------------------- PH-03


def test_ph03_n2_law_discrete_and_tang_eq8():
    fc = 3.5e9
    lam = rf.wavelength(fc)
    d = 2000.0
    tx = np.array([0, 0, d])
    rx = np.array([d * math.sin(math.radians(20)), 0, d * math.cos(math.radians(20))])
    vals = []
    for n_side, key in ((10, "PH03_N100_db"), (20, "PH03_N400_db"), (40, "PH03_N1600_db")):
        p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], n_side, n_side, lam / 2, lam / 2)
        h = db(k1_h(tx, rx, p, fc, focus_phases(tx, rx, p, fc)))
        eq8 = 20 * math.log10(tang_eq8_h_abs(p.n, (lam / 2) ** 2, d, d, 0.0, math.radians(20)))
        check(key, h)
        check(key, eq8)
        vals.append(h)
    check("PH03_slope_db_per_4x", vals[1] - vals[0])
    check("PH03_slope_db_per_4x", vals[2] - vals[1])


# --------------------------------------------------------------------------- PH-04


@pytest.mark.parametrize("n_side", [4, 8, 16, 32, 64, 128])
def test_ph04_far_specular_any_n(n_side):
    tx, rx = ph04_geometry(200.0, 30.0)
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, n_side, n_side)
    check("PH04_far_specular_200m_db", db(k1_h(tx, rx, p, 3.5e9)))


_PH04_NEAR_ANOMALOUS = (5.0, -10.0)
_PH04_SIDES = [4, 8, 16, 32, 64, 128]  # N = 16 ... 16384 cells on the 1 m^2 plate


@pytest.mark.parametrize("n_side", _PH04_SIDES)
def test_ph04_near_anomalous_oracle_value_at_lambda_over_40(n_side):
    """D5: po_patch at Delta = lambda/40 is -88.128 dB for every N (to its 3 printed decimals),
    and that is the ORACLE value (accept +-0.10 dB)."""
    fc = 3.5e9
    tx, rx = ph04_geometry(*_PH04_NEAR_ANOMALOUS)
    pitch = 1.0 / n_side
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, n_side, n_side)
    level = db(po_patch_h(tx, rx, p, fc, pitch, pitch, rf.wavelength(fc) / 40))
    assert abs(level - ORACLE["PH04_near_anomalous_5m_db"].value) <= 5e-4, level
    check("PH04_near_anomalous_5m_db", level)


def test_ph04_near_anomalous_continuous_limit_is_independent_of_the_cell_grid():
    """D5 cross-check without any cell machinery: the continuous-aperture integral (tensor
    Gauss-Legendre, converged to 1e-6 dB) is the Delta -> 0 limit; the midpoint (po_patch)
    error falls as Delta^2 and is 0.002 dB at lambda/40. The same integral gives the PH-04 far
    specular value (any N), so the plate integral itself is checked against a spec number."""
    fc = 3.5e9
    lam = rf.wavelength(fc)
    tx, rx = ph04_geometry(*_PH04_NEAR_ANOMALOUS)
    plate = ([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, fc)
    lim = [db(plate_k1_integral_h(tx, rx, *plate, n_nodes=n)) for n in (32, 64, 96)]
    assert max(lim) - min(lim) <= 1e-6
    check("PH04_near_anomalous_5m_db", lim[-1])
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, 16, 16)
    err = [
        db(po_patch_h(tx, rx, p, fc, 1 / 16, 1 / 16, lam / div)) - lim[-1] for div in (20, 40, 80)
    ]
    assert abs(err[1]) <= 0.005  # lambda/40 is converged well inside the 0.10 dB tolerance
    assert 3.0 <= err[0] / err[1] <= 5.0 and 3.0 <= err[1] / err[2] <= 5.0  # second order
    far_tx, far_rx = ph04_geometry(200.0, 30.0)
    check("PH04_far_specular_200m_db", db(plate_k1_integral_h(far_tx, far_rx, *plate)))


@pytest.mark.parametrize("n_side", _PH04_SIDES)
def test_ph04_near_anomalous_richardson_rule_reaches_the_oracle(n_side):
    """Section 5.4 rule from Delta = lambda/5 (compare Delta and Delta/2, halve at most twice
    while |dP| > 0.05 dB): the accepted level and its half-size check are both within the D5
    tolerance of -88.128 dB."""
    fc = 3.5e9
    tx, rx = ph04_geometry(*_PH04_NEAR_ANOMALOUS)
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, n_side, n_side)
    res = richardson_po_patch(tx, rx, p, fc, 1.0 / n_side, 1.0 / n_side)
    assert res.est_db <= 0.05
    check("PH04_near_anomalous_5m_db", db(res.h))
    check("PH04_near_anomalous_5m_db", db(res.h_half))


def test_ph04_single_level_po_patch_is_not_enough_teeth():
    """Why D5 asserts only converged values: at the first level (Delta = lambda/5) N = 16 and
    N = 64 sit 0.14 / 0.12 dB above -88.128 dB, outside +-0.10; the Richardson rule (which
    halves once for them) is what brings them in."""
    fc = 3.5e9
    tx, rx = ph04_geometry(*_PH04_NEAR_ANOMALOUS)
    o = ORACLE["PH04_near_anomalous_5m_db"]
    for n_side in (4, 8):
        p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, n_side, n_side)
        first = db(po_patch_h(tx, rx, p, fc, 1 / n_side, 1 / n_side, rf.wavelength(fc) / 5))
        assert abs(first - o.value) > o.tol
        res = richardson_po_patch(tx, rx, p, fc, 1 / n_side, 1 / n_side)
        assert res.halvings == 1
    with pytest.raises(RuntimeError, match="not converged"):
        richardson_po_patch(tx, rx, p, fc, 1 / 8, 1 / 8, tol_db=1e-4, max_halvings=0)


def test_ph04_teeth_legacy_v2_gains_12db_per_refinement():
    tx, rx = ph04_geometry(200.0, 30.0)
    vals = [
        db(
            legacy_v2_h(
                tx, rx, make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1, 1, n, n), 3.5e9
            )
        )
        for n in (16, 32, 64, 128)
    ]
    for a, b in itertools.pairwise(vals):
        check("PH04_legacy_v2_db_per_4x", b - a)


# --------------------------------------------------------------------------- PH-05


def test_ph05_oblique_plate_and_no_j_teeth():
    fc = 3.5e9
    lam = rf.wavelength(fc)
    th = math.radians(40)
    tx = np.array([-3 * math.sin(th), 0, 3 * math.cos(th)])
    rx = np.array([3 * math.sin(th), 0, 3 * math.cos(th)])
    ns = round(12.0 / (lam / 6))
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 12.0, 12.0, ns, ns)
    img = rf.image_theory_h(3.0, 3.0, fc)
    r = k1_h(tx, rx, p, fc) / img
    check("PH05_oblique_plate_db", db(r))
    check("PH05_oblique_plate_deg", math.degrees(np.angle(r)))
    rn = k1_h(tx, rx, p, fc, huygens_j=False) / img
    check("PH05_no_j_phase_deg", math.degrees(np.angle(rn)))


def test_ph05_disk_discrete_and_image_limits():
    fc = 3.5e9
    k = rf.wavenumber(fc)
    dm = 2.0
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 2.0, 2.0, 256, 256)
    mask = np.hypot(p.cells[:, 0], p.cells[:, 1]) <= 1.0
    disk = type(p)(
        p.cells[mask],
        p.area[mask],
        p.normal,
        p.u,
        p.v,
        p.iu[mask],
        p.iv[mask],
        np.arange(int(mask.sum()), dtype=np.int64),
    )
    pt = np.array([0, 0, dm])
    hs = k1_h(pt, pt, disk, fc, 1.0, obliquity="kirchhoff")
    check("PH05_disk_discrete_vs_closed_db", db(hs) - db(disk_kirchhoff_onaxis_h(dm, 1.0, fc)))
    hd = disk_kirchhoff_onaxis_h(dm, 200.0, fc)
    img = rf.wavelength(fc) / (4 * math.pi * 2 * dm) * np.exp(-1j * k * 2 * dm)
    check("PH05_disk_200m_vs_image_db", db(hd / img))
    check("PH05_disk_200m_vs_image_deg", math.degrees(np.angle(hd / img)))


# --------------------------------------------------------------------------- PH-06 / PH-07 / PH-09


def test_ph06_plate_rcs():
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, 40, 40)
    pt = np.array([0, 0, 400.0])
    check("PH06_plate_400m_db", db(k1_h(pt, pt, p, 3.5e9, 1.0)))


def test_ph07_focusing_saturation_and_passivity(rng):
    fc = 3.5e9
    lam = rf.wavelength(fc)
    ns = math.ceil(1.0 / (lam / 4))
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, ns, ns)
    pt = np.array([0, 0, 1.0])
    g = focus_phases(pt, pt, p, fc)
    check("PH07_focus_d1_L1_db", db(k1_h(pt, pt, p, fc, g)))
    check(
        "PH07_focus_d1_L1_db",
        20 * math.log10(rect_solid_angle_onaxis(0.5, 0.5, 1.0) / (4 * math.pi)),
    )
    check(
        "PH07_focus_limit_db",
        20 * math.log10(rect_solid_angle_onaxis(1e6, 1e6, 1.0) / (4 * math.pi)),
    )
    # |h| <= 0.5 for 10^3 random passive configurations (monostatic, d = 1 m, L = 1 m)
    bound = ORACLE["PH07_passive_bound_abs"].value
    for _ in range(1000):
        gam = np.exp(2j * np.pi * rng.random(p.n)) * rng.random(p.n)
        assert abs(k1_h(pt, pt, p, fc, gam)) <= bound
    assert abs(k1_h(pt, pt, p, fc, g)) <= bound


def test_ph09_backside_is_exactly_zero():
    p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 8, 8, 0.02, 0.02)
    assert k1_h([0.1, 0.0, -1.0], [0.2, 0.1, 2.0], p, 3.5e9) == 0.0
    assert k1_h([0.1, 0.0, 1.0], [0.2, 0.1, -2.0], p, 3.5e9) == 0.0


# --------------------------------------------------------------------------- PH-10


def test_ph10_two_ray_and_fresnel_limits():
    check("PH10_two_ray_52km_h1p5_db", -db(rf.two_ray_h(52000.0, 200.0, 1.5, 100e6)))
    ec = rf.complex_permittivity(15.0, 1e-3, 100e6)
    check("PH10_grazing_gamma_plus1_abs", float(abs(rf.fresnel_gamma_h(1e-4, ec) + 1)))
    check("PH10_grazing_gamma_plus1_abs", float(abs(rf.fresnel_gamma_v(1e-4, ec) + 1)))
    psi_b = rf.brewster_grazing_angle_rad(4.0)
    check("PH10_brewster_deg", math.degrees(psi_b))
    assert abs(rf.fresnel_gamma_v(psi_b, 4.0 + 0j)) <= 1e-12
    # PEC limit: sigma -> infinity gives Gamma_H -> -1, Gamma_V -> +1
    pec = rf.complex_permittivity(1.0, 1e12, 100e6)
    # (the limit is not uniform at grazing: Gamma_V - 1 ~ -2/(sqrt(eps_c) sin psi))
    for psi in (0.05, 0.3, 1.2):
        assert abs(rf.fresnel_gamma_h(psi, pec) + 1) <= 1e-5
        assert abs(rf.fresnel_gamma_v(psi, pec) - 1) <= 1e-5


# --------------------------------------------------------------------------- PH-11


def test_ph11_knife_edge():
    check("PH11_F0_abs", float(abs(rf.knife_edge_F(0.0))))
    nu = np.linspace(-0.78, 10.0, 20001)
    dev = float(np.max(np.abs(rf.itu_j_db(nu) - rf.knife_edge_loss_db(nu))))
    check("PH11_J_max_dev_db", dev)
    check("PH11_continuity_db", float(rf.knife_edge_loss_db(1e-9) - rf.knife_edge_loss_db(-1e-9)))
    check("PH11_nu10_db", float(rf.knife_edge_loss_db(10.0)))
    # asymptote |F| -> 1/(pi nu sqrt(2)) for large nu
    assert abs(float(abs(rf.knife_edge_F(200.0))) * math.pi * 200.0 * math.sqrt(2.0) - 1.0) <= 1e-4


# --------------------------------------------------------------------------- PH-12 (experimental)


@pytest.mark.experimental
def test_ph12_p526_regression():
    check("PH12_52km_h1p5_excess_db", p526.p526_smooth_excess_db(52.0, 200.0, 1.5, 100.0)[0])
    check("PH12_52km_h10_excess_db", p526.p526_smooth_excess_db(52.0, 200.0, 10.0, 100.0)[0])


# --------------------------------------------------------------------------- PH-13/14/15/25


def test_ph13_quantization_closed_forms_and_quadrature():
    for b, key in ((1, "PH13_eta_b1_db"), (2, "PH13_eta_b2_db"), (3, "PH13_eta_b3_db")):
        check(key, 10 * math.log10(ni.uniform_bbit_eta(b)))
        check(key, ni.alphabet_eta_db(ni.ideal_bbit(b)))
    lab = [np.exp(1j * math.radians(49.64)), np.exp(-1j * math.radians(153.77))]
    check("PH13_lab_pair_db", 10 * math.log10(ni.binary_eta(math.radians(49.64 + 153.77))))
    check("PH13_lab_pair_db", ni.alphabet_eta_db(lab))


def test_ph13_monte_carlo_matches_quadrature(rng):
    """Per-element best choice over N = 4096 uniform channel phases, 100 trials (+-0.15 dB)."""
    n = 4096
    for alphabet in (
        ni.ideal_bbit(1),
        ni.ideal_bbit(2),
        [np.exp(1j * math.radians(49.64)), np.exp(-1j * math.radians(153.77))],
    ):
        g = np.asarray(alphabet)
        losses = []
        for _ in range(100):
            c = np.exp(2j * np.pi * rng.random(n))
            # per-element best choice along the beam direction (asymptotically optimal)
            s = ni.quantize_argmax(np.conj(g), np.angle(c))  # argmax_s Re{Gamma_s c_n}
            losses.append((abs(np.sum(g[s] * c)) / n) ** 2)
        mc = 10 * math.log10(float(np.mean(losses)))
        assert abs(mc - ni.alphabet_eta_db(g)) <= ORACLE["PH13_mc_tol_db"].tol


def test_ph14_abeywickrama():
    check("PH14_bmin0p2_a1p6_db", ni.abeywickrama_eta_db(0.2, 1.6))
    check("PH14_bmin0p8_a1p6_db", ni.abeywickrama_eta_db(0.8, 1.6))
    for phi in (0.0, 1.0, -2.5):
        check(
            "PH14_phi_independence_db",
            ni.abeywickrama_eta_db(0.2, 1.6, phi) - ni.abeywickrama_eta_db(0.2, 1.6),
        )


def test_ph15_efficiency_and_dead_cells(rng):
    check("PH15_eta0p5_db", 20 * math.log10(ni.efficiency_amplitude(0.5)))
    check("PH15_dead10_db", ni.dead_cell_loss_db(0.10))
    # Monte Carlo of the closed form on a focused N = 1024 panel, 200 random maps
    fc = 3.5e9
    lam = rf.wavelength(fc)
    p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 32, 32, lam / 2, lam / 2)
    tx = np.array([-1.0, 0.2, 3.0])
    rx = np.array([1.5, -0.3, 2.5])
    g = focus_phases(tx, rx, p, fc)
    h0 = abs(k1_h(tx, rx, p, fc, g))
    vals = []
    for _ in range(200):
        dead = rng.permutation(p.n)[: round(0.10 * p.n)]
        gd = g.copy()
        gd[dead] = 0.0
        vals.append(abs(k1_h(tx, rx, p, fc, gd)) ** 2)
    check("PH15_dead10_db", 10 * math.log10(float(np.mean(vals)) / h0**2))


def test_ph25_binary_md_lab_pair():
    a = np.exp(1j * math.radians(49.64))
    b = np.exp(-1j * math.radians(153.77))
    m, d = ni.binary_md(a, b)
    assert abs((m + d) - a) <= 1e-15 and abs((m - d) - b) <= 1e-15
    check("PH25_d_db", db(d))
    check("PH25_m_db", db(m))
    check("PH25_m_deg", math.degrees(np.angle(m)))


# --------------------------------------------------------------------------- PH-17


def test_ph17_doppler_values():
    fc = 3.5e9
    check("PH17_los_fd_hz", mob.doppler_hz(30.0, fc))
    rx0 = np.array([50.0, 0.0, 0.0])
    v = np.array([-30.0, 0.0, 0.0])
    check("PH17_ris_fd_hz", mob.doppler_hz(mob.radial_speed_mps(rx0, v, [25.0, 10.0, 0.0]), fc))


def test_ph17_los_phase_over_one_second_and_block_constant_teeth():
    fc = 3.5e9
    k = rf.wavenumber(fc)
    fs = 10_000
    t = np.arange(fs + 1) / fs  # 1 s
    d = 50.0 - 30.0 * t
    phase = -k * d  # analytic carrier phase of the LOS path, -2 pi f_c d(t)/c
    f_inst = np.diff(np.unwrap(phase)) * fs / (2 * math.pi)
    check("PH17_los_fd_hz", float(np.mean(f_inst)))
    # The LOS oracle (friis_h, which carries the propagation phase) follows -2 pi f_c d(t)/c
    # along the 1 s trajectory: the residual phase is wrapped to (-pi, pi] and must vanish.
    # A sign slip (exp(+jkd)) or a lambda/C0 inconsistency would leave a ramp of up to
    # 2*k*30 m ~ 4.4e3 rad here, so this check has teeth.
    y = rf.friis_h(d, fc)
    resid = np.angle(y * np.exp(-1j * phase))
    check("PH17_phase_rad", float(np.max(np.abs(resid))))
    f_y = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * math.pi)
    check("PH17_los_fd_hz", float(np.mean(f_y)))  # positive on approach (section 3.1)
    # teeth: the conjugate (wrong-sign) coefficient fails the same residual check
    assert float(np.max(np.abs(np.angle(np.conj(y) * np.exp(-1j * phase))))) > 1.0


def test_ph17_teeth_block_constant_over_the_t9_window():
    """D7: the PH-17 teeth value -148.3 +- 1 Hz is the MEAN instantaneous frequency of the v2
    staircase (geometry frozen per 2 ms block) over the T9 window, 0.2 s at 10 kS/s.

    Computed two ways: sample by sample (wrapped phase steps of the staircase LOS phase) and
    by the closed form (99 block boundaries of wrap(k v T) over 1999 steps). Teeth: the
    long-run alias wrap(k v T)/(2 pi T) = -149.77 Hz is outside the tolerance, so a test that
    measures over a different window would not see -148.3 Hz.
    """
    fc, v, fs = PH17_T9_FC_HZ, PH17_T9_SPEED_MPS, PH17_T9_FS
    n = int(PH17_T9_WINDOW_S * fs)
    assert n == 2000
    spb = int(fs * PH17_T9_BLOCK_S)  # 20 samples per block (exact)
    k_blk = np.arange(n) // spb  # block index of each sample, exact integers
    t_blk = k_blk * float(PH17_T9_BLOCK_S)
    d = PH17_T9_RX0_M[0] - v * t_blk  # staircase Tx-Rx distance (Tx at the origin)
    yb = np.exp(-1j * rf.wavenumber(fc) * d)
    fb = float(np.mean(np.angle(yb[1:] * np.conj(yb[:-1])))) * float(fs) / (2 * math.pi)
    closed = mob.block_constant_mean_doppler_hz(fc, v, PH17_T9_BLOCK_S, fs, n)
    assert abs(fb - closed) <= 1e-9
    check("PH17_block_constant_hz", closed)
    close_jump = mob.block_constant_jump_rad(fc, v, PH17_T9_BLOCK_S)
    assert abs(close_jump - 2 * math.pi * mob.doppler_hz(v, fc) * 0.002) <= 1e-12
    alias = mob.block_constant_alias_hz(fc, v, PH17_T9_BLOCK_S)
    o = ORACLE["PH17_block_constant_hz"]
    assert abs(alias - o.value) > o.tol  # -149.77 Hz: the window matters (D7)
    # and the correct (continuous) geometry gives +350.242 Hz over the same window
    t9 = np.arange(n) / float(fs)
    yc = np.exp(-1j * rf.wavenumber(fc) * (PH17_T9_RX0_M[0] - v * t9))
    check(
        "PH17_los_fd_hz",
        float(np.mean(np.angle(yc[1:] * np.conj(yc[:-1])))) * float(fs) / (2 * math.pi),
    )


# --------------------------------------------------------------------------- PH-20


@pytest.mark.parametrize(
    ("taps", "band", "key"),
    [
        (8, 0.1, "PH20_kaiser8_b0p1_db"),
        (16, 0.3, "PH20_kaiser16_b0p3_db"),
        (32, 0.4, "PH20_kaiser32_b0p4_db"),
    ],
)
def test_ph20_kaiser_fd_rows(taps, band, key):
    mus = np.linspace(0.0, 0.999, 101)
    worst = max(rf.fd_inband_error_db(*rf.kaiser_sinc_taps(taps, 8.0, mu), band) for mu in mus)
    check(key, worst)
    assert worst <= ORACLE["PH20_fd_accept_db"].value


def test_ph20_teeth_lagrange4_fails_at_0p4():
    worst = max(
        rf.fd_inband_error_db(*rf.lagrange4_taps(mu), 0.4) for mu in np.linspace(0.0, 0.999, 101)
    )
    check("PH20_lagrange4_b0p4_db", worst)
    assert worst > ORACLE["PH20_fd_accept_db"].value


def test_ph20_narrowband_band_edge_error_bound():
    """At pi*B*dtau = 0.05 the per-path envelope error at the band edge is 20log10(0.025)."""
    x = ORACLE["PH20_narrowband_switch_rad"].value
    edge = 20 * math.log10(abs(1 - np.exp(1j * x / 2)))  # +-dtau/2 about the midpoint delay
    assert edge <= ORACLE["PH20_narrowband_edge_db"].value
    assert abs(edge - 20 * math.log10(0.025)) <= 1e-3


# --------------------------------------------------------------------------- PH-21 / PH-26


def test_ph21_noise_floor():
    check("PH21_noise_1MSps_dbm", rf.thermal_noise_dbm(1e6, 0.0))
    # the cross-RX bound 0.016 is 5 sigma of |rho| over 1e5 independent complex samples
    assert abs(mob.corr_5sigma(100_000) - ORACLE["PH21_cross_rx_corr"].tol) <= 5e-4


def test_ph26_point_vs_patch_specular():
    """16 x 16 lambda/2 panel at 3.5 GHz, 2 m, uniform Gamma: specular within 0.05 dB."""
    fc = 3.5e9
    lam = rf.wavelength(fc)
    p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 16, 16, lam / 2, lam / 2)
    q = subdivide(p, lam / 2, lam / 2, lam / 5)
    for th_deg in (0.0, 30.0):
        th = math.radians(th_deg)
        tx = np.array([-2 * math.sin(th), 0, 2 * math.cos(th)])
        rx = np.array([2 * math.sin(th), 0, 2 * math.cos(th)])
        check("PH26_specular_db", db(k1_h(tx, rx, p, fc)) - db(k1_h(tx, rx, q, fc)))


def test_ph26_patch_quadrature_converges_to_subaperture_sinc():
    """Far field, 45 deg retro: the patch integral tends to sinc(p (sin ti + sin ts)/lambda).

    Independent check of :func:`subdivide`: with many sub-cells the point-minus-patch
    difference approaches -20 log10|sinc(0.5 * 2 sin 45)| = 8.918 dB (the documented D6
    limit 8.92 dB).
    """
    fc = 3.5e9
    lam = rf.wavelength(fc)
    p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 4, 4, lam / 2, lam / 2)
    th = math.radians(45)
    pt = np.array([-2000 * math.sin(th), 0, 2000 * math.cos(th)])
    g = focus_phases(pt, pt, p, fc)
    sinc_db = -20 * math.log10(abs(float(np.sinc(0.5 * 2 * math.sin(th)))))
    q = subdivide(p, lam / 2, lam / 2, lam / 2 / 41)
    diff = db(k1_h(pt, pt, p, fc, g)) - db(k1_h(pt, pt, q, fc, g))
    assert abs(diff - sinc_db) <= 0.01
    check("PH26_retro45_sinc_limit_db", sinc_db)


_RETRO45_SIN_SUM = 2.0 * math.sin(math.radians(45.0))  # sin(theta_i) + sin(theta_s), retro


def test_ph26_retro45_closed_forms():
    """D6 values from the closed forms: a lambda/2 cell split into 3 sub-cells per axis
    (Delta <= lambda/5 gives ceil(2.5) = 3) loses -20 log10|(1 + 2 cos psi)/3| = 8.109 dB at
    45 deg retro (psi = 2 pi (1/6) sqrt 2); the continuous patch loses
    -20 log10|sinc(sqrt(2)/2)| = 8.918 dB. Both to the printed decimals of D6."""
    af3 = float(subaperture_factor(0.5, _RETRO45_SIN_SUM, 3))
    psi = 2 * math.pi * (0.5 / 3) * _RETRO45_SIN_SUM
    assert abs(af3 - (1 + 2 * math.cos(psi)) / 3) <= 1e-15
    d3 = -20 * math.log10(abs(af3))
    assert abs(d3 - ORACLE["PH26_retro45_db"].value) <= 5e-3
    check("PH26_retro45_db", d3)
    d_inf = -20 * math.log10(abs(float(subaperture_factor(0.5, _RETRO45_SIN_SUM))))
    check("PH26_retro45_sinc_limit_db", d_inf)
    # the m-point factor tends to the sinc as m grows (and m = 1 is the point model)
    assert float(subaperture_factor(0.5, _RETRO45_SIN_SUM, 1)) == 1.0
    big = float(subaperture_factor(0.5, _RETRO45_SIN_SUM, 4001))
    assert abs(big - float(np.sinc(0.5 * _RETRO45_SIN_SUM))) <= 1e-7


def test_ph26_retro45_k1_matches_closed_form_in_the_cell_far_field():
    """K1 point minus po_patch (Delta = lambda/5, 3 x 3 sub-cells) at 45 deg retro, focused:
    the 16 x 16 lambda/2 panel at 2 km and a single lambda/2 cell at 2 m (the cell is in its
    own far field there) both give the 3 x 3 closed form, hence 8.11 +- 0.1 dB (D6)."""
    fc = 3.5e9
    lam = rf.wavelength(fc)
    closed = -20 * math.log10(abs(float(subaperture_factor(0.5, _RETRO45_SIN_SUM, 3))))
    th = math.radians(45)
    for n_side, dist, tol_db in ((16, 2000.0, 1e-3), (1, 2.0, 0.01)):
        p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], n_side, n_side, lam / 2, lam / 2)
        q = subdivide(p, lam / 2, lam / 2, lam / 5)
        assert q.n == 9 * p.n
        pt = np.array([-dist * math.sin(th), 0, dist * math.cos(th)])
        g = focus_phases(pt, pt, p, fc)
        diff = db(k1_h(pt, pt, p, fc, g)) - db(k1_h(pt, pt, q, fc, g))
        assert abs(diff - closed) <= tol_db, (n_side, dist, diff)
        check("PH26_retro45_db", diff)


def test_ph26_retro45_panel_at_2m_follows_the_local_cell_angles():
    """The PH-26 panel (16 x 16 lambda/2, 3.5 GHz) at 2 m, 45 deg retro, focused: its cells
    see 35..55 deg, so the panel-level point-minus-patch difference is the amplitude-weighted
    mean of each cell's own 3 x 3 factor, sum|a_n| / sum|a_n| F_n (within 0.01 dB), which is
    7.68 dB, NOT the far-field 8.11 dB of D6 (outside +-0.1 dB). Recorded for the lead: the D6
    regression value applies to a cell (or the panel) in the far field, see the WP4 report."""
    fc = 3.5e9
    lam = rf.wavelength(fc)
    p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 16, 16, lam / 2, lam / 2)
    q = subdivide(p, lam / 2, lam / 2, lam / 5)
    th = math.radians(45)
    pt = np.array([-2.0 * math.sin(th), 0, 2.0 * math.cos(th)])
    g = focus_phases(pt, pt, p, fc)
    diff = db(k1_h(pt, pt, p, fc, g)) - db(k1_h(pt, pt, q, fc, g))
    a = k1_terms(pt, pt, p, fc, g)  # co-phased: real and positive
    assert np.max(np.abs(np.angle(a))) <= 1e-12
    f = local_subaperture_factor(pt, pt, p, fc, lam / 2, lam / 2, 3, 3)
    model = -20 * math.log10(float(np.sum(f * a.real) / np.sum(a.real)))
    assert abs(diff - model) <= 0.01
    o = ORACLE["PH26_retro45_db"]
    assert abs(diff - o.value) > o.tol, f"2 m panel-level difference {diff:.3f} dB"


# --------------------------------------------------------------------------- legacy_v1 (D4)


def test_legacy_v1_teeth_vs_k1_in_the_v1_default_geometry():
    """Section 5.14 / D4: legacy_v1 (whole-panel area per cell, abs(cos), q = 0.285, v1's
    c = 3e8) is +24.39 +- 0.05 dB above K1 in the v1 default geometry. Teeth: v1 evaluated
    with C0 instead of v1's constant lands outside the tolerance (+24.48 dB)."""
    s = v1_default_scene()
    k1 = db(k1_h(s.tx, s.rx, s.panel, s.fc_hz, s.gamma))
    v1 = db(legacy_v1_h(s.tx, s.rx, s.panel, s.fc_hz, s.panel_area_m2, s.gamma))
    check("LEGACY_v1_vs_k1_db", v1 - k1)
    v1_c0 = db(legacy_v1_h(s.tx, s.rx, s.panel, s.fc_hz, s.panel_area_m2, s.gamma, c_m_per_s=C0))
    o = ORACLE["LEGACY_v1_vs_k1_db"]
    assert abs((v1_c0 - k1) - o.value) > o.tol
    # v1 admits back-side cells (abs cos): mirroring the Tx behind the panel keeps |h|
    tx_back = s.tx * np.array([-1.0, 1.0, 1.0])
    back = legacy_v1_h(tx_back, s.rx, s.panel, s.fc_hz, s.panel_area_m2, s.gamma)
    assert abs(abs(back) - 10 ** (v1 / 20)) <= 1e-12 * 10 ** (v1 / 20)
    assert k1_h(tx_back, s.rx, s.panel, s.fc_hz, s.gamma) == 0.0  # K1 does not (PH-09)


# --------------------------------------------------------------------------- RT / V-CFO


def test_rt05_switch_offset_at_2km():
    check("RT05_samples_2km_23p04", 2000.0 / C0 * 23.04e6)


def test_rt09_teeth_v2_tick_phase_step():
    check("RT09_v2_step_rad", rf.v2_tick_boundary_phase_step_rad(500.0, V_CFO_FS, V_CFO_TAU))
    # the correct per-sample step is 2 pi df / f_s = 0.534 rad, so v2's boundary step is wrong
    assert (
        abs(2 * math.pi * 500 / 5880 - rf.v2_tick_boundary_phase_step_rad(500.0, 5880, V_CFO_TAU))
        > 0.3
    )


def test_vcfo_truncation_slope_and_crb():
    check("VCFO2_slope", float(rf.legacy_truncation_cfo_slope(V_CFO_FS, V_CFO_TAU)))
    for snr_db, key in (
        (0, "VCFO4_crb_0db_hz"),
        (10, "VCFO4_crb_10db_hz"),
        (20, "VCFO4_crb_20db_hz"),
    ):
        check(key, rf.crb_frequency_std_hz(5880.0, 10 ** (snr_db / 10), 2000))


# --------------------------------------------------------------------------- ST


def test_st02_lcr_at_minus_10db():
    check("ST02_lcr_m10db_hz", float(mob.rayleigh_lcr_hz(10 ** (-10 / 20), 50.0)))


def test_st01_start_point_ks_threshold_is_the_1pct_critical_value():
    assert abs(mob.ks_critical(10_000, 0.01) - ORACLE["ST01_ks_start"].tol) <= 5e-5


def test_v_cfo_constants():
    assert Fraction(V_CFO_FS) * V_CFO_TAU == Fraction(1176, 100)


# --------------------------------------------------------------------------- V-WCL W1


@pytest.mark.parametrize(
    ("fc", "key"),
    [
        (1e9, "W1_wcl_ratio_1GHz_db"),
        (2.4e9, "W1_wcl_ratio_2p4GHz_db"),
        (3.5e9, "W1_wcl_ratio_3p5GHz_db"),
    ],
)
def test_w1_wcl_eq2_over_k1_ratio_at_broadside(fc, key):
    """Section 5.14 teeth of ``legacy_wcl_eq2``: -0.52 / -8.1 / -11.4 dB at 1 / 2.4 / 3.5 GHz.

    Computed two ways: the closed form ``(pi lambda)^2`` and the letter's per-element
    amplitude divided by a one-cell K1 panel directly in front of Tx and Rx (cos = 1).
    """
    from tests.oracles.ris import wcl_eq2_amplitude, wcl_eq2_over_k1_power_ratio

    lam = rf.wavelength(fc)
    ratio = float(wcl_eq2_over_k1_power_ratio(lam, 1.0, 1.0))
    check(key, 10 * math.log10(ratio))
    w, h = lam / 4, 3.0
    cell = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 1, 1, w, w)
    pt = np.array([0.0, 0.0, h])  # Tx = Rx on the cell normal at distance h (monostatic)
    k1 = abs(k1_h(pt, pt, cell, fc))
    letter = float(wcl_eq2_amplitude(lam, w, h, h, h))
    assert abs((letter / k1) ** 2 / ratio - 1.0) <= ORACLE["W1_ratio_rel"].tol
