"""Oracle self-tests against ``physics_model_checks.json`` (written by ``pm_checks.py``).

Each test rebuilds the scratch geometry with the oracle functions and compares with the
stored value to its printed rounding (half a unit in the last stored decimal). Monte Carlo
entries that depend on the legacy global RNG are not reproduced (section 6.1).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from tests.oracles import nonideal as ni
from tests.oracles import p526, rf
from tests.oracles._data import pm_checks
from tests.oracles.ris import (
    ellingson_qgamma_h,
    focus_phases,
    k1_h,
    legacy_v2_h,
    los_h,
    make_panel,
    make_panel_extent,
    plate_monostatic_h_abs,
    rect_solid_angle_onaxis,
    tang_eq8_h_abs,
)

pytestmark = pytest.mark.physics

J = pm_checks()


def close(value: float, stored: float, decimals: int) -> None:
    """``value`` rounds to ``stored`` at ``decimals`` (half-unit tolerance plus float slack)."""
    tol = 0.5 * 10.0**-decimals + 1e-9
    assert abs(value - stored) <= tol, f"{value!r} vs stored {stored!r} (tol {tol:g})"


def db(h: complex | np.ndarray) -> float:
    return float(rf.db20(h))


# --------------------------------------------------------------------------- T1 / T17 (PH-01)


@pytest.mark.parametrize(
    ("fc", "d", "key"),
    [
        (3.5e9, 10.0, "T1_friis_dB_fc3500MHz_d10.00m"),
        (2.4e9, math.hypot(10, 6), "T1_friis_dB_fc2400MHz_d11.66m"),
        (100e6, 52000.0, "T1_friis_dB_fc100MHz_d52000.00m"),
    ],
)
def test_ph01_friis_matches_pm_checks_t1(fc, d, key):
    close(db(rf.friis_h(d, fc)), J[key], 3)


@pytest.mark.parametrize("fc", [3.5e9, 2.4e9, 100e6])
def test_t17_ants_eq1_missing_factor(fc):
    key = f"T17_ANTS_eq1_missing_factor_(4pi/lam)^2_dB_fc{fc / 1e6:g}MHz"
    close(20 * math.log10(4 * math.pi / rf.wavelength(fc)), J[key], 2)


# --------------------------------------------------------------------------- T2 (PH-04)

_T2 = {
    "near_specular_5m": (5.0, 5.0, 30.0),
    "far_specular_200m": (200.0, 200.0, 30.0),
    "near_anomalous_5m": (5.0, 5.0, -10.0),
}


def _t2_geometry(d1: float, d2: float, out_deg: float) -> tuple[np.ndarray, np.ndarray]:
    inc = math.radians(30)
    tx = np.array([-d1 * math.sin(inc), 0, d1 * math.cos(inc)])
    o = math.radians(out_deg)
    return tx, np.array([d2 * math.sin(o), 0, d2 * math.cos(o)])


@pytest.mark.parametrize("label", list(_T2))
def test_ph04_refinement_matches_pm_checks_t2(label):
    """K1 point sums and the legacy v2 kernel on a 1 m^2 plate refined from 16 to 16384 cells."""
    tx, rx = _t2_geometry(*_T2[label])
    fc = 3.5e9
    for row, n_side in zip(
        J[f"T2_refinement_uniformGamma_{label}"], (4, 8, 16, 32, 64, 128), strict=True
    ):
        p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, n_side, n_side)
        assert row["N"] == p.n
        close(db(k1_h(tx, rx, p, fc)), row["aperture_dB"], 3)
        close(db(legacy_v2_h(tx, rx, p, fc)), row["v2_dB"], 3)
        close(db(ellingson_qgamma_h(tx, rx, p, fc)), row["ellingson_q0_area_free_dB"], 3)
        close(1.0 / n_side / rf.wavelength(fc), row["cell_over_lambda"], 3)


def test_ph04_focused_refinement_matches_pm_checks_t2():
    tx, rx = _t2_geometry(5.0, 5.0, -10.0)
    fc = 3.5e9
    for row, n_side in zip(J["T2_refinement_focused_5m"], (8, 16, 32, 64, 128), strict=True):
        p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, n_side, n_side)
        g = focus_phases(tx, rx, p, fc)
        close(db(k1_h(tx, rx, p, fc, g)), row["aperture_dB"], 3)
        close(db(legacy_v2_h(tx, rx, p, fc, g * 1j)), row["v2_dB"], 3)


# --------------------------------------------------------------------------- T3 (PH-03)


def test_ph03_far_field_scaling_matches_pm_checks_t3():
    fc = 3.5e9
    lam = rf.wavelength(fc)
    d = 2000.0
    tx = np.array([0, 0, d])
    rx = np.array([d * math.sin(math.radians(20)), 0, d * math.cos(math.radians(20))])
    for row, n_side in zip(J["T3_far_field_beamforming_scaling"], (10, 20, 40), strict=True):
        L = n_side * lam / 2
        p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], L, L, n_side, n_side)
        h = k1_h(tx, rx, p, fc, focus_phases(tx, rx, p, fc))
        close(db(h), row["sum_dB"], 3)
        eq8 = tang_eq8_h_abs(p.n, float(p.area[0]), d, d, 0.0, math.radians(20))
        close(20 * math.log10(eq8), row["Tang2022_eq8_dB"], 3)


# --------------------------------------------------------------------------- T4 (PH-05)


def test_ph05_disk_closed_form_vs_image_matches_pm_checks_t4():
    from tests.oracles.ris import disk_kirchhoff_onaxis_h

    fc = 3.5e9
    lam = rf.wavelength(fc)
    k = rf.wavenumber(fc)
    dm = 2.0
    img = lam / (4 * math.pi * 2 * dm)
    for row in J["T4_disk_closed_form_vs_image_monostatic_d2m"]:
        hd = disk_kirchhoff_onaxis_h(dm, row["R_m"], fc)
        close(db(hd / img), row["abs_ratio_to_image_dB"], 3)
        ph = math.degrees(np.angle(hd / (img * np.exp(-1j * k * 2 * dm))))
        close(ph, row["phase_err_deg_vs_image"], 2)


def test_ph05_discrete_disk_matches_pm_checks_t4():
    from tests.oracles.ris import disk_kirchhoff_onaxis_h

    fc = 3.5e9
    dm = 2.0
    R = 1.0
    p_tx = np.array([0, 0, dm])
    for row in J["T4_discrete_disk_vs_closed_form_R1m_d2m"]:
        n_side = row["Nside"]
        p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 2 * R, 2 * R, n_side, n_side)
        mask = np.hypot(p.cells[:, 0], p.cells[:, 1]) <= R
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
        hs = k1_h(p_tx, p_tx, disk, fc, 1.0, obliquity="kirchhoff")
        hd = disk_kirchhoff_onaxis_h(dm, R, fc)
        close(db(hs), row["discrete_dB"], 3)
        close(db(hd), row["closed_form_dB"], 3)
        close(math.degrees(np.angle(hs / hd)), row["phase_diff_deg"], 2)


def test_ph05_oblique_plate_vs_image_matches_pm_checks_t4():
    fc = 3.5e9
    lam = rf.wavelength(fc)
    d1 = d2 = 3.0
    th = math.radians(40)
    tx = np.array([-d1 * math.sin(th), 0, d1 * math.cos(th)])
    rx = np.array([d2 * math.sin(th), 0, d2 * math.cos(th)])
    img = rf.image_theory_h(d1, d2, fc)
    for row in J["T4_oblique_specular_plate_vs_image_40deg_3m"]:
        L = row["L_m"]
        ns = round(L / (lam / 6))
        p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], L, L, ns, ns)
        assert p.n == row["cells"]
        hj = k1_h(tx, rx, p, fc, 1.0)
        hn = k1_h(tx, rx, p, fc, 1.0, huygens_j=False)
        close(db(hj / img), row["ratio_dB"], 3)
        close(math.degrees(np.angle(hj / img)), row["phase_err_deg_with_j"], 2)
        close(math.degrees(np.angle(hn / img)), row["phase_err_deg_without_j"], 2)


# --------------------------------------------------------------------------- T5 / T7 (PH-06/07)


def test_ph06_plate_matches_pm_checks_t5():
    fc = 3.5e9
    lam = rf.wavelength(fc)
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, 40, 40)
    tx = np.array([0, 0, 400.0])
    close(db(k1_h(tx, tx, p, fc, 1.0)), J["T5_plate_monostatic_far_sum_dB"], 3)
    close(
        20 * math.log10(plate_monostatic_h_abs(1.0, 400.0)),
        J["T5_plate_closed_form_A_over_4pi_d2_dB"],
        3,
    )
    sigma = 4 * math.pi / lam**2
    radar = 10 * math.log10(lam**2 * sigma / ((4 * math.pi) ** 3 * 400.0**4))
    close(radar, J["T5_radar_eq_dB"], 3)


def test_ph07_focusing_saturation_matches_pm_checks_t7():
    fc = 3.5e9
    lam = rf.wavelength(fc)
    for row in J["T7_monostatic_focus_d1m_3p5GHz"]:
        L = row["L_m"]
        om = rect_solid_angle_onaxis(L / 2, L / 2, 1.0)
        close(20 * math.log10(om / (4 * math.pi)), row["exact_Omega_over_4pi_dB"], 3)
        close(20 * math.log10(L * L / (4 * math.pi)), row["far_field_formula_dB"], 3)
        if "discrete_focus_sum_dB" in row:
            ns = math.ceil(L / (lam / 4))
            p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], L, L, ns, ns)
            pt = np.array([0, 0, 1.0])
            g = focus_phases(pt, pt, p, fc)
            close(db(k1_h(pt, pt, p, fc, g)), row["discrete_focus_sum_dB"], 3)
            close(db(legacy_v2_h(pt, pt, p, fc, g * 1j)), row["v2_isotropic_cells_dB"], 3)
    close(20 * math.log10(0.5), J["T7_limit_infinite_plate_dB"], 3)


# --------------------------------------------------------------------------- T9 (PH-17)


def test_ph17_doppler_matches_pm_checks_t9():
    fc = 3.5e9
    lam = rf.wavelength(fc)
    k = rf.wavenumber(fc)
    fs = 10e3
    t = np.arange(int(0.2 * fs)) / fs
    v = 30.0
    tx = np.zeros(3)
    rx0 = np.array([50.0, 0.0, 0.0])
    rx_t = rx0[None, :] + np.outer(t, [-v, 0, 0])
    r = np.linalg.norm(rx_t - tx, axis=1)
    y = np.exp(-1j * k * r)
    f_inst = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * math.pi)
    close(v / lam, J["T9_LOS_doppler_expected_Hz"], 3)
    close(float(f_inst.mean()), J["T9_LOS_doppler_measured_Hz_mean"], 3)
    tb_s = 0.002
    tb = np.floor(t / tb_s) * tb_s
    rb = np.linalg.norm((rx0[None, :] + np.outer(tb, [-v, 0, 0])) - tx, axis=1)
    yb = np.exp(-1j * k * rb)
    fb = np.angle(yb[1:] * np.conj(yb[:-1])) * fs / (2 * math.pi)
    close(2 * math.pi * (v / lam) * tb_s, J["T9_blockconst_phase_jump_per_block_rad"], 3)
    close(float(fb.mean()), J["T9_blockconst_apparent_mean_freq_Hz"], 2)
    # RIS path: 0.5 m x 0.5 m, 12 x 12 panel at (25, 10, 0) facing -y, focused on rx0
    p = make_panel_extent([25.0, 10.0, 0.0], [0, -1, 0], [1, 0, 0], 0.5, 0.5, 12, 12)
    g = focus_phases(tx, rx0, p, fc)
    yr = np.array([k1_h(tx, rx_t[i], p, fc, g) for i in range(0, len(t), 10)])
    fr = np.angle(yr[1:] * np.conj(yr[:-1])) * (fs / 10) / (2 * math.pi)
    u = (np.array([25.0, 10.0, 0.0]) - rx0) / np.linalg.norm(np.array([25.0, 10.0, 0.0]) - rx0)
    close(float(np.dot([-v, 0, 0], u) / lam), J["T9_RIS_path_doppler_expected_Hz_at_t0"], 3)
    close(float(fr[0]), J["T9_RIS_path_doppler_measured_Hz_first"], 3)


@pytest.mark.parametrize(("fc", "t_blk"), [(5.9e9, 0.002), (3.5e9, 0.010)])
def test_ph18_chord_error_budget_matches_pm_checks_t9(fc, t_blk):
    """Section 4.4 budget forms vs the T9 interpolation-bound numbers (v = 30 m/s, r = 10 m):
    chord phase error k r'' T^2/8 (deg), the T_curv for a 1 deg budget, and v T / lambda."""
    from tests.oracles import mobility as mob

    v, r = 30.0, 10.0
    tag = f"fc{fc / 1e9:g}GHz"
    err = mob.chord_phase_error_rad(fc, v, r, t_blk)
    close(math.degrees(err), J[f"T9_interp_err_bound_{tag}_v{v}_r{r}_T{t_blk * 1e3:g}ms_deg"], 3)
    t_1deg = mob.t_curv_s(math.radians(1.0), r, fc, v)
    close(t_1deg * 1e3, J[f"T9_max_block_for_1deg_{tag}_ms"], 3)
    assert abs(mob.chord_phase_error_rad(fc, v, r, t_1deg) - math.radians(1.0)) <= 1e-15
    close(
        v * t_blk / rf.wavelength(fc),
        J[f"T9_displacement_per_block_in_lambda_{tag}_T{t_blk * 1e3:g}ms"],
        3,
    )


# --------------------------------------------------------------------------- T13 (wideband RIS)


def test_k1_wideband_delay_spread_and_flatness_match_pm_checks_t13():
    """Focused 1 m^2 panel (46 x 46), 3.5 GHz: RIS delay spread and the |H(f)| flatness of the
    exact per-cell response over +-80 MHz (section 4.5 wideband reference)."""
    from tests.oracles.ris import k1_delays_s, k1_frequency_response

    fc = 3.5e9
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 1.0, 1.0, 46, 46)
    tx = np.array([-2.0, 0.0, 2.0])
    rx = np.array([1.0, 0.5, 1.5])
    g = focus_phases(tx, rx, p, fc)
    tau = k1_delays_s(tx, rx, p)
    close(float(tau.max() - tau.min()) * 1e9, J["T13_RIS_delay_spread_ns"], 3)
    f = np.linspace(-0.4 * 200e6, 0.4 * 200e6, 801)
    H = k1_frequency_response(tx, rx, p, fc, f, g)
    assert abs(H[400] - k1_h(tx, rx, p, fc, g)) <= 1e-13 * abs(H[400])  # f = 0 is K1
    mag = rf.db20(H)
    close(float(mag.max() - mag.min()), J["T13_flatness_dB_exact_over_pm80MHz"], 3)


# --------------------------------------------------------------------------- T10/T11 (PH-13/14/16)


def test_ph13_quantization_formula_matches_pm_checks_t10():
    for row in J["T10_quantisation_loss"]:
        close(10 * math.log10(ni.uniform_bbit_eta(row["bits"])), row["eta_formula_dB"], 3)
        close(ni.alphabet_eta_db(ni.ideal_bbit(row["bits"])), row["eta_formula_dB"], 3)


def test_ph16_phase_wrap_matches_pm_checks_t10():
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 0.3, 0.3, 8, 8)
    tx = np.array([0.2, 0.1, 2.0])
    rx = np.array([-0.5, 0.3, 1.5])
    g = focus_phases(tx, rx, p, 3.5e9)
    h1 = k1_h(tx, rx, p, 3.5e9, g)
    # m = 1: the PH-16 tolerance 1e-15 |h| holds for the float evaluation of exp(j 2 pi m)
    assert abs(h1 - k1_h(tx, rx, p, 3.5e9, g * np.exp(1j * 2 * np.pi))) <= 1e-15 * abs(h1)
    # m = 7 (the scratch value): exp(j*2*pi*7) itself carries ~1.7e-15 rounding in float64
    # (sin(fl(14 pi)) = -1.7e-15), so the stored scratch error is 1.7e-15 |h|. Exact 1e-15
    # needs the phase to be reduced or quantized before Gamma is formed (reported to WP2).
    h7 = k1_h(tx, rx, p, 3.5e9, g * np.exp(1j * 2 * np.pi * 7))
    assert abs(h1 - h7) <= 1e-14 * abs(h1)
    assert J["T10_phase_wrap_invariance_abs_err"] <= 1e-14 * abs(h1)
    assert abs(abs(h1 - h7) - J["T10_phase_wrap_invariance_abs_err"]) <= 1e-17


@pytest.mark.parametrize(("bmin", "alpha"), [(0.2, 1.6), (0.8, 1.6), (0.5, 1.6), (0.2, 3.0)])
def test_ph14_abeywickrama_matches_pm_checks_t11(bmin, alpha):
    close(
        ni.abeywickrama_eta_db(bmin, alpha),
        J[f"T11_Abeywickrama_eta_bmin{bmin}_alpha{alpha}_dB"],
        3,
    )


# --------------------------------------------------------------------------- T12 (antennas)


@pytest.mark.parametrize(
    ("a", "quoted"),
    [
        (3, "8 (9.03 dBi)"),
        (62, "21 dB"),
        (13, "14.5 dB"),
        (53.8, "20.4 dB"),
        (63.4, "21.1 dB"),
        (1, "cos pattern"),
    ],
)
def test_t12_cos_power_peak_gain_matches_pm_checks(a, quoted):
    from tests.oracles.ris import cos_power

    g = cos_power(a, [0, 0, 1])(np.array([[0.0, 0.0, 1.0]]))[0]
    close(10 * math.log10(g), J[f"T12_cos^{a}_gain_dBi(quoted {quoted})"], 2)


def test_t12_cos_power_normalization_integrates_to_4pi():
    """G_0 = 2(alpha+1) makes the pattern integrate to 4 pi over the sphere."""
    from scipy.integrate import quad

    from tests.oracles.ris import cos_power

    for a in (1.0, 3.0, 53.8):
        g = cos_power(a, [0, 0, 1])
        val = quad(
            lambda th, g=g: float(g(np.array([[0, 0, math.cos(th)]]))[0]) * math.sin(th), 0, math.pi
        )[0]
        assert abs(2 * math.pi * val - 4 * math.pi) <= 1e-9 * 4 * math.pi


# --------------------------------------------------------------------------- T15 (PH-10/12, V-FM)


def test_ph10_ph12_fm_direct_path_matches_pm_checks_t15():
    fc = 100e6
    lam = rf.wavelength(fc)
    for row in J["T15_FM_direct_path_loss"]:
        dk, hr = row["d_km"], row["hr_m"]
        close(-20 * math.log10(lam / (4 * math.pi * dk * 1000)), row["FSPL_dB"], 2)
        close(-db(rf.two_ray_h(dk * 1000, 200.0, hr, fc)), row["two_ray_flat_dB"], 2)
        exc, mode = p526.p526_smooth_excess_db(dk, 200.0, hr, 100.0)
        assert mode == row["P526_mode"]
        close(exc, row["P526_smooth_excess_dB"], 2)
        close(-20 * math.log10(lam / (4 * math.pi * dk * 1000)) + exc, row["P526_total_dB"], 2)
    for hr, key in (
        (1.5, "T15_FM_radio_horizon_km_h200_h1.5"),
        (34.9, "T15_FM_radio_horizon_km_h200_h34.9"),
    ):
        close(p526.radio_horizon_km(200.0, hr), J[key], 2)


def test_vfm_ris_path_with_ground_factors_matches_pm_checks_t15():
    """Focused-plate RIS bound with flat-ground image factors per hop (V-FM-1 inputs)."""
    fc = 100e6
    k = rf.wavenumber(fc)
    tx = np.array([0.0, 200.0, 100.0])
    ris_c = np.array([2000.0, 34.9, 52014.9])
    d1 = float(np.linalg.norm(ris_c - tx))
    close(d1, J["T15_FM_d_TX_RIS_m"], 1)
    a_ris = 30 * 0.25 * 30 * 0.25
    assert a_ris == J["T15_FM_RIS_area_m2"]
    for row in J["T15_FM_RIS_path_loss"]:
        d2 = row["d2_m"]
        pl = -20 * math.log10(a_ris / (4 * math.pi * d1 * d2))
        close(pl, row["RIS_free_space_PL_dB(upper bound, focused, cos=1)"], 2)
        f1 = abs(1 - np.exp(-1j * k * 2 * 200.0 * 34.9 / d1))
        for hr in (1.5, 10.0):
            f2 = abs(1 - np.exp(-1j * k * 2 * 34.9 * hr / d2))
            close(
                pl - 20 * math.log10(f1) - 20 * math.log10(f2),
                row["with_flat_ground_factors_dB"][f"hr{hr}"],
                2,
            )


# --------------------------------------------------------------------------- T16 / T18 / T19


def test_ph02_v1_default_geometry_k1_and_v2_match_pm_checks_t16():
    fc = 2399996950.0
    pitch = 0.06
    pos = np.array(
        [
            [0.0, 5.0 + 0.025 + 0.01 + n * pitch, 5.0 + 0.025 + 0.01 + m * pitch]
            for m in range(16)
            for n in range(16)
        ]
    )
    from tests.oracles.ris import Panel

    idx = np.arange(256, dtype=np.int64)
    p = Panel(
        pos,
        np.full(256, pitch**2),
        np.array([1.0, 0, 0]),
        np.array([0, 1.0, 0]),
        np.array([0, 0, 1.0]),
        idx,
        idx,
        idx,
    )
    tx = np.array([5.0, 2.0, 5.0])
    rx = np.array([15.0, 8.0, 5.0])
    g = np.exp(1j * np.pi / 4)
    close(db(los_h(tx, rx, fc)), J["T16_v1cfg_LOS_dB"], 2)
    close(db(k1_h(tx, rx, p, fc, g)), J["T16_v1cfg_RIS_recommended_dB"], 2)
    close(db(legacy_v2_h(tx, rx, p, fc, g)), J["T16_v1cfg_RIS_v2_dB"], 2)
    close(pitch / rf.wavelength(fc), J["T16_v1cfg_pitch_over_lambda"], 3)


def test_ph05_huygens_j_effect_on_los_plus_ris_matches_pm_checks_t18():
    fc = 3.5e9
    p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 0.4, 0.25, 10, 6)
    tx = np.array([-0.8, 0.0, 1.0])
    diffs = []
    for ang in np.radians(np.arange(5, 56, 10)):
        for rr in (0.9, 1.2, 1.5, 1.8, 2.1):
            rx = np.array([rr * math.sin(ang), 0.0, rr * math.cos(ang)])
            hl = rf.friis_h(np.linalg.norm(rx - tx), fc)
            g = np.exp(1j * np.radians(49.64))
            diffs.append(
                db(hl + k1_h(tx, rx, p, fc, g)) - db(hl + k1_h(tx, rx, p, fc, g, huygens_j=False))
            )
    close(
        max(abs(x) for x in diffs),
        J["T18_LOS_plus_RIS_total_power_diff_with_vs_without_j_dB_maxabs"],
        2,
    )


def test_ph03_anomalous_gradient_near_far_matches_pm_checks_t19():
    fc = 3.5e9
    lam = rf.wavelength(fc)
    k = rf.wavenumber(fc)
    th_s = math.radians(30)
    for row in J["T19_anomalous_gradient_2m_panel_0to30deg"]:
        dist = row["d1=d2_m"]
        ns = math.ceil(2.0 / (lam / 4))
        p = make_panel_extent([0, 0, 0], [0, 0, 1], [1, 0, 0], 2.0, 2.0, ns, ns)
        tx = np.array([0.0, 0.0, dist])
        rx = np.array([dist * math.sin(th_s), 0.0, dist * math.cos(th_s)])
        g = np.exp(-1j * k * p.cells[:, 0] * math.sin(th_s)) * (-1j)
        close(db(k1_h(tx, rx, p, fc, g)), row["sum_dB"], 2)
        close(20 * math.log10(lam / (4 * math.pi * 2 * dist)), row["weighted_sum_1/(d1+d2)_dB"], 2)
        prod = p.area[0] * p.n * math.sqrt(math.cos(th_s)) / (4 * math.pi * dist * dist)
        close(20 * math.log10(prod), row["product_far_field_dB"], 2)


def test_t12_published_fraunhofer_constants_match_pm_checks():
    for name, M, N, dx, dy, f in [
        ("Tang2021_RIS1", 102, 100, 0.01, 0.01, 10.5e9),
        ("Tang2021_RIS2", 34, 50, 0.01, 0.01, 10.5e9),
        ("Tang2021_small", 32, 8, 0.012, 0.012, 4.25e9),
        ("Tang2022_mmRIS1", 56, 20, 1.4e-3, 2.8e-3, 27e9),
        ("Tang2022_mmRIS2", 40, 40, 3.8e-3, 3.8e-3, 33e9),
    ]:
        close(2 * M * N * dx * dy / rf.wavelength(f), J[f"T12_{name}_2MNdxdy_over_lambda_m"], 2)


def test_vcfo_slope_constant():
    """11.76/11 (V-CFO-2) from the exact truncation formula."""
    assert rf.legacy_truncation_cfo_slope(5880, Fraction(2, 1000)) == Fraction(1176, 1100)


def test_g1_panel_cell_order_is_iv_major():
    """Cell order n = iv*n_u + iu (the convention the golden harness assumes)."""
    p = make_panel([0, 0, 0], [1, 0, 0], [0, 1, 0], 4, 3, 0.1, 0.2)
    n = np.arange(12)
    assert np.array_equal(p.iu, n % 4) and np.array_equal(p.iv, n // 4)
    # u along +y, v = n x u = +z; centred
    assert np.allclose(p.cells[0], [0, -0.15, -0.2], atol=1e-15)
    assert np.allclose(p.cells[11], [0, 0.15, 0.2], atol=1e-15)
