"""Spec oracle values (DESIGN.md sections 10.2, 10.3, 10.5, 10.6) as data, with tolerances.

Only numbers that the spec labels as oracle values (PH / RT / ST / G tables, or numbers a test
row cites) appear here (amendment A6). Every entry carries its spec ID; other work packages
may import these instead of re-typing the numbers. ``tests/oracles/test_oracle_tables.py``
recomputes each one from the closed forms in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "ORACLE",
    "PH17_T9_BLOCK_S",
    "PH17_T9_FC_HZ",
    "PH17_T9_FS",
    "PH17_T9_RX0_M",
    "PH17_T9_SPEED_MPS",
    "PH17_T9_WINDOW_S",
    "V_CFO_FS",
    "V_CFO_TAU",
    "OracleValue",
]


@dataclass(frozen=True)
class OracleValue:
    """One spec oracle number: ``value`` in ``unit``, accepted within ``tol`` (same unit)."""

    spec_id: str
    value: float
    tol: float
    unit: str
    what: str


def _v(spec_id: str, value: float, tol: float, unit: str, what: str) -> OracleValue:
    return OracleValue(spec_id, value, tol, unit, what)


ORACLE: dict[str, OracleValue] = {
    # PH-01 Friis (spec accept rel <= 1e-12 on h; the printed values carry 3 decimals)
    "PH01_friis_3p5GHz_10m_db": _v("PH-01", -63.329, 5e-4, "dB", "|h_LOS| 3.5 GHz, 10 m"),
    "PH01_friis_100MHz_52km_db": _v("PH-01", -106.768, 5e-4, "dB", "|h_LOS| 100 MHz, 52 km"),
    # PH-03 far-field focus N^2 law (Tang Eq. 8), lambda/2 cells, 2 km, 0 -> 20 deg, 3.5 GHz
    "PH03_N100_db": _v("PH-03", -169.027, 0.01, "dB", "N = 100"),
    "PH03_N400_db": _v("PH-03", -156.985, 0.01, "dB", "N = 400"),
    "PH03_N1600_db": _v("PH-03", -144.944, 0.01, "dB", "N = 1600"),
    "PH03_slope_db_per_4x": _v("PH-03", 12.04, 0.01, "dB", "slope per 4x cells"),
    # PH-04 fixed aperture 1 m^2, 3.5 GHz, Gamma = 1
    "PH04_far_specular_200m_db": _v("PH-04", -115.280, 0.01, "dB", "far specular 200 m, any N"),
    # D5: Richardson-converged po_patch; the oracle value is po_patch at Delta = lambda/40
    # (single-level values, e.g. Delta = lambda/5, are not asserted)
    "PH04_near_anomalous_5m_db": _v(
        "PH-04", -88.128, 0.10, "dB", "near anomalous 5 m, Richardson-converged po_patch (D5)"
    ),
    "PH04_legacy_v2_db_per_4x": _v("PH-04", 12.04, 0.01, "dB", "teeth: legacy_v2 per 4x"),
    # PH-05 image theory
    "PH05_oblique_plate_db": _v("PH-05", 0.0, 0.3, "dB", "40 deg plate 12 m vs image"),
    "PH05_oblique_plate_deg": _v("PH-05", 0.0, 3.0, "deg", "phase vs image"),
    "PH05_disk_discrete_vs_closed_db": _v("PH-05", 0.0, 0.01, "dB", "discrete disk vs closed form"),
    "PH05_disk_200m_vs_image_db": _v("PH-05", 0.0, 0.05, "dB", "disk R = 200 m vs image"),
    "PH05_disk_200m_vs_image_deg": _v("PH-05", 0.0, 2.0, "deg", "disk R = 200 m phase"),
    "PH05_no_j_phase_deg": _v("PH-05", -90.0, 3.0, "deg", "teeth: no_j phase vs image"),
    # PH-06 plate RCS
    "PH06_plate_400m_db": _v("PH-06", -126.067, 0.01, "dB", "1 m^2, 400 m monostatic"),
    # PH-07 focusing saturation
    "PH07_focus_d1_L1_db": _v("PH-07", -23.864, 0.01, "dB", "d = 1 m, L = 1 m"),
    "PH07_focus_limit_db": _v("PH-07", -6.021, 0.01, "dB", "L -> infinity"),
    "PH07_passive_bound_abs": _v("PH-07", 0.5, 0.0, "1", "|h| <= 0.5 (monostatic, any config)"),
    # PH-10 ground
    "PH10_two_ray_52km_h1p5_db": _v("PH-10", 139.10, 0.01, "dB", "path loss 200/1.5 m, 100 MHz"),
    "PH10_grazing_gamma_plus1_abs": _v("PH-10", 0.0, 1e-3, "1", "|Gamma_H,V + 1| at psi = 1e-4"),
    "PH10_brewster_deg": _v("PH-10", 26.565, 5e-4, "deg", "lossless eps_r = 4, |Gamma_V| = 0"),
    # PH-11 knife edge
    "PH11_F0_abs": _v("PH-11", 0.5, 1e-12, "1", "|F(0)|"),
    "PH11_J_max_dev_db": _v("PH-11", 0.123, 0.15 - 0.123, "dB", "max |J - exact| on [-0.78, 10]"),
    "PH11_continuity_db": _v("PH-11", 0.0, 1e-6, "dB", "loss continuity at nu = +-1e-9"),
    "PH11_nu10_db": _v("PH-11", 32.95, 0.01, "dB", "loss at nu = 10"),
    # PH-12 P.526 (EXPERIMENTAL, regression against scratch)
    "PH12_52km_h1p5_excess_db": _v("PH-12", 37.14, 0.01, "dB", "excess loss 200/1.5 m"),
    "PH12_52km_h10_excess_db": _v("PH-12", 20.28, 0.01, "dB", "excess loss 200/10 m"),
    # PH-13 quantization (quadrature +-0.001 dB, MC +-0.15 dB)
    "PH13_eta_b1_db": _v("PH-13", -3.922, 0.001, "dB", "eta(1)"),
    "PH13_eta_b2_db": _v("PH-13", -0.912, 0.001, "dB", "eta(2)"),
    "PH13_eta_b3_db": _v("PH-13", -0.224, 0.001, "dB", "eta(3)"),
    "PH13_lab_pair_db": _v("PH-13", -4.105, 0.001, "dB", "lab pair, Delta = 203.41 deg"),
    "PH13_mc_tol_db": _v("PH-13", 0.0, 0.15, "dB", "MC N = 4096, 100 trials vs quadrature"),
    # PH-14 Abeywickrama
    "PH14_bmin0p2_a1p6_db": _v("PH-14", -5.508, 0.01, "dB", "(beta_min, alpha) = (0.2, 1.6)"),
    "PH14_bmin0p8_a1p6_db": _v("PH-14", -1.085, 0.01, "dB", "(beta_min, alpha) = (0.8, 1.6)"),
    "PH14_phi_independence_db": _v("PH-14", 0.0, 1e-6, "dB", "independent of phi"),
    # PH-15 efficiency and failures
    "PH15_eta0p5_db": _v("PH-15", -3.0103, 5e-5, "dB", "eta = 0.5 on the RIS term"),
    "PH15_dead10_db": _v("PH-15", -0.915, 0.10, "dB", "10 % dead, focused, N = 1024, 200 maps"),
    # PH-16 phase wrap
    "PH16_wrap_rel": _v("PH-16", 0.0, 1e-15, "1", "|dh| <= 1e-15 |h| for Gamma*exp(j 2 pi m)"),
    # PH-17 Doppler (T9 geometry)
    "PH17_los_fd_hz": _v("PH-17", 350.242, 0.001, "Hz", "LOS 30 m/s, 3.5 GHz"),
    "PH17_ris_fd_hz": _v("PH-17", 325.192, 0.01, "Hz", "RIS path f_D(t0)"),
    # D7: mean over the T9 window (0.2 s at 10 kS/s, 2 ms blocks); see PH17_T9 below
    "PH17_block_constant_hz": _v(
        "PH-17", -148.3, 1.0, "Hz", "teeth: block_constant mean f over the T9 window (D7)"
    ),
    "PH17_phase_rad": _v("PH-17", 0.0, 1e-6, "rad", "phase vs -2 pi f_c d(t)/c over 1 s"),
    # PH-18 interpolation budget
    "PH18_eps_rel": _v("PH-18", 1e-3, 0.0, "1", "|dh|/|h| <= 1e-3"),
    # PH-20 FD filters (expected values cited in the PH-20 row; acceptance <= -60 dB)
    "PH20_fd_accept_db": _v("PH-20", -60.0, 0.0, "dB", "every FD row in band at worst mu"),
    "PH20_kaiser8_b0p1_db": _v("PH-20", -71.4, 0.1, "dB", "8 taps beta 8, B <= 0.2 f_s"),
    "PH20_kaiser16_b0p3_db": _v("PH-20", -75.5, 0.1, "dB", "16 taps beta 8, B <= 0.6 f_s"),
    "PH20_kaiser32_b0p4_db": _v("PH-20", -76.5, 0.1, "dB", "32 taps beta 8, B <= 0.8 f_s"),
    "PH20_lagrange4_b0p4_db": _v("PH-20", -5.2, 0.1, "dB", "teeth: 4-tap Lagrange at +-0.4 f_s"),
    "PH20_narrowband_switch_rad": _v("PH-20", 0.05, 0.0, "rad", "pi B dtau threshold"),
    "PH20_narrowband_edge_db": _v("PH-20", -32.0, 0.0, "dB", "band-edge error bound"),
    "PH20_delay_samples": _v("PH-20", 37.4, 0.01, "samples", "known delay recovered"),
    # PH-21 noise
    "PH21_noise_1MSps_dbm": _v(
        "PH-21", -113.975, 0.05, "dBm", "variance over 1e6 samples, F = 0 dB"
    ),
    "PH21_cross_rx_corr": _v("PH-21", 0.0, 0.016, "1", "cross-RX |rho| over 1e5 samples (5 sigma)"),
    # PH-25 m/d decomposition (lab pair, |Gamma| = 1)
    "PH25_d_db": _v("PH-25", -0.18, 0.01, "dB", "|d| lab pair"),
    "PH25_m_db": _v("PH-25", -13.86, 0.01, "dB", "|m| lab pair"),
    "PH25_m_deg": _v("PH-25", 127.9, 0.05, "deg", "arg m lab pair"),
    # PH-26 point vs patch (16 x 16, lambda/2, 3.5 GHz, 2 m)
    "PH26_specular_db": _v("PH-26", 0.0, 0.05, "dB", "point vs patch at specular"),
    # D6: 45 deg retro, Delta <= lambda/5 (3 x 3 sub-cells): point minus patch. 8.11 dB is the
    # 3 x 3 sub-aperture factor of a lambda/2 cell in its far field (exactly reproduced by a
    # single cell or by the panel in the far field; NOT by the 16 x 16 panel at 2 m, whose
    # cells see 35..55 deg: 7.68 dB focused; see test_ph26_* and the WP4 report).
    "PH26_retro45_db": _v("PH-26", 8.11, 0.1, "dB", "point - po_patch (3x3) at 45 deg retro (D6)"),
    "PH26_retro45_sinc_limit_db": _v(
        "PH-26", 8.92, 0.005, "dB", "documented Delta -> 0 sinc limit at 45 deg retro (D6)"
    ),
    # section 5.14 / D4: legacy_v1 teeth in the v1 default geometry (v1 uses c = 3e8)
    "LEGACY_v1_vs_k1_db": _v("D4", 24.39, 0.05, "dB", "legacy_v1 - K1, v1 default geometry"),
    # RT oracles
    "RT05_samples_2km_23p04": _v("RT-05", 154.0, 0.5, "samples", "RX at 2 km vs near RX (approx)"),
    "RT09_v2_step_rad": _v(
        "RT-09", 0.94, 0.005, "rad", "teeth: v2 per-tick CFO step, 500 Hz @ 5880"
    ),
    "RT11_delay_samples": _v("RT-11", 37.4, 0.01, "samples", "known delay through the engine"),
    # ST oracles
    "ST01_ks_position": _v("ST-01", 0.0, 0.02, "1", "KS D vs 3x^2/l^2 - 2x^3/l^3"),
    "ST01_ks_start": _v("ST-01", 0.0, 0.0163, "1", "KS D start points uniform"),
    "ST01_mean_leg_rel": _v("ST-01", 0.0, 0.025, "1", "mean |s-d| = l/3 +- 2.5 %"),
    "ST02_acf_dev": _v("ST-02", 0.0, 0.05, "1", "max |ACF - J0| for lags <= 40 ms"),
    "ST02_lcr_m10db_hz": _v("ST-02", 35.9, 0.10 * 35.9, "Hz", "LCR at -10 dB within 10 %"),
    "ST02_ks_envelope": _v("ST-02", 0.0, 0.01, "1", "envelope KS"),
    "ST05_rmse_over_crb": _v("ST-05", 1.5, 0.0, "1", "RMSE <= 1.5 x CRB at 10 and 20 dB"),
    # V-CFO numbers cited by test rows (V-CFO-2 slope; V-CFO-4 CRB std at N = 2000, 5880 S/s)
    "VCFO2_slope": _v("V-CFO-2", 1.0691, 0.0005, "1", "LegacyTruncationAdapter slope"),
    "VCFO4_crb_0db_hz": _v("V-CFO-4", 0.0256, 5e-5, "Hz", "CRB std, 0 dB"),
    "VCFO4_crb_10db_hz": _v("V-CFO-4", 0.0081, 5e-5, "Hz", "CRB std, 10 dB"),
    "VCFO4_crb_20db_hz": _v("V-CFO-4", 0.0026, 5e-5, "Hz", "CRB std, 20 dB"),
    # V-WCL W1 teeth (section 5.14 legacy_wcl_eq2 row, cited by W1): per-element power ratio of
    # the letter's Eq. (2)/(3) kernel to K1, (pi lambda)^2 (cos_i cos_s)^(2q-1), at broadside
    "W1_wcl_ratio_1GHz_db": _v("W1", -0.52, 0.005, "dB", "legacy_wcl_eq2 / K1, 1 GHz"),
    "W1_wcl_ratio_2p4GHz_db": _v("W1", -8.1, 0.05, "dB", "legacy_wcl_eq2 / K1, 2.4 GHz"),
    "W1_wcl_ratio_3p5GHz_db": _v("W1", -11.4, 0.05, "dB", "legacy_wcl_eq2 / K1, 3.5 GHz"),
    "W1_ratio_rel": _v("W1", 0.0, 1e-9, "1", "legacy_wcl_eq2 ratio vs the closed form"),
    # G goldens (section 10.6)
    "G1_rel": _v("G1", 0.0, 1e-9, "1", "max rel |dh| vs MATLAB per-element K1"),
    "G2_mainlobe_deg": _v("G2", 20.0, 0.5, "deg", "main lobe, -30 -> +20 deg"),
    "G2_levels_db": _v("G2", 0.0, 0.05, "dB", "levels above -30 dB after the cell-gain constant"),
    "G3_mirror_deg": _v("G3", 41.15, 0.25, "deg", "1-bit mirror lobe"),
    "G3_mirror_db": _v("G3", -0.97, 0.10, "dB", "1-bit mirror lobe level vs main lobe"),
    "G4_rel_rms": _v("G4", 0.0, 1e-9, "1", "max |dy| / RMS(y), comm.PhaseFrequencyOffset"),
    "G5_db": _v("G5", 0.0, 0.01, "dB", "comm.ThermalNoise power, 1e7 samples"),
    "G7_vs_etsi_db": _v("G7", 0.0, 1.0, "dB", "K1 at the focus vs ETSI far-field closed form"),
    "G7_po_bound_db": _v("G7", 0.0, 0.01, "dB", "never above the PO plate bound"),
    "G8_db": _v("G8", 0.0, 0.05, "dB", "twoRayChannel at 52 km"),
}
"""Spec oracle values keyed by a stable name (spec ID prefix)."""

V_CFO_FS = Fraction(5880)
"""V-CFO sample rate, S/s (section 10.10)."""
V_CFO_TAU = Fraction(2, 1000)
"""v1 tick duration, s (11.76/11 at 5880 S/s)."""

# PH-17 T9 geometry and window (``pm_checks.py`` T9; D7): Tx at the origin, Rx starting at
# (50, 0, 0) m and moving at (-30, 0, 0) m/s, 3.5 GHz; the RIS path uses a 0.5 m x 0.5 m,
# 12 x 12 panel centred at (25, 10, 0) m facing -y.
PH17_T9_FC_HZ = 3.5e9
"""T9 carrier, Hz."""
PH17_T9_SPEED_MPS = 30.0
"""T9 approach speed, m/s."""
PH17_T9_RX0_M = (50.0, 0.0, 0.0)
"""T9 Rx start position, m (Tx at the origin)."""
PH17_T9_FS = Fraction(10_000)
"""T9 sample rate, S/s."""
PH17_T9_WINDOW_S = Fraction(1, 5)
"""T9 window, s (2000 samples): the PH-17 teeth value is a mean over this window (D7)."""
PH17_T9_BLOCK_S = Fraction(1, 500)
"""v2 ``block_constant`` block length in T9, s (2 ms)."""
