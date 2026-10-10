"""units.py: constants and dB helpers against closed forms (sections 3.1, 5.12)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu import units

pytestmark = pytest.mark.unit


def test_constants_exact_si_values():
    # the only numeric c literal lives in units.py (grep test); compare against the SI text
    assert units.C0 == float("299792458")
    assert units.K_B == 1.380649e-23
    assert units.T0 == 290.0


def test_thermal_floor_per_hz_and_at_1msps():
    # section 3.1 [DERIVED]: -173.975 dBm/Hz; -113.975 dBm at 1 MS/s with F = 0 dB
    assert units.DBM_PER_HZ_THERMAL == pytest.approx(-173.975, abs=5e-4)
    assert units.thermal_noise_dbm(1e6) == pytest.approx(-113.975, abs=5e-4)
    assert units.thermal_noise_dbm(1e6, 7.0) == pytest.approx(-113.975 + 7.0, abs=5e-4)
    assert units.noise_power_w(1e6) == pytest.approx(units.K_B * 290.0 * 1e6, rel=1e-15)


def test_db_helpers_round_trip_scalar_and_array():
    x = np.array([1e-12, 1e-3, 0.5, 1.0, 7.0, 1e6])
    np.testing.assert_allclose(units.db_to_power_ratio(units.power_ratio_to_db(x)), x, rtol=1e-14)
    np.testing.assert_allclose(units.db_to_amp_ratio(units.amp_ratio_to_db(x)), x, rtol=1e-14)
    np.testing.assert_allclose(units.dbm_to_w(units.w_to_dbm(x)), x, rtol=1e-14)
    assert units.power_ratio_to_db(0.5) == pytest.approx(10 * math.log10(0.5), rel=1e-15)
    assert units.amp_ratio_to_db(0.5) == pytest.approx(20 * math.log10(0.5), rel=1e-15)
    assert units.w_to_dbm(1.0) == pytest.approx(30.0, abs=1e-12)
    assert units.dbm_to_w(0.0) == pytest.approx(1e-3, rel=1e-15)
    assert isinstance(units.power_ratio_to_db(2.0), float)
    assert units.power_ratio_to_db(0.0) == -math.inf


def test_friis_oracle_values_from_units_helpers():
    # PH-01 oracle values: 3.5 GHz / 10 m -> -63.329 dB; 100 MHz / 52 km -> -106.768 dB
    for fc, d, expected in ((3.5e9, 10.0, -63.329), (100e6, 52e3, -106.768)):
        h = units.wavelength_m(fc) / (4 * math.pi * d)
        assert units.amp_ratio_to_db(h) == pytest.approx(expected, abs=5e-4)
    assert units.wavenumber_rad_per_m(1e9) == pytest.approx(2 * math.pi * 1e9 / units.C0, rel=1e-15)
    with pytest.raises(ValueError):
        units.wavelength_m(0.0)
