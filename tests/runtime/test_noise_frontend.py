"""WP5: front end (DESIGN.md sections 3.3, 6.3): full scale, gain, ADC clip, sc16.

Oracles: the closed-form power mapping (0 dBm full scale -> |d|**2 reads in mW) and exact
int16 saturation arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.impair.frontend import RxFrontEnd, TxFrontEnd, from_sc16, full_scale_w, to_sc16

pytestmark = pytest.mark.runtime


def test_tx_full_scale_and_gain_closed_form() -> None:
    d = np.array([1.0 + 0j, 0.5j, -0.25], np.complex64)
    x = TxFrontEnd().apply(d)
    np.testing.assert_allclose(np.abs(x) ** 2, np.abs(d) ** 2 * 1e-3, rtol=1e-6)  # 0 dBm full scale
    x2 = TxFrontEnd(full_scale_dbm=10.0, gain_db=3.0).apply(d)
    assert abs(x2[0]) ** 2 == pytest.approx(10 ** ((10 + 3 - 30) / 10), rel=1e-6)
    assert full_scale_w(30.0) == pytest.approx(1.0, rel=1e-15)


def test_rx_digital_power_reads_in_dbm_by_default() -> None:
    p_dbm = -63.329
    y = np.full(4, math.sqrt(10 ** ((p_dbm - 30) / 10)), np.complex128)
    d = RxFrontEnd().apply(y)
    assert 10 * math.log10(float(np.abs(d[0]) ** 2)) == pytest.approx(p_dbm, abs=1e-5)
    d2 = RxFrontEnd(full_scale_dbm=-20.0, gain_db=6.0).apply(y)
    assert 10 * math.log10(float(np.abs(d2[0]) ** 2)) == pytest.approx(p_dbm + 20 + 6, abs=1e-5)
    # TX then RX with matching full scale is the identity on digital samples
    x = TxFrontEnd(full_scale_dbm=5.0).apply(np.array([0.3 - 0.4j], np.complex64))
    np.testing.assert_allclose(RxFrontEnd(full_scale_dbm=5.0).apply(x), [0.3 - 0.4j], rtol=1e-6)


def test_adc_clip_per_component() -> None:
    y = np.array([2.0 - 3.0j, 0.5 + 0.25j, -1.5 + 0.0j]) * math.sqrt(1e-3)
    np.testing.assert_array_equal(
        RxFrontEnd(adc_clip=True).apply(y), np.array([1 - 1j, 0.5 + 0.25j, -1 + 0j], np.complex64)
    )
    assert RxFrontEnd().apply(y)[0] == np.complex64(2 - 3j)


def test_sc16_rounding_and_saturation() -> None:
    d = np.array([1.0 + 0j, -1.0 - 1.0j, 2.0 - 2.0j, 0.5 / 32767 + 1.5j / 32767, np.nan + 0j])
    iq = to_sc16(d)
    assert iq.dtype == np.int16
    assert iq.tolist() == [
        32767,
        0,
        -32767,
        -32767,
        32767,
        -32768,
        0,
        2,
        0,
        0,
    ]  # rint: half to even
    back = from_sc16(iq)
    np.testing.assert_allclose(back[:2], [1.0, -1 - 1j], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(from_sc16(iq.reshape(-1, 2)), back)
    with pytest.raises(ValueError, match="even"):
        from_sc16(np.zeros(3, np.int16))
