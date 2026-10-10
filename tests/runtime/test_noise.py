"""WP5: thermal noise (PH-21, G5 emulator side) on the shared Philox stream (I5, I2a).

Oracles: Random123 Philox4x32-10 known-answer vectors (the generator noise depends on), the
closed-form floor ``k_B*T0*F*f_s`` (-113.975 dBm at 1 MS/s, F = 0 dB), and the sampling
distribution of the estimators (tolerances as stated in DESIGN.md section 10).
"""

from __future__ import annotations

import math

import numba
import numpy as np
import pytest

from sremu import rng
from sremu.impair.noise import (
    NOISE_LANE,
    noise_numba,
    noise_numpy,
    normals_numba,
    normals_numpy,
    thermal_sigma,
)
from sremu.units import K_B

pytestmark = pytest.mark.runtime

SEED = 20261010

# Random123 kat_vectors, philox4x32 R = 10: (counter, key, expected output)
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


def _key(rx_stream: str) -> tuple[int, int]:
    """The noise key of an RX stream (purpose ``noise``, section 6.2)."""
    return tuple(rng.derive_key(SEED, "noise", rx_stream))  # type: ignore[return-value]


@pytest.mark.parametrize(("ctr", "key", "expected"), KAT)
def test_i5_philox_known_answer_vectors(ctr, key, expected) -> None:
    assert tuple(int(v) for v in rng.philox4x32(np.array(ctr, np.uint64), key)) == expected


def test_i5_noise_uses_lane0_box_muller_of_the_philox_words() -> None:
    """Noise sample k is Box-Muller of the words of counter (k_lo, k_hi, 0, 0): recompute it
    from the raw words (across the 2**32 index boundary) for both backends."""
    key = _key("mu.rx0")
    k0 = 2**32 - 3
    words = rng.philox_block(key, k0, 6, sub=NOISE_LANE, backend="numba_cpu")
    assert np.array_equal(words, rng.philox_block(key, k0, 6, sub=NOISE_LANE))
    u1 = (words[:, 0].astype(np.float64) + 1.0) * 2.0**-32
    u2 = words[:, 1].astype(np.float64) * 2.0**-32
    expect = 0.3 * np.sqrt(-np.log(u1)) * np.exp(2j * np.pi * u2)
    for impl in (noise_numba, noise_numpy):
        np.testing.assert_allclose(impl(key, k0, 6, 0.3), expect, rtol=1e-6, atol=0)


def test_i5_noise_numba_vs_numpy() -> None:
    a = noise_numba((3, 4), 12_345, 200_000, 0.7)
    b = noise_numpy((3, 4), 12_345, 200_000, 0.7)
    assert a.dtype == b.dtype == np.complex64
    d = a.astype(np.complex128) - b.astype(np.complex128)
    assert np.sqrt(np.mean(np.abs(d) ** 2)) <= 1e-6 * np.sqrt(
        np.mean(np.abs(b.astype(np.complex128)) ** 2)
    )
    na = normals_numba((3, 4), 2, 10, 50_000)
    nb = normals_numpy((3, 4), 2, 10, 50_000)
    assert np.max(np.abs(na - nb)) <= 1e-12 * np.max(np.abs(nb))


@pytest.mark.parametrize("impl", [noise_numba, noise_numpy], ids=["numba", "numpy"])
def test_i2a_noise_chunk_invariance_bitwise(impl) -> None:
    whole = impl((9, 9), 5000, 1000, 1.0)
    parts = np.concatenate([impl((9, 9), 5000, 370, 1.0), impl((9, 9), 5370, 630, 1.0)])
    assert np.array_equal(whole, parts)
    odd = np.concatenate(
        [impl((9, 9), 5000 + a, b - a, 1.0) for a, b in ((0, 1), (1, 64), (64, 999), (999, 1000))]
    )
    assert np.array_equal(whole, odd)


def test_i2a_noise_thread_count_bitwise() -> None:
    prev = numba.get_num_threads()
    try:
        outs = []
        for nt in sorted(
            {1, min(3, numba.config.NUMBA_NUM_THREADS), numba.config.NUMBA_NUM_THREADS}
        ):
            numba.set_num_threads(nt)
            outs.append(noise_numba((5, 6), 777, 100_001, 1.0))
    finally:
        numba.set_num_threads(prev)
    assert all(np.array_equal(outs[0], o) for o in outs[1:])


def test_ph21_noise_power_minus113p975dbm_at_1msps() -> None:
    fs = 1e6
    sigma = thermal_sigma(fs, 0.0)
    floor_dbm = 10 * math.log10(K_B * 290.0 * fs) + 30
    assert floor_dbm == pytest.approx(-113.975, abs=0.001)  # PH-21 oracle value
    assert sigma**2 == pytest.approx(K_B * 290.0 * fs, rel=1e-15)
    w = noise_numba(_key("mu.rx0"), 0, 10**6, sigma).astype(np.complex128)
    p_dbm = 10 * math.log10(float(np.mean(np.abs(w) ** 2))) + 30
    assert p_dbm == pytest.approx(-113.975, abs=0.05)
    # each quadrature carries half the power and they are uncorrelated
    assert np.var(w.real) / np.var(w.imag) == pytest.approx(1.0, abs=0.01)
    assert abs(np.mean(w.real * w.imag)) / sigma**2 <= 0.005
    # noise figure adds exactly F dB
    assert 10 * math.log10(thermal_sigma(fs, 7.0) ** 2) + 30 == pytest.approx(-106.975, abs=0.001)


def test_ph21_cross_receiver_correlation_below_5_sigma() -> None:
    n = 10**5
    w1 = noise_numba(_key("rx1.rx0"), 0, n, 1.0).astype(np.complex128)
    w2 = noise_numba(_key("rx2.rx0"), 0, n, 1.0).astype(np.complex128)
    rho = abs(np.vdot(w2, w1)) / math.sqrt(np.vdot(w1, w1).real * np.vdot(w2, w2).real)
    assert rho <= 0.016
    # same RX, disjoint indices: independent as well
    w3 = noise_numba(_key("rx1.rx0"), n, n, 1.0).astype(np.complex128)
    assert abs(np.vdot(w3, w1)) / n <= 0.016


def test_ph21_box_muller_is_gaussian() -> None:
    w = noise_numpy((11, 12), 0, 400_000, math.sqrt(2.0)).astype(np.complex128)
    x = np.concatenate([w.real, w.imag])  # unit-variance components
    assert abs(x.mean()) <= 5 / math.sqrt(x.size)
    assert x.var() == pytest.approx(1.0, abs=5 * math.sqrt(2 / x.size))
    kurt = np.mean(x**4) / x.var() ** 2
    assert kurt == pytest.approx(3.0, abs=5 * math.sqrt(24 / x.size))


@pytest.mark.golden
def test_g5_emulator_side_thermal_noise_1e7_samples() -> None:
    """G5 (Python side): 290 K, 1 MS/s, 1e7 samples within +-0.01 dB of k_B*T*f_s.
    The MATLAB comm.ThermalNoise comparison is added by WP4's golden harness."""
    fs = 1e6
    w = noise_numba(_key("g5.rx0"), 0, 10**7, math.sqrt(K_B * 290.0 * fs)).astype(np.complex128)
    p_db = 10 * math.log10(float(np.mean(np.abs(w) ** 2)) / (K_B * 290.0 * fs))
    assert abs(p_db) <= 0.01


def test_noise_argument_validation() -> None:
    with pytest.raises(ValueError):
        noise_numba((2**32, 0), 0, 4, 1.0)
    with pytest.raises(ValueError):
        noise_numpy((1, 2), -1, 4, 1.0)
    with pytest.raises(ValueError):
        noise_numba((1, 2), 0, 4, -1.0)
    assert noise_numba((1, 2), 0, 0, 1.0).shape == (0,)
