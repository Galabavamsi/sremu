"""WP5: LO impairments: CFO (RT-09, G4 emulator side) and Wiener phase noise.

Oracles: exact rational phase ``frac(df*k/f_s)`` (Python ``Fraction``), the intra-block
phase step ``2*pi*df/f_s``, the v2 per-tick teeth closed form, and the Wiener variance law
``Var[phi(t+L) - phi(t)] = 2*pi*beta*L/f_s`` with its sampling distribution.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.impair.oscillator import (
    CFO_BLOCK,
    WienerPhaseNoise,
    apply_cfo,
    apply_lo,
    cfo_cycles,
    cfo_phasor,
)

pytestmark = pytest.mark.runtime


def _chunks(total: int, sizes: list[int]) -> list[tuple[int, int]]:
    out, k, i = [], 0, 0
    while k < total:
        n = min(sizes[i % len(sizes)], total - k)
        out.append((k, n))
        k += n
        i += 1
    return out


def _steps(p: np.ndarray) -> np.ndarray:
    return np.angle(p[1:] * np.conj(p[:-1]))


# --------------------------------------------------------------------------- RT-09


@pytest.mark.parametrize(
    "sizes", [[11], [1, 7, 1000, 64, 3001, 2], [CFO_BLOCK]], ids=["v1tick", "mixed", "block"]
)
def test_rt09_cfo_continuity_500hz_at_5880(sizes: list[int]) -> None:
    fs = Fraction(5880)
    df = 500.0
    total = 11_760  # 2 s
    p = np.concatenate([cfo_phasor(df, fs, k, n) for k, n in _chunks(total, sizes)])
    step = 2 * math.pi * df / 5880
    assert np.max(np.abs(_steps(p) - step)) <= 1e-9
    # complex64 output keeps the same continuity within float32 rounding
    y = np.concatenate(
        [apply_cfo(np.ones(n, np.complex64), df, fs, k) for k, n in _chunks(total, sizes)]
    )
    assert np.max(np.abs(_steps(y.astype(np.complex128)) - step)) <= 1e-6


def test_rt09_teeth_v2_per_tick_phase_gives_0p94_rad_steps() -> None:
    """v2: each 2 ms tick produces int(5880*0.002) = 11 samples but advances the CFO phase by a
    full tick, so the boundary step is 2*pi*500*(0.002 - 10/5880) = 0.940 rad."""
    fs, df, tau = 5880.0, 500.0, 0.002
    m = np.arange(11)
    p = np.concatenate([np.exp(2j * np.pi * df * (j * tau + m / fs)) for j in range(200)])
    steps = _steps(p)
    intra = 2 * math.pi * df / fs
    boundary = 2 * math.pi * df * (tau - 10 / fs)
    assert boundary == pytest.approx(0.94, abs=0.005)
    assert np.max(steps) == pytest.approx(boundary, abs=1e-9)
    assert np.max(np.abs(steps - intra)) > 1e-9  # the RT-09 check fails on the legacy phase


@pytest.mark.parametrize(
    "fs", [Fraction(5880), Fraction(23_040_000), Fraction(30_720_000), Fraction(10**6, 3)]
)
def test_cfo_phase_is_exact_for_large_indices(fs: Fraction) -> None:
    g = np.random.Generator(np.random.Philox(key=909))
    for _ in range(25):
        df = float(g.uniform(-2e4, 2e4))
        k0 = int(g.integers(0, 2**40))
        got = cfo_cycles(df, fs, k0, 3)
        for i in range(3):
            exact = (Fraction(df) * (k0 + i) / fs) % 1
            d = (got[i] - float(exact)) % 1.0
            assert min(d, 1.0 - d) <= 1e-12
        p = cfo_phasor(df, fs, k0, 3)
        ref = np.exp(
            2j * np.pi * np.array([float((Fraction(df) * (k0 + i) / fs) % 1) for i in range(3)])
        )
        assert np.max(np.abs(p - ref)) <= 1e-11


@pytest.mark.golden
def test_g4_emulator_side_cfo_arbitrary_chunks() -> None:
    """G4 (Python side): CFO applied with arbitrary chunk sizes equals the closed form
    x*exp(j 2 pi df k/f_s) to <= 1e-9 x RMS, and is chunk-invariant bitwise. The MATLAB
    comm.PhaseFrequencyOffset comparison on cfo_reference_iq.csv is added by WP4."""
    g = np.random.Generator(np.random.Philox(key=44))
    n = 20_000
    x = (g.standard_normal(n) + 1j * g.standard_normal(n)) / math.sqrt(2)
    fs = Fraction(10**6)
    df = 1234.5
    whole = x * cfo_phasor(df, fs, 0, n)
    pieces = np.concatenate(
        [x[k : k + c] * cfo_phasor(df, fs, k, c) for k, c in _chunks(n, [1, 333, 4096, 17, 999])]
    )
    assert np.array_equal(whole, pieces)
    exact = np.exp(2j * np.pi * np.array([float((Fraction(df) * k / fs) % 1) for k in range(n)]))
    rms = math.sqrt(float(np.mean(np.abs(x) ** 2)))
    assert np.max(np.abs(pieces - x * exact)) <= 1e-9 * rms


def test_apply_lo_directions_and_zero_offset() -> None:
    fs = Fraction(10**6)
    x = np.full(100, 0.5 + 0.5j, np.complex64)
    tx = apply_lo(x, 10, fs, cfo_hz=1000.0, direction="tx")
    rx = apply_lo(tx, 10, fs, cfo_hz=1000.0, direction="rx")
    np.testing.assert_allclose(rx, x, rtol=0, atol=1e-6)
    assert np.array_equal(apply_lo(x, 10, fs), x)
    np.testing.assert_array_equal(cfo_phasor(0.0, fs, 5, 4), np.ones(4))


# --------------------------------------------------------------------------- Wiener PN


def test_pn_bridge_and_block_increment_variances() -> None:
    beta, fs, n = 100.0, Fraction(10**6), 1 << 23
    pn = WienerPhaseNoise((21, 22), beta, fs)
    phi = pn.phase(0, n)
    s2 = 2 * math.pi * beta / float(fs)  # per-sample variance
    m_blk = pn.block
    nblk = n // m_blk
    # block boundaries carry the block phases exactly
    assert np.array_equal(phi[::m_blk][:64], pn.block_phases(0, 63))
    for lag, tol in ((37, 0.02), (1024, 0.07), (4096, 0.13)):
        d = phi[lag::lag] - phi[:-lag:lag][: phi[lag::lag].shape[0]]
        ratio = float(np.mean(d * d)) / (s2 * lag)
        assert ratio == pytest.approx(1.0, abs=tol), lag
    # in-block position: Var[phi(jM + m) - phi(jM)] = s2*m (a pure bridge would give m(M-m)/M)
    blocks = phi[: nblk * m_blk].reshape(nblk, m_blk)
    d = blocks[:, 300] - blocks[:, 0]
    assert float(np.mean(d * d)) / (s2 * 300) == pytest.approx(1.0, abs=0.07)


def test_pn_variance_independent_of_sample_rate() -> None:
    beta = 300.0
    for fs in (Fraction(10**6), Fraction(4 * 10**6)):
        pn = WienerPhaseNoise((31, 32), beta, fs)
        lag = int(fs / 1000)  # 1 ms
        phi = pn.phase(0, lag * 4000)
        d = phi[lag::lag] - phi[:-lag:lag][: phi[lag::lag].shape[0]]
        assert float(np.mean(d * d)) / (2 * math.pi * beta * 1e-3) == pytest.approx(1.0, abs=0.1)


def test_pn_is_pure_in_the_absolute_index() -> None:
    fs = Fraction(23_040_000)
    a = WienerPhaseNoise((41, 42), 50.0, fs)
    whole = a.phase(10_000, 300_000)
    b = WienerPhaseNoise((41, 42), 50.0, fs)  # fresh cache, different query order
    far = b.phase(5_000_000, 10)
    pieces = np.concatenate(
        [b.phase(10_000 + s, c) for s, c in ((0, 1), (1, 1023), (1024, 70_001), (71_025, 228_975))]
    )
    assert np.array_equal(whole, pieces)
    assert np.array_equal(far, a.phase(5_000_000, 10))
    c = WienerPhaseNoise((41, 42), 50.0, fs, backend="numpy_ref")
    np.testing.assert_allclose(c.phase(10_000, 3000), whole[:3000], rtol=0, atol=1e-12)
    assert np.all(WienerPhaseNoise((1, 1), 0.0, fs).phase(0, 10) == 0.0)
    np.testing.assert_allclose(np.abs(a.phasor(10_000, 100)), 1.0, rtol=0, atol=1e-15)
