"""WP5: fractional delay, narrowband switch, known delay and the TX ring.

PH-20 (FD tables and wideband), RT-11 (known 37.4-sample delay, renderer level; the engine
level belongs to WP6) and the section 4.5 ring rules. Oracles are closed forms: the ideal
delay response exp(-j 2 pi f D), the narrowband half-spread phase error, and exact integer
ring arithmetic.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.channel.fd import (
    FD_ROWS,
    choose_fd,
    fd_center,
    inband_error_db,
    integer_delay_allowed,
    interp_taps,
    kaiser_fd_spec,
    kaiser_sinc_taps,
    polyphase_table,
    worst_case_error_db,
)
from sremu.channel.interp import is_narrowband, plan_groups
from sremu.channel.render_numpy import get_backend
from sremu.channel.ring import TxRingBuffer, ring_capacity
from sremu.contracts import FDSpec, LinkJob, SegmentJob
from sremu.errors import ConfigurationError, FDBandwidthExceeded

pytestmark = pytest.mark.runtime

# PH-20 "expected" in-band errors (direct design, worst mu), spec section 4.5 / 10.2
PH20_EXPECTED_DB = {8: (0.1, -71.4), 16: (0.3, -75.5), 32: (0.4, -76.5)}


def _lagrange4(mu: float) -> np.ndarray:
    """Cubic Lagrange taps at n = 0..3 for delay D = 1 + mu (the rejected input-doc default)."""
    d = 1.0 + mu
    h = []
    for k in range(4):
        p = 1.0
        for m in range(4):
            if m != k:
                p *= (d - m) / (k - m)
        h.append(p)
    return np.array(h)


# --------------------------------------------------------------------------- PH-20 tables


@pytest.mark.parametrize("taps", [8, 16, 32])
def test_ph20_fd_table_row_inband_error_below_minus60db(taps: int) -> None:
    band, expected = PH20_EXPECTED_DB[taps]
    spec = kaiser_fd_spec(taps)
    # the polyphase table with linear interpolation between phases is what both backends apply
    err_table = worst_case_error_db(spec, band, n_mu=129)
    assert err_table <= -60.0
    # direct design reproduces the PH-20 expected values (fd_check.py grid of 101 mu values)
    err_direct = inband_error_db(
        lambda mu: kaiser_sinc_taps(mu, taps),
        lambda mu: fd_center(taps) + mu,
        band,
        np.linspace(0.0, 0.999, 101),
    )
    assert abs(err_direct - expected) <= 0.1
    # interpolation between the 1024 phases costs at most a fraction of a dB
    assert err_table <= err_direct + 0.2


def test_ph20_polyphase_rows_are_the_direct_design_and_interpolation_is_linear() -> None:
    table = polyphase_table(16)
    assert table.shape == (1025, 16)
    assert not table.flags.writeable
    for j in (0, 1, 511, 1023, 1024):
        np.testing.assert_allclose(table[j], kaiser_sinc_taps(j / 1024, 16), rtol=0, atol=1e-15)
    mu = (300 + 0.25) / 1024
    np.testing.assert_allclose(
        interp_taps(table, mu), 0.75 * table[300] + 0.25 * table[301], rtol=0, atol=1e-15
    )
    with pytest.raises(ValueError, match="mu"):
        interp_taps(table, 1.0)


def test_ph20_teeth_lagrange4_fails_the_inband_metric() -> None:
    """The 4-tap cubic Lagrange default of both input docs: -25.8/-13.2/-5.2 dB at 0.2/0.3/0.4."""
    mus = np.linspace(0.0, 0.999, 101)
    errs = {b: inband_error_db(_lagrange4, lambda mu: 1.0 + mu, b, mus) for b in (0.2, 0.3, 0.4)}
    assert errs[0.4] == pytest.approx(-5.2, abs=0.1)
    assert errs[0.3] == pytest.approx(-13.2, abs=0.1)
    assert errs[0.2] == pytest.approx(-25.8, abs=0.1)
    assert all(e > -60.0 for e in errs.values())


def test_ph20_48_taps_cannot_reach_the_target_beyond_0p8() -> None:
    err = inband_error_db(
        lambda mu: kaiser_sinc_taps(mu, 48),
        lambda mu: fd_center(48) + mu,
        0.45,
        np.linspace(0.0, 0.999, 101),
    )
    assert err == pytest.approx(-57.1, abs=0.2)
    assert err > -60.0


@pytest.mark.parametrize(
    ("frac", "expected"),
    [
        (0.1 / math.pi * (1 - 1e-12), "integer"),
        (0.1 / math.pi * 1.001, "kaiser8b8"),
        (0.2, "kaiser8b8"),
        (0.2000001, "kaiser16b8"),
        (0.6, "kaiser16b8"),
        (0.61, "kaiser32b8"),
        (0.8, "kaiser32b8"),
    ],
)
def test_ph20_choose_fd_follows_the_table(frac: float, expected: str) -> None:
    fs = Fraction(23_040_000)
    spec = choose_fd(frac * float(fs), fs)
    assert spec.filter_id == expected
    if expected != "integer":
        assert spec.lookahead == spec.taps // 2
        assert spec.n_phases == 1024 and spec.beta == 8.0


def test_ph20_choose_fd_rejects_above_0p8_and_respects_a_stricter_target() -> None:
    fs = Fraction(1_000_000)
    with pytest.raises(FDBandwidthExceeded):
        choose_fd(0.81e6, fs)
    # 8 taps reach -71.4 dB at +-0.1 f_s: a -70 dB target keeps 8 taps; -80 dB needs 16
    assert choose_fd(0.2e6, fs, target_db=-70.0).taps == 8
    assert choose_fd(0.2e6, fs, target_db=-80.0).taps == 16
    with pytest.raises(FDBandwidthExceeded):
        choose_fd(0.8e6, fs, target_db=-90.0)
    assert FD_ROWS == ((0.2, 8), (0.6, 16), (0.8, 32))
    assert integer_delay_allowed(0.0318 * 1e6, fs)
    assert not integer_delay_allowed(0.0319 * 1e6, fs)


# --------------------------------------------------------------------------- narrowband


def test_ph20_narrowband_switch_at_pi_b_dtau_0p05() -> None:
    b = 0.8e6
    dtau = 0.05 / (math.pi * b)
    assert is_narrowband(b, dtau)
    assert not is_narrowband(b, dtau * (1 + 1e-9))
    g, nb = plan_groups(np.array([1e-6, 1e-6 + dtau]), np.array([1e-6, 1e-6 + dtau]), b)
    assert nb and np.array_equal(g, [0, 0])
    g, nb = plan_groups(
        np.array([1e-6, 1e-6 + 1.5 * dtau, 1e-6 + 0.5 * dtau]),
        np.array([1e-6, 1e-6 + 1.5 * dtau, 1e-6 + 0.5 * dtau]),
        b,
    )
    assert not nb
    # clusters by increasing delay, each with spread <= 0.05/(pi B)
    assert g.tolist() == [0, 1, 0]


def _tone_ring(f_norm: float, n: int, fs: Fraction) -> TxRingBuffer:
    ring = TxRingBuffer("a.tx0", fs, 1 << 15)
    k = np.arange(n)
    ring.write(0, np.exp(2j * np.pi * f_norm * k).astype(np.complex64))
    return ring


def _job(
    ring: TxRingBuffer,
    fs: Fraction,
    k0: int,
    k1: int,
    amp: np.ndarray,
    tau: np.ndarray,
    group: np.ndarray,
    fd: FDSpec,
    fc: float = 2.0e9,
) -> SegmentJob:
    lj = LinkJob("a.tx0->b.rx0", ring, fc, 0.0, amp, amp.copy(), tau, tau.copy(), group, fd)
    return SegmentJob("b.rx0", fs, k0, k1, Fraction(0), Fraction(0), [lj])


def test_ph20_narrowband_band_edge_error_is_minus32db(backend_name: str) -> None:
    """Two paths with pi*B*dtau = 0.05 share one envelope at the midpoint delay: a tone at the
    band edge f = B/2 sees the half-spread error |1 - exp(j*pi*B*dtau/2)| = -32.04 dB."""
    fs = Fraction(1_000_000)
    b = 0.8e6
    dtau = 0.05 / (math.pi * b)
    ring = _tone_ring(b / 2 / float(fs), 6000, fs)
    tau = np.array([3.2e-6, 3.2e-6 + dtau])
    amp = np.array([1.0 + 0j, 0.0 + 0j])
    fd = choose_fd(b, fs)
    be = get_backend(backend_name)  # type: ignore[arg-type]
    shared = np.zeros(2000, np.complex128)
    be.render(_job(ring, fs, 2000, 4000, amp, tau, np.array([0, 0], np.int16), fd), shared)
    exact = np.zeros(2000, np.complex128)
    be.render(_job(ring, fs, 2000, 4000, amp, tau, np.array([0, 1], np.int16), fd), exact)
    err_db = 20 * np.log10(np.max(np.abs(shared - exact)) / np.min(np.abs(exact)))
    closed = 20 * np.log10(abs(1 - np.exp(1j * math.pi * b * dtau / 2)))
    assert closed == pytest.approx(-32.04, abs=0.01)
    assert err_db <= -32.0
    assert err_db == pytest.approx(closed, abs=0.05)


# --------------------------------------------------------------------------- known delay


@pytest.mark.parametrize("frac_bw", [0.6, 0.8])
def test_ph20_rt11_known_delay_37p4_samples_recovered(backend_name: str, frac_bw: float) -> None:
    """A band-limited periodic signal through a 37.4-sample path: the cross-spectrum phase
    slope recovers the delay to < 0.01 samples (wideband FD, B up to 0.8 f_s)."""
    fs = Fraction(1_000_000)
    period = 4096
    rng = np.random.Generator(np.random.Philox(key=20261010))
    spec = np.zeros(period, np.complex128)
    f = np.fft.fftfreq(period)
    band = np.abs(f) <= frac_bw / 2
    spec[band] = rng.standard_normal(band.sum()) + 1j * rng.standard_normal(band.sum())
    x = np.fft.ifft(spec)
    x = (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)
    ring = TxRingBuffer("a.tx0", fs, 1 << 15)
    ring.write(0, np.tile(x, 4))
    d = 37.4
    tau = np.array([d / float(fs)])
    out = np.zeros(period, np.complex128)
    get_backend(backend_name).render(  # type: ignore[arg-type]
        _job(
            ring,
            fs,
            2 * period,
            3 * period,
            np.array([1.0 + 0j]),
            tau,
            np.zeros(1, np.int16),
            choose_fd(frac_bw * float(fs), fs),
        ),
        out,
    )
    xin = np.fft.fft(x.astype(np.complex128))
    yout = np.fft.fft(out)
    inb = band & (np.abs(xin) > 1e-3 * np.max(np.abs(xin)))
    fk = f[inb]
    # cross spectrum about the nominal delay; remove the constant carrier phase first
    z = yout[inb] * np.conj(xin[inb]) * np.exp(2j * np.pi * fk * d)
    resid = np.angle(z * np.exp(-1j * np.angle(np.sum(z))))
    a = np.vstack([fk, np.ones_like(fk)]).T
    slope, _ = np.linalg.lstsq(a, resid, rcond=None)[0]
    d_hat = d - slope / (2 * np.pi)
    assert abs(d_hat - d) < 0.01


# --------------------------------------------------------------------------- wideband oracles
# (added in the WP5 review) The output of a multipath link with a band-limited periodic input
# is compared with the exact closed form: in the frequency domain
# Y(f) = X(f) * sum_p a_p exp(-j 2 pi (f_c + f) tau_p) (static), or in the time domain
# y(n) = a(n) exp(-j 2 pi f_c tau(n)) x(n - tau(n) f_s) with x the band-limited periodic
# interpolant (moving chord delay). Error bound (Parseval): per path, |a_p| times the FD row
# error (10**(-60/20), PH-20) plus the narrowband shared-envelope error
# max_{|f| <= B/2} |exp(-j 2 pi f (tau_g - tau_p)) - 1| = 2 sin(pi (B/2) |tau_g - tau_p|).


def _band_limited_periodic(period: int, frac_bw: float, key: int) -> tuple[np.ndarray, np.ndarray]:
    """complex64 periodic signal occupying |f| <= frac_bw/2 and its exact DFT (complex128)."""
    g = np.random.Generator(np.random.Philox(key=key))
    f = np.fft.fftfreq(period)
    band = np.abs(f) <= frac_bw / 2
    spec = np.zeros(period, np.complex128)
    spec[band] = g.standard_normal(band.sum()) + 1j * g.standard_normal(band.sum())
    x = np.fft.ifft(spec)
    x64 = (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)
    return x64, np.fft.fft(x64.astype(np.complex128))


def _rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(v) ** 2)))


@pytest.mark.parametrize("frac_bw", [0.2, 0.6, 0.8])
def test_ph20_wideband_multipath_matches_exact_frequency_response(
    backend_name: str, frac_bw: float
) -> None:
    fs = Fraction(1_000_000)
    b_hz = frac_bw * float(fs)
    period = 4096
    x, xf = _band_limited_periodic(period, frac_bw, 7)
    ring = TxRingBuffer("a.tx0", fs, 1 << 15)
    ring.write(0, np.tile(x, 5))
    fc = 3.5e9
    delta = 0.05 / (math.pi * b_hz)  # narrowband cluster width
    # two clusters (spread 0.6 and 0.9 delta) plus two isolated paths
    tau = np.array([2.13e-6, 2.13e-6 + 0.6 * delta, 5.71e-6, 11.234e-6, 11.234e-6 + 0.9 * delta])
    tau = np.concatenate([tau, [30.9e-6]])
    amp = np.array([1.0, 0.5 - 0.3j, -0.2 + 0.4j, 0.1j, 0.3 + 0.3j, -0.25 + 0j])
    group, nb = plan_groups(tau, tau, b_hz)
    assert not nb
    assert group.tolist() == [0, 0, 1, 2, 2, 3]
    fd = choose_fd(b_hz, fs)
    lj = LinkJob("a.tx0->b.rx0", ring, fc, 0.0, amp, amp.copy(), tau, tau.copy(), group, fd)
    k0 = 2 * period
    out = np.zeros(period, np.complex128)
    get_backend(backend_name).render(  # type: ignore[arg-type]
        SegmentJob("b.rx0", fs, k0, k0 + period, Fraction(0), Fraction(0), [lj]), out
    )
    f = np.fft.fftfreq(period) * float(fs)
    h = np.sum(amp[:, None] * np.exp(-2j * np.pi * (fc + f[None, :]) * tau[:, None]), axis=0)
    y_exact = np.fft.ifft(xf * h)
    # per-path bound: FD row error plus the shared-envelope (half cluster spread) error
    env = np.empty_like(tau)
    for g_id in np.unique(group):
        sel = group == g_id
        env[sel] = 0.5 * (tau[sel].min() + tau[sel].max())
    per_path = 10 ** (-60 / 20) + 2 * np.sin(np.pi * (b_hz / 2) * np.abs(env - tau))
    bound = float(np.sum(np.abs(amp) * per_path)) * _rms(x.astype(np.complex128))
    assert _rms(out - y_exact) <= bound
    # paths alone in their group carry only the FD error: <= -60 dB of their RMS
    single = LinkJob(
        "a.tx0->b.rx0", ring, fc, 0.0, amp[2:3], amp[2:3].copy(), tau[2:3], tau[2:3].copy(),
        np.zeros(1, np.int16), fd,
    )  # fmt: skip
    out1 = np.zeros(period, np.complex128)
    get_backend(backend_name).render(  # type: ignore[arg-type]
        SegmentJob("b.rx0", fs, k0, k0 + period, Fraction(0), Fraction(0), [single]), out1
    )
    y1 = np.fft.ifft(xf * amp[2] * np.exp(-2j * np.pi * (fc + f) * tau[2]))
    assert _rms(out1 - y1) <= 10 ** (-60 / 20) * _rms(y1)


@pytest.mark.parametrize("frac_bw", [0.2, 0.8])
def test_ph20_moving_chord_delay_matches_band_limited_closed_form(
    backend_name: str, frac_bw: float
) -> None:
    """Per-sample FD taps along a chord delay (1.7 samples over the interval) and a
    complex-linear amplitude: y(n) = a(n) e^{-j 2 pi f_c tau(n)} x(n - tau(n) f_s)."""
    fs = Fraction(1_000_000)
    period = 4096
    x, xf = _band_limited_periodic(period, frac_bw, 9)
    ring = TxRingBuffer("a.tx0", fs, 1 << 16)
    ring.write(0, np.tile(x, 10))
    fc = 2.4e9
    t_a, t_b = Fraction(9000, 10**6), Fraction(12000, 10**6)
    tau_a, tau_b = np.array([20.3e-6]), np.array([22.0e-6])
    amp_a, amp_b = np.array([1.0 + 0j]), np.array([0.8 + 0.1j])
    lj = LinkJob(
        "a.tx0->b.rx0", ring, fc, 0.0, amp_a, amp_b, tau_a, tau_b, np.zeros(1, np.int16),
        choose_fd(frac_bw * float(fs), fs),
    )  # fmt: skip
    k0, k1 = 10_000, 11_500  # strictly inside the interval [9000, 12000)
    out = np.zeros(k1 - k0, np.complex128)
    get_backend(backend_name).render(  # type: ignore[arg-type]
        SegmentJob("b.rx0", fs, k0, k1, t_a, t_b, [lj]), out
    )
    n = np.arange(k0, k1)
    w = (n - 9000) / 3000.0  # exact D-1 weight of this interval
    tau = tau_a[0] + w * (tau_b[0] - tau_a[0])
    a = amp_a[0] + w * (amp_b[0] - amp_a[0])
    kk = np.fft.fftfreq(period) * period
    tt = (n - tau * float(fs)) % period
    x_del = np.exp(2j * np.pi * np.outer(tt, kk) / period) @ xf / period
    cyc = fc * tau
    y_exact = a * np.exp(-2j * np.pi * (cyc - np.rint(cyc))) * x_del
    assert _rms(out - y_exact) <= 10 ** (-60 / 20) * _rms(y_exact)


@pytest.mark.parametrize(
    ("fs", "d_samples", "shift"),
    [
        (Fraction(1_000_000), 37.4, 37),
        (Fraction(1_000_000), 37.6, 38),
        (Fraction(1_000_000), 0.49, 0),
        (Fraction(2**20), 37.5, 38),  # tau*f_s is exactly 37.5 here: ties go up
        (Fraction(2**20), 36.5, 37),
    ],
)
def test_ph20_integer_only_delay_rounds_to_nearest_sample(
    backend_name: str, fs: Fraction, d_samples: float, shift: int
) -> None:
    """(review) Integer-only delay (B <= 0.0318 f_s) reads x[n - floor(d + 1/2)] with
    d = tau*f_s (nearest, ties up) and keeps the exact carrier phase exp(-j 2 pi f_c tau)."""
    g = np.random.Generator(np.random.Philox(key=4))
    x = (g.standard_normal(4000) + 1j * g.standard_normal(4000)).astype(np.complex64)
    ring = TxRingBuffer("a.tx0", fs, 1 << 13)
    ring.write(0, x)
    fd = choose_fd(0.03 * float(fs), fs)
    assert fd.filter_id == "integer"
    tau = np.array([d_samples / float(fs)])
    # the expected shift is the exact nearest-ties-up rounding of the float64 delay
    assert math.floor(Fraction(float(tau[0])) * fs + Fraction(1, 2)) == shift
    amp = np.array([0.7 - 0.2j])
    out = np.zeros(1000, np.complex128)
    get_backend(backend_name).render(  # type: ignore[arg-type]
        _job(ring, fs, 2000, 3000, amp, tau, np.zeros(1, np.int16), fd, fc=3.5e9), out
    )
    cyc = float((Fraction(3.5e9) * Fraction(float(tau[0]))) % 1)  # exact f_c*tau mod 1
    expect = amp[0] * np.exp(-2j * np.pi * cyc) * x[2000 - shift : 3000 - shift]
    np.testing.assert_allclose(out, expect.astype(np.complex128), rtol=1e-12, atol=0)


# --------------------------------------------------------------------------- ring


def test_ring_capacity_rule() -> None:
    # H = 2**ceil(log2(tau_max f_s + L_FD + 2P + 1024))
    assert ring_capacity(1e-5, "23.04e6", 32, 8192) == 32768  # 230.4 + 32 + 16384 + 1024
    assert ring_capacity(0.0, "1e6", 1, 512) == 4096  # 1 + 1024 + 1024 -> 2049
    assert ring_capacity(0.0, "1e6", 1, 511) == 2048  # 2047
    with pytest.raises(ConfigurationError):
        TxRingBuffer("a.tx0", "1e6", 1000)


def test_ring_semantics_gap_idle_wrap_and_overwrite() -> None:
    r = TxRingBuffer("a.tx0", "1e6", 16)
    assert r.committed_through() == 0 and r.oldest() == 0
    r.write(0, np.arange(1, 6, dtype=np.complex64))  # [0, 5)
    r.write(8, np.array([9, 10], dtype=np.complex64))  # gap [5, 8) is silence
    assert r.committed_through() == 10
    got = r.read(-3, 16)
    expect = np.array([0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0, 9, 10, 0, 0, 0], dtype=np.complex64)
    np.testing.assert_array_equal(got, expect)  # before 0 and after the head read as 0
    with pytest.raises(ValueError, match="immutable"):
        r.write(9, np.ones(1, np.complex64))
    r.idle_until(20)
    assert r.committed_through() == 20
    r.write(20, np.arange(100, 110, dtype=np.complex64))  # wraps the 16-sample buffer
    assert r.committed_through() == 30 and r.oldest() == 14
    np.testing.assert_array_equal(r.read(20, 10), np.arange(100, 110, dtype=np.complex64))
    np.testing.assert_array_equal(r.read(14, 6), np.zeros(6, np.complex64))  # idle silence
    with pytest.raises(IndexError, match="overwritten"):
        r.read(13, 2)
    out = np.empty(3, np.complex64)
    assert r.read(27, 3, out) is out
    np.testing.assert_array_equal(out, [107, 108, 109])


def test_ring_long_write_keeps_only_the_newest_capacity() -> None:
    r = TxRingBuffer("a.tx0", "1e6", 8)
    r.write(3, np.arange(20, dtype=np.complex64))
    assert r.committed_through() == 23 and r.oldest() == 15
    np.testing.assert_array_equal(r.read(15, 8), np.arange(12, 20, dtype=np.complex64))
