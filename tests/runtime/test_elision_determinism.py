"""Bitwise chunk invariance against NumPy temporary elision (D-4, I2a; Linux regression).

NumPy's SIMD complex128 multiply uses FMA, so it is not bitwise commutative: the imaginary
part ``fma(ar, bi, ai*br)`` of ``a*b`` and ``fma(br, ai, bi*ar)`` of ``b*a`` differ in the
last bit for about a third of the samples. NumPy builds with backtrace support (Linux/glibc,
macOS; not Windows) elide temporaries of at least 256 KiB: ``x * f(...)`` with an owning
temporary on the right is evaluated in place as ``f(...) *= x``, i.e. with the operands
swapped. A product written ``x * phasor(k0, n)`` therefore changed its bits at
n >= 16384 complex128 samples on Linux only, which broke the chunk-invariance claims there
while they held on Windows. These tests pin the evaluation order for chunks on both sides of
the 16384-sample threshold.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.channel.render_numpy import get_backend
from sremu.channel.ring import TxRingBuffer
from sremu.contracts import FDSpec, LinkJob, SegmentJob
from sremu.engine.emulator import Emulator, EngineConfig
from sremu.impair.oscillator import WienerPhaseNoise, apply_lo, cfo_phasor

from ._engine_support import client_run, lab_like_fake, stream_cfgs, waveform

pytestmark = pytest.mark.runtime

ELIDE_SAMPLES = (256 * 1024) // 16  # NumPy's elision threshold in complex128 samples
N = 40_000
# chunks on both sides of the threshold (16384 and 20000 are elision candidates)
SIZES = [ELIDE_SAMPLES, 1, 20_000, 333, ELIDE_SAMPLES - 1, 17, 4096]


def _hazard_live() -> bool:
    """Whether this NumPy build evaluates ``a * <temporary>`` as ``<temporary> * a`` with a
    different result (elision on, complex multiply not commutative), i.e. whether the
    regression tests below have teeth on this platform (Linux: yes; Windows: no)."""
    g = np.random.Generator(np.random.Philox(key=60))
    a = g.standard_normal(ELIDE_SAMPLES) + 1j * g.standard_normal(ELIDE_SAMPLES)
    b = np.exp(1j * g.uniform(-3.0, 3.0, ELIDE_SAMPLES))
    return not np.array_equal(a * b.copy(), np.multiply(a, b))


def _chunks(total: int, sizes: list[int]) -> list[tuple[int, int]]:
    out, k, i = [], 0, 0
    while k < total:
        n = min(sizes[i % len(sizes)], total - k)
        out.append((k, n))
        k += n
        i += 1
    return out


def _signal(tag: int, n: int) -> np.ndarray:
    g = np.random.Generator(np.random.Philox(key=tag))
    return (g.standard_normal(n) + 1j * g.standard_normal(n)) / math.sqrt(2)


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def test_cfo_phasor_products_are_evaluated_as_written() -> None:
    fs, df = Fraction(30_720_000), -2345.25
    x = _signal(61, N)
    p = cfo_phasor(df, fs, 0, N)
    assert _same_bits(x * cfo_phasor(df, fs, 0, N), np.multiply(x, p))
    assert _same_bits(cfo_phasor(df, fs, 0, N) * x, np.multiply(p, x))
    pieces = np.concatenate([x[k : k + c] * cfo_phasor(df, fs, k, c) for k, c in _chunks(N, SIZES)])
    assert _same_bits(x * cfo_phasor(df, fs, 0, N), pieces)


def test_phase_noise_phasor_products_are_evaluated_as_written() -> None:
    pn = WienerPhaseNoise((71, 72), 200.0, Fraction(30_720_000))
    x = _signal(62, N)
    p = pn.phasor(0, N)
    assert _same_bits(x * pn.phasor(0, N), np.multiply(x, p))
    assert _same_bits(pn.phasor(0, N) * x, np.multiply(p, x))
    pieces = np.concatenate([x[k : k + c] * pn.phasor(k, c) for k, c in _chunks(N, SIZES)])
    assert _same_bits(x * pn.phasor(0, N), pieces)


@pytest.mark.parametrize("n", [1, ELIDE_SAMPLES, N])
def test_phasors_are_non_owning_views_on_every_os(n: int) -> None:
    """Structural pin with teeth on every OS (the behavioural tests in this file can only
    fail where elision is live, i.e. not on Windows): the phasor helpers return arrays that
    do not own their data, so NumPy can never elide them as the right-hand operand of
    ``x * phasor``."""
    fs = Fraction(30_720_000)
    pn = WienerPhaseNoise((71, 72), 200.0, fs)
    pn0 = WienerPhaseNoise((71, 72), 0.0, fs)
    for p in (
        cfo_phasor(-2345.25, fs, 5, n),
        cfo_phasor(0.0, fs, 5, n),
        pn.phasor(5, n),
        pn0.phasor(5, n),
    ):
        assert p.shape == (n,) and p.dtype == np.complex128
        assert not p.flags.owndata


@pytest.mark.parametrize("direction", ["tx", "rx"])
def test_apply_lo_matches_the_fixed_order_composition(direction: str) -> None:
    """``apply_lo`` = x * (cfo * pn) (conjugated for RX), in that operand order, per chunk.

    The output is complex64, so this pins the composition and its chunk invariance only to
    complex64 precision: a 1-ulp complex128 operand-order change flips the float32 rounding
    with probability about 2**-29 per component, so this test does not detect the elision
    hazard (it passes with the pre-fix ``rot * pn.phasor(...)`` on Linux too). The complex128
    tests above and below do."""
    fs = Fraction(30_720_000)
    pn = WienerPhaseNoise((81, 82), 150.0, fs)
    x = _signal(63, N).astype(np.complex64)
    rot = np.multiply(cfo_phasor(777.5, fs, 0, N), pn.phasor(0, N))
    if direction == "rx":
        rot = np.conj(rot)
    ref = np.multiply(x.astype(np.complex128), rot).astype(np.complex64)
    whole = apply_lo(x, 0, fs, cfo_hz=777.5, pn=pn, direction=direction)  # type: ignore[arg-type]
    assert _same_bits(whole, ref)
    pieces = np.concatenate(
        [
            apply_lo(x[k : k + c], k, fs, cfo_hz=777.5, pn=pn, direction=direction)  # type: ignore[arg-type]
            for k, c in _chunks(N, SIZES)
        ]
    )
    assert _same_bits(pieces, ref)


def test_render_lo_offset_segment_cuts_bitwise_above_the_elision_threshold(
    backend_name: str,
) -> None:
    """The render LO rotation (``lo_offset_hz``) of one 20000-sample segment equals the same
    span rendered in cuts shorter than 16384 samples, bitwise, on both backends."""
    fs = Fraction(30_720_000)
    n, k0 = 20_000, 64
    ring = TxRingBuffer("a.tx0", fs, 1 << 15)
    ring.write(0, _signal(64, k0 + n + 64).astype(np.complex64))
    amp = _signal(65, 2)
    tau = np.array([0.0, 3.0 / 30_720_000])
    lj = LinkJob("a.tx0->b.rx0", ring, 3.5e9, 1234.567, amp, amp.copy(), tau, tau.copy(),
                 np.zeros(2, np.int16), FDSpec.integer())  # fmt: skip
    be = get_backend(backend_name)  # type: ignore[arg-type]
    whole = np.zeros(n, np.complex128)
    be.render(SegmentJob("b.rx0", fs, k0, k0 + n, Fraction(0), Fraction(0), [lj]), whole)
    cut = np.zeros(n, np.complex128)
    for a, b in ((0, 7), (7, ELIDE_SAMPLES - 1), (ELIDE_SAMPLES - 1, n)):
        job = SegmentJob("b.rx0", fs, k0 + a, k0 + b, Fraction(0), Fraction(0), [lj])
        be.render(job, cut[a:b])
    assert np.all(np.abs(whole) > 0)
    assert _same_bits(whole, cut)


def test_engine_rx_phase_noise_tap_is_chunk_invariant_at_16384_sample_pages() -> None:
    """RT-03 style at 30.72 MS/s (16384-sample pages) with RX CFO and Wiener phase noise:
    page-size client chunks vs 4096-sample chunks give the same complex128 RX tap and the
    same complex64 RX output, bitwise."""
    n_rx = 2 * ELIDE_SAMPLES + 5000

    def run(chunk: int) -> tuple[np.ndarray, np.ndarray, set[int]]:
        model = lab_like_fake("30.72e6")
        cfgs = stream_cfgs(
            model,
            bw=8.0e6,
            cfo={"ap.tx0": 137.0, "rx1.rx0": -421.5},
            extra={"rx1.rx0": {"linewidth_hz": 150.0}},
        )
        eng = Emulator(
            model, streams=cfgs, config=EngineConfig(backend="numba_cpu", render_tiles="model")
        )
        tap = eng.capture_accumulator("rx1.rx0")
        lens: set[int] = set()
        eng.on_page(lambda rx, k0, k1: lens.add(k1 - k0) if rx == "rx1.rx0" else None)
        y = client_run(
            eng, "ap.tx0", "rx1.rx0", waveform(n_rx + 200), n_rx, send_chunk=chunk, read_chunk=chunk
        )
        acc = np.concatenate([a for _, a in sorted(tap, key=lambda t: t[0])])[:n_rx]
        return y, acc, lens

    y_ref, acc_ref, lens_ref = run(ELIDE_SAMPLES)
    y, acc, lens = run(4096)
    # teeth: the reference processes whole 16384-sample pages, the chunked run only shorter
    # page portions, so the two runs sit on opposite sides of the elision threshold
    assert ELIDE_SAMPLES in lens_ref and max(lens) < ELIDE_SAMPLES
    assert _same_bits(acc, acc_ref)
    assert np.array_equal(y.view(np.uint64), y_ref.view(np.uint64))
