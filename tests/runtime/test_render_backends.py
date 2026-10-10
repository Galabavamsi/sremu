"""WP5: backend equivalence (I5) and linearity / time shift / seeds (PH-22).

I5: ``numba_cpu`` vs ``numpy_ref`` <= 1e-9 relative on the complex128 accumulator and
<= 1e-6 relative RMS after the complex64 cast, on jobs that exercise every kernel feature
(multi-link, LO offsets, multi-group wideband FD, all transition profiles, integer delay,
empty links) and on moving scenes. PH-22 oracles are exact algebraic identities.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.channel.fd import kaiser_fd_spec
from sremu.channel.render_numpy import get_backend
from sremu.channel.ring import TxRingBuffer
from sremu.contracts import FDSpec, LinkJob, PanelConfig, RenderBackend, SegmentJob, TransitionTerm
from sremu.impair.noise import noise_numpy

from ._render_support import SpanSpec, const_ring, focus_states, lab_walk, render_span, vehicle

pytestmark = pytest.mark.runtime


def _rng(tag: int) -> np.random.Generator:
    return np.random.Generator(np.random.Philox(key=tag))


def _cplx(g: np.random.Generator, n: int) -> np.ndarray:
    return g.standard_normal(n) + 1j * g.standard_normal(n)


def _feature_job(
    fs: Fraction, k0: int, k1: int, rings: list[TxRingBuffer], static: bool = False
) -> SegmentJob:
    g = _rng(5)
    links = []
    # link 0: wideband, 3 FD groups (32 taps), LO offset, three transition profiles
    p = 6
    tau_a = np.array([1.0e-6, 1.2e-6, 2.5e-6, 2.6e-6, 7.9e-6, 8.05e-6])
    tau_b = tau_a + (0.0 if static else np.array([1e-8, -2e-8, 3e-9, 0.0, 5e-8, -1e-8]))
    trans = [
        TransitionTerm(k0 + 300, "linear", 437.5, _cplx(g, p), _cplx(g, p)),
        TransitionTerm(k0 + 900, "first_order", 120.0, _cplx(g, p), _cplx(g, p)),
        TransitionTerm(k0 + 1500, "step", 0.0, _cplx(g, p), _cplx(g, p)),
    ]
    links.append(
        LinkJob(
            "a.tx0->r.rx0",
            rings[0],
            3.5e9,
            1234.567,
            _cplx(g, p),
            _cplx(g, p),
            tau_a,
            tau_b,
            np.array([0, 0, 1, 1, 2, 2], np.int16),
            kaiser_fd_spec(32),
            trans,
        )
    )
    # link 1: narrowband 8-tap, negative LO offset
    tau = np.array([3.3e-6, 3.3001e-6, 3.3002e-6])
    links.append(
        LinkJob(
            "b.tx0->r.rx0",
            rings[1],
            2.4e9,
            -77.25,
            _cplx(g, 3),
            _cplx(g, 3),
            tau,
            tau + 2e-9,
            np.zeros(3, np.int16),
            kaiser_fd_spec(8),
        )
    )
    # link 2: no paths at all (a link whose panel is absent)
    e = np.zeros(0)
    links.append(
        LinkJob(
            "c.tx0->r.rx0",
            rings[2],
            1e9,
            0.0,
            e.astype(np.complex128),
            e.astype(np.complex128),
            e,
            e.copy(),
            np.zeros(0, np.int16),
            kaiser_fd_spec(16),
        )
    )
    # link 3: integer delay, 2 paths, large delay (about 0.2 ms)
    tau = np.array([2.0e-4, 2.0e-4 + 1e-9])
    links.append(
        LinkJob(
            "d.tx0->r.rx0",
            rings[3],
            1e9,
            0.0,
            _cplx(g, 2),
            _cplx(g, 2),
            tau,
            tau + 3e-7,
            np.zeros(2, np.int16),
            FDSpec.integer(),
        )
    )
    return SegmentJob(
        "r.rx0", fs, k0, k1, Fraction(k0 - 37, int(fs)), Fraction(k1 + 400, int(fs)), links
    )


def _rings(fs: Fraction, n: int) -> list[TxRingBuffer]:
    out = []
    for i, tag in enumerate(("a", "b", "c", "d")):
        r = TxRingBuffer(f"{tag}.tx0", fs, 1 << 16)
        r.write(0, _cplx(_rng(100 + i), n).astype(np.complex64))
        out.append(r)
    return out


def _render(backend: str, job: SegmentJob) -> np.ndarray:
    out = np.zeros(job.k1 - job.k0, np.complex128)
    get_backend(backend).render(job, out)  # type: ignore[arg-type]
    return out


def _i5_check(ref: np.ndarray, got: np.ndarray) -> None:
    assert np.max(np.abs(got - ref)) <= 1e-9 * np.max(np.abs(ref))
    r64 = ref.astype(np.complex64).astype(np.complex128)
    g64 = got.astype(np.complex64).astype(np.complex128)
    assert np.sqrt(np.mean(np.abs(g64 - r64) ** 2)) <= 1e-6 * np.sqrt(np.mean(np.abs(r64) ** 2))


@pytest.mark.parametrize("static", [False, True], ids=["moving", "static_delays"])
def test_i5_backends_agree_on_a_feature_job(static: bool) -> None:
    """Moving delays use the per-sample tap path; constant delays the once-per-block path."""
    fs = Fraction(23_040_000)
    rings = _rings(fs, 30_000)
    job = _feature_job(fs, 10_000, 12_337, rings, static)
    ref = _render("numpy_ref", job)
    got = _render("numba_cpu", job)
    assert np.max(np.abs(ref)) > 0
    _i5_check(ref, got)


@pytest.mark.parametrize("scene_fn", [lab_walk, vehicle], ids=lambda f: f.__name__)
def test_i5_backends_agree_on_moving_scenes(scene_fn) -> None:
    sc = scene_fn()
    model = sc.build(2, 2)
    cfg = {
        sc.panel: PanelConfig(
            sc.panel, focus_states(model, sc.panel, sc.link, sc.t_focus_s, sc.fc_hz)
        )
    }
    ring = TxRingBuffer("tx.tx0", sc.f_s, 1 << 13)
    ring.write(0, _cplx(_rng(9), 6000).astype(np.complex64))
    k_lo = int(sc.window_s[0] * float(sc.f_s))
    spec = SpanSpec(
        model,
        cfg,
        sc.link,
        sc.fc_hz,
        ring,
        sc.f_s,
        Fraction(1, 1000),
        occupied_bw_hz=0.6 * float(sc.f_s),
        lo_offset_hz=321.0,
    )
    ref = render_span(spec, k_lo, k_lo + 400, "numpy_ref")
    got = render_span(spec, k_lo, k_lo + 400, "numba_cpu")
    _i5_check(ref, got)


def test_backends_are_render_backends_and_add_into_out() -> None:
    fs = Fraction(23_040_000)
    rings = _rings(fs, 30_000)
    job = _feature_job(fs, 10_000, 10_500, rings)
    for name in ("numpy_ref", "numba_cpu"):
        be = get_backend(name)  # type: ignore[arg-type]
        assert isinstance(be, RenderBackend) and be.name == name
        base = _cplx(_rng(3), 500)
        out = base.copy()
        be.render(job, out)
        np.testing.assert_allclose(out - base, _render(name, job), rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- PH-22


def _small_int_signal(g: np.random.Generator, n: int) -> np.ndarray:
    # exactly representable in complex64, so a*x1 + b*x2 is exact too
    return (g.integers(-8, 9, n) + 1j * g.integers(-8, 9, n)).astype(np.complex64)


def test_ph22_linearity(backend_name: str) -> None:
    fs = Fraction(23_040_000)
    g = _rng(22)
    x1 = _small_int_signal(g, 30_000)
    x2 = _small_int_signal(g, 30_000)
    a, b = 3, -2
    outs = []
    for x in (
        x1,
        x2,
        (a * x1.astype(np.complex128) + b * x2.astype(np.complex128)).astype(np.complex64),
    ):
        rings = []
        for i, tag in enumerate(("a", "b", "c", "d")):  # a different shift of x on every TX
            r = TxRingBuffer(f"{tag}.tx0", fs, 1 << 16)
            r.write(0, np.roll(x, 97 * i))
            rings.append(r)
        outs.append(_render(backend_name, _feature_job(fs, 10_000, 12_000, rings)))
    y1, y2, y3 = outs
    assert np.max(np.abs(y3 - (a * y1 + b * y2))) <= 1e-12 * np.max(np.abs(y3))


def test_ph22_time_shift_equivariance_static_channel(backend_name: str) -> None:
    fs = Fraction(1_000_000)
    g = _rng(23)
    x = _cplx(g, 5000).astype(np.complex64)
    shift = 1237
    r1 = TxRingBuffer("a.tx0", fs, 1 << 14)
    r1.write(0, x)
    r2 = TxRingBuffer("a.tx0", fs, 1 << 14)
    r2.write(shift, x)
    amp = _cplx(g, 4)
    tau = np.array([2.1e-6, 2.13e-6, 5.5e-6, 9.07e-6])

    def job(ring: TxRingBuffer, k0: int) -> SegmentJob:
        lj = LinkJob(
            "a.tx0->b.rx0",
            ring,
            3.5e9,
            0.0,
            amp,
            amp.copy(),
            tau,
            tau.copy(),
            np.array([0, 0, 1, 2], np.int16),
            kaiser_fd_spec(16),
        )
        return SegmentJob("b.rx0", fs, k0, k0 + 3000, Fraction(0), Fraction(0), [lj])

    y1 = _render(backend_name, job(r1, 500))
    y2 = _render(backend_name, job(r2, 500 + shift))
    assert np.max(np.abs(y2 - y1)) <= 1e-12 * np.max(np.abs(y1))


@pytest.mark.parametrize("key", [(0, 0), (1, 2), (0xFFFFFFFF, 0x12345678)])
def test_ph22_identical_output_across_seeds_with_noise_off(
    key: tuple[int, int], backend_name: str
) -> None:
    be = get_backend(backend_name)  # type: ignore[arg-type]
    z = be.noise(key, 1000, 500, 0.0)
    assert z.dtype == np.complex64 and np.all(z == 0)
    fs = Fraction(1_000_000)
    ring = const_ring("a.tx0", fs, 2000, 0.5 + 0.25j)
    one = np.ones(1, np.complex128)
    lj = LinkJob(
        "a.tx0->b.rx0",
        ring,
        3.5e9,
        0.0,
        one,
        one.copy(),
        np.array([1e-6]),
        np.array([1e-6]),
        np.zeros(1, np.int16),
        FDSpec.integer(),
    )
    y = np.zeros(500, np.complex128)
    be.render(SegmentJob("b.rx0", fs, 1000, 1500, Fraction(0), Fraction(0), [lj]), y)
    y_noisy = y + be.noise(key, 1000, 500, 0.0)
    assert np.array_equal(y, y_noisy)
    assert np.array_equal(noise_numpy(key, 1000, 500, 0.0), z)


def test_render_memo_is_content_addressed(backend_name: str) -> None:
    """Memoized preparation never serves stale data: mutating an input array in place (or
    rebuilding an equal job) gives exactly the output of a fresh job with those values."""
    fs = Fraction(23_040_000)
    rings = _rings(fs, 30_000)
    job = _feature_job(fs, 10_000, 10_700, rings)
    y1 = _render(backend_name, job)
    assert np.array_equal(y1, _render(backend_name, _feature_job(fs, 10_000, 10_700, rings)))
    job.links[0].amp_a[2] *= -3.0
    job.links[1].tau_b[0] += 1e-9
    y2 = _render(backend_name, job)
    fresh = _feature_job(fs, 10_000, 10_700, rings)
    fresh.links[0].amp_a[2] *= -3.0
    fresh.links[1].tau_b[0] += 1e-9
    assert not np.array_equal(y1, y2)
    assert np.array_equal(y2, _render(backend_name, fresh))


@pytest.mark.parametrize("chunk", [11, 64, 1000])
def test_rt09_render_lo_offset_continuity(backend_name: str, chunk: int) -> None:
    """RT-09 through the renderer: a 500 Hz carrier offset at 5880 S/s rendered in chunks has
    every sample-to-sample phase step equal to 2*pi*500/5880 within 1e-9 rad."""
    fs = Fraction(5880)
    n = 11_760
    ring = const_ring("a.tx0", fs, n + 64)
    one = np.ones(1, np.complex128)
    zero = np.zeros(1)
    lj = LinkJob("a.tx0->b.rx0", ring, 1e9, 500.0, one, one.copy(), zero, zero.copy(),
                 np.zeros(1, np.int16), FDSpec.integer())  # fmt: skip
    y = np.zeros(n, np.complex128)
    be = get_backend(backend_name)  # type: ignore[arg-type]
    for k in range(0, n, chunk):
        e = min(n, k + chunk)
        be.render(SegmentJob("b.rx0", fs, k, e, Fraction(0), Fraction(0), [lj]), y[k:e])
    steps = np.angle(y[1:] * np.conj(y[:-1]))
    assert np.max(np.abs(steps - 2 * np.pi * 500 / 5880)) <= 1e-9


@pytest.mark.parametrize("fs", [Fraction(23_040_000), Fraction(10**6, 3)], ids=["23p04M", "1M_3"])
def test_rt09_render_lo_offset_absolute_phase_at_large_indices(
    backend_name: str, fs: Fraction
) -> None:
    """(review) The render LO rotator is a pure function of the absolute RX index: near
    k = 2**40 it equals the exact rational phase 2*pi*frac(df*k/f_s) (Python Fraction) to
    1e-9 rad, for segments that start mid-block and straddle 64-sample anchors."""
    df = 1234.567
    k_base = 2**40 + 1_000_003
    ring = TxRingBuffer("a.tx0", fs, 1 << 12)
    ring.idle_until(k_base - 8)
    ring.write(k_base - 8, np.ones(400, np.complex64))
    one = np.ones(1, np.complex128)
    zero = np.zeros(1)
    lj = LinkJob("a.tx0->b.rx0", ring, 1e9, df, one, one.copy(), zero, zero.copy(),
                 np.zeros(1, np.int16), FDSpec.integer())  # fmt: skip
    be = get_backend(backend_name)  # type: ignore[arg-type]
    edges = [k_base, k_base + 5, k_base + 70, k_base + 199, k_base + 333]
    y = np.zeros(edges[-1] - edges[0], np.complex128)
    for a, b in itertools.pairwise(edges):
        be.render(
            SegmentJob("b.rx0", fs, a, b, Fraction(0), Fraction(0), [lj]),
            y[a - k_base : b - k_base],
        )
    exact = np.array(
        [float((Fraction(df) * k / fs) % 1) for k in range(edges[0], edges[-1])], dtype=np.float64
    )
    err = np.angle(y * np.exp(-2j * np.pi * exact))
    assert np.max(np.abs(err)) <= 1e-9
    assert np.max(np.abs(np.abs(y) - 1.0)) <= 1e-12


def test_i2a_i5_random_rational_rates_and_cuts() -> None:
    """(review) Seeded property loop: random rational rates, snapshot intervals with
    fractional n_a, moving multi-group FD paths, LO offsets and a linear transition. Each
    backend is bitwise invariant to random segment cuts (I2a) and numba_cpu matches
    numpy_ref within 1e-9 relative on the complex128 accumulator (I5)."""
    g = _rng(77)
    done = 0
    while done < 12:
        fs = Fraction(int(g.integers(1000, 10**7)), int(g.integers(1, 7)))
        t_a = Fraction(int(g.integers(0, 10**6)), int(g.integers(1, 10**4))) / 7
        scale = Fraction(int(g.integers(1, 9)), int(g.integers(1, 9)))
        t_b = t_a + Fraction(int(g.integers(50, 3000))) / fs * scale
        n_first, n_end = math.ceil(t_a * fs), math.ceil(t_b * fs)
        if n_end - n_first < 3:
            continue
        done += 1
        ring = TxRingBuffer("a.tx0", fs, 1 << 22)
        lo = max(0, n_first - 3000)
        ring.idle_until(lo)
        ring.write(lo, _cplx(g, n_end - lo + 100).astype(np.complex64))
        p = int(g.integers(1, 5))
        tau_a = g.uniform(0, 500, p) / float(fs)
        tau_b = np.abs(tau_a + g.uniform(-2, 2, p) / float(fs))
        trans = [
            TransitionTerm(
                int(g.integers(n_first, n_end)),
                "linear",
                float(g.uniform(1, 100)),
                _cplx(g, p),
                _cplx(g, p),
            )
        ]
        lj = LinkJob(
            "a.tx0->b.rx0", ring, float(g.uniform(1e8, 3e10)), float(g.uniform(-1e4, 1e4)),
            _cplx(g, p), _cplx(g, p), tau_a, tau_b, np.arange(p, dtype=np.int16),
            kaiser_fd_spec(16), trans,
        )  # fmt: skip
        cuts = sorted({n_first, n_end, *(int(c) for c in g.integers(n_first, n_end, 5))})
        full = {}
        for be in ("numpy_ref", "numba_cpu"):
            whole = _render(be, SegmentJob("r.rx0", fs, n_first, n_end, t_a, t_b, [lj]))
            parts = np.concatenate(
                [
                    _render(be, SegmentJob("r.rx0", fs, a, b, t_a, t_b, [lj]))
                    for a, b in itertools.pairwise(cuts)
                ]
            )
            assert np.array_equal(whole, parts), (be, fs, t_a, t_b)
            full[be] = whole
        ref = full["numpy_ref"]
        assert np.max(np.abs(full["numba_cpu"] - ref)) <= 1e-9 * np.max(np.abs(ref))
