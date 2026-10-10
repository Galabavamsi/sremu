"""Keyed Philox4x32-10 RNG (DESIGN.md 6.1, 6.2; I4, I5; RT-14 Philox parts).

Oracles:
* Random123 known-answer vectors for Philox4x32 with R = 10 rounds (``kat_vectors``);
* an independent pure-Python Philox implementation written here from the published round
  function (Salmon et al., SC'11) with arbitrary-precision ints;
* the spec's key formula: first 8 bytes of BLAKE2b(seed 0x1F purpose 0x1F name), LE words;
* closed forms of the uniform and Box-Muller transforms evaluated per element with ``math``;
* moments and KS distances of the variates against their distributions (fixed keys, so
  deterministic; thresholds are 5-sigma or alpha = 1e-3 critical values).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
from scipy import stats

from sremu import rng
from sremu.errors import ConfigurationError

pytestmark = pytest.mark.unit

BACKENDS = ("numpy_ref", "numba_cpu")
MASK = 0xFFFFFFFF

# Random123 kat_vectors, philox4x32 with 10 rounds: (counter, key, expected output)
KAT_R10 = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((MASK,) * 4, (MASK, MASK), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    (
        (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
        (0xA4093822, 0x299F31D0),
        (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
    ),
]


def ref_philox(ctr: tuple[int, ...], key: tuple[int, int]) -> tuple[int, int, int, int]:
    """Independent pure-Python Philox4x32-10 (Python ints)."""
    m0, m1, w0, w1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for _ in range(10):
        p0, p1 = m0 * c0, m1 * c2
        c0, c1, c2, c3 = (p1 >> 32) ^ c1 ^ k0, p1 & MASK, (p0 >> 32) ^ c3 ^ k1, p0 & MASK
        k0, k1 = (k0 + w0) & MASK, (k1 + w1) & MASK
    return c0, c1, c2, c3


def ref_block(key: tuple[int, int], k0: int, n: int, sub: int) -> np.ndarray:
    rows = [ref_philox(((k & MASK), (k >> 32), sub, 0), key) for k in range(k0, k0 + n)]
    return np.array(rows, dtype=np.uint32).reshape(n, 4)


# --------------------------------------------------------------------------- Philox core


@pytest.mark.parametrize(("ctr", "key", "expected"), KAT_R10)
def test_rt14_philox_known_answer_vectors(ctr, key, expected):
    out = rng.philox4x32(np.array(ctr, dtype=np.uint64), key)
    assert tuple(int(x) for x in out) == expected
    assert ref_philox(ctr, key) == expected  # the test's own reference agrees with Random123


@pytest.mark.parametrize("backend", BACKENDS)
def test_rt14_philox_block_zero_counter_kat(backend):
    # counter (k_lo, k_hi, sub, 0) = (0, 0, 0, 0) with key (0, 0) is the first KAT vector
    out = rng.philox_block((0, 0), 0, 1, sub=0, backend=backend)
    assert tuple(int(x) for x in out[0]) == KAT_R10[0][2]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("k0", "n", "sub"),
    [(0, 5, 0), (2**32 - 3, 7, 1), (2**40 + 11, 4, MASK), (2**63 - 4, 4, 9)],
)
def test_rt14_philox_block_matches_reference_counter_layout(backend, k0, n, sub):
    key = (0x9E3779B9, 0x7F4A7C15)
    assert np.array_equal(
        rng.philox_block(key, k0, n, sub=sub, backend=backend), ref_block(key, k0, n, sub)
    )


@pytest.mark.parametrize(
    ("k0", "n", "sub"),
    [(0, 0, 0), (0, 1, 0), (123_456, 1000, 3), (2**32 - 500, 1000, 0), (7, (1 << 18) + 3, 2)],
)
def test_i5_philox_uint32_numba_equals_numpy_bitwise(k0, n, sub):
    key = rng.derive_key(20261010, "noise", "rx1.rx0")
    a = rng.philox_block(key, k0, n, sub=sub, backend="numpy_ref")
    b = rng.philox_block(key, k0, n, sub=sub, backend="numba_cpu")
    assert a.dtype == b.dtype == np.uint32
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- key derivation


def test_rt14_key_is_first_64_bits_of_blake2b_spec_formula():
    key = rng.derive_key(20261010, "noise", "mu.rx0")
    digest = hashlib.blake2b(b"20261010\x1fnoise\x1fmu.rx0").digest()
    assert key.k0 == int.from_bytes(digest[0:4], "little")
    assert key.k1 == int.from_bytes(digest[4:8], "little")
    assert key.u64 == int.from_bytes(digest[0:8], "little")
    assert rng.PhiloxKey.from_u64(key.u64) == key
    assert rng.key_material(20261010, "noise", "mu.rx0") == b"20261010\x1fnoise\x1fmu.rx0"


def test_rng_scheme_and_purpose_registry_match_spec_6_2():
    assert rng.RNG_SCHEME == "philox4x32-10/blake2b-v1"
    assert set(rng.PURPOSES) == {
        "noise", "pn", "diffuse", "mobility", "failure", "table_mc",
        "ctrl.latency", "ctrl", "trial", "decoy", "traffic",
    }  # fmt: skip


def test_key_material_is_injective_and_validated():
    a = rng.derive_key(1, "trial", "ab/c")
    b = rng.derive_key(1, "trial", "a/bc")
    assert a != b
    assert rng.key_material(1, "ctrl", "x") != rng.key_material(1, "ctrl.latency", "x")
    assert rng.derive_key(1, "noise", "a.rx0") != rng.derive_key(2, "noise", "a.rx0")
    with pytest.raises(ValueError, match="separator"):
        rng.derive_key(1, "noise", "a\x1fb")
    with pytest.raises(ConfigurationError, match="unknown RNG purpose"):
        rng.derive_key(1, "noize", "a.rx0")
    rng.derive_key(1, "custom", "x", strict=False)  # explicit opt-out is allowed
    with pytest.raises(TypeError):
        rng.derive_key(True, "noise", "a.rx0")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rng.derive_key(1.0, "noise", "a.rx0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        rng.derive_key(-1, "noise", "a.rx0")


def test_rt14_adding_a_node_leaves_other_streams_variates_unchanged():
    """I4: keys and variates depend only on (seed, purpose, name, index)."""
    seed = 20261010
    before_rx = ["rx1.rx0", "rx2.rx0"]
    after_rx = ["rx0.rx0", "rx1.rx0", "rx15.rx0", "rx2.rx0", "rx2.rx1"]  # nodes/streams added
    links_before = ["ap.tx0->rx1.rx0", "ap.tx0->rx2.rx0"]
    links_after = ["ap.tx0->rx1.rx0", "ap.tx0->rx2.rx0", "ap.tx0->rx15.rx0", "ap2.tx0->rx1.rx0"]

    def variates(rx_ids, link_ids):
        out = {}
        for r in rx_ids:
            out["noise", r] = rng.complex_normal(rng.derive_key(seed, "noise", r), 10_000, 512, 0.3)
        for lk in link_ids:
            out["diffuse", lk] = rng.uniform(rng.derive_key(seed, "diffuse", lk), 0, 96, sub=5)
        return out

    va, vb = variates(before_rx, links_before), variates(after_rx, links_after)
    for k, v in va.items():
        assert np.array_equal(v, vb[k]), k
    # and the new streams are not copies of existing ones
    assert not np.array_equal(vb["noise", "rx15.rx0"], vb["noise", "rx1.rx0"])


# --------------------------------------------------------------------------- variates


@pytest.mark.parametrize("backend", BACKENDS)
def test_uniform_closed_form_from_words(backend):
    key = rng.derive_key(5, "ctrl.latency", "tdma")
    k0, n = 2**33 + 17, 2000
    words = ref_block(key, k0, n, 4)
    expect = np.array(
        [((int(r0) >> 5) * 2**26 + (int(r1) >> 6)) / 2**53 for r0, r1, _a, _b in words]
    )
    got = rng.uniform(key, k0, n, sub=4, backend=backend)
    assert np.array_equal(got, expect)  # exact: integer formula and an exact power-of-2 scale
    assert got.min() >= 0.0 and got.max() < 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_box_muller_closed_form_from_words(backend):
    key = rng.derive_key(5, "noise", "mu.rx0")
    k0, n, sigma = 999_983, 3000, 0.7
    words = ref_block(key, k0, n, 0)
    exp_c = np.empty(n, dtype=np.complex128)
    exp_r = np.empty(n)
    for i, (r0, r1, _a, _b) in enumerate(words):
        u1 = (int(r0) + 1) / 2**32
        u2 = int(r1) / 2**32
        rad = math.sqrt(-math.log(u1))
        exp_c[i] = complex(
            sigma * rad * math.cos(2 * math.pi * u2), sigma * rad * math.sin(2 * math.pi * u2)
        )
        exp_r[i] = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    got_c = rng.complex_normal(key, k0, n, sigma, backend=backend, dtype=np.complex128)
    got_r = rng.standard_normal(key, k0, n, backend=backend)
    # libm last-ulp differences only (bitwise on hosts where NumPy and numba share the CRT)
    np.testing.assert_allclose(got_c, exp_c, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(got_r, exp_r, rtol=1e-14, atol=1e-15)


def test_i5_variates_across_backends():
    key = rng.derive_key(77, "noise", "rx2.rx0")
    k0, n = 2**31 - 1000, 200_000
    u = [rng.uniform(key, k0, n, sub=1, backend=b) for b in BACKENDS]
    assert np.array_equal(u[0], u[1])  # integer arithmetic: bitwise
    z = [rng.complex_normal(key, k0, n, 1.5, backend=b, dtype=np.complex128) for b in BACKENDS]
    np.testing.assert_allclose(z[1], z[0], rtol=4e-16 * 8, atol=1e-15)
    z64 = [rng.complex_normal(key, k0, n, 1.5, backend=b) for b in BACKENDS]
    rel_rms = np.sqrt(np.mean(np.abs(z64[1] - z64[0]) ** 2) / np.mean(np.abs(z64[0]) ** 2))
    assert rel_rms <= 1e-6  # I5 bound on complex64 output
    # the complex64 stream is the rounded complex128 stream
    assert np.array_equal(z64[0], z[0].astype(np.complex64))


@pytest.mark.parametrize("backend", BACKENDS)
def test_rt14_chunk_invariance_370_plus_630(backend):
    key = rng.derive_key(9, "noise", "a.rx0")
    whole = rng.complex_normal(key, 5000, 1000, 1.0, backend=backend)
    parts = [
        rng.complex_normal(key, 5000, 370, 1.0, backend=backend),
        rng.complex_normal(key, 5370, 630, 1.0, backend=backend),
    ]
    assert np.array_equal(whole, np.concatenate(parts))
    # skipping a range does not change later samples
    assert np.array_equal(rng.complex_normal(key, 5900, 100, 1.0, backend=backend), whole[900:])


def test_rt14_numba_thread_count_invariance():
    numba = pytest.importorskip("numba")
    key = rng.derive_key(3, "noise", "x.rx0")
    n0 = numba.get_num_threads()
    try:
        ref = rng.complex_normal(key, 0, 50_000, 2.0, backend="numba_cpu")
        numba.set_num_threads(1)
        one = rng.complex_normal(key, 0, 50_000, 2.0, backend="numba_cpu")
    finally:
        numba.set_num_threads(n0)
    assert np.array_equal(ref, one)


def test_complex_normal_moments_and_distribution():
    sigma = 2.0
    n = 1_000_000
    z = rng.complex_normal(
        rng.derive_key(11, "noise", "stat.rx0"), 0, n, sigma, dtype=np.complex128
    )
    p = np.abs(z) ** 2  # exponential with mean sigma**2 and std sigma**2
    assert abs(p.mean() - sigma**2) <= 5 * sigma**2 / math.sqrt(n)
    for comp in (z.real, z.imag):
        assert abs(comp.mean()) <= 5 * (sigma / math.sqrt(2)) / math.sqrt(n)
        # var of the sample variance of N(0, s2): 2 s2^2 / n
        s2 = sigma**2 / 2
        assert abs(comp.var() - s2) <= 5 * s2 * math.sqrt(2 / n)
    assert abs(np.corrcoef(z.real, z.imag)[0, 1]) <= 5 / math.sqrt(n)
    # independence across absolute indices and lanes
    assert abs(np.mean(z[1:] * np.conj(z[:-1]))) / sigma**2 <= 5 / math.sqrt(n)
    w = rng.complex_normal(
        rng.derive_key(11, "noise", "stat.rx0"), 0, n, sigma, sub=1, dtype=np.complex128
    )
    assert abs(np.mean(z * np.conj(w))) / sigma**2 <= 5 / math.sqrt(n)
    # distribution: KS of Re against N(0, sigma^2/2) on the first 2e5 samples
    m = 200_000
    d = stats.kstest(z.real[:m] / (sigma / math.sqrt(2)), "norm").statistic
    assert d <= 1.95 / math.sqrt(m)  # alpha = 1e-3 critical value


def test_uniform_and_standard_normal_distribution():
    m = 200_000
    key = rng.derive_key(12, "trial", "ks/0")
    u = rng.uniform(key, 0, m)
    assert stats.kstest(u, "uniform").statistic <= 1.95 / math.sqrt(m)
    x = rng.standard_normal(key, 0, m, sub=2)
    assert stats.kstest(x, "norm").statistic <= 1.95 / math.sqrt(m)
    assert abs(x.mean()) <= 5 / math.sqrt(m)
    assert abs(x.var() - 1) <= 5 * math.sqrt(2 / m)


def test_complex_normal_out_parameter_and_validation():
    key = rng.derive_key(1, "noise", "o.rx0")
    out = np.empty(64, dtype=np.complex64)
    res = rng.complex_normal(key, 10, 64, 1.0, out=out)
    assert res is out
    assert np.array_equal(out, rng.complex_normal(key, 10, 64, 1.0))
    assert np.all(rng.complex_normal(key, 0, 16, 0.0) == 0)
    with pytest.raises(ValueError):
        rng.complex_normal(key, 0, 8, 1.0, out=np.empty(7, dtype=np.complex64))
    with pytest.raises(ValueError):
        rng.complex_normal(key, 0, 8, -1.0)
    with pytest.raises(ValueError):
        rng.complex_normal(key, 0, 8, 1.0, dtype=np.float64)
    with pytest.raises(ValueError):
        rng.philox_block(key, -1, 4)
    with pytest.raises(ValueError):
        rng.philox_block(key, 2**63 - 2, 3)
    with pytest.raises(ValueError):
        rng.philox_block(key, 0, 4, sub=2**32)
    with pytest.raises(ValueError):
        rng.philox_block((2**32, 0), 0, 4)
    with pytest.raises(TypeError):
        rng.philox_block(key, 1.0, 4)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        rng.uniform(key, 0, 4, backend="torch")  # type: ignore[arg-type]


def test_generator_factory_is_keyed_philox():
    g = rng.generator(20261010, "ctrl", "tdma")
    key = rng.derive_key(20261010, "ctrl", "tdma")
    ref = np.random.Generator(np.random.Philox(key=key.u64))
    assert np.array_equal(g.integers(0, 2**63, 64), ref.integers(0, 2**63, 64))
    a = rng.generator(20261010, "ctrl", "tdma").standard_normal(32)
    b = rng.generator(20261010, "ctrl", "tdma").standard_normal(32)
    c = rng.generator(20261010, "ctrl", "other").standard_normal(32)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generator_counter_selects_independent_streams():
    key = rng.derive_key(7, "mobility", "mu")
    for i in (0, 1, 2**40 + 3):
        g = rng.generator(7, "mobility", "mu", counter=i)
        ref = np.random.Generator(
            np.random.Philox(key=key.u64, counter=np.array([0, i, 0, 0], dtype=np.uint64))
        )
        assert np.array_equal(g.random(16), ref.random(16))
    legs = [rng.generator(7, "mobility", "mu", counter=i).random(64) for i in range(3)]
    assert not np.array_equal(legs[0], legs[1]) and not np.array_equal(legs[1], legs[2])
    assert np.array_equal(rng.generator(7, "mobility", "mu").random(8), legs[0][:8])
    with pytest.raises(ValueError):
        rng.generator(7, "mobility", "mu", counter=-1)
    with pytest.raises(TypeError):
        rng.generator(7, "mobility", "mu", counter=1.0)  # type: ignore[arg-type]
