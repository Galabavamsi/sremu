"""Property tests of the keyed Philox RNG (hypothesis; I4, I5, RT-14).

* any split of an absolute index range into chunks reproduces the single call bitwise;
* the NumPy twin equals the independent pure-Python Philox for random keys, counters, lanes;
* numba equals NumPy bitwise (uint32 words) for random ranges, including ranges that cross
  a 2**32 boundary of the index (the k_hi counter word).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sremu import rng
from tests.unit.test_rng import ref_block, ref_philox

hypothesis = pytest.importorskip("hypothesis")
st = hypothesis.strategies
given, settings = hypothesis.given, hypothesis.settings

pytestmark = pytest.mark.unit

u32 = st.integers(0, 2**32 - 1)
keys = st.tuples(u32, u32)


@settings(max_examples=60, deadline=None)
@given(ctr=st.tuples(u32, u32, u32, u32), key=keys)
def test_rt14_prop_numpy_twin_equals_pure_python(ctr, key):
    out = rng.philox4x32(np.array(ctr, dtype=np.uint64), key)
    assert tuple(int(x) for x in out) == ref_philox(ctr, key)


@settings(max_examples=40, deadline=None)
@given(
    key=keys,
    k0=st.one_of(
        st.integers(0, 2**20), st.integers(2**32 - 64, 2**32 + 64), st.integers(0, 2**63 - 2**12)
    ),
    n=st.integers(0, 300),
    sub=u32,
)
def test_i5_prop_numba_equals_numpy_and_reference(key, k0, n, sub):
    a = rng.philox_block(key, k0, n, sub=sub, backend="numpy_ref")
    b = rng.philox_block(key, k0, n, sub=sub, backend="numba_cpu")
    assert np.array_equal(a, b)
    if n <= 40:
        assert np.array_equal(a, ref_block(key, k0, n, sub))


@settings(max_examples=40, deadline=None)
@given(
    k0=st.integers(0, 2**45),
    cuts=st.lists(st.integers(0, 2000), min_size=0, max_size=6),
    backend=st.sampled_from(["numpy_ref", "numba_cpu"]),
    sub=st.integers(0, 3),
)
def test_rt14_prop_chunking_is_invisible(k0, cuts, backend, sub):
    key = rng.derive_key(42, "noise", "p.rx0")
    n = 2000
    whole = rng.complex_normal(key, k0, n, 1.0, sub=sub, backend=backend)
    bounds = [0, *sorted(set(cuts)), n]
    parts = [
        rng.complex_normal(key, k0 + a, b - a, 1.0, sub=sub, backend=backend)
        for a, b in itertools.pairwise(bounds)
    ]
    assert np.array_equal(whole, np.concatenate(parts))
    u_whole = rng.uniform(key, k0, n, sub=sub, backend=backend)
    u_parts = [
        rng.uniform(key, k0 + a, b - a, sub=sub, backend=backend)
        for a, b in itertools.pairwise(bounds)
    ]
    assert np.array_equal(u_whole, np.concatenate(u_parts))


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(0, 2**63 - 1),
    purpose=st.sampled_from(sorted(rng.PURPOSES)),
    name=st.text(min_size=0, max_size=20).filter(lambda s: "\x1f" not in s),
)
def test_rt14_prop_key_formula(seed, purpose, name):
    import hashlib

    digest = hashlib.blake2b(
        str(seed).encode() + b"\x1f" + purpose.encode() + b"\x1f" + name.encode()
    ).digest()
    key = rng.derive_key(seed, purpose, name)
    assert key.u64 == int.from_bytes(digest[:8], "little")
