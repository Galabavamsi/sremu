"""RT-02 / RT-01 property tests (hypothesis, with the ``tests._prop`` seeded fallback).

Properties are checked against definitions, not against the implementation: membership of
k/f in [t0, t1) by enumeration, the rounding bound |k/f - t| <= 1/(2f) with ties up, and
power-of-two page sizes nearest in the log domain.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.timebase import (
    PAGE_MAX,
    PAGE_MIN,
    Rate,
    TimeSpec,
    ceil_ticks,
    count,
    fraction_to_str,
    from_ticks,
    page_size,
    parse_exact,
    to_exact,
    to_ticks,
)

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

pytestmark = pytest.mark.unit

rates_st = st.builds(Fraction, st.integers(1, 10**9), st.integers(1, 1000)) | st.sampled_from(
    [Fraction(5880), Fraction(23_040_000), Fraction(30_720_000)]
)
times_st = st.fractions(min_value=0, max_value=10**6, max_denominator=10**9)
small_times_st = st.fractions(min_value=0, max_value=1, max_denominator=10**7)

FAST = settings(max_examples=300, deadline=None)


@FAST
@given(k=st.integers(0, 2**40), f=rates_st)
def test_rt02_prop_round_trip(k, f):
    assert to_ticks(from_ticks(k, f), f) == k
    assert TimeSpec.from_ticks(k, f).to_ticks(f) == k


@FAST
@given(t=times_st, f=rates_st)
def test_rt02_prop_nearest_within_half_sample(t, f):
    k = to_ticks(t, f)
    err = Fraction(k) / f - t
    assert -Fraction(1, 2) / f < err <= Fraction(1, 2) / f  # ties resolve upwards


@FAST
@given(t1=times_st, t2=times_st, f=rates_st)
def test_rt02_prop_monotone(t1, t2, f):
    lo, hi = min(t1, t2), max(t1, t2)
    assert to_ticks(lo, f) <= to_ticks(hi, f)
    assert ceil_ticks(lo, f) <= ceil_ticks(hi, f)


@FAST
@given(k=st.integers(-(2**40), 2**40), f=rates_st)
def test_rt02_prop_ties_go_up(k, f):
    assert to_ticks((Fraction(k) + Fraction(1, 2)) / f, f) == k + 1


@FAST
@given(a=small_times_st, b=small_times_st, c=small_times_st, f=rates_st)
def test_rt01_prop_counts_are_additive(a, b, c, f):
    x, y, z = sorted((a, b, c))
    assert count(x, y, f) + count(y, z, f) == count(x, z, f)
    assert count(0, z, f) == ceil_ticks(z, f) >= 0


@settings(max_examples=200, deadline=None)
@given(
    t0=st.fractions(min_value=0, max_value=10**6, max_denominator=10**6),
    width=st.fractions(min_value=0, max_value=1, max_denominator=10**6),
    f=st.builds(Fraction, st.integers(1, 2000), st.integers(1, 7)),
)
def test_rt01_prop_count_matches_enumeration(t0, width, f):
    t1 = t0 + width
    lo, hi = math.floor(t0 * f) - 1, math.ceil(t1 * f) + 1
    expected = sum(1 for k in range(lo, hi + 1) if t0 <= Fraction(k) / f < t1)
    assert count(t0, t1, f) == expected


@FAST
@given(
    f=rates_st,
    target=st.fractions(min_value=Fraction(1, 10**7), max_value=1, max_denominator=10**7),
)
def test_page_size_prop_power_of_two_nearest_in_log_domain(f, target):
    p = page_size(f, target)
    assert PAGE_MIN <= p <= PAGE_MAX
    assert p & (p - 1) == 0
    x = f * target
    m = p.bit_length() - 1
    if PAGE_MIN < p < PAGE_MAX:
        # 2**(m-1/2) <= x < 2**(m+1/2)  <=>  2**(2m-1) <= x**2 < 2**(2m+1)
        assert Fraction(2) ** (2 * m - 1) <= x * x < Fraction(2) ** (2 * m + 1)
    elif p == PAGE_MIN:
        assert x * x < Fraction(2) ** (2 * 6 + 1)
    else:
        assert x * x >= Fraction(2) ** (2 * 14 - 1)


@FAST
@given(q=st.fractions(min_value=Fraction(1, 10**9), max_value=10**12, max_denominator=10**9))
def test_rate_prop_string_round_trip(q):
    assert Rate.parse(fraction_to_str(q)) == q
    assert parse_exact(fraction_to_str(q)) == q


@FAST
@given(x=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False))
def test_to_exact_prop_uses_repr(x):
    assert to_exact(x) == Fraction(repr(x))
    with pytest.raises(TypeError):
        parse_exact(x)  # type: ignore[arg-type]


@FAST
@given(x=st.floats(width=32, allow_nan=False, allow_infinity=False))
def test_to_exact_prop_float32_shortest_decimal_round_trips(x):
    """R5 at float32 precision: the exact value is a decimal that rounds back to the same
    float32, and it is the decimal numpy prints for it (no float64 widening artefacts)."""
    v = np.float32(x)
    q = to_exact(v)
    assert np.float32(float(q)) == v
    assert q == Fraction(str(v))


@FAST
@given(full=st.integers(0, 10**6), frac=st.floats(0, 1, exclude_max=True))
def test_timespec_prop_exact_is_full_plus_repr_frac(full, frac):
    ts = TimeSpec(full, frac)
    assert ts.exact() == full + Fraction(repr(frac))
    assert TimeSpec.from_exact(ts.exact()) == ts
