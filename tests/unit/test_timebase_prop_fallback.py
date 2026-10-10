"""RT-01 / RT-02 property tests on the ``tests._prop`` seeded fallback (no hypothesis).

These run whether or not hypothesis is installed (DESIGN.md 2.4 / A5), so the fallback
path itself stays exercised. Properties are checked against definitions: the rounding
bound |k/f - t| <= 1/(2f) with ties up, exact round trips, and additivity of counts.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sremu.timebase import ceil_ticks, count, from_ticks, to_ticks
from tests import _prop

pytestmark = pytest.mark.unit


@_prop.given(n=400, k=_prop.integers(0, 2**40), f=_prop.rates())
def test_rt02_prop_fallback_round_trip(k, f):
    assert to_ticks(from_ticks(k, f), f) == k


@_prop.given(
    n=300,
    t=_prop.fractions(0, 10**6, max_denominator=10**9),
    f=_prop.sampled_from([Fraction(5880), Fraction(23_040_000), Fraction(30_720_000)]),
)
def test_rt02_prop_fallback_half_sample_bound(t, f):
    err = Fraction(to_ticks(t, f)) / f - t
    assert -Fraction(1, 2) / f < err <= Fraction(1, 2) / f


def test_prop_fallback_reports_failing_example():
    @_prop.given(n=50, k=_prop.integers(0, 100))
    def always_fails(k):
        assert k < 0

    with pytest.raises(AssertionError) as info:
        always_fails()
    assert any("_prop example #0" in note for note in getattr(info.value, "__notes__", []))


def test_prop_fallback_is_deterministic():
    seen: list[list[int]] = []
    for _ in range(2):
        drawn: list[int] = []

        @_prop.given(n=20, seed=3, k=_prop.integers(0, 2**70))
        def collect(k, drawn=drawn):
            drawn.append(k)

        collect()
        seen.append(drawn)
    assert seen[0] == seen[1]
    assert seen[0][:2] == [0, 2**70]  # boundary values first
    assert all(0 <= k <= 2**70 for k in seen[0])


@_prop.given(
    n=300,
    a=_prop.fractions(0, 1, max_denominator=10**7),
    b=_prop.fractions(0, 1, max_denominator=10**7),
    f=_prop.rates(),
)
def test_rt01_prop_fallback_counts_are_additive(a, b, f):
    lo, hi = min(a, b), max(a, b)
    assert count(0, lo, f) + count(lo, hi, f) == count(0, hi, f) == ceil_ticks(hi, f)
