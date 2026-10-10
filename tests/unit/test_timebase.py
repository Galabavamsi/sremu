"""RT-01 (I1) and RT-02 (R3/R5) exact time-base oracles, plus the page grid (section 4.3).

Oracle values are closed forms: hand-computed integer sample counts, brute-force
enumeration of the samples inside an interval, and the worked examples of DESIGN.md
sections 4.1, 4.3 and Appendix A.
"""

from __future__ import annotations

import itertools
import math
import pickle
from fractions import Fraction

import numpy as np
import pytest

from sremu.errors import ConfigurationError
from sremu.timebase import (
    DEFAULT_PAGE_TARGET_S,
    PAGE_MAX,
    PAGE_MIN,
    PageGrid,
    Rate,
    TimeSpec,
    ceil_ticks,
    count,
    fraction_to_str,
    from_ticks,
    iter_page_segments,
    page_size,
    parse_exact,
    round_log2,
    to_exact,
    to_ticks,
)

pytestmark = pytest.mark.unit

RATE_STRINGS = ("5880", "23.04e6", "30.72e6")
RATES = tuple(Rate.parse(s) for s in RATE_STRINGS)

# Hand-computed ceil(T*f) for T in {1 s, 7.3 ms, 1/3 s} at (5880, 23.04e6, 30.72e6) S/s.
#   7.3 ms: 5880*0.0073 = 42.924 -> 43; 23.04e6*0.0073 = 168192; 30.72e6*0.0073 = 224256.
#   1/3 s : 1960; 7680000; 10240000 (all exact).
EXPECTED_COUNTS = {
    "1": (5880, 23_040_000, 30_720_000),
    "7.3e-3": (43, 168_192, 224_256),
    "1/3": (1960, 7_680_000, 10_240_000),
}


def _brute_count(t0: Fraction, t1: Fraction, f: Fraction) -> int:
    """Independent oracle: enumerate sample indices k with t0 <= k/f < t1."""
    lo = math.floor(t0 * f) - 2
    hi = math.ceil(t1 * f) + 2
    return sum(1 for k in range(lo, hi + 1) if t0 <= Fraction(k) / f < t1)


# --------------------------------------------------------------------------- parsing / Rate


def test_rate_parse_exact_strings():
    assert Rate.parse("23.04e6") == Fraction(23_040_000)
    assert Rate.parse("30.72e6") == Fraction(30_720_000)
    assert Rate.parse("5880") == Fraction(5880)
    assert Rate.parse(" 1e6 ") == Fraction(10**6)
    assert Rate(10**6, 3) == Fraction(10**6, 3)
    assert Rate.parse("1000000/3") == Fraction(10**6, 3)


def test_rate_rejects_floats_on_scenario_path_and_accepts_them_on_api_path():
    with pytest.raises(TypeError):
        Rate.parse(23.04e6)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Rate(np.float64(1e6))  # type: ignore[arg-type]
    assert Rate.from_api(23.04e6) == Fraction(23_040_000)
    assert Rate.from_api(np.float32(5880.0)) == Fraction(5880)


@pytest.mark.parametrize("bad", ["0", "-1e6", 0, -5])
def test_rate_must_be_positive(bad):
    with pytest.raises(ConfigurationError):
        Rate(bad)


def test_rate_rejects_malformed_and_bool():
    with pytest.raises(ValueError):
        Rate.parse("23.04 MHz")
    with pytest.raises(ValueError):
        Rate.parse("nan")
    with pytest.raises(TypeError):
        Rate.parse(True)  # type: ignore[arg-type]


def test_rate_pickles_and_is_a_fraction():
    r = Rate.parse("23.04e6")
    r2 = pickle.loads(pickle.dumps(r))
    assert isinstance(r2, Rate)
    assert r2 == r
    assert hash(r) == hash(Fraction(23_040_000))
    assert isinstance(r * 2, Fraction)


def test_parse_exact_and_to_exact():
    assert parse_exact("1/200") == Fraction(1, 200)
    assert parse_exact("5e-3") == Fraction(1, 200)
    assert parse_exact(np.int64(7)) == Fraction(7)
    with pytest.raises(TypeError):
        parse_exact(5e-3)  # type: ignore[arg-type]
    # API floats go through repr: 0.5e-3 is exactly 1/2000, not the binary expansion
    assert to_exact(0.5e-3) == Fraction(1, 2000)
    assert to_exact(np.float64(0.5e-3)) == Fraction(1, 2000)
    assert to_exact(0.5e-3) != Fraction(0.5e-3)
    with pytest.raises(ValueError):
        to_exact(float("inf"))


def test_fraction_to_str_round_trips():
    for q in (Fraction(1, 200), Fraction(23_040_000), Fraction(-7, 3), Fraction(0)):
        assert parse_exact(fraction_to_str(q)) == q
    assert fraction_to_str(Fraction(1, 200)) == "1/200"
    assert fraction_to_str(Fraction(5)) == "5"


# --------------------------------------------------------------------------- RT-01 (I1)


@pytest.mark.parametrize("t_str", list(EXPECTED_COUNTS))
def test_rt01_exact_counts_three_rates_together(t_str):
    t_end = parse_exact(t_str)
    for f, expected in zip(RATES, EXPECTED_COUNTS[t_str], strict=True):
        assert count(0, t_end, f) == expected
        assert ceil_ticks(t_end, f) == expected


def test_rt01_one_million_awkward_epochs_sum_exactly():
    """[0, 1 s) split at 10**6 - 1 random cut points with an awkward common denominator
    (7 * 9973 * 1000003); zero-length epochs occur. All three rates are counted over the
    same epochs and must total exactly ceil(1 s * f)."""
    den = 7 * 9973 * 1_000_003
    gen = np.random.Generator(np.random.Philox(key=20261010))
    cuts = np.sort(gen.integers(1, den, size=10**6 - 1, dtype=np.int64))
    nums = [0, *cuts.tolist(), den]
    bounds = [Fraction(n, den) for n in nums]
    for f, expected in zip(RATES, EXPECTED_COUNTS["1"], strict=True):
        total = 0
        for a, b in itertools.pairwise(bounds):
            total += count(a, b, f)
        assert total == expected


@pytest.mark.parametrize("t_str", ["7.3e-3", "1/3"])
def test_rt01_awkward_epochs_other_durations(t_str):
    t_end = parse_exact(t_str)
    den = t_end.denominator * 9973 * 30011
    gen = np.random.Generator(np.random.Philox(key=7))
    n_end = t_end.numerator * (den // t_end.denominator)
    cuts = np.sort(gen.integers(1, n_end, size=10**5 - 1, dtype=np.int64))
    bounds = [Fraction(n, den) for n in [0, *cuts.tolist(), n_end]]
    assert bounds[-1] == t_end
    for f, expected in zip(RATES, EXPECTED_COUNTS[t_str], strict=True):
        assert sum(count(a, b, f) for a, b in itertools.pairwise(bounds)) == expected


def test_rt01_count_matches_brute_force_enumeration_including_large_epochs():
    gen = np.random.Generator(np.random.Philox(key=11))
    for f in RATES:
        for base in (Fraction(0), Fraction(10**6), Fraction(10**6) + Fraction(1, 3)):
            for _ in range(150):
                a = base + Fraction(int(gen.integers(0, 10**7)), 9973 * 30011)
                b = a + Fraction(int(gen.integers(0, 10**5)), 1_000_003)
                # restrict the enumeration cost at the high rates
                if (b - a) * f > 600:
                    b = a + Fraction(600) / f + Fraction(1, 7 * f)
                assert count(a, b, f) == _brute_count(a, b, f)


def test_rt01_counts_with_integer_samples_on_boundaries():
    f = Fraction(5880)
    # t0 exactly on sample 3, t1 exactly on sample 10: [3, 10) holds 7 samples
    assert count(Fraction(3, 5880), Fraction(10, 5880), f) == 7
    assert count(Fraction(3, 5880), Fraction(3, 5880), f) == 0
    with pytest.raises(ValueError):
        count(Fraction(1), Fraction(0), f)


def test_rt01_teeth_truncation_rule_gives_ants_fig2_slope():
    """Legacy ``int(f*tau)`` per 2 ms epoch at 5880 S/s yields 11 samples instead of 11.76 on
    average: the ANTS Fig. 2 slope 11.76/11 = 1.06909 (section 4.1 R4, FACT)."""
    f = Rate.parse("5880")
    tau = parse_exact("2e-3")
    epochs = 25
    exact = sum(count(i * tau, (i + 1) * tau, f) for i in range(epochs))
    legacy = sum(int(5880 * 0.002) for _ in range(epochs))
    assert exact == 294 == ceil_ticks(epochs * tau, f)
    assert legacy == 275
    assert exact / legacy == pytest.approx(11.76 / 11, abs=5e-4)


def test_rt01_count_rejects_floats():
    with pytest.raises(TypeError):
        count(0.0, 1.0, Fraction(5880))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        count(Fraction(0), Fraction(1), 5880.0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- RT-02 (R3/R5)

_K_EDGES = (0, 1, 2, 3, 2**20, 2**31 - 1, 2**31, 2**32, 2**39, 2**40 - 1, 2**40)


@pytest.mark.parametrize("f", RATES, ids=RATE_STRINGS)
def test_rt02_round_trip_up_to_2_pow_40(f):
    gen = np.random.Generator(np.random.Philox(key=int(f)))
    ks = list(_K_EDGES) + [int(k) for k in gen.integers(0, 2**40 + 1, size=20_000, dtype=np.int64)]
    for k in ks:
        assert to_ticks(from_ticks(k, f), f) == k
    for k in _K_EDGES:
        assert TimeSpec.from_ticks(k, f).to_ticks(f) == k
        assert TimeSpec.from_ticks(k, f).exact() == Fraction(k) / f


def test_rt02_float_half_ms_maps_to_11520_not_11521():
    f = Rate.parse("23.04e6")
    assert TimeSpec(0, 0.5e-3).to_ticks(f) == 11520
    assert to_ticks(to_exact(0.5e-3), f) == 11520
    # the binary value of the float lies slightly above 1/2000 ...
    exact_binary = Fraction(0.5e-3)
    assert exact_binary > Fraction(1, 2000)
    # ... so ceil would give 11521 (rejected rule) while nearest still gives 11520
    assert math.ceil(exact_binary * 23_040_000) == 11521
    assert to_ticks(exact_binary, f) == 11520


def test_rt02_float32_api_values_use_their_own_shortest_decimal():
    """R5 for NumPy float32/float16 API values: the decimal the user wrote, not the binary
    expansion of the widened float64. np.float32(0.9) s at 23.04 MS/s is sample
    0.9 * 23040000 = 20736000 exactly; widening first gives 0.89999997615... s, which lands
    on 20735999 (one sample early)."""
    f = Rate.parse("23.04e6")
    assert to_exact(np.float32(0.9)) == Fraction(9, 10)
    assert TimeSpec(0, np.float32(0.9)).to_ticks(f) == 20_736_000
    assert to_ticks(Fraction(repr(float(np.float32(0.9)))), f) == 20_735_999  # widened: wrong
    assert to_exact(np.float16(0.1)) == Fraction(1, 10)
    assert to_exact(np.float32(1e-10)) == Fraction(1, 10**10)
    assert to_exact(np.float64(0.9)) == Fraction(9, 10)
    with pytest.raises(ValueError):
        to_exact(np.float32("nan"))


def test_rt02_nearest_ties_up_and_error_bound():
    f = Fraction(23_040_000)
    for k in (0, 1, 11520, 2**40):
        assert to_ticks((Fraction(k) + Fraction(1, 2)) / f, f) == k + 1  # tie goes up
        assert to_ticks((Fraction(k) + Fraction(1, 2)) / f - Fraction(1, 10**30), f) == k
        assert to_ticks((Fraction(k) - Fraction(1, 2)) / f, f) == k
    t = Fraction(12345, 678901)
    assert abs(Fraction(to_ticks(t, f)) / f - t) <= Fraction(1, 2) / f


def test_rt02_negative_times_round_consistently():
    f = Fraction(1000)
    assert to_ticks(Fraction(-1, 2000), f) == 0  # -0.5 samples: tie goes up
    assert to_ticks(Fraction(-3, 2000), f) == -1  # -1.5 -> -1
    assert ceil_ticks(Fraction(-1, 2000), f) == 0


def test_rt02_to_ticks_rejects_floats_and_bad_rates():
    with pytest.raises(TypeError):
        to_ticks(0.5e-3, Fraction(23_040_000))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        from_ticks(1.0, Fraction(5880))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        to_ticks(Fraction(1), Fraction(0))


# --------------------------------------------------------------------------- TimeSpec


def test_timespec_exact_follows_r5():
    assert TimeSpec(0, 1e-3).exact() == Fraction(1, 1000)
    assert TimeSpec(1, 0.0).exact() == Fraction(1)
    assert TimeSpec(2, "1/3").exact() == Fraction(7, 3)
    assert TimeSpec(0, Fraction(1, 7)).exact() == Fraction(1, 7)
    assert TimeSpec.from_secs(2.5e-3).exact() == Fraction(1, 400)
    assert TimeSpec.from_exact("7/3") == TimeSpec(2, Fraction(1, 3))


def test_timespec_equality_hash_ordering_arithmetic():
    a = TimeSpec(0, 0.5)
    b = TimeSpec.from_exact(Fraction(1, 2))
    assert a == b
    assert hash(a) == hash(b)
    assert TimeSpec(0, 0.25) < a <= b < TimeSpec(1)
    assert (a + b) == TimeSpec(1)
    assert (a + Fraction(1, 4)).exact() == Fraction(3, 4)
    assert (TimeSpec(1) - a).exact() == Fraction(1, 2)
    assert (a - TimeSpec(1)).exact() == Fraction(-1, 2)


def test_timespec_rejects_float_full_secs_and_nonfinite():
    with pytest.raises(TypeError):
        TimeSpec(1e-3)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TimeSpec(0, float("nan"))


# --------------------------------------------------------------------------- page grid

PAGE_EXAMPLES = [("23.04e6", 8192), ("30.72e6", 16384), ("1e6", 512), ("10e3", 64), ("5880", 64)]


@pytest.mark.parametrize(("rate", "expected"), PAGE_EXAMPLES)
def test_page_size_worked_examples(rate, expected):
    assert page_size(Rate.parse(rate)) == expected
    assert page_size(Rate.parse(rate), DEFAULT_PAGE_TARGET_S) == expected


@pytest.mark.parametrize("target", ["2.5e-4", "5e-4", "1e-3", "1/7", "3e-6"])
@pytest.mark.parametrize("rate", ["5880", "1e6", "10e6", "23.04e6", "30.72e6", "61.44e6"])
def test_page_size_matches_float_formula_away_from_ties(rate, target):
    x = float(Rate.parse(rate) * parse_exact(target))
    lg = math.log2(x)
    assert abs((lg - math.floor(lg)) - 0.5) > 1e-6  # these cases are far from a tie
    expected = min(max(2 ** round(lg), PAGE_MIN), PAGE_MAX)
    assert page_size(Rate.parse(rate), target) == expected


def test_round_log2_exact_near_ties():
    # 2**(m + 1/2) is irrational, so rationals just below/above it round down/up
    for m in range(-20, 40):
        lo = Fraction(math.isqrt(2 ** (2 * m + 1) * 10**40 if m >= 0 else 2 * 10**40), 10**20)
        if m < 0:
            lo = lo / 2 ** (-m)
        # lo <= 2**(m+1/2) < lo + 1e-20*scale
        assert round_log2(lo) == m
        assert round_log2(lo * (1 + Fraction(1, 10**15))) == m + 1
    assert round_log2(Fraction(1)) == 0
    assert round_log2(Fraction(1, 3)) == -2  # log2(1/3) = -1.585
    with pytest.raises(ValueError):
        round_log2(Fraction(0))


def test_page_grid_segments_cover_range():
    grid = PageGrid.for_rate(Rate.parse("1e6"))
    assert grid.page == 512
    assert grid.page_duration_s() == Fraction(512, 10**6)
    segs = list(grid.segments(1000, 5000))
    assert segs[0][1] == 1000
    assert segs[-1][2] == 5000
    for (j, _a, b), (_, a2, _) in itertools.pairwise(segs):
        assert b == a2 == (j + 1) * 512
    assert all(grid.page_of(a) == j == grid.page_of(b - 1) for j, a, b in segs)
    assert grid.bounds(3) == (1536, 2048)
    assert list(iter_page_segments(7, 7, 64)) == []
