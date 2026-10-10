"""Exact time base: rates, times, rounding and the page grid (DESIGN.md sections 4.1, 4.3).

Normative rules implemented here:

* **R1** A rate is an exact ``Fraction`` in S/s parsed from a string or an int
  (``Rate.parse("23.04e6") == Fraction(23040000)``). Floats raise ``TypeError`` on the
  scenario-parsing path (``parse_exact``, ``Rate.parse``).
* **R2** Virtual time is an exact ``Fraction`` of seconds; index 0 of every stream is t = 0.
* **R3** ``to_ticks(t, f) = floor(t*f + 1/2)`` (nearest, ties up; UHD ``to_ticks``) for every
  time-to-sample mapping. ``from_ticks(k, f) = Fraction(k)/f`` is exact.
* **R4** ``count(t0, t1, f) = ceil(t1*f) - ceil(t0*f)`` samples on the half-open interval
  [t0, t1), in exact integer arithmetic.
* **R5** API floats are converted with ``Fraction(repr(x))`` (``to_exact``);
  ``TimeSpec(full_secs, frac_secs).exact() = full_secs + Fraction(repr(frac_secs))``.
* **R6** Wall time never enters this module (I7).
* **Page grid** ``P_r = clamp(2**round(log2(f_r*page_target_s)), 64, 16384)``, with
  ``page_target_s = "5e-4"`` by default; page j covers absolute samples [j*P, (j+1)*P).

All arithmetic is exact (Python ints and Fractions). Functions that take an exact quantity
reject floats with ``TypeError`` instead of silently rounding them.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from functools import total_ordering
from typing import Final, TypeAlias

from sremu.errors import ConfigurationError

__all__ = [
    "DEFAULT_PAGE_TARGET_S",
    "PAGE_MAX",
    "PAGE_MIN",
    "Exact",
    "PageGrid",
    "Rate",
    "TimeLike",
    "TimeSpec",
    "ceil_ticks",
    "count",
    "fraction_to_str",
    "from_ticks",
    "iter_page_segments",
    "page_bounds",
    "page_of",
    "page_size",
    "parse_exact",
    "round_log2",
    "to_exact",
    "to_ticks",
]

Exact: TypeAlias = Fraction | int
"""An exact real quantity: a ``Fraction`` or an ``int`` (never a float)."""

DEFAULT_PAGE_TARGET_S: Final[str] = "5e-4"
"""Default page duration target in seconds (scenario ``numerics.page_target_s``)."""

PAGE_MIN: Final[int] = 64
"""Smallest page size in samples."""

PAGE_MAX: Final[int] = 16384
"""Largest page size in samples."""

_HALF: Final[Fraction] = Fraction(1, 2)


# --------------------------------------------------------------------------- parsing


def parse_exact(value: str | int | Fraction, *, name: str = "value") -> Fraction:
    """Parse an exact quantity on the **scenario path** (section 3.1, R1).

    Accepts a decimal or rational string (``"23.04e6"``, ``"1/200"``, ``"5e-3"``), an int,
    or any ``numbers.Rational`` (``Fraction``, NumPy integers). Floats (including NumPy
    floats) and bools raise ``TypeError``; malformed strings raise ``ValueError``.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name}: bool is not an exact quantity")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, numbers.Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    if isinstance(value, str):
        text = value.strip()
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"{name}: cannot parse {value!r} as an exact quantity") from exc
    if isinstance(value, numbers.Real):
        raise TypeError(
            f"{name}: floats are rejected for exact quantities ({value!r}); "
            f"write the value as a string, e.g. {str(value)!r}"
        )
    raise TypeError(f"{name}: unsupported type {type(value).__name__} for an exact quantity")


def to_exact(value: float | str | int | Fraction, *, name: str = "value") -> Fraction:
    """Convert a value on the **API path** to an exact ``Fraction`` (R5).

    Floats are converted with ``Fraction(repr(x))`` (the shortest decimal that round-trips),
    so ``0.5e-3`` becomes exactly ``1/2000``. Other real floating types (``np.float32``,
    ``np.float16``, ``np.longdouble``) use the shortest round-trip decimal of their **own**
    precision (``str(x)``), never the binary expansion of a widened float:
    ``np.float32(0.9)`` is ``9/10``, not ``0.899999976...``, which would land one sample
    early at 23.04 MS/s. Non-finite values raise ``ValueError``. Everything else is handled
    by :func:`parse_exact`.
    """
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Rational | bool):
        x = float(value)
        if not math.isfinite(x):
            raise ValueError(f"{name}: non-finite value {value!r}")
        if isinstance(value, float):  # Python float, or np.float64 (a float subclass)
            return Fraction(repr(x))
        try:
            return Fraction(str(value).strip())
        except (ValueError, ZeroDivisionError):
            return Fraction(repr(x))
    return parse_exact(value, name=name)  # type: ignore[arg-type]


def fraction_to_str(x: Fraction | int) -> str:
    """Canonical exact text form used in JSON: ``"p/q"``, or ``"p"`` for integers."""
    q = parse_exact(x)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


# --------------------------------------------------------------------------- Rate


class Rate(Fraction):
    """An exact, strictly positive sample rate in S/s (R1).

    ``Rate("23.04e6") == Rate(23040000) == Fraction(23040000)``. Construction follows the
    scenario path: floats raise ``TypeError``. Use :meth:`from_api` to accept an API float
    via ``Fraction(repr(x))``. Arithmetic on a ``Rate`` returns plain ``Fraction`` values.
    """

    __slots__ = ()

    def __new__(
        cls,
        numerator: str | int | Fraction = 0,
        denominator: int | Fraction | None = None,
    ) -> Rate:
        value = parse_exact(numerator, name="rate")
        if denominator is not None:
            den = parse_exact(denominator, name="rate denominator")
            if den == 0:
                raise ZeroDivisionError("rate denominator is zero")
            value = value / den
        if value <= 0:
            raise ConfigurationError(f"sample rate must be > 0 S/s, got {value}")
        return super().__new__(cls, value.numerator, value.denominator)

    @classmethod
    def parse(cls, value: str | int | Fraction) -> Rate:
        """Parse a rate on the scenario path (floats raise ``TypeError``)."""
        return cls(value)

    @classmethod
    def from_api(cls, value: float | str | int | Fraction) -> Rate:
        """Convert an API value; floats go through ``Fraction(repr(x))`` (R5)."""
        return cls(to_exact(value, name="rate"))


# --------------------------------------------------------------------------- TimeSpec


@total_ordering
@dataclass(frozen=True, eq=False, slots=True)
class TimeSpec:
    """UHD-shaped virtual time in seconds since the emulator epoch (R2, R5).

    ``full_secs`` is an integer number of seconds. ``frac_secs`` is a float (converted with
    ``Fraction(repr(x))``), an int, an exact ``Fraction``, or an exact string. The value
    is ``exact() = full_secs + frac_secs`` as a ``Fraction``; comparison and hashing use it,
    so ``TimeSpec(0, 0.5) == TimeSpec.from_exact(Fraction(1, 2))``.
    """

    full_secs: int = 0
    frac_secs: float | Fraction | int | str = 0.0
    _exact: Fraction = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.full_secs, bool) or not isinstance(self.full_secs, numbers.Integral):
            raise TypeError(
                f"TimeSpec.full_secs must be an int, got {type(self.full_secs).__name__}; "
                "use TimeSpec.from_secs(x) for a single float"
            )
        frac = to_exact(self.frac_secs, name="TimeSpec.frac_secs")
        object.__setattr__(self, "_exact", Fraction(int(self.full_secs)) + frac)

    def exact(self) -> Fraction:
        """Exact time in seconds."""
        return self._exact

    def real_secs(self) -> float:
        """Time in seconds as a float (display only; never used for sample arithmetic)."""
        return float(self._exact)

    def to_ticks(self, rate: Exact) -> int:
        """Nearest sample index at ``rate`` (R3)."""
        return to_ticks(self._exact, rate)

    @classmethod
    def from_exact(cls, t: str | int | Fraction) -> TimeSpec:
        """Build from an exact time; ``frac_secs`` is stored as an exact ``Fraction``."""
        q = parse_exact(t, name="time")
        full = math.floor(q)
        return cls(full, q - full)

    @classmethod
    def from_secs(cls, secs: float | str | int | Fraction) -> TimeSpec:
        """Build from seconds given on the API path (a float goes through repr, R5)."""
        return cls.from_exact(to_exact(secs, name="time"))

    @classmethod
    def from_ticks(cls, k: int, rate: Exact) -> TimeSpec:
        """Exact time of sample ``k`` at ``rate``; ``from_ticks(k, f).to_ticks(f) == k``."""
        return cls.from_exact(from_ticks(k, rate))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSpec):
            return NotImplemented
        return self._exact == other._exact

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TimeSpec):
            return NotImplemented
        return self._exact < other._exact

    def __hash__(self) -> int:
        return hash(self._exact)

    def __add__(self, other: object) -> TimeSpec:
        if isinstance(other, TimeSpec):
            return TimeSpec.from_exact(self._exact + other._exact)
        if isinstance(other, Fraction | int) and not isinstance(other, bool):
            return TimeSpec.from_exact(self._exact + other)
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: object) -> TimeSpec:
        if isinstance(other, TimeSpec):
            return TimeSpec.from_exact(self._exact - other._exact)
        if isinstance(other, Fraction | int) and not isinstance(other, bool):
            return TimeSpec.from_exact(self._exact - other)
        return NotImplemented


TimeLike: TypeAlias = Fraction | int | TimeSpec
"""An exact time in seconds: ``Fraction``, ``int`` or ``TimeSpec``."""


# --------------------------------------------------------------------------- rounding


def _time_parts(t: TimeLike, name: str) -> tuple[int, int]:
    """(numerator, denominator > 0) of an exact time; floats raise ``TypeError``."""
    if isinstance(t, TimeSpec):
        q = t.exact()
        return q.numerator, q.denominator
    if isinstance(t, bool):
        raise TypeError(f"{name}: bool is not a time")
    if isinstance(t, Fraction):
        return t.numerator, t.denominator
    if isinstance(t, numbers.Integral):
        return int(t), 1
    if isinstance(t, numbers.Rational):
        return int(t.numerator), int(t.denominator)
    raise TypeError(
        f"{name} must be exact (Fraction, int or TimeSpec), got {type(t).__name__}; "
        "convert API floats with to_exact() first"
    )


def _rate_parts(f: Exact) -> tuple[int, int]:
    """(numerator > 0, denominator > 0) of an exact positive rate."""
    if isinstance(f, bool):
        raise TypeError("rate: bool is not a rate")
    if isinstance(f, Fraction):
        n, d = f.numerator, f.denominator
    elif isinstance(f, numbers.Integral):
        n, d = int(f), 1
    elif isinstance(f, numbers.Rational):
        n, d = int(f.numerator), int(f.denominator)
    else:
        raise TypeError(
            f"rate must be exact (Fraction or int), got {type(f).__name__}; "
            "use Rate.parse() or Rate.from_api()"
        )
    if n <= 0:
        raise ConfigurationError(f"sample rate must be > 0 S/s, got {f}")
    return n, d


def to_ticks(t: TimeLike, f: Exact) -> int:
    """Nearest sample index of time ``t`` (s) at rate ``f`` (S/s): ``floor(t*f + 1/2)`` (R3).

    Ties go up, as UHD ``time_spec_t::to_ticks``. The worst-case application error is
    ``1/(2f)`` seconds.
    """
    tn, td = _time_parts(t, "t")
    fn, fd = _rate_parts(f)
    n = tn * fn
    d = td * fd
    return (2 * n + d) // (2 * d)


def from_ticks(k: int, f: Exact) -> Fraction:
    """Exact time in seconds of sample index ``k`` at rate ``f``: ``Fraction(k)/f`` (R3)."""
    if isinstance(k, bool) or not isinstance(k, numbers.Integral):
        raise TypeError(f"sample index must be an int, got {type(k).__name__}")
    fn, fd = _rate_parts(f)
    return Fraction(int(k) * fd, fn)


def ceil_ticks(t: TimeLike, f: Exact) -> int:
    """``ceil(t*f)``: the first sample index at or after time ``t``.

    For ``t >= 0`` this is also the number of samples a stream at rate ``f`` has produced
    on [0, t) (invariant I1).
    """
    tn, td = _time_parts(t, "t")
    fn, fd = _rate_parts(f)
    return -((-tn * fn) // (td * fd))


def count(t0: TimeLike, t1: TimeLike, f: Exact) -> int:
    """Number of samples at rate ``f`` with time in the half-open interval [t0, t1) (R4).

    ``count(t0, t1, f) = ceil(t1*f) - ceil(t0*f)``, exact. Raises ``ValueError`` when
    ``t1 < t0``. Counts are additive: ``count(a, b) + count(b, c) == count(a, c)``.
    """
    n0, d0 = _time_parts(t0, "t0")
    n1, d1 = _time_parts(t1, "t1")
    if n1 * d0 < n0 * d1:
        raise ValueError(f"count: t1 < t0 ({Fraction(n1, d1)} < {Fraction(n0, d0)})")
    fn, fd = _rate_parts(f)
    return (-((-n1 * fn) // (d1 * fd))) - (-((-n0 * fn) // (d0 * fd)))


# --------------------------------------------------------------------------- page grid


def round_log2(x: Fraction | int) -> int:
    """``round(log2(x))`` for an exact ``x > 0``, computed without floating point.

    Returns m with ``2**(m - 1/2) <= x < 2**(m + 1/2)``. An exact tie would need
    ``x = 2**m * sqrt(2)``, which is irrational, so the result is unambiguous.
    """
    q = parse_exact(x, name="x")
    if q <= 0:
        raise ValueError(f"round_log2 needs x > 0, got {q}")
    n, d = q.numerator, q.denominator
    e = n.bit_length() - d.bit_length()  # floor(log2 x) is e or e - 1
    if (e >= 0 and n < (d << e)) or (e < 0 and (n << -e) < d):
        e -= 1
    # now 2**e <= x < 2**(e+1); round up iff x**2 >= 2**(2e+1)
    if 2 * e + 1 >= 0:
        up = n * n >= (d * d) << (2 * e + 1)
    else:
        up = (n * n) << -(2 * e + 1) >= d * d
    return e + 1 if up else e


def page_size(f: Exact, page_target_s: str | int | Fraction = DEFAULT_PAGE_TARGET_S) -> int:
    """Page size in samples: ``clamp(2**round(log2(f*page_target_s)), 64, 16384)`` (4.3).

    Worked examples with the default target 5e-4 s: 23.04 MS/s -> 8192, 30.72 MS/s ->
    16384, 1 MS/s -> 512, 10 kS/s -> 64, 5880 S/s -> 64.
    """
    fn, fd = _rate_parts(f)
    target = parse_exact(page_target_s, name="page_target_s")
    if target <= 0:
        raise ConfigurationError(f"page_target_s must be > 0, got {target}")
    m = round_log2(Fraction(fn, fd) * target)
    if m <= 6:
        return PAGE_MIN
    return min(1 << m, PAGE_MAX)


def page_of(k: int, page: int) -> int:
    """Index of the page containing absolute sample ``k`` (``k // page``)."""
    return int(k) // page


def page_bounds(j: int, page: int) -> tuple[int, int]:
    """Absolute sample range [j*page, (j+1)*page) of page ``j``."""
    return j * page, (j + 1) * page


def iter_page_segments(k0: int, k1: int, page: int) -> Iterator[tuple[int, int, int]]:
    """Split the absolute range [k0, k1) at page boundaries; yields ``(j, a, b)``."""
    if k1 < k0:
        raise ValueError(f"iter_page_segments: k1 < k0 ({k1} < {k0})")
    a = k0
    while a < k1:
        j = a // page
        b = min(k1, (j + 1) * page)
        yield j, a, b
        a = b


@dataclass(frozen=True, slots=True)
class PageGrid:
    """The absolute page grid of one RX stream (or rate group): ``rate`` in S/s, ``page``."""

    rate: Fraction
    page: int

    @classmethod
    def for_rate(
        cls, rate: Exact, page_target_s: str | int | Fraction = DEFAULT_PAGE_TARGET_S
    ) -> PageGrid:
        """Grid with the page size given by :func:`page_size`."""
        fn, fd = _rate_parts(rate)
        return cls(Fraction(fn, fd), page_size(rate, page_target_s))

    def page_of(self, k: int) -> int:
        """Page index of absolute sample ``k``."""
        return page_of(k, self.page)

    def bounds(self, j: int) -> tuple[int, int]:
        """Absolute sample range of page ``j``."""
        return page_bounds(j, self.page)

    def segments(self, k0: int, k1: int) -> Iterator[tuple[int, int, int]]:
        """Split [k0, k1) at page boundaries; yields ``(j, a, b)``."""
        return iter_page_segments(k0, k1, self.page)

    def page_duration_s(self) -> Fraction:
        """Exact page duration in seconds."""
        return Fraction(self.page) / self.rate
