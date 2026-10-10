"""Seeded property-loop fallback for environments without hypothesis (WP0).

hypothesis is installed in the project venv (DESIGN.md A5) and tests may use it directly.
This module keeps property tests runnable without it: ``given`` runs the test body ``n``
times with values drawn from keyed strategies. Boundary values are tried first; every
other draw comes from a Philox generator keyed by the test name and ``seed`` (never the
global RNG). A failing example is attached to the assertion as a note.

Example::

    from tests._prop import given, integers, rates

    @given(n=300, k=integers(0, 2**40), f=rates())
    def test_round_trip(k, f):
        assert to_ticks(from_ticks(k, f), f) == k
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import itertools
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

try:
    import hypothesis  # noqa: F401

    HAVE_HYPOTHESIS = True
except ImportError:  # pragma: no cover - hypothesis is installed in the project venv
    HAVE_HYPOTHESIS = False

__all__ = [
    "HAVE_HYPOTHESIS",
    "Strategy",
    "booleans",
    "floats",
    "fractions",
    "given",
    "integers",
    "rates",
    "sampled_from",
]


class Strategy:
    """A value generator: ``draw(rng)`` plus boundary ``edges`` tried first."""

    def __init__(self, draw: Callable[[np.random.Generator], Any], edges: Sequence[Any] = ()):
        self._draw = draw
        self.edges = tuple(edges)

    def draw(self, rng: np.random.Generator) -> Any:
        """Draw one value."""
        return self._draw(rng)

    def map(self, f: Callable[[Any], Any]) -> Strategy:
        """Strategy of ``f(value)``."""
        return Strategy(lambda rng: f(self._draw(rng)), [f(e) for e in self.edges])


def _uniform_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] for arbitrarily large Python ints (rejection sampling)."""
    span = hi - lo
    if span < 0:
        raise ValueError("empty integer range")
    if span < 2**62:
        return lo + int(rng.integers(0, span + 1))
    nbits = span.bit_length()
    nwords = (nbits + 31) // 32
    while True:
        words = rng.integers(0, 2**32, size=nwords, dtype=np.uint64)
        v = 0
        for w in words:
            v = (v << 32) | int(w)
        v >>= nwords * 32 - nbits
        if v <= span:
            return lo + v


def integers(min_value: int, max_value: int) -> Strategy:
    """Integers in [min_value, max_value] (inclusive; any size)."""
    edges = sorted({min_value, max_value, min(max(0, min_value), max_value)})
    return Strategy(lambda rng: _uniform_int(rng, min_value, max_value), edges)


def booleans() -> Strategy:
    """True or False."""
    return Strategy(lambda rng: bool(rng.integers(0, 2)), [False, True])


def floats(min_value: float, max_value: float) -> Strategy:
    """Uniform floats in [min_value, max_value]."""
    return Strategy(lambda rng: float(rng.uniform(min_value, max_value)), [min_value, max_value])


def sampled_from(values: Sequence[Any]) -> Strategy:
    """One of ``values``."""
    vals = list(values)
    return Strategy(lambda rng: vals[int(rng.integers(0, len(vals)))], vals[:1])


def fractions(
    min_value: Fraction | int, max_value: Fraction | int, max_denominator: int = 10**6
) -> Strategy:
    """Exact fractions in [min_value, max_value] with denominators <= ``max_denominator``."""
    lo, hi = Fraction(min_value), Fraction(max_value)

    def draw(rng: np.random.Generator) -> Fraction:
        d = _uniform_int(rng, 1, max_denominator)
        n_lo = -((-lo.numerator * d) // lo.denominator)  # ceil(lo*d)
        n_hi = (hi.numerator * d) // hi.denominator  # floor(hi*d)
        if n_hi < n_lo:
            return lo
        return Fraction(_uniform_int(rng, n_lo, n_hi), d)

    return Strategy(draw, [lo, hi])


def rates(max_numerator: int = 10**9, max_denominator: int = 1000) -> Strategy:
    """Random positive rational sample rates p/q in S/s."""

    def draw(rng: np.random.Generator) -> Fraction:
        return Fraction(_uniform_int(rng, 1, max_numerator), _uniform_int(rng, 1, max_denominator))

    return Strategy(draw, [Fraction(5880), Fraction(23_040_000), Fraction(30_720_000)])


def given(*, n: int = 200, seed: int | str = 0, **strategies: Strategy) -> Callable[..., Any]:
    """Run the decorated test ``n`` times with keyword arguments drawn from ``strategies``."""

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        key_src = f"{fn.__module__}.{fn.__qualname__}/{seed}".encode()
        key = int.from_bytes(hashlib.blake2b(key_src, digest_size=8).digest(), "little")
        names = list(strategies)
        edge_cases = list(itertools.product(*(strategies[k].edges or (None,) for k in names)))

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rng = np.random.Generator(np.random.Philox(key=key))
            for i in range(n):
                if i < len(edge_cases) and all(v is not None for v in edge_cases[i]):
                    example = dict(zip(names, edge_cases[i], strict=True))
                else:
                    example = {k: strategies[k].draw(rng) for k in names}
                try:
                    fn(*args, **kwargs, **example)
                except AssertionError as exc:
                    exc.add_note(f"_prop example #{i}: {example!r}")
                    raise

        sig = inspect.signature(fn)
        wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=[p for p in sig.parameters.values() if p.name not in strategies]
        )
        return wrapper

    return decorate
