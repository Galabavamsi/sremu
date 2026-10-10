"""Log-linear latency histogram (DESIGN.md section 12.1).

Page latencies, command latencies and stage times are recorded in a log-linear histogram
with 1 us resolution (the section 12.1 "log-linear histogram, 1 us resolution"):

* values are integers in a base unit (default 1 ns); values below ``resolution`` (default
  1000 = 1 us) fall into bucket 0;
* above that, each power-of-two range ``[2**e, 2**(e+1))`` (in resolution units) is split
  into ``sub_buckets`` equal linear buckets (default 64, so the relative bucket width is at
  most 1/64 = 1.6 %);
* ``percentile(p)`` returns the upper edge of the bucket that holds the p-th percentile
  (nearest-rank definition), clipped to the exact recorded maximum; ``max`` and ``min``
  are exact.

The histogram never reads a clock: callers pass durations.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Final

__all__ = ["DEFAULT_RESOLUTION_NS", "DEFAULT_SUB_BUCKETS", "LogLinearHistogram"]

DEFAULT_RESOLUTION_NS: Final[int] = 1000
"""Resolution of the default histogram: 1 us in ns."""

DEFAULT_SUB_BUCKETS: Final[int] = 64
"""Linear buckets per power of two."""


class LogLinearHistogram:
    """Log-linear histogram of non-negative integer values (see the module docstring)."""

    def __init__(
        self, resolution: int = DEFAULT_RESOLUTION_NS, sub_buckets: int = DEFAULT_SUB_BUCKETS
    ) -> None:
        if resolution < 1:
            raise ValueError(f"resolution must be >= 1, got {resolution}")
        if sub_buckets < 1 or sub_buckets & (sub_buckets - 1):
            raise ValueError(f"sub_buckets must be a power of two, got {sub_buckets}")
        self.resolution = int(resolution)
        self.sub_buckets = int(sub_buckets)
        self._sub_bits = self.sub_buckets.bit_length() - 1
        self._counts: dict[int, int] = {}
        self.count = 0
        self.total = 0
        self.min: int | None = None
        self.max: int | None = None

    # ------------------------------------------------------------------ bucketing
    def bucket_of(self, value: int) -> int:
        """Bucket index of ``value`` (0 for values below one resolution unit)."""
        u = int(value) // self.resolution
        if u < self.sub_buckets:
            return u
        e = u.bit_length() - 1  # u in [2**e, 2**(e+1)), e >= sub_bits
        shift = e - self._sub_bits
        return (shift + 1) * self.sub_buckets + ((u >> shift) - self.sub_buckets)

    def bucket_bounds(self, b: int) -> tuple[int, int]:
        """Value range ``[lo, hi)`` of bucket ``b`` in base units."""
        if b < self.sub_buckets:
            return b * self.resolution, (b + 1) * self.resolution
        shift = b // self.sub_buckets - 1
        m = b % self.sub_buckets + self.sub_buckets
        return (m << shift) * self.resolution, ((m + 1) << shift) * self.resolution

    # ------------------------------------------------------------------ recording
    def record(self, value: int, n: int = 1) -> None:
        """Record ``value`` (>= 0, base units) ``n`` times."""
        v = int(value)
        if v < 0:
            raise ValueError(f"histogram values must be >= 0, got {v}")
        if n <= 0:
            return
        b = self.bucket_of(v)
        self._counts[b] = self._counts.get(b, 0) + n
        self.count += n
        self.total += v * n
        self.min = v if self.min is None else min(self.min, v)
        self.max = v if self.max is None else max(self.max, v)

    def record_many(self, values: Iterable[int]) -> None:
        """Record every value of an iterable."""
        for v in values:
            self.record(v)

    def merge(self, other: LogLinearHistogram) -> None:
        """Add the counts of a histogram with the same layout."""
        if (other.resolution, other.sub_buckets) != (self.resolution, self.sub_buckets):
            raise ValueError("cannot merge histograms with different layouts")
        for b, c in other._counts.items():
            self._counts[b] = self._counts.get(b, 0) + c
        self.count += other.count
        self.total += other.total
        if other.min is not None:
            self.min = other.min if self.min is None else min(self.min, other.min)
        if other.max is not None:
            self.max = other.max if self.max is None else max(self.max, other.max)

    # ------------------------------------------------------------------ queries
    def percentile(self, p: float) -> int | None:
        """Upper edge of the bucket holding the ``p``-th percentile (0 < p <= 100).

        Nearest rank: the smallest recorded rank ``ceil(p/100*count)``. The result is
        clipped to the exact maximum, so ``percentile(100) == max``. None when empty.
        """
        if not 0.0 < p <= 100.0:
            raise ValueError(f"percentile must be in (0, 100], got {p}")
        if self.count == 0:
            return None
        rank = max(1, math.ceil(p / 100.0 * self.count))
        seen = 0
        for b in sorted(self._counts):
            seen += self._counts[b]
            if seen >= rank:
                hi = self.bucket_bounds(b)[1] - 1
                assert self.max is not None
                return min(hi, self.max)
        return self.max  # pragma: no cover - unreachable

    def mean(self) -> float | None:
        """Exact mean of the recorded values, or None when empty."""
        return None if self.count == 0 else self.total / self.count

    def counts(self) -> Mapping[int, int]:
        """Bucket index -> count (non-empty buckets only)."""
        return dict(self._counts)

    def summary(self, scale: float = 1e-3) -> dict[str, Any]:
        """``{count, min, p50, p95, p99, p999, max, mean}`` in base units times ``scale``
        (default ns -> us)."""

        def sc(v: float | None) -> float | None:
            return None if v is None else v * scale

        return {
            "count": self.count,
            "min": sc(self.min),
            "p50": sc(self.percentile(50)),
            "p95": sc(self.percentile(95)),
            "p99": sc(self.percentile(99)),
            "p999": sc(self.percentile(99.9)),
            "max": sc(self.max),
            "mean": sc(self.mean()),
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready layout and counts (bucket keys as strings)."""
        return {
            "resolution": self.resolution,
            "sub_buckets": self.sub_buckets,
            "count": self.count,
            "total": self.total,
            "min": self.min,
            "max": self.max,
            "buckets": {str(b): c for b, c in sorted(self._counts.items())},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LogLinearHistogram:
        """Inverse of :meth:`to_dict`."""
        h = cls(int(d["resolution"]), int(d["sub_buckets"]))
        h._counts = {int(b): int(c) for b, c in dict(d["buckets"]).items()}
        h.count = int(d["count"])
        h.total = int(d["total"])
        h.min = None if d.get("min") is None else int(d["min"])
        h.max = None if d.get("max") is None else int(d["max"])
        return h
