"""Real-time metrics (DESIGN.md section 12.1; the RTF definition of contradiction 36).

* **RTF (lockstep).** Virtual time advanced is the MINIMUM over streams of
  ``samples produced / f_s`` (exact), divided by the wall time. ``RTF_min`` is the minimum
  over consecutive 100 ms windows of virtual time ``[w*0.1, (w+1)*0.1)`` of
  ``0.1 s / (wall time spent advancing the minimum virtual time across the window)``.
  Partial windows at the end are not counted.
* **Aggregate throughput** is a separate metric: ``sum_s samples_s / wall`` in S/s.
* **Loss counters** (I6) use :data:`sremu.contracts.COUNTER_NAMES`.
* **Deadline misses** (paced) are counted per slack; page latencies and stage times go to
  :class:`~sremu.metrics.hist.LogLinearHistogram` (1 us resolution).

Wall time enters only here and in ``engine/pacer.py`` (I7): the meters receive integer
nanosecond timestamps from the caller's clock, so tests can drive them with a fake clock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Final

from sremu.contracts import COUNTER_NAMES, StreamId
from sremu.timebase import ceil_ticks, fraction_to_str, parse_exact

from .hist import LogLinearHistogram

__all__ = [
    "RTF_WINDOW_S",
    "Counters",
    "DeadlineStats",
    "RtfMeter",
    "RtfReport",
]

RTF_WINDOW_S: Final[Fraction] = Fraction(1, 10)
"""Virtual-time window of ``RTF_min``: 100 ms."""


class Counters(dict[str, int]):
    """The I6 loss counters (every name of ``COUNTER_NAMES`` starts at 0)."""

    def __init__(self, init: Mapping[str, int] | None = None) -> None:
        super().__init__(dict.fromkeys(COUNTER_NAMES, 0))
        if init:
            for k, v in init.items():
                self[k] = int(v)

    def add(self, name: str, n: int = 1) -> None:
        """Increase counter ``name`` by ``n`` (unknown names are created)."""
        self[name] = self.get(name, 0) + int(n)

    def delta(self, before: Mapping[str, int]) -> dict[str, int]:
        """Per-counter difference ``self - before``."""
        return {k: v - int(before.get(k, 0)) for k, v in self.items()}


@dataclass(frozen=True)
class RtfReport:
    """Summary of a run's real-time factor (section 12.1).

    ``virtual_s``: min over streams of samples/f_s (exact). ``wall_s``: wall time. ``rtf``:
    ``virtual_s / wall_s``. ``rtf_min``: minimum over complete 100 ms virtual windows (None
    when no window completed). ``aggregate_sps``: sum over streams of the samples produced
    in ``wall_s`` (since :meth:`RtfMeter.start` or the last :meth:`RtfMeter.restart`) /
    ``wall_s``. ``samples``: per stream, every sample produced (warm-up included).
    ``windows``: per complete window ``(t0_s, wall_s, rtf)``.
    """

    virtual_s: Fraction
    wall_s: float
    rtf: float | None
    rtf_min: float | None
    aggregate_sps: float | None
    samples: Mapping[StreamId, int]
    windows: tuple[tuple[Fraction, float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (exact virtual times as ``p/q`` strings)."""
        return {
            "virtual_s": fraction_to_str(self.virtual_s),
            "virtual_s_float": float(self.virtual_s),
            "wall_s": self.wall_s,
            "rtf": self.rtf,
            "rtf_min": self.rtf_min,
            "aggregate_sps": self.aggregate_sps,
            "samples": dict(self.samples),
            "n_windows": len(self.windows),
        }


class RtfMeter:
    """Tracks samples produced per stream against wall time (section 12.1 RTF).

    ``rates``: stream -> exact rate (S/s). Call :meth:`start` with the wall clock (ns), then
    :meth:`advance` whenever a stream produced samples (with the wall clock after the
    work), and :meth:`report` at the end. Streams that never produce keep the virtual time
    at 0, as the min-over-streams definition requires; pass only the streams that the run
    produces.
    """

    def __init__(
        self, rates: Mapping[StreamId, Fraction | int | str], window_s: Fraction = RTF_WINDOW_S
    ) -> None:
        if not rates:
            raise ValueError("RtfMeter needs at least one stream")
        self.rates = {s: parse_exact(f, name=f"rate of {s}") for s, f in rates.items()}
        self.window_s = parse_exact(window_s, name="window_s")
        self.samples: dict[StreamId, int] = dict.fromkeys(self.rates, 0)
        self._t0_ns: int | None = None
        self._last_ns: int | None = None
        self._win_index = 0  # next window whose end has not been reached
        self._win_start_ns: int | None = None
        self.windows: list[tuple[Fraction, float, float]] = []
        self._thr: dict[StreamId, int] | None = None
        self._v0 = Fraction(0)
        self._s0 = sum(self.samples.values())  # samples before the measurement (restart)
        self._pending_start = False

    def start(self, wall_ns: int) -> None:
        """Start the wall clock at ``wall_ns``."""
        self._t0_ns = int(wall_ns)
        self._last_ns = int(wall_ns)
        self._win_start_ns = int(wall_ns)
        self._s0 = sum(self.samples.values())

    def restart(self, wall_ns: int, samples: Mapping[StreamId, int]) -> None:
        """Restart the measurement at ``wall_ns`` with the streams already at ``samples``
        (after a warm-up). Windows start at the next 100 ms boundary of virtual time; the
        mean RTF counts only the virtual time advanced after the restart."""
        self.samples = {s: int(samples.get(s, 0)) for s in self.rates}
        v0 = self.virtual_s()
        self._v0 = v0
        self._s0 = sum(self.samples.values())
        self._win_index = int(v0 // self.window_s) + 1
        self.windows = []
        self._thr = None
        self._t0_ns = int(wall_ns)
        self._last_ns = int(wall_ns)
        self._win_start_ns = None  # set when the first full window starts
        self._pending_start = True

    @property
    def started(self) -> bool:
        """Whether :meth:`start` was called."""
        return self._t0_ns is not None

    def virtual_s(self) -> Fraction:
        """Min over streams of samples/f_s (exact)."""
        return min(Fraction(n) / self.rates[s] for s, n in self.samples.items())

    def _thresholds(self) -> dict[StreamId, int]:
        """Samples each stream needs for the virtual time to reach the end of the current
        window: ``ceil(t_end*f_s)`` (exact integers)."""
        t_end = (self._win_index + 1) * self.window_s
        return {s: ceil_ticks(t_end, f) for s, f in self.rates.items()}

    def advance(self, produced: Mapping[StreamId, int], wall_ns: int) -> None:
        """Add samples produced per stream; ``wall_ns`` is the wall clock after the work."""
        if self._t0_ns is None:
            self.start(wall_ns)
        for s, n in produced.items():
            if s in self.samples:
                self.samples[s] += int(n)
        self._last_ns = int(wall_ns)
        if self._pending_start:
            # a restart mid-window: the first full window starts when virtual time reaches
            # its start boundary
            t_start = self._win_index * self.window_s
            if all(self.samples[s] >= ceil_ticks(t_start, f) for s, f in self.rates.items()):
                self._win_start_ns = int(wall_ns)
                self._pending_start = False
            else:
                return
        if self._thr is None:
            self._thr = self._thresholds()
        while all(self.samples[s] >= k for s, k in self._thr.items()):
            assert self._win_start_ns is not None
            dt = (int(wall_ns) - self._win_start_ns) * 1e-9
            rtf = float(self.window_s) / dt if dt > 0 else float("inf")
            self.windows.append((self._win_index * self.window_s, dt, rtf))
            self._win_index += 1
            self._win_start_ns = int(wall_ns)
            self._thr = self._thresholds()

    def report(self, wall_ns: int | None = None) -> RtfReport:
        """RTF summary up to ``wall_ns`` (default: the last :meth:`advance`)."""
        if self._t0_ns is None:
            return RtfReport(Fraction(0), 0.0, None, None, None, dict(self.samples))
        end = self._last_ns if wall_ns is None else int(wall_ns)
        assert end is not None
        wall = (end - self._t0_ns) * 1e-9
        v = self.virtual_s() - self._v0
        rtf = float(v) / wall if wall > 0 else None
        # only the samples produced in the measured wall span (not the warm-up's)
        agg = (sum(self.samples.values()) - self._s0) / wall if wall > 0 else None
        rmin = min((w[2] for w in self.windows), default=None)
        return RtfReport(v, wall, rtf, rmin, agg, dict(self.samples), tuple(self.windows))


@dataclass
class DeadlineStats:
    """Paced-mode deadline bookkeeping (section 4.10): misses per slack and page latency.

    A page released at ``R`` (wall ns) and delivered at ``D`` misses the deadline of slack
    ``s`` when ``D > R + s``. ``latency`` holds ``D - max(R, t_inputs_complete)`` in ns.
    """

    slacks_ns: Sequence[int]
    misses: dict[int, int] = field(default_factory=dict)
    pages: int = 0
    latency: LogLinearHistogram = field(default_factory=LogLinearHistogram)
    utilization: LogLinearHistogram = field(
        default_factory=lambda: LogLinearHistogram(resolution=1, sub_buckets=64)
    )

    def __post_init__(self) -> None:
        self.slacks_ns = tuple(int(s) for s in self.slacks_ns)
        for s in self.slacks_ns:
            self.misses.setdefault(s, 0)

    def page(
        self,
        release_ns: int,
        delivered_ns: int,
        inputs_ns: int | None = None,
        compute_ns: int | None = None,
        page_ns: int | None = None,
    ) -> None:
        """Account one page: release time, delivery time, optional inputs-complete time,
        compute time and page duration (both ns; utilization is recorded in 1/1e6 units)."""
        self.pages += 1
        for s in self.slacks_ns:
            if delivered_ns > release_ns + s:
                self.misses[s] += 1
        ref = release_ns if inputs_ns is None else max(release_ns, inputs_ns)
        self.latency.record(max(0, delivered_ns - ref))
        if compute_ns is not None and page_ns:
            self.utilization.record(round(1e6 * compute_ns / page_ns))

    def miss_rate(self, slack_ns: int) -> float | None:
        """Misses per page at ``slack_ns`` (None before the first page)."""
        if self.pages == 0:
            return None
        return self.misses[int(slack_ns)] / self.pages

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary (latency in us; utilization as a fraction)."""
        u = self.utilization.summary(scale=1e-6)
        return {
            "pages": self.pages,
            "misses": {f"{s * 1e-6:g}ms": c for s, c in self.misses.items()},
            "miss_rate": {f"{s * 1e-6:g}ms": self.miss_rate(s) for s in self.slacks_ns},
            "latency_us": self.latency.summary(),
            "utilization": {k: u[k] for k in ("p50", "p99", "max")},
        }
