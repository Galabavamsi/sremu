"""WP6 unit tests: log-linear histogram, RTF meter (section 12.1 definition), deadline
statistics, counters and the event heap ordering (section 4.6)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import COUNTER_NAMES, EventPriority
from sremu.engine.events import EventHeap
from sremu.metrics.hist import LogLinearHistogram
from sremu.metrics.rt import Counters, DeadlineStats, RtfMeter

pytestmark = pytest.mark.unit


def test_hist_buckets_contiguous_and_relative_width() -> None:
    h = LogLinearHistogram()
    prev_hi = 0
    for b in range(0, 64 * 30):
        lo, hi = h.bucket_bounds(b)
        assert lo == prev_hi  # contiguous, no gaps or overlaps
        assert h.bucket_of(lo) == b
        assert h.bucket_of(hi - 1) == b
        if b >= 64:
            assert (hi - lo) / lo <= 1 / 64 + 1e-12  # log-linear: relative width <= 1/64
        else:
            assert hi - lo == 1000  # 1 us resolution in the linear range
        prev_hi = hi


def test_hist_percentiles_nearest_rank_within_bucket(rng: np.random.Generator) -> None:
    vals = rng.integers(0, 50_000_000, 20_000)  # ns
    h = LogLinearHistogram()
    h.record_many(int(v) for v in vals)
    s = np.sort(vals)
    for p in (50, 95, 99, 99.9, 100):
        exact = int(s[max(1, int(np.ceil(p / 100 * s.size))) - 1])
        got = h.percentile(p)
        assert got is not None
        lo, hi = h.bucket_bounds(h.bucket_of(exact))
        assert lo <= got <= hi - 1  # the bucket's upper edge (or the exact max)
        assert got >= exact
    assert h.percentile(100) == int(vals.max())
    assert h.min == int(vals.min())
    assert h.mean() == pytest.approx(float(vals.mean()))
    h2 = LogLinearHistogram.from_dict(h.to_dict())
    assert h2.counts() == h.counts()
    assert h2.percentile(99) == h.percentile(99)


def test_hist_merge_equals_joint_recording() -> None:
    a, b, c = LogLinearHistogram(), LogLinearHistogram(), LogLinearHistogram()
    for v in range(0, 3_000_000, 7919):
        (a if v % 2 else b).record(v)
        c.record(v)
    a.merge(b)
    assert a.counts() == c.counts()
    assert (a.count, a.min, a.max, a.total) == (c.count, c.min, c.max, c.total)


def test_rtf_meter_min_over_streams_and_windows() -> None:
    """RTF = min over streams of samples/f_s divided by wall time (section 12.1); RTF_min
    over 100 ms virtual windows."""
    m = RtfMeter({"a": Fraction(1000), "b": Fraction(10_000)})
    m.start(0)
    # 1 s of wall: stream a produces 1 s, stream b only 0.5 s -> virtual 0.5 s
    for i in range(1, 11):
        m.advance({"a": 100, "b": 500}, i * 100_000_000)
    r = m.report()
    assert r.virtual_s == Fraction(1, 2)
    assert r.wall_s == pytest.approx(1.0)
    assert r.rtf == pytest.approx(0.5)
    assert r.aggregate_sps == pytest.approx((1000 + 5000) / 1.0)
    assert len(r.windows) == 5  # five complete 100 ms windows of virtual time
    assert r.rtf_min == pytest.approx(0.5)
    # a slow window lowers RTF_min but not the mean much
    m.advance({"a": 100, "b": 1000}, 1_000_000_000 + 400_000_000)
    r2 = m.report()
    assert r2.rtf_min == pytest.approx(0.1 / 0.4)


def test_rtf_meter_restart_after_warmup() -> None:
    m = RtfMeter({"a": Fraction(1000)})
    m.start(0)
    m.advance({"a": 150}, 10**9)  # slow warm-up: 0.15 s in 1 s
    m.restart(10**9, {"a": 150})
    for i in range(1, 6):
        m.advance({"a": 100}, 10**9 + i * 10_000_000)
    r = m.report()
    assert r.virtual_s == Fraction(1, 2)
    assert r.wall_s == pytest.approx(0.05)
    assert r.rtf == pytest.approx(10.0)
    # windows start at the next 100 ms boundary (0.2 s); the warm-up is excluded
    assert [w[0] for w in r.windows] == [
        Fraction(2, 10),
        Fraction(3, 10),
        Fraction(4, 10),
        Fraction(5, 10),
    ]
    assert r.rtf_min == pytest.approx(10.0)


def test_deadline_stats_counts_per_slack() -> None:
    d = DeadlineStats([0, 1_000_000, 5_000_000])
    d.page(0, 0)
    d.page(0, 500_000)
    d.page(0, 2_000_000)
    d.page(0, 6_000_000, inputs_ns=1_000_000)
    assert d.misses == {0: 3, 1_000_000: 2, 5_000_000: 1}
    assert d.miss_rate(0) == pytest.approx(0.75)
    assert d.latency.max == 5_000_000  # delivered - max(release, inputs complete)


def test_counters_have_every_i6_name() -> None:
    c = Counters()
    assert set(COUNTER_NAMES) <= set(c)
    assert all(v == 0 for v in c.values())
    before = dict(c)
    c.add("late_samples", 5)
    assert c.delta(before)["late_samples"] == 5


def test_event_heap_orders_by_t_prio_seq() -> None:
    h = EventHeap()
    h.push(Fraction(2), EventPriority.MARKER, 1, "m")
    h.push(Fraction(1), EventPriority.RIS, 3, "r3")
    h.push(Fraction(1), EventPriority.GEOMETRY, 4, "g")
    h.push(Fraction(1), EventPriority.RIS, 2, "r2")
    cancelled = h.push(Fraction(1, 2), EventPriority.STREAM, 5, "x")
    cancelled.cancelled = True
    assert len(h) == 4
    assert [e.kind for e in h] == ["g", "r2", "r3", "m"]
    due = h.pop_due(Fraction(1))
    assert [e.kind for e in due] == ["g", "r2", "r3"]
    assert h.pop_due(Fraction(2), inclusive=False) == []
    assert h.pop() is not None
    assert h.pop() is None
    with pytest.raises(TypeError):
        h.push(0.5, 0, 9, "float time")  # type: ignore[arg-type]
