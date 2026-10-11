"""RtfMeter aggregate throughput after a restart (section 12.1).

``aggregate_sps`` is the samples produced in the measured wall span divided by that span.
After :meth:`RtfMeter.restart` (end of a warm-up) the warm-up's samples must not count: they
were produced before the measured span started.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sremu.metrics.rt import RtfMeter

pytestmark = pytest.mark.unit

S = 10**9  # ns per second


def test_aggregate_sps_excludes_samples_before_restart() -> None:
    """5 s of warm-up at 1 kS/s, restart, then 1 s more: 1000 S/s, not 6000."""
    m = RtfMeter({"a": Fraction(1000)})
    m.start(0)
    for i in range(1, 6):
        m.advance({"a": 1000}, i * S)
    m.restart(5 * S, {"a": 5000})
    m.advance({"a": 1000}, 6 * S)
    r = m.report()
    assert r.wall_s == pytest.approx(1.0)
    assert r.virtual_s == Fraction(1)
    assert r.rtf == pytest.approx(1.0)
    assert r.aggregate_sps == pytest.approx(1000.0)
    assert dict(r.samples) == {"a": 6000}  # cursors: the warm-up is still in the samples


def test_aggregate_sps_two_streams_after_restart() -> None:
    m = RtfMeter({"a": Fraction(1000), "b": Fraction(4000)})
    m.start(0)
    m.advance({"a": 3000, "b": 12_000}, 3 * S)
    m.restart(3 * S, {"a": 3000, "b": 12_000})
    m.advance({"a": 500, "b": 2000}, 3 * S + S // 2)
    r = m.report()
    assert r.wall_s == pytest.approx(0.5)
    assert r.aggregate_sps == pytest.approx((500 + 2000) / 0.5)


def test_aggregate_sps_without_restart_unchanged() -> None:
    m = RtfMeter({"a": Fraction(1000)})
    m.start(0)
    m.advance({"a": 250}, S // 2)
    assert m.report().aggregate_sps == pytest.approx(500.0)
