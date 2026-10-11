"""B2 feature set F5 (:func:`bench._engine_point.measure_zmq`): run length and floor status.

Root cause pinned here. B2 run 20261011T084048Z (c7i.48xlarge, ten concurrent workers):
L1_N240_R1_fs1e6_F5, L1_N240_R1_fs5.76e6_F5 and L4_N4096_R1_fs1e6_F5 failed with ``ZMQ
client failed (rx): recv error TIMEOUT`` after the RX client's 120 s ``recv`` timeout.
``measure_zmq`` served an engine whose run ended at the scenario's ``run.duration_s`` (1 s in
every benchmark scene) while its clients streamed ``warm-up + dur`` samples per stream, and
``dur`` is sized from the pilot RTF (``rtf_pilot * rep_wall_target_s``, up to
``target_pages`` pages). Any pilot RTF above ~0.16 (6 s target) sized ``dur`` >= 0.95 s; the
served run then ended (RX ``END_OF_STREAM`` at 1 s) before the RX client's ``num_done``
request was complete, and the client waited its whole timeout for samples that the run
never computes (no stall in the transport: every sample of the run was delivered). The
25 ms poll that the wake-up fix removed had kept F5 near or below that RTF; whether a point
fails follows its pilot RTF, which varies with the host load (a starved RX client reads the
samples buffered after its warm-up in a burst and can over-state it). Reproduced on Windows
and WSL2: every repetition with ``warm-up + dur`` > ``duration_s`` fails (the RX client
stops at exactly the run's last sample), none of 150 shorter ones under 4x CPU
oversubscription and delayed client starts. The fix serves a run of exactly the streamed
length. The second test pins the floor status of a point that the pilot already puts below
``rtf_floor``: no repetitions, which used to raise ``max() iterable argument is empty`` (B2
point L128_N240_R2_fs23.04e6_F5).

Both tests start client processes (``python -m bench._zmq_client``): integration, slow.
"""

from __future__ import annotations

import pytest

from bench import _engine_point, _scenes

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _scene(duration_s: str) -> dict[str, object]:
    return _scenes.bench_scenario(
        links=1, n_cells=40, n_ris=1, rate_sps="1e5", features="F5", duration_s=duration_s
    )


def test_f5_repetitions_longer_than_the_scenario_run_complete() -> None:
    """Pilot and repetition both stream more samples than the scenario's 0.05 s run: each
    served run must last as long as its clients stream (before the fix the RX client got
    the run's 5,000 samples and then timed out waiting for the rest)."""
    r = _engine_point.measure_zmq(
        _scene("0.05"),
        reps=1,
        warmup_s="0.02",
        min_virtual_s="0.15",
        target_pages=200,
        rep_wall_target_s=0.5,
        point_wall_cap_s=60.0,
    )
    assert r["status"] == "ok"
    assert r["reps"] == 1 and r["rep_virtual_s"] >= 0.15
    assert r["warmup_virtual_s"] + r["pilot_virtual_s"] > 0.05  # the pilot overran too
    assert r["pilot_rtf"] > 0 and r["rtf_median"] > 0
    assert r["counters"] == {"bad_packets": 0, "seq_errors": 0}


def test_f5_point_below_the_rtf_floor_reports_below_floor() -> None:
    """A pilot below ``rtf_floor``: no repetitions, ``status`` ``below_floor`` with the pilot
    RTF as the sizing estimate and the pilot's transport counters (both paths)."""
    r = _engine_point.measure_zmq(
        _scene("1"), reps=3, warmup_s="0.02", min_virtual_s="0.15", rtf_floor=1e9
    )
    assert r["status"] == "below_floor" and r["reps"] == 0
    assert r["rtf_median"] is None and r["rtf_reps"] == []
    assert r["pilot_rtf"] > 0
    assert r["counters"] == {"bad_packets": 0, "seq_errors": 0}
    sc = _scenes.bench_scenario(links=1, n_cells=40, n_ris=1, rate_sps="1e5", features="F1")
    q = _engine_point.measure_lockstep(
        sc, features="F1", reps=3, min_virtual_s="0.15", rtf_floor=1e9
    )
    assert q["status"] == "below_floor" and q["reps"] == 0
    assert q["rtf_median"] is None and q["pilot_rtf"] > 0
