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

The metric tests pin the end-to-end F5 RTF (server warm-up boundary to the RX client's last
sample) against an injected RX-client pause before its warm-up completes, with the served
engine slowed to a known rate (a sleep after every page): the pre-fix client-only figure
inflates (the server keeps computing into the loopback/ZMQ buffers while the client sleeps),
the end-to-end figure does not. The cap test lets the pilot over-state the repetitions' rate
by 50x and checks that ``point_wall_cap_s`` still bounds the point (truncated repetitions,
measured over what they delivered). The stress smoke test runs :mod:`bench.f5_stress`.

All tests start client processes (``python -m bench._zmq_client``): integration, slow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bench import _engine_point, _scenes, f5_stress

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SLOW_ENGINE = {"server_page_sleep_s": 0.006}
"""~6.4 ms of wall per 64-sample page at 1e5 S/s: RTF ~0.09 whatever the host speed."""


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


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(a), abs(b))


def test_f5_end_to_end_rtf_is_not_inflated_by_a_starved_rx_client() -> None:
    """Pilot-only points (0.2 s after the warm-up, ~2.4 s of wall) with the RX client asleep
    0.8 s (shorter than the run) and 3 s (longer) right after its first read, before its
    warm-up completes.

    0.8 s: the server keeps computing into the buffers (~7k samples), so its own figure
    stays put, the end-to-end figure stays within 15 % of the unpaused one and the
    client-only figure inflates (~1.37x on Windows and WSL2). 3 s: the run ends before the
    client wakes up; the samples reach it late, so the end-to-end figure may only fall. On
    Windows the whole run waits in the loopback buffers (client-only figure ~400x the
    end-to-end one); on WSL2 either that or part of it still waits in the server's queues
    when the client wakes and arrives ~0.9 s later (TCP zero-window probe backoff; ~4x).
    Whatever the buffering, the end-to-end figure never exceeds the engine-only one."""
    sc = _scene("1")
    kw: dict[str, Any] = {
        "reps": 0,
        "warmup_s": "0.02",
        "min_virtual_s": "0.6",
        "point_wall_cap_s": 60.0,
    }
    ref = _engine_point.measure_zmq(sc, inject=SLOW_ENGINE, **kw)
    short = _engine_point.measure_zmq(sc, inject={**SLOW_ENGINE, "rx_pause_s": 0.8}, **kw)
    long = _engine_point.measure_zmq(sc, inject={**SLOW_ENGINE, "rx_pause_s": 3.0}, **kw)
    for r in (ref, short, long):
        assert r["status"] == "ok" and r["reps"] == 0 and r["clock_shared"]
        assert r["pilot_rtf_basis"] == "end-to-end"
        assert r["pilot_virtual_s"] == pytest.approx(0.2, rel=0.01)
        assert 0.03 < r["pilot_rtf_server"] < 0.15  # the injected engine rate
        assert r["pilot_rtf"] <= 1.02 * r["pilot_rtf_server"]
    # unstarved: the three figures agree
    e2e = ref["pilot_rtf"]
    assert _close(e2e, ref["pilot_rtf_rx_client"], 0.10)
    assert _close(e2e, ref["pilot_rtf_server"], 0.10)
    # pause shorter than the run: end-to-end unchanged, client-only inflated
    assert _close(short["pilot_rtf_server"], ref["pilot_rtf_server"], 0.15)  # premise
    assert _close(short["pilot_rtf"], e2e, 0.15)
    assert short["pilot_rtf_rx_client"] > 1.25 * short["pilot_rtf"]
    # pause longer than the run: end-to-end never above the unpaused value
    assert long["pilot_rtf"] <= 1.05 * e2e
    assert long["pilot_rtf_rx_client"] > 2 * long["pilot_rtf"]
    pilot = long["f5_runs"][0]
    ts = pilot["timestamps_ns"]
    assert ts["launch"] <= ts["client_start"] <= ts["client_resume"] <= ts["server_warm"]
    assert ts["server_warm"] <= ts["server_last_page"] <= ts["client_end"] <= ts["done"]
    assert pilot["rx_pause_s"] == 3.0 and pilot["max_gap_s"] >= 2.9


def test_f5_wall_cap_bounds_repetitions_that_the_pilot_undersized() -> None:
    """The pilot runs the engine at full speed, the repetitions 50x slower (a sleep of 20 ms
    after every 0.64 ms page): the repetition length sized from the pilot would take minutes
    per repetition. The RX client's deadline truncates them, they are measured over what
    they delivered, the remaining ones are skipped, and the point ends near its cap."""
    cap = 14.0
    r = _engine_point.measure_zmq(
        _scene("1"),
        reps=3,
        warmup_s="0.02",
        min_virtual_s="0.3",
        target_pages=100_000,
        rep_wall_target_s=1.0,
        point_wall_cap_s=cap,
        inject={"rep_server_page_sleep_s": 0.02},
    )
    assert r["status"] == "ok" and r["clock_shared"]
    assert r["reps"] >= 1 and r["reps_truncated"] == r["reps"]
    assert r["reps"] + r["reps_skipped_cap"] <= 3
    assert r["point_wall_s"] <= cap + 6.0  # client start-up and teardown beyond the deadlines
    assert all(v < r["rep_virtual_s"] for v in r["rep_virtual_s_measured"])
    for x in r["f5_runs"][1:]:
        assert x["truncated_by_wall"] and x["virtual_s"] > 0
        assert x["rtf"] < 0.5 * r["pilot_rtf"]  # the slowed engine, measured
        assert _close(x["rtf"], x["rtf_server"], 0.25)
        assert x["run_wall_s"] <= x["wall_max_s"] + 6.0


def test_f5_stress_runs_concurrent_workers(tmp_path: Path) -> None:
    """Two concurrent workers, one small F5 point each (unpinned off Linux)."""
    rc = f5_stress.main(
        [
            "--runs",
            "2",
            "--concurrency",
            "2",
            "--cells",
            "40",
            "--fs",
            "1e5",
            "--reps",
            "1",
            "--rep-wall-s",
            "0.3",
            "--min-virtual-s",
            "0.15",
            "--threads",
            "1",
            "--timeout-min",
            "8",
            "--out",
            str(tmp_path),
        ]
    )
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert rc == 0, summ["errors"]
    assert summ["runs_ok"] == 2 and summ["runs_failed"] == 0 and summ["runs_stalled_or_lost"] == 0
    assert summ["clock_shared_all"] and summ["rtf_end_to_end"]["n"] == 2
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 2
