"""B4 checks: lockstep commands after the end of the run, paced repetition validity.

The B4 lockstep run of 20261011T074442Z (AWS c7i.8xlarge) reported ``lockstep_check_ok =
false`` only because the last barrier's command at 10 kHz (barrier one period before the end,
latency L_min + U[0, L_min] >= one period) switches after the end of the 1 s run and so stays
``scheduled``. These tests pin the corrected definition (exact count for new rows, a bound from
the controller's latency model for older rows) and the offline recomputation.

Paced: the B4 rerun at 81f2735 backlogged every 1 ms repetition (page latency p50 5-27 s), so
its timed commands arrived lead + lag ahead of their pages (late = 0 vacuous) and its visible
latency was the backlog. The tests pin the backlog rule, that a backlogged repetition never
contributes a latency or a late/exact fraction, the timed-command metrics (apply delay and
virtual margin against the pacer's wall schedule) and the summary of ``paced.json``. They
never assert a timing.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bench import b4_control, summarize


def _row(**kw: Any) -> dict[str, Any]:
    row = {
        "variant": "inproc",
        "rate_hz": 10_000,
        "virtual_s": 1.0,
        "commands": 10_000,
        "expected_commands": 10_000,
        "status": {"applied": 9_999, "scheduled": 1},
        "lateness_samples": {"n": 20_000.0},
        "lateness_nonzero": 0,
        "lateness_max": 0,
        "n_rx": 2,
    }
    row.update(kw)
    return row


def test_scheduled_after_end_counts_only_switches_beyond_the_run() -> None:
    ends = {"rx000.rx0": 1_000_000, "rx001.rx0": 1_000_000}

    def rec(status: str, k: int) -> SimpleNamespace:
        return SimpleNamespace(status=status, k_applied=dict.fromkeys(ends, k), k_req={})

    recs = [rec("applied", 999_960), rec("scheduled", 1_000_082), rec("scheduled", 999_999)]
    # 999_999 < end: the engine computed that sample, so the command should have applied
    assert b4_control.scheduled_after_end(recs, ends) == 1
    assert b4_control.scheduled_after_end([rec("scheduled", 1_000_000)], ends) == 1


def test_after_end_bound_follows_the_latency_model() -> None:
    # latency L_min + U[0, L_min] with L_min = 0.1 ms: only barriers in [end - 0.2 ms, end)
    assert b4_control.after_end_bound(10_000, 1.0) == 2  # t_b = 0.9998 s and 0.9999 s
    assert b4_control.after_end_bound(1_000, 1.0) == 0  # last barrier 0.999 s: always inside
    assert b4_control.after_end_bound(100, 2.0) == 0
    assert b4_control.after_end_bound(0, 1.0) == 0
    assert b4_control.after_end_bound(10_000, Fraction(1)) == 2


def test_lockstep_check_accepts_a_command_scheduled_after_the_end() -> None:
    # exact count recorded (new rows)
    assert b4_control.lockstep_problems(_row(scheduled_after_end=1)) == []
    # a scheduled command inside the run is still a failure
    assert b4_control.lockstep_problems(_row(scheduled_after_end=0))
    # older rows: bounded by the latency model (2 at 10 kHz over 1 s, 0 at 1 kHz)
    assert b4_control.lockstep_problems(_row()) == []
    assert b4_control.lockstep_problems(
        _row(status={"applied": 9_997, "scheduled": 3})
    )  # more than the bound
    one_k = _row(
        rate_hz=1_000,
        commands=1_000,
        expected_commands=1_000,
        status={"applied": 999, "scheduled": 1},
        lateness_samples={"n": 2_000.0},
    )
    assert b4_control.lockstep_problems(one_k)
    # every other rule still applies
    assert b4_control.lockstep_problems(_row(status={"applied": 9_999, "late_applied": 1}))
    assert b4_control.lockstep_problems(_row(lateness_nonzero=2, lateness_max=5))
    assert b4_control.lockstep_problems(_row(lateness_samples={"n": 10_000.0}))


def test_recompute_lockstep_summary_offline(tmp_path: Path) -> None:
    rows = [
        _row(
            variant="none", commands=0, expected_commands=0, status={}, lateness_samples={}, rtf=2.7
        ),
        _row(rtf=0.078),
        _row(variant="remote", rtf=0.021),
    ]
    (tmp_path / "lockstep.json").write_text(json.dumps(rows), encoding="utf-8")
    orig = {
        "lockstep_check_ok": False,
        "lockstep_check_problems": {"inproc@10000Hz": ["statuses other than applied"]},
        "lockstep_all_lateness_zero": True,
        "interrupted": None,
    }
    (tmp_path / "summary.json").write_text(json.dumps(orig), encoding="utf-8")
    assert b4_control.main(["--recompute", str(tmp_path)]) == 0
    out = json.loads((tmp_path / "summary_recomputed.json").read_text(encoding="utf-8"))
    assert out["lockstep_check_ok"] is True and out["lockstep_check_problems"] == {}
    assert out["lockstep_commands_total"] == 20_000
    assert out["recomputed"]["original"]["lockstep_check_ok"] is False
    assert out["lockstep_scheduled_at_end"]["inproc@10000Hz"] == {
        "scheduled": 1,
        "after_end": None,
        "after_end_bound": 2,
    }
    assert b"\r\n" not in (tmp_path / "summary_recomputed.json").read_bytes()


def _paced_row(key: str, rep: int, **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": key,
        "rate_hz": 100,
        "mode": "timed",
        "lead_s": 0.008,
        "page_samples": 1024,
        "page_s": 1.024e-3,
        "rep": rep,
        "wall_s": 65.0,
        "commands": 6000,
        "status": {"applied": 6000},
        "lateness_samples": {"p99": 0.0, "n": 12000.0},
        "lateness_max": 0,
        "page_latency_us": {"p50": 300.0, "p99": 900.0},
        "miss_rate": {"0ms": 0.2, "1ms": 0.01, "5ms": 0.0},
        "visible_latency_us": {"p50": 2000.0, "p99": 4000.0},
        "visible_minus_recv_us": {"p50": 2000.0, "p99": 4000.0},
        "late_fraction": 0.0,
        "exact_fraction": 1.0,
        "apply_delay_us": {"p50": 300.0, "p99": 1000.0},
        "virtual_margin_us": {"p1": 3000.0, "p50": 4000.0},
        "send_to_schedule_rtt_us": {"p99": 4500.0},
        "client": {"achieved_rate_hz": 100.0},
        "sustainable": False,
        "commands_per_virtual_s": 100.0,
        "engine_bound": False,
    }
    row.update(kw)
    reasons = b4_control.backlog_reasons(row["page_latency_us"], row["miss_rate"], row["page_s"])
    row.setdefault("backlogged", bool(reasons))
    row.setdefault("backlog_reasons", reasons)
    return row


def test_backlog_rule() -> None:
    page_s = 1.024e-3
    # p50 below one page and 5 ms misses at (not above) 1 %: valid
    assert b4_control.backlog_reasons({"p50": 900.0}, {"5ms": 0.01}, page_s) == []
    slow = b4_control.backlog_reasons({"p50": 5.2e6}, {"5ms": 0.0}, page_s)
    assert len(slow) == 1 and "page latency p50" in slow[0]
    miss = b4_control.backlog_reasons({"p50": 500.0}, {"5ms": 0.02}, page_s)
    assert len(miss) == 1 and "5 ms" in miss[0]
    assert len(b4_control.backlog_reasons({"p50": 2e3}, {"5ms": 1.0}, page_s)) == 2
    assert b4_control.backlog_reasons(None, None, page_s) == ["no paced page statistics"]
    # rows written before the flag existed are judged from their page statistics
    old = {"page_samples": 1024, "page_latency_us": {"p50": 1.4e6}, "miss_rate": {"5ms": 1.0}}
    assert b4_control.is_backlogged(old)
    fine = {**old, "page_latency_us": {"p50": 250.0}, "miss_rate": {"5ms": 0.0}}
    assert not b4_control.is_backlogged(fine)


def test_backlogged_reps_never_report_latency_or_late_fraction() -> None:
    key = "timed_lead8ms@100Hz"
    good = _paced_row(key, 0)
    # the 81f2735 failure mode: the engine lags by seconds, no command is late, and visible
    # - recv is lead + lag
    lag = _paced_row(
        key,
        1,
        page_latency_us={"p50": 5.2e6, "p99": 2.7e7},
        miss_rate={"0ms": 1.0, "1ms": 1.0, "5ms": 1.0},
        visible_latency_us={},
        visible_minus_recv_us={"p50": 5.2e6, "p99": 2.7e7},
        apply_delay_us={"p50": 5.2e6, "p99": 2.7e7},
    )
    late = _paced_row(
        key,
        2,
        status={"applied": 5400, "late_applied": 600},
        late_fraction=0.1,
        exact_fraction=0.9,
        visible_latency_us={"p50": 2500.0, "p99": 6000.0},
    )
    assert lag["backlogged"] and not good["backlogged"] and not late["backlogged"]
    by = b4_control.paced_by_key([good, lag, late], [(100, "timed", "8e-3")], True, 65.0)
    e = by[key]
    assert e["reps"] == 3 and e["backlogged_reps"] == 1
    assert e["visible_latency_p99_us_worst"] == 6000.0  # not the backlog's 2.7e7
    assert e["late_fraction"] == 600 / 12000  # over the two valid repetitions
    assert e["exact_fraction"] == (1.0 * 6000 + 0.9 * 6000) / 12000
    assert e["commands_valid"] == 12000 and e["commands_total"] == 18000
    # wall-clock figures keep every repetition: the backlog shows up as apply delay
    assert e["apply_delay_p99_us_worst"] == 2.7e7
    # every repetition backlogged: no latency and no fractions at all
    only = b4_control.paced_by_key([lag], [(100, "timed", "8e-3")], True, 65.0)[key]
    assert only["visible_latency_p99_us_worst"] is None
    assert only["late_fraction"] is None and only["exact_fraction"] is None
    assert only["lateness_p99_samples_median"] is None


def test_timed_metrics_apply_delay_and_virtual_margin() -> None:
    rate = Fraction(1_000_000)
    page = 1024
    t0, d_lag = 10_000_000_000, 1_000_000
    rx = ("rx000.rx0", "rx001.rx0")

    def rec(
        status: str, t_req: Fraction, k: int | None, recv: int, visible: int | None, late: int = 0
    ) -> SimpleNamespace:
        return SimpleNamespace(
            status=status,
            t_req=t_req,
            k_applied={} if k is None else dict.fromkeys(rx, k),
            lateness_samples={} if k is None else dict.fromkeys(rx, late),
            t_wall_recv_ns=recv,
            t_wall_visible_ns=visible,
        )

    # switch at sample 5000: its page [4096, 5120) is released at t0 + 5.12 ms + D_lag and the
    # record emitted 0.3 ms later; requested at t = 5 ms, received when the pacer's schedule
    # stood at virtual 1 ms (wall t0 + D_lag + 1 ms): margin 4 ms
    a = rec(
        "applied", Fraction(5, 1000), 5000, t0 + d_lag + 1_000_000, t0 + 5_120_000 + d_lag + 300_000
    )
    # applied 3 samples late, on the page ending at 3072, emitted 2 ms after its release;
    # requested 53 us behind the schedule at its receipt
    b = rec(
        "late_applied",
        Fraction(2047, 1_000_000),
        2050,
        t0 + d_lag + 2_100_000,
        t0 + 3_072_000 + d_lag + 2_000_000,
        late=3,
    )
    c = rec("rejected", Fraction(9, 1000), None, t0 + d_lag, None)
    m = b4_control.timed_metrics([a, b, c], page=page, rate=rate, t0_ns=t0, d_lag_ns=d_lag)
    assert m["commands"] == 3
    assert m["late_fraction"] == 1 / 3 and m["exact_fraction"] == 1 / 3
    assert (m["apply_delay_us"]["min"], m["apply_delay_us"]["max"]) == (300.0, 2000.0)
    assert m["apply_delay_us"]["n"] == 2  # the rejected command has no apply delay
    assert (m["virtual_margin_us"]["min"], m["virtual_margin_us"]["max"]) == (-53.0, 9000.0)
    assert m["virtual_margin_us"]["p50"] == 4000.0
    # the last page of a run is capped at its end sample
    end = b4_control.timed_metrics([a], page=page, rate=rate, t0_ns=t0, d_lag_ns=d_lag, end=5050)
    assert end["apply_delay_us"]["p50"] == 300.0 + (5120 - 5050)
    # asap commands: no virtual margin
    asap = b4_control.timed_metrics(
        [a], page=page, rate=rate, t0_ns=t0, d_lag_ns=d_lag, timed=False
    )
    assert asap["virtual_margin_us"] == {} and asap["apply_delay_us"]["p50"] == 300.0
    assert b4_control.timed_metrics([], page=page, rate=rate, t0_ns=t0, d_lag_ns=d_lag) == {
        "commands": 0,
        "late_fraction": None,
        "exact_fraction": None,
        "virtual_margin_us": {},
        "apply_delay_us": {},
    }


def test_summarize_b4_paced_reps(tmp_path: Path) -> None:
    key = "asap@100Hz"
    new = _paced_row(key, 0, mode="asap")
    old = _paced_row(
        key,
        1,
        mode="asap",
        page_latency_us={"p50": 1.4e6},
        miss_rate={"5ms": 1.0},
        visible_latency_us={"p50": 3.5e3, "p99": 5.6e3},
    )
    for k in (
        "backlogged",
        "backlog_reasons",
        "page_s",
        "late_fraction",
        "exact_fraction",
        "visible_minus_recv_us",
        "apply_delay_us",
        "virtual_margin_us",
        "send_to_schedule_rtt_us",
    ):
        old.pop(k)  # a row as 81f2735 wrote it
    (tmp_path / "paced.json").write_text(json.dumps([new, old]), encoding="utf-8")
    reps = summarize.b4_paced_reps(tmp_path)
    assert reps is not None and [r["backlogged"] for r in reps] == [False, True]
    assert [r["backlog_judged"] for r in reps] == [False, True]
    assert reps[0]["visible_latency_p99_us"] == 4000.0
    assert reps[1]["visible_latency_p99_us"] is None  # that 5.6 ms was the backlog's
    assert reps[1]["backlog_reasons"] and reps[1]["late_fraction"] is None
    assert reps[1]["apply_delay_p99_us"] is None
    assert summarize.b4_paced_reps(tmp_path / "missing") is None
