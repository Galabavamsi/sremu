"""B4 lockstep check: a command whose switch lies after the end of the run is not a failure.

The B4 lockstep run of 20261011T074442Z (AWS c7i.8xlarge) reported ``lockstep_check_ok =
false`` only because the last barrier's command at 10 kHz (barrier one period before the end,
latency L_min + U[0, L_min] >= one period) switches after the end of the 1 s run and so stays
``scheduled``. These tests pin the corrected definition (exact count for new rows, a bound from
the controller's latency model for older rows) and the offline recomputation. They never
assert a timing.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bench import b4_control


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
