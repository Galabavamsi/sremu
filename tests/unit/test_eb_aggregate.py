"""E-B aggregation (experiments/eb_aggregate.py): D24 labels, Es/N0 and geometry columns,
multi-panel VA-5 and the non-strict mode of the unattended driver (synthetic sweep roots)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from experiments import common, eb_aggregate

_GIT_STUB = {
    "commit": "test",
    "dirty": False,
    "dirty_patch_sha256": None,
    "dirty_patch_path": None,
    "reason": "test stub",
    "code": None,
}


@pytest.fixture(autouse=True)
def git_stub(monkeypatch):
    monkeypatch.setattr(common, "_git_cache", dict(_GIT_STUB))


def _eb2_summary(**over: Any) -> dict[str, Any]:
    s = {
        "alphabet": "lab",
        "los": "present",
        "guard_s": "1/2000",
        "jitter_s": "1/1000",
        "semantics": "timed",
        "lead_s": "1/2000",
        "n_packets": 2000,
        "packet_errors": 3,
        "per": 0.0015,
        "per_ci_low": 0.0003,
        "per_ci_high": 0.0044,
        "mean_evm_db": -20.0,
        "n_late": 900,
        "record_status_counts": {"applied": 2000},
        "es_n0_db": {"rx1.rx0": 25.0, "rx2.rx0": 23.4},
        "analytic_expectation": {"p_switch_in_packet": 0.45, "rule": "P(L > lead + G/2)"},
    }
    s.update(over)
    return s


def _write_root(root: Path, entries: list[dict[str, Any]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    idx = []
    for i, e in enumerate(entries):
        pid = f"p{i:05d}"
        (root / pid).mkdir(exist_ok=True)
        idx.append({"point_id": pid, "dir": pid, **e})
    (root / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    return root


def _rows(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_eb2_labels_and_columns(tmp_path: Path) -> None:
    old = _eb2_summary(semantics="legacy", lead_s=None, legacy_lost=35)  # former alias
    new = _eb2_summary(
        semantics="tick-based",
        lead_s=None,
        n_panels=6,
        tick_overwritten=210,
        record_status_counts={"applied": 11790, "superseded": 210},
        es_n0_target_db=15.0,
        scenario="scenarios/studies/lab_arc.yaml",
        geometry_label="modelled on the lab arrangement (SIM, nominal geometry)",
    )
    r1 = _write_root(tmp_path / "eb2_lab", [{"status": "ok", "summary": old}])
    r2 = _write_root(tmp_path / "eb2_lab_es15", [{"status": "ok", "summary": new}])
    rows = eb_aggregate.aggregate_eb2([r1, r2, tmp_path / "missing"], tmp_path / "out")
    assert len(rows) == 2
    csv_rows = _rows(tmp_path / "out" / "eb2_per.csv")
    assert [r["semantics"] for r in csv_rows] == ["tick-based", "tick-based"]
    assert "legacy" not in (tmp_path / "out" / "eb2_per.csv").read_text(encoding="utf-8")
    meta = (tmp_path / "out" / "meta.json").read_text(encoding="utf-8")
    assert "legacy" not in meta.lower()
    assert rows[0]["es_n0_target_db"] == 25.0 and rows[1]["es_n0_target_db"] == 15.0
    assert rows[0]["tick_overwritten_messages"] == 35.0
    assert rows[1]["tick_overwritten_messages"] == 35.0  # 210 panel commands / 6 panels
    assert rows[1]["superseded_messages"] == 35.0
    assert rows[0]["geometry_label"] == "lab-like SIM geometry (not the lab's)"
    assert rows[1]["scenario"] == "scenarios/studies/lab_arc.yaml"
    assert [r["sweep"] for r in rows] == ["eb2_lab", "eb2_lab_es15"]


def test_blocked_idealization_note(tmp_path: Path) -> None:
    """LOS-blocked rows whose summary carries design.blocker (non-default geometry) add the
    'LOS removed, RIS hops unattenuated' assumption to meta.json; other rows do not."""
    blk = {
        "vertices_m": [],
        "los_crossed": {"rx1.rx0": True, "rx2.rx0": True},
        "n_cells": 240,
        "tx_cell_segments_crossed": 152,
        "cell_rx_segments_crossed": {"rx1.rx0": 0, "rx2.rx0": 0},
    }
    arc = {
        "scenario": "scenarios/studies/lab_arc.yaml",
        "geometry_label": "modelled on the lab arrangement (SIM, nominal geometry)",
        "n_panels": 6,
    }
    plain = _write_root(tmp_path / "plain", [{"status": "ok", "summary": _eb2_summary()}])
    eb_aggregate.aggregate_eb2([plain], tmp_path / "o1")
    assert "unattenuated" not in (tmp_path / "o1" / "meta.json").read_text(encoding="utf-8")
    rows = [
        {"status": "ok", "summary": _eb2_summary(los="blocked", design={"blocker": blk}, **arc)},
        {"status": "ok", "summary": _eb2_summary(los="present", design={}, **arc)},
    ]
    root = _write_root(tmp_path / "eb2_lab", rows)
    eb_aggregate.aggregate_eb2([root], tmp_path / "o2")
    meta = json.loads((tmp_path / "o2" / "meta.json").read_text(encoding="utf-8"))
    notes = [a for a in meta["assumptions"] if "unattenuated" in a]
    assert len(notes) == 1 and "152/240 Tx->cell segments" in notes[0]
    s1 = _eb1_summary("step", 3576, 3576)
    s1.update(los="blocked", **arc)
    s1["design"]["blocker"] = blk
    va5 = eb_aggregate.aggregate_eb1(_eb1_root(tmp_path / "e1", [s1]), tmp_path / "o3")
    assert va5["pass"] is True
    meta1 = json.loads((tmp_path / "o3" / "meta.json").read_text(encoding="utf-8"))
    assert sum("152/240" in a for a in meta1["assumptions"]) == 1


def test_eb2_strict_and_lenient(tmp_path: Path) -> None:
    root = _write_root(
        tmp_path / "eb2_lab",
        [
            {"status": "ok", "summary": _eb2_summary()},
            {"status": "failed", "summary": None, "error": "RuntimeError: boom"},
        ],
    )
    with pytest.raises(RuntimeError, match="boom"):
        eb_aggregate.aggregate_eb2([root], tmp_path / "strict")
    rows = eb_aggregate.aggregate_eb2([root], tmp_path / "lenient", strict=False)
    assert len(rows) == 1
    failed = json.loads((tmp_path / "lenient" / "failed_points.json").read_text(encoding="utf-8"))
    assert failed == [{"sweep": "eb2_lab", "point_id": "p00001", "error": "RuntimeError: boom"}]


def _eb1_summary(transition: str, k_app: int, k_first: int, **va5_over: Any) -> dict[str, Any]:
    va5 = {
        "k_applied": k_app,
        "k_expected_to_ticks": k_app,
        "k_first_differing": k_first,
        "first_differing_minus_k_applied": k_first - k_app,
        "tx_active_at_switch": True,
        "symbols_before_first_affected": 100,
        "symbols_before_bitwise_identical": True,
        "pass_step_inside_packet": (k_first == k_app) if transition == "step" else None,
    }
    va5.update(va5_over)
    return {
        "alphabet": "lab",
        "los": "present",
        "transition": transition,
        "delta": "1/4",
        "t_sw_s": "1/250",
        "per_packet_evm_db": -10.0,
        "per_packet_evm_pct": 30.0,
        "bit_errors": 0,
        "ber": 0.0,
        "ref_evm_db": -25.0,
        "ref_bit_errors": 0,
        "es_n0_db": {"rx1.rx0": 25.0, "rx2.rx0": 24.8},
        "tx_dbm": -53.4,
        "va5": va5,
        "design": {
            "rx1_total_change_db": -10.5,
            "rx1_total_rotation_deg": -19.2,
            "rx1_ris_term_change_db": -12.0,
            "rx1_ris_term_rotation_deg": -30.0,
        },
    }


def _eb1_root(tmp_path: Path, summaries: list[dict[str, Any]], failed: bool = False) -> Path:
    entries: list[dict[str, Any]] = [{"status": "ok", "summary": s} for s in summaries]
    if failed:
        entries.append({"status": "failed", "summary": None, "error": "ValueError: x"})
    root = _write_root(tmp_path / "eb1", entries)
    for i in range(len(summaries)):
        (root / f"p{i:05d}" / "symbols.csv").write_text(
            "symbol,kind,t_center_s,evm_db,evm_ref_db\n0,preamble,0.003002,-25.0,-25.0\n",
            encoding="utf-8",
        )
    return root


def test_eb1_va5_single_panel(tmp_path: Path) -> None:
    root = _eb1_root(
        tmp_path, [_eb1_summary("step", 3576, 3576), _eb1_summary("linear20us", 3576, 3577)]
    )
    va5 = eb_aggregate.aggregate_eb1(root, tmp_path / "out")
    assert va5["pass"] is True
    assert va5["n_step_switches_inside_packet"] == 1 and va5["linear_first_minus_applied"] == [1]


def test_eb1_va5_multi_panel_mismatch_fails(tmp_path: Path) -> None:
    good = _eb1_summary(
        "step", 3576, 3576, n_panels=6, n_panels_changed=6, all_panels_k_applied_equal_expected=True
    )
    bad = _eb1_summary(
        "step",
        3576,
        3576,
        n_panels=6,
        n_panels_changed=6,
        all_panels_k_applied_equal_expected=False,
    )
    va5 = eb_aggregate.aggregate_eb1(_eb1_root(tmp_path / "a", [good]), tmp_path / "a_out")
    assert va5["pass"] is True
    va5 = eb_aggregate.aggregate_eb1(_eb1_root(tmp_path / "b", [good, bad]), tmp_path / "b_out")
    assert va5["pass"] is False and va5["all_panels_k_applied_equal_expected"] is False


def test_eb1_lenient_failure_blocks_va5(tmp_path: Path) -> None:
    root = _eb1_root(tmp_path, [_eb1_summary("step", 3576, 3576)], failed=True)
    with pytest.raises(RuntimeError):
        eb_aggregate.aggregate_eb1(root, tmp_path / "strict")
    va5 = eb_aggregate.aggregate_eb1(root, tmp_path / "lenient", strict=False)
    assert va5["n_failed_points"] == 1 and va5["pass"] is False
