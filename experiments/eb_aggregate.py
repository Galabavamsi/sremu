"""Aggregation of the E-B sweeps (F9 data, Table data, VA-5 acceptance).

Paper-facing labels follow D24: the untimed baseline is "tick-based" (point summaries
written with the former alias ``legacy`` are relabelled). ``strict=True`` (the default, as
used by ``experiments.run_sweeps``) raises on a failed point; ``strict=False`` (the
unattended ``python -m experiments.eb`` driver) aggregates the points that succeeded and
records the failed ones, and VA-5 then does not pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.common import RunRecord, read_csv, read_json, write_csv, write_json, write_meta

_SEMANTICS_LABEL = {"legacy": "tick-based"}
_DEFAULT_LABEL = "lab-like SIM geometry (not the lab's)"


def _points(
    root: Path, strict: bool = True, failed: list[dict[str, Any]] | None = None
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for e in read_json(root / "index.json"):
        if e["status"] != "ok":
            if strict:
                raise RuntimeError(f"{root}: {e['point_id']} failed: {e.get('error')}")
            if failed is not None:
                failed.append(
                    {"sweep": root.name, "point_id": e["point_id"], "error": e.get("error")}
                )
            continue
        out.append((e, e["summary"]))
    return out


def _label(s: dict[str, Any]) -> str:
    return str(s.get("geometry_label") or _DEFAULT_LABEL)


def _blocked_note(s: dict[str, Any]) -> str | None:
    """Assumption for LOS-blocked points of a geometry whose plate crosses RIS hops (the
    point summary's ``design.blocker``, written by ``experiments.eb`` for non-default
    geometries); None otherwise."""
    b = (s.get("design") or {}).get("blocker")
    if not b:
        return None
    return (
        f"{_label(s)}, LOS blocked: one opaque plate across both LOS segments; occluders act "
        "on the LOS only, so the RIS hops are unattenuated although the plate crosses "
        f"{b['tx_cell_segments_crossed']}/{b['n_cells']} Tx->cell segments (idealized regime "
        "'LOS removed, RIS hops unattenuated') [ENG]"
    )


def _add_note(notes: list[str], s: dict[str, Any]) -> None:
    n = _blocked_note(s)
    if n is not None and n not in notes:
        notes.append(n)


def aggregate_eb1(root: str | Path, out_dir: str | Path, *, strict: bool = True) -> dict[str, Any]:
    """E-B1: per-point table, per-symbol EVM traces (F9a) and the VA-5 verdict."""
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = RunRecord(out, 20261010, mode="analysis", patch_root=out)
    rows, traces = [], []
    failed: list[dict[str, Any]] = []
    labels: list[str] = []
    notes: list[str] = []
    for e, s in _points(root, strict, failed):
        v = s["va5"]
        if _label(s) not in labels:
            labels.append(_label(s))
        _add_note(notes, s)
        rows.append(
            {
                "alphabet": s["alphabet"],
                "los": s["los"],
                "transition": s["transition"],
                "delta": s["delta"],
                "t_sw_s": s["t_sw_s"],
                "evm_db": s["per_packet_evm_db"],
                "evm_pct": s["per_packet_evm_pct"],
                "bit_errors": s["bit_errors"],
                "ber": s["ber"],
                "ref_evm_db": s["ref_evm_db"],
                "ref_bit_errors": s["ref_bit_errors"],
                "es_n0_rx1_db": s["es_n0_db"]["rx1.rx0"],
                "tx_dbm": s["tx_dbm"],
                "k_applied": v["k_applied"],
                "k_first_differing": v["k_first_differing"],
                "first_minus_applied": v["first_differing_minus_k_applied"],
                "tx_active_at_switch": v["tx_active_at_switch"],
                "before_bitwise_identical": v["symbols_before_bitwise_identical"],
                "va5_pass_step_inside": v["pass_step_inside_packet"],
                "n_panels": s.get("n_panels", 1),
                "n_panels_changed": v.get("n_panels_changed", 1),
                "all_panels_k_applied_equal_expected": v.get(
                    "all_panels_k_applied_equal_expected",
                    v["k_applied"] == v["k_expected_to_ticks"],
                ),
                "rx1_total_change_db": s["design"]["rx1_total_change_db"],
                "rx1_total_rotation_deg": s["design"]["rx1_total_rotation_deg"],
                "rx1_ris_term_change_db": s["design"]["rx1_ris_term_change_db"],
                "rx1_ris_term_rotation_deg": s["design"]["rx1_ris_term_rotation_deg"],
                "scenario": s.get("scenario"),
                "geometry_label": _label(s),
            }
        )
        for r in read_csv(root / e["dir"] / "symbols.csv"):
            traces.append(
                {
                    "alphabet": s["alphabet"],
                    "los": s["los"],
                    "transition": s["transition"],
                    "delta": s["delta"],
                    **r,
                }
            )
    write_csv(out / "eb1_points.csv", rows)
    write_csv(out / "eb1_traces.csv", traces)
    inside = [r for r in rows if r["va5_pass_step_inside"] is not None]
    va5: dict[str, Any] = {
        "n_points": len(rows),
        "n_failed_points": len(failed),
        "failed_points": failed,
        "n_step_switches_inside_packet": len(inside),
        "n_pass": sum(1 for r in inside if r["va5_pass_step_inside"]),
        "all_before_bitwise_identical": all(r["before_bitwise_identical"] for r in rows),
        "all_panels_k_applied_equal_expected": all(
            r["all_panels_k_applied_equal_expected"] for r in rows
        ),
        "linear_first_minus_applied": sorted(
            {
                r["first_minus_applied"]
                for r in rows
                if r["transition"] != "step" and r["tx_active_at_switch"]
            }
        ),
        "criterion": "first RX1 sample differing from the no-switch run == to_ticks(t_sw + "
        "tau_RIS->RX1) == engine k_applied (step switch while the TX is active; with several "
        "panels, the earliest over the panels whose states change, and every panel's "
        "k_applied equals its own to_ticks(t_sw + tau_panel->RX1)); every sample before it "
        "bitwise identical",
        "geometry_labels": labels,
    }
    va5["pass"] = (
        va5["n_failed_points"] == 0
        and va5["n_step_switches_inside_packet"] > 0
        and va5["n_pass"] == va5["n_step_switches_inside_packet"]
        and va5["all_before_bitwise_identical"]
        and va5["all_panels_k_applied_equal_expected"]
    )
    write_json(out / "va5.json", va5)
    write_meta(
        out,
        experiment="E-B1",
        data_classes=["SIM-sremu", "ASSUMED"],
        description="E-B1 aggregation: per-symbol EVM traces (F9a), per-packet EVM/BER, VA-5",
        assumptions=[
            "lab |Gamma| ASSUMED (alphabet 'lab')",
            "transition path complex-linear [ENG]",
            *(labels or [_DEFAULT_LABEL]),
            *notes,
        ],
        files={
            "eb1_points.csv": "one row per (alphabet, LOS, transition, delta)",
            "eb1_traces.csv": "per-symbol EVM (dB) with and without the switch",
            "va5.json": "VA-5 acceptance",
        },
        extra={"sweep_root": str(root), "failed_points": failed},
    )
    run.finish()
    return va5


def aggregate_eb2(
    roots: Sequence[str | Path], out_dir: str | Path, *, strict: bool = True
) -> list[dict[str, Any]]:
    """E-B2: PER with Clopper-Pearson 95 % CI and mean EVM per point (F9b data)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = RunRecord(out, 20261010, mode="analysis", patch_root=out)
    rows = []
    failed: list[dict[str, Any]] = []
    labels: list[str] = []
    notes: list[str] = []
    for root in roots:
        root = Path(root)
        if not (root / "index.json").exists():
            continue
        for _e, s in _points(root, strict, failed):
            if _label(s) not in labels:
                labels.append(_label(s))
            _add_note(notes, s)
            n_pan = int(s.get("n_panels", 1))
            tick = s.get("tick_overwritten", s.get("legacy_lost"))
            sup = s["record_status_counts"].get("superseded", 0)
            rows.append(
                {
                    "alphabet": s["alphabet"],
                    "los": s["los"],
                    "es_n0_target_db": s.get("es_n0_target_db", 25.0),
                    "semantics": _SEMANTICS_LABEL.get(s["semantics"], s["semantics"]),
                    "lead_s": s["lead_s"],
                    "guard_s": s["guard_s"],
                    "jitter_s": s["jitter_s"],
                    "n_packets": s["n_packets"],
                    "packet_errors": s["packet_errors"],
                    "per": s["per"],
                    "per_ci_low": s["per_ci_low"],
                    "per_ci_high": s["per_ci_high"],
                    "mean_evm_db": s["mean_evm_db"],
                    "n_late": s["n_late"],
                    "tick_overwritten_messages": None if tick is None else tick / n_pan,
                    "superseded_messages": sup / n_pan,
                    "es_n0_rx1_db": s["es_n0_db"]["rx1.rx0"],
                    "es_n0_rx2_db": s["es_n0_db"]["rx2.rx0"],
                    "p_switch_in_packet_derived": s["analytic_expectation"]["p_switch_in_packet"],
                    "n_panels": n_pan,
                    "scenario": s.get("scenario"),
                    "geometry_label": _label(s),
                    "sweep": root.name,
                }
            )
    write_csv(out / "eb2_per.csv", rows)
    if failed:
        write_json(out / "failed_points.json", failed)
    write_meta(
        out,
        experiment="E-B2",
        data_classes=["SIM-sremu", "ASSUMED"],
        description="E-B2 aggregation: PER (Clopper-Pearson 95 %) and mean EVM vs latency jitter, "
        "guard and command semantics (F9b)",
        assumptions=[
            "command arrival = issue + keyed latency draw (0.2 ms + U[0, J]), one per guard",
            "timed: switch at max(requested, arrival); asap: at arrival; tick-based untimed "
            "semantics: next 2 ms tick after arrival, one pending slot per panel (overwrite)",
            "TX power calibrated for Es/N0 = es_n0_target_db at RX1 under its own optimum "
            "(25 dB; 15 dB = pre-declared sensitivity of the LOS-present slice)",
            "with several panels: one command per panel per guard (same latency draw); "
            "tick_overwritten_messages and superseded_messages count controller messages",
            "p_switch_in_packet_derived: [DERIVED] expectation, not a claim",
            *(labels or [_DEFAULT_LABEL]),
            *notes,
        ],
        files={
            "eb2_per.csv": "one row per (alphabet, LOS, Es/N0, semantics, lead, guard, J)",
            **({"failed_points.json": "points that failed (not aggregated)"} if failed else {}),
        },
        extra={"sweep_roots": [str(r) for r in roots], "failed_points": failed},
    )
    run.finish()
    return rows
