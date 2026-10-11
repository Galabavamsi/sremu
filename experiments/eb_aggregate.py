"""Aggregation of the E-B sweeps (F9 data, Table data, VA-5 acceptance)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.common import RunRecord, read_csv, read_json, write_csv, write_json, write_meta


def _points(root: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for e in read_json(root / "index.json"):
        if e["status"] != "ok":
            raise RuntimeError(f"{root}: {e['point_id']} failed: {e.get('error')}")
        out.append((e, e["summary"]))
    return out


def aggregate_eb1(root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """E-B1: per-point table, per-symbol EVM traces (F9a) and the VA-5 verdict."""
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = RunRecord(out, 20261010, mode="analysis", patch_root=out)
    rows, traces = [], []
    va5_rows = []
    for e, s in _points(root):
        v = s["va5"]
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
                "rx1_total_change_db": s["design"]["rx1_total_change_db"],
                "rx1_total_rotation_deg": s["design"]["rx1_total_rotation_deg"],
                "rx1_ris_term_change_db": s["design"]["rx1_ris_term_change_db"],
                "rx1_ris_term_rotation_deg": s["design"]["rx1_ris_term_rotation_deg"],
            }
        )
        va5_rows.append(rows[-1])
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
    inside = [r for r in va5_rows if r["va5_pass_step_inside"] is not None]
    va5 = {
        "n_step_switches_inside_packet": len(inside),
        "n_pass": sum(1 for r in inside if r["va5_pass_step_inside"]),
        "all_before_bitwise_identical": all(r["before_bitwise_identical"] for r in va5_rows),
        "linear_first_minus_applied": sorted(
            {
                r["first_minus_applied"]
                for r in va5_rows
                if r["transition"] != "step" and r["tx_active_at_switch"]
            }
        ),
        "criterion": "first RX1 sample differing from the no-switch run == to_ticks(t_sw + "
        "tau_RIS->RX1) == engine k_applied (step switch while the TX is active); every "
        "sample before it bitwise identical",
    }
    va5["pass"] = (
        va5["n_pass"] == va5["n_step_switches_inside_packet"]
        and va5["all_before_bitwise_identical"]
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
            "lab-like SIM geometry, not the lab's",
        ],
        files={
            "eb1_points.csv": "one row per (alphabet, LOS, transition, delta)",
            "eb1_traces.csv": "per-symbol EVM (dB) with and without the switch",
            "va5.json": "VA-5 acceptance",
        },
        extra={"sweep_root": str(root)},
    )
    run.finish()
    return va5


def aggregate_eb2(roots: Sequence[str | Path], out_dir: str | Path) -> list[dict[str, Any]]:
    """E-B2: PER with Clopper-Pearson 95 % CI and mean EVM per point (F9b data)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = RunRecord(out, 20261010, mode="analysis", patch_root=out)
    rows = []
    for root in roots:
        root = Path(root)
        if not (root / "index.json").exists():
            continue
        for _e, s in _points(root):
            rows.append(
                {
                    "alphabet": s["alphabet"],
                    "los": s["los"],
                    "semantics": s["semantics"],
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
                    "legacy_lost": s["legacy_lost"],
                    "superseded": s["record_status_counts"].get("superseded", 0),
                    "es_n0_rx1_db": s["es_n0_db"]["rx1.rx0"],
                    "es_n0_rx2_db": s["es_n0_db"]["rx2.rx0"],
                    "p_switch_in_packet_derived": s["analytic_expectation"]["p_switch_in_packet"],
                }
            )
    write_csv(out / "eb2_per.csv", rows)
    write_meta(
        out,
        experiment="E-B2",
        data_classes=["SIM-sremu", "ASSUMED"],
        description="E-B2 aggregation: PER (Clopper-Pearson 95 %) and mean EVM vs latency jitter, "
        "guard and command semantics (F9b)",
        assumptions=[
            "command arrival = issue + keyed latency draw (0.2 ms + U[0, J])",
            "timed: switch at max(requested, arrival); asap: at arrival; legacy: next 2 ms "
            "tick after arrival, single-slot overwrite",
            "p_switch_in_packet_derived: [DERIVED] expectation, not a claim",
        ],
        files={"eb2_per.csv": "one row per (alphabet, LOS, semantics, lead, guard, J)"},
        extra={"sweep_roots": [str(r) for r in roots]},
    )
    run.finish()
    return rows
