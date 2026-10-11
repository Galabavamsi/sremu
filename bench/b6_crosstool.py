"""B6: cross-tool context, time per RIS channel evaluation on one scene (section 12.2, RQ-44).

Usage::

    python -m bench.b6_crosstool [--sizes 256,1024,4096] [--no-matlab] [--matlab EXE] [--out DIR]

Scene (SIM): one link at 3.5 GHz; a square panel of N in {256, 1024, 4096} cells at 42.8 mm
pitch centred at the origin, facing +x; Tx 5 m at -30 deg and Rx 5 m at +20 deg (angles
from the normal in the x-y plane); one random 1-bit configuration (Gamma = +1/-1, the same
for every tool). One "channel evaluation" is h = LOS + RIS sum for that configuration.

Tools, all timed on this host in one session:

* **sremu** (``numba_cpu`` and ``numpy_ref``): ``ScenePropagation.snapshot(t)`` +
  ``combine`` (full evaluation for new positions, through the Python API) and ``combine``
  alone (configuration change, geometry cached);
* **MATLAB R2026a** (``bench/b6_matlab.m`` via ``matlab -batch``): the per-element K1 sum as
  an explicit ``for`` loop over the cells (a typical per-element script), the same sum
  vectorized, the toolbox-primitive path ``golden_util_k1`` (``tools/matlab``: rangeangle,
  ``phased.CosineAntennaElement``, fspl, plus an independent hand evaluation) and one
  ``phased.RISReflector`` step.

**Scope caveats** (also in ``meta.json`` and the ``scope`` column of ``context_table.csv``):
the tools do not all evaluate the same model. sremu and the three MATLAB K1 paths compute
the same near-field per-cell K1 sum (their |h| agreement is reported as a check);
``RISReflector`` is a far-field plane-wave model with isotropic unit cells and no cell-area
factor, timed only as tool context. MATLAB times exclude MATLAB start-up (``timeit`` inside
one session); sremu times exclude numba compilation (warm-up) and include its Python
dispatch. The literature numbers (ACHEM 82-167 us end to end, Colosseum 1 ms snapshots, ORIX
10 ms-1 s loop, TRACE 5.2 ms control RTT) are quoted with their own hardware as REPORTED
context, never as a speed comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

import bench
from bench._common import REPO, BenchRun, time_calls

SUITE = "b6"
FC_HZ = 3.5e9
PITCH_M = 0.0428
MATLAB_EXE = Path(r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe")
SCOPE = {
    "sremu_numba_cpu_snapshot+combine": (
        "near-field per-cell K1 sum + LOS for new positions (sremu API, numba backend)"
    ),
    "sremu_numba_cpu_combine_only": "configuration change, geometry cached (numba backend)",
    "sremu_numpy_ref_snapshot+combine": "same as numba, NumPy reference backend",
    "sremu_numpy_ref_combine_only": "configuration change, geometry cached (NumPy backend)",
    "matlab_hand_loop": "same K1 sum + LOS as an explicit per-element for-loop (MATLAB script)",
    "matlab_hand_vec": "same K1 sum + LOS, vectorized MATLAB (no toolbox calls)",
    "matlab_toolbox_golden_util_k1": (
        "same K1 sum from toolbox primitives (rangeangle, CosineAntennaElement, fspl) plus an "
        "independent hand evaluation and LOS (does more work than one evaluation)"
    ),
    "matlab_risreflector_step": (
        "phased.RISReflector step: far-field plane wave, isotropic unit cells, no cell area "
        "(a different model; tool context only)"
    ),
}
LITERATURE = [
    {
        "tool": "ACHEM (arXiv 2604.04742)",
        "quantity": "end-to-end latency",
        "value": "82-167 us",
        "hardware": "as reported by the authors",
        "data_class": "REPORTED",
    },
    {
        "tool": "Colosseum + RIS (Tsampazi 2025)",
        "quantity": "channel snapshot period",
        "value": "1 ms",
        "hardware": "Colosseum FPGA channel emulator",
        "data_class": "REPORTED",
    },
    {
        "tool": "ORIX / GoSimRIS",
        "quantity": "control loop",
        "value": "10 ms-1 s",
        "hardware": "as reported",
        "data_class": "REPORTED",
    },
    {
        "tool": "TRACE (arXiv 2608.11798)",
        "quantity": "control round trip (UDP)",
        "value": "5.2 ms",
        "hardware": "as reported",
        "data_class": "REPORTED",
    },
]


def scene(sizes: list[int], seed: int = 7) -> dict[str, Any]:
    """The B6 scene (positions, carrier, panels and their 1-bit configurations)."""
    rng = np.random.default_rng(seed)
    a_tx, a_rx = math.radians(-30.0), math.radians(20.0)
    panels = []
    for n in sizes:
        side = round(math.sqrt(n))
        if side * side != n:
            raise ValueError(f"B6 sizes must be squares, got {n}")
        bits = rng.integers(0, 2, n)
        panels.append(
            {
                "n_u": side,
                "n_v": side,
                "pitch_m": PITCH_M,
                "bits": bits.tolist(),
                "gamma": [1.0 if b == 0 else -1.0 for b in bits],
            }
        )
    return {
        "fc_hz": FC_HZ,
        "tx_m": [5 * math.cos(a_tx), 5 * math.sin(a_tx), 0.0],
        "rx_m": [5 * math.cos(a_rx), 5 * math.sin(a_rx), 0.0],
        "panels": panels,
    }


def scenario_dict(sc: dict[str, Any], pn: dict[str, Any]) -> dict[str, Any]:
    """The sremu scenario mapping of one B6 panel size."""
    n = pn["n_u"] * pn["n_v"]
    return {
        "schema_version": "1.0",
        "name": f"b6_n{n}",
        "seed": 1,
        "nodes": [
            {
                "id": "tx",
                "position_m": sc["tx_m"],
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": "tx0",
                        "direction": "tx",
                        "fc_hz": FC_HZ,
                        "rate_sps": "1e6",
                        "occupied_bw_hz": 8e5,
                    }
                ],
            },
            {
                "id": "rx",
                "position_m": sc["rx_m"],
                "antenna": {"pattern": "isotropic"},
                "streams": [{"name": "rx0", "direction": "rx", "fc_hz": FC_HZ, "rate_sps": "1e6"}],
            },
        ],
        "ris": [
            {
                "id": "ris0",
                "pose": {"center_m": [0, 0, 0], "normal": [1, 0, 0], "u_axis": [0, 1, 0]},
                "grid": {
                    "n_u": pn["n_u"],
                    "n_v": pn["n_v"],
                    "pitch_u_m": PITCH_M,
                    "pitch_v_m": PITCH_M,
                },
                "states": {"library": "ideal_antipodal"},
                "initial": {"fill": 0},
            }
        ],
        "run": {"duration_s": "1"},
    }


# =========================================================================== sremu


def sremu_rows(sc: dict[str, Any], run: BenchRun | None = None) -> list[dict[str, Any]]:
    """Time sremu's full evaluation and its configuration-only evaluation per size."""
    from bench._engine_point import _resolve
    from bench._scenes import ris_path_check
    from sremu.contracts import PanelConfig
    from sremu.physics.propagation import from_scenario

    out = []
    for pn in sc["panels"]:
        n = pn["n_u"] * pn["n_v"]
        res = _resolve(scenario_dict(sc, pn))
        sha = run.save_scenario(f"b6_n{n}", res) if run is not None else str(res.sha256())
        for be in ("numba_cpu", "numpy_ref"):
            model = from_scenario(res.scenario.model_dump(mode="python"), backend=be)
            cfg = {"ris0": PanelConfig("ris0", np.asarray(pn["bits"], dtype=np.int16))}
            ts = iter(Fraction(i, 1000) for i in range(10**9))
            full = time_calls(
                lambda m=model, c=cfg, ts=ts: m.combine(m.snapshot(next(ts)), c), min_time_s=0.3
            )
            snap = model.snapshot(Fraction(0))
            comb = time_calls(lambda m=model, s=snap, c=cfg: m.combine(s, c), min_time_s=0.3)
            h = model.evaluate(snap, cfg)
            chk = ris_path_check(model, cfg)  # raises on a zero RIS term (section 12.2)
            for kind, st in (("snapshot+combine", full), ("combine_only", comb)):
                out.append(
                    {
                        "tool": "sremu",
                        "method": f"sremu_{be}_{kind}",
                        "N": n,
                        "t_median_s": st["median"],
                        "t_min_s": st["min"],
                        "n_calls": int(st["n"]),
                        "h_abs_db": 20 * math.log10(abs(complex(h["tx.tx0->rx.rx0"]))),
                        "ris_minus_los_db": chk["min_ris_minus_los_db"],
                        "scenario_sha256": sha,
                    }
                )
            print(
                f"sremu {be} N={n}: full {full['median'] * 1e3:.3f} ms, "
                f"combine {comb['median'] * 1e3:.3f} ms",
                flush=True,
            )
    return out


# =========================================================================== MATLAB


def matlab_rows(
    sc_path: Path, out_dir: Path, exe: Path = MATLAB_EXE
) -> tuple[list[dict[str, Any]], str]:
    """Run ``bench/b6_matlab.m`` with ``matlab -batch``; returns rows and the console log."""
    if not exe.exists():
        return [], f"MATLAB not found at {exe}"
    out_csv = out_dir / "matlab_raw.csv"
    bench_dir = (REPO / "bench").as_posix()
    cmd = [
        str(exe),
        "-batch",
        f"addpath('{bench_dir}'); b6_matlab('{out_csv.as_posix()}', '{sc_path.as_posix()}')",
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - t0
    log = f"rc={r.returncode} wall={wall:.1f}s\n{r.stdout}\n{r.stderr}"
    rows = []
    if r.returncode == 0 and out_csv.exists():
        with out_csv.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(
                    {
                        "tool": "MATLAB",
                        "method": row["method"],
                        "N": int(row["N"]),
                        "t_median_s": float(row["t_median_s"]),
                        "t_min_s": None,
                        "n_calls": None,
                        "h_abs_db": float(row["h_abs_db"])
                        if row.get("h_abs_db") not in (None, "", "NaN")
                        else None,
                        "matlab_release": row.get("matlab_release"),
                    }
                )
    return rows, log


# =========================================================================== table


def context_table(rows: list[dict[str, Any]], platform: str) -> list[dict[str, Any]]:
    """One row per (method, N): time per evaluation, time per cell, |h| and its difference
    to sremu numba (same-model paths only), scope note and data class."""
    ref = {
        r["N"]: r["h_abs_db"]
        for r in rows
        if r["method"] == "sremu_numba_cpu_snapshot+combine" and r.get("h_abs_db") is not None
    }
    out = []
    for r in sorted(rows, key=lambda x: (x["N"], x["t_median_s"])):
        same_model = r["method"] != "matlab_risreflector_step"
        dh = (
            r["h_abs_db"] - ref[r["N"]]
            if same_model and r.get("h_abs_db") is not None and r["N"] in ref
            else None
        )
        out.append(
            {
                "N": r["N"],
                "tool": r["tool"],
                "method": r["method"],
                "t_per_eval_us": r["t_median_s"] * 1e6,
                "t_per_cell_ns": r["t_median_s"] / r["N"] * 1e9,
                "h_abs_db": r.get("h_abs_db"),
                "h_minus_sremu_db": dh,
                "same_model_as_sremu": same_model,
                "scope": SCOPE.get(r["method"], ""),
                "host": platform,
                "data_class": "MEAS",
            }
        )
    for lit in LITERATURE:
        out.append(
            {
                "N": None,
                "tool": lit["tool"],
                "method": lit["quantity"],
                "t_per_eval_us": None,
                "t_per_cell_ns": None,
                "h_abs_db": None,
                "h_minus_sremu_db": None,
                "same_model_as_sremu": None,
                "scope": f"{lit['value']} (literature; not a speed comparison)",
                "host": lit["hardware"],
                "data_class": lit["data_class"],
            }
        )
    return out


# =========================================================================== main


def main(argv: list[str] | None = None) -> int:
    """Run B6 and write its run directory."""
    bench.deterministic_env()
    ap = argparse.ArgumentParser(prog="bench.b6_crosstool", description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="256,1024,4096")
    ap.add_argument("--no-matlab", action="store_true")
    ap.add_argument("--matlab", type=Path, default=MATLAB_EXE)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wait-quiet-min", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: all)")
    a = ap.parse_args(argv)
    bench.set_threads(a.threads)
    sizes = [int(x) for x in a.sizes.split(",") if x]
    sc = scene(sizes)
    spec = {
        "threads": a.threads,
        "sizes": sizes,
        "scene": {k: sc[k] for k in ("fc_hz", "tx_m", "rx_m")},
        "matlab": not a.no_matlab,
        "scope": SCOPE,
    }
    run = BenchRun(
        SUITE,
        spec,
        out_root=a.out,
        tag=a.tag,
        data_classes={
            "times.csv": "MEAS",
            "context_table.csv": "MEAS (literature rows REPORTED, column data_class)",
            "literature.csv": "REPORTED",
            "*": "MEAS",
        },
        workload_classes={"times.csv": "SIM-sremu / MATLAB (column 'tool')", "*": "SIM-sremu"},
        caveats=[
            "Different tools evaluate different models: RISReflector is far-field plane-wave "
            "with isotropic unit cells and no cell-area factor; only sremu and the MATLAB "
            "hand-loop, vectorized and toolbox K1 paths compute the same near-field K1 sum.",
            "MATLAB times exclude MATLAB start-up (timeit inside one session); sremu times "
            "exclude numba compilation and include Python dispatch.",
            "Literature numbers are REPORTED with their own hardware; not a speed comparison.",
        ],
        wait_quiet_s=a.wait_quiet_min * 60.0,
        load_period_s=0.5,  # short sections: sample the host often
    )
    from bench._common import platform_label

    sc_path = run.path("scene.json")
    sc_path.write_text(json.dumps(sc), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    sections: dict[str, Any] = {}

    def section(name: str, fn: Any) -> Any:
        i0 = run.load.mark()
        t0 = time.perf_counter()
        res = fn()
        sections[name] = {
            "wall_s": time.perf_counter() - t0,
            "background_busy": run.load.window_stats(i0),
            "note": "MATLAB's own CPU use counts as background in the matlab section"
            if name == "matlab"
            else "other processes only",
        }
        return res

    with run.interruptible():
        rows += section("sremu", lambda: sremu_rows(sc, run))
        if not a.no_matlab:
            rm, logm = section("matlab", lambda: matlab_rows(sc_path, run.dir, a.matlab))
            rows += rm
            run.path("matlab_console.log").write_text(logm, encoding="utf-8")
    run.write_json("sections.json", sections)
    run.write_csv("times.csv", rows)
    table = context_table(rows, platform_label())
    run.write_csv("context_table.csv", table)
    run.write_csv("literature.csv", LITERATURE)
    agree = [
        abs(r["h_minus_sremu_db"])
        for r in table
        if r["h_minus_sremu_db"] is not None and r["tool"] == "MATLAB"
    ]
    summary = {
        "suite": SUITE,
        "per_N_us": {
            str(n): {r["method"]: r["t_median_s"] * 1e6 for r in rows if r["N"] == n} for n in sizes
        },
        "tools_run": sorted({r["tool"] for r in rows}),
        "matlab_same_model_max_abs_h_diff_db": max(agree) if agree else None,
        "background_busy_by_section": {k: v["background_busy"] for k, v in sections.items()},
    }
    run.finish(summary)
    print(f"B6 written to {run.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
