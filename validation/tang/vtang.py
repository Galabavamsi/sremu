"""V-TANG: Tang et al. 2022 published model curves and measured points (DESIGN.md 10.8).

Inputs (``data/validation/tang2022``, sha256-checked): Fig. 10 vector-extracted series
(DIG: ``refined_eq9`` published model curves and ``measured`` chamber points, 4 panels x 9
points) and the Tables III/IV parameters (REPORTED). The engine runs with the **shipped
default** ``cell_model`` (auto, which resolves to ``point`` for RIS1 at 27 GHz).

Received power (the V-TANG-1 test's link budget): CosPower(53.8) horn patterns aimed at the
panel centre carry the nominal peak gains G_0 = 2(alpha + 1) inside |h_RIS|; Tang's
measured calibration replaces them: ``P_R = P_T + 24.0 dB - 20 log10(G_0) + 20 log10|h_RIS|``.

* **V-TANG-1**: engine vs the published Eq. (9) curves, max |difference| <= 0.05 dB.
* **V-TANG-2**: measured - model (engine; and the published curve, for the wording):
  bias, SD, RMSE, n = 36; regression gate |RMSE(measured - engine) - 1.05| <= 0.05 dB.
* Paper wording template (DESIGN 10.8), filled from the numbers of this run.

Usage::

    python validation/tang/vtang.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vstudy as vs  # noqa: E402

STUDY_DIR = _HERE
DATA = vs.DATA_VALIDATION / "tang2022"
FIG = DATA / "tang2022_fig10_vector.json"
PARAMS = DATA / "tang2022_parameters.json"
LINK = "tx.tx0->rx.rx0"
TOL_CURVE_DB = 0.05
RMSE_REF_DB = 1.05
RMSE_TOL_DB = 0.05
WORDING = (
    "The kernel reproduces Tang et al.'s published model curves within {curve} dB. Their "
    "digitized chamber measurements lie {bias} dB (SD {sd} dB, n = {n}) above that "
    "published model."
)


def check_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the inputs after checking them against ``SHA256SUMS``."""
    sums = dict(line.split()[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines())
    for f in (FIG, PARAMS):
        got = vs.sha256_file(f)
        if got != sums[f.name]:
            raise SystemExit(f"{f.name}: sha256 {got} != SHA256SUMS {sums[f.name]}")
    return json.loads(FIG.read_bytes())["data"], json.loads(PARAMS.read_text(encoding="utf-8"))


def scenario(par: dict[str, Any], d1: float, d2: float, th_deg: float) -> dict[str, Any]:
    """Tang's specular setup: Tx at d1, Rx at d2, both at theta from the panel normal."""
    ris = par["ris"]
    alpha = 53.8
    th = math.radians(th_deg)
    ptx = [-d1 * math.sin(th), 0.0, d1 * math.cos(th)]
    prx = [d2 * math.sin(th), 0.0, d2 * math.cos(th)]

    def node(nid: str, pos: list[float], direction: str) -> dict[str, Any]:
        b = (-np.asarray(pos) / np.linalg.norm(pos)).tolist()
        stream: dict[str, Any] = {
            "name": f"{direction}0",
            "direction": direction,
            "fc_hz": ris["fc_hz"],
            "rate_sps": "1e6",
        }
        if direction == "tx":
            stream["occupied_bw_hz"] = 1e5
        return {
            "id": nid,
            "position_m": pos,
            "antenna": {"pattern": {"cos_power": {"alpha": alpha}}, "boresight": b},
            "streams": [stream],
        }

    return {
        "schema_version": "1.0",
        "name": "vtang",
        "seed": 1,
        "description": f"Tang 2022 Fig. 10 geometry d1 = {d1} m, d2 = {d2} m, theta = {th_deg} deg",
        "nodes": [node("tx", ptx, "tx"), node("rx", prx, "rx")],
        "ris": [
            {
                "id": "ris1",
                "pose": {"center_m": [0, 0, 0], "normal": [0, 0, 1], "u_axis": [1, 0, 0]},
                "grid": {
                    "n_u": ris["n_u"],
                    "n_v": ris["n_v"],
                    "pitch_u_m": ris["pitch_u_m"],
                    "pitch_v_m": ris["pitch_v_m"],
                    "cell_model": "auto",
                },
                "states": {"library": "tang2022_ris1_27ghz"},
                "initial": {"fill": 0},
            }
        ],
        "run": {"mode": "lockstep", "coupling": "barrier", "duration_s": "1/1000"},
    }


def engine_power_dbm(par: dict[str, Any], d1: float, d2: float, th: float) -> tuple[float, str]:
    """P_R (dBm) of the RIS path through ``from_scenario`` (cell_model auto)."""
    from sremu.physics.propagation import from_scenario

    model = from_scenario(
        vs.resolve_mapping(scenario(par, d1, d2, th)).scenario.model_dump(mode="python")
    )
    snap = model.snapshot(Fraction(0))
    lp = model.combine(snap, {"ris1": model.initial_config("ris1")})[LINK]
    h_ris = lp.amp[lp.tile >= 0]
    if h_ris.size != 1:
        raise RuntimeError("expected one RIS tile path")
    g0 = 2 * (53.8 + 1)
    p = par["link"]["p_t_dbm"] + par["link"]["gt_gr_gline_db"] - 20 * math.log10(g0)
    return p + 20 * math.log10(abs(h_ris[0])), model.cell_model("ris1")


def study(root: Path | None = None) -> Path:
    from sremu.analysis.stats import error_metrics

    fig, par = check_inputs()
    panels = par["fig10_panels"]
    base = scenario(par, panels["a"]["d1_m"], 1.0, panels["a"]["theta_deg"])
    run = vs.StudyRun(
        STUDY_DIR,
        "vtang",
        base,
        root=root,
        inputs=[(FIG, "DIG"), (PARAMS, "REPORTED")],
        assumptions=(
            "Tang Fig. 10 x positions snapped from 1.002 ... 5.002 m to the nominal 0.5 m grid",
            "link budget: P_T + G_t G_r G_line (measured, 24.0 dB) - nominal horn gains + |h_RIS|",
        ),
    )
    t0 = time.perf_counter()
    rows = []
    for pid in "abcd":
        d1 = panels[pid]["d1_m"]
        th = panels[pid]["theta_deg"]
        curve = fig[pid]["refined_eq9"]
        meas = fig[pid]["measured"]
        if [round(x * 2) / 2 for x, _ in curve] != [round(x * 2) / 2 for x, _ in meas]:
            raise RuntimeError(f"panel {pid}: curve and measured x grids differ")
        for (x_dig, p_curve), (_xm, p_meas) in zip(curve, meas, strict=True):
            d2 = round(x_dig * 2) / 2
            p_eng, cm = engine_power_dbm(par, d1, d2, th)
            rows.append(
                {
                    "panel": pid,
                    "d1_m": d1,
                    "theta_deg": th,
                    "d2_m": d2,
                    "d2_digitized_m": x_dig,
                    "published_eq9_dbm": p_curve,
                    "engine_dbm": p_eng,
                    "measured_dbm": p_meas,
                    "engine_minus_published_db": p_eng - p_curve,
                    "measured_minus_engine_db": p_meas - p_eng,
                    "measured_minus_published_db": p_meas - p_curve,
                    "cell_model": cm,
                }
            )
    run.time("engine_points", time.perf_counter() - t0)
    vs.write_csv(run.path("vtang_points.csv"), rows)
    eng = np.array([r["engine_dbm"] for r in rows])
    pub = np.array([r["published_eq9_dbm"] for r in rows])
    mea = np.array([r["measured_dbm"] for r in rows])
    d_curve = np.abs(eng - pub)
    m_eng = error_metrics(mea, eng)
    m_pub = error_metrics(mea, pub)
    per_panel = {}
    for pid in "abcd":
        sel = np.array([r["panel"] == pid for r in rows])
        per_panel[pid] = {
            "max_abs_engine_minus_published_db": float(np.max(d_curve[sel])),
            "measured_minus_engine": error_metrics(mea[sel], eng[sel]).as_dict(),
        }
    curve_max = float(np.max(d_curve))
    acc = {
        "V-TANG-1_max_abs_le_0.05dB": bool(curve_max <= TOL_CURVE_DB),
        "V-TANG-2_rmse_regression_1.05_pm_0.05dB": bool(
            abs(m_eng.rmse - RMSE_REF_DB) <= RMSE_TOL_DB
        ),
        "cell_model_auto_resolved": sorted({r["cell_model"] for r in rows}),
    }
    numbers = {
        "n_points": len(rows),
        "vtang1_max_abs_db": curve_max,
        "vtang1_mean_abs_db": float(np.mean(d_curve)),
        "vtang1_rmse_db": float(np.sqrt(np.mean((eng - pub) ** 2))),
        "vtang2_measured_minus_engine": m_eng.as_dict(),
        "vtang2_measured_minus_published": m_pub.as_dict(),
        "per_panel": per_panel,
    }
    # "within X dB" is a bound, so X is the maximum rounded *up* to 0.01 dB
    curve_bound = math.ceil(curve_max * 100.0 - 1e-9) / 100.0
    numbers["vtang1_bound_for_wording_db"] = curve_bound
    wording = WORDING.format(
        curve=f"{curve_bound:.2f}", bias=f"{m_pub.bias:+.2f}", sd=f"{m_pub.sd:.2f}", n=m_pub.n
    )
    numbers["wording"] = wording
    numbers["wording_note"] = (
        "Template of DESIGN.md 10.8 filled from this run (curve: max |engine - published| "
        "rounded up to 0.01 dB; bias/SD: measured - published Eq. 9 curve). Not presented as "
        "sremu accuracy."
    )
    vs.write_json(run.path("vtang_summary.json"), {"acceptance": acc, **numbers})
    meta = {
        "study": "V-TANG",
        "data_classes": ["DIG", "REPORTED", "SIM-sremu"],
        "acceptance": acc,
        "numbers": numbers,
        "figure": "F3(b): vtang_points.csv (published_eq9_dbm, engine_dbm, measured_dbm)",
    }
    path = run.finish(meta)
    print(f"V-TANG run directory: {run.run_dir}\nmeta: {path}\n{wording}")
    if root is None:
        print(f"report: {vs.write_report(STUDY_DIR, report(run.run_dir))}")
    return run.run_dir


# =========================================================================== report


def report(run_dir: Path) -> str:
    """``validation/tang/report.md`` rendered from a run directory's ``meta.json``."""
    meta = vs.read_meta(run_dir)
    f = vs.fnum
    acc = meta["acceptance"]
    num = meta["numbers"]
    me = num["vtang2_measured_minus_engine"]
    mp = num["vtang2_measured_minus_published"]
    s = []
    s.append("# V-TANG: Tang et al. 2022 model curves and measured points (DESIGN.md 10.8)\n")
    s.append(
        "V-TANG-1 compares sremu's area-tied K1 kernel, run through the scene model with the "
        "shipped default `cell_model` (auto), against the **published model curves** (Eq. 9) of "
        "Tang et al., Fig. 10: a model-to-model check. V-TANG-2 compares Tang's digitized "
        "chamber measurements with that model; it characterises Tang's published data and "
        "model and is **not** presented as sremu accuracy (DESIGN 10.8, 1.6). The Fig. 10 "
        "series are vector-extracted from the arXiv PDF (DIG; no measurement uncertainty is "
        f"stated in the source). Data classes: {', '.join(meta.get('data_classes', []))}.\n"
    )
    s.append("## Provenance\n")
    s.append(vs.run_header(meta) + "\n")
    s.append(
        "* Inputs: "
        + "; ".join(
            f"`{i['path']}` ({i['data_class']}, sha256 `{i['sha256'][:16]}...`)"
            for i in meta["inputs"]
        )
        + " (checked against `data/validation/tang2022/SHA256SUMS`; provenance in "
        "`data/validation/tang2022/PROVENANCE.md`).\n"
    )
    s.append("## Setup\n")
    s.append(
        "RIS1 of Tang 2022 Tables III/IV: 56 x 20 cells of 1.4 x 2.8 mm at 27 GHz, all cells in "
        "coding state '0' (Gamma = 0.9 at 165 deg, built-in table `tang2022_ris1_27ghz`); "
        "cos^53.8 horn patterns aimed at the panel centre; specular geometry (Tx at d1, Rx at "
        "d2, both at theta from the normal in the u-axis plane); four panels (d1, theta) = "
        "(1 m, 10 deg), (2 m, 10 deg), (1 m, 45 deg), (2 m, 45 deg), nine d2 = 1 ... 5 m each. "
        "Received power `P_R = P_T + G_t G_r G_line - 20 log10(G_0) + 20 log10 |h_RIS|` with "
        "P_T = 20 dBm and the measured G_t G_r G_line = 24.0 dB replacing the nominal horn peak "
        "gains G_0 = 2(alpha + 1) carried by the engine's CosPower antennas. `cell_model` auto "
        f"resolved to {acc['cell_model_auto_resolved']} (pitch < lambda/2). Digitized x "
        "positions (1.002 ... 5.002 m) are snapped to the nominal 0.5 m grid.\n"
    )
    s.append("## Acceptance\n")
    s.append(
        vs.md_table(
            ("Check", "Measured", "Result"),
            (
                (
                    "V-TANG-1: max |engine - published Eq. 9| <= 0.05 dB (36 points)",
                    f"{f(num['vtang1_max_abs_db'], 4)} dB",
                    vs.passfail(acc["V-TANG-1_max_abs_le_0.05dB"]),
                ),
                (
                    "V-TANG-2 regression: |RMSE(measured - engine) - 1.05| <= 0.05 dB",
                    f"RMSE {f(me['rmse'], 4)} dB",
                    vs.passfail(acc["V-TANG-2_rmse_regression_1.05_pm_0.05dB"]),
                ),
            ),
        )
        + "\n"
    )
    s.append("## Numbers (n = 36)\n")
    s.append(
        f"V-TANG-1: max |engine - published| = {f(num['vtang1_max_abs_db'], 4)} dB, mean "
        f"{f(num['vtang1_mean_abs_db'], 4)} dB, RMSE {f(num['vtang1_rmse_db'], 4)} dB.\n"
    )
    s.append(
        vs.md_table(
            (
                "V-TANG-2 errors",
                "n",
                "bias (dB)",
                "SD (dB)",
                "RMSE (dB)",
                "MAE (dB)",
                "max |e| (dB)",
            ),
            (
                (
                    lab,
                    m["n"],
                    f(m["bias"], 4, sign=True),
                    f(m["sd"], 4),
                    f(m["rmse"], 4),
                    f(m["mae"], 4),
                    f(m["max_abs"], 4),
                )
                for lab, m in (
                    ("measured - engine", me),
                    ("measured - published Eq. 9 curve", mp),
                )
            ),
        )
        + "\n"
    )
    s.append(
        vs.md_table(
            (
                "panel",
                "d1 (m), theta (deg)",
                "max |engine - published| (dB)",
                "measured - engine: bias (dB)",
                "SD (dB)",
                "RMSE (dB)",
            ),
            (
                (
                    pid,
                    {"a": "1, 10", "b": "2, 10", "c": "1, 45", "d": "2, 45"}[pid],
                    f(p["max_abs_engine_minus_published_db"], 4),
                    f(p["measured_minus_engine"]["bias"], 4, sign=True),
                    f(p["measured_minus_engine"]["sd"], 4),
                    f(p["measured_minus_engine"]["rmse"], 4),
                )
                for pid, p in sorted(num["per_panel"].items())
            ),
        )
        + "\n"
    )
    s.append("## Paper wording (DESIGN 10.8 template, filled from this run)\n")
    s.append(f"> {num['wording']}\n")
    s.append(
        "Filled as: curve = max |engine - published| rounded up to 0.01 dB (a bound); "
        "bias and SD = "
        'measured - published Eq. 9 curve (the sentence\'s "above that published model"). '
        f"With measured - engine instead, the bias is {f(me['bias'], 3, sign=True)} dB (SD "
        f"{f(me['sd'], 3)} dB). The DESIGN placeholders ({{0.02}}, {{+0.98}}, {{0.38}}) were "
        "scratch values; the paper must take these numbers from the pipeline macros, not from "
        "this report.\n"
    )
    s.append("## Files and run time\n")
    s.append(
        "`vtang_points.csv` (per point: panel, geometry, published, engine and measured dBm and "
        "their differences; figure F3(b) data), `vtang_summary.json`, `meta.json`, "
        f"`manifest.json`. Engine evaluation of the 36 points took "
        f"{f((meta.get('timings_s') or {}).get('engine_points'), 2)} s.\n"
    )
    return "\n".join(s)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V-TANG validation study (DESIGN.md 10.8)")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None, help="write report.md from a run dir")
    a = ap.parse_args(argv)
    if a.report is not None:
        print(vs.write_report(STUDY_DIR, report(a.report)))
        return 0
    study(root=a.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
