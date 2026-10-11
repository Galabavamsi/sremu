"""Diagnostic, NOT part of the protocol: which single additional lab fact raises C2 most.

PROTOCOL.md section 13.2. When the decoy power analysis labels the study "underpowered", this
module estimates how much one more piece of lab knowledge would raise the C2 detection rate,
so that the lead can tell the author which fact to ask for. Nothing here changes the
pre-registered prior, analysis or decision rules; the results are labelled
"diagnostic, not part of the protocol".

Two methods, both with the protocol's analysis code (:func:`.decoy.analyze_dataset`, primary
split, decisions C1-C3; the S-NF split is not run):

* **Subset** (facts about one parameter of a draw): the protocol's own tensor and decoys. For
  decoy ``i`` only the tensor draws whose parameter agrees with the decoy's truth within the
  fact's tolerance are kept, i.e. the prior conditioned on knowing that parameter to +-tol
  (the window is centred on the truth, a slightly optimistic reading of "known to +-tol").
  The bootstrap indices are the power analysis's (stream ``decoy_bootstrap``).
* **Restricted prior** (facts about the 30 point positions, which no tensor subset can
  express, and combinations): a smaller tensor and fresh decoys from a prior whose range is
  narrowed to +-tol around the nominal value, keyed streams ``vlab_diag_<fact>/<k>`` and
  ``vlab_diag_<fact>/<i>``; decoy and analysis share the narrowed prior.

Truth models M2 (detection) and M0 (false detection) at the nominal sigma_rep. No measured
data are used anywhere in this module.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from validation.ants_fig4b import decoy as dc
from validation.ants_fig4b.predict import (
    TensorCache,
    compute_tensor,
    evaluate_stream,
    load_cache,
    save_cache,
)
from validation.ants_fig4b.protocol import PROTOCOL_DIR, PROTOCOL_JSON, PROTOCOL_MD, load_protocol

from sremu import rng as sremu_rng
from sremu.analysis.stats import clopper_pearson
from sremu.record.manifest import utc_now_iso

__all__ = [
    "LABEL",
    "RESTRICTED_FACTS",
    "SUBSET_FACTS",
    "Fact",
    "restricted_protocol",
    "run_restricted_fact",
    "run_subset_fact",
    "subset_tensor",
    "write_protocol_block",
]

LABEL = "diagnostic, not part of the protocol"
BLOCK_BEGIN = "<!-- DIAGNOSTIC:BEGIN -->"
BLOCK_END = "<!-- DIAGNOSTIC:END -->"
_PER_DRAW_SKIP = ("sigma_pos_db", "pred_alt_cm_db")

MaskFn = Callable[[Mapping[str, np.ndarray], Mapping[str, np.ndarray], int], np.ndarray]


@dataclass(frozen=True)
class Fact:
    """One candidate lab fact: its id, the question to ask, a short description and either a
    subset mask or a prior override."""

    fact_id: str
    question: str
    description: str
    kind: str  # "subset" | "restricted"
    single: bool = True


def _near(key: str, tol: float, col: int | None = None) -> MaskFn:
    def fn(t: Mapping[str, np.ndarray], tr: Mapping[str, np.ndarray], i: int) -> np.ndarray:
        a, b = np.asarray(t[key]), np.asarray(tr[key])[i]
        if col is not None:
            a, b = a[:, col], b[col]
        return np.asarray(np.abs(a - b) <= tol)

    return fn


def _same(key: str) -> MaskFn:
    def fn(t: Mapping[str, np.ndarray], tr: Mapping[str, np.ndarray], i: int) -> np.ndarray:
        return np.asarray(np.asarray(t[key]) == np.asarray(tr[key])[i])

    return fn


def _all(*fns: MaskFn) -> MaskFn:
    def fn(t: Mapping[str, np.ndarray], tr: Mapping[str, np.ndarray], i: int) -> np.ndarray:
        m = np.ones(np.asarray(t["pred_db"]).shape[0], dtype=bool)
        for f in fns:
            m &= f(t, tr, i)
        return m

    return fn


SUBSET_FACTS: dict[str, tuple[Fact, MaskFn]] = {
    "R_pm10mm": (
        Fact("R_pm10mm", "Tx-to-tile distance R to +-1 cm", "|R - R_true| <= 10 mm", "subset"),
        _near("radius_m", 0.010),
    ),
    "R_pm2p5mm": (
        Fact("R_pm2p5mm", "Tx-to-tile distance R to +-2.5 mm", "|R - R_true| <= 2.5 mm", "subset"),
        _near("radius_m", 0.0025),
    ),
    "arc_angles": (
        Fact(
            "arc_angles",
            "angular positions of the tiles (arc centre to +-0.5 deg, span to +-1 deg)",
            "|centre - centre_true| <= 0.5 deg and |S - S_true| <= 1 deg",
            "subset",
        ),
        _all(_near("centre_deg", 0.5), _near("span_deg", 1.0)),
    ),
    "pitch": (
        Fact("pitch", "which unit cell (42.8 mm square or INDICON)", "same pitch", "subset"),
        _same("pitch_idx"),
    ),
    "antenna": (
        Fact(
            "antenna",
            "the whips' elevation pattern at 3.5 GHz (isotropic-like or dipole-like)",
            "same antenna option",
            "subset",
        ),
        _same("antenna_idx"),
    ),
    "gamma": (
        Fact(
            "gamma",
            "the two tile states' reflection coefficients (|Gamma| to +-0.1, phase to +-5 deg)",
            "|d|Gamma|| <= 0.1 and |d arg Gamma| <= 5 deg for both states",
            "subset",
        ),
        _all(
            _near("gamma_mag", 0.1, 0),
            _near("gamma_mag", 0.1, 1),
            _near("gamma_phase_offset_deg", 5.0, 0),
            _near("gamma_phase_offset_deg", 5.0, 1),
        ),
    ),
}


def _override_points(tol_m: float) -> Callable[[dict[str, Any]], None]:
    def fn(prior: dict[str, Any]) -> None:
        prior["point_error"] = {"xy_half_width_m": tol_m, "plane_height_half_width_m": tol_m}

    return fn


def _override_r_points(r_tol_m: float, p_tol_m: float) -> Callable[[dict[str, Any]], None]:
    def fn(prior: dict[str, Any]) -> None:
        r0 = float(np.mean(prior["geometry"]["radius_author_m"]))
        prior["geometry"]["radius_m"] = [r0 - r_tol_m, r0 + r_tol_m]
        _override_points(p_tol_m)(prior)

    return fn


def _override_tiles_known(prior: dict[str, Any]) -> None:
    """R = 1.95 m, S = 46 deg, centre 22.5 deg and the square cell, exactly."""
    g = prior["geometry"]
    r0 = float(np.mean(g["radius_author_m"]))
    s0 = float(np.mean(g["span_deg"]))
    g["radius_m"] = [r0, r0]
    g["span_deg"] = [s0, s0]
    g["centre_offset_deg"] = [0.0, 0.0]
    prior["pitch"] = {**prior["pitch"], "weights": [1.0, 0.0]}


def _override_tiles_points(p_tol_m: float | None) -> Callable[[dict[str, Any]], None]:
    def fn(prior: dict[str, Any]) -> None:
        _override_tiles_known(prior)
        if p_tol_m is not None:
            _override_points(p_tol_m)(prior)

    return fn


def _override_geometry_known(prior: dict[str, Any]) -> None:
    _override_tiles_points(0.0)(prior)


RESTRICTED_FACTS: dict[str, tuple[Fact, Callable[[dict[str, Any]], None]]] = {
    "points_pm5mm": (
        Fact(
            "points_pm5mm",
            "the 30 measurement positions and the plane height to +-5 mm",
            "point errors U[-5, 5] mm (in-plane per point, plane height)",
            "restricted",
        ),
        _override_points(0.005),
    ),
    "points_pm1mm": (
        Fact(
            "points_pm1mm",
            "the 30 measurement positions and the plane height to +-1 mm",
            "point errors U[-1, 1] mm",
            "restricted",
        ),
        _override_points(0.001),
    ),
    "R_pm2p5mm_points_pm5mm": (
        Fact(
            "R_pm2p5mm_points_pm5mm",
            "combination: R to +-2.5 mm and the positions to +-5 mm",
            "R ~ U[1.9475, 1.9525] m and point errors U[-5, 5] mm",
            "restricted",
            single=False,
        ),
        _override_r_points(0.0025, 0.005),
    ),
    "tiles_known": (
        Fact(
            "tiles_known",
            "combination: the as-built tile geometry exactly (R, arc angles, cell type); "
            "positions still +-2 cm",
            "R = 1.95 m, S = 46 deg, centre 22.5 deg, square pitch; point errors as v2",
            "restricted",
            single=False,
        ),
        _override_tiles_points(None),
    ),
    "tiles_known_points_pm5mm": (
        Fact(
            "tiles_known_points_pm5mm",
            "combination: the tile geometry exactly and the positions to +-5 mm",
            "as tiles_known, point errors U[-5, 5] mm",
            "restricted",
            single=False,
        ),
        _override_tiles_points(0.005),
    ),
    "tiles_known_points_pm1mm": (
        Fact(
            "tiles_known_points_pm1mm",
            "combination: the tile geometry exactly and the positions to +-1 mm",
            "as tiles_known, point errors U[-1, 1] mm",
            "restricted",
            single=False,
        ),
        _override_tiles_points(0.001),
    ),
    "geometry_known": (
        Fact(
            "geometry_known",
            "upper bound: every geometric quantity known exactly (R, arc angles, cell type, "
            "positions); Gamma, antenna pattern and cell model still unknown",
            "R = 1.95 m, S = 46 deg, centre 22.5 deg, square pitch, no point errors",
            "restricted",
            single=False,
        ),
        _override_geometry_known,
    ),
}


def restricted_protocol(
    protocol: Mapping[str, Any], fact_id: str, *, K: int, n_decoys: int
) -> dict[str, Any]:
    """The protocol of a restricted-prior fact: narrowed prior, keyed streams
    ``vlab_diag_<fact>/...``, ``K`` draws, no sigma_pos or paired draws."""
    out: dict[str, Any] = json.loads(json.dumps(protocol))
    _fact, override = RESTRICTED_FACTS[fact_id]
    override(out["prior"])
    out["rng"]["prior_draw"]["name"] = f"vlab_diag_{fact_id}/{{k}}"
    for key in ("decoy_truth", "decoy_noise", "decoy_bootstrap"):
        out["rng"][key]["name"] = f"vlab_diag_{fact_id}/{{i}}"
    out["primary_analysis"]["K"] = int(K)
    out["power_analysis"]["n_decoys"] = int(n_decoys)
    out["uncertainty_budget"]["sigma_pos"]["n_draws"] = 0
    out["models"]["near_field"]["point_vs_patch"]["n_draws"] = 0
    out.pop("power_analysis_result", None)
    out["variant"] = f"diagnostic/{fact_id}"
    return out


def subset_tensor(tensor: TensorCache, mask: np.ndarray) -> TensorCache:
    """The draws ``mask`` of a tensor (per-draw arrays sliced; first-n-draw extras dropped)."""
    k = tensor.K
    arrays = {
        name: a[mask]
        for name, a in tensor.arrays.items()
        if name not in _PER_DRAW_SKIP and a.ndim >= 1 and a.shape[0] == k
    }
    return TensorCache(arrays=arrays, meta=tensor.meta, path=tensor.path)


# =========================================================================== subset facts

_W: dict[str, Any] = {}


def _init(tensor_path: str, truth_path: str, protocol: dict[str, Any]) -> None:
    _W["tensor"] = load_cache(tensor_path, protocol=protocol)
    _W["truth"] = load_cache(truth_path, protocol=protocol)
    _W["protocol"] = protocol


def _row(i: int, fact: str, tm: str, srep: float, res: Mapping[str, Any], n: int) -> dict[str, Any]:
    d = res["primary"]["decisions"]
    return {
        "fact": fact,
        "decoy": i,
        "truth_model": tm,
        "sigma_rep_db": srep,
        "n_draws": n,
        "C1": d["C1"]["holds"],
        "C2": d["C2"]["holds"],
        "C3": d["C3"]["holds"],
        "c2_median_ok": d["C2"]["median_ok"],
        "c2_frac_ok": d["C2"]["frac_ok"],
        "spearman_ok": d["C2"]["spearman_ok"],
        "c2_median_delta_db": d["C2"]["median_delta_rmse_db"],
    }


def _subset_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    tensor: TensorCache = _W["tensor"]
    truth: TensorCache = _W["truth"]
    protocol = _W["protocol"]
    _fact, fn = SUBSET_FACTS[task["fact"]]
    st = protocol["rng"]["decoy_bootstrap"]
    rows = []
    for i in task["indices"]:
        mask = fn(tensor.arrays, truth.arrays, i)
        n = int(mask.sum())
        if n < int(task["min_draws"]):
            for tm in task["truth_models"]:
                rows.append({"fact": task["fact"], "decoy": i, "truth_model": tm, "n_draws": n})
            continue
        sub = subset_tensor(tensor, mask)
        name = str(st["name"]).format(i=i, k=i)
        for tm in task["truth_models"]:
            ds, _c = dc.decoy_dataset(
                i, truth.pred_db[i], protocol, sigma_rep_db=task["sigma_rep"], truth_model=tm
            )
            g = sremu_rng.generator(
                int(protocol["seed"]), str(st["purpose"]), name, counter=int(st["counter"])
            )
            res = dc.analyze_dataset(
                sub, ds, protocol, n_boot=task["n_boot"], rng=g, with_snf=False
            )
            rows.append(_row(i, task["fact"], tm, task["sigma_rep"], res, n))
    return rows


def run_subset_fact(
    fact_id: str,
    tensor_path: str | Path,
    truth_path: str | Path,
    protocol: Mapping[str, Any],
    *,
    n_decoys: int,
    n_boot: int,
    sigma_rep: float,
    truth_models: Sequence[str] = ("M2", "M0"),
    min_draws: int = 20,
    jobs: int = 1,
    chunk: int = 10,
) -> list[dict[str, Any]]:
    """Per-decoy decision rows of a subset fact (decoys with fewer than ``min_draws``
    consistent tensor draws are recorded without decisions)."""
    tasks = [
        {
            "fact": fact_id,
            "indices": list(range(i, min(n_decoys, i + chunk))),
            "truth_models": list(truth_models),
            "sigma_rep": float(sigma_rep),
            "n_boot": int(n_boot),
            "min_draws": int(min_draws),
        }
        for i in range(0, n_decoys, chunk)
    ]
    init = (str(tensor_path), str(truth_path), dict(protocol))
    rows: list[dict[str, Any]] = []
    if jobs <= 1:
        _init(*init)
        for t in tasks:
            rows.extend(_subset_task(t))
    else:
        with cf.ProcessPoolExecutor(max_workers=jobs, initializer=_init, initargs=init) as ex:
            for r in ex.map(_subset_task, tasks):
                rows.extend(r)
    rows.sort(key=lambda r: (r["truth_model"] != "M2", r["truth_model"], r["decoy"]))
    return rows


# =========================================================================== restricted


def run_restricted_fact(
    fact_id: str,
    protocol: Mapping[str, Any],
    run_dir: Path,
    *,
    K: int,
    n_decoys: int,
    n_boot: int,
    sigma_rep: float,
    jobs: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Tensor, fresh decoys and their analysis under the restricted prior of ``fact_id``."""
    proto = restricted_protocol(protocol, fact_id, K=K, n_decoys=n_decoys)
    t0 = time.perf_counter()
    res = compute_tensor(proto, jobs=jobs)
    tpath = run_dir / f"tensor_{fact_id}.npz"
    meta = save_cache(tpath, res, proto, extra_meta={"label": LABEL})
    tr = evaluate_stream(proto, proto["rng"]["decoy_truth"], n_decoys, jobs=jobs)
    trpath = run_dir / f"decoy_truth_{fact_id}.npz"
    tmeta = save_cache(trpath, tr, proto, extra_meta={"label": LABEL})
    rows = dc.run_power_analysis(
        tpath,
        tr["arrays"]["pred_db"],
        proto,
        n_decoys=n_decoys,
        sigma_reps=[float(sigma_rep)],
        truth_models=("M2", "M0"),
        n_boot=n_boot,
        jobs=jobs,
    )
    out = []
    for r in rows:
        out.append(
            {
                "fact": fact_id,
                "decoy": r["decoy"],
                "truth_model": r["truth_model"],
                "sigma_rep_db": r["sigma_rep_db"],
                "n_draws": int(K),
                **{
                    k: r[k]
                    for k in (
                        "C1",
                        "C2",
                        "C3",
                        "c2_median_ok",
                        "c2_frac_ok",
                        "spearman_ok",
                        "c2_median_delta_db",
                    )
                },
            }
        )
    info = {
        "tensor_content_sha256": meta["content_sha256"],
        "decoy_truth_content_sha256": tmeta["content_sha256"],
        "prediction_spec_sha256": meta["prediction_spec_sha256"],
        "rejected_candidates": res.get("rejected", {}),
        "wall_s": time.perf_counter() - t0,
    }
    return out, info


# =========================================================================== summary


def summarize(rows: Sequence[Mapping[str, Any]], facts: Sequence[str]) -> dict[str, Any]:
    """Per fact and truth model: decision rates (Clopper-Pearson 95 %) over the decoys with
    decisions, sub-condition rates, median Delta RMSE and the median subset size."""
    out: dict[str, Any] = {}
    for f in facts:
        rec: dict[str, Any] = {}
        for tm in ("M2", "M0"):
            rs = [r for r in rows if r["fact"] == f and r["truth_model"] == tm]
            ok = [r for r in rs if "C2" in r]
            n = len(ok)
            if n == 0:
                rec[tm] = {"n_decoys": 0, "n_skipped": len(rs)}
                continue
            k = int(sum(bool(r["C2"]) for r in ok))
            lo, hi = clopper_pearson(k, n, 0.95)
            rec[tm] = {
                "n_decoys": n,
                "n_skipped": len(rs) - n,
                "C2_rate": k / n,
                "C2_count": k,
                "C2_ci95": [lo, hi],
                "C1_rate": float(np.mean([bool(r["C1"]) for r in ok])),
                "C3_rate": float(np.mean([bool(r["C3"]) for r in ok])),
                "c2_median_ok_rate": float(np.mean([bool(r["c2_median_ok"]) for r in ok])),
                "c2_frac_ok_rate": float(np.mean([bool(r["c2_frac_ok"]) for r in ok])),
                "spearman_ok_rate": float(np.mean([bool(r["spearman_ok"]) for r in ok])),
                "median_delta_rmse_db": float(np.median([r["c2_median_delta_db"] for r in ok])),
                "median_n_draws": float(np.median([r["n_draws"] for r in ok])),
            }
        out[f] = rec
    return out


def _fact(fid: str) -> Fact:
    if fid in SUBSET_FACTS:
        return SUBSET_FACTS[fid][0]
    return RESTRICTED_FACTS[fid][0]


def protocol_block(
    summary: Mapping[str, Any], baseline: Mapping[str, Any], run: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    """The JSON record and the markdown block for PROTOCOL.md section 13.2."""
    rec = {
        "label": LABEL,
        "computed_utc": run["t_utc"],
        "run_directory": run["run_dir"],
        "sremu_commit": run["git_commit"],
        "sremu_dirty": run["git_dirty"],
        "baseline": baseline,
        "methods": {
            "subset": (
                "protocol tensor and decoys; per decoy only the draws consistent with the "
                "decoy's truth within the tolerance (window centred on the truth)"
            ),
            "restricted": (
                "smaller tensor and fresh decoys from a prior narrowed to +-tol around the "
                "nominal value (streams vlab_diag_<fact>/...)"
            ),
        },
        "facts": {
            fid: {
                "question": _fact(fid).question,
                "method": _fact(fid).kind,
                "single_fact": _fact(fid).single,
                **summary[fid],
            }
            for fid in summary
        },
        "settings": run["settings"],
    }
    ranked = sorted(
        (f for f in summary if _fact(f).single and summary[f]["M2"].get("n_decoys", 0) > 0),
        key=lambda f: (-summary[f]["M2"]["C2_rate"], -summary[f]["M2"]["median_delta_rmse_db"]),
    )
    top = ranked[0] if ranked else None
    rec["best_single_fact"] = (
        top if top is not None and summary[top]["M2"]["C2_rate"] > 0.0 else None
    )
    rec["closest_single_fact_by_delta_rmse"] = top
    b2, b0 = baseline["M2"], baseline["M0"]
    lines = [
        f"**Label: {LABEL}.** Computed {run['t_utc']} (sremu commit {run['git_commit']}, dirty "
        f"{run['git_dirty']}); run `{run['run_dir']}`"
        + (
            " (rows of `" + "`, `".join(run["settings"]["merged_runs"]) + "` merged)"
            if run["settings"].get("merged_runs")
            else ""
        )
        + ". Truth M2 (detection) and M0 (false "
        f"detection), sigma_rep {run['settings']['sigma_rep_db']:g} dB, "
        f"{run['settings']['n_boot']} bootstrap resamples per decoy. Subset facts use the "
        f"protocol's tensor and its {run['settings']['n_decoys_subset']} decoys (the prior "
        "conditioned on knowing the parameter to +-tol around the truth; decoys with fewer than "
        f"{run['settings']['min_draws']} consistent draws are skipped); restricted facts use a "
        f"K = {run['settings']['K_restricted']} tensor and {run['settings']['n_decoys_restricted']}"
        " fresh decoys from the narrowed prior.",
        "",
        "| Fact to ask the author for | Method | C2 rate, truth M2 (95 % CI) | C2 rate, truth M0 "
        "| median Delta RMSE, truth M2 (dB) | median draws |",
        "|---|---|---|---|---|---|",
        f"| none (protocol v2 as pre-registered) | power analysis | {b2['C2_rate']:.3f} "
        f"({b2['C2_ci95'][0]:.3f}-{b2['C2_ci95'][1]:.3f}) | {b0['C2_rate']:.3f} "
        f"| {b2['median_delta_rmse_db']:.2f} | {b2['median_n_draws']:.0f} |",
    ]
    for fid in summary:
        m2, m0 = summary[fid]["M2"], summary[fid]["M0"]
        f = _fact(fid)
        q = (f.question if f.single else f"({f.question})").replace("|", r"\|")
        if m2.get("n_decoys", 0) == 0:
            lines.append(f"| {q} | {f.kind} | no decoy with enough draws | | | |")
            continue
        lines.append(
            f"| {q} | {f.kind} | {m2['C2_rate']:.3f} ({m2['C2_ci95'][0]:.3f}-"
            f"{m2['C2_ci95'][1]:.3f}) | {m0.get('C2_rate', float('nan')):.3f} "
            f"| {m2['median_delta_rmse_db']:.2f} | {m2['median_n_draws']:.0f} |"
        )
    if top is not None:
        m2 = summary[top]["M2"]
        if m2["C2_rate"] > 0.0:
            lines += [
                "",
                f"Largest C2 detection rate from a single fact: **{_fact(top).question}** "
                f"({m2['C2_rate']:.3f}). "
                + (
                    "No single fact reaches the 0.5 power threshold."
                    if m2["C2_rate"] < 0.5
                    else "This reaches the 0.5 power threshold."
                ),
            ]
        else:
            lines += [
                "",
                "No single fact raises the C2 detection rate above 0; the closest by median "
                f"Delta RMSE is **{_fact(top).question}** "
                f"({m2['median_delta_rmse_db']:.2f} dB). No single fact reaches the 0.5 power "
                "threshold.",
            ]
    combos = [
        f for f in summary if not _fact(f).single and summary[f]["M2"].get("C2_rate", 0.0) >= 0.5
    ]
    if combos:
        lines.append(
            "Combinations that reach 0.5: "
            + "; ".join(f"{_fact(f).question} ({summary[f]['M2']['C2_rate']:.3f})" for f in combos)
            + "."
        )
    return rec, "\n".join(lines)


def write_protocol_block(
    record: Mapping[str, Any], block_md: str, protocol_dir: str | Path | None = None
) -> None:
    """Write the diagnostic into ``protocol.json`` (``diagnostic_not_part_of_protocol``) and
    the marked block of ``PROTOCOL.md`` (section 13.2)."""
    d = Path(protocol_dir) if protocol_dir is not None else PROTOCOL_DIR
    pj = d / PROTOCOL_JSON
    proto = json.loads(pj.read_text(encoding="utf-8"))
    proto["diagnostic_not_part_of_protocol"] = record
    pj.write_text(
        json.dumps(proto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path = d / PROTOCOL_MD
    md = md_path.read_text(encoding="utf-8")
    pat = re.compile(re.escape(BLOCK_BEGIN) + r".*?" + re.escape(BLOCK_END), re.S)
    if not pat.search(md):
        raise ValueError("PROTOCOL.md lacks the diagnostic markers")
    md = pat.sub(lambda _m: BLOCK_BEGIN + "\n" + block_md + "\n" + BLOCK_END, md)
    md_path.write_text(md, encoding="utf-8", newline="\n")


# =========================================================================== CLI


def _baseline(power_rows_csv: Path, sigma_rep: float) -> dict[str, Any]:
    import csv

    with power_rows_csv.open(encoding="utf-8") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if abs(float(r["sigma_rep_db"]) - sigma_rep) < 1e-9 and r["truth_model"] in ("M2", "M0")
        ]
    out: dict[str, Any] = {}
    for tm in ("M2", "M0"):
        rs = [r for r in rows if r["truth_model"] == tm]
        k = sum(r["C2"] == "True" for r in rs)
        lo, hi = clopper_pearson(k, len(rs), 0.95)
        out[tm] = {
            "n_decoys": len(rs),
            "C2_rate": k / len(rs),
            "C2_count": k,
            "C2_ci95": [lo, hi],
            "median_delta_rmse_db": float(np.median([float(r["c2_median_delta_db"]) for r in rs])),
            "median_n_draws": 2000.0,
        }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run the diagnostic into a new run directory (label: diagnostic, not part of the
    protocol)."""
    from validation.ants_fig4b import runlog

    ap = argparse.ArgumentParser(prog="python -m validation.ants_fig4b.diagnostic")
    ap.add_argument("--power-run", type=Path, required=True, help="run dir of decoy.py")
    ap.add_argument("--tensor", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs") / "vlab")
    ap.add_argument("--subset-facts", default=",".join(SUBSET_FACTS))
    ap.add_argument("--restricted-facts", default=",".join(RESTRICTED_FACTS))
    ap.add_argument("--n-decoys-subset", type=int, default=500)
    ap.add_argument("--K-restricted", type=int, default=1000)
    ap.add_argument("--n-decoys-restricted", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-draws", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--write-protocol-block", action="store_true")
    ap.add_argument(
        "--include-run",
        type=Path,
        action="append",
        default=[],
        help="earlier diagnostic run directory whose rows are merged into the summary",
    )
    args = ap.parse_args(argv)
    protocol = load_protocol()
    srep = float(protocol["power_analysis"]["sigma_rep_db"])
    rec = runlog.RunRecorder(args.out, "vlab_diagnostic", protocol, argv=sys.argv)
    truth_path = args.power_run / "decoy_truth.npz"
    rec.add_input(args.tensor, "SIM-sremu")
    rec.add_input(truth_path, "SIM-sremu")
    rows: list[dict[str, Any]] = []
    infos: dict[str, Any] = {}
    sub = [f for f in args.subset_facts.split(",") if f]
    res_f = [f for f in args.restricted_facts.split(",") if f]
    for f in res_f:
        r, info = run_restricted_fact(
            f,
            protocol,
            rec.run_dir,
            K=args.K_restricted,
            n_decoys=args.n_decoys_restricted,
            n_boot=args.n_boot,
            sigma_rep=srep,
            jobs=args.jobs,
        )
        rows += r
        infos[f] = info
        rec.add_output(f"tensor_{f}.npz", "SIM-sremu")
        rec.add_output(f"decoy_truth_{f}.npz", "SIM-sremu")
        print(f"  restricted fact {f}: {info['wall_s']:.0f} s", flush=True)
    for f in sub:
        t0 = time.perf_counter()
        rows += run_subset_fact(
            f,
            args.tensor,
            truth_path,
            protocol,
            n_decoys=args.n_decoys_subset,
            n_boot=args.n_boot,
            sigma_rep=srep,
            min_draws=args.min_draws,
            jobs=args.jobs,
        )
        infos[f] = {"wall_s": time.perf_counter() - t0}
        print(f"  subset fact {f}: {infos[f]['wall_s']:.0f} s", flush=True)
    facts = [*sub, *res_f]
    for prev in args.include_run:
        old = read_rows(prev / "diagnostic_rows.csv")
        rec.add_input(prev / "diagnostic_rows.csv", "SIM-sremu")
        prev_summary = json.loads((prev / "summary.json").read_text(encoding="utf-8"))
        for f in dict.fromkeys(r["fact"] for r in old):
            if f in facts:
                continue
            facts.append(f)
            rows += [r for r in old if r["fact"] == f]
            infos[f] = {
                **(prev_summary.get("fact_runs") or {}).get(f, {}),
                "run_directory": str(prev),
            }
    order = [*SUBSET_FACTS, *RESTRICTED_FACTS]
    facts.sort(key=lambda f: order.index(f) if f in order else len(order))
    summ = summarize(rows, facts)
    while not (args.power_run / "decoys.csv").is_file():  # the power run may still be busy
        time.sleep(30)
    rec.add_input(args.power_run / "decoys.csv", "SIM-sremu")
    base = _baseline(args.power_run / "decoys.csv", srep)
    run_info = {
        "t_utc": utc_now_iso(),
        "run_dir": str(rec.run_dir),
        "git_commit": rec.git.commit,
        "git_dirty": rec.git.dirty,
        "settings": {
            "sigma_rep_db": srep,
            "n_boot": args.n_boot,
            "n_decoys_subset": args.n_decoys_subset,
            "K_restricted": args.K_restricted,
            "n_decoys_restricted": args.n_decoys_restricted,
            "min_draws": args.min_draws,
            "power_run": str(args.power_run),
            "merged_runs": [str(x) for x in args.include_run],
        },
    }
    record, block = protocol_block(summ, base, run_info)
    record["fact_runs"] = infos
    rec.write_csv("diagnostic_rows.csv", rows_for_csv(rows))
    rec.add_output("diagnostic_rows.csv", "SIM-sremu")
    rec.write_json("summary.json", {"label": LABEL, **record, "block_md": block})
    rec.add_output("summary.json", "SIM-sremu")
    rec.finish(backend="numba_cpu", extra={"label": LABEL})
    print(block)
    print(f"run directory: {rec.run_dir}")
    if args.write_protocol_block:
        write_protocol_block(record, block)
        print("diagnostic written into protocol.json and PROTOCOL.md")
    return 0


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of an earlier ``diagnostic_rows.csv`` with their types restored (skipped decoys
    keep no decision fields)."""
    import csv

    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row: dict[str, Any] = {}
            for k, v in r.items():
                if v == "":
                    continue
                if v in ("True", "False"):
                    row[k] = v == "True"
                elif k in ("fact", "truth_model"):
                    row[k] = v
                elif k in ("decoy", "n_draws"):
                    row[k] = int(float(v))
                else:
                    row[k] = float(v)
            out.append(row)
    return out


def rows_for_csv(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows with a common key set (skipped decoys get empty decision fields)."""
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    return [{k: r.get(k, "") for k in keys} for r in rows]


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")
    sys.exit(main())
