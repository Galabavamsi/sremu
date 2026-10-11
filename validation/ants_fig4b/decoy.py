"""Synthetic decoys and the V-LAB power analysis (DESIGN.md section 10.9, "Power analysis").

``synthetic_decoy(i)``: the M2 prediction at a **fresh** prior draw (stream
``rng.decoy_truth`` = ``("decoy", "vlab/<i>", counter 0)``), shifted so that its noiseless
mean equals the mean of the disclosed ring means, plus ``sigma_rep z1 + sigma_dig z2`` with
``sigma_dig ~ U[0.045, 0.22]`` dB per cell and ``z1, z2 ~ N(0, 1)`` from
``("decoy", "vlab/<i>", counter 1)`` (draw order: sigma_dig[30], z1[30], z2[30]). The same
``z`` are used for every ``sigma_rep`` (common random numbers).

The power analysis runs the primary and secondary analyses of :mod:`.analysis` on every
decoy with the cached tensor and ``n_bootstrap = 2000`` resamples (indices from
``("decoy", "vlab/<i>", counter 2)``), and reports the C2 detection rate with a
Clopper-Pearson 95 % interval. If the rate is below 0.5 the study is labelled
"underpowered" whatever the outcome. Null decoys (truth M0, informational) estimate how
often C2 fires without a RIS path (the specificity check). The S-NF split (offsets fitted
without the near-field ring r = 1.8 m; the same test cells and bootstrap indices) is also
analysed on every decoy and reported as information only.

No measured data are used anywhere in this module.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from validation.ants_fig4b import analysis as an
from validation.ants_fig4b.predict import (
    MODELS,
    TensorCache,
    draw_and_evaluate,
    evaluate_stream,
    load_cache,
    save_cache,
)
from validation.ants_fig4b.prior import PriorDraw
from validation.ants_fig4b.protocol import (
    PROTOCOL_DIR,
    PROTOCOL_JSON,
    PROTOCOL_MD,
    cell_layout,
    load_protocol,
    prediction_spec_sha256,
    ring_split,
    sensitivity_run,
    split_masks,
)

from sremu import rng as sremu_rng
from sremu.analysis.stats import clopper_pearson
from sremu.record.manifest import utc_now_iso

__all__ = [
    "Decoy",
    "analyze_dataset",
    "decoy_dataset",
    "decoy_noise",
    "prediction_spread_comparison",
    "run_power_analysis",
    "synthetic_decoy",
    "write_protocol_result",
]

RESULT_BEGIN = "<!-- POWER_ANALYSIS_RESULT:BEGIN (written by decoy.py --write-protocol-result) -->"
RESULT_END = "<!-- POWER_ANALYSIS_RESULT:END -->"


@dataclass(frozen=True, eq=False)
class Decoy:
    """A synthetic dataset with its truth."""

    index: int
    dataset: an.Dataset
    truth_draw: PriorDraw | None
    truth_pred_db: np.ndarray
    truth_model: str
    offset_dbm: float
    sigma_rep_db: float


def _stream(protocol: Mapping[str, Any], key: str, i: int) -> tuple[str, str, int]:
    st = protocol["rng"][key]
    return str(st["purpose"]), str(st["name"]).format(i=i, k=i), int(st["counter"])


def decoy_noise(i: int, protocol: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(sigma_dig[30], z1[30], z2[30])`` of decoy ``i`` (keyed, draw order fixed)."""
    pa = protocol["power_analysis"]
    purpose, name, counter = _stream(protocol, "decoy_noise", i)
    g = sremu_rng.generator(int(protocol["seed"]), purpose, name, counter=counter)
    n = int(protocol["data"]["n_cells"])
    lo, hi = (float(x) for x in pa["sigma_dig_db_range"])
    s_dig = g.uniform(lo, hi, size=n)
    z1 = g.standard_normal(n)
    z2 = g.standard_normal(n)
    return s_dig, z1, z2


def _target_mean(protocol: Mapping[str, Any]) -> float:
    return float(np.mean(protocol["data"]["disclosed_descriptive_statistics"]["ring_means_dbm"]))


def decoy_dataset(
    i: int,
    truth_pred_db: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    sigma_rep_db: float | None = None,
    truth_model: str | None = None,
) -> tuple[an.Dataset, float]:
    """Decoy ``i`` from its truth predictions ``truth_pred_db[30, 4]`` (MODELS order)."""
    pa = protocol["power_analysis"]
    srep = float(pa["sigma_rep_db"] if sigma_rep_db is None else sigma_rep_db)
    tm = str(pa["truth_model"] if truth_model is None else truth_model)
    p = np.asarray(truth_pred_db, dtype=np.float64)[:, MODELS.index(tm)]
    c = _target_mean(protocol) - float(np.mean(p))
    s_dig, z1, z2 = decoy_noise(i, protocol)
    y = p + c + srep * z1 + s_dig * z2
    ds = an.Dataset(
        y_dbm=y,
        sigma_dig_db=s_dig,
        label=f"decoy {i} (truth {tm}, sigma_rep {srep:g} dB)",
        data_class="SIM-sremu",
    )
    return ds, c


def synthetic_decoy(
    seed: int,
    *,
    protocol: Mapping[str, Any] | None = None,
    sigma_rep_db: float | None = None,
    truth_model: str | None = None,
    backend: str = "numba_cpu",
) -> Decoy:
    """Decoy number ``seed``: M2 (or ``truth_model``) at a fresh keyed prior draw plus
    ``sigma_rep`` and ``sigma_dig`` noise. The development stand-in for the measured map."""
    proto = protocol if protocol is not None else load_protocol()
    purpose, name, counter = _stream(proto, "decoy_truth", int(seed))
    draw, dp = draw_and_evaluate(
        proto, purpose=purpose, name=name, counter=counter, backend=backend
    )
    pred = dp.power_db.T  # [30, 4]
    ds, c = decoy_dataset(
        int(seed), pred, proto, sigma_rep_db=sigma_rep_db, truth_model=truth_model
    )
    pa = proto["power_analysis"]
    return Decoy(
        index=int(seed),
        dataset=ds,
        truth_draw=draw,
        truth_pred_db=pred,
        truth_model=str(pa["truth_model"] if truth_model is None else truth_model),
        offset_dbm=c,
        sigma_rep_db=float(pa["sigma_rep_db"] if sigma_rep_db is None else sigma_rep_db),
    )


# =========================================================================== analysis


def analyze_dataset(
    tensor: TensorCache,
    data: an.Dataset,
    protocol: Mapping[str, Any],
    *,
    n_boot: int,
    rng: np.random.Generator,
    with_snf: bool = True,
) -> dict[str, Any]:
    """Primary (bootstrapped, decisions) and secondary analyses on the primary split; with
    ``with_snf`` also the S-NF split (offsets fitted without the 1.8 m ring) with the same
    bootstrap indices (its test cells are the primary ones)."""
    lay = cell_layout(protocol)
    train, test = split_masks(protocol, lay)["primary"]
    idx = rng.integers(0, int(test.sum()), size=(int(n_boot), int(test.sum())))
    prim = an.run_split(tensor, data, protocol, train=train, test=test, boot_idx=idx, layout=lay)
    sec = an.secondary(
        prim,
        data,
        train=train,
        test=test,
        sigma_tot_db=float(protocol["secondary_analysis"]["sigma_tot_db"]),
    )
    out: dict[str, Any] = {"primary": prim, "secondary": sec}
    if with_snf:
        sp = sensitivity_run(protocol, "S-NF")["split"]
        tr2, te2 = ring_split(protocol, sp["train_rings_m"], sp["test_rings_m"], lay)
        if not np.array_equal(te2, test):
            raise ValueError("S-NF must test the primary test cells (shared bootstrap indices)")
        out["S-NF"] = an.run_split(
            tensor, data, protocol, train=tr2, test=te2, boot_idx=idx, layout=lay
        )
    return out


def _decoy_row(
    i: int, res: Mapping[str, Any], truth_model: str, sigma_rep: float
) -> dict[str, Any]:
    prim, sec = res["primary"], res["secondary"]
    d = prim["decisions"]
    return {
        "decoy": i,
        "truth_model": truth_model,
        "sigma_rep_db": sigma_rep,
        "C1": d["C1"]["holds"],
        "C2": d["C2"]["holds"],
        "C3": d["C3"]["holds"],
        "c2_median_delta_db": d["C2"]["median_delta_rmse_db"],
        "c2_frac_ci_excl0": d["C2"]["frac_ci_excludes_zero"],
        "c2_median_ok": d["C2"]["median_ok"],
        "c2_frac_ok": d["C2"]["frac_ok"],
        "spearman_rho": d["C2"]["spearman_rho"],
        "spearman_ci_low": d["C2"]["spearman_ci_low"],
        "spearman_ok": d["C2"]["spearman_ok"],
        "c3_median_delta_db": d["C3"]["median_delta_rmse_db"],
        "c3_frac_ci_excl0": d["C3"]["frac_ci_excludes_zero"],
        "c1_median_abs_bias_db": d["C1"]["median_abs_bias_db"],
        "median_rmse_m2_db": d["C1"]["median_rmse_m2_db"],
        "median_rmse_m0n_db": d["C1"]["median_rmse_m0n_db"],
        "median_rmse_m0_db": prim["metrics"]["M0"]["rmse"]["median"],
        "secondary_delta_m0_m2_db": sec["delta_rmse"]["M0-M2"],
        "secondary_rmse_m2_db": sec["models"]["M2"]["rmse"],
        **_snf_fields(res),
    }


def _snf_fields(res: Mapping[str, Any]) -> dict[str, Any]:
    if "S-NF" not in res:
        return {}
    d = res["S-NF"]["decisions"]
    return {
        "snf_C1": d["C1"]["holds"],
        "snf_C2": d["C2"]["holds"],
        "snf_C3": d["C3"]["holds"],
        "snf_c2_median_delta_db": d["C2"]["median_delta_rmse_db"],
        "snf_c2_median_ok": d["C2"]["median_ok"],
        "snf_c2_frac_ok": d["C2"]["frac_ok"],
        "snf_spearman_ok": d["C2"]["spearman_ok"],
    }


_WORKER: dict[str, Any] = {}


def _init_decoy_worker(tensor_path: str, protocol: dict[str, Any], truth: np.ndarray) -> None:
    _WORKER["tensor"] = load_cache(tensor_path, protocol=protocol)
    _WORKER["protocol"] = protocol
    _WORKER["truth"] = truth


def _decoy_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    tensor: TensorCache = _WORKER["tensor"]
    protocol = _WORKER["protocol"]
    truth = _WORKER["truth"]
    rows = []
    for i in task["indices"]:
        purpose, name, _c = _stream(protocol, "decoy_bootstrap", i)
        counter = int(protocol["rng"]["decoy_bootstrap"]["counter"])
        for tm in task["truth_models"]:
            for s in task["sigma_reps"]:
                ds, _c0 = decoy_dataset(i, truth[i], protocol, sigma_rep_db=s, truth_model=tm)
                g = sremu_rng.generator(int(protocol["seed"]), purpose, name, counter=counter)
                res = analyze_dataset(tensor, ds, protocol, n_boot=task["n_boot"], rng=g)
                rows.append(_decoy_row(i, res, tm, s))
    return rows


def run_power_analysis(
    tensor_path: str | Path,
    truth_pred_db: np.ndarray,
    protocol: Mapping[str, Any],
    *,
    n_decoys: int | None = None,
    sigma_reps: Sequence[float] | None = None,
    truth_models: Sequence[str] = ("M2", "M0"),
    n_boot: int | None = None,
    jobs: int = 1,
    chunk: int = 5,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Per-decoy decision rows for every (truth model, sigma_rep); deterministic order."""
    pa = protocol["power_analysis"]
    n = int(pa["n_decoys"] if n_decoys is None else n_decoys)
    srs = (
        [float(pa["sigma_rep_db"]), *[float(x) for x in pa["sensitivity_sigma_rep_db"]]]
        if sigma_reps is None
        else [float(x) for x in sigma_reps]
    )
    nb = int(pa["n_bootstrap"] if n_boot is None else n_boot)
    tasks = [
        {
            "indices": list(range(i, min(n, i + chunk))),
            "sigma_reps": srs,
            "truth_models": list(truth_models),
            "n_boot": nb,
        }
        for i in range(0, n, chunk)
    ]
    init = (str(tensor_path), dict(protocol), np.asarray(truth_pred_db))
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if jobs <= 1:
        _init_decoy_worker(*init)
        for j, t in enumerate(tasks):
            rows.extend(_decoy_task(t))
            if progress:
                print(
                    f"  decoys {min((j + 1) * chunk, n)}/{n}, {time.perf_counter() - t0:.1f} s",
                    flush=True,
                )
    else:
        with cf.ProcessPoolExecutor(
            max_workers=jobs, initializer=_init_decoy_worker, initargs=init
        ) as ex:
            futs = [ex.submit(_decoy_task, t) for t in tasks]
            for j, f in enumerate(cf.as_completed(futs)):
                rows.extend(f.result())
                if progress and (j + 1) % max(1, len(futs) // 20) == 0:
                    print(
                        f"  chunks {j + 1}/{len(futs)}, {time.perf_counter() - t0:.1f} s",
                        flush=True,
                    )
    rows.sort(
        key=lambda r: (r["truth_model"] != "M2", r["truth_model"], r["sigma_rep_db"], r["decoy"])
    )
    return rows


def summarize_power(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Detection rates (Clopper-Pearson 95 %) per (truth model, sigma_rep) and the
    'underpowered' decision (truth M2, nominal sigma_rep)."""
    pa = protocol["power_analysis"]
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r["truth_model"]), float(r["sigma_rep_db"])), []).append(r)
    res: dict[str, Any] = {}
    first = str(pa["truth_model"])
    for (tm, s), rs in sorted(groups.items(), key=lambda kv: (kv[0][0] != first, kv[0])):
        n = len(rs)
        rec: dict[str, Any] = {"truth_model": tm, "sigma_rep_db": s, "n_decoys": n}
        for c in ("C1", "C2", "C3"):
            k = int(sum(bool(r[c]) for r in rs))
            lo, hi = clopper_pearson(k, n, 0.95)
            rec[f"{c}_rate"] = k / n
            rec[f"{c}_ci95"] = [lo, hi]
            rec[f"{c}_count"] = k
        for sub in ("c2_median_ok", "c2_frac_ok", "spearman_ok"):
            rec[f"{sub}_rate"] = float(np.mean([bool(r[sub]) for r in rs]))
        if all("snf_C2" in r for r in rs):
            for c in ("C1", "C2", "C3"):
                k = int(sum(bool(r[f"snf_{c}"]) for r in rs))
                rec[f"snf_{c}_rate"] = k / n
                rec[f"snf_{c}_count"] = k
                rec[f"snf_{c}_ci95"] = list(clopper_pearson(k, n, 0.95))
            for sub in ("c2_median_ok", "c2_frac_ok", "spearman_ok"):
                rec[f"snf_{sub}_rate"] = float(np.mean([bool(r[f"snf_{sub}"]) for r in rs]))
            rec["snf_c2_median_delta_db"] = an.summarize(
                np.array([r["snf_c2_median_delta_db"] for r in rs], dtype=np.float64)
            )
        for q in (
            "c2_median_delta_db",
            "c2_frac_ci_excl0",
            "spearman_rho",
            "c3_median_delta_db",
            "median_rmse_m2_db",
            "median_rmse_m0_db",
            "secondary_delta_m0_m2_db",
            "secondary_rmse_m2_db",
        ):
            rec[q] = an.summarize(np.array([r[q] for r in rs], dtype=np.float64))
        res[f"{tm}/sigma_rep_{s:g}"] = rec
    key = f"{pa['truth_model']}/sigma_rep_{float(pa['sigma_rep_db']):g}"
    main = res[key]
    thr = float(pa["underpowered_if_c2_rate_below"])
    null_key = f"{pa['null_decoys']['truth_model']}/sigma_rep_{float(pa['sigma_rep_db']):g}"
    spec: dict[str, Any] = {}
    for gk, rec in res.items():
        if rec["truth_model"] != pa["null_decoys"]["truth_model"]:
            continue
        lo, hi = rec["C2_ci95"]
        spec[gk] = {
            "c2_false_detection_rate": rec["C2_rate"],
            "c2_false_detection_ci95": [lo, hi],
            "specificity": 1.0 - rec["C2_rate"],
            "specificity_ci95": [1.0 - hi, 1.0 - lo],
        }
    return {
        "primary_key": key,
        "null_key": null_key if null_key in res else None,
        "c2_detection_rate": main["C2_rate"],
        "c2_detection_ci95": main["C2_ci95"],
        "underpowered_threshold": thr,
        "underpowered": bool(main["C2_rate"] < thr),
        "specificity": spec,
        "groups": res,
    }


def prediction_spread_comparison(
    tensor: TensorCache,
    protocol: Mapping[str, Any],
    reference: TensorCache | None = None,
    reference_label: str = "v1",
) -> dict[str, Any]:
    """Data-independent spread of the prior predictions (:func:`analysis.prediction_spread`)
    of ``tensor`` and, if given, of a ``reference`` tensor of an earlier protocol version
    (same cells and primary split)."""
    lay = cell_layout(protocol)
    train, test = split_masks(protocol, lay)["primary"]
    out: dict[str, Any] = {
        "definition": (
            "SD over prior draws of the fitted prediction per cell (sigma_geom, section 9; "
            "data-independent), its median over all cells and over the test cells; RIS "
            "signature = SD over the 30 cells of M2 - M0 per draw (quantiles over draws); "
            "ring-mean-removed M2 SD = median over test cells of the SD over draws of the "
            "ring-mean-removed M2 prediction"
        ),
        "current": an.prediction_spread(tensor.pred_db, train, test, lay.ring),
    }
    if reference is not None:
        out[reference_label] = an.prediction_spread(reference.pred_db, train, test, lay.ring)
        out["reference_label"] = reference_label
        out["reference_tensor_content_sha256"] = reference.meta.get("content_sha256")
    return out


# =========================================================================== protocol result


def write_protocol_result(
    summary: Mapping[str, Any], protocol_dir: str | Path | None = None
) -> None:
    """Write the power-analysis result into ``protocol.json`` (``power_analysis_result``) and
    the marked block of ``PROTOCOL.md``. Run before the lock."""
    d = Path(protocol_dir) if protocol_dir is not None else PROTOCOL_DIR
    pj = d / PROTOCOL_JSON
    proto = json.loads(pj.read_text(encoding="utf-8"))
    proto["power_analysis_result"] = summary["protocol_result"]
    pj.write_text(
        json.dumps(proto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path = d / PROTOCOL_MD
    md = md_path.read_text(encoding="utf-8")
    pat = re.compile(re.escape(RESULT_BEGIN) + r".*?" + re.escape(RESULT_END), re.S)
    if not pat.search(md):
        raise ValueError("PROTOCOL.md lacks the power-analysis result markers")
    md = pat.sub(
        lambda _m: RESULT_BEGIN + "\n" + summary["protocol_result_md"] + "\n" + RESULT_END, md
    )
    md_path.write_text(md, encoding="utf-8", newline="\n")


def _fmt_rate(rec: Mapping[str, Any], c: str) -> str:
    lo, hi = rec[f"{c}_ci95"]
    return (
        f"{rec[f'{c}_rate']:.3f} ({rec[f'{c}_count']}/{rec['n_decoys']}; 95 % CI {lo:.3f}-{hi:.3f})"
    )


def _protocol_result(
    summ: Mapping[str, Any], meta: Mapping[str, Any], run: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    g = summ["groups"]
    pr = {
        "computed_utc": run["t_utc"],
        "run_directory": run["run_dir"],
        "sremu_commit": run["git_commit"],
        "sremu_dirty": run["git_dirty"],
        "tensor_content_sha256": meta["content_sha256"],
        "decoy_truth_content_sha256": run["truth_sha256"],
        "prediction_spec_sha256": meta["prediction_spec_sha256"],
        "n_decoys": g[summ["primary_key"]]["n_decoys"],
        "n_bootstrap_per_decoy": run["n_boot"],
        "c2_detection_rate": summ["c2_detection_rate"],
        "c2_detection_ci95": summ["c2_detection_ci95"],
        "underpowered": summ["underpowered"],
        "rates": {k: {c: v[f"{c}_rate"] for c in ("C1", "C2", "C3")} for k, v in g.items()},
        "c2_subcondition_rates": {
            k: {c: v[f"{c}_rate"] for c in ("c2_median_ok", "c2_frac_ok", "spearman_ok")}
            for k, v in g.items()
        },
        "median_over_decoys": {
            k: {
                "c2_median_delta_db": v["c2_median_delta_db"]["median"],
                "secondary_delta_m0_m2_db": v["secondary_delta_m0_m2_db"]["median"],
                "median_rmse_m2_db": v["median_rmse_m2_db"]["median"],
                "median_rmse_m0_db": v["median_rmse_m0_db"]["median"],
            }
            for k, v in g.items()
        },
        "specificity": summ.get("specificity", {}),
    }
    snf = {k: v for k, v in g.items() if "snf_C2_rate" in v}
    if snf:
        pr["s_nf_informational"] = {
            k: {
                "C1": v["snf_C1_rate"],
                "C2": v["snf_C2_rate"],
                "C2_ci95": v["snf_C2_ci95"],
                "C3": v["snf_C3_rate"],
                "c2_median_ok": v["snf_c2_median_ok_rate"],
                "c2_frac_ok": v["snf_c2_frac_ok_rate"],
                "spearman_ok": v["snf_spearman_ok_rate"],
                "median_delta_rmse_db": v["snf_c2_median_delta_db"]["median"],
            }
            for k, v in snf.items()
        }
    if run.get("prediction_spread") is not None:
        pr["prediction_spread"] = run["prediction_spread"]
    lines = [
        f"Computed {run['t_utc']} (sremu commit {run['git_commit']}, dirty {run['git_dirty']}) "
        f"from tensor `{meta['content_sha256'][:16]}...` (K = {meta['K']}), "
        f"{pr['n_decoys']} decoys, {run['n_boot']} bootstrap resamples each.",
        "",
        "| Truth | sigma_rep (dB) | C1 rate | C2 rate (95 % CI) | C3 rate |",
        "|---|---|---|---|---|",
    ]
    for v in g.values():
        lines.append(
            f"| {v['truth_model']} | {v['sigma_rep_db']:g} | {v['C1_rate']:.3f} "
            f"| {_fmt_rate(v, 'C2')} | {v['C3_rate']:.3f} |"
        )
    lines += [
        "",
        "C2 sub-conditions (fraction of decoys meeting each) and medians over decoys:",
        "",
        "| Truth | sigma_rep (dB) | median Delta RMSE >= 1 dB | CI excludes 0 in >= 80 % "
        "of draws | Spearman | median Delta RMSE (dB) | secondary Delta RMSE (dB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for v in g.values():
        lines.append(
            f"| {v['truth_model']} | {v['sigma_rep_db']:g} | {v['c2_median_ok_rate']:.3f} "
            f"| {v['c2_frac_ok_rate']:.3f} | {v['spearman_ok_rate']:.3f} "
            f"| {v['c2_median_delta_db']['median']:.2f} "
            f"| {v['secondary_delta_m0_m2_db']['median']:.2f} |"
        )
    lines += [
        "",
        f"**C2 detection rate (truth M2, sigma_rep 1 dB): {summ['c2_detection_rate']:.3f}** "
        f"(95 % CI {summ['c2_detection_ci95'][0]:.3f}-{summ['c2_detection_ci95'][1]:.3f}). "
        + (
            "This is below 0.5, so the study is labelled **underpowered** whatever the outcome."
            if summ["underpowered"]
            else "This is at least 0.5, so the 'underpowered' label does not apply."
        ),
        "Rows with truth M0 are the informational specificity check (C2 should rarely hold).",
    ]
    for nk, sp in (summ.get("specificity") or {}).items():
        if nk.split("/", 1)[1] != str(summ["primary_key"]).split("/", 1)[1]:
            continue
        nv = g[nk]
        lo, hi = sp["c2_false_detection_ci95"]
        lines.append(
            f"Specificity (truth {nv['truth_model']}, no RIS path, sigma_rep "
            f"{nv['sigma_rep_db']:g} dB): C2 fires for {nv['C2_rate']:.3f} "
            f"({nv['C2_count']}/{nv['n_decoys']}; 95 % CI {lo:.3f}-{hi:.3f}) of the decoys, "
            f"i.e. specificity {sp['specificity']:.3f}."
        )
    pk = str(summ["primary_key"])
    pm = g[pk]
    lines.append(
        f"Under the {pm['truth_model']} truth (sigma_rep {pm['sigma_rep_db']:g} dB) C1 holds for "
        f"{pm['C1_rate']:.3f} and C3 for {pm['C3_rate']:.3f} of the decoys; C1 compares the "
        "held-out rings after an offset fitted on the training rings, so it does not test the "
        "absolute received power."
    )
    suffix = pk.split("/", 1)[1]
    for nk, nv in g.items():
        if nk == pk or nk.split("/", 1)[1] != suffix:
            continue
        a, b = float(pm["spearman_ok_rate"]), float(nv["spearman_ok_rate"])
        lines.append(
            f"The Spearman sub-condition of C2 holds for {b:.3f} of the decoys whose truth is "
            f"{nv['truth_model']} (no RIS path) and for {a:.3f} with the {pm['truth_model']} "
            "truth"
            + (", so on its own it does not indicate RIS-specific structure." if b >= a else ".")
        )
    if snf:
        lines += [
            "",
            "S-NF (information only; offsets fitted without the 1.8 m ring, same test cells and "
            "bootstrap indices):",
            "",
            "| Truth | sigma_rep (dB) | C1 rate | C2 rate (95 % CI) | C3 rate | median Delta "
            "RMSE (dB) |",
            "|---|---|---|---|---|---|",
        ]
        for v in snf.values():
            lo, hi = v["snf_C2_ci95"]
            lines.append(
                f"| {v['truth_model']} | {v['sigma_rep_db']:g} | {v['snf_C1_rate']:.3f} "
                f"| {v['snf_C2_rate']:.3f} ({v['snf_C2_count']}/{v['n_decoys']}; 95 % CI "
                f"{lo:.3f}-{hi:.3f}) | {v['snf_C3_rate']:.3f} "
                f"| {v['snf_c2_median_delta_db']['median']:.2f} |"
            )
    spread = run.get("prediction_spread")
    if spread is not None and spread.get("reference_label"):
        ref = spread["reference_label"]
        cur, old = spread["current"], spread[ref]
        lines += [
            "",
            f"Spread of the prior predictions ({ref} prior vs this prior; data-independent):",
            "",
            f"| Quantity | {ref} | this protocol |",
            "|---|---|---|",
        ]
        for m in ("M0", "M2"):
            lines.append(
                f"| median over cells of the SD over draws of the fitted {m} prediction (dB) "
                f"| {old['models'][m]['median_sd_all_cells_db']:.2f} "
                f"| {cur['models'][m]['median_sd_all_cells_db']:.2f} |"
            )
            lines.append(
                f"| the same over the test cells (dB) "
                f"| {old['models'][m]['median_sd_test_cells_db']:.2f} "
                f"| {cur['models'][m]['median_sd_test_cells_db']:.2f} |"
            )
        lines.append(
            "| RIS signature: median over draws of the SD over cells of M2 - M0 (dB) "
            f"| {old['ris_signature_db']['median']:.2f} | {cur['ris_signature_db']['median']:.2f} |"
        )
        lines.append(
            "| ring-mean-removed M2: median over test cells of the SD over draws (dB) "
            f"| {old['ring_removed_m2_sd_db']:.2f} | {cur['ring_removed_m2_sd_db']:.2f} |"
        )
        ref_delta = spread.get("reference_median_delta_rmse_truth_m2_db")
        if ref_delta is not None:
            lines.append(
                "| median over decoys of the median Delta RMSE (M0 - M2), truth M2, sigma_rep "
                f"1 dB (dB) | {ref_delta:.2f} | {pm['c2_median_delta_db']['median']:.2f} |"
            )
    return pr, "\n".join(lines)


# =========================================================================== CLI


def main(argv: Sequence[str] | None = None) -> int:
    """Decoy truths + power analysis into a new run directory."""
    from validation.ants_fig4b import runlog

    ap = argparse.ArgumentParser(prog="python -m validation.ants_fig4b.decoy")
    ap.add_argument("--tensor", type=Path, required=True, help="pred_tensor.npz of predict.py")
    ap.add_argument("--out", type=Path, default=Path("runs") / "vlab")
    ap.add_argument("--n", type=int, default=None, help="decoys (default: protocol)")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--truth-cache", type=Path, default=None, help="reuse decoy_truth.npz")
    ap.add_argument("--write-protocol-result", action="store_true")
    ap.add_argument(
        "--reference-tensor",
        type=Path,
        default=None,
        help="tensor of an earlier protocol version for the prediction-spread comparison",
    )
    ap.add_argument("--reference-label", default="v1")
    ap.add_argument("--name", default="vlab_power")
    ap.add_argument("--bg-seconds", type=float, default=5.0, help="background-load probe")
    args = ap.parse_args(argv)
    protocol = load_protocol()
    tensor = load_cache(args.tensor, protocol=protocol)
    pa = protocol["power_analysis"]
    n = int(pa["n_decoys"] if args.n is None else args.n)
    nb = int(pa["n_bootstrap"] if args.n_boot is None else args.n_boot)
    rec = runlog.RunRecorder(args.out, args.name, protocol, argv=sys.argv)
    rec.add_input(args.tensor, "SIM-sremu")
    bg_before = runlog.background_load(args.bg_seconds)
    rec.sample_load()
    t0 = time.perf_counter()
    if args.truth_cache is not None:
        tr = load_cache(args.truth_cache)
        truth_arrays, truth_meta = tr.arrays, dict(tr.meta)
        rec.add_input(args.truth_cache, "SIM-sremu")
    else:
        res = evaluate_stream(
            protocol, protocol["rng"]["decoy_truth"], n, jobs=args.jobs, progress=True
        )
        truth_meta = save_cache(
            rec.run_dir / "decoy_truth.npz", res, protocol, extra_meta={"kind": "vlab_decoy_truth"}
        )
        truth_arrays = res["arrays"]
        rec.add_output("decoy_truth.npz", "SIM-sremu")
        rec.write_csv("decoy_truth_draws.csv", [d.as_record() for d in res["draws"]])
        rec.add_output("decoy_truth_draws.csv", "ASSUMED")
    t_truth = time.perf_counter() - t0
    rec.sample_load()
    truth = truth_arrays["pred_db"]  # [n, 30, 4]
    rows = run_power_analysis(
        args.tensor, truth, protocol, n_decoys=n, n_boot=nb, jobs=args.jobs, progress=True
    )
    t_all = time.perf_counter() - t0
    rec.sample_load()
    summ = summarize_power(rows, protocol)
    reference = None
    if args.reference_tensor is not None:
        reference = load_cache(args.reference_tensor)
        rec.add_input(args.reference_tensor, "SIM-sremu")
    spread = prediction_spread_comparison(tensor, protocol, reference, args.reference_label)
    if reference is not None:
        for rh in protocol.get("revision_history", []):
            pw = rh.get("power") or {}
            if pw.get("tensor_content_sha256") == reference.meta.get("content_sha256"):
                spread["reference_median_delta_rmse_truth_m2_db"] = pw.get(
                    "median_delta_rmse_m0_m2_db_truth_m2_sigma_rep_1"
                )
    run_info = {
        "t_utc": utc_now_iso(),
        "run_dir": str(rec.run_dir),
        "git_commit": rec.git.commit,
        "git_dirty": rec.git.dirty,
        "truth_sha256": truth_meta["content_sha256"],
        "n_boot": nb,
        "prediction_spread": spread,
    }
    pr, pr_md = _protocol_result(summ, tensor.meta, run_info)
    summ_out = {
        **summ,
        "tensor": {k: tensor.meta[k] for k in ("content_sha256", "prediction_spec_sha256", "K")},
        "wall_s": {"truths": t_truth, "total": t_all},
        "background_cpu_busy_fraction_before": {"probe_s": args.bg_seconds, "value": bg_before},
        "jobs": args.jobs,
        "protocol_result": pr,
        "protocol_result_md": pr_md,
    }
    rec.write_csv("decoys.csv", rows)
    rec.add_output("decoys.csv", "SIM-sremu")
    rec.write_json("summary.json", summ_out)
    rec.add_output("summary.json", "SIM-sremu")
    rec.finish(
        backend="numba_cpu",
        extra={
            "power": {
                k: summ[k] for k in ("c2_detection_rate", "c2_detection_ci95", "underpowered")
            }
        },
    )
    print(
        json.dumps(
            {
                k: summ_out[k]
                for k in ("c2_detection_rate", "c2_detection_ci95", "underpowered", "wall_s")
            },
            indent=2,
        )
    )
    print(pr_md)
    print(f"run directory: {rec.run_dir}")
    if args.write_protocol_result:
        if prediction_spec_sha256(protocol) != tensor.meta["prediction_spec_sha256"]:
            raise SystemExit(
                "tensor prediction spec differs from the protocol; not writing the result"
            )
        write_protocol_result(summ_out)
        print("power-analysis result written into protocol.json and PROTOCOL.md")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")
    sys.exit(main())
