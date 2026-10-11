"""The V-LAB protocol pipeline (DESIGN.md section 10.9; every number from ``protocol.json``).

Inputs: the cached tensor (``pred[K, 30, models]``, ``dist_m[K, 30]``) and a dataset
``y[30]`` (dBm) with ``sigma_dig[30]`` (dB). For every draw ``k`` and model ``m``:

* offset ``C_km = mean_train(y - P_km)`` (one parameter; ``M0n`` fits ``(C, n)`` by OLS on
  ``x = -10 log10 d``); fitted prediction ``yhat = P + C``; errors ``e = y - yhat``;
* test RMSE, bias (mean e), MAE, max |e|; their median and 5-95 % over draws;
* ``Delta RMSE = RMSE(M0) - RMSE(M2)`` (C2) and ``RMSE(M1) - RMSE(M2)`` (C3) per draw, with
  a paired percentile bootstrap over test cells (indices ``rng.integers(0, n, (B, n))`` shared
  by all draws, offsets not refitted; the semantics of
  :func:`sremu.analysis.stats.paired_bootstrap` with ``rmse_difference``);
* Spearman rho (Fieller 95 % CI, :func:`sremu.analysis.stats.spearman`) between the
  ring-mean-removed measured test values and the ring-mean-removed prior-median M2
  prediction; per-draw rho is reported too;
* secondary "optimistic" analysis: weights ``w_k ~ exp(-SSE_train,k/(2 sigma_tot^2))``.

Decisions C1-C3 follow ``decision_rules`` exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from validation.ants_fig4b.predict import MODELS, TensorCache
from validation.ants_fig4b.protocol import CellLayout, cell_layout, split_masks

from sremu.analysis import stats as sstats

__all__ = [
    "ALL_MODELS",
    "Dataset",
    "bootstrap_delta_ci",
    "breakdown",
    "decide",
    "fitted_predictions",
    "pinned_cells",
    "prediction_spread",
    "ring_removed",
    "run_folds",
    "run_split",
    "secondary",
    "summarize",
    "test_metrics",
    "uncertainty_budget",
]

ALL_MODELS: tuple[str, ...] = ("M0", "M0n", "M1", "M2", "M3")


@dataclass(frozen=True, eq=False)
class Dataset:
    """A map in protocol cell order: ``y_dbm[30]``, ``sigma_dig_db[30]``, its label and data
    class (``DIG`` for the measured map, ``SIM-sremu`` for decoys)."""

    y_dbm: np.ndarray
    sigma_dig_db: np.ndarray
    label: str
    data_class: str


def summarize(x: np.ndarray, qs: Sequence[float] = (0.05, 0.5, 0.95)) -> dict[str, Any]:
    """``{"q05", "median", "q95", "mean"}`` of a 1-D array (NaN ignored)."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"q05": math.nan, "median": math.nan, "q95": math.nan, "mean": math.nan, "n": 0}
    q = np.quantile(a, qs)
    return {
        "q05": float(q[0]),
        "median": float(q[1]),
        "q95": float(q[2]),
        "mean": float(a.mean()),
        "n": int(a.size),
    }


# =========================================================================== fits


def fitted_predictions(
    tensor: TensorCache, y: np.ndarray, train: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fitted predictions ``{model: yhat[K, 30]}`` and fitted parameters
    (``C_<model>[K]``, ``n_M0n[K]``) with the offsets fitted on ``train`` cells only."""
    yy = np.asarray(y, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    par: dict[str, np.ndarray] = {}
    for m in MODELS:
        p = tensor.model(m)
        c = np.mean(yy[train][None, :] - p[:, train], axis=1)
        out[m] = p + c[:, None]
        par[f"C_{m}"] = c
    x = -10.0 * np.log10(tensor.arrays["dist_m"])
    xt = x[:, train]
    yt = yy[train]
    xm = xt.mean(axis=1)
    ym = yt.mean()
    sxx = np.sum((xt - xm[:, None]) ** 2, axis=1)
    sxy = np.sum((xt - xm[:, None]) * (yt - ym)[None, :], axis=1)
    n = sxy / sxx
    c0 = ym - n * xm
    out["M0n"] = c0[:, None] + n[:, None] * x
    par["C_M0n"] = c0
    par["n_M0n"] = n
    return out, par


def test_metrics(e: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Per-draw RMSE, bias, MAE and max |e| over the ``mask`` cells of ``e[K, 30]``."""
    et = e[:, mask]
    return {
        "rmse": sstats.rmse(et, axis=1),
        "bias": np.mean(et, axis=1),
        "mae": np.mean(np.abs(et), axis=1),
        "max_abs": np.max(np.abs(et), axis=1),
    }


def bootstrap_delta_ci(
    errors: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[str, str]],
    mask: np.ndarray,
    idx: np.ndarray,
    *,
    confidence: float = 0.95,
    chunk_bytes: int = 64 * 2**20,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Per-draw percentile CI of ``RMSE(a) - RMSE(b)`` over bootstrap resamples ``idx[B, n]``
    of the ``mask`` cells, for each ``(a, b)`` in ``pairs``. Identical, draw by draw, to
    :func:`sremu.analysis.stats.paired_bootstrap` of ``rmse_difference`` with the same
    indices."""
    models = sorted({m for p in pairs for m in p})
    et = {m: np.ascontiguousarray(errors[m][:, mask]) for m in models}
    k = next(iter(et.values())).shape[0]
    b, n = idx.shape
    alpha = (1.0 - confidence) / 2.0
    q = np.array([alpha, 1.0 - alpha])
    step = max(1, int(chunk_bytes // (b * n * 8 * max(1, len(models)))))
    lo = {p: np.empty(k) for p in pairs}
    hi = {p: np.empty(k) for p in pairs}
    for s in range(0, k, step):
        sl = slice(s, min(k, s + step))
        r = {m: sstats.rmse(et[m][sl][:, idx], axis=-1) for m in models}  # (c, B)
        for p in pairs:
            dist = r[p[0]] - r[p[1]]
            qq = np.quantile(dist, q, axis=1)
            lo[p][sl] = qq[0]
            hi[p][sl] = qq[1]
    return {p: (lo[p], hi[p]) for p in pairs}


def ring_removed(v: np.ndarray, mask: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Values on the ``mask`` cells (last axis) minus the mean of their ring's mask cells."""
    a = np.asarray(v, dtype=np.float64)[..., mask]
    rg = ring[mask]
    out = np.empty_like(a)
    for r in np.unique(rg):
        sel = rg == r
        out[..., sel] = a[..., sel] - a[..., sel].mean(axis=-1, keepdims=True)
    return out


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    da = a - a.mean(axis=-1, keepdims=True)
    db = b - b.mean(axis=-1, keepdims=True)
    num = np.sum(da * db, axis=-1)
    den = np.sqrt(np.sum(da * da, axis=-1) * np.sum(db * db, axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.clip(num / den, -1.0, 1.0)


def _spearman_block(
    y: np.ndarray,
    yhat: np.ndarray,
    mask: np.ndarray,
    ring: np.ndarray,
    *,
    ring_mean: bool,
    confidence: float,
) -> dict[str, Any]:
    if ring_mean:
        yr = ring_removed(y, mask, ring)
        pr = ring_removed(yhat, mask, ring)
    else:
        yr = np.asarray(y, dtype=np.float64)[mask]
        pr = yhat[:, mask]
    p_med = np.median(pr, axis=0)
    out: dict[str, Any] = {"n": int(yr.size), "ring_mean_removed": ring_mean}
    try:
        res = sstats.spearman(yr, p_med, confidence=confidence, ci_method="fieller")
        out.update(
            rho=res.rho,
            ci_low=res.ci_low,
            ci_high=res.ci_high,
            p_value=res.p_value,
            ci_method=res.ci_method,
        )
    except ValueError as exc:  # a constant sample: rho undefined
        out.update(
            rho=math.nan, ci_low=math.nan, ci_high=math.nan, p_value=math.nan, error=str(exc)
        )
    ry = sstats.rankdata(yr)
    rp = sstats.rankdata(pr, axis=-1)
    per = _pearson_rows(np.broadcast_to(ry, rp.shape), rp)
    out["per_draw"] = summarize(per)
    out["_per_draw"] = per
    out["_prior_median"] = p_med
    out["_measured"] = yr
    return out


# =========================================================================== one split


def run_split(
    tensor: TensorCache,
    data: Dataset,
    protocol: Mapping[str, Any],
    *,
    train: np.ndarray,
    test: np.ndarray,
    boot_idx: np.ndarray | None,
    layout: CellLayout | None = None,
    with_spearman: bool = True,
    ring_mean: bool = True,
) -> dict[str, Any]:
    """Fits, test metrics, Delta RMSE (with the bootstrap if ``boot_idx`` is given),
    Spearman and the decisions (when bootstrapped) for one train/test split. Keys starting
    with ``_`` hold arrays (per draw or per cell) for figures; the rest is JSON-ready."""
    lay = layout or cell_layout(protocol)
    y = np.asarray(data.y_dbm, dtype=np.float64)
    yhat, par = fitted_predictions(tensor, y, train)
    errs = {m: y[None, :] - yhat[m] for m in ALL_MODELS}
    met = {m: test_metrics(errs[m], test) for m in ALL_MODELS}
    out: dict[str, Any] = {
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "metrics": {m: {k: summarize(v) for k, v in met[m].items()} for m in ALL_MODELS},
        "fit": {k: summarize(v) for k, v in par.items()},
        "_yhat": yhat,
        "_metrics": met,
    }
    d02 = met["M0"]["rmse"] - met["M2"]["rmse"]
    d12 = met["M1"]["rmse"] - met["M2"]["rmse"]
    out["delta_rmse"] = {"M0-M2": summarize(d02), "M1-M2": summarize(d12)}
    out["_delta"] = {"M0-M2": d02, "M1-M2": d12}
    conf = float(protocol["bootstrap"]["confidence"])
    if with_spearman:
        out["spearman"] = _spearman_block(
            y,
            yhat["M2"],
            test,
            lay.ring,
            ring_mean=ring_mean,
            confidence=float(protocol["primary_analysis"]["spearman"]["confidence"]),
        )
    if boot_idx is not None:
        ci = bootstrap_delta_ci(errs, [("M0", "M2"), ("M1", "M2")], test, boot_idx, confidence=conf)
        out["bootstrap"] = {
            "n_resamples": int(boot_idx.shape[0]),
            "confidence": conf,
            "M0-M2": {
                "frac_ci_excludes_zero": float(np.mean(ci[("M0", "M2")][0] > 0.0)),
                "ci_low": summarize(ci[("M0", "M2")][0]),
                "ci_high": summarize(ci[("M0", "M2")][1]),
            },
            "M1-M2": {
                "frac_ci_excludes_zero": float(np.mean(ci[("M1", "M2")][0] > 0.0)),
                "ci_low": summarize(ci[("M1", "M2")][0]),
                "ci_high": summarize(ci[("M1", "M2")][1]),
            },
        }
        out["_ci"] = ci
        out["decisions"] = decide(out, protocol)
    return out


def decide(res: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    """C1-C3 from a bootstrapped :func:`run_split` result (``decision_rules``)."""
    dr = protocol["decision_rules"]
    met = res["_metrics"]
    c1r, c2r = dr["C1"], dr["C2"]
    med_abs_bias = float(np.median(np.abs(met["M2"]["bias"])))
    med_rmse_m2 = float(np.median(met["M2"]["rmse"]))
    med_rmse_m0n = float(np.median(met["M0n"]["rmse"]))
    c1_bias = med_abs_bias <= float(c1r["median_abs_bias_max_db"])
    c1_rmse = med_rmse_m2 <= med_rmse_m0n + float(c1r["rmse_margin_vs_m0n_db"])
    sp = res["spearman"]
    rho = float(sp["rho"])
    rho_ok = bool(
        math.isfinite(rho) and rho >= float(c2r["spearman_rho_min"]) and float(sp["ci_low"]) > 0.0
    )

    def c2_like(key: str) -> dict[str, Any]:
        med = float(np.median(res["_delta"][key]))
        frac = float(res["bootstrap"][key]["frac_ci_excludes_zero"])
        a = med >= float(c2r["median_delta_rmse_min_db"])
        b = frac >= float(c2r["ci_excludes_zero_min_fraction"])
        return {
            "median_delta_rmse_db": med,
            "median_ok": a,
            "frac_ci_excludes_zero": frac,
            "frac_ok": b,
            "spearman_rho": rho,
            "spearman_ci_low": float(sp["ci_low"]),
            "spearman_ok": rho_ok,
            "holds": bool(a and b and rho_ok),
        }

    c2 = c2_like("M0-M2")
    c3 = c2_like("M1-M2")
    return {
        "C1": {
            "median_abs_bias_db": med_abs_bias,
            "bias_ok": c1_bias,
            "median_rmse_m2_db": med_rmse_m2,
            "median_rmse_m0n_db": med_rmse_m0n,
            "rmse_ok": c1_rmse,
            "holds": bool(c1_bias and c1_rmse),
        },
        "C2": c2,
        "C3": c3,
    }


# =========================================================================== secondary


def secondary(
    split_result: Mapping[str, Any],
    data: Dataset,
    *,
    train: np.ndarray,
    test: np.ndarray,
    sigma_tot_db: float,
) -> dict[str, Any]:
    """The "optimistic" weighted analysis: per model, ``w_k ~ exp(-SSE_train,k/(2 sigma^2))``,
    weighted fitted prediction, its test metrics and the Kish ESS."""
    y = np.asarray(data.y_dbm, dtype=np.float64)
    out: dict[str, Any] = {"label": "optimistic", "sigma_tot_db": sigma_tot_db, "models": {}}
    preds: dict[str, np.ndarray] = {}
    for m in ALL_MODELS:
        yhat = split_result["_yhat"][m]
        sse = np.sum((y[None, train] - yhat[:, train]) ** 2, axis=1)
        z = -(sse - np.min(sse)) / (2.0 * sigma_tot_db**2)
        w = np.exp(z)
        w /= w.sum()
        pw = w @ yhat
        e = y - pw
        et = e[test]
        preds[m] = pw
        out["models"][m] = {
            "rmse": float(np.sqrt(np.mean(et**2))),
            "bias": float(np.mean(et)),
            "mae": float(np.mean(np.abs(et))),
            "max_abs": float(np.max(np.abs(et))),
            "ess": float(1.0 / np.sum(w**2)),
            "max_weight": float(np.max(w)),
        }
    mm = out["models"]
    out["delta_rmse"] = {
        "M0-M2": mm["M0"]["rmse"] - mm["M2"]["rmse"],
        "M1-M2": mm["M1"]["rmse"] - mm["M2"]["rmse"],
    }
    out["_pred"] = preds
    return out


# =========================================================================== folds etc.


def run_folds(tensor: TensorCache, data: Dataset, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Leave-one-ring-out and leave-one-wedge-out: per fold, median test metrics per model,
    Delta RMSE medians and (rings only) the prior-median Spearman rho. No decisions."""
    lay = cell_layout(protocol)
    out: dict[str, Any] = {}
    for name, (tr, te) in split_masks(protocol, lay).items():
        if name == "primary":
            continue
        ring_fold = name.startswith("loro")
        r = run_split(
            tensor,
            data,
            protocol,
            train=tr,
            test=te,
            boot_idx=None,
            layout=lay,
            with_spearman=ring_fold,
            ring_mean=False,
        )
        rec: dict[str, Any] = {
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "median_rmse": {m: r["metrics"][m]["rmse"]["median"] for m in ALL_MODELS},
            "median_abs_bias": {
                m: float(np.median(np.abs(r["_metrics"][m]["bias"]))) for m in ALL_MODELS
            },
            "delta_rmse_median": {k: v["median"] for k, v in r["delta_rmse"].items()},
        }
        if ring_fold:
            sp = r["spearman"]
            rec["spearman_prior_median"] = {k: sp[k] for k in ("rho", "ci_low", "ci_high", "n")}
        out[name] = rec
    return out


def pinned_cells(y: np.ndarray, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Indices of the colorbar-pinned cells (``data.pinned_rule``)."""
    d = protocol["data"]
    tol = float(d["pinned_match_tol_db"])
    yy = np.asarray(y, dtype=np.float64)
    idx: list[int] = []
    info: list[dict[str, Any]] = []
    for v in d["colorbar_pinned_dbm"]:
        hit = np.flatnonzero(np.abs(yy - float(v)) <= tol)
        if hit.size >= 1:
            i, fallback = int(hit[0]), False
        else:
            i = int(np.argmax(yy) if float(v) > float(np.median(yy)) else np.argmin(yy))
            fallback = True
        idx.append(i)
        info.append(
            {
                "pinned_dbm": float(v),
                "cell": i,
                "fallback_extreme": fallback,
                "n_matches": int(hit.size),
            }
        )
    return {"cells": sorted(set(idx)), "detail": info}


def uncertainty_budget(
    tensor: TensorCache,
    data: Dataset,
    primary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    test: np.ndarray,
) -> dict[str, Any]:
    """sigma_dig, sigma_pos (reference), sigma_rep, sigma_geom per cell and the test-cell
    coverage of the prior-median M2 prediction at sigma_rep in {nominal, sensitivity}."""
    ub = protocol["uncertainty_budget"]
    yhat = primary["_yhat"]["M2"]
    center = np.median(yhat, axis=0)
    s_geom = np.std(yhat, axis=0, ddof=1)
    s_dig = np.asarray(data.sigma_dig_db, dtype=np.float64)
    sp = tensor.arrays.get("sigma_pos_db")
    finite = None if sp is None else sp[np.all(np.isfinite(sp), axis=1)]
    s_pos = (
        np.median(finite, axis=0)
        if finite is not None and finite.shape[0] > 0
        else np.full(center.size, math.nan)
    )
    z = float(ub["coverage"]["z"])
    y = np.asarray(data.y_dbm, dtype=np.float64)
    cov = {}
    for s_rep in [float(ub["sigma_rep_db"]), *[float(x) for x in ub["sigma_rep_sensitivity_db"]]]:
        tot = np.sqrt(s_dig**2 + s_rep**2 + s_geom**2)
        inside = np.abs(y - center) <= z * tot
        cov[f"sigma_rep_{s_rep:g}"] = float(np.mean(inside[test]))
    return {
        "median_over_cells_db": {
            "sigma_dig": float(np.median(s_dig)),
            "sigma_pos": float(np.nanmedian(s_pos)) if np.any(np.isfinite(s_pos)) else None,
            "sigma_rep": float(ub["sigma_rep_db"]),
            "sigma_geom": float(np.median(s_geom)),
        },
        "coverage_test_cells": cov,
        "nominal_coverage": float(ub["coverage"]["nominal"]),
        "_sigma_geom": s_geom,
        "_sigma_pos": s_pos,
        "_center": center,
    }


def _group_stats(primary: Mapping[str, Any], sel: np.ndarray) -> dict[str, Any]:
    met = primary["_metrics"]
    if not np.any(sel):
        return {"n_draws": 0}
    return {
        "n_draws": int(sel.sum()),
        "median_rmse_m2": float(np.median(met["M2"]["rmse"][sel])),
        "median_rmse_m0": float(np.median(met["M0"]["rmse"][sel])),
        "median_delta_m0_m2": float(np.median(primary["_delta"]["M0-M2"][sel])),
        "median_delta_m1_m2": float(np.median(primary["_delta"]["M1-M2"][sel])),
    }


def breakdown(tensor: TensorCache, primary: Mapping[str, Any]) -> dict[str, Any]:
    """S4: median test RMSE(M2) and Delta RMSE per value of each discrete prior variable and,
    when the tensor has arc radii, per tertile of the radius R (edges: the tensor's own R
    tertiles, data-independent)."""
    out: dict[str, Any] = {}
    for key in tensor.meta["codes"]:
        lab = tensor.codes(key)
        out[key] = {str(v): _group_stats(primary, lab == v) for v in tensor.meta["codes"][key]}
    rad = tensor.arrays.get("radius_m")
    if rad is not None and np.all(np.isfinite(rad)):
        edges = np.quantile(rad, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        rec: dict[str, Any] = {}
        for i in range(3):
            lo, hi = float(edges[i]), float(edges[i + 1])
            sel = (rad >= lo) & ((rad < hi) if i < 2 else (rad <= hi))
            rec[f"[{lo:.4f}, {hi:.4f}] m"] = _group_stats(primary, sel)
        out["radius_tertile"] = rec
    return out


def prediction_spread(
    pred_db: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ring: np.ndarray,
    *,
    models: Sequence[str] = MODELS,
) -> dict[str, Any]:
    """Data-independent spread of the prior predictions (no measured or decoy values).

    The fitted prediction of draw ``k`` is ``P_k + mean_train(y) - mean_train(P_k)``, so its
    standard deviation over draws at cell ``i`` (``sigma_geom``, section 9) does not depend on
    the dataset: it is the SD over draws of ``P_k,i - mean_train(P_k)``. Per model: that SD's
    median over all cells and over the test cells. ``ris_signature_db``: quantiles over draws
    of the SD over the 30 cells of ``M2 - M0`` (the size of the RIS fingerprint a draw adds to
    LOS); ``ring_removed_m2_sd_db``: median over test cells of the SD over draws of the
    ring-mean-removed M2 prediction (the spread of the structure the Spearman condition
    uses)."""
    p = np.asarray(pred_db, dtype=np.float64)
    out: dict[str, Any] = {"K": int(p.shape[0]), "models": {}}
    for j, m in enumerate(models):
        pm = p[:, :, j]
        c = pm - pm[:, train].mean(axis=1, keepdims=True)
        sd = np.std(c, axis=0, ddof=1)
        out["models"][m] = {
            "median_sd_all_cells_db": float(np.median(sd)),
            "median_sd_test_cells_db": float(np.median(sd[test])),
        }
    if "M2" in models and "M0" in models:
        i2, i0 = list(models).index("M2"), list(models).index("M0")
        sig = np.std(p[:, :, i2] - p[:, :, i0], axis=1)
        q = np.quantile(sig, [0.05, 0.5, 0.95])
        out["ris_signature_db"] = {"q05": float(q[0]), "median": float(q[1]), "q95": float(q[2])}
        rr = ring_removed(p[:, :, i2], test, ring)
        out["ring_removed_m2_sd_db"] = float(np.median(np.std(rr, axis=0, ddof=1)))
    return out


def strip_private(obj: Any) -> Any:
    """Drop keys starting with ``_`` (array payloads) recursively."""
    if isinstance(obj, Mapping):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list | tuple):
        return [strip_private(v) for v in obj]
    return obj
