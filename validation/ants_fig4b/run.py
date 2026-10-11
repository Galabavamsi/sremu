"""The unblinded V-LAB run: primary, secondary and sensitivity analyses; F4 data; Table V.

Usage (only after the lead has locked the protocol, DESIGN.md section 10.9)::

    python -m validation.ants_fig4b.run --tensor runs/vlab/<...>_vlab_tensor/pred_tensor.npz \
        --tensor-sor runs/vlab/<...>_vlab_tensor_S-OR/pred_tensor.npz

1. :func:`validation.ants_fig4b.protocol_lock.check_lock` (refuses while
   ``PREREGISTERED_SHA256`` is empty or the protocol files changed); then a clean sremu work
   tree (the lock binds the protocol files only, the analysis code is pinned by the commit
   in the manifest; ``--allow-dirty`` overrides this and is recorded); then
   :func:`~validation.ants_fig4b.protocol_lock.load_measured`.
2. The tensor must carry the protocol's prediction-spec hash and, if the locked protocol
   records a power-analysis result, the same tensor content hash (``--allow-tensor-mismatch``
   overrides this and records it). The unblinded run also needs the S-OR tensor
   (``--tensor-sor``, the alternative prior of sensitivity run S-OR; refused before any data
   access when missing); it must carry the S-OR prediction-spec hash and, if
   ``protocol.json`` records one, the recorded content hash.
3. Primary analysis (paired bootstrap, ``bootstrap.n_resamples_primary``) with decisions
   C1-C3; secondary "optimistic" analysis; leave-one-ring-out and leave-one-wedge-out;
   sensitivity runs S1 (no (35 deg, 1.5 m) cell), S2 (no colorbar-pinned cells), S3 (coverage
   at sigma_rep 0.5 / 2 dB; decoy rates from the locked protocol), S4 (breakdown by
   hypothesis), S-OR (decisions under the v1 O_R geometry rows, own tensor) and S-NF
   (decisions with the offsets fitted without the near-field ring r = 1.8 m, plus the
   point-vs-patch spread at that ring); uncertainty budget.
4. Writes ``F4.csv`` (per cell: measured DIG, prior-median prediction SIM-sremu, residual),
   ``F4_delta_rmse.csv`` (per draw), ``F4.meta.json``, ``table_v.csv``/``table_v.json``,
   ``results.json`` and ``report.md`` into a new run directory with a manifest.

``--dry-run-decoy N`` runs the identical pipeline on ``synthetic_decoy(N)`` instead of the
measured map (no lock needed; every output is labelled DECOY and SIM-sremu). This is how the
pipeline is exercised before unblinding.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")

import numpy as np
from validation.ants_fig4b import analysis as an
from validation.ants_fig4b import protocol_lock
from validation.ants_fig4b.predict import TensorCache, load_cache, point_vs_patch_spread
from validation.ants_fig4b.protocol import (
    cell_layout,
    load_protocol,
    ring_split,
    sensitivity_run,
    split_masks,
    variant_protocol,
)

from sremu import rng as sremu_rng
from sremu.record.manifest import git_state

__all__ = ["analyze", "lock_state", "main", "table_v_rows"]


def _boot_idx(protocol: Mapping[str, Any], analysis: str, n_test: int, n_boot: int) -> np.ndarray:
    st = protocol["rng"]["analysis_bootstrap"]
    g = sremu_rng.generator(
        int(protocol["seed"]),
        str(st["purpose"]),
        str(st["name"]).format(analysis=analysis),
        counter=int(st["counter"]),
    )
    return g.integers(0, n_test, size=(n_boot, n_test))


def analyze(
    tensor: TensorCache,
    data: an.Dataset,
    protocol: Mapping[str, Any],
    *,
    n_boot: int | None = None,
    tensor_sor: TensorCache | None = None,
) -> dict[str, Any]:
    """Every pre-registered analysis of one dataset (arrays under ``_`` keys). ``tensor_sor``
    is the tensor of the alternative prior of S-OR (``sensitivity.S-OR`` is ``{"run": False}``
    without it; the unblinded CLI requires it)."""
    nb = int(protocol["bootstrap"]["n_resamples_primary"] if n_boot is None else n_boot)
    lay = cell_layout(protocol)
    train, test = split_masks(protocol, lay)["primary"]
    out: dict[str, Any] = {"dataset": {"label": data.label, "data_class": data.data_class}}
    prim = an.run_split(
        tensor,
        data,
        protocol,
        train=train,
        test=test,
        boot_idx=_boot_idx(protocol, "primary", int(test.sum()), nb),
        layout=lay,
    )
    out["primary"] = prim
    out["secondary"] = an.secondary(
        prim,
        data,
        train=train,
        test=test,
        sigma_tot_db=float(protocol["secondary_analysis"]["sigma_tot_db"]),
    )
    out["folds"] = an.run_folds(tensor, data, protocol)
    sens: dict[str, Any] = {}
    s1 = sensitivity_run(protocol, "S1")
    drop1 = [lay.index(float(t), float(r)) for t, r in s1["drop_cells_theta_r"]]
    pin = an.pinned_cells(data.y_dbm, protocol)
    for sid, drop in (("S1", drop1), ("S2", pin["cells"])):
        keep = np.ones(lay.n, dtype=bool)
        keep[drop] = False
        tr, te = train & keep, test & keep
        r = an.run_split(
            tensor,
            data,
            protocol,
            train=tr,
            test=te,
            boot_idx=_boot_idx(protocol, sid, int(te.sum()), nb),
            layout=lay,
        )
        sens[sid] = {"dropped_cells": [int(i) for i in drop], "result": r}
    sens["S2"]["pinned"] = pin
    snf = sensitivity_run(protocol, "S-NF")
    tr_nf, te_nf = ring_split(
        protocol, snf["split"]["train_rings_m"], snf["split"]["test_rings_m"], lay
    )
    sens["S-NF"] = {
        "train_rings_m": snf["split"]["train_rings_m"],
        "test_rings_m": snf["split"]["test_rings_m"],
        "result": an.run_split(
            tensor,
            data,
            protocol,
            train=tr_nf,
            test=te_nf,
            boot_idx=_boot_idx(protocol, "S-NF", int(te_nf.sum()), nb),
            layout=lay,
        ),
        "point_vs_patch": point_vs_patch_spread(tensor.arrays, tensor.meta["codes"], protocol),
    }
    if tensor_sor is not None:
        sens["S-OR"] = {
            "run": True,
            "tensor_content_sha256": tensor_sor.meta.get("content_sha256"),
            "result": an.run_split(
                tensor_sor,
                data,
                variant_protocol(protocol, "S-OR"),
                train=train,
                test=test,
                boot_idx=_boot_idx(protocol, "S-OR", int(test.sum()), nb),
                layout=lay,
            ),
        }
    else:
        sens["S-OR"] = {"run": False, "reason": "no S-OR tensor given"}
    out["sensitivity"] = sens
    out["budget"] = an.uncertainty_budget(tensor, data, prim, protocol, test=test)
    out["breakdown"] = an.breakdown(tensor, prim)
    pa_res = protocol.get("power_analysis_result")
    out["power_analysis"] = pa_res
    out["naming"] = protocol["naming"]["label"]
    return out


def table_v_rows(res: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Table V: per model, test RMSE / bias / MAE / max |e| (median [5-95 %]) and the
    secondary weighted RMSE."""
    prim, sec = res["primary"], res["secondary"]
    rows = []
    for m in an.ALL_MODELS:
        mm = prim["metrics"][m]
        rows.append(
            {
                "model": m,
                "rmse_median_db": mm["rmse"]["median"],
                "rmse_q05_db": mm["rmse"]["q05"],
                "rmse_q95_db": mm["rmse"]["q95"],
                "bias_median_db": mm["bias"]["median"],
                "bias_q05_db": mm["bias"]["q05"],
                "bias_q95_db": mm["bias"]["q95"],
                "mae_median_db": mm["mae"]["median"],
                "max_abs_median_db": mm["max_abs"]["median"],
                "secondary_rmse_db": sec["models"][m]["rmse"],
                "secondary_ess": sec["models"][m]["ess"],
            }
        )
    return rows


def _f4_rows(
    res: Mapping[str, Any], data: an.Dataset, protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    lay = cell_layout(protocol)
    train, _test = split_masks(protocol, lay)["primary"]
    prim = res["primary"]
    pm2 = np.median(prim["_yhat"]["M2"], axis=0)
    pm0 = np.median(prim["_yhat"]["M0"], axis=0)
    rows = []
    for i in range(lay.n):
        rows.append(
            {
                "cell": i,
                "theta_deg": float(lay.theta_deg[i]),
                "r_m": float(lay.r_m[i]),
                "split": "train" if train[i] else "test",
                "measured_dbm": float(data.y_dbm[i]),
                "sigma_dig_db": float(data.sigma_dig_db[i]),
                "pred_m2_prior_median_dbm": float(pm2[i]),
                "residual_m2_db": float(data.y_dbm[i] - pm2[i]),
                "pred_m0_prior_median_dbm": float(pm0[i]),
                "residual_m0_db": float(data.y_dbm[i] - pm0[i]),
                "sigma_geom_m2_db": float(res["budget"]["_sigma_geom"][i]),
            }
        )
    return rows


def _decisions_text(res: Mapping[str, Any]) -> str:
    d = res["primary"]["decisions"]
    return ", ".join(f"{c} {'holds' if d[c]['holds'] else 'fails'}" for c in ("C1", "C2", "C3"))


def lock_state() -> dict[str, Any]:
    """The protocol lock state recorded with every run (``locked`` is True only when
    ``PREREGISTERED_SHA256`` is set and equals the digest of the files on disk)."""
    digest = protocol_lock.protocol_digest()
    pre = str(protocol_lock.PREREGISTERED_SHA256)
    return {
        "locked": bool(pre) and pre == digest,
        "preregistered_sha256": pre or None,
        "protocol_digest": digest,
    }


def _report_md(res: Mapping[str, Any], dry: bool) -> str:
    prim = res["primary"]
    d = prim["decisions"]
    pa = res.get("power_analysis") or {}
    locked = bool((res.get("lock") or {}).get("locked"))
    lines = [
        "# V-LAB run report" + (" (DECOY DRY RUN: not the measured data)" if dry else ""),
        "",
        f"Dataset: {res['dataset']['label']} ({res['dataset']['data_class']}). "
        f"Study label: {res['naming']}.",
        "",
        f"Decisions: {_decisions_text(res)}.",
        "",
        "| Model | test RMSE median (5-95 %) dB | bias median dB |",
        "|---|---|---|",
    ]
    for m in an.ALL_MODELS:
        r = prim["metrics"][m]["rmse"]
        b = prim["metrics"][m]["bias"]
        lines.append(
            f"| {m} | {r['median']:.2f} ({r['q05']:.2f}-{r['q95']:.2f}) | {b['median']:.2f} |"
        )
    lines += [
        "",
        f"C2: median Delta RMSE {d['C2']['median_delta_rmse_db']:.2f} dB, CI excludes 0 in "
        f"{100 * d['C2']['frac_ci_excludes_zero']:.1f} % of draws, "
        f"Spearman rho {d['C2']['spearman_rho']:.3f} (CI low {d['C2']['spearman_ci_low']:.3f}).",
        f"C3: median Delta RMSE {d['C3']['median_delta_rmse_db']:.2f} dB, CI excludes 0 in "
        f"{100 * d['C3']['frac_ci_excludes_zero']:.1f} % of draws.",
        "Power analysis (protocol.json power_analysis_result; protocol "
        + ("locked" if locked else "NOT locked")
        + f"): C2 detection rate {pa.get('c2_detection_rate')}, "
        f"underpowered: {pa.get('underpowered')}.",
        "",
        "Secondary analysis (optimistic): weighted Delta RMSE M0-M2 "
        f"{res['secondary']['delta_rmse']['M0-M2']:.2f} dB.",
    ]
    for sid in ("S1", "S2", "S-NF", "S-OR"):
        sr = res["sensitivity"].get(sid) or {}
        if "result" not in sr:
            lines.append(f"Sensitivity {sid}: not run ({sr.get('reason', 'missing')}).")
            continue
        sd = sr["result"]["decisions"]
        lines.append(
            f"Sensitivity {sid}: "
            + ", ".join(f"{c} {'holds' if sd[c]['holds'] else 'fails'}" for c in ("C1", "C2", "C3"))
        )
    pvp = (res["sensitivity"].get("S-NF") or {}).get("point_vs_patch") or {}
    ring = (pvp.get("rings_m") or {}).get("1.8")
    if ring:
        lines.append(
            "Point-vs-patch spread at the 1.8 m ring (M2, offset removed): median "
            f"{ring['median_abs_centred_db']:.2f} dB, 95th percentile "
            f"{ring['q95_abs_centred_db']:.2f} dB ({pvp['n_draws']} paired draws)."
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """The unblinded run (or a decoy dry run)."""
    from validation.ants_fig4b import runlog
    from validation.ants_fig4b.decoy import synthetic_decoy

    ap = argparse.ArgumentParser(prog="python -m validation.ants_fig4b.run")
    ap.add_argument("--tensor", type=Path, required=True)
    ap.add_argument(
        "--tensor-sor",
        type=Path,
        default=None,
        help="tensor of the S-OR alternative prior (predict.py --variant S-OR); required "
        "for the unblinded run",
    )
    ap.add_argument("--out", type=Path, default=Path("runs") / "vlab")
    ap.add_argument(
        "--data", type=Path, default=None, help="measured CSV (default: $SREMU_PRIVATE_DATA)"
    )
    ap.add_argument("--dry-run-decoy", type=int, default=None, metavar="N")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--allow-tensor-mismatch", action="store_true")
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help="unblind although the sremu work tree is dirty (recorded in results and meta)",
    )
    args = ap.parse_args(argv)
    protocol = load_protocol()
    dry = args.dry_run_decoy is not None
    dirty_allowed = False
    if dry:
        dec = synthetic_decoy(int(args.dry_run_decoy), protocol=protocol)
        data = dec.dataset
        source: dict[str, Any] = {
            "decoy": int(args.dry_run_decoy),
            "truth": dec.truth_draw.as_record() if dec.truth_draw else None,
        }
    else:
        try:
            digest = protocol_lock.check_lock()  # refuses before any data access
        except protocol_lock.ProtocolLockError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        # The lock binds PROTOCOL.md and protocol.json only; the analysis code is pinned by
        # the commit recorded in the manifest, so unblinding needs a clean work tree.
        gs = git_state()
        if gs.dirty is not False:
            if not args.allow_dirty:
                print(
                    "refused: the sremu work tree is dirty or not a git work tree "
                    f"(dirty={gs.dirty}, reason={gs.reason}); the unblinded run must use "
                    "committed code. Pass --allow-dirty to proceed (recorded).",
                    file=sys.stderr,
                )
                return 2
            dirty_allowed = True
        if args.tensor_sor is None:
            print(
                "refused: the unblinded run needs the S-OR tensor (--tensor-sor; "
                "python -m validation.ants_fig4b.predict --variant S-OR)",
                file=sys.stderr,
            )
            return 2
        try:
            md = protocol_lock.load_measured(args.data)
        except protocol_lock.ProtocolLockError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        data = an.Dataset(
            y_dbm=md.y_dbm,
            sigma_dig_db=md.sigma_dig_db,
            label="ANTS Fig. 4(b) digitized map",
            data_class="DIG",
        )
        source = {
            "path": md.source_path,
            "sha256": md.sha256,
            "protocol_digest": digest,
            "columns": dict(md.columns),
        }
    tensor = load_cache(args.tensor, protocol=protocol)
    tensor_sor = None
    sor_mismatch = False
    if args.tensor_sor is not None:
        tensor_sor = load_cache(args.tensor_sor, protocol=variant_protocol(protocol, "S-OR"))
        rec_sor = sensitivity_run(protocol, "S-OR").get("tensor_result") or {}
        want = rec_sor.get("tensor_content_sha256")
        sor_mismatch = bool(want) and want != tensor_sor.meta["content_sha256"]
        if sor_mismatch and not (args.allow_tensor_mismatch or dry):
            raise SystemExit(
                f"the S-OR tensor differs from the one recorded in protocol.json ({want} vs "
                f"{tensor_sor.meta['content_sha256']}); use --allow-tensor-mismatch (recorded)"
            )
    pa: dict[str, Any] = dict(protocol.get("power_analysis_result") or {})
    mismatch = bool(pa) and pa.get("tensor_content_sha256") != tensor.meta["content_sha256"]
    if mismatch and not (args.allow_tensor_mismatch or dry):
        raise SystemExit(
            "the tensor differs from the one used by the locked power analysis "
            f"({pa.get('tensor_content_sha256')} vs {tensor.meta['content_sha256']}); "
            "use --allow-tensor-mismatch to proceed (recorded)"
        )
    rec = runlog.RunRecorder(
        args.out, "vlab_dryrun_decoy" if dry else "vlab_unblinded", protocol, argv=sys.argv
    )
    rec.add_input(args.tensor, "SIM-sremu")
    if args.tensor_sor is not None:
        rec.add_input(args.tensor_sor, "SIM-sremu")
    if not dry:
        rec.add_input(str(source["path"]), "DIG")
    res = analyze(tensor, data, protocol, n_boot=args.n_boot, tensor_sor=tensor_sor)
    res["lock"] = lock_state()
    res["dirty_tree_allowed"] = dirty_allowed
    dc_meas = "SIM-sremu" if dry else "DIG"
    rec.write_csv("F4.csv", _f4_rows(res, data, protocol))
    rec.add_output("F4.csv", dc_meas)
    delta_rows = [
        {"draw": k, "delta_m0_m2_db": float(a), "delta_m1_m2_db": float(b)}
        for k, (a, b) in enumerate(
            zip(res["primary"]["_delta"]["M0-M2"], res["primary"]["_delta"]["M1-M2"], strict=True)
        )
    ]
    rec.write_csv("F4_delta_rmse.csv", delta_rows)
    rec.add_output("F4_delta_rmse.csv", "SIM-sremu")
    rec.write_json(
        "F4.meta.json",
        {
            "figure": "F4",
            "dry_run_decoy": dry,
            "data_classes": sorted({dc_meas, "SIM-sremu", "ASSUMED"}),
            "columns": {
                "measured_dbm": dc_meas,
                "pred_m2_prior_median_dbm": "SIM-sremu",
                "residual_m2_db": f"{dc_meas} minus SIM-sremu",
            },
            "decoy_c2_detection_rate": (pa or {}).get("c2_detection_rate"),
            "underpowered": (pa or {}).get("underpowered"),
            "naming": res["naming"],
            "inputs": rec.inputs,
            "assumptions": list(runlog.ASSUMPTIONS),
            "sremu_commit": rec.git.commit,
        },
    )
    rec.add_output("F4.meta.json", "SIM-sremu")
    tv = table_v_rows(res)
    rec.write_csv("table_v.csv", tv)
    rec.add_output("table_v.csv", dc_meas)
    rec.write_json(
        "table_v.json",
        {
            "rows": tv,
            "decisions": res["primary"]["decisions"],
            "naming": res["naming"],
            "dry_run_decoy": dry,
            "power_analysis": pa,
            "secondary_label": "optimistic",
        },
    )
    rec.add_output("table_v.json", dc_meas)
    rec.write_json(
        "results.json",
        {
            "source": source,
            "tensor_mismatch_allowed": mismatch,
            "tensor_sor_mismatch_allowed": sor_mismatch,
            **an.strip_private(res),
        },
    )
    rec.add_output("results.json", dc_meas)
    (rec.run_dir / "report.md").write_text(_report_md(res, dry), encoding="utf-8", newline="\n")
    rec.add_output("report.md", dc_meas)
    rec.finish(
        backend="numba_cpu",
        extra={
            "dry_run_decoy": dry,
            "decisions": _decisions_text(res),
            "source": source,
            "dirty_tree_allowed": dirty_allowed,
        },
    )
    print(_report_md(res, dry))
    print(f"run directory: {rec.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
