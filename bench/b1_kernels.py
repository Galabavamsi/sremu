"""B1: kernel micro-benchmarks and the fitted cost model (DESIGN.md sections 7.4, 12.2).

Usage::

    python -m bench.b1_kernels [--quick] [--out runs/bench] [--skip-engine]

Measured, per backend (``numba_cpu``; ``numpy_ref`` on the smaller sizes):

* **snapshot** kernel: ``ris_kernel.k1_terms`` + ``accumulate_cells`` for R rows x Q
  quadrature points, R*Q from 40 to 1.7e7 (the K1 terms of every (link, cell) pair, which
  is what ``ScenePropagation.snapshot`` spends its time on), plus the full
  ``ScenePropagation.snapshot(t)`` and ``combine`` on benchmark scenes;
* **GEMV** (configuration change): ``ris_kernel.tile_sums`` in the separable (C, gamma)
  and B forms, R rows x N cells;
* **render**: ``render(job, out)`` of a synthetic ``SegmentJob`` with L links of P paths
  each, FD filter T_FD in {0 (integer delay), 8, 16, 32} taps, S samples per call (a page),
  chord delays (moving paths);
* **noise**: ``impair.noise.noise`` for n samples.

Fits (least squares in log space and NNLS in linear space):

* snapshot time vs R*Q: ``T = a + b (R Q)^alpha`` -> exponent ``alpha`` against O(L N);
* render time per call: ``log T = c0 + bL log L + bT log T_eff + bS log S (+ bP log P)``
  with ``T_eff = max(T_FD, 1)`` -> exponents against O(L T S);
* the section 7.4 engine cost model from lockstep engine runs (``--skip-engine`` omits
  them): ``T_page = a + b (R Q) P/(f_s T_u) + c L T_FD P + d n_rx P + e n_paths P``
  (non-negative least squares on relative error).

Timings are wall-clock medians of repeated calls on this host (platform and background
load in the manifest/meta). Nothing here is compared across machines.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

import bench
from bench._common import BenchRun, time_calls

SUITE = "b1"


# =========================================================================== geometry helpers


def _panel_points(q: int, pitch: float = 0.0428) -> np.ndarray:
    """(q, 3) points on a square-ish grid in the plane x = 0, centred at the origin."""
    n_u = math.ceil(math.sqrt(q))
    iu = np.arange(q) % n_u
    iv = np.arange(q) // n_u
    n_v = int(iv.max()) + 1
    pts = np.zeros((q, 3))
    pts[:, 1] = (iu - (n_u - 1) / 2) * pitch
    pts[:, 2] = (iv - (n_v - 1) / 2) * pitch
    return pts


def _rows(r: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    tx = np.column_stack([8 + rng.random(r), rng.uniform(-4, 4, r), np.ones(r)])
    rx = np.column_stack([5 + rng.random(r), rng.uniform(-3, 3, r), -0.5 * np.ones(r)])
    return tx, rx


# =========================================================================== snapshot / GEMV


def bench_snapshot_kernel(
    quick: bool, out: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """K1 terms + per-cell accumulation for R rows x Q points (rows appended to ``out``)."""
    from sremu.physics.ris_kernel import accumulate_cells, k1_terms

    rng = np.random.default_rng(1)
    rs = (1, 4, 16, 64, 256)
    qs = (40, 240, 900, 4096, 16384, 65536)
    out = [] if out is None else out
    for r, q in itertools.product(rs, qs):
        rq = r * q
        if quick and rq > 2e5:
            continue
        tx, rx = _rows(r, rng)
        pts = _panel_points(q)
        areas = np.full(q, 0.0428**2)
        normal = np.array([1.0, 0.0, 0.0])
        parent = np.arange(q, dtype=np.int64)
        for be in ("numba_cpu", "numpy_ref"):
            if be == "numpy_ref" and rq > (2e5 if quick else 1.1e6):
                continue

            def fn(
                be: str = be,
                tx: np.ndarray = tx,
                rx: np.ndarray = rx,
                pts: np.ndarray = pts,
                areas: np.ndarray = areas,
                normal: np.ndarray = normal,
                parent: np.ndarray = parent,
                q: int = q,
            ) -> None:
                t = k1_terms(tx, rx, pts, areas, normal, 3.5e9, backend=be)  # type: ignore[arg-type]
                accumulate_cells(t, parent, q, backend=be)  # type: ignore[arg-type]

            st = time_calls(fn, min_time_s=0.05 if quick else 0.25, max_calls=2000)
            out.append(
                {
                    "kernel": "snapshot_k1",
                    "backend": be,
                    "R": r,
                    "Q": q,
                    "RQ": rq,
                    "t_median_s": st["median"],
                    "t_min_s": st["min"],
                    "n_calls": int(st["n"]),
                    "ns_per_element": st["median"] / rq * 1e9,
                }
            )
            print(f"snapshot {be:9s} R={r:4d} Q={q:6d} {st['median'] * 1e3:9.3f} ms", flush=True)
    return out


def bench_gemv(quick: bool, out: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Configuration GEMV ``tile_sums`` (separable and B forms)."""
    from sremu.physics.ris_kernel import tile_sums

    rng = np.random.default_rng(2)
    out = [] if out is None else out
    for r, n, form in itertools.product((1, 16, 64), (40, 240, 900, 4096, 16384), ("C", "B")):
        if quick and r * n > 70_000:
            continue
        k = 3
        tile = (np.arange(n) * k // n).astype(np.int32)
        state = rng.integers(0, 2, n).astype(np.int16)
        c = rng.standard_normal((r, n)) + 1j * rng.standard_normal((r, n))
        g = np.where(state == 0, 1.0 + 0j, -1.0 + 0j)
        b = rng.standard_normal((r, 2, n)) + 1j * rng.standard_normal((r, 2, n))
        for be in ("numba_cpu", "numpy_ref"):

            def fn(
                be: str = be,
                form: str = form,
                tile: np.ndarray = tile,
                k: int = k,
                c: np.ndarray = c,
                g: np.ndarray = g,
                b: np.ndarray = b,
                state: np.ndarray = state,
            ) -> None:
                if form == "C":
                    tile_sums(tile, k, c=c, g=g, backend=be)  # type: ignore[arg-type]
                else:
                    tile_sums(tile, k, b=b, state=state, backend=be)  # type: ignore[arg-type]

            st = time_calls(fn, min_time_s=0.05 if quick else 0.2, max_calls=5000)
            out.append(
                {
                    "kernel": f"gemv_{form}",
                    "backend": be,
                    "R": r,
                    "N": n,
                    "RN": r * n,
                    "t_median_s": st["median"],
                    "t_min_s": st["min"],
                    "n_calls": int(st["n"]),
                }
            )
            print(f"gemv {form} {be:9s} R={r:3d} N={n:6d} {st['median'] * 1e6:9.1f} us", flush=True)
    return out


def bench_scene_snapshot(
    quick: bool, out: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Full ``ScenePropagation.snapshot`` and ``combine`` on benchmark scenes."""
    from bench._engine_point import _resolve
    from bench._scenes import bench_scenario, ris_path_check
    from sremu.physics.propagation import from_scenario

    out = [] if out is None else out
    grid = [(1, 40), (1, 900), (1, 16384), (16, 240), (16, 4096), (128, 240), (128, 4096)]
    if quick:
        grid = grid[:4]
    for links, n in grid:
        res = _resolve(bench_scenario(links=links, n_cells=n, n_ris=1, rate_sps="1e6"))
        d = res.scenario.model_dump(mode="python")
        model = from_scenario(d, backend="numba_cpu")
        cfg = {p: model.initial_config(p) for p in model.panel_ids()}
        chk = ris_path_check(model, cfg)  # raises on a zero RIS term (section 12.2)
        ts = iter(Fraction(i, 1000) for i in itertools.count())
        st = time_calls(
            lambda model=model, ts=ts: model.snapshot(next(ts)), min_time_s=0.05 if quick else 0.3
        )
        snap = model.snapshot(Fraction(0))
        sc = time_calls(
            lambda model=model, snap=snap, cfg=cfg: model.combine(snap, cfg),
            min_time_s=0.05 if quick else 0.3,
        )
        q = int(model.quadrature("ris0").points.shape[0])
        for kern, s in (("scene_snapshot", st), ("scene_combine", sc)):
            out.append(
                {
                    "kernel": kern,
                    "backend": "numba_cpu",
                    "L": links,
                    "N": n,
                    "Q": q,
                    "RQ": links * q,
                    "t_median_s": s["median"],
                    "t_min_s": s["min"],
                    "n_calls": int(s["n"]),
                    "min_ris_minus_los_db": chk["min_ris_minus_los_db"],
                }
            )
        print(
            f"scene L={links:3d} N={n:6d} snapshot {st['median'] * 1e3:8.3f} ms "
            f"combine {sc['median'] * 1e3:8.3f} ms",
            flush=True,
        )
    return out


# =========================================================================== render / noise


def _make_job(links: int, paths: int, taps: int, s: int, fs: Fraction, groups: str) -> Any:
    from sremu.channel.fd import kaiser_fd_spec
    from sremu.channel.ring import TxRingBuffer
    from sremu.contracts import FDSpec, LinkJob, SegmentJob

    rng = np.random.default_rng(3)
    cap = 1 << 20
    n_tx = cap
    ring = TxRingBuffer("tx.tx0", fs, cap)
    k = np.arange(n_tx)
    ring.write(0, np.exp(2j * np.pi * 0.1 * k).astype(np.complex64))
    fd = FDSpec.integer() if taps == 0 else kaiser_fd_spec(taps)
    jobs = []
    for li in range(links):
        tau_a = 20e-9 + rng.random(paths) * 80e-9 + (taps // 2) / float(fs)
        tau_b = tau_a + rng.uniform(-1e-9, 1e-9, paths)
        amp_a = (rng.standard_normal(paths) + 1j * rng.standard_normal(paths)) * 1e-4
        amp_b = amp_a * (1 + 0.01j)
        grp = (
            np.zeros(paths, dtype=np.int16) if groups == "one" else np.arange(paths, dtype=np.int16)
        )
        jobs.append(
            LinkJob(
                link=f"tx.tx0->rx{li}.rx0",
                ring=ring,
                fc_hz=3.5e9,
                lo_offset_hz=0.0,
                amp_a=amp_a,
                amp_b=amp_b,
                tau_a=tau_a,
                tau_b=tau_b,
                group=grp,
                fd=fd,
            )
        )
    # one long snapshot interval so the per-interval static is built once (steady state)
    t_b = Fraction(n_tx) / fs  # pages cycle mod 32: k1 <= 36*S <= 294912 < n_tx
    k0 = 4 * s

    def job(page: int) -> SegmentJob:
        a = k0 + page * s
        return SegmentJob(rx="rx", f_s=fs, k0=a, k1=a + s, t_a=Fraction(0), t_b=t_b, links=jobs)

    return job


def bench_render(quick: bool, out: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Render of synthetic segment jobs (steady state, one long snapshot interval)."""
    from sremu.channel import render_numba, render_numpy

    fs = Fraction(10**6)
    out = [] if out is None else out
    ls = (1, 4, 16, 64)
    ps = (1, 4, 16)
    ts = (0, 8, 16, 32)
    ss = (512, 8192)
    for links, paths, taps, s in itertools.product(ls, ps, ts, ss):
        if quick and (links > 16 or s > 512):
            continue
        for be, mod in (("numba_cpu", render_numba), ("numpy_ref", render_numpy)):
            if be == "numpy_ref" and (links * paths * s > 4 * 4 * 8192 or (quick and links > 4)):
                continue
            for groups in ("one", "per_path") if paths > 1 and taps > 0 else ("one",):
                if groups == "per_path" and (links > 16 or be == "numpy_ref"):
                    continue
                job = _make_job(links, paths, taps, s, fs, groups)
                buf = np.zeros(s, dtype=np.complex128)
                pages = itertools.count()

                def fn(
                    mod: Any = mod, job: Any = job, buf: np.ndarray = buf, pages: Any = pages
                ) -> None:
                    buf[:] = 0
                    mod.render(job(next(pages) % 32), buf)

                st = time_calls(fn, min_time_s=0.04 if quick else 0.2, max_calls=3000, warmup=2)
                n_groups = 1 if groups == "one" else paths
                out.append(
                    {
                        "kernel": "render",
                        "backend": be,
                        "L": links,
                        "P": paths,
                        "T_FD": taps,
                        "S": s,
                        "groups": groups,
                        "n_groups_per_link": n_groups,
                        "t_median_s": st["median"],
                        "t_min_s": st["min"],
                        "n_calls": int(st["n"]),
                        "ns_per_sample": st["median"] / s * 1e9,
                        "ns_per_link_path_sample": st["median"] / (s * links * paths) * 1e9,
                    }
                )
                print(
                    f"render {be:9s} L={links:3d} P={paths:3d} T={taps:2d} S={s:5d} "
                    f"{groups:8s} {st['median'] * 1e6:10.1f} us",
                    flush=True,
                )
    return out


def bench_noise(quick: bool, out: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Philox AWGN generation."""
    from sremu.impair.noise import noise

    out = [] if out is None else out
    for n in (512, 8192, 65536) if quick else (512, 2048, 8192, 65536, 262144):
        for be in ("numba_cpu", "numpy_ref"):
            k0 = itertools.count(0, n)
            st = time_calls(
                lambda be=be, n=n, k0=k0: noise((1, 2), next(k0), n, 1e-6, backend=be),
                min_time_s=0.05 if quick else 0.2,
            )
            out.append(
                {
                    "kernel": "noise",
                    "backend": be,
                    "n": n,
                    "t_median_s": st["median"],
                    "t_min_s": st["min"],
                    "n_calls": int(st["n"]),
                    "ns_per_sample": st["median"] / n * 1e9,
                }
            )
            print(f"noise {be:9s} n={n:7d} {st['median'] * 1e6:9.1f} us", flush=True)
    return out


# =========================================================================== fits


def fit_snapshot(rows: list[dict[str, Any]], tkey: str = "t_median_s") -> dict[str, Any]:
    """Per backend: log-log slope of T vs R*Q (R*Q >= 1e4) and NNLS ``T = a + b R Q``."""
    from scipy.optimize import nnls

    out: dict[str, Any] = {}
    for be in ("numba_cpu", "numpy_ref"):
        pts = [r for r in rows if r["kernel"] == "snapshot_k1" and r["backend"] == be]
        if len(pts) < 3:
            continue
        x = np.array([r["RQ"] for r in pts], dtype=float)
        y = np.array([r[tkey] for r in pts])
        big = x >= 1e4
        slope, _icpt = (
            np.polyfit(np.log(x[big]), np.log(y[big]), 1) if big.sum() >= 2 else (np.nan, np.nan)
        )
        a_mat = np.column_stack([np.ones_like(x), x]) / y[:, None]
        coef, _ = nnls(a_mat, np.ones_like(y))
        pred = coef[0] + coef[1] * x
        out[be] = {
            "exponent_alpha_vs_RQ": float(slope),
            "loglog_fit_points": int(big.sum()),
            "a_s": float(coef[0]),
            "b_s_per_element": float(coef[1]),
            "b_ns_per_element": float(coef[1] * 1e9),
            "rel_rms_linear_fit": float(np.sqrt(np.mean((pred / y - 1) ** 2))),
            "RQ_range": [float(x.min()), float(x.max())],
        }
    return out


def fit_render(rows: list[dict[str, Any]], tkey: str = "t_median_s") -> dict[str, Any]:
    """Per backend: ``log T = c0 + bL log L + bP log P + bT log T_eff + bS log S``."""
    out: dict[str, Any] = {}
    for be in ("numba_cpu", "numpy_ref"):
        pts = [
            r
            for r in rows
            if r["kernel"] == "render" and r["backend"] == be and r["groups"] == "one"
        ]
        if len(pts) < 6:
            continue
        lx = np.column_stack(
            [
                np.ones(len(pts)),
                np.log([r["L"] for r in pts]),
                np.log([r["P"] for r in pts]),
                np.log([max(r["T_FD"], 1) for r in pts]),
                np.log([r["S"] for r in pts]),
            ]
        )
        ly = np.log([r[tkey] for r in pts])
        cols = [0, 1, 2, 3, 4]
        if len({r["S"] for r in pts}) < 2:
            cols = [0, 1, 2, 3]
        coef, *_ = np.linalg.lstsq(lx[:, cols], ly, rcond=None)
        resid = ly - lx[:, cols] @ coef
        names = ["c0", "exp_L", "exp_P", "exp_T_eff", "exp_S"]
        d: dict[str, Any] = {names[c]: float(v) for c, v in zip(cols, coef, strict=True)}
        d["rms_log_residual"] = float(np.sqrt(np.mean(resid**2)))
        d["n_points"] = len(pts)
        # per-(L*P*S) and per-(L*T*S) cost in the regime the engine uses most
        lts = [r for r in pts if r["T_FD"] > 0]
        if lts:
            per = [r[tkey] / (r["L"] * r["T_FD"] * r["S"]) * 1e9 for r in lts]
            d["ns_per_L_T_S_median"] = statistics.median(per)
        lps = [r[tkey] / (r["L"] * r["P"] * r["S"]) * 1e9 for r in pts]
        d["ns_per_L_P_S_median"] = statistics.median(lps)
        # one combined exponent against O(L T S) (P fixed at its largest value)
        pmax = max(r["P"] for r in pts)
        sub = [r for r in pts if r["P"] == pmax]
        x = np.log([r["L"] * max(r["T_FD"], 1) * r["S"] for r in sub])
        if len(set(x.tolist())) > 1:
            d["exp_LTS_combined"] = float(np.polyfit(x, np.log([r[tkey] for r in sub]), 1)[0])
            d["exp_LTS_combined_P"] = pmax
        # the same regression on the FD-filter points only (T_FD > 0): the integer-delay
        # path (T_FD = 0, counted as T_eff = 1 above) has its own cost structure
        fd = [r for r in pts if r["T_FD"] > 0]
        if len(fd) >= 6 and len({r["T_FD"] for r in fd}) > 1:
            fx = np.column_stack(
                [
                    np.ones(len(fd)),
                    np.log([r["L"] for r in fd]),
                    np.log([r["P"] for r in fd]),
                    np.log([r["T_FD"] for r in fd]),
                    np.log([r["S"] for r in fd]),
                ]
            )
            fy = np.log([r[tkey] for r in fd])
            fcols = cols
            fc, *_ = np.linalg.lstsq(fx[:, fcols], fy, rcond=None)
            fres = fy - fx[:, fcols] @ fc
            d["fd_only"] = {
                **{
                    names[c].replace("T_eff", "T_FD"): float(v)
                    for c, v in zip(fcols, fc, strict=True)
                },
                "rms_log_residual": float(np.sqrt(np.mean(fres**2))),
                "n_points": len(fd),
            }
        out[be] = d
    return out


def fit_linear(
    rows: list[dict[str, Any]], kernel: str, xkey: str, tkey: str = "t_median_s"
) -> dict[str, Any]:
    """NNLS ``T = a + b x`` on relative error, per backend."""
    from scipy.optimize import nnls

    out: dict[str, Any] = {}
    for be in ("numba_cpu", "numpy_ref"):
        pts = [r for r in rows if r["kernel"] == kernel and r["backend"] == be]
        if len(pts) < 2:
            continue
        x = np.array([r[xkey] for r in pts], dtype=float)
        y = np.array([r[tkey] for r in pts])
        coef, _ = nnls(np.column_stack([np.ones_like(x), x]) / y[:, None], np.ones_like(y))
        slope = np.polyfit(np.log(x), np.log(y), 1)[0] if len(set(x)) > 1 else float("nan")
        out[be] = {
            "a_s": float(coef[0]),
            "b_s_per_unit": float(coef[1]),
            "loglog_slope": float(slope),
            "x": xkey,
        }
    return out


# =========================================================================== engine cost model


def engine_cost_points(
    quick: bool, out: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Lockstep engine runs spanning the section 7.4 regressors."""
    from bench._engine_point import measure_lockstep
    from bench._scenes import bench_scenario

    grid = [
        # (L, N, n_ris, f_s, features)
        (1, 240, 1, "1e6", "F1"),
        (4, 240, 1, "1e6", "F1"),
        (16, 240, 1, "1e6", "F1"),
        (64, 240, 1, "1e6", "F1"),
        (1, 16384, 1, "1e6", "F1"),
        (16, 4096, 1, "1e6", "F1"),
        (4, 240, 4, "1e6", "F1"),
        (16, 240, 0, "1e6", "F1"),
        (1, 240, 1, "1e5", "F1"),
        (4, 240, 1, "5.76e6", "F1"),
        (1, 240, 1, "23.04e6", "F1"),
        (4, 240, 1, "23.04e6", "F1"),
        (1, 240, 1, "1e6", "F3"),
        (4, 900, 1, "1e6", "F3"),
        (16, 240, 1, "1e6", "F3"),
        (1, 4096, 1, "1e6", "F3"),
        (4, 240, 2, "1e6", "F3"),
        (8, 900, 1, "5.76e6", "F1"),
        (32, 40, 1, "1e6", "F1"),
        (2, 900, 2, "1e6", "F1"),
    ]
    if quick:
        grid = [*grid[:4], grid[12]]
    out = [] if out is None else out
    for links, n, n_ris, fs, feat in grid:
        sc = bench_scenario(links=links, n_cells=n, n_ris=n_ris, rate_sps=fs, features=feat)
        r = measure_lockstep(
            sc,
            features=feat,
            reps=1 if quick else 3,
            target_pages=2000,
            rep_wall_target_s=1.0 if quick else 4.0,
            point_wall_cap_s=60.0 if quick else 120.0,
        )
        f = r["facts"]
        dec = r["decisions"]
        page = int(f["page"])
        rate = float(Fraction(fs))
        t_page = (page / rate) / r["rtf_median"] if r["rtf_median"] else None
        t_u = f["t_u_s"]
        taps = int(f["fd_taps"])
        groups = float(dec.get("mean_fd_groups") or 1.0)
        n_paths = f["n_links"] * (1 + f["ris_tile_paths"] / max(1, f["n_links"]))
        row = {
            "L": links,
            "N": n,
            "n_ris": n_ris,
            "f_s": fs,
            "features": feat,
            "page": page,
            "t_u_s": t_u,
            "fd_taps": taps,
            "fd_groups_mean": groups,
            "narrowband_fraction": dec.get("narrowband_fraction"),
            "RQ": f["rq"],
            "n_rx": f["n_rx"],
            "n_paths_total": n_paths,
            "x_snap": 0.0 if t_u is None else f["rq"] * page / (rate * t_u),
            "x_fd": links * taps * groups * page,
            "x_rx": f["n_rx"] * page,
            "x_paths": n_paths * page,
            "rtf_median": r["rtf_median"],
            "rtf_reps": r["rtf_reps"],
            "rep_pages": r["rep_pages"],
            "reps": r["reps"],
            "T_page_s": t_page,
            "T_page_best_s": (page / rate) / max(r["rtf_reps"]) if r["rtf_reps"] else None,
            "scenario_sha256": r["scenario_sha256"],
        }
        out.append(row)
        print(
            f"engine L={links:3d} N={n:6d} R={n_ris} fs={fs:8s} {feat} rtf={r['rtf_median']:.3f} "
            f"T_page={t_page * 1e6 if t_page else float('nan'):.1f} us",
            flush=True,
        )
    return out


def fit_engine(rows: list[dict[str, Any]], ykey: str = "T_page_s") -> dict[str, Any]:
    """NNLS of the section 7.4 model on relative error."""
    from scipy.optimize import nnls

    pts = [r for r in rows if r.get(ykey)]
    if len(pts) < 6:
        return {}
    keys = ("x_snap", "x_fd", "x_rx", "x_paths")
    x = np.array([[1.0] + [float(r[k]) for k in keys] for r in pts])
    y = np.array([r[ykey] for r in pts])
    coef, _ = nnls(x / y[:, None], np.ones_like(y))
    pred = x @ coef
    rel = pred / y - 1
    for r, p in zip(pts, pred, strict=True):
        sfx = "" if ykey == "T_page_s" else "_" + ykey
        r["T_page_pred_s" + sfx] = float(p)
        r["rel_err" + sfx] = float(p / r[ykey] - 1)
    return {
        "model": "T_page = a + b*(R*Q)*P/(f_s*T_u) + c*L*T_FD*G*P + d*n_rx*P + e*n_paths*P",
        "note": (
            "G = mean FD groups per link; T_FD = taps (1 for integer delay); n_paths = LOS + "
            "RIS tile paths over all links; static scenes have no snapshot term"
        ),
        "a_s": float(coef[0]),
        "b_ns_per_row_point": float(coef[1] * 1e9),
        "c_ns_per_link_tap_sample": float(coef[2] * 1e9),
        "d_ns_per_rx_sample": float(coef[3] * 1e9),
        "e_ns_per_path_sample": float(coef[4] * 1e9),
        "rel_rms": float(np.sqrt(np.mean(rel**2))),
        "rel_max_abs": float(np.max(np.abs(rel))),
        "n_points": len(pts),
    }


def fit_engine_extended(rows: list[dict[str, Any]], ykey: str = "T_page_s") -> dict[str, Any]:
    """Diagnostic companion of :func:`fit_engine` (the section 7.4 form stays primary): the
    same NNLS with three more terms, a per-link cost per page (``f L``), a cost per snapshot
    (``g P/(f_s T_u)``) and a per-link cost per snapshot (``h L P/(f_s T_u)``), i.e. the
    per-call dispatch work the section 7.4 form has no term for. Eight coefficients on few
    points: it shows which costs the primary form misses, it is not a predictive model."""
    from scipy.optimize import nnls

    pts = [r for r in rows if r.get(ykey)]
    if len(pts) < 10:
        return {}

    def nsnap(r: dict[str, Any]) -> float:
        t_u = r.get("t_u_s")
        if not t_u:
            return 0.0
        return float(r["page"]) / (float(Fraction(str(r["f_s"]))) * float(t_u))

    x = np.array(
        [
            [1.0]
            + [float(r[k]) for k in ("x_snap", "x_fd", "x_rx", "x_paths")]
            + [float(r["L"]), nsnap(r), float(r["L"]) * nsnap(r)]
            for r in pts
        ]
    )
    y = np.array([r[ykey] for r in pts])
    coef, _ = nnls(x / y[:, None], np.ones_like(y))
    rel = (x @ coef) / y - 1
    return {
        "model": (
            "section 7.4 form + f*L + g*P/(f_s*T_u) + h*L*P/(f_s*T_u) (diagnostic; the "
            "section 7.4 fit is the reported one)"
        ),
        "a_s": float(coef[0]),
        "b_ns_per_row_point": float(coef[1] * 1e9),
        "c_ns_per_link_tap_sample": float(coef[2] * 1e9),
        "d_ns_per_rx_sample": float(coef[3] * 1e9),
        "e_ns_per_path_sample": float(coef[4] * 1e9),
        "f_us_per_link_page": float(coef[5] * 1e6),
        "g_us_per_snapshot": float(coef[6] * 1e6),
        "h_us_per_link_snapshot": float(coef[7] * 1e6),
        "rel_rms": float(np.sqrt(np.mean(rel**2))),
        "rel_max_abs": float(np.max(np.abs(rel))),
        "n_points": len(pts),
        "n_coefficients": 8,
    }


# =========================================================================== main


def main(argv: list[str] | None = None) -> int:
    """Run B1 and write its run directory."""
    bench.deterministic_env()
    ap = argparse.ArgumentParser(prog="bench.b1_kernels", description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="small sizes only (smoke test)")
    ap.add_argument("--out", type=Path, default=None, help="output root (default runs/bench)")
    ap.add_argument("--skip-engine", action="store_true", help="skip the engine cost-model runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wait-quiet-min", type=float, default=0.0, help="wait for an idle host first")
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: all)")
    ap.add_argument(
        "--section-quiet-wait-s",
        type=float,
        default=0.0,
        help="before each kernel section wait up to this long for a quiet host",
    )
    ap.add_argument("--section-quiet-busy", type=float, default=0.30)
    a = ap.parse_args(argv)
    bench.set_threads(a.threads)
    import numba

    spec = {
        "section_quiet": {"wait_s": a.section_quiet_wait_s, "max_busy": a.section_quiet_busy},
        "quick": a.quick,
        "skip_engine": a.skip_engine,
        "numba_threads": numba.get_num_threads(),
        "kernels": ["snapshot_k1", "gemv_C", "gemv_B", "scene_snapshot", "render", "noise"],
    }
    run = BenchRun(
        SUITE,
        spec,
        out_root=a.out,
        tag=a.tag or ("quick" if a.quick else None),
        quiet_s=1.0 if a.quick else 3.0,
        caveats=[
            "Wall-clock medians on one host; not comparable across machines.",
            "Synthetic render jobs: one long snapshot interval (steady state, statics cached).",
            "numpy_ref is timed only on the smaller sizes.",
        ],
        wait_quiet_s=a.wait_quiet_min * 60.0,
    )
    t0 = time.perf_counter()
    snap: list[dict[str, Any]] = []
    gemv: list[dict[str, Any]] = []
    scene: list[dict[str, Any]] = []
    rend: list[dict[str, Any]] = []
    noi: list[dict[str, Any]] = []
    eng_rows: list[dict[str, Any]] = []
    sections: dict[str, Any] = {}

    def section(name: str, fn: Any, out: list[dict[str, Any]]) -> None:
        gate = run.load.wait_quiet(
            a.section_quiet_busy, 2.0, a.section_quiet_wait_s, tag=f"gate-{name}"
        )
        i0 = run.load.mark()
        ts = time.perf_counter()
        fn(a.quick, out)
        sections[name] = {
            "gate": gate,
            "background_busy": run.load.window_stats(i0),
            "wall_s": time.perf_counter() - ts,
            "rows": len(out),
        }

    with run.interruptible():
        section("snapshot_k1", bench_snapshot_kernel, snap)
        section("gemv", bench_gemv, gemv)
        section("scene_snapshot", bench_scene_snapshot, scene)
        section("render", bench_render, rend)
        section("noise", bench_noise, noi)
        if not a.skip_engine:
            section("engine_cost_points", engine_cost_points, eng_rows)
    run.write_json("sections.json", sections)
    run.write_csv("kernels_snapshot.csv", snap + scene)
    run.write_csv("kernels_gemv.csv", gemv)
    run.write_csv("kernels_render.csv", rend)
    run.write_csv("kernels_noise.csv", noi)
    fits: dict[str, Any] = {
        "snapshot": fit_snapshot(snap),
        "render": fit_render(rend),
        "gemv_C": fit_linear([r for r in gemv if r["kernel"] == "gemv_C"], "gemv_C", "RN"),
        "gemv_B": fit_linear([r for r in gemv if r["kernel"] == "gemv_B"], "gemv_B", "RN"),
        "noise": fit_linear(noi, "noise", "n"),
    }
    fits["min_based"] = {
        "note": (
            "same fits on per-call minima (engine: best repetition): a lower bound that "
            "intermittent background load cannot inflate; the median-based fits are primary"
        ),
        "snapshot": fit_snapshot(snap, "t_min_s"),
        "render": fit_render(rend, "t_min_s"),
        "noise": fit_linear(noi, "noise", "n", "t_min_s"),
    }
    if not a.skip_engine:
        fits["engine_cost_model"] = fit_engine(eng_rows)
        fits["engine_cost_model_extended"] = fit_engine_extended(eng_rows)
        fits["min_based"]["engine_cost_model"] = fit_engine(eng_rows, "T_page_best_s")
        run.write_csv("engine_cost_points.csv", eng_rows)
    run.write_json("fits.json", fits)
    summary = {
        "suite": SUITE,
        "wall_s": time.perf_counter() - t0,
        "numba_threads": numba.get_num_threads(),
        "snapshot_exponent_alpha": {
            k: v["exponent_alpha_vs_RQ"] for k, v in fits["snapshot"].items()
        },
        "render_exponents": {
            k: {e: v.get(e) for e in ("exp_L", "exp_P", "exp_T_eff", "exp_S")}
            for k, v in fits["render"].items()
        },
        "render_exponents_fd_only": {k: v.get("fd_only") for k, v in fits["render"].items()},
        "render_exp_LTS_combined": {
            k: v.get("exp_LTS_combined") for k, v in fits["render"].items()
        },
        "engine_cost_model": fits.get("engine_cost_model"),
        "engine_cost_model_best_rep": fits["min_based"].get("engine_cost_model"),
        "engine_cost_model_extended": fits.get("engine_cost_model_extended"),
        "snapshot_exponent_alpha_min_based": {
            k: v["exponent_alpha_vs_RQ"] for k, v in fits["min_based"]["snapshot"].items()
        },
        "background_busy_by_section": {k: v["background_busy"] for k, v in sections.items()},
    }
    run.finish(summary)
    print(f"B1 written to {run.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
