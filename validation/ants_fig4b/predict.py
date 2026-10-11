"""The V-LAB prediction tensor ``pred[K, 30, models]`` (DESIGN.md section 10.9, "Primary").

For each prior draw ``k`` (keyed, :func:`validation.ants_fig4b.prior.draw_prior`) one sremu
scene is built with :func:`sremu.physics.propagation.from_scenario` (numba backend; one RIS
panel per tile with its own pose; the draw's ``cell_model``, pitch, antennas and floor) and
evaluated at ``t = 0``. From the snapshot:

* ``h_los``: the LOS path, ``h_gnd``: the ground image path (``PathKind.GROUND``);
* per tile ``k``: ``G_k = sum_{rows (link, v)} sum_n C_k[r, n] exp(-j 2 pi f_c tau_tile[r])``
  from the separable RIS coefficients of panel ``t<k>`` (carrier phase restored exactly as
  ``ScenePropagation.evaluate``), split into the direct hop (variant 0) and the ground hop
  variants 1-3; every cell of a tile has the tile's state;
* the drawn ``Gamma_s = |Gamma_s| exp(j(arg Gamma_s^table + dphi_s))`` is applied to the tile
  states.

Models (``protocol.json`` ``models``): ``M0 = |h_los|^2``, ``M1 = |h_los + sum_k G0_k
Gamma_1|^2``, ``M2 = |h_los + sum_k G0_k Gamma[s_k]|^2``, ``M3 = |h_los + h_gnd + sum_k (G0_k +
Gimg_k) Gamma[s_k]|^2``, in dB (unit Tx power). ``dist_m[K, 30]`` (Tx-point distance) is the
``M0n`` regressor.

Near field (``models.near_field``): sremu's ``ValidityWarning`` for a node within 3 lambda of a
cell is suppressed (only that one) and counted per draw; other ``ValidityWarning`` instances
are counted separately; the minimum point-to-cell-centre and point-to-tile distances are
recorded; for the first ``models.near_field.point_vs_patch.n_draws`` draws the same geometry
is also evaluated with the other ``cell_model`` (``pred_alt_cm_db``, paired).

The cache is an ``.npz`` with a content SHA-256 over every array (name, dtype, shape, bytes)
and the SHA-256 of the protocol's prediction spec, written into a run directory with a
manifest. Draws are independent and keyed, so the tensor does not depend on the number of
worker processes or threads.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")  # deterministic mode (D-4)

import numpy as np
from validation.ants_fig4b.prior import (
    PriorDraw,
    draw_prior,
    geometry,
    node_tile_distance,
    scenario_dict,
)
from validation.ants_fig4b.protocol import (
    cell_layout,
    load_protocol,
    prediction_spec_sha256,
    split_masks,
    variant_protocol,
)

from sremu.contracts import PathKind
from sremu.errors import GeometryError, RefinementNotConverged, ValidityWarning
from sremu.physics.propagation import from_scenario
from sremu.units import C0

__all__ = [
    "MODELS",
    "NEARFIELD_WARNING",
    "DrawPrediction",
    "TensorCache",
    "alternative_cell_model",
    "compute_tensor",
    "content_sha256",
    "draw_and_evaluate",
    "evaluate_alternative",
    "evaluate_draw",
    "evaluate_stream",
    "load_cache",
    "n_point_vs_patch",
    "point_vs_patch_spread",
    "position_sigma",
    "save_cache",
]

MODELS: Final[tuple[str, ...]] = ("M0", "M1", "M2", "M3")
NEARFIELD_WARNING: Final[re.Pattern[str]] = re.compile(
    r"\(< 3 lambda\) from a cell; the point-scatterer approximation is marginal"
)
"""sremu's 3-lambda ``ValidityWarning`` (``ScenePropagation._guard_distance``): the only
warning suppressed (and counted) here (``models.near_field.warnings``)."""


@dataclass(frozen=True, eq=False)
class DrawPrediction:
    """Predictions of one draw: ``power_db[4, 30]`` (MODELS order), ``dist_m[30]`` and
    diagnostics (quadrature, Richardson and near-field bookkeeping)."""

    power_db: np.ndarray
    dist_m: np.ndarray
    n_quad: int
    richardson_level: int
    richardson_db: float
    subcell_max_wavelengths: float
    converged_subcell_m: float
    n_nearfield_warnings: int
    n_other_warnings: int
    min_point_cell_centre_m: float
    min_point_tile_m: float
    frac_points_behind_tiles: float
    ris_to_los_db_median: float
    eval_s: float


def _floor_db(p: np.ndarray, floor_db: float) -> np.ndarray:
    return np.asarray(10.0 * np.log10(np.maximum(p, 10.0 ** (floor_db / 10.0))), dtype=np.float64)


def _gamma(draw: PriorDraw, table_gamma: np.ndarray) -> np.ndarray:
    ph = np.angle(table_gamma) + np.radians(np.asarray(draw.gamma_phase_offset_deg))
    return np.asarray(draw.gamma_mag, dtype=np.float64) * np.exp(1j * ph)


def _hops(
    model: Any, snap: Any, links: Sequence[str], fc: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(gamma_table[S], G_direct[L, T], G_images[L, T])``: per link and tile (panel), the
    sum over the tile's cells of the separable coefficients with the carrier phase restored
    (a tile has one state, so its cells share the Gamma)."""
    pids = list(model.panel_ids())
    row_of = {lk: i for i, lk in enumerate(links)}
    g0 = np.zeros((len(links), len(pids)), dtype=np.complex128)
    gi = np.zeros((len(links), len(pids)), dtype=np.complex128)
    gam: np.ndarray | None = None
    for t, pid in enumerate(pids):
        rc = snap.ris[pid]
        if rc.C is None or rc.gamma is None:
            raise RuntimeError("expected separable RIS coefficients (angle-independent table)")
        if gam is None:
            gam = np.asarray(rc.gamma)
        elif not np.array_equal(gam, rc.gamma):
            raise RuntimeError("every tile must use the same state table")
        for r, (lk, var) in enumerate(rc.rows):
            ph = np.exp(-2j * np.pi * np.mod(fc * rc.tile_delay_s[r][rc.tile_of_cell], 1.0))
            term = complex(np.sum(rc.C[r] * ph))
            if var == 0:
                g0[row_of[lk], t] += term
            else:
                gi[row_of[lk], t] += term
    if gam is None:
        raise RuntimeError("the scene has no RIS panel")
    return gam, g0, gi


def _paths(snap: Any, links: Sequence[str], fc: float) -> tuple[np.ndarray, np.ndarray]:
    """``(h_los[L], h_ground[L])`` narrowband path sums (carrier phase included)."""
    los = np.zeros(len(links), dtype=np.complex128)
    gnd = np.zeros(len(links), dtype=np.complex128)
    for i, lk in enumerate(links):
        pb = snap.paths[lk]
        h = pb.amp * np.exp(-2j * np.pi * np.mod(fc * pb.delay_s, 1.0))
        los[i] = np.sum(h[pb.kind == PathKind.LOS])
        gnd[i] = np.sum(h[pb.kind == PathKind.GROUND])
    return los, gnd


def _rx_links(model: Any) -> list[str]:
    """Link ids ordered by point index ``p00`` ... ``p29``."""
    by_node = {model.link_streams(lk)[1].node: lk for lk in model.links()}
    return [by_node[f"p{i:02d}"] for i in range(len(by_node))]


def _split_warnings(caught: Sequence[warnings.WarningMessage]) -> tuple[int, int]:
    """``(near-field 3-lambda warnings, other ValidityWarnings)`` among ``caught``; warnings
    of other categories are re-emitted."""
    nf = other = 0
    for w in caught:
        if issubclass(w.category, ValidityWarning):
            if NEARFIELD_WARNING.search(str(w.message)):
                nf += 1
            else:
                other += 1
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return nf, other


def _build(
    draw: PriorDraw,
    protocol: Mapping[str, Any],
    backend: str,
    *,
    cell_model: str | None = None,
    **kw: Any,
) -> tuple[Any, float, list[warnings.WarningMessage]]:
    """``(model, subcell_max_wavelengths used, caught warnings)``: the ``po_patch`` quadrature
    escalation of ``models.po_patch_quadrature`` (sremu's default first, then finer initial
    sub-cells when sremu raises ``RefinementNotConverged``; the last failure propagates)."""
    levels = [
        float(x) for x in protocol["models"]["po_patch_quadrature"]["subcell_max_wavelengths"]
    ]
    cm = draw.cell_model if cell_model is None else cell_model
    if cm != "po_patch":
        levels = levels[:1]
    for j, sub in enumerate(levels):
        sc = scenario_dict(
            draw, protocol, backend=backend, subcell_max_wavelengths=sub, cell_model=cm, **kw
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                model = from_scenario(sc)
            except RefinementNotConverged:
                if j == len(levels) - 1:
                    raise
                continue
        return model, sub, list(caught)
    raise AssertionError("unreachable")


def evaluate_draw(
    draw: PriorDraw,
    protocol: Mapping[str, Any],
    *,
    backend: str = "numba_cpu",
    cell_model: str | None = None,
) -> DrawPrediction:
    """Model powers (dB) of ``draw`` at the 30 points: one sremu scene with the floor (M3);
    M0-M2 use its LOS path and direct-hop RIS rows only. ``cell_model`` overrides the draw's
    (paired point-vs-patch evaluation). Raises ``RefinementNotConverged`` when the
    ``po_patch`` quadrature escalation fails."""
    t0 = time.perf_counter()
    floor_db = float(protocol["models"]["power_floor_db"])
    model, sub, caught = _build(draw, protocol, backend, cell_model=cell_model, ground=True)
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        snap = model.snapshot(0)
    n_nf, n_other = _split_warnings([*caught, *caught2])
    fc = float(protocol["prior"]["fc_hz"])
    links = _rx_links(model)
    los, gnd = _paths(snap, links, fc)
    gtab, g0, gi = _hops(model, snap, links, fc)
    geo = geometry(draw, protocol)
    gam = _gamma(draw, gtab)
    s = geo.tile_state.astype(np.int64)
    h_ris_m1 = g0 @ np.full(s.size, gam[0])
    h_ris_m2 = g0 @ gam[s]
    h_ris_m3 = (g0 + gi) @ gam[s]
    p = np.stack(
        [
            np.abs(los) ** 2,
            np.abs(los + h_ris_m1) ** 2,
            np.abs(los + h_ris_m2) ** 2,
            np.abs(los + gnd + h_ris_m3) ** 2,
        ]
    )
    dist = np.linalg.norm(geo.points_m - geo.tx_m[None, :], axis=1)
    side = np.einsum(
        "tpj,tj->tp", geo.points_m[None, :, :] - geo.tile_center_m[:, None, :], geo.tile_normal
    )
    cc = geo.cell_centres()
    d_cc = np.linalg.norm(geo.points_m[:, None, :] - cc[None, :, :], axis=2)
    pids = list(model.panel_ids())
    rich = [model.richardson(pid) for pid in pids]
    quads = [model.quadrature(pid) for pid in pids]
    with np.errstate(divide="ignore"):
        ratio = 20.0 * np.log10(np.abs(h_ris_m2) / np.abs(los))
    rich_ok = [r for r in rich if r is not None]
    return DrawPrediction(
        power_db=_floor_db(p, floor_db),
        dist_m=dist,
        n_quad=int(sum(q.n_points for q in quads)),
        richardson_level=max((int(r.level) for r in rich_ok), default=-1),
        richardson_db=max((float(r.estimate_db) for r in rich_ok), default=math.nan),
        subcell_max_wavelengths=sub,
        converged_subcell_m=max(float(q.delta_m) for q in quads),
        n_nearfield_warnings=n_nf,
        n_other_warnings=n_other,
        min_point_cell_centre_m=float(np.min(d_cc)),
        min_point_tile_m=float(np.min(node_tile_distance(geo, geo.points_m))),
        frac_points_behind_tiles=float(np.mean(side <= 0.0)),
        ris_to_los_db_median=float(np.median(ratio)),
        eval_s=time.perf_counter() - t0,
    )


def alternative_cell_model(draw: PriorDraw) -> str:
    """The other cell model of the paired point-vs-patch evaluation."""
    return "po_patch" if draw.cell_model == "point" else "point"


def evaluate_alternative(
    draw: PriorDraw, protocol: Mapping[str, Any], *, backend: str = "numba_cpu"
) -> np.ndarray:
    """``power_db[4, 30]`` of ``draw`` with the other cell model (NaN if its ``po_patch``
    escalation does not converge; the draw itself stays accepted)."""
    try:
        res = evaluate_draw(
            draw, protocol, backend=backend, cell_model=alternative_cell_model(draw)
        )
    except RefinementNotConverged:
        return np.full((len(MODELS), int(protocol["data"]["n_cells"])), np.nan)
    return res.power_db


def position_sigma(
    draw: PriorDraw, protocol: Mapping[str, Any], *, backend: str = "numba_cpu"
) -> np.ndarray:
    """``sigma_pos[30]`` (dB) of the M2 prediction: central differences at fixed quadrature
    (the points move as external trajectories of one scene), propagated with the uniform
    standard deviations ``half_width/sqrt(3)`` (``uncertainty_budget.sigma_pos``). All NaN
    when a step brings a point within one wavelength of a cell (sremu ``GeometryError``; a
    feasible draw can lie within ``fd_step_m`` of that limit); such draws are excluded from
    the reported medians."""
    sp = protocol["uncertainty_budget"]["sigma_pos"]
    step = float(sp["fd_step_m"])
    sd_xy = float(sp["xy_half_width_m"]) / math.sqrt(3.0)
    sd_z = float(sp["z_half_width_m"]) / math.sqrt(3.0)
    fc = float(protocol["prior"]["fc_hz"])
    geo = geometry(draw, protocol)
    s = geo.tile_state.astype(np.int64)
    model, _sub, _caught = _build(draw, protocol, backend, ground=False, external_rx=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*" + NEARFIELD_WARNING.pattern, category=ValidityWarning
        )
        links = _rx_links(model)
        out = []
        t = 0
        for axis in range(3):
            pm = []
            for sign in (1.0, -1.0):
                t += 1
                for i, p in enumerate(geo.points_m):
                    q = p.copy()
                    q[axis] += sign * step
                    model.set_node_position(f"p{i:02d}", t, q)
                try:
                    snap = model.snapshot(t)
                except GeometryError:
                    return np.full(geo.points_m.shape[0], np.nan)
                los, _ = _paths(snap, links, fc)
                gtab, g0, _ = _hops(model, snap, links, fc)
                gam = _gamma(draw, gtab)
                pm.append(10.0 * np.log10(np.abs(los + g0 @ gam[s]) ** 2))
            out.append((pm[0] - pm[1]) / (2.0 * step))
    gx, gy, gz = out
    return np.asarray(np.sqrt((gx**2 + gy**2) * sd_xy**2 + gz**2 * sd_z**2), dtype=np.float64)


def draw_and_evaluate(
    protocol: Mapping[str, Any],
    *,
    purpose: str,
    name: str,
    counter: int = 0,
    backend: str = "numba_cpu",
) -> tuple[PriorDraw, DrawPrediction]:
    """A feasible prior draw and its predictions. A candidate whose ``po_patch`` quadrature
    does not converge (after the escalation) is rejected (reason ``po_patch_not_converged``)
    and redrawn from the same keyed stream, like a geometrically infeasible one."""
    box: dict[str, DrawPrediction] = {}

    def accept(cand: PriorDraw) -> str | None:
        try:
            box["pred"] = evaluate_draw(cand, protocol, backend=backend)
        except RefinementNotConverged:
            return "po_patch_not_converged"
        return None

    draw = draw_prior(protocol, purpose=purpose, name=name, counter=counter, accept=accept)
    return draw, box["pred"]


# =========================================================================== tensor


def _codes(protocol: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Value lists of the per-draw categorical labels (stored as ``<key>_idx``)."""
    pr = protocol["prior"]
    g = pr["geometry"]
    theta0 = tuple(g["theta0_direction"]["values"]) if "theta0_direction" in g else ("arc",)
    return {
        "geometry_model": (str(g["model"]),),
        "origin": (str(g["polar_origin"]),),
        "theta0": theta0,
        "pitch": tuple(pr["pitch"]["values"]),
        "cell_model": tuple(pr["cell_model"]["values"]),
        "antenna": tuple(pr["antenna"]["values"]),
        "polarization": tuple(pr["floor"]["polarization"]["values"]),
    }


def n_point_vs_patch(protocol: Mapping[str, Any]) -> int:
    """Draws evaluated with both cell models (``models.near_field.point_vs_patch.n_draws``)."""
    return int(protocol["models"]["near_field"]["point_vs_patch"]["n_draws"])


def _eval_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: evaluate draws ``task['ks']`` of stream ``task['stream']`` (picklable)."""
    protocol = task["protocol"]
    st = task["stream"]
    out: dict[str, Any] = {"k": [], "pred": [], "draw": [], "sigma_pos": {}, "alt": {}}
    for k in task["ks"]:
        name = str(st["name"]).format(k=k, i=k)
        draw, pred = draw_and_evaluate(
            protocol,
            purpose=st["purpose"],
            name=name,
            counter=int(st["counter"]),
            backend=task["backend"],
        )
        out["k"].append(k)
        out["pred"].append(pred)
        out["draw"].append(draw)
        if k < int(task.get("n_sigma_pos", 0)):
            out["sigma_pos"][k] = position_sigma(draw, protocol, backend=task["backend"])
        if k < int(task.get("n_alt", 0)):
            out["alt"][k] = evaluate_alternative(draw, protocol, backend=task["backend"])
    return out


def _init_worker(threads: int) -> None:
    import numba

    top = int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]
    numba.set_num_threads(max(1, min(threads, top)))


def evaluate_stream(
    protocol: Mapping[str, Any],
    stream: Mapping[str, Any],
    count: int,
    *,
    jobs: int = 1,
    backend: str = "numba_cpu",
    n_sigma_pos: int = 0,
    n_alt: int = 0,
    chunk: int = 8,
    progress: bool = False,
) -> dict[str, Any]:
    """Evaluate draws ``0..count-1`` of a keyed stream; arrays in index order. The first
    ``n_sigma_pos`` draws also get ``sigma_pos``, the first ``n_alt`` the paired evaluation
    with the other cell model."""
    ks = list(range(count))
    tasks = [
        {
            "protocol": dict(protocol),
            "stream": dict(stream),
            "ks": ks[i : i + chunk],
            "backend": backend,
            "n_sigma_pos": n_sigma_pos,
            "n_alt": n_alt,
        }
        for i in range(0, count, chunk)
    ]
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if jobs <= 1:
        for j, t in enumerate(tasks):
            results.append(_eval_task(t))
            if progress:
                n_done = min((j + 1) * chunk, count)
                print(f"  {n_done}/{count} draws, {time.perf_counter() - t0:.1f} s", flush=True)
    else:
        cpu = os.cpu_count() or 1
        threads = max(1, cpu // jobs)
        with cf.ProcessPoolExecutor(
            max_workers=jobs, initializer=_init_worker, initargs=(threads,)
        ) as ex:
            futs = [ex.submit(_eval_task, t) for t in tasks]
            for done, f in enumerate(cf.as_completed(futs), start=1):
                results.append(f.result())
                if progress and done % max(1, len(tasks) // 20) == 0:
                    print(
                        f"  {done}/{len(tasks)} chunks, {time.perf_counter() - t0:.1f} s",
                        flush=True,
                    )
    wall = time.perf_counter() - t0
    by_k: dict[int, tuple[DrawPrediction, PriorDraw]] = {}
    sig: dict[int, np.ndarray] = {}
    alt: dict[int, np.ndarray] = {}
    for r in results:
        for k, p, d in zip(r["k"], r["pred"], r["draw"], strict=True):
            by_k[int(k)] = (p, d)
        sig.update({int(k): v for k, v in r["sigma_pos"].items()})
        alt.update({int(k): v for k, v in r["alt"].items()})
    codes = _codes(protocol)
    preds = [by_k[k][0] for k in ks]
    draws = [by_k[k][1] for k in ks]
    arrays: dict[str, np.ndarray] = {
        "pred_db": np.stack([p.power_db.T for p in preds]).astype(np.float64),
        "dist_m": np.stack([p.dist_m for p in preds]).astype(np.float64),
        "n_quad": np.array([p.n_quad for p in preds], dtype=np.int64),
        "richardson_level": np.array([p.richardson_level for p in preds], dtype=np.int64),
        "richardson_db": np.array([p.richardson_db for p in preds], dtype=np.float64),
        "subcell_max_wavelengths": np.array(
            [p.subcell_max_wavelengths for p in preds], dtype=np.float64
        ),
        "converged_subcell_m": np.array([p.converged_subcell_m for p in preds], dtype=np.float64),
        "n_nearfield_warnings": np.array([p.n_nearfield_warnings for p in preds], dtype=np.int64),
        "n_other_warnings": np.array([p.n_other_warnings for p in preds], dtype=np.int64),
        "min_point_cell_centre_m": np.array([p.min_point_cell_centre_m for p in preds]),
        "min_point_tile_m": np.array([p.min_point_tile_m for p in preds]),
        "frac_points_behind_tiles": np.array([p.frac_points_behind_tiles for p in preds]),
        "ris_to_los_db_median": np.array([p.ris_to_los_db_median for p in preds]),
        "timing_eval_s": np.array([p.eval_s for p in preds], dtype=np.float64),
        "radius_m": np.array([d.radius_m for d in draws]),
        "span_deg": np.array([d.span_deg for d in draws]),
        "centre_deg": np.array([d.centre_deg for d in draws]),
        "d_m": np.array([d.d_m for d in draws]),
        "a_deg": np.array([d.a_deg for d in draws]),
        "gap_m": np.array([d.gap_m for d in draws]),
        "gamma_mag": np.array([d.gamma_mag for d in draws]),
        "gamma_phase_offset_deg": np.array([d.gamma_phase_offset_deg for d in draws]),
        "point_dxy_m": np.stack([d.point_dxy_m for d in draws]),
        "plane_dz_m": np.array([d.plane_dz_m for d in draws]),
        "height_m": np.array([d.height_m for d in draws]),
        "attempts": np.array([d.attempts for d in draws], dtype=np.int64),
    }
    for key, vals in codes.items():
        arrays[f"{key}_idx"] = np.array(
            [vals.index(getattr(d, key)) for d in draws], dtype=np.int64
        )
    if n_sigma_pos:
        arrays["sigma_pos_db"] = np.stack([sig[k] for k in range(min(n_sigma_pos, count))])
    if n_alt:
        arrays["pred_alt_cm_db"] = np.stack([alt[k].T for k in range(min(n_alt, count))]).astype(
            np.float64
        )
    rejected: dict[str, int] = {}
    for d in draws:
        for key, v in d.rejected.items():
            rejected[key] = rejected.get(key, 0) + v
    return {
        "arrays": arrays,
        "codes": {k: list(v) for k, v in codes.items()},
        "rejected": rejected,
        "wall_s": wall,
        "jobs": jobs,
        "draws": draws,
    }


def compute_tensor(
    protocol: Mapping[str, Any],
    *,
    K: int | None = None,
    jobs: int = 1,
    backend: str = "numba_cpu",
    n_sigma_pos: int | None = None,
    n_alt: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """The prediction tensor of ``K`` prior draws (default ``primary_analysis.K``) plus
    ``sigma_pos`` for the first ``uncertainty_budget.sigma_pos.n_draws`` draws and the paired
    other-cell-model predictions for the first ``models.near_field.point_vs_patch.n_draws``."""
    k = int(protocol["primary_analysis"]["K"] if K is None else K)
    nsp = int(
        protocol["uncertainty_budget"]["sigma_pos"]["n_draws"]
        if n_sigma_pos is None
        else n_sigma_pos
    )
    nalt = n_point_vs_patch(protocol) if n_alt is None else int(n_alt)
    res = evaluate_stream(
        protocol,
        protocol["rng"]["prior_draw"],
        k,
        jobs=jobs,
        backend=backend,
        n_sigma_pos=min(nsp, k),
        n_alt=min(nalt, k),
        progress=progress,
    )
    res["K"] = k
    return res


# =========================================================================== cache


def content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """SHA-256 over every array except wall-clock timings (``timing_*``): name, dtype, shape
    and C-order bytes, names sorted. Deterministic for a given code, protocol and host."""
    h = hashlib.sha256()
    for name in sorted(n for n in arrays if not n.startswith("timing_")):
        a = np.ascontiguousarray(arrays[name])
        h.update(
            name.encode() + b"\0" + a.dtype.str.encode() + b"\0" + repr(a.shape).encode() + b"\0"
        )
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class TensorCache:
    """A loaded prediction tensor."""

    arrays: Mapping[str, np.ndarray]
    meta: Mapping[str, Any]
    path: Path

    @property
    def pred_db(self) -> np.ndarray:
        """``[K, 30, 4]`` model powers (dB), MODELS order."""
        return self.arrays["pred_db"]

    @property
    def K(self) -> int:
        """Number of draws."""
        return int(self.pred_db.shape[0])

    def model(self, name: str) -> np.ndarray:
        """``[K, 30]`` powers of one model."""
        return self.pred_db[:, :, MODELS.index(name)]

    def codes(self, key: str) -> np.ndarray:
        """Per-draw string labels of a categorical prior variable."""
        vals = self.meta["codes"][key]
        return np.asarray(vals, dtype=object)[self.arrays[f"{key}_idx"]]


def save_cache(
    path: str | Path,
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``result['arrays']`` and its metadata to ``path`` (npz); return the metadata
    (``content_sha256``, ``prediction_spec_sha256``, ...)."""
    arrays = dict(result["arrays"])
    meta = {
        "kind": "vlab_prediction_tensor",
        "models": list(MODELS),
        "K": int(arrays["pred_db"].shape[0]),
        "codes": result["codes"],
        "prediction_spec_sha256": prediction_spec_sha256(protocol),
        "content_sha256": content_sha256(arrays),
        "rejected_candidates": result.get("rejected", {}),
        "data_class": "SIM-sremu",
        **dict(extra_meta or {}),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, meta_json=np.array(json.dumps(meta, sort_keys=True)), **arrays)
    os.replace(tmp, p)
    meta["file_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return meta


def load_cache(path: str | Path, *, protocol: Mapping[str, Any] | None = None) -> TensorCache:
    """Load and verify a tensor cache: the content hash must match its metadata and, if
    ``protocol`` is given, the prediction spec hash must match the protocol's."""
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        arrays = {k: z[k] for k in z.files if k != "meta_json"}
    got = content_sha256(arrays)
    if got != meta.get("content_sha256"):
        raise ValueError(f"{p}: content hash {got} != recorded {meta.get('content_sha256')}")
    if protocol is not None and meta.get("prediction_spec_sha256") != prediction_spec_sha256(
        protocol
    ):
        raise ValueError(
            f"{p}: computed for another prediction spec ({meta.get('prediction_spec_sha256')}) "
            f"than the protocol's ({prediction_spec_sha256(protocol)})"
        )
    return TensorCache(arrays=arrays, meta=meta, path=p)


# =========================================================================== CLI


def point_vs_patch_spread(
    arrays: Mapping[str, np.ndarray],
    codes: Mapping[str, Sequence[str]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Model-form spread of the paired draws (``models.near_field.point_vs_patch``): per ring,
    quantiles over draws and the ring's cells of ``|M2(point) - M2(po_patch)|`` after removing
    each draw's mean difference over the primary training cells (the fitted offset absorbs
    it), and of the raw difference. Data-independent."""
    alt = arrays.get("pred_alt_cm_db")
    if alt is None or alt.shape[0] == 0:
        return {"n_draws": 0}
    n = int(alt.shape[0])
    base = np.asarray(arrays["pred_db"][:n], dtype=np.float64)
    cm = np.asarray(codes["cell_model"], dtype=object)[arrays["cell_model_idx"][:n]]
    is_point = (cm == "point")[:, None]
    j = MODELS.index("M2")
    d = np.where(is_point, base[:, :, j], alt[:, :, j]) - np.where(
        is_point, alt[:, :, j], base[:, :, j]
    )
    ok = np.all(np.isfinite(d), axis=1)
    d = d[ok]
    lay = cell_layout(protocol)
    train, _test = split_masks(protocol, lay)["primary"]
    dc = d - d[:, train].mean(axis=1, keepdims=True)
    rings: dict[str, Any] = {}
    for ir, r in enumerate(lay.rings_m):
        sel = lay.ring == ir
        a, b = np.abs(dc[:, sel]).ravel(), np.abs(d[:, sel]).ravel()
        rings[f"{r:g}"] = {
            "median_abs_centred_db": float(np.median(a)) if a.size else math.nan,
            "q95_abs_centred_db": float(np.quantile(a, 0.95)) if a.size else math.nan,
            "median_abs_raw_db": float(np.median(b)) if b.size else math.nan,
            "q95_abs_raw_db": float(np.quantile(b, 0.95)) if b.size else math.nan,
        }
    return {
        "n_draws": n,
        "n_alternative_not_converged": int(np.sum(~ok)),
        "definition": "M2(point) - M2(po_patch) at the same geometry, dB",
        "rings_m": rings,
    }


def _q(x: np.ndarray, qs: Sequence[float] = (0.05, 0.5, 0.95)) -> list[float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return [float(v) for v in np.quantile(a, qs)] if a.size else [math.nan] * len(qs)


def _summary(res: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    a = res["arrays"]
    codes = res["codes"]
    k = a["pred_db"].shape[0]
    lam = C0 / float(protocol["prior"]["fc_hz"])
    cm = np.asarray(codes["cell_model"], dtype=object)[a["cell_model_idx"]]
    patch = cm == "po_patch"
    out: dict[str, Any] = {
        "K": int(k),
        "variant": protocol.get("variant", "primary"),
        "geometry_model": protocol["prior"]["geometry"]["model"],
        "wall_s": float(res["wall_s"]),
        "jobs": int(res["jobs"]),
        "eval_s_total": float(np.sum(a["timing_eval_s"])),
        "eval_s_by_cell_model": {
            str(c): {
                "n": int(np.sum(cm == c)),
                "mean_s": float(np.mean(a["timing_eval_s"][cm == c])) if np.any(cm == c) else None,
                "median_n_quad": float(np.median(a["n_quad"][cm == c]))
                if np.any(cm == c)
                else None,
            }
            for c in codes["cell_model"]
        },
        "candidates_total": int(np.sum(a["attempts"])),
        "acceptance_rate": float(k / np.sum(a["attempts"])),
        "rejected_by_reason": res.get("rejected", {}),
        "accepted_fraction": {
            key: {str(v): float(np.mean(a[f"{key}_idx"] == i)) for i, v in enumerate(codes[key])}
            for key in codes
        },
        "continuous_quantiles_q05_q50_q95": {
            key: _q(a[key])
            for key in ("radius_m", "span_deg", "centre_deg", "d_m", "a_deg", "gap_m")
            if np.any(np.isfinite(a[key]))
        },
        # near field (models.near_field): nodes between 1 and 3 wavelengths of a cell are
        # kept; sremu's 3-lambda ValidityWarning is suppressed and counted, other
        # ValidityWarnings are counted separately
        "nearfield_warnings_total": int(np.sum(a["n_nearfield_warnings"])),
        "nearfield_warning_draws": int(np.sum(a["n_nearfield_warnings"] > 0)),
        "fraction_draws_with_nearfield_warnings": float(np.mean(a["n_nearfield_warnings"] > 0)),
        "other_validity_warnings_total": int(np.sum(a["n_other_warnings"])),
        "min_point_cell_centre_m_q05_q50_q95": _q(a["min_point_cell_centre_m"]),
        "min_point_cell_centre_wavelengths_min": float(np.min(a["min_point_cell_centre_m"]) / lam),
        "min_point_tile_m_q05_q50_q95": _q(a["min_point_tile_m"]),
        "min_point_tile_wavelengths_min": float(np.min(a["min_point_tile_m"]) / lam),
        "fraction_draws_with_points_behind_a_tile": float(
            np.mean(a["frac_points_behind_tiles"] > 0.0)
        ),
        "po_patch_initial_subcell_counts": {
            f"{float(s):g}": int(np.sum((a["subcell_max_wavelengths"] == s) & patch))
            for s in np.unique(a["subcell_max_wavelengths"][patch])
        },
        "po_patch_converged_subcell_mm_counts": {
            f"{1e3 * float(s):.3f}": int(np.sum((a["converged_subcell_m"] == s) & patch))
            for s in np.unique(a["converged_subcell_m"][patch])
        },
        "richardson_level_counts": {
            str(int(v)): int(np.sum(a["richardson_level"] == v))
            for v in np.unique(a["richardson_level"])
        },
        "sd_over_cells_m2_minus_m0_db_q05_q50_q95": _q(
            np.std(a["pred_db"][:, :, 2] - a["pred_db"][:, :, 0], axis=1)
        ),
        "sigma_pos_db_median": (
            float(np.nanmedian(a["sigma_pos_db"]))
            if "sigma_pos_db" in a and np.any(np.isfinite(a["sigma_pos_db"]))
            else None
        ),
        "sigma_pos_nan_draws": (
            int(np.sum(np.all(np.isnan(a["sigma_pos_db"]), axis=1))) if "sigma_pos_db" in a else 0
        ),
        "ris_to_los_db_median_over_draws": float(np.median(a["ris_to_los_db_median"])),
        "point_vs_patch": point_vs_patch_spread(a, codes, protocol),
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Compute and cache the prediction tensor in a new run directory."""
    from validation.ants_fig4b import runlog

    ap = argparse.ArgumentParser(prog="python -m validation.ants_fig4b.predict")
    ap.add_argument("--out", type=Path, default=Path("runs") / "vlab", help="run root")
    ap.add_argument("--K", type=int, default=None, help="draws (default: protocol K)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--backend", default="numba_cpu", choices=["numba_cpu", "numpy_ref"])
    ap.add_argument("--n-sigma-pos", type=int, default=None)
    ap.add_argument("--n-alt", type=int, default=None, help="paired point-vs-patch draws")
    ap.add_argument(
        "--variant",
        default=None,
        help="alternative-prior sensitivity run (e.g. S-OR); default: the primary prior",
    )
    ap.add_argument("--name", default=None, help="run name (default vlab_tensor[_<variant>])")
    ap.add_argument("--bg-seconds", type=float, default=5.0, help="background-load probe")
    args = ap.parse_args(argv)
    protocol = variant_protocol(load_protocol(), args.variant)
    name = args.name or ("vlab_tensor" + (f"_{args.variant}" if args.variant else ""))
    rec = runlog.RunRecorder(args.out, name, protocol, argv=sys.argv)
    bg_before = runlog.background_load(args.bg_seconds)
    rec.sample_load()
    res = compute_tensor(
        protocol,
        K=args.K,
        jobs=args.jobs,
        backend=args.backend,
        n_sigma_pos=args.n_sigma_pos,
        n_alt=args.n_alt,
        progress=True,
    )
    rec.sample_load()
    bg_after = runlog.background_load(args.bg_seconds)
    path = rec.run_dir / "pred_tensor.npz"
    meta = save_cache(
        path,
        res,
        protocol,
        extra_meta={
            "git_commit": rec.git.commit,
            "git_dirty": rec.git.dirty,
            "variant": protocol.get("variant", "primary"),
        },
    )
    summ = _summary(res, protocol)
    summ.update({k: meta[k] for k in ("content_sha256", "file_sha256", "prediction_spec_sha256")})
    summ["background_cpu_busy_fraction"] = {
        "probe_s": args.bg_seconds,
        "before": bg_before,
        "after": bg_after,
    }
    rec.write_json("summary.json", summ)
    rec.write_csv("draws.csv", [d.as_record() for d in res["draws"]])
    rec.add_output("pred_tensor.npz", "SIM-sremu")
    rec.add_output("draws.csv", "ASSUMED")
    rec.add_output("summary.json", "SIM-sremu")
    rec.finish(backend=args.backend, extra={"tensor": summ})
    print(json.dumps(summ, indent=2))
    print(f"run directory: {rec.run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
