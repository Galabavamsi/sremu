"""The V-LAB prediction tensor ``pred[K, 30, models]`` (DESIGN.md section 10.9, "Primary").

For each prior draw ``k`` (keyed, :func:`validation.ants_fig4b.prior.draw_prior`) one sremu
scene is built with :func:`sremu.physics.propagation.from_scenario` (numba backend, the
draw's ``cell_model``, layout, pitch, gap, antennas and floor) and evaluated at ``t = 0``.
From the snapshot:

* ``h_los``: the LOS path, ``h_gnd``: the ground image path (``PathKind.GROUND``);
* ``G_v[n] = sum_{rows (link, v)} C[r, n] exp(-j 2 pi f_c tau_tile[r])`` from the separable
  RIS coefficients (carrier phase restored exactly as ``ScenePropagation.evaluate``), split
  into the direct hop (variant 0) and the ground hop variants 1-3;
* the drawn ``Gamma_s = |Gamma_s| exp(j(arg Gamma_s^table + dphi_s))`` is applied to the
  layout's states.

Models (``protocol.json`` ``models``): ``M0 = |h_los|^2``, ``M1 = |h_los + G_0 . Gamma_1|^2``,
``M2 = |h_los + G_0 . Gamma[s]|^2``, ``M3 = |h_los + h_gnd + (G_0 + G_img) . Gamma[s]|^2``, in
dB (unit Tx power). ``dist_m[K, 30]`` (Tx-point distance) is the ``M0n`` regressor.

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
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")  # deterministic mode (D-4)

import numpy as np
from validation.ants_fig4b.prior import PriorDraw, draw_prior, geometry, scenario_dict
from validation.ants_fig4b.protocol import (
    load_protocol,
    prediction_spec_sha256,
)

from sremu.contracts import PathKind
from sremu.errors import RefinementNotConverged, ValidityWarning
from sremu.physics.propagation import from_scenario

__all__ = [
    "MODELS",
    "DrawPrediction",
    "TensorCache",
    "compute_tensor",
    "content_sha256",
    "draw_and_evaluate",
    "evaluate_draw",
    "load_cache",
    "position_sigma",
    "save_cache",
]

MODELS: Final[tuple[str, ...]] = ("M0", "M1", "M2", "M3")
_CAT_FIELDS: Final[dict[str, tuple[str, ...]]] = {}


@dataclass(frozen=True, eq=False)
class DrawPrediction:
    """Predictions of one draw: ``power_db[4, 30]`` (MODELS order), ``dist_m[30]`` and
    diagnostics."""

    power_db: np.ndarray
    dist_m: np.ndarray
    n_quad: int
    richardson_level: int
    richardson_db: float
    subcell_max_wavelengths: float
    n_validity_warnings: int
    frac_points_behind_panel: float
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
    """``(gamma_table[S], G_direct[L, N], G_images[L, N])`` with the carrier phase restored."""
    rc = snap.ris["ris"]
    if rc.C is None or rc.gamma is None:
        raise RuntimeError("expected separable RIS coefficients (angle-independent table)")
    row_of = {lk: i for i, lk in enumerate(links)}
    n = rc.n_cells
    g0 = np.zeros((len(links), n), dtype=np.complex128)
    gi = np.zeros((len(links), n), dtype=np.complex128)
    for r, (lk, var) in enumerate(rc.rows):
        ph = np.exp(-2j * np.pi * np.mod(fc * rc.tile_delay_s[r][rc.tile_of_cell], 1.0))
        term = rc.C[r] * ph
        if var == 0:
            g0[row_of[lk]] += term
        else:
            gi[row_of[lk]] += term
    return np.asarray(rc.gamma), g0, gi


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


def _build(
    draw: PriorDraw, protocol: Mapping[str, Any], backend: str, **kw: Any
) -> tuple[Any, float, list[Any]]:
    """``(model, subcell_max_wavelengths used, caught warnings)``: the ``po_patch`` quadrature
    escalation of ``models.po_patch_quadrature`` (sremu's default first, then finer initial
    sub-cells when sremu raises ``RefinementNotConverged``; the last failure propagates)."""
    levels = [
        float(x) for x in protocol["models"]["po_patch_quadrature"]["subcell_max_wavelengths"]
    ]
    if draw.cell_model != "po_patch":
        levels = levels[:1]
    for j, sub in enumerate(levels):
        sc = scenario_dict(draw, protocol, backend=backend, subcell_max_wavelengths=sub, **kw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ValidityWarning)
            try:
                model = from_scenario(sc)
            except RefinementNotConverged:
                if j == len(levels) - 1:
                    raise
                continue
        return model, sub, list(caught)
    raise AssertionError("unreachable")


def evaluate_draw(
    draw: PriorDraw, protocol: Mapping[str, Any], *, backend: str = "numba_cpu"
) -> DrawPrediction:
    """Model powers (dB) of ``draw`` at the 30 points: one sremu scene with the floor (M3);
    M0-M2 use its LOS path and direct-hop RIS rows only. Raises
    ``RefinementNotConverged`` when the ``po_patch`` quadrature escalation fails."""
    t0 = time.perf_counter()
    floor_db = float(protocol["models"]["power_floor_db"])
    model, sub, caught = _build(draw, protocol, backend, ground=True)
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always", ValidityWarning)
        snap = model.snapshot(0)
    n_warn = sum(1 for w in [*caught, *caught2] if issubclass(w.category, ValidityWarning))
    fc = float(protocol["prior"]["fc_hz"])
    links = _rx_links(model)
    los, gnd = _paths(snap, links, fc)
    gtab, g0, gi = _hops(model, snap, links, fc)
    geo = geometry(draw, protocol)
    gam = _gamma(draw, gtab)
    s = geo.cell_state.astype(np.int64)
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
    side = (geo.points_m - geo.ris_center_m[None, :]) @ geo.normal
    rich = model.richardson("ris")
    with np.errstate(divide="ignore"):
        ratio = 20.0 * np.log10(np.abs(h_ris_m2) / np.abs(los))
    return DrawPrediction(
        power_db=_floor_db(p, floor_db),
        dist_m=dist,
        n_quad=int(model.quadrature("ris").n_points),
        richardson_level=-1 if rich is None else int(rich.level),
        richardson_db=math.nan if rich is None else float(rich.estimate_db),
        subcell_max_wavelengths=sub,
        n_validity_warnings=n_warn,
        frac_points_behind_panel=float(np.mean(side <= 0.0)),
        ris_to_los_db_median=float(np.median(ratio)),
        eval_s=time.perf_counter() - t0,
    )


def position_sigma(
    draw: PriorDraw, protocol: Mapping[str, Any], *, backend: str = "numba_cpu"
) -> np.ndarray:
    """``sigma_pos[30]`` (dB) of the M2 prediction: central differences at fixed quadrature
    (the points move as external trajectories of one scene), propagated with the uniform
    standard deviations ``half_width/sqrt(3)`` (``uncertainty_budget.sigma_pos``)."""
    sp = protocol["uncertainty_budget"]["sigma_pos"]
    step = float(sp["fd_step_m"])
    sd_xy = float(sp["xy_half_width_m"]) / math.sqrt(3.0)
    sd_z = float(sp["z_half_width_m"]) / math.sqrt(3.0)
    fc = float(protocol["prior"]["fc_hz"])
    geo = geometry(draw, protocol)
    s = geo.cell_state.astype(np.int64)
    model, _sub, _caught = _build(draw, protocol, backend, ground=False, external_rx=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
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
                snap = model.snapshot(t)
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


_ORIGINS = ("O_R", "O_T")
_THETA0 = ("normal", "plane")


def _codes(protocol: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    pr = protocol["prior"]
    return {
        "origin": tuple(pr["polar_origin"]["values"]),
        "theta0": tuple(pr["theta0_direction"]["values"]),
        "layout": tuple(pr["layout"]["values"]),
        "pitch": tuple(pr["pitch"]["values"]),
        "cell_model": tuple(pr["cell_model"]["values"]),
        "antenna": tuple(pr["antenna"]["values"]),
        "polarization": tuple(pr["floor"]["polarization"]["values"]),
    }


def _eval_task(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: evaluate draws ``task['ks']`` of stream ``task['stream']`` (picklable)."""
    protocol = task["protocol"]
    st = task["stream"]
    out: dict[str, Any] = {"k": [], "pred": [], "draw": [], "sigma_pos": {}}
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
    chunk: int = 8,
    progress: bool = False,
) -> dict[str, Any]:
    """Evaluate draws ``0..count-1`` of a keyed stream; arrays in index order."""
    ks = list(range(count))
    tasks = [
        {
            "protocol": dict(protocol),
            "stream": dict(stream),
            "ks": ks[i : i + chunk],
            "backend": backend,
            "n_sigma_pos": n_sigma_pos,
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
    for r in results:
        for k, p, d in zip(r["k"], r["pred"], r["draw"], strict=True):
            by_k[int(k)] = (p, d)
        sig.update({int(k): v for k, v in r["sigma_pos"].items()})
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
        "n_validity_warnings": np.array([p.n_validity_warnings for p in preds], dtype=np.int64),
        "frac_points_behind_panel": np.array([p.frac_points_behind_panel for p in preds]),
        "ris_to_los_db_median": np.array([p.ris_to_los_db_median for p in preds]),
        "timing_eval_s": np.array([p.eval_s for p in preds], dtype=np.float64),
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
    progress: bool = False,
) -> dict[str, Any]:
    """The prediction tensor of ``K`` prior draws (default ``primary_analysis.K``) plus
    ``sigma_pos`` for the first ``uncertainty_budget.sigma_pos.n_draws`` draws."""
    k = int(protocol["primary_analysis"]["K"] if K is None else K)
    nsp = int(
        protocol["uncertainty_budget"]["sigma_pos"]["n_draws"]
        if n_sigma_pos is None
        else n_sigma_pos
    )
    res = evaluate_stream(
        protocol,
        protocol["rng"]["prior_draw"],
        k,
        jobs=jobs,
        backend=backend,
        n_sigma_pos=min(nsp, k),
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


def _summary(res: Mapping[str, Any]) -> dict[str, Any]:
    a = res["arrays"]
    codes = res["codes"]
    k = a["pred_db"].shape[0]
    cm = np.asarray(codes["cell_model"], dtype=object)[a["cell_model_idx"]]
    out: dict[str, Any] = {
        "K": int(k),
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
        "n_validity_warnings_total": int(np.sum(a["n_validity_warnings"])),
        # draws with a node between 1 and 3 wavelengths of a cell (sremu ValidityWarning:
        # point-scatterer approximation marginal, DESIGN 5.3); kept, see PROTOCOL.md 4.1
        "draws_with_validity_warnings": int(np.sum(a["n_validity_warnings"] > 0)),
        "fraction_draws_with_validity_warnings": float(np.mean(a["n_validity_warnings"] > 0)),
        "fraction_draws_with_points_behind_panel": float(
            np.mean(a["frac_points_behind_panel"] > 0.0)
        ),
        "po_patch_initial_subcell_counts": {
            f"{float(s):g}": int(np.sum((a["subcell_max_wavelengths"] == s) & (cm == "po_patch")))
            for s in np.unique(a["subcell_max_wavelengths"])
        },
        "sd_over_cells_m2_minus_m0_db_q05_q50_q95": [
            float(v)
            for v in np.quantile(
                np.std(a["pred_db"][:, :, 2] - a["pred_db"][:, :, 0], axis=1), [0.05, 0.5, 0.95]
            )
        ],
        "sigma_pos_db_median": (
            float(np.median(a["sigma_pos_db"])) if "sigma_pos_db" in a else None
        ),
        "richardson_level_counts": {
            str(int(v)): int(np.sum(a["richardson_level"] == v))
            for v in np.unique(a["richardson_level"])
        },
        "ris_to_los_db_median_over_draws": float(np.median(a["ris_to_los_db_median"])),
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
    ap.add_argument("--name", default="vlab_tensor")
    ap.add_argument("--bg-seconds", type=float, default=5.0, help="background-load probe")
    args = ap.parse_args(argv)
    protocol = load_protocol()
    rec = runlog.RunRecorder(args.out, args.name, protocol, argv=sys.argv)
    bg_before = runlog.background_load(args.bg_seconds)
    rec.sample_load()
    res = compute_tensor(
        protocol,
        K=args.K,
        jobs=args.jobs,
        backend=args.backend,
        n_sigma_pos=args.n_sigma_pos,
        progress=True,
    )
    rec.sample_load()
    bg_after = runlog.background_load(args.bg_seconds)
    path = rec.run_dir / "pred_tensor.npz"
    meta = save_cache(
        path, res, protocol, extra_meta={"git_commit": rec.git.commit, "git_dirty": rec.git.dirty}
    )
    summ = _summary(res)
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
