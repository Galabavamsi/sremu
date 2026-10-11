"""E-C: non-ideality ablation and binary decomposition (DESIGN.md section 11, E-C; RQ-60).

Batch (coefficient-level) evaluation only, exact finite-alphabet optimum
(``sremu.physics.optimize.exact_alphabet_optimum``) per alphabet. The channel of every
geometry comes from the scene model (``sremu.physics.propagation.from_scenario`` +
``sremu.control.controller.csi_from_snapshot``) with a unit-amplitude separable alphabet,
which gives the Gamma-free cell factor C_n (cascade row of the 1 at 0 deg state); a row's
alphabet Gamma_s then gives ``Z[s, n] = Gamma_s C_n`` (angle-independent alphabets, as every
row here is).

Geometries:

1. ``lab``: lab-like panel (``scenarios/lab_like_A.yaml``: 24 x 5 cells at 42.8 mm, three
   tiles, 2 mm gaps, 3.5 GHz), Tx at the V-LAB prior median (2 m at -22.5 deg), Rx on the
   30-point polar grid (r in {0.6, 0.9, 1.2, 1.5, 1.8} m, theta in {0, 5, 15, 25, 35, 45}
   deg from the panel normal). SIM geometry; no measured data is read.
2. ``synth_near`` / ``synth_far``: 16 x 16 panel at lambda/2, 3.5 GHz, Tx 5 m at -30 deg,
   Rx 5 m / 500 m at +20 deg; ``synth_spec``: Rx 5 m at +30 deg (specular; added for the
   mirror-term row, as DESIGN.md asks for "the specular geometry").
3. ``corridor``: E-A A1 (1 GHz, h = 2 m, LOS partition 20 dB), averaged over the RWP
   stationary pdf ``f(x) = 6 x (l - x)/l^3`` on a 1 cm grid.

Rows: ideal continuous phase (reference), ideal 1-bit, ideal 2-bit, lab |Gamma| = 1, lab
|Gamma2| in {0.7, 0.5}, Tang RIS1 (0.9 at 165 deg, 0.7 at 0 deg; values of the published
table applied as an alphabet at each geometry's carrier [ENG]), Abeywickrama (beta_min 0.2,
alpha 1.6, b = 2, phi = 0), 10 % dead (absorbing), 10 % stuck at a random state, efficiency
0.5, and IdealDesigned (exact optimum for {1, -1} deployed on the lab states). Failure rows
are on the lab |Gamma| = 1 alphabet; failure maps from ``FailureMap.random`` with keyed
generators (purpose ``failure``).

Outputs: loss vs the ideal continuous phase of (a) the RIS term alone (z0 = 0) and (b) the
total received power (direct path included), power-averaged over each geometry's points;
the large-N eta column; the binary m/d decomposition and the mirror-term power share.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import (
    REPO,
    RunRecord,
    db10,
    quiet_warnings,
    write_csv,
    write_json,
    write_meta,
)
from sremu import rng as _rng
from sremu.control.controller import csi_from_snapshot
from sremu.physics.nonideal import (
    FailureMap,
    binary_md,
    eta_abeywickrama,
    eta_alphabet,
)
from sremu.physics.optimize import continuous_optimum, exact_alphabet_optimum
from sremu.physics.propagation import from_scenario
from sremu.physics.response import builtin
from sremu.scenario.load import parse_scenario, read_document, resolve_scenario
from sremu.units import C0

LAB = REPO / "scenarios" / "lab_like_A.yaml"
SEED = 20261010
N_MAPS = 200
FAIL_RATE = 0.10


def _phasor(mag: float, deg: float) -> complex:
    return mag * complex(math.cos(math.radians(deg)), math.sin(math.radians(deg)))


def lab_gammas(mag2: float = 1.0) -> np.ndarray:
    """(sigma1, sigma2) of ``lab_binary_3p5ghz`` with |sigma1| = 1, |sigma2| = mag2 (the
    library's phases; amplitudes ASSUMED)."""
    tab = builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1.0, "sigma2": float(mag2)})
    g = np.asarray(tab.gamma[:, 0, 0, 0], dtype=np.complex128)
    return g


def alphabet_rows() -> list[dict[str, Any]]:
    """Every ablation row: name, alphabet (S,) Gamma, kind and the large-N eta (power)."""
    lab1 = lab_gammas(1.0)
    tang = builtin("tang2022_ris1_27ghz")
    tang_g = np.asarray(tang.gamma[:, 0, 0, 0], dtype=np.complex128)
    th = 2 * np.pi * np.arange(4) / 4
    beta = (1 - 0.2) * ((np.sin(th) + 1) / 2) ** 1.6 + 0.2
    abey = beta * np.exp(1j * th)
    eta_lab = eta_alphabet(lab1)
    rows = [
        {"row": "ideal_continuous", "gamma": None, "kind": "continuous", "eta": 1.0},
        {"row": "ideal_1bit", "gamma": np.array([1.0, -1.0], complex), "kind": "alphabet"},
        {"row": "ideal_2bit", "gamma": np.exp(1j * th), "kind": "alphabet"},
        {"row": "lab_mag1", "gamma": lab1, "kind": "alphabet"},
        {"row": "lab_mag2_0p7", "gamma": lab_gammas(0.7), "kind": "alphabet"},
        {"row": "lab_mag2_0p5", "gamma": lab_gammas(0.5), "kind": "alphabet"},
        {"row": "tang_ris1", "gamma": tang_g, "kind": "alphabet"},
        {"row": "abeywickrama_b2", "gamma": abey, "kind": "alphabet"},
        {"row": "lab_dead10", "gamma": lab1, "kind": "dead", "eta": (1 - FAIL_RATE) ** 2 * eta_lab},
        {"row": "lab_stuck10", "gamma": lab1, "kind": "stuck", "eta": None},
        {"row": "lab_eff0p5", "gamma": lab1 * math.sqrt(0.5), "kind": "alphabet"},
        {"row": "ideal_designed_on_lab", "gamma": lab1, "kind": "ideal_designed", "eta": None},
    ]
    for r in rows:
        if "eta" not in r:
            r["eta"] = eta_alphabet(r["gamma"])
    rows[7]["eta_continuous_abeywickrama"] = eta_abeywickrama(0.2, 1.6)
    return rows


# =========================================================================== geometries


def _polar(r: float, deg: float) -> list[float]:
    a = math.radians(deg)
    return [r * math.cos(a), r * math.sin(a), 0.0]


def _node(nid: str, pos: Sequence[float], direction: str, fc: float) -> dict[str, Any]:
    st = {
        "name": "tx0" if direction == "tx" else "rx0",
        "direction": direction,
        "fc_hz": fc,
        "rate_sps": "1e6",
    }
    if direction == "tx":
        st["occupied_bw_hz"] = 1.0e4
    return {
        "id": nid,
        "position_m": [float(x) for x in pos],
        "antenna": {"pattern": "isotropic"},
        "streams": [st],
    }


def lab_geometry_scenario() -> tuple[dict[str, Any], list[tuple[float, float]]]:
    """Lab-like panel, Tx at 2 m / -22.5 deg, 30 Rx on the polar grid (ideal_antipodal)."""
    d = copy.deepcopy(read_document(LAB))
    fc = 3.5e9
    pts = [(r, th) for r in (0.6, 0.9, 1.2, 1.5, 1.8) for th in (0, 5, 15, 25, 35, 45)]
    nodes = [_node("ap", _polar(2.0, -22.5), "tx", fc)]
    for i, (r, th) in enumerate(pts):
        nodes.append(_node(f"r{i:02d}", _polar(r, th), "rx", fc))
    d.update(
        name="ec_lab_polar30",
        description="E-C lab-like panel, Tx at the V-LAB prior median, 30-point polar grid (SIM)",
        nodes=nodes,
        controllers=[],
        traffic=[],
    )
    d["ris"][0]["states"] = {"library": "ideal_antipodal"}
    d["ris"][0].pop("initial", None)
    return d, pts


def synth_scenario() -> dict[str, Any]:
    """16 x 16 panel at lambda/2 (3.5 GHz), Tx 5 m at -30 deg; Rx 5 m at +20 deg (near),
    500 m at +20 deg (far), 5 m at +30 deg (specular)."""
    fc = 3.5e9
    lam = C0 / fc
    nodes = [
        _node("ap", _polar(5.0, -30.0), "tx", fc),
        _node("near", _polar(5.0, 20.0), "rx", fc),
        _node("far", _polar(500.0, 20.0), "rx", fc),
        _node("spec", _polar(5.0, 30.0), "rx", fc),
    ]
    return {
        "schema_version": "1.0",
        "name": "ec_synth16",
        "seed": SEED,
        "description": "E-C synthetic 16 x 16 lambda/2 panel at 3.5 GHz (SIM)",
        "nodes": nodes,
        "ris": [
            {
                "id": "ris0",
                "pose": {"center_m": [0, 0, 0], "normal": [1, 0, 0], "u_axis": [0, 1, 0]},
                "grid": {"n_u": 16, "n_v": 16, "pitch_u_m": lam / 2, "pitch_v_m": lam / 2},
                "states": {"library": "ideal_antipodal"},
            }
        ],
        "run": {"duration_s": "1"},
    }


def _resolve(d: Mapping[str, Any]) -> Any:
    from sremu.scenario.guards import check_scenario

    quiet_warnings()
    sc = parse_scenario(dict(d), label=str(d.get("name")))
    res = resolve_scenario(sc, LAB.parent, warn=False)
    check_scenario(res).raise_for_errors()
    return res


@dataclass
class PointSet:
    """CSI of one geometry: per point ``h0`` and Gamma-free ``c`` (N,), with weights."""

    name: str
    h0: np.ndarray  # (P,)
    c: np.ndarray  # (P, N)
    w: np.ndarray  # (P,) weights (sum 1)
    labels: list[str]
    resolved: Any


def csi_points(d: Mapping[str, Any], rx_nodes: Sequence[str], name: str) -> PointSet:
    """``h0`` and ``C`` (cascade row of state 0 = 1 at 0 deg) for every Rx node."""
    quiet_warnings()
    res = _resolve(d)
    model = from_scenario(dict(d), backend="numpy_ref")
    snap = model.snapshot(Fraction(0))
    h0, cc = [], []
    for nd in rx_nodes:
        v = csi_from_snapshot(model, snap, f"ap.tx0->{nd}.rx0")
        z = v.cascade["ris0"]
        assert np.allclose(z[1], -z[0], rtol=0, atol=1e-30 + 1e-12 * np.abs(z[0]).max())
        h0.append(v.h_direct)
        cc.append(z[0])
    p = len(rx_nodes)
    return PointSet(name, np.asarray(h0), np.asarray(cc), np.full(p, 1.0 / p), list(rx_nodes), res)


def corridor_points(cache_dir: str | Path) -> PointSet:
    """E-A A1 corridor (1 GHz, h = 2 m, 20 dB) on a 1 cm grid, RWP stationary pdf weights."""
    from experiments import ea

    g = ea.build_geometry(1e9, 2, 20, cache_dir)
    step = g.den // 100
    xs = g.x[::step]
    keep = (xs > 0) & (xs < ea.L_M)
    l_m = ea.L_M
    f = 6 * xs * (l_m - xs) / l_m**3
    w = f * keep
    w = w / w.sum()
    z = g.z[::step]
    assert np.all(z[:, 0, :] == 0)  # on_off: OFF absorbing
    res = ea.resolve(ea.corridor_scenario(1e9, 2, 20))
    sel = np.nonzero(keep)[0]
    return PointSet(
        "corridor", g.hd[::step][sel], z[sel, 1, :], w[sel], [f"x={x:.2f}" for x in xs[sel]], res
    )


# =========================================================================== evaluation


def _opt(z0: complex, z: np.ndarray) -> float:
    return exact_alphabet_optimum(z0, z)[0] ** 2


def eval_row(
    row: Mapping[str, Any], ps: PointSet, *, n_maps: int = N_MAPS
) -> dict[str, np.ndarray]:
    """Optimal |h|^2 per point for the RIS term alone (z0 = 0) and with the direct path."""
    kind = row["kind"]
    n_pts, n = ps.c.shape
    ris = np.empty(n_pts)
    tot = np.empty(n_pts)
    for i in range(n_pts):
        c, h0 = ps.c[i], complex(ps.h0[i])
        if kind == "continuous":
            ris[i] = continuous_optimum(0j, c)[0] ** 2
            tot[i] = continuous_optimum(h0, c)[0] ** 2
            continue
        g = np.asarray(row["gamma"], dtype=np.complex128)
        z = g[:, None] * c[None, :]
        if kind == "alphabet":
            ris[i] = _opt(0j, z)
            tot[i] = _opt(h0, z)
        elif kind == "ideal_designed":
            zd = np.array([1.0, -1.0])[:, None] * c[None, :]
            cols = np.arange(n)
            _, s_r = exact_alphabet_optimum(0j, zd)
            _, s_t = exact_alphabet_optimum(h0, zd)
            ris[i] = abs(complex(z[s_r.astype(int), cols].sum())) ** 2
            tot[i] = abs(h0 + complex(z[s_t.astype(int), cols].sum())) ** 2
        elif kind in ("dead", "stuck"):
            acc_r = acc_t = 0.0
            m_use = n_maps if n_pts <= 64 else max(1, n_maps // 20)
            for m in range(m_use):
                gen = _rng.generator(SEED, "failure", f"ec/{ps.name}/{kind}/{i}/{m}")
                fm = FailureMap.random(n, FAIL_RATE, kind, gen, n_states=g.size)
                alive = ~fm.dead & (fm.stuck_at < 0)
                stuck = fm.stuck_at >= 0
                fixed = complex(np.sum(g[fm.stuck_at[stuck]] * c[stuck])) if np.any(stuck) else 0j
                zz = z[:, alive]
                acc_r += _opt(fixed, zz)
                acc_t += _opt(h0 + fixed, zz)
            ris[i] = acc_r / m_use
            tot[i] = acc_t / m_use
        else:
            raise ValueError(kind)
    return {"ris": ris, "tot": tot}


def run(out_dir: str | Path, cache_dir: str | Path, *, n_maps: int = N_MAPS) -> dict[str, Any]:
    """Compute Table VIII data and write the run directory."""
    quiet_warnings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d_lab, pts = lab_geometry_scenario()
    lab = csi_points(d_lab, [f"r{i:02d}" for i in range(len(pts))], "lab")
    syn = synth_scenario()
    near = csi_points(syn, ["near"], "synth_near")
    far = csi_points(syn, ["far"], "synth_far")
    spec = csi_points(syn, ["spec"], "synth_spec")
    cor = corridor_points(cache_dir)
    geos = [lab, near, far, spec, cor]
    run_rec = RunRecord(
        out, SEED, resolved=lab.resolved, mode="batch", coupling="none", patch_root=out
    )
    for ps in (near, cor):
        for inp in ps.resolved.inputs:
            run_rec.add_input(inp.path, inp.sha256, inp.data_class)
    rows = alphabet_rows()
    table: list[dict[str, Any]] = []
    per_point: list[dict[str, Any]] = []
    ref: dict[str, dict[str, np.ndarray]] = {}
    for ps in geos:
        ref[ps.name] = eval_row(rows[0], ps)
    for row in rows:
        rec: dict[str, Any] = {
            "row": row["row"],
            "eta_large_n_db": None if row["eta"] is None else db10(row["eta"]),
            "alphabet": None
            if row["gamma"] is None
            else [
                [abs(g), math.degrees(math.atan2(g.imag, g.real))] for g in np.asarray(row["gamma"])
            ],
        }
        if row["row"] == "abeywickrama_b2":
            rec["eta_continuous_phase_db"] = db10(row["eta_continuous_abeywickrama"])
        g = row["gamma"]
        if g is not None and np.asarray(g).size == 2:
            m, dd = binary_md(complex(g[0]), complex(g[1]))
            rec["md_abs_d_db"] = db10(abs(dd) ** 2)
            rec["md_abs_m_db"] = db10(abs(m) ** 2)
            rec["md_arg_m_deg"] = math.degrees(math.atan2(m.imag, m.real))
            for ps in (spec, near):
                c = ps.c[0]
                mirror = abs(m * complex(c.sum())) ** 2
                rec[f"mirror_share_{ps.name}_db"] = db10(mirror / float(np.sum(np.abs(c)) ** 2))
        for ps in geos:
            v = ref[ps.name] if row["kind"] == "continuous" else eval_row(row, ps, n_maps=n_maps)
            r0 = ref[ps.name]
            loss_r = float(np.dot(ps.w, v["ris"]) / np.dot(ps.w, r0["ris"]))
            loss_t = float(np.dot(ps.w, v["tot"]) / np.dot(ps.w, r0["tot"]))
            rec[f"loss_ris_db_{ps.name}"] = db10(loss_r)
            rec[f"loss_total_db_{ps.name}"] = db10(loss_t)
            rec[f"loss_ris_db_mean_{ps.name}"] = float(
                np.dot(ps.w, 10 * np.log10(v["ris"] / r0["ris"]))
            )
            for j in range(len(ps.labels)) if ps.name != "corridor" else ():
                per_point.append(
                    {
                        "geometry": ps.name,
                        "point": ps.labels[j],
                        "row": row["row"],
                        "p_ris": float(v["ris"][j]),
                        "p_ris_ref": float(r0["ris"][j]),
                        "p_tot": float(v["tot"][j]),
                        "p_tot_ref": float(r0["tot"][j]),
                        "loss_ris_db": db10(float(v["ris"][j] / r0["ris"][j])),
                        "loss_total_db": db10(float(v["tot"][j] / r0["tot"][j])),
                    }
                )
        table.append(rec)
        print(
            row["row"],
            {
                k: round(v, 3)
                for k, v in rec.items()
                if k.startswith("loss_ris_db_") and "mean" not in k
            },
            flush=True,
        )
    write_csv(out / "table8.csv", table)
    write_csv(out / "per_point.csv", per_point)
    geo_info = {
        ps.name: {
            "n_points": int(ps.c.shape[0]),
            "n_cells": int(ps.c.shape[1]),
            "direct_over_ris_cont_db": db10(
                float(np.dot(ps.w, np.abs(ps.h0) ** 2) / np.dot(ps.w, ref[ps.name]["ris"]))
            ),
        }
        for ps in geos
    }
    write_json(out / "geometries.json", geo_info)
    write_meta(
        out,
        experiment="E-C",
        data_classes=["SIM-sremu", "ANALYTIC", "ASSUMED", "REPORTED"],
        description="E-C non-ideality ablation (Table VIII / F11): loss vs ideal continuous phase "
        "per geometry, large-N eta, binary m/d decomposition and mirror share",
        assumptions=[
            "lab state amplitudes ASSUMED (|sigma1| = 1; |sigma2| as per row)",
            "Tang RIS1 alphabet: published 27 GHz table values (REPORTED) applied as an "
            "angle-independent alphabet at each geometry's carrier [ENG]",
            "Abeywickrama: beta_min 0.2, alpha 1.6, phi 0, b = 2 [ENG choice of b]",
            "failure rows on the lab |Gamma| = 1 alphabet; the optimizer knows the failed cells; "
            f"{n_maps} keyed maps per point (lab, synthetic), {max(1, n_maps // 20)} per position "
            "(corridor, averaged over the RWP pdf)",
            "loss = power-averaged optimum over the geometry's points / that of the ideal "
            "continuous phase (loss_ris_db_mean_*: mean of per-point dB losses)",
            "lab geometry: V-LAB prior median Tx (2 m, -22.5 deg) and the 30-point polar grid; "
            "SIM geometry, no measured data read",
        ],
        inputs=run_rec.inputs,
        files={
            "table8.csv": "one row per ablation row; loss columns per geometry",
            "per_point.csv": "per point and row (lab, synthetic)",
            "geometries.json": "points, cells and the direct/RIS power ratio per geometry",
        },
    )
    run_rec.finish()
    return {"table": table, "geometries": geo_info}


if __name__ == "__main__":  # pragma: no cover
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "runs" / "wp11" / "ec"
    cache = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "runs" / "wp11" / "_ea_cache"
    run(root, cache)
