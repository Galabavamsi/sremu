"""V-FM / E-D: model-based FM dimensioning (DESIGN.md 10.12 and 11 E-D; replaces ANTS Fig. 6).

Label: **model-based dimensioning, not validated.** 52 km link at 100 MHz, h_tx = 200 m,
h_r in {1.5, 10} m; a RIS of 30 x 30 cells of 0.25 m (A0 = 56.25 m^2, centre 34.9 m above
the ground, ``cell_model`` auto -> point), optimally focused (continuous phase per cell:
``sum_n |c_n|``, :func:`sremu.physics.optimize.continuous_optimum`), d2 in {0.5, 1, 2} km.

* **Direct path (primary):** flat-earth two-ray (``environment.ground``) plus one coherent
  knife edge (``environment.edges``): a ridge (horizontal edge, 400 m long, centred on the
  TX-RX line) 2 km from the RX (0.5 and 5 km: sensitivity), its height set so that the LOS
  ray has nu in [0, 5]; the ground path is diffracted by the same edge (WP3 leg rule).
  The **x axis** of the map is the ridge's obstruction (knife-edge) loss on the LOS ray,
  ``-20 log10 |F(nu)|`` (6.02 ... 26.9 dB). The net change of the two-ray + knife-edge
  direct path against the unobstructed two-ray is reported too; it can be negative
  (obstacle gain: the edge also shields the ground ray that nearly cancels the LOS at
  grazing incidence).
* **Reference row:** free-space LOS + knife edge (no ground), ``FSPL - 20 log10 |F(nu)|``.
* **RIS path:** K1 with the ground image expansion on both hops (hop variants 0-3), and
  without it (``ris_hops_enabled=False``: ground on the direct path only). RIS legs are not
  diffracted in release 1; the geometry keeps them clear of the ridge.
* **Geometries.** ``lateral`` (primary, [ENG]): the RIS d2 beside the village, its normal
  on the horizontal bisector of the directions to TX and RX (bistatic angle about 90 deg,
  so the obliquity costs about 3 dB); ``backreflector`` (sensitivity, [ENG]): the RIS on the
  far side of the village facing the TX (theta_i, theta_s near 0: the T15 cos = 1
  assumption; plausibility of the TX-RIS leg over the ridge not modelled).
* **Ground.** ``land_H``: eps_r = 15, sigma = 1e-3 S/m, H polarization (the P.526 module's
  defaults; P.2040 values UNVERIFIED) for the map; ``pec_H`` (Gamma_H = -1 exactly, library
  API) for the regression rows.
* **Required aperture** (coherent-aperture scaling, far field of the panel, [DERIVED]):
  ``A_req = A0 |h_dir| / |h_RIS(A0)|`` (RIS path power = direct path power) and
  ``A_req,3dB = (sqrt(2) - 1) A_req`` (coherent total +3 dB).
* **Model-form sensitivity row** [ENG]: a closed-form four-ray knife edge over the same
  flat ground (TX and RX ground images, F(nu) per ray; :func:`four_ray_knife_edge`), because
  the two-ray + knife-edge combination is a model choice (WP3 keeps one ground path through
  the unobstructed specular point with the dominant edge crossing of either leg).
* **EXPERIMENTAL sensitivity row:** the direct baseline from ITU-R P.526 smooth earth
  (``physics.experimental.p526``; 143.9 / 127.05 dB) plus the ridge's knife-edge loss
  (losses added in dB [ENG]).
* **V-FM-1 regression** (PEC ground, the T15 geometry: d_TX-RIS = 51953.7 m, RIS at 34.9 m,
  RX at d2 from the RIS near the panel normal): engine RIS-with-ground path loss vs the T15
  closed form ``-20 log10(A/(4 pi d1 d2)) - 20 log10 F1 - 20 log10 F2``,
  ``F = |1 - exp(-jk 2 h_a h_b / d)|`` (recomputed here and read from
  ``tests/oracles/data/physics_model_checks.json``), within +-0.1 dB at d2 = 1, 2 km; plus
  the two-ray direct path (PH-10: 139.10 / 122.63 dB) and the free-space focused RIS path
  (135.27 / 141.29 / 147.31 dB).

Usage::

    python validation/fm/vfm.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vstudy as vs  # noqa: E402

from sremu.units import C0  # noqa: E402

STUDY_DIR = _HERE
FC = 100e6
LAM = C0 / FC
K = 2 * math.pi / LAM
H_TX = 200.0
D_LINK = 52_000.0
H_RIS = 34.9
N_SIDE = 30
PITCH = 0.25
A0 = (N_SIDE * PITCH) ** 2
D2S = (500.0, 1000.0, 2000.0)
D2S_REG = (500.0, 1000.0, 2000.0, 3000.0)
HRS = (1.5, 10.0)
NUS = tuple(float(x) for x in np.round(np.arange(0.0, 5.0001, 0.1), 2))
S_RIDGE = 2000.0  # ridge distance from the RX, m [ENG]
S_RIDGES = (500.0, 2000.0, 5000.0)  # primary 2 km and sensitivity
RIDGE_HALF = 200.0  # half length of the ridge edge, m [ENG]
D_TX_RIS_T15 = 51953.7  # T15 geometry (pm_checks.py), m
LAND = {"eps_r": 15.0, "sigma_s_per_m": 1e-3, "polarization": "H", "height_m": 0.0}
PMC = vs.REPO / "tests" / "oracles" / "data" / "physics_model_checks.json"
TX = "tx.tx0"
RX = "rx.rx0"
LINK = f"{TX}->{RX}"
PANEL = "ris0"


# =========================================================================== geometry


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def ris_pose(
    geometry: str, d2: float, h_r: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(TX, RX, RIS centre, RIS normal) of a geometry (RIS-RX distance exactly d2)."""
    dz = H_RIS - h_r
    horiz = math.sqrt(d2 * d2 - dz * dz)
    tx = np.array([0.0, 0.0, H_TX])
    if geometry == "lateral":
        rx = np.array([0.0, D_LINK, h_r])
        c = np.array([horiz, D_LINK, H_RIS])
        ut = _unit(np.array([*(tx - c)[:2], 0.0]))
        ur = _unit(np.array([*(rx - c)[:2], 0.0]))
        n = _unit(ut + ur)
    elif geometry == "backreflector":
        rx = np.array([0.0, D_LINK, h_r])
        c = np.array([0.0, D_LINK + horiz, H_RIS])
        n = np.array([0.0, -1.0, 0.0])
    elif geometry == "t15":  # V-FM-1 regression: d_TX-RIS = 51953.7 m, RX toward the TX
        y_r = math.sqrt(D_TX_RIS_T15**2 - (H_TX - H_RIS) ** 2)
        c = np.array([0.0, y_r, H_RIS])
        rx = np.array([0.0, y_r - horiz, h_r])
        n = np.array([0.0, -1.0, 0.0])
    else:
        raise ValueError(geometry)
    return tx, rx, c, n


def ridge_edge(h_r: float, nu: float, s_ridge: float = 2000.0) -> tuple[dict[str, Any], float]:
    """Edge spec of the ridge ``s_ridge`` from the RX giving the LOS ray ``nu``, and the
    ridge-top height (m). Solved on the engine's own crossing geometry (KnifeEdge)."""
    from sremu.physics.diffraction import KnifeEdge

    tx = np.array([0.0, 0.0, H_TX])
    rx = np.array([0.0, D_LINK, h_r])
    y = D_LINK - s_ridge
    z_los = h_r + (H_TX - h_r) * s_ridge / D_LINK
    d1, d2 = D_LINK - s_ridge, s_ridge
    h_need = nu / math.sqrt(2 * (d1 + d2) / (LAM * d1 * d2))

    def edge(z: float) -> Any:
        return KnifeEdge("ridge", np.array([-RIDGE_HALF, y, z]), np.array([RIDGE_HALF, y, z]))

    z = z_los + h_need
    for _ in range(4):  # common-perpendicular correction (inclined LOS)
        cr = edge(z).crossing(tx, rx)
        z += h_need - cr.h_m
    spec = {"id": "ridge", "p0_m": [-RIDGE_HALF, y, z], "p1_m": [RIDGE_HALF, y, z]}
    return spec, z


def scenario(
    geometry: str,
    d2: float,
    h_r: float,
    *,
    ground: bool = True,
    edge: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Section 9.2 scenario mapping of one configuration (isotropic antennas)."""
    tx, rx, c, n = ris_pose(geometry, d2, h_r)
    u = _unit(np.cross(np.array([0.0, 0.0, 1.0]), n))
    env: dict[str, Any] = {}
    if ground:
        env["ground"] = dict(LAND)
    if edge is not None:
        env["edges"] = [dict(edge)]

    def node(nid: str, p: np.ndarray, direction: str) -> dict[str, Any]:
        s: dict[str, Any] = {
            "name": f"{direction}0",
            "direction": direction,
            "fc_hz": FC,
            "rate_sps": "1e6",
        }
        if direction == "tx":
            s["occupied_bw_hz"] = 2e5
        return {
            "id": nid,
            "position_m": p.tolist(),
            "antenna": {"pattern": "isotropic"},
            "streams": [s],
        }

    return {
        "schema_version": "1.0",
        "name": f"vfm_{geometry}",
        "seed": 1,
        "description": f"V-FM {geometry}: d2 = {d2} m, h_r = {h_r} m",
        "environment": env,
        "nodes": [node("tx", tx, "tx"), node("rx", rx, "rx")],
        "ris": [
            {
                "id": PANEL,
                "pose": {"center_m": c.tolist(), "normal": n.tolist(), "u_axis": u.tolist()},
                "grid": {
                    "n_u": N_SIDE,
                    "n_v": N_SIDE,
                    "pitch_u_m": PITCH,
                    "pitch_v_m": PITCH,
                    "cell_model": "auto",
                },
                "states": {"library": "ideal_antipodal"},
                "initial": {"fill": 0},
            }
        ],
        "run": {"mode": "lockstep", "coupling": "barrier", "duration_s": "1/1000"},
    }


def build(scen: Mapping[str, Any], *, ground: str | None, ris_hops: bool = True) -> Any:
    """Model with WP3 mechanisms: ``ground`` None (no ground), ``"land_H"`` (scenario) or
    ``"pec_H"`` (Gamma_H = -1, library API); ``ris_hops=False`` removes the image expansion
    on the RIS hops (direct-path ground kept)."""
    from sremu.physics.ground import GroundMechanism, GroundPlane
    from sremu.physics.propagation import from_scenario, wp3_mechanisms

    d = vs.resolve_mapping(scen).scenario.model_dump(mode="python")
    mechs = list(wp3_mechanisms(d))
    out = []
    for m in mechs:
        if isinstance(m, GroundMechanism):
            plane = GroundPlane(math.inf, math.inf, "H") if ground == "pec_H" else m.plane
            m = GroundMechanism(plane, m.occluders, ris_hops, edges=m.edges)
        out.append(m)
    return from_scenario(d, mechanisms=out) if out else from_scenario(d)


def channel(model: Any) -> dict[str, Any]:
    """Direct coefficient (with carrier phases) and focused RIS amplitude ``sum |c_n|``."""
    from sremu.physics.optimize import continuous_optimum

    snap = model.snapshot(Fraction(0))
    v = model.csi_view(snap, LINK)
    c = v.cascade[PANEL][0]  # state 0 = +1 (ideal antipodal): Gamma-free cascade
    ris_abs = float(np.sum(np.abs(c)))
    total, _ = continuous_optimum(v.h_direct, c)
    return {
        "h_dir": complex(v.h_direct),
        "ris_abs": ris_abs,
        "focused_total_abs": total,
        "n_cells": int(c.size),
        "cell_model": model.cell_model(PANEL),
    }


def pl(x: float) -> float:
    """Path loss (dB) of an amplitude."""
    return -20 * math.log10(x)


def t15_closed_form(d2: float, h_r: float) -> dict[str, float]:
    """The T15 focused-plate closed form (cos = 1) with flat-ground factors per hop."""
    fs = -20 * math.log10(A0 / (4 * math.pi * D_TX_RIS_T15 * d2))
    f1 = abs(1 - np.exp(-1j * K * 2 * H_TX * H_RIS / D_TX_RIS_T15))
    f2 = abs(1 - np.exp(-1j * K * 2 * H_RIS * h_r / d2))
    return {"free_space_db": fs, "with_ground_db": fs - 20 * math.log10(f1) - 20 * math.log10(f2)}


def four_ray_knife_edge(h_r: float, z_edge: float, s_ridge: float, *, detail: bool = False) -> Any:
    """Four-ray knife-edge direct path, an [ENG] model-form **sensitivity** (not sremu's
    model): sources TX and its ground image, sinks RX and its image; each of the four rays
    s -> r over the ridge gets ``F(nu_sr)`` (nu from the ridge top's height above the
    straight s-r line at the ridge, d1/d2 along that line), the land-H Fresnel coefficient
    of each reflection (TX side at the grazing angle TX -> ridge top, RX side at ridge top
    -> RX) and ``lambda/(4 pi d_sr) exp(-jk d_sr)``. sremu's WP3 rule instead keeps one
    ground path through the unobstructed specular point with the dominant edge crossing of
    either leg, so it has no TX-side reflection of the diffracted field. ``detail=True``
    returns ``(total, rays)`` with each ray's (source, sink, nu, coefficient)."""
    from sremu.physics.diffraction import knife_edge_F
    from sremu.physics.ground import complex_permittivity, fresnel_gamma_h

    eps = complex_permittivity(LAND["eps_r"], LAND["sigma_s_per_m"], FC)
    y_e = D_LINK - s_ridge
    g_tx = complex(fresnel_gamma_h(math.atan((H_TX + z_edge) / y_e), eps))
    g_rx = complex(fresnel_gamma_h(math.atan((z_edge + h_r) / s_ridge), eps))
    tot = 0j
    rays = []
    for src, s_z, gs in (("tx", H_TX, 1.0 + 0j), ("tx_image", -H_TX, g_tx)):
        for snk, r_z, gr in (("rx", h_r, 1.0 + 0j), ("rx_image", -h_r, g_rx)):
            d = math.hypot(D_LINK, r_z - s_z)
            z_line = s_z + (r_z - s_z) * y_e / D_LINK
            d1 = math.hypot(y_e, z_line - s_z)
            d2 = d - d1
            nu = (z_edge - z_line) * math.sqrt(2.0 * d / (LAM * d1 * d2))
            amp = LAM / (4 * math.pi * d) * complex(np.exp(-1j * K * d))
            c = gs * gr * amp * complex(knife_edge_F(nu))
            rays.append((src, snk, nu, c))
            tot += c
    return (tot, rays) if detail else tot


def obliquity_db(geometry: str, d2: float, h_r: float) -> float:
    """-20 log10 sqrt(cos_i cos_s) at the panel centre (dB loss)."""
    tx, rx, c, n = ris_pose(geometry, d2, h_r)
    ci = float(n @ _unit(tx - c))
    cs = float(n @ _unit(rx - c))
    return -10 * math.log10(ci * cs)


# =========================================================================== study


def study(root: Path | None = None) -> Path:
    from sremu.physics.diffraction import KnifeEdge, knife_edge_loss_db
    from sremu.physics.experimental.p526 import p526_smooth

    pmc = json.loads(PMC.read_bytes())
    base = scenario("lateral", 1000.0, 1.5)
    run = vs.StudyRun(
        STUDY_DIR,
        "vfm",
        base,
        root=root,
        inputs=[(PMC, "ANALYTIC")],
        assumptions=(
            "model-based dimensioning, not validated (DESIGN 10.12)",
            "lateral RIS placement with the normal on the horizontal bisector [ENG]",
            "ridge: horizontal knife edge 400 m long, 2 km from the RX "
            "(0.5, 5 km sensitivity) [ENG]",
            "ground eps_r = 15, sigma = 1e-3 S/m, H pol (P.526 module defaults) [ASSUMED]",
            "A_req by coherent-aperture scaling of the focused 56.25 m^2 panel [DERIVED]",
            "RIS legs are not diffracted (release 1); the geometry keeps them clear of the ridge",
            "x axis: knife-edge loss of the ridge on the LOS ray, -20 log10 |F(nu)| (exact F)",
            "four-ray knife-edge direct path (TX/RX ground images, F(nu) per ray): [ENG] "
            "model-form sensitivity row, not sremu's model",
        ),
    )
    t0 = time.perf_counter()
    # ---------------- direct path vs nu (primary ridge distance and sensitivity)
    direct = []
    base_pl: dict[float, dict[str, float]] = {}
    for h_r in HRS:
        tx, rx, _c, _n = ris_pose("lateral", 1000.0, h_r)
        h0 = channel(build(scenario("lateral", 1000.0, h_r), ground="land_H"))["h_dir"]
        hp = channel(build(scenario("lateral", 1000.0, h_r), ground="pec_H"))["h_dir"]
        p526, branch = p526_smooth(D_LINK / 1000, H_TX, h_r, FC / 1e6)
        fspl = pl(LAM / (4 * math.pi * float(np.linalg.norm(rx - tx))))
        base_pl[h_r] = {
            "two_ray_land": pl(abs(h0)),
            "two_ray_pec": pl(abs(hp)),
            "p526": fspl + p526,
            "p526_excess": p526,
            "fspl": fspl,
        }
        for s_ridge in S_RIDGES:
            for nu in NUS:
                spec, z = ridge_edge(h_r, nu, s_ridge)
                m = build(scenario("lateral", 1000.0, h_r, edge=spec), ground="land_H")
                e = KnifeEdge("r", np.array(spec["p0_m"]), np.array(spec["p1_m"]))
                nu_eng = e.crossing(tx, rx).nu(FC)
                hd = channel(m)["h_dir"]
                j = float(knife_edge_loss_db(nu_eng))
                h4 = four_ray_knife_edge(h_r, z, s_ridge)
                direct.append(
                    {
                        "h_r_m": h_r,
                        "ridge_distance_m": s_ridge,
                        "nu_target": nu,
                        "nu_engine_los": nu_eng,
                        "ridge_top_m": z,
                        "knife_edge_loss_db": j,
                        "pl_two_ray_knife_edge_db": pl(abs(hd)),
                        "pl_two_ray_unobstructed_db": pl(abs(h0)),
                        "net_change_vs_two_ray_db": pl(abs(hd)) - pl(abs(h0)),
                        "pl_four_ray_knife_edge_sensitivity_db": pl(abs(h4)),
                        "four_ray_minus_engine_db": pl(abs(h4)) - pl(abs(hd)),
                        "pl_free_space_knife_edge_db": fspl + j,
                        "pl_p526_knife_edge_db": fspl + p526 + j,
                        "p526_excess_db": p526,
                        "p526_branch": branch,
                        "fspl_db": fspl,
                    }
                )
    vs.write_csv(run.path("vfm_direct.csv"), direct)
    # ---------------- RIS paths
    ris = []
    for geometry in ("lateral", "backreflector"):
        for d2 in D2S:
            for h_r in HRS:
                for hops in (True, False):
                    m = build(scenario(geometry, d2, h_r), ground="land_H", ris_hops=hops)
                    ch = channel(m)
                    tx, rx, c, _n = ris_pose(geometry, d2, h_r)
                    ris.append(
                        {
                            "geometry": geometry,
                            "d2_m": d2,
                            "h_r_m": h_r,
                            "ground_on_ris_hops": hops,
                            "pl_ris_focused_db": pl(ch["ris_abs"]),
                            "d_tx_ris_m": float(np.linalg.norm(c - tx)),
                            "d_ris_rx_m": float(np.linalg.norm(rx - c)),
                            "obliquity_loss_db_centre": obliquity_db(geometry, d2, h_r),
                            "cell_model": ch["cell_model"],
                            "n_cells": ch["n_cells"],
                        }
                    )
    vs.write_csv(run.path("vfm_ris.csv"), ris)
    # ---------------- E-D map (primary ridge distance)
    rows = []
    for r in ris:
        for dr in direct:
            if dr["h_r_m"] != r["h_r_m"] or dr["ridge_distance_m"] != S_RIDGE:
                continue
            for baseline, pl_base in (
                ("two_ray_knife_edge", dr["pl_two_ray_knife_edge_db"]),
                ("p526_plus_knife_edge_experimental", dr["pl_p526_knife_edge_db"]),
                ("four_ray_knife_edge_sensitivity", dr["pl_four_ray_knife_edge_sensitivity_db"]),
                ("free_space_knife_edge_reference", dr["pl_free_space_knife_edge_db"]),
            ):
                a_req = A0 * 10 ** ((r["pl_ris_focused_db"] - pl_base) / 20)
                rows.append(
                    {
                        "geometry": r["geometry"],
                        "d2_m": r["d2_m"],
                        "h_r_m": r["h_r_m"],
                        "ground_on_ris_hops": r["ground_on_ris_hops"],
                        "baseline": baseline,
                        "nu": dr["nu_target"],
                        "knife_edge_loss_db": dr["knife_edge_loss_db"],
                        "net_change_vs_two_ray_db": dr["net_change_vs_two_ray_db"],
                        "pl_direct_db": pl_base,
                        "pl_ris_a0_db": r["pl_ris_focused_db"],
                        "ris_minus_direct_at_a0_db": pl_base - r["pl_ris_focused_db"],
                        "a_req_equal_power_m2": a_req,
                        "a_req_plus3db_m2": (math.sqrt(2) - 1) * a_req,
                        "a0_m2": A0,
                        "a0_sufficient_equal_power": bool(a_req <= A0),
                    }
                )
    vs.write_csv(run.path("vfm_ed_map.csv"), rows)
    # ---------------- V-FM-1 regression (PEC, T15 geometry)
    t15 = {r["d2_m"]: r for r in pmc["T15_FM_RIS_path_loss"]}
    reg = []
    for d2 in D2S_REG:
        for h_r in HRS:
            eng_g = pl(channel(build(scenario("t15", d2, h_r), ground="pec_H"))["ris_abs"])
            eng_f = pl(
                channel(build(scenario("t15", d2, h_r, ground=False), ground=None))["ris_abs"]
            )
            cf = t15_closed_form(d2, h_r)
            rec = t15[d2]
            rec_g = rec["with_flat_ground_factors_dB"][f"hr{h_r}"]
            rec_f = rec["RIS_free_space_PL_dB(upper bound, focused, cos=1)"]
            reg.append(
                {
                    "d2_m": d2,
                    "h_r_m": h_r,
                    "engine_with_ground_db": eng_g,
                    "t15_recorded_with_ground_db": rec_g,
                    "t15_recomputed_with_ground_db": cf["with_ground_db"],
                    "delta_with_ground_db": eng_g - rec_g,
                    "delta_vs_recomputed_db": eng_g - cf["with_ground_db"],
                    "engine_free_space_db": eng_f,
                    "t15_recorded_free_space_db": rec_f,
                    "delta_free_space_db": eng_f - rec_f,
                    "in_vfm1_range": d2 in (1000.0, 2000.0),
                    "pass_0p1": bool(abs(eng_g - rec_g) <= 0.1),
                    "obliquity_loss_db_centre": obliquity_db("t15", d2, h_r),
                }
            )
    vs.write_csv(run.path("vfm1_regression.csv"), reg)
    run.time("all", time.perf_counter() - t0)
    # ---------------- summary
    acc = {
        "V-FM-1_ris_with_ground_pm_0.1dB_d2_1_2km": all(
            r["pass_0p1"] for r in reg if r["in_vfm1_range"]
        ),
        "V-FM-1_all_d2_pm_0.1dB": all(r["pass_0p1"] for r in reg),
        "PH-10_two_ray_pec_139.10_122.63_pm_0.01dB": bool(
            abs(base_pl[1.5]["two_ray_pec"] - 139.10) <= 0.01
            and abs(base_pl[10.0]["two_ray_pec"] - 122.63) <= 0.01
        ),
    }

    def a_req(geom: str, hops: bool, base: str, d2: float, h_r: float, nu: float) -> float:
        return next(
            r["a_req_equal_power_m2"]
            for r in rows
            if r["geometry"] == geom
            and r["ground_on_ris_hops"] == hops
            and r["baseline"] == base
            and r["d2_m"] == d2
            and r["h_r_m"] == h_r
            and r["nu"] == nu
        )

    def net(h_r: float, s: float, nu: float) -> float:
        return next(
            d["net_change_vs_two_ray_db"]
            for d in direct
            if d["h_r_m"] == h_r and d["ridge_distance_m"] == s and d["nu_target"] == nu
        )

    def four_minus_engine(h_r: float) -> list[float]:
        v = [
            d["four_ray_minus_engine_db"]
            for d in direct
            if d["h_r_m"] == h_r and d["ridge_distance_m"] == S_RIDGE
        ]
        return [min(v), max(v)]

    numbers = {
        "a0_m2": A0,
        "four_ray_minus_engine_db_range_primary_ridge": {
            f"{h:g}": four_minus_engine(h) for h in HRS
        },
        "direct_baselines_db": {f"{h:g}": v for h, v in base_pl.items()},
        "net_change_vs_two_ray_db": {
            f"hr={h:g},ridge={s:g},nu={nu:g}": net(h, s, nu)
            for h in HRS
            for s in S_RIDGES
            for nu in (0.0, 1.0, 2.0, 5.0)
        },
        "a_req_m2_lateral_ground": {
            f"{base},d2={d2:g},hr={h:g},nu={nu:g}": a_req("lateral", True, base, d2, h, nu)
            for base in (
                "two_ray_knife_edge",
                "p526_plus_knife_edge_experimental",
                "four_ray_knife_edge_sensitivity",
            )
            for d2 in D2S
            for h in HRS
            for nu in (0.0, 2.0, 5.0)
        },
        "ris_minus_direct_db_at_a0_nu0": {
            f"{r['geometry']},hops={r['ground_on_ris_hops']},{r['baseline']},d2={r['d2_m']:g},"
            f"hr={r['h_r_m']:g}": r["ris_minus_direct_at_a0_db"]
            for r in rows
            if r["nu"] == 0.0
        },
        "vfm1": reg,
    }
    meta = {
        "study": "V-FM / E-D",
        "label": "model-based dimensioning, not validated",
        "data_classes": ["SIM-sremu", "ANALYTIC", "ASSUMED"],
        "experimental_rows": "baseline == p526_plus_knife_edge_experimental (ITU-R P.526 "
        "port, EXPERIMENTAL; not validated against ITU-R data)",
        "sensitivity_rows": "baseline == four_ray_knife_edge_sensitivity ([ENG] closed-form "
        "four-ray knife edge over flat ground; model-form sensitivity of the direct path)",
        "acceptance": acc,
        "numbers": numbers,
        "figure": "F12: vfm_ed_map.csv (a_req_equal_power_m2 vs knife_edge_loss_db)",
    }
    path = run.finish(meta)
    print(f"V-FM run directory: {run.run_dir}\nmeta: {path}")
    print(json.dumps(acc, indent=1))
    if root is None:
        print(f"report: {vs.write_report(STUDY_DIR, report(run.run_dir))}")
    return run.run_dir


# =========================================================================== report


def _csv(run_dir: Path, name: str) -> list[dict[str, str]]:
    import csv

    with (run_dir / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def report(run_dir: Path) -> str:
    """``validation/fm/report.md`` rendered from a run directory (meta.json + CSV tables)."""
    run_dir = Path(run_dir)
    meta = vs.read_meta(run_dir)
    f = vs.fnum
    acc = meta["acceptance"]
    num = meta["numbers"]
    reg = num["vfm1"]
    ris = _csv(run_dir, "vfm_ris.csv")
    edm = _csv(run_dir, "vfm_ed_map.csv")
    direct = _csv(run_dir, "vfm_direct.csv")
    base = num["direct_baselines_db"]
    s = []
    s.append("# V-FM / E-D: FM RIS dimensioning (DESIGN.md 10.12 and 11 E-D)\n")
    s.append(
        "**Label: model-based dimensioning, not validated.** All numbers are model outputs "
        "(two-ray flat earth, coherent knife edge, K1 RIS kernel with ground image expansion); "
        "no measurement enters. The ITU-R P.526 rows come from sremu's experimental port "
        "(`physics/experimental/p526.py`), which is not validated against ITU-R data and sits "
        "outside every gate: they are an **EXPERIMENTAL sensitivity row** only. Data classes: "
        f"{', '.join(meta.get('data_classes', []))}.\n"
    )
    s.append("## Provenance\n")
    s.append(vs.run_header(meta) + "\n")
    s.append(
        "* Assumptions recorded in the manifest: " + "; ".join(meta.get("assumptions", [])) + ".\n"
    )
    s.append("## Setup\n")
    s.append(
        f"52 km link at 100 MHz, h_tx = {H_TX:g} m, h_r in {{1.5, 10}} m; RIS of {N_SIDE} x "
        f"{N_SIDE} cells of {PITCH:g} m (A0 = {A0:g} m^2) centred {H_RIS:g} m above the ground, "
        "`cell_model` auto (point), focused with continuous per-cell phase (sum_n |c_n|), "
        "d2 (RIS-RX) in {0.5, 1, 2} km. Direct path: two-ray flat earth (ground eps_r = 15, "
        "sigma = 1e-3 S/m, H polarization [ASSUMED]) plus one coherent knife edge (a horizontal "
        "400 m ridge 2 km from the RX; 0.5 and 5 km as sensitivity) whose height gives the LOS "
        "ray nu in [0, 5]; the ground ray is diffracted by the same edge. x axis of the map: "
        "the ridge's knife-edge loss on the LOS ray, -20 log10 |F(nu)|. RIS legs are not "
        "diffracted (release 1); the geometry keeps them clear of the ridge. Geometries: "
        "`lateral` (primary [ENG]: the RIS beside the village, normal on the bisector, about "
        "3 dB obliquity) and `backreflector` (sensitivity [ENG]: behind the village facing the "
        "TX, theta near 0). Required aperture by coherent-aperture scaling [DERIVED]: "
        "A_req = A0 |h_dir| / |h_RIS(A0)| (RIS path equal to the direct path), and "
        "(sqrt(2) - 1) A_req for a +3 dB coherent total.\n"
    )
    s.append("## Acceptance (regression)\n")
    rows = [
        (
            "PH-10: two-ray (PEC, H) path loss 139.10 / 122.63 dB +- 0.01 dB",
            f"{f(base['1.5']['two_ray_pec'], 3)} / {f(base['10']['two_ray_pec'], 3)} dB",
            vs.passfail(acc["PH-10_two_ray_pec_139.10_122.63_pm_0.01dB"]),
        ),
        (
            "V-FM-1: RIS-with-ground path loss vs the T15 values, d2 = 1-2 km, +- 0.1 dB",
            "max |d| "
            + f(
                max(abs(r["delta_with_ground_db"]) for r in reg if r["in_vfm1_range"]),
                3,
            )
            + " dB",
            vs.passfail(acc["V-FM-1_ris_with_ground_pm_0.1dB_d2_1_2km"]),
        ),
        (
            "V-FM-1 extended to d2 = 0.5 and 3 km (+- 0.1 dB; not in the DESIGN row)",
            "max |d| " + f(max(abs(r["delta_with_ground_db"]) for r in reg), 3) + " dB",
            vs.passfail(acc["V-FM-1_all_d2_pm_0.1dB"]),
        ),
    ]
    s.append(vs.md_table(("Check", "Measured", "Result"), rows) + "\n")
    s.append(
        "V-FM-1 geometry (T15 of `pm_checks.py`): PEC ground (Gamma_H = -1), d_TX-RIS = "
        f"{D_TX_RIS_T15:g} m, RIS at {H_RIS:g} m, RX at d2 near the panel normal; reference = "
        "the recorded T15 values (`tests/oracles/data/physics_model_checks.json`) and the "
        "closed form recomputed here, -20 log10(A/(4 pi d1 d2)) - 20 log10 F1 - 20 log10 F2 "
        "with F = |1 - exp(-jk 2 h_a h_b / d)|.\n"
    )
    s.append(
        vs.md_table(
            (
                "d2 (m)",
                "h_r (m)",
                "engine with ground (dB)",
                "T15 recorded (dB)",
                "d (dB)",
                "d vs recomputed (dB)",
                "engine free space (dB)",
                "T15 free space (dB)",
                "result",
            ),
            (
                (
                    f(r["d2_m"], 0),
                    f(r["h_r_m"], 1),
                    f(r["engine_with_ground_db"], 3),
                    f(r["t15_recorded_with_ground_db"], 2),
                    f(r["delta_with_ground_db"], 3, sign=True),
                    f(r["delta_vs_recomputed_db"], 3, sign=True),
                    f(r["engine_free_space_db"], 3),
                    f(r["t15_recorded_free_space_db"], 2),
                    vs.passfail(r["pass_0p1"]),
                )
                for r in reg
            ),
        )
        + "\n"
    )
    s.append("## Direct-path baselines (no ridge)\n")
    s.append(
        vs.md_table(
            (
                "h_r (m)",
                "FSPL (dB)",
                "two-ray, land H (dB)",
                "two-ray, PEC H (dB)",
                "P.526 smooth earth, EXPERIMENTAL (dB)",
                "P.526 excess (dB)",
            ),
            (
                (
                    h,
                    f(b["fspl"], 3),
                    f(b["two_ray_land"], 3),
                    f(b["two_ray_pec"], 3),
                    f(b["p526"], 3),
                    f(b["p526_excess"], 3),
                )
                for h, b in base.items()
            ),
        )
        + "\n"
    )
    s.append("## RIS path at A0 = 56.25 m^2 (focused)\n")
    s.append(
        vs.md_table(
            (
                "geometry",
                "d2 (m)",
                "h_r (m)",
                "ground on RIS hops",
                "RIS path loss (dB)",
                "obliquity at centre (dB)",
            ),
            (
                (
                    r["geometry"],
                    f(float(r["d2_m"]), 0),
                    f(float(r["h_r_m"]), 1),
                    r["ground_on_ris_hops"],
                    f(float(r["pl_ris_focused_db"]), 3),
                    f(float(r["obliquity_loss_db_centre"]), 3),
                )
                for r in ris
            ),
        )
        + "\n"
    )

    def gap(geom: str, hops: str, baseline: str, d2s: tuple[float, ...]) -> tuple[float, float]:
        v = [
            -float(r["ris_minus_direct_at_a0_db"])
            for r in edm
            if r["geometry"] == geom
            and r["ground_on_ris_hops"] == hops
            and r["baseline"] == baseline
            and float(r["d2_m"]) in d2s
            and float(r["nu"]) == 0.0
        ]
        return min(v), max(v)

    g_t15 = [
        r["engine_with_ground_db"] - base[f"{r['h_r_m']:g}"]["p526"]
        for r in reg
        if r["d2_m"] in (1000.0, 2000.0)
    ]
    s.append("## E-D: required aperture vs obstruction (figure F12 data: `vfm_ed_map.csv`)\n")
    s.append(
        "Without obstruction (nu = 0 on the 2 km ridge, i.e. a grazing edge with 6.02 dB "
        "knife-edge loss on the LOS ray) and with ground on both RIS hops, the focused 56.25 m^2 "
        "RIS path is weaker than the direct path by (min to max over d2 = 1-2 km, both h_r):\n"
    )
    rows = []
    for geom in ("lateral", "backreflector"):
        for baseline, lab in (
            ("two_ray_knife_edge", "two-ray + knife edge (primary)"),
            ("four_ray_knife_edge_sensitivity", "four-ray knife edge ([ENG] sensitivity)"),
            ("p526_plus_knife_edge_experimental", "P.526 + knife edge (EXPERIMENTAL)"),
        ):
            lo, hi = gap(geom, "true", baseline, (1000.0, 2000.0))
            rows.append((geom, lab, f"{f(lo, 1)} to {f(hi, 1)} dB"))
    s.append(vs.md_table(("geometry", "direct-path model", "direct minus RIS"), rows) + "\n")
    s.append(
        "Against the unobstructed P.526 smooth-earth direct path (no ridge), the T15 "
        f"RIS-with-ground values are {f(min(g_t15), 1)} to {f(max(g_t15), 1)} dB weaker at "
        "d2 = 1-2 km (EXPERIMENTAL baseline).\n"
    )
    nus = (0.0, 1.0, 2.0, 3.0, 5.0)
    s.append(
        "Required aperture A_req (m^2) for RIS path = direct path, lateral geometry, ridge 2 km "
        "from the RX; columns are the ridge's knife-edge loss on the LOS ray (nu):\n"
    )
    jl = {
        float(r["nu"]): float(r["knife_edge_loss_db"])
        for r in edm
        if r["geometry"] == "lateral" and r["d2_m"] == "1000.0" and r["h_r_m"] == "1.5"
    }
    hdr = ["direct-path model", "ground on RIS hops", "d2 (m)", "h_r (m)"] + [
        f"nu={n:g} ({f(jl.get(n), 1)} dB)" for n in nus
    ]
    rows = []
    for baseline, lab in (
        ("two_ray_knife_edge", "two-ray + KE"),
        ("four_ray_knife_edge_sensitivity", "four-ray KE ([ENG] sensitivity)"),
        ("p526_plus_knife_edge_experimental", "P.526 + KE (EXPERIMENTAL)"),
        ("free_space_knife_edge_reference", "free space + KE (reference)"),
    ):
        for hops in ("true", "false"):
            for d2 in D2S:
                for h in HRS:
                    vals = {
                        float(r["nu"]): float(r["a_req_equal_power_m2"])
                        for r in edm
                        if r["geometry"] == "lateral"
                        and r["baseline"] == baseline
                        and r["ground_on_ris_hops"] == hops
                        and float(r["d2_m"]) == d2
                        and float(r["h_r_m"]) == h
                    }
                    rows.append([lab, hops, f(d2, 0), f(h, 1)] + [f(vals.get(n), 1) for n in nus])
    s.append(vs.md_table(hdr, rows) + "\n")
    # count per direct-path model: a pooled count over baselines would mix the primary model
    # with the EXPERIMENTAL and sensitivity rows (reviewer fix, WP10 review)
    suff = []
    for baseline, lab in (
        ("two_ray_knife_edge", "two-ray + KE (primary)"),
        ("free_space_knife_edge_reference", "free space + KE (reference)"),
        ("four_ray_knife_edge_sensitivity", "four-ray KE ([ENG] sensitivity)"),
        ("p526_plus_knife_edge_experimental", "P.526 + KE (EXPERIMENTAL)"),
    ):
        sel = [r for r in edm if r["baseline"] == baseline]
        k = sum(1 for r in sel if r["a0_sufficient_equal_power"] == "true")
        suff.append(f"{lab}: {k} of {len(sel)}")
    s.append(
        f"A0 = {A0:g} m^2 suffices (A_req <= A0) in these map rows per direct-path model (all "
        "geometries, ground options, d2, h_r and nu in 0 ... 5): " + "; ".join(suff) + ". "
        "A count pooled over the models is not meaningful.\n"
    )
    net = num["net_change_vs_two_ray_db"]
    s.append(
        "The ridge also shields the ground ray that nearly cancels the LOS at grazing "
        "incidence, so the two-ray + knife-edge direct path can be *stronger* than the "
        "unobstructed two-ray (negative net change, dB):\n"
    )
    s.append(
        vs.md_table(
            ("h_r (m)", "ridge from RX (m)", "nu = 0", "nu = 1", "nu = 2", "nu = 5"),
            (
                (
                    f"{h:g}",
                    f"{rd:g}",
                    *(
                        f(net[f"hr={h:g},ridge={rd:g},nu={n:g}"], 2, sign=True)
                        for n in (0, 1, 2, 5)
                    ),
                )
                for h in HRS
                for rd in S_RIDGES
            ),
        )
        + "\n"
    )
    fr = num.get("four_ray_minus_engine_db_range_primary_ridge") or {}
    if fr:
        s.append(
            "Model-form sensitivity of the direct path: the [ENG] four-ray knife edge minus "
            "sremu's two-ray + knife edge, ridge 2 km from the RX, nu in 0 ... 5: "
            + "; ".join(
                f"h_r = {h} m: {f(v[0], 1, sign=True)} to {f(v[1], 1, sign=True)} dB"
                for h, v in fr.items()
            )
            + ". A_req scales with the direct-path amplitude, so this spread carries straight "
            "into the map (factor 10^(spread/20)); the map is a dimensioning aid under a "
            "stated model, not a prediction.\n"
        )
    s.append(
        "Per-nu rows of the direct path (engine nu, ridge height, knife-edge loss, two-ray + "
        "KE, four-ray KE, free space + KE, P.526 + KE) are in `vfm_direct.csv` "
        f"({len(direct)} rows).\n"
    )
    s.append("## Files and run time\n")
    s.append(
        "`vfm_direct.csv`, `vfm_ris.csv`, `vfm_ed_map.csv` (F12), `vfm1_regression.csv`, "
        f"`meta.json`, `manifest.json`. Wall time of all evaluations "
        f"{f((meta.get('timings_s') or {}).get('all'), 1)} s.\n"
    )
    return "\n".join(s)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V-FM / E-D study (DESIGN.md 10.12, 11)")
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
