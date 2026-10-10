"""Geometry prior of the V-LAB protocol: keyed draws, lab hypotheses and sremu scenarios.

DESIGN.md section 10.9 ("Geometry prior"); every number comes from ``protocol.json``
(``prior``, ``frame``). A draw is a pure function of ``(seed, purpose, name, counter)``
through :func:`sremu.rng.generator` (D14); infeasible candidates (a node within one
wavelength of the panel rectangle, or a Tx-point segment crossing the panel) are rejected and
redrawn from the same stream (``prior.feasibility``).

Frames (``protocol.json`` ``frame``): z is up, the ground (M3 only) is ``z = 0``, the Tx,
the RIS centre and the measurement plane are at the antenna height ``h``. Measurement point
``i`` is ``(r_i cos theta_i + dx_i, r_i sin theta_i + dy_i, h + dz)`` around the polar
origin ``(0, 0)``:

* ``O_R/normal``: RIS centre at the origin, normal ``+x``, Tx ``(d cos a, d sin a, h)``;
* ``O_R/plane``: RIS centre at the origin, normal ``+y``, Tx ``(d sin a, d cos a, h)``;
* ``O_T/normal``: Tx at the origin, RIS centre ``(d cos a, d sin a, h)``, normal ``-x``;
* ``O_T/plane``: Tx at the origin, RIS centre ``(d cos a, d sin a, h)``, normal ``+y``;

with ``a in [-45, 0] deg`` and ``u = z x n``, ``v = +z``. The panel has ``tiles_u x
tiles_v`` tiles of ``8 x 5`` cells; tile column ``tu`` takes the state
``tile_column_states[layout][tu]`` (0 = sigma1, 1 = sigma2).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
from validation.ants_fig4b.protocol import cell_layout

from sremu import rng as sremu_rng
from sremu.physics.antennas import cos_power_alpha_for_gain_dbi
from sremu.physics.geometry import (
    PanelGrid,
    Pose,
    point_rectangle_distance,
    segments_hit_polygon,
)
from sremu.physics.walls import p2040_material
from sremu.units import C0

__all__ = [
    "ANTENNA_DIR",
    "Geometry",
    "PriorDraw",
    "antenna_table_path",
    "antenna_table_text",
    "draw_prior",
    "feasibility",
    "geometry",
    "panel_corners",
    "scenario_dict",
    "write_antenna_tables",
]

ANTENNA_DIR: Final[Path] = Path(__file__).resolve().parent / "antennas"
_N_CELLS: Final[int] = 30


@dataclass(frozen=True, eq=False)
class PriorDraw:
    """One draw from the geometry prior (all fields in protocol units)."""

    name: str
    origin: str
    theta0: str
    d_m: float
    a_deg: float
    layout: str
    pitch: str
    gap_m: float
    cell_model: str
    gamma_mag: tuple[float, float]
    gamma_phase_offset_deg: tuple[float, float]
    antenna: str
    point_dxy_m: np.ndarray
    plane_dz_m: float
    height_m: float
    polarization: str
    attempts: int = 1
    rejected: Mapping[str, int] = field(default_factory=dict)

    @property
    def hypothesis(self) -> str:
        """``"<origin>/<theta0>"``."""
        return f"{self.origin}/{self.theta0}"

    def as_record(self) -> dict[str, Any]:
        """Flat JSON-ready record (point errors summarized by their RMS)."""
        return {
            "name": self.name,
            "origin": self.origin,
            "theta0": self.theta0,
            "d_m": self.d_m,
            "a_deg": self.a_deg,
            "layout": self.layout,
            "pitch": self.pitch,
            "gap_m": self.gap_m,
            "cell_model": self.cell_model,
            "gamma_mag_s1": self.gamma_mag[0],
            "gamma_mag_s2": self.gamma_mag[1],
            "gamma_dphase_s1_deg": self.gamma_phase_offset_deg[0],
            "gamma_dphase_s2_deg": self.gamma_phase_offset_deg[1],
            "antenna": self.antenna,
            "plane_dz_m": self.plane_dz_m,
            "height_m": self.height_m,
            "polarization": self.polarization,
            "point_err_rms_m": float(np.sqrt(np.mean(self.point_dxy_m**2))),
            "attempts": self.attempts,
        }


def _categorical(rng: np.random.Generator, spec: Mapping[str, Any]) -> str:
    w = np.asarray(spec["weights"], dtype=np.float64)
    cdf = np.cumsum(w) / np.sum(w)
    u = rng.random()
    i = int(np.searchsorted(cdf, u, side="right"))
    return str(spec["values"][min(i, len(spec["values"]) - 1)])


def _uniform(rng: np.random.Generator, lohi: Any, size: Any = None) -> Any:
    lo, hi = float(lohi[0]), float(lohi[1])
    return rng.uniform(lo, hi, size=size)


def _candidate(rng: np.random.Generator, prior: Mapping[str, Any], name: str) -> PriorDraw:
    """One candidate in the documented ``prior.draw_order``."""
    origin = _categorical(rng, prior["polar_origin"])
    theta0 = _categorical(rng, prior["theta0_direction"])
    d_m = float(_uniform(rng, prior["placement"]["d_m"]))
    a_deg = float(_uniform(rng, prior["placement"]["a_deg"]))
    layout = _categorical(rng, prior["layout"])
    pitch = _categorical(rng, prior["pitch"])
    gap_m = float(_uniform(rng, prior["tile_gap_m"]))
    cell_model = _categorical(rng, prior["cell_model"])
    mag = _uniform(rng, prior["gamma"]["mag"], size=2)
    dph = _uniform(rng, prior["gamma"]["phase_offset_deg"], size=2)
    antenna = _categorical(rng, prior["antenna"])
    pe = prior["point_error"]
    hw = float(pe["xy_half_width_m"])
    dxy = rng.uniform(-hw, hw, size=(_N_CELLS, 2))
    hz = float(pe["plane_height_half_width_m"])
    dz = float(rng.uniform(-hz, hz))
    height = float(_uniform(rng, prior["floor"]["antenna_height_m"]))
    pol = _categorical(rng, prior["floor"]["polarization"])
    return PriorDraw(
        name=name,
        origin=origin,
        theta0=theta0,
        d_m=d_m,
        a_deg=a_deg,
        layout=layout,
        pitch=pitch,
        gap_m=gap_m,
        cell_model=cell_model,
        gamma_mag=(float(mag[0]), float(mag[1])),
        gamma_phase_offset_deg=(float(dph[0]), float(dph[1])),
        antenna=antenna,
        point_dxy_m=dxy,
        plane_dz_m=dz,
        height_m=height,
        polarization=pol,
    )


def draw_prior(
    protocol: Mapping[str, Any],
    *,
    purpose: str,
    name: str,
    counter: int = 0,
    seed: int | None = None,
    accept: Callable[[PriorDraw], str | None] | None = None,
) -> PriorDraw:
    """A feasible prior draw from stream ``(seed, purpose, name, counter)``.

    Candidates are drawn in ``prior.draw_order`` from one keyed generator; infeasible ones
    (:func:`feasibility`, then the optional ``accept`` hook, which returns a rejection reason
    or None) are rejected and the next candidate is drawn from the same stream, at most
    ``prior.feasibility.max_attempts`` times (``RuntimeError`` beyond)."""
    s = int(protocol["seed"] if seed is None else seed)
    gen = sremu_rng.generator(s, purpose, name, counter=counter)
    prior = protocol["prior"]
    max_att = int(prior["feasibility"]["max_attempts"])
    rejected: dict[str, int] = {}
    for att in range(1, max_att + 1):
        cand = _candidate(gen, prior, name)
        ok, reason = feasibility(geometry(cand, protocol), protocol)
        if ok and accept is not None:
            why = accept(cand)
            ok, reason = why is None, (why or "ok")
        if ok:
            return replace(cand, attempts=att, rejected=dict(rejected))
        key = f"{cand.hypothesis}:{reason}"
        rejected[key] = rejected.get(key, 0) + 1
    raise RuntimeError(f"prior draw {name}: no feasible candidate in {max_att} attempts")


# =========================================================================== geometry


@dataclass(frozen=True, eq=False)
class Geometry:
    """World geometry of one draw (m)."""

    tx_m: np.ndarray
    ris_center_m: np.ndarray
    normal: np.ndarray
    u_axis: np.ndarray
    points_m: np.ndarray
    n_u: int
    n_v: int
    n_tiles_u: int
    n_tiles_v: int
    pitch_u_m: float
    pitch_v_m: float
    gap_m: float
    cell_state: np.ndarray
    wavelength_m: float
    fc_hz: float

    def grid(self) -> PanelGrid:
        """The sremu :class:`PanelGrid` of the panel."""
        pose = Pose(self.ris_center_m, self.normal, self.u_axis)
        return PanelGrid(
            pose,
            self.n_u,
            self.n_v,
            self.pitch_u_m,
            self.pitch_v_m,
            self.n_tiles_u,
            self.n_tiles_v,
            self.gap_m,
        )


def _cell_state(protocol: Mapping[str, Any], layout: str) -> tuple[np.ndarray, int, int, int, int]:
    lay = protocol["prior"]["layout"]
    tc = protocol["prior"]["tile_cells"]
    tu_n, tv_n = (int(x) for x in lay["tiles_u_v"][layout])
    cu, cv = int(tc["n_u"]), int(tc["n_v"])
    cols = np.asarray(lay["tile_column_states"][layout], dtype=np.int16)
    if cols.size != tu_n:
        raise ValueError(f"layout {layout}: {cols.size} column states for {tu_n} tile columns")
    n_u, n_v = tu_n * cu, tv_n * cv
    iu = np.arange(n_u * n_v) % n_u  # cell n = iv*n_u + iu (D17)
    return cols[iu // cu].astype(np.int16), n_u, n_v, tu_n, tv_n


def geometry(draw: PriorDraw, protocol: Mapping[str, Any]) -> Geometry:
    """World coordinates of the Tx, panel and the 30 points for ``draw``."""
    fc = float(protocol["prior"]["fc_hz"])
    h = draw.height_m
    a = math.radians(draw.a_deg)
    d = draw.d_m
    if draw.origin == "O_R":
        c = np.array([0.0, 0.0, h])
        if draw.theta0 == "normal":
            n = np.array([1.0, 0.0, 0.0])
            tx = np.array([d * math.cos(a), d * math.sin(a), h])
        else:
            n = np.array([0.0, 1.0, 0.0])
            tx = np.array([d * math.sin(a), d * math.cos(a), h])
    elif draw.origin == "O_T":
        tx = np.array([0.0, 0.0, h])
        c = np.array([d * math.cos(a), d * math.sin(a), h])
        n = np.array([-1.0, 0.0, 0.0]) if draw.theta0 == "normal" else np.array([0.0, 1.0, 0.0])
    else:
        raise ValueError(f"unknown polar origin {draw.origin!r}")
    u = np.cross(np.array([0.0, 0.0, 1.0]), n)
    lay = cell_layout(protocol)
    th = np.radians(lay.theta_deg)
    pts = np.stack(
        [
            lay.r_m * np.cos(th) + draw.point_dxy_m[:, 0],
            lay.r_m * np.sin(th) + draw.point_dxy_m[:, 1],
            np.full(lay.n, h + draw.plane_dz_m),
        ],
        axis=1,
    )
    pi = protocol["prior"]["pitch"]["values"].index(draw.pitch)
    pu = float(protocol["prior"]["pitch"]["pitch_u_m"][pi])
    pv = float(protocol["prior"]["pitch"]["pitch_v_m"][pi])
    state, n_u, n_v, tu_n, tv_n = _cell_state(protocol, draw.layout)
    return Geometry(
        tx_m=tx,
        ris_center_m=c,
        normal=n,
        u_axis=u,
        points_m=pts,
        n_u=n_u,
        n_v=n_v,
        n_tiles_u=tu_n,
        n_tiles_v=tv_n,
        pitch_u_m=pu,
        pitch_v_m=pv,
        gap_m=draw.gap_m,
        cell_state=state,
        wavelength_m=C0 / fc,
        fc_hz=fc,
    )


def panel_corners(geo: Geometry) -> np.ndarray:
    """(4, 3) corners of the panel's outer rectangle (gaps included)."""
    grid = geo.grid()
    wu, wv = grid.size_m
    su = np.array([-1.0, 1.0, 1.0, -1.0]) * wu / 2
    sv = np.array([-1.0, -1.0, 1.0, 1.0]) * wv / 2
    return np.asarray(grid.pose.to_world(su, sv), dtype=np.float64)


def feasibility(geo: Geometry, protocol: Mapping[str, Any]) -> tuple[bool, str]:
    """``(ok, reason)``: every node at least ``min_node_panel_distance_wavelengths`` from the
    panel rectangle and no Tx-point segment crossing the panel (sremu geometry helpers)."""
    fe = protocol["prior"]["feasibility"]
    grid = geo.grid()
    dmin = float(fe["min_node_panel_distance_wavelengths"]) * geo.wavelength_m
    nodes = np.vstack([geo.tx_m[None, :], geo.points_m])
    if float(np.min(point_rectangle_distance(nodes, grid))) < dmin:
        return False, "near_panel"
    if bool(fe["los_must_not_cross_panel"]):
        a = np.broadcast_to(geo.tx_m, geo.points_m.shape)
        if bool(np.any(segments_hit_polygon(a, geo.points_m, panel_corners(geo)))):
            return False, "los_through_panel"
    return True, "ok"


# =========================================================================== antennas


def antenna_table_path(name: str, protocol: Mapping[str, Any]) -> Path:
    """Path of the tabulated pattern of antenna option ``name`` (``patch_*``)."""
    fbr = float(protocol["prior"]["antenna"]["front_to_back_db"])
    return ANTENNA_DIR / f"{name}_fbr{fbr:g}.csv"


def antenna_table_text(name: str, protocol: Mapping[str, Any]) -> str:
    """CSV text ``theta_deg,phi_deg,gain_dbi`` of a patch option: ``G0 max(cos^alpha psi,
    10^(-FBR/10))`` with ``alpha = G0/2 - 1`` (sremu ``cos_power_alpha_for_gain_dbi``)."""
    ant = protocol["prior"]["antenna"]
    i = ant["values"].index(name)
    g0_dbi = ant["g0_dbi"][i]
    if g0_dbi is None:
        raise ValueError(f"{name} is not a tabulated option")
    alpha = cos_power_alpha_for_gain_dbi(float(g0_dbi))
    floor = 10.0 ** (-float(ant["front_to_back_db"]) / 10.0)
    step = float(ant["table_step_deg"])
    th = np.arange(0.0, 180.0 + step / 2, step)
    c = np.cos(np.radians(th))
    rel = np.where(c > 0.0, np.clip(c, 0.0, None) ** alpha, 0.0)
    gdb = float(g0_dbi) + 10.0 * np.log10(np.maximum(rel, floor))
    lines = ["theta_deg,phi_deg,gain_dbi"]
    lines += [f"{t:.1f},0.0,{g:.6f}" for t, g in zip(th, gdb, strict=True)]
    return "\n".join(lines) + "\n"


def write_antenna_tables(protocol: Mapping[str, Any]) -> list[Path]:
    """(Re)write the tabulated patch patterns into :data:`ANTENNA_DIR`."""
    ANTENNA_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, g in zip(
        protocol["prior"]["antenna"]["values"], protocol["prior"]["antenna"]["g0_dbi"], strict=True
    ):
        if g is None:
            continue
        p = antenna_table_path(name, protocol)
        p.write_text(antenna_table_text(name, protocol), encoding="utf-8", newline="\n")
        out.append(p)
    return out


def _antenna_spec(name: str, protocol: Mapping[str, Any], boresight: np.ndarray) -> dict[str, Any]:
    if name == "isotropic":
        return {"pattern": "isotropic"}
    ant = protocol["prior"]["antenna"]
    return {
        "pattern": {
            "tabulated": {
                "file": str(antenna_table_path(name, protocol)),
                "provenance": {
                    "source_kind": "assumed",
                    "description": (
                        f"V-LAB prior antenna option {name}: CosPower front hemisphere with a "
                        f"{ant['front_to_back_db']} dB front-to-back floor [ENG]"
                    ),
                    "reference": "validation/ants_fig4b/PROTOCOL.md",
                },
            }
        },
        "boresight": [float(x) for x in boresight],
    }


# =========================================================================== scenario


def scenario_dict(
    draw: PriorDraw,
    protocol: Mapping[str, Any],
    *,
    ground: bool,
    backend: str = "numba_cpu",
    external_rx: bool = False,
    subcell_max_wavelengths: float = 0.2,
) -> dict[str, Any]:
    """The sremu scenario mapping (section 9.2 field names) of ``draw``.

    Nodes ``tx`` (stream ``tx0``) and ``p00`` ... ``p29`` (stream ``rx0``, cell order); one
    panel ``ris`` with the lab configuration as its initial state and the built-in table
    ``lab_binary_3p5ghz`` at ``assume_mag = 1`` (the drawn Gamma is applied to the separable
    coefficients by :mod:`.predict`). ``ground=True`` adds the concrete floor (M3).
    ``external_rx`` gives the points external trajectories (for position gradients);
    ``subcell_max_wavelengths`` is the initial ``po_patch`` sub-cell size (section 5.4)."""
    geo = geometry(draw, protocol)
    fc = float(protocol["prior"]["fc_hz"])
    rate = "1e6"
    nodes: list[dict[str, Any]] = [
        {
            "id": "tx",
            "position_m": [float(x) for x in geo.tx_m],
            "antenna": _antenna_spec(draw.antenna, protocol, geo.ris_center_m - geo.tx_m),
            "streams": [
                {
                    "name": "tx0",
                    "direction": "tx",
                    "fc_hz": fc,
                    "rate_sps": rate,
                    "occupied_bw_hz": 1.0e5,
                }
            ],
        }
    ]
    for i, p in enumerate(geo.points_m):
        nd: dict[str, Any] = {
            "id": f"p{i:02d}",
            "antenna": _antenna_spec(draw.antenna, protocol, geo.ris_center_m - p),
            "streams": [{"name": "rx0", "direction": "rx", "fc_hz": fc, "rate_sps": rate}],
        }
        if external_rx:
            nd["trajectory"] = {"model": "external", "initial_m": [float(x) for x in p]}
        else:
            nd["position_m"] = [float(x) for x in p]
        nodes.append(nd)
    env: dict[str, Any] = {}
    if ground:
        fl = protocol["prior"]["floor"]
        eps_r, sigma = p2040_material(str(fl["material"]), fc)
        env["ground"] = {
            "eps_r": eps_r,
            "sigma_s_per_m": sigma,
            "polarization": draw.polarization,
            "height_m": 0.0,
        }
    return {
        "schema_version": "1.0",
        "name": f"vlab_{draw.name.replace('/', '_')}",
        "seed": int(protocol["seed"]),
        "numerics": {"backend": backend, "subcell_max_wavelengths": subcell_max_wavelengths},
        "environment": env,
        "nodes": nodes,
        "ris": [
            {
                "id": "ris",
                "pose": {
                    "center_m": [float(x) for x in geo.ris_center_m],
                    "normal": [float(x) for x in geo.normal],
                    "u_axis": [float(x) for x in geo.u_axis],
                },
                "grid": {
                    "n_u": geo.n_u,
                    "n_v": geo.n_v,
                    "pitch_u_m": geo.pitch_u_m,
                    "pitch_v_m": geo.pitch_v_m,
                    "cell_model": draw.cell_model,
                },
                "tiles": {
                    "n_tiles_u": geo.n_tiles_u,
                    "n_tiles_v": geo.n_tiles_v,
                    "gap_m": geo.gap_m,
                },
                "states": {
                    "library": str(protocol["prior"]["gamma"]["table"]),
                    "assume_mag": {"sigma1": 1.0, "sigma2": 1.0},
                },
                "control": {"granularity": "element"},
                "initial": {"states": [int(s) for s in geo.cell_state]},
            }
        ],
        "run": {"duration_s": "10"},
    }
