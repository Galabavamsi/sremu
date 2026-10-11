"""Geometry prior of the V-LAB protocol v2: keyed draws, tile geometry and sremu scenarios.

DESIGN.md section 10.9 ("Geometry prior") as re-specified by protocol v2 from the
author-supplied lab facts (``protocol.json`` ``author_facts``; PROTOCOL.md "Revision
history"). Every number comes from ``protocol.json`` (``prior``, ``frame``). A draw is a pure
function of ``(seed, purpose, name, counter)`` through :func:`sremu.rng.generator` (D14);
infeasible candidates (``prior.feasibility``) are rejected and redrawn from the same stream.

Frames (``protocol.json`` ``frame``): z is up, the ground (M3 only) is ``z = 0``; the Tx, the
tile centres and the measurement plane are at the antenna height ``h``. Measurement point
``i`` is ``(r_i cos theta_i + dx_i, r_i sin theta_i + dy_i, h + dz)`` around the polar origin
``(0, 0)``. The RIS is ``n_tiles`` (six) tiles of ``n_u x n_v = 5 x 8`` cells, one sremu panel
per tile; tile ``k`` (from the lowest polar angle) takes the state
``tiles.states_by_position[k]`` (0 = sigma1, 1 = sigma2) in every cell.

* ``arc_facing_tx`` (primary, author facts 1-5): Tx at the origin; tile ``k`` centre
  ``(R cos phi_k, R sin phi_k, h)``, normal ``-(cos phi_k, sin phi_k, 0)`` (facing the Tx),
  ``u = z x n`` tangential and horizontal (the 5-cell side), ``v = +z`` (the 8-cell side),
  ``phi_k = 22.5 deg + offset + (k + 0.5 - 3) S/6``.
* ``flat_panel_v1`` (sensitivity run S-OR, the v1 O_R rows): RIS centre at the origin, normal
  ``+x`` (theta = 0 along the normal) or ``+y`` (theta = 0 in the panel plane), Tx at
  ``(d cos a, d sin a, h)`` or ``(d sin a, d cos a, h)``; the six tiles form one flat row along
  ``u = z x n`` with centre spacing ``n_u pitch_u + gap``.

Antennas (``prior.antenna``): isotropic, or a vertical short dipole
``G = 1.5 sin^2(psi)`` (``psi`` from +z, 40 dB null floor) tabulated every 0.5 deg; both are
azimuth-omnidirectional (the VERT2450/VERT900 whips of author fact 6), so no node is aimed.
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
    "ARC",
    "FLAT",
    "Geometry",
    "PriorDraw",
    "antenna_table_path",
    "antenna_table_text",
    "draw_prior",
    "feasibility",
    "geometry",
    "node_tile_distance",
    "panel_ids",
    "scenario_dict",
    "tile_corners",
    "write_antenna_tables",
]

ANTENNA_DIR: Final[Path] = Path(__file__).resolve().parent / "antennas"
ARC: Final[str] = "arc_facing_tx"
FLAT: Final[str] = "flat_panel_v1"
_N_CELLS: Final[int] = 30
_Z: Final[np.ndarray] = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True, eq=False)
class PriorDraw:
    """One draw from the geometry prior (all fields in protocol units).

    Geometry fields that the draw's ``geometry_model`` does not use are NaN (``radius_m``,
    ``span_deg``, ``centre_deg`` for the flat panel; ``d_m``, ``a_deg``, ``gap_m`` for the
    arc). ``theta0`` is ``"arc"`` for the arc model."""

    name: str
    geometry_model: str
    origin: str
    theta0: str
    radius_m: float
    span_deg: float
    centre_deg: float
    d_m: float
    a_deg: float
    gap_m: float
    pitch: str
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
        """``"<origin>/<theta0>"`` (``O_T/arc`` for the primary prior)."""
        return f"{self.origin}/{self.theta0}"

    def as_record(self) -> dict[str, Any]:
        """Flat JSON-ready record (point errors summarized by their RMS)."""
        return {
            "name": self.name,
            "geometry_model": self.geometry_model,
            "origin": self.origin,
            "theta0": self.theta0,
            "radius_m": self.radius_m,
            "span_deg": self.span_deg,
            "centre_deg": self.centre_deg,
            "d_m": self.d_m,
            "a_deg": self.a_deg,
            "gap_m": self.gap_m,
            "pitch": self.pitch,
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


def _geometry_fields(rng: np.random.Generator, g: Mapping[str, Any]) -> dict[str, Any]:
    """The geometry block of a candidate, in ``geometry.draw_order``."""
    nan = math.nan
    if g["model"] == ARC:
        radius = float(_uniform(rng, g["radius_m"]))
        span = float(_uniform(rng, g["span_deg"]))
        centre = float(g["centre_nominal_deg"]) + float(_uniform(rng, g["centre_offset_deg"]))
        return {
            "geometry_model": ARC,
            "origin": str(g["polar_origin"]),
            "theta0": "arc",
            "radius_m": radius,
            "span_deg": span,
            "centre_deg": centre,
            "d_m": nan,
            "a_deg": nan,
            "gap_m": nan,
        }
    if g["model"] == FLAT:
        theta0 = _categorical(rng, g["theta0_direction"])
        d_m = float(_uniform(rng, g["d_m"]))
        a_deg = float(_uniform(rng, g["a_deg"]))
        gap = float(_uniform(rng, g["tile_gap_m"]))
        return {
            "geometry_model": FLAT,
            "origin": str(g["polar_origin"]),
            "theta0": theta0,
            "radius_m": nan,
            "span_deg": nan,
            "centre_deg": nan,
            "d_m": d_m,
            "a_deg": a_deg,
            "gap_m": gap,
        }
    raise ValueError(f"unknown geometry model {g['model']!r}")


def _candidate(rng: np.random.Generator, prior: Mapping[str, Any], name: str) -> PriorDraw:
    """One candidate in the documented ``prior.draw_order``."""
    geo = _geometry_fields(rng, prior["geometry"])
    pitch = _categorical(rng, prior["pitch"])
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
        pitch=pitch,
        cell_model=cell_model,
        gamma_mag=(float(mag[0]), float(mag[1])),
        gamma_phase_offset_deg=(float(dph[0]), float(dph[1])),
        antenna=antenna,
        point_dxy_m=dxy,
        plane_dz_m=dz,
        height_m=height,
        polarization=pol,
        **geo,
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
    """World geometry of one draw (m): the Tx, the 30 points and the tiles (one sremu panel
    each, ``n_u x n_v`` cells of ``pitch_u x pitch_v``, uniform state ``tile_state[k]``)."""

    tx_m: np.ndarray
    points_m: np.ndarray
    tile_center_m: np.ndarray
    tile_normal: np.ndarray
    tile_u_axis: np.ndarray
    tile_state: np.ndarray
    n_u: int
    n_v: int
    pitch_u_m: float
    pitch_v_m: float
    wavelength_m: float
    fc_hz: float

    @property
    def n_tiles(self) -> int:
        """Number of tiles (panels)."""
        return int(self.tile_center_m.shape[0])

    @property
    def tile_phi_deg(self) -> np.ndarray:
        """Polar angle of every tile centre in the grid frame (deg)."""
        c = self.tile_center_m
        return np.degrees(np.arctan2(c[:, 1], c[:, 0]))

    @property
    def tile_size_m(self) -> tuple[float, float]:
        """Tile width along u and height along v (m)."""
        return self.n_u * self.pitch_u_m, self.n_v * self.pitch_v_m

    def grid(self, k: int) -> PanelGrid:
        """The sremu :class:`PanelGrid` of tile ``k`` (one tile, no gap)."""
        pose = Pose(self.tile_center_m[k], self.tile_normal[k], self.tile_u_axis[k])
        return PanelGrid(pose, self.n_u, self.n_v, self.pitch_u_m, self.pitch_v_m, 1, 1, 0.0)

    def grids(self) -> list[PanelGrid]:
        """Every tile's :class:`PanelGrid`."""
        return [self.grid(k) for k in range(self.n_tiles)]

    def cell_centres(self) -> np.ndarray:
        """``(n_tiles * n_u * n_v, 3)`` physical cell centres of every tile (tile-major)."""
        return np.concatenate([g.cell_centres() for g in self.grids()])


def _pitch(protocol: Mapping[str, Any], name: str) -> tuple[float, float]:
    pt = protocol["prior"]["pitch"]
    i = pt["values"].index(name)
    return float(pt["pitch_u_m"][i]), float(pt["pitch_v_m"][i])


def _tile_poses(
    draw: PriorDraw, protocol: Mapping[str, Any], h: float, tile_w: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(tx, centres[T, 3], normals[T, 3], u_axes[T, 3])`` of the draw's geometry model."""
    nt = int(protocol["prior"]["tiles"]["n_tiles"])
    k = np.arange(nt, dtype=np.float64)
    if draw.geometry_model == ARC:
        tx = np.array([0.0, 0.0, h])
        phi = np.radians(draw.centre_deg + (k + 0.5 - nt / 2.0) * draw.span_deg / nt)
        e = np.stack([np.cos(phi), np.sin(phi), np.zeros(nt)], axis=1)
        centres = draw.radius_m * e + np.array([0.0, 0.0, h])
        normals = -e
        u = np.cross(_Z[None, :], normals)
        return tx, centres, normals, u
    if draw.geometry_model == FLAT:
        a = math.radians(draw.a_deg)
        d = draw.d_m
        c = np.array([0.0, 0.0, h])
        if draw.theta0 == "normal":
            n = np.array([1.0, 0.0, 0.0])
            tx = np.array([d * math.cos(a), d * math.sin(a), h])
        elif draw.theta0 == "plane":
            n = np.array([0.0, 1.0, 0.0])
            tx = np.array([d * math.sin(a), d * math.cos(a), h])
        else:
            raise ValueError(f"unknown theta0 direction {draw.theta0!r}")
        u = np.cross(_Z, n)
        step = tile_w + draw.gap_m
        centres = c[None, :] + ((k - (nt - 1) / 2.0) * step)[:, None] * u[None, :]
        return tx, centres, np.tile(n, (nt, 1)), np.tile(u, (nt, 1))
    raise ValueError(f"unknown geometry model {draw.geometry_model!r}")


def geometry(draw: PriorDraw, protocol: Mapping[str, Any]) -> Geometry:
    """World coordinates of the Tx, the tiles and the 30 points for ``draw``."""
    fc = float(protocol["prior"]["fc_hz"])
    h = draw.height_m
    tl = protocol["prior"]["tiles"]
    n_u, n_v = int(tl["n_u"]), int(tl["n_v"])
    pu, pv = _pitch(protocol, draw.pitch)
    tx, centres, normals, u = _tile_poses(draw, protocol, h, n_u * pu)
    states = np.asarray(tl["states_by_position"], dtype=np.int16)
    if states.size != centres.shape[0]:
        raise ValueError(f"{states.size} tile states for {centres.shape[0]} tiles")
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
    return Geometry(
        tx_m=tx,
        points_m=pts,
        tile_center_m=centres,
        tile_normal=normals,
        tile_u_axis=u,
        tile_state=states,
        n_u=n_u,
        n_v=n_v,
        pitch_u_m=pu,
        pitch_v_m=pv,
        wavelength_m=C0 / fc,
        fc_hz=fc,
    )


def tile_corners(geo: Geometry) -> np.ndarray:
    """``(n_tiles, 4, 3)`` corners of every tile rectangle."""
    wu, wv = geo.tile_size_m
    su = np.array([-1.0, 1.0, 1.0, -1.0]) * wu / 2
    sv = np.array([-1.0, -1.0, 1.0, 1.0]) * wv / 2
    return np.stack([np.asarray(g.pose.to_world(su, sv), dtype=np.float64) for g in geo.grids()])


def node_tile_distance(geo: Geometry, nodes: np.ndarray) -> np.ndarray:
    """``(n_nodes,)`` distance from every node to the nearest tile rectangle (m)."""
    d = np.stack([point_rectangle_distance(nodes, g) for g in geo.grids()])
    return np.asarray(np.min(d, axis=0), dtype=np.float64)


def _tiles_overlap(geo: Geometry) -> bool:
    """Whether the interiors of two tile rectangles intersect. The tiles are vertical and
    share their centre height and size, so this is tested on their horizontal centre
    segments (touching at an end point is not an overlap)."""
    wu, _wv = geo.tile_size_m
    a = (geo.tile_center_m - geo.tile_u_axis * wu / 2)[:, :2]
    b = (geo.tile_center_m + geo.tile_u_axis * wu / 2)[:, :2]
    nt = geo.n_tiles
    return any(
        _segments_overlap_2d(a[i], b[i], a[j], b[j]) for i in range(nt) for j in range(i + 1, nt)
    )


def _segments_overlap_2d(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray, eps: float = 1e-12
) -> bool:
    """Proper crossing of two 2-D segments, or overlapping interiors when collinear."""

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    if abs(d1) <= eps and abs(d2) <= eps:  # collinear: compare the projections on p
        t = p2 - p1
        tt = float(t @ t)
        lo, hi = sorted((float((q1 - p1) @ t) / tt, float((q2 - p1) @ t) / tt))
        return hi > eps and lo < 1.0 - eps
    return d1 * d2 < 0 and d3 * d4 < 0


def feasibility(geo: Geometry, protocol: Mapping[str, Any]) -> tuple[bool, str]:
    """``(ok, reason)`` under ``prior.feasibility``: tiles must not overlap
    (``tiles_overlap``), every node at least ``min_node_tile_distance_wavelengths`` from
    every tile rectangle (``near_tile``) and, if ``los_must_not_cross_tiles``, no Tx-point
    segment crossing a tile (``los_through_tile``)."""
    fe = protocol["prior"]["feasibility"]
    if bool(fe["tiles_must_not_overlap"]) and _tiles_overlap(geo):
        return False, "tiles_overlap"
    dmin = float(fe["min_node_tile_distance_wavelengths"]) * geo.wavelength_m
    nodes = np.vstack([geo.tx_m[None, :], geo.points_m])
    if float(np.min(node_tile_distance(geo, nodes))) < dmin:
        return False, "near_tile"
    if bool(fe["los_must_not_cross_tiles"]):
        a = np.broadcast_to(geo.tx_m, geo.points_m.shape)
        for corners in tile_corners(geo):
            if bool(np.any(segments_hit_polygon(a, geo.points_m, corners))):
                return False, "los_through_tile"
    return True, "ok"


# =========================================================================== antennas


def _tabulated_options(protocol: Mapping[str, Any]) -> list[str]:
    ant = protocol["prior"]["antenna"]
    return [
        str(v) for v, g in zip(ant["values"], ant["peak_gain_linear"], strict=True) if g is not None
    ]


def antenna_table_path(name: str, protocol: Mapping[str, Any]) -> Path:
    """Path of the tabulated pattern of antenna option ``name`` (``short_dipole_z``)."""
    null_db = float(protocol["prior"]["antenna"]["null_floor_db"])
    return ANTENNA_DIR / f"{name}_null{null_db:g}.csv"


def antenna_table_text(name: str, protocol: Mapping[str, Any]) -> str:
    """CSV text ``theta_deg,phi_deg,gain_dbi`` of the vertical short-dipole option:
    ``G(psi) = G0 max(sin^2 psi, 10^(-null_floor_db/10))`` with ``psi`` the angle from the
    antenna axis (the table's ``theta``) and ``G0 = peak_gain_linear`` (1.5 for a short
    dipole). Azimuth-independent (one ``phi`` column)."""
    ant = protocol["prior"]["antenna"]
    i = ant["values"].index(name)
    g0 = ant["peak_gain_linear"][i]
    if g0 is None:
        raise ValueError(f"{name} is not a tabulated option")
    floor = 10.0 ** (-float(ant["null_floor_db"]) / 10.0)
    step = float(ant["table_step_deg"])
    th = np.arange(0.0, 180.0 + step / 2, step)
    rel = np.sin(np.radians(th)) ** 2
    gdb = 10.0 * np.log10(float(g0)) + 10.0 * np.log10(np.maximum(rel, floor))
    lines = ["theta_deg,phi_deg,gain_dbi"]
    lines += [f"{t:.1f},0.0,{g:.6f}" for t, g in zip(th, gdb, strict=True)]
    return "\n".join(lines) + "\n"


def write_antenna_tables(protocol: Mapping[str, Any]) -> list[Path]:
    """(Re)write the tabulated antenna patterns of the prior into :data:`ANTENNA_DIR`."""
    ANTENNA_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name in _tabulated_options(protocol):
        p = antenna_table_path(name, protocol)
        p.write_text(antenna_table_text(name, protocol), encoding="utf-8", newline="\n")
        out.append(p)
    return out


def _antenna_spec(name: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """sremu antenna spec of option ``name``: isotropic, or the tabulated vertical pattern
    with its axis (``boresight``) along ``prior.antenna.axis`` (+z) for every node; the
    pattern is azimuth-independent, so no node is aimed (``ref_axis`` +x only fixes the
    table's unused azimuth origin)."""
    if name == "isotropic":
        return {"pattern": "isotropic"}
    ant = protocol["prior"]["antenna"]
    g0 = ant["peak_gain_linear"][ant["values"].index(name)]
    return {
        "pattern": {
            "tabulated": {
                "file": str(antenna_table_path(name, protocol)),
                "ref_axis": [1.0, 0.0, 0.0],
                "provenance": {
                    "source_kind": "assumed",
                    "data_class": "ASSUMED",
                    "description": (
                        f"V-LAB prior antenna option {name}: vertical short dipole, "
                        f"G = {g0} sin^2(psi) from the vertical axis, {ant['null_floor_db']} dB "
                        "null floor; model of the VERT2450/VERT900 whips (author, 2026-10-11) "
                        "[ENG]"
                    ),
                    "reference": "validation/ants_fig4b/PROTOCOL.md",
                },
            }
        },
        "boresight": [float(x) for x in ant["axis"]],
    }


# =========================================================================== scenario


def panel_ids(n_tiles: int) -> list[str]:
    """sremu panel ids of the tiles, ``t0`` ... (tile order = increasing polar angle for the
    arc, increasing u for the flat row)."""
    return [f"t{k}" for k in range(n_tiles)]


def scenario_dict(
    draw: PriorDraw,
    protocol: Mapping[str, Any],
    *,
    ground: bool,
    backend: str = "numba_cpu",
    external_rx: bool = False,
    subcell_max_wavelengths: float = 0.2,
    cell_model: str | None = None,
) -> dict[str, Any]:
    """The sremu scenario mapping (section 9.2 field names) of ``draw``.

    Nodes ``tx`` (stream ``tx0``) and ``p00`` ... ``p29`` (stream ``rx0``, cell order); one
    RIS panel per tile (``t0`` ... ``t5``) with its own pose, ``n_u x n_v`` cells, the tile's
    state in every cell as its initial state, and the built-in table ``lab_binary_3p5ghz`` at
    ``assume_mag = 1`` (the drawn Gamma is applied to the separable coefficients by
    :mod:`.predict`). ``ground=True`` adds the concrete floor (M3). ``external_rx`` gives the
    points external trajectories (for position gradients); ``subcell_max_wavelengths`` is the
    initial ``po_patch`` sub-cell size (section 5.4); ``cell_model`` overrides the draw's
    (the paired point-vs-patch evaluation)."""
    geo = geometry(draw, protocol)
    fc = float(protocol["prior"]["fc_hz"])
    cm = draw.cell_model if cell_model is None else str(cell_model)
    rate = "1e6"
    nodes: list[dict[str, Any]] = [
        {
            "id": "tx",
            "position_m": [float(x) for x in geo.tx_m],
            "antenna": _antenna_spec(draw.antenna, protocol),
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
            "antenna": _antenna_spec(draw.antenna, protocol),
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
    n_cells = geo.n_u * geo.n_v
    ris = []
    for k, pid in enumerate(panel_ids(geo.n_tiles)):
        ris.append(
            {
                "id": pid,
                "pose": {
                    "center_m": [float(x) for x in geo.tile_center_m[k]],
                    "normal": [float(x) for x in geo.tile_normal[k]],
                    "u_axis": [float(x) for x in geo.tile_u_axis[k]],
                },
                "grid": {
                    "n_u": geo.n_u,
                    "n_v": geo.n_v,
                    "pitch_u_m": geo.pitch_u_m,
                    "pitch_v_m": geo.pitch_v_m,
                    "cell_model": cm,
                },
                "tiles": {"n_tiles_u": 1, "n_tiles_v": 1, "gap_m": 0.0},
                "states": {
                    "library": str(protocol["prior"]["gamma"]["table"]),
                    "assume_mag": {"sigma1": 1.0, "sigma2": 1.0},
                },
                "control": {"granularity": "element"},
                "initial": {"states": [int(geo.tile_state[k])] * n_cells},
            }
        )
    return {
        "schema_version": "1.0",
        "name": f"vlab_{draw.name.replace('/', '_')}",
        "seed": int(protocol["seed"]),
        "numerics": {"backend": backend, "subcell_max_wavelengths": subcell_max_wavelengths},
        "environment": env,
        "nodes": nodes,
        "ris": ris,
        "run": {"duration_s": "10"},
    }
