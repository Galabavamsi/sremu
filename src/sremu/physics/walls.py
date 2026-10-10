"""Order-1 image walls (DESIGN.md section 5.8; SHOULD; WP3).

A wall is a finite planar polygon with (``eps_r``, ``sigma_s_per_m``) and an optional RMS
roughness ``roughness_m``. Both faces reflect (same material). Release 1 models **image
order 1 only**: for each link and wall the Tx is mirrored in the wall plane; the path is
valid only when Tx and Rx lie strictly on the same side of the plane and the specular point
(where the segment ``img(p_tx) -> p_rx`` crosses the plane) lies inside the polygon or on
its boundary (``sremu.physics.geometry.segments_hit_polygon``). Order 2, wall-ground and
RIS-wall paths are cut (section 5.8). Walls do not block the LOS: blockage comes only from
explicit occluders (section 5.2, "scenario names never imply blockage").

**Amplitude** (excluding the propagation phase, contract ``PathBundle.amp``)::

    sqrt(G_T(unit(S - p_tx)) G_R(unit(S - p_rx))) * lambda/(4 pi d_img) * Gamma * rho_s * V

with ``S`` the specular point, ``d_img = |p_rx - img(p_tx)|`` (delay ``d_img/C0``), ``V``
the explicit occluder/partition factors on the legs Tx -> S -> Rx (section 5.9) times the
knife-edge factor ``F(nu)`` of the dominant edge crossing either leg (section 9.2: an edge
"applies to the paths it intersects"; one edge per path; the leg rule of
:mod:`sremu.physics.diffraction`), and the optional Rayleigh roughness factor
``rho_s = exp(-2 (k sigma_h sin psi)^2)`` (section 5.8), ``psi`` the grazing angle to the
wall plane.

**One Fresnel coefficient per bounce** (section 5.8), as a scalar co-polar coefficient
[ENG]. The scalar model carries one polarization ``p`` ("V": E along the vertical
direction projected transverse to the ray, ``unit(z - (z.k)k)``; "H": ``unit(k x z)``).
With the plane-of-incidence normal ``e_s = unit(k_i x n)`` (TE direction) and the TM basis
``e_p = e_s x k`` for the incident (``k_i``) and reflected (``k_r``) rays::

    Gamma = Gamma_H(psi) (p_i . e_s)(p_r . e_s) + Gamma_V(psi) (p_i . e_p,i)(p_r . e_p,r)

where ``Gamma_H``/``Gamma_V`` are the section 5.7 formulas (TE/TM in the ``e_p = e_s x k``
convention: PEC gives -1/+1) and ``p_i = p(k_i)``, ``p_r = p(k_r)``. For a horizontal
reflector this reduces exactly to ``Gamma_H`` (pol "H") or ``Gamma_V`` (pol "V") as for the
ground; for a vertical wall and horizontal rays a "V" link sees ``Gamma_H`` (TE). The form is
reciprocal: swapping Tx and Rx flips ``e_s`` and swaps the two TM basis vectors, and ``p`` is
even ("V") or odd ("H") in ``k``. At normal incidence any ``e_s`` transverse to the ray gives
the same value, ``Gamma_H (p_i . p_r)``. A vertically propagating ray has no defined
"V"/"H" direction; the x axis is used [ENG].

The polarization is the scenario's single scalar polarization: ``environment.ground.
polarization`` when a ground is given, else "V" [ENG; recorded in ``assumptions``]. The
section 9.2 wall schema has no polarization field.

Materials: the scenario gives ``eps_r`` and ``sigma_s_per_m`` directly.
:data:`P2040_TABLE3` transcribes Rec. ITU-R P.2040-3 (08/2023) Table 3 ("Material
properties", ``eta' = a f^b``, ``sigma = c f^d`` with f in GHz) for scenario authors,
with its citation; see :func:`p2040_material`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike

from sremu.contracts import PathKind
from sremu.errors import ConfigurationError, GeometryError, OutOfDomain
from sremu.physics.antennas import Antenna, Isotropic
from sremu.physics.diffraction import KnifeEdge, KnifeEdgeMechanism
from sremu.physics.geometry import segments_hit_polygon
from sremu.physics.ground import (
    Polarization,
    complex_permittivity,
    fresnel_gamma_sin,
    scenario_edges,
)
from sremu.physics.los import Occluder, los_visibility, occluders_from_spec
from sremu.units import C0

__all__ = [
    "P2040_CITATION",
    "P2040_TABLE3",
    "P2040Row",
    "Wall",
    "WallBounce",
    "WallsMechanism",
    "copol_gamma",
    "mechanism_from_scenario",
    "p2040_material",
    "scenario_polarization",
    "wall_bounce",
    "walls_from_spec",
]

_Z: Final = np.array([0.0, 0.0, 1.0])
_X: Final = np.array([1.0, 0.0, 0.0])
_DEGENERATE: Final[float] = 1e-12


# --------------------------------------------------------------------------- P.2040 Table 3


@dataclass(frozen=True)
class P2040Row:
    """One row of Rec. ITU-R P.2040-3 Table 3: ``eta' = a f^b``, ``sigma = c f^d`` (S/m),
    f in GHz, valid on ``[f_min_ghz, f_max_ghz]``."""

    material: str
    a: float
    b: float
    c: float
    d: float
    f_min_ghz: float
    f_max_ghz: float


P2040_CITATION: Final[str] = (
    "Recommendation ITU-R P.2040-3 (08/2023), 'Effects of building materials and structures "
    "on radiowave propagation above about 100 MHz', Annex 1, Table 3 'Material properties' "
    "(eta' = a f^b, sigma = c f^d, f in GHz). Source PDF: https://www.itu.int/dms_pubrec/"
    "itu-r/rec/p/R-REC-P.2040-3-202308-S!!PDF-E.pdf. Transcribed 2026-10-10 by WP3 from an "
    "automated text extraction of that PDF. P.2040-1 (07/2015) Table 3 differs (concrete, "
    "brick, plasterboard, glass and ceiling board were revised; plywood, marble and the "
    "220-450 GHz rows were added later). Re-checked 2026-10-10 (WP3 review) against a second, "
    "independent automated extraction of the same PDF: all 17 rows agree. P.2040-4 (09/2025, "
    "in force; R-REC-P.2040-4-202509-I) keeps these 17 rows unchanged per an automated "
    "extraction and adds rows not transcribed here (100-450 GHz entries; vinyl tile, carpet "
    "tile, asphalt concrete at 1-40 GHz); it calls the frequency ranges indicative, not hard "
    "limits. Both checks are text extractions: spot-check the PDF visually before quoting a "
    "value in the paper."
)
"""Citation and status of :data:`P2040_TABLE3` (DESIGN.md section 5.8)."""

P2040_TABLE3: Final[tuple[P2040Row, ...]] = (
    P2040Row("vacuum", 1.0, 0.0, 0.0, 0.0, 0.001, 100.0),
    P2040Row("concrete", 5.24, 0.0, 0.0462, 0.7822, 1.0, 100.0),
    P2040Row("brick", 3.91, 0.0, 0.0238, 0.16, 1.0, 40.0),
    P2040Row("plasterboard", 2.73, 0.0, 0.0085, 0.9395, 1.0, 100.0),
    P2040Row("wood", 1.99, 0.0, 0.0047, 1.0718, 0.001, 100.0),
    P2040Row("glass", 6.31, 0.0, 0.0036, 1.3394, 0.1, 100.0),
    P2040Row("glass", 5.79, 0.0, 0.0004, 1.658, 220.0, 450.0),
    P2040Row("ceiling_board", 1.48, 0.0, 0.0011, 1.0750, 1.0, 100.0),
    P2040Row("ceiling_board", 1.52, 0.0, 0.0029, 1.029, 220.0, 450.0),
    P2040Row("chipboard", 2.58, 0.0, 0.0217, 0.7800, 1.0, 100.0),
    P2040Row("plywood", 2.71, 0.0, 0.33, 0.0, 1.0, 40.0),
    P2040Row("marble", 7.074, 0.0, 0.0055, 0.9262, 1.0, 60.0),
    P2040Row("floorboard", 3.66, 0.0, 0.0044, 1.3515, 50.0, 100.0),
    P2040Row("metal", 1.0, 0.0, 1.0e7, 0.0, 1.0, 100.0),
    P2040Row("very_dry_ground", 3.0, 0.0, 0.00015, 2.52, 1.0, 10.0),
    P2040Row("medium_dry_ground", 15.0, -0.1, 0.035, 1.63, 1.0, 10.0),
    P2040Row("wet_ground", 30.0, -0.4, 0.15, 1.30, 1.0, 10.0),
)
"""Rec. ITU-R P.2040-3 Table 3 (see :data:`P2040_CITATION` for provenance and status)."""


def p2040_material(name: str, fc_hz: float) -> tuple[float, float]:
    """``(eps_r, sigma_s_per_m)`` of a P.2040-3 Table 3 material at ``fc_hz``.

    ``name`` is case-insensitive with spaces or underscores ("ceiling board"). A frequency
    outside every row's range raises ``OutOfDomain`` (no extrapolation)."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    rows = [r for r in P2040_TABLE3 if r.material == key]
    if not rows:
        known = sorted({r.material for r in P2040_TABLE3})
        raise ConfigurationError(f"unknown P.2040 material {name!r}; known: {known}")
    f_ghz = float(fc_hz) / 1e9
    for r in rows:
        if r.f_min_ghz <= f_ghz <= r.f_max_ghz:
            return r.a * f_ghz**r.b, r.c * f_ghz**r.d
    ranges = [(r.f_min_ghz, r.f_max_ghz) for r in rows]
    raise OutOfDomain(f"P.2040 {key}: {f_ghz:g} GHz is outside the table ranges {ranges} GHz")


# --------------------------------------------------------------------------- co-pol Gamma


def _unit(v: np.ndarray) -> np.ndarray:
    n = math.sqrt(float(v @ v))
    if n == 0.0:
        raise GeometryError("zero-length direction")
    return v / n


def _pol_vector(k: np.ndarray, pol: Polarization) -> np.ndarray:
    """Unit transverse polarization of a ray with unit direction ``k`` ([ENG])."""
    if pol == "V":
        v = _Z - float(_Z @ k) * k
    else:
        v = np.cross(k, _Z)
    n = math.sqrt(float(v @ v))
    if n <= _DEGENERATE:  # vertical ray: "V"/"H" undefined; use the x axis made transverse
        v = _X - float(_X @ k) * k
        n = math.sqrt(float(v @ v))
    return v / n


def copol_gamma(
    k_in: ArrayLike,
    normal: ArrayLike,
    eps_c: complex,
    pol: Polarization,
) -> complex:
    """Scalar co-polar reflection coefficient of a plane interface (module docstring).

    ``k_in``: unit direction of the incident ray (toward the surface); ``normal``: unit
    plane normal (either orientation)."""
    k_i = _unit(np.asarray(k_in, dtype=np.float64))
    n = _unit(np.asarray(normal, dtype=np.float64))
    kn = float(k_i @ n)
    k_r = k_i - 2.0 * kn * n
    sin_psi = abs(kn)
    gh = complex(fresnel_gamma_sin(sin_psi, eps_c, "H"))
    gv = complex(fresnel_gamma_sin(sin_psi, eps_c, "V"))
    es = np.cross(k_i, n)
    ns = math.sqrt(float(es @ es))
    if ns <= _DEGENERATE:  # normal incidence: any transverse e_s gives the same value
        es = np.cross(k_i, _X if abs(float(k_i @ _X)) < 0.9 else np.array([0.0, 1.0, 0.0]))
        ns = math.sqrt(float(es @ es))
    es = es / ns
    ep_i = np.cross(es, k_i)
    ep_r = np.cross(es, k_r)
    p_i = _pol_vector(k_i, pol)
    p_r = _pol_vector(k_r, pol)
    return gh * float(p_i @ es) * float(p_r @ es) + gv * float(p_i @ ep_i) * float(p_r @ ep_r)


# --------------------------------------------------------------------------- walls


@dataclass(frozen=True, eq=False)
class Wall:
    """A finite planar polygon wall (section 9.2 ``environment.walls[]``)."""

    id: str
    vertices_m: np.ndarray
    eps_r: float
    sigma_s_per_m: float
    roughness_m: float = 0.0
    origin: np.ndarray = field(init=False, repr=False)
    normal: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices_m, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3 or not np.all(np.isfinite(v)):
            raise ConfigurationError(f"wall {self.id!r}: vertices_m must be (M >= 3, 3) finite")
        complex_permittivity(self.eps_r, self.sigma_s_per_m, 1.0)  # validates the material
        if not (float(self.roughness_m) >= 0.0) or math.isinf(float(self.roughness_m)):
            raise ConfigurationError(f"wall {self.id!r}: roughness_m must be finite and >= 0")
        # Newell normal: robust for planar polygons in any orientation
        nrm = np.zeros(3)
        m = v.shape[0]
        for i in range(m):
            a, b = v[i], v[(i + 1) % m]
            nrm[0] += (a[1] - b[1]) * (a[2] + b[2])
            nrm[1] += (a[2] - b[2]) * (a[0] + b[0])
            nrm[2] += (a[0] - b[0]) * (a[1] + b[1])
        nn = float(np.linalg.norm(nrm))
        if nn == 0.0:
            raise ConfigurationError(f"wall {self.id!r}: degenerate polygon (zero area)")
        nrm = nrm / nn
        origin = v.mean(axis=0)
        if np.max(np.abs((v - origin) @ nrm)) > 1e-9 * max(1.0, float(np.ptp(v))):
            raise ConfigurationError(f"wall {self.id!r}: vertices are not coplanar")
        object.__setattr__(self, "vertices_m", v)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "normal", nrm)

    def signed_distance(self, p: ArrayLike) -> float:
        """Signed distance (m) of a point from the wall plane (along ``normal``)."""
        return float((np.asarray(p, dtype=np.float64) - self.origin) @ self.normal)

    def image(self, p: ArrayLike) -> np.ndarray:
        """Mirror image of a point in the wall plane."""
        q = np.asarray(p, dtype=np.float64)
        return q - 2.0 * self.signed_distance(q) * self.normal


@dataclass(frozen=True)
class WallBounce:
    """Geometry of a valid order-1 wall bounce."""

    specular_m: np.ndarray
    d_img_m: float
    k_in: np.ndarray
    sin_psi: float


def wall_bounce(wall: Wall, p_tx: ArrayLike, p_rx: ArrayLike) -> WallBounce | None:
    """The order-1 bounce Tx -> wall -> Rx, or None when it is not valid (section 5.8)."""
    a = np.asarray(p_tx, dtype=np.float64)
    b = np.asarray(p_rx, dtype=np.float64)
    sa = wall.signed_distance(a)
    sb = wall.signed_distance(b)
    if not sa * sb > 0.0:  # strictly on the same side (either face)
        return None
    a_img = wall.image(a)
    if not bool(segments_hit_polygon(a_img[None], b[None], wall.vertices_m)[0]):
        return None
    dv = b - a_img
    d = math.sqrt(float(dv @ dv))
    w = abs(sa) / (abs(sa) + abs(sb))
    s = a_img + w * dv
    k_in = _unit(s - a)
    sin_psi = (abs(sa) + abs(sb)) / d
    return WallBounce(s, d, k_in, sin_psi)


def _gain(ant: Antenna, src: np.ndarray, dst: np.ndarray) -> float:
    if isinstance(ant, Isotropic):
        return 1.0
    return float(ant.gain(_unit(dst - src)[None, :])[0])


@dataclass(eq=False)
class WallsMechanism:
    """WP2 propagation hook: one ``PathKind.WALL`` path per (link, valid wall), walls in
    scenario order (``link_paths``). ``occluders`` act on both legs (section 5.9); so does
    the dominant knife edge of ``edges`` crossing either leg (module docstring)."""

    walls: tuple[Wall, ...]
    polarization: Polarization = "V"
    occluders: tuple[Occluder, ...] = ()
    assumptions: tuple[str, ...] = ()
    edges: tuple[KnifeEdge, ...] = ()
    _diffraction: KnifeEdgeMechanism | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.walls = tuple(self.walls)
        self.occluders = tuple(self.occluders)
        self.edges = tuple(self.edges)
        self._diffraction = KnifeEdgeMechanism(self.edges) if self.edges else None
        ids = [w.id for w in self.walls]
        if len(set(ids)) != len(ids):
            raise ConfigurationError("wall ids must be unique")
        if self.polarization not in ("H", "V"):
            raise ConfigurationError(f"polarization must be 'H' or 'V', got {self.polarization!r}")

    def wall_path(
        self,
        wall: Wall,
        p_tx: ArrayLike,
        p_rx: ArrayLike,
        fc_hz: float,
        ant_tx: Antenna | None = None,
        ant_rx: Antenna | None = None,
        link: str = "",
    ) -> tuple[complex, float] | None:
        """(amplitude excluding the propagation phase, delay s) of the order-1 path via
        ``wall``, or None when the bounce is not valid. ``link`` names the path in a
        multi-edge warning."""
        a = np.asarray(p_tx, dtype=np.float64)
        b = np.asarray(p_rx, dtype=np.float64)
        bn = wall_bounce(wall, a, b)
        if bn is None:
            return None
        eps_c = complex_permittivity(wall.eps_r, wall.sigma_s_per_m, fc_hz)
        g = copol_gamma(bn.k_in, wall.normal, eps_c, self.polarization)
        lam = C0 / float(fc_hz)
        if wall.roughness_m > 0.0:
            k = 2.0 * math.pi / lam
            g *= math.exp(-2.0 * (k * float(wall.roughness_m) * bn.sin_psi) ** 2)
        v: complex | float = 1.0
        if self.occluders:
            v = float(los_visibility(a, bn.specular_m, self.occluders)) * float(
                los_visibility(bn.specular_m, b, self.occluders)
            )
        if self._diffraction is not None:
            v = v * self._diffraction.path_factor(
                np.stack([a, bn.specular_m, b]), fc_hz, f"{link or '?'} wall {wall.id}"
            )
        gt = _gain(ant_tx or Isotropic(), a, bn.specular_m)
        gr = _gain(ant_rx or Isotropic(), b, bn.specular_m)
        amp = math.sqrt(gt * gr) * lam / (4.0 * math.pi * bn.d_img_m) * g * v
        return complex(amp), bn.d_img_m / C0

    def link_paths(self, ctx: Any) -> list[tuple[complex, float, int]]:
        """Hook: ``[(amp, delay_s, PathKind.WALL)]`` for every wall with a valid bounce."""
        out: list[tuple[complex, float, int]] = []
        for w in self.walls:
            p = self.wall_path(
                w, ctx.p_tx, ctx.p_rx, ctx.fc_hz, ctx.ant_tx, ctx.ant_rx, str(ctx.link)
            )
            if p is not None:
                out.append((p[0], p[1], int(PathKind.WALL)))
        return out

    def describe(self) -> dict[str, Any]:
        """JSON-ready description (manifests)."""
        return {
            "mechanism": "walls",
            "order": 1,
            "polarization": self.polarization,
            "walls": [
                {
                    "id": w.id,
                    "eps_r": float(w.eps_r),
                    "sigma_s_per_m": float(w.sigma_s_per_m),
                    "roughness_m": float(w.roughness_m),
                    "n_vertices": int(w.vertices_m.shape[0]),
                }
                for w in self.walls
            ],
            "occluders": [o.id for o in self.occluders],
            "edges": [e.id for e in self.edges],
            "assumptions": list(self.assumptions),
        }


def walls_from_spec(items: Iterable[Mapping[str, Any]] | None) -> tuple[Wall, ...]:
    """``environment.walls`` (section 9.2) -> :class:`Wall` tuple."""
    out = []
    for it in items or ():
        try:
            out.append(
                Wall(
                    id=str(it["id"]),
                    vertices_m=np.asarray(it["vertices_m"], dtype=np.float64),
                    eps_r=float(it["eps_r"]),
                    sigma_s_per_m=float(it["sigma_s_per_m"]),
                    roughness_m=float(it.get("roughness_m", 0.0)),
                )
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"environment.walls entry is missing {exc.args[0]!r}"
            ) from None
    return tuple(out)


def scenario_polarization(scenario: Mapping[str, Any]) -> tuple[Polarization, tuple[str, ...]]:
    """The scenario's scalar polarization: ``environment.ground.polarization`` or "V" [ENG],
    with the assumption text when defaulted."""
    env = scenario.get("environment") or {}
    ground = env.get("ground") if isinstance(env, Mapping) else None
    if ground and ground.get("polarization") in ("H", "V"):
        return ground["polarization"], ()
    return "V", (
        "walls: scalar polarization 'V' (default; the scenario has no ground polarization) "
        "[ENG, section 5.8]",
    )


def mechanism_from_scenario(
    spec: Sequence[Mapping[str, Any]] | None,
    scenario: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> WallsMechanism | None:
    """D18 factory: the order-1 walls of ``environment.walls`` (None when the list is empty).
    The scene's occluders and knife edges act on the wall paths' legs (sections 5.9, 9.2)."""
    del base_dir
    walls = walls_from_spec(spec)
    if not walls:
        return None
    pol, assumptions = scenario_polarization(scenario)
    env = scenario.get("environment") or {}
    occ = occluders_from_spec(env.get("occluders") if isinstance(env, Mapping) else None)
    return WallsMechanism(walls, pol, occ, assumptions, edges=scenario_edges(scenario))
