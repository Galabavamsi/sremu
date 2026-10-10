"""Single knife-edge diffraction F(nu) (DESIGN.md section 5.9; WP3).

**Field factor** (exact, coherent)::

    F(nu) = ((1 + j)/2) * int_nu^inf exp(-j pi t^2 / 2) dt
          = ((1 + j)/2) * [(1/2 - C(nu)) - j (1/2 - S(nu))]

with the Fresnel integrals ``S, C`` of ``scipy.special.fresnel``
(``int_0^nu exp(-j pi t^2/2) dt = C(nu) - j S(nu)``, ``int_0^inf = (1 - j)/2``). ``F(0) = 1/2``,
``F(-inf) = 1``; for large ``nu`` the phase is ``-pi nu^2/2 - pi/4``, i.e. the extra path
``Delta = nu^2 lambda/4`` of the diffracted ray plus the -pi/4 edge phase, so
``h_LOS(d) * F(nu)`` rendered at the LOS delay is phase-consistent with the RIS and other
paths and sums coherently with them (section 5.9). The ITU-R P.526 approximation ``J(nu)``
is NOT used here; it is an oracle only (``tests/oracles/rf.py``, PH-11).

**Geometry** [ENG; release 1, one dominant edge]. An edge is the segment ``p0 -> p1``
(``environment.edges[]``): the top of a half-plane screen that hangs below it, toward -z
(the screen spans the edge direction ``e`` and ``d = unit(-z + (z.e) e)``; a vertical edge
has no "below" and is rejected). An edge applies to a link's LOS when the segment
Tx -> Rx crosses the screen's plane strictly between its ends at a point ``P`` whose
coordinate along the edge lies within the segment. Then

* ``h`` = signed distance between the edge line and the LOS line (common perpendicular),
  positive when ``P`` lies on the screen side (LOS obstructed);
* ``d1 = |P - p_tx|``, ``d2 = |p_rx - P|``;
* ``nu = h * sqrt(2 (d1 + d2) / (lambda d1 d2))`` (section 5.9).

The definition is symmetric in Tx and Rx (reciprocity, PH-08). An edge whose screen the
LOS does not cross leaves the LOS unchanged.

**Paths with legs** (section 9.2: an edge "applies to the paths it intersects"; section
5.9: one edge per path) [ENG]. A reflected path is the polyline ``p_tx -> S -> p_rx`` (ground
image path, order-1 wall path; ``S`` the specular point). Each leg that crosses an edge's
screen as above gives a crossing with ``h`` measured from that leg's line and ``d1``/``d2``
the path lengths from the path's start to the crossing and from the crossing to its end
(the unfolded path), so ``nu`` and ``F(nu)`` follow as for the LOS; a two-point path is
exactly the LOS rule. The path's factor is ``F`` of the dominant crossing.

When several crossings (edges, or legs) apply to one path, release 1 keeps only the dominant
one (largest ``nu``; Deygout is cut, section 5.9) and warns once per path
(``ValidityWarning``). Knife edges act on the LOS (``V = F(nu)``, section 5.2) and, through
:class:`sremu.physics.ground.GroundMechanism` and
:class:`sremu.physics.walls.WallsMechanism`, on the ground and wall image paths. RIS legs
are not diffracted in release 1 (the WP2 RIS rows have no leg-factor hook, like
occluders).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import fresnel

from sremu.errors import ConfigurationError, ValidityWarning
from sremu.units import C0

__all__ = [
    "EdgeCrossing",
    "KnifeEdge",
    "KnifeEdgeMechanism",
    "edges_from_spec",
    "knife_edge_F",
    "knife_edge_loss_db",
    "knife_edge_nu",
    "mechanism_from_scenario",
]

_Z: Final = np.array([0.0, 0.0, 1.0])
_VERTICAL_TOL: Final[float] = 1e-9


def knife_edge_F(nu: ArrayLike) -> NDArray[np.complex128] | complex:
    """Complex knife-edge field factor ``F(nu)`` (module docstring); scalar in, scalar out."""
    v = np.asarray(nu, dtype=np.float64)
    s, c = fresnel(v)
    out = (0.5 + 0.5j) * ((0.5 - c) - 1j * (0.5 - s))
    if v.ndim == 0:
        return complex(out)
    return np.asarray(out, dtype=np.complex128)


def knife_edge_loss_db(nu: ArrayLike) -> NDArray[np.float64] | float:
    """Diffraction loss ``-20 log10 |F(nu)|`` in dB (positive = loss)."""
    f = np.abs(np.asarray(knife_edge_F(nu)))
    out = -20.0 * np.log10(f)
    return float(out) if np.ndim(out) == 0 else np.asarray(out, dtype=np.float64)


def knife_edge_nu(h_m: float, d1_m: float, d2_m: float, fc_hz: float) -> float:
    """``nu = h sqrt(2 (d1 + d2)/(lambda d1 d2))`` (section 5.9), ``lambda = C0/fc``."""
    if not (d1_m > 0.0 and d2_m > 0.0 and fc_hz > 0.0):
        raise ConfigurationError("knife edge: d1, d2 and fc must be > 0")
    lam = C0 / float(fc_hz)
    return float(h_m) * math.sqrt(2.0 * (float(d1_m) + float(d2_m)) / (lam * d1_m * d2_m))


@dataclass(frozen=True)
class EdgeCrossing:
    """Where a LOS meets an edge's screen: clearance ``h_m`` (> 0 obstructed), ``d1_m``,
    ``d2_m`` and the crossing point ``point_m``."""

    h_m: float
    d1_m: float
    d2_m: float
    point_m: np.ndarray

    def nu(self, fc_hz: float) -> float:
        """Fresnel-Kirchhoff parameter at carrier ``fc_hz``."""
        return knife_edge_nu(self.h_m, self.d1_m, self.d2_m, fc_hz)


@dataclass(frozen=True, eq=False)
class KnifeEdge:
    """A knife edge given as a segment (section 9.2 ``environment.edges[]``)."""

    id: str
    p0_m: np.ndarray
    p1_m: np.ndarray
    e: np.ndarray = field(init=False, repr=False)
    down: np.ndarray = field(init=False, repr=False)
    m: np.ndarray = field(init=False, repr=False)
    length_m: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        p0 = np.asarray(self.p0_m, dtype=np.float64).reshape(-1)
        p1 = np.asarray(self.p1_m, dtype=np.float64).reshape(-1)
        if p0.shape != (3,) or p1.shape != (3,) or not np.all(np.isfinite(np.r_[p0, p1])):
            raise ConfigurationError(f"edge {self.id!r}: p0_m and p1_m must be 3 finite numbers")
        dv = p1 - p0
        length = math.sqrt(float(dv @ dv))
        if length == 0.0:
            raise ConfigurationError(f"edge {self.id!r}: p0_m == p1_m")
        e = dv / length
        d = -_Z + float(_Z @ e) * e
        nd = math.sqrt(float(d @ d))
        if nd < _VERTICAL_TOL:
            raise ConfigurationError(
                f"edge {self.id!r} is vertical: a knife edge is the top of a screen hanging "
                "below it (-z), which a vertical edge does not define (release 1)"
            )
        d = d / nd
        object.__setattr__(self, "p0_m", p0)
        object.__setattr__(self, "p1_m", p1)
        object.__setattr__(self, "e", e)
        object.__setattr__(self, "down", d)
        object.__setattr__(self, "m", np.cross(e, d))
        object.__setattr__(self, "length_m", length)

    def _leg_crossing(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[float, float, float, np.ndarray] | None:
        """``(h, s, |b - a|, P)`` where the segment a -> b crosses this edge's screen at
        ``P = a + s (b - a)``, or None when the edge does not apply to it."""
        k = b - a
        d = math.sqrt(float(k @ k))
        if d == 0.0:
            return None
        denom = float(k @ self.m)
        if denom == 0.0:
            return None
        s = float((self.p0_m - a) @ self.m) / denom
        if not 0.0 < s < 1.0:
            return None
        p = a + s * k
        rel = p - self.p0_m
        along = float(rel @ self.e)
        if along < 0.0 or along > self.length_m:
            return None
        h_vert = float(rel @ self.down)
        kh = k / d
        cr = np.cross(self.e, kh)
        h = h_vert * abs(float(kh @ self.m)) / math.sqrt(float(cr @ cr))
        return h, s, d, p

    def crossing(self, p_tx: ArrayLike, p_rx: ArrayLike) -> EdgeCrossing | None:
        """The LOS segment's crossing of this edge's screen, or None when the edge does not
        apply (module docstring)."""
        a = np.asarray(p_tx, dtype=np.float64)
        b = np.asarray(p_rx, dtype=np.float64)
        c = self._leg_crossing(a, b)
        if c is None:
            return None
        h, s, d, p = c
        return EdgeCrossing(h, s * d, (1.0 - s) * d, p)

    def path_crossings(self, vertices: ArrayLike) -> tuple[EdgeCrossing, ...]:
        """Crossings of this edge's screen by the polyline path ``vertices`` (V >= 2, 3):
        per crossing leg, ``h`` from that leg's line and ``d1``/``d2`` the path lengths from
        the start to the crossing and from the crossing to the end (module docstring). A
        two-point path gives exactly :meth:`crossing`."""
        pts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] < 2:
            raise ConfigurationError("a path needs at least two points")
        if pts.shape[0] == 2:
            c0 = self.crossing(pts[0], pts[1])
            return () if c0 is None else (c0,)
        lengths = [math.sqrt(float(dv @ dv)) for dv in np.diff(pts, axis=0)]
        out: list[EdgeCrossing] = []
        for i in range(len(lengths)):
            c = self._leg_crossing(pts[i], pts[i + 1])
            if c is None:
                continue
            h, s, d, p = c
            before = math.fsum(lengths[:i])
            after = math.fsum(lengths[i + 1 :])
            out.append(EdgeCrossing(h, before + s * d, (1.0 - s) * d + after, p))
        return tuple(out)


@dataclass(eq=False)
class KnifeEdgeMechanism:
    """WP2 propagation hook: ``los_factor(ctx) = F(nu)`` of the dominant applicable edge
    (1 when no edge applies). :meth:`path_factor` gives the same for a path with legs
    (ground and wall image paths, module docstring)."""

    edges: tuple[KnifeEdge, ...]
    _warned: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self.edges = tuple(self.edges)
        ids = [x.id for x in self.edges]
        if len(set(ids)) != len(ids):
            raise ConfigurationError("edge ids must be unique")

    def dominant_on_path(
        self, vertices: ArrayLike, fc_hz: float, key: str = ""
    ) -> tuple[KnifeEdge, float] | None:
        """(edge, nu) of the dominant crossing of the polyline path ``vertices`` (V >= 2,
        3), or None when no edge applies. ``key`` names the path in the warning."""
        pts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        found: list[tuple[float, int, KnifeEdge]] = []
        for i, edge in enumerate(self.edges):
            for cr in edge.path_crossings(pts):
                found.append((cr.nu(fc_hz), i, edge))
        if not found:
            return None
        if len(found) > 1:
            wkey = key or f"{tuple(np.round(pts, 9).ravel().tolist())}"
            if wkey not in self._warned:
                self._warned.add(wkey)
                warnings.warn(
                    f"path {key or '?'}: {len(found)} knife-edge crossings apply; release 1 "
                    "models one edge per path and keeps the dominant one (largest nu); "
                    "multi-edge (Deygout) diffraction is not modelled",
                    ValidityWarning,
                    stacklevel=3,
                )
        nu, _i, edge = max(found, key=lambda x: (x[0], -x[1]))
        return edge, nu

    def dominant(
        self, p_tx: ArrayLike, p_rx: ArrayLike, fc_hz: float, link: str = ""
    ) -> tuple[KnifeEdge, float] | None:
        """(edge, nu) of the dominant edge applicable to the LOS p_tx -> p_rx, or None."""
        pts = np.stack([np.asarray(p_tx, np.float64), np.asarray(p_rx, np.float64)])
        return self.dominant_on_path(pts, fc_hz, link)

    def path_factor(self, vertices: ArrayLike, fc_hz: float, key: str = "") -> complex:
        """Factor ``F(nu)`` of the dominant crossing of a polyline path (1 when none)."""
        dom = self.dominant_on_path(vertices, fc_hz, key)
        if dom is None:
            return 1.0 + 0.0j
        return complex(knife_edge_F(dom[1]))

    def factor(self, p_tx: ArrayLike, p_rx: ArrayLike, fc_hz: float, link: str = "") -> complex:
        """LOS factor ``F(nu)`` of the dominant edge (1 when none applies)."""
        dom = self.dominant(p_tx, p_rx, fc_hz, link)
        if dom is None:
            return 1.0 + 0.0j
        return complex(knife_edge_F(dom[1]))

    def los_factor(self, ctx: Any) -> complex:
        """Hook: multiplies the LOS ``V`` (section 5.2)."""
        return self.factor(ctx.p_tx, ctx.p_rx, ctx.fc_hz, str(ctx.link))

    def describe(self) -> dict[str, Any]:
        """JSON-ready description (manifests)."""
        return {
            "mechanism": "knife_edge",
            "model": (
                "F(nu) Fresnel-Kirchhoff, one dominant edge per path; LOS here, ground and "
                "wall image paths via their mechanisms; RIS legs not diffracted"
            ),
            "edges": [
                {"id": x.id, "p0_m": x.p0_m.tolist(), "p1_m": x.p1_m.tolist()} for x in self.edges
            ],
        }


def edges_from_spec(items: Iterable[Mapping[str, Any]] | None) -> tuple[KnifeEdge, ...]:
    """``environment.edges`` (section 9.2) -> :class:`KnifeEdge` tuple."""
    out = []
    for it in items or ():
        try:
            out.append(
                KnifeEdge(
                    str(it["id"]),
                    np.asarray(it["p0_m"], dtype=np.float64),
                    np.asarray(it["p1_m"], dtype=np.float64),
                )
            )
        except KeyError as exc:
            raise ConfigurationError(
                f"environment.edges entry is missing {exc.args[0]!r}"
            ) from None
    return tuple(out)


def mechanism_from_scenario(
    spec: Sequence[Mapping[str, Any]] | None,
    scenario: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> KnifeEdgeMechanism | None:
    """D18 factory: the knife edges of ``environment.edges`` (None when the list is empty)."""
    del scenario, base_dir
    edges = edges_from_spec(spec)
    return KnifeEdgeMechanism(edges) if edges else None
