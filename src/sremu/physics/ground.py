"""Flat ground: Fresnel coefficients, the image path and image expansion on RIS hops.

DESIGN.md section 5.7 (WP3). The ground is the plane ``z = height_m`` (z up, section 3.1)
with complex relative permittivity

    eps_c = eps_r - j 60 sigma lambda                               (section 5.7)

(``exp(+j omega t)`` convention of the baseband definition, section 3.1). Fresnel
coefficients at the grazing angle ``psi`` (angle between the ray and the ground plane):

    Gamma_H = (sin psi - sqrt(eps_c - cos^2 psi)) / (sin psi + sqrt(eps_c - cos^2 psi))
    Gamma_V = (eps_c sin psi - sqrt(eps_c - cos^2 psi)) / (eps_c sin psi + sqrt(eps_c - cos^2 psi))

with the principal square root. A perfect conductor (``sigma = inf`` or ``eps_r = inf``,
library API only; the scenario schema accepts finite numbers) gives exactly
``Gamma_H = -1`` and ``Gamma_V = +1``. The degenerate ``eps_c = 1`` at ``psi = 0`` (no
interface at grazing) is defined as ``Gamma = 0``, its value at every other angle.

**Image path** (``PathKind.GROUND``). The source is mirrored in the ground plane; the path
length is the image distance ``d_img = |p_rx - img(p_tx)|`` and the grazing angle follows
from ``sin psi = (a + b)/d_img`` with ``a``, ``b`` the heights of Tx and Rx above the
ground; the root is formed as ``sqrt((eps_c - 1) + sin^2 psi)`` (equal to the one above,
without cancellation near grazing or for ``eps_c`` near 1). Antenna gains are taken
toward the specular point (section 5.7: "toward the image direction"):
``G_T(unit(img(p_rx) - p_tx))`` and
``G_R(unit(img(p_tx) - p_rx))``. Amplitude (excluding the propagation phase, contract
``PathBundle.amp``) ``sqrt(G_T G_R) lambda/(4 pi d_img) Gamma V``; delay ``d_img/C0``.
``V`` is the product of the explicit occluder/partition factors (section 5.9) on the two
legs Tx -> specular point -> Rx [ENG: the same segment-polygon rule as the LOS] and the
knife-edge factor ``F(nu)`` of the dominant edge crossing either leg (section 9.2: an edge
"applies to the paths it intersects"; one edge per path, section 5.9; the leg rule of
:mod:`sremu.physics.diffraction`).

**Image expansion on RIS hops** (section 5.7; RIS rows with hop variants, section 3.4):
for each link and panel, variant 1 replaces the Tx by its image, variant 2 the Rx by its
image and variant 3 both. The K1 kernel then sees the unfolded distances and angles
(incidence from ``img(p_tx)``, reflection toward ``img(p_rx)``), weighted per quadrature
point by the ground coefficient(s) at the respective specular point(s), and the real
antennas' gains are taken toward the specular points:
``G_T(unit(img(q) - p_tx))`` and ``G_R(unit(img(q) - p_rx))``. Variant 0 (direct) is WP2's
row. Swapping Tx and Rx maps variant 1 to 2 and leaves 0 and 3 (reciprocity, PH-32).

Validity domain (model card): smooth, flat ground, no earth curvature (P.526 smooth earth
is the EXPERIMENTAL module ``physics/experimental/p526.py``, not used here). Every node and
every RIS quadrature point must be on or above the ground (``GeometryError`` otherwise).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sremu.contracts import PathKind
from sremu.errors import ConfigurationError, GeometryError
from sremu.physics.antennas import Antenna, Isotropic
from sremu.physics.diffraction import KnifeEdge, KnifeEdgeMechanism, edges_from_spec
from sremu.physics.los import Occluder, los_visibility, occluders_from_spec
from sremu.units import C0

__all__ = [
    "GROUND_PATH_KIND",
    "GroundMechanism",
    "GroundPlane",
    "Polarization",
    "complex_permittivity",
    "fresnel_gamma",
    "fresnel_gamma_h",
    "fresnel_gamma_sin",
    "fresnel_gamma_v",
    "ground_from_spec",
    "mechanism_from_scenario",
    "scenario_edges",
]

Polarization = Literal["H", "V"]

GROUND_PATH_KIND: Final[PathKind] = PathKind.GROUND


# --------------------------------------------------------------------------- Fresnel


def complex_permittivity(eps_r: float, sigma_s_per_m: float, fc_hz: float) -> complex:
    """``eps_c = eps_r - j 60 sigma lambda`` (section 5.7); ``lambda = C0/fc``.

    Returns ``complex(inf)`` for a perfect conductor (``sigma`` or ``eps_r`` infinite).
    """
    er = float(eps_r)
    sg = float(sigma_s_per_m)
    if not (er > 0.0) or not (sg >= 0.0) or math.isnan(er) or math.isnan(sg):
        raise ConfigurationError(f"need eps_r > 0 and sigma >= 0, got {eps_r!r}, {sigma_s_per_m!r}")
    if not (float(fc_hz) > 0.0):
        raise ConfigurationError(f"fc_hz must be > 0, got {fc_hz!r}")
    if math.isinf(er) or math.isinf(sg):
        return complex(math.inf, 0.0)
    return complex(er, -60.0 * sg * (C0 / float(fc_hz)))


def _is_pec(eps_c: complex) -> bool:
    return math.isinf(eps_c.real) or math.isinf(eps_c.imag)


def fresnel_gamma_sin(
    sin_psi: ArrayLike, eps_c: complex, pol: Polarization
) -> NDArray[np.complex128]:
    """Fresnel coefficient from ``sin psi`` (section 5.7), ``pol`` "H" or "V".

    The root is formed as ``sqrt((eps_c - 1) + sin^2 psi)`` (``= sqrt(eps_c - cos^2 psi)``),
    which has no cancellation near grazing or for ``eps_c`` near 1 (``eps_c = 1`` gives
    exactly 0). ``sin psi`` should come from the geometry without cancellation (the image
    geometry uses ``(a + b)/d_img``).

    Branch: ``Im sqrt <= 0``, the decaying transmitted wave for ``exp(+j omega t)``. The
    radicand keeps the sign of ``Im eps_c = -60 sigma lambda <= 0`` (``-0.0`` when lossless),
    so a lossless ``eps_r < 1`` beyond the critical angle gives the ``sigma -> 0+`` limit
    instead of the growing branch that ``+0j`` would select."""
    if pol not in ("H", "V"):
        raise ConfigurationError(f"polarization must be 'H' or 'V', got {pol!r}")
    s = np.asarray(sin_psi, dtype=np.float64)
    ec = complex(eps_c)
    if _is_pec(ec):
        return np.full(s.shape, -1.0 if pol == "H" else 1.0, dtype=np.complex128)
    z = np.empty(s.shape, dtype=np.complex128)
    z.real = (ec.real - 1.0) + s * s
    z.imag = math.copysign(abs(ec.imag), -1.0) if ec.imag <= 0.0 else ec.imag
    r = np.sqrt(z)
    num = s - r if pol == "H" else ec * s - r
    den = s + r if pol == "H" else ec * s + r
    with np.errstate(divide="ignore", invalid="ignore"):
        g = num / den
    # den == 0 only for eps_c == 1 at psi == 0 (no interface): Gamma = 0 as at every psi
    return np.asarray(np.where(den == 0, 0.0 + 0.0j, g), dtype=np.complex128)


def fresnel_gamma(psi_rad: ArrayLike, eps_c: complex, pol: Polarization) -> NDArray[np.complex128]:
    """Fresnel coefficient at grazing angle ``psi`` (rad), polarization "H" or "V"."""
    return fresnel_gamma_sin(np.sin(np.asarray(psi_rad, dtype=np.float64)), eps_c, pol)


def fresnel_gamma_h(psi_rad: ArrayLike, eps_c: complex) -> NDArray[np.complex128]:
    """Horizontal-polarization coefficient ``Gamma_H(psi)`` (section 5.7)."""
    return fresnel_gamma(psi_rad, eps_c, "H")


def fresnel_gamma_v(psi_rad: ArrayLike, eps_c: complex) -> NDArray[np.complex128]:
    """Vertical-polarization coefficient ``Gamma_V(psi)`` (section 5.7)."""
    return fresnel_gamma(psi_rad, eps_c, "V")


# --------------------------------------------------------------------------- ground plane


def _norm_rows(v: np.ndarray) -> np.ndarray:
    return np.sqrt((v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1]) + v[..., 2] * v[..., 2])


@dataclass(frozen=True)
class GroundPlane:
    """The flat ground ``z = height_m`` with (``eps_r``, ``sigma_s_per_m``) and the
    scenario polarization (section 9.2 ``environment.ground``)."""

    eps_r: float
    sigma_s_per_m: float
    polarization: Polarization = "V"
    height_m: float = 0.0

    def __post_init__(self) -> None:
        complex_permittivity(self.eps_r, self.sigma_s_per_m, 1.0)  # validates the material
        if self.polarization not in ("H", "V"):
            raise ConfigurationError(
                f"ground polarization must be 'H' or 'V', got {self.polarization!r}"
            )
        if not math.isfinite(float(self.height_m)):
            raise ConfigurationError("ground height_m must be finite")

    @property
    def is_pec(self) -> bool:
        """Perfect conductor (``Gamma_H = -1``, ``Gamma_V = +1`` exactly)."""
        return math.isinf(float(self.eps_r)) or math.isinf(float(self.sigma_s_per_m))

    def eps_c(self, fc_hz: float) -> complex:
        """Complex permittivity at ``fc_hz``."""
        return complex_permittivity(self.eps_r, self.sigma_s_per_m, fc_hz)

    def image(self, points: ArrayLike) -> NDArray[np.float64]:
        """Mirror images (..., 3) of points in the ground plane."""
        p = np.array(points, dtype=np.float64)
        p[..., 2] = 2.0 * float(self.height_m) - p[..., 2]
        return p

    def heights(self, points: ArrayLike, what: str = "point") -> NDArray[np.float64]:
        """Heights (...,) above the ground; ``GeometryError`` for any point below it."""
        h = np.asarray(points, dtype=np.float64)[..., 2] - float(self.height_m)
        if np.any(h < 0.0):
            raise GeometryError(
                f"{what} below the ground plane z = {self.height_m} m (min height "
                f"{float(np.min(h)):.6g} m); the flat-ground image model needs z >= ground"
            )
        return h

    def bounce(
        self, a: ArrayLike, b: ArrayLike, fc_hz: float, what: str = "point"
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Image distance ``|b - img(a)|`` (...,) and the ground coefficient (...,) of the
        specular bounce a -> ground -> b at carrier ``fc_hz`` (broadcast over rows)."""
        pa = np.asarray(a, dtype=np.float64)
        pb = np.asarray(b, dtype=np.float64)
        ha = self.heights(pa, what)
        hb = self.heights(pb, what)
        dx = pb[..., 0] - pa[..., 0]
        dy = pb[..., 1] - pa[..., 1]
        rho2 = dx * dx + dy * dy
        hs = ha + hb
        d = np.sqrt(rho2 + hs * hs)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(d > 0.0, hs / d, 1.0)
        g = fresnel_gamma_sin(s, self.eps_c(fc_hz), self.polarization)
        return d, g

    def specular_point(self, a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
        """Specular point (..., 3) on the ground of the bounce a -> ground -> b."""
        pa = np.asarray(a, dtype=np.float64)
        pb = np.asarray(b, dtype=np.float64)
        ha = self.heights(pa)
        hb = self.heights(pb)
        hs = ha + hb
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(hs > 0.0, ha / hs, 0.5)
        out = pa + w[..., None] * (pb - pa)
        out[..., 2] = float(self.height_m)
        return out


def _gain_dirs(ant: Antenna, src: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Power gain of ``ant`` at ``src`` toward ``dst_pts`` (Q, 3) (ones when isotropic)."""
    if isinstance(ant, Isotropic):
        return np.ones(dst_pts.shape[0], dtype=np.float64)
    d = dst_pts - src[None, :]
    n = _norm_rows(d)
    if np.any(n == 0.0):
        raise GeometryError("antenna direction toward a coincident point is undefined")
    return np.asarray(ant.gain(d / n[:, None]), dtype=np.float64)


# --------------------------------------------------------------------------- mechanism


@dataclass(eq=False)
class GroundMechanism:
    """WP2 propagation hook (``ScenePropagation(mechanisms=...)``): ground image path per
    link (``link_paths``) and RIS hop variants 1-3 per link and panel (``ris_hops``).

    ``occluders``: the scene's explicit occluders/partitions, applied to both legs of the
    ground path (section 5.9). ``edges``: the scene's knife edges; the dominant one crossing
    either leg multiplies the ground path by ``F(nu)`` (module docstring).
    ``ris_hops_enabled=False`` disables the RIS image expansion (library option for
    diagnostics; the scenario always enables it, section 5.7).
    """

    plane: GroundPlane
    occluders: tuple[Occluder, ...] = ()
    ris_hops_enabled: bool = True
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    edges: tuple[KnifeEdge, ...] = ()
    _diffraction: KnifeEdgeMechanism | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.occluders = tuple(self.occluders)
        self.edges = tuple(self.edges)
        self._diffraction = KnifeEdgeMechanism(self.edges) if self.edges else None

    # ---------------------------------------------------------------- link paths
    def ground_path(
        self,
        p_tx: ArrayLike,
        p_rx: ArrayLike,
        fc_hz: float,
        ant_tx: Antenna | None = None,
        ant_rx: Antenna | None = None,
        link: str = "",
    ) -> tuple[complex, float]:
        """(amplitude excluding the propagation phase, delay in s) of the ground path.
        ``link`` names the path in a multi-edge warning."""
        a = np.asarray(p_tx, dtype=np.float64)
        b = np.asarray(p_rx, dtype=np.float64)
        d_arr, g_arr = self.plane.bounce(a, b, fc_hz, what="node")
        d = float(d_arr)
        if not d > 0.0:
            raise GeometryError("TX and RX coincide on the ground; the image path is undefined")
        b_img = self.plane.image(b)
        a_img = self.plane.image(a)
        gt = float(_gain_dirs(ant_tx or Isotropic(), a, b_img[None, :])[0])
        gr = float(_gain_dirs(ant_rx or Isotropic(), b, a_img[None, :])[0])
        v: complex | float = 1.0
        if self.occluders or self._diffraction is not None:
            s = self.plane.specular_point(a, b)
            if self.occluders:
                v = float(los_visibility(a, s, self.occluders)) * float(
                    los_visibility(s, b, self.occluders)
                )
            if self._diffraction is not None:
                v = v * self._diffraction.path_factor(
                    np.stack([a, s, b]), fc_hz, f"{link or '?'} ground"
                )
        lam = C0 / float(fc_hz)
        amp = math.sqrt(gt * gr) * lam / (4.0 * math.pi * d) * complex(g_arr) * v
        return complex(amp), d / C0

    def link_paths(self, ctx: Any) -> list[tuple[complex, float, int]]:
        """Hook: the link's ground image path ``[(amp, delay_s, PathKind.GROUND)]``."""
        amp, tau = self.ground_path(
            ctx.p_tx, ctx.p_rx, ctx.fc_hz, ctx.ant_tx, ctx.ant_rx, str(ctx.link)
        )
        return [(amp, tau, int(PathKind.GROUND))]

    # ---------------------------------------------------------------- RIS hops
    def hop_weight(
        self, p_node: np.ndarray, points: np.ndarray, fc_hz: float
    ) -> NDArray[np.complex128]:
        """Ground coefficient (Q,) of the bounce node -> ground -> each quadrature point."""
        pts = np.asarray(points, dtype=np.float64)
        _d, g = self.plane.bounce(
            np.broadcast_to(np.asarray(p_node, np.float64), pts.shape), pts, fc_hz, what="RIS cell"
        )
        return np.asarray(g, dtype=np.complex128)

    def ris_hops(self, ctx: Any, panel: Any) -> list[Any]:
        """Hook: hop variants 1 (Tx image), 2 (Rx image) and 3 (both) of one link and panel
        (``sremu.physics.propagation.RisHopSource``), section 5.7."""
        if not self.ris_hops_enabled:
            return []
        from sremu.physics.propagation import RisHopSource  # same layer; avoids a cycle

        p_tx = np.asarray(ctx.p_tx, dtype=np.float64)
        p_rx = np.asarray(ctx.p_rx, dtype=np.float64)
        self.plane.heights(np.stack([p_tx, p_rx]), "node")
        self.plane.heights(panel.grid.cell_centres(), f"panel {panel.id} cell")
        fc = float(ctx.fc_hz)
        tx_img = self.plane.image(p_tx)
        rx_img = self.plane.image(p_rx)
        ant_tx: Antenna = ctx.ant_tx
        ant_rx: Antenna = ctx.ant_rx
        plane = self.plane

        def w_tx(pts: np.ndarray) -> np.ndarray:
            return self.hop_weight(p_tx, pts, fc)

        def w_rx(pts: np.ndarray) -> np.ndarray:
            return self.hop_weight(p_rx, pts, fc)

        def w_both(pts: np.ndarray) -> np.ndarray:
            return w_tx(pts) * w_rx(pts)

        def g_tx_img(pts: np.ndarray) -> np.ndarray:  # real Tx toward the specular point
            return _gain_dirs(ant_tx, p_tx, plane.image(pts))

        def g_rx_img(pts: np.ndarray) -> np.ndarray:  # real Rx toward the specular point
            return _gain_dirs(ant_rx, p_rx, plane.image(pts))

        def g_tx_dir(pts: np.ndarray) -> np.ndarray:
            return _gain_dirs(ant_tx, p_tx, np.asarray(pts, np.float64))

        def g_rx_dir(pts: np.ndarray) -> np.ndarray:
            return _gain_dirs(ant_rx, p_rx, np.asarray(pts, np.float64))

        iso_t = isinstance(ant_tx, Isotropic)
        iso_r = isinstance(ant_rx, Isotropic)
        return [
            RisHopSource(
                1,
                tx_img,
                p_rx,
                w_tx,
                gain_tx=None if iso_t else g_tx_img,
                gain_rx=None if iso_r else g_rx_dir,
            ),
            RisHopSource(
                2,
                p_tx,
                rx_img,
                w_rx,
                gain_tx=None if iso_t else g_tx_dir,
                gain_rx=None if iso_r else g_rx_img,
            ),
            RisHopSource(
                3,
                tx_img,
                rx_img,
                w_both,
                gain_tx=None if iso_t else g_tx_img,
                gain_rx=None if iso_r else g_rx_img,
            ),
        ]

    def describe(self) -> dict[str, Any]:
        """JSON-ready description (manifests)."""
        return {
            "mechanism": "ground",
            "eps_r": float(self.plane.eps_r),
            "sigma_s_per_m": float(self.plane.sigma_s_per_m),
            "polarization": self.plane.polarization,
            "height_m": float(self.plane.height_m),
            "ris_hop_variants": [1, 2, 3] if self.ris_hops_enabled else [],
            "occluders": [o.id for o in self.occluders],
            "edges": [e.id for e in self.edges],
            "assumptions": list(self.assumptions),
        }


def ground_from_spec(spec: Mapping[str, Any]) -> GroundPlane:
    """``environment.ground`` (section 9.2) -> :class:`GroundPlane`."""
    try:
        return GroundPlane(
            eps_r=float(spec["eps_r"]),
            sigma_s_per_m=float(spec["sigma_s_per_m"]),
            polarization=spec.get("polarization", "V"),
            height_m=float(spec.get("height_m", 0.0)),
        )
    except KeyError as exc:
        raise ConfigurationError(f"environment.ground is missing {exc.args[0]!r}") from None


def _scenario_occluders(scenario: Mapping[str, Any]) -> tuple[Occluder, ...]:
    env = scenario.get("environment") or {}
    items: Iterable[Mapping[str, Any]] | None = (
        env.get("occluders") if isinstance(env, Mapping) else None
    )
    return occluders_from_spec(items)


def scenario_edges(scenario: Mapping[str, Any]) -> tuple[KnifeEdge, ...]:
    """The scene's knife edges (``environment.edges``), for the image paths' legs."""
    env = scenario.get("environment") or {}
    items: Iterable[Mapping[str, Any]] | None = (
        env.get("edges") if isinstance(env, Mapping) else None
    )
    return edges_from_spec(items)


def mechanism_from_scenario(
    spec: Mapping[str, Any] | None,
    scenario: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> GroundMechanism | None:
    """D18 factory: the ground mechanism of ``environment.ground`` (None when absent).

    The scene's occluders and knife edges act on the ground path's legs (sections 5.9,
    9.2)."""
    del base_dir  # no files
    if not spec:
        return None
    return GroundMechanism(
        ground_from_spec(spec), _scenario_occluders(scenario), edges=scenario_edges(scenario)
    )
