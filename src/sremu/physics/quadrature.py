"""Cell model and aperture refinement (DESIGN.md section 5.4; WP2).

``cell_model`` in {"auto", "point", "po_patch"} per panel:

* ``point``: one quadrature point per physical cell (the model Tang validated).
* ``po_patch``: each cell is split into ``ceil(p_u/Delta) x ceil(p_v/Delta)`` sub-cells with
  ``Delta <= subcell_max_wavelengths * lambda`` (default lambda/5), sharing the parent's Gamma.
* ``auto``: ``point`` when ``max(p_u, p_v) <= point_max_pitch_wavelengths * lambda``
  (default lambda/2), ``po_patch`` otherwise.

Forcing ``point`` with a pitch above lambda/2 warns (``ValidityWarning``). Tables with
``includes_cell_pattern=True`` force ``point`` (the pattern is not counted twice).

Richardson check (``po_patch`` only): on up to 8 representative links (nearest, farthest,
most oblique incidence, most oblique reflection, four pseudo-random ones keyed through
``sremu.rng``, purpose ``trial``, name ``richardson/<panel>/<link>``, D14), the received power is
computed at ``Delta`` and ``Delta/2``. The power metric is evaluated for two configuration-
independent cases: all cells with Gamma = 1 (``P_uni = |sum_n b_n|^2``) and the coherent
(focused) configuration (``P_foc = (sum_n |b_n|)^2``); ``P_uni`` is skipped when it lies
more than 60 dB below ``P_foc`` (a numerical null). If the largest change exceeds
``richardson_tol_db`` (0.05 dB), ``Delta`` is halved, at most twice; then
``RefinementNotConverged`` is raised. The accepted ``Delta`` and its estimate are recorded.

Model form (section 5.4; PH-04, PH-26). The two cell models are not numerically equivalent:
a ``po_patch`` cell carries the sub-aperture factor of its uniform patch, which ``point``
omits. Near specular they agree (PH-26: <= 0.05 dB). For a lambda/2 cell at 45 deg retro the
continuous factor ``sinc(p (sin theta_i + sin theta_s)/lambda)`` is -8.92 dB (the Delta -> 0
limit); the shipped Delta <= lambda/5 (3 x 3 sub-cells) gives the 3-point midpoint value
``(1 + 2 cos(pi sqrt(2)/3))/3`` = -8.11 dB (D6 regression value, single angle). At finite
range the angles vary across the panel and the factor is that of each cell's local angles.
For electrically large cells the Richardson-converged ``po_patch`` value is the reported one
(D5: 1 m^2 plate, 3.5 GHz, 5 m near anomalous, -88.128 dB at Delta = lambda/40, +-0.10 dB).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from sremu import rng as _rng
from sremu.errors import ConfigurationError, RefinementNotConverged, ValidityWarning
from sremu.physics.antennas import Antenna, Isotropic
from sremu.physics.geometry import PanelGrid
from sremu.physics.ris_kernel import Backend, Obliquity, accumulate_cells, k1_terms

__all__ = [
    "RICHARDSON_RNG_PURPOSE",
    "CellModel",
    "LinkSample",
    "Quadrature",
    "RichardsonResult",
    "build_quadrature",
    "gap_quadrature",
    "representative_links",
    "resolve_cell_model",
    "richardson_check",
    "richardson_link_key",
    "subdivisions",
]

CellModel = Literal["auto", "point", "po_patch"]
_NULL_RATIO = 1e-6  # P_uni below -60 dB of P_foc is treated as a numerical null


@dataclass(frozen=True, eq=False)
class Quadrature:
    """Quadrature points of one panel.

    ``points`` (Q, 3) m; ``areas`` (Q,) m^2 (pitch areas; per cell they sum to the cell's
    pitch area); ``parent`` (Q,) int64 physical cell of each point, in parent-major order
    (all sub-cells of cell 0 first); ``m_u``/``m_v`` sub-cells per cell; ``delta_m`` the
    largest sub-cell side; ``level`` the number of Richardson halvings applied.
    """

    points: np.ndarray
    areas: np.ndarray
    parent: np.ndarray
    n_cells: int
    cell_model: Literal["point", "po_patch"]
    m_u: int
    m_v: int
    delta_m: float
    level: int = 0

    @property
    def n_points(self) -> int:
        """Number of quadrature points Q."""
        return int(self.points.shape[0])


def resolve_cell_model(
    requested: str,
    pitch_u_m: float,
    pitch_v_m: float,
    wavelength_m: float,
    *,
    point_max_pitch_wavelengths: float = 0.5,
    includes_cell_pattern: bool = False,
) -> Literal["point", "po_patch"]:
    """Resolve ``auto`` to ``point`` or ``po_patch`` (section 5.4)."""
    if requested not in ("auto", "point", "po_patch"):
        raise ConfigurationError(f"unknown cell_model {requested!r}")
    pitch = max(pitch_u_m, pitch_v_m)
    big = pitch > point_max_pitch_wavelengths * wavelength_m
    if includes_cell_pattern:
        if requested == "po_patch":
            raise ConfigurationError(
                "tables with includes_cell_pattern=True force cell_model='point'"
            )
        return "point"
    if requested == "auto":
        return "po_patch" if big else "point"
    if requested == "point" and big:
        warnings.warn(
            f"cell_model='point' forced with pitch {pitch:.4g} m > "
            f"{point_max_pitch_wavelengths} lambda = "
            f"{point_max_pitch_wavelengths * wavelength_m:.4g} m: unsubdivided cells larger "
            "than lambda/2 are outside the validity domain",
            ValidityWarning,
            stacklevel=2,
        )
    return requested  # type: ignore[return-value]


def subdivisions(
    pitch_u_m: float,
    pitch_v_m: float,
    wavelength_m: float,
    *,
    subcell_max_wavelengths: float = 0.2,
    level: int = 0,
) -> tuple[int, int]:
    """Sub-cells per cell ``(ceil(p_u/Delta), ceil(p_v/Delta))``, ``Delta = s lambda / 2^level``."""
    delta = subcell_max_wavelengths * wavelength_m / (2**level)
    # a tiny relative slack keeps exact multiples (p = 2 Delta) from rounding up
    mu = max(1, math.ceil(pitch_u_m / delta * (1 - 1e-12)))
    mv = max(1, math.ceil(pitch_v_m / delta * (1 - 1e-12)))
    return mu, mv


def build_quadrature(
    grid: PanelGrid,
    cell_model: Literal["point", "po_patch"],
    wavelength_m: float,
    *,
    subcell_max_wavelengths: float = 0.2,
    level: int = 0,
) -> Quadrature:
    """Quadrature of a panel for a resolved cell model."""
    centres = grid.cell_local()
    n = grid.n_cells
    if cell_model == "point":
        pts = grid.cell_centres()
        return Quadrature(
            points=pts,
            areas=np.full(n, grid.cell_area_m2),
            parent=np.arange(n, dtype=np.int64),
            n_cells=n,
            cell_model="point",
            m_u=1,
            m_v=1,
            delta_m=max(grid.pitch_u_m, grid.pitch_v_m),
            level=0,
        )
    if cell_model != "po_patch":
        raise ConfigurationError(f"build_quadrature: unresolved cell model {cell_model!r}")
    mu, mv = subdivisions(
        grid.pitch_u_m,
        grid.pitch_v_m,
        wavelength_m,
        subcell_max_wavelengths=subcell_max_wavelengths,
        level=level,
    )
    ou = ((np.arange(mu) + 0.5) / mu - 0.5) * grid.pitch_u_m
    ov = ((np.arange(mv) + 0.5) / mv - 0.5) * grid.pitch_v_m
    gv, gu = np.meshgrid(ov, ou, indexing="ij")
    su, sv = gu.ravel(), gv.ravel()  # (mu*mv,) offsets, v-major like cells
    lu = (centres[:, 0][:, None] + su[None, :]).ravel()
    lv = (centres[:, 1][:, None] + sv[None, :]).ravel()
    pts = grid.pose.to_world(lu, lv)
    m = mu * mv
    return Quadrature(
        points=pts,
        areas=np.full(n * m, grid.cell_area_m2 / m),
        parent=np.repeat(np.arange(n, dtype=np.int64), m),
        n_cells=n,
        cell_model="po_patch",
        m_u=mu,
        m_v=mv,
        delta_m=max(grid.pitch_u_m / mu, grid.pitch_v_m / mv),
        level=level,
    )


def gap_quadrature(
    grid: PanelGrid, wavelength_m: float, *, subcell_max_wavelengths: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature points (G, 3) and areas (G,) covering the inter-tile gaps (po_patch)."""
    delta = subcell_max_wavelengths * wavelength_m
    pts, areas = [], []
    for u0, u1, v0, v1 in grid.gap_strips():
        mu = max(1, math.ceil((u1 - u0) / delta))
        mv = max(1, math.ceil((v1 - v0) / delta))
        uu = u0 + (np.arange(mu) + 0.5) * (u1 - u0) / mu
        vv = v0 + (np.arange(mv) + 0.5) * (v1 - v0) / mv
        gv, gu = np.meshgrid(vv, uu, indexing="ij")
        pts.append(grid.pose.to_world(gu.ravel(), gv.ravel()))
        areas.append(np.full(mu * mv, (u1 - u0) * (v1 - v0) / (mu * mv)))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(pts), np.concatenate(areas)


@dataclass(frozen=True)
class LinkSample:
    """A (Tx, Rx) geometry used by the Richardson check."""

    name: str
    p_tx: np.ndarray
    p_rx: np.ndarray
    fc_hz: float
    ant_tx: Antenna = field(default_factory=Isotropic)
    ant_rx: Antenna = field(default_factory=Isotropic)


RICHARDSON_RNG_PURPOSE = "trial"
"""Keyed purpose of the pseudo-random Richardson link choice (D14; name
``richardson/<panel>/<link>``, i.e. study ``richardson``, trial ``<panel>/<link>``)."""


def richardson_link_key(seed: int, panel: str, link: str) -> int:
    """64-bit ``sremu.rng`` key of one link for the pseudo-random part of the Richardson
    choice (D14): ``derive_key(seed, "trial", "richardson/<panel>/<link>").u64``. A pure
    function of the link's own name, so the choice does not depend on link order."""
    return _rng.derive_key(int(seed), RICHARDSON_RNG_PURPOSE, f"richardson/{panel}/{link}").u64


def representative_links(
    samples: Sequence[LinkSample],
    grid: PanelGrid,
    n: int = 8,
    *,
    seed: int = 0,
    panel: str = "",
) -> list[int]:
    """Indices of up to ``n`` representative links: nearest, farthest, most oblique
    incidence, most oblique reflection, then pseudo-random ones in the order of
    :func:`richardson_link_key` (``sremu.rng`` derivation, D14)."""
    if not samples:
        return []
    c = grid.pose.center_m
    nrm = grid.pose.normal
    lens, ci, cs = [], [], []
    for s in samples:
        di, ds = s.p_tx - c, s.p_rx - c
        ri, rs = float(np.linalg.norm(di)), float(np.linalg.norm(ds))
        lens.append(ri + rs)
        ci.append(float(di @ nrm) / ri)
        cs.append(float(ds @ nrm) / rs)
    order: list[int] = []
    for idx in (
        int(np.argmin(lens)),
        int(np.argmax(lens)),
        int(np.argmin(np.where(np.asarray(ci) > 0, ci, np.inf))) if any(x > 0 for x in ci) else 0,
        int(np.argmin(np.where(np.asarray(cs) > 0, cs, np.inf))) if any(x > 0 for x in cs) else 0,
    ):
        if idx not in order:
            order.append(idx)
    rest = sorted(
        (i for i in range(len(samples)) if i not in order),
        key=lambda i: (richardson_link_key(seed, panel, samples[i].name), i),
    )
    order.extend(rest[: max(0, n - len(order))])
    return order[:n]


@dataclass(frozen=True)
class RichardsonResult:
    """Outcome of the Richardson check: accepted ``level`` (halvings), the estimate
    ``estimate_db`` (largest |Delta P| at the accepted level) and per-level history."""

    level: int
    estimate_db: float
    history_db: tuple[float, ...]
    links: tuple[str, ...]


def _cell_coeffs(
    grid: PanelGrid, quad: Quadrature, s: LinkSample, obliquity: Obliquity, backend: Backend
) -> np.ndarray:
    pts = quad.points

    def gains(ant: Antenna, src: np.ndarray) -> np.ndarray | None:
        if isinstance(ant, Isotropic):
            return None
        d = pts - src[None, :]
        return np.asarray(ant.gain(d / np.linalg.norm(d, axis=1)[:, None]))[None, :]

    terms = k1_terms(
        s.p_tx,
        s.p_rx,
        pts,
        quad.areas,
        grid.pose.normal,
        s.fc_hz,
        gain_tx=gains(s.ant_tx, s.p_tx),
        gain_rx=gains(s.ant_rx, s.p_rx),
        obliquity=obliquity,
        backend=backend,
    )
    return accumulate_cells(terms, quad.parent, quad.n_cells, backend=backend)[0, 0]


def _metric_db(b0: np.ndarray, b1: np.ndarray) -> float:
    foc0, foc1 = float(np.sum(np.abs(b0))) ** 2, float(np.sum(np.abs(b1))) ** 2
    if foc0 == 0.0 and foc1 == 0.0:
        return 0.0
    worst = abs(10 * math.log10(foc0 / foc1))
    uni0, uni1 = abs(complex(np.sum(b0))) ** 2, abs(complex(np.sum(b1))) ** 2
    if uni1 >= _NULL_RATIO * foc1 and uni0 > 0.0:
        worst = max(worst, abs(10 * math.log10(uni0 / uni1)))
    return worst


def richardson_check(
    grid: PanelGrid,
    wavelength_m: float,
    samples: Sequence[LinkSample],
    *,
    subcell_max_wavelengths: float = 0.2,
    tol_db: float = 0.05,
    max_halvings: int = 2,
    obliquity: Obliquity = "sqrt",
    backend: Backend = "numpy_ref",
    seed: int = 0,
    panel: str = "",
) -> RichardsonResult:
    """Richardson refinement check for a ``po_patch`` panel (section 5.4). ``seed`` and
    ``panel`` key the pseudo-random link choice (:func:`representative_links`, D14)."""
    idx = representative_links(samples, grid, seed=seed, panel=panel)
    chosen = [samples[i] for i in idx]
    if not chosen:
        return RichardsonResult(0, 0.0, (), ())
    history: list[float] = []
    prev = [
        _cell_coeffs(
            grid,
            build_quadrature(
                grid,
                "po_patch",
                wavelength_m,
                subcell_max_wavelengths=subcell_max_wavelengths,
                level=0,
            ),
            s,
            obliquity,
            backend,
        )
        for s in chosen
    ]
    for level in range(max_halvings + 1):
        quad = build_quadrature(
            grid,
            "po_patch",
            wavelength_m,
            subcell_max_wavelengths=subcell_max_wavelengths,
            level=level + 1,
        )
        nxt = [_cell_coeffs(grid, quad, s, obliquity, backend) for s in chosen]
        est = max(_metric_db(a, b) for a, b in zip(prev, nxt, strict=True))
        history.append(est)
        if est <= tol_db:
            return RichardsonResult(level, est, tuple(history), tuple(s.name for s in chosen))
        prev = nxt
    raise RefinementNotConverged(
        f"po_patch Richardson check: |dP| = {history[-1]:.3f} dB > {tol_db} dB after "
        f"{max_halvings} halvings (history {[f'{h:.3f}' for h in history]} dB)"
    )
