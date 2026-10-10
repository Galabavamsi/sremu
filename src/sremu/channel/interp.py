"""Snapshot interpolation, the T_u/tile policy, narrowband grouping and ``verify_interp``.

DESIGN.md sections 4.3 (D-1), 4.4 and 4.5. Normative definitions implemented here:

* **D-1 closed forms.** For RX sample ``n`` inside the snapshot interval [t_a, t_b):
  ``w(n) = ((n - n_a,int) - n_a,frac) * inv_len`` with ``n_a = t_a*f_s`` split exactly into
  integer and fractional parts and ``inv_len = 1/((t_b - t_a)*f_s)``; the chord delay is
  ``tau(n) = tau_a + w(n)*(tau_b - tau_a)`` and the carrier phase in cycles is
  ``c(n) = frac(f_c*tau_a) + w(n)*f_c*(tau_b - tau_a)``. ``frac(f_c*tau_a)`` is computed with an
  error-free (Dekker) product, so large absolute phases never lose precision.
  A degenerate interval (``t_b <= t_a``: static scene, event-only snapshots) has ``w = 0``.
* **Error budget** ``eps_rel = 1e-3`` (|dh|/|h|, -60 dB) is the TOTAL interpolation error
  target of PH-18 and ``verify_interp``. Lead decision D1 splits it in half between the two
  rules, ``eps_chord = eps_tile = eps_rel/2``, so the two error sources together stay within
  ``eps_rel``:

  - curvature (chord) ``k*rdd*T_u**2/8 <= eps_chord`` with ``rdd <= v_perp**2/r_min`` gives
    ``T_curv = sqrt(8*(eps_rel/2)*r_min/(k*v_perp**2))``;
  - tile residual midpoint dip ``dphi**2/8 <= eps_tile`` gives
    ``dphi_res <= sqrt(8*eps_rel/2) = 0.0632`` rad, and with
    ``dphi = 2*pi*v_rel*D_t*T_u/(lambda*R_min)`` the tile side
    ``D_t <= sqrt(8*eps_rel/2)*lambda*R_min/(2*pi*v_rel*T_u)``.
* **Policy** (:func:`choose_policy`): ``T_u0 = min(T_curv, 10 ms, scenario value)``; for
  m = 0..4, ``T_u = T_u0/2**m``, tile counts from D_t, cost
  ``c_geo*Q*R/T_u + c_rot*tiles*R*f_s`` (defaults 5 ns, 3 ns); pick the minimum cost among the
  candidates whose tile residual meets ``eps_tile``. Static scenes get one tile per panel and
  event-only snapshots. Panels enter as :class:`PanelPolicyInput`: the contract
  :class:`~sremu.contracts.PanelGeometry` (D11/D17; extents include physical-tile gaps) plus
  the panel's carrier and RIS row count (:func:`panel_policy_inputs` derives them from a
  ``PropagationModel`` snapshot; :func:`choose_policy_for_model` does the whole step).
* **Narrowband test** per link: ``pi*B*dtau <= 0.05`` with dtau the spread over all paths and
  tiles at both interval ends. Narrowband links share one envelope delay (group 0);
  otherwise paths are clustered so that within a cluster ``pi*B*delta <= 0.05``.
* **verify_interp**: ``|h_interp - h_true|/|h_true|`` at the interval midpoint per link, where
  ``h_interp`` is exactly what the renderer produces there and ``h_true`` comes from the exact
  midpoint snapshot. The engine calls it every 64th interval; runs above ``2*eps`` are flagged.

[ENG] choices: T_u is quantized down to a multiple of 1 us (exact ``Fraction``; a bound below
1 us is quantized on a finer grid, never rounded up); candidates
whose tile count is capped by the cell count and whose residual bound then exceeds
``eps_tile = eps_rel/2`` are infeasible.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import (
    FDSpec,
    LinkId,
    LinkJob,
    LinkPaths,
    MotionBounds,
    PanelConfig,
    PanelGeometry,
    PanelId,
    PropagationModel,
    Snapshot,
    TransitionTerm,
    TxRing,
)
from sremu.errors import ConfigurationError
from sremu.timebase import ceil_ticks, parse_exact
from sremu.units import C0

from .fd import DEFAULT_FD_TARGET_DB, choose_fd

__all__ = [
    "C_GEO_S",
    "C_ROT_S",
    "EPS_CHORD",
    "EPS_REL",
    "EPS_TILE",
    "M_MAX",
    "NB_EPS_RAD",
    "T_U_MAX_S",
    "T_U_QUANTUM_S",
    "VERIFY_INTERP_FLAG",
    "VERIFY_INTERP_STRIDE",
    "IntervalGrid",
    "LinkPlan",
    "PanelGeometry",
    "PanelPolicyInput",
    "PolicyCandidate",
    "SnapshotPolicy",
    "TilePlan",
    "choose_policy",
    "choose_policy_for_model",
    "delay_spread_s",
    "double_double",
    "frac_cycles_of_index",
    "frac_of_product",
    "group_envelope_delays",
    "interval_grid",
    "is_narrowband",
    "make_link_job",
    "panel_policy_inputs",
    "plan_groups",
    "plan_link",
    "residual_bound_rel",
    "should_verify",
    "split_product",
    "t_curv_s",
    "tile_side_max_m",
    "two_prod",
    "verify_interp",
    "weights",
]

EPS_REL: Final[float] = 1e-3
"""Total interpolation error target |dh|/|h| (-60 dB): PH-18 and ``verify_interp``."""

EPS_CHORD: Final[float] = EPS_REL / 2.0
"""Share of ``EPS_REL`` given to the chord (curvature) rule, ``eps_rel/2`` (D1)."""

EPS_TILE: Final[float] = EPS_REL / 2.0
"""Share of ``EPS_REL`` given to the tile-residual rule, ``eps_rel/2`` (D1)."""

T_U_MAX_S: Final[Fraction] = Fraction(1, 100)
"""Largest snapshot period, 10 ms."""

T_U_QUANTUM_S: Final[Fraction] = Fraction(1, 1_000_000)
"""T_u is quantized down to a multiple of 1 us [ENG]."""

M_MAX: Final[int] = 4
"""Candidates T_u0/2**m for m = 0..M_MAX."""

C_GEO_S: Final[float] = 5e-9
"""Default geometry cost per element-row and snapshot, s (refit in B1)."""

C_ROT_S: Final[float] = 3e-9
"""Default rotator cost per tile-row and sample, s (refit in B1)."""

NB_EPS_RAD: Final[float] = 0.05
"""Narrowband threshold: pi*B*dtau <= 0.05 rad."""

VERIFY_INTERP_STRIDE: Final[int] = 64
"""``verify_interp`` runs on every 64th snapshot interval."""

VERIFY_INTERP_FLAG: Final[float] = 2.0 * EPS_REL
"""Runs whose ``verify_interp`` maximum exceeds this are flagged."""

_SPLITTER: Final[float] = 134217729.0  # 2**27 + 1 (Veltkamp split)


# --------------------------------------------------------------------------- exact helpers


def two_prod(
    a: float | NDArray[np.float64], b: float | NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Error-free product (Dekker): returns ``(p, e)`` with ``p + e == a*b`` exactly."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    p = a * b
    t = _SPLITTER * a
    ahi = t - (t - a)
    alo = a - ahi
    t = _SPLITTER * b
    bhi = t - (t - b)
    blo = b - bhi
    e = ((ahi * bhi - p) + ahi * blo + alo * bhi) + alo * blo
    return p, e


def frac_of_product(
    a: float | NDArray[np.float64], b: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """``a*b`` reduced to [-0.5, 0.5] cycles, from the exact product (no precision loss)."""
    p, e = two_prod(a, b)
    r = p - np.rint(p)
    c = r + e
    return c - np.rint(c)


def split_product(
    a: float | NDArray[np.float64], b: float | NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Split ``a*b`` into an integer part and a small remainder (``a*b = i + f``, f ~ [0, 1))."""
    p, e = two_prod(a, b)
    i = np.floor(p)
    return i.astype(np.int64), (p - i) + e


def double_double(q: Fraction) -> tuple[float, float]:
    """``q`` as an unevaluated sum ``hi + lo`` of two floats (about 106 bits)."""
    hi = float(q)
    lo = float(q - Fraction(hi))
    return hi, lo


def frac_cycles_of_index(n: NDArray[np.int64], q_hi: float, q_lo: float) -> NDArray[np.float64]:
    """``frac(n*q)`` in [-0.5, 0.5] for absolute indices ``n`` and ``q = q_hi + q_lo`` cycles.

    A pure function of the absolute index: the result does not depend on how a range is
    segmented. ``n`` must satisfy ``|n| < 2**53``.
    """
    nf = np.asarray(n, dtype=np.int64).astype(np.float64)
    p, e = two_prod(nf, q_hi)
    r = p - np.rint(p)
    c = r + (e + nf * q_lo)
    return c - np.rint(c)


# --------------------------------------------------------------------------- D-1 grid


@dataclass(frozen=True)
class IntervalGrid:
    """Exact sample-domain description of a snapshot interval [t_a, t_b) at rate f_s (D-1).

    ``n_a_int``/``n_a_frac``: ``t_a*f_s`` split into integer and fractional parts.
    ``inv_len``: ``1/((t_b - t_a)*f_s)`` (0 for a degenerate interval). ``n_first``/``n_end``:
    the interval's samples are [ceil(t_a f_s), ceil(t_b f_s)); ``n_end`` is None when degenerate.
    """

    n_a_int: int
    n_a_frac: float
    inv_len: float
    n_first: int
    n_end: int | None

    @property
    def degenerate(self) -> bool:
        """True when ``t_b <= t_a`` (w = 0 everywhere)."""
        return self.n_end is None


def interval_grid(t_a: Fraction | int, t_b: Fraction | int, f_s: Fraction | int) -> IntervalGrid:
    """Exact D-1 description of [t_a, t_b) at rate ``f_s`` (all arguments exact)."""
    ta = parse_exact(t_a, name="t_a")
    tb = parse_exact(t_b, name="t_b")
    fs = parse_exact(f_s, name="f_s")
    if fs <= 0:
        raise ConfigurationError(f"sample rate must be > 0, got {fs}")
    na = ta * fs
    n_int = math.floor(na)
    n_frac = float(na - n_int)
    n_first = ceil_ticks(ta, fs)
    if tb <= ta:
        return IntervalGrid(n_int, n_frac, 0.0, n_first, None)
    length = (tb - ta) * fs
    return IntervalGrid(n_int, n_frac, float(1 / length), n_first, ceil_ticks(tb, fs))


def weights(grid: IntervalGrid, n: NDArray[np.int64]) -> NDArray[np.float64]:
    """Interpolation weights ``w(n) = ((n - n_a,int) - n_a,frac)*inv_len`` (D-1)."""
    m = (np.asarray(n, dtype=np.int64) - grid.n_a_int).astype(np.float64)
    return (m - grid.n_a_frac) * grid.inv_len


# --------------------------------------------------------------------------- error budget


def t_curv_s(r_min_m: float, v_perp_mps: float, fc_hz: float, eps: float = EPS_CHORD) -> float:
    """Curvature bound ``T_curv = sqrt(8*eps*r_min/(k*v_perp**2))`` in s (inf if static).

    ``eps`` is the chord share of the budget, by default ``eps_rel/2`` (D1).
    """
    if v_perp_mps <= 0.0:
        return math.inf
    if not r_min_m > 0.0:
        raise ConfigurationError(f"r_min must be > 0 m, got {r_min_m!r}")
    k = 2.0 * math.pi * fc_hz / C0
    return math.sqrt(8.0 * eps * r_min_m / (k * v_perp_mps**2))


def tile_side_max_m(
    r_min_m: float, v_rel_mps: float, fc_hz: float, t_u_s: float, eps: float = EPS_TILE
) -> float:
    """Largest tile side ``D_t = sqrt(8*eps)*lambda*R_min/(2*pi*v_rel*T_u)`` in m.

    ``eps`` is the tile-residual share of the budget, by default ``eps_rel/2`` (D1).
    """
    if v_rel_mps <= 0.0 or t_u_s <= 0.0:
        return math.inf
    lam = C0 / fc_hz
    return math.sqrt(8.0 * eps) * lam * r_min_m / (2.0 * math.pi * v_rel_mps * t_u_s)


def residual_bound_rel(
    tile_side_m: float, r_min_m: float, v_rel_mps: float, fc_hz: float, t_u_s: float
) -> float:
    """Residual midpoint-dip bound ``dphi**2/8`` for a tile of side ``tile_side_m``."""
    lam = C0 / fc_hz
    dphi = 2.0 * math.pi * v_rel_mps * tile_side_m * t_u_s / (lam * r_min_m)
    return dphi * dphi / 8.0


@dataclass(frozen=True)
class PanelPolicyInput:
    """What the T_u/tile policy needs about one RIS panel (D11).

    ``geometry``: the contract :class:`~sremu.contracts.PanelGeometry` of the panel (D17;
    ``PropagationModel.panel_geometry``). The policy uses its outer extents
    (``extent_u_m``/``extent_v_m``, physical-tile gaps included, so the tile side bound is
    conservative), its cell counts and its quadrature points Q (``n_quad``).
    ``fc_hz``: the panel's carrier, Hz: the largest ``link_fc_hz`` of the panel's links.
    ``n_rows``: RIS rows R (``RISCoefficients.n_rows``).
    """

    geometry: PanelGeometry
    fc_hz: float
    n_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, PanelGeometry):
            raise ConfigurationError(
                "PanelPolicyInput.geometry must be a sremu.contracts.PanelGeometry"
            )
        fc = float(self.fc_hz)
        if not (math.isfinite(fc) and fc > 0.0):
            raise ConfigurationError(f"PanelPolicyInput.fc_hz must be > 0, got {self.fc_hz!r}")
        rows = self.n_rows
        if isinstance(rows, bool) or int(rows) != rows or int(rows) < 1:
            raise ConfigurationError(f"PanelPolicyInput.n_rows must be an int >= 1, got {rows!r}")
        object.__setattr__(self, "fc_hz", fc)
        object.__setattr__(self, "n_rows", int(rows))

    @property
    def panel(self) -> PanelId:
        """Panel id."""
        return self.geometry.panel

    @property
    def size_u_m(self) -> float:
        """Outer extent along ``u_axis`` (gaps included), m."""
        return self.geometry.extent_u_m

    @property
    def size_v_m(self) -> float:
        """Outer extent along ``v_axis`` (gaps included), m."""
        return self.geometry.extent_v_m

    @property
    def n_u(self) -> int:
        """Physical cells along ``u_axis``."""
        return self.geometry.n_u

    @property
    def n_v(self) -> int:
        """Physical cells along ``v_axis``."""
        return self.geometry.n_v

    @property
    def q(self) -> int:
        """Quadrature points Q per row and snapshot."""
        return self.geometry.n_quad


def panel_policy_inputs(model: PropagationModel, snap: Snapshot) -> tuple[PanelPolicyInput, ...]:
    """Policy inputs of every panel in ``snap`` (panels without RIS rows are skipped).

    Geometry from ``model.panel_geometry``; carrier = the largest ``model.link_fc_hz`` over the
    links of the panel's rows; R = ``RISCoefficients.n_rows``.
    """
    out = []
    for pid, rc in snap.ris.items():
        if rc.n_rows == 0:
            continue
        fc = max(model.link_fc_hz(lk) for lk, _variant in rc.rows)
        out.append(PanelPolicyInput(model.panel_geometry(pid), fc, rc.n_rows))
    return tuple(out)


@dataclass(frozen=True)
class TilePlan:
    """Tile grid of one panel: ``tiles_u x tiles_v`` and the bound it achieves."""

    panel: PanelId
    tiles_u: int
    tiles_v: int
    tile_side_max_m: float
    residual_bound_rel: float

    @property
    def n_tiles(self) -> int:
        """Tiles per row K."""
        return self.tiles_u * self.tiles_v


@dataclass(frozen=True)
class PolicyCandidate:
    """One evaluated (T_u, tiles) candidate."""

    m: int
    t_u_s: Fraction
    tiles: Mapping[PanelId, TilePlan]
    cost: float
    feasible: bool


@dataclass(frozen=True)
class SnapshotPolicy:
    """Chosen snapshot period and tiles (logged in the manifest, section 3.6).

    ``t_u_s``: snapshot period, s (exact), or None for a static scene (event-only snapshots).
    ``t_curv_s``: the curvature bound for the chord share ``eps_rel/2``, s.
    ``curvature_bound_rel``/``residual_bound_rel``: bounds achieved by the choice (each at
    most ``eps_rel/2`` when ``feasible``); :attr:`total_bound_rel` is their sum (at most
    ``eps_rel``, D1). ``feasible`` is False when no candidate met the tile budget (the
    smallest T_u is used). ``eps_rel``: the total budget the policy was run with.
    """

    t_u_s: Fraction | None
    m: int
    tiles: Mapping[PanelId, TilePlan]
    cost: float
    t_curv_s: float
    curvature_bound_rel: float
    residual_bound_rel: float
    feasible: bool = True
    candidates: tuple[PolicyCandidate, ...] = field(default=())
    eps_rel: float = EPS_REL

    @property
    def total_bound_rel(self) -> float:
        """Chord plus tile-residual bound (D1: at most ``eps_rel`` when feasible)."""
        return self.curvature_bound_rel + self.residual_bound_rel

    def to_manifest(self) -> dict[str, object]:
        """JSON-ready summary (exact times as ``p/q`` strings)."""
        return {
            "t_u_s": None
            if self.t_u_s is None
            else f"{self.t_u_s.numerator}/{self.t_u_s.denominator}",
            "m": self.m,
            "tiles": {p: [tp.tiles_u, tp.tiles_v] for p, tp in self.tiles.items()},
            "cost": self.cost,
            "t_curv_s": self.t_curv_s,
            "eps_rel": self.eps_rel,
            "eps_chord": self.eps_rel / 2.0,
            "eps_tile": self.eps_rel / 2.0,
            "curvature_bound_rel": self.curvature_bound_rel,
            "residual_bound_rel": self.residual_bound_rel,
            "total_bound_rel": self.total_bound_rel,
            "feasible": self.feasible,
        }


def _quantize_down(t_s: float, quantum: Fraction) -> Fraction:
    """Largest multiple of ``quantum`` that does not exceed ``t_s`` (s, exact).

    A bound below one quantum is quantized on 1000x finer grids until it spans at least 1000
    steps (at most 0.1 % below the bound), so the result never exceeds ``t_s``: rounding
    T_curv *up* would break the curvature budget.
    """
    exact = Fraction(t_s)
    if not exact > 0:
        raise ConfigurationError(f"snapshot period bound must be > 0 s, got {t_s!r}")
    q = quantum
    steps = math.floor(exact / q)
    if steps >= 1:
        return q * steps
    while steps < 1000:
        q /= 1000
        steps = math.floor(exact / q)
    return q * steps


def choose_policy(
    bounds: MotionBounds,
    panels: Sequence[PanelPolicyInput],
    link_fc_hz: Mapping[LinkId, float],
    f_s: Fraction | int | float,
    *,
    eps: float = EPS_REL,
    t_u_max_s: Fraction = T_U_MAX_S,
    t_u_scenario_s: Fraction | None = None,
    c_geo_s: float = C_GEO_S,
    c_rot_s: float = C_ROT_S,
    m_max: int = M_MAX,
) -> SnapshotPolicy:
    """Choose ``T_u`` and the tile grids (section 4.4 policy with the D1 budget split).

    ``bounds``: the model's motion bounds. ``panels``: one :class:`PanelPolicyInput` per
    panel. ``link_fc_hz``: carrier of each link (for the non-RIS curvature bound). ``f_s``:
    the RX rate used in the cost. ``t_u_scenario_s``: a scenario value, which may only lower
    T_u. ``eps``: the TOTAL budget ``eps_rel``; the chord and tile rules each get ``eps/2``.
    """
    fs = float(f_s)
    for p in panels:
        if not isinstance(p, PanelPolicyInput):
            raise ConfigurationError(
                "choose_policy: panels must be PanelPolicyInput (contracts.PanelGeometry plus "
                f"carrier and row count), got {type(p).__name__}"
            )
    if not (math.isfinite(eps) and eps > 0.0):
        raise ConfigurationError(f"eps_rel must be > 0, got {eps!r}")
    eps_chord = eps / 2.0  # D1
    eps_tile = eps / 2.0  # D1
    if bounds.static:
        tiles = {p.panel: TilePlan(p.panel, 1, 1, math.inf, 0.0) for p in panels}
        return SnapshotPolicy(None, 0, tiles, 0.0, math.inf, 0.0, 0.0, True, (), eps)
    tc = math.inf
    for lk, fc in link_fc_hz.items():
        v = bounds.v_perp_max_mps.get(lk, bounds.v_rel_max_mps.get(lk, 0.0))
        r = bounds.r_min_m.get(lk, math.inf)
        if v > 0.0 and math.isfinite(r):
            tc = min(tc, t_curv_s(r, v, fc, eps_chord))
    for p in panels:
        v = bounds.panel_v_rel_max_mps.get(p.panel, 0.0)
        r = bounds.panel_r_min_m.get(p.panel, math.inf)
        if v > 0.0 and math.isfinite(r):
            tc = min(tc, t_curv_s(r, v, p.fc_hz, eps_chord))
    t0 = t_u_max_s
    if math.isfinite(tc):
        t0 = min(t0, _quantize_down(tc, T_U_QUANTUM_S))
    if t_u_scenario_s is not None:
        t0 = min(t0, parse_exact(t_u_scenario_s, name="t_u_scenario_s"))
    if t0 <= 0:
        raise ConfigurationError(f"snapshot period must be > 0, got {t0}")

    cands: list[PolicyCandidate] = []
    for m in range(m_max + 1):
        tu = t0 / (2**m)
        tus = float(tu)
        plans: dict[PanelId, TilePlan] = {}
        cost = 0.0
        feasible = True
        for p in panels:
            v = bounds.panel_v_rel_max_mps.get(p.panel, 0.0)
            r = bounds.panel_r_min_m.get(p.panel, math.inf)
            dt = tile_side_max_m(r, v, p.fc_hz, tus, eps_tile) if math.isfinite(r) else math.inf
            if math.isinf(dt):
                nu = nv = 1
                rb = 0.0
            else:
                nu = min(max(1, math.ceil(p.size_u_m / dt)), p.n_u)
                nv = min(max(1, math.ceil(p.size_v_m / dt)), p.n_v)
                side = max(p.size_u_m / nu, p.size_v_m / nv)
                rb = residual_bound_rel(side, r, v, p.fc_hz, tus) if v > 0.0 else 0.0
                if rb > eps_tile * (1.0 + 1e-12):
                    feasible = False
            plans[p.panel] = TilePlan(p.panel, nu, nv, dt, rb)
            cost += c_geo_s * p.q * p.n_rows / tus + c_rot_s * nu * nv * p.n_rows * fs
        cands.append(PolicyCandidate(m, tu, plans, cost, feasible))
    feasible_c = [c for c in cands if c.feasible]
    if feasible_c:
        best = min(feasible_c, key=lambda c: (c.cost, c.m))
    else:
        best = cands[-1]
    tus = float(best.t_u_s)
    curv = 0.0
    for lk, fc in link_fc_hz.items():
        v = bounds.v_perp_max_mps.get(lk, bounds.v_rel_max_mps.get(lk, 0.0))
        r = bounds.r_min_m.get(lk, math.inf)
        if v > 0.0 and math.isfinite(r):
            curv = max(curv, (2 * math.pi * fc / C0) * v * v / r * tus * tus / 8.0)
    for p in panels:
        v = bounds.panel_v_rel_max_mps.get(p.panel, 0.0)
        r = bounds.panel_r_min_m.get(p.panel, math.inf)
        if v > 0.0 and math.isfinite(r):
            curv = max(curv, (2 * math.pi * p.fc_hz / C0) * v * v / r * tus * tus / 8.0)
    resid = max((tp.residual_bound_rel for tp in best.tiles.values()), default=0.0)
    return SnapshotPolicy(
        best.t_u_s,
        best.m,
        best.tiles,
        best.cost,
        tc,
        curv,
        resid,
        best.feasible,
        tuple(cands),
        eps,
    )


def choose_policy_for_model(
    model: PropagationModel,
    f_s: Fraction | int | float,
    *,
    t_s: Fraction | int = 0,
    eps: float = EPS_REL,
    t_u_max_s: Fraction = T_U_MAX_S,
    t_u_scenario_s: Fraction | None = None,
    c_geo_s: float = C_GEO_S,
    c_rot_s: float = C_ROT_S,
    m_max: int = M_MAX,
) -> SnapshotPolicy:
    """:func:`choose_policy` for a ``PropagationModel``: its motion bounds, the panel inputs
    of its snapshot at ``t_s`` (:func:`panel_policy_inputs`) and every link's carrier.

    The keyword arguments are those of :func:`choose_policy`. Apply the result with
    ``model.with_render_tiles({p: (tp.tiles_u, tp.tiles_v) for p, tp in pol.tiles.items()})``.
    """
    t = parse_exact(t_s, name="t_s")
    panels = panel_policy_inputs(model, model.snapshot(t))
    fcs = {lk: model.link_fc_hz(lk) for lk in model.links()}
    return choose_policy(
        model.motion_bounds(),
        panels,
        fcs,
        f_s,
        eps=eps,
        t_u_max_s=t_u_max_s,
        t_u_scenario_s=t_u_scenario_s,
        c_geo_s=c_geo_s,
        c_rot_s=c_rot_s,
        m_max=m_max,
    )


# --------------------------------------------------------------------------- narrowband


def delay_spread_s(tau_a: NDArray[np.float64], tau_b: NDArray[np.float64]) -> float:
    """Delay spread over all paths at both interval ends, s (0 for fewer than two paths)."""
    if tau_a.size == 0:
        return 0.0
    return float(max(np.ptp(tau_a), np.ptp(tau_b)))


def is_narrowband(occupied_bw_hz: float, spread_s: float, eps: float = NB_EPS_RAD) -> bool:
    """Narrowband test ``pi*B*dtau <= eps`` (section 4.5)."""
    return math.pi * float(occupied_bw_hz) * float(spread_s) <= eps


def plan_groups(
    tau_a: NDArray[np.float64],
    tau_b: NDArray[np.float64],
    occupied_bw_hz: float,
    eps: float = NB_EPS_RAD,
) -> tuple[NDArray[np.int16], bool]:
    """FD envelope groups of a link's paths: ``(group, narrowband)``.

    Narrowband links put every path in group 0. Otherwise paths are sorted by mid-interval
    delay (stable by index) and clustered greedily so that each cluster's spread satisfies
    ``pi*B*delta <= eps`` at both interval ends; groups are numbered by increasing delay.
    """
    tau_a = np.asarray(tau_a, dtype=np.float64)
    tau_b = np.asarray(tau_b, dtype=np.float64)
    n = tau_a.shape[0]
    group = np.zeros(n, dtype=np.int16)
    if n == 0 or is_narrowband(occupied_bw_hz, delay_spread_s(tau_a, tau_b), eps):
        return group, True
    delta = eps / (math.pi * float(occupied_bw_hz))
    order = np.argsort(0.5 * (tau_a + tau_b), kind="stable")
    g = 0
    lo_a = hi_a = tau_a[order[0]]
    lo_b = hi_b = tau_b[order[0]]
    for idx in order:
        a, b = tau_a[idx], tau_b[idx]
        na_lo, na_hi = min(lo_a, a), max(hi_a, a)
        nb_lo, nb_hi = min(lo_b, b), max(hi_b, b)
        if na_hi - na_lo > delta or nb_hi - nb_lo > delta:
            g += 1
            if g > np.iinfo(np.int16).max:
                raise ConfigurationError("too many FD groups for int16")
            na_lo = na_hi = a
            nb_lo = nb_hi = b
        lo_a, hi_a, lo_b, hi_b = na_lo, na_hi, nb_lo, nb_hi
        group[idx] = g
    return group, False


def group_envelope_delays(
    tau: NDArray[np.float64], group: NDArray[np.int16]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Group labels (sorted) and each group's envelope delay ``(min + max)/2`` in s."""
    grp = np.asarray(group, dtype=np.int64)
    if grp.size and int(grp.min()) == int(grp.max()):  # one group (every narrowband link)
        env = 0.5 * (float(np.min(tau)) + float(np.max(tau)))
        return np.array([int(grp[0])], dtype=np.int64), np.array([env], dtype=np.float64)
    labels = np.unique(grp)
    out = np.empty(labels.shape[0], dtype=np.float64)
    for i, g in enumerate(labels):
        sel = tau[group == g]
        out[i] = 0.5 * (float(sel.min()) + float(sel.max()))
    return labels, out


@dataclass(frozen=True)
class LinkPlan:
    """Per-link rendering decision: groups, narrowband flag, FD filter and delay spread."""

    narrowband: bool
    group: NDArray[np.int16]
    fd: FDSpec
    spread_s: float


def plan_link(
    paths_a: LinkPaths,
    paths_b: LinkPaths,
    occupied_bw_hz: float,
    f_s: Fraction | int | float,
    *,
    fd_target_db: float = DEFAULT_FD_TARGET_DB,
) -> LinkPlan:
    """Narrowband/wideband decision, FD groups and FD filter for one link and interval."""
    if paths_a.n_paths != paths_b.n_paths:
        raise ConfigurationError(
            f"link {paths_a.link}: path count changes inside the interval "
            f"({paths_a.n_paths} -> {paths_b.n_paths})"
        )
    group, nb = plan_groups(paths_a.delay_s, paths_b.delay_s, occupied_bw_hz)
    fd = choose_fd(occupied_bw_hz, f_s, fd_target_db)
    return LinkPlan(nb, group, fd, delay_spread_s(paths_a.delay_s, paths_b.delay_s))


def make_link_job(
    link: LinkId,
    ring: TxRing,
    fc_hz: float,
    lo_offset_hz: float,
    paths_a: LinkPaths,
    paths_b: LinkPaths,
    plan: LinkPlan,
    trans: Sequence[TransitionTerm] = (),
) -> LinkJob:
    """Build a ``LinkJob`` from ``combine()`` outputs at the interval ends."""
    return LinkJob(
        link=link,
        ring=ring,
        fc_hz=float(fc_hz),
        lo_offset_hz=float(lo_offset_hz),
        amp_a=paths_a.amp,
        amp_b=paths_b.amp,
        tau_a=paths_a.delay_s,
        tau_b=paths_b.delay_s,
        group=plan.group,
        fd=plan.fd,
        trans=list(trans),
    )


# --------------------------------------------------------------------------- verify_interp


def should_verify(interval_index: int, stride: int = VERIFY_INTERP_STRIDE) -> bool:
    """Whether snapshot interval ``interval_index`` is checked (every 64th)."""
    return interval_index % stride == 0


def _channel(amp: NDArray[np.complex128], cycles: NDArray[np.float64]) -> complex:
    return complex(np.sum(amp * np.exp(-2j * np.pi * cycles)))


def verify_interp(
    model: PropagationModel,
    cfg: Mapping[PanelId, PanelConfig],
    t_a: Fraction,
    t_b: Fraction,
    fc_hz: Mapping[LinkId, float],
    *,
    snap_a: Snapshot | None = None,
    snap_b: Snapshot | None = None,
    links: Sequence[LinkId] | None = None,
    kinds: Sequence[int] | None = None,
) -> dict[LinkId, float]:
    """Midpoint interpolation error ``|h_interp - h_true|/|h_true|`` per link.

    ``h_interp`` is the renderer's channel at the interval midpoint (w = 1/2: complex-mean
    amplitudes, chord delays, exact carrier phase); ``h_true`` uses the exact midpoint
    snapshot (all elements). ``fc_hz`` gives each link's carrier. ``kinds`` optionally
    restricts both to paths of the given ``PathKind`` values (e.g. RIS tiles only; the
    default is the whole channel, as logged in the manifest). Returns 0 when both are 0.
    """
    ta = parse_exact(t_a, name="t_a")
    tb = parse_exact(t_b, name="t_b")
    sa = snap_a if snap_a is not None else model.snapshot(ta)
    sb = snap_b if snap_b is not None else model.snapshot(tb)
    sm = model.snapshot((ta + tb) / 2)
    pa = model.combine(sa, cfg)
    pb = model.combine(sb, cfg)
    pm = model.combine(sm, cfg)
    out: dict[LinkId, float] = {}
    for lk in links if links is not None else tuple(pa):
        fc = float(fc_hz[lk])
        a, b, m = pa[lk], pb[lk], pm[lk]
        if kinds is not None:
            kk = np.asarray(kinds, dtype=np.int8)
            sa_, sb_, sm_ = np.isin(a.kind, kk), np.isin(b.kind, kk), np.isin(m.kind, kk)
            a = LinkPaths(a.link, a.t, a.amp[sa_], a.delay_s[sa_], a.kind[sa_], a.tile[sa_])
            b = LinkPaths(b.link, b.t, b.amp[sb_], b.delay_s[sb_], b.kind[sb_], b.tile[sb_])
            m = LinkPaths(m.link, m.t, m.amp[sm_], m.delay_s[sm_], m.kind[sm_], m.tile[sm_])
        ca = frac_of_product(fc, a.delay_s)
        dc = fc * (b.delay_s - a.delay_s)
        h_int = _channel(a.amp + 0.5 * (b.amp - a.amp), ca + 0.5 * dc)
        h_true = _channel(m.amp, frac_of_product(fc, m.delay_s))
        diff = abs(h_int - h_true)
        if h_true == 0:
            out[lk] = 0.0 if diff == 0 else math.inf
        else:
            out[lk] = diff / abs(h_true)
    return out
