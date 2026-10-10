"""Built-in controllers (DESIGN.md section 8.7; WP9).

| Controller | Feedback | Behaviour |
|---|---|---|
| :class:`Fixed` | none | constant configuration (sent once) |
| :class:`RisAbsent` | none | ``enabled = False`` (the "RIS absent" baseline, sent once) |
| :class:`Urap` | none | WCL uniform random activation, redrawn each period |
| :class:`WclHeuristic` | positions | WCL letter Eqs. (7)-(8), (Theta, psi) grid search |
| :class:`ExactOptimum` | positions + model | exact finite-alphabet optimum (section 5.13) |
| :class:`GreedyQuantized` | positions + model | phase-align to the direct path, then quantize |
| :class:`IdealDesigned` | positions + model | design for an ideal alphabet, deploy the indices |
| :class:`CodebookSweep` | RX power | cyclic sweep over a codebook |
| :class:`ProbeBandit` | RX power | UCB1 over a codebook, reward = windowed RX power |
| :class:`GeneticAlgorithm` | positions + model | GA (W4 comparison only) |
| :class:`TdmaAlternate` | packet schedule | alternate configurations, one switch per guard |
| :class:`DelayedController` | wrapper | position updates every tau_SUI (age ``report_delay_s``) |

Randomness: every random draw uses the generator handed to ``setup`` (purpose
``ctrl/<name>``, ``sremu.rng``; D14), so decisions are reproducible given the seed.

**Position-driven controllers** evaluate the channel on their own model of the scene
(:mod:`sremu.control.controller`: ``SnapshotChannel``, ``GridChannel``, ``FunctionChannel``)
at the reported positions, which refer to ``t_b - position_delay_s``. Without a channel
model they use ``obs.csi_oracle`` (the engine flags ``oracle_csi_used``). The panel's own
cascade is optimized; other panels stay at their observed states.

**WCL heuristic (letter Eqs. 7-8), as implemented.** For a binary alphabet with an OFF
state ``off`` and an ON state ``on``: ``base = h_direct + sum_n Z[off, n]`` and the
steerable phasors ``d_n = Z[on, n] - Z[off, n]``. The relative phase of element n is
``delta_n = arg(base) - arg(d_n)``, which is ``phi_n(x) - phi(x)`` of the letter when the
phasors are ``a exp(-j phi)`` (``phase_source="phasor"``, default), or the letter's
path-length phases ``2 pi f_c (D_n(x) - d_LOS(x))/c`` (``phase_source="path_length"``,
which ignores the K1 factor j). The activation set (Eq. 7) is read as

    A(psi, Theta) = {n : (delta_n)_{2pi} < Theta - psi  or  (-delta_n)_{2pi} < psi},

the arc of width Theta containing ``phi(x)`` at distance ``psi`` from one edge, ``(a)_{2pi}``
being ``a`` reduced to [0, 2 pi). (Joined by "and" as the two printed conditions appear,
the set would be degenerate; this reading is an interpretation to be confirmed with the
letter's authors.) Eq. (8) evaluates ``|base + sum_{n in A} d_n|^2`` on the controller's
model for every candidate and keeps the best (the first on ties).

Candidate grid: ``L`` values ``Theta_l = l pi/(2L)`` (default L = 4: pi/8, pi/4, 3pi/8,
pi/2) and, per Theta, the nested uniform interior grids ``psi = j Theta/(m+1)``,
``1 <= j <= m <= M`` (default M = 3: Theta/2; Theta/3, 2Theta/3; Theta/4, Theta/2,
3Theta/4). That is ``K = L M (M+1)/2`` candidates (24 for L = 4, M = 3), the count the
letter and DESIGN.md section 8.7 give. The letter does not list the psi values; this
nested grid is an interpretation that reproduces its K (Theta/2, which Theorem 1.7 names as
the attractor, appears twice). Explicit ``thetas_rad``/``psi_fracs`` override it.

Theorem 1 quantities of the letter's corridor (W2) are available as
:func:`wcl_corridor_lengths` and :func:`wcl_phases`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import Any, Literal

import numpy as np

from sremu.contracts import (
    Action,
    CsiView,
    LatencyModel,
    LinkId,
    Observation,
    ObservationSpec,
    PanelId,
    PropagationModel,
    RisCapabilities,
)
from sremu.control.controller import (
    BaseController,
    ChannelModel,
    GridChannel,
    SnapshotChannel,
    aggregate_controls,
    link_nodes,
    position_time,
    wrap_phase,
)
from sremu.errors import ConfigurationError, UnknownTarget
from sremu.physics.optimize import exact_alphabet_optimum, per_element_choice
from sremu.timebase import parse_exact
from sremu.units import C0

__all__ = [
    "BUILTINS",
    "CodebookSweep",
    "DelayedController",
    "ExactOptimum",
    "Fixed",
    "GeneticAlgorithm",
    "GreedyQuantized",
    "IdealDesigned",
    "PositionDriven",
    "ProbeBandit",
    "RisAbsent",
    "TdmaAlternate",
    "Urap",
    "WclHeuristic",
    "build_controller",
    "ga_optimize",
    "optimum_codebook",
    "wcl_activation",
    "wcl_candidates",
    "wcl_corridor_lengths",
    "wcl_heuristic",
    "wcl_phases",
]

_TWO_PI = 2.0 * math.pi


def _states(x: Any, n: int | None, what: str) -> np.ndarray:
    s = np.asarray(x)
    if s.ndim != 1 or (s.size and not np.all(np.equal(np.mod(s, 1), 0))):
        raise ConfigurationError(f"{what}: states must be a 1-D integer vector")
    s = s.astype(np.int16)
    if n is not None and s.shape != (n,):
        raise ConfigurationError(f"{what}: needs {n} control states, got {s.shape[0]}")
    if s.size and int(s.min()) < 0:
        raise ConfigurationError(f"{what}: control states must be >= 0")
    return s


def _n_states(cap: RisCapabilities) -> int:
    return max(1, len(cap.states))


# =========================================================================== simple controllers


class Fixed(BaseController):
    """Constant configuration ``states`` ((N_ctrl,) state indices), sent at the first barrier.

    Other keywords: those of :class:`BaseController` (``period_s`` default 1 s).
    """

    def __init__(
        self,
        states: Sequence[int] | np.ndarray,
        *,
        panel: PanelId | None = None,
        name: str = "fixed",
        period_s: Fraction | int | str = 1,
        **kw: Any,
    ) -> None:
        super().__init__(name, period_s=period_s, **kw)
        self.states = _states(states, None, f"{name}")
        self.panel = panel
        self._sent = False

    def reset(self) -> None:
        self._sent = False
        pid = self._panel(self.panel)
        _states(self.states, self._n_controls(pid), self.name)

    def decide(self, obs: Observation) -> Sequence[Action]:
        """The configuration once; nothing afterwards."""
        if self._sent:
            return []
        self._sent = True
        return [Action(self._panel(self.panel), states=self.states.copy())]


class RisAbsent(BaseController):
    """Counterfactual "RIS absent" baseline: ``Action(panel, enabled=False)`` once (RQ-12)."""

    def __init__(
        self,
        *,
        panel: PanelId | None = None,
        name: str = "ris_absent",
        period_s: Fraction | int | str = 1,
        **kw: Any,
    ) -> None:
        super().__init__(name, period_s=period_s, **kw)
        self.panel = panel
        self._sent = False

    def reset(self) -> None:
        self._sent = False

    def decide(self, obs: Observation) -> Sequence[Action]:
        """``enabled = False`` once; nothing afterwards."""
        if self._sent:
            return []
        self._sent = True
        return [Action(self._panel(self.panel), enabled=False)]


class Urap(BaseController):
    """WCL uniformly randomized activation policy: each period every control is set to
    ``on_state`` with probability ``p`` and to ``off_state`` otherwise (``on_off`` alphabet:
    ON = 1, OFF = 0). Draws: ``rng.random(N_ctrl) < p`` per barrier."""

    def __init__(
        self,
        p: float = 0.2,
        *,
        panel: PanelId | None = None,
        name: str = "urap",
        period_s: Fraction | int | str = Fraction(1, 1000),
        on_state: int = 1,
        off_state: int = 0,
        **kw: Any,
    ) -> None:
        super().__init__(name, period_s=period_s, **kw)
        if not 0.0 <= float(p) <= 1.0:
            raise ConfigurationError(f"Urap: p must lie in [0, 1], got {p}")
        self.p = float(p)
        self.panel = panel
        self.on_state = int(on_state)
        self.off_state = int(off_state)

    def decide(self, obs: Observation) -> Sequence[Action]:
        """A fresh random activation vector."""
        pid = self._panel(self.panel)
        n = self._n_controls(pid)
        u = self._rng().random(n)
        s = np.where(u < self.p, self.on_state, self.off_state).astype(np.int16)
        return [Action(pid, states=s)]


# =========================================================================== position-driven


class PositionDriven(BaseController):
    """Base of the controllers that act on reported positions (section 8.7).

    * ``link``: the link whose channel is maximized (default: the only candidate link).
    * ``panel``: the controlled panel (default: the only panel in the capabilities).
    * ``channel``: a ``ChannelModel`` (``SnapshotChannel``, ``GridChannel``,
      ``FunctionChannel``); None means ``obs.csi_oracle`` (oracle CSI, flagged).
      ``SnapshotChannel`` ignores the reported positions, so ``setup`` rejects it when the
      observation spec has ``position_noise_m > 0``.
    * ``control_of_cell``: (N,) control index of each cell for granularities other than
      ``element`` (default: one control per cell).
    * ``skip_unchanged``: send nothing when the decision equals this controller's previous
      decision (its own memory, so the command log stays free of repeats; a configuration
      changed by another source is not re-asserted).
    * ``oracle_model``: True (default) when the channel model is the scene model itself
      (``SnapshotChannel``, a ``GridChannel`` sampled from the scene, or ``csi_oracle``):
      results are then oracle-geometry results and must be labelled so (section 8.7).
    """

    required_fields = ("positions",)

    def __init__(
        self,
        name: str,
        *,
        link: LinkId | None = None,
        panel: PanelId | None = None,
        channel: ChannelModel | None = None,
        control_of_cell: Sequence[int] | np.ndarray | None = None,
        skip_unchanged: bool = True,
        oracle_model: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(name, **kw)
        self.oracle_model = bool(oracle_model)
        self.link = link
        self.panel = panel
        self.channel = channel
        self.control_of_cell = (
            None if control_of_cell is None else np.asarray(control_of_cell, dtype=np.int64)
        )
        self.skip_unchanged = bool(skip_unchanged)
        if channel is None and "csi_oracle" not in self.spec.fields:
            self.spec = replace(self.spec, fields=(*self.spec.fields, "csi_oracle"))
        self.last: dict[str, Any] = {}
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        if isinstance(self.channel, SnapshotChannel) and self.spec.position_noise_m > 0.0:
            # A SnapshotChannel evaluates the exact scene at the position time and ignores the
            # reported positions, so position noise would be silently dropped.
            raise ConfigurationError(
                f"controller {self.name!r}: position_noise_m > 0 needs a channel that uses the "
                "reported positions (GridChannel or FunctionChannel), not SnapshotChannel"
            )
        self._prev = None
        self.last = {}

    # ---- channel access
    def _link(self, obs: Observation) -> LinkId:
        if self.link is not None:
            return self.link
        cands: list[LinkId] = []
        ch = self.channel
        if isinstance(ch, SnapshotChannel):
            cands = list(ch.model.links())
        elif isinstance(ch, GridChannel):
            cands = list(ch.views[0])
        elif ch is None and obs.csi_oracle is not None:
            cands = list(obs.csi_oracle)
        if len(cands) != 1:
            raise ConfigurationError(f"controller {self.name!r}: give link= (candidates {cands})")
        return cands[0]

    def view(self, obs: Observation) -> CsiView:
        """CSI of the target link at the reported positions."""
        link = self._link(obs)
        if self.channel is None:
            if obs.csi_oracle is None or link not in obs.csi_oracle:
                raise ConfigurationError(
                    f"controller {self.name!r}: no channel model and no csi_oracle for {link!r}"
                )
            return obs.csi_oracle[link]
        return self.channel.csi(link, obs.positions, position_time(obs, self.spec))

    def cascade(self, obs: Observation) -> tuple[complex, np.ndarray]:
        """``(z0, Z)``: the fixed part of the channel (direct paths plus every other panel at
        its observed state) and the (S, N_ctrl) cascade of the controlled panel."""
        pid = self._panel(self.panel)
        v = self.view(obs)
        if pid not in v.cascade:
            raise UnknownTarget(f"controller {self.name!r}: panel {pid!r} not in the CSI")
        z0 = complex(v.h_direct)
        for other, zo in v.cascade.items():
            if other == pid:
                continue
            s = np.asarray(obs.ris_state[other], dtype=np.int64)
            alive = s >= 0
            z0 += complex(zo[s[alive], np.nonzero(alive)[0]].sum())
        z = aggregate_controls(v.cascade[pid], self.control_of_cell, self._n_controls(pid))
        return z0, z

    def _emit(self, states: np.ndarray) -> list[Action]:
        s = np.asarray(states, dtype=np.int16)
        if self.skip_unchanged and self._prev is not None and np.array_equal(s, self._prev):
            return []
        self._prev = s.copy()
        return [Action(self._panel(self.panel), states=s.copy())]

    def _cache_key(self, obs: Observation) -> int | None:
        """Grid index when the decision depends on it alone (GridChannel, one panel)."""
        if isinstance(self.channel, GridChannel) and len(self.caps) <= 1:
            return self.channel.index(obs.positions)
        return None


class ExactOptimum(PositionDriven):
    """Exact finite-alphabet optimum (section 5.13) of ``|z0 + sum_g Z[s_g, g]|`` on the
    controller's model at the reported positions. ``last`` holds the predicted ``|h|``.
    With a ``GridChannel`` the optimum is cached per grid point."""

    def __init__(self, *, name: str = "exact_optimum", **kw: Any) -> None:
        super().__init__(name, **kw)
        self._cache: dict[int, tuple[float, np.ndarray]] = {}

    def reset(self) -> None:
        super().reset()
        self._cache = {}

    def decide(self, obs: Observation) -> Sequence[Action]:
        """Optimum configuration for the reported positions."""
        key = self._cache_key(obs)
        if key is not None and key in self._cache:
            value, s = self._cache[key]
        else:
            z0, z = self.cascade(obs)
            value, s = exact_alphabet_optimum(z0, z)
            if key is not None:
                self._cache[key] = (value, s)
        self.last = {"abs_h": float(value), "grid_index": key}
        return self._emit(s)


class GreedyQuantized(PositionDriven):
    """Phase-align to the fixed part, then quantize per control (section 5.6):
    ``s_g = argmax_s Re{Z[s, g] exp(-j arg z0)}`` (lowest state on ties; with ``z0 = 0`` the
    reference is the strongest cascade entry)."""

    def __init__(self, *, name: str = "greedy_quantized", **kw: Any) -> None:
        super().__init__(name, **kw)

    def decide(self, obs: Observation) -> Sequence[Action]:
        """Quantized phase alignment."""
        z0, z = self.cascade(obs)
        if z0 != 0:
            ref = math.atan2(z0.imag, z0.real)
        else:
            i = np.unravel_index(int(np.argmax(np.abs(z))), z.shape)
            ref = float(np.angle(z[i]))
        s = per_element_choice(z, ref).astype(np.int16)
        self.last = {"abs_h": abs(z0 + complex(np.sum(z[s, np.arange(z.shape[1])])))}
        return self._emit(s)


class IdealDesigned(PositionDriven):
    """Design for an ideal alphabet, deploy the state indices on the true panel (E-C).

    ``true_gamma`` (S,) is the panel's true alphabet at the carrier; the Gamma-free factor is
    recovered as ``C_g = Z[s_ref, g]/true_gamma[s_ref]`` (``s_ref`` = largest |Gamma|) for
    separable tables. The design is the exact optimum over ``design_gamma`` (default
    antipodal ``(1, -1)``) and its indices are deployed as they are.
    """

    def __init__(
        self,
        true_gamma: Sequence[complex] | np.ndarray,
        *,
        design_gamma: Sequence[complex] = (1.0, -1.0),
        name: str = "ideal_designed",
        **kw: Any,
    ) -> None:
        super().__init__(name, **kw)
        self.true_gamma = np.asarray(true_gamma, dtype=np.complex128).reshape(-1)
        self.design_gamma = np.asarray(design_gamma, dtype=np.complex128).reshape(-1)
        if self.design_gamma.size > self.true_gamma.size:
            raise ConfigurationError("IdealDesigned: the design alphabet has more states")
        if not np.any(np.abs(self.true_gamma) > 0):
            raise ConfigurationError("IdealDesigned: true_gamma is all zero")

    def decide(self, obs: Observation) -> Sequence[Action]:
        """Optimum for the ideal alphabet, deployed on the true states."""
        z0, z = self.cascade(obs)
        if z.shape[0] != self.true_gamma.size:
            raise ConfigurationError(
                f"IdealDesigned: CSI has {z.shape[0]} states, true_gamma {self.true_gamma.size}"
            )
        s_ref = int(np.argmax(np.abs(self.true_gamma)))
        c = z[s_ref] / self.true_gamma[s_ref]
        zd = self.design_gamma[:, None] * c[None, :]
        value, s = exact_alphabet_optimum(z0, zd)
        self.last = {"design_abs_h": float(value)}
        return self._emit(s)


# --------------------------------------------------------------------------- WCL heuristic


def wcl_corridor_lengths(
    x_m: float | np.ndarray, h_m: float, y_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Letter geometry: AP at the origin of the user line, RIS line at distance ``h_m``,
    element n at abscissa ``y_m[n]``, user at abscissa ``x_m``.

    Returns ``(D, d_los)``: ``D[..., n] = sqrt(y_n^2 + h^2) + sqrt((x - y_n)^2 + h^2)`` (m)
    and ``d_los = |x|`` (m), broadcasting ``x_m`` over a leading axis.
    """
    x = np.asarray(x_m, dtype=np.float64)[..., None]
    y = np.asarray(y_m, dtype=np.float64)
    dt = np.sqrt(y * y + h_m * h_m)
    dr = np.sqrt((x - y) * (x - y) + h_m * h_m)
    return dt + dr, np.abs(x[..., 0])


def wcl_phases(
    x_m: float | np.ndarray, h_m: float, y_m: np.ndarray, fc_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(phi_n(x), phi(x))`` of the letter: ``2 pi f_c D_n(x)/c`` and ``2 pi f_c x/c`` (rad,
    not reduced modulo 2 pi)."""
    d, dl = wcl_corridor_lengths(x_m, h_m, y_m)
    k = _TWO_PI * float(fc_hz) / C0
    return k * d, k * dl


def wcl_candidates(
    big_l: int = 4,
    big_m: int = 3,
    *,
    thetas_rad: Sequence[float] | None = None,
    psi_fracs: Sequence[float] | None = None,
) -> list[tuple[float, float]]:
    """(Theta, psi) candidates in evaluation order (see the module docstring).

    Default: ``Theta_l = l pi/(2L)`` for l = 1..L and the nested psi fractions
    ``j/(m+1)``, ``1 <= j <= m <= M``: ``L M (M+1)/2`` candidates. ``thetas_rad`` and
    ``psi_fracs`` (fractions of Theta in [0, 1]) override the two grids.
    """
    if thetas_rad is None:
        if int(big_l) < 1:
            raise ConfigurationError("WclHeuristic: L must be >= 1")
        thetas = [l_ * math.pi / (2 * int(big_l)) for l_ in range(1, int(big_l) + 1)]
    else:
        thetas = [float(t) for t in thetas_rad]
    if psi_fracs is None:
        if int(big_m) < 1:
            raise ConfigurationError("WclHeuristic: M must be >= 1")
        fracs = [j / (m + 1) for m in range(1, int(big_m) + 1) for j in range(1, m + 1)]
    else:
        fracs = [float(f) for f in psi_fracs]
    for t in thetas:
        if not 0.0 < t < math.pi:
            raise ConfigurationError(f"WclHeuristic: Theta must lie in (0, pi), got {t}")
    for f in fracs:
        if not 0.0 <= f <= 1.0:
            raise ConfigurationError(f"WclHeuristic: psi fractions must lie in [0, 1], got {f}")
    if not thetas or not fracs:
        raise ConfigurationError("WclHeuristic: empty candidate grid")
    return [(t, f * t) for t in thetas for f in fracs]


def wcl_activation(delta_rad: np.ndarray, theta: float, psi: float) -> np.ndarray:
    """Eq. (7) activation mask: ``(delta)_{2pi} < Theta - psi`` or ``(-delta)_{2pi} < psi``."""
    d = np.asarray(delta_rad, dtype=np.float64)
    return (np.mod(d, _TWO_PI) < theta - psi) | (np.mod(-d, _TWO_PI) < psi)


def wcl_heuristic(
    z0: complex,
    z: np.ndarray,
    candidates: Sequence[tuple[float, float]],
    *,
    delta_rad: np.ndarray | None = None,
    on_state: int = 1,
    off_state: int = 0,
) -> tuple[float, np.ndarray, tuple[float, float]]:
    """Eq. (8): the best activation set over ``candidates``.

    ``z`` is the (S, N) cascade; ``delta_rad`` the relative phases (default: phasor phases
    ``arg(base) - arg(d_n)``). Returns ``(|h|, states (N,) int16, (Theta*, psi*))``.
    """
    zz = np.asarray(z, dtype=np.complex128)
    base = complex(z0) + complex(np.sum(zz[off_state]))
    d = zz[on_state] - zz[off_state]
    if delta_rad is None:
        if base != 0:
            ref = math.atan2(base.imag, base.real)
        else:
            k = int(np.argmax(np.abs(d)))
            ref = float(np.angle(d[k]))
        delta = wrap_phase(ref - np.angle(d))
    else:
        delta = np.asarray(delta_rad, dtype=np.float64)
        if delta.shape != d.shape:
            raise ConfigurationError(f"delta_rad must be {d.shape}, got {delta.shape}")
    best_p, best_mask, best_c = -1.0, np.zeros(d.shape, dtype=bool), candidates[0]
    for th, ps in candidates:
        mask = wcl_activation(delta, th, ps)
        tot = base + complex(np.sum(d[mask]))
        p = tot.real * tot.real + tot.imag * tot.imag
        if p > best_p:
            best_p, best_mask, best_c = p, mask, (th, ps)
    states = np.where(best_mask, on_state, off_state).astype(np.int16)
    return math.sqrt(best_p), states, best_c


class WclHeuristic(PositionDriven):
    """WCL letter sub-optimal heuristic, Eqs. (7)-(8); see the module docstring.

    ``L``, ``M``: grid sizes (K = L M (M+1)/2 candidates); ``thetas_rad``/``psi_fracs``
    override them. ``phase_source``: ``"phasor"`` (default; phases of the controller's
    model phasors) or ``"path_length"`` (the letter's ``2 pi f_c (D_n - d_LOS)/c`` from the
    reported TX/RX positions and ``cell_centres_m`` (N, 3); element granularity only).
    ``on_state``/``off_state``: the binary alphabet's ON and OFF states. ``last`` holds the
    predicted |h|, the chosen (Theta, psi) and K.
    """

    def __init__(
        self,
        L: int = 4,  # the letter's symbols (L Theta values, M psi levels)
        M: int = 3,
        *,
        thetas_rad: Sequence[float] | None = None,
        psi_fracs: Sequence[float] | None = None,
        phase_source: Literal["phasor", "path_length"] = "phasor",
        cell_centres_m: np.ndarray | None = None,
        on_state: int = 1,
        off_state: int = 0,
        name: str = "wcl_heuristic",
        **kw: Any,
    ) -> None:
        super().__init__(name, **kw)
        self.candidates = wcl_candidates(L, M, thetas_rad=thetas_rad, psi_fracs=psi_fracs)
        if phase_source not in ("phasor", "path_length"):
            raise ConfigurationError(f"WclHeuristic: unknown phase_source {phase_source!r}")
        self.phase_source = phase_source
        self.cell_centres_m = (
            None if cell_centres_m is None else np.asarray(cell_centres_m, dtype=np.float64)
        )
        if phase_source == "path_length" and self.cell_centres_m is None:
            raise ConfigurationError(
                "WclHeuristic: phase_source='path_length' needs cell_centres_m"
            )
        self.on_state = int(on_state)
        self.off_state = int(off_state)
        self._cache: dict[int, tuple[float, np.ndarray, tuple[float, float]]] = {}

    @property
    def n_candidates(self) -> int:
        """K, the number of (Theta, psi) candidates evaluated per decision."""
        return len(self.candidates)

    def reset(self) -> None:
        super().reset()
        self._cache = {}

    def _path_length_delta(self, obs: Observation, n: int) -> np.ndarray:
        if self.control_of_cell is not None:
            raise ConfigurationError("WclHeuristic: path_length phases need element granularity")
        assert self.cell_centres_m is not None
        if self.cell_centres_m.shape != (n, 3):
            raise ConfigurationError(f"cell_centres_m must be ({n}, 3)")
        tx, rx = link_nodes(self._link(obs))
        if obs.positions is None or tx not in obs.positions or rx not in obs.positions:
            raise ConfigurationError("WclHeuristic: path_length phases need TX and RX positions")
        ptx = np.asarray(obs.positions[tx], dtype=np.float64)
        prx = np.asarray(obs.positions[rx], dtype=np.float64)
        c = self.cell_centres_m
        dn = np.linalg.norm(c - ptx[None, :], axis=1) + np.linalg.norm(c - prx[None, :], axis=1)
        dl = float(np.linalg.norm(prx - ptx))
        fc = float(self.view(obs).fc_hz)
        return _TWO_PI * fc * (dn - dl) / C0

    def decide(self, obs: Observation) -> Sequence[Action]:
        """Best activation set of Eq. (8) for the reported positions."""
        key = self._cache_key(obs) if self.phase_source == "phasor" else None
        if key is not None and key in self._cache:
            value, s, best = self._cache[key]
        else:
            z0, z = self.cascade(obs)
            if z.shape[0] <= max(self.on_state, self.off_state):
                raise ConfigurationError("WclHeuristic: the panel needs the ON and OFF states")
            delta = (
                self._path_length_delta(obs, z.shape[1])
                if self.phase_source == "path_length"
                else None
            )
            value, s, best = wcl_heuristic(
                z0,
                z,
                self.candidates,
                delta_rad=delta,
                on_state=self.on_state,
                off_state=self.off_state,
            )
            if key is not None:
                self._cache[key] = (value, s, best)
        self.last = {"abs_h": value, "theta": best[0], "psi": best[1], "K": self.n_candidates}
        return self._emit(s)


# --------------------------------------------------------------------------- GA (W4 only)


def ga_optimize(
    z0: complex,
    z: np.ndarray,
    rng: np.random.Generator,
    *,
    pop: int = 300,
    mut: float = 0.01,
    iters: int = 150,
    elite: int = 1,
) -> tuple[float, np.ndarray]:
    """Genetic algorithm for ``max |z0 + sum_n Z[s_n, n]|`` (the letter's GA; W4 only).

    Population ``pop`` of uniform random chromosomes; per generation: ``elite`` best kept,
    the rest bred by two size-2 tournaments, single-point crossover and per-gene mutation
    with probability ``mut`` (to a different state). Returns ``(|h|, s (N,) int16)``.
    """
    zz = np.asarray(z, dtype=np.complex128)
    s_count, n = zz.shape
    if s_count == 1:
        return abs(complex(z0) + complex(np.sum(zz[0]))), np.zeros(n, dtype=np.int16)
    if pop < 2 or not 0 <= elite < pop or iters < 0 or not 0.0 <= mut <= 1.0:
        raise ConfigurationError("ga_optimize: need pop >= 2, 0 <= elite < pop, iters >= 0")
    cols = np.arange(n)[None, :]

    def fitness(p: np.ndarray) -> np.ndarray:
        tot = complex(z0) + np.sum(zz[p, cols], axis=1)
        return tot.real * tot.real + tot.imag * tot.imag

    popu = rng.integers(0, s_count, size=(pop, n))
    fit = fitness(popu)
    n_child = pop - elite
    for _ in range(iters):
        order = np.argsort(-fit, kind="stable")
        elites = popu[order[:elite]]
        t1 = rng.integers(0, pop, size=(n_child, 2))
        t2 = rng.integers(0, pop, size=(n_child, 2))
        pa = np.where(fit[t1[:, 0]] >= fit[t1[:, 1]], t1[:, 0], t1[:, 1])
        pb = np.where(fit[t2[:, 0]] >= fit[t2[:, 1]], t2[:, 0], t2[:, 1])
        cut = rng.integers(1, n, size=n_child) if n > 1 else np.ones(n_child, dtype=np.int64)
        child = np.where(np.arange(n)[None, :] < cut[:, None], popu[pa], popu[pb])
        flip = rng.random((n_child, n)) < mut
        shift = rng.integers(1, s_count, size=(n_child, n))
        child = np.where(flip, (child + shift) % s_count, child)
        popu = np.vstack([elites, child])
        fit = fitness(popu)
    best = int(np.argmax(fit))
    return math.sqrt(float(fit[best])), popu[best].astype(np.int16)


class GeneticAlgorithm(PositionDriven):
    """The letter's GA (pop 300, mutation 0.01, 150 iterations by default); W4 only."""

    def __init__(
        self,
        pop: int = 300,
        mut: float = 0.01,
        iters: int = 150,
        *,
        name: str = "genetic_algorithm",
        **kw: Any,
    ) -> None:
        super().__init__(name, **kw)
        self.pop, self.mut, self.iters = int(pop), float(mut), int(iters)

    def decide(self, obs: Observation) -> Sequence[Action]:
        """GA configuration for the reported positions."""
        z0, z = self.cascade(obs)
        value, s = ga_optimize(z0, z, self._rng(), pop=self.pop, mut=self.mut, iters=self.iters)
        self.last = {"abs_h": value}
        return self._emit(s)


# =========================================================================== codebook / bandit


def optimum_codebook(
    views: Sequence[CsiView],
    panel: PanelId,
    *,
    control_of_cell: Sequence[int] | np.ndarray | None = None,
    n_controls: int | None = None,
) -> np.ndarray:
    """(K, N_ctrl) int16 codebook of exact optima, one per CSI view (e.g. E-A's 16 positions)."""
    rows = []
    for v in views:
        z = v.cascade[panel]
        n = z.shape[1] if n_controls is None else int(n_controls)
        coc = None if control_of_cell is None else np.asarray(control_of_cell, dtype=np.int64)
        zc = aggregate_controls(z, coc, n)
        rows.append(exact_alphabet_optimum(v.h_direct, zc)[1])
    if not rows:
        raise ConfigurationError("optimum_codebook: no views")
    return np.stack(rows).astype(np.int16)


def _codebook(spec: int | np.ndarray | Sequence[Sequence[int]]) -> int | np.ndarray:
    if isinstance(spec, int | np.integer) and not isinstance(spec, bool):
        if int(spec) < 1:
            raise ConfigurationError("codebook size must be >= 1")
        return int(spec)
    cb = np.asarray(spec)
    if cb.ndim != 2 or cb.shape[0] < 1:
        raise ConfigurationError(f"codebook must be (K, N_ctrl), got shape {cb.shape}")
    return _states(cb.ravel(), None, "codebook").reshape(cb.shape)


class _CodebookController(BaseController):
    """Shared codebook handling: an explicit (K, N_ctrl) array or a size K (uniform random
    codebook drawn at setup from the keyed generator)."""

    def __init__(
        self,
        name: str,
        codebook: int | np.ndarray | Sequence[Sequence[int]],
        *,
        panel: PanelId | None = None,
        rx: str | None = None,
        use_codebook_index: bool = False,
        **kw: Any,
    ) -> None:
        super().__init__(name, **kw)
        self._spec_cb = _codebook(codebook)
        self.panel = panel
        self.rx = rx
        self.use_codebook_index = bool(use_codebook_index)
        self.codebook = np.zeros((0, 0), dtype=np.int16)

    def reset(self) -> None:
        pid = self._panel(self.panel)
        n = self._n_controls(pid)
        if isinstance(self._spec_cb, int):
            s_count = _n_states(self.caps[pid])
            self.codebook = (
                self._rng().integers(0, s_count, size=(self._spec_cb, n)).astype(np.int16)
            )
        else:
            if self._spec_cb.shape[1] != n:
                raise ConfigurationError(
                    f"{self.name}: codebook rows need {n} controls, got {self._spec_cb.shape[1]}"
                )
            self.codebook = self._spec_cb.copy()

    def _action(self, i: int) -> Action:
        pid = self._panel(self.panel)
        if self.use_codebook_index:
            return Action(pid, codebook_index=int(i))
        return Action(pid, states=self.codebook[i].copy())

    def _power_mw(self, obs: Observation) -> float | None:
        """Observed RX power in mW (None when unavailable); ``rx`` defaults to the only RX."""
        p = obs.rx_power_dbm
        if self.rx is not None:
            if self.rx not in p:
                raise UnknownTarget(f"{self.name}: no RX power for {self.rx!r}")
            v = p[self.rx]
        elif len(p) == 1:
            v = next(iter(p.values()))
        elif not p:
            return None
        else:
            raise ConfigurationError(f"{self.name}: give rx= (observed streams {sorted(p)})")
        v = float(v)
        if math.isnan(v):
            return None
        return 0.0 if v == -math.inf else 10.0 ** (v / 10.0)


class CodebookSweep(_CodebookController):
    """Cyclic sweep: barrier m applies codebook entry ``m mod K``. The power observed at a
    barrier is credited to the entry applied at the previous barrier (``powers_mw``)."""

    def __init__(
        self,
        codebook: int | np.ndarray | Sequence[Sequence[int]] = 16,
        *,
        name: str = "codebook_sweep",
        **kw: Any,
    ) -> None:
        super().__init__(name, codebook, **kw)
        self.powers_mw: list[list[float]] = []
        self._m = 0
        self._last_arm: int | None = None

    def reset(self) -> None:
        super().reset()
        self.powers_mw = [[] for _ in range(self.codebook.shape[0])]
        self._m = 0
        self._last_arm = None

    def best_index(self) -> int | None:
        """Entry with the largest mean observed power (None before any feedback)."""
        means = [float(np.mean(p)) if p else -math.inf for p in self.powers_mw]
        return None if all(m == -math.inf for m in means) else int(np.argmax(means))

    def decide(self, obs: Observation) -> Sequence[Action]:
        """Next codebook entry."""
        if self._last_arm is not None:
            pw = self._power_mw(obs)
            if pw is not None:
                self.powers_mw[self._last_arm].append(pw)
        i = self._m % self.codebook.shape[0]
        self._m += 1
        self._last_arm = i
        return [self._action(i)]


class ProbeBandit(_CodebookController):
    """UCB1 over a codebook with windowed RX power as reward (section 8.7; SHOULD).

    Barrier period = ``dwell_s``. At each barrier the observed power (window ``window_s``,
    default ``dwell_s/2``, before t_b) is credited to the arm applied at the previous barrier,
    as ``r = P_mW / max P_mW seen`` in [0, 1]. Untried arms are played first (index order);
    then ``argmax_i mean_i + c sqrt(2 ln n / n_i)`` (lowest index on ties). The window must
    lie after the previous command took effect (latency < dwell - window).
    """

    def __init__(
        self,
        codebook: int | np.ndarray | Sequence[Sequence[int]] = 16,
        dwell_s: Fraction | int | str = "2e-2",
        *,
        c: float = 1.0,
        window_s: Fraction | int | str | None = None,
        name: str = "probe_bandit",
        spec: ObservationSpec | None = None,
        **kw: Any,
    ) -> None:
        dwell = parse_exact(dwell_s, name="dwell_s")
        win = dwell / 2 if window_s is None else parse_exact(window_s, name="window_s")
        if not 0 < win <= dwell:
            raise ConfigurationError("ProbeBandit: need 0 < window_s <= dwell_s")
        base = spec if spec is not None else ObservationSpec()
        if "rx_power_dbm" not in base.fields:
            base = replace(base, fields=(*base.fields, "rx_power_dbm"))
        super().__init__(name, codebook, period_s=dwell, spec=replace(base, window_s=win), **kw)
        if not c >= 0:
            raise ConfigurationError("ProbeBandit: c must be >= 0")
        self.c = float(c)
        self.counts = np.zeros(0, dtype=np.int64)
        self._raw: list[tuple[int, float]] = []
        self._last_arm: int | None = None

    def reset(self) -> None:
        super().reset()
        k = self.codebook.shape[0]
        self.counts = np.zeros(k, dtype=np.int64)
        self._raw = []
        self._last_arm = None

    def means(self) -> np.ndarray:
        """Mean normalized reward per arm (NaN for untried arms)."""
        k = self.codebook.shape[0]
        if not self._raw:
            return np.full(k, np.nan)
        top = max(p for _, p in self._raw)
        sums = np.zeros(k)
        for arm, p in self._raw:
            sums[arm] += p / top if top > 0 else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, sums / np.maximum(self.counts, 1), np.nan)

    def choose(self) -> int:
        """The next arm (UCB1)."""
        untried = np.nonzero(self.counts == 0)[0]
        if untried.size:
            return int(untried[0])
        n = int(self.counts.sum())
        idx = self.means() + self.c * np.sqrt(2.0 * math.log(n) / self.counts)
        return int(np.argmax(idx))

    def decide(self, obs: Observation) -> Sequence[Action]:
        """Credit the previous arm, then play the UCB1 choice."""
        if self._last_arm is not None:
            pw = self._power_mw(obs)
            if pw is not None:
                self.counts[self._last_arm] += 1
                self._raw.append((self._last_arm, pw))
        arm = self.choose()
        self._last_arm = arm
        return [self._action(arm)]


# =========================================================================== TDMA (E-B2)


class TdmaAlternate(BaseController):
    """Alternate per-packet configurations between targets, one switch per guard (E-B2).

    ``configs``: one (N_ctrl,) state vector per target (e.g. exact optima for RX1 and RX2).
    Barrier m (``t_b = m slot_s``, ``period_s = slot_s``) prepares guard m, after which
    packet m goes to target ``m mod T``:

    * ``timing="timed"``: the switch is requested at ``t_b + lead_s`` (the guard centre when
      the schedule puts guard centres at ``m slot_s + lead_s``); a late arrival follows the
      engine's late policy (``apply_on_arrival``).
    * ``"asap"`` and ``"legacy"``: untimed (``at_s`` None: ``t_b`` + latency); ``legacy`` is
      meant for the engine's ``LegacyNextTickAdapter``.

    :meth:`guard_centre_s` gives the schedule the controller assumes.
    """

    def __init__(
        self,
        configs: Sequence[Sequence[int]] | np.ndarray,
        *,
        slot_s: Fraction | int | str,
        timing: Literal["timed", "asap", "legacy"] = "timed",
        lead_s: Fraction | int | str = Fraction(1, 2000),
        panel: PanelId | None = None,
        name: str = "tdma_alternate",
        **kw: Any,
    ) -> None:
        super().__init__(name, period_s=slot_s, **kw)
        cb = np.asarray(configs)
        if cb.ndim != 2 or cb.shape[0] < 1:
            raise ConfigurationError("TdmaAlternate: configs must be (T, N_ctrl)")
        self.configs: np.ndarray = _states(cb.ravel(), None, name).reshape(cb.shape)
        if timing not in ("timed", "asap", "legacy"):
            raise ConfigurationError(f"TdmaAlternate: unknown timing {timing!r}")
        self.timing = timing
        self.lead_s = parse_exact(lead_s, name="lead_s")
        if timing == "timed" and self.lead_s < self.min_latency_s:
            raise ConfigurationError("TdmaAlternate: lead_s < min_latency_s (lookahead, 4.9)")
        self.panel = panel
        self._m = 0

    def reset(self) -> None:
        self._m = 0
        pid = self._panel(self.panel)
        if self.configs.shape[1] != self._n_controls(pid):
            raise ConfigurationError("TdmaAlternate: configs do not match the panel's controls")

    def guard_centre_s(self, m: int) -> Fraction:
        """Switch time the controller requests for guard m in ``timed`` mode, s."""
        return m * self.period_s + self.lead_s

    def decide(self, obs: Observation) -> Sequence[Action]:
        """Configuration of the next packet's target."""
        m = int(Fraction(obs.t_b) / self.period_s)
        self._m = m + 1
        target = m % self.configs.shape[0]
        at = Fraction(obs.t_b) + self.lead_s if self.timing == "timed" else None
        return [Action(self._panel(self.panel), states=self.configs[target].copy(), at_s=at)]


# =========================================================================== delay wrapper


class DelayedController:
    """Signalling delay wrapper (E-A): the inner position-driven controller decides every
    ``tau_sui_s`` (its barrier period) on positions ``report_delay_s`` old.

    The wrapper owns ``inner``: it sets the inner controller's period and observation spec
    (``positions`` added, ``position_delay_s = report_delay_s``). ``name`` defaults to the
    inner name, so the keyed generator is the same for every tau_SUI (paired comparisons).
    """

    def __init__(
        self,
        inner: BaseController,
        tau_sui_s: Fraction | int | str,
        report_delay_s: Fraction | int | str = 0,
        *,
        name: str | None = None,
    ) -> None:
        tau = parse_exact(tau_sui_s, name="tau_sui_s")
        if tau <= 0:
            raise ConfigurationError("DelayedController: tau_sui_s must be > 0")
        delay = parse_exact(report_delay_s, name="report_delay_s")
        if delay < 0:
            raise ConfigurationError("DelayedController: report_delay_s must be >= 0")
        self.inner = inner
        self.name = inner.name if name is None else name
        self.tau_sui_s = tau
        self.report_delay_s = delay
        self.period_s: Fraction = tau
        self.min_latency_s: Fraction = Fraction(inner.min_latency_s)
        self.latency: LatencyModel = inner.latency
        fields = tuple(inner.spec.fields)
        if "positions" not in fields:
            fields = (*fields, "positions")
        self.spec: ObservationSpec = replace(inner.spec, fields=fields, position_delay_s=delay)
        inner.spec = self.spec
        inner.period_s = tau

    def setup(self, caps: Mapping[PanelId, RisCapabilities], rng: np.random.Generator) -> None:
        """Set up the inner controller with the same capabilities and generator."""
        self.inner.setup(caps, rng)

    def decide(self, obs: Observation) -> Sequence[Action]:
        """The inner controller's decision."""
        return self.inner.decide(obs)

    @property
    def last(self) -> dict[str, Any]:
        """The inner controller's diagnostics."""
        return dict(getattr(self.inner, "last", {}))


# =========================================================================== factory


BUILTINS: dict[str, type] = {
    "Fixed": Fixed,
    "RisAbsent": RisAbsent,
    "Urap": Urap,
    "WclHeuristic": WclHeuristic,
    "ExactOptimum": ExactOptimum,
    "GreedyQuantized": GreedyQuantized,
    "IdealDesigned": IdealDesigned,
    "CodebookSweep": CodebookSweep,
    "ProbeBandit": ProbeBandit,
    "GeneticAlgorithm": GeneticAlgorithm,
    "TdmaAlternate": TdmaAlternate,
    "DelayedController": DelayedController,
}
"""Built-in controller classes by scenario ``impl`` name (section 9.2 ``controllers``)."""

_POSITION_DRIVEN = (WclHeuristic, ExactOptimum, GreedyQuantized, IdealDesigned, GeneticAlgorithm)


def _get(spec: Any, key: str, default: Any = None) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key, default)
    return getattr(spec, key, default)


def build_controller(
    spec: Any,
    *,
    model: PropagationModel | None = None,
    channel: ChannelModel | None = None,
) -> Any:
    """Build a built-in controller from a scenario ``controllers`` entry (section 9.2).

    ``spec`` is a ``scenario.models.ControllerModel`` or a mapping with the same keys
    (``name``, ``impl``, ``period_s``, ``min_latency_s``, ``latency {fixed_s, jitter}``,
    ``observation {fields, window_s, position_delay_s, position_noise_m}``, ``params``).
    Position-driven controllers use ``channel``, else ``SnapshotChannel(model)`` when a model
    is given (oracle geometry), else the observation's ``csi_oracle``.
    ``DelayedController`` takes ``params = {inner: <spec without name>, tau_sui_s,
    report_delay_s}``.
    """
    impl = _get(spec, "impl")
    if impl not in BUILTINS:
        raise ConfigurationError(f"unknown built-in controller {impl!r}; known: {sorted(BUILTINS)}")
    name = _get(spec, "name")
    params = dict(_get(spec, "params", {}) or {})
    if impl == "DelayedController":
        inner_spec = dict(params.pop("inner"))
        inner_spec.setdefault("name", name)
        for key in ("period_s", "min_latency_s", "latency", "observation"):
            if _get(spec, key) is not None:
                inner_spec.setdefault(key, _get(spec, key))
        inner = build_controller(inner_spec, model=model, channel=channel)
        return DelayedController(inner, **params)
    lat = _get(spec, "latency")
    latency = None
    if lat is not None:
        latency = LatencyModel(fixed_s=_get(lat, "fixed_s", 0), jitter=_get(lat, "jitter"))
    obs = _get(spec, "observation")
    ospec = None
    if obs is not None:
        ospec = ObservationSpec(
            fields=tuple(_get(obs, "fields", ("rx_power_dbm", "ris_state"))),
            window_s=_get(obs, "window_s", Fraction(1, 1000)),
            position_delay_s=_get(obs, "position_delay_s", Fraction(0)),
            position_noise_m=float(_get(obs, "position_noise_m", 0.0)),
        )
    kw: dict[str, Any] = {"name": name}
    if _get(spec, "min_latency_s") is not None:
        kw["min_latency_s"] = _get(spec, "min_latency_s")
    if latency is not None:
        kw["latency"] = latency
    cls = BUILTINS[impl]
    if cls is ProbeBandit:
        params.setdefault("dwell_s", _get(spec, "period_s"))
        if ospec is not None:
            kw["spec"] = ospec
    else:
        if _get(spec, "period_s") is not None:
            key = "slot_s" if cls is TdmaAlternate else "period_s"
            kw[key] = _get(spec, "period_s")
        if ospec is not None:
            kw["spec"] = ospec
    if cls in _POSITION_DRIVEN and "channel" not in params:
        if channel is not None:
            kw["channel"] = channel
        elif model is not None:
            kw["channel"] = SnapshotChannel(model)
    return cls(**params, **kw)
