"""Non-ideality pipeline and its analytic oracles (DESIGN.md section 5.6; WP2).

Normative order:

1. **Command**: ``states[N_ctrl]`` (int16), ``phases_rad[N_ctrl]`` or ``codebook_index``;
   granularity (element | column | tile | groups) expands N_ctrl to N physical cells.
2. **Quantizer** (continuous phases only): ``s = argmax_s Re{Gamma_ref(s) exp(-j phi)}``,
   correct for unequal amplitudes (ties resolve to the lowest state index).
3. **Failures**: :class:`FailureMap` (stuck-at and dead; dead -> state -1 -> ``gamma_dead``).
4. **Table lookup** -> ``B[r, s, n]`` at snapshot time (``propagation``).
5. **Efficiency**: amplitude times ``sqrt(eta_eff)``; refused for measured/digitized tables
   unless overridden (:func:`efficiency_factor`).
6. **Coupling hook** (off; separable tables only; no kappa values ship).
7. **Transition** in time (engine/render).

Phase wrap rule (D8, PH-16): a phase is reduced modulo one turn, in its own unit, BEFORE a
reflection coefficient is formed from it (:func:`wrap_phase_rad`, :func:`wrap_phase_deg`,
:func:`gamma_from_phase`; the quantizer of step 2 and the table importers of
``physics.response``). The reduction (``fmod``) is exact, so a phase given as ``phi + m``
turns yields bitwise the same Gamma as ``phi`` whenever the caller's ``phi + m`` turns is
itself exact in floating point (e.g. ``phase_deg + 360 m`` for table phases on a binary
grid). A float sum that already rounded (``phi + 2 pi m`` in radians, large ``|m|``) carries
an error of about ``ulp(2 pi |m|)/2`` rad that no reduction can remove; quantized phase
commands are unaffected (identical states, bitwise identical channel).

Analytic oracles: the large-N loss ``eta`` of finite alphabets and of the Abeywickrama
model, and the binary ``m/d`` decomposition ``Gamma = m + d s``, ``s = +/-1``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from sremu.contracts import Granularity, PanelConfig
from sremu.errors import ConfigurationError, NotPermitted

__all__ = [
    "CouplingHook",
    "FailureMap",
    "NoCoupling",
    "PanelPipeline",
    "binary_md",
    "control_map",
    "efficiency_factor",
    "eta_abeywickrama",
    "eta_alphabet",
    "eta_binary",
    "eta_bits",
    "expand_states",
    "gamma_from_phase",
    "md_channel",
    "quantize_phases",
    "wrap_phase_deg",
    "wrap_phase_rad",
]

_UNCERTAIN = ("measured", "digitized")
_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- phase wrap (D8)


def _wrap(x: np.ndarray | float, turn: float) -> np.ndarray:
    """``x mod turn`` in [0, turn): an exact ``fmod``, except that a tiny negative ``x``
    whose remainder rounds up to ``turn`` itself maps to 0 (the nearest equivalent phase)."""
    w = np.remainder(np.asarray(x, dtype=np.float64), turn)
    return np.where(w >= turn, 0.0, w)


def wrap_phase_rad(phase_rad: np.ndarray | float) -> np.ndarray:
    """Phase reduced to [0, 2 pi) rad with an exact ``fmod`` (D8: before Gamma is formed)."""
    return _wrap(phase_rad, _TWO_PI)


def wrap_phase_deg(phase_deg: np.ndarray | float) -> np.ndarray:
    """Phase reduced to [0, 360) deg with an exact ``fmod`` (D8: before Gamma is formed)."""
    return _wrap(phase_deg, 360.0)


def gamma_from_phase(phase_rad: np.ndarray | float, mag: np.ndarray | float = 1.0) -> np.ndarray:
    """``mag * exp(j phi)`` with ``phi`` first reduced modulo 2 pi (D8, PH-16)."""
    phi = wrap_phase_rad(phase_rad)
    return np.asarray(mag, dtype=np.float64) * (np.cos(phi) + 1j * np.sin(phi))


# --------------------------------------------------------------------------- granularity


def control_map(
    granularity: Granularity,
    n_u: int,
    n_v: int,
    *,
    tiles: tuple[int, int] = (1, 1),
    groups: Mapping[str, Sequence[int]] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Control index of every physical cell (N,) int64 and the control names.

    ``element``: one control per cell. ``column``: one per column ``iu`` (cells
    ``n = iv*n_u + iu``). ``tile``: one per physical tile (``tiles = (n_tiles_u,
    n_tiles_v)``). ``groups``: named disjoint groups covering every cell, in mapping order.
    """
    n = n_u * n_v
    cells = np.arange(n)
    iu, iv = cells % n_u, cells // n_u
    if granularity == "element":
        return cells.astype(np.int64), tuple(f"e{k}" for k in range(n))
    if granularity == "column":
        return iu.astype(np.int64), tuple(f"col{k}" for k in range(n_u))
    if granularity == "tile":
        tu_n, tv_n = tiles
        if n_u % tu_n or n_v % tv_n:
            raise ConfigurationError("tile granularity: tiles must divide the cell grid")
        tu = iu // (n_u // tu_n)
        tv = iv // (n_v // tv_n)
        return (tv * tu_n + tu).astype(np.int64), tuple(f"tile{k}" for k in range(tu_n * tv_n))
    if granularity == "groups":
        if not groups:
            raise ConfigurationError("granularity 'groups' needs control.groups")
        ctrl = np.full(n, -1, dtype=np.int64)
        names = tuple(groups)
        for k, name in enumerate(names):
            idx = np.asarray(list(groups[name]), dtype=np.int64)
            if idx.size and (idx.min() < 0 or idx.max() >= n):
                raise ConfigurationError(f"group {name!r} has cell indices outside [0, {n})")
            if np.any(ctrl[idx] >= 0):
                raise ConfigurationError(f"group {name!r} overlaps another group")
            ctrl[idx] = k
        if np.any(ctrl < 0):
            raise ConfigurationError("groups must cover every cell")
        return ctrl, names
    raise ConfigurationError(f"unknown granularity {granularity!r}")


def expand_states(cmd: np.ndarray, ctrl_of_cell: np.ndarray) -> np.ndarray:
    """Expand per-control states (N_ctrl,) to physical cells (N,) int16."""
    c = np.asarray(cmd)
    if c.ndim != 1 or c.size != int(ctrl_of_cell.max()) + 1:
        raise ConfigurationError(
            f"command has {c.size} controls, the panel expects {int(ctrl_of_cell.max()) + 1}"
        )
    return c.astype(np.int16)[ctrl_of_cell]


# --------------------------------------------------------------------------- quantizer


def quantize_phases(phases_rad: np.ndarray, gamma_ref: np.ndarray) -> np.ndarray:
    """``s_n = argmax_s Re{gamma_ref[s(, n)] exp(-j phi_n)}`` (int16), step 2.

    ``gamma_ref`` is (S,) for a panel-wide reference or (S, N) per element (e.g. per link).
    Ties resolve to the lowest state index; phases are reduced modulo 2 pi first (D8).
    """
    phi = np.asarray(phases_rad, dtype=np.float64)
    g = np.asarray(gamma_ref, dtype=np.complex128)
    if g.ndim == 1:
        g = g[:, None]
    if g.ndim != 2 or (g.shape[1] not in (1, phi.size)):
        raise ConfigurationError("gamma_ref must be (S,) or (S, N)")
    phi_r = wrap_phase_rad(phi)
    proj = g.real * np.cos(phi_r)[None, :] + g.imag * np.sin(phi_r)[None, :]
    return np.argmax(proj, axis=0).astype(np.int16)


# --------------------------------------------------------------------------- failures


@dataclass(frozen=True, eq=False)
class FailureMap:
    """Per-cell failures (step 3): ``stuck_at`` (N,) int16 (-1 healthy, else the stuck state)
    and ``dead`` (N,) bool (dead cells get state -1, i.e. ``gamma_dead``)."""

    stuck_at: np.ndarray
    dead: np.ndarray

    def __post_init__(self) -> None:
        s = np.asarray(self.stuck_at, dtype=np.int16)
        d = np.asarray(self.dead, dtype=bool)
        if s.ndim != 1 or d.shape != s.shape:
            raise ConfigurationError("stuck_at and dead must be (N,) arrays of equal length")
        if s.size and int(s.min()) < -1:
            raise ConfigurationError("stuck_at values must be >= -1")
        object.__setattr__(self, "stuck_at", s)
        object.__setattr__(self, "dead", d)

    @classmethod
    def healthy(cls, n: int) -> FailureMap:
        """No failures."""
        return cls(np.full(n, -1, dtype=np.int16), np.zeros(n, dtype=bool))

    @classmethod
    def random(
        cls,
        n: int,
        rate: float,
        mode: Literal["dead", "stuck"],
        rng: np.random.Generator,
        *,
        n_states: int = 2,
    ) -> FailureMap:
        """Each cell fails independently with probability ``rate``.

        ``mode="dead"``: failed cells are dead. ``mode="stuck"``: failed cells are stuck at a
        uniformly drawn state in [0, n_states). ``rng`` must be the keyed generator of
        purpose ``failure/<panel>`` (section 6.2).
        """
        if not 0.0 <= rate <= 1.0:
            raise ConfigurationError("failure rate must lie in [0, 1]")
        fail = rng.random(n) < rate
        if mode == "dead":
            return cls(np.full(n, -1, dtype=np.int16), fail)
        if mode == "stuck":
            st = rng.integers(0, n_states, size=n).astype(np.int16)
            return cls(np.where(fail, st, -1).astype(np.int16), np.zeros(n, dtype=bool))
        raise ConfigurationError(f"unknown failure mode {mode!r}")

    @classmethod
    def from_rates(
        cls,
        n: int,
        dead_rate: float,
        stuck_rate: float,
        rng: np.random.Generator,
        *,
        n_states: int = 2,
    ) -> FailureMap:
        """Disjoint dead and stuck cells with the given rates (scenario ``failures``)."""
        if dead_rate < 0 or stuck_rate < 0 or dead_rate + stuck_rate > 1:
            raise ConfigurationError("dead_rate, stuck_rate must be >= 0 with sum <= 1")
        u = rng.random(n)
        dead = u < dead_rate
        stuck = (u >= dead_rate) & (u < dead_rate + stuck_rate)
        st = rng.integers(0, n_states, size=n).astype(np.int16)
        return cls(np.where(stuck, st, -1).astype(np.int16), dead)

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Apply stuck-at then dead handling to a physical state vector (N,) int16."""
        s = np.asarray(state, dtype=np.int16)
        if s.shape != self.stuck_at.shape:
            raise ConfigurationError(f"state shape {s.shape} != failure map {self.stuck_at.shape}")
        out = np.where(self.stuck_at >= 0, self.stuck_at, s).astype(np.int16)
        out[self.dead] = -1
        return out

    @property
    def n_dead(self) -> int:
        """Number of dead cells."""
        return int(self.dead.sum())

    @property
    def n_stuck(self) -> int:
        """Number of stuck cells."""
        return int((self.stuck_at >= 0).sum())


# --------------------------------------------------------------------------- efficiency, coupling


def efficiency_factor(eta: float, source_kind: str, *, override: bool = False) -> float:
    """Amplitude factor ``sqrt(eta)`` of step 5.

    Raises ``NotPermitted`` for ``source_kind`` measured or digitized unless ``override``,
    because the measurement already contains the loss.
    """
    if not 0.0 < eta <= 1.0:
        raise ConfigurationError(f"efficiency must lie in (0, 1], got {eta}")
    if eta != 1.0 and source_kind in _UNCERTAIN and not override:
        raise NotPermitted(
            f"efficiency on a {source_kind} table would count the loss twice; "
            "set override_efficiency: true to force it"
        )
    return math.sqrt(eta)


@runtime_checkable
class CouplingHook(Protocol):
    """Neighbour-state coupling hook (step 6; off by default; no kappa values ship).

    ``delta(state)`` returns (N,) complex ``delta Gamma_n`` added to every cell's Gamma for
    the configuration ``state``; ``combine`` adds ``sum_n C[r, n] delta Gamma_n`` (separable
    tables only). No implementation other than :class:`NoCoupling` ships. Response tables are
    per state (Gamma(S, F, T, T)); a table measured in an array of identical neighbours
    contains the locally periodic coupling (ETSI GR RIS 003 section 5.1.3), and no
    configuration-dependent coupling is modelled.
    """

    def delta(self, state: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class NoCoupling:
    """The default hook: no coupling."""

    def delta(self, state: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(state).shape, dtype=np.complex128)


# --------------------------------------------------------------------------- pipeline


@dataclass(eq=False)
class PanelPipeline:
    """Steps 1-3 for one panel: command -> physical per-cell state -> ``PanelConfig``.

    ``ctrl_of_cell`` (N,) maps cells to controls (:func:`control_map`), ``gamma_ref`` (S,) is
    the quantizer reference (table at the panel reference geometry), ``failures`` the
    panel's :class:`FailureMap`, ``codebooks`` named (K, N_ctrl) int16 arrays registered via
    ``ris.set_codebook`` (the default codebook is ``"default"``).
    """

    panel: str
    ctrl_of_cell: np.ndarray
    gamma_ref: np.ndarray
    failures: FailureMap
    codebooks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_controls(self) -> int:
        """N_ctrl."""
        return int(self.ctrl_of_cell.max()) + 1

    @property
    def n_cells(self) -> int:
        """N."""
        return int(self.ctrl_of_cell.size)

    @property
    def n_states(self) -> int:
        """S."""
        return int(np.asarray(self.gamma_ref).size)

    def _finish(self, phys: np.ndarray, enabled: bool) -> PanelConfig:
        if phys.size and (int(phys.max()) >= self.n_states or int(phys.min()) < -1):
            raise ConfigurationError(
                f"panel {self.panel}: state index outside [-1, {self.n_states})"
            )
        return PanelConfig(self.panel, self.failures.apply(phys), enabled)

    def from_states(self, states: np.ndarray, *, enabled: bool = True) -> PanelConfig:
        """Per-control states (N_ctrl,) -> ``PanelConfig`` (expansion + failures)."""
        return self._finish(expand_states(np.asarray(states, np.int16), self.ctrl_of_cell), enabled)

    def from_phases(self, phases_rad: np.ndarray, *, enabled: bool = True) -> PanelConfig:
        """Per-control phases (rad) -> quantized -> expanded -> failures."""
        s = quantize_phases(np.asarray(phases_rad, np.float64), self.gamma_ref)
        if s.size != self.n_controls:
            raise ConfigurationError(f"{s.size} phases for {self.n_controls} controls")
        return self.from_states(s, enabled=enabled)

    def register_codebook(self, name: str, entries: np.ndarray) -> None:
        """Register a (K, N_ctrl) codebook of states."""
        e = np.asarray(entries, dtype=np.int16)
        if e.ndim != 2 or e.shape[1] != self.n_controls:
            raise ConfigurationError(f"codebook must be (K, {self.n_controls})")
        self.codebooks[name] = e

    def from_codebook(
        self, index: int, name: str = "default", *, enabled: bool = True
    ) -> PanelConfig:
        """Codebook entry -> ``PanelConfig``."""
        if name not in self.codebooks:
            raise ConfigurationError(f"panel {self.panel}: unknown codebook {name!r}")
        cb = self.codebooks[name]
        if not 0 <= index < cb.shape[0]:
            raise ConfigurationError(f"codebook index {index} outside [0, {cb.shape[0]})")
        return self.from_states(cb[index], enabled=enabled)


# --------------------------------------------------------------------------- analytic oracles


def eta_binary(delta_rad: float) -> float:
    """Large-N loss of an equal-amplitude binary alphabet separated by ``delta``:
    ``eta = (2 sin(delta'/2) / pi)^2`` with ``delta' = min(delta, 2 pi - delta)``."""
    d = math.fmod(abs(delta_rad), 2 * math.pi)
    dp = min(d, 2 * math.pi - d)
    return (2.0 * math.sin(dp / 2.0) / math.pi) ** 2


def eta_bits(b: int) -> float:
    """Large-N loss of the uniform b-bit alphabet: ``(2^b/pi sin(pi/2^b))^2``."""
    m = 2**b
    return (m / math.pi * math.sin(math.pi / m)) ** 2


def eta_alphabet(gammas: Sequence[complex] | np.ndarray) -> float:
    """Large-N loss of an arbitrary alphabet relative to an ideal continuous unit phase:

    ``eta = (E_phi[max_s Re{Gamma_s exp(-j phi)}])^2`` with ``phi`` uniform on [0, 2 pi),
    i.e. the per-element best choice for uniformly distributed channel phases. The upper
    envelope of the sinusoids is integrated exactly between its crossing points.
    """
    g = np.asarray(gammas, dtype=np.complex128).ravel()
    if g.size == 0:
        raise ConfigurationError("empty alphabet")
    amp, ang = np.abs(g), np.angle(g)
    # breakpoints: angles where two sinusoids a cos(phi - x) and b cos(phi - y) cross
    pts = [0.0, 2 * math.pi]
    for i in range(g.size):
        for j in range(i + 1, g.size):
            d = g[i] - g[j]  # Re{d e^{-j phi}} = 0 at phi = arg(d) +- pi/2
            if abs(d) > 0:
                for off in (math.pi / 2, -math.pi / 2):
                    pts.append(math.fmod(float(np.angle(d)) + off + 4 * math.pi, 2 * math.pi))
    pts = sorted(set(pts))
    total = 0.0
    for a, b in itertools.pairwise(pts):
        if b - a <= 0:
            continue
        mid = 0.5 * (a + b)
        k = int(np.argmax(amp * np.cos(mid - ang)))
        total += amp[k] * (math.sin(b - ang[k]) - math.sin(a - ang[k]))
    return (total / (2 * math.pi)) ** 2


def eta_abeywickrama(beta_min: float, alpha: float, phi_rad: float = 0.0) -> float:
    """Large-N loss of the Abeywickrama practical phase shifter with continuous phase:
    ``eta = (mean_theta beta(theta))^2`` with ``beta = (1 - beta_min)((sin(theta - phi) +
    1)/2)^alpha + beta_min``; the mean is ``Gamma(alpha+1/2)/(sqrt(pi) Gamma(alpha+1))`` of
    the shaped part (closed form, independent of ``phi``)."""
    del phi_rad  # the mean over a full period does not depend on phi
    shaped = math.exp(math.lgamma(alpha + 0.5) - math.lgamma(alpha + 1.0)) / math.sqrt(math.pi)
    return ((1.0 - beta_min) * shaped + beta_min) ** 2


def binary_md(a: complex, b: complex) -> tuple[complex, complex]:
    """Binary decomposition ``Gamma = m + d s`` with ``s = +1`` for ``a`` and ``-1`` for ``b``:
    ``m = (a + b)/2`` (fixed mirror term), ``d = (a - b)/2`` (steerable term)."""
    return (complex(a) + complex(b)) / 2.0, (complex(a) - complex(b)) / 2.0


def md_channel(z0: complex, c: np.ndarray, m: complex, d: complex, signs: np.ndarray) -> complex:
    """``z0 + m sum_n c_n + d sum_n c_n s_n`` for ``s_n = +/-1``."""
    cc = np.asarray(c, dtype=np.complex128)
    s = np.asarray(signs, dtype=np.float64)
    if np.any(np.abs(s) != 1.0):
        raise ConfigurationError("signs must be +1 or -1")
    return complex(z0) + m * complex(np.sum(cc)) + d * complex(np.sum(cc * s))
