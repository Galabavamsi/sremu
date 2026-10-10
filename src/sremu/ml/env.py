"""Gym-style RIS environment ``RisEnv`` (DESIGN.md section 8.8; WP9).

``RisEnv(scenario, panel="ris0", decision_period_s="1e-3", action="states" | "codebook",
reward="rx_power_dbm", traffic="pilot_tone", backend="batch")`` with
``reset(seed=None, options=None) -> (obs, info)`` and
``step(action) -> (obs, reward, terminated, truncated, info)``.

It subclasses ``gymnasium.Env`` when gymnasium imports and duck-types it otherwise (the
spaces then come from this module: :class:`Box`, :class:`Discrete`, :class:`MultiDiscrete`,
:class:`Dict`). No vectorized or GPU environment exists in release 1.

**Backend ``"batch"``** (the coefficient path, section 4.11). The scene is the scenario's
``PropagationModel`` (``sremu.physics.propagation.from_scenario``) or a model passed in.
Decision k covers ``[t_k, t_k + T)`` with ``t_k = k T`` (exact). The action is applied at
``t_k + latency_s``; before that instant the previous configuration holds. The reward is the
received pilot-tone power averaged over ``eval_points`` evaluation times
``t_k + (i + 1/2) T/E`` (exact midpoints of E equal sub-intervals):

    P = 10 log10( mean_i P_T |h(t_i; cfg(t_i))|^2 + P_noise )  [dBm]

with ``h`` the narrowband coefficient ``sum_p amp_p exp(-j 2 pi f_c tau_p)`` of
``combine(snapshot(t_i), cfg)`` for the target link, ``P_T`` the TX stream's full-scale
power times its gain (a unit-amplitude pilot tone, section 3.3) and ``P_noise`` the
expected RX noise power ``k_B T_0 F f_s`` when ``include_noise`` (otherwise 0). The batch
reward is the expectation; no noise sample is drawn.

The IQ backends (``"numba_cpu"``/``"numpy_ref"``) need the engine (WP6); until it exists
they raise ``ConfigurationError``.

**Observation** (a dict): ``rx_power_dbm`` (1,) float64, the power of the last decision
period (at ``reset``: the initial configuration evaluated at t = 0); ``ris_state``
(N_ctrl,) int64, the control states in effect at the end of the period; ``positions``
(nodes x 3) float64 m at the end of the period when requested in ``observe``.

**Info**: ``t_s`` (exact end time as a string), ``h`` (complex coefficients at the
evaluation times), ``optimum_dbm`` and ``regret_db``: the exact finite-alphabet optimum
(section 5.13) at every evaluation time from the oracle CSI, averaged like the reward,
and its gap to the achieved power (the regret reference of section 8.8; it ignores panel
failures, so it can be negative when cells are dead). ``oracle_reference`` is True.

Randomness: ``reset(seed)`` keys the space samplers (purpose ``ctrl``, names
``env/<panel>/action`` and ``env/<panel>/observation/<key>``; :func:`key_space`, also
with gymnasium, whose own ``space.seed`` is bypassed) and ``np_random`` (``env/<panel>``)
and, for an environment built from a scenario, rebuilds the scene with that scenario seed
(random waypoints and failure maps are keyed by it, section 6.2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sremu import rng as _rng
from sremu.contracts import (
    LinkId,
    PanelConfig,
    PanelId,
    PropagationModel,
    RisCapabilities,
    split_link_id,
)
from sremu.control.controller import (
    aggregate_controls,
    csi_from_snapshot,
    expand_controls,
    narrowband_coefficient,
)
from sremu.errors import ConfigurationError, UnknownTarget
from sremu.physics.optimize import exact_alphabet_optimum
from sremu.timebase import parse_exact
from sremu.units import K_B, T0

__all__ = [
    "HAVE_GYMNASIUM",
    "Box",
    "Dict",
    "Discrete",
    "MultiDiscrete",
    "RisEnv",
    "env_base",
    "key_space",
    "rollout",
]

try:  # optional dependency (extra "ml"); RisEnv duck-types gymnasium.Env without it
    import gymnasium as _gym  # type: ignore[import-not-found, unused-ignore]

    HAVE_GYMNASIUM = True
except ImportError:  # pragma: no cover - depends on the environment
    _gym = None
    HAVE_GYMNASIUM = False


# =========================================================================== duck-typed spaces


def key_space(space: Any, seed: int | None, name: str) -> None:
    """Key the sampler of ``space`` (a gymnasium space or a stand-in) and, recursively, of
    its named sub-spaces: purpose ``ctrl``, name ``name`` (sub-space ``k``:
    ``<name>/<k>``), master seed ``seed`` (None -> 0). gymnasium's own ``space.seed``
    derives its generator from a ``SeedSequence``; this keeps every sampler on the keyed
    ``sremu.rng`` streams (section 6.1) with or without gymnasium."""
    s = 0 if seed is None else int(seed)
    space._np_random = _rng.generator(s, "ctrl", name)
    subs = getattr(space, "spaces", None)
    if isinstance(subs, Mapping):
        for k, sub in subs.items():
            key_space(sub, s, f"{name}/{k}")


class _Space:
    """Minimal stand-in for ``gymnasium.spaces.Space`` (keyed sampling, membership).

    ``key_name`` is the RNG name of the sampler (purpose ``ctrl``); :class:`RisEnv` sets
    distinct names per space, so no two samplers share a stream."""

    key_name: str = "env/space"

    def __init__(self) -> None:
        self._np_random: np.random.Generator = _rng.generator(0, "ctrl", self.key_name)

    def seed(self, seed: int | None = None) -> list[int]:
        """Key the sampler: purpose ``ctrl``, name :attr:`key_name` (``seed`` None -> 0)."""
        key_space(self, seed, self.key_name)
        return [0 if seed is None else int(seed)]

    def __contains__(self, x: Any) -> bool:
        return self.contains(x)

    def contains(self, x: Any) -> bool:  # pragma: no cover - abstract
        """Whether ``x`` belongs to the space."""
        raise NotImplementedError


class Box(_Space):
    """Real box ``[low, high]`` of a fixed shape (float64)."""

    def __init__(self, low: float, high: float, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.low, self.high, self.shape, self.dtype = (
            float(low),
            float(high),
            tuple(shape),
            np.float64,
        )

    def sample(self) -> np.ndarray:
        """Uniform in the box (finite bounds) or standard normal (infinite bounds)."""
        if math.isfinite(self.low) and math.isfinite(self.high):
            return self._np_random.uniform(self.low, self.high, self.shape)
        return self._np_random.standard_normal(self.shape)

    def contains(self, x: Any) -> bool:
        """Shape and bounds check."""
        a = np.asarray(x, dtype=np.float64)
        return a.shape == self.shape and bool(np.all((a >= self.low) & (a <= self.high)))


class Discrete(_Space):
    """Integers ``0 .. n-1``."""

    def __init__(self, n: int) -> None:
        super().__init__()
        if int(n) < 1:
            raise ConfigurationError("Discrete space needs n >= 1")
        self.n, self.shape, self.dtype = int(n), (), np.int64

    def sample(self) -> int:
        """Uniform index."""
        return int(self._np_random.integers(0, self.n))

    def contains(self, x: Any) -> bool:
        """Integer in range."""
        try:
            v = int(x)
        except (TypeError, ValueError):
            return False
        return v == x and 0 <= v < self.n


class MultiDiscrete(_Space):
    """Integer vectors with ``0 <= x[i] < nvec[i]``."""

    def __init__(self, nvec: Sequence[int] | np.ndarray) -> None:
        super().__init__()
        self.nvec = np.asarray(nvec, dtype=np.int64)
        self.shape, self.dtype = self.nvec.shape, np.int64

    def sample(self) -> np.ndarray:
        """Uniform integer vector."""
        return self._np_random.integers(0, self.nvec).astype(np.int64)

    def contains(self, x: Any) -> bool:
        """Shape, integrality and range check."""
        a = np.asarray(x)
        if a.shape != self.shape or not np.all(np.equal(np.mod(a, 1), 0)):
            return False
        return bool(np.all((a >= 0) & (a < self.nvec)))


class Dict(_Space):
    """A mapping of named spaces."""

    def __init__(self, spaces: Mapping[str, _Space]) -> None:
        super().__init__()
        self.spaces = dict(spaces)

    def sample(self) -> dict[str, Any]:
        """A sample of every sub-space."""
        return {k: sp.sample() for k, sp in self.spaces.items()}  # type: ignore[attr-defined]

    def contains(self, x: Any) -> bool:
        """Same keys and every value in its space."""
        return (
            isinstance(x, Mapping)
            and set(x) == set(self.spaces)
            and all(self.spaces[k].contains(v) for k, v in x.items())
        )

    def __getitem__(self, key: str) -> _Space:
        return self.spaces[key]


def _spaces() -> Any:
    """gymnasium.spaces when available, else this module's stand-ins."""
    if HAVE_GYMNASIUM:
        return _gym.spaces
    import sys

    return sys.modules[__name__]


def env_base() -> type:
    """``gymnasium.Env`` when gymnasium imports, else ``object``."""
    return _gym.Env if HAVE_GYMNASIUM else object


# =========================================================================== scenario helpers


def _scenario_mapping(scenario: Any) -> tuple[dict[str, Any] | None, Path | None]:
    if isinstance(scenario, str | Path):
        from sremu.scenario.load import read_document

        path = Path(scenario)
        doc = read_document(path)
        if not isinstance(doc, Mapping):
            raise ConfigurationError(f"{path}: a scenario must be a mapping")
        return dict(doc), path.parent
    if hasattr(scenario, "model_dump"):
        return dict(scenario.model_dump(mode="python")), None
    if isinstance(scenario, Mapping):
        return dict(scenario), None
    return None, None


def _stream_entry(doc: Mapping[str, Any], sid: str) -> dict[str, Any]:
    node, _, name = sid.partition(".")
    for nd in doc.get("nodes") or []:
        if str(nd.get("id")) == node:
            for s in nd.get("streams") or []:
                if str(s.get("name")) == name:
                    return dict(s)
    raise UnknownTarget(f"stream {sid!r} not in the scenario")


# =========================================================================== environment


class RisEnv(env_base()):  # type: ignore[misc]
    """Gym-style environment over one RIS panel (section 8.8); see the module docstring.

    Args:
        scenario: scenario file path, mapping or pydantic ``Scenario``; or a
            ``PropagationModel`` (then ``tx_power_dbm`` and ``noise_floor_dbm`` are explicit).
        panel: controlled panel.
        decision_period_s: T, s (exact string/int/Fraction).
        action: ``"states"`` (N_ctrl state indices) or ``"codebook"`` (an index into
            ``codebook``, (K, N_ctrl)).
        reward: ``"rx_power_dbm"``.
        traffic: ``"pilot_tone"``.
        backend: ``"batch"`` (``"numba_cpu"``/``"numpy_ref"`` need the WP6 engine).
        link: target link (default: the only link with rows on the panel, else the first).
        codebook: (K, N_ctrl) states for ``action="codebook"``.
        horizon_s: episode length, s (truncation); default the scenario's
            ``run.duration_s`` (None: never truncated).
        eval_points: E reward evaluation times per period (>= 1).
        latency_s: action application delay within the period, s (0 <= latency < T).
        include_noise: add the RX stream's expected noise power ``k_B T_0 F f_s``.
        tx_power_dbm, noise_floor_dbm: overrides (required for a bare model).
        control_of_cell: (N,) control index per cell when the model has no ``pipeline``.
        observe: observation keys, from ``rx_power_dbm``, ``ris_state``, ``positions``.
        seed: initial seed of ``reset`` when ``reset`` gets none.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012 - gymnasium convention

    def __init__(
        self,
        scenario: Any,
        panel: PanelId = "ris0",
        decision_period_s: Fraction | int | str = "1e-3",
        action: Literal["states", "codebook"] = "states",
        reward: Literal["rx_power_dbm"] = "rx_power_dbm",
        traffic: Literal["pilot_tone"] = "pilot_tone",
        backend: Literal["batch", "numba_cpu", "numpy_ref"] = "batch",
        *,
        link: LinkId | None = None,
        codebook: np.ndarray | Sequence[Sequence[int]] | None = None,
        horizon_s: Fraction | int | str | None = None,
        eval_points: int = 1,
        latency_s: Fraction | int | str = 0,
        include_noise: bool = False,
        tx_power_dbm: float | None = None,
        noise_floor_dbm: float | None = None,
        control_of_cell: Sequence[int] | np.ndarray | None = None,
        observe: Sequence[str] = ("rx_power_dbm", "ris_state"),
        seed: int | None = None,
    ) -> None:
        if backend != "batch":
            raise ConfigurationError(
                f"RisEnv backend {backend!r} needs the IQ engine (WP6), which is not available "
                "in this build; use backend='batch'"
            )
        if reward != "rx_power_dbm":
            raise ConfigurationError(f"unknown reward {reward!r} (release 1: 'rx_power_dbm')")
        if traffic != "pilot_tone":
            raise ConfigurationError(f"unknown traffic {traffic!r} (release 1: 'pilot_tone')")
        if action not in ("states", "codebook"):
            raise ConfigurationError(f"unknown action kind {action!r}")
        bad = set(observe) - {"rx_power_dbm", "ris_state", "positions"}
        if bad or not observe:
            raise ConfigurationError(f"unknown or empty observation keys {sorted(bad)}")
        self.panel = panel
        self.backend = backend
        self.action_kind = action
        self.observe = tuple(observe)
        self.period = parse_exact(decision_period_s, name="decision_period_s")
        if self.period <= 0:
            raise ConfigurationError("decision_period_s must be > 0")
        self.latency = parse_exact(latency_s, name="latency_s")
        if not 0 <= self.latency < self.period:
            raise ConfigurationError("need 0 <= latency_s < decision_period_s")
        if int(eval_points) < 1:
            raise ConfigurationError("eval_points must be >= 1")
        self.eval_points = int(eval_points)
        self._doc, self._base_dir = _scenario_mapping(scenario)
        self._include_noise = bool(include_noise)
        self._tx_override = tx_power_dbm
        self._noise_override = noise_floor_dbm
        self._link_arg = link
        self._coc_arg = None if control_of_cell is None else np.asarray(control_of_cell, np.int64)
        self._default_seed = seed
        if self._doc is None:
            if not isinstance(scenario, PropagationModel):
                raise ConfigurationError("scenario must be a path, mapping, Scenario or model")
            self.model: PropagationModel = scenario
            self._model_seed: int | None = None
        else:
            self._model_seed = int(self._doc.get("seed", 0))
            self.model = self._build_model(self._model_seed)
        if horizon_s is not None:
            self.horizon: Fraction | None = parse_exact(horizon_s, name="horizon_s")
        elif self._doc is not None and (self._doc.get("run") or {}).get("duration_s") is not None:
            self.horizon = parse_exact(str(self._doc["run"]["duration_s"]), name="run.duration_s")
        else:
            self.horizon = None
        self._configure()
        if action == "codebook":
            if codebook is None:
                raise ConfigurationError("action='codebook' needs codebook=(K, N_ctrl)")
            cb = np.asarray(codebook)
            if cb.ndim != 2 or cb.shape[1] != self.n_controls or cb.shape[0] < 1:
                raise ConfigurationError(f"codebook must be (K, {self.n_controls})")
            if int(cb.min()) < 0 or int(cb.max()) >= self.n_states:
                raise ConfigurationError(f"codebook states must lie in [0, {self.n_states})")
            self.codebook: np.ndarray | None = cb.astype(np.int16)
        else:
            self.codebook = None
        sp = _spaces()
        if action == "states":
            self.action_space = sp.MultiDiscrete(np.full(self.n_controls, self.n_states))
        else:
            assert self.codebook is not None
            self.action_space = sp.Discrete(int(self.codebook.shape[0]))
        obs_sp: dict[str, Any] = {}
        if "rx_power_dbm" in self.observe:
            obs_sp["rx_power_dbm"] = (
                sp.Box(-np.inf, np.inf, (1,), np.float64)
                if HAVE_GYMNASIUM
                else Box(-math.inf, math.inf, (1,))
            )
        if "ris_state" in self.observe:
            obs_sp["ris_state"] = sp.MultiDiscrete(np.full(self.n_controls, self.n_states))
        if "positions" in self.observe:
            shape = (len(self._nodes), 3)
            obs_sp["positions"] = (
                sp.Box(-np.inf, np.inf, shape, np.float64)
                if HAVE_GYMNASIUM
                else Box(-math.inf, math.inf, shape)
            )
        self.observation_space = sp.Dict(obs_sp)
        self._key_spaces(seed)
        self._t = Fraction(0)
        self._ctrl = np.zeros(self.n_controls, dtype=np.int16)
        self._ready = False

    # ------------------------------------------------------------------ construction
    def _key_spaces(self, seed: int | None) -> None:
        """Key the action and observation samplers (purpose ``ctrl``, names
        ``env/<panel>/action`` and ``env/<panel>/observation[/<key>]``) and the episode
        generator ``np_random`` (``env/<panel>``)."""
        key_space(self.action_space, seed, f"env/{self.panel}/action")
        key_space(self.observation_space, seed, f"env/{self.panel}/observation")
        self.np_random = _rng.generator(
            0 if seed is None else int(seed), "ctrl", f"env/{self.panel}"
        )

    def _build_model(self, seed: int) -> PropagationModel:
        from sremu.physics.propagation import from_scenario

        assert self._doc is not None
        doc = dict(self._doc)
        doc["seed"] = int(seed)
        return from_scenario(doc, base_dir=self._base_dir)

    def _configure(self) -> None:
        m = self.model
        snap0 = m.snapshot(Fraction(0))
        if self.panel not in snap0.ris:
            raise UnknownTarget(f"unknown panel {self.panel!r} (panels: {sorted(snap0.ris)})")
        rc = snap0.ris[self.panel]
        rows = [lk for lk, _v in rc.rows]
        if self._link_arg is not None:
            if self._link_arg not in m.links():
                raise UnknownTarget(f"unknown link {self._link_arg!r}")
            self.link: LinkId = self._link_arg
        else:
            uniq = list(dict.fromkeys(rows)) or list(m.links())
            self.link = uniq[0]
        self.n_cells = rc.n_cells
        self.n_states = rc.n_states
        pipe = getattr(m, "pipeline", None)
        self._pipeline = pipe(self.panel) if callable(pipe) else None
        if self._pipeline is not None:
            self.n_controls = int(self._pipeline.n_controls)
            self._coc: np.ndarray | None = np.asarray(self._pipeline.ctrl_of_cell, np.int64)
        elif self._coc_arg is not None:
            if self._coc_arg.shape != (self.n_cells,):
                raise ConfigurationError(f"control_of_cell must be ({self.n_cells},)")
            self._coc = self._coc_arg
            self.n_controls = int(self._coc.max()) + 1
        else:
            self._coc = None
            self.n_controls = self.n_cells
        self._nodes = sorted(getattr(m, "nodes", {}) or {})
        if "positions" in self.observe and not hasattr(m, "node_position"):
            raise ConfigurationError("observe 'positions' needs a model with node_position()")
        tx, rx = split_link_id(self.link)
        if self._tx_override is not None:
            self.tx_power_dbm = float(self._tx_override)
        elif self._doc is not None:
            s = _stream_entry(self._doc, tx)
            self.tx_power_dbm = float(s.get("full_scale_dbm", 0.0)) + float(s.get("gain_db", 0.0))
        else:
            self.tx_power_dbm = 0.0
        if self._noise_override is not None:
            self.noise_mw = 10.0 ** (float(self._noise_override) / 10.0)
        elif self._include_noise:
            if self._doc is None:
                raise ConfigurationError("include_noise with a bare model needs noise_floor_dbm")
            s = _stream_entry(self._doc, rx)
            fs = float(parse_exact(str(s["rate_sps"]), name="rate_sps"))
            nf = float(s.get("noise_figure_db", 0.0))
            self.noise_mw = K_B * T0 * 10.0 ** (nf / 10.0) * fs * 1e3
        else:
            self.noise_mw = 0.0
        init = getattr(m, "initial_config", None)
        self._initial: PanelConfig | None = init(self.panel) if callable(init) else None

    # ------------------------------------------------------------------ mechanics
    def _config(self, ctrl_states: np.ndarray) -> PanelConfig:
        if self._pipeline is not None:
            return self._pipeline.from_states(ctrl_states)
        return PanelConfig(self.panel, expand_controls(ctrl_states, self._coc))

    def _cfg_map(
        self, cfg: PanelConfig, snap_panels: Sequence[PanelId]
    ) -> dict[PanelId, PanelConfig]:
        out: dict[PanelId, PanelConfig] = {}
        for pid in snap_panels:
            if pid == self.panel:
                out[pid] = cfg
            else:
                init = getattr(self.model, "initial_config", None)
                if callable(init):
                    out[pid] = init(pid)
        return out

    def _evaluate(self, t: Fraction, cfg: PanelConfig) -> tuple[complex, float]:
        """(h, optimum |h|) at exact time ``t`` under ``cfg``."""
        snap = self.model.snapshot(t)
        cfgs = self._cfg_map(cfg, list(snap.ris))
        h = narrowband_coefficient(self.model, snap, cfgs, self.link)
        view = csi_from_snapshot(self.model, snap, self.link)
        z0 = complex(view.h_direct)
        for pid, z in view.cascade.items():
            if pid == self.panel or pid not in cfgs or not cfgs[pid].enabled:
                continue
            s = np.asarray(cfgs[pid].state, dtype=np.int64)
            alive = s >= 0
            z0 += complex(z[s[alive], np.nonzero(alive)[0]].sum())
        zc = aggregate_controls(view.cascade[self.panel], self._coc, self.n_controls)
        opt, _ = exact_alphabet_optimum(z0, zc)
        return h, float(opt)

    def _p_tx_mw(self) -> float:
        return 10.0 ** (self.tx_power_dbm / 10.0)

    def _positions(self, t: Fraction) -> np.ndarray:
        pos = self.model.node_position  # type: ignore[attr-defined]
        return np.stack([np.asarray(pos(n, t), dtype=np.float64) for n in self._nodes])

    def _obs(self, power_dbm: float, t: Fraction) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if "rx_power_dbm" in self.observe:
            out["rx_power_dbm"] = np.array([power_dbm], dtype=np.float64)
        if "ris_state" in self.observe:
            out["ris_state"] = self._ctrl.astype(np.int64)
        if "positions" in self.observe:
            out["positions"] = self._positions(t)
        return out

    @staticmethod
    def _dbm(p_mw: float) -> float:
        return 10.0 * math.log10(p_mw) if p_mw > 0 else -math.inf

    def _action_states(self, action: Any) -> np.ndarray:
        if self.action_kind == "codebook":
            assert self.codebook is not None
            try:
                i = int(action)
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"codebook action must be an int, got {action!r}"
                ) from None
            if i != action or not 0 <= i < self.codebook.shape[0]:
                raise ConfigurationError(
                    f"codebook index {action!r} outside [0, {self.codebook.shape[0]})"
                )
            return self.codebook[i].copy()
        a = np.asarray(action)
        if a.shape != (self.n_controls,) or (a.size and not np.all(np.equal(np.mod(a, 1), 0))):
            raise ConfigurationError(f"states action must be ({self.n_controls},) integers")
        if a.size and (int(a.min()) < 0 or int(a.max()) >= self.n_states):
            raise ConfigurationError(f"states must lie in [0, {self.n_states})")
        return a.astype(np.int16)

    # ------------------------------------------------------------------ gym API
    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start an episode at t = 0 with the panel's initial configuration."""
        s = self._default_seed if seed is None else seed
        if s is not None:
            s = int(s)
            if self._doc is not None and s != self._model_seed:
                self.model = self._build_model(s)
                self._model_seed = s
                self._configure()
            self._key_spaces(s)
        self._t = Fraction(0)
        if self._initial is not None:
            st = np.asarray(self._initial.state, dtype=np.int64)
            ctrl = np.zeros(self.n_controls, dtype=np.int16)
            if self._coc is None:
                ctrl = np.where(st >= 0, st, 0).astype(np.int16)
            else:
                seen = np.zeros(self.n_controls, dtype=bool)
                for n, g in enumerate(self._coc):  # state of the first live cell per control
                    if st[n] >= 0 and not seen[g]:
                        ctrl[g] = st[n]
                        seen[g] = True
            self._ctrl = ctrl
            cfg = self._initial
        else:
            self._ctrl = np.zeros(self.n_controls, dtype=np.int16)
            cfg = self._config(self._ctrl)
        self._cfg = cfg
        h, opt = self._evaluate(Fraction(0), cfg)
        p = self._p_tx_mw() * abs(h) ** 2 + self.noise_mw
        self._last_power_dbm = self._dbm(p)
        self._ready = True
        info = {
            "t_s": str(self._t),
            "h": [h],
            "optimum_dbm": self._dbm(self._p_tx_mw() * opt * opt + self.noise_mw),
            "oracle_reference": True,
            "link": self.link,
            "seed": s,
        }
        return self._obs(self._dbm(p), self._t), info

    def step(self, action: Any) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply ``action`` at ``t + latency_s`` and advance by one decision period."""
        if not self._ready:
            raise ConfigurationError("call reset() before step()")
        states = self._action_states(action)
        return self._advance(self._config(states), states)

    def _advance(
        self, new_cfg: PanelConfig, states: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply ``new_cfg`` (control states ``states``) at ``t + latency_s``; advance."""
        t0 = self._t
        t_apply = t0 + self.latency
        e = self.eval_points
        hs: list[complex] = []
        p_sum = 0.0
        opt_sum = 0.0
        ptx = self._p_tx_mw()
        for i in range(e):
            t = t0 + (2 * i + 1) * self.period / (2 * e)
            cfg = new_cfg if t >= t_apply else self._cfg
            h, opt = self._evaluate(t, cfg)
            hs.append(h)
            p_sum += ptx * abs(h) ** 2
            opt_sum += ptx * opt * opt
        p_mw = p_sum / e + self.noise_mw
        opt_mw = opt_sum / e + self.noise_mw
        reward = self._dbm(p_mw)
        self._cfg = new_cfg
        self._ctrl = np.asarray(states, dtype=np.int16)
        self._t = t0 + self.period
        self._last_power_dbm = reward
        truncated = self.horizon is not None and self._t >= self.horizon
        info = {
            "t_s": str(self._t),
            "h": hs,
            "optimum_dbm": self._dbm(opt_mw),
            "regret_db": self._dbm(opt_mw) - reward,
            "oracle_reference": True,
            "link": self.link,
        }
        return self._obs(reward, self._t), float(reward), False, bool(truncated), info

    def render(self) -> None:
        """No rendering in release 1."""
        return None

    def close(self) -> None:
        """Nothing to release."""
        return None

    # ------------------------------------------------------------------ controller support
    @property
    def t(self) -> Fraction:
        """Current exact time, s (start of the next decision period)."""
        return self._t

    def capabilities(self) -> dict[PanelId, RisCapabilities]:
        """Panel capabilities: the model's ``capabilities(panel)`` when it has one, else
        element granularity with ``S`` analytic states."""
        cap = getattr(self.model, "capabilities", None)
        if callable(cap):
            return {self.panel: cap(self.panel)}
        geo = self.model.panel_geometry(self.panel)
        fc = float(self.model.link_fc_hz(self.link))
        return {
            self.panel: RisCapabilities(
                n_elements=self.n_cells,
                layout=(geo.n_u, geo.n_v),
                control_granularity="element",
                states=tuple(str(s) for s in range(self.n_states)),
                gamma_source="analytic:model",
                operating_band_hz=(fc, fc),
                max_bandwidth_hz=math.inf,
                n_controls=self.n_controls,
            )
        }


def _channel_models(controller: Any) -> list[Any]:
    """Scene models the controller's channel views hold (``channel.model``), through
    wrappers (``inner``)."""
    out = []
    c = controller
    for _ in range(8):  # wrappers nest shallowly
        model = getattr(getattr(c, "channel", None), "model", None)
        if model is not None:
            out.append(model)
        c = getattr(c, "inner", None)
        if c is None:
            break
    return out


def rollout(
    env: RisEnv,
    controller: Any,
    n_steps: int,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Drive a ``Controller`` (section 8.7) through ``env`` for ``n_steps`` periods.

    ``env.reset(seed)``, then ``controller.setup(caps, sremu.rng.generator(seed, "ctrl",
    name))``. Barriers fall at ``t_b = m T_c``; ``T_c`` (``controller.period_s``) must be a
    whole multiple of the decision period. The observation at a barrier carries the power of
    the previous period (``rx_power_dbm`` of the link's RX stream), the panel's per-cell
    state, positions at ``max(0, t_b - position_delay_s)`` when requested, and the oracle CSI
    at ``t_b`` when ``csi_oracle`` is requested. Actions on the panel (``states``,
    ``codebook_index`` with an env codebook, ``phases_rad`` with a physics pipeline,
    ``enabled``) take effect at ``t_b + env.latency_s``; their ``at_s`` and latency jitter are
    not modelled here (this is the environment's coefficient loop, not the WP6 batch path).
    Position noise is not supported. Returns ``rewards_dbm``, ``regret_db`` (arrays),
    ``n_commands`` and ``t_end_s``.

    The controller's master seed is ``seed``, else the environment's default seed, else the
    scenario seed (0 for a bare model). When ``reset(seed)`` rebuilds a scenario's scene
    (new seed, so new random waypoints), a controller whose channel model is the replaced
    scene raises ``ConfigurationError``: build it from ``env.model`` after
    ``env.reset(seed)``.
    """
    from sremu.contracts import Observation
    from sremu.control.controller import setup_controller

    before = env.model
    env.reset(seed=seed)
    if env.model is not before and any(m is before for m in _channel_models(controller)):
        raise ConfigurationError(
            "rollout: reset(seed) rebuilt the scene, but the controller's channel model is the "
            "previous scene; build the controller from env.model after env.reset(seed)"
        )
    if seed is not None:
        s = int(seed)
    elif env._default_seed is not None:
        s = int(env._default_seed)
    else:
        s = int(env._model_seed or 0)
    setup_controller(controller, env.capabilities(), s)
    spec = controller.spec
    if float(spec.position_noise_m) > 0.0:
        raise ConfigurationError("rollout does not model position noise")
    ratio = Fraction(controller.period_s) / env.period
    if ratio.denominator != 1 or ratio < 1:
        raise ConfigurationError(
            f"controller period {controller.period_s} s is not a whole multiple of the decision "
            f"period {env.period} s"
        )
    every = int(ratio)
    rx = split_link_id(env.link)[1]
    rewards: list[float] = []
    regrets: list[float] = []
    n_cmd = 0
    states = env._ctrl.copy()
    cfg = env._cfg
    for k in range(int(n_steps)):
        if k % every == 0:
            t_b = env.t
            positions = None
            if "positions" in spec.fields and hasattr(env.model, "node_position"):
                t_pos = max(Fraction(0), t_b - Fraction(spec.position_delay_s))
                positions = {
                    n: np.asarray(env.model.node_position(n, t_pos), dtype=np.float64)
                    for n in env._nodes
                }
            csi = None
            if "csi_oracle" in spec.fields:
                csi = {env.link: csi_from_snapshot(env.model, env.model.snapshot(t_b), env.link)}
            obs = Observation(
                t_b=t_b,
                k_b={},
                rx_power_dbm={rx: env._last_power_dbm},
                positions=positions,
                csi_oracle=csi,
                iq_window=None,
                ris_state={env.panel: np.asarray(cfg.state, dtype=np.int16).copy()},
                pending=(),
            )
            for a in controller.decide(obs):
                if a.panel != env.panel:
                    raise UnknownTarget(f"rollout: action for unknown panel {a.panel!r}")
                n_cmd += 1
                if a.enabled is not None:
                    cfg = PanelConfig(env.panel, np.asarray(cfg.state, np.int16), bool(a.enabled))
                    continue
                if a.states is not None:
                    states = env._action_states(a.states)
                elif a.codebook_index is not None:
                    if env.codebook is None:
                        raise ConfigurationError("rollout: codebook action without env codebook")
                    states = env.codebook[int(a.codebook_index)].copy()
                else:
                    if env._pipeline is None:
                        raise ConfigurationError("rollout: phases need a physics pipeline")
                    from sremu.physics.nonideal import quantize_phases

                    assert a.phases_rad is not None
                    states = np.asarray(
                        quantize_phases(np.asarray(a.phases_rad), env._pipeline.gamma_ref),
                        dtype=np.int16,
                    )
                c = env._config(states)
                cfg = PanelConfig(env.panel, np.asarray(c.state, np.int16), cfg.enabled)
        _, r, _, _, info = env._advance(cfg, states)
        rewards.append(r)
        regrets.append(float(info["regret_db"]))
    return {
        "rewards_dbm": np.asarray(rewards, dtype=np.float64),
        "regret_db": np.asarray(regrets, dtype=np.float64),
        "n_commands": n_cmd,
        "t_end_s": str(env.t),
    }
