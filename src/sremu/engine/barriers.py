"""Controller barriers (DESIGN.md sections 4.9, 8.7; I8, RT-08, RT-17).

* A registered controller with period ``T_c`` creates barriers at ``t_b = m*T_c`` (``m``
  from ``ceil(now/T_c)`` at registration).
* The engine computes every RX up to ``to_ticks(t_b + L_min, f_r)`` (exclusive), where
  ``L_min`` is the controller's declared minimum latency (conservative-PDES lookahead), and
  then runs the barrier. The :class:`~sremu.contracts.Observation` uses only samples
  ``< to_ticks(t_b, f_r)``.
* Actions default to ``at = t_b + max(latency, L_min)``; ``latency`` is ``fixed_s`` plus a
  jitter drawn from the keyed RNG purpose ``ctrl.latency/<name>`` (counter = barrier index
  m, lane = action index), converted to an exact time with ``Fraction(repr(x))``. An action
  with ``at < t_b + L_min`` raises ``LookaheadViolation`` (in-process); a timed command from
  a barrier-registered source earlier than ``t_b + L_min`` of its latest barrier is rejected
  with ``LOOKAHEAD_VIOLATION`` (control socket, RT-17).
* Because the frontier limit and the command mapping both use ``to_ticks`` (monotone) and
  tau >= 0, lateness is 0 by construction (I8).

**Coupling matrix (4.9).** lockstep + barrier (default, deterministic); lockstep +
unregistered timed commands (deterministic only if on time); paced + async; paced + barrier
for in-process controllers only; lockstep + async is rejected.

No wall clock here except the remote ack timeout, which the caller passes in.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from sremu import rng
from sremu.contracts import (
    Action,
    CommandRequest,
    CommandTime,
    Controller,
    CsiView,
    LatencyModel,
    LinkId,
    ObservationSpec,
    PanelId,
    PropagationModel,
    RemoteControllerProxy,
    RisCapabilities,
    Snapshot,
)
from sremu.errors import ConfigurationError, LookaheadViolation
from sremu.timebase import parse_exact, to_exact, to_ticks

__all__ = [
    "BarrierManager",
    "ControllerSlot",
    "action_request",
    "csi_view",
    "draw_latency_s",
]

_POSITION_COUNTER_BASE = 1 << 32  # ctrl/<name> generator counters for position noise


def draw_latency_s(seed: int, name: str, latency: LatencyModel, m: int, i: int = 0) -> Fraction:
    """Latency of action ``i`` at barrier ``m``: ``fixed_s`` + keyed jitter (exact, s).

    ``uniform:a,b`` -> ``a + (b - a)*u``; ``exp:mean`` -> ``-mean*ln(1 - u)``, with ``u``
    from ``rng.uniform(derive_key(seed, "ctrl.latency", name), m, 1, sub=i)``. The float
    jitter enters as ``Fraction(repr(x))`` (R5).
    """
    jit = latency.parsed_jitter()
    if jit is None:
        return Fraction(latency.fixed_s)
    key = rng.derive_key(seed, "ctrl.latency", name)
    u = float(rng.uniform(key, int(m), 1, sub=int(i))[0])
    kind, args = jit
    if kind == "uniform":
        a, b = float(args[0]), float(args[1])
        x = a + (b - a) * u
    else:
        x = -float(args[0]) * math.log1p(-u)
    return Fraction(latency.fixed_s) + to_exact(x, name="latency jitter")


def action_request(
    name: str, m: int, i: int, a: Action, at: Fraction, late_policy: str = "apply_on_arrival"
) -> CommandRequest:
    """The ``CommandRequest`` of controller action ``i`` at barrier ``m`` (``late_policy``:
    the run's ``run.late_policy``)."""
    cid = f"{name}:{m}:{i}"
    if (
        a.enabled is not None
        and a.states is None
        and a.phases_rad is None
        and a.codebook_index is None
    ):
        return CommandRequest(
            cid,
            "ris.set_enabled",
            name,
            a.panel,
            CommandTime.at(at),
            enabled=a.enabled,
            late_policy=late_policy,  # type: ignore[arg-type]
        )
    return CommandRequest(
        cid,
        "ris.set_config",
        name,
        a.panel,
        CommandTime.at(at),
        states=a.states,
        phases_rad=a.phases_rad,
        codebook_index=a.codebook_index,
        transition=a.transition,
        late_policy=late_policy,  # type: ignore[arg-type]
    )


def csi_view(model: PropagationModel, snap: Snapshot, link: LinkId, fc_hz: float) -> CsiView:
    """Oracle CSI of one link (``h_direct`` and per-panel (S, N) cascades with the carrier
    phase), from the contract objects (the model's own ``csi_view`` when it has one)."""
    own = getattr(model, "csi_view", None)
    if callable(own):
        view: CsiView = own(snap, link)
        return view
    pb = snap.paths[link]
    h0 = complex(np.sum(pb.amp * np.exp(-2j * np.pi * np.mod(fc_hz * pb.delay_s, 1.0))))
    cascade: dict[PanelId, NDArray[np.complex128]] = {}
    for pid, rc in snap.ris.items():
        acc = np.zeros((rc.n_states, rc.n_cells), dtype=np.complex128)
        for r, (rlk, _v) in enumerate(rc.rows):
            if rlk != link:
                continue
            ph = np.exp(-2j * np.pi * np.mod(fc_hz * rc.tile_delay_s[r][rc.tile_of_cell], 1.0))
            if rc.B is not None:
                acc += rc.B[r] * ph[None, :]
            else:
                assert rc.C is not None
                assert rc.gamma is not None
                acc += rc.gamma[:, None] * (rc.C[r] * ph)[None, :]
        cascade[pid] = acc
    return CsiView(link, snap.t, fc_hz, h0, cascade)


@dataclass(eq=False)
class ControllerSlot:
    """A registered controller and its barrier schedule."""

    ctrl: Controller | RemoteControllerProxy
    name: str
    period: Fraction
    l_min: Fraction
    spec: ObservationSpec
    latency: LatencyModel
    remote: bool
    m: int
    order: int
    t_b_last: Fraction | None = None

    @property
    def t_next(self) -> Fraction:
        """Time of the next barrier, ``m*T_c``."""
        return self.m * self.period

    def limit(self, f: Fraction) -> int:
        """RX frontier limit ``to_ticks(t_b + L_min, f)`` of the next barrier (exclusive)."""
        return to_ticks(self.t_next + self.l_min, f)

    def lookahead_bound(self) -> Fraction | None:
        """``t_b + L_min`` of the latest barrier (None before the first one)."""
        return None if self.t_b_last is None else self.t_b_last + self.l_min


class BarrierManager:
    """Barrier bookkeeping for every registered controller (see the module docstring)."""

    def __init__(self, seed: int, *, mode: str = "lockstep", coupling: str = "barrier") -> None:
        if mode == "lockstep" and coupling == "async":
            raise ConfigurationError(
                "lockstep + async coupling is rejected: non-deterministic by construction "
                "(section 4.9 coupling matrix)"
            )
        self.seed = int(seed)
        self.mode = mode
        self.coupling = coupling
        self.slots: list[ControllerSlot] = []

    def __len__(self) -> int:
        return len(self.slots)

    def register(
        self,
        ctrl: Controller | RemoteControllerProxy,
        now: Fraction,
        caps: Mapping[PanelId, RisCapabilities],
    ) -> ControllerSlot:
        """Register a controller; in-process controllers get ``setup`` with the keyed
        generator ``ctrl/<name>``."""
        name = str(ctrl.name)
        if any(s.name == name for s in self.slots):
            raise ConfigurationError(f"controller {name!r} is already registered")
        period = parse_exact(ctrl.period_s, name=f"{name}.period_s")
        if period <= 0:
            raise ConfigurationError(f"controller {name!r}: period_s must be > 0")
        lmin = parse_exact(ctrl.min_latency_s, name=f"{name}.min_latency_s")
        if lmin < 0:
            raise ConfigurationError(f"controller {name!r}: min_latency_s must be >= 0")
        remote = not isinstance(ctrl, Controller) and isinstance(ctrl, RemoteControllerProxy)
        if self.mode == "paced" and self.coupling == "barrier" and remote:
            raise ConfigurationError(
                "paced + barrier coupling allows in-process controllers only (section 4.9)"
            )
        if self.mode == "paced" and self.coupling == "async":
            raise ConfigurationError(
                "paced + async: controllers act through the control socket, not barriers"
            )
        lat = ctrl.latency if isinstance(ctrl.latency, LatencyModel) else LatencyModel()
        if not remote:
            ctrl.setup(dict(caps), rng.generator(self.seed, "ctrl", name))  # type: ignore[union-attr]
        m0 = math.ceil(now / period)
        slot = ControllerSlot(
            ctrl, name, period, lmin, ctrl.spec, lat, remote, int(m0), len(self.slots)
        )
        self.slots.append(slot)
        return slot

    def unregister(self, name: str) -> None:
        """Remove a controller (its pending barrier disappears)."""
        self.slots = [s for s in self.slots if s.name != name]

    def slot(self, name: str) -> ControllerSlot | None:
        """The slot of controller ``name`` (None when not registered)."""
        for s in self.slots:
            if s.name == name:
                return s
        return None

    def limit(self, f: Fraction) -> int | None:
        """Frontier limit at rate ``f``: min over controllers of ``to_ticks(t_b + L_min, f)``."""
        if not self.slots:
            return None
        return min(s.limit(f) for s in self.slots)

    def due(self, cursors: Sequence[tuple[Fraction, int, int | None]]) -> ControllerSlot | None:
        """Earliest barrier whose limit every group has reached.

        ``cursors``: per rate group ``(rate, cursor, end)``; an ended group counts as
        having reached every limit. Ties: earlier ``t_b``, then registration order.
        """
        best: ControllerSlot | None = None
        for s in sorted(self.slots, key=lambda s: (s.t_next, s.order)):
            ok = all(
                (end is not None and cur >= end) or cur >= s.limit(f) for f, cur, end in cursors
            )
            if ok:
                best = s
                break
        return best

    def lookahead_bound(self, src: str) -> Fraction | None:
        """``t_b + L_min`` of the latest barrier of a barrier-registered source."""
        s = self.slot(src)
        return None if s is None else s.lookahead_bound()

    def action_time(self, s: ControllerSlot, t_b: Fraction, i: int, a: Action) -> Fraction:
        """Switch time of action ``i``: ``a.at_s`` or ``t_b + max(latency, L_min)``;
        raises ``LookaheadViolation`` when earlier than ``t_b + L_min``."""
        if a.at_s is not None:
            at = Fraction(a.at_s)
        else:
            at = t_b + max(draw_latency_s(self.seed, s.name, s.latency, s.m, i), s.l_min)
        if at < t_b + s.l_min:
            raise LookaheadViolation(
                f"{s.name}: action at {at} < t_b + L_min = {t_b + s.l_min} (section 4.9)"
            )
        return at

    def observation_fields(self, s: ControllerSlot) -> tuple[str, ...]:
        """The fields the controller asked for."""
        return tuple(s.spec.fields)

    def position_noise(
        self, s: ControllerSlot, nodes: Sequence[str]
    ) -> dict[str, NDArray[np.float64]]:
        """Keyed additive position noise (std ``position_noise_m``) per node at barrier m."""
        if s.spec.position_noise_m <= 0.0:
            return {n: np.zeros(3) for n in nodes}
        gen = rng.generator(self.seed, "ctrl", s.name, counter=_POSITION_COUNTER_BASE + s.m)
        out = {}
        for n in nodes:
            out[n] = gen.normal(0.0, s.spec.position_noise_m, 3)
        return out
