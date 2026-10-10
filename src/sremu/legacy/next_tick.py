"""``LegacyNextTickAdapter``: v2 RIS command semantics (DESIGN.md 5.14; RT-07 teeth, E-B2).

v2 kept ONE pending RIS configuration (a single slot): a new command overwrote whatever was
pending, and the slot was applied at the next engine tick boundary ``ceil(t/tau)*tau``.
Two commands that arrive within the same tick therefore lose the first one (v2 N17).

The adapter maps that behaviour onto the sremu engine without changing it: a command that
arrives when the engine frontier is at time ``t_now`` is scheduled at the next tick
``t_k = (floor(t_now/tau) + 1)*tau`` (as an on-time command at the RIS). Two commands with the
same ``t_k`` hit the engine's same-time tie rule (the later ``seq`` wins and the earlier is
``superseded``), which is exactly the single-slot overwrite. Requested times in the command
are ignored (v2 had no timed RIS commands).

Raises ``NonPhysicalError`` unless ``allow_nonphysical=True``.
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction
from typing import Protocol

from sremu.contracts import CommandReceipt, CommandRequest, CommandTime
from sremu.errors import ConfigurationError, NonPhysicalError
from sremu.timebase import parse_exact

__all__ = ["LegacyNextTickAdapter"]


class _Engine(Protocol):
    def schedule(self, req: CommandRequest) -> CommandReceipt: ...

    def time_now(self) -> Fraction: ...


class LegacyNextTickAdapter:
    """Single-slot, next-tick RIS commands of v2 on top of an engine (see the module docs).

    ``engine``: anything with ``schedule`` and ``time_now`` (``EngineProtocol``).
    ``tau_s``: the v2 tick (default 2 ms). ``lost`` counts overwritten (superseded)
    commands.
    """

    def __init__(
        self,
        engine: _Engine,
        tau_s: Fraction | int | str = Fraction(1, 500),
        *,
        allow_nonphysical: bool = False,
    ) -> None:
        if not allow_nonphysical:
            raise NonPhysicalError(
                "LegacyNextTickAdapter reproduces v2's single-slot RIS commands; pass "
                "allow_nonphysical=True (teeth and E-B baseline only, section 5.14)"
            )
        self.engine = engine
        self.tau_s = parse_exact(tau_s, name="tau_s")
        if self.tau_s <= 0:
            raise ConfigurationError("tau_s must be > 0")
        self._slot: dict[str, tuple[Fraction, str]] = {}  # panel -> (tick, cmd_id)
        self.lost = 0

    def next_tick(self, t_now: Fraction) -> Fraction:
        """First tick boundary strictly after ``t_now``."""
        return (math.floor(t_now / self.tau_s) + 1) * self.tau_s

    def schedule(self, req: CommandRequest) -> CommandReceipt:
        """Submit a command the v2 way: overwrite the panel's slot, apply at the next tick."""
        t_k = self.next_tick(self.engine.time_now())
        prev = self._slot.get(req.target)
        if prev is not None and prev[0] == t_k:
            self.lost += 1
        self._slot[req.target] = (t_k, req.cmd_id)
        return self.engine.schedule(
            dataclasses.replace(req, at=CommandTime.at(t_k), late_policy="apply_on_arrival")
        )
