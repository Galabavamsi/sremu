"""Legacy ``block_constant`` Doppler: the v2 staircase geometry (DESIGN.md 4.4, 5.14; PH-17).

v2 froze the whole geometry for each engine block of ``T_b`` seconds: inside a block the
delay, and so the carrier phase, of every path stays at its value at the block start, and it
jumps at the next block. For the T9 geometry of ``pm_checks.py`` (3.5 GHz, 30 m/s approach,
2 ms blocks, 10 kS/s for 0.2 s) the mean instantaneous frequency is about -148.3 Hz instead
of the geometric +350.242 Hz (D7). sremu's normative renderer interpolates a chord delay
with an exact carrier phase instead; this mode exists only for the PH-17 teeth test.

:func:`block_constant_emulator` builds an :class:`~sremu.engine.emulator.Emulator` whose
snapshot grid is the block grid and whose intervals are rendered degenerate (``t_a = t_b``
at the block start, w = 0). Raises ``NonPhysicalError`` unless ``allow_nonphysical=True``.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from sremu.contracts import PropagationModel
from sremu.engine.emulator import Emulator, EngineConfig
from sremu.errors import ConfigurationError, NonPhysicalError
from sremu.timebase import parse_exact

__all__ = ["block_constant_emulator"]


def block_constant_emulator(
    model: PropagationModel,
    block_s: Fraction | int | str = Fraction(1, 500),
    *,
    allow_nonphysical: bool = False,
    config: EngineConfig | None = None,
    **kwargs: Any,
) -> Emulator:
    """An emulator with v2 staircase geometry over blocks of ``block_s`` (default 2 ms).

    ``kwargs`` go to :class:`~sremu.engine.emulator.Emulator` (streams, traffic, sinks, ...).
    ``verify_interp`` is off (the staircase is not an interpolation).
    """
    if not allow_nonphysical:
        raise NonPhysicalError(
            "block_constant Doppler reproduces v2's staircase geometry; pass "
            "allow_nonphysical=True (PH-17 teeth only, section 5.14)"
        )
    b = parse_exact(block_s, name="block_s")
    if b <= 0:
        raise ConfigurationError("block_s must be > 0")
    cfg = config or EngineConfig()
    import dataclasses

    cfg = dataclasses.replace(cfg, verify_interp=False, render_tiles="model")
    return Emulator(model, config=cfg, legacy_block_s=b, allow_nonphysical=True, **kwargs)
