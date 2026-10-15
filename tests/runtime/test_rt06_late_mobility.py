"""RT-06 under mobility: the late-command time is raised until no switch lands in the past.

``earliest_time`` starts from ``max_{r,j}(frontier_r/f_r - tau_{r,j})`` with tau at the
requested time and re-maps with tau at the new time. For a receiver approaching the panel
tau shrinks, so the one-shot start maps one sample below the frontier; the engine must raise
the time until every switch sample reaches the frontier. Teeth: the one-shot value is
asserted to be late, so the test fails if the raise loop is removed.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from sremu.engine.commands import CommandManager, RxPort, panel_specs_from_model
from sremu.timebase import from_ticks
from sremu.units import C0
from tests.runtime._engine_support import lab_like_fake

pytestmark = pytest.mark.runtime


RX = "rx1.rx0"
F_RX = Fraction(122_880_000)
FRONTIER = 1_228_800  # 10 ms at 122.88 MS/s


def _mgr(speed_m_s: float) -> CommandManager:
    model = lab_like_fake(far_m=2000.0)
    specs = panel_specs_from_model(model)

    def delay(panel: str, rx: str, t: Fraction, tile: int | None) -> float:
        return (5.0 - speed_m_s * float(t)) / C0  # receiver approaching from 5 m

    return CommandManager(specs, [RxPort(RX, F_RX)], delay, {"ris0": 1}, static_delays=False)


@pytest.mark.parametrize("speed", [30.0, 300.0, 3e5])
def test_late_command_never_lands_before_frontier(speed: float) -> None:
    mgr = _mgr(speed)
    t = mgr.earliest_time("ris0", {RX: FRONTIER}, Fraction(0))
    ks, _ = mgr.switch_samples("ris0", t)
    assert int(ks[RX].min()) == FRONTIER  # on the frontier, not before and not later


def test_one_shot_start_would_be_late_for_fast_approach() -> None:
    mgr = _mgr(300.0)
    one_shot = from_ticks(FRONTIER, F_RX) - Fraction(mgr.tau("ris0", RX, Fraction(0), None))
    ks_one_shot, _ = mgr.switch_samples("ris0", one_shot)
    assert int(ks_one_shot[RX].min()) == FRONTIER - 1  # tau re-evaluated at t': in the past
    t = mgr.earliest_time("ris0", {RX: FRONTIER}, Fraction(0))
    assert t > one_shot
    ks, _ = mgr.switch_samples("ris0", t)
    assert int(ks[RX].min()) >= FRONTIER
