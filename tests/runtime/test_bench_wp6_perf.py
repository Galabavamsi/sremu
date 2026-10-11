"""WP6 engine micro-benchmarks (``integration`` + ``slow``: nightly tier, not CI).

Wall-clock measurements on whatever machine runs them, so the bounds are generous and only
catch a return of the brute-force costs; each test prints its numbers. The paper's numbers
come from ``bench/`` runs, never from here.

* B4-style control loop: engine time per RIS command at 125/250/500 commands and 1 kHz.
* Insert cost into a book holding 250 commands vs an almost empty book.
* Status-update cost with 2000 scheduled commands vs 10.
* Lockstep RTF of the mobile (F3) B2 base scene with 1 and 8 links.

SIM geometry only (the B2 benchmark scene and FakePropagation).
"""

from __future__ import annotations

import time
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from sremu.contracts import CommandTime
from sremu.engine.commands import CommandManager, RxPort, panel_specs_from_model
from tests.runtime._engine_support import config_request, lab_like_fake

pytestmark = [pytest.mark.integration, pytest.mark.slow]

F = Fraction(10**6)
RXS = ("rx1.rx0", "rx2.rx0")


def _bench_engine(links: int, feat: str, dur: Fraction) -> Any:
    from bench._engine_point import _resolve, build_engine
    from bench._scenes import bench_scenario

    res = _resolve(
        bench_scenario(
            links=links, n_cells=240, n_ris=1, rate_sps="1e6", features=feat, duration_s=str(dur)
        )
    )
    return build_engine(res, feat)


def _b4_wall(n_cmd: int, controller: bool) -> float:
    from sremu.contracts import LatencyModel
    from sremu.control.builtin import Urap

    dur = Fraction(n_cmd, 1000)
    eng = _bench_engine(2, "F1", dur)
    if controller:
        lm = Fraction(1, 10_000)
        eng.register_controller(
            Urap(
                0.5,
                name="b4",
                period_s=Fraction(1, 1000),
                min_latency_s=lm,
                latency=LatencyModel(lm, f"uniform:0,{float(lm)!r}"),
            )
        )
    t0 = time.perf_counter()
    eng.run(until_s=dur)
    wall = time.perf_counter() - t0
    eng.close()
    return wall


@pytest.mark.parametrize("n_cmd", [125, 250, 500])
def test_bench_b4_engine_time_per_command(n_cmd: int, record_property: Any) -> None:
    """B4-style lockstep control loop (2 links, N = 240, 1 MS/s, F1, Urap in-process at
    1 kHz, L_min = 0.1 ms): engine wall time per command = (with controller - without) /
    commands, median of 3. The brute-force timeline cost 5-8 ms per command on the
    development laptop (14-23 times the wall time of rendering one 1 ms control period
    without a controller); the incremental one costs 1.3-2.1 ms (4-7 times). The bound only
    catches a return to the brute-force cost: it passes when the absolute cost is below
    3.5 ms or, on a slower or loaded machine (the nightly tier runs under xdist), when the
    cost is below 10 times this machine's own render time per period."""
    _b4_wall(50, True)  # numba kernels and caches warm
    ctl = sorted(_b4_wall(n_cmd, True) for _ in range(3))[1]
    ref = sorted(_b4_wall(n_cmd, False) for _ in range(3))[1]
    ms = (ctl - ref) / n_cmd * 1e3
    ms_period = ref / n_cmd * 1e3  # render wall time of one 1 ms period, no controller
    record_property("ms_per_command", ms)
    print(
        f"B4 {n_cmd} commands: {ms:.3f} ms per command (with {ctl:.3f} s, without {ref:.3f} s; "
        f"{ms / ms_period:.1f} x the render time per period)"
    )
    assert ms < 3.5 or ms < 10.0 * ms_period


def _manager() -> CommandManager:
    model = lab_like_fake(tiles=(3, 1))
    ports = [RxPort(r, F) for r in RXS]
    return CommandManager(
        panel_specs_from_model(model), ports, model.ris_delay_s, {"ris0": 3}, static_delays=True
    )


def test_bench_insert_cost_does_not_grow_with_the_book() -> None:
    """Scheduling into a book holding 250 commands costs about as much as into one holding
    5 (the brute-force book recomposed every held command at every insert)."""

    def per_insert(held: int) -> float:
        mgr = _manager()
        n = mgr.specs["ris0"].n_controls
        fr = dict.fromkeys(RXS, 0)
        rng = np.random.Generator(np.random.Philox(key=3))
        for i in range(held):
            st = rng.integers(0, 2, n).astype(np.int16)
            at = CommandTime.at(Fraction(i + 1, 10**4))
            mgr.schedule(config_request(f"h{i}", "ris0", st, at), fr)
        reqs = [
            config_request(
                f"x{i}",
                "ris0",
                rng.integers(0, 2, n).astype(np.int16),
                CommandTime.at(Fraction(held + i + 1, 10**4)),
            )
            for i in range(40)
        ]
        t0 = time.perf_counter()
        for r in reqs:
            mgr.schedule(r, fr)
        return (time.perf_counter() - t0) / len(reqs)

    small = min(per_insert(5) for _ in range(3))
    big = min(per_insert(250) for _ in range(3))
    print(f"insert: {small * 1e6:.1f} us (5 held), {big * 1e6:.1f} us (250 held)")
    assert big < 3.0 * small


def test_bench_status_update_cost_does_not_grow_with_pending() -> None:
    """A status update with nothing due costs about the same with 2000 scheduled commands
    as with 10 (the brute-force update scanned every scheduled command)."""

    def per_update(pending: int) -> float:
        mgr = _manager()
        n = mgr.specs["ris0"].n_controls
        fr = dict.fromkeys(RXS, 0)
        for i in range(pending):
            st = np.full(n, i % 2, np.int16)
            mgr.schedule(
                config_request(f"p{i}", "ris0", st, CommandTime.at(Fraction(i + 10, 1000))), fr
            )
        t0 = time.perf_counter()
        for k in range(500):
            mgr.update_statuses(dict.fromkeys(RXS, 1000 + k))
        return (time.perf_counter() - t0) / 500

    small = min(per_update(10) for _ in range(3))
    big = min(per_update(2000) for _ in range(3))
    print(f"status update: {small * 1e6:.2f} us (10 pending), {big * 1e6:.2f} us (2000 pending)")
    assert big < 3.0 * small + 5e-6


@pytest.mark.parametrize("links", [1, 8])
def test_bench_mobile_lockstep_rtf(links: int, record_property: Any) -> None:
    """Lockstep RTF of the B2 base scene with mobility (F3: RX at 1.5 m/s, automatic T_u,
    CFO, phase noise, thermal noise; N = 240, 1 MS/s), 0.5 s after a 0.05 s warm-up, best
    of 3. Reported only, with a floor far below the measured values."""
    best = 0.0
    for _ in range(3):
        eng = _bench_engine(links, "F3", Fraction(11, 20))
        eng.run(until_s=Fraction(1, 20))
        t0 = time.perf_counter()
        eng.run(until_s=Fraction(11, 20))
        best = max(best, 0.5 / (time.perf_counter() - t0))
        eng.close()
    record_property("rtf", best)
    print(f"mobile F3, {links} link(s): lockstep RTF {best:.3f}")
    assert best > (0.3 if links == 1 else 0.06)
