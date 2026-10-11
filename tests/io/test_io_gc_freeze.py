"""Served runs freeze the heap: no long garbage-collector pause (DESIGN.md section 4.10).

A served engine's process holds ~1.6e5 long-lived objects (interpreter, numba, pydantic,
zmq, the scenario); a full collection of that heap stalled every thread for ~55 ms and made
~150 pages late each in the 81f2735 B4 runs. ``Emulator.run_paced`` and ``Emulator.start``
run ``gc.collect(); gc.freeze()`` first, so later collections walk only what the run
allocated; the freeze ends with the loop (or ``Emulator.close``). The test serves a paced run
in a process with a large live heap, sends commands over the control socket while allocating
short-lived containers (gen 0/1/2 collections) and forcing one full collection, and records
every collection with ``gc.callbacks``: none in the measured span may exceed 5 ms (a full
collection of the unfrozen heap took 32 ms on the development laptop). Two more tests pin the
release of the freeze when a served paced run is stopped before its resume and when the
engine is closed while another thread is collecting for it (no freeze may outlive the engine).
"""

from __future__ import annotations

import gc
import threading
import time
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sremu.contracts import CommandTime
from sremu.engine import emulator
from sremu.engine.streams import ToneSource
from sremu.transport.client import connect_engine
from tests.io._wp8_support import served, wait_until
from tests.runtime._engine_support import config_request, lab_like_fake, make_engine

pytestmark = pytest.mark.io

F = Fraction(10**6)
N_COMMANDS = 30
MAX_PAUSE_NS = 5_000_000


def test_served_paced_run_has_no_long_collection() -> None:
    holders0 = emulator._gc_holders  # engines of other tests in this process may hold one
    ballast = [[i] for i in range(400_000)]  # long-lived containers, like a bench process
    eng = make_engine(
        lab_like_fake(),
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        mode="paced",
        coupling="async",
        duration_s=Fraction(60),  # ended by the server's stop after the commands
    )
    n_ctl = eng.commands.specs["ris0"].n_controls
    rng = np.random.default_rng(3)
    events: list[tuple[int, int, int]] = []  # (generation, start ns, duration ns)
    t_start = [0]

    def on_gc(phase: str, info: dict[str, Any]) -> None:
        if phase == "start":
            t_start[0] = time.perf_counter_ns()
        else:
            events.append(
                (int(info["generation"]), t_start[0], time.perf_counter_ns() - t_start[0])
            )

    gc.callbacks.append(on_gc)
    try:
        with served(eng, start_paused=True) as srv:
            remote = connect_engine(str(srv.endpoints["control"]), name="gc", timeout_s=30.0)
            try:
                remote.session.call("sim.resume")
                wait_until(lambda: eng._pacer is not None, 10.0, "the paced run")
                frozen = gc.get_freeze_count()
                t0 = time.perf_counter_ns()
                ok = 0
                for k in range(N_COMMANDS):
                    req = config_request(
                        f"gc:{k}",
                        "ris0",
                        rng.integers(0, 2, n_ctl),
                        CommandTime("asap"),
                        src="gc",
                    )
                    ok += bool(remote.schedule(req).ok)
                    junk = [[j] for j in range(5_000)]  # short-lived: gen 0/1/2 collections
                    del junk
                    if k == N_COMMANDS // 2:
                        gc.collect()  # the full collection a long run eventually triggers
                    time.sleep(0.005)
                t1 = time.perf_counter_ns()
                in_run = not srv.is_done()  # the measured span lies inside the run
            finally:
                remote.close()
    finally:
        gc.callbacks.remove(on_gc)
        eng.close()
    assert ok == N_COMMANDS and in_run
    assert frozen >= len(ballast)  # the run froze the heap, ballast included
    assert not eng._gc_frozen and emulator._gc_holders == holders0  # released by the run
    if holders0 == 0:  # and unfrozen (CPython keeps a few hundred startup objects frozen)
        assert gc.get_freeze_count() < len(ballast)
    span = [e for e in events if t0 <= e[1] <= t1]
    assert any(g == 2 for g, _, _ in span), span  # not vacuous: full collections happened
    worst = max(span, key=lambda e: e[2])
    assert worst[2] < MAX_PAUSE_NS, f"gen{worst[0]} collection took {worst[2] / 1e6:.1f} ms"


def test_stop_before_resume_releases_the_freeze() -> None:
    """The served paced driver freezes before ``sim.resume``; a server stopped before the
    resume never calls ``run_paced``, so the driver itself must release the freeze."""
    holders0 = emulator._gc_holders
    eng = make_engine(
        lab_like_fake(),
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        mode="paced",
        coupling="async",
        duration_s=Fraction(1),
    )
    try:
        with served(eng, start_paused=True):
            wait_until(lambda: eng._gc_frozen, 10.0, "the freeze before the resume")
        # stop() joined the paced driver thread, which never ran the paced loop
        assert not eng._gc_frozen and emulator._gc_holders == holders0
    finally:
        eng.close()


def test_close_during_the_collection_releases_the_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    """``close()`` from another thread while ``freeze_heap`` collects: the close waits for
    the freeze and releases it, and a closed engine never freezes again (a freeze that
    outlived its engine would keep the heap frozen for the rest of the process)."""
    entered, go = threading.Event(), threading.Event()
    calls = {"freeze": 0, "unfreeze": 0}

    def collect() -> int:
        entered.set()
        go.wait(10.0)
        return 0

    def count(name: str) -> None:
        calls[name] += 1

    shim = SimpleNamespace(
        collect=collect, freeze=lambda: count("freeze"), unfreeze=lambda: count("unfreeze")
    )
    monkeypatch.setattr(emulator, "gc", shim)
    monkeypatch.setattr(emulator, "_gc_holders", 0)
    eng = make_engine(lab_like_fake(), mode="paced", coupling="async", duration_s=Fraction(1))
    freezer = threading.Thread(target=eng.freeze_heap, daemon=True)
    freezer.start()
    assert entered.wait(10.0)
    closer = threading.Thread(target=eng.close, daemon=True)
    closer.start()
    time.sleep(0.05)  # let close() reach the freeze in progress (either order must work)
    go.set()
    freezer.join(10.0)
    closer.join(10.0)
    assert not freezer.is_alive() and not closer.is_alive()
    assert not eng._gc_frozen and emulator._gc_holders == 0
    assert calls == {"freeze": 1, "unfreeze": 1}
    eng.freeze_heap()  # closed: no new freeze
    assert calls == {"freeze": 1, "unfreeze": 1} and emulator._gc_holders == 0
