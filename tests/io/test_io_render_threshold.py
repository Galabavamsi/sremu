"""Render driver threshold per engine role (DESIGN.md section 4.10).

A served paced ``numba_cpu`` engine (driven by the control server, which marks it with
``Emulator.set_served`` before any server thread starts) renders with
``render_numba.PACED_PARALLEL_MIN_WORK`` (2**18): few short OpenMP regions beside the
server's control and transport threads. Every other engine keeps the lockstep
``render_numba.PARALLEL_MIN_WORK`` (2**16): lockstep engines, served or not, and headless
paced engines (B3, ``sremu run``), where 2**18 raised the page latency on AWS. The tests pin
the threshold each role reports and the one its render calls actually pass to
``render_numba.render_prepared`` (``min_work``; None selects ``PARALLEL_MIN_WORK``). The
threshold picks a driver only (both are bitwise identical), so no sample is compared here.
"""

from __future__ import annotations

import threading
from fractions import Fraction
from typing import Any

import pytest

from sremu.channel import render_numba
from sremu.engine.pacer import FakeClock
from sremu.engine.streams import ToneSource
from tests.io._wp8_support import served, wait_until
from tests.runtime._engine_support import lab_like_fake, make_engine

pytestmark = pytest.mark.io

F = Fraction(10**6)
LOCKSTEP = 1 << 16
PACED_SERVED = 1 << 18


def _engine(mode: str, **kw: Any) -> Any:
    return make_engine(
        lab_like_fake(),
        backend="numba_cpu",
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        mode=mode,
        coupling="async" if mode == "paced" else "barrier",
        always_active=True,
        **kw,
    )


@pytest.fixture
def thresholds(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[int]]:
    """Threshold of every ``render_numba`` render call, by the name of the calling thread."""
    seen: dict[str, list[int]] = {}
    lock = threading.Lock()
    orig = render_numba.render_prepared

    def spy(*args: Any, **kw: Any) -> None:
        mw = kw.get("min_work")
        with lock:
            seen.setdefault(threading.current_thread().name, []).append(
                render_numba.PARALLEL_MIN_WORK if mw is None else int(mw)
            )
        orig(*args, **kw)

    monkeypatch.setattr(render_numba, "render_prepared", spy)
    return seen


def test_threshold_constants() -> None:
    assert render_numba.PARALLEL_MIN_WORK == LOCKSTEP
    assert render_numba.PACED_PARALLEL_MIN_WORK == PACED_SERVED


def test_lockstep_engine_uses_the_lockstep_threshold(thresholds: dict[str, list[int]]) -> None:
    eng = _engine("lockstep", duration_s=Fraction(4 * 512) / F)
    try:
        assert not eng.served
        assert eng.render_parallel_min_work == LOCKSTEP
        eng.run()
    finally:
        eng.close()
    mine = thresholds.get(threading.current_thread().name, [])
    assert mine and set(mine) == {LOCKSTEP}
    info = eng.manifest_info()
    assert info["served"] is False and info["render_parallel_min_work"] == LOCKSTEP


def test_served_lockstep_engine_keeps_the_lockstep_threshold() -> None:
    eng = _engine("lockstep", duration_s=Fraction(1))
    try:
        with served(eng, start_paused=True):
            assert eng.served  # marked by the server
            assert eng.render_parallel_min_work == LOCKSTEP
    finally:
        eng.close()


def test_headless_paced_engine_uses_the_lockstep_threshold(
    thresholds: dict[str, list[int]],
) -> None:
    """B3 and ``sremu run``: ``run_paced`` without a server, in the caller's thread."""
    eng = _engine("paced", clock=FakeClock(), duration_s=Fraction(8 * 512) / F)
    try:
        assert not eng.served
        assert eng.render_parallel_min_work == LOCKSTEP
        s = eng.run_paced()
    finally:
        eng.close()
    assert s.paced is not None and s.paced["pages"] > 0
    mine = thresholds.get(threading.current_thread().name, [])
    assert mine and set(mine) == {LOCKSTEP}
    assert eng.manifest_info()["render_parallel_min_work"] == LOCKSTEP


def test_served_paced_engine_uses_the_paced_threshold(thresholds: dict[str, list[int]]) -> None:
    """The control server marks the engine before its paced driver thread renders."""
    eng = _engine("paced", clock=FakeClock(), duration_s=Fraction(8 * 512) / F)
    try:
        with served(eng, start_paused=True) as srv:
            assert eng.served
            assert eng.render_parallel_min_work == PACED_SERVED
            srv.resume()
            wait_until(srv.is_done, 30.0, "the end of the served paced run")
            assert srv.engine_error is None
    finally:
        eng.close()
    paced = thresholds.get("sremu-paced", [])  # the server's paced driver thread
    assert paced and set(paced) == {PACED_SERVED}
    info = eng.manifest_info()
    assert info["served"] is True and info["render_parallel_min_work"] == PACED_SERVED
    by = render_numba.driver_stats()["calls_by_min_work"]
    assert isinstance(by, dict) and str(PACED_SERVED) in by  # recorded for manifests


def test_set_served_switches_only_paced_numba_engines() -> None:
    paced = _engine("paced", duration_s=Fraction(1))
    lock = _engine("lockstep", duration_s=Fraction(1))
    ref = make_engine(lab_like_fake(), backend="numpy_ref", mode="paced", coupling="async")
    try:
        for e in (paced, lock, ref):
            e.set_served(True)
        assert paced.render_parallel_min_work == PACED_SERVED
        assert lock.render_parallel_min_work == LOCKSTEP
        assert ref.render_parallel_min_work is None  # numpy_ref has a single driver
        paced.set_served(False)
        assert not paced.served and paced.render_parallel_min_work == LOCKSTEP
    finally:
        for e in (paced, lock, ref):
            e.close()
