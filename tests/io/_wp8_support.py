"""Shared helpers for the WP8 transport tests (not collected: no ``test_`` prefix).

Servers bind ephemeral loopback ports (``tcp://127.0.0.1:0``) so tests can run in
parallel. Scenes are SIM geometry on ``FakePropagation`` (tests.runtime._engine_support);
nothing here is a physics result.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from sremu.transport.control_server import ControlServer, ServerConfig

REPO = Path(__file__).resolve().parents[2]

EPHEMERAL: dict[str, Any] = {
    "control_endpoint": "tcp://127.0.0.1:0",
    "pub_endpoint": "tcp://127.0.0.1:0",
    "tx_endpoint": "tcp://127.0.0.1:0",
}


def serve(engine: Any, **kw: Any) -> ControlServer:
    """A started server on ephemeral loopback ports."""
    return ControlServer(engine, ServerConfig(**{**EPHEMERAL, **kw})).start()


@contextmanager
def served(engine: Any, **kw: Any) -> Iterator[ControlServer]:
    """``serve`` as a context manager (always stopped)."""
    srv = serve(engine, **kw)
    try:
        yield srv
    finally:
        srv.stop()


def client_cmd(*args: str) -> list[str]:
    """Command line of the WP8 test client in a separate OS process."""
    return [sys.executable, "-m", "tests.io._wp8_client", *args]


def run_clients(cmds: Sequence[Sequence[str]], timeout_s: float = 300.0) -> list[str]:
    """Run client processes concurrently; assert exit code 0; return their stdout."""
    procs = [
        subprocess.Popen(
            list(c), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for c in cmds
    ]
    outs = []
    t0 = time.monotonic()
    for p in procs:
        try:
            out, err = p.communicate(timeout=max(1.0, timeout_s - (time.monotonic() - t0)))
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            raise
        assert p.returncode == 0, f"client failed ({p.args}):\n{out}\n{err}"
        outs.append(out)
    return outs


def wait_until(pred: Any, timeout_s: float = 10.0, what: str = "condition") -> None:
    """Poll ``pred()`` until true (wall clock; tests only)."""
    t0 = time.monotonic()
    while not pred():
        if time.monotonic() - t0 > timeout_s:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.005)


def bits(x: np.ndarray) -> np.ndarray:
    """Bit view for bitwise comparisons (complex64 -> uint64, complex128 -> 2 x uint64)."""
    a = np.ascontiguousarray(x)
    return a.view(np.uint64)


def vcfo_engine(f_hz: float, **kw: Any) -> Any:
    """The V-CFO-1 engine (tests/runtime/test_vcfo_engine.py) for an RX LO offset of
    ``-f_hz``, served-mode friendly (unopened TX streams are waited for)."""
    from tests.runtime._engine_support import make_engine
    from tests.runtime.test_vcfo_engine import _model

    kw.setdefault("backend", "numba_cpu")
    return make_engine(_model(), bw=588.0 * 2, cfo={"rx.rx0": -f_hz}, unopened_tx="wait", **kw)


def frac(x: str) -> Fraction:
    return Fraction(x)
