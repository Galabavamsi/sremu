"""Command latency stamps share one clock (DESIGN.md sections 4.6, 12.1; I7).

``visible latency = t_wall_visible_ns - t_wall_recv_ns``: the engine stamps the visible
time with its wall clock (``Emulator.clock``), so the control server must stamp the receive
time with the same clock. ``time.monotonic_ns`` is a different clock on Windows
(GetTickCount64: 15.6 ms ticks, tens of ms away from ``perf_counter``), which made served
visible latencies there meaningless. A fake engine clock shows which clock the server read.
"""

from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import StreamCmd, StreamSpec
from sremu.engine.pacer import FakeClock
from sremu.engine.streams import ToneSource
from sremu.testing.fake_engine import FakeEngine
from sremu.timebase import TimeSpec
from sremu.transport.client import RemoteEngine
from sremu.transport.control_server import ControlServer
from sremu.transport.messages import b64_of
from tests.io._wp8_support import served
from tests.runtime._engine_support import lab_like_fake, make_engine

pytestmark = pytest.mark.io

F = Fraction(10**6)
T0 = 10**15  # far from any real perf_counter or monotonic reading


def test_receive_and_visible_stamps_come_from_the_engine_clock() -> None:
    clock = FakeClock(T0)  # served mode never sleeps on it: it stays at T0
    eng = make_engine(lab_like_fake(), traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)}, clock=clock)
    n = eng.commands.specs["ris0"].n_controls
    with served(eng, start_paused=True) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
            cli.stream_cmd(h, StreamCmd("num_done", 4000, False, TimeSpec(0, 0)))
            r = cli.session.request(
                {
                    "id": "c1",
                    "op": "ris.set_config",
                    "panel": "ris0",
                    "at": {"t": "1/1000"},
                    "config": {"states_b64": b64_of(np.ones(n), "<i2"), "dtype": "i2"},
                }
            )
            assert r["ok"] and r["t_wall_recv_ns"] == T0
            cli.resume()
            got = 0
            while got < 4000:
                y, _md = cli.read_rx(h, 4096, 10.0)
                got += y.size
        finally:
            cli.close()
    (rec,) = eng.records
    assert rec.status == "applied"
    assert rec.t_wall_recv_ns == T0 and rec.t_wall_visible_ns == T0  # one clock


def test_engine_without_a_clock_gets_perf_counter() -> None:
    srv = ControlServer(FakeEngine())  # not started: no sockets
    assert srv._now_ns is time.perf_counter_ns
