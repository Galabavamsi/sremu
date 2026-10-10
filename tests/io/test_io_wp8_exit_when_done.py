"""``exit_when_done`` ends a served run whose consumers left early (DESIGN.md 4.9, 8.5).

Root cause of the CI hang of ``examples/controller_remote`` (GitHub Actions run 38066711708,
reproduced on Linux under WSL2): with ``exit_when_done`` the served lockstep engine held
virtual time whenever nothing consumed it, so a remote controller that unregistered before
the run's last barrier (the example's ``run_remote`` gave up after a 2 s idle timeout while
the server was still JIT-compiling its numba kernels) or an RX client that closed its stream
early left the run unfinished and ``sremu serve --exit-when-done`` waiting forever. Once a
consumer has attached, the run must go on to its end and the server must stop by itself.

Scenes are SIM geometry on ``FakePropagation``; nothing here is a physics result.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import zmq

from sremu.contracts import StreamCmd, StreamSpec
from sremu.control.remote import run_remote
from sremu.engine.streams import ToneSource
from sremu.timebase import TimeSpec
from sremu.transport.client import RemoteEngine
from tests.io._wp8_support import served, tcp_buffer_bytes
from tests.runtime._engine_support import (
    ScriptedController,
    alternate_script,
    lab_like_fake,
    make_engine,
)

pytestmark = pytest.mark.io

F = Fraction(10**6)
WAIT_S = 120.0  # wall-clock backstop only; a correct server stops within about a second


def _engine(duration_s: Fraction):
    return make_engine(
        lab_like_fake(), traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)}, duration_s=duration_s
    )


def _ran_to_the_end(eng) -> bool:
    return all(g.end is not None and g.cursor >= g.end for g in eng.groups)


def test_exit_when_done_after_the_remote_controller_unregistered_early() -> None:
    eng = _engine(Fraction(1, 50))  # 20 barriers at 1 ms; the controller serves 3
    n = eng.commands.specs["ris0"].n_controls
    with served(eng, exit_when_done=True) as srv:
        ctrl = ScriptedController("early", "1/1000", "2/10000", alternate_script("ris0", n))
        stats = run_remote(
            ctrl, str(srv.endpoints["control"]), max_barriers=3, idle_timeout_s=WAIT_S
        )
        assert stats.barriers == 3
        assert "early" not in srv.controllers  # unregistered
        assert srv.wait(WAIT_S), "the served run did not end after the controller left"
        assert srv.engine_error is None
        assert _ran_to_the_end(eng)
    assert [r.src for r in eng.records] == ["early"] * 3  # its three actions, nothing else


def test_exit_when_done_after_the_rx_client_closed_its_stream_early() -> None:
    # The client leaves after its first frame. While its stream is open, lockstep
    # backpressure lets the engine run ahead by at most the RX queue + the ZMQ high-water
    # marks + the OS TCP buffers (test_io_wp8_backpressure); on Linux the last term alone
    # can exceed 1e6 samples, so a 0.5 s run (5e5 samples) could end before the client
    # left. The run therefore ends beyond that bound (asserted below), so only the served
    # engine itself can finish it.
    duration = Fraction(4)
    eng = _engine(duration)
    with served(eng, exit_when_done=True, start_paused=True) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
            rcvhwm = cli._rx[h.id].sock.getsockopt(zmq.RCVHWM)
            ahead = (
                eng.rx["rx1.rx0"].queue_samples
                + (srv.config.rx_sndhwm + rcvhwm + 2) * h.page_size
                + tcp_buffer_bytes() // 8
            )
            assert ahead < duration * F
            cli.stream_cmd(h, StreamCmd("start_cont", 0, False, TimeSpec(0, 0)))
            cli.resume()
            y, md = cli.read_rx(h, 1000, 30.0)
            assert y.size > 0 and md.first_sample == 0
            cli.close_stream(h)
            assert not srv.is_done()  # the client left long before the end of the run
            assert srv.wait(WAIT_S), "the served run did not end after the RX client left"
            assert srv.engine_error is None
            assert _ran_to_the_end(eng)
        finally:
            cli.close()
