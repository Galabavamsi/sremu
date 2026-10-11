"""Remote round-trip latency: arrival wake-ups instead of a fixed poll (DESIGN.md 8.1, 8.3).

Root cause pinned here: ``sremu.Device`` polls the engine with timeout 0 under its lock and
waits ``DEFAULT_POLL_S`` (25 ms) between polls unless something in its own process calls
``Device.notify``. Over a remote session nothing did, so every ``recv`` that had to wait for
the server slept a whole poll interval: E-F measured a ~26 ms two-process round trip
(p50 25.9 ms, max 26.8 ms on a c7i.8xlarge) and 0.38 MS/s with 10,000-sample buffers. The
fix: :class:`~sremu.transport.zmq_native.ReadyWatcher` watches the client sockets'
``ZMQ_FD`` handles and calls the registered listeners (``Device.notify``) as soon as a
frame may have arrived; :meth:`RemoteEngine.add_wake_listener` registers them. A second,
smaller cause (tail only): the served engine loop could sleep its 5 ms idle wait through
an input that arrived between a fruitless ``step`` and the wait (``Emulator._kick``).

Checks. (a) The mechanism, whatever the host load: a Device whose poll fallback is
effectively off (``poll_s`` = 60 s) must finish every round trip in under 1 s, so every
wait was ended by a wake-up (a lost one is ended only by the ``recv`` timer, 5 s; Device
then retries once and returns the data, which is why the duration is checked), both with
the server threads in this process and with ``sremu serve`` in its own process; and
``Device.connect`` must register the wake-up (strict xfail until ``api/device.py`` does).
Together these exclude the root cause on any host. (b) The latency itself: two-process
round-trip p50 < 5 ms, asserted only when the test has the host to itself (not under
pytest-xdist). Measured with the fix: p50 1.1-1.7 ms on a quiet laptop (Windows 11, WSL2
Ubuntu); in the CI tier's configuration (``-n 8`` on four CPUs) up to 10.3 ms on WSL2 and
46-55 ms on Windows, where the bare transport (a blocking ``read_rx``, no Device) already
took 32-37 ms: under CPU oversubscription the OS scheduler, not the transport, sets the
latency, and no bound separates the 25 ms poll from it. The slowest woken round trip
there took 0.2 s, well below the 1 s mechanism bound. The first two round trips of the
two-process test get a 60 s timeout and no duration bound: with a cold numba cache the
served engine's first round trip took 4.3-4.6 s on a quiet laptop and 14.8 s on four
loaded CPUs (the second 2-5 ms). Without wake-ups the test still fails within about two
minutes (two 60 s polls, then the 1 s bound). Bitwise behaviour is covered by
IO-02/RT-03/RT-08.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zmq

from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TxMetadata
from sremu.contracts import StreamSpec
from sremu.errors import StreamError
from sremu.transport.client import RemoteEngine, connect_engine
from sremu.transport.zmq_native import ReadyWatcher, TransportCounters
from tests.io._wp8_support import REPO, served
from tests.runtime._engine_support import lab_like_fake, make_engine

pytestmark = pytest.mark.io

FC = 3.5e9
P50_BOUND_S = 5e-3
"""Two-process round-trip p50 bound, asserted when the test has the host to itself."""
NO_POLL_S = 60.0
"""``poll_s`` of the mechanism checks: no wait may rely on the poll fallback."""
RECV_TIMEOUT_S = 5.0
MAX_RTT_S = 1.0
"""Mechanism bound: a round trip whose wait missed its wake-up lasts >= RECV_TIMEOUT_S; the
slowest woken one took 0.2 s in the CI tier's configuration (module docstring)."""
COLD_TIMEOUT_S = 60.0
"""``recv`` timeout of the two-process test's first ``COLD_ROUND_TRIPS`` round trips."""
COLD_ROUND_TRIPS = 2
"""Round trips that may include the server's cold start (numba cache): no duration bound."""
BUF = 1000


def _round_trips(
    dev: Device,
    tx_stream: str,
    rx_stream: str,
    n_iter: int,
    n_warm: int,
    max_rtt_s: float | None = None,
    cold: int = 0,
) -> Any:
    """E-F loop: send ``BUF`` samples, then receive ``BUF`` samples; wall seconds of each of
    the ``n_iter`` round trips after ``n_warm`` unmeasured ones. Every ``recv`` must return
    samples within ``RECV_TIMEOUT_S`` (``COLD_TIMEOUT_S`` in the first ``cold`` round
    trips); with ``max_rtt_s`` every round trip after the first ``cold``, warm-up included,
    must be shorter than that."""
    x = np.exp(2j * np.pi * 0.01 * np.arange(BUF)).astype(np.complex64)
    y = np.zeros(BUF, dtype=np.complex64)
    tx = dev.get_tx_stream(StreamArgs(tx_stream, FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs(rx_stream, FC, "1e6"))
    rx.issue_stream_cmd(StreamCmd(mode="start_cont", stream_now=True))
    md = TxMetadata(start_of_burst=True)
    rtt = []
    for i in range(n_warm + n_iter):
        timeout = COLD_TIMEOUT_S if i < cold else RECV_TIMEOUT_S
        a = time.perf_counter()
        assert tx.send(x, md, timeout=2.0) == BUF
        got = 0
        while got < BUF:
            n = rx.recv(y[got:], RxMetadata(), timeout=timeout)
            assert n > 0, (
                f"recv timed out ({timeout} s) in round trip {i} after {got} samples: "
                "the server delivered nothing"
            )
            got += n
        dt = time.perf_counter() - a
        if max_rtt_s is not None and i >= cold:
            assert dt < max_rtt_s, (
                f"round trip {i} took {dt:.3f} s: a wait was not ended by a wake-up"
            )
        if i >= n_warm:
            rtt.append(dt)
        md = TxMetadata()
    return np.asarray(rtt)


# ------------------------------------------------------------------ ReadyWatcher unit tests


def _pair(ctx: zmq.Context[Any]) -> tuple[zmq.Socket[Any], zmq.Socket[Any]]:
    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.LINGER, 0)
    port = pull.bind_to_random_port("tcp://127.0.0.1")
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 0)
    push.connect(f"tcp://127.0.0.1:{port}")
    return pull, push


def _empty(sock: zmq.Socket[Any]) -> int:
    """Read everything queued (the reader 'finds the socket empty'); returns the count."""
    n = 0
    while sock.poll(0, zmq.POLLIN):
        sock.recv(zmq.NOBLOCK)
        n += 1
    return n


def test_ready_watcher_wakes_a_reader_that_found_its_socket_empty() -> None:
    """Each arrival after an empty read fires the listener; a socket nobody reads fires
    once and then stays quiet (no spinning) until its owner reads and re-arms it."""
    ctx: zmq.Context[Any] = zmq.Context()
    pull, push = _pair(ctx)
    push.send(b"sync")  # the connection is up before the watcher starts
    assert pull.poll(5000, zmq.POLLIN)
    pull.recv()
    w = ReadyWatcher(TransportCounters())
    fired = threading.Event()
    w.listen(fired.set)
    arm = w.add(int(pull.getsockopt(zmq.FD)))
    try:
        for k in range(20):
            _empty(pull)
            arm()
            fired.clear()
            if fired.wait(0.05):  # a connection event may fire once: spurious, re-arm
                _empty(pull)
                arm()
                fired.clear()
            push.send(b"frame %d" % k)
            assert fired.wait(5.0), f"no wake-up for arrival {k}"
            assert pull.poll(5000, zmq.POLLIN)  # the frame is there for the reader
        # nobody reads now: further arrivals cause no further wake-ups or select loops
        _empty(pull)
        arm()
        before = w.wakeups
        push.send(b"a")
        deadline = time.monotonic() + 5.0
        while w.wakeups == before and time.monotonic() < deadline:
            time.sleep(0.005)
        assert w.wakeups == before + 1
        it0 = w._iter
        for _ in range(50):
            push.send(b"b")
        time.sleep(0.3)
        assert w.wakeups == before + 1
        assert w._iter - it0 <= 2  # idle select timeouts only (WATCH_IDLE_S = 1 s)
        assert _empty(pull) == 51
    finally:
        w.stop()
        pull.close(0)
        push.close(0)
        ctx.term()


def test_ready_watcher_remove_stop_and_failing_listener() -> None:
    ctx: zmq.Context[Any] = zmq.Context()
    pull, push = _pair(ctx)
    counters = TransportCounters()
    w = ReadyWatcher(counters)
    calls = []

    def bad() -> None:
        raise RuntimeError("listener bug")

    w.listen(bad)
    w.listen(lambda: calls.append(1))
    fd = int(pull.getsockopt(zmq.FD))
    arm = w.add(fd)
    try:
        _empty(pull)
        arm()
        push.send(b"x")
        deadline = time.monotonic() + 5.0
        while not calls and time.monotonic() < deadline:
            time.sleep(0.005)
        assert calls, "the second listener must run although the first raised"
        assert counters["internal_errors"] >= 1
        assert w.thread.is_alive()
        w.remove(fd)  # returns only once no select includes the handle
        pull.close(0)  # safe right after remove()
        time.sleep(0.05)
        assert w.thread.is_alive()
        arm()  # arming a removed handle is a no-op
    finally:
        w.stop()
        w.stop()  # idempotent
        w.thread.join(5.0)
        pull.close(0)  # no-op when the body closed it; ctx.term() would hang on it
        push.close(0)
        ctx.term()
    assert not w.thread.is_alive()


# ------------------------------------------------------------------ engine served loop


def test_served_loop_does_not_sleep_through_an_input_that_arrived_during_its_step() -> None:
    """An input (TX, stream command, command, RX consumed) that lands while a fruitless
    ``step`` runs must make the served loop step again at once, not wait its 5 ms."""
    eng = make_engine(lab_like_fake())
    waits: list[float | None] = []

    class RecordingCondition(threading.Condition):
        def wait(self, timeout: float | None = None) -> bool:
            waits.append(timeout)
            return super().wait(0)

    eng._cond = RecordingCondition(eng._lock)
    eng._stop_evt = threading.Event()
    seen: list[int] = []

    def step(*, zero_fill: bool = False) -> bool:
        seen.append(len(waits))
        if len(seen) == 1:
            with eng._lock:
                eng._kick()  # e.g. submit_tx from the ingress thread, mid-step
        elif len(seen) == 3:
            assert eng._stop_evt is not None
            eng._stop_evt.set()
        return False

    eng.step = step  # type: ignore[method-assign]
    eng._serve_loop()
    # after step 1 (an input arrived meanwhile): no wait; after step 2 (none): one wait
    assert seen == [0, 0, 1]
    assert waits == [0.005]


# ------------------------------------------------------------------ remote round trips


def test_remote_device_waits_are_ended_by_wake_ups_not_by_the_poll() -> None:
    """A ``Device`` over a served engine (server threads in this process, native ZMQ over
    loopback) with ``Device.notify`` registered and its poll fallback at 60 s: each of the
    220 round trips takes under ``MAX_RTT_S``, so every ``recv`` that waited was woken by
    an arrival (a lost wake-up leaves the wait to the 5 s ``recv`` timer)."""
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    with served(eng) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        dev = Device(cli, owns_engine=True, poll_s=NO_POLL_S)
        cli.add_wake_listener(dev.notify)
        with dev:
            rtt = _round_trips(dev, "ap.tx0", "rx1.rx0", n_iter=200, n_warm=20, max_rtt_s=MAX_RTT_S)
            assert cli._watcher is not None and cli._watcher.wakeups > 0
    assert rtt.size == 200


def test_device_connect_registers_the_wake_listener() -> None:
    """``Device.connect`` wires ``Device.notify`` to the remote engine's wake-ups, without
    which every remote ``recv`` that waits sleeps a whole ``DEFAULT_POLL_S``."""
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    with served(eng) as srv:
        dev = Device.connect(str(srv.endpoints["control"]))
        with dev:
            w = getattr(dev.engine, "_watcher", None)
            assert w is not None, "no wake listener registered"
            assert dev.notify in w._listeners


def test_remote_engine_close_with_listener_and_closed_stream() -> None:
    """Closing streams and the engine with a watcher running never touches a closed
    socket (remove/stop order) and leaves no live watcher thread."""
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    with served(eng) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        cli.add_wake_listener(lambda: None)
        h = cli.open_stream(StreamSpec("rx1.rx0", "rx", FC, Fraction(10**6)))
        assert cli._rx[h.id].on_read is not None  # streams opened later are watched too
        cli.close_stream(h)
        w = cli._watcher
        assert w is not None and w.thread.is_alive()
        cli.close()
        w.thread.join(5.0)
        assert not w.thread.is_alive()
        assert cli.counters["internal_errors"] == 0
        with pytest.raises(StreamError, match="closed"):
            cli.add_wake_listener(lambda: None)


def _latency_scenario(tmp_path: Path) -> Path:
    """The quickstart scene without its tone generator (the client feeds ap.tx0), as E-F."""
    from sremu.scenario.load import read_document

    d = read_document(REPO / "scenarios" / "quickstart.yaml")
    d["name"] = "io_wake_latency"
    d["traffic"] = []
    d["run"]["duration_s"] = "1000"
    p = tmp_path / "latency.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


@pytest.mark.integration
def test_two_process_round_trips_are_woken_and_fast(tmp_path: Path) -> None:
    """The E-F topology: ``sremu serve`` in its own OS process and a Device over
    ``connect_engine`` (what ``Device.connect`` builds) with ``Device.notify`` registered and
    its poll fallback at 60 s. Every round trip after the first ``COLD_ROUND_TRIPS`` takes
    under ``MAX_RTT_S`` (no wait relied on the poll), and, when this test has the host to
    itself (not under pytest-xdist, whose other workers share the CPUs; module docstring),
    the round-trip p50 is below ``P50_BOUND_S``."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sremu.commands.serve",
            str(_latency_scenario(tmp_path)),
            "--serve",
            "tcp://127.0.0.1:0",
            "--pub",
            "none",
            "--tx",
            "tcp://127.0.0.1:0",
            "--no-paused",
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line.startswith("SREMU-SERVE "), line
        ep = json.loads(line.split(" ", 1)[1])["endpoints"]["control"]
        cli = connect_engine(ep)
        dev = Device(cli, owns_engine=True, poll_s=NO_POLL_S)
        cli.add_wake_listener(dev.notify)
        with dev:
            rtt = _round_trips(
                dev,
                "ap.tx0",
                "mu.rx0",
                n_iter=300,
                n_warm=30,
                max_rtt_s=MAX_RTT_S,
                cold=COLD_ROUND_TRIPS,
            )
            cli.shutdown()
        assert rtt.size == 300
        p50 = float(np.median(rtt))
        if not os.environ.get("PYTEST_XDIST_WORKER"):
            assert p50 < P50_BOUND_S, (
                f"round trip p50 {p50 * 1e3:.2f} ms (max {rtt.max() * 1e3:.1f})"
            )
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
