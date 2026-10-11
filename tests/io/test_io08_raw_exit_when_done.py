"""IO-08 regression: raw REQ/REP peers with ``exit_when_done`` (DESIGN.md 8.4, 8.5).

``sremu serve --exit-when-done`` stops by itself once the run has ended. In lockstep the
engine may reach the end of the run while up to ``rx_queue_pages`` pages are still queued
for a raw peer that reads slowly; the peer must nevertheless receive every RX sample of
the run, in order and exactly once, and the server must then stop by itself. The server
here is driven exactly as the CLI drives it (``serve_forever``: stop as soon as the
shutdown is set), so the test covers the hand-over from the end of the run to the stop.

The peer has to read to the end of the run: the engine cannot finish while a raw consumer
stops reading (RX backpressure), so the server would then never exit; this is documented
in ``sremu.transport.rawreqrep`` and is not exercised here.

Also pinned: the default TX request size of ``gnuradio_count`` peers (8 pages, at least
4096 and at most 65,536 samples), which halves the per-page REQ/REP cost of a
free-running GNU Radio transmitter (sdr_integration_feasibility.md, section 3.3).

Scenes are SIM geometry on ``FakePropagation``; nothing here is a physics result.
"""

from __future__ import annotations

import struct
import threading
import time
from fractions import Fraction

import numpy as np
import pytest
import zmq

from sremu.transport.rawreqrep import (
    GNURADIO_REQUEST_MAX_ITEMS,
    GNURADIO_REQUEST_MIN_ITEMS,
    GNURADIO_REQUEST_PAGES,
    RawPeerConfig,
    default_request_items,
)
from tests.io._wp8_support import bits, serve
from tests.runtime._engine_support import client_run, lab_like_fake, make_engine, waveform

pytestmark = pytest.mark.io

DURATION = Fraction(3, 25)  # 0.12 s = 120,000 samples at 1 MS/s, > 3 RX queues of 64 pages
N_RUN = 120_000
WAIT_S = 120.0  # wall-clock backstop only


def test_io08_default_request_items() -> None:
    assert (GNURADIO_REQUEST_PAGES, GNURADIO_REQUEST_MIN_ITEMS) == (8, 4096)
    assert GNURADIO_REQUEST_MAX_ITEMS == 65_536
    assert default_request_items("gnuradio_count", 512) == 4096
    assert default_request_items("gnuradio_count", 256) == 4096  # at least 4096
    assert default_request_items("gnuradio_count", 1024) == 8192
    assert default_request_items("gnuradio_count", 8192) == 65_536
    assert default_request_items("gnuradio_count", 16_384) == 65_536  # capped
    # srsRAN's TX REP ignores the request's content: the page size is kept
    assert default_request_items("srsran_dummy", 512) == 512


class _SlowPeer:
    """A GNU Radio- or srsRAN-style peer whose receiver is slow: its TX REP answers at
    once (a free-running transmitter), its RX REQ waits ``pause_s`` between requests so
    that the engine runs ahead by its whole RX queue. The RX side reads until it has
    ``n_total`` samples or no reply came for ``give_up_s``."""

    def __init__(self, fmt: str, x: np.ndarray, pause_s: float) -> None:
        self.fmt = fmt
        self.x = x
        self.pause_s = pause_s
        self.ctx = zmq.Context()
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.setsockopt(zmq.LINGER, 0)
        self.tx_port = f"tcp://127.0.0.1:{self.rep.bind_to_random_port('tcp://127.0.0.1')}"
        self.sent = 0
        self.tx_requests: list[bytes] = []
        self.received: list[np.ndarray] = []
        self.stop = threading.Event()

    def tx_main(self) -> None:
        while not self.stop.is_set():
            if not self.rep.poll(50):
                continue
            req = self.rep.recv()
            self.tx_requests.append(req)
            n = 1500 if self.fmt == "srsran_dummy" else int(struct.unpack("<I", req)[0])
            a = self.x[self.sent : self.sent + n]
            a = np.concatenate([a, np.zeros(n - a.size, np.complex64)])
            self.sent += n
            self.rep.send(a.astype("<c8").tobytes())

    def rx_main(self, rx_port: str, n_total: int, give_up_s: float) -> None:
        req = self.ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.connect(rx_port)
        got = 0
        try:
            while got < n_total and not self.stop.is_set():
                time.sleep(self.pause_s)
                req.send(b"\xff" if self.fmt == "srsran_dummy" else struct.pack("<I", 1000))
                if not req.poll(int(give_up_s * 1000)):
                    return  # nobody answers any more: the test sees the shortfall
                y = np.frombuffer(req.recv(), "<c8").astype(np.complex64)
                self.received.append(y)
                got += y.size
        finally:
            req.close(0)

    def close(self, *threads: threading.Thread) -> None:
        self.stop.set()
        for th in threads:
            th.join(5)
        self.rep.close(0)
        self.ctx.term()


def _engine(**kw: object):
    return make_engine(lab_like_fake(), noise=True, duration_s=DURATION, **kw)


@pytest.mark.parametrize("fmt", ["gnuradio_count", "srsran_dummy"])
def test_io08_exit_when_done_slow_raw_peer_gets_every_sample(fmt: str) -> None:
    x = waveform(N_RUN + 70_000, seed=31)
    peer = _SlowPeer(fmt, x, pause_s=0.002)
    cfg = RawPeerConfig(
        "slow",
        fmt,  # type: ignore[arg-type]
        tx_port=peer.tx_port,
        rx_port="tcp://127.0.0.1:0",
        tx_stream="ap.tx0",
        rx_stream="rx1.rx0",
        base_srate="1e6",
    )
    eng = _engine(unopened_tx="wait")
    tx_th = threading.Thread(target=peer.tx_main, daemon=True)
    tx_th.start()
    try:
        srv = serve(eng, exit_when_done=True, start_paused=False, raw_peers=(cfg,))
        raw = srv.raw_peers[0]
        page = int(raw.rx_handle.page_size)
        queue = int(eng.rx["rx1.rx0"].queue_samples)
        assert N_RUN > 3 * queue  # the engine can run a whole queue ahead of the peer
        # as the CLI does: stop as soon as the shutdown is set (exit_when_done)
        cli = threading.Thread(target=srv.serve_forever, daemon=True)
        cli.start()
        rx_th = threading.Thread(target=peer.rx_main, args=(raw.rx_endpoint, N_RUN, 10.0))
        rx_th.daemon = True
        rx_th.start()
        rx_th.join(WAIT_S)
        cli.join(WAIT_S)
        assert not cli.is_alive(), "the served run did not stop by itself"
        assert srv.engine_error is None
        st = raw.stats()
    finally:
        peer.close(tx_th)
    y = np.concatenate(peer.received) if peer.received else np.zeros(0, np.complex64)
    assert y.size == N_RUN, f"the raw peer got {y.size} of the {N_RUN} samples of the run"
    assert st["rx_samples"] == N_RUN and st["bad"] == 0
    if fmt == "gnuradio_count":
        ask = default_request_items(fmt, page)
        assert ask == 4096 and all(r == struct.pack("<I", ask) for r in peer.tx_requests)
    ref = client_run(_engine(), "ap.tx0", "rx1.rx0", x, N_RUN, send_chunk=4096, read_chunk=4096)
    assert np.array_equal(bits(y), bits(ref))
