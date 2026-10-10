"""IO-08: raw REQ/REP against stub peers for both request formats (DESIGN.md 8.4, 10.4).

The request formats are pinned from upstream source (see ``sremu.transport.rawreqrep``):

* srsRAN 4G (``rf_zmq_imp_rx.c`` / ``rf_zmq_imp_tx.c``): the RX side is a REQ that sends one
  byte ``0xFF`` and accepts any amount of raw baseband in reply; the TX side is a bound REP
  that ignores the request's content and answers with ``nsamples`` samples per call;
* GNU Radio 3.10 (``req_source_impl.cc`` / ``rep_sink_impl.cc``): the REQ sends a 4-byte
  host-order ``uint32`` item count; the REP answers ``min(available, count)`` items.

Each stub peer implements exactly that behaviour (its TX streams a known waveform in
irregular chunks, then zeros, like srsRAN's ``tx_zeros``; its RX asks with the pinned
request). Acceptance (sample-exact in lockstep): the samples the stub's RX collects are
bitwise equal to an in-process lockstep run with the same TX samples, nothing is dropped or
duplicated, and every reply carries exactly the requested count (``gnuradio_count``) or
``reply_items`` (``srsran_dummy``). A request in the wrong format is counted as a bad packet
and answered with an empty reply.
"""

from __future__ import annotations

import struct
import threading
from fractions import Fraction

import numpy as np
import pytest
import zmq

from sremu import rng
from sremu.transport.rawreqrep import (
    SRSRAN_DUMMY_REQUEST,
    RawPeerConfig,
    RawProtocolError,
    encode_request,
    parse_request,
)
from tests.io._wp8_support import bits, served, wait_until
from tests.runtime._engine_support import client_run, lab_like_fake, make_engine, waveform

pytestmark = pytest.mark.io

F = Fraction(10**6)
N_RX = 30_000
N_TX = N_RX + 8192


def test_io08_request_formats_pinned_from_source() -> None:
    assert encode_request("srsran_dummy", 1234) == b"\xff" == SRSRAN_DUMMY_REQUEST
    assert encode_request("gnuradio_count", 1234) == struct.pack("<I", 1234)
    assert encode_request("gnuradio_count", 2**32 - 1) == b"\xff\xff\xff\xff"
    assert parse_request("srsran_dummy", b"\xff") is None
    assert parse_request("gnuradio_count", (4096).to_bytes(4, "little")) == 4096
    for fmt, bad in (
        ("srsran_dummy", b""),
        ("srsran_dummy", b"\x00"),
        ("srsran_dummy", b"\xff\xff"),
        ("srsran_dummy", struct.pack("<I", 10)),
        ("gnuradio_count", b"\xff"),
        ("gnuradio_count", b"\x01\x00\x00"),
        ("gnuradio_count", b"\x01\x00\x00\x00\x00"),
    ):
        with pytest.raises(RawProtocolError):
            parse_request(fmt, bad)


class _StubPeer:
    """A srsRAN- or GNU Radio-style peer: TX REP bound (streams ``x`` then zeros in
    ``chunks``), RX REQ connected to the emulator (asks with the pinned request)."""

    def __init__(self, fmt: str, x: np.ndarray, chunks: list[int], asks: list[int]) -> None:
        self.fmt = fmt
        self.ctx = zmq.Context()
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.setsockopt(zmq.LINGER, 0)
        self.tx_port = f"tcp://127.0.0.1:{self.rep.bind_to_random_port('tcp://127.0.0.1')}"
        self.x = x
        self.chunks = chunks
        self.asks = asks
        self.sent = 0
        self.pending = np.zeros(0, np.complex64)  # GNU Radio: items in the sink's input buffer
        self.received: list[np.ndarray] = []
        self.reply_sizes: list[int] = []
        self.requested: list[int] = []
        self.stop = threading.Event()
        self.tx_requests: list[bytes] = []

    def _produce(self, n: int) -> np.ndarray:
        a = self.x[self.sent : self.sent + n]
        if a.size < n:
            a = np.concatenate([a, np.zeros(n - a.size, np.complex64)])
        self.sent += n
        return a.astype(np.complex64)

    def tx_main(self) -> None:
        i = 0
        while not self.stop.is_set():
            if not self.rep.poll(50):
                continue
            req = self.rep.recv()
            self.tx_requests.append(req)
            chunk = self._produce(self.chunks[i % len(self.chunks)])
            i += 1
            if self.fmt == "srsran_dummy":  # rf_zmq_imp_tx.c: request ignored, nsamples sent
                self.rep.send(chunk.astype("<c8").tobytes())
            else:  # rep_sink_impl.cc: min(available, req) when the request has >= 4 bytes
                self.pending = np.concatenate([self.pending, chunk])
                n = self.pending.size
                if len(req) >= 4:
                    n = min(n, int(struct.unpack("<I", req[:4])[0]))
                self.rep.send(self.pending[:n].astype("<c8").tobytes())
                self.pending = self.pending[n:]

    def rx_main(self, rx_port: str, n_total: int) -> None:
        req = self.ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.connect(rx_port)
        got = 0
        i = 0
        while got < n_total and not self.stop.is_set():
            if self.fmt == "srsran_dummy":
                req.send(b"\xff")  # rf_zmq_imp_rx.c: uint8_t dummy = 0xFF
            else:
                ask = self.asks[i % len(self.asks)]
                self.requested.append(ask)
                req.send(struct.pack("<I", ask))  # req_source_impl.cc: uint32 req_len
            i += 1
            while not req.poll(100):
                if self.stop.is_set():
                    req.close(0)
                    return
            msg = req.recv()
            assert len(msg) % 8 == 0
            y = np.frombuffer(msg, "<c8").astype(np.complex64)
            self.reply_sizes.append(y.size)
            self.received.append(y)
            got += y.size
        req.close(0)

    def close(self, *threads: threading.Thread) -> None:
        self.stop.set()
        for th in threads:
            th.join(5)
        self.rep.close(0)
        self.ctx.term()


def _engine(**kw: object):
    return make_engine(lab_like_fake(moving=True), noise=True, **kw)


@pytest.mark.parametrize("fmt", ["srsran_dummy", "gnuradio_count"])
def test_io08_raw_reqrep_sample_exact_vs_inprocess(fmt: str) -> None:
    g = rng.generator(8, "trial", f"wp8/io08/{fmt}")
    x = waveform(N_TX, seed=21)
    chunks = [int(v) for v in g.integers(1, 3000, 64)]
    asks = [int(v) for v in g.integers(1, 5000, 64)]
    peer = _StubPeer(fmt, x, chunks, asks)
    cfg = RawPeerConfig(
        "stub",
        fmt,  # type: ignore[arg-type]
        tx_port=peer.tx_port,
        rx_port="tcp://127.0.0.1:0",
        tx_stream="ap.tx0",
        rx_stream="rx1.rx0",
        base_srate="1e6",
        request_items=1000,
        reply_items=777,
    )
    eng = _engine(unopened_tx="wait")
    tx_th = threading.Thread(target=peer.tx_main, daemon=True)
    tx_th.start()
    try:
        with served(eng, raw_peers=(cfg,)) as srv:
            raw = srv.raw_peers[0]
            rx_th = threading.Thread(target=peer.rx_main, args=(raw.rx_endpoint, N_RX), daemon=True)
            rx_th.start()
            rx_th.join(120)
            assert not rx_th.is_alive(), "the stub RX did not get its samples"
            st = raw.stats()
            bad = srv.counters["bad_packets"]
    finally:
        peer.close(tx_th)
    y = np.concatenate(peer.received)
    if fmt == "srsran_dummy":
        assert all(n == 777 for n in peer.reply_sizes)
        assert all(r == b"\xff" for r in peer.tx_requests)
    else:
        assert peer.reply_sizes == peer.requested  # exactly the requested counts
        assert all(r == struct.pack("<I", 1000) for r in peer.tx_requests)
    assert bad == 0 and st["bad"] == 0
    assert st["rx_samples"] == y.size >= N_RX
    ref = client_run(_engine(), "ap.tx0", "rx1.rx0", x, y.size, send_chunk=4096, read_chunk=4096)
    assert np.array_equal(bits(y), bits(ref))
    assert float(np.mean(np.abs(y) ** 2)) > 0.0


@pytest.mark.parametrize(
    ("fmt", "wrong"), [("srsran_dummy", struct.pack("<I", 64)), ("gnuradio_count", b"\xff")]
)
def test_io08_wrong_request_format_counted_and_answered_empty(fmt: str, wrong: bytes) -> None:
    cfg = RawPeerConfig("p", fmt, rx_port="tcp://127.0.0.1:0", rx_stream="rx1.rx0")  # type: ignore[arg-type]
    eng = make_engine(lab_like_fake(), unopened_tx="silent")
    with served(eng, raw_peers=(cfg,)) as srv:
        ctx = zmq.Context()
        req = ctx.socket(zmq.REQ)
        req.setsockopt(zmq.LINGER, 0)
        req.connect(srv.raw_peers[0].rx_endpoint)
        for _ in range(5):
            req.send(wrong)
            assert req.poll(10_000)
            assert req.recv() == b""
        good = b"\xff" if fmt == "srsran_dummy" else struct.pack("<I", 100)
        req.send(good)
        assert req.poll(30_000)
        reply = req.recv()
        req.close(0)
        ctx.term()
        wait_until(lambda: srv.counters["bad_packets"] == 5, 10.0, "the bad requests")
        assert srv.counters.reasons_snapshot() == {"raw_request": 5}
    n = 512 if fmt == "srsran_dummy" else 100  # srsran: reply_items defaults to the page
    assert len(reply) == 8 * n


def test_io08_device_args_and_lockstep_only() -> None:
    c = RawPeerConfig.from_device_args(
        "enb",
        "srsran_dummy",
        "tx_port=tcp://*:2000,rx_port=tcp://localhost:2001,base_srate=23.04e6",
        tx_stream="enb.tx0",
        rx_stream="enb.rx0",
    )
    assert c.tx_port == "tcp://127.0.0.1:2000"
    assert c.rx_port == "tcp://127.0.0.1:2001"
    assert Fraction(c.base_srate) == Fraction(23_040_000)  # type: ignore[arg-type]
    from sremu.errors import NotPermitted

    eng = make_engine(lab_like_fake(), mode="paced", coupling="async")
    peer = RawPeerConfig("p", "srsran_dummy", rx_port="tcp://127.0.0.1:0", rx_stream="rx1.rx0")
    with pytest.raises(NotPermitted, match="lockstep"):
        served(eng, raw_peers=(peer,)).__enter__()
