"""E2 without GNU Radio: a stub peer with GNU Radio's pinned REQ/REP behaviour (8.4, IO-08).

    python examples/gnuradio_zmq/raw_peer_demo.py [--at 0.02] [--n 50000]

Serves ``gr_link.yaml`` with one raw REQ/REP peer in ``gnuradio_count`` format, exactly as
``gr_flowgraph.py`` would connect to it, but the peer is a stub implementing the blocks'
pinned behaviour (GNU Radio 3.10 ``rep_sink_impl.cc``: answer ``min(available, count)``
items to a 4-byte ``uint32`` request; ``req_source_impl.cc``: ask with a 4-byte ``uint32``
item count). The transmitter streams QPSK (4 samples per symbol, amplitude 0.5); a control
client schedules one RIS switch at RIS time ``--at`` before virtual time starts
(``flip_ris.py`` does the same against a separately served emulator).

The script prints the sample at which the receiver sees the switch (the command record's
``k_applied``, ``to_ticks(t_sw + tau_RIS->rx)``), the least-squares channel gain and the EVM
of each RIS state, and checks that the received waveform's change sits exactly at that
sample (the residual against the pre-switch channel jumps there and not before).
"""

from __future__ import annotations

import argparse
import base64
import struct
import sys
import threading
from pathlib import Path

import numpy as np
import zmq

from sremu import rng
from sremu.engine.emulator import Emulator
from sremu.transport.client import ClientSession
from sremu.transport.control_server import ControlServer, ServerConfig
from sremu.transport.rawreqrep import RawPeerConfig

HERE = Path(__file__).resolve().parent


def qpsk(n_sym: int, seed: int = 1) -> np.ndarray:
    """QPSK at 4 samples per symbol (rectangular), amplitude 0.5, keyed symbols."""
    k = rng.generator(seed, "trial", "examples/gnuradio_zmq").integers(0, 4, n_sym)
    pts = 0.5 * np.exp(1j * (np.pi / 4 + np.pi / 2 * k))
    return np.repeat(pts, 4).astype(np.complex64)


class StubGnuRadioPeer:
    """``rep_sink`` (TX, bound) and ``req_source`` (RX, connected) with GNU Radio's pinned
    behaviour; the sink's input buffer is refilled in irregular chunks."""

    def __init__(self, x: np.ndarray) -> None:
        self.ctx = zmq.Context()
        self.rep = self.ctx.socket(zmq.REP)
        self.tx_port = f"tcp://127.0.0.1:{self.rep.bind_to_random_port('tcp://127.0.0.1')}"
        self.x = x
        self.pos = 0
        self.buf = np.zeros(0, np.complex64)
        self.stop = threading.Event()
        self.rx: list[np.ndarray] = []

    def rep_sink(self) -> None:
        chunks = [1000, 4096, 333, 2048]
        i = 0
        while not self.stop.is_set():
            if not self.rep.poll(50):
                continue
            req = self.rep.recv()
            c = chunks[i % len(chunks)]
            i += 1
            more = self.x[self.pos : self.pos + c]
            more = np.concatenate([more, np.zeros(c - more.size, np.complex64)])
            self.pos += c
            self.buf = np.concatenate([self.buf, more])
            n = self.buf.size
            if len(req) >= 4:
                n = min(n, struct.unpack("<I", req[:4])[0])
            self.rep.send(self.buf[:n].astype("<c8").tobytes())
            self.buf = self.buf[n:]

    def req_source(self, rx_port: str, n_total: int) -> None:
        req = self.ctx.socket(zmq.REQ)
        req.connect(rx_port)
        got = 0
        while got < n_total:
            req.send(struct.pack("<I", min(4096, n_total - got)))
            y = np.frombuffer(req.recv(), "<c8").astype(np.complex64)
            self.rx.append(y)
            got += y.size
        req.close(0)

    def close(self, *threads: threading.Thread) -> None:
        self.stop.set()
        for th in threads:
            th.join(5)
        self.rep.close(0)
        self.ctx.term()


def run(at: str = "0.02", n: int = 50_000) -> dict[str, object]:
    """Run the demo; returns the switch sample, gains, EVMs and the waveforms."""
    x = qpsk(n // 4 + 4096)
    peer = StubGnuRadioPeer(x)
    tx_th = threading.Thread(target=peer.rep_sink, daemon=True)
    tx_th.start()
    eng = Emulator.from_scenario(HERE / "gr_link.yaml", unopened_tx="wait", external_inputs=True)
    cfg = RawPeerConfig(
        "gr",
        "gnuradio_count",
        tx_port=peer.tx_port,
        rx_port="tcp://127.0.0.1:0",
        tx_stream="gr.tx0",
        rx_stream="grrx.rx0",
        base_srate="1e6",
    )
    srv = ControlServer(
        eng,
        ServerConfig(
            control_endpoint="tcp://127.0.0.1:0",
            pub_endpoint=None,
            tx_endpoint="tcp://127.0.0.1:0",
            start_paused=True,
            raw_peers=(cfg,),
        ),
    ).start()
    try:
        s = ClientSession(str(srv.endpoints["control"]), name="flip")
        nc = int(s.call("caps.get")["caps"]["ris0"]["n_controls"])
        st = base64.b64encode(np.ones(nc, "<i2").tobytes()).decode("ascii")
        r = s.call(
            "ris.set_config", panel="ris0", at={"t": at}, config={"states_b64": st, "dtype": "i2"}
        )
        s.call("sim.resume")
        s.close()
        rx_th = threading.Thread(
            target=peer.req_source, args=(srv.raw_peers[0].rx_endpoint, n), daemon=True
        )
        rx_th.start()
        rx_th.join(120)
        if rx_th.is_alive():
            raise RuntimeError("the stub peer did not receive its samples")
    finally:
        srv.stop()
        peer.close(tx_th)
    y = np.concatenate(peer.rx)[:n].astype(np.complex128)
    rec = next(rr for rr in eng.records if rr.cmd_id == r["id"])
    k = int(rec.k_applied["grrx.rx0"])
    xs = x[:n].astype(np.complex128)
    out: dict[str, object] = {
        "k_switch": k,
        "k_req": r["k_req"],
        "status": rec.status,
        "lateness": rec.lateness_samples,
        "y": y,
        "x": xs,
    }
    for name, sl in (("state0", slice(16, k)), ("state1", slice(k, n))):
        h = complex(np.vdot(xs[sl], y[sl]) / np.vdot(xs[sl], xs[sl]))
        e = y[sl] - h * xs[sl]
        out[name] = {
            "gain_db": 20 * np.log10(abs(h)),
            "evm_pct": 100
            * float(np.sqrt(np.mean(np.abs(e) ** 2) / np.mean(np.abs(h * xs[sl]) ** 2))),
        }
    h0 = complex(np.vdot(xs[16:k], y[16:k]) / np.vdot(xs[16:k], xs[16:k]))
    resid = np.abs(y - h0 * xs)
    floor = float(np.max(resid[16:k]))
    out["first_jump"] = int(np.argmax(resid[16:] > 2 * floor)) + 16
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E2 with a stub GNU Radio peer.")
    ap.add_argument("--at", default="0.02")
    ap.add_argument("--n", type=int, default=50_000)
    args = ap.parse_args(argv)
    res = run(args.at, args.n)
    print(
        f"RIS switch at RIS time {args.at} s: receiver sees it at sample {res['k_switch']} "
        f"(k_req {res['k_req']}, status {res['status']}, lateness {res['lateness']})"
    )
    for name in ("state0", "state1"):
        d = res[name]
        assert isinstance(d, dict)
        print(f"  {name}: channel gain {d['gain_db']:.3f} dB, EVM {d['evm_pct']:.3f} %")
    print(f"  residual against the state-0 channel first jumps at sample {res['first_jump']}")
    return 0 if res["first_jump"] == res["k_switch"] else 1


if __name__ == "__main__":
    sys.exit(main())
