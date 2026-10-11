"""Benchmark clients for feature set F5 (run as ``python -m bench._zmq_client ...``).

* ``tx``: opens every TX stream of ``--streams`` through ``sremu.Device.connect`` and sends
  ``--n`` samples of a complex tone (``--tone`` Hz) per stream as one timed burst at sample
  0, in ``--chunk``-sample sends, round-robin over the streams.
* ``rx``: opens every RX stream, issues a timed ``num_done`` stream command for ``--n``
  samples at sample 0, resumes the paused server and reads round-robin until every stream
  has ``--n`` samples. It prints one JSON line: wall time after every stream passed
  ``--warm`` samples, samples read after the warm-up, and the total wall time.

The clients use only the public API and transport layers; they never build an engine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction

import numpy as np


def _tx(a: argparse.Namespace) -> int:
    from sremu.api import Device, StreamArgs, TimeSpec, TxMetadata

    streams = a.streams.split(",")
    n = int(a.n)
    chunk = int(a.chunk)
    rate = float(Fraction(a.rate))
    with Device.connect(a.endpoint, name="bench_tx") as dev:
        txs = [
            dev.get_tx_stream(StreamArgs(stream=s, fc_hz=a.fc, rate_sps=a.rate)) for s in streams
        ]
        sent = 0
        w = 2j * np.pi * float(a.tone) / rate
        while sent < n:
            m = min(chunk, n - sent)
            k = np.arange(sent, sent + m, dtype=np.float64)
            x = np.exp(w * k).astype(np.complex64)
            for tx in txs:
                md = TxMetadata(
                    has_time_spec=sent == 0,
                    time_spec=TimeSpec(0, 0),
                    start_of_burst=sent == 0,
                    end_of_burst=sent + m >= n,
                )
                got = tx.send(x, md)
                if got != m:
                    print(f"short send {got} != {m}", file=sys.stderr)
                    return 2
            sent += m
        for tx in txs:
            tx.recv_async_msg(timeout=60.0)
    print(json.dumps({"tx_sent_per_stream": n, "streams": len(streams)}))
    return 0


def _rx(a: argparse.Namespace) -> int:
    from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TimeSpec

    streams = a.streams.split(",")
    n = int(a.n)
    warm = int(a.warm)
    chunk = int(a.chunk)
    with Device.connect(a.endpoint, name="bench_rx") as dev:
        rxs = [
            dev.get_rx_stream(StreamArgs(stream=s, fc_hz=a.fc, rate_sps=a.rate)) for s in streams
        ]
        for rx in rxs:
            rx.issue_stream_cmd(
                StreamCmd(mode="num_done", num_samps=n, stream_now=False, time_spec=TimeSpec(0, 0))
            )
        buf = np.zeros(chunk, np.complex64)
        got = dict.fromkeys(streams, 0)
        md = RxMetadata()
        t0 = time.perf_counter()
        dev.engine.resume()  # type: ignore[attr-defined]
        t_warm = None
        while min(got.values()) < n:
            for s, rx in zip(streams, rxs, strict=True):
                if got[s] >= n:
                    continue
                c = min(chunk, n - got[s])
                k = rx.recv(buf[:c], md, timeout=120.0)
                if k == 0:
                    print(f"recv error {md.error_code.name} on {s}", file=sys.stderr)
                    return 3
                got[s] += k
            if t_warm is None and min(got.values()) >= warm:
                t_warm = time.perf_counter()
        t_end = time.perf_counter()
    if t_warm is None:
        t_warm = t0
    print(
        json.dumps(
            {
                "wall_total_s": t_end - t0,
                "wall_after_warm_s": t_end - t_warm,
                "samples_after_warm_total": sum(max(0, v - warm) for v in got.values()),
                "streams": len(streams),
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(prog="bench._zmq_client")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tx", "rx"):
        p = sub.add_parser(name)
        p.add_argument("--endpoint", required=True)
        p.add_argument("--streams", required=True)
        p.add_argument("--fc", type=float, required=True)
        p.add_argument("--rate", required=True)
        p.add_argument("--n", required=True)
        p.add_argument("--chunk", default="4096")
        if name == "tx":
            p.add_argument("--tone", default="0")
        else:
            p.add_argument("--warm", default="0")
    a = ap.parse_args(argv)
    from bench import pin_client_cpus

    pin_client_cpus()
    return _tx(a) if a.cmd == "tx" else _rx(a)


if __name__ == "__main__":
    sys.exit(main())
