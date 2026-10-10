"""WP8 test client run as a separate OS process (``python -m tests.io._wp8_client ...``).

Subcommands (one or more sessions, i.e. control endpoints, served in order):

* ``tx``: TX generator. Opens a TX stream through ``sremu.Device.connect`` and sends the
  waveform from ``--x`` (a .npy file) as one timed burst at sample 0, in chunks
  ``--chunks`` (cycled), EOB on the last chunk.
* ``rx``: RX sink. Opens the RX streams through ``sremu.Device.connect``, issues a timed
  ``num_done`` stream command at sample 0 for ``--n`` samples, sends ``sim.resume`` and
  reads with ``recv`` in chunks ``--chunk`` (alternating over the streams); writes
  ``<out>/<session>_<stream>.npy``.
* ``ctrl``: remote controller. Runs ``tests.runtime._engine_support.ScriptedController``
  (alternating states, keyed latency jitter) with ``sremu.control.remote.run_remote``.

The client imports only the API and transport layers (plus the test controller for
``ctrl``); it never builds an engine.
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

# Wall-clock waits for the served engine (backstops only; run_clients bounds the whole run at
# 300 s). The first samples can take tens of seconds on a loaded CI runner: the served engine
# JIT-compiles its numba kernels on a cold cache while the other xdist workers compete for
# 2 vCPUs (GitHub Actions run 38086720572: "recv error TIMEOUT" after the former 30 s).
WAIT_S = 120.0


def _log(msg: str) -> None:
    import os
    import time

    if os.environ.get("SREMU_WP8_CLIENT_LOG"):
        print(f"{time.monotonic():.3f} {msg}", file=sys.stderr, flush=True)


def _tx(args: argparse.Namespace) -> int:
    from sremu.api import Device, StreamArgs, TimeSpec, TxMetadata

    x = np.load(args.x).astype(np.complex64)
    chunks = [int(c) for c in args.chunks.split(",")]
    for ep in args.sessions.split(","):
        with Device.connect(ep, name="txgen") as dev:
            tx = dev.get_tx_stream(
                StreamArgs(stream=args.stream, fc_hz=args.fc, rate_sps=args.rate)
            )
            sent = 0
            i = 0
            while sent < x.size:
                part = x[sent : sent + chunks[i % len(chunks)]]
                md = TxMetadata(
                    has_time_spec=sent == 0,
                    time_spec=TimeSpec(0, 0),
                    start_of_burst=sent == 0,
                    end_of_burst=sent + part.size >= x.size,
                )
                n = tx.send(part, md)
                assert n == part.size, (n, part.size)
                sent += part.size
                i += 1
            ack = tx.recv_async_msg(timeout=WAIT_S)
            assert ack is not None and ack.event_code.name == "BURST_ACK", ack
            _log(f"tx session {ep} done")
    print(f"tx sent {x.size} samples per session")
    return 0


def _rx(args: argparse.Namespace) -> int:
    from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TimeSpec

    streams = args.stream.split(",")
    n = int(args.n)
    out = Path(args.out)
    chunks = [int(c) for c in args.chunk.split(",")]
    for j, ep in enumerate(args.sessions.split(",")):
        with Device.connect(ep, name="rxsink") as dev:
            rxs = [
                dev.get_rx_stream(StreamArgs(stream=s, fc_hz=args.fc, rate_sps=args.rate))
                for s in streams
            ]
            for rx in rxs:
                rx.issue_stream_cmd(
                    StreamCmd(
                        mode="num_done", num_samps=n, stream_now=False, time_spec=TimeSpec(0, 0)
                    )
                )
            dev.engine.resume()  # type: ignore[attr-defined]
            bufs = {s: np.zeros(n, np.complex64) for s in streams}
            got = dict.fromkeys(streams, 0)
            md = RxMetadata()
            i = 0
            while min(got.values()) < n:
                for s, rx in zip(streams, rxs, strict=True):
                    if got[s] >= n:
                        continue
                    c = min(chunks[i % len(chunks)], n - got[s])
                    k = rx.recv(bufs[s][got[s] : got[s] + c], md, timeout=WAIT_S)
                    if k == 0:
                        print(f"recv error {md.error_code.name} on {s}", file=sys.stderr)
                        return 3
                    assert md.first_sample == got[s], (md.first_sample, got[s])
                    got[s] += k
                    i += 1
            for s in streams:
                np.save(out / f"{j}_{s}.npy", bufs[s])
            _log(f"rx session {j} done")
    print(f"rx received {n} samples per stream per session")
    return 0


def _ctrl(args: argparse.Namespace) -> int:
    from sremu.contracts import LatencyModel
    from sremu.control.remote import run_remote
    from tests.runtime._engine_support import ScriptedController, alternate_script

    ctrl = ScriptedController(
        args.name,
        "1/1000",
        "2/10000",
        alternate_script(args.panel, int(args.n_controls)),
        latency=LatencyModel(Fraction(2, 10_000), "uniform:0,3e-4"),
        sleep_s=float(args.sleep_s),
    )
    stats = run_remote(ctrl, args.sessions, idle_timeout_s=float(args.idle_s), unregister=False)
    print(f"ctrl served {stats.barriers} barriers, {stats.commands} commands")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="_wp8_client")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tx")
    t.add_argument("--sessions", required=True)
    t.add_argument("--stream", required=True)
    t.add_argument("--fc", type=float, required=True)
    t.add_argument("--rate", required=True)
    t.add_argument("--x", required=True)
    t.add_argument("--chunks", default="1000,777")
    r = sub.add_parser("rx")
    r.add_argument("--sessions", required=True)
    r.add_argument("--stream", required=True)
    r.add_argument("--fc", type=float, required=True)
    r.add_argument("--rate", required=True)
    r.add_argument("--n", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--chunk", default="333,4096")
    c = sub.add_parser("ctrl")
    c.add_argument("--sessions", required=True)
    c.add_argument("--name", default="alt")
    c.add_argument("--panel", default="ris0")
    c.add_argument("--n-controls", required=True)
    c.add_argument("--sleep-s", default="0")
    c.add_argument("--idle-s", default="3")
    args = ap.parse_args(argv)
    return {"tx": _tx, "rx": _rx, "ctrl": _ctrl}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
