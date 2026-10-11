"""Benchmark clients for feature set F5 (run as ``python -m bench._zmq_client ...``).

* ``tx``: opens every TX stream of ``--streams`` through ``sremu.Device.connect`` and sends
  ``--n`` samples of a complex tone (``--tone`` Hz) per stream as one timed burst at sample
  0, in ``--chunk``-sample sends, round-robin over the streams.
* ``rx``: opens every RX stream, issues a timed ``num_done`` stream command for ``--n``
  samples at sample 0, resumes the paused server and reads round-robin until every stream
  has ``--n`` samples, or until ``--wall-max-s`` after the resume (then ``truncated``).

Each client prints one JSON line. Its timestamps are raw ``time.perf_counter_ns()`` values:
that clock is system-wide (Linux ``CLOCK_MONOTONIC``, Windows ``QueryPerformanceCounter``,
Python >= 3.10), so :func:`bench._engine_point.measure_zmq` compares them with the
timestamps the served engine takes in the benchmark process (and checks that they are
ordered with its own launch and completion times). The RX line holds ``t_start_ns``
(process start), ``t_resume_ns``, ``t_warm_ns`` (every stream has read ``--warm`` samples;
None if never), ``t_end_ns`` (the last sample was read; on truncation: the last read that
returned data), the samples read per stream, the largest gap between two reads that
returned data and the idle time before a truncation, plus the old client-only figures
(``wall_after_warm_s``: wall time after the client's own warm-up). Both clients report the
CPU affinity they ran with (Linux; None elsewhere) after :func:`bench.pin_client_cpus`.

``--pause-at K --pause-s S`` (RX, test injection): sleep ``S`` seconds once, right after the
first read round in which every stream has at least ``K`` samples (a starved client; with
``K`` below ``--warm`` the pause falls before the client's own warm-up completes).

The clients use only the public API and transport layers; they never build an engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

import numpy as np


def _affinity() -> str | None:
    """CPU affinity of this process (Linux cpulist syntax), None where unavailable."""
    get = getattr(os, "sched_getaffinity", None)
    if get is None:
        return None
    from bench._cpus import format_cpu_list

    return format_cpu_list(get(0))


def _tx(a: argparse.Namespace, t_start_ns: int) -> int:
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
        t_sent_ns = time.perf_counter_ns()
        for tx in txs:
            tx.recv_async_msg(timeout=60.0)
    print(
        json.dumps(
            {
                "tx_sent_per_stream": n,
                "streams": len(streams),
                "t_start_ns": t_start_ns,
                "t_sent_ns": t_sent_ns,
                "t_end_ns": time.perf_counter_ns(),
                "affinity": _affinity(),
            }
        )
    )
    return 0


def _rx(a: argparse.Namespace, t_start_ns: int) -> int:
    from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TimeSpec

    streams = a.streams.split(",")
    n = int(a.n)
    warm = int(a.warm)
    chunk = int(a.chunk)
    wall_max_ns = int(float(a.wall_max_s) * 1e9) if float(a.wall_max_s) > 0 else None
    recv_timeout = float(a.recv_timeout_s)
    pause_at = int(a.pause_at)
    pause_s = float(a.pause_s)
    paused = pause_s <= 0
    truncated = False
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
        t0 = time.perf_counter_ns()
        deadline = None if wall_max_ns is None else t0 + wall_max_ns
        dev.engine.resume()  # type: ignore[attr-defined]
        t_warm: int | None = None
        t_data: int | None = None  # last read that returned data
        max_gap_ns = 0
        while min(got.values()) < n and not truncated:
            for s, rx in zip(streams, rxs, strict=True):
                if got[s] >= n:
                    continue
                c = min(chunk, n - got[s])
                tmo = recv_timeout
                if deadline is not None:
                    left = (deadline - time.perf_counter_ns()) * 1e-9
                    if left <= 0:
                        truncated = True
                        break
                    tmo = min(tmo, max(left, 1e-3))
                k = rx.recv(buf[:c], md, timeout=tmo)
                now = time.perf_counter_ns()
                if k == 0:
                    if deadline is not None and now >= deadline:
                        truncated = True
                        break
                    print(f"recv error {md.error_code.name} on {s}", file=sys.stderr)
                    return 3
                if t_data is not None:
                    max_gap_ns = max(max_gap_ns, now - t_data)
                t_data = now
                got[s] += k
            if not paused and min(got.values()) >= pause_at:
                paused = True
                time.sleep(pause_s)  # injected starvation (tests, stress)
            if t_warm is None and min(got.values()) >= warm:
                t_warm = time.perf_counter_ns()
        t_stop = time.perf_counter_ns()
    t_end = t_data if t_data is not None else t_stop
    wall_after_warm = (t_end - (t_warm if t_warm is not None else t0)) * 1e-9
    print(
        json.dumps(
            {
                "streams": len(streams),
                "n": n,
                "warm": warm,
                "got": got,
                "got_min": min(got.values()),
                "truncated": truncated,
                "t_start_ns": t_start_ns,
                "t_resume_ns": t0,
                "t_warm_ns": t_warm,
                "t_end_ns": t_end,
                "max_gap_s": max_gap_ns * 1e-9,
                "idle_at_stop_s": (t_stop - t_end) * 1e-9,
                "wall_total_s": (t_end - t0) * 1e-9,
                "wall_after_warm_s": wall_after_warm,
                "samples_after_warm_total": sum(max(0, v - warm) for v in got.values()),
                "pause_s": pause_s,
                "affinity": _affinity(),
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    t_start_ns = time.perf_counter_ns()
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
            p.add_argument("--wall-max-s", default="0", help="stop reading after (0 = never)")
            p.add_argument("--recv-timeout-s", default="120", help="no data for this long: error")
            p.add_argument("--pause-at", default="0", help=argparse.SUPPRESS)
            p.add_argument("--pause-s", default="0", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    from bench import pin_client_cpus

    pin_client_cpus()
    return _tx(a, t_start_ns) if a.cmd == "tx" else _rx(a, t_start_ns)


if __name__ == "__main__":
    sys.exit(main())
