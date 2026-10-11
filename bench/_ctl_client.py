"""Benchmark controller processes for B4 and E-B3 (``python -m bench._ctl_client ...``).

* ``barrier``: a remote barrier controller (lockstep): WP9's ``Urap(p=0.5)`` with period
  ``1/--rate-hz`` and ``L_min = --l-min`` (latency ``L_min + U[0, L_min]``), run through
  ``sremu.control.remote.RemoteControllerRunner`` (the section 8.5 barrier protocol). After
  registering it sends ``sim.resume`` and serves barriers until none arrives for
  ``--idle-s`` seconds.
* ``async``: an asynchronous command source (paced + async coupling): every ``1/--rate-hz``
  wall seconds it sends one ``ris.set_config`` with random states, either timed
  (``at = t_virtual(now) + --lead-s``, where ``t_virtual`` is extrapolated from one
  ``time_now`` round trip at start, since paced virtual time follows the wall clock) or
  ``asap``. It resumes the paused server first and stops after ``--duration-s``.
* ``tdma``: like ``async`` timed, but switch times follow a TDMA packet schedule (E-B3):
  the switch of guard m (after packet m) is requested at its centre
  ``m*period_s + --guard-centre-s`` with the configuration of packet m+1's target
  (``configs[(m + 1) % T]`` from the ``--configs`` JSON file), sent ``--lead-s`` of wall time
  before the switch time.

Every subcommand prints one JSON line with its counts and round-trip percentiles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction

import numpy as np


def _pct(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)

    def q(p: float) -> float:
        return s[max(0, math.ceil(p / 100 * len(s)) - 1)]

    return {"n": len(s), "p50": q(50), "p90": q(90), "p99": q(99), "p99.9": q(99.9), "max": s[-1]}


def _sleep_until(t: float) -> None:
    rem = t - time.perf_counter()
    if rem > 0.0015:
        time.sleep(rem - 0.0015)
    while time.perf_counter() < t:
        pass


def _barrier(a: argparse.Namespace) -> int:
    from sremu.contracts import LatencyModel
    from sremu.control.builtin import Urap
    from sremu.control.remote import RemoteControllerRunner

    l_min = Fraction(a.l_min)
    ctrl = Urap(
        0.5,
        name=a.name,
        period_s=Fraction(1) / Fraction(a.rate_hz),
        min_latency_s=l_min,
        latency=LatencyModel(l_min, f"uniform:0,{float(l_min)!r}"),
    )
    runner = RemoteControllerRunner(ctrl, a.endpoint, timeout_s=120.0)
    runner.register()
    runner.session.call("sim.resume")
    t0 = time.perf_counter()
    stats = runner.serve(idle_timeout_s=float(a.idle_s))
    wall = time.perf_counter() - t0
    runner.unregister()
    runner.stop()
    print(json.dumps({"barriers": stats.barriers, "commands": stats.commands, "wall_s": wall}))
    return 0


def _async(a: argparse.Namespace) -> int:
    from sremu.contracts import CommandRequest, CommandTime
    from sremu.transport.client import connect_engine

    rate = float(Fraction(a.rate_hz))
    n_ctl = int(a.n_controls)
    rng = np.random.default_rng(int(a.seed))
    eng = connect_engine(a.endpoint, name=a.name, timeout_s=30.0)
    try:
        if a.resume:
            eng.session.call("sim.resume")
        time.sleep(float(a.start_delay_s))
        w0 = time.perf_counter()
        tv0 = eng.time_now()
        w1 = time.perf_counter()
        wsync = 0.5 * (w0 + w1)
        lead = Fraction(a.lead_s)
        rtts: list[float] = []
        status: dict[str, int] = {}
        n = int(float(a.duration_s) * rate)
        t_start = time.perf_counter()
        t_stop = t_start + float(a.duration_s)
        sent = 0
        for k in range(n):
            _sleep_until(t_start + k / rate)
            now = time.perf_counter()
            if now >= t_stop:  # replies slower than 1/rate: stop at the wall deadline, so
                break  # the client ends with the paced run (the achieved rate says it)
            if a.mode == "timed":
                tv = tv0 + Fraction(now - wsync).limit_denominator(10**9)
                at = CommandTime("t", tv + lead)
            else:
                at = CommandTime("asap")
            req = CommandRequest(
                cmd_id=f"{a.name}:{k}",
                op="ris.set_config",
                src=a.name,
                target=a.panel,
                at=at,
                states=rng.integers(0, 2, n_ctl).astype(np.int16),
            )
            s0 = time.perf_counter()
            rc = eng.schedule(req)
            rtts.append((time.perf_counter() - s0) * 1e6)
            key = rc.status if rc.ok else f"rejected:{rc.error.name}"
            status[key] = status.get(key, 0) + 1
            sent += 1
        wall = time.perf_counter() - t_start
    finally:
        eng.close()
    print(
        json.dumps(
            {
                "sent": sent,
                "target": n,
                "status": status,
                "rtt_us": _pct(rtts),
                "wall_s": wall,
                "achieved_rate_hz": sent / wall if wall > 0 else None,
            }
        )
    )
    return 0


def _tdma(a: argparse.Namespace) -> int:
    from sremu.contracts import CommandRequest, CommandTime
    from sremu.transport.client import connect_engine

    with open(a.configs, encoding="utf-8") as fh:
        configs = [np.asarray(c, dtype=np.int16) for c in json.load(fh)]
    period = Fraction(a.period_s)
    centre = Fraction(a.guard_centre_s)
    lead = float(Fraction(a.lead_s))
    eng = connect_engine(a.endpoint, name=a.name, timeout_s=30.0)
    rtts: list[float] = []
    status: dict[str, int] = {}
    try:
        if a.resume:
            eng.session.call("sim.resume")
        w0 = time.perf_counter()
        tv0 = eng.time_now()
        wsync = 0.5 * (w0 + time.perf_counter())
        m0 = math.ceil((tv0 + Fraction(1, 100)) / period)
        n = int(float(a.duration_s) / float(period))
        for i in range(n):
            m = m0 + i
            t_sw = m * period + centre
            # wall time at which to send: lead seconds before the switch (virtual ~ wall)
            _sleep_until(wsync + float(t_sw - tv0) - lead)
            req = CommandRequest(
                cmd_id=f"{a.name}:{m}",
                op="ris.set_config",
                src=a.name,
                target=a.panel,
                at=CommandTime("t", t_sw),
                states=configs[(m + 1) % len(configs)],  # guard m prepares packet m + 1
            )
            s0 = time.perf_counter()
            rc = eng.schedule(req)
            rtts.append((time.perf_counter() - s0) * 1e6)
            key = rc.status if rc.ok else f"rejected:{rc.error.name}"
            status[key] = status.get(key, 0) + 1
    finally:
        eng.close()
    print(json.dumps({"sent": n, "status": status, "rtt_us": _pct(rtts), "first_packet": m0}))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(prog="bench._ctl_client")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("barrier")
    b.add_argument("--endpoint", required=True)
    b.add_argument("--rate-hz", required=True)
    b.add_argument("--l-min", default="1/10000")
    b.add_argument("--idle-s", default="3")
    b.add_argument("--name", default="b4")
    s = sub.add_parser("async")
    s.add_argument("--endpoint", required=True)
    s.add_argument("--rate-hz", required=True)
    s.add_argument("--duration-s", required=True)
    s.add_argument("--mode", choices=("timed", "asap"), default="timed")
    s.add_argument("--lead-s", default="2e-3")
    s.add_argument("--panel", default="ris0")
    s.add_argument("--n-controls", required=True)
    s.add_argument("--seed", default="1")
    s.add_argument("--name", default="b4async")
    s.add_argument("--start-delay-s", default="0")
    s.add_argument("--resume", action="store_true")
    t = sub.add_parser("tdma")
    t.add_argument("--endpoint", required=True)
    t.add_argument("--configs", required=True)
    t.add_argument("--period-s", required=True)
    t.add_argument("--guard-centre-s", required=True)
    t.add_argument("--lead-s", default="2e-3")
    t.add_argument("--duration-s", required=True)
    t.add_argument("--panel", default="ris0")
    t.add_argument("--name", default="eb3")
    t.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    from bench import pin_client_cpus

    pin_client_cpus()
    return {"barrier": _barrier, "async": _async, "tdma": _tdma}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
