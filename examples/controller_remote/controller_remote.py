"""Run a barrier controller in its own process against a served emulator (DESIGN.md 4.9, 8.7).

    python examples/controller_remote/controller_remote.py                 # self-contained
    python examples/controller_remote/controller_remote.py --endpoint tcp://127.0.0.1:5600

Without ``--endpoint`` the script starts the served emulator itself, in a separate OS
process (``python -m sremu.commands.serve remote.yaml``; ``sremu run --serve`` once the CLI
routes it), on ephemeral loopback ports. ``--endpoint`` attaches to one you started with
``remote.yaml`` (the scenario declares the external controller ``probe``).

The controller, :class:`PowerProbe`, is an ordinary in-process controller class (the
``sremu.contracts.Controller`` protocol): at each barrier it reads the user's measured RX
power and alternates the panel between all-0 and all-1. ``sremu.control.remote.run_remote``
drives it over the control socket (``ctrl.barrier`` -> ``ris.set_config`` -> ``ctrl.ack``).

With ``--compare`` (default) the same class then runs in-process on the same scenario; the
script checks that both runs saw bit-identical observations (lockstep barriers make the
remote run deterministic, RT-08) and prints the table. Exit code 0 when they agree.

The remote controller stops after the run's last barrier (``t_b = m*T_c < duration``,
section 4.9; ten for ``remote.yaml``), not after a wall-clock idle timeout: a served
engine may spend seconds JIT-compiling its numba kernels (cold cache) before the first
barrier, so "no barrier for N seconds" does not mean the run is over. The served
emulator (``--exit-when-done``) finishes the run by itself and exits; the script always
stops it before returning, also when something failed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from sremu.contracts import Action, LatencyModel, Observation, ObservationSpec, RisCapabilities
from sremu.control.remote import run_remote

HERE = Path(__file__).resolve().parent
SCENARIO = HERE / "remote.yaml"
# Wall-clock backstops that only end a broken run (they never decide when a run is over):
# no barrier for this long, or the server still running this long after the last barrier.
IDLE_BACKSTOP_S = 180.0
SERVER_EXIT_S = 60.0
# The served emulator gives up on a barrier ack after this long (--ack-timeout-s; counted
# once the controller has attached), so it cannot outlive this script by more than that
# when the script is killed in the middle of the run (SIGKILL or TerminateProcess skip the
# finally below). The controller acks within milliseconds; the server's JIT and rendering
# happen before it sends a barrier, so they do not count against this timeout.
ACK_BACKSTOP_S = 120.0


class PowerProbe:
    """Alternates the panel between all-0 and all-1 and records the observed power."""

    def __init__(self) -> None:
        self.name = "probe"  # must match the scenario's external controller
        self.period_s = Fraction(1, 100)
        self.min_latency_s = Fraction(1, 1000)
        self.latency = LatencyModel(Fraction(1, 1000))
        self.spec = ObservationSpec(("rx_power_dbm", "ris_state"), Fraction(1, 1000))
        self.n = 0
        self.log: list[tuple[Fraction, float, int]] = []

    def setup(self, caps: Mapping[str, RisCapabilities], rng: np.random.Generator) -> None:
        self.n = int(caps["ris0"].n_controls or caps["ris0"].n_elements)

    def decide(self, obs: Observation) -> Sequence[Action]:
        state = int(obs.ris_state["ris0"][0])
        self.log.append((obs.t_b, float(obs.rx_power_dbm["mu.rx0"]), state))
        nxt = 1 - state
        return [Action("ris0", states=np.full(self.n, nxt, dtype=np.int16))]


def serve(scenario: Path) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    """Start the served emulator in its own process; returns it and its endpoints."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sremu.commands.serve",
            str(scenario),
            "--serve",
            "tcp://127.0.0.1:0",
            "--pub",
            "none",
            "--tx",
            "tcp://127.0.0.1:0",
            "--no-paused",
            "--exit-when-done",
            "--ack-timeout-s",
            str(ACK_BACKSTOP_S),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line.startswith("SREMU-SERVE "):
        proc.kill()
        raise RuntimeError(f"the server did not start: {line!r}")
    return proc, json.loads(line.split(" ", 1)[1])


def run_barriers(scenario: Path, period_s: Fraction) -> int:
    """Number of barriers of a run from t = 0: ``t_b = m*T_c < duration`` (section 4.9)."""
    from sremu.scenario.load import load_scenario

    duration = Fraction(load_scenario(scenario, warn=False).scenario.run.duration_s)
    return math.ceil(duration / Fraction(period_s))


def stop_server(proc: subprocess.Popen[str]) -> None:
    """Make sure the served emulator is gone (it must never outlive this script)."""
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    if proc.stdout is not None:
        proc.stdout.close()


def run_inprocess(scenario: Path) -> list[tuple[Fraction, float, int]]:
    """The same controller class in-process (reference for the comparison)."""
    from sremu.engine.emulator import Emulator

    ctrl = PowerProbe()
    eng = Emulator.from_scenario(scenario)
    eng.register_controller(ctrl)
    eng.run()
    return ctrl.log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", default=None, help="control endpoint of a running server")
    ap.add_argument("--no-compare", action="store_true", help="skip the in-process run")
    args = ap.parse_args(argv)
    ctrl = PowerProbe()
    n_barriers = run_barriers(SCENARIO, ctrl.period_s)
    proc = None
    endpoint = args.endpoint
    try:
        if endpoint is None:
            proc, info = serve(SCENARIO)
            endpoint = info["endpoints"]["control"]
            print(f"served emulator at {endpoint}", flush=True)
        # serve exactly the run's barriers, then unregister; the server runs on to the end
        stats = run_remote(ctrl, endpoint, max_barriers=n_barriers, idle_timeout_s=IDLE_BACKSTOP_S)
        if proc is not None:
            proc.wait(timeout=SERVER_EXIT_S)  # --exit-when-done: the run ends without us
    finally:
        if proc is not None:
            stop_server(proc)
    print(f"remote controller: {stats.barriers} barriers, {stats.commands} commands")
    if proc is not None and proc.returncode != 0:
        print(f"the served emulator failed (exit code {proc.returncode})", file=sys.stderr)
        return 1
    print("   t_b (s)   RX power (dBm)   state")
    for t_b, p, s in ctrl.log:
        print(f"  {float(t_b):8.3f}   {p:14.6f}   {s:5d}")
    if args.no_compare:
        return 0
    ref = run_inprocess(SCENARIO)
    same = [(t, s) for t, _, s in ref] == [
        (t, s) for t, _, s in ctrl.log
    ] and np.array_equal(  # bit-identical powers (NaN at t_b = 0: no samples before it)
        np.array([p for _, p, _ in ref]).view(np.uint64),
        np.array([p for _, p, _ in ctrl.log]).view(np.uint64),
    )
    print(f"in-process run of the same controller: identical observations: {same}")
    return 0 if same and stats.barriers == len(ref) else 1


if __name__ == "__main__":
    sys.exit(main())
