"""sremu quickstart: a tone through the quickstart scene with one timed RIS switch.

    python examples/quickstart/quickstart.py [--engine auto|inproc|fake] [--record DIR]

1. Open the scenario's TX (``ap.tx0``) and RX (``mu.rx0``) streams at 3.5 GHz, 1 MS/s.
2. Send 100 ms of a 1 kHz tone as one timed burst starting at t = 0.
3. Schedule ``ris0`` to switch every cell to state 1 at exactly t = 50 ms (RIS time).
4. Receive 100 ms and print the mean power before and after the switch sample.

``inproc`` runs ``Device.inproc("scenarios/quickstart.yaml", traffic={})`` (the WP6 engine:
LOS + RIS physics and thermal noise). The scenario drives ``ap.tx0`` with an in-process tone
generator for headless runs (``sremu run``); ``traffic={}`` removes it so that the tone this
script sends is what the AP transmits (the override is written into the run's
``scenario.resolved.yaml``). ``fake`` uses the identity-channel ``sremu.testing.FakeEngine``
with a TOY gain (1 with all cells in state 0, 0.5 otherwise) so the switch is visible; it is
not physics. ``auto`` picks ``inproc`` when available. With ``--record DIR`` the run is
recorded; ``sremu report DIR`` then renders it and ``sremu replay DIR`` replays it.
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TimeSpec, TxMetadata

REPO = Path(__file__).resolve().parents[2]
SCENARIO = REPO / "scenarios" / "quickstart.yaml"
RATE = Fraction(1_000_000)
N = 100_000  # 100 ms at 1 MS/s
T_SWITCH = Fraction(1, 20)  # 50 ms


def make_device(engine: str, record: str | None) -> tuple[Device, str]:
    """Device for ``engine`` and the engine actually used."""
    if engine in ("auto", "inproc"):
        try:
            # traffic={}: this script, not the scenario's tone generator, feeds ap.tx0
            return Device.inproc(SCENARIO, record=record, traffic={}), "inproc"
        except NotImplementedError as exc:
            if engine == "inproc":
                raise
            print(f"note: {exc}; using the identity-channel FakeEngine with a toy RIS gain")
    from sremu.contracts import RisCapabilities
    from sremu.testing import FakeEngine

    caps = {
        "ris0": RisCapabilities(
            256, (16, 16), "element", ("0", "1"), "analytic:toy", (3.4e9, 3.6e9), 1e6
        )
    }

    def toy_gain(cfg: Any) -> complex:
        return 1.0 if not cfg["ris0"].state.any() else 0.5

    return Device.from_engine(FakeEngine(caps, gain_fn=toy_gain), record=record), "fake"


def main(argv: list[str] | None = None) -> int:
    """Run the quickstart; returns 0."""
    ap = argparse.ArgumentParser(description="sremu quickstart")
    ap.add_argument("--engine", choices=("auto", "inproc", "fake"), default="auto")
    ap.add_argument("--record", default=None)
    args = ap.parse_args(argv)
    dev, used = make_device(args.engine, args.record)
    with dev:
        tx = dev.get_tx_stream(StreamArgs("ap.tx0", 3.5e9, "1e6"))
        rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
        n = np.arange(N)
        tone = np.exp(2j * np.pi * 1e3 * n / float(RATE)).astype(np.complex64)
        md = TxMetadata(
            has_time_spec=True, time_spec=TimeSpec(0, 0.0), start_of_burst=True, end_of_burst=True
        )
        print(f"sent {tx.send(tone, md)} samples ({used} engine)")
        receipt = dev.ris("ris0").set_config(
            states=np.ones(256, dtype=np.int16), time_spec=TimeSpec.from_exact(T_SWITCH)
        )
        print(f"RIS command: {receipt.status}, k_req {receipt.k_req}")
        rx.issue_stream_cmd(StreamCmd("num_done", num_samps=N))
        y = np.zeros(N, np.complex64)
        got = 0
        rmd = RxMetadata()
        while got < N:
            k = rx.recv(y[got:], rmd, timeout=1.0)
            if k == 0:
                print(f"recv stopped: {rmd.error_code.name}")
                break
            got += k
        k_sw = receipt.k_req["mu.rx0"]
        before = 10 * np.log10(np.mean(np.abs(y[:k_sw].astype(np.complex128)) ** 2))
        after = 10 * np.log10(np.mean(np.abs(y[k_sw:got].astype(np.complex128)) ** 2))
        print(f"received {got} samples; switch at sample {k_sw}")
        print(f"mean power before {before:.3f} dB, after {after:.3f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
