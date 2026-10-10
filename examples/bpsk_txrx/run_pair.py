"""Run the ported v1 BPSK TX/RX pair in one process (DESIGN.md 8.2; IO-07).

    python examples/bpsk_txrx/run_pair.py [--engine auto|inproc|fake] [--record DIR]

``inproc`` uses ``Device.inproc(bpsk.yaml)`` (the WP6 engine: free-space LOS with path loss
and thermal noise). ``fake`` uses ``sremu.testing.FakeEngine`` (identity channel, exact time
base): the receiver then gets the transmitted waveform bit for bit. ``auto`` (default) uses
``inproc`` when the engine is available, else ``fake``. Both scripts run unmodified through
``sremu.compat``, in the v1 order (transmitter first, then receiver); with ``--record`` the
run directory gets SigMF recordings and ``events.jsonl``.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sremu.api import Device
from sremu.compat import use_device

HERE = Path(__file__).resolve().parent


def make_device(engine: str, record: str | None) -> tuple[Device, str]:
    """Device for ``engine`` (``auto``, ``inproc`` or ``fake``) and the engine used."""
    if engine in ("auto", "inproc"):
        try:
            return Device.inproc(HERE / "bpsk.yaml", record=record), "inproc"
        except NotImplementedError as exc:
            if engine == "inproc":
                raise
            print(f"note: {exc}; using the identity-channel FakeEngine")
    from sremu.testing import FakeEngine

    class _Engine(FakeEngine):
        def tx_head(self, h: Any) -> int:
            """Absolute TX head, for recording untimed sends (CCR WP7-1)."""
            return int(self._tx[h.id].head)

    return Device.from_engine(_Engine(), record=record), "fake"


def run(engine: str = "auto", record: str | None = None) -> dict[str, Any]:
    """Run TX then RX; returns the waveforms, the samples sent and the engine used."""
    dev, used = make_device(engine, record)
    with dev, use_device(dev):
        tx = runpy.run_path(str(HERE / "usrp_tx.py"), run_name="__main__")
        rx = runpy.run_path(str(HERE / "usrp_rx.py"), run_name="__main__")
    return {
        "engine": used,
        "sent": tx["make_waveform"]()[0],
        "n_sent": tx["sent"],
        "received": rx["received"],
    }


def main(argv: list[str] | None = None) -> int:
    """Command line entry point; exit code 0 when 2389 samples were received."""
    ap = argparse.ArgumentParser(description="Run the ported v1 BPSK TX/RX pair.")
    ap.add_argument("--engine", choices=("auto", "inproc", "fake"), default="auto")
    ap.add_argument("--record", default=None, help="record the run into this directory")
    args = ap.parse_args(argv)
    res = run(args.engine, args.record)
    y = res["received"]
    print(f"engine {res['engine']}: sent {res['n_sent']}, received {y.size} samples")
    if res["engine"] == "fake":
        same = np.array_equal(y.view(np.uint32), np.asarray(res["sent"]).view(np.uint32))
        print(f"identity channel: received == sent bit for bit: {same}")
    return 0 if y.size == 2389 else 1


if __name__ == "__main__":
    sys.exit(main())
