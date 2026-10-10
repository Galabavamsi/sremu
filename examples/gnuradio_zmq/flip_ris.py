"""Schedule one RIS switch on a served emulator, then start virtual time (E2 helper).

    python examples/gnuradio_zmq/flip_ris.py --endpoint tcp://127.0.0.1:5600 --at 0.02

Sends ``ris.set_config`` (all controls to ``--state``) for RIS time ``--at`` (exact string,
seconds) over the control socket, prints the receipt (``k_req``: the sample at which each
receiver sees the switch, section 4.6) and sends ``sim.resume`` (``--no-resume`` to skip).
"""

from __future__ import annotations

import argparse
import base64
import sys

import numpy as np

from sremu.transport.client import ClientSession


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Schedule a RIS switch over the control socket.")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5600")
    ap.add_argument("--panel", default="ris0")
    ap.add_argument("--at", default="0.02", help="switch time at the RIS, s (exact string)")
    ap.add_argument("--state", type=int, default=1)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    s = ClientSession(args.endpoint, name="flip")
    try:
        caps = s.call("caps.get", panel=args.panel)["caps"][args.panel]
        n = int(caps.get("n_controls") or caps["n_elements"])
        states = base64.b64encode(np.full(n, args.state, "<i2").tobytes()).decode("ascii")
        r = s.request(
            {
                "op": "ris.set_config",
                "panel": args.panel,
                "at": {"t": args.at},
                "config": {"states_b64": states, "dtype": "i2"},
            }
        )
        print(f"ris.set_config at t = {args.at} s: {r}")
        if not args.no_resume:
            s.call("sim.resume")
            print("sim.resume sent")
        return 0 if r.get("ok") else 1
    finally:
        s.close()


if __name__ == "__main__":
    sys.exit(main())
