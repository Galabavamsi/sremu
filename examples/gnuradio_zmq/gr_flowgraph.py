"""GNU Radio 3.10 peer for the raw REQ/REP mode (E2; needs GNU Radio with gr-zeromq).

    python examples/gnuradio_zmq/gr_flowgraph.py --tx tcp://127.0.0.1:2000 \\
        --rx tcp://127.0.0.1:2001 --n 50000 --out rx.cf32

Transmit path: random QPSK symbols, 4 samples per symbol (rectangular), amplitude 0.5,
into a stock **ZMQ REP Sink** bound at ``--tx`` (the emulator's REQ pulls the samples with
4-byte ``uint32`` item counts). Receive path: a stock **ZMQ REQ Source** connected to
``--rx`` (the emulator's REP answers its ``uint32`` item-count requests) into a file sink
of ``--n`` complex64 samples.

Status: a minimal peer for the GNU Radio 3.10 Python API (``zeromq.rep_sink(itemsize, vlen,
address, timeout, pass_tags, hwm, bind)``, ``zeromq.req_source(...)``). It imports
``sremu.rng`` for the keyed symbols, so GNU Radio's interpreter must be able to import sremu.
The link-quality demonstration uses ``gr_dqpsk_modem.py`` instead, which imports nothing from
sremu and is tested against GNU Radio 3.10.9.2 (``tests/integration/test_gnuradio_raw.py``;
README.md). Linux / WSL2 only.
"""

from __future__ import annotations

import argparse
import sys


def build(tx: str, rx: str, n: int, out: str, seed: int = 1) -> object:
    """The flowgraph (a ``gr.top_block``)."""
    import numpy as np
    from gnuradio import blocks, gr, zeromq

    from sremu import rng

    class Peer(gr.top_block):  # type: ignore[misc]
        def __init__(self) -> None:
            gr.top_block.__init__(self, "sremu E2 peer")
            # keyed symbols (section 6.1), the same stream as raw_peer_demo.qpsk
            sym = rng.generator(seed, "trial", "examples/gnuradio_zmq").integers(0, 4, 1 << 16)
            pts = [
                complex(np.cos(np.pi / 4 + k * np.pi / 2), np.sin(np.pi / 4 + k * np.pi / 2))
                for k in range(4)
            ]
            src = blocks.vector_source_c([0.5 * pts[int(k)] for k in sym], True)
            sps = blocks.repeat(gr.sizeof_gr_complex, 4)
            sink = zeromq.rep_sink(gr.sizeof_gr_complex, 1, tx, 100, False, -1, True)
            source = zeromq.req_source(gr.sizeof_gr_complex, 1, rx, 100, False, -1, False)
            head = blocks.head(gr.sizeof_gr_complex, int(n))
            fsink = blocks.file_sink(gr.sizeof_gr_complex, out, False)
            self.connect(src, sps, sink)
            self.connect(source, head, fsink)

    return Peer()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GNU Radio 3.10 raw REQ/REP peer (E2).")
    ap.add_argument("--tx", default="tcp://127.0.0.1:2000")
    ap.add_argument("--rx", default="tcp://127.0.0.1:2001")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--out", default="rx.cf32")
    args = ap.parse_args(argv)
    try:
        tb = build(args.tx, args.rx, args.n, args.out)
    except ImportError:
        print("GNU Radio (gnuradio, gr-zeromq) is not installed; see README.md", file=sys.stderr)
        return 3
    tb.run()  # type: ignore[attr-defined]
    return 0


if __name__ == "__main__":
    sys.exit(main())
