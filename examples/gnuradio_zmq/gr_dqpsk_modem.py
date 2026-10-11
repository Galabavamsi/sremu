"""Stock GNU Radio DQPSK modem as an sremu raw REQ/REP peer (Linux / WSL2 only).

    /usr/bin/python3 examples/gnuradio_zmq/gr_dqpsk_modem.py --tx tcp://127.0.0.1:2000 \\
        --rx tcp://127.0.0.1:2001 --n 300000 --out RUN_DIR/gr

Runs under the interpreter that has GNU Radio 3.10 (on Ubuntu 24.04: ``apt install
gnuradio``, i.e. ``/usr/bin/python3`` with numpy 1.26). It imports nothing from sremu, which
needs numpy >= 2.2 and therefore runs in its own interpreter; the two meet only over ZMQ.
Only stock blocks, no knowledge of the emulator:

* TX (its own top block, free-running): ``vector_source_b`` (seeded 2-bit symbols, repeated)
  -> ``diff_encoder_bb(4)`` -> ``chunks_to_symbols_bc`` (QPSK at pi/4 + k pi/2) ->
  ``interp_fir_filter_ccf`` (root-raised cosine, ``--sps`` samples per symbol) ->
  ``zeromq.rep_sink`` bound at ``--tx`` (the emulator's REQ pulls TX with uint32 counts).
* RX (its own top block): ``zeromq.req_source`` connected to ``--rx`` -> ``head(--n)`` ->
  raw IQ file, and -> ``agc2_cc`` -> ``pfb_clock_sync_ccf`` -> ``costas_loop_cc(4)`` ->
  Costas-output file, and -> ``constellation_decoder_cb`` -> ``diff_decoder_bb(4)`` ->
  decision file. Blind timing and carrier recovery.

No finite sink hangs off the TX chain: a ``head`` on a TX side branch stops consuming once
full, GNU Radio then blocks the shared upstream block, the REP Sink starves and a lockstep
emulator cannot produce the RX samples the receiver waits for. The transmitted IQ is
instead regenerated offline from the symbol file and the filter taps written here
(``analyze_ris_ber.py``). The RX top block runs until ``head`` has passed ``--n`` samples
and the receiver chain has drained them (its output files stop growing for ``--settle-s``);
then both top blocks are stopped. The REQ Source cannot end by itself: after the run the
emulator no longer answers its last request, so GNU Radio would keep polling for it.

Outputs (prefix ``--out``): ``_rx.cf32`` (raw IQ received, complex64), ``_costas.cf32``,
``_dec.u8`` (decided data symbols 0..3), ``_tx_symbols.u8`` (the data symbols sent, one
period of the repeating source), ``_progress.csv`` (wall time vs RX samples written) and
``_modem.json`` (parameters, filter taps, versions, counts). Exit codes: 0 done, 3 GNU Radio
missing, 4 no RX progress for ``--stall-s`` or past ``--timeout``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # numpy is imported lazily (exit code 3 without GNU Radio)
    import numpy as np


def tx_symbols(seed: int, n: int) -> np.ndarray:
    """``n`` data symbols in 0..3: 2-bit fields (least significant first) of the raw 64-bit
    output of numpy's PCG64 bit generator seeded with ``seed``. The raw stream of a bit
    generator is stable across numpy versions, unlike ``Generator`` methods, so the 1.26
    of the GNU Radio interpreter and the 2.x of sremu produce the same symbols."""
    import numpy as np

    words = np.random.PCG64(seed).random_raw((n + 31) // 32)
    shifts = np.arange(0, 64, 2, dtype=np.uint64)
    fields = (words[:, None] >> shifts[None, :]) & np.uint64(3)
    return fields.astype(np.uint8).ravel()[:n]


def qpsk_points() -> list[complex]:
    """Constellation of ``chunks_to_symbols_bc``: exp(j (pi/4 + k pi/2)), k = 0..3."""
    return [
        complex(math.cos(math.pi / 4 + k * math.pi / 2), math.sin(math.pi / 4 + k * math.pi / 2))
        for k in range(4)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GNU Radio DQPSK modem as an sremu raw peer.")
    ap.add_argument("--tx", default="tcp://127.0.0.1:2000", help="REP Sink bind endpoint")
    ap.add_argument("--rx", default="tcp://127.0.0.1:2001", help="REQ Source connect endpoint")
    ap.add_argument("--n", type=int, required=True, help="RX samples to receive")
    ap.add_argument("--out", required=True, help="output path prefix")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--nsym", type=int, default=1 << 19, help="period of the symbol source")
    ap.add_argument("--sps", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--timeout", type=float, default=1800.0, help="wall-clock limit, s")
    ap.add_argument("--stall-s", type=float, default=120.0, help="give up without RX progress")
    ap.add_argument(
        "--settle-s",
        type=float,
        default=1.0,
        help="receiver outputs unchanged this long after the last RX sample: done",
    )
    a = ap.parse_args(argv)
    try:
        import numpy as np
        from gnuradio import analog, blocks, digital, filter, gr, zeromq
        from gnuradio.filter import firdes
    except ImportError as exc:
        print(f"GNU Radio is not importable here ({exc}); see README.md", file=sys.stderr)
        return 3

    sps, alpha, nfilts = a.sps, a.alpha, 32
    sym = tx_symbols(a.seed, a.nsym)
    sym.tofile(a.out + "_tx_symbols.u8")
    pts = qpsk_points()
    rrc_tx = list(firdes.root_raised_cosine(sps, sps, 1.0, alpha, 11 * sps))

    tx = gr.top_block("sremu raw peer TX")
    src = blocks.vector_source_b(sym.tolist(), True)
    denc = digital.diff_encoder_bb(4)
    c2s = digital.chunks_to_symbols_bc(pts)
    interp = filter.interp_fir_filter_ccf(sps, rrc_tx)
    # rep_sink(itemsize, vlen, address, timeout_ms, pass_tags, hwm, bind)
    txsink = zeromq.rep_sink(gr.sizeof_gr_complex, 1, a.tx, 100, False, -1, True)
    tx.connect(src, denc, c2s, interp, txsink)

    rx = gr.top_block("sremu raw peer RX")
    rxsrc = zeromq.req_source(gr.sizeof_gr_complex, 1, a.rx, 100, False, -1, False)
    head = blocks.head(gr.sizeof_gr_complex, a.n)
    rawfile = blocks.file_sink(gr.sizeof_gr_complex, a.out + "_rx.cf32", False)
    rawfile.set_unbuffered(True)  # the progress log reads the file size
    agc = analog.agc2_cc(1e-1, 1e-2, 1.0, 1.0)
    rrc_rx = firdes.root_raised_cosine(nfilts, nfilts * sps, 1.0, alpha, 11 * sps * nfilts)
    clk = digital.pfb_clock_sync_ccf(sps, 2 * math.pi / 100, rrc_rx, nfilts, nfilts // 2, 1.5, 1)
    costas = digital.costas_loop_cc(2 * math.pi / 100, 4, False)
    norm = digital.constellation.AMPLITUDE_NORMALIZATION
    const = digital.constellation_calcdist(pts, [], 4, 1, norm)
    dec = digital.constellation_decoder_cb(const.base())
    ddec = digital.diff_decoder_bb(4)
    costasfile = blocks.file_sink(gr.sizeof_gr_complex, a.out + "_costas.cf32", False)
    decfile = blocks.file_sink(1, a.out + "_dec.u8", False)
    for f in (costasfile, decfile):
        f.set_unbuffered(True)  # file sizes show when the receiver chain has drained
    rx.connect(rxsrc, head, rawfile)
    rx.connect(head, agc, clk, costas, dec, ddec, decfile)
    rx.connect(costas, costasfile)

    def sizes() -> tuple[int, int, int]:
        out = []
        for ext in ("_rx.cf32", "_costas.cf32", "_dec.u8"):
            try:
                out.append(os.path.getsize(a.out + ext))
            except OSError:
                out.append(0)
        return out[0] // 8, out[1], out[2]

    t0 = time.monotonic()
    tx.start()
    rx.start()
    rc = 0
    last, t_change, t_rx = sizes(), t0, t0
    t_complete = None
    with open(a.out + "_progress.csv", "w", encoding="utf-8", newline="\n") as log:
        log.write("wall_s,rx_samples\n")
        while True:
            time.sleep(0.05)
            now, cur = time.monotonic(), sizes()
            log.write(f"{now - t0:.3f},{cur[0]}\n")
            if cur != last:
                t_change = now
                if cur[0] != last[0]:
                    t_rx = now
                last = cur
            if cur[0] >= a.n:
                t_complete = t_complete or now
                if now - t_change >= a.settle_s:
                    break
            elif now - t_rx > a.stall_s or now - t0 > a.timeout:
                print(
                    f"no RX progress for {now - t_rx:.0f} s (or past --timeout); stopping",
                    file=sys.stderr,
                )
                rc = 4
                break
    wall = (t_complete or time.monotonic()) - t0
    rx.stop()
    rx.wait()
    tx.stop()
    tx.wait()

    def size(ext: str, item: int) -> int:
        return os.path.getsize(a.out + ext) // item

    info = {
        "peer": "examples/gnuradio_zmq/gr_dqpsk_modem.py",
        "gnuradio_version": gr.version(),
        "numpy_version": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tx_endpoint": a.tx,
        "rx_endpoint": a.rx,
        "n_requested": a.n,
        "rx_samples": size("_rx.cf32", 8),
        "costas_samples": size("_costas.cf32", 8),
        "decisions": size("_dec.u8", 1),
        "seed": a.seed,
        "symbol_generator": "numpy PCG64(seed).random_raw, 2-bit fields LSB first",
        "nsym": a.nsym,
        "sps": sps,
        "alpha": alpha,
        "rrc_tx_taps": [float(t) for t in rrc_tx],
        "constellation": [[p.real, p.imag] for p in pts],
        "rx_chain": {
            "agc2_cc": [1e-1, 1e-2, 1.0, 1.0],
            "pfb_clock_sync_ccf": {"nfilts": nfilts, "loop_bw": 2 * math.pi / 100, "max_dev": 1.5},
            "costas_loop_cc": {"loop_bw": 2 * math.pi / 100, "order": 4},
        },
        "wall_s": round(wall, 3),
        "exit": rc,
    }
    with open(a.out + "_modem.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(info, fh, indent=1)
        fh.write("\n")
    print(f"done rx_samples={info['rx_samples']} decisions={info['decisions']} wall={wall:.2f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
