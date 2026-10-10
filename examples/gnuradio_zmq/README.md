# GNU Radio over the raw REQ/REP mode (E2)

The emulator talks to a GNU Radio flowgraph through the stock **ZMQ REP Sink** (the
flowgraph's transmitter) and **ZMQ REQ Source** (its receiver), the way the GNU Radio broker
of the srsRAN ZMQ tutorials does (DESIGN.md sections 8.4 and 11 E-E2):

* the emulator **connects a REQ** to the flowgraph's REP Sink (`tx_port`) and pulls TX
  samples with 4-byte `uint32` item counts;
* the emulator **binds a REP** at `rx_port`; the flowgraph's REQ Source asks it for
  `uint32` item counts of RX samples.

There is no header: timing is implicit in the sample counts, so the mode is **lockstep
only**. The request formats are pinned from GNU Radio 3.10's `req_source_impl.cc` and
`rep_sink_impl.cc` (and srsRAN 4G's `rf_zmq_imp_rx.c` / `rf_zmq_imp_tx.c` for the
`srsran_dummy` format); see `sremu.transport.rawreqrep` for the exact lines. Wording until
E2 has run against real GNU Radio: "srsRAN-style REQ/REP raw-IQ mode", never
"wire-compatible".

## Without GNU Radio (runs anywhere)

```
python examples/gnuradio_zmq/raw_peer_demo.py --at 0.02
```

Serves `gr_link.yaml` (SIM geometry: a transmitter and a receiver around an ideal 1-bit
16 x 16 RIS, 1 MS/s) with one raw peer whose two blocks are stubs implementing the pinned
GNU Radio behaviour. The transmitter streams QPSK (4 samples per symbol); a control client
schedules one RIS switch at RIS time 20 ms before virtual time starts. The demo prints the
sample at which the receiver sees the switch (`k_applied`, lateness 0), the channel gain and
EVM of each RIS state, and checks that the received waveform changes exactly at that
sample. The IO-08 tests do the same for both request formats, sample-exact against an
in-process run.

## With GNU Radio 3.10 (E2; WSL2 Ubuntu or AWS)

```
# 1. the emulator, paused, with the flowgraph as a raw peer
python -m sremu.commands.serve examples/gnuradio_zmq/gr_link.yaml \
    --raw-peer "name=gr,format=gnuradio_count,tx_port=tcp://127.0.0.1:2000,rx_port=tcp://127.0.0.1:2001,tx_stream=gr.tx0,rx_stream=grrx.rx0"
# 2. the flowgraph: REP Sink bound at :2000, REQ Source connected to :2001
python examples/gnuradio_zmq/gr_flowgraph.py --tx tcp://127.0.0.1:2000 --rx tcp://127.0.0.1:2001 --n 50000
# 3. an external controller: one RIS switch at 20 ms, then start virtual time
python examples/gnuradio_zmq/flip_ris.py --endpoint tcp://127.0.0.1:5600 --at 0.02
```

`sremu run S.yaml --serve ...` takes the same options once the CLI routes `--serve` to
`sremu.commands.serve` (contract change request WP8-1).

`gr_flowgraph.py` is written against the GNU Radio 3.10 Python API but has **not been run
in this repository** (no GNU Radio on the development host); E2 and its EVM-per-state
figure are produced on a Linux host. srsRAN 4G (E3) uses `format=srsran_dummy`; its TX side
only answers after its PHY processed RX, so a raw srsRAN pair needs the E3 procedure
(STRETCH, time-boxed).
