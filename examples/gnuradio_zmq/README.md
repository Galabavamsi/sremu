# GNU Radio through the raw REQ/REP mode

The emulator talks to a GNU Radio flowgraph through the stock **ZMQ REP Sink** (the
flowgraph's transmitter) and **ZMQ REQ Source** (its receiver), the way the GNU Radio broker
of the srsRAN ZMQ tutorials does (DESIGN.md sections 8.4 and 11 E-E):

* the emulator **connects a REQ** to the flowgraph's REP Sink (`tx_port`) and pulls TX
  samples with 4-byte `uint32` item counts (8 pages per request by default);
* the emulator **binds a REP** at `rx_port`; the flowgraph's REQ Source asks it for
  `uint32` item counts of RX samples.

There is no header: timing is implicit in the sample counts, so the mode is **lockstep
only**: virtual time advances as the flowgraph exchanges samples, never with a wall clock.
The request formats are pinned from GNU Radio 3.10's `req_source_impl.cc` and
`rep_sink_impl.cc` (and srsRAN 4G's `rf_zmq_imp_rx.c` / `rf_zmq_imp_tx.c` for the
`srsran_dummy` format; see `sremu.transport.rawreqrep`). Wording: "srsRAN-style REQ/REP
raw-IQ mode", never "wire-compatible".

| File | Runs in | What |
|---|---|---|
| `raw_peer_demo.py` | sremu | stub peer with GNU Radio's pinned behaviour (runs anywhere) |
| `gr_link.yaml`, `gr_flowgraph.py`, `flip_ris.py` | sremu / GNU Radio | minimal QPSK peer and one RIS switch |
| `gr_ris_ber.yaml` | - | scene of the link-quality demonstration (RIS hop dominant) |
| `gr_dqpsk_modem.py` | **GNU Radio** | DQPSK modem of stock GNU Radio blocks as the raw peer |
| `ris_configs.py` | sremu | focusing, focusing with a fraction of cells inverted, uniform, random |
| `run_ris_ber.py` | sremu | one fresh served run + fresh modem per configuration; the switching run |
| `analyze_ris_ber.py` | sremu | SER, uncoded BLER, MER, LS gain/SNR, change points, hashes, RTF |
| `ee_ubuntu.sh` | shell | Ubuntu 24.04 / AWS: install, run the paper's E-E, export |

## Without GNU Radio (runs anywhere)

```
python examples/gnuradio_zmq/raw_peer_demo.py --at 0.02
```

Serves `gr_link.yaml` (SIM geometry: a transmitter and a receiver around an ideal 1-bit
16 x 16 RIS, 1 MS/s) with one raw peer whose two blocks are stubs implementing the pinned
GNU Radio behaviour. A control client schedules one RIS switch at RIS time 20 ms before
virtual time starts; the demo prints the sample at which the receiver sees the switch
(`k_applied`, lateness 0) and checks that the received waveform changes exactly there.

## With GNU Radio (Linux or WSL2 only)

GNU Radio and sremu run in **separate interpreters**: sremu needs numpy >= 2.2, while
Ubuntu 24.04's `gnuradio` package (3.10.9.2) is bound to the system Python's numpy 1.26.
They meet only over loopback ZMQ, so the ABI split is harmless. Never install sremu into
the system interpreter or GNU Radio into sremu's venv.

```
# 1. Ubuntu 24.04 (WSL2, native or AWS): GNU Radio from the distribution
sudo apt-get update && sudo apt-get install -y gnuradio python3-venv
/usr/bin/python3 -c "from gnuradio import gr, zeromq, digital; print(gr.version())"   # 3.10.9.2

# 2. sremu in its own venv, from the repository root
python3 -m venv ~/sremu-venv && ~/sremu-venv/bin/pip install -e ".[dev]"

# 3. link quality vs RIS configuration (10 fresh runs of 0.3 s) and the switching run
export SREMU_GR_PYTHON=/usr/bin/python3          # the interpreter that has GNU Radio
~/sremu-venv/bin/python examples/gnuradio_zmq/run_ris_ber.py --out runs/gr --configs all --switching

# 4. the gated tests (skipped with a reason where GNU Radio is missing)
~/sremu-venv/bin/python -m pytest tests/integration/test_gnuradio_raw.py -rs

# 5. the paper's E-E (2 independent repetitions + throughput) and its export
~/sremu-venv/bin/python -m experiments.ee_gnuradio run --root runs/ee --reps 2 --throughput
~/sremu-venv/bin/python -m experiments.ee_gnuradio export --root runs/ee --dest runs/ee_export
# or, on a fresh Ubuntu 24.04 host (e.g. AWS c7i):  OUT=/opt/job/out bash examples/gnuradio_zmq/ee_ubuntu.sh --setup
```

Each run of `run_ris_ber.py` starts `python -m sremu.commands.serve gr_ris_ber.yaml
--exit-when-done --out ... --raw-peer "name=gr,format=gnuradio_count,..."` (paused), then
`gr_dqpsk_modem.py` under `$SREMU_GR_PYTHON`, then schedules the RIS configuration over the
control socket and sends `sim.resume`. Static runs hold one configuration for 0.3 s; the
switching run holds uniform, focusing, random and focusing again for 0.4 s each, with the
three switches scheduled at exact RIS times before virtual time starts. Every run directory
holds GNU Radio's files (`gr_rx.cf32` received IQ, `gr_costas.cf32`, `gr_dec.u8` decisions,
`gr_tx_symbols.u8`, `gr_modem.json`), the emulator's run directory (`rec/`: manifest, events
with every applied command, metrics, the RX stream as SigMF), `run.json` and `analysis.json`.

`analyze_ris_ber.py` checks that GNU Radio received every sample of the run and that its
received IQ equals, bit for bit, the RX stream the emulator recorded itself; that every
manifest counter is zero and every command was applied with lateness 0; and, in the
switching run, that the maximum-likelihood change point of the received IQ lies at each
command's `k_applied`. It reports per configuration the SER of GNU Radio's decisions, the
**uncoded 200-bit block rate** (consecutive 100-symbol windows of the decision stream with an
error: no coding, no CRC, a re-expression of the SER stream, not a packet error rate), the
decision-directed MER, the LS gain and the SNR estimated from the received IQ, and the sha256
of the received IQ and of the decisions (equal across independent runs).

### Things that matter

* **No finite sink on the TX chain.** A `blocks.head` on a side branch of the TX chain stops
  consuming once full; GNU Radio then blocks the shared upstream block, the REP Sink starves,
  and lockstep cannot compute the RX samples the receiver waits for (a deadlock just short of
  the head length). The modem logs no TX; the analysis regenerates it from the symbol file
  and the filter taps (equal to GNU Radio's TX to float32 rounding; tested).
* **Read to the end of the run.** `--exit-when-done` needs the raw peer to read its RX stream
  to the end of the run: the engine cannot finish while a raw consumer stops reading
  (lockstep backpressure), so the server would never exit. At the end the peer still
  receives what the engine queued for it (IO-08). The modem's REQ Source cannot end by itself
  after the run (its last request is never answered), so the modem stops its flowgraphs once
  the receiver chain has drained.
* **Request size.** `request_items` defaults to 8 pages (>= 4096 samples) for
  `gnuradio_count`; one request per 0.5 ms page halved the served throughput at 1 MS/s.
* **Fresh receiver per configuration.** After a long link loss (the random configuration) the
  stock receiver's clock and carrier loops can stay in a false lock when the link returns;
  the static sweep therefore uses one fresh modem per configuration, and the switching run
  shows timing and gain steps.
* **Array factor.** `ris_configs.array_factor_db` (isotropic point cells) against the gain the
  receiver sees is an internal consistency check of the emulator against a simplified model,
  never a measured RIS gain or an accuracy statement.
* **Throughput.** Lockstep runs are virtual time: the real-time factor (virtual seconds per
  wall second of the served engine, which includes waiting for GNU Radio; `metrics.jsonl`,
  cross-checked with GNU Radio's RX progress) says how fast the loop advanced on that host.

## srsRAN (not run)

srsRAN 4G is not packaged for Ubuntu 24.04 and was not run. Its ZMQ RF driver pads its TX
stream only up to the end of the RX block it is about to read, while lockstep computes RX
samples only once the reaching TX is committed one fractional-delay look-ahead further; an
srsRAN pair would therefore stall at its first subframe without a per-peer TX delay of at
least that look-ahead and partial RX replies (`format=srsran_dummy` exists, these two
changes do not).
