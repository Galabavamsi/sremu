# Quickstart

A 1 kHz tone through `scenarios/quickstart.yaml` (one AP, one user, one 16 x 16 ideal 1-bit
RIS) with one RIS reconfiguration at exactly t = 50 ms, using the UHD-shaped Python API
(DESIGN.md section 8.1).

```
python examples/quickstart/quickstart.py                      # WP6 engine if available
python examples/quickstart/quickstart.py --engine fake        # identity channel, toy RIS gain
python examples/quickstart/quickstart.py --record runs/qs01   # record the run
sremu report runs/qs01 -o qs01.html                           # static HTML report
sremu replay runs/qs01                                        # bitwise replay (needs WP6)
```

What it shows:

* `Device.inproc(scenario, record=DIR, traffic={})` and `get_tx_stream` / `get_rx_stream`
  with `StreamArgs(stream, fc_hz, rate_sps)`; rates are exact (`"1e6"`). The scenario's
  in-process tone generator on `ap.tx0` (for headless `sremu run`) is removed with
  `traffic={}`, so the tone the script sends is what the AP transmits; the override is
  written into the run's `scenario.resolved.yaml`, so `sremu replay` reproduces it.
* A timed burst: `TxMetadata(has_time_spec=True, time_spec=TimeSpec(0, 0.0), ...)`.
* A timed RIS command: `dev.ris("ris0").set_config(states=..., time_spec=...)`; the receipt
  gives the RX sample `k_req` where the receiver sees the switch (`to_ticks(t_sw + tau)`).
* `issue_stream_cmd(StreamCmd("num_done", ...))` and `recv(buf, md, timeout)`.

The `fake` engine's gain (1 with every cell in state 0, 0.5 otherwise) is a toy to make the
switch visible before the WP6 engine is available; it is not a physical model.
