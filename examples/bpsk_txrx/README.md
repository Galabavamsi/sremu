# BPSK TX/RX pair (ported from v1)

The v1 scripts `ris_sim/external_traffic/usrp_tx.py` and `usrp_rx.py` ported to sremu
(DESIGN.md section 8.2, acceptance IO-07). The waveform is unchanged: 5880 S/s, a
200-symbol header of `1+1j`, 38 BPSK symbols at `+-0.1(1+j)`, 10 samples per symbol with a
rectangular pulse, 2389 samples in total, carrier `2.4e9 - 1800 - 800 - 450` Hz.

v1 exchanged requests through `config/nodes.json` (`send_to_simulator`,
`recieve_from_simulator`). The port calls the ANTS-named aliases
`sremu.compat.Send_to_sim` and `sremu.compat.Receive_from_sim` on the current device; the
v1 names still exist and emit a `DeprecationWarning`.

```
python examples/bpsk_txrx/run_pair.py --engine fake          # identity channel, bitwise check
python examples/bpsk_txrx/run_pair.py --engine inproc        # WP6 engine with bpsk.yaml
python examples/bpsk_txrx/run_pair.py --record runs/bpsk01   # also record SigMF + events
```

`run_pair.py` installs one in-process `Device` with `sremu.compat.use_device` and runs the
two scripts unmodified, transmitter first. In lockstep a receiver waits for its linked
transmitters; `Send_to_sim` ends its burst, so the stream is committed (silent) afterwards.
Each script can also run on its own against a served emulator by setting
`SREMU_CONNECT=tcp://127.0.0.1:5600` (needs the WP8 transport).

The `fake` engine is the identity-channel `sremu.testing.FakeEngine`: the receiver gets the
transmitted samples bit for bit, which is what the IO-07 test checks. With the real engine
the received waveform includes free-space path loss, propagation delay and thermal noise.
