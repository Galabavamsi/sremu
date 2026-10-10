# Remote barrier controller

A controller running in its own process steers the RIS of a served emulator over the
control socket (DESIGN.md sections 4.9, 8.5, 8.7). The protocol at every barrier
`t_b = m * T_c`:

```
engine     -> ctrl.barrier {t_b, k_b, m, observation}   (ROUTER -> DEALER, never PUB)
controller -> ris.set_config {at: t_b + latency, ...}   (zero or more)
controller -> ctrl.ack {t_b}
```

The engine computes every receiver up to `t_b + L_min` before the barrier and waits for
the ack, so in lockstep the result does not depend on how long the controller takes or on
process scheduling: the remote run is bit-identical to the same controller class running
in-process (RT-08).

```
python examples/controller_remote/controller_remote.py
python examples/controller_remote/controller_remote.py --endpoint tcp://127.0.0.1:5600
```

* `remote.yaml`: the quickstart scene (SIM geometry, 1-bit 16 x 16 RIS, 1 MS/s tone, 0.1 s)
  with an external controller `probe` (period 10 ms, `min_latency_s` 1 ms, observes
  `rx_power_dbm` and `ris_state`). External controllers are registered before the run
  starts, so barriers begin at `t_b = 0`; the remote side attaches by name with the same
  parameters.
* `controller_remote.py`: starts `python -m sremu.commands.serve remote.yaml` on ephemeral
  loopback ports (or uses `--endpoint`), runs `PowerProbe` (an ordinary controller class:
  alternate the panel between all-0 and all-1, record the observed power) through
  `sremu.control.remote.run_remote`, then runs the same class in-process and checks that
  both saw bit-identical observations.

Expected output: ten barriers; the measured user power alternates between two levels
(about -58.0 and -54.1 dBm in this SIM geometry) as the panel state alternates; the
first barrier reports `nan` (no samples exist before `t = 0`); and
`identical observations: True`.

Any `sremu.contracts.Controller` works the same way, including the built-ins of
`sremu.control.builtin`: `run_remote` hands `setup` the same keyed generator
(`rng.generator(seed, "ctrl", name)`, with the run's seed from `ctrl.register`) and draws
action latencies from the same keyed stream as the engine does in-process.
