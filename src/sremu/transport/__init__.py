"""WP8: transport and control plane (DESIGN.md sections 8.3-8.5, 8.10).

* :mod:`.framing` -- the 48-byte ``SRMU`` data-plane frame header and its validation;
* :mod:`.zmq_native` -- native framed ZMQ data plane: TX ingress (PULL), RX egress (PUSH per
  RX stream), client-side sender and receiver, HWM backpressure, BAD_PACKET/SEQ_ERROR
  counting;
* :mod:`.messages` -- control-plane JSON schemas (``schema/*.json``), validation before
  dispatch, size and element caps, codecs for commands, receipts and observations;
* :mod:`.control_server` -- the served emulator: control ROUTER, telemetry PUB, remote
  barrier controllers, localhost-only binds, rate limits, pause/resume;
* :mod:`.client` -- ``ClientSession`` and ``RemoteEngine`` (``sremu.Device.connect``);
* :mod:`.rawreqrep` -- raw REQ/REP mode for srsRAN- and GNU Radio-style peers (lockstep).

Importing this package imports nothing heavy; import the modules directly.
"""
