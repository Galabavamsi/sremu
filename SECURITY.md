# Security

sremu is a research emulator. Its network interfaces are designed for a single trusted
host.

## Control and data planes are localhost-only by default

- Every socket (control ROUTER `tcp://127.0.0.1:5600`, telemetry PUB `:5601`, TX ingress
  PULL `:5602`, RX egress PUSH on ephemeral ports) binds to `127.0.0.1`.
- Binding to any other address requires the explicit `--allow-remote` flag.
- **There is no authentication and no encryption.** Anyone who can reach an exposed port
  can send IQ samples and RIS commands. Optional ZMQ CURVE is future work. Do not expose
  sremu ports on untrusted networks; use an SSH tunnel if remote access is needed.

## Hardening by construction

- No web server and no browser-facing endpoint (the v2 path-traversal and cross-site
  WebSocket bug classes are removed by design).
- Control messages never carry filesystem paths.
- Every control message is schema-validated before dispatch; messages are capped at
  1 MiB and 65,536 elements per command, and the per-client command rate is limited
  (default 10 kHz). Data-plane frames with a bad header or size are dropped and counted
  (`BAD_PACKET`); the engine never raises on bad external input.
- Fuzz tests (IO-01) cover malformed frames and JSON.

## Reporting a problem

The repository is private while the licence is pending. Report security issues directly
to the maintainer (Galaba Vamsi, IIT Bhilai) rather than in a public tracker.
