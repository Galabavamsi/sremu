# Changelog

All notable changes are recorded here. Versions follow PEP 440; the cross-package
contract has its own version (`sremu.contracts.CONTRACT_VERSION`).

## [Unreleased] - 0.1.0.dev0

### Changed (transport and served control, 2026-10-11)

- Lockstep RX egress sends up to 8 pages per frame (`RX_BATCH_PAGES`; an explicit
  `rx_frame_samples` wins) and flushes at once when the engine is about to wait, on pause,
  at the end of the run, or after `ServerConfig.rx_flush_s` (10 ms). Paced mode sends one page
  per frame. Sample values, indices and flags do not depend on the framing. HWMs are counted
  in frames; `stream.open` returns `frame_samples`.
- Latency stamps (control-server receipt, frame `wall_ns`, ingress/egress timing) use the
  engine clock (`perf_counter_ns`).
- The paced engine's wait yields the CPU and the GIL until 0.1 ms before a page release.
- `RtfMeter.report().aggregate_sps` counts only the samples of the measured span.
- Raw REQ/REP peers: `RawPeer.stop()` waits (bounded) for the peer to read the RX samples
  queued at the end of the run; the default `gnuradio_count` request size is at least 8 pages.
- `examples/gnuradio_zmq/`: DQPSK link through stock GNU Radio 3.10 against the raw mode, with
  RIS configurations, a scheduled in-run switch and an analysis script (Linux/WSL).

### Changed (WP0 maintenance: lead decisions v1.2, **CONTRACT_VERSION 1.1**)

Full list in `docs/contracts_changelog.md`.

- `contracts`:
  - `PathKind.RIS_GAP = 6`; `PathBundle` rejects `RIS_TILE`, and `LinkPaths` enforces tile
    keys (>= 0 exactly for `RIS_TILE`) (D10).
  - `PropagationModel` gains `link_fc_hz`, `panel_geometry`, `ris_delay_s(..., tile=None)`,
    `with_render_tiles`, `set_node_position` and `set_node_trajectory` (D11).
  - New frozen `PanelGeometry` documents and checks the D17 cell layout (D11, D17).
  - `RISCoefficients` may carry both `B` and `C` (D11).
  - `LinkJob.rot_state` is optional (D11).
  - The energy budget is no longer a `Snapshot.diag` key (D9).
  - The render and noise contract is documented (D20), and `MAX_MESSAGE_BYTES` is documented
    as 1 MiB (D3).
- `units.C_LEGACY_V1 = 3.0e8`, for `physics/legacy.py` only. The literal scan allows it only
  in `units.py` (D4).
- New layer-0 `sremu.sysinfo` with the platform probes, used by `cli.py` (D12).
- CLI deterministic mode also sets `NUMBA_CACHE_DIR=<default cache>-nosvml` before numba is
  imported (D15).
- `testing.FakePropagation` implements the 1.1 protocol, including external nodes with timed
  position and trajectory events and exact piecewise motion bounds.
- `tools/tasks.py`: pytest runs with `-n 8`, and the new `paper-junit` task writes a junit
  file plus a `{sremu_commit, sremu_dirty}` sidecar (D21).
- README and CONTRIBUTING: offline install `uv pip install --offline --no-deps -e .` (D13).
- `.gitattributes`: explicit `*.csv` / `*.json` `text eol=lf` rules (DESIGN 2.2), placed before
  the byte-exact `-text` rules for hash-pinned data.

### Added (WP0, foundations and frozen contracts)

- Package skeleton (src layout) with one subpackage per work package (DESIGN.md 2.2) and a
  `sremu.commands` package for lazily registered CLI commands.
- `sremu.units`: `C0` (the single speed-of-light constant), `K_B`, `T0`, dB/dBm helpers,
  thermal noise power.
- `sremu.timebase`: exact `Rate`, `TimeSpec`, `to_ticks` (nearest, ties up), `from_ticks`,
  `ceil_ticks`, `count` (ceil-ceil on half-open intervals) and the page grid.
- `sremu.errors`: exception hierarchy and UHD-style `ErrorCode` / `AsyncCode`.
- `sremu.contracts`: **CONTRACT_VERSION 1.0** baseline with every section 3.4 type
  (paths, RIS coefficients, snapshots, configurations, controllers, commands, streams,
  metadata, render contract) and the auxiliary types of sections 4 and 8.
- `sremu.cli`: typer app with a lazy command registry and the `info` command.
- `sremu.testing`: `FakePropagation` (analytic LOS, constant-velocity nodes, simplified
  RIS panels), `FakeEngine` (identity channel, exact timing, commands and barriers) and
  scene builders.
- `tools/tasks.py` (ci, nightly, goldens, bench, paper-data, lint), pytest configuration
  with tier markers, layering / I7 / speed-of-light / global-RNG static checks, RT-01 and
  RT-02 tests.

### Changed (WP0 adversarial review)

- `cli`: deterministic mode now forces `NUMBA_DISABLE_INTEL_SVML=1` even over a pre-set `0`
  (D-4); a fresh-process test checks `numba.config.DISABLE_INTEL_SVML`.
- `timebase.to_exact`: NumPy `float32`/`float16`/`longdouble` API values use the shortest
  decimal of their own precision (R5); `np.float32(0.9)` s at 23.04 MS/s is now sample
  20736000, not 20735999.
- `contracts.check_link` rejects float rates; `CsiView.evaluate` rejects state vectors of the
  wrong length or with out-of-range states instead of silently dropping cells.
- `testing.FakeEngine`: element-subset commands compose in (t, seq) order rather than arrival
  order; superseded commands never take effect; every rejected command gets a seq and a
  `rejected` CommandRecord (I6); a barrier-registered (remote) source acting before
  `t_b + L_min` gets `LOOKAHEAD_VIOLATION` instead of a silent late application.
- Tiers: `validation_smoke` marker registered and selected by `tasks.py nightly` (2.6).
- Static checks: global-RNG ban extended from `src/` to every Python tree and to `.spawn()`;
  speed-of-light scan catches more spellings; fakes scanned for wall-clock imports.
- New tests: packaging/tier configuration vs DESIGN 2.1/2.4/2.6, lazy facade, float32 R5,
  FakeEngine barrier property (random rational rates), T9 RIS-path Doppler 325.192 Hz on the
  fake panel; `_prop` fallback tests moved out of the hypothesis-only module.
