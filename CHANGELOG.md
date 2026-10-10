# Changelog

All notable changes are recorded here. Versions follow PEP 440; the cross-package
contract has its own version (`sremu.contracts.CONTRACT_VERSION`).

## [Unreleased] - 0.1.0.dev0

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
