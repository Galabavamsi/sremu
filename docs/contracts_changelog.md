# Contract changelog (`sremu.contracts.CONTRACT_VERSION`)

Every change to `src/sremu/contracts.py` (and to the other CCR-controlled interfaces:
`cli.py` and the pyproject dependency lists) is a contract change request (CCR, DESIGN.md
14.3). The lead applies it, bumps `CONTRACT_VERSION` and records it here. Run manifests carry
`contract_version`, so a run can always be matched to the contract it used.

## 1.1 (10 Oct 2026): lead decisions v1.2 after the Stage A review

Applied by WP0 maintenance. Decision ids refer to "Lead decisions v1.2" at the top of
`docs/DESIGN.md`.

### Contracts (`src/sremu/contracts.py`)

- **Version.** `CONTRACT_VERSION = "1.1"`.
- **D3.** `MAX_MESSAGE_BYTES = 2**20` is unchanged and now documented as
  1,048,576 bytes = 1 MiB (IO-04 reads "1 MiB").
- **D9.** `Snapshot.diag` no longer lists the energy budget among its documented keys. The
  RIS energy budget is an on-demand diagnostic (`ris_kernel.energy_budget()`, PH-23,
  `sremu validate --energy`). Documented key form: `"<name>/<panel or link>"`.
- **D10.** New `PathKind.RIS_GAP = 6` for structural (non-configurable) gap reflections
  (`tiles.gap_gamma`).
  - `PathBundle` now rejects `RIS_TILE` (configurable RIS paths only come from `combine()`)
    and any value that is not a `PathKind`.
  - `LinkPaths` now checks tile keys: every `RIS_TILE` path has a tile key >= 0 and every
    other path (`RIS_GAP` included) has -1.
- **D11, protocol.** `PropagationModel` gains:
  - `link_fc_hz(link) -> float`: the carrier of the link's TX stream (the f_c of the
    propagation phase and of `LinkJob.fc_hz`);
  - `panel_geometry(panel) -> PanelGeometry`;
  - `ris_delay_s(panel, rx, t, tile=None)`: with `tile=k`, the delay from render tile k's
    reference point (the point used in `RISCoefficients.tile_delay_s`), for per-tile I3;
  - adopted from `ScenePropagation`: `with_render_tiles(tiles)`,
    `set_node_position(node, t, position_m)` and `set_node_trajectory(node, t, traj)` (timed
    node events; external-trajectory nodes only; events in time order).
  - `runtime_checkable` `isinstance` checks now require all of these names.
- **D11, new type.** `PanelGeometry` (frozen value type): `panel`, `center_m`, `normal`,
  `u_axis`, `extent_u_m`, `extent_v_m`, `n_u`, `n_v`, `pitch_u_m`, `pitch_v_m`, `n_quad` (Q),
  and optional physical tiles `n_tiles_u`, `n_tiles_v`, `gap_m`. It validates unit,
  orthogonal axes, tile divisibility, Q >= N and the D17 extents. Helpers: `from_cells()`,
  `v_axis`, `n_cells`, `q`, `cell_area_m2`, `cell_index()`, `cell_local_m()`,
  `cell_centres_m()`, `physical_tile_of_cell()`. New alias `Vec3`.
  - It carries what `channel.interp.choose_policy` needs about a panel's geometry. The
    per-panel carrier (max `link_fc_hz` of the panel's links) and the row count R
    (`RISCoefficients.n_rows`) are not geometry and are not in it.
  - `channel/interp.py` still defines its own `PanelGeometry` (panel, size_u_m, size_v_m,
    n_u, n_v, fc_hz, n_rows, n_quad). WP5 is asked to re-export or adapt to the contract
    type (see the WP0 report).
- **D11, RIS coefficients.** `RISCoefficients` may carry both `B` and `C`.
  - `B` alone, `C` + `gamma`, or `B` + `C` (with optional `gamma`) are valid.
  - With `B` present, `B` is authoritative for live cells and `separable` is False. A `C`
    alongside is the Gamma-free factor that dead cells (`gamma_dead * C`) and
    coupling-hook deltas need.
  - `gamma` needs `C`, and has S entries when `B` is present.
- **D11, render.** `LinkJob.rot_state` is documented as optional: backends may ignore it and
  callers may pass None.
- **D17.** Geometry conventions are documented in the module docstring, `PanelGeometry`,
  `PanelConfig` and `RISCoefficients`:
  - the cell index is `n = iv*n_u + iu` (`iu` along `u_axis`, `iv` along `normal x u_axis`);
  - `gap_m` is an extra inert gap, so spacing across a tile boundary is pitch + gap;
  - cells are centred on the panel centre;
  - the quadrature area is the pitch area;
  - render tile `k = tv*tiles_u + tu` with `tu = iu*tiles_u // n_u` and
    `tv = iv*tiles_v // n_v`.
- **D20.** The render contract is documented in `RenderFn`, `NoiseFn`, `RenderBackend` and
  `LinkJob`:
  - backends expose module-level `render(job, out)` in `sremu.channel.render_numpy` and
    `sremu.channel.render_numba`;
  - `noise(rx_key, k0, n, sigma)` lives in `sremu.impair.noise`;
  - `sigma**2 = E|w|**2` is the total complex variance;
  - `lo_offset_hz` rotates by `exp(+j 2 pi lo_offset_hz n / f_s)` at the RX rate from
    absolute sample index 0.

### Other WP0 interfaces

- **D4.** `sremu.units.C_LEGACY_V1 = 3.0e8`, for `physics/legacy.py` only.
  `tests/unit/test_layering.py` allows exactly two speed-of-light literals, both in
  `units.py` (`C0` and `C_LEGACY_V1`). It also checks that no module other than
  `physics/legacy.py` references `C_LEGACY_V1`.
- **D12.** New layer-0 module `sremu.sysinfo` holds the platform probes formerly in
  `cli.py`: versions, CPU model, power plan or governor, GPUs, AWS instance (now
  `{type, id}`), and numba SVML, threading and cache state. `cli.py` uses it, and
  `record/manifest.py` should too. The layering test classifies `sysinfo` as layer 0, and
  `src/sremu/commands/` is the CLI layer.
- **D15.** In deterministic mode the CLI sets `NUMBA_CACHE_DIR=<default cache>-nosvml`
  before numba is imported. The default cache is a pre-set `NUMBA_CACHE_DIR`, else numba's
  user-wide cache directory. `sremu info` reports `numba.cache_dir` and
  `numba.env_NUMBA_CACHE_DIR`.
- **D13.** README and CONTRIBUTING document the offline install as
  `uv pip install --offline --no-deps -e .`.
- **D21.** `tools/tasks.py` runs pytest with `-n 8`. A new `paper-junit PATH.xml` task runs
  the CI and nightly markers with `--junitxml` and writes `<stem>.meta.json` with
  `{sremu_commit, sremu_dirty}`, from `git rev-parse HEAD` and `git status --porcelain`
  read before the run.
- **Tests.** `tests/conftest.py` restores `NUMBA_DISABLE_INTEL_SVML` and `NUMBA_CACHE_DIR`
  after every test, because in-process CLI calls change them.

## 1.0 (10 Oct 2026): WP0 baseline

Every section 3.4 type plus the auxiliary types of sections 4 and 8.
