# Contributing to sremu

`docs/DESIGN.md` is the authoritative specification. Where this file and the spec differ,
the spec wins.

## 0. Install (offline, editable; DESIGN.md D13)

Use the project interpreter `.venv\Scripts\python.exe` (Python 3.12). With the virtual
environment active:

```powershell
uv pip install --offline --no-deps -e .
```

Without activating it, add `--python .venv\Scripts\python.exe`. New packages are installed
only through `uv pip install --python .venv\Scripts\python.exe <pkg>` and reported to the
lead (dependency lists change only through a CCR).

## 1. Ownership (DESIGN.md 2.2 and 14)

Every file has exactly one owning work package (WP). Edit only the files your WP owns plus
your own test files. Shared files are lead-owned (WP0) and change only through a contract
change request (CCR).

| Area | Owner |
|---|---|
| `pyproject.toml`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`, `.gitignore`, `.gitattributes`, `.github/` | WP0 |
| `src/sremu/{__init__,units,timebase,contracts,errors,cli,sysinfo}.py`, `py.typed`, `testing/` | WP0 |
| `tools/tasks.py`, `tests/conftest.py`, `tests/_prop.py`, `tests/unit/test_{timebase*,layering,cli,units,contracts*,packaging,sysinfo,fake_*}.py`, `docs/contracts_changelog.md` | WP0 |
| `src/sremu/commands/<name>.py` | the owner of command `<name>` (DESIGN.md 2.5, D12) |
| `rng.py`, `scenario/`, `record/manifest.py`, `scenarios/{quickstart,lab_like_A}.yaml`, `data/states/` | WP1 |
| `physics/` (core), `states` CLI | WP2 |
| `physics/{ground,walls,diffraction,diffuse}.py`, `physics/experimental/` | WP3 |
| `tests/oracles/`, `tests/golden/`, `tools/matlab/`, `verify` CLI | WP4 |
| `channel/`, `impair/` | WP5 |
| `engine/`, `metrics/`, `legacy/{truncation,next_tick}.py` | WP6 |
| `api/`, `compat.py`, `record/{sigmf,eventlog,replay,report}.py` | WP7 |
| `transport/`, `control/remote.py` | WP8 |
| `control/{controller,builtin}.py`, `ml/`, `analysis/`, `experiments/` | WP9 |
| `validation/`, `data/validation/` | WP10 |
| `bench/`, study scenarios | WP11 |
| `docs/` (except `DESIGN.md`), `CITATION.cff`, `paper/` | WP12 |

**CCRs.** `contracts.py`, `cli.py` and the dependency lists in `pyproject.toml` change only
through a CCR: describe the change and the reason in your WP report; the lead applies it,
bumps `CONTRACT_VERSION` and records it in `docs/contracts_changelog.md` (current version:
1.1).

**Git.** Agents never run git (no commit, branch, checkout, stash or reset). The lead
commits after each stage (DESIGN.md A4).

## 2. Layering (DESIGN.md 2.3; enforced by `tests/unit/test_layering.py`)

```
units, timebase, errors, contracts, sysinfo
  -> rng, scenario -> physics -> channel, impair -> engine, metrics, legacy
  -> api, record, transport, control, ml, compat -> analysis, experiments -> cli, commands
```

`sysinfo` (D12) holds the platform probes (CPU, power plan or governor, GPU, AWS instance,
numba state) shared by `cli.py` and `record/manifest.py`.

- Import only from your own layer or lower layers (also inside `TYPE_CHECKING` blocks).
- `sremu.testing` imports only the foundation layer; production code never imports it.
- Import names from their defining module, not from the lazy `sremu` facade.
- `tests/oracles` must not import `sremu.physics`, `sremu.channel` or `sremu.impair`.
- A new top-level module needs a layer: request it with a CCR.

## 3. Units and the time base (DESIGN.md 3.1, 4.1)

- `sremu.units.C0` is the only speed of light. No other literal for c (299792458, 3e8,
  2.998e8 in any spelling) may appear in any Python file; a token scan enforces it. The one
  exception (D4) is `units.C_LEGACY_V1 = 3.0e8`, defined in `units.py` and used only by
  `physics/legacy.py` to reproduce the v1 emulator.
- SI internally; dB and dBm only at I/O boundaries (`units.*_to_db`, `w_to_dbm`, ...).
- Every dimensional field name carries a unit suffix: `_hz _sps _m _s _dbm _db _deg _rad _mps _w`.
- Rates and every time that maps to a sample are exact `Fraction`s. Scenario values are
  strings (`"23.04e6"`, `"1/200"`); floats are rejected there (`timebase.parse_exact`,
  `Rate.parse`). API floats go through `Fraction(repr(x))` (`timebase.to_exact`).
- Use `to_ticks` (nearest, ties up) for every time-to-sample mapping and `count` (ceil-ceil)
  for samples on [t0, t1). Never `int(f*tau)`.
- IQ is complex64 at every interface; accumulators are complex128.

## 4. Randomness (DESIGN.md 6.1)

- No global RNG anywhere (`src`, `tests`, `tools`, `bench`, `validation`, `examples`): no
  `np.random.seed`, no legacy `np.random.<dist>` or `RandomState`, no stdlib `random`, no
  `.spawn()` (spawn-order dependent). A static test enforces it (ruff NPY002 as well).
- Engine variates come from the keyed Philox streams of `sremu.rng` (WP1):
  key = blake2b(seed, purpose, name), counter = absolute index.
- Controllers and analysis code use `np.random.Generator(np.random.Philox(key=...))` with
  a key derived from `(seed, "ctrl/<name>")`. Never `SeedSequence.spawn`.
- Tests use the `rng` fixture (a Philox generator keyed by the test id) or an explicit
  `np.random.Generator(np.random.Philox(key=...))`.

## 5. Wall clock (I7)

Modules under `physics/`, `channel/`, `impair/` and `engine/{scheduler,events,commands}.py`
must not import `time`, `datetime` or `perf_counter`. Wall clock is allowed only in
`engine/pacer.py`, `metrics/`, `transport/` and `record/`.

## 6. Tests

- Tests are oracles: assert numbers against independent closed forms or the spec's oracle
  tables, with the stated tolerances. Never assert only shapes, and never freeze current
  output as a golden unless the spec says so. Never loosen a spec tolerance to pass; report
  the failure instead.
- Put spec IDs in test names (`test_ph17_...`, `test_rt05_...`, `test_st01_...`, `test_g3_...`).
- Markers: `unit`, `physics`, `runtime`, `io`, `golden` (CI tier); `statistical`,
  `integration` (two-process), `validation_smoke` (nightly, DESIGN.md 2.6);
  `experimental`, `needs_private_data`, `slow`. A test without a tier marker gets the
  marker of its `tests/<dir>` automatically.
- Every `tests/<dir>` is a package (`__init__.py`); import helpers as `tests._prop` or
  `tests.oracles...`.
- `python tools/tasks.py ci` (8 xdist workers, D21) must pass, and stay within 8 minutes,
  before a WP is done. `python tools/tasks.py paper-junit PATH.xml` runs the CI and nightly
  tiers into a junit file plus a `{sremu_commit, sremu_dirty}` sidecar for the paper.
- Property tests may use hypothesis directly; `tests/_prop.py` is the seeded fallback.
- The fakes in `sremu.testing` (`FakePropagation`, `FakeEngine`) let you test against the
  contracts before the real components land.

## 7. CLI commands (DESIGN.md 2.5)

`sremu.cli.REGISTRY` pre-lists every command as `sremu.commands.<name>:<attr>`. To
implement your command, create `src/sremu/commands/<name>.py` defining that attribute:
a plain function (typer builds the options from its signature) or a `typer.Typer` for
subcommands (`states`). Until the module exists the command prints "not yet implemented"
and exits with code 3. Arguments after the command name, including `--help`, are
forwarded to your command.

Deterministic mode (the default; `--no-deterministic` turns it off) sets
`NUMBA_DISABLE_INTEL_SVML=1` and `NUMBA_CACHE_DIR=<default cache>-nosvml` before any command
module, and so numba, is imported (D-4, D15). A command module must therefore not be
imported by `cli.py` itself.

## 8. Package data

Data shipped in wheels must live under `src/sremu/` (`physics/tables/`,
`transport/schema/`, `scenario/schemas/`). Repository-root folders such as `scenarios/`
and `data/states/` are not part of a wheel.

## 9. Style

- `python tools/tasks.py lint` (ruff check, ruff format --check, mypy) must be clean for
  your files. Line length 100. Docstrings state units.
- Use `pathlib`, keep tests fast, and keep MATLAB and long statistical work behind markers.
