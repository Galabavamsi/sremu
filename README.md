# sremu

**Smart Radio Environment EMUlator**: a software-only, IQ-level radio-environment emulator
for reconfigurable intelligent surfaces (RIS). It is meant to be released as open source
once a licence is agreed (see [Licence](#licence)).

- RIS configuration is a **timestamped runtime command from an external controller**,
  applied at an **exact, logged sample** on every receiver.
- Per-state RIS responses can be **imported from measurements**.
- Wall-clock behaviour is **measured and reported**, never assumed.

sremu is not a digital twin, a ray tracer, a UHD drop-in or a hard-real-time emulator.

An earlier in-house simulator by the authors was used in [Mishra & Kherani 2025a, 2025b];
sremu is an independent new implementation.

- [Mishra & Kherani 2025a] S. Mishra and A. A. Kherani, "On the control of binary RIS
  under user mobility: structural results and implications," IEEE Wireless Commun. Lett.,
  vol. 14, no. 4, pp. 1139-1143, 2025, doi:10.1109/LWC.2025.3535557.
- [Mishra & Kherani 2025b] S. Mishra and A. A. Kherani, "Opportunistic sensing via
  ISAC-capable RIS," in Proc. National Conference on Communications (NCC), 2025,
  doi:10.1109/NCC63735.2025.10983019.

## Status

**Pre-release (0.1.0.dev0). Licence pending. The repository is private.** Interfaces may
still change through contract change requests (`docs/contracts_changelog.md`).

What exists:

- Exact time base, frozen cross-package contracts (`sremu.contracts`).
- Physics: line of sight, the area-tied RIS kernel K1 (`point` and `po_patch` cell models),
  measured-state response tables, flat-earth ground, first-order walls, a single knife
  edge, an additive diffuse term and non-idealities. Their models and validity domains are
  in `docs/model_card.md`.
- Two rendering backends (`numpy_ref` and `numba_cpu`), the engine (lockstep and
  wall-clock-paced modes, timed commands, barriers), a UHD-shaped Python API, a ZMQ
  transport whose sockets bind to localhost by default (`SECURITY.md`), recording (event
  log, SigMF IQ), replay and HTML reports, sweeps and the CLI below.

Verified (the code agrees with an independent reference; this says nothing about
agreement with the real world):

- Physics analytic oracles (PH), runtime oracles (RT) and interface tests (IO), run in the
  CI tier; seeded statistical oracles (ST), run in the nightly tier.
- MATLAB R2026a goldens (G1-G8). These are code-to-code checks only.
- V-WCL (published analytic results of the WCL letter) and V-CFO (receiver LO against
  exact expectations and the Cramer-Rao bound): checks against closed forms.
- V-TANG-1: the K1 kernel against the published model curves of Tang et al. 2022. This is
  a model-to-model check.

Compared with measurements (validation), with limits stated:

- V-TANG-2 compares Tang et al.'s digitized chamber measurements (DIG) with their
  published model, as reproduced by sremu. It describes the published data. It is not a
  claim about sremu's accuracy.
- V-LAB is a pre-registered consistency check against a digitized laboratory map collected
  by the authors (DIG). Its power analysis labelled it underpowered, and it did not
  separate the RIS model from a line-of-sight-only model.

Not validated:

- V-FM is model-based dimensioning. No measurement enters it.
- `sremu.physics.experimental.p526` (ITU-R P.526 smooth earth, PH-12) is experimental. It
  is not validated against ITU-R data, the core never imports it, and it is outside every
  gate. Its regression tests in `tests/experimental` are not part of the CI or nightly
  tiers.
- RIS state transitions use an engineering default until they are measured.
- Wall-clock characterization (benchmarks B1-B4, B6) is still in progress. Until it is
  reported for a stated host, no statement about wall-clock pacing capacity should be made.

Each study keeps its code, committed results and provenance under `validation/<study>/`.

## Install (offline, editable)

With the project virtual environment active (`.venv\Scripts\activate`):

```powershell
uv pip install --offline --no-deps -e .
sremu info
```

Without activating it, add `--python .venv\Scripts\python.exe` to the `uv` command. The
pip equivalent is `pip install -e . --no-build-isolation --no-deps`.

Python >= 3.12. Core dependencies: numpy, scipy, numba, pydantic, pyyaml, pyzmq, typer,
matplotlib. Extras: `dev`, `ml`, `sigmf`, `parquet`, `gpu`.

## Quick start

```powershell
sremu validate scenarios/quickstart.yaml            # schema, guards, derived quantities
sremu run scenarios/quickstart.yaml --out runs/qs   # headless lockstep run
sremu report runs/qs/<run> -o qs.html               # static HTML report of a run
sremu verify --quick                                # closed-form oracle subset
python examples/quickstart/quickstart.py            # Python API with a timed RIS command
```

`sremu replay <run>` replays a run bitwise; it needs a run recorded with IQ. Other commands:
`batch`, `sweep`, `states`, `schema`, `bench` (`sremu --help`). More examples are in
`examples/` (BPSK pair, remote controller, GNU Radio over ZMQ, Gymnasium).

## Tasks

```powershell
python tools/tasks.py ci        # unit, physics, runtime, io and golden tests (<= 8 min, -n 8)
python tools/tasks.py nightly   # statistical, integration and validation_smoke tests
python tools/tasks.py goldens   # Python side of the MATLAB goldens (--regenerate reruns MATLAB)
python tools/tasks.py bench     # benchmarks (see bench/README.md)
python tools/tasks.py lint      # ruff check, ruff format --check, mypy
python tools/tasks.py paper-junit ..\paper\generated\junit\sremu_paper.xml   # junit for the paper
python -m pytest tests/experimental -m experimental   # experimental modules (outside every gate)
```

`paper-junit` runs the CI and nightly tiers and writes `<stem>.meta.json` with
`{sremu_commit, sremu_dirty}` next to the junit file, so the paper pipeline can check which
commit the results came from. It does not run `tests/experimental`.

## Documentation

- `docs/DESIGN.md`: the authoritative design specification.
- `docs/model_card.md`: model, validity domain, evidence and default of each component.
- `CONTRIBUTING.md`: file ownership, units, time-base, RNG and testing rules.
- `SECURITY.md`: the localhost-only control plane.
- `bench/README.md`: the benchmark suites and the hosts they run on.

## Licence

Not yet chosen. See `LICENSE-PENDING.md`. The repository stays private until all authors
and IIT Bhilai agree on a licence. Third-party data under `data/` and `validation/` keep
their own terms, recorded in their provenance files.
