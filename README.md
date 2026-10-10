# sremu

**Smart Radio Environment EMUlator**: an open-source, software-only, IQ-level
radio-environment emulator for reconfigurable intelligent surfaces (RIS).

- RIS configuration is a **timestamped runtime command from an external controller**,
  applied at an **exact, logged sample** on every receiver.
- Per-state RIS responses can be **imported from measurements**.
- Wall-clock behaviour is **measured and reported**, never assumed.

sremu is not a digital twin, a ray tracer, a UHD drop-in or a hard-real-time emulator.

**Status: pre-release (0.1.0.dev0).** The foundations (exact time base, frozen
cross-package contracts, CLI, test doubles) are in place; the physics core, render kernels,
engine, API and transport are being built. Interfaces may still change through contract
change requests. Nothing here is a validated result yet.

## Lineage

sremu is the third generation of the IIT Bhilai RIS emulator:

| Generation | Reference | Notes |
|---|---|---|
| v1 | legacy repo, tag `v1-legacy` (03697a6) | Original emulator used by the ANTS manuscript and the WCL letter |
| v2 | legacy repo, tag `v2-legacy` (8abd671) | Rework with a web dashboard; audited in 2026 |
| sremu | this repository | New design: exact time base, K1 RIS kernel, measured-state tables, timed commands |

The audits of v1 and v2 and the design decisions that replace their behaviour are in
`docs/DESIGN.md` (sections 1, 4, 5.14 and 15).

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

## Tasks

```powershell
python tools/tasks.py ci        # unit, physics, runtime, io and golden tests (<= 8 min, -n 8)
python tools/tasks.py nightly   # statistical and integration tests
python tools/tasks.py lint      # ruff and mypy
python tools/tasks.py paper-junit ..\paper\data\sremu-junit.xml   # CI + nightly junit for the paper
```

`paper-junit` also writes `<stem>.meta.json` with `{sremu_commit, sremu_dirty}` next to the
junit file, so the paper pipeline can check which commit the results came from.

## Documentation

- `docs/DESIGN.md`: the authoritative design specification.
- `CONTRIBUTING.md`: file ownership, units, time-base, RNG and testing rules.
- `SECURITY.md`: the localhost-only control plane.

## Licence

Not yet chosen. See `LICENSE-PENDING.md`. The repository stays private until all authors
and IIT Bhilai agree on a licence.
