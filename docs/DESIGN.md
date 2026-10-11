# sremu: authoritative design specification

**Version** 1.0, 10 Oct 2026. **Product** `sremu` (Smart Radio Environment EMUlator), new git repo `D:\projects\ris\sremu`; paper repo `D:\projects\ris\paper`. **For** Galaba Vamsi (IIT Bhilai) and the coding agents that will build it. **Status: design only. No sremu code exists yet.** What exists: three design inputs, research notes, digitized data, MATLAB cross-checks, scratch scripts and micro-benchmarks (Appendix A).

This is the single spec the agents implement. It merges `physics_validation.md`, `runtime_interfaces.md` and `product_paper.md` under the two judges' verdicts. §15 lists every contradiction between them and how it was resolved.

### Lead amendments v1.1 (10 Oct 2026, after review; these override the text below)

- **A1. No generative-AI disclosure section in the paper.** The author has deferred that decision until he consults Dr. Kherani. Remove it from §13.3's appendix row and from H-8; `check_arxiv.py` must not require it. Revisit at journal submission (IEEE policy requires disclosure in the acknowledgments).
- **A2. Name.** `sremu` is free on PyPI (checked 10 Oct 2026). GitHub availability is still open (H-7).
- **A3. Staged build (usage-limited).** The work runs in stages with a lead checkpoint after each:
  - Stage A: WP0, then WP1, WP2, WP4, WP5 and the WP12 skeleton in parallel.
  - Stage B: WP3, WP6, WP7, WP8, WP9, then GATE-PHY and GATE-RT.
  - Stage C: WP10, WP11, the AWS window, and WP12 figures and writing.
- **A4. Shared working tree, no agent git operations.** Agents never run `git` (no commit, branch, checkout, stash or reset). They edit only their owned files (§2.2) plus their own tests; the lead commits after each stage. This replaces the per-WP branches of §14.3 for release 1.
- **A5. Interpreter.** Always use `D:\projects\ris\sremu\.venv\Scripts\python.exe` (Python 3.12, numpy 2.5.3, scipy 1.18.1, numba 0.68.0, pydantic 2.14, pyzmq 27.2, hypothesis 6.168 **installed**, ruff, mypy, pytest-xdist). hypothesis is available, so property tests may use it directly; `tests/_prop.py` remains the fallback for environments without it.
- **A6. Reproduce figures only from code.** Agents must not copy numbers from this spec into code as expected values unless the spec labels them as oracle values (PH/RT/ST/G tables). [PILOT] and [DERIVED] numbers are sizing aids, not test targets, unless a test row cites them.

### Lead decisions v1.2 (10 Oct 2026, after Stage A review; override the text below)

- **D1 PH-18 budget split.** The chord (curvature) and tile-residual rules each get half the budget: `ε_chord = ε_tile = ε_rel/2`, so `T_curv = √(8(ε_rel/2) r_min/(k v⊥²))` and `Δφ_res ≤ √(8·ε_rel/2) = 0.0632` rad. PH-18 keeps the total limit 1e-3; the two strict xfails must pass.
- **D2 Kirchhoff obliquity.** Stays a non-default option, emits a `ValidityWarning` ("not energy-bounded for anomalous reflection; measured excess up to about +0.9 dB"), and is documented in the model card. PH-23 applies to the default obliquity √(cos θi cos θs); the xfail becomes a test that asserts the warning and records the excess.
- **D3 Message cap** = 2^20 bytes (1 MiB). IO-04 reads "1 MiB".
- **D4 Legacy v1 constant.** `units.C_LEGACY_V1 = 3.0e8` (documented; used only by `physics/legacy.py`); the literal grep test allows it in `units.py` only. `legacy_v1` teeth: +24.39 ± 0.05 dB vs K1 in the v1 default geometry.
- **D5 PH-04 near anomalous** is evaluated with Richardson-converged `po_patch`; oracle −88.128 dB (Δ = λ/40) ± 0.10 dB. Single-level values are not asserted.
- **D6 PH-26** regression value: 8.11 dB point-minus-patch at 45° retro with Δ ≤ λ/5 (3×3 sub-cells), ±0.1 dB; the Δ→0 sinc limit 8.92 dB is documented.
- **D6b PH-26 at 2 m** (after Stage B gates): the 8.11 dB oracle is checked in the far field (200 m), where it equals the closed-form 3-point sub-aperture factor; the PH-26 row's 2 m geometry is pinned as a regression value 7.68 ± 0.1 dB, whose correctness is established by the per-cell local 3-point prediction test (≤ 0.05 dB).
- **D24 Fresh-attempt framing (author decision, 11 Oct 2026; overrides §1.6 rule 4's legacy-comparison provision, §12.2 B5, §13.2–13.3 lineage/before-after items, and every 'legacy'/'v1'/'v2' mention in paper text).** sremu is presented as a NEW emulator, judged on its own evidence. The paper contains NO v1/v2 comparisons, NO 'before/after' framing and NO speed-up vs v1. B5 (v1 vs sremu same-host) is dropped. Kept: the physical lab measurement of ANTS Fig. 4(b) as 'a laboratory measurement collected by the authors' (V-LAB), third-party published data (Tang 2022) and published analytic results (WCL letter). One transparency sentence is allowed: 'An earlier in-house simulator by the authors was used in [Mishra & Kherani WCL 2025, NCC 2025]; sremu is an independent new implementation.' Baselines are renamed generically: LegacyNextTickAdapter -> 'tick-based untimed semantics' in E-B2; the CFO truncation pitfall may be shown as a generic design lesson (integer samples per tick -> 6.9 % frequency scaling) without naming v1. Internal legacy teeth tests stay in the test suite but are not featured in the paper. Benchmarks use the fastest suitable hardware (AWS c7i family for quiet Linux timing; large instances for sweeps), with the hardware stated; no CPU-generation matching.
- **D25 NumPy elision and bitwise determinism (11 Oct 2026).** NumPy temporary elision (glibc Linux and macOS, temporaries >= 256 KiB, i.e. 16384 complex128 samples) can swap the operands of `a * b`, and NumPy's FMA-based complex multiply is not bitwise commutative (the imaginary part differs in ~1/3 of elements). Every product on the bitwise path must therefore fix operand order explicitly (`np.multiply(y, phasor)`, or non-owning phasor views). Regression tests: tests/runtime/test_elision_determinism.py (structural tests fail on every OS; behavioural tests fail on Linux).
- **D7 PH-17 teeth** uses the T9 window (0.2 s at 10 kS/s): −148.3 ± 1 Hz.
- **D8 PH-16.** Phases are reduced modulo 2π before Γ is formed; the 1e-15|h| tolerance applies.
- **D9 Energy diagnostic** is not part of `Snapshot.diag`. `ris_kernel.energy_budget()` is an on-demand diagnostic (PH-23, `sremu validate --energy`).
- **D10 `PathKind.RIS_GAP = 6`** for structural gap reflections. `RIS_TILE` paths always carry tile ≥ 0.
- **D11 PropagationModel protocol additions** (CONTRACT_VERSION 1.1): `link_fc_hz(link)`, `panel_geometry(panel)`, `ris_delay_s(panel, rx, t, tile=None)`; ScenePropagation's `with_render_tiles`, `set_node_position`, `set_node_trajectory` are adopted. `LinkJob.rot_state` is optional and may be ignored by backends. `RISCoefficients` may carry both `B` and `C`.
- **D12 CLI.** `src/sremu/commands/` is accepted (cli layer). Platform probes move to layer-0 `src/sremu/sysinfo.py`, shared by `cli.py` and `record/manifest.py`.
- **D13 Offline install** documented as `uv pip install --offline --no-deps -e .`.
- **D14 RNG.** Every keyed generator (mobility legs, failure maps, controllers) uses `sremu.rng` derivation (§6.1). WP2's private blake2b(digest_size=8) helper is replaced.
- **D15 Numba cache and SVML.** In deterministic mode the CLI sets `NUMBA_CACHE_DIR` to a separate `-nosvml` directory before numba is imported.
- **D16** `ResponseTable.from_state_library` accepts `angle_mapping: None` and per-entry σ values.
- **D17 Geometry conventions** (document in contracts and §9): `PanelConfig.state` index `n = iv·n_u + iu` (iu along `u_axis`, iv along `n̂ × û`); `tiles.gap_m` is an extra inert gap, so spacing across a tile boundary is pitch + gap; cells are centred on `pose.center_m`; quadrature area is the pitch area.
- **D18** Module-level `sremu.physics.propagation.from_scenario(scenario)` builds the full model and attaches ground/walls/diffraction/diffuse when the scenario enables them.
- **D19 Diffuse API.** `sremu.physics.diffuse.sos_gain(key, t_s, max_doppler_hz, n_sinusoids=48) -> complex128`, unit mean power, pure in absolute time.
- **D20 Render contract.** Backends expose module-level `render(job, out)`; `noise(rx_key, k0, n, sigma)` lives in `sremu.impair.noise`; `sigma² = E|w|²` (total complex variance); `lo_offset_hz` rotates at RX sample rate from absolute index 0.
- **D21 Tooling.** `tools/tasks.py ci` uses `-n 8` (a 16-worker xdist crash was seen); a `paper-junit` task writes the junit plus `{sremu_commit, sremu_dirty}` sidecar.
- **D22 Paper sources.** `build_paper.py` strips TeX comments into an `arxiv/` export directory; `check_arxiv.py` fails on internal notes (DRAFT, TODO, H-gates, co-author agreement notes) in the export.
- **D23** PH-18 and PH-21 tests are parametrized over both backends with `[numpy_ref]`/`[numba_cpu]` in their node ids, so gate accounting sees both.

---

## 0. How to read this document

**Normative words.** MUST, SHOULD and MAY have their RFC 2119 meanings. A numbered rule (R-, D-, I-) is normative.

**Evidence labels.**

| Label | Meaning |
|---|---|
| [FACT] | Stated in an input document, with its own provenance |
| [MEASURED] | Measured on this laptop by a saved script (Appendix A) |
| [DERIVED] | Arithmetic done for this spec (`research/scratch/design_checks/design_calc.py`) |
| [PILOT] | Output of `research/scratch/design_checks/*.py` or the E-B pilot. Used to size experiments; never quoted as a paper result |
| [TARGET] | A number the implementation must try to meet and the benchmarks must confirm or refute |
| [ENG] | An engineering choice with no literature source |
| [UNVERIFIED] | Must be checked before it is relied on |

**Precedence.** This document overrides the three inputs. Where it is silent:
- inside `physics/` and on verification and validation (V&V), `physics_validation.md` applies;
- on the time base, events, engine, transport, recording and metrics, `runtime_interfaces.md` applies;
- on positioning, paper process and honesty rules, `product_paper.md` applies.

All three are subject to §15.

**ID namespaces** (one per kind; the paper's tables are generated from CI output, never typed by hand):
- PH (physics analytic), ST (statistical), G1–G8 (MATLAB goldens), W1–W6 (WCL letter);
- I1–I8 (runtime invariants), RT (runtime oracles), IO (interfaces), VA (experiment consistency);
- V-TANG, V-LAB, V-CFO, V-WCL, V-FM (validation studies);
- E-A…E-G (experiments), B1–B6 (benchmarks), F1–F12 (figures), WP0–WP12 (work packages);
- GATE-PHY and GATE-RT (gates; deliberately not "G1/G2", which are goldens).

**The user's AWS question, answered once.** Do not move development to AWS for speed.
- The hot kernels take milliseconds on the laptop [MEASURED].
- All case-study compute is about 1–3 h of laptop wall time [DERIVED].
- AWS buys three things only: quiet Linux paced-mode timing numbers, one server-class scaling point, and an optional srsRAN demo.
- Expected cost is about US$20–60 [UNVERIFIED; check console prices]. Every launch needs the user's explicit approval, after a laptop pilot (§12.4).

---

## 1. Scope and non-goals

### 1.1 Product definition

sremu is an **open-source, software-only, IQ-level radio-environment emulator**:
- RIS configuration is a **timestamped runtime command from an external controller**, applied at an **exact, logged sample** on every receiver;
- per-state RIS responses can be **imported from measurements**;
- wall-clock behaviour is **measured and reported**, never assumed.

It is not a digital twin, a ray tracer, a UHD drop-in or a hard-real-time emulator.

### 1.2 Gap statement and differentiation

Gap, dated 10 Oct 2026, narrow by design. No surveyed open tool combines all four of:
- (i) multi-node IQ-level emulation behind an SDR-style interface;
- (ii) RIS configurations as runtime commands from an external controller, applied at an explicit, logged sample;
- (iii) per-state responses imported from measurements (state × f × θ_i × θ_s);
- (iv) characterized wall-clock behaviour.

The contribution is **integration plus validation**, not a new capability class. The paper says "to our knowledge, as of 10 Oct 2026" and points to Table I.

| Tool | What it has | What it lacks, relative to sremu |
|---|---|---|
| ACHEM (arXiv 2604.04742) | IQ level, unmodified UHD apps, real time, mobility, open | **No RIS** |
| OAI rfsimulator, srsRAN ZMQ | Lockstep IQ transport; OAI splits blocks at timestamped beam switches | No RIS model; no measured responses |
| Colosseum + RIS (Tsampazi 2025) | IQ, hardware real time | RIS precomputed offline (QuaDRiGa); no runtime RIS control; closed facility |
| GoSimRIS / ORIX | External RIS control (E2SM-RIS, 10 ms–1 s loop) | Coefficients only; no IQ; parametric amplitude model |
| QRIS | Measured/CST unit-cell patterns, multi-RIS | Offline, coefficient-level, request-gated |
| MATLAB R2026a `RectangularRIS`/`RISReflector` | Sample-level reflection, custom unit cell | Far field only (plane-wave), unit-gain cells, offline, commercial |
| Sionna RT | — | No RIS since v1.0, still none in v2.2.0 (29 Sep 2026) [FACT] |
| **TRACE** (co-authors, arXiv 2608.11798) | Socket-based modular framework; channel estimation and tracking (MMSE/OMP/DCAR) | Symbol-level `y = √P gᵀθx + ω`; i.i.d. Rayleigh / Gauss–Markov; ideal continuous phase, unit amplitude; UDP with 5.2 ms control RTT; no sample-accurate command semantics |

**TRACE wording** (agree it with Smriti Kumar and Mansi Ambwani before posting): "TRACE and sremu are complementary: TRACE studies estimation and tracking algorithms on symbol-level channels; sremu provides a sample-level radio environment with timed RIS control. A TRACE-style controller could drive sremu's control plane." No integration code is proposed or claimed.

### 1.3 Reviewer requests mapped to features

| Request (ANTS reviews) | Feature or evidence in sremu | Where |
|---|---|---|
| Quantitative validation, absolute power | K1 kernel; V-TANG; V-LAB pre-registered protocol in absolute dBm | §5.3, §10.8–10.9 |
| Honest real-time definitions and metrics | Lockstep / paced definitions; RTF, deadline misses, latency percentiles | §4, §12 |
| Scalability with stated features and RIS counts; comparison with other tools | B2/B3 sweeps over L, N, RIS count, f_s and feature sets; context table | §12 |
| Demonstrated mobility, multipath, Doppler | Geometry-derived Doppler (PH-17/18); order-1 walls; additive SoS diffuse; E-A | §4.4, §5.8–5.11 |
| RIS non-idealities (quantization, coupling, efficiency) | Ordered non-ideality pipeline; E-C ablation; coupling only through tables (scope paragraph) | §5.6, §11 E-C |
| Clear scope | §1 here; limitations table in the paper | §1.5, §13 |
| How traffic and RIS-control integration work | UHD-shaped API, native ZMQ, control socket; sequence and timing diagrams (F1, F2); E1/E2 | §8, §11 E-E |
| Related work and contributions | Table I; four contributions | §13 |
| Standards mapping (ETSI/TSDSI) | GR RIS 002/003 and GS RIS 008 mapping table; TSDSI sentence | §8.9 |
| Flow diagrams, future work, readable figures | F1, F2; future-work section; figure style rules | §13 |
| AI/ML hooks | Controller plugin API; gym-style `RisEnv`; exact optimum as upper bound | §8.7–8.8 |
| End-to-end metrics (RQ-37) | BPSK/QPSK BER vs Eb/N0 against theory; API latency percentiles | §11 E-F |

### 1.4 Release-1 feature tiers

| Tier | Features |
|---|---|
| **MUST** | Exact time base (§4.1); page/segment engine with bitwise invariance; K1 kernel with `cell_model`; ResponseTable with validators and built-ins; non-ideality pipeline; exact finite-alphabet optimum; LOS, antennas, ground image expansion, explicit occluders/partitions, single knife-edge; analytic trajectories (static, waypoints, constant velocity, RWP-1D/2D, trace); chord-delay Doppler with RIS tiles and `verify_interp`; TX ring with FD filters; Philox AWGN; CFO; timestamped RIS command semantics with per-RX application; transitions; lockstep with barrier controllers; paced mode with counters; batch (coefficient) path; UHD-shaped Python API; native framed ZMQ data plane; JSON control socket; SigMF + events.jsonl + manifest; replay; built-in controllers; legacy adapters for teeth tests; MATLAB goldens; all §10 gates; experiments E-A, E-B, E-C, E-E1, E-F, E-G |
| **SHOULD** | Order-1 image walls; additive SoS diffuse fading; Wiener phase noise; raw REQ/REP mode with a GNU Radio demo (E2); `probe_bandit` controller; E-D FM sizing figure; `sremu report` static HTML; paced run on a c6i.metal host |
| **STRETCH** | torch-CUDA f32-differential geometry backend behind a flag; srsRAN 4G attach demo (E3, 4 h time-box); vector extraction of the WCL letter figures for overlays; P.526 smooth earth beyond the experimental module |

### 1.5 Non-goals (stated in the paper's limitations table)

- UHD binary compatibility or a virtual-USRP driver (ACHEM is cited for that path).
- Hard real time; any paced-latency claim from Windows.
- Ray tracing, diffuse scattering beyond the Rayleigh factor, buildings, terrain profiles per hop, Deygout multi-edge, wall order 2 and RIS–wall paths.
- RIS–RIS multi-bounce; polarization and cross-pol (scalar co-pol model only).
- Mutual-coupling physics. Coupling enters only through measured or simulated per-configuration tables. A neighbour-state lookup **hook** exists with no κ values shipped. Mutual impedance (Gradoni–Di Renzo) is future work.
- Frequency-dependent Γ FIR fitting. A signal wider than a table's declared validity bandwidth is rejected.
- SFO, IQ imbalance, PA models, ADC quantization (beyond an optional full-scale clip and the sc16 wire format).
- Resampling between different TX/RX rates on overlapping bands (rejected with an error). MIMO (`nchan = 1`).
- Web server or dashboard. The v2 path-traversal and cross-site WebSocket bug classes are removed by construction.
- The O-RAN E2SM adapter; TSDSI conformance; GPU IQ path for socket clients; `VectorRisEnv`; CUDA graphs.
- Learning-based controller research. Hooks only, plus one bandit baseline (SHOULD).

### 1.6 Honesty rules (summary; enforced by §13.6)

1. Verification ≠ validation. MATLAB agreement is code-to-code only.
2. Digitized data is labelled DIG, with its uncertainty.
3. "Real-time" never appears unqualified.
4. No cross-machine speed-ups. Legacy code is compared only when rerun on the same host.
5. Banned words without a mapped test: "first", "only", "compliant", "digital twin", "accurately replicates", "no loss of functionality".
6. Nothing in the paper calls the WCL letter's Eq. (2) wrong until Mishra and Kherani confirm the dimensional check. The paper says "under the area-tied kernel the value is X".
7. Every number in the paper text comes from pipeline-generated macros.

---

## 2. Package layout and module responsibilities

### 2.1 Repositories

- `D:\projects\ris\sremu`: new git repo, package `sremu`, Python ≥ 3.12, src layout.
  - Before the first commit, set the git identity to the real author (RQ-07). The "Anamikarajesh" issue must not recur.
  - Tag the legacy repos `v1-legacy` (03697a6) and `v2-legacy` (8abd671) for internal record only. **Amended by D24:** the README does not state a lineage; it says "sremu is an independent new implementation".
  - **Licence is an ADMIN gate** (RQ-100: MIT per the existing metadata, or Apache-2.0). The repo stays private and `LICENSE-PENDING.md` stands in until all six authors and IIT Bhilai agree. pyproject uses `license = {text = "LicenseRef-Pending"}`.
- `D:\projects\ris\paper`: separate git repo, IEEEtran journal, pinned to an sremu commit by `sremu.lock`.

### 2.2 Tree and file ownership

Every file has exactly one owning work package (WP, §14). Shared files are lead-owned (WP0) and change only through a contract change request (§14.3).

```
sremu/
  pyproject.toml  README.md  LICENSE-PENDING.md  CHANGELOG.md  CONTRIBUTING.md  SECURITY.md   [WP0]
  CITATION.cff                                                                                [WP12]
  .gitignore .gitattributes (*.csv *.json text eol=lf)  .github/workflows/{ci,nightly}.yml    [WP0]
  tools/tasks.py            # task runner (no GNU make on Windows): ci | nightly | goldens | bench | paper-data  [WP0]
  tools/matlab/             # golden_*.m, run_goldens.ps1 (calls R2026a explicitly)          [WP4]
  src/sremu/
    __init__.py  py.typed                                                                     [WP0]
    units.py        C0=299_792_458.0 (the only c), K_B, T0=290, dB/dBm helpers                [WP0]
    timebase.py     Rate, TimeSpec, to_ticks, from_ticks, count, page grid                    [WP0]
    contracts.py    frozen cross-WP dataclasses and Protocols (§3.4)                          [WP0]
    errors.py       exceptions; UHD-style ErrorCode/AsyncCode enums                          [WP0]
    cli.py          typer app with a lazy command registry (module paths pre-listed)         [WP0]
    testing/        fake_propagation.py (LOS-only analytic PropagationModel), fake_engine.py (identity channel EngineProtocol), scenes.py  [WP0]
    rng.py          key derivation, Philox4x32-10 (numba + numpy twin), Generator factory     [WP1]
    scenario/       models.py (pydantic v2), load.py (YAML/JSON/TOML), guards.py, schema_export.py  [WP1]
    record/manifest.py                                                                        [WP1]
    physics/                                                                                  [WP2 unless noted]
      geometry.py   Pose, panel cell centres, visibility, segment–polygon test
      antennas.py   Isotropic, CosPower(alpha), Tabulated
      los.py
      quadrature.py cell_model auto/point/po_patch, sub-cells, Richardson check
      ris_kernel.py K1 (numpy f64 reference + numba)
      response.py   ResponseTable, Provenance, validators, interpolation, importers
      tables/       built-in table data (YAML/CSV) with provenance
      nonideal.py   quantizer, FailureMap, efficiency, CouplingHook, eta(), binary m/d decomposition
      optimize.py   exact_alphabet_optimum, continuous_optimum
      propagation.py  PropagationModel: snapshot(), combine(), ris_delay(), from_scenario()
      legacy.py     legacy_v1, legacy_v2, legacy_wcl_eq2, ellingson_qgamma, no_j (allow_nonphysical)
      ground.py  walls.py  diffraction.py  diffuse.py                                         [WP3]
      experimental/p526.py   (not imported by the core; tests in tests/experimental)          [WP3]
    channel/                                                                                  [WP5]
      ring.py  fd.py  interp.py (T_u/tile policy, chord, verify_interp)  transitions.py
      render_numpy.py  render_numba.py
    impair/  noise.py  oscillator.py (CFO, Wiener PN)  frontend.py (full scale, gain, clip)    [WP5]
    engine/                                                                                   [WP6]
      events.py  commands.py  streams.py  scheduler.py  barriers.py  pacer.py  emulator.py  batch.py
    metrics/ hist.py  rt.py                                                                   [WP6]
    legacy/  kernels re-exported from physics.legacy [WP2];  truncation.py, next_tick.py (adapters) [WP6]
    api/     device.py  metadata.py  streams.py                                               [WP7]
    compat.py                                                                                 [WP7]
    record/  sigmf.py  eventlog.py  replay.py  report.py                                      [WP7]
    transport/ framing.py  zmq_native.py  control_server.py  client.py  rawreqrep.py  schema/*.json  [WP8]
    control/ controller.py (helpers), builtin.py [WP9];  remote.py [WP8]
    ml/env.py                                                                                 [WP9]
    analysis/stats.py   experiments/sweep.py  experiments/registry.py                         [WP9]
  scenarios/   quickstart.yaml, lab_like_A.yaml [WP1];  study scenarios [WP10/WP11]
  data/states/*.yaml [WP1];  data/validation/{tang2022,cfo,wcl}/ [WP10]
  validation/  tang/  ants_fig4b/ (PROTOCOL.md, protocol.json, protocol_lock.py)  cfo/  wcl/  fm/   [WP10]
  bench/       b1_kernels.py … b6_crosstool.py, aws/ (cloud-init, run_all.sh, TEARDOWN.md)     [WP11]
  examples/    quickstart/, bpsk_txrx/ [WP7];  controller_remote/, gnuradio_zmq/ [WP8];  gym_quickstart/ [WP9]
  docs/        model_card.yaml (single source for paper Table II) + generated .md, timing_semantics.md, scenario_reference.md (generated), state_library.md, controllers.md, integration.md, reproducibility.md, limitations.md  [WP12]
  tests/
    oracles/   closed forms ported from pm_lib.py / pm_checks.py; MUST NOT import sremu.physics, sremu.channel or sremu.impair  [WP4]
    golden/matlab_R2026a/*.csv + *.meta.json, test_goldens.py                                 [WP4]
    unit/test_timebase*.py test_layering.py test_cli.py [WP0];  unit/test_scenario*.py test_rng*.py test_manifest*.py [WP1]
    physics/test_ph*.py   (PH ids in names)  [WP2: PH-01–07, 09, 13–16, 23–26, 28–31;  WP3: PH-08, 10, 11, 27, 32]
    runtime/test_render*.py test_noise*.py [WP5];  runtime/test_rt*.py [WP6]
    io/test_io*.py [WP7: IO-05..07;  WP8: IO-01..04, 08]
    statistical/ [ST-01 WP6, ST-02 WP3, ST-03 WP2, ST-04/ST-05 WP11/WP10]
    integration/ [WP8: two-process; WP10: validation smoke]
    experimental/ [WP3]
```

### 2.3 Layering (enforced by `tests/unit/test_layering.py`, which parses imports)

```
units, timebase, errors, contracts
  -> rng, scenario
  -> physics
  -> channel, impair
  -> engine, metrics, legacy
  -> api, record, transport, control, ml
  -> analysis, experiments
  -> cli
```

- A module may import only from its own layer or lower layers.
- `tests/oracles` must not import `sremu.physics`, `sremu.channel` or `sremu.impair`.
- I7 (static part): no module under `physics/`, `channel/`, `impair/`, `engine/scheduler.py`, `engine/events.py` or `engine/commands.py` may import `time`, `datetime` or `perf_counter`. Wall clock is allowed only in `engine/pacer.py`, `metrics/`, `transport/` and `record/`.

### 2.4 Dependencies (offline-installable on this laptop)

- **Core.** `numpy>=2.2`, `scipy>=1.14`, `numba>=0.62`, `pydantic>=2.12`, `pyyaml>=6`, `pyzmq>=27`, `typer>=0.12`, `matplotlib>=3.9`. All are installed [FACT: checked].
- **Extras.**
  - `dev` = pytest, hypothesis, ruff, mypy. hypothesis is **not installed**: property tests use `tests/_prop.py` (seeded loops) and use hypothesis only if it imports.
  - `ml` = gymnasium (not installed; `RisEnv` duck-types it).
  - `sigmf` = sigmf (not installed; metadata is written as plain JSON and checked structurally).
  - `parquet` = pyarrow (installed; sweep results only).
  - `gpu` = torch (STRETCH).
- **Build backend.** `setuptools>=70` (hatchling is not installed). Offline install: `pip install -e . --no-build-isolation --no-deps`.
- **Tooling.** latexmk and pdflatex come from MiKTeX. There is no GNU make, so `tools/tasks.py` and `paper/tools/build_paper.py` are Python entry points.

### 2.5 CLI (`sremu`, typer; lazy registry in `cli.py`)

| Command | Owner | Purpose |
|---|---|---|
| `validate S.yaml` | WP1 (+physics diagnostics, WP2) | Schema, guards and derived quantities: λ, Fraunhofer distances, f_D,max, auto T_u and tiles, narrowband or wideband per link, noise floor |
| `run S.yaml --out DIR [--mode paced] [--serve tcp://127.0.0.1:5600]` | WP6 / WP8 | Headless or served run |
| `batch S.yaml` | WP6 | Coefficient-only path |
| `sweep SPEC.yaml --jobs N` | WP9 | One run directory per point, plus an index |
| `replay RUN_DIR` | WP7 | Bitwise replay |
| `report RUN_DIR -o r.html` | WP7 | Static HTML, inline SVG, no JavaScript |
| `verify [--quick]` | WP4 | Oracle subset; prints model-card numbers (installed-wheel smoke test) |
| `states import X.csv` / `states check` | WP2 | Response-table tooling |
| `schema {scenario,states,manifest,control}` | WP1 / WP8 | JSON Schema export |
| `bench --suite {b1..b6}` | WP11 | Benchmarks with platform capture |
| `info` | WP0 | Versions, CPU, GPU, power plan or governor, numba threading and SVML state |

### 2.6 CI tiers (`tools/tasks.py`)

| Tier | Content | Budget | Runs |
|---|---|---|---|
| `ci` | Markers `unit or physics or runtime or io or golden`, on `numpy_ref` and `numba_cpu` | ≤ 8 min | Every merge to main |
| `nightly` | `statistical`, `integration` (two-process), `validation_smoke` | ≤ 30 min | Nightly or before each gate |
| `release` | V-* studies and paper data | ≤ 60 min with `PROFILE=fast` | Before figures |

GitHub Actions workflows (ubuntu-latest and windows-latest, py3.12) are committed but used only once the repo is public. CI timing is never quoted in the paper.

---

## 3. Core data model and units

### 3.1 Conventions (normative)

| Item | Rule |
|---|---|
| Speed of light | `units.C0 = 299_792_458.0`. No other literal appears anywhere (grep test) |
| Frames | Right-handed, metres, z up. Panel pose = (centre `c`, unit normal `n̂`, in-plane `û`); `v̂ = n̂ × û` |
| Baseband | `s(t) = √2·Re{x(t)e^{j2πf_c t}}`, so mean `s² = mean |x|²` in W. A path of delay τ maps `x(t) → a·x(t−τ)·e^{−j2πf_c τ}`. Doppler is positive on approach |
| Channel coefficient | Dimensionless and Friis-normalized: `P_R = P_T·|h|²`. Antenna gains sit inside h as `√(G_T G_R)` per path |
| Table angles | θ ∈ [−90°, 90°], measured from n̂ in the (û, n̂) plane and signed by the û component. θ_i points from the cell toward the Tx, θ_s from the cell toward the Rx. Reciprocity reads `Γ(s; θ_i, θ_s) = Γ(s; θ_s, θ_i)`. Specular reflection is `θ_s = −θ_i` |
| Noise | `CN(0, k_B T_0 F f_s)` W per **receiver** (not per link); −173.975 dBm/Hz + NF; −113.975 dBm at 1 MS/s with F = 0 dB [DERIVED] |
| Internal units | SI. dB and dBm appear only at I/O boundaries |
| Field naming | Every dimensional scenario or API field carries a unit suffix: `_hz _sps _m _s _dbm _db _deg _rad _mps _w` |
| Exact quantities | Sample rates, and every time that maps to a sample (durations, event times, periods, latencies), are **strings** parsed to `Fraction` (`"23.04e6"`, `"1/200"`, `"5e-3"`). Floats are rejected in scenario files for these fields |

### 3.2 Identifiers (also used as RNG keys, so they are stable strings)

| Kind | Form | Example |
|---|---|---|
| Node | id from the scenario | `ap`, `mu`, `rx07` |
| Stream | `<node>.<stream>` | `ap.tx0`, `mu.rx0` |
| Link | `<tx_stream>-><rx_stream>` | `ap.tx0->mu.rx0` |
| Panel | panel id | `ris0` |
| Command | client-supplied id (≤ 64 chars); the engine assigns a monotone `seq` | — |

A link exists only between streams of equal rate whose carriers satisfy `|f_c,tx − f_c,rx| < f_s/2`. Overlapping bands at different rates raise `RateMismatchError` (release 1).

### 3.3 Device power mapping

- **TX.** `x = d·√P_FS,tx·10^{G_tx/20}`, with `P_FS,tx` from `full_scale_dbm` (default 0 dBm) and `gain_db` (default 0).
- **RX.** `d = y·10^{G_rx/20}/√P_FS,rx`, with RX `full_scale_dbm` default 0 dBm. So by default `10·log10|d|²` reads directly in dBm.
- Optional `adc_clip: true` clips I and Q to ±1. The `sc16` wire format saturates.

The ANTS Fig. 4(b) comparison does not depend on these constants: one fitted offset C absorbs P_T, antenna gains and cables (§10.9).

### 3.4 `contracts.py` (frozen at the end of WP0; changed only by CCR, §14.3)

```python
CONTRACT_VERSION = "1.0"
StreamId = str; LinkId = str; PanelId = str

class PathKind(IntEnum): LOS = 0; GROUND = 1; WALL = 2; DIFF = 3; RIS_TILE = 4; DIFFUSE = 5

@dataclass(frozen=True)
class PathBundle:                 # non-RIS deterministic paths of one link at snapshot time t
    link: LinkId; t: Fraction
    amp: np.ndarray               # (P,) complex128: antenna gains, 1/(4πd)·λ, Fresnel Γ, F(ν), partition factor.
                                  #   EXCLUDES the propagation phase e^{-j2π f_c τ}
    delay_s: np.ndarray           # (P,) float64, absolute propagation delay
    kind: np.ndarray              # (P,) int8 PathKind

@dataclass(frozen=True)
class RISCoefficients:            # one panel, one snapshot
    panel: PanelId; t: Fraction
    rows: tuple[tuple[LinkId, int], ...]   # (link, hop_variant); variant 0 direct, 1 Tx-image, 2 Rx-image, 3 both (ground)
    tile_of_cell: np.ndarray      # (N,) int32 in [0, K)
    tile_delay_s: np.ndarray      # (R, K) float64 reference delay Tx -> tile ref point -> Rx (per row)
    B: np.ndarray | None          # (R, S, N) complex128 per state and physical cell: Γ(s,f,θi,θs)·A·o·√(G_T G_R)·ground factors·j/(4π r_i r_s)
                                  #   · e^{-jk(r_i + r_s)} · e^{+j2π f_c τ_tile}   (phase RELATIVE to the cell's tile reference)
    C: np.ndarray | None          # (R, N) separable form: same without Γ (tables that do not depend on angle)
    gamma: np.ndarray | None      # (S,) separable alphabet (efficiency already applied)
    gamma_dead: complex = 0j

@dataclass(frozen=True)
class Snapshot:
    t: Fraction
    paths: Mapping[LinkId, PathBundle]
    ris: Mapping[PanelId, RISCoefficients]
    diag: Mapping[str, float]     # energy budget, Richardson estimate, max off-plane angle, interpolation-bound inputs

@dataclass(frozen=True)
class PanelConfig:
    panel: PanelId
    state: np.ndarray             # (N,) int16 per PHYSICAL cell after granularity expansion, stuck-at and dead (-1) handling
    enabled: bool = True          # explicit "RIS absent" flag (RQ-12); distinct from every state index

@dataclass(frozen=True)
class LinkPaths:                  # combine() output: one link, configuration applied
    link: LinkId; t: Fraction
    amp: np.ndarray; delay_s: np.ndarray; kind: np.ndarray   # (P_tot,) incl. one RIS_TILE path per (panel,row,tile)
    tile: np.ndarray              # (P_tot,) int32, -1 for non-RIS, else a flat (panel,row,tile) key

class PropagationModel(Protocol):
    def links(self) -> tuple[LinkId, ...]: ...
    def snapshot(self, t: Fraction) -> Snapshot: ...                     # pure in (scenario, t)
    def combine(self, snap: Snapshot, cfg: Mapping[PanelId, PanelConfig]) -> Mapping[LinkId, LinkPaths]: ...  # linear in Γ
    def ris_delay_s(self, panel: PanelId, rx: StreamId, t: Fraction) -> float: ...   # τ_RIS→rx at the panel centre
    def motion_bounds(self) -> "MotionBounds": ...                       # v_rel,max, r_min per link (for the T_u/tile policy)

@dataclass(frozen=True)
class LatencyModel:
    fixed_s: Fraction = Fraction(0)
    jitter: str | None = None     # "uniform:0,2e-3" | "exp:1e-3"; drawn from keyed RNG purpose "ctrl.latency/<name>"
    mode: Literal["virtual", "wall"] = "virtual"

@dataclass(frozen=True)
class ObservationSpec:
    fields: tuple[str, ...] = ("rx_power_dbm", "ris_state")   # + "positions", "csi_oracle", "iq_window"
    window_s: Fraction = Fraction(1, 1000)
    position_delay_s: Fraction = Fraction(0); position_noise_m: float = 0.0

@dataclass(frozen=True)
class Observation:
    t_b: Fraction; k_b: Mapping[StreamId, int]
    rx_power_dbm: Mapping[StreamId, float]       # measured from real RX IQ (noise and impairments included)
    positions: Mapping[str, np.ndarray] | None
    csi_oracle: Mapping[LinkId, "CsiView"] | None  # use is flagged in the manifest as oracle
    iq_window: Mapping[StreamId, np.ndarray] | None
    ris_state: Mapping[PanelId, np.ndarray]; pending: tuple["CommandReceipt", ...]

@dataclass(frozen=True)
class Action:
    panel: PanelId
    states: np.ndarray | None = None; phases_rad: np.ndarray | None = None; codebook_index: int | None = None
    at_s: Fraction | None = None                 # default t_b + latency
    transition: "TransitionSpec | None" = None

class Controller(Protocol):
    name: str; period_s: Fraction; min_latency_s: Fraction; spec: ObservationSpec; latency: LatencyModel
    def setup(self, caps: Mapping[PanelId, "RisCapabilities"], rng: np.random.Generator) -> None: ...
    def decide(self, obs: Observation) -> Sequence[Action]: ...

@dataclass
class CommandRecord:              # one per command, JSONL + SigMF annotation
    cmd_id: str; seq: int; src: str; op: str; panel: PanelId
    t_req: Fraction; t_applied: Fraction | None              # switch time at the RIS
    k_req: dict[StreamId, int]; k_applied: dict[StreamId, int]
    lateness_samples: dict[StreamId, int]                    # k_applied - k_req
    t_wall_recv_ns: int | None; t_wall_computed_ns: int | None; t_wall_visible_ns: int | None
    status: Literal["scheduled", "applied", "late_applied", "rejected", "superseded"]
    config_hash: str              # blake2b-128 of (state int16 bytes, enabled flag)
    transition: dict | None; counterfactual: bool = False

class EngineProtocol(Protocol):   # what api/ and transport/ use; FakeEngine implements it in WP0
    def open_stream(self, spec: "StreamSpec") -> "StreamHandle": ...
    def submit_tx(self, h, samples: np.ndarray, md: "TxMetadata") -> int: ...
    def stream_cmd(self, h, cmd: "StreamCmd") -> None: ...
    def read_rx(self, h, n: int, timeout_s: float) -> tuple[np.ndarray, "RxMetadata"]: ...
    def schedule(self, req: "CommandRequest") -> "CommandReceipt": ...
    def register_controller(self, ctrl: Controller | "RemoteControllerProxy") -> None: ...
    def time_now(self) -> Fraction: ...
    def caps(self) -> Mapping[PanelId, "RisCapabilities"]: ...
    def recv_async(self, h, timeout_s: float) -> "AsyncMetadata | None": ...
```

**Render contract** (WP5 implements it, WP6 builds jobs):

```python
@dataclass
class LinkJob:
    link: LinkId; ring: "TxRing"; fc_hz: float; lo_offset_hz: float   # f_c,tx − f_c,rx
    amp_a: np.ndarray; amp_b: np.ndarray          # (P,) endpoint amplitudes at t_a, t_b (complex-linear between them)
    tau_a: np.ndarray; tau_b: np.ndarray          # (P,) endpoint delays (chord between them)
    group: np.ndarray                             # (P,) int16 FD group; one group per link when narrowband
    fd: "FDSpec"                                  # filter id, taps L, lookahead h_FD
    trans: list["TransitionTerm"]                 # (k_start, profile, duration_samples, dA_a, dA_b)
    rot_state: np.ndarray | None                  # numba rotator carry within a page (in/out)
@dataclass
class SegmentJob:
    rx: StreamId; f_s: Fraction; k0: int; k1: int      # absolute RX samples [k0, k1)
    t_a: Fraction; t_b: Fraction                        # enclosing snapshot interval
    links: list[LinkJob]
def render(job: SegmentJob, out: np.ndarray) -> None   # out: complex128 accumulator, len k1−k0; pure function
def noise(rx_key: tuple[int, int], k0: int, n: int, sigma: float) -> np.ndarray   # complex64, pure in absolute index
```

### 3.5 Shapes, dtypes, memory

| Quantity | Shape | dtype |
|---|---|---|
| Positions, quadrature points | (Q, 3) | f64 |
| Per-link distances | (R, Q) | f64; f32-differential only on the STRETCH GPU path |
| B tensor | (R, S, N) | c128. Example: 64 rows × 2 states × 4096 cells = 8 MB. S > 16 needs a separable table |
| IQ at every interface | (n,) | complex64. Accumulators are complex128 |
| State vectors | (N,) | int16. −1 = dead |
| Sample indices | — | Python int at the boundary, int64 in kernels (lasts 1.27e4 years at 23.04 MS/s) |

### 3.6 Run directory and manifest

```
runs/<UTC>_<name>/
  manifest.json  scenario.resolved.yaml  events.jsonl  metrics.jsonl  summary.json
  iq/<stream>.sigmf-meta + .sigmf-data (optional)  run.sigmf-collection
  snapshots/*.npz (optional, referenced by hash)  results.parquet (sweeps)  dirty.patch (only if dirty)
```

`manifest.json` MUST carry:
- `sremu_version`, `git_commit`, `git_dirty`, `dirty_patch_sha256`, `contract_version`;
- versions of python, numpy, scipy, numba, pydantic, pyzmq and torch;
- `platform`, `cpu_model`, `logical_cpus`, `power_plan_or_governor`, `gpu`, `aws_instance` (type and id, or null);
- `scenario_sha256` (canonical JSON of the resolved scenario);
- `inputs[{path, sha256, data_class}]` and `assumptions[]`;
- `seed`, `rng_scheme` (`"philox4x32-10/blake2b-v1"`);
- `mode`, `coupling`, `backend`, `threads`, `deterministic`, `numba_disable_svml`;
- per-stream `page_size`;
- per-link narrowband/wideband decision, FD filter, and the chosen T_u and tile counts with their error bounds;
- `verify_interp_max_rel`, `richardson_db`;
- counters summary, `nondeterministic_inputs`, `oracle_csi_used`;
- `argv`, `t_start_utc`, `t_end_utc`, `wall_s`, `host_load_samples`.

`data_class` ∈ {MEAS, DIG, REPORTED, ASSUMED, SIM-sremu, SIM-legacy-v1, SIM-legacy-v2, ANALYTIC, MATLAB}.

---

## 4. Time model and runtime

### 4.1 Rates, times and rounding

- **R1.** A rate is an exact `Fraction` parsed from a string or int (`Rate.parse("23.04e6") == Fraction(23040000)`). Floats raise `TypeError` in scenario files.
- **R2.** Virtual time is an exact `Fraction` of seconds. Index 0 of every stream is t = 0. The optional `epoch_datetime` goes only into SigMF `core:datetime`.
- **R3.** `to_ticks(t, f) = floor(t·f + 1/2)`: nearest, ties up (UHD `to_ticks`). It is used for **every** time→sample mapping: events, barriers, frontier limits, `TimeSpec`s. `from_ticks(k, f) = Fraction(k)/f` is exact. Worst-case application error is `1/(2f)`, reported in the paper.
- **R4.** Interval counts: stream s produces `count(t0, t1) = ceil(t1·f) − ceil(t0·f)` samples on the half-open interval [t0, t1), in exact integer arithmetic. This replaces `int(f·τ)`, the root cause of the ANTS Fig. 2 slope of 1.0691 (11.76/11 at 5880 S/s, τ = 2 ms) [FACT].
- **R5.** API floats are converted with `Fraction(repr(x))`. `TimeSpec(full_secs, frac_secs).exact() = full_secs + Fraction(repr(frac_secs))`. A float 0.5e-3 s at 23.04 MS/s maps to sample 11520 by both repr and nearest rounding; ceil would give 11521 [DERIVED].
- **R6.** Wall time never enters data-plane arithmetic (I7).

### 4.2 Runtime invariants (each has tests in §10.3)

| ID | Invariant | Tolerance |
|---|---|---|
| I1 | After virtual time T, stream s has produced exactly `ceil(T·f_s)` samples, for any mix of rates | exact |
| I2a | RX output does not depend on: client send/recv sizes; segmentation by events, barriers, epochs or availability; skipped inactive periods; lockstep vs paced without misses; numba thread count in deterministic mode | **bitwise** |
| I2b | RX output when `page_target_s` changes | ≤ 1e-9 relative (complex128 accumulators before the complex64 cast) |
| I3 | A RIS switch at RIS time t_sw takes effect on RX r at `k_r = to_ticks(t_sw + τ_RIS→r(t_sw), f_r)` (per tile when tiled) | exact |
| I4 | Every random variate is a pure function of `(master_seed, purpose, name, absolute index)`. Adding or removing a node does not change other streams' variates | bitwise |
| I5 | `numba_cpu` vs `numpy_ref`: ≤ 1e-9 relative on complex128 accumulators, ≤ 1e-6 relative RMS on complex64 output. Philox uint32 streams bitwise identical across the two | as stated |
| I6 | Nothing is lost silently. Every late, dropped, overflowed, underflowed, stalled or rejected sample or command is counted and logged with its sample index | exact counts |
| I7 | Wall clock never enters data-plane arithmetic | static import check + test |
| I8 | Lockstep, barrier-registered controller, every action at `t ≥ t_b + L_min`: `lateness == 0` on every RX, for random rational rates and times | exact (property test) |

### 4.3 Page grid, segments and determinism

- **Page size.** Each RX stream r has `P_r = clamp(2^round(log2(f_r·page_target_s)), 64, 16384)`, with default `page_target_s = "5e-4"`. Page j covers absolute samples [j·P_r, (j+1)·P_r).
  - 23.04 MS/s → 8192; 30.72 MS/s → 16384; 1 MS/s → 512; 10 kS/s → 64; 5880 S/s → 64.
- **Rate groups.** Streams with equal rate and page size form a rate group; one backend call per page per rate group.
- **Segments.** Events (per-RX `k_applied`), snapshot-grid boundaries, barrier limits and input availability may split a page into segments. Segments of a page are processed in order.
- **D-1, closed form.** Every per-sample quantity is a closed-form function of the absolute sample index n and the endpoints of the enclosing snapshot interval:
  - interpolation weight `w(n) = ((n − n_a,int) − n_a,frac)·inv_len`, where `n_a = t_a·f` is split exactly into integer and fractional parts;
  - delay `τ(n) = τ_a + w(n)(τ_b − τ_a)`;
  - phase in cycles `c(n) = frac(f_c τ_a) + w(n)·f_c(τ_b − τ_a)`, so large absolute phases never lose precision.
- **D-2, recursions.** Any recursion (the numba rotator `e^{−j2πc(n)}` and Wiener-bridge sums) restarts at **page starts and snapshot boundaries** from a closed-form anchor. Its state is **carried across segments within a page** (`LinkJob.rot_state`). Measured rotator drift is 5.0e-13 rad over 1 s at 23.04 MS/s [MEASURED].
- **D-3, summation order.** Sums over links and paths use a fixed order: links in scenario order; paths by (kind, index); tiles by index. Parallelism is over samples and RX streams, **never over the summation index**.
- **D-4, no fastmath** in any kernel on the bitwise path. In deterministic mode the CLI sets `NUMBA_DISABLE_INTEL_SVML=1` before numba is imported. SVML is off on this laptop (`USING_SVML=False`) but may be active on AWS Linux. Bitwise thread-count invariance is claimed **only in deterministic mode**; otherwise the claim is ≤ 1e-12 relative. Cross-OS and cross-machine results are claimed at ≤ 1e-12 relative, never bitwise.
- **D-5, GEMV order.** The configuration GEMV on the bitwise path is a numba loop in fixed order (no BLAS threading).
- **D-6, skipping.** Inactive RX pages may be skipped. Class (a) and (b) processes (§6.3) make this bitwise-safe.

### 4.4 Snapshot grid, Doppler and RIS tiles (normative definition)

**Definition (resolves physics vs runtime).**
- Geometry is evaluated at snapshot times `t_k = k·T_u`, aligned to absolute time; static scenes use snapshots only at events.
- Between snapshots, each non-RIS path has a **chord delay**, `τ_p(t)` linear between `τ_p(t_k)` and `τ_p(t_{k+1})` computed in f64. Its carrier phase is **exact for that delay**, `e^{−j2πf_c τ_p(t)}`, and its amplitude is complex-linear.
- Each RIS panel row is split into **tiles**. Tile k has a reference delay `τ_ref,k(t)`, rendered like a path (chord + exact phase), and a residual `g_k(t) = Σ_{n∈k} B[s_n, n](t)` (relative to the tile reference), interpolated complex-linearly.
- Because chord τ is linear in t, the phase is linear in t. The numba rotator `r = e^{−j2πΔc/(f_s T_u)}` is therefore exact up to rounding: the "analytic-ν trapezoid rotator" of the runtime doc is not needed. A test enforces numba-rotator vs numpy direct-sincos equivalence (I5).
- Configuration switches are exact discontinuities at sample indices, never interpolated across.
- The legacy `block_constant` mode exists only in `legacy/` for the teeth test (−148.3 Hz alias).

**Error budget.** One target, `ε_rel = 1e-3` relative |Δh|/|h| (−60 dB), shared by the T_u rule, the tile rule, PH-18 and `verify_interp`:
- curvature (chord) error `k·r̈·T_u²/8 ≤ ε_rel`, with `r̈ ≤ v⊥²/r_min`, which gives `T_curv = √(8ε_rel r_min/(k v⊥²))`;
- residual error: the linear-interpolation midpoint dip ≈ Δφ²/8 ≤ ε_rel, so the residual phase change per interval is `Δφ_res ≤ √(8ε_rel) = 0.0894` rad. With `Δφ_res = 2π·v_rel·D_t·T_u/(λ·R_min)` for tile side D_t (full side, conservative), the tile side is `D_t ≤ √(8ε_rel)·λ·R_min/(2π v_rel T_u)`.

**Policy (`channel/interp.py`).**
1. Compute `T_curv`. Take `T_u0 = min(T_curv, T_u,max = 10 ms)`. The scenario may only *lower* T_u.
2. For m = 0…4, with `T_u = T_u0/2^m`, compute D_t and the tile count, then the cost `c_geo·Q·R/T_u + c_rot·tiles·R·f_s`, using the coefficients fitted in B1 (defaults c_geo = 5 ns per element-row, c_rot = 3 ns per tile-sample) [ENG].
3. Pick the minimum-cost pair and log it in the manifest. Static scenes get one tile per panel and event-only snapshots.

**Worked numbers** [DERIVED, `design_checks/design_calc.py`; candidates m = 0, 1, 2]:

| Scenario | T_curv | T_u candidates (ms) | Tiles per dim |
|---|---|---|---|
| Lab walk, 3.5 GHz, 1.5 m/s, R_min 0.45 m, panel 1.08 m | 4.67 ms | 4.67 / 2.34 / 1.17 | 14 / 7 / 4 |
| WCL corridor, 1 GHz, 1 m/s, h = 2 m, 10 m line | 27.6 ms → 10 ms cap | 10 / 5 / 2.5 | 12 / 6 / 3 |
| WCL corridor, 2.4 GHz, same geometry | 17.8 ms → 10 ms | 10 / 5 / 2.5 | 29 / 15 / 8 |
| UAV, 3.5 GHz, 15 m/s, R_min 20 m, 0.4 m panel | 3.11 ms | 3.11 | 1 |
| Vehicle, 3.5 GHz, 30 m/s, R_min 5 m, 0.4 m | 0.78 ms | 0.78 / 0.39 | 2 / 1 |
| FR2, 28 GHz, 30 m/s, R_min 5 m, 0.4 m | 0.28 ms | 0.28 / 0.14 / 0.07 | 5 / 3 / 2 |

**`verify_interp`.** On in CI, in benchmarks and in every paper run. Every 64th snapshot interval, the engine evaluates the exact midpoint snapshot (all elements) and logs `max|h_interp − h_true|/|h|` per link. The manifest stores the maximum. Runs exceeding 2·ε_rel are flagged; the paper quotes the measured value.

Doppler examples [DERIVED]: 3.5 GHz at 30 m/s gives 350.242 Hz; 1 GHz at 1 m/s gives 3.336 Hz; 3.5 GHz at 1.5 m/s gives 17.51 Hz.

### 4.5 Envelope delay, narrowband test and fractional delay

- **TX history ring** per TX stream, length `H = 2^⌈log2(τ_max f_s + L_FD + 2P + 1024)⌉`. Delayed samples are read at the integer offset plus a fractional part.
- **Declared occupied bandwidth.** Each TX stream declares `occupied_bw_hz` (B; default `0.8·f_s`, with a warning when it is not declared).
- **Narrowband test** per link: `π·B·Δτ_link ≤ 0.05` rad, where Δτ_link is the spread over all paths and RIS tiles (snapshot maximum).
  - If narrowband, all paths share one envelope delay `τ_0(t)` = the chord-interpolated midpoint of the min and max delays, and keep their own exact carrier phases. The per-path envelope error is then ≤ −32.0 dB at the band edge and −36.8 dB RMS for a flat spectrum [DERIVED].
  - Otherwise, paths are clustered so that within a cluster `π·B·δ ≤ 0.05`. Each cluster gets its own FD filter at its chord delay.
- **Integer-only delay** (no FD filter) is allowed only when `π·B/(2f_s) ≤ 0.05`, i.e. `B ≤ 0.0318·f_s`: tones and CW measurements.
- **FD filter table** (Kaiser-windowed sinc, polyphase table of 1024 fractional phases with linear interpolation between phases). Worst case over μ, from `design_checks/fd_check.py` [PILOT]:

| Occupied band (B/f_s) | Filter | In-band error, worst μ | Lookahead h_FD |
|---|---|---|---|
| ≤ 0.2 | 8 taps, β = 8 | −71.4 dB | 4 |
| ≤ 0.6 | 16 taps, β = 8 | −75.5 dB | 8 |
| ≤ 0.8 | 32 taps, β = 8 | −76.5 dB | 16 |
| > 0.8 | rejected (`fd_error_target_db` defaults to −60) | 48 taps reach only −57.1 dB at ±0.45 f_s | — |

  The rejected options: 4-tap cubic Lagrange (both input docs' default) errs by −25.8 / −13.2 / −5.2 dB at ±0.2 / 0.3 / 0.4 f_s, and the runtime doc's `f_s·Δτ ≤ 0.1` rule allows −14.8 dB per path [PILOT].
- **Causality.** The FD filter is centred, so RX page [k0, k1) needs TX committed through `k1 + h_FD − ⌊τ_min f_s⌋` (§4.8). Timing semantics stay exact; no global latency offset is added.
- **Worked decisions** [DERIVED]: lab at 1 MS/s, B = 0.8 MHz, Δτ = 3 ns → π·B·Δτ = 0.0075, narrowband. Lab at 23.04 MS/s, B = 18.4 MHz → 0.17, wideband. FM at 1 MS/s, B = 200 kHz, Δτ = 6.7 µs → 4.2, wideband. WCL tone at 10 kS/s → narrowband.

### 4.6 Events and RIS command semantics

- **Event heap.** Events are ordered by `(t, prio, seq)`; prio 0 geometry, 1 RIS, 2 impairment, 3 stream command, 9 marker. There is **no single-slot pending configuration**: every command is kept. For two commands with the same t on the same panel, the later `seq` wins at that instant; both are logged and the earlier one gets status `superseded`.
- **Switch time.** `t_sw` is the switching time **at the RIS**. RX r sees the change on that panel's paths at `k_r = to_ticks(t_sw + τ_RIS→r(t_sw), f_r)`, with τ at the panel centre, or per tile when tiled.
  - The lab's 1.95 m is < 1 sample at any rate used. 2 km is 153.7 samples at 23.04 MS/s and 6.7 samples at 1 MS/s [DERIVED].
  - Option `ris_event_reference: "ris" | "rx"` (default `"ris"`).
- **TX-side events** act on TX samples before the delay line, so propagation is automatic. RX-side events act at RX samples.
- **Late commands.** `frontier_r` = the first uncomputed sample of RX r. A command is late if some affected RX has `k_r < frontier_r`.
  - `apply_on_arrival` (default, UHD semantics) moves the switch to the earliest **single physically consistent** time `t_sw' = max_r(frontier_r/f_r − τ_RIS→r)` and re-maps every RX with R3.
  - `reject` returns `LATE_COMMAND` and applies nothing.
  - In lockstep, any late command from a source that is not barrier-registered sets `nondeterministic_inputs = true` in the manifest.
- **Time forms accepted.** `at = {"t": "<exact>"}`; `{"sample": {"stream": s, "index": k}}`, converted to `t = from_ticks(k) − τ_RIS→s`; or `"asap"`, meaning `t_req` = the frontier time at arrival.
- **Command records.** Every command produces a `CommandRecord` (§3.4), written to `events.jsonl` (canonical, append-only, flushed per record) and mirrored as SigMF annotations when IQ is recorded.
- **Counterfactual events.** `ris.set_enabled` (RIS absent/present) is accepted as a timed event with `counterfactual: true` in its record. It exists for A/B baselines within one run and is not a physical action.

### 4.7 Transitions (non-instant reconfiguration)

- Elements are grouped into **transition classes** c, for example rising and falling edges. Each class has a profile `g_c(t)`: `step` (default), `linear` over T_c, or `first_order` `1 − e^{−t/τ_c}`.
- The default duration comes from the panel capability `reconfiguration_time_s` (the ETSI GS RIS 008 parameter).
- Exact, by linearity of `combine` in Γ, at RX time t:

  `h(t) = h_old(t) + Σ_c g_c(t − t_sw,r) · Δh_c(t)`, with `Δh_c = combine(cfg_old with class-c elements set to new) − combine(cfg_old)`

  Each Δh_c is interpolated per tile like any residual, so this stays correct under mobility.
- A command arriving mid-transition starts from the instantaneous per-element Γ, `Γ_old + g(t)(Γ_new − Γ_old)`.
- The complex-linear path is **[ENG]**. For the lab pair (203.41° apart) its midpoint is `m = (σ1+σ2)/2`, with `|m| = 0.203` (−13.86 dB) at ∠127.9° [DERIVED]. A panel MAY instead supply a `gamma_path` table, Γ(bias fraction) per state pair, which replaces complex-linear interpolation. Until a GS RIS 008 reconfiguration-time measurement exists, every transition result is labelled ENG.

### 4.8 Lockstep scheduler

```python
def run_lockstep(eng):
    while not eng.done():
        eng.ingest()                              # TX frames -> rings; control messages -> event heap (schema-validated)
        for grp in eng.rate_groups:
            j = grp.next_page; k0, k1 = j*grp.P, (j+1)*grp.P
            if not grp.has_active_rx(k0, k1): grp.skip_page(); continue          # D-6
            if not eng.tx_committed(grp, k1 + grp.h_fd):                         # §4.5 lookahead
                eng.wait_inputs(grp, timeout=eng.cfg.stall_timeout_s); continue
            limit = eng.barriers.limit(grp)       # = to_ticks(t_b + L_min, f) for the next unacked barrier (R3)
            seg_end = min(k1, limit)
            for seg in eng.segments(grp, grp.cursor, seg_end):   # split at events, snapshot boundaries
                eng.snapshots.ensure(seg.t_a, seg.t_b)            # geometry on the T_u grid
                eng.backend.render(eng.build_job(seg), grp.out[seg.slice])
            eng.post_rx(grp, grp.cursor, seg_end)  # noise, RX LO, front end, deliver, record, metrics
            grp.cursor = seg_end
            if seg_end == limit: eng.barriers.run_due(grp)        # in-process decide() or wait for remote ctrl.ack
```

- **TX commitment.** A TX stream is committed through sample K when any of these holds:
  - samples cover K;
  - its last burst ended with EOB (silence until its next timed burst);
  - it sent `idle_until(K)` (sremu extension; frame flag `IDLE_UNTIL`);
  - it is a raw REQ/REP peer whose replies cover K.
- **Stall.** After `stall_timeout_s` (default 2 s wall) the engine writes a deadlock report (which RX waits on which TX, which barriers are pending) and applies `on_stall: "wait" | "zero_fill"`. `zero_fill` counts `stall_zero_samples` and sets `nondeterministic_inputs`.

### 4.9 Controller in the loop

**Barriers.**
- A registered controller with period T_c creates barriers at `t_b = m·T_c`.
- The engine computes every RX up to `to_ticks(t_b + L_min, f_r)` (exclusive), where L_min is the controller's declared minimum latency (conservative-PDES lookahead).
- The `Observation` is built only from samples `< to_ticks(t_b, f_r)`.
- Actions default to `at = t_b + latency`, with latency drawn from `LatencyModel` and ≥ L_min. An action with `at < t_b + L_min` raises `LOOKAHEAD_VIOLATION`.
- Because the frontier limit and the command mapping both use `to_ticks`, which is monotone, and τ ≥ 0, lateness is 0 by construction (I8). The 0 ms vs 50 ms controller-sleep test checks this end to end.

**Remote controllers** use the barrier protocol over the control socket (§8.5): engine → `ctrl.barrier`, controller → zero or more `ris.set_config` → `ctrl.ack {t_b}`.

**Coupling-mode matrix** (resolves product's sync/async against runtime's lockstep/paced):

| Mode × coupling | Allowed? | Meaning |
|---|---|---|
| lockstep + barrier (sync) | **Default** | Deterministic; the τ_SUI and E-B studies are exactly reproducible |
| lockstep + unregistered timed commands | Allowed | Deterministic only if every command is on time; otherwise `nondeterministic_inputs` |
| paced + async | Allowed | Real wall-clock latency; lateness measured |
| paced + barrier | In-process controllers only | Barrier compute must fit within the page slack, or a deadline miss is counted |
| lockstep + async | **Rejected** | Non-deterministic by construction |

### 4.10 Paced mode (soft real time; opt-in)

- **Release.** Page j of a group is released at `R_j = t0 + k1/f + D_lag` (`D_lag` default 1 ms of TX input slack).
- **Sleep.** Hybrid: `sleep(R − now − 1.5 ms)`, then spin. Windows `time.sleep(0.5 ms)` overshoots by 0.51 ms at p50 and 0.88 ms at p99 [MEASURED], so paced numbers come from Linux.
- **Late TX data.**
  - Timed bursts are dropped, counted as `late_samples`, and reported with async `TIME_ERROR`.
  - Untimed streams get zeros and async `UNDERFLOW`; the write head slips to the frontier (`underflow_slip_samples`).
- **RX ring full** (client too slow): the page is dropped and the next `recv` returns `OVERFLOW` with the post-gap `time_spec` (UHD semantics). Counted as `overflow_pages`.
- **Deadline** = `R_j + slack`. Misses are reported at slack 0, 1 and 5 ms (5 ms is ACHEM's threshold).
- **Recorder.** The queue never blocks in paced mode; drops are counted as `rec_dropped`. In lockstep it blocks (backpressure).
- Controllers run asynchronously; their commands carry real lateness.

### 4.11 Batch (coefficient-only) path

- `engine/batch.py` evaluates `combine(snapshot(t), cfg(t))` on a time grid (default the auto T_u, or a fixed `dt_s`) and returns time-averaged powers and per-sample |h|². It uses the **same** PropagationModel, controllers, trajectories and RNG keys as the IQ engine.
- The command mapping uses the RIS-time convention; the ns-scale τ_RIS→r is negligible here and is logged.
- **Agreement contract.**
  - Noise off, paired trajectories: IQ time-averaged tone power vs batch ≤ 0.02 dB (W5, VA-1).
  - Noise on: within the 95% confidence interval (batch means).
  - τ_SUI → 0 with an in-process controller: equal to batch at 1e-9 (VA-4).

### 4.12 Replay

- `sremu replay RUN_DIR` re-injects the recorded TX SigMF streams and the commands from `events.jsonl` **at their recorded `t_applied`**, as on-time commands, with the resolved scenario and seeds.
- Acceptance (RT-16): RX bitwise equal to the recorded RX.
- This lets a reader reproduce a paced or hardware-in-the-loop run offline.

---

## 5. Physics model card

The source of truth for paper Table II is `docs/model_card.yaml`, which generates both the docs page and the LaTeX table.

### 5.1 Stack

```
Scene (nodes + analytic trajectories, antennas, RIS panels, ground, walls, occluders/partitions, knife edges)
  | snapshot(t): geometry in float64 at grid / event times
  v
PathBundle per link (LOS, ground image, order-1 walls, diffracted LOS)  +  RISCoefficients per panel (rows x states x cells, tile delays)
  | combine(snap, PanelConfig): gather / GEMV, linear in Γ
  v
LinkPaths  ->  render (§4.4–4.5): chord delay + exact carrier phase, tiles, transitions, FD  ->  + diffuse SoS  ->  + AWGN per RX
```

There are three time scales:
- geometry, O(R·Q) per snapshot;
- configuration, O(R·N) per command;
- samples, O(L·P·S).

### 5.2 LOS and antennas

- **LOS.** `h_LOS = √(G_T(r̂) G_R(−r̂)) · λ/(4πd) · V`. The propagation phase is applied by the renderer, so the path amplitude excludes `e^{−jkd}`.
- **V** is one of:
  - 1;
  - 0 for an explicit opaque occluder (segment–polygon test);
  - `10^{−L/20}` for an explicit **partition** with transmission loss L dB [ENG; used for the blocked-corridor experiment];
  - `F(ν)` for an explicit knife edge (§5.9).
- Scenario names never imply blockage.
- **Antennas.** `Isotropic`; `CosPower(α)` with `G_0 = 2(α+1)` and a boresight vector; `Tabulated(θ, φ, G_dBi)` with provenance. Scalar co-polar only.

### 5.3 RIS kernel K1 (the only physical kernel)

$$
h_{RIS}=\sum_{q}\sqrt{G_T(\hat r^{i}_q)\,G_R(-\hat r^{s}_q)}\;\Gamma_{n(q)}(s,f,\theta_{iq},\theta_{sq})\;\frac{A_q\,o(\theta_{iq},\theta_{sq})}{4\pi\, r_{iq}\, r_{sq}}\;j\,e^{-jk(r_{iq}+r_{sq})}\;\mathbb 1[\cos\theta_{iq}>0,\ \cos\theta_{sq}>0]
$$

| Symbol | Meaning | Unit |
|---|---|---|
| q | Quadrature point. One per physical cell for `cell_model="point"`; sub-cells for `"po_patch"` (§5.4) | — |
| n(q) | Parent physical cell, which carries the state and Γ | — |
| A_q | Pitch area of the quadrature cell, including gaps between cells; Σ_{q∈n} A_q = A_n | m² |
| r_iq, r_sq | Tx→cell and cell→Rx distances (f64) | m |
| θ_iq, θ_sq | Angles from the panel normal toward the Tx and the Rx (signed convention, §3.1) | rad |
| o(·) | Obliquity. Default `√(cos θ_i cos θ_s)` (Tang 2022's measured cos θ power pattern). Option `(cos θ_i + cos θ_s)/2` (Kirchhoff) | — |
| j | Huygens / physical-optics factor. Required: without it the phase relative to LOS is off by 90° | — |
| 𝟙[·] | Visibility: front-side incidence and reflection only | — |

**One model, three equivalent forms**, pinned at 1e-12 by PH-02 for `cell_model="point"`:
- Tang 2022 Eq. (12), the single-cell form without j;
- ETSI GR RIS 003 V1.2.1 §6.3.1.2 with the recommended cell gain `G = 4π·dx·dy/λ²`;
- Ellingson Eq. (10) with the element gain tied to the cell area, `G_e(ψ) = (4πA_n/λ²)cos ψ`.

**Guards.**
- A warning if `r < 3λ` (point-scatterer approximation); an error below λ.
- Back-side paths contribute exactly 0.
- Inter-tile gaps are explicit geometry: inert by default, or an optional structural Γ such as −1 for exposed ground plane. MATLAB showed that 2 mm gaps break the URA equivalence [FACT].

**Far-field closed form** (Tang Eq. 8): an accelerator only. It is allowed only when every node is beyond `2D²/λ`, and it is guarded by PH-03. The lab is near field: `2D²/λ` is 3.03 m for one 0.36 m tile and 27.3 m for three tiles, against Rx ranges of 0.45–1.95 m [FACT].

### 5.4 Cell model and aperture refinement (fixes physics D2)

`cell_model ∈ {"auto", "point", "po_patch"}` per panel; default `"auto"`:
- **`point`.** One quadrature point per physical cell. This is the model Tang validated and the one PH-02 equates.
- **`po_patch`.** Each physical cell is split into `⌈p_u/Δ⌉ × ⌈p_v/Δ⌉` sub-cells with `Δ ≤ λ/5` (`subcell_max_wavelengths = 0.2`), sharing the parent's Γ. This is a uniform-patch PO model.
- **`auto`** picks `point` when `max(p_u, p_v) ≤ 0.5λ` (`point_max_pitch_wavelengths`, recorded) and `po_patch` otherwise. Example: the lab's 5×8 tiles modelled as single uniform cells get `po_patch`; 42.8 mm cells at 3.5 GHz (λ/2 = 42.83 mm) get `point`.
- **The two models are not numerically equivalent.** For λ/2 cells at wide non-specular angles the sub-aperture factor differs by several dB: `sinc(p(sinθ_i + sinθ_s)/λ)` is −0.98 / −3.92 / −8.92 / −16.47 dB at 15 / 30 / 45 / 60° retro [PILOT, `fd_check.py`]. Near specular (`sinθ_i + sinθ_s ≈ 0`) they agree. The spread is **model-form uncertainty**: reported by PH-26 and sampled in the V-LAB prior.
- **Richardson check** (`po_patch` only): on 8 representative links (nearest, farthest, most oblique incidence, most oblique reflection, four random), compute P at Δ and at Δ/2. If max |ΔP| > 0.05 dB, halve Δ (at most twice), then raise `RefinementNotConverged`. The estimate goes to the manifest. For `point` the check is meaningless and is not run.
- Unsubdivided cells larger than λ/2 are forbidden unless `cell_model: "point"` is forced explicitly, which is logged with a warning.
- Tables with `includes_cell_pattern=True` (for example a single-cell bistatic measurement) force `point` and disable obliquity, so the pattern is not counted twice.
- Numbers [FACT T2]: 1 m² plate, 3.5 GHz, Γ = 1. Far specular at 200 m gives −115.28 dB for every N from 16 to 16384 under auto. Near anomalous at 5 m gives −88.10 dB at Δ ≤ λ/5. The legacy v2 kernel gains +12.04 dB per 4× refinement.

### 5.5 Per-state response tables

**Data model** (`physics/response.py`):
- `ResponseTable(labels[S], freqs_hz[F], theta_deg[T], gamma[S,F,T,T] complex128, provenance, validity_bw_hz)`.
- One θ grid is shared by the incidence and reflection axes. Degenerate shapes (S,1,1,1) and (S,F,1,1) are allowed; these are the angle-independent, separable tables.

**Load-time validators:**

| Rule | Criterion | On failure |
|---|---|---|
| Passivity | `|Γ| ≤ 1 + 1e−9` for analytic, full-wave, circuit or assumed tables; `≤ 1 + 3σ_mag` for measured or digitized ones | `PassivityError` |
| Reciprocity (angle tables only) | `max|Γ[s,f,i,j] − Γ[s,f,j,i]| ≤ tol`, with `tol = 1e−9` for σ = 0 sources, else `3·√2·σ_Γ` and `σ_Γ = |Γ|·√((ln10/20·σ_mag_db)² + (π/180·σ_phase_deg)²)` | `ReciprocityError`, unless `symmetrize="mean"`, which is logged in provenance and the manifest |
| Grid density | Adjacent `|ΔΓ|/max|Γ| ≤ 0.35` (≈ 20° at unit amplitude; complex-linear midpoint dip ≤ 0.134 dB) | `GridTooCoarse` |
| Domain | Query outside the f or θ grid | `OutOfDomain`. `extrapolate="nearest"` must be explicit and is recorded |
| Bandwidth | Declared occupied band of any linked TX stream exceeds `validity_bw_hz` | `BandwidthExceeded` (release 1 has no Γ FIR) |
| Off-plane | Off-plane angles are projected onto the (û, n̂) plane; the maximum is reported | `out_of_plane="error"` is available |

**Interpolation.** Complex-linear in (θ_i, θ_s) and linear in f. Never magnitude and unwrapped phase: that is ill-defined near |Γ| ≈ 0, and complex-linear keeps `combine` linear in Γ.

**Incidence-only tables** `g(θ)` are accepted only through the explicit reciprocal extension `Γ(θ_i, θ_s) := g(|θ_i − θ_s|/2)`, which gives `g(|θ_i|)` at specular. `angle_mapping: "half_difference"` is recorded in provenance [ENG]. Raw non-reciprocal tables (`reciprocal: false`) are rejected in release 1 (future work).

**Provenance** (mandatory): `source_kind` ∈ {measured, fullwave, circuit, analytic, digitized, assumed}; `description`; `reference`; `reference_plane_offset_m`; `polarization`; `includes_cell_pattern`; `sigma_mag_db`; `sigma_phase_deg` (scalar or per entry); `raw_sha256`; `license`; `created{date, tool, version}`; `assumptions[]`. `ResponseTable.sample(rng)` draws a perturbed table from the stated σ, for Monte Carlo.

**Built-ins:**

| Name | Content | source_kind |
|---|---|---|
| `lab_binary_3p5ghz` | σ1 = A1∠49.64°, σ2 = A2∠−153.77°, angle-independent. **Amplitudes must be set by the scenario (`assume_mag`)** and are recorded as ASSUMED. σ_phase = 15° from the 389°/mm etch sensitivity [FACT] | assumed |
| `tang2022_ris1_27ghz` / `tang2022_ris2_33ghz` | 0.9∠165°, 0.7∠0° / 0.8∠150°, 0.8∠0° (Tang 2022 Table III) | measured (published) |
| `ideal_bbit(b)` | `e^{j2πs/2^b}` | analytic |
| `ideal_antipodal` | {1, −1} | analytic |
| `on_off` | {0 (absorbing OFF), 1∠0°} (WCL letter) | analytic |
| `abeywickrama(b, β_min, φ, α)` | 2^b samples of `β(θ)e^{jθ}`, with `β(θ) = (1−β_min)((sin(θ−φ)+1)/2)^α + β_min` | analytic |

**Importers.**
- CSV long form `state,f_hz,theta_i_deg,theta_s_deg,mag_db,phase_deg[,sigma_mag_db,sigma_phase_deg]` plus a provenance JSON or YAML sidecar.
- MATLAB lossless S-curve CSVs (`unitcell_fullwave_*.csv`), phase only, labelled SIMULATED stand-in. The lossy infinite-array path is not trusted (non-passive |Γ| up to 12.5 [FACT]).

### 5.6 Non-ideality pipeline (normative order)

1. **Command.** `states[N_ctrl]` (int16), `phases_rad[N_ctrl]` or `codebook_index`. Granularity (`element | column | tile | groups`) expands N_ctrl to N physical cells.
2. **Quantizer** (continuous phases only): `s = argmax_s Re{Γ̄(s)·e^{−jφ}}`, with Γ̄ the table at the panel reference geometry (option: per link). Correct for unequal amplitudes.
3. **Failures.** `FailureMap(stuck_at[N], dead[N])`; dead → state −1 → `gamma_dead` (default 0, absorbing; option structural). `FailureMap.random(rate, mode, key)` draws from the keyed RNG.
4. **Table lookup** → `B[r, s, n]` at snapshot time.
5. **Efficiency.** Amplitude scaled by `√η_eff`. Refused for `source_kind ∈ {measured, digitized}` unless `override_efficiency: true`, because the measurement already contains the loss.
6. **Coupling hook** (off). `CouplingHook.delta(state) → (N,) complex` adds `Σ_n C[r,n]·δΓ_n`; separable tables only. **No κ values ship.** The paper says coupling enters only through measured or simulated per-configuration tables (ETSI GR RIS 003 §5.1.3), and mutual impedance is future work.
7. **Transition** in time (§4.7).

**Analytic oracles carried in `nonideal.py`:**
- **Large-N binary-alphabet loss.** With equal amplitudes, uniform channel phases and the per-element best choice, `η = (2 sin(Δ′/2)/π)²` with `Δ′ = min(Δ, 2π−Δ)`.
  - Δ = π: −3.922 dB.
  - Lab pair (Δ = 203.41°): −4.105 dB, i.e. 0.18 dB worse than ideal 1-bit.
  - Tang RIS1 / RIS2 alphabets (quadrature): −5.93 / −6.16 dB.
  - Uniform b-bit: `η(b) = (2^b/π·sin(π/2^b))²` = −3.922 / −0.912 / −0.224 dB.
- **Binary m/d decomposition.** `Γ = m + d·s`, `s ∈ {±1}`, `m = (a+b)/2`, `d = (a−b)/2`. Only d is steerable; m adds a fixed mirror path `m·Σ_n c_n`.

  | Alphabet | |d| (dB) | |m| (dB) |
  |---|---|---|
  | Lab pair, |Γ| = 1 | −0.18 | −13.86 ∠127.9° |
  | Lab, |Γ2| = 0.7 | −1.59 | −12.90 |
  | Lab, |Γ2| = 0.5 | −2.66 | −10.81 |
  | Ideal ON/OFF | −6.02 | −6.02 |

  [DERIVED]. Cite prior non-ideal and discrete-phase RIS work; this is not claimed as new.

### 5.7 Ground

- **Material.** `ε_c = ε_r − j60σλ`.
- **Fresnel coefficients**, grazing angle ψ:
  - `Γ_H = (sin ψ − √(ε_c − cos²ψ))/(sin ψ + √(ε_c − cos²ψ))`
  - `Γ_V = (ε_c sin ψ − √(ε_c − cos²ψ))/(ε_c sin ψ + √(ε_c − cos²ψ))`
- **Image path.** The source is mirrored in the ground plane; antenna gains are taken toward the image direction. Polarization (H or V) is declared per scenario.
- **Image expansion on RIS hops.** The kernel is evaluated for each (Tx or Tx-image, Rx or Rx-image) pair, weighted by the product of the ground Γ factors at the respective specular points. These are RIS rows with hop variants 0–3 (§3.4).
- Two-ray at 52 km, h = 200/1.5 m, 100 MHz, Γ = −1: 139.10 dB [FACT].

### 5.8 Walls (SHOULD)

- Finite planar polygons with (ε_r, σ). Material values come from Rec. ITU-R P.2040 Table 3 and are UNVERIFIED until transcribed with citation.
- **Image order 1 only.** A path is valid only if its specular point lies inside the polygon.
- One Fresnel coefficient per bounce; optional Rayleigh roughness factor `exp(−2(kσ_h sin ψ)²)`.
- Order 2 and RIS–wall paths are cut.

### 5.9 Occluders, partitions, knife edge; P.526

- **Occluder** (opaque): convex polygon; the segment–polygon test sets V = 0.
- **Partition:** polygon with `transmission_loss_db`; V = `10^{−L/20}` with no phase change [ENG].
- **Knife edge:**
  - `ν = h·√(2(d_1+d_2)/(λ d_1 d_2))`;
  - `F(ν) = ((1+j)/2)·∫_ν^∞ e^{−jπt²/2} dt` via `scipy.special.fresnel`;
  - the diffracted field `h_LOS(d)·F(ν)` is phase-consistent and sums coherently with RIS paths.
  - The ITU-R P.526 J(ν) is an **oracle only**: max 0.123 dB from exact on [−0.78, 10] [FACT].
- **One edge per path in release 1.** Deygout is cut.
- `physics/experimental/p526.py` ports `pm_checks.py::p526_smooth` (EXPERIMENTAL). It is not imported by the core and appears only as a labelled sensitivity row in E-D, with a regression test (PH-12).

### 5.10 Mobility (analytic functions of continuous time; positions never depend on any grid)

| Model | Definition |
|---|---|
| `static` | Fixed position |
| `waypoints` | Piecewise linear with timestamps |
| `constant_velocity` | Start position plus velocity |
| `rwp_1d` / `rwp_2d` | Random waypoint (Bettstetter): legs generated per leg index from keyed RNG purpose `mobility/<node>`, constant speed, optional pause |
| `trace` | CSV (t, x, y, z), cubic spline (C²), no extrapolation |
| `external` | Positions arrive as timed `node.set_position` / `node.set_trajectory` events |

Gauss–Markov and random walk are cut: they are τ-dependent in v1/v2 [FACT].

### 5.11 Diffuse multipath (SHOULD)

- An **additive** stochastic term per link: Clarke/Jakes sum of sinusoids, ≥ 48 sinusoids, with keyed amplitudes and phases. It is a pure function of absolute time and satisfies chunk and skip invariance.
- Optional exponential PDP of up to 4 taps, each with an independent SoS.
- Power is set by a K-factor relative to the deterministic paths.
- It **never multiplies** the deterministic channel. v2's composite multiplier made "RIS gain" fading-invariant by construction [FACT].
- TR 38.901 TDL profiles are background only.

### 5.12 Noise

- AWGN per **receiver**, `CN(0, k_B T_0 F f_s)` W, F from `noise_figure_db`. This fixes v2's 3.01 dB error [FACT].
- Generated by `impair/noise.py`: Philox4x32-10 plus Box–Muller, keyed per §6.1.

### 5.13 Exact finite-alphabet optimum (`physics/optimize.py`)

`exact_alphabet_optimum(z0, Z[S,N]) → (value, s*)` maximizes `|z0 + Σ_n Z[s_n, n]|` exactly.
- **Principle.** For a direction θ, the best per-element choice is `s_n(θ) = argmax_s Re{Z[s,n]e^{−jθ}}`. It changes only at the critical angles `arg(Z_a − Z_b) ± π/2` (S(S−1)N of them).
- **Proof sketch.** The optimum `s*` equals `s(θ*)` with `θ* = arg(total)`; otherwise switching an element would raise the projection onto θ*, and with it |total|.
- **Algorithm.** Sweep the sorted critical angles with incremental updates, O(S²N log(S²N)). A reference O(S²N²) evaluation is kept for tests.
- **Uses.** Controller upper bound and E-C. Checked against brute force (PH-24, PH-29). Physics measured 0 mismatches in 200 trials [FACT].
- **Not claimed as novel.** Cite prior optimal discrete-phase beamforming work (the WCL letter's own ref. [6], Zhao et al., IEEE WCL 13(2) 2024, and the linear-time optimal discrete beamforming literature). Both are [UNVERIFIED]; WP12 verifies them before citing.
- `continuous_optimum`: phases `−arg(c_n) + arg(h_0)` (closed form).

### 5.14 Legacy kernels and teeth (`physics/legacy.py`, `legacy/`)

Every legacy item raises unless called with `allow_nonphysical=True` or an explicit adapter.

| Item | Purpose | Teeth (it must FAIL the guard) |
|---|---|---|
| `legacy_v1` | Whole-panel area per cell, λ factor, `abs(cos)` | +24.4 dB vs K1 in the v1 default geometry |
| `legacy_v2` | `(λ/4π)²√(cos cos)/(r r)` | +12.04 ± 0.01 dB per 4× refinement |
| `legacy_wcl_eq2` | The letter's Eq. (2)/(3), q = 0.285 | Ratio to K1 is `(πλ)²(cos θ_i cos θ_s)^{2q−1}`: −0.52 / −8.1 / −11.4 dB at 1 / 2.4 / 3.5 GHz |
| `ellingson_qgamma` | Ellingson Eq. (10) with (q, γ) | — |
| `no_j` | K1 without j | −90 ± 3° phase vs image theory |
| `block_constant` Doppler | v2 staircase | −148.3 Hz alias instead of +350.2 Hz |
| `LegacyTruncationAdapter` | `int(f_s·τ)` per tick, CFO phase advanced by a full τ | CFO slope 1.0691 ± 0.0005 |
| `LegacyNextTickAdapter` | v2 RIS semantics: single-slot overwrite, applied at the next τ tick | Loses the first of two commands (v2 N17); used as the E-B baseline |

### 5.15 Validity domain (model card)

| Component | Model | Validity domain | Evidence | Default |
|---|---|---|---|---|
| LOS | Friis + patterns + explicit V | Far field of the antennas | PH-01 | On |
| RIS | K1, `point` | Cell in the far field of Tx and Rx (r ≥ 3λ); pitch ≤ λ/2; scalar co-pol | PH-02–07, G1–G3, V-TANG | auto |
| RIS | K1, `po_patch` | As above; uniform-patch PO; Richardson-converged | PH-04, PH-26 | Pitch > λ/2 |
| Γ tables | Complex-linear LUT | Inside the grid; band ≤ validity bandwidth; reciprocal | PH-28, PH-31 | — |
| Ground | Flat-earth image + Fresnel | Smooth, flat; no curvature (P.526 experimental) | PH-10, PH-32, G8 | Off |
| Walls | Image, order 1 | Finite polygons; specular | PH-08 | Off |
| Diffraction | Single knife edge F(ν) | One dominant edge | PH-11 | Off |
| Diffuse | Additive SoS | Statistical background only | ST-02, G6 | Off |
| Mobility/Doppler | Chord delay + exact phase + tiles | ε_rel ≤ 1e-3 (bound plus `verify_interp`) | PH-17–19 | Auto |
| Wideband | FD Kaiser 8/16/32 | B ≤ 0.8 f_s; ≤ −60 dB in band | PH-20 | Auto |
| Noise | `kT_0 F f_s` per RX | — | PH-21, G5 | On |
| Non-idealities | Pipeline (§5.6) | Tables, failures, efficiency; coupling via tables only | PH-13–15, 25, 30 | — |
| Transitions | Per-class g_c, complex-linear Γ path | [ENG] until measured | RT-12 | Step |

---

## 6. Impairments and RNG

### 6.1 Random number generation (`rng.py`)

- **Key.** The first 64 bits of `blake2b(master_seed ‖ purpose ‖ name)` (UTF-8, `‖` = 0x1F separator). They are split into two uint32 Philox key words.
- **Counter** (Philox4x32-10): `(k_lo, k_hi, sub, 0)`, where k is the **absolute** sample or step index and sub is a purpose-specific lane. Two outputs feed Box–Muller for one complex normal; the rest are reserved.
- **Starting point.** The `research/scratch/rt_micro2.py` prototype: 72–78 µs per 1 ms of 23.04 MS/s on 16 threads, variance 0.99961, a 370 + 630 split bitwise equal to one 1000-sample call [MEASURED].
- A NumPy twin (vectorized uint64 emulation of mulhi) MUST produce bitwise-identical uint32 streams (I5).
- **Controllers and analysis code** get `np.random.Generator(np.random.Philox(key=derive(seed, "ctrl/<name>")))`. There is no global RNG anywhere; a grep test bans `np.random.seed` and the legacy `np.random.*` functions.
- **Monte Carlo trials** use `purpose="trial"`, `name=f"{study}/{trial}"`.
- `SeedSequence.spawn` is **not** used, because it depends on spawn order.

### 6.2 Purpose registry (fixed strings; versioned in `rng.py`)

`noise/<rx_stream>`, `pn/<node>.<lo>`, `diffuse/<link>`, `mobility/<node>`, `failure/<panel>`, `table_mc/<panel>`, `ctrl.latency/<ctrl>`, `ctrl/<ctrl>`, `trial/<study>/<i>`, `decoy/<i>`, `traffic/<stream>`.

### 6.3 Impairments in release 1

Process classes (from the runtime doc): **(a)** pure function of absolute index or time; **(b)** hierarchical random access; **(c)** sequential with checkpointed state on an absolute grid.

| Impairment | Class | Model | Tier |
|---|---|---|---|
| AWGN | (a) | §5.12 | MUST |
| TX and RX LO CFO | (a) | `φ(k) = 2π·frac(Δf·k/f_s)`, page anchor × 1024-entry table | MUST |
| Wiener phase noise | (b) | Linewidth β: page increments `N(0, 2πβP/f_s)` keyed by page index; Brownian bridge inside the page; variance `2πβt`, independent of f_s (fixes v2 P4) | SHOULD |
| Diffuse SoS | (a) | §5.11 | SHOULD |
| Front end | (a) | Full scale, gain, optional clip, sc16 saturation | MUST |
| SFO, IQ imbalance, PA, ADC quantization | — | Cut (future work) | — |

**Chains.**
- RX: Σ links → diffuse → thermal noise → RX LO (CFO, PN) → gain/full scale → optional clip → deliver.
- TX: full scale/gain → TX LO (CFO, PN) → ring buffer.

---

## 7. Backends and performance targets

### 7.1 Backends (release 1)

| Backend | Geometry | IQ application | Role |
|---|---|---|---|
| `numpy_ref` | f64 | Direct per-sample sincos, complex128 accumulate | Oracle for everything; all tests run on it |
| `numba_cpu` (default) | f64, numba `prange` over rows | Rotators with anchors (D-2), polyphase FD, `prange` over samples, `nogil` | Default IQ path |
| `torch_cuda_geometry` | f32-differential (`r = r_ref(f64) + Δr(f32)`) | — (CPU renders) | STRETCH behind `--gpu-geometry`; accept ≤ 1e-4 rad and ≤ 2e-5 relative vs f64. The naive-f32 ablation must show ≈ 0.40 rad at 52 km [MEASURED] |

Cut: hybrid/resident GPU IQ, CUDA graphs, bitwise GPU Philox, `VectorRisEnv`. The precision rule stays documented as future work.

### 7.2 Precision rules

- Absolute path length is never computed in f32.
- Phase uses reduced cycles (D-1).
- complex128 accumulators; complex64 I/O.
- Delays are f64 seconds; the integer and fractional split is computed per sample from f64 `τ·f_s`. For delays above 1 s, an exact Fraction anchor per snapshot is used (guard).

### 7.3 Performance targets [TARGET; the paper reports measured values whatever they are]

| ID | Workload | Target | Basis |
|---|---|---|---|
| P-1 | Lab-like link (1 TX, 1 RX, 1 RIS ≤ 240 cells, static, 1 MS/s, narrowband, noise on), lockstep, laptop, 16 threads | RTF_min ≥ 5 | Python dispatch ≈ 30–100 µs per page × 2000 pages/s [UNVERIFIED per-call cost] |
| P-2 | Same link at 23.04 MS/s, B = 18.4 MHz (wideband, 2 FD groups × 32 taps) | RTF_min ≥ 2 | 32-tap FIR 493 µs per ms single thread [MEASURED], parallel over samples |
| P-3 | Configuration change (GEMV) 64 × 900 | ≤ 0.1 ms | 0.085 ms NumPy [MEASURED] |
| P-4 | Paced, 1 MS/s, 1 link, c6i.8xlarge Linux | Miss rate < 1e-3 at slack 1 ms over 60 s | — |
| P-5 | Paced, 1 MS/s, many links, c6i.8xlarge | Largest L with miss rate < 1e-3 reported; goal ≥ 32 | Runtime doc estimate (≥ 64) halved for safety |
| P-6 | Native ZMQ transport, TCP loopback | Ceiling reported; ≈ 322 MS/s cf32 | [MEASURED] |
| P-7 | Control round trip (REQ/REP small JSON) | p99 < 1 ms on Linux | 81 µs p50 on the loaded laptop [MEASURED] |

### 7.4 Cost model (fitted in B1, reported in the paper for RQ-43)

`T_page = a + b·(R·Q)·(P/(f_s T_u)) + c·L·T_FD·P + d·n_rx·P + e·n_tiles·P`

The paper reports fitted coefficients and exponents against O(L·N) per snapshot and O(L·T·S) per sample.

---

## 8. Interfaces

### 8.1 Python API (UHD-shaped, not UHD-pretending) — `api/`

```python
dev = sremu.Device.inproc("scenarios/lab_like_A.yaml", mode="lockstep", backend="numba_cpu", record="runs/r001")
# or: dev = sremu.Device.connect("tcp://127.0.0.1:5600")
tx = dev.get_tx_stream(StreamArgs(stream="ap.tx0", fc_hz=3.5e9, rate_sps="1e6", cpu="fc32"))
rx = dev.get_rx_stream(StreamArgs(stream="rx1.rx0", fc_hz=3.5e9, rate_sps="1e6", cpu="fc32"))
n = tx.send(buf, TxMetadata(has_time_spec=True, time_spec=TimeSpec(0, 1e-3), start_of_burst=True, end_of_burst=True), timeout=0.1)
rx.issue_stream_cmd(StreamCmd(mode="num_done", num_samps=10_000, stream_now=False, time_spec=TimeSpec(0, 1e-3)))
n = rx.recv(out, md := RxMetadata(), timeout=1.0)       # md.time_spec of first sample, md.error_code, md.end_of_burst
dev.ris("ris0").set_config(states=s, time_spec=TimeSpec(0, 5e-3), transition=Linear("2e-5"), late_policy="apply_on_arrival")
dev.set_command_time(TimeSpec(0, 8e-3)); dev.node("rx1").set_position([1.2, 0.3, 0.0]); dev.clear_command_time()
tx.idle_until(TimeSpec(1, 0.0))                          # sremu extension for lockstep TX commitment
msg = tx.recv_async_msg(timeout=0.0)                     # AsyncMetadata(event_code=BURST_ACK|UNDERFLOW|TIME_ERROR|SEQ_ERROR|...)
```

- **Error codes** mirror `uhd::rx_metadata_t`: NONE, TIMEOUT, LATE_COMMAND, BROKEN_CHAIN, OVERFLOW, ALIGNMENT, BAD_PACKET. sremu adds SCHEMA_ERROR, LOOKAHEAD_VIOLATION, UNKNOWN_TARGET, OUT_OF_DOMAIN, TOO_LARGE, NOT_PERMITTED, RATE_MISMATCH.
- **Async codes:** BURST_ACK, UNDERFLOW, SEQ_ERROR, TIME_ERROR, UNDERFLOW_IN_PACKET, SEQ_ERROR_IN_BURST.
- **Stream modes:** `start_cont`, `stop_cont`, `num_done`, `num_more`. Untimed calls start at the frontier.
- `nchan` is carried everywhere; values > 1 raise.

**Paper Table III** (generated from `api/metadata.py` docstrings): sremu ↔ UHD (`tx_streamer::send`, `rx_streamer::recv`, `tx/rx_metadata_t`, `set_command_time`) ↔ srsRAN ZMQ device args ↔ GNU Radio ZMQ blocks.

### 8.2 Compatibility wrappers (`compat.py`)

- `Send_to_sim(node, samples, fc, fs, gain=0, start_time=None)` and `Receive_from_sim(node, nsamps, fc, fs, gain=0, start_time=None)` are thin aliases with the ANTS names.
- `send_to_simulator` and `recieve_from_simulator` (v1 spelling) exist with a `DeprecationWarning`.
- No `MultiUSRP` signature mirroring (cut).
- `examples/bpsk_txrx/` ports the v1 BPSK TX/RX pair (5880 S/s, 200-symbol header of 1+1j, 38 BPSK symbols at ±0.1(1+j), 10 samples/symbol).

### 8.3 Native ZMQ data plane (`transport/zmq_native.py`, `framing.py`)

- **Sockets.**
  - TX ingress: one `PULL` bound at `tcp://127.0.0.1:5602`; clients `PUSH`.
  - RX egress: one `PUSH` per RX stream, bound to an ephemeral localhost port returned by `stream.open`; clients `PULL`.
  - Messages are multipart `[header, payload]` (zero-copy payload).
- **Header** (little-endian, 48 bytes, `struct "<IHHIIqIHHQQ"`):

```
uint32 magic        bytes b"SRMU" on the wire (uint32 value 0x554D5253)
uint16 version      1
uint16 flags        b0 SOB, b1 EOB, b2 HAS_TIME, b3 IDLE_UNTIL (empty payload; first_sample = idle-until index),
                    b4 OVERFLOW_BEFORE (RX gap precedes), b5 LATE, b6 END_OF_STREAM
uint32 stream_id    from stream.open
uint32 seq          per-stream frame counter; a gap -> SEQ_ERROR async
int64  first_sample absolute index at stream rate (TX: valid iff HAS_TIME, else frontier; RX: always)
uint32 nsamps       <= 2^22
uint16 nchan        == 1
uint16 dtype        1 cf32_le, 2 sc16_le
uint64 wall_ns      sender monotonic clock; metrics only, never data plane
uint64 reserved     0
```

- **Validation.** Bad magic, version, dtype, nchan or size, or a payload length mismatch → the frame is dropped and counted (`BAD_PACKET`). The engine never raises on bad input.
- **Backpressure.** TX `PUSH` blocks at HWM, giving lockstep backpressure. In paced mode RX sends with `NOBLOCK`; `EAGAIN` → `OVERFLOW_BEFORE` on the next frame.
- PUB/SUB never carries IQ (it drops at HWM).

### 8.4 Raw REQ/REP mode for srsRAN / GNU Radio (feasibility, gated) — `transport/rawreqrep.py`

- **Per peer** (device args `tx_port`, `rx_port`, `base_srate`), the emulator acts like the GNU Radio broker:
  - it **connects a REQ** to the peer's `tx_port` (REP) to pull TX samples;
  - it **binds a REP** at the peer's `rx_port` and answers RX requests with raw samples.
- No header: timing is implicit in sample counts, so the mode is **lockstep only**.
- **Per-peer option** `request_format: "srsran_dummy" | "gnuradio_count"`. srsRAN appears to send a dummy byte; gr-zeromq `req_source` sends an item count. Both are **[UNVERIFIED]**. WP8 MUST pin them from the source copies in `research/scratch/src_refs/` (srsRAN_4G `rf_zmq_imp_rx.c`, `rf_zmq_imp_tx.c`; gr-zeromq `rep_sink_impl.cc`, `req_source_impl.cc`) and add a stub-peer test (IO-08) for each format before any claim.
- **Wording** until IO-08 and E2 pass: "srsRAN-style REQ/REP raw-IQ mode", never "wire-compatible".
- srsRAN Project is archived (moved to OCUDU, Dec 2025); srsRAN 4G is the target for E3.

### 8.5 Control plane (`transport/control_server.py`, `transport/schema/*.json`)

**Sockets.**
- `ROUTER` at `tcp://127.0.0.1:5600`: requests and replies; registered controllers use `DEALER`.
- `PUB` at `:5601`: telemetry topics `ris.applied`, `cmd.late`, `cmd.rejected`, `stream.underflow`, `stream.overflow`, `metrics.window`, `sim.stall`.
- Barrier traffic goes over ROUTER → DEALER, **never PUB**, because PUB may drop.

**Envelope** (JSON, schema version 1, validated before dispatch):

```json
{"v":1,"id":"c-7f3a","op":"ris.set_config","src":"ctrl-a",
 "panel":"ris0","at":{"t":"5/1000"},
 "config":{"states_b64":"AQEAAQ...","dtype":"i2"},
 "elements":"all",
 "transition":{"model":"linear","duration_s":"2e-5"},
 "late_policy":"apply_on_arrival"}
```

- `at` alternatives: `{"sample":{"stream":"rx1.rx0","index":115200}}` or `"asap"`.
- `config` alternatives: `{"codebook_index":17}` or `{"phases_deg":[...],"quantizer":"max_real"}`.
- `elements` alternatives: `{"indices":[...]}` or `{"group":"tile2"}`.
- Success reply: `{"v":1,"id":"c-7f3a","ok":true,"status":"scheduled","seq":412,"t_req":"1/200","k_req":{"rx1.rx0":5000},"t_wall_recv_ns":...}`.
- Error reply: `{"v":1,"id":...,"ok":false,"error":{"code":"LATE_COMMAND","msg":"..."}}`.
- Every command later produces a `ris.applied` PUB message carrying its `CommandRecord`.

**Operations.**

| Op | Purpose |
|---|---|
| `hello` | Version and capability negotiation (`contract_version`) |
| `caps.get` | `RisCapabilities` per panel |
| `time.now` | Frontier time per stream |
| `ris.set_config` | Timed RIS configuration |
| `ris.set_codebook` | Register a named codebook (K × N_ctrl) |
| `ris.get_state` | Current applied and pending configuration |
| `ris.set_enabled` | Counterfactual on/off (logged as such) |
| `node.set_position`, `node.set_trajectory` | Timed geometry events |
| `stream.open`, `stream.close`, `stream.cmd` | Data plane |
| `ctrl.register` | `{name, period_s, min_latency_s, latency, observation}` |
| `ctrl.barrier` | Engine → controller: `{t_b, k_b, observation}` |
| `ctrl.ack` | Controller → engine: `{t_b}` |
| `ctrl.unregister` | — |
| `rec.marker` | Annotation in the recording |
| `stats.get` | Counters and metrics |
| `sim.pause`, `sim.resume`, `sim.shutdown` | Session control |

**`RisCapabilities`** (ETSI GS RIS 008 parameter names where they exist): `n_elements`, `layout`, `control_granularity`, `states`, `gamma_source` (`"measured:<sha256>"` or `"analytic:<model>"`), `operating_band_hz`, `max_bandwidth_hz`, `reconfiguration_time_s`, `max_command_rate_hz`, `static_power_w` (metadata only).

**Limits.** Message ≤ 1 MB; ≤ 65,536 elements per command; ≤ `max_command_rate_hz` per client (default 10 kHz), with excess rejected and counted; **no file paths in any message**.

### 8.6 Recording, event log, replay, report

- **SigMF.** One recording per stream (TX and RX), written as plain JSON (no sigmf dependency):
  - global: `core:datatype "cf32_le"`, `core:sample_rate`, `core:version` (the SigMF spec version at implementation time, v1.2.x; [UNVERIFIED] exact string), `core:extensions [{"name":"sremu","version":"1.0.0","optional":true}]`;
  - captures: `core:sample_start`, `core:global_index`, `core:frequency`, `core:datetime`;
  - annotations: one per applied event, with `core:sample_start = k_applied`, `core:label = "ris_cfg"`, `sremu:cmd_id`, `sremu:k_req`, `sremu:lateness`, `sremu:config_hash`, `sremu:status`.
  - A `.sigmf-collection` lists the meta files with SHA-512.
- **Event and metrics logs.** `events.jsonl` (canonical, append-only, flushed per record); `metrics.jsonl` (per 100 ms window).
- Replay per §4.12.
- `sremu report`: static HTML with inline SVG plots, the event table and the manifest. No server, no JavaScript.

### 8.7 Controller plugin API (`control/`)

- **Protocol** as in §3.4. In-process controllers run inside the engine thread at barriers (deterministic). `control/remote.py::run_remote(ctrl, "tcp://...")` wraps the same class for an external process.
- **Observation.** `rx_power_dbm` is measured from real RX IQ over `window_s` before t_b. `csi_oracle` (C[R,N] and direct paths) is available but its use is logged (`oracle_csi_used`).

**Built-ins** (`control/builtin.py`):

| Controller | Feedback | Behaviour |
|---|---|---|
| `Fixed(states)` | — | Constant configuration |
| `RisAbsent()` | — | Sets `enabled = False` (baseline) |
| `Urap(p)` | — | WCL uniform random activation, redrawn each period (`on_off` alphabet; for other alphabets, state 1 with probability p) |
| `WclHeuristic(L, M)` | Positions | Letter Eqs. (7)–(8); default L = 4 Θ values {π/8, π/4, 3π/8, π/2}, M = 3 ψ values, K = L·M(M+1)/2 = 24 |
| `ExactOptimum()` | Positions + model | §5.13 on the controller's own model of the scene (oracle geometry flagged) |
| `GreedyQuantized()` | Positions | Phase-align, then quantize (§5.6) |
| `IdealDesigned(target="antipodal")` | Positions | Designs for {1, −1}, deploys on the true states |
| `CodebookSweep(K)` | RX power | Cyclic sweep |
| `ProbeBandit(codebook=16, dwell_s="2e-2")` | RX power (IQ) | UCB1 over windowed power (SHOULD) |
| `GeneticAlgorithm(pop=300, mut=0.01, iters=150)` | Positions | Only for W4 |
| `TdmaAlternate(targets, timing="timed"\|"asap"\|"legacy", lead_s)` | Packet schedule | Alternates per-packet exact-optimum configurations between targets, scheduling switches in guards (E-B2) |

`DelayedController(inner, tau_sui_s, report_delay_s)` wraps any position-driven controller.

### 8.8 Gym-style environment (`ml/env.py`)

- `RisEnv(scenario, panel="ris0", decision_period_s="1e-3", action="states"|"codebook", reward="rx_power_dbm", traffic="pilot_tone", backend="numba_cpu"|"batch")` with `reset(seed) → (obs, info)` and `step(a) → (obs, r, terminated, truncated, info)`.
- It subclasses `gymnasium.Env` if gymnasium imports and duck-types otherwise. `exact_alphabet_optimum` provides the regret reference.
- No vectorized or GPU environment in release 1.

### 8.9 Standards mapping (paper Table VI; wording "aligned with", never "compliant")

| Document | Clause or parameter | sremu element |
|---|---|---|
| ETSI GR RIS 002 V1.2.1 | Architecture roles: RIS controller, RIS actuator, control link | External controller client; engine panel object; `LatencyModel` + transport |
| ETSI GR RIS 003 V1.2.1 | §5.1.2 Γ definition (locally periodic) | ResponseTable semantics |
| | §5.1.3 mutually coupled antenna model | Out of scope except through tables (stated) |
| | §6.3.1.2 path-loss model with G = 4π·dx·dy/λ² | K1 (PH-02) |
| | §6.6 multi-mode / PO integral (j factor) | K1 j; `po_patch` |
| ETSI GS RIS 008 V1.1.1 (Feb 2026) | Operating range; maximum bandwidth | Table f axis; `validity_bw_hz`, `max_bandwidth_hz` |
| | Bistatic RCS; reflected-beam spatial characteristics | (θ_i, θ_s) table axes; G2/G3 beam checks |
| | Total reflected power | Energy diagnostic (PH-23) |
| | Modulated-signal quality | E-B EVM/PER |
| | Reconfiguration time | Transition default `reconfiguration_time_s`; E-B |
| | Static power | Metadata only |
| TSDSI STD 5003 | — | One sentence: "The text of TSDSI STD 5003 was not accessed; no alignment is claimed." A mapping is added only if the text is obtained |
| ORIX E2SM-RIS | — | Future work (adapter cut) |

### 8.10 Security

- Every socket binds to `127.0.0.1` by default. `--allow-remote` is required for anything else, and `SECURITY.md` states there is no authentication (optional ZMQ CURVE is future work).
- No web server. No filesystem paths in messages. Schema validation and size caps on every message. Fuzz test IO-01.

---

## 9. Scenario schema

### 9.1 Rules

- pydantic v2 models in `scenario/models.py`. **YAML** is the primary format; JSON and TOML load through the same models. `sremu schema scenario` emits JSON Schema; `docs/scenario_reference.md` is generated from it.
- `extra="forbid"`. Units in field names. Exact quantities as strings (§3.1).
- Every default is recorded in `scenario.resolved.yaml`.

### 9.2 Model tree (field names are frozen by this spec)

```
Scenario: schema_version "1.0", name, seed:int, description
  numerics: page_target_s="5e-4", interp_rel_error=1e-3, t_u_max_s="1e-2", t_u_s?:str (may only lower auto),
            fd_error_target_db=-60, narrowband_phase_rad=0.05, point_max_pitch_wavelengths=0.5,
            subcell_max_wavelengths=0.2, richardson_tol_db=0.05, deterministic=true,
            backend="numba_cpu"|"numpy_ref", verify_interp=true
  environment:
    ground?: {eps_r, sigma_s_per_m, polarization: H|V, height_m=0}
    walls: [{id, vertices_m:[[x,y,z],...], eps_r, sigma_s_per_m, roughness_m=0}]
    occluders: [{id, vertices_m, kind: opaque|partition, transmission_loss_db?}]
    edges: [{id, p0_m, p1_m}]                     # knife edge as a segment; applies to the paths it intersects
    diffuse?: {k_factor_db, max_doppler_hz | from_motion: true, n_sinusoids=48, taps?: [{delay_s, rel_power_db}]}
  nodes: [{id,
           position_m?:[x,y,z] | trajectory?: {model: static|waypoints|constant_velocity|rwp_1d|rwp_2d|trace|external, ...},
           antenna: {pattern: isotropic | {cos_power: {alpha}} | {tabulated: {file, provenance}}, boresight?:[x,y,z]},
           streams: [{name, direction: tx|rx, fc_hz, rate_sps:str, occupied_bw_hz?, full_scale_dbm=0, gain_db=0,
                      noise_figure_db=0 (rx), lo: {cfo_hz=0, linewidth_hz=0}, adc_clip=false}]}]
  ris: [{id, enabled=true, pose: {center_m, normal, u_axis},
         grid: {n_u, n_v, pitch_u_m, pitch_v_m, cell_model: auto|point|po_patch},
         tiles?: {n_tiles_u, n_tiles_v, gap_m, gap_gamma?: complex|null},
         states: {library: <builtin> | file: <path>, assume_mag?: {state: mag}, efficiency=1.0, override_efficiency=false},
         control: {granularity: element|column|tile|groups, groups?: {...}},
         initial: {fill: int | states: [...]},
         failures?: {dead_rate=0, stuck_rate=0, gamma_dead=0},
         transition: {model: step|linear|first_order, duration_s?:str, classes?: by_edge|single},
         capabilities: {reconfiguration_time_s?, max_command_rate_hz=1e4, static_power_w?}}]
  controllers: [{name, kind: builtin|external, impl?, period_s:str, min_latency_s:str,
                 latency: {fixed_s:str, jitter?}, observation: {...}, params: {...}}]
  run: {mode: lockstep|paced, coupling: barrier|async, duration_s:str, late_policy: apply_on_arrival|reject,
        on_stall: wait|zero_fill, stall_timeout_s=2, paced: {d_lag_s="1e-3", slacks_s:["0","1e-3","5e-3"]}}
  record: {iq: false, events: true, metrics: true, snapshots: false}
  traffic?: [{stream, generator: tone|bpsk|qpsk_packets|file, params}]   # in-process generators for studies
```

### 9.3 Example (`scenarios/lab_like_A.yaml`; SIM geometry, not the lab's)

```yaml
schema_version: "1.0"
name: lab_like_A_eb
seed: 20261010
numerics: {backend: numba_cpu}
nodes:
  - id: ap
    position_m: [1.7320508, -1.0, 0.0]          # 2.0 m at -30 deg from the panel normal (+x)
    antenna: {pattern: isotropic}
    streams: [{name: tx0, direction: tx, fc_hz: 3.5e9, rate_sps: "1e6", occupied_bw_hz: 8.0e5, full_scale_dbm: 0}]
  - id: rx1
    position_m: [1.1591110, 0.3105829, 0.0]     # 1.2 m at 15 deg
    antenna: {pattern: isotropic}
    streams: [{name: rx0, direction: rx, fc_hz: 3.5e9, rate_sps: "1e6", noise_figure_db: 7}]
  - id: rx2
    position_m: [0.9829815, 0.6882918, 0.0]     # 1.2 m at 35 deg
    antenna: {pattern: isotropic}
    streams: [{name: rx0, direction: rx, fc_hz: 3.5e9, rate_sps: "1e6", noise_figure_db: 7}]
ris:
  - id: ris0
    pose: {center_m: [0, 0, 0], normal: [1, 0, 0], u_axis: [0, 1, 0]}
    grid: {n_u: 24, n_v: 5, pitch_u_m: 0.0428, pitch_v_m: 0.0428, cell_model: auto}
    tiles: {n_tiles_u: 3, n_tiles_v: 1, gap_m: 0.002}
    states: {library: lab_binary_3p5ghz, assume_mag: {sigma1: 1.0, sigma2: 1.0}}
    control: {granularity: element}
    initial: {fill: 0}
    transition: {model: step}
    capabilities: {reconfiguration_time_s: null}
controllers:
  - {name: tdma, kind: builtin, impl: TdmaAlternate, period_s: "2.304e-3", min_latency_s: "2e-4",
     latency: {fixed_s: "2e-4", jitter: "uniform:0,5e-4"}, params: {targets: [rx1.rx0, rx2.rx0]}}
run: {mode: lockstep, coupling: barrier, duration_s: "5", late_policy: apply_on_arrival}
record: {iq: true, events: true, metrics: true}
traffic: [{stream: ap.tx0, generator: qpsk_packets, params: {sps: 4, preamble_syms: 64, payload_syms: 512, guard_s: "5e-4"}}]
```

### 9.4 State-library file (`data/states/*.yaml`)

```yaml
schema_version: "1.0"
kind: ris_state_library
name: lab_binary_3p5ghz
provenance: {source_kind: assumed, description: "ANTS manuscript Sec. III phases; amplitudes unknown",
             reference: "unpublished ANTS 2025 manuscript", reference_plane_offset_m: 0.0, polarization: co,
             includes_cell_pattern: false, sigma_mag_db: 0.0, sigma_phase_deg: 15.0, raw_sha256: null,
             license: "lab permission pending", created: {date: "2026-10-10", tool: "manual"}}
frequency_hz: [3.5e9]
validity_bw_hz: 2.0e7          # [ENG] until measured
angle_grid: none               # angle-independent (separable)
reciprocal: true
reconfiguration_time_s: null   # GS RIS 008 parameter
states:
  - {id: 0, label: sigma1, gamma: [{f_hz: 3.5e9, mag: null, phase_deg: 49.64}]}
  - {id: 1, label: sigma2, gamma: [{f_hz: 3.5e9, mag: null, phase_deg: -153.77}]}
```

`mag: null` forces `assume_mag` in the scenario. The value is copied into the manifest `assumptions` and into the figure metadata.

### 9.5 Guards (`scenario/guards.py` plus physics diagnostics; each test-covered)

- **Errors:** unique ids; references resolve; rates exact; RATE_MISMATCH links; any cell < λ from a node; B > 0.8 f_s with the default target; table validators (§5.5); `t_u_s` above auto; an incidence-only table without `angle_mapping`.
- **Warnings:** a cell < 3λ from a node; an undeclared `occupied_bw_hz`; forced `point` with pitch > λ/2; absolute paths > 1 km (f64-reference note); recorder disk estimate > 10 GB.
- **`sremu validate` prints:** λ; Fraunhofer distances per panel; f_D,max per link; auto T_u and tiles with error bounds; narrowband or wideband per link and the FD choice; noise floor per RX; per-RX `τ_RIS→r` in samples.

---

## 10. Verification and validation matrix

### 10.1 Tiers and rules

| Tier | Content | Budget | Runs | Gate |
|---|---|---|---|---|
| T-A analytic | PH, RT, IO, I (f64 oracles, deterministic) | ≤ 8 min | CI | GATE-PHY / GATE-RT |
| T-B statistical | ST, seeded MC with stated σ | ≤ 30 min | Nightly | GATE-RT (ST-01), GATE-PHY (ST-02, ST-03) |
| T-C code-to-code | MATLAB goldens G1–G8 (CSV + meta in `tests/golden/matlab_R2026a/`) | ≤ 1 min, Python side | CI | Both |
| T-D measured / reproduction | V-TANG, V-LAB, V-CFO, V-WCL, V-FM | ≤ 30 min | Release / paper build | After both gates |

**Rules.**
- Oracles are independent: `tests/oracles/` is ported from `pm_lib.py`/`pm_checks.py` as closed forms and must not import physics, channel or impair code. WP4 writes them, and WP4 is not the physics author.
- Every input data file is copied into `data/` or `validation/` with sha256 and provenance. Private data (Fig. 4(b) digitized CSV, until lab permission) is read from `$SREMU_PRIVATE_DATA`, and its tests skip with the marker `needs_private_data` when it is absent.
- MATLAB is not a CI dependency.
- **MATLAB agreement is code-to-code only**, never validation.

### 10.2 Physics analytic oracles (PH)

The input design's numbers are kept; changes are marked **Δ**.

| ID | Check | Oracle / value | Accept |
|---|---|---|---|
| PH-01 | Friis | 3.5 GHz/10 m: −63.329 dB; 100 MHz/52 km: −106.768 dB | rel ≤ 1e-12 |
| PH-02 | One-model equivalence (**Δ** `cell_model="point"`) | K1 = Tang Eq. 12 = ETSI (G = 4π dx dy/λ²) = Ellingson with area gain | rel ≤ 1e-12 |
| PH-03 | N² law, far-field focus (Tang Eq. 8) | N = 100/400/1600 at λ/2, 2 km, 0°→20°: −169.027 / −156.985 / −144.944 dB | ±0.01 dB; slope 12.04 ± 0.01 dB per 4× |
| PH-04 | Fixed-aperture refinement (**Δ** auto policy) | 1 m², 3.5 GHz, Γ = 1. Far specular 200 m: −115.280 dB for N ∈ {16…16384}. Near anomalous 5 m: −88.10 dB (`po_patch`, Δ ≤ λ/5). Auto picks `po_patch` exactly when pitch > λ/2. **Teeth:** `legacy_v2` +12.04 dB per 4× | ±0.01 / ±0.10 dB; exact; ±0.01 dB |
| PH-05 | Image theory | Oblique 40° plate, 12 m, 3 + 3 m: Δ ≤ 0.3 dB, ≤ 3°. Disk on-axis closed form vs discrete ≤ 0.01 dB; vs image at 200 m ≤ 0.05 dB, 2°. **Teeth:** `no_j` phase −90 ± 3° | as stated |
| PH-06 | Plate RCS | 1 m², 400 m monostatic: −126.067 dB | ±0.01 dB |
| PH-07 | Focusing saturation, passivity | d = 1 m, L = 1 m: −23.864 dB; L → ∞: −6.021 dB; |h| ≤ 0.5 for 10³ random configurations | ±0.01 dB; bound exact |
| PH-08 | Reciprocity, full scene (**Δ** order-1 walls) | LOS + ground + 2 walls (order 1) + knife edge + RIS with a symmetric angle table, including ground hop variants | rel ≤ 1e-12; asymmetric analytic table raises |
| PH-09 | Visibility | Tx behind the panel | RIS term exactly 0 |
| PH-10 | Ground | Two-ray, 52 km, 200/1.5 m, 100 MHz, Γ = −1: 139.10 dB. Γ_H,V(ψ = 1e-4) → −1 (ε_r 15, σ 1e-3). Lossless ε_r = 4: |Γ_V| = 0 at ψ_B = 26.565°. PEC: Γ_H = −1, Γ_V = +1 | ±0.01 dB; |Γ+1| ≤ 1e-3; ≤ 1e-12; exact |
| PH-11 | Knife edge | |F(0)| = 0.5; J(ν) vs exact max 0.123 dB on [−0.78, 10]; continuity at ν = ±1e-9; ν = 10 asymptote 32.95 dB | 1e-12; ≤ 0.15 dB; ≤ 1e-6 dB; ≤ 0.01 dB |
| PH-12 | **Δ** P.526 (experimental, not in any gate) | 52 km, 200/1.5 m: excess 37.14 dB; 200/10 m: 20.28 dB (regression against scratch) | ±0.01 dB; label EXPERIMENTAL |
| PH-13 | Quantization | η(b) = −3.922 / −0.912 / −0.224 dB (quadrature); lab pair −4.105 dB; MC N = 4096, 100 trials | ±0.001 dB; MC ±0.15 dB |
| PH-14 | Abeywickrama | (β_min, α) = (0.2, 1.6): −5.508 dB; (0.8, 1.6): −1.085 dB; independent of φ | ±0.01 dB; 1e-6 dB |
| PH-15 | Efficiency and failures | η = 0.5 → −3.0103 dB on the RIS term; 10% dead, focused, N = 1024, 200 maps: −0.915 dB | 1e-9; ±0.10 dB |
| PH-16 | Phase wrap | Γ·e^{j2πm} | |Δh| ≤ 1e-15|h| |
| PH-17 | Geometric Doppler (T9 geometry from `pm_checks.py`) | LOS 30 m/s, 3.5 GHz: f_D = 350.242 Hz, phase vs −2πf_c d(t)/c over 1 s; RIS path 325.192 Hz. **Teeth:** `block_constant` −148.3 Hz | ≤ 1e-6 rad, ±0.001 Hz; ±0.01 Hz; ±1 Hz |
| PH-18 | Tiles and interpolation (**Δ** unified ε_rel) | Auto (T_u, tile) renderer vs brute force per element per sample: lab walk, vehicle, WCL corridor | |Δh|/|h| ≤ 1e-3; `verify_interp` reports the same value to 10% |
| PH-19 | Invariance | T_u ∈ {auto, auto/2, auto/4}: outputs agree within 2e-3 relative; positions at common times bitwise; client chunks 0.25/1/3.7 ms bitwise | as stated |
| PH-20 | **Δ** FD and wideband | Each FD row ≤ −60 dB in band at worst μ (expected −71.4 / −75.5 / −76.5 dB); polyphase-table interpolation still ≤ −60 dB; 37.4-sample delay recovered < 0.01 samples; narrowband switch at πBΔτ = 0.05 (unit test); narrowband band-edge error ≤ −32 dB. **Teeth:** 4-tap Lagrange at ±0.4 f_s ≈ −5.2 dB fails | as stated |
| PH-21 | Noise | Variance over 10⁶ samples −113.975 dBm (1 MS/s, F = 0 dB); cross-RX |ρ| over 10⁵ samples | ±0.05 dB; ≤ 0.016 (5σ) |
| PH-22 | Linearity, time shift, seeds | `y(ax₁ + bx₂) = a·y(x₁) + b·y(x₂)`; time-shift equivariance; identical output across seeds with noise and diffuse off | ≤ 1e-12 |
| PH-23 | Energy diagnostic | Scattered ≤ incident per panel for every tested configuration | always |
| PH-24 | Exact binary optimum | Sweep vs brute force (N = 20, 2²⁰ configurations, 100 positions) | rel ≤ 1e-12 |
| PH-25 | **New:** m/d decomposition | Decomposed vs direct evaluation; lab values |d| −0.18 dB, |m| −13.86 dB ∠127.9° | ≤ 1e-12; ±0.01 dB |
| PH-26 | **New:** point vs patch (model form) | 16×16 λ/2 panel, 3.5 GHz, 2 m. Specular: difference ≤ 0.05 dB. 45° retro: difference reported (≈ 8.9 dB) as a regression value | ≤ 0.05 dB; ±0.1 dB of the recorded value |
| PH-27 | **New:** occluders and partitions | Segment–polygon hit/miss cases; partition of L dB → exactly −L dB on LOS; opaque → 0 | exact |
| PH-28 | **New:** incidence-only extension | Extended table exactly symmetric; specular entry equals g(|θ_i|) | exact |
| PH-29 | **New:** general alphabet optimum | S = 4, N = 10 (4¹⁰ configurations), 50 random instances, including unequal amplitudes and a direct path | rel ≤ 1e-12 |
| PH-30 | **New:** quantizer | `argmax Re{Γe^{−jφ}}` equals per-element brute force (Tang alphabet, unequal amplitudes) | exact |
| PH-31 | **New:** table validators | Passivity, σ-based reciprocity tolerance (accepts noisy measured, rejects asymmetric analytic), grid density, OutOfDomain, BandwidthExceeded, missing provenance fields; complex-linear midpoint matches the closed form | exact / raises |
| PH-32 | **New:** ground on RIS hops | Γ_g = 0 reduces exactly to the free-space RIS term; four hop variants reciprocal | ≤ 1e-12 |

### 10.3 Runtime invariants and oracles (I / RT)

| ID | Check | Accept |
|---|---|---|
| RT-01 (I1) | Exact counts at 5880, 23.04e6 and 30.72e6 S/s together, for T ∈ {1 s, 7.3 ms, 1/3 s, 10⁶ awkward epochs} | exact |
| RT-02 (R3/R5) | `to_ticks(from_ticks(k)) == k` for k ≤ 2⁴⁰ at all three rates; float 0.5e-3 → 11520 (not 11521); random property tests (`_prop.py` / hypothesis) | exact |
| RT-03 (I2a) | Client chunks {1, 7, 1000, page, random}; epochs 0.25/1/3.7 ms; injected barrier splits; 1/4/16 threads in deterministic mode | bitwise |
| RT-04 (I2b) | `page_target_s` ∈ {0.25, 0.5, 1} ms | ≤ 1e-9 relative (complex128) |
| RT-05 (I3) | Config event at k on every RX; RX at 2 km and 23.04 MS/s gets `to_ticks(t_sw + τ)` (≈ 154 samples after the near RX) | exact |
| RT-06 (I6) | Late command: `apply_on_arrival` gives one physical time across RXs with exact lateness logged; `reject` → LATE_COMMAND, nothing applied | exact |
| RT-07 | Same-time tie: later seq wins, both logged, earlier `superseded`; two future commands both apply. **Teeth:** `LegacyNextTickAdapter` loses the first | exact |
| RT-08 (I8) | Barrier determinism: in-process and remote controllers sleeping 0 vs 50 ms give bitwise-identical RX; property test lateness == 0 for random rational rates, periods and L_min | bitwise / exact |
| RT-09 | CFO continuity: 500 Hz at 5880 S/s; no boundary step beyond the intra-block step ±1e-9 rad. **Teeth:** v2-style per-tick phase gives 0.94 rad | as stated |
| RT-10 | Skip invariance: RX inactive 0.3 s, then active, equals the always-active run | bitwise |
| RT-11 | Known delay through the engine (37.4 samples) | < 0.01 samples |
| RT-12 | Transitions: linear 20 µs ramp gives the analytic |h| to 1e-9; re-command mid-transition restarts from the instantaneous Γ; per-class deltas under mobility vs brute force ≤ 1e-3 | as stated |
| RT-13 | TX commitment, EOB, `idle_until`; stall → deadlock report; `zero_fill` counted | as stated |
| RT-14 (I4) | Adding a node leaves other streams' noise and diffuse variates unchanged; Philox numba == numpy | bitwise |
| RT-15 (I7) | Static import check plus a run with a monkeypatched clock gives identical output | pass |
| RT-16 | Replay: recorded TX + events reproduce RX | bitwise |
| RT-17 | Lookahead violation in lockstep → LOOKAHEAD_VIOLATION | exact |
| RT-18 | Paced counters: injected sleep load → miss count equals injected count; past TX burst → TIME_ERROR; unread RX → OVERFLOW with correct `time_spec` jump | exact |
| RT-19 | Backends: numba_cpu vs numpy_ref on the lab-walk, vehicle and FM scenes | I5 tolerances |

### 10.4 Interface tests (IO)

| ID | Check | Accept |
|---|---|---|
| IO-01 | 10⁴ malformed frames (random magic, sizes, dtypes, truncation) and malformed JSON | No crash; BAD_PACKET / SCHEMA_ERROR counted exactly |
| IO-02 | Two independent OS processes (TX generator, RX sink) over native ZMQ capture the expected waveform (v2 S1/S5 regression) | Bitwise equal to in-process lockstep |
| IO-03 | Socket command lands at the exact sample | lateness 0 in lockstep |
| IO-04 | 1 MB and 65,536-element caps; no paths accepted; non-localhost bind refused without `--allow-remote` | as stated |
| IO-05 | SigMF structural checks (required keys, annotation ordering, collection hashes); `sigmf` validator if installed | pass |
| IO-06 | API codes: TIMEOUT, OVERFLOW, LATE_COMMAND, TIME_ERROR, SEQ_ERROR | exact |
| IO-07 | Ported v1 BPSK scripts run through `Send_to_sim`/`Receive_from_sim` | 2389 samples delivered exactly |
| IO-08 | Raw REQ/REP against stub peers for both `request_format`s, as pinned from source | Sample-exact in lockstep |

### 10.5 Statistical oracles (ST)

| ID | Check | Accept |
|---|---|---|
| ST-01 | 1-D RWP, l = 10 m, v = 1 m/s, no pause, 10⁴ movement periods sampled at 10 Hz | KS D vs F(x) = 3x²/l² − 2x³/l³ ≤ 0.02; start points uniform D ≤ 0.0163; mean |s−d| = l/3 ± 2.5%; no jump > v·Δt + 1e-12 |
| ST-02 | SoS diffuse, f_D = 50 Hz, f_s = 10 kHz | max |ACF − J₀| ≤ 0.05 for lags ≤ 40 ms; LCR at −10 dB within 10% of 35.9 Hz; envelope KS < 0.01 |
| ST-03 | Quantization and failure MC | As PH-13 and PH-15 |
| ST-04 | BER vs Eb/N0 through the emulator (E-F reduced: 6 points, ≥ 100 errors or 2×10⁶ bits) | Theory inside the Clopper–Pearson 99% interval at every point (BPSK and QPSK) |
| ST-05 | CFO estimator vs CRB (V-CFO noisy part, reduced trials) | RMSE ≤ 1.5 × CRB at 10 and 20 dB |

### 10.6 MATLAB goldens (code-to-code; regenerated against K1)

Scripts live in `tools/matlab/` and run via `run_goldens.ps1`, which calls `C:\Program Files\MATLAB\R2026a\bin\matlab.exe -batch` explicitly: R2026b is first on PATH. Outputs carry `matlab_version`, `script_sha256` and `runtime_s`. The existing `D:\projects\ris\data\matlab` CSVs encode the v2 formula and are **not** used for G1–G3 or G7.

| ID | Script / reference | Accept |
|---|---|---|
| G1 | `golden_lab_polar.m`: per-element near-field K1 sum from toolbox primitives (`rangeangle` in panel axes, `phased.CosineAntennaElement`; confirm field vs power convention in R2026a help, `physconst`). 30 polar points × layouts A/B/C × 2 Tx hypotheses, plus the Tang Fig. 10 geometries | max rel |Δh| ≤ 1e-9 |
| G2 | `golden_beam_pattern.m`: `phased.RISReflector` and a hand array factor; 16×16 at λ/2, −30° → +20°, R = 500 m. Mind `pattern()` weight conjugation | Main lobe ±0.5°; levels above −30 dB within 0.05 dB after removing the documented constant 20·log(4π dx dy/λ²) |
| G3 | Same, 1-bit profile | Mirror lobe at 41.15° ± 0.25°, −0.97 ± 0.10 dB |
| G4 | `comm.PhaseFrequencyOffset` on `cfo_reference_iq.csv`; emulator fed arbitrary chunk sizes | ≤ 1e-9 × RMS |
| G5 | `comm.ThermalNoise`, 290 K, 1 MS/s | ±0.01 dB (10⁷ samples) |
| G6 | `comm.RayleighChannel` SoS ACF/LCR (existing CSVs reusable) | As ST-02 |
| G7 | `golden_fm_etsi.m`: ETSI far-field closed form at the focus, 100 MHz | Within 1 dB of ETSI; never above the PO plate bound (0.01 dB) |
| G8 | `phased.TwoRayChannel` at 52 km (verify property names before scripting) | ±0.05 dB |

The paper states: "G2/G3 agreement shows code agreement only; MATLAB's unit-gain cells share v2's normalization, so they do not validate absolute RIS power."

### 10.7 Teeth twins (each legacy guard must fail on the legacy item)

PH-04 (`legacy_v2`, +12.04 dB/4×), PH-05 (`no_j`, −90°), PH-17 (`block_constant`, −148.3 Hz), PH-20 (4-tap Lagrange), RT-07 (`LegacyNextTickAdapter`), RT-09 (per-tick CFO phase), V-CFO (`LegacyTruncationAdapter`, slope 1.0691), and V-WCL W1 (`legacy_wcl_eq2` ratio).

### 10.8 V-TANG (MEAS-3P, digitized)

- **Inputs.** Tang 2022 Tables III and IV; `D:\projects\ris\data\digitized\tang2022_fig10_vector.json` (vector-extracted), copied with sha256. 4 panels × 9 points. Run with the **shipped default** `cell_model` (auto).
- **V-TANG-1.** Engine vs the published model curves: max |Δ| ≤ 0.05 dB (scratch achieved 0.014 dB).
- **V-TANG-2.** Measured points: report bias, SD and RMSE (scratch: +0.98 dB, 0.38 dB, 1.05 dB, n = 36). Regression gate |RMSE − 1.05| ≤ 0.05 dB.
- **Paper wording:** "The kernel reproduces Tang et al.'s published model curves within {0.02} dB. Their digitized chamber measurements lie {+0.98} dB (SD {0.38} dB, n = 36) above that published model." This is not presented as sremu accuracy.

### 10.9 V-LAB: ANTS Fig. 4(b) protocol (digitized lab data)

**Data.**
- `fig4b_measured_rx_power.csv`: 30 cells, θ ∈ {0, 5, 15, 25, 35, 45}° (θ centres inferred), r ∈ {0.6, 0.9, 1.2, 1.5, 1.8} m.
- Per-cell uncertainty 0.045–0.22 dB (typical 0.07 dB). −10.06 and −33.20 dBm are pinned to the colorbar limits under the autoscale assumption.
- The physical reference of r and of θ = 0 is unknown.
- Already-computed descriptive statistics (disclosed): ring means −11.42 / −15.21 / −19.07 / −22.80 / −23.09 dBm; ring SDs 0.97 / 1.82 / 1.13 / 5.24 / 3.27 dB; log-distance n = 2.63 (RMSE 2.78 dB); ring-mean-removed SD 2.74 dB.

**Geometry prior** (discrete × continuous; `protocol.json` holds every number):

| Unknown | Prior |
|---|---|
| Polar origin | O_R, the RIS centre (P = 0.7), or O_T, the Tx (0.3). Under O_T, Tx and RIS centre swap roles in the next row |
| Direction of θ = 0 | Panel normal (0.5) or panel plane (0.5) |
| Tx position (under O_R) | d_T ~ U[1, 3] m, θ_T ~ U[−45°, 0°], same height |
| Layout of {σ1² σ2² σ1²} | A 3×1 tiles [σ1 σ2 σ1]; B 6×1 [σ1σ1σ2σ2σ1σ1]; C 3×2. Uniform |
| Cell pitch | 42.8 mm square, or 38 × 25.98 mm (INDICON cell); tile gap U[0, 5] mm |
| **cell_model** (**new**, model-form) | `point` (0.5) or `po_patch` (0.5) |
| |Γ(σ1)|, |Γ(σ2)| | U[0.6, 1.0] each; phases ± U[−15°, 15°] |
| Antennas | Isotropic, or CosPower patch with G ∈ {6, 9} dBi aimed at the RIS centre |
| Point position error | ±2 cm uniform; height plane ±5 cm |
| Floor (M3 only) | Concrete, ε_r from P.2040 [UNVERIFIED until transcribed]; antenna height U[0.8, 1.5] m |

**Models.** Each fits one offset C (dB) on training cells only. **Every model gets the same prior draws**, so Tx-position freedom is shared.

| Model | Definition |
|---|---|
| M0 | LOS only (Friis + patterns) |
| M0n | Log-distance (C, n), distance from the drawn Tx |
| M1 | Uniform all-σ1 mirror |
| M2 | Lab configuration |
| M3 | M2 + floor |

**Split.** Train on rings {0.6, 1.2, 1.8} m (18 cells); test on {0.9, 1.5} m (12 cells). Also leave-one-ring-out (5 folds) and leave-one-wedge-out (6 folds).

**Primary analysis (geometry is not fitted).**
- K = 2000 prior draws. Compute the prediction tensor **once** and cache it (`pred[K, 30, models]`; ≈ 1.2e9 evaluations, ~10 s on numba).
- Per draw: fit C on train, score on test.
- Report median and 5–95% of test RMSE, bias, MAE and max |e|, plus Spearman ρ between ring-mean-removed residuals and predictions.

**Secondary analysis, labelled "optimistic".** Hypothesis weights `w_h ∝ exp(−SSE_train,h/(2σ_tot²))` with σ_tot = 2.0 dB, and the weighted test prediction. Selecting geometry by calibration RMSE alone is not used.

**Uncertainty budget.** σ_dig per cell; σ_pos (±2 cm) through the local gradient; σ_rep = 1.0 dB assumed (sensitivity runs at 0.5 and 2 dB); σ_geom from the MC spread.

**Decision rules** (fixed before unblinding):

| Claim | Condition |
|---|---|
| C1 "absolute level consistent" | Median test |bias| ≤ 1.5 dB **and** median RMSE(M2) ≤ RMSE(M0n) + 0.5 dB |
| C2 "RIS model explains spatial structure" | All of: median ΔRMSE = RMSE(M0) − RMSE(M2) ≥ 1.0 dB; the paired-bootstrap 95% CI of ΔRMSE (10⁴ resamples over test cells) excludes 0 in ≥ 80% of draws; Spearman ρ ≥ 0.3 with its CI excluding 0 |
| C3 "configuration-specific" | C2 holds with M1 in place of M0 |

If C2 fails, the paper states the null result: the digitized map does not discriminate the RIS model from LOS-only at the available geometric knowledge.

**Sensitivity runs.** Without the (35°, 1.5 m) cell; without both colorbar-pinned cells. The primary analysis keeps all 30 cells.

**Power analysis (before unblinding).**
- 500 decoy datasets from M2 at a random prior draw, plus σ_rep and σ_dig noise. The full protocol runs on each, using the cached tensor and 2000 bootstrap resamples per decoy.
- If the C2 detection rate is < 0.5, the result is labelled **"underpowered"** whatever the outcome.

**Naming rule.**
- "Validation" only if the lab supplies the geometry. Then a single geometry is used, with σ_geom from the supplied tolerances.
- Otherwise: "consistency check under geometric uncertainty".

**Pre-registration.**
1. Commit `validation/ants_fig4b/PROTOCOL.md` and `protocol.json`.
2. Put their sha256 into `protocol_lock.py::PREREGISTERED_SHA256` in a separate commit.
3. `load_measured()` refuses to load the data unless the hash matches; development uses `synthetic_decoy(seed)` only.
4. The user deposits PROTOCOL.md and the hash on Zenodo or OSF before unblinding (human gate H-6).

Paper wording: "The analysis plan was fixed and time-stamped (commit {X}, {date}; deposited at {DOI}) after descriptive inspection of the digitized map but before any model–data comparison."

**Figure F4.** Measured (DIG), predicted (prior median) and residual maps on identical polar axes with one colorblind-safe colormap and measurement points marked (RQ-33). Inset: ΔRMSE distribution and decoy detection rate.

### 10.10 V-CFO (before and after; RQ-35)

- **Before.** Digitized slope 1.0681 (RMSE 57.1 Hz); v1 rerun 1.0691; prediction 11.76/11 = 1.06909 [FACT].
- **Setup.** Port of the v1 BPSK pair: 200-symbol header of 1+1j, 38 BPSK symbols at ±0.1(1+j), 10 samples/symbol, 2389 samples, 5880 S/s. RX LO offset swept from −1400 to +1400 Hz in 100 Hz steps.
- **V-CFO-1 (in process, noise off).** 2389 samples delivered exactly. Unwrapped-phase LS and lag-1 estimators on the header give |f̂ − f| ≤ 1e-6 Hz, identical for epochs of 0.25, 2 and 3.7 ms.
- **V-CFO-2.** `LegacyTruncationAdapter` reproduces slope 1.0691 ± 0.0005.
- **V-CFO-3 (E1, two OS processes over native ZMQ, lockstep).** Slope 1 ± 0.002 and |intercept| < 2 Hz, noise off. Output bitwise equal to V-CFO-1.
- **V-CFO-4 (noise from the engine's physical noise via TX power and NF).** Per-sample SNR {0, 10, 20} dB, 200 independently keyed trials per SNR at each of 7 CFO points. A periodogram + parabolic estimator must reach RMSE ≤ 1.5 × CRB at 10 and 20 dB; the 0 dB result is reported. CRB standard deviations for N = 2000 at 5880 S/s: 0.0256 / 0.0081 / 0.0026 Hz [DERIVED].
- **Figure F5a/b.** DIG points, LEGACY adapter, new engine; RMSE vs SNR against the CRB.

### 10.11 V-WCL (Mishra & Kherani, IEEE WCL 14(4), 2025, which used v1)

**Scenario as published.**
- AP at the corridor origin; MU moves on a line of length l = 10 m.
- RIS line parallel to it at distance h: N = ⌊l/k⌋ = 133 cells, w = k = λ/4, f_c = 1 GHz.
- v = 1 m/s, P_T = 0 dBm, G = 1; 1-D RWP with no pause; `on_off` table.
- **Documented deviation:** the AP is offset d0 = 0.5 m perpendicular to the user line, because with the AP on the user line the average LOS power diverges logarithmically as x → 0. Sensitivity runs use d0 ∈ {0.3, 1.0} m and an on-line AP with cutoffs {0.05, 0.1, 0.2} m.

**Findings handled in the reproduction.**
- Eq. 9 is the power of the mean field; the mean power adds `p(1−p)Σ|S_n|²`. Both forms are reported.
- The letter's Doppler counts the shift twice; sremu does not.
- Kernel ratio `(πλ)²(cos θ_i cos θ_s)^{2q−1}`: −0.52 dB at 1 GHz per element at broadside, −0.4 … +2.8 dB at h = 4 m with the angle factor, ≤ 0.2 dB after averaging. These are different metrics, all true. The paper states nothing about Eq. (2) until the letter's authors confirm.

**Tests.**

| ID | Check | Accept |
|---|---|---|
| W1 | RWP stationary pdf, uniform start points, mean period l/3; `legacy_wcl_eq2` ratio teeth | As ST-01; ratio within 1e-9 of the closed form |
| W2 | Theorem 1: argmin_n D_n at y = x/2; φ_n monotone in x | Within one element pitch; exact |
| W3 | URAP p = 0.2 vs h ∈ {1, 2, 4, 6} m: IQ time-averaged tone power vs analytic ∫E[P(x)]f_X dx (both forms); batch means of 10⁴ movement periods | |Δ| ≤ max(0.05 dB, CI half-width) |
| W4 | Fig. 6 geometry (l = 3, h = 2, N = 20): exact optimum, Eq. 8 heuristic, URAP, GA (300, 0.01, 150) | Optimum exact vs brute force; report P_SO/P_o vs the letter's "15–60%" |
| W5 | τ_SUI ∈ {0.01, 0.1, 0.3, 1, 3, 10} s: IQ (noise off) vs position-only semi-analytic oracle on paired trajectories | ≤ 0.02 dB; non-increasing in τ_SUI within CI |
| W6 | Doppler on/off (sample-level geometric vs position-snapshot average) | ≤ 0.01 dB (f_D ≤ 3.34 Hz) |

Pilot [PILOT, `wcl_check.py`]: under K1, the optimal binary gain over LOS is only 0.29 / 0.18 dB at h = 4 / 6 m (AP on the line, 0.2 m cutoff), and URAP ≤ 0.01 dB. **The letter's geometry has almost no RIS headroom.** This is why E-A adds a blocked-corridor regime.

Vector extraction of the letter's figures for overlays is STRETCH. Release 1 reproduces the letter's analytic expressions instead.

### 10.12 V-FM (model-based dimensioning; replaces ANTS Fig. 6)

- **Primary direct path:** flat-earth two-ray (PH-10), 139.10 dB at h_r = 1.5 m and 122.63 dB at h_r = 10 m.
- **Obstruction:** coherent knife edge F(ν).
- **RIS path:** K1 with ground image expansion on both hops (V-FM-1 regression: the T15 "RIS with ground" values, 159.6–171.6 dB at h_r = 1.5 m and 143.9–155.3 dB at h_r = 10 m for d₂ = 1–2 km, within ±0.1 dB).
- **EXPERIMENTAL sensitivity row:** P.526 direct path 143.9 / 127.05 dB.
- **Deliverable (E-D):** required aperture vs obstruction excess loss. Labelled "model-based dimensioning, not validated".

### 10.13 Gates

- **GATE-PHY** (WP2 + WP3 + WP4 done): PH-01…PH-11, PH-13…PH-16, PH-23…PH-32 on `numpy_ref`; ST-02, ST-03; G1, G2, G3, G6, G7, G8; V-TANG-1.
- **GATE-RT** (WP5 + WP6 + WP7 done): RT-01…RT-19; PH-17…PH-22 on both backends; ST-01; G4, G5; V-CFO-1, V-CFO-2.
- **Rule:** no paper figure (§13) is generated before both gates are green. `build_paper.py` reads the junit XML of the pinned commit and refuses otherwise.

---

## 11. Experiments and case studies (exact parameters)

All experiments run from `scenarios/studies/*.yaml` and `paper/sweeps/*.yaml`. Every figure meta carries data classes. Pilot numbers size the experiments and are never quoted as results.

### E-A: Signaling delay in closed-loop binary RIS control (WCL re-run and extension)

- **A0, reproduction.** V-WCL geometry, LOS unblocked, h ∈ {1, 2, 4, 6} m, f_c = 1 GHz.
- **A1, research regime.** The same corridor with a **partition** on the LOS path, transmission loss 20 dB (sensitivity 10 and 30 dB at h = 2 m); h ∈ {2, 4} m; f_c ∈ {1, 2.4} GHz with w = k = λ/4 over l = 10 m (N = 133 and 320).
- **Controllers:** `RisAbsent`, `Urap(0.2)`, `WclHeuristic(4,3)`, `ExactOptimum`, `ProbeBandit` (SHOULD; codebook = exact optima at 16 positions, dwell 20 ms). Position-driven controllers are wrapped in `DelayedController(τ_SUI, report_delay = 0)` with command latency 1 ms.
- **Sweep:** τ_SUI ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10} s.
- **Batch path:** 20 non-overlapping batches × 500 movement periods per configuration (≈ 3.3×10⁴ s simulated), 5 ms geometry step, exact optimum looked up on a 1 cm position grid. ≈ 150 configurations; minutes on the laptop.
- **IQ consistency subset (VA-1/W5; not a finding):** 1 GHz, h = 2, LOS −20 dB, τ_SUI ∈ {0.1, 1} s, {ExactOptimum, Urap}, 1000 s each, 10 kS/s tone, external controller process in barrier mode, external RX app estimating power over 10 ms windows. Accept ≤ 0.02 dB vs batch with noise off; within CI with noise on.
- **Pilot** [PILOT, `tsui_check.py`; h = 2, LOS −20 dB, 1 GHz; power over blocked LOS]: optimum 5.94 dB; at τ_SUI = 0.1 / 0.3 / 1 / 3 / 10 s → 4.61 / 2.73 / 1.43 / 0.88 / 0.69 dB; URAP ≈ 0.14 dB. At h = 4: 3.61 dB falling to 0.17 dB.
- **Hypotheses (pre-registered with V-LAB in `paper/preregistration/E-A.md`):**
  - **H-A1.** With retention `ρ(τ) = (P(τ) − P_stale)/(P(0) − P_stale)` in linear power, the half-retention delay τ½ of `ExactOptimum` at 1 GHz, h = 2 lies in [0.1, 1] s, i.e. ≥ 3× shorter than the mean movement period l/(3v) = 3.33 s.
  - **H-A2.** τ½ scales with λ/v: τ½(2.4 GHz)/τ½(1 GHz) ∈ [0.30, 0.55] (prediction 1/2.4 = 0.42).
  - **H-A3 (SHOULD).** `ProbeBandit` beats `Urap` at every τ_SUI ≥ 0.3 s. Report the absolute dB.
- **Reporting:** absolute dB effects with 95% CIs. Retention ratios only alongside absolute values. A0 results are presented as "under the area-tied kernel the value is X".

### E-B: Intra-frame RIS reconfiguration (headline use of sample-exact command semantics)

**Why the IQ level is necessary.** Coefficient-level tools (GoSimRIS/ORIX) and symbol-level, block-fading tools (TRACE) cannot place a reconfiguration inside a packet and observe per-symbol EVM. This experiment maps directly to GS RIS 008 "reconfiguration time" and "modulated-signal quality".

**Setup** (`scenarios/lab_like_A.yaml`, §9.3; SIM geometry labelled "lab-like", not the lab's):
- 3 tiles of 5×8 cells at 42.8 mm, 2 mm tile gaps, f_c = 3.5 GHz.
- Tx 2.0 m at −30°; RX1 1.2 m at 15°; RX2 1.2 m at 35°. LOS present; a sensitivity variant blocks LOS with an opaque occluder.
- **Alphabets:** lab σ1/σ2 with |Γ| = 1 (ASSUMED), and ideal 1-bit {1, −1}.
- **Waveform:** f_s = 1 MS/s; QPSK at 4 samples/symbol, rectangular pulses, integrate-and-dump; 64 known preamble symbols plus 512 payload symbols (2304 samples, T_p = 2.304 ms); one-tap LS equalizer from the preamble with no tracking; `occupied_bw_hz` = 0.8 MHz (narrowband: π·B·Δτ = 0.0075).
- **Noise:** NF 7 dB, physical; TX power set so that Es/N0 = 25 dB at RX1 under its own optimal configuration (computed from the snapshot and recorded).
- **Configurations:** `ExactOptimum` for RX1 and for RX2.
- **Pilot** [PILOT, `design_checks/eb_pilot.py`]: switching RX1's RIS channel to RX2's configuration changes it by −12.7 to −15.6 dB with 53–66° rotation. The complex-linear transition midpoint is −5 dB. The RIS term is comparable to LOS (−46.5 vs −46.4 dB).

**E-B1 (mechanism, deterministic).**
- Single packet to RX1, switch to the RX2 configuration at δ ∈ {−0.5, 0.05, 0.25, 0.5, 0.75, 1.05} × T_p after packet start (0.05 T_p falls in the preamble).
- Transitions {step, linear 20 µs, linear 200 µs}; alphabets {lab, ideal 1-bit}.
- Output: per-symbol EVM trace (F9a); per-packet EVM and BER table.
- Acceptance (mechanism, VA-5): the first affected sample on RX1 equals `to_ticks(t_sw + τ_RIS→RX1)` exactly, and symbols wholly before the switch are bitwise identical to the no-switch run.

**E-B2 (timing requirement).**
- 2000 packets alternating RX1/RX2 (`TdmaAlternate`); guards G ∈ {0.1, 0.5, 1.0} ms.
- Controller command latency L = 0.2 ms + U[0, J], J ∈ {0, 0.1, 0.3, 1, 3} ms (keyed `LatencyModel`, lockstep, deterministic).
- **Three command semantics:**
  - **timed:** `t_sw` = guard centre, issued with lead time ∈ {0.5, 2, 5} ms; late → `apply_on_arrival`;
  - **asap:** untimed, issued at guard start, applied on arrival;
  - **legacy:** `LegacyNextTickAdapter`, τ = 2 ms ticks, single-slot overwrite.
- **Metrics:** PER (≥ 1 bit error) with a Clopper–Pearson 95% CI; mean EVM.
- **Analytic expectations** [DERIVED; not claims]: timed PER ≈ P(L > lead); asap PER ≈ P(L > G/2); legacy PER ≈ the fraction of tick boundaries inside packets.
- Figure F9b: PER vs J at G = 0.5 ms for the three semantics.

**E-B3 (SHOULD; Linux, paced + async).** The same timed controller as a real external process over ZMQ, measured lateness distribution and resulting PER. A demonstration, labelled with the platform.

### E-C: Non-ideality ablation and binary decomposition (RQ-60; short analytic result)

- **Batch path only**, exact optimum per alphabet.
- **Geometries:**
  1. lab-like (E-B geometry, 30-point polar grid, Tx at the prior median);
  2. synthetic 16×16 λ/2 panel at 3.5 GHz, Tx 5 m at −30°, Rx 5 m at +20° and Rx 500 m (far);
  3. blocked corridor (E-A A1, h = 2), averaged over the RWP stationary pdf.
- **Rows:** ideal continuous phase; ideal 1-bit; ideal 2-bit; lab |Γ| = 1; lab |Γ2| ∈ {0.7, 0.5}; Tang RIS1; Abeywickrama (0.2, 1.6); 10% dead (absorbing); 10% stuck-at random; efficiency 0.5; `IdealDesigned` deployed on lab states (the cost of ignoring the true alphabet).
- **Outputs:** loss vs ideal continuous (dB) per geometry; the large-N η column; |m| mirror power in the specular geometry.
- Table VIII / F11.
- **Framing:** the m/d decomposition and η table are a short analytic result citing prior discrete and non-ideal RIS work. H-B1 ("lab states ≈ antipodal") is effectively settled by |d| = −0.18 dB and is **not** a headline research question. Compute goes to the unequal-amplitude rows and the specular geometry.

### E-D: FM dimensioning (half a page, one figure; SHOULD)

- 52 km link, 100 MHz, h_tx = 200 m, h_r ∈ {1.5, 10} m.
- Direct path: two-ray plus a knife-edge ridge with ν swept over [0, 5].
- RIS 30 × 30 × 0.25 m (56.25 m²), optimally focused, with and without ground on both hops; d₂ ∈ {0.5, 1, 2} km.
- F12: required aperture A_req (log) vs obstruction excess loss, marking 56.25 m², with the P.526 row as a labelled EXPERIMENTAL sensitivity.
- Physics facts carried into the text [FACT]: free-space focused RIS path 135.27 / 141.29 / 147.31 dB at 0.5 / 1 / 2 km. With ground on both hops the RIS is 16–28 dB weaker than the P.526 direct path at d₂ = 1–2 km unless terrain blocks the direct path.
- Label: "model-based dimensioning, not validated".

### E-E: Integration with external IQ applications

- **E1 (MUST):** V-CFO-3. Two processes, native ZMQ, the ANTS Fig. 2 regenerated.
- **E2 (SHOULD):** GNU Radio 3.10 (WSL2 Ubuntu or AWS) with stock ZMQ REQ Source / REP Sink against the raw mode, a QPSK burst stream, and an external controller flipping the RIS at a scheduled time. Shows the switch at the exact sample and EVM per state. Requires IO-08 to pass first.
- **E3 (STRETCH, 4 h time-box, Linux):** srsRAN 4G eNB + srsUE at 5.76 MS/s. Attach success, iperf throughput per RIS state, RTF (≤ 1, because srsRAN 4G sleeps on RX [FACT]). If the time-box is exceeded, the paper says nothing about srsRAN beyond the raw mode tested with GNU Radio.

### E-F: End-to-end link metrics (RQ-37)

- **BER.** Static lab-like LOS + RIS (fixed configuration); BPSK and QPSK at 4 samples/symbol; Eb/N0 ∈ {0, 2, 4, 6, 8, 10} dB set via TX power against the **physical** noise (no fudge factor). Coherent detection with the genie channel (oracle flagged) and with a 64-symbol pilot LS variant. ≥ 200 errors or 2×10⁷ bits per point. Theory `Q(√(2Eb/N0))`; Clopper–Pearson 95% intervals in F5c. ST-04 is the CI version.
- **API latency.** In-process `send`/`recv` call latency p50/p95/p99 for 1000-sample buffers; native-ZMQ two-process round trip; sustained throughput (MS/s). Platform stated.

### E-G: Scalability and wall-clock behaviour

See §12 (B1–B6) and F7/F8.

---

## 12. Benchmark protocol (laptop + AWS)

### 12.1 Metric definitions (fixes the RTF definition)

| Metric | Definition | Reported as |
|---|---|---|
| **RTF** (lockstep) | Virtual time advanced = **min over streams** of (samples produced / f_s), divided by wall time. Also `RTF_min` = minimum over consecutive 100 ms virtual-time windows | Mean and RTF_min |
| Aggregate throughput | Σ_s samples / wall time | MS/s (a separate metric) |
| Utilization U (paced) | Compute time of a page / page duration | p50, p99 |
| Page latency (paced) | `t_delivered − max(R_j, t_inputs_complete)`, with stage breakdown: snapshot, combine, render, noise, front end, transport in, transport out, record | p50/p99/p99.9/max (log-linear histogram, 1 µs resolution) |
| Deadline misses | Page delivered after `R_j + slack` | Rate at slack 0 / 1 / 5 ms |
| Command latency | `t_wall_visible − t_wall_recv`; lateness in samples | Distribution; lateness must be 0 in lockstep |
| Sustainable | A paced run of ≥ 60 s after 5 s warm-up with miss rate < 1e-3 at 1 ms slack and zero `late_samples`/`overflow_pages` | Max L (or Σf_s) per configuration |
| Loss counters | `late_samples`, `underflow_slip_samples`, `overflow_pages`, `bad_packets`, `rec_dropped`, `stall_zero_samples`, `rejected_cmds` | Per run |
| Fidelity | I1–I8 pass/fail; measured `verify_interp` | Table |

"Near-real-time" may appear only as shorthand, defined once in the paper as **Sustainable** on a named machine.

### 12.2 Sweeps

| ID | What | Grid |
|---|---|---|
| B1 | Kernel micro-benchmarks → cost-model coefficients (§7.4) | Snapshot R·Q ∈ {40 … 1.7e7}; GEMV; render with T_FD ∈ {0, 8, 16, 32}; noise; per backend |
| B2 | Lockstep RTF | L ∈ {1, 2, 4, …, 128}; N ∈ {40, 240, 900, 4096, 16384}; RIS count ∈ {0, 1, 2, 4}; f_s ∈ {0.1, 1, 5.76, 23.04} MS/s. Feature sets: F0 channel only → F1 + noise → F2 + CFO/PN → F3 + mobility (auto T_u, tiles) → F4 + recording → F5 + ZMQ transport. Every scenario asserts a **non-zero RIS path** (the v2 benchmark had |h_ris| = 0 [FACT]) |
| B3 | Paced sustainable links and latency CDFs | Same grid; `page_target_s` 0.25 / 0.5 / 1 ms |
| B4 | Control loop | Command rate 10 Hz–10 kHz; lateness and visible latency; lockstep (must be 0) vs paced async |
| B5 | Historical v1 workload | ANTS Fig. 3 workload (5000 S/s, N nodes) through v1-as-committed (with the one import fix from audit_v1) vs sremu, **same host only** |
| B6 | Cross-tool context (RQ-44) | Time per channel evaluation, same scene: sremu numba vs the MATLAB R2026a per-element script vs `RISReflector` vs v2. Scope caveats; literature numbers quoted with their hardware (ACHEM 82–167 µs end to end; Colosseum 1 ms snapshots; ORIX 10 ms–1 s loop; TRACE 5.2 ms control RTT) [FACT] |

### 12.3 Protocol

- 5 s warm-up, then ≥ 60 s and ≥ 10⁴ pages per point, 3 repetitions.
- Report the median of medians and the worst p99.
- State hardware, OS, power plan or governor, versions, threads, pinning (`taskset` on Linux) and sampled background load.
- Never compare across machines without rerunning both.
- Paced numbers in the paper come **only** from Linux. Laptop paced numbers may appear with their load stated, marked "Windows, not representative".

### 12.4 Machines and AWS plan (each launch needs the user's explicit approval)

| Host | Use | Instance-hours |
|---|---|---|
| Laptop (Ryzen 9 8945H 8C/16T, 31 GB, RTX 4070 Laptop, Windows 11 Balanced) | All development, tests, lockstep B1/B2, all case-study compute | — |
| WSL2 Ubuntu (installed) | GNU Radio E2 development | — |
| AWS **c6i.8xlarge**, Ubuntu 24.04, on-demand | B3/B4 paced numbers; B5 v1 rerun on the same host; E-B3 | ≈ 8 h |
| AWS **c6i.metal** (bare metal, 128 vCPU) | One quiet, hypervisor-free paced run plus the "large server" point (RQ-45) | ≈ 2 h |
| AWS c6i.2xlarge (optional) | E3 srsRAN 4G | ≤ 4 h |
| GPU instance | **None.** GPU is cut | — |

- **Cost** about US$20–60 including storage and transfer [UNVERIFIED; check the console].
- **Procedure.** Laptop pilot sizes each job → user approves each launch → everything is tagged `project=sremu` with an owner tag → `bench/aws/run_all.sh` → artifacts to one S3 bucket → terminate → `TEARDOWN.md` checklist; instance ids are recorded in manifests.
- No credentials in either repo.
- The Ice Lake c6i host is **not** a "comparison point" for the ANTS Xeon 6338: any v1 comparison reruns v1 on the same host.

---

## 13. Paper outline (IEEEtran journal, ≈ 13 pages + references)

### 13.1 Title and abstract

**Title (recommended):** *Evaluating RIS Control Under Signaling Delay and Non-Ideal Element States: An Open, Sample-Accurate IQ-Level Emulator*. Alternate: *Timed RIS Control in the Loop: An Open IQ-Level Emulator with Measured Element States and Characterized Timing*. Keywords include "smart radio environment" for continuity with ANTS.

**Abstract template** (≤ 200 words, ≤ 1920 characters; `{…}` filled only by the pipeline):

> Controllers for reconfigurable intelligent surfaces (RIS) are usually evaluated with ideal element states, instantaneous reconfiguration and channel coefficients rather than waveforms, so the effect of control timing and hardware states on a live link is hard to test before deployment. We present sremu, an open-source, software-only emulator that applies RIS-assisted propagation to complex-baseband sample streams exchanged with external applications through an SDR-style interface. RIS configurations are timestamped commands from an external controller; each takes effect at an exact, logged sample on every receiver, and per-state element responses can be imported from measurements. The area-tied propagation kernel reproduces published RIS path-loss model curves within {0.02} dB, and its absolute-power predictions are compared with digitized 3.5 GHz laboratory data under a pre-registered protocol ({V-LAB outcome}). In lockstep mode the emulator is sample-exact and deterministic; in wall-clock-paced mode on {platform} it sustains {X} links at {Y} MS/s with a deadline-miss rate below 10⁻³. Two studies use the sample-exact semantics: a RIS switch inside a packet {E-B result}, and the benefit of position-driven binary control decays with signaling delay on a time scale of {τ½}, set by λ/v rather than the mobility period.

### 13.2 Contributions (each mapped to evidence)

1. **Timed RIS control in an IQ emulator.**
   - The switch is defined at the RIS and applied per receiver at `to_ticks(t_sw + τ_RIS→r)`; late commands keep one physical switch time; every command is logged.
   - Deterministic controller-in-the-loop through virtual-time barriers, plus a measured paced mode.
   - UHD-shaped API, framed ZMQ data plane, GS RIS 008-aligned capability and response-table schema.
   - (Sec. III; F1, F2; Tables III, VI.)
2. **A verified, validation-referenced physical model.**
   - Area-tied PO kernel with an explicit cell model and model card.
   - {n_tests} oracles generated from CI; MATLAB code-to-code.
   - Tang curves (bias and SD stated); pre-registered lab consistency check with power analysis.
   - CFO root cause and fix (slope 1.0691 → 1.000).
   - (Secs. IV–V; F3–F6; Tables II, IV, V.)
3. **Characterized wall-clock behaviour.** Definitions; RTF, misses and latency percentiles; scaling over links × elements × RIS count × rate × features on a laptop and a Linux server; complexity fit; context table. (Sec. VI; F7, F8; Table VII.)
4. **Results only a sample-level emulator can produce.**
   - Intra-frame reconfiguration timing and the value of timestamped commands (E-B).
   - The signaling-delay time scale in a blocked corridor (E-A).
   - A non-ideality ablation (E-C) and an FM dimensioning negative result (E-D).
   - (Sec. VII; F9–F12.)

### 13.3 Sections

| § | Title | Pages | Content |
|---|---|---|---|
| I | Introduction | 1.0 | Problem; why coefficient-level and ideal-state evaluation is insufficient; contributions; scope box |
| II | Related work and gap | 1.0 | Table I (≈ 12 rows, versions and dates, "nv" kept distinct); TRACE paragraph; lineage: v1 was the WCL 2025 "in-house simulator" |
| III | Design | 2.0 | Architecture (F1); time model and command semantics (F2); interfaces (Table III); state library and GS RIS 008 (Table VI) |
| IV | Physical model | 1.5 | K1 equation, cell model, Doppler bound, noise; Table II model card with validity domains |
| V | Verification and validation | 2.0 | V-A oracle table (generated); V-B MATLAB; V-C Tang (F3b); V-D lab (F4, Table V); V-E CFO, BER (F5); V-F Doppler (F6) |
| VI | Wall-clock behaviour and scalability | 1.5 | Definitions; F7, F8; complexity fit; Table VII |
| VII | Case studies | 3.0 | E-B (F9), E-A (F10), E-C (F11), E-D (F12, short), E-E integration paragraph |
| VIII | Discussion | 0.75 | Standards mapping; AI/ML hooks (`RisEnv`, exact optimum as regret reference); limitations table |
| IX | Future work | 0.25 | UHD shim via ACHEM, E2SM adapter, coupling models, GPU-resident pipeline, multi-RIS interaction, rate conversion, second measured dataset |
| X | Conclusion | 0.25 | — |
| App. | Oracle details, reproducibility (one command), data provenance, generative-AI disclosure | — | — |

### 13.4 Figures (vector PDF, 3.5 in / 7.16 in widths, ≥ 8 pt fonts, Okabe–Ito palette, no titles inside, Type-42 fonts; every caption carries data classes)

| # | Content | Source | Answers |
|---|---|---|---|
| F1 | Architecture: TX app, RX app, controller processes; engine; native/raw data planes vs control plane; recorder | TikZ | Flow diagram; integration |
| F2 | (a) Page/segment processing flowchart; (b) RIS command lifecycle: wall receive → t_req at the RIS → per-RX k_applied with τ_RIS→r offsets; one late command; one barrier | TikZ | Flow diagram; RIS-control integration |
| F3 | (a) Fixed-aperture refinement: K1 flat at −115.28 dB vs v2's +12.04 dB/4×; (b) Tang Fig. 10 model curves vs sremu, with measured points (DIG) | PH-04, V-TANG | Physical correctness |
| F4 | Lab consistency check: measured / predicted / residual maps; ΔRMSE distribution; decoy power | V-LAB | Quantitative validation |
| F5 | (a) CFO before/after; (b) RMSE vs CRB; (c) BER vs Eb/N0 | V-CFO, E-F | Numeric metrics; end-to-end |
| F6 | Instantaneous frequency of LOS and RIS paths vs analytic (350.242 / 325.192 Hz); block-constant alias; `verify_interp` error vs T_u | PH-17/18 | Mobility, Doppler |
| F7 | (a) Lockstep RTF vs links (log–log) for N and feature sets, RTF = 1 line; (b) aggregate MS/s; laptop vs c6i | B2 | Scalability, RIS counts |
| F8 | Paced: miss rate vs slack; page latency CCDF with stage breakdown; command lateness | B3/B4 | Honest timing |
| F9 | E-B: (a) per-symbol EVM trace; (b) PER vs latency jitter for timed / asap / legacy | E-B | IQ-only result |
| F10 | E-A: power over blocked LOS vs τ_SUI (log x), controllers, 1 and 2.4 GHz; inset IQ vs batch agreement | E-A | Closed-loop control under mobility |
| F11 | E-C ablation bars across three geometries | E-C | Non-idealities |
| F12 | E-D sizing map | E-D | FM claim recast |

**Tables.** I tool comparison; II model card (generated from `model_card.yaml`); III API mapping; IV V&V summary (generated from junit); V lab metrics; VI standards mapping; VII platforms and versions; VIII ablation numbers.

### 13.5 Paper pipeline (`paper/tools/build_paper.py`; Python, not make)

Steps:
1. Check that `sremu.lock` equals the installed sremu commit and that both trees are clean; refuse otherwise.
2. Read the junit for GATE-PHY and GATE-RT; refuse if not green.
3. For each `figures/registry.yaml` entry, compute a content key = hash(commit, sweep spec, input hashes, script hash) and rerun only stale entries.
4. Write `figures/data/<fig>.csv` + `.meta.json` and vector PDFs.
5. Write `generated/numbers.tex` (`\newcommand` macros for every quoted number) and `generated/tables/*.tex`.
6. Run `check_claims.py`, `check_refs.py` (every entry has a DOI, arXiv ID or official URL, sourced from `refs_verified.bib` or verified anew) and `check_arxiv.py` (abstract ≤ 1920 characters; no `\today`, TODO or line numbers; filename characters; `.bbl` present; images < 34 MP).
7. Run latexmk.

`PROFILE=fast` must finish in < 20 min on the laptop; `PROFILE=full` produces the paper numbers.

### 13.6 Claims and honesty enforcement (minimal version)

- `claims.yaml` maps claim ids to `{text, evidence: [test ids | figure ids | citation keys]}`. `check_claims.py` fails if any evidence test is not "passed" in the pinned junit, if any figure meta is missing, or if a tex number is not a generated macro.
- Banned-word grep (§1.6) must return 0 without a mapped claim.
- Per-sentence `\claim{}` markers are optional (STRETCH).
- **References to verify and add before use** (not all are in `refs_verified.bib`): Tang 2022 mmWave path loss (arXiv:2101.08607), TRACE (arXiv:2608.11798), Wu & Zhang discrete phases (arXiv:1906.03165), Abeywickrama (arXiv:2002.10112), Matthiesen et al. (arXiv:2006.06991), Laakso et al. 1996, Zhao et al. WCL 2024 (the letter's ref. [6]), linear-time optimal discrete beamforming [UNVERIFIED], the SigMF spec, the UHD manual, ETSI GS RIS 008 (present).

---

## 14. Work breakdown (13 packages, four tracks plus lead)

### 14.1 Packages

Hours are agent wall-hours. Owned files follow §2.2. Acceptance tests are those named in §10. Interfaces are those of §3.4.

| WP | Track | Title | Depends on | Owns | Provides | Acceptance | h |
|---|---|---|---|---|---|---|---|
| **WP0** | Lead | Foundations and frozen contracts | — | pyproject, units, timebase, contracts, errors, cli, testing/ (FakePropagation, FakeEngine), tools/tasks.py, CI files, test_layering, `_prop.py` | Time base fully implemented; all §3.4 types; fakes for parallel work | RT-01, RT-02, RT-15 (static), layering test, CLI smoke, offline editable install | 3 |
| **WP1** | T | Scenario schema, RNG, manifest | WP0 | scenario/*, rng.py, record/manifest.py, data/states/*.yaml, scenarios/{quickstart, lab_like_A}.yaml | Pydantic models (§9), keyed Philox (numba + numpy), manifest writer | Schema round-trip and JSON Schema validation of all scenarios; guard tests; manifest dirty-detection test (temp repo); RT-14 Philox parts; numba == numpy Philox bitwise | 3 |
| **WP2** | P | Physics core | WP0 (WP1 for `from_scenario`) | physics/{geometry, antennas, los, quadrature, ris_kernel, response, tables, nonideal, optimize, propagation, legacy}.py, `states` CLI | `PropagationModel` (LOS + RIS), ResponseTable, optimum, legacy kernels | PH-01–07, 09, 13–16, 23–26, 28–31 on numpy_ref; numba kernel ≤ 1e-12 vs numpy; G1–G3 once WP4 lands; V-TANG-1 | 7 |
| **WP3** | P | Propagation extras | WP2 | physics/{ground, walls, diffraction, diffuse}.py, physics/experimental/ | Ground image expansion (incl. RIS hop variants), order-1 walls, occluders/partitions, knife edge, SoS diffuse, P.526 experimental | PH-08, 10, 11, 12 (experimental), 27, 32; ST-02; G6–G8 | 4 |
| **WP4** | P (independent agent) | Oracles and MATLAB goldens | WP0 | tests/oracles/, tests/golden/, tools/matlab/, `verify` CLI | Closed-form oracles; regenerated goldens G1–G8 (MATLAB R2026a, ≈ 5 min) | Oracle self-tests vs `physics_model_checks.json`; import-isolation test; goldens committed with meta; harness ready | 4 |
| **WP5** | T | Render kernels and impairments | WP0, WP1 | channel/*, impair/* | `render()` (numpy_ref, numba_cpu), FD tables, chord/tiles policy, `verify_interp`, transition math, noise, CFO, PN | PH-17–22 (with FakePropagation, then the real model), RT-09, RT-11, RT-12 math, I5, G4, G5 | 6 |
| **WP6** | T | Engine | WP0, WP1, WP5 (WP2 for integration) | engine/*, metrics/*, legacy/{truncation, next_tick}.py | Lockstep and paced scheduler, events/commands, barriers, batch path | RT-03–08, 10, 13, 16 (with WP7), 17, 18, 19; ST-01; VA-4; V-CFO-1/2 (with WP7 BPSK) | 7 |
| **WP7** | I | Python API, compat, recording | WP0 (FakeEngine), WP6 | api/*, compat.py, record/{sigmf, eventlog, replay, report}.py, examples/{quickstart, bpsk_txrx} | `Device.inproc`, streams, metadata, SigMF, replay, report | IO-05–07; RT-16; replay bitwise; report renders | 4 |
| **WP8** | I | Transport and control plane | WP7 | transport/*, control/remote.py, examples/{controller_remote, gnuradio_zmq} | Native ZMQ, control ROUTER/PUB, remote barriers, raw REQ/REP (gated) | IO-01–04, IO-08; remote vs in-process bitwise; V-CFO-3 (E1) | 5 |
| **WP9** | X | Controllers, env, analysis | WP0, WP2 (optimize), WP6 (batch) | control/{controller, builtin}.py, ml/env.py, analysis/stats.py, experiments/* | §8.7 built-ins, `RisEnv`, batch means, bootstrap, Clopper–Pearson, rank correlations, sweep runner | VA-3, W2, W4 (optimum and heuristic parts); env contract test; stats unit tests vs scipy | 4 |
| **WP10** | X | Validation studies | GATE-PHY, GATE-RT, WP9 | validation/*, data/validation/* | V-TANG, V-LAB (protocol, decoys, lock), V-CFO-4, V-WCL W1/W3/W5/W6, V-FM | §10.8–10.12 acceptance; reports with manifests | 6 |
| **WP11** | X | Experiments and benchmarks | GATE-PHY, GATE-RT, WP8, WP9 | scenarios/studies/*, bench/*, paper/sweeps/* (data side) | E-A, E-B, E-C, E-D, E-E2/3, E-F, B1–B6, AWS scripts | VA-1, VA-5; ST-04; sweep outputs with manifests; AWS teardown checklist | 7 |
| **WP12** | Lead/X | Paper repo, pipeline, docs | Skeleton: WP0. Figures: both gates | paper/*, docs/*, CITATION.cff, README final | `build_paper.py`, `check_*`, registry, `model_card.yaml` → Table II, section drafts | `PROFILE=fast` < 20 min from a clean clone; `check_claims`, `check_refs`, `check_arxiv` pass; banned-word grep 0 | 5 |

Total ≈ 65 agent-hours. With five parallel lanes, ≈ 1.5–2.5 days of wall time including integration. **Core** (WP0–WP9) ≈ 47 h; **studies and paper** (WP10–WP12) ≈ 18 h.

### 14.2 Schedule and critical path

```
Day 1  WP0 (3h) -> [WP1, WP2, WP4, WP5, WP12-skeleton in parallel; WP7 against FakeEngine]
Day 1–2  WP3 (after WP2), WP6 (after WP5; integrates WP2), WP8 (after WP7), WP9 (after WP2/WP6)
         -> GATE-PHY (WP2+WP3+WP4)   -> GATE-RT (WP5+WP6+WP7)
Day 2–3  WP10, WP11 (laptop) -> AWS window (user-approved) -> WP12 figures + writing
Critical path: WP0 -> WP5 -> WP6 -> GATE-RT -> WP11 (E-B, E-A) -> WP12
```

### 14.3 Integration rules

- **Branches and merges.** One branch per WP (`wp/NN-slug`). A WP edits only its owned files plus its own test files. The lead merges in dependency order; `tools/tasks.py ci` must pass on main after every merge.
- **Contract change requests.** `contracts.py`, `cli.py` and the pyproject dependency lists change only via a CCR in `docs/contracts_changelog.md`, applied by the lead with a `CONTRACT_VERSION` bump. Dependent WPs rebase.
- **Fakes first.** WP0 ships `FakePropagation` (analytic LOS-only, moving nodes supported) and `FakeEngine` (identity channel, exact time base), so WP5, WP6, WP7 and WP8 can start before WP2 or WP6 land. Their tests run against the fakes and are re-run against the real components at integration.
- **Oracle independence.** WP4's agent must not read WP2's implementation while writing oracles; it works from `pm_lib.py`, `pm_checks.py` and the equations in §5.

### 14.4 Cut order (cut the first item first)

1. GPU geometry backend (STRETCH).
2. E3 srsRAN.
3. Vector overlays of the WCL figures.
4. E2 GNU Radio and raw REQ/REP. Keep native ZMQ and E1.
5. `ProbeBandit` / H-A3.
6. Diffuse SoS fading. Keep the model-card statement; ST-02/G6 move to future work.
7. Wiener phase noise.
8. E-B3 (paced real-controller demo).
9. E-D FM figure. Keep a two-sentence link budget.
10. The 2.4 GHz arm of E-A (H-A2).
11. c6i.metal run. Keep c6i.8xlarge.

**Never cut:** GATE-PHY; GATE-RT; V-LAB protocol with decoys; V-CFO with E1; E-B1/E-B2; E-A batch plus the IQ consistency subset; E-F BER; B2 and the Linux B3 paced measurement; E-C table; the paper pipeline's gate checks.

---

## 15. Contradiction register

| # | Topic | Options in the inputs | Decision | Why |
|---|---|---|---|---|
| 1 | Time→sample rounding | product ceil; runtime nearest-ties-up | **Nearest, ties up** (R3), plus `Fraction(repr)` for API floats | Ceil maps a float 0.5e-3 s to 11521 |
| 2 | Interval counts | product floor–floor; runtime ceil–ceil | **ceil(t1 f) − ceil(t0 f)** on [t0, t1) | Consistent with the index rule (I1) |
| 3 | Where a switch takes effect | product: every stream at the same index; runtime: at the RIS + τ per RX | **Runtime**, per tile when tiled; one physical time for late commands | Physically right; the real differentiator |
| 4 | Barrier frontier vs command mapping | Unspecified | Both use `to_ticks`; property test I8 | Removes the off-by-one false lateness |
| 5 | Interpolation scheme | physics chord + exact phase; runtime analytic-ν rotator; product linear delay | **Chord delay + exact carrier phase + tiles** as the normative definition; the rotator is the numba implementation | Equivalent; no ν field needed; test-enforced |
| 6 | Interpolation tolerances | runtime ε_φ 0.01 rad + 0.25 rad spread; physics 1°, 0.05 rad, PH-18 1e-3 | **One target, ε_rel = 1e-3**, deriving both T_u and the tile size; `verify_interp` and PH-18 check it | The 0.25 rad spread bound gave ≈ 8e-3 |
| 7 | Narrowband switch | physics πBΔτ ≤ 0.05; runtime f_s Δτ ≤ 0.1; product B·Δτ > 0.1 warning | **πBΔτ ≤ 0.05 with B the declared occupied bandwidth** | The others allow −14.8 to −18 dB per-path error |
| 8 | FD filter | Both defaulted to 4-tap cubic | **Kaiser 8/16/32 by band, ≤ −60 dB, polyphase**; integer-only delay only for B ≤ 0.0318 f_s | Lagrange fails at LTE-like occupancy |
| 9 | Sub-cell default | physics λ/5 everywhere; product auto with warning; model card: only cells > λ/2 | **`cell_model` auto: point ≤ λ/2, po_patch above**; point-vs-patch spread in the V-LAB prior | λ/5 everywhere silently changes the validated model |
| 10 | Γ interpolation | physics complex-linear; product magnitude + unwrapped phase | **Complex-linear** + grid-density validator | Bounded, linear in Γ, no 2-D unwrap |
| 11 | Reciprocity tolerance | physics fixed 0.02; product rejects incidence-only | **σ-derived tolerance; logged symmetrize; incidence-only only via the half-difference extension; raw non-reciprocal tables rejected in r1** | Real measured tables need σ-aware tolerance; no modelling-induced non-reciprocity |
| 12 | Lab state amplitudes | physics built-in |Γ| = 1; product `mag: null` | **`mag: null` + mandatory `assume_mag`** recorded; σ_phase 15° for MC; E-C amplitude rows | Makes the assumption visible |
| 13 | Coefficient container | product CascadeCache (L, N); physics B[L,S,N] | **B[R,S,N] with a separable fast path**; R rows include ground hop variants | Angle-dependent measured tables (gap iii) need it |
| 14 | Fig. 4(b) protocol | physics prior-predictive C1–C3 + decoys; product geometry selection by calibration RMSE, "−2u" rule | **Physics primary**; product's naming rule kept; selection by RMSE dropped; weighted version labelled "optimistic" | ≈ 9 unknowns vs 18 training cells; equal freedom for baselines |
| 15 | Pre-registration credibility | In-code hash lock | Keep the lock; wording "fixed and timestamped after descriptive inspection, before model comparison"; Zenodo/OSF deposit (H-6) | Honest; the descriptive stats were already computed |
| 16 | Scenario format | runtime TOML + dataclasses; product YAML + pydantic | **pydantic v2, YAML primary**, JSON/TOML accepted; exact strings for rates and times | Better errors; installed |
| 17 | Event log | runtime JSONL; product Parquet | **events.jsonl canonical** + SigMF annotations; Parquet only for sweeps | Streamable, crash-safe, no dependency |
| 18 | Frame header | runtime 48-byte "SRMU" with seq and wall_ns; product 32-byte "SREU" | **48-byte**, bytes `b"SRMU"` | seq for SEQ_ERROR; wall_ns for latency |
| 19 | Control vocabulary | runtime `ris.set_config`/`ctrl.*`; product `ris.set`/`fence.advance` | **Runtime ops**; barrier traffic over ROUTER/DEALER, not PUB | PUB may drop |
| 20 | Coupling modes | product mode × coupling; runtime lockstep → barriers | Matrix in §4.9; **lockstep + async rejected** | Determinism |
| 21 | RNG keying | product SeedSequence spawn; runtime name-keyed Philox; physics (seed, rx) | **blake2b name-keyed Philox4x32-10, counter = absolute index** | I4: adding a node changes nothing else |
| 22 | Fading | physics additive persistent; runtime pure function of time; product stateful | **Additive SoS as a pure function of time**; never multiplies | Chunk and skip invariance; fixes v2 |
| 23 | Transitions | physics h-domain step 6; runtime per-class g_c | **Per-class g_c with exact Δh_c**, default t_sw from `reconfiguration_time_s`; complex-linear path labelled ENG; optional `gamma_path` | Correct under mobility |
| 24 | GPU | runtime four backends; physics optional; product stretch | **numpy_ref + numba_cpu only**; torch f32-differential geometry STRETCH | No case study needs a GPU; PCIe makes socket IQ a loss |
| 25 | P.526 | physics experimental port; product cuts it but quotes its numbers | **Experimental module outside the core**; two-ray + knife edge primary; P.526 only as a labelled sensitivity row | Unvalidated against ITU data |
| 26 | Coupling | physics κ proxy (off); product tables only | **Hook only, no κ values**; coupling through tables; scope paragraph | No supercell data exists to fit κ |
| 27 | TSDSI | runtime "inspired by"; product mapping only if read | **One sentence: not accessed, no alignment claimed** | Members-only text |
| 28 | Reference optimum | product quantizer + coordinate ascent; physics exact sweep | **Exact sweep for any finite alphabet** (PH-24/29); exhaustive search only as the N = 20 oracle; cited as known | Cheap and exact |
| 29 | IQ vs batch agreement | product within CI; physics 0.02 dB | **≤ 0.02 dB with noise off and paired trajectories; within CI with noise on** | Strong and testable |
| 30 | Geometry update for WCL | product fixed 1 ms; runtime 5.9 ms; physics auto | **Auto from ε_rel** (10 ms cap); the scenario may only lower it; logged | One policy |
| 31 | WCL discrepancy magnitude | physics "0.2 dB"; product "0.5 dB" | Both true at different levels (per element at broadside −0.52 dB; per element at h = 4 m −0.4…+2.8 dB; averaged ≤ 0.2 dB); **nothing in the paper until the authors confirm** | Co-authors' published letter |
| 32 | Case A emphasis | product H-A1 on retention at the letter's geometry | **Blocked-corridor regime (A1)** + λ/v scaling hypothesis; A0 reported as a reproduction | Under K1 the letter geometry has 0.2–0.4 dB of headroom [PILOT] |
| 33 | Case B emphasis | product RQ-B headline | **Short analytic result + E-C ablation**; not a headline | |d| = −0.18 dB nearly settles H-B1 |
| 34 | IQ-level necessity | product Case A IQ subset | **E-B intra-frame switching is the IQ-only headline**; the E-A IQ subset is a consistency check | W6: sample-level Doppler changes mean power by ≤ 0.01 dB |
| 35 | CFO "after" experiment | physics in-process; runtime two threads; product two processes | **Both**: in-process exactness (V-CFO-1) + two-process E1 (V-CFO-3), sharing one legacy adapter | Exactness and integration |
| 36 | RTF definition | runtime Σ over streams; RQ-40 wall/sim | **Min-over-streams virtual time / wall time**, with RTF_min over 100 ms windows; aggregate MS/s separate | Conventional direction; no stream double-counting |
| 37 | AWS | runtime c6i × 2 + g6; product c7i + metal + g6; physics none | **c6i.8xlarge ≈ 8 h + c6i.metal ≈ 2 h (+ optional E3)**; no GPU; same-host v1 rerun only; per-launch approval | Credibility, not speed |
| 38 | Phase plans | Three numberings | **13 WPs** (§14) with GATE-PHY / GATE-RT | One plan |
| 39 | Oracle naming | product T1–T18; physics PH/ST/G/W | **One namespace** (§0); counts generated by the pipeline | No hand-typed tables |
| 40 | Licence | product Apache-2.0; RQ-100 MIT | **Open ADMIN gate**; repo private until resolved | Needs all six authors + IIT Bhilai |
| 41 | Name collisions | Gates "G1/G2" vs goldens G1–G8 | Gates renamed **GATE-PHY / GATE-RT** | Unambiguous |
| 42 | "Real-time" in the abstract | product "paced on a laptop"; runtime Linux only | **Platform placeholder filled by the pipeline**; paced numbers from Linux | Windows sleep jitter ≈ 0.9 ms at p99 |

---

## 16. Open items and human gates

| ID | Gate | Who | Blocks |
|---|---|---|---|
| H-1 | Lab answers: Tx position, height and antenna; tile layout and pitch; whether σ1/σ2 were measured or designed, and |Γ|; Tx power and cables; **permission to publish the Fig. 4(b) CSV**; which v1 commit the WCL letter used | Author / lab | V-LAB "validation" wording, publication of the data file. Without H-1: consistency check only, data private |
| H-2 | Licence (MIT vs Apache-2.0) and arXiv licence | Six authors + IIT Bhilai | Public repo, CITATION.cff, arXiv |
| H-3 | WCL Eq. (2) dimensional check | Mishra, Kherani | Any statement about the letter's kernel |
| H-4 | TRACE differentiation wording | Kumar, Ambwani | Related-work text |
| H-5 | AWS launches | User, per instance | B3/B4/B5 numbers |
| H-6 | Zenodo/OSF deposit of the V-LAB protocol (and E-A hypotheses) before unblinding | User account | Pre-registration wording |
| H-7 | Name availability: "sremu" on PyPI and GitHub [UNVERIFIED] | Author | Public push |
| H-8 | Integrity items RQ-01…08 (phantom references, CAISc template, AI disclosure, git identity) and arXiv endorsement | Author | Posting |

---

## Appendix A. Evidence and recomputed numbers

**Inputs read for this spec** (all under `D:\projects\ris\research\` unless stated): `related_work.md`, `physics_models.md`, `emulation_practice.md`, `requirements_matrix.md`, `audit_v1.md`, `audit_v2.md`, `matlab_ris.md`, `design/{physics_validation, runtime_interfaces, product_paper}.md`, `D:\projects\ris\data\digitized\README.md`, `D:\projects\ris\data\papers\mishra2025_wcl_binary_ris_mobility.txt`, and `D:\projects\ris\data\papers\ext\trace_2608.11798.txt` (skimmed).

**Scripts and data relied on:**
- `research/scratch/pm_lib.py`, `pm_checks.py` (oracle sources; values in `data/physics_model_checks.json`).
- `research/scratch/rt_micro.py`, `rt_micro2.py` (Philox prototype; `data/emulation_bench/runtime_micro{1,2}_20261010.json`).
- `research/scratch/bench_cascaded.py`, `bench_extras.py` (`data/emulation_bench/bench_aggregate.md`).
- `research/scratch/design_checks/{pv_checks, pv_wcl, fd_check, tu_check, wcl_check, tsui_check}.py`. These are already copied out of the session temp directory, which settles the evidence-durability concern. `fd_check`, `wcl_check` and `tsui_check` were re-run for this spec on 10 Oct 2026; outputs are quoted as [PILOT].
- `research/scratch/design_checks/design_calc.py` and `eb_pilot.py` (written this session; [DERIVED] / [PILOT]). They are already copied out of the session temp directory.
- `data/matlab/*` (existing CSVs encode the v2 formula; regenerate G1–G3/G7); `data/digitized/*` (Fig. 4(b), Fig. 2, Tang vector JSON).

**Key numbers.**

| Quantity | Value | Label |
|---|---|---|
| Float 0.5e-3 s at 23.04 MS/s | ceil → 11521; nearest / repr → 11520 | DERIVED |
| Thermal floor | −173.975 dBm/Hz; −113.975 dBm at 1 MS/s | DERIVED |
| Doppler | 350.242 Hz (3.5 GHz, 30 m/s); 3.336 Hz (1 GHz, 1 m/s) | DERIVED |
| CRB SD, N = 2000, 5880 S/s | 0.0256 / 0.0081 / 0.0026 Hz at 0 / 10 / 20 dB | DERIVED |
| Binary losses | η(1) −3.922 dB; lab pair −4.105 dB; |d| −0.18 dB; |m| −13.86 dB ∠127.9° | DERIVED |
| Narrowband band-edge error at πBΔτ = 0.05 (midpoint reference) | −32.0 dB peak, −36.8 dB RMS | DERIVED |
| Per-RX switch offset at 2 km | 153.7 samples (23.04 MS/s); 6.67 samples (1 MS/s) | DERIVED |
| FD worst-μ in-band error | Lagrange-4: −25.8 / −13.2 / −5.2 dB at 0.2 / 0.3 / 0.4 f_s; Kaiser 8β8 −71.4 (0.1); 16β8 −75.5 (0.3); 32β8 −76.5 (0.4); 48β8 −57.1 (0.45) | PILOT |
| Sub-aperture factor, λ/2 cell, retro 15/30/45/60° | −0.98 / −3.92 / −8.92 / −16.47 dB | PILOT |
| WCL geometry, K1, opt − LOS | 0.29 dB (h = 4), 0.18 dB (h = 6); URAP ≤ 0.01 dB | PILOT |
| Blocked corridor (−20 dB), h = 2, 1 GHz | opt 5.94 dB; τ_SUI 0.1/0.3/1/3/10 s → 4.61/2.73/1.43/0.88/0.69 dB | PILOT |
| E-B lab-like switch at RX1 | RIS term −12.7 … −15.6 dB, 53–66° rotation; transition midpoint −5 dB | PILOT |
| Tang 2022 | Model curves ≤ 0.014 dB; measured +0.98 dB bias, 0.38 dB SD, 1.05 dB RMSE (n = 36) | FACT |
| ANTS Fig. 2 | Digitized slope 1.0681; v1 1.0691; 11.76/11 = 1.06909 | FACT |
| Fig. 4(b) | 30 cells; unc. 0.045–0.22 dB; ring-mean-removed SD 2.74 dB | FACT |
| Runtime micro-benchmarks | Philox 72–78 µs per ms at 23.04 MS/s (16 threads); TCP 322 MS/s; REQ/REP p50 81 µs; sleep(0.5 ms) overshoot p99 0.88 ms; rotator drift 5e-13 rad/s | MEASURED |

