# V-LAB pre-registered protocol: ANTS Fig. 4(b) consistency check

Protocol `V-LAB-ANTS-Fig4b`, version 1.0. Machine-readable twin: `protocol.json` in this
directory; where the two differ, `protocol.json` is authoritative and the difference is a
defect. Specification source: sremu `docs/DESIGN.md` section 10.9 (with the lead amendments
v1.1 and decisions v1.2).

## 0. Status and disclosure

* **Blinded.** This protocol was written, and the decoy power analysis below was run, before
  any model-data comparison. The author of this file (the WP10 agent) never opened, read,
  hashed or loaded the digitized measured file (`fig4b_measured_rx_power.csv`) or any copy of
  its values.
* **What was known.** Only the facts disclosed in DESIGN.md section 10.9: the cell geometry
  (theta centres {0, 5, 15, 25, 35, 45} deg, radii {0.6, 0.9, 1.2, 1.5, 1.8} m, 30 cells),
  the per-cell digitization uncertainty range (0.045-0.22 dB, typical 0.07 dB), the two
  colorbar-pinned values (-10.06 and -33.20 dBm) and the descriptive statistics of section 1
  (computed earlier by the project lead). The analysis plan therefore follows a descriptive
  inspection of the map, but precedes every model-data comparison.
* **Lock.** The lead commits this file and `protocol.json`, then sets
  `protocol_lock.py::PREREGISTERED_SHA256` to their digest (section 12) in a separate commit.
  `load_measured()` refuses to read the data until the digest matches. The user deposits this
  file and the digest on Zenodo or OSF before unblinding (human gate H-6).

## 1. Data

| Item | Value |
|---|---|
| File | `fig4b_measured_rx_power.csv`, read from `$SREMU_PRIVATE_DATA` (`{root}/fig4b_measured_rx_power.csv` or `{root}/fig4b_measured/fig4b_measured_rx_power.csv`); private until lab permission |
| Data class | DIG (digitized from the ANTS manuscript's Fig. 4(b)) |
| Quantity | received power, dBm, at 3.5 GHz |
| Cells | 30: theta in {0, 5, 15, 25, 35, 45} deg (centres inferred) x r in {0.6, 0.9, 1.2, 1.5, 1.8} m |
| Cell order | ring-major: cell i = 6 ir + it (ir indexes r, it indexes theta) |
| Matching | each CSV row is mapped to the protocol cell within 2.5 deg and 0.075 m; every cell exactly once |
| sigma_dig | per cell, from the CSV (disclosed range 0.045-0.22 dB, typical 0.07 dB) |
| Pinned cells | -10.06 and -33.20 dBm are pinned to the colorbar limits under the autoscale assumption; a cell is pinned if it lies within 0.005 dB of a pinned value (fallback: the maximum / minimum cell, reported) |
| Unknown | the physical reference of r and of theta = 0 |

Disclosed descriptive statistics (no model): ring means -11.42 / -15.21 / -19.07 / -22.80 /
-23.09 dBm; ring SDs 0.97 / 1.82 / 1.13 / 5.24 / 3.27 dB; log-distance exponent n = 2.63
(RMSE 2.78 dB); ring-mean-removed SD 2.74 dB.

## 2. Naming rule

The study is a **consistency check under geometric uncertainty**. The word "validation" is
used only if the lab supplies the geometry (human gate H-1); that case needs a protocol
amendment before unblinding (a single geometry, sigma_geom from the supplied tolerances).

## 3. Geometry frames

Units m and deg; z up. The ground (model M3 only) is the plane z = 0. The Tx, the RIS centre
and the measurement plane are at the antenna height h. Measurement point i is

    p_i = (r_i cos(theta_i) + dx_i,  r_i sin(theta_i) + dy_i,  h + dz)

around the polar origin (0, 0). The four discrete frame hypotheses (polar origin x direction
of theta = 0), with the placement distance d and angle a:

| Hypothesis | RIS centre | Panel normal | Tx |
|---|---|---|---|
| O_R / normal | (0, 0, h) | (1, 0, 0) | (d cos a, d sin a, h) |
| O_R / plane | (0, 0, h) | (0, 1, 0) | (d sin a, d cos a, h) |
| O_T / normal | (d cos a, d sin a, h) | (-1, 0, 0) | (0, 0, h) |
| O_T / plane | (d cos a, d sin a, h) | (0, 1, 0) | (0, 0, h) |

Under O_R, a is the Tx incidence angle from the panel normal (negative: the side of the
normal opposite to the measurement sector). Under O_T, the roles of the Tx and the RIS centre
swap (DESIGN 10.9): a is the polar angle of the RIS centre in the grid frame, and the panel
faces the Tx. The panel's u axis is z x n (horizontal) and its v axis is +z; cells are
numbered n = iv n_u + iu (D17). The panel consists of tiles of 8 (u) x 5 (v) cells.

## 4. Geometry prior

Every model uses the **same** K prior draws. Draws come from the keyed sremu RNG
(`sremu.rng.generator(seed = 20261010, "trial", "vlab_prior/<k>")`), in the fixed order of
`protocol.json` `prior.draw_order`.

| Unknown | Prior |
|---|---|
| Polar origin | O_R, the RIS centre (P = 0.7), or O_T, the Tx (P = 0.3) |
| Direction of theta = 0 | panel normal (0.5) or panel plane (0.5) |
| Placement (Tx under O_R, RIS centre under O_T) | d ~ U[1, 3] m, a ~ U[-45, 0] deg, same height |
| Layout of {sigma1^2 sigma2^2 sigma1^2} | uniform over A: 3 x 1 tiles, columns [s1 s2 s1]; B: 6 x 1, [s1 s1 s2 s2 s1 s1]; C: 3 x 2, columns [s1 s2 s1] in both tile rows |
| Cell pitch | 42.8 mm square (0.5), or 38 mm (u) x 25.98 mm (v) INDICON cell (0.5); tile gap U[0, 5] mm |
| cell_model (model form) | `point` (0.5) or `po_patch` (0.5) |
| abs Gamma(s1), abs Gamma(s2) | U[0.6, 1.0] each, independent |
| Gamma phases | nominal phases of sremu table `lab_binary_3p5ghz` (49.64 and -153.77 deg) plus independent U[-15, 15] deg offsets |
| Antennas | isotropic (0.5), or a patch with G0 = 6 dBi (0.25) or 9 dBi (0.25); the same option for the Tx and every Rx point, each aimed at the RIS centre |
| Point position error | dx, dy ~ U[-2, 2] cm independently per point; measurement-plane height dz ~ U[-5, 5] cm (one value per draw) |
| Antenna height h | U[0.8, 1.5] m (enters M3 only) |
| Floor (M3 only) | concrete, ITU-R P.2040-3 Table 3 via `sremu.physics.walls.p2040_material`: eps_r = 5.24, sigma = 0.1231 S/m at 3.5 GHz (UNVERIFIED until the PDF is spot-checked); polarization H (0.5) or V (0.5) |

**Patch pattern.** G(psi) = G0 max(cos(psi)^alpha, 10^(-20/10)) with alpha = G0/2 - 1
(sremu `CosPower` in the front hemisphere), tabulated every 0.5 deg
(`antennas/patch_6dbi_fbr20.csv`, `antennas/patch_9dbi_fbr20.csv`; sremu `Tabulated`).

**Feasibility.** A candidate is rejected and redrawn from the same keyed stream (at most 10000
attempts) if the Tx or any point lies closer than one wavelength (85.65 mm) to the panel
rectangle (the K1 validity limit, where sremu raises an error) or if a Tx-point segment
crosses the panel rectangle. The prior used is therefore the stated prior conditioned on
physical feasibility; the realized acceptance rates per hypothesis are reported with the
tensor.

**po_patch quadrature.** sremu's Richardson check (0.05 dB, at most two halvings) starts at
the default sub-cell size lambda/5. If sremu raises `RefinementNotConverged` (points close to
the panel), the scene is rebuilt with lambda/10 and then lambda/20; a candidate that still
does not converge is rejected (reason `po_patch_not_converged`) and redrawn.

### 4.1 Interpretations and completions of DESIGN 10.9 (fixed here, before unblinding)

1. **Frames.** DESIGN gives the hypotheses in words; section 3 fixes the coordinates. Under
   "panel plane" the placement angle is measured from the panel normal (otherwise the Tx would
   lie behind the panel).
2. **Antenna back lobe [ENG].** A pure CosPower pattern is zero in the back hemisphere, so an
   Rx patch aimed at the RIS would receive no LOS power at many points and M0 would predict
   minus infinity. A 20 dB front-to-back floor is added.
3. **Antenna option weights [ENG].** "Isotropic, or CosPower with G in {6, 9} dBi" is read as
   isotropic 0.5, 6 dBi 0.25, 9 dBi 0.25, shared by the Tx and the Rx.
4. **Floor polarization [ENG].** Not given in DESIGN; H and V with equal weight.
5. **Point errors.** "+-2 cm uniform" per in-plane coordinate and point; "height plane
   +-5 cm" as one offset of the whole measurement plane.
6. **Layouts.** Tiles of 8 x 5 cells as in `scenarios/lab_like_A.yaml`; the INDICON cell has
   its 38 mm side along u.
7. **Feasibility truncation** and **po_patch escalation** as above (both forced by sremu's
   validity guards).
8. **Spearman decision statistic** (section 7) and the bootstrap direction (the CI must exclude
   0 on the positive side, the claim direction).
9. **Validity domain (disclosure).** Feasibility uses sremu's error limit (one wavelength).
   Draws with a node between one and three wavelengths from a cell are kept; sremu then emits
   a `ValidityWarning` (the point-scatterer approximation is marginal, DESIGN 5.3). The tensor
   summary reports how many draws carry such warnings
   (`draws_with_validity_warnings`); no draw is dropped or reweighted for it.

## 5. Models

Each model is evaluated per draw with `sremu.physics.propagation.from_scenario` (numba
backend) at t = 0 as a narrowband channel h = sum amp exp(-j 2 pi f_c tau), unit Tx power.
The drawn Gamma is applied to the separable RIS coefficients of the snapshot (checked
against `ScenePropagation.evaluate` within 1e-9 dB by `tests/unit/test_vlab_protocol.py`).
Powers below -300 dB are floored.

| Model | Definition |
|---|---|
| M0 | LOS only (Friis and antenna patterns); RIS absent |
| M0n | log-distance y = C - n 10 log10(d), d = 3-D distance from the drawn Tx to the drawn point; (C, n) by OLS on the training cells |
| M1 | LOS + K1 RIS path, every cell in state sigma1 (a mirror: does the configuration matter?) |
| M2 | LOS + K1 RIS path, lab configuration (layout) |
| M3 | M2 + concrete floor: ground image path and RIS hop variants 1-3 (sremu ground mechanism) |

Each model fits **one offset C (dB) on the training cells only**: C = mean over the training
cells of (y - p) (unweighted least squares). M0n fits (C, n). Errors are e = measured -
fitted prediction.

## 6. Splits

* Primary: train on the rings {0.6, 1.2, 1.8} m (18 cells), test on {0.9, 1.5} m (12 cells).
* Leave-one-ring-out (5 folds: test one ring, train on the other 24 cells).
* Leave-one-wedge-out (6 folds: test one theta wedge, train on the other 25 cells).

Decision rules use the primary split only. The primary analysis keeps all 30 cells.

## 7. Primary analysis (geometry is not fitted)

* K = 2000 prior draws. The prediction tensor `pred[K, 30, {M0, M1, M2, M3}]` and the
  distances for M0n are computed **once** with sremu and cached (`predict.py`; npz with a
  content SHA-256 and the SHA-256 of the protocol's prediction spec).
* Per draw: fit C on train, score on test: RMSE, bias (mean e), MAE, max |e|. Report the
  median and the 5-95 % range over draws, per model.
* Delta RMSE per draw: RMSE(M0) - RMSE(M2) (C2) and RMSE(M1) - RMSE(M2) (C3).
* Paired bootstrap of Delta RMSE per draw: 10^4 resamples of the 12 test cells with
  replacement (indices `rng.integers(0, 12, size=(B, 12))` from
  `sremu.rng.generator(20261010, "trial", "vlab_bootstrap/primary")`, shared by all draws and
  models), offsets not refitted, 95 % percentile interval. This is exactly
  `sremu.analysis.stats.paired_bootstrap(..., rmse_difference)` per draw. "The CI excludes
  0" means its lower bound is > 0.
* Spearman rho between the ring-mean-removed measured test values and the ring-mean-removed
  **prior-median** M2 prediction (per cell, the median over draws of the ring-mean-removed
  fitted prediction; ring means over the test cells of each ring), with the Fieller 95 %
  interval (`sremu.analysis.stats.spearman`, n = 12). The per-draw rho distribution is also
  reported. If either sample is constant, rho is undefined and the Spearman condition fails.
* M3 is reported with the same metrics; no decision rule uses it.

## 8. Secondary analysis, labelled "optimistic"

Per model, hypothesis weights w_k proportional to exp(-SSE_train,k / (2 sigma_tot^2)) with
**sigma_tot = 2.0 dB** (SSE with the fitted offset), and the weighted test prediction
sum_k w_k yhat_k. Report its test RMSE, bias, MAE, max |e|, the weighted Delta RMSE and the
Kish effective sample size 1/sum(w^2). Selecting a geometry by calibration RMSE alone is not
used. With sigma_tot = 2.0 dB and 18 training cells the weights can concentrate on very few
draws (ESS near 1); the analysis then approaches that selection, which is why the ESS is
reported next to every weighted number and the analysis has no decision role.

## 9. Uncertainty budget

| Component | Value |
|---|---|
| sigma_dig | per cell, from the CSV |
| sigma_pos | +-2 cm (xy) and +-5 cm (z) propagated through the local M2 gradient: central differences with 1 mm steps at fixed quadrature, uniform standard deviation half-width/sqrt(3), first 200 draws; reported for reference (point errors are already sampled in the prior, so sigma_pos is contained in sigma_geom) |
| sigma_rep | 1.0 dB assumed; sensitivity 0.5 and 2.0 dB |
| sigma_geom | per cell, standard deviation over draws of the fitted M2 prediction |

Total sigma_tot,i = sqrt(sigma_dig,i^2 + sigma_rep^2 + sigma_geom,i^2). Reported: the
median of each component over cells and the test-cell coverage of the 90 % interval
(z = 1.645) around the prior-median M2 prediction, at sigma_rep in {0.5, 1.0, 2.0} dB.
Descriptive only.

## 10. Decision rules (fixed before unblinding)

| Claim | Condition |
|---|---|
| C1 "absolute level consistent" | median test abs(bias) of M2 <= 1.5 dB **and** median RMSE(M2) <= median RMSE(M0n) + 0.5 dB |
| C2 "RIS model explains spatial structure" | all of: median Delta RMSE = RMSE(M0) - RMSE(M2) >= 1.0 dB; the paired-bootstrap 95 % CI of Delta RMSE (10^4 resamples over test cells) excludes 0 in >= 80 % of draws; Spearman rho >= 0.3 with its CI excluding 0 |
| C3 "configuration-specific" | C2 holds with M1 in place of M0 (same thresholds; the Spearman condition is the one of C2) |

If C2 fails, the paper states the null result: **the digitized map does not discriminate the
RIS model from LOS-only at the available geometric knowledge.**

Interpretation of C1 (fixed here, before unblinding): every model's offset C is fitted on the
training rings, which absorbs the unknown Tx power, cable losses and antenna gains. C1
therefore tests the level of the held-out rings relative to the training rings (the range
trend), not the absolute received power; the paper must not describe it as a check of
absolute power unless the lab supplies the Tx power and losses (human gate H-1) and the
protocol is amended before unblinding.

## 11. Sensitivity runs

* S1: primary analysis and decisions without the (35 deg, 1.5 m) cell.
* S2: primary analysis and decisions without both colorbar-pinned cells.
* S3: sigma_rep in {0.5, 2.0} dB: decoy detection rates (section 13) and test-cell coverage.
* S4: descriptive breakdown by discrete hypothesis (origin, theta = 0 direction, layout,
  pitch, cell_model, antenna, floor polarization): median test RMSE(M2) and Delta RMSE; no
  decisions.

Sensitivity bootstraps use the streams `vlab_bootstrap/S1` and `vlab_bootstrap/S2`.

## 12. Pre-registration lock

    digest = sha256( PROTOCOL.md bytes with CRLF -> LF  ||  canonical JSON of protocol.json )
    canonical JSON = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False, allow_nan=False).encode("utf-8")

`python -m validation.ants_fig4b.protocol_lock` prints the digest and the lock state;
`--write-canonical FILE` writes the canonical JSON for an external check
(`cat PROTOCOL.md FILE | sha256sum`, with LF line endings). Steps:

1. The lead commits `PROTOCOL.md` and `protocol.json` (with the power-analysis result below).
2. In a separate commit the lead sets `PREREGISTERED_SHA256` to the digest.
3. `load_measured()` refuses unless the digest of the files on disk equals it; it checks
   before resolving the data path, so the data file is not touched otherwise.
4. The user deposits this file and the digest on Zenodo or OSF (H-6).
5. Only then: `python -m validation.ants_fig4b.run --tensor <pred_tensor.npz>`. The digest
   covers this file and `protocol.json`, not the analysis code; the code is pinned by the
   commit in the run manifest, so `run.py` refuses a dirty sremu work tree unless
   `--allow-dirty` is passed (recorded in `results.json` and `meta.json`).

## 13. Power analysis (before unblinding)

* 500 decoys. Decoy i: the M2 prediction at a **fresh** prior draw (stream
  `("decoy", "vlab/<i>", counter 0)`, same feasibility rules), shifted so that its noiseless
  mean equals the mean of the disclosed ring means (-18.318 dBm), plus sigma_rep z1 +
  sigma_dig z2 with sigma_rep = 1.0 dB, sigma_dig ~ U[0.045, 0.22] dB per cell, z1, z2 ~
  N(0, 1) (stream counter 1; the same z for every sigma_rep).
* The primary and secondary analyses run on every decoy with the cached tensor and 2000
  bootstrap resamples per decoy (stream counter 2).
* Detection: C2 holds. Rate with a Clopper-Pearson 95 % interval.
* **If the C2 detection rate is below 0.5, the result is labelled "underpowered", whatever
  the outcome.**
* Sensitivity: the same decoys at sigma_rep = 0.5 and 2.0 dB.
* Specificity check (informational, no decision use): the same decoys with M0 (no RIS path)
  as the truth.

### 13.1 Power analysis result

<!-- POWER_ANALYSIS_RESULT:BEGIN (written by decoy.py --write-protocol-result) -->
Computed 2026-10-10T17:13:02.012279Z (sremu commit 2a016a65efe156db33380abaf0bf5b76d9699979, dirty True) from tensor `71d31b58fb9c4906...` (K = 2000), 500 decoys, 2000 bootstrap resamples each.

| Truth | sigma_rep (dB) | C1 rate | C2 rate (95 % CI) | C3 rate |
|---|---|---|---|---|
| M2 | 0.5 | 0.046 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M2 | 1 | 0.048 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M2 | 2 | 0.056 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M0 | 0.5 | 0.004 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M0 | 1 | 0.006 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M0 | 2 | 0.010 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |

C2 sub-conditions (fraction of decoys meeting each) and medians over decoys:

| Truth | sigma_rep (dB) | median Delta RMSE >= 1 dB | CI excludes 0 in >= 80 % of draws | Spearman | median Delta RMSE (dB) | secondary Delta RMSE (dB) |
|---|---|---|---|---|---|---|
| M2 | 0.5 | 0.000 | 0.000 | 0.288 | -0.26 | -0.02 |
| M2 | 1 | 0.000 | 0.000 | 0.264 | -0.24 | -0.01 |
| M2 | 2 | 0.000 | 0.000 | 0.220 | -0.21 | -0.03 |
| M0 | 0.5 | 0.000 | 0.000 | 0.462 | -0.33 | -0.23 |
| M0 | 1 | 0.000 | 0.000 | 0.448 | -0.32 | -0.19 |
| M0 | 2 | 0.000 | 0.000 | 0.380 | -0.28 | -0.18 |

**C2 detection rate (truth M2, sigma_rep 1 dB): 0.000** (95 % CI 0.000-0.007). This is below 0.5, so the study is labelled **underpowered** whatever the outcome.
Rows with truth M0 are the informational specificity check (C2 should rarely hold).
Under the M2 truth (sigma_rep 1 dB) C1 holds for 0.048 and C3 for 0.000 of the decoys; C1 compares the held-out rings after an offset fitted on the training rings, so it does not test the absolute received power.
The Spearman sub-condition of C2 holds for 0.448 of the decoys whose truth is M0 (no RIS path) and for 0.264 with the M2 truth, so on its own it does not indicate RIS-specific structure.
<!-- POWER_ANALYSIS_RESULT:END -->

## 14. Outputs of the unblinded run

`run.py` writes a run directory with `manifest.json` (commit, dirty flag, seeds, input
hashes, platform), `meta.json` (data classes, protocol digest, lock state), `F4.csv` (per
cell: measured DIG, prior-median M2 and M0 predictions SIM-sremu, residuals, sigma_geom),
`F4_delta_rmse.csv` (per draw), `F4.meta.json`, `table_v.csv` / `table_v.json` (Table V),
`results.json` (every analysis of sections 7-11) and `report.md`.

## 15. Paper wording template

Placeholders `{...}` are filled only by the paper pipeline from the run outputs.

* "The analysis plan was fixed and time-stamped (commit {X}, {date}; deposited at {DOI})
  after descriptive inspection of the digitized map but before any model-data comparison."
* "Under geometric uncertainty (K = {K} prior draws), the lab-configuration model gives a
  median test RMSE of {rmse_m2} dB (5-95 %: {rmse_m2_lo}-{rmse_m2_hi} dB) against {rmse_m0}
  dB for LOS only (C1 {c1}, C2 {c2}, C3 {c3}); a decoy power analysis gives a C2 detection
  rate of {rate}."
* If C2 fails: "The digitized map does not discriminate the RIS model from LOS-only at the
  available geometric knowledge."
* If underpowered: "The comparison is underpowered (decoy C2 detection rate {rate} < 0.5), so
  this outcome is not evidence for or against the model." The decoy rates of C1 and C3 under
  the simulated true model ({rate_c1}, {rate_c3}) are reported with it.
* Data classes: measured DIG; predictions SIM-sremu; Gamma magnitudes and geometry ASSUMED.
* The study is called a "consistency check under geometric uncertainty" (section 2).

## 16. Reproduction

From the sremu repository root, with the project interpreter:

    python -m validation.ants_fig4b.predict --jobs 8          # tensor, K = 2000
    python -m validation.ants_fig4b.decoy --tensor <pred_tensor.npz> --jobs 8
    python -m validation.ants_fig4b.run --tensor <pred_tensor.npz> --dry-run-decoy 0   # pipeline check
