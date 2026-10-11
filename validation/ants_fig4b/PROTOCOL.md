# V-LAB pre-registered protocol: ANTS Fig. 4(b) consistency check

Protocol `V-LAB-ANTS-Fig4b`, version 2.0. Machine-readable twin: `protocol.json` in this
directory; where the two differ, `protocol.json` is authoritative and the difference is a
defect. Specification source: sremu `docs/DESIGN.md` section 10.9 (with the lead amendments
v1.1 and decisions v1.2), with the geometry prior re-specified in v2 from lab facts supplied
by the author (Revision history; `protocol.json` `author_facts`).

## Revision history

| Version | Date | State | Summary |
|---|---|---|---|
| 1.0 | 2026-10-10 | commit b03e52d; superseded before unblinding, never locked | Broad geometry prior of DESIGN 10.9: polar origin O_R (P = 0.7) or O_T (0.3); theta = 0 along the panel normal or in the panel plane; one flat panel with layouts A/B/C; Tx at d ~ U[1, 3] m, a ~ U[-45, 0] deg; isotropic or patch antennas aimed at the RIS; plane height +-5 cm. Power analysis: C2 detection rate 0.000 (0/500 decoys; 95 % CI 0.000-0.007), labelled **underpowered** (run `runs/vlab/20261010T165335Z_vlab_power`, tensor `71d31b58fb9c4906...`). |
| 2.0 | 2026-10-11 | current; not locked | Primary prior re-specified from the author-supplied lab facts 1-7 below (including the author's hand sketch on ANTS Fig. 4(b)): polar origin = Tx; six 5 x 8 tiles [s1 s1 s2 s2 s1 s1] on an arc centred on the Tx, each facing the Tx, just behind the 1.8 m ring; vertical whip antennas; plane height +-2 cm. Near-field handling fixed by a lead decision (section 4.2). Sensitivity runs S-OR (the v1 O_R rows) and S-NF (without the 1.8 m ring) added. New tensor and power analysis: C2 detection rate 0.000 (0/500 decoys; 95 % CI 0.000-0.007), still labelled **underpowered**; specificity 1.000 (section 13.1). |

**Blinding of v2.** Version 2.0 was specified before any unblinding. Neither the author of
v1 nor the author of v2 (WP10 agents) opened, read, listed, hashed or loaded the digitized
measured file or any copy of its values; v2 changes the prior only from the author's
statements about the lab, which do not refer to the measured values. Two intermediate v2
drafts (a flat-panel O_T prior, then an arc prior that forced `po_patch`) were stopped by the
lead before completion when the author supplied the arc placement; neither produced a power
result or was locked. With the v2 prior of this document, a first tensor run stopped with an
error in the sigma_pos step (a 1 mm finite-difference step brought a point within one
wavelength of a cell) before writing any result, and its run directory was deleted; the
sigma_pos NaN rule of section 9 was then added to `protocol.json` (it has no decision role),
and the tensor of section 13 was computed. (Recorded by the independent reviewer,
2026-10-11.)

**Independent review (2026-10-11, before unblinding; data-independent).** A reviewer who did
not write the WP10 code re-implemented the fits, metrics, paired bootstrap, Spearman/Fieller
interval and decision rules from this document with numpy/scipy (no import of the WP10
analysis code) and re-ran the power analysis on the cached tensor and decoy truths with the
protocol's keyed streams: all 3000 decoy rows (500 decoys x truths M2/M0 x sigma_rep
0.5/1/2 dB, primary and S-NF splits) agree with the power run (largest difference 4e-16; no
decision differs). With independent noise and bootstrap streams (200 decoys, sigma_rep 1 dB) C2
is again 0/200 (truth M2; median Delta RMSE -1.16 dB) and 0/200 (truth M0). An independent
numpy element-wise K1 sum built from facts 1-5 and 7 reproduces the point-model tensor rows
and decoy truths (M0, M1, M2) within 0.001 dB, and the `po_patch` rows within 0.03 dB at the
recorded sub-cell size, which checks the arc geometry, tile orientation, state order and the
M1 definition. The prediction-spec hash of this `protocol.json` equals the one the tensor and
the decoy truths were computed with.

**Author-supplied lab facts (provenance).** Galaba Vamsi (author of the ANTS manuscript),
chat with the project lead on 2026-10-11, including a hand sketch on ANTS Fig. 4(b); relayed
to the protocol author by the lead. Recorded as relayed:

1. *Polar measurement origin = the Tx (O_T).* The author first said "RIS centre", then
   corrected himself; in panel (a) of the ANTS figure the Tx triangle sits at the apex of the
   node fan, consistent with O_T. Primary prior: O_T with probability 1; one labelled
   sensitivity analysis S-OR uses the v1 O_R rows.
2. *RIS = six 5x8 tiles in the order s1 s1 s2 s2 s1 s1 placed along an arc centred on the
   Tx, just behind the last arc of measurement positions, each tile turned to face the Tx*
   (tile normal pointing at the Tx). Tile k centre at (R cos phi_k, R sin phi_k, h) in the
   grid frame, phi the same polar angle as the measurement theta; normal
   = -(cos phi_k, sin phi_k, 0); the tile's in-plane horizontal axis is tangential.
3. *Radius:* "right behind (~10-20 cm)" the last arc of positions (cell centres at
   r = 1.8 m), i.e. R ~ U[1.90, 2.00] m, one R per draw shared by all tiles; the protocol may
   widen this to U[1.88, 2.05] m if justified before computing (section 4.1, item 1).
4. *Tiles are mounted with the short side (5 cells) horizontal along the arc and the long
   side (8 cells) vertical.*
5. *The row covers the whole 0-45 deg sector, extending a little past both ends:* tile
   centres evenly spaced in phi over a span S ~ U[40, 52] deg whose centre is
   22.5 deg + U[-3, 3] deg (one value each per draw); phi_k = centre + (k + 0.5 - 3) S/6,
   k = 0..5, order from the lowest phi s1 s1 s2 s2 s1 s1 (symmetric). Gaps between tiles
   along the arc are inert.
6. *Radios:* Ettus B200 with VERT2450 and VERT900 vertical whips (neither specified for
   3.5 GHz; absolute gain absorbed by the fitted offset C), co-polarized vertical,
   azimuth-omnidirectional; all coplanar, so evaluated at about 0 deg elevation (isotropic,
   or a short-dipole elevation pattern as a 50/50 model-form alternative).
7. *Tx, RIS centres and Rx at the same height;* measurement-plane height uncertainty
   +-2 cm (one per draw); in-plane point error +-2 cm.

Unchanged from v1: the cell pitch options, the Gamma priors, the floor of M3, the models
M0/M0n/M1/M2/M3, the splits, the decision rules C1-C3, the bootstrap sizes, the 500-decoy
power analysis with its "underpowered" rule, and the naming rule.

## 0. Status and disclosure

* **Blinded.** This protocol was written, and the decoy power analysis below was run, before
  any model-data comparison (see "Blinding of v2" above).
* **What was known.** The facts disclosed in DESIGN.md section 10.9: the cell geometry
  (theta centres {0, 5, 15, 25, 35, 45} deg, radii {0.6, 0.9, 1.2, 1.5, 1.8} m, 30 cells),
  the per-cell digitization uncertainty range (0.045-0.22 dB, typical 0.07 dB), the two
  colorbar-pinned values (-10.06 and -33.20 dBm) and the descriptive statistics of section 1
  (computed earlier by the project lead); and, for v2, the author's lab facts above. The
  analysis plan therefore follows a descriptive inspection of the map, but precedes every
  model-data comparison.
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
| Reference of r and theta | v1: unknown. v2: r is measured from the Tx (fact 1); theta = 0 is fixed relative to the tile arc through the arc prior (fact 5) |

Disclosed descriptive statistics (no model): ring means -11.42 / -15.21 / -19.07 / -22.80 /
-23.09 dBm; ring SDs 0.97 / 1.82 / 1.13 / 5.24 / 3.27 dB; log-distance exponent n = 2.63
(RMSE 2.78 dB); ring-mean-removed SD 2.74 dB.

## 2. Naming rule

The study is a **consistency check under geometric uncertainty**. The word "validation" is
used only if the lab supplies the geometry (human gate H-1); that case needs a protocol
amendment before unblinding (a single geometry, sigma_geom from the supplied tolerances).
The author's facts of v2 are recollections with ranges, not a supplied geometry, so the label
is unchanged.

## 3. Geometry frames

Units m and deg; z up. The ground (model M3 only) is the plane z = 0. The Tx, the tile
centres and the measurement plane are at the antenna height h. Measurement point i is

    p_i = (r_i cos(theta_i) + dx_i,  r_i sin(theta_i) + dy_i,  h + dz)

around the polar origin (0, 0).

**Primary (O_T, arc).** The Tx is at (0, 0, h). The RIS is six tiles k = 0..5, numbered from
the lowest polar angle phi:

    phi_k   = centre + (k + 0.5 - 3) S/6,      centre = 22.5 deg + offset
    centre_k = (R cos phi_k, R sin phi_k, h)
    n_k     = -(cos phi_k, sin phi_k, 0)                (faces the Tx)
    u_k     = z x n_k = (sin phi_k, -cos phi_k, 0)      (horizontal, tangential: the 5-cell side)
    v_k     = n_k x u_k = (0, 0, 1)                     (vertical: the 8-cell side)

Tile k is one sremu RIS panel with this pose, 5 (u) x 8 (v) cells, cells numbered
n = iv n_u + iu (D17); every cell of tile k has the state of position k in
[s1 s1 s2 s2 s1 s1].

**Sensitivity run S-OR (the v1 O_R rows).** The RIS centre is the polar origin and the six
tiles form one flat row:

| Hypothesis | RIS centre | Panel normal | Tx |
|---|---|---|---|
| O_R / normal | (0, 0, h) | (1, 0, 0) | (d cos a, d sin a, h) |
| O_R / plane | (0, 0, h) | (0, 1, 0) | (d sin a, d cos a, h) |

with a the Tx incidence angle from the panel normal (negative: the side of the normal
opposite to the measurement sector); tile k centre c + (k - 2.5)(5 pitch_u + gap) u with
u = z x n (D17: spacing across a tile boundary is the tile width plus the gap).

## 4. Geometry prior

Every model uses the **same** K prior draws. Draws come from the keyed sremu RNG
(`sremu.rng.generator(seed = 20261010, "trial", "vlab_prior/<k>")`), in the fixed order of
`protocol.json` `prior.draw_order` (geometry block in `prior.geometry.draw_order`).

| Unknown | Prior (v2 primary) |
|---|---|
| Polar origin | O_T, the Tx (P = 1; fact 1). Alternative O_R: sensitivity run S-OR |
| Tile arc radius R (Tx to every tile centre) | U[1.88, 2.05] m, one per draw (fact 3; author's U[1.90, 2.00] m widened before computing, section 4.1) |
| Arc span S | U[40, 52] deg (fact 5) |
| Arc centre | 22.5 deg + U[-3, 3] deg (fact 5) |
| Tiles | six tiles of 5 (horizontal, along the arc) x 8 (vertical) cells, states [s1 s1 s2 s2 s1 s1] from the lowest phi, each facing the Tx (facts 2, 4, 5); gaps between tiles inert |
| Cell pitch | 42.8 mm square (0.5), or the INDICON cell (0.5) with 25.98 mm along the horizontal 5-cell side and 38 mm along the vertical 8-cell side (tile 214 x 342.4 mm or 129.9 x 304 mm) |
| cell_model (model form) | `point` (0.5) or `po_patch` (0.5) |
| abs Gamma(s1), abs Gamma(s2) | U[0.6, 1.0] each, independent |
| Gamma phases | nominal phases of sremu table `lab_binary_3p5ghz` (49.64 and -153.77 deg) plus independent U[-15, 15] deg offsets |
| Antennas (model form) | isotropic (0.5), or a vertical short dipole G = 1.5 sin^2(psi), psi from +z (0.5); the same option for the Tx and every Rx point; azimuth-omnidirectional, not aimed (fact 6) |
| Point position error | dx, dy ~ U[-2, 2] cm independently per point; measurement-plane height dz ~ U[-2, 2] cm (one value per draw) (fact 7) |
| Antenna height h | U[0.8, 1.5] m (enters M3 only) |
| Floor (M3 only) | concrete, ITU-R P.2040-3 Table 3 via `sremu.physics.walls.p2040_material`: eps_r = 5.24, sigma = 0.1231 S/m at 3.5 GHz (UNVERIFIED until the PDF is spot-checked); polarization V (vertical whips) |

**Dipole pattern.** G(psi) = 1.5 max(sin^2 psi, 10^(-40/10)), psi the angle from the
vertical antenna axis, tabulated every 0.5 deg (`antennas/short_dipole_z_null40.csv`, written
by `prior.write_antenna_tables`; sremu `Tabulated` with boresight +z).

**Feasibility.** A candidate is rejected and redrawn from the same keyed stream (at most
10000 attempts) if the Tx or any point lies closer than one wavelength (85.65 mm) to a tile
rectangle (reason `near_tile`; the K1 error limit, so no node is closer than one wavelength
to any cell or sub-cell) or if two tiles overlap (`tiles_overlap`; impossible within the
prior bounds: the smallest tile spacing, 40/6 = 6.67 deg, exceeds the largest angular tile
width, 2 atan(0.107/1.88) = 6.52 deg). The prior used is therefore the stated prior
conditioned on physical feasibility; the realized acceptance rate and the realized R
quantiles are reported with the tensor.

**po_patch quadrature.** sremu's Richardson check (0.05 dB, at most two halvings) runs per
tile and starts at the default sub-cell size lambda/5. If sremu raises
`RefinementNotConverged`, the scene is rebuilt with lambda/10 and then lambda/20; a candidate
that still does not converge is rejected (reason `po_patch_not_converged`) and redrawn. The
initial sub-cell size, the largest Richardson level and estimate over the tiles and the
converged sub-cell side are recorded per draw.

### 4.1 Interpretations and completions (fixed here, before unblinding)

1. **Arc radius widened [ENG].** Decided before any v2 prediction was computed. The author's
   "~10-20 cm" behind the 1.8 m ring is an approximate recollection, and it does not say
   whether the gap is measured to the tile face, its centre plane or its mount; the tiles are
   flat, so their edges lie up to 1.2 cm further out than their centres. A uniform prior on
   exactly [1.90, 2.00] m would give zero probability to a 9 cm or a 21 cm gap, which the
   statement does not exclude. The interval is therefore widened, as the lead allowed, to
   U[1.88, 2.05] m (gap 8-25 cm). Its lower end is truncated by the one-wavelength rule (a
   gap below 8.57 cm plus the outward point error is infeasible).
2. **Tile orientation and pitch.** Fact 4 puts the 5-cell side horizontal (u, tangential)
   and the 8-cell side vertical (v). The INDICON cell keeps its v1 orientation relative to
   the tile (38 mm along the 8-cell side), so in both pitch options the 5-cell side is the
   short side.
3. **One sremu panel per tile.** sremu supports several panels with independent poses; each
   tile is its own panel (one tile, no internal gap). The gaps between tiles along the arc
   carry no reflection (fact 5).
4. **Antenna elevation pattern [ENG].** Fact 6 leaves the elevation pattern open. All nodes
   are coplanar, but the outer points see the 0.30-0.34 m tall tiles from 0.10-0.20 m, i.e.
   cells at up to about 60 deg elevation, so the pattern matters at the 1.8 m ring. Neither
   whip is specified at 3.5 GHz, so neither pattern can be justified alone: both are kept as
   a 50/50 model-form alternative (like `cell_model`): isotropic, or the short dipole
   1.5 sin^2 psi (the broadest dipole-like doughnut). The 40 dB null floor only keeps the
   table finite. The LOS gain (psi near 90 deg) is absorbed by the fitted offset C.
5. **Floor polarization.** Vertical whips over a horizontal floor: the field lies in the
   plane of incidence, sremu polarization "V" (TM), with probability 1 (v1: H or V).
6. **LOS rule not applied to the arc.** Every tile point is at least R >= 1.88 m from the Tx
   and every measurement point at most 1.8 + 0.02 sqrt(2) < 1.83 m from it, so no Tx-point
   segment can reach a tile; the v1 rule `los_through_panel` can never fire and is not
   applied (`prior.feasibility.los_must_not_cross_tiles = false`). The one-wavelength rule
   (`near_panel` in v1, `near_tile` now) still applies and does truncate the prior. Both v1
   rules apply unchanged in S-OR.
7. **Point errors.** "+-2 cm" per in-plane coordinate and point; "+-2 cm" plane height as
   one offset of the whole measurement plane relative to the Tx and tile centres.
8. **Spearman decision statistic** (section 7) and the bootstrap direction (the CI must
   exclude 0 on the positive side, the claim direction), as in v1.
9. **S-OR prior.** "The v1 O_R rows" is read as: polar origin = RIS centre, theta = 0 along
   the panel normal or in the panel plane (0.5 / 0.5), Tx at d ~ U[1, 3] m and
   a ~ U[-45, 0] deg, the six tiles as one flat row with a tile gap U[0, 5] mm, and the v1
   feasibility rules; every other v2 entry (tiles, orientation, pitch, cell_model, Gamma,
   antennas, point errors, floor) is unchanged.

### 4.2 Near-field handling (lead decision, fixed before computing)

The outer ring (r = 1.8 m) is about 0.10-0.20 m (1.2-2.3 wavelengths at 3.5 GHz) from the
tiles.

* **Model.** The recommended analytic model here is the element-wise near-field summation
  (ETSI GR RIS 003 element-wise model; Tang et al. 2022 cell pattern), which is sremu's K1
  kernel with the exact distance from the Tx and from each point to every cell (`point`) or
  sub-cell (`po_patch`); no far-field or plane-wave approximation of a tile is made. Each unit
  cell (about lambda/2) is in its own far field beyond about 0.5 lambda, so the point-cell
  model remains applicable at >= 1 lambda.
* **Model form.** The v1 prior `cell_model` point / po_patch 0.5 / 0.5 is kept (`po_patch`
  is not forced). `po_patch` uses the Richardson refinement of section 4 and records the
  converged sub-cell size.
* **Warnings.** sremu emits a `ValidityWarning` for a node within 3 lambda of a cell
  ("point-scatterer approximation is marginal"); it is expected at the outer ring. Only this
  warning is suppressed; it is counted per draw, and the minimum point-to-cell distance is
  recorded per draw. Any other `ValidityWarning` is counted and reported.
* **Rejection.** A draw is rejected only if a node is < 1 lambda from a tile (section 4).
* **Validity statement.** The 1.8 m ring is at the edge of the element-wise model's
  validity. Mutual coupling between cells and the cells' reactive near field are not
  modelled (nor multiple scattering between the tiles, the antennas and their mounts). A
  full-wave (MoM/FDTD) model of the real tiles would be the rigorous reference; it is
  impossible here because the cell design files are unavailable.
* **Checks.** Sensitivity run S-NF (section 11) repeats the primary analysis without the
  1.8 m ring. The point-vs-patch spread at each ring is reported from paired evaluations: for
  the first 200 prior draws the same geometry is also evaluated with the other cell model
  (`pred_alt_cm_db` in the tensor).

## 5. Models

Each model is evaluated per draw with `sremu.physics.propagation.from_scenario` (numba
backend) at t = 0 as a narrowband channel h = sum amp exp(-j 2 pi f_c tau), unit Tx power,
with one RIS panel per tile. The drawn Gamma is applied to the separable RIS coefficients of
every tile in the snapshot (checked against `ScenePropagation.evaluate` within 1e-9 dB by
`tests/unit/test_vlab_protocol.py`). Powers below -300 dB are floored.

| Model | Definition |
|---|---|
| M0 | LOS only (Friis and antenna patterns); RIS absent |
| M0n | log-distance y = C - n 10 log10(d), d = 3-D distance from the drawn Tx to the drawn point; (C, n) by OLS on the training cells |
| M1 | LOS + K1 RIS paths of the six tiles, every cell in state sigma1 (a uniform mirror arc: does the configuration matter?) |
| M2 | LOS + K1 RIS paths of the six tiles, lab configuration [s1 s1 s2 s2 s1 s1] |
| M3 | M2 + concrete floor: ground image path and RIS hop variants 1-3 of every tile (sremu ground mechanism) |

Each model fits **one offset C (dB) on the training cells only**: C = mean over the training
cells of (y - p) (unweighted least squares). M0n fits (C, n). Errors are e = measured -
fitted prediction.

## 6. Splits

* Primary: train on the rings {0.6, 1.2, 1.8} m (18 cells), test on {0.9, 1.5} m (12 cells).
* Leave-one-ring-out (5 folds: test one ring, train on the other 24 cells).
* Leave-one-wedge-out (6 folds: test one theta wedge, train on the other 25 cells).
* S-NF (sensitivity, section 11): train on {0.6, 1.2} m (12 cells), test on {0.9, 1.5} m.

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
| sigma_pos | +-2 cm (xy) and +-2 cm (z) propagated through the local M2 gradient: central differences with 1 mm steps at fixed quadrature, uniform standard deviation half-width/sqrt(3), first 200 draws; a draw in which a 1 mm step brings a point within one wavelength of a cell (sremu `GeometryError`) gets NaN and is excluded; reported for reference (point errors are already sampled in the prior, so sigma_pos is contained in sigma_geom) |
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
* S4: descriptive breakdown by discrete prior variable (pitch, cell_model, antenna) and by
  tertile of the arc radius R: median test RMSE(M2) and Delta RMSE; no decisions.
* **S-OR** (alternative geometry prior): the v1 O_R rows (section 3; section 4.1, item 9)
  with every other v2 entry unchanged, its own keyed draws
  (`"trial", "vlab_prior_sor/<k>"`) and its own tensor (K = 2000, computed before
  unblinding with `predict.py --variant S-OR`; content hash in `protocol.json`
  `sensitivity_runs` S-OR `tensor_result`). Primary analysis and decisions C1-C3 on the
  primary split. Labelled sensitivity; it does not change the primary decisions. The S-OR
  tensor was computed on 2026-10-11 before unblinding (run
  `runs/vlab/20261010T223008Z_vlab_tensor_S-OR`, content SHA-256 `ce4d8d982233c8bc...`;
  acceptance rate 0.753, every rejection an O_R/plane draw with a point within one
  wavelength of a tile, so 34.5 % of the accepted draws are O_R/plane); `run.py` refuses
  another S-OR tensor unless `--allow-tensor-mismatch` is passed (recorded). **Implicit
  conditioning (recorded by the independent reviewer after the S-OR tensor was computed; the
  S-OR prior and tensor are unchanged):** under O_R/plane the theta = 0 points lie in the panel
  plane, and with the 42.8 mm cell the flat row of six tiles is 1.284 m long, so it passes
  within one wavelength of them; every accepted O_R/plane draw therefore has the INDICON cell
  (690 of 690, row 0.78 m long), against 639 of 1310 (48.8 %) among the O_R/normal draws. S-OR
  results for O_R/plane are conditional on the INDICON cell.
* **S-NF** (near field): primary analysis and decisions C1-C3 without the 1.8 m ring: the
  offsets are fitted on the rings {0.6, 1.2} m and the test cells are the primary ones
  ({0.9, 1.5} m). Also reported: the point-vs-patch spread per ring, in particular at the
  1.8 m ring (section 4.2). Labelled sensitivity; it does not change the primary decisions.

Sensitivity bootstraps use the streams `vlab_bootstrap/S1`, `vlab_bootstrap/S2`,
`vlab_bootstrap/S-OR` and `vlab_bootstrap/S-NF`.

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
5. Only then: `python -m validation.ants_fig4b.run --tensor <pred_tensor.npz> --tensor-sor
   <S-OR pred_tensor.npz>`. The digest covers this file and `protocol.json`, not the analysis
   code; the code is pinned by the commit in the run manifest, so `run.py` refuses a dirty
   sremu work tree unless `--allow-dirty` is passed (recorded in `results.json` and
   `meta.json`). It also refuses, before any data access, when the S-OR tensor is missing.

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
  as the truth; the C2 rate is the false-detection rate and 1 minus it the specificity.
* Information only: the S-NF split is also analysed on every decoy (same bootstrap indices).
* Spread of the prior predictions: v2 against the v1 tensor (data-independent).

### 13.1 Power analysis result

<!-- POWER_ANALYSIS_RESULT:BEGIN (written by decoy.py --write-protocol-result) -->
Computed 2026-10-10T23:02:26.242227Z (sremu commit 9dca79f37955dc38fa8890110bea0f342eea207d, dirty True) from tensor `d365f555a1809e36...` (K = 2000), 500 decoys, 2000 bootstrap resamples each.

| Truth | sigma_rep (dB) | C1 rate | C2 rate (95 % CI) | C3 rate |
|---|---|---|---|---|
| M2 | 0.5 | 0.000 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M2 | 1 | 0.000 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M2 | 2 | 0.002 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M0 | 0.5 | 0.000 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M0 | 1 | 0.000 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |
| M0 | 2 | 0.000 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 |

C2 sub-conditions (fraction of decoys meeting each) and medians over decoys:

| Truth | sigma_rep (dB) | median Delta RMSE >= 1 dB | CI excludes 0 in >= 80 % of draws | Spearman | median Delta RMSE (dB) | secondary Delta RMSE (dB) |
|---|---|---|---|---|---|---|
| M2 | 0.5 | 0.000 | 0.000 | 0.038 | -1.19 | -0.15 |
| M2 | 1 | 0.000 | 0.000 | 0.034 | -1.16 | -0.19 |
| M2 | 2 | 0.000 | 0.000 | 0.026 | -1.05 | -0.26 |
| M0 | 0.5 | 0.000 | 0.000 | 0.016 | -2.49 | -0.15 |
| M0 | 1 | 0.000 | 0.000 | 0.016 | -2.13 | -0.10 |
| M0 | 2 | 0.000 | 0.000 | 0.016 | -1.60 | -0.11 |

**C2 detection rate (truth M2, sigma_rep 1 dB): 0.000** (95 % CI 0.000-0.007). This is below 0.5, so the study is labelled **underpowered** whatever the outcome.
Rows with truth M0 are the informational specificity check (C2 should rarely hold).
Specificity (truth M0, no RIS path, sigma_rep 1 dB): C2 fires for 0.000 (0/500; 95 % CI 0.000-0.007) of the decoys, i.e. specificity 1.000.
Under the M2 truth (sigma_rep 1 dB) C1 holds for 0.000 and C3 for 0.000 of the decoys; C1 compares the held-out rings after an offset fitted on the training rings, so it does not test the absolute received power.
The Spearman sub-condition of C2 holds for 0.016 of the decoys whose truth is M0 (no RIS path) and for 0.034 with the M2 truth.

S-NF (information only; offsets fitted without the 1.8 m ring, same test cells and bootstrap indices):

| Truth | sigma_rep (dB) | C1 rate | C2 rate (95 % CI) | C3 rate | median Delta RMSE (dB) |
|---|---|---|---|---|---|
| M2 | 0.5 | 0.034 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 | -1.19 |
| M2 | 1 | 0.040 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 | -1.16 |
| M2 | 2 | 0.058 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 | -1.04 |
| M0 | 0.5 | 0.000 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 | -2.49 |
| M0 | 1 | 0.000 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 | -2.13 |
| M0 | 2 | 0.012 | 0.000 (0/500; 95 % CI 0.000-0.007) | 0.000 | -1.59 |

Spread of the prior predictions (v1 prior vs this prior; data-independent):

| Quantity | v1 | this protocol |
|---|---|---|
| median over cells of the SD over draws of the fitted M0 prediction (dB) | 3.36 | 0.08 |
| the same over the test cells (dB) | 2.90 | 0.09 |
| median over cells of the SD over draws of the fitted M2 prediction (dB) | 4.11 | 2.73 |
| the same over the test cells (dB) | 3.96 | 3.10 |
| RIS signature: median over draws of the SD over cells of M2 - M0 (dB) | 2.26 | 2.69 |
| ring-mean-removed M2: median over test cells of the SD over draws (dB) | 3.45 | 2.86 |
| median over decoys of the median Delta RMSE (M0 - M2), truth M2, sigma_rep 1 dB (dB) | -0.24 | -1.16 |
<!-- POWER_ANALYSIS_RESULT:END -->

**Reading of the v2 result (descriptive, written after the power analysis and before any
unblinding).** The author's facts remove almost all LOS uncertainty (the SD over draws of the
fitted M0 prediction falls from 3.4 dB to 0.08 dB, because the Tx is the polar origin) and make
the RIS signature large in every draw (SD over cells of M2 - M0: 5 % quantile 1.8 dB, against
0.1 dB in v1). The fitted M2 prediction, however, still varies by 2.7 dB (median over cells;
3.1 dB on the test cells) from draw to draw, as much as the signature itself: the RIS term
interferes with the LOS at path differences of about 2 (R - r), so the residual uncertainty of
R (+-8 cm) and of every point position (+-2 cm, giving a median sigma_pos of 2.6 dB) scrambles
the interference phase per cell. A draw of the prior therefore predicts a ripple that is
uncorrelated with the true one, so M2 adds error relative to M0 (median Delta RMSE -1.16 dB
under the M2 truth, against -0.24 dB in v1) and the prior-median structure carries little
information (the Spearman condition holds for 3 % of the M2-truth decoys). The point-vs-patch
spread at the 1.8 m ring (paired draws: median 0.12 dB, 95th percentile 0.65 dB after the
offset) is small compared with this, and S-NF (without the 1.8 m ring) is equally
underpowered, so the near-field ring is not what limits the power.

### 13.2 Diagnostic, not part of the protocol

<!-- DIAGNOSTIC:BEGIN -->
**Label: diagnostic, not part of the protocol.** Computed 2026-10-10T23:50:02.715465Z (sremu commit b4247316d882116b27e7e10768d537319c626941, dirty True); run `runs\vlab\20261010T232914Z_vlab_diagnostic` (rows of `runs\vlab\20261010T224804Z_vlab_diagnostic` merged). Truth M2 (detection) and M0 (false detection), sigma_rep 1 dB, 2000 bootstrap resamples per decoy. Subset facts use the protocol's tensor and its 500 decoys (the prior conditioned on knowing the parameter to +-tol around the truth; decoys with fewer than 20 consistent draws are skipped); restricted facts use a K = 1000 tensor and 200 fresh decoys from the narrowed prior.

| Fact to ask the author for | Method | C2 rate, truth M2 (95 % CI) | C2 rate, truth M0 | median Delta RMSE, truth M2 (dB) | median draws |
|---|---|---|---|---|---|
| none (protocol v2 as pre-registered) | power analysis | 0.000 (0.000-0.007) | 0.000 | -1.16 | 2000 |
| Tx-to-tile distance R to +-1 cm | subset | 0.000 (0.000-0.007) | 0.000 | -1.15 | 255 |
| Tx-to-tile distance R to +-2.5 mm | subset | 0.000 (0.000-0.007) | 0.000 | -1.13 | 64 |
| angular positions of the tiles (arc centre to +-0.5 deg, span to +-1 deg) | subset | 0.000 (0.000-0.007) | 0.000 | -1.15 | 53 |
| which unit cell (42.8 mm square or INDICON) | subset | 0.000 (0.000-0.007) | 0.000 | -1.19 | 964 |
| the whips' elevation pattern at 3.5 GHz (isotropic-like or dipole-like) | subset | 0.000 (0.000-0.007) | 0.000 | -1.17 | 999 |
| the two tile states' reflection coefficients (\|Gamma\| to +-0.1, phase to +-5 deg) | subset | 0.000 (0.000-0.008) | 0.000 | -1.15 | 37 |
| the 30 measurement positions and the plane height to +-5 mm | restricted | 0.000 (0.000-0.018) | 0.000 | -1.05 | 1000 |
| the 30 measurement positions and the plane height to +-1 mm | restricted | 0.000 (0.000-0.018) | 0.000 | -1.01 | 1000 |
| (combination: R to +-2.5 mm and the positions to +-5 mm) | restricted | 0.000 (0.000-0.018) | 0.000 | -0.36 | 1000 |
| (combination: the as-built tile geometry exactly (R, arc angles, cell type); positions still +-2 cm) | restricted | 0.000 (0.000-0.018) | 0.000 | -1.37 | 1000 |
| (combination: the tile geometry exactly and the positions to +-5 mm) | restricted | 0.000 (0.000-0.018) | 0.000 | 0.24 | 1000 |
| (combination: the tile geometry exactly and the positions to +-1 mm) | restricted | 0.530 (0.458-0.601) | 0.000 | 1.22 | 1000 |
| (upper bound: every geometric quantity known exactly (R, arc angles, cell type, positions); Gamma, antenna pattern and cell model still unknown) | restricted | 0.630 (0.559-0.697) | 0.000 | 1.30 | 1000 |

No single fact raises the C2 detection rate above 0; the closest by median Delta RMSE is **the 30 measurement positions and the plane height to +-1 mm** (-1.01 dB). No single fact reaches the 0.5 power threshold.
Combinations that reach 0.5: combination: the tile geometry exactly and the positions to +-1 mm (0.530); upper bound: every geometric quantity known exactly (R, arc angles, cell type, positions); Gamma, antenna pattern and cell model still unknown (0.630).
<!-- DIAGNOSTIC:END -->

**Reading (diagnostic, not part of the protocol).** No single additional fact (tile distance,
tile angles, cell type, antenna pattern, reflection coefficients, or the point positions)
raises the C2 detection rate above zero; the power appears only when the as-built tile
geometry is known exactly *and* the 30 positions to about +-1 mm (0.53), close to the upper
bound with every geometric quantity known (0.63), with no false detections under the M0 truth.
Partial knowledge (tiles exact with positions to +-5 mm; R to +-2.5 mm with positions to +-5
mm) still gives 0. In practice this means a surveyed geometry supplied by the lab (human gate
H-1), which by section 2 turns the study into a "validation" and needs a protocol amendment.
Intermediate tolerances on the tile geometry (between exact and R to +-2.5 mm) were not
explored.

## 14. Outputs of the unblinded run

`run.py` writes a run directory with `manifest.json` (commit, dirty flag, seeds, input
hashes, platform), `meta.json` (data classes, protocol digest, lock state), `F4.csv` (per
cell: measured DIG, prior-median M2 and M0 predictions SIM-sremu, residuals, sigma_geom),
`F4_delta_rmse.csv` (per draw), `F4.meta.json`, `table_v.csv` / `table_v.json` (Table V),
`results.json` (every analysis of sections 7-11, including S-OR and S-NF) and `report.md`.

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
* The near-field statement of section 4.2 (the 1.8 m ring is at the edge of the element-wise
  model's validity; coupling and the reactive near field are not modelled) accompanies every
  result.

## 16. Reproduction

From the sremu repository root, with the project interpreter:

    python -m validation.ants_fig4b.predict --jobs 8                     # tensor, K = 2000
    python -m validation.ants_fig4b.predict --jobs 8 --variant S-OR      # S-OR tensor
    python -m validation.ants_fig4b.decoy --tensor <pred_tensor.npz> --jobs 8 \
        --reference-tensor <v1 pred_tensor.npz> --write-protocol-result
    python -m validation.ants_fig4b.run --tensor <pred_tensor.npz> \
        --tensor-sor <S-OR pred_tensor.npz> --dry-run-decoy 0         # pipeline check
    python -m validation.ants_fig4b.diagnostic --power-run <decoy run dir> \
        --tensor <pred_tensor.npz> --jobs 8                           # section 13.2, not protocol
