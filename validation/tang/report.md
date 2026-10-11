# V-TANG: Tang et al. 2022 model curves and measured points (DESIGN.md 10.8)

V-TANG-1 compares sremu's area-tied K1 kernel, run through the scene model with the shipped default `cell_model` (auto), against the **published model curves** (Eq. 9) of Tang et al., Fig. 10: a model-to-model check. V-TANG-2 compares Tang's digitized chamber measurements with that model; it characterises Tang's published data and model and is **not** presented as sremu accuracy (DESIGN 10.8, 1.6). The Fig. 10 series are vector-extracted from the arXiv PDF (DIG; no measurement uncertainty is stated in the source). Data classes: DIG, REPORTED, SIM-sremu.

## Provenance

* Run directory: `validation/tang/results/20261010T232551Z_vtang` (manifest.json, meta.json, CSV/JSON data).
* sremu commit `9dca79f37955dc38fa8890110bea0f342eea207d`, dirty tree: True (dirty.patch sha256 `81af3aa08ec9060d3ca69033a7a75751954410438f1c83b551f1bbd2e09f4cac`).
* Platform: Windows-11-10.0.26200-SP0, AMD Ryzen 9 8945H w/ Radeon 780M Graphics, 16 logical CPUs, power plan Balanced.
* Started 2026-10-10T23:25:59.541470Z, ended 2026-10-10T23:26:04.561296Z, wall time 13.3 s.
* Host CPU busy fraction (all processes, sampled 2 times): median 0.95, max 0.99. Other agents' jobs shared the laptop; wall times are indicative, not benchmarks.

* Inputs: `data/validation/tang2022/tang2022_fig10_vector.json` (DIG, sha256 `cf8793e9f9ed5a79...`); `data/validation/tang2022/tang2022_parameters.json` (REPORTED, sha256 `aab055816fa41763...`) (checked against `data/validation/tang2022/SHA256SUMS`; provenance in `data/validation/tang2022/PROVENANCE.md`).

## Setup

RIS1 of Tang 2022 Tables III/IV: 56 x 20 cells of 1.4 x 2.8 mm at 27 GHz, all cells in coding state '0' (Gamma = 0.9 at 165 deg, built-in table `tang2022_ris1_27ghz`); cos^53.8 horn patterns aimed at the panel centre; specular geometry (Tx at d1, Rx at d2, both at theta from the normal in the u-axis plane); four panels (d1, theta) = (1 m, 10 deg), (2 m, 10 deg), (1 m, 45 deg), (2 m, 45 deg), nine d2 = 1 ... 5 m each. Received power `P_R = P_T + G_t G_r G_line - 20 log10(G_0) + 20 log10 |h_RIS|` with P_T = 20 dBm and the measured G_t G_r G_line = 24.0 dB replacing the nominal horn peak gains G_0 = 2(alpha + 1) carried by the engine's CosPower antennas. `cell_model` auto resolved to ['point'] (pitch < lambda/2). Digitized x positions (1.002 ... 5.002 m) are snapped to the nominal 0.5 m grid.

## Acceptance

| Check | Measured | Result |
|---|---|---|
| V-TANG-1: max \|engine - published Eq. 9\| <= 0.05 dB (36 points) | 0.0121 dB | PASS |
| V-TANG-2 regression: \|RMSE(measured - engine) - 1.05\| <= 0.05 dB | RMSE 1.0465 dB | PASS |

## Numbers (n = 36)

V-TANG-1: max |engine - published| = 0.0121 dB, mean 0.0080 dB, RMSE 0.0084 dB.

| V-TANG-2 errors | n | bias (dB) | SD (dB) | RMSE (dB) | MAE (dB) | max \|e\| (dB) |
|---|---|---|---|---|---|---|
| measured - engine | 36 | +0.9770 | 0.3802 | 1.0465 | 0.9770 | 1.5905 |
| measured - published Eq. 9 curve | 36 | +0.9850 | 0.3809 | 1.0542 | 0.9850 | 1.6000 |

| panel | d1 (m), theta (deg) | max \|engine - published\| (dB) | measured - engine: bias (dB) | SD (dB) | RMSE (dB) |
|---|---|---|---|---|---|
| a | 1, 10 | 0.0120 | +0.7907 | 0.3701 | 0.8643 |
| b | 2, 10 | 0.0120 | +1.1504 | 0.3895 | 1.2076 |
| c | 1, 45 | 0.0121 | +1.0568 | 0.2929 | 1.0922 |
| d | 2, 45 | 0.0113 | +0.9103 | 0.4158 | 0.9911 |

## Paper wording (DESIGN 10.8 template, filled from this run)

> The kernel reproduces Tang et al.'s published model curves within 0.02 dB. Their digitized chamber measurements lie +0.99 dB (SD 0.38 dB, n = 36) above that published model.

Filled as: curve = max |engine - published| rounded up to 0.01 dB (a bound); bias and SD = measured - published Eq. 9 curve (the sentence's "above that published model"). With measured - engine instead, the bias is +0.977 dB (SD 0.380 dB). The DESIGN placeholders ({0.02}, {+0.98}, {0.38}) were scratch values; the paper must take these numbers from the pipeline macros, not from this report.

## Files and run time

`vtang_points.csv` (per point: panel, geometry, published, engine and measured dBm and their differences; figure F3(b) data), `vtang_summary.json`, `meta.json`, `manifest.json`. Engine evaluation of the 36 points took 5.00 s.
