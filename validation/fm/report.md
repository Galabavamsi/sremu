# V-FM / E-D: FM RIS dimensioning (DESIGN.md 10.12 and 11 E-D)

**Label: model-based dimensioning, not validated.** All numbers are model outputs (two-ray flat earth, coherent knife edge, K1 RIS kernel with ground image expansion); no measurement enters. The ITU-R P.526 rows come from sremu's experimental port (`physics/experimental/p526.py`), which is not validated against ITU-R data and sits outside every gate: they are an **EXPERIMENTAL sensitivity row** only. Data classes: SIM-sremu, ANALYTIC, ASSUMED.

## Provenance

* Run directory: `validation/fm/results/20261010T232549Z_vfm` (manifest.json, meta.json, CSV/JSON data).
* sremu commit `9dca79f37955dc38fa8890110bea0f342eea207d`, dirty tree: True (dirty.patch sha256 `81af3aa08ec9060d3ca69033a7a75751954410438f1c83b551f1bbd2e09f4cac`).
* Platform: Windows-11-10.0.26200-SP0, AMD Ryzen 9 8945H w/ Radeon 780M Graphics, 16 logical CPUs, power plan Balanced.
* Started 2026-10-10T23:25:56.718392Z, ended 2026-10-10T23:26:14.290370Z, wall time 24.5 s.
* Host CPU busy fraction (all processes, sampled 2 times): median 0.96, max 0.98. Other agents' jobs shared the laptop; wall times are indicative, not benchmarks.

* Assumptions recorded in the manifest: model-based dimensioning, not validated (DESIGN 10.12); lateral RIS placement with the normal on the horizontal bisector [ENG]; ridge: horizontal knife edge 400 m long, 2 km from the RX (0.5, 5 km sensitivity) [ENG]; ground eps_r = 15, sigma = 1e-3 S/m, H pol (P.526 module defaults) [ASSUMED]; A_req by coherent-aperture scaling of the focused 56.25 m^2 panel [DERIVED]; RIS legs are not diffracted (release 1); the geometry keeps them clear of the ridge; x axis: knife-edge loss of the ridge on the LOS ray, -20 log10 |F(nu)| (exact F); four-ray knife-edge direct path (TX/RX ground images, F(nu) per ray): [ENG] model-form sensitivity row, not sremu's model.

## Setup

52 km link at 100 MHz, h_tx = 200 m, h_r in {1.5, 10} m; RIS of 30 x 30 cells of 0.25 m (A0 = 56.25 m^2) centred 34.9 m above the ground, `cell_model` auto (point), focused with continuous per-cell phase (sum_n |c_n|), d2 (RIS-RX) in {0.5, 1, 2} km. Direct path: two-ray flat earth (ground eps_r = 15, sigma = 1e-3 S/m, H polarization [ASSUMED]) plus one coherent knife edge (a horizontal 400 m ridge 2 km from the RX; 0.5 and 5 km as sensitivity) whose height gives the LOS ray nu in [0, 5]; the ground ray is diffracted by the same edge. x axis of the map: the ridge's knife-edge loss on the LOS ray, -20 log10 |F(nu)|. RIS legs are not diffracted (release 1); the geometry keeps them clear of the ridge. Geometries: `lateral` (primary [ENG]: the RIS beside the village, normal on the bisector, about 3 dB obliquity) and `backreflector` (sensitivity [ENG]: behind the village facing the TX, theta near 0). Required aperture by coherent-aperture scaling [DERIVED]: A_req = A0 |h_dir| / |h_RIS(A0)| (RIS path equal to the direct path), and (sqrt(2) - 1) A_req for a +3 dB coherent total.

## Acceptance (regression)

| Check | Measured | Result |
|---|---|---|
| PH-10: two-ray (PEC, H) path loss 139.10 / 122.63 dB +- 0.01 dB | 139.098 / 122.629 dB | PASS |
| V-FM-1: RIS-with-ground path loss vs the T15 values, d2 = 1-2 km, +- 0.1 dB | max \|d\| 0.034 dB | PASS |
| V-FM-1 extended to d2 = 0.5 and 3 km (+- 0.1 dB; not in the DESIGN row) | max \|d\| 0.054 dB | PASS |

V-FM-1 geometry (T15 of `pm_checks.py`): PEC ground (Gamma_H = -1), d_TX-RIS = 51953.7 m, RIS at 34.9 m, RX at d2 near the panel normal; reference = the recorded T15 values (`tests/oracles/data/physics_model_checks.json`) and the closed form recomputed here, -20 log10(A/(4 pi d1 d2)) - 20 log10 F1 - 20 log10 F2 with F = |1 - exp(-jk 2 h_a h_b / d)|.

| d2 (m) | h_r (m) | engine with ground (dB) | T15 recorded (dB) | d (dB) | d vs recomputed (dB) | engine free space (dB) | T15 free space (dB) | result |
|---|---|---|---|---|---|---|---|---|
| 500 | 1.5 | 147.585 | 147.60 | -0.015 | -0.014 | 135.283 | 135.27 | PASS |
| 500 | 10.0 | 134.464 | 134.41 | +0.054 | +0.057 | 135.279 | 135.27 | PASS |
| 1000 | 1.5 | 159.561 | 159.59 | -0.029 | -0.027 | 141.297 | 141.29 | PASS |
| 1000 | 10.0 | 143.874 | 143.88 | -0.006 | -0.008 | 141.296 | 141.29 | PASS |
| 2000 | 1.5 | 171.586 | 171.62 | -0.034 | -0.030 | 147.315 | 147.31 | PASS |
| 2000 | 10.0 | 155.303 | 155.33 | -0.027 | -0.025 | 147.315 | 147.31 | PASS |
| 3000 | 1.5 | 178.627 | 178.66 | -0.033 | -0.031 | 150.837 | 150.84 | PASS |
| 3000 | 10.0 | 162.235 | 162.26 | -0.025 | -0.028 | 150.837 | 150.84 | PASS |

## Direct-path baselines (no ridge)

| h_r (m) | FSPL (dB) | two-ray, land H (dB) | two-ray, PEC H (dB) | P.526 smooth earth, EXPERIMENTAL (dB) | P.526 excess (dB) |
|---|---|---|---|---|---|
| 1.5 | 106.768 | 139.071 | 139.098 | 143.903 | 37.135 |
| 10 | 106.768 | 122.637 | 122.629 | 127.048 | 20.280 |

## RIS path at A0 = 56.25 m^2 (focused)

| geometry | d2 (m) | h_r (m) | ground on RIS hops | RIS path loss (dB) | obliquity at centre (dB) |
|---|---|---|---|---|---|
| lateral | 500 | 1.5 | true | 150.709 | 2.979 |
| lateral | 500 | 1.5 | false | 138.260 | 2.979 |
| lateral | 500 | 10.0 | true | 137.664 | 2.974 |
| lateral | 500 | 10.0 | false | 138.256 | 2.974 |
| lateral | 1000 | 1.5 | true | 162.563 | 2.930 |
| lateral | 1000 | 1.5 | false | 144.234 | 2.930 |
| lateral | 1000 | 10.0 | true | 146.932 | 2.929 |
| lateral | 1000 | 10.0 | false | 144.232 | 2.929 |
| lateral | 2000 | 1.5 | true | 174.474 | 2.847 |
| lateral | 2000 | 1.5 | false | 150.176 | 2.847 |
| lateral | 2000 | 10.0 | true | 158.238 | 2.847 |
| lateral | 2000 | 10.0 | false | 150.176 | 2.847 |
| backreflector | 500 | 1.5 | true | 147.903 | 0.010 |
| backreflector | 500 | 1.5 | false | 135.374 | 0.010 |
| backreflector | 500 | 10.0 | true | 134.858 | 0.005 |
| backreflector | 500 | 10.0 | false | 135.370 | 0.005 |
| backreflector | 1000 | 1.5 | true | 159.958 | 0.002 |
| backreflector | 1000 | 1.5 | false | 141.470 | 0.002 |
| backreflector | 1000 | 10.0 | true | 144.328 | 0.001 |
| backreflector | 1000 | 10.0 | false | 141.469 | 0.001 |
| backreflector | 2000 | 1.5 | true | 172.262 | 6.26e-04 |
| backreflector | 2000 | 1.5 | false | 147.651 | 6.26e-04 |
| backreflector | 2000 | 10.0 | true | 156.025 | 3.57e-04 |
| backreflector | 2000 | 10.0 | false | 147.651 | 3.57e-04 |

## E-D: required aperture vs obstruction (figure F12 data: `vfm_ed_map.csv`)

Without obstruction (nu = 0 on the 2 km ridge, i.e. a grazing edge with 6.02 dB knife-edge loss on the LOS ray) and with ground on both RIS hops, the focused 56.25 m^2 RIS path is weaker than the direct path by (min to max over d2 = 1-2 km, both h_r):

| geometry | direct-path model | direct minus RIS |
|---|---|---|
| lateral | two-ray + knife edge (primary) | 28.5 to 41.3 dB |
| lateral | four-ray knife edge ([ENG] sensitivity) | 23.6 to 35.9 dB |
| lateral | P.526 + knife edge (EXPERIMENTAL) | 12.6 to 25.2 dB |
| backreflector | two-ray + knife edge (primary) | 25.9 to 39.1 dB |
| backreflector | four-ray knife edge ([ENG] sensitivity) | 21.0 to 33.7 dB |
| backreflector | P.526 + knife edge (EXPERIMENTAL) | 10.0 to 23.0 dB |

Against the unobstructed P.526 smooth-earth direct path (no ridge), the T15 RIS-with-ground values are 15.7 to 28.3 dB weaker at d2 = 1-2 km (EXPERIMENTAL baseline).

Required aperture A_req (m^2) for RIS path = direct path, lateral geometry, ridge 2 km from the RX; columns are the ridge's knife-edge loss on the LOS ray (nu):

| direct-path model | ground on RIS hops | d2 (m) | h_r (m) | nu=0 (6.0 dB) | nu=1 (13.9 dB) | nu=2 (19.1 dB) | nu=3 (22.5 dB) | nu=5 (26.9 dB) |
|---|---|---|---|---|---|---|---|---|
| two-ray + KE | true | 500 | 1.5 | 422.8 | 377.8 | 357.6 | 347.8 | 334.8 |
| two-ray + KE | true | 500 | 10.0 | 512.8 | 429.5 | 357.5 | 281.3 | 117.1 |
| two-ray + KE | true | 1000 | 1.5 | 1655.0 | 1478.9 | 1399.8 | 1361.4 | 1310.5 |
| two-ray + KE | true | 1000 | 10.0 | 1490.5 | 1248.6 | 1039.2 | 817.6 | 340.3 |
| two-ray + KE | true | 2000 | 1.5 | 6521.9 | 5828.0 | 5516.1 | 5364.6 | 5164.1 |
| two-ray + KE | true | 2000 | 10.0 | 5477.8 | 4588.9 | 3819.3 | 3004.7 | 1250.5 |
| two-ray + KE | false | 500 | 1.5 | 100.9 | 90.1 | 85.3 | 83.0 | 79.9 |
| two-ray + KE | false | 500 | 10.0 | 548.9 | 459.8 | 382.7 | 301.1 | 125.3 |
| two-ray + KE | false | 1000 | 1.5 | 200.6 | 179.3 | 169.7 | 165.0 | 158.8 |
| two-ray + KE | false | 1000 | 10.0 | 1092.3 | 915.0 | 761.6 | 599.1 | 249.4 |
| two-ray + KE | false | 2000 | 1.5 | 397.6 | 355.3 | 336.3 | 327.1 | 314.8 |
| two-ray + KE | false | 2000 | 10.0 | 2165.2 | 1813.9 | 1509.7 | 1187.7 | 494.3 |
| four-ray KE ([ENG] sensitivity) | true | 500 | 1.5 | 218.6 | 362.9 | 554.7 | 647.8 | 458.5 |
| four-ray KE ([ENG] sensitivity) | true | 500 | 10.0 | 327.8 | 529.1 | 631.8 | 497.7 | 35.2 |
| four-ray KE ([ENG] sensitivity) | true | 1000 | 1.5 | 855.8 | 1420.6 | 2171.6 | 2535.8 | 1794.9 |
| four-ray KE ([ENG] sensitivity) | true | 1000 | 10.0 | 952.8 | 1538.0 | 1836.5 | 1446.6 | 102.3 |
| four-ray KE ([ENG] sensitivity) | true | 2000 | 1.5 | 3372.4 | 5598.0 | 8557.3 | 9992.7 | 7073.1 |
| four-ray KE ([ENG] sensitivity) | true | 2000 | 10.0 | 3501.6 | 5652.6 | 6749.6 | 5316.7 | 375.9 |
| four-ray KE ([ENG] sensitivity) | false | 500 | 1.5 | 52.1 | 86.6 | 132.3 | 154.5 | 109.4 |
| four-ray KE ([ENG] sensitivity) | false | 500 | 10.0 | 350.9 | 566.4 | 676.4 | 532.8 | 37.7 |
| four-ray KE ([ENG] sensitivity) | false | 1000 | 1.5 | 103.7 | 172.2 | 263.2 | 307.4 | 217.6 |
| four-ray KE ([ENG] sensitivity) | false | 1000 | 10.0 | 698.2 | 1127.1 | 1345.9 | 1060.2 | 75.0 |
| four-ray KE ([ENG] sensitivity) | false | 2000 | 1.5 | 205.6 | 341.3 | 521.7 | 609.2 | 431.2 |
| four-ray KE ([ENG] sensitivity) | false | 2000 | 10.0 | 1384.1 | 2234.3 | 2667.9 | 2101.5 | 148.6 |
| P.526 + KE (EXPERIMENTAL) | true | 500 | 1.5 | 61.6 | 25.0 | 13.7 | 9.2 | 5.5 |
| P.526 + KE (EXPERIMENTAL) | true | 500 | 10.0 | 95.5 | 38.7 | 21.2 | 14.3 | 8.6 |
| P.526 + KE (EXPERIMENTAL) | true | 1000 | 1.5 | 241.0 | 97.7 | 53.5 | 36.1 | 21.7 |
| P.526 + KE (EXPERIMENTAL) | true | 1000 | 10.0 | 277.5 | 112.5 | 61.6 | 41.5 | 25.0 |
| P.526 + KE (EXPERIMENTAL) | true | 2000 | 1.5 | 949.8 | 385.0 | 210.9 | 142.1 | 85.5 |
| P.526 + KE (EXPERIMENTAL) | true | 2000 | 10.0 | 1020.0 | 413.5 | 226.5 | 152.6 | 91.8 |
| P.526 + KE (EXPERIMENTAL) | false | 500 | 1.5 | 14.7 | 6.0 | 3.3 | 2.2 | 1.3 |
| P.526 + KE (EXPERIMENTAL) | false | 500 | 10.0 | 102.2 | 41.4 | 22.7 | 15.3 | 9.2 |
| P.526 + KE (EXPERIMENTAL) | false | 1000 | 1.5 | 29.2 | 11.8 | 6.5 | 4.4 | 2.6 |
| P.526 + KE (EXPERIMENTAL) | false | 1000 | 10.0 | 203.4 | 82.4 | 45.2 | 30.4 | 18.3 |
| P.526 + KE (EXPERIMENTAL) | false | 2000 | 1.5 | 57.9 | 23.5 | 12.9 | 8.7 | 5.2 |
| P.526 + KE (EXPERIMENTAL) | false | 2000 | 10.0 | 403.2 | 163.4 | 89.5 | 60.3 | 36.3 |
| free space + KE (reference) | true | 500 | 1.5 | 4427.4 | 1794.6 | 983.2 | 662.4 | 398.5 |
| free space + KE (reference) | true | 500 | 10.0 | 986.0 | 399.7 | 219.0 | 147.5 | 88.7 |
| free space + KE (reference) | true | 1000 | 1.5 | 17331.5 | 7025.3 | 3848.8 | 2592.9 | 1559.8 |
| free space + KE (reference) | true | 1000 | 10.0 | 2866.2 | 1161.8 | 636.5 | 428.8 | 257.9 |
| free space + KE (reference) | true | 2000 | 1.5 | 68296.9 | 27683.9 | 15166.5 | 10217.4 | 6146.4 |
| free space + KE (reference) | true | 2000 | 10.0 | 10533.9 | 4269.9 | 2339.2 | 1575.9 | 948.0 |
| free space + KE (reference) | false | 500 | 1.5 | 1056.1 | 428.1 | 234.5 | 158.0 | 95.0 |
| free space + KE (reference) | false | 500 | 10.0 | 1055.6 | 427.9 | 234.4 | 157.9 | 95.0 |
| free space + KE (reference) | false | 1000 | 1.5 | 2100.8 | 851.5 | 466.5 | 314.3 | 189.1 |
| free space + KE (reference) | false | 1000 | 10.0 | 2100.5 | 851.4 | 466.4 | 314.2 | 189.0 |
| free space + KE (reference) | false | 2000 | 1.5 | 4163.9 | 1687.8 | 924.7 | 622.9 | 374.7 |
| free space + KE (reference) | false | 2000 | 10.0 | 4163.8 | 1687.8 | 924.6 | 622.9 | 374.7 |

A0 = 56.25 m^2 suffices (A_req <= A0) in these map rows per direct-path model (all geometries, ground options, d2, h_r and nu in 0 ... 5): two-ray + KE (primary): 0 of 1224; free space + KE (reference): 0 of 1224; four-ray KE ([ENG] sensitivity): 23 of 1224; P.526 + KE (EXPERIMENTAL): 844 of 1224. A count pooled over the models is not meaningful.

The ridge also shields the ground ray that nearly cancels the LOS at grazing incidence, so the two-ray + knife-edge direct path can be *stronger* than the unobstructed two-ray (negative net change, dB):

| h_r (m) | ridge from RX (m) | nu = 0 | nu = 1 | nu = 2 | nu = 5 |
|---|---|---|---|---|---|
| 1.5 | 500 | -11.04 | -10.50 | -10.13 | -9.01 |
| 1.5 | 2000 | -5.88 | -4.90 | -4.43 | -3.85 |
| 1.5 | 5000 | -2.77 | -1.27 | -0.56 | +0.07 |
| 10 | 500 | -0.15 | +1.98 | +3.20 | +5.77 |
| 10 | 2000 | -4.17 | -2.63 | -1.04 | +8.66 |
| 10 | 5000 | -2.53 | -0.89 | +0.38 | +5.57 |

Model-form sensitivity of the direct path: the [ENG] four-ray knife edge minus sremu's two-ray + knife edge, ridge 2 km from the RX, nu in 0 ... 5: h_r = 1.5 m: -5.5 to +5.7 dB; h_r = 10 m: -5.3 to +10.4 dB. A_req scales with the direct-path amplitude, so this spread carries straight into the map (factor 10^(spread/20)); the map is a dimensioning aid under a stated model, not a prediction.

Per-nu rows of the direct path (engine nu, ridge height, knife-edge loss, two-ray + KE, four-ray KE, free space + KE, P.526 + KE) are in `vfm_direct.csv` (306 rows).

## Files and run time

`vfm_direct.csv`, `vfm_ris.csv`, `vfm_ed_map.csv` (F12), `vfm1_regression.csv`, `meta.json`, `manifest.json`. Wall time of all evaluations 17.0 s.
