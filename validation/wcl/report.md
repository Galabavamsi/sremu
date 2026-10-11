# V-WCL: reproduction checks of the WCL 2025 letter (DESIGN.md 10.11)

Mishra & Kherani, IEEE WCL 14(4), 2025: published analytic results. An earlier in-house simulator by the authors was used in that letter; sremu is an independent new implementation (DESIGN D24: no comparison with that simulator is made here or in the paper). These are **verification** checks of sremu against independent closed forms and its own batch path under the area-tied K1 kernel, not a validation against measurements. Nothing here states anything about the letter's Eq. (2) (DESIGN 1.6 rule 6, H-3); numbers are phrased "under the area-tied kernel the value is X". Data classes: SIM-sremu, ANALYTIC, REPORTED, ASSUMED.

## Provenance

* Run directory: `validation/wcl/results/20261010T232549Z_vwcl` (manifest.json, meta.json, CSV/JSON data).
* sremu commit `9dca79f37955dc38fa8890110bea0f342eea207d`, dirty tree: True (dirty.patch sha256 `81af3aa08ec9060d3ca69033a7a75751954410438f1c83b551f1bbd2e09f4cac`).
* Platform: Windows-11-10.0.26200-SP0, AMD Ryzen 9 8945H w/ Radeon 780M Graphics, 16 logical CPUs, power plan Balanced.
* Started 2026-10-10T23:25:56.646528Z, ended 2026-10-11T00:26:35.057885Z, wall time 3645.7 s.
* Host CPU busy fraction (all processes, sampled 401 times): median 0.97, max 1.00. Other agents' jobs shared the laptop; wall times are indicative, not benchmarks.

* Profile `full`, parts ['w1', 'w3', 'w5']; sweep of 400 points with the sremu sweep runner, 12 workers, 0 failed, 0 resumed.

## Scenario as run

* Corridor of l = 10 m, MU on 1-D random waypoint at v = 1 m/s, no pause; RIS line at distance h with N = 133 cells of w = lambda/4 = 0.07495 m at f_c = 1 GHz; on_off alphabet (OFF absorbing, ON Gamma = 1); P_T = 0 dBm; isotropic antennas; IQ path: 10 kS/s unit tone, noise off, lockstep, numba_cpu.
* **Documented deviation:** AP offset d0 = 0.5 m perpendicular to the user line (+z); with the AP on the line the mean LOS power diverges as x -> 0. Sensitivity: d0 = 0.3 and 1.0 m, and an on-line AP with the MU segment starting at c = 0.05, 0.1, 0.2 m.
* Assumptions recorded in the manifest: V-WCL documented deviation: AP offset d0 = 0.5 m perpendicular to the user line (+z), not on the line as in the letter (mean LOS power diverges as x -> 0) [ENG]; URAP redraw period 0.1 s [ENG; the letter does not state it; the time average does not depend on it]; Coefficient path: left-point pieces on a 5 ms geometry grid (E-A batch setting); IQ path: 10 kS/s unit tone (P_T = 0 dBm), noise off, lockstep, numba_cpu.

## Acceptance summary

| Check | Measured | Result |
|---|---|---|
| W1 positions vs F(x) (KS D <= 0.02) | 0.0030 | PASS |
| W1 waypoints uniform (KS D <= 0.0163) | 0.0092 | PASS |
| W1 mean leg length l/3 +- 2.5 % | 3.3598 m (+0.79 %) | PASS |
| W1 no jump > v dt + 1e-12 | 0.100000 m | PASS |
| W1 teeth (internal, not for the paper per D24): legacy_wcl_eq2 / K1 ratio vs closed form (<= 1e-9) | 1.55e-15 | PASS |
| W3 h = 1 m: emulator vs analytic mean power, \|d\| <= max(0.05 dB, CI) = 0.157 dB | -0.0675 dB | PASS |
| W3 h = 2 m: emulator vs analytic mean power, \|d\| <= max(0.05 dB, CI) = 0.157 dB | -0.0672 dB | PASS |
| W3 h = 4 m: emulator vs analytic mean power, \|d\| <= max(0.05 dB, CI) = 0.157 dB | -0.0674 dB | PASS |
| W3 h = 6 m: emulator vs analytic mean power, \|d\| <= max(0.05 dB, CI) = 0.157 dB | -0.0674 dB | PASS |
| W3 same-trajectory h = 1 m (not a DESIGN gate): emulator vs E[P \| x(t)] mean power, within the paired 95 % CI (+-0.00072 dB) | +0.00008 dB | consistent |
| W3 same-trajectory h = 2 m (not a DESIGN gate): emulator vs E[P \| x(t)] mean power, within the paired 95 % CI (+-0.00046 dB) | -0.00002 dB | consistent |
| W3 same-trajectory h = 4 m (not a DESIGN gate): emulator vs E[P \| x(t)] mean power, within the paired 95 % CI (+-0.00016 dB) | -0.00001 dB | consistent |
| W3 same-trajectory h = 6 m (not a DESIGN gate): emulator vs E[P \| x(t)] mean power, within the paired 95 % CI (+-0.00007 dB) | +8.7472e-06 dB | consistent |
| W3 IQ subset h = 1 m: IQ vs paired batch path (<= 0.02 dB) | +0.00007 dB | PASS |
| W3 IQ subset h = 2 m: IQ vs paired batch path (<= 0.02 dB) | +3.6224e-06 dB | PASS |
| W3 IQ subset h = 4 m: IQ vs paired batch path (<= 0.02 dB) | -0.00003 dB | PASS |
| W3 IQ subset h = 6 m: IQ vs paired batch path (<= 0.02 dB) | +3.6186e-06 dB | PASS |
| W5 ExactOptimum tau_SUI = 1/100 s: IQ vs oracle (<= 0.02 dB) | +0.00002 dB | PASS |
| W5 ExactOptimum tau_SUI = 1/10 s: IQ vs oracle (<= 0.02 dB) | +0.00021 dB | PASS |
| W5 ExactOptimum tau_SUI = 3/10 s: IQ vs oracle (<= 0.02 dB) | +0.00006 dB | PASS |
| W5 ExactOptimum tau_SUI = 1 s: IQ vs oracle (<= 0.02 dB) | +0.00001 dB | PASS |
| W5 ExactOptimum tau_SUI = 3 s: IQ vs oracle (<= 0.02 dB) | -2.3990e-06 dB | PASS |
| W5 ExactOptimum tau_SUI = 10 s: IQ vs oracle (<= 0.02 dB) | -7.2783e-06 dB | PASS |
| W5 WclHeuristic tau_SUI = 1/100 s: IQ vs oracle (<= 0.02 dB) | +0.00001 dB | PASS |
| W5 WclHeuristic tau_SUI = 1/10 s: IQ vs oracle (<= 0.02 dB) | +0.00015 dB | PASS |
| W5 WclHeuristic tau_SUI = 3/10 s: IQ vs oracle (<= 0.02 dB) | +0.00004 dB | PASS |
| W5 WclHeuristic tau_SUI = 1 s: IQ vs oracle (<= 0.02 dB) | +4.9703e-06 dB | PASS |
| W5 WclHeuristic tau_SUI = 3 s: IQ vs oracle (<= 0.02 dB) | -4.0039e-06 dB | PASS |
| W5 WclHeuristic tau_SUI = 10 s: IQ vs oracle (<= 0.02 dB) | -7.6316e-06 dB | PASS |
| W5 ExactOptimum non-increasing 1/100 -> 1/10 s (within CI) | -0.1856 dB | PASS |
| W5 ExactOptimum non-increasing 1/10 -> 3/10 s (within CI) | -0.1516 dB | PASS |
| W5 ExactOptimum non-increasing 3/10 -> 1 s (within CI) | -0.0254 dB | PASS |
| W5 ExactOptimum non-increasing 1 -> 3 s (within CI) | -0.0016 dB | PASS |
| W5 ExactOptimum non-increasing 3 -> 10 s (within CI) | -0.0009 dB | PASS |
| W5 WclHeuristic non-increasing 1/100 -> 1/10 s (within CI) | -0.1369 dB | PASS |
| W5 WclHeuristic non-increasing 1/10 -> 3/10 s (within CI) | -0.1163 dB | PASS |
| W5 WclHeuristic non-increasing 3/10 -> 1 s (within CI) | -0.0178 dB | PASS |
| W5 WclHeuristic non-increasing 1 -> 3 s (within CI) | -0.0003 dB | PASS |
| W5 WclHeuristic non-increasing 3 -> 10 s (within CI) | -0.0009 dB | PASS |
| W6 ExactOptimum tau=1/100 h=4, reference auto_T_u: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00141 dB | PASS |
| W6 ExactOptimum tau=1/100 h=4, reference 5ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00169 dB | PASS |
| W6 ExactOptimum tau=1/100 h=4, reference 1ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00037 dB | PASS |
| W6 ExactOptimum tau=1/10 h=4, reference auto_T_u: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.01023 dB | FAIL |
| W6 ExactOptimum tau=1/10 h=4, reference 5ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.01054 dB | FAIL |
| W6 ExactOptimum tau=1/10 h=4, reference 1ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00195 dB | PASS |
| W6 ExactOptimum tau=1 h=4, reference auto_T_u: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00092 dB | PASS |
| W6 ExactOptimum tau=1 h=4, reference 5ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00094 dB | PASS |
| W6 ExactOptimum tau=1 h=4, reference 1ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00018 dB | PASS |
| W6 WclHeuristic tau=1/10 h=4, reference auto_T_u: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00753 dB | PASS |
| W6 WclHeuristic tau=1/10 h=4, reference 5ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00776 dB | PASS |
| W6 WclHeuristic tau=1/10 h=4, reference 1ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -0.00144 dB | PASS |
| W6 Urap tau=1/10 h=2, reference auto_T_u: IQ (Doppler) vs snapshot average (<= 0.01 dB) | +4.9443e-07 dB | PASS |
| W6 Urap tau=1/10 h=2, reference 5ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | +3.6224e-06 dB | PASS |
| W6 Urap tau=1/10 h=2, reference 1ms: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -1.4389e-06 dB | PASS |
| W6 ExactOptimum tau=1/100 h=4, reference 1ms_midpoint_oracle: IQ (Doppler) vs snapshot average (<= 0.01 dB) | +0.00002 dB | PASS |
| W6 ExactOptimum tau=1/10 h=4, reference 1ms_midpoint_oracle: IQ (Doppler) vs snapshot average (<= 0.01 dB) | +0.00021 dB | PASS |
| W6 ExactOptimum tau=1 h=4, reference 1ms_midpoint_oracle: IQ (Doppler) vs snapshot average (<= 0.01 dB) | +0.00001 dB | PASS |
| W6 WclHeuristic tau=1/10 h=4, reference 1ms_midpoint_oracle: IQ (Doppler) vs snapshot average (<= 0.01 dB) | +0.00015 dB | PASS |
| W6 Urap tau=1/10 h=2, reference 5ms_midpoint: IQ (Doppler) vs snapshot average (<= 0.01 dB) | -1.9346e-06 dB | PASS |

W6 summary: with references whose own quadrature error is small (left-point 1 ms, the independent oracle on 1 ms midpoints, the 5 ms midpoint average) every case is within 0.01 dB (PASS). With the batch path's own left-point grids (automatic T_u and 5 ms) 2 row(s) exceed 0.01 dB (largest 0.0105 dB); see the W6 section for why this is the reference's quadrature error and not the Doppler effect.

Machine-readable acceptance (meta.json): `{"w3_pass":true,"w3_iq_pass":true,"w5_pass":true,"w6_pass_by_grid":{"1ms":true,"1ms_midpoint_oracle":true,"5ms":false,"5ms_midpoint":true,"auto_T_u":false},"w6_doppler_isolated_pass":true,"W1":{"ks_positions_pass":true,"ks_starts_pass":true,"mean_leg_pass":true,"max_jump_pass":true,"teeth_pass":true}}`

## W1: mobility statistics and the legacy kernel teeth

333333 positions at 10 Hz over T = 100000/3 s (10000 mean movement periods; 9921 legs). The `legacy_wcl_eq2` per-element power ratio to sremu's K1 cascade equals (pi lambda)^2 (cos_i cos_s)^(2q-1) with q = 0.285 to 1.55e-15 (relative, 2660 element ratios); at broadside and 1 GHz the factor is -0.521 dB per element. This is a statement about two kernels' ratio, not about which is right; it is an internal teeth test and is not featured in the paper (DESIGN D24).

## W3: URAP (p = 0.2) time-averaged power vs the analytic integrals

Emulator side: the batch (coefficient) path `h = h_direct + sum_n cascade[s_n, n]` from `model.csi_view(model.snapshot(t))` on a 5 ms grid (left-point pieces, the batch path's semantics) over 10^4 mean movement periods per h, split into 20 consecutive batches (batch means, Student t 95 %); the URAP decisions come from the `Urap` controller itself (keyed generator, 0.1 s redraw). Analytic side: integrals over the stationary pdf of the letter's Eq. (9) (power of the mean field) and of the mean power (adds p(1-p) sum |S_n|^2), with an independent closed-form channel (Gauss-Legendre, 4000 nodes; 8000-node change reported).

| h (m) | emulator (dBm) | CI half (dB) | analytic mean power (dBm) | analytic Eq. 9 (dBm) | d vs mean power (dB) | d vs Eq. 9 (dB) | tol (dB) | result |
|---|---|---|---|---|---|---|---|---|
| 1 | -41.5600 | 0.1566 | -41.4925 | -41.4946 | -0.0675 | -0.0654 | 0.157 | PASS |
| 2 | -41.5667 | 0.1566 | -41.4995 | -41.5003 | -0.0672 | -0.0664 | 0.157 | PASS |
| 4 | -41.5674 | 0.1565 | -41.5000 | -41.5002 | -0.0674 | -0.0672 | 0.157 | PASS |
| 6 | -41.5673 | 0.1565 | -41.5000 | -41.5000 | -0.0674 | -0.0673 | 0.157 | PASS |

Why the interval is wide and the offset is the same at every h: with d0 = 0.5 m the LOS power, proportional to 1/(x^2 + d0^2), peaks near the AP, where the random waypoint rarely goes (f_X -> 0 as x -> 0), so the time average of the LOS power itself carries this mobility sampling error over 10^4 periods (emulator LOS -41.5676 dBm vs the stationary -41.5002 dBm, the same trajectory at every h), while the URAP increment over LOS is below 0.01 dB. The stationary comparison therefore passes through the CI rule and cannot separate the two forms. The same-trajectory comparison below removes the mobility noise.

**Same trajectory (not a DESIGN gate; added to make W3 discriminating).** On the same 5 ms grid and trajectory, the independent closed form gives the conditional expectations E[P | x(t)] of URAP in both forms; the emulator's batch means are compared per batch (paired Student-t 95 % interval over the 20 batches, only the URAP draws vary). The independent replica of the realised URAP power (same `Urap` decisions, closed-form channel) and of the LOS agrees with the emulator to the relative differences shown.

| h (m) | emulator (dBm) | E[P\|x] mean power (dBm) | E[P\|x] Eq. 9 (dBm) | d vs mean power (dB) | d vs Eq. 9 (dB) | paired CI half (dB) | forms differ by (dB) | URAP - LOS: emulator / mean power / Eq. 9 (dB) | replica max rel. diff |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -41.55997 | -41.56005 | -41.56216 | +0.00008 (in CI) | +0.00219 (outside CI) | 0.00072 | 0.00212 | +0.00763 / +0.00755 / +0.00544 | 4.0e-13 |
| 2 | -41.56672 | -41.56670 | -41.56751 | -0.00002 (in CI) | +0.00078 (outside CI) | 0.00046 | 0.00081 | +0.00087 / +0.00090 / +0.00009 | 4.0e-13 |
| 4 | -41.56735 | -41.56734 | -41.56755 | -0.00001 (in CI) | +0.00019 (outside CI) | 0.00016 | 0.00021 | +0.00025 / +0.00026 / +0.00005 | 4.0e-13 |
| 6 | -41.56733 | -41.56734 | -41.56742 | +8.7472e-06 (in CI) | +0.00008 (outside CI) | 0.00007 | 0.00007 | +0.00027 / +0.00026 / +0.00018 | 4.0e-13 |

Gains over LOS under the area-tied kernel (analytic stationary averages; emulator batch means in brackets), d0 = 0.5 m:

| h (m) | LOS (dBm) | URAP mean power - LOS (dB) | Eq. 9 - LOS (dB) | exact optimum - LOS (dB) | tau_SUI -> inf limit (dBm) | quadrature 8000 vs 4000 (dB) |
|---|---|---|---|---|---|---|
| 1 | -41.5002 [-41.5676] | +0.0077 [+0.0076] | +0.0056 | +1.0017 [+1.0064] | -41.4571 | +2.26663e-12 |
| 2 | -41.5002 [-41.5676] | +0.0007 [+0.0009] | -6.599e-05 | +0.7013 [+0.7069] | -41.4950 | +2.25242e-12 |
| 4 | -41.5002 [-41.5676] | +0.0003 [+0.0002] | +6.056e-05 | +0.3864 [+0.3899] | -41.4989 | +2.25953e-12 |
| 6 | -41.5002 [-41.5676] | +0.0003 [+0.0003] | +0.0002 | +0.2404 [+0.2427] | -41.4993 | +2.25953e-12 |

### W3 sensitivity to the AP offset (10^3 movement periods, 10 batches)

| case | h (m) | emulator (dBm) | CI half (dB) | d vs mean power (dB) | d vs Eq. 9 (dB) | within tol | URAP - LOS (dB) | opt - LOS (dB) |
|---|---|---|---|---|---|---|---|---|
| d0_0.3 | 1 | -40.5274 | 0.5146 | +0.0645 | +0.0664 | PASS | +0.0035 | +0.8987 |
| d0_0.3 | 2 | -40.5300 | 0.5173 | +0.0635 | +0.0642 | PASS | +0.0019 | +0.6103 |
| d0_0.3 | 4 | -40.5313 | 0.5162 | +0.0639 | +0.0640 | PASS | +0.0002 | +0.3284 |
| d0_0.3 | 6 | -40.5315 | 0.5161 | +0.0639 | +0.0639 | PASS | +3.737e-05 | +0.2028 |
| d0_1.0 | 1 | -43.0588 | 0.2403 | -0.0273 | -0.0252 | PASS | +0.0072 | +1.0721 |
| d0_1.0 | 2 | -43.0637 | 0.2409 | -0.0266 | -0.0256 | PASS | +0.0015 | +0.8406 |
| d0_1.0 | 4 | -43.0640 | 0.2413 | -0.0256 | -0.0254 | PASS | +0.0003 | +0.4899 |
| d0_1.0 | 6 | -43.0643 | 0.2413 | -0.0257 | -0.0256 | PASS | -2.693e-05 | +0.3088 |
| online_c0.05 | 1 | -39.1806 | 0.8990 | +0.1848 | +0.1863 | PASS | +0.0048 | +0.7312 |
| online_c0.05 | 2 | -39.1846 | 0.8991 | +0.1848 | +0.1853 | PASS | +0.0009 | +0.4783 |
| online_c0.05 | 4 | -39.1851 | 0.8991 | +0.1849 | +0.1850 | PASS | +0.0003 | +0.2549 |
| online_c0.05 | 6 | -39.1855 | 0.8991 | +0.1847 | +0.1848 | PASS | +5.165e-05 | +0.1566 |
| online_c0.1 | 1 | -40.1389 | 0.6130 | +0.1242 | +0.1260 | PASS | +0.0051 | +0.8495 |
| online_c0.1 | 2 | -40.1455 | 0.6131 | +0.1217 | +0.1223 | PASS | +0.0010 | +0.5653 |
| online_c0.1 | 4 | -40.1463 | 0.6132 | +0.1217 | +0.1218 | PASS | +0.0003 | +0.3038 |
| online_c0.1 | 6 | -40.1462 | 0.6130 | +0.1219 | +0.1220 | PASS | +0.0001 | +0.1874 |
| online_c0.2 | 1 | -41.2829 | 0.5121 | +0.0388 | +0.0409 | PASS | +0.0059 | +0.9929 |
| online_c0.2 | 2 | -41.2863 | 0.5136 | +0.0395 | +0.0403 | PASS | +0.0017 | +0.6786 |
| online_c0.2 | 4 | -41.2875 | 0.5134 | +0.0398 | +0.0400 | PASS | +0.0003 | +0.3692 |
| online_c0.2 | 6 | -41.2875 | 0.5138 | +0.0399 | +0.0399 | PASS | +0.0001 | +0.2286 |

### W3 IQ subset (the IQ engine where the test requires it)

Lockstep IQ engine with the in-process `Urap` barrier controller, 1000 s per h, paired with the batch path on the same trajectory and the same URAP draws (configuration mismatches counted).

| h (m) | IQ (dBm) | batch (dBm) | IQ - batch (dB) | IQ - batch midpoint (dB) | IQ - analytic (dB) | IQ CI half (dB) | mismatches | IQ run (s) | RTF |
|---|---|---|---|---|---|---|---|---|---|
| 1 | -41.30561 | -41.30569 | +0.000070 | -0.000003 | +0.1869 | 1.0596 | 0 | 1057 | 0.946 |
| 2 | -41.31240 | -41.31240 | +0.000004 | -0.000002 | +0.1871 | 1.0542 | 0 | 1044 | 0.958 |
| 4 | -41.31340 | -41.31337 | -0.000034 | -0.000008 | +0.1866 | 1.0567 | 0 | 645 | 1.549 |
| 6 | -41.31319 | -41.31319 | +0.000004 | -0.000008 | +0.1868 | 1.0562 | 0 | 638 | 1.566 |

The IQ - analytic column compares a 1000 s run (about 300 movement periods) with the stationary average; it is reported, not gated.

## W5: signaling delay tau_SUI, IQ vs the position-only oracle (h = 4 m)

`DelayedController(inner, tau_SUI, report_delay = 0)` with command latency 0, 1000 s per point, noise off. Oracle: the independent closed-form channel at the MU positions (piecewise-linear from the scenario's waypoints) on 1 ms midpoints, with its own exact ON/OFF optimum at the barrier positions (ExactOptimum) or with the engine-recorded decisions (WclHeuristic, which the oracle does not re-implement).

| controller | tau_SUI (s) | IQ (dBm) | oracle (dBm) | IQ - oracle (dB) | gain over LOS (dB) | decisions identical | barriers | IQ run (s) | RTF | result |
|---|---|---|---|---|---|---|---|---|---|---|
| ExactOptimum | 1/100 | -40.94691 | -40.94693 | +0.000023 | +0.3667 | 1.000 | 100000 | 2561 | 0.390 | PASS |
| ExactOptimum | 1/10 | -41.13246 | -41.13267 | +0.000210 | +0.1811 | 1.000 | 10000 | 744 | 1.345 | PASS |
| ExactOptimum | 3/10 | -41.28411 | -41.28417 | +0.000063 | +0.0295 | 1.000 | 3334 | 598 | 1.673 | PASS |
| ExactOptimum | 1 | -41.30949 | -41.30950 | +0.000010 | +0.0041 | 1.000 | 1000 | 526 | 1.900 | PASS |
| ExactOptimum | 3 | -41.31108 | -41.31108 | -0.000002 | +0.0025 | 1.000 | 334 | 499 | 2.002 | PASS |
| ExactOptimum | 10 | -41.31202 | -41.31201 | -0.000007 | +0.0016 | 1.000 | 100 | 478 | 2.094 | PASS |
| WclHeuristic | 1/100 | -41.04069 | -41.04071 | +0.000015 | +0.2729 | n/a (engine decisions) | 100000 | 1684 | 0.594 | PASS |
| WclHeuristic | 1/10 | -41.17759 | -41.17774 | +0.000152 | +0.1360 | n/a (engine decisions) | 10000 | 633 | 1.579 | PASS |
| WclHeuristic | 3/10 | -41.29390 | -41.29394 | +0.000045 | +0.0197 | n/a (engine decisions) | 3334 | 563 | 1.776 | PASS |
| WclHeuristic | 1 | -41.31169 | -41.31170 | +0.000005 | +0.0019 | n/a (engine decisions) | 1000 | 490 | 2.043 | PASS |
| WclHeuristic | 3 | -41.31204 | -41.31204 | -0.000004 | +0.0016 | n/a (engine decisions) | 334 | 501 | 1.996 | PASS |
| WclHeuristic | 10 | -41.31296 | -41.31296 | -0.000008 | +0.0006 | n/a (engine decisions) | 100 | 502 | 1.992 | PASS |

Context (stationary averages, h = 4 m): LOS -41.5002 dBm; exact optimum at tau_SUI -> 0 -41.1139 dBm; tau_SUI -> infinity limit (optimum of an independent position) -41.4989 dBm. The 1000 s runs cover about 300 movement periods, so their absolute levels differ from these by the trajectory's sampling error.

Monotonicity (paired per-second bins, batch-means CI of the difference, 20 batches): a step passes when the lower CI end of P(tau_b) - P(tau_a) is <= 0.

| controller | tau_a -> tau_b (s) | mean diff (dB) | result |
|---|---|---|---|
| ExactOptimum | 1/100 -> 1/10 | -0.18555 | PASS |
| ExactOptimum | 1/10 -> 3/10 | -0.15165 | PASS |
| ExactOptimum | 3/10 -> 1 | -0.02539 | PASS |
| ExactOptimum | 1 -> 3 | -0.00159 | PASS |
| ExactOptimum | 3 -> 10 | -0.00094 | PASS |
| WclHeuristic | 1/100 -> 1/10 | -0.13690 | PASS |
| WclHeuristic | 1/10 -> 3/10 | -0.11631 | PASS |
| WclHeuristic | 3/10 -> 1 | -0.01779 | PASS |
| WclHeuristic | 1 -> 3 | -0.00035 | PASS |
| WclHeuristic | 3 -> 10 | -0.00092 | PASS |

## W6: Doppler on/off (sample-level geometric vs position-snapshot average)

The IQ engine renders the chord-delay geometric Doppler at every sample (f_D <= 3.34 Hz here). The snapshot average uses |h(x(t))|^2 of frozen positions (no Doppler) with the engine-recorded configurations: left-point pieces (the batch path's semantics) at its automatic T_u, 5 ms and 1 ms; the independent oracle on 1 ms midpoints (the W5 reference of the same run); and the 5 ms midpoint average of the W3 IQ subset. The left-point rule has a first-order quadrature error that scales with the step: right after each barrier the configuration is the best one for the current position and the power then decays as the user moves, so holding the left value over-weights the start of every piece. The last column gives the ratio of the left-point difference to the 1 ms one (about 5 for a 5x step means first-order quadrature error); the midpoint references isolate the Doppler effect. The URAP rows have no systematic decay after a switch (random configurations), so they show no left-point bias and their ratios are noise.

| case | reference | IQ (dBm) | snapshot (dBm) | IQ - snapshot (dB) | result | ratio to 1 ms left |
|---|---|---|---|---|---|---|
| ExactOptimum tau=1/100 h=4 | auto_T_u | -40.946909 | -40.945498 | -0.001411 | PASS | 3.83 |
| ExactOptimum tau=1/100 h=4 | 5ms | -40.946909 | -40.945220 | -0.001689 | PASS | 4.58 |
| ExactOptimum tau=1/100 h=4 | 1ms | -40.946909 | -40.946541 | -0.000369 | PASS | - |
| ExactOptimum tau=1/10 h=4 | auto_T_u | -41.132460 | -41.122228 | -0.010232 | FAIL | 5.25 |
| ExactOptimum tau=1/10 h=4 | 5ms | -41.132460 | -41.121921 | -0.010539 | FAIL | 5.40 |
| ExactOptimum tau=1/10 h=4 | 1ms | -41.132460 | -41.130510 | -0.001951 | PASS | - |
| ExactOptimum tau=1 h=4 | auto_T_u | -41.309494 | -41.308578 | -0.000916 | PASS | 5.12 |
| ExactOptimum tau=1 h=4 | 5ms | -41.309494 | -41.308557 | -0.000937 | PASS | 5.24 |
| ExactOptimum tau=1 h=4 | 1ms | -41.309494 | -41.309315 | -0.000179 | PASS | - |
| WclHeuristic tau=1/10 h=4 | auto_T_u | -41.177592 | -41.170059 | -0.007533 | PASS | 5.24 |
| WclHeuristic tau=1/10 h=4 | 5ms | -41.177592 | -41.169831 | -0.007761 | PASS | 5.40 |
| WclHeuristic tau=1/10 h=4 | 1ms | -41.177592 | -41.176153 | -0.001438 | PASS | - |
| Urap tau=1/10 h=2 | auto_T_u | -41.312396 | -41.312396 | +4.94429e-07 | PASS | -0.34 |
| Urap tau=1/10 h=2 | 5ms | -41.312396 | -41.312399 | +0.000004 | PASS | -2.52 |
| Urap tau=1/10 h=2 | 1ms | -41.312396 | -41.312394 | -0.000001 | PASS | - |
| ExactOptimum tau=1/100 h=4 | 1ms_midpoint_oracle | -40.946909 | -40.946932 | +0.000023 | PASS | - |
| ExactOptimum tau=1/10 h=4 | 1ms_midpoint_oracle | -41.132460 | -41.132670 | +0.000210 | PASS | - |
| ExactOptimum tau=1 h=4 | 1ms_midpoint_oracle | -41.309494 | -41.309504 | +0.000010 | PASS | - |
| WclHeuristic tau=1/10 h=4 | 1ms_midpoint_oracle | -41.177592 | -41.177743 | +0.000152 | PASS | - |
| Urap tau=1/10 h=2 | 5ms_midpoint | -41.312396 | -41.312394 | -0.000002 | PASS | - |

`run_batch` itself (its own controller instance, 5 ms grid) over the first 100 s against the piecewise evaluation and the IQ run:

| case | run_batch vs piecewise (rel) | IQ - run_batch (dB) | run_batch (s) |
|---|---|---|---|
| ExactOptimum tau=1/100 h=4 | +0.000 | -0.001227 | 215.8 |
| ExactOptimum tau=1/10 h=4 | +3.33e-15 | -0.008488 | 57.6 |
| ExactOptimum tau=1 h=4 | -4.22e-15 | -0.000572 | 20.2 |
| WclHeuristic tau=1/10 h=4 | -5.44e-15 | -0.006212 | 76.3 |
| Urap tau=1/10 h=2 | -2.22e-16 | +0.000112 | 86.2 |

## Run times

| part | points | summed point wall (s) | longest point (s) |
|---|---|---|---|
| analytic | 24 | 795 | 99 |
| w3_chunk | 280 | 20966 | 395 |
| w3_cond | 80 | 724 | 14 |
| w3_iq | 4 | 4968 | 2093 |
| w5_iq | 12 | 13591 | 3631 |

Wall time per pass (manifests): final pass 3646 s; total 3646 s with 12 workers. In the final pass the sweep took 3633 s and W1 5 s. The table sums each point's own wall time over all passes. IQ RTF = simulated seconds / wall seconds of the lockstep engine in one worker process while the other workers (and other agents' jobs) loaded the CPU; these are not benchmark numbers (section 12 owns those).

## Files

`w1_position_histogram.csv`, `w1_summary.json`, `w3_conditional.csv` (same-trajectory W3), `w3_coefficient.csv` (main and sensitivity), `w3_iq.csv`, `w5.csv`, `w5_monotonicity.csv`, `w6.csv`, `w6_run_batch.csv`; per point `points/p*/` (point.json, summary.json, bins.npz of per-second powers for the IQ points); `meta.json` (all numbers, acceptance, data classes, output hashes); `manifest.json`.
