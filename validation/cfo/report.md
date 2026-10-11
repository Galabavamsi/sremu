# V-CFO: receiver-LO (CFO) verification and the noisy estimator study (DESIGN.md 10.10)

Verification of sremu's receiver-LO (CFO) path against exact expectations and the Cramer-Rao bound. **Paper use (DESIGN D24):** no comparison with earlier simulators and no before/after framing. The tick-truncation rows (`LegacyTruncationAdapter`: sremu's CFO-free output re-timed with an integer number of samples per tick) may appear only as a generic design lesson; the DIG rows (ANTS 2025 Fig. 2, digitized, whose quantity came from the authors' earlier in-house simulator plus an external software receiver, not an RF measurement) and the V-CFO-2 regression are kept for internal traceability and are not for the paper (`paper_use` in meta.json). Data classes: DIG, SIM-sremu, ANALYTIC.

## Provenance

* Run directory: `validation/cfo/results/20261010T232549Z_vcfo` (manifest.json, meta.json, CSV/JSON data).
* sremu commit `9dca79f37955dc38fa8890110bea0f342eea207d`, dirty tree: True (dirty.patch sha256 `81af3aa08ec9060d3ca69033a7a75751954410438f1c83b551f1bbd2e09f4cac`).
* Platform: Windows-11-10.0.26200-SP0, AMD Ryzen 9 8945H w/ Radeon 780M Graphics, 16 logical CPUs, power plan Balanced.
* Started 2026-10-10T23:25:56.515638Z, ended 2026-10-10T23:28:50.222097Z, wall time 180.8 s.
* Host CPU busy fraction (all processes, sampled 22 times): median 1.00, max 1.00. Other agents' jobs shared the laptop; wall times are indicative, not benchmarks.

* Inputs: `data/validation/cfo/fig2_cfo_estimated_vs_actual.csv` (DIG, sha256 `bb3622039b04092f...`); `data/validation/cfo/fig2_cfo_stats.json` (DIG, sha256 `2a1c4115939361e7...`).

## Setup

BPSK test burst (200-symbol 1+1j header, 38 BPSK symbols at +-0.1(1+j), 10 samples/symbol, 2389 samples at 5880 S/s) through the in-process lockstep engine and the WP2 propagation model of a static SIM link with a 16 x 16 RIS at 2.4 GHz - 3050 Hz (|h| = 0.000862). The RX LO is offset by -f, so the received baseband rotates at +f. Noise-free sweeps: f = -1400 ... +1400 Hz in 100 Hz steps; estimators on header samples [30, 1950). V-CFO-4: the engine's physical noise (k T0 F f_s per RX, F = 0 dB), per-sample header SNR set through the TX full scale, 200 independently keyed trials (purpose `trial`) at each of 7 CFO points (-1200 ... +1200 Hz) per SNR; estimators on the N = 2000 header samples.

## Acceptance

| Check | Measured | Result |
|---|---|---|
| V-CFO-1: engine LS and lag-1, max \|f_hat - f\| <= 1e-6 Hz (noise off) | LS 3.10e-10 Hz; lag-1 1.95e-08 Hz | PASS |
| V-CFO-2 (internal): tick-truncation adapter slope 1.0691 +- 0.0005 | 1.069091 | PASS |
| V-CFO-4 at 10 dB: periodogram RMSE <= 1.5 x CRB at every CFO point | max point 1.058 x CRB (pooled 0.987) | PASS |
| V-CFO-4 at 20 dB: periodogram RMSE <= 1.5 x CRB at every CFO point | max point 1.001 x CRB (pooled 0.957) | PASS |
| V-CFO-4 at 0 dB (reported, not gated) | periodogram max point 1.072 x CRB (pooled 0.996) | n/a |

## Noise-free sweeps (figure F5a data: `vcfo_sweep_noise_free.csv`)

Least-squares line of estimate vs actual CFO over the 29 sweep points; RMSE of (estimate - actual); residual RMSE after the fitted line.

| series | slope | intercept (Hz) | RMSE (Hz) | max \|err\| (Hz) | resid. after fit (Hz) |
|---|---|---|---|---|---|
| new engine, unwrapped-phase LS | 1.000000 | -2.02e-11 | 9.78e-11 | 3.10e-10 | 9.44e-11 |
| new engine, lag-1 | 1.000000 | +1.41e-09 | 8.00e-09 | 1.95e-08 | 7.79e-09 |
| new engine, periodogram + parabolic | 1.000000 | +2.20e-12 | 2.40e-07 | 3.29e-07 | 2.39e-07 |
| integer samples per tick (generic pitfall), unwrapped-phase LS | 1.069091 | +1.50e-11 | 57.806 | 96.728 | 6.49e-11 |
| integer samples per tick (generic pitfall), lag-1 | 1.062201 | +1.05e-07 | 52.105 | 81.550 | 2.577 |
| DIG (ANTS Fig. 2; internal only) vs nominal grid | 1.068147 | +0.476 | 57.122 | 100.600 | 3.449 |
| DIG (internal only) vs digitized x positions | 1.068730 | +0.351 | 57.635 | 99.600 | 4.311 |

The tick-truncation adapter delivers 2233 of 2389 samples (11 samples per 2 ms tick; apparent scale 1.069091 = 11.76/11). DIG points carry +-4.5 Hz.

Paper use of the columns of `vcfo_sweep_noise_free.csv` (D24): `engine_*` are the paper-facing F5a series; `legacy_*` (the internal adapter's name) may appear only as the generic design lesson, under a neutral label such as 'integer samples per tick', never as an earlier simulator or a before/after; `dig_*` are internal only and are not plotted.

## V-CFO-4: RMSE vs the CRB (figure F5b data: `vcfo4_rmse_vs_crb.csv`)

CRB of the frequency of a complex tone in complex AWGN with unknown amplitude and phase (Rife & Boorstyn 1974), recomputed: sigma_f = f_s/(2 pi) sqrt(6/(SNR N (N^2 - 1))), N = 2000, f_s = 5880 S/s. Lag-1 small-error theory [DERIVED in `vcfo.py`]: var = 1/(SNR (N-1)^2) + 1/(2 SNR^2 (N-1)) rad^2. RMSE pooled over 7 x 200 = 1400 trials per SNR.

| SNR (dB) | CRB (Hz) | DESIGN [DERIVED] (Hz) | realised SNR (dB) | periodogram RMSE (Hz) | /CRB pooled | /CRB max point | lag-1 RMSE (Hz) | lag-1 theory (Hz) | lag-1 /CRB | LS RMSE (Hz) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.025629 | 0.0256 | 0.004 | 0.025527 | 0.996 | 1.072 | 14.7334 | 14.8079 | 574.9 | 233.7207 |
| 10 | 0.008105 | 0.0081 | 10.002 | 0.008003 | 0.987 | 1.058 | 1.4720 | 1.4874 | 181.6 | 0.1185 |
| 20 | 0.002563 | 0.0026 | 19.999 | 0.002453 | 0.957 | 1.001 | 0.1562 | 0.1552 | 60.9 | 0.0025 |

The lag-1 estimator meets its own small-error prediction but not the CRB: its variance falls as 1/N, the bound's as 1/N^3. The unwrapped-phase LS estimator fails at 0 dB (phase-unwrapping errors) and is reported only. Per-point rows (bias, SD, RMSE, ratio to the CRB per estimator, realised SNR, wall time) are in `vcfo4_points.csv`; trials in `vcfo4/p*/trials.csv`.

### Checks of the noise and of the estimator

For an efficient estimator the pooled RMSE/CRB has a sampling SE of about 1/sqrt(2 n) = 0.0189, 0.0189, 0.0189 (n = 1400 per SNR). Engine-free reference (the same periodogram estimator on a numpy tone in complex AWGN, keyed trials): RMSE/CRB 0 dB 0.998 (SE 0.016, n = 2000); 10 dB 1.032 (SE 0.016, n = 2000); 20 dB 0.990 (SE 0.016, n = 2000). Engine noise with the signal off (30 keyed runs, 71670 samples): variance 0.9940 x k T0 F f_s (relative SE 0.0037); |mean normalised autocorrelation| at lags 1, 2, 3, 5, 10 = 0.0007, 0.0038, 0.0007, 0.0018, 0.0029 (SE 0.0037).

Distance of the pooled ratio from 1 and the lower-tail probability of the pooled sum of squared errors under an efficient estimator (sum e^2/CRB^2 ~ chi2_n): 0 dB -0.2 SE, P_low = 0.421; 10 dB -0.7 SE, P_low = 0.256; 20 dB -2.3 SE, P_low = 0.011. Borderline: the lowest ratio is below 1 at Bonferroni 3 min P_low = 0.034 one-sided but not two-sided (3 min P = 0.068); with the white, calibrated noise above it is read as sampling fluctuation, which more trials would confirm or refute. The gate (RMSE <= 1.5 x CRB at every CFO point) does not depend on this.

## Run times

Noise-free sweeps 1.1 s; V-CFO-4 sweep 82.8 s wall with 2 sweep-runner workers (summed point time 153 s, longest point 24.3 s); noise check and engine-free reference 81.4 s. Indicative only (shared laptop).
