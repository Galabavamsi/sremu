# V-LAB run report

Dataset: ANTS Fig. 4(b) digitized map (DIG). Study label: consistency check under geometric uncertainty.

Decisions: C1 fails, C2 fails, C3 fails.

| Model | test RMSE median (5-95 %) dB | bias median dB |
|---|---|---|
| M0 | 3.96 (3.92-3.99) | -0.59 |
| M0n | 3.77 (3.73-3.81) | -0.47 |
| M1 | 5.00 (3.36-6.75) | -0.64 |
| M2 | 5.11 (3.27-6.66) | -0.60 |
| M3 | 5.18 (3.30-6.79) | -0.63 |

C2: median Delta RMSE -1.15 dB, CI excludes 0 in 0.0 % of draws, Spearman rho -0.308 (CI low -0.758).
C3: median Delta RMSE -0.06 dB, CI excludes 0 in 4.6 % of draws.
Power analysis (protocol.json power_analysis_result; protocol locked): C2 detection rate 0.0, underpowered: True.

Secondary analysis (optimistic): weighted Delta RMSE M0-M2 -0.17 dB.
Sensitivity S1: C1 fails, C2 fails, C3 fails
Sensitivity S2: C1 fails, C2 fails, C3 fails
Sensitivity S-NF: C1 fails, C2 fails, C3 fails
Sensitivity S-OR: C1 fails, C2 fails, C3 fails
Point-vs-patch spread at the 1.8 m ring (M2, offset removed): median 0.12 dB, 95th percentile 0.65 dB (200 paired draws).
