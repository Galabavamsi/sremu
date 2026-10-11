# data/validation/cfo: ANTS 2025 Fig. 2 (CFO estimated vs injected), digitized

Owner: WP10 (V-CFO, DESIGN.md section 10.10). Data class: **DIG** (digitized from a raster
figure; not raw data). The underlying quantity is **emulated** (v1 emulator channel and an
external software BPSK receiver), not an RF measurement.

| File | sha256 (bytes as committed; `data/validation/** -text`) |
|---|---|
| `fig2_cfo_estimated_vs_actual.csv` | `bb3622039b04092fcdbd1b3836ac8b2e877e37fa3096ffe259a7f4af801210cc` |
| `fig2_cfo_stats.json` | `2a1c4115939361e7267940976d45448176b790fad57c6794dc8bb37927f28ea4` |

`SHA256SUMS` lists the same hashes; the study script (`validation/cfo/vcfo.py`) refuses to
run when they differ.

**Source.** Byte-exact copies (2026-10-10) of
`D:\projects\ris\data\digitized\fig2_cfo\fig2_cfo_estimated_vs_actual.csv` and
`fig2_cfo_stats.json`, produced by `research/scratch/s14_fig2.py` and
`s27_fig2_fig6_final.py` (sha256 `d768671a...fe36` and `cf05bf95...a03`) from Fig. 2 of the
ANTS 2025 manuscript (raster image Im2, 640 x 480 px; manuscript MD5
`cd4acf7c3ac5845bd9913354027410f6`). See `D:\projects\ris\data\digitized\README.md` section 3.

**Method and uncertainty.** Axes calibrated from tick-mark centroids (residual <= 0.43 px);
markers located on the raster with sub-pixel refinement. Per-point uncertainty +-4.5 Hz
(0.5 px marker localisation plus tick snapping at 8.09 Hz/px). The digitized x positions lie
within 3.5 Hz of the nominal -1400 ... +1400 Hz grid (100 Hz steps, 29 points).

**Recorded statistics** (`fig2_cfo_stats.json`, recomputed by `vcfo.py` from the CSV):
least-squares slope 1.0681, intercept +0.5 Hz, RMSE (estimate - actual) 57.1 Hz,
residual after the linear fit 3.4 Hz (digitization floor).

**Use.** Internal traceability only. Under DESIGN D24 (fresh-attempt framing, 11 Oct 2026) the
paper contains no comparison with the earlier simulator and no before/after framing, so these
points are not featured in the paper; `validation/cfo/vcfo.py` re-analyses them and labels them
`paper_use: internal only` in its `meta.json`. They are not a pass criterion of the engine.
