# data/validation/tang2022: Tang et al. 2022 Fig. 10 (DIG) and Tables III/IV (REPORTED)

Owner: WP10 (V-TANG, DESIGN.md section 10.8). Files are byte-exact (`data/validation/** -text`);
`SHA256SUMS` lists their hashes and both the V-TANG-1 test and `validation/tang/vtang.py`
refuse to run when a hash differs.

| File | Data class | sha256 |
|---|---|---|
| `tang2022_fig10_vector.json` | DIG | `cf8793e9f9ed5a799d9aad4eff799278b8cf0a9d26af9bbe7e3c736cc6f03a20` |
| `tang2022_parameters.json` | REPORTED | see `SHA256SUMS` |

## `tang2022_fig10_vector.json` (DIG)

* **Source.** W. Tang et al., "Path Loss Modeling and Measurements for Reconfigurable
  Intelligent Surfaces in the Millimeter-Wave Frequency Band", arXiv:2101.08607v2, Fig. 10
  (page 22): four panels (a) d1 = 1 m, theta = 10 deg; (b) 2 m, 10 deg; (c) 1 m, 45 deg;
  (d) 2 m, 45 deg; 9 points each at d2 = 1 ... 5 m.
* **Series.** `refined_eq9` (the paper's general model, Eq. 9), `model2_2021` (Eq. 2),
  `farfield_eq8` (Eq. 8) and `measured` (the paper's chamber measurements).
* **Extraction.** Vector paths of the PDF figure read with PyMuPDF `get_drawings()`; axes
  calibrated by a linear fit to the tick-label centres
  (`D:\projects\ris\research\scratch\tang_fig10_vec2.py`, sha256
  `6ced85dbb0e37c373effa6803662996150be2482913d335b257545ac16d8e248`). Original file:
  `D:\projects\ris\data\digitized\tang2022_fig10_vector.json` (same sha256; copied
  2026-10-10). A second, LF-normalized copy for the WP4 oracle self-tests lives in
  `tests/oracles/data/` (sha256 `1e250304...ca1b9d`, see its `PROVENANCE.json`).
* **Uncertainty.** Vector extraction: the values are the plotted polyline vertices and
  marker centres, so the figure-to-number error is set by the axis calibration only; the
  x positions carry a constant +0.002 m offset against the nominal 0.5 m grid (the studies
  snap to the grid). The source states no measurement uncertainty for the measured points.
  This is a digitized published figure, **not** raw measurement data.

## `tang2022_parameters.json` (REPORTED)

Values of Tang 2022 Tables III/IV used by V-TANG (RIS1 geometry and coding-state
reflection coefficient, P_T, the measured G_t G_r G_line calibration, the cos^53.8 horn
pattern and the Fig. 10 geometry). Transcribed by hand; no text or figure copied.

## Use

* V-TANG-1 (GATE-PHY): sremu's K1 kernel through the scene model vs the published
  `refined_eq9` curves, max |difference| <= 0.05 dB.
* V-TANG-2: the `measured` points vs the model: bias, SD, RMSE (n = 36); regression gate
  |RMSE - 1.05| <= 0.05 dB. Reported as a property of Tang's published data and model, not
  as sremu accuracy (DESIGN.md 10.8, 1.6).
