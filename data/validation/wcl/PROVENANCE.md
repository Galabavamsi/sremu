# data/validation/wcl: WCL 2025 letter parameters (REPORTED)

Owner: WP10 (V-WCL, DESIGN.md section 10.11). Files are byte-exact (`data/validation/** -text`);
`SHA256SUMS` lists their hashes and `validation/wcl/vwcl.py` refuses to run when a hash differs.

| File | Data class | sha256 |
|---|---|---|
| `wcl2025_parameters.json` | REPORTED | `83699e9bab3971ac87171831cea15cfcf656ccfce836fb08c6f4f17d796a6394` |

## Source

S. Mishra and A. A. Kherani, "On the Control of Binary RIS Under User Mobility", IEEE
Wireless Communications Letters 14(4), April 2025, pp. 1139-1143. Local copy
`D:\projects\ris\data\papers\mishra2025_wcl_binary_ris_mobility.pdf`
(sha256 `5c2a91d8f190ed21d204c91ca827643a2524132dbd357c6be3feb933707c6986`, checked
2026-10-11). An earlier in-house simulator by the authors was used in the letter; sremu is an
independent new implementation (DESIGN D24: no comparison with that simulator).

## Content and method

Scenario values of the letter's Sec. II-A, III-B and IV-A (corridor length l = 10 m,
f_c = 1 GHz, element size k = w = lambda/4, N = floor(l/k) = 133, v = 1 m/s, q = 0.285,
P_T = 0 dBm, G_t = G_r = 1, 1-D random waypoint without pause and its stationary pdf
(Eq. 1), URAP with p = 0.2 and the definition of its Eq. (9), h_L = 4 m and h_H = 6 m, the
Fig. 6 geometry, the State Update Interval tau_SUI, the AP at the corridor origin),
transcribed by hand on 2026-10-10 as values only (no text or figures copied); spot-checked
against the letter's text on 2026-10-11. The JSON also lists what the letter does not
state (the URAP redraw period; the plotted values of Figs. 5, 7 and 8; the v1 commit, H-1)
and sremu's documented deviations (the AP offset d0 = 0.5 m perpendicular to the user line
with its sensitivity runs; the area-tied K1 kernel instead of the letter's Eq. (2), about
which nothing is stated until the letter's authors confirm, DESIGN 1.6 rule 6 and H-3).

## Use

V-WCL W1/W3/W5/W6 (`validation/wcl/vwcl.py`, report in `validation/wcl/report.md`) and the
E-A A0 reproduction regime. These are parameters, not measurements; V-WCL is a verification
of sremu against closed forms under the area-tied kernel, not a validation.
