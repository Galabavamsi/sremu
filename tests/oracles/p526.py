"""EXPERIMENTAL: ITU-R P.526-16 smooth-earth diffraction, ported from ``pm_checks.py``.

Regression oracle for PH-12 (not in any gate; DESIGN.md sections 5.9, 10.2). Ported verbatim
from ``research/scratch/pm_checks.py::p526_smooth`` except that the wavelength uses
``sremu.units.C0`` (the scratch file used the literal 299.792458/f_MHz, which is the same
value). Not validated against ITU-R SG3 data.
"""

from __future__ import annotations

import math

from sremu.units import C0

__all__ = ["p526_smooth_excess_db", "radio_horizon_km"]


def radio_horizon_km(h1_m: float, h2_m: float, ae_km: float = 8500.0) -> float:
    """Smooth-earth line-of-sight distance ``sqrt(2 a_e)(sqrt(h1) + sqrt(h2))`` in km.

    ITU-R P.526-16 eq. (21) with ``a_e`` in km and heights converted to km; the branch point
    of :func:`p526_smooth_excess_db` (``"beyond_horizon"`` when ``d >= d_los``).
    """
    return math.sqrt(2.0 * ae_km) * (math.sqrt(h1_m / 1000.0) + math.sqrt(h2_m / 1000.0))


def p526_smooth_excess_db(
    d_km: float,
    h1_m: float,
    h2_m: float,
    f_mhz: float,
    ae_km: float = 8500.0,
    eps_r: float = 15.0,
    sigma_s_per_m: float = 1e-3,
    pol: str = "H",
) -> tuple[float, str]:
    """Excess loss over free space (dB, positive = loss) and the branch used.

    ITU-R P.526-16 section 3.2 (any distance, >= 10 MHz) with the section 3.1.1 first-term
    residue. Branches: ``"beyond_horizon"``, ``"clear"``, ``"Ah<0"``, ``"interpolated"``.
    """
    lam_m = C0 / (f_mhz * 1e6)

    def k_of(ae: float) -> float:
        kh = (
            0.36
            * (ae * f_mhz) ** (-1 / 3)
            * ((eps_r - 1) ** 2 + (18000 * sigma_s_per_m / f_mhz) ** 2) ** (-0.25)
        )
        return kh if pol == "H" else kh * (eps_r**2 + (18000 * sigma_s_per_m / f_mhz) ** 2) ** 0.5

    def first_term(d: float, ae: float) -> float:
        K = k_of(ae)
        beta = (1 + 1.6 * K**2 + 0.67 * K**4) / (1 + 4.5 * K**2 + 1.53 * K**4)
        X = 2.188 * beta * f_mhz ** (1 / 3) * ae ** (-2 / 3) * d
        Fx = (
            11 + 10 * math.log10(X) - 17.6 * X
            if X >= 1.6
            else -20 * math.log10(X) - 5.6488 * X**1.425
        )

        def G(h: float) -> float:
            Y = 9.575e-3 * beta * f_mhz ** (2 / 3) * ae ** (-1 / 3) * h
            B = beta * Y
            g = (
                17.6 * (B - 1.1) ** 0.5 - 5 * math.log10(B - 1.1) - 8
                if B > 2
                else 20 * math.log10(B + 0.1 * B**3)
            )
            return max(g, 2 + 20 * math.log10(K))

        return Fx + G(h1_m) + G(h2_m)

    d_los = radio_horizon_km(h1_m, h2_m, ae_km)
    if d_km >= d_los:
        return -first_term(d_km, ae_km), "beyond_horizon"
    d = d_km * 1000.0
    ae = ae_km * 1000.0
    c = (h1_m - h2_m) / (h1_m + h2_m)
    m = d**2 / (4 * ae * (h1_m + h2_m))
    bb = (
        2
        * math.sqrt((m + 1) / (3 * m))
        * math.cos(math.pi / 3 + (1 / 3) * math.acos((3 * c / 2) * math.sqrt(3 * m / (m + 1) ** 3)))
    )
    d1 = d / 2 * (1 + bb)
    d2 = d - d1
    hcl = ((h1_m - d1**2 / (2 * ae)) * d2 + (h2_m - d2**2 / (2 * ae)) * d1) / d
    hreq = 0.552 * math.sqrt(d1 * d2 * lam_m / d)
    if hcl >= hreq:
        return 0.0, "clear"
    aem = 0.5 * (d / (math.sqrt(h1_m) + math.sqrt(h2_m))) ** 2 / 1000.0
    Ah = -first_term(d_km, aem)
    if Ah < 0:
        return 0.0, "Ah<0"
    return (1 - hcl / hreq) * Ah, "interpolated"
