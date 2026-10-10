"""EXPERIMENTAL: ITU-R P.526-16 smooth-earth diffraction (DESIGN.md section 5.9; PH-12).

Port of ``research/scratch/pm_checks.py::p526_smooth``: ITU-R P.526-16 section 3.2 (any
distance, >= 10 MHz) with the section 3.1.1 first-term residue, effective earth radius
``a_e = 8500 km`` by default. The only change from the scratch function is that the
wavelength uses ``sremu.units.C0`` (the scratch file used the literal 299.792458/f_MHz, the
same value).

Status: EXPERIMENTAL. It is NOT imported by the core (``sremu.physics`` never imports this
package; a test enforces it), it is not in any gate, and it is not validated against ITU-R
SG3 data. It appears only as a labelled sensitivity row in E-D (section 11). Its regression
test (PH-12, ``tests/experimental``) pins the scratch values: excess loss 37.14 dB at
52 km, h = 200/1.5 m, and 20.28 dB at 200/10 m, 100 MHz.

Units follow the scratch function: distances in km, heights in m, frequency in MHz,
``a_e`` in km, conductivity in S/m. The result is the excess loss over free space in dB
(positive = loss) and the branch used: ``"beyond_horizon"``, ``"clear"``, ``"Ah<0"`` or
``"interpolated"``.
"""

from __future__ import annotations

import math
from typing import Final, Literal

from sremu.units import C0

__all__ = ["EXPERIMENTAL", "P526Branch", "p526_smooth", "radio_horizon_km"]

EXPERIMENTAL: Final[bool] = True
"""Label: this module is experimental (not validated; not in any gate)."""

P526Branch = Literal["beyond_horizon", "clear", "Ah<0", "interpolated"]


def radio_horizon_km(h1_m: float, h2_m: float, ae_km: float = 8500.0) -> float:
    """Smooth-earth LOS distance ``sqrt(2 a_e)(sqrt(h1) + sqrt(h2))`` (km; heights in km
    inside the root, P.526-16 eq. 21)."""
    return math.sqrt(2 * ae_km) * (math.sqrt(h1_m / 1000.0) + math.sqrt(h2_m / 1000.0))


def p526_smooth(
    d_km: float,
    h1: float,
    h2: float,
    f_mhz: float,
    ae_km: float = 8500.0,
    eps: float = 15.0,
    sig: float = 1e-3,
    pol: str = "H",
) -> tuple[float, P526Branch]:
    """EXPERIMENTAL excess loss over free space (dB, positive = loss) and its branch.

    ITU-R P.526-16 sec. 3.2 (any distance, >= 10 MHz) with the sec. 3.1.1 first-term
    residue; ``h1``, ``h2`` antenna heights in m, ``eps``/``sig`` ground permittivity and
    conductivity, ``pol`` "H" or "V". Ported from ``pm_checks.py::p526_smooth``.
    """
    lam_m = C0 / (f_mhz * 1e6)

    def k_of(ae: float) -> float:
        kh = (
            0.36
            * (ae * f_mhz) ** (-1 / 3)
            * ((eps - 1) ** 2 + (18000 * sig / f_mhz) ** 2) ** (-0.25)
        )
        return kh if pol == "H" else kh * (eps**2 + (18000 * sig / f_mhz) ** 2) ** 0.5

    def first_term(d: float, ae: float) -> float:
        k = k_of(ae)
        beta = (1 + 1.6 * k**2 + 0.67 * k**4) / (1 + 4.5 * k**2 + 1.53 * k**4)
        x = 2.188 * beta * f_mhz ** (1 / 3) * ae ** (-2 / 3) * d
        fx = (
            11 + 10 * math.log10(x) - 17.6 * x
            if x >= 1.6
            else -20 * math.log10(x) - 5.6488 * x**1.425
        )

        def g_of(h: float) -> float:
            y = 9.575e-3 * beta * f_mhz ** (2 / 3) * ae ** (-1 / 3) * h
            b = beta * y
            g = (
                17.6 * (b - 1.1) ** 0.5 - 5 * math.log10(b - 1.1) - 8
                if b > 2
                else 20 * math.log10(b + 0.1 * b**3)
            )
            return max(g, 2 + 20 * math.log10(k))

        return fx + g_of(h1) + g_of(h2)  # 20 log(E/E0)

    d_los = radio_horizon_km(h1, h2, ae_km)
    if d_km >= d_los:
        return -first_term(d_km, ae_km), "beyond_horizon"
    d = d_km * 1000.0
    ae = ae_km * 1000.0
    c = (h1 - h2) / (h1 + h2)
    m = d**2 / (4 * ae * (h1 + h2))
    bb = (
        2
        * math.sqrt((m + 1) / (3 * m))
        * math.cos(math.pi / 3 + (1 / 3) * math.acos((3 * c / 2) * math.sqrt(3 * m / (m + 1) ** 3)))
    )
    d1 = d / 2 * (1 + bb)
    d2 = d - d1
    hcl = ((h1 - d1**2 / (2 * ae)) * d2 + (h2 - d2**2 / (2 * ae)) * d1) / d
    hreq = 0.552 * math.sqrt(d1 * d2 * lam_m / d)
    if hcl >= hreq:
        return 0.0, "clear"
    aem = 0.5 * (d / (math.sqrt(h1) + math.sqrt(h2))) ** 2 / 1000.0  # km
    ah = -first_term(d_km, aem)
    if ah < 0:
        return 0.0, "Ah<0"
    return (1 - hcl / hreq) * ah, "interpolated"
