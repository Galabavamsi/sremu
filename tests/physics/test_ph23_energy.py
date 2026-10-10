"""PH-23: energy diagnostic, scattered <= incident per panel (section 10.2).

The far-zone scattered power of the K1 sources is integrated in closed form (hemisphere
integrals of the obliquity-weighted plane waves). It must not exceed the power the panel
intercepts for any passive configuration, for every tested geometry with pitch <= lambda/2.
A single-cell closed form checks the normalization: ``P_scat/P_inc = pi A / lambda^2``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import dblquad

from sremu.physics.optimize import continuous_optimum
from sremu.physics.ris_kernel import energy_budget, k1_terms
from sremu.units import C0
from tests.physics._closed_forms import make_panel

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
NRM = np.array([0.0, 0.0, 1.0])


def test_ph23_single_cell_closed_form():
    area = (LAM / 2) ** 2
    tx = np.array([0.4, -0.3, 2.0])
    p_scat, p_inc = energy_budget(tx, np.zeros((1, 3)), area, NRM, FC, 1.0)
    assert abs(p_scat / p_inc - math.pi * area / LAM**2) <= 1e-12
    # brute-force hemisphere integral of |h|^2 r^2 (4 pi / lambda^2) for the same cell
    r_far = 1e4
    ci = 2.0 / float(np.linalg.norm(tx))

    def integrand(theta: float, phi: float) -> float:
        cs = math.cos(theta)
        h = area * math.sqrt(ci * cs) / (4 * math.pi * float(np.linalg.norm(tx)) * r_far)
        return abs(h) ** 2 * r_far**2 * 4 * math.pi / LAM**2 * math.sin(theta)

    num, _ = dblquad(integrand, 0, 2 * math.pi, 0, math.pi / 2)
    assert abs(num / p_scat - 1) <= 1e-8


CASES = [
    (0.6, 0.5, (0.0, 0.0, 0.5)),
    (0.6, 0.25, (-0.8, 0.2, 1.0)),
    (1.0, 0.5, (2.0, 0.0, 0.7)),
    (2.0, 0.5, (0.0, 0.0, 20.0)),
]


def _configs(side: float, pitch_frac: float, tx: np.ndarray, rng: np.random.Generator):  # type: ignore[no-untyped-def]
    ns = math.ceil(side / (pitch_frac * LAM))
    pos, area = make_panel(side, side, ns, ns)
    configs = [np.ones(ns * ns, complex)]  # uniform
    for _ in range(3):  # random passive configurations
        configs.append(rng.uniform(0, 1, ns * ns) * np.exp(1j * rng.uniform(0, 2 * np.pi, ns * ns)))
    for _ in range(3):  # focused toward random receivers (strongly coherent)
        rx = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.5, 3)])
        c = k1_terms(tx, rx, pos, area, NRM, FC)[0]
        _, ph = continuous_optimum(0.0, c)
        configs.append(np.exp(1j * ph))
    return pos, area, configs


@pytest.mark.parametrize(("side", "pitch_frac", "tx"), CASES)
def test_ph23_scattered_not_above_incident(side: float, pitch_frac: float, tx: tuple, rng):  # type: ignore[no-untyped-def]
    """Default obliquity sqrt(cos cos): every tested configuration is passive."""
    t = np.asarray(tx, float)
    pos, area, configs = _configs(side, pitch_frac, t, rng)
    p_scat, p_inc = energy_budget(t, pos, area, NRM, FC, np.stack(configs), obliquity="sqrt")
    for ps in p_scat:
        assert 0 < ps <= p_inc * (1 + 1e-9), ps / p_inc
    one, _ = energy_budget(t, pos, area, NRM, FC, configs[-1], obliquity="sqrt")
    assert abs(one - p_scat[-1]) <= 1e-12 * p_scat[-1]  # batched == single


@pytest.mark.xfail(
    strict=True,
    reason="OPEN ISSUE: the optional Kirchhoff obliquity (cos_i + cos_s)/2 keeps a finite "
    "weight at grazing scattering; focused oblique configurations scatter up to 1.23x the "
    "incident power (+0.9 dB). The default obliquity satisfies PH-23.",
)
def test_ph23_kirchhoff_obliquity_energy_bound(rng: np.random.Generator):
    worst = 0.0
    for side, pitch_frac, tx in CASES:
        t = np.asarray(tx, float)
        pos, area, configs = _configs(side, pitch_frac, t, rng)
        p_scat, p_inc = energy_budget(
            t, pos, area, NRM, FC, np.stack(configs), obliquity="kirchhoff"
        )
        worst = max(worst, float(np.max(p_scat)) / p_inc)
    assert worst <= 1 + 1e-9, worst


def test_ph23_teeth_oversized_point_cells_violate():
    """Unsubdivided cells larger than lambda/2 are outside the validity domain: with
    lambda-sized point cells the diagnostic reports more scattered than incident power."""
    pos, area = make_panel(4 * LAM, 4 * LAM, 4, 4)
    p_scat, p_inc = energy_budget(np.array([0.0, 0.0, 3.0]), pos, area, NRM, FC, 1.0)
    assert p_scat > p_inc
