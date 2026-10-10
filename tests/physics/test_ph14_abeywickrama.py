"""PH-14: Abeywickrama practical phase shifter (section 10.2).

``eta = (mean_theta beta(theta))^2``: (beta_min, alpha) = (0.2, 1.6) -> -5.508 dB,
(0.8, 1.6) -> -1.085 dB (+-0.01 dB); independent of phi (1e-6 dB).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from sremu.physics.nonideal import eta_abeywickrama
from sremu.physics.response import abeywickrama_table, builtin

pytestmark = pytest.mark.physics


def _eta_quad(bmin: float, alpha: float, phi: float) -> float:
    def beta(th: float) -> float:
        return (1 - bmin) * ((math.sin(th - phi) + 1) / 2) ** alpha + bmin

    m = quad(beta, -math.pi, math.pi, epsabs=1e-14, epsrel=1e-13, limit=200)[0] / (2 * math.pi)
    return m * m


@pytest.mark.parametrize(("bmin", "alpha", "spec_db"), [(0.2, 1.6, -5.508), (0.8, 1.6, -1.085)])
def test_ph14_eta_matches_quadrature_and_oracle(bmin: float, alpha: float, spec_db: float):
    got_db = 10 * math.log10(eta_abeywickrama(bmin, alpha))
    assert abs(got_db - spec_db) <= 0.01
    for phi in (0.0, 0.43 * math.pi, 1.7, -2.9):
        q_db = 10 * math.log10(_eta_quad(bmin, alpha, phi))
        assert abs(q_db - got_db) <= 1e-6
        assert abs(10 * math.log10(eta_abeywickrama(bmin, alpha, phi)) - got_db) <= 1e-6


def test_ph14_table_samples_follow_the_model():
    bmin, alpha, phi = 0.2, 1.6, 0.43 * math.pi
    t = abeywickrama_table(8, bmin, math.degrees(phi), alpha)  # phi_deg (WP1 convention)
    g = t.gamma_at(3.5e9)
    th = 2 * np.pi * np.arange(256) / 256
    beta = (1 - bmin) * ((np.sin(th - phi) + 1) / 2) ** alpha + bmin
    assert np.max(np.abs(g - beta * np.exp(1j * th))) <= 1e-12
    # the sample mean of |Gamma| over a fine uniform grid is the model mean
    assert (
        abs(
            10 * math.log10(np.mean(np.abs(g)) ** 2)
            - 10 * math.log10(eta_abeywickrama(bmin, alpha))
        )
        <= 1e-6
    )
    named = builtin(
        f"abeywickrama(b=2, beta_min={bmin}, phi_deg={math.degrees(phi)}, alpha={alpha})"
    ).gamma_at(1e9)
    th2 = 2 * np.pi * np.arange(4) / 4
    b2 = (1 - bmin) * ((np.sin(th2 - phi) + 1) / 2) ** alpha + bmin
    assert np.max(np.abs(named - b2 * np.exp(1j * th2))) <= 1e-12
