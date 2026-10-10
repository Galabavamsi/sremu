"""PH-24: exact binary optimum, sweep vs brute force (N = 20, 2^20 configurations, 100
positions; rel <= 1e-12; section 10.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.physics.optimize import exact_alphabet_optimum, exact_alphabet_optimum_reference
from sremu.physics.response import builtin
from sremu.physics.ris_kernel import k1_terms
from sremu.units import C0
from tests.physics._closed_forms import make_panel, subset_sums

pytestmark = pytest.mark.physics

FC = 3.5e9


def test_ph24_sweep_equals_brute_force_100_positions(rng: np.random.Generator):
    lam = C0 / FC
    pos, area = make_panel(5 * lam / 2, 4 * lam / 2, 5, 4)  # N = 20 cells
    gam = builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1.0, "sigma2": 0.7}).gamma_at(FC)
    tx = np.array([-1.0, 0.3, 1.5])
    for _ in range(100):
        rx = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.5, 2.5)])
        c = k1_terms(tx, rx, pos, area, np.array([0.0, 0.0, 1.0]), FC)[0]
        d = float(np.linalg.norm(rx - tx))
        z0 = lam / (4 * math.pi * d) * np.exp(-2j * math.pi * d / lam)  # LOS term
        z = gam[:, None] * c[None, :]
        brute = float(np.max(np.abs(subset_sums(z0, z))))
        val, s = exact_alphabet_optimum(z0, z)
        assert abs(val - brute) <= 1e-12 * brute
        assert abs(abs(z0 + np.sum(z[s, np.arange(20)])) - val) <= 1e-12 * brute
        ref, _ = exact_alphabet_optimum_reference(z0, z)
        assert abs(ref - brute) <= 1e-12 * brute


def test_ph24_degenerate_inputs():
    z = np.ones((2, 5), complex)  # identical states: no critical angles
    val, s = exact_alphabet_optimum(1j, z)
    assert val == abs(1j + 5) and s.tolist() == [0] * 5
    z = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], complex)  # identical elements
    val, _ = exact_alphabet_optimum(0.0, z)
    assert val == 3.0
