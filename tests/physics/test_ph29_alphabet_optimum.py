"""PH-29: general finite-alphabet optimum, S = 4, N = 10 (4^10 configurations), 50 random
instances with unequal amplitudes and a direct path; rel <= 1e-12 vs brute force."""

from __future__ import annotations

import numpy as np
import pytest

from sremu.physics.optimize import exact_alphabet_optimum, exact_alphabet_optimum_reference
from tests.physics._closed_forms import subset_sums

pytestmark = pytest.mark.physics


def test_ph29_s4_n10_vs_brute_force(rng: np.random.Generator):
    for trial in range(50):
        alphabet = rng.uniform(0.3, 1.0, 4) * np.exp(1j * rng.uniform(0, 2 * np.pi, 4))
        c = rng.uniform(0.2, 1.0, 10) * np.exp(1j * rng.uniform(0, 2 * np.pi, 10))
        z = alphabet[:, None] * c[None, :]
        if trial % 2:  # some instances with per-element (non-separable) alphabets
            z = z * rng.uniform(0.5, 1.5, (4, 10))
        z0 = complex(rng.normal(0, 2), rng.normal(0, 2))
        brute = float(np.max(np.abs(subset_sums(z0, z))))
        val, s = exact_alphabet_optimum(z0, z)
        assert abs(val - brute) <= 1e-12 * brute
        assert s.dtype == np.int16 and s.min() >= 0 and s.max() <= 3
        ref, _ = exact_alphabet_optimum_reference(z0, z)
        assert abs(ref - brute) <= 1e-12 * brute
