"""PH-30: quantizer ``argmax_s Re{Gamma_s e^{-j phi}}`` equals a per-element brute force
(Tang alphabet, unequal amplitudes; exact)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.physics.nonideal import quantize_phases
from sremu.physics.response import builtin

pytestmark = pytest.mark.physics


def _brute(phi: float, alphabet: list[complex]) -> int:
    best, best_s = -math.inf, -1
    for s, g in enumerate(alphabet):
        v = abs(g) * math.cos(math.atan2(g.imag, g.real) - phi)
        if v > best + 1e-12:  # first maximum wins (ties resolve to the lowest index)
            best, best_s = v, s
    return best_s


@pytest.mark.parametrize("name", ["tang2022_ris1_27ghz", "tang2022_ris2_33ghz"])
def test_ph30_quantizer_matches_brute_force(name: str, rng: np.random.Generator):
    f = 27e9 if "ris1" in name else 33e9
    g = builtin(name).gamma_at(f)
    phi = rng.uniform(-3 * math.pi, 3 * math.pi, 2000)
    s = quantize_phases(phi, g)
    expected = [_brute(float(p), [complex(x) for x in g]) for p in phi]
    assert s.tolist() == expected


def test_ph30_per_element_reference(rng: np.random.Generator):
    n = 200
    gref = rng.uniform(0.2, 1, (3, n)) * np.exp(1j * rng.uniform(0, 2 * np.pi, (3, n)))
    phi = rng.uniform(0, 2 * math.pi, n)
    s = quantize_phases(phi, gref)
    for i in range(n):
        assert s[i] == _brute(float(phi[i]), [complex(x) for x in gref[:, i]])
