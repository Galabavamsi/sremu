"""PH-13: large-N quantization loss (section 10.2).

``eta(b) = (2^b/pi sin(pi/2^b))^2`` = -3.922 / -0.912 / -0.224 dB (quadrature, +-0.001 dB);
lab pair (203.41 deg) -4.105 dB; Tang RIS1 / RIS2 alphabets -5.93 / -6.16 dB; Monte Carlo
with N = 4096 cells and 100 trials within +-0.15 dB.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from sremu.physics.nonideal import eta_alphabet, eta_binary, eta_bits, quantize_phases
from sremu.physics.response import builtin

pytestmark = pytest.mark.physics


def _db(x: float) -> float:
    return 10 * math.log10(x)


def _eta_quadrature(gammas: np.ndarray) -> float:
    """(E_phi max_s Re{Gamma_s e^{-j phi}})^2 by adaptive quadrature (independent)."""
    g = np.asarray(gammas, complex)

    def f(phi: float) -> float:
        return float(np.max(np.real(g * np.exp(-1j * phi))))

    pts = sorted(
        {
            (math.atan2(d.imag, d.real) + o) % (2 * math.pi)
            for a in g
            for b in g
            if a != b
            for d in [a - b]
            for o in (math.pi / 2, -math.pi / 2)
        }
    )
    val, _ = quad(f, 0.0, 2 * math.pi, points=pts or None, limit=200, epsabs=1e-13)
    return (val / (2 * math.pi)) ** 2


@pytest.mark.parametrize(("bits", "spec_db"), [(1, -3.922), (2, -0.912), (3, -0.224)])
def test_ph13_uniform_bbit_eta(bits: int, spec_db: float):
    alphabet = np.exp(2j * np.pi * np.arange(2**bits) / 2**bits)
    q = _eta_quadrature(alphabet)
    assert abs(_db(eta_bits(bits)) - _db(q)) <= 0.001
    assert abs(_db(eta_bits(bits)) - spec_db) <= 0.001
    assert abs(eta_alphabet(alphabet) - eta_bits(bits)) <= 1e-12
    tab = builtin(f"ideal_bbit({bits})")
    assert abs(eta_alphabet(tab.gamma_at(1e9)) - eta_bits(bits)) <= 1e-12


def test_ph13_lab_pair_and_tang_alphabets():
    lab = builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1.0, "sigma2": 1.0}).gamma_at(3.5e9)
    delta = math.radians(49.64 - (-153.77))  # 203.41 deg
    assert abs(_db(eta_binary(delta)) - (-4.105)) <= 0.001
    assert abs(_db(eta_alphabet(lab)) - _db(eta_binary(delta))) <= 1e-9
    assert abs(_db(eta_alphabet(lab)) - _db(_eta_quadrature(lab))) <= 0.001
    for name, f, spec in (
        ("tang2022_ris1_27ghz", 27e9, -5.93),
        ("tang2022_ris2_33ghz", 33e9, -6.16),
    ):
        g = builtin(name).gamma_at(f)
        assert abs(_db(eta_alphabet(g)) - _db(_eta_quadrature(g))) <= 0.001
        assert abs(_db(eta_alphabet(g)) - spec) <= 0.005


@pytest.mark.parametrize("which", ["1bit", "2bit", "lab"])
def test_ph13_monte_carlo_n4096(which: str, rng: np.random.Generator):
    if which == "lab":
        g = builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1.0, "sigma2": 1.0}).gamma_at(3.5e9)
    else:
        b = int(which[0])
        g = np.exp(2j * np.pi * np.arange(2**b) / 2**b)
    n, trials = 4096, 100
    ratios = []
    for _ in range(trials):
        phi = rng.uniform(0, 2 * math.pi, n)  # channel phases
        # per-element best choice for the common direction 0: quantize the phase -phi
        s = quantize_phases(-phi, g)
        ratios.append(abs(np.sum(g[s] * np.exp(1j * phi))) ** 2 / n**2)
    assert abs(_db(float(np.mean(ratios))) - _db(eta_alphabet(g))) <= 0.15
