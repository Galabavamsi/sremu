"""PH-25: binary m/d decomposition (section 5.6 table; section 10.2).

``Gamma = m + d s`` with ``m = (a+b)/2``, ``d = (a-b)/2``: decomposed vs direct evaluation
<= 1e-12; lab pair |d| = -0.18 dB, |m| = -13.86 dB at 127.9 deg (+-0.01 dB).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.physics.nonideal import binary_md, md_channel
from sremu.physics.response import builtin

pytestmark = pytest.mark.physics


def _lab(mag2: float) -> tuple[complex, complex]:
    g = builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1.0, "sigma2": mag2}).gamma_at(3.5e9)
    return complex(g[0]), complex(g[1])


@pytest.mark.parametrize(
    ("mag2", "d_db", "m_db", "m_deg"),
    [(1.0, -0.18, -13.86, 127.9), (0.7, -1.59, -12.90, None), (0.5, -2.66, -10.81, None)],
)
def test_ph25_lab_values(mag2: float, d_db: float, m_db: float, m_deg: float | None):
    a, b = _lab(mag2)
    m, d = binary_md(a, b)
    assert abs(m - (a + b) / 2) == 0 and abs(d - (a - b) / 2) == 0
    assert abs(20 * math.log10(abs(d)) - d_db) <= 0.01
    assert abs(20 * math.log10(abs(m)) - m_db) <= 0.01
    if m_deg is not None:
        assert abs(math.degrees(math.atan2(m.imag, m.real)) - m_deg) <= 0.05


def test_ph25_ideal_on_off():
    g = builtin("on_off").gamma_at(1e9)
    m, d = binary_md(complex(g[1]), complex(g[0]))  # s = +1 is ON
    assert 20 * math.log10(abs(m)) == pytest.approx(-6.0206, abs=0.01)
    assert 20 * math.log10(abs(d)) == pytest.approx(-6.0206, abs=0.01)


def test_ph25_decomposed_equals_direct(rng: np.random.Generator):
    a, b = _lab(0.7)
    m, d = binary_md(a, b)
    for _ in range(50):
        n = 64
        c = rng.normal(size=n) + 1j * rng.normal(size=n)
        z0 = complex(rng.normal(), rng.normal())
        signs = rng.choice([-1.0, 1.0], size=n)
        direct = z0 + np.sum(np.where(signs > 0, a, b) * c)
        assert abs(md_channel(z0, c, m, d, signs) - direct) <= 1e-12 * max(1.0, abs(direct))
