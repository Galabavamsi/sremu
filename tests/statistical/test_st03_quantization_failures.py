"""ST-03: quantization and failure Monte Carlo (nightly tier; as PH-13 and PH-15).

Larger trial counts than the CI versions: N = 4096 cells, 400 trials per alphabet; 10 %
dead cells on a focused N = 1024 panel over 1000 maps. Every trial is keyed
(``trial/<study>/<i>``) through ``sremu.rng`` (D14), never the global RNG.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu import rng as sremu_rng
from sremu.physics.nonideal import FailureMap, eta_alphabet, quantize_phases
from sremu.physics.response import builtin

pytestmark = pytest.mark.statistical


@pytest.mark.parametrize("ref", ["ideal_bbit(b=1)", "ideal_bbit(b=2)", "tang2022_ris1_27ghz"])
def test_st03_quantization_monte_carlo(ref: str):
    f = 27e9 if "tang" in ref else 1e9
    g = builtin(ref).gamma_at(f)
    n, trials = 4096, 400
    vals = []
    for i in range(trials):
        rng = sremu_rng.generator(20261010, "trial", f"st03/{ref}/{i}")
        phi = rng.uniform(0, 2 * math.pi, n)
        s = quantize_phases(-phi, g)
        vals.append(abs(np.sum(g[s] * np.exp(1j * phi))) ** 2 / n**2)
    got = 10 * math.log10(float(np.mean(vals)))
    assert abs(got - 10 * math.log10(eta_alphabet(g))) <= 0.05


def test_st03_dead_cells_monte_carlo():
    n, p, maps = 1024, 0.1, 1000
    z = np.exp(
        1j * sremu_rng.generator(1, "trial", "st03/focus", counter=0).uniform(0, 1e-9, n)
    )  # focused
    healthy = abs(np.sum(z)) ** 2
    powers = []
    for i in range(maps):
        fm = FailureMap.random(n, p, "dead", sremu_rng.generator(20261010, "failure", f"st03/{i}"))
        powers.append(abs(np.sum(z[~fm.dead])) ** 2)
    got = 10 * math.log10(float(np.mean(powers)) / healthy)
    closed = 10 * math.log10((1 - p) ** 2 + p * (1 - p) / n)
    assert abs(got - closed) <= 0.02
