"""ST-02: SoS diffuse fading statistics (DESIGN.md sections 5.11, 10.5).

Spec row: f_D = 50 Hz, f_s = 10 kHz: max |ACF - J0| <= 0.05 for lags <= 40 ms; level
crossing rate at -10 dB within 10 % of 35.9 Hz; envelope KS < 0.01.

* ACF and LCR are checked on each of several independent keyed realizations (different
  links and a second tap), each a single 200 s series as in the G6 golden.
* The envelope KS is checked on the pooled envelopes of 8 independent 200 s realizations.
  A single realization's KS fluctuates widely because the time average of a Jakes-spectrum
  envelope converges slowly (the indicator covariance decays like J0^2, so its variance
  falls only as log(T)/T): measured on this host, 1 of 24 keys exceeded 0.01 at 200 s
  (max 0.013) and 1 of 8 at 1000 s. The pooled estimate converges to the M = 48 limit
  (KS 0.0024, the CLT correction of a 48-phasor sum, also checked below).

Statistics: ``tests/golden/harness.fading_stats`` (WP4, mirrors the MATLAB golden script);
theory: ``tests/oracles/mobility.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu import rng as srng
from sremu.physics.diffuse import sos_gain, sos_parameters
from tests.golden.harness import fading_stats
from tests.oracles import mobility as mob
from tests.oracles.tables import ORACLE

pytestmark = pytest.mark.statistical

FD, FS, N = 50.0, 10_000.0, 2_000_000
SEED = 20261010
CASES = [
    ("ap.tx0->mu.rx0", 0),
    ("ap.tx0->rx07.rx0", 0),
    ("ue.tx0->bs.rx0", 0),
    ("ap.tx0->mu.rx0", 1),
]


def _series(lk: str, tap: int = 0, n: int = N) -> np.ndarray:
    g = sos_gain(srng.derive_key(SEED, "diffuse", lk), np.arange(n) / FS, FD, 48, tap=tap)
    assert isinstance(g, np.ndarray)
    return g


def _ks_rayleigh(env: np.ndarray) -> float:
    return float(mob.ks_statistic(env, mob.rayleigh_cdf))


@pytest.mark.parametrize(("lk", "tap"), CASES)
def test_st02_acf_and_lcr_per_realization(lk: str, tap: int):
    g = _series(lk, tap)
    st = fading_stats(g, FS)
    m = st.lags_s <= 0.040 + 1e-12
    acf_dev = float(np.max(np.abs(st.acf[m] - mob.jakes_acf(st.lags_s[m], FD))))
    assert acf_dev <= ORACLE["ST02_acf_dev"].tol
    lcr_theory = float(mob.rayleigh_lcr_hz(10 ** (-10 / 20), FD))
    assert abs(lcr_theory - ORACLE["ST02_lcr_m10db_hz"].value) <= 0.05  # 35.9 Hz of the row
    assert abs(st.lcr_m10db_hz - lcr_theory) <= 0.10 * lcr_theory
    # one realization's time-average power is close to the unit ensemble mean
    assert abs(float(np.mean(np.abs(g) ** 2)) - 1.0) <= 0.02


def test_st02_envelope_ks_on_pooled_realizations():
    envs = []
    for j in range(8):
        g = _series(f"st02/{j}.tx0->rx.rx0")
        envs.append(np.abs(g) / math.sqrt(float(np.mean(np.abs(g) ** 2))))
    ks = _ks_rayleigh(np.concatenate(envs))
    assert ks < ORACLE["ST02_ks_envelope"].tol


def test_st02_m48_limit_distribution_is_within_the_ks_bound():
    """The long-time envelope distribution of one realization is that of a sum of M = 48
    unit phasors with independent uniform phases (distinct frequencies, Weyl). Its KS
    distance to Rayleigh is the CLT correction max|e^-x (2x - x^2)|/(4M) = 0.0024 (x the
    normalized power), well inside 0.01. Sampled with the realization's own keyed phases'
    distribution (uniform), 2e6 draws."""
    x = np.linspace(0.0, 20.0, 200_001)
    analytic = float(np.max(np.abs(np.exp(-x) * (2 * x - x * x)))) / (4 * 48)
    assert abs(analytic - 0.0024) <= 0.0001
    gen = srng.generator(SEED, "trial", "st02/weyl")
    acc = np.zeros(N, dtype=np.complex128)
    for _m in range(48):
        acc += np.exp(2j * math.pi * gen.random(N))
    ks = _ks_rayleigh(np.abs(acc) / math.sqrt(48))
    assert abs(ks - analytic) <= 5.0 / math.sqrt(N)  # sampling error of the empirical CDF
    assert ks < ORACLE["ST02_ks_envelope"].tol
    # and the realizations use exactly that limit: 48 equal amplitudes, uniform phases
    p = sos_parameters(srng.derive_key(SEED, "diffuse", CASES[0][0]), FD, 48)
    assert p.n_sinusoids == 48 and np.ptp(np.full(48, p.amplitude)) == 0


def test_st02_independent_links_are_uncorrelated():
    """Different links (keys) give uncorrelated diffuse processes: |rho| <= 5/sqrt(n_eff)
    with n_eff ~ 2 f_D T coherence intervals."""
    g1 = _series("a.tx0->b.rx0")
    g2 = _series("a.tx0->c.rx0")
    rho = abs(np.vdot(g1, g2)) / math.sqrt(float(np.vdot(g1, g1).real * np.vdot(g2, g2).real))
    n_eff = 2 * FD * N / FS
    assert rho <= 5.0 / math.sqrt(n_eff)
