"""ST-05: CFO estimator vs the Cramer-Rao bound through the engine (DESIGN.md 10.5, 10.10).

Reduced V-CFO-4: the BPSK test burst through the in-process lockstep engine with the engine's
physical noise (k T0 F f_s, F = 0 dB), per-sample header SNR 10 and 20 dB set through the
TX full scale, 2 CFO points x 30 independently keyed trials per SNR (the study runs
7 x 200). The periodogram + parabolic estimator on the N = 2000 header samples must reach
a pooled RMSE <= 1.5 x CRB at both SNRs. The study code (``validation/cfo/vcfo.py``) is
reused, so the test exercises exactly the study's link, keys and estimators.

With 60 trials, RMSE^2/CRB^2 ~ chi2_60/60 for an efficient estimator, so a false failure
(ratio > 1.5, i.e. chi2_60/60 > 2.25) has probability < 1e-6; the keys are fixed, so the
result is deterministic.
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.statistical

REPO = Path(__file__).resolve().parents[2]
CFO_POINTS = (-800.0, 400.0)
N_TRIALS = 30


def _vcfo() -> ModuleType:
    for p in (REPO / "validation", REPO / "validation" / "cfo"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module("vcfo")


def test_st05_crb_recomputed() -> None:
    vc = _vcfo()
    # the DESIGN 10.10 [DERIVED] values, N = 2000 at 5880 S/s
    for snr, ref in ((0.0, 0.0256), (10.0, 0.0081), (20.0, 0.0026)):
        assert round(vc.crb_std_hz(snr), 4) == ref
    # closed form: fs/(2 pi) sqrt(6/(SNR N (N^2-1)))
    n, fs = 2000, 5880.0
    assert vc.crb_std_hz(10.0) == pytest.approx(
        fs / (2 * math.pi) * math.sqrt(6 / (10 * n * (n * n - 1))), rel=1e-12
    )


@pytest.mark.parametrize("snr_db", [10.0, 20.0])
def test_st05_periodogram_rmse_within_1p5_crb(snr_db: float) -> None:
    vc = _vcfo()
    lk = vc.Link()
    fs_dbm = lk.full_scale_for_snr(snr_db)
    err = []
    snr_hat = []
    n0, n1 = vc.HEADER
    for f in CFO_POINTS:
        k = np.arange(n0, n1)
        ref = np.full(n1 - n0, 1 + 1j) * np.exp(2j * np.pi * f * k / float(vc.FS))
        for i in range(N_TRIALS):
            y = lk.receive(
                seed=vc.trial_seed(snr_db, f, i), cfo_hz=-f, full_scale_dbm=fs_dbm, noise=True
            )
            err.append(vc.est_periodogram(y) - f)
            seg = y[n0:n1].astype(np.complex128)
            a = np.vdot(ref, seg) / np.vdot(ref, ref)
            snr_hat.append(2 * abs(a) ** 2 / float(np.mean(np.abs(seg - a * ref) ** 2)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    crb = vc.crb_std_hz(snr_db)
    # the engine's physical noise realises the requested SNR (within 0.2 dB on average)
    assert abs(10 * math.log10(float(np.mean(snr_hat))) - snr_db) < 0.2
    assert rmse <= 1.5 * crb, (rmse, crb, rmse / crb)
