"""Teeth twins for the legacy kernels (sections 5.14, 10.7): each legacy item fails the
physical guard by the documented amount, and every item refuses to run without
``allow_nonphysical=True``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.errors import NonPhysicalError
from sremu.legacy import kernels as legacy_reexport
from sremu.physics import legacy
from sremu.physics.ris_kernel import k1_channel
from sremu.units import C0
from tests.physics._closed_forms import db20, make_panel

pytestmark = pytest.mark.physics


def test_legacy_items_raise_without_allow_nonphysical():
    pos, area = make_panel(0.2, 0.2, 4, 4)
    tx, rx, n = np.array([0.1, 0, 1.0]), np.array([-0.2, 0.1, 1.5]), np.array([0, 0, 1.0])
    calls = [
        lambda kw: legacy.legacy_v1(tx, rx, pos, 0.04, n, 3.5e9, **kw),
        lambda kw: legacy.legacy_v2(tx, rx, pos, n, 3.5e9, **kw),
        lambda kw: legacy.legacy_wcl_eq2(tx, rx, pos, area, n, 3.5e9, **kw),
        lambda kw: legacy.ellingson_qgamma(tx, rx, pos, n, 3.5e9, **kw),
        lambda kw: legacy.no_j(tx, rx, pos, area, n, 3.5e9, **kw),
        lambda kw: legacy.block_constant_delay(lambda t: t, np.zeros(3), 1e-3, **kw),
    ]
    for call in calls:
        with pytest.raises(NonPhysicalError):
            call({})
        call({"allow_nonphysical": True})
    assert set(legacy_reexport.LEGACY_ITEMS) == set(legacy.LEGACY_ITEMS)
    assert legacy_reexport.legacy_v2 is legacy.legacy_v2


def test_legacy_v1_is_24p4db_above_k1_in_v1_default_geometry():
    """v1 default configuration (T16 of pm_checks): 2.4 GHz, 16 x 16 cells at 6 cm pitch."""
    fc, pitch = 2399996950.0, 0.06
    pos = np.array(
        [
            [0.0, 5.0 + 0.025 + 0.01 + j * pitch, 5.0 + 0.025 + 0.01 + i * pitch]
            for i in range(16)
            for j in range(16)
        ]
    )
    nrm = np.array([1.0, 0.0, 0.0])
    tx, rx = np.array([5.0, 2.0, 5.0]), np.array([15.0, 8.0, 5.0])
    bn = np.exp(1j * math.pi / 4)
    h_k1 = k1_channel(tx, rx, pos, pitch**2, nrm, fc, bn)
    # independent K1 sum for normal +x
    k = 2 * math.pi * fc / C0
    vi, vs = tx - pos, rx - pos
    ri, rs = np.linalg.norm(vi, axis=1), np.linalg.norm(vs, axis=1)
    ci, cs = vi[:, 0] / ri, vs[:, 0] / rs
    ref = np.sum(
        1j
        * bn
        * pitch**2
        * np.sqrt(ci * cs)
        / (4 * math.pi * ri * rs)
        * np.exp(-1j * k * (ri + rs))
    )
    assert abs(h_k1 - ref) <= 1e-12 * abs(ref)
    # oracle table value (data/physics_model_checks.json, T16_v1cfg_RIS_recommended_dB, rounded
    # to 0.01 dB by pm_checks.py; same C0)
    assert abs(db20(h_k1) - (-99.36)) <= 0.005
    a_panel = (16 * pitch) ** 2
    h_v1 = legacy.legacy_v1(tx, rx, pos, a_panel, nrm, fc, allow_nonphysical=True)
    # independent transcription of v1 total_nlos_gain (whole-panel area in every cell)
    qv = legacy.WCL_Q
    p = (
        (qv * a_panel * C0) ** 2
        / (4 * math.pi * fc) ** 2
        * math.pi
        * np.abs(ci) ** (2 * qv)
        * (np.abs(cs) ** (2 * qv))
        / (ri**2 * rs**2)
    )
    v1_ref = np.sum(bn * np.sqrt(p) * np.exp(-2j * math.pi * fc * (ri + rs) / C0))
    assert abs(h_v1 - v1_ref) <= 1e-12 * abs(v1_ref)
    # Teeth: v1 overstates the RIS path by about 24 dB (spec: +24.4 dB with v1's own c = 3e8;
    # with units.C0, the only c allowed, the gap of this non-focused sum is 24.48 dB)
    gap = db20(h_v1) - db20(h_k1)
    assert 24.3 <= gap <= 24.6, gap


@pytest.mark.parametrize(("fc", "spec_db"), [(1e9, -0.52), (2.4e9, -8.1), (3.5e9, -11.4)])
def test_legacy_wcl_eq2_ratio_closed_form(fc: float, spec_db: float, rng: np.random.Generator):
    lam = C0 / fc
    q = legacy.WCL_Q
    area = (lam / 4) ** 2
    nrm = np.array([0.0, 0.0, 1.0])
    for _ in range(20):  # single elements at random geometry: ratio (pi lam)^2 (ci cs)^(2q-1)
        p = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0]])
        tx = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(1, 4)])
        rx = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(1, 4)])
        ci = (tx - p[0])[2] / np.linalg.norm(tx - p[0])
        cs = (rx - p[0])[2] / np.linalg.norm(rx - p[0])
        hw = legacy.legacy_wcl_eq2(tx, rx, p, area, nrm, fc, allow_nonphysical=True)
        hk = k1_channel(tx, rx, p, area, nrm, fc)
        ratio = abs(hw) ** 2 / abs(hk) ** 2
        closed = (math.pi * lam) ** 2 * (ci * cs) ** (2 * q - 1)
        assert abs(ratio / closed - 1) <= 1e-9
    assert abs(10 * math.log10((math.pi * lam) ** 2) - spec_db) <= 0.05  # broadside value


def test_ellingson_qgamma_matches_its_formula():
    fc, q = 3.5e9, 0.285
    lam = C0 / fc
    p = np.array([[0.1, -0.05, 0.0]])
    tx, rx = np.array([-1.0, 0.2, 2.0]), np.array([1.5, 0.1, 1.0])
    ri, rs = np.linalg.norm(tx - p[0]), np.linalg.norm(rx - p[0])
    ci, cs = (tx - p[0])[2] / ri, (rx - p[0])[2] / rs
    gq = 2 * (2 * q + 1)
    closed = (
        (lam / (4 * math.pi)) ** 2
        * math.sqrt(gq * ci ** (2 * q) * gq * cs ** (2 * q))
        / (ri * rs)
        * np.exp(-2j * math.pi * (ri + rs) / lam)
    )
    got = legacy.ellingson_qgamma(tx, rx, p, np.array([0, 0, 1.0]), fc, allow_nonphysical=True)
    assert abs(got - closed) <= 1e-12 * abs(closed)


def test_block_constant_doppler_alias_teeth():
    """T9 geometry: 3.5 GHz, Rx approaching at 30 m/s from 50 m, 10 kS/s, 0.2 s.
    Exact per-sample delays give +v/lambda = +350.242 Hz; the v2 2 ms staircase aliases to
    -148.3 Hz (+-1 Hz)."""
    fc, fs, v = 3.5e9, 10e3, 30.0
    t = np.arange(2000) / fs

    def tau(tt: np.ndarray) -> np.ndarray:
        return np.abs(50.0 - v * tt) / C0

    def mean_freq(tau_s: np.ndarray) -> float:
        y = np.exp(-2j * math.pi * fc * tau_s)
        return float(np.mean(np.angle(y[1:] * np.conj(y[:-1]))) * fs / (2 * math.pi))

    assert abs(mean_freq(tau(t)) - v * fc / C0) <= 0.001
    assert abs(v * fc / C0 - 350.242) <= 0.0005
    staircase = legacy.block_constant_delay(tau, t, 2e-3, allow_nonphysical=True)
    assert abs(mean_freq(staircase) - (-148.3)) <= 1.0
