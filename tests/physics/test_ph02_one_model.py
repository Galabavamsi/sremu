"""PH-02: one model, three equivalent forms for ``cell_model="point"`` (section 5.3).

K1 = Tang 2022 Eq. (12) (single-cell form, times the Huygens ``j``) = ETSI GR RIS 003
section 6.3.1.2 with ``G = 4 pi dx dy / lambda^2`` = Ellingson Eq. (10) with the
area-tied element gain ``G_e(psi) = (4 pi A / lambda^2) cos psi``; rel <= 1e-12.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.physics.antennas import CosPower
from sremu.physics.ris_kernel import k1_terms
from sremu.units import C0

pytestmark = pytest.mark.physics


def _geometry(rng: np.random.Generator):
    pos = np.column_stack([rng.uniform(-0.3, 0.3, 50), rng.uniform(-0.2, 0.2, 50), np.zeros(50)])
    tx = np.array([rng.uniform(-3, -1), rng.uniform(-1, 1), rng.uniform(1, 4)])
    rx = np.array([rng.uniform(1, 3), rng.uniform(-1, 1), rng.uniform(1, 4)])
    return pos, tx, rx


@pytest.mark.parametrize("fc_hz", [3.5e9, 27e9])
def test_ph02_k1_equals_tang_etsi_ellingson(fc_hz: float, rng: np.random.Generator):
    lam = C0 / fc_hz
    k = 2 * math.pi / lam
    dx, dy = lam / 2, lam / 3
    area = dx * dy
    for _ in range(5):
        pos, tx, rx = _geometry(rng)
        ant_t = CosPower(13.0, (np.zeros(3) - tx) / np.linalg.norm(tx))
        ant_r = CosPower(5.0, (np.zeros(3) - rx) / np.linalg.norm(rx))
        vi, vs = tx - pos, rx - pos
        rt, rr = np.linalg.norm(vi, axis=1), np.linalg.norm(vs, axis=1)
        ct, cr = vi[:, 2] / rt, vs[:, 2] / rr
        gt = ant_t.gain(-vi / rt[:, None])
        gr = ant_r.gain(-vs / rr[:, None])
        ph = np.exp(-1j * k * (rt + rr))
        # Tang 2022 Eq. (12): P_R/P_T = G_t G_r G dx dy lambda^2 F / (64 pi^3 r_t^2 r_r^2),
        # G = 4 pi dx dy / lambda^2, F = cos(theta_t) cos(theta_r) (cell power pattern)
        g_cell = 4 * math.pi * dx * dy / lam**2
        tang = (
            1j
            * np.sqrt(gt * gr * g_cell * dx * dy * lam**2 * ct * cr / (64 * math.pi**3))
            / (rt * rr)
            * ph
        )
        # ETSI GR RIS 003 6.3.1.2: sqrt(G_T G_R) G_cell lambda^2 sqrt(F) / ((4 pi)^2 r_t r_r)
        etsi = (
            1j
            * np.sqrt(gt * gr)
            * g_cell
            * lam**2
            * np.sqrt(ct * cr)
            / ((4 * math.pi) ** 2 * rt * rr)
            * ph
        )
        # Ellingson Eq. (10): (lambda/4pi)^2 sqrt(G_e(psi_i) G_e(psi_s)) / (r_i r_s)
        ge_i = 4 * math.pi * area / lam**2 * ct
        ge_s = 4 * math.pi * area / lam**2 * cr
        ell = (
            1j
            * np.sqrt(gt * gr)
            * (lam / (4 * math.pi)) ** 2
            * np.sqrt(ge_i * ge_s)
            / (rt * rr)
            * ph
        )
        k1 = k1_terms(
            tx,
            rx,
            pos,
            np.full(50, area),
            np.array([0.0, 0.0, 1.0]),
            fc_hz,
            gain_tx=gt[None, :],
            gain_rx=gr[None, :],
        )[0]
        for other in (tang, etsi, ell):
            assert np.max(np.abs(k1 - other) / np.abs(other)) <= 1e-12
