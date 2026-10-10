"""K1 oracle vs Tang et al. 2022 (arXiv:2101.08607) Fig. 10 published model curves (DIG).

Tang's refined model Eq. (9)/(12) is K1 (``cell_model="point"``) without the Huygens j, with
horn patterns cos^53.8 aimed at the RIS centre, mmRIS1 (56 x 20 cells, 1.4 x 2.8 mm) at
27 GHz, coding state '0' (0.9 at 165 deg), P_t = 20 dBm and the measured G_t*G_r*G_line =
24.0 dB (Table IV). The K1 oracle includes the horn peak gains G_0 = 2(alpha+1) in sqrt(G_T
G_R), so they are removed before adding Tang's measured calibration. This anchors the oracle
to a published third-party curve (V-TANG-1 acceptance: 0.05 dB).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.oracles import rf
from tests.oracles._data import tang_fig10
from tests.oracles.ris import cos_power, k1_h, make_panel

pytestmark = pytest.mark.physics

FC = 27e9
ALPHA = 53.8
GAMMA0 = 0.9 * np.exp(1j * math.radians(165.0))
PT_DBM = 20.0
CAL_DB = 24.0
PANELS = {"a": (1.0, 10.0), "b": (2.0, 10.0), "c": (1.0, 45.0), "d": (2.0, 45.0)}


def tang_received_power_dbm(d1: float, d2: float, theta_deg: float) -> float:
    """Tang's specular setup: Tx at d1, Rx at d2, both at theta from the normal."""
    th = math.radians(theta_deg)
    tx = np.array([-d1 * math.sin(th), 0.0, d1 * math.cos(th)])
    rx = np.array([d2 * math.sin(th), 0.0, d2 * math.cos(th)])
    p = make_panel([0, 0, 0], [0, 0, 1], [1, 0, 0], 56, 20, 1.4e-3, 2.8e-3)
    h = k1_h(tx, rx, p, FC, GAMMA0, g_t=cos_power(ALPHA, -tx), g_r=cos_power(ALPHA, -rx))
    g0_db = 20.0 * math.log10(2.0 * (ALPHA + 1.0))  # both horns' peak gains (power, dB)
    return PT_DBM + CAL_DB - g0_db + float(rf.db20(h))


@pytest.mark.parametrize("panel", list(PANELS))
def test_vtang1_k1_oracle_reproduces_tang_fig10_model_curves(panel):
    d1, theta = PANELS[panel]
    curve = tang_fig10()["data"][panel]["refined_eq9"]
    assert len(curve) == 9
    for d2_dig, p_dig in curve:
        d2 = round(d2_dig * 2) / 2  # digitized abscissae sit at the nominal 0.5 m grid (+2 mm)
        assert abs(d2_dig - d2) <= 0.005
        assert abs(tang_received_power_dbm(d1, d2, theta) - p_dig) <= 0.05
