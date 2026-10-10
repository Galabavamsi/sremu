"""PH-26: point vs po_patch model form (16 x 16 lambda/2 panel, 3.5 GHz, 2 m; section 10.2).

* Specular (uniform Gamma): the two cell models differ by <= 0.05 dB.
* 45 deg retro (focused, Tx = Rx): the difference is the sub-aperture factor of each cell.
  Its closed form, ``prod sinc(p (sin theta_i + sin theta_s) / lambda)`` per axis with the
  local direction cosines of every cell, predicts the engine within 0.1 dB. At 2 m the
  angles vary across the panel and the difference is about 8.45 dB; in the far field
  (200 m) it tends to the single-angle value 20 log10|sinc(sqrt 2 / 2)| = 8.92 dB
  (the "about 8.9 dB" of the spec).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.physics.geometry import PanelGrid, Pose
from sremu.physics.quadrature import build_quadrature
from sremu.physics.ris_kernel import accumulate_cells, k1_terms
from sremu.units import C0

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
P = LAM / 2
GRID = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 16, 16, P, P)


def _cells(model: str, level: int, tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
    q = build_quadrature(GRID, model, LAM, level=level)  # type: ignore[arg-type]
    t = k1_terms(tx, rx, q.points, q.areas, GRID.pose.normal, FC)
    return accumulate_cells(t, q.parent, q.n_cells)[0, 0]


def test_ph26_specular_models_agree():
    th = math.radians(30)
    tx = np.array([-2 * math.sin(th), 0, 2 * math.cos(th)])
    rx = np.array([2 * math.sin(th), 0, 2 * math.cos(th)])
    cp = _cells("point", 0, tx, rx)
    for level in (0, 1, 2):
        cq = _cells("po_patch", level, tx, rx)
        assert abs(20 * math.log10(abs(cp.sum()) / abs(cq.sum()))) <= 0.05


def _local_sinc_prediction(cp: np.ndarray, tx: np.ndarray) -> float:
    pts = GRID.cell_centres()
    d = tx[None, :] - pts
    u = d / np.linalg.norm(d, axis=1)[:, None]
    # retro: sin(theta_i) + sin(theta_s) = 2 * direction cosine, per axis
    fac = np.abs(np.sinc(P * 2 * u[:, 0] / LAM) * np.sinc(P * 2 * u[:, 1] / LAM))
    return 20 * math.log10(np.abs(cp).sum() / (np.abs(cp) * fac).sum())


@pytest.mark.parametrize(("dist", "closed_db"), [(2.0, None), (200.0, 8.92)])
def test_ph26_retro_45deg_difference(dist: float, closed_db: float | None):
    th = math.radians(45)
    tx = np.array([dist * math.sin(th), 0, dist * math.cos(th)])
    cp = _cells("point", 0, tx, tx)
    diffs = [
        20 * math.log10(np.abs(cp).sum() / np.abs(_cells("po_patch", lv, tx, tx)).sum())
        for lv in (3, 4)
    ]
    assert abs(diffs[1] - diffs[0]) <= 0.05  # converged sub-cell quadrature
    pred = _local_sinc_prediction(cp, tx)
    assert abs(diffs[1] - pred) <= 0.1
    if closed_db is not None:
        assert abs(closed_db - (-20 * math.log10(abs(np.sinc(math.sqrt(2) / 2))))) <= 0.005
        assert abs(diffs[1] - closed_db) <= 0.1
    else:
        # REGRESSION value (PH-26 is specified as a recorded regression): 8.45 dB at 2 m
        assert abs(diffs[1] - 8.45) <= 0.1
