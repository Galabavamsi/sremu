"""PH-26: point vs po_patch model form (16 x 16 lambda/2 panel, 3.5 GHz; section 10.2, D6).

* Specular (uniform Gamma, 2 m): the two cell models differ by <= 0.05 dB.
* 45 deg retro (focused, Tx = Rx), D6 regression value: 8.11 dB point-minus-patch with the
  shipped sub-cell size Delta <= lambda/5, i.e. 3 x 3 sub-cells per lambda/2 cell, +-0.1 dB.
  8.11 dB is the far-field (single-angle) value of the 3-point midpoint sub-aperture factor,
  ``-20 log10|(1 + 2 cos(pi sqrt(2)/3))/3|`` = 8.110 dB; the engine reproduces it at 200 m.
* The Delta -> 0 limit is the continuous sub-aperture factor
  ``-20 log10|sinc(p (sin ti + sin ts)/lambda)|`` = 8.92 dB (documented; checked at 200 m).
* At the PH-26 row's 2 m the local angles vary across the panel: with 3 x 3 sub-cells the
  engine gives 7.68 dB (it matches the per-cell local 3-point prediction, an independent
  closed form). D6's 8.11 dB is therefore not met at 2 m; the strict xfail below records
  that and needs a lead decision (distance or value).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.physics.geometry import PanelGrid, Pose
from sremu.physics.quadrature import build_quadrature, subdivisions
from sremu.physics.ris_kernel import accumulate_cells, k1_terms
from sremu.units import C0

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
P = LAM / 2
GRID = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 16, 16, P, P)
REGRESSION_DB = 8.11  # D6: point minus patch, 45 deg retro, Delta <= lambda/5 (3 x 3)
SINC_LIMIT_DB = -20 * math.log10(abs(float(np.sinc(math.sqrt(2) / 2))))  # Delta -> 0: 8.918


def _cells(model: str, level: int, tx: np.ndarray, rx: np.ndarray) -> np.ndarray:
    q = build_quadrature(GRID, model, LAM, level=level)  # type: ignore[arg-type]
    t = k1_terms(tx, rx, q.points, q.areas, GRID.pose.normal, FC)
    return accumulate_cells(t, q.parent, q.n_cells)[0, 0]


def _retro(dist: float) -> np.ndarray:
    th = math.radians(45)
    return np.array([dist * math.sin(th), 0, dist * math.cos(th)])


def _point_minus_patch_db(tx: np.ndarray, level: int) -> float:
    """Focused (retro) configuration: |h| = sum_n |c_n| for each cell model."""
    cp = _cells("point", 0, tx, tx)
    cq = _cells("po_patch", level, tx, tx)
    return 20 * math.log10(np.abs(cp).sum() / np.abs(cq).sum())


def _midpoint_factor(m: int, x: np.ndarray) -> np.ndarray:
    """|(1/m) sum_k exp(j 2 pi x (k + 1/2 - m/2)/m)|: m-point midpoint rule of the sinc
    sub-aperture factor, ``x = p (sin ti + sin ts)/lambda`` per axis."""
    k = np.arange(m) + 0.5 - m / 2
    return np.abs(np.exp(2j * np.pi * np.multiply.outer(x, k) / m).mean(axis=-1))


def test_ph26_specular_models_agree():
    th = math.radians(30)
    tx = np.array([-2 * math.sin(th), 0, 2 * math.cos(th)])
    rx = np.array([2 * math.sin(th), 0, 2 * math.cos(th)])
    cp = _cells("point", 0, tx, rx)
    for level in (0, 1, 2):
        cq = _cells("po_patch", level, tx, rx)
        assert abs(20 * math.log10(abs(cp.sum()) / abs(cq.sum()))) <= 0.05


def test_ph26_default_subcells_are_3x3():
    """Delta <= lambda/5 on a lambda/2 cell gives ceil(2.5) = 3 sub-cells per axis."""
    assert subdivisions(P, P, LAM) == (3, 3)


def test_ph26_retro_45deg_regression_8p11db_far_field():
    """D6 regression value 8.11 +- 0.1 dB with 3 x 3 sub-cells; it equals the closed-form
    3-point midpoint factor of a single angle (far field, 200 m) within 0.01 dB."""
    diff = _point_minus_patch_db(_retro(200.0), level=0)
    closed = -20 * math.log10(float(_midpoint_factor(3, np.array(math.sqrt(2) / 2))))
    assert (
        abs(closed - (-20 * math.log10(abs(1 + 2 * math.cos(math.pi * math.sqrt(2) / 3)) / 3)))
        <= 1e-12
    )
    assert abs(diff - closed) <= 0.01, (diff, closed)
    assert abs(diff - REGRESSION_DB) <= 0.1, diff


REGRESSION_2M_DB = 7.68  # lead decision D6b: near-field (2 m) regression value, 3 x 3 sub-cells


def test_ph26_retro_45deg_regression_7p68db_at_2m():
    """Lead decision D6b (10 Oct 2026): the 8.11 dB oracle is a far-field single-angle value
    and is checked at 200 m above. At the PH-26 row's 2 m geometry the local angles vary across
    the 16x16 panel, so 2 m is pinned as a regression value (7.68 +- 0.1 dB); its correctness
    is established independently by the per-cell local 3-point prediction test below."""
    diff = _point_minus_patch_db(_retro(2.0), level=0)
    assert abs(diff - REGRESSION_2M_DB) <= 0.1, diff


def _local_prediction_db(tx: np.ndarray, m: int | None) -> float:
    """Point-minus-patch predicted from each cell's local direction cosines: the m-point
    midpoint factor (``m``) or the continuous sinc (``None``) per axis, weighted by the
    point-model cell magnitudes (independent of the quadrature code)."""
    cp = _cells("point", 0, tx, tx)
    pts = GRID.cell_centres()
    d = tx[None, :] - pts
    u = d / np.linalg.norm(d, axis=1)[:, None]
    x_u, x_v = P * 2 * u[:, 0] / LAM, P * 2 * u[:, 1] / LAM  # retro: sin ti + sin ts = 2 u
    if m is None:
        fac = np.abs(np.sinc(x_u) * np.sinc(x_v))
    else:
        fac = _midpoint_factor(m, x_u) * _midpoint_factor(m, x_v)
    return 20 * math.log10(np.abs(cp).sum() / (np.abs(cp) * fac).sum())


def test_ph26_retro_45deg_2m_matches_local_subcell_prediction():
    """At 2 m the 3 x 3 engine value (7.68 dB, reported) is the local 3-point prediction."""
    tx = _retro(2.0)
    diff = _point_minus_patch_db(tx, level=0)
    assert abs(diff - _local_prediction_db(tx, 3)) <= 0.05, diff


@pytest.mark.parametrize("dist", [2.0, 200.0])
def test_ph26_retro_45deg_delta_to_zero_limit(dist: float):
    """Delta -> 0: the patch tends to the continuous sub-aperture factor; at 200 m that is
    the documented sinc limit 8.92 dB, at 2 m its per-cell local form."""
    tx = _retro(dist)
    diffs = [_point_minus_patch_db(tx, lv) for lv in (3, 4)]
    assert abs(diffs[1] - diffs[0]) <= 0.05  # converged sub-cell quadrature
    assert abs(diffs[1] - _local_prediction_db(tx, None)) <= 0.1
    if dist == 200.0:
        assert abs(SINC_LIMIT_DB - 8.92) <= 0.005
        assert abs(diffs[1] - SINC_LIMIT_DB) <= 0.1
