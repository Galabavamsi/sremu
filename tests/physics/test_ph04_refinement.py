"""PH-04: fixed-aperture refinement under the auto cell-model policy (section 10.2).

1 m^2 plate, 3.5 GHz, Gamma = 1, Tx at 30 deg incidence.
* Far specular 200 m: -115.280 dB for every N in {16 ... 16384} (+-0.01 dB) under ``auto``.
* Near anomalous 5 m (Rx at -10 deg): -88.10 dB with ``po_patch``, Delta <= lambda/5 and the
  Richardson check (+-0.10 dB).
* ``auto`` picks ``po_patch`` exactly when the pitch exceeds lambda/2.
* Teeth: ``legacy_v2`` grows by +12.04 +- 0.01 dB per 4x refinement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.errors import RefinementNotConverged, ValidityWarning
from sremu.physics.geometry import PanelGrid, Pose
from sremu.physics.legacy import legacy_v2
from sremu.physics.quadrature import (
    LinkSample,
    build_quadrature,
    resolve_cell_model,
    richardson_check,
)
from sremu.physics.ris_kernel import k1_terms
from sremu.units import C0
from tests.physics._closed_forms import db20

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
INC = math.radians(30)
POSE = Pose([0, 0, 0], [0, 0, 1], [1, 0, 0])
SIDES = (4, 8, 16, 32, 64, 128)


def _geom(d1: float, d2: float, out_deg: float) -> tuple[np.ndarray, np.ndarray]:
    o = math.radians(out_deg)
    return (
        np.array([-d1 * math.sin(INC), 0, d1 * math.cos(INC)]),
        np.array([d2 * math.sin(o), 0, d2 * math.cos(o)]),
    )


def _h_db(n_side: int, tx: np.ndarray, rx: np.ndarray, requested: str) -> tuple[float, str]:
    grid = PanelGrid(POSE, n_side, n_side, 1.0 / n_side, 1.0 / n_side)
    model = resolve_cell_model(requested, grid.pitch_u_m, grid.pitch_v_m, LAM)
    level = 0
    if model == "po_patch":
        level = richardson_check(grid, LAM, [LinkSample("l", tx, rx, FC)]).level
    quad = build_quadrature(grid, model, LAM, level=level)
    return db20(np.sum(k1_terms(tx, rx, quad.points, quad.areas, POSE.normal, FC))), model


def test_ph04_far_specular_200m_independent_of_n():
    tx, rx = _geom(200.0, 200.0, 30.0)
    # far-field plate closed form: A cos(30) / (4 pi d1 d2) (stationary phase over the plate)
    closed = db20(1.0 * math.cos(INC) / (4 * math.pi * 200.0 * 200.0))
    for n_side in SIDES:
        val, model = _h_db(n_side, tx, rx, "auto")
        assert model == ("po_patch" if 1.0 / n_side > LAM / 2 else "point")
        assert abs(val - (-115.280)) <= 0.01, (n_side, val)
        assert abs(val - closed) <= 0.01, (n_side, val, closed)


@pytest.mark.parametrize("n_side", [4, 8, 16, 32, 64])
def test_ph04_near_anomalous_5m_po_patch(n_side: int):
    tx, rx = _geom(5.0, 5.0, -10.0)
    val, model = _h_db(n_side, tx, rx, "po_patch")
    assert model == "po_patch"
    assert abs(val - (-88.10)) <= 0.10, (n_side, val)


def test_ph04_auto_picks_po_patch_exactly_above_half_wavelength():
    half = LAM / 2
    assert resolve_cell_model("auto", half, half, LAM) == "point"  # equal -> point
    assert resolve_cell_model("auto", half * (1 + 1e-12), half, LAM) == "po_patch"
    assert resolve_cell_model("auto", half * (1 - 1e-12), half * 0.5, LAM) == "point"
    assert resolve_cell_model("auto", 0.3 * LAM, 0.51 * LAM, LAM) == "po_patch"
    with pytest.warns(ValidityWarning):
        assert resolve_cell_model("point", LAM, LAM, LAM) == "point"
    assert resolve_cell_model("auto", LAM, LAM, LAM, includes_cell_pattern=True) == "point"


def test_ph04_teeth_legacy_v2_grows_12db_per_4x():
    tx, rx = _geom(200.0, 200.0, 30.0)
    vals = []
    for n_side in SIDES:
        grid = PanelGrid(POSE, n_side, n_side, 1.0 / n_side, 1.0 / n_side)
        vals.append(
            db20(legacy_v2(tx, rx, grid.cell_centres(), POSE.normal, FC, allow_nonphysical=True))
        )
    steps = np.diff(vals)
    assert np.all(np.abs(steps - 20 * math.log10(4)) <= 0.01), steps
    # and the legacy kernel fails the PH-04 guard that K1 passes
    assert all(abs(v - (-115.280)) > 0.01 for v in vals)


def test_ph04_richardson_raises_when_not_converged_and_is_recorded():
    """A 45 deg monostatic (retro) link on lambda/2 cells forced to po_patch does not converge
    within two halvings from lambda/5 at 0.05 dB: RefinementNotConverged (section 5.4)."""
    grid = PanelGrid(POSE, 16, 16, LAM / 2, LAM / 2)
    p = np.array([2 * math.sin(math.radians(45)), 0.0, 2 * math.cos(math.radians(45))])
    with pytest.raises(RefinementNotConverged):
        richardson_check(grid, LAM, [LinkSample("retro45", p, p, FC)])
    # a benign geometry converges and records its estimate and history
    res = richardson_check(grid, LAM, [LinkSample("ok", *_geom(5.0, 5.0, 30.0), FC)])
    assert res.estimate_db <= 0.05 and res.history_db[-1] == res.estimate_db
    assert res.links == ("ok",) and 0 <= res.level <= 2
