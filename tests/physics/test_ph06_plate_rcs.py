"""PH-06: plate RCS, 1 m^2 at 400 m monostatic: -126.067 dB (+-0.01 dB; section 10.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.physics.geometry import PanelGrid, Pose
from sremu.physics.quadrature import build_quadrature, resolve_cell_model
from sremu.physics.ris_kernel import k1_terms
from sremu.units import C0
from tests.physics._closed_forms import db20

pytestmark = pytest.mark.physics


def test_ph06_plate_rcs_monostatic_400m():
    fc, d, area = 3.5e9, 400.0, 1.0
    lam = C0 / fc
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 40, 40, 1 / 40, 1 / 40)
    model = resolve_cell_model("auto", grid.pitch_u_m, grid.pitch_v_m, lam)
    quad = build_quadrature(grid, model, lam)
    p = np.array([0.0, 0.0, d])
    h = np.sum(k1_terms(p, p, quad.points, quad.areas, grid.pose.normal, fc))
    closed = area / (4 * math.pi * d * d)  # A / (4 pi d^2)
    sigma = 4 * math.pi * area**2 / lam**2  # flat-plate RCS
    radar = math.sqrt(lam**2 * sigma / ((4 * math.pi) ** 3 * d**4))
    assert abs(closed - radar) <= 1e-12 * closed  # the two closed forms agree
    assert abs(db20(h) - db20(closed)) <= 0.01
    assert abs(db20(h) - (-126.067)) <= 0.01
