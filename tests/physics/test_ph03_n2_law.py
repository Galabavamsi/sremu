"""PH-03: N^2 law with far-field focusing (Tang 2022 Eq. 8; section 10.2).

N = 100 / 400 / 1600 cells at lambda/2, 2 km, Tx on the normal, Rx at 20 deg, focused:
-169.027 / -156.985 / -144.944 dB (+-0.01 dB), slope 12.04 +- 0.01 dB per 4x. The far-field
closed form is an accelerator only, guarded by the Fraunhofer distance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.errors import GeometryError
from sremu.physics.geometry import PanelGrid, Pose
from sremu.physics.optimize import continuous_optimum
from sremu.physics.quadrature import build_quadrature, resolve_cell_model
from sremu.physics.ris_kernel import accumulate_cells, farfield_focused_amplitude, k1_terms
from sremu.units import C0
from tests.physics._closed_forms import db20

pytestmark = pytest.mark.physics

FC = 3.5e9
D = 2000.0
TH_R = math.radians(20)
SPEC = {100: -169.027, 400: -156.985, 1600: -144.944}  # PH-03 oracle table


def _focused_db(n_side: int) -> tuple[float, float, PanelGrid]:
    lam = C0 / FC
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), n_side, n_side, lam / 2, lam / 2)
    model = resolve_cell_model("auto", lam / 2, lam / 2, lam)
    assert model == "point"
    quad = build_quadrature(grid, model, lam)
    tx = np.array([0.0, 0.0, D])
    rx = np.array([D * math.sin(TH_R), 0.0, D * math.cos(TH_R)])
    c = accumulate_cells(
        k1_terms(tx, rx, quad.points, quad.areas, grid.pose.normal, FC), quad.parent, quad.n_cells
    )[0, 0]
    value, phases = continuous_optimum(0.0, c)
    h = complex(np.sum(c * np.exp(1j * phases)))
    assert abs(abs(h) - value) <= 1e-12 * value
    return db20(h), lam, grid


def test_ph03_n2_law_far_field_focus():
    got = {}
    for n_side in (10, 20, 40):
        n = n_side * n_side
        val, lam, grid = _focused_db(n_side)
        got[n] = val
        # Tang Eq. (8) closed form: |h| = N A sqrt(cos 0 cos 20) / (4 pi d1 d2)
        closed = n * (lam / 2) ** 2 * math.sqrt(math.cos(TH_R)) / (4 * math.pi * D * D)
        assert abs(val - db20(closed)) <= 0.01
        assert abs(val - SPEC[n]) <= 0.01
        acc = farfield_focused_amplitude(
            n, (lam / 2) ** 2, D, D, 0.0, TH_R, FC, panel_diagonal_m=grid.diagonal_m
        )
        assert abs(acc - closed) <= 1e-12 * closed
    assert abs((got[400] - got[100]) - 20 * math.log10(4)) <= 0.01
    assert abs((got[1600] - got[400]) - 20 * math.log10(4)) <= 0.01


def test_ph03_farfield_accelerator_guard():
    lam = C0 / FC
    diag = math.hypot(40 * lam / 2, 40 * lam / 2)
    limit = 2 * diag**2 / lam
    with pytest.raises(GeometryError):
        farfield_focused_amplitude(
            1600, (lam / 2) ** 2, 0.5 * limit, D, 0.0, TH_R, FC, panel_diagonal_m=diag
        )
    farfield_focused_amplitude(
        1600, (lam / 2) ** 2, limit, limit, 0.0, TH_R, FC, panel_diagonal_m=diag
    )
