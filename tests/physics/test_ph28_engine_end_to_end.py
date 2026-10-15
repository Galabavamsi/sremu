"""PH-28 end to end: the engine feeds outward angles to an incidence-only table.

The table tests check ``Gamma(theta, -theta) == g(|theta|)``; this checks the convention that
connects them to the scene: theta_i (cell to TX) and theta_s (cell to RX) are both outward, so
specular reflection is theta_s = -theta_i. A one-cell panel with a strongly angle-dependent
g, divided by the same scene with a flat table, must give g(30) at 30 deg specular, g(0) at
30 deg retro-reflection and g(20) for normal incidence with a 40 deg exit. Teeth: a sign flip
between the angles and the table query swaps the specular and retro values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.contracts import PanelConfig
from sremu.physics.geometry import PanelGrid, Pose
from sremu.physics.propagation import PanelModel
from sremu.physics.response import ResponseTable
from sremu.units import C0
from tests.physics._wp3_scenes import analytic_provenance, link, row_values, two_node

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
TG = np.arange(0.0, 90.0 + 1e-9, 1.0)


def _g(th: float | np.ndarray) -> complex | np.ndarray:
    return (1 - np.asarray(th) / 100.0) * np.exp(1j * np.radians(3 * np.asarray(th)))


def _gamma_used(th_i: float, th_s: float) -> complex:
    g_tab = np.stack([_g(TG), _g(TG)])[:, None, :]
    tab = ResponseTable.from_incidence_only(["s0", "s1"], [FC], TG, g_tab, analytic_provenance())
    flat = ResponseTable(
        ["s0", "s1"], [FC], [0.0], np.ones((2, 1, 1, 1), complex), analytic_provenance()
    )
    tx = [2 * math.sin(math.radians(th_i)), 0.0, 2 * math.cos(math.radians(th_i))]
    rx = [3 * math.sin(math.radians(th_s)), 0.0, 3 * math.cos(math.radians(th_s))]
    vals = []
    for t in (tab, flat):
        grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 1, 1, LAM / 2, LAM / 2)
        m = two_node(tx, rx, FC, panels=[PanelModel("p", grid, t, cell_model="point")])
        cfg = {"p": PanelConfig("p", np.zeros(1, dtype=np.int16))}
        vals.append(row_values(m, "p", link(), cfg)[0])
    return complex(vals[0] / vals[1])


@pytest.mark.parametrize(
    ("th_i", "th_s", "expect"),
    [(-30.0, 30.0, 30.0), (60.0, -60.0, 60.0), (-30.0, -30.0, 0.0), (0.0, 40.0, 20.0)],
)
def test_incidence_only_table_sees_outward_angles(th_i: float, th_s: float, expect: float) -> None:
    assert abs(_gamma_used(th_i, th_s) - complex(_g(expect))) < 1e-12
