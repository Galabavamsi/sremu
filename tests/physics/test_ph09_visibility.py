"""PH-09: visibility. A Tx (or Rx) behind the panel gives a RIS term of exactly 0."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PanelConfig, StreamSpec
from sremu.physics.geometry import PanelGrid, Pose, Static
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation
from sremu.physics.response import builtin
from sremu.physics.ris_kernel import k1_channel, k1_terms
from tests.physics._closed_forms import make_panel

pytestmark = pytest.mark.physics

FC = 3.5e9
NRM = np.array([0.0, 0.0, 1.0])


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_ph09_tx_or_rx_behind_panel_is_exactly_zero(backend: str):
    pos, area = make_panel(0.4, 0.3, 10, 7)
    front = np.array([0.3, 0.1, 2.0])
    back = np.array([-0.2, 0.4, -1.5])
    assert k1_channel(back, front, pos, area, NRM, FC, 1.0, backend=backend) == 0j
    assert k1_channel(front, back, pos, area, NRM, FC, 1.0, backend=backend) == 0j
    t = k1_terms(back, front, pos, area, NRM, FC, backend=backend)
    assert np.all(t == 0)
    # grazing: a node in the panel plane is not in front (cos theta = 0) -> exactly 0
    graze = np.array([3.0, 0.0, 0.0])
    assert k1_channel(graze, front, pos, area, NRM, FC, 1.0, backend=backend) == 0j


def test_ph09_partial_visibility_counts_only_front_cells():
    # node level with the plate, offset sideways: half the cells see it from the front side
    pos, area = make_panel(1.0, 0.2, 20, 4)
    pos = pos.copy()
    pos[:, 2] = 0.05 * np.sign(pos[:, 0])  # a two-level (stepped) surface, same normal
    tx = np.array([0.0, 0.0, 0.0])  # below the right half, above the left half
    rx = np.array([0.1, 0.0, 3.0])
    t = k1_terms(tx, rx, pos, area, NRM, FC)[0]
    left = pos[:, 0] < 0
    assert np.all(t[~left] == 0)  # cells above the Tx see it from behind
    assert np.all(t[left] != 0)


def test_ph09_through_propagation_snapshot_and_combine():
    tx = StreamSpec("ap.tx0", "tx", FC, "1e6")  # type: ignore[arg-type]
    rx = StreamSpec("mu.rx0", "rx", FC, "1e6")  # type: ignore[arg-type]
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 8, 6, 0.04, 0.04)
    model = ScenePropagation(
        [
            NodeModel("ap", Static(np.array([0.5, 0.0, -2.0])), streams=(tx,)),
            NodeModel("mu", Static(np.array([-0.5, 0.2, 2.0])), streams=(rx,)),
        ],
        [PanelModel("ris0", grid, builtin("ideal_antipodal"))],
    )
    snap = model.snapshot(Fraction(0))
    rc = snap.ris["ris0"]
    assert rc.C is not None and np.all(rc.C == 0)
    lp = model.combine(snap, {"ris0": PanelConfig("ris0", np.ones(48, dtype=np.int16))})
    tiles = lp["ap.tx0->mu.rx0"]
    assert np.all(tiles.amp[tiles.tile >= 0] == 0)
