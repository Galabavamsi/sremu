"""PH-16: phase wrap invariance, ``Gamma e^{j 2 pi m}`` -> ``|dh| <= 1e-15 |h|``.

Phase commands go through the quantizer (pipeline step 2): wrapped phases select exactly the
same states, so the channel is bitwise identical for any m. A direct multiplication of
Gamma by ``e^{j 2 pi m}`` is bounded only by the representation error of that factor
(``|sin(2 pi m)| ~ 2.4e-16 m``), which stays below 1e-15 for m <= 3.
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import StreamSpec
from sremu.errors import OutOfDomain
from sremu.physics.geometry import PanelGrid, Pose, Static
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation
from sremu.physics.response import builtin
from sremu.physics.ris_kernel import k1_terms
from tests.physics._closed_forms import make_panel

pytestmark = pytest.mark.physics

FC = 3.5e9


def test_ph16_wrapped_phase_commands_give_identical_channel(rng: np.random.Generator):
    tx = StreamSpec("ap.tx0", "tx", FC, "1e6")  # type: ignore[arg-type]
    rx = StreamSpec("mu.rx0", "rx", FC, "1e6")  # type: ignore[arg-type]
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 8, 8, 0.3 / 8, 0.3 / 8)
    model = ScenePropagation(
        [
            NodeModel("ap", Static(np.array([0.2, 0.1, 2.0])), streams=(tx,)),
            NodeModel("mu", Static(np.array([-0.5, 0.3, 1.5])), streams=(rx,)),
        ],
        [PanelModel("ris0", grid, builtin("ideal_bbit(3)"))],
    )
    for mdl in (model,):
        snap = mdl.snapshot(Fraction(0))
        pipe = mdl.pipeline("ris0")
        phi = rng.uniform(-math.pi, math.pi, 64)
        h = mdl.evaluate(snap, {"ris0": pipe.from_phases(phi)})["ap.tx0->mu.rx0"]
        for m in (1, -3, 7, 1000, 123456):
            h2 = mdl.evaluate(snap, {"ris0": pipe.from_phases(phi + 2 * math.pi * m)})
            assert abs(h2["ap.tx0->mu.rx0"] - h) <= 1e-15 * abs(h)
    assert model.links() == ("ap.tx0->mu.rx0",)


def test_ph16_out_of_band_table_is_rejected():
    """A 27 GHz table on a 3.5 GHz link is outside its frequency domain (OutOfDomain)."""
    tx = StreamSpec("ap.tx0", "tx", FC, "1e6")  # type: ignore[arg-type]
    rx = StreamSpec("mu.rx0", "rx", FC, "1e6")  # type: ignore[arg-type]
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 8, 8, 0.3 / 8, 0.3 / 8)
    with pytest.raises(OutOfDomain):
        ScenePropagation(
            [
                NodeModel("ap", Static(np.array([0.2, 0.1, 2.0])), streams=(tx,)),
                NodeModel("mu", Static(np.array([-0.5, 0.3, 1.5])), streams=(rx,)),
            ],
            [PanelModel("ris0", grid, builtin("tang2022_ris1_27ghz"))],
        )


def test_ph16_direct_multiplication_bound():
    pos, area = make_panel(0.3, 0.3, 8, 8)
    t = k1_terms([0.2, 0.1, 2.0], [-0.5, 0.3, 1.5], pos, area, np.array([0.0, 0.0, 1.0]), FC)[0]
    g = np.exp(-1j * np.angle(t))  # focusing phases
    h = np.sum(g * t)
    for m in (1, 2, 3):
        h2 = np.sum(g * cmath.rect(1.0, 2 * math.pi * m) * t)
        assert abs(h2 - h) <= 1e-15 * abs(h)
