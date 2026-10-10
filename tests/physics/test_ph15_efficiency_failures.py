"""PH-15: efficiency and failures (section 10.2).

* ``eta = 0.5`` scales the RIS term by -3.0103 dB (1e-9), refused on measured tables unless
  overridden.
* 10 % dead cells (absorbing), focused, N = 1024, 200 maps: mean power -0.915 dB (+-0.10 dB)
  relative to the healthy panel; closed form ``10 log10((1-p)^2 + p(1-p)/N)``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PanelConfig, StreamSpec
from sremu.errors import NotPermitted
from sremu.physics.geometry import PanelGrid, Pose, Static
from sremu.physics.nonideal import FailureMap, efficiency_factor
from sremu.physics.optimize import continuous_optimum
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation
from sremu.physics.response import builtin
from sremu.physics.ris_kernel import k1_terms
from sremu.units import C0

pytestmark = pytest.mark.physics

FC = 3.5e9


def _model(table) -> ScenePropagation:  # type: ignore[no-untyped-def]
    tx = StreamSpec("ap.tx0", "tx", FC, "1e6")  # type: ignore[arg-type]
    rx = StreamSpec("mu.rx0", "rx", FC, "1e6")  # type: ignore[arg-type]
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 12, 10, 0.04, 0.04)
    return ScenePropagation(
        [
            NodeModel("ap", Static(np.array([-1.0, 0.0, 2.0])), streams=(tx,)),
            NodeModel("mu", Static(np.array([1.2, 0.3, 1.5])), streams=(rx,)),
        ],
        [PanelModel("ris0", grid, table)],
    )


def test_ph15_efficiency_half_is_minus_3dB_on_ris_term():
    base = builtin("ideal_bbit(2)")
    m1 = _model(base)
    m2 = _model(base.with_efficiency(0.5))
    cfg = {"ris0": PanelConfig("ris0", np.arange(120, dtype=np.int16) % 4)}
    a1 = m1.combine(m1.snapshot(Fraction(0)), cfg)["ap.tx0->mu.rx0"]
    a2 = m2.combine(m2.snapshot(Fraction(0)), cfg)["ap.tx0->mu.rx0"]
    ris = a1.tile >= 0
    ratio_db = 20 * math.log10(abs(a2.amp[ris][0]) / abs(a1.amp[ris][0]))
    assert abs(ratio_db - 10 * math.log10(0.5)) <= 1e-9
    assert a2.amp[~ris][0] == a1.amp[~ris][0]  # LOS untouched
    assert "efficiency" in " ".join(m2.panel("ris0").table.provenance.assumptions)


def test_ph15_efficiency_refused_on_measured_tables_unless_overridden():
    tang = builtin("tang2022_ris1_27ghz")
    with pytest.raises(NotPermitted):
        tang.with_efficiency(0.5)
    with pytest.raises(NotPermitted):
        efficiency_factor(0.5, "digitized")
    t2 = tang.with_efficiency(0.5, override=True)
    assert np.max(np.abs(t2.gamma - tang.gamma * math.sqrt(0.5))) <= 1e-15
    assert efficiency_factor(1.0, "measured") == 1.0  # no-op is allowed


def test_ph15_failure_map_semantics():
    fm = FailureMap(np.array([-1, 1, -1, 0], np.int16), np.array([False, False, True, True]))
    out = fm.apply(np.array([0, 0, 1, 1], np.int16))
    assert out.tolist() == [0, 1, -1, -1]  # stuck-at then dead


def test_ph15_ten_percent_dead_focused_n1024(rng: np.random.Generator):
    lam = C0 / FC
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 32, 32, lam / 2, lam / 2)
    tx = np.array([-3.0, 0.5, 4.0])
    rx = np.array([2.0, -1.0, 3.0])
    c = k1_terms(tx, rx, grid.cell_centres(), grid.cell_area_m2, grid.pose.normal, FC)[0]
    _, phases = continuous_optimum(0.0, c)
    z = c * np.exp(1j * phases)  # focused per-cell contributions
    p_healthy = abs(np.sum(z)) ** 2
    p = 0.1
    powers = []
    for _ in range(200):
        fm = FailureMap.random(1024, p, "dead", rng)
        alive = fm.apply(np.zeros(1024, np.int16)) >= 0
        powers.append(abs(np.sum(z[alive])) ** 2)
    got = 10 * math.log10(np.mean(powers) / p_healthy)
    n = 1024
    closed = 10 * math.log10((1 - p) ** 2 + p * (1 - p) * np.sum(np.abs(z) ** 2) / p_healthy)
    assert abs(got - (-0.915)) <= 0.10
    assert abs(got - closed) <= 0.10
    assert abs(closed - 20 * math.log10(1 - p)) <= 0.01  # mean-field term dominates at N = 1024
    assert n == grid.n_cells
