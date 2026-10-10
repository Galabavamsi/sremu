"""PH-07: near-field focusing saturation and passivity (section 10.2).

Monostatic focus at d = 1 m: a square plate of side L focused cell by cell gives
``|h| = Omega/(4 pi)`` (solid angle): L = 1 m -> -23.864 dB (+-0.01 dB). L -> inf: 1/2
(-6.021 dB). ``|h| <= 0.5`` for 10^3 random passive configurations (exact bound).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.physics.optimize import continuous_optimum
from sremu.physics.ris_kernel import k1_terms
from sremu.units import C0
from tests.physics._closed_forms import db20, make_panel, rect_solid_angle_onaxis

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
NRM = np.array([0.0, 0.0, 1.0])


@pytest.mark.parametrize(("side", "spec_db"), [(1.0, -23.864), (0.5, None), (2.0, None)])
def test_ph07_focused_plate_equals_solid_angle(side: float, spec_db: float | None):
    d = 1.0
    ns = math.ceil(side / (LAM / 4))
    pos, area = make_panel(side, side, ns, ns)
    p = np.array([0.0, 0.0, d])
    c = k1_terms(p, p, pos, area, NRM, FC)[0]
    value, _ = continuous_optimum(0.0, c)
    exact = rect_solid_angle_onaxis(side / 2, side / 2, d) / (4 * math.pi)
    assert abs(db20(value) - db20(exact)) <= 0.01
    if spec_db is not None:
        assert abs(db20(value) - spec_db) <= 0.01


def test_ph07_infinite_plate_limit_is_one_half():
    # closed form: Omega -> 2 pi for an infinite plate, |h| -> 1/2 = -6.021 dB
    assert abs(db20(rect_solid_angle_onaxis(1e7, 1e7, 1.0) / (4 * math.pi)) - (-6.021)) <= 0.001


def test_ph07_passivity_bound_random_configurations(rng: np.random.Generator):
    # the focused configuration is the maximum over all passive configurations
    for d, side in ((1.0, 4.0), (0.3, 2.0)):
        ns = math.ceil(side / (LAM / 4))
        pos, area = make_panel(side, side, ns, ns)
        p = np.array([0.0, 0.0, d])
        c = k1_terms(p, p, pos, area, NRM, FC)[0]
        best, _ = continuous_optimum(0.0, c)
        assert best <= 0.5
    # 10^3 random passive configurations (amplitudes in [0, 1], uniform phases), d = 0.3 m
    for _batch in range(10):
        mags = rng.uniform(0.0, 1.0, (100, c.size))
        phases = rng.uniform(0.0, 2 * math.pi, (100, c.size))
        h = np.abs((mags * np.exp(1j * phases)) @ c)
        assert np.all(h <= 0.5)
        assert np.all(h <= best * (1 + 1e-12))
