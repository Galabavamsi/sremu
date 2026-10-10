"""PH-16: phase wrap invariance, ``Gamma e^{j 2 pi m}`` -> ``|dh| <= 1e-15 |h|`` (D8).

D8: phases are reduced modulo one turn, in their own unit, before Gamma is formed.

* Phase commands go through the quantizer (pipeline step 2), which reduces modulo 2 pi:
  wrapped phases select exactly the same states, so the channel is bitwise identical for
  any m.
* Table phases (state libraries, degrees) are reduced modulo 360 deg before Gamma is formed:
  ``phase_deg + 360 m`` gives bitwise the same Gamma, hence the same channel, whenever that
  sum is exact in binary64 (phases on a 1/64 deg grid, any |m| <= 123456 here). Teeth:
  forming Gamma first (radians of the unreduced phase) misses 1e-15 for large m.
* Decimal phases (the lab's 49.64 / -153.77 deg) are themselves rounded; ``+-360`` then
  stays within 1e-15 |h|. Larger |m| carry the rounding of the caller's float sum, which no
  reduction can undo (reported, not a model property).
* A direct multiplication of Gamma by ``e^{j 2 pi m}`` is bounded only by the representation
  error of that factor (``|sin(2 pi m)| ~ 2.4e-16 m``), which stays below 1e-15 for m <= 3.
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PanelConfig, StreamSpec
from sremu.errors import OutOfDomain
from sremu.physics.geometry import PanelGrid, Pose, Static
from sremu.physics.nonideal import gamma_from_phase, wrap_phase_deg, wrap_phase_rad
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation
from sremu.physics.response import ResponseTable, builtin
from sremu.physics.ris_kernel import k1_terms
from tests.physics._closed_forms import make_panel

pytestmark = pytest.mark.physics

FC = 3.5e9
LINK = "ap.tx0->mu.rx0"
GRID = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 8, 8, 0.3 / 8, 0.3 / 8)
DYADIC_DEG = (49.640625, -153.765625)  # the lab phases on a 1/64 deg grid (exact sums)
DECIMAL_DEG = (49.64, -153.77)  # the lab phases as published (section 5.5)


def _model(table: ResponseTable) -> ScenePropagation:
    tx = StreamSpec("ap.tx0", "tx", FC, "1e6")  # type: ignore[arg-type]
    rx = StreamSpec("mu.rx0", "rx", FC, "1e6")  # type: ignore[arg-type]
    return ScenePropagation(
        [
            NodeModel("ap", Static(np.array([0.2, 0.1, 2.0])), streams=(tx,)),
            NodeModel("mu", Static(np.array([-0.5, 0.3, 1.5])), streams=(rx,)),
        ],
        [PanelModel("ris0", GRID, table)],
    )


def _h(model: ScenePropagation, state: np.ndarray) -> complex:
    snap = model.snapshot(Fraction(0))
    return model.evaluate(snap, {"ris0": PanelConfig("ris0", state.astype(np.int16))})[LINK]


def _library(phases_deg: tuple[float, ...], mags: tuple[float, ...] = (1.0, 0.8)) -> dict:
    return {
        "schema_version": "1.0",
        "kind": "ris_state_library",
        "name": "wrap_test",
        "provenance": {
            "source_kind": "analytic",
            "description": "PH-16 wrap test",
            "reference": "test",
            "reference_plane_offset_m": 0.0,
            "polarization": "co",
            "includes_cell_pattern": False,
            "sigma_mag_db": 0.0,
            "sigma_phase_deg": 0.0,
            "raw_sha256": None,
            "license": "test",
            "created": {"date": "2026-10-10", "tool": "pytest"},
        },
        "frequency_hz": [FC],
        "validity_bw_hz": 2e7,
        "angle_grid": "none",
        "reciprocal": True,
        "states": [
            {"id": s, "label": f"s{s}", "gamma": [{"f_hz": FC, "mag": a, "phase_deg": p}]}
            for s, (p, a) in enumerate(zip(phases_deg, mags, strict=True))
        ],
    }


def test_ph16_wrapped_phase_commands_give_identical_channel(rng: np.random.Generator):
    model = _model(builtin("ideal_bbit(3)"))
    snap = model.snapshot(Fraction(0))
    pipe = model.pipeline("ris0")
    phi = rng.uniform(-math.pi, math.pi, 64)
    h = model.evaluate(snap, {"ris0": pipe.from_phases(phi)})[LINK]
    for m in (1, -3, 7, 1000, 123456):
        h2 = model.evaluate(snap, {"ris0": pipe.from_phases(phi + 2 * math.pi * m)})
        assert abs(h2[LINK] - h) <= 1e-15 * abs(h)
    assert model.links() == (LINK,)


@pytest.mark.parametrize("m", [1, -3, 7, 1000, 123456])
def test_ph16_d8_table_phases_wrapped_by_whole_turns(m: int, rng: np.random.Generator):
    base = ResponseTable.from_state_library(_library(DYADIC_DEG))
    wrapped = ResponseTable.from_state_library(_library(tuple(p + 360.0 * m for p in DYADIC_DEG)))
    assert np.array_equal(wrapped.gamma, base.gamma)  # reduced before Gamma is formed
    state = rng.integers(0, 2, 64)
    h0 = _h(_model(base), state)
    h1 = _h(_model(wrapped), state)
    assert abs(h1 - h0) <= 1e-15 * abs(h0)
    # teeth: forming Gamma first (radians of the unreduced phase) is not exact for large m
    if abs(m) >= 1000:
        g_first = np.exp(1j * np.radians(DYADIC_DEG[0] + 360.0 * m))
        assert abs(g_first - base.gamma[0, 0, 0, 0]) > 1e-15


@pytest.mark.parametrize("m", [1, -1])
def test_ph16_d8_decimal_table_phases_one_turn(m: int, rng: np.random.Generator):
    base = ResponseTable.from_state_library(_library(DECIMAL_DEG))
    wrapped = ResponseTable.from_state_library(_library(tuple(p + 360.0 * m for p in DECIMAL_DEG)))
    state = rng.integers(0, 2, 64)
    h0 = _h(_model(base), state)
    h1 = _h(_model(wrapped), state)
    assert abs(h1 - h0) <= 1e-15 * abs(h0)


def test_ph16_d8_wrap_helpers():
    """The reductions are exact fmods into [0, one turn); Gamma is formed from the reduced
    phase (equal to forming it from an in-range phase, bitwise)."""
    x = np.array([-7.5, -1e-300, 0.0, 1.0, 2 * math.pi, 1e6, -123456.75])
    w = wrap_phase_rad(x)
    assert np.all((w >= 0) & (w < 2 * math.pi))
    # exact fmod; a tiny negative phase whose remainder rounds up to one full turn maps to 0
    assert np.remainder(-1e-300, 2 * math.pi) == 2 * math.pi and w[1] == 0.0
    keep = np.arange(x.size) != 1
    assert np.array_equal(w[keep], np.remainder(x, 2 * math.pi)[keep])
    d = np.array([-720.0, -153.765625, 0.0, 359.5, 360.0, 44444209.640625, -1e-300])
    wd = wrap_phase_deg(d)
    assert wd.tolist() == [0.0, 206.234375, 0.0, 359.5, 0.0, 49.640625, 0.0]  # exact
    phi = np.array([0.25, 1.5, 3.0, 6.0])
    assert np.array_equal(gamma_from_phase(phi, 0.7), 0.7 * (np.cos(phi) + 1j * np.sin(phi)))
    assert np.array_equal(
        gamma_from_phase(phi - 2 * math.pi), gamma_from_phase(wrap_phase_rad(phi - 2 * math.pi))
    )


def test_ph16_out_of_band_table_is_rejected():
    """A 27 GHz table on a 3.5 GHz link is outside its frequency domain (OutOfDomain)."""
    with pytest.raises(OutOfDomain):
        _model(builtin("tang2022_ris1_27ghz"))


def test_ph16_direct_multiplication_bound():
    pos, area = make_panel(0.3, 0.3, 8, 8)
    t = k1_terms([0.2, 0.1, 2.0], [-0.5, 0.3, 1.5], pos, area, np.array([0.0, 0.0, 1.0]), FC)[0]
    g = np.exp(-1j * np.angle(t))  # focusing phases
    h = np.sum(g * t)
    for m in (1, 2, 3):
        h2 = np.sum(g * cmath.rect(1.0, 2 * math.pi * m) * t)
        assert abs(h2 - h) <= 1e-15 * abs(h)
