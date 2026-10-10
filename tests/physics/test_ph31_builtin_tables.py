"""PH-31 (built-ins): the packaged state libraries are byte-identical copies of WP1's
``data/states`` (the source of truth), and both load paths give identical tables."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sremu.physics import response
from sremu.physics.response import BUILTIN_DIR, builtin

pytestmark = pytest.mark.physics

REPO = Path(__file__).resolve().parents[2]
WP1_STATES = REPO / "data" / "states"

REFS = [
    "ideal_antipodal",
    "on_off",
    "tang2022_ris1_27ghz",
    "tang2022_ris2_33ghz",
    "ideal_bbit(b=3)",
    "ideal_bbit(2)",
    "abeywickrama(b=2, beta_min=0.2, phi_deg=30, alpha=1.6)",
]


@pytest.mark.skipif(not WP1_STATES.is_dir(), reason="WP1 data/states not present")
def test_ph31_packaged_tables_equal_wp1_state_libraries():
    wp1 = sorted(p.name for p in WP1_STATES.glob("*.yaml"))
    ours = sorted(p.name for p in BUILTIN_DIR.glob("*.yaml"))
    assert wp1 == ours
    for name in wp1:
        assert (WP1_STATES / name).read_bytes() == (BUILTIN_DIR / name).read_bytes(), name


@pytest.mark.parametrize("ref", REFS)
def test_ph31_wp1_and_packaged_paths_agree(ref: str, monkeypatch: pytest.MonkeyPatch):
    a = builtin(ref)
    monkeypatch.setattr(response, "_wp1_states_dir", lambda: None)  # force the fallback
    b = builtin(ref)
    assert a.labels == b.labels and a.name == b.name
    assert np.array_equal(a.gamma, b.gamma)
    assert a.frequency_independent == b.frequency_independent
    assert a.validity_bw_hz == b.validity_bw_hz


def test_ph31_builtin_values_match_their_definitions():
    g = builtin("ideal_bbit(b=2)").gamma_at(1e9)
    assert g.tolist() == [1, 1j, -1, -1j]  # exact quadrants
    assert builtin("ideal_antipodal").gamma_at(5e9).tolist() == [1, -1]
    assert builtin("on_off").gamma_at(1e9).tolist() == [0, 1]
    t = builtin("tang2022_ris1_27ghz").gamma_at(27e9)
    assert abs(t[0] - 0.9 * np.exp(1j * math.radians(165))) <= 1e-15 and t[1] == 0.7
    lab = builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1.0, "sigma2": 1.0}).gamma_at(3.5e9)
    assert np.allclose(np.degrees(np.angle(lab)), [49.64, -153.77], atol=1e-12, rtol=0)
