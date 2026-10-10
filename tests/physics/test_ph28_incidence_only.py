"""PH-28: incidence-only tables through the half-difference extension (section 5.5).

``Gamma(theta_i, theta_s) := g(|theta_i - theta_s| / 2)``: the extended table is exactly
symmetric and its specular entry ``Gamma(theta, -theta)`` equals ``g(|theta|)`` exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from sremu.errors import TableValidationError
from sremu.physics.response import Provenance, ResponseTable

pytestmark = pytest.mark.physics


def _prov(kind: str = "fullwave") -> Provenance:
    return Provenance(
        kind,
        "test",
        "PH-28",
        0.0,
        "co",
        False,
        0.0,
        0.0,
        None,
        "test",  # type: ignore[arg-type]
        {"date": "2026-10-10"},
    )


def _g(theta: np.ndarray) -> np.ndarray:
    # smooth incidence-only response: two states, slowly varying in angle
    s0 = (0.95 - 0.1 * (theta / 90) ** 2) * np.exp(1j * np.radians(150 - 0.3 * theta))
    s1 = (0.9 - 0.05 * (theta / 90)) * np.exp(1j * np.radians(-10 + 0.2 * theta))
    return np.stack([s0, s1])[:, None, :]  # (S, F=1, Tg)


def test_ph28_extension_is_exactly_symmetric_with_exact_specular():
    tg = np.arange(0.0, 81.0, 5.0)
    t = ResponseTable.from_incidence_only(["a", "b"], [3.5e9], tg, _g(tg), _prov())
    gam = t.gamma
    assert np.array_equal(gam, np.swapaxes(gam, 2, 3))  # bitwise symmetric
    th = t.theta_deg
    g = _g(tg)
    for i, x in enumerate(th):
        j = int(np.nonzero(th == -x)[0][0])
        k = int(np.nonzero(tg == abs(x))[0][0])
        assert np.array_equal(gam[:, 0, i, j], g[:, 0, k])  # specular entry == g(|theta|)
    assert t.angle_mapping == "half_difference"
    assert any("half_difference" in a for a in t.provenance.assumptions)
    # queries are reciprocal too
    q1 = t.query(3.5e9, [10.0, -33.3], [47.5, 20.0])
    q2 = t.query(3.5e9, [47.5, 20.0], [10.0, -33.3])
    assert np.max(np.abs(q1 - q2)) <= 1e-15


def test_ph28_state_library_half_difference_and_guard():
    doc = {
        "kind": "ris_state_library",
        "name": "inc_only",
        "provenance": _prov().to_dict(),
        "frequency_hz": [3.5e9],
        "validity_bw_hz": 1e7,
        "angle_grid": {"theta_deg": [0.0, 10.0, 20.0, 30.0]},
        "angle_mapping": "half_difference",
        "reciprocal": True,
        "states": [
            {
                "id": 0,
                "label": "s0",
                "gamma": [
                    {"f_hz": 3.5e9, "theta_deg": th, "mag": 1.0, "phase_deg": 100.0 + th}
                    for th in (0.0, 10.0, 20.0, 30.0)
                ],
            }
        ],
    }
    t = ResponseTable.from_state_library(doc)
    assert np.array_equal(t.gamma, np.swapaxes(t.gamma, 2, 3))
    assert t.theta_deg.tolist() == [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    spec = t.query(3.5e9, [20.0], [-20.0])[0, 0]
    assert abs(spec - np.exp(1j * np.radians(120.0))) <= 1e-15
    bad = dict(doc, angle_mapping="full")
    with pytest.raises(TableValidationError):  # incidence-only data without the extension
        ResponseTable.from_state_library(bad)
    with pytest.raises(TableValidationError):  # raw non-reciprocal tables are rejected
        ResponseTable.from_state_library(dict(doc, reciprocal=False))
