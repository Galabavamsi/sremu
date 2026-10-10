"""PH-31 (D16): ``ResponseTable.from_state_library`` accepts ``angle_mapping: None`` and
per-entry sigma values (section 5.5 "sigma_phase_deg (scalar or per entry)", section 9.4).

* A WP1 ``StateLibrary`` of a full (theta_i, theta_s) table dumps ``angle_mapping: None``;
  it loads as a full table, entry by entry. ``incidence_only`` documents must name
  ``half_difference``.
* Per-entry ``sigma_mag_db`` / ``sigma_phase_deg`` override the provenance scalars in the
  validators: passivity bound ``1 + 3 (ln10/20) sigma_mag_db`` and reciprocity tolerance
  ``3 sqrt(2) |Gamma| sqrt((ln10/20 sigma_mag)^2 + (pi/180 sigma_phase)^2)`` per entry
  (teeth: the same tables without the entry sigmas are rejected).
* Half-difference tables extend entry sigmas with the Gamma weights; specular entries keep
  g's sigma at |theta|.
* ``to_state_library`` writes entry sigmas; the result validates against the WP1 schema and
  loads back to the same arrays.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from sremu import rng as sremu_rng
from sremu.errors import PassivityError, ReciprocityError, TableValidationError
from sremu.physics.response import ResponseTable

pytestmark = pytest.mark.physics

FC = 3.5e9
LN10_20 = math.log(10.0) / 20.0


def _prov(kind: str = "measured", sm: float = 0.0, sp: float = 0.0) -> dict:
    return {
        "source_kind": kind,
        "description": "D16 test table",
        "reference": "test",
        "reference_plane_offset_m": 0.0,
        "polarization": "co",
        "includes_cell_pattern": False,
        "sigma_mag_db": sm,
        "sigma_phase_deg": sp,
        "raw_sha256": None,
        "license": "test",
        "created": {"date": "2026-10-10", "tool": "pytest"},
        "assumptions": [],
    }


def _doc(states: list[dict], *, grid: object = "none", **extra: object) -> dict:
    d = {
        "schema_version": "1.0",
        "kind": "ris_state_library",
        "name": "d16_table",
        "provenance": _prov(),
        "frequency_hz": [FC],
        "validity_bw_hz": 2e7,
        "angle_grid": grid,
        "reciprocal": True,
        "reconfiguration_time_s": None,
        "states": states,
    }
    d.update(extra)
    return d


THETA = [-20.0, 0.0, 20.0]


def _full_states(asym: float = 0.0, sigma: dict | None = None) -> list[dict]:
    """Two states on a 3 x 3 (theta_i, theta_s) grid; state 1 has an asymmetry ``asym`` at
    (-20, 20) vs (20, -20); ``sigma`` adds entry sigmas to every state-1 entry."""
    out = []
    for s, base in enumerate((150.0, -30.0)):
        entries = []
        for ti in THETA:
            for ts in THETA:
                ph = base + 0.1 * (ti + ts)
                mag = 0.9
                if s == 1 and (ti, ts) == (-20.0, 20.0):
                    mag += asym
                e = {"f_hz": FC, "theta_i_deg": ti, "theta_s_deg": ts, "mag": mag, "phase_deg": ph}
                if s == 1 and sigma:
                    e.update(sigma)
                entries.append(e)
        out.append({"id": s, "label": f"s{s}", "gamma": entries})
    return out


def test_d16_wp1_model_full_angle_table_with_null_angle_mapping():
    models = pytest.importorskip("sremu.scenario.models")
    doc = _doc(_full_states(), grid={"theta_deg": THETA})
    lib = models.StateLibrary.model_validate(doc)
    assert lib.model_dump(mode="python")["angle_mapping"] is None  # what WP1 hands over
    t = ResponseTable.from_state_library(lib)
    assert t.angle_mapping == "full" and t.gamma.shape == (2, 1, 3, 3)
    for s, base in enumerate((150.0, -30.0)):
        for i, ti in enumerate(THETA):
            for j, ts in enumerate(THETA):
                ph = math.radians(base + 0.1 * (ti + ts))
                assert abs(t.gamma[s, 0, i, j] - 0.9 * complex(math.cos(ph), math.sin(ph))) <= 1e-15
    # the plain mapping with an explicit null loads the same table
    t2 = ResponseTable.from_state_library(dict(doc, angle_mapping=None))
    assert np.array_equal(t2.gamma, t.gamma)


def test_d16_incidence_only_needs_half_difference():
    states = [
        {
            "id": 0,
            "label": "s0",
            "gamma": [
                {"f_hz": FC, "theta_deg": th, "mag": 1.0, "phase_deg": th} for th in (0.0, 20.0)
            ],
        }
    ]
    doc = _doc(states, grid={"theta_deg": [0.0, 20.0]}, incidence_only=True, angle_mapping=None)
    with pytest.raises(TableValidationError, match="half_difference"):
        ResponseTable.from_state_library(doc)
    # null mapping on theta_deg entries without incidence_only: still not a full table
    with pytest.raises(TableValidationError, match="half_difference"):
        ResponseTable.from_state_library(dict(doc, incidence_only=False))
    t = ResponseTable.from_state_library(dict(doc, angle_mapping="half_difference"))
    assert t.angle_mapping == "half_difference"


def test_d16_entry_sigma_sets_the_passivity_bound():
    """|Gamma| = 1.05 (+0.42 dB) on a measured table: accepted with an entry
    sigma_mag_db = 0.2 (bound 1 + 3 (ln10/20) 0.2 = 1.069), rejected without it."""
    states = [
        {"id": 0, "label": "a", "gamma": [{"f_hz": FC, "mag": 0.9, "phase_deg": 10.0}]},
        {
            "id": 1,
            "label": "b",
            "gamma": [{"f_hz": FC, "mag": 1.05, "phase_deg": 190.0, "sigma_mag_db": 0.2}],
        },
    ]
    t = ResponseTable.from_state_library(_doc(states))
    sm = np.asarray(t.provenance.sigma_mag_db)
    assert sm.shape == t.gamma.shape and sm[0, 0, 0, 0] == 0.0 and sm[1, 0, 0, 0] == 0.2
    assert 1.05 <= 1 + 3 * LN10_20 * 0.2
    bare = copy.deepcopy(states)
    del bare[1]["gamma"][0]["sigma_mag_db"]
    with pytest.raises(PassivityError):
        ResponseTable.from_state_library(_doc(bare))
    # the Monte Carlo sampler uses the entry sigmas: state a (sigma 0) is never perturbed
    gen = sremu_rng.generator(1, "table_mc", "d16")
    for _ in range(5):
        smp = t.sample(gen)
        assert smp.gamma[0, 0, 0, 0] == t.gamma[0, 0, 0, 0]
        assert smp.gamma[1, 0, 0, 0] != t.gamma[1, 0, 0, 0]


def test_d16_entry_sigma_sets_the_reciprocity_tolerance():
    """An asymmetry of 0.02 in |Gamma| (state 1) exceeds the sigma-0 tolerance (1e-9) but is
    within 3 sqrt(2) sigma_Gamma for entry sigmas 0.1 dB / 1 deg on that state."""
    sig = {"sigma_mag_db": 0.1, "sigma_phase_deg": 1.0}
    tol = 3 * math.sqrt(2) * 0.9 * math.hypot(LN10_20 * 0.1, math.pi / 180 * 1.0)
    assert 0.02 <= tol
    t = ResponseTable.from_state_library(_doc(_full_states(0.02, sig), grid={"theta_deg": THETA}))
    err, tol_arr = t.reciprocity_error()
    assert float(np.max(err)) == pytest.approx(0.02, abs=1e-12)
    assert np.all(err <= tol_arr)
    with pytest.raises(ReciprocityError):
        ResponseTable.from_state_library(_doc(_full_states(0.02), grid={"theta_deg": THETA}))


def test_d16_half_difference_extends_entry_sigmas():
    grid = [0.0, 10.0, 20.0]
    sig_p = {0.0: 1.0, 10.0: 3.0, 20.0: 5.0}
    states = [
        {
            "id": 0,
            "label": "s0",
            "gamma": [
                {
                    "f_hz": FC,
                    "theta_deg": th,
                    "mag": 1.0,
                    "phase_deg": 2 * th,
                    "sigma_phase_deg": sig_p[th],
                }
                for th in grid
            ],
        }
    ]
    doc = _doc(
        states, grid={"theta_deg": grid}, incidence_only=True, angle_mapping="half_difference"
    )
    t = ResponseTable.from_state_library(doc)
    sp = np.asarray(t.provenance.sigma_phase_deg)
    th = t.theta_deg.tolist()
    assert th == [-20.0, -10.0, 0.0, 10.0, 20.0]
    for i, ti in enumerate(th):
        for j, ts in enumerate(th):
            x = abs(ti - ts) / 2  # half-difference; linear in the g grid
            assert sp[0, 0, i, j] == pytest.approx(np.interp(x, grid, [1.0, 3.0, 5.0]), abs=1e-12)
    # specular entries keep g's sigma at |theta|
    assert sp[0, 0, th.index(-20.0), th.index(20.0)] == 5.0
    assert sp[0, 0, th.index(10.0), th.index(-10.0)] == 3.0
    assert np.array_equal(sp, np.swapaxes(sp, 2, 3))


def test_d16_to_state_library_round_trip_validates_against_wp1_schema():
    models = pytest.importorskip("sremu.scenario.models")
    sig = {"sigma_mag_db": 0.1, "sigma_phase_deg": 1.0}
    t = ResponseTable.from_state_library(_doc(_full_states(0.02, sig), grid={"theta_deg": THETA}))
    doc = t.to_state_library()
    assert doc["angle_mapping"] is None
    assert doc["provenance"]["sigma_mag_db"] == 0.1 and doc["provenance"]["sigma_phase_deg"] == 1.0
    lib = models.StateLibrary.model_validate(doc)  # schema-compliant (scalar provenance sigma)
    back = ResponseTable.from_state_library(lib)
    assert np.max(np.abs(back.gamma - t.gamma)) <= 1e-15
    assert np.array_equal(
        np.asarray(back.provenance.sigma_mag_db), np.asarray(t.provenance.sigma_mag_db)
    )
    assert np.array_equal(
        np.asarray(back.provenance.sigma_phase_deg), np.asarray(t.provenance.sigma_phase_deg)
    )
