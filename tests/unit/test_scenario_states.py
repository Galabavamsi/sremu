"""Built-in state libraries (``data/states``; DESIGN.md 5.5, 5.6, 9.4; PH-13/14/25 values).

The data files are checked against the spec's oracle values, computed here from the stored
reflection coefficients with independent closed forms or quadrature:

* large-N alphabet loss ``eta = (mean_phi max_s Re{Gamma_s e^{-j phi}})**2`` (best
  per-element choice, uniform channel phase; midpoint rule on 2**16 phases): lab pair
  -4.105 dB, ideal b-bit -3.922 / -0.912 / -0.224 dB (PH-13, +/-0.001 dB), Tang RIS1/RIS2
  -5.93 / -6.16 dB (section 5.6, +/-0.01 dB);
* binary m/d decomposition of the lab pair at |Gamma| = 1: |d| = -0.18 dB,
  |m| = -13.86 dB at 127.9 deg (PH-25, +/-0.01 dB); ideal ON/OFF -6.02 / -6.02 dB;
* Abeywickrama: mean amplitude over the expanded 2**8-state family gives -5.508 dB for
  (beta_min, alpha) = (0.2, 1.6) and -1.085 dB for (0.8, 1.6), independent of phi
  (PH-14, +/-0.01 dB and 1e-6 dB);
* transcription of section 5.5 (phases, magnitudes, frequencies, source kinds).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sremu.errors import SchemaError
from sremu.scenario.load import (
    apply_assume_mag,
    builtin_library_names,
    builtin_states_dir,
    load_state_library,
    parse_library_ref,
    parse_state_library,
)
from sremu.scenario.models import StateLibrary

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]


def gammas(lib: StateLibrary) -> np.ndarray:
    """Angle-independent alphabet as complex values (single frequency)."""
    out = []
    for s in lib.states:
        (g,) = s.gamma
        assert g.mag is not None
        out.append(g.mag * np.exp(1j * math.radians(g.phase_deg)))
    return np.asarray(out)


def eta_db(gam: np.ndarray, n: int = 1 << 16) -> float:
    phi = 2 * np.pi * (np.arange(n) + 0.5) / n
    best = np.max(np.real(gam[:, None] * np.exp(-1j * phi)[None, :]), axis=0)
    return 20 * math.log10(best.mean())


def test_builtin_directory_and_names():
    assert builtin_states_dir().resolve() == (REPO / "data" / "states").resolve()
    assert set(builtin_library_names()) == {
        "lab_binary_3p5ghz", "tang2022_ris1_27ghz", "tang2022_ris2_33ghz", "ideal_bbit",
        "ideal_antipodal", "on_off", "abeywickrama",
    }  # fmt: skip


def test_lab_binary_transcription_and_mandatory_assume_mag():
    lib, path = load_state_library("lab_binary_3p5ghz")
    assert path.name == "lab_binary_3p5ghz.yaml"
    assert lib.provenance.source_kind == "assumed"
    assert lib.provenance.resolved_data_class() == "ASSUMED"
    assert lib.provenance.sigma_phase_deg == 15.0
    assert lib.frequency_hz == [3.5e9] and lib.validity_bw_hz == 2.0e7
    assert lib.labels == ("sigma1", "sigma2")
    assert [s.gamma[0].phase_deg for s in lib.states] == [49.64, -153.77]
    assert lib.null_mag_labels() == ("sigma1", "sigma2")
    with pytest.raises(SchemaError, match="mag: null"):
        apply_assume_mag(lib, None)
    with pytest.raises(SchemaError, match="mag: null"):
        apply_assume_mag(lib, {"sigma1": 1.0})
    with pytest.raises(SchemaError, match="unknown states"):
        apply_assume_mag(lib, {"sigma1": 1.0, "sigma2": 1.0, "sigma3": 1.0})


def test_ph13_ph25_lab_pair_values_from_the_data_file():
    lib, _ = load_state_library("lab_binary_3p5ghz")
    g = gammas(apply_assume_mag(lib, {"sigma1": 1.0, "sigma2": 1.0}))
    delta = (math.degrees(np.angle(g[0] / g[1]))) % 360.0
    assert delta == pytest.approx(203.41, abs=1e-9)
    dp = math.radians(min(delta, 360 - delta))
    closed = 20 * math.log10(2 * math.sin(dp / 2) / math.pi)
    assert closed == pytest.approx(-4.105, abs=0.001)  # PH-13 lab pair
    assert eta_db(g) == pytest.approx(closed, abs=1e-6)
    m, d = (g[0] + g[1]) / 2, (g[0] - g[1]) / 2
    assert 20 * math.log10(abs(d)) == pytest.approx(-0.18, abs=0.01)  # PH-25
    assert 20 * math.log10(abs(m)) == pytest.approx(-13.86, abs=0.01)
    assert math.degrees(np.angle(m)) == pytest.approx(127.9, abs=0.05)
    # unequal amplitudes (section 5.6 table): |Gamma2| = 0.7 and 0.5
    for mag2, d_db, m_db in ((0.7, -1.59, -12.90), (0.5, -2.66, -10.81)):
        g2 = gammas(apply_assume_mag(lib, {"sigma1": 1.0, "sigma2": mag2}))
        assert 20 * math.log10(abs((g2[0] - g2[1]) / 2)) == pytest.approx(d_db, abs=0.01)
        assert 20 * math.log10(abs((g2[0] + g2[1]) / 2)) == pytest.approx(m_db, abs=0.01)


@pytest.mark.parametrize(
    ("name", "freq", "values", "eta_expect"),
    [
        ("tang2022_ris1_27ghz", 2.7e10, [(0.9, 165.0), (0.7, 0.0)], -5.93),
        ("tang2022_ris2_33ghz", 3.3e10, [(0.8, 150.0), (0.8, 0.0)], -6.16),
    ],
)
def test_tang2022_transcription_and_alphabet_loss(name, freq, values, eta_expect):
    lib, _ = load_state_library(name)
    assert lib.provenance.source_kind == "measured"
    assert lib.provenance.resolved_data_class() == "REPORTED"
    assert "Table III" in lib.provenance.reference
    assert lib.frequency_hz == [freq]
    assert [(s.gamma[0].mag, s.gamma[0].phase_deg) for s in lib.states] == values
    assert eta_db(gammas(lib)) == pytest.approx(eta_expect, abs=0.01)


def test_ideal_antipodal_and_on_off():
    anti, _ = load_state_library("ideal_antipodal")
    np.testing.assert_allclose(gammas(anti), [1, -1], atol=1e-15)
    assert eta_db(gammas(anti)) == pytest.approx(-3.922, abs=0.001)
    onoff, _ = load_state_library("on_off")
    assert onoff.labels == ("off", "on")  # state 0 = absorbing OFF, state 1 = ON
    g = gammas(onoff)
    np.testing.assert_allclose(g, [0, 1], atol=0)
    m, d = (g[0] + g[1]) / 2, (g[1] - g[0]) / 2
    assert 20 * math.log10(abs(m)) == pytest.approx(-6.02, abs=0.01)
    assert 20 * math.log10(abs(d)) == pytest.approx(-6.02, abs=0.01)
    for lib in (anti, onoff):
        assert lib.frequency_hz == "any" and lib.validity_bw_hz is None
        assert lib.provenance.resolved_data_class() == "ANALYTIC"


@pytest.mark.parametrize(("b", "expect_db"), [(1, -3.922), (2, -0.912), (3, -0.224)])
def test_ph13_ideal_bbit_family_expansion(b, expect_db):
    lib, _ = load_state_library(f"ideal_bbit(b={b})")
    assert lib.n_states == 2**b
    assert lib.parameters == {"b": float(b)}
    g = gammas(lib)
    np.testing.assert_allclose(g, np.exp(2j * np.pi * np.arange(2**b) / 2**b), atol=1e-15)
    closed = 20 * math.log10(2**b / math.pi * math.sin(math.pi / 2**b))
    assert closed == pytest.approx(expect_db, abs=0.001)
    assert eta_db(g) == pytest.approx(closed, abs=1e-6)
    # positional argument and default
    assert load_state_library(f"ideal_bbit({b})")[0].states == lib.states
    assert load_state_library("ideal_bbit")[0].n_states == 2


@pytest.mark.parametrize(("beta_min", "expect_db"), [(0.2, -5.508), (0.8, -1.085)])
def test_ph14_abeywickrama_family_expansion(beta_min, expect_db):
    vals = []
    for phi in (0.0, 77.0, -130.0):
        lib, _ = load_state_library(
            f"abeywickrama(b=8, beta_min={beta_min}, phi_deg={phi}, alpha=1.6)"
        )
        assert lib.n_states == 256
        mags = np.array([s.gamma[0].mag for s in lib.states])
        th = 2 * np.pi * np.arange(256) / 256
        closed = (1 - beta_min) * ((np.sin(th - math.radians(phi)) + 1) / 2) ** 1.6 + beta_min
        np.testing.assert_allclose(mags, closed, rtol=1e-14)
        vals.append(20 * math.log10(mags.mean()))
    assert vals[0] == pytest.approx(expect_db, abs=0.01)
    assert max(vals) - min(vals) <= 1e-6  # independent of phi
    lib0, _ = load_state_library("abeywickrama")
    assert lib0.parameters == {"b": 1.0, "beta_min": 0.2, "phi_deg": 0.0, "alpha": 1.6}


@pytest.mark.parametrize(
    "ref",
    [
        "ideal_bbit(b=0)",
        "ideal_bbit(b=1.5)",
        "ideal_bbit(c=2)",
        "ideal_bbit(1, b=1)",
        "ideal_bbit(1, 2)",
        "abeywickrama(beta_min=1.5)",
        "ideal_antipodal(b=2)",
        "nonexistent",
        "ideal_bbit(b='x')",
        "ideal_bbit(b=__import__('os'))",
    ],
)
def test_bad_library_references(ref):
    with pytest.raises(SchemaError):
        load_state_library(ref)


def test_parse_library_ref():
    assert parse_library_ref("on_off") == ("on_off", (), {})
    assert parse_library_ref(" abeywickrama(2, alpha=1.6) ") == (
        "abeywickrama",
        (2,),
        {"alpha": 1.6},
    )


BASE = {
    "schema_version": "1.0",
    "kind": "ris_state_library",
    "name": "t",
    "provenance": {
        "source_kind": "analytic", "description": "d", "reference": "r",
        "reference_plane_offset_m": 0.0, "polarization": "co", "includes_cell_pattern": False,
        "sigma_mag_db": 0.0, "sigma_phase_deg": 0.0, "raw_sha256": None, "license": "x",
        "created": {"date": "2026-10-10", "tool": "t"},
    },
    "frequency_hz": [1e9],
    "validity_bw_hz": 1e6,
    "angle_grid": "none",
    "states": [{"id": 0, "label": "a", "gamma": [{"f_hz": 1e9, "mag": 1.0, "phase_deg": 0}]}],
}  # fmt: skip


def _lib(**over):
    d = {**BASE, **over}
    return parse_state_library(d)


def test_state_library_structure_rules():
    _lib()
    with pytest.raises(SchemaError, match="reciprocal"):
        _lib(reciprocal=False)
    with pytest.raises(SchemaError, match="angle_mapping"):
        _lib(incidence_only=True, angle_grid={"theta_deg": [0, 30]})
    with pytest.raises(SchemaError, match=r"0\.\.S-1"):
        _lib(states=[{"id": 1, "label": "a", "gamma": [{"f_hz": 1e9, "mag": 1.0, "phase_deg": 0}]}])
    with pytest.raises(SchemaError, match="not in frequency_hz"):
        _lib(states=[{"id": 0, "label": "a", "gamma": [{"f_hz": 2e9, "mag": 1.0, "phase_deg": 0}]}])
    one = [{"f_hz": 1e9, "mag": 0.5, "phase_deg": 0, "theta_i_deg": -30, "theta_s_deg": 30}]
    with pytest.raises(SchemaError, match="one entry per grid point"):
        _lib(
            angle_grid={"theta_deg": [-30, 30]},  # needs 4 (theta_i, theta_s) entries
            states=[{"id": 0, "label": "a", "gamma": one}],
        )
    grid = [(ti, ts) for ti in (-30.0, 30.0) for ts in (-30.0, 30.0)]
    full = [
        {"f_hz": 1e9, "mag": 0.5, "phase_deg": 0, "theta_i_deg": a, "theta_s_deg": b}
        for a, b in grid
    ]
    ok = _lib(angle_grid={"theta_deg": [-30, 30]}, states=[{"id": 0, "label": "a", "gamma": full}])
    assert not ok.is_separable()
    inc = [{"f_hz": 1e9, "mag": 0.5, "phase_deg": 0, "theta_deg": t} for t in (-30.0, 30.0)]
    _lib(
        incidence_only=True,
        angle_mapping="half_difference",
        angle_grid={"theta_deg": [-30, 30]},
        states=[{"id": 0, "label": "a", "gamma": inc}],
    )
    # incidence-only entries take theta_deg only; a stray theta_i_deg or theta_s_deg is an
    # error even when its value is 0.0 (falsy; review fix)
    for stray in ({"theta_s_deg": 0.0}, {"theta_s_deg": 30.0}, {"theta_i_deg": 0.0}):
        bad_inc = [{**e, **stray} for e in inc]
        with pytest.raises(SchemaError, match="theta_deg only"):
            _lib(
                incidence_only=True,
                angle_mapping="half_difference",
                angle_grid={"theta_deg": [-30, 30]},
                states=[{"id": 0, "label": "a", "gamma": bad_inc}],
            )
    with pytest.raises(SchemaError, match="family"):
        _lib(family={"name": "ideal_bbit", "formula": "f", "parameters": {}})
    with pytest.raises(SchemaError):
        _lib(provenance={**BASE["provenance"], "source_kind": "guessed"})
    with pytest.raises(SchemaError):
        _lib(provenance={k: v for k, v in BASE["provenance"].items() if k != "license"})
