"""PH-31: response-table validators and complex-linear interpolation (section 5.5)."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from sremu import rng as sremu_rng
from sremu.errors import (
    BandwidthExceeded,
    ConfigurationError,
    GridTooCoarse,
    MissingProvenanceError,
    OutOfDomain,
    PassivityError,
    ReciprocityError,
    TableValidationError,
)
from sremu.physics.response import Provenance, ResponseTable, builtin

pytestmark = pytest.mark.physics

LN10_20 = math.log(10) / 20


def prov(kind: str = "analytic", smag: float = 0.0, sph: float = 0.0, **kw) -> Provenance:  # type: ignore[no-untyped-def]
    d = {
        "source_kind": kind,
        "description": "PH-31",
        "reference": "test",
        "reference_plane_offset_m": 0.0,
        "polarization": "co",
        "includes_cell_pattern": False,
        "sigma_mag_db": smag,
        "sigma_phase_deg": sph,
        "raw_sha256": None,
        "license": "test",
        "created": {"date": "2026-10-10"},
    }
    d.update(kw)
    return Provenance.from_dict(d)


def angle_table(theta: np.ndarray) -> np.ndarray:
    ti, ts = np.meshgrid(theta, theta, indexing="ij")
    g0 = 0.9 * np.exp(1j * np.radians(0.25 * (ti + ts)))  # symmetric, smooth
    return np.stack([g0, -g0])[:, None]  # (2, 1, T, T)


THETA = np.arange(-60.0, 61.0, 10.0)


def test_ph31_passivity():
    g = np.array([1.0 + 2e-9, 0.5]).reshape(2, 1, 1, 1)
    with pytest.raises(PassivityError):
        ResponseTable(["a", "b"], [1e9], [0.0], g, prov("analytic"))
    ResponseTable(
        ["a", "b"], [1e9], [0.0], np.array([1 + 5e-10, 0.5]).reshape(2, 1, 1, 1), prov("analytic")
    )
    # measured: |Gamma| <= 1 + 3 (ln10/20) sigma_mag_db
    bound = 1 + 3 * LN10_20 * 0.1
    ok = np.array([bound * (1 - 1e-6), 0.5]).reshape(2, 1, 1, 1)
    ResponseTable(["a", "b"], [1e9], [0.0], ok, prov("measured", smag=0.1))
    bad = np.array([bound * (1 + 1e-6), 0.5]).reshape(2, 1, 1, 1)
    with pytest.raises(PassivityError):
        ResponseTable(["a", "b"], [1e9], [0.0], bad, prov("measured", smag=0.1))


def test_ph31_reciprocity_tolerance_from_sigma(rng: np.random.Generator):
    g = angle_table(THETA)
    smag, sph = 0.2, 2.0
    sig = np.abs(g) * np.sqrt((LN10_20 * smag) ** 2 + (math.pi / 180 * sph) ** 2)
    tol = 3 * math.sqrt(2) * sig
    # noisy measured: asymmetric perturbation within the sigma-derived tolerance -> accepted
    noise = (rng.uniform(-1, 1, g.shape) + 1j * rng.uniform(-1, 1, g.shape)) * 0.3 * sig
    ResponseTable(["a", "b"], [3.5e9], THETA, g + noise, prov("measured", smag, sph))
    # asymmetric analytic (sigma = 0, tol = 1e-9) -> raises
    asym = g.copy()
    asym[0, 0, 2, 5] *= np.exp(1j * 1e-6)
    with pytest.raises(ReciprocityError):
        ResponseTable(["a", "b"], [3.5e9], THETA, asym, prov("analytic"))
    # beyond the measured tolerance -> raises; symmetrize="mean" accepts and logs
    big = g.copy()
    big[1, 0, 1, 7] += 2 * tol[1, 0, 1, 7]
    with pytest.raises(ReciprocityError):
        ResponseTable(["a", "b"], [3.5e9], THETA, big, prov("measured", smag, sph))
    t = ResponseTable(
        ["a", "b"], [3.5e9], THETA, big, prov("measured", smag, sph), symmetrize="mean"
    )
    assert t.symmetrized
    assert np.array_equal(t.gamma, np.swapaxes(t.gamma, 2, 3))
    assert any("symmetrize" in a for a in t.provenance.assumptions)


def test_ph31_grid_density_limit():
    th = np.array([-10.0, 0.0, 10.0])
    ti, ts = np.meshgrid(th, th, indexing="ij")

    def table(step: float) -> np.ndarray:  # |dGamma|/max|Gamma| = 2 sin(step/2) per grid step
        ph = step * (ti + ts) / 10.0
        return np.exp(1j * ph)[None, None]

    ok_step = 2 * math.asin(0.34 / 2)
    bad_step = 2 * math.asin(0.36 / 2)
    ResponseTable(["a"], [1e9], th, table(ok_step), prov())
    with pytest.raises(GridTooCoarse):
        ResponseTable(["a"], [1e9], th, table(bad_step), prov())
    # complex-linear midpoint dip at unit amplitude: |mid| = sqrt(1 - delta^2/4) for an
    # adjacent |dGamma| = delta. At the 0.35 limit (20.16 deg) that is -0.1351 dB; the section
    # 5.5 figure "0.134 dB" is the small-angle form 20 log10(1 - delta^2/8) of the same bound.
    delta = 0.35
    t = ResponseTable(["a"], [1e9], th, table(2 * math.asin(delta / 2)), prov())
    mid = t.query(1e9, [-5.0], [0.0])[0, 0]  # halfway between theta_i = -10 and 0
    dip_db = 20 * math.log10(abs(mid))
    assert abs(dip_db - 10 * math.log10(1 - delta**2 / 4)) <= 1e-12
    assert abs(20 * math.log10(1 - delta**2 / 8) - (-0.134)) <= 0.0005
    assert -0.1352 <= dip_db <= -0.1350


def test_ph31_domain_bandwidth_and_extrapolation():
    t = ResponseTable(
        ["a", "b"],
        [3.4e9, 3.6e9],
        THETA,
        np.concatenate([angle_table(THETA)] * 2, axis=1),
        prov(),
        validity_bw_hz=2e7,
    )
    with pytest.raises(OutOfDomain):
        t.query(3.7e9, [0.0], [0.0])
    with pytest.raises(OutOfDomain):
        t.query(3.5e9, [65.0], [0.0])
    with pytest.raises(BandwidthExceeded):
        t.check_bandwidth(2.5e7, "ap.tx0")
    t.check_bandwidth(2.0e7, "ap.tx0")
    tn = ResponseTable(
        ["a", "b"],
        [3.4e9, 3.6e9],
        THETA,
        np.concatenate([angle_table(THETA)] * 2, axis=1),
        prov(),
        extrapolate="nearest",
    )
    assert any("nearest" in a for a in tn.provenance.assumptions)
    assert np.array_equal(tn.query(3.5e9, [75.0], [0.0]), tn.query(3.5e9, [60.0], [0.0]))
    single = builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1.0, "sigma2": 1.0})
    single.gamma_at(3.5e9 + 9e6)  # inside +- validity_bw/2
    with pytest.raises(OutOfDomain):
        single.gamma_at(3.5e9 + 1.1e7)


@pytest.mark.parametrize("field", Provenance.MANDATORY)
def test_ph31_missing_provenance_fields(field: str):
    d = prov().to_dict()
    del d[field]
    with pytest.raises(MissingProvenanceError):
        Provenance.from_dict(d)


def test_ph31_complex_linear_midpoints_match_closed_form():
    th = np.array([-10.0, 0.0, 10.0])
    a, b, c, d = 0.9 * np.exp(1j * 0.1), 0.8 * np.exp(1j * 0.2), 0.7 * np.exp(1j * 0.35), 0.85
    g = np.full((1, 2, 3, 3), 0.5 + 0j)
    g[0, 0, 0, 1], g[0, 0, 1, 0] = a, a  # symmetric pairs
    g[0, 0, 0, 0], g[0, 0, 1, 1] = b, c
    g[0, 1] = g[0, 0] * d
    t = ResponseTable(["a"], [1e9, 2e9], th, g, prov(), validate=False)  # coarse on purpose
    # bilinear center of the cell (theta_i, theta_s) in [-10, 0]^2 at f = 1 GHz
    got = t.query(1e9, [-5.0], [-5.0])[0, 0]
    assert abs(got - (b + a + a + c) / 4) <= 1e-15
    # linear in f at the midpoint frequency
    got_f = t.query(1.5e9, [-10.0], [0.0])[0, 0]
    assert abs(got_f - (a + a * d) / 2) <= 1e-15
    # never magnitude + unwrapped phase: the midpoint of 1 and -1 is 0
    g2 = np.ones((1, 1, 3, 3), complex)
    g2[0, 0, 0, 0] = -1
    t2 = ResponseTable(["a"], [1e9], th, g2 * 0.17 + 0.0, prov(), validate=False)
    mid = t2.query(1e9, [-10.0], [-5.0])[0, 0]
    assert abs(mid) <= 1e-15


def test_ph31_csv_import_round_trip_and_hash(tmp_path: Path):
    th = [-20.0, 0.0, 20.0]
    rows = []
    for s, ph0 in ((0, 40.0), (1, -140.0)):
        for i in th:
            for j in th:
                rows.append(
                    {
                        "state": s,
                        "f_hz": 3.5e9,
                        "theta_i_deg": i,
                        "theta_s_deg": j,
                        "mag_db": -0.5 - 0.001 * (i * i + j * j) / 100,
                        "phase_deg": ph0 + 0.1 * (i + j),
                    }
                )
    path = tmp_path / "t.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    side = prov("measured", 0.1, 1.0).to_dict()
    (tmp_path / "t.yaml").write_text(
        yaml.safe_dump({"provenance": side, "validity_bw_hz": 1e7}), encoding="utf-8"
    )
    t = ResponseTable.from_csv(path, tmp_path / "t.yaml")
    assert t.provenance.raw_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert t.gamma.shape == (2, 1, 3, 3) and t.validity_bw_hz == 1e7
    r = rows[5]
    expected = 10 ** (r["mag_db"] / 20) * np.exp(1j * np.radians(r["phase_deg"]))
    i, j = th.index(r["theta_i_deg"]), th.index(r["theta_s_deg"])
    assert abs(t.gamma[0, 0, i, j] - expected) <= 1e-15
    # mismatching hash in the sidecar is rejected
    bad = dict(side, raw_sha256="0" * 64)
    with pytest.raises(TableValidationError):
        ResponseTable.from_csv(path, {"provenance": bad})
    # an incomplete grid is rejected
    (tmp_path / "short.csv").write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(TableValidationError):
        ResponseTable.from_csv(tmp_path / "short.csv", {"provenance": side})


def test_ph31_mag_null_requires_assume_mag():
    with pytest.raises(ConfigurationError):
        builtin("lab_binary_3p5ghz")
    t = builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1.0, "sigma2": 0.7})
    assert abs(abs(t.gamma[1, 0, 0, 0]) - 0.7) <= 1e-15
    assert any("ASSUMED" in a for a in t.provenance.assumptions)
    with pytest.raises(ConfigurationError):
        builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1, "sigma2": 1, "nope": 1})


def test_ph31_sample_keeps_reciprocity_and_follows_sigma():
    th = np.array([-20.0, 0.0, 20.0])
    ti, ts = np.meshgrid(th, th, indexing="ij")
    g = (0.9 * np.exp(1j * np.radians(0.5 * (ti + ts))))[None, None]
    t = ResponseTable(["a"], [3.5e9], th, g, prov("measured", 0.2, 3.0))
    gen = sremu_rng.generator(20261010, "table_mc", "ph31/sample")  # D14
    samples = np.stack([t.sample(gen).gamma for _ in range(4000)])
    assert np.array_equal(samples, np.swapaxes(samples, 3, 4))  # exactly reciprocal
    mag_db = 20 * np.log10(np.abs(samples[:, 0, 0, 0, 1]) / 0.9)
    ph_deg = np.degrees(np.angle(samples[:, 0, 0, 0, 1] / g[0, 0, 0, 1]))
    assert abs(np.std(mag_db) - 0.2) <= 0.2 * 0.05  # 4000 draws: ~1.1 % standard error
    assert abs(np.std(ph_deg) - 3.0) <= 3.0 * 0.05
    assert any("Monte Carlo" in a for a in t.sample(gen).provenance.assumptions)


def test_ph31_matlab_scurve_importer(tmp_path: Path):
    head = "f_hz,tan_delta,patch_L_mm,gamma_phase_deg,gamma_abs\n"
    rows = [f"{f},0,{L},{150 - 10 * L + f / 1e9},1.0\n" for f in (3.4e9, 3.6e9) for L in (2, 10)]
    rows.append("3.4e9,0.01,2,1.0,0.5\n")  # lossy row: excluded (tan_delta != 0)
    path = tmp_path / "unitcell_fullwave_test.csv"
    path.write_text(head + "".join(rows), encoding="utf-8")
    t = ResponseTable.from_matlab_scurve(path, validity_bw_hz=5e7)
    assert t.labels == ("L2mm", "L10mm") and t.freqs_hz.tolist() == [3.4e9, 3.6e9]
    assert t.provenance.source_kind == "fullwave"
    assert t.provenance.raw_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    got = t.gamma_at(3.5e9)  # linear in f between the two frequencies
    want = 0.5 * (
        np.exp(1j * np.radians([130 + 3.4, 50 + 3.4]))
        + np.exp(1j * np.radians([130 + 3.6, 50 + 3.6]))
    )
    assert np.max(np.abs(got - want)) <= 1e-12
