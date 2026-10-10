"""MATLAB R2026a goldens G1-G8 (DESIGN.md section 10.6): integrity, oracle cross-checks and
the comparison harness for the sremu implementation.

Three kinds of tests:

* ``integrity``: every golden carries meta (MATLAB release, script/helper/output sha256,
  runtime) that matches the files on disk.
* ``oracle``: the independent closed forms of ``tests/oracles`` reproduce the MATLAB goldens
  (code-to-code between two independent implementations; runs now).
* ``sremu``: the implementation, reached only through the public contracts
  (PropagationModel / render / noise, see ``tests/golden/sremu_adapter.py``), against the
  goldens with the section 10.6 acceptance. Marked ``xfail(strict=False)`` only when the
  implementation module is not importable at collection time ("awaiting WP2/WP5").

MATLAB agreement is code-to-code only; it never validates absolute RIS power (section 1.6).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.contracts import PathKind
from sremu.units import C0, noise_power_w
from tests.golden import sremu_adapter as sa
from tests.golden.harness import (
    GOLDEN_DIR,
    GOLDEN_IDS,
    complex_col,
    fading_stats,
    integrity_errors,
    level_errors_db,
    load_csv,
    load_meta,
    max_rel,
    max_rel_floor,
    max_rel_to_peak,
    parabolic_peak,
    sha256_lf,
)
from tests.oracles import mobility as mob
from tests.oracles import rf
from tests.oracles.ris import Panel, cos_power, isotropic, k1_h, los_h, make_panel
from tests.oracles.tables import ORACLE

pytestmark = pytest.mark.golden

PROPAGATION = sa.propagation_available()
AWAIT_PROP = pytest.mark.xfail(
    not PROPAGATION,
    reason="awaiting WP2/WP5: sremu.physics.propagation / sremu.scenario.models not importable",
    strict=False,
)


def _backend_params() -> list:
    return [
        pytest.param(
            b,
            marks=pytest.mark.xfail(
                not sa.render_available(b),
                reason=f"awaiting WP2/WP5: {sa.BACKEND_MODULES[b]} not importable",
                strict=False,
            ),
        )
        for b in sa.BACKEND_MODULES
    ]


def tol(key: str) -> float:
    return ORACLE[key].tol


# =========================================================================== integrity


@pytest.mark.parametrize("golden_id", GOLDEN_IDS)
def test_golden_integrity(golden_id):
    """Release, script/helper sha256 and output sha256/row counts match the meta file."""
    assert integrity_errors(golden_id) == []


def test_golden_integrity_check_has_teeth(tmp_path, monkeypatch):
    """A changed output byte is reported."""
    import shutil

    from tests.golden import harness

    for f in GOLDEN_DIR.glob("G8_two_ray*"):
        shutil.copy(f, tmp_path / f.name)
    csv_path = tmp_path / "G8_two_ray.csv"
    csv_path.write_bytes(csv_path.read_bytes().replace(b"139.09", b"139.19", 1))
    monkeypatch.setattr(harness, "GOLDEN_DIR", tmp_path)
    harness.load_meta.cache_clear()
    harness._load_csv.cache_clear()
    try:
        errs = harness.integrity_errors("G8_two_ray")
    finally:
        harness.load_meta.cache_clear()
        harness._load_csv.cache_clear()
    assert any("sha256 differs" in e for e in errs)


def test_g4_input_provenance():
    import json

    prov = json.loads((GOLDEN_DIR / "G4_input_tx_iq.provenance.json").read_text(encoding="utf-8"))
    assert sha256_lf(GOLDEN_DIR / prov["file"]) == prov["sha256"]
    assert load_meta("G4_cfo")["parameters"]["input_sha256"] == prov["sha256"]
    assert prov["rows"] == 2000 and prov["data_class"] == "MATLAB"


# =========================================================================== G1


def _g1_rows():
    return load_csv("G1_lab_polar.csv")


def _g1_cells(geometry: str) -> tuple[Panel, np.ndarray]:
    c = load_csv("G1_cells.csv")
    m = c["geometry"] == geometry
    pos = np.stack([c["x"][m], c["y"][m], c["z"][m]], axis=1)
    g = load_meta("G1_lab_polar")["parameters"]["geometries"][geometry]
    n = np.asarray(g["normal"], float)
    u = np.asarray(g["u_axis"], float)
    idx = np.arange(pos.shape[0], dtype=np.int64)
    area = np.full(pos.shape[0], g["pitch_u_m"] * g["pitch_v_m"])
    p = Panel(
        pos,
        area,
        n,
        u,
        np.cross(n, u),
        c["iu"][m].astype(np.int64),
        c["iv"][m].astype(np.int64),
        idx,
    )
    return p, c["state"][m].astype(np.int64)


def _alphabet(g: dict) -> np.ndarray:
    return np.asarray(g["alphabet_mag"], float) * np.exp(
        1j * np.radians(np.asarray(g["alphabet_phase_deg"], float))
    )


def _oracle_antenna(label: str, pos: np.ndarray):
    if label == "iso":
        return isotropic
    alpha = float(label.removeprefix("cos"))
    return cos_power(alpha, -pos)  # aimed at the panel centre (origin)


def test_g1_matlab_toolbox_path_matches_hand_formula():
    r = _g1_rows()
    assert max_rel(complex_col(r, "h_ris"), complex_col(r, "h_ris_hand")) <= 1e-11
    chk = load_meta("G1_lab_polar")["checks"]
    assert (
        chk["cosine_element_field_pattern_max_dev"] <= 1e-12
    )  # CosinePower 0.5 is a field pattern


@pytest.mark.parametrize("geometry", ["lab_A_3x1", "lab_B_6x1", "lab_C_3x2", "tang_mmris1"])
def test_g1_oracle_panel_geometry_matches_matlab_cells(geometry):
    g = load_meta("G1_lab_polar")["parameters"]["geometries"][geometry]
    p_ml, state = _g1_cells(geometry)
    p = make_panel(
        g["center_m"],
        g["normal"],
        g["u_axis"],
        g["n_u"],
        g["n_v"],
        g["pitch_u_m"],
        g["pitch_v_m"],
        tiles=(g["n_tiles_u"], g["n_tiles_v"]),
        gap_m=g["gap_m"],
    )
    assert np.max(np.abs(p.cells - p_ml.cells)) <= 1e-12
    assert np.array_equal(p.iu, p_ml.iu) and np.array_equal(p.iv, p_ml.iv)
    if "tile_states" in g:
        ts = np.atleast_2d(np.asarray(g["tile_states"]))
        tu = p.iu // (g["n_u"] // g["n_tiles_u"])
        tv = p.iv // (g["n_v"] // g["n_tiles_v"])
        assert np.array_equal(state, ts[tv, tu])


def test_g1_oracle_k1_and_los_match_matlab():
    """Independent oracle K1 (tests/oracles/ris.py) vs the MATLAB toolbox path: rel <= 1e-9."""
    r = _g1_rows()
    meta = load_meta("G1_lab_polar")["parameters"]["geometries"]
    h_ris = np.empty(r["case_id"].size, complex)
    h_los = np.empty_like(h_ris)
    for i in range(r["case_id"].size):
        geom = str(r["geometry"][i])
        p, state = _g1_cells(geom)
        gam = _alphabet(meta[geom])[state]
        tx = np.array([r["tx_x"][i], r["tx_y"][i], r["tx_z"][i]])
        rx = np.array([r["rx_x"][i], r["rx_y"][i], r["rx_z"][i]])
        gt = _oracle_antenna(str(r["tx_antenna"][i]), tx)
        gr = _oracle_antenna(str(r["rx_antenna"][i]), rx)
        h_ris[i] = k1_h(tx, rx, p, r["fc_hz"][i], gam, g_t=gt, g_r=gr)
        h_los[i] = los_h(tx, rx, r["fc_hz"][i], gt, gr)
    assert max_rel(h_ris, complex_col(r, "h_ris")) <= tol("G1_rel")
    # LOS can be exactly 0 (Rx CosPower antenna facing away from the Tx): floor = isotropic LOS
    assert max_rel_floor(h_los, complex_col(r, "h_los"), _g1_iso_los(r)) <= tol("G1_rel")


def _g1_iso_los(r) -> np.ndarray:
    tx = np.stack([r["tx_x"], r["tx_y"], r["tx_z"]], axis=1)
    rx = np.stack([r["rx_x"], r["rx_y"], r["rx_z"]], axis=1)
    d = np.linalg.norm(rx - tx, axis=1)
    return C0 / r["fc_hz"] / (4 * math.pi * d)


def _g1_cases():
    r = _g1_rows()
    seen = []
    for g, v in zip(r["geometry"], r["variant"], strict=True):
        if (str(g), str(v)) not in seen:
            seen.append((str(g), str(v)))
    return seen


@AWAIT_PROP
@pytest.mark.filterwarnings(
    "ignore:cell_model='point' forced"
)  # lab pitch 44.8 mm > lambda/2: point forced on purpose
@pytest.mark.parametrize(("geometry", "variant"), _g1_cases())
def test_g1_sremu_k1_matches_golden(geometry, variant):
    """sremu PropagationModel (cell_model point) vs golden: max rel |dh| <= 1e-9 (RIS and LOS)."""
    r = _g1_rows()
    m = (r["geometry"] == geometry) & (r["variant"] == variant)
    g = load_meta("G1_lab_polar")["parameters"]["geometries"][geometry]
    _, state = _g1_cells(geometry)
    fc = float(r["fc_hz"][m][0])
    txp = np.array([r["tx_x"][m][0], r["tx_y"][m][0], r["tx_z"][m][0]])
    rxp = np.stack([r["rx_x"][m], r["rx_y"][m], r["rx_z"][m]], axis=1)

    def node(nid, pos, label, direction):
        if label == "iso":
            return sa.Node(nid, pos, direction)
        return sa.Node(
            nid, pos, direction, float(label.removeprefix("cos")), list(-pos / np.linalg.norm(pos))
        )

    nodes = [node("tx", txp, str(r["tx_antenna"][m][0]), "tx")]
    nodes += [
        node(f"rx{j:03d}", rxp[j], str(r["rx_antenna"][m][j]), "rx") for j in range(rxp.shape[0])
    ]
    panel = sa.PanelSpec(
        "ris0",
        g["center_m"],
        g["normal"],
        g["u_axis"],
        g["n_u"],
        g["n_v"],
        g["pitch_u_m"],
        g["pitch_v_m"],
        _alphabet(g),
        g["n_tiles_u"],
        g["n_tiles_v"],
        g["gap_m"],
        "point",
    )
    res = sa.evaluate_with_state_files(
        f"g1_{geometry}_{variant}", fc, nodes, [panel], {"ris0": state}
    )
    links = [sa.link_id("tx", f"rx{j:03d}") for j in range(rxp.shape[0])]
    h_ris = np.array([res[lk].kind(PathKind.RIS_TILE) for lk in links])
    h_los = np.array([res[lk].kind(PathKind.LOS) for lk in links])
    assert max_rel(h_ris, complex_col(r, "h_ris")[m]) <= tol("G1_rel")
    assert max_rel_floor(h_los, complex_col(r, "h_los")[m], _g1_iso_los(r)[m]) <= tol("G1_rel")


# =========================================================================== G2 / G3

_G23 = {"G2_beam_pattern": "state_continuous_4bit", "G3_beam_pattern_1bit": "state_onebit"}


def _g23(golden_id: str):
    meta = load_meta(golden_id)
    prm = meta["parameters"]
    cells = load_csv("G2_G3_cells.csv")
    state = cells[_G23[golden_id]].astype(np.int64)
    nb = prm["n_states"]
    alphabet = (
        np.exp(2j * np.pi * np.arange(nb) / nb) if nb > 2 else np.array([1.0 + 0j, -1.0 + 0j])
    )
    pos = np.stack([cells["x"], cells["y"], cells["z"]], axis=1)
    idx = np.arange(pos.shape[0], dtype=np.int64)
    p = Panel(
        pos,
        np.full(pos.shape[0], prm["pitch_m"] ** 2),
        np.array([1.0, 0, 0]),
        np.array([0, 1.0, 0]),
        np.array([0, 0, 1.0]),
        cells["iu"].astype(np.int64),
        cells["iv"].astype(np.int64),
        idx,
    )
    return meta, load_csv(f"{golden_id}.csv"), p, state, alphabet


def _cell_gain_db(meta) -> float:
    prm = meta["parameters"]
    lam = rf.wavelength(prm["fc_hz"])
    return 20 * math.log10(4 * math.pi * prm["pitch_m"] ** 2 / lam**2)


def _check_lobes(golden_id: str, az: np.ndarray, level_db: np.ndarray) -> None:
    main_az, main_db = parabolic_peak(az, level_db, (az > 0) & (az < 35))
    assert abs(main_az - ORACLE["G2_mainlobe_deg"].value) <= tol("G2_mainlobe_deg")
    if golden_id == "G3_beam_pattern_1bit":
        mir_az, mir_db = parabolic_peak(az, level_db, (az > 30) & (az < 60))
        assert abs(mir_az - ORACLE["G3_mirror_deg"].value) <= tol("G3_mirror_deg")
        assert abs((mir_db - main_db) - ORACLE["G3_mirror_db"].value) <= tol("G3_mirror_db")


@pytest.mark.parametrize("golden_id", list(_G23))
def test_g2_g3_matlab_reflector_matches_hand_af_and_lobes(golden_id):
    _, r, _, _, _ = _g23(golden_id)
    y = complex_col(r, "y_reflector")
    assert max_rel_to_peak(y, complex_col(r, "y_hand_af")) <= 1e-12
    _check_lobes(golden_id, r["az_deg"], 20 * np.log10(np.abs(y)))


@pytest.mark.parametrize("golden_id", list(_G23))
def test_g2_g3_oracle_k1_matches_matlab(golden_id):
    """Oracle K1 at 500 m vs the MATLAB K1 column (code-to-code, 1e-9 of the peak), and the
    section 10.6 acceptance applied to the oracle pattern (levels within 0.05 dB above -30 dB
    after the cell-gain constant; lobes)."""
    meta, r, p, state, alphabet = _g23(golden_id)
    tx = np.asarray(meta["parameters"]["tx_m"], float)
    fc = meta["parameters"]["fc_hz"]
    gam = alphabet[state]
    h = np.array(
        [
            k1_h(tx, [x, y, z], p, fc, gam)
            for x, y, z in zip(r["rx_x"], r["rx_y"], r["rx_z"], strict=True)
        ]
    )
    assert max_rel_to_peak(h, complex_col(r, "h_k1")) <= tol("G1_rel")
    test_db = 20 * np.log10(np.abs(h)) - _cell_gain_db(meta)
    ref_db = 20 * np.log10(np.abs(complex_col(r, "h_reflector_channel")))
    assert np.max(np.abs(level_errors_db(test_db, ref_db))) <= tol("G2_levels_db")
    _check_lobes(golden_id, r["az_deg"], 20 * np.log10(np.abs(h)))


def test_g2_level_criterion_has_teeth_unit_gain_cells_equal_legacy_v2():
    """Teeth of the G2 level rule, and the paper's statement that MATLAB's unit-gain cells
    share v2's normalization: the legacy v2 kernel reproduces the RISReflector channel within
    0.05 dB WITHOUT the cell-gain constant, so after removing 20 log(4 pi dx dy/lambda^2) (as
    the rule does for K1) it fails by about that constant (9.94 dB)."""
    from tests.oracles.ris import legacy_v2_h

    meta, r, p, state, alphabet = _g23("G2_beam_pattern")
    tx = np.asarray(meta["parameters"]["tx_m"], float)
    fc = meta["parameters"]["fc_hz"]
    sel = np.arange(0, r["az_deg"].size, 4)  # every 1 deg keeps the test fast
    h_v2 = np.array(
        [
            legacy_v2_h(tx, [r["rx_x"][i], r["rx_y"][i], r["rx_z"][i]], p, fc, alphabet[state])
            for i in sel
        ]
    )
    ref_db = 20 * np.log10(np.abs(complex_col(r, "h_reflector_channel")[sel]))
    v2_db = 20 * np.log10(np.abs(h_v2))
    assert np.max(np.abs(level_errors_db(v2_db, ref_db))) <= tol("G2_levels_db")
    err = level_errors_db(v2_db - _cell_gain_db(meta), ref_db)
    assert np.min(np.abs(err)) > 9.0  # the rule rejects a v2-normalized implementation


@AWAIT_PROP
@pytest.mark.parametrize("golden_id", list(_G23))
def test_g2_g3_sremu_beam_pattern(golden_id):
    """sremu vs RISReflector: levels above -30 dB within 0.05 dB after 20 log(4 pi dx dy/lambda^2),
    main lobe +-0.5 deg (G2), mirror lobe 41.15 +- 0.25 deg / -0.97 +- 0.10 dB (G3); plus the
    exact K1 column at 1e-9 of the peak (same kernel as G1)."""
    meta, r, _, state, alphabet = _g23(golden_id)
    prm = meta["parameters"]
    tx = np.asarray(prm["tx_m"], float)
    nodes = [sa.Node("tx", tx, "tx")]
    rxp = np.stack([r["rx_x"], r["rx_y"], r["rx_z"]], axis=1)
    # the Rx at az = -30 deg coincides with the Tx (monostatic point): its LOS is undefined,
    # so it is left out of the sremu scene (and out of every comparison below)
    keep = np.linalg.norm(rxp - tx, axis=1) > 1e-6
    assert int((~keep).sum()) == 1
    az = r["az_deg"][keep]
    nodes += [sa.Node(f"rx{j:03d}", rxp[j], "rx") for j in np.flatnonzero(keep)]
    panel = sa.PanelSpec(
        "ris0", [0, 0, 0], [1, 0, 0], [0, 1, 0], 16, 16, prm["pitch_m"], prm["pitch_m"], alphabet
    )
    res = sa.evaluate_with_state_files(
        f"g23_{golden_id}", prm["fc_hz"], nodes, [panel], {"ris0": state}
    )
    h = np.array(
        [res[sa.link_id("tx", f"rx{j:03d}")].kind(PathKind.RIS_TILE) for j in np.flatnonzero(keep)]
    )
    assert max_rel_to_peak(h, complex_col(r, "h_k1")[keep]) <= tol("G1_rel")
    test_db = 20 * np.log10(np.abs(h)) - _cell_gain_db(meta)
    ref_db = 20 * np.log10(np.abs(complex_col(r, "h_reflector_channel")[keep]))
    assert np.max(np.abs(level_errors_db(test_db, ref_db))) <= tol("G2_levels_db")
    _check_lobes(golden_id, az, 20 * np.log10(np.abs(h)))
    ml_az, _ = parabolic_peak(az, ref_db, (az > 0) & (az < 35))
    sr_az, _ = parabolic_peak(az, test_db, (az > 0) & (az < 35))
    assert abs(sr_az - ml_az) <= tol("G2_mainlobe_deg")


# =========================================================================== G4


def _g4():
    r = load_csv("G4_cfo.csv")
    x = r["x_re"] + 1j * r["x_im"]
    meta = load_meta("G4_cfo")
    return r, x, meta["parameters"]["offsets_hz"], int(meta["parameters"]["f_s_sps"])


def _g4_col(f0: int) -> str:
    return "y_" + (f"p{f0}" if f0 >= 0 else f"m{-f0}")


def test_g4_input_is_exactly_complex64():
    _, x, _, _ = _g4()
    assert np.array_equal(x.astype(np.complex64).astype(np.complex128), x)


def test_g4_oracle_cfo_matches_comm_phase_frequency_offset():
    r, x, offsets, fs = _g4()
    for f0 in offsets:
        y = complex_col(r, _g4_col(int(f0)))
        y_oracle = rf.cfo_rotate(x, int(f0), fs)
        rms = math.sqrt(float(np.mean(np.abs(y) ** 2)))
        assert np.max(np.abs(y_oracle - y)) <= tol("G4_rel_rms") * rms


@pytest.mark.parametrize("backend", _backend_params())
def test_g4_sremu_render_lo_offset_any_chunking(backend):
    """render() with lo_offset_hz = df on a zero-delay unit path, fed in irregular chunks:
    accumulator within 1e-9 x RMS of comm.PhaseFrequencyOffset."""
    r, x, offsets, fs = _g4()
    for f0 in offsets:
        y = complex_col(r, _g4_col(int(f0)))
        rms = math.sqrt(float(np.mean(np.abs(y) ** 2)))
        for chunks in ((2000,), (1, 7, 128, 333, 1000), (64,)):
            out = sa.render_cfo(backend, x.astype(np.complex64), fs, float(f0), chunks)
            assert np.max(np.abs(out - y)) <= tol("G4_rel_rms") * rms, (f0, chunks)


# =========================================================================== G5


def test_g5_matlab_thermal_noise_matches_kt0_f_fs():
    r = load_csv("G5_thermal_noise.csv")
    for i in range(r["config"].size):
        analytic = rf.thermal_noise_dbm(r["sample_rate_sps"][i], r["noise_figure_db"][i])
        assert abs(r["mean_power_dbm"][i] - analytic) <= tol("G5_db")
        # circular: power split equally between I and Q (1e7 samples, 5 sigma ~ 0.3 %)
        assert abs(r["var_re_w"][i] / r["var_im_w"][i] - 1.0) <= 5 * math.sqrt(2 / 5e6)


@pytest.mark.parametrize("backend", _backend_params())
def test_g5_sremu_noise_power_matches_comm_thermal_noise(backend):
    """sremu noise() with sigma^2 = k_B T0 F f_s over 1e7 samples (290 K, 1 MS/s): within
    0.01 dB of the comm.ThermalNoise golden."""
    r = load_csv("G5_thermal_noise.csv")
    i = int(np.flatnonzero(r["config"] == "T290_1MSps")[0])
    noise = sa.find_noise(backend)
    sigma = math.sqrt(noise_power_w(1e6, 0.0))
    w = noise((0x1234ABCD, 0x0BADF00D), 0, 10_000_000, sigma)
    assert w.dtype == np.complex64 and w.shape == (10_000_000,)
    p_dbm = 10 * math.log10(float(np.mean(np.abs(w.astype(np.complex128)) ** 2))) + 30
    assert abs(p_dbm - r["mean_power_dbm"][i]) <= tol("G5_db")


# =========================================================================== G6


def test_g6_matlab_sos_statistics_meet_st02():
    acf = load_csv("G6_fading_acf.csv")
    meta = load_meta("G6_fading_sos")["parameters"]
    fd = meta["max_doppler_hz"]
    lag = acf["lag_s"]
    m = lag <= 0.040 + 1e-12
    assert np.max(np.abs(acf["acf_re"][m] - mob.jakes_acf(lag[m], fd))) <= tol("ST02_acf_dev")
    assert np.max(np.abs(acf["j0"] - mob.jakes_acf(lag, fd))) <= 1e-12
    lcr = load_csv("G6_fading_lcr.csv")
    lcr10 = float(lcr["lcr_hz"][lcr["threshold_db"] == -10][0])
    theory = float(mob.rayleigh_lcr_hz(10 ** (-10 / 20), fd))
    assert abs(theory - ORACLE["ST02_lcr_m10db_hz"].value) <= 0.05
    assert abs(lcr10 - theory) <= 0.10 * theory
    cdf = load_csv("G6_fading_cdf.csv")
    ks_grid = np.max(np.abs(cdf["cdf"] - mob.rayleigh_cdf(10 ** (cdf["level_db"] / 20))))
    assert ks_grid <= tol("ST02_ks_envelope")


def _python_sos(rng: np.random.Generator, fd: float, fs: float, n: int, m: int = 48) -> np.ndarray:
    """Independent Clarke sum-of-sinusoids (fixed asymmetric arrival angles, random phases)."""
    t = np.arange(n) / fs
    alpha = (
        2 * np.pi * (np.arange(m) + 0.25) / m
    )  # distinct Doppler per sinusoid (no +-alpha pairs)
    phi = 2 * np.pi * rng.random(m)
    g = np.zeros(n, complex)
    for a, p in zip(alpha, phi, strict=True):
        g += np.exp(1j * (2 * np.pi * fd * np.cos(a) * t + p))
    return g / math.sqrt(m)


def test_g6_harness_statistics_accept_sos_and_reject_ar1(rng):
    """The G6/ST-02 harness passes an independent SoS process and fails an AR(1) (v2) one."""
    fd, fs, n = 50.0, 10_000.0, 400_000
    st = fading_stats(_python_sos(rng, fd, fs, n), fs)
    m = st.lags_s <= 0.040 + 1e-12
    assert np.max(np.abs(st.acf[m] - mob.jakes_acf(st.lags_s[m], fd))) <= tol("ST02_acf_dev")
    lcr_t = float(mob.rayleigh_lcr_hz(10 ** (-10 / 20), fd))
    assert abs(st.lcr_m10db_hz - lcr_t) <= 0.10 * lcr_t
    # teeth: AR(1) with the same 0.5-correlation lag as J0 is exponential, not Bessel
    rho = 0.5 ** (1 / (0.0049 * fs))
    w = (rng.normal(size=n) + 1j * rng.normal(size=n)) * math.sqrt((1 - rho**2) / 2)
    ar = np.empty(n, complex)
    ar[0] = w[0] / math.sqrt(1 - rho**2)
    for k in range(1, n):
        ar[k] = rho * ar[k - 1] + w[k]
    st_ar = fading_stats(ar, fs)
    assert np.max(np.abs(st_ar.acf[m] - mob.jakes_acf(st_ar.lags_s[m], fd))) > tol("ST02_acf_dev")


def _g6_provider():
    """WP3 diffuse SoS series provider (no frozen contract yet; see the WP4 CCR)."""
    if not sa.module_available("sremu.physics.diffuse"):
        return None
    import importlib

    return getattr(importlib.import_module("sremu.physics.diffuse"), "sos_gain", None)


@pytest.mark.xfail(
    _g6_provider() is None,
    reason="awaiting WP3: sremu.physics.diffuse.sos_gain(key, t_s, f_D, n_sinusoids) (CCR)",
    strict=False,
)
def test_g6_sremu_diffuse_sos_matches_golden_statistics():
    """sremu SoS diffuse series (f_D = 50 Hz, f_s = 10 kHz, 48 sinusoids) vs the
    comm.RayleighChannel SoS golden: ST-02 acceptance and ACF within 0.05 of the MATLAB ACF for
    lags <= 40 ms."""
    provider = _g6_provider()
    assert provider is not None
    fs, fd, n = 10_000.0, 50.0, 2_000_000
    g = provider((0x01234567, 0x89ABCDEF), np.arange(n) / fs, fd, 48)
    st = fading_stats(g, fs)
    gold = load_csv("G6_fading_acf.csv")
    m = st.lags_s <= 0.040 + 1e-12
    assert np.max(np.abs(st.acf[m] - mob.jakes_acf(st.lags_s[m], fd))) <= tol("ST02_acf_dev")
    assert np.max(np.abs(st.acf[m] - gold["acf_re"][: m.sum()])) <= tol("ST02_acf_dev")
    lcr_t = float(mob.rayleigh_lcr_hz(10 ** (-10 / 20), fd))
    assert abs(st.lcr_m10db_hz - lcr_t) <= 0.10 * lcr_t
    assert st.ks_envelope <= tol("ST02_ks_envelope")


# =========================================================================== G7


def _g7():
    meta = load_meta("G7_fm_etsi")
    r = load_csv("G7_fm_etsi.csv")
    cells = load_csv("G7_cells.csv")
    prm = meta["parameters"]
    pos = np.stack([cells["x"], cells["y"], cells["z"]], axis=1)
    pan = prm["panel"]
    n = np.asarray(pan["normal"], float)
    u = np.asarray(pan["u_axis"], float)
    idx = np.arange(pos.shape[0], dtype=np.int64)
    p = Panel(
        pos,
        np.full(pos.shape[0], pan["pitch_u_m"] * pan["pitch_v_m"]),
        n,
        u,
        np.cross(n, u),
        cells["iu"].astype(np.int64),
        cells["iv"].astype(np.int64),
        idx,
    )
    nb = 2 ** prm["n_bits"]
    alphabet = np.exp(2j * np.pi * np.arange(nb) / nb)
    return meta, r, p, cells["state"].astype(np.int64), alphabet


def test_g7_oracle_panel_and_k1_match_matlab():
    meta, r, p_ml, state, alphabet = _g7()
    pan = meta["parameters"]["panel"]
    p = make_panel(
        pan["center_m"],
        pan["normal"],
        pan["u_axis"],
        pan["n_u"],
        pan["n_v"],
        pan["pitch_u_m"],
        pan["pitch_v_m"],
    )
    assert np.max(np.abs(p.cells - p_ml.cells)) <= 1e-9  # 80 km coordinates
    fc = meta["parameters"]["fc_hz"]
    h = np.array([k1_h([0, 0, 0], [0, 0, z], p_ml, fc, alphabet[state]) for z in r["rx_z_m"]])
    assert max_rel(h, complex_col(r, "h_ris")) <= tol("G1_rel")
    hl = np.array([los_h([0, 0, 0], [0, 0, z], fc) for z in r["rx_z_m"]])
    assert max_rel(hl, complex_col(r, "h_los")) <= tol("G1_rel")


def test_g7_matlab_golden_meets_etsi_and_po_bound():
    from tests.oracles.ris import etsi_farfield_focus_h_abs, po_plate_bound_h_abs

    meta, r, _, _, _ = _g7()
    prm = meta["parameters"]
    pan = prm["panel"]
    d_t = prm["d_t_m"]
    etsi = etsi_farfield_focus_h_abs(
        pan["n_u"] * pan["n_v"], pan["pitch_u_m"], pan["pitch_v_m"], d_t, d_t - 78e3
    )
    assert abs(etsi - prm["etsi_focus_abs"]) <= 1e-12 * etsi
    h = np.abs(complex_col(r, "h_ris"))
    i_f = prm["focus_row_index_0based"]
    assert r["rx_z_m"][i_f] == 78e3
    assert abs(20 * math.log10(h[i_f] / etsi)) <= tol("G7_vs_etsi_db")
    area = pan["n_u"] * pan["n_v"] * pan["pitch_u_m"] * pan["pitch_v_m"]
    bound = np.array([po_plate_bound_h_abs(area, d_t, d_t - z) for z in r["rx_z_m"]])
    assert np.allclose(bound, r["po_bound_abs"], rtol=1e-12)
    assert np.max(20 * np.log10(h / bound)) <= tol("G7_po_bound_db")


@AWAIT_PROP
def test_g7_sremu_focused_panel_vs_etsi_and_po_bound():
    """sremu at 100 MHz: K1 at the focus within 1 dB of ETSI and never above the PO bound
    (+0.01 dB); also code-to-code with the MATLAB K1 column (1e-9)."""
    meta, r, _, state, alphabet = _g7()
    prm = meta["parameters"]
    pan = prm["panel"]
    nodes = [sa.Node("tx", [0.0, 0.0, 0.0], "tx")]
    nodes += [sa.Node(f"rx{j:03d}", [0.0, 0.0, float(z)], "rx") for j, z in enumerate(r["rx_z_m"])]
    panel = sa.PanelSpec(
        "ris0",
        pan["center_m"],
        pan["normal"],
        pan["u_axis"],
        pan["n_u"],
        pan["n_v"],
        pan["pitch_u_m"],
        pan["pitch_v_m"],
        alphabet,
    )
    res = sa.evaluate_with_state_files("g7_fm_etsi", prm["fc_hz"], nodes, [panel], {"ris0": state})
    h = np.array(
        [
            res[sa.link_id("tx", f"rx{j:03d}")].kind(PathKind.RIS_TILE)
            for j in range(r["rx_z_m"].size)
        ]
    )
    i_f = prm["focus_row_index_0based"]
    assert abs(20 * math.log10(abs(h[i_f]) / prm["etsi_focus_abs"])) <= tol("G7_vs_etsi_db")
    assert np.max(20 * np.log10(np.abs(h) / r["po_bound_abs"])) <= tol("G7_po_bound_db")
    assert max_rel(h, complex_col(r, "h_ris")) <= tol("G1_rel")


# =========================================================================== G8


def test_g8_oracle_two_ray_matches_two_ray_channel():
    r = load_csv("G8_two_ray.csv")
    fc = load_meta("G8_two_ray")["parameters"]["fc_hz"]
    for i in range(r["d_m"].size):
        pl = -float(rf.db20(rf.two_ray_h(r["d_m"][i], r["ht_m"][i], r["hr_m"][i], fc)))
        assert abs(pl - r["path_loss_db"][i]) <= tol("G8_db")
    i52 = int(np.flatnonzero((r["d_m"] == 52000) & (r["hr_m"] == 1.5))[0])
    assert abs(r["path_loss_db"][i52] - ORACLE["PH10_two_ray_52km_h1p5_db"].value) <= tol(
        "PH10_two_ray_52km_h1p5_db"
    )


@pytest.mark.xfail(
    not (PROPAGATION and sa.module_available("sremu.physics.ground")),
    reason="awaiting WP3: sremu.physics.ground not importable (ground image paths)",
    strict=False,
)
def test_g8_sremu_two_ray_with_pec_like_ground():
    """sremu LOS + ground image (eps_r 15, sigma 1e9 S/m: Gamma_H = -1 within 1e-7) vs
    twoRayChannel: path loss within 0.05 dB at 48/52/56 km and h_r = 1.5/10 m."""
    r = load_csv("G8_two_ray.csv")
    fc = load_meta("G8_two_ray")["parameters"]["fc_hz"]
    nodes = [sa.Node("tx", [0.0, 0.0, 200.0], "tx")]
    nodes += [
        sa.Node(f"rx{j:03d}", [float(r["d_m"][j]), 0.0, float(r["hr_m"][j])], "rx")
        for j in range(r["d_m"].size)
    ]
    ground = {"eps_r": 15.0, "sigma_s_per_m": 1e9, "polarization": "H", "height_m": 0.0}
    res = sa.evaluate_with_state_files("g8_two_ray", fc, nodes, [], {}, ground=ground)
    for j in range(r["d_m"].size):
        lc = res[sa.link_id("tx", f"rx{j:03d}")]
        assert set(lc.by_kind) <= {int(PathKind.LOS), int(PathKind.GROUND)}
        assert abs(-20 * math.log10(abs(lc.total)) - r["path_loss_db"][j]) <= tol("G8_db")
