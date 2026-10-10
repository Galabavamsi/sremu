"""PH-02, PH-09, PH-15, PH-16 through the full ``ScenePropagation`` path (review additions).

The unit-level PH tests drive ``k1_terms`` directly with test-supplied antenna gains. These
tests check the same closed forms through ``snapshot`` + ``combine`` so that the scene
plumbing (pose to world, sub-cell placement, antenna gain directions, failure handling,
``includes_cell_pattern``) is held to an independent oracle:

* K1 with CosPower antennas on both ends, for ``point`` and ``po_patch`` panels on an
  oblique pose: rel <= 1e-12 against the sum written out below (sub-cells placed
  independently as a uniform grid of pitch p/m, m = ceil(p / (lambda/5))).
* ``includes_cell_pattern=True`` forces ``point`` and obliquity 1 (section 5.4).
* Failures (stuck-at, dead with absorbing and structural ``gamma_dead``) give the closed form
  with the failed cells' Gamma replaced (section 5.6 step 3).
* PH-09 for the B form (angle-dependent table): a Tx behind the panel gives B == 0 exactly.
* PH-16 through the engine: a table multiplied by ``exp(j 2 pi m)`` changes h by
  <= 1e-15 |h| (m = 1..3).
"""

from __future__ import annotations

import cmath
import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PanelConfig, StreamSpec
from sremu.errors import ConfigurationError
from sremu.physics.antennas import CosPower
from sremu.physics.geometry import PanelGrid, Pose, Static
from sremu.physics.nonideal import FailureMap
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation
from sremu.physics.response import Provenance, ResponseTable, builtin
from sremu.units import C0

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
K = 2 * math.pi / LAM
LINK = "ap.tx0->mu.rx0"
TX = np.array([-1.0, 0.3, 2.0])
RX = np.array([1.4, -0.2, 1.6])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# an oblique pose: the panel normal is tilted, u is orthogonal to it
NRM = _unit(np.array([0.25, -0.15, 1.0]))
U_AX = _unit(np.cross(np.array([0.0, 1.0, 0.0]), NRM))
V_AX = np.cross(NRM, U_AX)
CEN = np.array([0.1, -0.05, 0.2])


def _cos_power(alpha: float, boresight: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """G = 2 (alpha + 1) cos^alpha on the front hemisphere (Tang 2022 Eq. 2), written out."""
    c = dirs @ boresight
    return np.where(c > 0, 2 * (alpha + 1) * np.clip(c, 0, None) ** alpha, 0.0)


def _k1_closed(
    pts: np.ndarray,
    area: float,
    gam: np.ndarray,
    *,
    ant: tuple[float, np.ndarray, float, np.ndarray] | None = None,
    obliquity: str = "sqrt",
) -> complex:
    """sum_q sqrt(G_T G_R) Gamma A o j exp(-jk(ri+rs)) / (4 pi ri rs), visible points only."""
    vi, vs = TX - pts, RX - pts
    ri, rs = np.linalg.norm(vi, axis=1), np.linalg.norm(vs, axis=1)
    ci, cs = (vi @ NRM) / ri, (vs @ NRM) / rs
    vis = (ci > 0) & (cs > 0)
    o = np.sqrt(np.clip(ci * cs, 0, None)) if obliquity == "sqrt" else np.ones_like(ci)
    if ant is None:
        g = np.ones_like(ri)
    else:
        at, bt, ar, br = ant
        g = _cos_power(at, bt, -vi / ri[:, None]) * _cos_power(ar, br, -vs / rs[:, None])
    terms = np.sqrt(g) * gam * area * o * 1j * np.exp(-1j * K * (ri + rs)) / (4 * math.pi * ri * rs)
    return complex(np.sum(np.where(vis, terms, 0.0)))


def _subcell_points(nu: int, nv: int, pitch: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform grid of (nu m) x (nv m) sub-cells of pitch p/m; parent cell iv*nu + iu."""
    s = pitch / m
    lu = (np.arange(nu * m) - (nu * m - 1) / 2) * s
    lv = (np.arange(nv * m) - (nv * m - 1) / 2) * s
    gv, gu = np.meshgrid(lv, lu, indexing="ij")
    pts = CEN + gu.ravel()[:, None] * U_AX + gv.ravel()[:, None] * V_AX
    iu_s = np.tile(np.arange(nu * m), nv * m)
    iv_s = np.repeat(np.arange(nv * m), nu * m)
    return pts, (iv_s // m) * nu + iu_s // m


def _ris_h(model: ScenePropagation, states: np.ndarray, pid: str = "ris0") -> complex:
    snap = model.snapshot(Fraction(0))
    lp = model.combine(snap, {pid: PanelConfig(pid, states.astype(np.int16))})[LINK]
    ris = lp.tile >= 0
    return complex(np.sum(lp.amp[ris] * np.exp(-2j * math.pi * FC * lp.delay_s[ris])))


def _scene(panel: PanelModel, ant: tuple[float, np.ndarray, float, np.ndarray] | None = None):  # type: ignore[no-untyped-def]
    a_tx = None if ant is None else CosPower(ant[0], ant[1])
    a_rx = None if ant is None else CosPower(ant[2], ant[3])
    kw_t = {} if a_tx is None else {"antenna": a_tx}
    kw_r = {} if a_rx is None else {"antenna": a_rx}
    return ScenePropagation(
        [
            NodeModel("ap", Static(TX), streams=(StreamSpec("ap.tx0", "tx", FC, "1e6"),), **kw_t),  # type: ignore[arg-type]
            NodeModel("mu", Static(RX), streams=(StreamSpec("mu.rx0", "rx", FC, "1e6"),), **kw_r),  # type: ignore[arg-type]
        ],
        [panel],
    )


@pytest.mark.parametrize(("pitch_wl", "model", "m_sub"), [(0.5, "point", 1), (0.9, "po_patch", 5)])
def test_ph02_scene_k1_with_cospower_antennas(pitch_wl: float, model: str, m_sub: int):
    nu, nv = 6, 5
    pitch = pitch_wl * LAM
    ant = (4.0, _unit(np.array([0.3, -0.1, -1.0])), 2.5, _unit(np.array([-0.5, 0.2, -1.0])))
    grid = PanelGrid(Pose(CEN, NRM, U_AX), nu, nv, pitch, pitch)
    scene = _scene(PanelModel("ris0", grid, builtin("ideal_bbit(2)")), ant)
    quad = scene.quadrature("ris0")
    assert scene.cell_model("ris0") == model
    # section 5.4: ceil(p / Delta) sub-cells per axis, Delta = lambda/5 (Richardson level 0)
    assert (quad.m_u, quad.m_v, quad.level) == (m_sub, m_sub, 0)
    states = np.arange(nu * nv) % 4
    pts, parent = _subcell_points(nu, nv, pitch, m_sub)
    gam = np.exp(2j * np.pi * states[parent] / 4)
    expect = _k1_closed(pts, pitch * pitch / m_sub**2, gam, ant=ant)
    got = _ris_h(scene, states)
    assert abs(got - expect) <= 1e-12 * abs(expect)
    # LOS: sqrt(G_T(r) G_R(-r)) lambda / (4 pi d), propagation phase excluded
    d = float(np.linalg.norm(RX - TX))
    r = (RX - TX) / d
    g_los = _cos_power(ant[0], ant[1], r[None])[0] * _cos_power(ant[2], ant[3], -r[None])[0]
    pb = scene.snapshot(Fraction(0)).paths[LINK]
    assert abs(pb.amp[0] - math.sqrt(g_los) * LAM / (4 * math.pi * d)) <= 1e-12 * abs(pb.amp[0])


def test_ph02_scene_includes_cell_pattern_forces_point_and_unit_obliquity():
    prov = Provenance(
        "measured",
        "single-cell bistatic measurement (pattern included)",
        "review test",
        0.0,
        "co",
        True,  # includes_cell_pattern
        0.0,
        0.0,
        None,
        "test",  # type: ignore[arg-type]
        {"date": "2026-10-10"},
    )
    g = np.array([0.8 * cmath.exp(1j * math.radians(30)), 0.6 * cmath.exp(-1j * math.radians(100))])
    table = ResponseTable(["a", "b"], [FC], [0.0], g.reshape(2, 1, 1, 1), prov, validity_bw_hz=1e8)
    nu, nv, pitch = 5, 4, 0.7 * LAM  # > lambda/2: auto would pick po_patch without the flag
    grid = PanelGrid(Pose(CEN, NRM, U_AX), nu, nv, pitch, pitch)
    scene = _scene(PanelModel("ris0", grid, table))
    assert scene.cell_model("ris0") == "point"
    states = np.arange(nu * nv) % 2
    pts, parent = _subcell_points(nu, nv, pitch, 1)
    expect = _k1_closed(pts, pitch * pitch, g[states[parent]], obliquity="none")
    assert abs(_ris_h(scene, states) - expect) <= 1e-12 * abs(expect)


@pytest.mark.parametrize("gamma_dead", [0j, -1 + 0j])
def test_ph15_scene_failures_follow_closed_form(gamma_dead: complex):
    nu, nv, pitch = 6, 5, LAM / 2
    n = nu * nv
    stuck = np.full(n, -1, np.int16)
    stuck[[2, 11, 17]] = [3, 0, 1]
    dead = np.zeros(n, bool)
    dead[[5, 11, 23, 29]] = True  # cell 11 is stuck and dead: dead wins
    grid = PanelGrid(Pose(CEN, NRM, U_AX), nu, nv, pitch, pitch)
    panel = PanelModel(
        "ris0",
        grid,
        builtin("ideal_bbit(2)"),
        failures=FailureMap(stuck, dead),
        gamma_dead=gamma_dead,
    )
    scene = _scene(panel)
    cmd = (np.arange(n) * 7) % 4
    cfg = scene.pipeline("ris0").from_states(cmd)
    st = cmd.copy()
    st[[2, 17]] = [3, 1]
    gam = np.exp(2j * np.pi * st / 4)
    gam[dead] = gamma_dead
    assert np.array_equal(cfg.state[~dead], st[~dead]) and np.all(cfg.state[dead] == -1)
    pts, parent = _subcell_points(nu, nv, pitch, 1)
    expect = _k1_closed(pts, pitch * pitch, gam[parent])
    snap = scene.snapshot(Fraction(0))
    lp = scene.combine(snap, {"ris0": cfg})[LINK]
    ris = lp.tile >= 0
    got = complex(np.sum(lp.amp[ris] * np.exp(-2j * math.pi * FC * lp.delay_s[ris])))
    assert abs(got - expect) <= 1e-12 * abs(expect)


def _angle_table() -> ResponseTable:
    th = np.arange(-80.0, 81.0, 10.0)
    ti, ts = np.meshgrid(th, th, indexing="ij")
    g = np.stack(
        [
            0.9 * np.exp(1j * np.radians(150 + 0.2 * (ti + ts))),
            0.8 * np.exp(1j * np.radians(-20 - 0.1 * (ti + ts))),
        ]
    )[:, None]
    prov = Provenance(
        "fullwave",
        "smooth synthetic",
        "review test",
        0.0,
        "co",
        False,
        0.0,
        0.0,
        None,
        "test",
        {"date": "2026-10-10"},  # type: ignore[arg-type]
    )
    return ResponseTable(["s0", "s1"], [FC], th, g, prov)


def test_ph09_b_form_tx_behind_panel_is_exactly_zero():
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 8, 6, LAM / 2, LAM / 2)
    scene = ScenePropagation(
        [
            NodeModel(
                "ap",
                Static(np.array([0.5, 0.0, -2.0])),
                streams=(StreamSpec("ap.tx0", "tx", FC, "1e6"),),
            ),  # type: ignore[arg-type]
            NodeModel(
                "mu",
                Static(np.array([-0.5, 0.2, 2.0])),
                streams=(StreamSpec("mu.rx0", "rx", FC, "1e6"),),
            ),  # type: ignore[arg-type]
        ],
        [PanelModel("ris0", grid, _angle_table())],
    )
    rc = scene.snapshot(Fraction(0)).ris["ris0"]
    assert not rc.separable and rc.B is not None and np.all(rc.B == 0)


def test_coupling_hook_and_structural_dead_refused_for_angle_tables():
    """Section 5.6: the coupling hook and a structural gamma_dead need the separable form;
    with an angle-dependent table both are refused when the model is built (never dropped
    silently at combine time)."""

    class Hook:
        def delta(self, state):  # type: ignore[no-untyped-def]
            return np.zeros(np.asarray(state).shape, complex)

    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 8, 6, LAM / 2, LAM / 2)
    with pytest.raises(ConfigurationError):
        _scene(PanelModel("ris0", grid, _angle_table(), coupling=Hook()))
    with pytest.raises(ConfigurationError):
        _scene(PanelModel("ris0", grid, _angle_table(), gamma_dead=-1 + 0j))


@pytest.mark.parametrize("m", [1, 2, 3])
def test_ph16_table_times_exp_j2pi_m_through_engine(m: int):
    nu, nv, pitch = 8, 6, LAM / 2
    grid = PanelGrid(Pose(CEN, NRM, U_AX), nu, nv, pitch, pitch)
    base = builtin("tang2022_ris1_27ghz")
    g0 = base.gamma[:, 0, 0, 0]
    prov = Provenance(
        "analytic",
        "Tang RIS1 alphabet as an analytic stand-in",
        "review test",
        0.0,
        "co",
        False,
        0.0,
        0.0,
        None,
        "test",
        {"date": "2026-10-10"},  # type: ignore[arg-type]
    )

    def table(gam: np.ndarray) -> ResponseTable:
        return ResponseTable(
            ["0", "1"], [0.0], [0.0], gam.reshape(2, 1, 1, 1), prov, frequency_independent=True
        )

    states = (np.arange(nu * nv) * 5 // 3) % 2
    h_ref = scene_h = None
    for gam in (g0, g0 * cmath.rect(1.0, 2 * math.pi * m)):
        scene = _scene(PanelModel("ris0", grid, table(gam)))
        snap = scene.snapshot(Fraction(0))
        scene_h = scene.evaluate(snap, {"ris0": PanelConfig("ris0", states.astype(np.int16))})[LINK]
        if h_ref is None:
            h_ref = scene_h
    assert h_ref is not None and scene_h is not None
    assert abs(scene_h - h_ref) <= 1e-15 * abs(h_ref)


def test_ph31_scene_bandwidth_exceeded_at_build():
    """Section 5.5: a linked TX whose occupied band exceeds validity_bw_hz is refused when
    the model is built (lab table: 20 MHz; default occupied band 0.8 f_s = 24.576 MHz)."""
    from sremu.errors import BandwidthExceeded

    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 8, 6, LAM / 2, LAM / 2)
    lab = builtin("lab_binary_3p5ghz", assume_mag={"sigma1": 1.0, "sigma2": 1.0})

    def scene(rate: str, bw: float | None) -> ScenePropagation:
        tx = StreamSpec("ap.tx0", "tx", FC, rate)  # type: ignore[arg-type]
        rx = StreamSpec("mu.rx0", "rx", FC, rate)  # type: ignore[arg-type]
        return ScenePropagation(
            [
                NodeModel("ap", Static(TX), streams=(tx,), occupied_bw_hz={"ap.tx0": bw}),
                NodeModel("mu", Static(RX), streams=(rx,)),
            ],
            [PanelModel("ris0", grid, lab)],
        )

    with pytest.raises(BandwidthExceeded):
        scene("30.72e6", None)
    scene("30.72e6", 18e6)  # declared 18 MHz <= 20 MHz
    scene("1e6", None)  # 0.8 MHz
