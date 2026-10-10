"""``ScenePropagation`` against the ``PropagationModel`` contract (sections 3.4, 4.4, 5.1-5.6).

Every expected value is an independent closed form: LOS Friis terms, the K1 sum written out
term by term (``tests/physics/_closed_forms.py``), path lengths through tile reference
points, exact closest approaches of constant-velocity motion.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.contracts import PanelConfig, PathKind, PropagationModel, StreamSpec
from sremu.errors import ConfigurationError, GeometryError, UnknownTarget, ValidityWarning
from sremu.physics.geometry import ConstantVelocity, PanelGrid, Pose, Static
from sremu.physics.los import Occluder
from sremu.physics.nonideal import FailureMap
from sremu.physics.propagation import NodeModel, PanelModel, RisHopSource, ScenePropagation
from sremu.physics.response import Provenance, ResponseTable, builtin
from sremu.units import C0
from tests.physics._closed_forms import k1_point_sum

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
LINK = "ap.tx0->mu.rx0"


def _streams(node: str, direction: str) -> tuple[StreamSpec, ...]:
    return (StreamSpec(f"{node}.{direction}0", direction, FC, "1e6"),)  # type: ignore[arg-type]


def _scene(
    table=None,
    *,
    tiles=(1, 1),
    tx=(-1.0, 0.3, 2.0),
    rx=(1.4, -0.2, 1.6),  # type: ignore[no-untyped-def]
    rx_traj=None,
    **panel_kw,
) -> ScenePropagation:
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 10, 6, LAM / 2, LAM / 2)
    panel = PanelModel(
        "ris0", grid, table or builtin("ideal_bbit(2)"), render_tiles=tiles, **panel_kw
    )
    return ScenePropagation(
        [
            NodeModel("ap", Static(np.array(tx)), streams=_streams("ap", "tx")),
            NodeModel("mu", rx_traj or Static(np.array(rx)), streams=_streams("mu", "rx")),
        ],
        [panel],
        horizon_s=10,
    )


def _absolute_h(model: ScenePropagation, t: Fraction, states: np.ndarray) -> complex:
    snap = model.snapshot(t)
    lp = model.combine(snap, {"ris0": PanelConfig("ris0", states.astype(np.int16))})[LINK]
    return complex(np.sum(lp.amp * np.exp(-2j * math.pi * FC * lp.delay_s)))


def test_contract_isinstance_and_links():
    m = _scene()
    assert isinstance(m, PropagationModel)
    assert m.links() == (LINK,)


@pytest.mark.parametrize("tiles", [(1, 1), (2, 3), (5, 2)])
def test_contract_tile_relative_phase_reconstructs_absolute_k1(
    tiles: tuple[int, int], rng: np.random.Generator
):
    m = _scene(tiles=tiles)
    states = rng.integers(0, 4, 60)
    h = _absolute_h(m, Fraction(0), states)
    grid = m.panel("ris0").grid
    gam = np.exp(2j * np.pi * states / 4)
    tx, rx = np.array([-1.0, 0.3, 2.0]), np.array([1.4, -0.2, 1.6])
    d = float(np.linalg.norm(rx - tx))
    los = LAM / (4 * math.pi * d) * np.exp(-2j * math.pi * d / LAM)
    ris = k1_point_sum(tx, rx, grid.cell_centres(), grid.cell_area_m2, FC, gam)
    assert abs(h - (los + ris)) <= 1e-12 * abs(los + ris)
    # re-tiling a model (render tiles chosen by the T_u/tile policy) leaves h unchanged
    m2 = m.with_render_tiles({"ris0": (3, 3)})
    assert m2.snapshot(Fraction(0)).ris["ris0"].n_tiles == 9
    assert abs(_absolute_h(m2, Fraction(0), states) - h) <= 1e-12 * abs(h)


def test_contract_snapshot_fields_match_closed_forms():
    m = _scene(tiles=(2, 1))
    snap = m.snapshot(Fraction(1, 3))
    assert snap.t == Fraction(1, 3)
    pb = snap.paths[LINK]
    tx, rx = np.array([-1.0, 0.3, 2.0]), np.array([1.4, -0.2, 1.6])
    d = float(np.linalg.norm(rx - tx))
    assert pb.kind.tolist() == [PathKind.LOS] and pb.delay_s[0] == pytest.approx(d / C0, rel=1e-15)
    rc = snap.ris["ris0"]
    assert rc.rows == ((LINK, 0),) and rc.n_tiles == 2 and rc.separable
    cells = m.panel("ris0").grid.cell_centres()
    for k in range(2):
        ref = cells[rc.tile_of_cell == k].mean(axis=0)
        lref = np.linalg.norm(ref - tx) + np.linalg.norm(ref - rx)
        assert abs(rc.tile_delay_s[0, k] - lref / C0) <= 1e-15 * lref / C0 * 4
    # coefficient phases are relative to the tile reference: C * exp(-j k L_ref) is absolute
    lref_c = rc.tile_delay_s[0][rc.tile_of_cell] * C0
    absolute = rc.C[0] * np.exp(-2j * math.pi * lref_c / LAM)
    expect = np.array([k1_point_sum(tx, rx, cells[i : i + 1], LAM**2 / 4, FC) for i in range(60)])
    assert np.max(np.abs(absolute - expect)) <= 1e-12 * np.max(np.abs(expect))


def test_contract_combine_linear_in_gamma_and_disabled_panel(rng: np.random.Generator):
    m = _scene()
    snap = m.snapshot(Fraction(0))
    s = rng.integers(0, 4, 60).astype(np.int16)
    mask = rng.random(60) < 0.5
    full = m.combine(snap, {"ris0": PanelConfig("ris0", s)})[LINK]
    a = m.combine(snap, {"ris0": PanelConfig("ris0", np.where(mask, s, -1).astype(np.int16))})[LINK]
    b = m.combine(snap, {"ris0": PanelConfig("ris0", np.where(mask, -1, s).astype(np.int16))})[LINK]
    dead = m.combine(snap, {"ris0": PanelConfig("ris0", np.full(60, -1, np.int16))})[LINK]
    ris = full.tile >= 0
    assert np.max(np.abs(a.amp[ris] + b.amp[ris] - full.amp[ris] - dead.amp[ris])) <= 1e-15 * max(
        1e-300, float(np.max(np.abs(full.amp[ris])))
    )
    assert np.all(dead.amp[ris] == 0)
    off = m.combine(snap, {"ris0": PanelConfig("ris0", s, enabled=False)})[LINK]
    none = m.combine(snap, {})[LINK]
    for lp in (off, none):
        assert lp.n_paths == full.n_paths and np.all(lp.amp[ris] == 0)
        assert np.array_equal(lp.delay_s, full.delay_s) and np.array_equal(lp.tile, full.tile)
    with pytest.raises(ConfigurationError):
        m.combine(snap, {"ris0": PanelConfig("ris0", np.full(60, 4, np.int16))})


def test_contract_angle_dependent_table_uses_b_form(rng: np.random.Generator):
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
        "synthetic",
        "test",
        0.0,
        "co",
        False,
        0.0,
        0.0,
        None,
        "t",  # type: ignore[arg-type]
        {"date": "2026-10-10"},
    )
    table = ResponseTable(["s0", "s1"], [FC], th, g, prov)
    m = _scene(table)
    snap = m.snapshot(Fraction(0))
    assert not snap.ris["ris0"].separable
    s = rng.integers(0, 2, 60)
    h = _absolute_h(m, Fraction(0), s)
    tx, rx = np.array([-1.0, 0.3, 2.0]), np.array([1.4, -0.2, 1.6])
    cells = m.panel("ris0").grid.cell_centres()
    vi, vs = tx - cells, rx - cells
    thi = np.degrees(np.arctan2(vi[:, 0], vi[:, 2]))  # signed by the u (= x) component
    ths = np.degrees(np.arctan2(vs[:, 0], vs[:, 2]))
    gam = np.array([table.query(FC, [thi[n]], [ths[n]])[s[n], 0] for n in range(60)])
    d = float(np.linalg.norm(rx - tx))
    expect = LAM / (4 * math.pi * d) * np.exp(-2j * math.pi * d / LAM) + k1_point_sum(
        tx, rx, cells, LAM**2 / 4, FC, gam
    )
    assert abs(h - expect) <= 1e-12 * abs(expect)
    assert snap.diag["max_off_plane_deg/ris0"] > 0  # the Rx is off the (u, n) plane


def test_contract_ris_delay_and_motion_bounds():
    traj = ConstantVelocity(np.array([5.0, 3.0, 2.0]), np.array([-1.0, 0.0, 0.0]))
    m = _scene(rx_traj=traj, tx=(0.0, 0.0, 2.0))
    t = Fraction(7, 2)
    p = np.array([5.0 - 3.5, 3.0, 2.0])
    assert m.ris_delay_s("ris0", "mu.rx0", t) == pytest.approx(np.linalg.norm(p) / C0, rel=1e-15)
    with pytest.raises(UnknownTarget):
        m.ris_delay_s("ris0", "nobody.rx0", t)
    mb = m.motion_bounds()
    assert not mb.static
    assert mb.v_rel_max_mps[LINK] == 1.0
    # closest approach of rx(t) = (5 - t, 3, 2) to the static Tx (0, 0, 2) over [0, 10]: 3 m
    assert mb.r_min_m[LINK] == pytest.approx(3.0, abs=1e-12)
    # the static Tx 2 m above the panel centre is the closest linked node (exact, no margin)
    assert mb.panel_r_min_m["ris0"] == pytest.approx(2.0, abs=1e-12)
    assert mb.panel_v_rel_max_mps["ris0"] == 1.0
    # moving Rx alone: rectangle |u| <= 2.5 lambda, |v| <= 1.5 lambda; sampled minimum minus
    # the margin v dt / 2 is a lower bound of the exact distance hypot(3 - 1.5 lambda, 2)
    m_rx = _scene(rx_traj=traj, tx=(0.0, 30.0, 2.0))
    got = m_rx.motion_bounds().panel_r_min_m["ris0"]
    exact = math.hypot(3.0 - 1.5 * LAM, 2.0)
    assert exact - 1.0 * (10 / 16) / 2 - 1e-12 <= got <= exact


def test_contract_guards_and_occluders():
    with pytest.raises(GeometryError):  # a node closer than one wavelength to a cell
        _scene(tx=(0.0, 0.0, 0.5 * LAM)).snapshot(Fraction(0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _scene(tx=(0.0, 0.0, 2 * LAM)).snapshot(Fraction(0))
    assert any(issubclass(x.category, ValidityWarning) for x in w)
    m = _scene()
    wall = Occluder(
        "p", np.array([[0.2, -2, 0], [0.2, 2, 0], [0.2, 2, 4], [0.2, -2, 4]]), "partition", 20.0
    )
    m2 = ScenePropagation(list(m.nodes.values()), [m.panel("ris0")], occluders=[wall])
    a1 = m.snapshot(Fraction(0)).paths[LINK].amp[0]
    a2 = m2.snapshot(Fraction(0)).paths[LINK].amp[0]
    assert abs(a2 / a1 - 0.1) <= 1e-15  # exactly -20 dB, no phase change


def test_contract_failures_initial_config_and_capabilities():
    fm = FailureMap(np.array([-1] * 58 + [1, -1], np.int16), np.array([False] * 59 + [True]))
    m = _scene(failures=fm, initial_state=np.zeros(60, np.int16))
    cfg = m.initial_config("ris0")
    assert cfg.state[58] == 1 and cfg.state[59] == -1 and cfg.state[:58].tolist() == [0] * 58
    caps = m.capabilities("ris0")
    assert (
        caps.n_elements == 60 and caps.layout == (10, 6) and caps.states == ("s0", "s1", "s2", "s3")
    )
    assert caps.gamma_source.startswith("analytic:")
    pipe = m.pipeline("ris0")
    assert pipe.from_states(np.full(60, 2, np.int16)).state[59] == -1


def test_contract_mechanism_hook_adds_paths_and_hop_rows():
    m = _scene()

    class Mech:
        def link_paths(self, ctx):  # type: ignore[no-untyped-def]
            return [(0.5 * 1e-3, 1e-8, PathKind.GROUND)]

        def los_factor(self, ctx):  # type: ignore[no-untyped-def]
            return 0.5

        def ris_hops(self, ctx, panel):  # type: ignore[no-untyped-def]
            img = ctx.p_tx * np.array([1.0, 1.0, 1.0])
            return [RisHopSource(1, img, ctx.p_rx, lambda pts: -np.ones(len(pts), complex))]

    m2 = ScenePropagation(list(m.nodes.values()), [m.panel("ris0")], mechanisms=[Mech()])
    s1, s2 = m.snapshot(Fraction(0)), m2.snapshot(Fraction(0))
    assert s2.paths[LINK].kind.tolist() == [PathKind.LOS, PathKind.GROUND]
    assert s2.paths[LINK].amp[0] == 0.5 * s1.paths[LINK].amp[0]
    rc = s2.ris["ris0"]
    assert rc.rows == ((LINK, 0), (LINK, 1))
    assert rc.C is not None and np.max(np.abs(rc.C[1] + rc.C[0])) <= 1e-15 * np.max(np.abs(rc.C[0]))
    lp = m2.combine(s2, {"ris0": PanelConfig("ris0", np.zeros(60, np.int16))})[LINK]
    assert np.sum(lp.tile >= 0) == 2  # one tile path per row


def test_contract_structural_gap_gamma_path():
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 10, 6, LAM / 2, LAM / 2, 2, 1, 0.01)
    panel = PanelModel("ris0", grid, builtin("ideal_bbit(1)"), gap_gamma=-1.0)
    m = ScenePropagation(
        [
            NodeModel("ap", Static(np.array([-1.0, 0.3, 2.0])), streams=_streams("ap", "tx")),
            NodeModel("mu", Static(np.array([1.4, -0.2, 1.6])), streams=_streams("mu", "rx")),
        ],
        [panel],
    )
    pb = m.snapshot(Fraction(0)).paths[LINK]
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.RIS_TILE]
    tx, rx = np.array([-1.0, 0.3, 2.0]), np.array([1.4, -0.2, 1.6])
    strip_u = grid.gap_strips()[0]
    assert strip_u[1] - strip_u[0] == pytest.approx(0.01)
    # closed form: K1 over the gap strip with Gamma = -1, relative to the panel-centre path
    nv = math.ceil((strip_u[3] - strip_u[2]) / (0.2 * LAM))
    vv = strip_u[2] + (np.arange(nv) + 0.5) * (strip_u[3] - strip_u[2]) / nv
    pts = np.column_stack([np.full(nv, 0.5 * (strip_u[0] + strip_u[1])), vv, np.zeros(nv)])
    area = 0.01 * (strip_u[3] - strip_u[2]) / nv
    lref = np.linalg.norm(tx) + np.linalg.norm(rx)
    expect = -k1_point_sum(tx, rx, pts, area, FC) * np.exp(2j * math.pi * lref / LAM)
    assert abs(pb.amp[1] - expect) <= 1e-12 * abs(expect)
    assert pb.delay_s[1] == pytest.approx(lref / C0, rel=1e-15)
    # combine: an enabled panel keeps the gap path; a disabled (absent, RQ-12) or
    # unconfigured panel has neither tile nor gap reflection; LOS and the path set are kept
    snap = m.snapshot(Fraction(0))
    on = m.combine(snap, {"ris0": PanelConfig("ris0", np.zeros(60, np.int16))})[LINK]
    off = m.combine(snap, {"ris0": PanelConfig("ris0", np.zeros(60, np.int16), enabled=False)})
    none = m.combine(snap, {})[LINK]
    assert on.amp[1] == pb.amp[1]
    for lp in (off[LINK], none):
        assert lp.amp[0] == pb.amp[0] and lp.amp[1] == 0 and np.all(lp.amp[lp.tile >= 0] == 0)
        assert np.array_equal(lp.delay_s, on.delay_s) and np.array_equal(lp.kind, on.kind)


def test_contract_from_scenario_rejects_environment_without_wp3():
    doc = {
        "nodes": [],
        "environment": {"ground": {"eps_r": 15, "sigma_s_per_m": 1e-3, "polarization": "H"}},
    }
    with pytest.raises(ConfigurationError):
        ScenePropagation.from_scenario(doc)


LAB_A = Path(__file__).resolve().parents[2] / "scenarios" / "lab_like_A.yaml"


@pytest.mark.skipif(not LAB_A.is_file(), reason="WP1 scenarios/lab_like_A.yaml not present")
def test_contract_from_scenario_file_and_wp1_model_agree():
    from sremu.physics.propagation import from_scenario
    from sremu.scenario.load import read_document
    from sremu.scenario.models import Scenario

    m1 = from_scenario(LAB_A)
    m2 = from_scenario(Scenario.model_validate(read_document(LAB_A)), base_dir=LAB_A.parent)
    assert m1.links() == m2.links() == ("ap.tx0->rx1.rx0", "ap.tx0->rx2.rx0")
    s1, s2 = m1.snapshot(Fraction(0)), m2.snapshot(Fraction(0))
    h1 = m1.evaluate(s1, {"ris0": m1.initial_config("ris0")})
    h2 = m2.evaluate(s2, {"ris0": m2.initial_config("ris0")})
    assert h1 == h2
    # the LOS term of each link is Friis at the scenario distances (closed form)
    for lk, pb in s1.paths.items():
        tx, rx = m1.link_streams(lk)
        d = float(np.linalg.norm(m1.node_position(rx.node, 0) - m1.node_position(tx.node, 0)))
        assert abs(pb.amp[0] - LAM / (4 * math.pi * d)) <= 1e-12 * abs(pb.amp[0])
    diag = m1.diagnostics()
    assert diag["static"] and diag["panels"]["ris0"]["cell_model"] in ("point", "po_patch")
    tau = diag["panels"]["ris0"]["tau_ris_rx_samples"]["rx1.rx0"]
    p_rx1 = m1.node_position("rx1", 0)
    assert tau == pytest.approx(np.linalg.norm(p_rx1) / C0 * 1e6, rel=1e-12)


def test_contract_coupling_hook_adds_c_delta(rng: np.random.Generator):
    class Hook:
        def delta(self, state):  # type: ignore[no-untyped-def]
            return 0.05j * (np.asarray(state) >= 0)

    m0 = _scene()
    m1 = _scene(coupling=Hook())
    s = rng.integers(0, 4, 60).astype(np.int16)
    s[3] = -1
    snap = m0.snapshot(Fraction(0))
    a0 = m0.tile_amplitudes(snap, {"ris0": PanelConfig("ris0", s)})["ris0"]
    a1 = m1.tile_amplitudes(snap, {"ris0": PanelConfig("ris0", s)})["ris0"]
    c = snap.ris["ris0"].C
    expected = a0 + np.sum(c * 0.05j * (s >= 0), axis=1)[:, None]
    assert np.max(np.abs(a1 - expected)) <= 1e-15 * np.max(np.abs(a0))
