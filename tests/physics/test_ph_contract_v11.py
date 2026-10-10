"""``ScenePropagation`` against contract 1.1 (lead decisions D11, D14, D17, D18).

Oracles are independent closed forms: the D17 cell-centre formula written out, tile
reference points as centroids of the D17 render-tile cells, Friis/K1 single-cell sums
(``tests/physics/_closed_forms.py``), and the ``sremu.rng`` derivation itself (D14).
"""

from __future__ import annotations

import math
import sys
import types
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu import rng as sremu_rng
from sremu.contracts import PanelConfig, PanelGeometry, PathKind, PropagationModel, StreamSpec
from sremu.errors import ConfigurationError, UnknownTarget
from sremu.physics import propagation as prop
from sremu.physics.geometry import (
    ConstantVelocity,
    ExternalTrajectory,
    PanelGrid,
    Pose,
    Static,
    keyed_generator,
    trajectory_from_spec,
)
from sremu.physics.nonideal import FailureMap
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation, from_scenario
from sremu.physics.quadrature import LinkSample, representative_links, richardson_link_key
from sremu.physics.response import builtin
from sremu.units import C0
from tests.physics._closed_forms import k1_point_sum

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
LINK = "ap.tx0->mu.rx0"
TX = np.array([-0.7, 0.4, 1.8])
RX = np.array([0.9, -0.3, 1.5])


def _streams(node: str, direction: str, fc: float = FC) -> tuple[StreamSpec, ...]:
    return (StreamSpec(f"{node}.{direction}0", direction, fc, "1e6"),)  # type: ignore[arg-type]


def _scene(panel: PanelModel, rx_traj=None) -> ScenePropagation:  # type: ignore[no-untyped-def]
    return ScenePropagation(
        [
            NodeModel("ap", Static(TX), streams=_streams("ap", "tx")),
            NodeModel("mu", rx_traj or Static(RX), streams=_streams("mu", "rx")),
        ],
        [panel],
        horizon_s=10,
    )


# panel in the z = 0 plane with u = +y, so v = n x u = -x (checks the v-axis sign of D17)
POSE = Pose([0.05, -0.02, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
N_U, N_V, TU, TV, GAP = 12, 6, 3, 2, 0.004
PITCH = LAM / 2


def _gapped_grid() -> PanelGrid:
    return PanelGrid(POSE, N_U, N_V, PITCH, PITCH, TU, TV, GAP)


def _d17_centre(iu: int, iv: int) -> np.ndarray:
    """D17 written out: u(iu) = (iu + 1/2) p + tu gap - extent_u/2 (likewise v)."""
    ext_u = N_U * PITCH + (TU - 1) * GAP
    ext_v = N_V * PITCH + (TV - 1) * GAP
    u = (iu + 0.5) * PITCH + (iu // (N_U // TU)) * GAP - ext_u / 2
    v = (iv + 0.5) * PITCH + (iv // (N_V // TV)) * GAP - ext_v / 2
    c = np.array([0.05, -0.02, 0.0])
    return c + u * np.array([0.0, 1.0, 0.0]) + v * np.array([-1.0, 0.0, 0.0])


# --------------------------------------------------------------------------- D11 / D17


def test_d11_d17_panel_geometry_is_the_contract_type():
    m = _scene(PanelModel("ris0", _gapped_grid(), builtin("ideal_bbit(1)")))
    assert isinstance(m, PropagationModel)
    pg = m.panel_geometry("ris0")
    assert isinstance(pg, PanelGeometry)
    assert (pg.n_u, pg.n_v, pg.n_tiles_u, pg.n_tiles_v) == (N_U, N_V, TU, TV)
    assert pg.gap_m == GAP and pg.pitch_u_m == PITCH and pg.n_quad == N_U * N_V  # point cells
    assert pg.v_axis == pytest.approx((-1.0, 0.0, 0.0), abs=1e-15)
    expect = np.stack([_d17_centre(n % N_U, n // N_U) for n in range(N_U * N_V)])
    assert np.max(np.abs(pg.cell_centres_m() - expect)) <= 1e-15
    assert np.max(np.abs(m.panel("ris0").grid.cell_centres() - expect)) <= 1e-15
    assert np.max(np.abs(m.quadrature("ris0").points - expect)) <= 1e-15
    assert np.array_equal(pg.physical_tile_of_cell(), m.panel("ris0").grid.physical_tile_of_cell())
    with pytest.raises(UnknownTarget):
        m.panel_geometry("nope")
    # a po_patch panel reports its quadrature size Q after the Richardson check
    big = PanelGrid(POSE, 4, 4, 0.8 * LAM, 0.8 * LAM)
    mp = _scene(PanelModel("ris1", big, builtin("ideal_bbit(1)")))
    assert mp.cell_model("ris1") == "po_patch"
    assert mp.panel_geometry("ris1").n_quad == mp.quadrature("ris1").n_points > 16


@pytest.mark.parametrize(("iu", "iv"), [(0, 0), (5, 0), (6, 2), (11, 5), (3, 4)])
def test_d17_state_index_n_drives_cell_iv_nu_plus_iu(iu: int, iv: int):
    """One live cell n = iv*n_u + iu (Gamma = 1), all others dead (absorbing): the RIS term
    equals K1 of a single cell at the D17 position (rel <= 1e-12)."""
    m = _scene(PanelModel("ris0", _gapped_grid(), builtin("ideal_bbit(1)")))
    n = iv * N_U + iu
    state = np.full(N_U * N_V, -1, np.int16)
    state[n] = 0
    snap = m.snapshot(Fraction(0))
    lp = m.combine(snap, {"ris0": PanelConfig("ris0", state)})[LINK]
    ris = lp.tile >= 0
    h = complex(np.sum(lp.amp[ris] * np.exp(-2j * math.pi * FC * lp.delay_s[ris])))
    ref = k1_point_sum(TX, RX, _d17_centre(iu, iv)[None, :], PITCH * PITCH, FC)
    assert abs(h - ref) <= 1e-12 * abs(ref)


def test_d11_link_fc_hz_per_link():
    nodes = [
        NodeModel(
            "ap",
            Static(TX),
            streams=(
                StreamSpec("ap.tx0", "tx", 3.5e9, "1e6"),  # type: ignore[arg-type]
                StreamSpec("ap.tx1", "tx", 28e9, "1e6"),  # type: ignore[arg-type]
            ),
        ),
        NodeModel(
            "mu",
            Static(RX),
            streams=(
                StreamSpec("mu.rx0", "rx", 3.5e9 + 1e5, "1e6"),  # type: ignore[arg-type]
                StreamSpec("mu.rx1", "rx", 28e9, "1e6"),  # type: ignore[arg-type]
            ),
        ),
    ]
    m = ScenePropagation(nodes)
    assert m.links() == ("ap.tx0->mu.rx0", "ap.tx1->mu.rx1")
    assert m.link_fc_hz("ap.tx0->mu.rx0") == 3.5e9  # the TX carrier, not the RX one
    assert m.link_fc_hz("ap.tx1->mu.rx1") == 28e9


def _tile_refs(tiles_u: int, tiles_v: int) -> np.ndarray:
    """Render tile k = tv*tiles_u + tu with tu = iu*tiles_u//n_u: centroid of its cells."""
    cells = np.stack([_d17_centre(n % N_U, n // N_U) for n in range(N_U * N_V)])
    iu, iv = np.arange(N_U * N_V) % N_U, np.arange(N_U * N_V) // N_U
    k = (iv * tiles_v // N_V) * tiles_u + iu * tiles_u // N_U
    return np.stack([cells[k == j].mean(axis=0) for j in range(tiles_u * tiles_v)])


def test_d11_ris_delay_per_tile_matches_tile_reference_points():
    traj = ConstantVelocity(np.array([0.9, -0.3, 1.5]), np.array([0.2, 0.1, 0.0]))
    m = _scene(PanelModel("ris0", _gapped_grid(), builtin("ideal_bbit(1)")), rx_traj=traj)
    m = m.with_render_tiles({"ris0": (4, 3)})
    t = Fraction(7, 2)
    p_rx = np.array([0.9, -0.3, 1.5]) + 3.5 * np.array([0.2, 0.1, 0.0])
    refs = _tile_refs(4, 3)
    snap = m.snapshot(t)
    rc = snap.ris["ris0"]
    assert rc.n_tiles == 12
    for k in range(12):
        tau = m.ris_delay_s("ris0", "mu.rx0", t, tile=k)
        exact = float(np.linalg.norm(p_rx - refs[k])) / C0
        assert tau == pytest.approx(exact, rel=1e-14)
        # the same reference point as the RIS row's tile delay (Tx -> ref -> Rx)
        tx_leg = float(np.linalg.norm(refs[k] - TX)) / C0
        assert rc.tile_delay_s[0, k] == pytest.approx(tx_leg + tau, rel=1e-14)
    centre = m.ris_delay_s("ris0", "mu.rx0", t)
    assert centre == pytest.approx(float(np.linalg.norm(p_rx - POSE.center_m)) / C0, rel=1e-15)
    for bad in (12, -1, True, 1.5):
        with pytest.raises(ConfigurationError):
            m.ris_delay_s("ris0", "mu.rx0", t, tile=bad)  # type: ignore[arg-type]
    with pytest.raises(UnknownTarget):
        m.ris_delay_s("ris0", "nobody.rx0", t, tile=0)
    with pytest.raises(UnknownTarget):
        m.ris_delay_s("nope", "mu.rx0", t)


def test_d11_timed_node_events_on_external_nodes():
    ext = trajectory_from_spec({"model": "external", "initial_m": [1.0, 0.0, 2.0]}, node="mu")
    assert isinstance(ext, ExternalTrajectory)
    m = _scene(PanelModel("ris0", _gapped_grid(), builtin("ideal_bbit(1)")), rx_traj=ext)

    def los_tau(t: Fraction) -> float:
        pb = m.snapshot(t).paths[LINK]
        return float(pb.delay_s[pb.kind == PathKind.LOS][0])

    def friis_tau(p: np.ndarray) -> float:
        return float(np.linalg.norm(p - TX)) / C0

    m.set_node_position("mu", Fraction(1), [2.0, 1.0, 1.0])
    assert los_tau(Fraction(1, 2)) == pytest.approx(friis_tau(np.array([1.0, 0.0, 2.0])), rel=1e-15)
    assert los_tau(Fraction(2)) == pytest.approx(friis_tau(np.array([2.0, 1.0, 1.0])), rel=1e-15)
    cv = ConstantVelocity(np.array([0.0, 2.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    m.set_node_trajectory("mu", Fraction(3), cv)
    assert los_tau(Fraction(5)) == pytest.approx(friis_tau(np.array([2.0, 2.0, 1.0])), rel=1e-15)
    # a section 9.2 trajectory mapping is accepted too (time measured from the event)
    spec = {"model": "constant_velocity", "start_m": [0.0, -1.0, 1.0], "velocity_mps": [0, 0.5, 0]}
    m.set_node_trajectory("mu", "6", spec)
    assert los_tau(Fraction(8)) == pytest.approx(friis_tau(np.array([0.0, 0.0, 1.0])), rel=1e-15)
    assert m.ris_delay_s("ris0", "mu.rx0", Fraction(8)) == pytest.approx(
        float(np.linalg.norm(np.array([0.0, 0.0, 1.0]) - POSE.center_m)) / C0, rel=1e-15
    )
    with pytest.raises(ConfigurationError):  # out of time order
        m.set_node_position("mu", Fraction(4), [0.0, 0.0, 1.0])
    with pytest.raises(ConfigurationError):  # not a trajectory
        m.set_node_trajectory("mu", Fraction(9), object())
    with pytest.raises(ConfigurationError):  # the Tx is not an external node
        m.set_node_position("ap", Fraction(9), [0.0, 0.0, 1.0])
    with pytest.raises(UnknownTarget):
        m.set_node_position("nobody", Fraction(9), [0.0, 0.0, 1.0])
    with pytest.raises(UnknownTarget):
        m.set_node_trajectory("nobody", Fraction(9), cv)


# --------------------------------------------------------------------------- D18


def _scenario_doc(environment: dict) -> dict:
    return {
        "seed": 3,
        "nodes": [
            {
                "id": "ap",
                "position_m": TX.tolist(),
                "streams": [{"name": "tx0", "direction": "tx", "fc_hz": FC, "rate_sps": "1e6"}],
            },
            {
                "id": "mu",
                "position_m": RX.tolist(),
                "streams": [{"name": "rx0", "direction": "rx", "fc_hz": FC, "rate_sps": "1e6"}],
            },
        ],
        "environment": environment,
    }


class _ImagePath:
    """Ground-like test mechanism: a flat-earth image path of reflection coefficient g."""

    def __init__(self, height_m: float, gamma: complex, kind: PathKind) -> None:
        self.h, self.g, self.kind = height_m, gamma, kind

    def link_paths(self, ctx):  # type: ignore[no-untyped-def]
        img = ctx.p_tx * np.array([1.0, 1.0, -1.0]) + np.array([0.0, 0.0, 2 * self.h])
        d = float(np.linalg.norm(ctx.p_rx - img))
        return [(self.g * (C0 / ctx.fc_hz) / (4 * math.pi * d), d / C0, self.kind)]


class _Factory:
    def __init__(self, make):  # type: ignore[no-untyped-def]
        self.calls: list[tuple] = []
        self.make = make

    def __call__(self, spec, scenario, *, base_dir):  # type: ignore[no-untyped-def]
        self.calls.append((spec, scenario, base_dir))
        return self.make(spec)


def _install(monkeypatch: pytest.MonkeyPatch, name: str, factory: object | None) -> None:
    mod = types.ModuleType(name)
    if factory is not None:
        mod.mechanism_from_scenario = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, mod)


class _Edge:
    def los_factor(self, ctx):  # type: ignore[no-untyped-def]
        return 0.5


@pytest.fixture
def fake_wp3(monkeypatch: pytest.MonkeyPatch) -> dict[str, _Factory]:
    facs = {
        "ground": _Factory(lambda s: _ImagePath(s["height_m"], -1.0, PathKind.GROUND)),
        "walls": _Factory(lambda s: [_ImagePath(w["h"], -0.5, PathKind.WALL) for w in s]),
        "edges": _Factory(lambda s: _Edge()),
        "diffuse": _Factory(lambda s: None),
    }
    for key, mod in prop.WP3_MECHANISM_MODULES:
        _install(monkeypatch, mod, facs[key])
    return facs


def test_d18_from_scenario_attaches_enabled_wp3_mechanisms(fake_wp3, tmp_path: Path):  # type: ignore[no-untyped-def]
    env = {
        "ground": {"eps_r": 15, "sigma_s_per_m": 1e-3, "polarization": "H", "height_m": -1.0},
        "walls": [{"id": "w1", "h": 4.0}, {"id": "w2", "h": 5.0}],
        "occluders": [],
        "edges": [{"id": "e1", "p0_m": [0, 0, 0], "p1_m": [0, 1, 0]}],
        "diffuse": {"k_factor_db": 10.0, "max_doppler_hz": 5.0},
    }
    doc = _scenario_doc(env)
    m = from_scenario(doc, base_dir=tmp_path)
    pb = m.snapshot(Fraction(0)).paths[LINK]
    # order: LOS, ground, walls (environment order of section 9.2)
    assert pb.kind.tolist() == [PathKind.LOS, PathKind.GROUND, PathKind.WALL, PathKind.WALL]
    d = float(np.linalg.norm(RX - TX))
    assert abs(pb.amp[0] - 0.5 * LAM / (4 * math.pi * d)) <= 1e-15 * LAM / (4 * math.pi * d)
    for idx, (h, g) in enumerate(((-1.0, -1.0), (4.0, -0.5), (5.0, -0.5)), start=1):
        img = TX * np.array([1.0, 1.0, -1.0]) + np.array([0.0, 0.0, 2 * h])
        di = float(np.linalg.norm(RX - img))
        assert pb.amp[idx] == pytest.approx(g * LAM / (4 * math.pi * di), rel=1e-15)
        assert pb.delay_s[idx] == pytest.approx(di / C0, rel=1e-15)
    for key, fac in fake_wp3.items():
        assert len(fac.calls) == 1
        spec, scenario, base_dir = fac.calls[0]
        assert spec == env[key] and scenario["environment"] is not None and base_dir == tmp_path
    # the classmethod builds the same model
    m2 = ScenePropagation.from_scenario(doc, base_dir=tmp_path)
    assert np.array_equal(m2.snapshot(Fraction(0)).paths[LINK].amp, pb.amp)


def test_d18_disabled_entries_are_not_imported(fake_wp3):  # type: ignore[no-untyped-def]
    env = {"ground": None, "walls": [], "occluders": [], "edges": [], "diffuse": None}
    m = from_scenario(_scenario_doc(env))
    assert all(not f.calls for f in fake_wp3.values())
    assert m.snapshot(Fraction(0)).paths[LINK].kind.tolist() == [PathKind.LOS]


def test_d18_missing_module_or_factory_raises(monkeypatch: pytest.MonkeyPatch):
    env = {"walls": [{"id": "w1", "h": 4.0}]}
    monkeypatch.setitem(sys.modules, "sremu.physics.walls", None)  # import fails: missing
    with pytest.raises(ConfigurationError, match=r"sremu\.physics\.walls"):
        from_scenario(_scenario_doc(env))
    _install(monkeypatch, "sremu.physics.walls", None)  # module without the factory
    with pytest.raises(ConfigurationError, match="mechanism_from_scenario"):
        from_scenario(_scenario_doc(env))


def test_d18_explicit_mechanisms_replace_the_automatic_ones(fake_wp3):  # type: ignore[no-untyped-def]
    env = {"ground": {"height_m": 0.0}}
    with pytest.raises(ConfigurationError):  # an enabled entry is never dropped silently
        from_scenario(_scenario_doc(env), mechanisms=())
    m = from_scenario(_scenario_doc(env), mechanisms=[_Edge()])
    assert not fake_wp3["ground"].calls
    pb = m.snapshot(Fraction(0)).paths[LINK]
    assert pb.kind.tolist() == [PathKind.LOS]


# --------------------------------------------------------------------------- D14


PHYSICS_SRC = Path(__file__).resolve().parents[2] / "src" / "sremu" / "physics"


def test_d14_no_private_key_derivation_in_physics():
    """Every keyed generator of the physics core comes from ``sremu.rng`` (section 6.1)."""
    banned = ("blake2b", "digest_size", "np.random.Philox", "default_rng", "SeedSequence")
    hits = [
        f"{p.relative_to(PHYSICS_SRC)}: {b}"
        for p in PHYSICS_SRC.rglob("*.py")
        for b in banned
        if b in p.read_text(encoding="utf-8")
    ]
    assert not hits, hits


def test_d14_keyed_generator_is_the_rng_derivation():
    for seed, purpose, name, ctr in (
        (7, "mobility", "mu", 0),
        (7, "mobility", "mu", 5),
        (9, "failure", "ris0", 0),
    ):
        a = keyed_generator(seed, purpose, name, ctr).random(16)
        b = sremu_rng.generator(seed, purpose, name, counter=ctr).random(16)
        assert np.array_equal(a, b)
    # random-waypoint legs: w_i = p0 + u_i (p1 - p0), u_i from mobility/<node> counter i
    rwp = trajectory_from_spec(
        {"model": "rwp_1d", "p0_m": [0, 0, 0], "p1_m": [10, 0, 0], "speed_mps": 1.0},
        node="mu",
        seed=11,
    )
    for i in range(4):
        u = sremu_rng.generator(11, "mobility", "mu", counter=i).random()
        assert rwp.waypoint(i)[0] == 10.0 * u  # type: ignore[attr-defined]


def test_d14_scenario_failure_map_uses_rng_failure_purpose():
    doc = _scenario_doc({})
    doc["ris"] = [
        {
            "id": "ris0",
            "pose": {"center_m": [0, 0, 0], "normal": [0, 0, 1], "u_axis": [1, 0, 0]},
            "grid": {"n_u": 10, "n_v": 10, "pitch_u_m": PITCH, "pitch_v_m": PITCH},
            "states": {"library": "ideal_bbit(2)"},
            "failures": {"dead_rate": 0.2, "stuck_rate": 0.1},
        }
    ]
    m = from_scenario(doc)
    want = FailureMap.from_rates(
        100, 0.2, 0.1, sremu_rng.generator(3, "failure", "ris0"), n_states=4
    )
    got = m.pipeline("ris0").failures
    assert np.array_equal(got.dead, want.dead) and np.array_equal(got.stuck_at, want.stuck_at)
    assert got.n_dead > 0 and got.n_stuck > 0


def _samples() -> list[LinkSample]:
    out = []
    for i in range(14):
        ang = 2 * math.pi * i / 14
        tx = np.array([0.3 * math.cos(ang), 0.3 * math.sin(ang), 2.0 + 0.01 * i])
        rx = np.array([-0.4 * math.sin(ang), 0.2 * math.cos(ang), 1.5 + 0.02 * i])
        out.append(LinkSample(f"l{i:02d}", tx, rx, FC))
    return out


def test_d14_richardson_link_choice_is_keyed_by_rng():
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 4, 4, LAM, LAM)
    samples = _samples()
    # the deterministic head (section 5.4): nearest, farthest, most oblique incidence and
    # most oblique reflection (panel at the origin, normal +z), duplicates removed
    lens = [np.linalg.norm(s.p_tx) + np.linalg.norm(s.p_rx) for s in samples]
    ci = [s.p_tx[2] / np.linalg.norm(s.p_tx) for s in samples]
    cs = [s.p_rx[2] / np.linalg.norm(s.p_rx) for s in samples]
    head = list(
        dict.fromkeys(
            [int(np.argmin(lens)), int(np.argmax(lens)), int(np.argmin(ci)), int(np.argmin(cs))]
        )
    )
    rest = [i for i in range(14) if i not in head]
    for seed in (0, 1, 2):
        idx = representative_links(samples, grid, seed=seed, panel="risA")
        assert len(idx) == 8 and len(set(idx)) == 8
        assert idx[: len(head)] == head
        # the tail: remaining links in the order of derive_key(seed, "trial", ...).u64
        keys = {
            i: sremu_rng.derive_key(seed, "trial", f"richardson/risA/{samples[i].name}").u64
            for i in rest
        }
        assert idx[len(head) :] == sorted(rest, key=lambda i: keys[i])[: 8 - len(head)]
        assert all(keys[i] == richardson_link_key(seed, "risA", samples[i].name) for i in rest)
        # a pure function of the link names: independent of the order of the samples
        perm = samples[::-1]
        names = [perm[i].name for i in representative_links(perm, grid, seed=seed, panel="risA")]
        assert sorted(names) == sorted(samples[i].name for i in idx)
    tails = {
        tuple(representative_links(samples, grid, seed=s, panel="risA")[len(head) :])
        for s in range(6)
    }
    assert len(tails) > 1  # the key depends on the master seed
