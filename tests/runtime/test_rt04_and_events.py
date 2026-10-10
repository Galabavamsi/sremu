"""RT-04 (I2b) page target invariance, plus engine events: geometry commands, counterfactual
``ris.set_enabled`` and observation fields (DESIGN.md sections 4.2, 4.3, 4.6, 4.9, 10.3)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.contracts import (
    CommandRequest,
    CommandTime,
    PanelConfig,
    PathKind,
)
from sremu.engine.streams import ToneSource
from sremu.errors import ErrorCode
from sremu.testing.fake_propagation import FakeNode, FakePropagation
from sremu.testing.scenes import stream
from sremu.timebase import to_ticks
from tests.runtime._engine_support import (
    ScriptedController,
    captured,
    config_request,
    lab_like_fake,
    make_engine,
)

pytestmark = pytest.mark.runtime

F = Fraction(10**6)


# --------------------------------------------------------------------------- RT-04 (I2b)


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_rt04_page_target_025_05_1ms_within_1e_9(backend: str) -> None:
    """page_target_s in {0.25, 0.5, 1} ms (pages of 256, 512, 1024 samples): complex128
    accumulators agree within 1e-9 relative (moving scene, commands, noise, CFO)."""
    outs = []
    for target in ("2.5e-4", "5e-4", "1e-3"):
        eng = make_engine(
            lab_like_fake(moving=True),
            backend=backend,
            noise=True,
            bw=8.0e5,
            cfo={"ap.tx0": 50.0, "rx1.rx0": -75.0},
            render_tiles="auto",
            traffic={"ap.tx0": ToneSource(F, 1.1e4, 1.0)},
            always_active=True,
            page_target_s=target,
            seed=4,
        )
        nc = eng.commands.specs["ris0"].n_controls
        eng.schedule(
            config_request(
                "c", "ris0", np.ones(nc, np.int16), CommandTime.at(Fraction(2345, 10**6))
            )
        )
        assert eng.groups[0].page == {"2.5e-4": 256, "5e-4": 512, "1e-3": 1024}[target]
        cap = eng.capture_accumulator("rx1.rx0")
        eng.run(Fraction(6, 1000))
        outs.append(captured(cap, 0, 6000))
    ref = outs[1]
    for y in (outs[0], outs[2]):
        assert np.max(np.abs(y - ref)) <= 1e-9 * np.max(np.abs(ref))


# --------------------------------------------------------------------------- geometry events


def _external_scene() -> FakePropagation:
    fc = 3.5e9
    nodes = [
        FakeNode("ap", (0.0, 0.0, 0.0), streams=(stream("ap.tx0", "tx", fc, F),)),
        FakeNode("mu", (10.0, 0.0, 0.0), streams=(stream("mu.rx0", "rx", fc, F),), external=True),
    ]
    return FakePropagation(nodes, horizon_s=1)


def test_node_set_position_event_applies_at_its_sample() -> None:
    """A timed ``node.set_position`` (static scene: event-only snapshots) changes the LOS
    from to_ticks(t) on; before it the output is unchanged; the record is applied."""
    model = _external_scene()
    eng = make_engine(model, traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)}, always_active=True)
    cap = eng.capture_accumulator("mu.rx0")
    t = Fraction(1500, 10**6)
    req = CommandRequest(
        "mv",
        "node.set_position",
        "test",
        "mu",
        CommandTime.at(t),
        position_m=np.array([20.0, 0.0, 0.0]),
    )
    assert eng.schedule(req).ok
    eng.run(Fraction(3, 1000))
    y = captured(cap, 0, 3000)
    k = to_ticks(t, F)
    assert np.all(y[:k] == y[0])
    assert np.all(y[k + 1 :] == y[k + 1])
    assert abs(abs(y[k + 1]) / abs(y[0]) - 0.5) < 1e-6  # 1/(4 pi d): 10 m -> 20 m
    rec = eng.records[0]
    assert rec.op == "node.set_position"
    assert rec.status == "applied"
    assert rec.k_applied == {"mu.rx0": k}


def test_node_set_trajectory_in_static_scene_is_not_permitted() -> None:
    eng = make_engine(_external_scene(), traffic={"ap.tx0": ToneSource(F)}, always_active=True)
    req = CommandRequest(
        "tr",
        "node.set_trajectory",
        "test",
        "mu",
        CommandTime.at(Fraction(1, 1000)),
        payload={
            "trajectory": {
                "model": "constant_velocity",
                "start_m": [10, 0, 0],
                "velocity_mps": [1, 0, 0],
            }
        },
    )
    r = eng.schedule(req)
    assert not r.ok
    assert r.error == ErrorCode.NOT_PERMITTED


# --------------------------------------------------------------------------- set_enabled


def test_set_enabled_counterfactual_removes_ris_paths_per_tile() -> None:
    """``ris.set_enabled(False)`` (counterfactual, logged as such): from the per-RX switch
    sample on, the output equals the non-RIS paths only (combine with the panel absent)."""
    model = lab_like_fake(tiles=(1, 1))
    eng = make_engine(model, traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)}, always_active=True)
    cap = eng.capture_accumulator("rx1.rx0")
    t = Fraction(1, 1000)
    req = CommandRequest("off", "ris.set_enabled", "test", "ris0", CommandTime.at(t), enabled=False)
    assert eng.schedule(req).ok
    eng.run(Fraction(2, 1000))
    y = captured(cap, 0, 2000)
    k = to_ticks(t + Fraction(model.ris_delay_s("ris0", "rx1.rx0", t)), F)
    snap = model.snapshot(Fraction(0))
    absent = model.combine(snap, {"ris0": PanelConfig("ris0", np.zeros(120, np.int16), False)})[
        "ap.tx0->rx1.rx0"
    ]
    present = model.combine(snap, {"ris0": PanelConfig("ris0", np.zeros(120, np.int16), True)})[
        "ap.tx0->rx1.rx0"
    ]
    xw = complex(np.complex64(np.sqrt(1e-3)))

    def h(lp):
        return complex(np.sum(lp.amp * np.exp(-2j * np.pi * np.mod(3.5e9 * lp.delay_s, 1.0)))) * xw

    assert abs(y[k - 1] - h(present)) <= 1e-12 * abs(h(present))
    assert abs(y[k] - h(absent)) <= 1e-12 * abs(h(absent))
    assert np.any(present.kind == PathKind.RIS_TILE)
    rec = eng.records[0]
    assert rec.counterfactual is True
    assert rec.status == "applied"
    assert rec.config_hash == PanelConfig("ris0", np.zeros(120, np.int16), False).config_hash()


# --------------------------------------------------------------------------- observations


def test_observation_positions_and_oracle_csi_flagged() -> None:
    model = lab_like_fake(moving=True)
    eng = make_engine(model, traffic={"ap.tx0": ToneSource(F, 0.0, 1.0)}, render_tiles="auto")
    ctrl = ScriptedController(
        "obs",
        "1/1000",
        "1/10000",
        lambda o, m: [],
        fields=("positions", "csi_oracle", "ris_state", "rx_power_dbm"),
    )
    eng.register_controller(ctrl)
    eng.run(Fraction(3, 1000))
    assert eng.oracle_csi_used
    assert eng.manifest_info()["oracle_csi_used"] is True
    o = ctrl.observations[-1]
    assert o.t_b == Fraction(2, 1000)
    assert o.positions is not None
    assert np.allclose(o.positions["rx1"], model.node_position("rx1", o.t_b))
    assert o.csi_oracle is not None
    csi = o.csi_oracle["ap.tx0->rx1.rx0"]
    st = o.ris_state["ris0"]
    snap = model.snapshot(o.t_b)
    lp = model.combine(snap, {"ris0": PanelConfig("ris0", st)})["ap.tx0->rx1.rx0"]
    h = complex(np.sum(lp.amp * np.exp(-2j * np.pi * np.mod(3.5e9 * lp.delay_s, 1.0))))
    assert abs(csi.evaluate({"ris0": st}) - h) <= 1e-9 * abs(h)
    assert np.isfinite(o.rx_power_dbm["rx1.rx0"])


def test_element_group_command_by_physical_tile() -> None:
    """``elements="tile1"`` addresses the controls of physical tile 1 of the WP2 lab-like
    panel (3 tiles of 8 x 5 cells); a ``states`` command on it changes only those cells."""
    from sremu.engine.commands import panel_specs_from_model
    from sremu.physics.propagation import from_scenario
    from sremu.scenario.load import load_scenario

    res = load_scenario(
        Path(__file__).resolve().parents[2] / "scenarios" / "lab_like_A.yaml", warn=False
    )
    model = from_scenario(res.scenario, base_dir=res.base_dir)
    spec = panel_specs_from_model(model)["ris0"]
    geo = model.panel_geometry("ris0")
    assert geo.n_tiles_u == 3
    idx = spec.resolve_elements("tile1")
    assert idx is not None
    tiles = geo.physical_tile_of_cell()
    assert set(idx.tolist()) == set(np.flatnonzero(tiles == 1).tolist())
    eng = make_engine(model, traffic={"ap.tx0": ToneSource(F)}, always_active=True)
    r = eng.schedule(
        config_request(
            "t1",
            "ris0",
            np.ones(idx.size, np.int16),
            CommandTime.at(Fraction(1, 1000)),
            elements="tile1",
        )
    )
    assert r.ok
    eng.run(Fraction(2, 1000))
    st = eng.panel_state("ris0", Fraction(1, 1000)).state
    assert np.all(st[tiles == 1] == 1)
    assert np.all(st[tiles != 1] == 0)
