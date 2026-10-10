"""RT-19 (I5): ``numba_cpu`` vs ``numpy_ref`` through the engine on the lab-walk, vehicle and
FM scenes: <= 1e-9 relative on the complex128 accumulators and <= 1e-6 relative RMS on the
complex64 output. DESIGN.md sections 4.2 (I5), 7.1, 10.3.

Scenes are SIM geometry on ``FakePropagation`` (simplified point-cell RIS) plus the WP2
physics model on the lab-like scenario; the engine chooses T_u and render tiles (4.4), FD
filters and narrowband/wideband grouping (4.5); noise and LO offsets are on.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.contracts import CommandTime, TransitionSpec
from sremu.engine.emulator import Emulator
from sremu.engine.streams import ToneSource
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import stream
from tests.runtime._engine_support import Collect, captured, config_request, make_engine

pytestmark = pytest.mark.runtime

F = Fraction(10**6)


def _scene(name: str) -> tuple[FakePropagation, float]:
    """(model, occupied bandwidth) of a named RT-19 scene."""
    if name == "lab_walk":  # 3.5 GHz, 1.5 m/s, R_min about 0.45 m, 1.08 m panel
        fc = 3.5e9
        nodes = [
            FakeNode("tx", (2.0, -1.2, 0.0), streams=(stream("tx.tx0", "tx", fc, F),)),
            FakeNode(
                "rx", (0.5, -0.6, 0.0), (0.0, 1.5, 0.0), streams=(stream("rx.rx0", "rx", fc, F),)
            ),
        ]
        panel = FakePanel("ris0", (0, 0, 0), (1, 0, 0), (0, 1, 0), 25, 25, 0.0432, 0.0432)
        return FakePropagation(nodes, [panel], horizon_s="1/10"), 8.0e5
    if name == "vehicle":  # 3.5 GHz, 30 m/s, R_min about 5 m, 0.4 m panel
        fc = 3.5e9
        nodes = [
            FakeNode("tx", (40.0, -30.0, 5.0), streams=(stream("tx.tx0", "tx", fc, F),)),
            FakeNode(
                "rx", (5.0, -2.0, 0.0), (0.0, 30.0, 0.0), streams=(stream("rx.rx0", "rx", fc, F),)
            ),
        ]
        panel = FakePanel("ris0", (0, 0, 0), (1, 0, 0), (0, 1, 0), 10, 10, 0.04, 0.04)
        return FakePropagation(nodes, [panel], horizon_s="1/10"), 6.0e5
    if name == "fm":  # 100 MHz, 52 km link, RIS 2 km from the RX: wideband groups
        fc = 1.0e8
        nodes = [
            FakeNode("tx", (-52_000.0, 0.0, 200.0), streams=(stream("tx.tx0", "tx", fc, F),)),
            FakeNode(
                "rx", (0.0, 0.0, 1.5), (1.0, 0.0, 0.0), streams=(stream("rx.rx0", "rx", fc, F),)
            ),
        ]
        c, sn = 0.5, float(np.sqrt(0.75))
        panel = FakePanel(
            "ris0", (-1000.0, 1732.0, 10.0), (c, -sn, 0.0), (sn, c, 0.0), 30, 30, 0.25, 0.25
        )
        return FakePropagation(nodes, [panel], horizon_s="1/10"), 2.0e5
    raise KeyError(name)


def _run(model: FakePropagation, bw: float, backend: str, n: int):
    sink = Collect()
    eng = make_engine(
        model,
        backend=backend,
        noise=True,
        bw=bw,
        cfo={"tx.tx0": 211.0, "rx.rx0": -97.0},
        render_tiles="auto",
        traffic={"tx.tx0": ToneSource(F, 3.1e4, 1.0)},
        always_active=True,
        sinks=[sink],
        seed=19,
    )
    nc = eng.commands.specs["ris0"].n_controls
    s = np.zeros(nc, np.int16)
    s[::2] = 1
    eng.schedule(
        config_request(
            "c",
            "ris0",
            s,
            CommandTime.at(Fraction(n // 2) / F),
            transition=TransitionSpec("linear", Fraction(2, 10**5)),
        )
    )
    cap = eng.capture_accumulator("rx.rx0")
    eng.run(Fraction(n) / F)
    return captured(cap, 0, n), sink.rx_array("rx.rx0", 0, n), eng


def _check(a: tuple, b: tuple) -> None:
    acc_a, out_a, _ = a
    acc_b, out_b, _ = b
    assert np.max(np.abs(acc_a - acc_b)) <= 1e-9 * np.max(np.abs(acc_a))
    oa = out_a.astype(np.complex128)
    ob = out_b.astype(np.complex128)
    assert np.sqrt(np.mean(np.abs(oa - ob) ** 2)) <= 1e-6 * np.sqrt(np.mean(np.abs(oa) ** 2))


@pytest.mark.parametrize("scene", ["lab_walk", "vehicle", "fm"])
def test_rt19_numba_vs_numpy_scenes(scene: str) -> None:
    model, bw = _scene(scene)
    n = 6000
    a = _run(model, bw, "numpy_ref", n)
    b = _run(_scene(scene)[0], bw, "numba_cpu", n)
    eng = a[2]
    assert eng.t_u is not None  # moving scene: snapshot grid
    dec = eng.manifest_info()["link_decisions"]["tx.tx0->rx.rx0"]
    if scene == "fm":
        assert dec["narrowband"] is False  # RIS path several us longer than LOS
        assert dec["n_groups"] >= 2
    _check(a, b)


def test_rt19_numba_vs_numpy_physics_model_lab_like() -> None:
    """WP2 ScenePropagation (lab_like_A.yaml, static, 32-tap FD) with a transition."""
    path = Path(__file__).resolve().parents[2] / "scenarios" / "lab_like_A.yaml"
    out = []
    for backend in ("numpy_ref", "numba_cpu"):
        sink = Collect()
        eng = Emulator.from_scenario(
            path,
            backend=backend,
            always_active=True,
            sinks=[sink],
            traffic={"ap.tx0": ToneSource(F, 4.0e4, 1.0)},
        )
        nc = eng.commands.specs["ris0"].n_controls
        eng.schedule(
            config_request(
                "c",
                "ris0",
                np.ones(nc, np.int16),
                CommandTime.at(Fraction(1, 1000)),
                transition=TransitionSpec("linear", Fraction(2, 10**5)),
            )
        )
        cap = eng.capture_accumulator("rx1.rx0")
        eng.run(Fraction(3, 1000))
        out.append((captured(cap, 0, 3000), sink.rx_array("rx1.rx0", 0, 3000), eng))
    _check(out[0], out[1])
