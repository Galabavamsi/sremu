"""WP5 renderer on the real WP2 ``ScenePropagation`` (PH-17, PH-18 "then the real model").

The scenes are built from section 9.2 scenario dicts (WP1 models, WP2 ``from_scenario``).
Oracles: the closed-form LOS path length of a constant-velocity node (PH-17) and the model's
own exact per-element evaluation at each sample time, ``evaluate(snapshot(t_n), cfg)``, as
the brute force (PH-18). The tests skip when WP1/WP2 cannot build the scene yet.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from sremu.channel.interp import EPS_REL, PanelGeometry, choose_policy, verify_interp
from sremu.contracts import PanelConfig
from sremu.units import C0

from ._render_support import SpanSpec, const_ring, render_span

pytestmark = pytest.mark.runtime

FC = 3.5e9


def _node(
    nid: str, rate: str, direction: str, pos: list[float], vel: list[float] | None = None
) -> dict:
    stream: dict[str, Any] = {
        "name": f"{direction}0",
        "direction": direction,
        "fc_hz": FC,
        "rate_sps": rate,
    }
    if direction == "tx":
        stream["occupied_bw_hz"] = 1.0  # a DC tone: integer delay
    node: dict[str, Any] = {"id": nid, "antenna": {"pattern": "isotropic"}, "streams": [stream]}
    if vel is None:
        node["position_m"] = pos
    else:
        node["trajectory"] = {"model": "constant_velocity", "start_m": pos, "velocity_mps": vel}
    return node


def _scenario(name: str, nodes: list[dict], ris: list[dict], duration: str) -> dict:
    return {
        "schema_version": "1.0",
        "name": name,
        "seed": 20261010,
        "numerics": {"backend": "numpy_ref"},
        "environment": {"walls": [], "occluders": [], "edges": []},
        "nodes": nodes,
        "ris": ris,
        "controllers": [],
        "run": {
            "mode": "lockstep",
            "coupling": "barrier",
            "duration_s": duration,
            "late_policy": "apply_on_arrival",
        },
        "record": {"iq": False, "events": False, "metrics": False, "snapshots": False},
    }


def _build(d: dict, horizon: str):
    models = pytest.importorskip("sremu.scenario.models")
    prop = pytest.importorskip("sremu.physics.propagation")
    last: Exception | None = None
    libs = ("ideal_bbit(b=4)", "ideal_bbit(4)")  # section 9.4 spelling first
    for lib in libs if d["ris"] else libs[:1]:
        for p in d["ris"]:
            p["states"] = {"library": lib}
        try:
            return prop.ScenePropagation.from_scenario(
                models.Scenario.model_validate(d), horizon_s=horizon
            )
        except (ValueError, TypeError, KeyError) as exc:  # pragma: no cover - integration guard
            last = exc
    pytest.skip(f"WP1/WP2 cannot build the scene yet: {last!r}")


def test_ph17_real_model_los_doppler(backend_name: str) -> None:
    d = _scenario(
        "wp5_t9",
        [
            _node("tx", "1e4", "tx", [0.0, 0.0, 0.0]),
            _node("rx", "1e4", "rx", [50.0, 0.0, 0.0], [-30.0, 0.0, 0.0]),
        ],
        [],
        "1",
    )
    model = _build(d, "1")
    link = model.links()[0]
    fs = Fraction(10_000)
    pol = choose_policy(model.motion_bounds(), [], {link: FC}, fs)
    n = 2000
    ring = const_ring("tx.tx0", fs, n + 64)
    y = render_span(SpanSpec(model, {}, link, FC, ring, fs, pol.t_u_s), 0, n, backend_name)
    cyc = FC * (50.0 - 30.0 * np.arange(n) / 1e4) / C0
    assert np.max(np.abs(np.angle(y * np.exp(2j * np.pi * (cyc - np.rint(cyc)))))) <= 1e-6
    fi = np.angle(y[1:] * np.conj(y[:-1])) * 1e4 / (2 * np.pi)
    assert np.max(np.abs(fi - 30.0 * FC / C0)) <= 0.001


def _lab_walk_real():
    """Lab-walk scene on the real model with auto render tiles, focused at t = 0.2 s.

    Returns (model, link, f_s, policy, cfg).
    """
    lam = C0 / FC
    pitch = 0.0428  # <= lambda/2: point cells
    assert pitch <= lam / 2
    panel = {
        "id": "ris0",
        "enabled": True,
        "pose": {"center_m": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0], "u_axis": [0.0, 1.0, 0.0]},
        "grid": {
            "n_u": 25,
            "n_v": 25,
            "pitch_u_m": pitch,
            "pitch_v_m": pitch,
            "cell_model": "point",
        },
        "states": {"library": "ideal_bbit(b=4)"},
        "control": {"granularity": "element"},
        "initial": {"fill": 0},
        "transition": {"model": "step"},
        "capabilities": {},
    }
    d = _scenario(
        "wp5_lab_walk",
        [
            _node("tx", "2e4", "tx", [1.5, -1.0, 0.3]),
            _node("rx", "2e4", "rx", [0.45, -0.3, 0.0], [0.0, 1.5, 0.0]),
        ],
        [panel],
        "4/10",
    )
    m1 = _build(d, "4/10")
    link = m1.links()[0]
    fs = Fraction(20_000)
    geo = PanelGeometry("ris0", 25 * pitch, 25 * pitch, 25, 25, FC, 1)
    pol = choose_policy(m1.motion_bounds(), [geo], {link: FC}, fs)
    tp = pol.tiles["ris0"]
    model = m1.with_render_tiles({"ris0": (tp.tiles_u, tp.tiles_v)})
    # focus the panel on the RX at t = 0.2 s with the oracle CSI (per-cell best state)
    cas = model.csi_view(model.snapshot(Fraction(1, 5)), link).cascade["ris0"]
    cfg = {"ris0": PanelConfig("ris0", np.argmax(cas.real, axis=0).astype(np.int16))}
    return model, link, fs, pol, cfg


def test_ph18_real_model_lab_walk_vs_per_element_evaluation() -> None:
    model, link, fs, pol, cfg = _lab_walk_real()
    m_a = math.floor(Fraction(19, 100) / pol.t_u_s)
    m_b = math.ceil(Fraction(21, 100) / pol.t_u_s)
    k_lo, k_hi = math.ceil(m_a * pol.t_u_s * fs), math.ceil(m_b * pol.t_u_s * fs)
    ring = const_ring("tx.tx0", fs, k_hi + 64)
    y = render_span(SpanSpec(model, cfg, link, FC, ring, fs, pol.t_u_s), k_lo, k_hi)
    h = np.array(
        [model.evaluate(model.snapshot(Fraction(k) / fs), cfg)[link] for k in range(k_lo, k_hi)]
    )
    rel = np.abs(y - h) / np.abs(h)
    assert rel.max() <= EPS_REL
    vi = [
        verify_interp(model, cfg, m * pol.t_u_s, (m + 1) * pol.t_u_s, {link: FC})[link]
        for m in range(m_a, m_b)
    ]
    assert max(vi) == pytest.approx(rel.max(), rel=0.10)


def test_ph19_real_model_t_u_and_chunk_invariance(backend_name: str) -> None:
    """(review) PH-19 on the real model: T_u in {auto, auto/2, auto/4} agree within 2e-3
    relative; client chunk edges at 0.25 / 1 / 3.7 ms leave the output bitwise equal;
    snapshots at a common time are bitwise identical whatever was evaluated before."""
    model, link, fs, pol, cfg = _lab_walk_real()
    k_lo, k_hi = 3700, 4300
    ring = const_ring("tx.tx0", fs, k_hi + 64)
    outs = [
        render_span(
            SpanSpec(model, cfg, link, FC, ring, fs, pol.t_u_s / div), k_lo, k_hi, backend_name
        )
        for div in (1, 2, 4)
    ]
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.max(np.abs(outs[i] - outs[j]) / np.abs(outs[0])) <= 2 * EPS_REL
    spec = SpanSpec(model, cfg, link, FC, ring, fs, pol.t_u_s)
    for chunk_ms in ("1/4", "1", "37/10"):
        chunk = Fraction(chunk_ms) / 1000 * fs
        edges = [math.ceil(i * chunk) for i in range(1, math.ceil(k_hi / chunk) + 1)]
        got = render_span(spec, k_lo, k_hi, backend_name, chunk_edges=edges)
        assert np.array_equal(outs[0], got), chunk_ms
    t_common = 9 * pol.t_u_s
    p1 = model.combine(model.snapshot(t_common), cfg)[link]
    model.snapshot(Fraction(1, 7))  # evaluation order must not matter (pure in t)
    p2 = model.combine(model.snapshot(t_common), cfg)[link]
    assert np.array_equal(p1.amp, p2.amp) and np.array_equal(p1.delay_s, p2.delay_s)
