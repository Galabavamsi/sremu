"""``RisEnv`` contract (DESIGN.md section 8.8; WP9 env contract test).

The reward is checked against an independent closed form of the FakePropagation scene
(Friis LOS plus the simplified point-cell RIS term evaluated directly from geometry, not
through ``combine``), the noise floor against ``k_B T_0 F f_s`` (-173.975 dBm/Hz + NF),
and the regret reference against the exhaustive optimum.
"""

from __future__ import annotations

import copy
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import StreamSpec
from sremu.control.builtin import ExactOptimum, RisAbsent, Urap
from sremu.control.controller import csi_from_snapshot
from sremu.errors import ConfigurationError
from sremu.ml.env import HAVE_GYMNASIUM, RisEnv, rollout
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
from sremu.units import C0

pytestmark = pytest.mark.unit

FC = 3.5e9
GAMMA = (1 + 0j, -1 + 0j, 0.6j)


def scene(moving: bool = True) -> FakePropagation:
    s = lambda sid, d: StreamSpec(sid, d, FC, Fraction(10**6))  # noqa: E731
    return FakePropagation(
        [
            FakeNode("ap", (0.0, 0.0, 0.0), streams=(s("ap.tx0", "tx"),)),
            FakeNode(
                "mu",
                (4.0, 0.5, 0.0),
                (0.0, 2.0, 0.0) if moving else (0.0, 0.0, 0.0),
                streams=(s("mu.rx0", "rx"),),
            ),
        ],
        [
            FakePanel(
                "ris0", (2.0, 2.0, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), 3, 2, 0.04, 0.04, GAMMA
            )
        ],
        horizon_s=10,
    )


def closed_form_h(t: float, states: np.ndarray) -> complex:
    """LOS + sum_n Gamma_{s_n} A j/(4 pi r_i r_s) exp(-jk(r_i + r_s)) from geometry."""
    lam = C0 / FC
    k = 2 * math.pi / lam
    ap = np.zeros(3)
    mu = np.array([4.0, 0.5 + 2.0 * t, 0.0])
    d = float(np.linalg.norm(mu - ap))
    h = lam / (4 * math.pi * d) * np.exp(-1j * k * d)
    c = np.array([2.0, 2.0, 0.0])
    u, v = np.array([1.0, 0, 0]), np.cross([0.0, -1.0, 0.0], [1.0, 0, 0])
    for n in range(6):
        iu, iv = n % 3, n // 3
        p = c + (iu - 1) * 0.04 * u + (iv - 0.5) * 0.04 * v
        ri, rs = np.linalg.norm(p - ap), np.linalg.norm(p - mu)
        h += (
            GAMMA[states[n]]
            * 0.04
            * 0.04
            * 1j
            / (4 * math.pi * ri * rs)
            * np.exp(-1j * k * (ri + rs))
        )
    return complex(h)


def test_env_api_spaces_and_types():
    env = RisEnv(scene(), decision_period_s="1e-3", tx_power_dbm=10.0)
    assert env.n_controls == 6 and env.n_states == 3 and env.link == "ap.tx0->mu.rx0"
    obs, info = env.reset(seed=1)
    assert set(obs) == {"rx_power_dbm", "ris_state"}
    assert obs in env.observation_space
    assert obs["rx_power_dbm"].shape == (1,) and obs["ris_state"].shape == (6,)
    assert info["oracle_reference"] is True and info["t_s"] == "0"
    a = env.action_space.sample()
    assert a in env.action_space and a.shape == (6,)
    out = env.step(a)
    assert len(out) == 5
    o2, r, term, trunc, info2 = out
    assert isinstance(r, float) and isinstance(term, bool) and isinstance(trunc, bool)
    assert o2 in env.observation_space and not term and not trunc
    np.testing.assert_array_equal(o2["ris_state"], a)
    assert info2["t_s"] == "1/1000"
    assert env.render() is None and env.close() is None
    assert HAVE_GYMNASIUM or type(env).__mro__[1] is object  # duck-typed without gymnasium


def test_env_reward_equals_closed_form_power_with_latency_and_eval_points():
    env = RisEnv(
        scene(), decision_period_s="1e-3", tx_power_dbm=10.0, eval_points=4, latency_s="3e-4"
    )
    env.reset()
    s_old = np.zeros(6, np.int64)
    rng_states = [np.array([0, 1, 2, 1, 0, 2]), np.array([2, 2, 1, 0, 0, 1])]
    for step, s_new in enumerate(rng_states):
        t0 = step * 1e-3
        _, r, _, _, info = env.step(s_new)
        ts = [t0 + (2 * i + 1) * 1e-3 / 8 for i in range(4)]
        hs = [closed_form_h(t, s_new if t >= t0 + 3e-4 else s_old) for t in ts]
        p_mw = 10.0 * np.mean([abs(h) ** 2 for h in hs])
        assert r == pytest.approx(10 * math.log10(p_mw), abs=1e-9)
        for h_env, h_ref in zip(info["h"], hs, strict=True):
            assert abs(h_env - h_ref) <= 1e-10 * abs(h_ref)
        s_old = s_new


def test_env_noise_floor_from_the_scenario(repo_root):
    path = repo_root / "scenarios" / "quickstart.yaml"
    env = RisEnv(path, include_noise=True)
    assert 10 * math.log10(env.noise_mw) == pytest.approx(-173.975 + 60.0 + 7.0, abs=1e-3)
    assert env.tx_power_dbm == 0.0 and env.horizon == Fraction(1, 10)
    obs, info = env.reset()
    h = info["h"][0]
    assert obs["rx_power_dbm"][0] == pytest.approx(
        10 * math.log10(abs(h) ** 2 + env.noise_mw), abs=1e-12
    )
    quiet = RisEnv(path)
    assert quiet.noise_mw == 0.0
    assert quiet.reset()[0]["rx_power_dbm"][0] == pytest.approx(20 * math.log10(abs(h)), abs=1e-9)


def test_env_regret_is_zero_at_the_exhaustive_optimum_and_nonnegative():
    env = RisEnv(scene(moving=False), decision_period_s="1e-3", tx_power_dbm=0.0)
    env.reset()
    model = env.model
    view = csi_from_snapshot(model, model.snapshot(Fraction(1, 2000)), env.link)
    best, arg = -1.0, None
    for cfg in itertools.product(range(3), repeat=6):
        v = abs(view.evaluate({"ris0": np.asarray(cfg, np.int16)}))
        if v > best:
            best, arg = v, cfg
    _, r, _, _, info = env.step(np.asarray(arg))
    assert abs(info["regret_db"]) <= 1e-9
    assert r == pytest.approx(20 * math.log10(best), abs=1e-9)
    for _ in range(10):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert info["regret_db"] >= -1e-9


def test_env_truncates_at_the_horizon_and_reset_restarts():
    env = RisEnv(scene(), decision_period_s="1e-3", horizon_s="3e-3", tx_power_dbm=0.0)
    env.reset()
    flags = [env.step(np.zeros(6, int))[3] for _ in range(3)]
    assert flags == [False, False, True]
    env.reset()
    assert env.t == 0
    with pytest.raises(ConfigurationError):
        RisEnv(scene(), decision_period_s="1e-3", tx_power_dbm=0.0).step(np.zeros(6, int))


def _rwp_scenario(repo_root):
    from sremu.scenario.load import read_document

    doc = read_document(repo_root / "scenarios" / "quickstart.yaml")
    doc = copy.deepcopy(doc)
    mu = doc["nodes"][1]
    del mu["position_m"]
    mu["trajectory"] = {
        "model": "rwp_1d",
        "p0_m": [4.7, 0.5, 0.0],
        "p1_m": [4.7, 2.5, 0.0],
        "speed_mps": 1.0,
    }
    return doc


def test_env_reset_seed_reproducible_and_keys_the_scene(repo_root):
    doc = _rwp_scenario(repo_root)
    env = RisEnv(doc, decision_period_s="1e-2", observe=("rx_power_dbm", "ris_state", "positions"))

    def episode(seed: int):
        obs, _ = env.reset(seed=seed)
        acts = [env.action_space.sample() for _ in range(5)]
        rows = [obs["positions"].copy()]
        rewards = []
        for a in acts:
            o, r, *_ = env.step(a)
            rows.append(o["positions"].copy())
            rewards.append(r)
        return np.stack(rows), np.asarray(rewards), np.stack(acts)

    p1, r1, a1 = episode(3)
    p2, r2, a2 = episode(3)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(a1, a2)
    p3, _, a3 = episode(4)
    assert not np.array_equal(p1, p3)  # the random-waypoint legs are keyed by the seed
    assert not np.array_equal(a1, a3)
    # positions are the model's positions at the end of each period
    np.testing.assert_array_equal(p3[2], env._positions(Fraction(2, 100)))


def test_env_space_samplers_are_distinct_keyed_streams():
    """Section 6.1: every sampler is a keyed sremu.rng stream (purpose ctrl), the action and
    observation samplers do not share a stream, and key_space works on any object with the
    gymnasium attributes (``_np_random``, ``spaces``), so the gymnasium path is keyed too."""
    from sremu import rng as sremu_rng
    from sremu.ml.env import key_space

    env = RisEnv(scene(), tx_power_dbm=0.0)
    env.reset(seed=12)
    nvec = np.full(6, 3)
    a = env.action_space.sample()
    ref = sremu_rng.generator(12, "ctrl", "env/ris0/action").integers(0, nvec)
    np.testing.assert_array_equal(a, ref)
    o = env.observation_space.spaces["ris_state"].sample()
    ref_o = sremu_rng.generator(12, "ctrl", "env/ris0/observation/ris_state").integers(0, nvec)
    np.testing.assert_array_equal(o, ref_o)
    env.reset(seed=12)
    seq_a = np.stack([env.action_space.sample() for _ in range(8)])
    seq_o = np.stack([env.observation_space.spaces["ris_state"].sample() for _ in range(8)])
    assert not np.array_equal(seq_a, seq_o)  # independent streams, not one shared key

    class _Leaf:
        _np_random = None

    class _Tree:
        _np_random = None

        def __init__(self) -> None:
            self.spaces = {"x": _Leaf(), "y": _Leaf()}

    t = _Tree()
    key_space(t, 5, "env/p/observation")
    for name, g in (
        ("env/p/observation", t._np_random),
        ("env/p/observation/x", t.spaces["x"]._np_random),
    ):
        assert g.integers(0, 2**62) == sremu_rng.generator(5, "ctrl", name).integers(0, 2**62)
    assert t.spaces["y"]._np_random is not t.spaces["x"]._np_random


def test_rollout_rejects_a_controller_bound_to_the_scene_reset_replaced(repo_root):
    """reset(seed) with a new seed rebuilds a scenario's scene; a controller whose
    SnapshotChannel holds the replaced scene would optimize stale random waypoints."""
    from sremu.control.builtin import DelayedController
    from sremu.control.controller import SnapshotChannel

    doc = _rwp_scenario(repo_root)
    for wrap in (False, True):
        env = RisEnv(doc, decision_period_s="1e-3")
        inner = ExactOptimum(period_s="1e-3", channel=SnapshotChannel(env.model))
        ctrl = DelayedController(inner, "1e-3") if wrap else inner
        with pytest.raises(ConfigurationError, match="rebuilt the scene"):
            rollout(env, ctrl, 3, seed=int(doc.get("seed", 0)) + 5)
    env = RisEnv(doc, decision_period_s="1e-3")
    env.reset(seed=int(doc.get("seed", 0)) + 5)  # the documented order works
    fresh = ExactOptimum(period_s="1e-3", channel=SnapshotChannel(env.model))
    res = rollout(env, fresh, 3, seed=int(doc.get("seed", 0)) + 5)
    assert np.all(res["regret_db"] <= 0.01)


def test_env_codebook_actions_match_state_actions():
    cb = np.array([[0, 1, 2, 0, 1, 2], [2, 2, 2, 1, 1, 1]])
    e1 = RisEnv(scene(), action="codebook", codebook=cb, tx_power_dbm=0.0)
    e2 = RisEnv(scene(), tx_power_dbm=0.0)
    assert e1.action_space.n == 2
    e1.reset()
    e2.reset()
    for i in (1, 0, 1):
        assert e1.step(i)[1] == e2.step(cb[i])[1]
    with pytest.raises(ConfigurationError):
        e1.step(2)
    with pytest.raises(ConfigurationError):
        RisEnv(scene(), action="codebook", tx_power_dbm=0.0)


def test_env_rejects_invalid_actions_and_settings():
    env = RisEnv(scene(), tx_power_dbm=0.0)
    env.reset()
    for bad in (np.zeros(5, int), np.full(6, 3), np.full(6, -1), np.full(6, 0.5)):
        with pytest.raises(ConfigurationError):
            env.step(bad)
    with pytest.raises(ConfigurationError):
        RisEnv(scene(), backend="numba_cpu")  # IQ backends need the WP6 engine
    with pytest.raises(ConfigurationError):
        RisEnv(scene(), reward="ber")
    with pytest.raises(ConfigurationError):
        RisEnv(scene(), latency_s="1e-3")  # latency must be < decision period
    with pytest.raises(TypeError):
        RisEnv(scene(), decision_period_s=0.001)  # exact times only (section 3.1)


def test_rollout_exact_optimum_has_zero_regret_and_ris_absent_drops_the_ris():
    env = RisEnv(scene(moving=False), decision_period_s="1e-3", tx_power_dbm=0.0)
    res = rollout(env, ExactOptimum(period_s="1e-3"), 5, seed=0)  # oracle CSI, static scene
    assert np.all(np.abs(res["regret_db"]) <= 1e-9)
    assert res["n_commands"] == 1  # unchanged decisions are not re-sent
    absent = rollout(env, RisAbsent(period_s="1e-3"), 3, seed=0)
    los = 20 * math.log10(C0 / FC / (4 * math.pi * math.hypot(4.0, 0.5)))
    np.testing.assert_allclose(absent["rewards_dbm"], los, atol=1e-9)
    with pytest.raises(ConfigurationError):
        rollout(env, Urap(period_s="15e-4"), 3)  # not a multiple of the decision period
