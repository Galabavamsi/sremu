"""D14 / section 6.1: controller decisions are reproducible given the seed (keyed RNG only).

Every random controller draws only from the generator handed to ``setup`` (purpose
``ctrl/<name>``, ``sremu.rng``; D14). So its decision sequence is a pure function of
(seed, name, observations): equal for equal seeds, different for different seeds,
independent of other controllers and of creation order, and identical through the
gym-style environment and the FakeEngine barrier path.

(VA-3 itself, "the exhaustive optimum is >= every controller at N = 20", is in
``test_wp9_va3_optimum_bound.py``.)
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu import rng as sremu_rng
from sremu.contracts import RisCapabilities, StreamCmd, StreamSpec, TxMetadata
from sremu.control.builtin import GeneticAlgorithm, ProbeBandit, Urap
from sremu.control.controller import (
    SnapshotChannel,
    controller_rng,
    make_observation,
    setup_controller,
)
from sremu.errors import ErrorCode
from sremu.ml.env import RisEnv, rollout
from sremu.testing import FakeEngine
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation

pytestmark = pytest.mark.unit

LINK = "ap.tx0->mu.rx0"


def _scene() -> FakePropagation:
    s = lambda sid, d: StreamSpec(sid, d, 3.5e9, Fraction(10**6))  # noqa: E731
    return FakePropagation(
        [
            FakeNode("ap", (0.0, 0.0, 0.0), streams=(s("ap.tx0", "tx"),)),
            FakeNode("mu", (4.0, 0.5, 0.0), (0.0, 1.0, 0.0), streams=(s("mu.rx0", "rx"),)),
        ],
        [FakePanel("ris0", (2.0, 2.0, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), 8, 4, 0.04, 0.04)],
        horizon_s=10,
    )


def _caps(n: int = 32, s: int = 2) -> dict[str, RisCapabilities]:
    return {
        "ris0": RisCapabilities(
            n, (n, 1), "element", tuple(f"s{i}" for i in range(s)), "analytic:t", (3e9, 4e9), 1e7
        )
    }


def _obs(m: int, power: float | None = None):
    rx = {} if power is None else {"mu.rx0": power}
    return make_observation(Fraction(m, 1000), ris_state={"ris0": np.zeros(32)}, rx_power_dbm=rx)


def _urap_sequence(seed: int, name: str = "urap", n: int = 50) -> np.ndarray:
    u = Urap(0.3, name=name)
    setup_controller(u, _caps(), seed)
    return np.stack([u.decide(_obs(m))[0].states for m in range(n)])


def test_d14_controller_rng_is_the_keyed_ctrl_stream():
    a = controller_rng(11, "urap").integers(0, 2**63, 16)
    b = sremu_rng.generator(11, "ctrl", "urap").integers(0, 2**63, 16)
    np.testing.assert_array_equal(a, b)
    c = controller_rng(11, "other").integers(0, 2**63, 16)
    assert not np.array_equal(a, c)


def test_d14_urap_same_seed_same_decisions_other_seed_differs():
    a, b = _urap_sequence(4), _urap_sequence(4)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, _urap_sequence(5))
    assert not np.array_equal(a, _urap_sequence(4, name="urap2"))


def test_d14_decisions_do_not_depend_on_other_controllers_or_order():
    alone = _urap_sequence(8, "a")
    u_a, u_b = Urap(0.3, name="a"), Urap(0.3, name="b")
    setup_controller(u_b, _caps(), 8)  # created and set up first
    setup_controller(u_a, _caps(), 8)
    inter = []
    for m in range(50):
        u_b.decide(_obs(m))  # interleaved draws of another controller
        inter.append(u_a.decide(_obs(m))[0].states)
    np.testing.assert_array_equal(alone, np.stack(inter))


def test_d14_setup_resets_the_decision_sequence():
    u = Urap(0.3)
    setup_controller(u, _caps(), 4)
    first = np.stack([u.decide(_obs(m))[0].states for m in range(20)])
    setup_controller(u, _caps(), 4)
    again = np.stack([u.decide(_obs(m))[0].states for m in range(20)])
    np.testing.assert_array_equal(first, again)


def _bandit_plays(seed: int) -> tuple[np.ndarray, list[int]]:
    b = ProbeBandit(6, "2e-2")
    setup_controller(b, _caps(), seed)
    plays = []
    for m in range(30):
        power = None if m == 0 else -40.0 - float(np.sum(plays[-1]) % 7)
        (a,) = b.decide(_obs(m, power))
        plays.append(a.states)
    return b.codebook.copy(), [int(np.sum(p)) for p in plays]


def test_d14_probe_bandit_random_codebook_and_plays_reproducible():
    cb1, p1 = _bandit_plays(21)
    cb2, p2 = _bandit_plays(21)
    np.testing.assert_array_equal(cb1, cb2)
    assert p1 == p2
    cb3, _ = _bandit_plays(22)
    assert not np.array_equal(cb1, cb3)


def test_d14_genetic_algorithm_reproducible():
    model = _scene()

    def run(seed: int) -> np.ndarray:
        g = GeneticAlgorithm(
            40, 0.02, 20, link=LINK, channel=SnapshotChannel(model), skip_unchanged=False
        )
        setup_controller(g, _caps(), seed)
        return np.stack([g.decide(_obs(m))[0].states for m in range(3)])

    np.testing.assert_array_equal(run(3), run(3))
    assert not np.array_equal(run(3), run(4))


def test_d14_environment_rollout_reproducible_end_to_end():
    def run(seed: int) -> np.ndarray:
        env = RisEnv(_scene(), decision_period_s="1e-3", tx_power_dbm=10.0)
        return rollout(env, Urap(0.5, period_s="2e-3"), 40, seed=seed)["rewards_dbm"]

    r1, r2 = run(7), run(7)
    np.testing.assert_array_equal(r1, r2)
    assert not np.array_equal(r1, run(8))


def test_d14_fake_engine_barrier_run_reproducible():
    """Two FakeEngine runs with a barrier-registered Urap give identical command records
    and RX samples (lockstep, deterministic)."""

    def run() -> tuple[list[str], np.ndarray]:
        cap = _caps(n=8, s=3)

        def gain(cfgs) -> complex:
            st = cfgs["ris0"].state
            return complex(1 + int(np.sum(st)))

        eng = FakeEngine(cap, gain_fn=gain)
        tx = eng.open_stream(StreamSpec("a.tx0", "tx", 3.5e9, Fraction(10**6)))  # type: ignore[arg-type]
        rx = eng.open_stream(StreamSpec("b.rx0", "rx", 3.5e9, Fraction(10**6)))  # type: ignore[arg-type]
        eng.register_controller(Urap(0.5, name="u", period_s="1e-3", min_latency_s="1e-4"))
        eng.submit_tx(tx, np.full(6000, 0.1, np.complex64), TxMetadata(end_of_burst=True))
        eng.stream_cmd(rx, StreamCmd("start_cont"))
        out = []
        got = 0
        while got < 5000:
            y, md = eng.read_rx(rx, 700, 0.0)
            assert md.error_code == ErrorCode.NONE
            out.append(y)
            got += y.size
        return [r.config_hash for r in eng.records], np.concatenate(out)[:5000]

    h1, y1 = run()
    h2, y2 = run()
    assert len(h1) >= 5 and h1 == h2
    np.testing.assert_array_equal(y1, y2)
    assert len(set(h1)) > 1  # the random policy did change the configuration
