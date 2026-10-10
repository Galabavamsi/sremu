"""Gym-style quickstart (DESIGN.md section 8.8; WP9).

Runs ``RisEnv`` on ``scenarios/quickstart.yaml`` (SIM geometry) with the coefficient
(batch) backend:

1. a uniformly random policy (the action space's keyed sampler), and
2. the built-in ``ExactOptimum`` controller driven through :func:`sremu.ml.env.rollout`,

and prints the mean received power and the mean regret to the exact finite-alphabet
optimum (the regret reference of section 8.8). Usage::

    python examples/gym_quickstart/gym_quickstart.py [--steps 20] [--seed 0]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from sremu.control.builtin import ExactOptimum
from sremu.ml.env import RisEnv, rollout

REPO = Path(__file__).resolve().parents[2]
SCENARIO = REPO / "scenarios" / "quickstart.yaml"


def mean_dbm(values_dbm: np.ndarray) -> float:
    """10 log10 of the mean linear power of a series of dBm values."""
    return 10.0 * math.log10(float(np.mean(10.0 ** (np.asarray(values_dbm) / 10.0))))


def main(steps: int = 20, seed: int = 0, scenario: Path = SCENARIO) -> dict[str, float]:
    """Run both policies and return their mean power (dBm) and mean regret (dB)."""
    env = RisEnv(scenario, panel="ris0", decision_period_s="1e-3", include_noise=False)
    _obs, _info = env.reset(seed=seed)
    rewards, regrets = [], []
    for _ in range(steps):
        _obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        rewards.append(r)
        regrets.append(info["regret_db"])
        if terminated or truncated:
            break
    res = rollout(env, ExactOptimum(period_s="1e-3"), steps, seed=seed)
    out = {
        "random_mean_dbm": mean_dbm(np.asarray(rewards)),
        "random_mean_regret_db": float(np.mean(regrets)),
        "optimum_mean_dbm": mean_dbm(res["rewards_dbm"]),
        "optimum_mean_regret_db": float(np.mean(res["regret_db"])),
    }
    for k, v in out.items():
        print(f"{k:24s} {v:10.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.steps, a.seed)
