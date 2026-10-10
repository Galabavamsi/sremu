"""ST-01: 1-D random waypoint (Bettstetter) through the scenario trajectory factory.

DESIGN.md sections 5.10 and 10.5: l = 10 m, v = 1 m/s, no pause, 10^4 movement periods
sampled at 10 Hz. Accept: KS D of the sampled positions against the stationary CDF
``F(x) = 3x^2/l^2 - 2x^3/l^3`` <= 0.02; start points (waypoints) uniform with D <= 0.0163;
mean leg length |s - d| = l/3 +- 2.5 %; no jump larger than ``v*dt + 1e-12``.

The trajectory comes from ``sremu.physics.geometry.trajectory_from_spec`` (keyed RNG
purpose ``mobility/<node>``, D14) exactly as the engine's scenarios build it; one seed, no
seed selection. The closed forms come from ``tests/oracles/mobility.py``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.physics.geometry import trajectory_from_spec
from tests.oracles.mobility import ks_statistic, rwp1d_cdf, rwp1d_mean_leg

pytestmark = pytest.mark.statistical

L_M = 10.0
V = 1.0
PERIODS = 10_000
SEED = 20261010


@pytest.fixture(scope="module")
def walk() -> tuple[np.ndarray, np.ndarray]:
    """(waypoints of 10^4 + 1 legs, positions sampled at 10 Hz over those legs)."""
    traj = trajectory_from_spec(
        {"model": "rwp_1d", "p0_m": [0.0, 0.0, 0.0], "p1_m": [L_M, 0.0, 0.0], "speed_mps": V},
        node="mu",
        seed=SEED,
    )
    horizon = 1.0
    starts = traj.leg_starts(horizon)
    while len(starts) < PERIODS + 2:
        horizon *= 2
        starts = traj.leg_starts(horizon)
    t_end = starts[PERIODS]
    wps = np.array([traj.waypoint(i)[0] for i in range(PERIODS + 1)])
    n = int(t_end * 10)
    xs = np.array([traj.position(Fraction(i, 10))[0] for i in range(n)])
    return wps, xs


def test_st01_rwp_1d_stationary_pdf_uniform_waypoints_mean_leg(
    walk: tuple[np.ndarray, np.ndarray],
) -> None:
    wps, xs = walk
    legs = np.abs(np.diff(wps))
    assert legs.size == PERIODS
    assert legs.mean() == pytest.approx(rwp1d_mean_leg(L_M), rel=0.025)  # l/3 +- 2.5 %
    d_wp = ks_statistic(np.sort(wps[:PERIODS]), lambda x: np.clip(x / L_M, 0.0, 1.0))
    assert d_wp <= 0.0163  # start points uniform
    d_pos = ks_statistic(np.sort(xs), lambda x: rwp1d_cdf(x, L_M))
    assert d_pos <= 0.02  # stationary pdf


def test_st01_rwp_no_jump_larger_than_v_dt(walk: tuple[np.ndarray, np.ndarray]) -> None:
    """No 10 Hz step exceeds v*dt + 1e-12 m over the whole walk (t up to ~3.3e4 s): the
    trajectory evaluates positions from the exact leg-relative time (gates round 1 fix of
    the former float(t) evaluation, which overshot by up to 6.3e-12 m)."""
    _wps, xs = walk
    assert xs[-1] != xs[0]  # the walk moves
    assert np.all(np.abs(np.diff(xs)) <= V * 0.1 + 1e-12)


def test_st01_teeth_float_absolute_time_overshoots_the_jump_bound(
    walk: tuple[np.ndarray, np.ndarray],
) -> None:
    """Teeth: the same walk evaluated from float absolute time (the pre-fix rule,
    ``a + (float(t) - t0)/(t1 - t0) (b - a)`` with float leg boundaries) breaks the bound."""
    traj = trajectory_from_spec(
        {"model": "rwp_1d", "p0_m": [0.0, 0.0, 0.0], "p1_m": [L_M, 0.0, 0.0], "speed_mps": V},
        node="mu",
        seed=SEED,
    )
    _wps, xs = walk
    n = xs.size
    n_legs = len(traj.leg_starts(n / 10 + 2 * L_M / V))  # legs covering the walk
    w3 = [traj.waypoint(i) for i in range(n_legs + 1)]
    wps = np.array([w[0] for w in w3])
    durations = [float(np.linalg.norm(w3[i + 1] - w3[i])) / V for i in range(n_legs)]
    t_end = np.cumsum(durations)  # sequential float accumulation, as before the fix
    t = np.arange(n) / 10.0  # float absolute times
    i = np.searchsorted(t_end, t, side="right")
    t0 = np.where(i > 0, t_end[np.maximum(i - 1, 0)], 0.0)
    legacy = wps[i] + ((t - t0) / (t_end[i] - t0)) * (wps[i + 1] - wps[i])
    assert np.max(np.abs(legacy - xs)) > 0.0
    assert np.any(np.abs(np.diff(legacy)) > V * 0.1 + 1e-12)


def test_st01_teeth_uniform_positions_fail_the_stationary_cdf() -> None:
    """The KS check has teeth: positions uniform on [0, l] (the start-point law, not the
    stationary law) fail D <= 0.02 at the same sample size."""
    from sremu import rng

    xs = np.sort(L_M * rng.uniform(rng.derive_key(SEED, "trial", "st01/teeth"), 0, 333_000))
    assert ks_statistic(xs, lambda x: rwp1d_cdf(x, L_M)) > 0.02
