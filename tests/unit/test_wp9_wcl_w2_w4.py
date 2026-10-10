"""V-WCL W2 and W4 (optimum and heuristic parts) for the WP9 controllers (DESIGN.md
section 10.11).

* W2, Theorem 1 of the letter on its corridor geometry (AP at the origin of the user line,
  RIS line at distance h, element n at abscissa ``y_n = (n - 1/2) k``): argmin_n D_n(x) lies
  within one element pitch of x/2 (items 1-2), the equal-phase element of two user
  positions is at their midpoint (item 4), phi(x) < phi_n(x) (item 5) and the
  monotonicity of item 3 in the form that holds (see ``test_w2_item3_*``).
* W4, Fig. 6 geometry (l = 3 m, h = 2 m, N = 20 cells of lambda/4 at 1 GHz, AP on the user
  line): the ``ExactOptimum`` controller's decision equals the brute-force optimum over all
  2^20 configurations (rel <= 1e-12) at every tested position; the ``WclHeuristic`` (and
  URAP) power ratios to the optimum are reported (junit properties), with the structural
  bound P_SO <= P_o asserted.

The scene uses the real K1 physics (``ScenePropagation`` with the ``on_off`` table); the
brute force is an independent exhaustive evaluation of the same CSI.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import StreamSpec
from sremu.control.builtin import (
    ExactOptimum,
    WclHeuristic,
    wcl_corridor_lengths,
    wcl_phases,
)
from sremu.control.controller import (
    GridChannel,
    make_observation,
    setup_controller,
)
from sremu.physics.geometry import ConstantVelocity, PanelGrid, Pose, Static
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation
from sremu.physics.response import builtin
from sremu.units import C0

pytestmark = pytest.mark.unit

FC = 1.0e9
LAM = C0 / FC
K = LAM / 4  # element width w = length k = lambda/4 (letter section IV-A)


# =========================================================================== W2 (Theorem 1)

W2_CASES = [  # (l, h, N, f_c): the letter's corridor at four heights and its Fig. 3 setup
    (10.0, 1.0, 133, 1.0e9),
    (10.0, 2.0, 133, 1.0e9),
    (10.0, 4.0, 133, 1.0e9),
    (10.0, 6.0, 133, 1.0e9),
    (25.0, 5.0, 333, C0 / 0.3),
]


def _y(n_cells: int, fc: float) -> np.ndarray:
    k = C0 / fc / 4
    return (np.arange(1, n_cells + 1) - 0.5) * k


@pytest.mark.parametrize(("l_m", "h_m", "n_cells", "fc"), W2_CASES)
def test_w2_item1_argmin_dn_within_one_pitch_of_half_x(l_m, h_m, n_cells, fc):
    """Theorem 1.1: y(argmin_n D_n(x)) = x/2, within one element pitch on the discrete line;
    D_n is unimodal in n (strictly decreasing, then strictly increasing)."""
    y = _y(n_cells, fc)
    k = y[1] - y[0]
    xs = np.linspace(0.01 * l_m, l_m, 2001)
    d, _ = wcl_corridor_lengths(xs, h_m, y)
    nstar = np.argmin(d, axis=1)
    inside = xs / 2 <= y[-1]  # x/2 must lie on the RIS line for the statement to apply
    assert np.all(np.abs(y[nstar] - xs / 2)[inside] <= k)
    for row, n in zip(d, nstar, strict=True):
        assert np.all(np.diff(row[: n + 1]) < 0) and np.all(np.diff(row[n:]) > 0)


@pytest.mark.parametrize(("l_m", "h_m", "n_cells", "fc"), W2_CASES)
def test_w2_item2_phase_argmin_matches_dn_argmin(l_m, h_m, n_cells, fc):
    """Theorem 1.2: phi_n(x) = 2 pi f_c D_n(x)/c has the same argmin (exact: same index)."""
    y = _y(n_cells, fc)
    xs = np.linspace(0.01 * l_m, l_m, 501)
    phi_n, _ = wcl_phases(xs, h_m, y, fc)
    d, _ = wcl_corridor_lengths(xs, h_m, y)
    np.testing.assert_array_equal(np.argmin(phi_n, axis=1), np.argmin(d, axis=1))
    np.testing.assert_allclose(phi_n, 2 * math.pi * fc * d / C0, rtol=1e-15)


def test_w2_continuous_minimizer_is_exactly_half_x():
    """Theorem 1.1 on a continuous line: dD/dy = 0 at y = x/2 (closed form, to rounding)."""
    for h in (1.0, 2.0, 4.0, 6.0):
        for x in (0.3, 1.0, 4.5, 9.9):
            y = np.array([x / 2])
            eps = 1e-6
            d0 = wcl_corridor_lengths(x, h, y)[0][0]
            dm = wcl_corridor_lengths(x, h, y - eps)[0][0]
            dp = wcl_corridor_lengths(x, h, y + eps)[0][0]
            assert dm > d0 and dp > d0
            assert abs(d0 - 2 * math.hypot(x / 2, h)) <= 1e-15 * d0 * 4


@pytest.mark.parametrize(("l_m", "h_m", "n_cells", "fc"), W2_CASES)
def test_w2_item3_phase_increases_with_x_beyond_the_element(l_m, h_m, n_cells, fc):
    """Theorem 1.3 in the form that holds exactly: phi_n(x) is strictly increasing in x for
    x > y_n (and strictly decreasing for x < y_n: phi_n is unimodal in x with its minimum at
    x = y_n), and the phase of the shortest RIS path, min_n phi_n(x), is strictly
    increasing in x once x/2 lies on the RIS line (x >= 2 y_1; continuous line: for every
    x > 0). (An unrestricted "phi_n increasing in x for every n" would contradict item 4,
    which has phi_n(x1) = phi_n(x2) at y_n = (x1 + x2)/2.)"""
    y = _y(n_cells, fc)
    xs = np.linspace(0.0, l_m, 4001)
    phi_n, _ = wcl_phases(xs, h_m, y, fc)
    dphi = np.diff(phi_n, axis=0)  # (x steps, n)
    after = xs[:-1, None] >= y[None, :]
    before = xs[1:, None] <= y[None, :]
    assert np.all(dphi[after] > 0)
    assert np.all(dphi[before] < 0)
    assert np.any(before)  # the decreasing branch exists for every geometry tested
    on_line = xs[:-1] >= 2 * y[0]
    assert np.all(np.diff(np.min(phi_n, axis=1))[on_line] > 0)
    cont = 2 * np.hypot(xs / 2, h_m)  # min over a continuous RIS line: 2 sqrt((x/2)^2 + h^2)
    assert np.all(np.diff(cont)[xs[:-1] > 0] > 0)
    assert np.all(np.min(wcl_corridor_lengths(xs, h_m, y)[0], axis=1) >= cont * (1 - 1e-15))


@pytest.mark.parametrize(("l_m", "h_m", "n_cells", "fc"), W2_CASES)
def test_w2_item4_equal_phase_element_is_the_midpoint(l_m, h_m, n_cells, fc, rng):
    """Theorem 1.4: phi_n(x1) = phi_n(x2) exactly at y = (x1 + x2)/2 and nowhere else (the
    difference changes sign there and only there)."""
    for _ in range(20):
        x1, x2 = sorted(rng.uniform(0.0, l_m, 2))
        ym = np.array([(x1 + x2) / 2])
        d1 = wcl_corridor_lengths(x1, h_m, ym)[0][0]
        d2 = wcl_corridor_lengths(x2, h_m, ym)[0][0]
        assert abs(d1 - d2) <= 4e-16 * d1 * 4
        yy = np.linspace(-l_m, 2 * l_m, 3001)
        diff = wcl_corridor_lengths(x1, h_m, yy)[0] - wcl_corridor_lengths(x2, h_m, yy)[0]
        sign_changes = np.nonzero(np.diff(np.sign(diff)) != 0)[0]
        assert sign_changes.size == 1
        assert yy[sign_changes[0]] <= ym[0] <= yy[sign_changes[0] + 1]


@pytest.mark.parametrize(("l_m", "h_m", "n_cells", "fc"), W2_CASES)
def test_w2_item5_los_phase_below_every_ris_phase(l_m, h_m, n_cells, fc):
    """Theorem 1.5: phi(x) < phi_n(x) for every n (D_n > x since h > 0)."""
    y = _y(n_cells, fc)
    xs = np.linspace(0.0, l_m, 1001)
    phi_n, phi = wcl_phases(xs, h_m, y, fc)
    assert np.all(phi[:, None] < phi_n)


# =========================================================================== W4 (Fig. 6)

N_FIG6 = 20
H_FIG6 = 2.0
L_FIG6 = 3.0
X0 = Fraction(1, 10)  # first grid position, m
DX = Fraction(1, 10)  # grid step, m (the MU moves at 1 m/s, so t = (x - x0)/v)
N_GRID = 29  # x = 0.1 ... 2.9 m


def _fig6_scene() -> ScenePropagation:
    """AP at the origin of the user line (letter geometry), MU moving along +x at 1 m/s
    from x0, RIS line at y = h with N = 20 cells of lambda/4 along x (u axis +x, normal -y),
    cell n at x = (n + 1/2) k, the ``on_off`` table (0 = OFF, absorbing; 1 = ON)."""
    rate = Fraction(10_000)
    ap = NodeModel("ap", Static(np.zeros(3)), streams=(StreamSpec("ap.tx0", "tx", FC, rate),))
    mu = NodeModel(
        "mu",
        ConstantVelocity(np.array([float(X0), 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        streams=(StreamSpec("mu.rx0", "rx", FC, rate),),
    )
    pose = Pose(
        np.array([N_FIG6 * K / 2, H_FIG6, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([1.0, 0, 0])
    )
    grid = PanelGrid(pose, N_FIG6, 1, K, K)
    panel = PanelModel("ris0", grid, builtin("on_off"))
    return ScenePropagation([ap, mu], [panel], horizon_s=3)


@pytest.fixture(scope="module")
def fig6():
    model = _fig6_scene()
    times = [i * DX for i in range(N_GRID)]
    channel = GridChannel.from_model(model, "mu", times, links=("ap.tx0->mu.rx0",))
    return model, channel


def _subset_sums(z: np.ndarray) -> np.ndarray:
    """All 2^N sums of an (N,) vector: bit n of the index selects z[n] (split in halves)."""
    n = z.size
    lo = n // 2

    def sums(v: np.ndarray) -> np.ndarray:
        idx = np.arange(1 << v.size)
        bits = (idx[:, None] >> np.arange(v.size)[None, :]) & 1
        return bits @ v

    return (sums(z[:lo])[:, None] + sums(z[lo:])[None, :]).reshape(-1)


def _caps(model: ScenePropagation):
    return {"ris0": model.capabilities("ris0")}


def test_w4_grid_positions_follow_the_trajectory(fig6):
    model, channel = fig6
    assert len(channel) == N_GRID
    x = channel.points[:, 0]
    np.testing.assert_allclose(x, [float(X0 + i * DX) for i in range(N_GRID)], rtol=0, atol=1e-15)
    cells = model.panel_geometry("ris0").cell_centres_m()
    np.testing.assert_allclose(cells[:, 0], (np.arange(N_FIG6) + 0.5) * K, rtol=0, atol=1e-15)
    np.testing.assert_allclose(cells[:, 1], H_FIG6, rtol=0, atol=0)


def test_w4_exact_optimum_equals_brute_force_2pow20(fig6, record_property):
    """W4: ExactOptimum's decision equals the exhaustive optimum over 2^20 configurations at
    every grid position (rel <= 1e-12), and its predicted |h| equals the evaluated one."""
    model, channel = fig6
    ctrl = ExactOptimum(link="ap.tx0->mu.rx0", channel=channel, skip_unchanged=False)
    setup_controller(ctrl, _caps(model), seed=7)
    worst = 0.0
    for i in range(N_GRID):
        pos = {"mu": channel.points[i], "ap": np.zeros(3)}
        obs = make_observation(i * DX, ris_state={"ris0": np.zeros(N_FIG6)}, positions=pos)
        (act,) = ctrl.decide(obs)
        view = channel.views[i]["ap.tx0->mu.rx0"]
        z = view.cascade["ris0"]
        assert np.all(z[0] == 0)  # OFF is absorbing
        brute = float(np.max(np.abs(view.h_direct + _subset_sums(z[1]))))
        got = abs(view.evaluate({"ris0": act.states}))
        assert abs(got - brute) <= 1e-12 * brute
        assert abs(ctrl.last["abs_h"] - got) <= 1e-12 * brute
        worst = max(worst, abs(got - brute) / brute)
    record_property("W4_exact_vs_bruteforce_max_rel", float(worst))


def _urap_mean_power(z0: complex, z1: np.ndarray, p: float) -> float:
    """E|z0 + sum_n b_n z1_n|^2 with b_n ~ Bernoulli(p) i.i.d. (mean field + variance)."""
    return abs(z0 + p * z1.sum()) ** 2 + p * (1 - p) * float(np.sum(np.abs(z1) ** 2))


@pytest.mark.parametrize("phase_source", ["phasor", "path_length"])
def test_w4_heuristic_ratio_to_optimum_reported(fig6, phase_source, record_property):
    """W4 heuristic part: P_SO/P_o of the letter's Eqs. (7)-(8) (K = 24 candidates) at each
    grid position; reported, with P_SO <= P_o (the optimum is exact) asserted. Also reported:
    URAP (p = 0.2) and LOS-only relative to P_o, and the letter's "15-60 %" under-performance
    for comparison (not a pass criterion; the kernels differ)."""
    model, channel = fig6
    cells = model.panel_geometry("ris0").cell_centres_m()
    heur = WclHeuristic(
        link="ap.tx0->mu.rx0",
        channel=channel,
        phase_source=phase_source,
        cell_centres_m=cells,
        skip_unchanged=False,
    )
    assert heur.n_candidates == 24
    setup_controller(heur, _caps(model), seed=7)
    ratios, urap, los = [], [], []
    for i in range(N_GRID):
        pos = {"mu": channel.points[i], "ap": np.zeros(3)}
        obs = make_observation(i * DX, ris_state={"ris0": np.zeros(N_FIG6)}, positions=pos)
        (act,) = heur.decide(obs)
        view = channel.views[i]["ap.tx0->mu.rx0"]
        z1 = view.cascade["ris0"][1]
        p_o = float(np.max(np.abs(view.h_direct + _subset_sums(z1)))) ** 2
        p_so = abs(view.evaluate({"ris0": act.states})) ** 2
        assert abs(math.sqrt(p_so) - heur.last["abs_h"]) <= 1e-12 * math.sqrt(p_o)
        assert p_so <= p_o * (1 + 1e-12)
        ratios.append(p_so / p_o)
        urap.append(_urap_mean_power(view.h_direct, z1, 0.2) / p_o)
        los.append(abs(view.h_direct) ** 2 / p_o)
    r = np.asarray(ratios)
    tag = f"W4_{phase_source}"
    record_property(f"{tag}_PSO_over_Po_min", float(r.min()))
    record_property(f"{tag}_PSO_over_Po_mean", float(r.mean()))
    record_property(f"{tag}_PSO_over_Po_max", float(r.max()))
    record_property(f"{tag}_underperformance_pct_min", float(100 * (1 - r.max())))
    record_property(f"{tag}_underperformance_pct_max", float(100 * (1 - r.min())))
    record_property("W4_URAP_over_Po_mean", float(np.mean(urap)))
    record_property("W4_LOS_over_Po_mean", float(np.mean(los)))
    print(
        f"\n{tag}: P_SO/P_o min {r.min():.4f} mean {r.mean():.4f} max {r.max():.4f}; "
        f"URAP/P_o mean {np.mean(urap):.4f}; LOS/P_o mean {np.mean(los):.4f}"
    )
    assert np.all((r > 0) & (r <= 1 + 1e-12))
