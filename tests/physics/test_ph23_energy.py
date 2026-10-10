"""PH-23: energy diagnostic, scattered <= incident per panel (section 10.2; D2, D9).

The far-zone scattered power of the K1 sources is integrated in closed form (hemisphere
integrals of the obliquity-weighted plane waves). With the default obliquity
``sqrt(cos cos)`` it must not exceed the power the panel intercepts for any passive
configuration, for every tested geometry with pitch <= lambda/2 (by Parseval the bound is
exact there; the largest ratio on this set is 0.9997, a uniform 2 m panel lit from 20 m). A
single-cell closed form checks the normalization: ``P_scat/P_inc = pi A / lambda^2``, and a
direct hemisphere quadrature checks the multi-point kernel for every obliquity.

D2: the optional Kirchhoff obliquity ``(cos_i + cos_s)/2`` is not energy-bounded. Using it
emits a ``ValidityWarning``; the excess over the PH-23 test set is measured and recorded
(junit property ``ph23_kirchhoff_excess_db``), and for steered far-field beams it follows the
closed-form obliquity power ratio ``(cos_i + cos_s)^2 / (4 cos_i cos_s)``.

D9: the energy budget is on demand only (never a ``Snapshot.diag`` key);
``ScenePropagation.energy_budget`` evaluates it for a scene's panels.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import dblquad

from sremu import rng as sremu_rng
from sremu.contracts import PanelConfig, StreamSpec
from sremu.errors import ConfigurationError, ValidityWarning
from sremu.physics.geometry import PanelGrid, Pose, Static
from sremu.physics.optimize import continuous_optimum
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation
from sremu.physics.response import builtin
from sremu.physics.ris_kernel import (
    KIRCHHOFF_WARNING,
    energy_budget,
    k1_terms,
    kirchhoff_power_factor,
)
from sremu.units import C0
from tests.physics._closed_forms import make_panel

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
NRM = np.array([0.0, 0.0, 1.0])


def test_ph23_single_cell_closed_form():
    area = (LAM / 2) ** 2
    tx = np.array([0.4, -0.3, 2.0])
    p_scat, p_inc = energy_budget(tx, np.zeros((1, 3)), area, NRM, FC, 1.0)
    assert abs(p_scat / p_inc - math.pi * area / LAM**2) <= 1e-12
    # brute-force hemisphere integral of |h|^2 r^2 (4 pi / lambda^2) for the same cell
    r_far = 1e4
    ci = 2.0 / float(np.linalg.norm(tx))

    def integrand(theta: float, phi: float) -> float:
        cs = math.cos(theta)
        h = area * math.sqrt(ci * cs) / (4 * math.pi * float(np.linalg.norm(tx)) * r_far)
        return abs(h) ** 2 * r_far**2 * 4 * math.pi / LAM**2 * math.sin(theta)

    num, _ = dblquad(integrand, 0, 2 * math.pi, 0, math.pi / 2)
    assert abs(num / p_scat - 1) <= 1e-8


@pytest.mark.parametrize("obliquity", ["sqrt", "kirchhoff", "none"])
def test_ph23_multi_point_kernel_matches_hemisphere_quadrature(obliquity: str):
    """The closed-form kernel (2 pi J1(x)/x; sin x/x and (sin x - x cos x)/x^3 terms) and the
    blocked quadratic form equal a direct numerical integral of the far-zone power
    ``(4 pi / lambda^2) int |sum_q A_q Gamma_q o_q(theta) e^{-jk r_iq} e^{jk rhat.rho_q}
    / (4 pi r_iq)|^2 dOmega`` over the front hemisphere (4 points within 0.8 lambda)."""
    gen = sremu_rng.generator(23, "trial", f"ph23/hemisphere/{obliquity}")
    pts = np.column_stack(
        [gen.uniform(-0.4, 0.4, 4) * LAM, gen.uniform(-0.4, 0.4, 4) * LAM, np.zeros(4)]
    )
    area = (LAM / 3) ** 2
    gam = gen.uniform(0.3, 1.0, 4) * np.exp(1j * gen.uniform(0, 2 * np.pi, 4))
    tx = np.array([0.3, -0.2, 1.1])
    k = 2 * math.pi / LAM
    d = tx[None, :] - pts
    ri = np.linalg.norm(d, axis=1)
    ci = d[:, 2] / ri
    src = area * gam * np.exp(-1j * k * ri) / (4 * math.pi * ri)

    def integrand(theta: float, phi: float) -> float:
        cs = math.cos(theta)
        rhat = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), cs])
        o = {"sqrt": np.sqrt(ci * cs), "kirchhoff": (ci + cs) / 2, "none": np.ones(4)}[obliquity]
        e = np.sum(src * o * np.exp(1j * k * (pts @ rhat)))
        return float(abs(e) ** 2) * 4 * math.pi / LAM**2 * math.sin(theta)

    num, _ = dblquad(integrand, 0, 2 * math.pi, 0, math.pi / 2, epsabs=0, epsrel=1e-11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)  # Kirchhoff warns by design (D2)
        p_scat, _ = energy_budget(tx, pts, area, NRM, FC, gam, obliquity=obliquity)  # type: ignore[arg-type]
    assert abs(p_scat / num - 1) <= 1e-8, (p_scat, num)


def test_ph23_row_blocks_do_not_change_the_result(monkeypatch: pytest.MonkeyPatch):
    """One-row blocks (forced) give the default-block result to rounding (rel <= 1e-13)."""
    import sremu.physics.ris_kernel as rk

    gen = sremu_rng.generator(23, "trial", "ph23/blocks")
    pos, area = make_panel(0.4, 0.4, 9, 9)
    tx = np.array([0.2, 0.1, 0.9])
    configs = np.exp(1j * gen.uniform(0, 2 * np.pi, (3, 81)))
    ref, p_inc = energy_budget(tx, pos, area, NRM, FC, configs)
    monkeypatch.setattr(rk, "_ENERGY_BLOCK_ELEMS", 1)
    got, p_inc2 = energy_budget(tx, pos, area, NRM, FC, configs)
    assert p_inc2 == p_inc
    assert np.max(np.abs(got / ref - 1)) <= 1e-13


CASES = [
    (0.6, 0.5, (0.0, 0.0, 0.5)),
    (0.6, 0.25, (-0.8, 0.2, 1.0)),
    (1.0, 0.5, (2.0, 0.0, 0.7)),
    (2.0, 0.5, (0.0, 0.0, 20.0)),
]


def _configs(side: float, pitch_frac: float, tx: np.ndarray, rng: np.random.Generator):  # type: ignore[no-untyped-def]
    ns = math.ceil(side / (pitch_frac * LAM))
    pos, area = make_panel(side, side, ns, ns)
    configs = [np.ones(ns * ns, complex)]  # uniform
    for _ in range(3):  # random passive configurations
        configs.append(rng.uniform(0, 1, ns * ns) * np.exp(1j * rng.uniform(0, 2 * np.pi, ns * ns)))
    for _ in range(3):  # focused toward random receivers (strongly coherent)
        rx = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(0.5, 3)])
        c = k1_terms(tx, rx, pos, area, NRM, FC)[0]
        _, ph = continuous_optimum(0.0, c)
        configs.append(np.exp(1j * ph))
    return pos, area, configs


@pytest.mark.parametrize(("side", "pitch_frac", "tx"), CASES)
def test_ph23_scattered_not_above_incident(side: float, pitch_frac: float, tx: tuple, rng):  # type: ignore[no-untyped-def]
    """Default obliquity sqrt(cos cos): every tested configuration is passive."""
    t = np.asarray(tx, float)
    pos, area, configs = _configs(side, pitch_frac, t, rng)
    p_scat, p_inc = energy_budget(t, pos, area, NRM, FC, np.stack(configs), obliquity="sqrt")
    for ps in p_scat:
        assert 0 < ps <= p_inc * (1 + 1e-9), ps / p_inc
    for m in (0, len(configs) - 1):  # batched == single, bitwise (same block reductions)
        one, _ = energy_budget(t, pos, area, NRM, FC, configs[m], obliquity="sqrt")
        assert one == p_scat[m]


def _kirchhoff_budget(t: np.ndarray, pos: np.ndarray, area: float, configs: np.ndarray):  # type: ignore[no-untyped-def]
    with pytest.warns(ValidityWarning, match="not energy-bounded"):
        return energy_budget(t, pos, area, NRM, FC, configs, obliquity="kirchhoff")


def test_ph23_kirchhoff_obliquity_warns_and_records_excess(record_property):  # type: ignore[no-untyped-def]
    """D2: the non-default Kirchhoff obliquity warns, and the PH-23 test set measures how far
    it exceeds the intercepted power (recorded; the default obliquity passes PH-23 above).

    The configurations are drawn from a keyed generator (``sremu.rng``, purpose ``trial``)
    so the recorded number is reproducible."""
    gen = sremu_rng.generator(20261010, "trial", "ph23/kirchhoff")
    worst = 0.0
    for side, pitch_frac, tx in CASES:
        t = np.asarray(tx, float)
        pos, area, configs = _configs(side, pitch_frac, t, gen)
        p_scat, p_inc = _kirchhoff_budget(t, pos, area, np.stack(configs))
        worst = max(worst, float(np.max(p_scat)) / p_inc)
    excess_db = 10 * math.log10(worst)
    record_property("ph23_kirchhoff_excess_db", round(excess_db, 4))
    # the warning is warranted: Kirchhoff scatters MORE than it intercepts on this set
    assert excess_db > 0.1, excess_db
    # the warning text quotes the measured excess on this test set ("about +0.9 dB")
    assert "+0.9 dB" in KIRCHHOFF_WARNING
    assert abs(excess_db - 0.9) <= 0.3, excess_db


@pytest.mark.parametrize("theta_i_deg", [0.0, 30.0])
@pytest.mark.parametrize("theta_s_deg", [0.0, 40.0, 60.0, 70.0])
def test_ph23_kirchhoff_steered_beam_follows_obliquity_power_ratio(
    theta_i_deg: float, theta_s_deg: float
):
    """D2 model-card statement: a far-field beam steered from theta_i to theta_s carries the
    Kirchhoff/default obliquity power ratio (cos_i + cos_s)^2/(4 cos_i cos_s) times the
    intercepted power (2 m x 2 m panel, lambda/2 cells, Tx 1 km, beam focused at 10 km;
    stationary-phase value at the beam centre; within 0.1 dB up to 70 deg). The same
    beams with the default obliquity stay within the PH-23 bound."""
    side = 2.0
    ns = math.ceil(side / (0.5 * LAM))
    pos, area = make_panel(side, side, ns, ns)
    ti, ts = math.radians(theta_i_deg), math.radians(theta_s_deg)
    t = 1e3 * np.array([-math.sin(ti), 0.0, math.cos(ti)])
    rx = 1e4 * np.array([math.sin(ts), 0.0, math.cos(ts)])
    _, ph = continuous_optimum(0.0, k1_terms(t, rx, pos, area, NRM, FC)[0])
    g = np.exp(1j * ph)
    pk, p_inc = _kirchhoff_budget(t, pos, area, g)
    ps, _ = energy_budget(t, pos, area, NRM, FC, g, obliquity="sqrt")
    closed_db = 10 * math.log10(kirchhoff_power_factor(ti, ts))
    assert abs(10 * math.log10(pk / p_inc) - closed_db) <= 0.1, (pk / p_inc, closed_db)
    assert ps <= p_inc * (1 + 1e-9)
    # the closed form itself: >= 1, equality exactly at |theta_s| = |theta_i|
    assert kirchhoff_power_factor(ti, ts) >= 1.0
    assert kirchhoff_power_factor(ti, -ti) == pytest.approx(1.0, abs=1e-15)


def test_ph23_kirchhoff_warning_from_kernel_and_scene():
    """D2: every entry point of the Kirchhoff option warns; the default never does."""
    pos, area = make_panel(0.3, 0.3, 4, 4)
    tx, rx = np.array([0.1, 0.0, 1.0]), np.array([-0.2, 0.1, 1.2])
    with pytest.warns(ValidityWarning, match="not energy-bounded"):
        k1_terms(tx, rx, pos, area, NRM, FC, obliquity="kirchhoff")
    with pytest.warns(ValidityWarning, match="not energy-bounded"):
        k1_terms(tx, rx, pos, area, NRM, FC, obliquity="kirchhoff", backend="numba_cpu")
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 4, 4, LAM / 2, LAM / 2)
    nodes = [
        NodeModel("ap", Static(tx), streams=(StreamSpec("ap.tx0", "tx", FC, "1e6"),)),  # type: ignore[arg-type]
        NodeModel("mu", Static(rx), streams=(StreamSpec("mu.rx0", "rx", FC, "1e6"),)),  # type: ignore[arg-type]
    ]
    with pytest.warns(ValidityWarning, match="not energy-bounded"):  # at build time
        ScenePropagation(
            nodes, [PanelModel("ris0", grid, builtin("ideal_bbit(1)"), obliquity="kirchhoff")]
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ValidityWarning)
        k1_terms(tx, rx, pos, area, NRM, FC)  # default: no warning
        energy_budget(tx, pos, area, NRM, FC, 1.0)
        m = ScenePropagation(nodes, [PanelModel("ris0", grid, builtin("ideal_bbit(1)"))])
        m.snapshot(Fraction(0))


def test_ph23_d9_energy_is_on_demand_not_in_snapshot_diag():
    """D9: no energy key in ``Snapshot.diag``; ``ScenePropagation.energy_budget`` equals the
    kernel diagnostic on the panel's quadrature points with the configured Gamma, and the
    default obliquity keeps scattered <= incident for passive configurations."""
    grid = PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 12, 12, LAM / 2, LAM / 2)
    tx = np.array([-0.6, 0.2, 1.1])
    rx = np.array([0.8, -0.3, 1.4])
    nodes = [
        NodeModel("ap", Static(tx), streams=(StreamSpec("ap.tx0", "tx", FC, "1e6"),)),  # type: ignore[arg-type]
        NodeModel("mu", Static(rx), streams=(StreamSpec("mu.rx0", "rx", FC, "1e6"),)),  # type: ignore[arg-type]
    ]
    m = ScenePropagation(nodes, [PanelModel("ris0", grid, builtin("ideal_bbit(2)"))])
    snap = m.snapshot(Fraction(0))
    assert not any("energy" in k for k in snap.diag)
    gen = sremu_rng.generator(7, "trial", "ph23/d9")
    for state in (np.zeros(144, np.int16), gen.integers(0, 4, 144).astype(np.int16)):
        cfg = {"ris0": PanelConfig("ris0", state)}
        got = m.energy_budget(Fraction(0), cfg)["ris0/ap.tx0"]
        gam = np.exp(2j * np.pi * state / 4)  # ideal 2-bit alphabet, written out
        pts = grid.cell_centres()
        p_scat, p_inc = energy_budget(tx, pts, grid.cell_area_m2, NRM, FC, gam)
        assert got["p_scat"] == pytest.approx(p_scat, rel=1e-12)
        assert got["p_inc"] == pytest.approx(p_inc, rel=1e-12)
        assert got["ratio"] <= 1.0 + 1e-9
        assert got["ratio_db"] == pytest.approx(10 * math.log10(p_scat / p_inc), abs=1e-12)
    # a dead cell (absorbing) removes its source; a disabled panel is skipped
    dead = np.zeros(144, np.int16)
    dead[:72] = -1
    half = m.energy_budget(0, {"ris0": PanelConfig("ris0", dead)})["ris0/ap.tx0"]
    p_scat_h, _ = energy_budget(
        tx, grid.cell_centres(), grid.cell_area_m2, NRM, FC, np.where(dead >= 0, 1.0, 0.0)
    )
    assert half["p_scat"] == pytest.approx(p_scat_h, rel=1e-12)
    off = {"ris0": PanelConfig("ris0", np.zeros(144, np.int16), enabled=False)}
    assert m.energy_budget(0, off) == {}
    with pytest.raises(ConfigurationError):
        m.energy_budget(0, max_points=100)  # O(Q^2) guard


def test_ph23_teeth_oversized_point_cells_violate():
    """Unsubdivided cells larger than lambda/2 are outside the validity domain: with
    lambda-sized point cells the diagnostic reports more scattered than incident power."""
    pos, area = make_panel(4 * LAM, 4 * LAM, 4, 4)
    p_scat, p_inc = energy_budget(np.array([0.0, 0.0, 3.0]), pos, area, NRM, FC, 1.0)
    assert p_scat > p_inc
