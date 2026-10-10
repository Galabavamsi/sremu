"""FakePropagation against the contract 1.1 protocol additions (D11, D17), closed forms only."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PanelConfig, PathKind, PropagationModel
from sremu.errors import ConfigurationError, UnknownTarget
from sremu.testing import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import doppler_t9, multi_rate_streams, near_far_rx, stream
from sremu.units import C0

pytestmark = pytest.mark.unit

PITCH = 0.0428  # near_far_rx panel: 24 x 5 cells, 3 x 1 render tiles, centre at the origin


def test_d11_link_fc_hz_is_the_tx_carrier():
    prop = multi_rate_streams()
    assert [prop.link_fc_hz(lk) for lk in prop.links()] == [1.0e9, 2.0e9, 3.0e9]
    with pytest.raises(UnknownTarget):
        prop.link_fc_hz("nobody.tx0->rxa.rx0")


def test_d11_d17_panel_geometry_of_the_fake_panel():
    prop = near_far_rx()
    g = prop.panel_geometry("ris0")
    assert (g.n_u, g.n_v, g.n_quad, g.n_tiles_u, g.n_tiles_v, g.gap_m) == (24, 5, 120, 1, 1, 0.0)
    assert (g.extent_u_m, g.extent_v_m) == pytest.approx((24 * PITCH, 5 * PITCH), rel=1e-15)
    assert (g.center_m, g.normal, g.u_axis) == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    # D17: n = iv*n_u + iu, centred cells; the fake panel uses the same layout
    np.testing.assert_allclose(
        g.cell_centres_m(), prop.panels["ris0"].cell_centres(), rtol=0, atol=1e-15
    )
    loc = g.cell_local_m()
    assert loc[1, 0] - loc[0, 0] == pytest.approx(PITCH, rel=1e-12)  # iu runs fastest
    assert loc[24, 1] - loc[0, 1] == pytest.approx(PITCH, rel=1e-12)
    with pytest.raises(UnknownTarget):
        prop.panel_geometry("ris9")


@pytest.mark.parametrize("tile", [0, 1, 2])
def test_d11_ris_delay_per_tile_hand_computed_and_consistent_with_tile_delay(tile):
    """Tile k of 3 along u covers iu in [8k, 8k + 8): its centroid is u = (8k - 8)*pitch on
    the panel's u axis (+y), v = 0. RX ``rx1`` sits at (1.2, 0, 0), TX at (1.7320508, -1, 0)."""
    prop = near_far_rx()
    t = Fraction(0)
    ref = np.array([0.0, (8 * tile - 8) * PITCH, 0.0])
    p_rx = np.array([1.2, 0.0, 0.0])
    p_tx = np.array([1.7320508, -1.0, 0.0])
    tau = prop.ris_delay_s("ris0", "rx1.rx0", t, tile=tile)
    assert tau * C0 == pytest.approx(float(np.linalg.norm(p_rx - ref)), rel=1e-14)
    rc = prop.snapshot(t).ris["ris0"]
    r = [lk for lk, _ in rc.rows].index("ap.tx0->rx1.rx0")
    assert rc.tile_delay_s[r, tile] == pytest.approx(
        tau + float(np.linalg.norm(p_tx - ref)) / C0, rel=1e-15
    )
    assert prop.ris_delay_s("ris0", "rx1.rx0", t) * C0 == pytest.approx(1.2, rel=1e-15)


def test_d11_ris_delay_rejects_bad_tiles_and_targets():
    prop = near_far_rx()
    for bad in (3, -1, True, 1.5):
        with pytest.raises(ConfigurationError):
            prop.ris_delay_s("ris0", "rx1.rx0", Fraction(0), tile=bad)  # type: ignore[arg-type]
    with pytest.raises(UnknownTarget):
        prop.ris_delay_s("ris9", "rx1.rx0", Fraction(0))
    with pytest.raises(UnknownTarget):
        prop.ris_delay_s("ris0", "ghost.rx0", Fraction(0))


def test_d11_with_render_tiles_returns_a_retiled_copy():
    base = doppler_t9(with_panel=True, tiles=1)
    tiled = base.with_render_tiles({"ris0": (3, 2)})
    assert isinstance(tiled, PropagationModel)
    t = Fraction(1, 7)
    rc0, rc1 = base.snapshot(t).ris["ris0"], tiled.snapshot(t).ris["ris0"]
    assert (rc0.n_tiles, rc1.n_tiles) == (1, 6)  # the original is unchanged
    n = np.arange(144)  # 12 x 12 cells: k = tv*3 + tu, tu = iu*3 // 12, tv = iv*2 // 12
    want = ((n // 12) * 2 // 12) * 3 + (n % 12) * 3 // 12
    assert rc1.tile_of_cell.tolist() == want.tolist()
    # tiles only change the reference split: the exact-phase sum over RIS paths is invariant
    state = np.zeros(144, np.int16)
    state[::3] = 1
    cfg = {"ris0": PanelConfig("ris0", state)}
    (lk,) = base.links()
    h = []
    for model in (base, tiled):
        lp = model.combine(model.snapshot(t), cfg)[lk]
        ris = lp.kind == PathKind.RIS_TILE
        h.append(complex(np.sum(lp.amp[ris] * np.exp(-2j * math.pi * 3.5e9 * lp.delay_s[ris]))))
    assert abs(h[1] - h[0]) <= 1e-9 * abs(h[0])
    with pytest.raises(UnknownTarget):
        base.with_render_tiles({"ris9": (2, 2)})
    with pytest.raises(ConfigurationError):
        base.with_render_tiles({"ris0": (13, 1)})  # more tiles than cells


def _two_nodes(external: bool = True, panel: bool = False) -> FakePropagation:
    nodes = [
        FakeNode("ap", (0.0, 0.0, 0.0), streams=(stream("ap.tx0", "tx"),)),
        FakeNode("mu", (10.0, 0.0, 0.0), streams=(stream("mu.rx0", "rx"),), external=external),
    ]
    panels = []
    if panel:
        panels.append(
            FakePanel("ris0", (6.0, 0.0, 3.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), 2, 2, 0.05, 0.05)
        )
    return FakePropagation(nodes, panels, horizon_s=4)


def _los_m(prop: FakePropagation, t: Fraction) -> float:
    (lk,) = prop.links()
    return float(prop.snapshot(t).paths[lk].delay_s[0]) * C0


def test_d11_set_node_position_holds_from_the_event_time():
    prop = _two_nodes()
    prop.set_node_position("mu", Fraction(1, 2), (3.0, 4.0, 0.0))
    assert _los_m(prop, Fraction(1, 4)) == pytest.approx(10.0, rel=1e-15)
    assert _los_m(prop, Fraction(1, 2)) == pytest.approx(5.0, rel=1e-15)  # applies at t
    assert _los_m(prop, Fraction(3)) == pytest.approx(5.0, rel=1e-15)
    prop.set_node_position("mu", "1/2", (6.0, 8.0, 0.0))  # same time: the later event wins
    assert _los_m(prop, Fraction(1, 2)) == pytest.approx(10.0, rel=1e-15)
    with pytest.raises(ConfigurationError, match="time order"):
        prop.set_node_position("mu", Fraction(1, 4), (1.0, 0.0, 0.0))
    with pytest.raises(ConfigurationError, match="external"):
        _two_nodes(external=False).set_node_position("mu", Fraction(1), (1.0, 0.0, 0.0))
    with pytest.raises(UnknownTarget):
        prop.set_node_position("ghost", Fraction(1), (1.0, 0.0, 0.0))


def test_d11_set_node_trajectory_follows_time_since_the_event():
    from sremu.physics.geometry import ConstantVelocity

    prop = _two_nodes()
    prop.set_node_trajectory(
        "mu", Fraction(1), ConstantVelocity(np.array([6.0, 8.0, 0.0]), np.array([0.0, -4.0, 0.0]))
    )
    # p(t) = (6, 8 - 4 (t - 1), 0): at t = 3 the RX is at (6, 0, 0)
    assert prop.node_position("mu", Fraction(3)) == pytest.approx([6.0, 0.0, 0.0], abs=1e-15)
    assert _los_m(prop, Fraction(3)) == pytest.approx(6.0, rel=1e-15)
    assert _los_m(prop, Fraction(1, 2)) == pytest.approx(10.0, rel=1e-15)
    with pytest.raises(TypeError, match="constant-velocity"):
        prop.set_node_trajectory("mu", Fraction(2), object())


def test_d11_motion_bounds_exact_for_piecewise_motion():
    """RX: static at (10, 0, 0) until t = 1, then (6, 8 - 4(t - 1), 0) until the 4 s horizon.
    Closest approach to the TX (origin): 6 m at t = 3; to the panel centre (6, 0, 3): 3 m."""
    from sremu.physics.geometry import ConstantVelocity

    prop = _two_nodes(panel=True)
    mb0 = prop.motion_bounds()
    (lk,) = prop.links()
    assert mb0.static and mb0.r_min_m[lk] == pytest.approx(10.0) and mb0.v_rel_max_mps[lk] == 0
    assert mb0.panel_r_min_m["ris0"] == pytest.approx(5.0)  # (10, 0, 0) to (6, 0, 3)
    prop.set_node_trajectory(
        "mu", Fraction(1), ConstantVelocity(np.array([6.0, 8.0, 0.0]), np.array([0.0, -4.0, 0.0]))
    )
    mb = prop.motion_bounds()
    assert not mb.static
    assert mb.v_rel_max_mps[lk] == pytest.approx(4.0, rel=1e-15)
    assert mb.r_min_m[lk] == pytest.approx(6.0, rel=1e-12)
    assert mb.panel_r_min_m["ris0"] == pytest.approx(3.0, rel=1e-12)
    assert mb.panel_v_rel_max_mps["ris0"] == pytest.approx(4.0, rel=1e-15)
    # an event after the horizon does not change the bounds
    prop.set_node_position("mu", Fraction(5), (0.5, 0.0, 0.0))
    assert prop.motion_bounds().r_min_m[lk] == pytest.approx(6.0, rel=1e-12)


def test_d11_fake_satisfies_the_v11_protocol():
    for prop in (near_far_rx(), _two_nodes(panel=True), doppler_t9(with_panel=True)):
        assert isinstance(prop, PropagationModel)
