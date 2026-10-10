"""Contract 1.1 (lead decisions v1.2: D3, D9, D10, D11, D17, D20).

Expected values are written out by hand from the decisions' text (D17 cell layout, D10 tile
keys), not read back from the implementation.
"""

from __future__ import annotations

import dataclasses
import inspect
from fractions import Fraction

import numpy as np
import pytest

from sremu import contracts as c
from sremu.errors import ConfigurationError

pytestmark = pytest.mark.unit

LINK = "a.tx0->b.rx0"


# --------------------------------------------------------------------------- D3


def test_d3_message_cap_is_one_mib():
    assert c.MAX_MESSAGE_BYTES == 1_048_576 == 2**20
    assert "1 MiB" in _attr_doc("MAX_MESSAGE_BYTES")


def _attr_doc(name: str) -> str:
    src = inspect.getsource(c)
    i = src.index(f"{name}: Final")
    return src[i : i + 300]


# --------------------------------------------------------------------------- D10


def _bundle(kind):
    n = len(kind)
    return c.PathBundle(
        LINK,
        Fraction(0),
        np.full(n, 1e-3 + 0j),
        np.linspace(1e-8, 2e-8, n),
        np.asarray(kind, dtype=np.int8),
    )


def test_d10_ris_gap_is_a_path_bundle_kind_and_ris_tile_is_not():
    pb = _bundle([c.PathKind.LOS, c.PathKind.RIS_GAP])
    assert pb.kind.tolist() == [0, 6]
    with pytest.raises(ValueError, match="RIS_TILE"):
        _bundle([c.PathKind.LOS, c.PathKind.RIS_TILE])
    with pytest.raises(ValueError, match="PathKind"):
        _bundle([0, 7])


def _paths(kind, tile):
    n = len(kind)
    return c.LinkPaths(
        LINK,
        Fraction(0),
        np.zeros(n, np.complex128),
        np.zeros(n),
        np.asarray(kind, np.int8),
        np.asarray(tile, np.int32),
    )


def test_d10_link_paths_tile_keys():
    lp = _paths([0, 6, 4, 4], [-1, -1, 0, 1])  # LOS, gap, two RIS tiles
    assert lp.n_paths == 4
    with pytest.raises(ValueError, match="RIS_TILE"):
        _paths([0, 4], [-1, -1])  # a RIS_TILE path without a tile key
    with pytest.raises(ValueError, match="-1"):
        _paths([6, 4], [0, 1])  # a gap path with a tile key
    with pytest.raises(ValueError, match="-1"):
        _paths([0], [3])
    with pytest.raises(ValueError, match="PathKind"):
        _paths([9], [-1])


# --------------------------------------------------------------------------- D11 RIS forms


def _ris(**kw):
    r, s, n, k = 2, 3, 4, 2
    base = {
        "panel": "ris0",
        "t": Fraction(0),
        "rows": ((LINK, 0), (LINK, 1)),
        "tile_of_cell": np.array([0, 0, 1, 1], np.int32),
        "tile_delay_s": np.zeros((r, k)),
        "B": np.ones((r, s, n), np.complex128),
        "C": np.ones((r, n), np.complex128),
        "gamma": None,
    }
    base.update(kw)
    return c.RISCoefficients(**base)


def test_d11_ris_coefficients_may_carry_both_b_and_c():
    both = _ris()
    assert both.B is not None and both.C is not None
    assert (both.separable, both.n_states, both.n_cells, both.n_tiles) == (False, 3, 4, 2)
    with_gamma = _ris(gamma=np.array([1, -1, 1j], np.complex128))
    assert with_gamma.n_states == 3 and not with_gamma.separable
    sep = _ris(B=None, gamma=np.array([1, -1], np.complex128))
    assert sep.separable and sep.n_states == 2


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"gamma": np.array([1, -1], np.complex128)}, "states"),  # S mismatch with B
        ({"C": None, "gamma": np.array([1, -1, 1], np.complex128)}, "gamma needs C"),
        ({"B": None}, "both C and gamma"),  # C alone is not a form
        ({"B": None, "C": None}, "give B"),
        ({"C": np.ones((2, 5), np.complex128)}, "C shape"),
        ({"B": np.ones((2, 3, 5), np.complex128)}, "B shape"),
        ({"C": np.ones((2, 4), np.complex64)}, "dtype"),
    ],
)
def test_d11_ris_coefficients_reject_inconsistent_forms(kw, match):
    with pytest.raises((ValueError, TypeError), match=match):
        _ris(**kw)


# --------------------------------------------------------------------------- D11 protocol

V11_MEMBERS = {
    "links",
    "link_fc_hz",
    "snapshot",
    "combine",
    "ris_delay_s",
    "panel_geometry",
    "motion_bounds",
    "with_render_tiles",
    "set_node_position",
    "set_node_trajectory",
}


def test_d11_propagation_model_members():
    members = {
        name
        for name, value in vars(c.PropagationModel).items()
        if callable(value) and not name.startswith("_")
    }
    assert members == V11_MEMBERS
    sig = inspect.signature(c.PropagationModel.ris_delay_s)
    assert list(sig.parameters) == ["self", "panel", "rx", "t", "tile"]
    assert sig.parameters["tile"].default is None
    sig = inspect.signature(c.PropagationModel.set_node_position)
    assert list(sig.parameters) == ["self", "node", "t", "position_m"]


def test_d11_runtime_check_requires_the_new_members():
    from sremu.testing.scenes import two_node_static

    model = two_node_static()
    assert isinstance(model, c.PropagationModel)

    class OldModel:  # a contract 1.0 model: no link_fc_hz / panel_geometry / node events
        def links(self):
            return ()

        def snapshot(self, t):
            raise NotImplementedError

        def combine(self, snap, cfg):
            raise NotImplementedError

        def ris_delay_s(self, panel, rx, t):
            raise NotImplementedError

        def motion_bounds(self):
            raise NotImplementedError

    assert not isinstance(OldModel(), c.PropagationModel)


def test_d11_link_job_rot_state_is_optional():
    fields = {f.name: f for f in dataclasses.fields(c.LinkJob)}
    assert fields["rot_state"].default is None
    assert "OPTIONAL" in (c.LinkJob.__doc__ or "")


# --------------------------------------------------------------------------- D17 geometry

U_EXPECTED = [-0.175, -0.075, 0.075, 0.175]  # 4 cells, pitch 0.1, 2 tiles, gap 0.05
V_EXPECTED = [-0.1, 0.1]  # 2 cells, pitch 0.2, 1 tile


def _geo(**kw):
    base = {
        "panel": "ris0",
        "center_m": (1.0, 2.0, 3.0),
        "normal": (0.0, -1.0, 0.0),
        "u_axis": (1.0, 0.0, 0.0),
        "n_u": 4,
        "n_v": 2,
        "pitch_u_m": 0.1,
        "pitch_v_m": 0.2,
        "n_tiles_u": 2,
        "gap_m": 0.05,
    }
    base.update(kw)
    return c.PanelGeometry.from_cells(**base)


def test_d17_cell_order_tiles_and_gaps_hand_computed():
    g = _geo()
    assert (g.extent_u_m, g.extent_v_m) == pytest.approx((0.45, 0.4), rel=1e-15)
    assert (g.n_cells, g.q, g.n_quad) == (8, 8, 8)
    assert g.cell_area_m2 == pytest.approx(0.02, rel=1e-15)
    assert g.v_axis == (0.0, 0.0, 1.0)  # normal x u = (0,-1,0) x (1,0,0)
    loc = g.cell_local_m()
    # n = iv*n_u + iu: the first n_u cells are row iv = 0
    want_u = np.array(U_EXPECTED * 2)
    want_v = np.repeat(V_EXPECTED, 4)
    np.testing.assert_allclose(loc[:, 0], want_u, rtol=0, atol=1e-15)
    np.testing.assert_allclose(loc[:, 1], want_v, rtol=0, atol=1e-15)
    # spacing across the tile boundary is pitch + gap
    assert loc[2, 0] - loc[1, 0] == pytest.approx(0.15, abs=1e-15)
    world = g.cell_centres_m()
    np.testing.assert_allclose(
        world, np.stack([1.0 + want_u, np.full(8, 2.0), 3.0 + want_v], axis=1), atol=1e-15
    )
    assert world.mean(axis=0) == pytest.approx([1.0, 2.0, 3.0], abs=1e-15)  # centred
    assert g.physical_tile_of_cell().tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert g.cell_index(3, 1) == 7 and g.cell_index(0, 1) == 4
    with pytest.raises(ConfigurationError):
        g.cell_index(4, 0)


def test_d17_geometry_matches_the_physics_panel_grid():
    """The documented convention is the one ``physics.geometry.PanelGrid`` implements."""
    from sremu.physics.geometry import PanelGrid, Pose

    pose = Pose(np.array([0.3, -0.2, 1.5]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    grid = PanelGrid(pose, 6, 4, 0.03, 0.05, n_tiles_u=3, n_tiles_v=2, gap_m=0.01)
    g = c.PanelGeometry.from_cells(
        "p", pose.center_m, pose.normal, pose.u_axis, 6, 4, 0.03, 0.05,
        n_tiles_u=3, n_tiles_v=2, gap_m=0.01,
    )  # fmt: skip
    assert (g.extent_u_m, g.extent_v_m) == pytest.approx(grid.size_m, rel=1e-15)
    np.testing.assert_allclose(g.cell_local_m(), grid.cell_local(), rtol=0, atol=1e-15)
    np.testing.assert_allclose(g.cell_centres_m(), grid.cell_centres(), rtol=0, atol=1e-15)
    assert np.array_equal(g.physical_tile_of_cell(), grid.physical_tile_of_cell())
    assert g.cell_area_m2 == grid.cell_area_m2


@pytest.mark.parametrize(
    "kw",
    [
        {"normal": (0.0, -2.0, 0.0)},  # not unit
        {"u_axis": (0.6, 0.8, 0.0)},  # not orthogonal to the normal
        {"n_tiles_u": 3},  # does not divide n_u = 4
        {"n_quad": 7},  # Q < N
        {"pitch_u_m": 0.0},
        {"gap_m": -0.01},
        {"center_m": (0.0, 1.0)},
        {"center_m": (0.0, float("nan"), 0.0)},
        {"n_u": 0},
    ],
)
def test_d17_panel_geometry_validation(kw):
    with pytest.raises(ConfigurationError):
        _geo(**kw)


def test_d17_extent_must_match_cells_tiles_and_gap():
    g = _geo()
    with pytest.raises(ConfigurationError, match="extent_u_m"):
        dataclasses.replace(g, extent_u_m=0.4)  # forgets the gap
    assert dataclasses.replace(g, n_quad=72).q == 72  # po_patch sub-cells: Q > N


def test_d17_panel_geometry_is_a_frozen_value():
    a, b = _geo(), _geo()
    assert a == b and hash(a) == hash(b)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.n_u = 3  # type: ignore[misc]
    assert isinstance(a.center_m, tuple) and a.center_m == (1.0, 2.0, 3.0)


# --------------------------------------------------------------------------- docs (D9, D17, D20)


def _doc(obj: object) -> str:
    return " ".join((obj.__doc__ or "").split())  # whitespace-normalized


def test_v11_docstrings_record_the_lead_decisions():
    assert "energy budget is NOT part of ``diag``" in _doc(c.Snapshot)  # D9
    assert "iv*n_u + iu" in _doc(c.PanelConfig)  # D17
    assert "iv*n_u + iu" in _doc(c.RISCoefficients)
    assert "RIS_GAP" in _doc(c.PathKind)  # D10
    noise = _doc(c.NoiseFn)
    assert "sremu.impair.noise" in noise and "sigma**2 = E|w|**2" in noise  # D20
    render = _doc(c.RenderFn)
    assert "render_numpy.render" in render and "render_numba.render" in render
    assert "absolute index 0" in _doc(c.LinkJob)
