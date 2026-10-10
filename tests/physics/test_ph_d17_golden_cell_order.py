"""D17: the physics panel grid uses the contract cell order of the MATLAB goldens.

``PanelConfig.state`` index ``n = iv*n_u + iu`` with ``iu`` along ``u_axis`` and ``iv`` along
``v = normal x u_axis``; inert tile gaps (spacing pitch + gap across a tile boundary); cells
centred on ``pose.center_m``. The oracle is the per-cell CSV written by the MATLAB golden
scripts (``tests/golden/matlab_R2026a/*_cells.csv``, columns ``n, iu, iv, x, y, z``): row
``n`` must be the centre of ``physics.geometry.PanelGrid`` cell ``n``, and the G1 per-cell
states must be the per-tile states expanded through ``PanelGrid.physical_tile_of_cell``.
So a golden state column drives ``ScenePropagation`` unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sremu.physics.geometry import PanelGrid, Pose
from tests.golden.harness import load_csv, load_meta

pytestmark = pytest.mark.physics


def _grid(p: dict[str, Any], n_u: int, n_v: int, pu: float, pv: float) -> PanelGrid:
    return PanelGrid(
        Pose(p["center_m"], p["normal"], p["u_axis"]),
        n_u,
        n_v,
        pu,
        pv,
        int(p.get("n_tiles_u", 1)),
        int(p.get("n_tiles_v", 1)),
        float(p.get("gap_m", 0.0)),
    )


def _panels() -> list[tuple[str, dict[str, np.ndarray], PanelGrid, float]]:
    out = []
    c1 = load_csv("G1_cells.csv")
    for geom, g in load_meta("G1_lab_polar")["parameters"]["geometries"].items():
        sel = c1["geometry"] == geom
        grid = _grid(g, g["n_u"], g["n_v"], g["pitch_u_m"], g["pitch_v_m"])
        out.append((f"G1/{geom}", {k: v[sel] for k, v in c1.items()}, grid, 1e-12))
    prm = load_meta("G2_beam_pattern")["parameters"]
    grid = _grid(prm["panel"], prm["n_u"], prm["n_v"], prm["pitch_m"], prm["pitch_m"])
    out.append(("G2_G3", load_csv("G2_G3_cells.csv"), grid, 1e-12))
    pan = load_meta("G7_fm_etsi")["parameters"]["panel"]
    grid = _grid(pan, pan["n_u"], pan["n_v"], pan["pitch_u_m"], pan["pitch_v_m"])
    out.append(("G7", load_csv("G7_cells.csv"), grid, 1e-9))  # 80 km coordinates
    return out


PANELS = _panels()


@pytest.mark.parametrize("panel", PANELS, ids=[p[0] for p in PANELS])
def test_d17_physics_grid_matches_golden_cell_files(panel):  # type: ignore[no-untyped-def]
    _, cols, grid, tol_m = panel
    n = np.arange(grid.n_cells)
    assert cols["n"].size == grid.n_cells
    assert np.array_equal(cols["n"].astype(np.int64), n)
    assert np.array_equal(cols["iu"].astype(np.int64), n % grid.n_u)
    assert np.array_equal(cols["iv"].astype(np.int64), n // grid.n_u)
    golden = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    assert np.max(np.abs(grid.cell_centres() - golden)) <= tol_m
    # v = normal x u: moving one row (iv + 1) inside a tile steps by one pitch along v
    v = np.cross(grid.pose.normal, grid.pose.u_axis)
    if grid.n_v > 1 and grid.n_v // grid.n_tiles_v > 1:
        step = golden[grid.n_u] - golden[0]
        assert np.max(np.abs(step - grid.pitch_v_m * v)) <= tol_m


def test_d17_physics_tile_index_matches_golden_states():
    """G1 lab layouts: per-cell state = per-tile state at ``tv*n_tiles_u + tu``."""
    for label, cols, grid, _ in PANELS:
        if not label.startswith("G1/lab"):
            continue
        g = load_meta("G1_lab_polar")["parameters"]["geometries"][label.removeprefix("G1/")]
        tile_state = np.asarray(g["tile_states"]).ravel()  # row-major (tv, tu)
        assert tile_state.size == grid.n_tiles_u * grid.n_tiles_v
        got = tile_state[grid.physical_tile_of_cell()]
        assert np.array_equal(cols["state"].astype(np.int64), got)
