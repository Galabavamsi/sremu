"""RIS configurations of the GNU Radio demonstration (sremu interpreter; numpy only).

Configurations of panel ``ris0`` for the link ``tx_node -> RIS -> rx_node`` of a scene
(``gr_ris_ber.yaml``), as state vectors in sremu's cell order (``n = iv * n_u + iu``,
:class:`sremu.physics.geometry.PanelGrid`); the ``ideal_antipodal`` library maps state 0 to
reflection +1 and state 1 to -1.

* ``focus``: 1-bit geometric focusing: each cell's two-hop phasor
  ``exp(-j k (d1 + d2)) / (d1 d2)`` is projected on a reference phase and the cell is set to
  -1 where the projection is negative; the reference phase (181 values in [0, pi]) that
  maximises the array factor is kept;
* ``focusq:Q``: ``focus`` with a seeded fraction ``Q`` of the cells inverted (permutation seed
  77; the same cells for every run);
* ``uniform``: every cell in state 0;
* ``random``: seeded random states (seed 2026).

:func:`array_factor_db` is the isotropic point-cell array factor of a configuration relative
to ``focus``. It ignores cell patterns and obliquity, so comparing it with the gain the
receiver sees is an internal consistency check of the emulator against a simplified model,
never a measurement or an accuracy statement.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sremu.physics.geometry import PanelGrid, Pose
from sremu.units import C0

FOCUS_Q = (0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
"""Fractions of inverted cells of the static sweep."""
STATIC = ("focus", *(f"focusq:{q:.2f}" for q in FOCUS_Q), "uniform", "random")
"""Every static configuration, best link first (as designed, before any run)."""
INVERT_SEED = 77
RANDOM_SEED = 2026


def load_scene(path: str | Path) -> dict[str, Any]:
    """The scene file as a dict."""
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _node(scene: Mapping[str, Any], node_id: str) -> np.ndarray:
    nd = next(n for n in scene["nodes"] if n["id"] == node_id)
    return np.asarray(nd["position_m"], float)


def panel_grid(scene: Mapping[str, Any], panel: str = "ris0") -> PanelGrid:
    """sremu's cell layout of ``panel``."""
    p = next(r for r in scene["ris"] if r["id"] == panel)
    pose = Pose(
        np.asarray(p["pose"]["center_m"], float),
        np.asarray(p["pose"]["normal"], float),
        np.asarray(p["pose"]["u_axis"], float),
    )
    g = p["grid"]
    return PanelGrid(
        pose, int(g["n_u"]), int(g["n_v"]), float(g["pitch_u_m"]), float(g["pitch_v_m"])
    )


class Configs:
    """Configurations and array-factor predictions for one scene and link."""

    def __init__(
        self,
        scene: Mapping[str, Any],
        tx_node: str = "gr",
        rx_node: str = "grrx",
        panel: str = "ris0",
    ) -> None:
        tx = next(n for n in scene["nodes"] if n["id"] == tx_node)
        fc = float(tx["streams"][0]["fc_hz"])
        k = 2.0 * np.pi * fc / C0
        cells = panel_grid(scene, panel).cell_centres()
        d1 = np.linalg.norm(cells - _node(scene, tx_node), axis=1)
        d2 = np.linalg.norm(cells - _node(scene, rx_node), axis=1)
        self.n_cells = cells.shape[0]
        self.phasor = np.exp(-1j * k * (d1 + d2)) / (d1 * d2)
        best: tuple[float, np.ndarray] | None = None
        for ref in np.linspace(0.0, np.pi, 181):
            st = (np.real(self.phasor * np.exp(-1j * ref)) < 0).astype(np.int16)
            v = self._af(st)
            if best is None or v > best[0]:
                best = (v, st)
        assert best is not None
        self.focus = best[1]

    def _af(self, states: np.ndarray) -> float:
        return float(abs(np.sum(self.phasor * np.where(states == 1, -1.0, 1.0))))

    def states(self, name: str) -> np.ndarray:
        """State vector (int16, cell order) of configuration ``name``."""
        if name == "focus":
            return self.focus.copy()
        if name == "uniform":
            return np.zeros(self.n_cells, np.int16)
        if name == "random":
            rnd = np.random.default_rng(RANDOM_SEED).integers(0, 2, self.n_cells)
            return np.asarray(rnd, dtype=np.int16)
        if name.startswith("focusq:"):
            q = float(name.split(":", 1)[1])
            st = self.focus.copy()
            idx = np.random.default_rng(INVERT_SEED).permutation(self.n_cells)
            idx = idx[: round(q * self.n_cells)]
            st[idx] = 1 - st[idx]
            return st
        raise ValueError(f"unknown RIS configuration {name!r}")

    def array_factor_db(self, name: str) -> float:
        """Isotropic point-cell array factor of ``name`` relative to ``focus``, dB."""
        return 20.0 * float(np.log10(self._af(self.states(name)) / self._af(self.focus)))
