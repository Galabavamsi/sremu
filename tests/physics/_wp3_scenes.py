"""Scene builders and evaluators shared by the WP3 physics tests (no expected values here).

Expected values in the WP3 tests come from independent closed forms
(``tests/oracles/rf.py``, ``tests/oracles/ris.py``) or hand geometry; this module only builds
``ScenePropagation`` instances and reads coefficients back through the public contract
(snapshot -> combine -> sum amp * exp(-j 2 pi f_c tau)).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from sremu.contracts import PanelConfig, StreamSpec
from sremu.physics.antennas import Antenna, Isotropic
from sremu.physics.geometry import Static
from sremu.physics.los import Occluder
from sremu.physics.propagation import NodeModel, PanelModel, ScenePropagation
from sremu.physics.response import Provenance


def streams(node: str, direction: str, fc_hz: float) -> tuple[StreamSpec, ...]:
    return (StreamSpec(f"{node}.{direction}0", direction, fc_hz, "1e6"),)  # type: ignore[arg-type]


def link(tx: str = "a", rx: str = "b") -> str:
    return f"{tx}.tx0->{rx}.rx0"


def two_node(
    p_tx: Sequence[float],
    p_rx: Sequence[float],
    fc_hz: float,
    *,
    mechanisms: Sequence[Any] = (),
    panels: Sequence[PanelModel] = (),
    occluders: Sequence[Occluder] = (),
    ant_tx: Antenna | None = None,
    ant_rx: Antenna | None = None,
    tx: str = "a",
    rx: str = "b",
) -> ScenePropagation:
    """A static two-node scene: ``tx`` transmits, ``rx`` receives."""
    return ScenePropagation(
        [
            NodeModel(
                tx, Static(np.asarray(p_tx, float)), ant_tx or Isotropic(), streams(tx, "tx", fc_hz)
            ),
            NodeModel(
                rx, Static(np.asarray(p_rx, float)), ant_rx or Isotropic(), streams(rx, "rx", fc_hz)
            ),
        ],
        list(panels),
        occluders=occluders,
        mechanisms=mechanisms,
        horizon_s=1,
    )


def by_kind(
    model: ScenePropagation,
    lk: str,
    cfg: Mapping[str, PanelConfig] | None = None,
    t: Fraction = Fraction(0),
) -> dict[int, complex]:
    """Complex coefficient at the carrier per path kind (and the total under key -1)."""
    snap = model.snapshot(t)
    lp = model.combine(snap, cfg or {pid: model.initial_config(pid) for pid in model.panel_ids()})[
        lk
    ]
    fc = model.link_fc_hz(lk)
    terms = lp.amp * np.exp(-2j * math.pi * np.mod(fc * lp.delay_s, 1.0))
    out = {int(k): complex(np.sum(terms[lp.kind == k])) for k in np.unique(lp.kind)}
    out[-1] = complex(np.sum(terms))
    return out


def row_values(
    model: ScenePropagation,
    pid: str,
    lk: str,
    cfg: Mapping[str, PanelConfig],
    t: Fraction = Fraction(0),
) -> dict[int, complex]:
    """Per hop variant of ``lk`` on panel ``pid``: the configured RIS row at the carrier,
    ``sum_k g_k exp(-j 2 pi f_c tau_k)`` over the row's tiles."""
    snap = model.snapshot(t)
    amps = model.tile_amplitudes(snap, cfg)[pid]
    assert amps is not None
    rc = snap.ris[pid]
    fc = model.link_fc_hz(lk)
    out: dict[int, complex] = {}
    for r, (rlk, var) in enumerate(rc.rows):
        if rlk == lk:
            ph = np.exp(-2j * math.pi * np.mod(fc * rc.tile_delay_s[r], 1.0))
            out[int(var)] = complex(np.sum(amps[r] * ph))
    return out


def analytic_provenance(description: str = "WP3 test table") -> Provenance:
    return Provenance.from_dict(
        {
            "source_kind": "analytic",
            "description": description,
            "reference": "tests/physics (WP3)",
            "reference_plane_offset_m": 0.0,
            "polarization": "co",
            "includes_cell_pattern": False,
            "sigma_mag_db": 0.0,
            "sigma_phase_deg": 0.0,
            "raw_sha256": None,
            "license": "test",
            "created": {"date": "2026-10-10"},
        }
    )
