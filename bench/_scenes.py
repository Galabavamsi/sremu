"""Benchmark scenes (SIM geometry; DESIGN.md sections 9.2, 12.2).

:func:`bench_scenario` builds a scenario mapping (section 9.2 field names) for one point of
the B2/B3 grids:

* ``links`` L = n_tx * n_rx (all TX-RX pairs at one carrier and rate are links). For
  L = 2**k: ``n_tx = 2**(k // 2)``, ``n_rx = 2**(k - k // 2)`` (1 -> 1x1, 2 -> 1x2,
  4 -> 2x2, 8 -> 2x4, ..., 128 -> 8x16). Noise and RX front ends scale with n_rx, the
  render work with L.
* ``n_cells`` N per panel: a fixed grid shape per N (:data:`GRID_SHAPES`) at 42.8 mm pitch
  (just below lambda/2 at 3.5 GHz, so ``cell_model: auto`` picks ``point``: Q = N).
* ``n_ris`` panels side by side on the plane x = 0 (normal +x), 1 m apart; every node is in
  front of every panel, so every RIS path is visible.
* ``rate_sps``: exact string; each TX declares ``occupied_bw_hz = 0.8 f_s`` (the default,
  declared explicitly) and sends an in-process tone at 0.1 f_s.
* Feature sets (cumulative): F0 channel only (noise off) -> F1 + thermal noise (NF 7 dB) ->
  F2 + RX CFO 500 Hz and Wiener phase noise 100 Hz -> F3 + mobility (RX nodes at 1.5 m/s,
  automatic T_u and tiles) -> F4 + recording (IQ, events, metrics) -> F5 + native ZMQ
  transport (TX and RX through client processes). F4/F5 are applied by the runner (B2);
  the scenario only differs up to F3.

Nothing here is a physics result: the geometry exists only to load the engine with a
known, non-degenerate set of LOS and RIS paths. :func:`ris_path_check` asserts that every
link of a RIS scene has a non-zero RIS term (a benchmark whose RIS term vanishes, e.g. with
every receiver behind the panel, would time a channel without its RIS path).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any, Final

import numpy as np

FC_HZ: Final[float] = 3.5e9
PITCH_M: Final[float] = 0.0428
GRID_SHAPES: Final[dict[int, tuple[int, int]]] = {
    40: (8, 5),
    240: (16, 15),
    900: (30, 30),
    4096: (64, 64),
    16384: (128, 128),
}
"""Panel grid (n_u, n_v) for each benchmark cell count N."""

FEATURES: Final[tuple[str, ...]] = ("F0", "F1", "F2", "F3", "F4", "F5")
FEATURE_TEXT: Final[dict[str, str]] = {
    "F0": "channel only (LOS + RIS tiles), noise off",
    "F1": "F0 + thermal noise (NF 7 dB)",
    "F2": "F1 + RX CFO 500 Hz + Wiener phase noise (linewidth 100 Hz)",
    "F3": "F2 + mobility (RX nodes 1.5 m/s; automatic T_u and tiles)",
    "F4": "F3 + recording (SigMF IQ of every RX, events.jsonl, metrics.jsonl)",
    "F5": "F4 + native ZMQ transport (TX and RX through client processes)",
}
RATES_SPS: Final[tuple[str, ...]] = ("1e5", "1e6", "5.76e6", "23.04e6")
LINKS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64, 128)
CELLS: Final[tuple[int, ...]] = (40, 240, 900, 4096, 16384)
RIS_COUNTS: Final[tuple[int, ...]] = (0, 1, 2, 4)


def split_links(links: int) -> tuple[int, int]:
    """``(n_tx, n_rx)`` with ``n_tx * n_rx == links`` (power-of-two split, see module doc)."""
    if links < 1 or links & (links - 1):
        raise ValueError(f"links must be a power of two >= 1, got {links}")
    k = links.bit_length() - 1
    return 2 ** (k // 2), 2 ** (k - k // 2)


def feature_index(features: str) -> int:
    """0 for F0 ... 5 for F5."""
    if features not in FEATURES:
        raise ValueError(f"unknown feature set {features!r} (F0..F5)")
    return FEATURES.index(features)


def _panel(pid: str, y_m: float, n_cells: int) -> dict[str, Any]:
    n_u, n_v = GRID_SHAPES[n_cells]
    return {
        "id": pid,
        "pose": {"center_m": [0.0, y_m, 0.0], "normal": [1.0, 0.0, 0.0], "u_axis": [0.0, 1.0, 0.0]},
        "grid": {"n_u": n_u, "n_v": n_v, "pitch_u_m": PITCH_M, "pitch_v_m": PITCH_M},
        "states": {"library": "ideal_antipodal"},
        "initial": {"fill": 0},
    }


def panel_side_m(n_cells: int) -> float:
    """Larger side of the benchmark panel, m."""
    n_u, n_v = GRID_SHAPES[n_cells]
    return max(n_u, n_v) * PITCH_M


def bench_scenario(
    *,
    links: int,
    n_cells: int,
    n_ris: int,
    rate_sps: str,
    features: str = "F1",
    seed: int = 0,
    duration_s: str = "1",
    page_target_s: str = "5e-4",
    mode: str = "lockstep",
    coupling: str = "barrier",
    name: str | None = None,
) -> dict[str, Any]:
    """Scenario mapping of one benchmark point (see the module docstring)."""
    fi = feature_index(features)
    n_tx, n_rx = split_links(links)
    if n_cells not in GRID_SHAPES:
        raise ValueError(f"n_cells must be one of {sorted(GRID_SHAPES)}, got {n_cells}")
    if n_ris < 0:
        raise ValueError("n_ris must be >= 0")
    rate = Fraction(rate_sps)
    bw = float(0.8 * rate)
    tone = float(rate) / 10.0
    side = panel_side_m(n_cells)
    spacing = side + 1.0
    panels = [_panel(f"ris{j}", (j - (n_ris - 1) / 2.0) * spacing, n_cells) for j in range(n_ris)]
    nodes: list[dict[str, Any]] = []
    traffic: list[dict[str, Any]] = []
    for i in range(n_tx):
        y = -4.0 + 8.0 * (i + 0.5) / n_tx
        nodes.append(
            {
                "id": f"tx{i:03d}",
                "position_m": [8.0, round(y, 6), 1.0],
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": "tx0",
                        "direction": "tx",
                        "fc_hz": FC_HZ,
                        "rate_sps": rate_sps,
                        "occupied_bw_hz": bw,
                    }
                ],
            }
        )
        traffic.append(
            {"stream": f"tx{i:03d}.tx0", "generator": "tone", "params": {"freq_hz": tone}}
        )
    for i in range(n_rx):
        y = -3.0 + 6.0 * (i + 0.5) / n_rx
        pos = [5.0, round(y, 6), -0.5]
        lo = {"cfo_hz": 500.0, "linewidth_hz": 100.0} if fi >= 2 else {}
        rx_stream: dict[str, Any] = {
            "name": "rx0",
            "direction": "rx",
            "fc_hz": FC_HZ,
            "rate_sps": rate_sps,
            "noise_figure_db": 7.0,
        }
        if lo:
            rx_stream["lo"] = lo
        node: dict[str, Any] = {
            "id": f"rx{i:03d}",
            "antenna": {"pattern": "isotropic"},
            "streams": [rx_stream],
        }
        if fi >= 3:
            node["trajectory"] = {
                "model": "constant_velocity",
                "start_m": pos,
                "velocity_mps": [0.0, 1.5, 0.0],
            }
        else:
            node["position_m"] = pos
        nodes.append(node)
    run: dict[str, Any] = {"mode": mode, "coupling": coupling, "duration_s": duration_s}
    return {
        "schema_version": "1.0",
        "name": name or f"b_L{links}_N{n_cells}_R{n_ris}_{features}",
        "seed": int(seed),
        "description": "sremu benchmark scene (SIM geometry, not a physical study)",
        "numerics": {"backend": "numba_cpu", "page_target_s": page_target_s},
        "nodes": nodes,
        "ris": panels,
        "run": run,
        "record": {"iq": fi >= 4, "events": True, "metrics": True},
        "traffic": traffic,
    }


def ris_path_check(model: Any, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assert a non-zero RIS term on every link of a model with panels (section 12.2).

    Evaluates ``combine(snapshot(0), cfg)`` (default: every panel's initial configuration)
    and returns per-link ``ris_db`` (RIS term power, dB re 1), ``los_db`` and
    ``ris_minus_los_db``, plus the minimum over links. Raises ``AssertionError`` when a
    link of a scene with panels has an exactly zero RIS term or no RIS path at all.
    """
    from sremu.contracts import PathKind

    ids = getattr(model, "panel_ids", ())
    pids = tuple(ids() if callable(ids) else ids)
    if cfg is None:
        cfg = {p: model.initial_config(p) for p in pids}
    snap = model.snapshot(Fraction(0))
    lp = model.combine(snap, cfg)
    per: dict[str, dict[str, float]] = {}
    worst = math.inf
    for lk, p in lp.items():
        ris = p.kind == PathKind.RIS_TILE
        los = p.kind == PathKind.LOS
        h_ris = complex(np.sum(p.amp[ris])) if np.any(ris) else 0j
        h_los = complex(np.sum(p.amp[los])) if np.any(los) else 0j
        if pids:
            if not np.any(ris):
                raise AssertionError(f"link {lk}: no RIS path although the scene has panels")
            if abs(h_ris) == 0.0:
                raise AssertionError(f"link {lk}: RIS term is exactly zero (|h_ris| = 0)")
        r_db = 20 * math.log10(abs(h_ris)) if abs(h_ris) > 0 else -math.inf
        l_db = 20 * math.log10(abs(h_los)) if abs(h_los) > 0 else -math.inf
        per[lk] = {
            "ris_db": r_db,
            "los_db": l_db,
            "ris_minus_los_db": r_db - l_db,
            "n_ris_paths": int(np.count_nonzero(ris)),
        }
        worst = min(worst, r_db - l_db)
    return {
        "links": per,
        "min_ris_minus_los_db": worst if pids else None,
        "n_panels": len(pids),
    }
