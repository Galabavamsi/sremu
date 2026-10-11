"""E-B: intra-frame RIS reconfiguration (DESIGN.md section 11, E-B; VA-5).

Geometries (point parameter ``scenario``, CLI ``--scenario``):

* default ``scenarios/lab_like_A.yaml`` (SIM geometry labelled "lab-like", not the lab's):
  3 tiles of 8 x 5 cells at 42.8 mm with 2 mm gaps, 3.5 GHz; Tx 2.0 m at -30 deg, RX1 1.2 m
  at 15 deg, RX2 1.2 m at 35 deg. Points without a ``scenario`` parameter use it, so the
  original sweep specs (``paper/sweeps/eb*.yaml``) reproduce unchanged;
* ``scenarios/studies/lab_arc.yaml``: modelled on the lab arrangement (SIM, nominal
  geometry): six 5 x 8 tiles (one panel each) on an arc of radius 1.95 m around the Tx,
  RX1 and RX2 at r = 1.2 m, theta = 15 and 35 deg.

Any scenario with nodes ``ap`` (stream ``tx0``), ``rx1`` and ``rx2`` (stream ``rx0``) at
static positions and element-controlled RIS panels works. Several panels act as one RIS:
the exact optimum is joint over every panel's cells (``|h0 + sum_panels sum_n Z|`` is the
single-panel problem on the concatenated cells, scenario panel order), and every
reconfiguration is one command per panel at the same time (one controller message, one
latency draw). Alphabets: ``lab`` (sigma1/sigma2 with |Gamma| = 1, ASSUMED) and ``ideal1``
(ideal 1-bit {1, -1}). LOS present; sensitivity variant ``blocked`` adds an opaque occluder
across both AP-RX LOS segments: the fixed plate :data:`BLOCKER` for the default geometry (it
misses every RIS hop), the plate of :func:`los_blocker` otherwise. sremu applies occluders
to the LOS path only (section 5.2), so RIS hops are never attenuated by it.

Waveform: 1 MS/s, QPSK at 4 samples/symbol, rectangular pulses, integrate-and-dump; 64
known preamble + 512 payload symbols (2304 samples, T_p = 2.304 ms); one-tap LS equalizer
from the preamble, no tracking. Physical noise (NF 7 dB); the TX power is set so that
Es/N0 = ``es_n0_db`` (default 25 dB) at RX1 under its own optimal configuration (computed
from the snapshot and recorded). Configurations: exact optimum (``sremu.physics.optimize``)
for RX1 and for RX2.

* **E-B1** (:func:`run_eb1_point`): one packet to RX1; switch to the RX2 configuration at
  ``delta * T_p`` after the packet start, with a step or a linear 20 us / 200 us transition;
  per-symbol EVM; VA-5: the first sample of RX1 that differs from the no-switch run equals
  ``to_ticks(t_sw + tau_RIS->RX1)`` and the engine's ``k_applied`` (with several panels:
  every panel's ``k_applied`` equals its own ``to_ticks(t_sw + tau_panel->RX1)`` and the
  first differing sample is the earliest of them over the panels whose states change);
  samples (and so whole symbols) before it are bitwise identical.
* **E-B2** (:func:`run_eb2_point`): 2000 packets alternating RX1/RX2. Commands come from
  the ``TdmaAlternate`` controller (one per guard; its decisions; one instance per panel).
  Their latency ``L = 0.2 ms + U[0, J]`` is the controller's keyed ``LatencyModel`` draw
  (``sremu.engine.barriers.draw_latency_s``: purpose ``ctrl.latency/<name>``, counter =
  barrier index), one draw per guard. Arrival = issue + L. Semantics:

  - ``timed``: requested switch at the guard centre, issued ``lead`` earlier; a late
    arrival applies on arrival (``apply_on_arrival``): t_sw = max(t_req, arrival);
  - ``asap``: issued at the guard start, applied on arrival;
  - ``tick-based`` (tick-based untimed semantics): issued at the guard start, applied at
    the next 2 ms tick after the arrival with one pending slot per panel (a second command
    in the same tick overwrites the first); implemented by
    ``sremu.legacy.next_tick.LegacyNextTickAdapter``. ``legacy`` is accepted as an alias.

  In lockstep with barrier coupling the engine evaluates a barrier controller's actions at
  ``t_b + L_min`` and ignores the latency of an action that carries ``at_s`` (it can never
  be late), so the arrival semantics are applied here and the resulting commands are
  scheduled in arrival order (later arrival = later ``seq``, which the engine's same-time
  rule needs for the tick-based overwrite). See the contract change request in the WP11
  report.

CLI (unattended E-B1 + E-B2 sweeps; one run directory with ``manifest.json`` per point,
``sweep.json``/``index.json`` per sweep, ``git_state.json``, the pre-declared plan
``eb_plan.json`` and the aggregation under ``<out>/results``)::

    python -m experiments.eb [--scenario PATH] [--out DIR] [--jobs N] [--resume] [--smoke]

The full design per scenario: E-B1 (72 points) and E-B2 at Es/N0 = 25 dB (``eb2_lab``: LOS
present and blocked, ``eb2_ideal1``: LOS present), plus the pre-declared Es/N0 = 15 dB
sensitivity of the LOS-present slice (``eb2_lab_es15``, ``eb2_ideal1_es15``); 2000 packets
per E-B2 point. ``--smoke`` runs one E-B1 point and one 50-packet E-B2 point instead.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import itertools
import math
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

if __package__ in (None, ""):  # ``python experiments/eb.py``: make the repo importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.common import (
    REPO,
    RunRecord,
    db10,
    export_git_info,
    quiet_warnings,
    read_json,
    write_csv,
    write_json,
    write_meta,
)
from sremu.analysis.stats import clopper_pearson
from sremu.contracts import (
    CommandRequest,
    CommandTime,
    LatencyModel,
    TransitionSpec,
)
from sremu.control.builtin import TdmaAlternate
from sremu.control.controller import csi_from_snapshot, make_observation, setup_controller
from sremu.engine.barriers import draw_latency_s
from sremu.engine.emulator import Emulator, EngineSink
from sremu.engine.streams import QpskPacketSource
from sremu.legacy.next_tick import LegacyNextTickAdapter
from sremu.physics.optimize import exact_alphabet_optimum
from sremu.physics.propagation import from_scenario
from sremu.scenario.load import parse_scenario, read_document, resolve_scenario
from sremu.timebase import to_ticks
from sremu.units import noise_power_w

LAB = REPO / "scenarios" / "lab_like_A.yaml"
ARC = REPO / "scenarios" / "studies" / "lab_arc.yaml"
GEOMETRY_LABELS = {
    "scenarios/lab_like_A.yaml": "lab-like SIM geometry (not the lab's)",
    "scenarios/studies/lab_arc.yaml": "modelled on the lab arrangement (SIM, nominal geometry)",
}
FS = Fraction(10**6)
SPS = 4
N_PRE = 64
N_PAY = 512
BODY = (N_PRE + N_PAY) * SPS  # 2304 samples
T_P = Fraction(BODY) / FS  # 2.304 ms
ES_N0_DB = 25.0
ES_N0_SENSITIVITY_DB = 15.0
NF_DB = 7.0
TX = "ap.tx0"
RX1 = "rx1.rx0"
RX2 = "rx2.rx0"
LINKS = {RX1: f"{TX}->{RX1}", RX2: f"{TX}->{RX2}"}
PANEL = "ris0"
TICK_S = Fraction(1, 500)  # 2 ms (tick-based untimed semantics)
LEGACY_TAU_S = TICK_S  # former name, kept for callers
LAT_FIXED_S = Fraction(1, 5000)  # 0.2 ms
CTRL_NAME = "tdma"
ALPHABETS: dict[str, dict[str, Any]] = {
    "lab": {"library": "lab_binary_3p5ghz", "assume_mag": {"sigma1": 1.0, "sigma2": 1.0}},
    "ideal1": {"library": "ideal_antipodal"},
}
TRANSITIONS = {
    "step": None,
    "linear20us": TransitionSpec("linear", Fraction(20, 10**6)),
    "linear200us": TransitionSpec("linear", Fraction(200, 10**6)),
}
BLOCKER = [[1.5, -0.6, -0.5], [1.5, -0.35, -0.5], [1.5, -0.35, 0.5], [1.5, -0.6, 0.5]]
SEMANTICS = ("timed", "asap", "tick-based")
SEMANTICS_ALIASES = {"legacy": "tick-based"}
# :func:`los_blocker` plate: across both LOS segments at BLOCK_AT x the shorter
# along-bisector LOS length from the Tx, BLOCK_MARGIN_M beyond both crossings, 1 m tall
BLOCK_AT = 0.5
BLOCK_MARGIN_M = 0.05
BLOCK_HALF_HEIGHT_M = 0.5
# E-B2 incremental command release (run_iq): run in 50 ms steps, schedule >= 50 ms ahead
CHUNK_S = Fraction(1, 20)
LOOKAHEAD_S = Fraction(1, 20)
_THREADS_ENV = "SREMU_EB_NUMBA_THREADS"


# =========================================================================== scenario


def scenario_path(scenario: str | Path | None = None) -> Path:
    """Absolute path of a scenario reference (None or "" = :data:`LAB`; relative paths are
    relative to the repository root)."""
    if scenario is None or str(scenario) == "":
        return LAB
    p = Path(scenario)
    if not p.is_absolute():
        p = REPO / p
    return p.resolve()


def scenario_ref(scenario: str | Path | None) -> str | None:
    """Point-parameter form of a scenario: None for the default geometry, else the POSIX
    path relative to the repository root (absolute when outside it)."""
    p = scenario_path(scenario)
    if p == LAB.resolve():
        return None
    try:
        return p.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def geometry_label(scenario: str | Path | None) -> str:
    """Paper-facing geometry label of a scenario (its description when not catalogued)."""
    p = scenario_path(scenario)
    try:
        rel = p.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        rel = p.as_posix()
    lab = GEOMETRY_LABELS.get(rel)
    if lab is not None:
        return lab
    return str(read_document(p).get("description") or p.stem)


def _panel_sizes(d: Mapping[str, Any]) -> list[int]:
    """Controls per panel (element granularity: n_u * n_v), scenario panel order."""
    out = []
    for p in d["ris"]:
        gran = (p.get("control") or {}).get("granularity", "element")
        if gran != "element":
            raise ValueError(f"E-B needs element control (panel {p['id']}: {gran})")
        out.append(int(p["grid"]["n_u"]) * int(p["grid"]["n_v"]))
    return out


def split_states(
    states: Sequence[int] | np.ndarray, panels: Sequence[str], sizes: Sequence[int]
) -> dict[str, np.ndarray]:
    """Split a concatenated (sum sizes,) state vector into one int16 vector per panel."""
    s = np.asarray(states).astype(np.int16).ravel()
    if s.size != int(sum(sizes)) or len(panels) != len(sizes):
        raise ValueError(f"{s.size} states for panels {list(panels)} of sizes {list(sizes)}")
    out, o = {}, 0
    for pid, n in zip(panels, sizes, strict=True):
        out[str(pid)] = s[o : o + int(n)].copy()
        o += int(n)
    return out


def _node_position(d: Mapping[str, Any], stream: str) -> np.ndarray:
    nid = stream.split(".", 1)[0]
    for nd in d["nodes"]:
        if nd["id"] == nid:
            if "position_m" not in nd:
                raise ValueError(f"E-B needs a static position for node {nid!r}")
            return np.asarray(nd["position_m"], dtype=np.float64)
    raise KeyError(nid)


def los_blocker(d: Mapping[str, Any]) -> list[list[float]]:
    """Opaque plate across both AP-RX LOS segments of scenario mapping ``d``.

    Vertical rectangle perpendicular to the horizontal bisector ``b`` of the Tx->RX1 and
    Tx->RX2 directions, at ``BLOCK_AT`` x the shorter along-``b`` LOS length from the Tx; it
    extends ``BLOCK_MARGIN_M`` beyond both LOS crossings sideways and ``BLOCK_HALF_HEIGHT_M``
    above and below the Tx height. Vertices rounded to 1e-9 m."""
    pt = _node_position(d, TX)
    ps = [_node_position(d, RX1), _node_position(d, RX2)]

    def hdir(p: np.ndarray) -> tuple[np.ndarray, float]:
        v = p - pt
        v[2] = 0.0
        n = float(np.linalg.norm(v))
        return v / n, n

    dirs = [hdir(p) for p in ps]
    b = dirs[0][0] + dirs[1][0]
    b /= float(np.linalg.norm(b))
    e = np.array([-b[1], b[0], 0.0])
    d_b = BLOCK_AT * min(n * float(u @ b) for u, n in dirs)
    s = [d_b * float(u @ e) / float(u @ b) for u, _n in dirs]
    lo, hi = min(s) - BLOCK_MARGIN_M, max(s) + BLOCK_MARGIN_M
    c = pt + d_b * b
    z0, z1 = pt[2] - BLOCK_HALF_HEIGHT_M, pt[2] + BLOCK_HALF_HEIGHT_M
    verts = []
    for lat, z in ((lo, z0), (hi, z0), (hi, z1), (lo, z1)):
        q = c + lat * e
        verts.append([round(float(q[0]), 9), round(float(q[1]), 9), round(float(z), 9)])
    return verts


def blocker_crossings(d: Mapping[str, Any], vertices: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Which segments the occluder polygon crosses: both LOS segments, every Tx -> cell
    centre and every cell centre -> RX segment (cell centres per D17). sremu attenuates only
    the LOS path; the RIS-hop counts document how idealized a blocked variant is."""
    from sremu.physics.geometry import PanelGrid, Pose, segments_hit_polygon

    poly = np.asarray(vertices, dtype=np.float64)
    pt = _node_position(d, TX)
    prx = {rx: _node_position(d, rx) for rx in (RX1, RX2)}
    cells = []
    for p in d["ris"]:
        pose = Pose(
            np.asarray(p["pose"]["center_m"], dtype=np.float64),
            np.asarray(p["pose"]["normal"], dtype=np.float64),
            np.asarray(p["pose"]["u_axis"], dtype=np.float64),
        )
        g, t = p["grid"], p.get("tiles") or {"n_tiles_u": 1, "n_tiles_v": 1, "gap_m": 0.0}
        grid = PanelGrid(
            pose,
            int(g["n_u"]),
            int(g["n_v"]),
            float(g["pitch_u_m"]),
            float(g["pitch_v_m"]),
            int(t["n_tiles_u"]),
            int(t["n_tiles_v"]),
            float(t["gap_m"]),
        )
        cells.append(np.asarray(grid.cell_centres(), dtype=np.float64))
    cc = np.concatenate(cells)
    tx_b = np.broadcast_to(pt, cc.shape)
    out: dict[str, Any] = {
        "vertices_m": [list(map(float, v)) for v in poly],
        "los_crossed": {
            rx: bool(segments_hit_polygon(pt[None, :], p[None, :], poly)[0])
            for rx, p in prx.items()
        },
        "n_cells": int(cc.shape[0]),
        "tx_cell_segments_crossed": int(np.count_nonzero(segments_hit_polygon(tx_b, cc, poly))),
        "cell_rx_segments_crossed": {
            rx: int(np.count_nonzero(segments_hit_polygon(cc, np.broadcast_to(p, cc.shape), poly)))
            for rx, p in prx.items()
        },
    }
    return out


def blocked_assumption(diag: Mapping[str, Any]) -> str | None:
    """Paper-facing assumption of a LOS-blocked point of a non-default geometry (``diag``:
    :attr:`Design.diag`, whose ``blocker`` entry is :func:`blocker_crossings`); None when the
    design has no such entry (LOS present, or the default geometry, whose plate misses every
    RIS hop)."""
    b = diag.get("blocker")
    if not b:
        return None
    return (
        "LOS blocked by one opaque plate across both Tx-RX LOS segments; occluders act on the "
        "LOS only (section 5.2), so the RIS hops are unattenuated although the plate crosses "
        f"{b['tx_cell_segments_crossed']}/{b['n_cells']} Tx->cell segments: idealized regime "
        "'LOS removed, RIS hops unattenuated' [ENG]"
    )


def lab_scenario(
    alphabet: str,
    los: str = "present",
    *,
    full_scale_dbm: float = 0.0,
    initial_states: Sequence[int] | np.ndarray | None = None,
    duration_s: Fraction | None = None,
    seed: int | None = None,
    scenario: str | Path | None = None,
) -> dict[str, Any]:
    """The E-B scenario (``lab_like_A.yaml`` by default, else ``scenario``) adapted for
    E-B: alphabet, LOS variant, TX power, initial configuration (concatenated over the
    panels in scenario order); no scenario controllers and no scenario traffic (the drivers
    supply them)."""
    path = scenario_path(scenario)
    default = path == LAB
    d = copy.deepcopy(read_document(path))
    stem = re.sub(r"[^A-Za-z0-9_.\-]", "_", path.stem)
    d["name"] = f"eb_{alphabet}_{los}" if default else f"eb_{stem}_{alphabet}_{los}"[:64]
    if seed is not None:
        d["seed"] = int(seed)
    pids = [str(p["id"]) for p in d["ris"]]
    init = None
    if initial_states is not None:
        init = split_states(initial_states, pids, _panel_sizes(d))
    for p in d["ris"]:
        p["states"] = dict(ALPHABETS[alphabet])
        if init is not None:
            p["initial"] = {"states": [int(s) for s in init[str(p["id"])]]}
    for nd in d["nodes"]:
        for s in nd["streams"]:
            if s["direction"] == "tx":
                s["full_scale_dbm"] = float(full_scale_dbm)
            else:
                s["noise_figure_db"] = NF_DB
    if los == "blocked":
        if default:
            d["environment"] = {
                "occluders": [{"id": "los_blocker", "kind": "opaque", "vertices_m": BLOCKER}]
            }
        else:
            env = d.setdefault("environment", {})
            env["occluders"] = [
                *list(env.get("occluders") or []),
                {"id": "los_blocker", "kind": "opaque", "vertices_m": los_blocker(d)},
            ]
    elif los != "present":
        raise ValueError(los)
    d["controllers"] = []
    d["traffic"] = []
    d["record"] = {"iq": False, "events": True, "metrics": False}
    if duration_s is not None:
        d["run"]["duration_s"] = str(duration_s)
    if default:
        d["description"] = (
            f"E-B lab-like SIM geometry (not the lab's), alphabet {alphabet}, LOS {los}"
        )
    else:
        d["description"] = f"E-B {geometry_label(path)}, alphabet {alphabet}, LOS {los}"
    return d


def resolve(d: Mapping[str, Any], base_dir: str | Path | None = None) -> Any:
    """Validate, resolve and guard a scenario mapping (relative files resolve against
    ``base_dir``, default the directory of :data:`LAB`)."""
    from sremu.scenario.guards import check_scenario

    quiet_warnings()
    sc = parse_scenario(dict(d), label=str(d.get("name")))
    res = resolve_scenario(sc, Path(base_dir) if base_dir is not None else LAB.parent, warn=False)
    check_scenario(res).raise_for_errors()
    return res


@dataclass
class Design:
    """Exact-optimum configurations, channels and the TX power of one (alphabet, LOS).

    ``s1``/``s2`` are concatenated over ``panels`` (scenario order, ``sizes`` controls
    each); :meth:`split` gives the per-panel vectors."""

    alphabet: str
    los: str
    s1: np.ndarray
    s2: np.ndarray
    h: dict[str, dict[str, complex]]  # rx -> {"s1": h, "s2": h, "direct": h}
    tx_dbm: float
    noise_w: float
    es_n0_db: dict[str, float]  # rx -> Es/N0 under its own optimal configuration
    diag: dict[str, Any]
    panels: tuple[str, ...] = (PANEL,)
    sizes: tuple[int, ...] = ()
    es_n0_target_db: float = ES_N0_DB
    scenario: str | None = None
    label: str = field(default=GEOMETRY_LABELS["scenarios/lab_like_A.yaml"])

    def split(self, s: Sequence[int] | np.ndarray) -> dict[str, np.ndarray]:
        """Per-panel state vectors of a concatenated configuration."""
        sizes = self.sizes or (int(np.asarray(s).size),)
        return split_states(s, self.panels, sizes)

    def changed_panels(self) -> list[str]:
        """Panels whose states differ between s1 and s2 (all panels when none do)."""
        a, b = self.split(self.s1), self.split(self.s2)
        ch = [p for p in self.panels if np.any(a[p] != b[p])]
        return ch or list(self.panels)


def design(
    alphabet: str,
    los: str,
    es_n0_db: float = ES_N0_DB,
    *,
    scenario: str | Path | None = None,
) -> Design:
    """Exact optima for RX1 and RX2 at t = 0 (static scene, joint over every panel) and the
    TX power that gives Es/N0 = ``es_n0_db`` at RX1 under s1 (``Es = P_rx * SPS / f_s``,
    ``N0 = k T0 F``)."""
    quiet_warnings()
    ref = scenario_ref(scenario)
    doc = lab_scenario(alphabet, los, scenario=scenario)
    pids = tuple(str(p["id"]) for p in doc["ris"])
    sizes = tuple(_panel_sizes(doc))
    model = from_scenario(doc, backend="numpy_ref")
    snap = model.snapshot(Fraction(0))
    views = {rx: csi_from_snapshot(model, snap, lk) for rx, lk in LINKS.items()}
    z = {rx: np.concatenate([v.cascade[p] for p in pids], axis=1) for rx, v in views.items()}
    _, s1 = exact_alphabet_optimum(views[RX1].h_direct, z[RX1])
    _, s2 = exact_alphabet_optimum(views[RX2].h_direct, z[RX2])
    c1, c2 = split_states(s1, pids, sizes), split_states(s2, pids, sizes)
    h: dict[str, dict[str, complex]] = {}
    for rx, v in views.items():
        h[rx] = {
            "s1": v.evaluate(c1),
            "s2": v.evaluate(c2),
            "direct": complex(v.h_direct),
        }
    nw = noise_power_w(float(FS), NF_DB)
    p_rx = 10 ** (es_n0_db / 10) * nw / SPS  # W
    p_tx = p_rx / abs(h[RX1]["s1"]) ** 2
    tx_dbm = 10 * math.log10(p_tx / 1e-3)
    es = {
        rx: 10 * math.log10(p_tx * abs(h[rx]["s1" if rx == RX1 else "s2"]) ** 2 * SPS / nw)
        for rx in (RX1, RX2)
    }
    v1 = views[RX1]
    cols = np.arange(s1.size)
    ris1_s1 = complex(z[RX1][s1.astype(int), cols].sum())
    ris1_s2 = complex(z[RX1][s2.astype(int), cols].sum())
    tau1 = {p: float(model.ris_delay_s(p, RX1, Fraction(0))) for p in pids}
    tau2 = {p: float(model.ris_delay_s(p, RX2, Fraction(0))) for p in pids}
    diag: dict[str, Any] = {
        "rx1_ris_term_change_db": db10(abs(ris1_s2) ** 2 / abs(ris1_s1) ** 2),
        "rx1_ris_term_rotation_deg": math.degrees(
            math.atan2((ris1_s2 / ris1_s1).imag, (ris1_s2 / ris1_s1).real)
        ),
        "rx1_ris_term_s1_db": db10(abs(ris1_s1) ** 2),
        "rx1_direct_db": db10(abs(v1.h_direct) ** 2),
        "rx1_total_change_db": db10(abs(h[RX1]["s2"]) ** 2 / abs(h[RX1]["s1"]) ** 2),
        "rx1_total_rotation_deg": math.degrees(
            math.atan2((h[RX1]["s2"] / h[RX1]["s1"]).imag, (h[RX1]["s2"] / h[RX1]["s1"]).real)
        ),
        "rx1_midpoint_db": db10(
            abs((h[RX1]["s1"] + h[RX1]["s2"]) / 2) ** 2 / abs(h[RX1]["s1"]) ** 2
        ),
        "n_differing_elements": int(np.count_nonzero(s1 != s2)),
        "tau_ris_rx1_s": min(tau1.values()),
        "tau_ris_rx2_s": min(tau2.values()),
    }
    if len(pids) > 1:
        diag["panels"] = list(pids)
        diag["n_differing_elements_by_panel"] = {
            p: int(np.count_nonzero(c1[p] != c2[p])) for p in pids
        }
        diag["tau_ris_rx1_s_by_panel"] = tau1
        diag["tau_ris_rx2_s_by_panel"] = tau2
    if ref is not None and los == "blocked":
        occ = [o for o in doc["environment"]["occluders"] if o["id"] == "los_blocker"]
        diag["blocker"] = blocker_crossings(doc, occ[0]["vertices_m"])
    return Design(
        alphabet,
        los,
        s1.astype(np.int16),
        s2.astype(np.int16),
        h,
        tx_dbm,
        nw,
        es,
        diag,
        panels=pids,
        sizes=sizes,
        es_n0_target_db=float(es_n0_db),
        scenario=ref,
        label=geometry_label(scenario),
    )


# =========================================================================== IQ helpers


class CaptureSink(EngineSink):
    """Keeps every RX sample (complex64) in preallocated arrays."""

    def __init__(self, n: int, streams: Sequence[str]) -> None:
        self.buf = {s: np.zeros(n, dtype=np.complex64) for s in streams}
        self.n = n

    def on_rx(self, stream: str, k0: int, d: np.ndarray) -> None:
        b = self.buf.get(stream)
        if b is None or k0 >= self.n:
            return
        k1 = min(self.n, k0 + d.shape[0])
        b[k0:k1] = d[: k1 - k0]


def run_iq(
    resolved: Any,
    traffic: Any,
    commands: Sequence[CommandRequest],
    n_samples: int,
    *,
    adapter_times: Sequence[Fraction] | None = None,
    release_times: Sequence[Fraction] | None = None,
    chunk_s: Fraction = CHUNK_S,
    lookahead_s: Fraction = LOOKAHEAD_S,
) -> tuple[CaptureSink, Emulator, Any]:
    """Lockstep run of ``n_samples`` RX samples with in-process traffic; ``commands`` are
    scheduled in the given order. With ``adapter_times``, each command goes through the
    tick-based adapter (``LegacyNextTickAdapter``) at its arrival time;
    ``eng.tick_overwritten`` then counts overwritten commands.

    Without ``release_times`` every command is scheduled before the run. With
    ``release_times`` (non-decreasing, one per command, each <= the command's effective
    time) the run advances in ``chunk_s`` steps and each command is scheduled before the
    step whose end + ``lookahead_s`` exceeds its release time, so it is always scheduled at
    least ``lookahead_s`` before the frontier reaches it (never late because of the
    release). Scheduling order (``seq``), arrival times and the adapter state are those of
    the upfront schedule; the engine prunes applied commands, so the per-command cost stays
    bounded instead of growing with the total command count (the upfront schedule is
    quadratic in the number of commands)."""
    quiet_warnings()
    sink = CaptureSink(n_samples, [RX1, RX2])
    eng = Emulator.from_scenario(resolved, traffic={TX: traffic}, sinks=[sink])
    ad: LegacyNextTickAdapter | None = None
    shim: _ArrivalEngine | None = None
    if adapter_times is not None:
        if len(adapter_times) != len(commands):
            raise ValueError("adapter_times needs one arrival time per command")
        shim = _ArrivalEngine(eng)
        ad = LegacyNextTickAdapter(shim, TICK_S, allow_nonphysical=True)

    def submit(i: int) -> None:
        req = commands[i]
        if ad is None:
            rc = eng.schedule(req)
        else:
            assert shim is not None
            assert adapter_times is not None
            shim.t = adapter_times[i]
            rc = ad.schedule(req)
        if not rc.ok:
            raise RuntimeError(f"command {req.cmd_id} rejected: {rc.error} {rc.msg}")

    t_end = Fraction(n_samples) / FS
    if release_times is None:
        for i in range(len(commands)):
            submit(i)
        summary = eng.run(t_end)
    else:
        if len(release_times) != len(commands):
            raise ValueError("release_times needs one time per command")
        if any(b < a for a, b in itertools.pairwise(release_times)):
            raise ValueError("release_times must be non-decreasing")
        i, t = 0, Fraction(0)
        while True:
            t_next = min(t_end, t + chunk_s)
            while i < len(commands) and release_times[i] < t_next + lookahead_s:
                submit(i)
                i += 1
            summary = eng.run(t_next)
            if t_next >= t_end:
                break
            t = t_next
        while i < len(commands):  # beyond the run (never rendered), as in the upfront case
            submit(i)
            i += 1
    if ad is not None:
        eng.tick_overwritten = ad.lost  # type: ignore[attr-defined]
    return sink, eng, summary


class _ArrivalEngine:
    """Engine view for the tick-based adapter: ``time_now`` is the command's arrival."""

    def __init__(self, eng: Emulator) -> None:
        self.eng = eng
        self.t = Fraction(0)

    def time_now(self) -> Fraction:
        return self.t

    def schedule(self, req: CommandRequest) -> Any:
        return self.eng.schedule(req)


def integrate_dump(y: np.ndarray, start: int, n_sym: int, sps: int = SPS) -> np.ndarray:
    """Integrate-and-dump outputs (sum over each symbol's ``sps`` samples), complex128."""
    seg = np.asarray(y[start : start + n_sym * sps], dtype=np.complex128)
    return seg.reshape(n_sym, sps).sum(axis=1)


def demod(y: np.ndarray, start: int, sym: np.ndarray) -> dict[str, Any]:
    """One-tap LS equalizer from the 64 preamble symbols; per-symbol EVM (|e|^2 / |s|^2) and
    payload bit errors (Gray QPSK: sign of I and Q)."""
    r = integrate_dump(y, start, N_PRE + N_PAY)
    pre = sym[:N_PRE]
    h = complex(np.vdot(pre, r[:N_PRE]) / np.vdot(pre, pre).real)
    z = r / h
    e2 = np.abs(z - sym) ** 2 / np.abs(sym) ** 2
    tx_bits = np.stack([sym.real < 0, sym.imag < 0], axis=1)[N_PRE:]
    rx_bits = np.stack([z.real < 0, z.imag < 0], axis=1)[N_PRE:]
    nerr = int(np.count_nonzero(tx_bits != rx_bits))
    return {
        "h_ls": h,
        "evm2": e2,
        "bit_errors": nerr,
        "evm_rms_payload_db": db10(float(np.mean(e2[N_PRE:]))),
        "evm_rms_payload_pct": 100 * math.sqrt(float(np.mean(e2[N_PRE:]))),
    }


def _cmd_id(base: str, panel: str, n_panels: int) -> str:
    """Command id: ``base`` for a single panel, ``base:<panel>`` with several."""
    return base if n_panels == 1 else f"{base}:{panel}"


def _joined_status(statuses: Sequence[str | None]) -> str | None:
    st = sorted({s for s in statuses if s is not None})
    if not st:
        return None
    return st[0] if len(st) == 1 else ",".join(st)


def _cap_threads() -> None:
    """Apply the CLI's per-worker numba thread count (``SREMU_EB_NUMBA_THREADS``); render
    output does not depend on the thread count (I2a)."""
    v = os.environ.get(_THREADS_ENV)
    if not v:
        return
    import numba

    cap = int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]
    n = max(1, min(int(v), cap))
    if numba.get_num_threads() != n:
        numba.set_num_threads(n)


# =========================================================================== E-B1


def run_eb1_point(ctx: Any) -> dict[str, Any]:
    """E-B1 sweep runner (``experiments.eb:run_eb1_point``). Parameters: ``alphabet``,
    ``los``, ``delta`` (fraction of T_p, string), ``transition`` (step | linear20us |
    linear200us); optional ``scenario`` (default lab-like) and ``es_n0_db`` (25)."""
    quiet_warnings()
    _cap_threads()
    prm = dict(ctx.params)
    alphabet, los = str(prm["alphabet"]), str(prm.get("los", "present"))
    delta = Fraction(str(prm["delta"]))
    trans = str(prm["transition"])
    sref = prm.get("scenario")
    es_target = float(prm.get("es_n0_db", ES_N0_DB))
    dsg = design(alphabet, los, es_target, scenario=sref)
    n_pan = len(dsg.panels)
    start = 3000  # packet start sample (3 ms)
    n = start + BODY + 1000
    scen = lab_scenario(
        alphabet,
        los,
        full_scale_dbm=dsg.tx_dbm,
        initial_states=dsg.s1,
        duration_s=Fraction(n) / FS,
        seed=int(ctx.seed),
        scenario=sref,
    )
    res = resolve(scen, scenario_path(sref).parent)
    run = RunRecord(
        Path(ctx.run_dir),
        int(res.scenario.seed),
        resolved=res,
        mode="lockstep",
        coupling="barrier",
        patch_root=Path(ctx.run_dir).parent,
    )
    src = QpskPacketSource(TX, int(res.scenario.seed), SPS, N_PRE, N_PAY, guard=10**9, start=start)
    sym = src.symbols(0)
    t_sw = Fraction(start) / FS + delta * T_P
    c2 = dsg.split(dsg.s2)
    reqs = [
        CommandRequest(
            _cmd_id("eb1_switch", pid, n_pan),
            "ris.set_config",
            "eb1",
            pid,
            CommandTime.at(t_sw),
            states=c2[pid],
            transition=TRANSITIONS[trans],
        )
        for pid in dsg.panels
    ]
    ref_sink, _ref_eng, _ = run_iq(res, src, [], n)
    sink, eng, summ = run_iq(res, src, reqs, n)
    recs = {r.cmd_id: r for r in eng.records}
    changed = set(dsg.changed_panels())
    per: dict[str, dict[str, Any]] = {}
    for pid in dsg.panels:
        rec = recs[_cmd_id("eb1_switch", pid, n_pan)]
        tau_p = float(eng.model.ris_delay_s(pid, RX1, t_sw))
        per[pid] = {
            "k_applied": int(rec.k_applied[RX1]),
            "k_expected": to_ticks(t_sw + Fraction(repr(tau_p)), FS),
            "tau_s": tau_p,
            "changed": pid in changed,
            "status": rec.status,
        }
    eff = [p for p in dsg.panels if per[p]["changed"]]
    k_app = min(per[p]["k_applied"] for p in eff)
    k_expect = min(per[p]["k_expected"] for p in eff)
    all_equal = all(v["k_applied"] == v["k_expected"] for v in per.values())
    y, y0 = sink.buf[RX1], ref_sink.buf[RX1]
    diff = np.nonzero(y != y0)[0]
    k_first = int(diff[0]) if diff.size else None
    # symbols wholly before the first affected sample: bitwise identical
    n_before = 0 if k_first is None else max(0, (k_first - start) // SPS)
    n_before = min(n_before, N_PRE + N_PAY) if k_first is not None else N_PRE + N_PAY
    seg, seg0 = y[start : start + n_before * SPS], y0[start : start + n_before * SPS]
    before_identical = bool(np.array_equal(seg.view(np.uint64), seg0.view(np.uint64)))
    tx_active = start - 16 <= k_app < start + BODY + 16
    d_sw = demod(y, start, sym)
    d_ref = demod(y0, start, sym)
    rows = [
        {
            "symbol": i,
            "kind": "preamble" if i < N_PRE else "payload",
            "t_center_s": float((start + i * SPS + SPS / 2) / FS),
            "evm_db": db10(float(d_sw["evm2"][i])),
            "evm_ref_db": db10(float(d_ref["evm2"][i])),
        }
        for i in range(N_PRE + N_PAY)
    ]
    write_csv(Path(ctx.run_dir) / "symbols.csv", rows)
    va5: dict[str, Any] = {
        "k_applied": k_app,
        "k_expected_to_ticks": k_expect,
        "k_first_differing": k_first,
        "first_differing_minus_k_applied": None if k_first is None else k_first - k_app,
        "tx_active_at_switch": bool(tx_active),
        "symbols_before_first_affected": n_before,
        "symbols_before_bitwise_identical": before_identical,
        "pass_step_inside_packet": (
            None
            if not (trans == "step" and tx_active)
            else bool(k_first == k_app == k_expect and before_identical and all_equal)
        ),
    }
    if n_pan > 1:
        va5["n_panels"] = n_pan
        va5["n_panels_changed"] = len(eff)
        va5["all_panels_k_applied_equal_expected"] = bool(all_equal)
        va5["per_panel"] = per
    out = {
        "alphabet": alphabet,
        "los": los,
        "delta": str(delta),
        "transition": trans,
        "t_sw_s": str(t_sw),
        "scenario": dsg.scenario,
        "geometry_label": dsg.label,
        "n_panels": n_pan,
        "es_n0_target_db": es_target,
        "tx_dbm": dsg.tx_dbm,
        "es_n0_db": dsg.es_n0_db,
        "design": dsg.diag,
        "per_packet_evm_db": d_sw["evm_rms_payload_db"],
        "per_packet_evm_pct": d_sw["evm_rms_payload_pct"],
        "bit_errors": d_sw["bit_errors"],
        "ber": d_sw["bit_errors"] / (2 * N_PAY),
        "ref_evm_db": d_ref["evm_rms_payload_db"],
        "ref_bit_errors": d_ref["bit_errors"],
        "command_status": _joined_status([v["status"] for v in per.values()]),
        "va5": va5,
        "verify_interp_max_rel": summ.verify_interp_max_rel,
    }
    write_json(Path(ctx.run_dir) / "result.json", out)
    info = eng.manifest_info()
    run.page_size = info["page_size"]
    run.verify_interp_max_rel = info["verify_interp_max_rel"]
    run.counters = {k: v for k, v in info["counters"].items() if isinstance(v, int)}
    run.add_assumption("lab |Gamma| ASSUMED" if alphabet == "lab" else "ideal 1-bit alphabet")
    run.add_assumption(
        f"TX power {dsg.tx_dbm:.3f} dBm for Es/N0 = {es_target:g} dB at RX1 under s1"
    )
    if n_pan > 1:
        run.add_assumption(
            f"{n_pan} panels act as one RIS: joint exact optimum over every cell; one command "
            "per panel at the same time"
        )
    blk = blocked_assumption(dsg.diag)
    if blk is not None:
        run.add_assumption(blk)
    write_meta(
        ctx.run_dir,
        experiment="E-B1",
        data_classes=["SIM-sremu", "ASSUMED"] if alphabet == "lab" else ["SIM-sremu"],
        description=(
            f"E-B1 mechanism: one packet to RX1, switch to RX2's optimum at {delta} T_p, "
            f"transition {trans}, alphabet {alphabet}, LOS {los}; {dsg.label}"
        ),
        assumptions=[*run.assumptions, "transition path complex-linear [ENG]"],
        inputs=run.inputs,
        files={"symbols.csv": "per-symbol EVM (dB) with and without the switch"},
    )
    run.finish()
    return {k: v for k, v in out.items() if k != "design"} | {"design": dsg.diag}


# =========================================================================== E-B2


def canonical_semantics(sem: str) -> str:
    """``timed`` | ``asap`` | ``tick-based`` (``legacy`` is an alias of ``tick-based``)."""
    s = SEMANTICS_ALIASES.get(str(sem), str(sem))
    if s not in SEMANTICS:
        raise ValueError(f"unknown E-B2 semantics {sem!r}; known {SEMANTICS}")
    return s


def packet_start(semantics: str, guard: int, lead: int) -> int:
    """First packet's start sample: guard centres at ``m*slot + lead`` (timed) or guard
    starts at ``m*slot`` (asap, tick-based)."""
    if semantics == "timed":
        return lead + guard // 2
    return guard


def run_eb2_point(ctx: Any) -> dict[str, Any]:
    """E-B2 sweep runner (``experiments.eb:run_eb2_point``). Parameters: ``alphabet``,
    ``guard_s`` (string), ``jitter_s`` (J, string), ``semantics`` (timed | asap |
    tick-based; ``legacy`` = tick-based), ``lead_s`` (timed), ``n_packets`` (default 2000);
    optional ``los`` (present), ``scenario`` (default lab-like), ``es_n0_db`` (25)."""
    quiet_warnings()
    _cap_threads()
    prm = dict(ctx.params)
    alphabet = str(prm.get("alphabet", "lab"))
    los = str(prm.get("los", "present"))
    g_s = Fraction(str(prm["guard_s"]))
    j_s = Fraction(str(prm["jitter_s"]))
    sem = canonical_semantics(str(prm["semantics"]))
    tick = sem == "tick-based"
    lead_s = Fraction(str(prm.get("lead_s", "5e-4")))
    n_pk = int(prm.get("n_packets", 2000))
    sref = prm.get("scenario")
    es_target = float(prm.get("es_n0_db", ES_N0_DB))
    guard = int(g_s * FS)
    lead = int(lead_s * FS)
    assert guard % 2 == 0
    assert guard * 1 == g_s * FS
    slot = BODY + guard
    slot_s = Fraction(slot) / FS
    start = packet_start(sem, guard, lead)
    n = start + n_pk * slot + 2000
    dsg = design(alphabet, los, es_target, scenario=sref)
    panels = dsg.panels
    n_pan = len(panels)
    scen = lab_scenario(
        alphabet,
        los,
        full_scale_dbm=dsg.tx_dbm,
        initial_states=dsg.s2,
        duration_s=Fraction(n) / FS,
        seed=int(ctx.seed),
        scenario=sref,
    )
    res = resolve(scen, scenario_path(sref).parent)
    seed = int(res.scenario.seed)
    run = RunRecord(
        Path(ctx.run_dir),
        seed,
        resolved=res,
        mode="lockstep",
        coupling="barrier",
        patch_root=Path(ctx.run_dir).parent,
    )
    src = QpskPacketSource(TX, seed, SPS, N_PRE, N_PAY, guard=guard, start=start)
    lat = LatencyModel(fixed_s=LAT_FIXED_S, jitter=None if j_s == 0 else f"uniform:0,{j_s}")
    c1, c2 = dsg.split(dsg.s1), dsg.split(dsg.s2)
    model = from_scenario(scen, backend="numpy_ref")
    caps = {pid: model.capabilities(pid) for pid in panels}
    ctrls = []
    for pid in panels:
        ctrl = TdmaAlternate(
            np.stack([c1[pid], c2[pid]]),
            slot_s=slot_s,
            timing=cast(Literal["timed", "asap", "legacy"], "legacy" if tick else sem),
            lead_s=lead_s if sem == "timed" else LAT_FIXED_S,
            panel=pid,
            name=CTRL_NAME if n_pan == 1 else f"{CTRL_NAME}_{pid}",
            min_latency_s=LAT_FIXED_S,
            latency=lat,
        )
        setup_controller(ctrl, caps, seed)
        ctrls.append(ctrl)
    ris_state = {pid: c1[pid] for pid in panels}
    cmds: list[dict[str, Any]] = []
    for m in range(n_pk):
        t_b = m * slot_s
        obs = make_observation(t_b, ris_state=ris_state)
        acts = [ctrl.decide(obs)[0] for ctrl in ctrls]
        ats = {a.at_s for a in acts}
        if len(ats) != 1:
            raise RuntimeError(f"guard {m}: panel controllers disagree on at_s {ats}")
        a0 = acts[0]
        L = draw_latency_s(seed, CTRL_NAME, lat, m, 0)  # one controller message per guard
        t_arr = t_b + L
        if sem == "timed":
            assert a0.at_s is not None
            t_req = Fraction(a0.at_s)
            t_eff = max(t_req, t_arr)
        else:
            t_req = t_arr
            t_eff = t_arr
        cmds.append(
            {
                "m": m,
                "target": RX1 if m % 2 == 0 else RX2,
                "t_issue": t_b,
                "latency": L,
                "t_arrival": t_arr,
                "t_req": t_req,
                "t_eff": t_eff,
                "late": bool(t_arr > t_req) if sem == "timed" else False,
                "states": {a.panel: np.asarray(a.states, dtype=np.int16) for a in acts},
            }
        )
    order = sorted(range(n_pk), key=lambda i: (cmds[i]["t_arrival"], i))
    reqs, arrivals = [], []
    for i in order:
        c = cmds[i]
        for pid in panels:
            reqs.append(
                CommandRequest(
                    _cmd_id(f"tdma:{c['m']}", pid, n_pan),
                    "ris.set_config",
                    CTRL_NAME,
                    pid,
                    CommandTime.at(c["t_eff"]),
                    states=c["states"][pid],
                )
            )
            arrivals.append(c["t_arrival"])
    # commands are released incrementally at their arrival times (<= every effective time:
    # timed max(t_req, arrival), asap arrival, tick-based next tick after arrival); this is
    # the upfront schedule's order and outcome without its quadratic bookkeeping cost
    sink, eng, summ = run_iq(
        res,
        src,
        reqs,
        n,
        adapter_times=arrivals if tick else None,
        release_times=arrivals,
    )
    recs = {r.cmd_id: r for r in eng.records}
    changed = dsg.changed_panels()
    rows = []
    n_err = 0
    evm_lin = []
    for m in range(n_pk):
        c = cmds[m]
        rx = c["target"]
        st = start + m * slot
        d = demod(sink.buf[rx], st, src.symbols(m))
        rs = [recs.get(_cmd_id(f"tdma:{m}", pid, n_pan)) for pid in changed]
        ks = [r.k_applied[rx] for r in rs if r is not None and r.k_applied and rx in r.k_applied]
        k_app = min(ks) if ks else None
        ts = [r.t_applied for r in rs if r is not None and r.t_applied is not None]
        err = d["bit_errors"] > 0
        n_err += int(err)
        evm_lin.append(float(np.mean(d["evm2"][N_PRE:])))
        rows.append(
            {
                "m": m,
                "target": rx,
                "t_issue_s": float(c["t_issue"]),
                "latency_s": float(c["latency"]),
                "t_arrival_s": float(c["t_arrival"]),
                "t_req_s": float(c["t_req"]),
                "t_eff_s": float(c["t_eff"]),
                "late": c["late"],
                "status": _joined_status([None if r is None else r.status for r in rs]),
                "t_applied_s": float(min(ts)) if ts else None,
                "k_applied_target": k_app,
                "packet_start": st,
                "packet_end": st + BODY,
                "switch_inside_packet": None if k_app is None else bool(st <= k_app < st + BODY),
                "bit_errors": d["bit_errors"],
                "evm_db": d["evm_rms_payload_db"],
            }
        )
    write_csv(Path(ctx.run_dir) / "packets.csv", rows)
    lo, hi = clopper_pearson(n_err, n_pk, 0.95)
    statuses: dict[str, int] = {}  # per panel command
    for r in eng.records:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    tick_lost = getattr(eng, "tick_overwritten", None)  # per panel command
    out = {
        "alphabet": alphabet,
        "los": los,
        "guard_s": str(g_s),
        "jitter_s": str(j_s),
        "semantics": sem,
        "lead_s": str(lead_s) if sem == "timed" else None,
        "scenario": dsg.scenario,
        "geometry_label": dsg.label,
        "n_panels": n_pan,
        "n_commands": len(reqs),
        "es_n0_target_db": es_target,
        "n_packets": n_pk,
        "packet_errors": n_err,
        "per": n_err / n_pk,
        "per_ci_low": lo,
        "per_ci_high": hi,
        "mean_evm_db": db10(float(np.mean(evm_lin))),
        "n_late": int(sum(c["late"] for c in cmds)),
        "record_status_counts": statuses,
        "tick_overwritten": tick_lost,
        "tick_overwritten_messages": None if tick_lost is None else tick_lost / n_pan,
        "tx_dbm": dsg.tx_dbm,
        "es_n0_db": dsg.es_n0_db,
        "verify_interp_max_rel": summ.verify_interp_max_rel,
        "analytic_expectation": analytic_per(sem, g_s, j_s, lead_s),
    }
    if n_pan > 1 or dsg.scenario is not None:  # the default geometry's result is unchanged
        out["design"] = dsg.diag
    write_json(Path(ctx.run_dir) / "result.json", out)
    info = eng.manifest_info()
    run.page_size = info["page_size"]
    run.verify_interp_max_rel = info["verify_interp_max_rel"]
    run.counters = {k: v for k, v in info["counters"].items() if isinstance(v, int)}
    run.add_assumption("lab |Gamma| ASSUMED" if alphabet == "lab" else "ideal 1-bit alphabet")
    run.add_assumption(
        f"TX power {dsg.tx_dbm:.3f} dBm for Es/N0 = {es_target:g} dB at RX1 under s1"
    )
    run.add_assumption(
        "command arrival = issue + keyed latency draw (one per guard); timed: switch at "
        "max(t_req, arrival) (apply_on_arrival); asap: at arrival; tick-based untimed "
        "semantics: next 2 ms tick after arrival, one pending slot per panel"
    )
    if n_pan > 1:
        run.add_assumption(
            f"{n_pan} panels act as one RIS: joint exact optimum over every cell; per guard "
            "one command per panel with the same latency draw and switch time"
        )
    blk = blocked_assumption(dsg.diag)
    if blk is not None:
        run.add_assumption(blk)
    write_meta(
        ctx.run_dir,
        experiment="E-B2",
        data_classes=["SIM-sremu", "ASSUMED"] if alphabet == "lab" else ["SIM-sremu"],
        description=(
            f"E-B2 timing requirement: {n_pk} packets alternating RX1/RX2, guard {g_s} s, "
            f"latency 0.2 ms + U[0, {j_s}] s, semantics {sem}"
            + (f" (lead {lead_s} s)" if sem == "timed" else "")
            + f", Es/N0 {es_target:g} dB at RX1; {dsg.label}"
        ),
        assumptions=run.assumptions,
        inputs=run.inputs,
        files={"packets.csv": "per packet: command timing, applied sample, bit errors, EVM"},
    )
    run.finish()
    return out


def analytic_per(sem: str, g: Fraction, j: Fraction, lead: Fraction) -> dict[str, Any]:
    """[DERIVED] expectations for the implemented timing (not claims): the probability
    that the switch lands inside a packet. timed: P(L > lead + G/2); asap: P(L > G);
    tick-based: not closed-form here (reported as None)."""
    sem = canonical_semantics(sem)
    lf = LAT_FIXED_S

    def p_gt(x: Fraction) -> float:
        if j == 0:
            return 1.0 if lf > x else 0.0
        return float(min(Fraction(1), max(Fraction(0), (lf + j - x) / j)))

    if sem == "timed":
        return {"p_switch_in_packet": p_gt(lead + g / 2), "rule": "P(L > lead + G/2)"}
    if sem == "asap":
        return {"p_switch_in_packet": p_gt(g), "rule": "P(L > G)"}
    return {"p_switch_in_packet": None, "rule": "next 2 ms tick after arrival"}


# =========================================================================== sweeps / CLI

SEED = 20261010
DELTAS = ["-0.5", "0.05", "0.25", "0.5", "0.75", "1.05"]
GUARDS = ["1e-4", "5e-4", "1e-3"]
JITTERS = ["0", "1e-4", "3e-4", "1e-3", "3e-3"]
_SEM5 = ["timed", "timed", "timed", "asap", "tick-based"]
_LEAD5 = ["5e-4", "2e-3", "5e-3", "5e-4", "5e-4"]  # lead_s is ignored by asap and tick-based
_R1 = "experiments.eb:run_eb1_point"
_R2 = "experiments.eb:run_eb2_point"


def sweep_specs(scenario: str | Path | None = None, *, smoke: bool = False) -> list[Any]:
    """The E-B sweep specs (``sremu.experiments.sweep.SweepSpec``) for one geometry.

    Full design: ``eb1`` (alphabet x LOS x transition x delta, 72 points); E-B2 at
    Es/N0 = 25 dB: ``eb2_lab`` (LOS present and blocked, 150 points) and ``eb2_ideal1``
    (LOS present, 75 points), each over semantics timed (lead 0.5/2/5 ms) | asap |
    tick-based x guard x J with 2000 packets; and the pre-declared Es/N0 = 15 dB
    sensitivity of the LOS-present slice, ``eb2_lab_es15`` and ``eb2_ideal1_es15`` (75
    points each). Grids match ``paper/sweeps/eb1.yaml``, ``eb2_lab.yaml`` and
    ``eb2_ideal1.yaml`` (``legacy`` there = ``tick-based`` here). Points of a non-default
    geometry carry ``scenario``. ``smoke``: one E-B1 point and one 50-packet E-B2 point."""
    from sremu.experiments.sweep import SweepSpec

    ref = scenario_ref(scenario)
    extra: dict[str, Any] = {} if ref is None else {"scenario": ref}
    geo = geometry_label(scenario)
    if smoke:
        return [
            SweepSpec.from_mapping(
                {
                    "name": "eb1_smoke",
                    "description": f"E-B1 smoke: lab, LOS present, step at 0.25 T_p; {geo}",
                    "runner": _R1,
                    "base": {
                        **extra,
                        "alphabet": "lab",
                        "los": "present",
                        "transition": "step",
                        "delta": "0.25",
                    },
                    "seeds": [SEED],
                }
            ),
            SweepSpec.from_mapping(
                {
                    "name": "eb2_smoke",
                    "description": f"E-B2 smoke: 50 packets, lab, timed lead 0.5 ms, G 0.5 ms, "
                    f"J 1 ms; {geo}",
                    "runner": _R2,
                    "base": {
                        **extra,
                        "alphabet": "lab",
                        "los": "present",
                        "semantics": "timed",
                        "lead_s": "5e-4",
                        "guard_s": "5e-4",
                        "jitter_s": "1e-3",
                        "n_packets": 50,
                    },
                    "seeds": [SEED],
                }
            ),
        ]
    grid2 = {"guard_s": list(GUARDS), "jitter_s": list(JITTERS)}
    specs = [
        {
            "name": "eb1",
            "description": f"E-B1 intra-frame switch: per-symbol EVM, per-packet EVM/BER, VA-5 "
            f"sample exactness; {geo}",
            "runner": _R1,
            "base": dict(extra),
            "grid": {
                "alphabet": ["lab", "ideal1"],
                "los": ["present", "blocked"],
                "transition": ["step", "linear20us", "linear200us"],
                "delta": list(DELTAS),
            },
        },
        {
            "name": "eb2_lab",
            "description": f"E-B2 PER and EVM vs latency jitter, lab alphabet, LOS present and "
            f"blocked, Es/N0 25 dB; {geo}",
            "runner": _R2,
            "base": {"alphabet": "lab", "n_packets": 2000, **extra},
            "zip": {
                "los": ["present"] * 5 + ["blocked"] * 5,
                "semantics": _SEM5 * 2,
                "lead_s": _LEAD5 * 2,
            },
            "grid": grid2,
        },
        {
            "name": "eb2_ideal1",
            "description": f"E-B2 PER and EVM vs latency jitter, ideal 1-bit alphabet, LOS "
            f"present, Es/N0 25 dB; {geo}",
            "runner": _R2,
            "base": {"alphabet": "ideal1", "los": "present", "n_packets": 2000, **extra},
            "zip": {"semantics": list(_SEM5), "lead_s": list(_LEAD5)},
            "grid": grid2,
        },
    ]
    for alph in ("lab", "ideal1"):
        specs.append(
            {
                "name": f"eb2_{alph}_es15",
                "description": f"E-B2 pre-declared sensitivity: Es/N0 {ES_N0_SENSITIVITY_DB:g} dB "
                f"at RX1 (instead of 25 dB), alphabet {alph}, LOS present; {geo}",
                "runner": _R2,
                "base": {
                    "alphabet": alph,
                    "los": "present",
                    "n_packets": 2000,
                    "es_n0_db": ES_N0_SENSITIVITY_DB,
                    **extra,
                },
                "zip": {"semantics": list(_SEM5), "lead_s": list(_LEAD5)},
                "grid": grid2,
            }
        )
    return [SweepSpec.from_mapping({**s, "seeds": [SEED]}) for s in specs]


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _plan(specs: Sequence[Any], scenario: str | None, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": "E-B",
        "scenario": scenario,
        "scenario_path": scenario_path(scenario).relative_to(REPO.resolve()).as_posix()
        if scenario_path(scenario).is_relative_to(REPO.resolve())
        else scenario_path(scenario).as_posix(),
        "geometry_label": geometry_label(scenario),
        "smoke": bool(args.smoke),
        "pre_declared": (
            "the full design below, including the Es/N0 15 dB sensitivity of the LOS-present "
            "slice (eb2_*_es15), is fixed in experiments/eb.py:sweep_specs and written here "
            "before any point runs"
        ),
        "specs": [
            {
                "name": s.name,
                "runner": s.runner,
                "spec_sha256": s.sha256,
                "n_points": len(s.points()),
                "base": dict(s.base),
                "grid": {k: list(v) for k, v in s.grid.items()},
                "zip": {k: list(v) for k, v in s.zip.items()},
                "seeds": list(s.seeds),
            }
            for s in specs
        ],
        "written_utc": _utc(),
        "argv": list(sys.argv),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Unattended E-B1 + E-B2 for one geometry (see the module docstring)."""
    ap = argparse.ArgumentParser(
        prog="python -m experiments.eb",
        description="E-B1 + E-B2 sweeps for one geometry, with per-point manifests.",
    )
    ap.add_argument(
        "--scenario",
        default=None,
        help="scenario file (repo-relative or absolute; default scenarios/lab_like_A.yaml)",
    )
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "eb", help="output root")
    ap.add_argument("--jobs", "-j", type=int, default=os.cpu_count() or 1)
    ap.add_argument(
        "--threads-per-job",
        type=int,
        default=None,
        help="numba threads per worker (default max(1, cpu_count // jobs))",
    )
    ap.add_argument("--resume", action="store_true", help="skip points already ok")
    ap.add_argument("--smoke", action="store_true", help="one E-B1 and one 50-packet E-B2 point")
    ap.add_argument("--only", choices=["eb1", "eb2"], default=None)
    ap.add_argument("--no-aggregate", action="store_true")
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    sp = scenario_path(args.scenario)
    if not sp.is_file():
        ap.error(f"scenario {sp} not found")
    ref = scenario_ref(args.scenario)
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tpj = args.threads_per_job or max(1, (os.cpu_count() or 1) // args.jobs)
    os.environ[_THREADS_ENV] = str(tpj)
    t0 = time.perf_counter()
    info = export_git_info(out)
    print(
        f"git {info['commit']} dirty={info['dirty']} patch={info['dirty_patch_sha256']}",
        flush=True,
    )
    for los in ("present", "blocked"):  # fail fast on a scenario E-B cannot use
        resolve(lab_scenario("lab", los, scenario=ref), sp.parent)
    specs = sweep_specs(ref, smoke=args.smoke)
    if args.only:
        specs = [s for s in specs if s.name.startswith(args.only)]
    plan = _plan(specs, ref, args)
    plan_path = out / ("eb_plan_smoke.json" if args.smoke else "eb_plan.json")
    if plan_path.exists():
        old = {s["name"]: s["spec_sha256"] for s in read_json(plan_path)["specs"]}
        new = {s["name"]: s["spec_sha256"] for s in plan["specs"]}
        clash = {k for k in new if k in old and old[k] != new[k]}
        if clash:
            print(f"refusing: {plan_path} pre-declares different specs {sorted(clash)}", flush=True)
            return 2
    else:
        write_json(plan_path, plan)
    print(
        f"E-B {plan['geometry_label']} ({plan['scenario_path']}): "
        f"{sum(s['n_points'] for s in plan['specs'])} points, jobs {args.jobs}, "
        f"numba threads/job {tpj}",
        flush=True,
    )
    from sremu.experiments.sweep import run_sweep

    summary: dict[str, Any] = {"specs": {}, "plan": plan_path.name, "t_start_utc": _utc()}
    for spec in specs:
        ts = time.perf_counter()

        def progress(rec: Mapping[str, Any], _name: str = spec.name) -> None:
            err = f" {rec.get('error')}" if rec["status"] != "ok" else ""
            print(
                f"[{_name}] {rec['point_id']} {rec['status']} {rec.get('wall_s', 0):.1f}s{err}",
                flush=True,
            )

        res = run_sweep(
            spec,
            out,
            jobs=args.jobs,
            resume=args.resume,
            allow_import=True,
            progress=progress,
            argv=list(sys.argv),
        )
        summary["specs"][spec.name] = {
            "n_points": len(res.records),
            "n_ok": res.n_ok,
            "n_failed": res.n_failed,
            "n_resumed": res.n_resumed,
            "failed": [
                {"point_id": r["point_id"], "error": r.get("error")}
                for r in res.records
                if r["status"] != "ok"
            ],
            "wall_s": time.perf_counter() - ts,
        }
        print(
            f"[{spec.name}] {res.n_ok} ok, {res.n_failed} failed, {res.n_resumed} resumed in "
            f"{time.perf_counter() - ts:.0f} s",
            flush=True,
        )
    va5: dict[str, Any] | None = None
    if not args.no_aggregate:
        from experiments import eb_aggregate

        results = out / ("results_smoke" if args.smoke else "results")
        eb1 = [out / s.name for s in specs if s.name.startswith("eb1")]
        eb2 = [out / s.name for s in specs if s.name.startswith("eb2")]
        if eb1:
            va5 = eb_aggregate.aggregate_eb1(eb1[0], results / "E-B1", strict=False)
            summary["va5"] = va5
        if eb2:
            rows = eb_aggregate.aggregate_eb2(eb2, results / "E-B2", strict=False)
            summary["eb2_rows"] = len(rows)
        summary["results"] = results.relative_to(out).as_posix()
    n_failed = sum(v["n_failed"] for v in summary["specs"].values())
    ok = n_failed == 0 and (va5 is None or bool(va5.get("pass")))
    summary["n_failed"] = n_failed
    summary["ok"] = ok
    summary["t_end_utc"] = _utc()
    summary["wall_s"] = time.perf_counter() - t0
    write_json(out / ("eb_summary_smoke.json" if args.smoke else "eb_summary.json"), summary)
    va5_pass = None if va5 is None else va5.get("pass")
    print(
        f"E-B done: ok={ok}, failed points {n_failed}, VA-5 {va5_pass}, {summary['wall_s']:.0f} s",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
