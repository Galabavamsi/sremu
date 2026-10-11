"""E-A: signaling delay in closed-loop binary RIS control (DESIGN.md section 11, E-A).

Pre-registration: ``paper/preregistration/E-A.md`` (written before any E-A sweep ran).

**Scenarios.** Every point is derived from ``scenarios/studies/ea_corridor.yaml`` by
:func:`corridor_scenario` (carrier, RIS offset h, LOS partition loss; A0 has no partition).

**Batch evaluation (vectorized).** ``sremu.engine.batch.run_batch`` evaluates one piece per
Python iteration (about 1.3 ms per piece on this laptop), i.e. about 2.4 h per
configuration of 33 333 s at a 5 ms step. This module evaluates the **same quantity**,
``h(t) = h_direct(x(t)) + sum_n Z[s_n(t), n](x(t))`` with ``P = P_T |h|^2`` on the 5 ms grid
split at every command switch time (the left-point piece rule of ``run_batch``), in
vectorized form:

* the channel of the scene model (``sremu.physics.propagation.from_scenario``,
  ``sremu.control.controller.csi_from_snapshot``) is sampled on a 1 mm grid of MU positions
  (a copy of the scenario whose MU moves at constant velocity, so ``snapshot(t)`` has the MU
  at x = t) and interpolated complex-linearly at the trajectory position; the interpolation
  error is measured against ``csi_from_snapshot`` of the RWP model at random times and
  stored with the cache;
* MU positions come from the scenario's own RWP trajectory (legs from
  ``RandomWaypoint1D.leg_starts``/``waypoint``), checked against ``node_position``;
* controller decisions come from the real controller objects (``sremu.control.builtin``):
  position-driven controllers with a 1 cm ``GridChannel`` are tabulated per grid index by
  calling their ``decide`` (their decisions are a function of that index; the table is
  checked against the live ``DelayedController`` on a sample of barriers); ``Urap`` and
  ``ProbeBandit`` are called barrier by barrier with the keyed generator
  ``ctrl/<name>`` of the run seed (the one ``run_batch`` and the engine hand them);
* a command issued at barrier t_b switches at t_b + 1 ms (RIS-time convention).

:func:`check_against_run_batch` compares this evaluator with ``run_batch`` on a window.

Sweep runner: ``experiments.ea:run_point`` (``paper/sweeps/ea_*.yaml``); aggregation:
:func:`aggregate` (gains, retention, tau_half, hypotheses H-A1..H-A3).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numba
import numpy as np

from experiments.common import (
    REPO,
    RunRecord,
    db10,
    quiet_warnings,
    read_json,
    write_csv,
    write_json,
    write_meta,
)
from sremu.analysis.stats import batch_means
from sremu.contracts import CsiView, LatencyModel
from sremu.control.builtin import (
    DelayedController,
    ExactOptimum,
    ProbeBandit,
    RisAbsent,
    Urap,
    WclHeuristic,
    optimum_codebook,
)
from sremu.control.controller import (
    GridChannel,
    csi_from_snapshot,
    make_observation,
    setup_controller,
)
from sremu.physics.propagation import from_scenario
from sremu.scenario.load import parse_scenario, read_document, resolve_scenario
from sremu.units import C0

TEMPLATE = REPO / "scenarios" / "studies" / "ea_corridor.yaml"
L_M = 10.0  # user line length l, m
D0_M = 0.5  # AP offset from the user line, m
V_MPS = 1.0  # MU speed, m/s
SEED = 20261010  # scenario seed (trajectory legs, controller generators)
LINK = "ap.tx0->mu.rx0"
PANEL = "ris0"
NODE = "mu"
RX = "mu.rx0"
AP_POS = np.array([0.0, -D0_M, 0.0])
P_T_W = 1e-3  # 0 dBm
DT_S = Fraction(1, 200)  # 5 ms evaluation step
LAT_S = Fraction(1, 1000)  # 1 ms command latency
COARSE_DEN = 100  # controller grid: 1 cm


def fine_den(fc_hz: float) -> int:
    """Fine CSI grid points per metre: 1 mm up to 1.5 GHz, 0.5 mm above (lambda/250 at
    2.4 GHz), so the complex-linear interpolation error stays below about 1e-4 relative."""
    return 1000 if float(fc_hz) <= 1.5e9 else 2000


N_BATCHES = 20
STEPS_PER_BATCH = 333_333  # 1666.665 s = 500 mean movement periods l/(3v)
TAUS = ("0.01", "0.03", "0.1", "0.3", "1", "3", "10")
BANDIT_DWELL_S = Fraction(1, 50)  # 20 ms native dwell
URAP_P = 0.2
CACHE_VERSION = "ea-geom-v1"


# =========================================================================== scenarios


def n_cells(fc_hz: float, l_m: float = L_M) -> tuple[int, float]:
    """(N, k): cell width = length k = lambda/4 and N = floor(l/k)."""
    k = C0 / float(fc_hz) / 4.0
    return int(math.floor(l_m / k)), k


def geometry_key(fc_hz: float, h_m: float, loss_db: float | None) -> str:
    """Stable short name of a geometry, e.g. ``f1.0G_h2_L20`` (``L0`` = no partition)."""
    lp = 0 if not loss_db else loss_db
    return f"f{float(fc_hz) / 1e9:.1f}G_h{h_m:g}_L{lp:g}"


def corridor_scenario(
    fc_hz: float,
    h_m: float,
    loss_db: float | None,
    *,
    survey: bool = False,
    seed: int = SEED,
    duration_s: str | None = None,
) -> dict[str, Any]:
    """The E-A corridor scenario mapping for one geometry (section 9.2 field names).

    ``loss_db`` None or 0: no partition (A0). ``survey``: the MU moves at 1 m/s along +x from
    x = 0 (``constant_velocity``), so ``snapshot(t)`` has the MU at x = t (CSI grid).
    """
    d = copy.deepcopy(read_document(TEMPLATE))
    n, k = n_cells(fc_hz)
    d["name"] = ("ea_survey_" if survey else "ea_") + geometry_key(fc_hz, h_m, loss_db).replace(
        ".", "p"
    )
    d["seed"] = int(seed)
    for nd in d["nodes"]:
        for s in nd["streams"]:
            s["fc_hz"] = float(fc_hz)
    p = d["ris"][0]
    p["pose"]["center_m"] = [n * k / 2.0, float(h_m), 0.0]
    p["grid"]["n_u"] = n
    p["grid"]["pitch_u_m"] = k
    p["grid"]["pitch_v_m"] = k
    if not loss_db:
        d["environment"] = {}
    else:
        d["environment"]["occluders"][0]["transmission_loss_db"] = float(loss_db)
    if survey:
        mu = next(nd for nd in d["nodes"] if nd["id"] == NODE)
        mu["trajectory"] = {
            "model": "constant_velocity",
            "start_m": [0.0, 0.0, 0.0],
            "velocity_mps": [V_MPS, 0.0, 0.0],
        }
    if duration_s is not None:
        d["run"]["duration_s"] = str(duration_s)
    d["description"] = (
        f"E-A corridor, f_c = {fc_hz / 1e9:g} GHz, h = {h_m:g} m, N = {n}, "
        + (f"LOS partition {loss_db:g} dB" if loss_db else "no partition (A0)")
        + ("; survey trajectory (CSI grid)" if survey else "; RWP 1-D MU")
        + "; SIM geometry"
    )
    return d


def resolve(d: Mapping[str, Any]) -> Any:
    """Validate, resolve and guard a scenario mapping (``ResolvedScenario``)."""
    from sremu.scenario.guards import check_scenario

    quiet_warnings()
    sc = parse_scenario(dict(d), label=str(d.get("name")))
    res = resolve_scenario(sc, TEMPLATE.parent, warn=False)
    rep = check_scenario(res)
    rep.raise_for_errors()
    return res


# =========================================================================== geometry cache


@dataclass
class Geometry:
    """Cached channel tables of one corridor geometry.

    ``x`` (J,) fine MU positions (m, 1 mm grid); ``hd`` (J,) h_direct; ``z`` (J, S, N) the
    cascade (carrier phase included); ``grid`` the 1 cm ``GridChannel``; ``codebook`` the
    16 exact optima of ``ProbeBandit``; ``caps`` the panel capabilities; ``scenario`` the RWP
    scenario mapping; ``check`` the interpolation check.
    """

    key: str
    den: int
    fc_hz: float
    h_m: float
    loss_db: float
    n: int
    k_m: float
    x: np.ndarray
    hd: np.ndarray
    z: np.ndarray
    grid: GridChannel
    codebook: np.ndarray
    caps: dict[str, Any]
    scenario: dict[str, Any]
    check: dict[str, Any]

    @property
    def n_coarse(self) -> int:
        return int(self.grid.points.shape[0])


def _fine_times(n_fine: int, den: int) -> list[Fraction]:
    return [Fraction(j, den) for j in range(n_fine)]


def _views_from_tables(x: np.ndarray, hd: np.ndarray, z: np.ndarray, fc: float) -> list[dict]:
    out = []
    for j in range(x.shape[0]):
        out.append({LINK: CsiView(LINK, Fraction(0), fc, complex(hd[j]), {PANEL: z[j]})})
    return out


def interp_csi(geom: Geometry, xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Complex-linear interpolation of (h_direct, cascade) at positions ``xq`` (m)."""
    u = np.asarray(xq, dtype=np.float64) * geom.den
    j = np.clip(np.floor(u).astype(np.int64), 0, geom.x.shape[0] - 2)
    w = u - j
    hd = geom.hd[j] * (1 - w) + geom.hd[j + 1] * w
    z = geom.z[j] * (1 - w)[:, None, None] + geom.z[j + 1] * w[:, None, None]
    return hd, z


def build_geometry(
    fc_hz: float,
    h_m: float,
    loss_db: float | None,
    cache_dir: str | Path,
    *,
    n_check: int = 400,
    check_horizon_s: float = 33_333.0,
) -> Geometry:
    """Build (or load) the cached channel tables of one geometry.

    The 1 mm grid is sampled from the survey model; the interpolation is checked against
    ``csi_from_snapshot`` of the RWP model at ``n_check`` keyed random times in
    ``[0, check_horizon_s]`` (relative error of h for the all-ON configuration and of
    h_direct; stored in ``check``).
    """
    quiet_warnings()
    loss = float(loss_db or 0.0)
    key = geometry_key(fc_hz, h_m, loss)
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    npz = cdir / f"geom_{key}.npz"
    meta_p = cdir / f"geom_{key}.json"
    scen = corridor_scenario(fc_hz, h_m, loss)
    n, k = n_cells(fc_hz)
    fc = float(fc_hz)
    den = fine_den(fc)
    if npz.exists() and meta_p.exists():
        meta = read_json(meta_p)
        if meta.get("version") == CACHE_VERSION:
            dat = np.load(npz)
            x, hd, z, cb = dat["x"], dat["hd"], dat["z"], dat["codebook"]
            coarse = slice(0, None, den // COARSE_DEN)
            views = _views_from_tables(x[coarse], hd[coarse], z[coarse], fc)
            pts = np.stack([x[coarse], np.zeros_like(x[coarse]), np.zeros_like(x[coarse])], 1)
            return Geometry(
                key,
                den,
                fc,
                h_m,
                loss,
                n,
                k,
                x,
                hd,
                z,
                GridChannel(NODE, pts, views),
                cb,
                meta["caps"],
                scen,
                meta["check"],
            )
    t0 = time.perf_counter()
    survey = from_scenario(corridor_scenario(fc_hz, h_m, loss, survey=True), backend="numpy_ref")
    n_fine = int(round(L_M * den)) + 1
    times = _fine_times(n_fine, den)
    x = np.empty(n_fine)
    hd = np.empty(n_fine, dtype=np.complex128)
    z = None
    for j, t in enumerate(times):
        snap = survey.snapshot(t)
        v = csi_from_snapshot(survey, snap, LINK)
        if z is None:
            z = np.empty((n_fine, *v.cascade[PANEL].shape), dtype=np.complex128)
        x[j] = float(survey.node_position(NODE, t)[0])
        hd[j] = v.h_direct
        z[j] = v.cascade[PANEL]
    assert z is not None
    np.testing.assert_allclose(x, np.arange(n_fine) / den, rtol=0, atol=1e-12)
    # ProbeBandit codebook: exact optima at x_i = (i + 1/2) l/16
    xs16 = [Fraction(int(L_M) * (2 * i + 1), 32) for i in range(16)]
    views16 = [csi_from_snapshot(survey, survey.snapshot(t), LINK) for t in xs16]
    cb = optimum_codebook(views16, PANEL)
    cap = survey.capabilities(PANEL)
    caps = {
        "n_elements": cap.n_elements,
        "n_controls": cap.n_controls,
        "states": list(cap.states),
        "gamma_source": cap.gamma_source,
    }
    t_build = time.perf_counter() - t0
    # interpolation check against the RWP model itself
    rwp = from_scenario(scen, backend="numpy_ref")
    from sremu import rng as _rng

    g = _rng.generator(SEED, "trial", f"ea/interp_check/{key}")
    tq = np.sort(g.uniform(0.0, check_horizon_s, n_check))
    errs_h, errs_d, errs_x = [], [], []
    geom_tmp = Geometry(key, den, fc, h_m, loss, n, k, x, hd, z, None, cb, caps, scen, {})  # type: ignore[arg-type]
    for t in tq:
        tf = Fraction(repr(float(t)))
        v = csi_from_snapshot(rwp, rwp.snapshot(tf), LINK)
        xm = float(rwp.node_position(NODE, tf)[0])
        hi, zi = interp_csi(geom_tmp, np.array([xm]))
        on = np.ones(n, dtype=np.int64)
        h_true = v.h_direct + complex(v.cascade[PANEL][on, np.arange(n)].sum())
        h_int = complex(hi[0]) + complex(zi[0][on, np.arange(n)].sum())
        errs_h.append(abs(h_int - h_true) / abs(h_true))
        errs_d.append(abs(complex(hi[0]) - v.h_direct) / abs(v.h_direct))
        errs_x.append(xm)
    check = {
        "n_check": n_check,
        "max_rel_err_h_all_on": float(np.max(errs_h)),
        "max_rel_err_h_direct": float(np.max(errs_d)),
        "max_db_err_all_on": float(np.max(np.abs(20 * np.log10(1 + np.asarray(errs_h))))),
        "build_s": t_build,
    }
    np.savez(npz.with_suffix(".tmp.npz"), x=x, hd=hd, z=z, codebook=cb)
    Path(npz.with_suffix(".tmp.npz")).replace(npz)
    write_json(
        meta_p,
        {
            "version": CACHE_VERSION,
            "key": key,
            "fc_hz": fc,
            "h_m": h_m,
            "loss_db": loss,
            "n_cells": n,
            "k_m": k,
            "caps": caps,
            "check": check,
            "sha256_npz": hashlib.sha256(npz.read_bytes()).hexdigest(),
        },
    )
    return build_geometry(fc_hz, h_m, loss, cache_dir)


# =========================================================================== trajectory


@dataclass
class Trajectory:
    """Vectorized MU positions of the scenario's RWP trajectory (legs from the trajectory
    object itself). ``t0`` (L,) leg start times, ``w0``/``w1`` (L,) leg end points (x, m),
    ``dur`` (L,) leg durations (s)."""

    t0: np.ndarray
    w0: np.ndarray
    w1: np.ndarray
    dur: np.ndarray
    check_max_abs_m: float

    def x(self, t: np.ndarray) -> np.ndarray:
        tt = np.asarray(t, dtype=np.float64)
        i = np.searchsorted(self.t0, tt, side="right") - 1
        a = np.clip((tt - self.t0[i]) / self.dur[i], 0.0, 1.0)
        return self.w0[i] + a * (self.w1[i] - self.w0[i])


def build_trajectory(
    scen: Mapping[str, Any], horizon_s: float, cache_dir: str | Path
) -> Trajectory:
    """Legs of the MU's RWP trajectory up to ``horizon_s`` (cached), checked against
    ``node_position`` at 2000 keyed random exact times (max |dx| stored)."""
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    p = cdir / f"traj_seed{int(scen['seed'])}_{int(horizon_s)}.npz"
    if p.exists():
        d = np.load(p)
        return Trajectory(d["t0"], d["w0"], d["w1"], d["dur"], float(d["check"]))
    model = from_scenario(dict(scen), backend="numpy_ref")
    traj = model.nodes[NODE].trajectory
    starts = traj.leg_starts(horizon_s + 100.0)
    nleg = len(starts)
    wp = np.array([float(traj.waypoint(i)[0]) for i in range(nleg + 1)])
    t0 = np.asarray(starts, dtype=np.float64)
    w0, w1 = wp[:-1], wp[1:]
    dur = np.abs(w1 - w0) / V_MPS
    tr = Trajectory(t0, w0, w1, dur, 0.0)
    from sremu import rng as _rng

    g = _rng.generator(int(scen["seed"]), "trial", "ea/trajectory_check")
    tq = g.uniform(0.0, horizon_s, 2000)
    xs = tr.x(tq)
    ref = np.array([float(model.node_position(NODE, Fraction(repr(float(t))))[0]) for t in tq])
    err = float(np.max(np.abs(xs - ref)))
    if err > 1e-9:
        raise AssertionError(f"vectorized trajectory differs from node_position by {err} m")
    np.savez(p.with_suffix(".tmp.npz"), t0=t0, w0=w0, w1=w1, dur=dur, check=err)
    Path(p.with_suffix(".tmp.npz")).replace(p)
    return Trajectory(t0, w0, w1, dur, err)


# =========================================================================== evaluation kernel


@numba.njit(cache=True)
def _eval_h(xq, cfg, hd, z, states, den, out):  # pragma: no cover - numba
    """h at positions ``xq`` (m) for configuration rows ``cfg`` (-1: RIS disabled); fine
    grid of ``den`` points per metre."""
    nj = hd.shape[0]
    nn = z.shape[2]
    for p in range(xq.shape[0]):
        u = xq[p] * den
        j = int(math.floor(u))
        if j < 0:
            j = 0
        if j > nj - 2:
            j = nj - 2
        w = u - j
        a = hd[j] * (1.0 - w) + hd[j + 1] * w
        c = cfg[p]
        if c >= 0:
            acc0 = 0j
            acc1 = 0j
            for n in range(nn):
                s = states[c, n]
                if s >= 0:
                    acc0 += z[j, s, n]
                    acc1 += z[j + 1, s, n]
            a += acc0 * (1.0 - w) + acc1 * w
        out[p] = a


def eval_power_w(geom: Geometry, xq: np.ndarray, cfg: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Received power ``P_T |h|^2`` (W) at positions ``xq`` with configuration rows ``cfg``
    of ``states`` (C, N) int16 (row -1: RIS disabled)."""
    out = np.empty(xq.shape[0], dtype=np.complex128)
    st = np.ascontiguousarray(states, dtype=np.int16)
    if st.ndim != 2 or st.shape[0] == 0:
        st = np.zeros((1, geom.n), dtype=np.int16)
    _eval_h(
        np.ascontiguousarray(xq, dtype=np.float64),
        np.ascontiguousarray(cfg, dtype=np.int64),
        geom.hd,
        geom.z,
        st,
        float(geom.den),
        out,
    )
    return P_T_W * (out.real * out.real + out.imag * out.imag)


def nearest_coarse(x: np.ndarray, n_coarse: int) -> np.ndarray:
    """Index of the nearest 1 cm grid point (ties to the lower index, as
    ``GridChannel.index``)."""
    u = np.asarray(x, dtype=np.float64) * COARSE_DEN - 0.5
    return np.clip(np.ceil(u), 0, n_coarse - 1).astype(np.int64)


# =========================================================================== controllers


class _Rewards(list):
    """Reward history of ``ProbeBandit`` (``(arm, p)`` pairs, appended by its ``decide``)
    that keeps the per-arm sums of ``p/top`` incrementally, in insertion order, and
    recomputes them (in insertion order) only when the running maximum ``top`` changes, so
    the sums are bitwise those of ``ProbeBandit.means``."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.top = -math.inf
        self.sums = np.zeros(k)

    def append(self, item: tuple[int, float]) -> None:  # type: ignore[override]
        super().append(item)
        arm, p = item
        if p > self.top:
            self.top = p
            self.sums = np.zeros(self.k)
            for a, q in self:
                self.sums[a] += q / self.top if self.top > 0 else 0.0
        else:
            self.sums[arm] += p / self.top if self.top > 0 else 0.0


class FastProbeBandit(ProbeBandit):
    """``ProbeBandit`` with O(K) ``means()`` (the parent rescans the whole reward history at
    every decision, O(n^2) per run). Decisions are bitwise those of the parent: same
    ``decide``/``choose``, and ``means`` returns the same floating-point values (see
    :class:`_Rewards`); :func:`check_fast_bandit` verifies the arm sequence."""

    def reset(self) -> None:
        super().reset()
        self._raw = _Rewards(self.codebook.shape[0])

    def means(self) -> np.ndarray:
        k = self.codebook.shape[0]
        if not self._raw:
            return np.full(k, np.nan)
        sums = np.asarray(self._raw.sums, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts > 0, sums / np.maximum(self.counts, 1), np.nan)


_CAPS: dict[str, dict] = {}


def _caps_obj(geom: Geometry) -> dict:
    if geom.key not in _CAPS:
        model = from_scenario(geom.scenario, backend="numpy_ref")
        _CAPS[geom.key] = {PANEL: model.capabilities(PANEL)}
    return _CAPS[geom.key]


def _obs(t_b: Fraction, x: float, state: np.ndarray, power_dbm: float | None = None) -> Any:
    return make_observation(
        t_b,
        ris_state={PANEL: state},
        positions={NODE: np.array([x, 0.0, 0.0]), "ap": AP_POS},
        rx_power_dbm=None if power_dbm is None else {RX: power_dbm},
    )


def make_inner(kind: str, geom: Geometry, *, skip_unchanged: bool = True) -> Any:
    """A position-driven controller with the 1 cm ``GridChannel`` (oracle geometry)."""
    kw = dict(
        link=LINK,
        panel=PANEL,
        channel=geom.grid,
        skip_unchanged=skip_unchanged,
        min_latency_s=LAT_S,
        latency=LatencyModel(fixed_s=LAT_S),
    )
    if kind == "exact":
        return ExactOptimum(name="exact_optimum", **kw)
    if kind == "wcl":
        return WclHeuristic(4, 3, name="wcl_heuristic", **kw)
    raise ValueError(kind)


_TABLE_DIR: dict[str, Path] = {}


def decision_table(kind: str, geom: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """``(table (G,) config row per 1 cm grid index, configs (C, N) int16)`` from the real
    controller's ``decide`` at every grid point (``skip_unchanged=False``); cached next to
    the geometry tables when :func:`set_table_dir` was called for the geometry."""
    cdir = _TABLE_DIR.get(geom.key)
    path = None if cdir is None else cdir / f"dtable_{kind}_{geom.key}.npz"
    if path is not None and path.exists():
        d = np.load(path)
        return d["table"], d["configs"]
    table, configs = _decision_table(kind, geom)
    if path is not None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, table=table, configs=configs)
        tmp.replace(path)
    return table, configs


def set_table_dir(geom: Geometry, cache_dir: str | Path) -> None:
    """Cache decision tables of ``geom`` in ``cache_dir``."""
    _TABLE_DIR[geom.key] = Path(cache_dir)


def _decision_table(kind: str, geom: Geometry) -> tuple[np.ndarray, np.ndarray]:
    ctrl = make_inner(kind, geom, skip_unchanged=False)
    setup_controller(ctrl, _caps_obj(geom), SEED)
    rows: list[np.ndarray] = []
    table = np.empty(geom.n_coarse, dtype=np.int64)
    index: dict[bytes, int] = {}
    zeros = np.zeros(geom.n, dtype=np.int16)
    for i in range(geom.n_coarse):
        (act,) = ctrl.decide(_obs(Fraction(0), float(geom.grid.points[i, 0]), zeros))
        s = np.asarray(act.states, dtype=np.int16)
        kb = s.tobytes()
        if kb not in index:
            index[kb] = len(rows)
            rows.append(s)
        table[i] = index[kb]
    return table, np.stack(rows)


def check_decision_table(
    kind: str,
    geom: Geometry,
    traj: Trajectory,
    tau: Fraction,
    table: np.ndarray,
    configs: np.ndarray,
    n_barriers: int = 300,
) -> dict[str, int]:
    """Run the live ``DelayedController`` on the first ``n_barriers`` barriers and compare its
    commands (issued or skipped, and the states) with the tabulated decisions."""
    ctrl = DelayedController(make_inner(kind, geom), tau)
    setup_controller(ctrl, _caps_obj(geom), SEED)
    tb = np.arange(n_barriers) * float(tau)
    xb = traj.x(tb)
    idx = nearest_coarse(xb, geom.n_coarse)
    prev = -1
    mism = 0
    idx_mism = 0
    state = np.zeros(geom.n, dtype=np.int16)
    for m in range(n_barriers):
        obs = _obs(m * tau, float(xb[m]), state)
        acts = list(ctrl.decide(obs))
        gi = geom.grid.index(obs.positions)
        if gi != idx[m]:
            idx_mism += 1
        c = int(table[idx[m]])
        issue = c != prev
        if issue != bool(acts) or (acts and not np.array_equal(acts[0].states, configs[c])):
            mism += 1
        if acts:
            state = np.asarray(acts[0].states, dtype=np.int16)
        prev = c
    return {"n_barriers": n_barriers, "decision_mismatches": mism, "index_mismatches": idx_mism}


@dataclass
class EvalResult:
    """Per-cell (5 ms) mean powers ``p_cell`` (W) and bookkeeping."""

    p_cell: np.ndarray
    n_commands: int
    extra: dict[str, Any]


def _cell_times(k0: int, k1: int) -> np.ndarray:
    return np.arange(k0, k1, dtype=np.float64) * float(DT_S)


def evaluate_open_loop(
    geom: Geometry,
    traj: Trajectory,
    n_cells_total: int,
    period_cells: int,
    decide: Any,
    *,
    chunk_cells: int = 40_000,
    rule: str = "mid",
) -> EvalResult:
    """Open-loop controllers: ``decide(m0, m1, t_b (M,), x_b (M,)) -> (issue (M,) bool,
    cfg (M,) int64, states (C, N))`` gives the commands of barriers [m0, m1); a command of
    barrier m switches at t_b + 1 ms (inside cell m*period_cells). ``rule``: each piece is
    evaluated at its start (``left``, the rule of ``run_batch``) or at its midpoint
    (``mid``)."""
    lat = float(LAT_S)
    dt = float(DT_S)
    off, off_a, off_b = _offsets(rule)
    p = np.empty(n_cells_total, dtype=np.float64)
    chunk = max(period_cells, (chunk_cells // period_cells) * period_cells)
    cur_cfg = 0  # initial configuration row (filled by decide's first table)
    cur_states = np.zeros((1, geom.n), dtype=np.int16)
    n_cmd = 0
    for k0 in range(0, n_cells_total, chunk):
        k1 = min(n_cells_total, k0 + chunk)
        m0 = k0 // period_cells
        m1 = -(-k1 // period_cells)
        tb = np.arange(m0, m1, dtype=np.float64) * period_cells * dt
        xb = traj.x(tb)
        issue, cfg_b, states = decide(m0, m1, tb, xb)
        # previous configuration as an extra row of this chunk's table
        prev_row = states.shape[0]
        st = np.concatenate(
            [states, cur_states[[cur_cfg]] if cur_cfg >= 0 else np.zeros((1, geom.n), np.int16)]
        )
        prev_disabled = cur_cfg < 0
        cells = np.arange(k0, k1)
        t = cells * dt
        x = traj.x(t + off)
        # config in force at the start of each cell: last issued command with barrier cell < k
        new_at = np.full(k1 - k0, -2, dtype=np.int64)
        kb = np.arange(m0, m1) * period_cells
        sel = issue & (kb + 1 < k1) & (kb + 1 >= k0)
        new_at[kb[sel] + 1 - k0] = cfg_b[sel]
        # a command of the barrier just before the chunk (kb = k0 - 1) cannot exist: kb is a
        # multiple of period_cells and chunks start at multiples of it
        first = -1 if prev_disabled else prev_row
        pos = np.where(new_at != -2, np.arange(k1 - k0), -1)
        np.maximum.accumulate(pos, out=pos)
        cfg_cell = np.where(pos >= 0, new_at[np.maximum(pos, 0)], first)
        pw = eval_power_w(geom, x, cfg_cell, st)
        # split cells (switch at kb*dt + lat)
        sw = issue & (kb >= k0) & (kb < k1)
        if np.any(sw):
            ks = kb[sw] - k0
            pw_old = (
                pw[ks]
                if off_a == off
                else eval_power_w(geom, traj.x(t[ks] + off_a), cfg_cell[ks], st)
            )
            xs = traj.x(t[ks] + lat + off_b)
            pw_new = eval_power_w(geom, xs, cfg_b[sw], st)
            pw[ks] = (lat * pw_old + (dt - lat) * pw_new) / dt
        p[k0:k1] = pw
        n_cmd += int(np.count_nonzero(issue))
        # carry the configuration in force at the end of the chunk
        last_issue = np.nonzero(issue)[0]
        if last_issue.size:
            c_last = int(cfg_b[last_issue[-1]])
            if c_last >= 0:
                cur_states = states[[c_last]]
                cur_cfg = 0
            else:
                cur_cfg = -1
    return EvalResult(p, n_cmd, {})


def run_controller(
    kind: str,
    geom: Geometry,
    traj: Trajectory,
    tau_s: str | Fraction | None,
    n_cells_total: int,
    *,
    check_barriers: int = 300,
    rule: str = "mid",
) -> EvalResult:
    """Evaluate one controller on ``n_cells_total`` cells of 5 ms (see the module docs);
    ``rule``: piece evaluation point, ``mid`` (default) or ``left`` (``run_batch``).

    ``kind``: ``absent`` | ``urap`` | ``exact`` | ``wcl`` | ``bandit`` | ``instant`` (exact
    optimum at the current position, no delay) | ``stale`` (exact optimum for the position at
    the same offset half the run away).
    """
    dt = float(DT_S)
    if kind == "absent":
        ctrl = RisAbsent(panel=PANEL, period_s=1, min_latency_s=LAT_S, latency=LatencyModel(LAT_S))
        setup_controller(ctrl, _caps_obj(geom), SEED)
        acts = list(ctrl.decide(_obs(Fraction(0), 0.0, np.zeros(geom.n, np.int16))))
        assert acts and acts[0].enabled is False
        period = n_cells_total  # one barrier (t = 0)

        def dec_absent(m0, m1, tb, xb):
            issue = np.zeros(m1 - m0, dtype=bool)
            cfg = np.full(m1 - m0, -1, dtype=np.int64)
            if m0 == 0:
                issue[0] = True
            return issue, cfg, np.zeros((0, geom.n), np.int16)

        return evaluate_open_loop(geom, traj, n_cells_total, period, dec_absent, rule=rule)
    if kind in ("instant", "stale"):
        table, configs = decision_table("exact", geom)
        p = np.empty(n_cells_total)
        chunk = 200_000
        half = n_cells_total // 2
        for k0 in range(0, n_cells_total, chunk):
            k1 = min(n_cells_total, k0 + chunk)
            cells = np.arange(k0, k1)
            off = _offsets(rule)[0]
            x = traj.x(cells * dt + off)
            if kind == "instant":
                xd = x
            else:  # the same offset in the batch half the run away
                xd = traj.x(((cells + half) % n_cells_total) * dt + off)
            cfg = table[nearest_coarse(xd, geom.n_coarse)]
            p[k0:k1] = eval_power_w(geom, x, cfg, configs)
        return EvalResult(p, 0, {"n_configs": int(configs.shape[0])})
    tau = Fraction(str(tau_s))
    if kind in ("exact", "wcl"):
        period = int(tau / DT_S)
        assert period * DT_S == tau
        table, configs = decision_table(kind, geom)
        chk = check_decision_table(kind, geom, traj, tau, table, configs, check_barriers)
        state = {"prev": -1}

        def dec_table(m0, m1, tb, xb):
            cfg = table[nearest_coarse(xb, geom.n_coarse)]
            prev = np.concatenate([[state["prev"]], cfg[:-1]])
            issue = cfg != prev
            state["prev"] = int(cfg[-1])
            return issue, cfg, configs

        r = evaluate_open_loop(geom, traj, n_cells_total, period, dec_table, rule=rule)
        r.extra.update({"n_configs": int(configs.shape[0]), "decision_check": chk})
        return r
    if kind == "urap":
        period = int(tau / DT_S)
        assert period * DT_S == tau
        ctrl = Urap(
            URAP_P,
            panel=PANEL,
            name="urap",
            period_s=tau,
            min_latency_s=LAT_S,
            latency=LatencyModel(LAT_S),
        )
        setup_controller(ctrl, _caps_obj(geom), SEED)
        zeros = np.zeros(geom.n, dtype=np.int16)

        def dec_urap(m0, m1, tb, xb):
            rows = []
            for m in range(m0, m1):
                (a,) = ctrl.decide(_obs(m * tau, 0.0, zeros))
                rows.append(a.states)
            st = np.stack(rows).astype(np.int16)
            return np.ones(m1 - m0, dtype=bool), np.arange(m1 - m0, dtype=np.int64), st

        return evaluate_open_loop(
            geom, traj, n_cells_total, period, dec_urap, chunk_cells=20_000, rule=rule
        )
    if kind == "bandit":
        return run_bandit(geom, traj, tau, n_cells_total, rule=rule)
    raise ValueError(f"unknown controller kind {kind!r}")


def config_channel_table(geom: Geometry, states: np.ndarray) -> np.ndarray:
    """``H[c, j] = h_direct(x_j) + sum_n Z[s_c,n, n](x_j)`` on the 1 mm grid for each
    configuration row of ``states`` (C, N) (state -1: dead, contributes 0)."""
    st = np.asarray(states, dtype=np.int64)
    cols = np.arange(geom.n)
    out = np.empty((st.shape[0], geom.x.shape[0]), dtype=np.complex128)
    for c in range(st.shape[0]):
        alive = st[c] >= 0
        out[c] = geom.hd + geom.z[:, st[c][alive], cols[alive]].sum(axis=1)
    return out


def table_power_w(table: np.ndarray, xq: np.ndarray, den: int) -> np.ndarray:
    """``P_T |h|^2`` at positions ``xq`` for every row of a channel table (complex-linear
    interpolation on the 1 mm grid, as :func:`eval_power_w`); returns (C, len(xq))."""
    u = np.asarray(xq, dtype=np.float64) * den
    j = np.clip(np.floor(u).astype(np.int64), 0, table.shape[1] - 2)
    w = u - j
    h = table[:, j] * (1.0 - w)[None, :] + table[:, j + 1] * w[None, :]
    return P_T_W * (h.real * h.real + h.imag * h.imag)


def run_bandit(
    geom: Geometry,
    traj: Trajectory,
    tau: Fraction,
    n_cells_total: int,
    *,
    slow_reference: bool = False,
    rule: str = "mid",
) -> EvalResult:
    """``ProbeBandit`` (UCB1, c = 1) over the 16-entry codebook, dwell = max(20 ms, tau),
    reward = power over the last dwell/2 before each barrier (closed loop, sequential)."""
    dwell = max(BANDIT_DWELL_S, tau)
    period = int(dwell / DT_S)
    assert period * DT_S == dwell
    win_cells = int((dwell / 2) / DT_S)
    assert win_cells * DT_S == dwell / 2
    cb = np.asarray(geom.codebook, dtype=np.int16)
    cls = ProbeBandit if slow_reference else FastProbeBandit
    ctrl = cls(
        cb,
        dwell,
        c=1.0,
        panel=PANEL,
        rx=RX,
        name="probe_bandit",
        min_latency_s=LAT_S,
        latency=LatencyModel(LAT_S),
    )
    setup_controller(ctrl, _caps_obj(geom), SEED)
    arms: list[int] = []
    dt = float(DT_S)
    lat = float(LAT_S)
    n_arm = cb.shape[0]
    st = np.concatenate([cb, np.zeros((1, geom.n), np.int16)])  # row n_arm: initial all-OFF
    table = config_channel_table(geom, st)
    p = np.empty(n_cells_total)
    chunk = max(period, (200_000 // period) * period)
    cur_row = n_arm
    n_cmd = 0
    counts = np.zeros(n_arm, dtype=np.int64)
    rows_of = {cb[a].tobytes(): a for a in range(n_arm - 1, -1, -1)}  # first row wins
    for k0 in range(0, n_cells_total, chunk):
        k1 = min(n_cells_total, k0 + chunk)
        cells = np.arange(k0, k1)
        off, off_a, off_b = _offsets(rule)
        pa = table_power_w(table, traj.x(cells * dt + off), geom.den)
        pa_old = pa if off_a == off else table_power_w(table, traj.x(cells * dt + off_a), geom.den)
        ps = table_power_w(table, traj.x(cells * dt + lat + off_b), geom.den)
        for kb in range(k0, k1, period):
            m = kb // period
            if kb - win_cells >= 0:
                pw = float(np.mean(p[kb - win_cells : kb]))
                pdbm = 10.0 * math.log10(pw / 1e-3) if pw > 0 else -math.inf
            else:
                pdbm = math.nan
            (a,) = ctrl.decide(_obs(m * dwell, 0.0, st[cur_row], pdbm))
            arm = rows_of[np.asarray(a.states, dtype=np.int16).tobytes()]
            i = kb - k0
            e = min(k1, kb + period) - k0
            p[kb] = (lat * pa_old[cur_row, i] + (dt - lat) * ps[arm, i]) / dt
            p[k0 + i + 1 : k0 + e] = pa[arm, i + 1 : e]
            cur_row = arm
            counts[arm] += 1
            n_cmd += 1
            arms.append(arm)
    return EvalResult(
        p, n_cmd, {"dwell_s": str(dwell), "arm_counts": counts.tolist(), "arms": arms}
    )


def _offsets(rule: str) -> tuple[float, float, float]:
    """Evaluation offsets (s) of a whole cell, of the part before a switch and of the part
    after it (from the switch time): 0 for ``left``, half the piece for ``mid``."""
    dt, lat = float(DT_S), float(LAT_S)
    if rule == "left":
        return 0.0, 0.0, 0.0
    if rule == "mid":
        return dt / 2, lat / 2, (dt - lat) / 2
    raise ValueError(f"unknown rule {rule!r}")


def check_fast_bandit(
    geom: Geometry, traj: Trajectory, tau: Fraction, n_cells: int
) -> dict[str, Any]:
    """Run ``ProbeBandit`` and ``FastProbeBandit`` on the same ``n_cells`` and compare the arm
    sequences and the powers (bitwise)."""
    a = run_bandit(geom, traj, tau, n_cells, slow_reference=True)
    b = run_bandit(geom, traj, tau, n_cells)
    return {
        "n_decisions": len(a.extra["arms"]),
        "arms_identical": a.extra["arms"] == b.extra["arms"],
        "powers_bitwise_identical": bool(
            np.array_equal(a.p_cell.view(np.uint64), b.p_cell.view(np.uint64))
        ),
    }


# =========================================================================== statistics


def batch_powers(p_cell: np.ndarray, n_batches: int = N_BATCHES) -> np.ndarray:
    """Per-batch mean power (W) of the 5 ms cells (equal-size, non-overlapping)."""
    k = p_cell.size // n_batches
    return p_cell[: k * n_batches].reshape(n_batches, k).mean(axis=1)


def summarize_power(pb: np.ndarray) -> dict[str, float]:
    """Mean received power with the Student-t 95 % batch-means interval (dBm)."""
    bm = batch_means(pb, pb.size)
    return {
        "p_mean_w": bm.mean,
        "p_mean_dbm": db10(bm.mean / 1e-3),
        "p_ci_low_dbm": db10(bm.ci_low / 1e-3) if bm.ci_low > 0 else -math.inf,
        "p_ci_high_dbm": db10(bm.ci_high / 1e-3),
        "p_half_width_w": bm.half_width,
    }


def mean_ci(v: np.ndarray) -> tuple[float, float, float]:
    """Mean and Student-t 95 % interval of per-batch values (one value per batch)."""
    a = np.asarray(v, dtype=np.float64)
    bm = batch_means(a, a.size)
    return bm.mean, bm.ci_low, bm.ci_high


def tau_half(taus: Sequence[float], rho: Sequence[float]) -> tuple[float, str]:
    """Smallest tau where rho (linear in rho, linear in log10 tau) falls to 0.5.

    Returns ``(tau_half, flag)`` with flag ``ok``, ``left`` (rho(tau_0) < 0.5; value = tau_0)
    or ``right`` (rho >= 0.5 throughout; value = tau_last)."""
    t = np.asarray(taus, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64)
    if r[0] < 0.5:
        return float(t[0]), "left"
    for i in range(len(t) - 1):
        if r[i] >= 0.5 > r[i + 1]:
            lt0, lt1 = math.log10(t[i]), math.log10(t[i + 1])
            f = (r[i] - 0.5) / (r[i] - r[i + 1])
            return 10.0 ** (lt0 + f * (lt1 - lt0)), "ok"
    return float(t[-1]), "right"


# =========================================================================== sweep runner


def _cache_dir(ctx: Any) -> Path:
    c = ctx.params.get("cache_dir")
    if c:
        return Path(str(c))
    return Path(ctx.run_dir).parent.parent / "_ea_cache"  # shared by the E-A sweeps of one root


def run_point(ctx: Any) -> dict[str, Any]:
    """Sweep runner (``experiments.ea:run_point``). Parameters: ``fc_hz``, ``h_m``,
    ``loss_db``, ``controller`` (kind), ``tau_sui_s`` (string; ignored by absent, instant,
    stale), optional ``n_batches``, ``steps_per_batch``, ``cache_dir``, ``regime``."""
    quiet_warnings()
    prm = dict(ctx.params)
    fc, h, loss = float(prm["fc_hz"]), float(prm["h_m"]), float(prm.get("loss_db") or 0.0)
    kind = str(prm["controller"])
    tau = prm.get("tau_sui_s")
    nb = int(prm.get("n_batches", N_BATCHES))
    spb = int(prm.get("steps_per_batch", STEPS_PER_BATCH))
    n_total = nb * spb
    horizon = n_total * float(DT_S) + 10.0
    cache = _cache_dir(ctx)
    geom = build_geometry(fc, h, loss, cache)
    set_table_dir(geom, cache)
    scen = corridor_scenario(fc, h, loss, seed=int(ctx.seed))
    res = resolve(scen)
    run2 = RunRecord(
        Path(ctx.run_dir),
        int(ctx.seed),
        resolved=res,
        mode="batch",
        coupling="barrier",
        backend="numpy_ref+numba(eval)",
        patch_root=Path(ctx.run_dir).parent,
    )
    traj = build_trajectory(scen, max(33_334.0, float(math.ceil(horizon - 10.0))), cache)
    t0 = time.perf_counter()
    rule = str(prm.get("rule", "mid"))
    r = run_controller(kind, geom, traj, tau, n_total, rule=rule)
    wall = time.perf_counter() - t0
    run2.sample_load()
    pb = batch_powers(r.p_cell, nb)
    summ = summarize_power(pb)
    out: dict[str, Any] = {
        "geometry": geom.key,
        "fc_hz": fc,
        "h_m": h,
        "loss_db": loss,
        "n_cells_ris": geom.n,
        "controller": kind,
        "tau_sui_s": None if tau is None else str(tau),
        "n_batches": nb,
        "steps_per_batch": spb,
        "duration_s": n_total * float(DT_S),
        "n_commands": r.n_commands,
        "rule": rule,
        "eval_wall_s": wall,
        "batch_p_w": pb.tolist(),
        "interp_check": geom.check,
        "trajectory_check_max_abs_m": traj.check_max_abs_m,
        **summ,
        **{f"x_{k}": v for k, v in r.extra.items() if k != "arms"},
    }
    rows = [
        {"batch": b, "p_w": float(pb[b]), "p_dbm": db10(float(pb[b]) / 1e-3)} for b in range(nb)
    ]
    write_csv(Path(ctx.run_dir) / "batches.csv", rows)
    # a decimated power trace for figures: 1 s means of the first 600 s
    n_dec = min(r.p_cell.size // 200, 600)
    trace = r.p_cell[: n_dec * 200].reshape(n_dec, 200).mean(axis=1)
    write_csv(
        Path(ctx.run_dir) / "trace_1s.csv",
        [{"t_s": float(i), "p_dbm": db10(float(trace[i]) / 1e-3)} for i in range(n_dec)],
    )
    write_json(Path(ctx.run_dir) / "result.json", out)
    for a in (
        "batch path evaluated in vectorized form (experiments/ea.py): channel sampled on a 1 mm "
        "(0.5 mm above 1.5 GHz) grid and interpolated complex-linearly; see interp_check",
        f"5 ms pieces split at switch times, each evaluated at its {rule} point",
        "controller channel model = scene model on a 1 cm GridChannel (oracle geometry)",
        "command latency 1 ms; switch at t_b + 1 ms (RIS-time convention)",
        f"controller {kind}, tau_SUI {tau}",
    ):
        run2.add_assumption(a)
    run2.oracle_csi_used = kind in ("exact", "wcl", "instant", "stale", "bandit")
    write_meta(
        ctx.run_dir,
        experiment="E-A",
        data_classes=["SIM-sremu"],
        description=(
            f"E-A batch point: {geom.key}, controller {kind}, tau_SUI {tau}; time-averaged "
            "received power over 20 non-overlapping batches of 500 mean movement periods"
        ),
        assumptions=run2.assumptions,
        inputs=run2.inputs,
        files={
            "batches.csv": "per-batch mean received power (W, dBm)",
            "trace_1s.csv": "1 s means of the received power over the first 600 s (dBm)",
            "result.json": "summary (also summary.json via the sweep runner)",
        },
    )
    run2.finish()
    return out


# =========================================================================== aggregation


def _load_points(sweep_roots: Sequence[Path]) -> list[dict[str, Any]]:
    pts = []
    for root in sweep_roots:
        idx = read_json(Path(root) / "index.json")
        for e in idx:
            if e["status"] != "ok":
                raise RuntimeError(f"{root}: point {e['point_id']} failed: {e.get('error')}")
            pts.append(e["summary"])
    return pts


def aggregate(sweep_roots: Sequence[str | Path], out_dir: str | Path) -> dict[str, Any]:
    """Gains over RisAbsent, retention, tau_half and the pre-registered hypotheses.

    Writes ``gains.csv``, ``retention.csv``, ``tau_half.csv``, ``hypotheses.json``,
    ``meta.json`` and ``manifest.json`` into ``out_dir``."""
    roots = [Path(r) for r in sweep_roots]
    pts = _load_points(roots)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = RunRecord(out, SEED, mode="analysis", patch_root=out)
    by_geo: dict[str, list[dict[str, Any]]] = {}
    for p in pts:
        by_geo.setdefault(p["geometry"], []).append(p)
    gains_rows: list[dict[str, Any]] = []
    ret_rows: list[dict[str, Any]] = []
    th_rows: list[dict[str, Any]] = []
    per_geo: dict[str, Any] = {}
    taus = [float(t) for t in TAUS]
    for geo, plist in sorted(by_geo.items()):
        ref = [p for p in plist if p["controller"] == "absent"]
        if not ref:
            continue
        pa = np.asarray(ref[0]["batch_p_w"])
        idx = {(p["controller"], p["tau_sui_s"]): p for p in plist}
        for p in plist:
            pb = np.asarray(p["batch_p_w"])
            g = 10 * np.log10(pb / pa)
            gm, gl, gh = mean_ci(g)
            gains_rows.append(
                {
                    "geometry": geo,
                    "fc_hz": p["fc_hz"],
                    "h_m": p["h_m"],
                    "loss_db": p["loss_db"],
                    "controller": p["controller"],
                    "tau_sui_s": p["tau_sui_s"],
                    "p_mean_dbm": p["p_mean_dbm"],
                    "p_ci_low_dbm": p["p_ci_low_dbm"],
                    "p_ci_high_dbm": p["p_ci_high_dbm"],
                    "gain_db": gm,
                    "gain_ci_low_db": gl,
                    "gain_ci_high_db": gh,
                    "gain_ratio_of_means_db": db10(float(pb.mean() / pa.mean())),
                    "n_commands": p["n_commands"],
                }
            )
        geo_info: dict[str, Any] = {}
        if (
            ("instant", None) in idx
            and ("stale", None) in idx
            and all(("exact", t) in idx for t in TAUS)
        ):
            p0 = np.asarray(idx[("instant", None)]["batch_p_w"])
            ps = np.asarray(idx[("stale", None)]["batch_p_w"])
            ptau = np.stack([np.asarray(idx[("exact", t)]["batch_p_w"]) for t in TAUS])
            rho_full = (ptau.mean(1) - ps.mean()) / (p0.mean() - ps.mean())
            th, flag = tau_half(taus, rho_full)
            rho_b = (ptau - ps[None, :]) / (p0 - ps)[None, :]
            thb = [tau_half(taus, rho_b[:, b]) for b in range(rho_b.shape[1])]
            lth = np.log10([v for v, _ in thb])
            lm, ll, lh = mean_ci(lth)
            flags_b = [f for _, f in thb]
            for i, t in enumerate(TAUS):
                rm, rl, rh = mean_ci(rho_b[i])
                ret_rows.append(
                    {
                        "geometry": geo,
                        "tau_sui_s": t,
                        "rho_full": float(rho_full[i]),
                        "rho_batch_mean": rm,
                        "rho_ci_low": rl,
                        "rho_ci_high": rh,
                        "p_tau_dbm": db10(float(ptau[i].mean()) / 1e-3),
                        "p0_dbm": db10(float(p0.mean()) / 1e-3),
                        "p_stale_dbm": db10(float(ps.mean()) / 1e-3),
                    }
                )
            geo_info = {
                "tau_half_s": th,
                "tau_half_flag": flag,
                "tau_half_ci_low_s": 10**ll,
                "tau_half_ci_high_s": 10**lh,
                "tau_half_batch_geomean_s": 10**lm,
                "n_censored_batches": int(sum(f != "ok" for f in flags_b)),
                "log10_tau_half_batches": lth.tolist(),
                "p0_dbm": db10(float(p0.mean()) / 1e-3),
                "p_stale_dbm": db10(float(ps.mean()) / 1e-3),
                "opt_gain_over_absent_db": db10(float(p0.mean() / pa.mean())),
            }
            th_rows.append(
                {
                    "geometry": geo,
                    **{k: v for k, v in geo_info.items() if k != "log10_tau_half_batches"},
                }
            )
        per_geo[geo] = geo_info
    write_csv(out / "gains.csv", gains_rows)
    write_csv(out / "retention.csv", ret_rows)
    write_csv(out / "tau_half.csv", th_rows)
    hyp = hypotheses(per_geo, by_geo)
    write_json(out / "hypotheses.json", hyp)
    point_assumptions: list[str] = []
    point_inputs: list[Any] = []
    for root in roots:
        for e in read_json(root / "index.json"):
            m = read_json(root / e["dir"] / "meta.json")
            for a in m.get("assumptions", []):
                if a not in point_assumptions and "tau_SUI" not in a:
                    point_assumptions.append(a)
            for i in m.get("inputs", []):
                if i not in point_inputs:
                    point_inputs.append(i)
    checks = {
        p["geometry"]: {
            "interp_max_db_err_all_on": p["interp_check"]["max_db_err_all_on"],
            "trajectory_check_max_abs_m": p["trajectory_check_max_abs_m"],
        }
        for p in pts
    }
    dchk = [p.get("x_decision_check") for p in pts if p.get("x_decision_check")]
    write_meta(
        out,
        experiment="E-A",
        data_classes=["SIM-sremu"],
        description="E-A aggregation: gains over RisAbsent, retention rho(tau), tau_half and "
        "the pre-registered hypotheses H-A1..H-A3 (paper/preregistration/E-A.md)",
        assumptions=[
            "gain CI: Student-t over 20 paired per-batch dB gains (19 d.o.f.)",
            "tau_half CI: Student-t over 20 per-batch log10 tau_half (censored batches clipped "
            "to the sweep range and counted)",
            "evaluation rule per point: " + ", ".join(sorted({str(p.get("rule")) for p in pts})),
            "AP offset d0 = 0.5 m from the user line, on the side away from the RIS (documented "
            "V-WCL deviation; side is an ENG choice)",
            "A1 partition: factor applied to the LOS path only (DESIGN 5.2); RIS hops "
            "unattenuated [ENG]",
            "one RWP trajectory realization (scenario seed 20261010) shared by every point (paired)",
            "ProbeBandit evaluated with FastProbeBandit (bitwise-identical decisions, O(K) means)",
            *point_assumptions,
        ],
        inputs=point_inputs,
        files={
            "gains.csv": "per geometry/controller/tau: absolute power and gain over RisAbsent",
            "retention.csv": "rho(tau) of ExactOptimum per geometry",
            "tau_half.csv": "tau_half per geometry with CI",
            "hypotheses.json": "H-A1, H-A2, H-A3 decisions per the pre-registration",
        },
        extra={
            "sweep_roots": [str(r) for r in roots],
            "checks_per_geometry": checks,
            "decision_table_checks": {
                "points": len(dchk),
                "barriers": int(sum(d["n_barriers"] for d in dchk)),
                "decision_mismatches": int(sum(d["decision_mismatches"] for d in dchk)),
                "index_mismatches": int(sum(d["index_mismatches"] for d in dchk)),
            },
        },
    )
    run.finish()
    return hyp


def _decide(ci: tuple[float, float], lo: float, hi: float) -> str:
    a, b = ci
    if a >= lo and b <= hi:
        return "supported"
    if b < lo or a > hi:
        return "not supported"
    return "inconclusive"


def hypotheses(per_geo: Mapping[str, Any], by_geo: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered decision rules (paper/preregistration/E-A.md section 4)."""
    out: dict[str, Any] = {}
    g1 = geometry_key(1e9, 2, 20)
    g24 = geometry_key(2.4e9, 2, 20)
    if per_geo.get(g1):
        info = per_geo[g1]
        ci = (info["tau_half_ci_low_s"], info["tau_half_ci_high_s"])
        out["H-A1"] = {
            "geometry": g1,
            "tau_half_s": info["tau_half_s"],
            "flag": info["tau_half_flag"],
            "ci_s": ci,
            "n_censored_batches": info["n_censored_batches"],
            "movement_period_over_tau_half": (L_M / (3 * V_MPS)) / info["tau_half_s"],
            "decision": _decide(ci, 0.1, 1.0),
        }
    if per_geo.get(g1) and per_geo.get(g24):
        a, b = per_geo[g1], per_geo[g24]
        lr = np.asarray(b["log10_tau_half_batches"]) - np.asarray(a["log10_tau_half_batches"])
        m, lo, hi = mean_ci(lr)
        ci = (10**lo, 10**hi)
        out["H-A2"] = {
            "geometries": [g1, g24],
            "ratio": b["tau_half_s"] / a["tau_half_s"],
            "ratio_batch_geomean": 10**m,
            "ci": ci,
            "prediction": 1 / 2.4,
            "decision": _decide(ci, 0.30, 0.55),
        }
        for hh in (4,):
            ga, gb = geometry_key(1e9, hh, 20), geometry_key(2.4e9, hh, 20)
            if per_geo.get(ga) and per_geo.get(gb):
                out[f"H-A2_secondary_h{hh}"] = {
                    "ratio": per_geo[gb]["tau_half_s"] / per_geo[ga]["tau_half_s"],
                    "note": "secondary, not decided",
                }
    for geo, plist in by_geo.items():
        idx = {(p["controller"], p["tau_sui_s"]): p for p in plist}
        if not all(("bandit", t) in idx and ("urap", t) in idx for t in ("0.3", "1", "3", "10")):
            continue
        rows = {}
        for t in ("0.3", "1", "3", "10"):
            d = 10 * np.log10(
                np.asarray(idx[("bandit", t)]["batch_p_w"])
                / np.asarray(idx[("urap", t)]["batch_p_w"])
            )
            m, lo, hi = mean_ci(d)
            rows[t] = {"delta_db": m, "ci_db": (lo, hi)}
        if all(r["ci_db"][0] > 0 for r in rows.values()):
            dec = "supported"
        elif any(r["ci_db"][1] < 0 for r in rows.values()):
            dec = "not supported"
        else:
            dec = "inconclusive"
        key = "H-A3" if geo == g1 else f"H-A3_report_{geo}"
        out[key] = {"geometry": geo, "per_tau": rows, "decision": dec}
    return out


def spec_points() -> list[dict[str, Any]]:
    """Every E-A sweep point (geometry x controller x tau), for documentation and tests."""
    geos = [(1e9, h, 0.0, "A0") for h in (1, 2, 4, 6)]
    geos += [(fc, h, 20.0, "A1") for fc in (1e9, 2.4e9) for h in (2, 4)]
    geos += [(fc, 2, lp, "A1s") for fc in (1e9, 2.4e9) for lp in (10.0, 30.0)]
    out = []
    for fc, h, lp, reg in geos:
        out.append({"fc_hz": fc, "h_m": h, "loss_db": lp, "regime": reg, "controller": "absent"})
        for kind in ("instant", "stale"):
            out.append({"fc_hz": fc, "h_m": h, "loss_db": lp, "regime": reg, "controller": kind})
        for kind in ("urap", "wcl", "exact", "bandit"):
            for t in TAUS:
                out.append(
                    {
                        "fc_hz": fc,
                        "h_m": h,
                        "loss_db": lp,
                        "regime": reg,
                        "controller": kind,
                        "tau_sui_s": t,
                    }
                )
    return out


def geometries() -> list[tuple[float, float, float]]:
    """(fc, h, loss) of every E-A geometry."""
    seen = []
    for p in spec_points():
        g = (p["fc_hz"], p["h_m"], p["loss_db"])
        if g not in seen:
            seen.append(g)
    return seen


def describe() -> str:
    """One-line description (used by drivers)."""
    return json.dumps({"n_points": len(spec_points()), "n_geometries": len(geometries())})


# =========================================================================== run_batch check


def check_against_run_batch(
    geom: Geometry,
    traj: Trajectory,
    kind: str,
    tau_s: str | None,
    duration_s: str = "60",
) -> dict[str, Any]:
    """Mean received power of this module's evaluator vs ``sremu.engine.batch.run_batch``
    (5 ms grid, the same controller class, seed and trajectory) over ``[0, duration_s)``.
    Returns both values in dBm and the difference in dB."""
    from sremu.engine.batch import run_batch

    quiet_warnings()
    dur = Fraction(str(duration_s))
    n = int(dur / DT_S)
    mine = run_controller(kind, geom, traj, tau_s, n, rule="left")
    p_mine = float(mine.p_cell.mean())
    model = from_scenario(geom.scenario, backend="numpy_ref")
    tau = None if tau_s is None else Fraction(str(tau_s))
    if kind == "absent":
        ctrl: Any = RisAbsent(
            panel=PANEL, period_s=dur, min_latency_s=LAT_S, latency=LatencyModel(LAT_S)
        )
    elif kind in ("exact", "wcl"):
        ctrl = DelayedController(make_inner(kind, geom), tau)
    elif kind == "urap":
        ctrl = Urap(
            URAP_P,
            panel=PANEL,
            name="urap",
            period_s=tau,
            min_latency_s=LAT_S,
            latency=LatencyModel(LAT_S),
        )
    elif kind == "bandit":
        dwell = max(BANDIT_DWELL_S, tau)
        ctrl = ProbeBandit(
            np.asarray(geom.codebook, dtype=np.int16),
            dwell,
            c=1.0,
            panel=PANEL,
            rx=RX,
            name="probe_bandit",
            min_latency_s=LAT_S,
            latency=LatencyModel(LAT_S),
        )
    else:
        raise ValueError(kind)
    t0 = time.perf_counter()
    rb = run_batch(model, dur, dt_s=DT_S, controllers=[ctrl], seed=SEED)
    wall = time.perf_counter() - t0
    p_rb = float(rb.mean_rx_power_w[RX])
    return {
        "controller": kind,
        "tau_sui_s": tau_s,
        "duration_s": str(dur),
        "p_vectorized_dbm": db10(p_mine / 1e-3),
        "p_run_batch_dbm": db10(p_rb / 1e-3),
        "diff_db": db10(p_mine / p_rb),
        "run_batch_pieces": len(rb.t),
        "run_batch_wall_s": wall,
        "n_commands_vectorized": mine.n_commands,
        "n_commands_run_batch": sum(
            1 for r in rb.records if r.status in ("applied", "late_applied")
        ),
    }


def prebuild(fc_hz: float, h_m: float, loss_db: float, cache_dir: str | Path) -> str:
    """Geometry tables and the ExactOptimum/WclHeuristic decision tables of one geometry."""
    geom = build_geometry(fc_hz, h_m, loss_db, cache_dir)
    set_table_dir(geom, cache_dir)
    for kind in ("exact", "wcl"):
        decision_table(kind, geom)
    return geom.key
