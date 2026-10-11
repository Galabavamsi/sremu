"""V-WCL: reproduction checks of Mishra & Kherani, IEEE WCL 14(4), 2025 (DESIGN.md 10.11).

Scenario as published (letter Sec. IV-A): AP at the corridor origin, MU on a line of
length l = 10 m (1-D random waypoint, v = 1 m/s, no pause), a RIS line parallel to it at
distance h with N = floor(l/k) = 133 cells of w = k = lambda/4 at f_c = 1 GHz, P_T = 0 dBm,
G = 1, the ``on_off`` alphabet (OFF absorbing). sremu frame: MU segment from (0,0,0) to
(l,0,0); RIS cells at ((n + 1/2) w, h, 0), n = 0..N-1, normal -y, u axis +x.

**Documented deviation** (DESIGN 10.11): the AP is offset by d0 = 0.5 m perpendicular to
the user line, here along +z (AP at (0, 0, d0)); with the AP on the line the mean LOS power
diverges logarithmically as x -> 0. Sensitivity: d0 in {0.3, 1.0} m, and the AP on the
line (d0 = 0) with the MU segment starting at a cutoff c in {0.05, 0.1, 0.2} m (the MU then
moves on [c, l]; the emulator cannot truncate otherwise). [ENG choices: the +z direction
of the offset; the cutoff as segment start.]

Parts (each writes one run directory under ``validation/wcl/results``):

* **W1** (``w1``): the scenario's RWP trajectory over 10^4 movement periods: stationary
  pdf (KS vs F(x) = 3x^2/l^2 - 2x^3/l^3), uniform waypoints (KS), mean movement length
  l/3, maximum step; ``legacy_wcl_eq2`` teeth: per-element power ratio to sremu's K1
  cascade vs ``(pi lambda)^2 (cos_i cos_s)^(2q-1)`` within 1e-9.
* **W3** (``w3``): URAP p = 0.2 at h in {1, 2, 4, 6} m. Emulator time-averaged tone power vs
  the analytic integral of E[P(x)] f_X(x) dx in both forms: the letter's Eq. (9) (power of
  the mean field) and the mean power (adds p(1-p) sum |S_n|^2). Heavy part: the
  coefficient path (combine(snapshot(t), cfg(t)) on a 5 ms grid; E-A batch settings) over
  10^4 movement periods, 20 non-overlapping batches; the URAP decisions come from the
  ``Urap`` controller itself (same keyed generator as in the engine). IQ subset: the
  lockstep IQ engine (10 kS/s tone, noise off, the in-process ``Urap`` barrier controller)
  over 1000 s per h, paired with the coefficient path on the same trajectory and draws.
  Sensitivity runs over the AP offset (10^3 movement periods each).
* **W5/W6** (``w5``): tau_SUI in {0.01, 0.1, 0.3, 1, 3, 10} s at h = 4 m (the letter's h_L)
  with ``ExactOptimum`` and ``WclHeuristic(4, 3)`` in ``DelayedController`` (command
  latency 0): IQ (noise off) vs a position-only semi-analytic oracle on the paired
  trajectory (W5, <= 0.02 dB; non-increasing in tau within CI); W6: the IQ (sample-level
  geometric Doppler) vs the position-snapshot average of ``run_batch`` (5 ms) over the
  first 300 s (<= 0.01 dB).

The **oracle** is independent of ``sremu.physics``: closed-form K1 point kernel of the
corridor (j A sqrt(cos_i cos_s)/(4 pi r_i r_s) exp(-jk(r_i + r_s)), A = w^2), LOS
lambda/(4 pi d) exp(-jkd), its own exact binary optimum (critical-angle enumeration) and its
own piecewise-linear evaluation of the trajectory from the scenario's waypoints.

Usage::

    python validation/wcl/vwcl.py w1|w3|w5|all [--jobs 12] [--fast | --smoke]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import math
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import vstudy as vs  # noqa: E402

from sremu.engine.emulator import EngineSink  # noqa: E402
from sremu.units import C0  # noqa: E402

STUDY_DIR = _HERE
FC = 1.0e9
L_M = 10.0
V_MPS = 1.0
P_URAP = 0.2
D0_M = 0.5
Q_LETTER = 0.285
H_LIST = (1.0, 2.0, 4.0, 6.0)
SEED = 20261010
LINK = "ap.tx0->mu.rx0"
RX = "mu.rx0"
PANEL = "ris0"
RATE = Fraction(10_000)  # IQ tone rate (E-A IQ subset)
URAP_PERIOD = Fraction(1, 10)  # [ENG] URAP redraw period
GEOM_DT = Fraction(1, 200)  # 5 ms geometry step of the coefficient path (E-A batch)
MOVE_PERIOD = Fraction(10, 3)  # l/(3v), s
N_PERIODS_MAIN = 10_000
TAUS = ("1/100", "1/10", "3/10", "1", "3", "10")
SENSITIVITY = (  # (label, d0_m, segment start c_m)
    ("d0_0.3", 0.3, 0.0),
    ("d0_1.0", 1.0, 0.0),
    ("online_c0.05", 0.0, 0.05),
    ("online_c0.1", 0.0, 0.1),
    ("online_c0.2", 0.0, 0.2),
)

ASSUMPTIONS = (
    "V-WCL documented deviation: AP offset d0 = 0.5 m perpendicular to the user line "
    "(+z), not on the line as in the letter (mean LOS power diverges as x -> 0) [ENG]",
    "URAP redraw period 0.1 s [ENG; the letter does not state it; the time average does "
    "not depend on it]",
    "Coefficient path: left-point pieces on a 5 ms geometry grid (E-A batch setting)",
    "IQ path: 10 kS/s unit tone (P_T = 0 dBm), noise off, lockstep, numba_cpu",
)


# =========================================================================== scenario


def cells(fc: float = FC, l_m: float = L_M) -> tuple[int, float]:
    """(N, w): cells of w = lambda/4 on the RIS line of length l."""
    w = C0 / fc / 4.0
    return math.floor(l_m / w), w


def corridor_scenario(
    h_m: float,
    *,
    d0_m: float = D0_M,
    seg_start_m: float = 0.0,
    fc: float = FC,
    seed: int = SEED,
    duration_s: str = "100",
    controllers: Sequence[Mapping[str, Any]] = (),
    name: str = "vwcl_corridor",
) -> dict[str, Any]:
    """The WCL corridor (module docstring) as a section 9.2 scenario mapping."""
    n, w = cells(fc)
    return {
        "schema_version": "1.0",
        "name": name,
        "seed": int(seed),
        "description": (
            f"WCL letter corridor: h = {h_m} m, AP at (0, 0, {d0_m}) m, MU rwp_1d on "
            f"[{seg_start_m}, {L_M}] m at {V_MPS} m/s, N = {n} cells of lambda/4 at {fc:g} Hz"
        ),
        "numerics": {"backend": "numba_cpu"},
        "nodes": [
            {
                "id": "ap",
                "position_m": [0.0, 0.0, float(d0_m)],
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": "tx0",
                        "direction": "tx",
                        "fc_hz": fc,
                        "rate_sps": "1e4",
                        "occupied_bw_hz": 100.0,
                        "full_scale_dbm": 0.0,
                    }
                ],
            },
            {
                "id": "mu",
                "trajectory": {
                    "model": "rwp_1d",
                    "p0_m": [float(seg_start_m), 0.0, 0.0],
                    "p1_m": [L_M, 0.0, 0.0],
                    "speed_mps": V_MPS,
                },
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": "rx0",
                        "direction": "rx",
                        "fc_hz": fc,
                        "rate_sps": "1e4",
                        "noise_figure_db": 0.0,
                    }
                ],
            },
        ],
        "ris": [
            {
                "id": PANEL,
                "pose": {
                    "center_m": [n * w / 2.0, float(h_m), 0.0],
                    "normal": [0, -1, 0],
                    "u_axis": [1, 0, 0],
                },
                "grid": {"n_u": n, "n_v": 1, "pitch_u_m": w, "pitch_v_m": w, "cell_model": "auto"},
                "states": {"library": "on_off"},
                "initial": {"fill": 0},
            }
        ],
        "controllers": [dict(c) for c in controllers],
        "run": {"mode": "lockstep", "coupling": "barrier", "duration_s": duration_s},
        "traffic": [
            {"stream": "ap.tx0", "generator": "tone", "params": {"freq_hz": 0.0, "amplitude": 1.0}}
        ],
    }


def urap_spec(p: float = P_URAP) -> dict[str, Any]:
    """Scenario controller entry of the URAP baseline."""
    return {
        "name": "urap",
        "kind": "builtin",
        "impl": "Urap",
        "period_s": f"{URAP_PERIOD.numerator}/{URAP_PERIOD.denominator}",
        "min_latency_s": "0",
        "latency": {"fixed_s": "0"},
        "params": {"p": p},
    }


def delayed_spec(inner: str, tau_s: str) -> dict[str, Any]:
    """``DelayedController(inner, tau_SUI, report_delay = 0)``, command latency 0."""
    params: dict[str, Any] = {"link": LINK}
    if inner == "WclHeuristic":
        params.update({"L": 4, "M": 3})
    return {
        "name": inner.lower(),
        "kind": "builtin",
        "impl": "DelayedController",
        "period_s": tau_s,
        "min_latency_s": "0",
        "latency": {"fixed_s": "0"},
        "params": {
            "inner": {"impl": inner, "params": params},
            "tau_sui_s": tau_s,
            "report_delay_s": "0",
        },
    }


def build_model(scen: Mapping[str, Any]) -> Any:
    """The full sremu model of a scenario mapping (``from_scenario``, D18)."""
    from sremu.physics.propagation import from_scenario

    res = vs.resolve_mapping(scen)
    return from_scenario(res.scenario.model_dump(mode="python"))


# =========================================================================== oracle


def oracle_channel(
    x_m: np.ndarray, h_m: float, d0_m: float, fc: float = FC
) -> tuple[np.ndarray, np.ndarray]:
    """Independent closed form: (h_LOS (X,), S (X, N)) with carrier phases, isotropic
    antennas, Gamma_ON = 1, A = w^2, K1 point kernel with sqrt(cos_i cos_s) obliquity and
    the PO factor j. MU at (x, 0, 0), AP at (0, 0, d0), cells at ((n + 1/2) w, h, 0)."""
    n, w = cells(fc)
    lam = C0 / fc
    k = 2.0 * math.pi / lam
    x = np.asarray(x_m, dtype=np.float64).reshape(-1)
    y = (np.arange(n, dtype=np.float64) + 0.5) * w
    ri = np.sqrt(y * y + h_m * h_m + d0_m * d0_m)
    rs = np.sqrt((x[:, None] - y[None, :]) ** 2 + h_m * h_m)
    ci = h_m / ri
    cs = h_m / rs
    s = 1j * (w * w) * np.sqrt(ci[None, :] * cs) / (4.0 * math.pi * ri[None, :] * rs)
    s = s * np.exp(-1j * k * (ri[None, :] + rs))
    dl = np.sqrt(x * x + d0_m * d0_m)
    h0 = lam / (4.0 * math.pi * dl) * np.exp(-1j * k * dl)
    return h0, s


def exact_binary(z0: complex, z: np.ndarray) -> tuple[float, np.ndarray]:
    """Independent exact maximiser of |z0 + sum_n b_n z_n| over b in {0,1}^N (ON/OFF):
    the optimal set is {n : Re(z_n e^{-j theta}) > 0} for theta = arg(total); the set only
    changes at arg(z_n) +- pi/2, so every candidate is a midpoint between consecutive
    critical angles (plus the empty set)."""
    zz = np.asarray(z, dtype=np.complex128)
    crit = np.sort(
        np.mod(np.concatenate([np.angle(zz) + np.pi / 2, np.angle(zz) - np.pi / 2]), 2 * np.pi)
    )
    mids = (crit + np.roll(crit, -1)) / 2.0
    mids[-1] = mids[-1] + np.pi  # wrap-around midpoint
    proj = np.real(zz[None, :] * np.exp(-1j * mids[:, None])) > 0.0
    vals = np.abs(z0 + proj.astype(np.float64) @ zz)
    i = int(np.argmax(vals))
    if abs(z0) >= vals[i]:
        return float(abs(z0)), np.zeros(zz.size, dtype=bool)
    return float(vals[i]), proj[i]


def rwp_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Stationary pdf of 1-D RWP on [a, b] (letter Eq. 1): 6u(L-u)/L^3, u = x - a."""
    lp = b - a
    u = np.asarray(x, dtype=np.float64) - a
    return 6.0 * u * (lp - u) / lp**3


def analytic_averages(
    h_m: float,
    d0_m: float,
    seg_start_m: float = 0.0,
    *,
    p: float = P_URAP,
    n_quad: int = 4000,
    with_opt: bool = True,
) -> dict[str, float]:
    """Integrals over the stationary pdf (Gauss-Legendre, ``n_quad`` nodes) of: LOS power,
    URAP mean field |h0 + p sum S|^2 (letter Eq. 9), URAP mean power (+ p(1-p) sum|S|^2),
    all-ON power and the exact binary optimum (mW at P_T = 0 dBm)."""
    a, b = float(seg_start_m), L_M
    t, wq = np.polynomial.legendre.leggauss(int(n_quad))
    x = a + (b - a) * (t + 1.0) / 2.0
    wt = wq * (b - a) / 2.0 * rwp_pdf(x, a, b)
    h0, s = oracle_channel(x, h_m, d0_m)
    ssum = s.sum(axis=1)
    mf = np.abs(h0 + p * ssum) ** 2
    mp = mf + p * (1.0 - p) * np.sum(np.abs(s) ** 2, axis=1)
    out = {
        "los": float(np.dot(wt, np.abs(h0) ** 2)),
        "urap_mean_field_eq9": float(np.dot(wt, mf)),
        "urap_mean_power": float(np.dot(wt, mp)),
        "all_on": float(np.dot(wt, np.abs(h0 + ssum) ** 2)),
        "weight_sum": float(np.sum(wt)),
    }
    if with_opt:
        po = np.array([exact_binary(h0[i], s[i])[0] ** 2 for i in range(x.size)])
        out["opt"] = float(np.dot(wt, po))
    return out


def stale_limit(h_m: float, d0_m: float, n_quad: int = 600) -> float:
    """tau_SUI -> infinity limit: the optimum for an independent position x' ~ f_X applied
    at x ~ f_X (double integral, Gauss-Legendre)."""
    t, wq = np.polynomial.legendre.leggauss(int(n_quad))
    x = L_M * (t + 1.0) / 2.0
    wt = wq * L_M / 2.0 * rwp_pdf(x, 0.0, L_M)
    h0, s = oracle_channel(x, h_m, d0_m)
    b = np.array([exact_binary(h0[i], s[i])[1] for i in range(x.size)], dtype=np.float64)
    p = np.abs(h0[:, None] + s @ b.T) ** 2  # p[i, j]: at x_i with the optimum of x_j
    return float(wt @ p @ wt)


# =========================================================================== trajectory


def mu_trajectory(model: Any) -> Any:
    """The MU trajectory object of a model (public ``nodes`` mapping)."""
    return model.nodes["mu"].trajectory


def waypoint_table(traj: Any, horizon_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(leg start times (K+1,), waypoint abscissae (K+2,)) covering [0, horizon_s]: leg i
    runs from w_i at t_i to w_{i+1} at t_{i+1} (no pause)."""
    starts = np.asarray(traj.leg_starts(horizon_s + 10.0), dtype=np.float64)
    wps = np.array([traj.waypoint(i)[0] for i in range(starts.size + 1)], dtype=np.float64)
    return starts, wps


def positions_x(
    t_s: np.ndarray, starts: np.ndarray, wps: np.ndarray, v: float = V_MPS
) -> np.ndarray:
    """Independent piecewise-linear evaluation of the MU abscissa at times ``t_s``."""
    t = np.asarray(t_s, dtype=np.float64)
    i = np.searchsorted(starts, t, side="right") - 1
    i = np.clip(i, 0, starts.size - 1)
    a = wps[i]
    b = wps[i + 1]
    dur = np.abs(b - a) / v
    frac = np.where(dur > 0, np.clip((t - starts[i]) / np.where(dur > 0, dur, 1.0), 0.0, 1.0), 1.0)
    return a + frac * (b - a)


# =========================================================================== controllers


def urap_states(model: Any, seed: int, m_end: int, m_start: int = 0) -> np.ndarray:
    """(m_end - m_start, N) int8 states: the decisions of the ``Urap`` controller at
    barriers m_start..m_end-1, made by the controller itself with its keyed generator
    (``ctrl/urap``), exactly as the engine's barrier manager sets it up. The controller is
    stepped from barrier 0 (its generator is sequential); only the rows from ``m_start``
    are kept (memory: a late W3 chunk would otherwise hold ~3e5 x 133 rows)."""
    from sremu.control.builtin import build_controller
    from sremu.control.controller import make_observation, setup_controller

    ctrl = build_controller(urap_spec(), model=model)
    setup_controller(ctrl, {PANEL: model.capabilities(PANEL)}, seed)
    n = model.panel_geometry(PANEL).n_cells
    out = np.zeros((max(m_end - m_start, 0), n), dtype=np.int8)
    prev = np.zeros(n, dtype=np.int16)
    for m in range(m_end):
        obs = make_observation(m * URAP_PERIOD, ris_state={PANEL: prev})
        (act,) = ctrl.decide(obs)
        prev = np.asarray(act.states, dtype=np.int16)
        if m >= m_start:
            out[m - m_start] = prev
    return out


# =========================================================================== coefficient path


def coefficient_power(
    model: Any, i0: int, i1: int, dt: Fraction, states_of: Any, *, opt_every: int = 0
) -> dict[str, Any]:
    """Coefficient path on grid points t_i = i dt, i0 <= i < i1 (left-point pieces of equal
    weight; equal to ``run_batch`` when every switch lies on the grid):
    ``h = h_direct + sum_n cascade[s_n, n]`` from ``model.csi_view(model.snapshot(t))``
    with the configuration ``states_of(t)`` ((N,) int states) in effect at t. Returns the
    per-point |h|^2 and LOS-only |h_direct|^2 and, with ``opt_every = k > 0``, the exact
    ON/OFF optimum's |h|^2 on every k-th point (:func:`exact_binary` on the engine's CSI)."""
    n = i1 - i0
    p = np.empty(n)
    plos = np.empty(n)
    popt: list[float] = []
    idx = None
    for j in range(n):
        t = (i0 + j) * dt
        snap = model.snapshot(t)
        v = model.csi_view(snap, LINK)
        z = v.cascade[PANEL]
        if idx is None:
            idx = np.arange(z.shape[1])
        s = states_of(t)
        h = v.h_direct + complex(z[s, idx].sum())
        p[j] = h.real * h.real + h.imag * h.imag
        plos[j] = abs(v.h_direct) ** 2
        if opt_every and (i0 + j) % opt_every == 0:
            popt.append(exact_binary(v.h_direct, z[1])[0] ** 2)
    out: dict[str, Any] = {"p": p, "p_los": plos}
    if opt_every:
        out["p_opt"] = np.asarray(popt)
    return out


def piecewise_power(
    model: Any,
    t_end: Fraction,
    dt: Fraction,
    switches: Sequence[tuple[Fraction, np.ndarray]],
    *,
    mode: str = "left",
    n_cells_: int | None = None,
) -> dict[str, Any]:
    """Time average of |h|^2 over [0, t_end) with ``run_batch`` piece semantics: pieces
    between consecutive points of {i dt} union {switch times}, weighted by duration, the
    configuration in effect at the piece start (last switch at or before it; all OFF
    before the first) and the channel at the piece's left point (``mode="left"``, as the
    batch path) or midpoint (``mode="mid"``). ``switches``: (RIS time, (N,) states)."""
    import bisect

    n = n_cells_ or model.panel_geometry(PANEL).n_cells
    sw_t = [t for t, _ in switches]
    sw_s = [np.asarray(s, dtype=np.int64) for _, s in switches]
    off = np.zeros(n, dtype=np.int64)
    pts = {i * dt for i in range(math.ceil(t_end / dt))}
    pts.update(t for t in sw_t if 0 < t < t_end)
    bps = sorted(pts)
    bps.append(t_end)
    idx = np.arange(n)
    acc = 0.0
    tot = Fraction(0)
    for a, b in itertools.pairwise(bps):
        k = bisect.bisect_right(sw_t, a) - 1
        s = off if k < 0 else sw_s[k]
        te = a if mode == "left" else (a + b) / 2
        v = model.csi_view(model.snapshot(te), LINK)
        h = v.h_direct + complex(v.cascade[PANEL][s, idx].sum())
        w = b - a
        acc += float(w) * (h.real * h.real + h.imag * h.imag)
        tot += w
    return {"mean_mw": acc / float(tot), "n_pieces": len(bps) - 1}


def grid_range(t_lo: Fraction, t_hi: Fraction, dt: Fraction) -> tuple[int, int]:
    """Grid indices i with t_lo <= i dt < t_hi."""
    return math.ceil(t_lo / dt), math.ceil(t_hi / dt)


# =========================================================================== sweep runners


OPT_EVERY = 20  # exact optimum on every 20th 5 ms point (0.1 s) of the W3 chunks


def run_w3_chunk(ctx: Any) -> dict[str, Any]:
    """Sweep runner: one batch (time chunk) of the W3 coefficient path."""
    p = dict(ctx.params)
    scen = corridor_scenario(
        p["h_m"],
        d0_m=p["d0_m"],
        seg_start_m=p["seg_start_m"],
        seed=int(p["seed"]),
        duration_s=str(p["T_s"]),
    )
    model = build_model(scen)
    t_total = Fraction(p["T_s"])
    k = int(p["n_chunks"])
    j = int(p["chunk"])
    i0, i1 = grid_range(t_total * j / k, t_total * (j + 1) / k, GEOM_DT)
    m_end = math.floor((i1 * GEOM_DT) / URAP_PERIOD) + 1
    m0 = math.floor((i0 * GEOM_DT) / URAP_PERIOD)
    t0 = time.perf_counter()
    sched = urap_states(model, int(p["seed"]), m_end, m_start=m0)
    t1 = time.perf_counter()

    def states_of(t: Fraction) -> np.ndarray:
        return sched[math.floor(t / URAP_PERIOD) - m0].astype(np.intp)

    r = coefficient_power(
        model, i0, i1, GEOM_DT, states_of, opt_every=OPT_EVERY if p.get("with_opt") else 0
    )
    t2 = time.perf_counter()
    out = {
        "h_m": p["h_m"],
        "d0_m": p["d0_m"],
        "seg_start_m": p["seg_start_m"],
        "chunk": j,
        "i0": i0,
        "i1": i1,
        "n": int(i1 - i0),
        "sum_p": float(np.sum(r["p"])),
        "sum_p_los": float(np.sum(r["p_los"])),
        "wall_urap_s": t1 - t0,
        "wall_coeff_s": t2 - t1,
    }
    if "p_opt" in r:
        out["sum_p_opt"] = float(np.sum(r["p_opt"]))
        out["n_opt"] = int(r["p_opt"].size)
    return out


def run_w3_cond(ctx: Any) -> dict[str, Any]:
    """Sweep runner: the trajectory-conditional oracle of one W3 batch. On the same 5 ms
    grid and the same MU trajectory as the matching ``w3_chunk`` point, the independent
    closed-form channel gives the conditional expectations E[P | x(t)] of URAP in both forms
    (mean power and the letter's Eq. 9, power of the mean field) and the LOS power, plus the
    realised URAP power with the ``Urap`` controller's own decisions (an independent replica
    of the emulator's batch path). Comparing the emulator with E[P | x(t)] on the same
    trajectory removes the mobility sampling noise that dominates the stationary comparison
    (only the URAP draws remain)."""
    p = dict(ctx.params)
    h = float(p["h_m"])
    d0 = float(p["d0_m"])
    scen = corridor_scenario(
        h,
        d0_m=d0,
        seg_start_m=p["seg_start_m"],
        seed=int(p["seed"]),
        duration_s=str(p["T_s"]),
    )
    model = build_model(scen)
    t_total = Fraction(p["T_s"])
    k = int(p["n_chunks"])
    j = int(p["chunk"])
    i0, i1 = grid_range(t_total * j / k, t_total * (j + 1) / k, GEOM_DT)
    m_end = math.floor((i1 * GEOM_DT) / URAP_PERIOD) + 1
    m0 = math.floor((i0 * GEOM_DT) / URAP_PERIOD)
    t0 = time.perf_counter()
    starts, wps = waypoint_table(mu_trajectory(model), float(i1 * GEOM_DT))
    sched = urap_states(model, int(p["seed"]), m_end, m_start=m0)
    ratio = GEOM_DT / URAP_PERIOD  # exact Fraction: barrier index m = floor(i * ratio)
    sums = {"mp": 0.0, "mf": 0.0, "los": 0.0, "real": 0.0}
    blk = 20_000
    for a in range(i0, i1, blk):
        idx = np.arange(a, min(i1, a + blk), dtype=np.int64)
        x = positions_x(idx * float(GEOM_DT), starts, wps)
        h0, sc = oracle_channel(x, h, d0)
        ssum = sc.sum(axis=1)
        mf = np.abs(h0 + P_URAP * ssum) ** 2
        mp = mf + P_URAP * (1.0 - P_URAP) * np.sum(np.abs(sc) ** 2, axis=1)
        m = (idx * ratio.numerator) // ratio.denominator
        cfg = sched[m - m0].astype(np.float64)
        real = np.abs(h0 + np.einsum("ij,ij->i", sc, cfg)) ** 2
        sums["mp"] += float(mp.sum())
        sums["mf"] += float(mf.sum())
        sums["los"] += float(np.sum(np.abs(h0) ** 2))
        sums["real"] += float(real.sum())
    return {
        "h_m": h,
        "d0_m": d0,
        "seg_start_m": p["seg_start_m"],
        "chunk": j,
        "i0": i0,
        "i1": i1,
        "n": int(i1 - i0),
        "sum_cond_mean_power": sums["mp"],
        "sum_cond_mean_field_eq9": sums["mf"],
        "sum_cond_los": sums["los"],
        "sum_oracle_realised": sums["real"],
        "wall_s": time.perf_counter() - t0,
    }


class _BinSink(EngineSink):
    """EngineSink: per-bin sums of |d|^2 (mW) of the RX stream and the applied configs."""

    def __init__(self, bin_samples: int) -> None:
        self.b = int(bin_samples)
        self.sums = np.zeros(0)
        self.counts = np.zeros(0, dtype=np.int64)
        self.applied: list[tuple[Fraction, np.ndarray, bool]] = []
        self.lateness: list[int] = []

    def on_rx(self, stream: str, k0: int, d: np.ndarray) -> None:
        if stream != RX:
            return
        x = np.asarray(d).astype(np.complex128)
        pw = x.real * x.real + x.imag * x.imag
        k = k0 + np.arange(pw.size)
        bins = k // self.b
        top = int(bins[-1]) + 1
        if top > self.sums.size:
            self.sums = np.concatenate([self.sums, np.zeros(top - self.sums.size)])
            self.counts = np.concatenate([self.counts, np.zeros(top - self.counts.size, np.int64)])
        np.add.at(self.sums, bins, pw)
        np.add.at(self.counts, bins, 1)

    def on_record(self, rec: Any, cfg: Any) -> None:
        if rec.status in ("applied", "late_applied") and cfg is not None:
            self.applied.append((rec.t_applied, np.asarray(cfg.state).copy(), bool(cfg.enabled)))
            self.lateness.extend(int(v) for v in rec.lateness_samples.values())


def run_iq_engine(scen: Mapping[str, Any], duration: Fraction) -> dict[str, Any]:
    """Lockstep IQ run of a corridor scenario (noise off, every RX page computed) with its
    in-process controllers; returns per-second power bins, applied configs and run facts."""
    from sremu.control.builtin import build_controller
    from sremu.engine.emulator import Emulator

    res = vs.resolve_mapping(scen)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = Emulator.from_scenario(res, noise=False, always_active=True)
    sink = _BinSink(int(RATE))
    eng.add_sink(sink)
    d = res.scenario.model_dump(mode="python")
    for c in d.get("controllers") or []:
        eng.register_controller(build_controller(c, model=eng.model))
    t1 = time.perf_counter()
    summ = eng.run(duration)
    t2 = time.perf_counter()
    info = eng.manifest_info()
    n_bins = int(duration * RATE) // int(RATE)
    return {
        "sums": sink.sums[:n_bins].copy(),
        "counts": sink.counts[:n_bins].copy(),
        "applied": sink.applied,
        "lateness_max": max(sink.lateness, default=0),
        "lateness_nonzero": int(sum(1 for v in sink.lateness if v != 0)),
        "n_records": len(eng.records),
        "build_s": t1 - t0,
        "run_s": t2 - t1,
        "rtf": float(duration) / (t2 - t1),
        "verify_interp_max_rel": summ.verify_interp_max_rel,
        "counters": dict(summ.counters),
        "oracle_csi_used": summ.oracle_csi_used,
        "t_u_s": info["snapshot_policy"].get("t_u_s_used"),
        "tiles": info["snapshot_policy"].get("tiles"),
        "link_decision": info["link_decisions"].get(LINK),
        "nondeterministic_inputs": summ.nondeterministic_inputs,
    }


def _switches(iq: Mapping[str, Any]) -> list[tuple[Fraction, np.ndarray]]:
    return [(t, np.asarray(s)) for t, s, _en in iq["applied"]]


def w6_rows(
    scen: Mapping[str, Any],
    model: Any,
    iq: Mapping[str, Any],
    dur: Fraction,
    batch_s: Fraction | None,
) -> dict[str, Any]:
    """W6: the IQ mean (sample-level geometric Doppler) vs position-snapshot averages with
    ``run_batch`` piece semantics (left-point holds) on the engine-recorded configurations,
    at the batch path's default grid (the auto T_u), 5 ms (E-A) and 1 ms, over the whole
    run; plus ``run_batch`` itself (its own controller, 5 ms) over [0, batch_s)."""
    from sremu.control.builtin import build_controller
    from sremu.engine.batch import run_batch

    iq_mean = float(iq["sums"].sum() / iq["counts"].sum())
    sw = _switches(iq)
    rows = []
    grids = (
        ("auto_T_u", Fraction(iq["t_u_s"])),
        ("5ms", Fraction(1, 200)),
        ("1ms", Fraction(1, 1000)),
    )
    for label, dt in grids:
        t0 = time.perf_counter()
        r = piecewise_power(model, dur, dt, sw, mode="left")
        rows.append(
            {
                "grid": label,
                "dt_s": dt,
                "snapshot_mean_mw": r["mean_mw"],
                "iq_mean_mw": iq_mean,
                "delta_db": vs.db10(iq_mean) - vs.db10(r["mean_mw"]),
                "n_pieces": r["n_pieces"],
                "wall_s": time.perf_counter() - t0,
            }
        )
    out: dict[str, Any] = {"T_s": str(dur), "rows": rows}
    if batch_s:
        res = vs.resolve_mapping(scen)
        d = res.scenario.model_dump(mode="python")
        bmodel = build_model(scen)
        ctrls = [build_controller(c, model=bmodel) for c in d.get("controllers") or []]
        t0 = time.perf_counter()
        br = run_batch(bmodel, batch_s, dt_s=GEOM_DT, controllers=ctrls, seed=int(d["seed"]))
        t1 = time.perf_counter()
        pw = piecewise_power(model, batch_s, GEOM_DT, sw, mode="left")
        nb = int(batch_s)
        iq_b = float(iq["sums"][:nb].sum() / iq["counts"][:nb].sum())
        out["run_batch"] = {
            "T_s": str(batch_s),
            "batch_mean_mw": float(br.mean_h2[LINK]),
            "piecewise_5ms_mean_mw": pw["mean_mw"],
            "iq_mean_mw": iq_b,
            "batch_vs_piecewise_rel": float(br.mean_h2[LINK]) / pw["mean_mw"] - 1.0,
            "delta_iq_vs_batch_db": vs.db10(iq_b) - vs.db10(float(br.mean_h2[LINK])),
            "batch_s": t1 - t0,
            "batch_pieces": len(br.t),
        }
    return out


def run_w3_iq(ctx: Any) -> dict[str, Any]:
    """Sweep runner: W3 IQ subset at one h (URAP), paired with the coefficient path."""
    p = dict(ctx.params)
    dur = Fraction(p["T_s"])
    scen = corridor_scenario(
        p["h_m"],
        seed=int(p["seed"]),
        duration_s=str(p["T_s"]),
        controllers=[urap_spec()],
        name="vwcl_w3_iq",
    )
    iq = run_iq_engine(scen, dur)
    model = build_model(scen)
    i0, i1 = grid_range(Fraction(0), dur, GEOM_DT)
    m_end = math.floor((i1 * GEOM_DT) / URAP_PERIOD) + 1
    t0 = time.perf_counter()
    sched = urap_states(model, int(p["seed"]), m_end).astype(np.int64)  # 1e4 rows
    # pairing check: the IQ engine applied exactly the controller's decisions
    mism = 0
    for t_app, st, _en in iq["applied"]:
        m = math.floor(t_app / URAP_PERIOD)
        mism += int(not np.array_equal(np.asarray(st, np.int64), sched[m]))
    r = coefficient_power(model, i0, i1, GEOM_DT, lambda t: sched[math.floor(t / URAP_PERIOD)])
    mid = piecewise_power(model, dur, GEOM_DT, _switches(iq), mode="mid")
    t1 = time.perf_counter()
    per_s = int(Fraction(1) / GEOM_DT)
    np.savez_compressed(
        ctx.run_dir / "bins.npz",
        iq_sum=iq["sums"],
        iq_count=iq["counts"],
        coef_mean=r["p"].reshape(-1, per_s).mean(axis=1),
        coef_los_mean=r["p_los"].reshape(-1, per_s).mean(1),
    )
    w6 = None
    if p.get("w6_batch_s"):
        w6 = w6_rows(scen, model, iq, dur, Fraction(p["w6_batch_s"]))
    return {
        "h_m": p["h_m"],
        "T_s": p["T_s"],
        "iq_mean_mw": float(iq["sums"].sum() / iq["counts"].sum()),
        "coef_mean_mw": float(r["p"].mean()),
        "coef_mid_mean_mw": float(mid["mean_mw"]),
        "coef_los_mean_mw": float(r["p_los"].mean()),
        "n_applied": len(iq["applied"]),
        "config_mismatches": mism,
        "iq_lateness_max": iq["lateness_max"],
        "iq_run_s": iq["run_s"],
        "iq_rtf": iq["rtf"],
        "iq_build_s": iq["build_s"],
        "coef_s": t1 - t0,
        "verify_interp_max_rel": iq["verify_interp_max_rel"],
        "t_u_s": iq["t_u_s"],
        "tiles": iq["tiles"],
        "counters": iq["counters"],
        "w6": w6,
    }


def run_w5_iq(ctx: Any) -> dict[str, Any]:
    """Sweep runner: W5 one (controller, tau_SUI) at h = 4: IQ vs semi-analytic oracle."""
    p = dict(ctx.params)
    dur = Fraction(p["T_s"])
    h = float(p["h_m"])
    tau = Fraction(p["tau_s"])
    scen = corridor_scenario(
        h,
        seed=int(p["seed"]),
        duration_s=str(p["T_s"]),
        controllers=[delayed_spec(p["controller"], p["tau_s"])],
        name="vwcl_w5",
    )
    iq = run_iq_engine(scen, dur)
    model = build_model(scen)
    n = model.panel_geometry(PANEL).n_cells
    t0 = time.perf_counter()
    starts, wps = waypoint_table(mu_trajectory(model), float(dur))
    # oracle decisions at the barriers t_b = m tau (positions at t_b; latency 0)
    m_end = math.ceil(dur / tau)
    tb = np.array([float(m * tau) for m in range(m_end)])
    xb = positions_x(tb, starts, wps)
    h0b, sb = oracle_channel(xb, h, D0_M)
    dec = np.array([exact_binary(h0b[i], sb[i])[1] for i in range(m_end)], dtype=bool)
    # engine decisions (applied configurations) mapped to barrier index
    eng_dec = np.zeros((m_end, n), dtype=bool)
    have = np.zeros(m_end, dtype=bool)
    for t_app, st, _en in iq["applied"]:
        m = math.floor(t_app / tau)
        if 0 <= m < m_end:
            eng_dec[m] = np.asarray(st) == 1
            have[m] = True
    for m in range(1, m_end):  # skip_unchanged: no command means "keep the previous"
        if not have[m]:
            eng_dec[m] = eng_dec[m - 1]
    same = np.all(dec == eng_dec, axis=1)
    # oracle power: 1 ms midpoints, piecewise configuration per barrier interval
    per_s = 1000
    nt = int(dur) * per_s
    tm = (np.arange(nt, dtype=np.float64) + 0.5) / per_s
    mb = np.minimum((tm / float(tau)).astype(np.int64), m_end - 1)
    p_or = np.empty(nt)
    p_or_eng = np.empty(nt)
    p_los = np.empty(nt)
    blk = 20_000
    for a in range(0, nt, blk):
        sl = slice(a, min(nt, a + blk))
        x = positions_x(tm[sl], starts, wps)
        h0, s = oracle_channel(x, h, D0_M)
        cfg = dec[mb[sl]].astype(np.float64)
        cfg_e = eng_dec[mb[sl]].astype(np.float64)
        p_or[sl] = np.abs(h0 + np.einsum("ij,ij->i", s, cfg)) ** 2
        p_or_eng[sl] = np.abs(h0 + np.einsum("ij,ij->i", s, cfg_e)) ** 2
        p_los[sl] = np.abs(h0) ** 2
    t1 = time.perf_counter()
    np.savez_compressed(
        ctx.run_dir / "bins.npz",
        iq_sum=iq["sums"],
        iq_count=iq["counts"],
        oracle_mean=p_or.reshape(-1, per_s).mean(axis=1),
        oracle_engdec_mean=p_or_eng.reshape(-1, per_s).mean(axis=1),
        los_mean=p_los.reshape(-1, per_s).mean(axis=1),
    )
    # trajectory pairing check (oracle positions vs the model's trajectory)
    probe = np.linspace(0.0, float(dur) - 1e-3, 257)
    xm = np.array([model.node_position("mu", Fraction(float(t)))[0] for t in probe])
    w6 = None
    if p.get("w6_batch_s"):
        w6 = w6_rows(scen, model, iq, dur, Fraction(p["w6_batch_s"]))
    iq_mean = float(iq["sums"].sum() / iq["counts"].sum())
    return {
        "controller": p["controller"],
        "tau_s": p["tau_s"],
        "h_m": h,
        "T_s": p["T_s"],
        "iq_mean_mw": iq_mean,
        "oracle_mean_mw": float(p_or.mean()),
        "oracle_engdec_mean_mw": float(p_or_eng.mean()),
        "los_mean_mw": float(p_los.mean()),
        "delta_db": vs.db10(iq_mean) - vs.db10(float(p_or.mean())),
        "delta_engdec_db": vs.db10(iq_mean) - vs.db10(float(p_or_eng.mean())),
        "decisions_identical_fraction": float(np.mean(same)),
        "n_barriers": m_end,
        "traj_max_abs_dx_m": float(np.max(np.abs(positions_x(probe, starts, wps) - xm))),
        "iq_lateness_max": iq["lateness_max"],
        "n_applied": len(iq["applied"]),
        "iq_run_s": iq["run_s"],
        "iq_rtf": iq["rtf"],
        "oracle_s": t1 - t0,
        "verify_interp_max_rel": iq["verify_interp_max_rel"],
        "t_u_s": iq["t_u_s"],
        "tiles": iq["tiles"],
        "oracle_csi_used": iq["oracle_csi_used"],
        "w6": w6,
    }


# =========================================================================== statistics


def batch_ci_db(values: np.ndarray, confidence: float = 0.95) -> dict[str, Any]:
    """Mean of equal-size batch values (linear power) with its Student-t half-width (mW;
    :func:`sremu.analysis.stats.batch_means` with one value per batch) and in dB:
    ``up = 10log10(1 + hw/m)``, ``down = -10log10(1 - hw/m)`` (None when the interval
    reaches 0); ``half_width_db = max(up, down)`` (None then: not informative)."""
    from sremu.analysis.stats import batch_means

    v = np.asarray(values, dtype=np.float64)
    k = v.size
    r = batch_means(v, k, confidence=confidence)
    m = float(r.mean)
    hw = float(r.half_width)
    up = vs.db10(1.0 + hw / m)
    down = -vs.db10(1.0 - hw / m) if hw < m else None
    return {
        "mean_mw": m,
        "half_width_mw": hw,
        "half_width_db_up": up,
        "half_width_db_down": down,
        "half_width_db": None if down is None else max(up, down),
        "n_batches": k,
        "mean_dbm": vs.db10(m),
    }


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_batches: int = 20) -> dict[str, float]:
    """Batch-means CI (Student t) of the mean of the paired difference ``b - a`` (per-bin
    linear powers)."""
    from sremu.analysis.stats import batch_means

    d = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    r = batch_means(d, min(n_batches, d.size))
    return {"mean_diff_mw": r.mean, "half_width_mw": r.half_width, "n_batches": r.n_batches}


# =========================================================================== W1


def w1_study(n_periods: int, seed: int = SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """W1: RWP statistics of the scenario's MU trajectory and the legacy_wcl_eq2 teeth."""
    from scipy import stats

    from sremu.physics.legacy import legacy_wcl_eq2

    t_total = MOVE_PERIOD * n_periods
    scen = corridor_scenario(2.0, seed=seed, duration_s=str(t_total))
    model = build_model(scen)
    traj = mu_trajectory(model)
    t0 = time.perf_counter()
    n_s = int(t_total * 10)
    xs = np.array([model.node_position("mu", Fraction(k, 10))[0] for k in range(n_s)])
    t1 = time.perf_counter()

    def cdf(x: np.ndarray) -> np.ndarray:
        return 3 * x**2 / L_M**2 - 2 * x**3 / L_M**3

    ks_pos = stats.kstest(xs, cdf)
    starts = traj.leg_starts(float(t_total))
    k_legs = len(starts) - 1  # legs completed inside [0, T)
    wps = np.array([traj.waypoint(i)[0] for i in range(k_legs + 1)])
    ks_start = stats.kstest(wps[:-1], stats.uniform(loc=0.0, scale=L_M).cdf)
    mean_len = float(np.mean(np.abs(np.diff(wps))))
    max_jump = float(np.max(np.abs(np.diff(xs))))
    hist, edges = np.histogram(xs, bins=50, range=(0.0, L_M), density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    rows = [
        {"x_m": float(c), "density_emulator": float(hh), "pdf_eq1": float(rwp_pdf(c, 0, L_M))}
        for c, hh in zip(centres, hist, strict=True)
    ]
    # legacy_wcl_eq2 teeth: per-element power ratio to sremu's K1 cascade
    lam = C0 / FC
    n, w = cells()
    worst = 0.0
    n_ratio = 0
    for h in H_LIST:
        for x in (0.5, 2.5, 5.0, 7.5, 9.5):
            sc = corridor_scenario(h, seed=seed)
            sc["nodes"][1] = dict(sc["nodes"][1])
            sc["nodes"][1].pop("trajectory")
            sc["nodes"][1]["position_m"] = [x, 0.0, 0.0]
            m = build_model(sc)
            z = m.csi_view(m.snapshot(Fraction(0)), LINK).cascade[PANEL][1]
            pts = m.panel_geometry(PANEL).cell_centres_m()
            for i in range(n):
                e2 = legacy_wcl_eq2(
                    [0.0, 0.0, D0_M],
                    [x, 0.0, 0.0],
                    pts[i : i + 1],
                    w * w,
                    [0.0, -1.0, 0.0],
                    FC,
                    allow_nonphysical=True,
                )
                ri = float(np.linalg.norm(pts[i] - np.array([0.0, 0.0, D0_M])))
                rs = float(np.linalg.norm(pts[i] - np.array([x, 0.0, 0.0])))
                ci, cs = h / ri, h / rs
                closed = (math.pi * lam) ** 2 * (ci * cs) ** (2 * Q_LETTER - 1)
                ratio = abs(e2) ** 2 / abs(z[i]) ** 2
                worst = max(worst, abs(ratio / closed - 1.0))
                n_ratio += 1
    summary = {
        "n_periods_requested": n_periods,
        "T_s": t_total,
        "n_position_samples": n_s,
        "sample_rate_hz": 10,
        "positions_wall_s": t1 - t0,
        "ks_positions_D": float(ks_pos.statistic),
        "ks_positions_accept_D_le": 0.02,
        "ks_positions_pass": bool(ks_pos.statistic <= 0.02),
        "n_legs": k_legs,
        "ks_starts_D": float(ks_start.statistic),
        "ks_starts_accept_D_le": 0.0163,
        "ks_starts_pass": bool(ks_start.statistic <= 0.0163),
        "mean_leg_length_m": mean_len,
        "mean_leg_length_ref_m": L_M / 3,
        "mean_leg_rel_err": mean_len / (L_M / 3) - 1.0,
        "mean_leg_pass": bool(abs(mean_len / (L_M / 3) - 1.0) <= 0.025),
        "max_jump_m": max_jump,
        "max_jump_bound_m": V_MPS * 0.1 + 1e-12,
        "max_jump_pass": bool(max_jump <= V_MPS * 0.1 + 1e-12),
        "teeth_eq2_ratio_max_rel_dev": worst,
        "teeth_eq2_n_ratios": n_ratio,
        "teeth_pass": bool(worst <= 1e-9),
        "teeth_ratio_broadside_db_1ghz": 20 * math.log10(math.pi * lam),
    }
    return rows, summary


# =========================================================================== sweep points


def run_point(ctx: Any) -> dict[str, Any]:
    """Sweep runner (``vwcl:run_point``): dispatch on ``params["kind"]``."""
    kind = ctx.params["kind"]
    if kind == "w3_chunk":
        return run_w3_chunk(ctx)
    if kind == "w3_iq":
        return run_w3_iq(ctx)
    if kind == "w5_iq":
        return run_w5_iq(ctx)
    if kind == "w3_cond":
        return run_w3_cond(ctx)
    if kind == "analytic":
        p = ctx.params
        t0 = time.perf_counter()
        out: dict[str, Any] = {
            "label": p["label"],
            "h_m": p["h_m"],
            "d0_m": p["d0_m"],
            "seg_start_m": p["seg_start_m"],
        }
        a = analytic_averages(
            p["h_m"], p["d0_m"], p["seg_start_m"], n_quad=int(p.get("n_quad") or 4000)
        )
        out.update({f"a4000_{k}": v for k, v in a.items()})
        out["n_quad"] = int(p.get("n_quad") or 4000)
        if p.get("convergence"):
            b = analytic_averages(p["h_m"], p["d0_m"], p["seg_start_m"], n_quad=8000)
            out.update({f"a8000_{k}": v for k, v in b.items()})
        if p.get("stale"):
            out["stale_limit_mw"] = stale_limit(p["h_m"], p["d0_m"])
        out["wall_s"] = time.perf_counter() - t0
        return out
    raise ValueError(f"unknown point kind {kind!r}")


_PKEYS = (
    "kind",
    "label",
    "h_m",
    "d0_m",
    "seg_start_m",
    "T_s",
    "n_chunks",
    "chunk",
    "with_opt",
    "controller",
    "tau_s",
    "w6_batch_s",
    "convergence",
    "stale",
    "n_quad",
)


def _frac_str(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


PROFILES: dict[str, dict[str, Any]] = {
    # full: the study; fast: minutes (development); smoke: the validation_smoke tests
    "full": {
        "n_main": N_PERIODS_MAIN,
        "n_sens": 1000,
        "k_main": 20,
        "k_sens": 10,
        "t_iq": "1000",
        "t_w6": "100",
        "taus": TAUS,
        "controllers": ("ExactOptimum", "WclHeuristic"),
        "h_list": H_LIST,
        "sens": True,
        "n_quad": 4000,
    },
    "fast": {
        "n_main": 60,
        "n_sens": 30,
        "k_main": 4,
        "k_sens": 2,
        "t_iq": "20",
        "t_w6": "10",
        "taus": ("1/10", "3/10", "1"),
        "controllers": ("ExactOptimum", "WclHeuristic"),
        "h_list": H_LIST,
        "sens": True,
        "n_quad": 4000,
    },
    "smoke": {
        "n_main": 6,
        "n_sens": 0,
        "k_main": 2,
        "k_sens": 0,
        "t_iq": "4",
        "t_w6": "2",
        "taus": ("1/10", "1"),
        "controllers": ("ExactOptimum",),
        "h_list": (2.0,),
        "sens": False,
        "n_quad": 400,
    },
}


def plan_points(profile: str, parts: Sequence[str]) -> list[dict[str, Any]]:
    """Every sweep point of the W3/W5/W6 parts (long IQ points first)."""
    pr = PROFILES[profile]
    n_main, n_sens, k_main, k_sens = pr["n_main"], pr["n_sens"], pr["k_main"], pr["k_sens"]
    t_iq, t_w6 = pr["t_iq"], pr["t_w6"]
    pts: list[dict[str, Any]] = []
    if "w5" in parts:
        for ctrl in pr["controllers"]:
            for tau in pr["taus"]:
                w6_cases = {
                    ("ExactOptimum", "1/100"),
                    ("ExactOptimum", "1/10"),
                    ("ExactOptimum", "1"),
                    ("WclHeuristic", "1/10"),
                }
                w6 = t_w6 if (ctrl, tau) in w6_cases else None
                pts.append(
                    {
                        "kind": "w5_iq",
                        "label": f"w5_{ctrl}_{tau}",
                        "h_m": 4.0,
                        "d0_m": D0_M,
                        "seg_start_m": 0.0,
                        "T_s": t_iq,
                        "controller": ctrl,
                        "tau_s": tau,
                        "w6_batch_s": w6,
                    }
                )
    if "w3" in parts:
        for h in pr["h_list"]:
            pts.append(
                {
                    "kind": "w3_iq",
                    "label": f"w3iq_h{h:g}",
                    "h_m": h,
                    "d0_m": D0_M,
                    "seg_start_m": 0.0,
                    "T_s": t_iq,
                    "w6_batch_s": t_w6 if h == 2.0 else None,
                }
            )
        t_main = MOVE_PERIOD * n_main
        for h in pr["h_list"]:
            for j in range(k_main):
                pts.append(
                    {
                        "kind": "w3_chunk",
                        "label": "main",
                        "h_m": h,
                        "d0_m": D0_M,
                        "seg_start_m": 0.0,
                        "T_s": _frac_str(t_main),
                        "n_chunks": k_main,
                        "chunk": j,
                        "with_opt": True,
                    }
                )
        t_sens = MOVE_PERIOD * max(n_sens, 1)
        for label, d0, c in SENSITIVITY if pr["sens"] else ():
            for h in pr["h_list"]:
                for j in range(k_sens):
                    pts.append(
                        {
                            "kind": "w3_chunk",
                            "label": label,
                            "h_m": h,
                            "d0_m": d0,
                            "seg_start_m": c,
                            "T_s": _frac_str(t_sens),
                            "n_chunks": k_sens,
                            "chunk": j,
                            "with_opt": True,
                        }
                    )
        labels = (("main", D0_M, 0.0), *SENSITIVITY) if pr["sens"] else (("main", D0_M, 0.0),)
        for label, d0, c in labels:
            for h in pr["h_list"]:
                pts.append(
                    {
                        "kind": "analytic",
                        "label": label,
                        "h_m": h,
                        "d0_m": d0,
                        "seg_start_m": c,
                        "convergence": label == "main" and profile == "full",
                        "stale": label == "main",
                        "n_quad": pr["n_quad"],
                    }
                )
    if "w3" in parts:
        # appended after every other point, so the ids of the original points never change
        # (resume-safe); same parameter keys as the w3_chunk points
        t_main = MOVE_PERIOD * n_main
        for h in pr["h_list"]:
            for j in range(k_main):
                pts.append(
                    {
                        "kind": "w3_cond",
                        "label": "main",
                        "h_m": h,
                        "d0_m": D0_M,
                        "seg_start_m": 0.0,
                        "T_s": _frac_str(t_main),
                        "n_chunks": k_main,
                        "chunk": j,
                    }
                )
    for p in pts:
        for k in _PKEYS:
            p.setdefault(k, None)
        p["seed"] = SEED
    return pts


def run_points(
    points: Sequence[Mapping[str, Any]],
    out_dir: Path,
    jobs: int,
    name: str,
    *,
    resume: bool = False,
    progress: Any = None,
) -> Any:
    """Run the points with the sremu sweep runner (``vwcl:run_point``, ``jobs`` workers;
    ``resume`` skips points already completed with the same parameters)."""
    from sremu.experiments.sweep import SweepSpec, run_sweep

    keys = list(points[0])
    spec = SweepSpec.from_mapping(
        {
            "name": name,
            "runner": "vwcl:run_point",
            "description": "V-WCL W3/W5/W6 points (validation/wcl/vwcl.py)",
            "zip": {k: [p[k] for p in points] for k in keys},
        }
    )
    return run_sweep(spec, out_dir, jobs=jobs, allow_import=True, resume=resume, progress=progress)


# =========================================================================== aggregation


def _summaries(sweep: Any, kind: str) -> list[dict[str, Any]]:
    out = []
    for r in sweep.records:
        if r["status"] != "ok":
            raise RuntimeError(f"sweep point {r['point_id']} failed: {r.get('error')}")
        if r["params"]["kind"] == kind:
            out.append(dict(r["summary"]) | {"point_id": r["point_id"], "wall_s": r["wall_s"]})
    return out


def _label_of(c: Mapping[str, Any]) -> str:
    for label, d0, seg in (("main", D0_M, 0.0), *SENSITIVITY):
        if abs(c["d0_m"] - d0) < 1e-12 and abs(c["seg_start_m"] - seg) < 1e-12:
            return label
    return "?"


def _w3_rows(
    an: Mapping[tuple[str, float], Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    order = sorted(an.items(), key=lambda kv: (kv[0][0] != "main", kv[0]))
    for (label, h), a in order:
        cs = sorted(
            (c for c in chunks if c["h_m"] == h and _label_of(c) == label),
            key=lambda c: c["chunk"],
        )
        if not cs:
            continue
        means = np.array([c["sum_p"] / c["n"] for c in cs])
        los = np.array([c["sum_p_los"] / c["n"] for c in cs])
        opt = np.array([c["sum_p_opt"] / c["n_opt"] for c in cs])
        ci, ci_los, ci_opt = batch_ci_db(means), batch_ci_db(los), batch_ci_db(opt)
        mp = a["a4000_urap_mean_power"]
        mf = a["a4000_urap_mean_field_eq9"]
        d_mp = vs.db10(ci["mean_mw"]) - vs.db10(mp)
        d_mf = vs.db10(ci["mean_mw"]) - vs.db10(mf)
        hw = ci["half_width_db"]
        tol = None if hw is None else max(0.05, hw)
        conv = a.get("a8000_urap_mean_power")
        stale = a.get("stale_limit_mw")
        rows.append(
            {
                "label": label,
                "h_m": h,
                "d0_m": a["d0_m"],
                "seg_start_m": a["seg_start_m"],
                "n_batches": ci["n_batches"],
                "n_grid_points": int(sum(c["n"] for c in cs)),
                "emu_urap_dbm": ci["mean_dbm"],
                "emu_urap_ci_half_db": ci["half_width_db"],
                "analytic_mean_power_dbm": vs.db10(mp),
                "analytic_mean_field_eq9_dbm": vs.db10(mf),
                "delta_vs_mean_power_db": d_mp,
                "delta_vs_mean_field_db": d_mf,
                "tolerance_db": tol,
                "pass_mean_power": None if tol is None else bool(abs(d_mp) <= tol),
                "mean_field_within_tol": None if tol is None else bool(abs(d_mf) <= tol),
                "emu_los_dbm": ci_los["mean_dbm"],
                "emu_los_ci_half_db": ci_los["half_width_db"],
                "analytic_los_dbm": vs.db10(a["a4000_los"]),
                "delta_los_db": ci_los["mean_dbm"] - vs.db10(a["a4000_los"]),
                "emu_opt_dbm": ci_opt["mean_dbm"],
                "emu_opt_ci_half_db": ci_opt["half_width_db"],
                "analytic_opt_dbm": vs.db10(a["a4000_opt"]),
                "delta_opt_db": ci_opt["mean_dbm"] - vs.db10(a["a4000_opt"]),
                "analytic_all_on_dbm": vs.db10(a["a4000_all_on"]),
                "gain_urap_over_los_db_analytic": vs.db10(mp / a["a4000_los"]),
                "gain_mean_field_over_los_db_analytic": vs.db10(mf / a["a4000_los"]),
                "gain_opt_over_los_db_analytic": vs.db10(a["a4000_opt"] / a["a4000_los"]),
                "gain_urap_over_los_db_emu": ci["mean_dbm"] - ci_los["mean_dbm"],
                "gain_opt_over_los_db_emu": ci_opt["mean_dbm"] - ci_los["mean_dbm"],
                "quad_convergence_db": None if conv is None else vs.db10(conv) - vs.db10(mp),
                "stale_limit_dbm": None if stale is None else vs.db10(stale),
                "wall_s_sum": float(sum(c["wall_s"] for c in cs)),
            }
        )
    return rows


def _w3_conditional_rows(
    chunks: Sequence[Mapping[str, Any]], conds: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """W3 on the same trajectory: emulator batch means vs the conditional expectations
    E[P | x(t)] (both forms), paired per batch (Student-t interval of the per-batch
    difference); plus the independent replica of the realised URAP power and the LOS."""
    from sremu.analysis.stats import batch_means

    rows = []
    for h in sorted({c["h_m"] for c in conds}):
        cs = {c["chunk"]: c for c in chunks if c["h_m"] == h and _label_of(c) == "main"}
        ks = {c["chunk"]: c for c in conds if c["h_m"] == h}
        common = sorted(set(cs) & set(ks))
        if len(common) < 2:
            continue
        if any(cs[i]["n"] != ks[i]["n"] or cs[i]["i0"] != ks[i]["i0"] for i in common):
            raise RuntimeError(f"W3 conditional: grids differ at h = {h}")
        n = np.array([cs[i]["n"] for i in common], dtype=np.float64)
        emu = np.array([cs[i]["sum_p"] for i in common]) / n
        emu_los = np.array([cs[i]["sum_p_los"] for i in common]) / n
        mp = np.array([ks[i]["sum_cond_mean_power"] for i in common]) / n
        mf = np.array([ks[i]["sum_cond_mean_field_eq9"] for i in common]) / n
        los = np.array([ks[i]["sum_cond_los"] for i in common]) / n
        real = np.array([ks[i]["sum_oracle_realised"] for i in common]) / n
        e, m_p, m_f = float(emu.mean()), float(mp.mean()), float(mf.mean())
        bp = batch_means(emu - mp, len(common))
        bf = batch_means(emu - mf, len(common))
        rows.append(
            {
                "h_m": h,
                "n_batches": len(common),
                "n_grid_points": int(n.sum()),
                "emu_urap_dbm": vs.db10(e),
                "cond_mean_power_dbm": vs.db10(m_p),
                "cond_mean_field_eq9_dbm": vs.db10(m_f),
                "delta_vs_cond_mean_power_db": vs.db10(e / m_p),
                "ci_half_vs_mean_power_db": vs.db10(1.0 + bp.half_width / m_p),
                "mean_power_within_ci": bool(abs(bp.mean) <= bp.half_width),
                "delta_vs_cond_mean_field_db": vs.db10(e / m_f),
                "ci_half_vs_mean_field_db": vs.db10(1.0 + bf.half_width / m_f),
                "mean_field_within_ci": bool(abs(bf.mean) <= bf.half_width),
                "forms_differ_db": vs.db10(m_p / m_f),
                "urap_increment_emu_db": vs.db10(e / float(emu_los.mean())),
                "urap_increment_cond_mean_power_db": vs.db10(m_p / float(los.mean())),
                "urap_increment_cond_eq9_db": vs.db10(m_f / float(los.mean())),
                "oracle_replica_max_rel_diff": float(np.max(np.abs(real / emu - 1.0))),
                "los_replica_max_rel_diff": float(np.max(np.abs(los / emu_los - 1.0))),
                "cond_wall_s_sum": float(sum(ks[i].get("wall_s", 0.0) for i in common)),
            }
        )
    return rows


def aggregate(sweep: Any, sweep_dir: Path) -> dict[str, Any]:
    """Tables and acceptance of W3 (main, sensitivity, IQ subset), W5 and W6."""
    out: dict[str, Any] = {"tables": {}}
    an = {(a["label"], a["h_m"]): a for a in _summaries(sweep, "analytic")}
    rows = _w3_rows(an, _summaries(sweep, "w3_chunk"))
    out["tables"]["w3_coefficient"] = rows
    main = [r for r in rows if r["label"] == "main"]
    if main:
        out["w3_pass"] = all(r["pass_mean_power"] is True for r in main)
    out["tables"]["w3_conditional"] = _w3_conditional_rows(
        _summaries(sweep, "w3_chunk"), _summaries(sweep, "w3_cond")
    )
    # ---------------- W3 IQ subset
    iq_rows = []
    for s in sorted(_summaries(sweep, "w3_iq"), key=lambda s: s["h_m"]):
        z = np.load(sweep_dir / s["point_id"] / "bins.npz")
        iq_bins = z["iq_sum"] / z["iq_count"]
        nb = 20 if iq_bins.size >= 40 else max(2, iq_bins.size // 2)
        ci = batch_ci_db(iq_bins[: (iq_bins.size // nb) * nb].reshape(nb, -1).mean(axis=1))
        a = an[("main", s["h_m"])]
        mp = a["a4000_urap_mean_power"]
        d_pair = vs.db10(s["iq_mean_mw"]) - vs.db10(s["coef_mean_mw"])
        d_mid = vs.db10(s["iq_mean_mw"]) - vs.db10(s["coef_mid_mean_mw"])
        d_an = vs.db10(s["iq_mean_mw"]) - vs.db10(mp)
        hw = ci["half_width_db"]
        iq_rows.append(
            {
                "h_m": s["h_m"],
                "T_s": s["T_s"],
                "iq_dbm": vs.db10(s["iq_mean_mw"]),
                "coef_dbm": vs.db10(s["coef_mean_mw"]),
                "delta_iq_vs_coef_db": d_pair,
                "pass_iq_vs_coef_0p02": bool(abs(d_pair) <= 0.02),
                "coef_mid_dbm": vs.db10(s["coef_mid_mean_mw"]),
                "delta_iq_vs_coef_mid_db": d_mid,
                "iq_ci_half_db": ci["half_width_db"],
                "n_batches": ci["n_batches"],
                "analytic_mean_power_dbm": vs.db10(mp),
                "delta_iq_vs_analytic_db": d_an,
                "pass_iq_vs_analytic": None if hw is None else bool(abs(d_an) <= max(0.05, hw)),
                "delta_iq_vs_mean_field_db": vs.db10(s["iq_mean_mw"])
                - vs.db10(a["a4000_urap_mean_field_eq9"]),
                "config_mismatches": s["config_mismatches"],
                "n_applied": s["n_applied"],
                "lateness_max": s["iq_lateness_max"],
                "iq_build_s": s["iq_build_s"],
                "iq_run_s": s["iq_run_s"],
                "iq_rtf": s["iq_rtf"],
                "coef_s": s["coef_s"],
                "t_u_s": s["t_u_s"],
                "tiles": str(s["tiles"]),
                "verify_interp_max_rel": s["verify_interp_max_rel"],
            }
        )
    out["tables"]["w3_iq"] = iq_rows
    if iq_rows:
        out["w3_iq_pass"] = all(
            r["pass_iq_vs_coef_0p02"] and r["config_mismatches"] == 0 for r in iq_rows
        )
    # ---------------- W5
    w5 = sorted(_summaries(sweep, "w5_iq"), key=lambda s: (s["controller"], Fraction(s["tau_s"])))
    w5_rows = []
    bins_by: dict[tuple[str, str], np.ndarray] = {}
    for s in w5:
        z = np.load(sweep_dir / s["point_id"] / "bins.npz")
        iq_bins = z["iq_sum"] / z["iq_count"]
        bins_by[(s["controller"], s["tau_s"])] = iq_bins
        nb = 20 if iq_bins.size >= 40 else max(2, iq_bins.size // 2)
        ci = batch_ci_db(iq_bins[: (iq_bins.size // nb) * nb].reshape(nb, -1).mean(axis=1))
        primary = s["delta_db"] if s["controller"] == "ExactOptimum" else s["delta_engdec_db"]
        a5 = an.get(("main", s["h_m"]), {})
        ctx5 = {
            "analytic_los_dbm": vs.db10(a5["a4000_los"]) if "a4000_los" in a5 else None,
            "analytic_opt_tau0_dbm": vs.db10(a5["a4000_opt"]) if "a4000_opt" in a5 else None,
            "stale_limit_dbm": vs.db10(a5["stale_limit_mw"]) if a5.get("stale_limit_mw") else None,
        }
        w5_rows.append(
            {
                "controller": s["controller"],
                "tau_s": s["tau_s"],
                "tau_float_s": float(Fraction(s["tau_s"])),
                "h_m": s["h_m"],
                "T_s": s["T_s"],
                "iq_dbm": vs.db10(s["iq_mean_mw"]),
                "iq_ci_half_db": ci["half_width_db"],
                "oracle_dbm": vs.db10(s["oracle_mean_mw"]),
                "delta_iq_vs_oracle_db": s["delta_db"],
                "oracle_engine_decisions_dbm": vs.db10(s["oracle_engdec_mean_mw"]),
                "delta_iq_vs_oracle_engdec_db": s["delta_engdec_db"],
                "primary_delta_db": primary,
                "pass_0p02": bool(abs(primary) <= 0.02),
                "oracle_decisions": "independent exact optimum"
                if s["controller"] == "ExactOptimum"
                else "engine-recorded decisions",
                "decisions_identical_fraction": s["decisions_identical_fraction"],
                "los_dbm": vs.db10(s["los_mean_mw"]),
                "gain_over_los_db": vs.db10(s["iq_mean_mw"]) - vs.db10(s["los_mean_mw"]),
                "n_barriers": s["n_barriers"],
                "n_applied": s["n_applied"],
                "lateness_max": s["iq_lateness_max"],
                "traj_max_abs_dx_m": s["traj_max_abs_dx_m"],
                "iq_run_s": s["iq_run_s"],
                "iq_rtf": s["iq_rtf"],
                "oracle_s": s["oracle_s"],
                "t_u_s": s["t_u_s"],
                "tiles": str(s["tiles"]),
                "verify_interp_max_rel": s["verify_interp_max_rel"],
                "oracle_csi_used": s["oracle_csi_used"],
                **ctx5,
            }
        )
    mono = []
    for ctrl in sorted({r["controller"] for r in w5_rows}):
        taus = sorted((t for (c, t) in bins_by if c == ctrl), key=Fraction)
        for t_a, t_b in itertools.pairwise(taus):
            a_bins, b_bins = bins_by[(ctrl, t_a)], bins_by[(ctrl, t_b)]
            n = min(a_bins.size, b_bins.size)
            ci = paired_diff_ci(a_bins[:n], b_bins[:n])
            mono.append(
                {
                    "controller": ctrl,
                    "tau_a_s": t_a,
                    "tau_b_s": t_b,
                    "mean_diff_mw": ci["mean_diff_mw"],
                    "half_width_mw": ci["half_width_mw"],
                    "mean_diff_db": vs.db10(float(b_bins[:n].mean()))
                    - vs.db10(float(a_bins[:n].mean())),
                    "non_increasing_within_ci": bool(
                        ci["mean_diff_mw"] - ci["half_width_mw"] <= 0.0
                    ),
                }
            )
    out["tables"]["w5"] = w5_rows
    out["tables"]["w5_monotonicity"] = mono
    if w5_rows:
        out["w5_pass"] = all(r["pass_0p02"] for r in w5_rows) and all(
            m["non_increasing_within_ci"] for m in mono
        )
    # ---------------- W6
    w6_rows = []
    w6_batch = []
    for s_ in _summaries(sweep, "w5_iq") + _summaries(sweep, "w3_iq"):
        w6 = s_.get("w6")
        if not w6:
            continue
        case = f"{s_.get('controller', 'Urap')} tau={s_.get('tau_s', '1/10')} h={s_['h_m']:g}"
        for r in w6["rows"]:
            w6_rows.append(
                {
                    "case": case,
                    "T_s": w6["T_s"],
                    "grid": r["grid"],
                    "dt_s": r["dt_s"],
                    "iq_dbm": vs.db10(r["iq_mean_mw"]),
                    "snapshot_dbm": vs.db10(r["snapshot_mean_mw"]),
                    "delta_db": r["delta_db"],
                    "pass_0p01": bool(abs(r["delta_db"]) <= 0.01),
                    "n_pieces": r["n_pieces"],
                    "wall_s": r["wall_s"],
                }
            )
        rb = w6.get("run_batch")
        if rb:
            w6_batch.append(
                {
                    "case": case,
                    **rb,
                    "iq_dbm": vs.db10(rb["iq_mean_mw"]),
                    "batch_dbm": vs.db10(rb["batch_mean_mw"]),
                }
            )
    # midpoint references of the same runs (no new simulation): the W5 oracle (independent
    # closed form on 1 ms midpoints) and the W3 IQ subset's 5 ms midpoint piecewise average
    for s_ in _summaries(sweep, "w5_iq"):
        if not s_.get("w6"):
            continue
        case = f"{s_['controller']} tau={s_['tau_s']} h={s_['h_m']:g}"
        d = s_["delta_db"] if s_["controller"] == "ExactOptimum" else s_["delta_engdec_db"]
        ref = (
            s_["oracle_mean_mw"]
            if s_["controller"] == "ExactOptimum"
            else s_["oracle_engdec_mean_mw"]
        )
        w6_rows.append(
            {
                "case": case,
                "T_s": s_["T_s"],
                "grid": "1ms_midpoint_oracle",
                "dt_s": "1/1000",
                "iq_dbm": vs.db10(s_["iq_mean_mw"]),
                "snapshot_dbm": vs.db10(ref),
                "delta_db": d,
                "pass_0p01": bool(abs(d) <= 0.01),
            }
        )
    for s_ in _summaries(sweep, "w3_iq"):
        if not s_.get("w6"):
            continue
        case = f"Urap tau=1/10 h={s_['h_m']:g}"
        d = vs.db10(s_["iq_mean_mw"]) - vs.db10(s_["coef_mid_mean_mw"])
        w6_rows.append(
            {
                "case": case,
                "T_s": s_["T_s"],
                "grid": "5ms_midpoint",
                "dt_s": "1/200",
                "iq_dbm": vs.db10(s_["iq_mean_mw"]),
                "snapshot_dbm": vs.db10(s_["coef_mid_mean_mw"]),
                "delta_db": d,
                "pass_0p01": bool(abs(d) <= 0.01),
            }
        )
    for r in w6_rows:  # left-point quadrature error scales with the step: ratio vs 1 ms
        if r["grid"] in ("5ms", "auto_T_u"):
            r1 = next((x for x in w6_rows if x["case"] == r["case"] and x["grid"] == "1ms"), None)
            if r1 is not None and r1["delta_db"] != 0:
                r["ratio_to_1ms_left"] = r["delta_db"] / r1["delta_db"]
    out["tables"]["w6"] = w6_rows
    out["tables"]["w6_run_batch"] = w6_batch
    if w6_rows:
        out["w6_pass_by_grid"] = {
            g: all(r["pass_0p01"] for r in w6_rows if r["grid"] == g)
            for g in sorted({r["grid"] for r in w6_rows})
        }
        iso = [r for r in w6_rows if r["grid"] in ("1ms", "1ms_midpoint_oracle", "5ms_midpoint")]
        out["w6_doppler_isolated_pass"] = bool(iso) and all(r["pass_0p01"] for r in iso)
    return out


# =========================================================================== main


def _check_params() -> Path:
    """The transcribed letter parameters, checked against ``SHA256SUMS``."""
    d = vs.DATA_VALIDATION / "wcl"
    f = d / "wcl2025_parameters.json"
    sums = dict(line.split()[::-1] for line in (d / "SHA256SUMS").read_text().splitlines())
    got = vs.sha256_file(f)
    if got != sums[f.name]:
        raise SystemExit(f"{f.name}: sha256 {got} != SHA256SUMS {sums[f.name]}")
    return f


def _cpu_by_kind(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per point kind: number of points, summed and maximum point wall time (s)."""
    out: dict[str, Any] = {}
    for r in records:
        k = r["params"]["kind"]
        d = out.setdefault(k, {"n": 0, "cpu_s_sum": 0.0, "point_wall_s_max": 0.0})
        w = float(r.get("wall_s") or 0.0)
        d["n"] += 1
        d["cpu_s_sum"] += w
        d["point_wall_s_max"] = max(d["point_wall_s_max"], w)
    return out


def study(
    parts: Sequence[str],
    *,
    jobs: int,
    profile: str = "full",
    root: Path | None = None,
    resume_dir: Path | None = None,
) -> Path:
    """Run the selected parts into one run directory (or continue ``resume_dir``)."""
    n_main = PROFILES[profile]["n_main"]
    t_main = MOVE_PERIOD * n_main
    base = corridor_scenario(
        2.0, duration_s=_frac_str(t_main), controllers=[urap_spec()], name="vwcl_base"
    )
    params_file = _check_params()
    run = vs.StudyRun(
        STUDY_DIR,
        "vwcl" if profile == "full" else f"vwcl_{profile}",
        base,
        inputs=[(params_file, "REPORTED")],
        assumptions=ASSUMPTIONS,
        root=root,
        resume_dir=resume_dir,
    )
    meta: dict[str, Any] = {
        "study": "V-WCL",
        "parts": list(parts),
        "profile": profile,
        "data_classes": ["SIM-sremu", "ANALYTIC", "REPORTED", "ASSUMED"],
        "paper_use": {
            "W1 positions/starts/mean leg/max jump, W3, W5, W6": "paper (verification under the "
            "area-tied kernel; DESIGN 1.6 rule 6 wording)",
            "W1 teeth_eq2_*": "internal only: kernel-ratio teeth test, not featured in the paper "
            "(DESIGN D24; nothing about the letter's Eq. 2, DESIGN 1.6 rule 6, H-3)",
        },
        "seeds": {
            "scenario_seed": SEED,
            "mobility_key": "mobility/mu (scenario seed)",
            "urap_key": "ctrl/urap (scenario seed)",
        },
        "env_threads": vs.env_threads(),
        "acceptance": {},
        "numbers": {},
    }
    sweep_parts = [p for p in parts if p in ("w3", "w5")]
    if sweep_parts:
        pts = plan_points(profile, sweep_parts)
        t0 = time.perf_counter()
        sw = run_points(
            pts, run.run_dir, jobs, "points", resume=resume_dir is not None, progress=run.progress
        )
        run.time("sweep_wall", time.perf_counter() - t0)
        meta["sweep"] = {
            "n_points": len(pts),
            "n_failed": sw.n_failed,
            "n_resumed": sw.n_resumed,
            "jobs": jobs,
            "cpu_s_sum": float(sum(r["wall_s"] for r in sw.records if r.get("wall_s"))),
            "by_kind": _cpu_by_kind(sw.records),
        }
        if sw.n_failed:
            vs.write_json(
                run.path("failed_points.json"), [r for r in sw.records if r["status"] != "ok"]
            )
            meta["error"] = "sweep points failed"
            run.finish(meta)
            raise SystemExit(f"{sw.n_failed} sweep points failed; see {run.run_dir}")
        agg = aggregate(sw, sw.root)
        for name, tab in agg["tables"].items():
            if tab:
                vs.write_csv(run.path(f"{name}.csv"), tab)
        for key in (
            "w3_pass",
            "w3_iq_pass",
            "w5_pass",
            "w6_pass_by_grid",
            "w6_doppler_isolated_pass",
        ):
            if key in agg:
                meta["acceptance"][key] = agg[key]
        meta["numbers"]["tables"] = agg["tables"]
    if "w1" in parts:
        t0 = time.perf_counter()
        rows, summ = w1_study(n_main)
        run.time("w1", time.perf_counter() - t0)
        vs.write_csv(run.path("w1_position_histogram.csv"), rows)
        vs.write_json(run.path("w1_summary.json"), summ)
        meta["acceptance"]["W1"] = {k: v for k, v in summ.items() if k.endswith("_pass")}
        meta["numbers"]["W1"] = summ
    path = run.finish(meta)
    print(f"V-WCL run directory: {run.run_dir}")
    print(f"meta: {path}")
    if root is None and profile == "full":
        print(f"report: {vs.write_report(STUDY_DIR, report(run.run_dir))}")
    return run.run_dir


class _Records:
    """Sweep records read back from a run directory's ``points/p*/`` (re-aggregation)."""

    def __init__(self, sweep_dir: Path) -> None:
        import json

        self.records: list[dict[str, Any]] = []
        for pj in sorted(sweep_dir.glob("p*/point.json")):
            rec = json.loads(pj.read_text(encoding="utf-8"))
            sj = pj.parent / "summary.json"
            rec["summary"] = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else None
            self.records.append(rec)


def reaggregate(run_dir: Path) -> Path:
    """Recompute the W3/W5/W6 tables and acceptance of a finished run directory from its
    saved per-point outputs (no simulation); rewrites the CSV tables and ``meta.json``."""
    run_dir = Path(run_dir)
    meta = vs.read_meta(run_dir)
    sw = _Records(run_dir / "points")
    agg = aggregate(sw, run_dir / "points")
    for name, tab in agg["tables"].items():
        if tab:
            vs.write_csv(run_dir / f"{name}.csv", tab)
    for key in ("w3_pass", "w3_iq_pass", "w5_pass", "w6_pass_by_grid", "w6_doppler_isolated_pass"):
        if key in agg:
            meta["acceptance"][key] = agg[key]
    meta["numbers"]["tables"] = agg["tables"]
    meta.setdefault("reaggregated_utc", []).append(_dt.datetime.now(_dt.UTC).isoformat())
    return vs.refresh_outputs(run_dir, meta)


# =========================================================================== report


def _pass_walls(run_dir: Path) -> list[tuple[str, Any]]:
    """(name, wall_s) of the earlier passes of a resumed run directory."""
    import json

    out: list[tuple[str, Any]] = []
    for mf in sorted(run_dir.glob("manifest.start*.json")):
        try:
            out.append(
                (mf.stem.replace("manifest.", ""), json.loads(mf.read_text("utf-8"))["wall_s"])
            )
        except (OSError, ValueError, KeyError):
            continue
    return out


def _run_history(run_dir: Path) -> list[str]:
    """Lines describing earlier passes of a resumed run directory (start-of-pass manifests
    kept as ``manifest.start<k>.json``, failures in ``failed_points.json``)."""
    import json

    out = []
    for mf in sorted(run_dir.glob("manifest.start*.json")):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append(
            f"`{mf.name}`: pass started {m.get('t_start_utc')}, ended {m.get('t_end_utc')}, "
            f"commit `{m.get('git_commit')}`, dirty {m.get('git_dirty')} (patch sha256 "
            f"`{m.get('dirty_patch_sha256')}`)."
        )
    fp = run_dir / "failed_points.json"
    if fp.exists():
        try:
            recs = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recs = []
        for r in recs:
            out.append(
                f"`failed_points.json` (first pass): {r.get('point_id')} "
                f"({(r.get('params') or {}).get('kind')}, h = "
                f"{(r.get('params') or {}).get('h_m')} m, chunk "
                f"{(r.get('params') or {}).get('chunk')}) failed with "
                f'"{str(r.get("error"))[:120]}"; it was re-run in a resumed pass after the '
                "decision table of `urap_states` was reduced to the chunk's rows (int8), with "
                "identical decisions (same keyed controller, stepped from barrier 0)."
            )
    return out


def _tab(meta: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    return list((meta.get("numbers") or {}).get("tables", {}).get(name) or [])


def report(run_dir: Path) -> str:
    """``validation/wcl/report.md`` rendered from a run directory's ``meta.json``."""
    meta = vs.read_meta(run_dir)
    f = vs.fnum
    acc = meta.get("acceptance", {})
    w1 = (meta.get("numbers") or {}).get("W1") or {}
    w3 = _tab(meta, "w3_coefficient")
    w3iq = _tab(meta, "w3_iq")
    w5 = _tab(meta, "w5")
    mono = _tab(meta, "w5_monotonicity")
    w6 = _tab(meta, "w6")
    w6b = _tab(meta, "w6_run_batch")
    w3c = _tab(meta, "w3_conditional")
    sweep = meta.get("sweep") or {}
    n, w = cells()
    s = []
    s.append("# V-WCL: reproduction checks of the WCL 2025 letter (DESIGN.md 10.11)\n")
    s.append(
        "Mishra & Kherani, IEEE WCL 14(4), 2025: published analytic results. An earlier "
        "in-house simulator by the authors was used in that letter; sremu is an independent new "
        "implementation (DESIGN D24: no comparison with that simulator is made here or in the "
        "paper). These are **verification** checks of sremu against independent closed forms "
        "and its own batch path under the area-tied K1 kernel, not a validation against "
        "measurements. Nothing here states anything about the letter's Eq. (2) (DESIGN 1.6 "
        'rule 6, H-3); numbers are phrased "under the area-tied kernel the value is X". Data '
        f"classes: {', '.join(meta.get('data_classes', []))}.\n"
    )
    s.append("## Provenance\n")
    s.append(vs.run_header(meta) + "\n")
    s.append(
        f"* Profile `{meta.get('profile')}`, parts {meta.get('parts')}; sweep of "
        f"{sweep.get('n_points')} points with the sremu sweep runner, {sweep.get('jobs')} "
        f"workers, {sweep.get('n_failed')} failed, {sweep.get('n_resumed', 0)} resumed.\n"
    )
    hist = _run_history(Path(run_dir))
    if hist:
        s.append("Run history (from the files kept in the run directory):\n")
        s.append("\n".join(f"* {h}" for h in hist) + "\n")
    s.append("## Scenario as run\n")
    s.append(
        f"* Corridor of l = {L_M:g} m, MU on 1-D random waypoint at v = {V_MPS:g} m/s, no "
        f"pause; RIS line at distance h with N = {n} cells of w = lambda/4 = {w:.5f} m at "
        f"f_c = {FC / 1e9:g} GHz; on_off alphabet (OFF absorbing, ON Gamma = 1); P_T = 0 dBm; "
        "isotropic antennas; IQ path: 10 kS/s unit tone, noise off, lockstep, numba_cpu.\n"
        f"* **Documented deviation:** AP offset d0 = {D0_M:g} m perpendicular to the user line "
        "(+z); with the AP on the line the mean LOS power diverges as x -> 0. Sensitivity: "
        "d0 = 0.3 and 1.0 m, and an on-line AP with the MU segment starting at c = 0.05, 0.1, "
        "0.2 m.\n"
        "* Assumptions recorded in the manifest: " + "; ".join(meta.get("assumptions", [])) + ".\n"
    )
    # ---------------- acceptance summary
    s.append("## Acceptance summary\n")
    rows = []
    if w1:
        rows += [
            (
                "W1 positions vs F(x) (KS D <= 0.02)",
                f(w1.get("ks_positions_D"), 4),
                vs.passfail(w1.get("ks_positions_pass")),
            ),
            (
                "W1 waypoints uniform (KS D <= 0.0163)",
                f(w1.get("ks_starts_D"), 4),
                vs.passfail(w1.get("ks_starts_pass")),
            ),
            (
                "W1 mean leg length l/3 +- 2.5 %",
                f"{f(w1.get('mean_leg_length_m'), 4)} m "
                f"({f(100 * w1.get('mean_leg_rel_err', 0), 2, sign=True)} %)",
                vs.passfail(w1.get("mean_leg_pass")),
            ),
            (
                "W1 no jump > v dt + 1e-12",
                f"{f(w1.get('max_jump_m'), 6)} m",
                vs.passfail(w1.get("max_jump_pass")),
            ),
            (
                "W1 teeth (internal, not for the paper per D24): legacy_wcl_eq2 / K1 ratio vs "
                "closed form (<= 1e-9)",
                f(w1.get("teeth_eq2_ratio_max_rel_dev"), 3),
                vs.passfail(w1.get("teeth_pass")),
            ),
        ]
    for r in (x for x in w3 if x["label"] == "main"):
        rows.append(
            (
                f"W3 h = {r['h_m']:g} m: emulator vs analytic mean power, |d| <= "
                f"max(0.05 dB, CI) = {f(r['tolerance_db'], 3)} dB",
                f"{f(r['delta_vs_mean_power_db'], 4, sign=True)} dB",
                vs.passfail(r["pass_mean_power"]),
            )
        )
    for r in w3c:
        rows.append(
            (
                f"W3 same-trajectory h = {r['h_m']:g} m (not a DESIGN gate): emulator vs "
                f"E[P | x(t)] mean power, within the paired 95 % CI "
                f"(+-{f(r['ci_half_vs_mean_power_db'], 5)} dB)",
                f"{f(r['delta_vs_cond_mean_power_db'], 5, sign=True)} dB",
                "consistent" if r["mean_power_within_ci"] else "outside CI",
            )
        )
    for r in w3iq:
        rows.append(
            (
                f"W3 IQ subset h = {r['h_m']:g} m: IQ vs paired batch path (<= 0.02 dB)",
                f"{f(r['delta_iq_vs_coef_db'], 5, sign=True)} dB",
                vs.passfail(r["pass_iq_vs_coef_0p02"] and r["config_mismatches"] == 0),
            )
        )
    for r in w5:
        rows.append(
            (
                f"W5 {r['controller']} tau_SUI = {r['tau_s']} s: IQ vs oracle (<= 0.02 dB)",
                f"{f(r['primary_delta_db'], 5, sign=True)} dB",
                vs.passfail(r["pass_0p02"]),
            )
        )
    for m in mono:
        rows.append(
            (
                f"W5 {m['controller']} non-increasing {m['tau_a_s']} -> {m['tau_b_s']} s "
                "(within CI)",
                f"{f(m['mean_diff_db'], 4, sign=True)} dB",
                vs.passfail(m["non_increasing_within_ci"]),
            )
        )
    for r in w6:
        rows.append(
            (
                f"W6 {r['case']}, reference {r['grid']}: IQ (Doppler) vs snapshot average "
                "(<= 0.01 dB)",
                f"{f(r['delta_db'], 5, sign=True)} dB",
                vs.passfail(r["pass_0p01"]),
            )
        )
    s.append(vs.md_table(("Check", "Measured", "Result"), rows) + "\n")
    if "w6_doppler_isolated_pass" in acc:
        bad = [r for r in w6 if r["grid"] in ("5ms", "auto_T_u") and not r["pass_0p01"]]
        s.append(
            "W6 summary: with references whose own quadrature error is small (left-point 1 ms, "
            "the independent oracle on 1 ms midpoints, the 5 ms midpoint average) every case "
            f"is within 0.01 dB ({vs.passfail(acc['w6_doppler_isolated_pass'])}). With the "
            "batch path's own left-point grids (automatic T_u and 5 ms) "
            + (
                f"{len(bad)} row(s) exceed 0.01 dB (largest "
                f"{f(max(abs(r['delta_db']) for r in bad), 4)} dB); see the W6 section for why "
                "this is the reference's quadrature error and not the Doppler effect."
                if bad
                else "every case is also within 0.01 dB."
            )
            + "\n"
        )
    s.append(f"Machine-readable acceptance (meta.json): `{json_compact(acc)}`\n")
    # ---------------- W1
    if w1:
        s.append("## W1: mobility statistics (and an internal kernel-ratio teeth test)\n")
        s.append(
            f"{w1.get('n_position_samples')} positions at {w1.get('sample_rate_hz')} Hz over "
            f"T = {w1.get('T_s')} s ({w1.get('n_periods_requested')} mean movement periods; "
            f"{w1.get('n_legs')} legs). The `legacy_wcl_eq2` per-element power ratio to "
            f"sremu's K1 cascade equals (pi lambda)^2 (cos_i cos_s)^(2q-1) with q = "
            f"{Q_LETTER} to {f(w1.get('teeth_eq2_ratio_max_rel_dev'), 3)} (relative, "
            f"{w1.get('teeth_eq2_n_ratios')} element ratios); at broadside and 1 GHz the "
            f"factor is {f(w1.get('teeth_ratio_broadside_db_1ghz'), 3)} dB per element. This "
            "is a statement about two kernels' ratio, not about which is right; it is an "
            "internal teeth test and is not featured in the paper (DESIGN D24).\n"
        )
    # ---------------- W3
    main = [r for r in w3 if r["label"] == "main"]
    if main:
        s.append("## W3: URAP (p = 0.2) time-averaged power vs the analytic integrals\n")
        s.append(
            "Emulator side: the batch (coefficient) path `h = h_direct + sum_n cascade[s_n, n]` "
            "from `model.csi_view(model.snapshot(t))` on a 5 ms grid (left-point pieces, the "
            "batch path's semantics) over 10^4 mean movement periods per h, split into 20 "
            "consecutive batches (batch means, Student t 95 %); the URAP decisions come from "
            "the `Urap` controller itself (keyed generator, 0.1 s redraw). Analytic side: "
            "integrals over the stationary pdf of the letter's Eq. (9) (power of the mean "
            "field) and of the mean power (adds p(1-p) sum |S_n|^2), with an independent "
            "closed-form channel (Gauss-Legendre, 4000 nodes; 8000-node change reported).\n"
        )
        s.append(
            vs.md_table(
                (
                    "h (m)",
                    "emulator (dBm)",
                    "CI half (dB)",
                    "analytic mean power (dBm)",
                    "analytic Eq. 9 (dBm)",
                    "d vs mean power (dB)",
                    "d vs Eq. 9 (dB)",
                    "tol (dB)",
                    "result",
                ),
                (
                    (
                        f(r["h_m"], 0),
                        f(r["emu_urap_dbm"], 4),
                        f(r["emu_urap_ci_half_db"], 4),
                        f(r["analytic_mean_power_dbm"], 4),
                        f(r["analytic_mean_field_eq9_dbm"], 4),
                        f(r["delta_vs_mean_power_db"], 4, sign=True),
                        f(r["delta_vs_mean_field_db"], 4, sign=True),
                        f(r["tolerance_db"], 3),
                        vs.passfail(r["pass_mean_power"]),
                    )
                    for r in main
                ),
            )
            + "\n"
        )
        los_e = main[0]["emu_los_dbm"]
        los_a = main[0]["analytic_los_dbm"]
        s.append(
            "Why the interval is wide and the offset is the same at every h: with d0 = "
            f"{D0_M:g} m the LOS power, proportional to 1/(x^2 + d0^2), peaks near the AP, "
            "where the random waypoint rarely goes (f_X -> 0 as x -> 0), so the time average "
            "of the LOS power itself carries this mobility sampling error over 10^4 periods "
            f"(emulator LOS {f(los_e, 4)} dBm vs the stationary {f(los_a, 4)} dBm, the same "
            "trajectory at every h), while the URAP increment over LOS is below 0.01 dB. The "
            "stationary comparison therefore passes through the CI rule and cannot separate "
            "the two forms. The same-trajectory comparison below removes the mobility noise.\n"
        )
        if w3c:
            s.append(
                "**Same trajectory (not a DESIGN gate; added to make W3 discriminating).** On "
                "the same 5 ms grid and trajectory, the independent closed form gives the "
                "conditional expectations E[P | x(t)] of URAP in both forms; the emulator's "
                "batch means are compared per batch (paired Student-t 95 % interval over the "
                "20 batches, only the URAP draws vary). The independent replica of the realised "
                "URAP power (same `Urap` decisions, closed-form channel) and of the LOS agrees "
                "with the emulator to the relative differences shown.\n"
            )
            s.append(
                vs.md_table(
                    (
                        "h (m)",
                        "emulator (dBm)",
                        "E[P|x] mean power (dBm)",
                        "E[P|x] Eq. 9 (dBm)",
                        "d vs mean power (dB)",
                        "d vs Eq. 9 (dB)",
                        "paired CI half (dB)",
                        "forms differ by (dB)",
                        "URAP - LOS: emulator / mean power / Eq. 9 (dB)",
                        "replica max rel. diff",
                    ),
                    (
                        (
                            f(r["h_m"], 0),
                            f(r["emu_urap_dbm"], 5),
                            f(r["cond_mean_power_dbm"], 5),
                            f(r["cond_mean_field_eq9_dbm"], 5),
                            f(r["delta_vs_cond_mean_power_db"], 5, sign=True)
                            + (" (in CI)" if r["mean_power_within_ci"] else " (outside CI)"),
                            f(r["delta_vs_cond_mean_field_db"], 5, sign=True)
                            + (" (in CI)" if r["mean_field_within_ci"] else " (outside CI)"),
                            f(r["ci_half_vs_mean_power_db"], 5),
                            f(r["forms_differ_db"], 5),
                            f"{f(r['urap_increment_emu_db'], 5, sign=True)} / "
                            f"{f(r['urap_increment_cond_mean_power_db'], 5, sign=True)} / "
                            f"{f(r['urap_increment_cond_eq9_db'], 5, sign=True)}",
                            f(
                                max(
                                    r["oracle_replica_max_rel_diff"], r["los_replica_max_rel_diff"]
                                ),
                                2,
                            ),
                        )
                        for r in w3c
                    ),
                )
                + "\n"
            )
        s.append(
            "Gains over LOS under the area-tied kernel (analytic stationary averages; emulator "
            "batch means in brackets), d0 = 0.5 m:\n"
        )
        s.append(
            vs.md_table(
                (
                    "h (m)",
                    "LOS (dBm)",
                    "URAP mean power - LOS (dB)",
                    "Eq. 9 - LOS (dB)",
                    "exact optimum - LOS (dB)",
                    "tau_SUI -> inf limit (dBm)",
                    "quadrature 8000 vs 4000 (dB)",
                ),
                (
                    (
                        f(r["h_m"], 0),
                        f"{f(r['analytic_los_dbm'], 4)} [{f(r['emu_los_dbm'], 4)}]",
                        f"{f(r['gain_urap_over_los_db_analytic'], 4, sign=True)} "
                        f"[{f(r['gain_urap_over_los_db_emu'], 4, sign=True)}]",
                        f(r["gain_mean_field_over_los_db_analytic"], 4, sign=True),
                        f"{f(r['gain_opt_over_los_db_analytic'], 4, sign=True)} "
                        f"[{f(r['gain_opt_over_los_db_emu'], 4, sign=True)}]",
                        f(r["stale_limit_dbm"], 4),
                        f(r["quad_convergence_db"], 6, sign=True),
                    )
                    for r in main
                ),
            )
            + "\n"
        )
    sens = [r for r in w3 if r["label"] != "main"]
    if sens:
        s.append("### W3 sensitivity to the AP offset (10^3 movement periods, 10 batches)\n")
        s.append(
            vs.md_table(
                (
                    "case",
                    "h (m)",
                    "emulator (dBm)",
                    "CI half (dB)",
                    "d vs mean power (dB)",
                    "d vs Eq. 9 (dB)",
                    "within tol",
                    "URAP - LOS (dB)",
                    "opt - LOS (dB)",
                ),
                (
                    (
                        r["label"],
                        f(r["h_m"], 0),
                        f(r["emu_urap_dbm"], 4),
                        f(r["emu_urap_ci_half_db"], 4),
                        f(r["delta_vs_mean_power_db"], 4, sign=True),
                        f(r["delta_vs_mean_field_db"], 4, sign=True),
                        vs.passfail(r["pass_mean_power"]),
                        f(r["gain_urap_over_los_db_analytic"], 4, sign=True),
                        f(r["gain_opt_over_los_db_analytic"], 4, sign=True),
                    )
                    for r in sens
                ),
            )
            + "\n"
        )
    if w3iq:
        s.append("### W3 IQ subset (the IQ engine where the test requires it)\n")
        s.append(
            "Lockstep IQ engine with the in-process `Urap` barrier controller, 1000 s per h, "
            "paired with the batch path on the same trajectory and the same URAP draws "
            "(configuration mismatches counted).\n"
        )
        s.append(
            vs.md_table(
                (
                    "h (m)",
                    "IQ (dBm)",
                    "batch (dBm)",
                    "IQ - batch (dB)",
                    "IQ - batch midpoint (dB)",
                    "IQ - analytic (dB)",
                    "IQ CI half (dB)",
                    "mismatches",
                    "IQ run (s)",
                    "RTF",
                ),
                (
                    (
                        f(r["h_m"], 0),
                        f(r["iq_dbm"], 5),
                        f(r["coef_dbm"], 5),
                        f(r["delta_iq_vs_coef_db"], 6, sign=True),
                        f(r["delta_iq_vs_coef_mid_db"], 6, sign=True),
                        f(r["delta_iq_vs_analytic_db"], 4, sign=True),
                        f(r["iq_ci_half_db"], 4),
                        r["config_mismatches"],
                        f(r["iq_run_s"], 0),
                        f(r["iq_rtf"], 3),
                    )
                    for r in w3iq
                ),
            )
            + "\n"
        )
        s.append(
            "The IQ - analytic column compares a 1000 s run (about 300 movement periods) with "
            "the stationary average; it is reported, not gated.\n"
        )
    # ---------------- W5
    if w5:
        s.append("## W5: signaling delay tau_SUI, IQ vs the position-only oracle (h = 4 m)\n")
        s.append(
            "`DelayedController(inner, tau_SUI, report_delay = 0)` with command latency 0, "
            "1000 s per point, noise off. Oracle: the independent closed-form channel at the "
            "MU positions (piecewise-linear from the scenario's waypoints) on 1 ms midpoints, "
            "with its own exact ON/OFF optimum at the barrier positions (ExactOptimum) or "
            "with the engine-recorded decisions (WclHeuristic, which the oracle does not "
            "re-implement).\n"
        )
        s.append(
            vs.md_table(
                (
                    "controller",
                    "tau_SUI (s)",
                    "IQ (dBm)",
                    "oracle (dBm)",
                    "IQ - oracle (dB)",
                    "gain over LOS (dB)",
                    "decisions identical",
                    "barriers",
                    "IQ run (s)",
                    "RTF",
                    "result",
                ),
                (
                    (
                        r["controller"],
                        r["tau_s"],
                        f(r["iq_dbm"], 5),
                        f(
                            r["oracle_dbm"]
                            if r["controller"] == "ExactOptimum"
                            else r["oracle_engine_decisions_dbm"],
                            5,
                        ),
                        f(r["primary_delta_db"], 6, sign=True),
                        f(r["gain_over_los_db"], 4, sign=True),
                        f(r["decisions_identical_fraction"], 3)
                        if r["controller"] == "ExactOptimum"
                        else "n/a (engine decisions)",
                        r["n_barriers"],
                        f(r["iq_run_s"], 0),
                        f(r["iq_rtf"], 3),
                        vs.passfail(r["pass_0p02"]),
                    )
                    for r in w5
                ),
            )
            + "\n"
        )
        r0 = w5[0]
        if r0.get("analytic_opt_tau0_dbm") is not None:
            s.append(
                f"Context (stationary averages, h = 4 m): LOS {f(r0.get('analytic_los_dbm'), 4)} "
                f"dBm; exact optimum at tau_SUI -> 0 {f(r0.get('analytic_opt_tau0_dbm'), 4)} dBm; "
                f"tau_SUI -> infinity limit (optimum of an independent position) "
                f"{f(r0.get('stale_limit_dbm'), 4)} dBm. The 1000 s runs cover about 300 "
                "movement periods, so their absolute levels differ from these by the "
                "trajectory's sampling error.\n"
            )
        if mono:
            s.append(
                "Monotonicity (paired per-second bins, batch-means CI of the difference, 20 "
                "batches): a step passes when the lower CI end of P(tau_b) - P(tau_a) is <= 0.\n"
            )
            s.append(
                vs.md_table(
                    ("controller", "tau_a -> tau_b (s)", "mean diff (dB)", "result"),
                    (
                        (
                            m["controller"],
                            f"{m['tau_a_s']} -> {m['tau_b_s']}",
                            f(m["mean_diff_db"], 5, sign=True),
                            vs.passfail(m["non_increasing_within_ci"]),
                        )
                        for m in mono
                    ),
                )
                + "\n"
            )
    # ---------------- W6
    if w6:
        s.append("## W6: Doppler on/off (sample-level geometric vs position-snapshot average)\n")
        s.append(
            "The IQ engine renders the chord-delay geometric Doppler at every sample (f_D <= "
            "3.34 Hz here). The snapshot average uses |h(x(t))|^2 of frozen positions (no "
            "Doppler) with the engine-recorded configurations: left-point pieces (the batch "
            "path's semantics) at its automatic T_u, 5 ms and 1 ms; the independent oracle on "
            "1 ms midpoints (the W5 reference of the same run); and the 5 ms midpoint average "
            "of the W3 IQ subset. The left-point rule has a first-order quadrature error that "
            "scales with the step: right after each barrier the configuration is the best one "
            "for the current position and the power then decays as the user moves, so holding "
            "the left value over-weights the start of every piece. The last column gives the "
            "ratio of the left-point difference to the 1 ms one (about 5 for a 5x step means "
            "first-order quadrature error); the midpoint references isolate the Doppler "
            "effect. The URAP rows have no systematic decay after a switch (random "
            "configurations), so they show no left-point bias and their ratios are noise.\n"
        )
        s.append(
            vs.md_table(
                (
                    "case",
                    "reference",
                    "IQ (dBm)",
                    "snapshot (dBm)",
                    "IQ - snapshot (dB)",
                    "result",
                    "ratio to 1 ms left",
                ),
                (
                    (
                        r["case"],
                        r["grid"],
                        f(r["iq_dbm"], 6),
                        f(r["snapshot_dbm"], 6),
                        f(r["delta_db"], 6, sign=True),
                        vs.passfail(r["pass_0p01"]),
                        f(r.get("ratio_to_1ms_left"), 2),
                    )
                    for r in w6
                ),
            )
            + "\n"
        )
        if w6b:
            s.append(
                "`run_batch` itself (its own controller instance, 5 ms grid) over the first "
                "100 s against the piecewise evaluation and the IQ run:\n"
            )
            s.append(
                vs.md_table(
                    (
                        "case",
                        "run_batch vs piecewise (rel)",
                        "IQ - run_batch (dB)",
                        "run_batch (s)",
                    ),
                    (
                        (
                            r["case"],
                            f(r["batch_vs_piecewise_rel"], 3, sign=True),
                            f(r["delta_iq_vs_batch_db"], 6, sign=True),
                            f(r["batch_s"], 1),
                        )
                        for r in w6b
                    ),
                )
                + "\n"
            )
    # ---------------- run times
    s.append("## Run times\n")
    bk = sweep.get("by_kind") or {}
    s.append(
        vs.md_table(
            ("part", "points", "summed point wall (s)", "longest point (s)"),
            (
                (k, v["n"], f(v["cpu_s_sum"], 0), f(v["point_wall_s_max"], 0))
                for k, v in sorted(bk.items())
            ),
        )
        + "\n"
    )
    passes = _pass_walls(Path(run_dir))
    passes.append(("final pass", meta.get("wall_s")))
    s.append(
        "Wall time per pass (manifests): "
        + "; ".join(f"{name} {f(wv, 0)} s" for name, wv in passes)
        + f"; total {f(sum(float(wv or 0) for _n, wv in passes), 0)} s with "
        f"{sweep.get('jobs')} workers. In the final pass the sweep took "
        f"{f((meta.get('timings_s') or {}).get('sweep_wall'), 0)} s and W1 "
        f"{f((meta.get('timings_s') or {}).get('w1'), 0)} s. The table sums each point's own "
        "wall time over all passes. IQ RTF = simulated seconds / wall seconds of the lockstep "
        "engine in one worker process while the other workers (and other agents' jobs) loaded "
        "the CPU; these are not benchmark numbers (section 12 owns those).\n"
    )
    s.append("## Files\n")
    s.append(
        "`w1_position_histogram.csv`, `w1_summary.json`, `w3_conditional.csv` (same-"
        "trajectory W3), `w3_coefficient.csv` (main and "
        "sensitivity), `w3_iq.csv`, `w5.csv`, `w5_monotonicity.csv`, `w6.csv`, "
        "`w6_run_batch.csv`; per point `points/p*/` (point.json, summary.json, bins.npz of "
        "per-second powers for the IQ points); `meta.json` (all numbers, acceptance, data "
        "classes, output hashes); `manifest.json`.\n"
    )
    return "\n".join(s)


def json_compact(x: Any) -> str:
    import json

    return json.dumps(vs.jsonable(x), separators=(",", ":"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V-WCL validation study (DESIGN.md 10.11)")
    ap.add_argument("parts", nargs="*", default=["all"], help="w1 w3 w5 | all")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--fast", action="store_true", help="development sizes (minutes)")
    ap.add_argument("--smoke", action="store_true", help="test sizes (validation_smoke)")
    ap.add_argument("--root", type=Path, default=None, help="results root (default results/)")
    ap.add_argument("--resume", type=Path, default=None, help="continue this run directory")
    ap.add_argument("--aggregate", type=Path, default=None, help="re-aggregate a run directory")
    ap.add_argument("--report", type=Path, default=None, help="write report.md from a run dir")
    a = ap.parse_args(argv)
    if a.aggregate is not None:
        print(reaggregate(a.aggregate))
        return 0
    if a.report is not None:
        print(vs.write_report(STUDY_DIR, report(a.report)))
        return 0
    parts = ["w1", "w3", "w5"] if "all" in a.parts else list(a.parts)
    profile = "smoke" if a.smoke else ("fast" if a.fast else "full")
    study(parts, jobs=a.jobs, profile=profile, root=a.root, resume_dir=a.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
