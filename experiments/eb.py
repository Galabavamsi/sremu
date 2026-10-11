"""E-B: intra-frame RIS reconfiguration (DESIGN.md section 11, E-B; VA-5).

Geometry ``scenarios/lab_like_A.yaml`` (SIM geometry labelled "lab-like", not the lab's):
3 tiles of 8 x 5 cells at 42.8 mm with 2 mm gaps, 3.5 GHz; Tx 2.0 m at -30 deg, RX1 1.2 m
at 15 deg, RX2 1.2 m at 35 deg. Alphabets: ``lab`` (sigma1/sigma2 with |Gamma| = 1,
ASSUMED) and ``ideal1`` (ideal 1-bit {1, -1}). LOS present; sensitivity variant
``blocked`` adds an opaque occluder across both AP-RX LOS segments (it misses every RIS
hop).

Waveform: 1 MS/s, QPSK at 4 samples/symbol, rectangular pulses, integrate-and-dump; 64
known preamble + 512 payload symbols (2304 samples, T_p = 2.304 ms); one-tap LS equalizer
from the preamble, no tracking. Physical noise (NF 7 dB); the TX power is set so that
Es/N0 = 25 dB at RX1 under its own optimal configuration (computed from the snapshot and
recorded). Configurations: exact optimum (``sremu.physics.optimize``) for RX1 and for RX2.

* **E-B1** (:func:`run_eb1_point`): one packet to RX1; switch to the RX2 configuration at
  ``delta * T_p`` after the packet start, with a step or a linear 20 us / 200 us transition;
  per-symbol EVM; VA-5: the first sample of RX1 that differs from the no-switch run equals
  ``to_ticks(t_sw + tau_RIS->RX1)`` and the engine's ``k_applied``; samples (and so whole
  symbols) before it are bitwise identical.
* **E-B2** (:func:`run_eb2_point`): 2000 packets alternating RX1/RX2. Commands come from
  the ``TdmaAlternate`` controller (one per guard; its decisions). Their latency
  ``L = 0.2 ms + U[0, J]`` is the controller's keyed ``LatencyModel`` draw
  (``sremu.engine.barriers.draw_latency_s``: purpose ``ctrl.latency/<name>``, counter =
  barrier index). Arrival = issue + L. Semantics:

  - ``timed``: requested switch at the guard centre, issued ``lead`` earlier; a late
    arrival applies on arrival (``apply_on_arrival``): t_sw = max(t_req, arrival);
  - ``asap``: issued at the guard start, applied on arrival;
  - ``legacy``: issued at the guard start, ``LegacyNextTickAdapter`` (2 ms ticks,
    single-slot overwrite) at the arrival time.

  In lockstep with barrier coupling the engine evaluates a barrier controller's actions at
  ``t_b + L_min`` and ignores the latency of an action that carries ``at_s`` (it can never
  be late), so the arrival semantics are applied here and the resulting commands are
  scheduled in arrival order (later arrival = later ``seq``, which the engine's same-time
  rule needs for the legacy overwrite). See the contract change request in the WP11 report.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import (
    REPO,
    RunRecord,
    db10,
    quiet_warnings,
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
FS = Fraction(10**6)
SPS = 4
N_PRE = 64
N_PAY = 512
BODY = (N_PRE + N_PAY) * SPS  # 2304 samples
T_P = Fraction(BODY) / FS  # 2.304 ms
ES_N0_DB = 25.0
NF_DB = 7.0
TX = "ap.tx0"
RX1 = "rx1.rx0"
RX2 = "rx2.rx0"
LINKS = {RX1: f"{TX}->{RX1}", RX2: f"{TX}->{RX2}"}
PANEL = "ris0"
LEGACY_TAU_S = Fraction(1, 500)  # 2 ms
LAT_FIXED_S = Fraction(1, 5000)  # 0.2 ms
CTRL_NAME = "tdma"
ALPHABETS = {
    "lab": {"library": "lab_binary_3p5ghz", "assume_mag": {"sigma1": 1.0, "sigma2": 1.0}},
    "ideal1": {"library": "ideal_antipodal"},
}
TRANSITIONS = {
    "step": None,
    "linear20us": TransitionSpec("linear", Fraction(20, 10**6)),
    "linear200us": TransitionSpec("linear", Fraction(200, 10**6)),
}
BLOCKER = [[1.5, -0.6, -0.5], [1.5, -0.35, -0.5], [1.5, -0.35, 0.5], [1.5, -0.6, 0.5]]


# =========================================================================== scenario


def lab_scenario(
    alphabet: str,
    los: str = "present",
    *,
    full_scale_dbm: float = 0.0,
    initial_states: Sequence[int] | None = None,
    duration_s: Fraction | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """``lab_like_A.yaml`` adapted for E-B: alphabet, LOS variant, TX power, initial
    configuration; no scenario controllers and no scenario traffic (the drivers supply
    them)."""
    d = copy.deepcopy(read_document(LAB))
    d["name"] = f"eb_{alphabet}_{los}"
    if seed is not None:
        d["seed"] = int(seed)
    p = d["ris"][0]
    p["states"] = dict(ALPHABETS[alphabet])
    if initial_states is not None:
        p["initial"] = {"states": [int(s) for s in initial_states]}
    for nd in d["nodes"]:
        for s in nd["streams"]:
            if s["direction"] == "tx":
                s["full_scale_dbm"] = float(full_scale_dbm)
            else:
                s["noise_figure_db"] = NF_DB
    if los == "blocked":
        d["environment"] = {
            "occluders": [{"id": "los_blocker", "kind": "opaque", "vertices_m": BLOCKER}]
        }
    elif los != "present":
        raise ValueError(los)
    d["controllers"] = []
    d["traffic"] = []
    d["record"] = {"iq": False, "events": True, "metrics": False}
    if duration_s is not None:
        d["run"]["duration_s"] = str(duration_s)
    d["description"] = f"E-B lab-like SIM geometry (not the lab's), alphabet {alphabet}, LOS {los}"
    return d


def resolve(d: Mapping[str, Any]) -> Any:
    """Validate, resolve and guard a scenario mapping."""
    from sremu.scenario.guards import check_scenario

    quiet_warnings()
    sc = parse_scenario(dict(d), label=str(d.get("name")))
    res = resolve_scenario(sc, LAB.parent, warn=False)
    check_scenario(res).raise_for_errors()
    return res


@dataclass
class Design:
    """Exact-optimum configurations, channels and the TX power of one (alphabet, LOS)."""

    alphabet: str
    los: str
    s1: np.ndarray
    s2: np.ndarray
    h: dict[str, dict[str, complex]]  # rx -> {"s1": h, "s2": h, "direct": h}
    tx_dbm: float
    noise_w: float
    es_n0_db: dict[str, float]  # rx -> Es/N0 under its own optimal configuration
    diag: dict[str, Any]


def design(alphabet: str, los: str) -> Design:
    """Exact optima for RX1 and RX2 at t = 0 (static scene) and the TX power that gives
    Es/N0 = 25 dB at RX1 under s1 (``Es = P_rx * SPS / f_s``, ``N0 = k T0 F``)."""
    quiet_warnings()
    model = from_scenario(lab_scenario(alphabet, los), backend="numpy_ref")
    snap = model.snapshot(Fraction(0))
    views = {rx: csi_from_snapshot(model, snap, lk) for rx, lk in LINKS.items()}
    _, s1 = exact_alphabet_optimum(views[RX1].h_direct, views[RX1].cascade[PANEL])
    _, s2 = exact_alphabet_optimum(views[RX2].h_direct, views[RX2].cascade[PANEL])
    h: dict[str, dict[str, complex]] = {}
    for rx, v in views.items():
        h[rx] = {
            "s1": v.evaluate({PANEL: s1}),
            "s2": v.evaluate({PANEL: s2}),
            "direct": complex(v.h_direct),
        }
    nw = noise_power_w(float(FS), NF_DB)
    p_rx = 10 ** (ES_N0_DB / 10) * nw / SPS  # W
    p_tx = p_rx / abs(h[RX1]["s1"]) ** 2
    tx_dbm = 10 * math.log10(p_tx / 1e-3)
    es = {
        rx: 10 * math.log10(p_tx * abs(h[rx]["s1" if rx == RX1 else "s2"]) ** 2 * SPS / nw)
        for rx in (RX1, RX2)
    }
    v1 = views[RX1]
    cols = np.arange(s1.size)
    ris1_s1 = complex(v1.cascade[PANEL][s1.astype(int), cols].sum())
    ris1_s2 = complex(v1.cascade[PANEL][s2.astype(int), cols].sum())
    diag = {
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
        "tau_ris_rx1_s": float(model.ris_delay_s(PANEL, RX1, Fraction(0))),
        "tau_ris_rx2_s": float(model.ris_delay_s(PANEL, RX2, Fraction(0))),
    }
    return Design(alphabet, los, s1.astype(np.int16), s2.astype(np.int16), h, tx_dbm, nw, es, diag)


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
) -> tuple[CaptureSink, Emulator, Any]:
    """Lockstep run of ``n_samples`` RX samples with in-process traffic; ``commands`` are
    scheduled before the run (in the given order). With ``adapter_times``, each command
    goes through ``LegacyNextTickAdapter`` at its arrival time."""
    quiet_warnings()
    sink = CaptureSink(n_samples, [RX1, RX2])
    eng = Emulator.from_scenario(resolved, traffic={TX: traffic}, sinks=[sink])
    if adapter_times is None:
        for req in commands:
            rc = eng.schedule(req)
            if not rc.ok:
                raise RuntimeError(f"command {req.cmd_id} rejected: {rc.error} {rc.msg}")
    else:
        shim = _ArrivalEngine(eng)
        ad = LegacyNextTickAdapter(shim, LEGACY_TAU_S, allow_nonphysical=True)
        for req, t_arr in zip(commands, adapter_times, strict=True):
            shim.t = t_arr
            rc = ad.schedule(req)
            if not rc.ok:
                raise RuntimeError(f"command {req.cmd_id} rejected: {rc.error} {rc.msg}")
        eng.legacy_lost = ad.lost  # type: ignore[attr-defined]
    summary = eng.run(Fraction(n_samples) / FS)
    return sink, eng, summary


class _ArrivalEngine:
    """Engine view for ``LegacyNextTickAdapter``: ``time_now`` is the command's arrival."""

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


# =========================================================================== E-B1


def run_eb1_point(ctx: Any) -> dict[str, Any]:
    """E-B1 sweep runner (``experiments.eb:run_eb1_point``). Parameters: ``alphabet``,
    ``los``, ``delta`` (fraction of T_p, string), ``transition`` (step | linear20us |
    linear200us)."""
    quiet_warnings()
    prm = dict(ctx.params)
    alphabet, los = str(prm["alphabet"]), str(prm.get("los", "present"))
    delta = Fraction(str(prm["delta"]))
    trans = str(prm["transition"])
    dsg = design(alphabet, los)
    start = 3000  # packet start sample (3 ms)
    n = start + BODY + 1000
    scen = lab_scenario(
        alphabet,
        los,
        full_scale_dbm=dsg.tx_dbm,
        initial_states=dsg.s1,
        duration_s=Fraction(n) / FS,
        seed=int(ctx.seed),
    )
    res = resolve(scen)
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
    req = CommandRequest(
        "eb1_switch",
        "ris.set_config",
        "eb1",
        PANEL,
        CommandTime.at(t_sw),
        states=dsg.s2,
        transition=TRANSITIONS[trans],
    )
    ref_sink, ref_eng, _ = run_iq(res, src, [], n)
    sink, eng, summ = run_iq(res, src, [req], n)
    recs = [r for r in eng.records if r.cmd_id == "eb1_switch"]
    rec = recs[-1]
    k_app = int(rec.k_applied[RX1])
    tau = float(eng.model.ris_delay_s(PANEL, RX1, t_sw))
    k_expect = to_ticks(t_sw + Fraction(repr(tau)), FS)
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
    va5 = {
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
            else bool(k_first == k_app == k_expect and before_identical)
        ),
    }
    out = {
        "alphabet": alphabet,
        "los": los,
        "delta": str(delta),
        "transition": trans,
        "t_sw_s": str(t_sw),
        "tx_dbm": dsg.tx_dbm,
        "es_n0_db": dsg.es_n0_db,
        "design": dsg.diag,
        "per_packet_evm_db": d_sw["evm_rms_payload_db"],
        "per_packet_evm_pct": d_sw["evm_rms_payload_pct"],
        "bit_errors": d_sw["bit_errors"],
        "ber": d_sw["bit_errors"] / (2 * N_PAY),
        "ref_evm_db": d_ref["evm_rms_payload_db"],
        "ref_bit_errors": d_ref["bit_errors"],
        "command_status": rec.status,
        "va5": va5,
        "verify_interp_max_rel": summ.verify_interp_max_rel,
    }
    write_json(Path(ctx.run_dir) / "result.json", out)
    info = eng.manifest_info()
    run.page_size = info["page_size"]
    run.verify_interp_max_rel = info["verify_interp_max_rel"]
    run.counters = {k: v for k, v in info["counters"].items() if isinstance(v, int)}
    run.add_assumption("lab |Gamma| ASSUMED" if alphabet == "lab" else "ideal 1-bit alphabet")
    run.add_assumption(f"TX power {dsg.tx_dbm:.3f} dBm for Es/N0 = 25 dB at RX1 under s1")
    write_meta(
        ctx.run_dir,
        experiment="E-B1",
        data_classes=["SIM-sremu", "ASSUMED"] if alphabet == "lab" else ["SIM-sremu"],
        description=(
            f"E-B1 mechanism: one packet to RX1, switch to RX2's optimum at {delta} T_p, "
            f"transition {trans}, alphabet {alphabet}, LOS {los}; lab-like SIM geometry"
        ),
        assumptions=run.assumptions + ["transition path complex-linear [ENG]"],
        inputs=run.inputs,
        files={"symbols.csv": "per-symbol EVM (dB) with and without the switch"},
    )
    run.finish()
    return {k: v for k, v in out.items() if k != "design"} | {"design": dsg.diag}


# =========================================================================== E-B2


def packet_start(semantics: str, guard: int, lead: int) -> int:
    """First packet's start sample: guard centres at ``m*slot + lead`` (timed) or guard
    starts at ``m*slot`` (asap, legacy)."""
    if semantics == "timed":
        return lead + guard // 2
    return guard


def run_eb2_point(ctx: Any) -> dict[str, Any]:
    """E-B2 sweep runner (``experiments.eb:run_eb2_point``). Parameters: ``alphabet``,
    ``guard_s`` (string), ``jitter_s`` (J, string), ``semantics`` (timed | asap | legacy),
    ``lead_s`` (timed), ``n_packets`` (default 2000)."""
    quiet_warnings()
    prm = dict(ctx.params)
    alphabet = str(prm.get("alphabet", "lab"))
    los = str(prm.get("los", "present"))
    g_s = Fraction(str(prm["guard_s"]))
    j_s = Fraction(str(prm["jitter_s"]))
    sem = str(prm["semantics"])
    lead_s = Fraction(str(prm.get("lead_s", "5e-4")))
    n_pk = int(prm.get("n_packets", 2000))
    guard = int(g_s * FS)
    lead = int(lead_s * FS)
    assert guard % 2 == 0 and guard * 1 == g_s * FS
    slot = BODY + guard
    slot_s = Fraction(slot) / FS
    start = packet_start(sem, guard, lead)
    n = start + n_pk * slot + 2000
    dsg = design(alphabet, los)
    scen = lab_scenario(
        alphabet,
        los,
        full_scale_dbm=dsg.tx_dbm,
        initial_states=dsg.s2,
        duration_s=Fraction(n) / FS,
        seed=int(ctx.seed),
    )
    res = resolve(scen)
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
    ctrl = TdmaAlternate(
        [dsg.s1, dsg.s2],
        slot_s=slot_s,
        timing=sem,
        lead_s=lead_s if sem == "timed" else LAT_FIXED_S,
        panel=PANEL,
        name=CTRL_NAME,
        min_latency_s=LAT_FIXED_S,
        latency=lat,
    )
    from sremu.physics.propagation import from_scenario as _fs

    caps = {PANEL: _fs(scen, backend="numpy_ref").capabilities(PANEL)}
    setup_controller(ctrl, caps, seed)
    cmds = []
    for m in range(n_pk):
        t_b = m * slot_s
        (a,) = ctrl.decide(make_observation(t_b, ris_state={PANEL: dsg.s1}))
        L = draw_latency_s(seed, CTRL_NAME, lat, m, 0)
        t_arr = t_b + L
        if sem == "timed":
            t_req = Fraction(a.at_s)
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
                "states": np.asarray(a.states, dtype=np.int16),
            }
        )
    order = sorted(range(n_pk), key=lambda i: (cmds[i]["t_arrival"], i))
    reqs = []
    for i in order:
        c = cmds[i]
        reqs.append(
            CommandRequest(
                f"tdma:{c['m']}",
                "ris.set_config",
                CTRL_NAME,
                PANEL,
                CommandTime.at(c["t_eff"]),
                states=c["states"],
            )
        )
    sink, eng, summ = run_iq(
        res,
        src,
        reqs,
        n,
        adapter_times=[cmds[i]["t_arrival"] for i in order] if sem == "legacy" else None,
    )
    recs = {r.cmd_id: r for r in eng.records}
    rows = []
    n_err = 0
    evm_lin = []
    for m in range(n_pk):
        c = cmds[m]
        rx = c["target"]
        st = start + m * slot
        d = demod(sink.buf[rx], st, src.symbols(m))
        r = recs.get(f"tdma:{m}")
        k_app = None if r is None or not r.k_applied else r.k_applied.get(rx)
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
                "status": None if r is None else r.status,
                "t_applied_s": None if r is None or r.t_applied is None else float(r.t_applied),
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
    statuses: dict[str, int] = {}
    for r in eng.records:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    out = {
        "alphabet": alphabet,
        "los": los,
        "guard_s": str(g_s),
        "jitter_s": str(j_s),
        "semantics": sem,
        "lead_s": str(lead_s) if sem == "timed" else None,
        "n_packets": n_pk,
        "packet_errors": n_err,
        "per": n_err / n_pk,
        "per_ci_low": lo,
        "per_ci_high": hi,
        "mean_evm_db": db10(float(np.mean(evm_lin))),
        "n_late": int(sum(c["late"] for c in cmds)),
        "record_status_counts": statuses,
        "legacy_lost": getattr(eng, "legacy_lost", None),
        "tx_dbm": dsg.tx_dbm,
        "es_n0_db": dsg.es_n0_db,
        "verify_interp_max_rel": summ.verify_interp_max_rel,
        "analytic_expectation": analytic_per(sem, g_s, j_s, lead_s),
    }
    write_json(Path(ctx.run_dir) / "result.json", out)
    info = eng.manifest_info()
    run.page_size = info["page_size"]
    run.verify_interp_max_rel = info["verify_interp_max_rel"]
    run.counters = {k: v for k, v in info["counters"].items() if isinstance(v, int)}
    run.add_assumption("lab |Gamma| ASSUMED" if alphabet == "lab" else "ideal 1-bit alphabet")
    run.add_assumption(
        "command arrival = issue + keyed latency draw; timed: switch at max(t_req, arrival) "
        "(apply_on_arrival); asap: at arrival; legacy: LegacyNextTickAdapter at arrival"
    )
    write_meta(
        ctx.run_dir,
        experiment="E-B2",
        data_classes=["SIM-sremu", "ASSUMED"] if alphabet == "lab" else ["SIM-sremu"],
        description=(
            f"E-B2 timing requirement: {n_pk} packets alternating RX1/RX2, guard {g_s} s, "
            f"latency 0.2 ms + U[0, {j_s}] s, semantics {sem}"
            + (f" (lead {lead_s} s)" if sem == "timed" else "")
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
    legacy: not closed-form here (reported as None)."""
    lf = LAT_FIXED_S

    def p_gt(x: Fraction) -> float:
        if j == 0:
            return 1.0 if lf > x else 0.0
        return float(min(1, max(0, (lf + j - x) / j)))

    if sem == "timed":
        return {"p_switch_in_packet": p_gt(lead + g / 2), "rule": "P(L > lead + G/2)"}
    if sem == "asap":
        return {"p_switch_in_packet": p_gt(g), "rule": "P(L > G)"}
    return {"p_switch_in_packet": None, "rule": "next 2 ms tick after arrival"}
