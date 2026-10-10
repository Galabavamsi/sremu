"""Batch (coefficient-only) path (DESIGN.md section 4.11; VA-1, VA-4, E-A, E-C).

:func:`run_batch` evaluates ``combine(snapshot(t), cfg(t))`` on a time grid and returns
per-link narrowband coefficients ``h(t) = sum_p a_p exp(-j 2 pi f_c tau_p)``, ``|h|**2`` and
their time averages. It uses the **same** ``PropagationModel``, panel command semantics
(:mod:`sremu.engine.commands`), controller barriers and keyed latency draws
(:mod:`sremu.engine.barriers`) and RNG keys as the IQ engine.

* **Grid.** ``dt_s`` if given, else the section 4.4 snapshot period (``T_u``); a static
  scene is evaluated only where something changes. Barrier times and switch times split
  the grid; inside each piece the configuration is constant (or follows its transition,
  evaluated at the piece start in RIS time). A running non-step transition adds its own
  sub-grid (``T/64`` over a linear ramp, ``T/8`` over the first ``32 T`` of a first-order
  profile), so the pieces follow the profile instead of holding the start mixture.
* **Command mapping.** The RIS-time convention: a switch at ``t_sw`` takes effect at
  ``t_sw`` for every receiver; the ns-scale ``tau_RIS->r`` is negligible here and is
  reported (``tau_ris_s``).
* **Weights.** Each piece holds its left-point value; it is weighted by its duration, or,
  with ``rate``, by ``count(t_i, t_{i+1}, rate)`` RX samples (R4), which matches a sampled
  receiver exactly (VA-4).
* **Agreement contract (4.11).** Noise off, paired trajectories: IQ time-averaged tone power
  vs batch within 0.02 dB (VA-1); noise on: within the 95 % CI of batch means; tau_SUI -> 0
  with an in-process controller: equal to batch at 1e-9 (VA-4).
* **Observations** for controllers carry ``rx_power_dbm`` from the channel itself
  (``10 log10(sum_l P_tx,l |h_l|**2 / 1 mW)``, averaged over ``window_s``; the batch path
  has no noise), positions, oracle CSI and the RIS state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from sremu.channel.interp import choose_policy_for_model, frac_of_product
from sremu.contracts import (
    CommandRecord,
    CommandRequest,
    Controller,
    LinkId,
    Observation,
    PanelConfig,
    PanelId,
    PropagationModel,
    StreamId,
)
from sremu.errors import ConfigurationError, ErrorCode, UnknownTarget
from sremu.timebase import count, parse_exact, to_ticks

from .barriers import BarrierManager, ControllerSlot, action_request, csi_view
from .commands import (
    CommandManager,
    PanelCommand,
    PanelSpec,
    RxPort,
    mixture_amplitudes,
    panel_combine_fn,
)
from .streams import StreamConfig, stream_configs_from_model

__all__ = ["BatchResult", "run_batch"]

_BATCH_RX = "batch.ris"  # pseudo receiver of the RIS-time convention


@dataclass
class BatchResult:
    """Output of :func:`run_batch`.

    ``t``: piece start times (exact); ``w``: piece weights (s, or samples with ``rate``);
    ``h``: per link complex coefficients at the piece starts; ``h2``: ``|h|**2``;
    ``mean_h2``: weighted time averages; ``rx_power_w``: per RX stream
    ``sum_l P_tx,l |h_l|**2`` per piece and ``mean_rx_power_w`` its average (W);
    ``records``: command records; ``tau_ris_s``: tau_RIS->r per (panel, rx) at t = 0 (s),
    the quantity the RIS-time convention neglects.
    """

    t: list[Fraction]
    w: NDArray[np.float64]
    h: dict[LinkId, NDArray[np.complex128]]
    h2: dict[LinkId, NDArray[np.float64]]
    mean_h2: dict[LinkId, float]
    rx_power_w: dict[StreamId, NDArray[np.float64]]
    mean_rx_power_w: dict[StreamId, float]
    records: list[CommandRecord]
    tau_ris_s: dict[str, float] = field(default_factory=dict)
    weights: str = "time"

    def mean_rx_power_dbm(self, rx: StreamId) -> float:
        """Average RX power in dBm."""
        p = self.mean_rx_power_w[rx]
        return 10.0 * math.log10(p / 1e-3) if p > 0 else -math.inf

    def times_s(self) -> NDArray[np.float64]:
        """Piece start times as float64 seconds (plots only)."""
        return np.array([float(x) for x in self.t], dtype=np.float64)


_LINEAR_STEPS = 64  # sub-pieces of a linear transition (step T/64)
_FIRST_ORDER_SPAN = 32  # a first-order transition is followed for 32 T (g = 1 - 1.3e-14)
_FIRST_ORDER_STEP = Fraction(1, 8)  # in units of T


def _transition_split(cmd: PanelCommand, t: Fraction) -> Fraction | None:
    """Next sub-grid point after ``t`` of a command's running non-step transition (None when
    it is a step, moves nothing, or has completed): ``T/64`` steps over a linear ramp and
    ``T/8`` steps over the first ``32 T`` of a first-order profile, so the left-point pieces
    follow ``g`` instead of holding the start mixture until the next grid point."""
    if cmd.is_step or not cmd.classes or not cmd.duration_s:
        return None
    dur = Fraction(cmd.duration_s)
    if cmd.profile == "linear":
        span, step = dur, dur / _LINEAR_STEPS
    else:
        span, step = _FIRST_ORDER_SPAN * dur, _FIRST_ORDER_STEP * dur
    dt = t - cmd.t
    if dt >= span:
        return None
    return cmd.t + (math.floor(dt / step) + 1) * step


def _tx_power_w(cfg: StreamConfig, digital_power: float) -> float:
    """TX power at the antenna port for a digital signal of mean power ``digital_power``."""
    return digital_power * 1e-3 * 10.0 ** (cfg.full_scale_dbm / 10.0) * 10.0 ** (cfg.gain_db / 10.0)


def run_batch(
    model: PropagationModel,
    duration_s: Fraction | int | str,
    *,
    dt_s: Fraction | int | str | None = None,
    rate: Fraction | int | str | None = None,
    panels: Mapping[PanelId, PanelSpec] | None = None,
    streams: Mapping[StreamId, StreamConfig] | None = None,
    controllers: Sequence[Controller] = (),
    commands: Sequence[CommandRequest] = (),
    seed: int = 0,
    tx_digital_power: Mapping[StreamId, float] | None = None,
    late_policy: str = "apply_on_arrival",
    eps: float = 1e-3,
) -> BatchResult:
    """Coefficient-only evaluation of a model over ``[0, duration_s)`` (section 4.11).

    ``commands``: timed requests scheduled up front (on time). ``controllers``: in-process
    barrier controllers (same barrier times, keyed latencies and action semantics as the IQ
    engine). ``tx_digital_power``: mean digital power per TX stream (default 1, a unit
    tone) for ``rx_power_w``. ``rate``: weight pieces by RX sample counts at this rate.
    """
    T = parse_exact(duration_s, name="duration_s")
    if T <= 0:
        raise ConfigurationError("duration_s must be > 0")
    rate_q = None if rate is None else parse_exact(rate, name="rate")
    configs = dict(streams if streams is not None else stream_configs_from_model(model))
    links = model.links()
    fcs = {lk: float(model.link_fc_hz(lk)) for lk in links}
    grid: Fraction | None
    if dt_s is not None:
        grid = parse_exact(dt_s, name="dt_s")
    else:
        rx_rates = [c.rate for c in configs.values() if c.direction == "rx"]
        pol = choose_policy_for_model(model, max(rx_rates, default=Fraction(1)), eps=eps)
        grid = pol.t_u_s  # None: static
    specs = dict(panels) if panels is not None else None
    if specs is None:
        from .commands import panel_specs_from_model

        specs = panel_specs_from_model(model)
    snap0 = model.snapshot(Fraction(0))
    n_tiles = {pid: rc.n_tiles for pid, rc in snap0.ris.items()}
    ref_rate = rate_q if rate_q is not None else Fraction(10**9)
    mgr = CommandManager(
        specs, [RxPort(_BATCH_RX, ref_rate)], model.ris_delay_s, n_tiles, reference="rx"
    )
    tau_ris: dict[str, float] = {}
    for pid in snap0.ris:
        for sid, c in configs.items():
            if c.direction == "rx":
                tau_ris[f"{pid}/{sid}"] = float(model.ris_delay_s(pid, sid, Fraction(0)))
    barriers = BarrierManager(seed)
    for ctrl in controllers:
        barriers.register(ctrl, Fraction(0), mgr.caps())
    for req in commands:
        out = mgr.schedule(req, {_BATCH_RX: 0}, late_policy=late_policy)
        if not out.receipt.ok:
            raise ConfigurationError(f"command {req.cmd_id!r} rejected: {out.receipt.error.name}")
    tx_p = {
        sid: _tx_power_w(c, float((tx_digital_power or {}).get(sid, 1.0)))
        for sid, c in configs.items()
        if c.direction == "tx"
    }
    rx_of = {lk: lk.split("->")[1] for lk in links}
    tx_of = {lk: lk.split("->")[0] for lk in links}

    t_list: list[Fraction] = []
    w_list: list[float] = []
    h_list: dict[LinkId, list[complex]] = {lk: [] for lk in links}
    pw_list: dict[StreamId, list[float]] = {
        sid: [] for sid, c in configs.items() if c.direction == "rx"
    }

    def power_window(t_b: Fraction, window: Fraction) -> dict[StreamId, float]:
        lo = t_b - window
        acc = dict.fromkeys(pw_list, 0.0)
        tot = 0.0
        for i, t in enumerate(t_list):
            t_end = t_list[i + 1] if i + 1 < len(t_list) else t_b
            a, b = max(t, lo), min(t_end, t_b)
            if b <= a:
                continue
            wgt = float(b - a)
            tot += wgt
            for r in acc:
                acc[r] += wgt * pw_list[r][i]
        return {
            r: (10.0 * math.log10(v / tot / 1e-3) if tot > 0 and v > 0 else math.nan)
            for r, v in acc.items()
        }

    def run_barrier(s: ControllerSlot) -> None:
        t_b = s.t_next
        s.t_b_last = t_b
        flds = set(s.spec.fields)
        positions = None
        if "positions" in flds:
            node_pos = getattr(model, "node_position", None)
            if node_pos is None:
                raise ConfigurationError("this model cannot report node positions")
            nodes = sorted({c.node for c in configs.values()})
            noise = barriers.position_noise(s, nodes)
            tp = max(t_b - s.spec.position_delay_s, Fraction(0))
            positions = {n: np.asarray(node_pos(n, tp), np.float64) + noise[n] for n in nodes}
        csi = None
        if "csi_oracle" in flds:
            snap = model.snapshot(t_b)
            csi = {lk: csi_view(model, snap, lk, fcs[lk]) for lk in links}
        obs = Observation(
            t_b=t_b,
            k_b={r: to_ticks(t_b, configs[r].rate) for r in pw_list},
            rx_power_dbm=power_window(t_b, s.spec.window_s),
            positions=positions,
            csi_oracle=csi,
            iq_window=None,
            ris_state=mgr.snapshot_states(t_b),
            pending=mgr.pending_receipts(t_b),
        )
        ctrl = s.ctrl
        assert isinstance(ctrl, Controller)
        for i, a in enumerate(ctrl.decide(obs)):
            at = barriers.action_time(s, t_b, i, a)
            req = action_request(s.name, s.m, i, a, at)
            fr = {_BATCH_RX: to_ticks(t_b, ref_rate)}
            out = mgr.schedule(req, fr, late_policy=late_policy)
            if not out.receipt.ok:
                if out.receipt.error == ErrorCode.UNKNOWN_TARGET:
                    raise UnknownTarget(out.receipt.msg)
                raise ConfigurationError(f"{s.name}: action rejected: {out.receipt.msg}")
        s.m += 1

    t = Fraction(0)
    while t < T:
        # barriers due at t run before the piece starting at t
        while barriers.slots:
            s = min(barriers.slots, key=lambda x: (x.t_next, x.order))
            if s.t_next <= t:
                run_barrier(s)
            else:
                break
        nxt = [T]
        if grid is not None:
            nxt.append((math.floor(t / grid) + 1) * grid)
        if barriers.slots:
            nxt.append(min(s.t_next for s in barriers.slots))
        for book in mgr.books.values():
            for cmd in book.active_config() + book.active_enabled():
                if cmd.t > t:
                    nxt.append(cmd.t)
            # a running non-step transition is followed on its own sub-grid (left points)
            run_ = [c for c in book.active_config() if c.t <= t]
            if run_:
                p = _transition_split(run_[-1], t)
                if p is not None:
                    nxt.append(p)
        t_end = min(nxt)
        snap = model.snapshot(t)
        # configuration at RIS time t (mixtures for running transitions)
        amps: dict[LinkId, NDArray[np.complex128]] | None = None
        cfg: dict[PanelId, PanelConfig] = {}
        mixed = False
        for pid, book in mgr.books.items():
            cfg[pid] = book.config_at(t)
            act = [c for c in book.active_config() if c.t <= t]
            if act and not act[-1].is_step:
                mixed = True
        paths = model.combine(snap, cfg)
        if mixed:
            amps = {lk: lp.amp.copy() for lk, lp in paths.items()}
            for pid, book in mgr.books.items():
                act = [c for c in book.active_config() if c.t <= t]
                if not act or act[-1].is_step:
                    continue
                mix = act[-1].frozen(t - act[-1].t)
                fn = panel_combine_fn(model, snap, pid)
                cur = fn(cfg[pid].state)
                new = mixture_amplitudes(mix, book.spec.n_states, fn)
                en = 1.0 if cfg[pid].enabled else 0.0
                for lk in amps:
                    amps[lk] = amps[lk] + en * (new[lk] - cur[lk])
        hv: dict[LinkId, complex] = {}
        for lk, lp in paths.items():
            a = lp.amp if amps is None else amps[lk]
            cyc = frac_of_product(fcs[lk], lp.delay_s)
            hv[lk] = complex(np.sum(a * np.exp(-2j * np.pi * cyc)))
        wgt = float(count(t, t_end, rate_q)) if rate_q is not None else float(t_end - t)
        t_list.append(t)
        w_list.append(wgt)
        for lk in links:
            h_list[lk].append(hv[lk])
        for r in pw_list:
            pw_list[r].append(
                sum(tx_p[tx_of[lk]] * abs(hv[lk]) ** 2 for lk in links if rx_of[lk] == r)
            )
        t = t_end
        mgr.update_statuses({_BATCH_RX: to_ticks(t, ref_rate)})

    w = np.asarray(w_list, dtype=np.float64)
    h = {lk: np.asarray(v, dtype=np.complex128) for lk, v in h_list.items()}
    h2 = {lk: np.abs(v) ** 2 for lk, v in h.items()}
    wsum = float(w.sum())
    mean_h2 = {lk: float(np.dot(w, v) / wsum) for lk, v in h2.items()}
    pw = {r: np.asarray(v, dtype=np.float64) for r, v in pw_list.items()}
    mean_pw = {r: float(np.dot(w, v) / wsum) for r, v in pw.items()}
    return BatchResult(
        t_list,
        w,
        h,
        h2,
        mean_h2,
        pw,
        mean_pw,
        list(mgr.records),
        tau_ris,
        "samples" if rate_q is not None else "time",
    )
