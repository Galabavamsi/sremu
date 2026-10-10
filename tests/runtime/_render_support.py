"""Shared helpers for the WP5 render tests (not collected: no ``test_`` prefix).

* :func:`render_span` drives a backend over consecutive snapshot intervals of a
  ``PropagationModel`` (a minimal stand-in for the WP6 job builder), optionally splitting
  segments at extra chunk edges (I2a / PH-19).
* :func:`ris_brute` evaluates the simplified ``FakePanel`` RIS channel per element per
  sample at exact positions (the PH-18 / RT-12 brute force), independent of the renderer.
* Scene builders for the PH-18 scenarios (SIM geometry only).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from sremu.channel.fd import kaiser_fd_spec
from sremu.channel.interp import PanelGeometry, make_link_job, plan_link
from sremu.channel.render_numpy import get_backend
from sremu.channel.ring import TxRingBuffer
from sremu.contracts import (
    FDSpec,
    LinkId,
    LinkPaths,
    PanelConfig,
    PanelId,
    PathKind,
    PropagationModel,
    SegmentJob,
    TransitionTerm,
)
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import stream
from sremu.timebase import ceil_ticks
from sremu.units import C0

DC_BW_HZ = 1.0  # a DC tone: integer-only delay is allowed


def const_ring(stream_id: str, f_s: Fraction, n: int, value: complex = 1.0 + 0.0j) -> TxRingBuffer:
    """A ring holding ``n`` samples of a constant (DC tone)."""
    ring = TxRingBuffer(stream_id, f_s, 1 << max(10, math.ceil(math.log2(n + 1))))
    ring.write(0, np.full(n, value, dtype=np.complex64))
    return ring


def select_paths(lp: LinkPaths, kinds: Sequence[int] | None) -> LinkPaths:
    """Keep only paths of the given kinds (None keeps everything)."""
    if kinds is None:
        return lp
    sel = np.isin(lp.kind, np.asarray(kinds, dtype=np.int8))
    return LinkPaths(lp.link, lp.t, lp.amp[sel], lp.delay_s[sel], lp.kind[sel], lp.tile[sel])


@dataclass
class SpanSpec:
    """What :func:`render_span` renders."""

    model: PropagationModel
    cfg: Mapping[PanelId, PanelConfig]
    link: LinkId
    fc_hz: float
    ring: TxRingBuffer
    f_s: Fraction
    t_u: Fraction
    occupied_bw_hz: float = DC_BW_HZ
    kinds: Sequence[int] | None = None
    lo_offset_hz: float = 0.0
    fd: FDSpec | None = None
    trans_fn: Callable[[int, LinkPaths, LinkPaths], list[TransitionTerm]] | None = None


def render_span(
    spec: SpanSpec,
    k_lo: int,
    k_hi: int,
    backend: str = "numpy_ref",
    chunk_edges: Sequence[int] = (),
) -> np.ndarray:
    """Render RX samples [k_lo, k_hi) of one link over the T_u snapshot grid."""
    be = get_backend(backend)  # type: ignore[arg-type]
    out = np.zeros(k_hi - k_lo, dtype=np.complex128)
    m0 = math.floor(Fraction(k_lo) / spec.f_s / spec.t_u)
    edges = sorted({int(e) for e in chunk_edges if k_lo < e < k_hi})
    cache: dict[int, Mapping[LinkId, LinkPaths]] = {}

    def paths(m: int) -> LinkPaths:
        if m not in cache:
            snap = spec.model.snapshot(m * spec.t_u)
            cache[m] = spec.model.combine(snap, spec.cfg)
        return select_paths(cache[m][spec.link], spec.kinds)

    m = m0
    while True:
        t_a = m * spec.t_u
        t_b = (m + 1) * spec.t_u
        n_first = ceil_ticks(t_a, spec.f_s)
        n_end = ceil_ticks(t_b, spec.f_s)
        if n_first >= k_hi:
            break
        a, b = max(n_first, k_lo), min(n_end, k_hi)
        if a < b:
            pa, pb = paths(m), paths(m + 1)
            plan = plan_link(pa, pb, spec.occupied_bw_hz, spec.f_s)
            if spec.fd is not None:
                plan = type(plan)(plan.narrowband, plan.group, spec.fd, plan.spread_s)
            trans = spec.trans_fn(m, pa, pb) if spec.trans_fn is not None else []
            lj = make_link_job(
                spec.link, spec.ring, spec.fc_hz, spec.lo_offset_hz, pa, pb, plan, trans
            )
            cuts = [a] + [e for e in edges if a < e < b] + [b]
            for s, e in itertools.pairwise(cuts):
                job = SegmentJob(rx="rx", f_s=spec.f_s, k0=s, k1=e, t_a=t_a, t_b=t_b, links=[lj])
                tmp = np.zeros(e - s, dtype=np.complex128)
                be.render(job, tmp)
                out[s - k_lo : e - k_lo] = tmp
        m += 1
    return out


def ris_brute(
    model: FakePropagation,
    panel: PanelId,
    link: LinkId,
    gamma_of_cell: Callable[[np.ndarray], np.ndarray] | np.ndarray,
    t: np.ndarray,
    fc_hz: float,
) -> np.ndarray:
    """Per-element per-sample RIS channel of a ``FakePanel`` at times ``t`` (s, float64).

    ``gamma_of_cell`` is the (N,) per-cell Gamma, or a function of the time index array
    returning (len(t), N). Coefficient per cell: ``A*j/(4 pi r_i r_s)*exp(-jk(r_i + r_s))``.
    """
    p = model.panels[panel]
    tx, rx = model.link_streams(link)
    ntx, nrx = model.nodes[tx.node], model.nodes[rx.node]
    cells = p.cell_centres()
    area = p.pitch_u_m * p.pitch_v_m
    k = 2.0 * math.pi * fc_hz / C0
    out = np.zeros(t.shape[0], dtype=np.complex128)
    for i0 in range(0, t.shape[0], 512):
        tt = t[i0 : i0 + 512]
        ptx = (
            np.asarray(ntx.position_m)[None, :]
            + np.asarray(ntx.velocity_mps)[None, :] * tt[:, None]
        )
        prx = (
            np.asarray(nrx.position_m)[None, :]
            + np.asarray(nrx.velocity_mps)[None, :] * tt[:, None]
        )
        ri = np.linalg.norm(cells[None, :, :] - ptx[:, None, :], axis=2)
        rs = np.linalg.norm(cells[None, :, :] - prx[:, None, :], axis=2)
        coef = area * 1j / (4.0 * math.pi * ri * rs) * np.exp(-1j * k * (ri + rs))
        g = (
            gamma_of_cell(np.arange(i0, i0 + tt.shape[0]))
            if callable(gamma_of_cell)
            else gamma_of_cell[None, :]
        )
        out[i0 : i0 + tt.shape[0]] = np.sum(coef * g, axis=1)
    return out


def focus_states(
    model: FakePropagation, panel: PanelId, link: LinkId, t_f: float, fc_hz: float, bits: int = 4
) -> np.ndarray:
    """int16 states of a 2**bits-PSK alphabet that co-phase every cell at time ``t_f``."""
    p = model.panels[panel]
    tx, rx = model.link_streams(link)
    ptx = model.nodes[tx.node].position(Fraction(t_f).limit_denominator(10**9))
    prx = model.nodes[rx.node].position(Fraction(t_f).limit_denominator(10**9))
    cells = p.cell_centres()
    k = 2.0 * math.pi * fc_hz / C0
    ph = np.angle(
        1j
        * np.exp(
            -1j * k * (np.linalg.norm(cells - ptx, axis=1) + np.linalg.norm(cells - prx, axis=1))
        )
    )
    s = np.rint(-ph / (2 * math.pi / 2**bits)).astype(np.int64) % 2**bits
    return s.astype(np.int16)


def psk_alphabet(bits: int) -> tuple[complex, ...]:
    """``exp(j 2 pi s / 2**bits)`` for s = 0..2**bits - 1."""
    m = 2**bits
    return tuple(complex(np.exp(2j * np.pi * s / m)) for s in range(m))


@dataclass(frozen=True)
class MobileScene:
    """A PH-18 scene: builder for a given tile grid, the RIS link and the check window."""

    name: str
    fc_hz: float
    build: Callable[[int, int], FakePropagation]
    link: LinkId
    panel: PanelId
    t_focus_s: float
    window_s: tuple[float, float]
    f_s: Fraction

    def geometry(self, model: FakePropagation) -> PanelGeometry:
        p = model.panels[self.panel]
        return PanelGeometry(
            self.panel,
            p.n_u * p.pitch_u_m,
            p.n_v * p.pitch_v_m,
            p.n_u,
            p.n_v,
            self.fc_hz,
            len(model.links()),
        )


def _scene(
    fc: float,
    rate: str,
    tx_pos: tuple[float, float, float],
    rx_pos: tuple[float, float, float],
    rx_vel: tuple[float, float, float],
    n_u: int,
    n_v: int,
    pitch: float,
    horizon: str,
    tiles: tuple[int, int],
    bits: int = 4,
) -> FakePropagation:
    nodes = [
        FakeNode("tx", tx_pos, streams=(stream("tx.tx0", "tx", fc, rate),)),
        FakeNode("rx", rx_pos, rx_vel, streams=(stream("rx.rx0", "rx", fc, rate),)),
    ]
    panel = FakePanel(
        "ris0",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        n_u,
        n_v,
        pitch,
        pitch,
        gamma=psk_alphabet(bits),
        tiles_u=tiles[0],
        tiles_v=tiles[1],
    )
    return FakePropagation(nodes, [panel], horizon_s=horizon)


def lab_walk() -> MobileScene:
    """3.5 GHz, 1.5 m/s, R_min 0.45 m, 1.08 m panel of 25 x 25 cells."""
    fc = 3.5e9
    return MobileScene(
        "lab_walk",
        fc,
        lambda tu, tv: _scene(
            fc,
            "2e4",
            (1.5, -1.0, 0.3),
            (0.45, -0.3, 0.0),
            (0.0, 1.5, 0.0),
            25,
            25,
            1.08 / 25,
            "4/10",
            (tu, tv),
        ),
        "tx.tx0->rx.rx0",
        "ris0",
        0.2,
        (0.17, 0.23),
        Fraction(20_000),
    )


def vehicle() -> MobileScene:
    """3.5 GHz, 30 m/s, R_min 5 m, 0.4 m panel of 9 x 9 cells."""
    fc = 3.5e9
    return MobileScene(
        "vehicle",
        fc,
        lambda tu, tv: _scene(
            fc,
            "2e4",
            (20.0, -10.0, 2.0),
            (5.0, -1.5, 0.0),
            (0.0, 30.0, 0.0),
            9,
            9,
            0.4 / 9,
            "1/10",
            (tu, tv),
        ),
        "tx.tx0->rx.rx0",
        "ris0",
        0.05,
        (0.035, 0.065),
        Fraction(20_000),
    )


def wcl_corridor() -> MobileScene:
    """1 GHz, 1 m/s, h = 2 m, a 10 m line of 67 cells (lambda/2 pitch)."""
    fc = 1e9
    return MobileScene(
        "wcl_corridor",
        fc,
        lambda tu, tv: _scene(
            fc,
            "5e3",
            (4.0, -3.0, 0.0),
            (2.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            67,
            1,
            0.15,
            "2",
            (tu, tv),
        ),
        "tx.tx0->rx.rx0",
        "ris0",
        1.0,
        (0.85, 1.15),
        Fraction(5_000),
    )


KAISER32 = kaiser_fd_spec(32)
RIS_KINDS = (int(PathKind.RIS_TILE),)
