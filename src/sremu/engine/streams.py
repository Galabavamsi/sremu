"""Stream runtime: TX ingestion, RX delivery and in-process traffic (DESIGN.md 3.3, 4.8, 6.3).

**TX chain** (section 6.3): full scale/gain -> TX LO (CFO, Wiener PN) -> ring buffer.
Samples submitted by a client (or produced by an in-process :class:`TrafficSource`) are
kept in a pending queue at their absolute indices and written into the TX history ring
(:class:`sremu.channel.ring.TxRingBuffer`) only when the renderer needs them, so a client
may send far ahead of the receivers without overwriting history the renderer still reads.

**TX commitment** (section 4.8). A TX stream is committed through sample K when its samples
cover K, when its last burst ended with EOB (silence until its next timed burst), after
``idle_until(K)``, or always when it has an in-process source. Unopened streams without a
source are silent (``unopened_tx="silent"``) or never committed (``"wait"``).

**Late TX data.** A timed burst that starts before the ring's committed head (samples the
renderer already used) is dropped: counted in ``late_samples`` with an async
``TIME_ERROR`` (UHD semantics). Untimed sends start at
``max(data end, ring head, frontier)``.

**RX chain** (section 6.3): sum of links (render) -> diffuse (paths) -> thermal noise ->
RX LO (CFO, PN) -> gain/full scale -> optional clip -> deliver. Every step is a pure
function of the absolute sample index, so chunking cannot change a bit (I2a).

**RX delivery** follows UHD stream modes (``start_cont``, ``stop_cont``, ``num_done``,
``num_more``); a stream command in the past returns ``LATE_COMMAND`` on the next read. In
paced mode a full RX queue drops the page (``overflow_pages``) and the next read after the
gap returns ``OVERFLOW`` with the post-gap ``time_spec``.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from sremu import rng
from sremu.channel.ring import TxRingBuffer
from sremu.contracts import (
    AsyncMetadata,
    BackendName,
    Direction,
    RxMetadata,
    StreamCmd,
    StreamId,
    StreamSpec,
    TxMetadata,
    split_stream_id,
)
from sremu.errors import AsyncCode, ConfigurationError, ErrorCode, StreamError, ValidityWarning
from sremu.impair.frontend import RxFrontEnd, TxFrontEnd
from sremu.impair.noise import noise as _noise
from sremu.impair.noise import thermal_sigma
from sremu.impair.oscillator import WienerPhaseNoise, cfo_phasor
from sremu.timebase import TimeSpec, parse_exact, to_ticks

__all__ = [
    "ArraySource",
    "BlockCache",
    "HistoryRing",
    "QpskPacketSource",
    "RxStream",
    "StreamConfig",
    "ToneSource",
    "TrafficSource",
    "TxStream",
    "bpsk_v1_waveform",
    "stream_configs_from_model",
    "stream_configs_from_scenario",
    "traffic_from_scenario",
]


# =========================================================================== configuration


@dataclass(frozen=True)
class StreamConfig:
    """Engine-side settings of one stream (scenario ``nodes[].streams[]``, section 9.2).

    ``rate``: exact S/s. ``occupied_bw_hz``: declared occupied band B of a TX stream (None:
    ``0.8*f_s``, with a ``ValidityWarning``). ``full_scale_dbm``/``gain_db``: device power
    mapping (section 3.3). ``noise_figure_db`` (RX), ``cfo_hz`` and ``linewidth_hz`` of the
    stream's LO, ``adc_clip`` (RX).
    """

    stream: StreamId
    direction: Direction
    fc_hz: float
    rate: Fraction
    occupied_bw_hz: float | None = None
    full_scale_dbm: float = 0.0
    gain_db: float = 0.0
    noise_figure_db: float = 0.0
    cfo_hz: float = 0.0
    linewidth_hz: float = 0.0
    adc_clip: bool = False

    def __post_init__(self) -> None:
        split_stream_id(self.stream)
        if self.direction not in ("tx", "rx"):
            raise ConfigurationError(f"{self.stream}: direction must be tx or rx")
        object.__setattr__(self, "rate", parse_exact(self.rate, name=f"{self.stream}.rate"))
        if self.rate <= 0:
            raise ConfigurationError(f"{self.stream}: rate must be > 0")

    @property
    def node(self) -> str:
        """Node id."""
        return split_stream_id(self.stream)[0]

    @property
    def bandwidth_hz(self) -> float:
        """Declared occupied bandwidth B (default ``0.8*f_s``)."""
        return 0.8 * float(self.rate) if self.occupied_bw_hz is None else float(self.occupied_bw_hz)

    def spec(self) -> StreamSpec:
        """The contract ``StreamSpec`` of this stream."""
        return StreamSpec(self.stream, self.direction, self.fc_hz, self.rate)

    @classmethod
    def from_spec(cls, spec: StreamSpec, **kw: Any) -> StreamConfig:
        """Defaults for a model ``StreamSpec`` (keyword overrides)."""
        return cls(spec.stream, spec.direction, float(spec.fc_hz), Fraction(spec.rate_sps), **kw)


def stream_configs_from_model(
    model: Any, overrides: Mapping[StreamId, Mapping[str, Any]] | None = None
) -> dict[StreamId, StreamConfig]:
    """Stream configs for every stream of a model (``model.streams``), with overrides."""
    streams: Mapping[StreamId, StreamSpec] = model.streams
    out: dict[StreamId, StreamConfig] = {}
    for sid, spec in streams.items():
        kw = dict((overrides or {}).get(sid, {}))
        bw = None
        nodes = getattr(model, "nodes", None)
        if nodes is not None and "occupied_bw_hz" not in kw:
            nd = nodes.get(spec.node)
            occ = getattr(nd, "occupied_bw_hz", None)
            if isinstance(occ, Mapping):
                bw = occ.get(sid)
        if bw is not None:
            kw.setdefault("occupied_bw_hz", float(bw))
        out[sid] = StreamConfig.from_spec(spec, **kw)
    return out


def stream_configs_from_scenario(scenario: Mapping[str, Any]) -> dict[StreamId, StreamConfig]:
    """Stream configs from a scenario mapping (``model_dump`` of the pydantic model)."""
    out: dict[StreamId, StreamConfig] = {}
    for nd in scenario.get("nodes") or []:
        for s in nd.get("streams") or []:
            sid = f"{nd['id']}.{s['name']}"
            lo = dict(s.get("lo") or {})
            bw = s.get("occupied_bw_hz")
            out[sid] = StreamConfig(
                stream=sid,
                direction=s["direction"],
                fc_hz=float(s["fc_hz"]),
                rate=parse_exact(s["rate_sps"], name=f"{sid}.rate_sps"),
                occupied_bw_hz=None if bw is None else float(bw),
                full_scale_dbm=float(s.get("full_scale_dbm", 0.0)),
                gain_db=float(s.get("gain_db", 0.0)),
                noise_figure_db=float(s.get("noise_figure_db", 0.0)),
                cfo_hz=float(lo.get("cfo_hz", 0.0)),
                linewidth_hz=float(lo.get("linewidth_hz", 0.0)),
                adc_clip=bool(s.get("adc_clip", False)),
            )
    return out


# =========================================================================== traffic


class TrafficSource(Protocol):
    """In-process TX generator: digital samples (full scale 1) for absolute TX indices
    ``[k0, k0 + n)``, a pure function of the index."""

    def __call__(self, k0: int, n: int) -> NDArray[np.complex64]: ...


class BlockCache:
    """Values of a pure function of the absolute sample index, generated in aligned blocks.

    ``fn(k0, n)`` must return the values of indices [k0, k0 + n) and depend only on the
    index (noise, LO phasors, tones), so serving a request from cached blocks of ``block``
    samples is bitwise identical to calling ``fn`` on the request directly (I2a); the
    blocks only amortize per-call overhead. Indices must be >= 0.
    """

    def __init__(
        self, fn: Callable[[int, int], NDArray[Any]], block: int = 16384, keep: int = 4
    ) -> None:
        self.fn = fn
        self.block = int(block)
        self.keep = int(keep)
        self._blocks: dict[int, NDArray[Any]] = {}

    def _get(self, b: int) -> NDArray[Any]:
        hit = self._blocks.get(b)
        if hit is None:
            hit = self.fn(b * self.block, self.block)
            self._blocks[b] = hit
            if len(self._blocks) > self.keep:
                del self._blocks[min(self._blocks)]
        return hit

    def __call__(self, k0: int, n: int) -> NDArray[Any]:
        if n <= 0:
            return self.fn(k0, 0)
        b0, b1 = k0 // self.block, (k0 + n - 1) // self.block
        if b0 == b1:
            off = k0 - b0 * self.block
            return self._get(b0)[off : off + n]
        parts = []
        for b in range(b0, b1 + 1):
            blk = self._get(b)
            lo = max(k0, b * self.block) - b * self.block
            hi = min(k0 + n, (b + 1) * self.block) - b * self.block
            parts.append(blk[lo:hi])
        return np.concatenate(parts)


def _phasor_aligned(df_hz: float, rate: Fraction, k0: int, n: int) -> NDArray[np.complex128]:
    """:func:`sremu.impair.oscillator.cfo_phasor` for a range aligned to its 1024-sample
    anchor blocks (``k0`` and ``n`` multiples of 1024): the same anchors and table, formed
    as an outer product (bitwise equal); other ranges fall back to ``cfo_phasor``."""
    from sremu.impair.oscillator import CFO_BLOCK, cycles_per_sample

    if k0 % CFO_BLOCK or n % CFO_BLOCK or n == 0:
        return cfo_phasor(df_hz, rate, k0, n)
    q_hi, q_lo = cycles_per_sample(df_hz, rate)
    if q_hi == 0.0 and q_lo == 0.0:
        return np.ones(n, dtype=np.complex128)
    from sremu.channel.interp import frac_cycles_of_index

    blocks = np.arange(k0 // CFO_BLOCK, (k0 + n) // CFO_BLOCK, dtype=np.int64)
    anchors = np.exp(2j * np.pi * frac_cycles_of_index(blocks * CFO_BLOCK, q_hi, q_lo))
    table = np.exp(
        2j * np.pi * frac_cycles_of_index(np.arange(CFO_BLOCK, dtype=np.int64), q_hi, q_lo)
    )
    return (anchors[:, None] * table[None, :]).reshape(-1)


class ToneSource:
    """``amplitude * exp(+j 2 pi frac(freq_hz*k/f_s))`` (exact phase; DC when freq is 0)."""

    def __init__(
        self, rate: Fraction | int | str, freq_hz: float = 0.0, amplitude: float = 1.0
    ) -> None:
        self.rate = parse_exact(rate, name="rate")
        self.freq_hz = float(freq_hz)
        self.amplitude = float(amplitude)
        self._cache = BlockCache(self._block)

    def _block(self, k0: int, n: int) -> NDArray[np.complex64]:
        ph = _phasor_aligned(self.freq_hz, self.rate, k0, n)
        return (self.amplitude * ph).astype(np.complex64)

    def __call__(self, k0: int, n: int) -> NDArray[np.complex64]:
        if self.freq_hz == 0.0:
            return np.full(n, self.amplitude, dtype=np.complex64)
        out: NDArray[np.complex64] = self._cache(k0, n)
        return out


@dataclass(frozen=True, eq=False)
class ArraySource:
    """A fixed waveform starting at absolute index ``start`` (zeros elsewhere); with
    ``period`` > 0 it repeats every ``period`` samples from ``start``."""

    samples: NDArray[np.complex64]
    start: int = 0
    period: int = 0

    def __call__(self, k0: int, n: int) -> NDArray[np.complex64]:
        out = np.zeros(n, dtype=np.complex64)
        x = np.asarray(self.samples, dtype=np.complex64)
        k = np.arange(k0, k0 + n, dtype=np.int64) - self.start
        if self.period > 0:
            ok = k >= 0
            idx = np.mod(k, self.period)
            ok &= idx < x.shape[0]
        else:
            idx = k
            ok = (k >= 0) & (k < x.shape[0])
        out[ok] = x[idx[ok]]
        return out


@dataclass(frozen=True, eq=False)
class QpskPacketSource:
    """QPSK packets with rectangular pulses (E-B waveform, section 11).

    Packet ``p`` starts at ``start + p*period`` with ``period = (preamble + payload)*sps +
    guard`` samples. The preamble symbols are fixed per stream (keyed ``traffic/<stream>``,
    lane 1, counter = symbol index); payload symbols are keyed by the absolute payload
    symbol index ``p*payload + i`` (lane 0). Symbols are ``(+-1 +- j)/sqrt(2)``.
    """

    stream: StreamId
    seed: int
    sps: int = 4
    preamble_syms: int = 64
    payload_syms: int = 512
    guard: int = 500
    start: int = 0

    @property
    def period(self) -> int:
        """Samples per packet including the guard."""
        return (self.preamble_syms + self.payload_syms) * self.sps + self.guard

    def symbols(self, p: int) -> NDArray[np.complex128]:
        """Preamble then payload symbols of packet ``p``."""
        key = rng.derive_key(self.seed, "traffic", self.stream)
        pre = rng.philox_block(key, 0, self.preamble_syms, sub=1)
        pay = rng.philox_block(key, p * self.payload_syms, self.payload_syms, sub=0)
        bits = np.concatenate([pre[:, :2], pay[:, :2]]) >> np.uint32(31)
        re = 1.0 - 2.0 * bits[:, 0].astype(np.float64)
        im = 1.0 - 2.0 * bits[:, 1].astype(np.float64)
        out = np.empty(re.shape[0], dtype=np.complex128)
        out.real = re / math.sqrt(2.0)
        out.imag = im / math.sqrt(2.0)
        return out

    def __call__(self, k0: int, n: int) -> NDArray[np.complex64]:
        out = np.zeros(n, dtype=np.complex64)
        per = self.period
        body = (self.preamble_syms + self.payload_syms) * self.sps
        lo = max(k0, self.start)
        if lo >= k0 + n:
            return out
        p0 = (lo - self.start) // per
        p1 = (k0 + n - 1 - self.start) // per
        for p in range(p0, p1 + 1):
            s0 = self.start + p * per
            a, b = max(s0, k0), min(s0 + body, k0 + n)
            if a >= b:
                continue
            sym = self.symbols(p)
            idx = (np.arange(a, b) - s0) // self.sps
            out[a - k0 : b - k0] = sym[idx].astype(np.complex64)
        return out


def bpsk_v1_waveform() -> NDArray[np.complex64]:
    """The v1 BPSK TX waveform (section 8.2, V-CFO): a 200-symbol header of ``1+1j``, then
    38 BPSK symbols at ``+-0.1(1+j)`` (bits 0000111111 then 28 zeros), 10 samples per
    symbol, rectangular pulse by convolution: 2389 samples at 5880 S/s."""
    bits = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] + [0] * 28
    sym = np.array([0.1 + 0.1j if b else -0.1 - 0.1j for b in bits])
    data = np.concatenate([np.full(200, 1 + 1j), sym])
    sps = 10
    d = np.zeros(data.shape[0] * sps, dtype=np.complex128)
    d[::sps] = data
    return np.convolve(d, np.ones(sps)).astype(np.complex64)


def traffic_from_scenario(
    scenario: Mapping[str, Any],
    configs: Mapping[StreamId, StreamConfig],
    *,
    base_dir: Path | None = None,
) -> dict[StreamId, TrafficSource]:
    """In-process generators of the scenario ``traffic`` list (section 9.2)."""
    out: dict[StreamId, TrafficSource] = {}
    seed = int(scenario.get("seed", 0))
    for t in scenario.get("traffic") or []:
        sid = str(t["stream"])
        if sid not in configs or configs[sid].direction != "tx":
            raise ConfigurationError(f"traffic stream {sid!r} is not a TX stream")
        rate = configs[sid].rate
        params = dict(t.get("params") or {})
        gen = t["generator"]
        if gen == "tone":
            out[sid] = ToneSource(
                rate, float(params.get("freq_hz", 0.0)), float(params.get("amplitude", 1.0))
            )
        elif gen == "qpsk_packets":
            guard = to_ticks(parse_exact(str(params.get("guard_s", "5e-4"))), rate)
            out[sid] = QpskPacketSource(
                sid,
                seed,
                int(params.get("sps", 4)),
                int(params.get("preamble_syms", 64)),
                int(params.get("payload_syms", 512)),
                guard,
            )
        elif gen == "bpsk":
            period = int(params.get("period_samples", 0))
            out[sid] = ArraySource(bpsk_v1_waveform(), 0, period)
        elif gen == "file":
            path = Path(str(params["file"]))
            if base_dir is not None and not path.is_absolute():
                path = base_dir / path
            data = np.fromfile(path, dtype=np.complex64)
            out[sid] = ArraySource(data, 0, data.shape[0] if params.get("repeat") else 0)
        else:
            raise ConfigurationError(f"unknown traffic generator {gen!r}")
    return out


# =========================================================================== TX


@dataclass
class _Chunk:
    k0: int
    x: NDArray[np.complex64]  # front-ended and LO-rotated (watts-referenced)

    @property
    def end(self) -> int:
        return self.k0 + self.x.shape[0]


class TxStream:
    """Runtime state of one TX stream (see the module docstring).

    ``ring_capacity``: initial ring length (grown when the renderer's look-back grows).
    ``seed``/``backend``: for the keyed Wiener phase noise of the TX LO.
    """

    def __init__(
        self,
        cfg: StreamConfig,
        ring_capacity: int,
        *,
        seed: int = 0,
        backend: BackendName = "numpy_ref",
        source: TrafficSource | None = None,
        unopened: str = "silent",
    ) -> None:
        if cfg.direction != "tx":
            raise ConfigurationError(f"{cfg.stream} is not a TX stream")
        self.cfg = cfg
        self.ring = TxRingBuffer(cfg.stream, cfg.rate, int(ring_capacity))
        self.scale = TxFrontEnd(cfg.full_scale_dbm, cfg.gain_db).scale
        self.pn = (
            WienerPhaseNoise(
                rng.derive_key(seed, "pn", f"{cfg.stream}.lo"),
                cfg.linewidth_hz,
                cfg.rate,
                backend=backend,
            )
            if cfg.linewidth_hz > 0.0
            else None
        )
        self.source = source
        self._lo = (
            BlockCache(lambda k, n: _phasor_aligned(cfg.cfo_hz, cfg.rate, k, n))
            if cfg.cfo_hz != 0.0
            else None
        )
        src = source
        self._source_prepared = BlockCache(
            lambda k, n: (
                self.prepare(k, src(k, n)) if src is not None else np.zeros(n, np.complex64)
            )
        )
        self.pending: deque[_Chunk] = deque()
        self.data_end = 0
        self.idle_until = 0
        self.opened = False
        self.silent = unopened == "silent"
        self.in_burst = False
        self.ended = False
        self.slip = False
        self.asyncq: deque[AsyncMetadata] = deque()
        self.zero_filled = 0

    @property
    def stream(self) -> StreamId:
        """Stream id."""
        return self.cfg.stream

    # ------------------------------------------------------------------ chain
    def prepare(self, k0: int, d: NDArray[np.complexfloating]) -> NDArray[np.complex64]:
        """TX chain for digital samples at absolute indices [k0, k0 + n): full scale/gain,
        then the TX LO (``exp(+j(phi_cfo + phi_pn))``), cast to complex64."""
        x = np.asarray(d).astype(np.complex128) * self.scale
        n = x.shape[0]
        if n and self._lo is not None:
            x *= self._lo(k0, n)
        if n and self.pn is not None:
            x *= self.pn.phasor(k0, n)
        return x.astype(np.complex64)

    # ------------------------------------------------------------------ commitment
    def committed(self) -> int | None:
        """First uncommitted absolute index, or None when committed indefinitely."""
        if self.source is not None or self.silent or self.ended:
            return None
        return max(self.data_end, self.idle_until, self.ring.committed_through())

    def head(self) -> int:
        """First index that a new (untimed) send may use."""
        return max(self.data_end, self.ring.committed_through())

    def open(self) -> None:
        """A client attached: commitment now follows its data."""
        self.opened = True
        if self.source is None:
            self.silent = False

    # ------------------------------------------------------------------ submission
    def _async(self, code: AsyncCode, k: int, n: int = 0) -> None:
        self.asyncq.append(
            AsyncMetadata(code, self.stream, 0, True, TimeSpec.from_ticks(k, self.cfg.rate), k, n)
        )

    def submit(
        self, samples: NDArray[np.complexfloating], md: TxMetadata, frontier: int
    ) -> tuple[int, int]:
        """Queue client samples; returns ``(accepted, late_samples)``.

        ``frontier``: the linked receivers' first uncomputed index (untimed sends start at
        or after it). Timed bursts that start before the ring head are dropped (late).
        """
        x = np.ascontiguousarray(samples, dtype=np.complex64).reshape(-1)
        f = self.cfg.rate
        if md.idle_until:
            if x.size:
                raise StreamError("idle_until frames carry no samples")
            self.idle_until = max(self.idle_until, to_ticks(md.time_spec, f))
            return 0, 0
        if md.has_time_spec:
            k0 = to_ticks(md.time_spec, f)
        elif self.in_burst and not self.slip:
            k0 = self.head()
        else:
            k0 = max(self.head(), int(frontier))
            self.slip = False
        if k0 < self.head():
            n = int(x.size)
            self._async(AsyncCode.TIME_ERROR, k0, n)
            return 0, n
        if x.size:
            self.pending.append(_Chunk(k0, self.prepare(k0, x)))
            self.data_end = k0 + int(x.size)
        else:
            self.data_end = max(self.data_end, k0)
        eob = bool(md.end_of_burst or md.end_of_stream)
        self.in_burst = not eob
        self.silent = eob
        if md.end_of_stream:
            self.ended = True
        if md.end_of_burst:
            self._async(AsyncCode.BURST_ACK, self.data_end)
        return int(x.size), 0

    # ------------------------------------------------------------------ ring
    def write_through(self, k_need: int, *, zero_fill: bool = False) -> int:
        """Commit samples into the ring up to absolute index ``k_need`` (exclusive).

        Pending client data and the in-process source are written; silence (EOB, idle,
        unopened) is committed as zeros. With ``zero_fill`` an uncommitted gap is filled
        with zeros too (stall/underflow policy); returns the number of zero-filled
        samples that were NOT committed by the client (0 otherwise).
        """
        ring = self.ring
        k_need = int(k_need)
        if k_need <= ring.committed_through():
            return 0
        if self.source is not None:
            k0 = ring.committed_through()
            ring.write(k0, self._source_prepared(k0, k_need - k0))
            return 0
        zeros = 0
        while ring.committed_through() < k_need:
            head = ring.committed_through()
            if self.pending:
                ch = self.pending[0]
                if ch.end <= head:  # already superseded by committed samples
                    self.pending.popleft()
                    continue
                if ch.k0 <= head:
                    b = min(ch.end, k_need)
                    ring.write(head, ch.x[head - ch.k0 : b - ch.k0])
                    if b >= ch.end:
                        self.pending.popleft()
                    continue
                # a gap before a later timed burst is idle (committed silence)
                ring.idle_until(min(ch.k0, k_need))
                continue
            if self.silent or self.ended:
                ring.idle_until(k_need)
                break
            lim = min(k_need, self.idle_until)
            if lim > head:
                ring.idle_until(lim)
                continue
            if zero_fill:
                zeros = k_need - head
                ring.idle_until(k_need)
                self.zero_filled += zeros
            break
        return zeros

    def drop_before(self, k: int) -> None:
        """Forget pending data before absolute index ``k`` (skipped receivers)."""
        while self.pending and self.pending[0].end <= k:
            self.pending.popleft()

    def ensure_capacity(self, span: int) -> None:
        """Grow the ring (power of two) so that it holds ``span`` samples of history."""
        if span <= self.ring.capacity:
            return
        cap = 1 << max(10, math.ceil(math.log2(span)))
        old = self.ring
        new = TxRingBuffer(self.stream, self.cfg.rate, cap)
        lo = old.oldest()
        hi = old.committed_through()
        if hi > lo:
            new.write(lo, old.read(lo, hi - lo))
        self.ring = new


# =========================================================================== RX


class HistoryRing:
    """Circular complex64 history of delivered RX samples (observation windows)."""

    def __init__(self, capacity: int) -> None:
        cap = 1 << max(6, math.ceil(math.log2(max(1, capacity))))
        self.capacity = cap
        self.mask = cap - 1
        self.buf = np.zeros(cap, dtype=np.complex64)
        self.end = 0  # first index not written

    def _put(self, k0: int, x: NDArray[np.complex64]) -> None:
        n = int(x.shape[0])
        start = k0 & self.mask
        first = min(n, self.capacity - start)
        self.buf[start : start + first] = x[:first]
        if first < n:
            self.buf[: n - first] = x[first:]

    def write(self, k0: int, x: NDArray[np.complex64]) -> None:
        """Store samples at absolute indices [k0, k0 + n) (skipped ranges read as 0)."""
        n = int(x.shape[0])
        if k0 > self.end:
            gap = min(k0 - self.end, self.capacity)
            self._put(k0 - gap, np.zeros(gap, dtype=np.complex64))
        if n >= self.capacity:
            x = x[n - self.capacity :]
            k0 += n - self.capacity
            n = self.capacity
        self._put(k0, x)
        self.end = max(self.end, k0 + n)

    def grow(self, capacity: int) -> None:
        """Hold at least ``capacity`` samples from now on (power of two); the retained
        samples are kept."""
        if capacity <= self.capacity:
            return
        lo = max(0, self.end - self.capacity)
        kept = self.read(lo, self.end - lo)
        cap = 1 << max(6, math.ceil(math.log2(capacity)))
        self.capacity = cap
        self.mask = cap - 1
        self.buf = np.zeros(cap, dtype=np.complex64)
        end = self.end
        self.end = lo
        if kept.size:
            self._put(lo, kept)
        self.end = end

    def read(self, k0: int, n: int) -> NDArray[np.complex64]:
        """Samples [k0, k0 + n); indices not retained (or never written) read as 0."""
        out = np.zeros(max(0, n), dtype=np.complex64)
        lo = max(k0, self.end - self.capacity, 0)
        hi = min(k0 + n, self.end)
        if hi > lo:
            start = lo & self.mask
            cnt = hi - lo
            first = min(cnt, self.capacity - start)
            o = lo - k0
            out[o : o + first] = self.buf[start : start + first]
            if first < cnt:
                out[o + first : o + cnt] = self.buf[: cnt - first]
        return out


@dataclass(eq=False)
class _Entry:
    k0: int
    data: NDArray[np.complex64] | None  # None: overflow gap marker
    sob: bool = False
    eob: bool = False
    k_after: int = 0  # gap marker: first index after the gap


class RxStream:
    """Runtime state of one RX stream (see the module docstring).

    ``noise_key``: the Philox key words of purpose ``noise/<stream>``; ``noise_on`` turns
    thermal noise on; ``backend`` selects the noise generator (I5).
    """

    def __init__(
        self,
        cfg: StreamConfig,
        *,
        seed: int = 0,
        backend: BackendName = "numpy_ref",
        noise_on: bool = True,
        history: int = 4096,
        queue_samples: int | None = None,
    ) -> None:
        if cfg.direction != "rx":
            raise ConfigurationError(f"{cfg.stream} is not an RX stream")
        self.cfg = cfg
        self.backend: BackendName = backend
        self.front = RxFrontEnd(cfg.full_scale_dbm, cfg.gain_db, cfg.adc_clip)
        self._plain_front = self.front.scale == 1.0 and not cfg.adc_clip
        nk = rng.derive_key(seed, "noise", cfg.stream)
        self.noise_key: tuple[int, int] = (int(nk[0]), int(nk[1]))
        self.sigma = thermal_sigma(float(cfg.rate), cfg.noise_figure_db) if noise_on else 0.0
        self.pn = (
            WienerPhaseNoise(
                rng.derive_key(seed, "pn", f"{cfg.stream}.lo"),
                cfg.linewidth_hz,
                cfg.rate,
                backend=backend,
            )
            if cfg.linewidth_hz > 0.0
            else None
        )
        key = self.noise_key
        sig = self.sigma
        be = backend
        self._noise = (
            BlockCache(lambda k, n: _noise(key, k, n, sig, backend=be)) if sig > 0.0 else None
        )
        self._lo = (
            BlockCache(lambda k, n: np.conj(_phasor_aligned(cfg.cfo_hz, cfg.rate, k, n)))
            if cfg.cfo_hz != 0.0
            else None
        )
        # conj(exp(+j*phi_pn(k))): a pure function of the absolute index (bitwise chunk
        # invariant, tests/runtime/test_elision_determinism.py), so whole blocks serve every
        # page instead of one Wiener-bridge evaluation per page
        pn = self.pn
        self._pn = BlockCache(lambda k, n: np.conj(pn.phasor(k, n))) if pn is not None else None
        self.history = HistoryRing(history)
        self.queue: deque[_Entry] = deque()
        self.queued = 0
        self.queue_samples = queue_samples
        self.opened = False
        self.streaming = False
        self.start: int | None = None
        self.remaining: int | None = None
        self.eob_at_end = True
        self.burst_started = False
        self.pending_error = ErrorCode.NONE
        self.capture: list[tuple[int, NDArray[np.complex128]]] | None = None
        self.overflow_pages = 0
        self.delivered = 0

    @property
    def stream(self) -> StreamId:
        """Stream id."""
        return self.cfg.stream

    # ------------------------------------------------------------------ chain
    def process(self, k0: int, acc: NDArray[np.complex128]) -> NDArray[np.complex64]:
        """RX chain on the rendered accumulator for absolute samples [k0, k0 + n):
        thermal noise -> RX LO (``exp(-j(phi_cfo + phi_pn))``) -> optional complex128 tap ->
        gain/full scale -> optional clip -> complex64."""
        n = acc.shape[0]
        y = acc
        if self._noise is not None and n:
            y = y + self._noise(k0, n)
        # np.multiply with y as the first operand: NumPy temporary elision (Linux/macOS, >= 16384
        # complex128 samples) can otherwise swap the operand order, and complex multiply is not
        # bitwise commutative, which broke chunk invariance on Linux (DESIGN D-4)
        if n and self._lo is not None:
            y = np.multiply(y, self._lo(k0, n))
        if n and self._pn is not None:
            y = np.multiply(y, self._pn(k0, n))
        if self.capture is not None:
            self.capture.append((k0, np.array(y, dtype=np.complex128, copy=True)))
        if self._plain_front:  # unit scale, no clip: RxFrontEnd.apply is a plain cast
            return np.asarray(y, dtype=np.complex128).astype(np.complex64)
        return self.front.apply(y)

    def power_w(self, d: NDArray[np.complex64]) -> float:
        """Mean power at the antenna port (W) of delivered digital samples."""
        if d.size == 0:
            return math.nan
        p = float(np.mean(np.abs(d.astype(np.complex128)) ** 2))
        return p / (self.front.scale**2)

    # ------------------------------------------------------------------ streaming
    def stream_cmd(self, cmd: StreamCmd, frontier: int) -> None:
        """Apply a UHD stream command (a start in the past -> LATE_COMMAND on next read)."""
        if cmd.mode == "stop_cont":
            self.streaming = False
            return
        start = frontier if cmd.stream_now else to_ticks(cmd.time_spec, self.cfg.rate)
        if start < frontier:
            self.pending_error = ErrorCode.LATE_COMMAND
            self.streaming = False
            return
        if cmd.mode == "num_more" and self.streaming and self.remaining is not None:
            self.remaining += int(cmd.num_samps)
            return
        self.streaming = True
        self.start = start
        self.burst_started = False
        self.remaining = None if cmd.mode == "start_cont" else int(cmd.num_samps)
        self.eob_at_end = cmd.mode == "num_done"

    def wants(self, k0: int, k1: int) -> bool:
        """Whether the client stream consumes any sample of [k0, k1)."""
        if not self.streaming or self.start is None:
            return False
        if self.remaining is not None and self.remaining <= 0:
            return False
        return k1 > self.start

    def deliver(self, k0: int, d: NDArray[np.complex64], *, drop_when_full: bool = False) -> bool:
        """Store computed samples [k0, k0 + n): history, then the client queue when
        streaming. Returns False when the page was dropped (queue full, paced mode)."""
        n = int(d.shape[0])
        self.history.write(k0, d)
        if not self.wants(k0, k0 + n):
            return True
        assert self.start is not None
        a = max(k0, self.start)
        b = k0 + n
        if self.remaining is not None:
            b = min(b, a + self.remaining)
        if b <= a:
            return True
        part = d[a - k0 : b - k0]
        if (
            drop_when_full
            and self.queue_samples is not None
            and self.queued + part.shape[0] > self.queue_samples
        ):
            self.overflow_pages += 1
            last = self.queue[-1] if self.queue else None
            if last is not None and last.data is None:
                last.k_after = b
            else:
                self.queue.append(_Entry(a, None, k_after=b))
            if self.remaining is not None:
                self.remaining -= b - a
                if self.remaining <= 0:
                    self.streaming = False
            self.start = b
            return False
        sob = not self.burst_started
        self.burst_started = True
        eob = False
        if self.remaining is not None:
            self.remaining -= b - a
            if self.remaining <= 0:
                self.streaming = False
                eob = self.eob_at_end
        self.queue.append(_Entry(a, np.array(part, copy=True), sob, eob))
        self.queued += part.shape[0]
        self.start = b if self.streaming else self.start
        return True

    def read(self, n: int) -> tuple[NDArray[np.complex64], RxMetadata]:
        """Pop up to ``n`` queued samples (never across an end of burst or a gap)."""
        md = RxMetadata()
        empty = np.zeros(0, dtype=np.complex64)
        if self.pending_error != ErrorCode.NONE:
            md.error_code, self.pending_error = self.pending_error, ErrorCode.NONE
            return empty, md
        if not self.queue:
            md.error_code = ErrorCode.TIMEOUT
            return empty, md
        head = self.queue[0]
        if head.data is None:
            self.queue.popleft()
            md.error_code = ErrorCode.OVERFLOW
            md.has_time_spec = True
            md.time_spec = TimeSpec.from_ticks(head.k_after, self.cfg.rate)
            md.first_sample = head.k_after
            return empty, md
        parts: list[NDArray[np.complex64]] = []
        got = 0
        first = head.k0
        md.start_of_burst = head.sob
        expect = first
        while self.queue and got < n:
            e = self.queue[0]
            if e.data is None or e.k0 != expect:
                break
            take = min(n - got, e.data.shape[0])
            parts.append(e.data[:take])
            got += take
            expect += take
            if take == e.data.shape[0]:
                self.queue.popleft()
                if e.eob:
                    md.end_of_burst = True
                    break
            else:
                e.data = e.data[take:]
                e.k0 += take
                e.sob = False
                break
        self.queued -= got
        self.delivered += got
        md.has_time_spec = True
        md.time_spec = TimeSpec.from_ticks(first, self.cfg.rate)
        md.first_sample = first
        out = np.concatenate(parts) if len(parts) > 1 else (parts[0] if parts else empty)
        if got == 0:
            md.error_code = ErrorCode.TIMEOUT
        return np.ascontiguousarray(out, dtype=np.complex64), md


def default_bandwidth_warning(configs: Sequence[StreamConfig]) -> None:
    """Warn once for every TX stream without a declared occupied bandwidth (section 4.5)."""
    for c in configs:
        if c.direction == "tx" and c.occupied_bw_hz is None:
            warnings.warn(
                f"{c.stream}: occupied_bw_hz not declared; using 0.8*f_s (section 4.5)",
                ValidityWarning,
                stacklevel=3,
            )
