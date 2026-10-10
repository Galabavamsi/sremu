"""Frozen cross-work-package contracts (DESIGN.md section 3.4; CONTRACT_VERSION "1.0").

This module is frozen at the end of WP0. It changes only through a contract change request
(CCR, section 14.3) applied by the lead together with a ``CONTRACT_VERSION`` bump.

Conventions (section 3.1, normative):

* SI units everywhere; every dimensional field name carries a unit suffix
  (``_hz _sps _m _s _dbm _db _deg _rad _mps _w``).
* Exact times and rates are ``Fraction`` (s, S/s). Where a dataclass accepts an exact
  quantity it also accepts an int or an exact string (``"2e-4"``, ``"1/200"``); floats are
  rejected with ``TypeError`` (convert API floats with ``sremu.timebase.to_exact``).
* Channel coefficients are dimensionless and Friis-normalized (``P_R = P_T*|h|**2``);
  path amplitudes EXCLUDE the propagation phase ``exp(-j*2*pi*f_c*tau)``, which the renderer
  applies exactly from the delay.
* IQ at every interface is complex64; accumulators are complex128 (section 3.5).
* Dataclasses holding arrays use ``eq=False`` (identity equality); treat their arrays as
  immutable after construction.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import IntEnum
from fractions import Fraction
from typing import Any, Final, Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np

from sremu.errors import AsyncCode, ConfigurationError, ErrorCode, RateMismatchError, TooLarge
from sremu.timebase import Rate, TimeSpec, fraction_to_str, parse_exact

__all__ = [
    "CONTRACT_VERSION",
    "COUNTER_NAMES",
    "DEFAULT_BIND_HOST",
    "DEFAULT_CONTROL_ENDPOINT",
    "DEFAULT_MAX_COMMAND_RATE_HZ",
    "DEFAULT_PUB_ENDPOINT",
    "DEFAULT_TX_INGRESS_ENDPOINT",
    "MAX_CMD_ID_LEN",
    "MAX_ELEMENTS_PER_COMMAND",
    "MAX_FRAME_SAMPLES",
    "MAX_MESSAGE_BYTES",
    "Action",
    "AsyncMetadata",
    "BackendName",
    "CommandOp",
    "CommandReceipt",
    "CommandRecord",
    "CommandRequest",
    "CommandStatus",
    "CommandTime",
    "Controller",
    "Coupling",
    "CpuFormat",
    "CsiView",
    "DataClass",
    "Direction",
    "EngineProtocol",
    "EventPriority",
    "EventReference",
    "FDSpec",
    "Granularity",
    "LatePolicy",
    "LatencyModel",
    "LinkId",
    "LinkJob",
    "LinkPaths",
    "MotionBounds",
    "NodeId",
    "NoiseFn",
    "Observation",
    "ObservationField",
    "ObservationSpec",
    "PanelConfig",
    "PanelId",
    "PathBundle",
    "PathKind",
    "PropagationModel",
    "RISCoefficients",
    "RemoteControllerProxy",
    "RenderBackend",
    "RenderFn",
    "RisCapabilities",
    "RunMode",
    "RxMetadata",
    "SegmentJob",
    "Snapshot",
    "SourceKind",
    "StreamCmd",
    "StreamHandle",
    "StreamId",
    "StreamMode",
    "StreamSpec",
    "TransitionModel",
    "TransitionSpec",
    "TransitionTerm",
    "TxMetadata",
    "TxRing",
    "check_link",
    "config_hash",
    "link_id",
    "split_link_id",
    "split_stream_id",
    "stream_id",
]

CONTRACT_VERSION: Final[str] = "1.0"

# --------------------------------------------------------------------------- identifiers
StreamId: TypeAlias = str
"""``<node>.<stream>``, e.g. ``"ap.tx0"`` (section 3.2)."""
LinkId: TypeAlias = str
"""``<tx_stream>-><rx_stream>``, e.g. ``"ap.tx0->mu.rx0"``."""
PanelId: TypeAlias = str
"""RIS panel id from the scenario, e.g. ``"ris0"``."""
NodeId: TypeAlias = str
"""Node id from the scenario, e.g. ``"ap"``."""

_ID_RE: Final = re.compile(r"^[A-Za-z0-9_\-]+$")


def stream_id(node: NodeId, stream: str) -> StreamId:
    """Compose a stream id ``<node>.<stream>``."""
    for part in (node, stream):
        if not _ID_RE.match(part):
            raise ConfigurationError(f"invalid id component {part!r} (allowed: A-Z a-z 0-9 _ -)")
    return f"{node}.{stream}"


def split_stream_id(sid: StreamId) -> tuple[NodeId, str]:
    """Split ``<node>.<stream>`` into ``(node, stream)``."""
    node, sep, stream = sid.partition(".")
    if not sep or not _ID_RE.match(node) or not _ID_RE.match(stream):
        raise ConfigurationError(f"invalid stream id {sid!r}; expected '<node>.<stream>'")
    return node, stream


def link_id(tx: StreamId, rx: StreamId) -> LinkId:
    """Compose a link id ``<tx_stream>-><rx_stream>``."""
    split_stream_id(tx)
    split_stream_id(rx)
    return f"{tx}->{rx}"


def split_link_id(lid: LinkId) -> tuple[StreamId, StreamId]:
    """Split ``<tx>-><rx>`` into ``(tx_stream, rx_stream)``."""
    tx, sep, rx = lid.partition("->")
    if not sep:
        raise ConfigurationError(f"invalid link id {lid!r}; expected '<tx>-><rx>'")
    split_stream_id(tx)
    split_stream_id(rx)
    return tx, rx


# --------------------------------------------------------------------------- vocabularies
Direction: TypeAlias = Literal["tx", "rx"]
LatePolicy: TypeAlias = Literal["apply_on_arrival", "reject"]
CommandStatus: TypeAlias = Literal["scheduled", "applied", "late_applied", "rejected", "superseded"]
StreamMode: TypeAlias = Literal["start_cont", "stop_cont", "num_done", "num_more"]
TransitionModel: TypeAlias = Literal["step", "linear", "first_order"]
Granularity: TypeAlias = Literal["element", "column", "tile", "groups"]
CpuFormat: TypeAlias = Literal["fc32", "sc16"]
RunMode: TypeAlias = Literal["lockstep", "paced"]
Coupling: TypeAlias = Literal["barrier", "async"]
BackendName: TypeAlias = Literal["numpy_ref", "numba_cpu"]
EventReference: TypeAlias = Literal["ris", "rx"]
CommandOp: TypeAlias = Literal[
    "ris.set_config", "ris.set_enabled", "node.set_position", "node.set_trajectory", "rec.marker"
]
ObservationField: TypeAlias = Literal[
    "rx_power_dbm", "ris_state", "positions", "csi_oracle", "iq_window"
]
DataClass: TypeAlias = Literal[
    "MEAS",
    "DIG",
    "REPORTED",
    "ASSUMED",
    "SIM-sremu",
    "SIM-legacy-v1",
    "SIM-legacy-v2",
    "ANALYTIC",
    "MATLAB",
]
"""Manifest ``data_class`` of an input or output (section 3.6)."""
SourceKind: TypeAlias = Literal[
    "measured", "fullwave", "circuit", "analytic", "digitized", "assumed"
]
"""Response-table provenance ``source_kind`` (section 5.5)."""

_COMMAND_OPS: Final = frozenset(
    {"ris.set_config", "ris.set_enabled", "node.set_position", "node.set_trajectory", "rec.marker"}
)
_OBS_FIELDS: Final = frozenset(
    {"rx_power_dbm", "ris_state", "positions", "csi_oracle", "iq_window"}
)

# --------------------------------------------------------------------------- limits / defaults
MAX_CMD_ID_LEN: Final[int] = 64
"""Maximum length of a client-supplied command id (section 3.2)."""
MAX_ELEMENTS_PER_COMMAND: Final[int] = 65_536
"""Maximum number of elements in one command (section 8.5)."""
MAX_MESSAGE_BYTES: Final[int] = 1 << 20
"""Maximum control message size, 1 MiB (section 8.5)."""
MAX_FRAME_SAMPLES: Final[int] = 1 << 22
"""Maximum ``nsamps`` of one data-plane frame (section 8.3)."""
DEFAULT_MAX_COMMAND_RATE_HZ: Final[float] = 1e4
"""Default per-client command rate limit, Hz (section 8.5)."""
DEFAULT_BIND_HOST: Final[str] = "127.0.0.1"
"""Every socket binds here unless ``--allow-remote`` is given (section 8.10)."""
DEFAULT_CONTROL_ENDPOINT: Final[str] = "tcp://127.0.0.1:5600"
"""Control ROUTER endpoint (section 8.5)."""
DEFAULT_PUB_ENDPOINT: Final[str] = "tcp://127.0.0.1:5601"
"""Telemetry PUB endpoint (section 8.5)."""
DEFAULT_TX_INGRESS_ENDPOINT: Final[str] = "tcp://127.0.0.1:5602"
"""Native TX ingress PULL endpoint (section 8.3)."""

COUNTER_NAMES: Final[tuple[str, ...]] = (
    "late_samples",
    "underflow_slip_samples",
    "overflow_pages",
    "bad_packets",
    "schema_errors",
    "seq_errors",
    "rec_dropped",
    "stall_zero_samples",
    "late_commands",
    "rejected_commands",
    "rate_limited_commands",
    "deadline_misses",
)
"""Canonical I6 counter names shared by engine, metrics, transport and record."""


class PathKind(IntEnum):
    """Kind of a propagation path (stored as int8 in path arrays)."""

    LOS = 0
    GROUND = 1
    WALL = 2
    DIFF = 3
    RIS_TILE = 4
    DIFFUSE = 5


class EventPriority(IntEnum):
    """Event-heap priority; events are ordered by ``(t, prio, seq)`` (section 4.6)."""

    GEOMETRY = 0
    RIS = 1
    IMPAIRMENT = 2
    STREAM = 3
    MARKER = 9


# --------------------------------------------------------------------------- helpers


def _set(obj: object, name: str, value: object) -> None:
    object.__setattr__(obj, name, value)


def _coerce_exact(obj: object, name: str, *, allow_none: bool = False, nonneg: bool = True) -> None:
    """Coerce an exact-quantity field (Fraction/int/str) of a frozen dataclass in place."""
    value = getattr(obj, name)
    if value is None:
        if allow_none:
            return
        raise TypeError(f"{type(obj).__name__}.{name} must not be None")
    q = parse_exact(value, name=f"{type(obj).__name__}.{name}")
    if nonneg and q < 0:
        raise ConfigurationError(f"{type(obj).__name__}.{name} must be >= 0, got {q}")
    _set(obj, name, q)


def _check_array(
    owner: str, name: str, arr: object, dtype: type[np.generic], ndim: int
) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{owner}.{name} must be a numpy array, got {type(arr).__name__}")
    if arr.dtype != np.dtype(dtype):
        raise TypeError(f"{owner}.{name} must have dtype {np.dtype(dtype)}, got {arr.dtype}")
    if arr.ndim != ndim:
        raise ValueError(f"{owner}.{name} must be {ndim}-D, got shape {arr.shape}")
    return arr


def config_hash(state: np.ndarray, enabled: bool = True) -> str:
    """blake2b-128 hex digest of a panel configuration (``CommandRecord.config_hash``).

    The hashed bytes are the state vector as little-endian int16 followed by one byte,
    ``0x01`` if the panel is enabled and ``0x00`` otherwise.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(state, dtype="<i2").tobytes())
    h.update(b"\x01" if enabled else b"\x00")
    return h.hexdigest()


def check_link(
    fc_tx_hz: float, rate_tx_sps: Fraction | int, fc_rx_hz: float, rate_rx_sps: Fraction | int
) -> bool:
    """Whether a TX stream links to an RX stream (section 3.2).

    A link exists between streams of equal rate whose carriers satisfy
    ``|fc_tx - fc_rx| < f_s/2``. Streams whose Nyquist bands overlap
    (``|fc_tx - fc_rx| < (f_s,tx + f_s,rx)/2``) at different rates raise
    ``RateMismatchError`` (release 1 has no resampler).
    """
    df = abs(float(fc_tx_hz) - float(fc_rx_hz))
    ftx = parse_exact(rate_tx_sps, name="rate_tx_sps")  # exact rates only (floats raise)
    frx = parse_exact(rate_rx_sps, name="rate_rx_sps")
    if ftx == frx:
        return df < ftx / 2
    if df < (ftx + frx) / 2:
        raise RateMismatchError(
            f"bands overlap at different rates ({ftx} vs {frx} S/s, |dfc| = {df} Hz)"
        )
    return False


# --------------------------------------------------------------------------- propagation


@dataclass(frozen=True, eq=False)
class PathBundle:
    """Non-RIS deterministic paths of one link at snapshot time ``t``.

    * ``link``: link id; ``t``: snapshot time, s (exact).
    * ``amp``: (P,) complex128, dimensionless. Antenna gains sqrt(G_T G_R), lambda/(4 pi d),
      Fresnel Gamma, F(nu), partition factor. EXCLUDES the propagation phase
      ``exp(-j 2 pi f_c tau)``.
    * ``delay_s``: (P,) float64, absolute propagation delay, s.
    * ``kind``: (P,) int8 ``PathKind`` values.
    """

    link: LinkId
    t: Fraction
    amp: np.ndarray
    delay_s: np.ndarray
    kind: np.ndarray

    def __post_init__(self) -> None:
        _coerce_exact(self, "t")
        n = _check_array("PathBundle", "amp", self.amp, np.complex128, 1).shape[0]
        for name, dt in (("delay_s", np.float64), ("kind", np.int8)):
            if _check_array("PathBundle", name, getattr(self, name), dt, 1).shape[0] != n:
                raise ValueError(f"PathBundle.{name} length differs from amp length {n}")

    @property
    def n_paths(self) -> int:
        """Number of paths P."""
        return int(self.amp.shape[0])


@dataclass(frozen=True, eq=False)
class RISCoefficients:
    """RIS coefficients of one panel at one snapshot (section 3.4).

    * ``rows``: R entries ``(link, hop_variant)``; variant 0 direct, 1 Tx-image,
      2 Rx-image, 3 both (ground image expansion).
    * ``tile_of_cell``: (N,) int32 in [0, K): tile of each physical cell.
    * ``tile_delay_s``: (R, K) float64 reference delay Tx -> tile reference point -> Rx, s.
    * ``B``: (R, S, N) complex128 per state and physical cell:
      Gamma(s,f,theta_i,theta_s) * A * o * sqrt(G_T G_R) * ground factors * j/(4 pi r_i r_s)
      * exp(-jk(r_i + r_s)) * exp(+j 2 pi f_c tau_tile), i.e. the phase RELATIVE to the
      cell's tile reference. ``None`` when the separable form is used.
    * ``C``: (R, N) complex128 separable form (same without Gamma), with ``gamma``: (S,)
      complex128 alphabet (efficiency already applied). Exactly one of ``B`` and
      (``C``, ``gamma``) is given.
    * ``gamma_dead``: reflection coefficient of dead cells (state -1); default 0 (absorbing).
    """

    panel: PanelId
    t: Fraction
    rows: tuple[tuple[LinkId, int], ...]
    tile_of_cell: np.ndarray
    tile_delay_s: np.ndarray
    B: np.ndarray | None
    C: np.ndarray | None
    gamma: np.ndarray | None
    gamma_dead: complex = 0j

    def __post_init__(self) -> None:
        _coerce_exact(self, "t")
        _set(self, "rows", tuple((str(lk), int(v)) for lk, v in self.rows))
        r = len(self.rows)
        if any(v not in (0, 1, 2, 3) for _, v in self.rows):
            raise ValueError("RISCoefficients.rows hop_variant must be 0, 1, 2 or 3")
        n = _check_array("RISCoefficients", "tile_of_cell", self.tile_of_cell, np.int32, 1).shape[0]
        td = _check_array("RISCoefficients", "tile_delay_s", self.tile_delay_s, np.float64, 2)
        if td.shape[0] != r:
            raise ValueError(f"tile_delay_s has {td.shape[0]} rows, expected R = {r}")
        k = td.shape[1]
        if n and (int(self.tile_of_cell.min()) < 0 or int(self.tile_of_cell.max()) >= k):
            raise ValueError(f"tile_of_cell values must lie in [0, {k})")
        separable = self.C is not None or self.gamma is not None
        if (self.B is None) == (not separable):
            raise ValueError("give exactly one of B or (C, gamma)")
        if self.B is not None:
            b = _check_array("RISCoefficients", "B", self.B, np.complex128, 3)
            if b.shape[0] != r or b.shape[2] != n:
                raise ValueError(f"B shape {b.shape} != (R={r}, S, N={n})")
        else:
            if self.C is None or self.gamma is None:
                raise ValueError("the separable form needs both C and gamma")
            c = _check_array("RISCoefficients", "C", self.C, np.complex128, 2)
            if c.shape != (r, n):
                raise ValueError(f"C shape {c.shape} != (R={r}, N={n})")
            _check_array("RISCoefficients", "gamma", self.gamma, np.complex128, 1)
        _set(self, "gamma_dead", complex(self.gamma_dead))

    @property
    def n_rows(self) -> int:
        """Number of rows R."""
        return len(self.rows)

    @property
    def n_cells(self) -> int:
        """Number of physical cells N."""
        return int(self.tile_of_cell.shape[0])

    @property
    def n_tiles(self) -> int:
        """Number of tiles K."""
        return int(self.tile_delay_s.shape[1])

    @property
    def n_states(self) -> int:
        """Number of states S."""
        if self.B is not None:
            return int(self.B.shape[1])
        assert self.gamma is not None
        return int(self.gamma.shape[0])

    @property
    def separable(self) -> bool:
        """True for the separable (C, gamma) form."""
        return self.B is None


@dataclass(frozen=True, eq=False)
class Snapshot:
    """Geometry evaluated at time ``t`` (s): non-RIS paths per link and RIS coefficients.

    ``diag`` carries diagnostics (energy budget, Richardson estimate, max off-plane angle,
    interpolation-bound inputs) as plain floats.
    """

    t: Fraction
    paths: Mapping[LinkId, PathBundle]
    ris: Mapping[PanelId, RISCoefficients]
    diag: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _coerce_exact(self, "t")


@dataclass(frozen=True, eq=False)
class PanelConfig:
    """Applied configuration of one panel.

    ``state``: (N,) int16 per PHYSICAL cell after granularity expansion, stuck-at and dead
    handling; -1 marks a dead cell. ``enabled`` is the explicit "RIS absent" flag (RQ-12),
    distinct from every state index.
    """

    panel: PanelId
    state: np.ndarray
    enabled: bool = True

    def __post_init__(self) -> None:
        st = _check_array("PanelConfig", "state", self.state, np.int16, 1)
        if st.size and int(st.min()) < -1:
            raise ValueError("PanelConfig.state values must be >= -1 (-1 = dead)")
        _set(self, "enabled", bool(self.enabled))

    def config_hash(self) -> str:
        """blake2b-128 of (state int16 bytes, enabled flag); see :func:`config_hash`."""
        return config_hash(self.state, self.enabled)


@dataclass(frozen=True, eq=False)
class LinkPaths:
    """``combine()`` output: all paths of one link with the configuration applied.

    ``amp`` (P_tot,) complex128 (excluding the propagation phase), ``delay_s`` (P_tot,)
    float64 s, ``kind`` (P_tot,) int8, ``tile`` (P_tot,) int32: -1 for non-RIS paths, else a
    flat (panel, row, tile) key. One RIS_TILE path per (panel, row, tile).
    """

    link: LinkId
    t: Fraction
    amp: np.ndarray
    delay_s: np.ndarray
    kind: np.ndarray
    tile: np.ndarray

    def __post_init__(self) -> None:
        _coerce_exact(self, "t")
        n = _check_array("LinkPaths", "amp", self.amp, np.complex128, 1).shape[0]
        for name, dt in (("delay_s", np.float64), ("kind", np.int8), ("tile", np.int32)):
            if _check_array("LinkPaths", name, getattr(self, name), dt, 1).shape[0] != n:
                raise ValueError(f"LinkPaths.{name} length differs from amp length {n}")

    @property
    def n_paths(self) -> int:
        """Number of paths P_tot."""
        return int(self.amp.shape[0])


@dataclass(frozen=True)
class MotionBounds:
    """Motion bounds for the T_u / tile policy (section 4.4).

    Per link: ``v_rel_max_mps`` (maximum relative speed of the link's endpoints, m/s),
    ``v_perp_max_mps`` (maximum speed component perpendicular to the path, m/s) and
    ``r_min_m`` (minimum path-segment length over the run, m). Per panel:
    ``panel_r_min_m`` (minimum distance from the panel to any linked node, m) and
    ``panel_v_rel_max_mps``. ``static`` is True when nothing moves (event-only snapshots).
    """

    static: bool
    v_rel_max_mps: Mapping[LinkId, float] = field(default_factory=dict)
    v_perp_max_mps: Mapping[LinkId, float] = field(default_factory=dict)
    r_min_m: Mapping[LinkId, float] = field(default_factory=dict)
    panel_r_min_m: Mapping[PanelId, float] = field(default_factory=dict)
    panel_v_rel_max_mps: Mapping[PanelId, float] = field(default_factory=dict)


@runtime_checkable
class PropagationModel(Protocol):
    """Geometry and configuration model used by the engine (WP2 implements; section 3.4)."""

    def links(self) -> tuple[LinkId, ...]:
        """All links in scenario order."""
        ...

    def snapshot(self, t: Fraction) -> Snapshot:
        """Geometry at time ``t`` (s); a pure function of (scenario, t)."""
        ...

    def combine(
        self, snap: Snapshot, cfg: Mapping[PanelId, PanelConfig]
    ) -> Mapping[LinkId, LinkPaths]:
        """Apply panel configurations; linear in Gamma."""
        ...

    def ris_delay_s(self, panel: PanelId, rx: StreamId, t: Fraction) -> float:
        """tau_RIS->rx in s at the panel centre, at time ``t``."""
        ...

    def motion_bounds(self) -> MotionBounds:
        """Motion bounds for the T_u / tile policy."""
        ...


# --------------------------------------------------------------------------- controllers


_JITTER_RE: Final = re.compile(r"^(uniform):([^,]+),([^,]+)$|^(exp):([^,]+)$")


@dataclass(frozen=True)
class LatencyModel:
    """Controller latency model (section 4.9).

    ``fixed_s``: fixed latency, s (exact). ``jitter``: ``"uniform:a,b"`` (s) or ``"exp:mean"``
    (s), drawn from the keyed RNG purpose ``"ctrl.latency/<name>"``. ``mode``: ``"virtual"``
    (lockstep) or ``"wall"`` (paced).
    """

    fixed_s: Fraction = Fraction(0)
    jitter: str | None = None
    mode: Literal["virtual", "wall"] = "virtual"

    def __post_init__(self) -> None:
        _coerce_exact(self, "fixed_s")
        if self.mode not in ("virtual", "wall"):
            raise ConfigurationError(
                f"LatencyModel.mode must be 'virtual' or 'wall', got {self.mode!r}"
            )
        self.parsed_jitter()

    def parsed_jitter(self) -> tuple[str, tuple[Fraction, ...]] | None:
        """``("uniform", (a, b))`` or ``("exp", (mean,))`` in exact seconds, or None."""
        if self.jitter is None:
            return None
        m = _JITTER_RE.match(self.jitter.replace(" ", ""))
        if m is None:
            raise ConfigurationError(
                f"LatencyModel.jitter {self.jitter!r}: expected 'uniform:a,b' or 'exp:mean'"
            )
        if m.group(1):
            a = parse_exact(m.group(2), name="jitter a")
            b = parse_exact(m.group(3), name="jitter b")
            if not 0 <= a <= b:
                raise ConfigurationError(f"uniform jitter needs 0 <= a <= b, got {a}, {b}")
            return "uniform", (a, b)
        mean = parse_exact(m.group(5), name="jitter mean")
        if mean < 0:
            raise ConfigurationError(f"exp jitter mean must be >= 0, got {mean}")
        return "exp", (mean,)


@dataclass(frozen=True)
class ObservationSpec:
    """What a controller observes at each barrier (sections 4.9, 8.7).

    ``fields`` from {rx_power_dbm, ris_state, positions, csi_oracle, iq_window};
    ``window_s``: RX power averaging window before t_b, s; ``position_delay_s``: age of the
    reported positions, s; ``position_noise_m``: std of additive position noise, m.
    """

    fields: tuple[str, ...] = ("rx_power_dbm", "ris_state")
    window_s: Fraction = Fraction(1, 1000)
    position_delay_s: Fraction = Fraction(0)
    position_noise_m: float = 0.0

    def __post_init__(self) -> None:
        _set(self, "fields", tuple(self.fields))
        bad = set(self.fields) - _OBS_FIELDS
        if bad:
            raise ConfigurationError(f"unknown observation fields {sorted(bad)}")
        _coerce_exact(self, "window_s")
        _coerce_exact(self, "position_delay_s")
        if self.window_s <= 0:
            raise ConfigurationError("ObservationSpec.window_s must be > 0")
        if not self.position_noise_m >= 0.0:
            raise ConfigurationError("ObservationSpec.position_noise_m must be >= 0")


@dataclass(frozen=True, eq=False)
class CsiView:
    """Oracle channel state of one link at time ``t`` (use is logged as ``oracle_csi_used``).

    ``h_direct``: sum of all non-RIS paths INCLUDING their carrier phase, dimensionless.
    ``cascade``: per panel, (S, N) complex128 contribution of cell n in state s, including
    the carrier phase (for separable tables row s equals ``C * gamma[s]``). Then
    ``h(cfg) = h_direct + sum_p sum_n cascade[p][s_n, n]``; dead cells (-1) contribute 0.
    ``fc_hz``: carrier of the link's TX stream, Hz.
    """

    link: LinkId
    t: Fraction
    fc_hz: float
    h_direct: complex
    cascade: Mapping[PanelId, np.ndarray]

    def __post_init__(self) -> None:
        _coerce_exact(self, "t")
        _set(self, "h_direct", complex(self.h_direct))
        for p, z in self.cascade.items():
            _check_array("CsiView", f"cascade[{p!r}]", z, np.complex128, 2)

    def evaluate(self, states: Mapping[PanelId, np.ndarray]) -> complex:
        """Channel coefficient for per-cell states (int16, -1 = dead) of every panel."""
        h = complex(self.h_direct)
        for p, z in self.cascade.items():
            s = np.asarray(states[p], dtype=np.int64)
            if s.shape != (z.shape[1],):
                raise ValueError(f"CsiView.evaluate: panel {p!r} needs ({z.shape[1]},) states")
            if s.size and (int(s.min()) < -1 or int(s.max()) >= z.shape[0]):
                raise ValueError(f"CsiView.evaluate: panel {p!r} state outside [-1, {z.shape[0]})")
            alive = s >= 0
            cols = np.nonzero(alive)[0]
            h += complex(z[s[alive], cols].sum())
        return h


@dataclass(frozen=True)
class TransitionSpec:
    """Non-instant reconfiguration request (section 4.7).

    ``model``: ``step`` | ``linear`` (over ``duration_s``) | ``first_order``
    (``1 - exp(-t/duration_s)``). ``duration_s``: s (exact); ``None`` means the panel's
    ``reconfiguration_time_s``. ``classes``: ``single`` (one class) or ``by_edge``
    (rising and falling edges are separate classes).
    """

    model: TransitionModel = "step"
    duration_s: Fraction | None = None
    classes: Literal["single", "by_edge"] = "single"

    def __post_init__(self) -> None:
        if self.model not in ("step", "linear", "first_order"):
            raise ConfigurationError(f"unknown transition model {self.model!r}")
        if self.classes not in ("single", "by_edge"):
            raise ConfigurationError(f"unknown transition classes {self.classes!r}")
        _coerce_exact(self, "duration_s", allow_none=True)


@dataclass(frozen=True, eq=False)
class Action:
    """A controller decision for one panel (section 4.9).

    Exactly one of ``states`` ((N_ctrl,) int16), ``phases_rad`` ((N_ctrl,) float64, rad,
    quantized by the panel) or ``codebook_index`` may be given; ``enabled`` (sremu addition,
    counterfactual ``ris.set_enabled``) may be combined with none of them. ``at_s``: switch
    time at the RIS, s (default ``t_b + latency``). ``transition`` overrides the panel default.
    """

    panel: PanelId
    states: np.ndarray | None = None
    phases_rad: np.ndarray | None = None
    codebook_index: int | None = None
    at_s: Fraction | None = None
    transition: TransitionSpec | None = None
    enabled: bool | None = None

    def __post_init__(self) -> None:
        given = [x is not None for x in (self.states, self.phases_rad, self.codebook_index)]
        if sum(given) > 1:
            raise ConfigurationError(
                "Action: give at most one of states, phases_rad, codebook_index"
            )
        if sum(given) == 0 and self.enabled is None:
            raise ConfigurationError(
                "Action: nothing to do (no states, phases, codebook or enabled)"
            )
        if sum(given) == 1 and self.enabled is not None:
            raise ConfigurationError("Action: 'enabled' cannot be combined with a configuration")
        if self.states is not None:
            _check_array("Action", "states", self.states, np.int16, 1)
        if self.phases_rad is not None:
            _check_array("Action", "phases_rad", self.phases_rad, np.float64, 1)
        _coerce_exact(self, "at_s", allow_none=True)


@dataclass(frozen=True, eq=False)
class Observation:
    """What a barrier-registered controller sees at barrier time ``t_b`` (section 4.9).

    ``k_b``: per RX stream, ``to_ticks(t_b, f_r)``; observation data use samples < k_b only.
    ``rx_power_dbm``: measured from real RX IQ over ``window_s`` (noise and impairments
    included), dBm. ``positions``: node id -> (3,) float64 m, or None. ``csi_oracle``: per
    link, or None (oracle use is flagged). ``iq_window``: per RX stream (n,) complex64, or
    None. ``ris_state``: per panel (N,) int16 applied at t_b. ``pending``: receipts of
    commands scheduled after t_b.
    """

    t_b: Fraction
    k_b: Mapping[StreamId, int]
    rx_power_dbm: Mapping[StreamId, float]
    positions: Mapping[str, np.ndarray] | None
    csi_oracle: Mapping[LinkId, CsiView] | None
    iq_window: Mapping[StreamId, np.ndarray] | None
    ris_state: Mapping[PanelId, np.ndarray]
    pending: tuple[CommandReceipt, ...]

    def __post_init__(self) -> None:
        _coerce_exact(self, "t_b")
        _set(self, "pending", tuple(self.pending))


@dataclass(frozen=True)
class RisCapabilities:
    """Panel capabilities, ETSI GS RIS 008 parameter names where they exist (8.5, 8.9).

    ``n_elements``: physical cells N. ``layout``: (n_u, n_v). ``control_granularity``:
    element | column | tile | groups. ``states``: state labels. ``gamma_source``:
    ``"measured:<sha256>"`` or ``"analytic:<model>"``. ``operating_band_hz``: (f_lo, f_hi),
    Hz. ``max_bandwidth_hz``: Hz. ``reconfiguration_time_s``: s (exact) or None.
    ``max_command_rate_hz``: Hz. ``static_power_w``: W (metadata only). ``n_controls``
    (sremu addition): N_ctrl, the length of a ``states`` command; defaults to ``n_elements``.
    """

    n_elements: int
    layout: tuple[int, int]
    control_granularity: Granularity
    states: tuple[str, ...]
    gamma_source: str
    operating_band_hz: tuple[float, float]
    max_bandwidth_hz: float
    reconfiguration_time_s: Fraction | None = None
    max_command_rate_hz: float = DEFAULT_MAX_COMMAND_RATE_HZ
    static_power_w: float | None = None
    n_controls: int | None = None

    def __post_init__(self) -> None:
        if self.n_elements < 1:
            raise ConfigurationError("RisCapabilities.n_elements must be >= 1")
        _set(self, "layout", (int(self.layout[0]), int(self.layout[1])))
        _set(self, "states", tuple(self.states))
        _set(
            self,
            "operating_band_hz",
            (float(self.operating_band_hz[0]), float(self.operating_band_hz[1])),
        )
        _coerce_exact(self, "reconfiguration_time_s", allow_none=True)
        if self.n_controls is None:
            _set(self, "n_controls", int(self.n_elements))
        if not (
            self.gamma_source.startswith("measured:") or self.gamma_source.startswith("analytic:")
        ):
            raise ConfigurationError("gamma_source must start with 'measured:' or 'analytic:'")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (exact times as ``"p/q"`` strings)."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["layout"] = list(self.layout)
        d["states"] = list(self.states)
        d["operating_band_hz"] = list(self.operating_band_hz)
        if self.reconfiguration_time_s is not None:
            d["reconfiguration_time_s"] = fraction_to_str(self.reconfiguration_time_s)
        return d


@runtime_checkable
class Controller(Protocol):
    """Controller plugin (sections 4.9, 8.7). In-process controllers run at barriers.

    Attributes: ``name``; ``period_s`` (barrier period T_c, s); ``min_latency_s`` (L_min, s);
    ``spec`` (ObservationSpec); ``latency`` (LatencyModel). ``setup`` receives the panel
    capabilities and a keyed ``np.random.Generator`` (purpose ``ctrl/<name>``).
    """

    name: str
    period_s: Fraction
    min_latency_s: Fraction
    spec: ObservationSpec
    latency: LatencyModel

    def setup(self, caps: Mapping[PanelId, RisCapabilities], rng: np.random.Generator) -> None:
        """Called once before the first barrier."""
        ...

    def decide(self, obs: Observation) -> Sequence[Action]:
        """Return zero or more actions for the barrier at ``obs.t_b``."""
        ...


@runtime_checkable
class RemoteControllerProxy(Protocol):
    """Engine-side stand-in for a controller registered over the control socket (8.5).

    The engine sends ``ctrl.barrier`` with the observation and waits for ``ctrl.ack {t_b}``;
    the remote controller's ``ris.set_config`` requests arrive through ``schedule()``.
    """

    name: str
    period_s: Fraction
    min_latency_s: Fraction
    spec: ObservationSpec
    latency: LatencyModel

    def send_barrier(self, obs: Observation) -> None:
        """Send ``ctrl.barrier {t_b, k_b, observation}`` to the remote controller."""
        ...

    def wait_ack(self, t_b: Fraction, timeout_s: float) -> bool:
        """Block until ``ctrl.ack {t_b}`` arrives; False on timeout (wall seconds)."""
        ...


# --------------------------------------------------------------------------- commands


@dataclass(frozen=True)
class CommandTime:
    """When a command applies (section 4.6): exact RIS time, a sample of a stream, or asap.

    * ``kind="t"``: ``t`` is the switch time at the RIS, s (exact).
    * ``kind="sample"``: sample ``index`` of ``stream``; converted to
      ``t = from_ticks(index) - tau_RIS->stream``.
    * ``kind="asap"``: ``t_req`` = the frontier time at arrival.
    """

    kind: Literal["t", "sample", "asap"]
    t: Fraction | None = None
    stream: StreamId | None = None
    index: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "t":
            _coerce_exact(self, "t")
            if self.stream is not None or self.index is not None:
                raise ConfigurationError("CommandTime(kind='t') takes only t")
        elif self.kind == "sample":
            if self.stream is None or self.index is None or self.t is not None:
                raise ConfigurationError("CommandTime(kind='sample') takes stream and index")
            if isinstance(self.index, bool) or int(self.index) != self.index or self.index < 0:
                raise ConfigurationError("CommandTime.index must be an int >= 0")
            _set(self, "index", int(self.index))
            split_stream_id(self.stream)
        elif self.kind == "asap":
            if self.t is not None or self.stream is not None or self.index is not None:
                raise ConfigurationError("CommandTime(kind='asap') takes no arguments")
        else:
            raise ConfigurationError(f"unknown CommandTime kind {self.kind!r}")

    @classmethod
    def at(cls, t: Fraction | int | str) -> CommandTime:
        """Switch at exact RIS time ``t`` (s)."""
        return cls("t", t=parse_exact(t, name="t"))

    @classmethod
    def at_sample(cls, stream: StreamId, index: int) -> CommandTime:
        """Switch so that ``stream`` sees it at sample ``index``."""
        return cls("sample", stream=stream, index=index)

    @classmethod
    def asap(cls) -> CommandTime:
        """Switch at the frontier time on arrival."""
        return cls("asap")

    def to_json(self) -> dict[str, Any] | str:
        """Control-envelope form: ``{"t": "p/q"}``, ``{"sample": {...}}`` or ``"asap"``."""
        if self.kind == "t":
            assert self.t is not None
            return {"t": fraction_to_str(self.t)}
        if self.kind == "sample":
            return {"sample": {"stream": self.stream, "index": self.index}}
        return "asap"

    @classmethod
    def from_json(cls, obj: Any) -> CommandTime:
        """Parse the control-envelope form (exact strings only for ``t``)."""
        if obj == "asap":
            return cls.asap()
        if isinstance(obj, Mapping) and set(obj) == {"t"}:
            return cls.at(obj["t"])
        if isinstance(obj, Mapping) and set(obj) == {"sample"}:
            s = obj["sample"]
            if not isinstance(s, Mapping) or set(s) != {"stream", "index"}:
                raise ConfigurationError(f"bad 'sample' time {s!r}")
            return cls.at_sample(str(s["stream"]), s["index"])
        raise ConfigurationError(f"bad command time {obj!r}")


@dataclass(frozen=True, eq=False)
class CommandRequest:
    """A timed command handed to ``EngineProtocol.schedule`` (sections 4.6, 8.5).

    ``cmd_id``: client id (<= 64 chars). ``op``: one of ``CommandOp``. ``src``: issuing
    client or controller. ``target``: panel id for ``ris.*``, node id for ``node.*``.
    ``at``: ``CommandTime``. Payload by op:

    * ``ris.set_config``: exactly one of ``states`` ((N_ctrl,) int16), ``phases_rad``
      ((N_ctrl,) float64 rad, with ``quantizer``) or ``codebook_index``; ``elements`` is
      ``"all"``, a tuple of element indices, or a group name; optional ``transition``.
    * ``ris.set_enabled``: ``enabled`` (counterfactual; logged as such).
    * ``node.set_position``: ``position_m`` (3,) float64, m.
    * ``node.set_trajectory``: ``payload`` holds the trajectory spec (scenario schema).
    * ``rec.marker``: ``payload`` holds the annotation.

    ``late_policy``: ``apply_on_arrival`` (default) or ``reject``. ``t_wall_recv_ns``:
    monotonic wall clock when the request was parsed (metrics only).
    """

    cmd_id: str
    op: CommandOp
    src: str
    target: str
    at: CommandTime
    states: np.ndarray | None = None
    phases_rad: np.ndarray | None = None
    quantizer: str = "max_real"
    codebook_index: int | None = None
    elements: str | tuple[int, ...] = "all"
    enabled: bool | None = None
    position_m: np.ndarray | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    transition: TransitionSpec | None = None
    late_policy: LatePolicy = "apply_on_arrival"
    t_wall_recv_ns: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= len(self.cmd_id) <= MAX_CMD_ID_LEN:
            raise TooLarge(f"cmd_id must have 1..{MAX_CMD_ID_LEN} characters")
        if self.op not in _COMMAND_OPS:
            raise ConfigurationError(f"unknown command op {self.op!r}")
        if self.late_policy not in ("apply_on_arrival", "reject"):
            raise ConfigurationError(f"unknown late_policy {self.late_policy!r}")
        if not isinstance(self.at, CommandTime):
            raise TypeError("CommandRequest.at must be a CommandTime")
        if not isinstance(self.elements, str):
            _set(self, "elements", tuple(int(i) for i in self.elements))
            if len(self.elements) > MAX_ELEMENTS_PER_COMMAND:
                raise TooLarge(f"more than {MAX_ELEMENTS_PER_COMMAND} elements in one command")
        for name in ("states", "phases_rad"):
            arr = getattr(self, name)
            if arr is not None:
                dt = np.int16 if name == "states" else np.float64
                if _check_array("CommandRequest", name, arr, dt, 1).size > MAX_ELEMENTS_PER_COMMAND:
                    raise TooLarge(f"more than {MAX_ELEMENTS_PER_COMMAND} elements in one command")
        if self.op == "ris.set_config":
            n = sum(x is not None for x in (self.states, self.phases_rad, self.codebook_index))
            if n != 1:
                raise ConfigurationError(
                    "ris.set_config needs exactly one of states, phases_rad, codebook_index"
                )
        elif self.op == "ris.set_enabled":
            if self.enabled is None:
                raise ConfigurationError("ris.set_enabled needs 'enabled'")
        elif self.op == "node.set_position":
            if self.position_m is None:
                raise ConfigurationError("node.set_position needs position_m")
            _check_array("CommandRequest", "position_m", self.position_m, np.float64, 1)
            if self.position_m.shape != (3,):
                raise ConfigurationError("position_m must have shape (3,)")

    @property
    def counterfactual(self) -> bool:
        """True for ``ris.set_enabled`` (an A/B baseline, not a physical action)."""
        return self.op == "ris.set_enabled"


@dataclass(frozen=True)
class CommandReceipt:
    """Reply to ``schedule`` (section 8.5 reply fields).

    On success ``ok`` is True, ``status`` is ``"scheduled"`` (or ``"late_applied"`` when moved
    by ``apply_on_arrival``), ``seq`` is the engine-assigned monotone sequence number, and
    ``t_req`` (s) and ``k_req`` (per affected RX stream) are the requested switch time and
    samples. On failure ``ok`` is False, ``status`` is ``"rejected"`` and ``error``/``msg``
    explain why.
    """

    cmd_id: str
    ok: bool
    status: CommandStatus
    seq: int | None = None
    t_req: Fraction | None = None
    k_req: Mapping[StreamId, int] = field(default_factory=dict)
    error: ErrorCode = ErrorCode.NONE
    msg: str = ""
    t_wall_recv_ns: int | None = None

    def __post_init__(self) -> None:
        _coerce_exact(self, "t_req", allow_none=True)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; exact times as ``"p/q"``; ``error`` by name."""
        d: dict[str, Any] = {"id": self.cmd_id, "ok": self.ok, "status": self.status}
        if self.ok:
            d.update(
                seq=self.seq,
                t_req=None if self.t_req is None else fraction_to_str(self.t_req),
                k_req=dict(self.k_req),
                t_wall_recv_ns=self.t_wall_recv_ns,
            )
        else:
            d["error"] = {"code": self.error.name, "msg": self.msg}
        return d


@dataclass
class CommandRecord:
    """One record per command: ``events.jsonl`` line and SigMF annotation (section 4.6).

    ``panel`` holds the command target (panel id for ``ris.*``, node id for ``node.*``).
    ``t_req``/``t_applied``: switch time at the RIS, s (exact; ``t_applied`` None until
    applied or when rejected). ``k_req``/``k_applied``: per affected RX stream sample index;
    ``lateness_samples = k_applied - k_req``. ``t_wall_*_ns``: monotonic wall clock (metrics
    only): request parsed, page with ``k_applied`` computed, page handed to the client.
    ``config_hash``: :func:`config_hash` of the applied configuration.
    """

    cmd_id: str
    seq: int
    src: str
    op: str
    panel: PanelId
    t_req: Fraction
    t_applied: Fraction | None
    k_req: dict[StreamId, int]
    k_applied: dict[StreamId, int]
    lateness_samples: dict[StreamId, int]
    t_wall_recv_ns: int | None
    t_wall_computed_ns: int | None
    t_wall_visible_ns: int | None
    status: CommandStatus
    config_hash: str
    transition: dict[str, Any] | None
    counterfactual: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict in field order; exact times as ``"p/q"`` strings."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["t_req"] = fraction_to_str(self.t_req)
        d["t_applied"] = None if self.t_applied is None else fraction_to_str(self.t_applied)
        for key in ("k_req", "k_applied", "lateness_samples"):
            d[key] = dict(d[key])
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CommandRecord:
        """Inverse of :meth:`to_dict` (exact strings back to ``Fraction``)."""
        kw = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kw["t_req"] = parse_exact(d["t_req"], name="t_req")
        kw["t_applied"] = None if d.get("t_applied") is None else parse_exact(d["t_applied"])
        for key in ("k_req", "k_applied", "lateness_samples"):
            kw[key] = {str(k): int(v) for k, v in d[key].items()}
        return cls(**kw)


# --------------------------------------------------------------------------- streams


@dataclass(frozen=True)
class StreamSpec:
    """A data-plane stream to open (normalized form of the API's ``StreamArgs``).

    ``stream``: ``<node>.<stream>``. ``direction``: tx | rx. ``fc_hz``: carrier, Hz.
    ``rate_sps``: exact sample rate, S/s (str/int/Fraction; floats rejected). ``cpu``: host
    sample format (fc32 = complex64, sc16 = int16 I/Q). ``nchan``: must be 1 (release 1).
    """

    stream: StreamId
    direction: Direction
    fc_hz: float
    rate_sps: Fraction
    cpu: CpuFormat = "fc32"
    nchan: int = 1

    def __post_init__(self) -> None:
        split_stream_id(self.stream)
        if self.direction not in ("tx", "rx"):
            raise ConfigurationError(f"direction must be 'tx' or 'rx', got {self.direction!r}")
        if self.cpu not in ("fc32", "sc16"):
            raise ConfigurationError(f"cpu format must be 'fc32' or 'sc16', got {self.cpu!r}")
        if self.nchan != 1:
            raise ConfigurationError(f"nchan must be 1 in release 1, got {self.nchan}")
        if not float(self.fc_hz) > 0.0:
            raise ConfigurationError(f"fc_hz must be > 0, got {self.fc_hz}")
        _set(self, "fc_hz", float(self.fc_hz))
        _set(self, "rate_sps", Fraction(Rate(self.rate_sps)))

    @property
    def node(self) -> NodeId:
        """Node id part of the stream id."""
        return split_stream_id(self.stream)[0]


@dataclass(frozen=True)
class StreamHandle:
    """An open stream: engine-assigned ``id`` (uint32, also the frame ``stream_id``),
    its ``spec`` and its ``page_size`` in samples (section 4.3)."""

    id: int
    spec: StreamSpec
    page_size: int


@dataclass
class TxMetadata:
    """UHD ``tx_metadata_t`` plus sremu extensions.

    ``has_time_spec``/``time_spec``: timed burst start (R3 mapping). ``start_of_burst``/
    ``end_of_burst``: burst flags; after EOB the stream is silent until its next timed burst.
    ``idle_until`` (sremu): with empty samples, commit the stream through
    ``to_ticks(time_spec)`` (frame flag IDLE_UNTIL). ``end_of_stream`` (sremu): the client
    will send nothing more.
    """

    has_time_spec: bool = False
    time_spec: TimeSpec = field(default_factory=TimeSpec)
    start_of_burst: bool = False
    end_of_burst: bool = False
    idle_until: bool = False
    end_of_stream: bool = False


@dataclass
class RxMetadata:
    """UHD ``rx_metadata_t`` plus sremu extensions (filled by ``recv``/``read_rx``).

    ``time_spec``: time of the first returned sample. ``error_code``: ``ErrorCode``.
    ``first_sample`` (sremu): absolute RX sample index of the first returned sample.
    """

    has_time_spec: bool = False
    time_spec: TimeSpec = field(default_factory=TimeSpec)
    more_fragments: bool = False
    fragment_offset: int = 0
    start_of_burst: bool = False
    end_of_burst: bool = False
    error_code: ErrorCode = ErrorCode.NONE
    out_of_sequence: bool = False
    first_sample: int | None = None


@dataclass(frozen=True)
class AsyncMetadata:
    """UHD ``async_metadata_t`` plus sremu extensions (TX events).

    ``event_code``: ``AsyncCode``. ``stream``: TX stream id. ``sample_index``: absolute TX
    sample index the event refers to; ``n_samples``: samples affected (I6 accounting).
    """

    event_code: AsyncCode
    stream: StreamId = ""
    channel: int = 0
    has_time_spec: bool = False
    time_spec: TimeSpec = field(default_factory=TimeSpec)
    sample_index: int | None = None
    n_samples: int = 0


@dataclass(frozen=True)
class StreamCmd:
    """UHD ``stream_cmd_t``: ``mode`` start_cont | stop_cont | num_done | num_more;
    ``num_samps``; ``stream_now`` (untimed: start at the frontier) or ``time_spec``."""

    mode: StreamMode
    num_samps: int = 0
    stream_now: bool = True
    time_spec: TimeSpec = field(default_factory=TimeSpec)

    def __post_init__(self) -> None:
        if self.mode not in ("start_cont", "stop_cont", "num_done", "num_more"):
            raise ConfigurationError(f"unknown stream mode {self.mode!r}")
        if self.num_samps < 0:
            raise ConfigurationError("StreamCmd.num_samps must be >= 0")


@runtime_checkable
class EngineProtocol(Protocol):
    """What ``api/`` and ``transport/`` use (WP6 implements; ``FakeEngine`` in WP0).

    ``h`` is a ``StreamHandle``. Times are exact virtual seconds; ``timeout_s`` values are
    wall seconds.
    """

    def open_stream(self, spec: StreamSpec) -> StreamHandle:
        """Open a TX or RX stream."""
        ...

    def submit_tx(self, h: StreamHandle, samples: np.ndarray, md: TxMetadata) -> int:
        """Queue TX samples (complex64); returns the number of samples accepted."""
        ...

    def stream_cmd(self, h: StreamHandle, cmd: StreamCmd) -> None:
        """Issue an RX stream command."""
        ...

    def read_rx(self, h: StreamHandle, n: int, timeout_s: float) -> tuple[np.ndarray, RxMetadata]:
        """Read up to ``n`` RX samples (complex64) and their metadata."""
        ...

    def schedule(self, req: CommandRequest) -> CommandReceipt:
        """Schedule a timed command."""
        ...

    def register_controller(self, ctrl: Controller | RemoteControllerProxy) -> None:
        """Register a barrier controller."""
        ...

    def time_now(self) -> Fraction:
        """Virtual time (s) through which every RX stream has been computed."""
        ...

    def caps(self) -> Mapping[PanelId, RisCapabilities]:
        """Capabilities of every panel."""
        ...

    def recv_async(self, h: StreamHandle, timeout_s: float) -> AsyncMetadata | None:
        """Next asynchronous TX event of ``h``, or None."""
        ...


# --------------------------------------------------------------------------- render contract


@dataclass(frozen=True)
class FDSpec:
    """Fractional-delay filter choice (section 4.5).

    ``filter_id``: e.g. ``"kaiser8b8"``, or ``"integer"`` for integer-only delay. ``taps``: L.
    ``lookahead``: h_FD in samples (TX must be committed through ``k1 + h_FD``). ``beta``:
    Kaiser beta. ``n_phases``: polyphase table size.
    """

    filter_id: str
    taps: int
    lookahead: int
    beta: float | None = None
    n_phases: int = 1024

    @classmethod
    def integer(cls) -> FDSpec:
        """Integer-only delay (allowed only when B <= 0.0318 f_s)."""
        return cls("integer", 1, 0, None, 1)


@dataclass(frozen=True, eq=False)
class TransitionTerm:
    """One transition class active in a segment (section 4.7).

    ``k_start``: absolute RX sample where ``g_c(0)`` applies (``to_ticks(t_sw + tau)``).
    ``profile``: step | linear | first_order. ``duration_samples``: T_c (linear) or tau_c
    (first order) times f_s, in samples (may be fractional). ``dA_a``/``dA_b``: (P,)
    complex128 endpoint amplitudes of Delta h_c at t_a and t_b, aligned with the job's paths.
    """

    k_start: int
    profile: TransitionModel
    duration_samples: float
    dA_a: np.ndarray
    dA_b: np.ndarray


@runtime_checkable
class TxRing(Protocol):
    """TX history ring of one TX stream (section 4.5).

    ``buffer``: (H,) complex64 circular storage with H a power of two; absolute sample k
    lives at ``buffer[k & mask]``. Samples never written read as 0 (silence).
    """

    stream: StreamId
    f_s: Fraction
    capacity: int
    mask: int
    buffer: np.ndarray

    def committed_through(self) -> int:
        """First absolute index NOT yet committed (section 4.8)."""
        ...

    def oldest(self) -> int:
        """Smallest absolute index still retained."""
        ...

    def write(self, k0: int, samples: np.ndarray) -> None:
        """Write samples at absolute indices [k0, k0 + len)."""
        ...

    def read(self, k0: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        """Read absolute indices [k0, k0 + n) as complex64."""
        ...


@dataclass(eq=False)
class LinkJob:
    """Render work for one link in one segment (section 3.4, render contract).

    ``fc_hz``: carrier for the propagation phase, Hz. ``lo_offset_hz``: f_c,tx - f_c,rx, Hz.
    ``amp_a``/``amp_b``: (P,) complex128 endpoint amplitudes at t_a, t_b (complex-linear
    between). ``tau_a``/``tau_b``: (P,) float64 endpoint delays, s (chord between).
    ``group``: (P,) int16 FD group (one group per link when narrowband). ``fd``: filter.
    ``trans``: active transition terms. ``rot_state``: numba rotator carry within a page.
    """

    link: LinkId
    ring: TxRing
    fc_hz: float
    lo_offset_hz: float
    amp_a: np.ndarray
    amp_b: np.ndarray
    tau_a: np.ndarray
    tau_b: np.ndarray
    group: np.ndarray
    fd: FDSpec
    trans: list[TransitionTerm] = field(default_factory=list)
    rot_state: np.ndarray | None = None


@dataclass(eq=False)
class SegmentJob:
    """Render work for one RX stream on absolute samples [k0, k1).

    ``f_s``: RX rate, S/s (exact). ``t_a``/``t_b``: enclosing snapshot interval, s (exact).
    ``links``: link jobs in scenario order (D-3 summation order).
    """

    rx: StreamId
    f_s: Fraction
    k0: int
    k1: int
    t_a: Fraction
    t_b: Fraction
    links: list[LinkJob]


class RenderFn(Protocol):
    """``render(job, out)``: add the segment's received signal into ``out``.

    ``out`` is a complex128 accumulator of length ``k1 - k0``. Pure function of its inputs.
    """

    def __call__(self, job: SegmentJob, out: np.ndarray) -> None: ...


class NoiseFn(Protocol):
    """``noise(rx_key, k0, n, sigma)``: complex64 AWGN for absolute samples [k0, k0 + n).

    ``rx_key``: the two uint32 Philox key words of the RX noise purpose. ``sigma``: total
    complex standard deviation (E|w|^2 = sigma^2), in sqrt(W). Pure in the absolute index.
    """

    def __call__(self, rx_key: tuple[int, int], k0: int, n: int, sigma: float) -> np.ndarray: ...


@runtime_checkable
class RenderBackend(Protocol):
    """A render backend (``numpy_ref`` or ``numba_cpu``, section 7.1)."""

    name: BackendName

    def render(self, job: SegmentJob, out: np.ndarray) -> None:
        """See :class:`RenderFn`."""
        ...

    def noise(self, rx_key: tuple[int, int], k0: int, n: int, sigma: float) -> np.ndarray:
        """See :class:`NoiseFn`."""
        ...
