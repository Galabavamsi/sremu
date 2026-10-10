"""UHD-shaped TX and RX streamers (DESIGN.md section 8.1).

* :class:`StreamArgs` names a scenario stream (``<node>.<stream>``) with its carrier, exact
  rate and host sample format. API floats go through ``Fraction(repr(x))`` (R5), so
  ``rate_sps=5880.0`` is exactly 5880 S/s.
* :class:`TxStreamer` ``send`` hands samples and :class:`~sremu.contracts.TxMetadata` to the
  engine: timed bursts start at ``to_ticks(time_spec, f)`` (R3); untimed bursts start at
  the frontier. ``recv_async_msg`` returns BURST_ACK, UNDERFLOW, TIME_ERROR, SEQ_ERROR, ...
  ``idle_until`` and ``end_of_stream`` are the sremu lockstep-commitment extensions.
* :class:`RxStreamer` ``issue_stream_cmd`` and ``recv`` follow ``uhd::rx_streamer``:
  ``recv`` fills the buffer from one contiguous run of samples; ``md`` is filled in place
  (``time_spec`` of the first sample, ``error_code``, ``end_of_burst``, ``first_sample``).
  An error met after some samples were already copied is **queued for the next call**
  (UHD behaviour), so a partial buffer comes back with ``NONE`` and the next ``recv``
  returns 0 samples with the error (e.g. ``OVERFLOW`` with the post-gap ``time_spec``).
  Samples after a discontinuity are held back for the next call, so one ``recv`` never
  spans a gap.

**Formats.** ``fc32``: complex64 buffers of shape (n,) or (1, n). ``sc16``: int16 buffers
of shape (n, 2) or (1, n, 2) holding (I, Q); full scale is ``SC16_SCALE`` (32767), TX
``x = (I + jQ)/32767``, RX ``round(32767*x)`` saturated to the int16 range (the sc16 format
saturates, section 3.3). Recordings always hold the complex64 values the engine saw.

**Blocking.** The Device serializes engine calls with one lock and implements ``timeout``
(wall seconds) by polling the engine with timeout 0 and waiting on a condition that every
TX send, idle promise and command of this process signals (and at least every
``poll_s``); a ``threading.Timer`` ends the wait after ``timeout``, so wake-ups by other
threads never shorten it (no clock is read here; I7).
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction
from typing import TYPE_CHECKING, Final

import numpy as np

from sremu.contracts import (
    AsyncMetadata,
    CpuFormat,
    Direction,
    RxMetadata,
    StreamCmd,
    StreamHandle,
    StreamSpec,
    TxMetadata,
)
from sremu.errors import ConfigurationError, ErrorCode, StreamError
from sremu.timebase import Rate, TimeSpec, to_ticks

if TYPE_CHECKING:
    from sremu.api.device import Device

__all__ = ["SC16_SCALE", "RxStreamer", "StreamArgs", "TxStreamer"]

SC16_SCALE: Final[float] = 32767.0
"""Full scale of the sc16 host format: int16 32767 <-> 1.0 in fc32."""

_RETRYABLE: Final = frozenset({ErrorCode.TIMEOUT})


class StreamArgs:
    """Arguments of ``Device.get_tx_stream`` / ``get_rx_stream``.

    ``stream``: ``"<node>.<stream>"``. ``fc_hz``: carrier, Hz. ``rate_sps``: sample rate in
    S/s, exact (str, int or Fraction) or a float converted with ``Fraction(repr(x))`` (R5).
    ``cpu``: ``"fc32"`` (complex64) or ``"sc16"`` (int16 I/Q). ``nchan``: must be 1.
    """

    __slots__ = ("cpu", "fc_hz", "nchan", "rate_sps", "stream")

    def __init__(
        self,
        stream: str,
        fc_hz: float,
        rate_sps: str | int | Fraction | float,
        cpu: CpuFormat = "fc32",
        nchan: int = 1,
    ) -> None:
        self.stream = stream
        self.fc_hz = float(fc_hz)
        self.rate_sps: Fraction = Fraction(Rate.from_api(rate_sps))
        self.cpu: CpuFormat = cpu
        self.nchan = int(nchan)
        self.spec("rx")  # validate now (stream id, cpu format, nchan, fc)

    def spec(self, direction: Direction) -> StreamSpec:
        """The engine's ``StreamSpec`` for ``direction``."""
        return StreamSpec(self.stream, direction, self.fc_hz, self.rate_sps, self.cpu, self.nchan)

    def __repr__(self) -> str:
        return (
            f"StreamArgs(stream={self.stream!r}, fc_hz={self.fc_hz!r}, "
            f"rate_sps={str(self.rate_sps)!r}, cpu={self.cpu!r}, nchan={self.nchan})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StreamArgs):
            return NotImplemented
        return (self.stream, self.fc_hz, self.rate_sps, self.cpu, self.nchan) == (
            other.stream,
            other.fc_hz,
            other.rate_sps,
            other.cpu,
            other.nchan,
        )

    def __hash__(self) -> int:
        return hash((self.stream, self.fc_hz, self.rate_sps, self.cpu, self.nchan))


# =========================================================================== conversions


def _one_channel(arr: np.ndarray, ndim: int, what: str) -> np.ndarray:
    """Strip a leading channel axis of length 1; reject nchan > 1."""
    if arr.ndim == ndim + 1:
        if arr.shape[0] != 1:
            raise ConfigurationError(f"{what}: nchan must be 1, got a buffer for {arr.shape[0]}")
        arr = arr[0]
    if arr.ndim != ndim:
        raise ValueError(f"{what}: unexpected buffer shape {arr.shape}")
    return arr


def tx_to_complex64(buf: object, cpu: CpuFormat) -> np.ndarray:
    """Host TX buffer -> contiguous complex64 samples (see module docstring)."""
    arr = np.asarray(buf)
    if cpu == "fc32":
        arr = _one_channel(arr, 1, "send")
        if not np.iscomplexobj(arr):
            raise TypeError(f"fc32 TX buffers must be complex, got dtype {arr.dtype}")
        return np.ascontiguousarray(arr, dtype=np.complex64)
    if arr.dtype != np.int16:
        raise TypeError(f"sc16 TX buffers must be int16 (I, Q) pairs, got dtype {arr.dtype}")
    arr = _one_channel(arr, 2, "send")
    if arr.shape[-1] != 2:
        raise TypeError(
            f"sc16 TX buffers must be int16 of shape (n, 2), got {arr.dtype} {arr.shape}"
        )
    out = (arr[:, 0].astype(np.float32) + 1j * arr[:, 1].astype(np.float32)) / np.float32(
        SC16_SCALE
    )
    return np.ascontiguousarray(out, dtype=np.complex64)


def _rx_view(out: np.ndarray, cpu: CpuFormat) -> np.ndarray:
    """Writable single-channel view of a host RX buffer."""
    if not isinstance(out, np.ndarray):
        raise TypeError("recv needs a preallocated numpy buffer")
    if cpu == "fc32":
        v = _one_channel(out, 1, "recv")
        if v.dtype != np.complex64:
            raise TypeError(f"fc32 RX buffers must be complex64, got {v.dtype}")
        return v
    v = _one_channel(out, 2, "recv")
    if v.dtype != np.int16 or v.shape[-1] != 2:
        raise TypeError(f"sc16 RX buffers must be int16 of shape (n, 2), got {v.dtype} {v.shape}")
    return v


def complex64_to_sc16(x: np.ndarray) -> np.ndarray:
    """(n,) complex64 -> (n, 2) int16 with ``round(32767*x)`` saturated to int16."""
    s = np.empty((x.size, 2), dtype=np.float64)
    s[:, 0] = np.rint(x.real.astype(np.float64) * SC16_SCALE)
    s[:, 1] = np.rint(x.imag.astype(np.float64) * SC16_SCALE)
    return np.clip(s, -32768, 32767).astype(np.int16)


def _copy_md(dst: RxMetadata, src: RxMetadata) -> None:
    for name in (
        "has_time_spec",
        "time_spec",
        "more_fragments",
        "fragment_offset",
        "start_of_burst",
        "end_of_burst",
        "error_code",
        "out_of_sequence",
        "first_sample",
    ):
        setattr(dst, name, getattr(src, name))


# =========================================================================== TX


class TxStreamer:
    """A TX stream of a :class:`~sremu.api.device.Device` (UHD ``tx_streamer``)."""

    def __init__(self, device: Device, handle: StreamHandle) -> None:
        self._dev = device
        self.handle = handle
        self._in_burst = False
        self._head: int | None = None  # first index after the last accepted sample

    @property
    def stream(self) -> str:
        """Stream id ``<node>.<stream>``."""
        return self.handle.spec.stream

    @property
    def rate_sps(self) -> Fraction:
        """Exact sample rate, S/s."""
        return self.handle.spec.rate_sps

    def get_num_channels(self) -> int:
        """Always 1 in release 1 (UHD ``get_num_channels``)."""
        return self.handle.spec.nchan

    def send(self, buf: object, md: TxMetadata | None = None, timeout: float = 0.1) -> int:
        """Send samples; returns the number accepted (0 when a late timed burst is dropped,
        reported by async ``TIME_ERROR``). ``timeout`` is accepted for UHD symmetry; the
        in-process engines never block a send."""
        md = TxMetadata() if md is None else md
        x = tx_to_complex64(buf, self.handle.spec.cpu)
        if md.idle_until and x.size:
            raise StreamError("idle_until carries no samples")
        return self._dev._send(self, x, md)

    # state used by Device-level recording ---------------------------------------------
    def _start_index(self, md: TxMetadata) -> int | None:
        """Absolute start index of a send when the API can know it (timed, or continuing
        a burst); None for an untimed burst start (the engine places it)."""
        if md.has_time_spec:
            return to_ticks(md.time_spec, self.rate_sps)
        if self._in_burst and self._head is not None:
            return self._head
        return None

    def _after_send(self, k0: int | None, n_in: int, n_ok: int, md: TxMetadata) -> None:
        if md.idle_until:
            return  # a commitment promise: no samples, no burst state (section 4.8)
        if n_in and not n_ok:
            return  # dropped (late): the engine's state did not change
        if k0 is not None:
            self._head = k0 + n_ok
        self._in_burst = not (md.end_of_burst or md.end_of_stream)

    # sremu extensions ----------------------------------------------------------------
    def idle_until(self, time_spec: TimeSpec) -> None:
        """Commit this stream (silence) through ``to_ticks(time_spec)`` (lockstep, 4.8)."""
        self._dev._send(
            self,
            np.zeros(0, np.complex64),
            TxMetadata(has_time_spec=True, time_spec=time_spec, idle_until=True),
        )

    def end_of_stream(self) -> None:
        """Declare that this stream sends nothing more (silence from now on)."""
        self._dev._send(self, np.zeros(0, np.complex64), TxMetadata(end_of_stream=True))

    def recv_async_msg(self, timeout: float = 0.0) -> AsyncMetadata | None:
        """Next asynchronous TX event (UHD ``recv_async_msg``), or None after ``timeout``."""
        return self._dev._retry(
            lambda: self._dev._engine.recv_async(self.handle, 0.0),
            lambda m: m is not None,
            timeout,
        )


# =========================================================================== RX


class RxStreamer:
    """An RX stream of a :class:`~sremu.api.device.Device` (UHD ``rx_streamer``)."""

    def __init__(self, device: Device, handle: StreamHandle) -> None:
        self._dev = device
        self.handle = handle
        self._held: deque[tuple[np.ndarray, RxMetadata]] = deque()

    @property
    def stream(self) -> str:
        """Stream id ``<node>.<stream>``."""
        return self.handle.spec.stream

    @property
    def rate_sps(self) -> Fraction:
        """Exact sample rate, S/s."""
        return self.handle.spec.rate_sps

    def get_num_channels(self) -> int:
        """Always 1 in release 1."""
        return self.handle.spec.nchan

    def issue_stream_cmd(self, cmd: StreamCmd) -> None:
        """Start or stop streaming (``start_cont``, ``stop_cont``, ``num_done``,
        ``num_more``); a timed command in the past makes the next ``recv`` return
        ``LATE_COMMAND``."""
        self._dev._stream_cmd(self, cmd)

    def _read(self, n: int) -> tuple[np.ndarray, RxMetadata]:
        """One chunk: a held one first, else one engine read (records what it reads)."""
        if self._held:
            return self._held.popleft()
        y, md = self._dev._engine.read_rx(self.handle, n, 0.0)
        y = np.asarray(y, dtype=np.complex64).reshape(-1)
        if y.size:
            if md.first_sample is None:
                md.first_sample = to_ticks(md.time_spec, self.rate_sps)
            self._dev._record_rx(self, md.first_sample, y)
        return y, md

    def recv(
        self,
        out: np.ndarray,
        md: RxMetadata | None = None,
        timeout: float = 1.0,
        one_packet: bool = False,
    ) -> int:
        """Fill ``out`` with up to ``len(out)`` samples; returns how many. ``md`` (filled in
        place) describes the first returned sample (see the module docstring)."""
        view = _rx_view(out, self.handle.spec.cpu)
        cap = int(view.shape[0])
        meta = RxMetadata() if md is None else md
        _copy_md(meta, RxMetadata())
        if cap == 0:
            return 0
        with self._dev._cond:
            y, m = self._dev._retry(
                lambda: self._read(cap),
                lambda r: r[0].size > 0 or r[1].error_code not in _RETRYABLE,
                timeout,
            )
            if y.size > cap:  # a held chunk can be longer than this buffer
                self._held.appendleft(self._split(y, m, cap))
                y = y[:cap]
            _copy_md(meta, m)
            self._put(view, 0, y)
            got = int(y.size)
            if m.error_code != ErrorCode.NONE or not got:
                self._dev._sync_records()
                return got
            expected = (m.first_sample or 0) + got
            while got < cap and not meta.end_of_burst and not one_packet:
                y, m = self._read(cap - got)
                if not y.size and m.error_code in _RETRYABLE:
                    break  # nothing more available now: return the partial buffer
                if m.error_code != ErrorCode.NONE or not y.size or m.first_sample != expected:
                    self._held.appendleft((y, m))  # error or gap: the next call returns it
                    break
                if y.size > cap - got:
                    self._held.appendleft(self._split(y, m, cap - got))
                    y = y[: cap - got]
                self._put(view, got, y)
                got += int(y.size)
                expected += int(y.size)
                meta.end_of_burst = m.end_of_burst
            self._dev._sync_records()
            return got

    def _split(self, y: np.ndarray, md: RxMetadata, k: int) -> tuple[np.ndarray, RxMetadata]:
        """The part of chunk ``(y, md)`` after its first ``k`` samples."""
        rest = RxMetadata()
        _copy_md(rest, md)
        f = self.rate_sps
        rest.first_sample = (md.first_sample or 0) + k
        rest.time_spec = TimeSpec.from_ticks(rest.first_sample, f)
        rest.start_of_burst = False
        md.end_of_burst = False
        return y[k:], rest

    def _put(self, view: np.ndarray, at: int, y: np.ndarray) -> None:
        if not y.size:
            return
        if self.handle.spec.cpu == "fc32":
            view[at : at + y.size] = y
        else:
            view[at : at + y.size] = complex64_to_sc16(y)
