"""Compatibility wrappers with the ANTS / v1 names (DESIGN.md section 8.2).

``Send_to_sim(node, samples, fc, fs, gain=0, start_time=None)`` and
``Receive_from_sim(node, nsamps, fc, fs, gain=0, start_time=None)`` are thin aliases of
``TxStreamer.send`` and ``RxStreamer.recv`` on the streams ``<node>.tx0`` / ``<node>.rx0``
(override with ``stream=``) of the current :class:`~sremu.api.device.Device`:

* the device is the one installed with :func:`use_device` / :func:`set_device`, given with
  ``device=``, or (WP8) a session opened with ``Device.connect($SREMU_CONNECT)``;
* ``fc`` (Hz) and ``fs`` (S/s; floats via ``Fraction(repr(x))``, R5) open the stream on
  first use; a later call with a different ``fc`` or ``fs`` raises ``ConfigurationError``
  (no retuning in release 1);
* ``start_time`` (s; float, exact string, ``Fraction`` or ``TimeSpec``) makes the call
  timed (R3 mapping); ``None`` starts at the frontier (untimed);
* ``gain`` (dB) is an extra digital gain applied at the API boundary, ``10**(gain/20)`` on
  the samples (in addition to the scenario stream's ``gain_db``, section 3.3); ``gain=0``
  leaves the samples bit-for-bit unchanged.

``Send_to_sim`` sends the samples as one burst (start and end of burst, so the stream is
silent afterwards and lockstep receivers are not blocked) and returns the number of samples
accepted. ``Receive_from_sim`` streams exactly ``nsamps`` samples (``num_done``) and returns
them as complex64; it raises ``SremuTimeout`` if they do not all arrive (a lockstep
receiver waits for every linked transmitter: send first, or use ``idle_until``),
``LateCommandError`` for a ``start_time`` in the past and ``StreamError`` for other codes.

``send_to_simulator(data, fc, sampling_rate, node_id)`` and
``recieve_from_simulator(num_samps, fc, sampling_rate, node_id)`` keep the v1 spelling and
argument order and emit a ``DeprecationWarning``. There is no ``MultiUSRP`` mirror (cut).
"""

from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from typing import Final

import numpy as np

from sremu.api.device import Device
from sremu.api.streams import RxStreamer, StreamArgs, TxStreamer
from sremu.contracts import RxMetadata, StreamCmd, TxMetadata
from sremu.errors import (
    ConfigurationError,
    ErrorCode,
    LateCommandError,
    SremuTimeout,
    StreamError,
)
from sremu.timebase import TimeSpec

__all__ = [
    "CONNECT_ENV",
    "Receive_from_sim",
    "Send_to_sim",
    "clear_device",
    "get_device",
    "recieve_from_simulator",
    "send_to_simulator",
    "set_device",
    "use_device",
]

CONNECT_ENV: Final[str] = "SREMU_CONNECT"
"""Environment variable naming a served emulator endpoint for scripts run on their own."""

DEFAULT_TIMEOUT_S: Final[float] = 1.0
"""Wall seconds ``Receive_from_sim`` waits without progress before ``SremuTimeout``."""

_lock = threading.RLock()
_current: list[Device] = []


def set_device(dev: Device) -> None:
    """Install ``dev`` as the current device (replacing any previous one)."""
    with _lock:
        _current.clear()
        _current.append(dev)


def clear_device() -> None:
    """Remove the current device."""
    with _lock:
        _current.clear()


@contextmanager
def use_device(dev: Device) -> Iterator[Device]:
    """Make ``dev`` the current device inside a ``with`` block (restores the previous one)."""
    with _lock:
        prev = list(_current)
        _current[:] = [dev]
    try:
        yield dev
    finally:
        with _lock:
            _current[:] = prev


def get_device() -> Device:
    """The current device; with none installed, ``Device.connect($SREMU_CONNECT)`` (WP8)."""
    with _lock:
        if _current:
            return _current[0]
        endpoint = os.environ.get(CONNECT_ENV)
        if endpoint:
            dev = Device.connect(endpoint)
            _current.append(dev)
            return dev
    raise ConfigurationError(
        "no sremu device: wrap the calls in sremu.compat.use_device(dev) or set "
        f"${CONNECT_ENV} to a served emulator endpoint"
    )


def _gain(x: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0:
        return x
    return (x * np.float32(10.0 ** (float(gain_db) / 20.0))).astype(np.complex64)


def _time(start_time: object) -> TimeSpec:
    if isinstance(start_time, TimeSpec):
        return start_time
    return TimeSpec.from_secs(start_time)  # type: ignore[arg-type]


def _open(
    dev: Device, node: str, stream: str, direction: str, fc: float, fs: object
) -> TxStreamer | RxStreamer:
    args = StreamArgs(f"{node}.{stream}", fc, fs, "fc32")  # type: ignore[arg-type]
    return dev.get_tx_stream(args) if direction == "tx" else dev.get_rx_stream(args)


def Send_to_sim(
    node: str,
    samples: object,
    fc: float,
    fs: float | str | int | Fraction,
    gain: float = 0,
    start_time: float | str | Fraction | TimeSpec | None = None,
    *,
    stream: str = "tx0",
    device: Device | None = None,
) -> int:
    """Send ``samples`` from ``<node>.<stream>`` as one burst; returns samples accepted."""
    dev = device or get_device()
    tx = _open(dev, node, stream, "tx", fc, fs)
    assert isinstance(tx, TxStreamer)
    x = np.asarray(samples)
    if x.ndim == 2 and x.shape[0] == 1:  # v1 passed shape (1, n)
        x = x[0]
    if x.ndim != 1:
        raise ConfigurationError(f"samples must be (n,) or (1, n) for nchan = 1, got {x.shape}")
    x = _gain(np.ascontiguousarray(x, dtype=np.complex64), gain)
    md = TxMetadata(start_of_burst=True, end_of_burst=True)
    if start_time is not None:
        md.has_time_spec = True
        md.time_spec = _time(start_time)
    return tx.send(x, md)


def Receive_from_sim(
    node: str,
    nsamps: int,
    fc: float,
    fs: float | str | int | Fraction,
    gain: float = 0,
    start_time: float | str | Fraction | TimeSpec | None = None,
    *,
    stream: str = "rx0",
    device: Device | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> np.ndarray:
    """Receive exactly ``nsamps`` complex64 samples on ``<node>.<stream>``."""
    n = int(nsamps)
    if n < 0:
        raise ConfigurationError("nsamps must be >= 0")
    dev = device or get_device()
    rx = _open(dev, node, stream, "rx", fc, fs)
    assert isinstance(rx, RxStreamer)
    out = np.zeros(n, dtype=np.complex64)
    if n == 0:
        return out
    if start_time is None:
        rx.issue_stream_cmd(StreamCmd("num_done", num_samps=n, stream_now=True))
    else:
        rx.issue_stream_cmd(
            StreamCmd("num_done", num_samps=n, stream_now=False, time_spec=_time(start_time))
        )
    got = 0
    md = RxMetadata()
    while got < n:
        k = rx.recv(out[got:], md, timeout=timeout)
        if md.error_code == ErrorCode.TIMEOUT or (k == 0 and md.error_code == ErrorCode.NONE):
            raise SremuTimeout(
                f"{rx.stream}: received {got} of {n} samples; the linked transmitters are not "
                "committed (lockstep: send first, end the burst or use idle_until)"
            )
        if md.error_code == ErrorCode.LATE_COMMAND:
            raise LateCommandError(f"{rx.stream}: start_time is in the past")
        if md.error_code != ErrorCode.NONE:
            raise StreamError(f"{rx.stream}: {md.error_code.name} after {got} samples")
        got += k
    return _gain(out, gain)


def send_to_simulator(
    data: object, fc: float, sampling_rate: float | str | int | Fraction, node_id: str
) -> int:
    """v1 spelling of :func:`Send_to_sim` (v1 argument order); deprecated."""
    warnings.warn(
        "send_to_simulator is deprecated; use sremu.compat.Send_to_sim(node, samples, fc, fs)",
        DeprecationWarning,
        stacklevel=2,
    )
    return Send_to_sim(node_id, data, fc, sampling_rate)


def recieve_from_simulator(
    num_samps: int, fc: float, sampling_rate: float | str | int | Fraction, node_id: str
) -> np.ndarray:
    """v1 spelling (sic) of :func:`Receive_from_sim` (v1 argument order); deprecated."""
    warnings.warn(
        "recieve_from_simulator is deprecated; use "
        "sremu.compat.Receive_from_sim(node, nsamps, fc, fs)",
        DeprecationWarning,
        stacklevel=2,
    )
    return Receive_from_sim(node_id, num_samps, fc, sampling_rate)
