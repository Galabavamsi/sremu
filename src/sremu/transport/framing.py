"""Native data-plane frames: the 48-byte ``SRMU`` header (DESIGN.md section 8.3).

A frame is a two-part ZMQ message ``[header, payload]``; the payload is zero-copy. The
header is little-endian, ``struct "<IHHIIqIHHQQ"`` (48 bytes)::

    uint32 magic        bytes b"SRMU" on the wire (uint32 value 0x554D5253)
    uint16 version      1
    uint16 flags        b0 SOB, b1 EOB, b2 HAS_TIME, b3 IDLE_UNTIL (empty payload;
                        first_sample = idle-until index), b4 OVERFLOW_BEFORE (an RX gap
                        precedes), b5 LATE, b6 END_OF_STREAM
    uint32 stream_id    from stream.open
    uint32 seq          per-stream frame counter; a gap -> SEQ_ERROR async
    int64  first_sample absolute index at the stream rate (TX: valid iff HAS_TIME, else the
                        frontier; RX: always)
    uint32 nsamps       <= 2**22
    uint16 nchan        == 1
    uint16 dtype        1 cf32_le, 2 sc16_le
    uint64 wall_ns      sender monotonic clock; metrics only, never the data plane
    uint64 reserved     0

**Validation** (:func:`parse_frame`). Version 1 is strict: a bad magic, version, dtype,
nchan or size, a payload length mismatch, a non-zero ``reserved`` word, unknown flag bits,
an ``IDLE_UNTIL`` frame with samples or a negative timed ``first_sample`` raise
:class:`BadFrame` with a short ``reason``; the receiver drops the frame and counts it
(``BAD_PACKET``, I6). Nothing here raises on bad input except :class:`BadFrame`.

**Sample formats.** ``cf32_le`` is interleaved little-endian float32 I/Q (numpy
``<c8``). ``sc16_le`` is interleaved little-endian int16 I/Q with full scale
:data:`SC16_SCALE` = 32767 (the API's sc16 host format, section 8.1):
``x = (I + jQ)/32767`` on input and ``round(32767*x)`` saturated to int16 on output.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import MAX_FRAME_SAMPLES
from sremu.errors import ErrorCode

__all__ = [
    "BYTES_PER_SAMPLE",
    "HEADER_SIZE",
    "HEADER_STRUCT",
    "KNOWN_FLAGS",
    "MAGIC",
    "MAGIC_U32",
    "SC16_SCALE",
    "VERSION",
    "BadFrame",
    "Dtype",
    "Flag",
    "Frame",
    "FrameHeader",
    "decode_payload",
    "encode_payload",
    "make_frame",
    "parse_frame",
]

HEADER_STRUCT: Final = struct.Struct("<IHHIIqIHHQQ")
"""Header layout (section 8.3); ``HEADER_STRUCT.size == 48``."""
HEADER_SIZE: Final[int] = HEADER_STRUCT.size
MAGIC: Final[bytes] = b"SRMU"
"""Magic bytes as they appear on the wire."""
MAGIC_U32: Final[int] = int.from_bytes(MAGIC, "little")
"""The magic as the little-endian uint32 of the header (0x554D5253)."""
VERSION: Final[int] = 1
"""Frame format version."""
SC16_SCALE: Final[float] = 32767.0
"""Full scale of ``sc16_le``: int16 32767 <-> 1.0 (same as the API's sc16 host format)."""

_U32_MAX: Final[int] = (1 << 32) - 1
_U64_MAX: Final[int] = (1 << 64) - 1
_I64_MIN: Final[int] = -(1 << 63)
_I64_MAX: Final[int] = (1 << 63) - 1


class Flag(IntFlag):
    """Header flag bits (section 8.3)."""

    SOB = 1 << 0
    EOB = 1 << 1
    HAS_TIME = 1 << 2
    IDLE_UNTIL = 1 << 3
    OVERFLOW_BEFORE = 1 << 4
    LATE = 1 << 5
    END_OF_STREAM = 1 << 6


KNOWN_FLAGS: Final[int] = 0x7F
"""Mask of the flag bits defined in version 1 (any other bit is a bad frame)."""


class Dtype(IntEnum):
    """Payload sample format."""

    CF32_LE = 1
    SC16_LE = 2


BYTES_PER_SAMPLE: Final[dict[int, int]] = {Dtype.CF32_LE: 8, Dtype.SC16_LE: 4}
"""Bytes per complex sample of each dtype."""


class BadFrame(Exception):
    """A frame failed validation; ``reason`` names the check (e.g. ``"magic"``)."""

    code = ErrorCode.BAD_PACKET

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True)
class FrameHeader:
    """A decoded frame header (field meanings in the module docstring)."""

    flags: int
    stream_id: int
    seq: int
    first_sample: int
    nsamps: int
    dtype: int = Dtype.CF32_LE
    nchan: int = 1
    wall_ns: int = 0
    version: int = VERSION
    reserved: int = 0

    def has(self, flag: Flag) -> bool:
        """Whether ``flag`` is set."""
        return bool(self.flags & flag)

    def pack(self) -> bytes:
        """The 48 header bytes; ``ValueError`` when a field does not fit its width."""
        for name, lo, hi in (
            ("flags", 0, 0xFFFF),
            ("stream_id", 0, _U32_MAX),
            ("seq", 0, _U32_MAX),
            ("first_sample", _I64_MIN, _I64_MAX),
            ("nsamps", 0, _U32_MAX),
            ("dtype", 0, 0xFFFF),
            ("nchan", 0, 0xFFFF),
            ("wall_ns", 0, _U64_MAX),
            ("version", 0, 0xFFFF),
            ("reserved", 0, _U64_MAX),
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not lo <= int(v) <= hi:
                raise ValueError(f"frame header field {name}={v!r} outside [{lo}, {hi}]")
        return HEADER_STRUCT.pack(
            MAGIC_U32,
            int(self.version),
            int(self.flags),
            int(self.stream_id),
            int(self.seq),
            int(self.first_sample),
            int(self.nsamps),
            int(self.nchan),
            int(self.dtype),
            int(self.wall_ns),
            int(self.reserved),
        )


@dataclass(frozen=True, eq=False)
class Frame:
    """A validated frame: header and complex64 samples (``(nsamps,)``)."""

    header: FrameHeader
    samples: NDArray[np.complex64]


def encode_payload(x: NDArray[np.complexfloating], dtype: int = Dtype.CF32_LE) -> bytes:
    """Payload bytes of complex samples (``(n,)``) in ``dtype``."""
    z = np.ascontiguousarray(np.asarray(x).reshape(-1), dtype=np.complex64)
    if dtype == Dtype.CF32_LE:
        return z.astype("<c8", copy=False).tobytes()
    if dtype == Dtype.SC16_LE:
        iq = np.empty((z.shape[0], 2), dtype=np.float64)
        iq[:, 0] = z.real
        iq[:, 1] = z.imag
        q = np.clip(np.rint(iq * SC16_SCALE), -32768, 32767).astype("<i2")
        return q.tobytes()
    raise ValueError(f"unknown dtype {dtype}")


def decode_payload(dtype: int, payload: bytes | memoryview, nsamps: int) -> NDArray[np.complex64]:
    """complex64 samples of a payload (lengths must already match)."""
    if dtype == Dtype.CF32_LE:
        return np.frombuffer(payload, dtype="<c8", count=nsamps).astype(np.complex64)
    iq = np.frombuffer(payload, dtype="<i2", count=2 * nsamps).astype(np.float32)
    iq /= np.float32(SC16_SCALE)
    return iq.view(np.complex64).copy()


def make_frame(
    header: FrameHeader, samples: NDArray[np.complexfloating] | None = None
) -> list[bytes]:
    """The two parts ``[header, payload]`` of a frame; ``header.nsamps`` must match."""
    payload = b"" if samples is None else encode_payload(samples, header.dtype)
    n = len(payload) // BYTES_PER_SAMPLE.get(header.dtype, 8) if payload else 0
    if n != header.nsamps:
        raise ValueError(f"header nsamps {header.nsamps} != {n} samples in the payload")
    return [header.pack(), payload]


def _buf(part: object) -> memoryview:
    if isinstance(part, bytes | bytearray | memoryview):
        return memoryview(part)
    buf = getattr(part, "buffer", None)  # zmq.Frame
    if buf is not None:
        return memoryview(buf)
    raise BadFrame("parts", f"unsupported message part {type(part).__name__}")


def parse_header(hdr: bytes | memoryview) -> FrameHeader:
    """Decode and validate a header (everything except the payload length)."""
    if len(hdr) != HEADER_SIZE:
        raise BadFrame("header_size", f"{len(hdr)} bytes, expected {HEADER_SIZE}")
    (magic, version, flags, sid, seq, first, nsamps, nchan, dtype, wall, reserved) = (
        HEADER_STRUCT.unpack(hdr)
    )
    if magic != MAGIC_U32:
        raise BadFrame("magic", f"0x{magic:08x}")
    if version != VERSION:
        raise BadFrame("version", str(version))
    if flags & ~KNOWN_FLAGS:
        raise BadFrame("flags", f"unknown bits 0x{flags & ~KNOWN_FLAGS:04x}")
    if nchan != 1:
        raise BadFrame("nchan", str(nchan))
    if dtype not in BYTES_PER_SAMPLE:
        raise BadFrame("dtype", str(dtype))
    if nsamps > MAX_FRAME_SAMPLES:
        raise BadFrame("nsamps", f"{nsamps} > {MAX_FRAME_SAMPLES}")
    if reserved != 0:
        raise BadFrame("reserved", str(reserved))
    if flags & Flag.IDLE_UNTIL and nsamps != 0:
        raise BadFrame("flags", "IDLE_UNTIL frame with samples")
    if flags & (Flag.HAS_TIME | Flag.IDLE_UNTIL) and first < 0:
        raise BadFrame("first_sample", str(first))
    return FrameHeader(flags, sid, seq, first, nsamps, dtype, nchan, wall, version, reserved)


def parse_frame(parts: Sequence[object]) -> Frame:
    """Validate a received multipart message and decode it (:class:`BadFrame` otherwise)."""
    if len(parts) != 2:
        raise BadFrame("parts", f"{len(parts)} message parts, expected 2")
    hdr = parse_header(_buf(parts[0]))
    payload = _buf(parts[1])
    want = hdr.nsamps * BYTES_PER_SAMPLE[hdr.dtype]
    if len(payload) != want:
        raise BadFrame("payload_size", f"{len(payload)} bytes, expected {want}")
    return Frame(hdr, decode_payload(hdr.dtype, payload, hdr.nsamps))
