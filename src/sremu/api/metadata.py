"""UHD-shaped metadata, codes and transition helpers of the Python API (DESIGN.md 8.1).

sremu's API is **UHD-shaped, not UHD-pretending**: the metadata types and status codes
mirror ``uhd::tx_metadata_t``, ``uhd::rx_metadata_t``, ``uhd::async_metadata_t`` and
``uhd::stream_cmd_t`` (names and semantics), but there is no UHD binary compatibility.
The types themselves are the frozen contract types (``sremu.contracts``, ``sremu.timebase``,
``sremu.errors``) re-exported here.

* **RX error codes** (``RxMetadata.error_code``, also in command receipts): NONE, TIMEOUT,
  LATE_COMMAND, BROKEN_CHAIN, OVERFLOW, ALIGNMENT, BAD_PACKET; sremu adds SCHEMA_ERROR,
  LOOKAHEAD_VIOLATION, UNKNOWN_TARGET, OUT_OF_DOMAIN, TOO_LARGE, NOT_PERMITTED,
  RATE_MISMATCH.
* **Async TX codes** (``AsyncMetadata.event_code``): BURST_ACK, UNDERFLOW, SEQ_ERROR,
  TIME_ERROR, UNDERFLOW_IN_PACKET, SEQ_ERROR_IN_BURST.
* **Stream modes** (``StreamCmd.mode``): start_cont, stop_cont, num_done, num_more.
  Untimed commands start at the frontier.
* ``nchan`` is carried everywhere; values > 1 raise ``ConfigurationError``.

**Paper Table III** is generated from the rows below by :func:`table_iii` (do not edit the
table by hand elsewhere). Each row starts with ``- sremu:`` and has the columns sremu, UHD,
srsRAN ZMQ device args and GNU Radio ZMQ blocks. Entries marked [UNVERIFIED] are pinned by
WP8 (section 8.4) before any claim.

Table III:

- sremu: ``TxStreamer.send(buf, TxMetadata, timeout)``
  UHD: ``tx_streamer::send``
  srsRAN: ``tx_port`` (raw samples; the emulator's REQ pulls them) [UNVERIFIED]
  GNU Radio: ``zeromq.rep_sink``; native ``PUSH`` frames
- sremu: ``RxStreamer.recv(buf, RxMetadata, timeout)``
  UHD: ``rx_streamer::recv``
  srsRAN: ``rx_port`` (the emulator's REP answers RX requests) [UNVERIFIED]
  GNU Radio: ``zeromq.req_source``; native ``PULL`` frames
- sremu: ``TxMetadata`` (SOB, EOB, ``time_spec``; sremu: ``idle_until``, ``end_of_stream``)
  UHD: ``tx_metadata_t``
  srsRAN: none (timing implicit in sample counts)
  GNU Radio: none (stream tags not used)
- sremu: ``RxMetadata`` (``time_spec``, ``error_code``, EOB; sremu: ``first_sample``)
  UHD: ``rx_metadata_t``
  srsRAN: none
  GNU Radio: none
- sremu: ``TxStreamer.recv_async_msg(timeout)`` returning ``AsyncMetadata``
  UHD: ``tx_streamer::recv_async_msg``, ``async_metadata_t``
  srsRAN: none
  GNU Radio: none
- sremu: ``RxStreamer.issue_stream_cmd(StreamCmd)``
  UHD: ``rx_streamer::issue_stream_cmd``, ``stream_cmd_t``
  srsRAN: implicit (each REQ asks for samples)
  GNU Radio: implicit
- sremu: ``TimeSpec(full_secs, frac_secs)`` (exact; floats via ``Fraction(repr(x))``)
  UHD: ``time_spec_t``
  srsRAN: none
  GNU Radio: none
- sremu: ``Device.set_command_time`` / ``Device.clear_command_time``
  UHD: ``multi_usrp::set_command_time`` / ``multi_usrp::clear_command_time``
  srsRAN: none
  GNU Radio: none
- sremu: ``StreamArgs(stream, fc_hz, rate_sps, cpu)``
  UHD: ``stream_args_t`` with ``set_tx_freq``/``set_rx_freq``, ``set_tx_rate``/``set_rx_rate``
  srsRAN: ``base_srate``; ``tx_freq``/``rx_freq`` [UNVERIFIED]
  GNU Radio: block ``address`` and item size
- sremu: ``TxStreamer.idle_until(TimeSpec)`` (sremu extension)
  UHD: none (closest: end of burst)
  srsRAN: none
  GNU Radio: none
- sremu: ``Device.ris(panel).set_config(...)`` (timed RIS command)
  UHD: none
  srsRAN: none
  GNU Radio: none
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from sremu.contracts import AsyncMetadata, RxMetadata, StreamCmd, TransitionSpec, TxMetadata
from sremu.errors import AsyncCode, ErrorCode
from sremu.timebase import TimeSpec, to_exact

__all__ = [
    "ApiMapping",
    "AsyncCode",
    "AsyncMetadata",
    "ErrorCode",
    "FirstOrder",
    "Linear",
    "RxMetadata",
    "Step",
    "StreamCmd",
    "TimeSpec",
    "TxMetadata",
    "table_iii",
    "table_iii_markdown",
]

TABLE_III_COLUMNS: Final[tuple[str, str, str, str]] = (
    "sremu",
    "UHD",
    "srsRAN ZMQ device args",
    "GNU Radio ZMQ blocks",
)


@dataclass(frozen=True)
class ApiMapping:
    """One row of paper Table III."""

    sremu: str
    uhd: str
    srsran_zmq: str
    gnuradio_zmq: str


_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("sremu", "- sremu:"),
    ("uhd", "UHD:"),
    ("srsran_zmq", "srsRAN:"),
    ("gnuradio_zmq", "GNU Radio:"),
)


def table_iii() -> tuple[ApiMapping, ...]:
    """Rows of paper Table III, parsed from the ``Table III:`` block of this module's
    docstring (four prefixed lines per row, in column order)."""
    doc = __doc__ or ""
    _, sep, rest = doc.partition("Table III:\n")
    if not sep:
        raise RuntimeError("api.metadata docstring lost its 'Table III:' block")
    lines = [ln.strip() for ln in rest.splitlines() if ln.strip()]
    if len(lines) % len(_FIELDS):
        raise RuntimeError("Table III block must have four lines per row")
    rows = []
    for i in range(0, len(lines), len(_FIELDS)):
        cells = []
        for (_, prefix), line in zip(_FIELDS, lines[i : i + len(_FIELDS)], strict=True):
            if not line.startswith(prefix):
                raise RuntimeError(f"Table III line {line!r} should start with {prefix!r}")
            cells.append(line[len(prefix) :].strip())
        rows.append(ApiMapping(*cells))
    return tuple(rows)


def table_iii_markdown() -> str:
    """Table III as a Markdown table (header plus one line per row)."""
    lines = [
        "| " + " | ".join(TABLE_III_COLUMNS) + " |",
        "|" + "---|" * len(TABLE_III_COLUMNS),
    ]
    for r in table_iii():
        lines.append(f"| {r.sremu} | {r.uhd} | {r.srsran_zmq} | {r.gnuradio_zmq} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- transitions


def Step() -> TransitionSpec:
    """Instant switch (the default transition)."""
    return TransitionSpec("step")


def Linear(duration_s: str | int | Fraction | float | None = None) -> TransitionSpec:
    """Complex-linear ramp over ``duration_s`` (s; exact, or a float converted with
    ``Fraction(repr(x))`` as every API float, R5; None = the panel's
    ``reconfiguration_time_s``), e.g. ``Linear("2e-5")`` (section 4.7)."""
    return TransitionSpec(
        "linear", None if duration_s is None else to_exact(duration_s, name="duration_s")
    )


def FirstOrder(tau_s: str | int | Fraction | float | None = None) -> TransitionSpec:
    """First-order profile ``1 - exp(-t/tau_s)`` (s; exact or an R5 float; None = panel
    default)."""
    return TransitionSpec("first_order", None if tau_s is None else to_exact(tau_s, name="tau_s"))
