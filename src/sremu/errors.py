"""Exceptions, warnings and UHD-style status codes (DESIGN.md sections 4, 5.5, 8.1, 8.5).

Two kinds of failure exist in sremu:

* **Status codes** (``ErrorCode``, ``AsyncCode``) travel in metadata and control replies,
  exactly like ``uhd::rx_metadata_t::error_code_t`` and ``uhd::async_metadata_t::event_code_t``.
  The engine never raises on bad external input; it returns a code and counts it (I6).
  The **names** are the wire contract (JSON replies carry ``"code": "LATE_COMMAND"``). The
  numeric values of the UHD-derived members mirror ``uhd/types/metadata.hpp`` for
  familiarity only and are not part of any protocol.
* **Exceptions** signal programming or configuration errors (invalid scenarios, invalid
  tables, physics guards). Every sremu exception derives from ``SremuError``; the ones
  that correspond to a status code carry it in ``.code``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import ClassVar

__all__ = [
    "AsyncCode",
    "BandwidthExceeded",
    "ConfigurationError",
    "ContractVersionError",
    "ErrorCode",
    "FDBandwidthExceeded",
    "GeometryError",
    "GridTooCoarse",
    "LateCommandError",
    "LookaheadViolation",
    "MissingProvenanceError",
    "NonPhysicalError",
    "NondeterminismWarning",
    "NotPermitted",
    "OutOfDomain",
    "PassivityError",
    "RateMismatchError",
    "ReciprocityError",
    "RefinementNotConverged",
    "SchemaError",
    "SremuError",
    "SremuTimeout",
    "SremuWarning",
    "StreamError",
    "TableValidationError",
    "TooLarge",
    "UnknownTarget",
    "ValidityWarning",
]


class ErrorCode(IntEnum):
    """Per-call status code (RX metadata, control replies, receipts).

    UHD ``rx_metadata_t`` names first; sremu additions from 0x100 (section 8.1).
    """

    NONE = 0x0
    TIMEOUT = 0x1
    LATE_COMMAND = 0x2
    BROKEN_CHAIN = 0x4
    OVERFLOW = 0x8
    ALIGNMENT = 0xC
    BAD_PACKET = 0xF
    # sremu extensions
    SCHEMA_ERROR = 0x100
    LOOKAHEAD_VIOLATION = 0x101
    UNKNOWN_TARGET = 0x102
    OUT_OF_DOMAIN = 0x103
    TOO_LARGE = 0x104
    NOT_PERMITTED = 0x105
    RATE_MISMATCH = 0x106


class AsyncCode(IntEnum):
    """Asynchronous TX event code (``async_metadata_t::event_code_t`` names, section 8.1)."""

    BURST_ACK = 0x1
    UNDERFLOW = 0x2
    SEQ_ERROR = 0x4
    TIME_ERROR = 0x8
    UNDERFLOW_IN_PACKET = 0x10
    SEQ_ERROR_IN_BURST = 0x20


# --------------------------------------------------------------------------- exceptions


class SremuError(Exception):
    """Base class of every sremu exception. ``code`` maps it to a wire status when one exists."""

    code: ClassVar[ErrorCode | None] = None


class ConfigurationError(SremuError, ValueError):
    """Invalid configuration or argument (for example ``nchan > 1``, a non-positive rate)."""


class SchemaError(ConfigurationError):
    """A scenario, state library or control message failed schema validation."""

    code = ErrorCode.SCHEMA_ERROR


class ContractVersionError(SremuError):
    """A peer announced an incompatible ``contract_version`` during ``hello``."""


class RateMismatchError(ConfigurationError):
    """Two streams overlap in band at different sample rates (section 3.2; release 1)."""

    code = ErrorCode.RATE_MISMATCH


class UnknownTarget(SremuError, KeyError):
    """A command or query names a stream, node, panel or controller that does not exist."""

    code = ErrorCode.UNKNOWN_TARGET

    def __str__(self) -> str:  # KeyError quotes its argument; keep the plain message
        return str(self.args[0]) if self.args else ""


class TooLarge(SremuError, ValueError):
    """A message exceeds a size cap (1 MB, 65,536 elements, 2**22 samples per frame)."""

    code = ErrorCode.TOO_LARGE


class NotPermitted(SremuError):
    """The operation is not allowed in the current mode or by the current policy."""

    code = ErrorCode.NOT_PERMITTED


class StreamError(SremuError):
    """Misuse of a stream handle (closed stream, wrong direction, bad dtype)."""


class SremuTimeout(SremuError, TimeoutError):
    """A blocking call timed out."""

    code = ErrorCode.TIMEOUT


class LateCommandError(SremuError):
    """A timed command arrived after its sample was computed and ``late_policy="reject"``."""

    code = ErrorCode.LATE_COMMAND


class LookaheadViolation(SremuError):
    """A barrier-registered controller acted at ``at < t_b + L_min`` (section 4.9, RT-17)."""

    code = ErrorCode.LOOKAHEAD_VIOLATION


class NonPhysicalError(SremuError):
    """A legacy or non-physical item was called without ``allow_nonphysical=True`` (5.14)."""


class GeometryError(ConfigurationError):
    """Invalid geometry, e.g. a RIS cell closer than one wavelength to a node (section 5.3)."""


class RefinementNotConverged(SremuError):
    """The ``po_patch`` Richardson check did not converge after two halvings (section 5.4)."""


class FDBandwidthExceeded(ConfigurationError):
    """Occupied bandwidth above 0.8 f_s with the default FD error target (section 4.5)."""


# Response-table validators (section 5.5) ------------------------------------------------


class TableValidationError(ConfigurationError):
    """Base class of the response-table load-time validators (section 5.5)."""


class PassivityError(TableValidationError):
    """|Gamma| exceeds 1 (+ tolerance) for a state, frequency and angle pair."""


class ReciprocityError(TableValidationError):
    """Gamma(s; theta_i, theta_s) != Gamma(s; theta_s, theta_i) beyond tolerance."""


class GridTooCoarse(TableValidationError):
    """Adjacent table entries differ by more than 0.35 of max |Gamma|."""


class OutOfDomain(TableValidationError):
    """A query lies outside the table's frequency or angle grid."""

    code = ErrorCode.OUT_OF_DOMAIN


class BandwidthExceeded(TableValidationError):
    """A linked TX stream's occupied band exceeds the table's ``validity_bw_hz``."""


class MissingProvenanceError(TableValidationError):
    """A mandatory provenance field of a response table is missing."""


# --------------------------------------------------------------------------- warnings


class SremuWarning(UserWarning):
    """Base class of sremu warnings."""


class ValidityWarning(SremuWarning):
    """A model is used near the edge of its validity domain (e.g. r < 3 lambda)."""


class NondeterminismWarning(SremuWarning):
    """The run is no longer bitwise reproducible (``nondeterministic_inputs``)."""
