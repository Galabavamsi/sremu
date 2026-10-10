"""Canonical event log ``events.jsonl`` (DESIGN.md sections 3.6, 4.6, 8.6; contradiction 17).

``events.jsonl`` is the canonical, append-only command log of a run. Every command produces
a :class:`sremu.contracts.CommandRecord`; a record's ``status`` (and, for late commands, its
applied time) changes while the run proceeds, so the log is **event-sourced**:

* each line is one JSON object: the fields of ``CommandRecord.to_dict()`` (exact times as
  ``"p/q"`` strings) plus ``"request"``, the command as issued (:func:`request_to_json`, the
  control-envelope form of section 8.5), or ``null`` when the producer did not have it;
* a line is appended whenever a record first appears or changes, and every line is flushed
  when it is written (optionally fsync'ed), so a crashed run keeps every completed line;
* the final state of a command is its **last** line with that ``seq``
  (:func:`read_events`); ``seq`` is the engine-assigned monotone sequence number, unique
  per command (rejected commands get one too).

Request form (``"request"``), following the control envelope of section 8.5:

* ``cmd_id``, ``op``, ``src``, ``target`` (panel id for ``ris.*``, node id for ``node.*``),
  ``at`` (``{"t": "p/q"}``, ``{"sample": {"stream": s, "index": k}}`` or ``"asap"``),
  ``late_policy``, ``t_wall_recv_ns``;
* ``config`` for ``ris.set_config``: ``{"states_b64": ..., "dtype": "i2"}`` (little-endian
  int16 bytes, base64), ``{"phases_rad": [...], "quantizer": q}`` (JSON floats round-trip
  float64 exactly) or ``{"codebook_index": k}``;
* ``elements``: ``"all"``, ``{"indices": [...]}`` or ``{"group": name}``;
* ``transition``: ``{"model", "duration_s" ("p/q" or null), "classes"}`` or null;
* ``enabled`` (``ris.set_enabled``), ``position_m`` (``node.set_position``, 3 floats, m) and
  ``payload`` (trajectory spec or marker fields; ``Fraction`` values become ``"p/q"``).

Applied configuration (``"applied_config"``, optional): when the producer knows the panel
configuration in effect right after the switch (the WP6 engine reports it with each final
record), the line carries ``{"states_b64": ..., "dtype": "i2", "enabled": bool}`` (per
physical cell, D17 order). It is the replay fallback for commands whose request the
recorder never saw (in-process or remote controllers).

The request lets :mod:`sremu.record.replay` re-inject the command exactly (section 4.12).
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import numpy as np

from sremu.contracts import CommandRecord, CommandRequest, CommandTime, TransitionSpec
from sremu.timebase import fraction_to_str, parse_exact

__all__ = [
    "EVENTLOG_FORMAT",
    "EVENTS_NAME",
    "AppliedConfig",
    "EventLog",
    "LoggedCommand",
    "iter_event_lines",
    "line_of",
    "read_events",
    "request_from_json",
    "request_to_json",
]

EVENTS_NAME: Final[str] = "events.jsonl"
"""File name of the event log inside a run directory (section 3.6)."""

EVENTLOG_FORMAT: Final[int] = 1
"""Version of the line format described in the module docstring."""

_STATES_DTYPE: Final[str] = "i2"  # little-endian int16 (section 8.5 "dtype":"i2")


# =========================================================================== JSON helpers


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return fraction_to_str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable in events.jsonl")


def _dumps(obj: Any) -> str:
    return json.dumps(
        obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=_json_default
    )


# =========================================================================== requests


def request_to_json(req: CommandRequest) -> dict[str, Any]:
    """The control-envelope form of a command request (module docstring)."""
    d: dict[str, Any] = {
        "cmd_id": req.cmd_id,
        "op": req.op,
        "src": req.src,
        "target": req.target,
        "at": req.at.to_json(),
        "late_policy": req.late_policy,
        "t_wall_recv_ns": req.t_wall_recv_ns,
    }
    if req.op == "ris.set_config":
        if req.states is not None:
            raw = np.ascontiguousarray(req.states, dtype="<i2").tobytes()
            d["config"] = {"states_b64": base64.b64encode(raw).decode("ascii"), "dtype": "i2"}
        elif req.phases_rad is not None:
            d["config"] = {
                "phases_rad": [float(x) for x in req.phases_rad],
                "quantizer": req.quantizer,
            }
        else:
            d["config"] = {"codebook_index": int(req.codebook_index or 0)}
        if req.elements == "all":
            d["elements"] = "all"
        elif isinstance(req.elements, str):
            d["elements"] = {"group": req.elements}
        else:
            d["elements"] = {"indices": [int(i) for i in req.elements]}
        tr = req.transition
        d["transition"] = (
            None
            if tr is None
            else {
                "model": tr.model,
                "duration_s": None if tr.duration_s is None else fraction_to_str(tr.duration_s),
                "classes": tr.classes,
            }
        )
    if req.enabled is not None:
        d["enabled"] = bool(req.enabled)
    if req.position_m is not None:
        d["position_m"] = [float(x) for x in req.position_m]
    if req.payload:
        d["payload"] = json.loads(_dumps(dict(req.payload)))
    return d


def request_from_json(d: Mapping[str, Any]) -> CommandRequest:
    """Inverse of :func:`request_to_json` (bitwise for states and phases)."""
    kw: dict[str, Any] = {}
    cfg = d.get("config")
    if cfg is not None:
        if "states_b64" in cfg:
            if cfg.get("dtype", _STATES_DTYPE) != _STATES_DTYPE:
                raise ValueError(f"unsupported states dtype {cfg.get('dtype')!r}")
            raw = base64.b64decode(cfg["states_b64"].encode("ascii"), validate=True)
            kw["states"] = np.frombuffer(raw, dtype="<i2").astype(np.int16)
        elif "phases_rad" in cfg:
            kw["phases_rad"] = np.asarray(cfg["phases_rad"], dtype=np.float64)
            kw["quantizer"] = str(cfg.get("quantizer", "max_real"))
        elif "codebook_index" in cfg:
            kw["codebook_index"] = int(cfg["codebook_index"])
        else:
            raise ValueError(f"unknown config form {sorted(cfg)}")
    el = d.get("elements", "all")
    if el == "all":
        kw["elements"] = "all"
    elif isinstance(el, Mapping) and set(el) == {"indices"}:
        kw["elements"] = tuple(int(i) for i in el["indices"])
    elif isinstance(el, Mapping) and set(el) == {"group"}:
        kw["elements"] = str(el["group"])
    else:
        raise ValueError(f"bad elements {el!r}")
    tr = d.get("transition")
    if tr is not None:
        dur = tr.get("duration_s")
        kw["transition"] = TransitionSpec(
            model=tr["model"],
            duration_s=None if dur is None else parse_exact(dur, name="duration_s"),
            classes=tr.get("classes", "single"),
        )
    if "enabled" in d:
        kw["enabled"] = bool(d["enabled"])
    if "position_m" in d:
        kw["position_m"] = np.asarray(d["position_m"], dtype=np.float64)
    if "payload" in d:
        kw["payload"] = dict(d["payload"])
    return CommandRequest(
        cmd_id=str(d["cmd_id"]),
        op=d["op"],
        src=str(d["src"]),
        target=str(d["target"]),
        at=CommandTime.from_json(d["at"]),
        late_policy=d.get("late_policy", "apply_on_arrival"),
        t_wall_recv_ns=d.get("t_wall_recv_ns"),
        **kw,
    )


# =========================================================================== applied config


@dataclass(frozen=True, eq=False)
class AppliedConfig:
    """Panel configuration in effect right after a command: ``state`` (N,) int16 per
    physical cell (-1 dead) and the ``enabled`` flag."""

    state: np.ndarray
    enabled: bool = True

    def to_json(self) -> dict[str, Any]:
        """``{"states_b64", "dtype": "i2", "enabled"}``."""
        raw = np.ascontiguousarray(self.state, dtype="<i2").tobytes()
        return {
            "states_b64": base64.b64encode(raw).decode("ascii"),
            "dtype": _STATES_DTYPE,
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> AppliedConfig:
        """Inverse of :meth:`to_json`."""
        if d.get("dtype", _STATES_DTYPE) != _STATES_DTYPE:
            raise ValueError(f"unsupported states dtype {d.get('dtype')!r}")
        raw = base64.b64decode(str(d["states_b64"]).encode("ascii"), validate=True)
        state = np.frombuffer(raw, dtype="<i2").astype(np.int16)
        return cls(state, bool(d.get("enabled", True)))

    @classmethod
    def of(cls, cfg: Any) -> AppliedConfig | None:
        """From any object with ``state`` and ``enabled`` (e.g. ``PanelConfig``), or None."""
        if cfg is None or not hasattr(cfg, "state"):
            return None
        state = np.asarray(cfg.state, dtype=np.int16).copy()
        return cls(state, bool(getattr(cfg, "enabled", True)))


# =========================================================================== lines


def line_of(record: CommandRecord, request: CommandRequest | None = None) -> dict[str, Any]:
    """One ``events.jsonl`` object: ``record.to_dict()`` plus ``"request"``."""
    d = record.to_dict()
    d["request"] = None if request is None else request_to_json(request)
    return d


@dataclass(frozen=True)
class LoggedCommand:
    """Final state of one command in an event log.

    ``record``: the record as of its last line. ``request``: the issued command (the last
    non-null ``request`` seen for this ``seq``), or None. ``n_lines``: how many lines the
    command has (its status history length). ``history``: the statuses in line order.
    ``applied``: the last ``applied_config`` seen for this ``seq``, or None.
    """

    record: CommandRecord
    request: CommandRequest | None
    n_lines: int
    history: tuple[str, ...]
    applied: AppliedConfig | None = None


def iter_event_lines(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects of ``events.jsonl`` (a file or a run directory).

    Blank lines are skipped. A truncated final line (a crash in the middle of a write) is
    ignored; malformed lines elsewhere raise ``ValueError`` with the line number.
    """
    p = Path(path)
    if p.is_dir():
        p = p / EVENTS_NAME
    lines = p.read_text(encoding="utf-8").split("\n")
    last = len(lines) - 1
    for i, text in enumerate(lines):
        if not text.strip():
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            if i == last:  # no trailing newline: an interrupted append
                return
            raise ValueError(f"{p}:{i + 1}: malformed event line: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{p}:{i + 1}: an event line must be a JSON object")
        yield obj


def read_events(path: str | Path) -> list[LoggedCommand]:
    """Final state of every command in ``events.jsonl``, ordered by ``seq``."""
    last: dict[int, dict[str, Any]] = {}
    req: dict[int, dict[str, Any]] = {}
    count: dict[int, int] = {}
    hist: dict[int, list[str]] = {}
    applied: dict[int, dict[str, Any]] = {}
    for obj in iter_event_lines(path):
        seq = int(obj["seq"])
        last[seq] = obj
        count[seq] = count.get(seq, 0) + 1
        hist.setdefault(seq, []).append(str(obj.get("status")))
        if obj.get("request") is not None:
            req[seq] = obj["request"]
        if obj.get("applied_config") is not None:
            applied[seq] = obj["applied_config"]
    out = []
    for seq in sorted(last):
        r = req.get(seq)
        a = applied.get(seq)
        out.append(
            LoggedCommand(
                record=CommandRecord.from_dict(last[seq]),
                request=None if r is None else request_from_json(r),
                n_lines=count[seq],
                history=tuple(hist[seq]),
                applied=None if a is None else AppliedConfig.from_json(a),
            )
        )
    return out


# =========================================================================== writer


class EventLog:
    """Append-only ``events.jsonl`` writer (flushed per line; ``fsync=True`` also syncs).

    :meth:`append` writes a line when a command first appears or when its record changed
    since the last line written for the same ``seq`` (identical states are skipped), so a
    producer may call it for every record at every synchronization point.
    """

    def __init__(self, path: str | Path, *, fsync: bool = False, append: bool = False) -> None:
        p = Path(path)
        if p.is_dir():
            p = p / EVENTS_NAME
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = p
        self._fsync = fsync
        self._f = p.open("a" if append else "w", encoding="utf-8", newline="\n")
        self._last: dict[int, dict[str, Any]] = {}
        self._requests: dict[int, CommandRequest] = {}
        self._applied: dict[int, AppliedConfig] = {}
        self._records: dict[int, CommandRecord] = {}
        self.n_lines = 0

    def append(
        self,
        record: CommandRecord,
        request: CommandRequest | None = None,
        applied: AppliedConfig | None = None,
    ) -> bool:
        """Log ``record`` (with its request and applied configuration, when given) if it is
        new or changed; True if a line was written. A request or applied configuration is
        kept from the first line that has one and repeated on later lines."""
        if self._f.closed:
            raise ValueError("EventLog is closed")
        seq = int(record.seq)
        rec = record.to_dict()
        new_request = request is not None and seq not in self._requests
        new_applied = applied is not None and seq not in self._applied
        if self._last.get(seq) == rec and not new_request and not new_applied:
            return False
        if new_request:
            assert request is not None
            self._requests[seq] = request
        if new_applied:
            assert applied is not None
            self._applied[seq] = applied
        line = dict(rec)
        line["request"] = request_to_json(self._requests[seq]) if seq in self._requests else None
        if seq in self._applied:
            line["applied_config"] = self._applied[seq].to_json()
        self._f.write(_dumps(line) + "\n")
        self._f.flush()
        if self._fsync:
            os.fsync(self._f.fileno())
        self._last[seq] = rec
        self._records[seq] = CommandRecord.from_dict(rec)
        self.n_lines += 1
        return True

    def final(self) -> list[LoggedCommand]:
        """Final state of every command logged so far (in memory), ordered by ``seq``."""
        return [
            LoggedCommand(self._records[s], self._requests.get(s), 0, (), self._applied.get(s))
            for s in sorted(self._records)
        ]

    def close(self) -> None:
        """Close the file (idempotent)."""
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
