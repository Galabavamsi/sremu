"""SigMF recordings, the run recorder and structural validators (DESIGN.md sections 3.6, 8.6).

**Recordings.** One SigMF recording per stream (TX and RX) under ``<run>/iq/``:
``<stream>.sigmf-data`` (``cf32_le``: interleaved little-endian float32 I/Q) and
``<stream>.sigmf-meta``, written as plain JSON (the ``sigmf`` package is not needed):

* ``global``: ``core:datatype "cf32_le"``, ``core:sample_rate`` (float), ``core:version``
  (:data:`SIGMF_VERSION`), ``core:num_channels`` 1, ``core:recorder``,
  ``core:extensions [{"name": "sremu", "version": "1.0.0", "optional": true}]`` and the
  sremu keys ``sremu:stream``, ``sremu:direction``, ``sremu:rate_sps`` (the exact rate,
  ``"p/q"``), ``sremu:cpu_format``, ``sremu:open_index`` (stream open order, which fixes
  the summation order on replay) and ``sremu:contract_version``.
* ``captures``: one segment per contiguous run of absolute samples, with
  ``core:sample_start`` (index into the dataset), ``core:global_index`` (absolute stream
  sample index, so gaps are explicit), ``core:frequency`` (carrier, Hz) and, only when an
  epoch is configured (R2), ``core:datetime`` = epoch + ``global_index``/f_s (ISO-8601, UTC,
  nanoseconds, ``Z``).
* ``annotations`` (RX recordings): one per applied command visible in the recording,
  sorted by ``core:sample_start``. ``core:sample_start`` is the dataset index of the
  stream's ``k_applied`` (equal to ``k_applied`` for a recording that starts at absolute
  sample 0 without gaps; a ``k_applied`` inside a gap maps to the first recorded sample
  after it). ``core:label`` is ``"ris_cfg"`` for ``ris.*`` commands, ``"node_cfg"`` for
  ``node.*`` and ``"marker"`` for ``rec.marker``. sremu keys: ``sremu:cmd_id``,
  ``sremu:seq``, ``sremu:op``, ``sremu:target``, ``sremu:k_applied`` (absolute),
  ``sremu:k_req``, ``sremu:lateness`` (samples), ``sremu:t_applied`` (``"p/q"`` s),
  ``sremu:config_hash``, ``sremu:status`` and ``sremu:counterfactual``.
* ``<run>/run.sigmf-collection`` lists every recording's meta file with its SHA-512.

**Run recorder.** :class:`RunRecorder` is the sink a producer feeds while a run executes
(the WP6 engine through :class:`EngineRecorderSink`, and ``sremu.api.Device``): stream
opens, samples at absolute indices, command records (to ``events.jsonl``,
:mod:`sremu.record.eventlog`) and metrics windows (to ``metrics.jsonl``, one JSON object per
line, flushed per line; exact times as ``"p/q"``, non-finite numbers as null).
:meth:`RunRecorder.close` writes the meta files (annotations come from the final command
records) and the collection.

**Validators** (IO-05): :func:`validate_meta`, :func:`validate_recording` and
:func:`validate_collection` return a list of problems (empty when valid): required keys,
types, the dataset-format grammar, capture and annotation ordering, declared extension
namespaces, ``core:datetime`` form, dataset size, sremu index consistency and collection
hashes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Protocol, TypeGuard, runtime_checkable

import numpy as np

from sremu.contracts import (
    CONTRACT_VERSION,
    CommandRecord,
    CommandRequest,
    StreamId,
    StreamSpec,
)
from sremu.record.eventlog import EVENTS_NAME, AppliedConfig, EventLog, LoggedCommand
from sremu.timebase import fraction_to_str, parse_exact

__all__ = [
    "COLLECTION_NAME",
    "IQ_DIR",
    "METRICS_NAME",
    "SIGMF_VERSION",
    "SREMU_EXTENSION",
    "Capture",
    "EngineRecorderSink",
    "RecordSink",
    "Recording",
    "RecordingIndex",
    "RunRecorder",
    "SigMFWriter",
    "annotation_label",
    "file_index",
    "list_recordings",
    "read_recording",
    "validate_collection",
    "validate_meta",
    "validate_recording",
    "write_collection",
]

SIGMF_VERSION: Final[str] = "1.2.6"
"""SigMF specification version written to ``core:version`` (sigmf.org shows "SigMF
Specification Version v1.2.6", checked 10 Oct 2026; DESIGN.md 8.6 left it as v1.2.x)."""

SREMU_EXTENSION: Final[dict[str, Any]] = {"name": "sremu", "version": "1.0.0", "optional": True}
"""The ``core:extensions`` entry declaring the ``sremu:`` namespace (section 8.6)."""

IQ_DIR: Final[str] = "iq"
COLLECTION_NAME: Final[str] = "run.sigmf-collection"
METRICS_NAME: Final[str] = "metrics.jsonl"
_DATATYPE: Final[str] = "cf32_le"
_META_EXT: Final[str] = ".sigmf-meta"
_DATA_EXT: Final[str] = ".sigmf-data"
_APPLIED: Final[frozenset[str]] = frozenset({"applied", "late_applied"})

_DATATYPE_RE: Final = re.compile(r"^(r|c)(?:(f32|f64|i32|i16|u32|u16)_(le|be)|(i8|u8))$")
_VERSION_RE: Final = re.compile(r"^\d+\.\d+\.\d+$")
_DATETIME_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_BYTES: Final[dict[str, int]] = {
    "f32": 4,
    "f64": 8,
    "i32": 4,
    "i16": 2,
    "u32": 4,
    "u16": 2,
    "i8": 1,
    "u8": 1,
}


def annotation_label(op: str) -> str:
    """``core:label`` of a command annotation: ``ris_cfg``, ``node_cfg`` or ``marker``."""
    if op.startswith("ris."):
        return "ris_cfg"
    if op.startswith("node."):
        return "node_cfg"
    return "marker"


# =========================================================================== captures


@dataclass(frozen=True)
class Capture:
    """A capture segment: dataset index ``sample_start``, absolute ``global_index`` and the
    number of samples ``length`` until the next segment (or the end of the dataset)."""

    sample_start: int
    global_index: int
    length: int

    @property
    def global_end(self) -> int:
        """First absolute index after the segment."""
        return self.global_index + self.length


def file_index(captures: Sequence[Capture], k: int) -> int | None:
    """Dataset index of absolute sample ``k`` (module docstring): inside a segment its
    position; inside a gap the first recorded sample after it; None before the first or
    after the last recorded sample."""
    for i, c in enumerate(captures):
        if k < c.global_index:
            return None if i == 0 else c.sample_start
        if k < c.global_end:
            return c.sample_start + (k - c.global_index)
    return None


def _format_datetime(epoch: _dt.datetime, offset_s: Fraction) -> str:
    """ISO-8601 UTC ``epoch + offset_s`` with nanoseconds and a ``Z`` suffix."""
    base = epoch.astimezone(_dt.UTC)
    unix = _dt.datetime(1970, 1, 1, tzinfo=_dt.UTC)
    base_ns = ((base - unix) // _dt.timedelta(microseconds=1)) * 1000
    off = offset_s * 1_000_000_000
    total = base_ns + (off.numerator * 2 + off.denominator) // (2 * off.denominator)
    sec, ns = divmod(total, 1_000_000_000)
    stamp = (unix + _dt.timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{ns:09d}Z"


def _parse_epoch(epoch: _dt.datetime | str | None) -> _dt.datetime | None:
    if epoch is None:
        return None
    if isinstance(epoch, str):
        text = epoch.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        epoch = _dt.datetime.fromisoformat(text)
    if epoch.tzinfo is None:
        raise ValueError("epoch_datetime must be timezone-aware (UTC)")
    return epoch


# =========================================================================== writer


class SigMFWriter:
    """One SigMF recording (``<base>.sigmf-data`` + ``<base>.sigmf-meta``).

    :meth:`write` appends complex64 samples at absolute stream index ``k0``; a write that
    does not continue the previous one starts a new capture segment (``core:global_index``
    = ``k0``). Writes must not go backwards (``ValueError``). :meth:`close` writes the meta
    file with the given annotations and returns its path.
    """

    def __init__(
        self,
        base: str | Path,
        *,
        stream: StreamId,
        direction: str,
        rate_sps: Fraction | int | str,
        fc_hz: float,
        cpu_format: str = "fc32",
        open_index: int = 0,
        epoch_datetime: _dt.datetime | str | None = None,
        description: str = "",
    ) -> None:
        from sremu import __version__

        self.base = Path(base)
        self.base.parent.mkdir(parents=True, exist_ok=True)
        self.stream = stream
        self.direction = direction
        self.rate = parse_exact(rate_sps, name="rate_sps")
        self.fc_hz = float(fc_hz)
        self.epoch = _parse_epoch(epoch_datetime)
        self._global: dict[str, Any] = {
            "core:datatype": _DATATYPE,
            "core:sample_rate": float(self.rate),
            "core:version": SIGMF_VERSION,
            "core:num_channels": 1,
            "core:recorder": f"sremu {__version__}",
            "core:extensions": [dict(SREMU_EXTENSION)],
            "core:description": description or f"sremu {direction.upper()} stream {stream}",
            "sremu:stream": stream,
            "sremu:direction": direction,
            "sremu:rate_sps": fraction_to_str(self.rate),
            "sremu:cpu_format": cpu_format,
            "sremu:open_index": int(open_index),
            "sremu:contract_version": CONTRACT_VERSION,
        }
        self._data = self.data_path.open("wb")
        self._caps: list[list[int]] = []  # [sample_start, global_index, length]
        self.n_samples = 0
        self.closed = False

    @property
    def data_path(self) -> Path:
        """Path of the ``.sigmf-data`` file."""
        return self.base.with_name(self.base.name + _DATA_EXT)

    @property
    def meta_path(self) -> Path:
        """Path of the ``.sigmf-meta`` file."""
        return self.base.with_name(self.base.name + _META_EXT)

    @property
    def captures(self) -> tuple[Capture, ...]:
        """Capture segments written so far."""
        return tuple(Capture(s, g, n) for s, g, n in self._caps)

    def write(self, k0: int, samples: np.ndarray) -> None:
        """Append ``samples`` (cast to complex64) at absolute stream index ``k0``."""
        if self.closed:
            raise ValueError(f"recording {self.stream!r} is closed")
        x = np.ascontiguousarray(samples, dtype=np.complex64).reshape(-1)
        if x.size == 0:
            return
        k0 = int(k0)
        if k0 < 0:
            raise ValueError(f"{self.stream}: negative sample index {k0}")
        if self._caps:
            _s, g, n = self._caps[-1]
            end = g + n
            if k0 < end:
                raise ValueError(
                    f"{self.stream}: write at {k0} overlaps recorded samples (end {end})"
                )
            if k0 == end:
                self._caps[-1][2] += x.size
            else:
                self._caps.append([self.n_samples, k0, x.size])
        else:
            self._caps.append([0, k0, x.size])
        self._data.write(x.astype("<c8", copy=False).tobytes())
        self.n_samples += x.size

    def meta(self, annotations: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
        """The metadata object (annotations sorted by ``core:sample_start``)."""
        caps: list[dict[str, Any]] = []
        for s, g, _n in self._caps or [[0, 0, 0]]:
            c: dict[str, Any] = {
                "core:sample_start": s,
                "core:global_index": g,
                "core:frequency": self.fc_hz,
            }
            if self.epoch is not None:
                c["core:datetime"] = _format_datetime(self.epoch, Fraction(g) / self.rate)
            caps.append(c)
        ann = sorted(
            (dict(a) for a in annotations),
            key=lambda a: (int(a["core:sample_start"]), int(a.get("sremu:seq", 0))),
        )
        return {"global": dict(self._global), "captures": caps, "annotations": ann}

    def close(self, annotations: Iterable[Mapping[str, Any]] = ()) -> Path:
        """Close the data file and write the meta file; returns the meta path."""
        if not self.closed:
            self._data.close()
            self.closed = True
        text = json.dumps(self.meta(annotations), indent=2, ensure_ascii=False, allow_nan=False)
        self.meta_path.write_text(text + "\n", encoding="utf-8", newline="\n")
        return self.meta_path

    def annotations_for(self, commands: Iterable[LoggedCommand]) -> list[dict[str, Any]]:
        """Annotations of the applied commands visible in this recording."""
        caps = self.captures
        out = []
        for lc in commands:
            r = lc.record
            if r.status not in _APPLIED or self.stream not in r.k_applied:
                continue
            k = int(r.k_applied[self.stream])
            idx = file_index(caps, k)
            if idx is None:
                continue
            out.append(
                {
                    "core:sample_start": idx,
                    "core:label": annotation_label(r.op),
                    "sremu:cmd_id": r.cmd_id,
                    "sremu:seq": int(r.seq),
                    "sremu:op": r.op,
                    "sremu:target": r.panel,
                    "sremu:k_applied": k,
                    "sremu:k_req": int(r.k_req.get(self.stream, k)),
                    "sremu:lateness": int(r.lateness_samples.get(self.stream, 0)),
                    "sremu:t_applied": None
                    if r.t_applied is None
                    else fraction_to_str(r.t_applied),
                    "sremu:config_hash": r.config_hash,
                    "sremu:status": r.status,
                    "sremu:counterfactual": bool(r.counterfactual),
                }
            )
        return out


# =========================================================================== run recorder


@runtime_checkable
class RecordSink(Protocol):
    """What a producer (engine or Device) feeds while a run executes.

    ``open_stream(spec)`` once per stream, in open order; ``write(stream, k0, samples)`` with
    accepted TX samples or delivered RX samples at absolute stream indices (increasing per
    stream); ``on_command(record, request, applied)`` whenever a command record appears or
    changes (``applied``: the panel configuration right after the switch, when known);
    ``close()`` once at the end.
    """

    def open_stream(self, spec: StreamSpec) -> None: ...

    def write(self, stream: StreamId, k0: int, samples: np.ndarray) -> None: ...

    def on_command(
        self, record: CommandRecord, request: CommandRequest | None = None, applied: object = None
    ) -> None: ...

    def close(self) -> object: ...


@dataclass
class RecordingIndex:
    """What :meth:`RunRecorder.close` wrote."""

    run_dir: Path
    metas: dict[StreamId, Path] = field(default_factory=dict)
    collection: Path | None = None
    events: Path | None = None


class RunRecorder:
    """Records a run directory: per-stream SigMF (``iq``), ``events.jsonl`` (``events``) and
    the SigMF collection. Thread-safe; implements :class:`RecordSink`."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        iq: bool = True,
        events: bool = True,
        metrics: bool = True,
        epoch_datetime: _dt.datetime | str | None = None,
        fsync: bool = False,
        description: str = "",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.iq = iq
        self._epoch = epoch_datetime
        self._description = description
        self._lock = threading.RLock()
        self._writers: dict[StreamId, SigMFWriter] = {}
        self._specs: dict[StreamId, StreamSpec] = {}
        self._log = EventLog(self.run_dir / EVENTS_NAME, fsync=fsync) if events else None
        self._metrics_on = metrics
        self._metrics: Any = None
        self._fsync = fsync
        self._index: RecordingIndex | None = None
        self.ignored_samples: dict[StreamId, int] = {}

    @property
    def streams(self) -> tuple[StreamId, ...]:
        """Opened streams in open order."""
        return tuple(self._specs)

    def open_stream(self, spec: StreamSpec) -> None:
        """Start the recording of a stream (no-op when already open)."""
        with self._lock:
            if spec.stream in self._specs:
                return
            self._specs[spec.stream] = spec
            if self.iq:
                self._writers[spec.stream] = SigMFWriter(
                    self.run_dir / IQ_DIR / spec.stream,
                    stream=spec.stream,
                    direction=spec.direction,
                    rate_sps=spec.rate_sps,
                    fc_hz=spec.fc_hz,
                    cpu_format=spec.cpu,
                    open_index=len(self._specs) - 1,
                    epoch_datetime=self._epoch,
                    description=self._description,
                )

    def write(self, stream: StreamId, k0: int, samples: np.ndarray) -> None:
        """Record samples of an open stream at absolute index ``k0``. Samples arriving
        after :meth:`close` are not written; they are counted in ``ignored_samples``."""
        with self._lock:
            if stream not in self._specs:
                raise KeyError(f"stream {stream!r} was not opened in the recorder")
            if self._index is not None:
                self.ignored_samples[stream] = self.ignored_samples.get(stream, 0) + int(
                    np.size(samples)
                )
                return
            w = self._writers.get(stream)
            if w is not None:
                w.write(k0, samples)

    def on_command(
        self,
        record: CommandRecord,
        request: CommandRequest | None = None,
        applied: object = None,
    ) -> None:
        """Log a new or changed command record (``events.jsonl``); ``applied`` is the panel
        configuration right after the switch (``PanelConfig``-like), when known."""
        with self._lock:
            if self._log is not None and self._index is None:
                self._log.append(record, request, AppliedConfig.of(applied))

    def on_metrics(self, window: Mapping[str, Any]) -> None:
        """Append one metrics window to ``metrics.jsonl`` (flushed per line)."""
        with self._lock:
            if not self._metrics_on or self._index is not None:
                return
            if self._metrics is None:
                self._metrics = (self.run_dir / METRICS_NAME).open(
                    "w", encoding="utf-8", newline="\n"
                )
            line = json.dumps(_finite(dict(window)), default=_jsonable, allow_nan=False)
            self._metrics.write(line + "\n")
            self._metrics.flush()
            if self._fsync:
                os.fsync(self._metrics.fileno())

    def commands(self) -> list[LoggedCommand]:
        """Final state of the commands logged so far."""
        with self._lock:
            return [] if self._log is None else self._log.final()

    def close(self) -> RecordingIndex:
        """Write the meta files and the collection; idempotent."""
        with self._lock:
            if self._index is not None:
                return self._index
            idx = RecordingIndex(self.run_dir)
            cmds = self.commands()
            for sid, w in self._writers.items():
                ann = w.annotations_for(cmds) if w.direction == "rx" else []
                idx.metas[sid] = w.close(ann)
            if self._writers:
                idx.collection = write_collection(self.run_dir, idx.metas.values())
            if self._log is not None:
                self._log.close()
                idx.events = self._log.path
            if self._metrics is not None:
                self._metrics.close()
            self._index = idx
            return idx


def _finite(obj: Any) -> Any:
    """Replace non-finite floats by None (JSON has no NaN or infinity), recursively."""
    if isinstance(obj, float | np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_finite(v) for v in obj]
    return obj


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return fraction_to_str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


class EngineRecorderSink:
    """Adapter from the WP6 engine's sink callbacks to a :class:`RunRecorder`.

    ``on_tx``/``on_rx`` record samples of the streams opened in the recorder (samples of
    other streams are counted in ``recorder.ignored_samples``), ``on_record`` logs final
    command records, ``on_metrics`` appends ``metrics.jsonl`` windows; other callbacks
    (``on_stall``, future hooks) are accepted and ignored.
    """

    def __init__(self, recorder: RunRecorder) -> None:
        self.recorder = recorder

    def _samples(self, stream: StreamId, k0: int, x: np.ndarray) -> None:
        rec = self.recorder
        if stream in rec.streams:
            rec.write(stream, int(k0), x)
        else:
            rec.ignored_samples[stream] = rec.ignored_samples.get(stream, 0) + int(np.size(x))

    def on_tx(self, stream: StreamId, k0: int, x: np.ndarray) -> None:
        """Accepted client TX samples at absolute index ``k0``."""
        self._samples(stream, k0, x)

    def on_rx(self, stream: StreamId, k0: int, d: np.ndarray) -> None:
        """Computed RX samples at absolute index ``k0``."""
        self._samples(stream, k0, d)

    def on_record(self, rec: CommandRecord, cfg: object = None) -> None:
        """A command record became final; ``cfg`` is the configuration after it."""
        self.recorder.on_command(rec, None, cfg)

    def on_metrics(self, window: Mapping[str, Any]) -> None:
        """A metrics window closed."""
        self.recorder.on_metrics(window)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)


def _sha512(path: Path) -> str:
    return hashlib.sha512(path.read_bytes()).hexdigest()


def write_collection(run_dir: str | Path, metas: Iterable[Path], description: str = "") -> Path:
    """Write ``run.sigmf-collection`` listing ``metas`` (names relative to ``run_dir``)."""
    root = Path(run_dir)
    streams = []
    for m in metas:
        rel = m.relative_to(root).as_posix()
        streams.append({"name": rel[: -len(_META_EXT)], "hash": _sha512(m)})
    obj = {
        "collection": {
            "core:version": SIGMF_VERSION,
            "core:description": description or "sremu run recordings",
            "core:extensions": [dict(SREMU_EXTENSION)],
            "core:streams": streams,
        }
    }
    path = root / COLLECTION_NAME
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


# =========================================================================== reading


@dataclass(frozen=True, eq=False)
class Recording:
    """A SigMF recording read back: meta, data (complex64) and decoded sremu fields."""

    meta_path: Path
    meta: dict[str, Any]
    data: np.ndarray
    stream: StreamId
    direction: str
    rate_sps: Fraction
    fc_hz: float
    cpu_format: str
    open_index: int
    captures: tuple[Capture, ...]

    def segments(self) -> list[tuple[int, np.ndarray]]:
        """``(global_index, samples)`` per capture segment."""
        return [
            (c.global_index, self.data[c.sample_start : c.sample_start + c.length])
            for c in self.captures
        ]


def _captures_of(meta: Mapping[str, Any], n_samples: int) -> tuple[Capture, ...]:
    caps = meta.get("captures", [])
    out = []
    for i, c in enumerate(caps):
        s = int(c["core:sample_start"])
        g = int(c.get("core:global_index", s))
        nxt = int(caps[i + 1]["core:sample_start"]) if i + 1 < len(caps) else n_samples
        out.append(Capture(s, g, nxt - s))
    return tuple(out)


def read_recording(meta_path: str | Path) -> Recording:
    """Read a ``cf32_le`` sremu recording (meta + data)."""
    mp = Path(meta_path)
    if mp.suffix != _META_EXT:
        mp = mp.with_name(mp.name + _META_EXT)
    meta = json.loads(mp.read_text(encoding="utf-8"))
    g = meta["global"]
    if g.get("core:datatype") != _DATATYPE:
        raise ValueError(f"{mp}: only {_DATATYPE} recordings are supported")
    data_path = mp.with_name(mp.name[: -len(_META_EXT)] + _DATA_EXT)
    data = np.fromfile(data_path, dtype="<c8").astype(np.complex64)
    caps = _captures_of(meta, int(data.size))
    fc = float(meta["captures"][0].get("core:frequency", 0.0)) if meta.get("captures") else 0.0
    return Recording(
        meta_path=mp,
        meta=meta,
        data=data,
        stream=str(g["sremu:stream"]),
        direction=str(g["sremu:direction"]),
        rate_sps=parse_exact(g["sremu:rate_sps"], name="sremu:rate_sps"),
        fc_hz=fc,
        cpu_format=str(g.get("sremu:cpu_format", "fc32")),
        open_index=int(g.get("sremu:open_index", 0)),
        captures=caps,
    )


def list_recordings(run_dir: str | Path) -> list[Recording]:
    """Every recording of a run (from the collection when present, else ``iq/``), sorted
    by ``sremu:open_index``."""
    root = Path(run_dir)
    coll = root / COLLECTION_NAME
    if coll.is_file():
        obj = json.loads(coll.read_text(encoding="utf-8"))
        metas = [root / f"{s['name']}{_META_EXT}" for s in obj["collection"]["core:streams"]]
    else:
        metas = sorted((root / IQ_DIR).glob(f"*{_META_EXT}"))
    recs = [read_recording(m) for m in metas]
    return sorted(recs, key=lambda r: r.open_index)


# =========================================================================== validators


def _is_int(x: Any) -> TypeGuard[int]:
    return isinstance(x, int) and not isinstance(x, bool)


def _namespaces(obj: Mapping[str, Any], where: str, declared: set[str], out: list[str]) -> None:
    for key in obj:
        ns, sep, _ = str(key).partition(":")
        if not sep:
            out.append(f"{where}: key {key!r} has no namespace")
        elif ns != "core" and ns not in declared:
            out.append(f"{where}: namespace {ns!r} of {key!r} is not declared in core:extensions")


def validate_meta(meta: Any, *, n_samples: int | None = None) -> list[str]:
    """Structural problems of a SigMF metadata object (empty list when valid).

    ``n_samples``: dataset length, when known, for range checks.
    """
    p: list[str] = []
    if not isinstance(meta, Mapping):
        return ["metadata must be a JSON object"]
    for k in ("global", "captures", "annotations"):
        if k not in meta:
            p.append(f"missing top-level {k!r}")
    extra = set(meta) - {"global", "captures", "annotations"}
    if extra:
        p.append(f"unexpected top-level keys {sorted(extra)}")
    if p:
        return p
    g = meta["global"]
    if not isinstance(g, Mapping):
        return ["'global' must be an object"]
    declared: set[str] = set()
    ext = g.get("core:extensions", [])
    if not isinstance(ext, list):
        p.append("global: core:extensions must be an array")
        ext = []
    for e in ext:
        if not isinstance(e, Mapping) or not {"name", "version", "optional"} <= set(e):
            p.append(f"global: extension entry {e!r} needs name, version and optional")
            continue
        if not isinstance(e["optional"], bool):
            p.append(f"global: extension {e['name']!r} optional must be a boolean")
        declared.add(str(e["name"]))
    dt = g.get("core:datatype")
    m = _DATATYPE_RE.match(dt) if isinstance(dt, str) else None
    if m is None:
        p.append(f"global: core:datatype {dt!r} is missing or not a SigMF dataset format")
    ver = g.get("core:version")
    if not (isinstance(ver, str) and _VERSION_RE.match(ver)):
        p.append(f"global: core:version {ver!r} is missing or not X.Y.Z")
    sr = g.get("core:sample_rate")
    if sr is not None and not (isinstance(sr, int | float) and not isinstance(sr, bool) and sr > 0):
        p.append(f"global: core:sample_rate {sr!r} must be a positive number")
    nch = g.get("core:num_channels")
    if nch is not None and not (_is_int(nch) and nch >= 1):
        p.append(f"global: core:num_channels {nch!r} must be an integer >= 1")
    _namespaces(g, "global", declared, p)
    if "sremu" in declared:
        for k in ("sremu:stream", "sremu:direction", "sremu:rate_sps"):
            if k not in g:
                p.append(f"global: missing {k!r}")
        if "sremu:rate_sps" in g and isinstance(sr, int | float):
            try:
                if float(parse_exact(g["sremu:rate_sps"])) != float(sr):
                    p.append("global: sremu:rate_sps disagrees with core:sample_rate")
            except (TypeError, ValueError):
                p.append(f"global: sremu:rate_sps {g['sremu:rate_sps']!r} is not exact")
    caps = meta["captures"]
    if not isinstance(caps, list):
        return [*p, "'captures' must be an array"]
    prev_s = prev_g = None
    for i, c in enumerate(caps):
        where = f"captures[{i}]"
        if not isinstance(c, Mapping):
            p.append(f"{where}: must be an object")
            continue
        s = c.get("core:sample_start")
        if not (_is_int(s) and s >= 0):
            p.append(f"{where}: core:sample_start {s!r} must be an integer >= 0")
            continue
        if prev_s is not None and s <= prev_s:
            p.append(f"{where}: captures must be sorted by core:sample_start (ascending)")
        gi = c.get("core:global_index")
        if gi is not None:
            if not (_is_int(gi) and gi >= 0):
                p.append(f"{where}: core:global_index {gi!r} must be an integer >= 0")
            elif prev_s is not None and prev_g is not None and gi - prev_g < s - prev_s:
                p.append(f"{where}: core:global_index overlaps the previous segment")
            else:
                prev_g = gi
        cdt = c.get("core:datetime")
        if cdt is not None and not (isinstance(cdt, str) and _DATETIME_RE.match(cdt)):
            p.append(f"{where}: core:datetime {cdt!r} is not ISO-8601 UTC with 'Z'")
        if n_samples is not None and s > n_samples:
            p.append(f"{where}: core:sample_start {s} beyond the dataset ({n_samples} samples)")
        _namespaces(c, where, declared, p)
        prev_s = s
    anns = meta["annotations"]
    if not isinstance(anns, list):
        return [*p, "'annotations' must be an array"]
    caps_ok = all(
        isinstance(c, Mapping)
        and _is_int(c.get("core:sample_start"))
        and _is_int(c.get("core:global_index", 0))
        for c in caps
    )
    cap_objs = _captures_of(meta, n_samples) if n_samples is not None and caps_ok else None
    prev = None
    for i, a in enumerate(anns):
        where = f"annotations[{i}]"
        if not isinstance(a, Mapping):
            p.append(f"{where}: must be an object")
            continue
        s = a.get("core:sample_start")
        if not (_is_int(s) and s >= 0):
            p.append(f"{where}: core:sample_start {s!r} must be an integer >= 0")
            continue
        if prev is not None and s < prev:
            p.append(f"{where}: annotations must be sorted by core:sample_start (ascending)")
        prev = s
        cnt = a.get("core:sample_count")
        if cnt is not None and not (_is_int(cnt) and cnt >= 0):
            p.append(f"{where}: core:sample_count {cnt!r} must be an integer >= 0")
        if n_samples is not None and s >= max(n_samples, 1):
            p.append(f"{where}: core:sample_start {s} beyond the dataset ({n_samples} samples)")
        if "sremu:k_applied" in a and cap_objs is not None:
            want = file_index(cap_objs, int(a["sremu:k_applied"]))
            if want != s:
                p.append(
                    f"{where}: core:sample_start {s} != dataset index {want} of "
                    f"sremu:k_applied {a['sremu:k_applied']}"
                )
        _namespaces(a, where, declared, p)
    return p


def _bytes_per_sample(datatype: str) -> int | None:
    m = _DATATYPE_RE.match(datatype)
    if m is None:
        return None
    size = _BYTES[m.group(2) or m.group(4)]
    return size * (2 if m.group(1) == "c" else 1)


def validate_recording(meta_path: str | Path) -> list[str]:
    """Problems of a recording on disk: :func:`validate_meta` plus the data file (exists,
    whole number of samples)."""
    mp = Path(meta_path)
    if mp.suffix != _META_EXT:
        mp = mp.with_name(mp.name + _META_EXT)
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{mp.name}: cannot read metadata: {exc}"]
    data = mp.with_name(mp.name[: -len(_META_EXT)] + _DATA_EXT)
    n: int | None = None
    probs: list[str] = []
    dt = meta.get("global", {}).get("core:datatype") if isinstance(meta, Mapping) else None
    bps = _bytes_per_sample(dt) if isinstance(dt, str) else None
    if not data.is_file():
        probs.append(f"{data.name}: dataset file is missing")
    elif bps is not None:
        size = data.stat().st_size
        if size % bps:
            probs.append(f"{data.name}: {size} bytes is not a whole number of {dt} samples")
        n = size // bps
    return [*probs, *(f"{mp.name}: {x}" for x in validate_meta(meta, n_samples=n))]


def validate_collection(path: str | Path) -> list[str]:
    """Problems of a ``.sigmf-collection``: structure, SHA-512 of every listed meta file,
    and each listed recording (:func:`validate_recording`)."""
    cp = Path(path)
    if cp.is_dir():
        cp = cp / COLLECTION_NAME
    try:
        obj = json.loads(cp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{cp.name}: cannot read collection: {exc}"]
    if not isinstance(obj, Mapping) or set(obj) != {"collection"}:
        return [f"{cp.name}: the collection object must be the only top-level key"]
    c = obj["collection"]
    p: list[str] = []
    ver = c.get("core:version")
    if not (isinstance(ver, str) and _VERSION_RE.match(ver)):
        p.append(f"{cp.name}: core:version {ver!r} is missing or not X.Y.Z")
    declared = {str(e.get("name")) for e in c.get("core:extensions", []) if isinstance(e, Mapping)}
    _namespaces(c, cp.name, declared, p)
    streams = c.get("core:streams")
    if not isinstance(streams, list):
        return [*p, f"{cp.name}: core:streams must be an array"]
    for i, s in enumerate(streams):
        if not isinstance(s, Mapping) or not {"name", "hash"} <= set(s):
            p.append(f"{cp.name}: core:streams[{i}] needs name and hash")
            continue
        meta = cp.parent / f"{s['name']}{_META_EXT}"
        if not meta.is_file():
            p.append(f"{cp.name}: {meta.name} is missing")
            continue
        if _sha512(meta) != s["hash"]:
            p.append(f"{cp.name}: SHA-512 of {meta.name} does not match")
        p.extend(validate_recording(meta))
    return p
