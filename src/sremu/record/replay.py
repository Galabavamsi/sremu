"""Bitwise replay of a recorded run (DESIGN.md section 4.12; acceptance RT-16).

``sremu replay RUN_DIR`` rebuilds the engine (lockstep) from the run directory, re-injects

* every recorded TX SigMF stream: each capture segment as one timed burst
  (``time_spec = from_ticks(core:global_index)``, SOB and EOB, so gaps are silence; a TX
  stream with no samples is committed with an empty EOB), and
* every command of ``events.jsonl`` **at its recorded** ``t_applied``, as an on-time
  command (``late_policy="reject"``), in the original ``seq`` order with the original
  ``cmd_id``, ``src`` and payload. Rejected commands, and commands that never got a
  ``t_applied``, are skipped; superseded ones are re-injected so the engine supersedes them
  again exactly as in the run (a superseded or never-applied command without any payload
  never took effect and is skipped). A command logged without its request (one issued inside the
  engine, e.g. by an in-process or remote controller) is re-injected from its
  ``applied_config``: the full panel configuration after the switch, as an all-elements
  ``ris.set_config`` (or ``ris.set_enabled``) with the recorded transition. Engines
  with ``schedule_physical`` (WP6) apply that per-cell configuration directly, for any
  control granularity and with dead or stuck cells; through ``ris.set_config`` it needs
  element-granularity control without dead cells (one control per physical cell),

then reads every recorded RX stream over the span of its recorded capture segments (one
``num_done`` per stream issued before any sample is computed; round robin across streams,
so lockstep frontiers advance together; samples in gaps are computed and discarded, which
I2a makes harmless) and compares the recorded samples **bitwise** with
the recording (complex64 bit patterns; the acceptance of RT-16). It also checks that each
re-injected command lands at the recorded ``k_applied`` on every RX stream.

The engine comes from an *engine factory* ``factory(run_dir) -> EngineProtocol``. The
default (:func:`default_engine_factory`) builds the WP6 ``Emulator`` from
``scenario.resolved.yaml`` in lockstep with the backend recorded in ``manifest.json`` and
the engine overrides of ``engine_overrides.json`` (written by ``Device.inproc`` for
overrides that are not scenario fields, e.g. ``noise=False``; :func:`write_engine_overrides`).
Seeds and in-process traffic generators come from the scenario, so generated TX streams
are regenerated, not re-injected. Streams are opened in their recorded open order
(``sremu:open_index``), which fixes the summation order of the replayed run.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np

from sremu.contracts import (
    CommandReceipt,
    CommandRequest,
    CommandTime,
    EngineProtocol,
    PanelConfig,
    StreamCmd,
    TransitionSpec,
    TxMetadata,
)
from sremu.errors import ErrorCode, SremuError
from sremu.record.eventlog import EVENTS_NAME, LoggedCommand, read_events
from sremu.record.sigmf import Recording, RunRecorder, list_recordings
from sremu.timebase import TimeSpec, fraction_to_str

__all__ = [
    "ENGINE_OVERRIDES_NAME",
    "EngineFactory",
    "ReplayError",
    "ReplayResult",
    "StreamComparison",
    "default_engine_factory",
    "load_engine_factory",
    "read_engine_overrides",
    "replay",
    "write_engine_overrides",
]

EngineFactory = Callable[[Path], EngineProtocol]
"""Builds a fresh engine for a run directory."""

ENGINE_OVERRIDES_NAME: Final[str] = "engine_overrides.json"
"""Run-directory file with the engine overrides that are not scenario fields."""

_NO_DATA_EFFECT: Final = frozenset({"clock", "sinks"})
"""Engine arguments that never change samples (wall clock, telemetry); not recorded."""

_REPLAY_FORCED: Final = frozenset({"mode", "coupling"})
"""Replay always runs lockstep with barrier coupling (section 4.12)."""

_SKIP = frozenset({"rejected"})


def _plain(v: Any) -> Any:
    """JSON form of an override value (exact quantities as ``"p/q"``); ``TypeError`` for
    anything that is not plain data."""
    if v is None or isinstance(v, bool | int | float | str):
        return v
    if isinstance(v, Fraction):
        return fraction_to_str(v)
    if isinstance(v, list | tuple):
        return [_plain(x) for x in v]
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    raise TypeError(type(v).__name__)


def write_engine_overrides(
    run_dir: str | Path, overrides: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Write the plain-data engine ``overrides`` that change samples to
    ``<run_dir>/engine_overrides.json`` (only when there are any; a stale file from an
    earlier run in the same directory is removed otherwise); returns ``(saved, unsaved)``:
    the JSON values written and the names of overrides that are not plain data (which a
    replay cannot reproduce). ``clock`` and ``sinks`` are ignored."""
    saved: dict[str, Any] = {}
    unsaved: list[str] = []
    for k, v in overrides.items():
        if k in _NO_DATA_EFFECT:
            continue
        try:
            saved[k] = _plain(v)
        except TypeError:
            unsaved.append(k)
    path = Path(run_dir) / ENGINE_OVERRIDES_NAME
    if saved:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"format": 1, "overrides": saved}, indent=2, allow_nan=False)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
    elif path.is_file():
        path.unlink()
    return saved, unsaved


def read_engine_overrides(run_dir: str | Path) -> dict[str, Any]:
    """The overrides of ``<run_dir>/engine_overrides.json`` (empty when absent)."""
    path = Path(run_dir) / ENGINE_OVERRIDES_NAME
    if not path.is_file():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, Mapping) or obj.get("format") != 1:
        raise ReplayError(f"{path}: unsupported engine overrides format")
    return dict(obj.get("overrides") or {})


class ReplayError(SremuError):
    """The run directory cannot be replayed (missing recordings, payloads or engine)."""


@dataclass(frozen=True)
class StreamComparison:
    """Replayed vs recorded samples of one RX stream."""

    stream: str
    n_samples: int
    n_read: int
    n_mismatch: int
    max_abs_diff: float

    @property
    def bitwise_equal(self) -> bool:
        """True when every recorded sample was reproduced with identical bits."""
        return self.n_read == self.n_samples and self.n_mismatch == 0


@dataclass
class ReplayResult:
    """Outcome of :func:`replay`."""

    run_dir: Path
    streams: dict[str, StreamComparison] = field(default_factory=dict)
    n_commands: int = 0
    skipped: list[tuple[int, str]] = field(default_factory=list)
    command_mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Every RX stream bitwise equal and every command at its recorded samples."""
        return all(s.bitwise_equal for s in self.streams.values()) and not self.command_mismatches

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary."""
        return {
            "run_dir": str(self.run_dir),
            "ok": self.ok,
            "n_commands": self.n_commands,
            "skipped": [{"seq": s, "status": st} for s, st in self.skipped],
            "command_mismatches": list(self.command_mismatches),
            "streams": {
                k: {
                    "n_samples": v.n_samples,
                    "n_read": v.n_read,
                    "n_mismatch": v.n_mismatch,
                    "max_abs_diff": v.max_abs_diff,
                    "bitwise_equal": v.bitwise_equal,
                }
                for k, v in self.streams.items()
            },
        }


# =========================================================================== factories


def default_engine_factory(run_dir: Path) -> EngineProtocol:
    """WP6 engine from ``scenario.resolved.yaml`` (lockstep, backend from ``manifest.json``,
    plus the overrides of ``engine_overrides.json``)."""
    from sremu.api.device import _wp6_module

    sc_path = Path(run_dir) / "scenario.resolved.yaml"
    if not sc_path.is_file():
        raise ReplayError(f"{sc_path} is missing; give an engine factory for this run")
    emu = _wp6_module()
    overrides: dict[str, Any] = {
        k: v for k, v in read_engine_overrides(run_dir).items() if k not in _REPLAY_FORCED
    }
    overrides.update(mode="lockstep", coupling="barrier")
    man = Path(run_dir) / "manifest.json"
    if man.is_file():
        backend = json.loads(man.read_text(encoding="utf-8")).get("backend")
        if backend:
            overrides["backend"] = backend
    engine: EngineProtocol = emu.Emulator.from_scenario(sc_path, **overrides)
    return engine


def load_engine_factory(spec: str) -> EngineFactory:
    """``"package.module:function"`` -> the engine factory it names."""
    mod_name, sep, attr = spec.partition(":")
    if not sep or not mod_name or not attr:
        raise ReplayError(f"engine factory {spec!r} must look like 'package.module:function'")
    fn = getattr(importlib.import_module(mod_name), attr, None)
    if not callable(fn):
        raise ReplayError(f"{spec!r} is not a callable engine factory")
    return fn


# =========================================================================== replay


def _bits(x: np.ndarray) -> np.ndarray:
    """(n, 2) uint32 bit patterns of complex64 samples (I, Q)."""
    return np.ascontiguousarray(x, dtype=np.complex64).view(np.uint32).reshape(-1, 2)


def _replay_request(lc: LoggedCommand) -> CommandRequest:
    """The command to re-inject: the logged request at ``t_applied``, else (no request,
    e.g. an in-process or remote controller's command) the applied configuration as a
    full-panel ``ris.set_config`` / ``ris.set_enabled`` at ``t_applied``."""
    rec = lc.record
    assert rec.t_applied is not None
    at = CommandTime.at(rec.t_applied)
    if lc.request is not None:
        return replace(lc.request, at=at, late_policy="reject", t_wall_recv_ns=None)
    a = lc.applied
    if a is None or rec.op not in ("ris.set_config", "ris.set_enabled"):
        raise ReplayError(
            f"command seq {rec.seq} ({rec.cmd_id}) has no request payload and no applied "
            "configuration in events.jsonl; it cannot be re-injected"
        )
    if rec.op == "ris.set_enabled":
        return CommandRequest(
            rec.cmd_id,
            "ris.set_enabled",
            rec.src,
            rec.panel,
            at,
            enabled=a.enabled,
            late_policy="reject",
        )
    tr = rec.transition
    return CommandRequest(
        rec.cmd_id,
        "ris.set_config",
        rec.src,
        rec.panel,
        at,
        states=a.state.copy(),
        transition=None
        if tr is None
        else TransitionSpec(tr["model"], tr.get("duration_s"), tr.get("classes", "single")),
        late_policy="reject",
    )


def _reinject(dev: Any, engine: EngineProtocol, lc: LoggedCommand) -> CommandReceipt:
    """Re-inject one logged command on time at its recorded ``t_applied``.

    A ``ris.set_config`` logged without its request but with its applied configuration goes
    through the engine's optional ``schedule_physical(panel, t, cfg, transition=, cmd_id=,
    src=)`` (WP6, section 4.12) when it has one: the recorded per-cell configuration is
    applied as is, which works for every control granularity and for dead or stuck cells.
    Everything else goes through :func:`_replay_request` and ``Device.schedule``.
    """
    rec = lc.record
    a = lc.applied
    physical = getattr(engine, "schedule_physical", None)
    if lc.request is None and a is not None and rec.op == "ris.set_config" and callable(physical):
        assert rec.t_applied is not None
        receipt: CommandReceipt = physical(
            rec.panel,
            rec.t_applied,
            PanelConfig(rec.panel, a.state.copy(), bool(a.enabled)),
            transition=rec.transition,
            cmd_id=rec.cmd_id,
            src=rec.src,
        )
        return receipt
    out: CommandReceipt = dev.schedule(_replay_request(lc))
    return out


def replay(
    run_dir: str | Path,
    *,
    engine_factory: EngineFactory | None = None,
    out: str | Path | None = None,
    chunk: int = 4096,
) -> ReplayResult:
    """Replay ``run_dir`` (module docstring) and compare RX bitwise.

    ``out``: optional directory where the replayed run is recorded (SigMF + events).
    ``chunk``: RX samples per read in the round robin.
    """
    from sremu.api.device import Device, engine_command_log
    from sremu.api.streams import StreamArgs

    root = Path(run_dir)
    recs = list_recordings(root)
    if not recs:
        raise ReplayError(f"{root}: no SigMF recordings (record with iq enabled)")
    rx_recs = [r for r in recs if r.direction == "rx"]
    if not rx_recs:
        raise ReplayError(f"{root}: no RX recording to compare")
    events = read_events(root / EVENTS_NAME) if (root / EVENTS_NAME).is_file() else []
    engine = (engine_factory or default_engine_factory)(root)
    rec_out = RunRecorder(out) if out is not None else None
    dev = Device(engine, recorder=rec_out, src="replay")
    result = ReplayResult(root)
    try:
        streams: dict[str, Any] = {}
        for r in recs:
            args = StreamArgs(r.stream, r.fc_hz, r.rate_sps, "fc32")
            streams[r.stream] = (
                dev.get_tx_stream(args) if r.direction == "tx" else dev.get_rx_stream(args)
            )
        # commands, in seq order, at their recorded t_applied
        sent: list[tuple[LoggedCommand, int]] = []
        for lc in events:
            st = lc.record.status
            never_applied = st in ("scheduled", "superseded")
            no_payload = lc.request is None and lc.applied is None
            if st in _SKIP or lc.record.t_applied is None or (never_applied and no_payload):
                result.skipped.append((lc.record.seq, st))
                continue
            receipt = _reinject(dev, engine, lc)
            if not receipt.ok or receipt.seq is None:
                result.command_mismatches.append(
                    f"seq {lc.record.seq} ({lc.record.cmd_id}): re-injection rejected: "
                    f"{receipt.error.name} {receipt.msg}"
                )
                continue
            sent.append((lc, int(receipt.seq)))
        result.n_commands = len(sent)
        # TX: one timed burst per capture segment
        for r in recs:
            if r.direction != "tx":
                continue
            tx = streams[r.stream]
            segs = [(g, x) for g, x in r.segments() if x.size]
            for g, x in segs:
                md = TxMetadata(
                    has_time_spec=True,
                    time_spec=TimeSpec.from_ticks(g, r.rate_sps),
                    start_of_burst=True,
                    end_of_burst=True,
                )
                if tx.send(x, md) != x.size:
                    raise ReplayError(f"{r.stream}: the engine refused TX samples at {g}")
            if not segs:
                tx.send(np.zeros(0, np.complex64), TxMetadata(end_of_burst=True))
        # RX: round robin over the recorded capture segments
        got = _read_all(rx_recs, streams, chunk)
        for r in rx_recs:
            want = r.data
            have = got[r.stream]
            n = min(want.size, have.size)
            mism = int(np.count_nonzero(np.any(_bits(want[:n]) != _bits(have[:n]), axis=-1)))
            diff = float(np.max(np.abs(want[:n].astype(np.complex128) - have[:n]))) if n else 0.0
            result.streams[r.stream] = StreamComparison(
                r.stream, int(want.size), int(have.size), mism + (want.size - n), diff
            )
        # commands landed where they did in the run
        log = {int(rec.seq): rec for rec, _ in engine_command_log(engine)}
        for lc, seq in sent:
            now = log.get(seq)
            if now is None:
                continue
            if lc.record.status in ("applied", "late_applied"):
                for s, k in lc.record.k_applied.items():
                    if now.k_applied.get(s) != k:
                        result.command_mismatches.append(
                            f"seq {lc.record.seq} ({lc.record.cmd_id}) on {s}: "
                            f"k_applied {now.k_applied.get(s)} != recorded {k}"
                        )
    finally:
        dev.close()
    return result


def _read_all(rx_recs: list[Recording], streams: dict[str, Any], chunk: int) -> dict[str, Any]:
    """Read every RX stream over the span of its recorded captures (one ``num_done`` per
    stream, issued before any sample is computed, so no command can be late), round robin
    in ``chunk``-sample reads; returns the recorded segments of each stream concatenated."""
    from sremu.contracts import RxMetadata

    span: dict[str, tuple[int, int]] = {}
    for r in rx_recs:
        caps = [c for c in r.captures if c.length]
        if caps:
            span[r.stream] = (caps[0].global_index, caps[-1].global_end)
    for r in rx_recs:
        if r.stream in span:
            k0, k1 = span[r.stream]
            streams[r.stream].issue_stream_cmd(
                StreamCmd(
                    "num_done",
                    num_samps=k1 - k0,
                    stream_now=False,
                    time_spec=TimeSpec.from_ticks(k0, r.rate_sps),
                )
            )
    bufs = {s: np.zeros(k1 - k0, np.complex64) for s, (k0, k1) in span.items()}
    got = dict.fromkeys(span, 0)
    idle_rounds = 0
    while any(got[s] < bufs[s].size for s in span):
        progressed = False
        for s in span:
            left = bufs[s].size - got[s]
            if not left:
                continue
            md = RxMetadata()
            n = streams[s].recv(bufs[s][got[s] : got[s] + min(chunk, left)], md, timeout=0.0)
            if n:
                if md.first_sample != span[s][0] + got[s]:
                    raise ReplayError(f"{s}: replay returned samples from {md.first_sample}")
                got[s] += n
                progressed = True
            elif md.error_code not in (ErrorCode.TIMEOUT, ErrorCode.NONE):
                raise ReplayError(f"{s}: RX error {md.error_code.name} during replay")
        idle_rounds = 0 if progressed else idle_rounds + 1
        if idle_rounds > 1:
            stuck = sorted(s for s in span if got[s] < bufs[s].size)
            raise ReplayError(f"replay stalled: no RX progress on {stuck} (lockstep inputs)")
    out: dict[str, Any] = {}
    for r in rx_recs:
        if r.stream not in span:
            out[r.stream] = np.zeros(0, np.complex64)
            continue
        k0 = span[r.stream][0]
        parts = [
            bufs[r.stream][c.global_index - k0 : c.global_end - k0] for c in r.captures if c.length
        ]
        out[r.stream] = np.concatenate(parts)
    return out
