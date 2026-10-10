"""``sremu run S.yaml --out DIR [--mode paced]`` (WP6; DESIGN.md sections 2.5, 3.6, 8.6).

Headless run of a scenario through :class:`sremu.engine.emulator.Emulator`: the scenario's
in-process traffic generators drive the TX streams, every RX stream is computed, and the run
directory receives

* ``manifest.json`` and ``scenario.resolved.yaml`` (:mod:`sremu.record.manifest`);
* ``events.jsonl``, ``metrics.jsonl`` and, with ``record.iq``, SigMF recordings of every RX
  stream plus ``run.sigmf-collection``, written by WP7's ``sremu.record.sigmf.RunRecorder``
  (the event log keeps each command's applied configuration, which ``sremu replay`` needs);
* ``summary.json``: samples, counters, RTF, flags and the paced deadline statistics.

Without WP7's recorder an interim writer (:class:`RunDirSink`) produces ``events.jsonl``
(one ``CommandRecord`` per line), ``applied_configs.jsonl`` (applied physical states as
``states_b64``), ``metrics.jsonl`` and raw ``iq/<stream>.sigmf-data`` files.

``--serve`` (control socket and native data plane) belongs to WP8 and exits with code 3
until ``sremu.transport`` provides it. Scenario controllers of kind ``builtin`` are created
from ``sremu.control.builtin`` (WP9); ``external`` controllers need ``--serve``.

Exit codes: 0 success; 1 run failure (stall, rejected action); 2 scenario error; 3 not yet
implemented (``--serve`` or missing controller package).
"""

from __future__ import annotations

import base64
import importlib
import json
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, TextIO

import numpy as np
import typer

from sremu.contracts import CommandRecord, PanelConfig
from sremu.engine.emulator import EngineSink
from sremu.errors import SremuError
from sremu.timebase import fraction_to_str

__all__ = ["RunDirSink", "command", "controllers_from_scenario", "make_recorder"]

NOT_IMPLEMENTED_EXIT = 3


class RunDirSink(EngineSink):
    """Writes ``events.jsonl``, ``applied_configs.jsonl``, ``metrics.jsonl`` and IQ files."""

    def __init__(
        self,
        run_dir: Path,
        *,
        iq: bool,
        events: bool,
        metrics: bool,
        rates: Mapping[str, Any],
        fcs: Mapping[str, float],
    ) -> None:
        self.run_dir = run_dir
        self._events: TextIO | None = (
            (run_dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") if events else None
        )
        self._configs: TextIO | None = (
            (run_dir / "applied_configs.jsonl").open("a", encoding="utf-8", newline="\n")
            if events
            else None
        )
        self._metrics: TextIO | None = (
            (run_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n")
            if metrics
            else None
        )
        self._iq: dict[str, Any] = {}
        self._iq_next: dict[str, int] = {}
        self.rates = dict(rates)
        self.fcs = dict(fcs)
        if iq:
            (run_dir / "iq").mkdir(exist_ok=True)
        self.iq = iq
        self.records = 0

    def on_record(self, rec: CommandRecord, cfg: PanelConfig | None) -> None:
        self.records += 1
        if self._events is not None:
            self._events.write(json.dumps(rec.to_dict(), sort_keys=False) + "\n")
            self._events.flush()
        if (
            self._configs is not None
            and cfg is not None
            and rec.status in ("applied", "late_applied")
        ):
            st = np.ascontiguousarray(cfg.state, dtype="<i2")
            line = {
                "seq": rec.seq,
                "cmd_id": rec.cmd_id,
                "panel": rec.panel,
                "t_applied": None if rec.t_applied is None else fraction_to_str(rec.t_applied),
                "states_b64": base64.b64encode(st.tobytes()).decode("ascii"),
                "dtype": "i2",
                "enabled": bool(cfg.enabled),
                "config_hash": cfg.config_hash(),
            }
            self._configs.write(json.dumps(line) + "\n")
            self._configs.flush()

    def on_metrics(self, window: Mapping[str, Any]) -> None:
        if self._metrics is not None:
            self._metrics.write(json.dumps(dict(window)) + "\n")

    def on_rx(self, stream: str, k0: int, d: np.ndarray) -> None:
        if not self.iq:
            return
        fh = self._iq.get(stream)
        if fh is None:
            fh = (self.run_dir / "iq" / f"{stream}.sigmf-data").open("wb")
            self._iq[stream] = fh
            self._iq_next[stream] = k0
            self._write_meta(stream, k0)
        if k0 != self._iq_next[stream]:  # skipped samples: keep the file gap-free (zeros)
            fh.write(np.zeros(k0 - self._iq_next[stream], dtype="<c8").tobytes())
        fh.write(np.ascontiguousarray(d, dtype="<c8").tobytes())
        self._iq_next[stream] = k0 + int(d.shape[0])

    def _write_meta(self, stream: str, k0: int) -> None:
        rate = self.rates[stream]
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": float(rate),
                "core:version": "1.2.0",
                "core:description": "sremu RX stream (interim WP6 writer)",
                "core:extensions": [{"name": "sremu", "version": "1.0.0", "optional": True}],
                "sremu:sample_rate_exact": fraction_to_str(rate),
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:global_index": int(k0),
                    "core:frequency": self.fcs[stream],
                }
            ],
            "annotations": [],
        }
        (self.run_dir / "iq" / f"{stream}.sigmf-meta").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    def close(self) -> None:
        for fh in (self._events, self._configs, self._metrics, *self._iq.values()):
            if fh is not None:
                fh.close()


def make_recorder(
    run_dir: Path, eng: Any, *, iq: bool, events: bool, metrics: bool
) -> tuple[Any, Any]:
    """The run directory writer: WP7's ``RunRecorder`` (canonical SigMF, ``events.jsonl``
    with applied configurations, ``metrics.jsonl``) through its ``EngineRecorderSink`` when
    available, else the interim :class:`RunDirSink`. Every RX stream is recorded; TX streams
    driven by in-process generators are regenerated from the scenario on replay. Returns
    ``(sink, close)``."""
    try:
        from sremu.record.sigmf import EngineRecorderSink, RunRecorder
    except ModuleNotFoundError:
        fallback = RunDirSink(
            run_dir,
            iq=iq,
            events=events,
            metrics=metrics,
            rates={s: c.rate for s, c in eng.configs.items()},
            fcs={s: c.fc_hz for s, c in eng.configs.items()},
        )
        return fallback, fallback.close
    rr = RunRecorder(run_dir, iq=iq, events=events, metrics=metrics)
    for c in eng.configs.values():
        if c.direction == "rx":
            rr.open_stream(c.spec())
    return EngineRecorderSink(rr), rr.close


def controllers_from_scenario(scenario: Mapping[str, Any], model: Any = None) -> list[Any]:
    """In-process controllers of the scenario (``kind: builtin``) from WP9's
    ``sremu.control.builtin.build_controller(spec, model=...)``. Raises
    ``ModuleNotFoundError`` when WP9 is missing and ``SremuError`` for external controllers
    (they need ``--serve``, WP8)."""
    out: list[Any] = []
    specs = list(scenario.get("controllers") or [])
    if not specs:
        return out
    for c in specs:
        if c.get("kind") == "external":
            raise SremuError(f"controller {c['name']!r} is external: run with --serve (WP8)")
    builtin = importlib.import_module("sremu.control.builtin")
    factory = getattr(builtin, "build_controller", None)
    for c in specs:
        if factory is not None:
            out.append(factory(c, model=model))
            continue
        cls = getattr(builtin, str(c.get("impl")), None)
        if cls is None:
            raise SremuError(f"controller {c['name']!r}: unknown builtin {c.get('impl')!r}")
        out.append(cls(**dict(c.get("params") or {})))
    return out


def _apply_overrides(
    res: Any, *, mode: str | None, backend: str | None, duration_s: str | None
) -> Any:
    """A ``ResolvedScenario`` with ``run.mode``, ``numerics.backend`` and
    ``run.duration_s`` replaced (validated again by the scenario model)."""
    import dataclasses

    from sremu.scenario.models import Scenario

    if mode is None and backend is None and duration_s is None:
        return res
    d = res.scenario.model_dump(mode="python")
    if mode is not None:
        d["run"]["mode"] = mode
    if backend is not None:
        d.setdefault("numerics", {})["backend"] = backend
    if duration_s is not None:
        d["run"]["duration_s"] = duration_s
    return dataclasses.replace(res, scenario=Scenario.model_validate(d))


def command(
    scenario: Annotated[Path, typer.Argument(help="Scenario file (.yaml, .json or .toml).")],
    out: Annotated[Path, typer.Option("--out", help="Root directory of run directories.")] = Path(
        "runs"
    ),
    mode: Annotated[
        str | None, typer.Option("--mode", help="lockstep or paced (default: scenario).")
    ] = None,
    serve: Annotated[
        str | None, typer.Option("--serve", help="Serve on a control endpoint (WP8).")
    ] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", help="numba_cpu or numpy_ref.")
    ] = None,
    duration_s: Annotated[
        str | None,
        typer.Option("--duration-s", help="Exact run duration (overrides run.duration_s)."),
    ] = None,
    name: Annotated[
        str | None, typer.Option("--name", help="Run name (default: scenario name).")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
) -> None:
    """Run a scenario headless and write a run directory."""
    if serve is not None:
        try:
            importlib.import_module("sremu.transport.control_server")
        except ModuleNotFoundError:
            typer.echo(
                "sremu run --serve: needs the WP8 transport (sremu.transport.control_server)",
                err=True,
            )
            raise typer.Exit(NOT_IMPLEMENTED_EXIT) from None
        typer.echo("sremu run --serve is provided by WP8; use its entry point", err=True)
        raise typer.Exit(NOT_IMPLEMENTED_EXIT)
    from sremu.engine.emulator import Emulator
    from sremu.record.manifest import ManifestRecorder, create_run_dir
    from sremu.scenario.load import load_scenario

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = load_scenario(scenario)
    except (SremuError, OSError) as exc:
        typer.echo(f"INVALID {scenario}\n{exc}", err=True)
        raise typer.Exit(2) from None
    d = res.scenario.model_dump(mode="python")
    try:
        if d.get("controllers"):
            importlib.import_module("sremu.control.builtin")
    except ModuleNotFoundError:
        typer.echo(
            "scenario controllers need WP9 (sremu.control.builtin), not available yet", err=True
        )
        raise typer.Exit(NOT_IMPLEMENTED_EXIT) from None
    except SremuError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(NOT_IMPLEMENTED_EXIT) from None
    # command-line overrides become part of the resolved scenario, so the manifest and
    # scenario.resolved.yaml describe what actually ran
    try:
        res = _apply_overrides(res, mode=mode, backend=backend, duration_s=duration_s)
    except (SremuError, ValueError, TypeError) as exc:
        typer.echo(f"bad override: {exc}", err=True)
        raise typer.Exit(2) from None
    d = res.scenario.model_dump(mode="python")
    run_dir = create_run_dir(out, name or res.scenario.name)
    recorder = ManifestRecorder(res, run_dir, argv=sys.argv)
    rec = d.get("record") or {}
    try:
        eng = Emulator.from_scenario(res, always_active=True)
    except SremuError as exc:
        typer.echo(f"cannot build the engine: {exc}", err=True)
        raise typer.Exit(2) from None
    sink, close_sink = make_recorder(
        run_dir,
        eng,
        iq=bool(rec.get("iq", False)),
        events=bool(rec.get("events", True)),
        metrics=bool(rec.get("metrics", True)),
    )
    eng.add_sink(sink)
    try:
        for c in controllers_from_scenario(d, eng.model):
            eng.register_controller(c)
    except SremuError as exc:
        close_sink()
        typer.echo(f"cannot build the scenario controllers: {exc}", err=True)
        raise typer.Exit(2) from None
    rc = 0
    try:
        summary = eng.run_paced() if eng.cfg.mode == "paced" else eng.run()
    except SremuError as exc:
        typer.echo(f"run failed: {type(exc).__name__}: {exc}", err=True)
        summary = eng.summary()
        rc = 1
    finally:
        close_sink()
    info = eng.manifest_info()
    for r, p in info["page_size"].items():
        recorder.set_page_size(r, p)
    pol = info["snapshot_policy"]
    for lk, dec in info["link_decisions"].items():
        recorder.set_link(
            lk,
            {
                "narrowband": dec["narrowband"],
                "fd_filter": dec["fd_filter"],
                "t_u_s": pol.get("t_u_s_used"),
                "tiles": {p: int(t[0]) * int(t[1]) for p, t in pol.get("tiles", {}).items()},
                "interp_error_bound": pol.get("total_bound_rel"),
                "details": {"delay_spread_s": dec["delay_spread_s"], "n_groups": dec["n_groups"]},
            },
        )
    if summary.verify_interp_max_rel is not None:
        recorder.update_verify_interp(summary.verify_interp_max_rel)
    recorder.add_counters(summary.counters)
    if summary.nondeterministic_inputs:
        recorder.mark_nondeterministic()
    if summary.oracle_csi_used:
        recorder.mark_oracle_csi_used()
    recorder.finish()
    out_d = {
        "run_dir": str(run_dir),
        "samples": summary.samples,
        "counters": summary.counters,
        "rtf": summary.rtf,
        "nondeterministic_inputs": summary.nondeterministic_inputs,
        "oracle_csi_used": summary.oracle_csi_used,
        "stalls": summary.stalls,
        "verify_interp_max_rel": summary.verify_interp_max_rel,
        "paced": summary.paced,
        "records": len(eng.records),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(out_d, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if json_out:
        typer.echo(json.dumps(out_d, indent=2, default=str))
    else:
        rtf = summary.rtf.get("rtf")
        typer.echo(f"run {run_dir}")
        v = summary.rtf.get("virtual_s_float")
        typer.echo(f"  virtual {v} s, RTF {rtf}, RTF_min {summary.rtf.get('rtf_min')}")
        typer.echo(f"  records {len(eng.records)}; counters {summary.counters}")
        if summary.nondeterministic_inputs:
            typer.echo("  nondeterministic_inputs: true")
    if rc:
        raise typer.Exit(rc)
