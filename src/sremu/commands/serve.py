"""Served runs: ``sremu run S.yaml --serve tcp://127.0.0.1:5600`` (WP8; DESIGN.md 2.5, 8.3-8.5).

Builds the scenario's emulator (lockstep by default), registers its controllers in
scenario order (``builtin`` ones in-process, ``external`` ones as remote proxies that the
remote side attaches to by name, section 4.9) and serves it with
:class:`sremu.transport.control_server.ControlServer`:

* control ``ROUTER`` (``--serve``, default ``tcp://127.0.0.1:5600``), telemetry ``PUB``
  (``--pub``, ``:5601``; ``none`` disables), native TX ingress (``--tx``, ``:5602``); RX
  egress ports are ephemeral and returned by ``stream.open``;
* every bind must be a loopback address unless ``--allow-remote`` (section 8.10);
* ``--paused`` (default): virtual time starts at the first ``sim.resume``, so clients can
  open streams and issue timed stream commands first; TX streams no client has opened are
  waited for (``--unopened-tx wait``, default);
* raw REQ/REP peers: ``--raw-peer "name=enb,format=srsran_dummy,tx_port=tcp://127.0.0.1:2000,
  rx_port=tcp://127.0.0.1:2001,tx_stream=enb.tx0,rx_stream=enb.rx0"`` (lockstep only;
  ``request_items`` defaults to 8 pages for ``format=gnuradio_count``; GNU Radio example in
  ``examples/gnuradio_zmq``);
* ``--out DIR``: record a run directory as ``sremu run`` does (manifest, events, metrics,
  SigMF of every RX stream with ``record.iq``).

On start one line ``SREMU-SERVE {json}`` with the bound endpoints is printed to stdout
(flushed), so scripts can use ephemeral ports (``tcp://127.0.0.1:0``). The server stops on
``sim.shutdown``, on ``--exit-when-done`` once the run has ended and every RX egress has
been drained, or on Ctrl-C. With raw peers, ``--exit-when-done`` needs each peer to read its
RX stream to the end of the run (the engine cannot finish while a raw consumer stops
reading); at the end the peer still gets the samples queued for it.

Until the CLI registry routes ``sremu run --serve`` here (contract change request WP8-1),
run it as ``python -m sremu.commands.serve S.yaml [options]``.

Exit codes: 0 stopped normally; 1 the engine failed; 2 scenario or option error.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Annotated, Any

import typer

__all__ = ["command", "main", "parse_raw_peer", "run_served"]


def parse_raw_peer(text: str) -> Any:
    """A :class:`~sremu.transport.rawreqrep.RawPeerConfig` from ``key=value,...``
    (keys: name, format, tx_port, rx_port, tx_stream, rx_stream, sample_format,
    base_srate, request_items, reply_items)."""
    from sremu.transport.rawreqrep import RawPeerConfig

    d: dict[str, Any] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise typer.BadParameter(f"--raw-peer item {item!r} is not key=value")
        k, v = item.split("=", 1)
        d[k.strip()] = v.strip()
    if "format" in d:
        d["request_format"] = d.pop("format")
    for k in ("request_items", "reply_items"):
        if k in d:
            d[k] = int(d[k])
    try:
        return RawPeerConfig(**d)
    except TypeError as exc:
        raise typer.BadParameter(f"--raw-peer: {exc}") from None


def _scenario_mode(scenario: Path) -> str | None:
    """``run.mode`` of a scenario file read without validation (None when absent or unreadable;
    the real load validates it later)."""
    try:
        if scenario.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(scenario.read_text(encoding="utf-8"))
        elif scenario.suffix.lower() == ".json":
            import json

            data = json.loads(scenario.read_text(encoding="utf-8"))
        else:
            import tomllib

            data = tomllib.loads(scenario.read_text(encoding="utf-8"))
        run = data.get("run") if isinstance(data, dict) else None
        mode = run.get("mode") if isinstance(run, dict) else None
        return str(mode) if mode is not None else None
    except Exception:
        return None


def _passive_omp_for_paced() -> None:
    """A paced server runs few, short OpenMP regions (served paced pages render serially below
    ``render_numba.PACED_PARALLEL_MIN_WORK``); with the runtime's default wait policy the idle
    workers spin for milliseconds after each region beside the control and transport threads.
    Set ``OMP_WAIT_POLICY=PASSIVE`` before numba starts, as the B4 and E-B3 suites measure it
    (DESIGN 4.10). An explicit environment setting wins, and nothing changes once numba is
    imported."""
    import os
    import sys

    if "OMP_WAIT_POLICY" not in os.environ and "numba" not in sys.modules:
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"


def run_served(
    scenario: Path,
    *,
    control: str = "tcp://127.0.0.1:5600",
    pub: str | None = "tcp://127.0.0.1:5601",
    tx: str = "tcp://127.0.0.1:5602",
    allow_remote: bool = False,
    paused: bool = True,
    exit_when_done: bool = False,
    out: Path | None = None,
    mode: str | None = None,
    backend: str | None = None,
    duration_s: str | None = None,
    name: str | None = None,
    raw_peers: tuple[Any, ...] = (),
    unopened_tx: str = "wait",
    ack_timeout_s: float | None = None,
    echo: Any = print,
) -> int:
    """Serve a scenario until shutdown (see the module docstring); returns the exit code."""
    if mode == "paced" or (mode is None and _scenario_mode(scenario) == "paced"):
        _passive_omp_for_paced()
    from sremu.commands.run import _apply_overrides, make_recorder
    from sremu.engine.emulator import Emulator
    from sremu.errors import SremuError
    from sremu.scenario.load import load_scenario
    from sremu.transport.control_server import ControlServer, ServerConfig

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = load_scenario(scenario)
        res = _apply_overrides(res, mode=mode, backend=backend, duration_s=duration_s)
    except (SremuError, OSError, ValueError, TypeError) as exc:
        echo(f"INVALID {scenario}\n{exc}", file=sys.stderr)
        return 2
    d = res.scenario.model_dump(mode="python")
    try:
        eng = Emulator.from_scenario(res, unopened_tx=unopened_tx, external_inputs=True)
    except SremuError as exc:
        echo(f"cannot build the engine: {exc}", file=sys.stderr)
        return 2
    recorder = close_sink = None
    run_dir = None
    if out is not None:
        from sremu.record.manifest import ManifestRecorder, create_run_dir

        run_dir = create_run_dir(out, name or res.scenario.name)
        recorder = ManifestRecorder(res, run_dir, argv=sys.argv)
        rec = d.get("record") or {}
        sink, close_sink = make_recorder(
            run_dir,
            eng,
            iq=bool(rec.get("iq", False)),
            events=bool(rec.get("events", True)),
            metrics=bool(rec.get("metrics", True)),
        )
        eng.add_sink(sink)
    try:
        builtin = None
        if any(c.get("kind") == "builtin" for c in d.get("controllers") or []):
            from sremu.control.builtin import build_controller

            def builtin(spec: Any) -> Any:
                return build_controller(spec, model=eng.model)

        cfg = ServerConfig(
            control_endpoint=control,
            pub_endpoint=pub,
            tx_endpoint=tx,
            allow_remote=allow_remote,
            start_paused=paused,
            exit_when_done=exit_when_done,
            raw_peers=tuple(raw_peers),
            ack_timeout_s=ack_timeout_s,
        )
        server = ControlServer(eng, cfg, scenario=d, builtin_factory=builtin)
        server.start()
    except SremuError as exc:
        if close_sink is not None:
            close_sink()
        echo(f"cannot serve: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    echo("SREMU-SERVE " + json.dumps({"endpoints": server.endpoints, "paused": server.paused}))
    sys.stdout.flush()
    rc = 0
    try:
        while not server.wait(0.2):
            pass
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        server.stop()
        if close_sink is not None:
            close_sink()
    if server.engine_error is not None:
        echo(f"engine failed: {server.engine_error}", file=sys.stderr)
        rc = 1
    if recorder is not None and run_dir is not None:
        summary = eng.summary()
        info = eng.manifest_info()
        for r, p in info["page_size"].items():
            recorder.set_page_size(r, p)
        recorder.add_counters({**summary.counters, **_prefixed(server.counters.snapshot())})
        if summary.nondeterministic_inputs:
            recorder.mark_nondeterministic()
        if summary.oracle_csi_used:
            recorder.mark_oracle_csi_used()
        if summary.verify_interp_max_rel is not None:
            recorder.update_verify_interp(summary.verify_interp_max_rel)
        recorder.add_assumption(
            "served run (WP8): inputs arrived over the control socket and the native data "
            f"plane at {server.endpoints}"
        )
        recorder.finish()
    return rc


def _typer_echo(msg: str, file: Any = None) -> None:
    typer.echo(msg, err=file is sys.stderr)


def _prefixed(d: dict[str, int]) -> dict[str, int]:
    return {f"transport.{k}": int(v) for k, v in d.items()}


def command(
    scenario: Annotated[Path, typer.Argument(help="Scenario file (.yaml, .json or .toml).")],
    serve: Annotated[
        str, typer.Option("--serve", "--control", help="Control ROUTER endpoint.")
    ] = "tcp://127.0.0.1:5600",
    pub: Annotated[
        str, typer.Option("--pub", help="Telemetry PUB endpoint, or 'none'.")
    ] = "tcp://127.0.0.1:5601",
    tx: Annotated[str, typer.Option("--tx", help="Native TX ingress PULL endpoint.")] = (
        "tcp://127.0.0.1:5602"
    ),
    allow_remote: Annotated[
        bool, typer.Option("--allow-remote", help="Allow non-loopback binds (no authentication).")
    ] = False,
    paused: Annotated[
        bool, typer.Option("--paused/--no-paused", help="Hold virtual time until sim.resume.")
    ] = True,
    exit_when_done: Annotated[
        bool, typer.Option("--exit-when-done", help="Stop once the run ended and RX drained.")
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", help="Record a run directory under this root.")
    ] = None,
    mode: Annotated[str | None, typer.Option("--mode", help="lockstep or paced.")] = None,
    backend: Annotated[
        str | None, typer.Option("--backend", help="numba_cpu or numpy_ref.")
    ] = None,
    duration_s: Annotated[
        str | None, typer.Option("--duration-s", help="Exact run duration (overrides).")
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Run name.")] = None,
    raw_peer: Annotated[
        list[str] | None,
        typer.Option("--raw-peer", help="Raw REQ/REP peer: key=value,... (repeatable)."),
    ] = None,
    unopened_tx: Annotated[
        str, typer.Option("--unopened-tx", help="wait (default) or silent.")
    ] = "wait",
    ack_timeout_s: Annotated[
        float | None, typer.Option("--ack-timeout-s", help="Remote barrier ack timeout (wall s).")
    ] = None,
) -> None:
    """Serve a scenario over the control socket and the native ZMQ data plane."""
    peers = tuple(parse_raw_peer(p) for p in (raw_peer or []))
    rc = run_served(
        scenario,
        control=serve,
        pub=None if pub.lower() == "none" else pub,
        tx=tx,
        allow_remote=allow_remote,
        paused=paused,
        exit_when_done=exit_when_done,
        out=out,
        mode=mode,
        backend=backend,
        duration_s=duration_s,
        name=name,
        raw_peers=peers,
        unopened_tx=unopened_tx,
        ack_timeout_s=ack_timeout_s,
        echo=_typer_echo,
    )
    if rc:
        raise typer.Exit(rc)


def main(argv: list[str] | None = None) -> int:
    """``python -m sremu.commands.serve``: deterministic numba settings first (D-4, D15),
    as the ``sremu`` CLI does, unless ``--no-deterministic`` is the first argument."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--no-deterministic":
        args = args[1:]
    else:
        from sremu import sysinfo

        os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
        os.environ["NUMBA_CACHE_DIR"] = sysinfo.nosvml_cache_dir(os.environ.get("NUMBA_CACHE_DIR"))
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
    app.command()(command)
    try:
        app(args=args, prog_name="sremu serve")  # standalone: usage errors exit with 2
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else (0 if code is None else 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
