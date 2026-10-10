"""``sremu`` command line (typer) with a lazy command registry (DESIGN.md section 2.5).

Each command is registered by name in :data:`REGISTRY` as ``"module.path:attribute"``.
The CLI registers a thin typer command per entry; the target module is imported only when
that command runs, so ``sremu --help`` and ``sremu info`` stay fast, and a command whose
module does not exist yet prints a clear "not yet implemented" message and exits with
code 3 (:data:`NOT_IMPLEMENTED_EXIT`) instead of crashing on import.

**How a work package adds its command.** Create the module named in :data:`REGISTRY`
(``src/sremu/commands/<name>.py``) and define the attribute as either

* a plain function: typer turns its signature into arguments and options, or
* a ``typer.Typer`` app: for commands with subcommands (``sremu states import|check``).

All arguments after the command name (including ``--help``) are forwarded to it.
``cli.py`` itself changes only through a CCR (section 14.3).

**Determinism (D-4, D15).** Unless ``--no-deterministic`` is given, the CLI sets
``NUMBA_DISABLE_INTEL_SVML=1`` (overriding any pre-set value) and
``NUMBA_CACHE_DIR=<default cache>-nosvml`` (:func:`sremu.sysinfo.nosvml_cache_dir`; the
default cache is a pre-set ``NUMBA_CACHE_DIR``, else numba's user-wide cache directory)
before any command module, and so numba, is imported.

Platform probes live in the layer-0 module :mod:`sremu.sysinfo` (D12), shared with the run
manifest.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import typer

from sremu import sysinfo

__all__ = [
    "NOT_IMPLEMENTED_EXIT",
    "REGISTRY",
    "CommandEntry",
    "app",
    "build_app",
    "collect_info",
    "load_command",
]

NOT_IMPLEMENTED_EXIT = 3
"""Exit code of a registered command whose module does not exist yet."""


@dataclass(frozen=True)
class CommandEntry:
    """A lazily imported command: ``target`` is ``"module.path:attribute"``."""

    target: str
    owner: str
    help: str


REGISTRY: dict[str, CommandEntry] = {
    "validate": CommandEntry(
        "sremu.commands.validate:command",
        "WP1",
        "Validate a scenario: schema, guards and derived quantities.",
    ),
    "run": CommandEntry(
        "sremu.commands.run:command",
        "WP6/WP8",
        "Run a scenario headless (--out DIR) or served (--serve tcp://127.0.0.1:5600).",
    ),
    "batch": CommandEntry(
        "sremu.commands.batch:command", "WP6", "Coefficient-only (batch) evaluation of a scenario."
    ),
    "sweep": CommandEntry(
        "sremu.commands.sweep:command", "WP9", "Parameter sweep: one run directory per point."
    ),
    "replay": CommandEntry(
        "sremu.commands.replay:command", "WP7", "Bitwise replay of a recorded run directory."
    ),
    "report": CommandEntry(
        "sremu.commands.report:command", "WP7", "Static HTML report of a run directory."
    ),
    "verify": CommandEntry(
        "sremu.commands.verify:command", "WP4", "Run the oracle subset; print model-card numbers."
    ),
    "states": CommandEntry(
        "sremu.commands.states:app", "WP2", "Response-table tooling (states import / check)."
    ),
    "schema": CommandEntry(
        "sremu.commands.schema:command",
        "WP1/WP8",
        "Export JSON Schema: scenario, states, manifest or control.",
    ),
    "bench": CommandEntry(
        "sremu.commands.bench:command", "WP11", "Benchmarks (b1..b6) with platform capture."
    ),
    "info": CommandEntry(
        "sremu.cli:info",
        "WP0",
        "Versions, CPU, GPU, power plan, numba threading layer and SVML state.",
    ),
}
"""Command name -> lazily imported implementation (section 2.5)."""


# --------------------------------------------------------------------------- lazy loading


def _apply_determinism(deterministic: bool) -> None:
    """Set the numba environment of deterministic mode (D-4, D15); no-op otherwise.

    SVML is always disabled (a pre-set ``NUMBA_DISABLE_INTEL_SVML=0`` is overridden; use
    ``--no-deterministic`` to keep SVML), and ``NUMBA_CACHE_DIR`` becomes
    ``<default cache>-nosvml`` so kernels cached with SVML are never loaded.
    """
    if deterministic:
        os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
        os.environ["NUMBA_CACHE_DIR"] = sysinfo.nosvml_cache_dir(os.environ.get("NUMBA_CACHE_DIR"))


def load_command(name: str, entry: CommandEntry | None = None) -> Any | None:
    """Import the command registered as ``name``; None when its module does not exist yet.

    ``entry`` defaults to ``REGISTRY[name]``. Returns a click-compatible command object (it
    has ``.main(args=..., prog_name=..., standalone_mode=...)``). An ``ImportError`` raised
    *inside* an existing command module propagates unchanged.
    """
    entry = REGISTRY[name] if entry is None else entry
    module_name, _, attr = entry.target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if module_name == missing or module_name.startswith(missing + "."):
            return None
        raise
    obj = getattr(module, attr, None)
    if obj is None:
        return None
    if isinstance(obj, typer.Typer):
        return typer.main.get_command(obj)
    if callable(getattr(obj, "main", None)) and hasattr(obj, "params"):
        return obj  # already a click-style command object
    if callable(obj):
        sub = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
        sub.command(name=name, help=entry.help)(obj)
        return typer.main.get_command(sub)
    raise TypeError(f"{entry.target} is neither a function nor a typer.Typer")


def _lazy_command(name: str, entry: CommandEntry) -> Callable[[typer.Context], None]:
    def _invoke(ctx: typer.Context) -> None:
        cmd = load_command(name, entry)
        if cmd is None:
            module = entry.target.partition(":")[0]
            typer.echo(
                f"sremu {name}: not yet implemented (owner {entry.owner}; expected "
                f"'{entry.target.partition(':')[2]}' in src/{module.replace('.', '/')}.py)",
                err=True,
            )
            raise typer.Exit(NOT_IMPLEMENTED_EXIT)
        rc = cmd.main(args=list(ctx.args), prog_name=f"sremu {name}", standalone_mode=False)
        if isinstance(rc, int) and not isinstance(rc, bool) and rc != 0:
            raise typer.Exit(rc)

    _invoke.__name__ = f"lazy_{name}"
    _invoke.__doc__ = entry.help
    return _invoke


def _version_callback(value: bool) -> None:
    if value:
        from sremu import __version__

        typer.echo(f"sremu {__version__}")
        raise typer.Exit()


def main(
    version_: bool = typer.Option(
        False,
        "--version",
        help="Print the sremu version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    deterministic: bool = typer.Option(
        True,
        "--deterministic/--no-deterministic",
        help=(
            "Force NUMBA_DISABLE_INTEL_SVML=1 and a separate '-nosvml' NUMBA_CACHE_DIR "
            "before numba is imported (D-4, D15)."
        ),
    ),
) -> None:
    """sremu command line."""
    _apply_determinism(deterministic)


# --------------------------------------------------------------------------- info


def collect_info(probe_threads: bool = False) -> dict[str, Any]:
    """Platform and dependency information (probes in :mod:`sremu.sysinfo`, D12)."""
    from sremu import __version__
    from sremu.contracts import CONTRACT_VERSION

    deps = {d: sysinfo.dist_version(d) for d in sysinfo.INFO_DISTRIBUTIONS}
    try:
        import zmq

        deps["libzmq"] = zmq.zmq_version()
    except ImportError:
        deps["libzmq"] = None
    return {
        "sremu_version": __version__,
        "contract_version": CONTRACT_VERSION,
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "python_executable": sys.executable,
        **sysinfo.platform_info(),
        "versions": deps,
        "numba": sysinfo.numba_state(probe_threads),
    }


def _flatten(d: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            rows.extend(_flatten(v, key + "."))
        else:
            rows.append((key, v))
    return rows


def info(
    json_out: bool = typer.Option(False, "--json", help="Print the information as JSON."),
    probe_threads: bool = typer.Option(
        False,
        "--probe-threads",
        help="Compile a tiny parallel kernel to report the active numba threading layer.",
    ),
) -> None:
    """Print versions, CPU, GPU, power plan, numba threading layer and SVML state."""
    data = collect_info(probe_threads)
    if json_out:
        typer.echo(json.dumps(data, indent=2, sort_keys=False))
        return
    rows = _flatten(data)
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v) if v else "none"
        typer.echo(f"{k.ljust(width)}  {v if v is not None else 'n/a'}")


def build_app(registry: Mapping[str, CommandEntry] | None = None) -> typer.Typer:
    """Build the typer app: one command per registry entry (lazy unless in this module)."""
    reg = REGISTRY if registry is None else registry
    application = typer.Typer(
        name="sremu",
        help="sremu: Smart Radio Environment EMUlator (IQ-level RIS radio-environment emulator).",
        no_args_is_help=True,
        add_completion=False,
        pretty_exceptions_enable=False,
    )
    application.callback()(main)
    for name, entry in reg.items():
        module_name, _, attr = entry.target.partition(":")
        if module_name == __name__:
            application.command(name=name, help=entry.help)(globals()[attr])
            continue
        application.command(
            name=name,
            help=f"{entry.help} [owner {entry.owner}]",
            context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
            add_help_option=False,
        )(_lazy_command(name, entry))
    return application


app = build_app()
"""The ``sremu`` console-script entry point."""


if __name__ == "__main__":  # pragma: no cover
    app()
