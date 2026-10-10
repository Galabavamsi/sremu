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

**Determinism (D-4).** Unless ``--no-deterministic`` is given, the CLI sets
``NUMBA_DISABLE_INTEL_SVML=1`` (overriding any pre-set value) before any command module,
and so numba, is imported.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import typer

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
    # D-4: deterministic mode always runs with SVML disabled. A pre-set
    # NUMBA_DISABLE_INTEL_SVML=0 is overridden; use --no-deterministic to keep SVML.
    if deterministic:
        os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"


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
        help="Force NUMBA_DISABLE_INTEL_SVML=1 before numba is imported (D-4).",
    ),
) -> None:
    """sremu command line."""
    _apply_determinism(deterministic)


# --------------------------------------------------------------------------- info


def _run(args: list[str], timeout_s: float = 10.0) -> str | None:
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _dist_version(dist: str) -> str | None:
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _cpu_model() -> str | None:
    if sys.platform == "win32":
        import winreg

        try:
            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or None


def _power_plan_or_governor() -> str | None:
    if sys.platform == "win32":
        out = _run(["powercfg", "/getactivescheme"])
        if out:
            m = re.search(r"\(([^)]*)\)\s*$", out)
            return m.group(1) if m else out
        return None
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return gov.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _gpus() -> list[str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    out = _run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return [line.strip() for line in out.splitlines() if line.strip()] if out else []


def _aws_instance() -> str | None:
    base = Path("/sys/devices/virtual/dmi/id")
    try:
        if "Amazon EC2" in (base / "sys_vendor").read_text(encoding="utf-8"):
            return (base / "board_asset_tag").read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        pass
    return None


def _numba_state(probe_threads: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "env_NUMBA_DISABLE_INTEL_SVML": os.environ.get("NUMBA_DISABLE_INTEL_SVML")
    }
    try:
        import numba
    except ImportError as exc:
        state["error"] = f"numba not importable: {exc}"
        return state
    cfg = numba.config
    state["version"] = numba.__version__
    state["using_svml"] = bool(getattr(cfg, "USING_SVML", False))
    state["disable_intel_svml"] = bool(getattr(cfg, "DISABLE_INTEL_SVML", False))
    state["threading_layer_config"] = str(getattr(cfg, "THREADING_LAYER", "default"))
    state["num_threads"] = int(getattr(cfg, "NUMBA_NUM_THREADS", 0))
    available = []
    for layer, mod in (("tbb", "tbbpool"), ("omp", "omppool"), ("workqueue", "workqueue")):
        try:
            importlib.import_module(f"numba.np.ufunc.{mod}")
            available.append(layer)
        except Exception:  # any failure means "not available"
            pass
    state["threading_layers_available"] = available
    if probe_threads:
        try:

            @numba.njit(parallel=True, cache=False)
            def _probe(n: int) -> float:
                s = 0.0
                for i in numba.prange(n):
                    s += i
                return s

            _probe(64)
            state["threading_layer_active"] = numba.threading_layer()
        except Exception as exc:
            state["threading_layer_active"] = f"probe failed: {exc}"
    return state


def collect_info(probe_threads: bool = False) -> dict[str, Any]:
    """Platform and dependency information (also useful for manifests and benchmarks)."""
    from sremu import __version__
    from sremu.contracts import CONTRACT_VERSION

    deps = {
        d: _dist_version(d)
        for d in (
            "numpy",
            "scipy",
            "numba",
            "llvmlite",
            "pydantic",
            "pyzmq",
            "typer",
            "PyYAML",
            "matplotlib",
            "torch",
        )
    }
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
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "power_plan_or_governor": _power_plan_or_governor(),
        "gpu": _gpus(),
        "aws_instance": _aws_instance(),
        "versions": deps,
        "numba": _numba_state(probe_threads),
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
