"""CLI smoke tests (section 2.5): lazy registry, ``info``, determinism switch, console script."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from sremu import cli
from sremu.contracts import CONTRACT_VERSION

pytestmark = pytest.mark.unit

SPEC_COMMANDS = {
    "validate",
    "run",
    "batch",
    "sweep",
    "replay",
    "report",
    "verify",
    "states",
    "schema",
    "bench",
    "info",
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_cli_registry_lists_every_section_2_5_command():
    assert set(cli.REGISTRY) == SPEC_COMMANDS
    for name, entry in cli.REGISTRY.items():
        module, sep, attr = entry.target.partition(":")
        assert sep and module and attr, name
        assert module == "sremu.cli" or module == f"sremu.commands.{name}", name


def test_cli_help_lists_commands_without_importing_them(runner):
    before = {m for m in sys.modules if m.startswith("sremu.commands.")}
    res = runner.invoke(cli.app, ["--help"])
    assert res.exit_code == 0, res.output
    for name in SPEC_COMMANDS:
        assert name in res.output
    after = {m for m in sys.modules if m.startswith("sremu.commands.")}
    assert after == before


def test_cli_version(runner):
    from sremu import __version__

    res = runner.invoke(cli.app, ["--version"])
    assert res.exit_code == 0
    assert res.output.strip() == f"sremu {__version__}"


def test_cli_info_text(runner):
    res = runner.invoke(cli.app, ["info"])
    assert res.exit_code == 0, res.output
    lines = dict(
        line.split(None, 1) for line in res.output.splitlines() if len(line.split(None, 1)) == 2
    )
    assert lines["contract_version"] == CONTRACT_VERSION
    assert lines["versions.numpy"] == np.__version__
    assert lines["numba.using_svml"] in ("True", "False")
    assert "numba.threading_layers_available" in lines
    assert lines["logical_cpus"] == str(os.cpu_count())


def test_cli_info_json(runner):
    res = runner.invoke(cli.app, ["info", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["contract_version"] == CONTRACT_VERSION
    assert data["versions"]["numpy"] == np.__version__
    for key in ("cpu_model", "logical_cpus", "power_plan_or_governor", "gpu", "platform"):
        assert key in data
    assert isinstance(data["gpu"], list)
    assert {"using_svml", "threading_layer_config", "num_threads"} <= set(data["numba"])


def test_cli_deterministic_switch_sets_svml_env(runner, monkeypatch):
    monkeypatch.delenv("NUMBA_DISABLE_INTEL_SVML", raising=False)
    res = runner.invoke(cli.app, ["--no-deterministic", "info", "--json"])
    assert res.exit_code == 0
    assert "NUMBA_DISABLE_INTEL_SVML" not in os.environ
    res = runner.invoke(cli.app, ["info", "--json"])
    assert res.exit_code == 0
    assert os.environ["NUMBA_DISABLE_INTEL_SVML"] == "1"
    # a pre-set "0" does not survive deterministic mode (D-4)
    monkeypatch.setenv("NUMBA_DISABLE_INTEL_SVML", "0")
    res = runner.invoke(cli.app, ["info", "--json"])
    assert res.exit_code == 0
    assert os.environ["NUMBA_DISABLE_INTEL_SVML"] == "1"


def _info_in_fresh_process(args: list[str], svml_env: str | None) -> dict:
    """Run ``sremu <args> info --json`` in a new interpreter (numba not yet imported)."""
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_DISABLE_INTEL_SVML"}
    if svml_env is not None:
        env["NUMBA_DISABLE_INTEL_SVML"] = svml_env
    res = subprocess.run(
        [sys.executable, "-c", "from sremu.cli import app; app()", *args, "info", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)["numba"]


def test_cli_d4_deterministic_mode_disables_svml_before_numba_import():
    """D-4: numba reads NUMBA_DISABLE_INTEL_SVML once, at import. ``numba.config`` in a
    fresh process therefore shows whether the CLI set it *before* numba was imported, and
    deterministic mode must win even over a pre-set ``0``."""
    for pre in (None, "0"):
        state = _info_in_fresh_process([], pre)
        assert state["env_NUMBA_DISABLE_INTEL_SVML"] == "1", pre
        assert state["disable_intel_svml"] is True, pre
        assert state["using_svml"] is False, pre
    state = _info_in_fresh_process(["--no-deterministic"], None)
    assert state["env_NUMBA_DISABLE_INTEL_SVML"] is None
    assert state["disable_intel_svml"] is False


def test_cli_import_does_not_import_numba_or_commands():
    """``import sremu.cli`` must stay light: numba only after the D-4 switch."""
    code = (
        "import sys, sremu.cli; "
        "bad = [m for m in sys.modules if m == 'numba' or m.startswith('sremu.commands.')]; "
        "print(','.join(bad))"
    )
    res = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == ""


def test_cli_not_yet_implemented_exits_3(runner):
    reg = {"ghost": cli.CommandEntry("sremu.commands.ghost_cmd_xyz:command", "WP99", "Ghost.")}
    app = cli.build_app(reg)
    res = runner.invoke(app, ["ghost", "scenario.yaml", "--out", "x"])
    assert res.exit_code == cli.NOT_IMPLEMENTED_EXIT
    assert "not yet implemented" in res.output
    assert "WP99" in res.output
    assert cli.load_command("ghost", reg["ghost"]) is None


def _write_module(tmp_path: Path, name: str, body: str) -> None:
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")


def test_cli_lazy_function_command_forwards_arguments(runner, tmp_path, monkeypatch):
    _write_module(
        tmp_path,
        "sremu_fake_cmd_fn",
        """
        import typer

        def command(n: int, flag: bool = typer.Option(False, "--flag")) -> None:
            typer.echo(f"n={n} flag={flag}")
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    reg = {"fake": cli.CommandEntry("sremu_fake_cmd_fn:command", "WP0", "Fake.")}
    app = cli.build_app(reg)
    assert "sremu_fake_cmd_fn" not in sys.modules  # building the app imports nothing
    res = runner.invoke(app, ["fake", "5", "--flag"])
    assert res.exit_code == 0, res.output
    assert "n=5 flag=True" in res.output
    res = runner.invoke(app, ["fake", "--help"])
    assert res.exit_code == 0
    assert "--flag" in res.output
    res = runner.invoke(app, ["fake", "not-an-int"])
    assert res.exit_code == 2  # usage error from the real command


def test_cli_lazy_typer_subcommands(runner, tmp_path, monkeypatch):
    _write_module(
        tmp_path,
        "sremu_fake_cmd_app",
        """
        import typer

        app = typer.Typer()

        @app.command("check")
        def check(path: str) -> None:
            typer.echo(f"check {path}")

        @app.command("import")
        def import_(path: str) -> None:
            typer.echo(f"import {path}")
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    app = cli.build_app({"states": cli.CommandEntry("sremu_fake_cmd_app:app", "WP2", "States.")})
    res = runner.invoke(app, ["states", "import", "x.csv"])
    assert res.exit_code == 0, res.output
    assert "import x.csv" in res.output


def test_cli_import_error_inside_command_module_propagates(runner, tmp_path, monkeypatch):
    _write_module(tmp_path, "sremu_fake_cmd_broken", "import sremu_no_such_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    app = cli.build_app({"broken": cli.CommandEntry("sremu_fake_cmd_broken:command", "WP0", "B.")})
    res = runner.invoke(app, ["broken"])
    assert res.exit_code != 0
    assert isinstance(res.exception, ModuleNotFoundError)
    assert res.exception.name == "sremu_no_such_dependency_xyz"


def test_cli_console_script_info_runs():
    exe = shutil.which("sremu", path=str(Path(sys.executable).parent))
    if exe is None:
        pytest.skip("sremu console script not installed in this environment")
    res = subprocess.run(
        [exe, "info", "--json"], capture_output=True, text=True, timeout=120, check=False
    )
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["contract_version"] == CONTRACT_VERSION
