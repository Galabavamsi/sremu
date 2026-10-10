"""``sremu replay RUN_DIR``: bitwise replay of a recorded run (WP7; DESIGN.md 4.12, RT-16).

Exit codes: 0 when every recorded RX stream is reproduced bitwise and every command lands
at its recorded samples; 1 on any mismatch; 2 when the run cannot be replayed (any sremu
error, e.g. a missing recording or a refused stream); 3 when the engine is not available
yet (WP6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from sremu.errors import SremuError
from sremu.record.replay import load_engine_factory, replay

__all__ = ["command"]


def command(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to replay.")],
    engine: Annotated[
        str | None,
        typer.Option(
            "--engine",
            help="Engine factory 'package.module:function' (default: the WP6 engine built "
            "from scenario.resolved.yaml).",
        ),
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Also record the replayed run here.")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the result as JSON.")] = False,
) -> None:
    """Re-inject recorded TX and events.jsonl commands; compare RX bitwise."""
    try:
        factory = load_engine_factory(engine) if engine else None
        res = replay(run_dir, engine_factory=factory, out=out)
    except NotImplementedError as exc:
        typer.echo(f"sremu replay: {exc}", err=True)
        raise typer.Exit(3) from None
    except (SremuError, OSError, KeyError, ValueError) as exc:  # incl. ReplayError
        typer.echo(f"sremu replay: cannot replay {run_dir}: {exc}", err=True)
        raise typer.Exit(2) from None
    if json_out:
        typer.echo(json.dumps(res.to_dict(), indent=2))
    else:
        for name, s in res.streams.items():
            state = "bitwise equal" if s.bitwise_equal else "MISMATCH"
            typer.echo(
                f"{name}: {s.n_read}/{s.n_samples} samples, {s.n_mismatch} differ "
                f"(max |diff| {s.max_abs_diff:.3g}): {state}"
            )
        typer.echo(f"commands re-injected: {res.n_commands}; skipped: {len(res.skipped)}")
        for m in res.command_mismatches:
            typer.echo(f"command mismatch: {m}")
        typer.echo("replay: OK" if res.ok else "replay: MISMATCH")
    if not res.ok:
        raise typer.Exit(1)
