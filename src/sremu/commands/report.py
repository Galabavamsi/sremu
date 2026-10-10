"""``sremu report RUN_DIR -o r.html``: static HTML report (WP7; DESIGN.md 2.5, 8.6).

One self-contained file: inline CSS and SVG, no JavaScript, no external resources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sremu.record.report import write_report

__all__ = ["command"]


def command(
    run_dir: Annotated[Path, typer.Argument(help="Run directory.")],
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Output HTML file (default RUN_DIR/report.html)."),
    ] = None,
    max_points: Annotated[
        int, typer.Option("--max-points", help="Points per power plot.", min=2)
    ] = 800,
) -> None:
    """Write the static HTML report of a run directory."""
    if not run_dir.is_dir():
        typer.echo(f"sremu report: {run_dir} is not a directory", err=True)
        raise typer.Exit(2)
    path = write_report(run_dir, output, max_points=max_points)
    typer.echo(f"report written to {path}")
