"""``sremu sweep SPEC.yaml --jobs N`` (WP9; DESIGN.md section 2.5).

Runs every point of a sweep spec (:mod:`sremu.experiments.sweep`) in its own run directory
under ``--out`` (default ``runs/``), plus ``sweep.json``, ``index.json``, ``index.csv`` and,
when pyarrow is installed, ``results.parquet``.

Exit code 0: every point succeeded; 1: at least one point failed (recorded in the index);
2: the spec is invalid or unreadable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from sremu.errors import SremuError

__all__ = ["command"]


def command(
    spec: Annotated[Path, typer.Argument(help="Sweep spec (.yaml, .json or .toml).")],
    jobs: Annotated[int, typer.Option("--jobs", "-j", min=1, help="Worker processes.")] = 1,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output root; the sweep goes to OUT/<name>/.")
    ] = Path("runs"),
    resume: Annotated[
        bool, typer.Option("--resume", help="Skip points already completed with the same params.")
    ] = False,
    allow_import: Annotated[
        bool,
        typer.Option(
            "--allow-import",
            help="Allow 'module:function' runners outside the sremu package.",
        ),
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No per-point lines.")] = False,
) -> None:
    """Parameter sweep: one run directory per point, plus an index."""
    from sremu.experiments.sweep import load_spec, run_sweep

    try:
        sp = load_spec(spec)
    except (OSError, SremuError, ValueError) as exc:
        typer.echo(f"sremu sweep: invalid spec {spec}: {exc}", err=True)
        raise typer.Exit(2) from None

    def progress(rec: Any) -> None:
        if quiet:
            return
        tag = " (resumed)" if rec.get("resumed") else ""
        err = f": {rec.get('error')}" if rec["status"] != "ok" else ""
        typer.echo(f"{rec['point_id']} seed={rec['seed']} {rec['status']}{tag}{err}")

    try:
        res = run_sweep(
            sp, out, jobs=jobs, resume=resume, allow_import=allow_import, progress=progress
        )
    except SremuError as exc:
        typer.echo(f"sremu sweep: {exc}", err=True)
        raise typer.Exit(2) from None
    typer.echo(
        f"sweep {sp.name}: {res.n_ok} ok, {res.n_failed} failed, {res.n_resumed} resumed; "
        f"index {res.root / 'index.json'}" + (" (+ results.parquet)" if res.parquet else "")
    )
    if not res.ok:
        raise typer.Exit(1)
