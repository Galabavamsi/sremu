"""``sremu bench --suite {b1,b2,b3,b4,b6,eb3} [suite args...]`` (WP11; DESIGN.md sections 2.5, 12).

Runs a benchmark suite of the repository's ``bench/`` directory (``python -m
bench.<module>`` with the current interpreter, working directory = the repository root, in
the deterministic numba environment the CLI has already set). Extra arguments are passed to
the suite unchanged, e.g. ``sremu bench --suite b2 --quick --out runs/bench``. The suites
exist only in a source checkout; an installed wheel has no ``bench/`` (exit code 3).

| Suite | Module | What |
|---|---|---|
| b1 | ``bench.b1_kernels`` | kernel micro-benchmarks and the section 7.4 cost-model fit |
| b2 | ``bench.b2_lockstep`` | lockstep RTF sweep (slices; server plan with pinned workers) |
| b3 | ``bench.b3_paced`` | paced sustainable links and latency CDFs (Linux numbers) |
| b4 | ``bench.b4_control`` | control loop: lockstep (lateness 0) and paced async |
| b6 | ``bench.b6_crosstool`` | cross-tool context (MATLAB per-element script, RISReflector) |
| eb3 | ``bench.eb3_paced`` | E-B3 timing demonstration (external TDMA controller) |

There is no B5: the benchmark set is B1-B4 and B6 (DESIGN.md D24).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

__all__ = ["SUITES", "app", "bench_dir", "command", "run_suite"]

SUITES: dict[str, str] = {
    "b1": "bench.b1_kernels",
    "b2": "bench.b2_lockstep",
    "b3": "bench.b3_paced",
    "b4": "bench.b4_control",
    "b6": "bench.b6_crosstool",
    "eb3": "bench.eb3_paced",
}
"""Suite name -> module under the repository's ``bench/`` package."""


def bench_dir() -> Path | None:
    """The repository root holding ``bench/`` (source checkout), or None."""
    root = Path(__file__).resolve().parents[3]
    return root if (root / "bench" / "__init__.py").exists() else None


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Benchmarks (b1-b4, b6, eb3) with platform capture; extra arguments go to the suite.",
)
def run_suite(
    ctx: typer.Context,
    suite: str = typer.Option(..., "--suite", help="b1 | b2 | b3 | b4 | b6 | eb3"),
) -> None:
    """Run one benchmark suite."""
    if suite not in SUITES:
        typer.echo(f"sremu bench: unknown suite {suite!r} ({', '.join(SUITES)})", err=True)
        raise typer.Exit(2)
    root = bench_dir()
    if root is None:
        typer.echo("sremu bench: the bench/ suites need a source checkout of sremu", err=True)
        raise typer.Exit(3)
    rc = subprocess.call([sys.executable, "-m", SUITES[suite], *ctx.args], cwd=root)
    if rc:
        raise typer.Exit(rc)


command = typer.main.get_command(app)
"""The click command registered as ``sremu bench`` (passes unknown options to the suite)."""
