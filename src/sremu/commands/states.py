"""``sremu states import|check``: response-table tooling (DESIGN.md sections 2.5, 5.5; WP2).

* ``sremu states import TABLE.csv --provenance SIDE.yaml [--out LIB.yaml]``: import a
  long-form CSV (``state,f_hz,theta_i_deg,theta_s_deg,mag_db,phase_deg[,sigma...]``) or,
  with ``--format matlab``, a MATLAB lossless S-curve CSV; run every validator and write a
  state-library YAML (section 9.4).
* ``sremu states check NAME_OR_FILE [--assume-mag label=mag ...]``: load a built-in name or
  a state-library/CSV file, run the validators and print a summary (JSON with ``--json``).

Exit code 0 on success, 1 when a validator fails.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from sremu.errors import SremuError
from sremu.physics.nonideal import binary_md, eta_alphabet
from sremu.physics.response import ResponseTable, builtin, builtin_names

app = typer.Typer(
    name="states",
    help="Response-table tooling: import a CSV into a state library, or check a table.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _parse_assume(items: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for it in items or []:
        key, sep, val = it.partition("=")
        if not sep:
            raise typer.BadParameter(f"--assume-mag expects label=mag, got {it!r}")
        out[key.strip()] = float(val)
    return out


def _load(target: str, assume: dict[str, float], provenance: Path | None) -> ResponseTable:
    p = Path(target)
    if p.suffix.lower() == ".csv":
        side = provenance or p.with_suffix(".yaml")
        return ResponseTable.from_csv(p, side)
    if p.suffix.lower() in (".yaml", ".yml", ".json") or p.exists():
        return ResponseTable.from_state_library(p, assume_mag=assume or None)
    return builtin(target, assume_mag=assume or None)


def _report(table: ResponseTable) -> dict[str, Any]:
    info = table.summary()
    if table.angle_independent:
        f0 = 0.0 if table.frequency_independent else float(table.freqs_hz[0])
        g = table.gamma_at(f0)
        info["eta_large_n_db"] = 10 * math.log10(eta_alphabet(g)) if abs(g).max() > 0 else None
        if g.size == 2:
            m, d = binary_md(complex(g[0]), complex(g[1]))
            info["md_db"] = {
                "m_db": 20 * math.log10(abs(m)) if abs(m) > 0 else None,
                "m_deg": math.degrees(math.atan2(m.imag, m.real)),
                "d_db": 20 * math.log10(abs(d)) if abs(d) > 0 else None,
            }
    return info


@app.command("check")
def check(
    target: Annotated[str, typer.Argument(help="Built-in name or state library / CSV path.")],
    assume_mag: Annotated[
        list[str] | None, typer.Option("--assume-mag", help="label=mag for mag: null.")
    ] = None,
    provenance: Annotated[
        Path | None, typer.Option("--provenance", help="Sidecar for a CSV table.")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
) -> None:
    """Load a table, run every validator and print a summary."""
    try:
        table = _load(target, _parse_assume(assume_mag), provenance)
    except (SremuError, OSError, ValueError, KeyError) as exc:
        typer.echo(f"FAIL {target}: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from None
    info = _report(table)
    if json_out:
        typer.echo(json.dumps(info, indent=2, default=str))
    else:
        typer.echo(f"OK {target}")
        for k, v in info.items():
            typer.echo(f"  {k}: {v}")


@app.command("import")
def import_(
    csv_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Table CSV.")],
    provenance: Annotated[
        Path | None, typer.Option("--provenance", help="Provenance sidecar (JSON/YAML).")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", help="State-library YAML to write.")] = None,
    fmt: Annotated[
        str, typer.Option("--format", help="'long' (section 5.5 CSV) or 'matlab'.")
    ] = "long",
    symmetrize: Annotated[
        str | None, typer.Option("--symmetrize", help="'mean' to accept and log.")
    ] = None,
    name: Annotated[str | None, typer.Option("--name", help="Table name.")] = None,
) -> None:
    """Import a CSV table, validate it and write a state-library YAML."""
    if symmetrize not in (None, "mean"):  # never ignore an unknown option silently
        typer.echo(f"FAIL {csv_path}: --symmetrize accepts only 'mean'", err=True)
        raise typer.Exit(1)
    try:
        if fmt == "matlab":
            table = ResponseTable.from_matlab_scurve(csv_path, name=name)
        elif fmt == "long":
            side = provenance or csv_path.with_suffix(".yaml")
            table = ResponseTable.from_csv(
                csv_path, side, symmetrize="mean" if symmetrize == "mean" else None, name=name
            )
        else:
            raise typer.BadParameter(f"unknown --format {fmt!r}")
    except (SremuError, OSError, ValueError, KeyError) as exc:
        typer.echo(f"FAIL {csv_path}: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(1) from None
    doc = table.to_state_library()
    target = out or csv_path.with_name(f"{table.name or csv_path.stem}.states.yaml")
    target.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    typer.echo(
        f"OK {csv_path} -> {target} ({len(table.labels)} states, shape {list(table.gamma.shape)})"
    )


@app.command("list")
def list_() -> None:
    """List the built-in tables."""
    for n in builtin_names():
        typer.echo(n)
