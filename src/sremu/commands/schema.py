"""``sremu schema {scenario,states,manifest,control}`` (WP1 / WP8; DESIGN.md section 2.5).

Prints a JSON Schema (draft 2020-12) to stdout, or writes it with ``--out``. ``control``
prints the control-plane message schemas shipped by WP8 in ``sremu/transport/schema/``
(as one object keyed by file name) once they exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

__all__ = ["command"]

_KINDS = ("scenario", "states", "manifest", "control")


def _control() -> dict[str, Any] | None:
    import sremu

    d = Path(sremu.__file__).resolve().parent / "transport" / "schema"
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    if not files:
        return None
    return {f.name: json.loads(f.read_text(encoding="utf-8")) for f in files}


def command(
    kind: Annotated[str, typer.Argument(help="One of: scenario, states, manifest, control.")],
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write to this file.")] = None,
) -> None:
    """Export a JSON Schema: scenario, states, manifest or control."""
    from sremu.scenario.schema_export import (
        scenario_json_schema,
        schema_text,
        state_library_json_schema,
    )

    if kind not in _KINDS:
        typer.echo(f"unknown schema {kind!r}; choose one of {', '.join(_KINDS)}", err=True)
        raise typer.Exit(2)
    if kind == "scenario":
        schema: dict[str, Any] | None = scenario_json_schema()
    elif kind == "states":
        schema = state_library_json_schema()
    elif kind == "manifest":
        from sremu.record.manifest import manifest_json_schema

        schema = manifest_json_schema()
    else:
        schema = _control()
        if schema is None:
            typer.echo(
                "sremu schema control: not yet implemented (owner WP8; expected JSON files "
                "in src/sremu/transport/schema/)",
                err=True,
            )
            raise typer.Exit(3)
    assert schema is not None
    text = schema_text(schema)
    if out is None:
        typer.echo(text, nl=False)
    else:
        out.write_text(text, encoding="utf-8", newline="\n")
        typer.echo(f"wrote {out}")
