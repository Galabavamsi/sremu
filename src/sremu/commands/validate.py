"""``sremu validate S.yaml`` (WP1; DESIGN.md sections 2.5, 9.5).

Loads the scenario (schema), resolves it (state libraries, computed defaults), runs the
guards and prints every error and warning plus the derived quantities: wavelength,
Fraunhofer distances per panel, maximum Doppler bound per link, the narrowband estimate and
FD row per link, a conservative T_curv bound, the noise floor per RX and ``tau_RIS->r`` in
samples. The automatic T_u and tile counts come from the engine's interpolation policy
(WP5) and are printed by it when available.

**Physics diagnostics** (section 2.5, "WP1 + physics diagnostics, WP2"): every resolved
panel state library is also loaded as a ``physics.response.ResponseTable``, which runs the
section 5.5 load-time validators (passivity, sigma-based reciprocity, grid density,
mandatory provenance). A failure is reported as error ``TABLE_VALIDATOR`` with the
validator's exception class. The scenario guards cannot do this themselves (layer 1 may not
import ``sremu.physics``, section 2.3).

Exit code 0: valid (warnings allowed); 1: guard errors; 2: schema or file errors.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Annotated, Any

import typer

from sremu.errors import SremuError
from sremu.scenario.guards import GuardReport, check_scenario
from sremu.scenario.load import (
    ResolvedScenario,
    parse_scenario,
    read_document,
    resolve_scenario,
)

__all__ = ["command", "physics_diagnostics"]


def physics_diagnostics(res: ResolvedScenario, rep: GuardReport) -> None:
    """Run the section 5.5 table validators of ``physics.response.ResponseTable`` on every
    resolved panel library; failures become ``TABLE_VALIDATOR`` errors in ``rep``."""
    from sremu.physics.response import ResponseTable

    for pid, lib in res.libraries.items():
        doc = lib.model_dump(mode="python")
        if doc.get("angle_mapping") is None:
            # A full (theta_i, theta_s) table: WP2 reads an absent key as "full" but rejects
            # an explicit null (reported to WP2 in the WP1 review).
            doc.pop("angle_mapping", None)
        try:
            ResponseTable.from_state_library(doc)
        except SremuError as exc:
            rep.error(
                "TABLE_VALIDATOR",
                f"ris[{pid}].states",
                f"{type(exc).__name__}: {exc} (section 5.5)",
                type(exc),
            )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isinf(v):
            return "inf"
        return f"{v:.6g}"
    return str(v)


def command(
    scenario: Annotated[Path, typer.Argument(help="Scenario file (.yaml, .json or .toml).")],
    json_out: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Validate a scenario: schema, guards and derived quantities."""
    try:
        sc = parse_scenario(read_document(scenario), label=str(scenario))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = resolve_scenario(sc, scenario.resolve().parent, warn=False, strict=False)
    except (SremuError, OSError) as exc:
        if json_out:
            typer.echo(json.dumps({"valid": False, "schema_error": str(exc)}, indent=2))
        else:
            typer.echo(f"INVALID {scenario}\n{exc}", err=True)
        raise typer.Exit(2) from None
    rep = check_scenario(res)
    physics_diagnostics(res, rep)
    if json_out:
        out = {
            "valid": not rep.errors,
            "scenario": sc.name,
            "scenario_sha256": res.sha256(),
            "errors": [vars(i) | {"exc": i.exc.__name__} for i in rep.errors],
            "warnings": [vars(i) | {"exc": i.exc.__name__} for i in rep.warnings],
            "assumptions": list(res.assumptions),
            "inputs": [r.to_dict() for r in res.inputs],
            "derived": rep.derived,
        }
        typer.echo(json.dumps(out, indent=2, default=str))
    else:
        status = "VALID" if not rep.errors else "INVALID"
        typer.echo(f"{status} {scenario}  (name {sc.name}, sha256 {res.sha256()[:16]}...)")
        for i in rep.errors:
            typer.echo(f"  error   {i}")
        for i in rep.warnings:
            typer.echo(f"  warning {i}")
        for a in res.assumptions:
            typer.echo(f"  assumption: {a}")
        d = rep.derived
        for sid, row in d.get("streams", {}).items():
            extra = f", page {row['page_size']}" if "page_size" in row else ""
            typer.echo(f"  stream {sid}: lambda {_fmt(row['lambda_m'])} m{extra}")
        for pid, row in d.get("panels", {}).items():
            ff = ", ".join(f"{_fmt(v)} m at {k} Hz" for k, v in row["fraunhofer_m"].items())
            typer.echo(
                f"  panel {pid}: {row['n_cells']} cells, {row['n_controls']} controls, "
                f"size {_fmt(row['size_m'][0])} x {_fmt(row['size_m'][1])} m, "
                f"Fraunhofer 2D^2/lambda {ff or 'n/a'}"
            )
        for lid, row in d.get("links", {}).items():
            nb = row.get("narrowband")
            nb_txt = "n/a" if nb is None else ("narrowband" if nb else "wideband")
            fd = row["fd"]
            fd_txt = f"{fd['taps']} taps" if fd["taps"] else "rejected (> 0.8 f_s)"
            parts = [
                f"f_D,max <= {_fmt(row['fd_max_hz_bound'])} Hz"
                if row["fd_max_hz_bound"] is not None
                else "f_D,max unknown (external motion)",
                f"{nb_txt} (pi B dtau = {_fmt(row.get('narrowband_metric_rad', float('nan')))} "
                "rad, t = 0 estimate)",
                f"FD {fd_txt}",
            ]
            if "t_curv_s_bound" in row:
                parts.append(f"T_curv >= {_fmt(row['t_curv_s_bound'])} s")
            typer.echo(f"  link {lid}: " + "; ".join(parts))
        for sid, row in d.get("receivers", {}).items():
            tau = ", ".join(
                f"{p}: {_fmt(v)} samples" for p, v in row.get("tau_ris_samples", {}).items()
            )
            typer.echo(
                f"  rx {sid}: noise floor {_fmt(row['noise_floor_dbm'])} dBm"
                + (f"; tau_RIS->r {tau}" if tau else "")
            )
        typer.echo(
            "  auto T_u and tiles: decided by the engine's interpolation policy (channel/interp)"
        )
    if rep.errors:
        raise typer.Exit(1)
