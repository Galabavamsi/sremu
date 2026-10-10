"""``sremu batch S.yaml`` (WP6; DESIGN.md sections 2.5, 4.11): the coefficient-only path.

Evaluates ``combine(snapshot(t), cfg(t))`` on the section 4.4 snapshot grid (or ``--dt-s``)
with the scenario's model, controllers and RNG keys (:func:`sremu.engine.batch.run_batch`)
and prints the time-averaged ``|h|^2`` per link and the mean RX power per receiver for unit
digital TX power (``P_TX`` from ``full_scale_dbm``/``gain_db``). ``--rate`` weights pieces
by RX sample counts (R4). ``--out FILE`` writes the per-piece table as JSON.

Scenario controllers of kind ``builtin`` need WP9 (``sremu.control.builtin``); without it
the command exits with code 3 unless ``--no-controllers`` is given. Exit code 2: scenario
error.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Annotated, Any

import typer

from sremu.errors import SremuError

__all__ = ["command"]


def command(
    scenario: Annotated[Path, typer.Argument(help="Scenario file (.yaml, .json or .toml).")],
    duration_s: Annotated[
        str | None, typer.Option("--duration-s", help="Exact duration (default: run.duration_s).")
    ] = None,
    dt_s: Annotated[
        str | None, typer.Option("--dt-s", help="Fixed exact grid step (default: auto T_u).")
    ] = None,
    rate: Annotated[
        str | None, typer.Option("--rate", help="Weight pieces by RX samples at this exact rate.")
    ] = None,
    no_controllers: Annotated[
        bool, typer.Option("--no-controllers", help="Ignore scenario controllers.")
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", help="Write the per-piece table as JSON.")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Print the summary as JSON.")] = False,
) -> None:
    """Coefficient-only (batch) evaluation of a scenario."""
    from sremu.engine.batch import run_batch
    from sremu.engine.streams import stream_configs_from_scenario
    from sremu.physics.propagation import from_scenario
    from sremu.scenario.load import load_scenario
    from sremu.timebase import parse_exact

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = load_scenario(scenario)
            d = res.scenario.model_dump(mode="python")
            model = from_scenario(d, base_dir=res.base_dir)
    except (SremuError, OSError) as exc:
        typer.echo(f"INVALID {scenario}\n{exc}", err=True)
        raise typer.Exit(2) from None
    ctrls: list[Any] = []
    if not no_controllers and d.get("controllers"):
        from sremu.commands.run import controllers_from_scenario

        try:
            ctrls = controllers_from_scenario(d, model)
        except ModuleNotFoundError:
            typer.echo(
                "scenario controllers need WP9 (sremu.control.builtin); use --no-controllers",
                err=True,
            )
            raise typer.Exit(3) from None
    dur = parse_exact(duration_s) if duration_s is not None else d["run"]["duration_s"]
    result = run_batch(
        model,
        dur,
        dt_s=dt_s,
        rate=rate,
        streams=stream_configs_from_scenario(d),
        controllers=ctrls,
        seed=int(d.get("seed", 0)),
    )
    h2_db = {lk: (10 * math.log10(v) if v > 0 else -math.inf) for lk, v in result.mean_h2.items()}
    p_dbm = {r: result.mean_rx_power_dbm(r) for r in result.mean_rx_power_w}
    summary: dict[str, Any] = {
        "scenario": res.scenario.name,
        "duration_s": str(dur),
        "pieces": len(result.t),
        "weights": result.weights,
        "mean_h2_db": h2_db,
        "mean_rx_power_dbm": p_dbm,
        "tau_ris_s": result.tau_ris_s,
        "records": len(result.records),
    }
    if out is not None:
        table = {
            "t_s": [f"{t.numerator}/{t.denominator}" for t in result.t],
            "w": result.w.tolist(),
            "h2": {lk: v.tolist() for lk, v in result.h2.items()},
            "rx_power_w": {r: v.tolist() for r, v in result.rx_power_w.items()},
            "summary": summary,
        }
        out.write_text(json.dumps(table) + "\n", encoding="utf-8")
    if json_out:
        typer.echo(json.dumps(summary, indent=2, default=str))
    else:
        head = f"batch {res.scenario.name}: {len(result.t)} pieces over {dur} s"
        typer.echo(f"{head} ({result.weights} weights)")
        for lk, v in h2_db.items():
            typer.echo(f"  link {lk}: mean |h|^2 {v:.4f} dB")
        for r, v in p_dbm.items():
            typer.echo(f"  rx {r}: mean power {v:.4f} dBm (unit digital TX power)")
