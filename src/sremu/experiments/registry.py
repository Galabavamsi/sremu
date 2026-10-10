"""Experiment runner registry (DESIGN.md sections 2.5, 11; WP9).

A **runner** computes one sweep point: ``runner(ctx: PointContext) -> Mapping[str, Any]``.
It receives the point's parameters, its master seed and its own run directory (where it
may write anything: a manifest, IQ, tables), and returns a JSON-able summary that the sweep
writes to ``summary.json`` and to the sweep index.

Runners are referenced from a sweep spec by **name**. Built-in names are registered here;
studies register theirs with :func:`register`. A ``"module:function"`` reference is
resolved by import only for modules under ``sremu.`` unless the caller allows other
imports explicitly (``allow_import``; the CLI's ``--allow-import``), because a sweep spec is
a data file.

Built-in runners:

* ``echo``: returns the parameters, the seed and one keyed uniform draw (purpose
  ``trial``, name ``<sweep>/<point_id>``); used by tests and as a template.
* ``controller_rollout``: drives a built-in controller (section 8.7) through
  ``sremu.ml.env.RisEnv`` (batch backend) and reports the mean received power with a
  batch-means interval and the mean regret to the exact optimum.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sremu import rng as _rng
from sremu.errors import ConfigurationError

__all__ = [
    "RUNNERS",
    "PointContext",
    "RunnerEntry",
    "register",
    "resolve",
]


@dataclass(frozen=True)
class PointContext:
    """What a runner gets for one sweep point.

    ``sweep``: sweep name. ``point_id``: stable id (``p00012``). ``index``: position in the
    sweep's point order. ``params``: merged parameters (``base`` + grid values).
    ``seed``: the point's master seed. ``run_dir``: the point's directory (exists).
    """

    sweep: str
    point_id: str
    index: int
    params: Mapping[str, Any]
    seed: int
    run_dir: Path
    extra: Mapping[str, Any] = field(default_factory=dict)

    def trial_rng(self, name: str = "") -> Any:
        """Keyed generator, purpose ``trial``, name ``<sweep>/<point_id>[/<name>]``."""
        key = f"{self.sweep}/{self.point_id}" + (f"/{name}" if name else "")
        return _rng.generator(self.seed, "trial", key)


Runner = Callable[[PointContext], Mapping[str, Any]]


@dataclass(frozen=True)
class RunnerEntry:
    """A registered runner: ``fn`` and a one-line ``description``."""

    fn: Runner
    description: str = ""


RUNNERS: dict[str, RunnerEntry] = {}
"""Runner name -> entry."""


def register(name: str, fn: Runner | None = None, *, description: str = "") -> Any:
    """Register ``fn`` under ``name`` (usable as a decorator). Re-registering a name raises."""

    def deco(f: Runner) -> Runner:
        if not name or ":" in name:
            raise ConfigurationError(f"invalid runner name {name!r}")
        if name in RUNNERS and RUNNERS[name].fn is not f:
            raise ConfigurationError(f"runner {name!r} is already registered")
        RUNNERS[name] = RunnerEntry(f, description or (f.__doc__ or "").strip().split("\n")[0])
        return f

    return deco if fn is None else deco(fn)


def resolve(ref: str, *, allow_import: bool = False) -> Runner:
    """The runner for a name or a ``"module:function"`` reference.

    ``module:function`` imports ``module`` only when it is under ``sremu.`` or when
    ``allow_import`` is True (``ConfigurationError`` otherwise).
    """
    if ref in RUNNERS:
        return RUNNERS[ref].fn
    mod, sep, attr = ref.partition(":")
    if not sep:
        raise ConfigurationError(f"unknown runner {ref!r}; registered: {sorted(RUNNERS)}")
    if not (mod == "sremu" or mod.startswith("sremu.")) and not allow_import:
        raise ConfigurationError(
            f"runner {ref!r}: importing modules outside sremu needs allow_import "
            "(CLI: --allow-import)"
        )
    try:
        module = importlib.import_module(mod)
    except ImportError as exc:
        raise ConfigurationError(f"runner {ref!r}: cannot import {mod!r}: {exc}") from exc
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise ConfigurationError(f"runner {ref!r}: {attr!r} is not a callable in {mod!r}")
    return fn


# =========================================================================== built-ins


@register("echo", description="Return the parameters, the seed and one keyed uniform draw.")
def echo(ctx: PointContext) -> Mapping[str, Any]:
    """Return the parameters, the seed and one keyed uniform draw (purpose ``trial``)."""
    u = float(ctx.trial_rng().random())
    if ctx.params.get("fail"):
        raise RuntimeError(f"echo: point {ctx.point_id} asked to fail")
    out: dict[str, Any] = {"seed": ctx.seed, "u": u}
    for k, v in ctx.params.items():
        if isinstance(v, int | float | str | bool) or v is None:
            out[f"param.{k}"] = v
    return out


@register(
    "controller_rollout",
    description="Built-in controller through RisEnv (batch backend): mean power, regret.",
)
def controller_rollout(ctx: PointContext) -> Mapping[str, Any]:
    """Drive a built-in controller through ``RisEnv`` (batch backend).

    Parameters: ``scenario`` (path, relative to the sweep spec's directory when given in
    ``ctx.extra["spec_dir"]``), ``controller`` (a section 9.2 controller mapping: ``impl``,
    ``period_s``, ``params``...), ``decision_period_s``, ``n_steps``, optional ``panel``,
    ``link``, ``eval_points``, ``include_noise`` and ``n_batches`` (default 10).
    Returns ``mean_power_dbm`` (10 log10 of the mean linear power), its batch-means
    interval in dB, ``mean_regret_db`` and ``n_steps``.
    """
    from sremu.analysis.stats import batch_means
    from sremu.control.builtin import build_controller
    from sremu.ml.env import RisEnv, rollout

    p = dict(ctx.params)
    for key in ("scenario", "controller", "decision_period_s", "n_steps"):
        if key not in p:
            raise ConfigurationError(f"controller_rollout: missing parameter {key!r}")
    scen = Path(str(p["scenario"]))
    base = ctx.extra.get("spec_dir")
    if not scen.is_absolute() and base is not None:
        scen = Path(base) / scen
    env = RisEnv(
        scen,
        panel=p.get("panel", "ris0"),
        decision_period_s=str(p["decision_period_s"]),
        link=p.get("link"),
        eval_points=int(p.get("eval_points", 1)),
        include_noise=bool(p.get("include_noise", False)),
    )
    spec = dict(p["controller"])
    spec.setdefault("name", "ctrl")
    # reset first: a point seed other than the scenario's rebuilds the scene (random
    # waypoints are keyed by it), and the controller's model must be that scene
    env.reset(seed=ctx.seed)
    ctrl = build_controller(spec, model=env.model)
    res = rollout(env, ctrl, int(p["n_steps"]), seed=ctx.seed)
    lin = [10.0 ** (r / 10.0) for r in res["rewards_dbm"]]
    k = min(int(p.get("n_batches", 10)), len(lin))
    out: dict[str, Any] = {
        "n_steps": len(lin),
        "mean_power_dbm": 10.0 * math.log10(sum(lin) / len(lin)),
        "mean_regret_db": float(sum(res["regret_db"]) / len(res["regret_db"])),
        "n_commands": int(res["n_commands"]),
    }
    if k >= 2:
        bm = batch_means(lin, k)
        out["ci_low_dbm"] = 10.0 * math.log10(bm.ci_low) if bm.ci_low > 0 else -math.inf
        out["ci_high_dbm"] = 10.0 * math.log10(bm.ci_high)
        out["n_batches"] = bm.n_batches
    return out
