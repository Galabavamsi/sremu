"""Sweep runner: one run directory per point, plus an index (DESIGN.md sections 2.5, 3.6;
WP9).

A sweep spec (YAML, JSON or TOML; read with the scenario reader) is a mapping::

    name: ea_tsui                 # safe directory name
    runner: controller_rollout    # registered runner, or "sremu.module:function"
    base: {scenario: corridor.yaml, n_steps: 1000}   # parameters shared by every point
    grid:                         # Cartesian product; the first key varies slowest
      tau_sui_s: ["0.01", "0.1", "1"]
      controller: [{impl: Urap, ...}, {impl: ExactOptimum, ...}]
    zip:                          # optional: lists of equal length that vary together,
      h_m: [2, 4]                 #   crossed with the grid (outermost)
      label: [low, high]
    seeds: [0, 1, 2]              # optional; or ``repeats: 3`` (seeds 0..2); default [0]

Points are ordered (zip index, grid index, seed) and get stable ids ``p00000``,
``p00001``, ... The output tree is::

    <out>/<name>/
      sweep.json      spec, spec sha256, runner, point count, environment, times, argv
      index.json      one entry per point: id, params, seed, status, wall_s, summary
      index.csv       the same, flattened scalar columns (``param.*``, ``result.*``)
      results.parquet the same as a table, when pyarrow is installed (optional)
      p00000/         point.json (params, seed, params_sha256, status, error), summary.json,
      ...             and whatever the runner writes

``jobs > 1`` runs points in worker processes (``ProcessPoolExecutor``); the index is
always written in point order, so it does not depend on scheduling. Workers resolve the
runner again: a runner registered by name must be registered when its module is imported
(on Windows the workers are spawned and re-import the main module), else use a
``"module:function"`` reference; otherwise every point fails with "unknown runner".
``resume`` skips points whose ``point.json`` says ``ok`` for the same runner, parameters
and seed. A failing point is recorded with status ``failed`` and its error; the others
still run.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from sremu.errors import ConfigurationError
from sremu.experiments.registry import PointContext, resolve

__all__ = [
    "INDEX_CSV",
    "INDEX_JSON",
    "PARQUET_NAME",
    "POINT_JSON",
    "SUMMARY_JSON",
    "SWEEP_JSON",
    "Point",
    "SweepResult",
    "SweepSpec",
    "jsonable",
    "load_spec",
    "run_sweep",
]

SWEEP_JSON = "sweep.json"
INDEX_JSON = "index.json"
INDEX_CSV = "index.csv"
PARQUET_NAME = "results.parquet"
POINT_JSON = "point.json"
SUMMARY_JSON = "summary.json"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_SPEC_KEYS = {"name", "runner", "base", "grid", "zip", "seeds", "repeats", "description"}


# =========================================================================== JSON helpers


def jsonable(x: Any) -> Any:
    """Convert to JSON-ready values: numpy scalars and arrays, ``Fraction`` (``"p/q"``
    string), complex (``[re, im]``), paths (strings); non-finite floats become None."""
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v) for v in x.tolist()]
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, complex | np.complexfloating):
        return [jsonable(complex(x).real), jsonable(complex(x).imag)]
    if isinstance(x, Fraction):
        return str(x)
    if isinstance(x, Path):
        return str(x)
    if x is None or isinstance(x, str):
        return x
    return str(x)


def _dumps(obj: Any) -> str:
    return json.dumps(jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False, sort_keys=False)


def _canonical(obj: Any) -> str:
    return json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


# =========================================================================== spec


@dataclass(frozen=True)
class Point:
    """One sweep point: ``index``, stable ``point_id``, merged ``params`` and ``seed``."""

    index: int
    point_id: str
    params: Mapping[str, Any]
    seed: int

    @property
    def params_sha256(self) -> str:
        """SHA-256 of the canonical JSON of (params, seed)."""
        return hashlib.sha256(
            _canonical({"params": self.params, "seed": self.seed}).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class SweepSpec:
    """A validated sweep spec (see the module docstring)."""

    name: str
    runner: str
    base: Mapping[str, Any] = field(default_factory=dict)
    grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    zip: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    seeds: tuple[int, ...] = (0,)
    description: str = ""
    source: Path | None = None
    sha256: str = ""

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any], *, source: Path | None = None) -> SweepSpec:
        """Validate a spec mapping."""
        if not isinstance(d, Mapping):
            raise ConfigurationError("a sweep spec must be a mapping")
        unknown = set(d) - _SPEC_KEYS
        if unknown:
            raise ConfigurationError(f"unknown sweep spec keys {sorted(unknown)}")
        name = d.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name) or len(name) > 64:
            raise ConfigurationError(f"sweep name {name!r} is not a safe directory name")
        runner = d.get("runner")
        if not isinstance(runner, str) or not runner:
            raise ConfigurationError("sweep spec needs a runner name")
        base = dict(d.get("base") or {})
        grid = dict(d.get("grid") or {})
        zp = dict(d.get("zip") or {})
        for label, block in (("grid", grid), ("zip", zp)):
            for k, v in block.items():
                if not isinstance(v, list) or not v:
                    raise ConfigurationError(f"{label}.{k} must be a non-empty list")
        if set(grid) & set(zp) or (set(grid) | set(zp)) & set(base):
            raise ConfigurationError("a parameter may appear in only one of base, grid, zip")
        if zp and len({len(v) for v in zp.values()}) != 1:
            raise ConfigurationError("zip lists must have equal lengths")
        if "seeds" in d and "repeats" in d:
            raise ConfigurationError("give seeds or repeats, not both")
        if "repeats" in d:
            r = d["repeats"]
            if isinstance(r, bool) or not isinstance(r, int) or r < 1:
                raise ConfigurationError("repeats must be an int >= 1")
            seeds: tuple[int, ...] = tuple(range(r))
        else:
            raw = d.get("seeds", [0])
            if not isinstance(raw, list) or not raw:
                raise ConfigurationError("seeds must be a non-empty list of ints")
            for s in raw:
                if isinstance(s, bool) or not isinstance(s, int) or s < 0:
                    raise ConfigurationError(f"seed {s!r} must be an int >= 0")
            if len(set(raw)) != len(raw):
                raise ConfigurationError("seeds must be distinct")
            seeds = tuple(int(s) for s in raw)
        sha = hashlib.sha256(_canonical(dict(d)).encode("utf-8")).hexdigest()
        return cls(
            name=name,
            runner=runner,
            base=base,
            grid=grid,
            zip=zp,
            seeds=seeds,
            description=str(d.get("description", "")),
            source=source,
            sha256=sha,
        )

    def points(self) -> list[Point]:
        """Every point in (zip, grid, seed) order with stable ids."""
        zkeys = list(self.zip)
        zlen = len(next(iter(self.zip.values()))) if self.zip else 1
        gkeys = list(self.grid)
        combos = list(itertools.product(*(self.grid[k] for k in gkeys))) if gkeys else [()]
        out: list[Point] = []
        for zi in range(zlen):
            zvals = {k: self.zip[k][zi] for k in zkeys}
            for combo in combos:
                params = {**self.base, **zvals, **dict(zip(gkeys, combo, strict=True))}
                for seed in self.seeds:
                    i = len(out)
                    out.append(Point(i, f"p{i:05d}", params, seed))
        return out


def load_spec(path: str | Path) -> SweepSpec:
    """Read and validate a sweep spec file (YAML/JSON/TOML)."""
    from sremu.scenario.load import read_document

    p = Path(path)
    return SweepSpec.from_mapping(read_document(p), source=p.resolve())


# =========================================================================== execution


def _utc() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run_point(task: dict[str, Any]) -> dict[str, Any]:
    """Run one point (top level, so worker processes can import it)."""
    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "sweep": task["sweep"],
        "point_id": task["point_id"],
        "index": task["index"],
        "params": task["params"],
        "seed": task["seed"],
        "params_sha256": task["params_sha256"],
        "runner": task["runner"],
        "status": "running",
        "started_utc": _utc(),
    }
    t0 = time.perf_counter()
    summary: Any = None
    try:
        fn = resolve(task["runner"], allow_import=task["allow_import"])
        ctx = PointContext(
            sweep=task["sweep"],
            point_id=task["point_id"],
            index=task["index"],
            params=task["params"],
            seed=task["seed"],
            run_dir=run_dir,
            extra=task["extra"],
        )
        out = fn(ctx)
        if not isinstance(out, Mapping):
            raise ConfigurationError(f"runner returned {type(out).__name__}, expected a mapping")
        summary = jsonable(out)
        _write_text(run_dir / SUMMARY_JSON, _dumps(summary))
        record["status"] = "ok"
        record["error"] = None
    except Exception as exc:  # recorded per point; the sweep continues
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["ended_utc"] = _utc()
    record["wall_s"] = time.perf_counter() - t0
    _write_text(run_dir / POINT_JSON, _dumps(record))
    record["summary"] = summary
    return record


def _resumable(run_dir: Path, pt: Point, runner: str) -> dict[str, Any] | None:
    try:
        rec = json.loads((run_dir / POINT_JSON).read_text(encoding="utf-8"))
        summary = json.loads((run_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        rec.get("status") != "ok"
        or rec.get("params_sha256") != pt.params_sha256
        or rec.get("runner") != runner
    ):
        return None
    rec["summary"] = summary
    rec["resumed"] = True
    return dict(rec)


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, bool | int | float | str) or value is None:
        out[prefix] = value
    else:
        out[prefix] = json.dumps(jsonable(value), sort_keys=True)


def _index_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        row: dict[str, Any] = {
            "point_id": r["point_id"],
            "index": r["index"],
            "seed": r["seed"],
            "status": r["status"],
            "wall_s": r.get("wall_s"),
        }
        _flatten("param", r["params"], row)
        if isinstance(r.get("summary"), Mapping):
            _flatten("result", r["summary"], row)
        rows.append(row)
    return rows


def _write_index(root: Path, records: Sequence[Mapping[str, Any]]) -> bool:
    entries = [
        {
            "point_id": r["point_id"],
            "index": r["index"],
            "dir": r["point_id"],
            "params": r["params"],
            "seed": r["seed"],
            "status": r["status"],
            "error": r.get("error"),
            "wall_s": r.get("wall_s"),
            "summary": r.get("summary"),
        }
        for r in records
    ]
    _write_text(root / INDEX_JSON, _dumps(entries))
    rows = _index_rows(records)
    cols: list[str] = []
    for row in rows:
        for c in row:
            if c not in cols:
                cols.append(c)
    tmp = root / (INDEX_CSV + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({c: jsonable(row.get(c)) for c in cols})
    os.replace(tmp, root / INDEX_CSV)
    try:
        import pyarrow as pa  # type: ignore[import-not-found, unused-ignore]
        import pyarrow.parquet as pq  # type: ignore[import-not-found, unused-ignore]
    except ImportError:
        return False
    table = pa.Table.from_pylist([{c: jsonable(row.get(c)) for c in cols} for row in rows])
    pq.write_table(table, root / PARQUET_NAME)
    return True


@dataclass(frozen=True)
class SweepResult:
    """Outcome of :func:`run_sweep`: output ``root``, per-point ``records`` (point order),
    counts and whether ``results.parquet`` was written."""

    root: Path
    records: tuple[Mapping[str, Any], ...]
    n_ok: int
    n_failed: int
    n_resumed: int
    parquet: bool

    @property
    def ok(self) -> bool:
        """True when every point succeeded."""
        return self.n_failed == 0


def run_sweep(
    spec: SweepSpec,
    out_dir: str | Path,
    *,
    jobs: int = 1,
    resume: bool = False,
    allow_import: bool = False,
    environment: Mapping[str, Any] | None = None,
    argv: Sequence[str] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> SweepResult:
    """Run every point of ``spec`` under ``out_dir/<spec.name>/`` (see the module docstring).

    ``environment`` is recorded in ``sweep.json`` (default:
    ``sremu.record.manifest.environment_info()``). ``progress`` is called with each point's
    record as it completes (completion order).
    """
    if int(jobs) < 1:
        raise ConfigurationError("jobs must be >= 1")
    resolve(spec.runner, allow_import=allow_import)  # fail early on an unknown runner
    root = Path(out_dir) / spec.name
    root.mkdir(parents=True, exist_ok=True)
    if environment is None:
        from sremu.record.manifest import environment_info

        environment = environment_info()
    pts = spec.points()
    extra = {"spec_dir": None if spec.source is None else str(spec.source.parent)}
    sweep_doc: dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "runner": spec.runner,
        "spec_sha256": spec.sha256,
        "spec_path": None if spec.source is None else str(spec.source),
        "spec": {
            "base": spec.base,
            "grid": spec.grid,
            "zip": spec.zip,
            "seeds": list(spec.seeds),
        },
        "n_points": len(pts),
        "jobs": int(jobs),
        "environment": dict(environment),
        "argv": list(sys.argv if argv is None else argv),
        "t_start_utc": _utc(),
        "t_end_utc": None,
    }
    _write_text(root / SWEEP_JSON, _dumps(sweep_doc))
    records: dict[int, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    for pt in pts:
        run_dir = root / pt.point_id
        if resume:
            rec = _resumable(run_dir, pt, spec.runner)
            if rec is not None:
                records[pt.index] = rec
                if progress is not None:
                    progress(rec)
                continue
        tasks.append(
            {
                "sweep": spec.name,
                "point_id": pt.point_id,
                "index": pt.index,
                "params": jsonable(pt.params),
                "seed": pt.seed,
                "params_sha256": pt.params_sha256,
                "runner": spec.runner,
                "allow_import": allow_import,
                "run_dir": str(run_dir),
                "extra": extra,
            }
        )
    if int(jobs) == 1 or len(tasks) <= 1:
        for task in tasks:
            rec = _run_point(task)
            records[rec["index"]] = rec
            if progress is not None:
                progress(rec)
    else:
        with cf.ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            futs = {pool.submit(_run_point, t): t for t in tasks}
            for fut in cf.as_completed(futs):
                rec = fut.result()
                records[rec["index"]] = rec
                if progress is not None:
                    progress(rec)
    ordered = tuple(records[i] for i in sorted(records))
    parquet = _write_index(root, ordered)
    sweep_doc["t_end_utc"] = _utc()
    n_failed = sum(1 for r in ordered if r["status"] != "ok")
    sweep_doc["n_failed"] = n_failed
    _write_text(root / SWEEP_JSON, _dumps(sweep_doc))
    return SweepResult(
        root=root,
        records=ordered,
        n_ok=len(ordered) - n_failed,
        n_failed=n_failed,
        n_resumed=sum(1 for r in ordered if r.get("resumed")),
        parquet=parquet,
    )
