"""Shared helpers of the WP11 experiment drivers: run directories, manifests, meta, CSV.

Every experiment output is a run directory (DESIGN.md section 3.6) holding

* ``manifest.json``: the :class:`sremu.record.manifest.Manifest` model (sremu version, git
  commit and dirty flag, dirty-patch hash, package versions, platform, CPU, power plan,
  scenario hash, inputs, assumptions, seed, RNG scheme, mode, backend, threads, wall time,
  host load samples). The git state is read once per process (read-only ``git rev-parse``,
  ``status``, ``diff``, ``ls-files`` inside :func:`sremu.record.manifest.git_state`) and a
  dirty tree's ``dirty.patch`` is written once per sweep root (its SHA-256 is in every
  manifest), not once per point;
* ``scenario.resolved.yaml`` when the run has a scenario;
* CSV and/or JSON data;
* ``meta.json``: experiment id, data classes (DESIGN.md 3.6 ``data_class`` values),
  assumptions, inputs and a description.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import sys
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sremu import sysinfo
from sremu.contracts import COUNTER_NAMES
from sremu.experiments.sweep import jsonable
from sremu.record.manifest import (
    PATCH_NAME,
    RESOLVED_SCENARIO_NAME,
    HostLoadSampler,
    Manifest,
    environment_info,
    git_state,
    utc_now_iso,
    write_manifest,
)
from sremu.rng import RNG_SCHEME

REPO = Path(__file__).resolve().parents[1]
"""The sremu repository root (this file is ``<repo>/experiments/common.py``)."""

PAPER = REPO.parent / "paper"
"""The paper repository (separate git repo)."""

DATA_CLASSES = (
    "MEAS",
    "DIG",
    "REPORTED",
    "ASSUMED",
    "SIM-sremu",
    "SIM-legacy-v1",
    "SIM-legacy-v2",
    "ANALYTIC",
    "MATLAB",
)

_GIT_ENV = "SREMU_EXP_GIT_JSON"
_git_cache: dict[str, Any] | None = None
_env_cache: dict[str, Any] | None = None


# =========================================================================== small I/O


def write_json(path: str | Path, obj: Any) -> Path:
    """Write JSON (2-space indent, UTF-8, LF; numpy/Fraction made JSON-ready)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps(jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(tmp, p)
    return p


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cell(v: Any) -> Any:
    v = jsonable(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list | dict):
        return json.dumps(v, sort_keys=True)
    return "" if v is None else v


def write_csv(
    path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None
) -> Path:
    """Write rows as CSV (header = ``columns`` or the union of keys in first-seen order).
    Floats are written with ``repr`` (round-trip exact); None as an empty cell."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = list(columns) if columns is not None else []
    if columns is None:
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([_cell(r.get(c)) for c in cols])
    os.replace(tmp, p)
    return p


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV written by :func:`write_csv` (values as strings)."""
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def db10(x: float) -> float:
    """``10 log10(x)`` (``-inf`` for 0, NaN for negative)."""
    if x > 0:
        return 10.0 * math.log10(x)
    return -math.inf if x == 0 else math.nan


# =========================================================================== git / environment


def git_info(patch_root: str | Path | None = None) -> dict[str, Any]:
    """Git state of the sremu tree, read once per process.

    Returns ``{commit, dirty, dirty_patch_sha256, dirty_patch_path}``. When the tree is dirty
    and ``patch_root`` is given, ``<patch_root>/dirty.patch`` is written (once). A driver may
    pass the result to worker processes through the environment variable
    ``SREMU_EXP_GIT_JSON`` (a JSON file path), so workers do not run git themselves.
    """
    global _git_cache
    if _git_cache is not None:
        return _git_cache
    env_path = os.environ.get(_GIT_ENV)
    if env_path and Path(env_path).is_file():
        _git_cache = read_json(env_path)
        return _git_cache
    gs = git_state(REPO)
    info: dict[str, Any] = {
        "commit": gs.commit,
        "dirty": gs.dirty,
        "dirty_patch_sha256": gs.patch_sha256,
        "dirty_patch_path": None,
        "reason": gs.reason,
    }
    if gs.dirty and gs.patch is not None and patch_root is not None:
        pr = Path(patch_root)
        pr.mkdir(parents=True, exist_ok=True)
        # the shared tree changes while runs go on: one patch file per distinct state
        pp = pr / f"dirty_{gs.patch_sha256[:16]}.patch"
        if not pp.exists():
            tmp = pp.with_name(pp.name + f".{os.getpid()}.tmp")
            tmp.write_bytes(gs.patch)
            os.replace(tmp, pp)
        info["dirty_patch_path"] = str(pp)
    info["code"] = code_hashes()
    _git_cache = info
    return info


def code_hashes() -> dict[str, Any]:
    """SHA-256 of every experiment driver (``experiments/*.py``), study scenario
    (``scenarios/studies/*``, ``scenarios/lab_like_A.yaml``), sweep spec (``paper/sweeps/*.yaml``)
    and one combined hash of ``src/sremu/**/*.py`` (sorted paths), at the time of the call."""
    import hashlib

    def h(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    files: dict[str, str] = {}
    for pat in ("experiments/*.py", "scenarios/studies/*", "scenarios/lab_like_A.yaml"):
        for p in sorted(REPO.glob(pat)):
            if p.is_file():
                files[p.relative_to(REPO).as_posix()] = h(p)
    for p in sorted((PAPER / "sweeps").glob("*.yaml")):
        files["paper/" + p.relative_to(PAPER).as_posix()] = h(p)
    src = hashlib.sha256()
    for p in sorted((REPO / "src" / "sremu").rglob("*.py")):
        src.update(p.relative_to(REPO).as_posix().encode())
        src.update(bytes(1))
        src.update(p.read_bytes())
    return {"files": files, "src_sremu_tree_sha256": src.hexdigest()}


def export_git_info(root: str | Path) -> dict[str, Any]:
    """Read the git state once (writing ``dirty.patch`` into ``root`` when dirty), save it
    as ``<root>/git_state.json`` and point ``SREMU_EXP_GIT_JSON`` at it for workers."""
    info = git_info(root)
    p = write_json(Path(root) / "git_state.json", info)
    os.environ[_GIT_ENV] = str(p)
    return info


def environment() -> dict[str, Any]:
    """:func:`sremu.record.manifest.environment_info`, cached per process."""
    global _env_cache
    if _env_cache is None:
        _env_cache = environment_info()
    return _env_cache


# =========================================================================== run directory


@dataclass
class RunRecord:
    """A run directory being written: call :meth:`finish` at the end.

    ``resolved``: a :class:`sremu.scenario.load.ResolvedScenario` (or None for runs without
    a scenario, then ``scenario_name``/``scenario_sha256`` describe the inputs).
    """

    run_dir: Path
    seed: int
    resolved: Any = None
    scenario_name: str = "none"
    scenario_sha256: str = "0" * 64
    mode: str = "batch"
    coupling: str = "none"
    backend: str = "none"
    deterministic: bool = True
    threads: int | None = None
    argv: list[str] = field(default_factory=lambda: list(sys.argv))
    assumptions: list[str] = field(default_factory=list)
    inputs: list[dict[str, str]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    links: dict[str, Any] = field(default_factory=dict)
    page_size: dict[str, int] = field(default_factory=dict)
    verify_interp_max_rel: float | None = None
    richardson_db: float | None = None
    nondeterministic_inputs: bool = False
    oracle_csi_used: bool = False
    patch_root: Path | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.perf_counter()
        self._t_start = utc_now_iso()
        self._sampler = HostLoadSampler()
        self._load: list[dict[str, Any]] = []
        if self.resolved is not None:
            sc = self.resolved.scenario
            self.scenario_name = sc.name
            self.scenario_sha256 = self.resolved.sha256()
            (self.run_dir / RESOLVED_SCENARIO_NAME).write_text(
                self.resolved.to_yaml(), encoding="utf-8", newline="\n"
            )
            for r in self.resolved.inputs:
                d = r.to_dict()
                if d not in self.inputs:
                    self.inputs.append(d)
            for a in self.resolved.assumptions:
                if a not in self.assumptions:
                    self.assumptions.append(a)
            if self.backend == "none":
                self.backend = sc.numerics.backend
            self.deterministic = bool(sc.numerics.deterministic)

    def sample_load(self) -> None:
        """Append a host-load sample (busy CPU fraction since the previous sample)."""
        self._load.append(self._sampler.sample().model_dump())

    def add_assumption(self, text: str) -> None:
        if text not in self.assumptions:
            self.assumptions.append(text)

    def add_input(self, path: str, sha256: str, data_class: str) -> None:
        d = {"path": str(path), "sha256": sha256, "data_class": data_class}
        if d not in self.inputs:
            self.inputs.append(d)

    def finish(self) -> Manifest:
        """Write ``manifest.json`` and return it."""
        self.sample_load()
        g = git_info(self.patch_root)
        env = environment()
        counters = dict.fromkeys(COUNTER_NAMES, 0)
        counters.update({k: int(v) for k, v in self.counters.items()})
        data = {
            **env,
            "git_commit": g["commit"],
            "git_dirty": g["dirty"],
            "dirty_patch_sha256": g["dirty_patch_sha256"],
            "scenario_name": self.scenario_name,
            "scenario_sha256": self.scenario_sha256,
            "inputs": self.inputs,
            "assumptions": self.assumptions,
            "seed": int(self.seed),
            "rng_scheme": RNG_SCHEME,
            "mode": self.mode,
            "coupling": self.coupling,
            "backend": self.backend,
            "threads": self.threads if self.threads is not None else sysinfo.numba_threads(),
            "deterministic": bool(self.deterministic),
            "numba_disable_svml": sysinfo.numba_svml_disabled(),
            "page_size": self.page_size,
            "links": self.links,
            "verify_interp_max_rel": self.verify_interp_max_rel,
            "richardson_db": self.richardson_db,
            "counters": counters,
            "nondeterministic_inputs": self.nondeterministic_inputs,
            "oracle_csi_used": self.oracle_csi_used,
            "argv": self.argv,
            "t_start_utc": self._t_start,
            "t_end_utc": utc_now_iso(),
            "wall_s": time.perf_counter() - self._t0,
            "host_load_samples": self._load,
        }
        m = Manifest.model_validate(data)
        write_manifest(self.run_dir, m)
        return m


def write_meta(
    run_dir: str | Path,
    *,
    experiment: str,
    data_classes: Iterable[str],
    description: str,
    assumptions: Iterable[str] = (),
    inputs: Iterable[Mapping[str, str]] = (),
    files: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``meta.json``: experiment id, data classes (validated against DESIGN.md 3.6),
    description, assumptions, inputs, the data files with their meaning, and extras."""
    dcs = list(dict.fromkeys(data_classes))
    bad = [d for d in dcs if d not in DATA_CLASSES]
    if bad:
        raise ValueError(f"unknown data classes {bad}; allowed {DATA_CLASSES}")
    doc: dict[str, Any] = {
        "experiment": experiment,
        "data_classes": dcs,
        "description": description,
        "assumptions": list(dict.fromkeys(assumptions)),
        "inputs": list(inputs),
        "files": dict(files or {}),
        "code": git_info().get("code"),
        "written_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
    }
    if extra:
        doc.update(dict(extra))
    return write_json(Path(run_dir) / "meta.json", doc)


def quiet_warnings() -> None:
    """Silence ``ValidityWarning``-style warnings in bulk sweeps (they are recorded in the
    resolved scenario's assumptions instead)."""
    warnings.simplefilter("ignore")


def stats_dict(x: np.ndarray) -> dict[str, float]:
    """min/mean/max of an array (JSON-ready)."""
    a = np.asarray(x, dtype=np.float64)
    return {"min": float(a.min()), "mean": float(a.mean()), "max": float(a.max())}
