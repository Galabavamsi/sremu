"""Shared helpers of the WP10 validation studies V-TANG, V-CFO, V-WCL and V-FM.

Every study writes one run directory ``validation/<study>/results/<UTC>_<name>/`` with

* ``manifest.json`` and ``scenario.resolved.yaml`` (:class:`sremu.record.manifest.ManifestRecorder`
  of the study's base scenario: commit, dirty flag (+ ``dirty.patch``), seed, platform,
  versions, inputs with sha256 and data class, assumptions, wall time, host load);
* CSV and JSON data files written by the study;
* ``meta.json``: study id, data classes, inputs, every output file with its sha256,
  acceptance results, the numbers the paper may quote, seeds, run times and host load.

Numbers are written exactly as measured (``repr`` floats in CSV/JSON). Nothing here
interprets a result; the study scripts state acceptance against DESIGN.md section 10.
This module only uses public sremu APIs (scenario loader, manifest recorder, sysinfo).
``validation/<study>/report.md`` is rendered from a run directory's ``meta.json`` by the
study script (``--report RUN_DIR``), so every number in it is the measured value.

**Git state.** :func:`study_git_state` chooses the commit and dirty flag every run records,
with the rules of the WP11 experiments: ``$SREMU_EXP_GIT_JSON`` first (the state a driver
such as ``experiments/regenerate_paper_data.sh`` fixed once for all its processes; read as
``experiments/common.py:git_info`` reads it), then the work tree (read-only git), and for a
job bundle without ``.git`` ``$SREMU_GIT_COMMIT`` / ``$SREMU_GIT_DIRTY`` (set by
``research/aws/runner.py``; read as ``bench/_common.py:bundle_commit`` reads them).
``meta.json`` names the source (``git_state_source``).

**Numba environment.** Importing this module from a study script run as ``__main__`` (or its
spawned sweep workers), before numba is imported, applies the deterministic-mode
environment of the CLI (D-4, D15): ``NUMBA_DISABLE_INTEL_SVML=1`` and
the ``-nosvml`` cache directory, plus ``NUMBA_NUM_THREADS=1`` unless set (the studies
parallelise over sweep-runner worker processes, so each worker stays single-threaded).
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any


def _run_as_study_script() -> bool:
    """True when ``__main__`` is a script under ``validation/`` (a study run or one of its
    spawned sweep workers), never inside pytest or another host process."""
    f = getattr(sys.modules.get("__main__"), "__file__", None)
    if not f:
        return False
    try:
        Path(f).resolve().relative_to(Path(__file__).resolve().parent)
    except ValueError:
        return False
    return True


if "numba" not in sys.modules and _run_as_study_script():  # D-4, D15; see the docstring
    from sremu import sysinfo as _sysinfo

    os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
    os.environ["NUMBA_CACHE_DIR"] = _sysinfo.nosvml_cache_dir(os.environ.get("NUMBA_CACHE_DIR"))
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from sremu.record.manifest import GitState, git_state  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VALIDATION = REPO / "validation"
DATA_VALIDATION = REPO / "data" / "validation"

__all__ = [
    "DATA_VALIDATION",
    "REPO",
    "VALIDATION",
    "StudyRun",
    "db10",
    "fnum",
    "jsonable",
    "load_summary",
    "md_table",
    "passfail",
    "read_meta",
    "refresh_outputs",
    "rel",
    "resolve_mapping",
    "run_header",
    "sha256_file",
    "study_git_state",
    "write_csv",
    "write_json",
    "write_report",
]

GIT_JSON_ENV = "SREMU_EXP_GIT_JSON"
"""Path of a JSON git state ``{commit, dirty, dirty_patch_sha256, dirty_patch_path}`` fixed by
a driver for all its processes (the variable of ``experiments/common.py``)."""


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: str | Path) -> str:
    """Path relative to the sremu repository (posix), or absolute when outside it."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def db10(x: float) -> float:
    """``10 log10(x)`` (``-inf`` for x <= 0)."""
    return 10.0 * math.log10(x) if x > 0 else -math.inf


def jsonable(x: Any) -> Any:
    """JSON-ready values: numpy scalars/arrays, Fractions (``"p/q"``), complex
    (``[re, im]``), paths; non-finite floats become None."""
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
        return f"{x.numerator}/{x.denominator}" if x.denominator != 1 else str(x.numerator)
    if isinstance(x, Path):
        return rel(x)
    return x


def write_json(path: str | Path, obj: Any) -> Path:
    """Write JSON (2-space indent, UTF-8, LF, no NaN)."""
    p = Path(path)
    p.write_text(
        json.dumps(jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return p


def _cell(v: Any) -> Any:
    if isinstance(v, float | np.floating):
        f = float(v)
        return repr(f) if math.isfinite(f) else ""
    if isinstance(v, bool | np.bool_):
        return "true" if v else "false"
    if isinstance(v, Fraction):
        return jsonable(v)
    if v is None:
        return ""
    return v


def write_csv(
    path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None
) -> Path:
    """Write rows as CSV (LF line ends; floats as ``repr``; empty for None/NaN)."""
    p = Path(path)
    cols = list(columns) if columns is not None else []
    if columns is None:
        for r in rows:
            for c in r:
                if c not in cols:
                    cols.append(c)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: _cell(r.get(c)) for c in cols})
    return p


def resolve_mapping(d: Mapping[str, Any], base_dir: str | Path | None = None) -> Any:
    """Validate and resolve a scenario mapping (``sremu.scenario.load``); warnings off."""
    from sremu.scenario.load import parse_scenario, resolve_scenario

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return resolve_scenario(parse_scenario(dict(d)), base_dir, warn=False)


@dataclasses.dataclass(frozen=True)
class RecordedGitState(GitState):
    """A git state taken from a driver's JSON record: keeps the recorded patch SHA-256 when
    the patch bytes are not at hand (``patch`` None)."""

    recorded_patch_sha256: str | None = None

    @property
    def patch_sha256(self) -> str | None:
        return self.recorded_patch_sha256 if self.patch is None else super().patch_sha256


def study_git_state(repo_dir: str | Path | None = None) -> tuple[GitState, str]:
    """The git state a study run records, and where it came from (module docstring).

    1. ``$SREMU_EXP_GIT_JSON`` naming an existing file (otherwise ignored, as in
       ``experiments/common.py:git_info``). A dirty state's patch bytes are read from its
       ``dirty_patch_path`` when that file exists and has the recorded SHA-256 (the run
       directory then gets ``dirty.patch``); otherwise only the recorded SHA-256 is kept.
    2. The work tree containing ``repo_dir`` (default: this repository; read-only git).
    3. When that gives no commit (no ``.git``, e.g. an AWS job bundle): ``$SREMU_GIT_COMMIT``
       and ``$SREMU_GIT_DIRTY`` (``1``/``0``; unset = unknown), as ``bench/_common.py``.
    """
    env_path = os.environ.get(GIT_JSON_ENV)
    if env_path and Path(env_path).is_file():
        d = json.loads(Path(env_path).read_text(encoding="utf-8"))
        rec_sha = d.get("dirty_patch_sha256")
        patch = None
        pp = d.get("dirty_patch_path")
        if d.get("dirty") and pp and Path(pp).is_file():
            b = Path(pp).read_bytes()
            patch = b if hashlib.sha256(b).hexdigest() == rec_sha else None
        gs = RecordedGitState(
            d.get("commit"), d.get("dirty"), patch, d.get("reason"), recorded_patch_sha256=rec_sha
        )
        return gs, f"${GIT_JSON_ENV} ({env_path})"
    gs = git_state(REPO if repo_dir is None else repo_dir)
    if gs.commit is None:
        commit = os.environ.get("SREMU_GIT_COMMIT", "").strip() or None
        if commit is not None:
            raw = os.environ.get("SREMU_GIT_DIRTY", "").strip()
            dirty = None if raw == "" else raw not in ("0", "false", "False")
            note = f"no git work tree ({gs.reason}); $SREMU_GIT_COMMIT/$SREMU_GIT_DIRTY"
            return GitState(commit, dirty, None, note), "$SREMU_GIT_COMMIT/$SREMU_GIT_DIRTY"
    return gs, "work tree (read-only git)"


class StudyRun:
    """One study run directory with its manifest and ``meta.json``.

    ``study_dir``: ``validation/<study>``. ``name``: run name. ``scenario``: the study's
    base scenario mapping (section 9.2), resolved and written as the manifest scenario.
    ``inputs``: (path, data_class) of input data files (copied under ``data/validation``).
    """

    def __init__(
        self,
        study_dir: str | Path,
        name: str,
        scenario: Mapping[str, Any],
        *,
        inputs: Iterable[tuple[str | Path, str]] = (),
        assumptions: Iterable[str] = (),
        argv: Sequence[str] | None = None,
        root: str | Path | None = None,
        resume_dir: str | Path | None = None,
    ) -> None:
        from sremu.record.manifest import ManifestRecorder, create_run_dir

        self.t0 = time.perf_counter()
        base = Path(root) if root is not None else Path(study_dir) / "results"
        self.resumed_from: list[dict[str, str]] = []
        if resume_dir is not None:
            # continue an interrupted run in place: keep the start-of-run patch/manifest
            self.run_dir = Path(resume_dir)
            if not self.run_dir.is_dir():
                raise SystemExit(f"--resume: {self.run_dir} is not a directory")
            for nm in ("dirty.patch", "manifest.json"):
                p = self.run_dir / nm
                if p.exists():
                    k = 1
                    while (self.run_dir / f"{p.stem}.start{k}{p.suffix}").exists():
                        k += 1
                    q = self.run_dir / f"{p.stem}.start{k}{p.suffix}"
                    p.rename(q)
                    self.resumed_from.append({"file": q.name, "sha256": sha256_file(q)})
        else:
            self.run_dir = create_run_dir(base, name)
        self.resolved = resolve_mapping(scenario)
        gs, self.git_source = study_git_state()
        self.recorder = ManifestRecorder(
            self.resolved, self.run_dir, argv=list(sys.argv if argv is None else argv), git=gs
        )
        self.recorder.sample_host_load()
        self.inputs: list[dict[str, str]] = []
        for path, dc in inputs:
            self.add_input(path, dc)
        for a in assumptions:
            self.recorder.add_assumption(a)
        self.timings: dict[str, float] = {}

    @property
    def git_commit(self) -> str | None:
        return self.recorder.git.commit

    @property
    def git_dirty(self) -> bool | None:
        return self.recorder.git.dirty

    def path(self, name: str) -> Path:
        """A file path inside the run directory."""
        return self.run_dir / name

    def add_input(self, path: str | Path, data_class: str) -> None:
        """Record an input data file (path relative to the repo, sha256, data class)."""
        entry = {"path": rel(path), "sha256": sha256_file(path), "data_class": data_class}
        if entry not in self.inputs:
            self.inputs.append(entry)
            self.recorder.add_input(entry)

    def add_assumption(self, text: str) -> None:
        self.recorder.add_assumption(text)

    def time(self, label: str, seconds: float) -> None:
        """Record a wall-clock duration (s) under ``label``."""
        self.timings[label] = float(seconds)

    def progress(self, _record: Mapping[str, Any] | None = None) -> None:
        """Sweep-runner progress hook: one host-load sample per completed point (so the
        manifest shows the background load over a long run)."""
        with contextlib.suppress(Exception):  # a failed probe must never stop a study
            self.recorder.sample_host_load()

    def finish(self, meta: Mapping[str, Any]) -> Path:
        """Write ``manifest.json`` and ``meta.json`` (with every output's sha256)."""
        man = self.recorder.finish()
        outputs = {}
        for p in sorted(self.run_dir.rglob("*")):
            if p.is_file() and p.name not in ("meta.json",):
                outputs[p.relative_to(self.run_dir).as_posix()] = sha256_file(p)
        loads = [s.cpu_busy_fraction for s in man.host_load_samples]
        doc = {
            "study": meta.get("study"),
            "run_dir": rel(self.run_dir),
            "sremu_commit": man.git_commit,
            "sremu_dirty": man.git_dirty,
            "dirty_patch_sha256": man.dirty_patch_sha256,
            "git_state_source": self.git_source,
            "seed": man.seed,
            "platform": man.platform,
            "cpu_model": man.cpu_model,
            "logical_cpus": man.logical_cpus,
            "power_plan_or_governor": man.power_plan_or_governor,
            "t_start_utc": man.t_start_utc,
            "t_end_utc": man.t_end_utc,
            "wall_s": man.wall_s,
            "timings_s": self.timings,
            "host_cpu_busy_fraction_samples": loads,
            "host_cpu_busy_fraction_summary": load_summary(loads),
            "resumed_from": self.resumed_from,
            "inputs": self.inputs,
            "assumptions": list(man.assumptions),
            **{k: v for k, v in meta.items() if k != "study"},
            "outputs": outputs,
        }
        return write_json(self.run_dir / "meta.json", doc)


def load_summary(loads: Sequence[float]) -> dict[str, Any]:
    """n, min, median, mean and max of host CPU busy-fraction samples (None when empty)."""
    v = [float(x) for x in loads if x is not None and math.isfinite(float(x))]
    if not v:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    a = np.asarray(v)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "max": float(a.max()),
    }


def refresh_outputs(run_dir: str | Path, meta: dict[str, Any]) -> Path:
    """Recompute ``meta["outputs"]`` (sha256 of every file but meta.json) and rewrite
    ``meta.json`` (used after a re-aggregation of an existing run directory)."""
    rd = Path(run_dir)
    meta["outputs"] = {
        p.relative_to(rd).as_posix(): sha256_file(p)
        for p in sorted(rd.rglob("*"))
        if p.is_file() and p.name != "meta.json"
    }
    return write_json(rd / "meta.json", meta)


# =========================================================================== reports


def fnum(x: Any, nd: int = 3, *, sign: bool = False) -> str:
    """Format a number for a report table (``-`` for None/NaN; ints unchanged)."""
    if x is None:
        return "-"
    if isinstance(x, bool | np.bool_):
        return "yes" if x else "no"
    if isinstance(x, int | np.integer):
        return str(int(x))
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(f):
        return "-"
    if f != 0 and (abs(f) < 10 ** (-nd) or abs(f) >= 1e7):
        return f"{f:+.{max(nd - 1, 1)}e}" if sign else f"{f:.{max(nd - 1, 1)}e}"
    return f"{f:+.{nd}f}" if sign else f"{f:.{nd}f}"


def md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A GitHub-flavoured markdown table (cells already formatted, or str()-ed; a literal
    ``|`` inside a cell is escaped)."""

    def cell(c: Any) -> str:
        return str(c).replace("|", "\\|")

    out = [
        "| " + " | ".join(cell(h) for h in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for r in rows:
        out.append("| " + " | ".join(cell(c) for c in r) + " |")
    return "\n".join(out)


def passfail(x: Any) -> str:
    """PASS / FAIL / n/a for an acceptance boolean."""
    if x is True:
        return "PASS"
    if x is False:
        return "FAIL"
    return "n/a"


def run_header(meta: Mapping[str, Any]) -> str:
    """Common provenance block of a report (commit, dirty flag, platform, times, load)."""
    ls = meta.get("host_cpu_busy_fraction_summary") or {}
    lines = [
        f"* Run directory: `{meta.get('run_dir')}` (manifest.json, meta.json, CSV/JSON data).",
        f"* sremu commit `{meta.get('sremu_commit')}`, dirty tree: {meta.get('sremu_dirty')} "
        f"(dirty.patch sha256 `{meta.get('dirty_patch_sha256')}`).",
        f"* Platform: {meta.get('platform')}, {meta.get('cpu_model')}, "
        f"{meta.get('logical_cpus')} logical CPUs, "
        f"power plan {meta.get('power_plan_or_governor')}.",
        f"* Started {meta.get('t_start_utc')}, ended {meta.get('t_end_utc')}, "
        f"wall time {fnum(meta.get('wall_s'), 1)} s.",
        f"* Host CPU busy fraction (all processes, sampled {ls.get('n', 0)} times): "
        f"median {fnum(ls.get('median'), 2)}, max {fnum(ls.get('max'), 2)}. Other jobs shared "
        "the host; wall times are indicative, not benchmarks.",
    ]
    if meta.get("resumed_from"):
        lines.append(
            "* Resumed in place; the provenance above is the last pass's. Start-of-pass files "
            f"kept: {meta['resumed_from']}."
        )
    return "\n".join(lines)


def write_report(study_dir: str | Path, text: str) -> Path:
    """Write ``<study_dir>/report.md`` (UTF-8, LF)."""
    p = Path(study_dir) / "report.md"
    p.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    return p


def read_meta(run_dir: str | Path) -> dict[str, Any]:
    """``meta.json`` of a run directory."""
    return json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))


def env_threads() -> dict[str, str | None]:
    """Thread-related environment (reported with timings)."""
    keys = ("NUMBA_NUM_THREADS", "NUMBA_DISABLE_INTEL_SVML", "OMP_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys}
