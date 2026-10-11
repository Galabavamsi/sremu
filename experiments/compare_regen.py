"""Compare a regenerated results tree with the paper's committed results.

    python experiments/compare_regen.py NEW [--ref REF] [--json REPORT.json] [--smoke]
        [--validation-new DIR] [--sremu-root DIR] [--no-code-check]
        [--rtol 1e-9] [--atol 1e-12] [--atol-db 1e-9] [--atol-hz 1e-7]

``NEW`` is ``<job out>/paper_results`` of ``experiments/regenerate_paper_data.sh``; ``REF``
defaults to ``<sremu>/../paper/sweeps/results``. Run it from anywhere; it needs only the
Python standard library.

Scopes:

* **WP11** (``REF``): the root ``summary.json`` / ``meta.json`` and ``E-A``, ``E-A-IQ``,
  ``E-B1``, ``E-B2``, ``E-C``, ``E-F/ber``;
* **E-B-arc** (``REF/E-B-arc``): the lab-arc E-B results (E-B1, E-B2, VA-5, plan, run
  summary, per-sweep ``index.csv`` / ``sweep.json``, ``meta.json``); ``host/``,
  ``git_state.json`` and ``run_manifests.json`` are provenance, not compared;
* **validation** (``--validation-new``, default ``<NEW>/../validation``): the regenerated
  V-TANG, V-CFO and V-FM run directories (``<dir>/<study>/<UTC>_<name>/``) against the run
  directories the paper reads (``TANG_RUN``, ``CFO_RUN``, ``FM_RUN`` of
  ``paper/tools/gen_results.py``, under the sremu checkout): every file but ``manifest.json``
  and ``dirty.patch``. V-LAB is not re-run (locked protocol); its integrity check from the job
  (``_regen/vlab_check.json``) is reported, and a full comparison fails when that record is
  missing or not ok.

What is compared, and how:

* **Exact** (deterministic quantities and decisions): row keys, integers (bit and packet
  error counts, command counts, sample indices, trial seeds), booleans (acceptance, CI and
  pass/fail decisions), exact fractions (``1/2000``), labels, spec hashes, and decision
  strings (``supported``, ``pass``, ...).
* **Tolerance** (float aggregates): every other number. A value passes when
  ``|new - ref| <= atol_db`` for a dB quantity (column or key name contains ``db``),
  ``|new - ref| <= atol_hz + rtol |ref|`` for a V-CFO frequency (name ends in ``_hz``), and
  ``|new - ref| <= atol + rtol |ref|`` otherwise. Max absolute and relative deviations and
  the number of bit-identical values are reported per column, whether or not they pass.
  Why these values: the paper rounds every quoted value to at most 3 decimals (dB, s) or 2
  significant digits, so 1e-9 (relative, and absolute in dB) is far below anything it
  prints, and far above float64 round-off (the earlier Windows/Linux checks of the same code
  differed by at most 7e-15 dB). ``atol_hz`` = 1e-7 Hz is 10x below the V-CFO-1 acceptance
  limit (1e-6 Hz): the noise-free CFO estimator errors are float64 round-off (3e-10 Hz, LS)
  or deterministic algorithmic residuals (2e-8 Hz lag-1, 3e-7 Hz periodogram) whose value is
  set by round-off near zero, where a relative tolerance means nothing. A changed paper macro
  is found by ``tools/gen_results.py --check`` on the installed tree, not here.
* **meta.json**, in part: data classes, descriptions, assumptions, inputs, acceptance and
  the deterministic checks, with the rules above; file hashes, code hashes, times, local
  paths, git fields and host fields are not compared (``META_SKIP``, ``RUN_FIELDS``).
* **Not compared** (host- or provenance-specific): wall times, real-time factors, CPU model,
  platform, instance type, data source, git fields, worker counts, argv, environment and
  times; ``manifest.json``, ``git_state.json``, ``run_manifests.json``, ``*.patch``; and in
  the validation runs the report sentences ``wording`` / ``wording_note``, which the study
  renders from numbers that are compared (``DERIVED_TEXT``; V-TANG's template changed after
  the reference run: its bias is now the exact decimal mean, 0.985 -> +0.98 dB instead of
  +0.99 dB, the value ``gen_results.py`` already computes itself). The
  provenance of both trees is reported separately: a regenerated tree is *pinned* when every
  manifest, VA-1 row, the E-B arc run and every validation run names the same commit, none is
  dirty, and every code record names the same ``src/sremu`` tree hash. A smoke tree
  (``summary.json`` ``meta.smoke``) never passes a full comparison.
* **Code check** (unless ``--no-code-check``): the job's git state and code hashes
  (``<NEW>/../_regen``) against the local sremu checkout (``--sremu-root``, default this
  repository): HEAD (read from ``.git``, no git process) equal to the job's commit, and the
  same ``src/sremu`` tree hash, experiment/scenario hashes and validation-code hashes. Whether
  the checkout is clean is not checked here (``git status`` is the lead's step).
* **E-F latency** (``E-F/latency``, collected only with ``finalize --with-latency``): MEAS of
  the recording host; its values are never compared, only its data class and host record.

``--smoke`` is for the reduced smoke run: reference rows absent from NEW are counted as "not
regenerated" (not failures), and an unpinned NEW tree or a failed code check is a warning,
not a failure. Rows present in both trees are compared with the same rules as a full run.

Exit status: 0 when every comparison passes (and NEW is pinned, the code check passes and the
job's V-LAB check record exists and passes, unless ``--smoke``), 1 otherwise, 2 on a usage
error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
SREMU_ROOT = HERE.parents[1]
DEFAULT_REF = HERE.parents[2] / "paper" / "sweeps" / "results"

ARC = "E-B-arc"
CSV_KEYS: dict[str, tuple[str, ...]] = {
    "E-A/gains.csv": ("geometry", "controller", "tau_sui_s"),
    "E-A/retention.csv": ("geometry", "tau_sui_s"),
    "E-A/tau_half.csv": ("geometry",),
    "E-A-IQ/va1.csv": ("controller", "tau_sui_s", "noise"),
    "E-B1/eb1_points.csv": ("alphabet", "los", "transition", "delta"),
    "E-B1/eb1_traces.csv": ("alphabet", "los", "transition", "delta", "symbol"),
    "E-B2/eb2_per.csv": ("sweep", "alphabet", "los", "semantics", "lead_s", "guard_s", "jitter_s"),
    "E-C/table8.csv": ("row",),
    "E-C/per_point.csv": ("geometry", "point", "row"),
    "E-F/ber/ber.csv": ("modulation", "ebn0_db"),
}
for _k in ("E-B1/eb1_points.csv", "E-B1/eb1_traces.csv", "E-B2/eb2_per.csv"):
    CSV_KEYS[f"{ARC}/{_k}"] = CSV_KEYS[_k]
"""Row keys of the known result tables (``.../index.csv`` of a sweep: ``point_id``; an
unknown CSV is compared row by row in order)."""

LIST_KEYS: dict[str, tuple[str, ...]] = {
    "tau_half": CSV_KEYS["E-A/tau_half.csv"],
    "rows": CSV_KEYS["E-A-IQ/va1.csv"],
    "f9b_rows_guard_0p5ms": CSV_KEYS["E-B2/eb2_per.csv"],
    "table8": CSV_KEYS["E-C/table8.csv"],
    "ber": CSV_KEYS["E-F/ber/ber.csv"],
}
"""Table rows copied into ``summary.json`` lists, matched by the same keys."""

HOST_FIELDS = frozenset(
    {
        "rtf",
        "wall_s",
        "iq_wall_s",
        "batch_wall_s",
        "eval_wall_s",
        "cpu_model",
        "platform",
        "source",
        "aws_instance_type",
        "git_commit",
        "git_dirty",
        "src_sremu_tree_sha256",
        "ea_iq_py_sha256",
        "batch_source",
        "batch_cpu_model",
        "hosts",
        "batch_reference_cross_host_check",
    }
)
"""Fields that describe the recording host or the code state, not the result."""

RUN_FIELDS = frozenset(
    {
        "t_start_utc",
        "t_end_utc",
        "started_utc",
        "ended_utc",
        "written_utc",
        "argv",
        "environment",
        "jobs",
        "source_dir",
        "run_dir",
        "timings_s",
        "host_cpu_busy_fraction_samples",
        "host_cpu_busy_fraction_summary",
        "power_plan_or_governor",
        "logical_cpus",
        "resumed_from",
        "outputs",
        "git_state_source",
        "sremu_commit",
        "sremu_dirty",
        "dirty_patch_sha256",
        "cpu_s_sum",
        "point_wall_s_max",
        "sweep_dir",
        "env_threads",
    }
)
"""Further run-, host- and time-specific fields of the E-B arc and validation files (sweep
``sweep.json`` / ``index.json`` / ``point.json``, plan and run summary, study ``meta.json``);
for CSV columns the last dotted component counts (``result.wall_s``)."""

DERIVED_TEXT = frozenset({"wording", "wording_note"})
"""Report sentences of the validation studies, rendered from compared numbers."""

PROVENANCE_FILES = frozenset({"manifest.json", "git_state.json", "run_manifests.json"})
"""Files not compared at all (git state, host, times), nor any ``*.patch``. ``meta.json`` is
compared in part."""

META_SKIP = frozenset(
    {
        "files",
        "code",
        "written_utc",
        "sweep_roots",
        "sweep_root",
        "out_root",
        "root",
        "imported_runs",
        "summary_sha256",
        "smoke",
    }
)
"""``meta.json`` fields not compared (file hashes, code hashes, times, local paths, import
records). Everything else is compared: ``experiment``, ``data_classes``, ``description`` and
``assumptions`` (the paper reads data classes and some setup numbers from them), ``inputs``
(path, data class, and SHA-256 except for run ``manifest.json`` inputs), and the
deterministic checks (E-A ``checks_per_geometry``, ``decision_table_checks``; E-A-IQ
``complete``, ``missing_rows``, ``all_accept``; ``failed_points``)."""
META_SKIP_EXTRA = {f"{ARC}/meta.json": frozenset({"run", "host", "sremu"})}
"""Per-file additions: the E-B arc ``meta.json`` provenance blocks (run, host, git state)."""
LATENCY_DIR = "E-F/latency"
_LATENCY_NOTE = "E-F API latency"  # the root meta.json assumption that says whether it is in
EXPERIMENTS = ("E-A", "E-A-IQ", "E-B1", "E-B2", "E-C", "E-F/ber")

VALIDATION_STUDIES: dict[str, tuple[str, str, str]] = {
    "V-TANG": ("tang", "vtang", "TANG_RUN"),
    "V-CFO": ("cfo", "vcfo", "CFO_RUN"),
    "V-FM": ("fm", "vfm", "FM_RUN"),
}
"""Study -> (subdirectory of the job's ``validation/``, run name, constant of
``paper/tools/gen_results.py`` naming the run directory the paper reads)."""
VALIDATION_FALLBACK = {
    "V-TANG": "validation/tang/results/20261010T232551Z_vtang",
    "V-CFO": "validation/cfo/results/20261010T232549Z_vcfo",
    "V-FM": "validation/fm/results/20261010T232549Z_vfm",
}
"""Reference run directories when ``gen_results.py`` is not at hand (its values of 555d9b0)."""
ATOL_HZ_SCOPE = ("V-CFO/",)
"""Files whose ``*_hz`` values use ``atol_hz`` (the V-CFO frequencies)."""

_INT = re.compile(r"^[+-]?\d+$")
_FRAC = re.compile(r"^[+-]?\d+/\d+$")


# =========================================================================== scalar rules


def _num(s: Any) -> float | None:
    """A float from a non-bool number or a numeric string, else None."""
    if isinstance(s, bool) or s is None:
        return None
    if isinstance(s, int | float):
        return float(s)
    if isinstance(s, str):
        t = s.strip()
        if not t or _FRAC.match(t):
            return None
        try:
            return float(t)
        except ValueError:
            return None
    return None


def _is_int(s: Any) -> bool:
    if isinstance(s, bool):
        return False
    if isinstance(s, int):
        return True
    return isinstance(s, str) and bool(_INT.match(s.strip()))


class Tol:
    """Tolerances: ``atol_db`` for dB quantities, ``atol_hz + rtol |ref|`` for frequencies
    (``*_hz``) when ``atol_hz`` is set, ``atol + rtol |ref|`` otherwise."""

    def __init__(
        self, rtol: float, atol: float, atol_db: float, atol_hz: float | None = None
    ) -> None:
        self.rtol, self.atol, self.atol_db, self.atol_hz = rtol, atol, atol_db, atol_hz

    def with_hz(self, atol_hz: float | None) -> Tol:
        return Tol(self.rtol, self.atol, self.atol_db, atol_hz)

    def ok(self, name: str, new: float, ref: float) -> bool:
        if math.isnan(new) or math.isnan(ref):
            return math.isnan(new) and math.isnan(ref)
        if math.isinf(new) or math.isinf(ref):
            return new == ref
        d = abs(new - ref)
        if "db" in name.lower():
            return d <= self.atol_db
        if self.atol_hz is not None and name.lower().endswith("_hz"):
            return d <= self.atol_hz + self.rtol * abs(ref)
        return d <= self.atol + self.rtol * abs(ref)


class ColumnStats:
    """Deviation statistics of one float column (or JSON key path family)."""

    def __init__(self) -> None:
        self.n = 0
        self.identical = 0
        self.failed = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.where_abs: str | None = None
        self.where_fail: str | None = None

    def add(self, new: float, ref: float, ok: bool, where: str) -> None:
        self.n += 1
        same = new == ref or (math.isnan(new) and math.isnan(ref))
        self.identical += int(same)
        if not ok:
            self.failed += 1
            self.where_fail = self.where_fail or where
        if not same and math.isfinite(new) and math.isfinite(ref):
            d = abs(new - ref)
            r = d / abs(ref) if ref != 0 else math.inf
            if d > self.max_abs:
                self.max_abs, self.where_abs = d, where
            self.max_rel = max(self.max_rel, r)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "bit_identical": self.identical,
            "failed": self.failed,
            "max_abs_dev": self.max_abs,
            "max_rel_dev": self.max_rel,
            "at": self.where_abs,
            "first_failure_at": self.where_fail,
        }


class FileResult:
    """Comparison result of one file. ``skip``: field names not compared."""

    def __init__(self, path: str, kind: str, skip: frozenset[str] = HOST_FIELDS) -> None:
        self.path = path
        self.kind = kind
        self.skip = skip
        self.rows_ref = 0
        self.rows_new = 0
        self.rows_compared = 0
        self.not_regenerated: list[str] = []
        self.unexpected: list[str] = []
        self.exact_mismatches: list[str] = []
        self.n_exact = 0
        self.floats: dict[str, ColumnStats] = {}
        self.not_compared: set[str] = set()
        self.notes: list[str] = []

    def skipped(self, name: str) -> bool:
        return name in self.skip or name.rsplit(".", 1)[-1] in self.skip

    def failed(self, smoke: bool) -> bool:
        return bool(
            self.exact_mismatches
            or self.unexpected
            or any(s.failed for s in self.floats.values())
            or (self.not_regenerated and not smoke)
        )

    def as_dict(self, smoke: bool) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": "FAIL" if self.failed(smoke) else "ok",
            "rows_ref": self.rows_ref,
            "rows_new": self.rows_new,
            "rows_compared": self.rows_compared,
            "exact_values_compared": self.n_exact,
            "exact_mismatches": len(self.exact_mismatches),
            "exact_mismatch_examples": self.exact_mismatches[:20],
            "not_regenerated": len(self.not_regenerated),
            "not_regenerated_examples": self.not_regenerated[:10],
            "unexpected_rows": len(self.unexpected),
            "unexpected_examples": self.unexpected[:10],
            "float_columns": {k: v.as_dict() for k, v in sorted(self.floats.items())},
            "not_compared": sorted(self.not_compared),
            "notes": self.notes,
        }


def compare_scalar(fr: FileResult, tol: Tol, name: str, new: Any, ref: Any, where: str) -> None:
    """Exact for ints, booleans, fractions and text; tolerance for other numbers."""
    a, b = _num(new), _num(ref)
    if a is not None and b is not None and not (_is_int(new) and _is_int(ref)):
        fr.floats.setdefault(name, ColumnStats()).add(a, b, tol.ok(name, a, b), where)
        return
    fr.n_exact += 1
    if _is_int(new) and _is_int(ref):
        same = int(str(new).strip()) == int(str(ref).strip())
    else:
        same = new == ref
    if not same:
        fr.exact_mismatches.append(f"{where} [{name}]: new {new!r} != ref {ref!r}")


def compare_tree(fr: FileResult, tol: Tol, new: Any, ref: Any, path: str, smoke: bool) -> None:
    """Recursive JSON comparison (dict keys by name, lists by position)."""
    leaf = path.rsplit(".", 1)[-1].split("[", 1)[0]
    if leaf in fr.skip:
        fr.not_compared.add(leaf)
        return
    if isinstance(ref, dict) and isinstance(new, dict):
        for k in ref:
            if k in fr.skip:
                fr.not_compared.add(k)
                continue
            if k not in new:
                (fr.not_regenerated if smoke else fr.exact_mismatches).append(
                    f"{path}.{k}: missing in new"
                )
                continue
            compare_tree(fr, tol, new[k], ref[k], f"{path}.{k}", smoke)
        for k in new:
            if k not in ref and k not in fr.skip:
                fr.notes.append(f"{path}.{k}: only in new")
        return
    if isinstance(ref, list) and isinstance(new, list):
        lk = LIST_KEYS.get(leaf)
        if lk and all(isinstance(x, dict) for x in [*ref, *new]):
            _compare_keyed(fr, tol, new, ref, path, lk, smoke)
            return
        if len(ref) != len(new):
            fr.exact_mismatches.append(f"{path}: list length new {len(new)} != ref {len(ref)}")
        for i, (a, b) in enumerate(zip(new, ref, strict=False)):
            compare_tree(fr, tol, a, b, f"{path}[{i}]", smoke)
        return
    if isinstance(ref, dict | list) or isinstance(new, dict | list):
        fr.n_exact += 1
        fr.exact_mismatches.append(f"{path}: structure differs")
        return
    compare_scalar(fr, tol, re.sub(r"\[\d+\]", "", path).split(".")[-1], new, ref, path)


def _compare_keyed(
    fr: FileResult,
    tol: Tol,
    new: list[dict[str, Any]],
    ref: list[dict[str, Any]],
    path: str,
    key: tuple[str, ...],
    smoke: bool,
) -> None:
    """Rows of a table copied into a JSON list (``summary.json``), matched by key."""

    def k_of(r: dict[str, Any]) -> str:
        return " | ".join(f"{c}={r.get(c, '')}" for c in key)

    ni = {k_of(r): r for r in new}
    for r in ref:
        k = k_of(r)
        if k not in ni:
            fr.not_regenerated.append(f"{path}: {k}")
            continue
        compare_tree(fr, tol, ni[k], r, f"{path}[{k}]", smoke)
    rk = {k_of(r) for r in ref}
    fr.unexpected += [f"{path}: {k}" for k in ni if k not in rk]


def _cell_json(v: str) -> Any:
    t = v.strip()
    if t[:1] in "[{":
        try:
            return json.loads(t)
        except ValueError:
            return v
    return v


# =========================================================================== file types


def _read_csv(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def csv_key(rel: str) -> tuple[str, ...] | None:
    """Row key of a result table (None: rows in order)."""
    if rel in CSV_KEYS:
        return CSV_KEYS[rel]
    if rel.endswith("/index.csv"):
        return ("point_id",)
    return None


def compare_csv(
    rel: str, new_p: Path, ref_p: Path, tol: Tol, smoke: bool, skip: frozenset[str] = HOST_FIELDS
) -> FileResult:
    fr = FileResult(rel, "csv", skip)
    ref, new = _read_csv(ref_p), _read_csv(new_p)
    fr.rows_ref, fr.rows_new = len(ref), len(new)
    key = csv_key(rel)
    cols_ref = list(ref[0]) if ref else []
    cols_new = list(new[0]) if new else []
    for c in cols_ref:
        if new and c not in cols_new and not fr.skipped(c):
            fr.exact_mismatches.append(f"column {c!r} missing in new")
    for c in cols_new:
        if ref and c not in cols_ref:
            fr.notes.append(f"column {c!r} only in new")

    def k_of(r: dict[str, str], i: int) -> str:
        return " | ".join(f"{c}={r.get(c, '')}" for c in key) if key else f"row {i}"

    def index(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for i, r in enumerate(rows):
            k = k_of(r, i)
            if k in out:
                fr.exact_mismatches.append(f"duplicate key in {label}: {k}")
            out[k] = r
        return out

    ri, ni = index(ref, "ref"), index(new, "new")
    for k, r in ri.items():
        n = ni.get(k)
        if n is None:
            fr.not_regenerated.append(k)
            continue
        fr.rows_compared += 1
        for c in cols_ref:
            if fr.skipped(c):
                fr.not_compared.add(c)
                continue
            if key and c in key:
                continue
            if c not in n:
                continue
            a, b = _cell_json(n[c]), _cell_json(r[c])
            if isinstance(a, dict | list) or isinstance(b, dict | list):
                compare_tree(fr, tol, a, b, f"{k} [{c}]", smoke)
            else:
                compare_scalar(fr, tol, c, a, b, k)
    fr.unexpected = [k for k in ni if k not in ri]
    return fr


def compare_json(
    rel: str, new_p: Path, ref_p: Path, tol: Tol, smoke: bool, skip: frozenset[str] = HOST_FIELDS
) -> FileResult:
    fr = FileResult(rel, "json", skip)
    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    new = json.loads(new_p.read_text(encoding="utf-8"))
    if rel == "summary.json":
        ref = {k: v for k, v in ref.items() if k != "meta"}
        new = {k: v for k, v in new.items() if k != "meta"}
        fr.not_compared.add("meta")
    compare_tree(fr, tol, new, ref, rel, smoke)
    return fr


def _meta_view(d: dict[str, Any], smoke: bool, rel: str = "") -> dict[str, Any]:
    """The compared part of a ``meta.json`` (see ``META_SKIP``, ``META_SKIP_EXTRA``)."""
    skip = META_SKIP | META_SKIP_EXTRA.get(rel, frozenset())
    out = {k: v for k, v in d.items() if k not in skip}
    inputs = out.get("inputs")
    if isinstance(inputs, list):
        out["inputs"] = [
            {
                k: v
                for k, v in i.items()
                if not (k == "sha256" and str(i.get("path", "")).endswith("manifest.json"))
            }
            if isinstance(i, dict)
            else i
            for i in inputs
        ]
    if isinstance(out.get("assumptions"), list):
        a = out["assumptions"]
        if smoke:
            a = [x for x in a if not str(x).startswith("SMOKE:")]
        if rel == "meta.json":  # E-F latency in or out (finalize --with-latency) is reported apart
            a = [x for x in a if not str(x).startswith(_LATENCY_NOTE)]
        out["assumptions"] = a
    if rel == "meta.json" and isinstance(out.get("experiments"), list):
        out["experiments"] = [e for e in out["experiments"] if e != LATENCY_DIR]
    return out


def compare_meta(
    rel: str, new_p: Path, ref_p: Path, tol: Tol, smoke: bool, skip: frozenset[str] = HOST_FIELDS
) -> FileResult:
    fr = FileResult(rel, "meta", skip)
    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    new = json.loads(new_p.read_text(encoding="utf-8"))
    fr.not_compared |= {k for k in ref if k in META_SKIP | META_SKIP_EXTRA.get(rel, frozenset())}
    compare_tree(fr, tol, _meta_view(new, smoke, rel), _meta_view(ref, smoke, rel), rel, smoke)
    return fr


def compare_text(rel: str, new_p: Path, ref_p: Path) -> FileResult:
    fr = FileResult(rel, "text")
    a = new_p.read_text(encoding="utf-8").replace("\r\n", "\n")
    b = ref_p.read_text(encoding="utf-8").replace("\r\n", "\n")
    fr.n_exact = 1
    if a != b:
        la, lb = a.splitlines(), b.splitlines()
        i = next((j for j, (x, y) in enumerate(zip(la, lb, strict=False)) if x != y), None)
        i = min(len(la), len(lb)) if i is None else i
        fr.exact_mismatches.append(f"text differs from line {i + 1}")
    return fr


def compare_file(
    rel: str, new_p: Path, ref_p: Path, tol: Tol, smoke: bool, skip: frozenset[str] = HOST_FIELDS
) -> FileResult:
    """Dispatch on the file type (``rel`` names the file in the report and selects keys)."""
    suf = Path(rel).suffix.lower()
    if suf == ".csv":
        return compare_csv(rel, new_p, ref_p, tol, smoke, skip)
    if Path(rel).name == "meta.json":
        return compare_meta(rel, new_p, ref_p, tol, smoke, skip)
    if suf == ".json":
        return compare_json(rel, new_p, ref_p, tol, smoke, skip)
    return compare_text(rel, new_p, ref_p)


# =========================================================================== provenance


def _json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def provenance(root: Path, validation: dict[str, Path] | None = None) -> dict[str, Any]:
    """Git state, code hash and host of every experiment of a results tree, of its E-B arc
    run and of the given validation run directories (study -> run dir)."""
    per: dict[str, Any] = {}
    for e in ("", *EXPERIMENTS):
        d = root / e if e else root
        if not d.is_dir():
            continue
        m = _json(d / "manifest.json")
        meta = _json(d / "meta.json")
        aws = m.get("aws_instance") or {}
        per[e or "(root)"] = {
            "git_commit": m.get("git_commit"),
            "git_dirty": m.get("git_dirty"),
            "dirty_patch_sha256": m.get("dirty_patch_sha256"),
            "src_sremu_tree_sha256": (meta.get("code") or {}).get("src_sremu_tree_sha256"),
            "cpu_model": m.get("cpu_model"),
            "platform": m.get("platform"),
            "aws_instance_type": aws.get("type") if isinstance(aws, dict) else None,
        }
    if (root / ARC / "git_state.json").is_file():
        gs = _json(root / ARC / "git_state.json")
        sh = _json(root / ARC / "run_manifests.json").get("shared") or {}
        aws = sh.get("aws_instance") or {}
        per[ARC] = {
            "git_commit": gs.get("commit"),
            "git_dirty": gs.get("dirty"),
            "dirty_patch_sha256": gs.get("dirty_patch_sha256"),
            "src_sremu_tree_sha256": (gs.get("code") or {}).get("src_sremu_tree_sha256"),
            "manifests_git_commit": sh.get("git_commit"),
            "cpu_model": sh.get("cpu_model"),
            "platform": sh.get("platform"),
            "aws_instance_type": aws.get("type") if isinstance(aws, dict) else None,
        }
    for study, d in (validation or {}).items():
        meta = _json(d / "meta.json")
        per[study] = {
            "git_commit": meta.get("sremu_commit"),
            "git_dirty": meta.get("sremu_dirty"),
            "dirty_patch_sha256": meta.get("dirty_patch_sha256"),
            "src_sremu_tree_sha256": None,  # study runs record no code hash
            "run_dir": str(d),
            "cpu_model": meta.get("cpu_model"),
            "platform": meta.get("platform"),
        }
    va1 = root / "E-A-IQ" / "va1.csv"
    rows = _read_csv(va1) if va1.is_file() else []
    va1_states = sorted(
        {(r.get("git_commit") or "", r.get("git_dirty") or "", r.get("source") or "") for r in rows}
    )
    commits = {v["git_commit"] for v in per.values()} | {s[0] or None for s in va1_states}
    commits |= {v["manifests_git_commit"] for v in per.values() if "manifests_git_commit" in v}
    dirty = {v["git_dirty"] for v in per.values()} | {
        {"True": True, "False": False}.get(s[1]) for s in va1_states
    }
    trees = {v["src_sremu_tree_sha256"] for k, v in per.items() if k not in (validation or {})}
    pinned = (
        bool(per)
        and len(commits) == 1
        and None not in commits
        and dirty == {False}
        and len(trees) == 1
        and None not in trees
    )
    return {
        "pinned": pinned,
        "commits": sorted(str(c) for c in commits),
        "dirty_flags": sorted(str(d) for d in dirty),
        "src_sremu_tree_sha256": sorted(str(t) for t in trees),
        "va1_row_states": [
            {"git_commit": a or None, "git_dirty": b or None, "source": c} for a, b, c in va1_states
        ],
        "per_experiment": per,
    }


# =========================================================================== code check


def git_head(repo: Path) -> str | None:
    """HEAD commit of ``repo`` from its ``.git`` files (no git process; a worktree's ``.git``
    file and ``packed-refs`` are followed)."""
    g = repo / ".git"
    try:
        if g.is_file():
            g = Path(g.read_text(encoding="utf-8").split(":", 1)[1].strip())
        head = (g / "HEAD").read_text(encoding="utf-8").strip()
    except (OSError, IndexError):
        return None
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    common = g
    if (g / "commondir").is_file():
        common = (g / (g / "commondir").read_text(encoding="utf-8").strip()).resolve()
    for base in (g, common):
        if (base / ref).is_file():
            return (base / ref).read_text(encoding="utf-8").strip()
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref:
                return parts[0]
    return None


def code_hashes(repo: Path) -> dict[str, Any]:
    """``experiments/common.py:code_hashes`` of the checkout ``repo`` (standard library only;
    the same files and the same combined ``src/sremu`` hash), without the paper's spec files."""
    files: dict[str, str] = {}
    for pat in ("experiments/*.py", "scenarios/studies/*", "scenarios/lab_like_A.yaml"):
        for p in sorted(repo.glob(pat)):
            if p.is_file():
                files[p.relative_to(repo).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    src = hashlib.sha256()
    for p in sorted((repo / "src" / "sremu").rglob("*.py")):
        src.update(p.relative_to(repo).as_posix().encode())
        src.update(bytes(1))
        src.update(p.read_bytes())
    return {"files": files, "src_sremu_tree_sha256": src.hexdigest()}


def validation_code_hashes(repo: Path) -> dict[str, str]:
    """SHA-256 of the validation-study code and protocol files (``validation/**``, run
    directories, caches and reports excluded); the job records the same in
    ``_regen/provenance.json`` ``validation_code``."""
    out: dict[str, str] = {}
    v = repo / "validation"
    for p in sorted(v.rglob("*")):
        rel = p.relative_to(repo).as_posix()
        if not p.is_file() or "/results/" in rel or "__pycache__" in rel or p.name == "report.md":
            continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def code_check(job_out: Path, repo: Path) -> dict[str, Any]:
    """The job's git state and code hashes against the local checkout ``repo``."""
    gs = _json(job_out / "_regen" / "git_state.json")
    prov = _json(job_out / "_regen" / "provenance.json")
    if not gs:
        return {"performed": False, "ok": None, "reason": f"{job_out / '_regen'} has no git state"}
    head = git_head(repo)
    local = code_hashes(repo)
    job_code = gs.get("code") or {}
    job_files = {
        k: v for k, v in (job_code.get("files") or {}).items() if not k.startswith("paper/")
    }
    differ = sorted(
        k for k in set(job_files) | set(local["files"]) if job_files.get(k) != local["files"].get(k)
    )
    job_val = prov.get("validation_code")
    loc_val = validation_code_hashes(repo)
    val_differ = (
        None
        if job_val is None
        else sorted(k for k in set(job_val) | set(loc_val) if job_val.get(k) != loc_val.get(k))
    )
    res = {
        "performed": True,
        "sremu_root": str(repo),
        "job_commit": gs.get("commit"),
        "job_dirty": gs.get("dirty"),
        "local_head": head,
        "head_equals_job_commit": head is not None and head == gs.get("commit"),
        "src_sremu_tree_sha256_job": job_code.get("src_sremu_tree_sha256"),
        "src_sremu_tree_sha256_local": local["src_sremu_tree_sha256"],
        "src_equal": job_code.get("src_sremu_tree_sha256") == local["src_sremu_tree_sha256"],
        "files_compared": len(set(job_files) | set(local["files"])),
        "files_differing": differ,
        "validation_code_compared": None if job_val is None else len(set(job_val) | set(loc_val)),
        "validation_code_differing": val_differ,
        "clean_tree": "not checked here (no git process): run git status in the checkout",
    }
    res["ok"] = bool(
        res["head_equals_job_commit"]
        and res["src_equal"]
        and not differ
        and val_differ is not None
        and not val_differ
    )
    return res


# =========================================================================== driver


def in_scope(rel: str) -> bool:
    """Files this regeneration produces: the root files, the experiment directories of
    ``EXPERIMENTS`` and the E-B arc results (not its ``host/`` records). ``E-F/latency/``
    (MEAS, collected only with ``finalize --with-latency``) is checked apart."""
    if rel.startswith(f"{ARC}/host/"):
        return False
    return "/" not in rel or any(rel.startswith(e + "/") for e in (*EXPERIMENTS, ARC))


def _files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name not in PROVENANCE_FILES and p.suffix != ".patch"
    )


def validation_refs(paper_tools: Path) -> tuple[dict[str, str], str]:
    """Reference run directories (sremu-relative) of the validation studies: the constants of
    ``paper/tools/gen_results.py`` when it is readable, else ``VALIDATION_FALLBACK``."""
    p = paper_tools / "gen_results.py"
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return dict(VALIDATION_FALLBACK), "fallback (gen_results.py not found)"
    out = {}
    for study, (_sub, _name, const) in VALIDATION_STUDIES.items():
        m = re.search(rf'^{const} = "([^"]+)"', text, re.M)
        out[study] = m.group(1) if m else VALIDATION_FALLBACK[study]
    return out, str(p)


def find_validation_runs(new_val: Path) -> dict[str, Path]:
    """The regenerated run directory of every study under ``new_val`` (one per study)."""
    out = {}
    for study, (sub, name, _c) in VALIDATION_STUDIES.items():
        runs = sorted(
            p
            for p in (new_val / sub).glob(f"*_{name}*")
            if p.is_dir() and re.fullmatch(rf"\d{{8}}T\d{{6}}Z_{name}(_fast)?", p.name)
        )
        if runs:
            out[study] = runs[-1]
    return out


def compare_validation(
    runs: dict[str, Path], refs: dict[str, str], repo: Path, tol: Tol, smoke: bool
) -> tuple[dict[str, FileResult], list[str], list[str]]:
    """Every regenerated validation run against the run directory the paper reads."""
    files: dict[str, FileResult] = {}
    missing: list[str] = []
    extra: list[str] = []
    skip = HOST_FIELDS | RUN_FIELDS | DERIVED_TEXT
    for study in VALIDATION_STUDIES:
        ref_dir = repo / refs[study]
        if study not in runs:
            missing.append(f"{study}/ (no regenerated run)")
            continue
        new_dir = runs[study]
        if not ref_dir.is_dir():
            missing.append(f"{study}/ (reference {ref_dir} not found)")
            continue
        in_hz_scope = any(f"{study}/".startswith(s) for s in ATOL_HZ_SCOPE)
        t = tol.with_hz(tol.atol_hz if in_hz_scope else None)
        ref_files, new_files = set(_files(ref_dir)), set(_files(new_dir))
        for rel in sorted(ref_files):
            key = f"{study}/{rel}"
            if rel not in new_files:
                missing.append(key)
                continue
            files[key] = compare_file(key, new_dir / rel, ref_dir / rel, t, smoke, skip)
        extra += [f"{study}/{rel}" for rel in sorted(new_files - ref_files)]
    return files, missing, extra


def latency_check(new_root: Path) -> dict[str, Any] | None:
    """E-F latency, when collected: MEAS only, with its host record (values not compared)."""
    d = new_root / LATENCY_DIR
    if not d.is_dir():
        return None
    meta, man = _json(d / "meta.json"), _json(d / "manifest.json")
    aws = man.get("aws_instance") or {}
    res = {
        "data_classes": meta.get("data_classes"),
        "cpu_model": man.get("cpu_model"),
        "platform": man.get("platform"),
        "aws_instance_type": aws.get("type") if isinstance(aws, dict) else None,
        "values_compared": False,
        "note": "MEAS of the recording host only; no reference values (host-specific)",
    }
    res["ok"] = meta.get("data_classes") == ["MEAS"] and bool(man.get("cpu_model"))
    return res


def compare(
    new_root: Path,
    ref_root: Path,
    tol: Tol,
    smoke: bool,
    *,
    validation_new: Path | None = None,
    repo: Path | None = SREMU_ROOT,
    code: bool = True,
) -> dict[str, Any]:
    files: dict[str, FileResult] = {}
    missing: list[str] = []
    all_ref = _files(ref_root)
    ref_files = [r for r in all_ref if in_scope(r)]
    out_of_scope = sorted(
        {
            f"{ARC}/host/" if r.startswith(f"{ARC}/host/") else r.split("/")[0] + "/"
            for r in all_ref
            if not in_scope(r)
        }
    )
    new_files = set(_files(new_root))
    base = tol.with_hz(None)  # atol_hz is for the V-CFO frequencies only
    for rel in ref_files:
        if rel not in new_files:
            missing.append(rel)
            continue
        skip = HOST_FIELDS | RUN_FIELDS if rel.startswith(f"{ARC}/") else HOST_FIELDS
        files[rel] = compare_file(rel, new_root / rel, ref_root / rel, base, smoke, skip)
    extra = sorted(new_files - set(ref_files))
    job_out = new_root.parent
    val_root = validation_new if validation_new is not None else job_out / "validation"
    val: dict[str, Any] = {"new": str(val_root), "performed": val_root.is_dir()}
    runs: dict[str, Path] = {}
    if val_root.is_dir() and repo is not None:
        refs, ref_src = validation_refs(ref_root.parents[1] / "tools")
        runs = find_validation_runs(val_root)
        vfiles, vmissing, vextra = compare_validation(runs, refs, repo, tol, smoke)
        files.update(vfiles)
        missing += vmissing
        extra += vextra
        val.update(
            {
                "reference_runs": refs,
                "reference_source": ref_src,
                "new_runs": {k: str(v) for k, v in runs.items()},
            }
        )
    elif not smoke:
        missing.append(f"validation runs ({val_root} not found)")
    prov_new, prov_ref = provenance(new_root, runs), provenance(ref_root)
    cc = code_check(job_out, repo) if (code and repo is not None) else {"performed": False}
    vlab = _json(job_out / "_regen" / "vlab_check.json") or None
    lat = latency_check(new_root)
    failed = [r for r, f in files.items() if f.failed(smoke)]
    new_is_smoke = bool((_json(new_root / "summary.json").get("meta") or {}).get("smoke"))
    ok = (
        not failed
        and not missing
        and (prov_new["pinned"] or smoke)
        and (smoke or not new_is_smoke)  # a smoke tree never passes a full comparison
        and (smoke or not cc.get("performed") or bool(cc.get("ok")))
        and (smoke or (vlab is not None and bool(vlab.get("ok"))))  # a missing record fails too
        and (lat is None or bool(lat["ok"]))
    )
    return {
        "new": str(new_root),
        "ref": str(ref_root),
        "mode": "smoke" if smoke else "full",
        "new_is_smoke_tree": new_is_smoke,
        "tolerances": {
            "rtol": tol.rtol,
            "atol": tol.atol,
            "atol_db": tol.atol_db,
            "atol_hz": tol.atol_hz,
            "atol_hz_scope": list(ATOL_HZ_SCOPE),
        },
        "ok": ok,
        "failed_files": failed,
        "missing_files": missing,
        "extra_files": extra,
        "ref_out_of_scope": out_of_scope,
        "validation": val,
        "files": {r: f.as_dict(smoke) for r, f in files.items()},
        "provenance": {"new": prov_new, "ref": prov_ref},
        "code_check": cc,
        "vlab_check": vlab,
        "latency": lat,
    }


def _fmt(x: float) -> str:
    return "0" if x == 0 else f"{x:.3g}"


def print_report(rep: dict[str, Any]) -> None:
    print(f"compare {rep['new']}\n   vs   {rep['ref']}   ({rep['mode']})")
    t = rep["tolerances"]
    print(
        f"tolerances: dB quantities |d| <= {t['atol_db']:g} dB; V-CFO *_hz |d| <= "
        f"{t['atol_hz']:g} Hz + {t['rtol']:g} |ref|; others |d| <= {t['atol']:g} + "
        f"{t['rtol']:g} |ref|; ints, booleans, fractions, labels exact"
    )
    for rel, f in rep["files"].items():
        print(
            f"\n[{f['status']}] {rel}: rows ref {f['rows_ref']} new {f['rows_new']} compared "
            f"{f['rows_compared']}; exact values {f['exact_values_compared']} "
            f"(mismatches {f['exact_mismatches']}); not regenerated {f['not_regenerated']}; "
            f"unexpected {f['unexpected_rows']}"
        )
        for m in f["exact_mismatch_examples"][:5]:
            print(f"    exact: {m}")
        for u in f["unexpected_examples"][:3]:
            print(f"    unexpected row: {u}")
        for c, s in f["float_columns"].items():
            flag = "FAIL" if s["failed"] else "ok"
            print(
                f"    {flag:4} {c}: n {s['n']}, bit-identical {s['bit_identical']}, max |d| "
                f"{_fmt(s['max_abs_dev'])}, max rel {_fmt(s['max_rel_dev'])}"
                + (f" at {s['at']}" if s["max_abs_dev"] else "")
            )
    if rep["missing_files"]:
        print(f"\nmissing in new: {rep['missing_files']}")
    if rep["extra_files"]:
        print(f"only in new: {rep['extra_files']}")
    if rep["ref_out_of_scope"]:
        print(f"reference entries out of scope (not regenerated here): {rep['ref_out_of_scope']}")
    v = rep["validation"]
    if v.get("reference_runs"):
        print(f"\nvalidation references ({v['reference_source']}): {v['reference_runs']}")
        print(f"validation regenerated: {v['new_runs']}")
    for side in ("new", "ref"):
        p = rep["provenance"][side]
        print(
            f"\nprovenance {side}: pinned {p['pinned']}; commits {p['commits']}; dirty "
            f"{p['dirty_flags']}; src trees {[s[:12] for s in p['src_sremu_tree_sha256']]}"
        )
    cc = rep["code_check"]
    if cc.get("performed"):
        print(
            f"\ncode check vs {cc['sremu_root']}: ok {cc['ok']}; job commit {cc['job_commit']} "
            f"(dirty {cc['job_dirty']}), local HEAD {cc['local_head']}; src tree equal "
            f"{cc['src_equal']}; differing files {cc['files_differing'][:5]}; differing "
            f"validation files {(cc['validation_code_differing'] or [])[:5]}; {cc['clean_tree']}"
        )
    else:
        print(f"\ncode check not performed ({cc.get('reason', 'disabled')})")
    if rep["vlab_check"] is not None:
        vl = rep["vlab_check"]
        print(f"V-LAB check (job): ok {vl.get('ok')}; {vl.get('summary')}")
    else:
        print("V-LAB check (job): no _regen/vlab_check.json (a full comparison fails)")
    if rep["latency"] is not None:
        print(f"E-F latency (MEAS, values not compared): {rep['latency']}")
    if rep["new_is_smoke_tree"]:
        print("\nNEW is a smoke tree (summary.json meta.smoke = true): never paper data")
    print(
        f"\nRESULT: {'PASS' if rep['ok'] else 'FAIL'}"
        + (f" (failed: {rep['failed_files']})" if rep["failed_files"] else "")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("new", type=Path, help="regenerated results root (<job out>/paper_results)")
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF, help="reference results root")
    ap.add_argument("--json", type=Path, default=None, help="write the full report here")
    ap.add_argument("--smoke", action="store_true", help="subset comparison of a smoke run")
    ap.add_argument(
        "--validation-new",
        type=Path,
        default=None,
        help="regenerated validation runs (default <NEW>/../validation)",
    )
    ap.add_argument(
        "--sremu-root",
        type=Path,
        default=SREMU_ROOT,
        help="sremu checkout holding the reference validation runs and checked against the "
        "job's code (default: this repository)",
    )
    ap.add_argument("--no-code-check", action="store_true", help="skip the code check")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--atol-db", type=float, default=1e-9)
    ap.add_argument("--atol-hz", type=float, default=1e-7)
    a = ap.parse_args(argv)
    if not a.new.is_dir() or not a.ref.is_dir():
        print(f"not a directory: {a.new if not a.new.is_dir() else a.ref}", file=sys.stderr)
        return 2
    rep = compare(
        a.new.resolve(),
        a.ref.resolve(),
        Tol(a.rtol, a.atol, a.atol_db, a.atol_hz),
        a.smoke,
        validation_new=a.validation_new.resolve() if a.validation_new else None,
        repo=a.sremu_root.resolve(),
        code=not a.no_code_check,
    )
    if a.json is not None:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        with a.json.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rep, indent=1, default=str) + "\n")
    print_report(rep)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
