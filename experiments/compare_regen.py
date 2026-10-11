"""Compare a regenerated WP11 results tree with the paper's committed results tree.

    python experiments/compare_regen.py NEW [--ref REF] [--json REPORT.json] [--smoke]
        [--rtol 1e-9] [--atol 1e-12] [--atol-db 1e-9]

``NEW`` is ``<job out>/paper_results`` of ``experiments/regenerate_paper_data.sh``; ``REF``
defaults to ``<sremu>/../paper/sweeps/results``. Run it from anywhere; it needs only the
Python standard library.

What is compared, and how:

* **Exact** (deterministic quantities and decisions): row keys, integers (bit and packet
  error counts, command counts, sample indices), booleans (acceptance and CI decisions),
  exact fractions (``1/2000``), labels, and decision strings of ``hypotheses.json`` and
  ``va5.json`` (``supported``, ``pass``, ...).
* **Tolerance** (float aggregates): every other number. A value passes when
  ``|new - ref| <= atol_db`` for a dB quantity (column or key name contains ``db``) and
  ``|new - ref| <= atol + rtol |ref|`` otherwise. Max absolute and relative deviations and
  the number of bit-identical values are reported per column, whether or not they pass.
* **meta.json**, in part: data classes, descriptions, assumptions (the paper reads data
  classes and some setup numbers from them), inputs and the deterministic checks, with the
  rules above; file hashes, code hashes, times and local paths are not compared
  (``META_SKIP``).
* **Not compared** (host- or provenance-specific): wall times, real-time factors, CPU model,
  platform, instance type, data source, git fields and code hashes in rows;
  ``manifest.json`` and ``git_state.json``. The provenance of both trees is reported
  separately: a regenerated tree is *pinned* when every manifest and VA-1 row names the same
  commit, none is dirty, and every ``meta.json`` names the same ``src/sremu`` tree hash.
  A smoke tree (``summary.json`` ``meta.smoke``) never passes a full comparison.

``--smoke`` is for the reduced smoke run: reference rows absent from NEW are counted as "not
regenerated" (not failures), and an unpinned NEW tree is a warning, not a failure. Rows
present in both trees are compared with the same rules as a full run.

Exit status: 0 when every comparison passes (and NEW is pinned, unless ``--smoke``), 1
otherwise, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
DEFAULT_REF = HERE.parents[2] / "paper" / "sweeps" / "results"

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
"""Row keys of the known result tables (an unknown CSV is compared row by row in order)."""

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

PROVENANCE_FILES = frozenset({"manifest.json", "git_state.json"})
"""Files not compared at all (git state, host, times). ``meta.json`` is compared in part."""

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
EXPERIMENTS = ("E-A", "E-A-IQ", "E-B1", "E-B2", "E-C", "E-F/ber")

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
    """Tolerances: ``atol_db`` for dB quantities, ``atol + rtol |ref|`` otherwise."""

    def __init__(self, rtol: float, atol: float, atol_db: float) -> None:
        self.rtol, self.atol, self.atol_db = rtol, atol, atol_db

    def ok(self, name: str, new: float, ref: float) -> bool:
        if math.isnan(new) or math.isnan(ref):
            return math.isnan(new) and math.isnan(ref)
        if math.isinf(new) or math.isinf(ref):
            return new == ref
        d = abs(new - ref)
        if "db" in name.lower():
            return d <= self.atol_db
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
    """Comparison result of one file."""

    def __init__(self, path: str, kind: str) -> None:
        self.path = path
        self.kind = kind
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
    if leaf in HOST_FIELDS:
        fr.not_compared.add(leaf)
        return
    if isinstance(ref, dict) and isinstance(new, dict):
        for k in ref:
            if k in HOST_FIELDS:
                fr.not_compared.add(k)
                continue
            if k not in new:
                (fr.not_regenerated if smoke else fr.exact_mismatches).append(
                    f"{path}.{k}: missing in new"
                )
                continue
            compare_tree(fr, tol, new[k], ref[k], f"{path}.{k}", smoke)
        for k in new:
            if k not in ref and k not in HOST_FIELDS:
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


def compare_csv(rel: str, new_p: Path, ref_p: Path, tol: Tol, smoke: bool) -> FileResult:
    fr = FileResult(rel, "csv")
    ref, new = _read_csv(ref_p), _read_csv(new_p)
    fr.rows_ref, fr.rows_new = len(ref), len(new)
    key = CSV_KEYS.get(rel)
    cols_ref = list(ref[0]) if ref else []
    cols_new = list(new[0]) if new else []
    for c in cols_ref:
        if new and c not in cols_new and c not in HOST_FIELDS:
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
            if c in HOST_FIELDS:
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


def compare_json(rel: str, new_p: Path, ref_p: Path, tol: Tol, smoke: bool) -> FileResult:
    fr = FileResult(rel, "json")
    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    new = json.loads(new_p.read_text(encoding="utf-8"))
    if rel == "summary.json":
        ref = {k: v for k, v in ref.items() if k != "meta"}
        new = {k: v for k, v in new.items() if k != "meta"}
        fr.not_compared.add("meta")
    compare_tree(fr, tol, new, ref, rel, smoke)
    return fr


def _meta_view(d: dict[str, Any], smoke: bool) -> dict[str, Any]:
    """The compared part of a ``meta.json`` (see ``META_SKIP``)."""
    out = {k: v for k, v in d.items() if k not in META_SKIP}
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
    if smoke and isinstance(out.get("assumptions"), list):
        out["assumptions"] = [a for a in out["assumptions"] if not str(a).startswith("SMOKE:")]
    return out


def compare_meta(rel: str, new_p: Path, ref_p: Path, tol: Tol, smoke: bool) -> FileResult:
    fr = FileResult(rel, "meta")
    ref = json.loads(ref_p.read_text(encoding="utf-8"))
    new = json.loads(new_p.read_text(encoding="utf-8"))
    fr.not_compared |= {k for k in ref if k in META_SKIP}
    compare_tree(fr, tol, _meta_view(new, smoke), _meta_view(ref, smoke), rel, smoke)
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


# =========================================================================== provenance


def _json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def provenance(root: Path) -> dict[str, Any]:
    """Git state, code hash and host of every experiment of a results tree."""
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
    va1 = root / "E-A-IQ" / "va1.csv"
    rows = _read_csv(va1) if va1.is_file() else []
    va1_states = sorted(
        {(r.get("git_commit") or "", r.get("git_dirty") or "", r.get("source") or "") for r in rows}
    )
    commits = {v["git_commit"] for v in per.values()} | {s[0] or None for s in va1_states}
    dirty = {v["git_dirty"] for v in per.values()} | {
        {"True": True, "False": False}.get(s[1]) for s in va1_states
    }
    trees = {v["src_sremu_tree_sha256"] for v in per.values()}
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


# =========================================================================== driver


def in_scope(rel: str) -> bool:
    """Files this regeneration produces: the root ``summary.json`` and the experiment
    directories of ``EXPERIMENTS`` (other studies in the tree, e.g. ``E-B-arc/``, and the
    not-collected ``E-F/latency/`` are out of scope)."""
    return "/" not in rel or any(rel.startswith(e + "/") for e in EXPERIMENTS)


def _files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name not in PROVENANCE_FILES
    )


def compare(new_root: Path, ref_root: Path, tol: Tol, smoke: bool) -> dict[str, Any]:
    files: dict[str, FileResult] = {}
    missing: list[str] = []
    all_ref = _files(ref_root)
    ref_files = [r for r in all_ref if in_scope(r)]
    out_of_scope = sorted({r.split("/")[0] + "/" for r in all_ref if not in_scope(r)})
    new_files = set(_files(new_root))
    for rel in ref_files:
        if rel not in new_files:
            missing.append(rel)
            continue
        a, b = new_root / rel, ref_root / rel
        suf = Path(rel).suffix.lower()
        if suf == ".csv":
            files[rel] = compare_csv(rel, a, b, tol, smoke)
        elif Path(rel).name == "meta.json":
            files[rel] = compare_meta(rel, a, b, tol, smoke)
        elif suf == ".json":
            files[rel] = compare_json(rel, a, b, tol, smoke)
        else:
            files[rel] = compare_text(rel, a, b)
    extra = sorted(new_files - set(ref_files))
    prov_new, prov_ref = provenance(new_root), provenance(ref_root)
    failed = [r for r, f in files.items() if f.failed(smoke)]
    new_is_smoke = bool((_json(new_root / "summary.json").get("meta") or {}).get("smoke"))
    ok = (
        not failed
        and not missing
        and (prov_new["pinned"] or smoke)
        and (smoke or not new_is_smoke)  # a smoke tree never passes a full comparison
    )
    return {
        "new": str(new_root),
        "ref": str(ref_root),
        "mode": "smoke" if smoke else "full",
        "new_is_smoke_tree": new_is_smoke,
        "tolerances": {"rtol": tol.rtol, "atol": tol.atol, "atol_db": tol.atol_db},
        "ok": ok,
        "failed_files": failed,
        "missing_files": missing,
        "extra_files": extra,
        "ref_out_of_scope": out_of_scope,
        "files": {r: f.as_dict(smoke) for r, f in files.items()},
        "provenance": {"new": prov_new, "ref": prov_ref},
    }


def _fmt(x: float) -> str:
    return "0" if x == 0 else f"{x:.3g}"


def print_report(rep: dict[str, Any]) -> None:
    print(f"compare {rep['new']}\n   vs   {rep['ref']}   ({rep['mode']})")
    t = rep["tolerances"]
    print(
        f"tolerances: dB quantities |d| <= {t['atol_db']:g} dB; others |d| <= "
        f"{t['atol']:g} + {t['rtol']:g} |ref|; ints, booleans, fractions, labels exact"
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
    for side in ("new", "ref"):
        p = rep["provenance"][side]
        print(
            f"\nprovenance {side}: pinned {p['pinned']}; commits {p['commits']}; dirty "
            f"{p['dirty_flags']}; src trees {[s[:12] for s in p['src_sremu_tree_sha256']]}"
        )
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
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("--atol-db", type=float, default=1e-9)
    a = ap.parse_args(argv)
    if not a.new.is_dir() or not a.ref.is_dir():
        print(f"not a directory: {a.new if not a.new.is_dir() else a.ref}", file=sys.stderr)
        return 2
    rep = compare(a.new.resolve(), a.ref.resolve(), Tol(a.rtol, a.atol, a.atol_db), a.smoke)
    if a.json is not None:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    print_report(rep)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
