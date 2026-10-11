"""Collect the WP11 results into ``paper/sweeps/results/`` and write ``summary.json``.

    python -m experiments.finalize [--runs runs/wp11]
    python -m experiments.finalize --import-aws-ea-iq <AWS job results dir> [--note TEXT]

* ``E-A/``: written by ``experiments.run_sweeps`` (``ea.aggregate``); read here, not rewritten;
* ``E-A-IQ/``: VA-1 table from ``<runs>/ea_iq`` (:func:`experiments.ea_iq.aggregate`, strict:
  every (controller, tau_SUI, noise) row of the design must be present, else this script
  fails before anything reaches the paper);
* ``E-B1/``, ``E-B2/``: re-aggregated from the original lab-like sweeps ``<runs>/eb1``,
  ``<runs>/eb2_lab`` and ``<runs>/eb2_ideal1`` (``scenarios/lab_like_A.yaml``) with
  :mod:`experiments.eb_aggregate`, which writes the D24 labels (the untimed baseline is
  ``tick-based``: tick-based untimed semantics); a point of another geometry fails the run;
* ``E-C/``: the E-C run directory (Table VIII data) copied as is;
* ``E-F/ber/``: BER table (:func:`experiments.ef.aggregate_ber`);
* E-F API latency is **not** collected (a transport latency fix is pending and the latency
  will be re-measured); a stale ``E-F/latency/`` is removed;
* D24 check: no ``legacy``/``v1``/``v2`` naming in any paper-facing result file (the RNG
  scheme id ``philox4x32-10/blake2b-v1`` excepted); a violation fails the run;
* ``summary.json``: the key numbers of every experiment (machine-readable, for the paper
  pipeline's macros; every value is copied from the result files, nothing is typed) and a
  ``meta`` block (data classes, git state, manifest and file SHA-256 per experiment);
  ``meta.json`` and ``manifest.json`` of this step next to it.

``--import-aws-ea-iq`` first copies the E-A IQ run directories of a lead AWS job
(``<job>/out/ea_iq/*``, see ``research/aws/runner.py``) into ``<runs>/ea_iq`` with a
``provenance.json`` per directory (job id, launch record, instance, CPU, OS, bundle SHA-256,
file hashes, the local directory it replaced and, for batch references, a cross-host check
against the replaced local result); the job-level files go to ``<runs>/ea_iq/_aws_<job>/``
and replaced local directories are moved (never deleted) to ``<runs>/ea_iq_replaced_<UTC>/``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.common import (  # noqa: E402
    PAPER,
    RunRecord,
    export_git_info,
    git_info,
    read_json,
    write_json,
    write_meta,
)

RESULTS = PAPER / "sweeps" / "results"
EXPERIMENT_DIRS = ("E-A", "E-A-IQ", "E-B1", "E-B2", "E-C", "E-F/ber")
"""Result directories collected into ``summary.json`` (each with meta.json + manifest.json)."""

EXCLUDED = {
    "E-F/latency": "E-F API latency is not reported: a transport latency fix is pending and the "
    "latency will be re-measured",
}
LABELS = {
    "tick-based": "tick-based untimed semantics (E-B2 baseline: next 2 ms tick after "
    "arrival, one pending slot per panel)"
}

_D24_PATTERN = re.compile(r"legacy|\bv[12]\b", re.IGNORECASE)
_D24_ALLOWED = ("philox4x32-10/blake2b-v1",)  # RNG scheme id (DESIGN.md 6.1), not a product version
_D24_SUFFIXES = (".csv", ".json", ".yaml", ".yml", ".txt", ".md", ".tex")
_LAB_LIKE_SCENARIO = "lab_like_A.yaml"


# =========================================================================== helpers


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree_hashes(d: Path) -> dict[str, str]:
    return {f.relative_to(d).as_posix(): _sha256(f) for f in sorted(d.rglob("*")) if f.is_file()}


def _copy_run(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.patch", "*.tmp"))


def _rows(p: Path) -> list[dict[str, str]]:
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _utc() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# =========================================================================== AWS import


def _cross_host_check(local: Path, imported: Path) -> dict[str, Any]:
    """Compare a replaced local batch reference with the imported one (same code path, other
    host): every numeric field and whether it is bit-identical."""
    a = read_json(local / "result.json")
    b = read_json(imported / "result.json")
    keys = ("p_batch_dbm", "p_vectorized_mid_dbm", "p_vectorized_left_dbm", "pieces", "n_commands")
    fields = {
        k: {"local": a.get(k), "imported": b.get(k), "equal": a.get(k) == b.get(k)} for k in keys
    }
    lm = local / "manifest.json"
    return {
        "local_cpu_model": read_json(lm).get("cpu_model") if lm.is_file() else None,
        "local_platform": read_json(lm).get("platform") if lm.is_file() else None,
        "fields": fields,
        "all_equal": all(v["equal"] for v in fields.values()),
    }


def import_aws_ea_iq(
    job_dir: str | Path, dest: str | Path, *, note: str | None = None
) -> dict[str, Any]:
    """Copy the E-A IQ run directories of a lead AWS job into ``dest`` with provenance.

    ``job_dir`` is a fetched job directory of ``research/aws/runner.py``
    (``<results>/<run_id>/out/...``); the launch record is read from ``launches.jsonl`` two
    levels up when present. Returns the job record written to ``dest/_aws_<run_id>/job.json``.
    """
    job_dir, dest = Path(job_dir).resolve(), Path(dest).resolve()
    out = job_dir / "out"
    src = out / "ea_iq"
    if not src.is_dir():
        raise FileNotFoundError(f"{src} is not a directory")
    run_id = job_dir.name
    launch = None
    lp = job_dir.parent.parent / "launches.jsonl"
    if lp.is_file():
        for line in lp.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("run_id") == run_id:
                launch = json.loads(line)

    def text(name: str) -> str:
        p = out / name
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""

    m_exit = re.search(r"^JOB_EXIT=(\d+)", text("job_stdout.txt"), re.M)
    m_cpu = re.search(r"^Model name:\s*(.+?)\s*$", text("lscpu.txt"), re.M)
    job_files = {
        n: _sha256(out / n)
        for n in ("job_stdout.txt", "lscpu.txt", "meminfo.txt", "sremu-job.log")
        if (out / n).is_file()
    }
    if (job_dir / "results.tgz").is_file():
        job_files["../results.tgz"] = _sha256(job_dir / "results.tgz")
    job: dict[str, Any] = {
        "run_id": run_id,
        "origin": "lead AWS job (research/aws/runner.py): bundle of the sremu repo, outputs "
        "under $SREMU_JOB_OUT, returned as results.tgz",
        "source_dir": str(job_dir),
        "launch": launch,
        "region": "ap-south-1" if launch else None,
        "job_exit": int(m_exit.group(1)) if m_exit else None,
        "cpu_model_lscpu": m_cpu.group(1) if m_cpu else None,
        "os": "Ubuntu 24.04 LTS (noble)"
        if re.search(r"\bnoble\b", text("sremu-job.log"))
        else None,
        "job_files_sha256": job_files,
        "imported_utc": _utc(),
        "note": note,
    }
    side = dest / f"_aws_{run_id}"
    side.mkdir(parents=True, exist_ok=True)
    prev = read_json(side / "job.json") if (side / "job.json").is_file() else {}
    prev_entries = (
        {e["dir"]: e for e in prev.get("imported", [])} if prev.get("run_id") == run_id else {}
    )
    if prev_entries:
        job["imported_utc"] = prev.get("imported_utc", job["imported_utc"])
    for n in ("job_stdout.txt", "lscpu.txt", "meminfo.txt"):
        if (out / n).is_file():
            shutil.copy2(out / n, side / n)
    for n, m in (("git_state.json", "git_state.json"), ("va1.csv", "va1_as_returned.csv")):
        if (src / n).is_file():
            shutil.copy2(src / n, side / m)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    imported = []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        target = dest / d.name
        files = _tree_hashes(d)
        replaced: dict[str, Any] | None = None
        if target.exists():
            pp = target / "provenance.json"
            if pp.is_file():
                old = read_json(pp)
                if old.get("run_id") == run_id and old.get("files_sha256") == files:
                    imported.append(prev_entries.get(d.name, {"dir": d.name, "status": "imported"}))
                    continue
            moved = dest.parent / f"{dest.name}_replaced_{stamp}" / d.name
            moved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(moved))
            replaced = {
                "moved_to": str(moved),
                "had_result": (moved / "result.json").is_file(),
                "had_manifest": (moved / "manifest.json").is_file(),
            }
        shutil.copytree(d, target)
        prov: dict[str, Any] = {
            **{
                k: job[k]
                for k in ("run_id", "origin", "region", "job_exit", "cpu_model_lscpu", "os", "note")
            },
            "instance_type": (launch or {}).get("type"),
            "instance_id": (launch or {}).get("instance_id"),
            "ami": (launch or {}).get("ami"),
            "bundle_sha256": (launch or {}).get("bundle_sha256"),
            "job_cmd": (launch or {}).get("cmd"),
            "source_path": str(d),
            "files_sha256": files,
            "imported_utc": job["imported_utc"],
            "replaced_local": replaced,
        }
        if replaced and replaced["had_result"] and d.name.startswith("batch_"):
            prov["cross_host_check"] = _cross_host_check(Path(replaced["moved_to"]), target)
        write_json(target / "provenance.json", prov)
        imported.append({"dir": d.name, "status": "imported", "replaced_local": replaced})
    job["imported"] = imported
    write_json(side / "job.json", job)
    return job


# =========================================================================== E-B, original lab-like


def aggregate_eb_lab_like(runs: Path) -> bool:
    """Re-aggregate the original lab-like E-B sweeps into ``E-B1/`` and ``E-B2/`` (D24 labels
    from :mod:`experiments.eb_aggregate`). Returns False (results left as they are) when the
    sweep directories are absent; raises when a point is of another geometry."""
    from experiments import eb_aggregate

    eb1 = runs / "eb1"
    eb2 = [runs / "eb2_lab", runs / "eb2_ideal1"]
    if not (eb1 / "index.json").exists() or not all((r / "index.json").exists() for r in eb2):
        return False
    for name in ("E-B1", "E-B2"):
        tmp = RESULTS / f".{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        # strict by default: a failed point raises
        if name == "E-B1":
            eb_aggregate.aggregate_eb1(eb1, tmp)
        else:
            eb_aggregate.aggregate_eb2(eb2, tmp)
        table = tmp / ("eb1_points.csv" if name == "E-B1" else "eb2_per.csv")
        other = sorted(
            {
                r.get("scenario") or ""
                for r in _rows(table)
                if (r.get("scenario") or "") and not str(r["scenario"]).endswith(_LAB_LIKE_SCENARIO)
            }
        )
        if other:
            shutil.rmtree(tmp)
            raise RuntimeError(
                f"{name}: points of another geometry {other} in the original lab-like sweeps"
            )
        dst = RESULTS / name
        if dst.exists():
            shutil.rmtree(dst)
        tmp.rename(dst)
    return True


# =========================================================================== D24 check


def d24_violations(root: Path = RESULTS, skip_top: tuple[str, ...] = ()) -> list[str]:
    """Lines of paper-facing result files with ``legacy``/``v1``/``v2`` naming (D24);
    ``skip_top``: names of top-level files to leave out (this step's own outputs, before
    they are rewritten)."""
    bad = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _D24_SUFFIXES:
            continue
        if p.parent == root and p.name in skip_top:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            t = line
            for a in _D24_ALLOWED:
                t = t.replace(a, "")
            for m in _D24_PATTERN.finditer(t):
                bad.append(
                    f"{p.relative_to(root).as_posix()}:{i}: {m.group(0)!r}: {line.strip()[:160]}"
                )
    return bad


# =========================================================================== summary


def _experiment_record(d: Path) -> dict[str, Any]:
    meta = read_json(d / "meta.json") if (d / "meta.json").is_file() else {}
    man = read_json(d / "manifest.json") if (d / "manifest.json").is_file() else {}
    return {
        "data_classes": meta.get("data_classes"),
        "description": meta.get("description"),
        "manifest": {
            "sha256": _sha256(d / "manifest.json") if man else None,
            "git_commit": man.get("git_commit"),
            "git_dirty": man.get("git_dirty"),
            "dirty_patch_sha256": man.get("dirty_patch_sha256"),
            "platform": man.get("platform"),
            "cpu_model": man.get("cpu_model"),
            "aws_instance": man.get("aws_instance"),
            "t_end_utc": man.get("t_end_utc"),
        },
        "code_src_sremu_tree_sha256": (meta.get("code") or {}).get("src_sremu_tree_sha256"),
        "files_sha256": _tree_hashes(d),
    }


def summary() -> dict[str, Any]:
    """Key numbers from the result files. Raises when a required result is missing; VA-1
    ``all_accept`` exists only over the complete design (else ``VA1IncompleteError``)."""
    from experiments import ea_iq

    out: dict[str, Any] = {}
    hyp = RESULTS / "E-A" / "hypotheses.json"
    if not hyp.exists():
        raise FileNotFoundError(f"{hyp} missing (run experiments.run_sweeps ea first)")
    h = read_json(hyp)
    out["E-A"] = {
        "hypotheses": {k: v for k, v in h.items() if not k.startswith("H-A3_report")},
        "H-A3_reports": {k: v["decision"] for k, v in h.items() if k.startswith("H-A3_report")},
        "tau_half": _rows(RESULTS / "E-A" / "tau_half.csv"),
    }
    va1 = _rows(RESULTS / "E-A-IQ" / "va1.csv")
    ea_iq.check_va1_complete(va1)  # fails loudly on any missing (controller, tau, noise) row
    va1_meta = read_json(RESULTS / "E-A-IQ" / "meta.json")
    out["VA-1"] = {
        "rows": va1,
        "n_rows": len(va1),
        "complete": True,
        "all_accept": all(r["accept"] == "True" for r in va1),
        "max_abs_diff_noise_off_db": max(
            abs(float(r["diff_iq_minus_vectorized_mid_db"])) for r in va1 if r["noise"] == "off"
        ),
        "hosts": va1_meta.get("hosts"),
        "batch_reference_cross_host_check": va1_meta.get("batch_reference_cross_host_check"),
    }
    va5 = RESULTS / "E-B1" / "va5.json"
    if not va5.exists():
        raise FileNotFoundError(f"{va5} missing")
    out["VA-5"] = read_json(va5)
    eb2 = _rows(RESULTS / "E-B2" / "eb2_per.csv")
    if not eb2:
        raise FileNotFoundError(RESULTS / "E-B2" / "eb2_per.csv")
    out["E-B2"] = {
        "n_points": len(eb2),
        "semantics": sorted({r["semantics"] for r in eb2}),
        "labels": LABELS,
        "f9b_rows_guard_0p5ms": [r for r in eb2 if r["guard_s"] == "1/2000"],
    }
    t8 = _rows(RESULTS / "E-C" / "table8.csv")
    if not t8:
        raise FileNotFoundError(RESULTS / "E-C" / "table8.csv")
    out["E-C"] = {"table8": t8}
    ber = _rows(RESULTS / "E-F" / "ber" / "ber.csv")
    if not ber:
        raise FileNotFoundError(RESULTS / "E-F" / "ber" / "ber.csv")
    out["E-F"] = {
        "ber": ber,
        "n_points": len(ber),
        "genie_theory_in_ci95_all": all(r["genie_theory_in_ci95"] == "True" for r in ber),
    }
    g = git_info()
    out["meta"] = {
        "generated_utc": _utc(),
        "sremu_git": {k: g.get(k) for k in ("commit", "dirty", "dirty_patch_sha256")},
        "experiments": {e: _experiment_record(RESULTS / e) for e in EXPERIMENT_DIRS},
        "excluded": EXCLUDED,
        "labels": LABELS,
        "d24_check": {"checked_by": "experiments.finalize.d24_violations", "violations": 0},
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--runs", type=Path, default=REPO / "runs" / "wp11")
    ap.add_argument(
        "--import-aws-ea-iq",
        type=Path,
        default=None,
        metavar="JOB_DIR",
        help="fetched AWS job directory whose out/ea_iq run dirs are imported first",
    )
    ap.add_argument("--note", default=None, help="provenance note stored with the import")
    args = ap.parse_args(argv)
    runs = args.runs.resolve()
    if args.import_aws_ea_iq is not None:
        job = import_aws_ea_iq(args.import_aws_ea_iq, runs / "ea_iq", note=args.note)
        print(json.dumps({"imported": job["imported"]}, default=str))
    export_git_info(runs)  # git state read once; a dirty patch goes to runs/, not the paper repo
    from experiments import ea_iq

    ea_iq.aggregate(runs / "ea_iq", RESULTS / "E-A-IQ")  # strict: raises on a missing row
    if not aggregate_eb_lab_like(runs):
        print("E-B: original lab-like sweep directories absent; E-B1/E-B2 results left as they are")
    _copy_run(runs / "ec", RESULTS / "E-C")
    if (runs / "ef_ber").exists():
        from experiments import ef

        ef.aggregate_ber(runs / "ef_ber", RESULTS / "E-F" / "ber")
    stale = RESULTS / "E-F" / "latency"
    if stale.exists():
        shutil.rmtree(stale)
        print(f"removed {stale} ({EXCLUDED['E-F/latency']})")
    bad = d24_violations(RESULTS, skip_top=("summary.json", "meta.json", "manifest.json"))
    if bad:
        raise RuntimeError("D24 naming violations in paper-facing results:\n" + "\n".join(bad[:50]))
    s = summary()
    write_json(RESULTS / "summary.json", s)
    rec = RunRecord(RESULTS, 20261010, mode="analysis", patch_root=runs)
    classes = sorted(
        {c for e in s["meta"]["experiments"].values() for c in (e["data_classes"] or [])}
    )
    write_meta(
        RESULTS,
        experiment="WP11-results",
        data_classes=classes,
        description="Index of the WP11 paper results: summary.json holds the key numbers of E-A, "
        "VA-1 (E-A-IQ), E-B1/VA-5, E-B2, E-C and E-F BER, copied from the result files, with data "
        "classes, git state and manifest/file SHA-256 per experiment",
        assumptions=[EXCLUDED["E-F/latency"], *(f"label {k!r}: {v}" for k, v in LABELS.items())],
        files={"summary.json": "key numbers and provenance index"},
        extra={
            "experiments": list(EXPERIMENT_DIRS),
            "summary_sha256": _sha256(RESULTS / "summary.json"),
        },
    )
    rec.finish()
    bad = d24_violations(RESULTS)
    if bad:
        raise RuntimeError("D24 naming violations in paper-facing results:\n" + "\n".join(bad[:50]))
    print(
        json.dumps({k: list(v) if isinstance(v, dict) else v for k, v in s.items()}, default=str)[
            :2000
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
