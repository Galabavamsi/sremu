"""Install a compared pinned-commit regeneration into the paper tree (and the validation runs
into the sremu tree).

    python experiments/install_regen.py JOB_OUT [--report REPORT.json] [--apply]
        [--accept-differences] [--with-latency] [--aws-run-id RUN_ID]
        [--paper DIR] [--sremu DIR]

``JOB_OUT`` is the fetched job output (``<research>/aws/results/<run_id>/out``) of
``experiments/regenerate_paper_data.sh``; ``REPORT.json`` (default
``JOB_OUT/compare_report.json``) is the report of ``experiments/compare_regen.py`` run on the
local, clean checkout of the job's commit. Without ``--apply`` the plan is printed and nothing
is written.

Refuses unless the report is a full (not smoke) comparison of a pinned tree whose code check
passed (local HEAD = the job's commit, same code hashes) and whose V-LAB check record exists
and passed. Differences found by the comparison
also refuse, unless ``--accept-differences`` says the lead reviewed them and the regenerated
values become the paper's (the paper then quotes the pinned-commit values;
``tools/gen_results.py --check`` lists the macros that change).

What it copies (and nothing else):

* ``JOB_OUT/paper_results/{E-A, E-A-IQ, E-B1, E-B2, E-C, E-F/ber, E-B-arc}`` and the root
  ``summary.json``, ``meta.json``, ``manifest.json`` into ``paper/sweeps/results/``, each
  replacing the old one; ``E-F/latency`` only with ``--with-latency`` (WAITING for the
  transport fix; MEAS of the job's host). ``bench/`` and anything else is left alone.
* ``JOB_OUT/validation/<study>/<UTC>_<name>/`` (V-TANG, V-CFO, V-FM) into
  ``sremu/validation/<study>/results/`` as new run directories (old runs stay). They go to the
  sremu tree, not the paper tree, because the paper reads them there today and because the
  V-CFO record carries internal material that must not enter the paper repository (DESIGN
  D24: ``legacy_*`` series names and digitized internal reference rows). Their ``meta.json``
  keeps the job's ``run_dir`` (where the run was made); the installed location is the
  directory name.
* the run-directory names in the paper sources that read those runs: ``TANG_RUN``,
  ``CFO_RUN``, ``FM_RUN`` of ``tools/gen_results.py``, the run constants of
  ``figures/src/f03_aperture_tang.py``, ``f05_cfo_ber.py``, ``f12_fm_sizing.py`` and the
  inputs of F3, F5 and F12 in ``figures/registry.yaml`` (exact string replacement, each
  occurrence count checked first).

With ``--aws-run-id``, ``E-B-arc/meta.json`` ``run`` gets the runner's run id and launch
record (``research/aws/launches.jsonl``). V-LAB is never touched.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

SREMU = Path(__file__).resolve().parents[1]
PAPER = SREMU.parent / "paper"
LAUNCHES = SREMU.parent / "research" / "aws" / "launches.jsonl"

RESULT_DIRS = ("E-A", "E-A-IQ", "E-B1", "E-B2", "E-C", "E-F/ber", "E-B-arc")
ROOT_FILES = ("summary.json", "meta.json", "manifest.json")
LATENCY_DIR = "E-F/latency"
STUDIES = {  # study -> (job subdirectory, run name, gen_results constant)
    "V-TANG": ("tang", "vtang", "TANG_RUN"),
    "V-CFO": ("cfo", "vcfo", "CFO_RUN"),
    "V-FM": ("fm", "vfm", "FM_RUN"),
}
READERS = {  # paper file -> {study: expected occurrences of the run path}
    "tools/gen_results.py": {"V-TANG": 1, "V-CFO": 1, "V-FM": 1},
    "figures/src/f03_aperture_tang.py": {"V-TANG": 1},
    "figures/src/f05_cfo_ber.py": {"V-CFO": 1},
    "figures/src/f12_fm_sizing.py": {"V-FM": 1},
    "figures/registry.yaml": {"V-TANG": 2, "V-CFO": 3, "V-FM": 2},
}


def _json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_text(p: Path, text: str) -> None:
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def preconditions(rep: dict[str, Any], accept: bool) -> list[str]:
    """Reasons to refuse (empty: install allowed)."""
    bad = []
    if rep.get("mode") != "full" or rep.get("new_is_smoke_tree"):
        bad.append("the report is not a full comparison of a full (non-smoke) tree")
    if not (rep.get("provenance") or {}).get("new", {}).get("pinned"):
        bad.append("the regenerated tree is not pinned (one commit, clean, one code hash)")
    cc = rep.get("code_check") or {}
    if not (cc.get("performed") and cc.get("ok")):
        bad.append(f"the code check did not pass ({cc.get('reason') or 'see code_check'})")
    if rep.get("missing_files"):
        bad.append(f"files missing in the regeneration: {rep['missing_files'][:5]}")
    vl = rep.get("vlab_check")
    if vl is None:
        bad.append("the report has no V-LAB check record (_regen/vlab_check.json of the job)")
    elif not vl.get("ok"):
        bad.append(f"the V-LAB check failed: {vl.get('summary')}")
    if rep.get("failed_files") and not accept:
        bad.append(
            f"{len(rep['failed_files'])} files differ ({rep['failed_files'][:5]}); review them "
            "and pass --accept-differences to install the regenerated values"
        )
    return bad


def plan(job: Path, paper: Path, sremu: Path, with_latency: bool) -> dict[str, Any]:
    """Everything the install would do, checked against the trees (raises on a problem)."""
    new = job / "paper_results"
    dirs = [*RESULT_DIRS, *([LATENCY_DIR] if with_latency else [])]
    copies = []
    for d in dirs:
        if not (new / d).is_dir():
            raise SystemExit(f"{new / d} missing")
        copies.append((new / d, paper / "sweeps" / "results" / d))
    for f in ROOT_FILES:
        if not (new / f).is_file():
            raise SystemExit(f"{new / f} missing")
        copies.append((new / f, paper / "sweeps" / "results" / f))
    gr = (paper / "tools" / "gen_results.py").read_text(encoding="utf-8")
    runs = []
    renames: dict[str, tuple[str, str]] = {}
    for study, (sub, name, const) in STUDIES.items():
        found = sorted(
            p
            for p in (job / "validation" / sub).glob(f"*_{name}")
            if p.is_dir() and re.fullmatch(rf"\d{{8}}T\d{{6}}Z_{name}", p.name)
        )
        if len(found) != 1:
            raise SystemExit(f"{study}: {len(found)} full runs under {job / 'validation' / sub}")
        dst = sremu / "validation" / sub / "results" / found[0].name
        if dst.exists():
            raise SystemExit(f"{dst} exists")
        runs.append((found[0], dst))
        m = re.search(rf'^{const} = "([^"]+)"', gr, re.M)
        if not m:
            raise SystemExit(f"tools/gen_results.py: {const} not found")
        renames[study] = (m.group(1), f"validation/{sub}/results/{found[0].name}")
    edits = []
    for rel, want in READERS.items():
        text = (paper / rel).read_text(encoding="utf-8")
        for study, n in want.items():
            old = renames[study][0]
            if text.count(old) != n:
                raise SystemExit(f"{rel}: {text.count(old)} occurrences of {old}, expected {n}")
        edits.append(rel)
    return {"copies": copies, "runs": runs, "renames": renames, "edits": edits}


def apply(p: dict[str, Any], paper: Path, aws_run_id: str | None) -> None:
    for src, dst in p["copies"]:
        if dst.is_dir():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copyfile(src, dst)
    for src, dst in p["runs"]:
        shutil.copytree(src, dst)
    for rel in p["edits"]:
        f = paper / rel
        text = f.read_text(encoding="utf-8")
        for old, new in p["renames"].values():
            text = text.replace(old, new)
        _write_text(f, text)
    if aws_run_id:
        mp = paper / "sweeps" / "results" / "E-B-arc" / "meta.json"
        meta = _json(mp)
        launch = None
        if LAUNCHES.is_file():
            for line in LAUNCHES.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("run_id") == aws_run_id:
                    launch = json.loads(line)
        meta["run"]["aws_run_id"] = aws_run_id
        meta["run"]["launch"] = launch
        _write_text(mp, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("job", type=Path, help="fetched job output (<run_id>/out)")
    ap.add_argument("--report", type=Path, default=None, help="compare_regen.py --json report")
    ap.add_argument("--apply", action="store_true", help="write (default: print the plan)")
    ap.add_argument("--accept-differences", action="store_true")
    ap.add_argument("--with-latency", action="store_true", help="also E-F/latency (MEAS)")
    ap.add_argument("--aws-run-id", default=None)
    ap.add_argument("--paper", type=Path, default=PAPER)
    ap.add_argument("--sremu", type=Path, default=SREMU)
    a = ap.parse_args(argv)
    job = a.job.resolve()
    rep = _json(a.report or job / "compare_report.json")
    bad = preconditions(rep, a.accept_differences)
    if bad:
        print("refusing:\n  " + "\n  ".join(bad))
        return 1
    p = plan(job, a.paper.resolve(), a.sremu.resolve(), a.with_latency)
    for src, dst in p["copies"]:
        print(f"copy   {src} -> {dst}")
    for src, dst in p["runs"]:
        print(f"run    {src} -> {dst}")
    for study, (old, new) in p["renames"].items():
        print(f"rename {study}: {old} -> {new} in {', '.join(p['edits'])}")
    if not a.apply:
        print("dry run: nothing written (add --apply)")
        return 0
    apply(p, a.paper.resolve(), a.aws_run_id)
    print(
        "installed; next: git status in both repositories, then the steps of "
        "research/analysis/regen_coverage.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
