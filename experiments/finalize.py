"""Collect the WP11 results into ``paper/sweeps/results/`` and write ``summary.json``.

    python -m experiments.finalize [--runs runs/wp11]
    python -m experiments.finalize --import-aws-ea-iq <AWS job results dir> [--note TEXT]
    python -m experiments.finalize --runs <job out> --out <job out>/paper_results [--smoke]
        [--eb-arc <job out>/eb_arc] [--with-latency]

``--out`` sets the results root (default ``paper/sweeps/results/``); a regeneration run
(``experiments/regenerate_paper_data.sh``) writes into its own job directory, never into the
paper repository. Every path below is relative to that root. ``--smoke`` is for the reduced
local check of that script only: it needs a non-default ``--out``, aggregates VA-1 without
the completeness rule (short smoke runs are not VA-1 rows, so VA-1 has no verdict) and marks
``summary.json`` as ``smoke: true``. Smoke output is never paper data.

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
* E-F API latency is **not** collected by default (a transport latency fix is pending and the
  latency will be re-measured); a stale ``E-F/latency/`` is removed. ``--with-latency``
  (WAITING for that fix: use it only for a run of the fixed transport) copies the run
  directory ``<runs>/ef_latency`` (:func:`experiments.ef.run_latency`) to ``E-F/latency/``
  and adds its percentiles to ``summary.json`` as MEAS of the recording host only (CPU,
  platform and instance type next to the numbers; never a claim for another host);
* ``E-B-arc/`` (``--eb-arc DIR``): an E-B run on the lab arc (``python -m experiments.eb
  --scenario scenarios/studies/lab_arc.yaml --out DIR``) in the layout
  ``tools/gen_results.py`` reads (:func:`collect_eb_arc`). Not part of ``summary.json``
  (the WP11 index), as in the paper tree;
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
LATENCY_DIR = "E-F/latency"
LATENCY_NOTE = (
    "E-F API latency: MEAS of the recording host only (CPU, platform and instance type in "
    "E-F/latency/manifest.json and in summary.json next to the numbers); no claim for another "
    "host and no cross-host speed-up"
)

ARC_DIR = "E-B-arc"
ARC_SCENARIO = "scenarios/studies/lab_arc.yaml"
ARC_SPECS = ("eb1", "eb2_lab", "eb2_ideal1", "eb2_lab_es15", "eb2_ideal1_es15")
ARC_TEXT: dict[str, Any] = {
    # the wording of paper/sweeps/results/E-B-arc/meta.json (compare_regen.py compares it)
    "description": "E-B (intra-frame RIS reconfiguration: E-B1 mechanism + VA-5, E-B2 timing) "
    "on scenarios/studies/lab_arc.yaml; aggregated results of one AWS run, copied unchanged from "
    "the run's results directory",
    "data_class_notes": {
        "SIM-sremu": "every result value (EVM, BER, PER, VA-5) comes from sremu runs",
        "ASSUMED": "an input, not a result: the two lab cell states have |Gamma| = 1 (the "
        "library has no measured magnitudes)",
    },
    "geometry_note": "nominal values (middle of the author-supplied intervals: R 1.95 m, span "
    "46 deg, centre 22.5 deg, height 1.0 m); not a measured geometry; no measurement is "
    "compared with these runs",
    "assumptions": [
        "lab state amplitudes |Gamma| = 1 ASSUMED (alphabet 'lab'); ideal 1-bit {1, -1} also run",
        "transition path complex-linear [ENG]",
        "six tiles act as one RIS: joint exact optimum over every cell; one command per panel at "
        "the same time",
        "LOS blocked: one opaque plate across both LOS segments; occluders act on the LOS only, "
        "so the RIS hops are unattenuated (idealized regime 'LOS removed, RIS hops "
        "unattenuated') [ENG]",
        "TX power set for Es/N0 = 25 dB at RX1 under its own optimum; Es/N0 = 15 dB is the "
        "pre-declared sensitivity of the LOS-present slice (eb_plan.json 'pre_declared')",
        "free space with LOS plus the RIS; isotropic antennas in the horizontal plane [ENG]",
    ],
    "not_in_wp11_index": "this run is not part of sweeps/results/summary.json (the WP11 index); "
    "tools/gen_results.py reads E-B-arc directly",
}
_ARC_SHARED = (
    "sremu_version",
    "git_commit",
    "git_dirty",
    "dirty_patch_sha256",
    "contract_version",
    "versions",
    "platform",
    "cpu_model",
    "logical_cpus",
    "aws_instance",
    "rng_scheme",
    "deterministic",
    "numba_disable_svml",
    "seed",
)
"""Manifest fields every point of one E-B arc run must share (``run_manifests.json``)."""

_D24_PATTERN = re.compile(r"legacy|\bv[12]\b", re.IGNORECASE)
_D24_ALLOWED = (
    "philox4x32-10/blake2b-v1",  # RNG scheme id (DESIGN.md 6.1), not a product version
    "Spectre v1",  # lscpu vulnerability names in recorded host files, not a product version
    "Spectre v2",
)
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


def aggregate_eb_lab_like(runs: Path, results: Path = RESULTS) -> bool:
    """Re-aggregate the original lab-like E-B sweeps into ``<results>/E-B1/`` and
    ``<results>/E-B2/`` (D24 labels from :mod:`experiments.eb_aggregate`). Returns False
    (results left as they are) when the sweep directories are absent; raises when a point is
    of another geometry."""
    from experiments import eb_aggregate

    eb1 = runs / "eb1"
    eb2 = [runs / "eb2_lab", runs / "eb2_ideal1"]
    if not (eb1 / "index.json").exists() or not all((r / "index.json").exists() for r in eb2):
        return False
    for name in ("E-B1", "E-B2"):
        tmp = results / f".{name}.tmp"
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
        dst = results / name
        if dst.exists():
            shutil.rmtree(dst)
        tmp.rename(dst)
    return True


# =========================================================================== E-B, lab arc


def collect_eb_arc(src: str | Path, results: Path = RESULTS, *, smoke: bool = False) -> Path:
    """Copy an E-B run on the lab arc into ``<results>/E-B-arc/`` in the layout of the paper
    tree (the one ``tools/gen_results.py`` and ``figures/src/f09_intraframe.py`` read).

    ``src`` is the ``--out`` of ``python -m experiments.eb --scenario
    scenarios/studies/lab_arc.yaml`` (with ``--smoke``: its smoke files). Copied unchanged:
    the run's aggregation ``results/E-B1``, ``results/E-B2``, ``eb_plan.json``,
    ``eb_summary.json``, ``git_state.json`` and ``sweep.json`` / ``index.csv`` of every sweep
    (under ``sweeps/``). Written: ``run_manifests.json`` (SHA-256 of every point manifest and
    the fields they share) and ``meta.json`` (provenance: run, host, git state and code hashes
    of the run; the git state is the one every point manifest records). Raises, writing
    nothing, when the run has failed points or VA-5 failed, the plan is not the lab-arc plan,
    or a point manifest disagrees with ``git_state.json``.
    """
    src = Path(src).resolve()
    sfx = "_smoke" if smoke else ""
    plan_p, summ_p = src / f"eb_plan{sfx}.json", src / f"eb_summary{sfx}.json"
    gs_p = src / "git_state.json"
    agg = src / f"results{sfx}"
    for p in (plan_p, summ_p, gs_p, agg / "E-B1", agg / "E-B2"):
        if not p.exists():
            raise FileNotFoundError(f"E-B arc: {p} missing")
    plan, summ, gs = read_json(plan_p), read_json(summ_p), read_json(gs_p)
    if plan.get("scenario_path") != ARC_SCENARIO:
        raise RuntimeError(f"{plan_p}: scenario {plan.get('scenario_path')!r}, not {ARC_SCENARIO}")
    specs = [s["name"] for s in plan["specs"]]
    if not smoke and tuple(specs) != ARC_SPECS:
        raise RuntimeError(f"{plan_p}: sweeps {specs}, expected {list(ARC_SPECS)}")
    if not (summ.get("ok") and summ.get("n_failed") == 0 and (summ.get("va5") or {}).get("pass")):
        raise RuntimeError(f"{summ_p}: the run is not ok (failed points or VA-5 failed)")
    tmp = results / f".{ARC_DIR}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    for exp in ("E-B1", "E-B2"):
        shutil.copytree(agg / exp, tmp / exp, ignore=shutil.ignore_patterns("*.patch", "*.tmp"))
    for p, name in ((plan_p, "eb_plan.json"), (summ_p, "eb_summary.json"), (gs_p, gs_p.name)):
        shutil.copyfile(p, tmp / name)
    points: dict[str, dict[str, str]] = {}
    shared: dict[str, Any] | None = None
    jobs = set()
    for s in specs:
        (tmp / "sweeps" / s).mkdir(parents=True)
        for name in ("index.csv", "sweep.json"):
            shutil.copyfile(src / s / name, tmp / "sweeps" / s / name)
        jobs.add(read_json(src / s / "sweep.json").get("jobs"))
        points[s] = {}
        for d in sorted(p for p in (src / s).iterdir() if p.is_dir() and p.name.startswith("p")):
            points[s][d.name] = _sha256(d / "manifest.json")
            m = read_json(d / "manifest.json")
            cur = {k: m.get(k) for k in _ARC_SHARED}
            if shared is None:
                shared = cur
            elif cur != shared:
                shutil.rmtree(tmp)
                diff = sorted(k for k in _ARC_SHARED if cur[k] != shared[k])
                raise RuntimeError(f"E-B arc {s}/{d.name}: manifest differs in {diff}")
    if shared is None or (shared["git_commit"], shared["git_dirty"]) != (
        gs.get("commit"),
        gs.get("dirty"),
    ):
        shutil.rmtree(tmp)
        raise RuntimeError("E-B arc: the point manifests do not record git_state.json's state")
    write_json(
        tmp / "run_manifests.json",
        {
            "description": "sha256 of every per-point manifest.json of the run (run directories "
            "kept in the job output, not copied); 'shared' holds the fields identical in all of "
            "them",
            "n_manifests": sum(len(v) for v in points.values()),
            "shared": shared,
            "points": points,
        },
    )
    aws = shared.get("aws_instance") or {}
    code = gs.get("code") or {}
    scen_sha = (code.get("files") or {}).get(ARC_SCENARIO) or _sha256(REPO / ARC_SCENARIO)
    prov_p = src.parent / "_regen" / "provenance.json"  # a regenerate_paper_data.sh job
    run_id = f"regen_{read_json(prov_p)['started_utc']}" if prov_p.is_file() else src.name
    meta = {
        "experiment": ARC_DIR,
        "description": ARC_TEXT["description"],
        "data_classes": ["SIM-sremu", "ASSUMED"],
        "data_class_notes": ARC_TEXT["data_class_notes"],
        "geometry_label": plan["geometry_label"],
        "geometry_note": ARC_TEXT["geometry_note"],
        "scenario": {"path": ARC_SCENARIO, "sha256": scen_sha},
        "assumptions": list(ARC_TEXT["assumptions"]),
        "run": {
            "run_id": run_id,
            "source_dir": str(src),
            "t_start_utc": summ.get("t_start_utc"),
            "t_end_utc": summ.get("t_end_utc"),
            "wall_s": summ.get("wall_s"),
            "points": {s: summ["specs"][s]["n_points"] for s in specs},
            "failed_points": summ["n_failed"],
            "va5_pass": summ["va5"]["pass"],
            "argv": plan.get("argv"),
            **({"smoke": True} if smoke else {}),
        },
        "host": {
            "provider": "AWS EC2" if aws.get("type") else None,
            "instance_type": aws.get("type"),
            "instance_id": aws.get("id"),
            "platform": shared["platform"],
            "cpu_model": shared["cpu_model"],
            "logical_cpus": shared["logical_cpus"],
            "versions": shared["versions"],
            "jobs": sorted(j for j in jobs if j is not None),
        },
        "sremu": {
            "commit": gs.get("commit"),
            "dirty": gs.get("dirty"),
            "dirty_patch_sha256": gs.get("dirty_patch_sha256"),
            "commit_source": "the run's git state, recorded in git_state.json and in every point "
            "manifest (run_manifests.json 'shared'); in a regeneration job it is the state fixed "
            "once in _regen/git_state.json (SREMU_GIT_COMMIT/SREMU_GIT_DIRTY of the AWS job "
            "bundle, or the work tree)",
            "code_hashes_run": code,
        },
        "not_in_wp11_index": ARC_TEXT["not_in_wp11_index"],
        "files": _tree_hashes(tmp),
        "written_utc": _utc(),
    }
    write_json(tmp / "meta.json", meta)
    dst = results / ARC_DIR
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return dst


# =========================================================================== E-F latency


def collect_latency(runs: Path, results: Path = RESULTS) -> Path:
    """Copy the E-F API latency run directory ``<runs>/ef_latency`` to
    ``<results>/E-F/latency/`` (MEAS, the recording host only). WAITING for the transport
    latency fix: call it only for a run of the fixed transport (``--with-latency``)."""
    src = runs / "ef_latency"
    for name in ("latency.json", "manifest.json", "meta.json"):
        if not (src / name).is_file():
            raise FileNotFoundError(f"E-F latency: {src / name} missing")
    if read_json(src / "meta.json").get("data_classes") != ["MEAS"]:
        raise RuntimeError(f"{src / 'meta.json'}: E-F latency must be MEAS only")
    dst = results / LATENCY_DIR
    _copy_run(src, dst)
    return dst


def latency_summary(results: Path = RESULTS) -> dict[str, Any]:
    """Key numbers of ``E-F/latency/`` with the recording host next to them (MEAS)."""
    d = results / LATENCY_DIR
    lat, man = read_json(d / "latency.json"), read_json(d / "manifest.json")
    aws = man.get("aws_instance") or {}
    inproc, zmq = lat["inproc"], lat["zmq_two_process"]
    return {
        "data_class": "MEAS",
        "host_specific": LATENCY_NOTE,
        "host": {
            "cpu_model": man.get("cpu_model"),
            "logical_cpus": man.get("logical_cpus"),
            "platform": man.get("platform"),
            "aws_instance_type": aws.get("type") if isinstance(aws, dict) else None,
        },
        "buffer_samples": inproc.get("buffer"),
        "inproc_send": inproc["send"],
        "inproc_recv": inproc["recv"],
        "inproc_sustained_msps": inproc["sustained_msps"],
        "zmq_round_trip": zmq["round_trip"],
        "zmq_sustained_msps": zmq["sustained_msps"],
        "zmq_sustained_buffer_samples": zmq.get("sustained_buffer"),
    }


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


def summary(
    results: Path = RESULTS, *, smoke: bool = False, latency: bool = False
) -> dict[str, Any]:
    """Key numbers from the result files under ``results``. Raises when a required result is
    missing; VA-1 ``all_accept`` exists only over the complete design (else
    ``VA1IncompleteError``). ``smoke``: the VA-1 completeness rule is not applied, VA-1 gets
    ``complete: false`` and no verdict, and ``meta.smoke`` is true (never paper data).
    ``latency``: ``E-F/latency/`` was collected (:func:`latency_summary`, MEAS)."""
    from experiments import ea_iq

    out: dict[str, Any] = {}
    hyp = results / "E-A" / "hypotheses.json"
    if not hyp.exists():
        raise FileNotFoundError(f"{hyp} missing (run experiments.run_sweeps ea first)")
    h = read_json(hyp)
    out["E-A"] = {
        "hypotheses": {k: v for k, v in h.items() if not k.startswith("H-A3_report")},
        "H-A3_reports": {k: v["decision"] for k, v in h.items() if k.startswith("H-A3_report")},
        "tau_half": _rows(results / "E-A" / "tau_half.csv"),
    }
    va1 = _rows(results / "E-A-IQ" / "va1.csv")
    if not smoke:
        ea_iq.check_va1_complete(va1)  # fails loudly on any missing (controller, tau, noise) row
    va1_meta = read_json(results / "E-A-IQ" / "meta.json")
    off = [abs(float(r["diff_iq_minus_vectorized_mid_db"])) for r in va1 if r["noise"] == "off"]
    out["VA-1"] = {
        "rows": va1,
        "n_rows": len(va1),
        "complete": not smoke,
        "all_accept": None if smoke else all(r["accept"] == "True" for r in va1),
        "max_abs_diff_noise_off_db": max(off) if off else None,
        "hosts": va1_meta.get("hosts"),
        "batch_reference_cross_host_check": va1_meta.get("batch_reference_cross_host_check"),
    }
    if smoke:
        out["VA-1"]["missing_rows"] = va1_meta.get("missing_rows")
    va5 = results / "E-B1" / "va5.json"
    if not va5.exists():
        raise FileNotFoundError(f"{va5} missing")
    out["VA-5"] = read_json(va5)
    eb2 = _rows(results / "E-B2" / "eb2_per.csv")
    if not eb2:
        raise FileNotFoundError(results / "E-B2" / "eb2_per.csv")
    out["E-B2"] = {
        "n_points": len(eb2),
        "semantics": sorted({r["semantics"] for r in eb2}),
        "labels": LABELS,
        "f9b_rows_guard_0p5ms": [r for r in eb2 if r["guard_s"] == "1/2000"],
    }
    t8 = _rows(results / "E-C" / "table8.csv")
    if not t8:
        raise FileNotFoundError(results / "E-C" / "table8.csv")
    out["E-C"] = {"table8": t8}
    ber = _rows(results / "E-F" / "ber" / "ber.csv")
    if not ber:
        raise FileNotFoundError(results / "E-F" / "ber" / "ber.csv")
    out["E-F"] = {
        "ber": ber,
        "n_points": len(ber),
        "genie_theory_in_ci95_all": all(r["genie_theory_in_ci95"] == "True" for r in ber),
    }
    if latency:
        out["E-F-latency"] = latency_summary(results)
    g = git_info()
    dirs = (*EXPERIMENT_DIRS, LATENCY_DIR) if latency else EXPERIMENT_DIRS
    out["meta"] = {
        "generated_utc": _utc(),
        "smoke": smoke,
        "sremu_git": {k: g.get(k) for k in ("commit", "dirty", "dirty_patch_sha256")},
        "experiments": {e: _experiment_record(results / e) for e in dirs},
        "excluded": {} if latency else EXCLUDED,
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
    ap.add_argument(
        "--out",
        type=Path,
        default=RESULTS,
        help="results root (default: the paper repo's sweeps/results)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="reduced local check of regenerate_paper_data.sh: VA-1 aggregated without the "
        "completeness rule, summary marked smoke; needs a non-default --out",
    )
    ap.add_argument(
        "--eb-arc",
        type=Path,
        default=None,
        metavar="DIR",
        help="E-B lab-arc run (--out of python -m experiments.eb --scenario "
        "scenarios/studies/lab_arc.yaml) collected into E-B-arc/",
    )
    ap.add_argument(
        "--with-latency",
        action="store_true",
        help="collect <runs>/ef_latency into E-F/latency (MEAS, recording host only). WAITING "
        "for the transport latency fix: only for a run of the fixed transport",
    )
    args = ap.parse_args(argv)
    runs = args.runs.resolve()
    res = args.out.resolve()
    if args.smoke and res == RESULTS.resolve():
        ap.error("--smoke needs a non-default --out (smoke output is never paper data)")
    if args.import_aws_ea_iq is not None:
        job = import_aws_ea_iq(args.import_aws_ea_iq, runs / "ea_iq", note=args.note)
        print(json.dumps({"imported": job["imported"]}, default=str))
    export_git_info(runs)  # git state read once; a dirty patch goes to runs/, not the paper repo
    from experiments import ea_iq

    # strict (default): raises on a missing row
    ea_iq.aggregate(runs / "ea_iq", res / "E-A-IQ", strict=not args.smoke)
    if not aggregate_eb_lab_like(runs, res):
        print("E-B: original lab-like sweep directories absent; E-B1/E-B2 results left as they are")
    _copy_run(runs / "ec", res / "E-C")
    if (runs / "ef_ber").exists():
        from experiments import ef

        ef.aggregate_ber(runs / "ef_ber", res / "E-F" / "ber")
    if args.with_latency:
        print(f"collected {collect_latency(runs, res)} ({LATENCY_NOTE})")
    else:
        stale = res / LATENCY_DIR
        if stale.exists():
            shutil.rmtree(stale)
            print(f"removed {stale} ({EXCLUDED[LATENCY_DIR]})")
    if args.eb_arc is not None:
        print(f"collected {collect_eb_arc(args.eb_arc, res, smoke=args.smoke)}")
    bad = d24_violations(res, skip_top=("summary.json", "meta.json", "manifest.json"))
    if bad:
        raise RuntimeError("D24 naming violations in paper-facing results:\n" + "\n".join(bad[:50]))
    s = summary(res, smoke=args.smoke, latency=args.with_latency)
    write_json(res / "summary.json", s)
    rec = RunRecord(res, 20261010, mode="analysis", patch_root=runs)
    classes = sorted(
        {c for e in s["meta"]["experiments"].values() for c in (e["data_classes"] or [])}
    )
    smoke_note = (
        ["SMOKE: reduced subsets from regenerate_paper_data.sh --smoke; not paper data"]
        if args.smoke
        else []
    )
    write_meta(
        res,
        experiment="WP11-results",
        data_classes=classes,
        description="Index of the WP11 paper results: summary.json holds the key numbers of E-A, "
        "VA-1 (E-A-IQ), E-B1/VA-5, E-B2, E-C and E-F BER, copied from the result files, with data "
        "classes, git state and manifest/file SHA-256 per experiment",
        assumptions=[
            *smoke_note,
            LATENCY_NOTE if args.with_latency else EXCLUDED[LATENCY_DIR],
            *(f"label {k!r}: {v}" for k, v in LABELS.items()),
        ],
        files={"summary.json": "key numbers and provenance index"},
        extra={
            "experiments": list(s["meta"]["experiments"]),
            "summary_sha256": _sha256(res / "summary.json"),
            **({"smoke": True} if args.smoke else {}),
        },
    )
    rec.finish()
    bad = d24_violations(res)
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
