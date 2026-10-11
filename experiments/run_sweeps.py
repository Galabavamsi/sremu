"""Drive the WP11 sweeps with the sremu sweep runner (``sremu.experiments.sweep``).

    python -m experiments.run_sweeps ea   [--jobs 12] [--out runs/wp11] [--resume]
    python -m experiments.run_sweeps eb1  [...]
    python -m experiments.run_sweeps eb2  [...]

Run from the sremu repository root with the sremu venv. For each experiment: the git state
is read once (``dirty.patch`` written into the output root when the tree is dirty; every
point manifest carries its hash), shared caches are prebuilt, every sweep spec of
``paper/sweeps/`` is run with ``--jobs`` worker processes (one run directory per point),
then the aggregation writes ``paper/sweeps/results/<experiment>/`` (CSV/JSON, meta.json,
manifest.json).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.common import PAPER, export_git_info  # noqa: E402
from sremu.experiments.sweep import load_spec, run_sweep  # noqa: E402

SPECS = {
    "ea": ["ea_ref.yaml", "ea_tau.yaml"],
    "eb1": ["eb1.yaml"],
    "eb2": ["eb2_lab.yaml", "eb2_ideal1.yaml"],
}
RESULTS = PAPER / "sweeps" / "results"


def _prebuild_ea(out: Path, jobs: int, spec_paths: list[Path]) -> None:
    from experiments import ea

    cache = out / "_ea_cache"
    geos: list[tuple[float, float, float]] = []
    for sp in spec_paths:
        for pt in load_spec(sp).points():
            g = (
                float(pt.params["fc_hz"]),
                float(pt.params["h_m"]),
                float(pt.params.get("loss_db") or 0),
            )
            if g not in geos:
                geos.append(g)
    with cf.ProcessPoolExecutor(max_workers=max(1, min(jobs, len(geos)))) as pool:
        futs = [pool.submit(ea.prebuild, fc, h, lp, cache) for fc, h, lp in geos]
        futs.append(
            pool.submit(ea.build_trajectory, ea.corridor_scenario(1e9, 2, 20), 33_334.0, cache)
        )
        for f in cf.as_completed(futs):
            r = f.result()
            print("prebuilt", r if isinstance(r, str) else "trajectory", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment", choices=sorted(SPECS))
    ap.add_argument("--jobs", "-j", type=int, default=12)
    ap.add_argument("--out", type=Path, default=REPO / "runs" / "wp11")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", default=None, help="run only this spec file name")
    ap.add_argument("--no-aggregate", action="store_true")
    ap.add_argument("--spec-dir", type=Path, default=PAPER / "sweeps", help="sweep spec directory")
    ap.add_argument("--results", type=Path, default=RESULTS, help="aggregated results root")
    args = ap.parse_args(argv)
    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    info = export_git_info(out)
    print(
        f"git {info['commit']} dirty={info['dirty']} patch={info['dirty_patch_sha256']}", flush=True
    )
    if args.experiment == "ea":
        _prebuild_ea(out, args.jobs, [args.spec_dir / n for n in SPECS["ea"]])
    roots = []
    for name in SPECS[args.experiment]:
        if args.only and name != args.only:
            continue
        spec = load_spec(args.spec_dir / name)
        t0 = time.perf_counter()

        def progress(rec: dict) -> None:
            err = f" {rec.get('error')}" if rec["status"] != "ok" else ""
            print(
                f"[{spec.name}] {rec['point_id']} {rec['status']} {rec.get('wall_s', 0):.1f}s{err}",
                flush=True,
            )

        res = run_sweep(
            spec,
            out,
            jobs=args.jobs,
            resume=args.resume,
            allow_import=True,
            progress=progress,
            argv=sys.argv,
        )
        print(
            f"[{spec.name}] {res.n_ok} ok, {res.n_failed} failed, {res.n_resumed} resumed in "
            f"{time.perf_counter() - t0:.0f} s",
            flush=True,
        )
        roots.append(res.root)
    if args.no_aggregate or args.only:
        return 0
    if args.experiment == "ea":
        from experiments import ea

        hyp = ea.aggregate([out / "ea_ref", out / "ea_tau"], args.results / "E-A")
        print(hyp, flush=True)
    elif args.experiment == "eb1":
        from experiments import eb_aggregate

        eb_aggregate.aggregate_eb1(out / "eb1", args.results / "E-B1")
    elif args.experiment == "eb2":
        from experiments import eb_aggregate

        eb_aggregate.aggregate_eb2([out / "eb2_lab", out / "eb2_ideal1"], args.results / "E-B2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
