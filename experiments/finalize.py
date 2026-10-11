"""Collect the WP11 results into ``paper/sweeps/results/`` and write ``summary.json``.

    python -m experiments.finalize [--runs runs/wp11]

* ``E-A-IQ/``: VA-1 table from ``<runs>/ea_iq`` (:func:`experiments.ea_iq.aggregate`);
* ``E-C/``: the E-C run directory (Table VIII data) copied as is;
* ``E-F/ber/``: BER table (:func:`experiments.ef.aggregate_ber`);
* ``E-F/latency/``: the latency run directory copied as is;
* ``summary.json``: the key numbers of every experiment (machine-readable, for the paper
  pipeline's macros; every value is copied from the result files, nothing is typed).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.common import PAPER, export_git_info, read_json, write_json  # noqa: E402

RESULTS = PAPER / "sweeps" / "results"


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


def summary() -> dict[str, Any]:
    """Key numbers from the result files."""
    out: dict[str, Any] = {}
    hyp = RESULTS / "E-A" / "hypotheses.json"
    if hyp.exists():
        h = read_json(hyp)
        out["E-A"] = {
            "hypotheses": {k: v for k, v in h.items() if not k.startswith("H-A3_report")},
            "H-A3_reports": {k: v["decision"] for k, v in h.items() if k.startswith("H-A3_report")},
            "tau_half": _rows(RESULTS / "E-A" / "tau_half.csv"),
        }
    va1 = _rows(RESULTS / "E-A-IQ" / "va1.csv")
    if va1:
        out["VA-1"] = {
            "rows": va1,
            "all_accept": all(r["accept"] == "True" for r in va1),
        }
    va5 = RESULTS / "E-B1" / "va5.json"
    if va5.exists():
        out["VA-5"] = read_json(va5)
    eb2 = _rows(RESULTS / "E-B2" / "eb2_per.csv")
    if eb2:
        out["E-B2"] = {
            "n_points": len(eb2),
            "f9b_rows_guard_0p5ms": [r for r in eb2 if r["guard_s"] == "1/2000"],
        }
    t8 = _rows(RESULTS / "E-C" / "table8.csv")
    if t8:
        out["E-C"] = {"table8": t8}
    ber = _rows(RESULTS / "E-F" / "ber" / "ber.csv")
    if ber:
        out["E-F"] = {
            "ber": ber,
            "genie_theory_in_ci95_all": all(r["genie_theory_in_ci95"] == "True" for r in ber),
        }
    lat = RESULTS / "E-F" / "latency" / "latency.json"
    if lat.exists():
        out.setdefault("E-F", {})["latency"] = read_json(lat)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=REPO / "runs" / "wp11")
    args = ap.parse_args(argv)
    runs = args.runs.resolve()
    export_git_info(runs)  # git state read once; a dirty patch goes to runs/, not the paper repo
    if (runs / "ea_iq").exists():
        from experiments import ea_iq

        ea_iq.aggregate(runs / "ea_iq", RESULTS / "E-A-IQ")
    _copy_run(runs / "ec", RESULTS / "E-C")
    if (runs / "ef_ber").exists():
        from experiments import ef

        ef.aggregate_ber(runs / "ef_ber", RESULTS / "E-F" / "ber")
    _copy_run(runs / "ef_latency", RESULTS / "E-F" / "latency")
    s = summary()
    write_json(RESULTS / "summary.json", s)
    print(json.dumps({k: list(v) if isinstance(v, dict) else v for k, v in s.items()}, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
