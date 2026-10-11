"""Collect the headline numbers of the latest run of every benchmark suite (WP11).

Usage::

    python -m bench.summarize [--root runs/bench] [--out runs/bench/bench_summary.json]

For each run name (suite plus tag: ``b1``, ``b2``, ``b4_lockstep``, ``b3_windows``, ...)
the most recent complete run directory under ``--root`` (with ``manifest.json``) is read;
its ``summary.json``, the manifest's git commit/dirty flag, platform and the
background-load summary of ``meta.json`` are copied into one JSON file, with the run
directory path, so the paper pipeline (WP12) and the lead can map every quoted number to
its data. Nothing is recomputed or rounded here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from bench._common import RUNS_ROOT

SUITES = ("b1", "b2", "b3", "b4", "b6", "eb3")


def latest_runs(root: Path) -> dict[str, list[Path]]:
    """Complete run directories per run name (suite plus tag, e.g. ``b4_lockstep``), newest
    last (directory names start with the UTC stamp; a ``-N`` collision suffix is ignored)."""
    out: dict[str, list[Path]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (d / "manifest.json").exists() or not (d / "summary.json").exists():
            continue
        if "_" not in d.name:
            continue
        name = re.sub(r"-\d+$", "", d.name.split("_", 1)[1])
        if name.split("_", 1)[0] in SUITES:
            out.setdefault(name, []).append(d)
    return out


def summarize(root: Path) -> dict[str, Any]:
    """Headline numbers per suite and run (see the module docstring)."""
    res: dict[str, Any] = {}
    for suite, dirs in latest_runs(root).items():
        entries = []
        for d in dirs:
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            entries.append(
                {
                    "run_dir": str(d),
                    "git_commit": man.get("git_commit"),
                    "git_dirty": man.get("git_dirty"),
                    "platform": man.get("platform"),
                    "cpu_model": man.get("cpu_model"),
                    "aws_instance": man.get("aws_instance"),
                    "threads": man.get("threads"),
                    "platform_label": meta.get("platform_label"),
                    "background_load": meta.get("background_load"),
                    "quiet_gate": meta.get("quiet_gate"),
                    "data_classes": meta.get("data_classes"),
                    "workload_data_classes": meta.get("workload_data_classes"),
                    "interrupted": meta.get("interrupted"),
                    "summary": json.loads((d / "summary.json").read_text(encoding="utf-8")),
                }
            )
        res[suite] = {"latest": entries[-1] if entries else None, "n_runs": len(entries)}
    return res


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(prog="bench.summarize", description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or a.root / "bench_summary.json"
    data = summarize(a.root)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
