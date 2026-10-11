"""E-E: stock GNU Radio through the raw REQ/REP mode (DESIGN.md 8.4, 11 E-E; paper Section VII,
"Integration with external IQ applications"). Linux / WSL2 only.

    python -m experiments.ee_gnuradio run --root ROOT [--reps 2] [--throughput]
        [--rates 1e6,1.92e6,3.84e6,5.76e6] [--throughput-s 2] [--note TEXT]
    python -m experiments.ee_gnuradio export --root ROOT --dest ../paper/sweeps/results/E-E
    python -m experiments.ee_gnuradio export --root R1 [--root R2 ...] --dest DIR --throughput-only

``run`` executes ``--reps`` independent repetitions (``rep1`` ...) of the whole design of
``examples/gnuradio_zmq/run_ris_ber.py``: one fresh served run with one fresh GNU Radio
3.10 DQPSK modem (stock gr-zeromq and gr-digital blocks, its own interpreter, ``--gr-python``
or ``$SREMU_GR_PYTHON``) per static RIS configuration (``ris_configs.STATIC``: 1-bit
focusing, focusing with 10-50 % of the cells inverted, all cells in state 0, random; 0.3 s
of virtual time each) and one run with three RIS switches scheduled over the control socket
at exact RIS times (0.4 s per configuration). Repetitions share nothing but the code, so equal
sha256 of the IQ GNU Radio received and of its decisions across repetitions is the
bitwise-reproducibility check. ``--throughput`` adds one run per rate of ``--rates`` (focusing,
``--throughput-s`` of virtual time, no IQ recording) for the lockstep real-time factor: virtual
seconds per wall second of the GNU Radio + sremu loop, from the emulator's per-0.1 s metrics
(median after the first window, which holds the numba JIT) and from GNU Radio's RX progress.
Lockstep runs are virtual time: the factor says how fast the loop advanced on this host, not
that it kept pace with a clock.

``export`` checks every run (complete, all counters zero, GNU Radio's IQ equal to the
emulator's own RX record, every command applied with lateness 0, identical hashes across
repetitions) and writes ``summary.json``, ``table_ee.csv``, ``meta.json`` and the small
per-run files (``analysis.json``, ``run.json``, the emulator's ``manifest.json``,
``events.jsonl``, ``metrics.jsonl``) to ``--dest``; IQ files stay in ``ROOT``.

Data classes: SIM-sremu (everything computed from the emulated IQ, including GNU Radio's
decisions), ANALYTIC (array-factor prediction, DQPSK theory), MEAS (wall-clock throughput of
the stated host). The array-factor comparison is an internal consistency check, never a
measured RIS gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
EXAMPLE = REPO / "examples" / "gnuradio_zmq"
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))

from experiments.common import (  # noqa: E402
    environment,
    export_git_info,
    read_json,
    write_csv,
    write_json,
    write_meta,
)

STATIC_DURATION_S = "0.3"
SEGMENT_S = "0.4"
DEFAULT_RATES = "1e6,1.92e6,3.84e6,5.76e6"
SMALL_FILES = ("analysis.json", "run.json")
REC_FILES = ("manifest.json", "events.jsonl", "metrics.jsonl")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jobs(rep: str) -> list[tuple[str, dict[str, Any]]]:
    from ris_configs import STATIC
    from run_ris_ber import SWITCHING

    jobs = [
        (
            f"static_{c.replace(':', '')}",
            {"kind": "static", "configs": [c], "duration_s": STATIC_DURATION_S},
        )
        for c in STATIC
    ]
    dur = str(Decimal(SEGMENT_S) * len(SWITCHING) + Decimal("0.05"))
    jobs.append(
        (
            "switching",
            {
                "kind": "switching",
                "configs": list(SWITCHING),
                "duration_s": dur,
                "segment_s": SEGMENT_S,
            },
        )
    )
    return jobs


def cmd_run(a: argparse.Namespace) -> int:
    from run_ris_ber import one_run, summary_line

    if platform.system() != "Linux":
        print("E-E needs Linux or WSL2 with GNU Radio", file=sys.stderr)
        return 3
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    git = export_git_info(root)
    write_json(
        root / "host.json",
        {"environment": environment(), "git": git, "note": a.note, "argv": sys.argv},
    )
    ok = True
    for r in range(1, a.reps + 1):
        rep = f"rep{r}"
        for name, kw in _jobs(rep):
            res = one_run(root / rep / name, gr_python=a.gr_python, **kw)
            ok &= bool(res["ok"])
            print(("OK   " if res["ok"] else "FAIL ") + f"{rep} " + summary_line(res), flush=True)
    if a.throughput:
        for rate in [float(x) for x in a.rates.split(",") if x.strip()]:
            res = one_run(
                root / "throughput" / f"rate_{rate:g}",
                kind="static",
                configs=["focus"],
                duration_s=a.throughput_s,
                rate=rate,
                gr_python=a.gr_python,
                record_iq=False,
            )
            ok &= bool(res["ok"])
            print(
                ("OK   " if res["ok"] else "FAIL ")
                + f"rate {rate:g}: "
                + json.dumps(res.get("throughput")),
                flush=True,
            )
    return 0 if ok else 1


def _seg(a: dict[str, Any]) -> dict[str, Any]:
    return a["segments"][0]


def _copy_run(src: Path, out: Path, dest: Path, data_class: str) -> list[dict[str, str]]:
    """Copy the small files of one run; their input records for meta.json."""
    out.mkdir(parents=True)
    for f in SMALL_FILES:
        shutil.copy2(src / f, out / f)
    rec = next((src / "rec").glob("*/manifest.json")).parent
    for f in REC_FILES:
        shutil.copy2(rec / f, out / f)
    return [
        {"path": f.relative_to(dest).as_posix(), "sha256": _sha(f), "data_class": data_class}
        for f in sorted(out.iterdir())
    ]


def _throughput_rows(tput: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for _name, an in sorted(tput.items(), key=lambda kv: kv[1]["rate_sps"]):
        tp = an.get("throughput", {})
        rows.append(
            {
                "rate_sps": an["rate_sps"],
                "duration_s": an["n_run"] / an["rate_sps"],
                "sremu_rtf_median": tp.get("sremu_rtf_median"),
                "sremu_rtf_windows": tp.get("sremu_rtf_windows"),
                "gr_rtf_steady": tp.get("gr_rtf_steady"),
                "ser": _seg(an)["ser"],
                "page_size": (an.get("manifest") or {}).get("page_size"),
                "ok": an.get("ok"),
            }
        )
    return rows


def export_throughput(roots: list[Path], dest: Path) -> int:
    """``export --throughput-only``: the throughput runs of one or more sweeps (one ``run``
    root each, in the order given), their hosts and notes."""
    sweeps: list[dict[str, Any]] = []
    problems: list[str] = []
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    inputs = []
    for k, root in enumerate(roots, start=1):
        tput = {
            d.name: read_json(d / "analysis.json")
            for d in sorted((root / "throughput").glob("*"))
            if (d / "analysis.json").is_file()
        }
        if not tput:
            print(f"no throughput runs under {root}", file=sys.stderr)
            return 2
        host = read_json(root / "host.json") if (root / "host.json").is_file() else {}
        problems += [
            f"{root.name}/throughput/{n} not ok" for n, an in tput.items() if not an.get("ok")
        ]
        for name in tput:
            inputs += _copy_run(
                root / "throughput" / name, dest / "runs" / f"sweep{k}" / name, dest, "MEAS"
            )
        sweeps.append(
            {
                "sweep": k,
                "root": str(root),
                "throughput": _throughput_rows(tput),
                "host": host.get("environment"),
                "git": host.get("git"),
                "note": host.get("note"),
            }
        )
    write_json(
        dest / "summary.json",
        {
            "experiment": "E-E (throughput only)",
            "checks": {"all_ok": not problems, "problems": problems},
            "sweeps": sweeps,
            "host": sweeps[0]["host"],
        },
    )
    write_meta(
        dest,
        experiment="E-E",
        data_classes=["MEAS"],
        description="Lockstep real-time factor of the GNU Radio + sremu loop (virtual seconds "
        "per wall second) on one host: single runs per rate, one or more sweeps",
        assumptions=[
            "lockstep: virtual time follows the sample exchange; the real-time factor "
            "is not a real-time claim",
            "single run per rate and sweep; host load not controlled",
        ],
        inputs=inputs,
        files={"summary.json": "throughput rows per sweep and host"},
        extra={"roots": [str(r) for r in roots]},
    )
    print(f"exported {len(sweeps)} throughput sweeps to {dest}; problems: {problems or 'none'}")
    return 0 if not problems else 1


def cmd_export(a: argparse.Namespace) -> int:
    dest = Path(a.dest)
    if a.throughput_only:
        return export_throughput([Path(r) for r in a.root], dest)
    if len(a.root) != 1:
        print("export takes one --root (several only with --throughput-only)", file=sys.stderr)
        return 2
    root = Path(a.root[0])
    reps = sorted(p.name for p in root.glob("rep*") if p.is_dir())
    if not reps:
        print(f"no repetitions under {root}", file=sys.stderr)
        return 2
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for rep in reps:
        for d in sorted((root / rep).iterdir()):
            if (d / "analysis.json").is_file():
                runs.setdefault(d.name, {})[rep] = read_json(d / "analysis.json")
    tput = (
        {
            d.name: read_json(d / "analysis.json")
            for d in sorted((root / "throughput").glob("*"))
            if (d / "analysis.json").is_file()
        }
        if (root / "throughput").is_dir()
        else {}
    )
    problems: list[str] = []
    for name, by_rep in runs.items():
        if set(by_rep) != set(reps):
            problems.append(f"{name}: missing in {sorted(set(reps) - set(by_rep))}")
        for rep, an in by_rep.items():
            if not an.get("ok"):
                problems.append(f"{rep}/{name} not ok")
    for name, an in tput.items():
        if not an.get("ok"):
            problems.append(f"throughput/{name} not ok")
    identical = {
        name: len({json.dumps(an["sha256"], sort_keys=True) for an in by_rep.values()}) == 1
        for name, by_rep in runs.items()
    }
    first = reps[0]
    focus_gain = _seg(runs["static_focus"][first])["ls_gain_db"]
    static = []
    for by_rep in runs.values():
        an = by_rep[first]
        if an["kind"] != "static":
            continue
        s = _seg(an)
        static.append(
            {
                "config": s["config"],
                "af_pred_db": s.get("af_pred_db"),
                "gain_rel_focus_db": s["ls_gain_db"] - focus_gain,
                "ls_snr_db": s["ls_snr_db"],
                "es_n0_db": s["es_n0_db"],
                "symbols": s["symbols"],
                "symbol_errors": s["symbol_errors"],
                "ser": s["ser"],
                "ser_theory": s["ser_theory"],
                "implied_loss_db": s["implied_loss_db"],
                "bler_uncoded_200bit": s["bler_uncoded_200bit"],
                "blocks": s["blocks"],
                "mer_db": s["mer_db"],
                "worst_20ms_ser": s["worst_20ms_ser"],
            }
        )
    static.sort(key=lambda r: -r["gain_rel_focus_db"])
    sw = runs.get("switching", {}).get(first)
    rows_tp = _throughput_rows(tput)
    all_an = [an for by_rep in runs.values() for an in by_rep.values()] + list(tput.values())
    host = read_json(root / "host.json") if (root / "host.json").is_file() else {}
    import yaml

    ref = runs["static_focus"][first]
    summary = {
        "experiment": "E-E",
        "repetitions": reps,
        "runs_per_repetition": len(runs),
        "design": {
            "rate_sps": ref["rate_sps"],
            "sps": ref["sps"],
            "static_duration_s": STATIC_DURATION_S,
            "segment_s": SEGMENT_S,
            "modem": read_json(root / first / "static_focus" / "gr_modem.json").get("rx_chain"),
            "scene": yaml.safe_load(
                (root / first / "static_focus" / "scene.yaml").read_text(encoding="utf-8")
            ),
        },
        "checks": {
            "all_ok": not problems,
            "problems": problems,
            "all_complete": all(an["complete"] for an in all_an),
            "all_counters_zero": all(not an.get("counters_nonzero") for an in all_an),
            "all_rx_equal_emulator_record": all(
                an.get("rx_equals_emulator_record", True) for an in all_an
            ),
            "all_commands_applied_lateness0": all(
                c["status"] == "applied" and c["lateness_samples"] == 0
                for an in all_an
                for c in an["commands"]
            ),
            "bitwise_identical_across_repetitions": all(identical.values()),
            "identical_per_run": identical,
        },
        "static": static,
        "switching": None
        if sw is None
        else {
            "segments": sw["segments"],
            "change_points": sw["change_points"],
            "commands": sw["commands"],
        },
        "sha256": {
            name: {rep: an["sha256"] for rep, an in by_rep.items()} for name, by_rep in runs.items()
        },
        "throughput": rows_tp,
        "gnuradio_version": runs["static_focus"][first].get("gnuradio_version"),
        "manifest_example": runs["static_focus"][first].get("manifest"),
        "host": host.get("environment"),
        "git": host.get("git"),
        "note": host.get("note"),
    }
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    inputs = []
    for rep in reps:
        for name in runs:
            inputs += _copy_run(root / rep / name, dest / "runs" / rep / name, dest, "SIM-sremu")
    for name in tput:
        inputs += _copy_run(
            root / "throughput" / name, dest / "runs" / "throughput" / name, dest, "MEAS"
        )
    write_json(dest / "summary.json", summary)
    write_csv(dest / "table_ee.csv", static)
    ex = {p.name: _sha(p) for p in sorted(EXAMPLE.iterdir()) if p.is_file()}
    write_meta(
        dest,
        experiment="E-E",
        data_classes=["SIM-sremu", "ANALYTIC", "MEAS"],
        description="Stock GNU Radio 3.10 DQPSK modem (gr-zeromq REQ/REP, gr-digital) through "
        "sremu's raw REQ/REP mode: link quality vs RIS configuration, in-run switch "
        "timing, bitwise reproducibility, lockstep real-time factor",
        assumptions=[
            "SIM geometry gr_ris_ber.yaml: ideal 1-bit 16 x 16 RIS at 3.5 GHz, TX/RX 5 m at "
            "-30/+20 deg, direct path behind a 40 dB partition, 1 MS/s, NF 7 dB",
            "BLER = fraction of consecutive uncoded 100-symbol (200-bit) windows of GNU Radio's "
            "decision stream with an error (no coding, no CRC)",
            "SNR estimated from the received IQ against the regenerated TX IQ (LS fit)",
            "array-factor prediction: isotropic point cells, no cell pattern or obliquity "
            "(internal consistency check, not a measurement)",
            "lockstep: virtual time follows the sample exchange; the real-time factor is "
            "virtual seconds per wall second on the stated host (MEAS)",
            "GNU Radio decisions of the first 20 ms (30 ms after a switch) are not scored",
        ],
        inputs=inputs,
        files={
            "summary.json": "checks, per-configuration results, switching run, hashes, "
            "throughput, host",
            "table_ee.csv": "one row per static RIS configuration (repetition rep1)",
            "runs/": "per run: analysis.json, run.json and the emulator's manifest.json, "
            "events.jsonl, metrics.jsonl",
        },
        extra={
            "examples_gnuradio_zmq_sha256": ex,
            "git": host.get("git"),
            "note": host.get("note"),
            "root": str(root),
        },
    )
    print(
        f"exported {len(runs)} runs x {len(reps)} repetitions + {len(tput)} throughput runs "
        f"to {dest}; problems: {problems or 'none'}; identical across repetitions: "
        f"{all(identical.values())}"
    )
    return 0 if not problems and all(identical.values()) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E-E: GNU Radio through the raw REQ/REP mode.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--throughput", action="store_true")
    r.add_argument("--rates", default=DEFAULT_RATES)
    r.add_argument("--throughput-s", default="2")
    r.add_argument("--gr-python", default=os.environ.get("SREMU_GR_PYTHON", "/usr/bin/python3"))
    r.add_argument("--note", default="", help="free text stored with the host record")
    e = sub.add_parser("export")
    e.add_argument(
        "--root",
        required=True,
        action="append",
        help="a run root (repeat with --throughput-only for several sweeps)",
    )
    e.add_argument("--dest", required=True)
    e.add_argument(
        "--throughput-only",
        action="store_true",
        help="export only the throughput runs (a frozen host diagnostic)",
    )
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_export(a)


if __name__ == "__main__":
    sys.exit(main())
