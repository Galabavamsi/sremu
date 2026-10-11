"""Checks of the Linux parts of the pinned-commit regeneration (``regenerate_paper_data.sh``):
GNU Radio on the job host, the E-E export and the paper junit.

    python -m experiments.regen_checks gnuradio --out _regen/gnuradio.json
        [--gr-python /usr/bin/python3] [--install auto|never]
    python -m experiments.regen_checks ee --root OUT/ee --export OUT/paper_results/E-E
        --git-state _regen/git_state.json --out _regen/ee_check.json
        [--reps 2] [--rates 1e6,...] [--ref PAPER/sweeps/results/E-E]
    python -m experiments.regen_checks junit --xml OUT/junit/sremu_paper.xml
        --out _regen/junit_check.json [--mode full|smoke] [--scope full|subset]
        [--expect-gnuradio yes|no] [--claims PAPER/claims.yaml] [--ref-junit XML]
    python -m experiments.regen_checks host-label --provenance _regen/provenance.json

``gnuradio``: probes ``from gnuradio import gr, zeromq, digital`` in the GNU Radio
interpreter (never sremu's: sremu needs numpy >= 2.2, Ubuntu 24.04's ``gnuradio`` package is
bound to the system numpy 1.26); with ``--install auto`` on Linux, when it is missing, runs
``apt-get install -y gnuradio`` non-interactively (as root, else ``sudo -n``; three attempts,
waiting up to 15 min for the dpkg lock that unattended upgrades hold after boot); as root it
also installs git when it is missing (the junit's manifest tests run git in temporary
repositories and skip without it; the job itself never runs git). Records the GNU Radio,
VOLK, ZeroMQ and system numpy package versions and the OS. Exit 0 when GNU Radio is usable.

``ee``: the E-E export (``experiments.ee_gnuradio export``) of the job: every run ok and
complete, counters zero, GNU Radio's IQ equal to the emulator's record, commands applied with
lateness 0, the repetitions bitwise identical (sha256 of the received IQ, the Costas output
and the decisions, recomputed here per run), the throughput rows of every requested rate,
the git state of the export equal to the job's (``_regen/git_state.json``) with the same
``src/sremu`` tree hash, no D24 naming. Reported, not required: the git state the served
emulator wrote into each run's ``manifest.json`` (``sremu.commands.serve`` reads git itself
and records null in a bundle without ``.git``), and, with ``--ref``, whether the IQ and
decision hashes equal those of the reference run (another host can legitimately differ in
float rounding of GNU Radio's VOLK kernels or numba's code generation).

``junit``: the paper junit of the job (``tools/tasks.py paper-junit``): counts, failures,
skips, skipped tests carrying a Table IV spec id (``tools/gen_results.py`` refuses those),
the GNU Radio tests, the VA-1 record test, and the junit status of the test/pytest evidence
of the claims (``--claims``: every claim of the paper's ``claims.yaml``; default: the claims
this regeneration is meant to turn green, ``CLAIM_EVIDENCE``). The status rules mirror
``paper/tools/paper_common.py`` (``spec_id_status``) and ``paper/tools/check_claims.py``
(``pytest_status``). ``--mode smoke``: the VA-1 record of a smoke run is incomplete by
design, so its test may fail (it must run); everything else as in full mode.

``host-label``: the host string the paper's provenance table prints for the junit
(``AWS <instance type>``, ``laptop (WSL2)``, ``laptop (Windows)``, ``Linux``).

Standard library only, except ``ee`` (D24 check of ``experiments.finalize``) and ``--claims``
(PyYAML).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GR_PACKAGES = ("gnuradio", "libvolk*", "libzmq5", "python3-numpy")
APT_ATTEMPTS = 3
APT_LOCK_TIMEOUT_S = 900

EE_CHECKS = (
    "all_ok",
    "all_complete",
    "all_counters_zero",
    "all_rx_equal_emulator_record",
    "all_commands_applied_lateness0",
    "bitwise_identical_across_repetitions",
)
"""``summary.json`` ``checks`` of the E-E export that must all be true (gen_results.py
refuses the E-E record otherwise)."""

GR_TESTS = ("tests/integration/test_gnuradio_raw.py", "tests/integration/test_gnuradio_analysis.py")
VA1_TEST = "tests/validation/test_va1_record.py"
TABLE4_IDS = (
    [f"PH-{i:02d}" for i in range(1, 33)]
    + [f"RT-{i:02d}" for i in range(1, 20)]
    + [f"IO-{i:02d}" for i in range(1, 9)]
    + [f"ST-{i:02d}" for i in range(1, 6)]
    + [f"G{i}" for i in range(1, 9)]
)
"""Spec ids of Table IV (``gen_results.FAMILIES`` plus PH-12): a skipped test carrying one of
them makes ``gen_results.py`` stop."""

CLAIM_EVIDENCE: dict[str, list[str]] = {
    "C-VA1": ["test:VA-1"],
    "C-CONTRIB-4": ["test:VA-5", "test:VA-1", "test:E-B"],
    "C-ABS-EA": ["test:VA-1", "test:E-A"],
    "C-RAW": ["test:IO-08", "pytest:tests/integration/test_gnuradio_raw.py"],
    "C-EE": [
        "test:V-CFO-3",
        "test:IO-02",
        "test:IO-08",
        "pytest:tests/integration/test_gnuradio_raw.py",
    ],
    "C-EE-LINK": [
        "pytest:tests/integration/test_gnuradio_raw.py",
        "pytest:tests/integration/test_gnuradio_analysis.py",
    ],
    "C-EE-TIMING": [
        "pytest:tests/integration/test_gnuradio_raw.py::"
        "test_gnuradio_dqpsk_switching_run_through_raw_mode",
        "pytest:tests/integration/test_gnuradio_raw.py::"
        "test_gnuradio_static_runs_are_bitwise_reproducible",
        "pytest:tests/io/test_io08_raw_exit_when_done.py",
        "test:IO-03",
    ],
    "C-EE-SPEED": ["pytest:tests/integration/test_gnuradio_raw.py"],
}
"""Test/pytest evidence of the claims this regeneration turns green (paper ``claims.yaml``
of 2026-10-11; figure, table, citation and spec evidence is not junit evidence)."""


# =========================================================================== small helpers


def _json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, default=str) + "\n")


def _run(cmd: list[str], env: dict[str, str] | None = None, timeout: float = 3600) -> int:
    print("+ " + " ".join(cmd), flush=True)
    try:
        return subprocess.run(cmd, env=env, timeout=timeout, check=False).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  failed: {exc}", flush=True)
        return 127


# =========================================================================== gnuradio


def gr_probe(gr_python: str) -> dict[str, Any]:
    """GNU Radio version and interpreter facts of ``gr_python`` (``available`` False with the
    reason when it cannot import gr, zeromq and digital)."""
    if not Path(gr_python).is_file():
        return {"available": False, "reason": f"no interpreter at {gr_python}"}
    code = (
        "from gnuradio import gr, zeromq, digital; import sys, numpy; print(gr.version()); "
        "print(sys.version.split()[0]); print(numpy.__version__)"
    )
    try:
        res = subprocess.run(
            [gr_python, "-I", "-c", code], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"{gr_python} does not run: {exc}"}
    if res.returncode != 0:
        return {"available": False, "reason": res.stderr.strip()[-400:]}
    lines = res.stdout.strip().splitlines()
    return {
        "available": True,
        "gnuradio_version": lines[-3],
        "gr_python_version": lines[-2],
        "gr_python_numpy": lines[-1],
    }


def dpkg_versions(patterns: tuple[str, ...] = GR_PACKAGES) -> dict[str, str]:
    """Installed Debian package versions matching ``patterns`` (empty without dpkg-query)."""
    exe = shutil.which("dpkg-query")
    if exe is None:
        return {}
    res = subprocess.run(
        [exe, "-W", "-f", "${db:Status-Abbrev} ${Package} ${Version}\n", *patterns],
        capture_output=True,
        text=True,
        check=False,
    )
    out = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith("ii"):
            out[parts[1]] = parts[2]
    return out


def os_release() -> dict[str, str]:
    p = Path("/etc/os-release")
    if not p.is_file():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        k, sep, v = line.partition("=")
        if sep and k in ("ID", "VERSION_ID", "PRETTY_NAME", "VERSION_CODENAME"):
            out[k] = v.strip().strip('"')
    return out


def apt_install(packages: list[str]) -> list[dict[str, Any]]:
    """``apt-get update`` + ``apt-get install -y`` (non-interactive, waiting for the dpkg
    lock), up to ``APT_ATTEMPTS`` times. Returns one record per command run."""
    prefix: list[str] = []
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        sudo = shutil.which("sudo")
        if sudo is None:
            return [{"cmd": "apt-get", "rc": None, "reason": "not root and no sudo"}]
        prefix = [sudo, "-n"]
    # no prompts, and no service restarts by needrestart's apt hook while the job runs
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive", "NEEDRESTART_SUSPEND": "1"}
    lock = ["-o", f"DPkg::Lock::Timeout={APT_LOCK_TIMEOUT_S}"]
    log: list[dict[str, Any]] = []
    for attempt in range(1, APT_ATTEMPTS + 1):
        for cmd in (
            [*prefix, "apt-get", *lock, "update"],
            [*prefix, "apt-get", *lock, "install", "-y", *packages],
        ):
            t0 = time.monotonic()
            rc = _run(cmd, env=env)
            log.append({"attempt": attempt, "cmd": " ".join(cmd), "rc": rc,
                        "wall_s": round(time.monotonic() - t0, 1)})  # fmt: skip
            if rc != 0:
                break
        else:
            return log
        if attempt < APT_ATTEMPTS:
            time.sleep(30 * attempt)
    return log


def cmd_gnuradio(a: argparse.Namespace) -> int:
    t0 = time.monotonic()
    rec: dict[str, Any] = {
        "gr_python": a.gr_python,
        "platform": platform.platform(),
        "os_release": os_release(),
        "installed_by_job": False,
        "apt": [],
    }
    probe = gr_probe(a.gr_python)
    # git is not used by the job; the junit's manifest tests run it in temporary repositories
    # and skip without it. Installed only as root (the AWS job host), never via sudo.
    root = hasattr(os, "geteuid") and os.geteuid() == 0
    want = [] if probe["available"] else ["gnuradio"]
    if shutil.which("git") is None and root:
        want.append("git")
    if want and a.install == "auto":
        if platform.system() != "Linux" or shutil.which("apt-get") is None:
            probe["install_skipped"] = "not Linux with apt-get"
        else:
            print(f"installing {want} (GNU Radio: {probe.get('reason', 'present')[-200:]})")
            rec["apt"] = apt_install(want)
            rec["installed_by_job"] = "gnuradio" in want
            rec["apt_packages"] = want
            probe = gr_probe(a.gr_python)
    rec.update(probe)
    rec["git_on_path"] = shutil.which("git") is not None
    rec["packages"] = dpkg_versions((*GR_PACKAGES, "git"))
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    _write(Path(a.out), rec)
    print(json.dumps({k: rec.get(k) for k in ("available", "gnuradio_version", "packages")}))
    return 0 if rec["available"] else 1


# =========================================================================== E-E


def _tree_hashes(d: Path) -> dict[str, str]:
    return {
        f.relative_to(d).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(d.rglob("*"))
        if f.is_file()
    }


def ee_check(
    export: Path,
    git_state: dict[str, Any],
    *,
    reps: int,
    rates: list[float],
    ref: Path | None = None,
    d24: bool = True,
) -> dict[str, Any]:
    """The E-E export against the job's design and git state (module docstring)."""
    s = _json(export / "summary.json")
    meta = _json(export / "meta.json")
    problems: list[str] = []
    if not s:
        return {"ok": False, "problems": [f"{export / 'summary.json'} missing or unreadable"]}
    checks = s.get("checks") or {}
    for k in EE_CHECKS:
        if checks.get(k) is not True:
            problems.append(f"checks.{k} is {checks.get(k)!r} ({checks.get('problems')})")
    want_reps = [f"rep{i}" for i in range(1, reps + 1)]
    if s.get("repetitions") != want_reps:
        problems.append(f"repetitions {s.get('repetitions')}, expected {want_reps}")
    sha = s.get("sha256") or {}
    identical = {
        run: len({json.dumps(h, sort_keys=True) for h in by_rep.values()}) == 1
        and sorted(by_rep) == want_reps
        for run, by_rep in sha.items()
    }
    if not sha or not all(identical.values()):
        problems.append(
            f"hashes differ across repetitions: {[r for r, v in identical.items() if not v]}"
        )
    if s.get("runs_per_repetition") != len(sha) or len(sha) < 2:
        problems.append(
            f"runs_per_repetition {s.get('runs_per_repetition')}, hashed runs {len(sha)}"
        )
    tp = {float(r["rate_sps"]): r for r in s.get("throughput") or []}
    if rates and sorted(tp) != sorted(rates):
        problems.append(f"throughput rates {sorted(tp)}, requested {sorted(rates)}")
    if any(not r.get("ok") for r in tp.values()):
        problems.append("a throughput run is not ok")
    g = s.get("git") or {}
    if (g.get("commit"), g.get("dirty")) != (git_state.get("commit"), git_state.get("dirty")):
        problems.append(
            f"export git {g.get('commit')}/{g.get('dirty')} differs from the job's "
            f"{git_state.get('commit')}/{git_state.get('dirty')}"
        )
    tree_job = (git_state.get("code") or {}).get("src_sremu_tree_sha256")
    tree_meta = (meta.get("code") or {}).get("src_sremu_tree_sha256")
    if tree_job is None or tree_meta != tree_job:
        problems.append(f"meta.json src tree {tree_meta} differs from the job's {tree_job}")
    if not s.get("gnuradio_version"):
        problems.append("no GNU Radio version recorded")
    states = Counter()
    for m in sorted((export / "runs").rglob("manifest.json")):
        d = _json(m)
        states[f"{d.get('git_commit')}|{d.get('git_dirty')}"] += 1
    job_state = f"{git_state.get('commit')}|{git_state.get('dirty')}"
    bad_d24: list[str] = []
    if d24:
        from experiments.finalize import d24_violations

        bad_d24 = d24_violations(export)
        if bad_d24:
            problems.append(f"D24 naming in the export: {bad_d24[:5]}")
    host = s.get("host") or {}
    aws = host.get("aws_instance") or {}
    rep: dict[str, Any] = {
        "export": str(export),
        "repetitions": s.get("repetitions"),
        "runs_per_repetition": s.get("runs_per_repetition"),
        "identical_across_repetitions": identical,
        "gnuradio_version": s.get("gnuradio_version"),
        "git": {"commit": g.get("commit"), "dirty": g.get("dirty")},
        "src_sremu_tree_sha256": tree_meta,
        "host": {
            "cpu_model": host.get("cpu_model"),
            "platform": host.get("platform"),
            "aws_instance_type": aws.get("type") if isinstance(aws, dict) else None,
        },
        "throughput_meas": {
            f"{k:g}": {
                "sremu_rtf_median": r.get("sremu_rtf_median"),
                "gr_rtf_steady": r.get("gr_rtf_steady"),
            }
            for k, r in sorted(tp.items())
        },
        "run_manifest_git_states": dict(states),
        "run_manifests_record_job_commit": bool(states) and set(states) == {job_state},
        "run_manifests_note": "sremu.commands.serve records the git state it reads itself "
        "(null in a job bundle without .git); the export's summary.json/meta.json carry the "
        "job's state",
        "d24_violations": len(bad_d24),
        "files": len(_tree_hashes(export)) if export.is_dir() else 0,
    }
    if ref is not None and (ref / "summary.json").is_file():
        rs = _json(ref / "summary.json")
        rsha = rs.get("sha256") or {}
        per = {}
        for run, by_rep in sha.items():
            new_h = next(iter(by_rep.values()), None)
            ref_h = next(iter((rsha.get(run) or {}).values()), None)
            per[run] = None if ref_h is None or new_h is None else {
                k: new_h.get(k) == ref_h.get(k) for k in sorted(set(new_h) | set(ref_h))
            }  # fmt: skip
        rh = rs.get("host") or {}
        rep["vs_reference"] = {
            "ref": str(ref),
            "ref_git": (rs.get("git") or {}).get("commit"),
            "ref_cpu_model": rh.get("cpu_model"),
            "ref_platform": rh.get("platform"),
            "gnuradio_version_equal": rs.get("gnuradio_version") == s.get("gnuradio_version"),
            "hashes_equal_per_run": per,
            "all_hashes_equal": bool(per)
            and all(v is not None and all(v.values()) for v in per.values()),
            "note": "informational: the reference was run on another host/commit; equal hashes "
            "show the same IQ and decisions bit for bit",
        }
    rep["problems"] = problems
    rep["ok"] = not problems
    return rep


def cmd_ee(a: argparse.Namespace) -> int:
    rates = [float(x) for x in a.rates.split(",") if x.strip()] if a.rates else []
    rep = ee_check(
        Path(a.export),
        _json(Path(a.git_state)),
        reps=a.reps,
        rates=rates,
        ref=Path(a.ref) if a.ref else None,
    )
    rep["root"] = a.root
    _write(Path(a.out), rep)
    print(
        json.dumps({k: rep.get(k) for k in ("ok", "problems", "run_manifests_record_job_commit")})
    )
    if "vs_reference" in rep:
        print(f"hashes equal to the reference: {rep['vs_reference']['all_hashes_equal']}")
    return 0 if rep["ok"] else 1


# =========================================================================== junit


@dataclass(frozen=True)
class Case:
    nodeid: str  # lower-case "<classname>::<name>", as paper_common.load_junit
    status: str  # passed | failed | skipped
    message: str = ""


def load_junit(path: Path) -> list[Case]:
    cases = []
    for tc in ET.parse(path).getroot().iter("testcase"):
        nodeid = f"{tc.get('classname', '')}::{tc.get('name', '')}".lower()
        bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        if bad is not None:
            cases.append(Case(nodeid, "failed", (bad.get("message") or "")[:300]))
        elif tc.find("skipped") is not None:
            cases.append(Case(nodeid, "skipped", (tc.find("skipped").get("message") or "")[:300]))
        else:
            cases.append(Case(nodeid, "passed"))
    return cases


def spec_id_pattern(spec_id: str) -> re.Pattern[str]:
    """As ``paper/tools/paper_common.py`` ``spec_id_pattern``."""
    parts = re.findall(r"[A-Za-z]+|\d+", spec_id)
    pieces = [f"0*{int(p)}(?!\\d)" if p.isdigit() else re.escape(p.lower()) for p in parts]
    tail = "" if parts[-1].isdigit() else "(?![a-z])"
    return re.compile(r"(?<![a-z0-9])" + r"[-_]?".join(pieces) + tail)


def is_implementation_test(nodeid: str) -> bool:
    """As ``paper_common.is_implementation_test``: oracle self-tests and golden-harness tests
    are not evidence."""
    module, _, name = nodeid.lower().partition("::")
    packages = re.split(r"[./\\]", module)[:-1]
    if "oracles" in packages:
        return False
    return not ("golden" in packages and "sremu" not in name)


def spec_id_status(spec_id: str, cases: list[Case]) -> str:
    """As ``paper_common.spec_id_status`` (without the backend rules)."""
    pat = spec_id_pattern(spec_id)
    hits = [c for c in cases if pat.search(c.nodeid) and is_implementation_test(c.nodeid)]
    if not hits:
        return "missing"
    if any(c.status == "failed" for c in hits):
        return "failed"
    if any(c.status == "skipped" for c in hits):
        return "skipped"
    return "passed"


_PYTEST_REF = re.compile(r"^(tests/[A-Za-z0-9_/]+\.py)(?:::([A-Za-z_][A-Za-z0-9_]*))?$")


def pytest_status(ref: str, cases: list[Case]) -> str:
    """As ``paper/tools/check_claims.py`` ``pytest_status``."""
    m = _PYTEST_REF.match(ref)
    if not m:
        return "missing"
    module = m.group(1)[: -len(".py")].replace("/", ".").lower()
    name = (m.group(2) or "").lower()
    hits = []
    for c in cases:
        cls, _, test = c.nodeid.partition("::")
        if not (cls == module or cls.startswith(module + ".")):
            continue
        if name and not (test == name or test.startswith(name + "[")):
            continue
        hits.append(c)
    if not hits:
        return "missing"
    if any(c.status == "failed" for c in hits):
        return "failed"
    if any(c.status == "skipped" for c in hits):
        return "skipped"
    return "passed"


def evidence_status(ev: str, cases: list[Case]) -> str | None:
    kind, _, ident = ev.partition(":")
    if kind == "test":
        return spec_id_status(ident, cases)
    if kind == "pytest":
        return pytest_status(ident, cases)
    return None  # not junit evidence


def claims_evidence(claims_yaml: Path | None) -> tuple[dict[str, list[str]], str]:
    if claims_yaml is None or not claims_yaml.is_file():
        return dict(CLAIM_EVIDENCE), "embedded CLAIM_EVIDENCE"
    import yaml

    doc = yaml.safe_load(claims_yaml.read_text(encoding="utf-8")) or {}
    out = {}
    for cid, c in (doc.get("claims") or {}).items():
        ev = [str(e) for e in (c or {}).get("evidence") or []]
        ev = [e for e in ev if e.split(":", 1)[0] in ("test", "pytest")]
        if ev:
            out[str(cid)] = ev
    return out, str(claims_yaml)


def junit_check(
    xml: Path,
    *,
    mode: str = "full",
    scope: str = "full",
    expect_gnuradio: bool = True,
    claims_yaml: Path | None = None,
    ref_junit: Path | None = None,
) -> dict[str, Any]:
    """The job's paper junit (module docstring)."""
    if not xml.is_file():
        return {"ok": False, "xml": str(xml), "problems": [f"{xml} missing"]}
    cases = load_junit(xml)
    n = Counter(c.status for c in cases)
    problems: list[str] = []
    failed = [c for c in cases if c.status == "failed"]
    va1_failed = [c for c in failed if pytest_status(VA1_TEST, [c]) == "failed"]
    if mode == "smoke":
        if len(failed) > len(va1_failed):
            problems.append(f"{len(failed) - len(va1_failed)} failed tests besides the VA-1 one")
    elif failed:
        problems.append(f"{len(failed)} failed tests")
    skipped = [c for c in cases if c.status == "skipped"]
    skipped_ids = sorted(
        {c.nodeid for c in skipped for i in TABLE4_IDS if spec_id_pattern(i).search(c.nodeid)}
    )
    if skipped_ids:
        problems.append(f"skipped tests with a Table IV spec id: {skipped_ids[:5]}")
    required = {VA1_TEST: pytest_status(VA1_TEST, cases)}
    for t in GR_TESTS:
        required[t] = pytest_status(t, cases)
    want_va1 = {"passed", "failed"} if mode == "smoke" else {"passed"}
    if required[VA1_TEST] not in want_va1:
        problems.append(f"{VA1_TEST}: {required[VA1_TEST]}")
    raw, analysis = GR_TESTS
    if required[analysis] != "passed":  # no GNU Radio needed: must pass everywhere
        problems.append(f"{analysis}: {required[analysis]}")
    if expect_gnuradio and required[raw] != "passed":
        problems.append(f"{raw}: {required[raw]} (GNU Radio expected on this host)")
    elif not expect_gnuradio and required[raw] not in ("passed", "skipped"):
        problems.append(f"{raw}: {required[raw]}")
    ev, ev_src = claims_evidence(claims_yaml)
    claims = {}
    for cid, items in ev.items():
        st = {e: evidence_status(e, cases) for e in items}
        claims[cid] = {"evidence": st, "junit_green": all(v == "passed" for v in st.values())}
    side = xml.with_suffix(".meta.json")
    rep: dict[str, Any] = {
        "xml": str(xml),
        "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "sidecar": _json(side) if side.is_file() else None,
        "mode": mode,
        "scope": scope,
        "expect_gnuradio": expect_gnuradio,
        "tests": len(cases),
        "passed": n["passed"],
        "failed": n["failed"],
        "skipped": n["skipped"],
        "failed_tests": [f"{c.nodeid}: {c.message}" for c in failed][:50],
        "skipped_tests": [f"{c.nodeid}: {c.message}" for c in skipped][:200],
        "skipped_with_table4_id": skipped_ids,
        "required_tests": required,
        "claims_source": ev_src,
        "claims": claims,
        "claims_junit_green": sorted(k for k, v in claims.items() if v["junit_green"]),
        "claims_not_junit_green": sorted(k for k, v in claims.items() if not v["junit_green"]),
        # the claims this regeneration is meant to turn green (CLAIM_EVIDENCE)
        "target_claims_junit_green": {
            k: claims[k]["junit_green"] for k in CLAIM_EVIDENCE if k in claims
        },
    }
    if scope != "full":
        rep["note"] = "subset junit (smoke): not a paper record"
    if ref_junit is not None and ref_junit.is_file():
        rc = load_junit(ref_junit)
        ref_pass = {c.nodeid for c in rc if c.status == "passed"}
        new_pass = {c.nodeid for c in cases if c.status == "passed"}
        rep["vs_reference_junit"] = {
            "ref": str(ref_junit),
            "ref_counts": dict(Counter(c.status for c in rc)),
            "passed_in_ref_not_here": sorted(ref_pass - new_pass)[:100],
            "n_passed_in_ref_not_here": len(ref_pass - new_pass),
            "n_passed_here_not_in_ref": len(new_pass - ref_pass),
            "note": "informational (other commit, platform or test subset)",
        }
    rep["problems"] = problems
    rep["ok"] = not problems
    return rep


def cmd_junit(a: argparse.Namespace) -> int:
    rep = junit_check(
        Path(a.xml),
        mode=a.mode,
        scope=a.scope,
        expect_gnuradio=a.expect_gnuradio == "yes",
        claims_yaml=Path(a.claims) if a.claims else None,
        ref_junit=Path(a.ref_junit) if a.ref_junit else None,
    )
    _write(Path(a.out), rep)
    keys = ("ok", "tests", "passed", "failed", "skipped", "problems", "target_claims_junit_green")
    print(json.dumps({k: rep.get(k) for k in keys}))
    return 0 if rep["ok"] else 1


# =========================================================================== host label


def host_label(host: dict[str, Any]) -> str:
    """The provenance-table host string of ``paper/tools/gen_results.py`` (``_host``), with
    WSL2 named as the E-E rows name it."""
    aws = host.get("aws_instance") or {}
    if isinstance(aws, dict) and aws.get("type"):
        return f"AWS {aws['type']}"
    plat = str(host.get("platform") or "")
    if "microsoft" in plat.lower():
        return "laptop (WSL2)"
    if plat.startswith("Windows"):
        return "laptop (Windows)"
    if plat.startswith("Linux"):
        return "Linux"
    return "n/r"


def cmd_host_label(a: argparse.Namespace) -> int:
    print(host_label(_json(Path(a.provenance)).get("host") or {}))
    return 0


# =========================================================================== main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gnuradio")
    g.add_argument("--gr-python", default=os.environ.get("SREMU_GR_PYTHON", "/usr/bin/python3"))
    g.add_argument("--install", choices=("auto", "never"), default="never")
    g.add_argument("--out", required=True)
    e = sub.add_parser("ee")
    e.add_argument("--root", required=True)
    e.add_argument("--export", required=True)
    e.add_argument("--git-state", required=True)
    e.add_argument("--reps", type=int, default=2)
    e.add_argument("--rates", default="")
    e.add_argument("--ref", default=None)
    e.add_argument("--out", required=True)
    j = sub.add_parser("junit")
    j.add_argument("--xml", required=True)
    j.add_argument("--mode", choices=("full", "smoke"), default="full")
    j.add_argument("--scope", choices=("full", "subset"), default="full")
    j.add_argument("--expect-gnuradio", choices=("yes", "no"), default="yes")
    j.add_argument("--claims", default=None)
    j.add_argument("--ref-junit", default=None)
    j.add_argument("--out", required=True)
    h = sub.add_parser("host-label")
    h.add_argument("--provenance", required=True)
    a = ap.parse_args(argv)
    return {"gnuradio": cmd_gnuradio, "ee": cmd_ee, "junit": cmd_junit,
            "host-label": cmd_host_label}[a.cmd](a)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
