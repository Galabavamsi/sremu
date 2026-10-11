#!/usr/bin/env bash
# Regenerate the paper's WP11 results at ONE committed sremu revision (pinned provenance).
#
# The paper's E-A, E-B (lab-like), E-C and E-F results were produced from dirty trees on
# several commits. This script re-runs all of them from one tree, records one git state for
# every run directory, and collects the results with experiments/finalize.py. It never
# writes to the paper repository.
#
#   SREMU_JOB_OUT=/opt/job/out bash experiments/regenerate_paper_data.sh [options]
#
# Options:
#   --jobs N          CPU budget (default: nproc). The sweep pools get N - R workers; the R
#                     reserved CPUs (R = 16 when N > 32, else N/4) carry the 12 VA-1
#                     processes and E-C, which run beside the sweeps.
#   --deadline-min M  stop starting work after M minutes and stop running steps then
#                     (default 0 = none); the closing checks, finalize and compare still run.
#                     Use it on AWS, below the runner's --maxmin, so that partial results
#                     are still uploaded.
#   --smoke           tiny subsets for a local check of this script (NOT paper data): one
#                     E-A geometry, 12 E-B1 points, 15 E-B2 points of 40 packets, 4 s VA-1
#                     runs, 4 E-C failure maps, 6 BER points.
#   --plan            run only the checks (specs, git state, provenance) and print the
#                     steps; no experiment runs. It writes _regen/, so use a fresh
#                     SREMU_JOB_OUT for the real run.
#
# Environment:
#   SREMU_JOB_OUT     output root (required; must not hold an earlier regeneration).
#   SREMU_PYTHON      interpreter (default: python, i.e. the active venv).
#   SREMU_GIT_COMMIT, SREMU_GIT_DIRTY
#                     git state of a job bundle without .git (research/aws/runner.py sets
#                     both). Without them the tree must be a git work tree; the experiment
#                     code then reads its state (read-only git).
#   SREMU_SPEC_DIR    directory with the paper sweep specs, or "embedded" (default:
#                     ../paper/sweeps when it exists, else the copies embedded below). Every
#                     spec is checked against its pinned content hash (SweepSpec.sha256 of
#                     the paper runs).
#   SREMU_REGEN_REF   reference results tree for the final comparison (default:
#                     ../paper/sweeps/results when it exists; the comparison is skipped
#                     otherwise, e.g. on AWS).
#   SREMU_REGEN_ALLOW_DIRTY=1
#                     full run on a dirty tree. The run is then NOT pinned; avoid.
#
# Exit status: 0 done; 1 a step failed (finalize is then skipped); 2 usage or an existing
# output root; 3 a setup check failed; 4 (full mode) every step ran but compare_regen.py
# found differences from the reference tree. A smoke run's comparison always shows
# differences (reduced designs), so it does not change the exit status.
#
# What runs (all under $SREMU_JOB_OUT):
#   1. E-F API latency       ef_latency/            first, on an otherwise idle host; MEAS of
#                                                   this host only; finalize does not collect
#                                                   it (a transport latency fix is pending)
#   2. E-A prebuild          _ea_cache/             geometry and decision tables
#      E-F BER (12 points)   ef_ber/<mod>_<EbN0>dB/ in parallel with the prebuild
#   3. VA-1 IQ subset        ea_iq/                 4 batch references and 8 served IQ runs
#                                                   (Exact, Urap; tau 0.1 and 1 s; noise
#                                                   off/on; 1000 s), each its own process
#      E-C                   ec/                    beside them
#      E-A batch sweep       ea_ref/, ea_tau/       experiments.run_sweeps ea, which also
#                                                   writes paper_results/E-A/
#      E-B lab-like          eb1/, eb2_lab/, eb2_ideal1/  (paper specs, default scenario)
#   4. finalize              paper_results/         experiments.finalize --out
#   5. compare               compare_report.json    experiments/compare_regen.py, only when a
#                                                   reference tree is available
#   Provenance and logs:     _regen/ (provenance.json, steps.tsv, logs/, specs/)
#
# Not covered (the paper quotes them, but this script does not re-run them):
#   V-TANG, V-CFO, V-FM  validation/<study>/results/20261010T23*: sremu 9dca79f, DIRTY tree.
#                        Each takes under 4 min on the laptop. Their manifests read git
#                        directly (they ignore SREMU_GIT_COMMIT), so in a job bundle without
#                        .git they would record no commit; re-run them from a clean checkout.
#   E-B-arc              paper/sweeps/results/E-B-arc: AWS run at sremu 5be649a, clean.
#   V-LAB                validation/ants_fig4b/results/20261011T003044Z_*: 7a63287, clean.
#
# Sizing (c7i.48xlarge, 192 vCPU): an estimate, not a measurement on this instance type, of
# 30-50 min of job time, plus about 5 min of instance setup by the runner. The critical path
# is the VA-1 batch reference for tau = 0.1 s (one process): 1314 s on an idle c7i.8xlarge
# with the code behind the paper rows, 2014 s on the laptop with the working tree of
# 309c633 (another job shared the laptop for part of that time). E-C (one process) took
# 855 s on the laptop beside the sweeps, the 1000 s IQ runs about 600 s each. The sweeps
# finish first: on the laptop, 12 workers ran all 372 E-A points in 1172 s and all 297 E-B
# points in 119 s; on 176 workers that is about 10 min at most. A smaller instance (e.g.
# 64 vCPU) would therefore take about as long. Memory: of four E-A points
# measured on the laptop, the largest peak working set was 0.77 GiB (2.4 GHz, bandit,
# tau 0.01 s), so 176 sweep workers should stay well inside the instance's 384 GiB.
#
# Thread settings: sweep workers run with one numba and one BLAS thread each; VA-1 and E-C
# processes with 4 numba threads. Engine output does not depend on the numba thread count
# (DESIGN I2a); compare_regen.py checks the values against the paper tree.
# SVML: this script always sets NUMBA_DISABLE_INTEL_SVML=1 (DESIGN D15). The earlier runs
# differ: the local runs behind E-A, E-B, E-C and E-F BER recorded numba_disable_svml false,
# the AWS VA-1 rows true. On the laptop with SVML disabled (working tree of 309c633), every
# E-A, E-B, E-C and E-F BER value and 5 of the 12 VA-1 pieces (batch references exact
# tau 0.1 and 1 s; IQ runs exact tau 1 s noise off and on, urap tau 1 s noise off) were
# bit-identical to the paper tree. The Linux run checks everything again on one host.
set -uo pipefail

usage() { awk 'NR > 1 && /^set -uo pipefail/ { exit } NR > 1 { print }' "$0"; }

JOBS=""
DEADLINE_MIN=0
MODE=full
PLAN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --deadline-min) DEADLINE_MIN="${2:?--deadline-min needs a value}"; shift 2 ;;
    --deadline-min=*) DEADLINE_MIN="${1#*=}"; shift ;;
    --smoke) MODE=smoke; shift ;;
    --plan) PLAN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

to_native() {  # Git Bash on Windows: C:/... paths for the Windows Python; unchanged elsewhere
  if command -v cygpath > /dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT" || exit 2
PY="${SREMU_PYTHON:-python}"
OUT_RAW="${SREMU_JOB_OUT:?SREMU_JOB_OUT must be set (output root)}"
mkdir -p "$OUT_RAW" || exit 2
OUT="$(to_native "$(cd "$OUT_RAW" && pwd)")"
REG="$OUT/_regen"
LOGS="$REG/logs"
STEPS="$REG/steps.tsv"
JOBS="${JOBS:-$(nproc)}"
case "$JOBS" in ''|*[!0-9]*) echo "--jobs must be a positive integer" >&2; exit 2 ;; esac
case "$DEADLINE_MIN" in ''|*[!0-9]*) echo "--deadline-min must be an integer" >&2; exit 2 ;; esac
[ "$JOBS" -ge 1 ] || { echo "--jobs must be >= 1" >&2; exit 2; }
if [ "$JOBS" -gt 32 ]; then RESERVE=16; else RESERVE=$(( JOBS / 4 )); fi
SWEEP_JOBS=$(( JOBS - RESERVE ))
[ "$SWEEP_JOBS" -ge 1 ] || SWEEP_JOBS=1
T_START=$(date +%s)
DEADLINE_EPOCH=$(( T_START + DEADLINE_MIN * 60 ))

log() { printf '[regen %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

for d in ea_ref ea_tau ea_iq eb1 eb2_lab eb2_ideal1 ec ef_ber ef_latency _ea_cache \
         paper_results _regen; do
  if [ -e "$OUT/$d" ]; then
    echo "refusing: $OUT/$d exists (use a fresh SREMU_JOB_OUT)" >&2
    exit 2
  fi
done
mkdir -p "$LOGS"
printf 'step\tstart_epoch\tend_epoch\twall_s\texit\n' > "$STEPS"

export PYTHONUNBUFFERED=1
export NUMBA_DISABLE_INTEL_SVML=1  # fixed for every process (a caller's value is overridden)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SWEEP_ENV=(env NUMBA_NUM_THREADS=1 SREMU_EB_NUMBA_THREADS=1)
SINGLE_ENV=(env NUMBA_NUM_THREADS=4)
if [ "$MODE" = smoke ]; then
  export SREMU_EAIQ_DURATION_S=4  # VA-1 smoke runs (experiments/ea_iq.py: smoke tests only)
else
  unset SREMU_EAIQ_DURATION_S  # a value left in the caller's shell would shorten the VA-1 runs
fi

# ----------------------------------------------------------------------------- steps
FAILED=()
NO_DEADLINE=0  # 1 for the short closing steps (checks, finalize, compare)
BG_NAMES=()
BG_PIDS=()

run_step() {  # run_step NAME CMD... : log to _regen/logs/NAME.log, record in steps.tsv
  local name="$1" logf="$LOGS/$1.log" t0 t1 rc lim
  shift
  t0=$(date +%s)
  log "start $name"
  if [ "$DEADLINE_MIN" -gt 0 ] && [ "$NO_DEADLINE" != 1 ]; then
    lim=$(( DEADLINE_EPOCH - t0 ))
    if [ "$lim" -le 0 ]; then
      echo "deadline reached before the step started" > "$logf"
      rc=124
    else
      timeout --kill-after=30 "${lim}s" "$@" > "$logf" 2>&1
      rc=$?
    fi
  else
    "$@" > "$logf" 2>&1
    rc=$?
  fi
  t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$t0" "$t1" "$(( t1 - t0 ))" "$rc" >> "$STEPS"
  if [ "$rc" -eq 0 ]; then
    log "done  $name ($(( t1 - t0 )) s)"
  else
    log "FAIL  $name exit $rc ($(( t1 - t0 )) s); last lines of $logf:"
    tail -n 15 "$logf" | sed 's/^/    /'
  fi
  return "$rc"
}

fg_step() { run_step "$@" || FAILED+=("$1"); }

bg_step() {
  run_step "$@" &
  BG_NAMES+=("$1")
  BG_PIDS+=("$!")
}

wait_bg() {
  local i
  for i in "${!BG_PIDS[@]}"; do
    wait "${BG_PIDS[$i]}" || FAILED+=("${BG_NAMES[$i]}")
  done
  BG_NAMES=()
  BG_PIDS=()
}

# ----------------------------------------------------------------------------- sweep specs
# Copies of paper/sweeps/*.yaml (paper repo d7e67e7) for job bundles without the paper
# repo. setup_regen checks every copy and the chosen source against the pinned hashes.
EMB="$REG/specs_embedded"
mkdir -p "$EMB"
cat > "$EMB/ea_ref.yaml" <<'YAML'
# E-A reference points (DESIGN.md section 11, E-A; pre-registration paper/preregistration/E-A.md).
# Per geometry: RisAbsent (baseline), the instantaneous exact optimum P(0) (tau_SUI -> 0, no
# latency) and the stale reference P_stale (exact optimum for an independent position).
# Runner: sremu/experiments/ea.py:run_point (batch path, vectorized; 20 batches x 333 333
# steps of 5 ms = 33 333.3 s per point). Run with
#   python -m experiments.run_sweeps ea --jobs 12      (from the sremu repo root)
name: ea_ref
description: "E-A reference points: RisAbsent, instantaneous optimum, stale optimum per geometry"
runner: "experiments.ea:run_point"
base:
  n_batches: 20
  steps_per_batch: 333333
zip:
  regime:  [A0, A0, A0, A0, A1, A1, A1, A1, A1s, A1s, A1s, A1s]
  fc_hz:   [1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 2.4e9, 2.4e9, 1.0e9, 1.0e9, 2.4e9, 2.4e9]
  h_m:     [1, 2, 4, 6, 2, 4, 2, 4, 2, 2, 2, 2]
  loss_db: [0, 0, 0, 0, 20, 20, 20, 20, 10, 30, 10, 30]
grid:
  controller: [absent, instant, stale]
seeds: [20261010]
YAML
cat > "$EMB/ea_tau.yaml" <<'YAML'
# E-A signaling-delay sweep (DESIGN.md section 11, E-A; pre-registration
# paper/preregistration/E-A.md). Controllers: Urap(0.2) (period tau_SUI), WclHeuristic(4,3)
# and ExactOptimum in DelayedController(tau_SUI, report_delay 0) with a 1 cm GridChannel,
# ProbeBandit (16 exact optima, dwell max(20 ms, tau_SUI)); command latency 1 ms.
# Runner: sremu/experiments/ea.py:run_point (batch path, vectorized; 20 batches x 333 333
# steps of 5 ms = 33 333.3 s per point). Run with
#   python -m experiments.run_sweeps ea --jobs 12      (from the sremu repo root)
name: ea_tau
description: "E-A controllers vs tau_SUI per geometry (A0 reproduction, A1 blocked corridor)"
runner: "experiments.ea:run_point"
base:
  n_batches: 20
  steps_per_batch: 333333
zip:
  regime:  [A0, A0, A0, A0, A1, A1, A1, A1, A1s, A1s, A1s, A1s]
  fc_hz:   [1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 1.0e9, 2.4e9, 2.4e9, 1.0e9, 1.0e9, 2.4e9, 2.4e9]
  h_m:     [1, 2, 4, 6, 2, 4, 2, 4, 2, 2, 2, 2]
  loss_db: [0, 0, 0, 0, 20, 20, 20, 20, 10, 30, 10, 30]
grid:
  controller: [exact, wcl, urap, bandit]
  tau_sui_s: ["0.01", "0.03", "0.1", "0.3", "1", "3", "10"]
seeds: [20261010]
YAML
cat > "$EMB/eb1.yaml" <<'YAML'
# E-B1 mechanism (DESIGN.md section 11, E-B; VA-5). One QPSK packet (64 + 512 symbols,
# 4 samples/symbol, 1 MS/s) to RX1 of scenarios/lab_like_A.yaml (lab-like SIM geometry);
# the RIS switches from RX1's to RX2's exact optimum at delta * T_p after the packet start.
# Runner: sremu/experiments/eb.py:run_eb1_point. Run with
#   python -m experiments.run_sweeps eb1 --jobs 12      (from the sremu repo root)
name: eb1
description: "E-B1 intra-frame switch: per-symbol EVM, per-packet EVM/BER, VA-5 sample exactness"
runner: "experiments.eb:run_eb1_point"
grid:
  alphabet: [lab, ideal1]
  los: [present, blocked]
  transition: [step, linear20us, linear200us]
  delta: ["-0.5", "0.05", "0.25", "0.5", "0.75", "1.05"]
seeds: [20261010]
YAML
cat > "$EMB/eb2_ideal1.yaml" <<'YAML'
# E-B2 timing requirement (DESIGN.md section 11, E-B), ideal 1-bit alphabet, LOS present.
# Same protocol as eb2_lab.yaml. Runner: sremu/experiments/eb.py:run_eb2_point.
name: eb2_ideal1
description: "E-B2 PER and EVM vs latency jitter, ideal 1-bit alphabet, LOS present"
runner: "experiments.eb:run_eb2_point"
base:
  alphabet: ideal1
  los: present
  n_packets: 2000
zip:
  semantics: [timed, timed, timed, asap, legacy]
  lead_s:    ["5e-4", "2e-3", "5e-3", "5e-4", "5e-4"]
grid:
  guard_s: ["1e-4", "5e-4", "1e-3"]
  jitter_s: ["0", "1e-4", "3e-4", "1e-3", "3e-3"]
seeds: [20261010]
YAML
cat > "$EMB/eb2_lab.yaml" <<'YAML'
# E-B2 timing requirement (DESIGN.md section 11, E-B), lab alphabet (|Gamma| = 1 ASSUMED),
# LOS present and the LOS-blocked sensitivity variant. 2000 packets alternating RX1/RX2
# (TdmaAlternate decisions), guards G, command latency 0.2 ms + U[0, J] (keyed LatencyModel),
# semantics timed (lead 0.5 / 2 / 5 ms; late -> apply_on_arrival), asap, legacy
# (LegacyNextTickAdapter, 2 ms ticks). lead_s is ignored by asap and legacy.
# Runner: sremu/experiments/eb.py:run_eb2_point. Run with
#   python -m experiments.run_sweeps eb2 --jobs 12      (from the sremu repo root)
name: eb2_lab
description: "E-B2 PER and EVM vs latency jitter, lab alphabet, LOS present and blocked"
runner: "experiments.eb:run_eb2_point"
base:
  alphabet: lab
  n_packets: 2000
zip:
  los:       [present, present, present, present, present, blocked, blocked, blocked, blocked, blocked]
  semantics: [timed, timed, timed, asap, legacy, timed, timed, timed, asap, legacy]
  lead_s:    ["5e-4", "2e-3", "5e-3", "5e-4", "5e-4", "5e-4", "2e-3", "5e-3", "5e-4", "5e-4"]
grid:
  guard_s: ["1e-4", "5e-4", "1e-3"]
  jitter_s: ["0", "1e-4", "3e-4", "1e-3", "3e-3"]
seeds: [20261010]
YAML

SPEC_SRC=""
if [ "${SREMU_SPEC_DIR:-}" = embedded ]; then
  SPEC_SRC=""
elif [ -n "${SREMU_SPEC_DIR:-}" ]; then
  SPEC_SRC="$(to_native "$SREMU_SPEC_DIR")"
elif [ -f "$ROOT/../paper/sweeps/ea_ref.yaml" ]; then
  SPEC_SRC="$(to_native "$(cd "$ROOT/../paper/sweeps" && pwd)")"
fi
SCRIPT_SHA="$(sha256sum "$HERE/regenerate_paper_data.sh" | cut -d' ' -f1)"

# ----------------------------------------------------------------------------- setup
# Checks the specs, writes the smoke subsets, fixes the git state for every later process
# (SREMU_EXP_GIT_JSON) and writes _regen/provenance.json.
log "mode $MODE, jobs $JOBS (sweep pools $SWEEP_JOBS, reserved $RESERVE), out $OUT"
REGEN_OUT="$OUT" REGEN_MODE="$MODE" REGEN_SPEC_SRC="$SPEC_SRC" REGEN_JOBS="$JOBS" \
REGEN_SWEEP_JOBS="$SWEEP_JOBS" REGEN_SCRIPT_SHA="$SCRIPT_SHA" \
REGEN_DEADLINE_MIN="$DEADLINE_MIN" \
"$PY" - > "$LOGS/setup.log" 2>&1 <<'PYEOF'
import datetime as dt
import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
os.environ.pop("SREMU_EXP_GIT_JSON", None)  # the git state is decided here, not inherited
out = Path(os.environ["REGEN_OUT"])
reg = out / "_regen"
mode = os.environ["REGEN_MODE"]

# SweepSpec.sha256 (canonical content) of paper/sweeps/*.yaml, equal to sweep.json
# "spec_sha256" of the runs behind the paper results.
EXPECTED = {
    "ea_ref": "fdf696a7d2631038d7631bfee07c27f5487a29ce0ed637f0a968ce569bcd24e2",
    "ea_tau": "e90840cc1bb72263a1fa9ca9bb05b21c82568c2ce4bcb8e1f6d4e649cec3cedc",
    "eb1": "f78c111cf70660d5f66adb303be1fcfdf4c8a5dd20cbe4695c3ce34025933de7",
    "eb2_lab": "061468462f5462017e07dc17b9c1475594f900fe612b49a00c175e43a4703755",
    "eb2_ideal1": "c2c9c4e39d16aa569223a3516885a0cafae17b98fc1d6afbee56058565f83532",
}
# smoke subsets: E-A geometry A1 (1 GHz, h 2 m, 20 dB), the one H-A1 and the VA-1/E-C
# corridor use; E-B small grids; E-B2 with 40 packets (so E-B2 cannot match the paper)
SMOKE = {
    "ea_ref": {"zip_idx": [4]},
    "ea_tau": {"zip_idx": [4]},
    "eb1": {"grid": {"alphabet": ["lab"], "los": ["present"], "transition": ["step", "linear20us"]}},
    "eb2_lab": {
        "zip_idx": [0, 1, 2, 3, 4],
        "grid": {"guard_s": ["5e-4"], "jitter_s": ["0", "1e-3"]},
        "base": {"n_packets": 40},
    },
    "eb2_ideal1": {
        "grid": {"guard_s": ["5e-4"], "jitter_s": ["1e-3"]},
        "base": {"n_packets": 40},
    },
}


def fail(msg: str) -> None:
    print("SETUP FAILED: " + msg, flush=True)
    sys.exit(3)


import hashlib  # noqa: E402

from experiments.common import code_hashes, git_info, write_json  # noqa: E402
from sremu.experiments.sweep import SweepSpec, load_spec  # noqa: E402
from sremu.record.manifest import environment_info  # noqa: E402
from sremu.scenario.load import read_document  # noqa: E402


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---- specs: the embedded copies are always checked; the chosen source must match too
src = os.environ.get("REGEN_SPEC_SRC") or ""
specs = reg / "specs"
specs.mkdir(parents=True, exist_ok=True)
spec_rec = {}
for name, exp in EXPECTED.items():
    emb = reg / "specs_embedded" / f"{name}.yaml"
    emb_sha = load_spec(emb).sha256
    if emb_sha != exp:
        fail(f"embedded copy of {name}.yaml has spec hash {emb_sha}, pinned {exp}")
    p = Path(src) / f"{name}.yaml" if src else emb
    if not p.is_file():
        fail(f"{p} not found")
    dst = specs / f"{name}.yaml"
    shutil.copyfile(p, dst)
    spec = load_spec(dst)
    if spec.sha256 != exp:
        fail(f"{p}: spec hash {spec.sha256} differs from the pinned {exp}")
    spec_rec[name] = {
        "source": str(p),
        "file_sha256": sha_file(dst),
        "spec_sha256": spec.sha256,
        "pinned_spec_sha256": exp,
        "n_points": len(spec.points()),
    }
spec_dir = specs
smoke_rec = {}
if mode == "smoke":
    spec_dir = reg / "specs_smoke"
    spec_dir.mkdir(parents=True, exist_ok=True)
    for name, s in SMOKE.items():
        d = dict(read_document(specs / f"{name}.yaml"))
        if "zip_idx" in s:
            d["zip"] = {k: [v[i] for i in s["zip_idx"]] for k, v in d["zip"].items()}
        for k, v in s.get("grid", {}).items():
            if not all(x in d["grid"][k] for x in v):
                fail(f"smoke grid {name}.{k} is not a subset")
            d["grid"][k] = v
        for k, v in s.get("base", {}).items():
            d["base"][k] = v
        d["description"] = f"SMOKE subset (not paper data) of {name}: {d.get('description', '')}"
        sp = SweepSpec.from_mapping(d)
        (spec_dir / f"{name}.yaml").write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
        smoke_rec[name] = {"spec_sha256": sp.sha256, "n_points": len(sp.points()), "subset": s}

# ---- git state, fixed once for every later process (experiments.common.git_info reads it)
gs_path = reg / "git_state.json"
commit = os.environ.get("SREMU_GIT_COMMIT", "").strip()
allow_dirty = os.environ.get("SREMU_REGEN_ALLOW_DIRTY", "") == "1"
if commit:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        fail(f"SREMU_GIT_COMMIT {commit!r} is not a full 40-hex commit id")
    flag = os.environ.get("SREMU_GIT_DIRTY", "").strip()
    if flag not in ("0", "1"):
        fail("SREMU_GIT_DIRTY must be 0 or 1 when SREMU_GIT_COMMIT is set")
    dirty = flag == "1"
    if dirty and mode == "full" and not allow_dirty:
        fail(
            "SREMU_GIT_DIRTY=1: the bundle is not the committed code (runner.py sets it for "
            "--extra paths or a dirty launch tree); launch from a clean tree without --extra"
        )
    info = {
        "commit": commit,
        "dirty": dirty,
        "dirty_patch_sha256": None,
        "dirty_patch_path": None,
        "reason": "from SREMU_GIT_COMMIT/SREMU_GIT_DIRTY (job bundle without .git)",
        "code": code_hashes(),
    }
    git_source = "env"
else:
    info = dict(git_info(reg))  # read-only git; a dirty tree's patch goes to _regen/
    if info.get("commit") is None:
        fail(f"no git state ({info.get('reason')}); set SREMU_GIT_COMMIT and SREMU_GIT_DIRTY")
    if info.get("dirty") and mode == "full" and not allow_dirty:
        fail("the work tree is dirty; regenerate from a clean checkout of one commit")
    git_source = "git"
write_json(gs_path, info)

host = dict(environment_info())
if host.get("aws_instance") is not None:
    host["aws_instance"] = host["aws_instance"].model_dump(mode="json")
host = json.loads(json.dumps(host, default=str))
env_keys = (
    "NUMBA_DISABLE_INTEL_SVML",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "SREMU_EAIQ_DURATION_S",
    "SREMU_BUNDLE_SHA256",
)
prov = {
    "tool": "experiments/regenerate_paper_data.sh",
    "script_sha256": os.environ["REGEN_SCRIPT_SHA"],
    "mode": mode,
    "paper_data": mode == "full",
    "started_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git": {
        "source": git_source,
        "commit": info["commit"],
        "dirty": info["dirty"],
        "dirty_patch_sha256": info.get("dirty_patch_sha256"),
        "allow_dirty": allow_dirty,
    },
    "code": info["code"],
    "specs": spec_rec,
    "smoke_specs": smoke_rec,
    "spec_dir_used": str(spec_dir),
    "jobs": int(os.environ["REGEN_JOBS"]),
    "sweep_jobs": int(os.environ["REGEN_SWEEP_JOBS"]),
    "deadline_min": int(os.environ["REGEN_DEADLINE_MIN"]),
    "thread_settings": {
        "sweep_workers": {"NUMBA_NUM_THREADS": 1, "SREMU_EB_NUMBA_THREADS": 1},
        "va1_and_ec_processes": {"NUMBA_NUM_THREADS": 4},
        "latency": "numba default",
    },
    "env": {k: os.environ.get(k) for k in env_keys},
    "host": host,
    "python": platform.python_version(),
}
write_json(reg / "provenance.json", prov)
print(json.dumps({"spec_dir": str(spec_dir), "git": prov["git"]}), flush=True)
print("SPEC_DIR=" + str(spec_dir), flush=True)
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  log "setup failed (exit $rc):"
  sed 's/^/    /' "$LOGS/setup.log"
  exit 3
fi
SPEC_DIR="$(sed -n 's/^SPEC_DIR=//p' "$LOGS/setup.log" | tail -n 1)"
export SREMU_EXP_GIT_JSON="$REG/git_state.json"
log "specs verified ($SPEC_DIR); git: $(grep -m 1 '^{' "$LOGS/setup.log")"
if command -v lscpu > /dev/null 2>&1; then lscpu > "$REG/lscpu.txt" 2>&1; fi
"$PY" -m pip freeze > "$REG/pip_freeze.txt" 2>&1 || true
if [ "$PLAN" = 1 ]; then
  cat <<PLAN
plan ($MODE; checks passed; provenance in $REG/provenance.json):
  1  ef_latency     experiments.ef.run_latency -> $OUT/ef_latency (alone; numba default threads)
  2  ef_ber_*       experiments.ef.run_ber_point x $( [ "$MODE" = smoke ] && echo 6 || echo 12 ) -> $OUT/ef_ber (parallel)
     ea_prebuild    experiments.run_sweeps._prebuild_ea -> $OUT/_ea_cache ($JOBS processes max)
  3  va1_*          experiments.ea_iq.batch_reference x 4, run_point x 8 -> $OUT/ea_iq (background)
     ec             experiments.ec.run -> $OUT/ec (background)
     ea_sweeps      experiments.run_sweeps ea --jobs $SWEEP_JOBS --spec-dir $SPEC_DIR --results $OUT/paper_results
     eb1_sweep      experiments.run_sweeps eb1 --jobs $SWEEP_JOBS --no-aggregate
     eb2_sweeps     experiments.run_sweeps eb2 --jobs $SWEEP_JOBS --no-aggregate
  4  check_runs, finalize --runs $OUT --out $OUT/paper_results$( [ "$MODE" = smoke ] && echo " --smoke" )
  5  compare (when a reference tree exists), provenance
PLAN
  exit 0
fi

# ----------------------------------------------------------------------------- phase 1
# E-F API latency, alone on the host (numba default threads)
fg_step ef_latency env -u NUMBA_NUM_THREADS -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS \
  -u MKL_NUM_THREADS "$PY" -c \
  "import sys; from experiments.ef import run_latency; run_latency(sys.argv[1])" \
  "$OUT/ef_latency"

# ----------------------------------------------------------------------------- phase 2
# E-F BER points in parallel with the E-A prebuild
if [ "$MODE" = smoke ]; then BER_DB="0 4 8"; else BER_DB="0 2 4 6 8 10"; fi
for mod in bpsk qpsk; do
  for e in $BER_DB; do
    dir="$OUT/ef_ber/$(printf '%s_%02ddB' "$mod" "$e")"
    bg_step "ef_ber_${mod}_${e}dB" "${SWEEP_ENV[@]}" "$PY" -c \
      "import json, sys; from experiments.ef import run_ber_point; \
r = run_ber_point(sys.argv[1], float(sys.argv[2]), sys.argv[3]); \
print(json.dumps({k: r[k] for k in ('modulation', 'ebn0_db', 'bits', 'genie_errors', \
'ls64_errors', 'genie_theory_in_ci95')}))" "$mod" "$e" "$dir"
  done
done
fg_step ea_prebuild "${SWEEP_ENV[@]}" "$PY" -c \
  "import sys; from pathlib import Path; from experiments.run_sweeps import SPECS, _prebuild_ea; \
_prebuild_ea(Path(sys.argv[1]), int(sys.argv[2]), [Path(sys.argv[3]) / n for n in SPECS['ea']])" \
  "$OUT" "$JOBS" "$SPEC_DIR"
wait_bg

# ----------------------------------------------------------------------------- phase 3
# long single-process work in the background: VA-1 (4 batch references, 8 IQ runs), E-C
if [ -f "$OUT/_ea_cache/geom_f1.0G_h2_L20.npz" ]; then
  for kt in exact:0.1 exact:1 urap:0.1 urap:1; do
    k="${kt%%:*}"
    t="${kt##*:}"
    bg_step "va1_batch_${k}_tau${t}" "${SINGLE_ENV[@]}" "$PY" -c \
      "import json, sys; from pathlib import Path; from experiments import ea_iq; \
print(json.dumps(ea_iq.batch_reference(sys.argv[1], sys.argv[2], Path(sys.argv[3]), \
Path(sys.argv[4]))))" "$k" "$t" "$OUT/ea_iq" "$OUT/_ea_cache"
    for noise in off on; do
      bg_step "va1_iq_${k}_tau${t}_noise${noise}" "${SINGLE_ENV[@]}" "$PY" -c \
        "import json, sys; from pathlib import Path; from experiments import ea_iq; \
r = ea_iq.run_point(sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])); \
print(json.dumps({k: r[k] for k in ('kind', 'tau_sui_s', 'noise', 'duration_s', 'p_iq_dbm', \
'p_iq_ci_dbm', 'wall_s')}))" "$k" "$t" "$noise" "$OUT/ea_iq" "$OUT/_ea_cache"
      sleep 1  # stagger the process launches (each IQ run starts three processes)
    done
  done
  if [ "$MODE" = smoke ]; then NMAPS=4; else NMAPS=200; fi
  bg_step ec "${SINGLE_ENV[@]}" "$PY" -c \
    "import sys; from experiments.ec import run; run(sys.argv[1], sys.argv[2], n_maps=int(sys.argv[3]))" \
    "$OUT/ec" "$OUT/_ea_cache" "$NMAPS"
else
  log "E-A cache missing: VA-1 and E-C not started"
  FAILED+=(va1 ec)
fi

# sweeps in the foreground, one after the other (each run_sweeps writes git_state.json once)
fg_step ea_sweeps "${SWEEP_ENV[@]}" "$PY" -m experiments.run_sweeps ea --jobs "$SWEEP_JOBS" \
  --out "$OUT" --spec-dir "$SPEC_DIR" --results "$OUT/paper_results"
fg_step eb1_sweep "${SWEEP_ENV[@]}" "$PY" -m experiments.run_sweeps eb1 --jobs "$SWEEP_JOBS" \
  --out "$OUT" --spec-dir "$SPEC_DIR" --results "$OUT/paper_results" --no-aggregate
fg_step eb2_sweeps "${SWEEP_ENV[@]}" "$PY" -m experiments.run_sweeps eb2 --jobs "$SWEEP_JOBS" \
  --out "$OUT" --spec-dir "$SPEC_DIR" --results "$OUT/paper_results" --no-aggregate
log "sweeps done; waiting for VA-1 and E-C"
wait_bg

# ----------------------------------------------------------------------------- checks
# every sweep point ok, every VA-1 run present, one git state in every manifest
NO_DEADLINE=1
fg_step check_runs "$PY" - "$OUT" "$MODE" <<'PYEOF'
import json
import sys
from collections import Counter
from pathlib import Path

out, mode = Path(sys.argv[1]), sys.argv[2]
bad = []
counts = {}
for sw in ("ea_ref", "ea_tau", "eb1", "eb2_lab", "eb2_ideal1"):
    p = out / sw / "index.json"
    if not p.is_file():
        bad.append(f"{sw}: index.json missing")
        continue
    st = Counter(e["status"] for e in json.loads(p.read_text(encoding="utf-8")))
    counts[sw] = dict(st)
    if set(st) != {"ok"}:
        bad.append(f"{sw}: {dict(st)}")
for k, t in (("exact", "0.1"), ("exact", "1"), ("urap", "0.1"), ("urap", "1")):
    for d in [f"batch_{k}_tau{t}", f"{k}_tau{t}_noiseoff", f"{k}_tau{t}_noiseon"]:
        if not (out / "ea_iq" / d / "manifest.json").is_file():
            bad.append(f"ea_iq/{d}: no manifest")
n_ber = len(list(out.glob("ef_ber/*_*dB/manifest.json")))
want_ber = 6 if mode == "smoke" else 12
if n_ber != want_ber:
    bad.append(f"ef_ber: {n_ber} points, expected {want_ber}")
if not (out / "ec" / "table8.csv").is_file():
    bad.append("ec: table8.csv missing")
states = Counter()
for m in out.rglob("manifest.json"):
    if "_regen" in m.parts:
        continue
    d = json.loads(m.read_text(encoding="utf-8"))
    states[(d.get("git_commit"), d.get("git_dirty"), d.get("dirty_patch_sha256"))] += 1
rep = {
    "sweep_status": counts,
    "ber_points": n_ber,
    "manifest_git_states": [
        {"git_commit": c, "git_dirty": dd, "dirty_patch_sha256": p, "n": n}
        for (c, dd, p), n in states.items()
    ],
    "problems": bad,
}
gs = json.loads((out / "_regen" / "git_state.json").read_text(encoding="utf-8"))
if len(states) != 1:
    bad.append(f"{len(states)} distinct git states in the run manifests")
elif next(iter(states))[:2] != (gs.get("commit"), gs.get("dirty")):
    bad.append(f"manifest git state {next(iter(states))[:2]} differs from _regen/git_state.json")
(out / "_regen" / "check_runs.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
print(json.dumps(rep, indent=1))
sys.exit(1 if bad else 0)
PYEOF

# ----------------------------------------------------------------------------- finalize
if [ "${#FAILED[@]}" -eq 0 ]; then
  if [ "$MODE" = smoke ]; then FIN_SMOKE=(--smoke); else FIN_SMOKE=(); fi
  fg_step finalize "$PY" -m experiments.finalize --runs "$OUT" --out "$OUT/paper_results" \
    "${FIN_SMOKE[@]}"
else
  log "finalize skipped: failed steps: ${FAILED[*]}"
fi

# ----------------------------------------------------------------------------- compare
REF=""
if [ -n "${SREMU_REGEN_REF:-}" ]; then
  REF="$(to_native "$SREMU_REGEN_REF")"
elif [ -d "$ROOT/../paper/sweeps/results" ]; then
  REF="$(to_native "$(cd "$ROOT/../paper/sweeps/results" && pwd)")"
fi
COMPARE_RC=""
if [ -n "$REF" ] && [ -f "$OUT/paper_results/summary.json" ]; then
  if [ "$MODE" = smoke ]; then CMP_SMOKE=(--smoke); else CMP_SMOKE=(); fi
  run_step compare "$PY" experiments/compare_regen.py "$OUT/paper_results" --ref "$REF" \
    --json "$OUT/compare_report.json" "${CMP_SMOKE[@]}"
  COMPARE_RC=$?
  tail -n 8 "$LOGS/compare.log" | sed 's/^/    /'
  if [ "$COMPARE_RC" -eq 0 ]; then log "compare: PASS"; else
    log "compare: FAIL or mismatches (exit $COMPARE_RC); see $OUT/compare_report.json"; fi
else
  log "comparison skipped (no reference tree or no finalized results); run it later with:"
  log "  python experiments/compare_regen.py <out>/paper_results --ref <paper>/sweeps/results"
fi

# ----------------------------------------------------------------------------- provenance
T_END=$(date +%s)
"$PY" - "$OUT" "$T_START" "$T_END" "${COMPARE_RC:-none}" "${FAILED[*]:-}" \
  > "$LOGS/provenance_finish.log" 2>&1 <<'PYEOF'
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
reg = out / "_regen"
t0, t1, cmp_rc, failed = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split()
prov = json.loads((reg / "provenance.json").read_text(encoding="utf-8"))
with (reg / "steps.tsv").open(encoding="utf-8", newline="") as fh:
    steps = list(csv.DictReader(fh, delimiter="\t"))
cache = {}
for p in sorted((out / "_ea_cache").glob("*")):
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        cache[p.name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
check = reg / "check_runs.json"
prov.update(
    {
        "ended_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_s": t1 - t0,
        "steps": steps,
        "failed_steps": failed,
        "finalized": (out / "paper_results" / "summary.json").is_file() and not failed,
        "compare_exit": None if cmp_rc == "none" else int(cmp_rc),
        "check_runs": json.loads(check.read_text(encoding="utf-8")) if check.is_file() else None,
        "ea_cache_files": cache,
    }
)
(reg / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: prov[k] for k in ("wall_s", "failed_steps", "finalized", "compare_exit")}))
PYEOF
cat "$LOGS/provenance_finish.log"

if [ "${#FAILED[@]}" -gt 0 ]; then
  log "FAILED steps: ${FAILED[*]} (total $(( T_END - T_START )) s)"
  exit 1
fi
log "done in $(( T_END - T_START )) s; results in $OUT/paper_results (mode $MODE)"
if [ "$MODE" = full ] && [ -n "$COMPARE_RC" ] && [ "$COMPARE_RC" -ne 0 ]; then
  log "the regenerated results differ from the reference tree (see compare_report.json)"
  exit 4
fi
exit 0
