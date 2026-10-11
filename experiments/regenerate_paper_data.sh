#!/usr/bin/env bash
# Regenerate every simulated number of the paper at ONE committed sremu revision (pinned
# provenance).
#
# The paper's E-A, E-B (lab-like and lab arc), E-C, E-F BER, V-TANG, V-CFO and V-FM results
# were produced on several commits, partly from dirty trees. This script re-runs all of them
# from one tree, records one git state for every run directory, collects the WP11 results and
# the lab-arc E-B run with experiments/finalize.py and leaves the validation runs in their
# own directory. On Linux it also runs the E-E GNU Radio demonstration (installing GNU Radio
# from the Ubuntu archive when it is missing) and the paper junit (tools/tasks.py paper-junit,
# with the GNU Radio tests and the VA-1 record check on the regenerated VA-1 result). It never
# writes to the paper or sremu repository (experiments/install_regen.py does that, after
# compare_regen.py passed).
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
#                     runs, 4 E-C failure maps, 6 BER points, the E-B arc smoke (one E-B1
#                     point, one 50-packet E-B2 point), V-CFO --fast (4 trials per point);
#                     V-TANG and V-FM run in full (under a minute each).
#   --collect-latency WAITING for the transport latency fix: collect the E-F API latency into
#                     paper_results/E-F/latency (finalize --with-latency, MEAS of this host
#                     only) when the latency step succeeded. Without it the latency is
#                     measured into ef_latency/ but not collected. Either way a failed latency
#                     step is reported (exit 1) but does not stop finalize.
#   --no-ee           skip E-E (GNU Radio through the raw REQ/REP mode). E-E runs only on Linux
#                     (elsewhere it is skipped with a note): 2 independent repetitions of the
#                     11 runs of experiments/ee_gnuradio.py (10 static RIS configurations of
#                     0.3 s and the switching run), then, after every other experiment, the
#                     throughput runs at 1, 1.92, 3.84 and 5.76 MS/s (2 s each) on the
#                     otherwise idle host, exported to paper_results/E-E (the layout of
#                     paper/sweeps/results/E-E).
#   --no-junit        skip the paper junit (junit/sremu_paper.xml + .meta.json:
#                     tools/tasks.py paper-junit --git-state _regen/git_state.json, the CI and
#                     nightly markers, SREMU_GR_PYTHON set, SREMU_VA1_RECORD = the regenerated
#                     paper_results/E-A-IQ). In --smoke it runs a subset (the VA-1 record, GNU
#                     Radio and regeneration-tool tests) unless SREMU_REGEN_JUNIT_FULL=1.
#                     The paper reads every generated/junit/*.xml as one record, so this Linux
#                     junit REPLACES the Windows one there (install_regen.py --with-junit);
#                     adding it beside would double Table IV's counts and leave the GNU Radio
#                     tests "skipped" (skipped in any file = not green). A final build needs the
#                     junit of the sremu.lock commit: if the lock moves to the results-only
#                     commit after install, re-run this junit there on Linux.
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
#                     code then reads its state (read-only git). Either way the state is
#                     fixed once in _regen/git_state.json, and every process (experiments
#                     and validation studies) records it through SREMU_EXP_GIT_JSON.
#   SREMU_SPEC_DIR    directory with the paper sweep specs, or "embedded" (default:
#                     ../paper/sweeps when it exists, else the copies embedded below). Every
#                     spec is checked against its pinned content hash (SweepSpec.sha256 of
#                     the paper runs); so are the E-B arc sweeps (experiments.eb.sweep_specs)
#                     and scenarios/studies/lab_arc.yaml.
#   SREMU_REGEN_REF   reference results tree for the final comparison (default:
#                     ../paper/sweeps/results when it exists; the comparison is skipped
#                     otherwise, e.g. on AWS).
#   SREMU_REGEN_ALLOW_DIRTY=1
#                     full run on a dirty tree. The run is then NOT pinned; avoid.
#   SREMU_REGEN_LATENCY_CAP_S
#                     time limit of the E-F latency step in s (default 900).
#   SREMU_GR_PYTHON   the interpreter with GNU Radio (default /usr/bin/python3; Ubuntu 24.04's
#                     gnuradio 3.10.9.2 with the system numpy 1.26). sremu never runs in it and
#                     GNU Radio never runs in SREMU_PYTHON (numpy >= 2.2); they meet over
#                     loopback ZMQ only.
#   SREMU_REGEN_EE_THREADS
#                     NUMBA_NUM_THREADS of the served engine in the E-E runs (default 16, the
#                     thread count of the laptop E-E run; the output does not depend on it).
#   SREMU_REGEN_JUNIT_FULL=1
#                     the full paper junit in --smoke too.
#   SREMU_REGEN_JUNIT_CAP_S
#                     time limit of the junit step in s (default 2400).
#
# Exit status: 0 done; 1 a step failed (finalize is skipped when a WP11 step failed; a failed
# E-B arc run, validation study, V-LAB check or latency run is reported but does not stop
# finalize, which then runs without --eb-arc / --with-latency; a failed GNU Radio install,
# E-E step, E-E check or junit check is reported the same way); 2 usage or an existing
# output root; 3 a setup check failed; 4 (full mode) every step ran but compare_regen.py
# found differences from the reference tree. A
# smoke run's comparison always shows differences (reduced designs), so it does not change
# the exit status.
#
# What runs (all under $SREMU_JOB_OUT):
#   1. V-LAB check           _regen/vlab_check.json V-LAB is NOT re-run (the locked,
#                                                   pre-registered comparison of 7a63287): its
#                                                   protocol digest must equal
#                                                   PREREGISTERED_SHA256 and the run's record,
#                                                   and validation/ants_fig4b/ must be
#                                                   byte-identical to its files at 7a63287
#      E-F API latency       ef_latency/            on an otherwise idle host; MEAS of this
#                                                   host only; capped at 15 min; collected only
#                                                   with --collect-latency and only when it
#                                                   succeeded (fix pending)
#   2. E-A prebuild          _ea_cache/             geometry and decision tables
#      E-F BER (12 points)   ef_ber/<mod>_<EbN0>dB/ in parallel with the prebuild
#   3. VA-1 IQ subset        ea_iq/                 4 batch references and 8 served IQ runs
#                                                   (Exact, Urap; tau 0.1 and 1 s; noise
#                                                   off/on; 1000 s), each its own process
#      E-C                   ec/                    beside them
#      E-A batch sweep       ea_ref/, ea_tau/       experiments.run_sweeps ea, which also
#                                                   writes paper_results/E-A/
#      E-B lab-like          eb1/, eb2_lab/, eb2_ideal1/  (paper specs, default scenario)
#      E-B lab arc           eb_arc/                python -m experiments.eb --scenario
#                                                   scenarios/studies/lab_arc.yaml (447 points)
#      V-TANG, V-FM, V-CFO   validation/<study>/<UTC>_<name>/   validation/<study>/v*.py
#                                                   --root (after the sweeps, beside VA-1)
#      GNU Radio (Linux)     _regen/gnuradio.json   experiments.regen_checks gnuradio --install
#                                                   auto: apt-get install gnuradio when missing
#                                                   (background, from the end of step 1)
#      E-E repetitions       ee/rep1, ee/rep2       experiments.ee_gnuradio run --reps 2 (after
#                                                   the sweeps, beside VA-1 and E-C; lockstep, so
#                                                   load changes no value or hash)
#   4. finalize              paper_results/         experiments.finalize --out ... --eb-arc
#                                                   eb_arc [--with-latency]
#      E-E throughput        ee/throughput/         ee_gnuradio run --reps 0 --throughput, alone on
#                                                   the otherwise idle host (MEAS of this host)
#      E-E export + check    paper_results/E-E/, _regen/ee_check.json (repetitions bitwise
#                                                   identical, git state = the job's, D24)
#      paper junit           junit/sremu_paper.xml  + .meta.json (commit from git_state.json, host);
#                                                   _regen/junit_check.json (failures, skipped spec
#                                                   ids, GNU Radio and VA-1 tests, claim evidence)
#   5. compare               compare_report.json    experiments/compare_regen.py (WP11, E-B
#                                                   arc, validation runs, E-E, junit record, code
#                                                   check), only when a reference tree is available
#   Provenance and logs:     _regen/ (provenance.json, steps.tsv, logs/, specs/)
#
# Not re-run here (research/analysis/regen_coverage.md has the reasons and the procedure):
#   V-LAB     validation/ants_fig4b/results/20261011T003044Z_vlab_unblinded, 7a63287, clean:
#             the locked pre-registered comparison; re-running it with changed code would
#             break the pre-registration. The paper cites it at 7a63287; step 1 checks that
#             the pinned tree still carries its protocol and analysis code unchanged.
#   V-WCL     no paper number or figure reads it.
#   bench/    B1-B4 and B6 (paper/sweeps/results/bench): MEAS on dedicated hosts
#             (bench/aws/job.sh; MATLAB on the laptop for B6); running them beside this job
#             would spoil the timing.
#
# Sizing (c7i.48xlarge, 192 vCPU; job time, plus about 5 min of instance setup by the
# runner): an estimate from recorded run times, not a measurement on this instance type.
# Critical path: the VA-1 batch reference for tau = 0.1 s (one process) beside everything
# else: 1317 s and 1298 s (exact, urap) on an idle c7i.8xlarge (AWS run
# 20261010T232822Z_ea_iq_ef_18ffd1, the same Xeon 8488C), so about 25-35 min on the loaded
# instance; the 1000 s IQ runs took 588-602 s each there, E-C 855-1035 s on the laptop.
# The foreground chain is shorter: E-A (372 points; 16 349 CPU-s summed over the laptop
# points, longest point 232 s) about 5 min on 176 workers; E-B lab-like (297 points) 119 s
# on 12 laptop workers; E-B arc (447 points: 30 640 CPU-s on the Xeon at 5be649a, before the
# O(log n) command timeline of 309c633; with the current engine one laptop worker took 1.1 s
# for an E-B1 point and 7.8-10.8 s for a 2000-packet E-B2 point, about 3 600 CPU-s in all)
# 2-4 min, bounded by the five sweeps run one after the other; V-CFO 181 s on the laptop
# with 2 workers (V-TANG 13 s, V-FM 24 s), beside VA-1. Linux parts: apt-get install gnuradio
# about 3-6 min in the background (not measured on AWS; it waits up to 15 min for the dpkg
# lock); E-E repetitions: 93 s for both in the WSL2 smoke of this script (Ryzen 9 8945H, full
# E-E design, beside the smoke VA-1 and E-C), 7.5 min in the paper's laptop run (both
# repetitions + the four throughput runs, run walls 101.5 s + 56.3 s + 26.3 s), so about
# 2-8 min beside VA-1 on the Xeon and 1-2 min for the throughput runs at the end; the paper
# junit took 106-112 s in that WSL2 smoke (2125 tests, -n 8) and 82 s on Windows (2017 tests),
# 2-4 min expected here. The tail after VA-1 (checks, finalize, E-E throughput and export,
# junit) adds about 5-8 min. Expected job time 35-50 min: use --deadline-min 100 with runner
# --maxmin 130. Memory: of four E-A points measured on the
# laptop, the largest peak working set was 0.77 GiB (2.4 GHz, bandit, tau 0.01 s), so 176
# sweep workers stay well inside the instance's 384 GiB.
#
# Thread settings: sweep workers (E-A, E-B, E-B arc, V-CFO-4) run with one numba and one
# BLAS thread each; VA-1 and E-C processes with 4 numba threads; V-TANG and V-FM with one.
# Engine output does not depend on the numba thread count (DESIGN I2a); compare_regen.py
# checks the values against the paper tree.
# SVML: this script always sets NUMBA_DISABLE_INTEL_SVML=1 (DESIGN D15). The earlier runs
# differ: the local runs behind E-A, E-B, E-C and E-F BER recorded numba_disable_svml false,
# the AWS VA-1 rows and the E-B arc run true; the validation studies set it themselves. On
# the laptop with SVML disabled (working tree of 309c633), every E-A, E-B, E-C and E-F BER
# value and the 5 VA-1 pieces re-run there (batch references exact tau 0.1 and 1 s; IQ runs
# exact tau 1 s noise off and on, urap tau 1 s noise off) were bit-identical to the paper
# tree; the other 7 VA-1 pieces were not re-run locally. The Linux run checks everything
# again on one host. The 8 VA-1 IQ runs are served over the native ZMQ data plane (the
# transport whose intermittent stall is being fixed): launch the paper run only from a
# commit that contains that fix, with or without --collect-latency.
set -uo pipefail

usage() { awk 'NR > 1 && /^set -uo pipefail/ { exit } NR > 1 { print }' "$0"; }

JOBS=""
DEADLINE_MIN=0
MODE=full
PLAN=0
COLLECT_LATENCY=0
RUN_EE=1
RUN_JUNIT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    --jobs=*) JOBS="${1#*=}"; shift ;;
    --deadline-min) DEADLINE_MIN="${2:?--deadline-min needs a value}"; shift 2 ;;
    --deadline-min=*) DEADLINE_MIN="${1#*=}"; shift ;;
    --smoke) MODE=smoke; shift ;;
    --collect-latency) COLLECT_LATENCY=1; shift ;;
    --no-ee) RUN_EE=0; shift ;;
    --no-junit) RUN_JUNIT=0; shift ;;
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
VCFO_JOBS=$(( SWEEP_JOBS < 21 ? SWEEP_JOBS : 21 ))  # V-CFO-4 has 21 points
T_START=$(date +%s)
DEADLINE_EPOCH=$(( T_START + DEADLINE_MIN * 60 ))
# E-E and GNU Radio (Linux only; examples/gnuradio_zmq/README.md)
GR_PY="${SREMU_GR_PYTHON:-/usr/bin/python3}"
EE_THREADS="${SREMU_REGEN_EE_THREADS:-16}"
case "$EE_THREADS" in ''|*[!0-9]*) echo "SREMU_REGEN_EE_THREADS must be an integer" >&2; exit 2 ;; esac
if [ "$MODE" = smoke ]; then
  EE_RATES="1e6,5.76e6"; EE_TPUT_S="0.5"  # smoke: two short throughput runs
else
  EE_RATES="1e6,1.92e6,3.84e6,5.76e6"; EE_TPUT_S="2"  # the rates and length the paper quotes
fi
if [ "$(uname -s)" = Linux ]; then IS_LINUX=1; else IS_LINUX=0; fi
export HOME="${HOME:-/root}"  # cloud-init may leave HOME unset; GNU Radio keeps prefs there

log() { printf '[regen %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

for d in ea_ref ea_tau ea_iq eb1 eb2_lab eb2_ideal1 ec ef_ber ef_latency _ea_cache \
         eb_arc validation paper_results _regen ee junit; do
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
FAILED=()      # WP11 steps: any failure skips finalize
FAILED_AUX=()  # E-B arc, validation studies, V-LAB check, latency: reported only
NO_DEADLINE=0  # 1 for the short closing steps (checks, finalize, compare)
STEP_CAP_S=0   # > 0: a further time limit (s) for the steps started while it is set
BG_NAMES=()
BG_PIDS=()
BG_AUX=()

run_step() {  # run_step NAME CMD... : log to _regen/logs/NAME.log, record in steps.tsv
  local name="$1" logf="$LOGS/$1.log" t0 t1 rc lim=0
  shift
  t0=$(date +%s)
  log "start $name"
  if [ "$DEADLINE_MIN" -gt 0 ] && [ "$NO_DEADLINE" != 1 ]; then
    lim=$(( DEADLINE_EPOCH - t0 ))
    [ "$lim" -gt 0 ] || lim=-1
  fi
  if [ "$STEP_CAP_S" -gt 0 ] && { [ "$lim" -eq 0 ] || [ "$lim" -gt "$STEP_CAP_S" ]; }; then
    lim="$STEP_CAP_S"
  fi
  if [ "$lim" -lt 0 ]; then
    echo "deadline reached before the step started" > "$logf"
    rc=124
  elif [ "$lim" -gt 0 ]; then
    timeout --kill-after=30 "${lim}s" "$@" > "$logf" 2>&1
    rc=$?
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
fg_aux() { run_step "$@" || FAILED_AUX+=("$1"); }

bg_step() {
  run_step "$@" &
  BG_NAMES+=("$1")
  BG_PIDS+=("$!")
  BG_AUX+=(0)
}

bg_aux() {
  run_step "$@" &
  BG_NAMES+=("$1")
  BG_PIDS+=("$!")
  BG_AUX+=(1)
}

wait_bg() {
  local i
  for i in "${!BG_PIDS[@]}"; do
    if ! wait "${BG_PIDS[$i]}"; then
      if [ "${BG_AUX[$i]}" = 1 ]; then FAILED_AUX+=("${BG_NAMES[$i]}"); else
        FAILED+=("${BG_NAMES[$i]}"); fi
    fi
  done
  BG_NAMES=()
  BG_PIDS=()
  BG_AUX=()
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
REGEN_DEADLINE_MIN="$DEADLINE_MIN" REGEN_COLLECT_LATENCY="$COLLECT_LATENCY" \
REGEN_EE="$RUN_EE" REGEN_JUNIT="$RUN_JUNIT" REGEN_IS_LINUX="$IS_LINUX" REGEN_GR_PY="$GR_PY" \
REGEN_EE_THREADS="$EE_THREADS" REGEN_EE_RATES="$EE_RATES" REGEN_EE_TPUT_S="$EE_TPUT_S" \
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
# E-B on the lab arc: SweepSpec.sha256 of experiments.eb.sweep_specs(lab_arc) and SHA-256 of
# the scenario file, equal to paper/sweeps/results/E-B-arc/eb_plan.json and git_state.json
# (AWS run 20261011T060442Z_eb_arc_d41a75)
ARC_SCENARIO = "scenarios/studies/lab_arc.yaml"
ARC_SCENARIO_SHA256 = "f78c4c8615746d899eabd4b5f1cbdefbd82a2e70993f6643a28c23ea5beb060a"
EXPECTED_ARC = {
    "eb1": "c2bc8c151704f099995c2c609e803423b5262197118601c313b3f545d84953b5",
    "eb2_lab": "853504438d587444f457efcabce59152a2a1ec993accc03b0042f9cea9a5c54d",
    "eb2_ideal1": "035d142d37b59fb38c4150ab352864bfa6811857883bb12c25b12c032b0d15a0",
    "eb2_lab_es15": "c4447bae9b4e0255f8ee5f8d1acaa2f4de69c05a772e1824454daacd85c68140",
    "eb2_ideal1_es15": "baa6bd9576f0d5b389cb83eab0bbb3be88e8842acb8198f7502665c7f77eda18",
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

# ---- E-B on the lab arc: the code-defined design and the scenario must be the paper's
from experiments import eb  # noqa: E402
from experiments.compare_regen import validation_code_hashes  # noqa: E402

arc_sha = sha_file(REPO / ARC_SCENARIO)
if arc_sha != ARC_SCENARIO_SHA256:
    fail(f"{ARC_SCENARIO}: sha256 {arc_sha}, pinned {ARC_SCENARIO_SHA256}")
arc_rec = {}
for s in eb.sweep_specs(eb.scenario_ref(ARC_SCENARIO)):
    if s.sha256 != EXPECTED_ARC.get(s.name):
        fail(f"E-B arc sweep {s.name}: spec hash {s.sha256}, pinned {EXPECTED_ARC.get(s.name)}")
    arc_rec[s.name] = {"spec_sha256": s.sha256, "n_points": len(s.points())}
if set(arc_rec) != set(EXPECTED_ARC):
    fail(f"E-B arc sweeps {sorted(arc_rec)}, pinned {sorted(EXPECTED_ARC)}")

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
    "SREMU_GR_PYTHON",
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
    "validation_code": validation_code_hashes(REPO),
    "specs": spec_rec,
    "smoke_specs": smoke_rec,
    "arc": {"scenario": ARC_SCENARIO, "scenario_sha256": arc_sha, "specs": arc_rec},
    "spec_dir_used": str(spec_dir),
    "jobs": int(os.environ["REGEN_JOBS"]),
    "sweep_jobs": int(os.environ["REGEN_SWEEP_JOBS"]),
    "deadline_min": int(os.environ["REGEN_DEADLINE_MIN"]),
    "collect_latency": os.environ["REGEN_COLLECT_LATENCY"] == "1",
    "ee": {
        "enabled": os.environ["REGEN_EE"] == "1",
        "runs_here": os.environ["REGEN_EE"] == "1" and os.environ["REGEN_IS_LINUX"] == "1",
        "gr_python": os.environ["REGEN_GR_PY"],
        "repetitions": 2,
        "throughput_rates_sps": [float(x) for x in os.environ["REGEN_EE_RATES"].split(",")],
        "throughput_s": os.environ["REGEN_EE_TPUT_S"],
        "repetitions_phase": "after the sweeps, beside the VA-1 and E-C processes",
        "throughput_phase": "after every other experiment, alone on the host",
    },
    "junit": {
        "enabled": os.environ["REGEN_JUNIT"] == "1",
        "scope": "subset"
        if mode == "smoke" and os.environ.get("SREMU_REGEN_JUNIT_FULL") != "1"
        else "full",
    },
    "thread_settings": {
        "sweep_workers": {"NUMBA_NUM_THREADS": 1, "SREMU_EB_NUMBA_THREADS": 1},
        "eb_arc_workers": {"NUMBA_NUM_THREADS": 1, "threads_per_job": 1},
        "va1_and_ec_processes": {"NUMBA_NUM_THREADS": 4},
        "validation_studies": {"NUMBA_NUM_THREADS": 1},
        "ee_served_engine": {"NUMBA_NUM_THREADS": int(os.environ["REGEN_EE_THREADS"])},
        "latency": "numba default",
        "junit": "numba default, pytest -n 8 (tools/tasks.py)",
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
if [ "$RUN_JUNIT" = 1 ] && [ "$PLAN" != 1 ] && \
   ! "$PY" -c "import sigmf, jsonschema" > /dev/null 2>&1; then
  # the optional packages behind IO-05 (official SigMF validator) and IO-01 (jsonschema oracle):
  # without them those spec ids are skipped and gen_results.py refuses the junit. Versions of
  # the laptop venv that produced the paper junit; numpy and the other pins stay as they are.
  run_step junit_deps "$PY" -m pip install --disable-pip-version-check \
    "SigMF==1.13.0" "jsonschema==4.26.0" || FAILED_AUX+=(junit_deps)
fi
"$PY" -m pip freeze > "$REG/pip_freeze.txt" 2>&1 || true
if [ "$PLAN" = 1 ]; then
  cat <<PLAN
plan ($MODE; checks passed; provenance in $REG/provenance.json):
  1  vlab_check     experiments.vlab_check -> $REG/vlab_check.json (V-LAB is not re-run)
     ef_latency     experiments.ef.run_latency -> $OUT/ef_latency (alone; numba default threads; capped at ${SREMU_REGEN_LATENCY_CAP_S:-900} s; $( [ "$COLLECT_LATENCY" = 1 ] && echo "collected if it succeeds" || echo "not collected" ))
  2  ef_ber_*       experiments.ef.run_ber_point x $( [ "$MODE" = smoke ] && echo 6 || echo 12 ) -> $OUT/ef_ber (parallel)
     ea_prebuild    experiments.run_sweeps._prebuild_ea -> $OUT/_ea_cache ($JOBS processes max)
  3  va1_*          experiments.ea_iq.batch_reference x 4, run_point x 8 -> $OUT/ea_iq (background)
     ec             experiments.ec.run -> $OUT/ec (background)
     ea_sweeps      experiments.run_sweeps ea --jobs $SWEEP_JOBS --spec-dir $SPEC_DIR --results $OUT/paper_results
     eb1_sweep      experiments.run_sweeps eb1 --jobs $SWEEP_JOBS --no-aggregate
     eb2_sweeps     experiments.run_sweeps eb2 --jobs $SWEEP_JOBS --no-aggregate
     eb_arc         experiments.eb --scenario scenarios/studies/lab_arc.yaml --out $OUT/eb_arc --jobs $SWEEP_JOBS$( [ "$MODE" = smoke ] && echo " --smoke" )
     v_tang, v_fm   validation/{tang/vtang,fm/vfm}.py --root $OUT/validation/<study> (background)
     v_cfo          validation/cfo/vcfo.py --jobs $VCFO_JOBS$( [ "$MODE" = smoke ] && echo " --fast" ) --root $OUT/validation/cfo (background)
     gr_setup       experiments.regen_checks gnuradio --gr-python $GR_PY --install auto -> $REG/gnuradio.json (background from the end of step 1; $( [ "$IS_LINUX" = 1 ] && echo "Linux" || echo "not Linux: skipped" ))
     ee_reps        experiments.ee_gnuradio run --root $OUT/ee --reps 2 (background after eb_arc; NUMBA_NUM_THREADS=$EE_THREADS; $( [ "$RUN_EE" = 1 ] && [ "$IS_LINUX" = 1 ] && echo "enabled" || echo "skipped" ))
  4  check_runs, check_aux, finalize --runs $OUT --out $OUT/paper_results --eb-arc $OUT/eb_arc$( [ "$MODE" = smoke ] && echo " --smoke" )$( [ "$COLLECT_LATENCY" = 1 ] && echo " --with-latency (if ef_latency succeeded)" )
     ee_throughput  experiments.ee_gnuradio run --root $OUT/ee --reps 0 --throughput --rates $EE_RATES --throughput-s $EE_TPUT_S (alone on the host)
     ee_export      experiments.ee_gnuradio export --root $OUT/ee --dest $OUT/paper_results/E-E; ee_check -> $REG/ee_check.json
     junit          tools/tasks.py paper-junit $OUT/junit/sremu_paper.xml --git-state $REG/git_state.json ($( [ "$RUN_JUNIT" = 1 ] && echo "enabled" || echo "skipped" ); SREMU_GR_PYTHON=$GR_PY, SREMU_VA1_RECORD=$OUT/paper_results/E-A-IQ); junit_check -> $REG/junit_check.json
  5  compare (when a reference tree exists), provenance
PLAN
  exit 0
fi

# ----------------------------------------------------------------------------- phase 1
# V-LAB integrity (no re-run; reads files only), then the E-F API latency alone on the host
# (numba default threads). The latency run is capped at 15 min (a stalled transport must not
# hold the job); its failure is reported but never stops finalize of the WP11 results (with
# --collect-latency, finalize then runs without --with-latency).
fg_aux vlab_check "$PY" -m experiments.vlab_check --out "$REG/vlab_check.json"
STEP_CAP_S="${SREMU_REGEN_LATENCY_CAP_S:-900}"
case "$STEP_CAP_S" in ''|*[!0-9]*) STEP_CAP_S=900 ;; esac
fg_aux ef_latency env -u NUMBA_NUM_THREADS -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS \
  -u MKL_NUM_THREADS "$PY" -c \
  "import sys; from experiments.ef import run_latency; run_latency(sys.argv[1])" \
  "$OUT/ef_latency"
STEP_CAP_S=0

# GNU Radio for E-E and the junit's GNU Radio tests (Linux): probed in its own interpreter and
# installed from the Ubuntu archive when missing, in the background from here on (after the
# latency step, which needs the idle host). Not in BG_PIDS: wait_gr waits for it when needed.
GR_PID=""
GR_OK=0
GR_WAITED=0
if { [ "$RUN_EE" = 1 ] || [ "$RUN_JUNIT" = 1 ]; } && [ "$IS_LINUX" = 1 ]; then
  run_step gr_setup "$PY" -m experiments.regen_checks gnuradio --gr-python "$GR_PY" \
    --install auto --out "$REG/gnuradio.json" &
  GR_PID=$!
elif [ "$IS_LINUX" != 1 ]; then
  log "GNU Radio: not Linux ($(uname -s)); E-E skipped, the junit's GNU Radio tests skip"
fi
wait_gr() {
  if [ "$GR_WAITED" = 1 ] || [ -z "$GR_PID" ]; then return 0; fi
  GR_WAITED=1
  if wait "$GR_PID"; then GR_OK=1; else FAILED_AUX+=(gr_setup); fi
}

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
# E-B on the lab arc (code-defined design, hash-checked in setup; aggregated by eb.py itself)
if [ "$MODE" = smoke ]; then ARC_SMOKE=(--smoke); else ARC_SMOKE=(); fi
fg_aux eb_arc "${SWEEP_ENV[@]}" "$PY" -m experiments.eb --scenario scenarios/studies/lab_arc.yaml \
  --out "$OUT/eb_arc" --jobs "$SWEEP_JOBS" --threads-per-job 1 "${ARC_SMOKE[@]}"
# validation studies the paper quotes, beside VA-1 and E-C (each a run directory under
# validation/<study>/; they record the job's git state through SREMU_EXP_GIT_JSON)
mkdir -p "$OUT/validation"
bg_aux v_tang env NUMBA_NUM_THREADS=1 "$PY" validation/tang/vtang.py --root "$OUT/validation/tang"
bg_aux v_fm env NUMBA_NUM_THREADS=1 "$PY" validation/fm/vfm.py --root "$OUT/validation/fm"
if [ "$MODE" = smoke ]; then VCFO_FAST=(--fast); else VCFO_FAST=(); fi
bg_aux v_cfo env NUMBA_NUM_THREADS=1 "$PY" validation/cfo/vcfo.py --jobs "$VCFO_JOBS" \
  --root "$OUT/validation/cfo" "${VCFO_FAST[@]}"
# E-E repetitions (Linux): after the sweeps, beside VA-1, E-C and the validation studies. The
# runs are lockstep (virtual time follows the sample exchange), so the load changes no value
# and no hash; their wall times are not quoted. The throughput runs follow at the end, alone.
EE_STARTED=0
EE_NOTE=""
if [ "$RUN_EE" = 1 ] && [ "$IS_LINUX" = 1 ]; then
  wait_gr
  if [ "$GR_OK" = 1 ]; then
    EE_COMMIT="$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['commit'])" \
      "$REG/git_state.json" 2>/dev/null)"
    EE_GRV="$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['gnuradio_version'])" \
      "$REG/gnuradio.json" 2>/dev/null)"
    EE_NOTE="pinned regeneration job (experiments/regenerate_paper_data.sh, mode $MODE) at sremu"
    EE_NOTE="$EE_NOTE $EE_COMMIT on $(hostname) ($(nproc) logical CPUs); GNU Radio $EE_GRV from"
    EE_NOTE="$EE_NOTE the Ubuntu archive (_regen/gnuradio.json); served engine NUMBA_NUM_THREADS="
    EE_NOTE="$EE_NOTE$EE_THREADS; repetitions run beside the VA-1 and E-C processes (lockstep:"
    EE_NOTE="$EE_NOTE values and hashes do not depend on load); throughput runs alone on the"
    EE_NOTE="$EE_NOTE otherwise idle host after every other experiment (MEAS of this host)"
    STEP_CAP_S=2700
    bg_aux ee_reps env NUMBA_NUM_THREADS="$EE_THREADS" SREMU_GR_PYTHON="$GR_PY" \
      "$PY" -m experiments.ee_gnuradio run --root "$OUT/ee" --reps 2 --gr-python "$GR_PY" \
      --note "$EE_NOTE"
    STEP_CAP_S=0
    EE_STARTED=1
  else
    log "E-E not started: GNU Radio is not usable (see $LOGS/gr_setup.log)"
  fi
elif [ "$RUN_EE" = 1 ]; then
  log "E-E skipped: it needs Linux with GNU Radio (this is $(uname -s))"
fi
log "sweeps done; waiting for VA-1, E-C, the validation studies$( [ "$EE_STARTED" = 1 ] && echo " and the E-E repetitions")"
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
    rel = m.relative_to(out).parts
    # E-E: the served emulator (sremu.commands.serve) records the git state it reads itself;
    # experiments.regen_checks ee reports those manifests
    if "_regen" in rel or rel[0] == "ee" or rel[:2] == ("paper_results", "E-E"):
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

# E-B arc and validation runs: complete, ok, and on the job's git state (reported; a failure
# here does not stop finalize of the WP11 results)
fg_aux check_aux "$PY" - "$OUT" "$MODE" <<'PYEOF'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

out, mode = Path(sys.argv[1]), sys.argv[2]
gs = json.loads((out / "_regen" / "git_state.json").read_text(encoding="utf-8"))
state = (gs.get("commit"), gs.get("dirty"))
bad = []
sfx = "_smoke" if mode == "smoke" else ""
arc = out / "eb_arc"
summ = arc / f"eb_summary{sfx}.json"
arc_rep = None
if not summ.is_file():
    bad.append(f"eb_arc: {summ.name} missing")
else:
    s = json.loads(summ.read_text(encoding="utf-8"))
    arc_rep = {
        "ok": s.get("ok"),
        "n_failed": s.get("n_failed"),
        "va5_pass": (s.get("va5") or {}).get("pass"),
        "points": {k: v.get("n_ok") for k, v in (s.get("specs") or {}).items()},
    }
    if not (s.get("ok") and s.get("n_failed") == 0 and arc_rep["va5_pass"]):
        bad.append(f"eb_arc: not ok {arc_rep}")
    want = 2 if mode == "smoke" else 447
    if sum(arc_rep["points"].values()) != want:
        bad.append(f"eb_arc: {sum(arc_rep['points'].values())} points ok, expected {want}")
    ags = json.loads((arc / "git_state.json").read_text(encoding="utf-8"))
    if (ags.get("commit"), ags.get("dirty")) != state:
        bad.append("eb_arc: git_state.json differs from _regen/git_state.json")
val = {}
for study, sub, name in (("V-TANG", "tang", "vtang"), ("V-FM", "fm", "vfm"), ("V-CFO", "cfo", "vcfo")):
    runs = sorted(p for p in (out / "validation" / sub).glob(f"*_{name}*") if p.is_dir())
    if len(runs) != 1 or not (runs[0] / "meta.json").is_file():
        bad.append(f"{study}: {len(runs)} run directories with meta.json under validation/{sub}")
        continue
    meta = json.loads((runs[0] / "meta.json").read_text(encoding="utf-8"))
    acc = meta.get("acceptance") or {}
    val[study] = {
        "run_dir": runs[0].name,
        "sremu_commit": meta.get("sremu_commit"),
        "sremu_dirty": meta.get("sremu_dirty"),
        "git_state_source": meta.get("git_state_source"),
        "acceptance": acc,
    }
    if (meta.get("sremu_commit"), meta.get("sremu_dirty")) != state:
        bad.append(f"{study}: meta.json git state differs from _regen/git_state.json")
    if study == "V-CFO":
        idx = runs[0] / "vcfo4" / "index.csv"
        with idx.open(encoding="utf-8", newline="") as fh:
            st = Counter(r["status"] for r in csv.DictReader(fh))
        val[study]["vcfo4_status"] = dict(st)
        if set(st) != {"ok"}:
            bad.append(f"V-CFO: vcfo4 points {dict(st)}")
rep = {"eb_arc": arc_rep, "validation": val, "problems": bad}
(out / "_regen" / "check_aux.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
print(json.dumps(rep, indent=1))
sys.exit(1 if bad else 0)
PYEOF

# ----------------------------------------------------------------------------- finalize
if [ "${#FAILED[@]}" -eq 0 ]; then
  if [ "$MODE" = smoke ]; then FIN_ARGS=(--smoke); else FIN_ARGS=(); fi
  case " ${FAILED_AUX[*]:-} " in
    *" eb_arc "*) log "E-B arc failed: finalize runs without --eb-arc" ;;
    *) FIN_ARGS+=(--eb-arc "$OUT/eb_arc") ;;
  esac
  if [ "$COLLECT_LATENCY" = 1 ]; then
    case " ${FAILED_AUX[*]:-} " in
      *" ef_latency "*) log "E-F latency failed: finalize runs without --with-latency" ;;
      *) FIN_ARGS+=(--with-latency) ;;
    esac
  fi
  fg_step finalize "$PY" -m experiments.finalize --runs "$OUT" --out "$OUT/paper_results" \
    "${FIN_ARGS[@]}"
else
  log "finalize skipped: failed steps: ${FAILED[*]}"
fi

# reference tree (paper/sweeps/results) and paper repository, when at hand (not on AWS)
REF=""
PAPER_DIR=""
if [ -n "${SREMU_REGEN_REF:-}" ]; then
  REF="$(to_native "$SREMU_REGEN_REF")"
  case "${SREMU_REGEN_REF%/}" in
    */sweeps/results) PAPER_DIR="$(to_native "$(cd "$SREMU_REGEN_REF/../.." && pwd)")" ;;
  esac
elif [ -d "$ROOT/../paper/sweeps/results" ]; then
  REF="$(to_native "$(cd "$ROOT/../paper/sweeps/results" && pwd)")"
  PAPER_DIR="$(to_native "$(cd "$ROOT/../paper" && pwd)")"
fi

# ----------------------------------------------------------------------------- E-E tail
# Throughput runs alone on the otherwise idle host (every other experiment has finished), then
# the export in the layout of paper/sweeps/results/E-E and its check. ee_gnuradio run rewrites
# ee/host.json (same host, git state and note; its argv becomes the throughput call): the
# repetitions' record is kept as ee/host_reps.json.
if [ "$EE_STARTED" = 1 ] && [ -d "$OUT/ee/rep1" ]; then
  cp "$OUT/ee/host.json" "$OUT/ee/host_reps.json" 2>/dev/null || true
  NO_DEADLINE=0
  STEP_CAP_S=900
  fg_aux ee_throughput env NUMBA_NUM_THREADS="$EE_THREADS" SREMU_GR_PYTHON="$GR_PY" \
    "$PY" -m experiments.ee_gnuradio run --root "$OUT/ee" --reps 0 --throughput \
    --rates "$EE_RATES" --throughput-s "$EE_TPUT_S" --gr-python "$GR_PY" --note "$EE_NOTE"
  STEP_CAP_S=0
  NO_DEADLINE=1
  fg_aux ee_export "$PY" -m experiments.ee_gnuradio export --root "$OUT/ee" \
    --dest "$OUT/paper_results/E-E"
  EE_REF=()
  if [ -n "$REF" ] && [ -f "$REF/E-E/summary.json" ]; then EE_REF=(--ref "$REF/E-E"); fi
  fg_aux ee_check "$PY" -m experiments.regen_checks ee --root "$OUT/ee" \
    --export "$OUT/paper_results/E-E" --git-state "$REG/git_state.json" --reps 2 \
    --rates "$EE_RATES" --out "$REG/ee_check.json" "${EE_REF[@]}"
fi

# ----------------------------------------------------------------------------- junit
# The paper junit (tools/tasks.py paper-junit: CI + nightly markers, -n 8) on this host after
# everything else: the git state from _regen/git_state.json (no .git in a job bundle), the
# GNU Radio tests with SREMU_GR_PYTHON, the VA-1 record test on the regenerated E-A-IQ.
# The step's own exit code (failing tests) is judged by junit_check (smoke: the VA-1 record
# of a smoke run is incomplete by design, so that one test may fail).
if [ "$RUN_JUNIT" = 1 ]; then
  wait_gr
  mkdir -p "$OUT/junit"
  if [ "$MODE" = smoke ] && [ "${SREMU_REGEN_JUNIT_FULL:-0}" != 1 ]; then
    JUNIT_SCOPE=subset
    JUNIT_SEL=(tests/validation/test_va1_record.py tests/integration/test_gnuradio_raw.py
               tests/integration/test_gnuradio_analysis.py tests/unit/test_regen_tools.py)
  else
    JUNIT_SCOPE=full
    JUNIT_SEL=()
  fi
  HOST_LABEL="$("$PY" -m experiments.regen_checks host-label --provenance "$REG/provenance.json" \
    2>/dev/null || echo n/r)"
  JUNIT_NOTE="tools/tasks.py paper-junit (CI + nightly markers, -n 8) in the pinned regeneration"
  JUNIT_NOTE="$JUNIT_NOTE job (experiments/regenerate_paper_data.sh, mode $MODE, scope"
  JUNIT_NOTE="$JUNIT_NOTE $JUNIT_SCOPE) on $HOST_LABEL; git state from _regen/git_state.json;"
  JUNIT_NOTE="$JUNIT_NOTE SREMU_GR_PYTHON=$GR_PY; SREMU_VA1_RECORD = the job's paper_results/E-A-IQ"
  # Without a paper repository next to sremu (the AWS bundle), tests/unit/test_eb_arc.py's
  # paper-spec test would skip, and a skipped test carrying E-B leaves the E-B claims
  # pending. It reads ../paper/sweeps/eb*.yaml only: give it the job's copies of the paper
  # sweep specs (_regen/specs, each checked against its pinned SweepSpec hash in setup) for
  # the junit, and remove them afterwards. An existing ../paper is never touched.
  PAPER_MIRROR=""
  if [ ! -e "$ROOT/../paper" ] && [ "$JUNIT_SCOPE" = full ]; then
    PAPER_MIRROR="$ROOT/../paper"
    if mkdir -p "$PAPER_MIRROR/sweeps" && cp "$REG/specs/"*.yaml "$PAPER_MIRROR/sweeps/"; then
      log "junit: $PAPER_MIRROR/sweeps holds the job's hash-checked paper sweep specs (removed after)"
      JUNIT_NOTE="$JUNIT_NOTE; ../paper/sweeps = the job's hash-checked copies of the paper sweep specs"
    fi
  fi
  NO_DEADLINE=0
  STEP_CAP_S="${SREMU_REGEN_JUNIT_CAP_S:-2400}"
  case "$STEP_CAP_S" in ''|*[!0-9]*) STEP_CAP_S=2400 ;; esac
  run_step junit env -u SREMU_EXP_GIT_JSON -u OMP_NUM_THREADS -u OPENBLAS_NUM_THREADS \
    -u MKL_NUM_THREADS SREMU_GR_PYTHON="$GR_PY" SREMU_VA1_RECORD="$OUT/paper_results/E-A-IQ" \
    "$PY" tools/tasks.py paper-junit "$OUT/junit/sremu_paper.xml" \
    --git-state "$REG/git_state.json" --host "$HOST_LABEL" --note "$JUNIT_NOTE" "${JUNIT_SEL[@]}"
  log "junit exit $? (judged by junit_check)"
  STEP_CAP_S=0
  NO_DEADLINE=1
  if [ -n "$PAPER_MIRROR" ]; then  # only what was created above
    rm -f "$PAPER_MIRROR/sweeps/"*.yaml
    rmdir "$PAPER_MIRROR/sweeps" "$PAPER_MIRROR" 2> /dev/null || log "junit: $PAPER_MIRROR left in place"
  fi
  JC_ARGS=()
  if [ -n "$PAPER_DIR" ] && [ -f "$PAPER_DIR/claims.yaml" ]; then
    JC_ARGS+=(--claims "$PAPER_DIR/claims.yaml"); fi
  if [ -n "$PAPER_DIR" ] && [ -f "$PAPER_DIR/generated/junit/sremu_paper.xml" ]; then
    JC_ARGS+=(--ref-junit "$PAPER_DIR/generated/junit/sremu_paper.xml"); fi
  if [ "$IS_LINUX" = 1 ]; then JC_GR=yes; else JC_GR=no; fi
  fg_aux junit_check "$PY" -m experiments.regen_checks junit --xml "$OUT/junit/sremu_paper.xml" \
    --mode "$MODE" --scope "$JUNIT_SCOPE" --expect-gnuradio "$JC_GR" \
    --out "$REG/junit_check.json" "${JC_ARGS[@]}"
fi

# ----------------------------------------------------------------------------- compare
COMPARE_RC=""
if [ -n "$REF" ] && [ -f "$OUT/paper_results/summary.json" ]; then
  if [ "$MODE" = smoke ]; then CMP_SMOKE=(--smoke); else CMP_SMOKE=(); fi
  run_step compare "$PY" experiments/compare_regen.py "$OUT/paper_results" --ref "$REF" \
    --validation-new "$OUT/validation" --json "$OUT/compare_report.json" "${CMP_SMOKE[@]}"
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
  "${FAILED_AUX[*]:-}" > "$LOGS/provenance_finish.log" 2>&1 <<'PYEOF'
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
reg = out / "_regen"
t0, t1, cmp_rc, failed = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5].split()
failed_aux = sys.argv[6].split()
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


def rec(name: str):
    p = reg / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


prov.update(
    {
        "ended_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_s": t1 - t0,
        "steps": steps,
        "failed_steps": failed,
        "failed_aux_steps": failed_aux,
        "finalized": (out / "paper_results" / "summary.json").is_file() and not failed,
        "eb_arc_collected": (out / "paper_results" / "E-B-arc" / "meta.json").is_file(),
        "latency_collected": (out / "paper_results" / "E-F" / "latency").is_dir(),
        "compare_exit": None if cmp_rc == "none" else int(cmp_rc),
        "check_runs": rec("check_runs.json"),
        "check_aux": rec("check_aux.json"),
        "vlab_check": rec("vlab_check.json"),
        "gnuradio": rec("gnuradio.json"),
        "ee_collected": (out / "paper_results" / "E-E" / "summary.json").is_file(),
        "ee_check": rec("ee_check.json"),
        "junit_written": (out / "junit" / "sremu_paper.xml").is_file(),
        "junit_check": rec("junit_check.json"),
        "ea_cache_files": cache,
    }
)
with (reg / "provenance.json").open("w", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(prov, indent=1) + "\n")
keys = ("wall_s", "failed_steps", "failed_aux_steps", "finalized", "eb_arc_collected",
        "ee_collected", "junit_written", "compare_exit")
print(json.dumps({k: prov[k] for k in keys}))
PYEOF
cat "$LOGS/provenance_finish.log"

if [ "${#FAILED[@]}" -gt 0 ] || [ "${#FAILED_AUX[@]}" -gt 0 ]; then
  log "FAILED steps: ${FAILED[*]:-} ${FAILED_AUX[*]:-} (total $(( T_END - T_START )) s)"
  exit 1
fi
log "done in $(( T_END - T_START )) s; results in $OUT/paper_results (mode $MODE)"
if [ "$MODE" = full ] && [ -n "$COMPARE_RC" ] && [ "$COMPARE_RC" -ne 0 ]; then
  log "the regenerated results differ from the reference tree (see compare_report.json)"
  exit 4
fi
exit 0
