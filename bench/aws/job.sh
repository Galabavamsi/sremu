#!/usr/bin/env bash
# sremu benchmark job entry point for the lead's AWS runner (DESIGN.md 12.3-12.4; WP11).
#
# The runner (research/aws/runner.py) boots Ubuntu 24.04, unpacks the bundle, activates the
# venv, exports SREMU_JOB_OUT and runs ONE shell command from the bundle root; it uploads
# SREMU_JOB_OUT afterwards and powers the instance off. This script is that command:
#
#   bash bench/aws/job.sh JOB [run_job.py options]      e.g.
#   bash bench/aws/job.sh b1b2 --deadline-min 210       (c7i.48xlarge)
#   bash bench/aws/job.sh b3b4 --deadline-min 210       (c7i.8xlarge)
#
# JOB: b1 | b2 | b1b2 | b3 | b4 | b4classic | b3b4 | eb3 | smoke (see bench/aws/run_job.py).
# The bundle has no .git: prefix the command with SREMU_GIT_COMMIT=<sha> SREMU_GIT_DIRTY=0|1
# so every manifest records the source commit.
# It quiets the host (stops package timers and snaps; best effort, needs root), asks for the
# performance governor where the VM exposes cpufreq, records both, then runs the job.
set -uo pipefail

JOB="${1:?usage: bash bench/aws/job.sh JOB [options]}"
shift
OUT="${SREMU_JOB_OUT:?SREMU_JOB_OUT must be set (the runner exports it)}"
PY="${SREMU_PYTHON:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
mkdir -p "$OUT"
cd "$ROOT" || exit 2

quiet_host() {
  local u
  if [ "${SREMU_SKIP_HOST_PREP:-0}" = "1" ]; then
    echo "SREMU_SKIP_HOST_PREP=1: host left as is"
    return 0
  fi
  if [ "$(id -u)" != "0" ] || ! command -v systemctl > /dev/null 2>&1; then
    echo "not root or no systemctl: background services left running"
    return 0
  fi
  for u in unattended-upgrades.service apt-daily.timer apt-daily-upgrade.timer \
           apt-daily.service apt-daily-upgrade.service snapd.service snapd.socket \
           fwupd-refresh.timer motd-news.timer man-db.timer; do
    systemctl stop "$u" > /dev/null 2>&1 && echo "stopped $u"
  done
  return 0
}

set_governor() {
  local g
  if [ "${SREMU_SKIP_HOST_PREP:-0}" = "1" ]; then
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2> /dev/null || echo "unavailable"
    return 0
  fi
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -w "$g" ]; then
      echo performance > "$g" 2> /dev/null || true
    fi
  done
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2> /dev/null \
    || echo "unavailable"
}

{
  echo "job=$JOB date=$(date -u +%Y-%m-%dT%H:%M:%SZ) root=$ROOT"
  quiet_host
} > "$OUT/job_host_prep.txt" 2>&1
GOV="$(set_governor)"
echo "governor=$GOV" >> "$OUT/job_host_prep.txt"
export SREMU_GOVERNOR="$GOV"
export NUMBA_DISABLE_INTEL_SVML="${NUMBA_DISABLE_INTEL_SVML:-1}"
echo "[job.sh] job=$JOB governor=$GOV out=$OUT python=$(command -v "$PY")"
exec "$PY" bench/aws/run_job.py "$JOB" --out "$OUT" "$@"
