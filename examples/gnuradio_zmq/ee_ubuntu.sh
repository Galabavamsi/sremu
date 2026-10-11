#!/usr/bin/env bash
# E-E on an Ubuntu 24.04 host (WSL2, native, or an AWS c7i instance): stock GNU Radio through
# sremu's raw REQ/REP mode. Run from the sremu checkout:
#
#   OUT=/opt/job/out bash examples/gnuradio_zmq/ee_ubuntu.sh [--setup] [--rates LIST] [--reps N]
#
#   --setup        apt-get install gnuradio (Ubuntu 24.04: 3.10.9.2, system python3 with numpy
#                  1.26) and create the sremu venv $VENV (default $HOME/sremu-venv; numpy >= 2.2).
#                  The two interpreters never mix: they meet only over loopback ZMQ.
#   --rates LIST   throughput rates, S/s (default 1e6,1.92e6,3.84e6,5.76e6,11.52e6,23.04e6)
#   --reps N       independent repetitions of the full design (default 2)
#
# Writes $OUT/ee (run directories with the IQ, about 0.3 GB per repetition plus the throughput
# runs) and $OUT/paper_results/E-E (summary.json, table_ee.csv, meta.json, small per-run files:
# what paper/sweeps/results/E-E holds). Lockstep: the real-time factors are virtual seconds per
# wall second of this host (MEAS), not a real-time claim. Exit code: that of the export (0 when
# every run passed its checks and the repetitions are bitwise identical).
set -euo pipefail
: "${OUT:?set OUT to an empty output directory}"
VENV=${VENV:-$HOME/sremu-venv}
RATES="1e6,1.92e6,3.84e6,5.76e6,11.52e6,23.04e6"
REPS=2
SETUP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --setup) SETUP=1 ;;
    --rates) RATES=$2; shift ;;
    --reps) REPS=$2; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
. /etc/os-release
[ "${VERSION_ID:-}" = "24.04" ] || echo "warning: tested on Ubuntu 24.04 only (this is ${VERSION_ID:-?})" >&2
if [ "$SETUP" = 1 ]; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y gnuradio python3-venv
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -e ".[dev]"
fi
export SREMU_GR_PYTHON=${SREMU_GR_PYTHON:-/usr/bin/python3}
"$SREMU_GR_PYTHON" -I -c "from gnuradio import gr, zeromq, digital; print('GNU Radio', gr.version())"
"$VENV/bin/python" -c "import sremu, numpy; print('sremu', sremu.__file__, 'numpy', numpy.__version__)"
mkdir -p "$OUT"
"$VENV/bin/python" -m experiments.ee_gnuradio run --root "$OUT/ee" --reps "$REPS" \
  --throughput --rates "$RATES" --throughput-s 2 --note "${EE_NOTE:-ee_ubuntu.sh on $(hostname)}" \
  | tee "$OUT/ee_run.log" || echo "ee_gnuradio run reported a failed run (see ee_run.log)" >&2
# the export checks every run again and fails (non-zero) on any problem
"$VENV/bin/python" -m experiments.ee_gnuradio export --root "$OUT/ee" --dest "$OUT/paper_results/E-E"
