# sremu benchmarks (WP11; DESIGN.md §12)

| Suite | Command | Host | What it reports |
|---|---|---|---|
| B1 | `python -m bench.b1_kernels` | laptop; Linux job `b1` (one pinned 16-vCPU slot) | kernel timings (K1 snapshot, GEMV, render with T_FD ∈ {0, 8, 16, 32}, noise) per backend; exponents vs O(L·N) and O(L·T·S); NNLS fit of the §7.4 page-cost model from engine runs |
| B2 | `python -m bench.b2_lockstep` | laptop (plan `slices`); Linux job `b2` (plan `server`, c7i.48xlarge) | lockstep RTF and RTF_min (100 ms windows) over L × N × RIS count × f_s × F0..F5, pruned by a plan and by series pruning; non-zero RIS path asserted per scene; `dropped.csv` gives the reason for every product point not run |
| B3 | `python -m bench.b3_paced` | **Linux job `b3`** (quiet c7i.8xlarge); laptop only as "Windows, not representative" | paced sustainable L per series (f_s, page target, N, R, F), miss rate at 0/1/5 ms slack, page-latency histograms (CCDF); `protocol_12_3` in the summary |
| B4 | `python -m bench.b4_control --part lockstep` / `--part paced` | laptop / **Linux job `b4`** | command lateness (lockstep must be 0) and visible latency, 10 Hz–10 kHz, in-process, remote barrier and paced async, on the base scene with `--links` L (default 2). The lockstep pass needs real evidence (`lockstep_check_ok`: ≥ 90 % of the expected commands, all `applied`, one lateness value per command and RX, all 0). Every paced repetition states whether it was sustainable, the rate the client really achieved (REQ/REP: a slow engine caps it) and the command rate inside the measured virtual window (`engine_bound` when it is below half the requested rate) |
| B6 | `python -m bench.b6_crosstool` | laptop | time per RIS channel evaluation: sremu (numba, NumPy) vs MATLAB R2026a (per-element loop script, vectorized, toolbox path, `phased.RISReflector`); `context_table.csv` with a scope note per row and REPORTED literature numbers |
| E-B3 | `python -m bench.eb3_paced` | Linux | external TDMA controller over ZMQ, paced + async: lateness and switch-inside-packet rate (not a PER) |

There is no B5 (DESIGN.md D24: sremu is judged on its own evidence).

`sremu bench --suite {b1,b2,b3,b4,b6,eb3} [args]` runs the same scripts. `python -m bench.run_laptop`
runs the laptop campaign (B1, B2, B4 lockstep, B6, B3 "Windows") with load gates, and
`python -m bench.summarize` collects the headline numbers of the latest run of every suite
into `runs/bench/bench_summary.json`.

Every suite writes one run directory `<out>/<UTC>_<suite>[_<tag>]/` with `manifest.json`
(git commit and dirty flag, versions, platform, power plan or governor, threads, host-load
samples), `spec.json` (hashed into `scenario_sha256`), CSV/JSON data, `scenarios/*.yaml`
(every resolved scenario, hashed in the tables), `meta.json` (data classes, platform label,
caveats, background load, quiet gate, **pinning**: affinity, `NUMBA_NUM_THREADS`,
`SREMU_CPUS`, `SREMU_CLIENT_CPUS`, governor; `interrupted`) and `summary.json`. Numbers are
reported exactly as measured; nothing is compared across machines.

## B2 plans and pruning

* **`slices`** (laptop): one-factor slices S1–S6 through the base point (N = 240, R = 1,
  1 MS/s, F1), sequential, in-process.
* **`server`** (large Linux host): tier 0 the slices; tier 1 the mobility grid (F3:
  L × N × R ∈ {1, 2, 4} × f_s, plus R = 0); tier 2 the static grid (F1: N ∈ {240, 16384});
  tier 3 F0..F5 at three anchors (N, R, f_s); tier 4 S7 thread scaling (T ∈ {32, 48, 96}) run
  alone on the host. 909 points in 118 series; the other 2952 product points are listed with
  their reason (R = 0 makes N irrelevant; in a static scene N only enters the one-time
  snapshot; feature sets only at the base and anchors).
* **Series pruning**: points of a series (same N, R, f_s, F, threads) run in increasing L;
  a point whose wall-limited pilot RTF is below `--rtf-floor` (default 0.005) is recorded as
  `below_floor` (pilot RTF only, never reported as a measured RTF) and the larger-L points
  of its series are dropped with the reason. Failures and timeouts stop a series too.
* **Parallel workers** (`--workers auto --cpus-per-point 16`): one worker process per point,
  pinned to a slot of whole physical cores inside one NUMA node (`sched_setaffinity` on every
  thread of the worker, including the OpenBLAS pool that `import numpy` starts before the pin;
  each row records `thread_affinities_end` and `all_threads_in_slot`); core 0
  is kept for the OS and the orchestrator. On c7i.48xlarge (96 cores, 2 nodes) that is 11
  concurrent 16-vCPU slots. Concurrent points share L3, memory bandwidth and power, so the
  run ends with an **isolation check** (`isolation.csv`: reference points rerun alone,
  `rtf_isolated / rtf_parallel`).

## Linux jobs (lead's AWS runner)

Transport is the runner's (`research/aws/runner.py`: bundle, presigned upload, power-off).
A job is one shell command from the bundle root with `SREMU_JOB_OUT` set:

```
bash bench/aws/job.sh JOB [--deadline-min M] [--cpus-per-slot 16] [--client-cores 2] [--dry]
```

`job.sh` stops package timers and snaps (root, best effort), asks for the performance
governor where cpufreq exists (c7i usually exposes none; recorded as "unavailable") and runs
`bench/aws/run_job.py`, which writes `sysinfo.json`, `status.json`, `<step>.log` and the run
directories under `$SREMU_JOB_OUT/runs/`. At `--deadline-min` a running suite gets SIGINT,
writes its partial run directory and the job ends, so the upload still happens.

| Job | Steps | Pinning | Instance |
|---|---|---|---|
| `b1b2` | B1 (all kernels + cost-model fit), then B2 `--plan server --workers auto` | B1 on slot 0 (16 vCPUs, node 0, rest idle); B2 orchestrator on core 0, workers on their slots | c7i.48xlarge |
| `b3b4` | B3 `--plan server`, B4 lockstep, B4 paced | engine on 8 cores (16 vCPUs) after core 0, `NUMBA_NUM_THREADS=16`; clients on the next 2 cores; rest idle | c7i.8xlarge |
| `b1`, `b2`, `b3`, `b4`, `eb3` | single suites (`eb3`: E-B3, pinned as B3) | as above | |
| `smoke` | quick B1, B2 (2 workers), B3, B4 | as above | any |

AWS commands (PowerShell, from `D:\projects\ris\research\aws`; each launch needs the user's
approval). The bundle is a `git archive` without `.git`, so the command passes the commit in
`SREMU_GIT_COMMIT`/`SREMU_GIT_DIRTY` and every manifest records it. After `bench/` and
`src/sremu/commands/bench.py` are committed:

```
$c = git -C D:\projects\ris\sremu rev-parse HEAD
python runner.py launch b1b2 --type c7i.48xlarge --maxmin 240 --disk 60 --run "SREMU_GIT_COMMIT=$c SREMU_GIT_DIRTY=0 bash bench/aws/job.sh b1b2 --deadline-min 210"
python runner.py launch b3 --type c7i.8xlarge --maxmin 210 --disk 40 --run "SREMU_GIT_COMMIT=$c SREMU_GIT_DIRTY=0 bash bench/aws/job.sh b3 --deadline-min 180"
# B4 on its own quiet host, best after the per-command engine cost is fixed:
python runner.py launch b4 --type c7i.8xlarge --maxmin 120 --disk 40 --run "SREMU_GIT_COMMIT=$c SREMU_GIT_DIRTY=0 bash bench/aws/job.sh b4 --deadline-min 95"
# or B3 then B4 on one host:
python runner.py launch b3b4 --type c7i.8xlarge --maxmin 270 --disk 40 --run "SREMU_GIT_COMMIT=$c SREMU_GIT_DIRTY=0 bash bench/aws/job.sh b3b4 --deadline-min 240"
```

While they are uncommitted, add `--extra bench src/sremu/commands/bench.py` and set
`SREMU_GIT_DIRTY=1` (the runner never overrides files that are in `HEAD`, so an older
committed `bench/` would win over the extras). Estimates: B2 server plan about 1 h of
parallel sweep; B3 about 2.5 h, capped by its budget (the lowest-priority series are dropped
first, `dropped.csv`); B4 about 1 h (lockstep about 5 min; paced repetitions at >= 100
commands/s run about 2.4x their virtual length while a configuration command costs
milliseconds of engine time, and the client's achieved rate is reported). A suite still
running at `--deadline-min` gets SIGINT and writes its partial run directory; the runner
powers the instance off when the job ends, so `--maxmin` only caps a hang.
