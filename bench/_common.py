"""Shared benchmark plumbing: run directories, manifests, load sampling, timers (WP11).

Every suite writes ONE run directory ``<out>/<UTC>_<suite>[_<tag>]/`` (default
``runs/bench/``; ``runs/`` is git-ignored) containing:

* ``manifest.json``: the section 3.6 manifest (:class:`sremu.record.manifest.Manifest`):
  git commit and dirty flag (+ ``dirty.patch``), versions, platform, CPU, power plan or
  governor, AWS instance, seed, RNG scheme, threads, SVML state, argv, start/end times and
  host-load samples. ``scenario_name`` is ``bench-<suite>``; ``scenario_sha256`` is the
  SHA-256 of the canonical JSON of the suite spec (grid, durations, options), so two runs
  with the same spec share it. Per-point scenarios are stored as resolved YAML under
  ``scenarios/`` and their own SHA-256 appears in the point tables.
* ``spec.json``: the suite spec that was hashed.
* CSV/JSON data files written by the suite.
* ``meta.json``: data classes, platform label (a Windows paced number is always labelled
  "Windows, not representative"), caveats, files and the background-load summary.
* ``summary.json``: the suite's headline numbers (exactly as measured).

**Interruption.** A suite wraps its measurement loops in :meth:`BenchRun.interruptible`:
SIGINT (``timeout --signal=INT`` on the AWS instance, Ctrl-C) or SIGTERM (converted to
``KeyboardInterrupt`` by :class:`BenchRun`) ends the loops, and the suite still writes its
partial data, manifest, ``meta.json`` and ``summary.json``, both with
``"interrupted": "<reason>"`` (``null`` for a complete run).

**Background load.** :class:`LoadMonitor` samples the system busy-CPU fraction (Windows
``GetSystemTimes``, Linux ``/proc/stat``, via :class:`sremu.record.manifest.HostLoadSampler`)
every ``period_s`` in a daemon thread, together with this process's CPU time, and reports
``background_busy = busy - own_cpu/(n_cpus*wall)`` (other processes' share; child processes
such as MATLAB, ZMQ clients or controller processes count as background). A short quiet
window before the suite records the idle baseline.
"""

from __future__ import annotations

import contextlib
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO: Final[Path] = Path(__file__).resolve().parents[1]
RUNS_ROOT: Final[Path] = REPO / "runs" / "bench"
WINDOWS_PACED_LABEL: Final[str] = "Windows, not representative"

__all__ = [
    "REPO",
    "RUNS_ROOT",
    "WINDOWS_PACED_LABEL",
    "BenchRun",
    "LoadMonitor",
    "canonical_sha256",
    "jsonable",
    "percentiles",
    "platform_label",
    "time_calls",
    "write_csv",
]


# =========================================================================== helpers


def jsonable(x: Any) -> Any:
    """JSON-ready values (numpy, Fraction, complex, Path; non-finite floats -> None)."""
    from sremu.experiments.sweep import jsonable as _j

    return _j(x)


def canonical_sha256(obj: Any) -> str:
    """SHA-256 of the canonical JSON (sorted keys, compact) of ``obj``."""
    text = json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> Path:
    """Write ``obj`` as indented JSON (UTF-8, LF)."""
    path.write_text(
        json.dumps(jsonable(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write rows (union of keys, first-seen order) as CSV with LF line endings."""
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_cell(r.get(k)) for k in keys})
    return path


def _csv_cell(v: Any) -> Any:
    v = jsonable(v)
    if isinstance(v, dict | list):
        return json.dumps(v, separators=(",", ":"))
    if v is None:
        return ""
    return v


def percentiles(xs: Sequence[float], ps: Iterable[float] = (50, 90, 99)) -> dict[str, float]:
    """Nearest-rank percentiles ``{"p50": ..}`` plus min/max/mean/n (empty -> {})."""
    if not xs:
        return {}
    s = sorted(float(x) for x in xs)
    out: dict[str, float] = {"n": float(len(s)), "min": s[0], "max": s[-1]}
    out["mean"] = statistics.fmean(s)
    for p in ps:
        k = max(1, math.ceil(p / 100.0 * len(s)))
        out[f"p{p:g}"] = s[k - 1]
    return out


def time_calls(
    fn: Callable[[], Any],
    *,
    min_time_s: float = 0.2,
    min_calls: int = 3,
    max_calls: int = 10_000,
    warmup: int = 1,
) -> dict[str, float]:
    """Time repeated calls of ``fn`` (``perf_counter_ns``): ``warmup`` untimed calls, then
    at least ``min_calls`` and until ``min_time_s`` has elapsed (capped at ``max_calls``).
    Returns per-call ``min``, ``median``, ``mean``, ``p90`` (s) and ``n``."""
    for _ in range(warmup):
        fn()
    ts: list[float] = []
    t_end = time.perf_counter() + min_time_s
    while len(ts) < min_calls or (time.perf_counter() < t_end and len(ts) < max_calls):
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) * 1e-9)
    ts.sort()
    return {
        "min": ts[0],
        "median": statistics.median(ts),
        "mean": statistics.fmean(ts),
        "p90": ts[max(0, math.ceil(0.9 * len(ts)) - 1)],
        "n": float(len(ts)),
    }


PINNING_ENV: Final[tuple[str, ...]] = (
    "NUMBA_NUM_THREADS",
    "SREMU_CPUS",
    "SREMU_CLIENT_CPUS",
    "SREMU_GOVERNOR",
)


def pinning_info() -> dict[str, Any]:
    """CPU affinity of this process (Linux), the pinning environment set by
    ``bench/aws/run_job.py`` and the governor or power plan (section 12.3: "state ...
    threads, pinning")."""
    from bench._cpus import format_cpu_list, online_cpus
    from sremu import sysinfo

    return {
        "affinity": format_cpu_list(online_cpus()),
        "affinity_supported": hasattr(os, "sched_getaffinity"),
        "env": {k: os.environ.get(k) for k in PINNING_ENV},
        "power_plan_or_governor": sysinfo.power_plan_or_governor(),
    }


def platform_label(paced: bool = False) -> str:
    """Human label of the host for tables and figure metadata.

    Paced numbers from Windows are always labelled ``"Windows, not representative"``
    (section 12.3); lockstep numbers carry the OS and CPU.
    """
    from sremu import sysinfo

    info = sysinfo.platform_info()
    aws = info.get("aws_instance")
    host = f"{info.get('cpu_model')}, {info.get('logical_cpus')} logical CPUs, {platform.system()}"
    if aws:
        host = f"AWS {aws.get('type')} ({host})"
    if paced and sys.platform == "win32":
        return f"{host} [{WINDOWS_PACED_LABEL}]"
    return host


# =========================================================================== load monitor


class LoadMonitor:
    """Background CPU-load sampler (see the module docstring)."""

    def __init__(self, period_s: float = 2.0) -> None:
        from sremu.record.manifest import HostLoadSampler

        self.period_s = float(period_s)
        self._sampler = HostLoadSampler()
        self._ncpu = os.cpu_count() or 1
        self._last_wall = time.perf_counter()
        self._last_cpu = time.process_time()
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def sample(self, tag: str = "") -> dict[str, Any]:
        """Take one sample now (busy fraction since the previous sample)."""
        with self._lock:
            s = self._sampler.sample()
            wall, cpu = time.perf_counter(), time.process_time()
            dw, dc = wall - self._last_wall, cpu - self._last_cpu
            self._last_wall, self._last_cpu = wall, cpu
            own = dc / (self._ncpu * dw) if dw > 0 else None
            busy = s.cpu_busy_fraction
            bg = None if busy is None or own is None else max(0.0, busy - own)
            rec = {
                "t_utc": s.t_utc,
                "tag": tag,
                "cpu_busy_fraction": busy,
                "own_cpu_fraction": own,
                "background_busy_fraction": bg,
                "loadavg_1min": s.loadavg_1min,
                "interval_s": dw,
            }
            self.samples.append(rec)
            return rec

    def quiet_window(self, seconds: float = 3.0, tag: str = "baseline") -> dict[str, Any]:
        """System busy fraction over a ``seconds`` window (own sampler, so the periodic
        samples cannot shorten the window); the record is appended with ``tag``."""
        from sremu.record.manifest import HostLoadSampler

        own = HostLoadSampler()
        w0, c0 = time.perf_counter(), time.process_time()
        time.sleep(seconds)
        s = own.sample()
        dw, dc = time.perf_counter() - w0, time.process_time() - c0
        own_frac = dc / (self._ncpu * dw) if dw > 0 else None
        busy = s.cpu_busy_fraction
        rec = {
            "t_utc": s.t_utc,
            "tag": tag,
            "cpu_busy_fraction": busy,
            "own_cpu_fraction": own_frac,
            "background_busy_fraction": None
            if busy is None or own_frac is None
            else max(0.0, busy - own_frac),
            "loadavg_1min": s.loadavg_1min,
            "interval_s": dw,
        }
        with self._lock:
            self.samples.append(rec)
        return rec

    def wait_quiet(
        self,
        max_busy: float = 0.20,
        window_s: float = 5.0,
        timeout_s: float = 0.0,
        tag: str = "baseline",
    ) -> dict[str, Any]:
        """Wait (up to ``timeout_s``) until the system busy fraction over a ``window_s``
        window is at most ``max_busy``; returns the gate result (never raises)."""
        t0 = time.perf_counter()
        last = self.quiet_window(window_s, tag)
        while (
            last["cpu_busy_fraction"] or 0.0
        ) > max_busy and time.perf_counter() - t0 + window_s <= timeout_s:
            last = self.quiet_window(window_s, tag)
        busy = last["cpu_busy_fraction"]
        return {
            "max_busy": max_busy,
            "window_s": window_s,
            "timeout_s": timeout_s,
            "waited_s": time.perf_counter() - t0,
            "last_busy_fraction": busy,
            "met": busy is not None and busy <= max_busy,
        }

    def mark(self) -> int:
        """Index of the next sample (start of a measurement window)."""
        with self._lock:
            return len(self.samples)

    def window_stats(self, i0: int) -> dict[str, Any]:
        """Background busy fraction (mean, max, n) of the periodic samples since ``mark``
        ``i0`` (one fresh sample is taken first so short windows are covered)."""
        self.sample("window")
        with self._lock:
            xs = [
                r["background_busy_fraction"]
                for r in self.samples[i0:]
                if r["tag"] in ("running", "window") and r["background_busy_fraction"] is not None
            ]
        if not xs:
            return {"n": 0, "mean": None, "max": None}
        return {"n": len(xs), "mean": statistics.fmean(xs), "max": max(xs)}

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            self.sample("running")

    def start(self) -> LoadMonitor:
        """Start periodic sampling in a daemon thread."""
        self.sample("start")
        self._thread = threading.Thread(target=self._run, name="bench-load", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop sampling (takes a final sample)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5.0)
        self.sample("stop")

    def summary(self) -> dict[str, Any]:
        """Background-load statistics over the running samples (and the baseline)."""
        run = [
            s["background_busy_fraction"]
            for s in self.samples
            if s["tag"] in ("running", "window", "stop")
            and s["background_busy_fraction"] is not None
        ]
        base = [
            s["cpu_busy_fraction"]
            for s in self.samples
            if s["tag"] == "baseline" and s["cpu_busy_fraction"] is not None
        ]
        return {
            "n_samples": len(self.samples),
            "period_s": self.period_s,
            "baseline_busy_fraction": base[-1] if base else None,
            "background_busy": percentiles(run, (50, 90, 99)) if run else {},
            "note": (
                "background_busy = system busy fraction minus this process's CPU share; "
                "child processes (MATLAB, ZMQ clients, controller processes, parallel point "
                "workers) count as background"
            ),
        }


# =========================================================================== run directory


def _sigterm_to_keyboard_interrupt() -> Any:
    """Make SIGTERM raise ``KeyboardInterrupt`` in the main thread (so a terminated suite
    still writes its partial run directory). Returns the previous handler (to restore), or
    None when nothing was installed (not the main thread, or SIGTERM not handleable)."""
    if threading.current_thread() is not threading.main_thread():
        return None

    def _handler(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    try:
        return signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError, AttributeError):  # pragma: no cover - platform specific
        return None


def _manifest_dict(
    run_dir: Path,
    *,
    suite: str,
    spec_sha: str,
    seed: int,
    mode: str,
    coupling: str,
    backend: str,
    inputs: Sequence[Mapping[str, str]],
    assumptions: Sequence[str],
    argv: Sequence[str],
) -> dict[str, Any]:
    from sremu import sysinfo
    from sremu.contracts import COUNTER_NAMES
    from sremu.record.manifest import PATCH_NAME, environment_info, git_state, utc_now_iso
    from sremu.rng import RNG_SCHEME

    env = environment_info()
    gs = git_state(REPO)
    if gs.dirty and gs.patch is not None:
        (run_dir / PATCH_NAME).write_bytes(gs.patch)
    commit, dirty = gs.commit, gs.dirty
    notes: list[str] = []
    if commit is None:
        commit, dirty, note = bundle_commit()
        if note:
            notes.append(note)
    aws = env["aws_instance"]
    return {
        **env,
        "aws_instance": None if aws is None else aws.model_dump(),
        "git_commit": commit,
        "git_dirty": dirty,
        "dirty_patch_sha256": gs.patch_sha256,
        "scenario_name": f"bench-{suite}",
        "scenario_sha256": spec_sha,
        "inputs": [dict(i) for i in inputs],
        "assumptions": [*assumptions, *notes],
        "seed": int(seed),
        "rng_scheme": RNG_SCHEME,
        "mode": mode,
        "coupling": coupling,
        "backend": backend,
        "threads": sysinfo.numba_threads(),
        "deterministic": os.environ.get("NUMBA_DISABLE_INTEL_SVML", "") not in ("", "0"),
        "numba_disable_svml": sysinfo.numba_svml_disabled(),
        "page_size": {},
        "links": {},
        "verify_interp_max_rel": None,
        "richardson_db": None,
        "counters": dict.fromkeys(COUNTER_NAMES, 0),
        "nondeterministic_inputs": False,
        "oracle_csi_used": False,
        "argv": list(argv),
        "t_start_utc": utc_now_iso(),
        "t_end_utc": None,
        "wall_s": None,
        "host_load_samples": [],
    }


def bundle_commit() -> tuple[str | None, bool | None, str | None]:
    """Commit of a source tree without ``.git`` (an AWS job bundle is a ``git archive``):
    ``$SREMU_GIT_COMMIT`` and ``$SREMU_GIT_DIRTY`` (``1``/``0``), set by the launch command.
    Returns ``(commit, dirty, note)``; all None when the variables are unset."""
    commit = os.environ.get("SREMU_GIT_COMMIT", "").strip() or None
    if commit is None:
        return None, None, None
    raw = os.environ.get("SREMU_GIT_DIRTY", "").strip()
    dirty = None if raw == "" else raw not in ("0", "false", "False")
    note = (
        "git_commit/git_dirty from $SREMU_GIT_COMMIT/$SREMU_GIT_DIRTY (job bundle without "
        ".git; set by the launch command)"
    )
    return commit, dirty, note


class BenchRun:
    """One suite run directory with manifest, data files, ``meta.json`` and summary.

    ``spec`` is hashed into ``scenario_sha256`` and written to ``spec.json``.
    ``data_classes`` maps output file names (or ``"*"``) to data classes (section 3.6
    ``data_class`` values). Benchmark numbers are wall-clock measurements on the stated host,
    so the default is ``{"*": "MEAS"}``; ``workload_classes`` (default
    ``{"*": "SIM-sremu"}``) states what was timed (``SIM-sremu``, ``MATLAB``, ...). Both go
    to ``meta.json``.
    """

    def __init__(
        self,
        suite: str,
        spec: Mapping[str, Any],
        *,
        out_root: Path | str | None = None,
        tag: str | None = None,
        seed: int = 0,
        mode: str = "lockstep",
        coupling: str = "barrier",
        backend: str = "numba_cpu",
        data_classes: Mapping[str, str] | None = None,
        workload_classes: Mapping[str, str] | None = None,
        caveats: Sequence[str] = (),
        paced: bool = False,
        argv: Sequence[str] | None = None,
        load_period_s: float = 2.0,
        quiet_s: float = 3.0,
        wait_quiet_s: float = 0.0,
        quiet_max_busy: float = 0.20,
    ) -> None:
        from sremu.record.manifest import create_run_dir

        self.suite = suite
        self.spec = dict(spec)
        self.spec_sha = canonical_sha256(self.spec)
        root = (Path(out_root) if out_root is not None else RUNS_ROOT).resolve()
        name = suite if not tag else f"{suite}_{tag}"
        self.dir = create_run_dir(root, name)
        self._t0 = time.perf_counter()
        self.inputs: list[dict[str, str]] = []
        self.assumptions: list[str] = []
        self.caveats = list(caveats)
        self.data_classes = dict(data_classes or {"*": "MEAS"})
        self.workload_classes = dict(workload_classes or {"*": "SIM-sremu"})
        self.paced = paced
        self.files: list[str] = []
        self._argv = list(sys.argv if argv is None else argv)
        self._manifest = _manifest_dict(
            self.dir,
            suite=suite,
            spec_sha=self.spec_sha,
            seed=seed,
            mode=mode,
            coupling=coupling,
            backend=backend,
            inputs=(),
            assumptions=(),
            argv=self._argv,
        )
        write_json(self.dir / "spec.json", {"suite": suite, "sha256": self.spec_sha, **self.spec})
        self.interrupted: str | None = None
        self._prev_sigterm = _sigterm_to_keyboard_interrupt()
        self.load = LoadMonitor(load_period_s)
        self.quiet_gate: dict[str, Any] | None = None
        if wait_quiet_s > 0:
            self.quiet_gate = self.load.wait_quiet(quiet_max_busy, max(quiet_s, 1.0), wait_quiet_s)
            print(f"[bench] quiet gate: {self.quiet_gate}", flush=True)
        elif quiet_s > 0:
            self.load.quiet_window(quiet_s)
        self.load.start()
        self.finished = False

    # ---- content
    def path(self, name: str) -> Path:
        """Absolute path of a file in the run directory (parents created)."""
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write_csv(self, name: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        """Write a CSV data file and register it."""
        p = write_csv(self.path(name), rows)
        self.files.append(name)
        return p

    def write_json(self, name: str, obj: Any) -> Path:
        """Write a JSON data file and register it."""
        p = write_json(self.path(name), obj)
        self.files.append(name)
        return p

    def add_input(self, path: str, sha256: str, data_class: str) -> None:
        """Register an input file for the manifest (deduplicated)."""
        d = {"path": path, "sha256": sha256, "data_class": data_class}
        if d not in self.inputs:
            self.inputs.append(d)

    def add_assumption(self, text: str) -> None:
        """Register an assumption (manifest ``assumptions``)."""
        if text not in self.assumptions:
            self.assumptions.append(text)

    def save_scenario(self, point_id: str, resolved: Any) -> str:
        """Store a resolved scenario under ``scenarios/<point_id>.yaml``; returns its
        SHA-256 and registers its inputs and assumptions."""
        p = self.path(f"scenarios/{point_id}.yaml")
        p.write_text(resolved.to_yaml(), encoding="utf-8", newline="\n")
        for r in resolved.inputs:
            self.add_input(r.path, r.sha256, r.data_class)
        for a in resolved.assumptions:
            self.add_assumption(a)
        return str(resolved.sha256())

    def set_manifest(self, **fields: Any) -> None:
        """Override manifest fields (e.g. ``threads``, ``page_size``, ``counters``)."""
        self._manifest.update(fields)

    @contextlib.contextmanager
    def interruptible(self) -> Iterator[None]:
        """Context for measurement loops: a ``KeyboardInterrupt`` (SIGINT, or SIGTERM via
        :func:`_sigterm_to_keyboard_interrupt`) ends the block and is recorded in
        :attr:`interrupted`; the suite then writes its partial results as usual."""
        try:
            yield
        except KeyboardInterrupt:
            self.interrupted = (
                f"KeyboardInterrupt at {time.perf_counter() - self._t0:.1f} s into the run "
                "(SIGINT or SIGTERM); results are partial"
            )
            print(f"[bench] {self.interrupted}", flush=True)

    # ---- finish
    def finish(
        self, summary: Mapping[str, Any], *, extra_meta: Mapping[str, Any] | None = None
    ) -> Path:
        """Stop load sampling and write ``summary.json``, ``meta.json`` and the manifest."""
        from sremu import sysinfo
        from sremu.record.manifest import (
            HostLoadSample,
            Manifest,
            utc_now_iso,
            write_manifest,
        )

        self.load.stop()
        if self._prev_sigterm is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            self._prev_sigterm = None
        write_json(self.dir / "summary.json", {**summary, "interrupted": self.interrupted})
        load = self.load.summary()
        meta = {
            "suite": self.suite,
            "spec_sha256": self.spec_sha,
            "interrupted": self.interrupted,
            "data_classes": self.data_classes,
            "workload_data_classes": self.workload_classes,
            "platform_label": platform_label(self.paced),
            "paced": self.paced,
            "windows_paced_not_representative": bool(self.paced and sys.platform == "win32"),
            "caveats": self.caveats,
            "files": sorted(set(self.files)),
            "background_load": load,
            "quiet_gate": self.quiet_gate,
            "pinning": pinning_info(),
            "created_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **(dict(extra_meta) if extra_meta else {}),
        }
        write_json(self.dir / "meta.json", meta)
        m = dict(self._manifest)
        m["inputs"] = self.inputs
        m["assumptions"] = [*m.get("assumptions", []), *self.assumptions]
        m["threads"] = m.get("threads") if m.get("threads") is not None else sysinfo.numba_threads()
        m["numba_disable_svml"] = sysinfo.numba_svml_disabled()
        m["t_end_utc"] = utc_now_iso()
        m["wall_s"] = time.perf_counter() - self._t0
        m["host_load_samples"] = [
            HostLoadSample(
                t_utc=s["t_utc"],
                cpu_busy_fraction=s["cpu_busy_fraction"],
                loadavg_1min=s["loadavg_1min"],
            ).model_dump()
            for s in self.load.samples
        ]
        write_manifest(self.dir, Manifest.model_validate(m))
        self.finished = True
        return self.dir
