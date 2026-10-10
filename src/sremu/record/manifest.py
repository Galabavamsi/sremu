"""Run directory and ``manifest.json`` (DESIGN.md section 3.6).

A run directory is ``runs/<UTC>_<name>/`` with ``manifest.json``,
``scenario.resolved.yaml`` and, only when the source tree is dirty, ``dirty.patch``.
:class:`ManifestRecorder` captures the environment and git state when a run starts, collects
what the engine reports while it runs, and writes the manifest when the run finishes.

``manifest.json`` carries every field required by section 3.6 (:class:`Manifest`, JSON
Schema via ``sremu schema manifest``):

* ``sremu_version``, ``git_commit``, ``git_dirty``, ``dirty_patch_sha256``,
  ``contract_version``;
* ``versions`` of python, numpy, scipy, numba, pydantic, pyzmq and torch (null if absent);
* ``platform``, ``cpu_model``, ``logical_cpus``, ``power_plan_or_governor``, ``gpu``,
  ``aws_instance`` (``{type, id}`` or null);
* ``scenario_sha256`` (canonical JSON of the resolved scenario), ``inputs[{path, sha256,
  data_class}]``, ``assumptions[]``;
* ``seed``, ``rng_scheme`` (``"philox4x32-10/blake2b-v1"``);
* ``mode``, ``coupling``, ``backend``, ``threads``, ``deterministic``,
  ``numba_disable_svml``;
* per-stream ``page_size``; per-link decisions (``links``: narrowband, FD filter, T_u,
  tiles, error bounds);
* ``verify_interp_max_rel``, ``richardson_db``;
* ``counters``, ``nondeterministic_inputs``, ``oracle_csi_used``;
* ``argv``, ``t_start_utc``, ``t_end_utc``, ``wall_s``, ``host_load_samples``.

**Git.** ``git_commit`` is ``HEAD`` of the repository containing the sremu sources (or the
``repo_dir`` given). The tree is dirty when ``git status --porcelain`` lists anything
(modified, staged or untracked non-ignored files). ``dirty.patch`` is
``git diff HEAD --binary`` followed by a ``/dev/null`` diff for every untracked file, so
``git apply dirty.patch`` on a clean checkout of ``git_commit`` reproduces the tree;
``dirty_patch_sha256`` is the SHA-256 of its bytes. Git is only read, never written.
"""

from __future__ import annotations

import ctypes
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from sremu.contracts import CONTRACT_VERSION, COUNTER_NAMES, DataClass
from sremu.rng import RNG_SCHEME
from sremu.scenario.load import InputRecord, ResolvedScenario
from sremu.timebase import page_size

__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "PATCH_NAME",
    "RESOLVED_SCENARIO_NAME",
    "AwsInstance",
    "GitState",
    "HostLoadSample",
    "HostLoadSampler",
    "InputEntry",
    "LinkDecision",
    "Manifest",
    "ManifestRecorder",
    "create_run_dir",
    "environment_info",
    "git_state",
    "manifest_json_schema",
    "read_manifest",
    "utc_now_iso",
    "write_manifest",
]

MANIFEST_VERSION: Final[str] = "1.0"
MANIFEST_NAME: Final[str] = "manifest.json"
PATCH_NAME: Final[str] = "dirty.patch"
RESOLVED_SCENARIO_NAME: Final[str] = "scenario.resolved.yaml"
_VERSION_DISTS: Final[dict[str, str]] = {
    "numpy": "numpy",
    "scipy": "scipy",
    "numba": "numba",
    "pydantic": "pydantic",
    "pyzmq": "pyzmq",
    "torch": "torch",
}


# =========================================================================== model


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class InputEntry(_M):
    """An input data file: path, SHA-256 and data class."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_class: DataClass


class AwsInstance(_M):
    """EC2 instance type and id."""

    type: str | None
    id: str | None


class LinkDecision(_M):
    """Per-link numerical decisions (section 3.6). ``tiles``: panel id -> tile count;
    ``interp_error_bound``: the T_u/tile error bound (relative |dh|/|h|)."""

    narrowband: bool | None = None
    fd_filter: str | None = None
    t_u_s: str | None = None
    tiles: dict[str, int] = Field(default_factory=dict)
    interp_error_bound: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class HostLoadSample(_M):
    """Host load at a wall-clock instant: busy CPU fraction since the previous sample and,
    on POSIX, the 1-minute load average."""

    t_utc: str
    cpu_busy_fraction: float | None = None
    loadavg_1min: float | None = None


class Manifest(_M):
    """``manifest.json`` (section 3.6)."""

    manifest_version: Literal["1.0"] = "1.0"
    sremu_version: str
    git_commit: str | None
    git_dirty: bool | None
    dirty_patch_sha256: str | None
    contract_version: str
    versions: dict[str, str | None]
    platform: str
    cpu_model: str | None
    logical_cpus: int | None
    power_plan_or_governor: str | None
    gpu: list[str]
    aws_instance: AwsInstance | None
    scenario_name: str
    scenario_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inputs: list[InputEntry]
    assumptions: list[str]
    seed: int
    rng_scheme: str
    mode: str
    coupling: str
    backend: str
    threads: int | None
    deterministic: bool
    numba_disable_svml: bool | None
    page_size: dict[str, int]
    links: dict[str, LinkDecision]
    verify_interp_max_rel: float | None
    richardson_db: float | None
    counters: dict[str, int]
    nondeterministic_inputs: bool
    oracle_csi_used: bool
    argv: list[str]
    t_start_utc: str
    t_end_utc: str | None
    wall_s: float | None
    host_load_samples: list[HostLoadSample]


def manifest_json_schema() -> dict[str, Any]:
    """JSON Schema (draft 2020-12) of ``manifest.json``."""
    body = Manifest.model_json_schema(mode="validation")
    body.pop("title", None)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:sremu:schema:manifest:{MANIFEST_VERSION}",
        "title": "sremu run manifest",
        **body,
    }


# =========================================================================== git


@dataclass(frozen=True)
class GitState:
    """Git state of a source tree. ``commit``/``dirty`` are None when it is not a git work
    tree (or git is unavailable); ``patch`` holds the dirty patch bytes when dirty."""

    commit: str | None
    dirty: bool | None
    patch: bytes | None
    reason: str | None = None

    @property
    def patch_sha256(self) -> str | None:
        """SHA-256 hex of the patch bytes, or None."""
        return None if self.patch is None else hashlib.sha256(self.patch).hexdigest()


def _git(repo: Path, *args: str, ok: Iterable[int] = (0,)) -> subprocess.CompletedProcess[bytes]:
    exe = shutil.which("git")
    if exe is None:
        raise FileNotFoundError("git")
    cmd = [exe, "-C", str(repo), "-c", "core.quotepath=off", "-c", "color.ui=never", *args]
    res = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    if res.returncode not in tuple(ok):
        raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout, res.stderr)
    return res


def default_repo_dir() -> Path:
    """The directory containing the sremu sources (the repo root in a source checkout)."""
    return Path(__file__).resolve().parents[3]


def git_state(repo_dir: str | Path | None = None) -> GitState:
    """Commit, dirty flag and dirty patch of the work tree containing ``repo_dir``
    (default: the sremu sources). Read-only: runs ``rev-parse``, ``status``, ``diff`` and
    ``ls-files`` only."""
    repo = Path(repo_dir) if repo_dir is not None else default_repo_dir()
    try:
        top = _git(repo, "rev-parse", "--show-toplevel").stdout.decode().strip()
    except FileNotFoundError:
        return GitState(None, None, None, "git not found")
    except (subprocess.CalledProcessError, OSError, subprocess.SubprocessError):
        return GitState(None, None, None, f"{repo} is not inside a git work tree")
    root = Path(top)
    try:
        commit: str | None = _git(root, "rev-parse", "--verify", "HEAD").stdout.decode().strip()
    except subprocess.CalledProcessError:
        commit = None  # no commits yet
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all").stdout
    dirty = bool(status)
    if not dirty:
        return GitState(commit, False, None)
    parts: list[bytes] = []
    base = "HEAD" if commit is not None else "--cached"
    diff_args = ["diff", "--binary", "--no-color", "--no-ext-diff", "--full-index"]
    parts.append(_git(root, *diff_args, base).stdout)
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout
    for name in sorted(n for n in untracked.decode("utf-8").split("\0") if n):
        parts.append(
            _git(root, *diff_args, "--no-index", "--", "/dev/null", name, ok=(0, 1)).stdout
        )
    return GitState(commit, True, b"".join(parts))


# =========================================================================== environment


def _run(args: list[str], timeout_s: float = 10.0) -> str | None:
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _dist_version(dist: str) -> str | None:
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _cpu_model() -> str | None:
    if sys.platform == "win32":
        import winreg

        try:
            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or None


def _power_plan_or_governor() -> str | None:
    if sys.platform == "win32":
        out = _run(["powercfg", "/getactivescheme"])
        if out:
            m = re.search(r"\(([^)]*)\)\s*$", out)
            return m.group(1) if m else out
        return None
    try:
        return (
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        return None


def _gpus() -> list[str]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    out = _run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return [line.strip() for line in out.splitlines() if line.strip()] if out else []


def _aws_instance() -> AwsInstance | None:
    base = Path("/sys/devices/virtual/dmi/id")
    try:
        if "Amazon EC2" not in (base / "sys_vendor").read_text(encoding="utf-8"):
            return None
    except OSError:
        return None

    def read(name: str) -> str | None:
        try:
            return (base / name).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    return AwsInstance(type=read("product_name"), id=read("board_asset_tag"))


def _numba_svml_disabled() -> bool | None:
    nb = sys.modules.get("numba")
    if nb is not None:
        return bool(getattr(nb.config, "DISABLE_INTEL_SVML", False))
    env = os.environ.get("NUMBA_DISABLE_INTEL_SVML")
    return None if env is None else env not in ("", "0")


def _numba_threads() -> int | None:
    nb = sys.modules.get("numba")
    if nb is None:
        return None
    try:
        return int(nb.get_num_threads())
    except Exception:  # numba not fully initialised
        return None


def environment_info() -> dict[str, Any]:
    """Platform fields of the manifest (versions, CPU, power plan, GPU, AWS instance)."""
    from sremu import __version__

    versions: dict[str, str | None] = {"python": platform.python_version()}
    versions.update({k: _dist_version(d) for k, d in _VERSION_DISTS.items()})
    return {
        "sremu_version": __version__,
        "contract_version": CONTRACT_VERSION,
        "versions": versions,
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "power_plan_or_governor": _power_plan_or_governor(),
        "gpu": _gpus(),
        "aws_instance": _aws_instance(),
    }


class _FileTime(ctypes.Structure):
    _fields_ = (("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32))

    def value(self) -> int:
        return (self.hi << 32) | self.lo


class HostLoadSampler:
    """Busy-CPU fraction between successive samples (Windows ``GetSystemTimes``, Linux
    ``/proc/stat``) and the POSIX 1-minute load average."""

    def __init__(self) -> None:
        self._last: tuple[int, int] | None = self._times()

    @staticmethod
    def _times() -> tuple[int, int] | None:
        """(busy, total) CPU time counters, or None when unavailable."""
        if sys.platform == "win32":
            idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
            try:
                ok = ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined,unused-ignore]
                    ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
                )
            except (AttributeError, OSError):
                return None
            if not ok:
                return None
            total = kernel.value() + user.value()  # kernel time includes idle time
            return total - idle.value(), total
        try:
            fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        except (OSError, IndexError):
            return None
        vals = [int(x) for x in fields]
        idle_all = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        return total - idle_all, total

    def sample(self) -> HostLoadSample:
        """One sample; the busy fraction covers the time since the previous sample."""
        now = self._times()
        busy = None
        if now is not None and self._last is not None:
            d_total = now[1] - self._last[1]
            if d_total > 0:
                busy = min(1.0, max(0.0, (now[0] - self._last[0]) / d_total))
        self._last = now
        load = None
        if hasattr(os, "getloadavg"):
            try:
                load = float(os.getloadavg()[0])
            except OSError:
                load = None
        return HostLoadSample(t_utc=utc_now_iso(), cpu_busy_fraction=busy, loadavg_1min=load)


# =========================================================================== run directory


def utc_now_iso() -> str:
    """Current UTC time, ISO 8601 with microseconds and a ``Z`` suffix."""
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def create_run_dir(root: str | Path, name: str, t_utc: _dt.datetime | None = None) -> Path:
    """Create ``<root>/<UTC>_<name>/`` (``UTC`` = ``YYYYmmddTHHMMSSZ``); a numeric suffix
    ``-2``, ``-3``, ... is added if the directory already exists."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]*", name):
        raise ValueError(f"run name {name!r} is not a safe directory name")
    t = t_utc or _dt.datetime.now(_dt.UTC)
    stamp = t.astimezone(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root) / f"{stamp}_{name}"
    path, k = base, 1
    while True:
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            k += 1
            path = base.with_name(f"{base.name}-{k}")


def write_manifest(run_dir: str | Path, manifest: Manifest) -> Path:
    """Write ``manifest.json`` (2-space indent, keys in model order, UTF-8, LF)."""
    path = Path(run_dir) / MANIFEST_NAME
    text = json.dumps(
        manifest.model_dump(mode="json"), indent=2, ensure_ascii=False, allow_nan=False
    )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def read_manifest(path: str | Path) -> Manifest:
    """Read and validate a ``manifest.json`` (file or run directory)."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_NAME
    return Manifest.model_validate_json(p.read_text(encoding="utf-8"))


# =========================================================================== recorder


class ManifestRecorder:
    """Collects the manifest of one run.

    Construction captures the environment, the git state (writing ``dirty.patch`` into
    ``run_dir`` when dirty), ``scenario.resolved.yaml``, the start time and default per-RX
    page sizes. The engine then reports decisions and counters; :meth:`finish` stamps the
    end time and writes ``manifest.json``.
    """

    def __init__(
        self,
        resolved: ResolvedScenario,
        run_dir: str | Path,
        *,
        argv: Sequence[str] | None = None,
        repo_dir: str | Path | None = None,
        threads: int | None = None,
        environment: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        sc = resolved.scenario
        self._t0 = time.perf_counter()
        self._sampler = HostLoadSampler()
        env = dict(environment) if environment is not None else environment_info()
        gs = git_state(repo_dir)
        if gs.dirty and gs.patch is not None:
            (self.run_dir / PATCH_NAME).write_bytes(gs.patch)
        (self.run_dir / RESOLVED_SCENARIO_NAME).write_text(
            resolved.to_yaml(), encoding="utf-8", newline="\n"
        )
        pages = {
            sid: page_size(s.rate_sps, sc.numerics.page_target_s)
            for sid, (_n, s) in sc.streams().items()
            if s.direction == "rx"
        }
        self.git = gs
        self._data: dict[str, Any] = {
            **env,
            "git_commit": gs.commit,
            "git_dirty": gs.dirty,
            "dirty_patch_sha256": gs.patch_sha256,
            "scenario_name": sc.name,
            "scenario_sha256": resolved.sha256(),
            "inputs": [r.to_dict() for r in resolved.inputs],
            "assumptions": list(resolved.assumptions),
            "seed": sc.seed,
            "rng_scheme": RNG_SCHEME,
            "mode": sc.run.mode,
            "coupling": sc.run.coupling,
            "backend": sc.numerics.backend,
            "threads": threads if threads is not None else _numba_threads(),
            "deterministic": sc.numerics.deterministic,
            "numba_disable_svml": _numba_svml_disabled(),
            "page_size": pages,
            "links": {},
            "verify_interp_max_rel": None,
            "richardson_db": None,
            "counters": dict.fromkeys(COUNTER_NAMES, 0),
            "nondeterministic_inputs": False,
            "oracle_csi_used": False,
            "argv": list(sys.argv if argv is None else argv),
            "t_start_utc": utc_now_iso(),
            "t_end_utc": None,
            "wall_s": None,
            "host_load_samples": [],
        }
        Manifest.model_validate(self._data)  # fail early on a malformed environment

    # ---- updates reported while running
    def set_page_size(self, stream: str, size: int) -> None:
        """Record the page size actually used by an RX stream."""
        self._data["page_size"][stream] = int(size)

    def set_link(self, link: str, decision: LinkDecision | Mapping[str, Any]) -> None:
        """Record a link's narrowband/FD/T_u/tile decision."""
        d = decision if isinstance(decision, LinkDecision) else LinkDecision(**decision)
        self._data["links"][link] = d.model_dump(mode="json")

    def set_threads(self, threads: int | None) -> None:
        """Record the worker thread count."""
        self._data["threads"] = threads

    def update_verify_interp(self, max_rel: float) -> None:
        """Keep the maximum ``verify_interp`` relative error."""
        cur = self._data["verify_interp_max_rel"]
        self._data["verify_interp_max_rel"] = max_rel if cur is None else max(cur, max_rel)

    def set_richardson_db(self, value_db: float | None) -> None:
        """Record the Richardson estimate (``po_patch`` only)."""
        self._data["richardson_db"] = value_db

    def add_counters(self, counters: Mapping[str, int]) -> None:
        """Add to the I6 counters (new names are accepted)."""
        for k, v in counters.items():
            self._data["counters"][k] = self._data["counters"].get(k, 0) + int(v)

    def mark_nondeterministic(self) -> None:
        """Set ``nondeterministic_inputs`` (e.g. a late unregistered command, zero_fill)."""
        self._data["nondeterministic_inputs"] = True

    def mark_oracle_csi_used(self) -> None:
        """Set ``oracle_csi_used`` (a controller observed ``csi_oracle``)."""
        self._data["oracle_csi_used"] = True

    def add_input(self, record: InputRecord | Mapping[str, str]) -> None:
        """Add an input data file entry."""
        d = record.to_dict() if isinstance(record, InputRecord) else dict(record)
        if d not in self._data["inputs"]:
            self._data["inputs"].append(InputEntry.model_validate(d).model_dump())

    def add_assumption(self, text: str) -> None:
        """Add an assumption line (deduplicated)."""
        if text not in self._data["assumptions"]:
            self._data["assumptions"].append(text)

    def sample_host_load(self) -> HostLoadSample:
        """Append a host-load sample."""
        s = self._sampler.sample()
        self._data["host_load_samples"].append(s.model_dump())
        return s

    def manifest(self) -> Manifest:
        """The manifest as collected so far (validated)."""
        return Manifest.model_validate(self._data)

    def finish(self) -> Manifest:
        """Stamp ``t_end_utc`` and ``wall_s``, write ``manifest.json`` and return it."""
        self.sample_host_load()
        self._data["t_end_utc"] = utc_now_iso()
        self._data["wall_s"] = time.perf_counter() - self._t0
        m = self.manifest()
        write_manifest(self.run_dir, m)
        return m
