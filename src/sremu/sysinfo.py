"""Platform probes shared by ``sremu info`` and the run manifest (layer 0; D12).

Everything here reads the host: interpreter and package versions, CPU model, power plan
(Windows) or CPU governor (Linux), GPUs (``nvidia-smi``), the AWS instance (DMI), and the
numba threading / SVML / cache state. Only the standard library is imported at module level:
numba is imported lazily by :func:`numba_state`, so importing this module (and therefore
``sremu.cli``) never fixes numba's configuration before the CLI's deterministic switch
(D-4, D15).

Numba cache and SVML (D15): numba's on-disk cache does not separate kernels compiled with and
without Intel SVML, so deterministic mode (``NUMBA_DISABLE_INTEL_SVML=1``) uses its own cache
directory ``<default cache>-nosvml`` (:func:`nosvml_cache_dir`). The default cache is
``$NUMBA_CACHE_DIR`` when set, else numba's user-wide cache directory
(:func:`default_numba_cache_dir`).
"""

from __future__ import annotations

import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Final

__all__ = [
    "AWS_DMI_DIR",
    "INFO_DISTRIBUTIONS",
    "LINUX_GOVERNOR_FILE",
    "NOSVML_SUFFIX",
    "aws_instance",
    "cpu_model",
    "default_numba_cache_dir",
    "dist_version",
    "gpus",
    "nosvml_cache_dir",
    "numba_state",
    "numba_svml_disabled",
    "numba_threads",
    "parse_powercfg",
    "platform_info",
    "power_plan_or_governor",
    "read_governor",
    "run",
]

NOSVML_SUFFIX: Final[str] = "-nosvml"
"""Suffix of the deterministic-mode numba cache directory (D15)."""

AWS_DMI_DIR: Final[Path] = Path("/sys/devices/virtual/dmi/id")
"""Linux DMI directory read by :func:`aws_instance`."""

LINUX_GOVERNOR_FILE: Final[Path] = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
"""CPU frequency governor read by :func:`read_governor` (Linux)."""

INFO_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "numpy",
    "scipy",
    "numba",
    "llvmlite",
    "pydantic",
    "pyzmq",
    "typer",
    "PyYAML",
    "matplotlib",
    "torch",
)
"""Distributions whose versions ``sremu info`` reports."""


def run(args: list[str], timeout_s: float = 10.0) -> str | None:
    """Stripped stdout of a command, or None if it cannot run, fails or times out."""
    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def dist_version(dist: str) -> str | None:
    """Installed version of distribution ``dist``, or None."""
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def cpu_model() -> str | None:
    """CPU model string (Windows registry, ``/proc/cpuinfo`` or ``sysctl``), or None."""
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
        out = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out
    return platform.processor() or None


def parse_powercfg(out: str) -> str:
    """Plan name from ``powercfg /getactivescheme`` output (``... (Balanced)`` -> ``Balanced``)."""
    m = re.search(r"\(([^)]*)\)\s*$", out)
    return m.group(1) if m else out


def read_governor(governor_file: Path = LINUX_GOVERNOR_FILE) -> str | None:
    """CPU frequency governor from ``governor_file``, or None if it cannot be read."""
    try:
        return governor_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def power_plan_or_governor() -> str | None:
    """Active Windows power plan, or the CPU frequency governor elsewhere; None if unknown."""
    if sys.platform == "win32":
        out = run(["powercfg", "/getactivescheme"])
        return parse_powercfg(out) if out else None
    return read_governor()


def gpus() -> list[str]:
    """``name, driver, memory`` of every NVIDIA GPU (``nvidia-smi``); empty if none."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    out = run([exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return [line.strip() for line in out.splitlines() if line.strip()] if out else []


def aws_instance(dmi_dir: Path = AWS_DMI_DIR) -> dict[str, str | None] | None:
    """``{"type": <instance type>, "id": <instance id>}`` on Amazon EC2 (DMI), else None."""
    try:
        if "Amazon EC2" not in (dmi_dir / "sys_vendor").read_text(encoding="utf-8"):
            return None
    except OSError:
        return None

    def read(name: str) -> str | None:
        try:
            return (dmi_dir / name).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    return {"type": read("product_name"), "id": read("board_asset_tag")}


def platform_info() -> dict[str, Any]:
    """Host fields shared by ``sremu info`` and the manifest (section 3.6).

    Keys: ``platform``, ``cpu_model``, ``logical_cpus``, ``power_plan_or_governor``, ``gpu``
    (list of str) and ``aws_instance`` (``{type, id}`` or None).
    """
    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "power_plan_or_governor": power_plan_or_governor(),
        "gpu": gpus(),
        "aws_instance": aws_instance(),
    }


# --------------------------------------------------------------------------- numba


def default_numba_cache_dir(env: dict[str, str] | None = None) -> str:
    """The numba cache directory sremu treats as the default (D15), without importing numba.

    ``$NUMBA_CACHE_DIR`` when set and non-empty; otherwise numba's user-wide cache directory
    (numba ``AppDirs("numba", appauthor=False).user_cache_dir``): ``%LOCALAPPDATA%\\numba\\Cache``
    on Windows, ``~/Library/Caches/numba`` on macOS, ``$XDG_CACHE_HOME/numba`` (default
    ``~/.cache/numba``) elsewhere.
    """
    environ = os.environ if env is None else env
    preset = environ.get("NUMBA_CACHE_DIR", "")
    if preset:
        return preset
    if sys.platform == "win32":
        base = environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return os.path.join(os.path.normpath(base), "numba", "Cache")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/numba")
    base = environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "numba")


def nosvml_cache_dir(base: str | None = None) -> str:
    """Deterministic-mode numba cache directory ``<base>-nosvml`` (D15).

    ``base`` defaults to :func:`default_numba_cache_dir`. Idempotent: a directory that already
    ends in ``-nosvml`` is returned unchanged; trailing path separators are dropped first.
    """
    root = base if base else default_numba_cache_dir()
    stripped = root.rstrip("/\\") or root
    if stripped.endswith(NOSVML_SUFFIX):
        return stripped
    return stripped + NOSVML_SUFFIX


def numba_svml_disabled() -> bool | None:
    """Whether numba runs without SVML: numba's config if numba is imported, else the
    environment (None when neither says)."""
    nb = sys.modules.get("numba")
    if nb is not None:
        return bool(getattr(nb.config, "DISABLE_INTEL_SVML", False))
    env = os.environ.get("NUMBA_DISABLE_INTEL_SVML")
    return None if env is None else env not in ("", "0")


def numba_threads() -> int | None:
    """numba's current thread count if numba is imported, else None."""
    nb = sys.modules.get("numba")
    if nb is None:
        return None
    try:
        return int(nb.get_num_threads())
    except Exception:  # numba not fully initialised
        return None


def numba_state(probe_threads: bool = False) -> dict[str, Any]:
    """numba version, SVML, threading and cache state (imports numba).

    ``probe_threads`` compiles a tiny parallel kernel to report the active threading layer.
    """
    state: dict[str, Any] = {
        "env_NUMBA_DISABLE_INTEL_SVML": os.environ.get("NUMBA_DISABLE_INTEL_SVML"),
        "env_NUMBA_CACHE_DIR": os.environ.get("NUMBA_CACHE_DIR"),
    }
    try:
        import numba
    except ImportError as exc:
        state["error"] = f"numba not importable: {exc}"
        return state
    cfg = numba.config
    state["version"] = numba.__version__
    state["using_svml"] = bool(getattr(cfg, "USING_SVML", False))
    state["disable_intel_svml"] = bool(getattr(cfg, "DISABLE_INTEL_SVML", False))
    state["cache_dir"] = str(getattr(cfg, "CACHE_DIR", ""))
    state["threading_layer_config"] = str(getattr(cfg, "THREADING_LAYER", "default"))
    state["num_threads"] = int(getattr(cfg, "NUMBA_NUM_THREADS", 0))
    available = []
    for layer, mod in (("tbb", "tbbpool"), ("omp", "omppool"), ("workqueue", "workqueue")):
        try:
            importlib.import_module(f"numba.np.ufunc.{mod}")
            available.append(layer)
        except Exception:  # any failure means "not available"
            pass
    state["threading_layers_available"] = available
    if probe_threads:
        try:

            @numba.njit(parallel=True, cache=False)
            def _probe(n: int) -> float:
                s = 0.0
                for i in numba.prange(n):
                    s += i
                return s

            _probe(64)
            state["threading_layer_active"] = numba.threading_layer()
        except Exception as exc:
            state["threading_layer_active"] = f"probe failed: {exc}"
    return state
