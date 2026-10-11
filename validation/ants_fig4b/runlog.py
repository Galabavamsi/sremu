"""Run directories of the V-LAB study: ``manifest.json`` + data + ``meta.json``.

Every output directory is ``<root>/<UTC>_<name>/`` (:func:`sremu.record.manifest.
create_run_dir`) with

* ``manifest.json``: the sremu manifest model (section 3.6) built from
  :func:`sremu.record.manifest.environment_info` and :func:`~sremu.record.manifest.git_state`
  (commit, dirty flag and ``dirty.patch``), the protocol seed, RNG scheme, inputs with
  SHA-256 and data class, host-load samples and wall time;
* ``meta.json``: study metadata (protocol digest, lock state, prediction-spec hash, every
  output with SHA-256 and data class, the set of data classes) plus run-specific fields;
* the data files (CSV/JSON/NPZ).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from validation.ants_fig4b.protocol import (
    PROTOCOL_DIR,
    PROTOCOL_JSON,
    PROTOCOL_MD,
    prediction_spec_sha256,
)

from sremu import sysinfo
from sremu.record.manifest import (
    PATCH_NAME,
    HostLoadSampler,
    Manifest,
    create_run_dir,
    environment_info,
    git_state,
    utc_now_iso,
    write_manifest,
)
from sremu.rng import RNG_SCHEME

__all__ = ["ASSUMPTIONS", "RunRecorder", "background_load", "jsonable", "sha256_file"]

ASSUMPTIONS: tuple[str, ...] = (
    "V-LAB geometry prior of protocol.json v2: author-supplied lab facts (2026-10-11) with "
    "ranges; the exact lab geometry is not supplied (human gate H-1)",
    "Gamma magnitudes U[0.6, 1.0] and phase offsets U[-15, 15] deg around "
    "lab_binary_3p5ghz (ASSUMED)",
    "antennas: isotropic or vertical short dipole (1.5 sin^2, 40 dB null floor) for the "
    "VERT2450/VERT900 whips [ENG]",
    "floor polarization V (vertical whips); concrete from P.2040-3 (UNVERIFIED)",
    "element-wise near-field K1 summation; mutual coupling and the cells' reactive near field "
    "not modelled; nodes within one wavelength of a tile rejected",
)


def background_load(seconds: float = 5.0) -> float | None:
    """Busy CPU fraction of the whole host over ``seconds`` of idling by this process (the
    background load before a timed run)."""
    s = HostLoadSampler()
    time.sleep(seconds)
    return s.sample().cpu_busy_fraction


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(x: Any) -> Any:
    """JSON-ready values (numpy scalars/arrays; non-finite floats become None)."""
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, bool | np.bool_):
        return bool(x)
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, Path):
        return str(x)
    return x


class RunRecorder:
    """One V-LAB run directory."""

    def __init__(
        self,
        root: str | Path,
        name: str,
        protocol: Mapping[str, Any],
        *,
        argv: Sequence[str] | None = None,
        repo_dir: str | Path | None = None,
    ) -> None:
        self.run_dir = create_run_dir(root, name)
        self.name = name
        self.protocol = protocol
        self._t0 = time.perf_counter()
        self._t_start = utc_now_iso()
        self._sampler = HostLoadSampler()
        self._load: list[dict[str, Any]] = []
        self.env = environment_info()
        self.git = git_state(repo_dir)
        if self.git.dirty and self.git.patch is not None:
            (self.run_dir / PATCH_NAME).write_bytes(self.git.patch)
        self.argv = list(sys.argv if argv is None else argv)
        self.inputs: list[dict[str, str]] = []
        self.outputs: list[dict[str, str]] = []
        for fname in (PROTOCOL_MD, PROTOCOL_JSON):
            self.add_input(PROTOCOL_DIR / fname, "ASSUMED")

    # ---- bookkeeping
    def sample_load(self) -> None:
        """Append a host-load sample (busy CPU fraction since the previous sample)."""
        self._load.append(self._sampler.sample().model_dump())

    def add_input(self, path: str | Path, data_class: str) -> None:
        """Record an input file with its SHA-256 and data class."""
        p = Path(path)
        self.inputs.append({"path": str(p), "sha256": sha256_file(p), "data_class": data_class})

    def add_output(self, rel: str, data_class: str) -> None:
        """Record an output file (relative to the run directory)."""
        self.outputs.append(
            {"path": rel, "sha256": sha256_file(self.run_dir / rel), "data_class": data_class}
        )

    def write_json(self, rel: str, obj: Any) -> Path:
        """Write ``obj`` as indented JSON (LF)."""
        p = self.run_dir / rel
        p.write_text(
            json.dumps(jsonable(obj), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return p

    def write_csv(self, rel: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        """Write rows (dicts with the same keys) as CSV (LF)."""
        p = self.run_dir / rel
        with p.open("w", encoding="utf-8", newline="") as fh:
            if not rows:
                fh.write("")
                return p
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({k: _cell(v) for k, v in r.items()})
        return p

    # ---- finish
    def finish(self, *, backend: str, extra: Mapping[str, Any] | None = None) -> Manifest:
        """Write ``manifest.json`` and ``meta.json``."""
        from validation.ants_fig4b.protocol_lock import PREREGISTERED_SHA256, protocol_digest

        self.sample_load()
        data = {
            **self.env,
            "git_commit": self.git.commit,
            "git_dirty": self.git.dirty,
            "dirty_patch_sha256": self.git.patch_sha256,
            "scenario_name": f"vlab/{self.name}",
            "scenario_sha256": prediction_spec_sha256(self.protocol),
            "inputs": self.inputs,
            "assumptions": list(ASSUMPTIONS),
            "seed": int(self.protocol["seed"]),
            "rng_scheme": RNG_SCHEME,
            "mode": "validation-study",
            "coupling": "none",
            "backend": backend,
            "threads": sysinfo.numba_threads(),
            "deterministic": True,
            "numba_disable_svml": sysinfo.numba_svml_disabled(),
            "page_size": {},
            "links": {},
            "verify_interp_max_rel": None,
            "richardson_db": None,
            "counters": {},
            "nondeterministic_inputs": False,
            "oracle_csi_used": False,
            "argv": self.argv,
            "t_start_utc": self._t_start,
            "t_end_utc": utc_now_iso(),
            "wall_s": time.perf_counter() - self._t0,
            "host_load_samples": self._load,
        }
        man = Manifest.model_validate(data)
        write_manifest(self.run_dir, man)
        meta = {
            "study": "V-LAB",
            "run": self.name,
            "protocol_id": self.protocol["protocol_id"],
            "protocol_version": self.protocol["protocol_version"],
            "protocol_digest": protocol_digest(),
            "preregistered_sha256": PREREGISTERED_SHA256 or None,
            "locked": bool(PREREGISTERED_SHA256),
            "prediction_spec_sha256": prediction_spec_sha256(self.protocol),
            "git_commit": self.git.commit,
            "git_dirty": self.git.dirty,
            "outputs": self.outputs,
            "data_classes": sorted({o["data_class"] for o in self.outputs}),
            "wall_s": data["wall_s"],
            **dict(extra or {}),
        }
        self.write_json("meta.json", meta)
        return man


def _cell(v: Any) -> Any:
    if isinstance(v, float | np.floating):
        f = float(v)
        return repr(f) if math.isfinite(f) else ""
    if isinstance(v, np.integer):
        return int(v)
    return v
