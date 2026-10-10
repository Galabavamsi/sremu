"""Glue between the MATLAB goldens and the sremu implementation, written ONLY against the
public contracts (``sremu.contracts``: PropagationModel, PanelConfig, LinkPaths, render and
noise) and the frozen scenario field names of DESIGN.md section 9.2 / state-library format of
section 9.4. Every place that depends on an API not frozen by the spec is isolated here so
that integration needs one edit at most.

Assumptions (also listed as contract change requests in the WP4 report):

* A1 ``PanelConfig.state`` is ordered ``n = iv*n_u + iu`` with ``iu`` along ``pose.u_axis``
  and ``iv`` along ``normal x u_axis`` (the ``FakePanel`` convention of WP0).
* A2 Physical tiles (``ris[].tiles``) add an inert ``gap_m`` between adjacent tiles; cells
  are centred on ``pose.center_m``.
* A3 A PropagationModel is built by ``sremu.physics.propagation.from_scenario(scenario)``
  where ``scenario`` is ``sremu.scenario.models.Scenario.model_validate(dict)``; if
  ``from_scenario`` rejects that object it is retried with the path of a YAML file.
* A4 ``antenna.boresight`` is a direction; every boresight used here also passes through the
  aimed point, so a "point" reading gives the same pattern.
* A5 The render backends expose module-level ``render(job, out)`` (and ``noise``) in
  ``sremu.channel.render_numpy`` / ``sremu.channel.render_numba``; ``noise`` may instead live
  in ``sremu.impair.noise``.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import FDSpec, LinkJob, PanelConfig, PathKind, SegmentJob

__all__ = [
    "BACKEND_MODULES",
    "AdapterError",
    "LinkCoefficients",
    "Node",
    "PanelSpec",
    "SimpleTxRing",
    "evaluate_links",
    "evaluate_with_state_files",
    "find_noise",
    "find_render",
    "link_id",
    "module_available",
    "propagation_available",
    "render_available",
    "render_cfo",
    "scenario_dict",
    "state_library",
]

BACKEND_MODULES = {
    "numpy_ref": "sremu.channel.render_numpy",
    "numba_cpu": "sremu.channel.render_numba",
}


class AdapterError(RuntimeError):
    """The implementation exists but does not match an assumption of this adapter."""


def module_available(name: str) -> bool:
    """True when ``name`` can be found without importing it (parents may be imported)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def propagation_available() -> bool:
    """WP1 scenario models and the WP2 PropagationModel factory module exist."""
    return module_available("sremu.physics.propagation") and module_available(
        "sremu.scenario.models"
    )


def render_available(backend: str) -> bool:
    """The WP5 render module of ``backend`` exists."""
    return module_available(BACKEND_MODULES[backend])


# --------------------------------------------------------------------------- scenario dicts


@dataclass(frozen=True)
class Node:
    """A static node with one stream. ``alpha`` None means an isotropic antenna."""

    id: str
    position_m: Sequence[float]
    direction: str  # "tx" | "rx"
    alpha: float | None = None
    boresight: Sequence[float] | None = None


@dataclass(frozen=True)
class PanelSpec:
    """One RIS panel (DESIGN.md section 9.2 field names)."""

    id: str
    center_m: Sequence[float]
    normal: Sequence[float]
    u_axis: Sequence[float]
    n_u: int
    n_v: int
    pitch_u_m: float
    pitch_v_m: float
    alphabet: Sequence[complex]
    n_tiles_u: int = 1
    n_tiles_v: int = 1
    gap_m: float = 0.0
    cell_model: str = "point"


def state_library(
    name: str, fc_hz: float, alphabet: Sequence[complex], validity_bw_hz: float = 1e6
) -> dict[str, Any]:
    """An angle-independent (separable) state library in the section 9.4 format."""
    return {
        "schema_version": "1.0",
        "kind": "ris_state_library",
        "name": name,
        "provenance": {
            "source_kind": "analytic",
            "description": "MATLAB golden alphabet (DESIGN.md section 10.6)",
            "reference": "tests/golden (WP4)",
            "reference_plane_offset_m": 0.0,
            "polarization": "co",
            "includes_cell_pattern": False,
            "sigma_mag_db": 0.0,
            "sigma_phase_deg": 0.0,
            "raw_sha256": None,
            "license": "n/a",
            "created": {
                "date": "2026-10-10",
                "tool": "tests/golden/sremu_adapter.py",
                "version": "1",
            },
            "assumptions": [],
        },
        "frequency_hz": [float(fc_hz)],
        "validity_bw_hz": float(validity_bw_hz),
        "angle_grid": "none",
        "reciprocal": True,
        "reconfiguration_time_s": None,
        "states": [
            {
                "id": s,
                "label": f"s{s}",
                "gamma": [
                    {
                        "f_hz": float(fc_hz),
                        "mag": float(abs(g)),
                        "phase_deg": float(np.degrees(np.angle(g))),
                    }
                ],
            }
            for s, g in enumerate(alphabet)
        ],
    }


def _antenna(n: Node) -> dict[str, Any]:
    if n.alpha is None:
        return {"pattern": "isotropic"}
    return {
        "pattern": {"cos_power": {"alpha": float(n.alpha)}},
        "boresight": [float(x) for x in n.boresight or ()],
    }


def scenario_dict(
    name: str,
    fc_hz: float,
    nodes: Sequence[Node],
    panels: Sequence[PanelSpec] = (),
    state_files: Mapping[str, Path] | None = None,
    ground: Mapping[str, Any] | None = None,
    rate_sps: str = "1e6",
) -> dict[str, Any]:
    """A section 9.2 scenario: static nodes, one stream each, optional panels and ground."""
    state_files = state_files or {}
    out: dict[str, Any] = {
        "schema_version": "1.0",
        "name": name,
        "seed": 20261010,
        "description": "WP4 MATLAB golden comparison scene (SIM)",
        "numerics": {"backend": "numpy_ref"},
        "environment": {"walls": [], "occluders": [], "edges": []},
        "nodes": [
            {
                "id": n.id,
                "position_m": [float(x) for x in n.position_m],
                "antenna": _antenna(n),
                "streams": [
                    {
                        "name": f"{n.direction}0",
                        "direction": n.direction,
                        "fc_hz": float(fc_hz),
                        "rate_sps": rate_sps,
                        **({"occupied_bw_hz": 1.0e4} if n.direction == "tx" else {}),
                    }
                ],
            }
            for n in nodes
        ],
        "ris": [],
        "controllers": [],
        "run": {
            "mode": "lockstep",
            "coupling": "barrier",
            "duration_s": "1e-3",
            "late_policy": "apply_on_arrival",
        },
        "record": {"iq": False, "events": False, "metrics": False, "snapshots": False},
    }
    if ground is not None:
        out["environment"]["ground"] = dict(ground)
    for p in panels:
        entry: dict[str, Any] = {
            "id": p.id,
            "enabled": True,
            "pose": {
                "center_m": [float(x) for x in p.center_m],
                "normal": [float(x) for x in p.normal],
                "u_axis": [float(x) for x in p.u_axis],
            },
            "grid": {
                "n_u": p.n_u,
                "n_v": p.n_v,
                "pitch_u_m": float(p.pitch_u_m),
                "pitch_v_m": float(p.pitch_v_m),
                "cell_model": p.cell_model,
            },
            "states": {"file": str(state_files[p.id])},
            "control": {"granularity": "element"},
            "initial": {"fill": 0},
            "transition": {"model": "step"},
            "capabilities": {},
        }
        if p.n_tiles_u > 1 or p.n_tiles_v > 1 or p.gap_m:
            entry["tiles"] = {
                "n_tiles_u": p.n_tiles_u,
                "n_tiles_v": p.n_tiles_v,
                "gap_m": float(p.gap_m),
            }
        out["ris"].append(entry)
    return out


# --------------------------------------------------------------------------- PropagationModel


@dataclass(frozen=True)
class LinkCoefficients:
    """Full complex-baseband coefficients of one link at the carrier, by path kind."""

    total: complex
    by_kind: Mapping[int, complex]

    def kind(self, k: PathKind) -> complex:
        return self.by_kind.get(int(k), 0j)


def _build_model(scn: dict[str, Any], workdir: Path) -> Any:
    import yaml

    models = importlib.import_module("sremu.scenario.models")
    prop = importlib.import_module("sremu.physics.propagation")
    factory = getattr(prop, "from_scenario", None)
    if factory is None:
        for obj in vars(prop).values():
            if (
                isinstance(obj, type)
                and obj.__module__ == prop.__name__
                and callable(getattr(obj, "from_scenario", None))
            ):
                factory = obj.from_scenario
                break
    if factory is None:
        raise AdapterError("sremu.physics.propagation has no from_scenario (assumption A3)")
    scenario_cls = getattr(models, "Scenario", None)
    path = workdir / f"{scn['name']}.yaml"
    path.write_text(yaml.safe_dump(scn, sort_keys=False), encoding="utf-8")
    errors: list[str] = []
    if scenario_cls is not None:
        try:
            return factory(scenario_cls.model_validate(scn))
        except Exception as exc:  # report and fall back to the file route
            errors.append(
                f"from_scenario(Scenario.model_validate(dict)): {type(exc).__name__}: {exc}"
            )
    for arg in (path, str(path)):
        try:
            return factory(arg)
        except Exception as exc:
            errors.append(f"from_scenario({type(arg).__name__}): {type(exc).__name__}: {exc}")
    raise AdapterError(
        "could not build a PropagationModel (assumption A3):\n  " + "\n  ".join(errors)
    )


def evaluate_links(
    scn: dict[str, Any],
    states: Mapping[str, NDArray[np.integer]],
    fc_hz: float,
    t: Fraction = Fraction(0),
) -> dict[str, LinkCoefficients]:
    """Evaluate every link at time ``t``: snapshot -> combine -> sum amp*exp(-j2 pi fc tau).

    ``states``: per panel id, the (N,) state of every physical cell (order of A1). Uses
    only ``PropagationModel.snapshot/combine`` and ``LinkPaths`` from ``sremu.contracts``.
    """
    with tempfile.TemporaryDirectory(prefix="sremu_golden_") as tmp:
        model = _build_model(scn, Path(tmp))
        snap = model.snapshot(t)
        cfg = {
            pid: PanelConfig(pid, np.ascontiguousarray(s, dtype=np.int16))
            for pid, s in states.items()
        }
        paths = model.combine(snap, cfg)
    out: dict[str, LinkCoefficients] = {}
    for link, lp in paths.items():
        ph = np.exp(-2j * math.pi * fc_hz * np.asarray(lp.delay_s, dtype=np.float64))
        terms = np.asarray(lp.amp, dtype=np.complex128) * ph
        kinds = np.asarray(lp.kind)
        by_kind = {int(k): complex(np.sum(terms[kinds == k])) for k in np.unique(kinds)}
        out[str(link)] = LinkCoefficients(complex(np.sum(terms)), by_kind)
    return out


def evaluate_with_state_files(
    name: str,
    fc_hz: float,
    nodes: Sequence[Node],
    panels: Sequence[PanelSpec],
    states: Mapping[str, NDArray[np.integer]],
    ground: Mapping[str, Any] | None = None,
) -> dict[str, LinkCoefficients]:
    """Write one state-library YAML per panel (section 9.4), build the scenario, evaluate."""
    import yaml

    with tempfile.TemporaryDirectory(prefix="sremu_states_") as tmp:
        files = {}
        for p in panels:
            f = Path(tmp) / f"{name}_{p.id}_states.yaml"
            f.write_text(
                yaml.safe_dump(state_library(f"{name}_{p.id}", fc_hz, p.alphabet), sort_keys=False),
                "utf-8",
            )
            files[p.id] = f
        scn = scenario_dict(name, fc_hz, nodes, panels, files, ground)
        return evaluate_links(scn, states, fc_hz)


def link_id(tx: str, rx: str) -> str:
    """``<tx>.tx0-><rx>.rx0`` for the single-stream nodes built here."""
    return f"{tx}.tx0->{rx}.rx0"


# --------------------------------------------------------------------------- render / noise


class SimpleTxRing:
    """Minimal :class:`sremu.contracts.TxRing` holding a finite complex64 waveform from k = 0."""

    def __init__(
        self, stream: str, f_s: Fraction, samples: NDArray[np.complex64], margin: int = 4096
    ) -> None:
        n = int(samples.shape[0])
        cap = 1 << math.ceil(math.log2(n + 2 * margin))
        self.stream = stream
        self.f_s = Fraction(f_s)
        self.capacity = cap
        self.mask = cap - 1
        self.buffer = np.zeros(cap, dtype=np.complex64)
        self.buffer[:n] = samples.astype(np.complex64)
        self._n = n

    def committed_through(self) -> int:
        return self._n

    def oldest(self) -> int:
        return 0

    def write(self, k0: int, samples: np.ndarray) -> None:
        idx = (np.arange(k0, k0 + samples.shape[0]) & self.mask).astype(np.int64)
        self.buffer[idx] = samples.astype(np.complex64)
        self._n = max(self._n, k0 + int(samples.shape[0]))

    def read(self, k0: int, n: int, out: np.ndarray | None = None) -> np.ndarray:
        k = np.arange(k0, k0 + n)
        vals = np.where((k >= 0) & (k < self._n), self.buffer[k & self.mask], 0).astype(
            np.complex64
        )
        if out is not None:
            out[:] = vals
            return out
        return vals


def find_render(backend: str) -> Callable[[SegmentJob, np.ndarray], None]:
    """The module-level ``render(job, out)`` of a backend (assumption A5)."""
    mod = importlib.import_module(BACKEND_MODULES[backend])
    fn = getattr(mod, "render", None)
    if not callable(fn):
        raise AdapterError(
            f"{BACKEND_MODULES[backend]} has no module-level render(job, out) (assumption A5)"
        )
    return fn


def find_noise(backend: str) -> Callable[[tuple[int, int], int, int, float], np.ndarray]:
    """``noise(rx_key, k0, n, sigma)`` from the backend module or ``sremu.impair.noise``."""
    for name in (BACKEND_MODULES[backend], "sremu.impair.noise"):
        if module_available(name):
            fn = getattr(importlib.import_module(name), "noise", None)
            if callable(fn):
                return fn
    raise AdapterError("no noise(rx_key, k0, n, sigma) found (assumption A5)")


def render_cfo(
    backend: str, x: NDArray[np.complex64], f_s: int, lo_offset_hz: float, chunks: Sequence[int]
) -> NDArray[np.complex128]:
    """Render one zero-delay unit path with ``lo_offset_hz`` over [0, len(x)) in ``chunks``.

    The same LinkJob is reused across consecutive segments so a backend can carry
    ``rot_state``; returns the complex128 accumulator (before any complex64 cast).
    """
    render = find_render(backend)
    n = int(x.shape[0])
    fs = Fraction(f_s)
    ring = SimpleTxRing("tx.tx0", fs, x)
    job_link = LinkJob(
        link="tx.tx0->rx.rx0",
        ring=ring,
        fc_hz=3.5e9,
        lo_offset_hz=float(lo_offset_hz),
        amp_a=np.ones(1, dtype=np.complex128),
        amp_b=np.ones(1, dtype=np.complex128),
        tau_a=np.zeros(1, dtype=np.float64),
        tau_b=np.zeros(1, dtype=np.float64),
        group=np.zeros(1, dtype=np.int16),
        fd=FDSpec.integer(),
        trans=[],
        rot_state=None,
    )
    out = np.zeros(n, dtype=np.complex128)
    k0 = 0
    i = 0
    while k0 < n:
        k1 = min(n, k0 + int(chunks[i % len(chunks)]))
        seg = SegmentJob(
            rx="rx.rx0",
            f_s=fs,
            k0=k0,
            k1=k1,
            t_a=Fraction(0),
            t_b=Fraction(n) / fs,
            links=[job_link],
        )
        buf = np.zeros(k1 - k0, dtype=np.complex128)
        render(seg, buf)
        out[k0:k1] = buf
        k0 = k1
        i += 1
    return out
