"""Protocol loader, cell grid, splits and folds (DESIGN.md section 10.9).

``protocol.json`` holds every number of the V-LAB protocol; this module reads it and derives
the cell bookkeeping. Cells are ring-major: cell ``i = 6*ir + it`` with ``ir`` indexing
``data.r_m`` and ``it`` indexing ``data.theta_deg``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "PROTOCOL_DIR",
    "PROTOCOL_JSON",
    "PROTOCOL_MD",
    "CellLayout",
    "canonical_json_bytes",
    "cell_layout",
    "load_protocol",
    "prediction_spec",
    "prediction_spec_sha256",
    "prior_variants",
    "ring_split",
    "sensitivity_run",
    "split_masks",
    "variant_protocol",
]

PROTOCOL_DIR: Final[Path] = Path(__file__).resolve().parent
PROTOCOL_MD: Final[str] = "PROTOCOL.md"
PROTOCOL_JSON: Final[str] = "protocol.json"


def load_protocol(protocol_dir: str | Path | None = None) -> dict[str, Any]:
    """The parsed ``protocol.json`` of ``protocol_dir`` (default: this package)."""
    d = Path(protocol_dir) if protocol_dir is not None else PROTOCOL_DIR
    with (d / PROTOCOL_JSON).open("r", encoding="utf-8") as fh:
        out = json.load(fh)
    if not isinstance(out, dict) or out.get("schema") != "sremu.vlab.protocol/1":
        raise ValueError(
            f"{d / PROTOCOL_JSON}: not a V-LAB protocol (schema sremu.vlab.protocol/1)"
        )
    return out


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, no whitespace, UTF-8, no NaN (the lock's definition)."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def prediction_spec(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """The protocol subset that fixes the prediction tensor (cell grid, frame, prior,
    models, K, seed, RNG streams, sigma_pos settings; the variant id of an alternative-prior
    sensitivity run). The tensor cache records its hash; the power-analysis result added to
    ``protocol.json`` later does not change it."""
    out = {
        "seed": protocol["seed"],
        "rng": protocol["rng"],
        "grid": {k: protocol["data"][k] for k in ("fc_hz", "theta_deg", "r_m", "cell_order")},
        "frame": protocol["frame"],
        "prior": protocol["prior"],
        "models": protocol["models"],
        "K": protocol["primary_analysis"]["K"],
        "sigma_pos": protocol["uncertainty_budget"]["sigma_pos"],
    }
    if protocol.get("variant") is not None:
        out["variant"] = protocol["variant"]
    return out


def prediction_spec_sha256(protocol: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of :func:`prediction_spec`."""
    return hashlib.sha256(canonical_json_bytes(prediction_spec(protocol))).hexdigest()


def sensitivity_run(protocol: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """The ``sensitivity_runs`` entry with ``id == run_id`` (``KeyError`` if absent)."""
    for r in protocol.get("sensitivity_runs", []):
        if str(r["id"]) == run_id:
            return dict(r)
    raise KeyError(f"no sensitivity run {run_id!r}")


def prior_variants(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    """Ids of the sensitivity runs that use an alternative prior (``prior_overrides``)."""
    return tuple(
        str(r["id"]) for r in protocol.get("sensitivity_runs", []) if "prior_overrides" in r
    )


def variant_protocol(protocol: Mapping[str, Any], variant: str | None = None) -> dict[str, Any]:
    """The protocol of a labelled alternative-prior sensitivity run (PROTOCOL.md section 11).

    A deep copy of ``protocol`` in which every key of the run's ``prior_overrides`` replaces
    the same key of ``prior`` (it must exist there) and every entry of ``rng_names`` replaces
    the stream name of the same ``rng`` key; models, analysis, decisions and the power-analysis
    design are unchanged. ``power_analysis_result`` (which belongs to the primary prior) is
    dropped and ``variant`` records the run id, so the variant's prediction-spec hash differs
    from the primary one. ``variant`` None or ``"primary"`` returns a plain deep copy.
    """
    out: dict[str, Any] = json.loads(json.dumps(protocol))
    if variant in (None, "primary"):
        return out
    runs = {str(r["id"]): r for r in protocol.get("sensitivity_runs", [])}
    spec = runs.get(str(variant))
    if spec is None or "prior_overrides" not in spec:
        raise KeyError(
            f"{variant!r} is not an alternative-prior sensitivity run "
            f"(known: {list(prior_variants(protocol))})"
        )
    for key, val in spec["prior_overrides"].items():
        if key not in out["prior"]:
            raise KeyError(f"{variant}: prior_overrides key {key!r} is not a prior key")
        out["prior"][key] = json.loads(json.dumps(val))
    for key, name in spec.get("rng_names", {}).items():
        out["rng"][key]["name"] = str(name)
    out.pop("power_analysis_result", None)
    out["variant"] = str(variant)
    return out


@dataclass(frozen=True)
class CellLayout:
    """The 30 cells in protocol order: polar angle (deg), radius (m), ring and wedge index."""

    theta_deg: np.ndarray
    r_m: np.ndarray
    ring: np.ndarray
    wedge: np.ndarray
    rings_m: tuple[float, ...]
    thetas_deg: tuple[float, ...]

    @property
    def n(self) -> int:
        """Number of cells."""
        return int(self.theta_deg.size)

    def index(
        self, theta_deg: float, r_m: float, *, tol_theta: float = 1e-6, tol_r: float = 1e-6
    ) -> int:
        """Index of the cell at (theta, r); ``KeyError`` if none."""
        hit = np.flatnonzero(
            (np.abs(self.theta_deg - theta_deg) <= tol_theta) & (np.abs(self.r_m - r_m) <= tol_r)
        )
        if hit.size != 1:
            raise KeyError(f"no unique cell at theta={theta_deg} deg, r={r_m} m")
        return int(hit[0])


def cell_layout(protocol: Mapping[str, Any]) -> CellLayout:
    """Cells in ring-major order (``i = n_theta*ir + it``)."""
    th = [float(x) for x in protocol["data"]["theta_deg"]]
    rr = [float(x) for x in protocol["data"]["r_m"]]
    ir, it = np.meshgrid(np.arange(len(rr)), np.arange(len(th)), indexing="ij")
    ring = ir.ravel().astype(np.int64)
    wedge = it.ravel().astype(np.int64)
    out = CellLayout(
        theta_deg=np.asarray(th, dtype=np.float64)[wedge],
        r_m=np.asarray(rr, dtype=np.float64)[ring],
        ring=ring,
        wedge=wedge,
        rings_m=tuple(rr),
        thetas_deg=tuple(th),
    )
    if out.n != int(protocol["data"]["n_cells"]):
        raise ValueError("protocol cell count does not match theta x r")
    return out


def _ring_mask(layout: CellLayout, rings_m: list[float]) -> np.ndarray:
    idx = [layout.rings_m.index(float(r)) for r in rings_m]
    return np.isin(layout.ring, idx)


def ring_split(
    protocol: Mapping[str, Any],
    train_rings_m: list[float],
    test_rings_m: list[float],
    layout: CellLayout | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(train_mask, test_mask)`` of a ring split (disjoint; need not cover every cell)."""
    lay = layout or cell_layout(protocol)
    train = _ring_mask(lay, [float(r) for r in train_rings_m])
    test = _ring_mask(lay, [float(r) for r in test_rings_m])
    if np.any(train & test) or not train.any() or not test.any():
        raise ValueError("a ring split needs disjoint, non-empty train and test rings")
    return train, test


def split_masks(
    protocol: Mapping[str, Any], layout: CellLayout | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{name: (train_mask, test_mask)}``: ``primary``, ``loro_r<ring m>`` (5 folds) and
    ``lowo_t<theta deg>`` (6 folds)."""
    lay = layout or cell_layout(protocol)
    sp = protocol["split"]["primary"]
    train = _ring_mask(lay, sp["train_rings_m"])
    test = _ring_mask(lay, sp["test_rings_m"])
    if np.any(train & test) or not np.all(train | test):
        raise ValueError("primary split must partition the cells")
    out = {"primary": (train, test)}
    for ir, r in enumerate(lay.rings_m):
        t = lay.ring == ir
        out[f"loro_r{r:g}"] = (~t, t)
    for it, th in enumerate(lay.thetas_deg):
        t = lay.wedge == it
        out[f"lowo_t{th:g}"] = (~t, t)
    return out
