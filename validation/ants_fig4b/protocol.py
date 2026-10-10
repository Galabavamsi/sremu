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
    "split_masks",
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
    models, K, seed, RNG streams, sigma_pos settings). The tensor cache records its hash;
    the power-analysis result added to ``protocol.json`` later does not change it."""
    return {
        "seed": protocol["seed"],
        "rng": protocol["rng"],
        "grid": {k: protocol["data"][k] for k in ("fc_hz", "theta_deg", "r_m", "cell_order")},
        "frame": protocol["frame"],
        "prior": protocol["prior"],
        "models": protocol["models"],
        "K": protocol["primary_analysis"]["K"],
        "sigma_pos": protocol["uncertainty_budget"]["sigma_pos"],
    }


def prediction_spec_sha256(protocol: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of :func:`prediction_spec`."""
    return hashlib.sha256(canonical_json_bytes(prediction_spec(protocol))).hexdigest()


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
