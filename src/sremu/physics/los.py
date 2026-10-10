"""Line-of-sight path and explicit visibility factors (DESIGN.md sections 5.2, 5.9; WP2).

``h_LOS = sqrt(G_T(r) G_R(-r)) * lambda/(4 pi d) * V`` with ``r`` the unit vector Tx -> Rx.
The propagation phase ``exp(-j k d)`` is applied by the renderer from the delay ``d/C0``,
so the path amplitude returned here EXCLUDES it (contract ``PathBundle.amp``).

``V`` is 1, or the product of explicit factors of the polygons the segment crosses: 0 for
an opaque occluder and ``10**(-L/20)`` for a partition with transmission loss ``L`` dB
(no phase change) [ENG]. Scenario names never imply blockage. Knife-edge ``F(nu)`` factors
come from WP3 (``physics/diffraction.py``) through the propagation mechanism hook.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from sremu.errors import ConfigurationError, GeometryError
from sremu.physics.antennas import Antenna, Isotropic
from sremu.physics.geometry import segments_hit_polygon
from sremu.units import C0

__all__ = [
    "Occluder",
    "friis_amplitude",
    "los_path",
    "los_visibility",
    "occluders_from_spec",
]


def friis_amplitude(d_m: float | np.ndarray, fc_hz: float) -> np.ndarray:
    """Free-space amplitude ``lambda/(4 pi d)`` (isotropic antennas, no phase)."""
    d = np.asarray(d_m, dtype=np.float64)
    if np.any(d <= 0.0):
        raise GeometryError("distance must be > 0 m")
    lam = C0 / float(fc_hz)
    return lam / (4.0 * math.pi * d)


@dataclass(frozen=True, eq=False)
class Occluder:
    """An explicit planar polygon on propagation paths (section 5.9).

    ``kind="opaque"`` sets ``V = 0``; ``kind="partition"`` multiplies by
    ``10**(-transmission_loss_db/20)`` per crossing.
    """

    id: str
    vertices_m: np.ndarray
    kind: Literal["opaque", "partition"] = "opaque"
    transmission_loss_db: float | None = None

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices_m, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
            raise ConfigurationError(f"occluder {self.id!r}: vertices_m must be (M>=3, 3)")
        object.__setattr__(self, "vertices_m", v)
        if self.kind not in ("opaque", "partition"):
            raise ConfigurationError(f"occluder {self.id!r}: unknown kind {self.kind!r}")
        if self.kind == "partition" and (
            self.transmission_loss_db is None or not self.transmission_loss_db >= 0.0
        ):
            raise ConfigurationError(f"partition {self.id!r} needs transmission_loss_db >= 0")

    @property
    def factor(self) -> float:
        """Amplitude factor applied per crossing."""
        if self.kind == "opaque":
            return 0.0
        assert self.transmission_loss_db is not None
        return 10.0 ** (-self.transmission_loss_db / 20.0)


def los_visibility(a: np.ndarray, b: np.ndarray, occluders: Iterable[Occluder] = ()) -> np.ndarray:
    """Visibility factor ``V`` (...,) of segments a -> b (..., 3) through ``occluders``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v = np.ones(np.broadcast_shapes(a.shape, b.shape)[:-1], dtype=np.float64)
    for occ in occluders:
        hit = segments_hit_polygon(a, b, occ.vertices_m)
        v = np.where(hit, v * occ.factor, v)
    return v


def los_path(
    p_tx: Sequence[float] | np.ndarray,
    p_rx: Sequence[float] | np.ndarray,
    fc_hz: float,
    ant_tx: Antenna | None = None,
    ant_rx: Antenna | None = None,
    visibility: complex | float = 1.0,
) -> tuple[complex, float]:
    """LOS amplitude (excluding the propagation phase) and absolute delay ``d/C0`` (s)."""
    a = np.asarray(p_tx, dtype=np.float64)
    b = np.asarray(p_rx, dtype=np.float64)
    dv = b - a
    d = math.sqrt(
        (float(dv[0]) * float(dv[0]) + float(dv[1]) * float(dv[1])) + float(dv[2]) * float(dv[2])
    )
    if not d > 0.0:
        raise GeometryError("TX and RX are co-located")
    r = dv / d
    gt = float((ant_tx or Isotropic()).gain(r))
    gr = float((ant_rx or Isotropic()).gain(-r))
    lam = C0 / float(fc_hz)
    amp = math.sqrt(gt * gr) * lam / (4.0 * math.pi * d) * complex(visibility)
    return complex(amp), d / C0


def occluders_from_spec(items: Iterable[Mapping[str, Any]] | None) -> tuple[Occluder, ...]:
    """Scenario ``environment.occluders`` (section 9.2) -> :class:`Occluder` tuple."""
    out = []
    for it in items or ():
        out.append(
            Occluder(
                id=str(it["id"]),
                vertices_m=np.asarray(it["vertices_m"], dtype=np.float64),
                kind=it.get("kind", "opaque"),
                transmission_loss_db=it.get("transmission_loss_db"),
            )
        )
    ids = [o.id for o in out]
    if len(set(ids)) != len(ids):
        raise ConfigurationError("occluder ids must be unique")
    return tuple(out)
