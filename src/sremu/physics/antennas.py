"""Antenna power patterns (DESIGN.md section 5.2; WP2).

Scalar co-polar only. ``gain(dirs)`` returns the linear power gain toward unit direction
vectors ``dirs`` (..., 3) that point FROM the antenna TOWARD the far end of the path:

* LOS: ``G_T(r) * G_R(-r)`` with ``r`` the unit vector Tx -> Rx, i.e. ``tx.gain(r)`` and
  ``rx.gain(-r)``.
* RIS cell q: ``tx.gain(unit(q - p_tx))`` and ``rx.gain(unit(q - p_rx))``.

Patterns:

* :class:`Isotropic`: G = 1.
* :class:`CosPower`: ``G = 2 (alpha + 1) cos(psi)**alpha`` for ``cos(psi) > 0`` (else 0),
  with ``psi`` the angle from ``boresight``; ``G_0 = 2 (alpha + 1)`` (Tang 2022 Eq. 2).
* :class:`Tabulated`: G(theta, phi) in dBi on a grid with provenance; ``theta`` from the
  boresight, ``phi`` the azimuth around it measured from ``ref_axis``; bilinear
  interpolation in dBi, periodic in ``phi``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sremu.errors import ConfigurationError, MissingProvenanceError
from sremu.physics.geometry import vec3

__all__ = [
    "Antenna",
    "CosPower",
    "Isotropic",
    "Tabulated",
    "antenna_from_spec",
    "cos_power_alpha_for_gain_dbi",
]


@runtime_checkable
class Antenna(Protocol):
    """A scalar power pattern."""

    def gain(self, dirs: np.ndarray) -> np.ndarray:
        """Linear power gain (...,) toward unit directions (..., 3)."""
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-ready description (for manifests)."""
        ...


@dataclass(frozen=True)
class Isotropic:
    """Isotropic radiator, G = 1 (0 dBi)."""

    def gain(self, dirs: np.ndarray) -> np.ndarray:
        return np.ones(np.asarray(dirs).shape[:-1], dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        return {"pattern": "isotropic"}


@dataclass(frozen=True, eq=False)
class CosPower:
    """``G = 2 (alpha+1) cos^alpha(psi)`` on the front hemisphere of ``boresight``, else 0."""

    alpha: float
    boresight: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        if not self.alpha >= 0.0:
            raise ConfigurationError(f"cos_power alpha must be >= 0, got {self.alpha}")
        b = vec3(self.boresight, "boresight")
        nb = float(np.linalg.norm(b))
        if nb == 0.0:
            raise ConfigurationError("boresight must be non-zero")
        object.__setattr__(self, "boresight", b / nb)

    @property
    def peak_gain(self) -> float:
        """G_0 = 2 (alpha + 1)."""
        return 2.0 * (self.alpha + 1.0)

    def gain(self, dirs: np.ndarray) -> np.ndarray:
        c = np.asarray(dirs, dtype=np.float64) @ self.boresight
        cp = np.clip(c, 0.0, None)
        return np.where(c > 0.0, self.peak_gain * cp**self.alpha, 0.0)

    def describe(self) -> dict[str, Any]:
        return {"pattern": "cos_power", "alpha": self.alpha, "boresight": self.boresight.tolist()}


_PROV_REQUIRED = ("source_kind", "description", "reference")


@dataclass(frozen=True, eq=False)
class Tabulated:
    """Tabulated pattern ``gain_dbi[T, P]`` on ``theta_deg[T]`` (0..180) x ``phi_deg[P]``.

    ``theta`` is the angle from ``boresight``; ``phi`` the azimuth around it from
    ``ref_axis`` (projected orthogonal to the boresight). Interpolation is bilinear in dBi,
    periodic in ``phi``; ``theta`` outside the grid holds the edge value.
    """

    theta_deg: np.ndarray
    phi_deg: np.ndarray
    gain_dbi: np.ndarray
    provenance: Mapping[str, Any]
    boresight: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    ref_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self) -> None:
        th = np.asarray(self.theta_deg, dtype=np.float64)
        ph = np.asarray(self.phi_deg, dtype=np.float64)
        g = np.asarray(self.gain_dbi, dtype=np.float64)
        if g.shape != (th.size, ph.size):
            raise ConfigurationError(f"gain_dbi shape {g.shape} != ({th.size}, {ph.size})")
        if np.any(np.diff(th) <= 0) or (ph.size > 1 and np.any(np.diff(ph) <= 0)):
            raise ConfigurationError("theta_deg and phi_deg must be strictly increasing")
        if not np.all(np.isfinite(g)):
            raise ConfigurationError("gain_dbi must be finite (use a large negative dBi for nulls)")
        missing = [k for k in _PROV_REQUIRED if k not in self.provenance]
        if missing:
            raise MissingProvenanceError(f"tabulated antenna provenance lacks {missing}")
        b = vec3(self.boresight, "boresight")
        b = b / float(np.linalg.norm(b))
        r = vec3(self.ref_axis, "ref_axis")
        r = r - (r @ b) * b
        if float(np.linalg.norm(r)) < 1e-9:
            raise ConfigurationError("ref_axis must not be parallel to the boresight")
        r = r / float(np.linalg.norm(r))
        for name, val in (
            ("theta_deg", th),
            ("phi_deg", ph),
            ("gain_dbi", g),
            ("boresight", b),
            ("ref_axis", r),
        ):
            object.__setattr__(self, name, val)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        provenance: Mapping[str, Any],
        boresight: np.ndarray | None = None,
        ref_axis: np.ndarray | None = None,
    ) -> Tabulated:
        """Long-form CSV ``theta_deg,phi_deg,gain_dbi`` covering a full grid."""
        data: Any = np.genfromtxt(Path(path), delimiter=",", names=True, dtype=np.float64)
        th_u = np.unique(data["theta_deg"])
        ph_u = np.unique(data["phi_deg"])
        g = np.full((th_u.size, ph_u.size), np.nan)
        it = np.searchsorted(th_u, data["theta_deg"])
        ip = np.searchsorted(ph_u, data["phi_deg"])
        g[it, ip] = data["gain_dbi"]
        if np.isnan(g).any():
            raise ConfigurationError(f"{path}: pattern grid is incomplete")
        kw: dict[str, Any] = {}
        if boresight is not None:
            kw["boresight"] = boresight
        if ref_axis is not None:
            kw["ref_axis"] = ref_axis
        return cls(th_u, ph_u, g, dict(provenance), **kw)

    def gain(self, dirs: np.ndarray) -> np.ndarray:
        d = np.asarray(dirs, dtype=np.float64)
        b, r = self.boresight, self.ref_axis
        s = np.cross(b, r)
        c = np.clip(d @ b, -1.0, 1.0)
        theta = np.degrees(np.arccos(c))
        phi = np.degrees(np.arctan2(d @ s, d @ r)) % 360.0
        th, ph, g = self.theta_deg, self.phi_deg, self.gain_dbi
        tq = np.clip(theta, th[0], th[-1])
        i = np.clip(np.searchsorted(th, tq, side="right") - 1, 0, max(th.size - 2, 0))
        if th.size > 1:
            wt = (tq - th[i]) / (th[i + 1] - th[i])
        else:
            wt = np.zeros_like(tq)
        if ph.size == 1:
            g0 = g[i, 0]
            g1 = g[np.minimum(i + 1, th.size - 1), 0]
            gdb = g0 + wt * (g1 - g0)
        else:
            phx = np.concatenate([ph, [ph[0] + 360.0]])
            gx = np.concatenate([g, g[:, :1]], axis=1)
            pq = np.where(phi < ph[0], phi + 360.0, phi)
            j = np.clip(np.searchsorted(phx, pq, side="right") - 1, 0, phx.size - 2)
            wp = (pq - phx[j]) / (phx[j + 1] - phx[j])
            i1 = np.minimum(i + 1, th.size - 1)
            g00, g01 = gx[i, j], gx[i, j + 1]
            g10, g11 = gx[i1, j], gx[i1, j + 1]
            gdb = (1 - wt) * ((1 - wp) * g00 + wp * g01) + wt * ((1 - wp) * g10 + wp * g11)
        return 10.0 ** (gdb / 10.0)

    def describe(self) -> dict[str, Any]:
        return {
            "pattern": "tabulated",
            "boresight": self.boresight.tolist(),
            "ref_axis": self.ref_axis.tolist(),
            "provenance": dict(self.provenance),
        }


def antenna_from_spec(spec: Mapping[str, Any] | None, *, base_dir: Path | None = None) -> Antenna:
    """Scenario form (section 9.2): ``{pattern: isotropic | {cos_power: {alpha}} |
    {tabulated: {file, provenance}}, boresight?: [x, y, z]}``."""
    if spec is None:
        return Isotropic()
    pattern = spec.get("pattern", "isotropic")
    boresight = spec.get("boresight")
    if pattern == "isotropic":
        return Isotropic()
    if isinstance(pattern, Mapping) and "cos_power" in pattern:
        alpha = float(pattern["cos_power"]["alpha"])
        if boresight is None:
            raise ConfigurationError("cos_power antennas need a boresight")
        return CosPower(alpha, np.asarray(boresight, dtype=np.float64))
    if isinstance(pattern, Mapping) and "tabulated" in pattern:
        tab = pattern["tabulated"]
        path = Path(tab["file"])
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        bs = None if boresight is None else np.asarray(boresight, dtype=np.float64)
        ref = tab.get("ref_axis")
        return Tabulated.from_csv(
            path, tab["provenance"], bs, None if ref is None else np.asarray(ref, dtype=np.float64)
        )
    raise ConfigurationError(f"unknown antenna pattern {pattern!r}")


def cos_power_alpha_for_gain_dbi(g0_dbi: float) -> float:
    """Inverse of G_0 = 2(alpha+1): the CosPower exponent giving peak gain ``g0_dbi``."""
    return 10.0 ** (g0_dbi / 10.0) / 2.0 - 1.0
