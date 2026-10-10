"""Per-state RIS response tables (DESIGN.md section 5.5; WP2).

``ResponseTable(labels[S], freqs_hz[F], theta_deg[T], gamma[S, F, T, T], provenance,
validity_bw_hz)``: one theta grid is shared by the incidence and reflection axes (table angle
convention of section 3.1). Degenerate shapes (S, 1, 1, 1) and (S, F, 1, 1) are the
angle-independent (separable) tables.

Load-time validators (raise the ``sremu.errors`` table errors):

* passivity: ``|Gamma| <= 1 + 1e-9`` (analytic, fullwave, circuit, assumed) or
  ``<= 1 + 3 sigma_mag`` with ``sigma_mag = (ln10/20) sigma_mag_db`` (measured, digitized);
* reciprocity (angle tables): ``max|Gamma[s,f,i,j] - Gamma[s,f,j,i]| <= tol`` with
  ``tol = 1e-9`` when every sigma is 0, else ``3 sqrt(2) sigma_Gamma``,
  ``sigma_Gamma = |Gamma| sqrt((ln10/20 sigma_mag_db)^2 + (pi/180 sigma_phase_deg)^2)``;
  ``symmetrize="mean"`` replaces a failing table by ``(Gamma + Gamma^T)/2`` and records it;
* grid density: adjacent ``|dGamma| / max|Gamma| <= 0.35`` along every interpolated axis;
* domain: queries outside the grid raise ``OutOfDomain`` unless ``extrapolate="nearest"``
  was requested explicitly (recorded);
* bandwidth: :meth:`ResponseTable.check_bandwidth` raises ``BandwidthExceeded``;
* provenance: every mandatory field must be present (``MissingProvenanceError``).

Interpolation is complex-linear in (theta_i, theta_s) and linear in f: never magnitude and
unwrapped phase, so ``combine`` stays linear in Gamma.

Incidence-only tables ``g(theta)`` enter only through the reciprocal half-difference
extension ``Gamma(theta_i, theta_s) := g(|theta_i - theta_s|/2)`` (``angle_mapping:
half_difference``, recorded); raw non-reciprocal tables are rejected in release 1.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import yaml

from sremu.contracts import SourceKind
from sremu.errors import (
    BandwidthExceeded,
    ConfigurationError,
    GridTooCoarse,
    MissingProvenanceError,
    OutOfDomain,
    PassivityError,
    ReciprocityError,
    TableValidationError,
)
from sremu.physics.nonideal import efficiency_factor
from sremu.timebase import fraction_to_str, parse_exact

__all__ = [
    "BUILTIN_DIR",
    "DENSITY_MAX",
    "Provenance",
    "ResponseTable",
    "abeywickrama_table",
    "builtin",
    "builtin_names",
    "ideal_bbit_table",
    "load_table",
]

BUILTIN_DIR = Path(__file__).resolve().parent / "tables"
"""Directory of the packaged built-in state libraries (YAML, section 9.4 format)."""

DENSITY_MAX = 0.35
"""Grid-density limit on adjacent ``|dGamma|/max|Gamma|`` (about 20 deg at unit amplitude)."""

_SOURCE_KINDS = ("measured", "fullwave", "circuit", "analytic", "digitized", "assumed")
_UNCERTAIN_KINDS = ("measured", "digitized")
_LN10_20 = math.log(10.0) / 20.0
_PASSIVE_TOL = 1e-9
_RECIP_TOL0 = 1e-9


# --------------------------------------------------------------------------- provenance


@dataclass(frozen=True, eq=False)
class Provenance:
    """Mandatory provenance of a response table (section 5.5).

    ``sigma_mag_db`` and ``sigma_phase_deg`` are scalars or arrays broadcastable to the
    table's ``gamma`` shape (S, F, T, T). ``created`` holds ``{date, tool, version}``.
    ``assumptions`` collects every transformation applied after loading (symmetrize,
    assume_mag, efficiency, half-difference extension, extrapolation, sampling).
    """

    source_kind: SourceKind
    description: str
    reference: str
    reference_plane_offset_m: float
    polarization: str
    includes_cell_pattern: bool
    sigma_mag_db: float | np.ndarray
    sigma_phase_deg: float | np.ndarray
    raw_sha256: str | None
    license: str
    created: Mapping[str, Any]
    assumptions: tuple[str, ...] = ()

    MANDATORY: ClassVar[tuple[str, ...]] = (
        "source_kind",
        "description",
        "reference",
        "reference_plane_offset_m",
        "polarization",
        "includes_cell_pattern",
        "sigma_mag_db",
        "sigma_phase_deg",
        "raw_sha256",
        "license",
        "created",
    )

    def __post_init__(self) -> None:
        if self.source_kind not in _SOURCE_KINDS:
            raise MissingProvenanceError(
                f"source_kind must be one of {_SOURCE_KINDS}, got {self.source_kind!r}"
            )
        for name in ("sigma_mag_db", "sigma_phase_deg"):
            v = getattr(self, name)
            arr = np.asarray(v, dtype=np.float64)
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                raise TableValidationError(f"provenance {name} must be finite and >= 0")
            object.__setattr__(self, name, float(arr) if arr.ndim == 0 else arr)
        object.__setattr__(self, "includes_cell_pattern", bool(self.includes_cell_pattern))
        object.__setattr__(self, "reference_plane_offset_m", float(self.reference_plane_offset_m))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if not isinstance(self.created, Mapping):
            raise MissingProvenanceError("provenance 'created' must be a mapping {date, tool, ...}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Provenance:
        """Build from a mapping; every mandatory field must be present."""
        names = cls.MANDATORY
        missing = [k for k in names if k not in d]
        if missing:
            raise MissingProvenanceError(f"provenance lacks mandatory fields {missing}")
        kw = {k: d[k] for k in names}
        if d.get("sigma_mag_db") is None or d.get("sigma_phase_deg") is None:
            raise MissingProvenanceError("sigma_mag_db and sigma_phase_deg must not be null")
        kw["assumptions"] = tuple(d.get("assumptions", ()) or ())
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-ready dict (array sigmas as nested lists)."""
        out: dict[str, Any] = {}
        for k in self.MANDATORY:
            v = getattr(self, k)
            out[k] = (
                v.tolist()
                if isinstance(v, np.ndarray)
                else (dict(v) if isinstance(v, Mapping) else v)
            )
        out["assumptions"] = list(self.assumptions)
        return out

    def with_assumption(self, text: str) -> Provenance:
        """Copy with ``text`` appended to ``assumptions``."""
        return replace(self, assumptions=(*self.assumptions, text))

    @property
    def sigma_is_zero(self) -> bool:
        """True when both sigmas are identically 0."""
        return not (
            np.any(np.asarray(self.sigma_mag_db)) or np.any(np.asarray(self.sigma_phase_deg))
        )


# --------------------------------------------------------------------------- table


AngleMapping = Literal["full", "none", "half_difference"]


def _axis_index(grid: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left index i (clipped to [0, n-2]) and weight w in [0, 1] for linear interpolation."""
    i = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2)
    w = (x - grid[i]) / (grid[i + 1] - grid[i])
    return i, w


class ResponseTable:
    """Per-state complex reflection coefficients Gamma(s, f, theta_i, theta_s).

    Parameters: ``labels`` (S,), ``freqs_hz`` (F,) increasing (ignored when
    ``frequency_independent``), ``theta_deg`` (T,) increasing in [-90, 90] (any value when
    T = 1), ``gamma`` (S, F, T, T) complex, ``provenance``. ``validity_bw_hz``: declared
    validity bandwidth (None: unbounded, e.g. analytic tables). ``extrapolate``: ``"error"``
    (default) or ``"nearest"`` (explicit; recorded). ``symmetrize``: None or ``"mean"``.
    ``angle_mapping``: ``"none"`` (T = 1), ``"full"`` or ``"half_difference"`` (built by
    :meth:`from_incidence_only`). ``reconfiguration_time_s``: GS RIS 008 parameter or None.
    """

    def __init__(
        self,
        labels: Sequence[str],
        freqs_hz: Sequence[float] | np.ndarray,
        theta_deg: Sequence[float] | np.ndarray,
        gamma: np.ndarray,
        provenance: Provenance,
        *,
        validity_bw_hz: float | None = None,
        name: str = "",
        frequency_independent: bool = False,
        extrapolate: Literal["error", "nearest"] = "error",
        symmetrize: Literal["mean"] | None = None,
        angle_mapping: AngleMapping | None = None,
        reconfiguration_time_s: Fraction | None = None,
        validate: bool = True,
    ) -> None:
        g = np.array(gamma, dtype=np.complex128)  # copy
        if g.ndim != 4 or g.shape[2] != g.shape[3]:
            raise TableValidationError(f"gamma must be (S, F, T, T), got shape {g.shape}")
        s_count, f_count, t_count, _ = g.shape
        self.labels = tuple(str(x) for x in labels)
        if len(self.labels) != s_count or s_count < 1:
            raise TableValidationError(f"{len(self.labels)} labels for {s_count} states")
        if len(set(self.labels)) != s_count:
            raise TableValidationError("state labels must be unique")
        f = np.asarray(freqs_hz, dtype=np.float64).reshape(-1)
        th = np.asarray(theta_deg, dtype=np.float64).reshape(-1)
        if f.size != f_count or th.size != t_count:
            raise TableValidationError("freqs_hz / theta_deg sizes do not match gamma")
        if not frequency_independent and (np.any(f <= 0) or np.any(np.diff(f) <= 0)):
            raise TableValidationError("freqs_hz must be > 0 and strictly increasing")
        if t_count > 1 and (np.any(np.diff(th) <= 0) or th[0] < -90 or th[-1] > 90):
            raise TableValidationError("theta_deg must be strictly increasing within [-90, 90]")
        if not np.all(np.isfinite(g)):
            raise TableValidationError("gamma must be finite")
        if extrapolate not in ("error", "nearest"):
            raise ConfigurationError(
                f"extrapolate must be 'error' or 'nearest', got {extrapolate!r}"
            )
        if validity_bw_hz is not None and not validity_bw_hz > 0:
            raise TableValidationError("validity_bw_hz must be > 0 (or null)")
        if not isinstance(provenance, Provenance):
            raise MissingProvenanceError("a ResponseTable needs a Provenance")
        self.freqs_hz = f
        self.theta_deg = th if t_count > 1 else np.zeros(1)
        self.provenance = provenance
        self.validity_bw_hz = None if validity_bw_hz is None else float(validity_bw_hz)
        self.name = name
        self.frequency_independent = bool(frequency_independent)
        self.extrapolate = extrapolate
        self.reconfiguration_time_s = reconfiguration_time_s
        self.angle_mapping: AngleMapping = angle_mapping or ("none" if t_count == 1 else "full")
        if extrapolate == "nearest" and not any("extrapolate" in a for a in provenance.assumptions):
            self.provenance = provenance.with_assumption("extrapolate='nearest' (explicit)")
        self.symmetrized = False
        self.gamma = g
        if validate:
            self._validate(symmetrize)
        self.gamma.setflags(write=False)

    # ------------------------------------------------------------------ validators
    def _sigma_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        sm = np.broadcast_to(np.asarray(self.provenance.sigma_mag_db, np.float64), self.gamma.shape)
        sp = np.broadcast_to(
            np.asarray(self.provenance.sigma_phase_deg, np.float64), self.gamma.shape
        )
        return sm, sp

    def _validate(self, symmetrize: str | None) -> None:
        self.check_passivity()
        if self.n_theta > 1:
            err, tol = self.reciprocity_error()
            if np.any(err > tol):
                if symmetrize == "mean":
                    g = 0.5 * (self.gamma + np.swapaxes(self.gamma, 2, 3))
                    self.gamma = g
                    self.symmetrized = True
                    self.provenance = self.provenance.with_assumption(
                        f"symmetrize='mean' applied (max reciprocity error "
                        f"{float(np.max(err)):.3g} > tolerance)"
                    )
                    self.check_passivity()
                else:
                    i = np.unravel_index(int(np.argmax(err - tol)), err.shape)
                    raise ReciprocityError(
                        f"table {self.name!r}: |Gamma(theta_i, theta_s) - Gamma(theta_s, theta_i)|"
                        f" = {float(err[i]):.3g} > tol {float(tol[i]):.3g} at state "
                        f"{self.labels[i[0]]!r}, f index {i[1]}, theta "
                        f"({self.theta_deg[i[2]]}, {self.theta_deg[i[3]]}) deg; use "
                        "symmetrize='mean' to accept (logged)"
                    )
            elif symmetrize is not None and symmetrize != "mean":
                raise ConfigurationError(f"unknown symmetrize option {symmetrize!r}")
        self.check_grid_density()

    def check_passivity(self) -> None:
        """Raise ``PassivityError`` when ``|Gamma|`` exceeds the passivity bound."""
        mag = np.abs(self.gamma)
        if self.provenance.source_kind in _UNCERTAIN_KINDS:
            sm, _ = self._sigma_arrays()
            bound = 1.0 + 3.0 * _LN10_20 * sm
        else:
            bound = np.full(mag.shape, 1.0 + _PASSIVE_TOL)
        bad = mag > bound
        if np.any(bad):
            i = np.unravel_index(int(np.argmax(mag - bound)), mag.shape)
            raise PassivityError(
                f"table {self.name!r}: |Gamma| = {float(mag[i]):.6g} > {float(bound[i]):.6g} "
                f"(state {self.labels[i[0]]!r}, source_kind {self.provenance.source_kind})"
            )

    def reciprocity_error(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-entry ``|Gamma - Gamma^T|`` and its tolerance (both (S, F, T, T))."""
        g = self.gamma
        gt = np.swapaxes(g, 2, 3)
        err = np.abs(g - gt)
        if self.provenance.sigma_is_zero:
            return err, np.full(err.shape, _RECIP_TOL0)
        sm, sp = self._sigma_arrays()
        smx = np.maximum(sm, np.swapaxes(sm, 2, 3))
        spx = np.maximum(sp, np.swapaxes(sp, 2, 3))
        mag = np.maximum(np.abs(g), np.abs(gt))
        sig = mag * np.sqrt((_LN10_20 * smx) ** 2 + (math.pi / 180.0 * spx) ** 2)
        return err, np.maximum(3.0 * math.sqrt(2.0) * sig, _RECIP_TOL0)

    def grid_density(self) -> float:
        """Largest adjacent ``|dGamma|/max|Gamma|`` over the interpolated axes (per state)."""
        g = self.gamma
        mx = np.max(np.abs(g).reshape(g.shape[0], -1), axis=1)
        mx = np.where(mx > 0, mx, 1.0)[:, None, None, None]
        worst = 0.0
        axes = [2, 3] if self.n_theta > 1 else []
        if self.n_freqs > 1 and not self.frequency_independent:
            axes.append(1)
        for ax in axes:
            d = np.abs(np.diff(g, axis=ax))
            mxx = np.broadcast_to(mx, d.shape)
            worst = max(worst, float(np.max(d / mxx)))
        return worst

    def check_grid_density(self) -> None:
        """Raise ``GridTooCoarse`` above :data:`DENSITY_MAX`."""
        dens = self.grid_density()
        if dens > DENSITY_MAX:
            raise GridTooCoarse(
                f"table {self.name!r}: adjacent |dGamma|/max|Gamma| = {dens:.3f} > {DENSITY_MAX}"
                " (complex-linear interpolation would dip too much); refine the grid"
            )

    def check_bandwidth(self, occupied_bw_hz: float, stream: str = "") -> None:
        """Raise ``BandwidthExceeded`` if a linked TX band exceeds ``validity_bw_hz``."""
        if self.validity_bw_hz is not None and occupied_bw_hz > self.validity_bw_hz:
            raise BandwidthExceeded(
                f"table {self.name!r}: occupied band {occupied_bw_hz:.6g} Hz of {stream or 'a TX'}"
                f" exceeds validity_bw_hz {self.validity_bw_hz:.6g} Hz (release 1 has no Gamma FIR)"
            )

    # ------------------------------------------------------------------ properties
    @property
    def n_states(self) -> int:
        """S."""
        return int(self.gamma.shape[0])

    @property
    def n_freqs(self) -> int:
        """F."""
        return int(self.gamma.shape[1])

    @property
    def n_theta(self) -> int:
        """T."""
        return int(self.gamma.shape[2])

    @property
    def angle_independent(self) -> bool:
        """True for (S, F, 1, 1) tables (separable per carrier)."""
        return self.n_theta == 1

    @property
    def includes_cell_pattern(self) -> bool:
        """``provenance.includes_cell_pattern``."""
        return self.provenance.includes_cell_pattern

    def operating_band_hz(self) -> tuple[float, float]:
        """(f_lo, f_hi) of the frequency domain (inf for frequency-independent tables)."""
        if self.frequency_independent:
            return (0.0, math.inf)
        half = 0.0 if self.validity_bw_hz is None or self.n_freqs > 1 else self.validity_bw_hz / 2
        return (float(self.freqs_hz[0]) - half, float(self.freqs_hz[-1]) + half)

    def sha256(self) -> str:
        """Content hash of labels, grids and gamma (for ``gamma_source`` strings)."""
        h = hashlib.sha256()
        h.update("\x1f".join(self.labels).encode())
        for arr in (self.freqs_hz, self.theta_deg):
            h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(self.gamma, dtype="<c16").tobytes())
        return h.hexdigest()

    # ------------------------------------------------------------------ queries
    def _freq_weights(self, f_hz: float) -> tuple[int, float]:
        f = self.freqs_hz
        if self.frequency_independent or self.n_freqs == 1:
            if not self.frequency_independent:
                lo, hi = self.operating_band_hz()
                tol = 1e-9 * float(f[0])
                if not (lo - tol <= f_hz <= hi + tol) and self.extrapolate == "error":
                    raise OutOfDomain(
                        f"table {self.name!r}: f = {f_hz:.9g} Hz outside [{lo:.9g}, {hi:.9g}] Hz"
                    )
            return 0, 0.0
        if f_hz < f[0] or f_hz > f[-1]:
            if self.extrapolate == "error":
                raise OutOfDomain(
                    f"table {self.name!r}: f = {f_hz:.9g} Hz outside [{f[0]:.9g}, {f[-1]:.9g}] Hz"
                )
            f_hz = min(max(f_hz, float(f[0])), float(f[-1]))
        i, w = _axis_index(f, np.asarray([f_hz]))
        return int(i[0]), float(w[0])

    def gamma_at(self, f_hz: float) -> np.ndarray:
        """(S,) Gamma of an angle-independent table at ``f_hz`` (linear in f)."""
        if not self.angle_independent:
            raise ConfigurationError(f"table {self.name!r} depends on angle; use query()")
        i, w = self._freq_weights(f_hz)
        g = self.gamma[:, :, 0, 0]
        if self.n_freqs == 1 or w == 0.0:
            return g[:, i].copy()
        return (1.0 - w) * g[:, i] + w * g[:, i + 1]

    def query(
        self,
        f_hz: float,
        theta_i_deg: np.ndarray | float,
        theta_s_deg: np.ndarray | float,
        states: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Gamma (S_sel, Q) at carrier ``f_hz`` and angle pairs (Q,), complex-linear."""
        ti = np.atleast_1d(np.asarray(theta_i_deg, dtype=np.float64))
        ts = np.atleast_1d(np.asarray(theta_s_deg, dtype=np.float64))
        ti, ts = np.broadcast_arrays(ti, ts)
        sel = np.arange(self.n_states) if states is None else np.asarray(states, dtype=np.int64)
        fi, fw = self._freq_weights(f_hz)
        if self.angle_independent:
            g = self.gamma[sel, :, 0, 0]
            val = (
                g[:, fi]
                if (self.n_freqs == 1 or fw == 0.0)
                else ((1 - fw) * g[:, fi] + fw * g[:, fi + 1])
            )
            return np.repeat(val[:, None], ti.size, axis=1)
        th = self.theta_deg
        lo, hi = float(th[0]), float(th[-1])
        if np.any(ti < lo) or np.any(ti > hi) or np.any(ts < lo) or np.any(ts > hi):
            if self.extrapolate == "error":
                raise OutOfDomain(
                    f"table {self.name!r}: angle query outside [{lo}, {hi}] deg "
                    f"(theta_i in [{ti.min():.3f}, {ti.max():.3f}], theta_s in "
                    f"[{ts.min():.3f}, {ts.max():.3f}])"
                )
            ti = np.clip(ti, lo, hi)
            ts = np.clip(ts, lo, hi)
        i, wi = _axis_index(th, ti.ravel())
        j, wj = _axis_index(th, ts.ravel())

        def plane(fidx: int) -> np.ndarray:
            g = self.gamma[sel, fidx]  # (S, T, T)
            g00, g01 = g[:, i, j], g[:, i, j + 1]
            g10, g11 = g[:, i + 1, j], g[:, i + 1, j + 1]
            return (1 - wi) * ((1 - wj) * g00 + wj * g01) + wi * ((1 - wj) * g10 + wj * g11)

        out = plane(fi)
        if self.n_freqs > 1 and fw != 0.0 and not self.frequency_independent:
            out = (1 - fw) * out + fw * plane(fi + 1)
        return out.reshape(sel.size, *ti.shape) if ti.ndim > 1 else out

    # ------------------------------------------------------------------ transforms
    def with_efficiency(self, eta: float, *, override: bool = False) -> ResponseTable:
        """Scale amplitudes by ``sqrt(eta)`` (pipeline step 5).

        Refused (``NotPermitted``) for measured or digitized tables unless ``override``,
        because the measurement already contains the loss.
        """
        factor = efficiency_factor(eta, self.provenance.source_kind, override=override)
        if eta == 1.0:
            return self
        text = f"efficiency eta={eta!r} applied (sqrt on |Gamma|)"
        if override and self.provenance.source_kind in _UNCERTAIN_KINDS:
            text += f"; override_efficiency on a {self.provenance.source_kind} table"
        return self._derived(self.gamma * factor, self.provenance.with_assumption(text))

    def _derived(self, gamma: np.ndarray, prov: Provenance, **kw: Any) -> ResponseTable:
        t = ResponseTable(
            self.labels,
            self.freqs_hz,
            self.theta_deg,
            gamma,
            prov,
            validity_bw_hz=self.validity_bw_hz,
            name=kw.get("name", self.name),
            frequency_independent=self.frequency_independent,
            extrapolate=self.extrapolate,
            angle_mapping=self.angle_mapping,
            reconfiguration_time_s=self.reconfiguration_time_s,
            validate=kw.get("validate", True),
        )
        t.symmetrized = self.symmetrized
        return t

    def sample(self, rng: np.random.Generator) -> ResponseTable:
        """A perturbed copy drawn from the stated sigmas (Monte Carlo; keyed ``rng``).

        ``mag_db += N(0, sigma_mag_db)`` and ``phase += N(0, sigma_phase_deg)`` independently
        per entry; angle tables are perturbed on the upper triangle and mirrored, so the
        sample stays exactly reciprocal. Validators are not re-run (a sample may exceed
        ``|Gamma| = 1`` by construction of the error model).
        """
        sm, sp = self._sigma_arrays()
        shape = self.gamma.shape
        dm = rng.standard_normal(shape) * sm
        dp = rng.standard_normal(shape) * sp
        if self.n_theta > 1:
            iu = np.triu_indices(self.n_theta, 1)
            for arr in (dm, dp):
                arr[:, :, iu[1], iu[0]] = arr[:, :, iu[0], iu[1]]
        g = self.gamma * 10.0 ** (dm / 20.0) * np.exp(1j * np.radians(dp))
        prov = self.provenance.with_assumption("Monte Carlo sample of the stated sigmas")
        return self._derived(g, prov, validate=False)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_incidence_only(
        cls,
        labels: Sequence[str],
        freqs_hz: Sequence[float] | np.ndarray,
        theta_g_deg: Sequence[float] | np.ndarray,
        g: np.ndarray,
        provenance: Provenance,
        *,
        theta_deg: Sequence[float] | np.ndarray | None = None,
        **kw: Any,
    ) -> ResponseTable:
        """Reciprocal half-difference extension ``Gamma(ti, ts) := g(|ti - ts|/2)``.

        ``theta_g_deg`` (Tg,) is the incidence-angle grid of ``g`` (S, F, Tg), increasing
        from 0 to at most 90 deg. The extended table lives on ``theta_deg`` (default
        ``sorted(-theta_g U theta_g)``); off-grid half-differences are interpolated
        complex-linearly in ``g``. The result is exactly symmetric and its specular entry
        ``Gamma(theta, -theta)`` equals ``g(|theta|)``.
        """
        tg = np.asarray(theta_g_deg, dtype=np.float64)
        gg = np.asarray(g, dtype=np.complex128)
        if tg.ndim != 1 or tg.size < 2 or tg[0] != 0.0 or tg[-1] > 90 or np.any(np.diff(tg) <= 0):
            raise TableValidationError(
                "incidence-only grid must increase from 0 to <= 90 deg (g(0) is needed)"
            )
        if gg.ndim != 3 or gg.shape[2] != tg.size:
            raise TableValidationError("g must be (S, F, Tg)")
        th = (
            np.unique(np.concatenate([-tg, tg]))
            if theta_deg is None
            else np.asarray(theta_deg, dtype=np.float64)
        )
        x = np.abs(th[:, None] - th[None, :]) / 2.0  # exactly symmetric
        if x.max() > tg[-1]:
            raise TableValidationError("half-differences exceed the incidence-only grid")
        i, w = _axis_index(tg, x.ravel())
        lo = gg[:, :, i]
        hi = gg[:, :, i + 1]
        ext = ((1 - w) * lo + w * hi).reshape(gg.shape[0], gg.shape[1], th.size, th.size)
        prov = provenance.with_assumption(
            "angle_mapping='half_difference': Gamma(ti, ts) = g(|ti - ts|/2) [ENG]"
        )
        return cls(labels, freqs_hz, th, ext, prov, angle_mapping="half_difference", **kw)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        sidecar: str | Path | Mapping[str, Any],
        *,
        symmetrize: Literal["mean"] | None = None,
        extrapolate: Literal["error", "nearest"] = "error",
        name: str | None = None,
    ) -> ResponseTable:
        """Import the long-form CSV of section 5.5 plus a provenance sidecar.

        Columns ``state,f_hz,theta_i_deg,theta_s_deg,mag_db,phase_deg`` with optional
        ``sigma_mag_db,sigma_phase_deg`` (per entry). Empty or NaN angle columns make an
        angle-independent table. Every (state, f, theta_i, theta_s) of the grid must appear
        exactly once. The sidecar (JSON/YAML path or mapping) holds the provenance fields,
        either flat or under ``provenance``, plus optional ``validity_bw_hz`` and ``name``;
        ``raw_sha256`` defaults to the CSV's SHA-256 and must match it when given.
        """
        p = Path(path)
        raw = p.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        meta = _load_mapping(sidecar)
        prov_d = dict(meta.get("provenance", meta))
        prov_d.pop("validity_bw_hz", None)
        prov_d.pop("name", None)
        if prov_d.get("raw_sha256") in (None, ""):
            prov_d["raw_sha256"] = digest
        elif prov_d["raw_sha256"] != digest:
            raise TableValidationError(
                f"{p.name}: raw_sha256 in the sidecar does not match the file ({digest})"
            )
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        need = {"state", "f_hz", "theta_i_deg", "theta_s_deg", "mag_db", "phase_deg"}
        if not rows or not need <= set(rows[0]):
            raise TableValidationError(f"{p.name}: CSV needs columns {sorted(need)}")

        def num(x: str | None) -> float:
            return float("nan") if x is None or x.strip() == "" else float(x)

        states: list[str] = []
        for r in rows:
            lab = r["state"].strip()
            if lab not in states:
                states.append(lab)
        if all(re.fullmatch(r"-?\d+", s) for s in states):
            states.sort(key=int)
        f_all = np.array([num(r["f_hz"]) for r in rows])
        ti_all = np.array([num(r["theta_i_deg"]) for r in rows])
        ts_all = np.array([num(r["theta_s_deg"]) for r in rows])
        angle_free = bool(np.all(np.isnan(ti_all)) and np.all(np.isnan(ts_all)))
        if not angle_free and (np.any(np.isnan(ti_all)) or np.any(np.isnan(ts_all))):
            raise TableValidationError(f"{p.name}: angle columns are partly empty")
        freqs = np.unique(f_all)
        theta = np.zeros(1) if angle_free else np.unique(np.concatenate([ti_all, ts_all]))
        shape = (len(states), freqs.size, theta.size, theta.size)
        gam = np.full(shape, np.nan + 0j)
        sig_m = np.zeros(shape)
        sig_p = np.zeros(shape)
        has_sig = "sigma_mag_db" in rows[0] and "sigma_phase_deg" in rows[0]
        for r, f, ti, ts in zip(rows, f_all, ti_all, ts_all, strict=True):
            s = states.index(r["state"].strip())
            fi = int(np.searchsorted(freqs, f))
            ii = 0 if angle_free else int(np.searchsorted(theta, ti))
            jj = 0 if angle_free else int(np.searchsorted(theta, ts))
            if not np.isnan(gam[s, fi, ii, jj]):
                raise TableValidationError(f"{p.name}: duplicate entry {r}")
            gam[s, fi, ii, jj] = 10.0 ** (num(r["mag_db"]) / 20.0) * _unit_deg(num(r["phase_deg"]))
            if has_sig:
                sig_m[s, fi, ii, jj] = num(r["sigma_mag_db"])
                sig_p[s, fi, ii, jj] = num(r["sigma_phase_deg"])
        if np.any(np.isnan(gam)):
            raise TableValidationError(
                f"{p.name}: the (state, f, theta_i, theta_s) grid is incomplete"
            )
        if has_sig:
            prov_d["sigma_mag_db"] = sig_m
            prov_d["sigma_phase_deg"] = sig_p
        prov = Provenance.from_dict(prov_d)
        return cls(
            states,
            freqs,
            theta,
            gam,
            prov,
            validity_bw_hz=_float_or_none(meta.get("validity_bw_hz")),
            name=name or meta.get("name", p.stem),
            extrapolate=extrapolate,
            symmetrize=symmetrize,
        )

    @classmethod
    def from_matlab_scurve(
        cls,
        path: str | Path,
        *,
        patch_lengths_mm: Sequence[float] | None = None,
        tan_delta: float = 0.0,
        validity_bw_hz: float | None = None,
        name: str | None = None,
    ) -> ResponseTable:
        """MATLAB lossless S-curve CSV (``unitcell_fullwave_*.csv``), phase only.

        Each selected ``patch_L_mm`` becomes a state with ``Gamma = exp(j phase)`` (lossless,
        labelled a SIMULATED stand-in); every selected state must exist at every frequency
        of the file. The lossy infinite-array path is not trusted (non-passive |Gamma|).
        """
        p = Path(path)
        raw = p.read_bytes()
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        need = {"f_hz", "tan_delta", "patch_L_mm", "gamma_phase_deg"}
        if not rows or not need <= set(rows[0]):
            raise TableValidationError(f"{p.name}: not a MATLAB S-curve CSV (need {sorted(need)})")
        rows = [r for r in rows if float(r["tan_delta"]) == tan_delta]
        if not rows:
            raise TableValidationError(f"{p.name}: no rows with tan_delta = {tan_delta}")
        freqs = np.unique([float(r["f_hz"]) for r in rows])
        lengths = (
            sorted({float(r["patch_L_mm"]) for r in rows})
            if patch_lengths_mm is None
            else [float(x) for x in patch_lengths_mm]
        )
        gam = np.full((len(lengths), freqs.size, 1, 1), np.nan + 0j)
        for r in rows:
            L = float(r["patch_L_mm"])
            if L not in lengths:
                continue
            gam[lengths.index(L), int(np.searchsorted(freqs, float(r["f_hz"]))), 0, 0] = _unit_deg(
                float(r["gamma_phase_deg"])
            )
        if np.any(np.isnan(gam)):
            raise TableValidationError(
                f"{p.name}: selected patch lengths missing at some frequency"
            )
        prov = Provenance(
            source_kind="fullwave",
            description=(
                "MATLAB R2026a lossless unit-cell S-curve, phase only; SIMULATED "
                "stand-in for measured states"
            ),
            reference=p.name,
            reference_plane_offset_m=0.0,
            polarization="co",
            includes_cell_pattern=False,
            sigma_mag_db=0.0,
            sigma_phase_deg=0.0,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            license="internal (generated by the authors)",
            created={"tool": "MATLAB R2026a", "importer": "ResponseTable.from_matlab_scurve"},
            assumptions=(
                "|Gamma| set to 1 (lossless S-curve; phase only)",
                f"tan_delta = {tan_delta}",
            ),
        )
        labels = [f"L{x:g}mm" for x in lengths]
        return cls(
            labels, freqs, [0.0], gam, prov, validity_bw_hz=validity_bw_hz, name=name or p.stem
        )

    @classmethod
    def from_state_library(
        cls,
        doc: Any,
        *,
        assume_mag: Mapping[str, float] | None = None,
        base_dir: Path | None = None,
        extrapolate: Literal["error", "nearest"] = "error",
        symmetrize: Literal["mean"] | None = None,
    ) -> ResponseTable:
        """Load a state-library document (``data/states/*.yaml`` format, section 9.4): a
        mapping, a file path or a WP1 ``StateLibrary`` model (``model_dump``).

        ``frequency_hz: null`` marks a frequency-independent table. ``angle_grid: none``
        (or absent) gives an angle-independent table; ``angle_grid: {theta_deg: [...]}`` an
        angle table whose gamma entries carry ``theta_i_deg``/``theta_s_deg``;
        ``angle_mapping: half_difference`` with entries carrying ``theta_deg`` gives the
        incidence-only extension. ``mag: null`` forces ``assume_mag[label]`` (recorded).
        ``reciprocal: false`` is rejected (release 1).
        """
        if isinstance(doc, str | Path):
            p = Path(doc)
            if base_dir is not None and not p.is_absolute():
                p = base_dir / p
            d = _read_library(p)
        elif hasattr(doc, "model_dump"):  # a sremu.scenario.models.StateLibrary (WP1)
            d = doc.model_dump(mode="python")
        else:
            d = dict(doc)
        if d.get("family") is not None:  # an unexpanded analytic family: use its defaults
            d = _expand_family(d, (), {})
        if d.get("kind", "ris_state_library") != "ris_state_library":
            raise TableValidationError(f"not a ris_state_library document (kind={d.get('kind')!r})")
        if d.get("reciprocal", True) is False:
            raise TableValidationError(
                "raw non-reciprocal tables (reciprocal: false) are rejected in release 1"
            )
        prov = Provenance.from_dict(d.get("provenance") or {})
        name = str(d.get("name", ""))
        states = d.get("states") or []
        if not states:
            raise TableValidationError(f"state library {name!r} has no states")
        labels = [str(s.get("label", s.get("id"))) for s in states]
        fdecl = d.get("frequency_hz")
        freq_indep = fdecl is None or fdecl == "any"
        freqs = np.array([0.0]) if freq_indep else np.asarray(fdecl, dtype=np.float64)
        grid = d.get("angle_grid", "none")
        mapping = d.get("angle_mapping", "none" if grid in (None, "none") else "full")
        assumptions: list[str] = []
        am = {str(k): float(v) for k, v in (assume_mag or {}).items()}

        def mag_of(label: str, e: Mapping[str, Any]) -> float:
            m = e.get("mag")
            if m is None and "mag_db" in e and e["mag_db"] is not None:
                return 10.0 ** (float(e["mag_db"]) / 20.0)
            if m is None:
                if label not in am:
                    raise ConfigurationError(
                        f"state library {name!r}: state {label!r} has mag: null; the scenario "
                        "must set assume_mag for it"
                    )
                return am[label]
            return float(m)

        for lab in labels:
            if lab in am and any(
                e.get("mag") is None
                for s in states
                if str(s.get("label", s.get("id"))) == lab
                for e in s.get("gamma", [])
            ):
                assumptions.append(f"assume_mag: |Gamma({lab})| = {am[lab]!r} (ASSUMED)")
        unknown = set(am) - set(labels)
        if unknown:
            raise ConfigurationError(f"assume_mag names unknown states {sorted(unknown)}")
        if d.get("parameters"):
            assumptions.append(f"analytic family parameters {dict(d['parameters'])}")

        def fidx(e: Mapping[str, Any]) -> int:
            if freq_indep:
                if e.get("f_hz") is not None:
                    raise TableValidationError("frequency-independent tables need f_hz: null")
                return 0
            if e.get("f_hz") is None:
                raise TableValidationError(f"state library {name!r}: gamma entry without f_hz")
            f = float(e["f_hz"])
            k = int(np.argmin(np.abs(freqs - f)))
            if abs(freqs[k] - f) > 1e-9 * f:
                raise TableValidationError(f"gamma entry at f = {f} not in frequency_hz")
            return k

        if grid in (None, "none"):
            gam = np.full((len(labels), freqs.size, 1, 1), np.nan + 0j)
            for s, st in enumerate(states):
                for e in st.get("gamma", []):
                    gam[s, fidx(e), 0, 0] = mag_of(labels[s], e) * _unit_deg(float(e["phase_deg"]))
            theta = np.zeros(1)
        else:
            theta = np.asarray(grid["theta_deg"], dtype=np.float64)
            if mapping == "half_difference":
                g = np.full((len(labels), freqs.size, theta.size), np.nan + 0j)
                for s, st in enumerate(states):
                    for e in st.get("gamma", []):
                        t = int(np.argmin(np.abs(theta - float(e["theta_deg"]))))
                        g[s, fidx(e), t] = mag_of(labels[s], e) * _unit_deg(float(e["phase_deg"]))
                if np.any(np.isnan(g)):
                    raise TableValidationError(f"state library {name!r}: incomplete grid")
                for a in assumptions:
                    prov = prov.with_assumption(a)
                return cls.from_incidence_only(
                    labels,
                    freqs,
                    theta,
                    g,
                    prov,
                    validity_bw_hz=_float_or_none(d.get("validity_bw_hz")),
                    name=name,
                    frequency_independent=freq_indep,
                    extrapolate=extrapolate,
                    reconfiguration_time_s=_exact_or_none(d.get("reconfiguration_time_s")),
                )
            if mapping != "full":
                raise TableValidationError(
                    f"angle tables need angle_mapping 'full' or 'half_difference', got {mapping!r}"
                )
            if any(
                "theta_deg" in e and "theta_i_deg" not in e
                for st in states
                for e in st.get("gamma", [])
            ):
                raise TableValidationError(
                    f"state library {name!r}: incidence-only entries (theta_deg) need "
                    "angle_mapping: half_difference (raw incidence-only tables are not reciprocal)"
                )
            gam = np.full((len(labels), freqs.size, theta.size, theta.size), np.nan + 0j)
            for s, st in enumerate(states):
                for e in st.get("gamma", []):
                    i = int(np.argmin(np.abs(theta - float(e["theta_i_deg"]))))
                    j = int(np.argmin(np.abs(theta - float(e["theta_s_deg"]))))
                    gam[s, fidx(e), i, j] = mag_of(labels[s], e) * _unit_deg(float(e["phase_deg"]))
        if np.any(np.isnan(gam)):
            raise TableValidationError(f"state library {name!r}: incomplete (state, f, angle) grid")
        for a in assumptions:
            prov = prov.with_assumption(a)
        return cls(
            labels,
            freqs,
            theta,
            gam,
            prov,
            validity_bw_hz=_float_or_none(d.get("validity_bw_hz")),
            name=name,
            frequency_independent=freq_indep,
            extrapolate=extrapolate,
            symmetrize=symmetrize,
            reconfiguration_time_s=_exact_or_none(d.get("reconfiguration_time_s")),
        )

    def to_state_library(self) -> dict[str, Any]:
        """The section 9.4 document form of this table (amplitudes always explicit)."""
        states = []
        for s, lab in enumerate(self.labels):
            entries = []
            for fi in range(self.n_freqs):
                f = None if self.frequency_independent else float(self.freqs_hz[fi])
                for i in range(self.n_theta):
                    for j in range(self.n_theta):
                        g = complex(self.gamma[s, fi, i, j])
                        e: dict[str, Any] = {}
                        if f is not None:
                            e["f_hz"] = f
                        if self.n_theta > 1:
                            e["theta_i_deg"] = float(self.theta_deg[i])
                            e["theta_s_deg"] = float(self.theta_deg[j])
                        e["mag"] = abs(g)
                        e["phase_deg"] = math.degrees(math.atan2(g.imag, g.real))
                        entries.append(e)
            states.append({"id": s, "label": lab, "gamma": entries})
        return {
            "schema_version": "1.0",
            "kind": "ris_state_library",
            "name": self.name,
            "provenance": self.provenance.to_dict(),
            "frequency_hz": None if self.frequency_independent else self.freqs_hz.tolist(),
            "validity_bw_hz": self.validity_bw_hz,
            "angle_grid": "none" if self.n_theta == 1 else {"theta_deg": self.theta_deg.tolist()},
            "angle_mapping": "none" if self.n_theta == 1 else "full",
            "reciprocal": True,
            "reconfiguration_time_s": None
            if self.reconfiguration_time_s is None
            else fraction_to_str(self.reconfiguration_time_s),
            "states": states,
        }

    def summary(self) -> dict[str, Any]:
        """Short description for CLI and manifests."""
        err = 0.0
        if self.n_theta > 1:
            e, _ = self.reciprocity_error()
            err = float(e.max())
        return {
            "name": self.name,
            "states": list(self.labels),
            "shape": list(self.gamma.shape),
            "frequency_independent": self.frequency_independent,
            "freqs_hz": [] if self.frequency_independent else self.freqs_hz.tolist(),
            "angle_independent": self.angle_independent,
            "angle_mapping": self.angle_mapping,
            "validity_bw_hz": self.validity_bw_hz,
            "max_abs_gamma": float(np.max(np.abs(self.gamma))),
            "max_reciprocity_error": err,
            "grid_density": self.grid_density(),
            "symmetrized": self.symmetrized,
            "source_kind": self.provenance.source_kind,
            "assumptions": list(self.provenance.assumptions),
            "sha256": self.sha256(),
        }

    def __repr__(self) -> str:
        return (
            f"ResponseTable(name={self.name!r}, states={self.labels}, shape={self.gamma.shape},"
            f" source_kind={self.provenance.source_kind!r})"
        )


def _unit_deg(phase_deg: float) -> complex:
    """``exp(j phase)`` from degrees, exact at multiples of 90 deg (``cosdg``/``sindg``)."""
    from scipy.special import cosdg, sindg

    return complex(float(cosdg(phase_deg)), float(sindg(phase_deg)))


def _float_or_none(x: Any) -> float | None:
    """YAML 1.1 reads ``2.0e7`` as a string: coerce explicitly."""
    return None if x is None else float(x)


def _exact_or_none(x: Any) -> Fraction | None:
    if x is None:
        return None
    return parse_exact(
        str(x) if not isinstance(x, Fraction | int) else x, name="reconfiguration_time_s"
    )


def _load_mapping(src: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(src, Mapping):
        return dict(src)
    p = Path(src)
    text = p.read_text(encoding="utf-8")
    out = yaml.safe_load(text)  # JSON is a subset of YAML
    if not isinstance(out, Mapping):
        raise TableValidationError(f"{p.name}: sidecar must be a mapping")
    return dict(out)


# --------------------------------------------------------------------------- built-ins


def _analytic_provenance(description: str, reference: str) -> Provenance:
    return Provenance(
        source_kind="analytic",
        description=description,
        reference=reference,
        reference_plane_offset_m=0.0,
        polarization="co",
        includes_cell_pattern=False,
        sigma_mag_db=0.0,
        sigma_phase_deg=0.0,
        raw_sha256=None,
        license="n/a (analytic)",
        created={"tool": "sremu.physics.response", "date": "2026-10-10"},
    )


def _family_gamma(family: str, params: Mapping[str, float]) -> tuple[list[str], np.ndarray]:
    """Expanded analytic family (the WP1 ``data/states`` convention): labels ``s0..``, phases
    ``360 s / 2^b`` deg, ``ideal_bbit`` unit amplitude, ``abeywickrama`` amplitude
    ``(1 - beta_min)((sin(theta - phi) + 1)/2)^alpha + beta_min`` with ``phi = phi_deg``."""
    b = int(params["b"])
    if not 1 <= b <= 8:
        raise ConfigurationError(f"{family}: b must lie in [1, 8]")
    n = 1 << b
    out = []
    for k in range(n):
        unit = _unit_deg(360.0 * k / n)
        if family == "ideal_bbit":
            mag = 1.0
        elif family == "abeywickrama":
            bmin, alpha = float(params["beta_min"]), float(params["alpha"])
            if not (0.0 <= bmin <= 1.0 and alpha >= 0.0):
                raise ConfigurationError("abeywickrama needs 0 <= beta_min <= 1, alpha >= 0")
            th = 2.0 * math.pi * k / n
            phi = math.radians(float(params["phi_deg"]))
            mag = (1.0 - bmin) * ((math.sin(th - phi) + 1.0) / 2.0) ** alpha + bmin
        else:
            raise ConfigurationError(f"unknown analytic family {family!r}")
        out.append(mag * unit)
    return [f"s{k}" for k in range(n)], np.asarray(out, dtype=np.complex128)


def _num_text(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def _expand_family(
    d: Mapping[str, Any], pos: Sequence[float], kw: Mapping[str, float]
) -> dict[str, Any]:
    """Expand a ``family`` state-library document into explicit states (fallback when WP1's
    ``sremu.scenario.load.expand_family`` is unavailable; same conventions)."""
    fam = d["family"]
    spec = dict(fam.get("parameters") or {})
    names = list(spec)
    if len(pos) > len(names):
        raise ConfigurationError(f"{fam['name']}: too many arguments")
    bound = dict(zip(names, pos, strict=False))
    for k, v in kw.items():
        if k not in spec or k in bound:
            raise ConfigurationError(f"{fam['name']}: bad parameter {k!r}")
        bound[k] = v
    params: dict[str, float] = {}
    for k in names:
        val = bound.get(k, (spec[k] or {}).get("default"))
        if val is None:
            raise ConfigurationError(f"{fam['name']}: parameter {k!r} is required")
        params[k] = float(val)
    labels, g = _family_gamma(str(fam["name"]), params)
    out = dict(d)
    out.update(
        name=f"{fam['name']}(" + ", ".join(f"{k}={_num_text(v)}" for k, v in params.items()) + ")",
        family=None,
        parameters=params,
        frequency_hz="any",
        states=[
            {
                "id": k,
                "label": lab,
                "gamma": [{"f_hz": None, "mag": abs(complex(z)), "phase_deg": 360.0 * k / len(g)}],
            }
            for k, (lab, z) in enumerate(zip(labels, g, strict=True))
        ],
    )
    return out


def ideal_bbit_table(b: int) -> ResponseTable:
    """Ideal b-bit phase alphabet ``exp(j 2 pi s / 2^b)``, s = 0..2^b-1 (analytic)."""
    labels, g = _family_gamma("ideal_bbit", {"b": b})
    prov = _analytic_provenance(
        f"ideal {b}-bit phase alphabet exp(j 2 pi s / 2^{b})",
        "Wu & Zhang, arXiv:1906.03165 (discrete phase shifts)",
    )
    return ResponseTable(
        labels,
        [0.0],
        [0.0],
        g.reshape(-1, 1, 1, 1),
        prov,
        name=f"ideal_bbit(b={b})",
        frequency_independent=True,
    )


def abeywickrama_table(b: int, beta_min: float, phi_deg: float, alpha: float) -> ResponseTable:
    """2^b samples of ``beta(theta) exp(j theta)``, ``theta = 2 pi s / 2^b``, with
    ``beta(theta) = (1 - beta_min)((sin(theta - phi) + 1)/2)^alpha + beta_min``
    (Abeywickrama et al., arXiv:2002.10112, Eq. 5); ``phi`` in degrees (WP1 convention)."""
    params = {"b": b, "beta_min": beta_min, "phi_deg": phi_deg, "alpha": alpha}
    labels, g = _family_gamma("abeywickrama", params)
    prov = _analytic_provenance(
        f"Abeywickrama practical phase-shift model {params}",
        "Abeywickrama et al., arXiv:2002.10112, Eq. 5",
    )
    name = "abeywickrama(" + ", ".join(f"{k}={_num_text(v)}" for k, v in params.items()) + ")"
    return ResponseTable(
        labels, [0.0], [0.0], g.reshape(-1, 1, 1, 1), prov, name=name, frequency_independent=True
    )


_REF_RE = re.compile(r"^\s*([A-Za-z0-9_\-.]+)\s*(?:\((.*)\))?\s*$", re.S)


def _parse_ref(ref: str) -> tuple[str, tuple[float, ...], dict[str, float]]:
    m = _REF_RE.match(ref)
    if m is None:
        raise ConfigurationError(f"bad state library reference {ref!r}")
    name, args = m.group(1), m.group(2)
    pos: list[float] = []
    kw: dict[str, float] = {}
    for a in [x.strip() for x in (args or "").split(",") if x.strip()]:
        if "=" in a:
            k, v = a.split("=", 1)
            kw[k.strip()] = float(v)
        else:
            pos.append(float(a))
    return name, tuple(pos), kw


def _wp1_states_dir() -> Path | None:
    """WP1's built-in state-library directory (``data/states``) when it can be found."""
    try:
        from sremu.scenario.load import builtin_states_dir
    except ImportError:
        return None
    try:
        d = builtin_states_dir()
    except Exception:  # directory absent (e.g. a wheel without data/states)
        return None
    return d if d.is_dir() else None


def _read_library(path: Path) -> dict[str, Any]:
    """Read a state-library file with WP1's YAML 1.2 reader when available."""
    try:
        from sremu.scenario.load import read_document
    except ImportError:
        out = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        out = read_document(path)
    if not isinstance(out, Mapping):
        raise TableValidationError(f"{path.name}: a state library must be a mapping")
    return dict(out)


def builtin_names() -> list[str]:
    """Built-in state libraries: WP1's ``data/states`` when present, else the packaged copies
    in ``physics/tables`` (byte-identical, checked by the WP2 tests)."""
    wp1 = _wp1_states_dir()
    root = wp1 if wp1 is not None else BUILTIN_DIR
    return sorted(p.stem for p in root.glob("*.yaml"))


def builtin(name: str, *, assume_mag: Mapping[str, float] | None = None) -> ResponseTable:
    """A built-in table by reference: ``lab_binary_3p5ghz``, ``tang2022_ris1_27ghz``,
    ``tang2022_ris2_33ghz``, ``ideal_antipodal``, ``on_off`` or a parametric family
    ``ideal_bbit(b=2)`` / ``abeywickrama(b=2, beta_min=0.2, phi_deg=0, alpha=1.6)``
    (positional arguments follow the family's parameter order).

    WP1's ``data/states`` libraries are used when present (through
    ``sremu.scenario.load.load_state_library``); otherwise the packaged copies.
    """
    if _wp1_states_dir() is not None:
        from sremu.scenario.load import load_state_library

        lib, _path = load_state_library(name)
        return ResponseTable.from_state_library(lib, assume_mag=assume_mag)
    stem, pos, kw = _parse_ref(name)
    path = BUILTIN_DIR / f"{stem}.yaml"
    if not path.is_file():
        raise ConfigurationError(f"unknown built-in table {stem!r}; known: {builtin_names()}")
    d = _read_library(path)
    if d.get("family") is not None:
        d = _expand_family(d, pos, kw)
    elif pos or kw:
        raise ConfigurationError(f"{stem} is not a parametric family; it takes no arguments")
    return ResponseTable.from_state_library(d, assume_mag=assume_mag)


def load_table(
    spec: Mapping[str, Any] | str,
    *,
    base_dir: Path | None = None,
) -> ResponseTable:
    """Scenario ``ris[].states`` (section 9.2): ``{library: <builtin> | file: <path>,
    assume_mag?: {label: mag}}``, or a bare built-in name. Efficiency is applied by the
    caller (:meth:`ResponseTable.with_efficiency`)."""
    if isinstance(spec, str):
        return builtin(spec)
    am = spec.get("assume_mag")
    if "library" in spec and spec["library"] is not None:
        lib = spec["library"]
        if isinstance(lib, Mapping):  # {name: ideal_bbit, b: 2}
            nm = lib["name"]
            args = ", ".join(f"{k}={v}" for k, v in lib.items() if k != "name")
            return builtin(f"{nm}({args})" if args else nm, assume_mag=am)
        return builtin(str(lib), assume_mag=am)
    if "file" in spec and spec["file"] is not None:
        p = Path(spec["file"])
        if base_dir is not None and not p.is_absolute():
            p = base_dir / p
        if p.suffix.lower() == ".csv":
            side = spec.get("provenance") or p.with_suffix(".yaml")
            return ResponseTable.from_csv(
                p,
                side,
                symmetrize=spec.get("symmetrize"),
                extrapolate=spec.get("extrapolate", "error"),
            )
        return ResponseTable.from_state_library(
            p,
            assume_mag=am,
            extrapolate=spec.get("extrapolate", "error"),
            symmetrize=spec.get("symmetrize"),
        )
    raise ConfigurationError("states needs 'library' or 'file'")
