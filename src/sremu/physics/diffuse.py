"""Additive diffuse multipath: Clarke/Jakes sum of sinusoids (DESIGN.md section 5.11; WP3).

**Process (D19).** :func:`sos_gain` ``(key, t_s, max_doppler_hz, n_sinusoids=48)`` returns
the complex gain ``g(t)`` of one diffuse tap at absolute times ``t_s``::

    g(t) = (1/sqrt(M)) sum_{m=0}^{M-1} exp(j (2 pi f_D cos(alpha_m) t + phi_m))
    alpha_m = 2 pi (m + delta)/M

* ``M >= 48`` sinusoids (section 5.11) with equal amplitudes, so ``E|g|^2 = 1`` exactly
  (unit mean power) and the time-average power of one realization tends to 1.
* Keyed randomness (``sremu.rng`` only; purpose ``diffuse``, name ``<link>``, section 6.2):
  the phases ``phi_m = 2 pi u_m`` with ``u_m`` = Philox uniform at index ``m`` of lane
  ``2*tap``, and the arrival-angle offset ``delta = 1/8 + u/4`` with ``u`` = Philox uniform at
  index 0 of lane ``2*tap + 1``. Every parameter is a pure function of
  ``(seed, link, tap, m)``.
* ``delta`` in [1/8, 3/8] keeps the angle set away from its mirror image (``delta`` in
  {0, 1/2} would pair ``cos(alpha_m)`` values), so all M Doppler frequencies are distinct
  with spacing at least ``0.25 (2 pi/M)^2 f_D / 2``; the time-average statistics of a single
  realization then converge [ENG].
* Autocorrelation: ``E[g(t+tau) g*(t)] = (1/M) sum_m exp(j 2 pi f_D cos(alpha_m) tau)
  = J0(2 pi f_D tau) + 2 J_M(2 pi f_D tau) cos(2 pi delta) + O(J_2M)`` (Jacobi-Anger), real and
  equal to the Clarke ``J0`` to about ``J_48(12.6) ~ 1e-20`` at the 40 ms lag of ST-02.
* ``g`` is a **pure function of absolute time**: each sample is computed from ``t`` alone in a
  fixed order over ``m`` (cycles reduced modulo 1 before the phase is formed), so chunking,
  skipping and the order of evaluation never change it (chunk and skip invariance, I2a/I4).

**Use in the channel** (section 5.11). The diffuse term is **additive** per link and never
multiplies the deterministic channel (v2's composite multiplier made "RIS gain"
fading-invariant). Per link, up to 4 taps (scenario ``environment.diffuse.taps``; one tap at
0 s / 0 dB by default), each with an independent SoS (lane ``tap``). Power is set by the
Rician K-factor relative to the link's deterministic non-RIS paths [ENG]:

    P_ref(t) = sum |amp_p(t)|^2 over the PathBundle paths of kind LOS, GROUND, WALL, DIFF
    P_tap(t) = P_ref(t) * w_tap / K,  w = 10^(rel_power_db/10) normalized to sum 1

(incoherent sum, so the reference does not fade; RIS paths are excluded so that no RIS
configuration changes the diffuse power). Tap ``i`` arrives at ``tau_LOS(t) + delay_s[i]``.
The contribution of tap ``i`` to the RX baseband is ``sqrt(P_tap(t)) g_i(t) x(t - tau_i(t))``
with the envelope delay only: the carrier phase of the diffuse term is the SoS phase itself
(adding ``exp(-j 2 pi f_c tau)`` would shift the Clarke spectrum by the reference path's
Doppler). :class:`DiffuseMechanism` computes these terms per snapshot
(:meth:`DiffuseMechanism.terms`); a renderer evaluates ``g`` at absolute sample indices
with :func:`sos_gain_samples` (``t = k/f_s`` correctly rounded from the exact rate, so the
value depends on ``k`` only) and interpolates ``sqrt(P_tap)`` and the delay between
snapshots like a path amplitude and a chord delay. The render hook that adds the terms to
the IQ is a contract change request (``LinkJob`` has no diffuse field in contract 1.1).

``max_doppler_hz`` is given, or ``from_motion: true`` takes ``v_rel,max/lambda`` of the
link over the run horizon (``PropagationModel.motion_bounds``).
"""

from __future__ import annotations

import contextlib
import math
import numbers
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sremu import rng as _rng
from sremu.contracts import PathBundle, PathKind
from sremu.errors import ConfigurationError
from sremu.timebase import parse_exact
from sremu.units import C0

__all__ = [
    "DETERMINISTIC_KINDS",
    "MAX_TAPS",
    "MIN_SINUSOIDS",
    "DiffuseMechanism",
    "DiffuseTap",
    "DiffuseTerm",
    "SosParameters",
    "find_diffuse",
    "mechanism_from_scenario",
    "reference_power",
    "sample_times_s",
    "sos_gain",
    "sos_gain_samples",
    "sos_parameters",
]

MIN_SINUSOIDS: Final[int] = 48
"""Minimum number of sinusoids (section 5.11)."""

MAX_TAPS: Final[int] = 4
"""Maximum number of diffuse taps per link (section 5.11)."""

DETERMINISTIC_KINDS: Final[tuple[int, ...]] = (
    int(PathKind.LOS),
    int(PathKind.GROUND),
    int(PathKind.WALL),
    int(PathKind.DIFF),
)
"""Path kinds whose incoherent power sum is the K-factor reference."""

_TWO_PI: Final[float] = 2.0 * math.pi
_CHUNK: Final[int] = 1 << 16


@dataclass(frozen=True)
class SosParameters:
    """Sinusoid frequencies ``f_D cos(alpha_m)`` (Hz), phases (rad), the common amplitude
    ``1/sqrt(M)`` and the angle offset ``delta`` of one diffuse tap."""

    freqs_hz: NDArray[np.float64]
    phases_rad: NDArray[np.float64]
    amplitude: float
    angle_offset: float

    @property
    def n_sinusoids(self) -> int:
        """M."""
        return int(self.freqs_hz.shape[0])


def _check_m(n_sinusoids: int) -> int:
    if isinstance(n_sinusoids, bool) or not isinstance(n_sinusoids, numbers.Integral):
        raise ConfigurationError(f"n_sinusoids must be an int, got {n_sinusoids!r}")
    m = int(n_sinusoids)
    if m < MIN_SINUSOIDS:
        raise ConfigurationError(f"n_sinusoids must be >= {MIN_SINUSOIDS} (section 5.11), got {m}")
    return m


def _check_fd(max_doppler_hz: float) -> float:
    fd = float(max_doppler_hz)
    if not (math.isfinite(fd) and fd >= 0.0):
        raise ConfigurationError(f"max_doppler_hz must be finite and >= 0, got {max_doppler_hz!r}")
    return fd


def _check_tap(tap: int) -> int:
    if (
        isinstance(tap, bool)
        or not isinstance(tap, numbers.Integral)
        or not 0 <= int(tap) < 1 << 30
    ):
        raise ConfigurationError(f"tap must be an int in [0, 2**30), got {tap!r}")
    return int(tap)


def sos_parameters(
    key: _rng.PhiloxKey | tuple[int, int],
    max_doppler_hz: float,
    n_sinusoids: int = MIN_SINUSOIDS,
    *,
    tap: int = 0,
) -> SosParameters:
    """The keyed SoS parameters of one tap (module docstring). Pure in its arguments."""
    m = _check_m(n_sinusoids)
    fd = _check_fd(max_doppler_hz)
    tp = _check_tap(tap)
    u_phase = _rng.uniform(key, 0, m, sub=2 * tp)
    u_off = float(_rng.uniform(key, 0, 1, sub=2 * tp + 1)[0])
    delta = 0.125 + 0.25 * u_off
    alpha = _TWO_PI * (np.arange(m, dtype=np.float64) + delta) / m
    return SosParameters(
        freqs_hz=fd * np.cos(alpha),
        phases_rad=_TWO_PI * u_phase,
        amplitude=1.0 / math.sqrt(m),
        angle_offset=delta,
    )


def _sos_eval(p: SosParameters, t: NDArray[np.float64]) -> NDArray[np.complex128]:
    out = np.empty(t.shape, dtype=np.complex128)
    flat_t = t.reshape(-1)
    flat_o = out.reshape(-1)
    for a in range(0, flat_t.size, _CHUNK):
        tt = flat_t[a : a + _CHUNK]
        re = np.zeros(tt.shape, dtype=np.float64)
        im = np.zeros(tt.shape, dtype=np.float64)
        for f, phi in zip(p.freqs_hz.tolist(), p.phases_rad.tolist(), strict=True):
            ph = _TWO_PI * np.mod(f * tt, 1.0) + phi
            re += np.cos(ph)
            im += np.sin(ph)
        flat_o[a : a + _CHUNK] = (re + 1j * im) * p.amplitude
    return out


def sos_gain(
    key: _rng.PhiloxKey | tuple[int, int],
    t_s: ArrayLike,
    max_doppler_hz: float,
    n_sinusoids: int = MIN_SINUSOIDS,
    *,
    tap: int = 0,
) -> NDArray[np.complex128] | complex:
    """Diffuse SoS gain ``g(t)`` at absolute times ``t_s`` (s), complex128, unit mean power
    (D19; module docstring). ``key``: Philox key ``(k0, k1)``, normally
    ``sremu.rng.derive_key(seed, "diffuse", link)``; ``tap`` selects an independent process of
    the same key. A scalar time gives a complex scalar; an array gives an array of its shape.
    """
    p = sos_parameters(key, max_doppler_hz, n_sinusoids, tap=tap)
    t = np.asarray(t_s, dtype=np.float64)
    if not np.all(np.isfinite(t)):
        raise ConfigurationError("t_s must be finite")
    g = _sos_eval(p, np.atleast_1d(t))
    if t.ndim == 0:
        return complex(g[0])
    return g


_MAX_EXACT: Final[int] = 1 << 53


def sample_times_s(k0: int, n: int, f_s: Fraction | int | str) -> NDArray[np.float64]:
    """Absolute times ``k/f_s`` (s) of samples ``k0 .. k0+n-1`` of a stream at the exact rate
    ``f_s``: each value is the correctly rounded float64 of the exact rational ``k*den/num``
    (one division of exactly representable integers), so it depends on ``k`` and ``f_s`` only
    (never on the chunking). Needs ``k*den < 2**53``."""
    if isinstance(k0, bool) or not isinstance(k0, numbers.Integral) or int(k0) < 0:
        raise ConfigurationError(f"k0 must be an int >= 0, got {k0!r}")
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or int(n) < 0:
        raise ConfigurationError(f"n must be an int >= 0, got {n!r}")
    rate = parse_exact(f_s, name="f_s")
    if rate <= 0:
        raise ConfigurationError(f"f_s must be > 0, got {f_s!r}")
    num, den = rate.numerator, rate.denominator
    k_end = int(k0) + int(n)
    if k_end * den > _MAX_EXACT or num > _MAX_EXACT:
        raise ConfigurationError("sample index too large for exact float64 sample times")
    idx = np.arange(int(k0), k_end, dtype=np.int64)
    return (idx * den).astype(np.float64) / float(num)


def sos_gain_samples(
    key: _rng.PhiloxKey | tuple[int, int],
    k0: int,
    n: int,
    f_s: Fraction | int | str,
    max_doppler_hz: float,
    n_sinusoids: int = MIN_SINUSOIDS,
    *,
    tap: int = 0,
) -> NDArray[np.complex128]:
    """:func:`sos_gain` at the absolute samples ``k0 .. k0+n-1`` of a stream of exact rate
    ``f_s`` (the renderer's view, like ``noise(rx_key, k0, n, sigma)``): a pure function of
    the absolute index (:func:`sample_times_s`)."""
    g = sos_gain(key, sample_times_s(k0, n, f_s), max_doppler_hz, n_sinusoids, tap=tap)
    assert isinstance(g, np.ndarray)
    return g


# --------------------------------------------------------------------------- per-link terms


@dataclass(frozen=True)
class DiffuseTap:
    """One tap: delay relative to the LOS (s) and relative power (dB)."""

    delay_s: float = 0.0
    rel_power_db: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.delay_s) and self.delay_s >= 0.0):
            raise ConfigurationError(
                f"diffuse tap delay_s must be finite and >= 0, got {self.delay_s}"
            )
        if not math.isfinite(self.rel_power_db):
            raise ConfigurationError("diffuse tap rel_power_db must be finite")


@dataclass(frozen=True)
class DiffuseTerm:
    """The diffuse contribution of one (link, tap) at one snapshot time.

    ``amp``: ``sqrt(P_tap)`` (Friis-normalized amplitude, like ``PathBundle.amp``);
    ``delay_s``: absolute envelope delay ``tau_LOS + tap delay``. The RX contribution is
    ``amp * sos_gain(key, t, max_doppler_hz, n_sinusoids, tap=tap) * x(t - delay_s)`` with
    no extra carrier phase (module docstring)."""

    link: str
    tap: int
    key: _rng.PhiloxKey
    amp: float
    delay_s: float
    max_doppler_hz: float
    n_sinusoids: int

    def gain(self, t_s: ArrayLike) -> NDArray[np.complex128] | complex:
        """``amp * g(t)`` at absolute times ``t_s``."""
        g = sos_gain(self.key, t_s, self.max_doppler_hz, self.n_sinusoids, tap=self.tap)
        return self.amp * g

    def gain_samples(self, k0: int, n: int, f_s: Fraction | int | str) -> NDArray[np.complex128]:
        """``amp * g`` at absolute samples ``k0 .. k0+n-1`` of a stream of rate ``f_s``."""
        g = sos_gain_samples(
            self.key, k0, n, f_s, self.max_doppler_hz, self.n_sinusoids, tap=self.tap
        )
        return self.amp * g


def reference_power(bundle: PathBundle) -> float:
    """K-factor reference: ``sum |amp|^2`` of the bundle's LOS, GROUND, WALL and DIFF paths."""
    sel = np.isin(bundle.kind, np.asarray(DETERMINISTIC_KINDS, dtype=np.int8))
    a = bundle.amp[sel]
    return float(np.sum(a.real * a.real + a.imag * a.imag))


def _reference_delay(bundle: PathBundle) -> float:
    los = np.flatnonzero(bundle.kind == PathKind.LOS)
    if los.size:
        return float(bundle.delay_s[los[0]])
    if bundle.n_paths == 0:
        raise ConfigurationError(f"link {bundle.link}: no paths to reference the diffuse delay")
    return float(np.min(bundle.delay_s))


@dataclass(eq=False)
class DiffuseMechanism:
    """Diffuse-term provider (section 5.11; scenario ``environment.diffuse``).

    It has none of the WP2 path hooks (the term is stochastic and additive, not a
    deterministic path): :func:`sremu.physics.propagation.from_scenario` keeps it in
    ``ScenePropagation.mechanisms`` (:func:`find_diffuse`) and the renderer asks it for
    :meth:`terms` per snapshot.
    """

    k_factor_db: float
    max_doppler_hz: float | None = None
    from_motion: bool = False
    n_sinusoids: int = MIN_SINUSOIDS
    taps: tuple[DiffuseTap, ...] = (DiffuseTap(),)
    seed: int = 0
    _fd_cache: weakref.WeakKeyDictionary[Any, dict[str, float]] = field(
        default_factory=weakref.WeakKeyDictionary, repr=False
    )

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.k_factor_db)):
            raise ConfigurationError("diffuse k_factor_db must be finite")
        if (self.max_doppler_hz is None) == (not self.from_motion):
            raise ConfigurationError("diffuse: give exactly one of max_doppler_hz or from_motion")
        if self.max_doppler_hz is not None:
            self.max_doppler_hz = _check_fd(self.max_doppler_hz)
        self.n_sinusoids = _check_m(self.n_sinusoids)
        self.taps = tuple(self.taps)
        if not 1 <= len(self.taps) <= MAX_TAPS:
            raise ConfigurationError(f"diffuse: 1 to {MAX_TAPS} taps, got {len(self.taps)}")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, numbers.Integral)
            or self.seed < 0
        ):
            raise ConfigurationError(f"seed must be an int >= 0, got {self.seed!r}")

    @property
    def k_linear(self) -> float:
        """K = 10^(k_factor_db/10)."""
        return 10.0 ** (float(self.k_factor_db) / 10.0)

    @property
    def tap_weights(self) -> NDArray[np.float64]:
        """Tap power fractions (sum 1)."""
        w = np.array([10.0 ** (t.rel_power_db / 10.0) for t in self.taps], dtype=np.float64)
        return w / float(np.sum(w))

    def key(self, link: str) -> _rng.PhiloxKey:
        """Philox key of the link's diffuse processes (purpose ``diffuse``, name ``link``)."""
        return _rng.derive_key(int(self.seed), "diffuse", str(link))

    def doppler_hz(self, model: Any, link: str) -> float:
        """Maximum Doppler of the link's SoS: given, or ``v_rel,max/lambda`` (from motion)."""
        if self.max_doppler_hz is not None:
            return float(self.max_doppler_hz)
        try:
            cache = self._fd_cache.get(model)
        except TypeError:  # not weak-referenceable: no caching
            cache = None
        if cache is None:
            mb = model.motion_bounds()
            cache = {
                lk: float(mb.v_rel_max_mps.get(lk, 0.0)) * float(model.link_fc_hz(lk)) / C0
                for lk in model.links()
            }
            with contextlib.suppress(TypeError):  # not weak-referenceable: no caching
                self._fd_cache[model] = cache
        return cache[str(link)]

    def link_terms(self, bundle: PathBundle, max_doppler_hz: float) -> tuple[DiffuseTerm, ...]:
        """The diffuse terms of one link at the bundle's snapshot time."""
        p_ref = reference_power(bundle)
        tau0 = _reference_delay(bundle)
        key = self.key(bundle.link)
        w = self.tap_weights
        fd = _check_fd(max_doppler_hz)
        return tuple(
            DiffuseTerm(
                bundle.link,
                i,
                key,
                math.sqrt(p_ref * float(w[i]) / self.k_linear),
                tau0 + float(tap.delay_s),
                fd,
                self.n_sinusoids,
            )
            for i, tap in enumerate(self.taps)
        )

    def terms(self, model: Any, snap: Any) -> dict[str, tuple[DiffuseTerm, ...]]:
        """Per link of ``snap.paths``: the diffuse terms at the snapshot time ``snap.t``."""
        return {
            lk: self.link_terms(pb, self.doppler_hz(model, lk)) for lk, pb in snap.paths.items()
        }

    def describe(self) -> dict[str, Any]:
        """JSON-ready description (manifests)."""
        return {
            "mechanism": "diffuse_sos",
            "k_factor_db": float(self.k_factor_db),
            "max_doppler_hz": self.max_doppler_hz,
            "from_motion": bool(self.from_motion),
            "n_sinusoids": int(self.n_sinusoids),
            "taps": [{"delay_s": t.delay_s, "rel_power_db": t.rel_power_db} for t in self.taps],
            "reference": "incoherent power of LOS/GROUND/WALL/DIFF paths [ENG]",
            "rng": "sremu.rng purpose 'diffuse', name <link>; lanes 2*tap, 2*tap+1",
        }


def find_diffuse(model: Any) -> DiffuseMechanism | None:
    """The :class:`DiffuseMechanism` attached to a ``ScenePropagation`` (or None)."""
    for mech in getattr(model, "mechanisms", ()):
        if isinstance(mech, DiffuseMechanism):
            return mech
    return None


def mechanism_from_scenario(
    spec: Mapping[str, Any] | None,
    scenario: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> DiffuseMechanism | None:
    """D18 factory: the diffuse provider of ``environment.diffuse`` (None when absent)."""
    del base_dir
    if not spec:
        return None
    taps_spec: Sequence[Mapping[str, Any]] | None = spec.get("taps")
    taps = (
        tuple(
            DiffuseTap(float(t.get("delay_s", 0.0)), float(t.get("rel_power_db", 0.0)))
            for t in taps_spec
        )
        if taps_spec
        else (DiffuseTap(),)
    )
    fd = spec.get("max_doppler_hz")
    return DiffuseMechanism(
        k_factor_db=float(spec["k_factor_db"]),
        max_doppler_hz=None if fd is None else float(fd),
        from_motion=bool(spec.get("from_motion", False)),
        n_sinusoids=int(spec.get("n_sinusoids", MIN_SINUSOIDS)),
        taps=taps,
        seed=int(scenario.get("seed", 0)),
    )
