"""Fractional-delay (FD) filters (DESIGN.md section 4.5; PH-20).

Normative choices:

* **Filter.** Kaiser-windowed sinc with ``L`` taps and ``beta = 8``. For a fractional
  part ``mu`` in [0, 1) the taps are ``h[m] = sinc(m - D) * w(m - D)``, ``m = 0..L-1``, with
  ``D = c + mu`` and ``c = (L - 1)//2``. The window is evaluated at the fractional offset,
  ``w(x) = I0(beta*sqrt(1 - (2x/L)**2))/I0(beta)`` for ``|2x/L| <= 1`` and 0 elsewhere (the
  design of ``research/scratch/design_checks/fd_check.py``). ``y[n] = sum_m h[m] x[n - m]``
  approximates ``x(n - D)``.
* **Polyphase table.** The filter actually applied by both backends is a table of
  ``n_phases + 1 = 1025`` rows (``mu = j/1024``, ``j = 0..1024``) with **linear interpolation
  between adjacent rows**: ``h(mu) = T[j] + alpha*(T[j+1] - T[j])``, ``j = floor(mu*1024)``,
  ``alpha = mu*1024 - j``. The table is the filter's definition; ``numpy_ref`` and
  ``numba_cpu`` use the same rows and the same interpolation formula (I5).
* **Choice by declared occupied bandwidth B** (spec table, default target -60 dB):
  ``B/f_s <= 0.2`` -> 8 taps, ``<= 0.6`` -> 16 taps, ``<= 0.8`` -> 32 taps (lookahead
  ``h_FD = L/2``), ``> 0.8`` -> :class:`~sremu.errors.FDBandwidthExceeded`. Integer-only delay
  (rounded to the nearest sample) is used only when ``pi*B/(2 f_s) <= 0.05``, i.e.
  ``B <= 0.0318 f_s`` (tones and CW).
* **Error metric.** ``max_{|f| <= B/2} |H(f) - exp(-j 2 pi f D)|`` in dB (f normalized to
  f_s), worst case over ``mu``.

The 4-tap cubic Lagrange interpolator of the input documents is deliberately absent (it
fails the metric: about -5.2 dB at +-0.4 f_s); the PH-20 teeth test builds it locally.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction
from functools import lru_cache
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import FDSpec
from sremu.errors import ConfigurationError, FDBandwidthExceeded

__all__ = [
    "DEFAULT_FD_TARGET_DB",
    "FD_ROWS",
    "INTEGER_MAX_PHASE_RAD",
    "KAISER_BETA",
    "N_PHASES",
    "choose_fd",
    "fd_center",
    "inband_error_db",
    "integer_delay_allowed",
    "interp_taps",
    "kaiser_fd_spec",
    "kaiser_sinc_taps",
    "polyphase_table",
    "table_for",
    "worst_case_error_db",
]

KAISER_BETA: Final[float] = 8.0
"""Kaiser window beta of every release-1 FD filter."""

N_PHASES: Final[int] = 1024
"""Number of fractional phases of the polyphase table (rows j/1024, plus the row at mu=1)."""

DEFAULT_FD_TARGET_DB: Final[float] = -60.0
"""Default in-band error target ``fd_error_target_db`` (section 4.5)."""

FD_ROWS: Final[tuple[tuple[float, int], ...]] = ((0.2, 8), (0.6, 16), (0.8, 32))
"""(max occupied band B/f_s, taps) rows of the section 4.5 table (beta = 8)."""

INTEGER_MAX_PHASE_RAD: Final[float] = 0.05
"""Integer-only delay is allowed when ``pi*B/(2 f_s) <= 0.05`` rad (half-sample error)."""

_CANDIDATE_TAPS: Final[tuple[int, ...]] = (8, 16, 32)


def fd_center(taps: int) -> int:
    """Integer centre ``c = (L - 1)//2`` of an ``L``-tap FD filter (D = c + mu)."""
    if taps < 1:
        raise ConfigurationError(f"FD filter needs >= 1 tap, got {taps}")
    return (taps - 1) // 2


def kaiser_sinc_taps(
    mu: float | NDArray[np.float64], taps: int, beta: float = KAISER_BETA
) -> NDArray[np.float64]:
    """Direct Kaiser-windowed sinc taps for fractional delay ``mu`` (scalar or array).

    Returns shape ``(taps,)`` for a scalar ``mu`` and ``(len(mu), taps)`` for an array.
    The filter delay is ``D = (taps - 1)//2 + mu`` samples.
    """
    mu_arr = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    m = np.arange(taps, dtype=np.float64)
    x = m[None, :] - (fd_center(taps) + mu_arr[:, None])
    t = 2.0 * x / taps
    arg = np.sqrt(np.clip(1.0 - t * t, 0.0, 1.0))
    w = np.where(np.abs(t) <= 1.0, np.i0(beta * arg) / np.i0(beta), 0.0)
    h = np.sinc(x) * w
    return h[0] if np.ndim(mu) == 0 else h


@lru_cache(maxsize=32)
def polyphase_table(
    taps: int, beta: float = KAISER_BETA, n_phases: int = N_PHASES
) -> NDArray[np.float64]:
    """Read-only polyphase table, shape ``(n_phases + 1, taps)``: row j is ``mu = j/n_phases``."""
    if n_phases < 1:
        raise ConfigurationError(f"n_phases must be >= 1, got {n_phases}")
    mus = np.arange(n_phases + 1, dtype=np.float64) / n_phases
    table = np.ascontiguousarray(kaiser_sinc_taps(mus, taps, beta))
    table.setflags(write=False)
    return table


def table_for(spec: FDSpec) -> NDArray[np.float64]:
    """Polyphase table of an FD spec (``integer`` specs have no table)."""
    if spec.filter_id == "integer":
        raise ConfigurationError("integer-only delay has no polyphase table")
    if spec.beta is None:
        raise ConfigurationError(f"FD spec {spec.filter_id!r} has no Kaiser beta")
    return polyphase_table(int(spec.taps), float(spec.beta), int(spec.n_phases))


def interp_taps(table: NDArray[np.float64], mu: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Taps at fractional delay ``mu`` by linear interpolation between table rows.

    Uses exactly the formula of both render backends:
    ``j = floor(mu*n_phases)``, ``alpha = mu*n_phases - j``,
    ``h = T[j] + alpha*(T[j+1] - T[j])``.
    """
    n_phases = table.shape[0] - 1
    mu_arr = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    if np.any(mu_arr < 0.0) or np.any(mu_arr >= 1.0):
        raise ValueError("mu must lie in [0, 1)")
    jf = mu_arr * n_phases
    j = np.minimum(np.floor(jf).astype(np.int64), n_phases - 1)
    alpha = jf - j
    lo = table[j]
    hi = table[j + 1]
    h = lo + alpha[:, None] * (hi - lo)
    return h[0] if np.ndim(mu) == 0 else h


def kaiser_fd_spec(taps: int, beta: float = KAISER_BETA, n_phases: int = N_PHASES) -> FDSpec:
    """FD spec of an ``L``-tap Kaiser filter; lookahead ``h_FD = L/2`` (section 4.5)."""
    if taps < 2:
        raise ConfigurationError(f"a Kaiser FD filter needs >= 2 taps, got {taps}")
    return FDSpec(
        filter_id=f"kaiser{taps}b{beta:g}",
        taps=int(taps),
        lookahead=int(taps) // 2,
        beta=float(beta),
        n_phases=int(n_phases),
    )


def integer_delay_allowed(occupied_bw_hz: float, f_s: Fraction | int | float) -> bool:
    """Whether integer-only delay is allowed: ``pi*B/(2 f_s) <= 0.05`` (B <= 0.0318 f_s)."""
    return math.pi * float(occupied_bw_hz) / (2.0 * float(f_s)) <= INTEGER_MAX_PHASE_RAD


def inband_error_db(
    taps_fn: Callable[[float], NDArray[np.float64]],
    delay_fn: Callable[[float], float],
    band_frac: float,
    mus: NDArray[np.float64],
    n_freq: int = 801,
) -> float:
    """Worst in-band error in dB over the given fractional delays.

    ``taps_fn(mu)`` returns the taps, ``delay_fn(mu)`` the ideal delay D in samples, and the
    band is ``|f| <= band_frac`` (normalized to f_s). Returns
    ``20*log10(max_mu max_f |H(f) - exp(-j 2 pi f D)|)``.
    """
    f = np.linspace(-band_frac, band_frac, n_freq)
    worst = 0.0
    for mu in np.asarray(mus, dtype=np.float64):
        h = np.asarray(taps_fn(float(mu)), dtype=np.float64)
        m = np.arange(h.shape[0], dtype=np.float64)
        resp = np.exp(-2j * np.pi * f[:, None] * m[None, :]) @ h
        ideal = np.exp(-2j * np.pi * f * delay_fn(float(mu)))
        worst = max(worst, float(np.max(np.abs(resp - ideal))))
    return 20.0 * math.log10(worst) if worst > 0.0 else -math.inf


def worst_case_error_db(spec: FDSpec, band_frac: float, n_mu: int = 257) -> float:
    """Worst in-band error (dB) of the table-interpolated filter of ``spec``.

    The fractional delays include every 4th table row and the midpoints between rows,
    where linear interpolation errs most.
    """
    table = table_for(spec)
    c = fd_center(spec.taps)
    grid = np.linspace(0.0, 1.0, n_mu, endpoint=False)
    mids = (np.floor(grid * spec.n_phases) + 0.5) / spec.n_phases
    mus = np.unique(np.concatenate([grid, mids]))
    return inband_error_db(lambda mu: interp_taps(table, mu), lambda mu: c + mu, band_frac, mus)


@lru_cache(maxsize=64)
def _row_error_db(taps: int, band_frac: float) -> float:
    return worst_case_error_db(kaiser_fd_spec(taps), band_frac, n_mu=65)


def choose_fd(
    occupied_bw_hz: float,
    f_s: Fraction | int | float,
    target_db: float = DEFAULT_FD_TARGET_DB,
    *,
    allow_integer: bool = True,
) -> FDSpec:
    """FD filter for a TX stream with declared occupied bandwidth ``B`` (section 4.5).

    Integer-only delay when allowed (``B <= 0.0318 f_s``); otherwise the section 4.5 table
    row. With the default target the table is used as is. For a stricter target, a row whose
    computed worst-case in-band error exceeds the target steps up to the next larger filter.
    Raises :class:`FDBandwidthExceeded` when ``B > 0.8 f_s`` or no filter meets the target.
    """
    b = float(occupied_bw_hz)
    fs = float(f_s)
    if not (b >= 0.0 and math.isfinite(b)):
        raise ConfigurationError(f"occupied_bw_hz must be finite and >= 0, got {occupied_bw_hz!r}")
    if not fs > 0.0:
        raise ConfigurationError(f"sample rate must be > 0, got {f_s!r}")
    if allow_integer and integer_delay_allowed(b, fs):
        return FDSpec.integer()
    frac = b / fs
    row = next((t for max_frac, t in FD_ROWS if frac <= max_frac), None)
    if row is None:
        raise FDBandwidthExceeded(
            f"occupied band B/f_s = {frac:.4f} > 0.8: no release-1 FD filter reaches "
            f"{target_db:g} dB (48 taps reach only about -57 dB at +-0.45 f_s)"
        )
    taps = row
    if target_db == DEFAULT_FD_TARGET_DB:
        return kaiser_fd_spec(taps)
    for cand in _CANDIDATE_TAPS:
        if cand < taps:
            continue
        if _row_error_db(cand, round(frac / 2.0, 9)) <= target_db:
            return kaiser_fd_spec(cand)
    raise FDBandwidthExceeded(
        f"no FD filter with <= {_CANDIDATE_TAPS[-1]} taps reaches {target_db:g} dB "
        f"for B/f_s = {frac:.4f}"
    )
