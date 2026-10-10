"""Exact finite-alphabet optimum and the continuous optimum (DESIGN.md section 5.13; WP2).

``exact_alphabet_optimum(z0, Z[S, N])`` maximizes ``|z0 + sum_n Z[s_n, n]|`` exactly.

Principle: for a direction theta, the best per-element choice is ``s_n(theta) = argmax_s
Re{Z[s, n] exp(-j theta)}``. It changes only at the critical angles
``arg(Z[a, n] - Z[b, n]) +- pi/2`` (S(S-1)N of them). The optimum ``s*`` equals
``s(theta*)`` with ``theta* = arg(total*)``: otherwise switching one element would raise
the projection onto ``theta*`` and with it ``|total|``. So it suffices to evaluate one
configuration per arc between consecutive critical angles.

* :func:`exact_alphabet_optimum`: sweep over the sorted critical angles with incremental
  updates, O(S^2 N log(S^2 N)).
* :func:`exact_alphabet_optimum_reference`: the same candidate set, every arc evaluated from
  scratch, O(S^3 N^2); kept for tests.

Not claimed as novel: cite prior optimal discrete-phase beamforming work (section 5.13).
"""

from __future__ import annotations

import math

import numpy as np

from sremu.errors import ConfigurationError

__all__ = [
    "continuous_optimum",
    "critical_angles",
    "exact_alphabet_optimum",
    "exact_alphabet_optimum_reference",
    "per_element_choice",
]

_TWO_PI = 2.0 * math.pi


def _check(z0: complex, z: np.ndarray) -> np.ndarray:
    zz = np.asarray(z, dtype=np.complex128)
    if zz.ndim != 2 or zz.shape[0] < 1 or zz.shape[1] < 1:
        raise ConfigurationError("Z must be (S, N) with S, N >= 1")
    if not (np.all(np.isfinite(zz)) and math.isfinite(abs(complex(z0)))):
        raise ConfigurationError("z0 and Z must be finite")
    return zz


def per_element_choice(z: np.ndarray, theta: float) -> np.ndarray:
    """``s_n(theta) = argmax_s Re{Z[s, n] exp(-j theta)}`` (lowest index on ties), int64."""
    c, s = math.cos(theta), math.sin(theta)
    return np.argmax(z.real * c + z.imag * s, axis=0)


def critical_angles(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Critical angles in [0, 2 pi) and their element indices (unsorted)."""
    zz = np.asarray(z, dtype=np.complex128)
    s_count = zz.shape[0]
    angs, elems = [], []
    for a in range(s_count):
        for b in range(a + 1, s_count):
            d = zz[a] - zz[b]
            ok = d != 0
            base = np.angle(d[ok])
            idx = np.nonzero(ok)[0]
            for off in (0.5 * math.pi, -0.5 * math.pi):
                angs.append(np.mod(base + off, _TWO_PI))
                elems.append(idx)
    if not angs:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    ang = np.concatenate(angs)
    ang = np.where(ang >= _TWO_PI, ang - _TWO_PI, ang)
    return ang, np.concatenate(elems).astype(np.int64)


def _arcs(ang: np.ndarray) -> np.ndarray:
    """Distinct sorted angles."""
    return np.unique(ang)


def _value(z0: complex, z: np.ndarray, s: np.ndarray) -> complex:
    return complex(z0) + complex(np.sum(z[s, np.arange(z.shape[1])]))


def exact_alphabet_optimum(z0: complex, z: np.ndarray) -> tuple[float, np.ndarray]:
    """Maximize ``|z0 + sum_n Z[s_n, n]|`` over ``s`` in {0..S-1}^N exactly.

    Returns ``(value, s*)`` with ``s*`` (N,) int16. Sweep over the sorted critical angles:
    one configuration per arc, updated incrementally (only the elements whose critical angle
    is crossed change). The returned value is recomputed from ``s*`` with a fresh sum.
    """
    zz = _check(z0, z)
    n = zz.shape[1]
    ang, elem = critical_angles(zz)
    if ang.size == 0:  # every element has identical states
        s = np.zeros(n, dtype=np.int64)
        return abs(_value(z0, zz, s)), s.astype(np.int16)
    order = np.argsort(ang, kind="stable")
    ang_s, elem_s = ang[order], elem[order]
    cuts = np.nonzero(np.diff(ang_s) > 0)[0] + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [ang_s.size]])
    uniq = ang_s[starts]
    m = uniq.size
    mids = np.empty(m)
    mids[:-1] = 0.5 * (uniq[:-1] + uniq[1:])
    mids[-1] = 0.5 * (uniq[-1] + uniq[0] + _TWO_PI)
    # arc m-1 (wrapping) is the starting configuration
    s = per_element_choice(zz, float(mids[-1]))
    cols = np.arange(n)
    total = complex(z0) + complex(np.sum(zz[s, cols]))
    best_val = abs(total)
    best_arc = m - 1
    re, im = zz.real, zz.imag
    for k in range(m - 1):  # crossing uniq[k] enters arc k
        th = float(mids[k])
        c, sn = math.cos(th), math.sin(th)
        idx = np.unique(elem_s[starts[k] : ends[k]])  # an element may repeat at one angle
        new = np.argmax(re[:, idx] * c + im[:, idx] * sn, axis=0)
        old = s[idx]
        changed = new != old
        if np.any(changed):
            ii = idx[changed]
            total += complex(np.sum(zz[new[changed], ii] - zz[old[changed], ii]))
            s[ii] = new[changed]
        v = abs(total)
        if v > best_val:
            best_val, best_arc = v, k
    s_best = per_element_choice(zz, float(mids[best_arc]))
    return abs(_value(z0, zz, s_best)), s_best.astype(np.int16)


def exact_alphabet_optimum_reference(z0: complex, z: np.ndarray) -> tuple[float, np.ndarray]:
    """Reference O(S^3 N^2) version: every arc's configuration is evaluated from scratch."""
    zz = _check(z0, z)
    n = zz.shape[1]
    ang, _ = critical_angles(zz)
    if ang.size == 0:
        s = np.zeros(n, dtype=np.int64)
        return abs(_value(z0, zz, s)), s.astype(np.int16)
    uniq = _arcs(ang)
    mids = np.concatenate([0.5 * (uniq[:-1] + uniq[1:]), [0.5 * (uniq[-1] + uniq[0] + _TWO_PI)]])
    best_val, best_s = -1.0, np.zeros(n, dtype=np.int64)
    for th in mids:
        s = per_element_choice(zz, float(th))
        v = abs(_value(z0, zz, s))
        if v > best_val:
            best_val, best_s = v, s
    return best_val, best_s.astype(np.int16)


def continuous_optimum(h0: complex, c: np.ndarray) -> tuple[float, np.ndarray]:
    """Unit-amplitude continuous-phase optimum: ``phi_n = arg(h0) - arg(c_n)``.

    Returns ``(|h0| + sum_n |c_n|, phases_rad (N,))``; with ``h0 = 0`` the reference phase
    is 0.
    """
    cc = np.asarray(c, dtype=np.complex128)
    ref = math.atan2(complex(h0).imag, complex(h0).real) if h0 != 0 else 0.0
    phases = np.mod(ref - np.angle(cc), _TWO_PI)
    return abs(complex(h0)) + float(np.sum(np.abs(cc))), phases
