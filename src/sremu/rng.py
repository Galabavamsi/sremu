"""Keyed counter-based random numbers (DESIGN.md sections 6.1, 6.2; invariants I4, I5).

Scheme ``"philox4x32-10/blake2b-v1"`` (recorded in every manifest as ``rng_scheme``):

* **Key.** The first 64 bits of ``blake2b(master_seed || purpose || name)`` with the default
  64-byte BLAKE2b digest, where ``master_seed`` is the decimal text of the integer seed, every
  part is UTF-8 and ``||`` is the separator byte ``0x1F``. The 8 bytes are read
  little-endian: bytes 0..3 give key word ``k0``, bytes 4..7 give ``k1``. The 64-bit key
  integer is ``k0 | k1 << 32``.
* **Counter** (Philox4x32-10): ``(k_lo, k_hi, sub, 0)``, where ``k`` is the **absolute**
  sample or step index (``0 <= k < 2**63``) and ``sub`` is a purpose-specific lane
  (``0 <= sub < 2**32``).
* **Variates.** Output words ``r0, r1`` feed one draw; ``r2, r3`` are reserved.

  - ``uniform``: ``((r0 >> 5)*2**26 + (r1 >> 6)) * 2**-53`` in [0, 1) (53-bit resolution).
  - ``complex_normal`` (Box-Muller): ``u1 = (r0 + 1)*2**-32`` in (0, 1],
    ``u2 = r1*2**-32`` in [0, 1); ``z = sigma*sqrt(-ln u1)*(cos 2 pi u2 + j sin 2 pi u2)``,
    so ``E|z|**2 = sigma**2`` (each of Re, Im has variance ``sigma**2/2``). The 32-bit ``u1``
    bounds ``|z| <= sigma*sqrt(32 ln 2) = 4.71 sigma``; the truncated tail has probability
    ``2**-32``.
  - ``standard_normal``: ``sqrt(-2 ln u1)*cos(2 pi u2)``, N(0, 1).

Every variate is a pure function of ``(master_seed, purpose, name, sub, absolute index)``
(I4): splitting a range into chunks, skipping ranges, the thread count and adding or
removing other nodes never change it.

Two backends compute the same streams:

* ``numpy_ref``: a vectorized NumPy twin (32x32 -> 64-bit products in ``uint64``, which is
  exact, then split into high and low words);
* ``numba_cpu``: Philox4x32-10 in numba (``parallel=True`` over samples, no fastmath, D-4),
  compiled lazily on first use and cached on disk.

The uint32 streams of the two backends are bitwise identical (I5); so are the uniforms,
which use integer operations and one exact scaling. The normals go through ``log``, ``cos``
and ``sin``, whose last-ulp behaviour depends on the math library (NumPy's SIMD kernels vs
the C runtime used by numba); they agree to within a few ulps. (On the development laptop,
an AVX2 host, they were bitwise equal over 2e6 draws; this is not claimed elsewhere.)

Every keyed ``np.random.Generator`` (random-waypoint legs, failure maps, controllers,
trials; D14) comes from :func:`generator`:
``np.random.Generator(np.random.Philox(key=derive_key(seed, purpose, name).u64))``; an
optional ``counter`` selects independent per-index streams of one key (e.g. random-waypoint
legs).
``SeedSequence.spawn`` is never used (it depends on spawn order) and there is no global RNG.
"""

from __future__ import annotations

import functools
import hashlib
import numbers
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Any, Final, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from sremu.errors import ConfigurationError

__all__ = [
    "MAX_INDEX",
    "PURPOSES",
    "PURPOSE_REGISTRY_VERSION",
    "RNG_SCHEME",
    "SEPARATOR",
    "Backend",
    "PhiloxKey",
    "complex_normal",
    "derive_key",
    "generator",
    "key_material",
    "philox4x32",
    "philox_block",
    "standard_normal",
    "uniform",
]

RNG_SCHEME: Final[str] = "philox4x32-10/blake2b-v1"
"""Identifier of this scheme, recorded in the manifest (section 3.6)."""

PURPOSE_REGISTRY_VERSION: Final[int] = 1
"""Version of :data:`PURPOSES`. Changing a purpose string changes every derived stream."""

PURPOSES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "noise": "AWGN per receiver; name = <rx_stream>",
        "pn": "Wiener phase noise per LO; name = <node>.<lo>",
        "diffuse": "diffuse sum-of-sinusoids multipath per link; name = <link>",
        "mobility": "random-waypoint legs per node; name = <node>",
        "failure": "random failure maps per panel; name = <panel>",
        "table_mc": "Monte Carlo perturbed response tables per panel; name = <panel>",
        "ctrl.latency": "controller latency jitter; name = <ctrl>",
        "ctrl": "controller-private generator; name = <ctrl>",
        "trial": "Monte Carlo trials; name = <study>/<i>",
        "decoy": "validation decoys; name = <i>",
        "traffic": "in-process traffic generators; name = <stream>",
    }
)
"""Purpose registry (section 6.2): purpose string -> meaning of ``name``."""

SEPARATOR: Final[bytes] = b"\x1f"
"""Separator byte between seed, purpose and name in the key material."""

MAX_INDEX: Final[int] = 1 << 63
"""Exclusive upper bound of absolute indices (int64 in kernels, section 3.5)."""

Backend = Literal["numpy_ref", "numba_cpu"]

_MASK32: Final[int] = 0xFFFFFFFF
_M0: Final[int] = 0xD2511F53
_M1: Final[int] = 0xCD9E8D57
_W0: Final[int] = 0x9E3779B9
_W1: Final[int] = 0xBB67AE85
_ROUNDS: Final[int] = 10
_TWO_M32: Final[float] = 2.3283064365386963e-10  # 2**-32, exact in float64
_TWO_M53: Final[float] = 1.1102230246251565e-16  # 2**-53, exact in float64
_TWO_PI: Final[float] = 6.283185307179586  # float64(2*pi)
_CHUNK: Final[int] = 1 << 18  # NumPy twin works in chunks to bound temporary memory


# --------------------------------------------------------------------------- keys


class PhiloxKey(NamedTuple):
    """Two uint32 Philox key words. It is a ``tuple[int, int]`` (the ``rx_key`` of the
    render contract's ``noise()``)."""

    k0: int
    k1: int

    @property
    def u64(self) -> int:
        """The 64-bit key integer ``k0 | k1 << 32`` (the NumPy ``Philox(key=...)`` value)."""
        return self.k0 | (self.k1 << 32)

    @classmethod
    def from_u64(cls, key: int) -> PhiloxKey:
        """Split a 64-bit key integer into its two words."""
        if isinstance(key, bool) or not isinstance(key, numbers.Integral):
            raise TypeError(f"key must be an int, got {type(key).__name__}")
        key = int(key)
        if not 0 <= key < (1 << 64):
            raise ValueError(f"key must be in [0, 2**64), got {key}")
        return cls(key & _MASK32, key >> 32)


def _check_seed(master_seed: int) -> int:
    if isinstance(master_seed, bool) or not isinstance(master_seed, numbers.Integral):
        raise TypeError(f"master_seed must be an int, got {type(master_seed).__name__}")
    seed = int(master_seed)
    if seed < 0:
        raise ValueError(f"master_seed must be >= 0, got {seed}")
    return seed


def key_material(master_seed: int, purpose: str, name: str, *, strict: bool = True) -> bytes:
    """The hashed bytes ``seed || purpose || name`` (UTF-8, ``0x1F`` separators).

    With ``strict`` (the default) ``purpose`` must be in :data:`PURPOSES`. Neither part may
    contain the separator byte, so the encoding is injective.
    """
    seed = _check_seed(master_seed)
    if not isinstance(purpose, str) or not isinstance(name, str):
        raise TypeError("purpose and name must be str")
    if strict and purpose not in PURPOSES:
        raise ConfigurationError(
            f"unknown RNG purpose {purpose!r}; registered purposes: {sorted(PURPOSES)}"
        )
    if not purpose:
        raise ValueError("purpose must be non-empty")
    if "\x1f" in purpose or "\x1f" in name:
        raise ValueError("purpose and name must not contain the separator byte 0x1F")
    return SEPARATOR.join((str(seed).encode("utf-8"), purpose.encode("utf-8"), name.encode()))


def derive_key(master_seed: int, purpose: str, name: str, *, strict: bool = True) -> PhiloxKey:
    """Philox key of stream ``(master_seed, purpose, name)`` (section 6.1).

    The first 8 bytes of the 64-byte BLAKE2b digest of :func:`key_material`, read
    little-endian into ``(k0, k1)``.
    """
    digest = hashlib.blake2b(key_material(master_seed, purpose, name, strict=strict)).digest()
    return PhiloxKey(int.from_bytes(digest[0:4], "little"), int.from_bytes(digest[4:8], "little"))


def generator(
    master_seed: int, purpose: str, name: str, *, counter: int = 0
) -> np.random.Generator:
    """Keyed NumPy generator of stream ``(master_seed, purpose, name)`` (section 6.1; D14).

    This is the one helper every keyed ``np.random.Generator`` in sremu comes from:
    random-waypoint legs (``"mobility"``, ``<node>``), failure maps (``"failure"``,
    ``<panel>``), Monte Carlo tables (``"table_mc"``, ``<panel>``), controllers
    (``"ctrl"``, ``<ctrl>``), trials (``"trial"``, ``"<study>/<i>"``) and the other
    purposes of :data:`PURPOSES`. Packages must not build their own key (no private hash
    helpers, no ``SeedSequence``): the result is exactly::

        np.random.Generator(np.random.Philox(
            key=derive_key(master_seed, purpose, name).u64,
            counter=np.array([0, counter, 0, 0], dtype=np.uint64)))

    Args:
        master_seed: the scenario ``seed`` (int >= 0).
        purpose: a registered purpose string (:data:`PURPOSES`; an unknown purpose raises
            ``ConfigurationError``).
        name: the purpose-specific name (node, panel, controller, ...), section 6.2.
        counter: ``0 <= counter < 2**64`` selects an independent stream of the same key in
            the second 64-bit Philox counter word (NumPy's Philox advances the first), e.g.
            random-waypoint leg ``i`` of ``mobility/<node>``. Default 0.

    The stream depends only on these four values, never on creation order or on other
    generators (never spawned, section 6.1). The raw Philox stream is fixed by the key and
    counter; values drawn through ``Generator`` methods follow NumPy's stream-compatibility
    policy (NEP 19), so the NumPy version is recorded in the manifest.
    """
    if isinstance(counter, bool) or not isinstance(counter, numbers.Integral):
        raise TypeError(f"counter must be an int, got {type(counter).__name__}")
    if not 0 <= int(counter) < (1 << 64):
        raise ValueError(f"counter must be in [0, 2**64), got {counter}")
    key = derive_key(master_seed, purpose, name)
    ctr = np.array([0, int(counter), 0, 0], dtype=np.uint64)
    return np.random.Generator(np.random.Philox(key=key.u64, counter=ctr))


def _as_key(key: PhiloxKey | tuple[int, int]) -> tuple[int, int]:
    if len(key) != 2:
        raise ValueError("key must be two uint32 words (k0, k1)")
    k0, k1 = key
    for w in (k0, k1):
        if isinstance(w, bool) or not isinstance(w, numbers.Integral) or not 0 <= w <= _MASK32:
            raise ValueError(f"key words must be ints in [0, 2**32), got {key!r}")
    return int(k0), int(k1)


def _check_range(k0: int, n: int, sub: int) -> tuple[int, int, int]:
    for label, v in (("k0", k0), ("n", n), ("sub", sub)):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            raise TypeError(f"{label} must be an int, got {type(v).__name__}")
    k0, n, sub = int(k0), int(n), int(sub)
    if k0 < 0 or n < 0:
        raise ValueError(f"need k0 >= 0 and n >= 0, got k0={k0}, n={n}")
    if k0 + n > MAX_INDEX:
        raise ValueError(f"absolute index range [{k0}, {k0 + n}) exceeds 2**63")
    if not 0 <= sub <= _MASK32:
        raise ValueError(f"sub lane must be in [0, 2**32), got {sub}")
    return k0, n, sub


# --------------------------------------------------------------------------- NumPy twin


def _philox_np(
    c0: NDArray[np.uint32],
    c1: NDArray[np.uint32],
    c2: NDArray[np.uint32],
    c3: NDArray[np.uint32],
    k0: int,
    k1: int,
) -> tuple[NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32], NDArray[np.uint32]]:
    """Ten Philox4x32 rounds on uint32 arrays; ``mulhi``/``mullo`` via exact uint64 products."""
    m0 = np.uint64(_M0)
    m1 = np.uint64(_M1)
    s32 = np.uint64(32)
    for _ in range(_ROUNDS):
        p0 = m0 * c0.astype(np.uint64)
        p1 = m1 * c2.astype(np.uint64)
        hi0 = (p0 >> s32).astype(np.uint32)
        lo0 = p0.astype(np.uint32)  # truncation = low 32 bits
        hi1 = (p1 >> s32).astype(np.uint32)
        lo1 = p1.astype(np.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ np.uint32(k0), lo1, hi0 ^ c3 ^ np.uint32(k1), lo0
        k0 = (k0 + _W0) & _MASK32
        k1 = (k1 + _W1) & _MASK32
    return c0, c1, c2, c3


def philox4x32(counters: NDArray[Any], key: PhiloxKey | tuple[int, int]) -> NDArray[np.uint32]:
    """Philox4x32-10 of explicit counters (NumPy twin).

    ``counters``: (n, 4) or (4,) array of uint32 words ``(c0, c1, c2, c3)``. Returns the
    output words with the same shape, dtype uint32.
    """
    k0, k1 = _as_key(key)
    c = np.asarray(counters)
    if c.shape[-1:] != (4,) or c.ndim > 2:
        raise ValueError(f"counters must have shape (4,) or (n, 4), got {c.shape}")
    if c.dtype.kind not in "ui" or (c.size and (int(c.min()) < 0 or int(c.max()) > _MASK32)):
        raise ValueError("counters must be integers in [0, 2**32)")
    c = c.astype(np.uint32)
    words = _philox_np(c[..., 0], c[..., 1], c[..., 2], c[..., 3], k0, k1)
    return np.stack(words, axis=-1)


def _block_np(k0: int, n: int, sub: int, key: tuple[int, int]) -> NDArray[np.uint32]:
    out = np.empty((n, 4), dtype=np.uint32)
    for a in range(0, n, _CHUNK):
        b = min(n, a + _CHUNK)
        k = np.arange(k0 + a, k0 + b, dtype=np.uint64)
        c0 = (k & np.uint64(_MASK32)).astype(np.uint32)
        c1 = (k >> np.uint64(32)).astype(np.uint32)
        c2 = np.full(b - a, sub, dtype=np.uint32)
        c3 = np.zeros(b - a, dtype=np.uint32)
        words = _philox_np(c0, c1, c2, c3, key[0], key[1])
        for j in range(4):
            out[a:b, j] = words[j]
    return out


def _uniform_np(r: NDArray[np.uint32]) -> NDArray[np.float64]:
    hi = (r[:, 0] >> np.uint32(5)).astype(np.uint64)
    lo = (r[:, 1] >> np.uint32(6)).astype(np.uint64)
    return ((hi << np.uint64(26)) + lo).astype(np.float64) * _TWO_M53


def _bm_parts_np(r: NDArray[np.uint32]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    u1 = (r[:, 0].astype(np.float64) + 1.0) * _TWO_M32
    u2 = r[:, 1].astype(np.float64) * _TWO_M32
    return u1, u2


# --------------------------------------------------------------------------- numba kernels


@functools.cache
def _nb() -> SimpleNamespace:
    """Compile (lazily, cached on disk) the numba kernels. No fastmath (D-4)."""
    import numba as nb

    m0 = np.uint64(_M0)
    m1 = np.uint64(_M1)
    w0 = np.uint32(_W0)
    w1 = np.uint32(_W1)
    mask = np.uint64(_MASK32)
    s32 = np.uint64(32)
    two_m32 = _TWO_M32
    two_m53 = _TWO_M53
    two_pi = _TWO_PI

    @nb.njit(inline="always", cache=True, nogil=True)
    def philox10(c0, c1, c2, c3, k0, k1):
        for _ in range(10):
            p0 = m0 * np.uint64(c0)
            p1 = m1 * np.uint64(c2)
            hi0 = np.uint32(p0 >> s32)
            lo0 = np.uint32(p0 & mask)
            hi1 = np.uint32(p1 >> s32)
            lo1 = np.uint32(p1 & mask)
            n0 = np.uint32(hi1 ^ c1 ^ k0)
            n2 = np.uint32(hi0 ^ c3 ^ k1)
            c0 = n0
            c1 = lo1
            c2 = n2
            c3 = lo0
            k0 = np.uint32(np.uint64(k0) + np.uint64(w0))
            k1 = np.uint32(np.uint64(k1) + np.uint64(w1))
        return c0, c1, c2, c3

    @nb.njit(inline="always", cache=True, nogil=True)
    def counter_words(k_start, i, sub, key0, key1):
        k = np.uint64(k_start + i)
        return philox10(
            np.uint32(k & mask),
            np.uint32(k >> s32),
            np.uint32(sub),
            np.uint32(0),
            np.uint32(key0),
            np.uint32(key1),
        )

    @nb.njit(parallel=True, cache=True, nogil=True)
    def block(out, k_start, sub, key0, key1):
        for i in nb.prange(out.shape[0]):
            r0, r1, r2, r3 = counter_words(k_start, i, sub, key0, key1)
            out[i, 0] = r0
            out[i, 1] = r1
            out[i, 2] = r2
            out[i, 3] = r3

    @nb.njit(parallel=True, cache=True, nogil=True)
    def uniform(out, k_start, sub, key0, key1):
        for i in nb.prange(out.shape[0]):
            r0, r1, _r2, _r3 = counter_words(k_start, i, sub, key0, key1)
            hi = np.uint64(r0 >> np.uint32(5))
            lo = np.uint64(r1 >> np.uint32(6))
            out[i] = np.float64((hi << np.uint64(26)) + lo) * two_m53

    @nb.njit(parallel=True, cache=True, nogil=True)
    def cnormal(out, k_start, sub, key0, key1, sigma):
        for i in nb.prange(out.shape[0]):
            r0, r1, _r2, _r3 = counter_words(k_start, i, sub, key0, key1)
            u1 = (np.float64(r0) + 1.0) * two_m32
            u2 = np.float64(r1) * two_m32
            rad = sigma * np.sqrt(-np.log(u1))
            ang = two_pi * u2
            out[i] = complex(rad * np.cos(ang), rad * np.sin(ang))

    @nb.njit(parallel=True, cache=True, nogil=True)
    def snormal(out, k_start, sub, key0, key1):
        for i in nb.prange(out.shape[0]):
            r0, r1, _r2, _r3 = counter_words(k_start, i, sub, key0, key1)
            u1 = (np.float64(r0) + 1.0) * two_m32
            u2 = np.float64(r1) * two_m32
            out[i] = np.sqrt(-2.0 * np.log(u1)) * np.cos(two_pi * u2)

    return SimpleNamespace(block=block, uniform=uniform, cnormal=cnormal, snormal=snormal)


def _check_backend(backend: str) -> None:
    if backend not in ("numpy_ref", "numba_cpu"):
        raise ConfigurationError(f"unknown backend {backend!r}; use 'numpy_ref' or 'numba_cpu'")


# --------------------------------------------------------------------------- public draws


def philox_block(
    key: PhiloxKey | tuple[int, int],
    k0: int,
    n: int,
    *,
    sub: int = 0,
    backend: Backend = "numpy_ref",
) -> NDArray[np.uint32]:
    """Raw Philox output words for absolute indices [k0, k0 + n), lane ``sub``.

    Returns an (n, 4) uint32 array; row i holds ``philox4x32((k_lo, k_hi, sub, 0), key)``
    with ``k = k0 + i``. Bitwise identical across backends (I5).
    """
    _check_backend(backend)
    kk = _as_key(key)
    k0, n, sub = _check_range(k0, n, sub)
    if backend == "numpy_ref":
        return _block_np(k0, n, sub, kk)
    out = np.empty((n, 4), dtype=np.uint32)
    if n:
        _nb().block(out, np.int64(k0), np.int64(sub), np.int64(kk[0]), np.int64(kk[1]))
    return out


def uniform(
    key: PhiloxKey | tuple[int, int],
    k0: int,
    n: int,
    *,
    sub: int = 0,
    backend: Backend = "numpy_ref",
) -> NDArray[np.float64]:
    """Uniform variates in [0, 1) with 53-bit resolution for indices [k0, k0 + n).

    ``((r0 >> 5)*2**26 + (r1 >> 6))*2**-53``; bitwise identical across backends.
    """
    _check_backend(backend)
    kk = _as_key(key)
    k0, n, sub = _check_range(k0, n, sub)
    if backend == "numpy_ref":
        out = np.empty(n, dtype=np.float64)
        for a in range(0, n, _CHUNK):
            b = min(n, a + _CHUNK)
            out[a:b] = _uniform_np(_block_np(k0 + a, b - a, sub, kk))
        return out
    out = np.empty(n, dtype=np.float64)
    if n:
        _nb().uniform(out, np.int64(k0), np.int64(sub), np.int64(kk[0]), np.int64(kk[1]))
    return out


def standard_normal(
    key: PhiloxKey | tuple[int, int],
    k0: int,
    n: int,
    *,
    sub: int = 0,
    backend: Backend = "numpy_ref",
) -> NDArray[np.float64]:
    """Real N(0, 1) variates ``sqrt(-2 ln u1)*cos(2 pi u2)`` for indices [k0, k0 + n)."""
    _check_backend(backend)
    kk = _as_key(key)
    k0, n, sub = _check_range(k0, n, sub)
    out = np.empty(n, dtype=np.float64)
    if backend == "numpy_ref":
        for a in range(0, n, _CHUNK):
            b = min(n, a + _CHUNK)
            u1, u2 = _bm_parts_np(_block_np(k0 + a, b - a, sub, kk))
            out[a:b] = np.sqrt(-2.0 * np.log(u1)) * np.cos(_TWO_PI * u2)
        return out
    if n:
        _nb().snormal(out, np.int64(k0), np.int64(sub), np.int64(kk[0]), np.int64(kk[1]))
    return out


def complex_normal(
    key: PhiloxKey | tuple[int, int],
    k0: int,
    n: int,
    sigma: float = 1.0,
    *,
    sub: int = 0,
    backend: Backend = "numpy_ref",
    dtype: type[np.complexfloating[Any, Any]] | np.dtype[Any] = np.complex64,
    out: NDArray[np.complexfloating[Any, Any]] | None = None,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Circular complex normal variates CN(0, sigma**2) for absolute indices [k0, k0 + n).

    ``sigma``: total complex standard deviation (``E|z|**2 = sigma**2``), in the caller's
    amplitude unit (sqrt(W) for receiver noise). Computed in float64 and stored as
    ``dtype`` (complex64 by default, the IQ interface type; complex128 allowed). ``out``
    may be given (contiguous, length n, complex64 or complex128). Pure in the absolute index.
    """
    _check_backend(backend)
    kk = _as_key(key)
    k0, n, sub = _check_range(k0, n, sub)
    sig = float(sigma)
    if not (np.isfinite(sig) and sig >= 0.0):
        raise ValueError(f"sigma must be finite and >= 0, got {sigma!r}")
    if out is None:
        dt = np.dtype(dtype)
        if dt not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError(f"dtype must be complex64 or complex128, got {dt}")
        out = np.empty(n, dtype=dt)
    elif (
        out.shape != (n,)
        or out.dtype not in (np.dtype(np.complex64), np.dtype(np.complex128))
        or not out.flags.c_contiguous
    ):
        raise ValueError("out must be a contiguous (n,) complex64 or complex128 array")
    if backend == "numpy_ref":
        for a in range(0, n, _CHUNK):
            b = min(n, a + _CHUNK)
            u1, u2 = _bm_parts_np(_block_np(k0 + a, b - a, sub, kk))
            rad = sig * np.sqrt(-np.log(u1))
            ang = _TWO_PI * u2
            seg = np.empty(b - a, dtype=np.complex128)
            seg.real = rad * np.cos(ang)
            seg.imag = rad * np.sin(ang)
            out[a:b] = seg
        return out
    if n:
        _nb().cnormal(out, np.int64(k0), np.int64(sub), np.int64(kk[0]), np.int64(kk[1]), sig)
    return out
