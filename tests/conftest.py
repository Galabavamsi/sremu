"""Shared pytest configuration (WP0).

* Sets ``NUMBA_DISABLE_INTEL_SVML=1`` before any test imports numba (D-4, deterministic mode).
* Gives every collected test a tier marker: a test without one of the tier markers gets
  the marker of its top-level ``tests/<dir>`` (``unit``, ``physics``, ``runtime``, ``io``,
  ``golden``, ``statistical``, ``integration``, ``experimental``), so the CI selection
  ``-m "unit or physics or runtime or io or golden"`` never silently drops a test.
* Skips ``needs_private_data`` tests when ``$SREMU_PRIVATE_DATA`` is not set.
* Fixtures: ``repo_root``, ``rng`` (keyed Philox generator per test, never the global RNG)
  and ``backend_name`` (parametrized over ``$SREMU_BACKENDS``, default both backends).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")

import numpy as np  # imported after the SVML switch on purpose
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"

TIER_MARKERS = (
    "unit",
    "physics",
    "runtime",
    "io",
    "golden",
    "statistical",
    "integration",
    "validation_smoke",
    "experimental",
)
_DIR_TIER = {
    "unit": "unit",
    "physics": "physics",
    "oracles": "physics",
    "runtime": "runtime",
    "io": "io",
    "golden": "golden",
    "statistical": "statistical",
    "integration": "integration",
    "experimental": "experimental",
}
ALL_BACKENDS = ("numpy_ref", "numba_cpu")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    have_private = bool(os.environ.get("SREMU_PRIVATE_DATA"))
    skip_private = pytest.mark.skip(reason="needs $SREMU_PRIVATE_DATA (private lab data)")
    for item in items:
        if not any(item.get_closest_marker(m) for m in TIER_MARKERS):
            try:
                rel = Path(str(item.path)).resolve().relative_to(TESTS_ROOT)
                tier = _DIR_TIER.get(rel.parts[0], "unit")
            except ValueError:
                tier = "unit"
            item.add_marker(getattr(pytest.mark, tier))
        if item.get_closest_marker("needs_private_data") and not have_private:
            item.add_marker(skip_private)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root directory."""
    return REPO_ROOT


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """A Philox generator keyed by the test's node id (reproducible, independent per test)."""
    digest = hashlib.blake2b(request.node.nodeid.encode("utf-8"), digest_size=8).digest()
    return np.random.Generator(np.random.Philox(key=int.from_bytes(digest, "little")))


def _backends() -> list[str]:
    raw = os.environ.get("SREMU_BACKENDS", ",".join(ALL_BACKENDS))
    chosen = [b.strip() for b in raw.split(",") if b.strip()]
    unknown = set(chosen) - set(ALL_BACKENDS)
    if unknown:
        raise pytest.UsageError(f"unknown backends in SREMU_BACKENDS: {sorted(unknown)}")
    return chosen


@pytest.fixture(params=_backends())
def backend_name(request: pytest.FixtureRequest) -> str:
    """Backend name, parametrized over ``$SREMU_BACKENDS`` (default: numpy_ref, numba_cpu)."""
    return str(request.param)
