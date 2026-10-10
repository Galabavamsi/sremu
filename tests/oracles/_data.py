"""Loaders for the oracle input data in ``tests/oracles/data`` (with sha256 provenance)."""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parent / "data"

__all__ = ["DATA", "load_json", "pm_checks", "provenance", "sha256_lf", "tang_fig10"]


def sha256_lf(path: Path) -> str:
    """sha256 of a text file with CRLF normalized to LF (robust to git autocrlf checkouts)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@cache
def load_json(name: str) -> Any:
    """Parse ``tests/oracles/data/<name>``."""
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def provenance() -> dict[str, Any]:
    """The provenance sidecar ``PROVENANCE.json``."""
    return load_json("PROVENANCE.json")


def pm_checks() -> dict[str, Any]:
    """``physics_model_checks.json`` written by ``research/scratch/pm_checks.py``."""
    return load_json("physics_model_checks.json")


def tang_fig10() -> dict[str, Any]:
    """Vector-extracted Tang et al. 2022 Fig. 10 curves (DIG)."""
    return load_json("tang2022_fig10_vector.json")
