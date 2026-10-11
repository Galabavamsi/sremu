"""`sremu serve` sets OMP_WAIT_POLICY=PASSIVE for paced scenarios before numba starts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sremu.commands import serve


def _write(tmp_path: Path, mode: str) -> Path:
    p = tmp_path / "s.yaml"
    p.write_text(f"run: {{mode: {mode}, duration_s: '1'}}\n", encoding="utf-8")
    return p


def test_scenario_mode_is_read_without_validation(tmp_path: Path) -> None:
    assert serve._scenario_mode(_write(tmp_path, "paced")) == "paced"
    assert serve._scenario_mode(_write(tmp_path, "lockstep")) == "lockstep"
    assert serve._scenario_mode(tmp_path / "missing.yaml") is None


def test_passive_policy_only_before_numba_and_never_over_an_explicit_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    monkeypatch.delitem(sys.modules, "numba", raising=False)
    serve._passive_omp_for_paced()
    import os

    assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"
    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    serve._passive_omp_for_paced()
    assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE"
