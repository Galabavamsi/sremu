"""Smoke test of ``examples/gym_quickstart`` (WP9): it runs, the exact-optimum controller
has zero regret on the static quickstart scene and beats the random policy."""

from __future__ import annotations

import runpy

import pytest

pytestmark = pytest.mark.unit


def test_gym_quickstart_example_runs(repo_root, capsys):
    mod = runpy.run_path(str(repo_root / "examples" / "gym_quickstart" / "gym_quickstart.py"))
    out = mod["main"](steps=5, seed=1)
    assert abs(out["optimum_mean_regret_db"]) <= 1e-9
    assert out["random_mean_regret_db"] > 0.0
    assert out["optimum_mean_dbm"] > out["random_mean_dbm"]
    assert "optimum_mean_dbm" in capsys.readouterr().out
