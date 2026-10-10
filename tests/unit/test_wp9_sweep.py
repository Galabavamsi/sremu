"""Sweep runner, runner registry and ``sremu sweep`` (DESIGN.md sections 2.5, 3.6; WP9).

Oracles: the point order and ids follow the documented (zip, grid, seed) product; the echo
runner's keyed draw equals ``sremu.rng.generator(seed, "trial", "<sweep>/<point_id>")``;
the index does not depend on the number of worker processes.
"""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sremu import cli
from sremu import rng as sremu_rng
from sremu.errors import ConfigurationError
from sremu.experiments.registry import RUNNERS, PointContext, register, resolve
from sremu.experiments.sweep import (
    INDEX_CSV,
    INDEX_JSON,
    POINT_JSON,
    SUMMARY_JSON,
    SWEEP_JSON,
    SweepSpec,
    jsonable,
    load_spec,
    run_sweep,
)

pytestmark = pytest.mark.unit

SPEC = {
    "name": "demo",
    "runner": "echo",
    "base": {"label": "x"},
    "grid": {"a": [1, 2], "b": ["u", "v", "w"]},
    "zip": {"h": [2.0, 4.0], "tag": ["lo", "hi"]},
    "seeds": [5, 6],
}


def test_spec_points_order_ids_and_params():
    sp = SweepSpec.from_mapping(SPEC)
    pts = sp.points()
    assert len(pts) == 2 * (2 * 3) * 2 == 24
    assert [p.point_id for p in pts[:3]] == ["p00000", "p00001", "p00002"]
    assert pts[0].params == {"label": "x", "h": 2.0, "tag": "lo", "a": 1, "b": "u"}
    assert (pts[0].seed, pts[1].seed) == (5, 6)
    assert pts[2].params["b"] == "v" and pts[2].params["a"] == 1  # b varies faster than a
    assert pts[12].params["h"] == 4.0 and pts[12].params["tag"] == "hi"  # zip outermost
    assert len({p.params_sha256 for p in pts}) == 24
    sp2 = SweepSpec.from_mapping({"name": "r", "runner": "echo", "repeats": 3})
    assert [p.seed for p in sp2.points()] == [0, 1, 2]
    assert SweepSpec.from_mapping({"name": "d", "runner": "echo"}).seeds == (0,)


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "../x", "runner": "echo"},
        {"name": "x", "runner": "echo", "oops": 1},
        {"name": "x"},
        {"name": "x", "runner": "echo", "grid": {"a": []}},
        {"name": "x", "runner": "echo", "grid": {"a": 1}},
        {"name": "x", "runner": "echo", "zip": {"a": [1, 2], "b": [1]}},
        {"name": "x", "runner": "echo", "base": {"a": 1}, "grid": {"a": [1]}},
        {"name": "x", "runner": "echo", "seeds": [1, 1]},
        {"name": "x", "runner": "echo", "seeds": [1], "repeats": 2},
        {"name": "x", "runner": "echo", "seeds": [-1]},
    ],
)
def test_spec_validation_rejects(bad):
    with pytest.raises(ConfigurationError):
        SweepSpec.from_mapping(bad)


def _strip(entries: list[dict]) -> list[dict]:
    return [{k: v for k, v in e.items() if k != "wall_s"} for e in entries]


def test_run_sweep_tree_index_and_keyed_draws(tmp_path):
    sp = SweepSpec.from_mapping(SPEC)
    res = run_sweep(sp, tmp_path, environment={"host": "test"}, argv=["x"])
    assert res.ok and res.n_ok == 24 and res.n_failed == 0
    root = tmp_path / "demo"
    assert res.root == root
    sweep = json.loads((root / SWEEP_JSON).read_text(encoding="utf-8"))
    assert sweep["n_points"] == 24 and sweep["spec_sha256"] == sp.sha256
    assert sweep["environment"] == {"host": "test"} and sweep["t_end_utc"]
    index = json.loads((root / INDEX_JSON).read_text(encoding="utf-8"))
    assert [e["point_id"] for e in index] == [f"p{i:05d}" for i in range(24)]
    for e, pt in zip(index, sp.points(), strict=True):
        assert e["status"] == "ok" and e["params"] == jsonable(pt.params) and e["seed"] == pt.seed
        u = sremu_rng.generator(pt.seed, "trial", f"demo/{pt.point_id}").random()
        assert e["summary"]["u"] == u
        assert e["summary"]["param.a"] == pt.params["a"]
        rec = json.loads((root / pt.point_id / POINT_JSON).read_text(encoding="utf-8"))
        assert rec["params_sha256"] == pt.params_sha256 and rec["status"] == "ok"
        summ = json.loads((root / pt.point_id / SUMMARY_JSON).read_text(encoding="utf-8"))
        assert summ == e["summary"]
    with (root / INDEX_CSV).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 24 and rows[1]["param.b"] == "u" and rows[1]["result.seed"] == "6"
    assert rows[3]["param.b"] == "v" and rows[3]["seed"] == "6"


def test_parallel_jobs_give_the_same_index(tmp_path):
    sp = SweepSpec.from_mapping({**SPEC, "name": "par"})
    run_sweep(sp, tmp_path / "one", jobs=1, environment={})
    run_sweep(sp, tmp_path / "two", jobs=2, environment={})
    a = json.loads((tmp_path / "one" / "par" / INDEX_JSON).read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "two" / "par" / INDEX_JSON).read_text(encoding="utf-8"))
    assert _strip(a) == _strip(b)


def test_failures_are_recorded_and_resume_reruns_only_them(tmp_path):
    spec = {"name": "f", "runner": "echo", "grid": {"fail": [False, True, False]}}
    sp = SweepSpec.from_mapping(spec)
    res = run_sweep(sp, tmp_path, environment={})
    assert not res.ok and res.n_failed == 1 and res.n_ok == 2
    bad = res.records[1]
    assert bad["status"] == "failed" and "asked to fail" in bad["error"]
    assert not (tmp_path / "f" / "p00001" / SUMMARY_JSON).exists()
    res2 = run_sweep(sp, tmp_path, resume=True, environment={})
    assert res2.n_resumed == 2 and res2.n_failed == 1
    assert [r.get("resumed", False) for r in res2.records] == [True, False, True]
    # same params and seeds but another runner: nothing is reused
    sp4 = SweepSpec.from_mapping({**spec, "runner": "sremu.experiments.registry:echo"})
    assert sp4.points()[0].params_sha256 == sp.points()[0].params_sha256
    assert run_sweep(sp4, tmp_path, resume=True, environment={}).n_resumed == 0
    sp3 = SweepSpec.from_mapping({**spec, "base": {"extra": 1}})  # new params: nothing reused
    assert run_sweep(sp3, tmp_path, resume=True, environment={}).n_resumed == 0


def test_runner_registry_and_import_policy():
    assert resolve("echo") is RUNNERS["echo"].fn
    assert resolve("sremu.experiments.registry:echo") is RUNNERS["echo"].fn
    with pytest.raises(ConfigurationError):
        resolve("nope")
    with pytest.raises(ConfigurationError):
        resolve("json:dumps")  # outside sremu without allow_import
    assert resolve("json:dumps", allow_import=True) is json.dumps
    with pytest.raises(ConfigurationError):
        resolve("sremu.experiments.registry:nothing_here")

    @register("wp9_test_runner")
    def _r(ctx: PointContext):
        return {"x": ctx.params.get("x", 0) * 2}

    assert resolve("wp9_test_runner") is _r
    with pytest.raises(ConfigurationError):
        register("wp9_test_runner", lambda ctx: {})
    del RUNNERS["wp9_test_runner"]


def test_controller_rollout_runner(tmp_path, repo_root):
    spec = {
        "name": "roll",
        "runner": "controller_rollout",
        "base": {
            "scenario": str(repo_root / "scenarios" / "quickstart.yaml"),
            "decision_period_s": "1e-3",
            "n_steps": 6,
            "n_batches": 3,
        },
        "grid": {
            "controller": [
                {"impl": "Urap", "period_s": "1e-3", "params": {"p": 0.5}},
                {"impl": "ExactOptimum", "period_s": "1e-3"},
            ]
        },
    }
    res = run_sweep(SweepSpec.from_mapping(spec), tmp_path, environment={})
    assert res.ok, [r.get("error") for r in res.records]
    urap, opt = (r["summary"] for r in res.records)
    assert opt["mean_regret_db"] == pytest.approx(0.0, abs=1e-9)  # static scene, oracle CSI
    assert urap["mean_regret_db"] > 0.0 and opt["mean_power_dbm"] > urap["mean_power_dbm"]
    assert urap["n_steps"] == 6 and urap["n_batches"] == 3


def test_controller_rollout_uses_the_scene_of_the_point_seed(tmp_path, repo_root):
    """A point seed other than the scenario's rebuilds the random-waypoint scene; the
    controller must optimize that scene, not the scenario-seed one. Oracle: ExactOptimum
    decides at t_b on the true scene, and the reward is evaluated at t_b + T/2, so its regret
    is bounded by the motion within half a period (0.5 mm at 1 m/s, lambda = 8.6 cm):
    <= 0.01 dB. A controller left on the stale trajectory loses dBs."""
    import copy

    from sremu.scenario.load import read_document

    doc = copy.deepcopy(read_document(repo_root / "scenarios" / "quickstart.yaml"))
    mu = doc["nodes"][1]
    del mu["position_m"]
    mu["trajectory"] = {
        "model": "rwp_1d",
        "p0_m": [4.7, 0.5, 0.0],
        "p1_m": [4.7, 2.5, 0.0],
        "speed_mps": 1.0,
    }
    scen = tmp_path / "rwp.json"
    scen.write_text(json.dumps(doc), encoding="utf-8")
    other = int(doc.get("seed", 0)) + 76
    spec = {
        "name": "rwp",
        "runner": "controller_rollout",
        "base": {
            "scenario": str(scen),
            "decision_period_s": "1e-3",
            "n_steps": 40,
            "n_batches": 4,
            "controller": {"impl": "ExactOptimum", "period_s": "1e-3"},
        },
        "seeds": [int(doc.get("seed", 0)), other],
    }
    res = run_sweep(SweepSpec.from_mapping(spec), tmp_path / "out", environment={})
    assert res.ok, [r.get("error") for r in res.records]
    same, rebuilt = (r["summary"] for r in res.records)
    assert 0.0 <= same["mean_regret_db"] <= 0.01
    assert 0.0 <= rebuilt["mean_regret_db"] <= 0.01
    assert same["mean_power_dbm"] != rebuilt["mean_power_dbm"]  # the seeds give other walks


def test_jsonable_conversions():
    import math
    from fractions import Fraction

    import numpy as np

    out = jsonable(
        {
            "a": np.float64(1.5),
            "b": np.int16(3),
            "c": Fraction(1, 3),
            "d": 1 + 2j,
            "e": math.inf,
            "f": np.array([1, 2]),
            "g": Path("x"),
        }
    )
    assert out == {"a": 1.5, "b": 3, "c": "1/3", "d": [1.0, 2.0], "e": None, "f": [1, 2], "g": "x"}


def _write_spec(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_cli_sweep_exit_codes(tmp_path):
    runner = CliRunner()
    good = _write_spec(
        tmp_path,
        """
        name: cli_demo
        runner: echo
        grid: {a: [1, 2]}
        seeds: [0]
        """,
    )
    out = tmp_path / "runs"
    res = runner.invoke(cli.app, ["sweep", str(good), "--out", str(out), "-j", "1"])
    assert res.exit_code == 0, res.output
    assert "2 ok, 0 failed" in res.output
    assert (out / "cli_demo" / INDEX_JSON).is_file()
    assert load_spec(good).name == "cli_demo"
    failing = _write_spec(
        tmp_path,
        """
        name: cli_fail
        runner: echo
        grid: {fail: [true]}
        """,
    )
    res = runner.invoke(cli.app, ["sweep", str(failing), "--out", str(out), "-q"])
    assert res.exit_code == 1, res.output
    bad = _write_spec(tmp_path, "name: '../bad'\nrunner: echo\n")
    res = runner.invoke(cli.app, ["sweep", str(bad), "--out", str(out)])
    assert res.exit_code == 2
    res = runner.invoke(cli.app, ["sweep", str(tmp_path / "missing.yaml"), "--out", str(out)])
    assert res.exit_code == 2
