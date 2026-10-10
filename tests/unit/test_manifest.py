"""Run directory and manifest (DESIGN.md 3.6).

Git state is tested on temporary repositories created by the tests themselves (never the
sremu working tree). The dirty-patch oracle is reconstruction: cloning the commit into a
second directory and running ``git apply dirty.patch`` must reproduce the dirty tree byte
for byte, including staged, unstaged, deleted, binary and untracked files.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import warnings
from pathlib import Path

import pytest

from sremu.contracts import CONTRACT_VERSION, COUNTER_NAMES
from sremu.errors import ValidityWarning
from sremu.record.manifest import (
    MANIFEST_NAME,
    PATCH_NAME,
    RESOLVED_SCENARIO_NAME,
    HostLoadSampler,
    LinkDecision,
    Manifest,
    ManifestRecorder,
    create_run_dir,
    git_state,
    manifest_json_schema,
    read_manifest,
)
from sremu.scenario.load import load_scenario, parse_scenario, read_document, scenario_sha256
from tests.unit.test_scenario_schema import validate_instance

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")

REQUIRED_FIELDS = {  # section 3.6, verbatim list
    "sremu_version", "git_commit", "git_dirty", "dirty_patch_sha256", "contract_version",
    "platform", "cpu_model", "logical_cpus", "power_plan_or_governor", "gpu", "aws_instance",
    "scenario_sha256", "inputs", "assumptions", "seed", "rng_scheme", "mode", "coupling",
    "backend", "threads", "deterministic", "numba_disable_svml", "page_size",
    "verify_interp_max_rel", "richardson_db", "nondeterministic_inputs", "oracle_csi_used",
    "argv", "t_start_utc", "t_end_utc", "wall_s", "host_load_samples",
}  # fmt: skip

FAKE_ENV = {
    "sremu_version": "0.1.0.dev0", "contract_version": CONTRACT_VERSION,
    "versions": {"python": platform.python_version(), "numpy": "x", "scipy": "x",
                 "numba": "x", "pydantic": "x", "pyzmq": "x", "torch": None},
    "platform": "test", "cpu_model": None, "logical_cpus": 1,
    "power_plan_or_governor": None, "gpu": [], "aws_instance": None,
}  # fmt: skip


def git(repo: Path, *args: str) -> str:
    """Run git in a test-owned temporary repository."""
    assert GIT is not None
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}
    res = subprocess.run(
        [GIT, "-C", str(repo), "-c", "user.name=sremu-test", "-c", "user.email=t@example.invalid",
         "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", *args],
        capture_output=True, check=True, env=env,
    )  # fmt: skip
    return res.stdout.decode()


def make_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "false")  # local config of the temporary repo
    (repo / "a.txt").write_bytes(b"alpha\nbeta\n")
    (repo / "b.txt").write_bytes(b"to be deleted\n")
    (repo / "bin.dat").write_bytes(bytes(range(256)))
    (repo / ".gitignore").write_bytes(b"ignored/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "initial")
    return repo


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and ".git" not in p.relative_to(root).parts
        and "ignored" not in p.relative_to(root).parts
    }


# --------------------------------------------------------------------------- git state


@needs_git
def test_git_state_clean_repo(tmp_path):
    repo = make_repo(tmp_path)
    gs = git_state(repo)
    assert gs.commit == git(repo, "rev-parse", "HEAD").strip()
    assert len(gs.commit) == 40
    assert gs.dirty is False and gs.patch is None and gs.patch_sha256 is None
    (repo / "ignored").mkdir()
    (repo / "ignored" / "x.bin").write_bytes(b"ignored files do not make the tree dirty")
    assert git_state(repo).dirty is False
    assert git_state(repo / "ignored").commit == gs.commit  # any directory inside the tree


@needs_git
def test_git_state_dirty_patch_reconstructs_the_tree(tmp_path):
    repo = make_repo(tmp_path)
    head = git(repo, "rev-parse", "HEAD").strip()
    (repo / "a.txt").write_bytes(b"alpha\nBETA changed\ngamma\n")  # unstaged edit
    (repo / "b.txt").unlink()  # deletion
    (repo / "bin.dat").write_bytes(bytes(range(255, -1, -1)))  # binary edit
    (repo / "staged.txt").write_bytes(b"staged new file\n")
    git(repo, "add", "staged.txt")
    (repo / "sub").mkdir()
    (repo / "sub" / "untracked file.txt").write_bytes(b"untracked\nwith space in name\n")
    gs = git_state(repo)
    assert gs.commit == head and gs.dirty is True and gs.patch
    assert gs.patch_sha256 == hashlib.sha256(gs.patch).hexdigest()
    # reconstruction oracle
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(repo), str(clone))
    git(clone, "config", "core.autocrlf", "false")
    git(clone, "checkout", "-q", head)
    patch = tmp_path / "dirty.patch"
    patch.write_bytes(gs.patch)
    git(clone, "apply", "--binary", str(patch))
    assert tree_bytes(clone) == tree_bytes(repo)


@needs_git
def test_git_state_untracked_only_and_no_commits(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "new.txt").write_bytes(b"new\n")
    gs = git_state(repo)
    assert gs.dirty is True and b"new.txt" in gs.patch
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q")
    assert git_state(empty).commit is None and git_state(empty).dirty is False
    (empty / "f.txt").write_bytes(b"x\n")
    gs2 = git_state(empty)
    assert gs2.commit is None and gs2.dirty is True and b"f.txt" in gs2.patch


def test_git_state_outside_a_work_tree(tmp_path, monkeypatch):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    gs = git_state(plain)
    assert (gs.commit, gs.dirty, gs.patch) == (None, None, None)
    assert gs.reason


# --------------------------------------------------------------------------- run directory


def test_create_run_dir_naming_and_collisions(tmp_path):
    t = dt.datetime(2026, 10, 10, 12, 30, 5, tzinfo=dt.UTC)
    a = create_run_dir(tmp_path / "runs", "lab_like_A_eb", t)
    b = create_run_dir(tmp_path / "runs", "lab_like_A_eb", t)
    assert a.name == "20261010T123005Z_lab_like_A_eb" and a.is_dir()
    assert b.name == "20261010T123005Z_lab_like_A_eb-2"
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    c = create_run_dir(tmp_path / "runs", "x", dt.datetime(2026, 10, 10, 18, 0, 5, tzinfo=ist))
    assert c.name == "20261010T123005Z_x"
    with pytest.raises(ValueError, match="safe directory name"):
        create_run_dir(tmp_path, "../escape", t)


@needs_git
def test_manifest_recorder_full_run(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "a.txt").write_bytes(b"dirty\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        res = load_scenario(REPO / "scenarios" / "lab_like_A.yaml")
    run_dir = create_run_dir(tmp_path / "runs", res.scenario.name)
    rec = ManifestRecorder(
        res, run_dir, argv=["sremu", "run", "lab_like_A.yaml"], repo_dir=repo, threads=4
    )
    rec.set_link(
        "ap.tx0->rx1.rx0",
        LinkDecision(
            narrowband=True,
            fd_filter="kaiser32",
            t_u_s="0.01",
            tiles={"ris0": 1},
            interp_error_bound=1e-3,
        ),
    )
    rec.update_verify_interp(2e-4)
    rec.update_verify_interp(1e-4)  # the maximum is kept
    rec.add_counters({"late_commands": 2})
    rec.add_counters({"late_commands": 1, "custom_counter": 5})
    rec.mark_nondeterministic()
    rec.add_assumption("extra assumption")
    rec.sample_host_load()
    m = rec.finish()

    data = json.loads((run_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert REQUIRED_FIELDS <= set(data)
    assert set(data["versions"]) == {
        "python",
        "numpy",
        "scipy",
        "numba",
        "pydantic",
        "pyzmq",
        "torch",
    }
    assert data["versions"]["python"] == platform.python_version()
    assert validate_instance(data, manifest_json_schema()) == []
    assert read_manifest(run_dir) == m == Manifest.model_validate(data)
    # git
    assert data["git_commit"] == git(repo, "rev-parse", "HEAD").strip()
    assert data["git_dirty"] is True
    patch = (run_dir / PATCH_NAME).read_bytes()
    assert data["dirty_patch_sha256"] == hashlib.sha256(patch).hexdigest()
    # scenario hash = canonical JSON of the resolved scenario, recomputed from the file
    resolved = parse_scenario(read_document(run_dir / RESOLVED_SCENARIO_NAME))
    assert data["scenario_sha256"] == scenario_sha256(resolved) == res.sha256()
    assert data["scenario_name"] == "lab_like_A_eb"
    lib = REPO / "data" / "states" / "lab_binary_3p5ghz.yaml"
    assert data["inputs"] == [
        {
            "path": "data/states/lab_binary_3p5ghz.yaml",
            "sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
            "data_class": "ASSUMED",
        }
    ]
    assert sum("ASSUMED" in a for a in data["assumptions"]) == 2
    assert "extra assumption" in data["assumptions"]
    # run settings
    assert (data["seed"], data["rng_scheme"]) == (20261010, "philox4x32-10/blake2b-v1")
    assert (
        data["mode"],
        data["coupling"],
        data["backend"],
        data["threads"],
        data["deterministic"],
    ) == ("lockstep", "barrier", "numba_cpu", 4, True)
    assert data["page_size"] == {"rx1.rx0": 512, "rx2.rx0": 512}  # 1 MS/s (4.3)
    assert data["links"]["ap.tx0->rx1.rx0"]["tiles"] == {"ris0": 1}
    assert data["verify_interp_max_rel"] == 2e-4 and data["richardson_db"] is None
    assert set(COUNTER_NAMES) <= set(data["counters"])
    assert data["counters"]["late_commands"] == 3 and data["counters"]["custom_counter"] == 5
    assert all(data["counters"][k] == 0 for k in COUNTER_NAMES if k != "late_commands")
    assert data["nondeterministic_inputs"] is True and data["oracle_csi_used"] is False
    assert data["argv"] == ["sremu", "run", "lab_like_A.yaml"]
    t0 = dt.datetime.fromisoformat(data["t_start_utc"].replace("Z", "+00:00"))
    t1 = dt.datetime.fromisoformat(data["t_end_utc"].replace("Z", "+00:00"))
    assert t1 >= t0 and data["wall_s"] >= 0.0
    assert len(data["host_load_samples"]) == 2
    for s in data["host_load_samples"]:
        assert s["cpu_busy_fraction"] is None or 0.0 <= s["cpu_busy_fraction"] <= 1.0


@needs_git
def test_clean_tree_writes_no_patch(tmp_path):
    repo = make_repo(tmp_path)
    res = load_scenario(REPO / "scenarios" / "quickstart.yaml")
    run_dir = tmp_path / "run"
    m = ManifestRecorder(res, run_dir, repo_dir=repo, environment=FAKE_ENV).finish()
    assert m.git_dirty is False and m.dirty_patch_sha256 is None
    assert not (run_dir / PATCH_NAME).exists()
    assert m.page_size == {"mu.rx0": 512}
    assert m.inputs[0].data_class == "ANALYTIC"


def test_manifest_without_git(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain = tmp_path / "plain"
    plain.mkdir()
    res = load_scenario(REPO / "scenarios" / "quickstart.yaml")
    m = ManifestRecorder(res, tmp_path / "run", repo_dir=plain, environment=FAKE_ENV).finish()
    assert (m.git_commit, m.git_dirty, m.dirty_patch_sha256) == (None, None, None)


def test_host_load_sampler():
    s = HostLoadSampler()
    a = s.sample()
    sum(i * i for i in range(200_000))  # some CPU work between samples
    b = s.sample()
    for x in (a, b):
        assert x.cpu_busy_fraction is None or 0.0 <= x.cpu_busy_fraction <= 1.0
        assert x.t_utc.endswith("Z")
