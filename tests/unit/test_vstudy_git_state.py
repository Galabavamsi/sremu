"""Git state of the WP10 validation-study runs (``validation/vstudy.py``): the rules of the
WP11 experiments (``$SREMU_EXP_GIT_JSON`` first, as ``experiments/common.py:git_info``),
then the work tree, then ``$SREMU_GIT_COMMIT`` / ``$SREMU_GIT_DIRTY`` for a job bundle
without ``.git`` (as ``bench/_common.py:bundle_commit``), and ``ManifestRecorder(git=...)``.

No test starts a git process: the work-tree reader is replaced by a stub.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from validation import vstudy

from sremu.record import manifest as mf
from sremu.record.manifest import GitState, ManifestRecorder, read_manifest
from sremu.scenario.load import read_document

REPO = Path(__file__).resolve().parents[2]
COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def _scenario() -> dict:
    return dict(read_document(REPO / "scenarios" / "quickstart.yaml"))


@pytest.fixture
def no_git(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The work-tree reader of vstudy and of the manifest module, replaced: records calls and
    reports 'not a git work tree' (an AWS job bundle)."""
    calls: list[str] = []

    def fake(repo_dir=None) -> GitState:
        calls.append(str(repo_dir))
        return GitState(None, None, None, "not inside a git work tree (stub)")

    monkeypatch.setattr(vstudy, "git_state", fake)
    monkeypatch.setattr(mf, "git_state", fake)
    for k in ("SREMU_EXP_GIT_JSON", "SREMU_GIT_COMMIT", "SREMU_GIT_DIRTY"):
        monkeypatch.delenv(k, raising=False)
    return calls


def _write_state(tmp_path: Path, **kw) -> Path:
    p = tmp_path / "git_state.json"
    p.write_text(json.dumps(kw), encoding="utf-8")
    return p


def test_exp_git_json_is_used_without_reading_git(tmp_path, monkeypatch, no_git) -> None:
    p = _write_state(tmp_path, commit=COMMIT, dirty=False, dirty_patch_sha256=None)
    monkeypatch.setenv("SREMU_EXP_GIT_JSON", str(p))
    monkeypatch.setenv("SREMU_GIT_COMMIT", OTHER)  # the driver's record wins
    monkeypatch.setenv("SREMU_GIT_DIRTY", "1")
    gs, src = vstudy.study_git_state()
    assert (gs.commit, gs.dirty, gs.patch_sha256) == (COMMIT, False, None)
    assert "SREMU_EXP_GIT_JSON" in src and no_git == []

    run = vstudy.StudyRun(tmp_path, "t", _scenario(), root=tmp_path / "runs")
    meta_path = run.finish({"study": "T"})
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    man = read_manifest(run.run_dir)
    assert (man.git_commit, man.git_dirty) == (COMMIT, False)
    assert (meta["sremu_commit"], meta["sremu_dirty"]) == (COMMIT, False)
    assert "SREMU_EXP_GIT_JSON" in meta["git_state_source"]
    assert not (run.run_dir / "dirty.patch").exists()
    assert no_git == []


def test_exp_git_json_dirty_patch_copied_or_hash_kept(tmp_path, monkeypatch, no_git) -> None:
    patch = b"diff --git a/x b/x\n+dirty\n"
    sha = hashlib.sha256(patch).hexdigest()
    pp = tmp_path / f"dirty_{sha[:16]}.patch"
    pp.write_bytes(patch)
    p = _write_state(
        tmp_path, commit=COMMIT, dirty=True, dirty_patch_sha256=sha, dirty_patch_path=str(pp)
    )
    monkeypatch.setenv("SREMU_EXP_GIT_JSON", str(p))
    run = vstudy.StudyRun(tmp_path, "t", _scenario(), root=tmp_path / "runs")
    run.finish({"study": "T"})
    man = read_manifest(run.run_dir)
    assert (man.git_commit, man.git_dirty, man.dirty_patch_sha256) == (COMMIT, True, sha)
    assert (run.run_dir / "dirty.patch").read_bytes() == patch

    # patch file gone (or not the recorded bytes): the recorded hash is kept, no patch written
    pp.write_bytes(b"something else\n")
    run2 = vstudy.StudyRun(tmp_path, "t2", _scenario(), root=tmp_path / "runs")
    run2.finish({"study": "T"})
    man2 = read_manifest(run2.run_dir)
    assert (man2.git_dirty, man2.dirty_patch_sha256) == (True, sha)
    assert not (run2.run_dir / "dirty.patch").exists()
    assert no_git == []


def test_missing_json_file_falls_back_like_experiments(tmp_path, monkeypatch, no_git) -> None:
    monkeypatch.setenv("SREMU_EXP_GIT_JSON", str(tmp_path / "absent.json"))
    gs, src = vstudy.study_git_state()
    assert gs.commit is None and src == "work tree (read-only git)"
    assert len(no_git) == 1


@pytest.mark.parametrize(("flag", "dirty"), [("0", False), ("1", True), ("", None)])
def test_bundle_without_git_uses_sremu_git_commit(tmp_path, monkeypatch, no_git, flag, dirty):
    monkeypatch.setenv("SREMU_GIT_COMMIT", COMMIT)
    if flag:
        monkeypatch.setenv("SREMU_GIT_DIRTY", flag)
    gs, src = vstudy.study_git_state()
    assert (gs.commit, gs.dirty, gs.patch) == (COMMIT, dirty, None)
    assert src == "$SREMU_GIT_COMMIT/$SREMU_GIT_DIRTY"
    run = vstudy.StudyRun(tmp_path, "t", _scenario(), root=tmp_path / "runs")
    meta = json.loads(run.finish({"study": "T"}).read_text(encoding="utf-8"))
    assert (meta["sremu_commit"], meta["sremu_dirty"]) == (COMMIT, dirty)
    assert meta["git_state_source"] == "$SREMU_GIT_COMMIT/$SREMU_GIT_DIRTY"


def test_work_tree_wins_over_bundle_variables(monkeypatch, no_git) -> None:
    monkeypatch.setattr(vstudy, "git_state", lambda repo_dir=None: GitState(OTHER, False, None))
    monkeypatch.setenv("SREMU_GIT_COMMIT", COMMIT)
    monkeypatch.setenv("SREMU_GIT_DIRTY", "0")
    gs, src = vstudy.study_git_state()
    assert gs.commit == OTHER and src == "work tree (read-only git)"


def test_manifest_recorder_records_a_given_git_state(tmp_path, no_git) -> None:
    from sremu.scenario.load import parse_scenario, resolve_scenario

    res = resolve_scenario(parse_scenario(_scenario()), None, warn=False)
    rec = ManifestRecorder(res, tmp_path / "r", git=GitState(COMMIT, True, b"p\n"))
    man = rec.finish()
    assert (man.git_commit, man.git_dirty) == (COMMIT, True)
    assert man.dirty_patch_sha256 == hashlib.sha256(b"p\n").hexdigest()
    assert (tmp_path / "r" / "dirty.patch").read_bytes() == b"p\n"
    assert no_git == []  # the given state is recorded; the reader is never called
