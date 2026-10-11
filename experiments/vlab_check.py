"""Integrity check of the locked V-LAB comparison (DESIGN.md section 10.9) at a later commit.

V-LAB is the pre-registered, locked lab comparison: it ran once, at sremu 7a63287 (run
``validation/ants_fig4b/results/20261011T003044Z_vlab_unblinded``), and is never re-run with
changed code. The paper cites that run at that commit. This check states whether the tree it
runs in still carries the V-LAB protocol and analysis code unchanged, so that the paper can
say so for the pinned commit:

* the protocol digest of ``PROTOCOL.md`` + ``protocol.json``
  (:func:`validation.ants_fig4b.protocol_lock.protocol_digest`) equals
  ``PREREGISTERED_SHA256`` and the digest the run recorded (``meta.json``);
* ``PROTOCOL.md`` and ``protocol.json`` have the SHA-256 the run's manifest recorded as inputs;
* every file of ``validation/ants_fig4b/`` (run directories and caches excluded) is
  byte-identical to the file at 7a63287 (``FILES_7A63287``), and there is no other file.

``FILES_7A63287`` was read from the files of a checkout of 7a63287 (the worktree
``research/scratch/vlab_unblind_7a63287`` the run was made from, detached HEAD 7a63287) and
re-derived, read-only and without a git process, from the blobs of commit 7a63287 in
``.git/objects`` (identical, 13 of 13). The authoritative check is
``git diff --exit-code 7a63287 <pinned commit> -- validation/ants_fig4b`` (the lead's step);
this module lets a job bundle without ``.git`` (AWS) record the same facts.

Not covered: the sremu modules the V-LAB code imports (``sremu.analysis.stats``,
``sremu.rng``, ``sremu.physics.*``, ``sremu.record.manifest``, ...). Some of them differ
between 7a63287 and later commits, so this check does not show that a re-run at the pinned
commit would reproduce the run; the paper cites the run at 7a63287 (``not_covered`` in the
record).

    python -m experiments.vlab_check [--out _regen/vlab_check.json]

Reads files only (no git process, the measured data is never opened). Exit 0 when every
check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VLAB_DIR = "validation/ants_fig4b"
VLAB_RUN = f"{VLAB_DIR}/results/20261011T003044Z_vlab_unblinded"
VLAB_COMMIT = "7a632876ae731f708c081fabc09b255bb3b29339"
SHA256SUMS_7A63287 = """
8792ffb49431a23515680b075cd3b645c263db45320f2d77dc60a29739d4627a  PROTOCOL.md
62a4552d072774e668634fa35f8221f71859c653e01d6066b1e13fc435cbed9c  __init__.py
94508052be3c40046be69725b9277d331428751dfe92b72f084a0593fbf88c99  analysis.py
fa40e1970af22ebe59b4f349df5508de43a099f74eab28b1e31421942564140e  antennas/short_dipole_z_null40.csv
9cd3ffa1b3a59a79ce8c55a8bc7e1500de963c889a59b435315b16b2d9a6ed1f  decoy.py
1b5b41d83c724dc1808f533d8c1674f023d83e268cbd8204fcf6bfe34f45117d  diagnostic.py
7fd3d3960274965bd0517767371597aa87d6621fc5d20c7257fa8aecc7ee6338  predict.py
272e2bc37d382046b401ffd08838d90c1155f6887e1072b7f035b6ae43a3b9b4  prior.py
fe8a90188464f9ccf972c8918e4f7fc06b12ea7e9616f731b9114b6fca4ca9c8  protocol.json
191cec6acb53574b9151cb1abed11ea3e704b5e04102ee23c3e5978bdf053354  protocol.py
b0d6ab0d9563fec028d3c6265b823fd5d8b0dd2ed1b40abd0e39abc0f542b5a7  protocol_lock.py
39a97b59a66078281352bad8d9a345a6f624099212f21e0f7bb97534790d6617  run.py
23f6a64d648f571048d7447c905f8deb4159e90ece2f5b29a07463aa05579da7  runlog.py
"""
"""``sha256sum`` of every file of ``validation/ants_fig4b/`` at 7a63287 (results excluded)."""
FILES_7A63287 = {
    f"{VLAB_DIR}/{name}": h
    for h, name in (ln.split() for ln in SHA256SUMS_7A63287.splitlines() if ln.strip())
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def vlab_files(repo: Path = REPO) -> dict[str, str]:
    """SHA-256 of the V-LAB code and protocol files of ``repo`` (run directories, caches
    excluded)."""
    d = repo / VLAB_DIR
    out = {}
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(repo).as_posix()
        if p.is_file() and "/results/" not in rel and "__pycache__" not in p.parts:
            out[rel] = _sha(p)
    return out


def check(repo: Path = REPO) -> dict[str, Any]:
    """The checks of the module docstring, as a JSON-ready record with ``ok``."""
    from validation.ants_fig4b import protocol_lock

    digest = protocol_lock.protocol_digest(repo / VLAB_DIR)
    m = re.search(
        r'^PREREGISTERED_SHA256: Final\[str\] = "([0-9a-f]*)"',
        (repo / VLAB_DIR / "protocol_lock.py").read_text(encoding="utf-8"),
        re.M,
    )
    locked = m.group(1) if m else ""
    run = repo / VLAB_RUN
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    recorded = {Path(i["path"].replace("\\", "/")).name: i["sha256"] for i in man["inputs"]}
    proto_files = {
        name: {
            "sha256": _sha(repo / VLAB_DIR / name),
            "run_input_sha256": recorded.get(name),
        }
        for name in ("PROTOCOL.md", "protocol.json")
    }
    files = vlab_files(repo)
    differing = sorted(k for k in FILES_7A63287 if files.get(k) != FILES_7A63287[k])
    extra = sorted(set(files) - set(FILES_7A63287))
    res: dict[str, Any] = {
        "vlab_run": VLAB_RUN,
        "run_git_commit": meta.get("git_commit"),
        "run_git_dirty": meta.get("git_dirty"),
        "run_decisions": meta.get("decisions"),
        "protocol_digest": digest,
        "preregistered_sha256": locked,
        "run_protocol_digest": meta.get("protocol_digest"),
        "digest_equals_lock": digest == locked,
        "lock_equals_run": locked
        == meta.get("protocol_digest")
        == meta.get("preregistered_sha256"),
        "protocol_files": proto_files,
        "protocol_files_equal_run_inputs": all(
            v["sha256"] == v["run_input_sha256"] for v in proto_files.values()
        ),
        "files_compared": len(FILES_7A63287),
        "files_differing_from_7a63287": differing,
        "files_not_at_7a63287": extra,
        "reference": f"FILES_7A63287: the files of a checkout of {VLAB_COMMIT} (experiments/"
        "vlab_check.py); authoritative: git diff --exit-code 7a63287 <commit> -- "
        "validation/ants_fig4b",
        "not_covered": "the sremu modules the V-LAB code imports (sremu.analysis.stats, "
        "sremu.rng, sremu.physics.*, sremu.record.manifest, ...): this check does not show "
        "that a re-run at this commit would reproduce the 7a63287 run",
    }
    res["ok"] = bool(
        res["digest_equals_lock"]
        and res["lock_equals_run"]
        and res["protocol_files_equal_run_inputs"]
        and meta.get("git_commit") == VLAB_COMMIT
        and meta.get("git_dirty") is False
        and not differing
        and not extra
    )
    res["summary"] = (
        f"protocol digest {'=' if res['digest_equals_lock'] else '!='} lock "
        f"{'=' if res['lock_equals_run'] else '!='} run record; "
        f"{len(FILES_7A63287) - len(differing)}/{len(FILES_7A63287)} files byte-identical to "
        f"7a63287, {len(extra)} other; run at {str(meta.get('git_commit'))[:7]} "
        f"dirty {meta.get('git_dirty')}"
    )
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="write the record (JSON) here")
    ap.add_argument("--repo", type=Path, default=REPO, help="sremu tree to check")
    a = ap.parse_args(argv)
    res = check(a.repo.resolve())
    text = json.dumps(res, indent=1) + "\n"
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print(text, end="")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
