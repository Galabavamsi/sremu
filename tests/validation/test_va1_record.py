"""VA-1 record check (DESIGN.md section 11, E-A; paper Section VII-B and Fig. 10(c)).

VA-1 is the IQ consistency subset of E-A: served IQ runs (external controller process in
barrier mode, external RX application) against the batch evaluators, one row per
(controller, tau_SUI, noise). Its acceptance is pre-specified: with noise off,
|IQ - vectorized midpoint evaluator| <= 0.02 dB; with noise on, the vectorized value lies in
the 95 % batch-means interval of the IQ window powers. The paper quotes the largest
|IQ - batch| over all rows against the 0.02 dB limit.

The record is the E-A-IQ aggregate that ``experiments/ea_iq.py`` writes (``aggregate``, run
by ``experiments/finalize.py``) and that ``experiments/install_regen.py`` installs into the
paper repository: ``va1.csv``, ``meta.json`` and ``manifest.json`` in
``sweeps/results/E-A-IQ``. This test reads the record, recomputes every row's IQ-minus-batch
difference from the recorded powers and checks the design and the acceptance, so the paper's
test record (``tools/tasks.py paper-junit``) carries a VA-1 node id. It runs no emulator.

Where the record is looked up: ``$SREMU_VA1_RECORD`` when set (e.g. a regeneration job's
``paper_results/E-A-IQ`` before install; a missing record there fails the test), otherwise
``$SREMU_PAPER_DIR/sweeps/results/E-A-IQ`` with ``<repo>/../paper`` as the default paper
directory. Without ``$SREMU_VA1_RECORD`` the test skips, with the reason, only when no record
exists (CI has no paper repository). Whether
every row comes from one clean commit is checked by ``experiments/compare_regen.py`` before
the regenerated record is installed, and the paper lists each row's commit and host in its
provenance table (``tools/gen_results.py``); this test does not repeat that check.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path

import pytest
from experiments import ea_iq

pytestmark = pytest.mark.validation_smoke

REPO = Path(__file__).resolve().parents[2]
VA1_TOL_DB = 0.02  # pre-specified VA-1 limit (DESIGN.md section 11, E-A; paper numVaOneTolDb)
RECOMPUTE_ATOL_DB = 1e-9  # recorded diff column vs the difference of the recorded powers


@pytest.fixture(scope="module")
def va1_record() -> Path:
    explicit = os.environ.get("SREMU_VA1_RECORD")
    if explicit:
        if not (Path(explicit) / "va1.csv").is_file():
            pytest.fail(f"SREMU_VA1_RECORD={explicit} holds no va1.csv")
        return Path(explicit)
    paper = Path(os.environ.get("SREMU_PAPER_DIR", REPO.parent / "paper"))
    record = paper / "sweeps" / "results" / "E-A-IQ"
    if (record / "va1.csv").is_file():
        return record
    pytest.skip(
        f"VA-1 record absent (no va1.csv in {record}); set SREMU_VA1_RECORD or SREMU_PAPER_DIR"
    )


def _rows(record: Path) -> list[dict[str, str]]:
    with (record / "va1.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_va1_record_meets_the_pre_specified_tolerance(va1_record: Path) -> None:
    meta = json.loads((va1_record / "meta.json").read_text(encoding="utf-8"))
    manifest = json.loads((va1_record / "manifest.json").read_text(encoding="utf-8"))
    assert meta["experiment"] == "E-A-IQ"
    assert "git_commit" in manifest, "the aggregate has no run manifest"
    # The record states the rule it was judged by; it must be the pre-specified one.
    rule = next((a for a in meta["assumptions"] if a.startswith("acceptance (noise off)")), None)
    assert rule is not None, "meta.json does not state the noise-off acceptance rule"
    m = re.search(r"<= (\d+(?:\.\d+)?) dB", rule)
    assert m is not None
    assert float(m.group(1)) == VA1_TOL_DB

    rows = _rows(va1_record)
    ea_iq.check_va1_complete(rows)  # every design row once, each a full-length run
    assert meta["complete"] is True
    assert meta["expected_rows"] == len(rows) == len(ea_iq.expected_va1_keys())

    diffs = []
    for r in rows:
        key = (r["controller"], r["tau_sui_s"], r["noise"])
        p_iq, ref = float(r["p_iq_dbm"]), float(r["p_vectorized_mid_dbm"])
        diff = p_iq - ref
        assert math.isfinite(diff), key
        assert abs(diff - float(r["diff_iq_minus_vectorized_mid_db"])) <= RECOMPUTE_ATOL_DB, key
        if r["noise"] == "off":
            accept = abs(diff) <= VA1_TOL_DB
        else:
            accept = float(r["iq_ci_low_dbm"]) <= ref <= float(r["iq_ci_high_dbm"])
        assert accept, f"VA-1 row {key} fails its acceptance (IQ - batch = {diff:.6f} dB)"
        assert r["accept"] == str(accept), key
        diffs.append(abs(diff))
    max_abs = max(diffs)
    # The paper quotes the largest |IQ - batch| over every row against the noise-off limit.
    assert max_abs <= VA1_TOL_DB, f"largest |IQ - batch| {max_abs:.6f} dB > {VA1_TOL_DB} dB"
    assert meta["all_accept"] is True
