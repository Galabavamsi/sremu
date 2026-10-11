"""V-LAB pre-registration lock (DESIGN.md section 10.9, "Pre-registration"; WP10).

These tests never touch the digitized measurements: every data file used here is a
synthetic stand-in written to a temporary directory, and ``$SREMU_PRIVATE_DATA`` is pointed
at that directory (or removed).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from validation.ants_fig4b import protocol_lock as pl
from validation.ants_fig4b.protocol import (
    PROTOCOL_DIR,
    PROTOCOL_JSON,
    PROTOCOL_MD,
    canonical_json_bytes,
    cell_layout,
    load_protocol,
)

pytestmark = pytest.mark.unit


def _copy_protocol(tmp_path: Path) -> Path:
    d = tmp_path / "proto"
    d.mkdir()
    for f in (PROTOCOL_MD, PROTOCOL_JSON):
        shutil.copyfile(PROTOCOL_DIR / f, d / f)
    return d


def _synthetic_csv(path: Path, header: str = "theta_deg,r_m,rx_power_dbm,unc_db") -> np.ndarray:
    """A synthetic (not measured) map in shuffled row order; returns y in cell order."""
    lay = cell_layout(load_protocol())
    y = -10.0 - 8.0 * np.log2(lay.r_m / 0.6) - 0.1 * lay.theta_deg
    order = np.arange(lay.n)[::-1]
    lines = [header] + [f"{lay.theta_deg[i]:.1f},{lay.r_m[i]:.2f},{y[i]:.4f},0.07" for i in order]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return y


@pytest.fixture
def no_data_access(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Fail the test if the measured-file reader is ever reached."""
    calls: list[Path] = []

    def boom(path: Path) -> bytes:
        calls.append(path)
        raise AssertionError(f"data file accessed: {path}")

    monkeypatch.setattr(pl, "_read_data_file", boom)
    return calls


def test_committed_constant_is_empty_or_matches_the_files() -> None:
    """Before the lock the constant is the empty placeholder; after the lead's lock commit it
    must equal the digest of the committed files (any later edit breaks CI)."""
    value = pl.PREREGISTERED_SHA256
    if value == "":
        assert not pl.is_locked()
        return
    assert re.fullmatch(r"[0-9a-f]{64}", value)
    assert pl.protocol_digest() == value


def test_digest_definition() -> None:
    md = (PROTOCOL_DIR / PROTOCOL_MD).read_bytes().replace(b"\r\n", b"\n")
    obj = json.loads((PROTOCOL_DIR / PROTOCOL_JSON).read_text(encoding="utf-8"))
    cj = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert pl.protocol_digest() == hashlib.sha256(md + cj).hexdigest()
    assert pl.digest_input_bytes() == md + cj


def test_refuses_when_not_locked_without_touching_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_data_access: list[Path]
) -> None:
    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", "")
    f = tmp_path / "fig4b_measured_rx_power.csv"
    _synthetic_csv(f)
    monkeypatch.setenv("SREMU_PRIVATE_DATA", str(tmp_path))
    with pytest.raises(pl.ProtocolNotLocked):
        pl.load_measured()
    with pytest.raises(pl.ProtocolNotLocked):
        pl.load_measured(f)
    with pytest.raises(pl.ProtocolNotLocked):
        pl.check_lock()
    assert no_data_access == []


def test_refuses_on_wrong_hash_without_touching_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_data_access: list[Path]
) -> None:
    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", "0" * 64)
    f = tmp_path / "x.csv"
    _synthetic_csv(f)
    with pytest.raises(pl.ProtocolHashMismatch):
        pl.load_measured(f)
    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", "not-a-digest")
    with pytest.raises(pl.ProtocolHashMismatch):
        pl.load_measured(f)
    assert no_data_access == []


def test_refuses_after_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_data_access: list[Path]
) -> None:
    d = _copy_protocol(tmp_path)
    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", pl.protocol_digest(d))
    assert pl.check_lock(d) == pl.protocol_digest(d)
    # a one-character change of PROTOCOL.md
    md = (d / PROTOCOL_MD).read_text(encoding="utf-8")
    (d / PROTOCOL_MD).write_text(
        md.replace("sigma_tot = 2.0 dB", "sigma_tot = 2.5 dB", 1), encoding="utf-8"
    )
    with pytest.raises(pl.ProtocolHashMismatch):
        pl.load_measured(tmp_path / "x.csv", protocol_dir=d)
    # restore it, then change one number of protocol.json
    (d / PROTOCOL_MD).write_text(md, encoding="utf-8")
    pl.check_lock(d)
    obj = json.loads((d / PROTOCOL_JSON).read_text(encoding="utf-8"))
    obj["decision_rules"]["C2"]["median_delta_rmse_min_db"] = 0.9
    (d / PROTOCOL_JSON).write_text(json.dumps(obj, indent=2), encoding="utf-8")
    with pytest.raises(pl.ProtocolHashMismatch):
        pl.load_measured(tmp_path / "x.csv", protocol_dir=d)
    assert no_data_access == []


def test_digest_ignores_line_endings_and_json_layout(tmp_path: Path) -> None:
    d = _copy_protocol(tmp_path)
    ref = pl.protocol_digest(d)
    md = (d / PROTOCOL_MD).read_bytes().replace(b"\r\n", b"\n")
    (d / PROTOCOL_MD).write_bytes(md.replace(b"\n", b"\r\n"))
    obj = json.loads((d / PROTOCOL_JSON).read_text(encoding="utf-8"))
    (d / PROTOCOL_JSON).write_text(json.dumps(obj, indent=7, sort_keys=False), encoding="utf-8")
    assert pl.protocol_digest(d) == ref
    assert canonical_json_bytes(obj) == canonical_json_bytes(json.loads(json.dumps(obj, indent=1)))


def test_loads_synthetic_file_when_locked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    d = _copy_protocol(tmp_path)
    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", pl.protocol_digest(d))
    data_root = tmp_path / "private"
    (data_root / "fig4b_measured").mkdir(parents=True)
    f = data_root / "fig4b_measured" / "fig4b_measured_rx_power.csv"
    y = _synthetic_csv(f)
    monkeypatch.setenv("SREMU_PRIVATE_DATA", str(data_root))
    md = pl.load_measured(protocol_dir=d)
    np.testing.assert_allclose(md.y_dbm, np.round(y, 4))
    np.testing.assert_array_equal(md.sigma_dig_db, np.full(30, 0.07))
    assert md.sha256 == hashlib.sha256(f.read_bytes()).hexdigest()
    assert md.protocol_digest == pl.protocol_digest(d)
    assert md.data_class == "DIG"
    assert md.columns["y_dbm"] == "rx_power_dbm"


def test_missing_env_after_lock_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    d = _copy_protocol(tmp_path)
    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", pl.protocol_digest(d))
    monkeypatch.delenv("SREMU_PRIVATE_DATA", raising=False)
    with pytest.raises(FileNotFoundError, match="SREMU_PRIVATE_DATA"):
        pl.load_measured(protocol_dir=d)


@pytest.mark.parametrize(
    "header",
    [
        "theta_deg,r_m,rx_power_dbm,unc_db",
        "Theta (deg),R [m],P_rx [dBm],sigma_dB",
        "angle_deg,radius_m,power_dbm,uncertainty_db",
    ],
)
def test_csv_header_matching(tmp_path: Path, header: str) -> None:
    f = tmp_path / "a.csv"
    y = _synthetic_csv(f, header)
    yy, sg, cmap = pl.parse_measured_csv(f.read_text(encoding="utf-8"), load_protocol())
    np.testing.assert_allclose(yy, np.round(y, 4))
    np.testing.assert_array_equal(sg, np.full(30, 0.07))
    assert set(cmap) == {"theta_deg", "r_m", "y_dbm", "sigma_db"}


def test_csv_rejects_ambiguous_or_incomplete(tmp_path: Path) -> None:
    proto = load_protocol()
    f = tmp_path / "a.csv"
    _synthetic_csv(f, "theta_deg,r_m,power_dbm,rx_dbm,unc_db")
    text = f.read_text(encoding="utf-8")
    with pytest.raises(pl.MeasuredDataError, match="cannot map"):
        pl.parse_measured_csv(text, proto)
    _synthetic_csv(f)
    lines = f.read_text(encoding="utf-8").splitlines()
    with pytest.raises(pl.MeasuredDataError, match="missing"):
        pl.parse_measured_csv("\n".join(lines[:-1]), proto)
    with pytest.raises(pl.MeasuredDataError, match="duplicate"):
        pl.parse_measured_csv("\n".join([*lines, lines[-1]]), proto)
    assert ",0.60," in lines[-1]
    bad = [*lines[:-1], lines[-1].replace(",0.60,", ",0.70,")]
    with pytest.raises(pl.MeasuredDataError, match="not a protocol cell"):
        pl.parse_measured_csv("\n".join(bad), proto)


def test_run_cli_refuses_a_dirty_tree_without_touching_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_data_access: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reviewer addition: the lock binds the protocol files only, so the unblinded run also
    needs committed code. A dirty (or non-git) tree is refused before any data access unless
    --allow-dirty is given; with it, the gate passes and the data reader is reached."""
    from validation.ants_fig4b import run

    from sremu.record.manifest import GitState

    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", pl.protocol_digest())
    f = tmp_path / "x.csv"
    _synthetic_csv(f)
    base = [
        "--tensor",
        str(tmp_path / "missing.npz"),
        "--tensor-sor",
        str(tmp_path / "missing_sor.npz"),
        "--out",
        str(tmp_path / "runs"),
    ]
    for state in (GitState("a" * 40, True, b"patch"), GitState(None, None, None, "no git")):
        monkeypatch.setattr(run, "git_state", lambda *a, _s=state, **k: _s)
        rc = run.main([*base, "--data", str(f)])
        assert rc == 2
        assert "dirty" in capsys.readouterr().err
        assert no_data_access == []
        assert not (tmp_path / "runs").exists()
    monkeypatch.setattr(run, "git_state", lambda *a, **k: GitState("a" * 40, True, b"patch"))
    with pytest.raises(AssertionError, match="data file accessed"):
        run.main([*base, "--data", str(f), "--allow-dirty"])
    assert no_data_access == [f]


def test_run_cli_refuses_while_unlocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_data_access: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from validation.ants_fig4b import run

    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", "")
    rc = run.main(["--tensor", str(tmp_path / "missing.npz"), "--out", str(tmp_path / "runs")])
    assert rc == 2
    assert "not locked" in capsys.readouterr().err
    assert no_data_access == []
    assert not (tmp_path / "runs").exists()


def test_run_cli_refuses_without_the_s_or_tensor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    no_data_access: list[Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Protocol v2: the unblinded run needs the pre-computed S-OR tensor (sensitivity run
    S-OR); without it the run is refused before any data access."""
    from validation.ants_fig4b import run

    from sremu.record.manifest import GitState

    monkeypatch.setattr(pl, "PREREGISTERED_SHA256", pl.protocol_digest())
    monkeypatch.setattr(run, "git_state", lambda *a, **k: GitState("a" * 40, False, None))
    f = tmp_path / "x.csv"
    _synthetic_csv(f)
    rc = run.main(
        [
            "--tensor",
            str(tmp_path / "missing.npz"),
            "--out",
            str(tmp_path / "runs"),
            "--data",
            str(f),
        ]
    )
    assert rc == 2
    assert "S-OR" in capsys.readouterr().err
    assert no_data_access == []
    assert not (tmp_path / "runs").exists()
