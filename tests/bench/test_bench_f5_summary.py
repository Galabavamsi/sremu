"""``bench.summarize`` copies the F5 columns of a B2 run (``f5_points``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from bench._common import write_csv

from bench import summarize

pytestmark = pytest.mark.unit


def _row(features: str, **kw: object) -> dict[str, object]:
    base = {
        "point_id": f"L1_N240_R1_fs1e6_{features}",
        "L": 1,
        "N": 240,
        "n_ris": 1,
        "f_s": "1e6",
        "features": features,
        "threads": 16,
        "status": "ok",
        "rtf_median": 1.45 if features == "F4" else 0.88,
        "rtf_reps": [0.87, 0.88, 0.9],
    }
    return {**base, **kw}


def test_f5_points_copy_diagnostics_settings_and_the_f4_twin(tmp_path: Path) -> None:
    d = tmp_path / "20261012T000000Z_b2_server"
    d.mkdir()
    for name, body in (("manifest.json", {"git_commit": "abc"}), ("meta.json", {})):
        (d / name).write_text(json.dumps(body), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({"suite": "b2"}), encoding="utf-8")
    write_csv(
        d / "points.csv",
        [
            _row("F4"),
            _row(
                "F5",
                rtf_rx_client_median=0.9,
                rtf_server_median=0.91,
                client_affinity_mode="separate",
                client_cpus="161-162",
                client_affinity={"rx": "161-162", "tx": "161-162"},
                clock_shared=True,
                rx_frame_samples=4096,
                tx_chunk_samples=4096,
            ),
            _row("F5", point_id="L2_N240_R1_fs1e6_F5", L=2, status="failed", rtf_median=None),
        ],
    )
    res = summarize.summarize(tmp_path)
    pts = res["b2_server"]["latest"]["f5_points"]
    assert [p["point_id"] for p in pts] == ["L1_N240_R1_fs1e6_F5", "L2_N240_R1_fs1e6_F5"]
    p = pts[0]
    assert (p["rtf_median"], p["rtf_rx_client_median"], p["rtf_server_median"]) == (
        0.88,
        0.9,
        0.91,
    )
    assert p["rtf_reps"] == [0.87, 0.88, 0.9]
    assert (p["client_affinity_mode"], p["client_cpus"]) == ("separate", "161-162")
    assert p["client_affinity"] == {"rx": "161-162", "tx": "161-162"}
    assert p["clock_shared"] is True
    assert (p["rx_frame_samples"], p["tx_chunk_samples"]) == (4096, 4096)
    assert p["f4_rtf_median"] == 1.45  # the F4 point with the same L, N, R, f_s, threads
    assert pts[1]["status"] == "failed" and pts[1]["f4_rtf_median"] is None
    assert pts[1]["rtf_median"] is None


def test_f5_points_absent_without_points_csv(tmp_path: Path) -> None:
    assert summarize.f5_points(tmp_path) is None
