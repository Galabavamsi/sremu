"""B2 feature set F5: the end-to-end RTF metric, the point filter and the client CPU placement.

Pins the metric of :func:`bench._engine_point.measure_zmq` without processes
(:func:`bench._engine_point.f5_rates` on synthetic timelines, :class:`ServerMarks`) and the
B2 plumbing added with it (``--features`` / ``--points``, ``--f5-clients separate``). The
process-level checks (injected client pause, wall cap) are in ``test_bench_zmq_point.py``.

Why: the pre-fix F5 RTF was timed by the RX client from the moment it had *read* its warm-up
samples. A client starved before that moment (on AWS the clients shared each slot's CPUs with
16 numba threads) lets the served engine run ahead into the loopback/ZMQ buffers and then
reads the backlog in a burst, so its post-warm-up span shrinks and the RTF inflates (7-9
instead of ~0.17 in a WSL2 test with a 3 s pause); the inflated pilot then sized repetitions
far beyond ``rep_wall_target_s``. The end-to-end span starts at the server's warm-up boundary,
so a client cannot shrink it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bench import _common, _cpus, _engine_point, b2_lockstep

pytestmark = pytest.mark.unit

MS = 1_000_000  # ns
N_WARM = 1_000
RATE = 100_000.0  # 0.01 s warm-up, 0.1 s after it in every timeline below


def _client(t_warm: int | None, t_end: int, *, got: int = 11_000, t0: int = 200 * MS) -> dict:
    return {
        "got_min": got,
        "t_start_ns": 100 * MS,
        "t_resume_ns": t0,
        "t_warm_ns": t_warm,
        "t_end_ns": t_end,
        "samples_after_warm_total": max(0, got - N_WARM),
    }


SERVER = {"t_warm_ns": 210 * MS, "k_min": 11_000, "t_last_ns": 310 * MS}  # engine RTF 1.0


def _rates(client: dict, server: dict = SERVER, done: int = 900 * MS) -> dict[str, Any]:
    return _engine_point.f5_rates(
        client, server, n_warm=N_WARM, rate=RATE, t_launch_ns=0, t_done_ns=done
    )


def test_unstarved_client_all_three_figures_agree() -> None:
    r = _rates(_client(211 * MS, 311 * MS))
    assert r["clock_ok"] and r["rtf_basis"] == "end-to-end"
    assert r["virtual_s"] == pytest.approx(0.1)
    assert r["rtf_server"] == pytest.approx(1.0)
    assert r["rtf"] == pytest.approx(0.1 / 0.101)
    assert r["rtf_rx_client"] == pytest.approx(1.0)
    assert r["rtf"] <= min(r["rtf_server"], r["rtf_rx_client"])
    assert r["aggregate_sps"] == pytest.approx(10_000 / 0.101)


def test_client_starved_before_its_warm_up_inflates_only_the_client_figure() -> None:
    """The client slept 50 ms before reading its warm-up samples while the server kept
    computing into the buffers; it read the backlog in a burst and still ended 1 ms after the
    server. Old figure: 2x; end-to-end figure: unchanged."""
    ref = _rates(_client(211 * MS, 311 * MS))
    r = _rates(_client(261 * MS, 311 * MS))
    assert r["rtf"] == pytest.approx(ref["rtf"])
    assert r["rtf_rx_client"] == pytest.approx(2.0)
    assert r["rtf_rx_client"] > 1.9 * r["rtf"]


def test_starvation_longer_than_the_run_lowers_the_end_to_end_figure() -> None:
    """A 3 s pause: the samples reached the client 3 s late. The client figure explodes
    (the WSL2 symptom), the end-to-end figure falls, it never rises above the engine's."""
    r = _rates(_client(3_200 * MS, 3_201 * MS), done=3_300 * MS)
    assert r["rtf_rx_client"] == pytest.approx(100.0)
    assert r["rtf"] == pytest.approx(0.1 / 2.991)
    assert r["rtf"] < r["rtf_server"]


def test_process_clocks_not_comparable_fall_back_to_the_smaller_figure() -> None:
    """Client timestamps from a process-relative clock (Python < 3.10 on Windows): the
    ordering check fails and the figure falls back to min(server, client), flagged."""
    c = {
        **_client(1 * MS, 101 * MS),
        "t_start_ns": 10,
        "t_resume_ns": 20,
    }
    r = _rates(c)
    assert not r["clock_ok"] and r["rtf_basis"].startswith("min(server, rx client)")
    assert r["rtf"] == pytest.approx(min(r["rtf_server"], r["rtf_rx_client"]))


def test_run_truncated_before_the_warm_up_has_only_the_whole_run_rate() -> None:
    r = _rates(
        _client(None, 260 * MS, got=600),
        {"t_warm_ns": None, "k_min": 640, "t_last_ns": 259 * MS},
    )
    assert r["rtf"] is None and r["rtf_rx_client"] is None and r["rtf_server"] is None
    assert r["rtf_whole_run"] == pytest.approx(0.006 / 0.06)
    assert r["clock_ok"]


def test_server_marks_warm_boundary_needs_every_rx_stream() -> None:
    m = _engine_point.ServerMarks(["a", "b"], n_warm=128)
    m("a", 0, 64)
    m("b", 0, 64)
    m("zz", 0, 4096)  # not one of the measured streams
    m("a", 64, 128)
    assert m.t_warm_ns is None
    m("b", 64, 128)
    t = m.t_warm_ns
    assert t is not None
    m("a", 128, 192)
    m("b", 128, 192)
    assert m.t_warm_ns == t  # first crossing only
    k, t_last = m.progress()
    assert k == 192 and t_last is not None and t_last >= t
    assert "end-to-end" in _engine_point.F5_METRIC


# =========================================================================== B2 plumbing


def test_filter_points_by_feature_and_id() -> None:
    pts = b2_lockstep.server_points(16, host_cpus=192)
    f5, rest = b2_lockstep.filter_points(pts, ["F5"])
    assert f5 and all(p["F"] == "F5" for p in f5) and len(f5) + len(rest) == len(pts)
    # the four F5 series of the server plan: base and the three anchors, L = 1..128 each
    assert {(p["N"], p["R"], p["fs"]) for p in f5} == {
        (240, 1, "1e6"),
        (240, 1, "5.76e6"),
        (4096, 1, "1e6"),
        (240, 2, "23.04e6"),
    }
    assert len(f5) == 32
    ids = ["L1_N240_R1_fs1e6_F5", "L4_N4096_R1_fs1e6_F5"]
    one, _ = b2_lockstep.filter_points(pts, None, ids)
    assert [b2_lockstep.point_id(p) for p in one] == ids
    with pytest.raises(ValueError, match="not in the plan"):
        b2_lockstep.filter_points(pts, None, ["L3_N240_R1_fs1e6_F5"])
    with pytest.raises(ValueError, match="unknown feature"):
        b2_lockstep.filter_points(pts, ["F9"])


def _c7i_48xlarge() -> list[_cpus.Core]:
    """96 cores (SMT siblings c, c + 96), NUMA node 0 = cores 0-47, node 1 = 48-95."""
    return [_cpus.Core(0 if c < 48 else 1, (c, c + 96)) for c in range(96)]


def test_client_cpu_sets_use_free_cores_of_the_slot_node() -> None:
    cores = _c7i_48xlarge()
    slots, reserved = _cpus.plan_slots(cores, 16, reserve_cores=1)
    assert len(slots) == 11  # the B2 layout of run 20261011T084048Z
    sets = b2_lockstep.client_cpu_sets(cores, slots, reserved)
    used = {c for s in slots for c in s} | set(reserved)
    node0_free = [*range(41, 48), *range(137, 144)]
    for i, s in enumerate(slots):
        assert sets[i] and not used.intersection(sets[i])
        if s[0] < 48:  # node-0 slots split cores 41-47 among themselves (2, 2, 1, 1, 1)
            assert set(sets[i]) <= set(node0_free)
        else:  # node 1 has no free core: its slots share every free CPU of the host
            assert sets[i] == sorted(node0_free)
    assert [len(sets[i]) for i in range(5)] == [4, 4, 2, 2, 2]
    assert len({c for i in range(5) for c in sets[i]}) == 14  # disjoint
    # --workers 4: the first four slots (node 0) get dedicated cores 33-47 of their node
    four = b2_lockstep.client_cpu_sets(cores, slots[:4], reserved)
    assert [len(x) for x in four] == [8, 8, 8, 6]
    assert all(min(x) >= 33 and max(x) < 144 for x in four)
    # every CPU in a slot or reserved: no client CPUs (the worker falls back to its slot)
    full = [list(k.cpus) for k in cores[1:]]
    assert b2_lockstep.client_cpu_sets(cores, full, [0, 96]) == [[]] * len(full)


def test_b2_point_filter_runs_only_f5_and_accounts_for_every_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.bench.test_bench_suites import _check_accounting, _fake_result

    calls: list[str] = []

    def fake_measure(p: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
        calls.append(b2_lockstep.point_id(p))
        return _fake_result(p, 0.5, settings["rtf_floor"])

    monkeypatch.setattr(b2_lockstep, "measure_point", fake_measure)
    monkeypatch.setattr(b2_lockstep, "jit_warmup", lambda features="F3": None)
    monkeypatch.setattr(_common.BenchRun, "save_scenario", lambda self, pid, res: "0" * 64)
    monkeypatch.delenv("SREMU_CLIENT_CPUS", raising=False)
    common = ["--quick", "--rtf-floor", "0", "--point-quiet-window-s", "0"]
    rc = b2_lockstep.main([*common, "--features", "F5", "--out", str(tmp_path / "a")])
    assert rc == 0
    (run_dir,) = list((tmp_path / "a").iterdir())
    rows, dropped = _check_accounting(run_dir)
    assert calls == ["L1_N240_R1_fs1e6_F5", "L4_N240_R1_fs1e6_F5"]
    assert [r["features"] for r in rows] == ["F5", "F5"]
    assert all(r["client_affinity_mode"] == "unpinned" for r in rows)
    filt = [d for d in dropped if d["reason"] == b2_lockstep.FILTER_REASON]
    assert filt and all(d["F"] != "F5" for d in filt)
    spec = json.loads((run_dir / "spec.json").read_text())
    assert spec["point_filter"]["features"] == ["F5"] and len(spec["points"]) == 2
    summ = json.loads((run_dir / "summary.json").read_text())
    assert summ["points_filtered"] == len(filt) and summ["points_run"] == 2
    # explicit ids; an id outside the plan is refused before any run directory is made
    calls.clear()
    rc = b2_lockstep.main(
        [*common, "--points", "L4_N240_R1_fs1e6_F5", "--out", str(tmp_path / "b")]
    )
    assert rc == 0 and calls == ["L4_N240_R1_fs1e6_F5"]
    (run_b,) = list((tmp_path / "b").iterdir())
    _check_accounting(run_b)
    rc = b2_lockstep.main(
        [*common, "--points", "L2_N240_R1_fs1e6_F5", "--out", str(tmp_path / "c")]
    )
    assert rc == 2 and not (tmp_path / "c").exists()


def test_b2_row_carries_the_f5_diagnostics() -> None:
    from tests.bench.test_bench_suites import _fake_result

    p = {"L": 1, "N": 240, "R": 1, "fs": "1e6", "F": "F5", "T": None, "slices": ["S2"]}
    r = _fake_result(p, 0.4, None)
    r.update(dict.fromkeys(b2_lockstep.F5_ROW_KEYS))
    r.update(rtf_rx_client_median=0.9, rtf_server_median=0.41, client_cpus_requested="41,137")
    row = b2_lockstep._row(p, r, None, 1.0, 16)
    assert row["rtf_median"] == 0.4 and row["rtf_rx_client_median"] == 0.9
    assert row["rtf_server_median"] == 0.41 and row["client_cpus"] == "41,137"
    plain = b2_lockstep._row({**p, "F": "F4"}, _fake_result(p, 0.4, None), None, 1.0, 16)
    assert "rtf_rx_client_median" not in plain
