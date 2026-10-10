"""events.jsonl: canonical append-only command log (DESIGN.md 3.6, 4.6, 8.6; contradiction 17).

Oracles: the line format of :mod:`sremu.record.eventlog` (``CommandRecord.to_dict()`` plus
``request``), bitwise round trips of the request payloads, the event-sourcing rule (the last
line of a ``seq`` is its final state) and flush-per-line durability.
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import CommandRecord, CommandRequest, CommandTime, TransitionSpec
from sremu.record.eventlog import (
    EventLog,
    iter_event_lines,
    line_of,
    read_events,
    request_from_json,
    request_to_json,
)

pytestmark = pytest.mark.io


def _record(seq=3, status="scheduled", **kw):
    d = {
        "cmd_id": "c-7f3a",
        "seq": seq,
        "src": "ctrl-a",
        "op": "ris.set_config",
        "panel": "ris0",
        "t_req": Fraction(1, 200),
        "t_applied": Fraction(1, 200),
        "k_req": {"rx1.rx0": 5000},
        "k_applied": {"rx1.rx0": 5000},
        "lateness_samples": {"rx1.rx0": 0},
        "t_wall_recv_ns": 123,
        "t_wall_computed_ns": None,
        "t_wall_visible_ns": None,
        "status": status,
        "config_hash": "ab" * 16,
        "transition": None,
        "counterfactual": False,
    }
    d.update(kw)
    return CommandRecord(**d)


def _roundtrip(req):
    return request_from_json(json.loads(json.dumps(request_to_json(req))))


def test_eventlog_request_roundtrip_states_bitwise(rng):
    states = rng.integers(-1, 4, size=4096).astype(np.int16)
    req = CommandRequest(
        "c1",
        "ris.set_config",
        "api",
        "ris0",
        CommandTime.at("1/200"),
        states=states,
        elements=tuple(range(4096)),
        transition=TransitionSpec("linear", "2e-5", "by_edge"),
        late_policy="reject",
        t_wall_recv_ns=42,
    )
    d = request_to_json(req)
    assert d["at"] == {"t": "1/200"}
    assert d["config"]["dtype"] == "i2"
    assert d["transition"] == {"model": "linear", "duration_s": "1/50000", "classes": "by_edge"}
    back = _roundtrip(req)
    assert back.states.dtype == np.int16
    np.testing.assert_array_equal(back.states, states)
    assert back.elements == tuple(range(4096))
    assert back.transition == TransitionSpec("linear", Fraction(1, 50000), "by_edge")
    assert (back.at, back.late_policy, back.t_wall_recv_ns) == (req.at, "reject", 42)
    assert (back.cmd_id, back.op, back.src, back.target) == ("c1", "ris.set_config", "api", "ris0")


def test_eventlog_request_roundtrip_phases_float64_bitwise(rng):
    ph = rng.uniform(-np.pi, np.pi, 257)
    req = CommandRequest(
        "c2",
        "ris.set_config",
        "api",
        "ris0",
        CommandTime.at_sample("rx1.rx0", 115200),
        phases_rad=ph,
        quantizer="nearest",
        elements="tile2",
    )
    back = _roundtrip(req)
    assert back.phases_rad.view(np.uint64).tolist() == ph.view(np.uint64).tolist()
    assert back.quantizer == "nearest"
    assert back.elements == "tile2"
    assert back.at == CommandTime.at_sample("rx1.rx0", 115200)


@pytest.mark.parametrize(
    "req",
    [
        CommandRequest("c3", "ris.set_config", "a", "ris0", CommandTime.asap(), codebook_index=17),
        CommandRequest("c4", "ris.set_enabled", "a", "ris0", CommandTime.at(0), enabled=False),
        CommandRequest(
            "c5",
            "node.set_position",
            "a",
            "rx1",
            CommandTime.at("8e-3"),
            position_m=np.array([1.2, 0.3, 0.1]),
        ),
        CommandRequest(
            "c6",
            "node.set_trajectory",
            "a",
            "rx1",
            CommandTime.at(1),
            payload={"model": "static", "position_m": [1.0, 2.0, 0.0], "t": Fraction(1, 3)},
        ),
        CommandRequest("c7", "rec.marker", "a", "rec", CommandTime.at(0), payload={"label": "x"}),
    ],
)
def test_eventlog_request_roundtrip_other_ops(req):
    back = _roundtrip(req)
    assert back.op == req.op
    assert back.codebook_index == req.codebook_index
    assert back.enabled == req.enabled
    if req.position_m is not None:
        assert back.position_m.tolist() == req.position_m.tolist()
    if req.op == "node.set_trajectory":
        assert back.payload == {"model": "static", "position_m": [1.0, 2.0, 0.0], "t": "1/3"}
    elif req.payload:
        assert back.payload == dict(req.payload)


def test_eventlog_line_is_record_dict_plus_request():
    rec = _record()
    line = line_of(rec)
    assert line["request"] is None
    assert {k: v for k, v in line.items() if k != "request"} == rec.to_dict()
    assert line["t_req"] == "1/200"
    assert CommandRecord.from_dict(line).to_dict() == rec.to_dict()


def test_eventlog_append_dedupes_and_tracks_status_history(tmp_path):
    path = tmp_path / "events.jsonl"
    req = CommandRequest(
        "c-7f3a",
        "ris.set_config",
        "ctrl-a",
        "ris0",
        CommandTime.at("1/200"),
        states=np.array([1, 0, 1], np.int16),
    )
    log = EventLog(path)
    rec = _record(seq=3)
    assert log.append(rec) is True
    assert log.append(rec) is False  # identical record: no new line
    assert log.append(rec, req) is True  # first time with a request: one more line
    assert log.append(rec, req) is False
    rec.status = "applied"
    assert log.append(rec) is True
    other = _record(
        seq=4, cmd_id="c-2", status="rejected", t_applied=None, k_applied={}, lateness_samples={}
    )
    assert log.append(other) is True
    # flushed per line: a second reader sees every line before close()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == log.n_lines == 4
    assert [json.loads(x)["status"] for x in lines] == [
        "scheduled",
        "scheduled",
        "applied",
        "rejected",
    ]
    log.close()
    final = read_events(path)
    assert [f.record.seq for f in final] == [3, 4]
    assert final[0].record.status == "applied"
    assert final[0].history == ("scheduled", "scheduled", "applied")
    assert final[0].n_lines == 3
    np.testing.assert_array_equal(final[0].request.states, [1, 0, 1])
    assert final[1].request is None
    assert final[1].record.t_applied is None
    with pytest.raises(ValueError, match="closed"):
        log.append(rec)


def test_eventlog_truncated_last_line_is_ignored_and_bad_middle_line_raises(tmp_path):
    path = tmp_path / "events.jsonl"
    good = json.dumps(line_of(_record(seq=1)))
    path.write_text(good + "\n" + good[: len(good) // 2], encoding="utf-8")
    assert [o["seq"] for o in iter_event_lines(path)] == [1]
    path.write_text(good[:10] + "\n" + good + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"events\.jsonl:1"):
        list(iter_event_lines(path))


def test_eventlog_reads_a_run_directory(tmp_path):
    with EventLog(tmp_path) as log:
        log.append(_record(seq=9))
    assert (tmp_path / "events.jsonl").is_file()
    assert [f.record.seq for f in read_events(tmp_path)] == [9]


def test_eventlog_applied_config_roundtrip_and_kept_on_later_lines(tmp_path, rng):
    from sremu.contracts import PanelConfig
    from sremu.record.eventlog import AppliedConfig

    state = rng.integers(-1, 4, size=300).astype(np.int16)
    a = AppliedConfig.of(PanelConfig("ris0", state, enabled=False))
    assert AppliedConfig.of(None) is None
    back = AppliedConfig.from_json(json.loads(json.dumps(a.to_json())))
    np.testing.assert_array_equal(back.state, state)
    assert back.state.dtype == np.int16
    assert back.enabled is False
    log = EventLog(tmp_path)
    rec = _record(seq=5, status="applied")
    assert log.append(rec) is True
    assert log.append(rec, applied=a) is True  # first applied config: one more line
    assert log.append(rec, applied=a) is False
    log.close()
    lines = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text("utf-8").splitlines()]
    assert "applied_config" not in lines[0]
    assert lines[1]["applied_config"] == a.to_json()
    (final,) = read_events(tmp_path)
    np.testing.assert_array_equal(final.applied.state, state)
    assert final.request is None
