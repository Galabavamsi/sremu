"""Control-plane codecs and schema conformance (DESIGN.md sections 8.5, 4.9; WP8).

* Every codec is exact: observations (float64 through JSON ``repr``, arrays as base64 of
  little-endian bytes, non-finite powers), capabilities (non-finite floats), commands (all
  ops, all ``config``/``at``/``elements`` forms; ``phases_rad`` bit-exact) and receipts
  round-trip to equal objects.
* Messages the server actually produces (replies, ``ctrl.barrier``, ``stream.async`` and PUB
  telemetry captured from a served run) validate against the shipped schema files, with the
  package's validator and with the reference ``jsonschema`` implementation.
* ``sremu schema control`` exports the schema files.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction

import numpy as np
import pytest
import zmq
from typer.testing import CliRunner

from sremu import rng
from sremu.cli import build_app
from sremu.contracts import (
    CommandReceipt,
    CommandRequest,
    CommandTime,
    CsiView,
    LatencyModel,
    Observation,
    RisCapabilities,
    StreamCmd,
    StreamSpec,
    TransitionSpec,
    TxMetadata,
)
from sremu.control.remote import RemoteControllerRunner
from sremu.engine.streams import ToneSource
from sremu.errors import ErrorCode
from sremu.timebase import TimeSpec, fraction_to_str, to_ticks
from sremu.transport.client import ClientSession, RemoteEngine
from sremu.transport.messages import (
    SCHEMA_FILES,
    caps_from_json,
    caps_to_json,
    command_from_request,
    command_to_request,
    decode_json,
    encode_json,
    observation_from_json,
    observation_to_json,
    parse_request,
    receipt_from_reply,
    receipt_reply,
    schema,
    validator,
)
from tests.io._wp8_support import bits, served, wait_until
from tests.runtime._engine_support import (
    ScriptedController,
    alternate_script,
    lab_like_fake,
    make_engine,
)

pytestmark = pytest.mark.io

F = Fraction(10**6)


def _wire(obj: dict) -> dict:
    return decode_json(encode_json(obj))


def test_rt08_observation_roundtrip_is_exact() -> None:
    g = rng.generator(7, "trial", "wp8/messages")
    z = (g.standard_normal((2, 5)) + 1j * g.standard_normal((2, 5))).astype(np.complex128)
    obs = Observation(
        t_b=Fraction(7, 3000),
        k_b={"a.rx0": 2333, "b.rx0": 17},
        rx_power_dbm={"a.rx0": -71.123456789012345, "b.rx0": -math.inf, "c.rx0": math.nan},
        positions={"a": np.array([0.1, 1e-300, -3.0]), "b": np.array([1 / 3, 2.0, 5e10])},
        csi_oracle={
            "x.tx0->a.rx0": CsiView(
                "x.tx0->a.rx0", Fraction(7, 3000), 3.5e9, 0.1 / 3 - 2j / 7, {"ris0": z}
            )
        },
        iq_window={"a.rx0": (g.standard_normal(9) + 1j).astype(np.complex64)},
        ris_state={"ris0": np.array([-1, 0, 1, 32767], np.int16)},
        pending=(
            CommandReceipt(
                "c:1:0", True, "scheduled", seq=4, t_req=Fraction(1, 3), k_req={"a.rx0": 333333}
            ),
        ),
    )
    back = observation_from_json(_wire(observation_to_json(obs)))
    assert back.t_b == obs.t_b and back.k_b == obs.k_b
    assert back.rx_power_dbm["a.rx0"] == obs.rx_power_dbm["a.rx0"]
    assert back.rx_power_dbm["b.rx0"] == -math.inf and math.isnan(back.rx_power_dbm["c.rx0"])
    for n in obs.positions:
        assert np.array_equal(bits(back.positions[n]), bits(obs.positions[n]))
    v, w = back.csi_oracle["x.tx0->a.rx0"], obs.csi_oracle["x.tx0->a.rx0"]
    assert (v.t, v.fc_hz, v.h_direct) == (w.t, w.fc_hz, w.h_direct)
    assert v.cascade["ris0"].dtype == np.complex128
    assert np.array_equal(bits(v.cascade["ris0"]), bits(w.cascade["ris0"]))
    assert np.array_equal(bits(back.iq_window["a.rx0"]), bits(obs.iq_window["a.rx0"]))
    assert back.ris_state["ris0"].dtype == np.int16
    assert np.array_equal(back.ris_state["ris0"], obs.ris_state["ris0"])
    p = back.pending[0]
    assert (p.cmd_id, p.seq, p.t_req, dict(p.k_req)) == (
        "c:1:0",
        4,
        Fraction(1, 3),
        {"a.rx0": 333333},
    )


def test_rt08_caps_roundtrip_with_nonfinite_floats() -> None:
    caps = {
        "ris0": RisCapabilities(
            120,
            (24, 5),
            "element",
            ("s0", "s1"),
            "analytic:fake",
            (3.0e9, math.inf),
            math.inf,
            Fraction(2, 10**5),
            1e4,
            math.nan,
        ),
        "ris1": RisCapabilities(
            4, (2, 2), "tile", ("a",), "measured:" + "0" * 64, (1e9, 2e9), 1e8, n_controls=2
        ),
    }
    back = caps_from_json(_wire(caps_to_json(caps)))
    assert back["ris1"] == caps["ris1"]
    b0 = back["ris0"]
    assert b0.operating_band_hz == (3.0e9, math.inf) and b0.max_bandwidth_hz == math.inf
    assert b0.reconfiguration_time_s == Fraction(2, 10**5) and math.isnan(b0.static_power_w)


def _requests() -> list[CommandRequest]:
    st = np.array([0, 1, -1, 1], np.int16)
    return [
        CommandRequest(
            "a",
            "ris.set_config",
            "c1",
            "ris0",
            CommandTime.at(Fraction(1, 3)),
            states=st,
            transition=TransitionSpec("linear", Fraction(2, 10**5), "by_edge"),
            late_policy="reject",
        ),
        CommandRequest(
            "b",
            "ris.set_config",
            "c1",
            "ris0",
            CommandTime.at_sample("rx1.rx0", 99),
            codebook_index=3,
            elements=(0, 2, 5),
        ),
        CommandRequest(
            "c",
            "ris.set_config",
            "c1",
            "ris0",
            CommandTime.asap(),
            phases_rad=np.array([0.1, -np.pi / 3, 2.0000000000000004]),
            elements="tile2",
        ),
        CommandRequest("d", "ris.set_enabled", "c1", "ris0", CommandTime.at(5), enabled=False),
        CommandRequest(
            "e",
            "node.set_position",
            "c1",
            "rx1",
            CommandTime.at(Fraction(1, 7)),
            position_m=np.array([1 / 3, -2.0, 1e-9]),
        ),
        CommandRequest(
            "f",
            "node.set_trajectory",
            "c1",
            "rx1",
            CommandTime.asap(),
            payload={
                "trajectory": {
                    "model": "constant_velocity",
                    "start_m": [0.0, 0.0, 0.0],
                    "velocity_mps": [1.5, 0.0, 0.0],
                }
            },
        ),
        CommandRequest(
            "g",
            "rec.marker",
            "c1",
            "rec",
            CommandTime.at(Fraction(1, 2)),
            payload={"label": "x", "k": 3, "s": "y"},
        ),
    ]


@pytest.mark.parametrize("req", _requests(), ids=lambda r: r.cmd_id)
def test_io03_command_roundtrip_through_the_wire(req: CommandRequest) -> None:
    msg = parse_request(encode_json(command_to_request(req)))
    back = command_from_request(msg, src=req.src)
    assert (back.cmd_id, back.op, back.src, back.target, back.at, back.late_policy) == (
        req.cmd_id,
        req.op,
        req.src,
        req.target,
        req.at,
        req.late_policy,
    )
    for name in ("states", "phases_rad", "position_m"):
        a, b = getattr(req, name), getattr(back, name)
        assert (a is None) == (b is None)
        if a is not None:
            assert a.dtype == b.dtype and np.array_equal(bits(a), bits(b))
    assert back.codebook_index == req.codebook_index and back.elements == req.elements
    assert back.enabled == req.enabled and back.transition == req.transition
    assert dict(back.payload) == dict(req.payload)


def test_io03_phases_deg_form_converts_to_radians() -> None:
    msg = parse_request(
        encode_json(
            {
                "v": 1,
                "id": "p",
                "op": "ris.set_config",
                "panel": "ris0",
                "at": "asap",
                "config": {"phases_deg": [0, 90, -180, 45.5]},
            }
        )
    )
    req = command_from_request(msg, src="x")
    assert np.array_equal(req.phases_rad, np.array([0, 90, -180, 45.5]) * (np.pi / 180.0))


def test_io03_receipt_reply_roundtrip() -> None:
    ok = CommandReceipt(
        "a",
        True,
        "scheduled",
        seq=3,
        t_req=Fraction(1, 200),
        k_req={"r.rx0": 5000},
        t_wall_recv_ns=123,
    )
    r = receipt_from_reply(_wire(receipt_reply(ok)))
    assert (r.ok, r.status, r.seq, r.t_req, dict(r.k_req), r.t_wall_recv_ns) == (
        True,
        "scheduled",
        3,
        Fraction(1, 200),
        {"r.rx0": 5000},
        123,
    )
    bad = CommandReceipt("b", False, "rejected", error=ErrorCode.LATE_COMMAND, msg="late")
    rb = receipt_from_reply(_wire(receipt_reply(bad)))
    assert (rb.ok, rb.status, rb.error, rb.msg) == (
        False,
        "rejected",
        ErrorCode.LATE_COMMAND,
        "late",
    )
    assert validator("reply.schema.json").is_valid(receipt_reply(ok))
    assert validator("reply.schema.json").is_valid(receipt_reply(bad))


def _check(name: str, msg: dict) -> None:
    errs = validator(name).errors(msg)
    assert not errs, (name, msg, errs)
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - the reference check needs the dev package
        return
    jsonschema.Draft202012Validator(schema(name)).validate(msg)


def test_io01_server_messages_conform_to_the_schemas() -> None:
    eng = make_engine(
        lab_like_fake(),
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        duration_s=Fraction(21, 100),
        always_active=True,
    )  # two metrics windows
    n = eng.commands.specs["ris0"].n_controls
    with served(eng, start_paused=True) as srv:
        sub_ctx = zmq.Context()
        sub = sub_ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.connect(str(srv.endpoints["pub"]))
        s = ClientSession(str(srv.endpoints["control"]), name="probe")
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            replies = [s.info]
            for op, kw in (
                ("caps.get", {}),
                ("time.now", {}),
                ("stats.get", {}),
                ("ris.get_state", {"panel": "ris0"}),
                ("sim.pause", {}),
            ):
                replies.append(s.call(op, **kw))
            replies.append(s.request({"op": "caps.get", "panel": "nope"}))
            for r in replies:
                _check("reply.schema.json", r)
            # ctrl.barrier events, as the remote runner receives them
            ctrl = ScriptedController(
                "alt",
                "1/20",
                "2/10000",
                alternate_script("ris0", n),
                latency=LatencyModel(Fraction(2, 10_000)),
                fields=("rx_power_dbm", "ris_state", "iq_window", "csi_oracle"),
            )
            runner = RemoteControllerRunner(ctrl, str(srv.endpoints["control"]))
            seen: list[dict] = []
            orig = runner.handle_barrier

            def keep(ev: dict) -> list:
                seen.append(ev)
                return orig(ev)

            runner.handle_barrier = keep  # type: ignore[method-assign]
            runner.start()
            srv.resume()
            wait_until(srv.is_done, 60.0, "the end of the run")
            runner.stop()
            assert len(seen) == 5  # t_b = 0, 50, ..., 200 ms
            for m, e in enumerate(seen):
                _check("event.schema.json", e)
                # section 4.9: ctrl.barrier {t_b, k_b, observation}; k_b = to_ticks(t_b, f_r)
                assert e["m"] == m and e["t_b"] == fraction_to_str(Fraction(m, 20))
                assert (
                    e["k_b"]
                    == e["observation"]["k_b"]
                    == {r: to_ticks(Fraction(m, 20), F) for r in ("rx1.rx0", "rx2.rx0")}
                )
            assert seen[0]["observation"]["csi_oracle"] is not None
            assert seen[0]["observation"]["iq_window"] is not None
            pubs = []
            while sub.poll(200):
                topic, payload = sub.recv_multipart()
                body = json.loads(payload)
                assert body["topic"] == topic.decode()
                pubs.append(body)
            assert {"ris.applied", "metrics.window"} <= {b["topic"] for b in pubs}
            for b in pubs:
                _check("telemetry.schema.json", b)
        finally:
            cli.close()
            s.close()
            sub.close(0)
            sub_ctx.term()


def test_io01_stream_async_events_conform_to_the_schema() -> None:
    """BURST_ACK and TIME_ERROR of a client-fed TX stream arrive as ``stream.async``."""
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    with served(eng) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        raw: list[dict] = []
        orig = cli.session._dispatch

        def keep(msg: object) -> None:
            if isinstance(msg, dict):
                raw.append(msg)
            orig(msg)

        cli.session._dispatch = keep  # type: ignore[method-assign]
        try:
            h = cli.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
            md = TxMetadata(True, TimeSpec(0, 0), True, True)
            cli.submit_tx(h, np.ones(8, np.complex64), md)
            cli.submit_tx(h, np.ones(3, np.complex64), md)  # starts before the head: late
            got = []
            for _ in range(200):
                ev = cli.recv_async(h, 0.05)
                if ev is not None:
                    got.append(ev)
                if len(got) == 2:
                    break
            assert [e.event_code.name for e in got] == ["BURST_ACK", "TIME_ERROR"]
            assert got[1].sample_index == 0 and got[1].n_samples == 3
            assert len(raw) == 2
            for m in raw:
                _check("event.schema.json", m)
        finally:
            cli.close()


def test_io04_stream_async_events_reach_the_owner_only() -> None:
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    with served(eng) as srv:
        a = RemoteEngine(str(srv.endpoints["control"]), name="a")
        b = ClientSession(str(srv.endpoints["control"]), name="b")
        try:
            h = a.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
            a.submit_tx(h, np.ones(4, np.complex64), TxMetadata(True, TimeSpec(0, 0), True, True))
            ev = None
            for _ in range(100):
                ev = a.recv_async(h, 0.05)
                if ev is not None:
                    break
            assert ev is not None and ev.event_code.name == "BURST_ACK" and ev.sample_index == 4
            b.pump(0.2)
            assert not b.async_events and not b.events
            r = b.request({"op": "stream.cmd", "stream_id": h.id, "mode": "start_cont"})
            assert r["error"]["code"] in ("NOT_PERMITTED",)
        finally:
            a.close()
            b.close()


def test_io01_schema_control_cli_exports_every_file() -> None:
    res = CliRunner().invoke(build_app(), ["schema", "control"])
    assert res.exit_code == 0, res.output
    d = json.loads(res.output)
    assert sorted(d) == sorted(SCHEMA_FILES)
    assert d["request.schema.json"]["$id"] == "urn:sremu:control:v1:request"


def test_io03_timed_stream_cmd_is_exact_over_the_wire() -> None:
    """stream.cmd carries the exact start time of a timed UHD stream command."""
    eng = make_engine(lab_like_fake())
    with served(eng, start_paused=True) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
            cli.stream_cmd(h, StreamCmd("num_done", 10, False, TimeSpec.from_exact(Fraction(1, 3))))
            st = eng.rx["rx1.rx0"]
            assert st.streaming and st.start == 333333 and st.remaining == 10
        finally:
            cli.close()
