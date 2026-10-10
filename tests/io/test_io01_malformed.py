"""IO-01: malformed frames and malformed JSON (DESIGN.md sections 8.3, 8.5, 8.10, 10.4).

10**4 malformed data-plane frames (random magic, sizes, dtypes, truncation, flags, ...)
and malformed control messages reach a served emulator. Acceptance: no crash, and the
``BAD_PACKET`` / ``SCHEMA_ERROR`` counts are exact. Every malformed frame carries exactly
one defect, so its expected drop reason is known; a final valid frame on the same ``PUSH``
socket (ZMQ keeps per-connection order) marks the end of the burst.
"""

from __future__ import annotations

import base64
import contextlib
import json
import struct
from collections import Counter

import numpy as np
import pytest
import zmq

from sremu import rng
from sremu.contracts import MAX_FRAME_SAMPLES, StreamHandle, StreamSpec
from sremu.transport.client import ClientSession, RemoteEngine
from sremu.transport.framing import (
    HEADER_SIZE,
    HEADER_STRUCT,
    MAGIC,
    MAGIC_U32,
    BadFrame,
    Dtype,
    Flag,
    FrameHeader,
    make_frame,
    parse_frame,
)
from sremu.transport.messages import validator
from sremu.transport.zmq_native import RxReceiver
from tests.io._wp8_support import served, wait_until
from tests.runtime._engine_support import lab_like_fake, make_engine

pytestmark = pytest.mark.io

N_BAD = 10_000
KINDS = (
    "parts",
    "header_size",
    "magic",
    "version",
    "flags",
    "idle_with_samples",
    "nchan",
    "dtype",
    "nsamps",
    "reserved",
    "payload_size",
    "first_sample",
    "stream_id",
)
REASON = {k: k for k in KINDS} | {"idle_with_samples": "flags"}


def _hdr(sid: int, **kw: int) -> dict[str, int]:
    d = {
        "magic": MAGIC_U32,
        "version": 1,
        "flags": int(Flag.HAS_TIME | Flag.SOB),
        "stream_id": sid,
        "seq": 0,
        "first_sample": 0,
        "nsamps": 4,
        "nchan": 1,
        "dtype": 1,
        "wall_ns": 0,
        "reserved": 0,
    }
    d.update(kw)
    return d


def _pack(d: dict[str, int]) -> bytes:
    return HEADER_STRUCT.pack(
        d["magic"],
        d["version"],
        d["flags"],
        d["stream_id"],
        d["seq"],
        d["first_sample"],
        d["nsamps"],
        d["nchan"],
        d["dtype"],
        d["wall_ns"],
        d["reserved"],
    )


def malformed(kind: str, sid: int, g: np.random.Generator) -> list[bytes]:
    """One frame with exactly one defect of ``kind``."""
    payload = np.zeros(4, np.complex64).tobytes()
    if kind == "parts":
        return [_pack(_hdr(sid))] if g.random() < 0.5 else [_pack(_hdr(sid)), payload, b"x"]
    if kind == "header_size":
        n = int(g.integers(0, 100))
        n = n if n != HEADER_SIZE else 47
        return [g.bytes(n), payload]
    if kind == "magic":
        m = int(g.integers(0, 2**32))
        return [_pack(_hdr(sid, magic=m if m != MAGIC_U32 else 0)), payload]
    if kind == "version":
        v = int(g.integers(2, 2**16))
        return [_pack(_hdr(sid, version=v if g.random() < 0.9 else 0)), payload]
    if kind == "flags":
        bit = 1 << int(g.integers(7, 16))
        return [_pack(_hdr(sid, flags=int(Flag.HAS_TIME) | bit)), payload]
    if kind == "idle_with_samples":
        return [_pack(_hdr(sid, flags=int(Flag.IDLE_UNTIL))), payload]
    if kind == "nchan":
        return [_pack(_hdr(sid, nchan=int(g.choice([0, 2, 3, 65535])))), payload]
    if kind == "dtype":
        return [_pack(_hdr(sid, dtype=int(g.choice([0, 3, 4, 255, 65535])))), payload]
    if kind == "nsamps":
        n = int(g.integers(MAX_FRAME_SAMPLES + 1, 2**32))
        return [_pack(_hdr(sid, nsamps=n)), payload]
    if kind == "reserved":
        return [_pack(_hdr(sid, reserved=int(g.integers(1, 2**63)))), payload]
    if kind == "payload_size":  # truncated or extended: any length but 32 bytes
        n = int(g.integers(0, 64))
        return [_pack(_hdr(sid)), (payload + payload)[: n if n != 32 else 64]]
    if kind == "first_sample":
        return [_pack(_hdr(sid, first_sample=-int(g.integers(1, 2**62)))), payload]
    if kind == "stream_id":
        return [_pack(_hdr(sid + 1000 + int(g.integers(0, 1000)))), payload]
    raise AssertionError(kind)


def test_io01_header_layout_is_the_spec() -> None:
    assert HEADER_STRUCT.format == "<IHHIIqIHHQQ"
    assert HEADER_SIZE == 48
    assert MAGIC_U32 == 0x554D5253
    hdr = FrameHeader(int(Flag.SOB | Flag.EOB), 7, 9, 123456789012, 3, Dtype.SC16_LE, 1, 42)
    raw = hdr.pack()
    assert raw[:4] == MAGIC == b"SRMU"
    assert struct.unpack_from("<H", raw, 4)[0] == 1  # version
    assert struct.unpack_from("<q", raw, 16)[0] == 123456789012  # first_sample at offset 16
    assert struct.unpack_from("<Q", raw, 32)[0] == 42  # wall_ns at offset 32
    fr = parse_frame(make_frame(hdr, np.array([0.5 + 0.25j, -1.0, 1.0j], np.complex64)))
    assert fr.header == hdr
    # sc16 full scale 32767 (round(32767 x), saturated)
    assert np.array_equal(
        fr.samples, (np.array([16384 + 8192j, -32767, 32767j]) / 32767).astype(np.complex64)
    )


@pytest.mark.parametrize("kind", KINDS[:-1])
def test_io01_parse_frame_rejects_each_defect(kind: str) -> None:
    g = rng.generator(1, "trial", f"wp8/io01/{kind}")
    for _ in range(50):
        with pytest.raises(BadFrame) as ei:
            parse_frame(malformed(kind, 1, g))
        assert ei.value.reason == REASON[kind]


def test_io01_10k_malformed_frames_counted_exactly_no_crash() -> None:
    eng = make_engine(lab_like_fake(), unopened_tx="wait")
    g = rng.generator(2026, "trial", "wp8/io01/frames")
    with served(eng) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]))
        try:
            h = cli.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, "1e6"))
            push = cli.ctx.socket(zmq.PUSH)
            push.setsockopt(zmq.LINGER, 0)
            push.connect(str(srv.endpoints["tx"]))
            expected: Counter[str] = Counter()
            kinds = g.integers(0, len(KINDS), N_BAD)
            for k in kinds:
                kind = KINDS[int(k)]
                push.send_multipart(malformed(kind, h.id, g))
                expected[REASON[kind]] += 1
            good = FrameHeader(int(Flag.HAS_TIME | Flag.SOB), h.id, 0, 0, 10)
            push.send_multipart(make_frame(good, np.ones(10, np.complex64)))
            wait_until(lambda: srv.counters["frames_in"] == 1, 60.0, "the sentinel frame")
            st = cli.stats()
            push.close(0)
            assert st["transport"]["bad_packets"] == N_BAD
            assert st["bad_packet_reasons"] == dict(expected)
            assert st["transport"]["seq_errors"] == 0
            assert st["engine_error"] is None
            assert eng.tx_head(h) == 10  # only the sentinel reached the engine
            assert cli.session.call("hello")["ok"]  # still serving
        finally:
            cli.close()


MALFORMED_JSON: list[bytes] = [
    b"",
    b"\xff\xfe\x00{",
    b"{",
    b"[]",
    b"null",
    b"1",
    b'"hello"',
    b'{"v":1,"id":"a","op":"hello"',
    b'{"v":1,"id":"a","op":"hello","client":NaN}',
    b'{"v":1,"id":"a","op":"hello","client":Infinity}',
    b'{"v":1,"v":1,"id":"a","op":"hello"}',
    b'{"v":2,"id":"a","op":"hello"}',
    b'{"v":1,"op":"hello"}',
    b'{"v":1,"id":"a b","op":"hello"}',
    b'{"v":1,"id":"' + b"x" * 65 + b'","op":"hello"}',
    b'{"v":1,"id":"a","op":"rm -rf"}',
    b'{"v":1,"id":"a","op":"ris.set_config","panel":"ris0","at":"asap"}',
    b'{"v":1,"id":"a","op":"ris.set_config","panel":5,"at":"asap","config":{"codebook_index":1}}',
    b'{"v":1,"id":"a","op":"ris.set_config","panel":"ris0","at":{"t":0.5},"config":{"codebook_index":1}}',
    b'{"v":1,"id":"a","op":"ris.set_config","panel":"ris0","at":{"t":"1.5.3"},"config":{"codebook_index":1}}',
    b'{"v":1,"id":"a","op":"ris.set_config","panel":"ris0","at":"asap","config":{"states_b64":"!!","dtype":"i2"}}',
    b'{"v":1,"id":"a","op":"ris.set_config","panel":"ris0","at":"asap","config":{"codebook_index":1,"phases_deg":[1]}}',
    b'{"v":1,"id":"a","op":"hello","file":"/etc/passwd"}',
    b'{"v":1,"id":"a","op":"stream.open","stream":"ap.tx0","direction":"tx","fc_hz":3.5e9,"rate_sps":"1/0"}',
    b'{"v":1,"id":"a","op":"ctrl.ack","controller":"c","t_b":"-1"}',
    b"[" * 100_000 + b"]" * 100_000,
]
"""Each one must be refused with SCHEMA_ERROR (counted once)."""


def test_io01_malformed_json_counted_exactly_no_crash() -> None:
    eng = make_engine(lab_like_fake())
    g = rng.generator(2026, "trial", "wp8/io01/json")
    fuzz = [g.bytes(int(n)) for n in g.integers(1, 300, 2000)]
    with served(eng) as srv:
        s = ClientSession(str(srv.endpoints["control"]), hello=False)
        try:
            msgs = MALFORMED_JSON + fuzz
            for m in msgs:
                s.send_raw(m)
            s.sock.send_multipart([b'{"v":1,"id":"a","op":"hello"}', b"second part"])
            replies = []
            for _ in range(len(msgs) + 1):
                assert s.sock.poll(10_000)
                replies.append(json.loads(s.sock.recv()))
            assert all(r["ok"] is False for r in replies)
            assert [r["error"]["code"] for r in replies] == ["SCHEMA_ERROR"] * (len(msgs) + 1)
            # the id is echoed when it could be read
            assert replies[16]["id"] == "a"
            assert replies[0]["id"] is None
            st = s.call("stats.get")
            assert st["transport"]["schema_errors"] == len(msgs) + 1
            assert st["transport"]["too_large_messages"] == 0
            assert st["engine_error"] is None
            assert s.call("hello")["ok"]
        finally:
            s.close()


def test_io01_client_receiver_counts_bad_frames() -> None:
    """The client side drops and counts malformed RX frames too."""
    ctx = zmq.Context()
    try:
        push = ctx.socket(zmq.PUSH)
        port = push.bind_to_random_port("tcp://127.0.0.1")
        h = StreamHandle(5, StreamSpec("rx1.rx0", "rx", 3.5e9, "1e6"), 512)
        rx = RxReceiver(ctx, f"tcp://127.0.0.1:{port}", h)
        g = rng.generator(3, "trial", "wp8/io01/client")
        for i in range(200):
            push.send_multipart(malformed(KINDS[i % (len(KINDS) - 1)], 5, g))
        push.send_multipart(
            make_frame(FrameHeader(int(Flag.HAS_TIME), 5, 0, 100, 3), np.ones(3, np.complex64))
        )
        y, md = rx.read(10, 5.0)
        while not y.size:
            y, md = rx.read(10, 5.0)
        assert rx.counters["bad_packets"] == 200
        assert md.first_sample == 100 and y.shape == (3,)
        rx.close()
        push.close(0)
    finally:
        ctx.term()


# --------------------------------------------------------------------------- schema oracle


def _valid_requests() -> list[dict]:
    st = base64.b64encode(np.arange(-1, 7, dtype="<i2").tobytes()).decode()
    return [
        {"v": 1, "id": "c-1", "op": "hello", "client": "x", "contract_version": "1.1"},
        {"v": 1, "id": "c-2", "op": "caps.get"},
        {"v": 1, "id": "c-3", "op": "time.now"},
        {
            "v": 1,
            "id": "c-7f3a",
            "op": "ris.set_config",
            "src": "ctrl-a",
            "panel": "ris0",
            "at": {"t": "5/1000"},
            "config": {"states_b64": st, "dtype": "i2"},
            "elements": "all",
            "transition": {"model": "linear", "duration_s": "2e-5"},
            "late_policy": "apply_on_arrival",
        },
        {
            "v": 1,
            "id": "c-5",
            "op": "ris.set_config",
            "panel": "ris0",
            "at": {"sample": {"stream": "rx1.rx0", "index": 115200}},
            "config": {"codebook_index": 17},
            "elements": {"indices": [0, 3, 5]},
        },
        {
            "v": 1,
            "id": "c-6",
            "op": "ris.set_config",
            "panel": "ris0",
            "at": "asap",
            "config": {"phases_deg": [0.0, 90.0], "quantizer": "max_real"},
            "elements": {"group": "tile2"},
        },
        {
            "v": 1,
            "id": "c-7",
            "op": "ris.set_config",
            "panel": "ris0",
            "at": {"t": 3},
            "config": {"phases_rad": [0.25, -1.5]},
        },
        {
            "v": 1,
            "id": "c-8",
            "op": "ris.set_codebook",
            "panel": "ris0",
            "name": "default",
            "codebook_b64": st,
            "dtype": "i2",
            "shape": [2, 4],
        },
        {"v": 1, "id": "c-9", "op": "ris.get_state", "panel": "ris0"},
        {
            "v": 1,
            "id": "c-10",
            "op": "ris.set_enabled",
            "panel": "ris0",
            "at": "asap",
            "enabled": False,
        },
        {
            "v": 1,
            "id": "c-11",
            "op": "node.set_position",
            "node": "mu",
            "at": {"t": "1/2"},
            "position_m": [1.0, 2, 3.5],
        },
        {
            "v": 1,
            "id": "c-12",
            "op": "node.set_trajectory",
            "node": "mu",
            "at": "asap",
            "trajectory": {
                "model": "waypoints",
                "t_s": ["0", "1/2"],
                "positions_m": [[0, 0, 0], [1, 0, 0]],
            },
        },
        {
            "v": 1,
            "id": "c-13",
            "op": "node.set_trajectory",
            "node": "mu",
            "at": "asap",
            "trajectory": {"model": "rwp_2d", "bounds_m": [[0, 0], [5, 5]], "speed_mps": 1.0},
        },
        {
            "v": 1,
            "id": "c-14",
            "op": "stream.open",
            "stream": "ap.tx0",
            "direction": "tx",
            "fc_hz": 3.5e9,
            "rate_sps": "23.04e6",
            "cpu": "sc16",
            "nchan": 1,
        },
        {"v": 1, "id": "c-15", "op": "stream.close", "stream_id": 3},
        {
            "v": 1,
            "id": "c-16",
            "op": "stream.cmd",
            "stream_id": 2,
            "mode": "num_done",
            "num_samps": 1000,
            "stream_now": False,
            "t": "1e-3",
        },
        {
            "v": 1,
            "id": "c-17",
            "op": "ctrl.register",
            "name": "c1",
            "period_s": "1/1000",
            "min_latency_s": "2e-4",
            "latency": {"fixed_s": "2e-4", "jitter": "uniform:0,3e-4"},
            "observation": {"fields": ["rx_power_dbm", "iq_window"], "window_s": "1e-3"},
        },
        {"v": 1, "id": "c-18", "op": "ctrl.ack", "controller": "c1", "t_b": "1/1000"},
        {"v": 1, "id": "c-19", "op": "ctrl.unregister", "controller": "c1"},
        {"v": 1, "id": "c-20", "op": "rec.marker", "label": "start", "fields": {"k": 1, "s": "x"}},
        {"v": 1, "id": "c-21", "op": "stats.get"},
        {"v": 1, "id": "c-22", "op": "sim.pause"},
        {"v": 1, "id": "c-23", "op": "sim.resume"},
        {"v": 1, "id": "c-24", "op": "sim.shutdown"},
    ]


def _mutations(msg: dict, g: np.random.Generator) -> list[dict]:
    out = []
    for k in list(msg):
        m = dict(msg)
        del m[k]
        out.append(m)
        m = dict(msg)
        m[k] = [None, 1.5, "x", True, {}, [], -1][int(g.integers(0, 7))]
        out.append(m)
    m = dict(msg)
    m["extra"] = 1
    out.append(m)
    return out


def test_io01_schema_validator_agrees_with_reference_jsonschema() -> None:
    """The dependency-free validator gives the reference implementation's verdict on every
    valid request and on single-field mutations of each (the oracle is ``jsonschema``)."""
    jsonschema = pytest.importorskip("jsonschema")
    from sremu.transport.messages import schema

    ref = jsonschema.Draft202012Validator(schema("request.schema.json"))
    mine = validator("request.schema.json")
    g = rng.generator(5, "trial", "wp8/io01/schema")
    corpus: list[object] = []
    for msg in _valid_requests():
        assert ref.is_valid(msg), (msg, [e.message for e in ref.iter_errors(msg)])
        assert mine.is_valid(msg), (msg, mine.errors(msg))
        corpus.extend(_mutations(msg, g))
    for raw in MALFORMED_JSON[:-1]:
        with contextlib.suppress(ValueError, UnicodeDecodeError):
            corpus.append(json.loads(raw))
    n_invalid = 0
    for inst in corpus:
        r = ref.is_valid(inst)
        n_invalid += not r
        assert mine.is_valid(inst) == r, (inst, mine.errors(inst))
    assert n_invalid > 50  # the mutations really exercise rejection paths
    for name in ("request", "reply", "event", "telemetry", "frame"):
        jsonschema.Draft202012Validator.check_schema(schema(f"{name}.schema.json"))
