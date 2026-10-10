"""IO-04: size caps, element caps, no paths, localhost-only binds (DESIGN.md 8.5, 8.10, 10.4).

* A request of exactly 1 MiB (2**20 bytes, D3) is accepted; 1 MiB + 1 byte is refused with
  ``TOO_LARGE`` and counted.
* 65,536 elements per command pass the caps; 65,537 states, phases or indices are refused
  with ``TOO_LARGE`` and counted.
* No request field is a path: the request schema has no path-like property, unknown fields
  are refused, and the only scenario object that names a file (trajectory ``trace``) is
  refused (``SCHEMA_ERROR``).
* Binding anything but a loopback address is refused without ``allow_remote``
  (``--allow-remote``), for the control, PUB, TX and raw REP sockets and on the CLI.
* Commands above ``max_command_rate_hz`` per client are refused and counted.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys

import numpy as np
import pytest

from sremu.contracts import MAX_ELEMENTS_PER_COMMAND, MAX_MESSAGE_BYTES
from sremu.errors import NotPermitted
from sremu.transport.client import ClientSession
from sremu.transport.control_server import (
    ControlServer,
    ServerConfig,
    check_bind_endpoint,
    is_loopback_endpoint,
)
from sremu.transport.messages import schema
from sremu.transport.rawreqrep import RawPeerConfig
from tests.io._wp8_support import EPHEMERAL, REPO, served
from tests.runtime._engine_support import lab_like_fake, make_engine

pytestmark = pytest.mark.io


def _engine():
    return make_engine(lab_like_fake())


def _exact_size(msg: dict, size: int) -> bytes:
    """``msg`` as compact JSON padded with insignificant whitespace to ``size`` bytes."""
    raw = json.dumps(msg, separators=(",", ":")).encode()
    assert len(raw) <= size
    return raw + b" " * (size - len(raw))


def _send(s: ClientSession, raw: bytes) -> dict:
    s.send_raw(raw)
    assert s.sock.poll(30_000)
    return json.loads(s.sock.recv())


def test_io04_message_cap_is_exactly_one_mib() -> None:
    assert MAX_MESSAGE_BYTES == 1 << 20 == 1_048_576
    with served(_engine()) as srv:
        s = ClientSession(str(srv.endpoints["control"]), hello=False)
        try:
            ok = _send(s, _exact_size({"v": 1, "id": "big", "op": "time.now"}, MAX_MESSAGE_BYTES))
            assert ok["ok"] is True and ok["id"] == "big"
            too = _send(
                s, _exact_size({"v": 1, "id": "big2", "op": "time.now"}, MAX_MESSAGE_BYTES + 1)
            )
            assert too["ok"] is False and too["error"]["code"] == "TOO_LARGE"
            assert srv.counters["too_large_messages"] == 1
            assert srv.counters["schema_errors"] == 0
            # far above the cap (8 MiB): still a counted TOO_LARGE reply, no crash
            huge = _send(s, b" " * (8 << 20))
            assert huge["error"]["code"] == "TOO_LARGE"
            assert srv.counters["too_large_messages"] == 2
            assert s.call("hello")["ok"]
        finally:
            s.close()


def _set_config(n: int, form: str) -> dict:
    msg: dict = {
        "v": 1,
        "id": f"e-{form}-{n}",
        "op": "ris.set_config",
        "panel": "ris0",
        "at": "asap",
    }
    if form == "states":
        st = base64.b64encode(np.zeros(n, "<i2").tobytes()).decode()
        msg["config"] = {"states_b64": st, "dtype": "i2"}
    elif form == "phases":
        msg["config"] = {"phases_deg": [0] * n}
    else:
        msg["config"] = {"codebook_index": 0}
        msg["elements"] = {"indices": list(range(n))}
    return msg


@pytest.mark.parametrize("form", ["states", "phases", "indices"])
def test_io04_element_cap_65536(form: str) -> None:
    assert MAX_ELEMENTS_PER_COMMAND == 65_536
    with served(_engine()) as srv:
        s = ClientSession(str(srv.endpoints["control"]))
        try:
            at_cap = s.request(_set_config(MAX_ELEMENTS_PER_COMMAND, form))
            # within the caps: the engine decides (this 120-cell panel cannot take them)
            assert at_cap["ok"] is False and at_cap["error"]["code"] != "TOO_LARGE", at_cap
            over = s.request(_set_config(MAX_ELEMENTS_PER_COMMAND + 1, form))
            assert over["ok"] is False and over["error"]["code"] == "TOO_LARGE", over
            assert srv.counters["too_large_messages"] == 1
        finally:
            s.close()


PATHLIKE = {
    "file",
    "files",
    "path",
    "paths",
    "filename",
    "dir",
    "directory",
    "folder",
    "out",
    "output",
    "run_dir",
    "scenario",
    "url",
    "uri",
}


def _property_names(node: object) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            out |= set(props)
        for v in node.values():
            out |= _property_names(v)
    elif isinstance(node, list):
        for v in node:
            out |= _property_names(v)
    return out


def test_io04_no_path_fields_in_requests() -> None:
    names = _property_names(schema("request.schema.json"))
    assert "trajectory" in names and "stream" in names  # the walk sees the definitions
    assert not names & PATHLIKE, names & PATHLIKE
    with served(_engine()) as srv:
        s = ClientSession(str(srv.endpoints["control"]))
        try:
            trace = s.request(
                {
                    "op": "node.set_trajectory",
                    "node": "rx1",
                    "at": "asap",
                    "trajectory": {"model": "trace", "file": "C:/secret.csv", "data_class": "MEAS"},
                }
            )
            assert trace["error"]["code"] == "SCHEMA_ERROR"
            for op, extra in (
                ("hello", {"path": "/etc/passwd"}),
                ("stats.get", {"file": "x"}),
                ("caps.get", {"scenario": "S.yaml"}),
            ):
                r = s.request({"op": op, **extra})
                assert r["error"]["code"] == "SCHEMA_ERROR", r
            assert srv.counters["schema_errors"] == 4
        finally:
            s.close()


@pytest.mark.parametrize(
    ("endpoint", "loopback"),
    [
        ("tcp://127.0.0.1:5600", True),
        ("tcp://127.0.0.1:0", True),
        ("tcp://127.8.9.10:5600", True),
        ("tcp://localhost:5600", True),
        ("tcp://[::1]:5600", True),
        ("inproc://x", True),
        ("ipc://sremu", True),
        ("tcp://*:5600", False),
        ("tcp://0.0.0.0:5600", False),
        ("tcp://[::]:5600", False),
        ("tcp://192.168.1.10:5600", False),
        ("tcp://eth0:5600", False),
        ("tcp://example.org:5600", False),
        ("udp://127.0.0.1:5600", False),
    ],
)
def test_io04_loopback_classification(endpoint: str, loopback: bool) -> None:
    assert is_loopback_endpoint(endpoint) is loopback
    if loopback:
        check_bind_endpoint(endpoint, allow_remote=False)
    else:
        with pytest.raises(NotPermitted):
            check_bind_endpoint(endpoint, allow_remote=False)
        check_bind_endpoint(endpoint, allow_remote=True)


@pytest.mark.parametrize("which", ["control_endpoint", "pub_endpoint", "tx_endpoint"])
def test_io04_non_localhost_bind_refused_without_allow_remote(which: str) -> None:
    """Refused before any socket exists (the check precedes every bind; binding a wildcard
    address in a test would also trigger host firewall prompts)."""
    for ep in ("tcp://0.0.0.0:0", "tcp://*:0", "tcp://192.168.1.10:0"):
        srv = ControlServer(_engine(), ServerConfig(**{**EPHEMERAL, which: ep}))
        with pytest.raises(NotPermitted, match="only loopback"):
            srv.start()
        assert srv.ctx is None  # nothing was bound


def test_io04_raw_rep_bind_refused_without_allow_remote() -> None:
    peer = RawPeerConfig("p", "srsran_dummy", rx_port="tcp://*:0", rx_stream="rx1.rx0")
    with pytest.raises(NotPermitted):
        ControlServer(_engine(), ServerConfig(**EPHEMERAL, raw_peers=(peer,))).start()


def test_io04_cli_refuses_non_localhost_bind() -> None:
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "sremu.commands.serve",
            "--no-deterministic",
            "examples/bpsk_txrx/bpsk.yaml",
            "--serve",
            "tcp://0.0.0.0:0",
            "--pub",
            "none",
            "--tx",
            "tcp://127.0.0.1:0",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 2
    assert "only loopback" in p.stderr


def test_io04_command_rate_limit_per_client_counted() -> None:
    with served(_engine(), max_command_rate_hz=0.5) as srv:  # bucket of 1 token, 2 s refill
        s = ClientSession(str(srv.endpoints["control"]))
        s2 = ClientSession(str(srv.endpoints["control"]), name="other")
        try:
            marker = {"op": "rec.marker", "label": "m"}
            assert s.request(marker)["ok"]
            r2 = s.request(marker)
            r3 = s.request(marker)
            assert [r2["error"]["code"], r3["error"]["code"]] == ["NOT_PERMITTED"] * 2
            assert "rate" in r2["error"]["msg"]
            assert s2.request(marker)["ok"]  # per client
            assert s.call("stats.get")["transport"]["rate_limited_commands"] == 2
            assert s.call("time.now")["ok"]  # queries are not rate limited
        finally:
            s.close()
            s2.close()
