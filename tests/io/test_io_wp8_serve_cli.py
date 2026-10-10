"""The served-run command (``python -m sremu.commands.serve``; ``sremu run --serve`` after
CCR WP8-1) with the real scenario physics (DESIGN.md sections 2.5, 3.6, 8.3-8.5).

The ported v1 BPSK pair scenario (``examples/bpsk_txrx/bpsk.yaml``: WP2 propagation model,
thermal noise, numba backend) is served by a separate OS process on ephemeral loopback
ports; a client in this process sends the 2389-sample waveform and receives it through
``sremu.Device.connect``. The received samples are bitwise equal to the same API calls on
``Device.inproc`` (lockstep, I2a); the server records a run directory (``--out``) and exits
by itself once the run has ended and the RX egress has been drained (``--exit-when-done``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TimeSpec, TxMetadata
from sremu.engine.streams import bpsk_v1_waveform
from tests.io._wp8_support import REPO, bits

pytestmark = pytest.mark.io

SCENARIO = REPO / "examples" / "bpsk_txrx" / "bpsk.yaml"
FC = 2399996950.0


def _session(dev: Device, x: np.ndarray, resume: bool) -> np.ndarray:
    rx = dev.get_rx_stream(StreamArgs(stream="node_2.rx0", fc_hz=FC, rate_sps="5880"))
    rx.issue_stream_cmd(
        StreamCmd(mode="num_done", num_samps=x.size, stream_now=False, time_spec=TimeSpec(0, 0))
    )
    tx = dev.get_tx_stream(StreamArgs(stream="node_1.tx0", fc_hz=FC, rate_sps="5880"))
    if resume:
        dev.engine.resume()  # type: ignore[attr-defined]
    assert (
        tx.send(
            x,
            TxMetadata(
                has_time_spec=True, time_spec=TimeSpec(0, 0), start_of_burst=True, end_of_burst=True
            ),
        )
        == x.size
    )
    out = np.zeros(x.size, np.complex64)
    md = RxMetadata()
    got = 0
    while got < x.size:
        k = rx.recv(out[got:], md, timeout=30.0)
        assert k > 0, md.error_code
        got += k
    assert md.end_of_burst
    return out


def test_io02_serve_cli_bpsk_bitwise_equal_inprocess_and_records(tmp_path: Path) -> None:
    x = bpsk_v1_waveform()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sremu.commands.serve",
            "--no-deterministic",
            str(SCENARIO),
            "--serve",
            "tcp://127.0.0.1:0",
            "--pub",
            "tcp://127.0.0.1:0",
            "--tx",
            "tcp://127.0.0.1:0",
            "--out",
            str(tmp_path / "runs"),
            "--name",
            "served",
            "--exit-when-done",
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline() if proc.stdout else ""
        assert line.startswith("SREMU-SERVE "), (line, proc.stderr.read() if proc.stderr else "")
        info = json.loads(line.split(" ", 1)[1])
        assert info["paused"] is True
        with Device.connect(info["endpoints"]["control"]) as dev:
            served_y = _session(dev, x, resume=True)
        rc = proc.wait(timeout=120)
        err = proc.stderr.read() if proc.stderr else ""
    finally:
        proc.kill()
    assert rc == 0, err
    with Device.inproc(SCENARIO) as dev:
        ref = _session(dev, x, resume=False)
    assert np.array_equal(bits(served_y), bits(ref))
    assert float(np.mean(np.abs(ref[:2000]) ** 2)) > 0.0
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    rd = run_dirs[0]
    man = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    assert man["mode"] == "lockstep"
    assert any("served run" in a for a in man["assumptions"])
    assert (rd / "events.jsonl").exists() and (rd / "scenario.resolved.yaml").exists()
    counters = man["counters"]
    assert counters["transport.bad_packets"] == 0
    assert counters["transport.frames_in"] >= 1  # the waveform arrived over the data plane


def test_io04_serve_cli_usage_errors() -> None:
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "sremu.commands.serve",
            "--no-deterministic",
            "nope.yaml",
            "--serve",
            "tcp://127.0.0.1:0",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 2
    assert "INVALID" in p.stderr
    from sremu.commands.serve import parse_raw_peer

    c = parse_raw_peer(
        "name=enb,format=gnuradio_count,rx_port=tcp://127.0.0.1:0,rx_stream=a.rx0,reply_items=64"
    )
    assert c.request_format == "gnuradio_count" and c.reply_items == 64
    import typer

    with pytest.raises(typer.BadParameter):
        parse_raw_peer("name=enb,format")
