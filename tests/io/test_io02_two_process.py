"""IO-02: two independent OS processes over native ZMQ (DESIGN.md sections 8.3, 10.4).

A TX generator process and an RX sink process (``tests/io/_wp8_client.py``, each its own
``python`` process using ``sremu.Device.connect``) talk to a served lockstep emulator over
the control socket and the native framed data plane. The RX waveforms they capture must be
**bitwise equal** to an in-process lockstep run with the same TX samples (I2a: client
chunking, process boundaries and wall-clock timing cannot change a bit).

Scene: the E-B-like SIM scene of the WP6 tests (``lab_like_fake``): a moving receiver, a
RIS, thermal noise on, two receivers, 1 MS/s (the v2 S1/S5 regression setting of a TX
generator and an RX sink in separate processes).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.engine.streams import bpsk_v1_waveform
from tests.io._wp8_support import bits, client_cmd, run_clients, served, vcfo_engine
from tests.runtime._engine_support import client_run, lab_like_fake, make_engine, waveform

pytestmark = [pytest.mark.io, pytest.mark.integration]

F = Fraction(10**6)
FC = 3.5e9
N_RX = 20_000
N_TX = 24_000


def _engine(**kw: object):
    return make_engine(lab_like_fake(moving=True), noise=True, backend="numba_cpu", **kw)


def _reference(x: np.ndarray, rx: str) -> np.ndarray:
    """In-process lockstep: timed burst at 0 in 1000-sample sends, 4096-sample reads."""
    return client_run(_engine(), "ap.tx0", rx, x, N_RX, send_chunk=1000, read_chunk=4096)


def test_io02_two_processes_bitwise_equal_to_inprocess(tmp_path: Path) -> None:
    x = waveform(N_TX, seed=11)
    np.save(tmp_path / "x.npy", x)
    eng = _engine(unopened_tx="wait")
    with served(eng, start_paused=True) as srv:
        ep = str(srv.endpoints["control"])
        outs = run_clients(
            [
                client_cmd(
                    "rx",
                    "--sessions",
                    ep,
                    "--stream",
                    "rx1.rx0,rx2.rx0",
                    "--fc",
                    str(FC),
                    "--rate",
                    "1e6",
                    "--n",
                    str(N_RX),
                    "--out",
                    str(tmp_path),
                    "--chunk",
                    "333,4096,1",
                ),
                client_cmd(
                    "tx",
                    "--sessions",
                    ep,
                    "--stream",
                    "ap.tx0",
                    "--fc",
                    str(FC),
                    "--rate",
                    "1e6",
                    "--x",
                    str(tmp_path / "x.npy"),
                    "--chunks",
                    "1000,777,4093",
                ),
            ]
        )
        stats = srv.counters.snapshot()
        assert not eng.nondeterministic_inputs
    assert "rx received" in outs[0] and "tx sent" in outs[1]
    assert stats["bad_packets"] == 0 and stats["seq_errors"] == 0
    assert stats["frames_in"] >= N_TX // 4093
    for rx in ("rx1.rx0", "rx2.rx0"):
        y = np.load(tmp_path / f"0_{rx}.npy")
        assert y.dtype == np.complex64 and y.shape == (N_RX,)
        ref = _reference(x, rx)
        assert np.array_equal(bits(y), bits(ref)), rx
        # the waveform really went through the channel (not zeros, not the raw TX)
        assert float(np.mean(np.abs(y[1000:]) ** 2)) > 0.0
        assert not np.array_equal(bits(y), bits(x[:N_RX]))


def test_io02_bpsk_v1_pair_two_processes(tmp_path: Path) -> None:
    """The ported v1 BPSK pair (2389 samples at 5880 S/s) captured exactly by a separate
    RX process, bitwise equal to the in-process V-CFO-1 engine at f = 0."""
    from tests.runtime.test_vcfo_engine import FC as VFC
    from tests.runtime.test_vcfo_engine import _receive

    x = bpsk_v1_waveform()
    np.save(tmp_path / "x.npy", x)
    with served(vcfo_engine(0.0), start_paused=True) as srv:
        ep = str(srv.endpoints["control"])
        run_clients(
            [
                client_cmd(
                    "rx",
                    "--sessions",
                    ep,
                    "--stream",
                    "rx.rx0",
                    "--fc",
                    repr(VFC),
                    "--rate",
                    "5880",
                    "--n",
                    str(x.size),
                    "--out",
                    str(tmp_path),
                ),
                client_cmd(
                    "tx",
                    "--sessions",
                    ep,
                    "--stream",
                    "tx.tx0",
                    "--fc",
                    repr(VFC),
                    "--rate",
                    "5880",
                    "--x",
                    str(tmp_path / "x.npy"),
                    "--chunks",
                    "12",
                ),
            ]
        )
    y = np.load(tmp_path / "0_rx.rx0.npy")
    assert y.shape == (2389,)
    assert np.array_equal(bits(y), bits(_receive(0.0, Fraction(2, 1000))))
