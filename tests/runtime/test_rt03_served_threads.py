"""Served lockstep (engine thread) with client threads: I2a bitwise equality with the
in-process pull model (RT-03 part: "segmentation by availability"), backpressure, stall
report and stop (DESIGN.md sections 4.2, 4.8)."""

from __future__ import annotations

import threading
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import StreamCmd, StreamSpec, TxMetadata
from sremu.timebase import TimeSpec
from tests.runtime._engine_support import client_run, lab_like_fake, make_engine, waveform

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
N_RX = 9000


def _engine(**kw: object):
    return make_engine(
        lab_like_fake(moving=True),
        backend="numba_cpu",
        noise=True,
        bw=8.0e5,
        cfo={"rx1.rx0": 33.0},
        render_tiles="auto",
        seed=8,
        **kw,
    )


def test_rt03_served_engine_thread_with_client_threads_bitwise() -> None:
    ref = client_run(
        _engine(), "ap.tx0", "rx1.rx0", waveform(N_RX + 200), N_RX, send_chunk=512, read_chunk=512
    )
    eng = _engine(rx_queue_pages=4)
    htx = eng.open_stream(StreamSpec("ap.tx0", "tx", 3.5e9, F))
    hrx = eng.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
    eng.stream_cmd(
        hrx, StreamCmd("num_done", num_samps=N_RX, stream_now=False, time_spec=TimeSpec(0, 0))
    )
    eng.start()
    x = waveform(N_RX + 200)

    def sender() -> None:
        for k in range(0, x.size, 333):
            part = x[k : k + 333]
            eng.submit_tx(htx, part, TxMetadata(k == 0, TimeSpec(0, 0), k == 0, k + 333 >= x.size))

    got: list[np.ndarray] = []

    def reader() -> None:
        total = 0
        while total < N_RX:
            d, md = eng.read_rx(hrx, 777, 1.0)
            got.append(d)
            total += d.size
            if md.end_of_burst:
                break

    ts = [threading.Thread(target=sender), threading.Thread(target=reader)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    eng.stop()
    y = np.concatenate(got)
    assert y.size == N_RX
    assert np.array_equal(y.view(np.uint64), ref.view(np.uint64))


def test_served_engine_stall_report_and_stop() -> None:
    eng = _engine(stall_timeout_s=0.05, unopened_tx="wait")
    hrx = eng.open_stream(StreamSpec("rx1.rx0", "rx", 3.5e9, F))
    eng.stream_cmd(hrx, StreamCmd("start_cont", stream_now=True))
    eng.start()
    d, _md = eng.read_rx(hrx, 100, 0.3)  # nothing can be computed: TX never commits
    assert d.size == 0
    eng.stop()
    assert len(eng.stall_reports) == 1
    rep = eng.stall_reports[0].to_dict()
    assert "ap.tx0" in rep["groups"][0]["waiting_on_tx"]
    eng.stop()  # idempotent
