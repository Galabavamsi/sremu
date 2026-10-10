"""WP6 unit tests: stream runtime (sections 3.3, 4.8, 6.3) and the legacy adapters (5.14)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu import rng
from sremu.contracts import StreamCmd, TxMetadata
from sremu.engine.streams import (
    ArraySource,
    BlockCache,
    HistoryRing,
    QpskPacketSource,
    RxStream,
    StreamConfig,
    ToneSource,
    TxStream,
    bpsk_v1_waveform,
)
from sremu.errors import AsyncCode, ErrorCode, NonPhysicalError
from sremu.impair.noise import noise
from sremu.impair.oscillator import cfo_phasor
from sremu.legacy.truncation import LegacyTruncationAdapter
from sremu.timebase import TimeSpec

pytestmark = pytest.mark.unit

F = Fraction(10**6)


@pytest.mark.parametrize(
    "fn",
    [
        lambda k, n: noise(tuple(rng.derive_key(1, "noise", "a.rx0")), k, n, 0.3),
        lambda k, n: cfo_phasor(-321.75, F, k, n),
    ],
    ids=["noise", "cfo"],
)
def test_block_cache_bitwise_equals_direct(fn: object) -> None:
    """Serving a pure-in-index function from aligned blocks cannot change a bit (I2a)."""
    cache = BlockCache(fn, block=4096)  # type: ignore[arg-type]
    for k0, n in [(0, 10), (4090, 20), (12_000, 9000), (100_000, 1), (5, 0)]:
        a = cache(k0, n)
        b = fn(k0, n)  # type: ignore[operator]
        assert np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8))


def test_tone_source_matches_cfo_phasor_and_is_pure() -> None:
    src = ToneSource(F, 1234.5, 0.5)
    ref = (0.5 * cfo_phasor(1234.5, F, 7000, 3000)).astype(np.complex64)
    assert np.array_equal(src(7000, 3000), ref)
    parts = np.concatenate([src(7000, 1), src(7001, 2999)])
    assert np.array_equal(parts, ref)
    assert np.all(ToneSource(F, 0.0, 2.0)(0, 4) == 2.0)


def test_qpsk_packets_pure_and_framed() -> None:
    src = QpskPacketSource("ap.tx0", 9, sps=4, preamble_syms=8, payload_syms=16, guard=20)
    assert src.period == (8 + 16) * 4 + 20
    whole = src(0, 3 * src.period)
    chunked = np.concatenate([src(k, 37) for k in range(0, 3 * src.period, 37)])[: 3 * src.period]
    assert np.array_equal(whole, chunked)
    body = (8 + 16) * 4
    assert np.all(whole[body : src.period] == 0)  # guard is silent
    assert np.allclose(np.abs(whole[:body]), 1.0)
    assert np.array_equal(whole[: 8 * 4], whole[src.period : src.period + 8 * 4])  # same preamble
    assert not np.array_equal(whole[32:body], whole[src.period + 32 : src.period + body])


def test_bpsk_v1_waveform_2389_samples() -> None:
    x = bpsk_v1_waveform()
    assert x.shape == (2389,)
    # one impulse per 10-sample rectangular window: each sample is its symbol's value
    assert np.all(x[:2000] == np.complex64(1 + 1j))  # 200-symbol header of 1+1j
    assert np.allclose(x[2000:2040], -0.1 - 0.1j)  # bits 0000 -> -0.1(1+j)
    assert np.allclose(x[2040:2100], 0.1 + 0.1j)  # bits 111111 -> +0.1(1+j)
    assert np.allclose(x[2370:2380], -0.1 - 0.1j)  # last symbol
    assert np.all(x[2380:] == 0)  # the 9-sample convolution tail


def test_array_source_repeat() -> None:
    src = ArraySource(np.arange(5, dtype=np.complex64), start=3, period=7)
    out = src(0, 20)
    assert out[:3].tolist() == [0, 0, 0]
    assert out[3:8].real.tolist() == [0, 1, 2, 3, 4]
    assert out[8:10].tolist() == [0, 0]
    assert out[10:15].real.tolist() == [0, 1, 2, 3, 4]


def _tx(**kw: object) -> TxStream:
    cfg = StreamConfig("a.tx0", "tx", 1e9, F, occupied_bw_hz=1.0)
    st = TxStream(cfg, 1 << 12, **kw)  # type: ignore[arg-type]
    st.open()
    return st


def test_tx_commitment_rules() -> None:
    st = _tx()
    assert st.committed() == 0
    x = np.ones(100, np.complex64)
    assert st.submit(x, TxMetadata(True, TimeSpec(0, 0), True, False), 0) == (100, 0)
    assert st.committed() == 100
    st.submit(
        np.zeros(0, np.complex64), TxMetadata(True, TimeSpec.from_ticks(250, F), idle_until=True), 0
    )
    assert st.committed() == 250
    # a timed burst after a gap: the gap is committed silence
    st.submit(x, TxMetadata(True, TimeSpec.from_ticks(400, F), True, True), 0)
    assert st.committed() is None  # EOB: silent until the next timed burst
    assert st.write_through(600) == 0
    ring = st.ring.read(0, 600)
    assert np.allclose(ring[:100], np.sqrt(1e-3))  # front end: full scale 0 dBm
    assert np.all(ring[100:400] == 0)
    assert np.allclose(ring[400:500], np.sqrt(1e-3))
    assert np.all(ring[500:] == 0)
    # a burst in the past is late: TIME_ERROR, counted
    acc, late = st.submit(x[:10], TxMetadata(True, TimeSpec.from_ticks(450, F), True, True), 0)
    assert (acc, late) == (0, 10)
    codes = [e.event_code for e in st.asyncq]
    assert AsyncCode.BURST_ACK in codes
    assert codes[-1] == AsyncCode.TIME_ERROR


def test_tx_write_through_zero_fill_counts_only_uncommitted() -> None:
    st = _tx()
    st.submit(np.ones(50, np.complex64), TxMetadata(True, TimeSpec(0, 0), True, False), 0)
    assert st.write_through(80) == 0  # cannot commit past 50 without zero fill
    assert st.ring.committed_through() == 50
    assert st.write_through(80, zero_fill=True) == 30
    assert st.zero_filled == 30


def test_tx_lo_cfo_applied_at_absolute_index() -> None:
    cfg = StreamConfig("a.tx0", "tx", 1e9, F, occupied_bw_hz=1.0, cfo_hz=1000.0)
    st = TxStream(cfg, 1 << 12)
    st.open()
    st.submit(
        np.ones(64, np.complex64), TxMetadata(True, TimeSpec.from_ticks(100, F), True, True), 0
    )
    st.write_through(200)
    y = st.ring.read(100, 64).astype(np.complex128)
    expect = np.sqrt(1e-3) * cfo_phasor(1000.0, F, 100, 64)
    assert np.allclose(y, expect, rtol=1e-6, atol=0)


def _rx(**kw: object) -> RxStream:
    return RxStream(StreamConfig("b.rx0", "rx", 1e9, F), noise_on=False, **kw)  # type: ignore[arg-type]


def test_rx_num_done_eob_and_reads() -> None:
    st = _rx()
    st.stream_cmd(
        StreamCmd(
            "num_done", num_samps=150, stream_now=False, time_spec=TimeSpec.from_ticks(30, F)
        ),
        0,
    )
    d = (np.arange(100) + 1j).astype(np.complex64)
    st.deliver(0, d)
    st.deliver(100, d)
    a, ma = st.read(64)
    assert a.size == 64
    assert ma.first_sample == 30
    assert ma.start_of_burst
    b, mb = st.read(1000)
    assert b.size == 150 - 64
    assert mb.end_of_burst
    c, mc = st.read(10)
    assert c.size == 0
    assert mc.error_code == ErrorCode.TIMEOUT
    st.stream_cmd(
        StreamCmd("start_cont", stream_now=False, time_spec=TimeSpec.from_ticks(5, F)), 200
    )
    _, md = st.read(1)
    assert md.error_code == ErrorCode.LATE_COMMAND


def test_rx_overflow_marker_carries_post_gap_time() -> None:
    st = _rx(queue_samples=100)
    st.stream_cmd(StreamCmd("start_cont"), 0)
    for k in range(0, 500, 50):
        st.deliver(k, np.ones(50, np.complex64), drop_when_full=True)
    assert st.overflow_pages == 8
    a, _ = st.read(1000)
    assert a.size == 100
    z, m = st.read(10)
    assert z.size == 0
    assert m.error_code == ErrorCode.OVERFLOW
    assert m.first_sample == 500


def test_history_ring_wraps_and_zero_fills_gaps() -> None:
    h = HistoryRing(64)
    h.write(0, np.arange(50, dtype=np.complex64))
    h.write(60, np.arange(100, 130, dtype=np.complex64))  # gap 50..60 reads as 0
    out = h.read(40, 50)
    assert out[:10].real.tolist() == list(range(40, 50))
    assert np.all(out[10:20] == 0)
    assert out[20:50].real.tolist() == list(range(100, 130))
    assert np.all(h.read(0, 20) == 0)  # overwritten (capacity 64)


def test_legacy_truncation_adapter_v1_numbers() -> None:
    with pytest.raises(NonPhysicalError):
        LegacyTruncationAdapter(5880)
    ad = LegacyTruncationAdapter(Fraction(5880), "2e-3", allow_nonphysical=True)
    assert ad.samples_per_tick == 11  # int(11.76)
    assert ad.ticks(2389) == 203
    assert ad.delivered(2389) == 2233
    t = ad.tick_times(23)
    assert t[11] == pytest.approx(0.002)  # tick 1 starts a full tau later
    assert t[10] == pytest.approx(10 / 5880)
    y = ad.deliver(np.ones(2389, np.complex64), 100.0)
    assert y.shape == (2233,)
    assert np.angle(y[11] * np.conj(y[10])) == pytest.approx(2 * np.pi * 100 * (0.002 - 10 / 5880))
    ad2 = LegacyTruncationAdapter(
        Fraction(5880), "2e-3", intra_tick_dt_s=1 / 6000, allow_nonphysical=True
    )
    assert ad2.tick_times(2)[1] == pytest.approx(1 / 6000)
