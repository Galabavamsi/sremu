"""V-CFO-1 and V-CFO-2 (DESIGN.md section 10.10; RQ-35).

Port of the v1 BPSK pair: a 200-symbol header of 1+1j, 38 BPSK symbols at +-0.1(1+j),
10 samples per symbol (2389 samples) at 5880 S/s, through the in-process engine (static SIM
link with a RIS, noise off). The receiver's LO is offset by -f, so the received baseband
rotates at f; the sweep is f = -1400 ... +1400 Hz in 100 Hz steps.

* V-CFO-1: 2389 samples delivered exactly; unwrapped-phase LS and lag-1 estimators on the
  header give |f_hat - f| <= 1e-6 Hz, identical for client epochs of 0.25, 2 and 3.7 ms.
* V-CFO-2: the ``LegacyTruncationAdapter`` (v1 time base: int(f_s tau) samples per 2 ms tick,
  CFO phase advanced by a full tau) reproduces the slope 1.0691 +- 0.0005.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import StreamCmd, StreamSpec, TxMetadata
from sremu.engine.streams import bpsk_v1_waveform
from sremu.errors import NonPhysicalError
from sremu.legacy.truncation import LegacyTruncationAdapter
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import stream
from sremu.timebase import TimeSpec
from tests.oracles.tables import ORACLE, V_CFO_FS, V_CFO_TAU
from tests.runtime._engine_support import make_engine

pytestmark = pytest.mark.runtime

FC = 2.4e9 - 3050.0  # v1 carrier (audit_v1 section 10.1)
SWEEP = np.arange(-1400, 1401, 100, dtype=np.float64)
N0, N1 = 30, 1950  # header samples used by the estimators (as in the v1 audit)


def _model() -> FakePropagation:
    nodes = [
        FakeNode("tx", (5.0, 2.0, 5.0), streams=(stream("tx.tx0", "tx", FC, V_CFO_FS),)),
        FakeNode("rx", (15.0, 8.0, 5.0), streams=(stream("rx.rx0", "rx", FC, V_CFO_FS),)),
    ]
    panel = FakePanel(
        "ris0", (10.0, 0.0, 5.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), 16, 16, 0.0625, 0.0625
    )
    return FakePropagation(nodes, [panel], horizon_s=1)


def _receive(f_hz: float, epoch_s: Fraction, backend: str = "numba_cpu") -> np.ndarray:
    """Send the v1 waveform and receive 2389 samples, sending and reading ``epoch_s`` worth
    of samples per call."""
    x = bpsk_v1_waveform()
    eng = make_engine(_model(), backend=backend, bw=588.0 * 2, cfo={"rx.rx0": -f_hz})
    htx = eng.open_stream(StreamSpec("tx.tx0", "tx", FC, V_CFO_FS))
    hrx = eng.open_stream(StreamSpec("rx.rx0", "rx", FC, V_CFO_FS))
    eng.stream_cmd(
        hrx, StreamCmd("num_done", num_samps=x.size, stream_now=False, time_spec=TimeSpec(0, 0))
    )
    c = max(1, math.ceil(epoch_s * V_CFO_FS))
    got: list[np.ndarray] = []
    sent = 0
    eob = False
    while not eob:
        if sent < x.size:
            part = x[sent : sent + c]
            eng.submit_tx(
                htx,
                part,
                TxMetadata(sent == 0, TimeSpec(0, 0), sent == 0, sent + part.size >= x.size),
            )
            sent += part.size
        d, md = eng.read_rx(hrx, c, 0.0)
        got.append(d)
        eob = md.end_of_burst
    return np.concatenate(got)


def _est_ls(y: np.ndarray, fs: float) -> float:
    seg = y[N0:N1].astype(np.complex128)
    ph = np.unwrap(np.angle(seg))
    n = np.arange(seg.size, dtype=np.float64)
    slope = np.polyfit(n, ph, 1)[0]
    return float(slope * fs / (2 * np.pi))


def _est_lag1(y: np.ndarray, fs: float) -> float:
    seg = y[N0:N1].astype(np.complex128)
    return float(np.angle(np.sum(seg[1:] * np.conj(seg[:-1]))) * fs / (2 * np.pi))


@pytest.fixture(scope="module")
def sweep_2ms() -> dict[float, np.ndarray]:
    return {float(f): _receive(float(f), Fraction(2, 1000)) for f in SWEEP}


def test_vcfo1_delivers_2389_samples_exactly(sweep_2ms: dict[float, np.ndarray]) -> None:
    for y in sweep_2ms.values():
        assert y.shape == (2389,)
        assert y.dtype == np.complex64


def test_vcfo1_estimators_exact_to_1e_6_hz(sweep_2ms: dict[float, np.ndarray]) -> None:
    fs = float(V_CFO_FS)
    for f, y in sweep_2ms.items():
        assert abs(_est_ls(y, fs) - f) <= 1e-6, f
        assert abs(_est_lag1(y, fs) - f) <= 1e-6, f


@pytest.mark.parametrize(
    "epoch", [Fraction(1, 4000), Fraction(37, 10_000)], ids=["0.25ms", "3.7ms"]
)
def test_vcfo1_identical_for_epochs(sweep_2ms: dict[float, np.ndarray], epoch: Fraction) -> None:
    for f in (-1400.0, -300.0, 700.0, 1400.0):
        y = _receive(f, epoch)
        assert np.array_equal(y.view(np.uint64), sweep_2ms[f].view(np.uint64))


def test_vcfo2_legacy_truncation_slope_1p0691(sweep_2ms: dict[float, np.ndarray]) -> None:
    with pytest.raises(NonPhysicalError):
        LegacyTruncationAdapter(V_CFO_FS, V_CFO_TAU)
    ad = LegacyTruncationAdapter(V_CFO_FS, V_CFO_TAU, allow_nonphysical=True)
    assert ad.samples_per_tick == 11
    assert ad.delivered(2389) == 2233  # v1 lost 6.5 % of the waveform (audit_v1 B6)
    y0 = sweep_2ms[0.0]  # the engine's CFO-free output
    est = []
    for f in SWEEP:
        yv1 = ad.deliver(y0, float(f))
        assert yv1.shape == (2233,)
        est.append(_est_ls(yv1, float(V_CFO_FS)))
    slope = float(np.polyfit(SWEEP, np.asarray(est), 1)[0])
    ref = ORACLE["VCFO2_slope"]
    assert slope == pytest.approx(ref.value, abs=ref.tol)
    assert ad.apparent_scale == pytest.approx(11.76 / 11, rel=1e-12)
    # the fixed engine's slope is 1 (V-CFO-1 above): the teeth fail the exactness check
    assert abs(slope - 1.0) > 0.05
