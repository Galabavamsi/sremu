"""RT-11: a known delay through the engine is recovered within 0.01 samples (DESIGN.md
sections 4.5, 10.3; the renderer-level part is WP5's ``test_render_fd.py``).

LOS-only SIM link (``FakePropagation``) whose length gives ``tau*f_s = 37.4`` samples at
1 MS/s, declared band B = 0.8 f_s (32-tap FD filter), noise off, a TX made of 31 complex
tones on exact DFT bins of a 4096-sample window. The oracle is the ideal delay response:
over the window each tone comes out as ``h0 * exp(-j 2 pi k D / N)``, so after removing
the integer part (37 samples) the residual phase is linear in k with slope
``-2 pi (D - 37)/N``; the fitted D must be 37.4 +- 0.01 samples.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.testing.fake_propagation import FakeNode, FakePropagation
from sremu.testing.scenes import stream
from sremu.units import C0
from tests.oracles.tables import ORACLE
from tests.runtime._engine_support import captured, make_engine

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
N = 4096
BINS = np.arange(-1500, 1501, 100)  # |f| <= 0.37 f_s, inside B = 0.8 f_s
N0 = 1000  # window start, well after the TX start plus the FD transient


class _Tones:
    """Sum of unit complex tones on bins ``k`` of an N-point DFT (pure in the index)."""

    def __call__(self, k0: int, n: int) -> np.ndarray:
        idx = np.arange(k0, k0 + n, dtype=np.int64)
        ph = np.mod(np.outer(idx, BINS), N).astype(np.float64) / N  # exact cycles
        return (np.exp(2j * np.pi * ph).sum(axis=1) / BINS.size).astype(np.complex64)


def test_rt11_known_delay_37p4_samples_through_the_engine(backend_name: str) -> None:
    ref = ORACLE["RT11_delay_samples"]
    d_m = ref.value / float(F) * C0
    nodes = [
        FakeNode("tx", (0.0, 0.0, 0.0), streams=(stream("tx.tx0", "tx", 1.0e9, F),)),
        FakeNode("rx", (d_m, 0.0, 0.0), streams=(stream("rx.rx0", "rx", 1.0e9, F),)),
    ]
    model = FakePropagation(nodes, horizon_s=1)
    eng = make_engine(
        model, backend=backend_name, bw=0.8e6, traffic={"tx.tx0": _Tones()}, always_active=True
    )
    cap = eng.capture_accumulator("rx.rx0")
    eng.run(Fraction(N0 + N) / F)
    dec_fd = eng.manifest_info()["link_decisions"]["tx.tx0->rx.rx0"]["fd_filter"]
    assert "32" in str(dec_fd)  # B = 0.8 f_s: the 32-tap row of the section 4.5 table
    y = captured(cap, N0, N0 + N)
    n = np.arange(N0, N0 + N, dtype=np.int64)
    proj = np.array(
        [np.mean(y * np.exp(-2j * np.pi * np.mod(n * k, N) / N)) for k in BINS],
        dtype=np.complex128,
    )
    resid = proj * np.exp(2j * np.pi * BINS * 37 / N)  # remove the integer part
    phase = np.unwrap(np.angle(resid[np.argsort(BINS)]))
    slope = np.polyfit(np.sort(BINS).astype(np.float64), phase, 1)[0]
    d_hat = 37.0 - slope * N / (2 * math.pi)
    assert abs(d_hat - ref.value) < ref.tol, d_hat  # an integer-only delay gives 37.0
