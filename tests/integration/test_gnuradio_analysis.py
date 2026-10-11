"""Analysis helpers of the GNU Radio demonstration (``examples/gnuradio_zmq``), without GNU
Radio: synthetic signals with known answers. Runs everywhere (CI tier ``unit``).

* the TX regeneration is differential QPSK, zero-stuffed and filtered from a zero state;
* the maximum-likelihood change point finds a gain step at its exact sample at moderate SNR;
* decision alignment, symbol errors and the uncoded 200-bit block rate;
* the DQPSK theory and the implied implementation loss invert each other;
* the RIS configurations: focusing maximizes the isotropic array factor, the inverted
  fractions and the random configuration lose gain monotonically, and the predictions of
  the GNU Radio feasibility investigation are reproduced (same seeds, same geometry).

Scenes are SIM geometry; nothing here is a physics result.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pytest

from tests.io._wp8_support import REPO

pytestmark = pytest.mark.unit

EXAMPLE = REPO / "examples" / "gnuradio_zmq"
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))

import analyze_ris_ber as an  # noqa: E402
from gr_dqpsk_modem import qpsk_points, tx_symbols  # noqa: E402
from ris_configs import STATIC, Configs, load_scene  # noqa: E402


def test_tx_symbols_are_pinned_to_the_raw_bit_stream() -> None:
    s = tx_symbols(11, 64)
    w = np.random.PCG64(11).random_raw(2)
    assert s.dtype == np.uint8 and s.size == 64
    assert [int(v) for v in s[:4]] == [
        int((w[0] >> np.uint64(2 * i)) & np.uint64(3)) for i in range(4)
    ]
    assert np.allclose(qpsk_points()[1], np.exp(1j * 3 * np.pi / 4))


def test_regenerate_tx_is_differential_qpsk_through_the_filter() -> None:
    sym = np.array([1, 3, 0, 2, 2, 1], np.uint8)
    taps = np.array([0.5, 1.0, 0.25])
    x = an.regenerate_tx(sym, taps, 2, 20)
    d = np.cumsum(np.tile(sym, 3).astype(int)) % 4
    pts = np.exp(1j * (np.pi / 4 + np.pi / 2 * d))
    up = np.zeros(pts.size * 2, complex)
    up[::2] = pts
    assert np.allclose(x, np.convolve(up, taps)[:20])


def test_ml_change_point_lands_on_the_switch_sample() -> None:
    g = np.random.default_rng(5)
    n, k = 40_000, 21_337
    x = (g.standard_normal(n) + 1j * g.standard_normal(n)) / np.sqrt(2)
    h = np.where(np.arange(n) < k, 0.5 * np.exp(0.3j), 0.2 * np.exp(-1.1j))
    noise = 0.1 * (g.standard_normal(n) + 1j * g.standard_normal(n)) / np.sqrt(2)
    y = h * x + noise  # 6 dB SNR after the step
    assert an.ml_change_point(x, y, 0, n) == k
    assert an.ml_change_point(x, y, 5_000, 35_000) == k


def test_alignment_errors_and_uncoded_blocks() -> None:
    sym = np.random.default_rng(1).integers(0, 4, 8192).astype(np.uint8)
    lag = 1234
    dec = sym[(np.arange(6000) + lag) % sym.size].copy()
    dec[[100, 101, 4050]] ^= 1
    got, peak = an.align(dec, sym, [0, 3000])
    assert got == lag and peak > 0.99
    err = an.symbol_errors(dec, sym, got, 0, 6000)
    assert int(err.sum()) == 3
    b, nb = an.bler(err)
    assert nb == 60 and b == pytest.approx(2 / 60)


def test_dqpsk_theory_and_implied_loss_invert() -> None:
    assert an.ser_dqpsk_theory(30.0) < 1e-12
    assert 0.0 < an.ser_dqpsk_theory(8.0) < an.ser_dqpsk_theory(5.0) < 0.75
    ser = an.ser_dqpsk_theory(10.0)
    assert an.implied_loss_db(ser, 10.5) == pytest.approx(0.5, abs=1e-6)
    assert an.implied_loss_db(0.0, 10.0) is None


def test_ris_configurations_of_the_scene() -> None:
    c = Configs(load_scene(EXAMPLE / "gr_ris_ber.yaml"))
    assert c.n_cells == 256
    af = {name: c.array_factor_db(name) for name in STATIC}
    assert af["focus"] == 0.0 and all(v <= 0.0 for v in af.values())
    q = [af[f"focusq:{x:.2f}"] for x in (0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)]
    assert all(a > b for a, b in itertools.pairwise(q))
    # the GNU Radio feasibility investigation (sdr_integration_feasibility.md, 3.6), same seeds
    for name, v in (
        ("uniform", -8.32),
        ("random", -23.71),
        ("focusq:0.35", -10.95),
        ("focusq:0.10", -1.98),
    ):
        assert af[name] == pytest.approx(v, abs=0.006)
    assert int((c.states("focusq:0.35") != c.focus).sum()) == round(0.35 * 256)
    assert set(np.unique(c.states("random"))) == {0, 1}
