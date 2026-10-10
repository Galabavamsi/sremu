"""IO-07: the ported v1 BPSK scripts run through Send_to_sim/Receive_from_sim (DESIGN 8.2).

Acceptance: 2389 samples delivered exactly. Oracles: the v1 waveform rebuilt independently
(``np.repeat`` of the 238 symbols by 10 plus the 9-sample convolution tail of zeros, stored
through float32 as v1 did), the exact v1 rate (5880 S/s via R5) and the identity channel of
``FakeEngine`` (RX = TX bit for bit).
"""

from __future__ import annotations

import importlib.util
import runpy
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu import compat
from sremu.api import Device
from sremu.compat import Receive_from_sim, Send_to_sim, use_device
from sremu.errors import ConfigurationError, LateCommandError, SremuTimeout
from sremu.record.sigmf import read_recording, validate_collection
from sremu.timebase import Rate, TimeSpec
from tests.io import _wp7_support as sup

pytestmark = pytest.mark.io

EX = Path(__file__).resolve().parents[2] / "examples" / "bpsk_txrx"
FC_V1 = 2399996950.0


def _v1_waveform_oracle() -> np.ndarray:
    header = np.full(200, 1 + 1j)
    bits = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] + [0] * 28
    sym = np.where(np.asarray(bits) == 1, 0.1 + 0.1j, -0.1 - 0.1j)
    x = np.concatenate([np.repeat(np.concatenate([header, sym]), 10), np.zeros(9)])
    re, im = x.real.astype(np.float32), x.imag.astype(np.float32)
    return (re + 1j * im).astype(np.complex64)


@pytest.fixture
def tx_mod():
    return runpy.run_path(str(EX / "usrp_tx.py"))  # not __main__: definitions only


def test_io07_v1_waveform_and_rate_are_preserved(tx_mod):
    w = tx_mod["make_waveform"]()
    assert w.shape == (1, 2389)
    assert w.dtype == np.complex64
    np.testing.assert_array_equal(w[0].view(np.uint32), _v1_waveform_oracle().view(np.uint32))
    assert tx_mod["sampling_rate"]() == 5880.0
    assert Rate.from_api(tx_mod["sampling_rate"]()) == Fraction(5880)
    assert tx_mod["CENTER_FREQ"] == FC_V1


def test_io07_ported_scripts_deliver_2389_samples_exactly(capsys):
    eng = sup.make_engine()
    dev = Device.from_engine(eng)
    with dev, use_device(dev):
        tx = runpy.run_path(str(EX / "usrp_tx.py"), run_name="__main__")
        rx = runpy.run_path(str(EX / "usrp_rx.py"), run_name="__main__")
    y = rx["received"]
    assert tx["sent"] == 2389
    assert isinstance(y, np.ndarray)
    assert y.dtype == np.complex64
    assert y.shape == (2389,)
    np.testing.assert_array_equal(y.view(np.uint32), _v1_waveform_oracle().view(np.uint32))
    # the streams carry the v1 parameters exactly
    specs = {h.spec.stream: h.spec for h in (s.handle for s in dev._tx.values())}
    specs.update({h.spec.stream: h.spec for h in (s.handle for s in dev._rx.values())})
    assert specs["node_1.tx0"].rate_sps == Fraction(5880) == specs["node_2.rx0"].rate_sps
    assert specs["node_1.tx0"].fc_hz == FC_V1 == specs["node_2.rx0"].fc_hz
    out = capsys.readouterr().out
    assert "sent 2389 samples from node_1.tx0" in out
    assert "received 2389 samples at node_2.rx0" in out


def test_io07_run_pair_fake_engine_with_recording(tmp_path, capsys):
    run_pair = runpy.run_path(str(EX / "run_pair.py"))
    rec = tmp_path / "bpsk"
    assert run_pair["main"](["--engine", "fake", "--record", str(rec)]) == 0
    out = capsys.readouterr().out
    assert "engine fake: sent 2389, received 2389 samples" in out
    assert "received == sent bit for bit: True" in out
    assert validate_collection(rec) == []
    rx = read_recording(rec / "iq" / "node_2.rx0.sigmf-meta")
    tx = read_recording(rec / "iq" / "node_1.tx0.sigmf-meta")
    assert rx.data.size == tx.data.size == 2389
    assert rx.rate_sps == Fraction(5880)
    np.testing.assert_array_equal(rx.data.view(np.uint32), _v1_waveform_oracle().view(np.uint32))


def test_io07_v1_names_are_deprecated_aliases_with_v1_argument_order(tx_mod):
    w = tx_mod["make_waveform"]()
    dev = Device.from_engine(sup.make_engine())
    with use_device(dev):
        with pytest.warns(DeprecationWarning, match="Send_to_sim"):
            n = compat.send_to_simulator(w, FC_V1, 5880.0, "node_1")
        with pytest.warns(DeprecationWarning, match="Receive_from_sim"):
            y = compat.recieve_from_simulator(2389, FC_V1, 5880.0, "node_2")
    assert n == 2389
    np.testing.assert_array_equal(y.view(np.uint32), w[0].view(np.uint32))


def test_io07_ants_names_raise_no_warning(tx_mod):
    dev = Device.from_engine(sup.make_engine())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Send_to_sim("node_1", tx_mod["make_waveform"](), FC_V1, 5880.0, device=dev)
        assert Receive_from_sim("node_2", 2389, FC_V1, 5880.0, device=dev).size == 2389


# --------------------------------------------------------------------------- compat semantics


def test_compat_timed_start_uses_r5_and_r3():
    dev = Device.from_engine(sup.make_engine())
    x = (np.arange(1, 101) * (1 + 0.5j)).astype(np.complex64)
    Send_to_sim("ap", x, 3.5e9, 1e6, start_time=0.5e-3, device=dev)  # sample 500, not 501
    y = Receive_from_sim("mu", 600, 3.5e9, 1e6, device=dev)
    assert not y[:500].any()
    np.testing.assert_array_equal(y[500:], x)
    y2 = Receive_from_sim("ue", 100, 3.5e9, "1e6", start_time=TimeSpec(0, 5e-4), device=dev)
    np.testing.assert_array_equal(y2, x)


def test_compat_gain_is_digital_and_zero_gain_is_bitwise_identity():
    dev = Device.from_engine(sup.make_engine())
    x = (np.linspace(-1, 1, 64) + 0.25j).astype(np.complex64)
    Send_to_sim("ap", x, 3.5e9, 1e6, gain=20.0, device=dev)  # factor 10**(20/20) = 10
    y = Receive_from_sim("mu", 64, 3.5e9, 1e6, device=dev)
    np.testing.assert_array_equal(y, (x * np.float32(10.0)).astype(np.complex64))
    y2 = Receive_from_sim("ue", 64, 3.5e9, 1e6, gain=-20.0, device=dev)  # RX gain 0.1
    np.testing.assert_array_equal(y2, (y * np.float32(0.1)).astype(np.complex64))


def test_compat_errors(monkeypatch):
    monkeypatch.delenv(compat.CONNECT_ENV, raising=False)
    compat.clear_device()
    with pytest.raises(ConfigurationError, match="no sremu device"):
        compat.get_device()
    dev = Device.from_engine(sup.make_engine(), poll_s=0.01)
    dev.get_tx_stream(compat.StreamArgs("ap.tx0", 3.5e9, "1e6"))  # linked, never sends
    with pytest.raises(SremuTimeout, match="received 0 of 10"):
        Receive_from_sim("mu", 10, 3.5e9, 1e6, device=dev, timeout=0.0)
    with pytest.raises(ConfigurationError, match="already open"):
        Send_to_sim("ap", np.ones(4, np.complex64), 3.5e9, 2e6, device=dev)
    dev2 = Device.from_engine(sup.make_engine())
    Send_to_sim("ap", np.ones(100, np.complex64), 3.5e9, 1e6, device=dev2)
    Receive_from_sim("mu", 100, 3.5e9, 1e6, device=dev2)
    with pytest.raises(LateCommandError):
        Receive_from_sim("mu", 10, 3.5e9, 1e6, start_time=1e-5, device=dev2)
    with pytest.raises(ConfigurationError, match="nchan"):
        Send_to_sim("bs", np.ones((2, 4), np.complex64), 3.5e9, 1e6, device=dev2)


def test_compat_use_device_nests_and_restores():
    a = Device.from_engine(sup.make_engine())
    b = Device.from_engine(sup.make_engine())
    compat.clear_device()
    with use_device(a):
        assert compat.get_device() is a
        with use_device(b):
            assert compat.get_device() is b
        assert compat.get_device() is a
    compat.set_device(b)
    assert compat.get_device() is b
    compat.clear_device()


def test_io07_example_scenario_validates_with_the_script_constants(tx_mod):
    from sremu.scenario.load import load_scenario

    res = load_scenario(EX / "bpsk.yaml")
    streams = res.scenario.streams()
    tx, rx = streams["node_1.tx0"][1], streams["node_2.rx0"][1]
    assert tx.rate_sps == rx.rate_sps == Fraction(5880)
    assert tx.fc_hz == rx.fc_hz == tx_mod["CENTER_FREQ"]


@pytest.mark.skipif(
    importlib.util.find_spec("sremu.engine.emulator") is None,
    reason="WP6 engine (sremu.engine.emulator) not available yet",
)
def test_io07_ported_scripts_through_the_wp6_engine(capsys):
    """Through the real engine (bpsk.yaml: free space, 3 m, LOS) exactly 2389 samples
    arrive and the received waveform is the TX waveform times the Friis LOS coefficient
    ``h = lambda/(4 pi d) exp(-j 2 pi f_c d/c)`` (section 3.1). Tolerance: the RX noise
    floor (-136.3 dBm at 5880 S/s, F = 0 dB) is 89.7 dB below the received power, so the
    least-squares estimate of h is good to about 1e-6; 1e-4 leaves room for the FD filter
    at a 10 ns (5.9e-5 sample) delay."""
    from sremu.units import C0

    run_pair = runpy.run_path(str(EX / "run_pair.py"))
    res = run_pair["run"]("inproc")
    assert res["engine"] == "inproc"
    y = res["received"]
    assert y.shape == (2389,)
    assert y.dtype == np.complex64
    x = _v1_waveform_oracle().astype(np.complex128)
    h_est = np.vdot(x, y.astype(np.complex128)) / np.vdot(x, x)
    d = 3.0
    h = C0 / FC_V1 / (4 * np.pi * d) * np.exp(-2j * np.pi * FC_V1 * d / C0)
    assert abs(h_est / h - 1) < 1e-4
    assert "received 2389 samples at node_2.rx0" in capsys.readouterr().out
