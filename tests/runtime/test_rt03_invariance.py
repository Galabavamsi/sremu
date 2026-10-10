"""RT-03 (I2a): RX output does not depend on client chunking, epochs, barrier splits or the
numba thread count (deterministic mode). Bitwise. DESIGN.md sections 4.2, 4.3, 10.3.

The scene moves (snapshot grid, chord delays, render tiles), carries two timed RIS
commands (one with a linear transition), thermal noise and RX/TX CFO, so every page is cut
by snapshot boundaries, switch samples and transition terms.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import CommandTime, TransitionSpec
from tests.runtime._engine_support import (
    ScriptedController,
    client_run,
    config_request,
    lab_like_fake,
    make_engine,
    waveform,
)

pytestmark = pytest.mark.runtime

N_RX = 12_000  # 12 ms at 1 MS/s
N_TX = 12_200
CFO = {"ap.tx0": 137.0, "rx1.rx0": -421.5}


def _engine(backend: str, **kw: object):
    eng = make_engine(
        lab_like_fake(moving=True),
        backend=backend,
        noise=True,
        bw=8.0e5,
        cfo=CFO,
        render_tiles="auto",
        seed=11,
        **kw,
    )
    n = eng.commands.specs["ris0"].n_controls
    s1 = np.zeros(n, np.int16)
    s1[::3] = 1
    eng.schedule(config_request("c1", "ris0", s1, CommandTime.at(Fraction(3, 1000))))
    eng.schedule(
        config_request(
            "c2",
            "ris0",
            np.ones(n, np.int16),
            CommandTime.at(Fraction(7777, 1_000_000)),
            transition=TransitionSpec("linear", Fraction(20, 1_000_000)),
        )
    )
    return eng


def _run(backend: str, send: object, read: object, **kw: object) -> np.ndarray:
    eng = _engine(backend, **kw)
    return client_run(
        eng, "ap.tx0", "rx1.rx0", waveform(N_TX), N_RX, send_chunk=send, read_chunk=read
    )  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    """Reference outputs per backend: one-page chunks."""
    return {b: _run(b, 512, 512) for b in ("numpy_ref", "numba_cpu")}


@pytest.mark.parametrize(
    "chunks",
    [(1, 1), (7, 7), (1000, 1000), (512, 512), ([3, 999, 250, 17, 4096], [700, 1, 33, 2500])],
    ids=["chunk1", "chunk7", "chunk1000", "page", "random"],
)
def test_rt03_client_chunks_bitwise(
    reference: dict[str, np.ndarray], chunks: tuple, backend_name: str
) -> None:
    send, read = chunks
    y = _run(backend_name, send, read)
    assert y.dtype == np.complex64
    assert np.array_equal(y.view(np.uint64), reference[backend_name].view(np.uint64))


@pytest.mark.parametrize("epoch_s", ["0.25e-3", "1e-3", "3.7e-3"])
def test_rt03_epochs_bitwise(
    reference: dict[str, np.ndarray], epoch_s: str, backend_name: str
) -> None:
    """Client epochs of 0.25, 1 and 3.7 ms (send and read that much per call)."""
    n = int(Fraction(epoch_s) * 10**6)
    y = _run(backend_name, n, n)
    assert np.array_equal(y.view(np.uint64), reference[backend_name].view(np.uint64))


def test_rt03_injected_barrier_splits_bitwise(
    reference: dict[str, np.ndarray], backend_name: str
) -> None:
    """A registered controller that never acts splits pages at its barrier limits
    (period 0.37 ms, L_min 0.123 ms); the output is unchanged."""
    eng = _engine(backend_name)
    ctrl = ScriptedController("probe", "37/100000", "123/1000000", lambda obs, m: [])
    eng.register_controller(ctrl)
    y = client_run(eng, "ap.tx0", "rx1.rx0", waveform(N_TX), N_RX, send_chunk=512, read_chunk=512)
    assert len(ctrl.observations) > 20
    assert np.array_equal(y.view(np.uint64), reference[backend_name].view(np.uint64))


@pytest.mark.parametrize("threads", [1, 4, 16])
def test_rt03_numba_threads_bitwise_deterministic_mode(
    reference: dict[str, np.ndarray], threads: int
) -> None:
    """1/4/16 numba threads (capped at the machine's NUMBA_NUM_THREADS) in deterministic
    mode (SVML off, conftest): bitwise identical."""
    import numba

    n = min(threads, numba.config.NUMBA_NUM_THREADS)
    old = numba.get_num_threads()
    numba.set_num_threads(n)
    try:
        y = _run("numba_cpu", 512, 512)
    finally:
        numba.set_num_threads(old)
    assert np.array_equal(y.view(np.uint64), reference["numba_cpu"].view(np.uint64))


def test_rt03_reference_is_not_trivial(reference: dict[str, np.ndarray]) -> None:
    """Teeth for the bitwise checks: the reference carries signal, the switch and noise
    (changing the second command's time changes the output)."""
    y = reference["numpy_ref"]
    assert np.all(np.isfinite(y))
    assert float(np.mean(np.abs(y.astype(np.complex128)) ** 2)) > 0
    eng = make_engine(
        lab_like_fake(moving=True),
        backend="numpy_ref",
        noise=True,
        bw=8.0e5,
        cfo=CFO,
        render_tiles="auto",
        seed=11,
    )
    n = eng.commands.specs["ris0"].n_controls
    eng.schedule(
        config_request("c1", "ris0", np.ones(n, np.int16), CommandTime.at(Fraction(3, 1000)))
    )
    z = client_run(eng, "ap.tx0", "rx1.rx0", waveform(N_TX), N_RX, send_chunk=512, read_chunk=512)
    assert not np.array_equal(z, y)
    assert np.array_equal(z[:2900], y[:2900])  # identical before the first switch
