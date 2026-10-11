"""Stock GNU Radio 3.10 through the raw REQ/REP mode (``examples/gnuradio_zmq``; gated).

GNU Radio is never imported into sremu's interpreter: sremu needs numpy >= 2.2, while Ubuntu
24.04's ``gnuradio`` package is bound to the system numpy 1.26. The tests probe a separate
interpreter, ``$SREMU_GR_PYTHON`` (default ``/usr/bin/python3``), and run GNU Radio there as
a subprocess (``gr_dqpsk_modem.py``: gr-zeromq REP Sink / REQ Source, gr-digital DQPSK
modem). They are skipped, with the reason, on any OS other than Linux and when that
interpreter or its GNU Radio (gr-zeromq, gr-digital) is missing.

* The TX IQ that the analysis regenerates from the modem's symbol file and filter taps
  equals what GNU Radio's own TX chain produces (to float32 rounding).
* One served lockstep run with RIS switches scheduled over the control socket at exact RIS
  times (uniform, focusing, random, focusing; 0.1 s each): GNU Radio receives every sample of
  the run, bit for bit the RX stream the emulator recorded itself; every manifest counter is
  zero; every command is applied with lateness 0, and the maximum-likelihood change point of
  the received IQ lies exactly at each command's ``k_applied``; the stock receiver decodes
  the focusing segment without a symbol error.
* Two independent static runs (fresh processes, other ports) give identical sha256 of the IQ
  GNU Radio received and of its decisions.

Scene: ``gr_ris_ber.yaml`` (SIM geometry). The paper's E-E numbers come from
``experiments/ee_gnuradio.py``, not from this test.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.io._wp8_support import REPO

pytestmark = pytest.mark.integration

EXAMPLE = REPO / "examples" / "gnuradio_zmq"
if str(EXAMPLE) not in sys.path:
    sys.path.insert(0, str(EXAMPLE))


def _gr_python() -> str:
    return os.environ.get("SREMU_GR_PYTHON", "/usr/bin/python3")


def _skip_reason() -> str | None:
    if platform.system() != "Linux":
        return f"GNU Radio peer runs on Linux/WSL2 only (this is {platform.system()})"
    py = _gr_python()
    if not Path(py).is_file():
        return f"no GNU Radio interpreter at {py} (set SREMU_GR_PYTHON)"
    try:
        res = subprocess.run(
            [py, "-I", "-c", "from gnuradio import gr, zeromq, digital; print(gr.version())"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{py} does not run: {exc}"
    if res.returncode != 0:
        return f"{py} has no GNU Radio with gr-zeromq and gr-digital: {res.stderr.strip()[-200:]}"
    return None


@pytest.fixture(scope="module")
def gr_python() -> str:
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    return _gr_python()


_TX_SCRIPT = r"""
import importlib.util, json, sys
import numpy as np
from gnuradio import blocks, digital, filter, gr
from gnuradio.filter import firdes
spec = importlib.util.spec_from_file_location("modem", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sym = m.tx_symbols(11, 2048)
taps = list(firdes.root_raised_cosine(4, 4, 1.0, 0.35, 44))
tb = gr.top_block()
src = blocks.vector_source_b(sym.tolist(), True)
hd = blocks.head(gr.sizeof_gr_complex, 20000)
vs = blocks.vector_sink_c()
tb.connect(src, digital.diff_encoder_bb(4), digital.chunks_to_symbols_bc(m.qpsk_points()),
           filter.interp_fir_filter_ccf(4, taps), hd, vs)
tb.run()
np.array(vs.data(), np.complex64).tofile(sys.argv[2] + ".cf32")
sym.tofile(sys.argv[2] + ".u8")
json.dump({"taps": [float(t) for t in taps]}, open(sys.argv[2] + ".json", "w"))
"""


def test_gnuradio_tx_regeneration_matches_the_flowgraph(gr_python: str, tmp_path: Path) -> None:
    from analyze_ris_ber import regenerate_tx
    from gr_dqpsk_modem import tx_symbols

    script = tmp_path / "tx_chain.py"
    script.write_text(_TX_SCRIPT, encoding="utf-8")
    res = subprocess.run(
        [gr_python, "-I", str(script), str(EXAMPLE / "gr_dqpsk_modem.py"), str(tmp_path / "tx")],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    g = np.fromfile(tmp_path / "tx.cf32", np.complex64)
    sym = np.fromfile(tmp_path / "tx.u8", np.uint8)
    assert np.array_equal(sym, tx_symbols(11, 2048))  # numpy 1.26 and 2.x: same symbols
    taps = np.asarray(json.loads((tmp_path / "tx.json").read_text())["taps"])
    r = regenerate_tx(sym, taps, 4, g.size)
    assert g.size == 20000
    assert float(np.max(np.abs(g - r))) < 1e-5


def test_gnuradio_dqpsk_switching_run_through_raw_mode(gr_python: str, tmp_path: Path) -> None:
    from run_ris_ber import SWITCHING, one_run

    a = one_run(
        tmp_path / "switching",
        kind="switching",
        configs=list(SWITCHING),
        duration_s="0.45",
        segment_s="0.1",
        gr_python=gr_python,
        timeout_s=600.0,
    )
    assert a["ok"], json.dumps(
        {
            k: a.get(k)
            for k in ("complete", "counters_nonzero", "rx_equals_emulator_record", "commands")
        }
    )
    assert a["complete"] and a["rx_samples"] == 450_000
    assert a["rx_equals_emulator_record"] is True
    assert not a["counters_nonzero"]
    assert [c["k_applied"] for c in a["commands"]] == [100_000, 200_000, 300_000]
    assert all(c["status"] == "applied" and c["lateness_samples"] == 0 for c in a["commands"])
    assert [c["offset"] for c in a["change_points"]] == [0, 0, 0]
    segs = a["segments"]
    assert [s["config"] for s in segs] == list(SWITCHING)
    assert segs[1]["ser"] == 0.0 and segs[1]["symbols"] > 10_000  # focusing decodes cleanly
    assert segs[2]["ls_gain_db"] < segs[1]["ls_gain_db"] - 15.0  # random loses the link gain


def test_gnuradio_static_runs_are_bitwise_reproducible(gr_python: str, tmp_path: Path) -> None:
    from run_ris_ber import one_run

    hashes = []
    for i in range(2):
        a = one_run(
            tmp_path / f"focus_{i}",
            kind="static",
            configs=["focus"],
            duration_s="0.1",
            gr_python=gr_python,
            timeout_s=600.0,
        )
        assert a["ok"] and a["segments"][0]["ser"] == 0.0
        hashes.append(a["sha256"])
    assert hashes[0] == hashes[1]
