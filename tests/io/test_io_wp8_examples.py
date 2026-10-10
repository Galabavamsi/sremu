"""The WP8 examples run (examples/controller_remote, examples/gnuradio_zmq).

* ``controller_remote.py``: a remote barrier controller in its own process alternates the
  RIS of a separately served emulator (real scenario physics, noise on); its observations
  are bit-identical to the same controller class in-process, and the two RIS states give
  two distinct power levels.
* ``raw_peer_demo.py``: a stub GNU Radio peer (pinned REQ/REP behaviour) through the raw
  mode; the RIS switch scheduled over the control socket is seen exactly at the recorded
  ``k_applied`` (lateness 0).
* ``gr_flowgraph.py`` needs GNU Radio; without it the script says so (exit code 3).
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from tests.io._wp8_support import REPO

pytestmark = pytest.mark.io

EX = REPO / "examples"


def test_rt08_example_controller_remote_matches_inprocess() -> None:
    p = subprocess.run(
        [sys.executable, str(EX / "controller_remote" / "controller_remote.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert "10 barriers, 10 commands" in p.stdout
    assert "identical observations: True" in p.stdout
    rows = [ln.split() for ln in p.stdout.splitlines() if ln.strip()[:1].isdigit()]
    powers = {int(r[2]): float(r[1]) for r in rows[1:]}  # state -> dBm (t_b = 0 has no data)
    assert abs(powers[0] - powers[1]) > 1.0  # the RIS state changes the received power


def test_io08_example_raw_peer_demo_switch_at_recorded_sample() -> None:
    mod = runpy.run_path(str(EX / "gnuradio_zmq" / "raw_peer_demo.py"))
    res = mod["run"]("0.02", 50_000)
    k = res["k_switch"]
    assert res["status"] == "applied" and res["lateness"] == {"grrx.rx0": 0}
    assert res["k_req"] == {"grrx.rx0": k}
    # the receiver is ~5 m from the RIS: tau << 1 sample at 1 MS/s, so k = to_ticks(0.02 s)
    assert k == int(Fraction("0.02") * 10**6)
    assert res["first_jump"] == k
    y, x = res["y"], res["x"]
    assert y.shape == x.shape == (50_000,)
    assert res["state0"]["gain_db"] != pytest.approx(res["state1"]["gain_db"], abs=0.5)
    assert np.isfinite(res["state0"]["evm_pct"]) and np.isfinite(res["state1"]["evm_pct"])


def test_io08_example_gr_flowgraph_without_gnuradio() -> None:
    try:
        import gnuradio  # noqa: F401
    except ImportError:
        p = subprocess.run(
            [sys.executable, str(EX / "gnuradio_zmq" / "gr_flowgraph.py")],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == 3 and "not installed" in p.stderr
    else:  # pragma: no cover - GNU Radio hosts run E2 instead
        pytest.skip("GNU Radio is installed: run E2 (examples/gnuradio_zmq/README.md)")


def test_io03_example_flip_ris_against_a_served_engine() -> None:
    from tests.io._wp8_support import served
    from tests.runtime._engine_support import lab_like_fake, make_engine

    eng = make_engine(lab_like_fake())
    with served(eng, start_paused=True) as srv:
        mod = runpy.run_path(str(EX / "gnuradio_zmq" / "flip_ris.py"))
        rc = mod["main"](["--endpoint", str(srv.endpoints["control"]), "--at", "1/100"])
        assert rc == 0
        assert not srv.paused
    rec = eng.records[-1]
    assert rec.op == "ris.set_config" and rec.t_req == Fraction(1, 100)
