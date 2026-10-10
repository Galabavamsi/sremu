"""IO-03: a socket command lands at the exact sample; lateness 0 in lockstep (DESIGN.md
sections 4.6, 8.5, 10.4).

Scene: the E-B-like SIM scene with ``rx2`` 2 km away along the panel normal (``far_m``),
so the two receivers see the switch at different samples (I3:
``k_r = to_ticks(t_sw + tau_RIS->r(t_sw), f_r)``; about 6.7 samples apart at 1 MS/s). The TX
is an in-process tone, noise is off. Commands arrive over the control socket:

* A: ``at {"t": "5/1000"}`` before the run starts (paused);
  one render tile, so the whole panel switches at the panel-centre sample;
* B: ``at {"sample": {"stream": "rx2.rx0", "index": 61234}}``;
* C: ``at {"t": "3/20"}`` sent while the run is going, with the frontier still before it.

Oracles: ``k_req`` equals the closed form above (FakePropagation's analytic RIS delay);
every record ends ``applied`` with ``k_applied == k_req`` and lateness 0 on every RX; the
served RX equals, bitwise, an in-process run with the same commands; and that run equals a
command-free run on ``[0, k_A)`` and differs from it at ``k_A`` (the switch is exactly there).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import CommandTime, StreamCmd, StreamSpec
from sremu.engine.streams import ToneSource
from sremu.timebase import TimeSpec, from_ticks, to_ticks
from sremu.transport.client import RemoteEngine
from sremu.transport.messages import b64_of
from tests.io._wp8_support import bits, served
from tests.runtime._engine_support import config_request, lab_like_fake, make_engine

pytestmark = pytest.mark.io

F = Fraction(10**6)
N = 200_000
RXS = ("rx1.rx0", "rx2.rx0")
T_A = Fraction(5, 1000)
K_B = 61_234
T_C = Fraction(3, 20)


def _engine(**kw: object):
    return make_engine(
        lab_like_fake(far_m=2000.0, tiles=(1, 1)),
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        **kw,
    )


def _n(eng) -> int:
    return eng.commands.specs["ris0"].n_controls


def _read_all(cli: RemoteEngine, hs: dict, n: int, first: int = 0) -> dict[str, np.ndarray]:
    out = {r: [] for r in RXS}
    got = dict.fromkeys(RXS, 0)
    while min(got.values()) < n:
        for r in RXS:
            if got[r] >= n:
                continue
            y, md = cli.read_rx(hs[r], min(4096, n - got[r]), 10.0)
            assert md.error_code.name in ("NONE", "TIMEOUT"), md.error_code
            if y.size:
                assert md.first_sample == first + got[r]
                out[r].append(y)
                got[r] += y.size
    return {r: np.concatenate(v) for r, v in out.items()}


def _inprocess(commands: bool) -> dict[str, np.ndarray]:
    eng = _engine()
    n = _n(eng)
    hs = {r: eng.open_stream(StreamSpec(r, "rx", 3.5e9, F)) for r in RXS}
    for h in hs.values():
        eng.stream_cmd(h, StreamCmd("num_done", N, False, TimeSpec(0, 0)))
    if commands:
        for cid, at, s in (
            ("A", CommandTime.at(T_A), 1),
            ("B", CommandTime.at_sample("rx2.rx0", K_B), 0),
            ("C", CommandTime.at(T_C), 1),
        ):
            assert eng.schedule(config_request(cid, "ris0", np.full(n, s, np.int16), at)).ok
    out = {}
    for r, h in hs.items():
        parts = []
        while sum(p.size for p in parts) < N:
            y, _ = eng.read_rx(h, N, 0.0)
            parts.append(y)
        out[r] = np.concatenate(parts)
    return out


def test_io03_socket_command_lands_at_exact_sample_lateness_zero() -> None:
    eng = _engine(unopened_tx="wait")
    n = _n(eng)
    model = eng.model
    with served(eng, start_paused=True) as srv:
        cli = RemoteEngine(str(srv.endpoints["control"]), name="io03")
        try:
            hs = {r: cli.open_stream(StreamSpec(r, "rx", 3.5e9, F)) for r in RXS}
            for h in hs.values():
                cli.stream_cmd(h, StreamCmd("num_done", N, False, TimeSpec(0, 0)))

            def send(cid: str, at: object, s: int) -> dict:
                return cli.session.request(
                    {
                        "id": cid,
                        "op": "ris.set_config",
                        "panel": "ris0",
                        "at": at,
                        "config": {"states_b64": b64_of(np.full(n, s), "<i2"), "dtype": "i2"},
                    }
                )

            ra = send("A", {"t": "5/1000"}, 1)
            rb = send("B", {"sample": {"stream": "rx2.rx0", "index": K_B}}, 0)
            assert ra["ok"] and ra["status"] == "scheduled", ra
            assert rb["ok"] and rb["status"] == "scheduled", rb
            # I3 closed form with the model's analytic RIS delay (static scene; the float
            # delay enters exactly)
            for r in RXS:
                tau = model.ris_delay_s("ris0", r, T_A)
                assert ra["k_req"][r] == to_ticks(T_A + Fraction(tau), F)
            assert rb["k_req"]["rx2.rx0"] == K_B
            assert ra["k_req"]["rx2.rx0"] - ra["k_req"]["rx1.rx0"] in (6, 7)
            cli.resume()
            first = _read_all(cli, hs, 1000)
            fr = cli.frontier()
            assert max(fr.values()) < to_ticks(T_C, F)  # C is sent ahead of the frontier
            rc = send("C", {"t": "3/20"}, 1)
            assert rc["ok"] and rc["status"] == "scheduled", rc
            rest = _read_all(cli, hs, N - 1000, first=1000)
            served_y = {r: np.concatenate([first[r], rest[r]]) for r in RXS}
        finally:
            cli.close()
    recs = {r.cmd_id: r for r in eng.records}
    assert set(recs) == {"A", "B", "C"}
    for r in recs.values():
        assert r.status == "applied", r
        assert r.k_applied == r.k_req
        assert all(v == 0 for v in r.lateness_samples.values()), r
    assert recs["B"].k_applied["rx2.rx0"] == K_B
    assert not eng.nondeterministic_inputs
    ref = _inprocess(commands=True)
    base = _inprocess(commands=False)
    for r in RXS:
        assert np.array_equal(bits(served_y[r]), bits(ref[r])), r
        k = recs["A"].k_applied[r]
        assert np.array_equal(bits(ref[r][:k]), bits(base[r][:k]))
        assert ref[r][k] != base[r][k]  # the switch is exactly at k_A on this RX
    assert from_ticks(recs["A"].k_applied["rx1.rx0"], F) >= T_A
