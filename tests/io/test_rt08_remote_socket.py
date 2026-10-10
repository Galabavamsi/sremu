"""RT-08 remote half and RT-17 over the control socket (DESIGN.md sections 4.9, 8.5, 8.7).

A barrier controller running remotely (``sremu.control.remote``: the same controller class,
driven over the control socket's ``ctrl.barrier`` / ``ris.set_config`` / ``ctrl.ack``
protocol, in a thread or in another OS process) must give RX output **bitwise identical**
to the same controller in-process, whether it answers at once or sleeps 50 ms per barrier;
every action has lateness 0. The scene and controller are those of the in-process RT-08
test (tests/runtime/test_rt08_rt17_barriers.py): moving receiver, RIS, thermal noise, a
scripted controller alternating the panel with keyed latency jitter; plus WP9's ``Urap``,
whose decisions come from the keyed generator handed to ``setup`` (seed parity).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import Action, LatencyModel, Observation
from sremu.control.builtin import Urap
from sremu.control.remote import RemoteControllerRunner
from sremu.engine.streams import ToneSource
from sremu.errors import ErrorCode, LookaheadViolation
from tests.io._wp8_support import bits, client_cmd, served, wait_until
from tests.runtime._engine_support import (
    ScriptedController,
    alternate_script,
    captured,
    lab_like_fake,
    make_engine,
)

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
RXS = ("rx1.rx0", "rx2.rx0")
N = 8000


def _engine(backend: str, **kw: object):
    eng = make_engine(
        lab_like_fake(moving=True),
        traffic={"ap.tx0": ToneSource(F, 1.0e3, 1.0)},
        noise=True,
        always_active=True,
        render_tiles="auto",
        backend=backend,
        **kw,
    )
    return eng, {r: eng.capture_accumulator(r) for r in RXS}


def _scripted(n: int, sleep_s: float = 0.0) -> ScriptedController:
    return ScriptedController(
        "alt",
        "1/1000",
        "2/10000",
        alternate_script("ris0", n),
        latency=LatencyModel(Fraction(2, 10_000), "uniform:0,3e-4"),
        sleep_s=sleep_s,
    )


def _urap() -> Urap:
    return Urap(
        0.5,
        name="urap",
        period_s="1/1000",
        min_latency_s="1/10000",
        latency=LatencyModel(Fraction(1, 10_000), "exp:1e-4"),
    )


def _n(eng) -> int:
    return eng.commands.specs["ris0"].n_controls


def _inprocess(make_ctrl, backend: str) -> tuple[dict[str, np.ndarray], list]:
    eng, caps = _engine(backend)
    eng.register_controller(make_ctrl(_n(eng)))
    eng.run(Fraction(N) / F)
    return {r: captured(c, 0, N) for r, c in caps.items()}, list(eng.records)


def _remote(make_ctrl, backend: str) -> tuple[dict[str, np.ndarray], list, object]:
    eng, caps = _engine(backend, duration_s=Fraction(N) / F)
    with served(eng, start_paused=True) as srv:
        runner = RemoteControllerRunner(make_ctrl(_n(eng)), str(srv.endpoints["control"]))
        runner.start()
        srv.resume()
        wait_until(srv.is_done, 60.0, "the end of the served run")
        runner.stop()
        assert srv.engine_error is None, srv.engine_error
    assert runner.error is None, runner.error
    return {r: captured(c, 0, N) for r, c in caps.items()}, list(eng.records), runner


def _check_records(recs: list, src: str, n_min: int) -> None:
    mine = [r for r in recs if r.src == src]
    assert len(mine) >= n_min
    for r in mine:
        assert r.status == "applied", r
        assert all(v == 0 for v in r.lateness_samples.values()), r


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
@pytest.mark.parametrize("sleep_s", [0.0, 0.05], ids=["sleep0", "sleep50ms"])
def test_rt08_remote_controller_bitwise_equal_inprocess(backend: str, sleep_s: float) -> None:
    ref, ref_recs = _inprocess(lambda n: _scripted(n), backend)
    got, recs, runner = _remote(lambda n: _scripted(n, sleep_s), backend)
    _check_records(ref_recs, "alt", 7)
    _check_records(recs, "alt", 7)
    assert runner.stats.barriers == len([r for r in recs if r.src == "alt"])
    # the same commands, at the same times, with the same ids
    key = lambda r: (r.cmd_id, r.t_req, r.t_applied, tuple(sorted(r.k_applied.items())))  # noqa: E731
    assert sorted(map(key, recs)) == sorted(map(key, ref_recs))
    for r in RXS:
        assert np.array_equal(bits(got[r]), bits(ref[r])), r


def test_rt08_remote_builtin_urap_keyed_rng_parity() -> None:
    """``Urap`` draws from the generator given to ``setup``: the remote runner rebuilds it
    from the run's seed, so its random configurations equal the in-process ones."""
    ref, ref_recs = _inprocess(lambda n: _urap(), "numpy_ref")
    got, recs, _ = _remote(lambda n: _urap(), "numpy_ref")
    _check_records(recs, "urap", 7)
    hashes = [r.config_hash for r in recs if r.src == "urap"]
    assert hashes == [r.config_hash for r in ref_recs if r.src == "urap"]
    assert len(set(hashes)) > 3  # really random configurations
    for r in RXS:
        assert np.array_equal(bits(got[r]), bits(ref[r])), r


def test_rt08_remote_controller_in_separate_os_process() -> None:
    ref, _ = _inprocess(lambda n: _scripted(n), "numpy_ref")
    eng, caps = _engine("numpy_ref", duration_s=Fraction(N) / F)
    with served(eng, start_paused=True) as srv:
        import subprocess

        from tests.io._wp8_support import REPO

        proc = subprocess.Popen(
            client_cmd(
                "ctrl",
                "--sessions",
                str(srv.endpoints["control"]),
                "--n-controls",
                str(_n(eng)),
                "--sleep-s",
                "0.02",
                "--idle-s",
                "2",
            ),
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            wait_until(
                lambda: "alt" in srv.controllers and srv.controllers["alt"].ident,
                60.0,
                "the remote controller to register",
            )
            srv.resume()
            wait_until(srv.is_done, 120.0, "the end of the served run")
            out, err = proc.communicate(timeout=60)
        finally:
            proc.kill()
        assert proc.returncode == 0, err
    assert "ctrl served" in out
    _check_records(list(eng.records), "alt", 7)
    for r in RXS:
        assert np.array_equal(bits(captured(caps[r], 0, N)), bits(ref[r])), r


def test_rt17_remote_action_before_lookahead_rejected() -> None:
    """A remote action earlier than t_b + L_min reaches the engine, which rejects it with
    LOOKAHEAD_VIOLATION; the runner raises LookaheadViolation (as the in-process engine
    would abort the run) after acking and unregistering."""

    def bad(n: int) -> ScriptedController:
        def script(obs: Observation, m: int) -> list[Action]:
            return [Action("ris0", states=np.ones(n, np.int16), at_s=obs.t_b + Fraction(1, 10**4))]

        return ScriptedController("bad", "1/1000", "2/10000", script)

    eng, _ = _engine("numpy_ref", duration_s=Fraction(3, 1000))
    with served(eng, start_paused=True) as srv:
        runner = RemoteControllerRunner(bad(_n(eng)), str(srv.endpoints["control"]))
        runner.start()
        srv.resume()
        wait_until(lambda: runner.error is not None, 30.0, "the runner to fail")
        wait_until(srv.is_done, 30.0, "the run to end without the controller")
        runner.stop()
    assert isinstance(runner.error, LookaheadViolation)
    assert runner.stats.receipts[0].error == ErrorCode.LOOKAHEAD_VIOLATION
    rec = [r for r in eng.records if r.src == "bad"]
    assert rec and rec[0].status == "rejected"


def test_rt08_scenario_external_controller_waits_for_attach() -> None:
    """A scenario controller of kind ``external`` is registered before the run: its
    barriers start at t_b = 0 and the engine waits (publishing sim.stall) until the remote
    side attaches by name with matching parameters; a mismatching registration is refused."""
    from sremu.transport.client import ClientSession, RemoteError
    from sremu.transport.control_server import ControlServer, ServerConfig
    from tests.io._wp8_support import EPHEMERAL

    eng, caps = _engine("numpy_ref", duration_s=Fraction(N) / F)
    scenario = {
        "nodes": [],
        "controllers": [
            {
                "name": "alt",
                "kind": "external",
                "period_s": Fraction(1, 1000),
                "min_latency_s": Fraction(2, 10_000),
                "latency": {"fixed_s": Fraction(2, 10_000), "jitter": "uniform:0,3e-4"},
                "observation": {
                    "fields": ["rx_power_dbm", "ris_state"],
                    "window_s": Fraction(1, 1000),
                },
            }
        ],
    }
    srv = ControlServer(eng, ServerConfig(**EPHEMERAL), scenario=scenario).start()
    try:
        assert "alt" in srv.controllers and srv.controllers["alt"].preregistered
        wait_until(lambda: srv.controllers["alt"].pending == 0, 30.0, "the first barrier")
        s = ClientSession(str(srv.endpoints["control"]))
        with pytest.raises(RemoteError) as ei:
            s.call("ctrl.register", name="alt", period_s="1/500", min_latency_s="2e-4")
        assert ei.value.code == ErrorCode.NOT_PERMITTED
        s.close()
        runner = RemoteControllerRunner(_scripted(_n(eng)), str(srv.endpoints["control"]))
        runner.start()
        wait_until(srv.is_done, 60.0, "the end of the served run")
        runner.stop()
    finally:
        srv.stop()
    ref, _ = _inprocess(lambda n: _scripted(n), "numpy_ref")
    _check_records(list(eng.records), "alt", 7)
    for r in RXS:
        assert np.array_equal(bits(captured(caps[r], 0, N)), bits(ref[r])), r
