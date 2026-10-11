"""VA-5 on the lab arc (DESIGN.md section 11, E-B1): sample exactness of an intra-frame
switch of a six-panel RIS (one panel per tile, ``scenarios/studies/lab_arc.yaml``).

A QPSK packet goes to RX1; every panel switches from the joint RX1 optimum to the joint RX2
optimum at the same requested time (one command per panel). Each panel's change must be
applied on RX1 at ``to_ticks(t_sw + tau_panel->RX1)`` (its ``k_applied``); the first RX1
sample that differs from the no-switch run (same keyed noise) is the earliest of them, and
every earlier sample is bitwise identical. ``test_va5_arc_straddling_tick`` puts the
requested time where the tiles' RIS->RX1 delays straddle a rounding boundary, so different
panels switch on different samples; a run that switches only the early panels must match
the full run up to and including the early sample and differ on the next one.
Also: the E-B1/E-B2 point runners on the arc (plumbing, tick-based semantics) and the
``--smoke`` CLI.
"""

from __future__ import annotations

import json
import types
from fractions import Fraction

import numpy as np
import pytest
from experiments import common, eb

from sremu.contracts import CommandRequest, CommandTime
from sremu.engine.streams import QpskPacketSource
from sremu.timebase import to_ticks

pytestmark = pytest.mark.runtime

ARC_REF = "scenarios/studies/lab_arc.yaml"
START = 3000
_GIT_STUB = {
    "commit": "test",
    "dirty": False,
    "dirty_patch_sha256": None,
    "dirty_patch_path": None,
    "reason": "test stub",
    "code": None,
}


@pytest.fixture(scope="module")
def setup():
    dsg = eb.design("lab", "present", scenario=ARC_REF)
    n = START + eb.BODY + 600
    scen = eb.lab_scenario(
        "lab",
        "present",
        full_scale_dbm=dsg.tx_dbm,
        initial_states=dsg.s1,
        duration_s=Fraction(n) / eb.FS,
        scenario=ARC_REF,
    )
    res = eb.resolve(scen, eb.ARC.parent)
    src = QpskPacketSource(
        eb.TX, int(res.scenario.seed), eb.SPS, eb.N_PRE, eb.N_PAY, guard=10**9, start=START
    )
    ref, eng, _ = eb.run_iq(res, src, [], n)
    tau = {p: float(eng.model.ris_delay_s(p, eb.RX1, Fraction(0))) for p in dsg.panels}
    return dsg, res, src, n, ref, tau


def _switch(dsg, t_sw: Fraction, transition: str, panels=None) -> list[CommandRequest]:
    c2 = dsg.split(dsg.s2)
    return [
        CommandRequest(
            f"va5:{pid}",
            "ris.set_config",
            "va5",
            pid,
            CommandTime.at(t_sw),
            states=c2[pid],
            transition=eb.TRANSITIONS[transition],
        )
        for pid in (dsg.panels if panels is None else panels)
    ]


def _expected(eng, panels, t_sw: Fraction) -> dict[str, int]:
    return {
        pid: to_ticks(t_sw + Fraction(repr(float(eng.model.ris_delay_s(pid, eb.RX1, t_sw)))), eb.FS)
        for pid in panels
    }


def _first_diff(y: np.ndarray, y0: np.ndarray) -> int:
    diff = np.nonzero(y.view(np.uint64) != y0.view(np.uint64))[0]
    assert diff.size > 0
    return int(diff[0])


@pytest.mark.parametrize(("transition", "offset"), [("step", 0), ("linear20us", 1)])
def test_va5_arc_first_affected_sample(setup, transition: str, offset: int) -> None:
    dsg, res, src, n, ref, _tau = setup
    assert len(dsg.panels) == 6
    t_sw = Fraction(START) / eb.FS + Fraction(1, 4) * eb.T_P
    sink, eng, _ = eb.run_iq(res, src, _switch(dsg, t_sw, transition), n)
    recs = {r.cmd_id: r for r in eng.records}
    want = _expected(eng, dsg.panels, t_sw)
    for pid in dsg.panels:
        assert recs[f"va5:{pid}"].status == "applied"
        assert recs[f"va5:{pid}"].k_applied[eb.RX1] == want[pid]
    k0 = min(want[p] for p in dsg.changed_panels())
    y, y0 = sink.buf[eb.RX1], ref.buf[eb.RX1]
    k_first = _first_diff(y, y0)
    assert k_first == k0 + offset
    assert np.array_equal(y[:k_first].view(np.uint64), y0[:k_first].view(np.uint64))
    # the switch changes RX1's channel (not a no-op)
    assert abs(dsg.h[eb.RX1]["s2"] - dsg.h[eb.RX1]["s1"]) > 0.1 * abs(dsg.h[eb.RX1]["s1"])


def test_va5_arc_straddling_tick(setup) -> None:
    """Panels whose RIS->RX1 delay falls on either side of a rounding boundary switch on
    consecutive samples; each sample reflects exactly the panels applied by then."""
    dsg, res, src, n, ref, tau = setup
    taus = {p: Fraction(repr(tau[p])) * eb.FS for p in dsg.panels}  # in samples
    order = sorted(dsg.panels, key=lambda p: taus[p])
    gaps = [(taus[order[i + 1]] - taus[order[i]], i) for i in range(len(order) - 1)]
    _, i_gap = max(gaps)
    cut = (taus[order[i_gap]] + taus[order[i_gap + 1]]) / 2  # in samples
    early, late = order[: i_gap + 1], order[i_gap + 1 :]
    k = START + 600
    t_sw = (Fraction(2 * k + 1, 2) - cut) / eb.FS  # t_sw*f + tau crosses k + 1/2 at the cut
    sink, eng, _ = eb.run_iq(res, src, _switch(dsg, t_sw, "step"), n)
    recs = {r.cmd_id: r for r in eng.records}
    want = _expected(eng, dsg.panels, t_sw)
    assert {want[p] for p in early} == {k} and {want[p] for p in late} == {k + 1}
    for pid in dsg.panels:
        assert recs[f"va5:{pid}"].k_applied[eb.RX1] == want[pid]
    changed = set(dsg.changed_panels())
    assert changed & set(early) and changed & set(late)
    y, y0 = sink.buf[eb.RX1], ref.buf[eb.RX1]
    assert _first_diff(y, y0) == k
    part, _, _ = eb.run_iq(res, src, _switch(dsg, t_sw, "step", early), n)
    yp = part.buf[eb.RX1]
    assert np.array_equal(yp[: k + 1].view(np.uint64), y[: k + 1].view(np.uint64))
    assert yp[k + 1] != y[k + 1]


@pytest.mark.parametrize("tick", [False, True])
def test_incremental_release_is_bitwise_upfront(setup, tick: bool) -> None:
    """``run_iq(release_times=...)`` (E-B2's incremental command release, many chunk
    boundaries here) gives the same RX samples and command records as scheduling every
    command before the run, with and without the tick-based adapter (incl. overwrites)."""
    dsg, res, src, n, _ref, _tau = setup
    cfg = [dsg.split(dsg.s1), dsg.split(dsg.s2)]
    reqs, arr = [], []
    for g in range(12):
        t = Fraction(400 * g + 37 * (g % 3), 10**6)  # non-decreasing arrivals
        for pid in dsg.panels:
            reqs.append(
                CommandRequest(
                    f"g{g}:{pid}",
                    "ris.set_config",
                    "t",
                    pid,
                    CommandTime.at(t),
                    states=cfg[(g + 1) % 2][pid],
                )
            )
            arr.append(t)
    kw = {"adapter_times": arr} if tick else {}
    a, ea, _ = eb.run_iq(res, src, reqs, n, **kw)
    b, eb2, _ = eb.run_iq(
        res,
        src,
        reqs,
        n,
        release_times=arr,
        chunk_s=Fraction(1, 1000),
        lookahead_s=Fraction(1, 2000),
        **kw,
    )
    for rx in (eb.RX1, eb.RX2):
        assert np.array_equal(a.buf[rx].view(np.uint64), b.buf[rx].view(np.uint64))

    def recs(e):
        return [(r.cmd_id, r.status, r.t_applied, dict(r.k_applied)) for r in e.records]

    assert recs(ea) == recs(eb2)
    assert len(ea.records) == len(reqs)
    if tick:
        assert ea.tick_overwritten == eb2.tick_overwritten > 0
    with pytest.raises(ValueError):
        eb.run_iq(res, src, reqs[:2], n, release_times=[arr[6], arr[0]])


@pytest.fixture
def git_stub(monkeypatch):
    monkeypatch.setattr(common, "_git_cache", dict(_GIT_STUB))
    monkeypatch.setenv("SREMU_EXP_GIT_JSON", "")
    monkeypatch.setenv(eb._THREADS_ENV, "")


def test_eb1_point_arc_blocked(tmp_path, git_stub) -> None:
    ctx = types.SimpleNamespace(
        params={
            "alphabet": "ideal1",
            "los": "blocked",
            "transition": "step",
            "delta": "0.25",
            "scenario": ARC_REF,
        },
        seed=20261010,
        run_dir=tmp_path / "p0",
    )
    out = eb.run_eb1_point(ctx)
    v = out["va5"]
    assert v["pass_step_inside_packet"] is True
    assert v["n_panels"] == 6 and v["all_panels_k_applied_equal_expected"] is True
    assert sorted(v["per_panel"]) == [f"t{k}" for k in range(6)]
    assert out["design"]["rx1_direct_db"] == float("-inf")  # LOS removed
    assert out["design"]["blocker"]["los_crossed"] == {eb.RX1: True, eb.RX2: True}
    assert out["geometry_label"] == "modelled on the lab arrangement (SIM, nominal geometry)"
    for f in ("manifest.json", "meta.json", "result.json", "symbols.csv", "scenario.resolved.yaml"):
        assert (ctx.run_dir / f).is_file()
    # the idealization of the blocked arc variant is a paper-facing assumption
    note = eb.blocked_assumption(out["design"])
    assert note is not None and "RIS hops unattenuated" in note
    for f in ("meta.json", "manifest.json"):
        assert note in json.loads((ctx.run_dir / f).read_text(encoding="utf-8"))["assumptions"]


def test_eb2_point_arc_tick_based(tmp_path, git_stub) -> None:
    n_pk = 8
    ctx = types.SimpleNamespace(
        params={
            "alphabet": "lab",
            "los": "present",
            "semantics": "tick-based",
            "guard_s": "1e-4",
            "jitter_s": "3e-3",
            "n_packets": n_pk,
            "scenario": ARC_REF,
            "es_n0_db": 15.0,
        },
        seed=20261010,
        run_dir=tmp_path / "p1",
    )
    out = eb.run_eb2_point(ctx)
    assert out["semantics"] == "tick-based" and out["n_panels"] == 6
    assert out["n_commands"] == 6 * n_pk
    assert sum(out["record_status_counts"].values()) == 6 * n_pk
    lost = out["tick_overwritten"]
    # overwrites occur here (J = 3 ms > the 2 ms tick) and hit whole controller messages
    assert lost is not None and lost >= 6 and lost % 6 == 0
    assert out["record_status_counts"].get("superseded", 0) == lost
    assert out["tick_overwritten_messages"] == lost / 6
    assert abs(out["es_n0_db"][eb.RX1] - 15.0) < 1e-9 and out["es_n0_target_db"] == 15.0
    assert "legacy" not in json.dumps(out)
    rows = (ctx.run_dir / "packets.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == n_pk + 1


def test_cli_smoke_arc(tmp_path, git_stub) -> None:
    import numba

    out = tmp_path / "eb_arc"
    rc = eb.main(
        [
            "--scenario",
            ARC_REF,
            "--out",
            str(out),
            "--jobs",
            "1",
            "--threads-per-job",
            str(numba.get_num_threads()),
            "--smoke",
        ]
    )
    assert rc == 0
    summ = json.loads((out / "eb_summary_smoke.json").read_text(encoding="utf-8"))
    assert summ["ok"] is True and summ["n_failed"] == 0 and summ["va5"]["pass"] is True
    plan = json.loads((out / "eb_plan_smoke.json").read_text(encoding="utf-8"))
    assert plan["scenario"] == ARC_REF and [s["name"] for s in plan["specs"]] == [
        "eb1_smoke",
        "eb2_smoke",
    ]
    for d in ("eb1_smoke/p00000", "eb2_smoke/p00000"):
        assert (out / d / "manifest.json").is_file()
    assert (out / "results_smoke" / "E-B2" / "eb2_per.csv").is_file()
