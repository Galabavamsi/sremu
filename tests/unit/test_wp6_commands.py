"""WP6 unit tests for :mod:`sremu.engine.commands` (DESIGN.md sections 4.6, 4.7, 5.6)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import CommandRequest, CommandTime, PanelConfig, TransitionSpec
from sremu.engine.commands import (
    CommandManager,
    Mixture,
    PanelBook,
    RxPort,
    mixture_amplitudes,
    panel_specs_from_model,
    profile_value,
)
from sremu.errors import ErrorCode
from sremu.timebase import from_ticks, to_ticks
from tests.runtime._engine_support import config_request, lab_like_fake

pytestmark = pytest.mark.unit

F = Fraction(10**6)


def _mgr(**kw: object) -> tuple[CommandManager, object]:
    model = lab_like_fake(far_m=2000.0)
    specs = panel_specs_from_model(model)
    ports = [RxPort("rx1.rx0", F), RxPort("rx2.rx0", F)]
    mgr = CommandManager(specs, ports, model.ris_delay_s, {"ris0": 3}, static_delays=True, **kw)  # type: ignore[arg-type]
    return mgr, model


def test_profile_values_in_ris_time() -> None:
    d = Fraction(20, 10**6)
    assert profile_value("step", Fraction(0), d) == 1.0
    assert profile_value("linear", Fraction(-1, 10**6), d) == 0.0
    assert profile_value("linear", Fraction(5, 10**6), d) == 0.25
    assert profile_value("linear", Fraction(30, 10**6), d) == 1.0
    assert profile_value("first_order", d, d) == pytest.approx(1 - np.exp(-1), rel=1e-15)
    assert profile_value("linear", Fraction(1), None) == 1.0  # no duration: a step


def test_mixture_apply_dense_and_collapse() -> None:
    base = np.array([0, 0, 1, -1], np.int16)
    tgt = np.array([1, 1, 1, -1], np.int16)
    mix = Mixture(base)
    moving = mix.moving(tgt, 2)
    assert moving.tolist() == [True, True, False, False]  # dead cells never move
    half = mix.apply([moving], tgt, [0.25])
    w = half.dense(np.arange(3), 2)
    assert np.allclose(w, [[0.75, 0.25], [0.75, 0.25], [0.0, 1.0]])
    full = half.apply([half.moving(tgt, 2)], tgt, [1.0])
    assert full.pure
    assert full.base.tolist() == tgt.tolist()


def test_mixture_amplitudes_linear_in_gamma() -> None:
    """h(mixture) by the reference identity equals the explicit per-element sum."""
    model = lab_like_fake()
    snap = model.snapshot(Fraction(0))
    rc = snap.ris["ris0"]
    n = rc.n_cells
    rng_ = np.random.Generator(np.random.Philox(key=5))
    a = rng_.integers(0, 2, n).astype(np.int16)
    b = 1 - a
    mask = rng_.random(n) < 0.5
    mix = Mixture(a.copy()).apply([mask], b, [0.3])

    def comb(states: np.ndarray) -> dict:
        return {
            lk: lp.amp
            for lk, lp in model.combine(snap, {"ris0": PanelConfig("ris0", states)}).items()
        }

    got = mixture_amplitudes(mix, 2, comb)
    # explicit: Gamma_n = (1 - w) Gamma(a_n) + w Gamma(b_n) on masked cells
    gam = np.asarray(rc.gamma)
    g_cell = gam[a.astype(int)].astype(np.complex128)
    g_cell[mask] = 0.7 * gam[a[mask].astype(int)] + 0.3 * gam[b[mask].astype(int)]
    for lk, amp in got.items():
        r = next(i for i, (rl, _v) in enumerate(rc.rows) if rl == lk)
        expect = np.zeros(rc.n_tiles, np.complex128)
        np.add.at(expect, rc.tile_of_cell, rc.C[r] * g_cell)
        lp = model.combine(snap, {})[lk]
        tiles = amp[lp.tile >= 0]
        assert np.allclose(tiles, expect, rtol=1e-12, atol=0)


def test_book_same_time_supersede_and_subset_composition_order() -> None:
    mgr, _ = _mgr()
    fr = {"rx1.rx0": 0, "rx2.rx0": 0}
    n = mgr.specs["ris0"].n_controls
    # arrival order differs from time order: the subset command at t=2 ms is composed
    # on top of the full command at t=1 ms, whatever the arrival order
    sub = config_request(
        "sub", "ris0", np.ones(3, np.int16), CommandTime.at(Fraction(2, 1000)), elements=(0, 1, 2)
    )
    full = config_request(
        "full", "ris0", np.zeros(n, np.int16) + 1, CommandTime.at(Fraction(1, 1000))
    )
    assert mgr.schedule(sub, fr).receipt.ok
    assert mgr.schedule(full, fr).receipt.ok
    book = mgr.books["ris0"]
    act = book.active_config()
    assert [c.cmd_id for c in act] == ["full", "sub"]
    assert np.all(act[1].target == 1)
    # same t: later seq wins, earlier superseded (never takes effect)
    zeros = config_request("z", "ris0", np.zeros(n, np.int16), CommandTime.at(Fraction(1, 1000)))
    out = mgr.schedule(zeros, fr)
    assert [r.cmd_id for r in out.final] == ["full"]
    assert out.final[0].status == "superseded"
    act = book.active_config()
    assert [c.cmd_id for c in act] == ["z", "sub"]
    assert act[1].target.tolist()[:4] == [1, 1, 1, 0]  # subset over the new base
    assert book.config_at(Fraction(15, 10_000)).state.sum() == 0


def test_manager_time_forms_sample_and_asap() -> None:
    mgr, model = _mgr()
    n = mgr.specs["ris0"].n_controls
    fr = {"rx1.rx0": 1234, "rx2.rx0": 1234}
    req = CommandRequest(
        "s",
        "ris.set_config",
        "t",
        "ris0",
        CommandTime.at_sample("rx2.rx0", 5000),
        states=np.ones(n, np.int16),
    )
    out = mgr.schedule(req, fr)
    tau = model.ris_delay_s("ris0", "rx2.rx0", Fraction(0))
    assert out.receipt.t_req == from_ticks(5000, F) - Fraction(tau)
    assert out.command is not None
    assert out.command.k_centre["rx2.rx0"] == 5000  # lands exactly on the requested sample
    asap = CommandRequest(
        "a", "ris.set_config", "t", "ris0", CommandTime.asap(), states=np.zeros(n, np.int16)
    )
    out = mgr.schedule(asap, fr)
    assert out.receipt.status == "scheduled"
    assert out.command is not None
    for r in ("rx1.rx0", "rx2.rx0"):
        assert int(out.command.k_tiles[r].min()) >= 1234  # never late
    assert int(out.command.k_tiles["rx1.rx0"].min()) == 1234  # the binding RX at its frontier


@pytest.mark.parametrize(
    ("req_kw", "code"),
    [
        ({"target": "nope"}, ErrorCode.UNKNOWN_TARGET),
        ({"states": np.ones(7, np.int16)}, ErrorCode.SCHEMA_ERROR),
        ({"states_val": 5}, ErrorCode.SCHEMA_ERROR),
        ({"transition": TransitionSpec("linear", None)}, ErrorCode.SCHEMA_ERROR),
        ({"at": CommandTime.at_sample("ghost.rx0", 3)}, ErrorCode.UNKNOWN_TARGET),
        ({"elements": "tile99"}, ErrorCode.SCHEMA_ERROR),
    ],
    ids=[
        "unknown_panel",
        "wrong_length",
        "bad_state",
        "no_duration",
        "unknown_stream",
        "bad_group",
    ],
)
def test_manager_rejections_are_logged_with_seq(req_kw: dict, code: ErrorCode) -> None:
    mgr, _ = _mgr()
    n = mgr.specs["ris0"].n_controls
    states = np.full(n, req_kw.pop("states_val", 1), np.int16)
    kw = {"target": "ris0", "at": CommandTime.at(Fraction(1)), "states": states}
    kw.update(req_kw)
    req = CommandRequest(
        "x",
        "ris.set_config",
        "t",
        kw["target"],
        kw["at"],
        states=kw["states"],
        transition=kw.get("transition"),
        elements=kw.get("elements", "all"),
    )
    out = mgr.schedule(req, {"rx1.rx0": 0, "rx2.rx0": 0})
    assert not out.receipt.ok
    assert out.receipt.error == code
    assert mgr.records[-1].status == "rejected"
    assert mgr.records[-1].seq == mgr.seq


def test_manager_switch_samples_rx_reference() -> None:
    """``ris_event_reference="rx"``: k_r = to_ticks(t, f_r) on every RX and tile."""
    mgr, _ = _mgr(reference="rx")
    t = Fraction(12345, 10**7)
    ks, kc = mgr.switch_samples("ris0", t)
    for r in ("rx1.rx0", "rx2.rx0"):
        assert kc[r] == to_ticks(t, F)
        assert ks[r].tolist() == [to_ticks(t, F)] * 3


def test_book_prune_keeps_the_rendered_timeline() -> None:
    """Pruning (memory bound) drops commands no RX renders again; the remaining epochs keep
    their identity (seq, version) and targets."""
    mgr, _ = _mgr()
    n = mgr.specs["ris0"].n_controls
    fr = {"rx1.rx0": 0, "rx2.rx0": 0}
    for i in range(300):
        st = np.full(n, i % 2, np.int16)
        st[: i % n] = 1 - (i % 2)
        mgr.schedule(config_request(f"c{i}", "ris0", st, CommandTime.at(Fraction(i, 1000))), fr)
    book: PanelBook = mgr.books["ris0"]
    before = {c.seq: (c.version, c.target.copy()) for c in book.active_config()}
    mgr.update_statuses({"rx1.rx0": 280_000, "rx2.rx0": 280_000})
    after = book.active_config()
    assert len(after) < 300
    assert after[-1].cmd_id == "c299"
    for c in after:
        v, tgt = before[c.seq]
        assert c.version == v
        assert np.array_equal(c.target, tgt)
    assert book.config_at(Fraction(299, 1000)).state.tolist() == before[after[-1].seq][1].tolist()
