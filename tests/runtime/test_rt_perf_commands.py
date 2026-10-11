"""WP6 engine performance: the incremental command timeline and the dispatch shortcuts are
exact (DESIGN.md sections 4.6, 4.7; RT-06/07 semantics, I2a).

* **Equivalence.** The incremental panel book (bisect insert, partial recompose, maintained
  per-RX arrays, slice prune) and the heap-based status update are driven in lockstep with
  the pre-change brute-force implementation (``_ref_commands``) on seeded random timelines:
  config and enable commands, subsets, every transition profile and class rule, same-time
  ties, out-of-order and late arrivals under both late policies, asap and sample forms,
  geometry re-maps, prunes and independent RX frontiers. Every observable (records,
  statuses and their order, versions, targets, mixtures, epochs, per-RX views, queries)
  must be identical after every step.
* **Structure** (deterministic, no timing): appending a command recomposes one command; a
  status update with nothing due pops no heap entry; a later insert keeps the current
  piece's cached jobs; the jobs cache key is complete (rendering with the cache disabled is
  bitwise the same); the block caches (RX phase noise, latency uniforms) and the shared
  plan combine return bitwise the values they replace.
* Timing benchmarks live in ``test_bench_wp6_perf.py`` (nightly, not CI).

SIM geometry only (FakePropagation and the B2 benchmark scene); nothing here is a physics
result.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from sremu.contracts import CommandRequest, CommandTime, TransitionSpec
from sremu.engine import commands as cmdmod
from sremu.engine.commands import CommandManager, PanelSpec, RxPort, panel_specs_from_model
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import stream
from sremu.timebase import from_ticks
from tests.runtime._engine_support import config_request, lab_like_fake, make_engine
from tests.runtime._ref_commands import RefManager

pytestmark = pytest.mark.runtime

F = Fraction(10**6)
RXS = ("rx1.rx0", "rx2.rx0")


# =========================================================================== helpers


def _scene(moving: bool) -> FakePropagation:
    """The E-B-like fake scene with rx2 on an external trajectory (timed position events
    move it) far along the panel normal, so a move shifts its switch samples."""
    fc = 3.5e9
    nodes = [
        FakeNode("ap", (1.732050808, -1.0, 0.0), streams=(stream("ap.tx0", "tx", fc, "1e6"),)),
        FakeNode(
            "rx1",
            (1.159110992, 0.310582854, 0.0),
            (0.0, 1.5, 0.0) if moving else (0.0, 0.0, 0.0),
            streams=(stream("rx1.rx0", "rx", fc, "1e6"),),
        ),
        FakeNode(
            "rx2", (900.0, 0.0, 0.0), streams=(stream("rx2.rx0", "rx", fc, "1e6"),), external=True
        ),
    ]
    p = 0.0428
    panel = FakePanel(
        "ris0", (0, 0, 0), (1, 0, 0), (0, 1, 0), 24, 5, p, p, gamma=(1 + 0j, -1 + 0j), tiles_u=3
    )
    return FakePropagation(nodes, [panel], horizon_s="2")


def _grouped_stuck(spec: PanelSpec) -> PanelSpec:
    """Two cells per control; cells 0-23 (controls 0-11) and a few others stuck at state 0:
    different controls can give the same physical target (the early stop of the recompose
    must compare controls, not only targets)."""
    n = spec.n_cells
    stuck = np.zeros(n, dtype=bool)
    stuck[:24] = True
    stuck[[29, 41, 57]] = True

    def failures(s: np.ndarray) -> np.ndarray:
        out = np.asarray(s, np.int16).copy()
        out[stuck] = 0
        return out

    return dataclasses.replace(
        spec,
        ctrl_of_cell=np.arange(n, dtype=np.int64) // 2,
        initial_controls=np.zeros(n // 2, np.int16),
        failures=failures,
        control_names=tuple(f"e{k}" for k in range(n // 2)),
    )


def _managers(
    moving: bool, reference: str = "ris", grouped: bool = False
) -> tuple[CommandManager, RefManager, FakePropagation]:
    model = _scene(moving)
    ports = [RxPort(r, F) for r in RXS]
    kw: dict[str, Any] = {"static_delays": not moving, "reference": reference}
    out = []
    for cls in (CommandManager, RefManager):
        specs = panel_specs_from_model(model)
        if grouped:
            specs = {pid: _grouped_stuck(sp) for pid, sp in specs.items()}
        out.append(cls(specs, ports, model.ris_delay_s, {"ris0": 3}, **kw))
    return out[0], out[1], model  # type: ignore[return-value]


def _mix_key(m: Any) -> Any:
    if m is None:
        return None
    return (
        m.base.tobytes(),
        tuple((la.mask.tobytes(), la.target.tobytes(), la.g) for la in m.layers),
    )


def _arr(a: Any) -> Any:
    return None if a is None else (a.dtype.str, a.shape, a.tobytes())


def _state(mgr: CommandManager, probes: list[Fraction], views: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "has_pending": mgr.has_pending,
        "seq": mgr.seq,
        "records": [
            (
                r.cmd_id,
                r.seq,
                r.status,
                r.t_req,
                r.t_applied,
                sorted(r.k_req.items()),
                sorted(r.k_applied.items()),
                sorted(r.lateness_samples.items()),
                r.config_hash,
            )
            for r in mgr.records
        ],
        "counts": dict(mgr.counts),
    }
    for pid, b in mgr.books.items():
        out[pid] = {
            "version": b.version,
            "head": None if b._head is None else b._head.seq,
            "init_epoch": b._initial_epoch.key,
            "config": [(c.seq, c.superseded) for c in b.config],
            "enabled": [(c.seq, c.superseded) for c in b.enabled],
            "act": [
                (
                    c.seq,
                    c.version,
                    _arr(c.target),
                    _arr(c.controls),
                    _arr(c.last),
                    tuple(_arr(m) for m in c.classes),
                    _mix_key(c.start),
                )
                for c in b.active_config()
            ],
            "act_en": [(c.seq, c.enabled) for c in b.active_enabled()],
            "epochs": [(e.key, e.is_step, _mix_key(e.start), _arr(e.target)) for e in b.epochs()],
        }
        q = []
        for t in probes:
            cfg = b.config_at(t)
            last = b.latest_config_at(t)
            q.append(
                (
                    cfg.state.tobytes(),
                    cfg.enabled,
                    [c.seq for c in b.pending_after(t)],
                    None if last is None else last.seq,
                    b.next_time_after(t),
                )
            )
        out[pid]["queries"] = q
        out[pid]["receipts"] = [
            (r.cmd_id, r.seq, r.t_req, sorted(r.k_req.items()))
            for r in mgr.pending_receipts(probes[0])
        ]
        if views:
            for rx in RXS:
                v = b.view(rx)
                out[f"{pid}/{rx}"] = (
                    _arr(v.cfg_k),
                    _arr(v.en_k),
                    _arr(v.en_kc),
                    _arr(v.en_vals),
                    _arr(v.splits),
                    v.en_initial,
                    [e.key for e in v.cfg_epochs],
                    [_arr(v.epochs_at(int(a))) for a in v.splits[:8]],
                )
    return out


def _random_request(
    rng: np.random.Generator, i: int, n_ctrl: int, frontier_t: Fraction, times: list[Fraction]
) -> CommandRequest:
    """A random ris.* request (see the module docstring for the mix)."""
    kind = rng.random()
    late_policy = "reject" if rng.random() < 0.2 else "apply_on_arrival"
    r = rng.random()
    if r < 0.06:
        at = CommandTime.asap()
    elif r < 0.12:
        at = CommandTime.at_sample(RXS[int(rng.integers(2))], int(rng.integers(0, 60_000)))
    elif r < 0.30 and times:
        at = CommandTime.at(times[int(rng.integers(len(times)))])  # a tie
    else:
        # mostly ahead of the frontier, sometimes late, sometimes far out of order
        off = Fraction(int(rng.integers(-300, 6000)), 10**6)
        at = CommandTime.at(max(Fraction(0), frontier_t + off))
    if kind < 0.18:
        return CommandRequest(
            f"e{i}",
            "ris.set_enabled",
            "t",
            "ris0",
            at,
            enabled=bool(rng.random() < 0.5),
            late_policy=late_policy,  # type: ignore[arg-type]
        )
    model = ("step", "linear", "first_order")[int(rng.integers(3))]
    classes = "by_edge" if rng.random() < 0.5 else "single"
    trans = (
        None
        if rng.random() < 0.3
        else TransitionSpec(model, Fraction(int(rng.integers(1, 400)), 10**6), classes)
    )
    if rng.random() < 0.3:
        k = int(rng.integers(1, 12))
        pool = 16 if rng.random() < 0.5 else n_ctrl  # often only the first controls
        elems = tuple(int(x) for x in rng.choice(pool, size=min(k, pool), replace=False))
        k = len(elems)
        states = rng.integers(0, 2, k).astype(np.int16)
        return config_request(
            f"c{i}", "ris0", states, at, transition=trans, late_policy=late_policy, elements=elems
        )
    states = rng.integers(0, 2, n_ctrl).astype(np.int16)
    return config_request(f"c{i}", "ris0", states, at, transition=trans, late_policy=late_policy)


# =========================================================================== equivalence


@pytest.mark.filterwarnings("ignore::sremu.errors.NondeterminismWarning")
@pytest.mark.parametrize(
    ("seed", "moving", "reference", "grouped"),
    [
        (1, False, "ris", False),
        (2, True, "ris", False),
        (3, False, "rx", False),
        (4, True, "ris", True),
        (5, False, "ris", True),
    ],
    ids=["static", "moving", "rx_reference", "moving_grouped_stuck", "static_grouped_stuck"],
)
def test_incremental_timeline_equals_bruteforce(
    seed: int, moving: bool, reference: str, grouped: bool
) -> None:
    new, ref, model = _managers(moving, reference, grouped)
    n_ctrl = new.specs["ris0"].n_controls
    rng = np.random.Generator(np.random.Philox(key=900 + seed))
    fr = dict.fromkeys(RXS, 0)
    times: list[Fraction] = []
    t_geo = Fraction(0)
    for i in range(260):
        ft = max(from_ticks(fr[r], F) for r in RXS)
        req = _random_request(rng, i, n_ctrl, ft, times)
        a = new.schedule(req, dict(fr))
        b = ref.schedule(req, dict(fr))
        assert (a.receipt.ok, a.receipt.status, a.receipt.error, a.late) == (
            b.receipt.ok,
            b.receipt.status,
            b.receipt.error,
            b.late,
        )
        assert [r.cmd_id for r in a.final] == [r.cmd_id for r in b.final]
        if a.command is not None:
            times.append(a.command.t)
        op = rng.random()
        if op < 0.35:  # advance the RX frontiers independently
            for r in RXS:
                fr[r] += int(rng.integers(0, 900))
            out_a = new.update_statuses(dict(fr))
            out_b = ref.update_statuses(dict(fr))
            assert [(r.cmd_id, r.status) for r in out_a] == [(r.cmd_id, r.status) for r in out_b]
        elif op < 0.42:
            keep = int(rng.integers(2, 12))
            new.books["ris0"].prune(dict(fr), keep=keep)
            ref.books["ris0"].prune(dict(fr), keep=keep)
        elif op < 0.48:  # rx2 jumps (hundreds of m: microseconds of delay) and re-maps
            t_e = max(from_ticks(fr[r], F) for r in RXS) + Fraction(int(rng.integers(0, 3000)), F)
            t_geo = max(t_geo, t_e)
            model.set_node_position("rx2", t_geo, (float(rng.uniform(300.0, 3000.0)), 0.0, 0.0))
            new.geometry_changed(t_geo)
            ref.geometry_changed(t_geo)
        probes = [Fraction(int(x), 10**6) for x in rng.integers(0, 70_000, 6)]
        probes.append(times[-1] if times else Fraction(0))
        views = bool(rng.random() < 0.5)  # leave some RX arrays unbuilt (lazy path)
        assert _state(new, probes, views) == _state(ref, probes, views), f"step {i}"
    # drain: everything ends applied in the same order
    big = dict.fromkeys(RXS, 10**9)
    assert [r.cmd_id for r in new.update_statuses(big)] == [
        r.cmd_id for r in ref.update_statuses(big)
    ]
    assert not new.has_pending
    assert _state(new, [Fraction(0)], True) == _state(ref, [Fraction(0)], True)


# =========================================================================== structure


def test_append_recomposes_only_the_new_command(monkeypatch: pytest.MonkeyPatch) -> None:
    new, _, _ = _managers(False)
    n = new.specs["ris0"].n_controls
    fr = dict.fromkeys(RXS, 0)
    for i in range(120):
        st = np.full(n, i % 2, np.int16)
        new.schedule(config_request(f"c{i}", "ris0", st, CommandTime.at(Fraction(i + 1, 1000))), fr)
    calls = {"n": 0}
    orig = cmdmod._classes

    def counting(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(cmdmod, "_classes", counting)
    st = np.ones(n, np.int16)
    new.schedule(config_request("last", "ris0", st, CommandTime.at(Fraction(200, 1000))), fr)
    assert calls["n"] == 1  # the full recompose would have processed 121 commands


def test_status_update_with_nothing_due_pops_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    new, _, _ = _managers(False)
    n = new.specs["ris0"].n_controls
    fr = dict.fromkeys(RXS, 0)
    for i in range(1000):
        st = np.full(n, i % 2, np.int16)
        t = Fraction(i + 10, 1000)
        new.schedule(config_request(f"c{i}", "ris0", st, CommandTime.at(t)), fr)
    pops = {"n": 0}
    real = cmdmod.heapq.heappop

    class _Heapq:
        heappush = staticmethod(cmdmod.heapq.heappush)

        @staticmethod
        def heappop(h: list) -> Any:
            pops["n"] += 1
            return real(h)

    monkeypatch.setattr(cmdmod, "heapq", _Heapq)
    for k in range(50):
        assert new.update_statuses(dict.fromkeys(RXS, 1000 + k)) == []
    assert pops["n"] == 0  # a scan would have visited 1000 commands per call
    done = new.update_statuses(dict.fromkeys(RXS, 10_100))  # past the first command only
    assert [r.cmd_id for r in done] == ["c0"]
    assert pops["n"] == len(RXS)


def _engine(moving: bool, **kw: Any) -> Any:
    from sremu.engine.streams import ToneSource

    model = lab_like_fake(moving=moving, tiles=(3, 1))
    return make_engine(
        model,
        backend="numba_cpu",
        noise=True,
        bw=None,
        traffic={"ap.tx0": ToneSource(F, 5e4, 1.0)},
        always_active=True,
        render_tiles="model",
        t_u_s=Fraction(1, 2000) if moving else None,
        **kw,
    )


def _engine_wide(moving: bool) -> Any:
    """A deliberately oversized SIM panel (24 x 5 cells at 20 m pitch, 3 x 1 render tiles)
    at 10 MS/s, so the tiles of one command switch at different samples and the per-tile
    epoch and enable flags differ inside a piece (not a physical geometry)."""
    from sremu.engine.streams import ToneSource

    fc, rate = 3.5e9, "1e7"
    nodes = [
        FakeNode("ap", (1500.0, -800.0, 0.0), streams=(stream("ap.tx0", "tx", fc, rate),)),
        FakeNode(
            "rx1",
            (2000.0, 300.0, 0.0),
            (0.0, 30.0, 0.0) if moving else (0.0, 0.0, 0.0),
            streams=(stream("rx1.rx0", "rx", fc, rate),),
        ),
        FakeNode("rx2", (1800.0, -200.0, 50.0), streams=(stream("rx2.rx0", "rx", fc, rate),)),
    ]
    panel = FakePanel(
        "ris0", (0, 0, 0), (1, 0, 0), (0, 1, 0), 24, 5, 20.0, 20.0, gamma=(1, -1), tiles_u=3
    )
    model = FakePropagation(nodes, [panel], horizon_s="2")
    return make_engine(
        model,
        backend="numba_cpu",
        noise=True,
        bw=None,
        traffic={"ap.tx0": ToneSource(Fraction(10**7), 7e5, 1.0)},
        always_active=True,
        render_tiles="model",
        t_u_s=Fraction(1, 2000) if moving else None,
    )


def test_future_insert_keeps_the_current_jobs_cached() -> None:
    eng = _engine(False)
    n = eng.commands.specs["ris0"].n_controls
    eng.run(until_s=Fraction(3, 1000))
    rx = "rx1.rx0"
    f = eng.configs[rx].rate
    a = eng.groups[0].cursor
    iv = eng.snaps.interval_for(a, f)
    before = eng.builder.jobs(rx, f, iv, a)
    st = np.ones(n, np.int16)
    eng.schedule(config_request("far", "ris0", st, CommandTime.at(Fraction(1, 10))))
    after = eng.builder.jobs(rx, f, iv, a)
    assert after is before  # the old key (book version) forced a rebuild here


class _NoStore(OrderedDict):  # type: ignore[type-arg]
    """A job cache that never keeps an entry: every piece builds its jobs."""

    def __setitem__(self, key: Any, value: Any) -> None:
        return None


@pytest.mark.filterwarnings("ignore::sremu.errors.NondeterminismWarning")
@pytest.mark.parametrize("moving", [False, True], ids=["static", "moving"])
def test_jobs_cache_key_is_complete(moving: bool) -> None:
    """Rendering with the jobs cache disabled is bitwise the same (the signature holds every
    timeline input of a piece: per-tile epochs, running-transition switch samples, enable
    flags), over transitions, by-edge classes, enable toggles, ties and late commands."""

    def run(cache: bool) -> dict[str, np.ndarray]:
        eng = _engine_wide(moving)
        k = eng.commands.books["ris0"].n_tiles
        assert k == 3
        if not cache:
            eng.builder._jobs = _NoStore()
        caps = {r: eng.capture_accumulator(r) for r in eng.rx}
        n = eng.commands.specs["ris0"].n_controls
        rng = np.random.Generator(np.random.Philox(key=77))
        times: list[Fraction] = []
        for chunk in range(6):
            ft = Fraction(chunk * 4, 1000)
            for i in range(12):
                req = _random_request(rng, chunk * 100 + i, n, ft, times)
                rc = eng.schedule(req)
                if rc.ok and rc.t_req is not None:
                    times.append(rc.t_req)
            eng.run(until_s=ft + Fraction(4, 1000))
        # teeth: some command switches its tiles at different samples on some RX
        book = eng.commands.books["ris0"]
        spread = [int(np.ptp(c.k_tiles[r])) for c in book.config for r in c.k_tiles]
        assert max(spread) > 0
        return {
            r: np.concatenate([y for _, y in sorted(c, key=lambda p: p[0])])
            for r, c in caps.items()
        }

    a, b = run(True), run(False)
    for r in a:
        assert a[r].tobytes() == b[r].tobytes()


def test_rx_phase_noise_block_cache_is_bitwise() -> None:
    from sremu.engine.streams import RxStream, stream_configs_from_model

    model = lab_like_fake()
    cfg = stream_configs_from_model(
        model, {"rx1.rx0": {"linewidth_hz": 150.0, "cfo_hz": 321.0, "noise_figure_db": 5.0}}
    )["rx1.rx0"]
    st = RxStream(cfg, seed=4, backend="numba_cpu")
    assert st.pn is not None
    rng = np.random.Generator(np.random.Philox(key=5))
    k = 0
    for _ in range(40):
        n = int(rng.integers(1, 40_000))
        acc = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-6
        got = st.process(k, acc)
        # the uncached chain (the pre-change expression for the phase-noise factor)
        y = acc + st._noise(k, n) if st._noise is not None else acc
        y = np.multiply(y, st._lo(k, n))
        y = np.multiply(y, np.conj(st.pn.phasor(k, n)))
        assert got.tobytes() == st.front.apply(y).tobytes()
        k += n


def test_latency_uniform_block_cache_is_bitwise() -> None:
    from sremu import rng as srng
    from sremu.engine.barriers import _latency_uniform

    for name in ("a", "ctrl-b"):
        key = srng.derive_key(17, "ctrl.latency", name)
        for m in (0, 1, 255, 256, 257, 1000, 77_777):
            for i in (0, 3):
                direct = float(srng.uniform(key, m, 1, sub=i)[0])
                assert _latency_uniform(17, name, m, i) == direct


def test_plan_delays_equal_unconfigured_combine() -> None:
    """The plan reads the path structure from the snapshot's configured combine; it equals
    the unconfigured combine's bitwise (paths of absent panels keep zero amplitude)."""
    eng = _engine(True)
    eng.run(until_s=Fraction(2, 1000))
    f = eng.configs["rx1.rx0"].rate
    iv = eng.snaps.interval_for(eng.groups[0].cursor, f)
    plans = eng.builder.plans(iv)
    pa = eng.model.combine(iv.snap_a, {})
    pb = eng.model.combine(iv.snap_b, {})
    for lk, pl in plans.items():
        assert pl.tau_a.tobytes() == pa[lk].delay_s.tobytes()
        assert pl.tau_b.tobytes() == pb[lk].delay_s.tobytes()
        assert pl.layout.n_paths == pa[lk].n_paths
