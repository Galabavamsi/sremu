"""Bitwise output recorder for engine refactors (not collected: no ``test_`` prefix).

Records, for a fixed set of engine runs, the SHA-256 of every RX stream's samples (complex64
digital and, where captured, the complex128 accumulator), every command record (status,
times, switch samples, lateness, config hash), the order and configuration of the records
handed to sinks, counters and per-link decisions. Run it once on the code before a change
and once after, then compare: any difference in any output byte shows up as a mismatch.

    python -m tests.runtime._perf_record record OUT.json [--only NAME[,NAME]]
    python -m tests.runtime._perf_record compare BEFORE.json AFTER.json

Cases: the B4 control scene (Urap at 1 kHz, static and mobile), mobile F3 scenes with 1 and
8 links, scripted command timelines on the WP2 lab-like model (transitions, by-edge
classes, ties, enable commands, asap and sample forms, late commands under both policies, a
geometry event re-mapping pending commands; static, mobile, both backends), a two-panel
FakePropagation scene, 600 up-front commands, the batch path, an E-B2 point (timed, asap and
tick-based semantics) and in-process E-A IQ points on the mobile corridor (Urap and the
delayed exact optimum). Everything is SIM geometry; nothing here is a physics result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import warnings
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _h(a: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _fr(x: Any) -> Any:
    if isinstance(x, Fraction):
        return f"{x.numerator}/{x.denominator}"
    return x


def _rec(r: Any) -> list[Any]:
    return [
        r.cmd_id,
        r.seq,
        r.src,
        r.op,
        r.panel,
        r.status,
        _fr(r.t_req),
        _fr(r.t_applied),
        sorted(r.k_req.items()),
        sorted(r.k_applied.items()),
        sorted(r.lateness_samples.items()),
        r.config_hash,
        r.transition,
        bool(r.counterfactual),
    ]


class _Sink:
    """Collects RX samples per stream (digital complex64) and the emitted records."""

    def __init__(self) -> None:
        from sremu.engine.emulator import EngineSink

        self.rx: dict[str, list[tuple[int, np.ndarray]]] = {}
        self.emitted: list[list[Any]] = []
        self._base = EngineSink

    def attach(self, eng: Any) -> None:
        sink = self
        from sremu.engine.emulator import EngineSink

        class S(EngineSink):
            def on_rx(self, stream: str, k0: int, d: np.ndarray) -> None:
                sink.rx.setdefault(stream, []).append((int(k0), np.array(d, copy=True)))

            def on_record(self, rec: Any, cfg: Any) -> None:
                sink.emitted.append(
                    [
                        rec.cmd_id,
                        rec.status,
                        None if cfg is None else cfg.config_hash(),
                    ]
                )

        eng.add_sink(S())

    def digest(self) -> dict[str, Any]:
        out = {}
        for s, parts in sorted(self.rx.items()):
            parts = sorted(parts, key=lambda p: p[0])
            k = 0
            for k0, d in parts:
                assert k0 == k, f"{s}: gap or overlap at {k0} (expected {k})"
                k += d.shape[0]
            y = np.concatenate([d for _, d in parts]) if parts else np.zeros(0, np.complex64)
            out[s] = {"n": int(y.shape[0]), "sha": _h(y)}
        return out


def _engine_digest(eng: Any, sink: _Sink, caps: dict[str, list] | None = None) -> dict[str, Any]:
    s = eng.summary()
    info = eng.manifest_info()
    out: dict[str, Any] = {
        "rx": sink.digest(),
        "records": [_rec(r) for r in eng.records],
        "emitted": sink.emitted,
        "counters": dict(sorted(s.counters.items())),
        "samples": sorted(s.samples.items()),
        "verify_interp_max_rel": s.verify_interp_max_rel,
        "nondeterministic_inputs": s.nondeterministic_inputs,
        "link_decisions": info["link_decisions"],
    }
    if caps:
        out["acc"] = {
            r: _h(np.concatenate([y for _, y in sorted(c, key=lambda p: p[0])]))
            for r, c in sorted(caps.items())
        }
    return out


# =========================================================================== cases


def _bench_engine(links: int, feat: str, dur: str, **kw: Any) -> Any:
    from bench._engine_point import _resolve, build_engine
    from bench._scenes import bench_scenario

    res = _resolve(
        bench_scenario(
            links=links, n_cells=240, n_ris=1, rate_sps="1e6", features=feat, duration_s=dur
        )
    )
    return build_engine(res, feat, **kw)


def case_b4(feat: str, dur: str) -> dict[str, Any]:
    from sremu.contracts import LatencyModel
    from sremu.control.builtin import Urap

    eng = _bench_engine(2, feat, dur)
    sink = _Sink()
    sink.attach(eng)
    lm = Fraction(1, 10_000)
    eng.register_controller(
        Urap(
            0.5,
            name="b4",
            period_s=Fraction(1, 1000),
            min_latency_s=lm,
            latency=LatencyModel(lm, f"uniform:0,{float(lm)!r}"),
        )
    )
    caps = {r: eng.capture_accumulator(r) for r in eng.rx}
    eng.run(until_s=Fraction(dur))
    return _engine_digest(eng, sink, caps)


def case_mobile(links: int, feat: str, dur: str) -> dict[str, Any]:
    eng = _bench_engine(links, feat, dur)
    sink = _Sink()
    sink.attach(eng)
    caps = {r: eng.capture_accumulator(r) for r in eng.rx}
    eng.run(until_s=Fraction(dur))
    return _engine_digest(eng, sink, caps)


def _lab_dict(mobile: bool) -> tuple[dict[str, Any], Path]:
    from sremu.scenario.load import load_scenario

    res = load_scenario(REPO / "scenarios" / "lab_like_A.yaml", warn=False)
    d = copy.deepcopy(res.scenario.model_dump(mode="python"))
    d["controllers"] = []
    d["traffic"] = [
        {"stream": "ap.tx0", "generator": "tone", "params": {"freq_hz": 1.0e5}},
    ]
    d["run"]["duration_s"] = Fraction(1, 20)
    rx2 = next(n for n in d["nodes"] if n["id"] == "rx2")
    rx2["trajectory"] = {"model": "external", "initial_m": list(rx2.pop("position_m"))}
    if mobile:
        rx1 = next(n for n in d["nodes"] if n["id"] == "rx1")
        pos = rx1.pop("position_m")
        rx1["trajectory"] = {
            "model": "constant_velocity",
            "start_m": list(pos),
            "velocity_mps": [0.0, 1.5, 0.0],
        }
    assert res.base_dir is not None
    return d, res.base_dir


def case_lab_scripted(mobile: bool, backend: str) -> dict[str, Any]:
    from sremu.contracts import CommandRequest, CommandTime, TransitionSpec
    from sremu.engine.emulator import Emulator

    d, base = _lab_dict(mobile)
    eng = Emulator.from_scenario(d, base_dir=base, backend=backend, always_active=True)
    sink = _Sink()
    sink.attach(eng)
    caps = {r: eng.capture_accumulator(r) for r in eng.rx}
    n = eng.commands.specs["ris0"].n_controls
    rng = np.random.Generator(np.random.Philox(key=11))
    receipts: list[Any] = []

    def cfg(cid: str, at: Any, states: Any = None, **kw: Any) -> None:
        st = rng.integers(0, 2, n).astype(np.int16) if states is None else states
        receipts.append(
            eng.schedule(
                CommandRequest(
                    cid, "ris.set_config", kw.pop("src", "t"), "ris0", at, states=st, **kw
                )
            )
        )

    def en(cid: str, at: Any, flag: bool, **kw: Any) -> None:
        receipts.append(
            eng.schedule(
                CommandRequest(cid, "ris.set_enabled", "t", "ris0", at, enabled=flag, **kw)
            )
        )

    ms = Fraction(1, 1000)
    lin = TransitionSpec("linear", Fraction(40, 10**6), "single")
    fo = TransitionSpec("first_order", Fraction(15, 10**6), "by_edge")
    cfg("a", CommandTime.at(2 * ms))
    cfg("b", CommandTime.at(3 * ms), transition=lin)
    cfg("b2", CommandTime.at(3 * ms + Fraction(20, 10**6)), transition=fo)  # inside b's ramp
    cfg("c", CommandTime.at(5 * ms))
    cfg("c_tie", CommandTime.at(5 * ms))  # supersedes c
    cfg("sub", CommandTime.at(6 * ms), states=np.ones(5, np.int16), elements=(0, 3, 7, 11, 19))
    en("off", CommandTime.at(7 * ms), False)
    en("on", CommandTime.at(Fraction(85, 10_000)), True)
    en("on_tie", CommandTime.at(Fraction(85, 10_000)), True)
    cfg("late_insert", CommandTime.at(Fraction(45, 10_000)), transition=lin)  # between c and b2
    cfg("s1", CommandTime.at_sample("rx2.rx0", 9_500))
    for i in range(40):  # a burst that exercises insertion in the middle of the timeline
        t = 10 * ms + Fraction((i * 7919) % 40, 4000)
        cfg(f"burst{i}", CommandTime.at(t), transition=lin if i % 3 == 0 else None)
    eng.run(until_s=12 * ms)
    cfg("asap1", CommandTime.asap())
    cfg("late_apply", CommandTime.at(5 * ms))  # late: moved to the frontier
    cfg("late_rej", CommandTime.at(5 * ms), late_policy="reject")
    en("late_en", CommandTime.at(4 * ms), False)
    cfg("future", CommandTime.at(25 * ms), transition=fo)
    cfg("future2", CommandTime.at(26 * ms))
    receipts.append(
        eng.schedule(
            CommandRequest(
                "move",
                "node.set_position",
                "t",
                "rx2",
                CommandTime.at(20 * ms),
                position_m=np.array([0.9, 0.75, 0.05]),
            )
        )
    )
    eng.run(until_s=18 * ms)
    for i in range(30):
        cfg(f"tail{i}", CommandTime.at(30 * ms + i * Fraction(1, 2000)))
    eng.run(until_s=Fraction(1, 20))
    out = _engine_digest(eng, sink, caps)
    out["receipts"] = [
        [r.cmd_id, bool(r.ok), r.status, int(r.error), r.seq, _fr(r.t_req), sorted(r.k_req.items())]
        for r in receipts
    ]
    return out


def case_fake_two_panel(moving: bool) -> dict[str, Any]:
    from sremu.contracts import Action, LatencyModel, TransitionSpec
    from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
    from sremu.testing.scenes import stream
    from tests.runtime._engine_support import ScriptedController, make_engine

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
            "rx2", (0.982982453, 0.688291724, 0.0), streams=(stream("rx2.rx0", "rx", fc, "1e6"),)
        ),
    ]
    p = 0.0428
    panels = [
        FakePanel(
            "ris0",
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            24,
            5,
            p,
            p,
            gamma=(1 + 0j, -1 + 0j),
            tiles_u=3,
            tiles_v=1,
        ),
        FakePanel(
            "ris1",
            (0, 1.5, 0),
            (1, 0, 0),
            (0, 1, 0),
            8,
            5,
            p,
            p,
            gamma=(1 + 0j, 1j),
            tiles_u=2,
            tiles_v=1,
        ),
    ]
    model = FakePropagation(nodes, panels, horizon_s="2")
    from sremu.engine.streams import ToneSource

    eng = make_engine(
        model,
        backend="numba_cpu",
        noise=True,
        bw=None,
        traffic={"ap.tx0": ToneSource(Fraction(10**6), 5e4, 1.0)},
        always_active=True,
        render_tiles="auto",
        t_u_s=Fraction(1, 2000) if moving else None,
    )
    sink = _Sink()
    sink.attach(eng)
    n0 = eng.commands.specs["ris0"].n_controls
    n1 = eng.commands.specs["ris1"].n_controls
    lin = TransitionSpec("linear", Fraction(30, 10**6), "by_edge")

    def script(obs: Any, m: int) -> list[Action]:
        r = np.random.Generator(np.random.Philox(key=100 + m))
        acts = [
            Action(
                "ris0",
                states=r.integers(0, 2, n0).astype(np.int16),
                transition=lin if m % 2 else None,
            )
        ]
        if m % 3 == 0:
            acts.append(Action("ris1", states=r.integers(0, 2, n1).astype(np.int16)))
        if m % 5 == 4:
            acts.append(Action("ris1", enabled=bool(m % 2)))
        return acts

    eng.register_controller(
        ScriptedController(
            "sc",
            "1/1250",
            "1/10000",
            script,
            latency=LatencyModel(Fraction(1, 10000), "uniform:0,2e-4"),
        )
    )
    caps = {r: eng.capture_accumulator(r) for r in eng.rx}
    eng.run(until_s=Fraction(3, 10))
    return _engine_digest(eng, sink, caps)


def case_upfront(n_cmd: int) -> dict[str, Any]:
    from sremu.contracts import CommandRequest, CommandTime
    from sremu.engine.emulator import Emulator

    d, base = _lab_dict(False)
    d["run"]["duration_s"] = Fraction(n_cmd, 2000) - Fraction(1, 20)
    eng = Emulator.from_scenario(d, base_dir=base, always_active=True)
    sink = _Sink()
    sink.attach(eng)
    n = eng.commands.specs["ris0"].n_controls
    rng = np.random.Generator(np.random.Philox(key=3))
    for i in range(n_cmd):
        eng.schedule(
            CommandRequest(
                f"u{i}",
                "ris.set_config",
                "up",
                "ris0",
                CommandTime.at(Fraction(i, 2000) + Fraction(1, 10**5)),
                states=rng.integers(0, 2, n).astype(np.int16),
            )
        )
    eng.run()
    return _engine_digest(eng, sink)


def case_batch() -> dict[str, Any]:
    from sremu.contracts import LatencyModel
    from sremu.control.builtin import Urap
    from sremu.engine.batch import run_batch
    from sremu.engine.streams import stream_configs_from_scenario
    from sremu.physics.propagation import from_scenario

    d, base = _lab_dict(True)
    model = from_scenario(d, base_dir=base, backend="numba_cpu")
    ctrl = Urap(
        0.4,
        name="u",
        period_s=Fraction(1, 1000),
        min_latency_s=Fraction(1, 10**4),
        latency=LatencyModel(Fraction(1, 10**4), "uniform:0,1e-4"),
    )
    res = run_batch(
        model,
        Fraction(1, 5),
        streams=stream_configs_from_scenario(d),
        controllers=[ctrl],
        seed=5,
        rate=Fraction(10**6),
    )
    return {
        "t": [_fr(t) for t in res.t],
        "w": _h(res.w),
        "h": {k: _h(v) for k, v in sorted(res.h.items())},
        "rx_power_w": {k: _h(v) for k, v in sorted(res.rx_power_w.items())},
        "mean_h2": {k: float(v).hex() for k, v in sorted(res.mean_h2.items())},
        "records": [_rec(r) for r in res.records],
    }


_EB_CAPTURE: dict[str, Any] = {}


def case_eb2(sem: str, jitter: str, n_packets: int) -> dict[str, Any]:
    import tempfile
    from types import SimpleNamespace

    from experiments import eb

    orig = eb.run_iq

    def wrapped(*a: Any, **k: Any) -> Any:
        sink, eng, summ = orig(*a, **k)
        _EB_CAPTURE["rx"] = {s: _h(b) for s, b in sorted(sink.buf.items())}
        _EB_CAPTURE["records"] = [_rec(r) for r in eng.records]
        _EB_CAPTURE["counters"] = dict(sorted(eng.summary().counters.items()))
        return sink, eng, summ

    eb.run_iq = wrapped
    try:
        with tempfile.TemporaryDirectory() as td:
            ctx = SimpleNamespace(
                params={
                    "alphabet": "lab",
                    "guard_s": "5e-4",
                    "jitter_s": jitter,
                    "semantics": sem,
                    "lead_s": "5e-4",
                    "n_packets": n_packets,
                },
                seed=20261010,
                run_dir=td,
            )
            res = eb.run_eb2_point(ctx)
            pk = (Path(td) / "packets.csv").read_bytes()
    finally:
        eb.run_iq = orig
    keep = {k: v for k, v in res.items() if k not in ("design",)}
    return {
        "result": json.loads(json.dumps(keep, default=str)),
        "packets_sha": hashlib.sha256(pk).hexdigest(),
        **copy.deepcopy(_EB_CAPTURE),
    }


def case_ea_iq(kind: str, tau: str, dur_s: int) -> dict[str, Any]:
    from experiments import ea, ea_iq

    from sremu.engine.emulator import Emulator

    cache = REPO / "runs" / "wp11" / "_ea_cache"
    if kind == "exact" and not (cache / "geom_f1.0G_h2_L20.npz").exists():
        return {"skipped": "no E-A geometry cache"}
    d = ea.corridor_scenario(1e9, 2, 20, duration_s=str(dur_s))
    d["controllers"] = []
    res = ea.resolve(d)
    eng = Emulator.from_scenario(res, noise=True, always_active=True)
    sink = _Sink()
    sink.attach(eng)
    eng.register_controller(ea_iq.make_controller(kind, tau, cache))
    eng.run(until_s=dur_s)
    return _engine_digest(eng, sink)


CASES: dict[str, Callable[[], dict[str, Any]]] = {
    "b4_F1": lambda: case_b4("F1", "0.3"),
    "b4_F3": lambda: case_b4("F3", "0.15"),
    "mobile_L1_F3": lambda: case_mobile(1, "F3", "0.1"),
    "mobile_L8_F3": lambda: case_mobile(8, "F3", "0.05"),
    "static_L8_F2": lambda: case_mobile(8, "F2", "0.05"),
    "lab_static_numba": lambda: case_lab_scripted(False, "numba_cpu"),
    "lab_mobile_numba": lambda: case_lab_scripted(True, "numba_cpu"),
    "lab_static_numpy": lambda: case_lab_scripted(False, "numpy_ref"),
    "lab_mobile_numpy": lambda: case_lab_scripted(True, "numpy_ref"),
    "fake_two_panel_static": lambda: case_fake_two_panel(False),
    "fake_two_panel_moving": lambda: case_fake_two_panel(True),
    "upfront_600": lambda: case_upfront(600),
    "batch": case_batch,
    "eb2_timed": lambda: case_eb2("timed", "1e-3", 300),
    "eb2_asap": lambda: case_eb2("asap", "5e-4", 200),
    "eb2_tick": lambda: case_eb2("tick-based", "5e-4", 200),
    "ea_iq_urap": lambda: case_ea_iq("urap", "0.1", 20),
    "ea_iq_exact": lambda: case_ea_iq("exact", "0.1", 20),
}


def record(out: Path, only: list[str] | None) -> None:
    warnings.simplefilter("ignore")
    res: dict[str, Any] = {}
    for name, fn in CASES.items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        res[name] = fn()
        print(f"{name}: {time.perf_counter() - t0:.1f} s", flush=True)
    out.write_text(json.dumps(res, indent=1, default=str))


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present on one side only")
            else:
                _diff(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def compare(a_p: Path, b_p: Path) -> int:
    a = json.loads(a_p.read_text())
    b = json.loads(b_p.read_text())
    bad = 0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            print(f"{name}: only in one file")
            bad += 1  # a case missing on one side is not a pass
            continue
        out: list[str] = []
        _diff(a[name], b[name], name, out)
        print(f"{name}: {'IDENTICAL' if not out else f'{len(out)} DIFFERENCES'}")
        for line in out[:20]:
            print("   ", line)
        bad += bool(out)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("out")
    r.add_argument("--only", default="")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "record":
        record(Path(args.out), [s for s in args.only.split(",") if s] or None)
        return 0
    return compare(Path(args.a), Path(args.b))


if __name__ == "__main__":
    raise SystemExit(main())
