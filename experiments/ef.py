"""E-F: end-to-end link metrics (DESIGN.md section 11, E-F; RQ-37; ST-04 is the CI version).

**BER.** Static lab-like LOS + RIS (``scenarios/lab_like_A.yaml`` geometry, lab alphabet
with |Gamma| = 1 ASSUMED, fixed configuration = exact optimum for RX1; RX2 removed), BPSK
and QPSK at 4 samples/symbol with rectangular pulses, frames of 64 known pilot + 512 data
symbols back to back. Eb/N0 in {0, 2, ..., 10} dB is set through the TX power against the
engine's **physical** noise (``N0 = k T0 F``, NF 7 dB; no fudge factor):
``P_rx = (Eb/N0) N0 f_s b / SPS`` with b bits per symbol. Receiver: integrate-and-dump over
each symbol, coherent detection with

* ``genie``: the true narrowband channel from the scene model (oracle, flagged), and
* ``ls64``: a one-tap LS estimate from each frame's 64 pilots.

Each point runs until both variants have >= 200 bit errors or 2e7 data bits. Theory:
``Q(sqrt(2 Eb/N0))`` (both modulations, Gray QPSK). Clopper-Pearson 95 % intervals.

**API latency.** In-process ``send``/``recv`` call latency of 1000-sample buffers
(``sremu.api.Device.inproc``), the native-ZMQ two-process round trip (served emulator in a
second OS process, ``Device.connect``) and the sustained streaming rate (MS/s). The host's
background load is sampled and recorded; numbers are this laptop's, not a claim about any
other platform.
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import time
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erfc

from experiments.common import (
    REPO,
    RunRecord,
    db10,
    quiet_warnings,
    write_csv,
    write_json,
    write_meta,
)
from experiments.eb import RX1, TX, design, lab_scenario, resolve
from sremu import rng as _rng
from sremu.analysis.stats import clopper_pearson
from sremu.control.controller import csi_from_snapshot
from sremu.engine.emulator import Emulator, EngineSink
from sremu.physics.propagation import from_scenario
from sremu.units import K_B, T0

FS = 10**6
SPS = 4
N_PIL = 64
N_DAT = 512
FRAME_SYM = N_PIL + N_DAT
FRAME = FRAME_SYM * SPS  # 2304 samples
NF_DB = 7.0
EBN0_DB = (0, 2, 4, 6, 8, 10)
MIN_ERR = 200
MAX_BITS = 20_000_000
SEED = 20261010
PAGE_TARGET_S = "4e-3"  # 4096-sample pages: fewer Python dispatches (I2b: <= 1e-9 relative)


def q_ber(ebn0_db: float) -> float:
    """``Q(sqrt(2 Eb/N0))``."""
    return 0.5 * float(erfc(math.sqrt(10 ** (ebn0_db / 10))))


class FramedSource:
    """BPSK or QPSK frames (64 pilots + 512 data symbols, 4 samples/symbol, rectangular),
    back to back from sample 0; symbols keyed by purpose ``traffic/<stream>`` and the
    absolute symbol index (a pure function of the sample index)."""

    def __init__(self, stream: str, seed: int, mod: str) -> None:
        self.key = _rng.derive_key(seed, "traffic", stream)
        self.mod = mod
        self._cache: dict[int, np.ndarray] = {}

    def symbols(self, f: int) -> np.ndarray:
        """Unit-energy symbols of frame ``f`` (pilots first)."""
        s = self._cache.get(f)
        if s is not None:
            return s
        w = _rng.philox_block(self.key, f * FRAME_SYM, FRAME_SYM, sub=2)
        b = (w[:, :2] >> np.uint32(31)).astype(np.float64)
        if self.mod == "bpsk":
            s = (1.0 - 2.0 * b[:, 0]).astype(np.complex128)
        else:
            s = ((1.0 - 2.0 * b[:, 0]) + 1j * (1.0 - 2.0 * b[:, 1])) / math.sqrt(2.0)
        if len(self._cache) > 64:
            self._cache.pop(min(self._cache))
        self._cache[f] = s
        return s

    def __call__(self, k0: int, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.complex64)
        f0, f1 = k0 // FRAME, (k0 + n - 1) // FRAME
        for f in range(f0, f1 + 1):
            a, b = max(k0, f * FRAME), min(k0 + n, (f + 1) * FRAME)
            idx = (np.arange(a, b) - f * FRAME) // SPS
            out[a - k0 : b - k0] = self.symbols(f)[idx].astype(np.complex64)
        return out


class BerSink(EngineSink):
    """Demodulates whole frames of one RX stream as they arrive; counts data-bit errors."""

    def __init__(self, src: FramedSource, h_genie: complex, stream: str = RX1) -> None:
        self.src = src
        self.h = complex(h_genie) * SPS  # integrate-and-dump gain
        self.stream = stream
        self.buf = np.zeros(0, dtype=np.complex64)
        self.k_buf = 0
        self.frames = 0
        self.bits = 0
        self.err = {"genie": 0, "ls64": 0}
        self.h_ls_err: list[float] = []

    def on_rx(self, stream: str, k0: int, d: np.ndarray) -> None:
        if stream != self.stream:
            return
        if k0 != self.k_buf + self.buf.size:
            raise RuntimeError(f"non-contiguous RX at {k0}")
        self.buf = np.concatenate([self.buf, d])
        nfr = (self.k_buf + self.buf.size) // FRAME - self.k_buf // FRAME
        if nfr <= 0:
            return
        f0 = self.k_buf // FRAME
        assert self.k_buf % FRAME == 0
        for i in range(nfr):
            seg = self.buf[i * FRAME : (i + 1) * FRAME].astype(np.complex128)
            r = seg.reshape(FRAME_SYM, SPS).sum(axis=1)
            s = self.src.symbols(f0 + i)
            pil, dat = s[:N_PIL], s[N_PIL:]
            h_ls = complex(np.vdot(pil, r[:N_PIL]) / np.vdot(pil, pil).real)
            self.h_ls_err.append(abs(h_ls - self.h) / abs(self.h))
            for name, h in (("genie", self.h), ("ls64", h_ls)):
                z = r[N_PIL:] / h
                if self.src.mod == "bpsk":
                    e = int(np.count_nonzero((z.real < 0) != (dat.real < 0)))
                else:
                    e = int(np.count_nonzero((z.real < 0) != (dat.real < 0)))
                    e += int(np.count_nonzero((z.imag < 0) != (dat.imag < 0)))
                self.err[name] += e
            self.bits += N_DAT * (1 if self.src.mod == "bpsk" else 2)
            self.frames += 1
        self.buf = self.buf[nfr * FRAME :]
        self.k_buf += nfr * FRAME

    def done(self) -> bool:
        return self.bits >= MAX_BITS or min(self.err.values()) >= MIN_ERR


def ber_scenario() -> tuple[dict[str, Any], Any, np.ndarray]:
    """Lab-like scenario with RX2 removed and RX1's exact optimum as the fixed config."""
    dsg = design("lab", "present")
    d = lab_scenario("lab", "present", initial_states=dsg.s1, seed=SEED)
    d["nodes"] = [nd for nd in d["nodes"] if nd["id"] != "rx2"]
    d["name"] = "ef_ber_lab"
    d["description"] = "E-F BER: lab-like SIM geometry, fixed RIS configuration (RX1 optimum)"
    return d, dsg, dsg.s1


def run_ber_point(
    mod: str, ebn0_db: float, out_dir: str | Path, *, chunk_s: str = "1"
) -> dict[str, Any]:
    """One (modulation, Eb/N0) BER point; writes a run directory."""
    quiet_warnings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d0, dsg, s1 = ber_scenario()
    model = from_scenario(d0, backend="numpy_ref")
    v = csi_from_snapshot(model, model.snapshot(Fraction(0)), f"{TX}->{RX1}")
    h = v.evaluate({"ris0": s1})
    n0 = K_B * T0 * 10 ** (NF_DB / 10)
    bps = 1 if mod == "bpsk" else 2
    p_rx = 10 ** (ebn0_db / 10) * n0 * FS * bps / SPS
    p_tx = p_rx / abs(h) ** 2
    d = copy.deepcopy(d0)
    for nd in d["nodes"]:
        for s in nd["streams"]:
            if s["direction"] == "tx":
                s["full_scale_dbm"] = 10 * math.log10(p_tx / 1e-3)
    d["run"]["duration_s"] = "100"
    d.setdefault("numerics", {})["page_target_s"] = PAGE_TARGET_S
    res = resolve(d)
    run = RunRecord(
        out, SEED, resolved=res, mode="lockstep", coupling="barrier", patch_root=out.parent
    )
    src = FramedSource(TX, SEED, mod)
    # device units (section 3.3): d = y/sqrt(P_FS,rx) with x = sqrt(P_T) * symbol, P_FS,rx = 1 mW
    sink = BerSink(src, h * math.sqrt(p_tx / 1e-3))
    eng = Emulator.from_scenario(res, traffic={TX: src}, sinks=[sink])
    t = Fraction(0)
    step = Fraction(chunk_s)
    t0 = time.perf_counter()
    while not sink.done():
        t += step
        eng.run(t)
        run.sample_load()
    wall = time.perf_counter() - t0
    summ = eng.summary()
    theory = q_ber(ebn0_db)
    row: dict[str, Any] = {
        "modulation": mod,
        "ebn0_db": ebn0_db,
        "bits": sink.bits,
        "frames": sink.frames,
        "theory_ber": theory,
        "tx_dbm": 10 * math.log10(p_tx / 1e-3),
        "p_rx_dbm": 10 * math.log10(p_rx / 1e-3),
        "h_genie_db": db10(abs(h) ** 2),
        "ls_rel_err_rms": float(np.sqrt(np.mean(np.square(sink.h_ls_err)))),
        "virtual_s": float(t),
        "wall_s": wall,
        "rtf": float(t) / wall,
    }
    for name in ("genie", "ls64"):
        k = sink.err[name]
        lo, hi = clopper_pearson(k, sink.bits, 0.95)
        lo99, hi99 = clopper_pearson(k, sink.bits, 0.99)
        row[f"{name}_errors"] = k
        row[f"{name}_ber"] = k / sink.bits
        row[f"{name}_ci95"] = [lo, hi]
        row[f"{name}_ci99"] = [lo99, hi99]
        row[f"{name}_theory_in_ci95"] = bool(lo <= theory <= hi)
    write_json(out / "result.json", row)
    info = eng.manifest_info()
    run.page_size = info["page_size"]
    run.verify_interp_max_rel = info["verify_interp_max_rel"]
    run.counters = {k: v for k, v in info["counters"].items() if isinstance(v, int)}
    run.oracle_csi_used = True
    run.add_assumption("genie detector uses the scene model's narrowband channel (oracle CSI)")
    run.add_assumption("lab |Gamma| ASSUMED; fixed configuration = RX1 exact optimum")
    run.add_assumption(f"numerics.page_target_s = {PAGE_TARGET_S} (performance only, I2b)")
    write_meta(
        out,
        experiment="E-F",
        data_classes=["SIM-sremu", "ANALYTIC", "ASSUMED"],
        description=f"E-F BER point: {mod}, Eb/N0 {ebn0_db} dB, physical noise (NF 7 dB)",
        assumptions=run.assumptions,
        inputs=run.inputs,
        files={"result.json": "errors, bits, BER, Clopper-Pearson intervals, theory"},
    )
    run.finish()
    return row


def run_ber(
    out_root: str | Path, mods: Sequence[str] = ("bpsk", "qpsk"), points: Sequence[float] = EBN0_DB
) -> list[dict[str, Any]]:
    """Every BER point (sequential; each point is its own run directory) + summary CSV."""
    root = Path(out_root)
    rows = []
    for mod in mods:
        for e in points:
            r = run_ber_point(mod, float(e), root / f"{mod}_{int(e):02d}dB")
            print(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "modulation",
                            "ebn0_db",
                            "bits",
                            "genie_errors",
                            "ls64_errors",
                            "theory_ber",
                            "genie_ber",
                            "genie_theory_in_ci95",
                            "rtf",
                        )
                    }
                ),
                flush=True,
            )
            rows.append(r)
    return rows


# =========================================================================== API latency


def _pct(x: Sequence[float]) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64) * 1e6
    return {
        "n": int(a.size),
        "p50_us": float(np.percentile(a, 50)),
        "p95_us": float(np.percentile(a, 95)),
        "p99_us": float(np.percentile(a, 99)),
        "max_us": float(a.max()),
        "mean_us": float(a.mean()),
    }


def latency_scenario() -> dict[str, Any]:
    """The quickstart scene without its tone generator (the client feeds ap.tx0)."""
    from sremu.scenario.load import read_document

    d = copy.deepcopy(read_document(REPO / "scenarios" / "quickstart.yaml"))
    d["name"] = "ef_latency"
    d["traffic"] = []
    d["run"]["duration_s"] = "1000"
    return d


def api_latency_inproc(n_iter: int = 5000, n_warm: int = 500, buf: int = 1000) -> dict[str, Any]:
    """In-process ``send``/``recv`` call latency for ``buf``-sample buffers."""
    from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TxMetadata

    quiet_warnings()
    res = resolve(latency_scenario())
    dev = Device.inproc(res, traffic={})
    x = (np.exp(2j * np.pi * 0.01 * np.arange(buf))).astype(np.complex64)
    y = np.zeros(buf, dtype=np.complex64)
    t_send, t_recv = [], []
    with dev:
        tx = dev.get_tx_stream(StreamArgs("ap.tx0", 3.5e9, "1e6"))
        rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
        rx.issue_stream_cmd(StreamCmd(mode="start_cont", stream_now=True))
        md = TxMetadata(start_of_burst=True)
        for i in range(n_warm + n_iter):
            a = time.perf_counter()
            tx.send(x, md, timeout=1.0)
            b = time.perf_counter()
            got = 0
            while got < buf:
                got += rx.recv(y[got:], RxMetadata(), timeout=1.0)
            c = time.perf_counter()
            md = TxMetadata()
            if i >= n_warm:
                t_send.append(b - a)
                t_recv.append(c - b)
        t0 = time.perf_counter()
        n_sus = 2000
        for _ in range(n_sus):
            tx.send(x, md, timeout=1.0)
            got = 0
            while got < buf:
                got += rx.recv(y[got:], RxMetadata(), timeout=1.0)
        sus = n_sus * buf / (time.perf_counter() - t0) / 1e6
    return {"send": _pct(t_send), "recv": _pct(t_recv), "sustained_msps": sus, "buffer": buf}


def _serve(scen_path: Path) -> tuple[subprocess.Popen[str], dict[str, Any]]:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sremu.commands.serve",
            str(scen_path),
            "--serve",
            "tcp://127.0.0.1:0",
            "--pub",
            "none",
            "--tx",
            "tcp://127.0.0.1:0",
            "--no-paused",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline()
    if not line.startswith("SREMU-SERVE "):
        proc.kill()
        raise RuntimeError(f"server did not start: {line!r}")
    return proc, json.loads(line.split(" ", 1)[1])


def api_latency_zmq(
    scen_path: Path, n_iter: int = 3000, n_warm: int = 300, buf: int = 1000
) -> dict[str, Any]:
    """Native-ZMQ two-process round trip (send ``buf`` samples, receive ``buf`` samples)
    against a served emulator in its own OS process, and the sustained streaming rate."""
    from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TxMetadata

    proc, info = _serve(scen_path)
    try:
        dev = Device.connect(info["endpoints"]["control"])
        x = (np.exp(2j * np.pi * 0.01 * np.arange(buf))).astype(np.complex64)
        y = np.zeros(buf, dtype=np.complex64)
        rtt = []
        with dev:
            tx = dev.get_tx_stream(StreamArgs("ap.tx0", 3.5e9, "1e6"))
            rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
            rx.issue_stream_cmd(StreamCmd(mode="start_cont", stream_now=True))
            md = TxMetadata(start_of_burst=True)
            for i in range(n_warm + n_iter):
                a = time.perf_counter()
                tx.send(x, md, timeout=2.0)
                got = 0
                while got < buf:
                    got += rx.recv(y[got:], RxMetadata(), timeout=2.0)
                c = time.perf_counter()
                md = TxMetadata()
                if i >= n_warm:
                    rtt.append(c - a)
            big = 10_000
            xb = np.tile(x, big // buf)
            yb = np.zeros(big, dtype=np.complex64)
            t0 = time.perf_counter()
            n_sus = 300
            for _ in range(n_sus):
                tx.send(xb, md, timeout=2.0)
                got = 0
                while got < big:
                    got += rx.recv(yb[got:], RxMetadata(), timeout=2.0)
            sus = n_sus * big / (time.perf_counter() - t0) / 1e6
            dev.engine.shutdown() if hasattr(dev.engine, "shutdown") else None
        return {
            "round_trip": _pct(rtt),
            "sustained_msps": sus,
            "buffer": buf,
            "sustained_buffer": big,
        }
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # pragma: no cover
            proc.kill()


def run_latency(out_dir: str | Path) -> dict[str, Any]:
    """API latency study: in-process and two-process, with host load samples."""
    quiet_warnings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = latency_scenario()
    res = resolve(d)
    scen_path = out / "ef_latency.yaml"
    scen_path.write_text(res.to_yaml(), encoding="utf-8", newline="\n")
    run = RunRecord(
        out,
        int(res.scenario.seed),
        resolved=res,
        mode="lockstep",
        coupling="none",
        patch_root=out.parent,
    )
    run.sample_load()
    inproc = api_latency_inproc()
    run.sample_load()
    zmq = api_latency_zmq(scen_path)
    run.sample_load()
    result = {"inproc": inproc, "zmq_two_process": zmq, "host_load": run._load}
    write_json(out / "latency.json", result)
    rows = []
    for name, st in (
        ("inproc_send", inproc["send"]),
        ("inproc_recv", inproc["recv"]),
        ("zmq_round_trip", zmq["round_trip"]),
    ):
        rows.append({"metric": name, **st})
    rows.append(
        {"metric": "inproc_sustained_msps", "mean_us": None, "p50_us": inproc["sustained_msps"]}
    )
    rows.append({"metric": "zmq_sustained_msps", "mean_us": None, "p50_us": zmq["sustained_msps"]})
    write_csv(out / "latency.csv", rows)
    run.add_assumption(
        "wall-clock measurements on a shared, loaded Windows laptop (see host_load_samples)"
    )
    write_meta(
        out,
        experiment="E-F",
        data_classes=["MEAS"],
        description="E-F API latency: in-process send/recv (1000-sample buffers), native-ZMQ "
        "two-process round trip, sustained MS/s; this laptop only",
        assumptions=run.assumptions,
        inputs=run.inputs,
        files={"latency.json": "percentiles and sustained rates", "latency.csv": "flat table"},
    )
    run.finish()
    return result


def aggregate_ber(root: str | Path, results_dir: str | Path) -> list[dict[str, Any]]:
    """BER table (F5c data) from the point run directories under ``root``."""
    root, out = Path(root), Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = RunRecord(out, SEED, mode="analysis", patch_root=out)
    rows = []
    for p in sorted(root.glob("*_*dB/result.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        rows.append(
            {
                "modulation": r["modulation"],
                "ebn0_db": r["ebn0_db"],
                "bits": r["bits"],
                "theory_ber": r["theory_ber"],
                "genie_errors": r["genie_errors"],
                "genie_ber": r["genie_ber"],
                "genie_ci95_low": r["genie_ci95"][0],
                "genie_ci95_high": r["genie_ci95"][1],
                "genie_theory_in_ci95": r["genie_theory_in_ci95"],
                "ls64_errors": r["ls64_errors"],
                "ls64_ber": r["ls64_ber"],
                "ls64_ci95_low": r["ls64_ci95"][0],
                "ls64_ci95_high": r["ls64_ci95"][1],
                "ls64_theory_in_ci95": r["ls64_theory_in_ci95"],
                "tx_dbm": r["tx_dbm"],
                "rtf": r["rtf"],
            }
        )
    rows.sort(key=lambda x: (x["modulation"], x["ebn0_db"]))
    write_csv(out / "ber.csv", rows)
    write_meta(
        out,
        experiment="E-F",
        data_classes=["SIM-sremu", "ANALYTIC", "ASSUMED"],
        description="E-F BER vs Eb/N0 through the emulator (physical noise), genie and 64-pilot LS "
        "detection, Clopper-Pearson 95 % intervals, theory Q(sqrt(2 Eb/N0)) (F5c data)",
        assumptions=[
            "genie = scene-model channel (oracle CSI)",
            "lab |Gamma| ASSUMED",
            ">= 200 errors or 2e7 data bits per point (both detectors)",
        ],
        files={"ber.csv": "one row per (modulation, Eb/N0)"},
        extra={"root": str(root)},
    )
    rec.finish()
    return rows
