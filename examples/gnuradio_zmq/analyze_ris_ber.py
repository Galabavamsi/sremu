"""Analysis of one GNU Radio run of ``run_ris_ber.py`` (sremu interpreter; numpy, scipy).

    python examples/gnuradio_zmq/analyze_ris_ber.py RUN_DIR [RUN_DIR ...]

Reads what GNU Radio wrote (``gr_*``), the emulator's run directory (``rec/*``) and the
runner's ``run.json``; writes ``RUN_DIR/analysis.json`` and prints a summary line.

Per segment (a static run has one; the in-run switching run one per RIS configuration):

* **SER**: GNU Radio's decided symbols against the transmitted data symbols, aligned by a
  circular correlation over the symbol period; the receiver's acquisition after the start
  (``SKIP_START_S``) or after a switch (``SKIP_SWITCH_S``) is not scored;
* **BLER (uncoded 200-bit blocks)**: the fraction of consecutive 100-symbol windows of the
  decision stream that hold at least one symbol error. There is no coding and no CRC: this
  is a re-expression of the SER stream, not a packet error rate;
* **MER**: decision-directed, from the Costas-loop output;
* **LS gain and SNR**: least-squares fit ``y = h x`` of the IQ GNU Radio received (``y``)
  against the transmitted IQ ``x``, regenerated from the symbol file and the filter taps
  (GNU Radio's own TX chain differs from the regeneration by float32 rounding only); the SNR
  is estimated from the received IQ (residual power), it is not an emulator setting;
* **theory**: coherent QPSK with differential decoding at ``Es/N0 = SNR + 10 log10(sps)``
  and, where the run has at least ``MIN_ERRORS`` symbol errors, the implementation loss
  implied by the measured SER.

Per switch (switching run): the maximum-likelihood change point of the two-segment LS model
``y = h1 x`` before and ``y = h2 x`` after, searched over the whole window around the
switch, against the command record's ``k_applied``.

Run checks: GNU Radio received every sample of the run; its received IQ equals, bit for bit,
the RX stream the emulator recorded itself (SigMF), so every sample crossed the raw REQ/REP
plane once and in order; every manifest counter is zero. sha256 of the received IQ, of the
Costas output and of the decisions, for comparing independent runs. Throughput: the
emulator's real-time factor (virtual seconds per wall second of the served lockstep engine,
which includes waiting for GNU Radio) per 0.1 s window of ``metrics.jsonl``, and GNU Radio's
own RX progress; lockstep runs are virtual time, so neither is a real-time claim.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SKIP_START_S = 0.02
SKIP_SWITCH_S = 0.03
SKIP_END_S = 0.01
LS_SKIP_S = 0.01
BLOCK_SYMBOLS = 100  # 200 bits of DQPSK
MIN_ERRORS = 100
WORST_WINDOW_S = 0.02
CP_EDGE = 1000  # change point search: samples kept on each side
CP_HALF_WINDOW = 20_000


# --------------------------------------------------------------------------- signals
def regenerate_tx(symbols: np.ndarray, taps: np.ndarray, sps: int, n: int) -> np.ndarray:
    """The TX IQ of ``gr_dqpsk_modem.py``: repeated data symbols, differential encoding
    (``out = (in + previous out) mod 4``, zero initial state), QPSK points
    ``exp(j (pi/4 + k pi/2))``, zero-stuffed by ``sps`` and filtered with ``taps`` from a
    zero state (GNU Radio's ``interp_fir_filter_ccf``); the first ``n`` samples."""
    need = -(-n // sps) + taps.size
    reps = -(-need // symbols.size)
    d = np.cumsum(np.tile(symbols.astype(np.int64), reps)[:need]) % 4
    up = np.zeros(need * sps, np.complex128)
    up[::sps] = np.exp(1j * (np.pi / 4 + np.pi / 2 * d))
    return np.convolve(up, taps.astype(np.float64))[:n]


def align(
    dec: np.ndarray, symbols: np.ndarray, anchors: list[int], length: int = 4096
) -> tuple[int, float]:
    """Lag ``L`` with ``dec[i] == symbols[(i + L) mod Ns]`` from the best circular
    correlation of ``exp(j pi/2 s)`` over ``length`` decisions at any of ``anchors``;
    returns (lag, normalized correlation peak)."""
    ns = symbols.size
    z = np.fft.fft(np.exp(1j * np.pi / 2 * symbols.astype(float)))
    best: tuple[float, int] | None = None
    for a0 in anchors:
        a0 = max(0, min(int(a0), dec.size - length))
        w = np.exp(1j * np.pi / 2 * dec[a0 : a0 + length].astype(float))
        cc = np.abs(np.fft.ifft(z * np.conj(np.fft.fft(w, ns)))) / length
        i = int(np.argmax(cc))
        if best is None or cc[i] > best[0]:
            best = (float(cc[i]), (i - a0) % ns)
    assert best is not None
    return best[1], best[0]


def symbol_errors(dec: np.ndarray, symbols: np.ndarray, lag: int, i0: int, i1: int) -> np.ndarray:
    """Boolean error vector of decisions ``i0:i1``."""
    idx = (np.arange(i0, i1) + lag) % symbols.size
    return np.asarray(dec[i0:i1] != symbols[idx])


def bler(err: np.ndarray, block: int = BLOCK_SYMBOLS) -> tuple[float | None, int]:
    """Fraction of consecutive ``block``-symbol windows with an error, and their number."""
    nb = err.size // block
    if nb == 0:
        return None, 0
    return float(np.mean(err[: nb * block].reshape(nb, block).any(axis=1))), nb


def mer_db(costas: np.ndarray) -> float:
    """Decision-directed modulation error ratio of QPSK points (any amplitude), dB."""
    c = costas.astype(np.complex128)
    amp = np.sqrt(np.mean(np.abs(c) ** 2))
    pts = amp * np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    d = pts[np.argmin(np.abs(c[:, None] - pts[None, :]), axis=1)]
    return float(10 * np.log10(np.mean(np.abs(d) ** 2) / np.mean(np.abs(c - d) ** 2)))


def ls_fit(
    x: np.ndarray, y: np.ndarray, lo: int, hi: int, delays: range = range(-2, 9)
) -> dict[str, float]:
    """LS gain ``h`` of ``y[lo:hi] = h x[lo-D:hi-D]`` for the integer delay D with the best
    fit; gain (dB), phase (deg), SNR estimate (dB), D."""
    best: dict[str, float] | None = None
    for dl in delays:
        if lo - dl < 0 or hi - dl > x.size:
            continue
        xs, ys = x[lo - dl : hi - dl], y[lo:hi]
        h = np.vdot(xs, ys) / np.vdot(xs, xs)
        res = ys - h * xs
        snr = 10 * np.log10(np.mean(np.abs(h * xs) ** 2) / np.mean(np.abs(res) ** 2))
        if best is None or snr > best["snr_db"]:
            best = {
                "gain_db": float(20 * np.log10(abs(h))),
                "phase_deg": float(np.degrees(np.angle(h))),
                "snr_db": float(snr),
                "delay_samples": dl,
            }
    assert best is not None
    return best


def ml_change_point(x: np.ndarray, y: np.ndarray, lo: int, hi: int, edge: int = CP_EDGE) -> int:
    """Maximum-likelihood change point of ``y = h1 x`` on ``[lo, c)`` and ``y = h2 x`` on
    ``[c, hi)`` (complex Gaussian noise of one variance): the ``c`` in
    ``[lo + edge, hi - edge]`` that minimizes the summed LS residuals, i.e. maximizes
    ``|S_xy(lo, c)|^2 / S_xx(lo, c) + |S_xy(c, hi)|^2 / S_xx(c, hi)``."""
    xs, ys = x[lo:hi], y[lo:hi]
    sxx = np.concatenate([[0.0], np.cumsum(np.abs(xs) ** 2)])
    sxy = np.concatenate([[0.0], np.cumsum(np.conj(xs) * ys)])
    c = np.arange(edge, xs.size - edge + 1)
    a = np.abs(sxy[c]) ** 2 / sxx[c]
    b = np.abs(sxy[-1] - sxy[c]) ** 2 / (sxx[-1] - sxx[c])
    return lo + int(c[int(np.argmax(a + b))])


def q_func(v: float) -> float:
    return 0.5 * math.erfc(v / math.sqrt(2.0))


def ser_dqpsk_theory(es_n0_db: float) -> float:
    """SER of coherent Gray QPSK followed by differential decoding at Es/N0 (dB): a decided
    symbol is right when both coherent symbols are right or both rotated alike."""
    p = q_func(math.sqrt(10 ** (es_n0_db / 10)))
    pc, pr = (1 - p) ** 2, p * (1 - p)
    return 1.0 - (pc**2 + 2 * pr**2 + p**4)


def implied_loss_db(ser: float, es_n0_db: float) -> float | None:
    """Es/N0 the theory needs for ``ser``, subtracted from ``es_n0_db``."""
    from scipy.optimize import brentq

    if not 0.0 < ser < 0.7:
        return None
    need = brentq(lambda v: ser_dqpsk_theory(v) - ser, -30.0, 40.0)
    return float(es_n0_db - need)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- one run
def _rec_dir(run: Path) -> Path | None:
    mf = sorted(glob.glob(str(run / "rec" / "*" / "manifest.json")))
    return Path(mf[0]).parent if mf else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def throughput(rec: Path | None, progress: Path, rate: float) -> dict[str, Any]:
    """Real-time factors of the run (virtual time per wall time; lockstep)."""
    out: dict[str, Any] = {}
    if rec is not None:
        rtf = [float(m["rtf"]) for m in _jsonl(rec / "metrics.jsonl") if m.get("rtf")]
        if len(rtf) > 1:
            out["sremu_rtf_median"] = float(np.median(rtf[1:]))  # the first window holds JIT
            out["sremu_rtf_windows"] = len(rtf) - 1
    if progress.is_file():
        p = np.loadtxt(progress, delimiter=",", skiprows=1, ndmin=2)
        p = p[p[:, 1] > 0]
        if p.shape[0] > 3:
            last = p[:, 1].max()
            p = p[: int(np.argmax(p[:, 1] >= last)) + 1]
            sel = p[p[:, 1] >= 0.2 * last]
            if sel.shape[0] > 1 and sel[-1, 0] > sel[0, 0]:
                out["gr_rtf_steady"] = float(
                    (sel[-1, 1] - sel[0, 1]) / rate / (sel[-1, 0] - sel[0, 0])
                )
            out["gr_wall_first_to_last_s"] = float(p[-1, 0] - p[0, 0])
    return out


def analyze_run(run: str | Path) -> dict[str, Any]:
    """Analyse one run directory; writes and returns ``analysis.json``."""
    run = Path(run)
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    modem = json.loads((run / "gr_modem.json").read_text(encoding="utf-8"))
    rate = float(meta["rate_sps"])
    sps = int(modem["sps"])
    fsym = rate / sps
    n_run = int(meta["n_run"])
    y = np.fromfile(run / "gr_rx.cf32", np.complex64)
    dec = np.fromfile(run / "gr_dec.u8", np.uint8)
    costas = np.fromfile(run / "gr_costas.cf32", np.complex64)
    sym = np.fromfile(run / "gr_tx_symbols.u8", np.uint8)
    taps = np.asarray(modem["rrc_tx_taps"], float)
    x = regenerate_tx(sym, taps, sps, y.size)
    yc = y.astype(np.complex128)
    rec = _rec_dir(run)
    out: dict[str, Any] = {
        "run": run.name,
        "kind": meta["kind"],
        "rate_sps": rate,
        "sps": sps,
        "n_run": n_run,
        "rx_samples": int(y.size),
        "complete": int(y.size) == n_run,
        "decisions": int(dec.size),
        "gnuradio_version": modem.get("gnuradio_version"),
        "sha256": {
            "rx_iq": sha256_file(run / "gr_rx.cf32"),
            "costas": sha256_file(run / "gr_costas.cf32"),
            "decisions": sha256_file(run / "gr_dec.u8"),
        },
    }
    # --- the emulator's own record of the run
    events = _jsonl(rec / "events.jsonl") if rec else []
    cmds = sorted(
        (e for e in events if e.get("op") == "ris.set_config"),
        key=lambda e: int((e.get("k_applied") or {}).get(meta["rx_stream"], 0)),
    )
    rx_id = meta["rx_stream"]
    out["commands"] = [
        {
            "cmd_id": e.get("cmd_id"),
            "t_applied": e.get("t_applied"),
            "k_req": (e.get("k_req") or {}).get(rx_id),
            "k_applied": (e.get("k_applied") or {}).get(rx_id),
            "lateness_samples": (e.get("lateness_samples") or {}).get(rx_id),
            "status": e.get("status"),
        }
        for e in cmds
    ]
    if rec is not None:
        man = json.loads((rec / "manifest.json").read_text(encoding="utf-8"))
        out["counters_nonzero"] = {k: v for k, v in man.get("counters", {}).items() if v}
        out["manifest"] = {
            k: man.get(k)
            for k in (
                "mode",
                "backend",
                "threads",
                "deterministic",
                "nondeterministic_inputs",
                "page_size",
                "git_commit",
                "git_dirty",
                "wall_s",
                "sremu_version",
                "cpu_model",
                "platform",
            )
        }
        rec_iq = sorted(rec.rglob(f"{rx_id}*.sigmf-data"))
        if rec_iq:
            r = np.fromfile(rec_iq[0], np.complex64)
            out["rx_equals_emulator_record"] = bool(
                r.size >= y.size and np.array_equal(r[: y.size].view(np.uint64), y.view(np.uint64))
            )
            out["emulator_record_samples"] = int(r.size)
    # --- segments: static = whole run; switching = between the applied switches
    names = list(meta["configs"])
    starts = [0] + [int(c["k_applied"]) for c in out["commands"] if meta["kind"] == "switching"]
    if meta["kind"] == "static":
        starts = [0]
    bounds = list(zip(starts, [*starts[1:], y.size], strict=True))
    anchors = [int(SKIP_START_S * fsym), dec.size // 2, max(0, dec.size - 5000)]
    g_lag, g_peak = align(dec, sym, anchors)
    segs = []
    for name, (k0, k1) in zip(names, bounds, strict=False):
        skip = SKIP_START_S if k0 == 0 else SKIP_SWITCH_S
        i0 = int((k0 / rate + skip) * fsym)
        i1 = min(dec.size, int((k1 / rate - SKIP_END_S) * fsym)) if k1 < y.size else dec.size
        lag, peak = (
            align(dec[:i1], sym, [i0 + (i1 - i0) // 2 - 2048])
            if i1 - i0 > 8192
            else (g_lag, g_peak)
        )
        if peak < 0.5:  # link lost in this segment: keep the run's alignment
            lag = g_lag
        err = symbol_errors(dec, sym, lag, i0, i1)
        b, nb = bler(err)
        win = int(WORST_WINDOW_S * fsym)
        worst = max(
            (float(err[j : j + win].mean()) for j in range(0, err.size - win + 1, win)),
            default=None,
        )
        lo = k0 + int(LS_SKIP_S * rate)
        hi = min(k1 - int(LS_SKIP_S * rate) if k1 < y.size else y.size - 64, y.size - 64)
        ls = ls_fit(x, yc, lo, hi)
        es = ls["snr_db"] + 10 * math.log10(sps)
        n_err = int(err.sum())
        seg = {
            "config": name,
            "k0": k0,
            "k1": k1,
            "symbols": int(err.size),
            "symbol_errors": n_err,
            "ser": float(err.mean()),
            "bler_uncoded_200bit": b,
            "blocks": nb,
            "worst_20ms_ser": worst,
            "mer_db": mer_db(costas[i0:i1]),
            "ls_gain_db": ls["gain_db"],
            "ls_phase_deg": ls["phase_deg"],
            "ls_snr_db": ls["snr_db"],
            "ls_delay_samples": ls["delay_samples"],
            "es_n0_db": es,
            "ser_theory": ser_dqpsk_theory(es),
            "implied_loss_db": implied_loss_db(float(err.mean()), es)
            if n_err >= MIN_ERRORS
            else None,
            "lag": lag,
            "corr_peak": peak,
        }
        if "af_pred_db" in meta:
            seg["af_pred_db"] = meta["af_pred_db"].get(name)
        segs.append(seg)
    out["segments"] = segs
    # --- switch timing: ML change point around every applied switch
    if meta["kind"] == "switching":
        cps = []
        for c in out["commands"]:
            k = int(c["k_applied"])
            half = min(CP_HALF_WINDOW, int(0.45 * float(meta["segment_s"]) * rate))
            cp = ml_change_point(x, yc, k - half, k + half)
            cps.append(
                {
                    "k_applied": k,
                    "ml_change_point": cp,
                    "offset": cp - k,
                    "window": [k - half, k + half],
                }
            )
        out["change_points"] = cps
    out["throughput"] = throughput(rec, run / "gr_progress.csv", rate)
    (run / "analysis.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return out


def summary_line(a: dict[str, Any]) -> str:
    segs = "; ".join(
        f"{s['config']}: SER {s['ser']:.3g}, BLER(200b) {s['bler_uncoded_200bit']}, "
        f"MER {s['mer_db']:.2f} dB, LS {s['ls_gain_db']:.2f} dB, SNR {s['ls_snr_db']:.2f} dB"
        for s in a["segments"]
    )
    cps = a.get("change_points")
    cp = f" | change points {[c['offset'] for c in cps]}" if cps else ""
    return (
        f"{a['run']}: complete {a['complete']}, counters {a.get('counters_nonzero')}, "
        f"rx==record {a.get('rx_equals_emulator_record')} | {segs}{cp}"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    for r in args:
        print(summary_line(analyze_run(r)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
