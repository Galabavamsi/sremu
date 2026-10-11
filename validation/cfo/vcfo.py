"""V-CFO: receiver-LO (CFO) verification (DESIGN.md section 10.10; RQ-35) and the noisy
study V-CFO-4.

Setup (the BPSK test burst of section 10.10; internally a port of the earlier simulator's
pair, section 8.2): a 200-symbol header of 1+1j, 38 BPSK symbols
at +-0.1(1+j), 10 samples/symbol, 2389 samples at 5880 S/s (``bpsk_v1_waveform``), through
the in-process lockstep engine with the WP2 physics model of a static SIM link with a RIS
(16 x 16 cells, 2.4 GHz - 3050 Hz, the v1 carrier). The RX LO is offset by -f, so the
received baseband rotates at +f.

* **Noise-free sweeps** (f = -1400 ... +1400 Hz, 100 Hz steps; the F5a data): the new
  engine and the ``LegacyTruncationAdapter`` (v1 time base, ``int(f_s tau)`` samples per
  2 ms tick) applied to the engine's CFO-free output, each with the unwrapped-phase LS
  estimator on header samples [30, 1950) (as V-CFO-1/2) and the lag-1 estimator; slope,
  intercept and RMSE of (estimate - actual). The DIG points of ANTS Fig. 2 are re-analysed
  the same way.
* **V-CFO-4** (noise from the engine's physical noise, kT0 F f_s per RX with F = 0 dB):
  per-sample SNR in {0, 10, 20} dB on the header, set through the TX full scale from the
  model's |h|; 200 independently keyed trials (purpose ``trial``) at each of 7 CFO points;
  estimators on the N = 2000 header samples: periodogram (zero-padded FFT) + parabolic
  peak interpolation, lag-1 and unwrapped-phase LS. RMSE vs the CRB
  ``sigma_f = f_s/(2 pi) sqrt(6/(SNR N (N^2 - 1)))`` (complex tone, unknown amplitude and
  phase; Rife & Boorstyn 1974). Acceptance: periodogram RMSE <= 1.5 x CRB at 10 and 20 dB
  at every CFO point; 0 dB reported.

**Paper use (DESIGN D24, fresh-attempt framing).** The paper shows no v1/v2 comparison and
no before/after framing. Paper-facing: the new engine's sweeps and V-CFO-4 (F5a/b), and the
tick-truncation effect only as a generic design lesson (integer samples per tick -> 6.9 %
frequency scaling), computed by ``LegacyTruncationAdapter`` without naming v1. The ANTS
Fig. 2 DIG points (an output of the earlier simulator) and the V-CFO-2 slope regression stay
internal (traceability and teeth tests); ``meta.json`` carries a ``paper_use`` label per
series.

Usage::

    python validation/cfo/vcfo.py [--jobs 12] [--fast]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import warnings
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import vstudy as vs  # noqa: E402

from sremu.engine.emulator import EngineSink  # noqa: E402
from sremu.units import K_B, T0  # noqa: E402

STUDY_DIR = _HERE
DATA = vs.DATA_VALIDATION / "cfo"
FS = Fraction(5880)
TAU_V1 = Fraction(1, 500)
FC = 2.4e9 - 3050.0  # v1 carrier (audit_v1 section 10.1)
N_WAVE = 2389
HEADER = (0, 2000)  # N = 2000 header samples (V-CFO-4, CRB)
LS_WIN = (30, 1950)  # V-CFO-1/2 estimator window
SWEEP = np.arange(-1400, 1401, 100, dtype=np.float64)
CFO_POINTS = (-1200.0, -800.0, -400.0, 0.0, 400.0, 800.0, 1200.0)  # [ENG] 7 points
SNRS_DB = (0.0, 10.0, 20.0)
N_TRIALS = 200
MASTER_SEED = 20261010
TX = "tx.tx0"
RX = "rx.rx0"
LINK = f"{TX}->{RX}"
NFFT = 1 << 18


def scenario(seed: int = MASTER_SEED, cfo_hz: float = 0.0, full_scale_dbm: float = 0.0) -> dict:
    """Static SIM link with a RIS (16 x 16 cells of 0.06 m < lambda/2, point model)."""
    return {
        "schema_version": "1.0",
        "name": "vcfo_link",
        "seed": int(seed),
        "description": "V-CFO SIM link: TX (5,2,5), RX (15,8,5), RIS at (10,0,5) facing +y",
        "numerics": {"backend": "numba_cpu"},
        "nodes": [
            {
                "id": "tx",
                "position_m": [5.0, 2.0, 5.0],
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": "tx0",
                        "direction": "tx",
                        "fc_hz": FC,
                        "rate_sps": "5880",
                        "occupied_bw_hz": 1176.0,
                        "full_scale_dbm": float(full_scale_dbm),
                    }
                ],
            },
            {
                "id": "rx",
                "position_m": [15.0, 8.0, 5.0],
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": "rx0",
                        "direction": "rx",
                        "fc_hz": FC,
                        "rate_sps": "5880",
                        "noise_figure_db": 0.0,
                        "lo": {"cfo_hz": float(cfo_hz)},
                    }
                ],
            },
        ],
        "ris": [
            {
                "id": "ris0",
                "pose": {"center_m": [10.0, 0.0, 5.0], "normal": [0, 1, 0], "u_axis": [1, 0, 0]},
                "grid": {
                    "n_u": 16,
                    "n_v": 16,
                    "pitch_u_m": 0.06,
                    "pitch_v_m": 0.06,
                    "cell_model": "auto",
                },
                "states": {"library": "ideal_antipodal"},
                "initial": {"fill": 0},
            }
        ],
        "run": {"mode": "lockstep", "coupling": "barrier", "duration_s": "2389/5880"},
        "traffic": [{"stream": TX, "generator": "bpsk", "params": {}}],
    }


class _Collect(EngineSink):
    def __init__(self) -> None:
        self.parts: list[tuple[int, np.ndarray]] = []

    def on_rx(self, stream: str, k0: int, d: np.ndarray) -> None:
        if stream == RX:
            self.parts.append((k0, np.array(d, copy=True)))

    def samples(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.complex64)
        for k0, d in self.parts:
            a, b = max(k0, 0), min(k0 + d.size, n)
            if b > a:
                out[a:b] = d[a - k0 : b - k0]
        return out


class Link:
    """The scenario's model built once; engines per trial (seed, CFO, TX full scale)."""

    def __init__(self) -> None:
        from sremu.physics.propagation import from_scenario

        self.base = scenario()
        res = vs.resolve_mapping(self.base)
        self.d = res.scenario.model_dump(mode="python")
        self.model = from_scenario(self.d)
        snap = self.model.snapshot(Fraction(0))
        cfg = {"ris0": self.model.initial_config("ris0")}
        self.h = complex(self.model.evaluate(snap, cfg)[LINK])

    def full_scale_for_snr(self, snr_db: float) -> float:
        """TX full scale (dBm) giving per-sample SNR ``snr_db`` on the header (|x|^2 = 2):
        ``2 P_FS |h|^2 / (k_B T0 F f_s)`` with F = 0 dB (noise in mW)."""
        n_mw = K_B * T0 * float(FS) * 1e3
        return snr_db + 10 * math.log10(n_mw) - 10 * math.log10(2.0 * abs(self.h) ** 2)

    def receive(
        self, *, seed: int, cfo_hz: float, full_scale_dbm: float, noise: bool
    ) -> np.ndarray:
        """RX samples [0, 2389) of one engine run."""
        from sremu.engine.emulator import Emulator, EngineConfig
        from sremu.engine.streams import ArraySource, bpsk_v1_waveform, stream_configs_from_scenario

        d = dict(self.d)
        d["nodes"] = [dict(n) for n in d["nodes"]]
        d["nodes"][0]["streams"] = [
            dict(d["nodes"][0]["streams"][0], full_scale_dbm=full_scale_dbm)
        ]
        rx = dict(d["nodes"][1]["streams"][0])
        rx["lo"] = dict(rx.get("lo") or {}, cfo_hz=float(cfo_hz))
        d["nodes"][1]["streams"] = [rx]
        cfgs = stream_configs_from_scenario(d)
        sink = _Collect()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eng = Emulator(
                self.model,
                streams=cfgs,
                config=EngineConfig(
                    seed=int(seed), noise=noise, always_active=True, backend="numba_cpu"
                ),
                traffic={TX: ArraySource(bpsk_v1_waveform())},
                sinks=[sink],
            )
            eng.run(Fraction(N_WAVE) / FS)
        return sink.samples(N_WAVE)


# =========================================================================== estimators


def est_ls(y: np.ndarray, win: tuple[int, int] = LS_WIN) -> float:
    """Unwrapped-phase least squares (V-CFO-1/2 estimator), Hz."""
    seg = np.asarray(y[win[0] : win[1]], dtype=np.complex128)
    ph = np.unwrap(np.angle(seg))
    n = np.arange(seg.size, dtype=np.float64)
    return float(np.polyfit(n, ph, 1)[0] * float(FS) / (2 * np.pi))


def est_lag1(y: np.ndarray, win: tuple[int, int] = LS_WIN) -> float:
    """Lag-1 autocorrelation phase, Hz."""
    seg = np.asarray(y[win[0] : win[1]], dtype=np.complex128)
    return float(np.angle(np.sum(seg[1:] * np.conj(seg[:-1]))) * float(FS) / (2 * np.pi))


def est_periodogram(y: np.ndarray, win: tuple[int, int] = HEADER, nfft: int = NFFT) -> float:
    """Zero-padded periodogram peak with parabolic interpolation of |Y|^2 over 3 bins, Hz."""
    seg = np.asarray(y[win[0] : win[1]], dtype=np.complex128)
    p = np.abs(np.fft.fft(seg, nfft)) ** 2
    k = int(np.argmax(p))
    a, b, c = p[(k - 1) % nfft], p[k], p[(k + 1) % nfft]
    den = a - 2 * b + c
    delta = 0.5 * (a - c) / den if den != 0 else 0.0
    f = (k + delta) * float(FS) / nfft
    return float(f - float(FS) if f >= float(FS) / 2 else f)


def crb_std_hz(snr_db: float, n: int = HEADER[1] - HEADER[0], fs: float = float(FS)) -> float:
    """CRB standard deviation of the frequency of a complex tone in complex AWGN (unknown
    amplitude and phase): ``fs/(2 pi) sqrt(6/(SNR N (N^2-1)))``."""
    snr = 10 ** (snr_db / 10)
    return fs / (2 * math.pi) * math.sqrt(6.0 / (snr * n * (n * n - 1)))


def lag1_std_hz(snr_db: float, n: int = HEADER[1] - HEADER[0], fs: float = float(FS)) -> float:
    """Small-error standard deviation of the lag-1 estimator on ``n`` samples of a complex
    tone in complex AWGN [DERIVED here]: with ``R = sum z_n z*_{n-1}``, the signal-noise
    cross terms cancel in Im(R e^{-j omega}) except at the two ends, so
    ``var(omega_hat) = 1/(SNR (n-1)^2) + 1/(2 SNR^2 (n-1))``; ``f = fs omega/(2 pi)``.
    The lag-1 estimator is not efficient for large n (its variance falls as 1/n, the CRB's
    as 1/n^3); it is reported against the CRB, not gated."""
    snr = 10 ** (snr_db / 10)
    var = 1.0 / (snr * (n - 1) ** 2) + 1.0 / (2.0 * snr * snr * (n - 1))
    return fs / (2 * math.pi) * math.sqrt(var)


def fit(actual: np.ndarray, est: np.ndarray) -> dict[str, float]:
    """Slope, intercept, RMSE of (est - actual), residual RMSE after the linear fit."""
    a = np.asarray(actual, np.float64)
    e = np.asarray(est, np.float64)
    slope, icpt = np.polyfit(a, e, 1)
    res = e - (slope * a + icpt)
    return {
        "slope": float(slope),
        "intercept_hz": float(icpt),
        "rmse_hz": float(np.sqrt(np.mean((e - a) ** 2))),
        "mean_err_hz": float(np.mean(e - a)),
        "max_abs_err_hz": float(np.max(np.abs(e - a))),
        "rmse_after_fit_hz": float(np.sqrt(np.mean(res**2))),
        "n": int(a.size),
    }


# =========================================================================== V-CFO-4 runner


_LINK: Link | None = None


def _link() -> Link:
    global _LINK
    if _LINK is None:
        _LINK = Link()
    return _LINK


def trial_seed(snr_db: float, cfo_hz: float, i: int) -> int:
    """Engine master seed of one trial: keyed purpose ``trial``, name
    ``vcfo4/snr<..>/f<..>/<i>`` (section 6.1)."""
    from sremu import rng

    g = rng.generator(MASTER_SEED, "trial", f"vcfo4/snr{snr_db:g}/f{cfo_hz:g}/{i}")
    return int(g.integers(0, 2**62))


def run_point(ctx: Any) -> dict[str, Any]:
    """Sweep runner (``vcfo:run_point``): ``n_trials`` noisy trials at one (SNR, CFO)."""
    p = ctx.params
    snr = float(p["snr_db"])
    f = float(p["cfo_hz"])
    n_trials = int(p["n_trials"])
    lk = _link()
    fs_dbm = lk.full_scale_for_snr(snr)
    t0 = time.perf_counter()
    rows = []
    n0, n1 = HEADER
    hdr = np.full(n1 - n0, 1 + 1j)
    for i in range(n_trials):
        seed = trial_seed(snr, f, i)
        y = lk.receive(seed=seed, cfo_hz=-f, full_scale_dbm=fs_dbm, noise=True)
        seg = y[n0:n1].astype(np.complex128)
        # realised per-sample SNR: signal = LS fit of the known rotating header
        k = np.arange(n0, n1)
        ref = hdr * np.exp(2j * np.pi * f * k / float(FS))
        a = np.vdot(ref, seg) / np.vdot(ref, ref)
        resid = seg - a * ref
        snr_hat = (abs(a) ** 2 * 2.0) / float(np.mean(np.abs(resid) ** 2))
        rows.append(
            {
                "trial": i,
                "seed": seed,
                "periodogram_hz": est_periodogram(y),
                "lag1_hz": est_lag1(y, HEADER),
                "ls_hz": est_ls(y, HEADER),
                "snr_hat_db": 10 * math.log10(snr_hat),
            }
        )
    vs.write_csv(ctx.run_dir / "trials.csv", rows)
    out: dict[str, Any] = {
        "snr_db": snr,
        "cfo_hz": f,
        "n_trials": n_trials,
        "tx_full_scale_dbm": fs_dbm,
        "wall_s": time.perf_counter() - t0,
        "crb_std_hz": crb_std_hz(snr),
        "snr_hat_db_mean": float(np.mean([r["snr_hat_db"] for r in rows])),
    }
    for est in ("periodogram", "lag1", "ls"):
        e = np.array([r[f"{est}_hz"] for r in rows]) - f
        out[f"{est}_bias_hz"] = float(np.mean(e))
        out[f"{est}_sd_hz"] = float(np.std(e, ddof=1))
        out[f"{est}_rmse_hz"] = float(np.sqrt(np.mean(e * e)))
        out[f"{est}_sum_sq_hz2"] = float(np.sum(e * e))
        out[f"{est}_rmse_over_crb"] = out[f"{est}_rmse_hz"] / out["crb_std_hz"]
    return out


# =========================================================================== study


def _check_data() -> list[tuple[Path, str]]:
    sums = dict(line.split()[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines())
    files = []
    for name in ("fig2_cfo_estimated_vs_actual.csv", "fig2_cfo_stats.json"):
        got = vs.sha256_file(DATA / name)
        if got != sums[name]:
            raise SystemExit(f"{name}: sha256 {got} != SHA256SUMS {sums[name]}")
        files.append((DATA / name, "DIG"))
    return files


def noise_free(lk: Link) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """V-CFO-1/2 figure data: engine and legacy sweeps (noise off) and the DIG re-analysis."""
    from sremu.legacy.truncation import LegacyTruncationAdapter

    rows: list[dict[str, Any]] = []
    y0 = lk.receive(seed=MASTER_SEED, cfo_hz=0.0, full_scale_dbm=0.0, noise=False)
    ad = LegacyTruncationAdapter(FS, TAU_V1, allow_nonphysical=True)
    with (DATA / "fig2_cfo_estimated_vs_actual.csv").open(encoding="utf-8") as fh:
        dig = list(csv.DictReader(fh))
    dig_by = {float(r["actual_cfo_hz_nominal"]): r for r in dig}
    est = {k: [] for k in ("eng_ls", "eng_lag1", "eng_pgram", "leg_ls", "leg_lag1")}
    for f in SWEEP:
        y = lk.receive(seed=MASTER_SEED, cfo_hz=-float(f), full_scale_dbm=0.0, noise=False)
        yv1 = ad.deliver(y0, float(f))
        r = {
            "actual_cfo_hz": float(f),
            "engine_ls_hz": est_ls(y),
            "engine_lag1_hz": est_lag1(y),
            "engine_periodogram_hz": est_periodogram(y),
            "engine_samples": int(y.size),
            "legacy_ls_hz": est_ls(yv1),
            "legacy_lag1_hz": est_lag1(yv1),
            "legacy_samples": int(yv1.size),
        }
        d = dig_by.get(float(f))
        if d is not None:
            r["dig_actual_hz"] = float(d["actual_cfo_hz_digitized"])
            r["dig_estimated_hz"] = float(d["estimated_cfo_hz"])
            r["dig_unc_hz"] = float(d["estimated_cfo_unc_hz"])
        rows.append(r)
        est["eng_ls"].append(r["engine_ls_hz"])
        est["eng_lag1"].append(r["engine_lag1_hz"])
        est["eng_pgram"].append(r["engine_periodogram_hz"])
        est["leg_ls"].append(r["legacy_ls_hz"])
        est["leg_lag1"].append(r["legacy_lag1_hz"])
    summ = {
        "engine_ls": fit(SWEEP, np.array(est["eng_ls"])),
        "engine_lag1": fit(SWEEP, np.array(est["eng_lag1"])),
        "engine_periodogram": fit(SWEEP, np.array(est["eng_pgram"])),
        "legacy_ls": fit(SWEEP, np.array(est["leg_ls"])),
        "legacy_lag1": fit(SWEEP, np.array(est["leg_lag1"])),
        "dig_vs_nominal": fit(
            np.array([float(r["actual_cfo_hz_nominal"]) for r in dig]),
            np.array([float(r["estimated_cfo_hz"]) for r in dig]),
        ),
        "dig_vs_digitized_actual": fit(
            np.array([float(r["actual_cfo_hz_digitized"]) for r in dig]),
            np.array([float(r["estimated_cfo_hz"]) for r in dig]),
        ),
        "legacy_samples_per_tick": ad.samples_per_tick,
        "legacy_apparent_scale": ad.apparent_scale,
        "legacy_delivered": ad.delivered(N_WAVE),
        "engine_max_abs_err_ls_hz": float(np.max(np.abs(np.array(est["eng_ls"]) - SWEEP))),
        "engine_max_abs_err_lag1_hz": float(np.max(np.abs(np.array(est["eng_lag1"]) - SWEEP))),
        "engine_max_abs_err_periodogram_hz": float(
            np.max(np.abs(np.array(est["eng_pgram"]) - SWEEP))
        ),
    }
    summ["acceptance"] = {
        "V-CFO-1_engine_ls_and_lag1_le_1e-6_hz": bool(
            summ["engine_max_abs_err_ls_hz"] <= 1e-6 and summ["engine_max_abs_err_lag1_hz"] <= 1e-6
        ),
        "V-CFO-2_legacy_slope_1.0691_pm_0.0005": bool(
            abs(summ["legacy_ls"]["slope"] - 1.0691) <= 0.0005
        ),
    }
    return rows, summ


def noise_check(lk: Link, n_runs: int = 30) -> dict[str, Any]:
    """The engine's RX noise with the signal off (TX full scale -300 dBm): variance against
    k_B T0 F f_s (F = 0 dB) and the normalised autocorrelation at lags 1, 2, 3, 5, 10
    (whiteness), over ``n_runs`` keyed runs of 2389 samples."""
    from sremu import rng

    n_mw = K_B * T0 * float(FS) * 1e3
    var, acs = [], []
    for i in range(n_runs):
        seed = int(rng.generator(MASTER_SEED, "trial", f"vcfo4/noisecheck/{i}").integers(0, 2**62))
        y = lk.receive(seed=seed, cfo_hz=0.0, full_scale_dbm=-300.0, noise=True)
        y = y.astype(np.complex128)
        var.append(float(np.mean(np.abs(y) ** 2)))
        acs.append([complex(np.vdot(y[:-g], y[g:]) / np.vdot(y, y)) for g in (1, 2, 3, 5, 10)])
    a = np.asarray(acs)
    n_s = n_runs * N_WAVE
    return {
        "n_runs": n_runs,
        "n_samples": n_s,
        "variance_over_kT0Ffs": float(np.mean(var)) / n_mw,
        "variance_rel_se": 1.0 / math.sqrt(n_s),
        "abs_mean_rho_lags_1_2_3_5_10": [float(v) for v in np.abs(a.mean(axis=0))],
        "rho_se": 1.0 / math.sqrt(n_s),
    }


def reference_mc(snr_db: float, n_trials: int = 2000) -> dict[str, Any]:
    """Engine-free reference: the same periodogram estimator on a numpy complex tone
    (|x|^2 = 2, random phase, CFO uniform in [-1200, 1200] Hz) in complex AWGN at the same
    per-sample SNR, N = 2000 (keyed generator, purpose ``trial``)."""
    from sremu import rng

    g = rng.generator(MASTER_SEED, "trial", f"vcfo4/reference_mc/snr{snr_db:g}")
    snr = 10 ** (snr_db / 10)
    n = HEADER[1] - HEADER[0]
    k = np.arange(n)
    err = np.empty(n_trials)
    for i in range(n_trials):
        f = g.uniform(-1200.0, 1200.0)
        w = (g.standard_normal(n) + 1j * g.standard_normal(n)) * math.sqrt(1.0 / snr)
        y = (1 + 1j) * np.exp(2j * np.pi * f * k / float(FS) + 1j * g.uniform(0, 2 * np.pi)) + w
        err[i] = est_periodogram(y) - f
    crb = crb_std_hz(snr_db)
    r = float(np.sqrt(np.mean(err * err)))
    return {
        "snr_db": snr_db,
        "n_trials": n_trials,
        "rmse_hz": r,
        "rmse_over_crb": r / crb,
        "rmse_over_crb_se": r / crb / math.sqrt(2.0 * n_trials),
        "bias_hz": float(np.mean(err)),
    }


def noisy(
    run_dir: Path, *, jobs: int, n_trials: int, progress: Any = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """V-CFO-4 through the sremu sweep runner (one point per (SNR, CFO))."""
    from sremu.experiments.sweep import SweepSpec, run_sweep

    spec = SweepSpec.from_mapping(
        {
            "name": "vcfo4",
            "runner": "vcfo:run_point",
            "description": "V-CFO-4 noisy trials (validation/cfo/vcfo.py)",
            "base": {"n_trials": n_trials},
            "grid": {"snr_db": list(SNRS_DB), "cfo_hz": list(CFO_POINTS)},
        }
    )
    sw = run_sweep(spec, run_dir, jobs=jobs, allow_import=True, progress=progress)
    if sw.n_failed:
        raise SystemExit(
            f"V-CFO-4: {sw.n_failed} points failed: "
            f"{[r.get('error') for r in sw.records if r['status'] != 'ok']}"
        )
    rows = [dict(r["summary"]) | {"point_id": r["point_id"]} for r in sw.records]
    by_snr: dict[str, Any] = {}
    for snr in SNRS_DB:
        rs = [r for r in rows if r["snr_db"] == snr]
        n = sum(r["n_trials"] for r in rs)
        crb = crb_std_hz(snr)
        d: dict[str, Any] = {
            "crb_std_hz": crb,
            "lag1_theory_std_hz": lag1_std_hz(snr),
            "n_trials_total": n,
            "snr_hat_db_mean": float(np.mean([r["snr_hat_db_mean"] for r in rs])),
        }
        for est in ("periodogram", "lag1", "ls"):
            pooled = math.sqrt(sum(r[f"{est}_sum_sq_hz2"] for r in rs) / n)
            d[f"{est}_rmse_pooled_hz"] = pooled
            d[f"{est}_rmse_over_crb_pooled"] = pooled / crb
            d[f"{est}_rmse_over_crb_max_point"] = max(r[f"{est}_rmse_over_crb"] for r in rs)
            d[f"{est}_rmse_over_crb_min_point"] = min(r[f"{est}_rmse_over_crb"] for r in rs)
            d[f"{est}_bias_max_abs_point_hz"] = max(abs(r[f"{est}_bias_hz"]) for r in rs)
        d["lag1_rmse_over_theory_pooled"] = d["lag1_rmse_pooled_hz"] / d["lag1_theory_std_hz"]
        by_snr[f"{snr:g}"] = d
    acc = {
        f"V-CFO-4_periodogram_le_1.5crb_every_point_snr{snr:g}dB": bool(
            by_snr[f"{snr:g}"]["periodogram_rmse_over_crb_max_point"] <= 1.5
        )
        for snr in (10.0, 20.0)
    }
    return rows, {
        "by_snr": by_snr,
        "acceptance": acc,
        "sweep_dir": vs.rel(sw.root),
        "cpu_s_sum": float(sum(r["wall_s"] for r in sw.records)),
        "point_wall_s_max": float(max(r["wall_s"] for r in sw.records)),
        "jobs": int(jobs),
    }


def study(*, jobs: int, fast: bool, root: Path | None = None) -> Path:
    inputs = _check_data()
    run = vs.StudyRun(
        STUDY_DIR,
        "vcfo_fast" if fast else "vcfo",
        scenario(),
        inputs=inputs,
        root=root,
        assumptions=(
            "V-CFO SIM link geometry (TX, RX, 16x16 RIS) [ENG]; static, isotropic antennas",
            "CFO points of V-CFO-4: -1200 ... +1200 Hz in 400 Hz steps [ENG]",
            "per-sample SNR defined on the 1+1j header (|x|^2 = 2) against k_B T0 F f_s, F = 0 dB",
        ),
    )
    lk = Link()
    t0 = time.perf_counter()
    rows, summ = noise_free(lk)
    run.time("noise_free_sweeps", time.perf_counter() - t0)
    vs.write_csv(run.path("vcfo_sweep_noise_free.csv"), rows)
    t0 = time.perf_counter()
    nrows, nsumm = noisy(
        run.run_dir, jobs=jobs, n_trials=4 if fast else N_TRIALS, progress=run.progress
    )
    run.time("vcfo4_sweep_wall", time.perf_counter() - t0)
    vs.write_csv(run.path("vcfo4_points.csv"), nrows)
    t0 = time.perf_counter()
    ncheck = noise_check(lk, n_runs=4 if fast else 30)
    ref = {f"{snr:g}": reference_mc(snr, n_trials=50 if fast else 2000) for snr in SNRS_DB}
    run.time("noise_check_and_reference_mc", time.perf_counter() - t0)
    vs.write_json(run.path("vcfo4_checks.json"), {"noise_check": ncheck, "reference_mc": ref})
    crb_rows = [
        {
            "snr_db": snr,
            "n_samples": HEADER[1] - HEADER[0],
            "fs_sps": float(FS),
            "crb_std_hz": crb_std_hz(snr),
            "design_derived_hz": {0.0: 0.0256, 10.0: 0.0081, 20.0: 0.0026}[snr],
            "crb_matches_design_4dp": round(crb_std_hz(snr), 4)
            == {0.0: 0.0256, 10.0: 0.0081, 20.0: 0.0026}[snr],
            "efficient_pooled_ratio_se": 1.0
            / math.sqrt(2.0 * nsumm["by_snr"][f"{snr:g}"]["n_trials_total"]),
            **{k: v for k, v in nsumm["by_snr"][f"{snr:g}"].items() if k != "crb_std_hz"},
        }
        for snr in SNRS_DB
    ]
    vs.write_csv(run.path("vcfo4_rmse_vs_crb.csv"), crb_rows)
    meta = {
        "study": "V-CFO",
        "fast": fast,
        "data_classes": ["DIG", "SIM-sremu", "ANALYTIC"],
        "data_class_notes": {
            "legacy_*": "SIM-sremu: the engine's CFO-free output re-timed by the "
            "LegacyTruncationAdapter (integer samples per tick), not output of another tool",
            "dig_*": "DIG: ANTS Fig. 2 (emulated channel of an earlier in-house simulator + "
            "external SW receiver), +-4.5 Hz",
            "crb": "ANALYTIC",
        },
        "paper_use": {
            "engine_*": "paper (F5a): the new engine, slope 1",
            "vcfo4_*, crb": "paper (F5b): RMSE vs CRB",
            "legacy_*": "paper only as a generic design lesson (integer samples per tick -> "
            "frequency scaling 11.76/11), never named as v1 or as a before/after (D24)",
            "dig_*": "internal only: output of the earlier simulator; not featured in the "
            "paper (D24: no v1/v2 comparisons)",
            "V-CFO-2": "internal teeth/regression test (D24)",
        },
        "seeds": {"master_seed": MASTER_SEED, "trial_key": "trial/vcfo4/snr<s>/f<f>/<i>"},
        "env_threads": vs.env_threads(),
        "acceptance": {**summ["acceptance"], **nsumm["acceptance"]},
        "numbers": {
            "noise_free": summ,
            "vcfo4": nsumm,
            "crb": crb_rows,
            "noise_check": ncheck,
            "reference_mc": ref,
            "h_abs": abs(lk.h),
            "nfft": NFFT,
            "header_window": list(HEADER),
            "ls_window_noise_free": list(LS_WIN),
        },
    }
    path = run.finish(meta)
    print(f"V-CFO run directory: {run.run_dir}\nmeta: {path}")
    if root is None and not fast:
        print(f"report: {vs.write_report(STUDY_DIR, report(run.run_dir))}")
    return run.run_dir


# =========================================================================== report


def report(run_dir: Path) -> str:
    """``validation/cfo/report.md`` rendered from a run directory's ``meta.json``."""
    meta = vs.read_meta(run_dir)
    f = vs.fnum
    acc = meta["acceptance"]
    nf = meta["numbers"]["noise_free"]
    v4 = meta["numbers"]["vcfo4"]
    crb = meta["numbers"]["crb"]
    s = []
    s.append(
        "# V-CFO: receiver-LO (CFO) verification and the noisy estimator study (DESIGN.md 10.10)\n"
    )
    s.append(
        "Verification of sremu's receiver-LO (CFO) path against exact expectations and the "
        "Cramer-Rao bound. **Paper use (DESIGN D24):** no comparison with earlier simulators "
        "and no before/after framing. The tick-truncation rows (`LegacyTruncationAdapter`: "
        "sremu's CFO-free output re-timed with an integer number of samples per tick) may "
        "appear only as a generic design lesson; the DIG rows (ANTS 2025 Fig. 2, digitized, "
        "whose quantity came from the authors' earlier in-house simulator plus an external "
        "software receiver, not an RF measurement) and the V-CFO-2 regression are kept for "
        "internal traceability and are not for the paper (`paper_use` in meta.json). Data "
        f"classes: {', '.join(meta.get('data_classes', []))}.\n"
    )
    s.append("## Provenance\n")
    s.append(vs.run_header(meta) + "\n")
    s.append(
        "* Inputs: "
        + "; ".join(
            f"`{i['path']}` ({i['data_class']}, sha256 `{i['sha256'][:16]}...`)"
            for i in meta["inputs"]
        )
        + ".\n"
    )
    s.append("## Setup\n")
    s.append(
        "BPSK test burst (200-symbol 1+1j header, 38 BPSK symbols at +-0.1(1+j), 10 "
        "samples/symbol, 2389 samples at 5880 S/s) through the in-process lockstep engine and "
        "the WP2 propagation model of a static SIM link with a 16 x 16 RIS at 2.4 GHz - 3050 Hz "
        f"(|h| = {f(meta['numbers']['h_abs'], 6)}). The RX LO is offset by -f, so the received "
        "baseband rotates at +f. Noise-free sweeps: f = -1400 ... +1400 Hz in 100 Hz steps; "
        "estimators on header samples [30, 1950). V-CFO-4: the engine's physical noise "
        "(k T0 F f_s per RX, F = 0 dB), per-sample header SNR set through the TX full scale, "
        "200 independently keyed trials (purpose `trial`) at each of 7 CFO points (-1200 ... "
        "+1200 Hz) per SNR; estimators on the N = 2000 header samples.\n"
    )
    s.append("## Acceptance\n")
    rows = [
        (
            "V-CFO-1: engine LS and lag-1, max |f_hat - f| <= 1e-6 Hz (noise off)",
            f"LS {f(nf['engine_max_abs_err_ls_hz'], 3)} Hz; lag-1 "
            f"{f(nf['engine_max_abs_err_lag1_hz'], 3)} Hz",
            vs.passfail(acc["V-CFO-1_engine_ls_and_lag1_le_1e-6_hz"]),
        ),
        (
            "V-CFO-2 (internal): tick-truncation adapter slope 1.0691 +- 0.0005",
            f(nf["legacy_ls"]["slope"], 6),
            vs.passfail(acc["V-CFO-2_legacy_slope_1.0691_pm_0.0005"]),
        ),
    ]
    for snr in ("10", "20"):
        d = v4["by_snr"][snr]
        rows.append(
            (
                f"V-CFO-4 at {snr} dB: periodogram RMSE <= 1.5 x CRB at every CFO point",
                f"max point {f(d['periodogram_rmse_over_crb_max_point'], 3)} x CRB "
                f"(pooled {f(d['periodogram_rmse_over_crb_pooled'], 3)})",
                vs.passfail(acc[f"V-CFO-4_periodogram_le_1.5crb_every_point_snr{snr}dB"]),
            )
        )
    d0 = v4["by_snr"]["0"]
    rows.append(
        (
            "V-CFO-4 at 0 dB (reported, not gated)",
            f"periodogram max point {f(d0['periodogram_rmse_over_crb_max_point'], 3)} x CRB "
            f"(pooled {f(d0['periodogram_rmse_over_crb_pooled'], 3)})",
            "n/a",
        )
    )
    s.append(vs.md_table(("Check", "Measured", "Result"), rows) + "\n")
    s.append("## Noise-free sweeps (figure F5a data: `vcfo_sweep_noise_free.csv`)\n")
    s.append(
        "Least-squares line of estimate vs actual CFO over the 29 sweep points; RMSE of "
        "(estimate - actual); residual RMSE after the fitted line.\n"
    )
    names = (
        ("engine_ls", "new engine, unwrapped-phase LS"),
        ("engine_lag1", "new engine, lag-1"),
        ("engine_periodogram", "new engine, periodogram + parabolic"),
        ("legacy_ls", "integer samples per tick (generic pitfall), unwrapped-phase LS"),
        ("legacy_lag1", "integer samples per tick (generic pitfall), lag-1"),
        ("dig_vs_nominal", "DIG (ANTS Fig. 2; internal only) vs nominal grid"),
        ("dig_vs_digitized_actual", "DIG (internal only) vs digitized x positions"),
    )
    s.append(
        vs.md_table(
            (
                "series",
                "slope",
                "intercept (Hz)",
                "RMSE (Hz)",
                "max |err| (Hz)",
                "resid. after fit (Hz)",
            ),
            (
                (
                    lab,
                    f(nf[k]["slope"], 6),
                    f(nf[k]["intercept_hz"], 3, sign=True),
                    f(nf[k]["rmse_hz"], 3),
                    f(nf[k]["max_abs_err_hz"], 3),
                    f(nf[k]["rmse_after_fit_hz"], 3),
                )
                for k, lab in names
            ),
        )
        + "\n"
    )
    s.append(
        f"The tick-truncation adapter delivers {nf['legacy_delivered']} of 2389 samples "
        f"({nf['legacy_samples_per_tick']} samples per 2 ms tick; apparent scale "
        f"{f(nf['legacy_apparent_scale'], 6)} = 11.76/11). DIG points carry +-4.5 Hz.\n"
    )
    s.append(
        "Paper use of the columns of `vcfo_sweep_noise_free.csv` (D24): `engine_*` are the "
        "paper-facing F5a series; `legacy_*` (the internal adapter's name) may appear only as "
        "the generic design lesson, under a neutral label such as 'integer samples per tick', "
        "never as an earlier simulator or a before/after; `dig_*` are internal only and are not "
        "plotted.\n"
    )
    s.append("## V-CFO-4: RMSE vs the CRB (figure F5b data: `vcfo4_rmse_vs_crb.csv`)\n")
    s.append(
        "CRB of the frequency of a complex tone in complex AWGN with unknown amplitude and "
        "phase (Rife & Boorstyn 1974), recomputed: sigma_f = f_s/(2 pi) sqrt(6/(SNR N "
        "(N^2 - 1))), N = 2000, f_s = 5880 S/s. Lag-1 small-error theory [DERIVED in "
        "`vcfo.py`]: var = 1/(SNR (N-1)^2) + 1/(2 SNR^2 (N-1)) rad^2. RMSE pooled over "
        "7 x 200 = 1400 trials per SNR.\n"
    )
    s.append(
        vs.md_table(
            (
                "SNR (dB)",
                "CRB (Hz)",
                "DESIGN [DERIVED] (Hz)",
                "realised SNR (dB)",
                "periodogram RMSE (Hz)",
                "/CRB pooled",
                "/CRB max point",
                "lag-1 RMSE (Hz)",
                "lag-1 theory (Hz)",
                "lag-1 /CRB",
                "LS RMSE (Hz)",
            ),
            (
                (
                    f(r["snr_db"], 0),
                    f(r["crb_std_hz"], 6),
                    f(r["design_derived_hz"], 4),
                    f(r["snr_hat_db_mean"], 3),
                    f(r["periodogram_rmse_pooled_hz"], 6),
                    f(r["periodogram_rmse_over_crb_pooled"], 3),
                    f(r["periodogram_rmse_over_crb_max_point"], 3),
                    f(r["lag1_rmse_pooled_hz"], 4),
                    f(r.get("lag1_theory_std_hz"), 4),
                    f(r["lag1_rmse_over_crb_pooled"], 1),
                    f(r["ls_rmse_pooled_hz"], 4),
                )
                for r in crb
            ),
        )
        + "\n"
    )
    s.append(
        "The lag-1 estimator meets its own small-error prediction but not the CRB: its "
        "variance falls as 1/N, the bound's as 1/N^3. The unwrapped-phase LS estimator fails "
        "at 0 dB (phase-unwrapping errors) and is reported only. Per-point rows (bias, SD, "
        "RMSE, ratio to the CRB per estimator, realised SNR, wall time) are in "
        "`vcfo4_points.csv`; trials in `vcfo4/p*/trials.csv`.\n"
    )
    nc = meta["numbers"].get("noise_check")
    ref = meta["numbers"].get("reference_mc")
    if nc and ref:
        s.append("### Checks of the noise and of the estimator\n")
        s.append(
            "For an efficient estimator the pooled RMSE/CRB has a sampling SE of about "
            "1/sqrt(2 n) = "
            + ", ".join(f"{f(r['efficient_pooled_ratio_se'], 4)}" for r in crb)
            + f" (n = {v4['by_snr']['10']['n_trials_total']} per SNR). Engine-free reference "
            "(the same periodogram estimator on "
            "a numpy tone in complex AWGN, keyed trials): RMSE/CRB "
            + "; ".join(
                f"{k} dB {f(v['rmse_over_crb'], 3)} (SE {f(v['rmse_over_crb_se'], 3)}, "
                f"n = {v['n_trials']})"
                for k, v in ref.items()
            )
            + f". Engine noise with the signal off ({nc['n_runs']} keyed runs, "
            f"{nc['n_samples']} samples): variance {f(nc['variance_over_kT0Ffs'], 4)} x "
            f"k T0 F f_s (relative SE {f(nc['variance_rel_se'], 4)}); |mean normalised "
            "autocorrelation| at lags 1, 2, 3, 5, 10 = "
            + ", ".join(f(x, 4) for x in nc["abs_mean_rho_lags_1_2_3_5_10"])
            + f" (SE {f(nc['rho_se'], 4)}).\n"
        )
        # how unusual the pooled ratios are for an efficient estimator: sum of squared errors
        # over CRB^2 ~ chi2_n (n = trials per SNR), lower-tail probability
        from scipy import stats

        tails, p_low, p_two = [], [], []
        for r in crb:
            n = int(r["n_trials_total"])
            ratio = float(r["periodogram_rmse_over_crb_pooled"])
            z = (ratio - 1.0) / float(r["efficient_pooled_ratio_se"])
            p = float(stats.chi2.cdf(n * ratio * ratio, n))
            p_low.append(p)
            p_two.append(2.0 * min(p, 1.0 - p))
            tails.append(f"{f(r['snr_db'], 0)} dB {z:+.1f} SE, P_low = {p:.3f}")
        b_low, b_two = min(1.0, 3 * min(p_low)), min(1.0, 3 * min(p_two))
        if b_low > 0.05:
            verdict = (
                "No pooled ratio is significantly below 1 once the three SNRs are accounted for "
                f"(Bonferroni 3 min P_low = {b_low:.3f}); with the white, calibrated noise above "
                "this is read as sampling fluctuation, not as an estimator beating the bound"
            )
        elif b_two > 0.05:
            verdict = (
                f"Borderline: the lowest ratio is below 1 at Bonferroni 3 min P_low = {b_low:.3f} "
                f"one-sided but not two-sided (3 min P = {b_two:.3f}); with the white, calibrated "
                "noise above it is read as sampling fluctuation, which more trials would confirm "
                "or refute"
            )
        else:
            verdict = (
                f"At least one pooled ratio differs from 1 significantly (Bonferroni two-sided "
                f"3 min P = {b_two:.3f}): an estimator cannot beat the bound on average, so the "
                "noise or SNR calibration needs investigation"
            )
        s.append(
            "Distance of the pooled ratio from 1 and the lower-tail probability of the pooled sum "
            "of squared errors under an efficient estimator (sum e^2/CRB^2 ~ chi2_n): "
            + "; ".join(tails)
            + f". {verdict}. The gate (RMSE <= 1.5 x CRB at every CFO point) does not depend on "
            "this.\n"
        )
    s.append("## Run times\n")
    t = meta.get("timings_s", {})
    s.append(
        f"Noise-free sweeps {f(t.get('noise_free_sweeps'), 1)} s; V-CFO-4 sweep "
        f"{f(t.get('vcfo4_sweep_wall'), 1)} s wall with {v4.get('jobs')} sweep-runner workers "
        f"(summed point time {f(v4.get('cpu_s_sum'), 0)} s, longest point "
        f"{f(v4.get('point_wall_s_max'), 1)} s); noise check and engine-free reference "
        f"{f(t.get('noise_check_and_reference_mc'), 1)} s. Indicative only (shared laptop).\n"
    )
    return "\n".join(s)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="V-CFO validation study (DESIGN.md 10.10)")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None, help="write report.md from a run dir")
    a = ap.parse_args(argv)
    if a.report is not None:
        print(vs.write_report(STUDY_DIR, report(a.report)))
        return 0
    study(jobs=a.jobs, fast=a.fast, root=a.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
