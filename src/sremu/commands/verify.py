"""``sremu verify [--quick] [--json]``: oracle subset and model-card numbers (DESIGN.md 2.5).

An installed-wheel smoke test (WP4). Two groups of checks:

* ``oracle``: closed forms of the model card recomputed here (no sremu implementation code)
  and compared with the spec oracle values (sections 5, 10.2, 10.10): Friis, Tang Eq. (8)
  N^2 law, plate RCS, focusing saturation, two-ray, Brewster angle, knife edge, quantization
  and binary-alphabet losses, Abeywickrama, m/d decomposition, Doppler, thermal noise floor,
  the v1 truncation slope and the CRB of V-CFO-4.
* ``implementation`` (skipped with ``--quick``): the installed implementation reached only
  through the public contracts: PH-01 Friis through ``PropagationModel`` (LOS), PH-04 far
  specular plate through the RIS kernel, and PH-21 noise power through the render backend's
  ``noise``. A check whose implementation module is not installed reports ``SKIP``.

Exit status 0 when nothing FAILs, 1 otherwise. The closed forms are deliberately duplicated
from ``tests/oracles`` (which is not part of the wheel) and kept tiny.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import typer

from sremu.units import C0, K_B, T0

__all__ = ["CheckResult", "command", "implementation_checks", "oracle_checks", "run_checks"]


@dataclass(frozen=True)
class CheckResult:
    """One verify line: ``status`` is PASS, FAIL or SKIP."""

    group: str
    spec_id: str
    quantity: str
    value: float | None
    expected: float
    tol: float
    unit: str
    status: str
    detail: str = ""


def _judge(
    group: str, spec_id: str, quantity: str, value: float, expected: float, tol: float, unit: str
) -> CheckResult:
    ok = math.isfinite(value) and abs(value - expected) <= tol + 1e-12
    return CheckResult(
        group, spec_id, quantity, value, expected, tol, unit, "PASS" if ok else "FAIL"
    )


# --------------------------------------------------------------------------- closed forms


def _db20(x: complex) -> float:
    return 20.0 * math.log10(abs(x))


def _friis_db(d_m: float, fc_hz: float) -> float:
    return _db20(C0 / fc_hz / (4.0 * math.pi * d_m))


def _two_ray_loss_db(d_m: float, ht_m: float, hr_m: float, fc_hz: float) -> float:
    lam = C0 / fc_hz
    k = 2.0 * math.pi / lam
    d0 = math.hypot(d_m, ht_m - hr_m)
    d1 = math.hypot(d_m, ht_m + hr_m)
    dd = 4.0 * ht_m * hr_m / (d0 + d1)
    h = lam / (4.0 * math.pi) * (1.0 / d0 - np.exp(-1j * k * dd) / d1)
    return -_db20(complex(h))


def _knife_edge_loss_db(nu: float) -> float:
    from scipy.special import fresnel

    s, c = fresnel(nu)
    f = (1.0 + 1.0j) / 2.0 * ((0.5 - c) - 1j * (0.5 - s))
    return -_db20(complex(f))


def _eta_bbit_db(b: int) -> float:
    m = 2**b
    return 10.0 * math.log10((m / math.pi * math.sin(math.pi / m)) ** 2)


def _eta_binary_db(delta_rad: float) -> float:
    d = math.fmod(abs(delta_rad), 2.0 * math.pi)
    d = min(d, 2.0 * math.pi - d)
    return 10.0 * math.log10((2.0 * math.sin(d / 2.0) / math.pi) ** 2)


def _abeywickrama_db(beta_min: float, alpha: float, phi: float = 0.43 * math.pi) -> float:
    from scipy.integrate import quad

    def beta(t: float) -> float:
        return (1.0 - beta_min) * ((math.sin(t - phi) + 1.0) / 2.0) ** alpha + beta_min

    return 20.0 * math.log10(quad(beta, -math.pi, math.pi, epsabs=1e-13)[0] / (2.0 * math.pi))


def _tang_eq8_db(n_cells: int, cell_area: float, d: float, theta_r: float) -> float:
    pl = 16.0 * math.pi**2 * (d * d) ** 2 / ((n_cells * cell_area) ** 2 * math.cos(theta_r))
    return -10.0 * math.log10(pl)


def oracle_checks() -> list[CheckResult]:
    """Closed-form model-card numbers vs the spec oracle values."""
    g = "oracle"
    lam35 = C0 / 3.5e9
    out = [
        _judge(g, "PH-01", "Friis 3.5 GHz, 10 m", _friis_db(10.0, 3.5e9), -63.329, 5e-4, "dB"),
        _judge(g, "PH-01", "Friis 100 MHz, 52 km", _friis_db(52e3, 100e6), -106.768, 5e-4, "dB"),
    ]
    for n_side, exp in ((10, -169.027), (20, -156.985), (40, -144.944)):
        val = _tang_eq8_db(n_side * n_side, (lam35 / 2) ** 2, 2000.0, math.radians(20))
        out.append(
            _judge(g, "PH-03", f"Tang Eq. 8, N = {n_side * n_side}, 2 km", val, exp, 0.01, "dB")
        )
    out.append(
        _judge(
            g,
            "PH-06",
            "plate 1 m^2, 400 m monostatic",
            _db20(1.0 / (4 * math.pi * 400.0**2)),
            -126.067,
            0.01,
            "dB",
        )
    )
    om = 4.0 * math.atan(0.25 / (1.0 * math.sqrt(0.25 + 0.25 + 1.0)))
    out.append(
        _judge(
            g, "PH-07", "focusing, d = 1 m, L = 1 m", _db20(om / (4 * math.pi)), -23.864, 0.01, "dB"
        )
    )
    out.append(_judge(g, "PH-07", "focusing limit L -> inf", _db20(0.5), -6.021, 0.01, "dB"))
    out.append(
        _judge(
            g,
            "PH-10",
            "two-ray 52 km, 200/1.5 m, 100 MHz",
            _two_ray_loss_db(52e3, 200.0, 1.5, 100e6),
            139.10,
            0.01,
            "dB",
        )
    )
    out.append(
        _judge(
            g,
            "PH-10",
            "Brewster grazing angle, eps_r = 4",
            math.degrees(math.asin(1 / math.sqrt(5.0))),
            26.565,
            5e-4,
            "deg",
        )
    )
    out.append(
        _judge(
            g, "PH-11", "|F(0)| knife edge", 10 ** (-_knife_edge_loss_db(0.0) / 20), 0.5, 1e-12, "1"
        )
    )
    out.append(
        _judge(
            g, "PH-11", "knife-edge loss at nu = 10", _knife_edge_loss_db(10.0), 32.95, 0.01, "dB"
        )
    )
    for b, exp in ((1, -3.922), (2, -0.912), (3, -0.224)):
        out.append(
            _judge(g, "PH-13", f"eta({b}) uniform {b}-bit", _eta_bbit_db(b), exp, 0.001, "dB")
        )
    out.append(
        _judge(
            g,
            "PH-13",
            "eta lab pair (203.41 deg)",
            _eta_binary_db(math.radians(203.41)),
            -4.105,
            0.001,
            "dB",
        )
    )
    out.append(
        _judge(
            g, "PH-14", "Abeywickrama (0.2, 1.6)", _abeywickrama_db(0.2, 1.6), -5.508, 0.01, "dB"
        )
    )
    out.append(
        _judge(
            g, "PH-14", "Abeywickrama (0.8, 1.6)", _abeywickrama_db(0.8, 1.6), -1.085, 0.01, "dB"
        )
    )
    a = np.exp(1j * math.radians(49.64))
    b2 = np.exp(-1j * math.radians(153.77))
    m, d = (a + b2) / 2, (a - b2) / 2
    out.append(_judge(g, "PH-25", "|d| lab pair", _db20(complex(d)), -0.18, 0.01, "dB"))
    out.append(_judge(g, "PH-25", "|m| lab pair", _db20(complex(m)), -13.86, 0.01, "dB"))
    out.append(
        _judge(g, "PH-25", "arg m lab pair", math.degrees(float(np.angle(m))), 127.9, 0.05, "deg")
    )
    out.append(
        _judge(g, "PH-17", "LOS Doppler 30 m/s, 3.5 GHz", 30.0 * 3.5e9 / C0, 350.242, 0.001, "Hz")
    )
    rx0, ris_c = np.array([50.0, 0.0, 0.0]), np.array([25.0, 10.0, 0.0])
    v_app = float(np.array([-30.0, 0.0, 0.0]) @ ((ris_c - rx0) / np.linalg.norm(ris_c - rx0)))
    out.append(
        _judge(
            g, "PH-17", "RIS-path Doppler (T9 geometry)", v_app * 3.5e9 / C0, 325.192, 0.01, "Hz"
        )
    )
    out.append(
        _judge(
            g,
            "PH-21",
            "noise floor 1 MS/s, F = 0 dB",
            10 * math.log10(K_B * T0 * 1e6) + 30,
            -113.975,
            5e-4,
            "dBm",
        )
    )
    slope = Fraction(5880) * Fraction(2, 1000)
    out.append(
        _judge(
            g,
            "V-CFO-2",
            "v1 truncation slope 11.76/11",
            float(slope / math.floor(slope)),
            1.0691,
            5e-4,
            "1",
        )
    )
    for snr_db, exp in ((0, 0.0256), (10, 0.0081), (20, 0.0026)):
        n = 2000.0
        crb = 5880.0 / (2 * math.pi) * math.sqrt(6.0 / (10 ** (snr_db / 10) * n * (n * n - 1)))
        out.append(_judge(g, "V-CFO-4", f"CRB std, {snr_db} dB, N = 2000", crb, exp, 5e-5, "Hz"))
    return out


# --------------------------------------------------------------------------- implementation


def _available(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _scenario(
    name: str, fc: float, nodes: list[tuple[str, list[float], str]], ris: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "name": name,
        "seed": 1,
        "description": "sremu verify smoke scene (SIM)",
        "numerics": {"backend": "numpy_ref"},
        "nodes": [
            {
                "id": nid,
                "position_m": pos,
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": f"{d}0",
                        "direction": d,
                        "fc_hz": fc,
                        "rate_sps": "1e6",
                        **({"occupied_bw_hz": 1e4} if d == "tx" else {}),
                    }
                ],
            }
            for nid, pos, d in nodes
        ],
        "ris": ris,
        "controllers": [],
        "run": {
            "mode": "lockstep",
            "coupling": "barrier",
            "duration_s": "1e-3",
            "late_policy": "apply_on_arrival",
        },
        "record": {"iq": False, "events": False, "metrics": False, "snapshots": False},
    }


def _from_scenario() -> Callable[[Any], Any]:
    """The PropagationModel factory: ``propagation.from_scenario`` (module function or the
    ``from_scenario`` classmethod of a class defined there; DESIGN.md section 2.2)."""
    from sremu.physics import propagation

    fn = getattr(propagation, "from_scenario", None)
    if callable(fn):
        return fn
    for obj in vars(propagation).values():
        if isinstance(obj, type) and obj.__module__ == propagation.__name__:
            fn = getattr(obj, "from_scenario", None)
            if callable(fn):
                return fn
    raise AttributeError("sremu.physics.propagation defines no from_scenario")


def _link_coefficients(
    scn: dict[str, Any], states: dict[str, np.ndarray], fc: float
) -> dict[str, dict[int, complex]]:
    from sremu.contracts import PanelConfig
    from sremu.scenario.models import Scenario

    model = _from_scenario()(Scenario.model_validate(scn))
    snap = model.snapshot(Fraction(0))
    cfg = {p: PanelConfig(p, np.asarray(s, dtype=np.int16)) for p, s in states.items()}
    out: dict[str, dict[int, complex]] = {}
    for link, lp in model.combine(snap, cfg).items():
        terms = np.asarray(lp.amp) * np.exp(-2j * math.pi * fc * np.asarray(lp.delay_s))
        kinds = np.asarray(lp.kind)
        out[str(link)] = {int(k): complex(np.sum(terms[kinds == k])) for k in np.unique(kinds)}
    return out


def _impl_friis() -> float:
    from sremu.contracts import PathKind

    fc = 3.5e9
    scn = _scenario(
        "verify_friis", fc, [("ap", [0.0, 0.0, 0.0], "tx"), ("mu", [10.0, 0.0, 0.0], "rx")], []
    )
    h = _link_coefficients(scn, {}, fc)["ap.tx0->mu.rx0"][int(PathKind.LOS)]
    return _db20(h)


def _impl_far_specular_plate() -> float:
    import yaml

    from sremu.contracts import PathKind

    fc = 3.5e9
    inc = math.radians(30.0)
    tx = [-200.0 * math.sin(inc), 0.0, 200.0 * math.cos(inc)]
    rx = [200.0 * math.sin(inc), 0.0, 200.0 * math.cos(inc)]
    lib = {
        "schema_version": "1.0",
        "kind": "ris_state_library",
        "name": "verify_unit",
        "provenance": {
            "source_kind": "analytic",
            "description": "Gamma = 1 (verify)",
            "reference": "sremu verify",
            "reference_plane_offset_m": 0.0,
            "polarization": "co",
            "includes_cell_pattern": False,
            "sigma_mag_db": 0.0,
            "sigma_phase_deg": 0.0,
            "raw_sha256": None,
            "license": "n/a",
            "created": {"date": "2026-10-10", "tool": "sremu verify", "version": "1"},
            "assumptions": [],
        },
        "frequency_hz": [fc],
        "validity_bw_hz": 1e6,
        "angle_grid": "none",
        "reciprocal": True,
        "reconfiguration_time_s": None,
        "states": [
            {"id": 0, "label": "unit", "gamma": [{"f_hz": fc, "mag": 1.0, "phase_deg": 0.0}]}
        ],
    }
    with tempfile.TemporaryDirectory(prefix="sremu_verify_") as tmp:
        lib_path = Path(tmp) / "verify_unit.yaml"
        lib_path.write_text(yaml.safe_dump(lib, sort_keys=False), encoding="utf-8")
        ris = [
            {
                "id": "ris0",
                "enabled": True,
                "pose": {
                    "center_m": [0.0, 0.0, 0.0],
                    "normal": [0.0, 0.0, 1.0],
                    "u_axis": [1.0, 0.0, 0.0],
                },
                "grid": {
                    "n_u": 16,
                    "n_v": 16,
                    "pitch_u_m": 1.0 / 16,
                    "pitch_v_m": 1.0 / 16,
                    "cell_model": "auto",
                },
                "states": {"file": str(lib_path)},
                "control": {"granularity": "element"},
                "initial": {"fill": 0},
                "transition": {"model": "step"},
                "capabilities": {},
            }
        ]
        scn = _scenario("verify_plate", fc, [("tx", tx, "tx"), ("rx", rx, "rx")], ris)
        h = _link_coefficients(scn, {"ris0": np.zeros(256)}, fc)["tx.tx0->rx.rx0"][
            int(PathKind.RIS_TILE)
        ]
    return _db20(h)


def _impl_noise_dbm() -> float:
    noise: Callable[..., np.ndarray] | None = None
    for mod in ("sremu.channel.render_numpy", "sremu.impair.noise"):
        if _available(mod):
            noise = getattr(importlib.import_module(mod), "noise", None)
            if callable(noise):
                break
    if noise is None:
        raise ModuleNotFoundError(
            "no noise(rx_key, k0, n, sigma) in sremu.channel.render_numpy / sremu.impair.noise"
        )
    from sremu.units import noise_power_w

    w = noise((0x5EED0001, 0x5EED0002), 0, 1_000_000, math.sqrt(noise_power_w(1e6, 0.0)))
    return 10.0 * math.log10(float(np.mean(np.abs(np.asarray(w, dtype=np.complex128)) ** 2))) + 30.0


def implementation_checks() -> list[CheckResult]:
    """Installed implementation vs model-card numbers (SKIP when a module is missing)."""
    g = "implementation"
    specs: list[tuple[str, str, Callable[[], float], float, float, str, tuple[str, ...]]] = [
        (
            "PH-01",
            "Friis via PropagationModel (LOS), 3.5 GHz, 10 m",
            _impl_friis,
            -63.329,
            5e-4,
            "dB",
            ("sremu.physics.propagation", "sremu.scenario.models"),
        ),
        (
            "PH-04",
            "1 m^2 plate, far specular 200 m (K1, cell_model auto)",
            _impl_far_specular_plate,
            -115.280,
            0.01,
            "dB",
            ("sremu.physics.propagation", "sremu.scenario.models"),
        ),
        (
            "PH-21",
            "noise power over 1e6 samples, 1 MS/s, F = 0 dB",
            _impl_noise_dbm,
            -113.975,
            0.05,
            "dBm",
            (),
        ),
    ]
    out = []
    for spec_id, what, fn, exp, tol, unit, mods in specs:
        missing = [m for m in mods if not _available(m)]
        if missing:
            out.append(
                CheckResult(
                    g,
                    spec_id,
                    what,
                    None,
                    exp,
                    tol,
                    unit,
                    "SKIP",
                    f"not installed: {', '.join(missing)}",
                )
            )
            continue
        try:
            out.append(_judge(g, spec_id, what, fn(), exp, tol, unit))
        except ModuleNotFoundError as exc:
            out.append(
                CheckResult(g, spec_id, what, None, exp, tol, unit, "SKIP", f"not installed: {exc}")
            )
        except Exception as exc:  # report, do not crash the smoke test
            out.append(
                CheckResult(
                    g, spec_id, what, None, exp, tol, unit, "FAIL", f"{type(exc).__name__}: {exc}"
                )
            )
    return out


def run_checks(quick: bool = False) -> list[CheckResult]:
    """Oracle checks, plus implementation checks unless ``quick``."""
    res = oracle_checks()
    if not quick:
        res += implementation_checks()
    return res


def _format(results: list[CheckResult]) -> str:
    head = f"{'group':<14} {'id':<8} {'quantity':<54} {'value':>14} {'expected':>12} {'tol':>8}"
    lines = [head + " unit  status"]
    for r in results:
        val = "-" if r.value is None else f"{r.value:.6g}"
        detail = f"  ({r.detail})" if r.detail else ""
        lines.append(
            f"{r.group:<14} {r.spec_id:<8} {r.quantity:<54} {val:>14} {r.expected:>12.6g} "
            f"{r.tol:>8.2g} {r.unit:<5} {r.status}{detail}"
        )
    n_fail = sum(r.status == "FAIL" for r in results)
    n_skip = sum(r.status == "SKIP" for r in results)
    n_pass = len(results) - n_fail - n_skip
    lines.append(f"{len(results)} checks: {n_pass} pass, {n_fail} fail, {n_skip} skipped")
    return "\n".join(lines)


def command(
    quick: bool = typer.Option(
        False, "--quick", help="Closed-form oracle subset only (no implementation calls)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the results as JSON."),
) -> None:
    """Run the oracle subset and print the model-card numbers (installed-wheel smoke test)."""
    results = run_checks(quick)
    if json_out:
        typer.echo(json.dumps([asdict(r) for r in results], indent=1))
    else:
        typer.echo(_format(results))
    if any(r.status == "FAIL" for r in results):
        raise typer.Exit(1)
