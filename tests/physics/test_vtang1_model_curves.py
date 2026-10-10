"""V-TANG-1: engine vs Tang et al. 2022 published model curves (Fig. 10, Eq. 9) within
0.05 dB (section 10.8), run with the shipped default ``cell_model`` (auto).

Inputs: ``data/validation/tang2022/tang2022_fig10_vector.json`` (vector-extracted from the
arXiv PDF; DIG, checked against ``SHA256SUMS``) and Tang 2022 Tables III/IV: RIS1 (56 x 20
cells, 1.4 x 2.8 mm, 27 GHz, coding state '0' = 0.9 at 165 deg), P_T = 20 dBm, horn
patterns cos^53.8 pointed at the panel centre, measured G_t G_r G_line = 24.0 dB
(system B). The engine's CosPower antennas carry the nominal peak gains
G_0 = 2(alpha + 1); the measured calibration replaces them, so
``P_R = P_T + 24.0 dB - 10 log10(G_0^2) + 20 log10 |h_RIS|``.
"""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.physics.propagation import ScenePropagation

pytestmark = pytest.mark.physics

DATA = Path(__file__).resolve().parents[2] / "data" / "validation" / "tang2022"
PT_DBM = 20.0  # Tang 2022 Table IV
CAL_DB = 24.0  # measured Gt*Gr*Gline, Tang 2022 Table IV (system B)
ALPHA = 53.8  # horn cos^alpha pattern, Tang 2022
PANELS = {"a": (1.0, 10.0), "b": (2.0, 10.0), "c": (1.0, 45.0), "d": (2.0, 45.0)}


def _data() -> dict:
    raw = (DATA / "tang2022_fig10_vector.json").read_bytes()
    sums = dict(line.split()[::-1] for line in (DATA / "SHA256SUMS").read_text().splitlines())
    assert hashlib.sha256(raw).hexdigest() == sums["tang2022_fig10_vector.json"]
    return json.loads(raw)["data"]


def _scenario(d1: float, d2: float, th_deg: float) -> dict:
    th = math.radians(th_deg)
    ptx = [-d1 * math.sin(th), 0.0, d1 * math.cos(th)]
    prx = [d2 * math.sin(th), 0.0, d2 * math.cos(th)]

    def node(nid: str, pos: list[float], direction: str) -> dict:
        b = (-np.asarray(pos) / np.linalg.norm(pos)).tolist()  # boresight at the panel centre
        stream = {"name": f"{direction}0", "direction": direction, "fc_hz": 27e9, "rate_sps": "1e6"}
        if direction == "tx":
            stream["occupied_bw_hz"] = 1e5
        return {
            "id": nid,
            "position_m": pos,
            "antenna": {"pattern": {"cos_power": {"alpha": ALPHA}}, "boresight": b},
            "streams": [stream],
        }

    return {
        "seed": 1,
        "nodes": [node("tx", ptx, "tx"), node("rx", prx, "rx")],
        "ris": [
            {
                "id": "ris1",
                "pose": {"center_m": [0, 0, 0], "normal": [0, 0, 1], "u_axis": [1, 0, 0]},
                "grid": {
                    "n_u": 56,
                    "n_v": 20,
                    "pitch_u_m": 1.4e-3,
                    "pitch_v_m": 2.8e-3,
                    "cell_model": "auto",
                },
                "states": {"library": "tang2022_ris1_27ghz"},
                "initial": {"fill": 0},
            }
        ],
    }


def test_vtang1_engine_reproduces_published_eq9_curves():
    data = _data()
    g0 = 2 * (ALPHA + 1)
    worst = 0.0
    n = 0
    for panel, (d1, th) in PANELS.items():
        for d2_dig, p_dig in data[panel]["refined_eq9"]:
            d2 = round(d2_dig * 2) / 2  # digitizer x offset (+0.002 m) on the 0.5 m grid
            model = ScenePropagation.from_scenario(_scenario(d1, d2, th))
            assert model.cell_model("ris1") == "point"  # auto: pitch < lambda/2 at 27 GHz
            snap = model.snapshot(Fraction(0))
            lp = model.combine(snap, {"ris1": model.initial_config("ris1")})["tx.tx0->rx.rx0"]
            h_ris = lp.amp[lp.tile >= 0]
            assert h_ris.size == 1
            p_dbm = PT_DBM + CAL_DB - 20 * math.log10(g0) + 20 * math.log10(abs(h_ris[0]))
            worst = max(worst, abs(p_dbm - p_dig))
            n += 1
    assert n == 36
    assert worst <= 0.05, worst
