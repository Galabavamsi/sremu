"""PH-01: Friis LOS amplitude (DESIGN.md section 10.2)."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import PathKind, StreamSpec
from sremu.physics.antennas import CosPower, Isotropic
from sremu.physics.geometry import Static
from sremu.physics.los import friis_amplitude, los_path
from sremu.physics.propagation import NodeModel, ScenePropagation
from sremu.units import C0
from tests.physics._closed_forms import db20

pytestmark = pytest.mark.physics


@pytest.mark.parametrize(
    ("fc_hz", "d_m", "spec_db"),
    [(3.5e9, 10.0, -63.329), (100e6, 52000.0, -106.768)],  # PH-01 oracle table
)
def test_ph01_friis_matches_closed_form_and_oracle(fc_hz: float, d_m: float, spec_db: float):
    closed = C0 / fc_hz / (4 * math.pi * d_m)
    amp, tau = los_path([0.0, 0.0, 0.0], [d_m, 0.0, 0.0], fc_hz)
    assert abs(amp - closed) / closed <= 1e-12
    assert amp.imag == 0.0  # the propagation phase is excluded from the path amplitude
    assert tau == d_m / C0
    assert abs(db20(amp) - spec_db) <= 0.0005  # spec value is rounded to 3 decimals
    assert abs(float(friis_amplitude(d_m, fc_hz)) - closed) / closed <= 1e-12


def test_ph01_friis_with_antenna_gains_and_oblique_geometry():
    fc, a, b = 2.4e9, np.array([1.0, -2.0, 0.5]), np.array([7.0, 6.0, 2.0])
    d = float(np.linalg.norm(b - a))
    r = (b - a) / d
    tx = CosPower(3.0, r)  # boresight on the link: G = 2(3+1) = 8
    rx = CosPower(1.0, np.array([0.0, 0.0, 1.0]))  # off boresight: G = 4 cos(psi)
    cos_psi = float(-r @ np.array([0.0, 0.0, 1.0]))
    expected = math.sqrt(8.0 * (4.0 * max(cos_psi, 0.0))) * C0 / fc / (4 * math.pi * d)
    amp, _ = los_path(a, b, fc, tx, rx)
    assert abs(amp - expected) <= 1e-12 * expected


def test_ph01_through_propagation_snapshot():
    fc = 3.5e9
    tx = StreamSpec("ap.tx0", "tx", fc, "1e6")  # type: ignore[arg-type]
    rx = StreamSpec("mu.rx0", "rx", fc, "1e6")  # type: ignore[arg-type]
    model = ScenePropagation(
        [
            NodeModel("ap", Static(np.zeros(3)), Isotropic(), (tx,)),
            NodeModel("mu", Static(np.array([10.0, 0.0, 0.0])), Isotropic(), (rx,)),
        ]
    )
    snap = model.snapshot(Fraction(0))
    pb = snap.paths["ap.tx0->mu.rx0"]
    closed = C0 / fc / (4 * math.pi * 10.0)
    assert pb.kind.tolist() == [PathKind.LOS]
    assert abs(pb.amp[0] - closed) / closed <= 1e-12
    assert pb.delay_s[0] == 10.0 / C0
    assert abs(db20(pb.amp[0]) - (-63.329)) <= 0.0005
