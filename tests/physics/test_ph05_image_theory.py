"""PH-05: image theory and the Huygens ``j`` (section 10.2).

* Oblique 40 deg plate, 12 m side, Tx and Rx at 3 m (specular): K1 within 0.3 dB and 3 deg
  of the image ``lambda/(4 pi (d1+d2)) exp(-jk(d1+d2))``.
* Teeth: the ``no_j`` legacy kernel is off by -90 +- 3 deg.
* Disk on axis (R = 1 m, d = 2 m, Kirchhoff obliquity): discrete K1 within 0.01 dB of the
  exact closed form; the closed form itself tends to the image at R = 200 m (0.05 dB, 2 deg).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sremu.physics.legacy import no_j
from sremu.physics.ris_kernel import k1_channel
from sremu.units import C0
from tests.physics._closed_forms import db20, disk_kirchhoff_onaxis, make_panel

pytestmark = pytest.mark.physics

FC = 3.5e9
LAM = C0 / FC
K = 2 * math.pi / LAM
NRM = np.array([0.0, 0.0, 1.0])


def test_ph05_oblique_plate_matches_image_and_no_j_teeth():
    d1 = d2 = 3.0
    th = math.radians(40)
    tx = np.array([-d1 * math.sin(th), 0, d1 * math.cos(th)])
    rx = np.array([d2 * math.sin(th), 0, d2 * math.cos(th)])
    img = LAM / (4 * math.pi * (d1 + d2)) * np.exp(-1j * K * (d1 + d2))
    side = 12.0
    ns = round(side / (LAM / 6))
    pos, area = make_panel(side, side, ns, ns)
    h = k1_channel(tx, rx, pos, area, NRM, FC, 1.0)
    ratio = h / img
    assert abs(db20(ratio)) <= 0.3
    assert abs(math.degrees(np.angle(ratio))) <= 3.0
    hn = no_j(tx, rx, pos, area, NRM, FC, 1.0, allow_nonphysical=True)
    assert abs(math.degrees(np.angle(hn / img)) - (-90.0)) <= 3.0
    assert abs(hn) == pytest.approx(abs(h), rel=1e-12)


# the closed form is the Kirchhoff-obliquity disk; the option warns by design (D2)
@pytest.mark.filterwarnings("ignore:obliquity='kirchhoff':sremu.errors.ValidityWarning")
def test_ph05_disk_on_axis_discrete_vs_closed_form():
    d, radius = 2.0, 1.0
    p = np.array([0.0, 0.0, d])
    pos, area = make_panel(2 * radius, 2 * radius, 256, 256)
    disk = pos[np.hypot(pos[:, 0], pos[:, 1]) <= radius]
    h = k1_channel(p, p, disk, area, NRM, FC, 1.0, obliquity="kirchhoff")
    exact = disk_kirchhoff_onaxis(d, radius, FC)
    assert abs(db20(h) - db20(exact)) <= 0.01


def test_ph05_disk_closed_form_tends_to_image_oracle():
    """Consistency of the closed-form oracle itself (no engine code involved)."""
    d = 2.0
    img = LAM / (4 * math.pi * 2 * d) * np.exp(-1j * K * 2 * d)
    exact = disk_kirchhoff_onaxis(d, 200.0, FC)
    assert abs(db20(exact / img)) <= 0.05
    assert abs(math.degrees(np.angle(exact / img))) <= 2.0
