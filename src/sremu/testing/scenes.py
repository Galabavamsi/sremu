"""Small scene builders for tests (WP0). All geometry is SIM, never the lab's.

Each builder returns a :class:`FakePropagation`; stream ids follow ``<node>.<stream>``.
"""

from __future__ import annotations

from fractions import Fraction

from sremu.contracts import StreamSpec
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation

__all__ = [
    "doppler_t9",
    "multi_rate_streams",
    "near_far_rx",
    "stream",
    "two_node_static",
]


def stream(
    sid: str, direction: str, fc_hz: float = 3.5e9, rate_sps: str | int | Fraction = "1e6"
) -> StreamSpec:
    """Shorthand for a :class:`StreamSpec` (rates as exact strings or ints)."""
    return StreamSpec(sid, direction, fc_hz, rate_sps)  # type: ignore[arg-type]


def two_node_static(
    d_m: float = 10.0, fc_hz: float = 3.5e9, rate_sps: str | int | Fraction = "1e6"
) -> FakePropagation:
    """TX ``ap.tx0`` at the origin and RX ``mu.rx0`` at (d, 0, 0), both static.

    PH-01 geometry with the defaults: |h| = lambda/(4 pi d) = -63.329 dB at 3.5 GHz, 10 m.
    """
    return FakePropagation(
        [
            FakeNode("ap", (0.0, 0.0, 0.0), streams=(stream("ap.tx0", "tx", fc_hz, rate_sps),)),
            FakeNode("mu", (d_m, 0.0, 0.0), streams=(stream("mu.rx0", "rx", fc_hz, rate_sps),)),
        ]
    )


def doppler_t9(
    with_panel: bool = False, rate_sps: str | int | Fraction = "1e4", tiles: int = 1
) -> FakePropagation:
    """The T9 geometry of ``pm_checks.py`` (PH-17): 3.5 GHz, TX at the origin, RX starting
    at (50, 0, 0) and approaching at 30 m/s along -x (LOS Doppler +350.242 Hz).

    ``with_panel`` adds a 0.5 m x 0.5 m, 12 x 12 simplified panel at (25, 10, 0) facing -y
    with ``tiles x tiles`` tiles.
    """
    fc = 3.5e9
    nodes = [
        FakeNode("tx", (0.0, 0.0, 0.0), streams=(stream("tx.tx0", "tx", fc, rate_sps),)),
        FakeNode(
            "rx",
            (50.0, 0.0, 0.0),
            (-30.0, 0.0, 0.0),
            streams=(stream("rx.rx0", "rx", fc, rate_sps),),
        ),
    ]
    panels = []
    if with_panel:
        pitch = 0.5 / 12
        panels.append(
            FakePanel(
                "ris0",
                (25.0, 10.0, 0.0),
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
                12,
                12,
                pitch,
                pitch,
                tiles_u=tiles,
                tiles_v=tiles,
            )
        )
    return FakePropagation(nodes, panels, horizon_s=1)


def near_far_rx(
    rate_sps: str | int | Fraction = "23.04e6", far_m: float = 2000.0, fc_hz: float = 3.5e9
) -> FakePropagation:
    """RT-05 geometry: a panel at the origin facing +x, TX ``ap.tx0`` 2 m away, a near RX
    ``rx1.rx0`` 1.2 m away and a far RX ``rx2.rx0`` at ``far_m`` along the normal.

    At 23.04 MS/s the far RX sees a switch about 153.7 samples after the panel.
    """
    nodes = [
        FakeNode("ap", (1.7320508, -1.0, 0.0), streams=(stream("ap.tx0", "tx", fc_hz, rate_sps),)),
        FakeNode("rx1", (1.2, 0.0, 0.0), streams=(stream("rx1.rx0", "rx", fc_hz, rate_sps),)),
        FakeNode("rx2", (far_m, 0.0, 0.0), streams=(stream("rx2.rx0", "rx", fc_hz, rate_sps),)),
    ]
    pitch = 0.0428
    panel = FakePanel(
        "ris0",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        24,
        5,
        pitch,
        pitch,
        gamma=(1 + 0j, -1 + 0j),
        tiles_u=3,
    )
    return FakePropagation(nodes, [panel], horizon_s=10)


def multi_rate_streams() -> FakePropagation:
    """Three non-overlapping links at 5880 S/s, 23.04 MS/s and 30.72 MS/s (RT-01 rates)."""
    rates = (("a", "5880", 1.0e9), ("b", "23.04e6", 2.0e9), ("c", "30.72e6", 3.0e9))
    nodes = []
    for i, (tag, rate, fc) in enumerate(rates):
        nodes.append(
            FakeNode(
                f"tx{tag}", (0.0, float(i), 0.0), streams=(stream(f"tx{tag}.tx0", "tx", fc, rate),)
            )
        )
        nodes.append(
            FakeNode(
                f"rx{tag}", (5.0, float(i), 0.0), streams=(stream(f"rx{tag}.rx0", "rx", fc, rate),)
            )
        )
    return FakePropagation(nodes)
