"""Shared helpers of the WP7 tests (API, compat, recording, replay, report).

* :class:`RecFakeEngine`: ``sremu.testing.FakeEngine`` plus ``tx_head(h)``, the optional
  engine method that Device-level recording uses to place an untimed TX burst start
  (contract change request WP7-1; WP0 owns FakeEngine).
* :func:`make_engine` / :func:`replay_engine`: a FakeEngine with one 8-element panel whose
  configuration is visible in the RX output through a complex, element-dependent gain, so
  every state change (including element subsets and ``set_enabled``) changes the samples.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from sremu.contracts import PanelConfig, RisCapabilities, StreamHandle
from sremu.testing import FakeEngine

N_EL = 8
WEIGHTS = np.exp(1j * (0.7 + 1.3 * np.arange(N_EL))) / N_EL  # fixed, distinct per element


def caps() -> dict[str, RisCapabilities]:
    """One 2 x 4 binary panel ``ris0``."""
    return {
        "ris0": RisCapabilities(
            n_elements=N_EL,
            layout=(4, 2),
            control_granularity="element",
            states=("0", "1"),
            gamma_source="analytic:test",
            operating_band_hz=(3.0e9, 4.0e9),
            max_bandwidth_hz=1.0e6,
        )
    }


def gain(cfg: Mapping[str, PanelConfig]) -> complex:
    """``1 + sum_n w_n s_n`` while the panel is enabled, else ``1``."""
    c = cfg["ris0"]
    if not c.enabled:
        return 1.0 + 0.0j
    return complex(1.0 + np.sum(WEIGHTS * c.state.astype(np.float64)))


class RecFakeEngine(FakeEngine):
    """FakeEngine with the CCR WP7-1 ``tx_head`` method."""

    def tx_head(self, h: StreamHandle) -> int:
        """First absolute index after the last accepted sample of TX stream ``h``."""
        return int(self._tx[h.id].head)


def make_engine(*, with_head: bool = True) -> FakeEngine:
    """A fresh engine with :func:`caps` and :func:`gain`."""
    cls = RecFakeEngine if with_head else FakeEngine
    return cls(caps(), gain_fn=gain)


def replay_engine(run_dir: Path) -> FakeEngine:
    """Engine factory for ``replay`` / ``sremu replay --engine``."""
    return make_engine()


class ToggleController:
    """In-process barrier controller: at barrier m it sets every element of ``panel`` to
    ``m % 2`` (cells ``n_el``), ``latency_s`` after the barrier (period ``period_s``)."""

    def __init__(
        self,
        panel: str = "ris0",
        n_el: int = N_EL,
        period_s: str = "1e-3",
        latency_s: str = "1e-4",
        name: str = "toggle",
    ) -> None:
        from fractions import Fraction

        from sremu.contracts import LatencyModel, ObservationSpec

        self.name = name
        self.panel = panel
        self.n_el = n_el
        self.period_s = Fraction(period_s)
        self.min_latency_s = Fraction(latency_s)
        self.spec = ObservationSpec(window_s=Fraction(latency_s))
        self.latency = LatencyModel(fixed_s=Fraction(latency_s))
        self.m = 0

    def setup(self, caps, rng) -> None:
        """Nothing to prepare."""

    def decide(self, obs):
        """One full-panel action per barrier."""
        from sremu.contracts import Action

        state = np.full(self.n_el, self.m % 2, dtype=np.int16)
        self.m += 1
        return [Action(self.panel, states=state)]
