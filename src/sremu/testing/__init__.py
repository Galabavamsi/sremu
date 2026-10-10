"""Test doubles shared by all work packages (WP0).

* :class:`FakePropagation` - analytic LOS-only ``PropagationModel`` (constant-velocity
  nodes, optional simplified RIS panels).
* :class:`FakeEngine` - identity-channel ``EngineProtocol`` with the exact time base.
* :mod:`sremu.testing.scenes` - small scene builders.

Layering rule: this package imports only the foundation layer (units, timebase, errors,
contracts), and no production module may import it.
"""

from sremu.testing.fake_engine import FakeEngine
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation

__all__ = ["FakeEngine", "FakeNode", "FakePanel", "FakePropagation"]
