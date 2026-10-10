"""Legacy kernels re-exported from :mod:`sremu.physics.legacy` (WP2; DESIGN.md section 2.2).

Every item raises ``NonPhysicalError`` unless called with ``allow_nonphysical=True``. They
exist only for the teeth tests of section 10.7.
"""

from __future__ import annotations

from sremu.physics.legacy import (
    LEGACY_ITEMS,
    WCL_Q,
    block_constant_delay,
    ellingson_qgamma,
    legacy_v1,
    legacy_v2,
    legacy_wcl_eq2,
    no_j,
)

__all__ = [
    "LEGACY_ITEMS",
    "WCL_Q",
    "block_constant_delay",
    "ellingson_qgamma",
    "legacy_v1",
    "legacy_v2",
    "legacy_wcl_eq2",
    "no_j",
]
