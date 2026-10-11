"""ST-04 (DESIGN.md section 10.5): BER vs Eb/N0 through the emulator (E-F reduced).

Lab-like LOS + RIS with a fixed configuration (RX1's exact optimum), BPSK and QPSK at 4
samples/symbol, Eb/N0 in {0, 2, 4, 6, 8, 10} dB set through the TX power against the engine's
physical noise (NF 7 dB), coherent detection with the scene model's channel (genie). Each point
runs until >= 100 bit errors or >= 2e6 data bits. Accept: Q(sqrt(2 Eb/N0)) inside the
Clopper-Pearson 99 % interval at every point.
"""

from __future__ import annotations

import pytest
from experiments import ef

pytestmark = pytest.mark.statistical


@pytest.mark.parametrize("mod", ["bpsk", "qpsk"])
@pytest.mark.parametrize("ebn0_db", [0, 2, 4, 6, 8, 10])
def test_st04_ber_theory_inside_cp99(
    mod: str, ebn0_db: int, tmp_path, monkeypatch, record_property
) -> None:
    monkeypatch.setattr(ef, "MIN_ERR", 100)
    monkeypatch.setattr(ef, "MAX_BITS", 2_000_000)
    r = ef.run_ber_point(mod, float(ebn0_db), tmp_path / f"{mod}_{ebn0_db}", chunk_s="0.5")
    lo, hi = r["genie_ci99"]
    record_property("ST04_bits", r["bits"])
    record_property("ST04_errors", r["genie_errors"])
    record_property("ST04_ber", r["genie_ber"])
    record_property("ST04_theory", r["theory_ber"])
    assert r["genie_errors"] >= 100 or r["bits"] >= 2_000_000
    assert lo <= r["theory_ber"] <= hi
