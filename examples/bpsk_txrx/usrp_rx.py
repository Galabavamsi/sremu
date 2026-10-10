"""v1 BPSK receiver ported to sremu (DESIGN.md 8.2; acceptance IO-07).

Port of ``ris_sim/external_traffic/usrp_rx.py`` (v1): receive 2389 samples at 5880 S/s on
the carrier ``2.4e9 - 1800 - 800 - 450`` Hz at ``node_2``. v1 wrote a request with
``recieve_from_simulator``; the port calls :func:`sremu.compat.Receive_from_sim`, which
returns the samples (complex64).
"""

from __future__ import annotations

import numpy as np

from sremu.compat import Receive_from_sim

CENTER_FREQ = 2.4e9 - 1800 - 800 - 450
SAMPLING = 5880.0
NUM_SAMPS = 2389
NODE = "node_2"


def main() -> np.ndarray:
    """Receive the v1 packet at ``node_2``."""
    y = Receive_from_sim(NODE, NUM_SAMPS, CENTER_FREQ, SAMPLING)
    print(f"received {y.size} samples at {NODE}.rx0")
    return y


if __name__ == "__main__":
    received = main()
