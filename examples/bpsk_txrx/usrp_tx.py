"""v1 BPSK transmitter ported to sremu (DESIGN.md 8.2; acceptance IO-07).

Port of ``ris_sim/external_traffic/usrp_tx.py`` (v1). The waveform arithmetic is kept as in
v1: a 200-symbol header of ``1+1j``, 38 BPSK symbols at ``+-0.1(1+j)``, 10 samples per
symbol with a rectangular pulse (``np.convolve`` with ``ones(10)``, so 238*10 + 9 = 2389
samples), stored through float32 I/Q as v1 did. The sample rate is v1's
``(38 + 200)*10 / (0.75*(238/441))``, which evaluates to exactly 5880.0 S/s (and reaches
the engine exactly through ``Fraction(repr(x))``, R5); the carrier is
``2.4e9 - 1800 - 800 - 450`` Hz.

v1 wrote a request into ``config/nodes.json`` with ``send_to_simulator``; the port calls the
ANTS-named alias :func:`sremu.compat.Send_to_sim` on the current device (installed by
``run_pair.py``, or ``$SREMU_CONNECT`` for a served emulator).
"""

from __future__ import annotations

import numpy as np

from sremu.compat import Send_to_sim

BITS = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1] + [0] * 28  # v1's 38-bit packet
HEADER_SIZE = 200
SYMBOL_RATE = 441
SAMPLES_PER_SYMBOL = 10
PACKET_SIZE = 38  # symbols
CENTER_FREQ = 2.4e9 - 1800 - 800 - 450
NODE = "node_1"


def bits_to_symbols(bitstream: list[int]) -> np.ndarray:
    """v1 BPSK mapping: 1 -> 0.1+0.1j, 0 -> -0.1-0.1j."""
    mapping = {1: 0.1 + 0.1j, 0: -0.1 - 0.1j}
    return np.array([mapping[b] for b in bitstream], dtype=complex)


def sampling_rate() -> float:
    """v1's rate arithmetic (evaluates to exactly 5880.0)."""
    symbol_duration = 1 / SYMBOL_RATE
    time_bet_msg = (HEADER_SIZE + PACKET_SIZE) * symbol_duration
    msg_time = 0.75 * time_bet_msg
    return ((PACKET_SIZE + HEADER_SIZE) * SAMPLES_PER_SYMBOL) / msg_time


def make_waveform() -> np.ndarray:
    """The v1 transmit array, shape (1, 2389), complex64."""
    symbols = bits_to_symbols(BITS)
    header = np.tile(1 + 1j, HEADER_SIZE)
    data = np.concatenate((header, symbols))
    transmit = []
    for pkt in [data]:
        pulse = np.ones(SAMPLES_PER_SYMBOL)
        d = np.zeros(len(pkt) * SAMPLES_PER_SYMBOL, dtype=complex)
        d[::SAMPLES_PER_SYMBOL] = pkt
        x = np.convolve(d, pulse)
        t = np.zeros(len(x) * 2, dtype=np.float32)
        t[::2] = x.real
        t[1::2] = x.imag
        transmit.append(t[::2] + 1j * t[1::2])
    return np.array(transmit, dtype=np.complex64)


def main() -> int:
    """Send the v1 packet from ``node_1``; returns the number of samples accepted."""
    transmit = make_waveform()
    fs = sampling_rate()
    print(f"total samples: {transmit.shape[1]}  s_m: {fs}")
    n = Send_to_sim(NODE, transmit, CENTER_FREQ, fs)
    print(f"sent {n} samples from {NODE}.tx0")
    return n


if __name__ == "__main__":
    sent = main()
