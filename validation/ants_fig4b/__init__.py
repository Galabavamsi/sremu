"""V-LAB: pre-registered consistency check against the ANTS Fig. 4(b) digitized map.

DESIGN.md section 10.9 (WP10). Blinded phase: nothing in this package reads the digitized
measurements except :func:`validation.ants_fig4b.protocol_lock.load_measured`, which refuses
to run until the protocol hash is locked (``PREREGISTERED_SHA256``) and matches
``PROTOCOL.md`` + ``protocol.json``.

Modules:

* :mod:`.protocol`: loads ``protocol.json``; cell grid, splits and folds.
* :mod:`.protocol_lock`: the hash lock and ``load_measured()``.
* :mod:`.prior`: prior draws (keyed RNG), the tile geometry (v2: six tiles on an arc facing
  the Tx; S-OR: the v1 O_R flat row), sremu scenarios (one panel per tile).
* :mod:`.predict`: the prediction tensor ``pred[K, 30, models]`` (sremu, cached npz).
* :mod:`.analysis`: the protocol pipeline (fits, metrics, bootstrap, decisions C1-C3).
* :mod:`.decoy`: ``synthetic_decoy(seed)`` and the decoy power analysis.
* :mod:`.run`: the unblinded run (primary, secondary, sensitivity incl. S-OR and S-NF; F4
  data, Table V).
* :mod:`.diagnostic`: which additional lab fact would raise the C2 detection rate (diagnostic,
  not part of the protocol).
"""
