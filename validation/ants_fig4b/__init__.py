"""V-LAB: pre-registered consistency check against the ANTS Fig. 4(b) digitized map.

DESIGN.md section 10.9 (WP10). Blinded phase: nothing in this package reads the digitized
measurements except :func:`validation.ants_fig4b.protocol_lock.load_measured`, which refuses
to run until the protocol hash is locked (``PREREGISTERED_SHA256``) and matches
``PROTOCOL.md`` + ``protocol.json``.

Modules:

* :mod:`.protocol`: loads ``protocol.json``; cell grid, splits and folds.
* :mod:`.protocol_lock`: the hash lock and ``load_measured()``.
* :mod:`.prior`: prior draws (keyed RNG), lab geometry hypotheses, sremu scenarios.
* :mod:`.predict`: the prediction tensor ``pred[K, 30, models]`` (sremu, cached npz).
* :mod:`.analysis`: the protocol pipeline (fits, metrics, bootstrap, decisions C1-C3).
* :mod:`.decoy`: ``synthetic_decoy(seed)`` and the decoy power analysis.
* :mod:`.run`: the unblinded run (primary, secondary, sensitivity; F4 data, Table V).
"""
