"""WP6: the engine (DESIGN.md section 4).

Modules (import them directly; this package imports nothing heavy):

* :mod:`.emulator` -- :class:`~sremu.engine.emulator.Emulator` (``EngineProtocol``), lockstep
  and paced runs, served mode, sinks, run facts for the manifest;
* :mod:`.scheduler` -- snapshot grid, segments, render jobs, rate-group page advance (no
  wall clock);
* :mod:`.commands` -- RIS command semantics: switch time at the RIS, per-RX and per-tile
  switch samples, late policies, ties, element subsets, transitions (no wall clock);
* :mod:`.events` -- the ``(t, prio, seq)`` event heap (no wall clock);
* :mod:`.barriers` -- controller barriers, keyed latencies, observations, lookahead;
* :mod:`.streams` -- TX ingestion and commitment, RX chain and delivery, traffic sources;
* :mod:`.pacer` -- paced-mode release schedule, hybrid sleep, deadlines;
* :mod:`.batch` -- the coefficient-only (batch) path.
"""
