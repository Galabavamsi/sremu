"""WP11 experiment drivers (DESIGN.md section 11: E-A, E-B, E-C, E-F).

Not part of the installed ``sremu`` package: these modules are study scripts that use the
public sremu APIs (``sremu.physics.propagation.from_scenario``, ``sremu.engine``,
``sremu.control``, ``sremu.analysis.stats``, ``sremu.experiments.sweep``,
``sremu.record.manifest``). Every run writes a run directory with ``manifest.json``,
CSV/JSON data and ``meta.json`` (data classes); see :mod:`experiments.common`.
"""
