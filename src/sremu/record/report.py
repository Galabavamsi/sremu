"""Static HTML report of a run directory (DESIGN.md sections 2.5, 8.6).

``sremu report RUN_DIR -o r.html`` writes one self-contained HTML file: inline CSS and
inline SVG plots, **no JavaScript, no external resources, no server**. Every value read from
the run directory is HTML-escaped. Sections (each skipped when its input is absent):

* **Run**: key manifest fields (versions, git state, scenario, seed, mode, backend,
  determinism, timing), the I6 counters and the full ``manifest.json`` in a collapsible
  ``<details>`` block.
* **Streams**: one row per SigMF recording (direction, exact rate, carrier, samples,
  capture segments, annotations) and the IO-05 structural check of the recordings.
* **Power plots**: per recording, windowed power ``10*log10(mean |x|^2)`` in dB (dBm with
  the default full-scale mapping of section 3.3) against absolute stream time; each capture
  segment is its own polyline (gaps are not bridged); RX plots mark every annotated command
  with a dashed line at its ``k_applied``. Each ``<svg>`` carries its axis ranges as
  ``data-t-min``, ``data-t-max``, ``data-p-min`` and ``data-p-max`` attributes so the plotted
  points can be checked against the data.
* **Events**: the final state of every command in ``events.jsonl`` with its status history.
* **Metrics**: the windows of ``metrics.jsonl`` (first 200 rows).
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np

from sremu.record.eventlog import EVENTS_NAME, LoggedCommand, read_events
from sremu.record.sigmf import COLLECTION_NAME, Recording, list_recordings, validate_collection
from sremu.timebase import fraction_to_str

__all__ = [
    "PLOT_FLOOR_DB",
    "power_trace",
    "render_report",
    "svg_power_plot",
    "write_report",
]

PLOT_FLOOR_DB: Final[float] = -200.0
"""Power plotted for an all-zero window (log of zero), dB."""

_W: Final[int] = 760
_H: Final[int] = 260
_ML, _MR, _MT, _MB = 64, 16, 14, 40  # plot margins, px
_LINE: Final[str] = "#0072B2"  # Okabe-Ito blue
_MARK: Final[str] = "#D55E00"  # Okabe-Ito vermillion
_MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "scenario_name",
    "sremu_version",
    "contract_version",
    "git_commit",
    "git_dirty",
    "scenario_sha256",
    "seed",
    "rng_scheme",
    "mode",
    "coupling",
    "backend",
    "threads",
    "deterministic",
    "numba_disable_svml",
    "platform",
    "cpu_model",
    "power_plan_or_governor",
    "t_start_utc",
    "t_end_utc",
    "wall_s",
    "verify_interp_max_rel",
    "richardson_db",
    "nondeterministic_inputs",
    "oracle_csi_used",
)

_CSS: Final[str] = """
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px;color:#1b1b1b;
background:#fff;max-width:1100px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.15rem;margin:28px 0 8px;
border-bottom:1px solid #ddd;padding-bottom:4px}h3{font-size:1rem;margin:16px 0 6px}
table{border-collapse:collapse;font-size:.85rem;margin:6px 0}
th,td{border:1px solid #d6d6d6;padding:3px 7px;text-align:left;vertical-align:top}
th{background:#f3f3f3}td.num{text-align:right;font-variant-numeric:tabular-nums}
code,pre{font-family:ui-monospace,Consolas,monospace;font-size:.8rem}
pre{background:#f7f7f7;padding:8px;overflow:auto;max-height:420px}
.ok{color:#007a3d}.bad{color:#b00020}.muted{color:#666}
.scroll{overflow-x:auto}svg{display:block;margin:4px 0 12px;max-width:100%;height:auto}
table{display:block;overflow-x:auto;max-width:100%}p code{overflow-wrap:anywhere}
"""


def _e(x: object) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, Mapping):
        return ", ".join(f"{k}: {v}" for k, v in x.items())
    if isinstance(x, list):
        return ", ".join(str(v) for v in x)
    return "" if x is None else str(x)


def _time_cell(t: Fraction | None) -> str:
    if t is None:
        return "<td></td>"
    return f'<td class="num" title="{_e(fraction_to_str(t))} s">{float(t):.9g}</td>'


# =========================================================================== power traces


def power_trace(x: np.ndarray, *, max_points: int = 800) -> tuple[np.ndarray, np.ndarray, int]:
    """Windowed power of ``x``: ``(starts, p_db, window)``.

    ``window = max(1, ceil(n/max_points))`` samples; window i covers samples
    ``[i*window, min(n, (i+1)*window))`` and ``p_db[i] = 10*log10(mean |x|^2)`` (complex128),
    or :data:`PLOT_FLOOR_DB` for an all-zero window; ``starts`` are the window start indices.
    """
    n = int(x.size)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0), 1
    w = max(1, math.ceil(n / max(1, max_points)))
    starts = np.arange(0, n, w, dtype=np.int64)
    p2 = np.abs(x.astype(np.complex128)) ** 2
    sums = np.add.reduceat(p2, starts)
    counts = np.diff(np.append(starts, n))
    p = sums / counts
    with np.errstate(divide="ignore"):
        db = np.where(p > 0, 10.0 * np.log10(np.where(p > 0, p, 1.0)), PLOT_FLOOR_DB)
    return starts, db, w


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not hi > lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    out = []
    t = first
    while t <= hi + 1e-12 * abs(step):
        out.append(round(t, 12))
        t += step
    return out


def svg_power_plot(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    markers: Iterable[tuple[float, str]] = (),
    title: str = "",
) -> str:
    """Inline SVG: one polyline per ``(t_s, p_db)`` segment, dashed markers ``(t_s, label)``.

    The axis ranges are written as ``data-*`` attributes (module docstring); x maps
    ``t -> ML + (t - t_min)/(t_max - t_min)*plot_w`` and y maps
    ``p -> MT + (p_max - p)/(p_max - p_min)*plot_h``.
    """
    segs = [(np.asarray(t, float), np.asarray(p, float)) for t, p in segments if len(t)]
    marks = list(markers)
    if not segs:
        return '<p class="muted">no samples</p>'
    t_all = np.concatenate([t for t, _ in segs] + [np.array([m for m, _ in marks], float)])
    p_all = np.concatenate([p for _, p in segs])
    t_min, t_max = float(t_all.min()), float(t_all.max())
    if t_max <= t_min:
        t_max = t_min + 1.0
    p_min, p_max = float(p_all.min()), float(p_all.max())
    pad = max(1.0, 0.05 * (p_max - p_min))
    p_min, p_max = math.floor(p_min - pad), math.ceil(p_max + pad)
    pw, ph = _W - _ML - _MR, _H - _MT - _MB

    def fx(t: float) -> float:
        return _ML + (t - t_min) / (t_max - t_min) * pw

    def fy(p: float) -> float:
        return _MT + (p_max - p) / (p_max - p_min) * ph

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}" role="img" aria-label="{_e(title)}" '
        f'data-t-min="{t_min!r}" data-t-max="{t_max!r}" data-p-min="{p_min!r}" '
        f'data-p-max="{p_max!r}" data-plot="{_ML} {_MT} {pw} {ph}">',
        f"<title>{_e(title)}</title>",
        f'<rect x="{_ML}" y="{_MT}" width="{pw}" height="{ph}" fill="#fff" stroke="#999"/>',
    ]
    for tv in _ticks(t_min, t_max):
        x = fx(tv)
        parts.append(
            f'<line x1="{x:.3f}" y1="{_MT + ph}" x2="{x:.3f}" y2="{_MT + ph + 4}" stroke="#666"/>'
            f'<text x="{x:.3f}" y="{_MT + ph + 16}" font-size="10" text-anchor="middle">'
            f"{tv:.6g}</text>"
        )
    for pv in _ticks(p_min, p_max):
        y = fy(pv)
        parts.append(
            f'<line x1="{_ML - 4}" y1="{y:.3f}" x2="{_ML + pw}" y2="{y:.3f}" stroke="#eee"/>'
            f'<text x="{_ML - 6}" y="{y + 3:.3f}" font-size="10" text-anchor="end">{pv:.6g}</text>'
        )
    parts.append(
        f'<text x="{_ML + pw / 2:.1f}" y="{_H - 6}" font-size="11" text-anchor="middle">'
        "time (s)</text>"
        f'<text x="12" y="{_MT + ph / 2:.1f}" font-size="11" text-anchor="middle" '
        f'transform="rotate(-90 12 {_MT + ph / 2:.1f})">power (dB)</text>'
    )
    for t, p in segs:
        pts = " ".join(f"{fx(a):.3f},{fy(b):.3f}" for a, b in zip(t, p, strict=True))
        parts.append(
            f'<polyline class="power" fill="none" stroke="{_LINE}" stroke-width="1.2" '
            f'points="{pts}"/>'
        )
    for tm, label in marks:
        x = fx(float(tm))
        parts.append(
            f'<line class="marker" x1="{x:.3f}" y1="{_MT}" x2="{x:.3f}" y2="{_MT + ph}" '
            f'stroke="{_MARK}" stroke-dasharray="4 3"><title>{_e(label)}</title></line>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _recording_plot(r: Recording, max_points: int) -> str:
    segs = []
    for c in r.captures:
        x = r.data[c.sample_start : c.sample_start + c.length]
        starts, db, _w = power_trace(x, max_points=max(1, max_points // max(1, len(r.captures))))
        t = np.array([float(Fraction(c.global_index + int(s)) / r.rate_sps) for s in starts])
        segs.append((t, db))
    marks = []
    for a in r.meta.get("annotations", []):
        k = a.get("sremu:k_applied")
        if k is None:
            continue
        marks.append(
            (
                float(Fraction(int(k)) / r.rate_sps),
                f"{a.get('sremu:cmd_id', '')} {a.get('core:label', '')} "
                f"k={k} {a.get('sremu:status', '')}",
            )
        )
    return svg_power_plot(segs, markers=marks, title=f"{r.direction.upper()} {r.stream} power")


# =========================================================================== sections


def _kv_table(rows: Iterable[tuple[str, object]]) -> str:
    body = "".join(f"<tr><th>{_e(k)}</th><td>{_e(_fmt(v))}</td></tr>" for k, v in rows)
    return f"<table>{body}</table>"


def _run_section(root: Path) -> str:
    p = root / "manifest.json"
    if not p.is_file():
        return '<h2>Run</h2><p class="muted">no manifest.json in this run directory</p>'
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f'<h2>Run</h2><p class="bad">manifest.json unreadable: {_e(exc)}</p>'
    rows = [(k, m.get(k)) for k in _MANIFEST_KEYS if k in m]
    out = ["<h2>Run</h2>", _kv_table(rows)]
    ver = m.get("versions")
    if isinstance(ver, Mapping):
        out += ["<h3>Versions</h3>", _kv_table(sorted(ver.items()))]
    cnt = m.get("counters")
    if isinstance(cnt, Mapping):
        out += ["<h3>Counters (I6)</h3>", _kv_table(sorted(cnt.items()))]
    full = json.dumps(m, indent=2, ensure_ascii=False)
    out.append(f"<details><summary>manifest.json</summary><pre>{_e(full)}</pre></details>")
    return "".join(out)


def _streams_section(root: Path, recs: list[Recording], max_points: int) -> str:
    if not recs:
        return '<h2>Streams</h2><p class="muted">no SigMF recordings</p>'
    rows = []
    for r in recs:
        rows.append(
            "<tr>"
            f"<td>{_e(r.open_index)}</td><td><code>{_e(r.stream)}</code></td>"
            f"<td>{_e(r.direction)}</td><td class=num>{_e(fraction_to_str(r.rate_sps))}</td>"
            f"<td class=num>{r.fc_hz:.10g}</td><td class=num>{r.data.size}</td>"
            f"<td class=num>{len(r.captures)}</td>"
            f"<td class=num>{len(r.meta.get('annotations', []))}</td></tr>"
        )
    head = (
        "<tr><th>#</th><th>stream</th><th>dir</th><th>rate (S/s)</th><th>f_c (Hz)</th>"
        "<th>samples</th><th>captures</th><th>annotations</th></tr>"
    )
    out = ["<h2>Streams</h2>", f'<div class="scroll"><table>{head}{"".join(rows)}</table></div>']
    if (root / COLLECTION_NAME).is_file():
        probs = validate_collection(root / COLLECTION_NAME)
        if probs:
            items = "".join(f"<li>{_e(x)}</li>" for x in probs)
            out.append(f'<p class="bad">SigMF structural check: {len(probs)} problem(s)</p>')
            out.append(f"<ul>{items}</ul>")
        else:
            out.append('<p class="ok">SigMF structural check (IO-05): no problems</p>')
    out.append("<h2>Power plots</h2>")
    for r in recs:
        out.append(f"<h3>{_e(r.direction.upper())} <code>{_e(r.stream)}</code></h3>")
        out.append(_recording_plot(r, max_points))
    return "".join(out)


def _events_section(cmds: list[LoggedCommand]) -> str:
    if not cmds:
        return '<h2>Events</h2><p class="muted">no events.jsonl entries</p>'
    head = (
        "<tr><th>seq</th><th>cmd_id</th><th>src</th><th>op</th><th>target</th>"
        "<th>t_req (s)</th><th>t_applied (s)</th><th>k_req</th><th>k_applied</th>"
        "<th>lateness</th><th>status</th><th>history</th><th>config_hash</th>"
        "<th>counterfactual</th></tr>"
    )
    rows = []
    for lc in cmds:
        r = lc.record
        cls = "bad" if r.status in ("rejected",) else ""
        rows.append(
            "<tr>"
            f"<td class=num>{r.seq}</td><td><code>{_e(r.cmd_id)}</code></td><td>{_e(r.src)}</td>"
            f"<td>{_e(r.op)}</td><td>{_e(r.panel)}</td>"
            f"{_time_cell(r.t_req)}{_time_cell(r.t_applied)}"
            f"<td>{_e(_fmt(r.k_req))}</td><td>{_e(_fmt(r.k_applied))}</td>"
            f"<td>{_e(_fmt(r.lateness_samples))}</td>"
            f'<td class="{cls}">{_e(r.status)}</td><td>{_e(" > ".join(lc.history))}</td>'
            f"<td><code>{_e(r.config_hash[:12])}</code></td><td>{_e(r.counterfactual)}</td></tr>"
        )
    return f'<h2>Events</h2><div class="scroll"><table>{head}{"".join(rows)}</table></div>'


def _metrics_section(root: Path, max_rows: int = 200) -> str:
    p = root / "metrics.jsonl"
    if not p.is_file():
        return ""
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    if not rows:
        return '<h2>Metrics</h2><p class="muted">metrics.jsonl is empty</p>'
    keys: list[str] = []
    for r in rows[:max_rows]:
        keys += [k for k in r if k not in keys]
    head = "".join(f"<th>{_e(k)}</th>" for k in keys)
    body = "".join(
        "<tr>" + "".join(f"<td>{_e(_fmt(r.get(k)))}</td>" for k in keys) + "</tr>"
        for r in rows[:max_rows]
    )
    note = f"<p class=muted>{len(rows)} windows</p>"
    return f'<h2>Metrics</h2>{note}<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


# =========================================================================== entry points


def render_report(run_dir: str | Path, *, max_points: int = 800, title: str | None = None) -> str:
    """The report of ``run_dir`` as one HTML document (module docstring)."""
    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"run directory {root} does not exist")
    recs = (
        list_recordings(root)
        if (root / "iq").is_dir() or (root / COLLECTION_NAME).is_file()
        else []
    )
    cmds = read_events(root / EVENTS_NAME) if (root / EVENTS_NAME).is_file() else []
    heading = title or f"sremu run {root.name}"
    body = [
        f"<h1>{_e(heading)}</h1>",
        f'<p class="muted">run directory <code>{_e(root.resolve())}</code></p>',
        _run_section(root),
        _streams_section(root, recs, max_points),
        _events_section(cmds),
        _metrics_section(root),
    ]
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(heading)}</title><style>{_CSS}</style></head>"
        f"<body>{''.join(body)}</body></html>\n"
    )


def write_report(run_dir: str | Path, out: str | Path | None = None, **kw: Any) -> Path:
    """Write the report (default ``<run_dir>/report.html``); returns its path."""
    path = Path(out) if out is not None else Path(run_dir) / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(run_dir, **kw), encoding="utf-8", newline="\n")
    return path
