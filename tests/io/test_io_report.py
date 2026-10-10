"""``sremu report``: static HTML report renders (DESIGN.md 2.5, 8.6; WP7 acceptance).

Oracles: the report must be a single self-contained document with no JavaScript and no
external resources, every value from the run directory HTML-escaped, the event table and
manifest fields present, and the plotted points must invert (through the axis ranges the
``<svg>`` declares) to the windowed power computed independently from the raw samples.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from html.parser import HTMLParser

import numpy as np
import pytest
from typer.testing import CliRunner

from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TimeSpec, TxMetadata
from sremu.cli import build_app
from sremu.contracts import CommandRecord, StreamSpec
from sremu.record.report import PLOT_FLOOR_DB, power_trace, render_report, write_report
from sremu.record.sigmf import RunRecorder
from tests.io import _wp7_support as sup

pytestmark = pytest.mark.io

VOID = {"meta", "br", "hr", "img", "input", "link", "col", "area", "base", "wbr"}
SVG_SELF = {"rect", "line", "polyline", "path", "circle"}


class _Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.attrs: list[tuple[str, str, str | None]] = []
        self.svgs: list[dict[str, str | None]] = []
        self.polylines: list[list[str]] = []
        self.cur_svg_lines: list[str] = []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            self.attrs.append((tag, k, v))
        if tag == "svg":
            self.svgs.append(dict(attrs))
            self.polylines.append([])
        if tag == "polyline":
            self.polylines[-1].append(dict(attrs)["points"])
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        for k, v in attrs:
            self.attrs.append((tag, k, v))
        if tag == "polyline":
            self.polylines[-1].append(dict(attrs)["points"])

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
            return
        self.stack.pop()


def _parse(html: str) -> _Tags:
    p = _Tags()
    p.feed(html)
    p.close()
    return p


def _record_run(run, rng, *, cmd_label="api"):
    dev = Device.from_engine(sup.make_engine(), record=run, src=cmd_label)
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", 3.5e9, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", 3.5e9, "1e6"))
    x = (0.5 * np.exp(2j * np.pi * np.arange(4000) / 50)).astype(np.complex64)
    tx.send(x, TxMetadata(True, TimeSpec(0, 0.0), True, True))
    r = dev.ris("ris0").set_config(states=np.ones(8, np.int16), time_spec=TimeSpec(0, 2e-3))
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=4000))
    out = np.zeros(4000, np.complex64)
    got = 0
    while got < 4000:
        got += rx.recv(out[got:], RxMetadata(), timeout=0.0)
    dev.close()
    return x, out, r


def test_report_power_trace_closed_forms():
    a = 0.5
    x = np.full(1000, a * np.exp(1j * 0.3), np.complex64)
    starts, db, w = power_trace(x, max_points=100)
    assert w == 10
    assert starts.tolist() == list(range(0, 1000, 10))
    z = complex(np.complex64(a * np.exp(1j * 0.3)))  # the stored complex64 value, exactly
    np.testing.assert_allclose(db, 10 * math.log10(z.real**2 + z.imag**2), rtol=0, atol=1e-12)
    assert abs(db[0] - 20 * math.log10(a)) < 1e-6  # = -6.0206 dB up to complex64 rounding
    # partial last window, and an all-zero window plots at the floor
    y = np.concatenate([np.zeros(10, np.complex64), np.ones(5, np.complex64) * 2])
    starts, db, w = power_trace(y, max_points=2)  # window ceil(15/2) = 8
    assert w == 8
    assert starts.tolist() == [0, 8]
    assert db[0] == PLOT_FLOOR_DB
    assert db[1] == pytest.approx(10 * math.log10((5 * 4) / 7), abs=1e-12)  # 2 zeros + 5 x |2|^2
    assert power_trace(np.zeros(0, np.complex64))[0].size == 0


def test_report_renders_static_html_with_events_plots_and_checks(tmp_path, rng):
    run = tmp_path / "run"
    _x, _y, receipt = _record_run(run, rng)
    html = render_report(run)
    p = _parse(html)
    assert p.errors == []
    assert p.stack == []
    low = html.lower()
    assert "<script" not in low
    assert "javascript:" not in low
    assert not [a for a in p.attrs if a[1].startswith("on")]  # no event handlers
    assert not [a for a in p.attrs if a[1] in ("src", "href")]  # no external resources
    assert html.startswith("<!DOCTYPE html>")
    assert "SigMF structural check (IO-05): no problems" in html
    assert f"<code>{receipt.cmd_id}</code>" in html
    assert "scheduled &gt; applied" in html  # status history of the command
    assert "no manifest.json" in html
    # one plot per recording; RX has one dashed marker at the switch
    assert len(p.svgs) == 2
    assert html.count('class="marker"') == 1


@pytest.mark.parametrize("which", [0, 1])
def test_report_plotted_points_invert_to_the_windowed_power(tmp_path, rng, which):
    run = tmp_path / "run"
    x, y, _ = _record_run(run, rng)
    html = render_report(run, max_points=200)
    p = _parse(html)
    svg = p.svgs[which]
    t_min, t_max = float(svg["data-t-min"]), float(svg["data-t-max"])
    p_min, p_max = float(svg["data-p-min"]), float(svg["data-p-max"])
    ml, mt, pw, ph = (float(v) for v in svg["data-plot"].split())
    (pts,) = p.polylines[which]
    xy = np.array([[float(a) for a in s.split(",")] for s in pts.split()])
    t = t_min + (xy[:, 0] - ml) / pw * (t_max - t_min)
    pv = p_max - (xy[:, 1] - mt) / ph * (p_max - p_min)
    sig = x if which == 0 else y
    w = math.ceil(sig.size / 200)
    want_db = [
        10 * math.log10(np.mean(np.abs(sig[i : i + w].astype(np.complex128)) ** 2))
        for i in range(0, sig.size, w)
    ]
    want_t = [i / 1e6 for i in range(0, sig.size, w)]
    tol_db = 0.0006 / ph * (p_max - p_min)  # points are written with 3 decimals
    np.testing.assert_allclose(pv, want_db, rtol=0, atol=tol_db)
    np.testing.assert_allclose(t, want_t, rtol=0, atol=0.0006 / pw * (t_max - t_min))
    if which == 1:  # the switch at 2 ms changes the RX power by |g|^2
        g = sup.gain({"ris0": _cfg(np.ones(8))})
        k = 2000 // w
        assert pv[k + 1] - pv[k - 1] == pytest.approx(20 * math.log10(abs(g)), abs=1e-2)


def _cfg(s):
    from sremu.contracts import PanelConfig

    return PanelConfig("ris0", np.asarray(s, np.int16))


def test_report_escapes_everything(tmp_path):
    run = tmp_path / "evil"
    rec = RunRecorder(run)
    rec.open_stream(StreamSpec("mu.rx0", "rx", 3.5e9, Fraction(1000)))
    rec.write("mu.rx0", 0, np.ones(10, np.complex64))
    bad = "<script>alert(1)</script>"
    rec.on_command(
        CommandRecord(
            cmd_id=bad,
            seq=1,
            src='"><img src=x onerror=alert(1)>',
            op="ris.set_config",
            panel="ris0",
            t_req=Fraction(0),
            t_applied=Fraction(0),
            k_req={"mu.rx0": 0},
            k_applied={"mu.rx0": 0},
            lateness_samples={"mu.rx0": 0},
            t_wall_recv_ns=None,
            t_wall_computed_ns=None,
            t_wall_visible_ns=None,
            status="applied",
            config_hash="0" * 32,
            transition=None,
        )
    )
    rec.close()
    (run / "manifest.json").write_text(
        json.dumps({"scenario_name": "<b>x</b>", "seed": 3, "counters": {"late<x>": 1}}),
        encoding="utf-8",
    )
    html = render_report(run)
    p = _parse(html)
    assert p.errors == []
    assert "<script" not in html.lower()
    assert "<img" not in html.lower()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;x&lt;/b&gt;" in html
    assert not [a for a in p.attrs if a[1].startswith("on")]


def test_report_manifest_and_metrics_sections(tmp_path):
    run = tmp_path / "m"
    run.mkdir()
    man = {
        "scenario_name": "quickstart",
        "seed": 1,
        "backend": "numba_cpu",
        "mode": "lockstep",
        "git_commit": "abc123",
        "git_dirty": False,
        "versions": {"numpy": "2.5.3"},
        "counters": {"late_commands": 2, "overflow_pages": 0},
    }
    (run / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"t_s": i / 10, "rtf": 1.5 + i}) for i in range(3)) + "\n",
        encoding="utf-8",
    )
    html = render_report(run)
    assert _parse(html).errors == []
    for needle in (
        "<th>scenario_name</th><td>quickstart</td>",
        "<th>backend</th><td>numba_cpu</td>",
        "<th>late_commands</th><td>2</td>",
        "<th>numpy</th><td>2.5.3</td>",
        "3 windows",
        "no SigMF recordings",
        "no events.jsonl entries",
    ):
        assert needle in html


def test_report_cli(tmp_path, rng):
    run = tmp_path / "run"
    _record_run(run, rng)
    runner = CliRunner()
    app = build_app()
    out = tmp_path / "r.html"
    res = runner.invoke(app, ["report", str(run), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert out.read_text(encoding="utf-8") == render_report(run)
    res = runner.invoke(app, ["report", str(run)])
    assert res.exit_code == 0
    assert (run / "report.html").is_file()
    assert runner.invoke(app, ["report", str(tmp_path / "nope")]).exit_code == 2
    assert write_report(run, tmp_path / "x" / "y.html").is_file()
    assert re.search(r"<svg [^>]*data-t-min=", out.read_text(encoding="utf-8"))
