"""Scenario loading, formats, round trips, resolution and hashing (DESIGN.md 3.1, 3.6, 9.1).

Oracles: the YAML 1.2 core schema rules; the same scenario written in YAML, JSON and TOML
must give the same model and the same ``scenario_sha256``; ``scenario_sha256`` equals
SHA-256 of the canonical JSON computed here with ``json.dumps(sort_keys=True,
separators=(",", ":"))``; dumping and reloading is the identity.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from fractions import Fraction
from pathlib import Path

import pytest

from sremu.errors import SchemaError, ValidityWarning
from sremu.scenario.load import (
    canonical_json,
    dump_yaml,
    load_scenario,
    parse_document,
    parse_scenario,
    read_document,
    resolve_scenario,
    scenario_sha256,
    sha256_file,
)
from sremu.scenario.models import FloatNotExactError, Scenario
from sremu.timebase import Rate

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
QUICK = REPO / "scenarios" / "quickstart.yaml"
LAB = REPO / "scenarios" / "lab_like_A.yaml"

QUICK_TOML = """
schema_version = "1.0"
name = "quickstart"
seed = 1
description = "Minimal static LOS + RIS scenario for the README quickstart"

[[nodes]]
id = "ap"
position_m = [4.330127019, -2.5, 0.0]
antenna = { pattern = "isotropic" }

[[nodes.streams]]
name = "tx0"
direction = "tx"
fc_hz = 3.5e9
rate_sps = "1e6"
occupied_bw_hz = 1.0e4

[[nodes]]
id = "mu"
position_m = [4.698463104, 1.710100717, 0.0]
antenna = { pattern = "isotropic" }

[[nodes.streams]]
name = "rx0"
direction = "rx"
fc_hz = 3.5e9
rate_sps = "1e6"
noise_figure_db = 7

[[ris]]
id = "ris0"
pose = { center_m = [0, 0, 0], normal = [1, 0, 0], u_axis = [0, 1, 0] }
grid = { n_u = 16, n_v = 16, pitch_u_m = 0.0428, pitch_v_m = 0.0428 }
states = { library = "ideal_antipodal" }
initial = { fill = 0 }

[run]
duration_s = "0.1"

[[traffic]]
stream = "ap.tx0"
generator = "tone"
params = { freq_hz = 1.0e3 }
"""


# --------------------------------------------------------------------------- YAML 1.2 core


def test_yaml_core_schema_scalars():
    doc = parse_document(
        "a: 1e6\nb: 3.5e9\nc: on\nd: off\ne: yes\nf: 010\ng: 2026-10-10\nh: true\n"
        "i: '1e6'\nj: ~\nk: .inf\nl: 0x1F\nm: 1.0e-20\nn: -7\n",
        "yaml",
    )
    assert doc["a"] == 1e6 and isinstance(doc["a"], float)  # YAML 1.1 would give a string
    assert doc["b"] == 3.5e9 and isinstance(doc["b"], float)
    assert doc["c"] == "on" and doc["d"] == "off" and doc["e"] == "yes"  # not booleans
    assert doc["f"] == 10 and isinstance(doc["f"], int)  # not octal 8
    assert doc["g"] == "2026-10-10"  # no implicit timestamps
    assert doc["h"] is True
    assert doc["i"] == "1e6"
    assert doc["j"] is None
    assert doc["k"] == float("inf")
    assert doc["l"] == 31
    assert doc["m"] == 1e-20
    assert doc["n"] == -7


def test_yaml_dump_quotes_what_the_loader_would_retype():
    data = {
        "s1": "1e6",
        "s2": "5e-4",
        "s3": "1000000",
        "s4": "true",
        "s5": "null",
        "s6": "on",
        "s7": "0.0005",
        "f": 3.5e9,
        "g": 1e-20,
        "i": 10,
        "b": False,
        "n": None,
    }
    text = dump_yaml(data)
    assert parse_document(text, "yaml") == data


def test_duplicate_keys_are_errors_in_yaml_and_json():
    with pytest.raises(SchemaError, match="duplicate"):
        parse_document("a: 1\na: 2\n", "yaml")
    with pytest.raises(SchemaError, match="duplicate"):
        parse_document('{"a": 1, "a": 2}', "json")
    with pytest.raises(SchemaError, match="non-finite"):
        parse_document('{"a": NaN}', "json")
    with pytest.raises(SchemaError, match="cannot parse"):
        parse_document("a: [1, 2\n", "yaml")


def test_unquoted_yaml_rate_is_a_float_and_rejected():
    text = QUICK.read_text(encoding="utf-8").replace('rate_sps: "1e6"', "rate_sps: 1e6", 1)
    with pytest.raises(FloatNotExactError, match="floats are rejected") as info:
        parse_scenario(parse_document(text, "yaml"))
    assert isinstance(info.value, TypeError) and isinstance(info.value, SchemaError)


# --------------------------------------------------------------------------- formats


def test_yaml_json_toml_load_to_the_same_model_and_hash(tmp_path):
    doc = read_document(QUICK)
    (tmp_path / "q.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "q.toml").write_text(QUICK_TOML, encoding="utf-8")
    a = parse_scenario(doc)
    b = parse_scenario(read_document(tmp_path / "q.json"))
    c = parse_scenario(read_document(tmp_path / "q.toml"))
    assert a == b == c
    assert scenario_sha256(a) == scenario_sha256(b) == scenario_sha256(c)
    with pytest.raises(SchemaError, match="unknown document format"):
        read_document(tmp_path / "q.txt")


def test_exact_fields_parse_to_exact_fractions():
    sc = parse_scenario(read_document(LAB))
    tx = sc.nodes[0].streams[0]
    assert tx.rate_sps == Fraction(1_000_000) and isinstance(tx.rate_sps, Fraction)
    assert Rate(tx.rate_sps) == Rate.parse("1e6")
    assert sc.controllers[0].period_s == Fraction(36, 15625)  # "2.304e-3"
    assert sc.controllers[0].min_latency_s == Fraction(1, 5000)
    assert sc.run.duration_s == 5
    assert sc.numerics.page_target_s == Fraction(1, 2000)  # default "5e-4"
    assert sc.run.paced.slacks_s == [Fraction(0), Fraction(1, 1000), Fraction(1, 200)]


# --------------------------------------------------------------------------- round trips


@pytest.mark.parametrize("path", [QUICK, LAB], ids=lambda p: p.name)
def test_schema_round_trip_python_json_yaml(path):
    sc = parse_scenario(read_document(path))
    assert Scenario.model_validate(sc.model_dump()) == sc
    js = sc.model_dump(mode="json")
    assert parse_scenario(json.loads(json.dumps(js))) == sc
    assert parse_scenario(parse_document(dump_yaml(js), "yaml")) == sc
    # model_dump(json) is a fixed point
    assert parse_scenario(js).model_dump(mode="json") == js


@pytest.mark.parametrize("path", [QUICK, LAB], ids=lambda p: p.name)
def test_resolved_yaml_reloads_to_the_same_scenario_and_hash(path, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValidityWarning)
        res = load_scenario(path)
    text = res.to_yaml()
    f = tmp_path / "scenario.resolved.yaml"
    f.write_text(text, encoding="utf-8")
    again = parse_scenario(read_document(f))
    assert again == res.scenario
    assert scenario_sha256(again) == res.sha256()
    # resolution is idempotent
    res2 = resolve_scenario(again, path.parent, warn=False)
    assert res2.scenario == res.scenario
    # every default is written out
    assert "page_target_s: '0.0005'" in text and "verify_interp: true" in text


def test_scenario_sha256_is_sha256_of_canonical_json():
    sc = parse_scenario(read_document(LAB))
    obj = sc.model_dump(mode="json")
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert canonical_json(obj) == text
    assert scenario_sha256(sc) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    # a semantic change changes the hash; an equal value written differently does not
    doc = read_document(LAB)
    doc["controllers"][0]["period_s"] = "36/15625"  # same value as "2.304e-3"
    assert scenario_sha256(parse_scenario(doc)) == scenario_sha256(sc)
    doc["seed"] = 20261011
    assert scenario_sha256(parse_scenario(doc)) != scenario_sha256(sc)


# --------------------------------------------------------------------------- resolution


def test_resolution_fills_occupied_bandwidth_and_warns():
    doc = read_document(QUICK)
    del doc["nodes"][0]["streams"][0]["occupied_bw_hz"]
    sc = parse_scenario(doc)
    assert sc.nodes[0].streams[0].occupied_bw_hz is None
    with pytest.warns(ValidityWarning, match="occupied_bw_hz not declared"):
        res = resolve_scenario(sc, QUICK.parent)
    assert res.scenario.nodes[0].streams[0].occupied_bw_hz == 0.8 * 1e6  # 0.8 f_s (4.5)
    assert res.defaulted_bw == ("ap.tx0",)
    assert any("0.8*f_s" in a for a in res.assumptions)


def test_resolution_records_inputs_and_assumed_magnitudes():
    res = resolve_scenario(parse_scenario(read_document(LAB)), LAB.parent, warn=False)
    lib_path = REPO / "data" / "states" / "lab_binary_3p5ghz.yaml"
    assert [r.to_dict() for r in res.inputs] == [
        {
            "path": "data/states/lab_binary_3p5ghz.yaml",
            "sha256": sha256_file(lib_path),
            "data_class": "ASSUMED",
        }
    ]
    assert sha256_file(lib_path) == hashlib.sha256(lib_path.read_bytes()).hexdigest()
    mags = [a for a in res.assumptions if "assumed 1 (ASSUMED" in a]
    assert len(mags) == 2  # sigma1 and sigma2
    lib = res.libraries["ris0"]
    assert [g.mag for s in lib.states for g in s.gamma] == [1.0, 1.0]


def test_missing_assume_mag_is_an_error():
    doc = read_document(LAB)
    del doc["ris"][0]["states"]["assume_mag"]
    sc = parse_scenario(doc)
    with pytest.raises(SchemaError, match="mag: null"):
        resolve_scenario(sc, LAB.parent, warn=False)
    res = resolve_scenario(sc, LAB.parent, warn=False, strict=False)
    assert "ris0" in res.library_errors


def test_trace_file_is_an_input_with_its_data_class(tmp_path):
    (tmp_path / "walk.csv").write_text("t_s,x_m,y_m,z_m\n0,5,0,0\n1,5,1,0\n", encoding="utf-8")
    doc = read_document(QUICK)
    doc["nodes"][1].pop("position_m")
    doc["nodes"][1]["trajectory"] = {"model": "trace", "file": "walk.csv", "data_class": "MEAS"}
    res = resolve_scenario(parse_scenario(doc), tmp_path, warn=False)
    rec = [r for r in res.inputs if r.path == "walk.csv"]
    assert rec and rec[0].data_class == "MEAS"
    assert rec[0].sha256 == hashlib.sha256((tmp_path / "walk.csv").read_bytes()).hexdigest()
    with pytest.raises(SchemaError, match="not found"):
        resolve_scenario(parse_scenario(doc), tmp_path / "elsewhere", warn=False)


def test_load_scenario_warns_once_per_issue(tmp_path):
    doc = read_document(QUICK)
    del doc["nodes"][0]["streams"][0]["occupied_bw_hz"]
    f = tmp_path / "q.yaml"
    f.write_text(dump_yaml(doc), encoding="utf-8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = load_scenario(f)
    msgs = [str(x.message) for x in w if issubclass(x.category, ValidityWarning)]
    assert len(msgs) == 1 and "OCCUPIED_BW_UNDECLARED" in msgs[0]
    assert res.scenario.nodes[0].streams[0].occupied_bw_hz == 800000.0
