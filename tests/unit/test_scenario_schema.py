"""JSON Schema export and validation of every scenario and state library (DESIGN.md 9.1).

``jsonschema`` is not installed (and not a dependency), so this module carries a small,
independent JSON Schema (draft 2020-12) validator for the keywords pydantic emits. It is
an oracle separate from pydantic: every shipped scenario and state library, read as a raw
document, must validate against the exported schema, and documents pydantic rejects for
type reasons (floats for exact quantities, unknown keys, bad enums) must fail it too. The
validator has its own teeth tests.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from pydantic_core import PydanticCustomError

from sremu.errors import SchemaError
from sremu.record.manifest import manifest_json_schema
from sremu.scenario.load import parse_scenario, read_document
from sremu.scenario.models import EXACT_PATTERN, exact_to_str, parse_exact_field
from sremu.scenario.schema_export import (
    SCHEMA_FILES,
    packaged_schema_dir,
    scenario_json_schema,
    schema_text,
    state_library_json_schema,
)

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
SCENARIOS = sorted((REPO / "scenarios").glob("*.yaml"))
STATES = sorted((REPO / "data" / "states").glob("*.yaml"))


# =========================================================================== mini validator


def _is_type(x: Any, t: str) -> bool:
    if t == "object":
        return isinstance(x, dict)
    if t == "array":
        return isinstance(x, list)
    if t == "string":
        return isinstance(x, str)
    if t == "boolean":
        return isinstance(x, bool)
    if t == "null":
        return x is None
    if t == "number":
        return isinstance(x, int | float) and not isinstance(x, bool)
    if t == "integer":
        return (isinstance(x, int) and not isinstance(x, bool)) or (
            isinstance(x, float) and x.is_integer()
        )
    raise ValueError(f"unknown JSON type {t}")


def _json_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def ecma_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a JSON Schema ``pattern`` with ECMA-262 semantics (the JSON Schema regex
    dialect) for the constructs our schemas use: ``$`` (outside character classes) matches
    only at the end of the input -- Python's ``$`` also matches before a final newline -- and
    character classes are ASCII (``re.ASCII``)."""
    out: list[str] = []
    i, in_class = 0, False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            out.append(pattern[i : i + 2])
            i += 2
            continue
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
        elif ch == "$":
            ch = r"\Z"
        out.append(ch)
        i += 1
    return re.compile("".join(out), re.ASCII)


def validate_instance(x: Any, schema: Any, root: Any = None, path: str = "$") -> list[str]:
    """Errors of instance ``x`` against a draft 2020-12 ``schema`` (subset)."""
    root = schema if root is None else root
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: false schema"]
    errs: list[str] = []
    if "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/"), ref
        target = root
        for part in ref[2:].split("/"):
            target = target[part]
        errs += validate_instance(x, target, root, path)
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(x, t) for t in types):
            return [*errs, f"{path}: expected {types}, got {type(x).__name__}"]
    if "const" in schema and not _json_equal(x, schema["const"]):
        errs.append(f"{path}: expected const {schema['const']!r}")
    if "enum" in schema and not any(_json_equal(x, e) for e in schema["enum"]):
        errs.append(f"{path}: {x!r} not in enum {schema['enum']}")
    if isinstance(x, str):
        if "pattern" in schema and ecma_pattern(schema["pattern"]).search(x) is None:
            errs.append(f"{path}: {x!r} does not match {schema['pattern']}")
        if "minLength" in schema and len(x) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(x) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
    if _is_type(x, "number"):
        if "minimum" in schema and x < schema["minimum"]:
            errs.append(f"{path}: {x} < minimum {schema['minimum']}")
        if "maximum" in schema and x > schema["maximum"]:
            errs.append(f"{path}: {x} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and x <= schema["exclusiveMinimum"]:
            errs.append(f"{path}: {x} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and x >= schema["exclusiveMaximum"]:
            errs.append(f"{path}: {x} >= exclusiveMaximum {schema['exclusiveMaximum']}")
    if isinstance(x, list):
        prefix = schema.get("prefixItems", [])
        for i, (item, sub) in enumerate(zip(x, prefix, strict=False)):
            errs += validate_instance(item, sub, root, f"{path}[{i}]")
        if "items" in schema:
            for i, item in enumerate(x[len(prefix) :], start=len(prefix)):
                errs += validate_instance(item, schema["items"], root, f"{path}[{i}]")
        if "minItems" in schema and len(x) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(x) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
    if isinstance(x, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in x:
                errs.append(f"{path}: missing required {k!r}")
        for k, v in x.items():
            if k in props:
                errs += validate_instance(v, props[k], root, f"{path}.{k}")
            elif "additionalProperties" in schema:
                errs += validate_instance(v, schema["additionalProperties"], root, f"{path}.{k}")
    for key in ("anyOf", "oneOf"):
        if key in schema:
            ok = [not validate_instance(x, s, root, path) for s in schema[key]]
            if key == "anyOf" and not any(ok):
                errs.append(f"{path}: matches no anyOf branch")
            if key == "oneOf" and sum(ok) != 1:
                errs.append(f"{path}: matches {sum(ok)} oneOf branches (need exactly 1)")
    for s in schema.get("allOf", []):
        errs += validate_instance(x, s, root, path)
    if "not" in schema and not validate_instance(x, schema["not"], root, path):
        errs.append(f"{path}: matches 'not' schema")
    return errs


# --------------------------------------------------------------------------- validator teeth


def test_mini_validator_teeth():
    schema = {
        "$defs": {
            "P": {
                "type": "object",
                "properties": {"x": {"type": "integer", "minimum": 0}},
                "required": ["x"],
                "additionalProperties": False,
            }
        },
        "type": "object",
        "properties": {
            "p": {"$ref": "#/$defs/P"},
            "e": {"enum": ["a", "b"]},
            "s": {"type": "string", "pattern": "^[a-z]+$"},
            "v": {
                "type": "array",
                "prefixItems": [{"type": "number"}] * 2,
                "minItems": 2,
                "maxItems": 2,
            },
            "u": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "o": {"oneOf": [{"type": "number"}, {"type": "integer"}]},
            "c": {"const": True},
        },
        "additionalProperties": False,
    }
    good = {"p": {"x": 3}, "e": "a", "s": "abc", "v": [1, 2.5], "u": None, "o": 1.5, "c": True}
    assert validate_instance(good, schema) == []
    bad = [
        {"p": {"x": -1}},
        {"p": {}},
        {"p": {"x": 1, "y": 2}},
        {"p": {"x": True}},
        {"e": "c"},
        {"s": "AB"},
        {"s": "abc\n"},  # ECMA-262: "$" does not match before a final newline
        {"s": " abc"},
        {"v": [1]},
        {"v": [1, "2"]},
        {"u": 3},
        {"o": 2},  # both oneOf branches match
        {"c": 1},  # 1 is not the boolean true
        {"zz": 0},
    ]
    for doc in bad:
        assert validate_instance(doc, schema), doc


# =========================================================================== the schemas


def test_shipped_schema_copies_are_current():
    expect = {"scenario": scenario_json_schema(), "states": state_library_json_schema()}
    for kind, fn in SCHEMA_FILES.items():
        path = packaged_schema_dir() / fn
        assert path.read_text(encoding="utf-8") == schema_text(expect[kind]), (
            f"{path} is stale; regenerate with sremu.scenario.schema_export.write_schemas()"
        )


def test_schemas_are_draft_2020_12_and_self_describing():
    for s in (scenario_json_schema(), state_library_json_schema(), manifest_json_schema()):
        assert s["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert s["$id"].startswith("urn:sremu:schema:")
        assert s["type"] == "object"
        assert s["additionalProperties"] is False  # extra="forbid"
        json.dumps(s)


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.name)
def test_scenario_files_validate_against_json_schema(path):
    doc = read_document(path)
    assert validate_instance(doc, scenario_json_schema()) == []
    # and the resolved (all defaults written out) JSON dump validates too
    dumped = parse_scenario(doc).model_dump(mode="json")
    assert validate_instance(dumped, scenario_json_schema()) == []


@pytest.mark.parametrize("path", STATES, ids=lambda p: p.name)
def test_state_library_files_validate_against_json_schema(path):
    assert validate_instance(read_document(path), state_library_json_schema()) == []


def test_all_shipped_files_were_found():
    assert {p.name for p in SCENARIOS} >= {"quickstart.yaml", "lab_like_A.yaml"}
    assert {p.stem for p in STATES} == {
        "lab_binary_3p5ghz", "tang2022_ris1_27ghz", "tang2022_ris2_33ghz", "ideal_bbit",
        "ideal_antipodal", "on_off", "abeywickrama",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("mutate", "where"),
    [
        (lambda d: d["run"].__setitem__("duration_s", 2.5), "float for an exact time"),
        (lambda d: d["controllers"][0].__setitem__("period_s", 2.304e-3), "float period"),
        (lambda d: d["nodes"][0]["streams"][0].__setitem__("rate_sps", 5880.5), "float rate"),
        (lambda d: d["nodes"][0]["streams"][0].__setitem__("rate_sps", "1e6 S/s"), "bad text"),
        (lambda d: d["run"].__setitem__("mode", "realtime"), "bad enum"),
        (lambda d: d["nodes"][0].__setitem__("color", "red"), "unknown key"),
        (lambda d: d.__setitem__("seed", "7"), "string seed"),
        (lambda d: d["nodes"][0]["streams"][0].__setitem__("fc_hz", "3.5e9"), "string number"),
        (lambda d: d["nodes"][0].__setitem__("position_m", [0, 0]), "short vector"),
        (lambda d: d["ris"][0]["grid"].__setitem__("n_u", 0), "n_u below 1"),
        (lambda d: d["ris"][0]["grid"].__setitem__("n_u", True), "bool for int"),
        (lambda d: d.__setitem__("schema_version", "2.0"), "schema version"),
    ],
)
def test_type_violations_fail_both_pydantic_and_json_schema(mutate, where):
    doc = read_document(REPO / "scenarios" / "lab_like_A.yaml")
    mutate(doc)
    assert validate_instance(doc, scenario_json_schema()), where
    with pytest.raises(Exception, match="schema error"):
        parse_scenario(doc)


def test_integral_floats_for_exact_fields_are_rejected_by_the_loader():
    """JSON Schema cannot tell 1e6 from 1000000 (both are the integer 1000000 in its data
    model), so this part of R1 is enforced by the loader only."""
    from sremu.scenario.models import FloatNotExactError

    doc = read_document(REPO / "scenarios" / "lab_like_A.yaml")
    doc["nodes"][0]["streams"][0]["rate_sps"] = 1e6
    assert validate_instance(doc, scenario_json_schema()) == []  # documented limitation
    with pytest.raises(FloatNotExactError) as info:
        parse_scenario(doc)
    assert isinstance(info.value, TypeError)


def test_json_schema_defaults_are_the_spec_texts_and_equal_the_model_defaults():
    from sremu.scenario.models import Numerics, ObservationModel, PacedModel, Rwp1dTrajectory

    defs = scenario_json_schema()["$defs"]
    cases = [
        ("Numerics", "page_target_s", "5e-4", Numerics().page_target_s),
        ("Numerics", "t_u_max_s", "1e-2", Numerics().t_u_max_s),
        ("PacedModel", "d_lag_s", "1e-3", PacedModel().d_lag_s),
        ("ObservationModel", "window_s", "1e-3", ObservationModel().window_s),
        ("ObservationModel", "position_delay_s", "0", ObservationModel().position_delay_s),
        ("Rwp1dTrajectory", "pause_s", "0", Rwp1dTrajectory.model_fields["pause_s"].default),
    ]
    for model, field, text, value in cases:
        assert defs[model]["properties"][field]["default"] == text
        assert Fraction(text) == value
    slacks = defs["PacedModel"]["properties"]["slacks_s"]["default"]
    assert slacks == ["0", "1e-3", "5e-3"]
    assert [Fraction(x) for x in slacks] == PacedModel().slacks_s


# --------------------------------------------------------------------------- exact syntax


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("23.04e6", Fraction(23_040_000)),
        ("1/200", Fraction(1, 200)),
        ("5e-3", Fraction(1, 200)),
        ("2.304e-3", Fraction(36, 15625)),
        ("0.5e-3", Fraction(1, 2000)),
        ("-7", Fraction(-7)),
        (".5", Fraction(1, 2)),
        ("1/3", Fraction(1, 3)),
    ],
)
def test_exact_pattern_and_canonical_text(text, value):
    assert re.fullmatch(EXACT_PATTERN, text)
    assert Fraction(text) == value
    canon = exact_to_str(value)
    assert re.fullmatch(EXACT_PATTERN, canon)
    assert Fraction(canon) == value


@pytest.mark.parametrize("text", ["1e6 ", "1/0", "1.5/2", "nan", "inf", "0x10", "1_000", "", "/3"])
def test_exact_pattern_rejects(text):
    assert re.fullmatch(EXACT_PATTERN, text) is None
    with pytest.raises(PydanticCustomError):
        parse_exact_field(text)


# Candidate texts for the loader/schema agreement sweep: valid forms, whitespace in every
# position, Unicode digits and look-alikes, and malformed numbers.
EXACT_CANDIDATES = [
    "5", "+5", "-5", "5.", ".5", "0.5", "5e3", "5E+3", "5e-03", "1/200", "01/0200", "-1/3",
    "23.04e6", "2.304e-3",
    " 5", "5 ", "\t5", "5\t", "5\n", "\n5", "5\r", "5\r\n", " 1/200 ", "1 /2", "1/ 2",
    " 5", "5 ", "５", "٥", "5​",
    "1_000", "1/0", "1/00", "1.5/2", "1/2/3", "1e", "e5", ".", "", "/3", "nan", "inf",
    "0x10", "--5", "5-", "1e6 S/s",
]  # fmt: skip


@pytest.mark.parametrize("text", EXACT_CANDIDATES)
def test_s9_1_exact_loader_accepts_exactly_what_the_json_schema_accepts(text):
    """Strip-vs-schema consistency (section 9.1): the loader accepts an exact-quantity string
    if and only if the exported JSON Schema (ECMA-262 ``pattern``) does; nothing is stripped,
    and an accepted string's value is ``Fraction(text)``."""
    exact_schema = scenario_json_schema()["$defs"]["RunModel"]["properties"]["duration_s"]
    schema_ok = not validate_instance(text, exact_schema)
    try:
        value = parse_exact_field(text)
        loader_ok = True
    except PydanticCustomError:
        loader_ok = False
    assert loader_ok is schema_ok, (text, loader_ok, schema_ok)
    if loader_ok:
        assert value == Fraction(text)


@pytest.mark.parametrize("text", [" 1e6", "1e6 ", "\t1e6", "1e6\n", "1e6\r\n", " 1e6 "])
@pytest.mark.parametrize(
    "where",
    [
        ("nodes", 0, "streams", 0, "rate_sps"),
        ("run", "duration_s"),
        ("controllers", 0, "period_s"),
    ],
)
def test_s9_1_exact_strings_with_surrounding_whitespace_are_rejected(text, where):
    """Surrounding whitespace is rejected by the loader (named in the message) and by the
    JSON Schema alike, in every exact field (rates, durations, periods)."""
    doc = read_document(REPO / "scenarios" / "lab_like_A.yaml")
    node: Any = doc
    for key in where[:-1]:
        node = node[key]
    node[where[-1]] = text
    assert validate_instance(doc, scenario_json_schema()), (where, text)
    with pytest.raises(SchemaError, match="surrounding whitespace"):
        parse_scenario(doc)
    node[where[-1]] = text.strip()  # control: the stripped text is valid in this field
    assert validate_instance(doc, scenario_json_schema()) == []
    parse_scenario(doc)


def test_exact_to_str_is_decimal_when_finite():
    assert exact_to_str(Fraction(1, 2000)) == "0.0005"
    assert exact_to_str(Fraction(23_040_000)) == "23040000"
    assert exact_to_str(Fraction(-3, 8)) == "-0.375"
    assert exact_to_str(Fraction(1, 3)) == "1/3"
    assert exact_to_str(Fraction(36, 15625)) == "0.002304"
    assert math.isclose(float(exact_to_str(Fraction(1, 10**30))), 1e-30)
