"""Loading, resolving and hashing scenarios and state libraries (DESIGN.md 3.6, 5.5, 9.1).

* **Formats.** YAML is primary; JSON and TOML load through the same models. YAML is read
  with YAML 1.2 core-schema scalars (:class:`YamlLoader`): ``3.5e9`` and ``1e6`` are
  floats exactly as in JSON, ``true``/``false`` are the only booleans (``on``, ``off``,
  ``yes``, ``no`` stay strings), ``010`` is the integer 10 and dates stay strings.
  Duplicate mapping keys are errors in every format. So an exact field written as a
  bare ``rate_sps: 1e6`` is rejected (write ``"1e6"``), as in JSON.
* **Errors.** Schema violations raise :class:`~sremu.errors.SchemaError` with every error
  listed; a float given for an exact quantity raises
  :class:`~sremu.scenario.models.FloatNotExactError` (a ``SchemaError`` and a
  ``TypeError``, R1).
* **State libraries.** Built-ins (``data/states/<name>.yaml``) are referenced by name, or
  for parametric analytic families as ``"ideal_bbit(b=2)"`` /
  ``"abeywickrama(b=2, beta_min=0.2, phi_deg=0, alpha=1.6)"`` (positional arguments follow
  the family's parameter order). Files are resolved relative to the scenario file.
* **Resolution** (:func:`resolve_scenario`) fills computed defaults (TX ``occupied_bw_hz =
  0.8 f_s``), loads every panel's state library, applies ``assume_mag`` and collects the
  manifest ``inputs`` and ``assumptions``. ``scenario.resolved.yaml`` is the YAML dump of
  the resolved model (every default written out); ``scenario_sha256`` is the SHA-256 of
  its canonical JSON (sorted keys, no whitespace, UTF-8).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import tomllib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import BaseModel, ValidationError

from sremu.errors import SchemaError, ValidityWarning
from sremu.scenario.models import (
    FloatNotExactError,
    GammaPoint,
    Scenario,
    StateEntry,
    StateLibrary,
    StreamModel,
)

__all__ = [
    "BUILTIN_STATES_ENV",
    "InputRecord",
    "ResolvedScenario",
    "YamlDumper",
    "YamlLoader",
    "apply_assume_mag",
    "builtin_library_names",
    "builtin_states_dir",
    "canonical_json",
    "dump_yaml",
    "expand_family",
    "load_scenario",
    "load_state_library",
    "parse_document",
    "parse_library_ref",
    "parse_scenario",
    "parse_state_library",
    "read_document",
    "resolve_scenario",
    "scenario_sha256",
    "sha256_file",
    "validation_error_to_schema_error",
]

BUILTIN_STATES_ENV: Final[str] = "SREMU_STATES_DIR"
"""Environment variable that overrides the built-in state-library directory."""

_OCCUPIED_BW_DEFAULT_FRACTION: Final[Fraction] = Fraction(4, 5)


# =========================================================================== YAML 1.2 core


_NULL_RE = re.compile(r"^(?:~|null|Null|NULL|)$")
_BOOL_RE = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
_INT_RE = re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$")
_FLOAT_RE = re.compile(
    r"^(?:[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?[0-9]+[eE][-+]?[0-9]+"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
)


def _install_core_resolvers(cls: Any) -> None:
    cls.yaml_implicit_resolvers = {}
    cls.add_implicit_resolver("tag:yaml.org,2002:null", _NULL_RE, ["~", "n", "N", ""])
    cls.add_implicit_resolver("tag:yaml.org,2002:bool", _BOOL_RE, list("tTfF"))
    cls.add_implicit_resolver("tag:yaml.org,2002:int", _INT_RE, list("-+0123456789"))
    cls.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT_RE, list("-+0123456789."))


class YamlLoader(yaml.SafeLoader):
    """Safe YAML loader with YAML 1.2 core-schema scalars and duplicate-key detection."""


_install_core_resolvers(YamlLoader)


def _construct_int(loader: YamlLoader, node: yaml.ScalarNode) -> int:
    text = str(loader.construct_scalar(node))
    sign = -1 if text.startswith("-") else 1
    body = text.lstrip("+-")
    if body.startswith("0o"):
        return sign * int(body[2:], 8)
    if body.startswith("0x"):
        return sign * int(body[2:], 16)
    return sign * int(body, 10)


def _construct_bool(loader: YamlLoader, node: yaml.ScalarNode) -> bool:
    return str(loader.construct_scalar(node)).lower() == "true"


def _construct_mapping(loader: YamlLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    out: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str | int | float | bool) and key is not None:
            raise yaml.constructor.ConstructorError(
                None, None, f"unsupported mapping key {key!r}", key_node.start_mark
            )
        if key in out:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        out[key] = loader.construct_object(value_node, deep=True)
    return out


YamlLoader.add_constructor("tag:yaml.org,2002:int", _construct_int)
YamlLoader.add_constructor("tag:yaml.org,2002:bool", _construct_bool)
YamlLoader.add_constructor("tag:yaml.org,2002:map", _construct_mapping)


class YamlDumper(yaml.SafeDumper):
    """Safe YAML dumper whose quoting matches :class:`YamlLoader` (round-trip safe)."""


_install_core_resolvers(YamlDumper)


def dump_yaml(data: Any) -> str:
    """YAML text of JSON-compatible ``data`` (key order kept); loads back with
    :class:`YamlLoader` to an equal object."""
    return yaml.dump(
        data,
        Dumper=YamlDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=100,
    )


# =========================================================================== documents


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise SchemaError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def _json_constant(name: str) -> Any:
    raise SchemaError(f"non-finite JSON number {name} is not allowed")


_FORMATS: Final[dict[str, str]] = {
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
}


def read_document(path: str | Path, *, fmt: str | None = None) -> Any:
    """Read a YAML, JSON or TOML file; the format follows the suffix (``.yaml``/``.yml``,
    ``.json``, ``.toml``) unless ``fmt`` (``"yaml"``, ``"json"``, ``"toml"``) is given."""
    p = Path(path)
    if fmt is None:
        fmt = _FORMATS.get(p.suffix.lower())
        if fmt is None:
            raise SchemaError(f"{p}: unknown document format (use .yaml, .json or .toml)")
    return parse_document(p.read_text(encoding="utf-8"), fmt, label=str(p))


def parse_document(text: str, fmt: str, *, label: str = "<text>") -> Any:
    """Parse document text in format ``fmt`` (``"yaml"``, ``"json"`` or ``"toml"``)."""
    try:
        if fmt == "yaml":
            return yaml.load(text, Loader=YamlLoader)  # YamlLoader is a SafeLoader
        if fmt == "json":
            return json.loads(text, object_pairs_hook=_json_pairs, parse_constant=_json_constant)
        if fmt == "toml":
            return tomllib.loads(text)
    except SchemaError:
        raise
    except (yaml.YAMLError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SchemaError(f"{label}: cannot parse {fmt}: {exc}") from exc
    raise SchemaError(f"unknown document format {fmt!r}")


def validation_error_to_schema_error(exc: ValidationError, label: str) -> SchemaError:
    """Map a pydantic ``ValidationError`` to a ``SchemaError`` listing every error (a
    :class:`FloatNotExactError` when an exact field got a float)."""
    lines = []
    is_float = False
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  {loc}: {err['msg']}")
        is_float = is_float or err["type"] == "exact_float"
    cls = FloatNotExactError if is_float else SchemaError
    return cls(f"{label}: {len(lines)} schema error(s):\n" + "\n".join(lines))


def _validate_model[M: BaseModel](model: type[M], data: Any, label: str) -> M:
    if not isinstance(data, Mapping):
        raise SchemaError(f"{label}: the document root must be a mapping")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise validation_error_to_schema_error(exc, label) from None


def parse_scenario(data: Mapping[str, Any], *, label: str = "<scenario>") -> Scenario:
    """Validate a scenario document (schema only; no guards, no file access)."""
    return _validate_model(Scenario, data, label)


def parse_state_library(data: Mapping[str, Any], *, label: str = "<state library>") -> StateLibrary:
    """Validate a state-library document (schema only)."""
    return _validate_model(StateLibrary, data, label)


# =========================================================================== state libraries


def builtin_states_dir() -> Path:
    """Directory of the built-in state libraries.

    ``$SREMU_STATES_DIR`` if set; else ``<package>/data/states`` if it exists (wheel
    layout); else ``<repo>/data/states`` (source checkout, src layout).
    """
    import os

    env = os.environ.get(BUILTIN_STATES_ENV)
    if env:
        return Path(env)
    pkg = Path(__file__).resolve().parents[1]  # src/sremu
    for cand in (pkg / "data" / "states", pkg.parents[1] / "data" / "states"):
        if cand.is_dir():
            return cand
    raise SchemaError(
        "built-in state libraries not found (looked in <package>/data/states and "
        f"<repo>/data/states); set ${BUILTIN_STATES_ENV}"
    )


def builtin_library_names() -> tuple[str, ...]:
    """Names of the built-in state libraries."""
    return tuple(sorted(p.stem for p in builtin_states_dir().glob("*.yaml")))


_REF_RE = re.compile(r"^\s*([A-Za-z0-9_\-.]+)\s*(?:\((.*)\))?\s*$", re.S)


def parse_library_ref(ref: str) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    """Split ``"name"`` or ``"name(a, k=v)"`` into (name, positional, keyword) with literal
    numeric arguments."""
    m = _REF_RE.match(ref)
    if m is None:
        raise SchemaError(f"bad state library reference {ref!r}")
    name, args = m.group(1), m.group(2)
    if args is None or not args.strip():
        return name, (), {}
    try:
        call = ast.parse(f"f({args})", mode="eval").body
    except SyntaxError as exc:
        raise SchemaError(f"bad arguments in state library reference {ref!r}") from exc
    assert isinstance(call, ast.Call)

    def lit(node: ast.expr) -> Any:
        value = ast.literal_eval(node)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SchemaError(f"{ref!r}: arguments must be numbers")
        return value

    try:
        pos = tuple(lit(a) for a in call.args)
        kw = {k.arg: lit(k.value) for k in call.keywords if k.arg is not None}
    except ValueError as exc:
        raise SchemaError(f"{ref!r}: arguments must be numeric literals") from exc
    if len(kw) != len(call.keywords):
        raise SchemaError(f"{ref!r}: **kwargs are not allowed")
    return name, pos, kw


def _bind_params(lib: StateLibrary, pos: Sequence[Any], kw: Mapping[str, Any]) -> dict[str, float]:
    assert lib.family is not None
    spec = lib.family.parameters
    names = list(spec)
    if len(pos) > len(names):
        raise SchemaError(f"{lib.name}: too many arguments ({len(pos)} > {len(names)})")
    bound: dict[str, Any] = dict(zip(names, pos, strict=False))
    for k, v in kw.items():
        if k not in spec:
            raise SchemaError(f"{lib.name}: unknown parameter {k!r}; parameters: {names}")
        if k in bound:
            raise SchemaError(f"{lib.name}: parameter {k!r} given twice")
        bound[k] = v
    out: dict[str, float] = {}
    for k, p in spec.items():
        v = bound.get(k, p.default)
        if v is None:
            raise SchemaError(f"{lib.name}: parameter {k!r} is required")
        if p.integer and (not float(v).is_integer()):
            raise SchemaError(f"{lib.name}: parameter {k!r} must be an integer, got {v}")
        if (p.min is not None and v < p.min) or (p.max is not None and v > p.max):
            raise SchemaError(f"{lib.name}: parameter {k!r} = {v} outside [{p.min}, {p.max}]")
        out[k] = float(v)
    return out


def expand_family(lib: StateLibrary, params: Mapping[str, float]) -> StateLibrary:
    """Expand an analytic family into explicit states (frequency independent).

    * ``ideal_bbit(b)``: ``Gamma_s = exp(j 2 pi s / 2**b)``, s = 0..2**b - 1.
    * ``abeywickrama(b, beta_min, phi_deg, alpha)``: ``theta_s = 2 pi s / 2**b``,
      ``Gamma_s = beta(theta_s) exp(j theta_s)`` with
      ``beta(theta) = (1 - beta_min)((sin(theta - phi) + 1)/2)**alpha + beta_min``.

    Phases are written in degrees in [0, 360). The parameters are recorded in the expanded
    library's ``parameters`` and its name becomes ``"family(k=v, ...)"``.
    """
    if lib.family is None:
        return lib
    b = int(params["b"])
    n = 1 << b
    states = []
    for s in range(n):
        theta = 2.0 * math.pi * s / n
        phase_deg = 360.0 * s / n
        if lib.family.name == "ideal_bbit":
            mag = 1.0
        else:
            beta_min = params["beta_min"]
            phi = math.radians(params["phi_deg"])
            mag = (1.0 - beta_min) * ((math.sin(theta - phi) + 1.0) / 2.0) ** params[
                "alpha"
            ] + beta_min
        states.append(
            StateEntry(
                id=s, label=f"s{s}", gamma=[GammaPoint(f_hz=None, mag=mag, phase_deg=phase_deg)]
            )
        )
    arg_text = ", ".join(f"{k}={_num_text(v)}" for k, v in params.items())
    data = lib.model_dump(mode="python")
    data.update(
        name=f"{lib.family.name}({arg_text})",
        family=None,
        parameters=dict(params),
        states=[s.model_dump(mode="python") for s in states],
    )
    return StateLibrary.model_validate(data)


def _num_text(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def load_state_library(
    ref: str | None = None, *, file: str | Path | None = None, base_dir: Path | None = None
) -> tuple[StateLibrary, Path]:
    """Load a built-in (``ref``) or file state library, expanding analytic families.

    Returns the (expanded) library and the file it came from.
    """
    if (ref is None) == (file is None):
        raise ValueError("give exactly one of ref or file")
    if file is not None:
        path = Path(file)
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        pos: tuple[Any, ...] = ()
        kw: dict[str, Any] = {}
    else:
        assert ref is not None
        name, pos, kw = parse_library_ref(ref)
        path = builtin_states_dir() / f"{name}.yaml"
        if not path.is_file():
            raise SchemaError(
                f"unknown built-in state library {name!r}; built-ins: "
                f"{', '.join(builtin_library_names())}"
            )
    if not path.is_file():
        raise SchemaError(f"state library file not found: {path}")
    lib = parse_state_library(read_document(path), label=str(path))
    if lib.family is None:
        if pos or kw:
            raise SchemaError(f"{lib.name} is not a parametric family; it takes no arguments")
        return lib, path
    return expand_family(lib, _bind_params(lib, pos, kw)), path


def apply_assume_mag(lib: StateLibrary, assume_mag: Mapping[str, float] | None) -> StateLibrary:
    """Replace ``mag: null`` entries by ``assume_mag[label]``.

    Every null-magnitude state needs a value; ``assume_mag`` keys must be labels of
    null-magnitude states (a value for a state whose magnitude is defined is an error).
    """
    nulls = set(lib.null_mag_labels())
    given = dict(assume_mag or {})
    unknown = set(given) - set(lib.labels)
    if unknown:
        raise SchemaError(
            f"{lib.name}: assume_mag names unknown states {sorted(unknown)}; "
            f"states: {list(lib.labels)}"
        )
    defined = set(given) - nulls
    if defined:
        raise SchemaError(
            f"{lib.name}: assume_mag given for states with defined magnitudes {sorted(defined)}"
        )
    missing = nulls - set(given)
    if missing:
        raise SchemaError(
            f"{lib.name}: states {sorted(missing)} have mag: null; the scenario must set "
            "states.assume_mag for them (recorded as ASSUMED)"
        )
    if not nulls:
        return lib
    states = []
    for s in lib.states:
        if s.label in nulls:
            gam = [
                g.model_copy(update={"mag": given[s.label]}) if g.mag is None else g
                for g in s.gamma
            ]
            s = s.model_copy(update={"gamma": gam})
        states.append(s)
    return lib.model_copy(update={"states": states})


# =========================================================================== resolution


@dataclass(frozen=True)
class InputRecord:
    """One manifest ``inputs`` entry (section 3.6)."""

    path: str
    sha256: str
    data_class: str

    def to_dict(self) -> dict[str, str]:
        """JSON-ready dict."""
        return {"path": self.path, "sha256": self.sha256, "data_class": self.data_class}


@dataclass(frozen=True)
class ResolvedScenario:
    """A validated scenario with computed defaults filled in.

    ``scenario``: the resolved model (dumped to ``scenario.resolved.yaml``).
    ``libraries``: panel id -> state library with ``assume_mag`` applied.
    ``inputs`` / ``assumptions``: manifest entries. ``notes``: resolution messages
    (each also emitted as a ``ValidityWarning`` when ``warn=True``). ``library_errors``:
    panel id -> message for state libraries that could not be loaded or completed (only
    with ``strict=False``; the guards report them as errors).
    """

    scenario: Scenario
    base_dir: Path | None
    libraries: Mapping[str, StateLibrary]
    library_files: Mapping[str, Path]
    inputs: tuple[InputRecord, ...]
    assumptions: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())
    library_errors: Mapping[str, str] = field(default_factory=dict)
    defaulted_bw: tuple[str, ...] = ()

    def to_yaml(self) -> str:
        """``scenario.resolved.yaml`` text (every default written out)."""
        return dump_yaml(self.scenario.model_dump(mode="json"))

    def sha256(self) -> str:
        """``scenario_sha256``: SHA-256 of the canonical JSON of the resolved scenario."""
        return scenario_sha256(self.scenario)


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Canonical JSON text: sorted keys, ``(",", ":")`` separators, UTF-8 characters kept,
    no NaN or infinity. Used for every hash over structured data."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def scenario_sha256(scenario: Scenario) -> str:
    """SHA-256 hex of ``canonical_json(scenario.model_dump(mode="json"))``."""
    text = canonical_json(scenario.model_dump(mode="json"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rel_label(path: Path, base_dir: Path | None) -> str:
    path = path.resolve()
    for root in [p for p in (base_dir,) if p is not None] + [_repo_root()]:
        try:
            return path.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_stream(
    s: StreamModel, notes: list[str], sid: str, defaulted: list[str]
) -> StreamModel:
    if s.direction == "tx" and s.occupied_bw_hz is None:
        bw = float(_OCCUPIED_BW_DEFAULT_FRACTION * s.rate_sps)
        defaulted.append(sid)
        notes.append(f"{sid}: occupied_bw_hz not declared; using 0.8*f_s = {bw:g} Hz (section 4.5)")
        return s.model_copy(update={"occupied_bw_hz": bw})
    return s


def resolve_scenario(
    scenario: Scenario,
    base_dir: str | Path | None = None,
    *,
    warn: bool = True,
    strict: bool = True,
) -> ResolvedScenario:
    """Fill computed defaults, load state libraries and collect manifest inputs/assumptions.

    With ``strict`` (the default), an unresolvable library or an ``assume_mag`` problem
    raises ``SchemaError``; otherwise it is recorded in ``library_errors``. Missing trace
    or antenna files always raise. The guards are separate
    (:func:`sremu.scenario.guards.check_scenario`).
    """
    bdir = Path(base_dir) if base_dir is not None else None
    notes: list[str] = []
    assumptions: list[str] = []
    inputs: list[InputRecord] = []
    defaulted: list[str] = []

    nodes = []
    for n in scenario.nodes:
        streams = [_resolve_stream(s, notes, f"{n.id}.{s.name}", defaulted) for s in n.streams]
        nodes.append(n.model_copy(update={"streams": streams}))
        traj = n.trajectory
        if traj is not None and traj.model == "trace":
            fp = _abs(traj.file, bdir)
            inputs.append(InputRecord(_rel_label(fp, bdir), sha256_file(fp), str(traj.data_class)))
        pat = n.antenna.pattern
        if not isinstance(pat, str) and hasattr(pat, "tabulated"):
            tab = pat.tabulated
            fp = _abs(tab.file, bdir)
            inputs.append(
                InputRecord(_rel_label(fp, bdir), sha256_file(fp), str(tab.provenance.data_class))
            )

    libraries: dict[str, StateLibrary] = {}
    lib_files: dict[str, Path] = {}
    lib_errors: dict[str, str] = {}
    for p in scenario.ris:
        try:
            lib, lib_files[p.id] = load_state_library(
                p.states.library, file=p.states.file, base_dir=bdir
            )
            completed = apply_assume_mag(lib, p.states.assume_mag)
        except SchemaError as exc:
            if strict:
                raise
            lib_errors[p.id] = str(exc)
            continue
        rec = InputRecord(
            _rel_label(lib_files[p.id], bdir),
            sha256_file(lib_files[p.id]),
            lib.provenance.resolved_data_class(),
        )
        if rec not in inputs:
            inputs.append(rec)
        for label in lib.null_mag_labels():
            assert p.states.assume_mag is not None
            assumptions.append(
                f"ris {p.id}: |Gamma| of state '{label}' of library '{lib.name}' assumed "
                f"{p.states.assume_mag[label]:g} (ASSUMED; the library has mag: null)"
            )
        libraries[p.id] = completed
        for a in lib.provenance.assumptions:
            assumptions.append(f"library {lib.name}: {a}")
    resolved = scenario.model_copy(update={"nodes": nodes})
    # Re-validate so the resolved object is exactly what the resolved YAML would load as.
    resolved = Scenario.model_validate(resolved.model_dump(mode="python"))
    assumptions.extend(notes)
    if warn:
        for msg in notes:
            warnings.warn(msg, ValidityWarning, stacklevel=2)
    return ResolvedScenario(
        scenario=resolved,
        base_dir=bdir,
        libraries=libraries,
        library_files=lib_files,
        inputs=tuple(inputs),
        assumptions=tuple(dict.fromkeys(assumptions)),
        notes=tuple(notes),
        library_errors=lib_errors,
        defaulted_bw=tuple(defaulted),
    )


def _abs(file: str, base_dir: Path | None) -> Path:
    p = Path(file)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    if not p.is_file():
        raise SchemaError(f"referenced file not found: {p}")
    return p


def load_scenario(
    path: str | Path,
    *,
    fmt: str | None = None,
    check: bool = True,
    warn: bool = True,
) -> ResolvedScenario:
    """Read, validate, resolve and (with ``check``) guard a scenario file.

    Guard errors raise (the first error's exception class, message listing all errors);
    guard warnings are emitted as ``ValidityWarning`` when ``warn``.
    """
    p = Path(path)
    sc = parse_scenario(read_document(p, fmt=fmt), label=str(p))
    # with check, the guard report carries the resolution notes as warnings (no duplicates)
    resolved = resolve_scenario(sc, p.resolve().parent, warn=warn and not check, strict=not check)
    if check:
        from sremu.scenario.guards import check_scenario

        report = check_scenario(resolved)
        if warn:
            report.emit_warnings()
        report.raise_for_errors()
    return resolved
