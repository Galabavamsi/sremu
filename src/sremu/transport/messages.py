"""Control-plane messages: JSON schemas, validation and codecs (DESIGN.md sections 8.5, 8.10).

**Schemas.** The normative wire contract is the set of JSON Schema (draft 2020-12) files in
``sremu/transport/schema/`` (exported by ``sremu schema control``): ``request`` (client to
emulator), ``reply``, ``event`` (unsolicited ``ctrl.barrier`` / ``stream.async`` on the
ROUTER/DEALER connection), ``telemetry`` (PUB topics) and ``frame`` (the binary data-plane
header, documented as JSON).

**Validation before dispatch** (:func:`parse_request`), in this order, so every refusal has
one well-defined code (I6 counts it):

1. size: more than :data:`~sremu.contracts.MAX_MESSAGE_BYTES` (1 MiB) -> ``TOO_LARGE``;
2. syntax: not UTF-8, not JSON, duplicate keys, ``NaN``/``Infinity`` tokens, not an object
   -> ``SCHEMA_ERROR``;
3. element caps: more than :data:`~sremu.contracts.MAX_ELEMENTS_PER_COMMAND` (65,536) states,
   phases, indices or codebook entries -> ``TOO_LARGE``;
4. schema: the ``request`` schema -> ``SCHEMA_ERROR``.

The validator is a small, dependency-free implementation of the JSON Schema keywords the
files use (``type const enum properties required additionalProperties patternProperties
propertyNames min/maxProperties items min/maxItems min/maxLength pattern minimum maximum
exclusiveMinimum exclusiveMaximum allOf anyOf oneOf not if/then/else $ref``); its verdicts
are cross-checked against the reference ``jsonschema`` package in the tests. ``pattern``
follows ECMA-262 for a final ``$`` (end of string, so ``"x\\n"`` does not match ``^x$``).

**No file paths.** No request field is ever interpreted as a filesystem path; the only
scenario object that names a file (trajectory ``"trace"``) has no schema form here, so it
cannot be sent (section 8.10).

**Codecs** convert between envelopes and the contract types: ``CommandRequest``,
``CommandReceipt``, ``Observation`` (arrays as base64 of little-endian bytes, float64 values
through JSON exactly), ``RisCapabilities`` and stream metadata. Exact times travel as
``"p/q"`` strings (:func:`~sremu.timebase.fraction_to_str`).
"""

from __future__ import annotations

import base64
import binascii
import functools
import json
import math
import re
from collections.abc import Callable, Mapping
from fractions import Fraction
from importlib import resources
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import (
    MAX_ELEMENTS_PER_COMMAND,
    MAX_MESSAGE_BYTES,
    CommandReceipt,
    CommandRequest,
    CommandTime,
    CsiView,
    Observation,
    RisCapabilities,
    TransitionSpec,
)
from sremu.errors import ErrorCode
from sremu.timebase import fraction_to_str, parse_exact

__all__ = [
    "SCHEMA_FILES",
    "SCHEMA_VERSION",
    "ProtocolError",
    "SchemaValidator",
    "at_from_json",
    "caps_from_json",
    "caps_to_json",
    "command_from_request",
    "command_to_request",
    "decode_json",
    "encode_json",
    "error_reply",
    "exact_from_json",
    "num_from_json",
    "num_to_json",
    "observation_from_json",
    "observation_to_json",
    "parse_request",
    "receipt_from_reply",
    "receipt_reply",
    "schema",
    "validator",
]

SCHEMA_VERSION: Final[int] = 1
"""Envelope ``v`` of this implementation."""
SCHEMA_FILES: Final[tuple[str, ...]] = (
    "request.schema.json",
    "reply.schema.json",
    "event.schema.json",
    "telemetry.schema.json",
    "frame.schema.json",
)
"""The schema files of ``sremu/transport/schema/``."""

_DEG: Final[float] = math.pi / 180.0
_NONFINITE: Final[dict[str, float]] = {
    "-Infinity": -math.inf,
    "Infinity": math.inf,
    "NaN": math.nan,
}


class ProtocolError(Exception):
    """A request is refused before dispatch: ``code`` (an ``ErrorCode``), ``msg`` and the
    request ``msg_id`` when it could be read (else None)."""

    def __init__(self, code: ErrorCode, msg: str, msg_id: str | None = None) -> None:
        super().__init__(f"{code.name}: {msg}")
        self.code = code
        self.msg = msg
        self.msg_id = msg_id


# =========================================================================== schemas


@functools.cache
def schema(name: str) -> dict[str, Any]:
    """The parsed schema file ``name`` (e.g. ``"request.schema.json"``)."""
    if name not in SCHEMA_FILES:
        raise KeyError(f"unknown schema {name!r}")
    text = resources.files("sremu.transport").joinpath("schema", name).read_text("utf-8")
    out: dict[str, Any] = json.loads(text)
    return out


@functools.cache
def validator(name: str) -> SchemaValidator:
    """A cached :class:`SchemaValidator` for schema file ``name``."""
    return SchemaValidator(schema(name))


def _json_equal(a: Any, b: Any) -> bool:
    """JSON value equality (booleans are not numbers; 1 == 1.0)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


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
        if isinstance(x, bool):
            return False
        return isinstance(x, int) or (isinstance(x, float) and x.is_integer())
    raise ValueError(f"unknown JSON type {t!r}")


class SchemaValidator:
    """Validates JSON instances against one schema document (see the module docstring)."""

    def __init__(self, root: Mapping[str, Any]) -> None:
        self.root = root
        self._re: dict[str, re.Pattern[str]] = {}

    def _pattern(self, p: str) -> re.Pattern[str]:
        rx = self._re.get(p)
        if rx is None:
            src = p[:-1] + r"\Z" if p.endswith("$") and not p.endswith(r"\$") else p
            rx = re.compile(src)
            self._re[p] = rx
        return rx

    def _resolve(self, ref: str) -> Any:
        if not ref.startswith("#"):
            raise ValueError(f"only local $ref is supported, got {ref!r}")
        node: Any = self.root
        for tok in [t for t in ref[1:].split("/") if t]:
            tok = tok.replace("~1", "/").replace("~0", "~")
            node = node[tok]
        return node

    def errors(self, inst: Any, limit: int = 8) -> list[str]:
        """Validation errors (JSON-path prefixed, at most ``limit``); empty when valid."""
        out: list[str] = []
        self._check(inst, self.root, "$", out, limit)
        return out[:limit]

    def is_valid(self, inst: Any) -> bool:
        """Whether ``inst`` validates."""
        return not self.errors(inst, limit=1)

    def _ok(self, inst: Any, sch: Any, path: str) -> bool:
        return not self._sub(inst, sch, path)

    def _sub(self, inst: Any, sch: Any, path: str) -> list[str]:
        out: list[str] = []
        self._check(inst, sch, path, out, 1)
        return out

    def _check(self, x: Any, s: Any, path: str, out: list[str], limit: int) -> None:
        if len(out) >= limit:
            return
        if s is True:
            return
        if s is False:
            out.append(f"{path}: not allowed")
            return
        if "$ref" in s:
            self._check(x, self._resolve(s["$ref"]), path, out, limit)
        t = s.get("type")
        if t is not None:
            types = t if isinstance(t, list) else [t]
            if not any(_is_type(x, tt) for tt in types):
                out.append(f"{path}: expected {'/'.join(types)}, got {type(x).__name__}")
                return
        if "const" in s and not _json_equal(x, s["const"]):
            out.append(f"{path}: must be {s['const']!r}")
        if "enum" in s and not any(_json_equal(x, e) for e in s["enum"]):
            out.append(f"{path}: {x!r} not in {s['enum']!r}")
        if isinstance(x, str):
            if "minLength" in s and len(x) < s["minLength"]:
                out.append(f"{path}: shorter than {s['minLength']}")
            if "maxLength" in s and len(x) > s["maxLength"]:
                out.append(f"{path}: longer than {s['maxLength']}")
            if "pattern" in s and self._pattern(s["pattern"]).search(x) is None:
                out.append(f"{path}: does not match {s['pattern']!r}")
        if isinstance(x, int | float) and not isinstance(x, bool):
            if "minimum" in s and x < s["minimum"]:
                out.append(f"{path}: below {s['minimum']}")
            if "maximum" in s and x > s["maximum"]:
                out.append(f"{path}: above {s['maximum']}")
            if "exclusiveMinimum" in s and x <= s["exclusiveMinimum"]:
                out.append(f"{path}: not above {s['exclusiveMinimum']}")
            if "exclusiveMaximum" in s and x >= s["exclusiveMaximum"]:
                out.append(f"{path}: not below {s['exclusiveMaximum']}")
        if isinstance(x, list):
            if "minItems" in s and len(x) < s["minItems"]:
                out.append(f"{path}: fewer than {s['minItems']} items")
            if "maxItems" in s and len(x) > s["maxItems"]:
                out.append(f"{path}: more than {s['maxItems']} items")
                return
            if "items" in s:
                for i, item in enumerate(x):
                    self._check(item, s["items"], f"{path}[{i}]", out, limit)
                    if len(out) >= limit:
                        return
        if isinstance(x, dict):
            if "minProperties" in s and len(x) < s["minProperties"]:
                out.append(f"{path}: fewer than {s['minProperties']} properties")
            if "maxProperties" in s and len(x) > s["maxProperties"]:
                out.append(f"{path}: more than {s['maxProperties']} properties")
                return
            for k in s.get("required", ()):
                if k not in x:
                    out.append(f"{path}: missing {k!r}")
            props = s.get("properties", {})
            pats = s.get("patternProperties", {})
            if "propertyNames" in s:
                for k in x:
                    self._check(k, s["propertyNames"], f"{path}.<{k}>", out, limit)
            for k, v in x.items():
                matched = False
                if k in props:
                    matched = True
                    self._check(v, props[k], f"{path}.{k}", out, limit)
                for p, ps in pats.items():
                    if self._pattern(p).search(k) is not None:
                        matched = True
                        self._check(v, ps, f"{path}.{k}", out, limit)
                if not matched and "additionalProperties" in s:
                    ap = s["additionalProperties"]
                    if ap is False:
                        out.append(f"{path}: unexpected property {k!r}")
                    else:
                        self._check(v, ap, f"{path}.{k}", out, limit)
                if len(out) >= limit:
                    return
        for sub in s.get("allOf", ()):
            self._check(x, sub, path, out, limit)
        if "anyOf" in s and not any(self._ok(x, sub, path) for sub in s["anyOf"]):
            out.append(f"{path}: matches none of the allowed forms")
        if "oneOf" in s:
            n = sum(self._ok(x, sub, path) for sub in s["oneOf"])
            if n != 1:
                out.append(f"{path}: matches {n} of the forms, exactly one required")
        if "not" in s and self._ok(x, s["not"], path):
            out.append(f"{path}: matches a forbidden form")
        if "if" in s:
            if self._ok(x, s["if"], path):
                if "then" in s:
                    self._check(x, s["then"], path, out, limit)
            elif "else" in s:
                self._check(x, s["else"], path, out, limit)


# =========================================================================== JSON


def _no_constant(name: str) -> Any:
    raise ValueError(f"JSON constant {name} is not allowed")


def _no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k!r}")
        d[k] = v
    return d


def decode_json(raw: bytes | str) -> Any:
    """Strict JSON decode (UTF-8; no duplicate keys, no NaN/Infinity); ``ValueError``."""
    text = raw.decode("utf-8") if isinstance(raw, bytes | bytearray | memoryview) else raw
    try:
        return json.loads(text, parse_constant=_no_constant, object_pairs_hook=_no_dupes)
    except RecursionError as exc:
        raise ValueError("JSON nested too deeply") from exc


def encode_json(obj: Any) -> bytes:
    """Compact UTF-8 JSON (``ValueError`` on non-finite floats)."""
    return json.dumps(obj, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode(
        "utf-8"
    )


def error_reply(msg_id: str | None, code: ErrorCode, msg: str) -> dict[str, Any]:
    """An error reply envelope."""
    return {
        "v": SCHEMA_VERSION,
        "id": msg_id,
        "ok": False,
        "error": {"code": code.name, "msg": msg},
    }


# =========================================================================== parse


def _b64(s: str, what: str) -> bytes:
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(ErrorCode.SCHEMA_ERROR, f"{what}: invalid base64") from exc


def _element_caps(msg: Mapping[str, Any]) -> None:
    """Raise ``TOO_LARGE`` when a command carries more than 65,536 elements."""
    cap = MAX_ELEMENTS_PER_COMMAND
    cfg = msg.get("config")
    if isinstance(cfg, dict):
        st = cfg.get("states_b64")
        if isinstance(st, str) and len(st) // 4 * 3 > 2 * cap + 2:
            raise ProtocolError(ErrorCode.TOO_LARGE, f"more than {cap} states in one command")
        for key in ("phases_deg", "phases_rad"):
            ph = cfg.get(key)
            if isinstance(ph, list) and len(ph) > cap:
                raise ProtocolError(ErrorCode.TOO_LARGE, f"more than {cap} phases in one command")
    el = msg.get("elements")
    if isinstance(el, dict) and isinstance(el.get("indices"), list) and len(el["indices"]) > cap:
        raise ProtocolError(ErrorCode.TOO_LARGE, f"more than {cap} element indices")
    cb = msg.get("codebook_b64")
    if isinstance(cb, str) and len(cb) // 4 * 3 > 2 * cap + 2:
        raise ProtocolError(ErrorCode.TOO_LARGE, f"codebook with more than {cap} entries")
    shape = msg.get("shape")
    if (
        isinstance(shape, list)
        and len(shape) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in shape)
        and shape[0] * shape[1] > cap
    ):
        raise ProtocolError(ErrorCode.TOO_LARGE, f"codebook with more than {cap} entries")
    traj = msg.get("trajectory")
    if isinstance(traj, dict):
        for key in ("t_s", "positions_m"):
            v = traj.get(key)
            if isinstance(v, list) and len(v) > cap:
                raise ProtocolError(ErrorCode.TOO_LARGE, f"more than {cap} waypoints")


def parse_request(raw: bytes, *, max_bytes: int = MAX_MESSAGE_BYTES) -> dict[str, Any]:
    """Parse and validate one request message (see the module docstring for the order of
    the checks); returns the envelope dict or raises :class:`ProtocolError`."""
    if len(raw) > max_bytes:
        raise ProtocolError(ErrorCode.TOO_LARGE, f"message of {len(raw)} bytes > {max_bytes}")
    try:
        msg = decode_json(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(ErrorCode.SCHEMA_ERROR, f"malformed JSON: {exc}") from None
    if not isinstance(msg, dict):
        raise ProtocolError(ErrorCode.SCHEMA_ERROR, "a request must be a JSON object")
    mid = msg.get("id")
    msg_id = mid if isinstance(mid, str) and 0 < len(mid) <= 64 else None
    try:
        _element_caps(msg)
    except ProtocolError as exc:
        exc.msg_id = msg_id
        raise
    errs = validator("request.schema.json").errors(msg, limit=4)
    if errs:
        raise ProtocolError(ErrorCode.SCHEMA_ERROR, "; ".join(errs), msg_id)
    return msg


# =========================================================================== codecs


def exact_from_json(v: Any, name: str = "value") -> Fraction:
    """An exact quantity from its wire form (string or integer)."""
    if isinstance(v, float):
        if not v.is_integer():
            raise ProtocolError(ErrorCode.SCHEMA_ERROR, f"{name}: floats are not exact")
        v = int(v)
    try:
        return parse_exact(v, name=name)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ProtocolError(ErrorCode.SCHEMA_ERROR, f"{name}: {exc}") from None


def at_from_json(v: Any) -> CommandTime:
    """``CommandTime`` of an ``at`` field (default ``asap`` when absent)."""
    if v is None or v == "asap":
        return CommandTime.asap()
    if isinstance(v, dict) and "t" in v:
        return CommandTime.at(exact_from_json(v["t"], "at.t"))
    s = v["sample"]
    return CommandTime.at_sample(str(s["stream"]), int(s["index"]))


def _transition_from_json(v: Any) -> TransitionSpec | None:
    if v is None:
        return None
    d = v.get("duration_s")
    return TransitionSpec(
        v["model"],
        None if d is None else exact_from_json(d, "transition.duration_s"),
        v.get("classes", "single"),
    )


def _transition_to_json(t: TransitionSpec) -> dict[str, Any]:
    return {
        "model": t.model,
        "duration_s": None if t.duration_s is None else fraction_to_str(t.duration_s),
        "classes": t.classes,
    }


def _int16_b64(s: str, what: str) -> NDArray[np.int16]:
    b = _b64(s, what)
    if len(b) % 2:
        raise ProtocolError(ErrorCode.SCHEMA_ERROR, f"{what}: odd byte count for int16")
    return np.frombuffer(b, dtype="<i2").astype(np.int16)


def b64_of(arr: NDArray[Any], dtype: str) -> str:
    """Base64 of the little-endian bytes of ``arr`` cast to ``dtype`` (``"<i2"``...)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def command_from_request(
    msg: Mapping[str, Any], *, src: str, t_wall_recv_ns: int | None = None
) -> CommandRequest:
    """The ``CommandRequest`` of a validated ``ris.set_config``, ``ris.set_enabled``,
    ``node.set_position``, ``node.set_trajectory`` or ``rec.marker`` envelope (``src``:
    the issuing client or controller; ``t_wall_recv_ns``: when the server parsed it, a
    metrics-only wall clock). Raises :class:`ProtocolError`."""
    op = msg["op"]
    cid = str(msg["id"])
    at = at_from_json(msg.get("at"))
    late = msg.get("late_policy", "apply_on_arrival")
    common: dict[str, Any] = {"t_wall_recv_ns": t_wall_recv_ns}
    try:
        if op == "ris.set_config":
            cfg = msg["config"]
            kw: dict[str, Any] = {}
            if "states_b64" in cfg:
                kw["states"] = _int16_b64(cfg["states_b64"], "config.states_b64")
            elif "codebook_index" in cfg:
                kw["codebook_index"] = int(cfg["codebook_index"])
            elif "phases_deg" in cfg:
                kw["phases_rad"] = np.asarray(cfg["phases_deg"], dtype=np.float64) * _DEG
                kw["quantizer"] = cfg.get("quantizer", "max_real")
            else:
                kw["phases_rad"] = np.asarray(cfg["phases_rad"], dtype=np.float64)
                kw["quantizer"] = cfg.get("quantizer", "max_real")
            el = msg.get("elements", "all")
            if isinstance(el, dict):
                el = tuple(int(i) for i in el["indices"]) if "indices" in el else str(el["group"])
            return CommandRequest(
                cid,
                "ris.set_config",
                src,
                msg["panel"],
                at,
                elements=el,
                transition=_transition_from_json(msg.get("transition")),
                late_policy=late,
                **kw,
                **common,
            )
        if op == "ris.set_enabled":
            return CommandRequest(
                cid,
                op,
                src,
                msg["panel"],
                at,
                enabled=bool(msg["enabled"]),
                late_policy=late,
                **common,
            )
        if op == "node.set_position":
            return CommandRequest(
                cid,
                op,
                src,
                msg["node"],
                at,
                position_m=np.asarray(msg["position_m"], dtype=np.float64),
                late_policy=late,
                **common,
            )
        if op == "node.set_trajectory":
            return CommandRequest(
                cid,
                op,
                src,
                msg["node"],
                at,
                payload={"trajectory": dict(msg["trajectory"])},
                late_policy=late,
                **common,
            )
        if op == "rec.marker":
            payload = {"label": msg["label"], **dict(msg.get("fields") or {})}
            return CommandRequest(cid, op, src, "rec", at, payload=payload, **common)
    except ProtocolError:
        raise
    except Exception as exc:  # contract validation (ConfigurationError, TooLarge, ...)
        code = getattr(exc, "code", None)
        raise ProtocolError(
            code if isinstance(code, ErrorCode) else ErrorCode.SCHEMA_ERROR, str(exc), cid
        ) from None
    raise ProtocolError(ErrorCode.NOT_PERMITTED, f"{op} is not a command", cid)


def command_to_request(req: CommandRequest, *, msg_id: str | None = None) -> dict[str, Any]:
    """The request envelope of a ``CommandRequest`` (client side); ``phases_rad`` travel
    as the exact ``phases_rad`` form."""
    msg: dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "id": msg_id or req.cmd_id,
        "op": req.op,
        "src": req.src,
        "at": req.at.to_json(),
    }
    if req.op in ("ris.set_config", "ris.set_enabled"):
        msg["panel"] = req.target
        msg["late_policy"] = req.late_policy
    if req.op == "ris.set_config":
        if req.states is not None:
            msg["config"] = {"states_b64": b64_of(req.states, "<i2"), "dtype": "i2"}
        elif req.codebook_index is not None:
            msg["config"] = {"codebook_index": int(req.codebook_index)}
        else:
            assert req.phases_rad is not None
            msg["config"] = {
                "phases_rad": [float(v) for v in np.asarray(req.phases_rad, np.float64)],
                "quantizer": req.quantizer,
            }
        if isinstance(req.elements, str):
            msg["elements"] = "all" if req.elements == "all" else {"group": req.elements}
        else:
            msg["elements"] = {"indices": [int(i) for i in req.elements]}
        if req.transition is not None:
            msg["transition"] = _transition_to_json(req.transition)
    elif req.op == "ris.set_enabled":
        msg["enabled"] = bool(req.enabled)
    elif req.op == "node.set_position":
        assert req.position_m is not None
        msg["node"] = req.target
        msg["position_m"] = [float(v) for v in np.asarray(req.position_m, np.float64)]
        msg["late_policy"] = req.late_policy
    elif req.op == "node.set_trajectory":
        msg["node"] = req.target
        msg["trajectory"] = dict(req.payload.get("trajectory", req.payload))
        msg["late_policy"] = req.late_policy
    elif req.op == "rec.marker":
        p = dict(req.payload)
        msg["label"] = str(p.pop("label", "marker"))
        if p:
            msg["fields"] = p
    return msg


def receipt_reply(r: CommandReceipt) -> dict[str, Any]:
    """Reply envelope of a command receipt (section 8.5 reply fields)."""
    if r.ok:
        return {"v": SCHEMA_VERSION, **r.to_dict()}
    out = error_reply(r.cmd_id, r.error, r.msg or r.error.name)
    out["status"] = "rejected"
    return out


def receipt_from_reply(d: Mapping[str, Any]) -> CommandReceipt:
    """``CommandReceipt`` of a reply envelope (or of an observation's pending entry)."""
    cid = str(d.get("id") or "")
    if d.get("ok"):
        t = d.get("t_req")
        return CommandReceipt(
            cid,
            True,
            d.get("status", "scheduled"),
            seq=d.get("seq"),
            t_req=None if t is None else parse_exact(t),
            k_req={str(k): int(v) for k, v in dict(d.get("k_req") or {}).items()},
            t_wall_recv_ns=d.get("t_wall_recv_ns"),
        )
    err = dict(d.get("error") or {})
    code = ErrorCode.__members__.get(str(err.get("code")), ErrorCode.NOT_PERMITTED)
    return CommandReceipt(cid, False, "rejected", error=code, msg=str(err.get("msg", "")))


# --------------------------------------------------------------------------- caps


_CAPS_FLOATS: Final = ("max_bandwidth_hz", "max_command_rate_hz", "static_power_w")


def num_to_json(x: float) -> float | str:
    """A float for JSON: finite values as numbers (exact through ``repr``), non-finite ones
    as ``"Infinity"``, ``"-Infinity"`` or ``"NaN"`` (JSON has no such numbers)."""
    x = float(x)
    if math.isfinite(x):
        return x
    return "NaN" if math.isnan(x) else ("Infinity" if x > 0 else "-Infinity")


def num_from_json(x: Any) -> float:
    """Inverse of :func:`num_to_json`."""
    return _NONFINITE[x] if isinstance(x, str) else float(x)


def caps_to_json(caps: Mapping[str, RisCapabilities]) -> dict[str, Any]:
    """``{panel: RisCapabilities.to_dict()}`` with non-finite floats as strings."""
    out = {}
    for p, c in caps.items():
        d = c.to_dict()
        d["operating_band_hz"] = [num_to_json(v) for v in d["operating_band_hz"]]
        for k in _CAPS_FLOATS:
            if d.get(k) is not None:
                d[k] = num_to_json(d[k])
        out[p] = d
    return out


def caps_from_json(d: Mapping[str, Any]) -> dict[str, RisCapabilities]:
    """Inverse of :func:`caps_to_json`."""
    out = {}
    for p, c in d.items():
        kw = dict(c)
        kw["layout"] = tuple(kw["layout"])
        kw["states"] = tuple(kw["states"])
        kw["operating_band_hz"] = tuple(num_from_json(v) for v in kw["operating_band_hz"])
        for k in _CAPS_FLOATS:
            if kw.get(k) is not None:
                kw[k] = num_from_json(kw[k])
        if kw.get("reconfiguration_time_s") is not None:
            kw["reconfiguration_time_s"] = parse_exact(kw["reconfiguration_time_s"])
        out[str(p)] = RisCapabilities(**kw)
    return out


# --------------------------------------------------------------------------- observation

_dbm_to_json = num_to_json
_dbm_from_json = num_from_json


def observation_to_json(obs: Observation) -> dict[str, Any]:
    """Wire form of an ``Observation`` (exact: float64 values through JSON ``repr``)."""
    csi = None
    if obs.csi_oracle is not None:
        csi = {
            lk: {
                "t": fraction_to_str(v.t),
                "fc_hz": float(v.fc_hz),
                "h_direct": [float(v.h_direct.real), float(v.h_direct.imag)],
                "cascade": {
                    p: {"shape": [int(z.shape[0]), int(z.shape[1])], "c16_b64": b64_of(z, "<c16")}
                    for p, z in v.cascade.items()
                },
            }
            for lk, v in obs.csi_oracle.items()
        }
    return {
        "t_b": fraction_to_str(obs.t_b),
        "k_b": {r: int(k) for r, k in obs.k_b.items()},
        "rx_power_dbm": {r: _dbm_to_json(p) for r, p in obs.rx_power_dbm.items()},
        "positions": None
        if obs.positions is None
        else {
            n: [float(v) for v in np.asarray(p, np.float64).reshape(-1)]
            for n, p in obs.positions.items()
        },
        "csi_oracle": csi,
        "iq_window": None
        if obs.iq_window is None
        else {r: {"c8_b64": b64_of(y, "<c8")} for r, y in obs.iq_window.items()},
        "ris_state": {p: {"i2_b64": b64_of(s, "<i2")} for p, s in obs.ris_state.items()},
        "pending": [{"v": SCHEMA_VERSION, **r.to_dict()} for r in obs.pending],
    }


def _arr(s: str, dtype: str, out: Callable[[NDArray[Any]], NDArray[Any]]) -> NDArray[Any]:
    return out(np.frombuffer(base64.b64decode(s, validate=True), dtype=dtype))


def observation_from_json(d: Mapping[str, Any]) -> Observation:
    """Inverse of :func:`observation_to_json`."""
    csi = None
    if d.get("csi_oracle") is not None:
        csi = {}
        for lk, v in d["csi_oracle"].items():
            cascade = {}
            for p, z in v["cascade"].items():
                a = _arr(z["c16_b64"], "<c16", lambda x: x.astype(np.complex128))
                cascade[p] = a.reshape(int(z["shape"][0]), int(z["shape"][1]))
            csi[lk] = CsiView(
                lk,
                parse_exact(v["t"]),
                float(v["fc_hz"]),
                complex(float(v["h_direct"][0]), float(v["h_direct"][1])),
                cascade,
            )
    pos = d.get("positions")
    iq = d.get("iq_window")
    return Observation(
        t_b=parse_exact(d["t_b"]),
        k_b={str(r): int(k) for r, k in d["k_b"].items()},
        rx_power_dbm={str(r): _dbm_from_json(p) for r, p in d["rx_power_dbm"].items()},
        positions=None
        if pos is None
        else {str(n): np.asarray(p, dtype=np.float64) for n, p in pos.items()},
        csi_oracle=csi,
        iq_window=None
        if iq is None
        else {
            str(r): _arr(v["c8_b64"], "<c8", lambda x: x.astype(np.complex64))
            for r, v in iq.items()
        },
        ris_state={
            str(p): _arr(v["i2_b64"], "<i2", lambda x: x.astype(np.int16))
            for p, v in d["ris_state"].items()
        },
        pending=tuple(receipt_from_reply(r) for r in d.get("pending", [])),
    )
