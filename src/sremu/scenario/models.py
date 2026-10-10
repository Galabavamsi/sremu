"""Scenario and state-library models (pydantic v2; DESIGN.md sections 3.1, 5.5, 9.1-9.4).

Field names are frozen by section 9.2. Rules (section 9.1):

* ``extra="forbid"`` everywhere; every dimensional field carries a unit suffix
  (``_hz _sps _m _s _dbm _db _deg _rad _mps _w``).
* **Exact quantities** (sample rates and every time that maps to a sample: durations, event
  times, periods, latencies) are strings or ints parsed to ``Fraction``
  (``"23.04e6"``, ``"1/200"``, ``"5e-3"``). Floats are rejected (:class:`FloatNotExactError`,
  which is both a ``SchemaError`` and a ``TypeError``, R1). In JSON they serialize as exact
  decimal strings (``"0.0005"``) or, when the value has no finite decimal expansion, as
  ``"p/q"``.
* Other numbers are finite JSON numbers: strings and bools are not coerced to numbers, and
  ints are never coerced from floats or strings (``seed: "7"`` is an error).
* Every default is explicit here and so appears in ``scenario.resolved.yaml``.

The models check each object on its own (types, ranges, local consistency). Checks that need
the whole scenario, files on disk or physics (unique ids, references, links, distances to
cells, state libraries, bandwidths) are the guards in :mod:`sremu.scenario.guards`.
"""

from __future__ import annotations

import itertools
import math
import re
from fractions import Fraction
from typing import Annotated, Any, Final, Literal, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    WithJsonSchema,
    model_validator,
)
from pydantic_core import PydanticCustomError

from sremu.contracts import DataClass, LatencyModel, ObservationField, SourceKind
from sremu.errors import SchemaError

__all__ = [
    "EXACT_PATTERN",
    "SCHEMA_VERSION",
    "AngleGrid",
    "Antenna",
    "CapabilitiesModel",
    "CompactFamily",
    "ComplexValue",
    "ConstantVelocityTrajectory",
    "ControlModel",
    "ControllerModel",
    "Created",
    "Diffuse",
    "DiffuseTap",
    "Edge",
    "Environment",
    "Exact",
    "ExactNonNeg",
    "ExactPos",
    "ExternalTrajectory",
    "FailuresModel",
    "FamilyParameter",
    "FloatNotExactError",
    "GammaPoint",
    "Grid",
    "Ground",
    "InitialModel",
    "LatencyModelSpec",
    "LoModel",
    "Node",
    "Numerics",
    "ObservationModel",
    "Occluder",
    "PacedModel",
    "Pose",
    "Provenance",
    "RecordModel",
    "RisPanel",
    "RunModel",
    "Rwp1dTrajectory",
    "Rwp2dTrajectory",
    "Scenario",
    "StateEntry",
    "StateLibrary",
    "StatesModel",
    "StaticTrajectory",
    "StreamModel",
    "TabulatedPattern",
    "Tiles",
    "TraceTrajectory",
    "Traffic",
    "Trajectory",
    "TransitionModelSpec",
    "Vec3",
    "Wall",
    "WaypointsTrajectory",
    "exact_to_str",
    "parse_exact_field",
]

SCHEMA_VERSION: Final[str] = "1.0"
"""Scenario and state-library schema version (``schema_version``)."""


class FloatNotExactError(SchemaError, TypeError):
    """A float was given for an exact quantity (rate or sample-mapped time) in a scenario.

    Raised by the loaders; it is a ``SchemaError`` (so a ``ConfigurationError``) and a
    ``TypeError`` (R1: "floats raise TypeError in scenario files").
    """


# --------------------------------------------------------------------------- exact numbers

EXACT_PATTERN: Final[str] = (
    r"^[+-]?(?:[0-9]+/0*[1-9][0-9]*|(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)$"
)
"""Accepted text of an exact quantity: a decimal (optionally with exponent) or ``p/q``."""

_EXACT_RE: Final = re.compile(EXACT_PATTERN)


def parse_exact_field(value: Any) -> Fraction:
    """Validate an exact scenario quantity (strings or ints; floats rejected, R1)."""
    if isinstance(value, bool):
        raise PydanticCustomError("exact_type", "bool is not an exact quantity")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        hint = repr(value) if math.isfinite(value) else "<finite value>"
        raise PydanticCustomError(
            "exact_float",
            "floats are rejected for exact quantities (rates and sample-mapped times); "
            "write the value as a string, e.g. '{hint}'",
            {"hint": hint},
        )
    if isinstance(value, str):
        text = value.strip()
        if not _EXACT_RE.match(text):
            raise PydanticCustomError(
                "exact_syntax",
                "'{text}' is not an exact quantity (decimal like '23.04e6' or ratio like '1/200')",
                {"text": value},
            )
        return Fraction(text)
    raise PydanticCustomError(
        "exact_type",
        "expected a string or int for an exact quantity, got {t}",
        {"t": type(value).__name__},
    )


def exact_to_str(q: Fraction | int) -> str:
    """Canonical exact text: a plain decimal when the value has a finite decimal expansion
    (``"0.0005"``, ``"23040000"``), else ``"p/q"`` (``"1/3"``). ``Fraction(s) == q``."""
    q = Fraction(q)
    num, den = q.numerator, q.denominator
    if den == 1:
        return str(num)
    d = den
    twos = fives = 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return f"{num}/{den}"
    digits = max(twos, fives)
    scaled = abs(num) * (10**digits // den)
    sign = "-" if num < 0 else ""
    whole, frac = divmod(scaled, 10**digits)
    return f"{sign}{whole}.{frac:0{digits}d}"


_EXACT_JSON_SCHEMA: Final[dict[str, Any]] = {
    "anyOf": [{"type": "string", "pattern": EXACT_PATTERN}, {"type": "integer"}],
    "description": "Exact quantity: decimal or 'p/q' string, or an integer. Floats are rejected.",
}


def _exact_pos(value: Any) -> Fraction:
    q = parse_exact_field(value)
    if q <= 0:
        raise PydanticCustomError("exact_positive", "must be > 0, got {q}", {"q": str(q)})
    return q


def _exact_nonneg(value: Any) -> Fraction:
    q = parse_exact_field(value)
    if q < 0:
        raise PydanticCustomError("exact_nonneg", "must be >= 0, got {q}", {"q": str(q)})
    return q


_exact_ser = PlainSerializer(exact_to_str, return_type=str, when_used="json")

Exact: TypeAlias = Annotated[
    Fraction, PlainValidator(parse_exact_field), _exact_ser, WithJsonSchema(_EXACT_JSON_SCHEMA)
]
"""An exact quantity (any sign)."""
ExactPos: TypeAlias = Annotated[
    Fraction,
    PlainValidator(_exact_pos),
    _exact_ser,
    WithJsonSchema({**_EXACT_JSON_SCHEMA, "description": "Exact quantity > 0 (string or int)."}),
]
"""An exact quantity > 0 (rates, durations, periods)."""
ExactNonNeg: TypeAlias = Annotated[
    Fraction,
    PlainValidator(_exact_nonneg),
    _exact_ser,
    WithJsonSchema({**_EXACT_JSON_SCHEMA, "description": "Exact quantity >= 0 (string or int)."}),
]
"""An exact quantity >= 0 (times, latencies)."""


# --------------------------------------------------------------------------- plain numbers


def _real_number(value: Any) -> Any:
    """Accept int/float only: no bools and no numeric strings (JSON-Schema-consistent)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PydanticCustomError(
            "number_type", "expected a number, got {t}", {"t": type(value).__name__}
        )
    return value


Float: TypeAlias = Annotated[float, BeforeValidator(_real_number)]
PosFloat: TypeAlias = Annotated[float, BeforeValidator(_real_number), Field(gt=0)]
NonNegFloat: TypeAlias = Annotated[float, BeforeValidator(_real_number), Field(ge=0)]
UnitFloat: TypeAlias = Annotated[float, BeforeValidator(_real_number), Field(ge=0, le=1)]
StrictInt: TypeAlias = Annotated[int, Field(strict=True)]
StrictBool: TypeAlias = Annotated[bool, Field(strict=True)]
Vec3: TypeAlias = tuple[Float, Float, Float]
"""A 3-vector (x, y, z); positions in m, directions dimensionless."""

IdStr: TypeAlias = Annotated[str, Field(pattern=r"^[A-Za-z0-9_\-]+$", min_length=1, max_length=64)]
"""A node, stream, panel or object id (also an RNG key component, section 3.2)."""
StreamRef: TypeAlias = Annotated[
    str, Field(pattern=r"^[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+$", max_length=129)
]
"""A stream id ``<node>.<stream>``."""


def _complex(value: Any) -> complex:
    if isinstance(value, complex):
        z = value
    elif isinstance(value, bool):
        raise PydanticCustomError("complex_type", "bool is not a complex number")
    elif isinstance(value, int | float):
        z = complex(float(value), 0.0)
    elif isinstance(value, list | tuple) and len(value) == 2:
        re_, im_ = value
        for part in (re_, im_):
            if isinstance(part, bool) or not isinstance(part, int | float):
                raise PydanticCustomError("complex_type", "[re, im] must be two numbers")
        z = complex(float(re_), float(im_))
    else:
        raise PydanticCustomError(
            "complex_type", "expected a real number or [re, im], got {v!r}", {"v": value}
        )
    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
        raise PydanticCustomError("complex_finite", "complex value must be finite")
    return z


ComplexValue: TypeAlias = Annotated[
    complex,
    PlainValidator(_complex),
    PlainSerializer(lambda z: [z.real, z.imag], return_type=list[float], when_used="json"),
    WithJsonSchema(
        {
            "anyOf": [
                {"type": "number"},
                {
                    "type": "array",
                    "prefixItems": [{"type": "number"}, {"type": "number"}],
                    "minItems": 2,
                    "maxItems": 2,
                },
            ],
            "description": "Complex number: a real number or [re, im].",
        }
    ),
]
"""A complex value written as a real number or ``[re, im]``; serialized as ``[re, im]``."""


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
        populate_by_name=False,
        use_enum_values=True,
    )


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


# =========================================================================== scenario


class Numerics(_Model):
    """Numerical policy (section 9.2). Defaults are the spec's."""

    page_target_s: ExactPos = Field(
        default=Fraction(1, 2000),
        json_schema_extra={"default": "5e-4"},
        description="Page duration target, s (4.3).",
    )
    interp_rel_error: PosFloat = Field(default=1e-3, description="Shared epsilon_rel (4.4).")
    t_u_max_s: ExactPos = Field(
        default=Fraction(1, 100),
        json_schema_extra={"default": "1e-2"},
        description="Snapshot interval cap, s (4.4).",
    )
    t_u_s: ExactPos | None = Field(
        default=None, description="Snapshot interval, s; may only LOWER the automatic value."
    )
    fd_error_target_db: Float = Field(
        default=-60.0, description="FD in-band error target, dB (4.5)."
    )
    narrowband_phase_rad: PosFloat = Field(default=0.05, description="pi*B*dtau threshold, rad.")
    point_max_pitch_wavelengths: PosFloat = Field(default=0.5, description="auto -> point limit.")
    subcell_max_wavelengths: PosFloat = Field(default=0.2, description="po_patch sub-cell size.")
    richardson_tol_db: PosFloat = Field(default=0.05, description="Richardson tolerance, dB (5.4).")
    deterministic: StrictBool = True
    backend: Literal["numba_cpu", "numpy_ref"] = "numba_cpu"
    verify_interp: StrictBool = True

    @model_validator(mode="after")
    def _t_u_below_cap(self) -> Numerics:
        if self.t_u_s is not None and self.t_u_s > self.t_u_max_s:
            raise ValueError(
                f"numerics.t_u_s = {exact_to_str(self.t_u_s)} s exceeds t_u_max_s = "
                f"{exact_to_str(self.t_u_max_s)} s; t_u_s may only lower the automatic value"
            )
        return self


class Ground(_Model):
    """Flat ground plane z = ``height_m`` with complex permittivity (section 5.7)."""

    eps_r: PosFloat
    sigma_s_per_m: NonNegFloat
    polarization: Literal["H", "V"]
    height_m: Float = 0.0


class Wall(_Model):
    """Finite planar polygon wall, image order 1 (section 5.8)."""

    id: IdStr
    vertices_m: list[Vec3] = Field(min_length=3)
    eps_r: PosFloat
    sigma_s_per_m: NonNegFloat
    roughness_m: NonNegFloat = 0.0


class Occluder(_Model):
    """Opaque occluder or partition polygon (section 5.9)."""

    id: IdStr
    vertices_m: list[Vec3] = Field(min_length=3)
    kind: Literal["opaque", "partition"]
    transmission_loss_db: NonNegFloat | None = None

    @model_validator(mode="after")
    def _loss_iff_partition(self) -> Occluder:
        if self.kind == "partition" and self.transmission_loss_db is None:
            raise ValueError("a partition needs transmission_loss_db")
        if self.kind == "opaque" and self.transmission_loss_db is not None:
            raise ValueError("an opaque occluder takes no transmission_loss_db")
        return self


class Edge(_Model):
    """Knife edge given as a segment; applies to the paths it intersects (section 5.9)."""

    id: IdStr
    p0_m: Vec3
    p1_m: Vec3


class DiffuseTap(_Model):
    """One exponential-PDP tap of the diffuse term."""

    delay_s: NonNegFloat
    rel_power_db: Float


class Diffuse(_Model):
    """Additive sum-of-sinusoids diffuse multipath (section 5.11)."""

    k_factor_db: Float
    max_doppler_hz: NonNegFloat | None = None
    from_motion: StrictBool = False
    n_sinusoids: StrictInt = Field(default=48, ge=48)
    taps: list[DiffuseTap] | None = Field(default=None, min_length=1, max_length=4)

    @model_validator(mode="after")
    def _doppler_source(self) -> Diffuse:
        if (self.max_doppler_hz is None) == (not self.from_motion):
            raise ValueError("give exactly one of max_doppler_hz or from_motion: true")
        return self


class Environment(_Model):
    """Ground, walls, occluders, knife edges and diffuse multipath (section 9.2)."""

    ground: Ground | None = None
    walls: list[Wall] = Field(default_factory=list)
    occluders: list[Occluder] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    diffuse: Diffuse | None = None


# --------------------------------------------------------------------------- trajectories


class StaticTrajectory(_Model):
    """Fixed position (section 5.10)."""

    model: Literal["static"]
    position_m: Vec3


class WaypointsTrajectory(_Model):
    """Piecewise linear through ``positions_m`` at exact times ``t_s`` (strictly
    increasing, same length); held before the first and after the last waypoint."""

    model: Literal["waypoints"]
    t_s: list[ExactNonNeg] = Field(min_length=1)
    positions_m: list[Vec3] = Field(min_length=1)

    @model_validator(mode="after")
    def _increasing(self) -> WaypointsTrajectory:
        if len(self.t_s) != len(self.positions_m):
            raise ValueError("waypoints need as many t_s as positions_m")
        if any(b <= a for a, b in itertools.pairwise(self.t_s)):
            raise ValueError("waypoint times must be strictly increasing")
        return self


class ConstantVelocityTrajectory(_Model):
    """``position(t) = start_m + velocity_mps * t``."""

    model: Literal["constant_velocity"]
    start_m: Vec3
    velocity_mps: Vec3


class Rwp1dTrajectory(_Model):
    """1-D random waypoint on the segment [p0_m, p1_m] (Bettstetter), constant speed, legs
    keyed by RNG purpose ``mobility/<node>``."""

    model: Literal["rwp_1d"]
    p0_m: Vec3
    p1_m: Vec3
    speed_mps: PosFloat
    pause_s: ExactNonNeg = Field(default=Fraction(0), json_schema_extra={"default": "0"})

    @model_validator(mode="after")
    def _nondegenerate(self) -> Rwp1dTrajectory:
        d = (self.p1_m[0] - self.p0_m[0], self.p1_m[1] - self.p0_m[1], self.p1_m[2] - self.p0_m[2])
        if _norm(d) == 0.0:
            raise ValueError("rwp_1d needs p0_m != p1_m")
        return self


class Rwp2dTrajectory(_Model):
    """2-D random waypoint in the rectangle ``bounds_m = [[x0, y0], [x1, y1]]`` at height
    ``z_m``."""

    model: Literal["rwp_2d"]
    bounds_m: tuple[tuple[Float, Float], tuple[Float, Float]]
    z_m: Float = 0.0
    speed_mps: PosFloat
    pause_s: ExactNonNeg = Field(default=Fraction(0), json_schema_extra={"default": "0"})

    @model_validator(mode="after")
    def _ranges(self) -> Rwp2dTrajectory:
        (x0, y0), (x1, y1) = self.bounds_m
        if not (x0 < x1 and y0 < y1):
            raise ValueError("rwp_2d bounds_m must be [[x0, y0], [x1, y1]] with x0 < x1, y0 < y1")
        return self


class TraceTrajectory(_Model):
    """CSV trace ``t_s,x_m,y_m,z_m`` (cubic spline, no extrapolation); path relative to the
    scenario file."""

    model: Literal["trace"]
    file: str = Field(min_length=1)
    data_class: DataClass


class ExternalTrajectory(_Model):
    """Positions arrive as timed ``node.set_position`` / ``node.set_trajectory`` events."""

    model: Literal["external"]
    initial_m: Vec3


Trajectory: TypeAlias = Annotated[
    StaticTrajectory
    | WaypointsTrajectory
    | ConstantVelocityTrajectory
    | Rwp1dTrajectory
    | Rwp2dTrajectory
    | TraceTrajectory
    | ExternalTrajectory,
    Field(discriminator="model"),
]


# --------------------------------------------------------------------------- antennas


class CosPowerParams(_Model):
    """``CosPower(alpha)``: G0 = 2(alpha + 1) (section 5.2)."""

    alpha: NonNegFloat


class CosPowerPattern(_Model):
    cos_power: CosPowerParams


class AntennaProvenance(_Model):
    """Provenance of a tabulated antenna pattern."""

    description: str
    reference: str
    data_class: DataClass


class TabulatedParams(_Model):
    file: str = Field(min_length=1)
    provenance: AntennaProvenance


class TabulatedPattern(_Model):
    tabulated: TabulatedParams


class Antenna(_Model):
    """Scalar co-polar antenna (section 5.2); ``boresight`` required unless isotropic."""

    pattern: Literal["isotropic"] | CosPowerPattern | TabulatedPattern
    boresight: Vec3 | None = None

    @model_validator(mode="after")
    def _boresight(self) -> Antenna:
        if self.pattern == "isotropic":
            if self.boresight is not None:
                raise ValueError("an isotropic antenna takes no boresight")
        elif self.boresight is None or _norm(self.boresight) == 0.0:
            raise ValueError("a directional antenna needs a nonzero boresight")
        return self


# --------------------------------------------------------------------------- nodes


class LoModel(_Model):
    """Local oscillator offsets (section 6.3)."""

    cfo_hz: Float = 0.0
    linewidth_hz: NonNegFloat = 0.0


class StreamModel(_Model):
    """A TX or RX stream of a node (sections 3.3, 4.5, 9.2).

    ``occupied_bw_hz`` (TX only) defaults to ``0.8*f_s`` at resolution, with a warning.
    ``noise_figure_db`` and ``adc_clip`` are RX-only.
    """

    name: IdStr
    direction: Literal["tx", "rx"]
    fc_hz: PosFloat
    rate_sps: ExactPos
    occupied_bw_hz: PosFloat | None = None
    full_scale_dbm: Float = 0.0
    gain_db: Float = 0.0
    noise_figure_db: NonNegFloat = 0.0
    lo: LoModel = Field(default_factory=LoModel)
    adc_clip: StrictBool = False

    @model_validator(mode="after")
    def _direction_fields(self) -> StreamModel:
        if self.direction == "tx":
            if "noise_figure_db" in self.model_fields_set and self.noise_figure_db != 0.0:
                raise ValueError("noise_figure_db applies to RX streams only")
            if self.adc_clip:
                raise ValueError("adc_clip applies to RX streams only")
        elif self.occupied_bw_hz is not None:
            raise ValueError("occupied_bw_hz is declared on TX streams only")
        return self


class Node(_Model):
    """A node: exactly one of ``position_m`` or ``trajectory``."""

    id: IdStr
    position_m: Vec3 | None = None
    trajectory: Trajectory | None = None
    antenna: Antenna
    streams: list[StreamModel] = Field(min_length=1)

    @model_validator(mode="after")
    def _one_position_source(self) -> Node:
        if (self.position_m is None) == (self.trajectory is None):
            raise ValueError("give exactly one of position_m or trajectory")
        return self


# --------------------------------------------------------------------------- RIS panels


class Pose(_Model):
    """Panel pose: centre (m), unit normal n and in-plane u axis; v = n x u (section 3.1)."""

    center_m: Vec3
    normal: Vec3
    u_axis: Vec3

    @model_validator(mode="after")
    def _orthonormal(self) -> Pose:
        nn, nu = _norm(self.normal), _norm(self.u_axis)
        if nn == 0.0 or nu == 0.0:
            raise ValueError("pose.normal and pose.u_axis must be nonzero")
        dot = sum(a * b for a, b in zip(self.normal, self.u_axis, strict=True)) / (nn * nu)
        if abs(dot) > 1e-6:
            raise ValueError(f"pose.u_axis must be orthogonal to pose.normal (cos = {dot:.3g})")
        return self


class Grid(_Model):
    """Cell grid; cell index ``n = iv*n_u + iu`` (u fastest)."""

    n_u: StrictInt = Field(ge=1)
    n_v: StrictInt = Field(ge=1)
    pitch_u_m: PosFloat
    pitch_v_m: PosFloat
    cell_model: Literal["auto", "point", "po_patch"] = "auto"


class Tiles(_Model):
    """Tile layout; ``gap_m`` between adjacent tiles; ``gap_gamma`` optional structural
    reflection of exposed gaps (None = inert)."""

    n_tiles_u: StrictInt = Field(ge=1)
    n_tiles_v: StrictInt = Field(ge=1)
    gap_m: NonNegFloat
    gap_gamma: ComplexValue | None = None

    @model_validator(mode="after")
    def _passive_gap(self) -> Tiles:
        if self.gap_gamma is not None and abs(self.gap_gamma) > 1.0 + 1e-9:
            raise ValueError("tiles.gap_gamma must be passive (|gap_gamma| <= 1)")
        return self


class StatesModel(_Model):
    """Response states: a built-in ``library`` (``"name"`` or ``"name(k=v, ...)"`` for
    parametric families) or a state-library ``file``; ``assume_mag`` per state label for
    ``mag: null`` states (recorded as ASSUMED)."""

    library: str | None = None
    file: str | None = None
    assume_mag: dict[str, UnitFloat] | None = None
    efficiency: Annotated[float, BeforeValidator(_real_number), Field(gt=0, le=1)] = 1.0
    override_efficiency: StrictBool = False

    @model_validator(mode="after")
    def _one_source(self) -> StatesModel:
        if (self.library is None) == (self.file is None):
            raise ValueError("give exactly one of states.library or states.file")
        return self


class ControlModel(_Model):
    """Control granularity; ``groups`` lists physical cell indices per control group."""

    granularity: Literal["element", "column", "tile", "groups"] = "element"
    groups: list[list[Annotated[int, Field(strict=True, ge=0)]]] | None = None

    @model_validator(mode="after")
    def _groups(self) -> ControlModel:
        if self.granularity == "groups" and not self.groups:
            raise ValueError("granularity 'groups' needs control.groups")
        if self.granularity != "groups" and self.groups is not None:
            raise ValueError("control.groups is only used with granularity 'groups'")
        return self


class InitialModel(_Model):
    """Initial configuration: a ``fill`` state for every control or explicit ``states``."""

    fill: Annotated[int, Field(strict=True, ge=0)] | None = None
    states: list[Annotated[int, Field(strict=True, ge=0)]] | None = None

    @model_validator(mode="after")
    def _one(self) -> InitialModel:
        if (self.fill is None) == (self.states is None):
            raise ValueError("give exactly one of initial.fill or initial.states")
        return self


class FailuresModel(_Model):
    """Random failure map rates (keyed RNG purpose ``failure/<panel>``)."""

    dead_rate: UnitFloat = 0.0
    stuck_rate: UnitFloat = 0.0
    gamma_dead: ComplexValue = 0.0

    @model_validator(mode="after")
    def _rates(self) -> FailuresModel:
        if self.dead_rate + self.stuck_rate > 1.0:
            raise ValueError("dead_rate + stuck_rate must be <= 1")
        if abs(self.gamma_dead) > 1.0 + 1e-9:
            raise ValueError("gamma_dead must be passive (|gamma_dead| <= 1)")
        return self


class TransitionModelSpec(_Model):
    """Default transition (section 4.7); ``duration_s`` defaults to the panel's
    ``reconfiguration_time_s``."""

    model: Literal["step", "linear", "first_order"] = "step"
    duration_s: ExactPos | None = None
    classes: Literal["single", "by_edge"] = "single"


class CapabilitiesModel(_Model):
    """ETSI GS RIS 008-style capabilities (sections 8.5, 8.9)."""

    reconfiguration_time_s: ExactNonNeg | None = None
    max_command_rate_hz: PosFloat = 1e4
    static_power_w: NonNegFloat | None = None


class RisPanel(_Model):
    """A RIS panel (section 9.2)."""

    id: IdStr
    enabled: StrictBool = True
    pose: Pose
    grid: Grid
    tiles: Tiles | None = None
    states: StatesModel
    control: ControlModel = Field(default_factory=ControlModel)
    initial: InitialModel = Field(default_factory=lambda: InitialModel(fill=0))
    failures: FailuresModel | None = None
    transition: TransitionModelSpec = Field(default_factory=TransitionModelSpec)
    capabilities: CapabilitiesModel = Field(default_factory=CapabilitiesModel)

    @model_validator(mode="after")
    def _layout(self) -> RisPanel:
        if self.tiles is not None and (
            self.grid.n_u % self.tiles.n_tiles_u or self.grid.n_v % self.tiles.n_tiles_v
        ):
            raise ValueError(
                "grid.n_u and grid.n_v must be multiples of tiles.n_tiles_u and n_tiles_v"
            )
        if self.control.granularity == "tile" and self.tiles is None:
            raise ValueError("granularity 'tile' needs a tiles layout")
        if self.control.groups is not None:
            n = self.n_cells
            seen = [c for g in self.control.groups for c in g]
            if sorted(seen) != list(range(n)):
                raise ValueError(f"control.groups must partition the {n} cells exactly once")
        if self.initial.states is not None and len(self.initial.states) != self.n_controls:
            raise ValueError(
                f"initial.states has {len(self.initial.states)} entries; the panel has "
                f"{self.n_controls} controls"
            )
        if (
            self.transition.model != "step"
            and self.transition.duration_s is None
            and self.capabilities.reconfiguration_time_s is None
        ):
            raise ValueError(
                f"transition model '{self.transition.model}' needs transition.duration_s or "
                "capabilities.reconfiguration_time_s"
            )
        return self

    @property
    def n_cells(self) -> int:
        """Physical cell count N = n_u*n_v."""
        return self.grid.n_u * self.grid.n_v

    @property
    def n_controls(self) -> int:
        """Length N_ctrl of a ``states`` command for this granularity."""
        g = self.control.granularity
        if g == "element":
            return self.n_cells
        if g == "column":
            return self.grid.n_u
        if g == "tile":
            assert self.tiles is not None
            return self.tiles.n_tiles_u * self.tiles.n_tiles_v
        assert self.control.groups is not None
        return len(self.control.groups)


# --------------------------------------------------------------------------- controllers


class LatencyModelSpec(_Model):
    """Controller latency: ``fixed_s`` plus optional ``jitter`` ``"uniform:a,b"`` or
    ``"exp:mean"`` (s), drawn from RNG purpose ``ctrl.latency/<name>``."""

    fixed_s: ExactNonNeg
    jitter: str | None = None

    @model_validator(mode="after")
    def _jitter(self) -> LatencyModelSpec:
        if self.jitter is not None:
            try:
                LatencyModel(self.fixed_s, self.jitter)
            except Exception as exc:  # ConfigurationError / ValueError / TypeError
                raise ValueError(str(exc)) from None
        return self


_DEFAULT_OBSERVATION: Final[tuple[ObservationField, ...]] = ("rx_power_dbm", "ris_state")


class ObservationModel(_Model):
    """What the controller observes (mirrors ``contracts.ObservationSpec``)."""

    fields: list[ObservationField] = Field(default_factory=lambda: list(_DEFAULT_OBSERVATION))
    window_s: ExactPos = Field(default=Fraction(1, 1000), json_schema_extra={"default": "1e-3"})
    position_delay_s: ExactNonNeg = Field(default=Fraction(0), json_schema_extra={"default": "0"})
    position_noise_m: NonNegFloat = 0.0


class ControllerModel(_Model):
    """A controller (section 9.2). Built-ins name their class in ``impl``."""

    name: IdStr
    kind: Literal["builtin", "external"]
    impl: str | None = None
    period_s: ExactPos
    min_latency_s: ExactNonNeg
    latency: LatencyModelSpec
    observation: ObservationModel = Field(default_factory=ObservationModel)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _impl_and_latency(self) -> ControllerModel:
        if self.kind == "builtin" and not self.impl:
            raise ValueError("a builtin controller needs impl")
        if self.kind == "external" and self.impl is not None:
            raise ValueError("an external controller takes no impl")
        if self.latency.fixed_s < self.min_latency_s:
            raise ValueError(
                f"latency.fixed_s = {exact_to_str(self.latency.fixed_s)} s is below "
                f"min_latency_s = {exact_to_str(self.min_latency_s)} s (actions would violate "
                "the lookahead, 4.9)"
            )
        return self


# --------------------------------------------------------------------------- run / record


class PacedModel(_Model):
    """Paced-mode settings (section 4.10)."""

    d_lag_s: ExactNonNeg = Field(default=Fraction(1, 1000), json_schema_extra={"default": "1e-3"})
    slacks_s: list[ExactNonNeg] = Field(
        default_factory=lambda: [Fraction(0), Fraction(1, 1000), Fraction(1, 200)],
        json_schema_extra={"default": ["0", "1e-3", "5e-3"]},
    )


class RunModel(_Model):
    """Run mode and policies (sections 4.6-4.10). ``stall_timeout_s`` is wall time."""

    mode: Literal["lockstep", "paced"] = "lockstep"
    coupling: Literal["barrier", "async"] = "barrier"
    duration_s: ExactPos
    late_policy: Literal["apply_on_arrival", "reject"] = "apply_on_arrival"
    on_stall: Literal["wait", "zero_fill"] = "wait"
    stall_timeout_s: PosFloat = 2.0
    paced: PacedModel = Field(default_factory=PacedModel)

    @model_validator(mode="after")
    def _coupling(self) -> RunModel:
        if self.mode == "lockstep" and self.coupling == "async":
            raise ValueError(
                "mode 'lockstep' with coupling 'async' is rejected (non-deterministic, 4.9)"
            )
        return self


class RecordModel(_Model):
    """What a run records (section 3.6)."""

    iq: StrictBool = False
    events: StrictBool = True
    metrics: StrictBool = True
    snapshots: StrictBool = False


class Traffic(_Model):
    """An in-process traffic generator feeding a TX stream (studies)."""

    stream: StreamRef
    generator: Literal["tone", "bpsk", "qpsk_packets", "file"]
    params: dict[str, Any] = Field(default_factory=dict)


class Scenario(_Model):
    """Root of a scenario file (section 9.2)."""

    schema_version: Literal["1.0"]
    name: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$", max_length=64)]
    seed: Annotated[int, Field(strict=True, ge=0, lt=2**63)]
    description: str = ""
    numerics: Numerics = Field(default_factory=Numerics)
    environment: Environment = Field(default_factory=Environment)
    nodes: list[Node] = Field(min_length=1)
    ris: list[RisPanel] = Field(default_factory=list)
    controllers: list[ControllerModel] = Field(default_factory=list)
    run: RunModel
    record: RecordModel = Field(default_factory=RecordModel)
    traffic: list[Traffic] = Field(default_factory=list)

    # ---- convenience views (no validation side effects)
    def streams(self) -> dict[str, tuple[Node, StreamModel]]:
        """Stream id ``<node>.<stream>`` -> (node, stream), in scenario order."""
        return {f"{n.id}.{s.name}": (n, s) for n in self.nodes for s in n.streams}

    def node(self, node_id: str) -> Node:
        """The node with id ``node_id`` (KeyError if absent)."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def panel(self, panel_id: str) -> RisPanel:
        """The RIS panel with id ``panel_id`` (KeyError if absent)."""
        for p in self.ris:
            if p.id == panel_id:
                return p
        raise KeyError(panel_id)


# =========================================================================== state libraries


class Created(_Model):
    """Creation record of a table."""

    date: str
    tool: str
    version: str | None = None


class Provenance(_Model):
    """Mandatory response-table provenance (section 5.5). ``data_class`` (sremu addition)
    is the manifest data class of the table's values; when omitted it follows from
    ``source_kind``."""

    source_kind: SourceKind
    description: str
    reference: str
    reference_plane_offset_m: Float
    polarization: Literal["co", "cross"]
    includes_cell_pattern: StrictBool
    sigma_mag_db: NonNegFloat
    sigma_phase_deg: NonNegFloat
    raw_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None
    license: str
    created: Created
    assumptions: list[str] = Field(default_factory=list)
    data_class: DataClass | None = None

    def resolved_data_class(self) -> str:
        """``data_class`` or the default for ``source_kind``."""
        if self.data_class is not None:
            return str(self.data_class)
        return {
            "measured": "MEAS",
            "digitized": "DIG",
            "assumed": "ASSUMED",
            "analytic": "ANALYTIC",
            "circuit": "ANALYTIC",
            "fullwave": "REPORTED",
        }[self.source_kind]


class AngleGrid(_Model):
    """Angle axis shared by incidence and reflection (section 5.5), degrees in [-90, 90]."""

    theta_deg: list[Annotated[float, BeforeValidator(_real_number), Field(ge=-90, le=90)]] = Field(
        min_length=2
    )

    @model_validator(mode="after")
    def _increasing(self) -> AngleGrid:
        t = self.theta_deg
        if any(b <= a for a, b in itertools.pairwise(t)):
            raise ValueError("angle_grid.theta_deg must be strictly increasing")
        return self


class GammaPoint(_Model):
    """One response sample: magnitude (linear; ``null`` = must be assumed by the scenario)
    and phase in degrees at ``f_hz`` (``null`` for frequency-independent libraries) and,
    for angle tables, ``theta_i_deg``/``theta_s_deg`` (or ``theta_deg`` for incidence-only
    tables). Optional per-entry uncertainties override the provenance scalars."""

    f_hz: PosFloat | None = None
    mag: NonNegFloat | None
    phase_deg: Float
    theta_i_deg: Float | None = None
    theta_s_deg: Float | None = None
    theta_deg: Float | None = None
    sigma_mag_db: NonNegFloat | None = None
    sigma_phase_deg: NonNegFloat | None = None


class StateEntry(_Model):
    """One state of the alphabet."""

    id: Annotated[int, Field(strict=True, ge=0)]
    label: Annotated[str, Field(pattern=r"^[A-Za-z0-9_\-+.]+$", max_length=64)]
    gamma: list[GammaPoint] = Field(min_length=1)


class FamilyParameter(_Model):
    """A parameter of an analytic state family."""

    default: Float | None = None
    description: str
    unit: str = ""
    integer: StrictBool = False
    min: Float | None = None
    max: Float | None = None


class CompactFamily(_Model):
    """An analytic, parametric state family expanded at load time (``ideal_bbit``,
    ``abeywickrama``); see :func:`sremu.scenario.load.expand_family`."""

    name: Literal["ideal_bbit", "abeywickrama"]
    formula: str
    parameters: dict[str, FamilyParameter]


class StateLibrary(_Model):
    """A RIS state-library file (``data/states/*.yaml``, section 9.4).

    Either explicit ``states`` or an analytic ``family`` (expanded with parameters into
    explicit states). ``frequency_hz: "any"`` marks a frequency-independent analytic table;
    ``validity_bw_hz: null`` means no bandwidth limit. ``angle_grid: "none"`` marks an
    angle-independent (separable) table.
    """

    schema_version: Literal["1.0"]
    kind: Literal["ris_state_library"]
    name: Annotated[
        str, Field(pattern=r"^[A-Za-z0-9_\-.]+(?:\([A-Za-z0-9_.,=+\- ]*\))?$", max_length=160)
    ]
    provenance: Provenance
    frequency_hz: Annotated[list[PosFloat], Field(min_length=1)] | Literal["any"]
    validity_bw_hz: PosFloat | None
    angle_grid: Literal["none"] | AngleGrid
    incidence_only: StrictBool = False
    angle_mapping: Literal["half_difference"] | None = None
    reciprocal: StrictBool = True
    reconfiguration_time_s: ExactNonNeg | None = None
    family: CompactFamily | None = None
    parameters: dict[str, Float] | None = Field(
        default=None, description="Bound parameters of an expanded analytic family (provenance)."
    )
    states: list[StateEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _structure(self) -> StateLibrary:
        if not self.reciprocal:
            raise ValueError("non-reciprocal tables (reciprocal: false) are rejected in release 1")
        if self.angle_mapping is not None and not self.incidence_only:
            raise ValueError("angle_mapping is only used by incidence-only tables")
        if self.incidence_only and self.angle_mapping is None:
            raise ValueError(
                "an incidence-only table needs angle_mapping: half_difference (section 5.5)"
            )
        if self.incidence_only and self.angle_grid == "none":
            raise ValueError("an incidence-only table needs an angle_grid")
        if self.family is None:
            if not self.states:
                raise ValueError("a state library needs states (or an analytic family)")
            self._check_states()
        else:
            if self.states:
                raise ValueError("give either states or family, not both")
            if self.parameters is not None:
                raise ValueError(
                    "an unexpanded family takes no parameters (defaults live in "
                    "family.parameters; parameters records an expansion)"
                )
        return self

    def _check_states(self) -> None:
        ids = [s.id for s in self.states]
        if ids != list(range(len(ids))):
            raise ValueError(f"state ids must be 0..S-1 in order, got {ids}")
        labels = [s.label for s in self.states]
        if len(set(labels)) != len(labels):
            raise ValueError(f"state labels must be unique, got {labels}")
        freqs = None if self.frequency_hz == "any" else list(self.frequency_hz)
        if freqs is not None and any(b <= a for a, b in itertools.pairwise(freqs)):
            raise ValueError("frequency_hz must be strictly increasing")
        grid = None if self.angle_grid == "none" else list(self.angle_grid.theta_deg)
        for s in self.states:
            keys: list[tuple[float | None, ...]] = []
            for g in s.gamma:
                if freqs is None:
                    if g.f_hz is not None:
                        raise ValueError(
                            f"state {s.label}: f_hz must be null for frequency_hz: any"
                        )
                elif g.f_hz not in freqs:
                    raise ValueError(f"state {s.label}: f_hz {g.f_hz} is not in frequency_hz")
                if grid is None:
                    if (g.theta_i_deg, g.theta_s_deg, g.theta_deg) != (None, None, None):
                        raise ValueError(
                            f"state {s.label}: angles given for an angle-independent table"
                        )
                    keys.append((g.f_hz,))
                elif self.incidence_only:
                    if (
                        g.theta_deg is None
                        or g.theta_i_deg is not None
                        or g.theta_s_deg is not None
                    ):
                        raise ValueError(
                            f"state {s.label}: incidence-only entries take theta_deg only"
                        )
                    if g.theta_deg not in grid:
                        raise ValueError(f"state {s.label}: theta_deg {g.theta_deg} not in grid")
                    keys.append((g.f_hz, g.theta_deg))
                else:
                    if g.theta_i_deg is None or g.theta_s_deg is None or g.theta_deg is not None:
                        raise ValueError(
                            f"state {s.label}: angle entries need theta_i_deg and theta_s_deg"
                        )
                    if g.theta_i_deg not in grid or g.theta_s_deg not in grid:
                        raise ValueError(f"state {s.label}: angle not in angle_grid")
                    keys.append((g.f_hz, g.theta_i_deg, g.theta_s_deg))
            n_f = 1 if freqs is None else len(freqs)
            n_a = 1 if grid is None else (len(grid) if self.incidence_only else len(grid) ** 2)
            if len(set(keys)) != len(keys) or len(keys) != n_f * n_a:
                raise ValueError(
                    f"state {s.label}: needs exactly one entry per grid point "
                    f"({n_f * n_a}), got {len(keys)} (duplicates: {len(keys) - len(set(keys))})"
                )

    # ---- views
    @property
    def labels(self) -> tuple[str, ...]:
        """State labels in id order."""
        return tuple(s.label for s in self.states)

    @property
    def n_states(self) -> int:
        """Alphabet size S (0 for an unexpanded family)."""
        return len(self.states)

    def null_mag_labels(self) -> tuple[str, ...]:
        """Labels of states with at least one ``mag: null`` entry."""
        return tuple(s.label for s in self.states if any(g.mag is None for g in s.gamma))

    def is_separable(self) -> bool:
        """True for angle-independent tables (S,1,1,1) or (S,F,1,1)."""
        return self.angle_grid == "none"
