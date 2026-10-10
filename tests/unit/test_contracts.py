"""contracts.py: every section 3.4 type exists and validates its documented invariants."""

from __future__ import annotations

import dataclasses
import hashlib
import struct
from fractions import Fraction

import numpy as np
import pytest

from sremu import contracts as c
from sremu.errors import ConfigurationError, ErrorCode, RateMismatchError, TooLarge

pytestmark = pytest.mark.unit

SECTION_3_4_NAMES = [
    "CONTRACT_VERSION",
    "StreamId",
    "LinkId",
    "PanelId",
    "PathKind",
    "PathBundle",
    "RISCoefficients",
    "Snapshot",
    "PanelConfig",
    "LinkPaths",
    "PropagationModel",
    "LatencyModel",
    "ObservationSpec",
    "Observation",
    "Action",
    "Controller",
    "CommandRecord",
    "EngineProtocol",
    "LinkJob",
    "SegmentJob",
    "RenderFn",
    "NoiseFn",
    # auxiliary types referenced by sections 3.4, 4 and 8
    "StreamSpec",
    "StreamHandle",
    "TxMetadata",
    "RxMetadata",
    "AsyncMetadata",
    "StreamCmd",
    "CommandRequest",
    "CommandReceipt",
    "CommandTime",
    "RisCapabilities",
    "TransitionSpec",
    "TransitionTerm",
    "FDSpec",
    "CsiView",
    "MotionBounds",
    "TxRing",
    "RemoteControllerProxy",
    "RenderBackend",
]
FROZEN = [
    "PathBundle",
    "RISCoefficients",
    "Snapshot",
    "PanelConfig",
    "LinkPaths",
    "LatencyModel",
    "ObservationSpec",
    "Observation",
    "Action",
    "StreamSpec",
    "StreamHandle",
    "StreamCmd",
    "CommandRequest",
    "CommandReceipt",
    "CommandTime",
    "RisCapabilities",
    "TransitionSpec",
    "TransitionTerm",
    "FDSpec",
    "CsiView",
    "MotionBounds",
    "AsyncMetadata",
]


def test_contract_version_and_all_names():
    assert c.CONTRACT_VERSION == "1.0"
    for name in SECTION_3_4_NAMES:
        assert hasattr(c, name), name
    for name in c.__all__:
        assert hasattr(c, name), name


@pytest.mark.parametrize("name", FROZEN)
def test_frozen_dataclasses(name):
    cls = getattr(c, name)
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_mutable_records_per_spec():
    for name in ("CommandRecord", "LinkJob", "SegmentJob", "TxMetadata", "RxMetadata"):
        assert not getattr(c, name).__dataclass_params__.frozen, name


def test_path_kind_values():
    assert [int(k) for k in c.PathKind] == [0, 1, 2, 3, 4, 5]
    assert c.PathKind.RIS_TILE == 4


def _bundle(**kw):
    base = {
        "link": "a.tx0->b.rx0",
        "t": Fraction(1, 3),
        "amp": np.array([1e-3 + 0j], dtype=np.complex128),
        "delay_s": np.array([1e-8]),
        "kind": np.array([0], dtype=np.int8),
    }
    base.update(kw)
    return c.PathBundle(**base)


def test_path_bundle_validates_dtypes_and_lengths():
    pb = _bundle(t=0)
    assert pb.t == Fraction(0) and isinstance(pb.t, Fraction)
    assert pb.n_paths == 1
    with pytest.raises(TypeError):
        _bundle(amp=np.array([1e-3], dtype=np.complex64))
    with pytest.raises(ValueError):
        _bundle(delay_s=np.array([1e-8, 2e-8]))
    with pytest.raises(TypeError):
        _bundle(t=0.5)
    with pytest.raises(TypeError):
        _bundle(kind=np.array([0], dtype=np.int64))


def _ris(**kw):
    r, s, n, k = 2, 2, 6, 3
    base = {
        "panel": "ris0",
        "t": Fraction(0),
        "rows": (("a.tx0->b.rx0", 0), ("a.tx0->b.rx0", 3)),
        "tile_of_cell": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        "tile_delay_s": np.zeros((r, k)),
        "B": None,
        "C": np.ones((r, n), dtype=np.complex128),
        "gamma": np.array([1, -1], dtype=np.complex128),
    }
    base.update(kw)
    assert s == 2
    return c.RISCoefficients(**base)


def test_ris_coefficients_forms_and_shapes():
    rc = _ris()
    assert (rc.n_rows, rc.n_cells, rc.n_tiles, rc.n_states, rc.separable) == (2, 6, 3, 2, True)
    b = c.RISCoefficients(
        panel="ris0",
        t=0,
        rows=(("l", 0),),
        tile_of_cell=np.zeros(4, np.int32),
        tile_delay_s=np.zeros((1, 1)),
        B=np.zeros((1, 3, 4), np.complex128),
        C=None,
        gamma=None,
    )
    assert (b.n_states, b.separable) == (3, False)
    with pytest.raises(ValueError):
        _ris(B=np.zeros((2, 2, 6), np.complex128))  # both forms
    with pytest.raises(ValueError):
        _ris(C=None, gamma=None)  # neither form
    with pytest.raises(ValueError):
        _ris(rows=(("l", 0), ("l", 4)))  # hop variant out of range
    with pytest.raises(ValueError):
        _ris(tile_of_cell=np.array([0, 0, 1, 1, 2, 3], dtype=np.int32))  # tile >= K
    with pytest.raises(ValueError):
        _ris(C=np.ones((2, 5), dtype=np.complex128))


def test_panel_config_hash_matches_independent_blake2b():
    state = np.array([0, 1, 1, -1, 3], dtype=np.int16)
    raw = struct.pack("<5h", 0, 1, 1, -1, 3)
    on = hashlib.blake2b(raw + b"\x01", digest_size=16).hexdigest()
    off = hashlib.blake2b(raw + b"\x00", digest_size=16).hexdigest()
    assert c.PanelConfig("ris0", state).config_hash() == on
    assert c.PanelConfig("ris0", state, enabled=False).config_hash() == off
    assert c.config_hash(state.astype(">i2"), True) == on  # byte order normalized
    assert len(on) == 32
    with pytest.raises(TypeError):
        c.PanelConfig("ris0", state.astype(np.int32))
    with pytest.raises(ValueError):
        c.PanelConfig("ris0", np.array([-2], dtype=np.int16))


def test_latency_model_parsing():
    lm = c.LatencyModel(fixed_s="2e-4", jitter="uniform:0,2e-3")  # type: ignore[arg-type]
    assert lm.fixed_s == Fraction(1, 5000)
    assert lm.parsed_jitter() == ("uniform", (Fraction(0), Fraction(1, 500)))
    assert c.LatencyModel(jitter="exp:1e-3").parsed_jitter() == ("exp", (Fraction(1, 1000),))
    assert c.LatencyModel().parsed_jitter() is None
    for bad in ("gauss:1", "uniform:2e-3,0", "uniform:1", "exp:-1"):
        with pytest.raises(ConfigurationError):
            c.LatencyModel(jitter=bad)
    with pytest.raises(TypeError):
        c.LatencyModel(fixed_s=2e-4)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        c.LatencyModel(mode="sleepy")  # type: ignore[arg-type]


def test_observation_spec_and_action_validation():
    spec = c.ObservationSpec(fields=("rx_power_dbm", "csi_oracle"), window_s="1/500")  # type: ignore[arg-type]
    assert spec.window_s == Fraction(1, 500)
    with pytest.raises(ConfigurationError):
        c.ObservationSpec(fields=("telepathy",))
    with pytest.raises(ConfigurationError):
        c.ObservationSpec(window_s=Fraction(0))
    a = c.Action("ris0", states=np.zeros(4, np.int16), at_s="5e-3")  # type: ignore[arg-type]
    assert a.at_s == Fraction(1, 200)
    assert c.Action("ris0", enabled=False).enabled is False
    with pytest.raises(ConfigurationError):
        c.Action("ris0")
    with pytest.raises(ConfigurationError):
        c.Action("ris0", states=np.zeros(4, np.int16), codebook_index=3)
    with pytest.raises(ConfigurationError):
        c.Action("ris0", codebook_index=1, enabled=True)


def test_command_time_json_round_trip():
    for ct in (
        c.CommandTime.at("1/200"),
        c.CommandTime.at_sample("rx1.rx0", 115200),
        c.CommandTime.asap(),
    ):
        assert c.CommandTime.from_json(ct.to_json()) == ct
    assert c.CommandTime.at("5e-3").to_json() == {"t": "1/200"}
    with pytest.raises(ConfigurationError):
        c.CommandTime("t", t=Fraction(1), stream="a.b")
    with pytest.raises(ConfigurationError):
        c.CommandTime.from_json({"time": "1"})
    with pytest.raises(TypeError):
        c.CommandTime.from_json({"t": 0.005})


def test_command_request_limits_and_payload_rules():
    at = c.CommandTime.at(Fraction(1, 200))
    ok = c.CommandRequest(
        "c-1",
        "ris.set_config",
        "ctrl",
        "ris0",
        at,
        states=np.zeros(8, np.int16),
        elements=[0, 1, 2, 3, 4, 5, 6, 7],
    )  # type: ignore[arg-type]
    assert ok.elements == tuple(range(8))
    assert not ok.counterfactual
    with pytest.raises(TooLarge):
        c.CommandRequest("x" * 65, "ris.set_config", "s", "ris0", at, codebook_index=1)
    with pytest.raises(TooLarge):
        c.CommandRequest(
            "c",
            "ris.set_config",
            "s",
            "ris0",
            at,
            states=np.zeros(c.MAX_ELEMENTS_PER_COMMAND + 1, np.int16),
        )
    with pytest.raises(ConfigurationError):
        c.CommandRequest("c", "ris.set_config", "s", "ris0", at)
    with pytest.raises(ConfigurationError):
        c.CommandRequest("c", "ris.explode", "s", "ris0", at)  # type: ignore[arg-type]
    en = c.CommandRequest("c", "ris.set_enabled", "s", "ris0", at, enabled=False)
    assert en.counterfactual
    with pytest.raises(ConfigurationError):
        c.CommandRequest("c", "node.set_position", "s", "rx1", at, position_m=np.zeros(2))


def test_command_record_round_trip_and_receipt_dict():
    rec = c.CommandRecord(
        cmd_id="c-7f3a",
        seq=412,
        src="ctrl-a",
        op="ris.set_config",
        panel="ris0",
        t_req=Fraction(1, 200),
        t_applied=Fraction(1, 200),
        k_req={"rx1.rx0": 5000},
        k_applied={"rx1.rx0": 5000},
        lateness_samples={"rx1.rx0": 0},
        t_wall_recv_ns=10,
        t_wall_computed_ns=None,
        t_wall_visible_ns=None,
        status="applied",
        config_hash="0" * 32,
        transition={"model": "linear", "duration_s": "1/50000"},
    )
    d = rec.to_dict()
    assert d["t_req"] == "1/200"
    assert list(d) == [f.name for f in dataclasses.fields(c.CommandRecord)]
    assert c.CommandRecord.from_dict(d) == rec
    r = c.CommandReceipt(
        "c-7f3a", True, "scheduled", seq=412, t_req=Fraction(1, 200), k_req={"rx1.rx0": 5000}
    )
    assert r.to_dict() == {
        "id": "c-7f3a",
        "ok": True,
        "status": "scheduled",
        "seq": 412,
        "t_req": "1/200",
        "k_req": {"rx1.rx0": 5000},
        "t_wall_recv_ns": None,
    }
    e = c.CommandReceipt("c-1", False, "rejected", error=ErrorCode.LATE_COMMAND, msg="late")
    assert e.to_dict()["error"] == {"code": "LATE_COMMAND", "msg": "late"}


def test_stream_spec_and_ids():
    s = c.StreamSpec("ap.tx0", "tx", 3.5e9, "23.04e6")  # type: ignore[arg-type]
    assert s.rate_sps == Fraction(23_040_000)
    assert s.node == "ap"
    with pytest.raises(ConfigurationError):
        c.StreamSpec("ap.tx0", "tx", 3.5e9, "1e6", nchan=2)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        c.StreamSpec("aptx0", "tx", 3.5e9, "1e6")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        c.StreamSpec("ap.tx0", "tx", 3.5e9, 1e6)  # type: ignore[arg-type]
    assert c.link_id("ap.tx0", "mu.rx0") == "ap.tx0->mu.rx0"
    assert c.split_link_id("ap.tx0->mu.rx0") == ("ap.tx0", "mu.rx0")
    assert c.stream_id("rx07", "rx0") == "rx07.rx0"
    with pytest.raises(ConfigurationError):
        c.split_link_id("ap.tx0-mu.rx0")


def test_check_link_rule_section_3_2():
    f = Fraction(10**6)
    assert c.check_link(3.5e9, f, 3.5e9 + 0.49e6, f)
    assert not c.check_link(3.5e9, f, 3.5e9 + 0.5e6, f)  # |dfc| < fs/2 is strict
    assert not c.check_link(1e9, f, 3.5e9, Fraction(23_040_000))  # far apart, different rates
    with pytest.raises(RateMismatchError):
        c.check_link(3.5e9, f, 3.5e9, Fraction(23_040_000))
    # overlap at different rates uses (fs_tx + fs_rx)/2: 1 MS/s and 3 MS/s overlap below 2 MHz
    with pytest.raises(RateMismatchError):
        c.check_link(1e9, f, 1e9 + 1.99e6, Fraction(3 * 10**6))
    assert not c.check_link(1e9, f, 1e9 + 2.0e6, Fraction(3 * 10**6))
    with pytest.raises(TypeError):
        c.check_link(3.5e9, 1e6, 3.5e9, 1e6)  # type: ignore[arg-type]  # exact rates only


def test_ris_capabilities_defaults_and_dict():
    cap = c.RisCapabilities(
        n_elements=120,
        layout=(24, 5),
        control_granularity="element",
        states=("sigma1", "sigma2"),
        gamma_source="analytic:lab_binary_3p5ghz",
        operating_band_hz=(3.4e9, 3.6e9),
        max_bandwidth_hz=2e7,
        reconfiguration_time_s="2e-5",  # type: ignore[arg-type]
    )
    assert cap.n_controls == 120
    assert cap.reconfiguration_time_s == Fraction(1, 50000)
    assert cap.max_command_rate_hz == 1e4
    d = cap.to_dict()
    assert d["reconfiguration_time_s"] == "1/50000"
    assert d["layout"] == [24, 5]
    with pytest.raises(ConfigurationError):
        dataclasses.replace(cap, gamma_source="magic")


def test_csi_view_evaluate_matches_manual_sum():
    z = np.array([[1 + 1j, 2, 3j], [-1, 0.5j, 4]], dtype=np.complex128)
    csi = c.CsiView("a.tx0->b.rx0", Fraction(0), 3.5e9, 0.25 + 0j, {"ris0": z})
    st = np.array([1, -1, 0], dtype=np.int16)
    assert csi.evaluate({"ris0": st}) == 0.25 + (-1) + 3j
    with pytest.raises(ValueError):  # a short state vector must not drop cells silently
        csi.evaluate({"ris0": np.array([1, 0], dtype=np.int16)})
    with pytest.raises(ValueError):
        csi.evaluate({"ris0": np.array([2, 0, 0], dtype=np.int16)})  # S = 2


def test_transition_and_fd_specs():
    t = c.TransitionSpec("linear", "2e-5")  # type: ignore[arg-type]
    assert t.duration_s == Fraction(1, 50000)
    with pytest.raises(ConfigurationError):
        c.TransitionSpec("cubic")  # type: ignore[arg-type]
    assert c.FDSpec.integer().lookahead == 0
    with pytest.raises(ConfigurationError):
        c.StreamCmd("num_done", num_samps=-1)


def test_metadata_defaults_mirror_uhd():
    md = c.RxMetadata()
    assert md.error_code == ErrorCode.NONE and not md.has_time_spec
    assert c.TxMetadata().time_spec.exact() == 0
    assert c.StreamCmd("start_cont").stream_now
    assert set(c.COUNTER_NAMES) >= {
        "late_samples",
        "underflow_slip_samples",
        "overflow_pages",
        "bad_packets",
        "rec_dropped",
        "stall_zero_samples",
    }


def test_protocols_are_satisfied_by_the_fakes():
    from sremu.testing import FakeEngine
    from sremu.testing.scenes import two_node_static

    assert isinstance(two_node_static(), c.PropagationModel)
    assert isinstance(FakeEngine(), c.EngineProtocol)
