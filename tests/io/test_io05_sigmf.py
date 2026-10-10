"""IO-05: SigMF structural checks (DESIGN.md 8.6, 10.4).

Oracles: the SigMF v1.2 rules (required ``core:datatype``/``core:version``, captures and
annotations sorted by ``core:sample_start``, declared extension namespaces, collection
``core:streams`` hashed with SHA-512 of the meta file), the sremu conventions of section 8.6
(``cf32_le``, ``core:global_index`` = absolute index, one annotation per applied command at
``k_applied``) and hand-computed byte layouts and timestamps.
"""

from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from sremu.api import Device, RxMetadata, StreamArgs, StreamCmd, TimeSpec, TxMetadata
from sremu.contracts import CommandRecord, StreamSpec
from sremu.record.sigmf import (
    COLLECTION_NAME,
    SIGMF_VERSION,
    SREMU_EXTENSION,
    Capture,
    RunRecorder,
    SigMFWriter,
    file_index,
    read_recording,
    validate_collection,
    validate_meta,
    validate_recording,
)
from tests.io import _wp7_support as sup

pytestmark = pytest.mark.io

FC = 3.5e9


def _rec(seq, status, k_applied, *, op="ris.set_config", k_req=None, cmd_id=None):
    k_req = dict(k_applied) if k_req is None else k_req
    return CommandRecord(
        cmd_id=cmd_id or f"c{seq}",
        seq=seq,
        src="test",
        op=op,
        panel="ris0",
        t_req=Fraction(0),
        t_applied=Fraction(seq, 1000),
        k_req=k_req,
        k_applied=dict(k_applied),
        lateness_samples={s: k_applied[s] - k_req[s] for s in k_applied},
        t_wall_recv_ns=None,
        t_wall_computed_ns=None,
        t_wall_visible_ns=None,
        status=status,
        config_hash=f"{seq:032x}",
        transition=None,
    )


def _samples(rng, n):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


# --------------------------------------------------------------------------- writer


def test_io05_writer_layout_captures_and_bytes(tmp_path, rng):
    w = SigMFWriter(
        tmp_path / "iq" / "mu.rx0", stream="mu.rx0", direction="rx", rate_sps="1e6", fc_hz=FC
    )
    a, b, c = _samples(rng, 100), _samples(rng, 40), _samples(rng, 50)
    w.write(0, a)
    w.write(100, b)  # contiguous: same capture
    w.write(250, c)  # gap of 110 samples: new capture
    meta_path = w.close()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    g = meta["global"]
    assert g["core:datatype"] == "cf32_le"
    assert g["core:sample_rate"] == 1e6
    assert g["core:version"] == SIGMF_VERSION == "1.2.6"
    assert g["core:extensions"] == [{"name": "sremu", "version": "1.0.0", "optional": True}]
    assert g["sremu:rate_sps"] == "1000000"
    assert g["sremu:stream"] == "mu.rx0"
    assert g["sremu:direction"] == "rx"
    assert meta["captures"] == [
        {"core:sample_start": 0, "core:global_index": 0, "core:frequency": FC},
        {"core:sample_start": 140, "core:global_index": 250, "core:frequency": FC},
    ]
    assert meta["annotations"] == []
    # cf32_le: interleaved little-endian float32 I, Q
    raw = (tmp_path / "iq" / "mu.rx0.sigmf-data").read_bytes()
    want = np.concatenate([a, b, c])
    inter = np.empty(2 * want.size, "<f4")
    inter[0::2], inter[1::2] = want.real, want.imag
    assert raw == inter.tobytes()
    assert validate_recording(meta_path) == []
    r = read_recording(meta_path)
    assert r.captures == (Capture(0, 0, 140), Capture(140, 250, 50))
    assert r.rate_sps == Fraction(1_000_000)
    np.testing.assert_array_equal(r.data.view(np.uint32), want.view(np.uint32))


def test_io05_writer_rejects_overlap_and_backwards(tmp_path):
    w = SigMFWriter(tmp_path / "x", stream="a.tx0", direction="tx", rate_sps=10, fc_hz=FC)
    w.write(5, np.ones(10, np.complex64))
    with pytest.raises(ValueError, match="overlaps"):
        w.write(14, np.ones(3, np.complex64))
    with pytest.raises(ValueError, match="overlaps"):
        w.write(0, np.ones(1, np.complex64))


def test_io05_file_index_mapping():
    caps = (Capture(0, 10, 5), Capture(5, 30, 4))  # absolute [10,15) and [30,34)
    assert file_index(caps, 9) is None  # before the recording
    assert [file_index(caps, k) for k in (10, 14)] == [0, 4]
    assert file_index(caps, 20) == 5  # in the gap: first recorded sample after it
    assert [file_index(caps, k) for k in (30, 33)] == [5, 8]
    assert file_index(caps, 34) is None  # after the end


def test_io05_datetime_from_epoch_to_the_nanosecond(tmp_path):
    w = SigMFWriter(
        tmp_path / "x",
        stream="a.rx0",
        direction="rx",
        rate_sps=3,
        fc_hz=FC,
        epoch_datetime="2026-10-10T00:00:00Z",
    )
    w.write(1, np.ones(1, np.complex64))  # t = 1/3 s
    w.write(5, np.ones(1, np.complex64))  # t = 5/3 s
    meta = json.loads(w.close().read_text(encoding="utf-8"))
    assert [c["core:datetime"] for c in meta["captures"]] == [
        "2026-10-10T00:00:00.333333333Z",  # 333333333.33 ns rounds down
        "2026-10-10T00:00:01.666666667Z",  # 1666666666.67 ns rounds up
    ]
    assert validate_meta(meta, n_samples=2) == []


# --------------------------------------------------------------------------- annotations


def test_io05_annotations_one_per_applied_command_sorted(tmp_path, rng):
    rec = RunRecorder(tmp_path)
    rec.open_stream(StreamSpec("ap.tx0", "tx", FC, Fraction(1000)))
    rec.open_stream(StreamSpec("mu.rx0", "rx", FC, Fraction(1000)))
    rec.write("ap.tx0", 0, _samples(rng, 50))
    rec.write("mu.rx0", 10, _samples(rng, 20))  # absolute [10, 30)
    rec.write("mu.rx0", 40, _samples(rng, 10))  # absolute [40, 50), dataset [20, 30)
    k = "mu.rx0"
    records = [
        _rec(1, "applied", {k: 45}),  # in the 2nd segment -> 20 + 5 = 25
        _rec(2, "late_applied", {k: 12}, k_req={k: 9}),  # -> 2, lateness 3
        _rec(3, "superseded", {k: 15}),  # never takes effect: no annotation
        _rec(4, "rejected", {}),
        _rec(5, "scheduled", {k: 20}),  # not applied (yet): no annotation
        _rec(6, "applied", {k: 35}, op="node.set_position"),  # gap -> next segment start 20
        _rec(7, "applied", {k: 5}),  # before the recording: not visible
        _rec(8, "applied", {k: 50}),  # after the end: not visible
        _rec(9, "applied", {k: 10}, op="rec.marker"),
    ]
    for r in records:
        rec.on_command(r)
    idx = rec.close()
    meta = json.loads(idx.metas["mu.rx0"].read_text(encoding="utf-8"))
    ann = meta["annotations"]
    got = [(a["core:sample_start"], a["sremu:seq"], a["core:label"]) for a in ann]
    assert got == [(0, 9, "marker"), (2, 2, "ris_cfg"), (20, 6, "node_cfg"), (25, 1, "ris_cfg")]
    a2 = ann[1]
    assert a2["sremu:k_applied"] == 12
    assert a2["sremu:k_req"] == 9
    assert a2["sremu:lateness"] == 3
    assert a2["sremu:status"] == "late_applied"
    assert a2["sremu:config_hash"] == f"{2:032x}"
    assert a2["sremu:t_applied"] == "1/500"
    tx_meta = json.loads(idx.metas["ap.tx0"].read_text(encoding="utf-8"))
    assert tx_meta["annotations"] == []  # commands map to RX samples only
    assert validate_collection(tmp_path) == []


# --------------------------------------------------------------------------- collection


def test_io05_collection_lists_meta_files_with_sha512(tmp_path, rng):
    rec = RunRecorder(tmp_path)
    for sid, d in (("ap.tx0", "tx"), ("mu.rx0", "rx")):
        rec.open_stream(StreamSpec(sid, d, FC, Fraction(1000)))
        rec.write(sid, 0, _samples(rng, 8))
    idx = rec.close()
    coll = json.loads((tmp_path / COLLECTION_NAME).read_text(encoding="utf-8"))
    assert set(coll) == {"collection"}
    c = coll["collection"]
    assert c["core:version"] == SIGMF_VERSION
    assert c["core:extensions"] == [SREMU_EXTENSION]
    want = [
        {
            "name": f"iq/{sid}",
            "hash": hashlib.sha512(idx.metas[sid].read_bytes()).hexdigest(),
        }
        for sid in ("ap.tx0", "mu.rx0")
    ]
    assert c["core:streams"] == want
    assert validate_collection(tmp_path) == []
    # teeth: any change to a meta file breaks its hash
    p = idx.metas["mu.rx0"]
    p.write_bytes(p.read_bytes().replace(b'"rx"', b'"RX"'))
    probs = validate_collection(tmp_path)
    assert any("SHA-512 of mu.rx0.sigmf-meta does not match" in x for x in probs)


# --------------------------------------------------------------------------- validator teeth


def _good_meta():
    return {
        "global": {
            "core:datatype": "cf32_le",
            "core:version": "1.2.6",
            "core:sample_rate": 1000.0,
            "core:extensions": [dict(SREMU_EXTENSION)],
            "sremu:stream": "mu.rx0",
            "sremu:direction": "rx",
            "sremu:rate_sps": "1000",
        },
        "captures": [
            {"core:sample_start": 0, "core:global_index": 0},
            {"core:sample_start": 10, "core:global_index": 30},
        ],
        "annotations": [
            {"core:sample_start": 3, "sremu:k_applied": 3},
            {"core:sample_start": 12, "sremu:k_applied": 32},
        ],
    }


def test_io05_validator_accepts_the_reference_meta():
    assert validate_meta(_good_meta(), n_samples=20) == []


def _drop(path):
    def f(m):
        obj = m
        for k in path[:-1]:
            obj = obj[k]
        del obj[path[-1]]

    return f


def _set(path, value):
    def f(m):
        obj = m
        for k in path[:-1]:
            obj = obj[k]
        obj[path[-1]] = value

    return f


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (_drop(["global", "core:datatype"]), "core:datatype"),
        (_set(["global", "core:datatype"], "cf33_le"), "core:datatype"),
        (_drop(["global", "core:version"]), "core:version"),
        (_set(["global", "core:version"], "1.2"), "core:version"),
        (_drop(["captures"]), "missing top-level 'captures'"),
        (_drop(["annotations"]), "missing top-level 'annotations'"),
        (_set(["captures", 1, "core:sample_start"], 0), "sorted by core:sample_start"),
        (_set(["captures", 1, "core:global_index"], 5), "overlaps the previous segment"),
        (_set(["annotations", 0, "core:sample_start"], 15), "sorted by core:sample_start"),
        (_drop(["annotations", 1, "core:sample_start"]), "core:sample_start None"),
        (_set(["annotations", 1, "core:sample_start"], 20), "beyond the dataset"),
        (_set(["annotations", 1, "sremu:k_applied"], 33), "dataset index 13"),
        (_set(["annotations", 0, "acme:x"], 1), "namespace 'acme'"),
        (_set(["global", "nonamespace"], 1), "has no namespace"),
        (_set(["captures", 0, "core:datetime"], "2026-10-10 00:00:00"), "core:datetime"),
        (_set(["global", "core:extensions"], [{"name": "sremu", "version": "1"}]), "optional"),
        (_set(["global", "sremu:rate_sps"], "999"), "disagrees with core:sample_rate"),
        (_set(["global", "core:sample_rate"], -1.0), "positive number"),
    ],
)
def test_io05_validator_teeth(mutate, needle):
    m = copy.deepcopy(_good_meta())
    mutate(m)
    probs = validate_meta(m, n_samples=20)
    assert any(needle in p for p in probs), probs


def test_io05_validator_dataset_size_teeth(tmp_path):
    w = SigMFWriter(tmp_path / "a", stream="a.rx0", direction="rx", rate_sps=10, fc_hz=FC)
    w.write(0, np.ones(4, np.complex64))
    meta_path = w.close()
    assert validate_recording(meta_path) == []
    data = tmp_path / "a.sigmf-data"
    data.write_bytes(data.read_bytes()[:-3])  # 29 bytes: not whole cf32 samples
    assert any("not a whole number" in p for p in validate_recording(meta_path))
    data.unlink()
    assert any("dataset file is missing" in p for p in validate_recording(meta_path))


def test_io05_sigmf_package_validator_if_installed(tmp_path, rng):
    """The official ``sigmf`` validator (extra ``sigmf``) accepts sremu recordings with
    gaps, epoch timestamps and sremu annotations, and the collection hashes verify."""
    sigmf = pytest.importorskip("sigmf")
    from sigmf import sigmffile

    rec = RunRecorder(tmp_path, epoch_datetime="2026-10-10T12:00:00Z")
    rec.open_stream(StreamSpec("ap.tx0", "tx", FC, Fraction(1000)))
    rec.open_stream(StreamSpec("mu.rx0", "rx", FC, Fraction(1000)))
    rec.write("ap.tx0", 0, _samples(rng, 64))
    rec.write("mu.rx0", 0, _samples(rng, 32))
    rec.write("mu.rx0", 40, _samples(rng, 24))
    rec.on_command(_rec(1, "applied", {"mu.rx0": 7}))
    rec.on_command(_rec(2, "late_applied", {"mu.rx0": 45}, k_req={"mu.rx0": 41}))
    idx = rec.close()
    for meta_path in idx.metas.values():
        handle = sigmffile.fromfile(str(meta_path))
        handle.validate()
        assert handle.get_global_field("core:datatype") == "cf32_le"
    rx = sigmffile.fromfile(str(idx.metas["mu.rx0"]))
    assert [a["core:sample_start"] for a in rx.get_annotations()] == [7, 37]
    assert rx.read_samples().shape == (56,)
    coll = sigmffile.fromfile(str(tmp_path / COLLECTION_NAME))
    assert isinstance(coll, sigmf.sigmffile.SigMFCollection)
    coll.verify_stream_hashes()
    assert sorted(coll.get_stream_names()) == ["iq/ap.tx0", "iq/mu.rx0"]
    # teeth: the official reader rejects a collection whose meta hash no longer matches
    p = idx.metas["mu.rx0"]
    p.write_bytes(p.read_bytes().replace(b'"rx"', b'"RX"'))
    with pytest.raises(sigmf.error.SigMFFileError, match="does not match"):
        sigmffile.fromfile(str(tmp_path / COLLECTION_NAME))


# --------------------------------------------------------------------------- end to end


def test_io05_device_recording_end_to_end(tmp_path, rng):
    run = tmp_path / "run"
    dev = Device.from_engine(sup.make_engine(), record=run)
    tx = dev.get_tx_stream(StreamArgs("ap.tx0", FC, "1e6"))
    rx = dev.get_rx_stream(StreamArgs("mu.rx0", FC, "1e6"))
    x = _samples(rng, 1000)
    md = TxMetadata(
        has_time_spec=True, time_spec=TimeSpec(0, 0.0), start_of_burst=True, end_of_burst=True
    )
    assert tx.send(x, md) == 1000
    r = dev.ris("ris0").set_config(states=np.ones(8, np.int16), time_spec=TimeSpec(0, 4e-4))
    assert r.ok
    assert r.k_req == {"mu.rx0": 400}
    rx.issue_stream_cmd(StreamCmd("num_done", num_samps=1000))
    out = np.zeros(1000, np.complex64)
    got = 0
    md_rx = RxMetadata()
    while got < 1000:
        got += rx.recv(out[got:], md_rx, timeout=0.0)
    dev.close()
    assert validate_collection(run) == []
    tx_rec = read_recording(run / "iq" / "ap.tx0.sigmf-meta")
    rx_rec = read_recording(run / "iq" / "mu.rx0.sigmf-meta")
    np.testing.assert_array_equal(tx_rec.data.view(np.uint32), x.view(np.uint32))
    np.testing.assert_array_equal(rx_rec.data.view(np.uint32), out.view(np.uint32))
    ann = rx_rec.meta["annotations"]
    assert [(a["core:sample_start"], a["sremu:cmd_id"], a["sremu:status"]) for a in ann] == [
        (400, "api-1", "applied")
    ]
    assert tx_rec.meta["global"]["sremu:open_index"] == 0
    assert rx_rec.meta["global"]["sremu:open_index"] == 1


def test_io05_meta_is_plain_json_utf8_lf(tmp_path):
    w = SigMFWriter(tmp_path / "a", stream="a.rx0", direction="rx", rate_sps=10, fc_hz=FC)
    w.write(0, np.ones(2, np.complex64))
    raw = Path(w.close()).read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"}\n")
    assert json.loads(raw.decode("utf-8"))["global"]["core:num_channels"] == 1


def test_io05_recorder_writes_metrics_jsonl(tmp_path):
    rec = RunRecorder(tmp_path)
    rec.on_metrics({"t_s": Fraction(1, 10), "rtf": 1.25, "lat_p99_s": float("nan")})
    rec.on_metrics({"t_s": Fraction(2, 10), "rtf": float("inf"), "n": np.int64(3)})
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x) for x in lines] == [
        {"t_s": "1/10", "rtf": 1.25, "lat_p99_s": None},
        {"t_s": "1/5", "rtf": None, "n": 3},
    ]
    rec.close()
    rec.on_metrics({"t_s": 1})  # after close: ignored
    assert len((tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2
