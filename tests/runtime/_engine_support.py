"""Shared helpers for the WP6 engine tests (not collected: no ``test_`` prefix).

Scenes are SIM geometry built on ``FakePropagation`` (simplified point-cell RIS) unless a
test says it uses the WP2 physics model; nothing here is a physics result.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np

from sremu.contracts import (
    Action,
    CommandRequest,
    CommandTime,
    LatencyModel,
    Observation,
    ObservationSpec,
    PanelId,
    RisCapabilities,
    StreamCmd,
    StreamSpec,
    TransitionSpec,
    TxMetadata,
)
from sremu.engine.emulator import Emulator, EngineConfig, EngineSink
from sremu.engine.streams import StreamConfig, TrafficSource, stream_configs_from_model
from sremu.testing.fake_propagation import FakeNode, FakePanel, FakePropagation
from sremu.testing.scenes import stream
from sremu.timebase import TimeSpec

DC_BW_HZ = 1.0  # a DC tone or narrow CW: integer-only delay is allowed (section 4.5)


# =========================================================================== scenes


def lab_like_fake(
    rate: str | int | Fraction = "1e6",
    *,
    moving: bool = False,
    tiles: tuple[int, int] = (3, 1),
    far_m: float | None = None,
    horizon: str = "2",
) -> FakePropagation:
    """E-B-like SIM geometry (section 9.3) with a simplified 24 x 5 panel of (1, -1) cells:
    TX ``ap.tx0`` 2 m at -30 deg, ``rx1.rx0`` 1.2 m at 15 deg and ``rx2.rx0`` 1.2 m at 35 deg
    (or ``far_m`` along the normal). ``moving``: rx1 walks at 1.5 m/s along +y."""
    fc = 3.5e9
    rx2 = (far_m, 0.0, 0.0) if far_m is not None else (0.982982453, 0.688291724, 0.0)
    nodes = [
        FakeNode("ap", (1.732050808, -1.0, 0.0), streams=(stream("ap.tx0", "tx", fc, rate),)),
        FakeNode(
            "rx1",
            (1.159110992, 0.310582854, 0.0),
            (0.0, 1.5, 0.0) if moving else (0.0, 0.0, 0.0),
            streams=(stream("rx1.rx0", "rx", fc, rate),),
        ),
        FakeNode("rx2", rx2, streams=(stream("rx2.rx0", "rx", fc, rate),)),
    ]
    pitch = 0.0428
    panel = FakePanel(
        "ris0",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        24,
        5,
        pitch,
        pitch,
        gamma=(1 + 0j, -1 + 0j),
        tiles_u=tiles[0],
        tiles_v=tiles[1],
    )
    return FakePropagation(nodes, [panel], horizon_s=horizon)


def stream_cfgs(
    model: Any,
    *,
    bw: float | None = DC_BW_HZ,
    cfo: Mapping[str, float] | None = None,
    nf_db: float = 0.0,
    extra: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, StreamConfig]:
    """Stream configs with a declared TX bandwidth, optional LO offsets and NF."""
    ov: dict[str, dict[str, Any]] = {}
    for sid, spec in model.streams.items():
        d: dict[str, Any] = {}
        if spec.direction == "tx" and bw is not None:
            d["occupied_bw_hz"] = bw
        if spec.direction == "rx":
            d["noise_figure_db"] = nf_db
        if cfo and sid in cfo:
            d["cfo_hz"] = cfo[sid]
        d.update((extra or {}).get(sid, {}))
        ov[sid] = d
    return stream_configs_from_model(model, ov)


def make_engine(
    model: Any,
    *,
    backend: str = "numpy_ref",
    noise: bool = False,
    bw: float | None = DC_BW_HZ,
    cfo: Mapping[str, float] | None = None,
    traffic: Mapping[str, TrafficSource] | None = None,
    sinks: Sequence[EngineSink] = (),
    clock: Any = None,
    **cfg: Any,
) -> Emulator:
    """An emulator on ``model`` with test-friendly defaults (noise off, DC bandwidth)."""
    cfg.setdefault("render_tiles", "model")
    config = EngineConfig(backend=backend, noise=noise, **cfg)  # type: ignore[arg-type]
    return Emulator(
        model,
        streams=stream_cfgs(model, bw=bw, cfo=cfo),
        config=config,
        traffic=traffic,
        sinks=sinks,
        clock=clock,
    )


class Collect(EngineSink):
    """Collects computed RX samples, TX submissions and final records."""

    def __init__(self) -> None:
        self.rx: dict[str, list[tuple[int, np.ndarray]]] = {}
        self.tx: dict[str, list[tuple[int, np.ndarray]]] = {}
        self.records: list[Any] = []
        self.configs: list[Any] = []
        self.metrics: list[Any] = []
        self.stalls: list[Any] = []

    def on_rx(self, stream: str, k0: int, d: np.ndarray) -> None:
        self.rx.setdefault(stream, []).append((k0, np.array(d, copy=True)))

    def on_tx(self, stream: str, k0: int, x: np.ndarray) -> None:
        self.tx.setdefault(stream, []).append((k0, np.array(x, copy=True)))

    def on_record(self, rec: Any, cfg: Any) -> None:
        self.records.append(rec)
        self.configs.append(cfg)

    def on_metrics(self, window: Any) -> None:
        self.metrics.append(window)

    def on_stall(self, report: Any) -> None:
        self.stalls.append(report)

    def rx_array(self, stream: str, k0: int, k1: int) -> np.ndarray:
        """Assembled complex64 RX samples [k0, k1) (the chunks must cover them)."""
        out = np.zeros(k1 - k0, dtype=np.complex64)
        got = np.zeros(k1 - k0, dtype=bool)
        for a, d in self.rx.get(stream, []):
            lo, hi = max(a, k0), min(a + d.shape[0], k1)
            if lo < hi:
                out[lo - k0 : hi - k0] = d[lo - a : hi - a]
                got[lo - k0 : hi - k0] = True
        assert got.all(), f"{stream}: samples missing in [{k0}, {k1})"
        return out


def captured(cap: Sequence[tuple[int, np.ndarray]], k0: int, k1: int) -> np.ndarray:
    """Assemble complex128 accumulator captures over [k0, k1)."""
    out = np.zeros(k1 - k0, dtype=np.complex128)
    got = np.zeros(k1 - k0, dtype=bool)
    for a, y in cap:
        lo, hi = max(a, k0), min(a + y.shape[0], k1)
        if lo < hi:
            out[lo - k0 : hi - k0] = y[lo - a : hi - a]
            got[lo - k0 : hi - k0] = True
    assert got.all(), f"captured samples missing in [{k0}, {k1})"
    return out


def waveform(n: int, seed: int = 7) -> np.ndarray:
    """A deterministic QPSK-like complex64 test waveform of unit mean power."""
    from sremu import rng

    key = rng.derive_key(seed, "trial", "wp6/waveform")
    w = rng.philox_block(key, 0, n)
    re = 1.0 - 2.0 * (w[:, 0] >> np.uint32(31)).astype(np.float64)
    im = 1.0 - 2.0 * (w[:, 1] >> np.uint32(31)).astype(np.float64)
    return ((re + 1j * im) / math.sqrt(2.0)).astype(np.complex64)


def client_run(
    eng: Emulator,
    tx: str,
    rx: str,
    x: np.ndarray,
    n_rx: int,
    *,
    send_chunk: int | Sequence[int],
    read_chunk: int | Sequence[int],
) -> np.ndarray:
    """Drive the in-process API: timed send of ``x`` from sample 0 in chunks (EOB at the
    end), timed RX start at 0, reads in chunks; returns the first ``n_rx`` RX samples."""
    ctx = eng.configs[tx]
    crx = eng.configs[rx]
    htx = eng.open_stream(StreamSpec(tx, "tx", ctx.fc_hz, ctx.rate))
    hrx = eng.open_stream(StreamSpec(rx, "rx", crx.fc_hz, crx.rate))
    eng.stream_cmd(hrx, StreamCmd("start_cont", stream_now=False, time_spec=TimeSpec(0, 0)))
    sc = list(send_chunk) if not isinstance(send_chunk, int) else None
    rc = list(read_chunk) if not isinstance(read_chunk, int) else None
    sent = 0
    got: list[np.ndarray] = []
    total = 0
    i = 0
    guard = 0
    while total < n_rx:
        guard += 1
        assert guard < 10**6, "client loop does not progress"
        if sent < x.shape[0]:
            c = sc[i % len(sc)] if sc else int(send_chunk)  # type: ignore[arg-type]
            chunk = x[sent : sent + c]
            md = TxMetadata(
                has_time_spec=sent == 0,
                time_spec=TimeSpec(0, 0),
                start_of_burst=sent == 0,
                end_of_burst=sent + chunk.shape[0] >= x.shape[0],
            )
            assert eng.submit_tx(htx, chunk, md) == chunk.shape[0]
            sent += chunk.shape[0]
        r = rc[i % len(rc)] if rc else int(read_chunk)  # type: ignore[arg-type]
        data, _md = eng.read_rx(hrx, min(r, n_rx - total), 0.0)
        if data.size:
            got.append(data)
            total += data.size
        i += 1
    return np.concatenate(got)[:n_rx]


# =========================================================================== controllers


class ScriptedController:
    """An in-process controller returning scripted actions per barrier (tests only).

    ``script(obs, m) -> list[Action]``; ``sleep_s`` sleeps in ``decide`` (wall clock, to
    show it cannot change lockstep output).
    """

    def __init__(
        self,
        name: str,
        period_s: Fraction | str,
        min_latency_s: Fraction | str,
        script: Any,
        *,
        latency: LatencyModel | None = None,
        fields: tuple[str, ...] = ("rx_power_dbm", "ris_state"),
        window_s: str = "1e-3",
        sleep_s: float = 0.0,
    ) -> None:
        self.name = name
        self.period_s = Fraction(period_s)
        self.min_latency_s = Fraction(min_latency_s)
        self.spec = ObservationSpec(fields, Fraction(window_s))
        self.latency = latency or LatencyModel(Fraction(min_latency_s))
        self.script = script
        self.sleep_s = sleep_s
        self.observations: list[Observation] = []
        self.caps: Mapping[PanelId, RisCapabilities] | None = None
        self._m = 0

    def setup(self, caps: Mapping[PanelId, RisCapabilities], rng: np.random.Generator) -> None:
        self.caps = caps

    def decide(self, obs: Observation) -> list[Action]:
        if self.sleep_s:
            import time

            time.sleep(self.sleep_s)
        self.observations.append(obs)
        out = list(self.script(obs, self._m))
        self._m += 1
        return out


def alternate_script(panel: str, n: int, *, states: tuple[int, int] = (0, 1)) -> Any:
    """Alternate all-``states[0]`` / all-``states[1]`` every barrier."""

    def fn(obs: Observation, m: int) -> list[Action]:
        s = states[m % 2]
        return [Action(panel, states=np.full(n, s, dtype=np.int16))]

    return fn


def config_request(
    cmd_id: str,
    panel: str,
    states: np.ndarray,
    at: CommandTime,
    *,
    src: str = "test",
    transition: TransitionSpec | None = None,
    late_policy: str = "apply_on_arrival",
    elements: Any = "all",
) -> CommandRequest:
    """A ``ris.set_config`` request."""
    return CommandRequest(
        cmd_id,
        "ris.set_config",
        src,
        panel,
        at,
        states=np.asarray(states, dtype=np.int16),
        transition=transition,
        late_policy=late_policy,  # type: ignore[arg-type]
        elements=elements,
    )
