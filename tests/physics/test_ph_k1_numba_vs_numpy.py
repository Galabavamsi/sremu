"""WP2 acceptance: the numba K1 kernel matches the numpy float64 reference to <= 1e-12
relative (section 14.1, WP2 row; backends of section 7.1)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from sremu.contracts import StreamSpec
from sremu.physics.antennas import CosPower
from sremu.physics.geometry import PanelGrid, Pose, Static
from sremu.physics.propagation import NodeModel, Numerics, PanelModel, ScenePropagation
from sremu.physics.response import Provenance, ResponseTable, builtin
from sremu.physics.ris_kernel import accumulate_cells, k1_terms
from sremu.units import C0

pytestmark = pytest.mark.physics


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    scale = np.max(np.abs(b), axis=-1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    return float(np.max(np.abs(a - b) / scale))


@pytest.mark.filterwarnings("ignore:obliquity='kirchhoff':sremu.errors.ValidityWarning")  # D2
@pytest.mark.parametrize("obliquity", ["sqrt", "kirchhoff", "none"])
@pytest.mark.parametrize("huygens", [True, False])
def test_k1_terms_numba_matches_numpy(obliquity: str, huygens: bool, rng: np.random.Generator):
    q, r = 3000, 5
    pts = np.column_stack([rng.uniform(-0.5, 0.5, q), rng.uniform(-0.4, 0.4, q), np.zeros(q)])
    txs = np.column_stack([rng.uniform(-3, 3, r), rng.uniform(-3, 3, r), rng.uniform(-1, 50, r)])
    rxs = np.column_stack([rng.uniform(-3, 3, r), rng.uniform(-3, 3, r), rng.uniform(0.5, 2000, r)])
    fcs = rng.uniform(1e9, 30e9, r)
    ref = rng.uniform(0, 100, (r, q))
    gt, gr = rng.uniform(0, 20, (r, q)), rng.uniform(0, 20, (r, q))
    kw = {
        "ref_len_m": ref,
        "gain_tx": gt,
        "gain_rx": gr,
        "obliquity": obliquity,
        "huygens_j": huygens,
    }
    a = k1_terms(
        txs, rxs, pts, np.full(q, 1e-4), np.array([0, 0, 1.0]), fcs, backend="numpy_ref", **kw
    )  # type: ignore[arg-type]
    b = k1_terms(
        txs, rxs, pts, np.full(q, 1e-4), np.array([0, 0, 1.0]), fcs, backend="numba_cpu", **kw
    )  # type: ignore[arg-type]
    assert np.array_equal(a == 0, b == 0)  # identical visibility decisions
    assert _rel(b, a) <= 1e-12
    parent = np.repeat(np.arange(q // 4), 4)
    gam = rng.normal(size=(r, 3, q)) + 1j * rng.normal(size=(r, 3, q))
    for g in (None, gam):
        ca = accumulate_cells(a, parent, q // 4, g, backend="numpy_ref")
        cb = accumulate_cells(a, parent, q // 4, g, backend="numba_cpu")
        assert _rel(cb, ca) <= 1e-12


def _angle_table() -> ResponseTable:
    th = np.arange(-80.0, 81.0, 10.0)
    ti, ts = np.meshgrid(th, th, indexing="ij")
    g0 = 0.9 * np.exp(1j * np.radians(150 + 0.2 * (ti + ts)))
    g1 = 0.8 * np.exp(1j * np.radians(-20 - 0.1 * (ti + ts)))
    prov = Provenance(
        "fullwave",
        "smooth synthetic angle table",
        "test",
        0.0,
        "co",
        False,
        0.0,  # type: ignore[arg-type]
        0.0,
        None,
        "test",
        {"date": "2026-10-10"},
    )
    return ResponseTable(
        ["s0", "s1"], [3.5e9], th, np.stack([g0, g1])[:, None], prov, validity_bw_hz=1e8
    )


@pytest.mark.parametrize("table", ["separable", "angle"])
def test_snapshot_numba_matches_numpy(table: str):
    fc = 3.5e9
    lam = C0 / fc
    tab = builtin("ideal_bbit(2)") if table == "separable" else _angle_table()
    streams = {
        "ap": (StreamSpec("ap.tx0", "tx", fc, "1e6"),),  # type: ignore[arg-type]
        "u1": (StreamSpec("u1.rx0", "rx", fc, "1e6"),),  # type: ignore[arg-type]
        "u2": (StreamSpec("u2.rx0", "rx", fc, "1e6"),),  # type: ignore[arg-type]
    }
    pos = {"ap": [-1.0, 0.2, 2.0], "u1": [1.5, -0.3, 1.2], "u2": [0.4, 0.9, 3.0]}

    def build(backend: str) -> ScenePropagation:
        nodes = [
            NodeModel(
                k,
                Static(np.array(pos[k])),
                CosPower(3.0, -np.array(pos[k]) / np.linalg.norm(pos[k])),
                streams[k],
            )
            for k in pos
        ]
        panels = [
            PanelModel(
                "p_point",
                PanelGrid(Pose([0, 0, 0], [0, 0, 1], [1, 0, 0]), 12, 10, lam / 2, lam / 2),
                tab,
                render_tiles=(3, 2),
            ),
            PanelModel(
                "p_patch",
                PanelGrid(Pose([0, 2.5, 0.5], [0, -1, 0], [1, 0, 0]), 4, 3, 0.9 * lam, 0.8 * lam),
                tab,
            ),
        ]
        # finer starting sub-cells: the oblique po_patch panel does not converge within two
        # halvings from lambda/5 (RefinementNotConverged, as specified)
        return ScenePropagation(
            nodes,
            panels,
            backend=backend,  # type: ignore[arg-type]
            numerics=Numerics(subcell_max_wavelengths=0.1),
        )

    sa = build("numpy_ref").snapshot(Fraction(0))
    sb = build("numba_cpu").snapshot(Fraction(0))
    for pid in ("p_point", "p_patch"):
        ra, rb = sa.ris[pid], sb.ris[pid]
        assert ra.separable == rb.separable == (table == "separable")
        assert np.array_equal(ra.tile_delay_s, rb.tile_delay_s)
        xa = ra.C if ra.separable else ra.B
        xb = rb.C if rb.separable else rb.B
        assert xa is not None and xb is not None
        assert _rel(xb.reshape(xb.shape[0], -1), xa.reshape(xa.shape[0], -1)) <= 1e-12


def test_tile_gemv_numba_matches_numpy_and_parallel_equals_serial(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
):
    from sremu.physics import ris_kernel

    r, s, n, k = 64, 3, 900, 7
    c = rng.normal(size=(r, n)) + 1j * rng.normal(size=(r, n))
    b = rng.normal(size=(r, s, n)) + 1j * rng.normal(size=(r, s, n))
    g = rng.normal(size=n) + 1j * rng.normal(size=n)
    state = rng.integers(-1, s, n)
    tile = rng.integers(0, k, n)
    for kw in ({"c": c, "g": g}, {"b": b, "state": state}):
        a = ris_kernel.tile_sums(tile, k, backend="numpy_ref", **kw)
        z = ris_kernel.tile_sums(tile, k, backend="numba_cpu", **kw)
        assert _rel(z, a) <= 1e-12
        # independent closed form: explicit loops
        ref = np.zeros((r, k), complex)
        for i in range(n):
            if "c" in kw:
                ref[:, tile[i]] += c[:, i] * g[i]
            elif state[i] >= 0:
                ref[:, tile[i]] += b[:, state[i], i]
        assert _rel(a, ref) <= 1e-12
    # the row-parallel K1 drivers give bitwise the same result as the serial ones
    pts = np.column_stack([rng.uniform(-0.5, 0.5, 600), rng.uniform(-0.5, 0.5, 600), np.zeros(600)])
    txs = np.column_stack([rng.uniform(-2, 2, 8), rng.uniform(-2, 2, 8), rng.uniform(1, 3, 8)])
    rxs = np.column_stack([rng.uniform(-2, 2, 8), rng.uniform(-2, 2, 8), rng.uniform(1, 3, 8)])
    nrm = np.array([0.0, 0.0, 1.0])
    serial = k1_terms(txs, rxs, pts, 1e-4, nrm, 3.5e9, backend="numba_cpu")
    acc_s = accumulate_cells(serial, np.arange(600) // 3, 200, backend="numba_cpu")
    monkeypatch.setattr(ris_kernel, "PARALLEL_MIN_WORK", 0)
    assert ris_kernel._use_parallel(8, 600)
    par = k1_terms(txs, rxs, pts, 1e-4, nrm, 3.5e9, backend="numba_cpu")
    acc_p = accumulate_cells(par, np.arange(600) // 3, 200, backend="numba_cpu")
    assert np.array_equal(serial, par)
    assert np.array_equal(acc_s, acc_p)
