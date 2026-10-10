"""Diffuse SoS (DESIGN.md sections 5.11, 6.1, 6.2; D19 API; supports ST-02 and G6).

Deterministic properties of :func:`sremu.physics.diffuse.sos_gain` and of the per-link
diffuse terms. Expected values: the Clarke autocorrelation ``J0`` (scipy, via
``tests/oracles/mobility.py``), keyed derivation by ``sremu.rng``, and K-factor arithmetic
on the snapshot's own path amplitudes. The statistical acceptance (ST-02) is in
``tests/statistical/test_st02_diffuse_sos.py``.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from sremu import rng as srng
from sremu.contracts import PathKind
from sremu.errors import ConfigurationError
from sremu.physics.diffuse import (
    DiffuseMechanism,
    DiffuseTap,
    find_diffuse,
    mechanism_from_scenario,
    reference_power,
    sos_gain,
    sos_parameters,
)
from sremu.physics.geometry import ConstantVelocity, Static
from sremu.physics.ground import GroundMechanism, GroundPlane
from sremu.physics.propagation import NodeModel, ScenePropagation, from_scenario
from sremu.units import C0
from tests.oracles import mobility as mob
from tests.physics._wp3_scenes import link, streams, two_node

pytestmark = pytest.mark.physics

KEY = (0x01234567, 0x89ABCDEF)


def test_d19_signature_shapes_and_dtype():
    t = np.arange(1000) / 1e4
    g = sos_gain(KEY, t, 50.0)
    assert isinstance(g, np.ndarray) and g.dtype == np.complex128 and g.shape == (1000,)
    g2 = sos_gain(KEY, t.reshape(10, 100), 50.0, 48)
    assert g2.shape == (10, 100) and np.array_equal(g2.ravel(), g)
    s = sos_gain(KEY, float(t[123]), 50.0)
    assert isinstance(s, complex)
    assert s == g[123]


def test_d19_pure_in_absolute_time_chunk_and_skip_invariance(rng: np.random.Generator):
    """Each sample depends on its absolute time only: arbitrary chunkings, a reversed order
    and isolated scalar evaluations give bitwise identical values (I2a)."""
    t = np.arange(100_000) / 1e4 + 1234.5
    full = sos_gain(KEY, t, 77.0, 64, tap=2)
    cuts = np.sort(rng.choice(np.arange(1, t.size), 9, replace=False))
    parts = [sos_gain(KEY, seg, 77.0, 64, tap=2) for seg in np.split(t, cuts)]
    assert np.array_equal(np.concatenate(parts), full)
    assert np.array_equal(sos_gain(KEY, t[::-1], 77.0, 64, tap=2)[::-1], full)
    for i in rng.integers(0, t.size, 20):
        assert sos_gain(KEY, float(t[i]), 77.0, 64, tap=2) == full[i]


def test_d19_unit_mean_power_and_parameters():
    for m in (48, 64, 101):
        p = sos_parameters(KEY, 50.0, m)
        assert p.n_sinusoids == m
        assert abs(m * p.amplitude**2 - 1.0) <= 1e-15  # E|g|^2 = M a^2 = 1
        assert np.all(np.abs(p.freqs_hz) <= 50.0)
        assert np.all((p.phases_rad >= 0) & (p.phases_rad < 2 * math.pi))
        assert 0.125 <= p.angle_offset <= 0.375


def test_d19_ensemble_acf_is_clarke_j0():
    """(1/M) sum_m exp(j 2 pi f_m tau) = J0(2 pi f_D tau) + 2 J_M cos(2 pi delta) + ..., i.e.
    J0 to 1e-12 over the ST-02 lags (<= 40 ms at 50 Hz), real part and imaginary part."""
    lags = np.linspace(0.0, 0.040, 401)
    for key in (KEY, (1, 2), (0xFFFFFFFF, 0)):
        for tap in (0, 3):
            p = sos_parameters(key, 50.0, 48, tap=tap)
            acf = np.mean(np.exp(2j * math.pi * p.freqs_hz[None, :] * lags[:, None]), axis=1)
            assert np.max(np.abs(acf - mob.jakes_acf(lags, 50.0))) <= 1e-12


def test_d19_doppler_frequencies_are_distinct():
    """delta in [1/8, 3/8] keeps every cos(alpha_m) distinct: spacing >= the documented
    0.25 (2 pi/M)^2 f_D / 2 (minus rounding)."""
    for key in ((7, 9), (123, 456), KEY):
        for m in (48, 96):
            f = np.sort(sos_parameters(key, 50.0, m).freqs_hz)
            bound = 0.25 * (2 * math.pi / m) ** 2 / 2 * 50.0
            assert np.min(np.diff(f)) >= bound * (1 - 1e-6)


def test_d19_keyed_by_sremu_rng_and_independent_taps():
    p = sos_parameters(KEY, 10.0, 48, tap=1)
    assert np.array_equal(p.phases_rad, 2 * math.pi * srng.uniform(KEY, 0, 48, sub=2))
    assert p.angle_offset == 0.125 + 0.25 * float(srng.uniform(KEY, 0, 1, sub=3)[0])
    t = np.arange(2000) / 1e3
    g0, g1 = sos_gain(KEY, t, 10.0), sos_gain(KEY, t, 10.0, tap=1)
    gk = sos_gain((KEY[0] ^ 1, KEY[1]), t, 10.0)
    assert not np.allclose(g0, g1) and not np.allclose(g0, gk)
    assert np.array_equal(g0, sos_gain(srng.PhiloxKey(*KEY), t, 10.0))


def test_d19_zero_doppler_is_static_and_validation():
    g = sos_gain(KEY, np.linspace(0, 100, 11), 0.0)
    assert np.all(g == g[0]) and abs(g[0]) > 0
    with pytest.raises(ConfigurationError, match="48"):
        sos_gain(KEY, 0.0, 10.0, 47)
    with pytest.raises(ConfigurationError):
        sos_gain(KEY, 0.0, -1.0)
    with pytest.raises(ConfigurationError):
        sos_gain(KEY, np.array([0.0, math.nan]), 1.0)
    with pytest.raises(ValueError):
        sos_gain((1 << 32, 0), 0.0, 1.0)


# --------------------------------------------------------------------------- per-link terms


def _scene_with_ground(fc: float = 3.5e9) -> ScenePropagation:
    return two_node(
        [0.0, 0.0, 2.0],
        [25.0, 3.0, 1.5],
        fc,
        mechanisms=[GroundMechanism(GroundPlane(15, 0.01, "V"))],
    )


def test_diffuse_terms_k_factor_taps_and_delays():
    model = _scene_with_ground()
    snap = model.snapshot(Fraction(0))
    pb = snap.paths[link()]
    taps = (DiffuseTap(0.0, 0.0), DiffuseTap(50e-9, -3.0), DiffuseTap(120e-9, -6.0))
    mech = DiffuseMechanism(k_factor_db=7.0, max_doppler_hz=12.0, taps=taps, seed=99)
    terms = mech.terms(model, snap)[link()]
    p_ref = abs(pb.amp[0]) ** 2 + abs(pb.amp[1]) ** 2  # LOS + GROUND, incoherent
    assert reference_power(pb) == pytest.approx(p_ref, rel=1e-15)
    w = np.array([1.0, 10**-0.3, 10**-0.6])
    w = w / w.sum()
    k = 10**0.7
    assert [t.tap for t in terms] == [0, 1, 2]
    for i, t in enumerate(terms):
        assert t.amp == pytest.approx(math.sqrt(p_ref * w[i] / k), rel=1e-14)
        assert t.delay_s == pb.delay_s[0] + taps[i].delay_s
        assert t.max_doppler_hz == 12.0 and t.n_sinusoids == 48
        assert t.key == srng.derive_key(99, "diffuse", link())
    assert sum(t.amp**2 for t in terms) == pytest.approx(p_ref / k, rel=1e-14)
    tt = np.arange(100) / 1e3
    assert np.array_equal(terms[1].gain(tt), terms[1].amp * sos_gain(terms[1].key, tt, 12.0, tap=1))


def test_diffuse_reference_excludes_ris_and_diffuse_paths():
    from sremu.contracts import PathBundle

    kinds = np.array(
        [
            PathKind.LOS,
            PathKind.GROUND,
            PathKind.WALL,
            PathKind.DIFF,
            PathKind.RIS_GAP,
            PathKind.DIFFUSE,
        ],
        np.int8,
    )
    amp = np.array([1.0, 2j, 0.5, -0.25, 7.0, 9.0], np.complex128)
    pb = PathBundle("x.tx0->y.rx0", Fraction(0), amp, np.arange(6, dtype=np.float64) * 1e-9, kinds)
    assert reference_power(pb) == 1.0 + 4.0 + 0.25 + 0.0625


def test_diffuse_doppler_from_motion():
    """from_motion: f_D = v_rel,max / lambda (3.5 GHz at 1.5 m/s: 17.51 Hz, section 4.4)."""
    fc = 3.5e9
    nodes = [
        NodeModel("a", Static(np.array([0.0, 0.0, 1.5])), streams=streams("a", "tx", fc)),
        NodeModel(
            "b",
            ConstantVelocity(np.array([5.0, 0.0, 1.5]), np.array([1.5, 0.0, 0.0])),
            streams=streams("b", "rx", fc),
        ),
    ]
    model = ScenePropagation(nodes, [], horizon_s=2)
    mech = DiffuseMechanism(k_factor_db=10.0, from_motion=True)
    fd = mech.doppler_hz(model, link())
    assert fd == pytest.approx(1.5 * fc / C0, rel=1e-12)
    assert abs(fd - 17.51) <= 0.005
    assert mech.terms(model, model.snapshot(Fraction(1)))[link()][0].max_doppler_hz == fd


def test_diffuse_adding_a_node_changes_nothing_for_other_links():
    """I4: keyed by (seed, "diffuse", link), so a new node/link leaves other links' diffuse
    processes bitwise unchanged."""
    fc = 2.4e9
    mech = DiffuseMechanism(k_factor_db=3.0, max_doppler_hz=5.0, seed=7)
    m1 = two_node([0, 0, 1.0], [10, 0, 1.0], fc)
    extra = NodeModel("c", Static(np.array([0.0, 5.0, 1.0])), streams=streams("c", "rx", fc))
    nodes = [*m1.nodes.values(), extra]
    m2 = ScenePropagation(nodes, [], horizon_s=1)
    t1 = mech.terms(m1, m1.snapshot(0))
    t2 = mech.terms(m2, m2.snapshot(0))
    assert set(t2) == {link(), link("a", "c")}
    tt = np.arange(500) / 1e3
    assert np.array_equal(t1[link()][0].gain(tt), t2[link()][0].gain(tt))
    assert not np.allclose(
        t2[link()][0].gain(tt) / t2[link()][0].amp,
        t2[link("a", "c")][0].gain(tt) / t2[link("a", "c")][0].amp,
    )


def test_diffuse_mechanism_validation():
    with pytest.raises(ConfigurationError, match="exactly one"):
        DiffuseMechanism(k_factor_db=3.0)
    with pytest.raises(ConfigurationError, match="exactly one"):
        DiffuseMechanism(k_factor_db=3.0, max_doppler_hz=1.0, from_motion=True)
    with pytest.raises(ConfigurationError, match="taps"):
        DiffuseMechanism(
            k_factor_db=3.0, max_doppler_hz=1.0, taps=tuple(DiffuseTap() for _ in range(5))
        )
    with pytest.raises(ConfigurationError):
        DiffuseTap(-1e-9, 0.0)
    with pytest.raises(ConfigurationError):
        DiffuseMechanism(k_factor_db=math.inf, max_doppler_hz=1.0)


def test_diffuse_from_scenario_is_attached_and_found(tmp_path):
    doc = {
        "schema_version": "1.0",
        "name": "diffuse",
        "seed": 4242,
        "nodes": [
            {
                "id": "a",
                "position_m": [0.0, 0.0, 1.5],
                "antenna": {"pattern": "isotropic"},
                "streams": [
                    {
                        "name": "tx0",
                        "direction": "tx",
                        "fc_hz": 1e9,
                        "rate_sps": "1e6",
                        "occupied_bw_hz": 1e5,
                    }
                ],
            },
            {
                "id": "b",
                "position_m": [6.0, 1.0, 1.5],
                "antenna": {"pattern": "isotropic"},
                "streams": [{"name": "rx0", "direction": "rx", "fc_hz": 1e9, "rate_sps": "1e6"}],
            },
        ],
        "environment": {
            "diffuse": {
                "k_factor_db": 6.0,
                "max_doppler_hz": 3.0,
                "n_sinusoids": 64,
                "taps": [
                    {"delay_s": 0.0, "rel_power_db": 0.0},
                    {"delay_s": 1e-7, "rel_power_db": -4.0},
                ],
            }
        },
        "run": {
            "mode": "lockstep",
            "coupling": "barrier",
            "duration_s": "1",
            "late_policy": "apply_on_arrival",
        },
    }
    model = from_scenario(doc, base_dir=tmp_path)
    mech = find_diffuse(model)
    assert isinstance(mech, DiffuseMechanism)
    assert mech.seed == 4242 and mech.n_sinusoids == 64 and len(mech.taps) == 2
    # it is not a deterministic path: the snapshot holds the LOS only
    assert model.snapshot(0).paths[link()].kind.tolist() == [PathKind.LOS]
    assert find_diffuse(two_node([0, 0, 1.0], [1, 0, 1.0], 1e9)) is None
    assert mechanism_from_scenario(None, doc) is None
    assert mechanism_from_scenario({"k_factor_db": 1.0, "from_motion": True}, {"seed": 1}).taps == (
        DiffuseTap(),
    )  # type: ignore[union-attr]


@pytest.mark.parametrize("rate", ["23.04e6", "5880", "1e6", Fraction(30_720_000, 7)])
def test_d19_sample_times_are_correctly_rounded_exact_rationals(rate, rng: np.random.Generator):  # type: ignore[no-untyped-def]
    from sremu.physics.diffuse import sample_times_s
    from sremu.timebase import parse_exact

    fr = parse_exact(rate)
    k0 = int(rng.integers(0, 10**9))
    t = sample_times_s(k0, 257, rate)
    assert t.dtype == np.float64 and t.shape == (257,)
    for i in (0, 1, 100, 256):
        assert t[i] == float(Fraction(k0 + i) / fr)  # Fraction -> float is correctly rounded


def test_d19_sample_api_is_pure_in_the_absolute_index(rng: np.random.Generator):
    """Chunked renders over absolute indices give bitwise identical diffuse gains (I2a)."""
    from sremu.physics.diffuse import sample_times_s, sos_gain_samples

    key = srng.derive_key(5, "diffuse", "ap.tx0->mu.rx0")
    k_start, n = 123_456_789, 40_000
    full = sos_gain_samples(key, k_start, n, "23.04e6", 350.242, 48, tap=1)
    cuts = np.sort(rng.choice(np.arange(1, n), 7, replace=False))
    bounds = [0, *cuts.tolist(), n]
    parts = [
        sos_gain_samples(key, k_start + a, b - a, "23.04e6", 350.242, 48, tap=1)
        for a, b in itertools.pairwise(bounds)
    ]
    assert np.array_equal(np.concatenate(parts), full)
    ref = sos_gain(key, sample_times_s(k_start, n, "23.04e6"), 350.242, 48, tap=1)
    assert np.array_equal(full, ref)
    with pytest.raises(ConfigurationError):
        sos_gain_samples(key, -1, 10, "1e6", 1.0)
    with pytest.raises(ConfigurationError):
        sos_gain_samples(key, 1 << 60, 10, "1e6", 1.0)


def test_d19_term_gain_samples():
    mech = DiffuseMechanism(k_factor_db=3.0, max_doppler_hz=20.0, seed=11)
    model = two_node([0, 0, 1.0], [10, 0, 1.0], 1e9)
    term = mech.terms(model, model.snapshot(0))[link()][0]
    got = term.gain_samples(1000, 64, "1e6")
    want = term.amp * sos_gain(term.key, (1000 + np.arange(64)) / 1e6, 20.0, 48)
    assert np.array_equal(got, want)
