"""Transition math for non-instant RIS reconfiguration (DESIGN.md section 4.7; RT-12).

Exact by linearity of ``combine`` in Gamma, at RX sample n of receiver r:

    h(n) = h_old(n) + sum_c g_c(n - k_r) * dh_c(n),
    dh_c = combine(cfg_old with the class-c elements set to new) - combine(cfg_old)

* ``k_r = to_ticks(t_sw + tau_RIS->r, f_r)`` is the RX sample where ``g_c(0)`` applies (I3),
  per tile when tiled (pass one ``k_start`` per path).
* Profiles (argument in samples, ``x = n - k_start``; ``g = 0`` for ``x < 0``):
  ``step`` -> 1; ``linear`` -> ``min(x/T, 1)``; ``first_order`` -> ``1 - exp(-x/tau)``
  (``-expm1``). A non-positive duration is a step.
* ``dh_c`` is evaluated at both snapshot ends and interpolated complex-linearly like any
  residual (``TransitionTerm.dA_a``/``dA_b``), so the result stays correct under mobility.
* Transition classes: ``single`` (every changed element) or ``by_edge`` (one class per
  distinct ``(from, to)`` state pair, ordered by the pair).
* **Re-command mid-transition** (start from the instantaneous per-element Gamma
  ``Gamma_old + g(t)(Gamma_new - Gamma_old)``): :func:`freeze_terms` turns the running terms
  into step terms scaled by their value ``g_c(k_now - k_c)`` at the re-command sample, and
  :func:`recommand_deltas` gives the new classes' deltas relative to that instantaneous
  state, ``D_c'(new2) - sum_c g_c * D_(c & c')(new1)`` with ``D_S(s) = combine(base with S
  set to s) - combine(base)``. The frozen value is taken per RX at the RX re-command sample
  (continuity of h at that sample) [ENG].

The complex-linear Gamma path is [ENG] (section 4.7); every transition result is labelled
ENG until a GS RIS 008 reconfiguration-time measurement exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray

from sremu.contracts import (
    LinkId,
    PanelConfig,
    PanelId,
    PropagationModel,
    Snapshot,
    TransitionModel,
    TransitionSpec,
    TransitionTerm,
)
from sremu.errors import ConfigurationError
from sremu.timebase import parse_exact

__all__ = [
    "PROFILE_CODES",
    "ClassMask",
    "build_terms",
    "class_config",
    "class_deltas",
    "freeze_terms",
    "profile_code",
    "profile_values",
    "recommand_classes",
    "recommand_deltas",
    "terms_from_deltas",
    "transition_classes",
]

PROFILE_CODES: Final[dict[str, int]] = {"step": 0, "linear": 1, "first_order": 2}
"""Integer codes of the profiles used by the numba kernel."""


def profile_code(profile: str, duration_samples: float) -> int:
    """Kernel code of a profile; a non-positive duration is a step (code 0)."""
    if profile not in PROFILE_CODES:
        raise ConfigurationError(f"unknown transition profile {profile!r}")
    if profile != "step" and not duration_samples > 0.0:
        return 0
    return PROFILE_CODES[profile]


def profile_values(
    profile: str, x_samples: float | NDArray[np.floating], duration_samples: float
) -> NDArray[np.float64]:
    """``g(x)`` for sample offsets ``x = n - k_start`` (0 for x < 0)."""
    x = np.asarray(x_samples, dtype=np.float64)
    code = profile_code(profile, duration_samples)
    on = x >= 0.0
    if code == 0:
        return np.where(on, 1.0, 0.0)
    if code == 1:
        return np.where(on, np.minimum(x / duration_samples, 1.0), 0.0)
    return np.where(on, -np.expm1(-np.maximum(x, 0.0) / duration_samples), 0.0)


@dataclass(frozen=True, eq=False)
class ClassMask:
    """One transition class: a label, the elements in it, and its profile."""

    label: str
    mask: NDArray[np.bool_]
    profile: TransitionModel = "step"
    duration_s: Fraction = Fraction(0)


def transition_classes(
    state_from: NDArray[np.int16],
    state_to: NDArray[np.int16],
    classes: str = "single",
) -> list[tuple[str, NDArray[np.bool_]]]:
    """Masks of the changed elements per class: ``single`` or ``by_edge``.

    ``by_edge`` gives one class per distinct ``(from, to)`` pair, ordered by the pair and
    labelled ``"<from>-><to>"``. Unchanged elements belong to no class.
    """
    a = np.asarray(state_from)
    b = np.asarray(state_to)
    if a.shape != b.shape:
        raise ConfigurationError(f"state shapes differ: {a.shape} vs {b.shape}")
    changed = a != b
    if not changed.any():
        return []
    if classes == "single":
        return [("all", changed)]
    if classes != "by_edge":
        raise ConfigurationError(f"unknown transition classes {classes!r}")
    pairs = sorted({(int(x), int(y)) for x, y in zip(a[changed], b[changed], strict=True)})
    return [(f"{x}->{y}", changed & (a == x) & (b == y)) for x, y in pairs]


def class_config(
    base: PanelConfig, target_state: NDArray[np.int16], mask: NDArray[np.bool_]
) -> PanelConfig:
    """``base`` with the masked elements set to ``target_state``."""
    st = base.state.copy()
    st[mask] = np.asarray(target_state, dtype=np.int16)[mask]
    return PanelConfig(base.panel, st, base.enabled)


def class_deltas(
    model: PropagationModel,
    snap: Snapshot,
    cfg_base: Mapping[PanelId, PanelConfig],
    panel: PanelId,
    target_state: NDArray[np.int16],
    masks: Sequence[NDArray[np.bool_]],
) -> list[dict[LinkId, NDArray[np.complex128]]]:
    """``D_S = combine(base with S set to target) - combine(base)`` per mask, per link.

    The path sets must not depend on the configuration (the delays are checked).
    """
    base = model.combine(snap, cfg_base)
    out: list[dict[LinkId, NDArray[np.complex128]]] = []
    for mask in masks:
        cfg = dict(cfg_base)
        cfg[panel] = class_config(cfg_base[panel], target_state, mask)
        paths = model.combine(snap, cfg)
        d: dict[LinkId, NDArray[np.complex128]] = {}
        for lk, lp in paths.items():
            b = base[lk]
            if lp.n_paths != b.n_paths or not np.array_equal(lp.delay_s, b.delay_s):
                raise ConfigurationError(
                    f"link {lk}: combine() changed the path set with the configuration"
                )
            d[lk] = lp.amp - b.amp
        out.append(d)
    return out


def terms_from_deltas(
    d_a: NDArray[np.complex128],
    d_b: NDArray[np.complex128],
    profile: TransitionModel,
    duration_samples: float,
    k_start: int | NDArray[np.integer],
) -> list[TransitionTerm]:
    """Transition terms for one class on one link.

    ``k_start`` is one RX sample for every path, or one per path (tiles see the switch at
    their own sample); paths sharing a ``k_start`` share a term, ordered by ``k_start``.
    """
    d_a = np.asarray(d_a, dtype=np.complex128)
    d_b = np.asarray(d_b, dtype=np.complex128)
    if d_a.shape != d_b.shape or d_a.ndim != 1:
        raise ConfigurationError("delta endpoints must be equal-length 1-D arrays")
    if np.ndim(k_start) == 0:
        return [TransitionTerm(int(k_start), profile, float(duration_samples), d_a, d_b)]
    ks = np.asarray(k_start, dtype=np.int64)
    if ks.shape != d_a.shape:
        raise ConfigurationError("per-path k_start must have one entry per path")
    terms = []
    for k in np.unique(ks):
        sel = ks == k
        ta = np.where(sel, d_a, 0.0).astype(np.complex128)
        tb = np.where(sel, d_b, 0.0).astype(np.complex128)
        terms.append(TransitionTerm(int(k), profile, float(duration_samples), ta, tb))
    return terms


def build_terms(
    model: PropagationModel,
    snap_a: Snapshot,
    snap_b: Snapshot,
    cfg_base: Mapping[PanelId, PanelConfig],
    panel: PanelId,
    target_state: NDArray[np.int16],
    spec: TransitionSpec,
    f_s: Fraction | int,
    k_start: Mapping[LinkId, int | NDArray[np.integer]],
    *,
    default_duration_s: Fraction | None = None,
    class_profiles: Mapping[str, tuple[TransitionModel, Fraction]] | None = None,
) -> dict[LinkId, list[TransitionTerm]]:
    """Transition terms of one command, per link, for the interval [snap_a.t, snap_b.t].

    ``cfg_base``: configuration before the command (all panels). ``spec``: the command's
    transition (``duration_s`` None means ``default_duration_s``, the panel's
    ``reconfiguration_time_s``). ``k_start``: per link, the RX sample (or per-path samples)
    where the switch reaches that link's receiver. ``class_profiles`` may override the
    profile and duration per class label.
    """
    fs = parse_exact(f_s, name="f_s")
    dur = spec.duration_s if spec.duration_s is not None else default_duration_s
    if spec.model != "step" and dur is None:
        raise ConfigurationError("a non-step transition needs a duration (panel default missing)")
    dur = Fraction(0) if dur is None else parse_exact(dur, name="duration_s")
    classes = transition_classes(cfg_base[panel].state, target_state, spec.classes)
    if not classes:
        return {lk: [] for lk in k_start}
    masks = [m for _, m in classes]
    da = class_deltas(model, snap_a, cfg_base, panel, target_state, masks)
    db = class_deltas(model, snap_b, cfg_base, panel, target_state, masks)
    out: dict[LinkId, list[TransitionTerm]] = {lk: [] for lk in k_start}
    for ci, (label, _mask) in enumerate(classes):
        prof: TransitionModel = spec.model
        d = dur
        if class_profiles is not None and label in class_profiles:
            prof, d = class_profiles[label]
            d = parse_exact(d, name="duration_s")
        for lk, ks in k_start.items():
            out[lk].extend(terms_from_deltas(da[ci][lk], db[ci][lk], prof, float(d * fs), ks))
    return out


def freeze_terms(terms: Sequence[TransitionTerm], k_now: int) -> list[TransitionTerm]:
    """Running terms frozen at RX sample ``k_now`` (re-command mid-transition).

    Each term becomes a step term at its own ``k_start`` scaled by ``g(k_now - k_start)``;
    terms not yet started (g = 0) are dropped. Valid for segments starting at ``k_now``.
    """
    out = []
    for t in terms:
        g = float(profile_values(t.profile, float(int(k_now) - t.k_start), t.duration_samples))
        if g == 0.0:
            continue
        out.append(TransitionTerm(t.k_start, "step", 0.0, g * t.dA_a, g * t.dA_b))
    return out


def recommand_classes(
    state_base: NDArray[np.int16],
    state_new1: NDArray[np.int16],
    frozen: Sequence[tuple[NDArray[np.bool_], float]],
    state_new2: NDArray[np.int16],
    classes: str = "single",
) -> list[tuple[str, NDArray[np.bool_]]]:
    """Classes of a command issued mid-transition.

    ``frozen``: ``(mask_c, g_c)`` of the interrupted command's classes at the re-command
    sample (as for :func:`recommand_deltas`). Elements of those classes were last commanded
    to ``state_new1``; the rest hold ``state_base``. An element moves if it is still in
    transition (``g_c < 1``) or its last commanded state differs from ``state_new2``.
    ``by_edge`` pairs use (last commanded state, ``state_new2``).
    """
    base = np.asarray(state_base)
    new1 = np.asarray(state_new1)
    new2 = np.asarray(state_new2)
    commanded = np.zeros(base.shape, dtype=bool)
    moving_now = np.zeros(base.shape, dtype=bool)
    for mask_c, g_c in frozen:
        m = np.asarray(mask_c, dtype=bool)
        commanded |= m
        if g_c < 1.0:
            moving_now |= m
    last = np.where(commanded, new1, base)
    moving = moving_now | (last != new2)
    if not moving.any():
        return []
    if classes == "single":
        return [("all", moving)]
    if classes != "by_edge":
        raise ConfigurationError(f"unknown transition classes {classes!r}")
    pairs = sorted({(int(x), int(y)) for x, y in zip(last[moving], new2[moving], strict=True)})
    return [(f"{x}->{y}", moving & (last == x) & (new2 == y)) for x, y in pairs]


def recommand_deltas(
    model: PropagationModel,
    snap: Snapshot,
    cfg_base: Mapping[PanelId, PanelConfig],
    panel: PanelId,
    state_new1: NDArray[np.int16],
    frozen: Sequence[tuple[NDArray[np.bool_], float]],
    state_new2: NDArray[np.int16],
    masks_new2: Sequence[NDArray[np.bool_]],
) -> list[dict[LinkId, NDArray[np.complex128]]]:
    """Deltas of a new command relative to the instantaneous mid-transition Gamma.

    ``cfg_base``: configuration before the first (interrupted) transition toward
    ``state_new1``. ``frozen``: ``(mask_c, g_c)`` of each interrupted class at the
    re-command sample. Returns, per new class ``c'``,
    ``D_c'(new2) - sum_c g_c * D_(c & c')(new1)``.
    """
    d2 = class_deltas(model, snap, cfg_base, panel, state_new2, masks_new2)
    out = []
    for ci, m2 in enumerate(masks_new2):
        acc = {lk: v.copy() for lk, v in d2[ci].items()}
        for mask_c, g_c in frozen:
            inter = np.asarray(mask_c) & np.asarray(m2)
            if g_c == 0.0 or not inter.any():
                continue
            d1 = class_deltas(model, snap, cfg_base, panel, state_new1, [inter])[0]
            for lk in acc:
                acc[lk] = acc[lk] - g_c * d1[lk]
        out.append(acc)
    return out
