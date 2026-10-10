"""Remote controllers: run any ``Controller`` against a served emulator (DESIGN.md 4.9, 8.7).

``run_remote(ctrl, "tcp://127.0.0.1:5600")`` wraps the same controller class that runs
in-process for an external process. The barrier protocol over the control socket
(section 8.5) is

    engine -> ctrl.barrier {t_b, k_b, m, observation}
    controller -> zero or more ris.set_config / ris.set_enabled (src = controller name)
    controller -> ctrl.ack {t_b}

**Bitwise parity with the in-process controller (RT-08).** The runner reproduces what the
engine does for an in-process controller, so a lockstep run gives bitwise-identical RX:

* registration returns the run's master seed; ``setup`` receives the same keyed generator
  ``rng.generator(seed, "ctrl", name)`` (section 6.1);
* each action ``i`` at barrier ``m`` switches at ``a.at_s`` or
  ``t_b + max(latency, L_min)`` with the latency drawn by
  :func:`sremu.engine.barriers.draw_latency_s` (purpose ``ctrl.latency/<name>``,
  counter ``m``, lane ``i``), and becomes the same ``CommandRequest``
  (:func:`sremu.engine.barriers.action_request`: id ``<name>:<m>:<i>``, the run's
  ``late_policy``);
* the observation travels exactly (float64 through JSON ``repr``, arrays as base64 of
  their little-endian bytes); phase actions use the exact ``phases_rad`` form.

An action earlier than ``t_b + L_min`` is still sent; the engine rejects it with
``LOOKAHEAD_VIOLATION`` (RT-17) and the runner raises
:class:`~sremu.errors.LookaheadViolation` after acking and unregistering, as the in-process
engine would abort. Any other rejected action raises ``ConfigurationError`` the same way.

No wall clock is read here (waits use socket polls and events; I7 allows the clock only in
``transport/``).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from sremu import rng
from sremu.contracts import DEFAULT_CONTROL_ENDPOINT, CommandReceipt, Controller, LatencyModel
from sremu.engine.barriers import action_request, draw_latency_s
from sremu.errors import ConfigurationError, ErrorCode, LookaheadViolation, SremuError
from sremu.timebase import fraction_to_str, parse_exact
from sremu.transport.client import ClientSession
from sremu.transport.messages import (
    caps_from_json,
    command_to_request,
    observation_from_json,
    receipt_from_reply,
)

__all__ = ["RemoteControllerRunner", "RemoteRunStats", "register_message", "run_remote"]


@dataclass
class RemoteRunStats:
    """What a remote controller did: barriers served, commands sent and their receipts."""

    barriers: int = 0
    commands: int = 0
    receipts: list[CommandReceipt] = field(default_factory=list)
    t_b: list[Fraction] = field(default_factory=list)


def register_message(ctrl: Controller) -> dict[str, Any]:
    """The ``ctrl.register`` fields of a controller (its declared attributes)."""
    lat = ctrl.latency if isinstance(ctrl.latency, LatencyModel) else LatencyModel()
    spec = ctrl.spec
    return {
        "name": str(ctrl.name),
        "period_s": fraction_to_str(parse_exact(ctrl.period_s)),
        "min_latency_s": fraction_to_str(parse_exact(ctrl.min_latency_s)),
        "latency": {"fixed_s": fraction_to_str(lat.fixed_s), "jitter": lat.jitter},
        "observation": {
            "fields": list(spec.fields),
            "window_s": fraction_to_str(spec.window_s),
            "position_delay_s": fraction_to_str(spec.position_delay_s),
            "position_noise_m": float(spec.position_noise_m),
        },
    }


class RemoteControllerRunner:
    """Runs one controller over its own control connection (see the module docstring).

    ``endpoint``: the served emulator's control endpoint. ``timeout_s``: reply timeout of
    each request. ``session``: an existing :class:`ClientSession` (else one is opened).
    """

    def __init__(
        self,
        ctrl: Controller,
        endpoint: str = DEFAULT_CONTROL_ENDPOINT,
        *,
        timeout_s: float = 30.0,
        session: ClientSession | None = None,
    ) -> None:
        self.ctrl = ctrl
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self._session = session
        self._own_session = session is None
        self.name = str(ctrl.name)
        self.stats = RemoteRunStats()
        self.registered: dict[str, Any] | None = None
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.seed = 0
        self.late_policy = "apply_on_arrival"
        self.l_min = parse_exact(ctrl.min_latency_s)
        self.latency = ctrl.latency if isinstance(ctrl.latency, LatencyModel) else LatencyModel()

    @property
    def session(self) -> ClientSession:
        """The control connection (opened on first use)."""
        if self._session is None:
            self._session = ClientSession(self.endpoint, name=self.name, timeout_s=self.timeout_s)
        return self._session

    # ------------------------------------------------------------------ protocol
    def register(self) -> dict[str, Any]:
        """``ctrl.register``; then ``ctrl.setup(caps, rng.generator(seed, "ctrl", name))``."""
        reply = self.session.call("ctrl.register", **register_message(self.ctrl))
        self.registered = reply
        self.seed = int(reply["seed"])
        self.late_policy = str(reply.get("late_policy", "apply_on_arrival"))
        caps = caps_from_json(reply.get("caps") or {})
        self.ctrl.setup(caps, rng.generator(self.seed, "ctrl", self.name))
        return reply

    def handle_barrier(self, ev: Mapping[str, Any]) -> list[CommandReceipt]:
        """Decide at one ``ctrl.barrier``, send the commands, then ``ctrl.ack``."""
        obs = observation_from_json(ev["observation"])
        t_b = obs.t_b
        m = int(ev["m"])
        actions = list(self.ctrl.decide(obs))
        receipts: list[CommandReceipt] = []
        first_error: CommandReceipt | None = None
        for i, a in enumerate(actions):
            if a.at_s is not None:
                at = Fraction(a.at_s)
            else:
                at = t_b + max(draw_latency_s(self.seed, self.name, self.latency, m, i), self.l_min)
            req = action_request(self.name, m, i, a, at, self.late_policy)
            r = receipt_from_reply(self.session.request(command_to_request(req)))
            receipts.append(r)
            if not r.ok and first_error is None:
                first_error = r
        self.session.call("ctrl.ack", controller=self.name, t_b=fraction_to_str(t_b))
        self.stats.barriers += 1
        self.stats.commands += len(actions)
        self.stats.receipts.extend(receipts)
        self.stats.t_b.append(t_b)
        if first_error is not None:
            self._unregister_quietly()
            if first_error.error == ErrorCode.LOOKAHEAD_VIOLATION:
                raise LookaheadViolation(f"{self.name}: {first_error.msg}")
            raise ConfigurationError(
                f"{self.name}: action rejected: {first_error.error.name}: {first_error.msg}"
            )
        return receipts

    def serve(
        self, *, max_barriers: int | None = None, idle_timeout_s: float | None = None
    ) -> RemoteRunStats:
        """Serve barriers until :meth:`stop`, ``max_barriers`` barriers, or no barrier for
        ``idle_timeout_s`` wall seconds (None: wait forever).

        The idle timeout counts from registration and is a backstop, not an end-of-run
        signal: a served engine can be silent for many seconds before a barrier (it computes
        every receiver up to ``t_b + L_min`` first, and on a cold cache JIT-compiles its numba
        kernels, which takes seconds before the first barrier and longer on a loaded CI
        runner). When the run's barrier count is known (``t_b = m*T_c < duration``), pass
        ``max_barriers``."""
        if self.registered is None:
            self.register()
        idle = 0.0
        while not self._stop.is_set():
            if max_barriers is not None and self.stats.barriers >= max_barriers:
                break
            ev = self.session.next_event("ctrl.barrier", timeout_s=0.1)
            if ev is None:
                idle += 0.1
                if idle_timeout_s is not None and idle >= idle_timeout_s:
                    break
                continue
            idle = 0.0
            if ev.get("controller") != self.name:
                continue
            self.handle_barrier(ev)
        return self.stats

    def _unregister_quietly(self) -> None:
        # short timeout: the server may already have stopped (e.g. --exit-when-done)
        with contextlib.suppress(SremuError):
            self.session.call("ctrl.unregister", controller=self.name, timeout_s=2.0)

    def unregister(self) -> None:
        """``ctrl.unregister`` (the engine stops waiting for this controller)."""
        self._unregister_quietly()

    # ------------------------------------------------------------------ thread
    def _main(self) -> None:
        try:
            self.serve()
        except BaseException as exc:  # kept for the owner; the thread must not die silently
            self.error = exc

    def start(self) -> None:
        """Register now (errors raise here), then serve barriers in a background thread."""
        self.register()
        self._thread = threading.Thread(
            target=self._main, name=f"sremu-ctrl-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        """Stop serving and close the connection (when the runner opened it)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout_s)
        if self._own_session and self._session is not None:
            self._session.close()


def run_remote(
    ctrl: Controller,
    endpoint: str = DEFAULT_CONTROL_ENDPOINT,
    *,
    max_barriers: int | None = None,
    idle_timeout_s: float | None = None,
    timeout_s: float = 30.0,
    unregister: bool = True,
) -> RemoteRunStats:
    """Register ``ctrl`` with the served emulator at ``endpoint`` and serve its barriers in
    this thread until ``max_barriers`` barriers were served or none arrived for
    ``idle_timeout_s`` wall seconds; then unregister (unless ``unregister=False``)."""
    runner = RemoteControllerRunner(ctrl, endpoint, timeout_s=timeout_s)
    try:
        stats = runner.serve(max_barriers=max_barriers, idle_timeout_s=idle_timeout_s)
        if unregister:
            runner.unregister()
        return stats
    finally:
        runner.stop()
