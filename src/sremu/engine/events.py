"""Event heap (DESIGN.md section 4.6). No wall clock here (I7).

Events are ordered by ``(t, prio, seq)`` with ``t`` the exact event time (s), ``prio`` an
:class:`~sremu.contracts.EventPriority` (0 geometry, 1 RIS, 2 impairment, 3 stream command,
9 marker) and ``seq`` the engine-wide monotone sequence number assigned on arrival. There is
no single-slot pending configuration: every event is kept until it is popped.

RIS commands keep their own per-panel books (:mod:`sremu.engine.commands`) because every
receiver sees them at its own sample; this heap carries the events that act at one
absolute time for everybody (geometry ``node.set_position`` / ``node.set_trajectory`` and
recording markers).
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from sremu.contracts import EventPriority
from sremu.timebase import parse_exact

__all__ = ["Event", "EventHeap"]


@dataclass(order=False, eq=False)
class Event:
    """One timed event: exact time ``t`` (s), priority, engine ``seq``, kind and payload."""

    t: Fraction
    prio: int
    seq: int
    kind: str
    payload: Any = None
    cancelled: bool = field(default=False)

    @property
    def key(self) -> tuple[Fraction, int, int]:
        """Sort key ``(t, prio, seq)``."""
        return (self.t, self.prio, self.seq)

    def __lt__(self, other: Event) -> bool:
        return self.key < other.key


class EventHeap:
    """Min-heap of :class:`Event` ordered by ``(t, prio, seq)``."""

    def __init__(self) -> None:
        self._heap: list[Event] = []

    def __len__(self) -> int:
        return sum(1 for e in self._heap if not e.cancelled)

    def push(
        self,
        t: Fraction | int | str,
        prio: int | EventPriority,
        seq: int,
        kind: str,
        payload: Any = None,
    ) -> Event:
        """Insert an event; returns it (``cancel`` it to drop it)."""
        ev = Event(parse_exact(t, name="event time"), int(prio), int(seq), str(kind), payload)
        heapq.heappush(self._heap, ev)
        return ev

    def _prune(self) -> None:
        while self._heap and self._heap[0].cancelled:
            heapq.heappop(self._heap)

    def peek(self) -> Event | None:
        """Earliest pending event, or None."""
        self._prune()
        return self._heap[0] if self._heap else None

    def pop(self) -> Event | None:
        """Remove and return the earliest pending event, or None."""
        self._prune()
        return heapq.heappop(self._heap) if self._heap else None

    def pop_due(self, t: Fraction, *, inclusive: bool = True) -> list[Event]:
        """Pop every event with time ``<= t`` (``< t`` when not ``inclusive``), in order."""
        out: list[Event] = []
        while True:
            ev = self.peek()
            if ev is None or ev.t > t or (not inclusive and ev.t == t):
                return out
            heapq.heappop(self._heap)
            out.append(ev)

    def times(self) -> list[Fraction]:
        """Sorted distinct times of the pending events."""
        return sorted({e.t for e in self._heap if not e.cancelled})

    def __iter__(self) -> Iterator[Event]:
        """Pending events in ``(t, prio, seq)`` order (does not pop)."""
        return iter(sorted(e for e in self._heap if not e.cancelled))
