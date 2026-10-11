"""CPU topology, slot planning and pinning for the Linux benchmark jobs (DESIGN.md 12.3).

Section 12.3 asks for pinning (``taskset`` on Linux) to be stated with every number. The
jobs pin at two levels:

* **Suite level** (B1, B3, B4 on a quiet host): ``bench/aws/run_job.py`` runs the suite
  under ``taskset -c <engine CPUs>`` and exports ``SREMU_CLIENT_CPUS`` (client and
  controller processes, :func:`bench.pin_client_cpus`) and ``NUMBA_NUM_THREADS``.
* **Point level** (B2 on a large host): the B2 orchestrator starts one worker process per
  point with its own CPU slot (:func:`plan_slots`), so concurrent points never share a
  core.

Topology comes from ``/sys/devices/system`` (SMT siblings from
``cpu*/topology/thread_siblings_list``, NUMA nodes from ``node*/cpulist``); a slot is a set
of whole physical cores (with all their SMT siblings) inside one NUMA node. Only CPUs in
the process's current affinity are used. On a host without ``/sys`` (Windows), every
logical CPU is its own core in node 0 and pinning is a recorded no-op.

``python -m bench._cpus`` prints the topology and a slot plan (used by ``run_job.py``)::

    python -m bench._cpus plan --cpus-per-slot 16 --reserve-cores 1 [--slots 1] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYS_ROOT = Path("/sys/devices/system")

__all__ = [
    "Core",
    "all_cpus",
    "format_cpu_list",
    "online_cpus",
    "pin_self",
    "plan_slots",
    "read_topology",
]


@dataclass(frozen=True)
class Core:
    """One physical core: its NUMA node and its logical CPUs (SMT siblings), sorted."""

    node: int
    cpus: tuple[int, ...]


def _parse_list(text: str) -> list[int]:
    from bench import parse_cpu_list

    return parse_cpu_list(text.strip())


def format_cpu_list(cpus: Iterable[int]) -> str:
    """``[0, 1, 2, 5, 96, 97]`` -> ``"0-2,5,96-97"`` (Linux cpulist syntax; empty -> "")."""
    xs = sorted({int(c) for c in cpus})
    parts: list[str] = []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        parts.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(parts)


def online_cpus() -> list[int]:
    """CPUs this process may run on (its affinity on Linux, else ``range(os.cpu_count())``)."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def all_cpus(sys_root: Path = SYS_ROOT) -> list[int]:
    """Every online CPU of the host (``<sys_root>/cpu/online``), whatever this process's own
    affinity is (a planner pinned to the OS core still plans the whole host); falls back to
    ``range(os.cpu_count())``."""
    try:
        return _parse_list((sys_root / "cpu" / "online").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return list(range(os.cpu_count() or 1))


def read_topology(sys_root: Path = SYS_ROOT, cpus: Sequence[int] | None = None) -> list[Core]:
    """Physical cores among ``cpus`` (default :func:`online_cpus`), ordered by node, then by
    their first CPU. Without topology files every CPU is a one-thread core in node 0."""
    allowed = set(online_cpus() if cpus is None else cpus)
    node_of: dict[int, int] = {}
    node_dir = sys_root / "node"
    if node_dir.is_dir():
        for d in sorted(node_dir.glob("node[0-9]*")):
            try:
                for c in _parse_list((d / "cpulist").read_text(encoding="utf-8")):
                    node_of[c] = int(d.name[4:])
            except (OSError, ValueError):
                continue
    seen: set[int] = set()
    cores: list[Core] = []
    for c in sorted(allowed):
        if c in seen:
            continue
        sib_file = sys_root / "cpu" / f"cpu{c}" / "topology" / "thread_siblings_list"
        try:
            sibs = [s for s in _parse_list(sib_file.read_text(encoding="utf-8")) if s in allowed]
        except (OSError, ValueError):
            sibs = [c]
        if c not in sibs:
            sibs = [c, *sibs]
        seen.update(sibs)
        cores.append(Core(node_of.get(c, 0), tuple(sorted(sibs))))
    cores.sort(key=lambda k: (k.node, k.cpus[0]))
    return cores


def plan_slots(
    cores: Sequence[Core],
    cpus_per_slot: int,
    *,
    reserve_cores: int = 1,
    max_slots: int | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Disjoint CPU slots of ``cpus_per_slot`` logical CPUs, each made of whole physical
    cores inside one NUMA node, after reserving the first ``reserve_cores`` cores of the
    first node (OS, orchestrator, load sampling). Returns ``(slots, reserved_cpus)``.

    A slot takes whole cores until it has at least ``cpus_per_slot`` CPUs (so an odd count
    on an SMT-2 host gets one extra sibling); a node's remainder that cannot fill a slot
    stays idle. Raises ``ValueError`` when not even one slot fits."""
    if cpus_per_slot < 1:
        raise ValueError("cpus_per_slot must be >= 1")
    by_node: dict[int, list[Core]] = {}
    for k in cores:
        by_node.setdefault(k.node, []).append(k)
    nodes = sorted(by_node)
    reserved: list[int] = []
    if nodes and reserve_cores > 0:
        first = by_node[nodes[0]]
        for k in first[:reserve_cores]:
            reserved.extend(k.cpus)
        by_node[nodes[0]] = first[reserve_cores:]
    slots: list[list[int]] = []
    for n in nodes:
        avail = by_node[n]
        i = 0
        while i < len(avail):
            got: list[int] = []
            j = i
            while j < len(avail) and len(got) < cpus_per_slot:
                got.extend(avail[j].cpus)
                j += 1
            if len(got) < cpus_per_slot:
                break
            slots.append(sorted(got))  # whole cores: one more CPU than asked when odd
            i = j
            if max_slots is not None and len(slots) >= max_slots:
                return slots, sorted(reserved)
    if not slots:
        raise ValueError(
            f"no slot of {cpus_per_slot} CPUs fits in {sum(len(k.cpus) for k in cores)} CPUs "
            f"(reserve_cores={reserve_cores})"
        )
    return slots, sorted(reserved)


def pin_self(cpus: Sequence[int] | None) -> dict[str, Any]:
    """Pin this process (every thread, :func:`bench.set_process_affinity`; Linux
    ``sched_setaffinity``) to ``cpus`` before numba starts its threads. Returns
    ``{"requested", "applied", "affinity", "threads_pinned", "thread_affinities", "note"}``
    for the manifest (``thread_affinities``: the distinct masks of all threads read back)."""
    from bench import set_process_affinity, thread_affinities

    rec: dict[str, Any] = {"requested": None if cpus is None else format_cpu_list(cpus)}
    if not cpus:
        rec.update(applied=False, note="no CPU list")
    elif hasattr(os, "sched_setaffinity"):
        n = set_process_affinity(list(cpus))
        rec.update(applied=True, threads_pinned=n, note="sched_setaffinity (all threads)")
    else:
        rec.update(applied=False, note=f"pinning not supported on {sys.platform}")
    rec["affinity"] = format_cpu_list(online_cpus())
    rec["thread_affinities"] = thread_affinities()
    return rec


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bench._cpus", description="CPU topology and slot plan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="print a slot plan")
    p.add_argument("--cpus-per-slot", type=int, required=True)
    p.add_argument("--reserve-cores", type=int, default=1)
    p.add_argument("--slots", type=int, default=None, help="at most this many slots")
    p.add_argument("--json", action="store_true")
    sub.add_parser("topology", help="print the cores")
    a = ap.parse_args(argv)
    cores = read_topology()
    if a.cmd == "topology":
        print(json.dumps([{"node": k.node, "cpus": list(k.cpus)} for k in cores]))
        return 0
    slots, reserved = plan_slots(
        cores, a.cpus_per_slot, reserve_cores=a.reserve_cores, max_slots=a.slots
    )
    used = {c for s in slots for c in s} | set(reserved)
    rest = [c for k in cores for c in k.cpus if c not in used]
    if a.json:
        print(json.dumps({"slots": slots, "reserved": reserved, "unused": rest}))
    else:  # shell-friendly: one line per slot, then the reserved and unused CPU lists
        for s in slots:
            print(format_cpu_list(s))
        print("reserved", format_cpu_list(reserved) or "-")
        print("unused", format_cpu_list(rest) or "-")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
