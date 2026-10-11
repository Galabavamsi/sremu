"""sremu: Smart Radio Environment EMUlator.

A software-only, IQ-level radio-environment emulator in which RIS
configuration is a timestamped runtime command from an external controller, applied at an
exact, logged sample on every receiver (DESIGN.md section 1.1).

Importing ``sremu`` is cheap: public names are resolved lazily on first access, so the
foundation modules (``sremu.units``, ``sremu.timebase``, ...) never pull in the engine.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__: str = version("sremu")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.1.0.dev0"

# public name -> ("module", "attribute", owning work package)
_LAZY: dict[str, tuple[str, str, str]] = {
    "CONTRACT_VERSION": ("sremu.contracts", "CONTRACT_VERSION", "WP0"),
    "Rate": ("sremu.timebase", "Rate", "WP0"),
    "TimeSpec": ("sremu.timebase", "TimeSpec", "WP0"),
    "TxMetadata": ("sremu.contracts", "TxMetadata", "WP0"),
    "RxMetadata": ("sremu.contracts", "RxMetadata", "WP0"),
    "AsyncMetadata": ("sremu.contracts", "AsyncMetadata", "WP0"),
    "StreamCmd": ("sremu.contracts", "StreamCmd", "WP0"),
    "ErrorCode": ("sremu.errors", "ErrorCode", "WP0"),
    "AsyncCode": ("sremu.errors", "AsyncCode", "WP0"),
    "Device": ("sremu.api.device", "Device", "WP7"),
    "StreamArgs": ("sremu.api.streams", "StreamArgs", "WP7"),
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr, owner = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'sremu' has no attribute {name!r}") from None
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if module_name == missing or module_name.startswith(missing + "."):
            raise AttributeError(
                f"sremu.{name} is not yet implemented (expected in {module_name}, owner {owner})"
            ) from exc
        raise
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
