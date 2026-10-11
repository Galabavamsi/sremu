"""Pre-registration lock of the V-LAB protocol (DESIGN.md section 10.9, "Pre-registration").

Procedure:

1. The lead commits ``PROTOCOL.md`` and ``protocol.json``.
2. In a separate commit the lead sets :data:`PREREGISTERED_SHA256` to
   :func:`protocol_digest` (print it with ``python -m validation.ants_fig4b.protocol_lock``).
3. :func:`load_measured` refuses to touch the digitized data unless the digest of the
   protocol files on disk equals :data:`PREREGISTERED_SHA256`. Development uses
   :func:`validation.ants_fig4b.decoy.synthetic_decoy` only.
4. The user deposits ``PROTOCOL.md`` and the digest on Zenodo or OSF before unblinding
   (human gate H-6).

**Digest.** ``sha256(md || cj)`` where ``md`` is ``PROTOCOL.md`` with every ``CRLF``
replaced by ``LF`` and ``cj`` is the canonical JSON of ``protocol.json``
(``json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)``, UTF-8).
The canonical form makes the digest independent of whitespace and key order in
``protocol.json``; the line-ending normalization makes it independent of the checkout's
``core.autocrlf``. Any change of content changes it. Reproduce it outside sremu with
``--write-canonical`` and ``cat``/``sha256sum``.

The lock is checked **before** the data path is resolved: when it fails, the data file is
never opened, read, hashed or stat'ed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from validation.ants_fig4b.protocol import (
    PROTOCOL_DIR,
    PROTOCOL_JSON,
    PROTOCOL_MD,
    canonical_json_bytes,
    cell_layout,
    load_protocol,
)

__all__ = [
    "PREREGISTERED_SHA256",
    "MeasuredData",
    "MeasuredDataError",
    "ProtocolHashMismatch",
    "ProtocolLockError",
    "ProtocolNotLocked",
    "check_lock",
    "digest_input_bytes",
    "is_locked",
    "load_measured",
    "parse_measured_csv",
    "protocol_digest",
]

PREREGISTERED_SHA256: Final[str] = "d8623254ac6f9785f147b38fd8449b457f87c0c658baaae688eff597b09e0b7c"
"""SHA-256 of the locked protocol (:func:`protocol_digest`). Empty until the lead locks it
in a separate commit; while it is empty :func:`load_measured` refuses."""


class ProtocolLockError(RuntimeError):
    """The protocol lock does not allow unblinding."""


class ProtocolNotLocked(ProtocolLockError):
    """``PREREGISTERED_SHA256`` is empty: the protocol has not been locked yet."""


class ProtocolHashMismatch(ProtocolLockError):
    """The protocol files on disk differ from the locked ones."""


class MeasuredDataError(ValueError):
    """The measured CSV cannot be mapped onto the protocol's 30 cells."""


# =========================================================================== digest


def digest_input_bytes(protocol_dir: str | Path | None = None) -> bytes:
    """The exact bytes whose SHA-256 is the protocol digest: ``PROTOCOL.md`` (CRLF -> LF)
    followed by the canonical JSON of ``protocol.json``."""
    d = Path(protocol_dir) if protocol_dir is not None else PROTOCOL_DIR
    md = (d / PROTOCOL_MD).read_bytes().replace(b"\r\n", b"\n")
    with (d / PROTOCOL_JSON).open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    return md + canonical_json_bytes(obj)


def protocol_digest(protocol_dir: str | Path | None = None) -> str:
    """SHA-256 hex digest of the protocol files (see the module docstring)."""
    return hashlib.sha256(digest_input_bytes(protocol_dir)).hexdigest()


def _locked_value() -> str:
    # read the module attribute at call time (tests monkeypatch it)
    return str(sys.modules[__name__].PREREGISTERED_SHA256)


def is_locked() -> bool:
    """Whether :data:`PREREGISTERED_SHA256` has been set."""
    return bool(_locked_value())


def check_lock(protocol_dir: str | Path | None = None) -> str:
    """Return the protocol digest if it equals :data:`PREREGISTERED_SHA256`; raise
    :class:`ProtocolNotLocked` if the constant is empty and :class:`ProtocolHashMismatch`
    if the files changed. Touches only the two protocol files."""
    expected = _locked_value()
    if not expected:
        raise ProtocolNotLocked(
            "V-LAB protocol is not locked: PREREGISTERED_SHA256 is empty. The measured data "
            "stay blinded; use synthetic_decoy(seed) for development."
        )
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ProtocolHashMismatch(f"PREREGISTERED_SHA256 is not a sha256 hex digest: {expected!r}")
    got = protocol_digest(protocol_dir)
    if got != expected:
        raise ProtocolHashMismatch(
            f"V-LAB protocol files changed since the lock: digest {got} != locked {expected}. "
            "Refusing to load the measured data."
        )
    return got


# =========================================================================== data


@dataclass(frozen=True)
class MeasuredData:
    """The digitized map in protocol cell order (data class DIG)."""

    y_dbm: np.ndarray
    sigma_dig_db: np.ndarray
    theta_deg: np.ndarray
    r_m: np.ndarray
    source_path: str
    sha256: str
    protocol_digest: str
    columns: Mapping[str, str] = field(default_factory=dict)
    data_class: str = "DIG"


_COLUMN_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "sigma_db": (r"unc", r"sigma", r"std", r"err", r"^u_?db$", r"^u$"),
    "theta_deg": (r"theta", r"angle", r"azimuth", r"^th(_deg)?$", r"^phi(_deg)?$"),
    "r_m": (r"^r(_m)?$", r"radius", r"range", r"dist"),
    "y_dbm": (r"dbm", r"power", r"^p(_?rx)?$", r"rx"),
}


def _norm(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)  # drop bracketed units
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _match_columns(header: Sequence[str], override: Mapping[str, str] | None) -> dict[str, str]:
    """Map the four required quantities to header names (explicit override wins)."""
    if override:
        missing = [k for k in _COLUMN_PATTERNS if k not in override]
        if missing:
            raise MeasuredDataError(f"column override lacks {missing}")
        for k, v in override.items():
            if v not in header:
                raise MeasuredDataError(
                    f"override column {v!r} for {k} not in header {list(header)}"
                )
        return dict(override)
    out: dict[str, str] = {}
    taken: set[str] = set()
    for key, pats in _COLUMN_PATTERNS.items():  # sigma first, so 'power_unc' is not power
        hits = [h for h in header if h not in taken and any(re.search(p, _norm(h)) for p in pats)]
        if len(hits) != 1:
            raise MeasuredDataError(
                f"cannot map {key!r} onto the header {list(header)} (candidates {hits}); pass "
                "columns={...} explicitly"
            )
        out[key] = hits[0]
        taken.add(hits[0])
    return out


def parse_measured_csv(
    text: str,
    protocol: Mapping[str, Any],
    *,
    columns: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Parse the digitized CSV into ``(y_dbm[30], sigma_dig_db[30], column map)`` in protocol
    cell order. Long format, one row per cell; lines starting with ``#`` are comments.
    Column names are matched by keyword (or ``columns``); every protocol cell must appear
    exactly once within ``data.match_tolerance``. Error messages name header fields and
    cell coordinates only, never measured values."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise MeasuredDataError("empty CSV")
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header = [h.strip() for h in rows[0]]
    cmap = _match_columns(header, columns)
    col = {k: header.index(v) for k, v in cmap.items()}
    lay = cell_layout(protocol)
    tol = protocol["data"]["match_tolerance"]
    y = np.full(lay.n, np.nan)
    sg = np.full(lay.n, np.nan)
    for ln, row in enumerate(rows[1:], start=2):
        try:
            th = float(row[col["theta_deg"]])
            rr = float(row[col["r_m"]])
            yy = float(row[col["y_dbm"]])
            ss = float(row[col["sigma_db"]])
        except (ValueError, IndexError) as exc:
            raise MeasuredDataError(f"line {ln}: cannot read the four fields") from exc
        try:
            i = lay.index(th, rr, tol_theta=float(tol["theta_deg"]), tol_r=float(tol["r_m"]))
        except KeyError:
            raise MeasuredDataError(
                f"line {ln}: (theta={th:g} deg, r={rr:g} m) is not a protocol cell"
            ) from None
        if not np.isnan(y[i]):
            raise MeasuredDataError(f"line {ln}: duplicate cell (theta={th:g}, r={rr:g})")
        if not (np.isfinite(yy) and np.isfinite(ss) and ss >= 0.0):
            raise MeasuredDataError(f"line {ln}: non-finite value or negative uncertainty")
        y[i], sg[i] = yy, ss
    if np.isnan(y).any():
        miss = [(float(lay.theta_deg[i]), float(lay.r_m[i])) for i in np.flatnonzero(np.isnan(y))]
        raise MeasuredDataError(f"cells missing from the CSV: {miss}")
    return y, sg, cmap


def _resolve_data_path(path: str | Path | None, protocol: Mapping[str, Any]) -> Path:
    """Explicit ``path``, else the protocol's search paths under ``$SREMU_PRIVATE_DATA``.
    Called only after the lock check."""
    if path is not None:
        return Path(path)
    data = protocol["data"]
    root = os.environ.get(str(data["env_var"]))
    if not root:
        raise FileNotFoundError(
            f"${data['env_var']} is not set; pass path=... or set it to the private data root"
        )
    for tmpl in data["search_paths"]:
        p = Path(str(tmpl).format(root=root))
        if p.is_file():
            return p
    raise FileNotFoundError(f"{data['file_name']} not found under ${data['env_var']}")


def _read_data_file(path: Path) -> bytes:
    """The single place where the measured file is read (tests assert it is not reached
    when the lock fails)."""
    return path.read_bytes()


def load_measured(
    path: str | Path | None = None,
    *,
    protocol_dir: str | Path | None = None,
    columns: Mapping[str, str] | None = None,
) -> MeasuredData:
    """Load the digitized Fig. 4(b) map, **only** if the protocol lock holds.

    Order: (1) :func:`check_lock` (raises :class:`ProtocolNotLocked` or
    :class:`ProtocolHashMismatch`; the data file is not touched); (2) resolve ``path`` (or
    ``$SREMU_PRIVATE_DATA``); (3) read, hash and parse. ``protocol_dir`` selects the
    protocol files (default: this package)."""
    digest = check_lock(protocol_dir)
    protocol = load_protocol(protocol_dir)
    p = _resolve_data_path(path, protocol)
    raw = _read_data_file(p)
    y, sg, cmap = parse_measured_csv(raw.decode("utf-8-sig"), protocol, columns=columns)
    lay = cell_layout(protocol)
    return MeasuredData(
        y_dbm=y,
        sigma_dig_db=sg,
        theta_deg=lay.theta_deg.copy(),
        r_m=lay.r_m.copy(),
        source_path=str(p),
        sha256=hashlib.sha256(raw).hexdigest(),
        protocol_digest=digest,
        columns=cmap,
    )


# =========================================================================== CLI


def main(argv: Sequence[str] | None = None) -> int:
    """Print the protocol digest and the lock state; optionally write the digest input."""
    ap = argparse.ArgumentParser(prog="python -m validation.ants_fig4b.protocol_lock")
    ap.add_argument("--protocol-dir", type=Path, default=None)
    ap.add_argument(
        "--write-canonical",
        type=Path,
        default=None,
        help="write the canonical JSON of protocol.json to this file (for external checks)",
    )
    args = ap.parse_args(argv)
    d = args.protocol_dir
    digest = protocol_digest(d)
    print(f"protocol digest      {digest}")
    locked = _locked_value()
    print(f"PREREGISTERED_SHA256 {locked or '(empty: not locked)'}")
    if locked:
        print("lock state           " + ("MATCH" if locked == digest else "MISMATCH"))
    if args.write_canonical is not None:
        dd = Path(d) if d is not None else PROTOCOL_DIR
        obj: Any = json.loads((dd / PROTOCOL_JSON).read_text(encoding="utf-8"))
        args.write_canonical.write_bytes(canonical_json_bytes(obj))
        print(f"wrote {args.write_canonical} (digest = sha256(PROTOCOL.md LF bytes || this file))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
