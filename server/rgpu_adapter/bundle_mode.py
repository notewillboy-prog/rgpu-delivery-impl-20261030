"""Explicit, sealed execution mode for RGPU bundles.

A bundle is built for exactly one of two operations:

* ``prepare_only`` -> ``prepare.py`` registers the parent and dataset and stops;
* ``train``        -> ``run.py`` prepares and executes exactly one training job.

The mode marker is a manifest member, so it cannot be changed without changing
the externally delivered manifest SHA-256.  The entry scripts verify the marker
before doing any work; a receipt that names a different mode/entry is rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

MODE_FILE = "bundle_mode.json"
SCHEMA = 1
MODES = {
    "prepare_only": "prepare.py",
    "train": "run.py",
}


def build_record(mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(f"unsupported bundle mode: {mode!r}")
    return {"schema": SCHEMA, "mode": mode, "entry_script": MODES[mode]}


def read_record(root: Path) -> dict | None:
    path = Path(root) / MODE_FILE
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("bundle mode marker is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("bundle mode marker is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"schema", "mode", "entry_script"}:
        raise ValueError("bundle mode marker has an invalid shape")
    if value.get("schema") != SCHEMA or value.get("mode") not in MODES:
        raise ValueError("bundle mode marker has an invalid schema/mode")
    if value.get("entry_script") != MODES[value["mode"]]:
        raise ValueError("bundle mode marker entry script does not match its mode")
    return value


def enforce(root: Path, expected_mode: str, requested_mode: str | None = None) -> dict | None:
    """Validate the sealed marker and the caller's requested mode.

    A missing marker is tolerated only for legacy bundles that have no mode
    marker (the old test fixtures).  New bundles always ship one.  If a caller
    supplies ``requested_mode`` a missing marker is a hard error.
    """
    if expected_mode not in MODES:
        raise ValueError(f"unsupported expected mode: {expected_mode!r}")
    if requested_mode is not None and requested_mode != expected_mode:
        raise ValueError(
            f"bundle mode mismatch: requested {requested_mode!r}, entry expects {expected_mode!r}"
        )
    record = read_record(root)
    if record is None:
        if requested_mode is not None:
            raise ValueError("bundle mode marker is missing")
        return None
    if record["mode"] != expected_mode:
        raise ValueError(
            f"bundle mode mismatch: bundle is {record['mode']!r}, entry expects {expected_mode!r}"
        )
    if record["entry_script"] != MODES[expected_mode]:
        raise ValueError("bundle mode entry script mismatch")
    return record
