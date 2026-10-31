"""Validate the provenance of the parent model class map used in a bundle.

The class map is not guessed from the family.  It must come from a JSON
metadata record whose archive SHA-256 binds it to the exact parent archive and
whose ``source`` names where the class information came from.  An empty map is
legal, but only when the metadata explicitly records the archive scan/index
that found no class registry entries.  A non-empty map additionally needs
per-class evidence with a source document and SHA-256.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{64}\Z")
FAMILIES = frozenset({"locate", "sam3"})


def _require_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"parent metadata {label} is required")
    return value


def _require_sha(value, label):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f"parent metadata {label} is not a SHA-256")
    return value


def load(path, *, family, archive_sha256):
    """Load and validate a parent metadata sidecar for one family/archive."""
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("parent metadata is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("schema") != 1:
        raise ValueError("parent metadata schema is invalid")
    if value.get("family") not in FAMILIES or value.get("family") != family:
        raise ValueError("parent metadata family does not match the requested bundle")
    actual_archive = _require_sha(value.get("archive_sha256"), "archive_sha256")
    if actual_archive != _require_sha(archive_sha256, "actual archive"):
        raise ValueError("parent metadata archive SHA-256 mismatch")
    approved_fp = _require_sha(value.get("approved_base_fingerprint"),
                               "approved_base_fingerprint")
    parent_fp = _require_sha(value.get("parent_fingerprint"), "parent_fingerprint")
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("parent metadata source must be an object")
    _require_text(source.get("kind"), "source.kind")
    _require_text(source.get("detail"), "source.detail")
    classes = value.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("parent metadata classes must be an object")
    for key, prompt in classes.items():
        if not isinstance(key, str) or not key.strip() or len(key) > 200:
            raise ValueError("parent metadata has an invalid class name")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 300:
            raise ValueError("parent metadata has an invalid class prompt")
    evidence = value.get("class_evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("parent metadata class_evidence must be a list")
    if classes:
        if not evidence:
            raise ValueError(
                "non-empty parent classes require class_evidence; refusing a guessed map"
            )
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("parent class evidence entry is invalid")
            _require_text(item.get("source"), "class_evidence.source")
            _require_sha(item.get("sha256"), "class_evidence.sha256")
            item_classes = item.get("classes")
            if not isinstance(item_classes, dict) or item_classes != classes:
                raise ValueError("parent class evidence does not match the class map")
    elif evidence:
        raise ValueError("empty parent class map must not carry class evidence")
    return {
        "schema": 1,
        "family": family,
        "archive_sha256": actual_archive,
        "approved_base_fingerprint": approved_fp,
        "parent_fingerprint": parent_fp,
        "classes": dict(classes),
        "source": dict(source),
        "class_evidence": list(evidence),
        "path": str(path.resolve()),
    }
