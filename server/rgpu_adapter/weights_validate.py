"""Pure-stdlib validator prototype for a training ``weights.zip``.

The real training service writes ZIP_STORED archives with:

    model/**                 complete merged weights + config + runtime
    adapter/**               peft adapter (locate) / sam3 adapter files
    evaluation.json
    class_prompts.json       {class_name: prompt}
    training_config.json
    result_manifest.json     metadata incl. weight_fingerprint/weight_hashes
    training.json
    source_manifest.json
    file_checksums.json      {member: sha256}, does NOT include itself

This prototype validates exactly those semantics without importing torch.  It
is intentionally strict: any unlisted member, unsafe path, checksum mismatch,
wrong family/artifact/quality marker, or recomputed weight fingerprint mismatch
fails the archive.  It does not claim model quality.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

MAX_MEMBERS = 100_000
MAX_UNCOMPRESSED = 64 * 1024**3
CHUNK = 4 * 1024 * 1024
REQUIRED = {
    "result_manifest.json",
    "file_checksums.json",
    "class_prompts.json",
    "evaluation.json",
    "training_config.json",
}


class ValidationError(Exception):
    pass


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    while block := stream.read(CHUNK):
        digest.update(block)
    return digest.hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0o120000


def _safe_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValidationError("empty member name")
    if "\\" in name or ":" in name:
        raise ValidationError(f"unsafe member name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValidationError(f"unsafe member name: {name!r}")


def _load_json(archive: zipfile.ZipFile, name: str, limit: int = 16 * 1024**2) -> dict:
    info = archive.getinfo(name)
    if info.file_size > limit:
        raise ValidationError(f"{name} is too large")
    try:
        value = json.loads(archive.read(name).decode("utf-8"))
    except (KeyError, UnicodeDecodeError, ValueError) as exc:
        raise ValidationError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be a JSON object")
    return value


def _fingerprint(hashes: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def validate_weights_zip(
    path: str | Path,
    *,
    family: str,
    expected_parent_fingerprint: str | None = None,
    expected_parent_model_id: str | None = None,
    expected_parent_model_ids: dict[str, str] | None = None,
    expected_classes: set[str] | dict | None = None,
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValidationError(f"archive not found: {path}")
    result: dict = {"family": family, "checksums_ok": False, "quality_status": None}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValidationError("too many members")
        total = sum(info.file_size for info in infos)
        if total > MAX_UNCOMPRESSED:
            raise ValidationError("uncompressed size exceeds limit")
        names: list[str] = []
        lower: set[str] = set()
        for info in infos:
            _safe_name(info.filename)
            if info.is_dir():
                continue
            if _is_symlink(info):
                raise ValidationError(f"symlink member: {info.filename}")
            key = info.filename.casefold()
            if key in lower:
                raise ValidationError(f"duplicate member (case-insensitive): {info.filename}")
            lower.add(key)
            names.append(info.filename)
        bad = archive.testzip()
        if bad:
            raise ValidationError(f"CRC failure: {bad}")
        missing = REQUIRED - set(names)
        if missing:
            raise ValidationError(f"missing members: {sorted(missing)}")

        manifest = _load_json(archive, "result_manifest.json")
        checksums = _load_json(archive, "file_checksums.json")
        classes = _load_json(archive, "class_prompts.json")
        _load_json(archive, "evaluation.json", limit=64 * 1024**2)
        _load_json(archive, "training_config.json")

        if manifest.get("model_family") != family:
            raise ValidationError("result_manifest.model_family mismatch")
        if manifest.get("artifact_type") != "complete_merged_model":
            raise ValidationError("result_manifest.artifact_type mismatch")
        if manifest.get("standalone") is not True:
            raise ValidationError("result_manifest.standalone must be true")
        quality = manifest.get("quality_status")
        if quality != "needs_review":
            raise ValidationError(f"result_manifest.quality_status must be needs_review, got {quality!r}")
        result["quality_status"] = quality
        if expected_parent_fingerprint is not None:
            # If the caller supplies an expected parent fingerprint, the
            # archive MUST carry the same non-empty fingerprint.  A missing or
            # empty field is a rejection, not an optional-field warning.
            if not _is_sha256(expected_parent_fingerprint):
                raise ValidationError("expected parent fingerprint is not a SHA-256")
            actual_parent_fp = manifest.get("parent_model_fingerprint")
            if not isinstance(actual_parent_fp, str) or not actual_parent_fp:
                raise ValidationError(
                    "result_manifest.parent_model_fingerprint is missing"
                )
            if not _is_sha256(actual_parent_fp):
                raise ValidationError(
                    "result_manifest.parent_model_fingerprint is not a SHA-256"
                )
            if actual_parent_fp != expected_parent_fingerprint:
                raise ValidationError("result_manifest.parent_model_fingerprint mismatch")
            result["parent_fingerprint_in_manifest"] = True
        else:
            result["parent_fingerprint_in_manifest"] = bool(
                manifest.get("parent_model_fingerprint")
            )

        if expected_parent_model_id is not None:
            if not isinstance(expected_parent_model_id, str) or not expected_parent_model_id:
                raise ValidationError("expected parent model id is empty")
            actual_model_id = manifest.get("parent_model_version_id")
            if not isinstance(actual_model_id, str) or not actual_model_id:
                raise ValidationError("result_manifest.parent_model_version_id is missing")
            if actual_model_id != expected_parent_model_id:
                raise ValidationError("result_manifest.parent_model_version_id mismatch")
            result["parent_model_id_ok"] = True
            result["parent_model_version_id"] = actual_model_id

        if expected_parent_model_ids is not None:
            # Mapping is trusted-local-id -> remote-id.  The archive stores the
            # remote id; the local alias must be present and unambiguous.
            if not isinstance(expected_parent_model_ids, dict) or not expected_parent_model_ids:
                raise ValidationError("expected parent model id mapping is empty")
            normalized: dict[str, str] = {}
            for local, remote in expected_parent_model_ids.items():
                if not isinstance(local, str) or not local or not isinstance(remote, str) or not remote:
                    raise ValidationError("parent model id mapping is invalid")
                normalized[local] = remote
            remote_values = set(normalized.values())
            if len(remote_values) != 1:
                raise ValidationError("parent model id mapping has multiple remote ids")
            mapped_remote = next(iter(remote_values))
            if expected_parent_model_id is not None and mapped_remote != expected_parent_model_id:
                raise ValidationError("parent model id mapping disagrees with expected remote id")
            actual_model_id = manifest.get("parent_model_version_id")
            if actual_model_id != mapped_remote:
                raise ValidationError(
                    "result_manifest.parent_model_version_id not in trusted mapping"
                )
            locals_for_actual = sorted(k for k, v in normalized.items() if v == actual_model_id)
            if not locals_for_actual:
                raise ValidationError("no trusted local id maps to archive parent id")
            result["parent_model_ids"] = normalized
            result["local_parent_model_ids"] = locals_for_actual

        if expected_classes is not None:
            if isinstance(expected_classes, dict):
                wanted = {
                    str(k): str(v) for k, v in expected_classes.items()
                }
                if classes != wanted:
                    missing = sorted(set(wanted) - set(classes))
                    extra = sorted(set(classes) - set(wanted))
                    changed = sorted(
                        name for name in set(classes) & set(wanted)
                        if classes[name] != wanted[name]
                    )
                    raise ValidationError(
                        "class_prompts.json mismatch; "
                        f"missing={missing}, extra={extra}, changed={changed}"
                    )
            else:
                wanted = set(expected_classes)
                if set(classes) != wanted:
                    raise ValidationError("class_prompts.json class set mismatch")
        result["classes"] = sorted(classes)
        result["class_prompts"] = dict(classes)

        # Every member except file_checksums.json must be listed and hash-match.
        listed = set(checksums)
        actual = set(names) - {"file_checksums.json"}
        if listed != actual:
            extra = sorted(actual - listed)
            absent = sorted(listed - actual)
            raise ValidationError(f"file_checksums coverage mismatch; extra={extra}, missing={absent}")
        hashes = dict(checksums)
        for name, expected in hashes.items():
            _safe_name(name)
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValidationError(f"invalid checksum for {name}")
            with archive.open(name) as stream:
                actual_sha = sha256_stream(stream)
            if actual_sha != expected:
                raise ValidationError(f"checksum mismatch: {name}")
        result["checksums_ok"] = True

        weight_files = {
            info.filename: info
            for info in infos
            if not info.is_dir()
            and info.filename.startswith("model/")
            and info.filename.count("/") == 1
            and info.filename.endswith(".safetensors")
        }
        if not weight_files:
            raise ValidationError("no top-level model/*.safetensors")
        weight_hashes: dict[str, str] = {}
        for name in sorted(weight_files):
            with archive.open(name) as stream:
                weight_hashes[name.split("/", 1)[1]] = sha256_stream(stream)
        recomputed = _fingerprint(weight_hashes)
        if manifest.get("weight_fingerprint") != recomputed:
            raise ValidationError("recomputed weight_fingerprint mismatch")
        if manifest.get("weight_hashes") and manifest["weight_hashes"] != weight_hashes:
            raise ValidationError("result_manifest.weight_hashes mismatch")
        result.update({
            "weight_fingerprint": recomputed,
            "weight_hashes": weight_hashes,
            "member_count": len(names),
            "uncompressed_bytes": total,
        })
        result["ok"] = True
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--family", required=True, choices=["locate", "sam3"])
    parser.add_argument("--parent-fingerprint", default=None)
    parser.add_argument("--parent-model-id", default=None)
    args = parser.parse_args()
    print(json.dumps(validate_weights_zip(
        args.archive, family=args.family,
        expected_parent_fingerprint=args.parent_fingerprint,
        expected_parent_model_id=args.parent_model_id,
    ), ensure_ascii=False, indent=2))
