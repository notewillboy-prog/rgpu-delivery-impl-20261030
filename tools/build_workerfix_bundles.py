#!/usr/bin/env python3
"""Build fresh RGPU bundles that pin the patched worker.

The upstream ``rgpu_bundle.build_bundle`` copies the family source tree from the
live project root, which still contains the old worker.  This driver overlays the
patched worker *before* the one post-build manifest rewrite, adds the new
``run.py``/``weights_validate.py``, injects the trusted local parent id, and then
seals the bundle.  It never starts ``prepare.py``, a service, or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGES = {"locate": "training_service", "sam3": "sam3_training_service"}
FAMILIES = tuple(PACKAGES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def fresh_destination(dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle destination: {dest}")
    for marker in ("prepare.events.jsonl", "prepared_runtime", "outputs"):
        if (dest / marker).exists():
            raise RuntimeError(f"destination already has execution artefacts: {dest / marker}")


def seal(dest: Path) -> tuple[dict, str]:
    excluded = {"bundle_manifest.json", "prepare.events.jsonl", "prepare.out", "build.log"}
    manifest = {}
    for path in sorted(dest.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(dest).as_posix()
        if path.name in excluded or rel.startswith(("prepared_runtime/", "outputs/")):
            continue
        manifest[rel] = sha256(path)
    write_json(dest / "bundle_manifest.json", manifest)
    return manifest, sha256(dest / "bundle_manifest.json")


def patch_one(family: str, *, base_root: Path, data_root: Path, archive_root: Path,
              patched_root: Path, adapter_root: Path, dataset: Path | None = None,
              class_name: str = "fish", class_prompt: str = "fish",
              parent_classes: dict[str, str] | None = None,
              name_prefix: str = "workerfix") -> dict:
    package = PACKAGES[family]
    dest = data_root / "bundles" / family
    receipt_path = data_root / "bundles" / f"{family}.bundle_receipt.json"
    fresh_destination(dest)

    sys.path.insert(0, str(base_root / "src"))
    from label_workbench import rgpu_bundle  # noqa: PLC0415

    if Path(rgpu_bundle.PROJECT) != base_root:
        raise RuntimeError(f"builder project root {rgpu_bundle.PROJECT} != {base_root}")

    approved = archive_root / "locate_anything_3b.zip" if family == "locate" else archive_root / "sam3_image_base.zip"
    parent = archive_root / "locate_anything_3b.zip" if family == "locate" else archive_root / "sam3_parent_native.zip"
    dataset = Path(dataset) if dataset is not None else archive_root / "fishing_min.zip"
    parent_classes = dict(parent_classes or {"fish": "fish"})
    fingerprints = {
        "locate": "f07a9ae5eaa737c26061f722c117a1bf552d548c9f2afecbf5858ec43f25da1e",
        "sam3": "cb3452ff8bf09c969d7ec8869f746abdfb27606616464a7b416f7a47d3c747ba",
    }
    training = {
        "locate": {
            "class_prompts": [{"class_name": class_name, "prompt": class_prompt}],
            "steps": 1, "image_tokens": 1024, "max_train_per_class": 4,
            "max_eval_per_class": 2, "seed": 20261022,
            "name": f"{name_prefix}-locate-1step", "training_mode": "new_only",
        },
        "sam3": {
            "class_prompts": [{"class_name": class_name, "prompt": class_prompt}],
            "steps": 1, "resolution": 1008, "max_train_per_class": 4,
            "max_eval_per_class": 2, "seed": 20261022,
            "name": f"{name_prefix}-sam3-1step", "training_mode": "new_only",
        },
    }
    for path in (approved, parent, dataset):
        if not path.is_file():
            raise FileNotFoundError(path)

    builder_receipt = rgpu_bundle.build_bundle(
        dest, family, approved, parent, dataset,
        approved_base_fingerprint=fingerprints[family],
        parent_fingerprint=fingerprints[family],
        parent_classes=parent_classes,
        training=training[family],
    )

    # Overlay the independently patched producer BEFORE the only manifest seal.
    target_worker = dest / "server" / package / "worker.py"
    source_worker = patched_root / package / "worker.py"
    if target_worker.is_symlink() or not target_worker.is_file():
        raise RuntimeError(f"builder did not create a regular worker: {target_worker}")
    shutil.copyfile(source_worker, target_worker)
    if sha256(target_worker) != sha256(source_worker):
        raise RuntimeError("patched worker copy mismatch")

    for name in ("run.py", "weights_validate.py"):
        source = adapter_root / name
        target = dest / name
        shutil.copyfile(source, target)
        if sha256(target) != sha256(source):
            raise RuntimeError(f"{name} copy mismatch")

    request_path = dest / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    local_parent_id = "sha256:" + request["parent_model"]["sha256"]
    request["parent_model_id"] = local_parent_id
    request["parent_model"]["id"] = local_parent_id
    write_json(request_path, request)

    manifest, manifest_sha = seal(dest)
    rgpu_bundle.verify_bundle_tree(dest.resolve(), manifest)
    worker_sha = sha256(target_worker)
    worker_source_sha = sha256(source_worker)
    worker_text = target_worker.read_text(encoding="utf-8")
    if worker_text.count("parent_model_fingerprint") != 2:
        raise RuntimeError("patched worker is missing the binding fields")
    scripts = {name: sha256(dest / name) for name in ("run.py", "weights_validate.py")}
    final = {
        "schema": 1,
        "directory": str(dest.resolve()),
        "family": family,
        "request_id": builder_receipt.get("request_id"),
        "manifest_sha256": manifest_sha,
        "entry_script_sha256": scripts["run.py"],
        "mode": "prepare_only",
        "entry": ["python", "-I", "-B", "-X", "utf8", "run.py", "--manifest-sha256", manifest_sha],
        "outputs": ["outputs/**"],
        "builder_manifest_sha256": builder_receipt.get("manifest_sha256"),
        "approved_base_fingerprint": fingerprints[family],
        "parent_fingerprint": fingerprints[family],
        "parent_model_id": local_parent_id,
        "scripts": scripts,
        "dataset": {"path": str(dataset.resolve()), "sha256": sha256(dataset)},
        "parent_classes": parent_classes,
        "new_class": {"name": class_name, "prompt": class_prompt},
        "worker": {
            "package": package,
            "source": str(source_worker),
            "source_sha256": worker_source_sha,
            "bundle_sha256": worker_sha,
            "parent_model_fingerprint_fields": 2,
        },
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(receipt_path, final)
    if sha256(dest / "bundle_manifest.json") != manifest_sha:
        raise RuntimeError("manifest changed while writing receipt")
    if manifest.get(f"server/{package}/worker.py") != worker_sha:
        raise RuntimeError("manifest does not bind the patched worker")
    return {
        "family": family,
        "bundle": str(dest.resolve()),
        "receipt": str(receipt_path.resolve()),
        "manifest_sha256": manifest_sha,
        "manifest_files": len(manifest),
        "worker_sha256": worker_sha,
        "worker_source_sha256": worker_source_sha,
        "scripts": scripts,
        "parent_model_id": local_parent_id,
        "dataset_sha256": sha256(dataset),
        "parent_classes": parent_classes,
        "new_class": {"name": class_name, "prompt": class_prompt},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, default=Path("/home/rgpu/rgpu-live-20261030"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--patched-root", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--class-name", default="fish")
    parser.add_argument("--class-prompt", default="fish")
    parser.add_argument("--parent-classes-json", default='{"fish": "fish"}')
    parser.add_argument("--name-prefix", default="workerfix")
    parser.add_argument("--only", action="append", choices=FAMILIES, default=None)
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    (args.data_root / "bundles").mkdir(parents=True, exist_ok=True)
    only = args.only or list(FAMILIES)
    built = []
    for family in only:
        built.append(patch_one(
            family, base_root=args.base_root.resolve(),
            data_root=args.data_root.resolve(), archive_root=args.archive_root.resolve(),
            patched_root=args.patched_root.resolve(), adapter_root=args.adapter_root.resolve(),
            dataset=args.dataset.resolve() if args.dataset is not None else None,
            class_name=args.class_name, class_prompt=args.class_prompt,
            parent_classes=json.loads(args.parent_classes_json),
            name_prefix=args.name_prefix,
        ))
    write_json(args.data_root / "build_result.json", {"ok": True, "built": built})
    print(json.dumps({"ok": True, "built": built}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
