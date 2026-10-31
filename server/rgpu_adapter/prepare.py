"""Prepare, but deliberately do not execute, an isolated RGPU training job.

The adapter reuses the existing family-specific upload and job validators.  It
must not run a real model until the RGPU server has a verified external GPU
coordination mechanism and the caller has a durable submission receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / 'server'
sys.path.insert(0, str(ROOT))
sys.path[:0] = [str(SERVER), str(SERVER / 'vendor_sam3')]
FAMILIES = {'locate': 'training_service', 'sam3': 'sam3_training_service'}
try:
    from . import bundle_mode
except ImportError:  # script execution: ``python prepare.py``
    import bundle_mode  # type: ignore

SHA = re.compile(r'[0-9a-f]{64}\Z')
PREPARE_RESERVE = 8 * 1024**3
METADATA_RESERVE = 64 * 1024**2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def checked_file(root, entry, *, allow_id=False):
    keys = {'path', 'sha256'} | ({'id'} if allow_id else set())
    if not isinstance(entry, dict) or set(entry) not in ({'path', 'sha256'}, keys):
        raise ValueError('input identity is invalid')
    if not SHA.fullmatch(entry['sha256']):
        raise ValueError('input identity is invalid')
    if 'id' in entry and entry['id'] != 'sha256:' + entry['sha256']:
        raise ValueError('input content-address does not match SHA-256')
    rel = Path(entry['path'])
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('input path is invalid')
    candidate = root / rel
    if candidate.is_symlink() or not candidate.is_file() or not candidate.resolve().is_relative_to(root):
        raise ValueError('input is not a file inside this bundle')
    if digest(candidate) != entry['sha256']:
        raise ValueError('input SHA-256 mismatch')
    return candidate


def verify_manifest(root, expected_sha):
    """Check contents against the separately delivered sender receipt."""
    if not isinstance(expected_sha,str) or not SHA.fullmatch(expected_sha):
        raise ValueError('external manifest SHA-256 is required')
    manifest_path = root / 'bundle_manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file() or digest(manifest_path)!=expected_sha:
        raise ValueError('external manifest SHA-256 mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or not {'prepare.py','request.json'}<=manifest.keys():
        raise ValueError('bundle manifest invalid')
    expected=set();dirs={'.'};lower=set()
    for name, value in manifest.items():
        if not isinstance(name, str) or not isinstance(value, str) or not SHA.fullmatch(value):
            raise ValueError('bundle manifest record invalid')
        rel=PurePosixPath(name)
        if ('\\' in name or ':' in name or rel.is_absolute() or '..' in rel.parts or not rel.parts
                or str(rel)!=name or name in {'bundle_manifest.json','prepare.events.jsonl'}):
            raise ValueError('unsafe bundle path')
        if name.casefold() in lower:raise ValueError('duplicate bundle path')
        lower.add(name.casefold());expected.add(name)
        dirs.update(parent.as_posix() for parent in rel.parents)
    allowed=expected|{'bundle_manifest.json','prepare.events.jsonl'};found=set()
    def fail(error):raise error
    for folder,subdirs,files in os.walk(root,followlinks=False,onerror=fail):
        for name in subdirs+files:
            path=Path(folder)/name;rel=path.relative_to(root).as_posix();info=path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(path,'is_junction',lambda:False)():
                raise ValueError('bundle contains a symbolic link or junction')
            if stat.S_ISDIR(info.st_mode):
                if rel not in dirs:raise ValueError('bundle contains an unlisted directory: '+rel)
            elif stat.S_ISREG(info.st_mode):
                if rel not in allowed:raise ValueError('bundle contains an unlisted file: '+rel)
                found.add(rel)
            else:raise ValueError('bundle contains a non-regular file')
    if not (expected|{'bundle_manifest.json'})<=found:raise ValueError('bundle is missing files')
    for name,value in manifest.items():checked_file(root,{'path':name,'sha256':value})


def unpack_bytes(path):
    with zipfile.ZipFile(path) as archive:
        items=archive.infolist()
        if len(items)>100000:raise ValueError('ZIP file count exceeds limit')
        total=sum(item.file_size for item in items)
        if total>64*1024**3:raise ValueError('ZIP uncompressed size exceeds limit')
        return total


def prepare_space_requirement(family,approved,parent,dataset):
    base_bytes=unpack_bytes(approved)
    parent_bytes=unpack_bytes(parent)
    data_bytes=unpack_bytes(dataset)
    # Native SAM3 checkpoint tensors may use a larger dtype than the approved
    # model; reserve up to four times the entire approved uncompressed archive
    # or the parent archive, whichever is larger, for a converted copy.
    conversion=max(parent_bytes,4*base_bytes) if family=='sam3' else parent_bytes
    payload=base_bytes+parent_bytes+conversion+data_bytes+METADATA_RESERVE
    reserve=max(PREPARE_RESERVE,(payload+9)//10)
    return {'base_unpacked_bytes':base_bytes,'parent_unpacked_bytes':parent_bytes,
            'conversion_allowance_bytes':conversion,'dataset_unpacked_bytes':data_bytes,
            'metadata_bytes':METADATA_RESERVE,'reserve_bytes':reserve,'required_bytes':payload+reserve}


def _prepare(root, expected_manifest_sha, event):
    verify_manifest(root, expected_manifest_sha)
    event('manifest_verified',manifest_sha256=expected_manifest_sha)
    spec = json.loads((root / 'request.json').read_text(encoding='utf-8'))
    if spec.get('schema') != 1 or spec.get('family') not in FAMILIES:
        raise ValueError('unknown request schema or family')
    family = spec['family']
    approved = checked_file(root, spec['approved_base'])
    parent = checked_file(root, spec['parent_model'], allow_id=True)
    local_parent_id = 'sha256:' + spec['parent_model']['sha256']
    if 'parent_model_id' in spec and spec['parent_model_id'] != local_parent_id:
        raise ValueError('parent model content-address mismatch')
    dataset = checked_file(root, spec['dataset'])
    expected = spec.get('parent_fingerprint')
    if not isinstance(expected, str) or not SHA.fullmatch(expected):
        raise ValueError('parent fingerprint missing')
    classes = spec.get('parent_classes')
    if not isinstance(classes, dict) or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip() for k, v in classes.items()):
        raise ValueError('parent classes invalid')
    budget=prepare_space_requirement(family,approved,parent,dataset)
    available=shutil.disk_usage(root).free
    event('space_checked',available_bytes=available,**budget)
    if available<budget['required_bytes']:
        raise ValueError('insufficient disk space for unpacking, conversion, and reserve')
    package = FAMILIES[family]
    assets = importlib.import_module(package + '.assets')
    core = importlib.import_module(package + '.core')
    tasks = importlib.import_module(package + '.tasks')
    api = importlib.import_module(package + '.api')
    workspace = root / 'prepared_runtime'
    workspace.mkdir(mode=0o700, exist_ok=False)  # no blind replay after interruption
    event('workspace_created')
    assets.unpack(approved, workspace / 'approved_base')
    base = assets.find_root(workspace / 'approved_base', 'config.json')
    if assets.model_fingerprint(base)[0] != spec['approved_base_fingerprint']:
        raise ValueError('approved base fingerprint mismatch')
    event('approved_base_verified')
    store = core.Store(workspace / 'store')
    assets.bootstrap_model(store, base)
    parent_id = core.identifier('asset')
    (store.root / 'assets' / parent_id).mkdir()
    model_asset = assets.register_archive(store, 'model', parent, parent_id, base)
    if model_asset['metadata']['fingerprint'] != expected:
        raise ValueError('parent weight fingerprint mismatch')
    event('parent_verified')
    model_id = model_asset['metadata']['model_version_id']
    model = store.get('models', model_id)
    model['classes'] = classes
    with store.connect() as db:
        db.execute("UPDATE objects SET data=? WHERE kind='models' AND id=?", (json.dumps(model, ensure_ascii=False), model_id))
    dataset_id = core.identifier('asset')
    (store.root / 'assets' / dataset_id).mkdir()
    assets.register_archive(store, 'dataset', dataset, dataset_id, base)
    event('dataset_registered')
    request = dict(spec['training'], model_version_id=model_id, dataset_asset_id=dataset_id,
                   training_mode='new_only', replay_asset_id=None, replay_class_prompts=[])
    request = api.TrainingRequest.model_validate(request).model_dump()
    job = tasks.create_job(store, request, idem=spec['request_id'])
    result = {'schema': 1, 'phase': 'prepared_only', 'family': family,
              'request_id': spec['request_id'], 'parent_fingerprint': expected,
              'bundle_manifest_sha256': expected_manifest_sha, 'remote_parent_model_id': model_id,
              'job_id': job['id'], 'job_state': job['state'], 'quality_status': 'not_trained',
              'note': 'GPU execution deliberately disabled until resource coordination is verified'}
    out = root / 'outputs'; out.mkdir(exist_ok=False)
    with (out/'prepared.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    return result


def prepare(root=ROOT, expected_manifest_sha=None, expected_mode="prepare_only", requested_mode=None):
    root=Path(root).resolve(strict=True)
    bundle_mode.enforce(root, expected_mode, requested_mode)
    # Exclusive journal creation prevents automatic retries after a crash,
    # failure, or success. A human must inspect and explicitly create a new run.
    try: stream=(root/'prepare.events.jsonl').open('x',encoding='utf-8')
    except FileExistsError:raise FileExistsError('preparation already attempted; inspect prepare.events.jsonl before creating a new run') from None
    with stream:
        def event(kind,**detail):
            stream.write(json.dumps({'event':kind,**detail},ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        event('started')
        try:
            result=_prepare(root,expected_manifest_sha,event)
            event('complete',job_id=result['job_id'],phase=result['phase'])
            return result
        except BaseException as exc:
            event('failed',error_type=type(exc).__name__,detail=str(exc))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256',required=True,help='sender receipt SHA-256; independently verify prepare.py before execution')
    parser.add_argument('--bundle-mode',choices=sorted(bundle_mode.MODES),default=None,
                        help='assert the sealed bundle mode; mismatch is rejected')
    args=parser.parse_args()
    result = prepare(expected_manifest_sha=args.manifest_sha256,requested_mode=args.bundle_mode)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
