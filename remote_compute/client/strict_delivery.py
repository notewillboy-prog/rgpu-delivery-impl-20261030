"""Opt-in one-shot delivery with durable intent and query-only recovery.

This module does not supply a production SSH/TLS verifier. A trusted caller
must provide a verifier that authenticates the actual live channel and stable
server identity. Without one, every entry fails before network activity.
The caller must also use the same protected receipt directory for every run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re

from common import protocol
from common.models import JobSpec, validate_job_spec
from .durable_receipt import DurableReceipt, ReceiptConflict
from .uploader import upload_file, verify_local_entries

_SHA = re.compile(r'[0-9a-f]{64}\Z')
_KNOWN_STATES = {'uploading', 'queued', 'env_building', 'running',
                 'succeeded', 'failed', 'timeout', 'cancelled'}
_META_KEY = '_strict_delivery'


@dataclass(frozen=True)
class VerifiedTransport:
    """Evidence returned only by an independently trusted transport verifier.

    The verifier, not this data class, is responsible for validating SSH/TLS
    and binding that proof to the very same live connection. A self-reported
    hello hostname or server_id alone is not evidence of authentication.
    """
    connection: object
    endpoint: str
    identity_sha256: str
    authenticated_transport: bool
    target_bound: bool


class DeliveryStopped(RuntimeError):
    def __init__(self, stage, *, job_id=None):
        super().__init__(f'严格投送在 {stage} 停止；请检查原任务，禁止自动重投。')
        self.stage = stage
        # A seen ID may not have been durably persisted; it is only a clue.
        self.observed_job_id = job_id


def _canonical_fingerprint(spec):
    data = spec.to_dict()
    data['inputs'] = sorted(data.get('inputs') or [], key=lambda item: str(item.get('path')))
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _bound_spec(spec, request_id, identity_sha256):
    if not isinstance(identity_sha256, str) or not _SHA.fullmatch(identity_sha256):
        raise ValueError('trusted server identity must be a SHA-256')
    # Copy and normalize before validation so the submitted spec cannot alias a
    # caller-owned mutable object. The identity binding is covered by spec SHA.
    data = json.loads(json.dumps(spec.to_dict(), ensure_ascii=False))
    meta = data.setdefault('meta', {})
    if _META_KEY in meta:
        raise ValueError('reserved strict delivery metadata already exists')
    meta[_META_KEY] = {'request_id': request_id, 'server_identity_sha256': identity_sha256}
    result = JobSpec.from_dict(data)
    validate_job_spec(result)
    return result


def _verify_transport(connection, endpoint, identity_sha256, verifier):
    if not callable(verifier):
        raise ValueError('trusted authenticated transport verifier is required')
    evidence = verifier(connection, endpoint, identity_sha256)
    if (not isinstance(evidence, VerifiedTransport) or evidence.connection is not connection
            or evidence.endpoint != endpoint or evidence.identity_sha256 != identity_sha256
            or evidence.authenticated_transport is not True or evidence.target_bound is not True):
        raise ValueError('server identity or channel binding was not verified')


def _missing(spec, accepted):
    """Reject ambiguous or altered server file lists before starting upload."""
    index = spec.input_index()
    missing, present = accepted.get('missing'), accepted.get('present')
    if not isinstance(missing, list) or not isinstance(present, list):
        raise ValueError('missing/present lists are required')
    missing_paths, present_paths = [], []
    for item in missing:
        name = item.get('path') if isinstance(item, dict) else item
        if not isinstance(name, str) or name not in index:
            raise ValueError('unknown missing input')
        if isinstance(item, dict):
            if item.get('size') != index[name].size or item.get('sha256') != index[name].sha256:
                raise ValueError('server input identity mismatch')
        missing_paths.append(name)
    for item in present:
        name = item.get('path') if isinstance(item, dict) else item
        if not isinstance(name, str) or name not in index:
            raise ValueError('unknown present input')
        present_paths.append(name)
    all_paths = missing_paths + present_paths
    if len(set(all_paths)) != len(all_paths) or set(all_paths) != set(index):
        raise ValueError('server input list is incomplete or duplicated')
    return [index[name] for name in missing_paths]


class StrictDelivery:
    def __init__(self, connection, receipt_root, request_id, endpoint,
                 server_identity_sha256, spec, project_dir, *, verifier=None):
        self.connection = connection
        self.endpoint = endpoint
        self.identity = server_identity_sha256
        self.verifier = verifier
        self.request_id = request_id
        self.receipt = DurableReceipt.for_request(receipt_root, request_id)
        self.spec = _bound_spec(spec, request_id, server_identity_sha256)
        self.fingerprint = _canonical_fingerprint(self.spec)
        self.project_dir = project_dir

    def _verify(self):
        _verify_transport(self.connection, self.endpoint, self.identity, self.verifier)

    def _unchanged(self):
        if _canonical_fingerprint(self.spec) != self.fingerprint:
            raise ValueError('job spec changed after binding')

    def submit_once(self):
        """Submit once, upload once, start once; never reconnect or resubmit."""
        self._verify()
        self._unchanged()
        problems = verify_local_entries(self.spec.inputs, self.project_dir)
        if problems:
            raise ValueError('input files changed or unavailable')
        if not self.receipt.begin_once(self.request_id, self.endpoint, self.fingerprint):
            raise ReceiptConflict('existing receipt forbids any new submission')
        observed = None
        stage = 'submit_unknown'
        try:
            self.connection.send_json({'kind': protocol.KIND_JOB_SUBMIT,
                                       'spec': self.spec.to_dict(), 'resume_job_id': None})
            accepted = self.connection.wait_for([protocol.KIND_JOB_ACCEPTED])
            observed = accepted.get('job_id')
            stage = 'accepted_not_durable'
            self.receipt.record_accepted(observed)
            stage = 'upload_or_start_unknown'
            self._unchanged()
            missing = _missing(self.spec, accepted)
            for entry in missing:
                upload_file(self.connection, entry, self.project_dir,
                            max_chunk_retries=0, max_file_retries=0)
            self._unchanged()
            self.receipt.update_state(observed, 'tracking')
            self.connection.send_json({'kind': protocol.KIND_JOB_START, 'job_id': observed})
            response = self.connection.wait_for([protocol.KIND_JOB_STARTED, protocol.KIND_JOB_FINISHED])
            if response.get('job_id') != observed:
                raise ValueError('server returned a different job ID')
            # No output or model-quality claim is made here.
            return {'job_id': observed, 'phase': 'start_acknowledged',
                    'remote_message': response, 'quality_status': 'needs_review'}
        except Exception as exc:
            raise DeliveryStopped(stage, job_id=observed) from exc

    def query_existing(self):
        """Only read persisted ID and query state; never submit/start/upload."""
        self._verify()
        self._unchanged()
        plan = self.receipt.recovery()
        r = plan['receipt']
        if (r['request_id'] != self.request_id or r['endpoint'] != self.endpoint
                or r['spec_sha256'] != self.fingerprint):
            raise ReceiptConflict('receipt and verified target/spec do not match')
        if plan['action'] != 'query_existing':
            raise DeliveryStopped('accepted_id_unknown')
        job_id = r['job_id']
        try:
            self.connection.send_json({'kind': protocol.KIND_JOB_STATUS, 'job_id': job_id})
            status = self.connection.wait_for([protocol.KIND_JOB_STATUS_OK])
            if status.get('job_id') != job_id or status.get('state') not in _KNOWN_STATES:
                raise ValueError('unverified status response')
            return {'job_id': job_id, 'remote_status': status,
                    'quality_status': 'needs_review', 'automatic_resubmit': False}
        except Exception as exc:
            raise DeliveryStopped('status_unknown', job_id=job_id) from exc
