"""Strict delivery contract using only injected in-memory channels."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers
from client import packager
from client import strict_delivery as mod
from client.durable_receipt import ReceiptConflict
from server.jobstore import canonical_spec_fingerprint


ENDPOINT = 'tcp://localhost:23456'
IDENTITY = 'b' * 64


class FakeConnection:
    def __init__(self, events, *, accepted=None, status=None, error_kind=None):
        self.events = events
        self.accepted = accepted
        self.status = status
        self.error_kind = error_kind

    def send_json(self, message):
        self.events.append(('send', message['kind']))
        if self.error_kind == 'send_' + message['kind']:
            raise OSError('simulated unknown send outcome')

    def wait_for(self, kinds):
        self.events.append(('wait', kinds[0]))
        if self.error_kind == 'wait_' + kinds[0]:
            raise OSError('simulated lost response')
        if kinds[0] == 'job_accepted': return self.accepted
        if kinds[0] == 'job_status_ok': return self.status
        return {'kind':'job_started', 'job_id':'job-one'}


def verified(conn, endpoint, identity):
    return mod.VerifiedTransport(conn, endpoint, identity, True, True)


@pytest.fixture
def case(tmp_path):
    project = helpers.copy_example_project(tmp_path / 'project')
    root = tmp_path / 'receipts'; root.mkdir()
    spec = packager.make_job_spec(project, ['python', 'train.py'], timeout_s=120)
    events = []
    accepted = {'kind':'job_accepted', 'job_id':'job-one',
                'missing':[e.to_dict() for e in spec.inputs], 'present':[]}
    conn = FakeConnection(events, accepted=accepted)
    delivery = mod.StrictDelivery(conn, root, 'request-1', ENDPOINT, IDENTITY,
                                  spec, project, verifier=verified)
    return delivery, conn, events


def test_intent_then_accepted_fsync_before_upload_and_start(case, monkeypatch):
    d, conn, events = case
    original_send = conn.send_json
    def send(message):
        state = d.receipt.recovery()['receipt']
        if message['kind'] == 'job_submit': assert state['state'] == 'intent' and state['job_id'] is None
        if message['kind'] == 'job_start': assert state['state'] == 'tracking' and state['job_id'] == 'job-one'
        original_send(message)
    conn.send_json = send
    def upload(_conn, entry, _path, **kwargs):
        state = d.receipt.recovery()['receipt']
        assert state['job_id'] == 'job-one' and state['state'] == 'accepted'
        assert kwargs['max_chunk_retries'] == kwargs['max_file_retries'] == 0
        events.append(('upload',entry.path))
    monkeypatch.setattr(mod, 'upload_file', upload)
    assert d.fingerprint == canonical_spec_fingerprint(d.spec)
    assert d.submit_once()['phase'] == 'start_acknowledged'
    assert [x for x in events if x == ('send','job_submit')] == [('send','job_submit')]
    assert events.index(('send','job_submit')) < next(i for i,x in enumerate(events) if x[0] == 'upload') < events.index(('send','job_start'))


@pytest.mark.parametrize('error', ['send_job_submit', 'wait_job_accepted'])
def test_submit_uncertain_never_resubmits_or_queries_unknown_id(case, error):
    d, conn, events = case; conn.error_kind = error
    with pytest.raises(mod.DeliveryStopped) as exc: d.submit_once()
    assert exc.value.stage == 'submit_unknown'
    assert d.receipt.recovery()['action'] == 'manual_unknown'
    with pytest.raises(ReceiptConflict): d.submit_once()
    with pytest.raises(mod.DeliveryStopped): d.query_existing()
    assert events.count(('send','job_submit')) == 1
    assert ('send','job_start') not in events
    assert ('send','job_status') not in events


def test_accepted_persistence_failure_stops_before_upload(case, monkeypatch):
    d, conn, events = case
    def failed(_job): raise OSError('disk full')
    monkeypatch.setattr(d.receipt, 'record_accepted', failed)
    monkeypatch.setattr(mod, 'upload_file', lambda *_a, **_k: pytest.fail('must not upload'))
    with pytest.raises(mod.DeliveryStopped) as exc: d.submit_once()
    assert exc.value.stage == 'accepted_not_durable'
    assert exc.value.observed_job_id == 'job-one'
    assert d.receipt.recovery()['action'] == 'manual_unknown'
    assert ('send','job_start') not in events
    with pytest.raises(ReceiptConflict): d.submit_once()


def test_reject_or_failed_upload_never_restarts(case, monkeypatch):
    d, conn, events = case
    def fail(*_a, **_kw): raise OSError('upload disconnected')
    monkeypatch.setattr(mod, 'upload_file', fail)
    with pytest.raises(mod.DeliveryStopped): d.submit_once()
    assert d.receipt.recovery()['receipt']['job_id'] == 'job-one'
    assert ('send','job_start') not in events
    conn.status = {'kind':'job_status_ok','job_id':'job-one','state':'uploading'}
    assert d.query_existing()['remote_status']['state'] == 'uploading'
    assert [x[1] for x in events if x[0] == 'send'] == ['job_submit','job_status']


def test_server_rejection_is_not_retried(case):
    from client.connection import RemoteError
    d, conn, events = case
    def rejected(_kinds): raise RemoteError('BUSY', 'simulated rejection')
    conn.wait_for = rejected
    with pytest.raises(mod.DeliveryStopped): d.submit_once()
    assert events == [('send','job_submit')]
    assert d.receipt.recovery()['action'] == 'manual_unknown'
    with pytest.raises(ReceiptConflict): d.submit_once()
    assert events == [('send','job_submit')]


def test_tracking_persist_failure_prevents_start(case, monkeypatch):
    d, conn, events = case
    monkeypatch.setattr(mod, 'upload_file', lambda *_a, **_kw: None)
    def fail(*_a): raise OSError('sync failed')
    monkeypatch.setattr(d.receipt, 'update_state', fail)
    with pytest.raises(mod.DeliveryStopped): d.submit_once()
    assert d.receipt.recovery()['receipt']['job_id'] == 'job-one'
    assert ('send','job_start') not in events


def test_start_response_lost_never_sends_another_start(case, monkeypatch):
    d, conn, events = case
    monkeypatch.setattr(mod, 'upload_file', lambda *_a, **_kw: None)
    conn.error_kind = 'wait_job_started'
    with pytest.raises(mod.DeliveryStopped): d.submit_once()
    conn.error_kind = None
    conn.status = {'kind':'job_status_ok','job_id':'job-one','state':'running'}
    assert d.query_existing()['remote_status']['state'] == 'running'
    assert events.count(('send','job_start')) == 1
    assert events.count(('send','job_submit')) == 1


def test_server_missing_input_identity_mismatch_never_uploads(case, monkeypatch):
    d, conn, events = case
    conn.accepted['missing'][0]['sha256'] = 'f'*64
    monkeypatch.setattr(mod, 'upload_file', lambda *_a, **_kw: pytest.fail('must not upload'))
    with pytest.raises(mod.DeliveryStopped): d.submit_once()
    assert d.receipt.recovery()['receipt']['job_id'] == 'job-one'
    assert ('send','job_start') not in events


def test_no_trusted_identity_or_channel_fails_before_receipt_and_send(case):
    d, conn, events = case
    d.verifier = None
    with pytest.raises(ValueError): d.submit_once()
    assert not d.receipt.path.exists() and not events
    for evidence in (mod.VerifiedTransport(conn, ENDPOINT, IDENTITY, False, True),
                     mod.VerifiedTransport(conn, ENDPOINT, 'c'*64, True, True),
                     mod.VerifiedTransport(object(), ENDPOINT, IDENTITY, True, True)):
        d.verifier = lambda *_a, e=evidence: e
        with pytest.raises(ValueError): d.submit_once()
        assert not d.receipt.path.exists() and not events


def test_recovery_validates_identity_spec_and_response_is_query_only(case):
    d, conn, events = case
    d.receipt.begin_once(d.request_id, ENDPOINT, d.fingerprint)
    d.receipt.record_accepted('job-one')
    conn.status = {'kind':'job_status_ok','job_id':'other','state':'running'}
    with pytest.raises(mod.DeliveryStopped): d.query_existing()
    assert events == [('send','job_status'),('wait','job_status_ok')]
    events.clear()
    other = mod.StrictDelivery(conn, d.receipt.path.parent, d.request_id, ENDPOINT, 'c'*64,
                               # remove bound metadata by using original-like copy
                               _strip(d.spec), d.project_dir, verifier=verified)
    with pytest.raises(ReceiptConflict): other.query_existing()
    assert not events


def _strip(spec):
    from common.models import JobSpec
    data = spec.to_dict(); data['meta'].pop('_strict_delivery', None)
    return JobSpec.from_dict(data)


def test_concurrent_attempts_allow_only_one_submit(case, monkeypatch):
    d, conn, events = case
    monkeypatch.setattr(mod, 'upload_file', lambda *_a, **_kw: None)
    def run():
        try: return d.submit_once()['job_id']
        except ReceiptConflict: return 'blocked'
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda _: run(), range(2)))
    assert sorted(result) == ['blocked','job-one']
    assert events.count(('send','job_submit')) == 1
