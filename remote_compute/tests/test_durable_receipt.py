"""Durability and conflict tests; no networking or real credentials."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from client.durable_receipt import DurableReceipt, ReceiptConflict


SHA = 'a' * 64
ENDPOINT = 'tcp://127.0.0.1:12345'


def receipt(tmp_path):
    return DurableReceipt(tmp_path / 'receipt.json')


def test_intent_and_state_are_idempotent_and_immutable(tmp_path):
    r = receipt(tmp_path)
    assert r.create_intent('request-1', ENDPOINT, SHA)['state'] == 'intent'
    assert r.recovery()['action'] == 'manual_unknown'
    assert not r.recovery()['automatic_resubmit']
    r.record_accepted('job-1')
    assert r.create_intent('request-1', ENDPOINT, SHA)['job_id'] == 'job-1'
    assert r.record_accepted('job-1')['state'] == 'accepted'
    r.update_state('job-1', 'tracking')
    assert r.record_accepted('job-1')['state'] == 'tracking'
    r.update_state('job-1', 'succeeded')
    assert r.update_state('job-1', 'succeeded')['state'] == 'succeeded'
    assert r.recovery()['action'] == 'query_existing'
    for action in (lambda: r.record_accepted('job-2'),
                   lambda: r.update_state('job-1', 'failed'),
                   lambda: r.update_state('job-2', 'succeeded'),
                   lambda: r.create_intent('request-2', ENDPOINT, SHA),
                   lambda: r.create_intent('request-1', 'tcp://127.0.0.1:2', SHA),
                   lambda: r.create_intent('request-1', ENDPOINT, 'b'*64)):
        with pytest.raises(ReceiptConflict): action()


@pytest.mark.parametrize('bad', ['tcp://user:pw@host:12', 'tcp://host:12/?token=secret',
                                 'tcp://host:12/path', 'tcp://host', 'http://host:1'])
def test_no_credentials_or_noncanonical_endpoint_in_receipt(tmp_path, bad):
    with pytest.raises(ValueError): receipt(tmp_path).create_intent('request', bad, SHA)
    assert not (tmp_path / 'receipt.json').exists()


def test_failed_atomic_replace_keeps_persisted_intent(tmp_path, monkeypatch):
    r = receipt(tmp_path)
    r.create_intent('request', ENDPOINT, SHA)
    before = r.path.read_bytes()
    def fail_replace(*_): raise OSError('simulated interruption before replace')
    monkeypatch.setattr(os, 'replace', fail_replace)
    with pytest.raises(OSError): r.record_accepted('job-1')
    assert r.path.read_bytes() == before
    assert r.recovery()['action'] == 'manual_unknown'
    assert not list(tmp_path.glob('*.tmp'))


def test_process_crash_after_fsync_recovers_persisted_job(tmp_path):
    path = tmp_path / 'receipt.json'
    script = '''from client.durable_receipt import DurableReceipt\nimport os,sys\nr=DurableReceipt(sys.argv[1]);r.create_intent("request", "tcp://localhost:1234", "a"*64)\nr.record_accepted("job-fixed");os._exit(17)\n'''
    result = subprocess.run([sys.executable, '-c', script, str(path)], cwd=Path(__file__).resolve().parents[1], timeout=15)
    assert result.returncode == 17
    plan = DurableReceipt(path).recovery()
    assert plan['action'] == 'query_existing' and plan['receipt']['job_id'] == 'job-fixed'
    assert plan['automatic_resubmit'] is False


def test_process_crash_after_intent_never_infers_resubmit(tmp_path):
    path = tmp_path / 'receipt.json'
    script = '''from client.durable_receipt import DurableReceipt\nimport os,sys\nDurableReceipt(sys.argv[1]).create_intent("request", "tcp://localhost:1234", "a"*64)\nos._exit(18)\n'''
    result = subprocess.run([sys.executable, '-c', script, str(path)], cwd=Path(__file__).resolve().parents[1], timeout=15)
    assert result.returncode == 18
    plan = DurableReceipt(path).recovery()
    assert plan['action'] == 'manual_unknown' and plan['receipt']['job_id'] is None
    assert plan['automatic_resubmit'] is False


def test_concurrent_distinct_job_ids_cannot_both_be_accepted(tmp_path):
    path = tmp_path / 'receipt.json'
    DurableReceipt(path).create_intent('request', ENDPOINT, SHA)
    gate = tmp_path / 'go'
    script = '''from client.durable_receipt import DurableReceipt,ReceiptConflict\nimport sys,time\nfrom pathlib import Path\nwhile not Path(sys.argv[2]).exists():time.sleep(.01)\ntry:DurableReceipt(sys.argv[1]).record_accepted(sys.argv[3]);print("accepted")\nexcept ReceiptConflict:print("conflict")\n'''
    cwd = Path(__file__).resolve().parents[1]
    procs = [subprocess.Popen([sys.executable, '-c', script, str(path), str(gate), job], cwd=cwd,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for job in ('job-one', 'job-two')]
    gate.touch()
    outcomes = []
    try:
        for proc in procs:
            out, err = proc.communicate(timeout=20)
            assert proc.returncode == 0, err
            outcomes.append(out.strip())
    finally:
        for proc in procs:
            if proc.poll() is None: proc.kill(); proc.wait()
    assert sorted(outcomes) == ['accepted', 'conflict']
    assert DurableReceipt(path).recovery()['receipt']['job_id'] in ('job-one', 'job-two')


def test_concurrent_identical_job_id_is_idempotent(tmp_path):
    path = tmp_path / 'receipt.json'
    DurableReceipt(path).create_intent('request', ENDPOINT, SHA)
    gate = tmp_path / 'go'
    script = '''from client.durable_receipt import DurableReceipt\nimport sys,time\nfrom pathlib import Path\nwhile not Path(sys.argv[2]).exists():time.sleep(.01)\nprint(DurableReceipt(sys.argv[1]).record_accepted("job-same")["job_id"])\n'''
    cwd = Path(__file__).resolve().parents[1]
    procs = [subprocess.Popen([sys.executable, '-c', script, str(path), str(gate)], cwd=cwd,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    gate.touch()
    try:
        for proc in procs:
            out, err = proc.communicate(timeout=20)
            assert proc.returncode == 0, err
            assert out.strip() == 'job-same'
    finally:
        for proc in procs:
            if proc.poll() is None: proc.kill(); proc.wait()


def test_begin_once_rejects_paths_not_derived_from_request(tmp_path):
    with pytest.raises(ValueError, match='request-derived'):
        receipt(tmp_path).begin_once('request', ENDPOINT, SHA)
    for bad in ('../request', 'CON', 'nul', 'com1', 'Lpt9', 'bad/name'):
        with pytest.raises(ValueError): DurableReceipt.for_request(tmp_path, bad)
    r = DurableReceipt.for_request(tmp_path, 'request')
    assert r.begin_once('request', ENDPOINT, SHA) is True
    assert r.begin_once('request', ENDPOINT, SHA) is False
    with pytest.raises(ReceiptConflict): r.begin_once('request', ENDPOINT, 'b'*64)


def test_begin_once_concurrent_processes_have_exactly_one_winner(tmp_path):
    gate = tmp_path / 'go'
    script = '''from client.durable_receipt import DurableReceipt\nimport sys,time\nfrom pathlib import Path\nwhile not Path(sys.argv[2]).exists():time.sleep(.01)\nr=DurableReceipt.for_request(sys.argv[1],"request");print(r.begin_once("request","tcp://localhost:1234","a"*64))\n'''
    cwd = Path(__file__).resolve().parents[1]
    procs = [subprocess.Popen([sys.executable, '-c', script, str(tmp_path), str(gate)], cwd=cwd,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    gate.touch()
    results = []
    try:
        for proc in procs:
            out, err = proc.communicate(timeout=20)
            assert proc.returncode == 0, err
            results.append(out.strip())
    finally:
        for proc in procs:
            if proc.poll() is None: proc.kill(); proc.wait()
    assert sorted(results) == ['False', 'True']
    assert DurableReceipt.for_request(tmp_path, 'request').recovery()['action'] == 'manual_unknown'


def test_begin_once_crash_after_winning_never_allows_resubmit(tmp_path):
    script = '''from client.durable_receipt import DurableReceipt\nimport os,sys\nr=DurableReceipt.for_request(sys.argv[1],"request")\nassert r.begin_once("request","tcp://localhost:1234","a"*64)\nos._exit(19)\n'''
    cwd = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=cwd, timeout=15)
    assert child.returncode == 19
    r = DurableReceipt.for_request(tmp_path, 'request')
    assert r.begin_once('request', 'tcp://localhost:1234', SHA) is False
    assert r.recovery()['action'] == 'manual_unknown'


def test_begin_once_failed_write_does_not_report_winner(tmp_path, monkeypatch):
    r = DurableReceipt.for_request(tmp_path, 'request')
    def fail(*_): raise OSError('disk error')
    monkeypatch.setattr(os, 'replace', fail)
    with pytest.raises(OSError): r.begin_once('request', ENDPOINT, SHA)
    assert not r.path.exists()
