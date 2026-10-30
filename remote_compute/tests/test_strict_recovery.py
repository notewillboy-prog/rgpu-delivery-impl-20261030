"""An unknown pre-restart process must never become an executable runtime."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import helpers
import rawflow
from client import packager
from server.jobstore import JobStore


@pytest.mark.parametrize('old_state', ['uploading', 'queued', 'env_building', 'running'])
def test_restart_unknown_nonterminal_does_not_mutate_or_resume(tmp_path, old_state):
    project = helpers.copy_example_project(tmp_path / 'project')
    spec = packager.make_job_spec(project, ['python', 'train.py'], timeout_s=120)
    data = tmp_path / 'service'
    server = helpers.ServerProcess(data_dir=data, extra_args=['--strict-recovery']).start()
    try:
        client, _ = rawflow.connect(server.address)
        accepted = rawflow.submit(client, spec)
        job_id = accepted['job_id']
        client.close()
        # A lost connection while the same server owns the job is safe.
        client, _ = rawflow.connect(server.address)
        assert rawflow.submit(client, spec, resume_job_id=job_id)['job_id'] == job_id
        client.close()
    finally:
        server.stop()
    store = JobStore(data)
    store.update_state(job_id, state=old_state)
    state_path = store.state_path(job_id)
    before = state_path.read_bytes()
    server2 = helpers.ServerProcess(data_dir=data, extra_args=['--strict-recovery']).start()
    try:
        client, _ = rawflow.connect(server2.address)
        try:
            client.send_json({'kind':'job_status','job_id':job_id})
            assert client.expect_error('CONFLICT')['message'].find('未知') >= 0
            client.send_json({'kind':'job_submit','spec':spec.to_dict(),'resume_job_id':job_id})
            assert client.expect_error('CONFLICT')['message'].find('核查') >= 0
            client.send_json({'kind':'job_start','job_id':job_id})
            assert client.expect_error('CONFLICT')['message']
        finally:
            client.close()
        assert state_path.read_bytes() == before
        assert len([p for p in store.jobs_dir.iterdir() if p.is_dir()]) == 1
    finally:
        server2.stop()


def test_terminal_state_is_still_queryable_after_restart(tmp_path):
    data = tmp_path / 'service'
    project = helpers.copy_example_project(tmp_path / 'project')
    spec = packager.make_job_spec(project, ['python', 'train.py'], timeout_s=120)
    server = helpers.ServerProcess(data_dir=data, extra_args=['--strict-recovery']).start()
    try:
        client, _ = rawflow.connect(server.address)
        job_id = rawflow.submit(client, spec)['job_id']
        client.close()
    finally:
        server.stop()
    store = JobStore(data)
    store.update_state(job_id, state='succeeded', exit_code=0, outputs=[])
    before = store.state_path(job_id).read_bytes()
    server2 = helpers.ServerProcess(data_dir=data, extra_args=['--strict-recovery']).start()
    try:
        client, _ = rawflow.connect(server2.address)
        try:
            assert rawflow.query_status(client, job_id)['state'] == 'succeeded'
            client.send_json({'kind':'output_list','job_id':job_id})
            assert client.recv_json()['kind'] == 'output_list_ok'
        finally:
            client.close()
        assert store.state_path(job_id).read_bytes() == before
    finally:
        server2.stop()


def test_live_process_restart_refuses_reattach_without_changing_disk(tmp_path):
    data = tmp_path / 'service'
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'train.py').write_text(
        'import time\nprint("WORK_STARTED", flush=True)\ntime.sleep(3)\n', encoding='utf-8')
    spec = packager.make_job_spec(project, ['python', 'train.py'], timeout_s=20)
    server = helpers.ServerProcess(data_dir=data, extra_args=['--strict-recovery']).start()
    try:
        client, _ = rawflow.connect(server.address)
        job_id = rawflow.submit(client, spec)['job_id']
        for entry in spec.inputs:
            assert rawflow.upload_entry(client, entry, project).get('done')
        client.send_json({'kind':'job_start','job_id':job_id})
        while True:
            msg = client.recv_json()
            assert msg.get('kind') not in ('error', 'job_finished'), msg
            if msg.get('kind') == 'job_event' and 'WORK_STARTED' in msg.get('data', ''):
                break
        client.close()
    finally:
        server.stop()
    store = JobStore(data)
    before = store.state_path(job_id).read_bytes()
    assert store.load_state(job_id)['state'] == 'running'
    server2 = helpers.ServerProcess(data_dir=data, extra_args=['--strict-recovery']).start()
    try:
        client, _ = rawflow.connect(server2.address)
        try:
            client.send_json({'kind':'job_submit','spec':spec.to_dict(),'resume_job_id':job_id})
            assert client.expect_error('CONFLICT')
            client.send_json({'kind':'job_status','job_id':job_id})
            assert client.expect_error('CONFLICT')
        finally:
            client.close()
        assert store.state_path(job_id).read_bytes() == before
    finally:
        server2.stop()
