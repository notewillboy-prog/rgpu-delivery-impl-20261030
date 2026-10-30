"""A failed resume must never silently submit a second remote job."""
from types import SimpleNamespace
import pytest
from common import protocol
from common.models import JobSpec
from client.connection import ConnectionLost, RemoteError
from client.runner import JobRunner


class FakeConnection:
    def __init__(self, response):
        self.config = SimpleNamespace(chunk_size=1024)
        self.max_frame = 1024
        self.response = response
        self.messages = []

    def send_json(self, message):
        self.messages.append(message)

    def wait_for(self, *_args, **_kwargs):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def make_runner(response):
    connection = FakeConnection(response)
    runner = JobRunner(connection, JobSpec(job_name='test', entry_argv=['python', 'task.py']), '.')
    return runner, connection


@pytest.mark.parametrize('code', [protocol.ERR_NOT_FOUND, protocol.ERR_CONFLICT])
def test_failed_resume_preserves_id_and_never_creates_new_job(code):
    runner, conn = make_runner(RemoteError(code, 'unknown'))
    runner.job_id = 'known-job'
    with pytest.raises(RemoteError, match='人工核查'):
        runner._resubmit_for_resume()
    assert runner.job_id == 'known-job'
    assert len(conn.messages) == 1
    assert conn.messages[0]['resume_job_id'] == 'known-job'


def test_mismatched_accepted_id_is_rejected_without_losing_original():
    runner, conn = make_runner({'job_id':'different-job', 'missing':[], 'present':[]})
    runner.job_id = 'known-job'
    with pytest.raises(RemoteError, match='人工核查'):
        runner._resubmit_for_resume()
    assert runner.job_id == 'known-job'
    assert len(conn.messages) == 1


def test_unknown_accepted_does_not_retry_new_job():
    runner, conn = make_runner(ConnectionLost('accepted response lost'))
    result = runner.run()
    assert not result.ok and result.job_id == ''
    assert len(conn.messages) == 1
    assert conn.messages[0]['resume_job_id'] is None


def test_valid_resume_keeps_original_job():
    runner, conn = make_runner({'job_id':'known-job', 'missing':[], 'present':[]})
    runner.job_id = 'known-job'
    assert runner._resubmit_for_resume()['job_id'] == 'known-job'
    assert len(conn.messages) == 1
