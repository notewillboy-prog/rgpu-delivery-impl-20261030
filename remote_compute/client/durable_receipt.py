"""Local durable send intent; this module never submits or contacts a server.

An INTENT means submission may or may not have happened. Only a durably
recorded accepted job ID makes it safe for a caller to query that exact job.
The caller must not infer that an unknown outcome permits another submission.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit

_ID = re.compile(r'[A-Za-z0-9_-]{1,200}\Z')
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_STATES = {'accepted': 0, 'tracking': 1, 'succeeded': 2, 'failed': 2, 'cancelled': 2}
_TERMINAL = {'succeeded', 'failed', 'cancelled'}
_RESERVED = re.compile(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\Z', re.I)


class ReceiptConflict(ValueError):
    """The proposed mutation conflicts with an already persisted identity."""


def _identity(request_id, endpoint, spec_sha256):
    if (not isinstance(request_id, str) or not _ID.fullmatch(request_id)
            or _RESERVED.fullmatch(request_id)):
        raise ValueError('invalid request id')
    if not isinstance(spec_sha256, str) or not _SHA.fullmatch(spec_sha256):
        raise ValueError('invalid spec fingerprint')
    if not isinstance(endpoint, str) or len(endpoint) > 500:
        raise ValueError('invalid endpoint')
    try:
        url = urlsplit(endpoint)
        if (url.scheme not in ('tcp', 'https') or not url.hostname or not url.port
                or url.username is not None or url.password is not None
                or url.path not in ('', '/') or url.query or url.fragment):
            raise ValueError()
    except ValueError:
        raise ValueError('endpoint must be tcp/https host and port without credentials') from None
    return {'request_id': request_id, 'endpoint': endpoint, 'spec_sha256': spec_sha256}


def _sync_dir(directory):
    # POSIX rename durability requires syncing the parent directory. Windows
    # does not permit opening directories this way; file fsync still applies.
    if os.name != 'nt':
        fd = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


@contextmanager
def _lock(path, timeout=10):
    if path.is_symlink():
        raise ValueError('receipt lock cannot be a symlink')
    flags = os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, 'r+b') as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b'\0'); stream.flush(); os.fsync(stream.fileno())
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('receipt lock timed out') from None
                time.sleep(.05)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class DurableReceipt:
    """One file per logical request, with immutable ID/endpoint/spec bindings."""

    def __init__(self, path):
        self.path = Path(path).absolute()
        self.lock_path = self.path.with_name(self.path.name + '.lock')
        if not self.path.parent.is_dir():
            raise ValueError('receipt directory must already exist')

    @classmethod
    def for_request(cls, directory, request_id):
        """Resolve one predictable file per request within a trusted directory."""
        if (not isinstance(request_id, str) or not _ID.fullmatch(request_id)
                or _RESERVED.fullmatch(request_id)):
            raise ValueError('invalid request id')
        root = Path(directory)
        if root.is_symlink() or not root.is_dir():
            raise ValueError('receipt directory must be an existing trusted directory')
        return cls(root / (request_id + '.json'))

    def _read(self):
        if self.path.is_symlink() or not self.path.is_file():
            raise FileNotFoundError('receipt is absent or unsafe')
        value = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(value, dict) or set(value) != {'schema', 'request_id', 'endpoint', 'spec_sha256', 'job_id', 'state'} or value['schema'] != 1:
            raise ValueError('receipt schema invalid')
        _identity(value['request_id'], value['endpoint'], value['spec_sha256'])
        if value['job_id'] is None:
            if value['state'] != 'intent': raise ValueError('receipt state invalid')
        elif (not isinstance(value['job_id'], str) or not _ID.fullmatch(value['job_id'])
              or value['state'] not in _STATES):
            raise ValueError('receipt job or state invalid')
        return value

    def _write(self, value):
        if self.path.is_symlink():
            raise ValueError('receipt cannot be a symlink')
        temp = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                             prefix='.' + self.path.name + '.', suffix='.tmp', delete=False) as f:
                temp = Path(f.name)
                json.dump(value, f, ensure_ascii=False, sort_keys=True)
                f.write('\n'); f.flush(); os.fsync(f.fileno())
            os.replace(temp, self.path)
            temp = None
            _sync_dir(self.path.parent)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    def create_intent(self, request_id, endpoint, spec_sha256):
        """Idempotent metadata creation; never use its return to authorize send.

        Use begin_once with for_request for the one allowed first submission.
        """
        identity = _identity(request_id, endpoint, spec_sha256)
        with _lock(self.lock_path):
            if self.path.exists() or self.path.is_symlink():
                current = self._read()
                if any(current[k] != v for k, v in identity.items()):
                    raise ReceiptConflict('request identity conflicts with existing receipt')
                return current
            value = {'schema': 1, **identity, 'job_id': None, 'state': 'intent'}
            self._write(value)
            return value

    def begin_once(self, request_id, endpoint, spec_sha256):
        """Create the intent atomically; only the creator may attempt one send.

        False means an existing matching intent or job and forbids new send,
        even if the previous process may have died before transmitting. A
        caller must use one trusted common directory for all requests.
        """
        identity = _identity(request_id, endpoint, spec_sha256)
        if self.path.name != request_id + '.json':
            raise ValueError('begin_once requires a request-derived receipt path')
        with _lock(self.lock_path):
            if self.path.exists() or self.path.is_symlink():
                current = self._read()
                if any(current[k] != v for k, v in identity.items()):
                    raise ReceiptConflict('request identity conflicts with existing receipt')
                return False
            self._write({'schema': 1, **identity, 'job_id': None, 'state': 'intent'})
            return True

    def record_accepted(self, job_id):
        if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
            raise ValueError('invalid accepted job id')
        with _lock(self.lock_path):
            current = self._read()
            if current['job_id'] is not None:
                if current['job_id'] != job_id:
                    raise ReceiptConflict('accepted job id conflicts with persisted job')
                return current
            current.update(job_id=job_id, state='accepted')
            self._write(current)
            return current

    def update_state(self, job_id, state):
        if state not in _STATES:
            raise ValueError('invalid receipt state')
        with _lock(self.lock_path):
            current = self._read()
            if current['job_id'] is None or current['job_id'] != job_id:
                raise ReceiptConflict('state update does not match accepted job id')
            old = current['state']
            if old == state:
                return current
            if old in _TERMINAL or _STATES[state] <= _STATES[old]:
                raise ReceiptConflict('receipt state cannot move backwards or change terminal result')
            current['state'] = state
            self._write(current)
            return current

    def recovery(self):
        """Return query-only plan or unknown outcome; no network side effects."""
        with _lock(self.lock_path):
            current = self._read()
            return {'receipt': current, 'action': 'query_existing' if current['job_id'] else 'manual_unknown',
                    'automatic_resubmit': False}
