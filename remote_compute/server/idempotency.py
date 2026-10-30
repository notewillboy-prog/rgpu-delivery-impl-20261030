"""File-backed idempotency index for ``job_submit``.

One JSON record per key::

    <data_dir>/idempotency/<sha256(key)>.json
    {"key": ..., "job_id": ..., "spec_fingerprint": ..., "created_at": ...}

The index is deliberately small: it does not own job state, it only maps a
caller-supplied key to the already-created job.  ``get_or_create`` holds a
per-key lock (thread + cross-process) across lookup/create/record, so two
concurrent submissions of the same key create at most one job in one process.

Crash window: if the process dies after ``create_job`` but before ``record``,
a retry cannot see the orphaned job and will create a second one.  Keep the
window small; callers should use a durable client receipt in addition to this
index.  A future version can reserve the job id before creating the directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

_KEY_RE = re.compile(r"[\x20-\x7e]{1,200}\Z")
_JOB_RE = re.compile(r"job-[A-Za-z0-9_-]{1,100}\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class IdempotencyError(Exception):
    """Base class for index failures."""


class IdempotencyConflict(IdempotencyError):
    """The key was already used for a different spec or job."""


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


@contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "r+b", buffering=0) as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class IdempotencyIndex:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    # -- paths ------------------------------------------------------------
    @staticmethod
    def validate_key(key: str) -> str:
        if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
            raise ValueError("idempotency key must be 1..200 printable ASCII chars")
        if "/" in key or "\\" in key:
            raise ValueError("idempotency key must not contain path separators")
        return key

    def record_path(self, key: str) -> Path:
        digest = hashlib.sha256(self.validate_key(key).encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def _thread_lock(self, key: str) -> threading.RLock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                self._locks[key] = lock
            return lock

    @contextmanager
    def guard(self, key: str):
        with self._thread_lock(key):
            with _file_lock(self.root / ".locks" / f"{hashlib.sha256(key.encode()).hexdigest()}.lock"):
                yield

    # -- records ----------------------------------------------------------
    def lookup(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.record_path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise IdempotencyError(f"idempotency record is unreadable: {key}: {exc}") from exc
        if not isinstance(data, dict):
            raise IdempotencyError("idempotency record is not an object")
        if data.get("key") != key:
            raise IdempotencyError("idempotency record key mismatch")
        if not isinstance(data.get("job_id"), str) or not _JOB_RE.fullmatch(data["job_id"]):
            raise IdempotencyError("idempotency record job_id is invalid")
        if not isinstance(data.get("spec_fingerprint"), str) or not _SHA_RE.fullmatch(data["spec_fingerprint"]):
            raise IdempotencyError("idempotency record spec fingerprint is invalid")
        return data

    def record(self, key: str, job_id: str, spec_fingerprint: str) -> Dict[str, Any]:
        key = self.validate_key(key)
        if not isinstance(job_id, str) or not _JOB_RE.fullmatch(job_id):
            raise ValueError("invalid job id")
        if not isinstance(spec_fingerprint, str) or not _SHA_RE.fullmatch(spec_fingerprint):
            raise ValueError("invalid spec fingerprint")
        payload = {
            "key": key,
            "job_id": job_id,
            "spec_fingerprint": spec_fingerprint,
            "created_at": time.time(),
        }
        _atomic_write(self.record_path(key), payload)
        return payload

    def get_or_create(
        self,
        key: str,
        spec_fingerprint: str,
        create: Callable[[], str],
    ) -> Tuple[str, bool]:
        """Return ``(job_id, reused)``; create at most once per key in-process."""
        key = self.validate_key(key)
        if not isinstance(spec_fingerprint, str) or not _SHA_RE.fullmatch(spec_fingerprint):
            raise ValueError("invalid spec fingerprint")
        with self.guard(key):
            existing = self.lookup(key)
            if existing is not None:
                if existing["spec_fingerprint"] != spec_fingerprint:
                    raise IdempotencyConflict(
                        "idempotency key already used for a different spec"
                    )
                return existing["job_id"], True
            job_id = str(create())
            if not _JOB_RE.fullmatch(job_id):
                raise IdempotencyError("created job id is invalid")
            # Record while still holding the key lock.  A crash between create
            # and record leaves an orphan; documented in the module docstring.
            self.record(key, job_id, spec_fingerprint)
            return job_id, False


__all__ = [
    "IdempotencyError",
    "IdempotencyConflict",
    "IdempotencyIndex",
]
