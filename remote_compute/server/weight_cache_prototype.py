"""Local, immutable-by-convention content-addressed weight cache prototype.

Only suitable inside a private directory controlled by the service account. It
does not implement network transport, integration, eviction or GPU training.
Cross-process locks coordinate users of this module; they cannot prevent other
programs with write access from changing files. Every hit is fully rehashed.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CHUNK = 1024 * 1024


def _regular(path: Path) -> None:
    for parent in path.absolute().parents:
        if parent.is_symlink():
            raise ValueError('file path contains a symbolic link')
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError("expected an ordinary file without symbolic links")


def _private_directory(path: Path) -> Path:
    path = path.absolute()
    # Never follow symlinked parents, including pre-existing ones.
    for parent in reversed((path, *path.parents)):
        if parent.exists() or parent.is_symlink():
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("cache path contains a symlink or non-directory")
        else:
            parent.mkdir(mode=0o700, exist_ok=True)
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("cache path changed during creation")
    return path


def _identity(path: Path) -> tuple[str, int]:
    _regular(path)
    h, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(_CHUNK), b''):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def _sync_directory(path: Path) -> None:
    if os.name == 'nt':
        return  # Windows directory fsync is unavailable via os.open.
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _lock(path: Path):
    # Open without following a link on platforms which support O_NOFOLLOW.
    flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
    if path.is_symlink():
        raise ValueError('lock cannot be a symlink')
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, 'r+b', buffering=0) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('lock must be a regular file')
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b'\0')
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class WeightCache:
    """SHA-256 cache; controlled files only, no automatic removal or overwrite."""

    def __init__(self, root, *, quota_bytes: int, reserve_bytes: int = 1024**3):
        if isinstance(quota_bytes, bool) or not isinstance(quota_bytes, int) or quota_bytes <= 0:
            raise ValueError('quota must be a positive byte count')
        if isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, int) or reserve_bytes < 0:
            raise ValueError('reserve must be a nonnegative byte count')
        self.root = _private_directory(Path(root))
        self.objects = _private_directory(self.root / 'objects')
        self.locks = _private_directory(self.root / 'locks')
        self.quota, self.reserve = quota_bytes, reserve_bytes

    def _object(self, sha: str) -> Path:
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError('invalid SHA-256')
        return self.objects / sha

    def _usage(self) -> int:
        total = 0
        for path in self.objects.iterdir():
            if path.name.startswith('.incoming-'):
                _regular(path)
                # Existing partial files count against quota too.
            elif not _SHA.fullmatch(path.name):
                raise ValueError('unexpected cache object')
            _regular(path)
            total += path.stat().st_size
        return total

    def _verified(self, sha: str, size: int) -> bool:
        path = self._object(sha)
        if not path.exists() and not path.is_symlink():
            return False
        actual, length = _identity(path)
        if length != size or actual != sha:
            raise ValueError('cache object is corrupt or has an unexpected size')
        return True

    def contains(self, sha: str, size: int) -> bool:
        self._check_size(size)
        path = self._object(sha)
        with _lock(self.locks / (path.name + '.lock')):
            return self._verified(sha, size)

    @staticmethod
    def _check_size(size):
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError('size must be a nonnegative integer')

    def put(self, source, sha: str, size: int) -> bool:
        """Validate and copy the source. Return True when a new object is stored."""
        self._check_size(size)
        path = self._object(sha)
        source = Path(source)
        if _identity(source) != (sha, size):
            raise ValueError('source SHA-256 or size mismatch')
        # Global lock makes the quota check and publication atomic across cache
        # users; per-object lock also protects concurrent reads/materialization.
        with _lock(self.locks / 'quota.lock'), _lock(self.locks / (sha + '.lock')):
            if self._verified(sha, size):
                return False
            if self._usage() + size > self.quota:
                raise ValueError('cache quota insufficient')
            if shutil.disk_usage(self.objects).free < size + self.reserve:
                raise ValueError('insufficient free disk space')
            fd, temp_name = tempfile.mkstemp(prefix='.incoming-', dir=self.objects)
            temp = Path(temp_name)
            try:
                with os.fdopen(fd, 'wb') as dst, source.open('rb') as src:
                    h, written = hashlib.sha256(), 0
                    for block in iter(lambda: src.read(_CHUNK), b''):
                        written += len(block)
                        if written > size:
                            raise ValueError('source changed while copying')
                        h.update(block)
                        dst.write(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                if (h.hexdigest(), written) != (sha, size):
                    raise ValueError('source changed while copying')
                # Atomic no-overwrite publication, same filesystem; this link is
                # only to our private temporary file, not to any user file.
                os.link(temp, path, follow_symlinks=False)
                _sync_directory(self.objects)
            finally:
                temp.unlink(missing_ok=True)
            return True

    def materialize(self, sha: str, size: int, target) -> Path:
        """Copy a verified object without overwriting or linking the caller's file."""
        self._check_size(size)
        source = self._object(sha)
        target = Path(target).absolute()
        parent = _private_directory(target.parent)
        if target.exists() or target.is_symlink():
            raise FileExistsError('target already exists')
        with _lock(self.locks / (sha + '.lock')):
            if not self._verified(sha, size):
                raise FileNotFoundError('cache object missing')
            if shutil.disk_usage(parent).free < size + self.reserve:
                raise ValueError('insufficient space to materialize')
            fd, name = tempfile.mkstemp(prefix='.weight-copy-', dir=parent)
            temp = Path(name)
            try:
                with os.fdopen(fd, 'wb') as dst, source.open('rb') as src:
                    h, written = hashlib.sha256(), 0
                    for block in iter(lambda: src.read(_CHUNK), b''):
                        h.update(block)
                        written += len(block)
                        dst.write(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                if (h.hexdigest(), written) != (sha, size):
                    raise ValueError('cache object changed during materialization')
                os.link(temp, target, follow_symlinks=False)
                _sync_directory(parent)
            finally:
                temp.unlink(missing_ok=True)
        return target
