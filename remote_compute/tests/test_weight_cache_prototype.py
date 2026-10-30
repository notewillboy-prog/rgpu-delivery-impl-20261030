"""Exercise cache integrity, quotas, target isolation and concurrent writers."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from server.weight_cache_prototype import WeightCache


def identity(data):
    return hashlib.sha256(data).hexdigest(), len(data)


def cache(tmp_path, quota=10_000, reserve=0):
    return WeightCache(tmp_path / 'cache', quota_bytes=quota, reserve_bytes=reserve)


def test_put_hit_and_materialized_copy_are_independent(tmp_path):
    data = b'weight contents'; source = tmp_path/'source'
    source.write_bytes(data); sha, size = identity(data); instance = cache(tmp_path)
    assert instance.put(source, sha, size)
    assert not instance.put(source, sha, size)
    assert instance.contains(sha, size)
    output = instance.materialize(sha, size, tmp_path/'out'/'weight.bin')
    output.write_bytes(b'X' * size)
    assert instance.contains(sha, size)
    assert (instance.objects / sha).read_bytes() == data
    with pytest.raises(FileExistsError):
        instance.materialize(sha, size, output)
    assert output.read_bytes() == b'X' * size


def test_corrupt_object_never_counts_as_hit_and_is_not_overwritten(tmp_path):
    source=tmp_path/'source'; source.write_bytes(b'abcdef'); sha,size=identity(b'abcdef')
    instance=cache(tmp_path); instance.put(source,sha,size)
    (instance.objects/sha).write_bytes(b'ghijkl')
    for action in (lambda: instance.contains(sha,size),
                   lambda: instance.put(source,sha,size),
                   lambda: instance.materialize(sha,size,tmp_path/'out')):
        with pytest.raises(ValueError,match='corrupt'):
            action()
    assert (instance.objects/sha).read_bytes()==b'ghijkl'
    assert not (tmp_path/'out').exists()


def test_invalid_source_and_paths_are_rejected(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'content');instance=cache(tmp_path)
    sha,size=identity(b'other')
    with pytest.raises(ValueError,match='mismatch'):instance.put(source,sha,size)
    with pytest.raises(ValueError,match='SHA'):instance.contains('../bad',1)
    for size in (-1,True):
        with pytest.raises(ValueError):instance.contains('0'*64,size)
    link=tmp_path/'link'
    try:link.symlink_to(source)
    except OSError:pytest.skip('cannot create symlink')
    sha,size=identity(b'content')
    with pytest.raises(ValueError):instance.put(link,sha,size)


def test_quota_and_free_space_fail_before_publication(tmp_path,monkeypatch):
    data=b'abcdefgh';source=tmp_path/'source';source.write_bytes(data);sha,size=identity(data)
    instance=cache(tmp_path,quota=7)
    with pytest.raises(ValueError,match='quota'):instance.put(source,sha,size)
    assert not (instance.objects/sha).exists()
    instance=cache(tmp_path,quota=100,reserve=2)
    monkeypatch.setattr('server.weight_cache_prototype.shutil.disk_usage',lambda _:SimpleNamespace(free=9))
    with pytest.raises(ValueError,match='free disk'):instance.put(source,sha,size)
    assert not (instance.objects/sha).exists()


def test_concurrent_processes_share_one_object(tmp_path):
    data=os.urandom(1024*128);src=tmp_path/'source';src.write_bytes(data);sha,size=identity(data)
    root=tmp_path/'cache'
    program=('from server.weight_cache_prototype import WeightCache\n'
             f'c=WeightCache({str(root)!r},quota_bytes={size},reserve_bytes=0)\n'
             f'print(c.put({str(src)!r},{sha!r},{size}))\n')
    cwd=Path(__file__).resolve().parents[1]
    a=subprocess.Popen([sys.executable,'-c',program],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    b=subprocess.Popen([sys.executable,'-c',program],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    oa,ea=a.communicate(timeout=30);ob,eb=b.communicate(timeout=30)
    assert a.returncode == b.returncode == 0, (ea,eb)
    assert sorted([oa.strip(),ob.strip()]) == ['False','True']
    assert (root/'objects'/sha).read_bytes()==data


def test_distinct_objects_cannot_race_past_quota(tmp_path):
    size=64 * 1024; root=tmp_path/'cache'; sources=[]
    for i in range(2):
        data=bytes([i+1])*size; source=tmp_path/f'source{i}';source.write_bytes(data)
        sha,_=identity(data);sources.append((source,sha))
    code=('from server.weight_cache_prototype import WeightCache\n'
          'import sys\n'
          'c=WeightCache(sys.argv[1],quota_bytes=int(sys.argv[4]),reserve_bytes=0)\n'
          'try: c.put(sys.argv[2],sys.argv[3],int(sys.argv[4])); print("stored")\n'
          'except ValueError as exc: print(str(exc))\n')
    cwd=Path(__file__).resolve().parents[1]
    processes=[subprocess.Popen([sys.executable,'-c',code,str(root),str(src),sha,str(size)],
                                cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
               for src,sha in sources]
    results=[p.communicate(timeout=30) for p in processes]
    assert all(p.returncode==0 for p in processes),results
    assert sorted(out.strip() for out,_ in results)==['cache quota insufficient','stored']
    assert len(list((root/'objects').iterdir()))==1
