"""End-to-end cache reuse: first job fills, second job receives without upload."""

from __future__ import annotations

from pathlib import Path

import helpers
import rawflow
from client import packager

ENTRY = ["python", "train.py", "--epochs", "1", "--data", "data.csv",
         "--out", "outputs", "--model", "model.json"]


def test_second_job_hits_weight_cache(tmp_path: Path) -> None:
    server = helpers.ServerProcess(
        port=helpers.find_free_port(), token="cache-token",
        data_dir=tmp_path / "server",
        extra_args=["--weight-cache-bytes", str(2 * 1024 ** 3)],
        ready_timeout=20.0,
    ).start()
    try:
        project = helpers.copy_example_project(tmp_path / "proj")
        spec = packager.make_job_spec(project, ENTRY, timeout_s=60)
        client, _hello = rawflow.connect(f"127.0.0.1:{server.port}", token="cache-token")
        try:
            accepted = rawflow.submit(client, spec)
            job_id = accepted["job_id"]
            chunk = rawflow.chunk_size_of(accepted)
            for path in rawflow.missing_paths(accepted):
                result = rawflow.upload_entry(client, spec.input_index()[path], project, chunk)
                assert result.get("done") is True, result
            finished = rawflow.wait_finished(client, job_id, timeout=120)
            assert finished["state"] == "succeeded", finished
        finally:
            client.close()

        second = packager.make_job_spec(project, ENTRY, timeout_s=60)
        client, _hello = rawflow.connect(f"127.0.0.1:{server.port}", token="cache-token")
        try:
            accepted = rawflow.submit(client, second)
            assert accepted["missing"] == [], accepted
            assert set(accepted["cache_hits"]) == {entry.path for entry in second.inputs}
            assert accepted["received"] == {}
        finally:
            client.close()
    finally:
        server.stop(cleanup_data_dir=True)
