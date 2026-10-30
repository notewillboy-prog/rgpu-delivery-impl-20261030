"""Server reports partial upload offsets so reconnect can resume instead of restarting."""

from __future__ import annotations

import hashlib
from pathlib import Path

import helpers
import rawflow
from client import packager


def test_upload_partial_then_reconnect_reports_received(tmp_path: Path) -> None:
    server = helpers.ServerProcess(
        port=helpers.find_free_port(), token="resume-token",
        data_dir=tmp_path / "server", ready_timeout=20.0,
    ).start()
    try:
        project = helpers.copy_example_project(tmp_path / "proj")
        spec = packager.make_job_spec(
            project,
            ["python", "train.py", "--epochs", "1", "--data", "data.csv",
             "--out", "outputs", "--model", "model.json"],
            timeout_s=60,
        )
        client, _hello = rawflow.connect(f"127.0.0.1:{server.port}", token="resume-token")
        try:
            accepted = rawflow.submit(client, spec)
            job_id = accepted["job_id"]
            path = rawflow.missing_paths(accepted)[0]
            entry = spec.input_index()[path]
            data = (project / path).read_bytes()
            partial = data[:7]
            client.send_json({
                "kind": "file_chunk", "path": path, "offset": 0,
                "length": len(partial), "sha256": hashlib.sha256(partial).hexdigest(),
            })
            client.send_blob(partial)
            assert client.recv_json()["kind"] == "file_ok"
        finally:
            client.close()

        client, _hello = rawflow.connect(f"127.0.0.1:{server.port}", token="resume-token")
        try:
            resumed = rawflow.submit(client, spec, resume_job_id=job_id)
            assert resumed["job_id"] == job_id
            assert resumed["received"].get(path) == 7
            assert path in rawflow.missing_paths(resumed)
        finally:
            client.close()
    finally:
        server.stop(cleanup_data_dir=True)
