import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json, owned
from roop_3d.engine.pipeline_runner import MasterPipelineRunner, PipelineCheckpoint
from roop_3d.jobs.project_store import settings_digest
from roop_3d.jobs.schema import JobRequest, JobState
from roop_3d.tests.test_stage14_workstation import LocalCatalog, project_fixture


def test_atomic_json_and_binary_checkpoint_creation_and_fields(tmp_path):
    app, payload, model = project_fixture(tmp_path)
    request = JobRequest.model_validate(payload)
    job, _ = app.state.jobs.create(request, "cp-test-job")

    target_hash = sha256_file(app.state.media.path(request.target_path))[1]
    faceset_id = request.face_mappings[0].faceset_id
    curves = {"swap_weight": [{"frame": 0, "value": 0.5, "easing": "linear"}]}
    pose_cache = {"track_0000": [10.5, -5.2, 1.1]}
    mask = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    cp = PipelineCheckpoint(
        job_id=job.job_id,
        input_sha256=target_hash,
        request_sha256=MasterPipelineRunner._request_digest(job),
        settings_sha256=settings_digest(app.state.settings),
        input_hashes={"target": target_hash, "inspection": payload["inspection_id"]},
        faceset_ids=(faceset_id,),
        pose_cache=pose_cache,
        keyframe_curves=curves,
        manual_mask=mask,
        execution_provider="CUDAExecutionProvider",
        completed_frame=12,
        completed_segment=1,
        total_segments=3,
    )

    runner = MasterPipelineRunner(app.state.root, app.state.projects.catalog, app.state.media, lambda: app.state.settings)
    runner._write_checkpoint(cp)

    json_path = tmp_path / f"roop_3d/.runtime/jobs/{job.job_id}/checkpoint.json"
    bin_path = tmp_path / f"roop_3d/.runtime/jobs/{job.job_id}/checkpoint.bin"

    assert json_path.is_file(), "checkpoint.json must exist"
    assert bin_path.is_file(), "checkpoint.bin must exist"

    # Verify JSON content
    data = json.loads(json_path.read_text("utf-8"))
    assert data["job_id"] == job.job_id
    assert data["input_hashes"]["target"] == target_hash
    assert data["faceset_ids"] == [faceset_id]
    assert data["pose_cache"] == pose_cache
    assert data["keyframe_curves"] == curves
    assert data["manual_mask"] == mask
    assert data["execution_provider"] == "CUDAExecutionProvider"
    assert data["completed_frame"] == 12
    assert data["completed_segment"] == 1
    assert data["total_segments"] == 3

    # Verify Binary content
    raw_bin = bin_path.read_bytes()
    assert raw_bin.startswith(b"R3DCP001")
    json_len = int.from_bytes(raw_bin[8:12], "big")
    extracted_json = json.loads(raw_bin[12:12 + json_len].decode("utf-8"))
    assert extracted_json["job_id"] == job.job_id
    assert extracted_json["execution_provider"] == "CUDAExecutionProvider"


def test_project_validation_badges_inputs_verified_vs_missing_media(tmp_path):
    app, payload, model = project_fixture(tmp_path)
    request = JobRequest.model_validate(payload)
    job, _ = app.state.jobs.create(request, "badge-test-job")
    for state in (JobState.preparing, JobState.running, JobState.pausing, JobState.paused):
        job = app.state.jobs.transition(job.job_id, state)

    target_hash = sha256_file(app.state.media.path(request.target_path))[1]
    cp = PipelineCheckpoint(
        job_id=job.job_id,
        input_sha256=target_hash,
        request_sha256=MasterPipelineRunner._request_digest(job),
        settings_sha256=settings_digest(app.state.settings),
        completed_frame=5,
    )
    runner = MasterPipelineRunner(app.state.root, app.state.projects.catalog, app.state.media, lambda: app.state.settings)
    runner._write_checkpoint(cp)

    frame = tmp_path / f"roop_3d/.runtime/jobs/{job.job_id}/frames/00000000.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(cv2.imencode(".png", np.zeros((64, 96, 3), np.uint8))[1].tobytes())

    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        # Save project
        resp = client.post("/api/v1/projects", json={"name": "Badge Project", "request": payload, "checkpoint_job_id": job.job_id})
        assert resp.status_code == 201
        project_id = resp.json()["project_id"]

        # Validate project: should be Inputs Verified
        v_resp = client.get(f"/api/v1/projects/{project_id}/validate")
        assert v_resp.status_code == 200
        val = v_resp.json()
        assert val["inputs_verified"] is True
        assert val["status"] == "Inputs Verified"
        assert val["checkpoint"]["job_id"] == job.job_id

        # Projects list should include validation badge fields
        p_list = client.get("/api/v1/projects").json()
        matching = next(p for p in p_list if p["project_id"] == project_id)
        assert matching["inputs_verified"] is True
        assert matching["status"] == "Inputs Verified"

        # Now simulate missing source media: delete target video
        (tmp_path / payload["target_path"]).unlink()

        v_resp2 = client.get(f"/api/v1/projects/{project_id}/validate")
        assert v_resp2.status_code == 200
        val2 = v_resp2.json()
        assert val2["inputs_verified"] is False
        assert val2["status"] == "Missing Source Media"


def test_media_split_boundaries_endpoint(tmp_path):
    app, payload, model = project_fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        resp = client.post("/api/v1/media/split-boundaries", json={
            "target_path": payload["target_path"],
            "max_segment_duration": 0.5,
        })
        assert resp.status_code == 200, resp.text
        segments = resp.json()
        assert isinstance(segments, list)
        assert len(segments) >= 1
        assert "start_seconds" in segments[0]
        assert "end_seconds" in segments[0]
        assert "segment_index" in segments[0]
        assert segments[0]["segment_index"] == 0
