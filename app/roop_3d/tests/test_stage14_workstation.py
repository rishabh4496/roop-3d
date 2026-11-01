from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json
from roop_3d.engine.pipeline_runner import MasterPipelineRunner, PipelineCheckpoint
from roop_3d.jobs.project_store import settings_digest
from roop_3d.jobs.schema import JobRecord, JobRequest, JobState
from roop_3d.server.app import create_app
from roop_3d.tests.test_stage12_worker import _fixture, _video


class LocalCatalog:
    """Small hashed weight dependency; project hashing and storage remain real."""
    def __init__(self, path: Path):
        self.path = path
        path.write_bytes(b"fixture weight identity")
        self.record = {"model_id": "fixture", "sha256": sha256_file(path)[1], "name": "fixture.onnx"}
    def detector(self): return self.record
    def recognizer(self): return self.record
    def records(self): return [self.record]
    def get(self, model_id):
        assert model_id == "fixture"
        return self.record
    def verify(self, record):
        if sha256_file(self.path)[1] != record["sha256"]:
            raise DomainError("MODEL_DRIFT", "Fixture model bytes changed", 409)


def project_fixture(tmp_path):
    app, worker, payload = _fixture(tmp_path)
    model = tmp_path / "roop_3d/.runtime/fixture_weight.onnx"
    app.state.projects.catalog = LocalCatalog(model)
    return app, payload, model


def checkpoint(app, payload):
    request = JobRequest.model_validate(payload)
    job, _ = app.state.jobs.create(request, "project-job")
    for state in (JobState.preparing, JobState.running, JobState.pausing, JobState.paused):
        job = app.state.jobs.transition(job.job_id, state)
    value = PipelineCheckpoint(job_id=job.job_id, input_sha256=sha256_file(app.state.media.path(request.target_path))[1],
        request_sha256=MasterPipelineRunner._request_digest(job), settings_sha256=settings_digest(app.state.settings), completed_frame=0)
    atomic_json(app.state.root, f"roop_3d/.runtime/jobs/{job.job_id}/checkpoint.json", value.model_dump(mode="json"))
    frame = app.state.root / f"roop_3d/.runtime/jobs/{job.job_id}/frames/00000000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(cv2.imencode(".png", np.zeros((64, 96, 3), np.uint8))[1].tobytes())
    return job


def test_versioned_project_reopen_and_verified_checkpoint_resume(tmp_path):
    app, payload, model = project_fixture(tmp_path)
    job = checkpoint(app, payload)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.post("/api/v1/projects", json={"name": "Observed identity edit", "request": payload, "checkpoint_job_id": job.job_id})
        assert response.status_code == 201, response.text
        project = response.json()
        assert project["schema_version"] == 1 and project["last_completed_frame"] == 0
        assert project["model_digests"] == {"fixture": sha256_file(model)[1]}
        assert len(project["inputs"]) == 3
        assert "embedding" not in json.dumps(project) and "consent_token" not in json.dumps(project)
        project_id = project["project_id"]
        assert client.get(f"/api/v1/projects/{project_id}").status_code == 200
        exported = client.get(f"/api/v1/projects/{project_id}/download")
        assert exported.status_code == 200 and ".r3dp" in exported.headers["content-disposition"]
        assert client.post(f"/api/v1/projects/{project_id}/resume").json()["state"] == "queued"


@pytest.mark.parametrize("change", ["target", "faceset", "inspection", "model", "settings", "frames"])
def test_project_drift_blocks_checkpoint_continuation_without_modifying_manifest(tmp_path, change):
    app, payload, model = project_fixture(tmp_path)
    job = checkpoint(app, payload)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        project = client.post("/api/v1/projects", json={"name": "Integrity fixture", "request": payload, "checkpoint_job_id": job.job_id}).json()
        path = tmp_path / app.state.projects.relative(project["project_id"])
        before = path.read_bytes()
        paths = {"target": tmp_path / payload["target_path"], "faceset": tmp_path / "roop_3d/outputs/facesets/alice.fsz",
                 "inspection": tmp_path / f"roop_3d/.runtime/inspections/{payload['inspection_id']}.json", "model": model}
        if change in paths: paths[change].write_bytes(b"changed input")
        elif change == "settings": app.state.settings = app.state.settings.model_copy(update={"max_concurrent_jobs": 2})
        else: (tmp_path / f"roop_3d/.runtime/jobs/{job.job_id}/frames/00000000.png").unlink()
        response = client.post(f"/api/v1/projects/{project['project_id']}/resume")
        assert response.status_code == 409, response.text
        assert app.state.jobs.get(job.job_id).state == JobState.paused
        assert path.read_bytes() == before


def test_untrimmed_job_identity_remains_compatible_and_trim_bounds_validate():
    original = {"target_path": "roop_3d/.runtime/media/video.mp4", "inspection_id": "a" * 32,
                "face_mappings": [{"track_id": "track_0000", "faceset_id": "alice"}], "processing": "local", "remote_consent": False}
    value = JobRequest.model_validate(original)
    legacy = hashlib.sha256(json.dumps(original, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert MasterPipelineRunner._request_digest(JobRecord(request=value)) == legacy
    with pytest.raises(ValueError):
        JobRequest.model_validate({**original, "trim_start_seconds": 2, "trim_end_seconds": 1})


def test_trim_bounds_reach_real_timestamp_writer(tmp_path):
    app, worker, payload = _fixture(tmp_path, count=8)
    payload.update(trim_start_seconds=.08, trim_end_seconds=.24)
    record, _ = app.state.jobs.create(JobRequest.model_validate(payload), "trim-fixture")
    from roop_3d.engine.vram_governor import Admission
    output = worker.runner.run(record, Admission("cpu", "CPUExecutionProvider", 0, 0, "fixture"),
        control=lambda: JobState.running, mark_running=lambda: None, report_progress=lambda value: None)
    assert output.frames == 4
    manifest = json.loads((tmp_path / output.manifest_path).read_text("utf-8"))
    assert manifest["trim"] == {"start": "2/25", "end": "6/25"}


def test_production_root_spa_assets_and_api_404_are_separate(tmp_path):
    dist = tmp_path / "roop_3d_ui/dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<!doctype html><title>Real workstation build</title>', encoding="utf-8")
    (dist / "assets/app.js").write_text("window.workstation=true", encoding="utf-8")
    app = create_app(Settings(), repo_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        assert "Real workstation build" in client.get("/").text
        assert "Real workstation build" in client.get("/projects/library").text
        assert client.get("/assets/app.js").status_code == 200
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/api/v1/missing").status_code == 404
        assert client.get("/missing").status_code == 404
        assert client.get("/api/v1/health").json()["application"] == "roop_3d"


def send_upload(client, name, data):
    response = client.post("/api/v1/assets/uploads", json={"filename": name, "size_bytes": len(data)})
    assert response.status_code == 201, response.text
    upload_id = response.json()["upload_id"]
    assert client.put(f"/api/v1/assets/uploads/{upload_id}?offset=1", content=data).status_code == 409
    assert client.put(f"/api/v1/assets/uploads/{upload_id}?offset=0", content=data).status_code == 200
    return client.post(f"/api/v1/assets/uploads/{upload_id}/complete")


def test_chunk_upload_streaming_range_and_saved_inspection(tmp_path):
    app, payload, _ = project_fixture(tmp_path)
    data = (tmp_path / payload["target_path"]).read_bytes()
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = send_upload(client, "clip.mp4", data)
        assert response.status_code == 200, response.text
        target = response.json()["path"]
        response = client.get("/api/v1/media/content", params={"path": target}, headers={"Range": "bytes=0-31"})
        assert response.status_code == 206 and response.content == data[:32]
        assert client.get("/api/v1/media/keyframes", params={"path": target}).json()["timestamps"]
        assert client.get(f"/api/v1/media/inspections/{payload['inspection_id']}", params={"path": payload["target_path"]}).status_code == 200
        assert client.get("/api/v1/media/content", params={"path": "../outside.mp4"}).status_code == 422
        assert client.post("/api/v1/assets/uploads", json={"filename": "../escape.mp4", "size_bytes": 20}).status_code == 422


def test_fsz_import_and_coverage_use_stage04_validation_without_raw_embedding_response(tmp_path):
    app = create_app(Settings(), repo_root=tmp_path)
    crop = cv2.imencode(".png", np.full((64, 64, 3), 90, np.uint8))[1].tobytes()
    source = {"filename": "0.png", "identity": {"arcface_embedding": [1.0] + [0.0] * 511},
              "pose": [20, 0, 0]}
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("metadata.json", json.dumps({"source_name": "Observed fixture", "sources": [source]}))
        archive.writestr("0.png", crop)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = send_upload(client, "observed.fsz", data.getvalue())
        assert response.status_code == 200, response.text
        faceset_id = response.json()["faceset_id"]
        assert len(client.get("/api/v1/facesets").json()) == 1
        response = client.get(f"/api/v1/facesets/{faceset_id}/coverage")
        assert response.status_code == 200, response.text
        assert response.json()["views"][0]["yaw"] == 20
        assert "embedding" not in response.text
        malicious = io.BytesIO()
        with zipfile.ZipFile(malicious, "w") as archive:
            archive.writestr("../escape.png", crop)
        assert send_upload(client, "unsafe.fsz", malicious.getvalue()).status_code == 400
        assert not (tmp_path / "escape.png").exists()


def test_project_import_rejects_unknown_versions_and_changed_inputs(tmp_path):
    app, payload, _ = project_fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        saved = client.post("/api/v1/projects", json={"name": "Version fixture", "request": payload}).json()
        saved["project_id"] = "f" * 32
        assert client.post("/api/v1/projects/import", json={**saved, "schema_version": 99}).status_code == 422
        (tmp_path / payload["target_path"]).write_bytes(b"changed target")
        assert client.post("/api/v1/projects/import", json=saved).status_code == 409
        assert not (tmp_path / app.state.projects.relative(saved["project_id"])).exists()


def test_reconstruction_view_returns_real_hashed_geometry_and_rejects_wrong_source(tmp_path):
    app, payload, _ = project_fixture(tmp_path)
    base = tmp_path / "roop_3d/.runtime/reconstructions"
    base.mkdir(parents=True)
    mesh = base / "observed.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    reconstruction_id = "b" * 32
    relative = f"roop_3d/.runtime/reconstructions/{reconstruction_id}.json"
    receipt = {"method": "tps_2_5d", "coordinate_system": "opencv", "source_sha256": sha256_file(tmp_path / payload["target_path"])[1],
               "mesh_path": mesh.relative_to(tmp_path).as_posix(), "mesh_sha256": sha256_file(mesh)[1]}
    atomic_json(tmp_path, relative, receipt)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.get(f"/api/v1/reconstructions/{reconstruction_id}", params={"target_path": payload["target_path"]})
        assert response.status_code == 200, response.text
        assert response.json()["vertices"] == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        assert response.json()["method"] == "tps_2_5d"
        assert response.json()["texture"] is None
        receipt["source_sha256"] = "0" * 64
        atomic_json(tmp_path, relative, receipt)
        assert client.get(f"/api/v1/reconstructions/{reconstruction_id}", params={"target_path": payload["target_path"]}).status_code == 409


def test_prepared_checkpoint_refuses_changed_faceset_archive_before_model_creation(tmp_path):
    app, worker, payload = _fixture(tmp_path)
    archive = tmp_path / "roop_3d/outputs/facesets/alice.fsz"
    crop = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))[1].tobytes()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("0.png", crop)
        output.writestr("metadata.json", json.dumps({"sources": [{"filename": "0.png", "identity": {"arcface_embedding": [1.0] + [0.0] * 511}}]}))
    from roop_3d.faceset.faceset_reader import FaceSetReader
    worker.runner.faceset_reader = FaceSetReader(app.state.models, cache_root=tmp_path / "roop_3d/.runtime/cache")
    record, _ = app.state.jobs.create(JobRequest.model_validate(payload), "source-drift")
    prepared = worker.runner.prepare(record)
    assert prepared.checkpoint.faceset_sha256["alice"] == sha256_file(archive)[1]
    with zipfile.ZipFile(archive, "a") as output:
        output.comment = b"source archive bytes changed"
    with pytest.raises(DomainError) as error:
        worker.runner.prepare(record)
    assert error.value.code == "JOB_SOURCE_CHANGED"


def test_build_and_download_faceset(tmp_path):
    import base64
    app, payload, _ = project_fixture(tmp_path)
    tiny_png = cv2.imencode(".png", np.zeros((48, 48, 3), dtype=np.uint8))[1].tobytes()
    b64 = base64.b64encode(tiny_png).decode("utf-8")

    build_payload = {
        "name": "Actor Hero",
        "faceset_id": "actor_hero",
        "views": [
            {
                "image_base64": b64,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "embedding": [1.0] + [0.0] * 511,
            },
            {
                "image_base64": b64,
                "yaw": -25.0,
                "pitch": 5.0,
                "roll": 0.0,
                "embedding": [0.98] + [0.02] * 511,
            },
            {
                "image_base64": b64,
                "yaw": 40.0,
                "pitch": -2.0,
                "roll": 0.0,
                "embedding": [0.95] + [0.05] * 511,
            },
        ],
    }

    for view in build_payload['views']:
        view.update(landmarks=[[10,10],[30,10],[20,20],[12,32],[28,32]],bbox=[0,0,48,48])
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        res = client.post("/api/v1/facesets/build", json=build_payload)
        assert res.status_code == 201, res.text
        data = res.json()
        assert data["faceset_id"] == "actor_hero"
        assert data["item_count"] == 3

        # Coverage inspection
        cov_res = client.get("/api/v1/facesets/actor_hero/coverage")
        assert cov_res.status_code == 200
        cov = cov_res.json()
        assert len(cov["views"]) == 3

        # Download archive
        dl_res = client.get("/api/v1/facesets/actor_hero/download")
        assert dl_res.status_code == 200
        assert dl_res.headers["content-type"] == "application/zip"
        assert len(dl_res.content) > 0

        # Thumb inspection
        th_res = client.get("/api/v1/facesets/actor_hero/thumb")
        assert th_res.status_code == 200
        assert th_res.headers["content-type"] == "image/png"
        assert len(th_res.content) > 0

        # Rename faceset
        rn_res = client.post("/api/v1/facesets/actor_hero/rename", json={"name": "Renamed Hero"})
        assert rn_res.status_code == 200
        assert rn_res.json()["name"] == "Renamed Hero"

        # Open folder and rebuild thumbs
        f_res = client.post("/api/v1/facesets/open-folder")
        assert f_res.status_code == 200
        assert "path" in f_res.json()

        rb_res = client.post("/api/v1/facesets/rebuild-thumbs")
        assert rb_res.status_code == 200
        assert rb_res.json()["rebuilt"] >= 1

        # Delete faceset
        del_res = client.delete("/api/v1/facesets/actor_hero")
        assert del_res.status_code == 200
        assert del_res.json()["deleted"] == "actor_hero"


def test_concat_media_segments(tmp_path):
    app, payload, _ = project_fixture(tmp_path)
    seg1 = payload["target_path"]
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        res = client.post("/api/v1/media/concat", json={"segment_paths": [seg1, seg1]})
        assert res.status_code == 201, res.text
        data = res.json()
        assert data["segment_count"] == 2
        assert "concat_" in data["output_path"]
