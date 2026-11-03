"""Stage 8 integration gates. Contract sessions are explicit fixtures; codecs/storage/API are real.

The separate production-browser acceptance script uses actual local model inference.
"""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import av
import pytest
from fastapi.testclient import TestClient

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json
from roop_3d.engine.pipeline_runner import MasterPipelineRunner
from roop_3d.jobs.schema import JobState
from roop_3d.media.probe import probe
from roop_3d.media.video_reader import VideoReader
from roop_3d.tests.test_stage05_media import audio_payloads, frame_times, make_video
from roop_3d.tests.test_stage12_worker import _fixture, _wait_state
from roop_3d.tests.test_stage14_workstation import LocalCatalog, project_fixture


def test_two_projects_revisioned_save_preview_history_and_exports(tmp_path):
    app, worker, payload = _fixture(tmp_path)
    catalog = LocalCatalog(tmp_path / "roop_3d/.runtime/weight.onnx")
    app.state.projects.catalog = worker.runner.catalog = catalog
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        preview = {"timestamp": .12, "selected_track": "track_0000", "source_faceset_id": "alice",
                   "viewport": {"compare_mode": "diff", "view_mode": "compare", "zoom": 2}}
        draft = {"name": "First edit", "request": payload, "preview": preview}
        first = client.post("/api/v1/projects", json=draft)
        assert first.status_code == 201, first.text
        first = first.json()
        second = client.post("/api/v1/projects", json={**draft, "name": "Second edit"}).json()
        assert first["project_id"] != second["project_id"]
        path = f"/api/v1/projects/{first['project_id']}"
        updated = client.put(path, json={**draft, "name": "Revised edit", "revision": 0})
        assert updated.status_code == 200, updated.text
        assert updated.json()["revision"] == 1
        assert client.put(path, json={**draft, "revision": 0}).status_code == 409
        assert client.get(path).json()["preview"]["viewport"]["zoom"] == 2
        assert client.get(f"/api/v1/projects/{second['project_id']}").json()["revision"] == 0

        created = client.post(path + "/jobs", json={}, headers={"Idempotency-Key": "first-job"})
        assert created.status_code == 201, created.text
        job = created.json()
        assert job["project_id"] == first["project_id"] and job["effective_settings"]
        assert client.post(path + "/jobs", json={}, headers={"Idempotency-Key": "first-job"}).status_code == 200
        assert client.post(f"/api/v1/projects/{second['project_id']}/jobs", json={},
                           headers={"Idempotency-Key": "first-job"}).status_code == 409
        assert client.get(f"/api/v1/jobs/{job['job_id']}/artifacts/video").status_code == 409
        history = client.get(path).json()["output_history"]
        assert history[0]["state"] == "queued" and history[0]["output"] is None
        # The job's effective settings survive unrelated global config changes.
        app.state.settings = app.state.settings.model_copy(update={"temporal_beta": .09})
        worker.start()
        try:
            completed = _wait_state(client, job["job_id"], "succeeded", timeout=30)
        finally:
            worker.stop()
        exported = client.get(f"/api/v1/jobs/{job['job_id']}/artifacts/video")
        manifest = client.get(f"/api/v1/jobs/{job['job_id']}/artifacts/manifest")
        assert exported.status_code == manifest.status_code == 200
        assert manifest.json()["output_sha256"] == completed["output"]["output_sha256"]
        assert manifest.json()["stage12"]["effective_settings"]["temporal_beta"] != .09
        downloaded = client.get(path + "/download").json()
        assert downloaded["output_history"][0]["output"]["frames"] == 8
        assert client.get(f"/api/v1/projects/{second['project_id']}").json()["output_history"] == []
        # A modified artifact cannot be exported as the verified completed render.
        (tmp_path / completed["output"]["output_path"]).write_bytes(b"corrupted local output")
        assert client.get(f"/api/v1/jobs/{job['job_id']}/artifacts/video").status_code == 409


@pytest.mark.parametrize("change,code", [("model", "JOB_MODELS_CHANGED"),
                                         ("inspection", "JOB_INSPECTION_CHANGED"),
                                         ("legacy", "JOB_CHECKPOINT_MODELS_UNKNOWN"),
                                         ("request", "JOB_CHECKPOINT_MISMATCH")])
def test_resume_binds_models_inspection_and_request(tmp_path, monkeypatch, change, code):
    app, worker, payload = _fixture(tmp_path)
    catalog = LocalCatalog(tmp_path / "roop_3d/.runtime/weight.onnx")
    runner = worker.runner
    runner.catalog = catalog
    monkeypatch.setattr(runner, "_hardware_state", lambda settings: {"probe": "not needed by contract test"})
    from roop_3d.jobs.schema import JobRequest
    record, _ = app.state.jobs.create(JobRequest.model_validate(payload), "identity-test")
    prepared = runner.prepare(record)
    assert prepared.checkpoint.model_digests == {"fixture": catalog.record["sha256"]}
    runner._save_frame(record.job_id, 0, np.zeros((64, 96, 3), np.uint8), (64, 96, 3))
    saved = runner._checkpoint(prepared.checkpoint, completed_frame=0)
    if change == "model":
        catalog.path.write_bytes(b"a different verified model version")
        catalog.record = {**catalog.record, "sha256": sha256_file(catalog.path)[1]}
    elif change == "inspection":
        path = tmp_path / f"roop_3d/.runtime/inspections/{payload['inspection_id']}.json"
        path.write_text(path.read_text("utf-8") + "\n", encoding="utf-8")
    elif change == "legacy":
        runner._write_checkpoint(saved.model_copy(update={"model_digests": {}}))
    else:
        record = record.model_copy(update={"request": record.request.model_copy(update={"trim_end_seconds": .2})})
    with pytest.raises(DomainError) as failure:
        runner.prepare(record)
    assert failure.value.code == code


def test_full_worker_preserves_vfr_orientation_color_and_all_audio_tracks(tmp_path):
    app, worker, payload = _fixture(tmp_path, settings=Settings(hardware_profile="cpu",
        preferred_provider="CPUExecutionProvider", audio_policy="copy"))
    target = make_video(tmp_path / payload["target_path"], vfr=True, rotation=90, audio=2, audio_codec="aac", sar=Fraction(4, 3))
    receipt_path = f"roop_3d/.runtime/inspections/{payload['inspection_id']}.json"
    receipt = json.loads((tmp_path / receipt_path).read_text("utf-8"))
    receipt["target_sha256"] = sha256_file(target)[1]
    atomic_json(tmp_path, receipt_path, receipt)
    worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            response = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "vfr-worker"})
            assert response.status_code == 201, response.text
            job = _wait_state(client, response.json()["job_id"], "succeeded", timeout=30)
    finally:
        worker.stop()
    output = tmp_path / job["output"]["output_path"]
    assert frame_times(target) == frame_times(output)
    assert len(set(b - a for a, b in zip(frame_times(output), frame_times(output)[1:]))) > 1
    observed = probe(output)
    assert (observed.videos[0].width, observed.videos[0].height, observed.videos[0].rotation) == (64, 96, 0)
    assert (observed.videos[0].colorspace, observed.videos[0].color_primaries, observed.videos[0].color_trc) == (1, 1, 1)
    assert [track.language for track in observed.audio] == ["eng", "hin"]
    assert audio_payloads(target) == audio_payloads(output)
    with av.open(str(output)) as container:
        assert container.streams.video[0].sample_aspect_ratio == Fraction(3, 4)
    assert len(list(VideoReader(output, hardware="software").frames())) == 8


def test_model_file_drift_is_reported_in_project_validation(tmp_path):
    app, payload, weight = project_fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        saved = client.post("/api/v1/projects", json={"name": "Model integrity", "request": payload}).json()
        weight.write_bytes(b"changed model file")
        result = client.get(f"/api/v1/projects/{saved['project_id']}/validate").json()
        assert result["inputs_verified"] is False
        assert client.get(f"/api/v1/projects/{saved['project_id']}").status_code == 409


def test_one_corrupt_project_does_not_hide_other_local_projects(tmp_path):
    app, payload, _ = project_fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        saved = client.post("/api/v1/projects", json={"name": "Healthy edit", "request": payload}).json()
        broken = tmp_path / app.state.projects.relative("f" * 32)
        broken.write_bytes(b"{corrupt project")
        listed = client.get("/api/v1/projects")
        assert listed.status_code == 200
        records = {item["project_id"]: item for item in listed.json()}
        assert records[saved["project_id"]]["inputs_verified"]
        assert not records["f" * 32]["inputs_verified"]
        assert broken.read_bytes() == b"{corrupt project"


def test_dependency_snapshot_checks_bytes_keeps_missing_optional_models_optional(tmp_path):
    from roop_3d.models.catalog import render_dependencies
    catalog = LocalCatalog(tmp_path / "weight.onnx")
    optional = {"model_id": "optional", "name": "gpen_bfr_512.onnx", "sha256": "c" * 64}
    catalog.records = lambda: [catalog.record, optional]
    original = catalog.verify
    def verify(record):
        if record["model_id"] == "optional":
            raise DomainError("MODEL_NOT_INSTALLED", "Optional model weights absent", 404)
        original(record)
    catalog.verify = verify
    assert render_dependencies(catalog) == {"fixture": catalog.record["sha256"]}
    catalog.path.write_bytes(b"corrupt bytes with an unchanged registry entry")
    with pytest.raises(DomainError, match="Fixture model bytes changed"):
        render_dependencies(catalog)


def test_full_inspection_reuses_tracker_to_eof_without_legacy_sampling_caps(tmp_path, monkeypatch):
    from roop_3d.media import service
    from roop_3d.media.schema import InspectRequest
    class Tracker:
        # Explicit no-face contract fixture: real decode, limits and report writing, no model inference.
        scene_cut = False
        events = []
        events_dropped = 0
        def __init__(self, *args, **kwargs): pass
        def process(self, image): return ()
        def release(self): pass
    app, _, payload = _fixture(tmp_path, count=8)
    monkeypatch.setattr(service, "FaceTracker", Tracker)
    request = {"path": payload["target_path"], "identity_lock": True, "max_samples": 1, "max_decoded_frames": 1}
    bounded = app.state.media.inspect(InspectRequest.model_validate(request))
    full = app.state.media.inspect(InspectRequest.model_validate({**request, "full_video": True}))
    assert bounded.decoded_frames == 1 and bounded.truncated
    assert full.decoded_frames == 8 and not full.truncated
    # A full traversal still fails visibly at its time budget, without claiming EOF.
    ticks = iter([0.0] + [100.0] * 32)
    from types import SimpleNamespace
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    limited = app.state.media.inspect(InspectRequest.model_validate({**request, "full_video": True}))
    assert limited.truncated and limited.decoded_frames == 1


def test_resume_rejects_changed_3d_asset_before_new_frame_inference(tmp_path, monkeypatch):
    from roop_3d.tests.test_stage12_worker import _Session
    from roop_3d.jobs.schema import JobRequest
    from roop_3d.engine.vram_governor import Admission
    sessions = []
    class AssetSession(_Session):
        def capability(self):
            return {**super().capability(), "render_modes": {"identity_assets": {
                "alice": {"available": True, "build_key": "b" * 64}}}}
    def factory(catalog, config):
        session = AssetSession(config)
        sessions.append(session)
        return session
    app, worker, payload = _fixture(tmp_path, session_factory=factory)
    monkeypatch.setattr(worker.runner, "_hardware_state", lambda settings: {})
    record, _ = app.state.jobs.create(JobRequest.model_validate(payload), "asset-identity")
    prepared = worker.runner.prepare(record)
    saved = prepared.checkpoint.model_copy(update={"completed_frame": 0, "identity_assets": {"alice": "a" * 64}})
    from dataclasses import replace
    prepared = replace(prepared, checkpoint=saved)
    with pytest.raises(DomainError) as error:
        worker.runner.run_prepared(record, Admission(profile="cpu", provider="CPUExecutionProvider", free_vram_mib=0,
            reserved_vram_mib=0, reason="contract test"), prepared=prepared, control=lambda: None,
            mark_running=lambda: pytest.fail("Changed geometry must fail before rendering"), report_progress=lambda value: None)
    assert error.value.code == "JOB_IDENTITY_ASSET_CHANGED"
    assert sessions[-1].released


def test_vfr_preview_uses_true_frame_ordinals_and_distinct_timestamp_cache(tmp_path, monkeypatch):
    from roop_3d.tests.test_workstation_preview import preview_fixture, FixtureSession
    from roop_3d.tests.test_remediation_stage4 import live_payload
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    request = fixture["request"]
    target = make_video(tmp_path / request["target_path"], vfr=True)
    receipt_path = f"roop_3d/.runtime/inspections/{request['inspection_id']}.json"
    receipt = json.loads((tmp_path / receipt_path).read_text("utf-8"))
    receipt["target_sha256"] = sha256_file(target)[1]
    atomic_json(tmp_path, receipt_path, receipt)
    payload = live_payload(fixture)
    payload.pop("frame_index", None)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        first = client.post("/api/preview", json={**payload, "timestamp": .1})
        assert first.status_code == 200, first.text
        assert json.loads(first.headers["x-preview-meta"])["decoded_frame_index"] == 2
        assert FixtureSession.instances[-1].inferred == 3
        second = client.post("/api/preview", json={**payload, "timestamp": .13})
        assert second.status_code == 200, second.text
        assert json.loads(second.headers["x-preview-meta"])["decoded_frame_index"] == 3
        assert first.headers["x-preview-config-hash"] == second.headers["x-preview-config-hash"]
        assert second.headers["x-preview-cache"] == "MISS"
        repeated = client.post("/api/preview", json={**payload, "timestamp": .1})
        assert repeated.headers["x-preview-cache"] == "HIT"
        assert json.loads(repeated.headers["x-preview-meta"])["decoded_frame_index"] == 2
        exact = client.post("/api/preview", json={**payload, "frame_index": 5})
        assert exact.status_code == 200, exact.text
        assert json.loads(exact.headers["x-preview-meta"])["decoded_frame_index"] == 5
