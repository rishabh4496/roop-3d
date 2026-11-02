from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.engine.registry import EngineRegistry
from roop_3d.jobs.schema import JobOutput, JobRequest, JobState
from roop_3d.jobs.store import JobStore
from roop_3d.jobs.worker import JobWorker
from roop_3d.server.app import create_app


def request(**extra):
    return JobRequest(target_path="local.mp4", inspection_id="a" * 32,
                      face_mappings=({"track_id": "track_0000", "faceset_id": "alice"},), **extra)


def test_older_jobs_and_explicit_reference_keep_original_fingerprints(tmp_path):
    old = request().model_dump(mode="json", exclude={"engine_id"})
    restored = JobRequest.model_validate(old)
    assert restored.engine_id == "reference_onnx"
    assert "engine_id" not in restored.identity_payload()
    assert restored.identity_payload() == request(engine_id="reference_onnx").identity_payload()
    store = JobStore(tmp_path)
    first, _ = store.create(restored, "old-key")
    same, created = store.create(request(engine_id="reference_onnx"), "old-key")
    assert not created and same.job_id == first.job_id
    assert request(engine_id="local_fixture").identity_payload()["engine_id"] == "local_fixture"


@pytest.mark.parametrize("engine_id,code", [("minimax_h3_ref2va_experimental", "ENGINE_UNAVAILABLE"),
                                            ("bfs_qwen_image21_experimental", "ENGINE_UNAVAILABLE"),
                                            ("unknown_engine", "ENGINE_NOT_FOUND")])
def test_unavailable_selection_rejected_before_media_access(tmp_path, engine_id, code):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.post("/api/v1/jobs", json=request(engine_id=engine_id).model_dump(mode="json"),
                               headers={"Idempotency-Key": "unavailable"})
        assert response.json()["error"]["code"] == code
        assert client.get("/api/v1/jobs").json() == []


def test_worker_dispatches_explicit_adapter_and_never_falls_back(tmp_path):
    calls = []

    class Runner:
        def prepare(self, record, settings):
            calls.append(record.request.engine_id)
            return "prepared"

        def retry_profile(self, job_id):
            return None

        def run_prepared(self, record, admission, **kwargs):
            assert kwargs["prepared"] == "prepared"
            kwargs["mark_running"]()
            return JobOutput(output_path="roop_3d/.runtime/output.mp4", manifest_path="roop_3d/.runtime/output.json",
                             output_sha256="a" * 64, frames=1, audio_policy="drop")

    store = JobStore(tmp_path)
    reference = Runner()
    worker = JobWorker(tmp_path, store, Settings, None, None, pipeline_runner=reference)
    alternate = Runner()
    worker.engines.register("local_fixture", alternate)
    assert worker.engines.resolve("reference_onnx") is reference
    assert worker.engines.resolve("local_fixture") is alternate
    worker.scheduler = SimpleNamespace(try_admit=lambda *args, **kwargs: SimpleNamespace(
        lease=SimpleNamespace(admission=None)), release=lambda job_id: None)
    selected, _ = store.create(request(engine_id="local_fixture"), "selected")
    worker._execute_job(selected.job_id)
    assert calls == ["local_fixture"]
    assert store.get(selected.job_id).state == JobState.succeeded
    unsupported, _ = store.create(request(engine_id="unknown_engine"), "unsupported")
    worker._execute_job(unsupported.job_id)
    assert store.get(unsupported.job_id).failure.code == "ENGINE_NOT_FOUND"
    assert calls == ["local_fixture"]


def test_registry_refuses_incomplete_or_incompatible_adapters():
    registry = EngineRegistry()
    with pytest.raises(ValueError):
        registry.register("incomplete", object())
    with pytest.raises(ValueError):
        registry.register("image_only", object(), job_contract="images")
    with pytest.raises(DomainError):
        registry.resolve("reference_onnx")
