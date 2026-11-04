from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.downloads import ModelDownloadManager
from roop_3d.models.hub import ModelHubService
from roop_3d.models.registry import Manifest, ModelRegistry
from roop_3d.server.app import create_app
from roop_3d.tests.test_stage03_registry import artifact, reviewed_artifact


def test_hub_exposes_registry_evidence_and_keeps_experimental_engines_gated(tmp_path):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.get("/api/v1/models/hub")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["execution_policy"] == "offline_by_default"
        assert set(body["model_states"]) == {"AVAILABLE", "NOT_INSTALLED", "LICENSE_RESTRICTED", "INCOMPATIBLE",
                                             "UNVERIFIED", "BROKEN", "QUARANTINED", "READY_FOR_INFERENCE"}
        ids = {item["model_id"] for item in body["models"]}
        assert {"minimax_h3_ref2va_faceswap_lora", "bfs_qwen21_dare60_avgrank30",
                "bfs_qwen21_prune20_dare50_avgrank26", "bfs_qwen21_prune20_dare60_avgrank30"} <= ids
        required = {"version", "family", "engine_type", "purpose", "source_repository", "weight_files", "license",
                    "minimum_vram_gib", "recommended_vram_gib", "dependencies", "supported_precision",
                    "supported_execution_providers", "status", "benchmark_status", "actual_inference_available"}
        assert all(required <= item.keys() for item in body["models"])
        h3 = next(item for item in body["models"] if item["model_id"] == "minimax_h3_ref2va_faceswap_lora")
        assert h3["license"]["id"] == "Apache-2.0"
        assert h3["weight_files"][0]["sha256"] == "1e032cf519cc143f434e67516d8ad0aacf4c6e146315b2dcc3b1c2800470326d"
        assert h3["minimum_vram_gib"] is None and h3["recommended_vram_gib"] is None
        assert h3["download_allowed"] is True
        engines = {item["engine_id"]: item for item in body["engines"]}
        assert engines["reference_onnx"]["selected"] is True
        assert engines["minimax_h3_ref2va_experimental"]["selectable"] is False
        assert engines["bfs_qwen_image21_experimental"]["selectable"] is False
        blocked = client.post("/api/v1/models/minimax_h3_ref2va_faceswap_lora/downloads", json={})
        assert blocked.status_code == 403
        assert blocked.json()["error"]["code"] == "LICENSE_ACCEPTANCE_REQUIRED"


@pytest.mark.parametrize("family", ["MiniMax H3 Ref2VA LoRA", "Qwen-Image-2.1 face-swap LoRA", "other reviewed model"])
def test_hub_finds_hash_verified_artifacts_in_configured_local_model_paths(tmp_path, family):
    data = b"verified local adapter bytes"
    record = artifact(data, status="verified_remote", upstream_url="https://github.com/fixture/model.onnx",
                      provenance_state="VERIFIED_REMOTE", license_id="Apache-2.0", commercial_use="allow",
                      manual_acceptance_required=True, model_family=family)
    registry = ModelRegistry(manifest=Manifest(evidence_scope="Fixture", models=(record,)))
    catalog = ModelCatalog(cache_root=tmp_path, registry=registry)
    search_root = tmp_path / "roop_3d/.runtime/weights"
    local_copy = search_root / "nested" / record.filenames[0]
    local_copy.parent.mkdir(parents=True)
    local_copy.write_bytes(data)
    catalog.provider.search_paths = lambda: (str(search_root),)

    row = ModelHubService(catalog).model(record.model_id)
    assert row["status"] == "AVAILABLE"
    assert row["actual_inference_available"] is False
    assert row["download_allowed"] is False
    assert "hash-verified" in row["status_evidence"]


def test_hub_uses_runtime_first_match_for_duplicate_core_model_names(tmp_path, monkeypatch):
    digest = "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af"
    first = artifact(b"first registry fixture", model_id="inswapper_first", name="inswapper_128.onnx",
                     role="identity_swapper", expected_sha256=digest, byte_size=1)
    second = artifact(b"second registry fixture", model_id="inswapper_second", name="inswapper_128.onnx",
                      role="identity_swapper", expected_sha256=digest, byte_size=1)
    registry = ModelRegistry(manifest=Manifest(evidence_scope="Fixture", models=(first, second)))
    catalog = ModelCatalog(cache_root=tmp_path, registry=registry)
    def resolve(record):
        if record.model_id == first.model_id:
            raise DomainError("MODEL_NOT_INSTALLED", "first runtime copy is absent", 404)
        return Path("second-copy.onnx")
    monkeypatch.setattr(catalog.provider, "resolve", resolve)

    service = ModelHubService(catalog)
    selected, _ = service._core["identity_swapper"]
    assert selected.model_id == first.model_id
    assert service._core_status(selected)[0] == "NOT_INSTALLED"


def test_legacy_download_and_trt_routes_no_longer_bulk_download_or_write_fake_engines(tmp_path):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        bulk = client.post("/api/v1/models/download", json={})
        assert bulk.status_code == 422
        assert bulk.json()["error"]["code"] == "MODEL_SELECTION_REQUIRED"
        compile_response = client.post("/api/v1/models/compile_trt", json={"model_id": "inswapper_128"})
        assert compile_response.status_code == 409
        assert compile_response.json()["error"]["code"] == "TRT_BUILD_UNAVAILABLE"
    assert not (tmp_path / "app/models/trt_cache").exists()


def test_legacy_registry_does_not_treat_engine_filename_as_inference_receipt(tmp_path, monkeypatch):
    from roop_3d.model_registry import UnifiedModelRegistry

    monkeypatch.setattr(UnifiedModelRegistry, "get_full_readiness_report", lambda self: [
        {"model_id": "inswapper_128", "readiness_state": "READY + TRT ENGINE",
         "trt_engine_path": "fake.engine", "hash_valid": True},
        {"model_id": "stage16_liveportrait_eye", "readiness_state": "READY + TRT ENGINE",
         "trt_engine_path": "receipt-verified.engine", "hash_valid": True},
    ])
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.get("/api/models/registry")
        assert response.status_code == 200
        rows = {row["model_id"]: row for row in response.json()}
        legacy = rows["inswapper_128"]
        assert legacy["readiness_state"] == "DOWNLOADED (NEEDS TRT)"
        assert legacy["trt_engine_path"] is None
        assert legacy["actual_inference_available"] is False
        # Stage 16 has its own receipt-based validator and retains its own evidence state.
        assert rows["stage16_liveportrait_eye"]["readiness_state"] == "READY + TRT ENGINE"


def test_verify_route_does_not_promote_downloaded_adapter_bytes_to_inference_ready(tmp_path, monkeypatch):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    model_id = "minimax_h3_ref2va_faceswap_lora"
    monkeypatch.setattr(app.state.models.provider, "resolve", lambda requested: Path("h3_ref2va.safetensors"))
    monkeypatch.setattr(app.state.model_hub, "model", lambda requested: {
        "status": "AVAILABLE", "actual_inference_available": False,
    })
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.post("/api/v1/models/hub/verify", json={"model_id": model_id})
        assert response.status_code == 200, response.text
        row = response.json()["models"][0]
        assert row["bytes_verified"] is True
        assert row["status"] == "AVAILABLE"
        assert row["actual_inference_available"] is False


def test_hub_verify_preserves_license_and_unknown_model_states(tmp_path):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        restricted = client.post("/api/v1/models/hub/verify", json={"model_id": "flame-2020"})
        assert restricted.status_code == 200
        assert restricted.json()["models"][0]["status"] == "LICENSE_RESTRICTED"
        assert restricted.json()["models"][0]["actual_inference_available"] is False
        missing = client.post("/api/v1/models/hub/verify", json={"model_id": "not-registered"})
        assert missing.status_code == 200
        assert missing.json()["models"][0]["status"] == "NOT_INSTALLED"


def test_active_downloads_are_listed_for_ui_reconnection(tmp_path):
    record = reviewed_artifact(b"unneeded fixture bytes", filenames=["fixture.onnx"], status="verified_remote", upstream_url="https://github.com/fixture/model.onnx",
                      provenance_state="VERIFIED_REMOTE", license_id="LicenseRef-Fixture", commercial_use="allow",
                      manual_acceptance_required=True)
    registry = ModelRegistry(manifest=Manifest(evidence_scope="Fixture", models=(record,)))
    catalog = ModelCatalog(cache_root=tmp_path, registry=registry)
    manager = ModelDownloadManager(catalog)
    entered, release = threading.Event(), threading.Event()
    def blocked_download(*args, **kwargs):
        entered.set()
        release.wait(3)
    catalog.downloader.download = blocked_download
    job = manager.start(record.model_id, accepted_license=True)
    try:
        assert entered.wait(3)
        duplicate = manager.start(record.model_id, accepted_license=True)
        assert duplicate["download_id"] == job["download_id"]
        assert [item["download_id"] for item in manager.active()] == [job["download_id"]]
    finally:
        release.set()
        manager.close()


def test_active_download_list_route_exposes_a_started_hub_job(tmp_path, monkeypatch):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    fixture = reviewed_artifact(b"download fixture", model_id="download_fixture",
                                status="verified_remote", upstream_url="https://github.com/fixture/download.onnx",
                                provenance_state="VERIFIED_REMOTE", manual_acceptance_required=True)
    app.state.models.registry._records[fixture.model_id] = fixture
    entered, release = threading.Event(), threading.Event()

    def blocked_download(*args, **kwargs):
        entered.set()
        release.wait(3)
        if kwargs["cancel_event"].is_set():
            raise DomainError("DOWNLOAD_CANCELLED", "fixture cancellation", 499)

    monkeypatch.setattr(app.state.models.downloader, "download", blocked_download)
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            started = client.post("/api/v1/models/download_fixture/downloads",
                                  json={"accepted_license": True})
            assert started.status_code == 202, started.text
            assert entered.wait(3)
            active = client.get("/api/v1/models/downloads")
            assert active.status_code == 200
            assert active.json()[0]["download_id"] == started.json()["download_id"]
            cancelled = client.delete(f"/api/v1/models/downloads/{started.json()['download_id']}")
            assert cancelled.status_code == 200
            assert cancelled.json()["state"] == "cancelling"
    finally:
        release.set()


def test_download_manager_cancellation_is_resumable_and_license_gated(tmp_path):
    data = b"verified adapter fixture" * 4096
    record = reviewed_artifact(data, filenames=["adapter.safetensors"], serialization_format="SAFETENSORS", status="verified_remote", upstream_url="https://github.com/fixture/adapter.safetensors",
                      provenance_state="VERIFIED_REMOTE", license_id="LicenseRef-Fixture", commercial_use="allow",
                      manual_acceptance_required=True)
    registry = ModelRegistry(manifest=Manifest(evidence_scope="Fixture", models=(record,)))
    catalog = ModelCatalog(cache_root=tmp_path, registry=registry)
    entered, release = threading.Event(), threading.Event()
    class SlowStream(httpx.SyncByteStream):
        def __iter__(self):
            yield data[:65536]
            entered.set()
            release.wait(5)
            yield data[65536:]
    catalog.downloader.transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=SlowStream(), headers={"ETag": '"fixture-v1"'}))
    manager = ModelDownloadManager(catalog)
    try:
        with pytest.raises(DomainError) as blocked:
            manager.start(record.model_id)
        assert blocked.value.code == "LICENSE_ACCEPTANCE_REQUIRED"
        job = manager.start(record.model_id, accepted_license=True)
        assert entered.wait(3)
        cancelling = manager.cancel(job["download_id"])
        assert cancelling["state"] == "cancelling"
        release.set()
        deadline = time.monotonic() + 3
        result = manager.get(job["download_id"])
        while result["state"] not in {"cancelled", "failed"} and time.monotonic() < deadline:
            time.sleep(0.01)
            result = manager.get(job["download_id"])
        assert result["state"] == "cancelled"
        assert result["lifecycle_state"] == "NOT_INSTALLED"
        final, partial, metadata = catalog.downloader.paths(record)
        assert not (tmp_path / final).exists()
        assert (tmp_path / partial).read_bytes() == data[:65536]
        assert (tmp_path / metadata).is_file()
        assert hashlib.sha256(data[:65536]).hexdigest() != record.expected_sha256
    finally:
        release.set()
        manager.close()
