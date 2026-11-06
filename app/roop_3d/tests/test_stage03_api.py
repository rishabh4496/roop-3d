from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import httpx

from roop_3d.core.settings import Settings
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.local_provider import LocalWeightProvider
from roop_3d.models.stage16 import STAGE16_IDS
from roop_3d.server.app import create_app
from roop_3d.tests.test_stage03_downloader import DATA, remote
from roop_3d.tests.test_stage03_registry import artifact, registry_for, reviewed_artifact

STAGE6_IDS = ("minimax_h3_ref2va_faceswap_lora", "bfs_qwen21_dare60_avgrank30",
              "bfs_qwen21_prune20_dare50_avgrank26", "bfs_qwen21_prune20_dare60_avgrank30")


def test_api_exposes_authoritative_registry_notices_and_blocked_license(tmp_path):
    app = create_app(Settings(), repo_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.get("/api/v1/models/registry")
        assert response.status_code == 200
        expected = app.state.models.registry.records()
        assert {item["model_id"] for item in response.json()} == {record.model_id for record in expected}
        assert len(response.json()) == len(expected)
        assert set(STAGE16_IDS) <= {item["model_id"] for item in response.json()}
        assert set(STAGE6_IDS) <= {item["model_id"] for item in response.json()}
        assert sum(item["status"] == "verified_local" for item in response.json()) == sum(record.status == "verified_local" for record in expected)
        flame = next(item for item in response.json() if item["model_id"] == "flame-2020")
        assert flame["expected_sha256"] is None and flame["upstream_url"] is None
        notice = client.get("/api/v1/models/flame-2020/license").json()
        assert notice["manual_acceptance_required"] and not notice["automatic_download_allowed"]
        h3_notice = client.get("/api/v1/models/minimax_h3_ref2va_faceswap_lora/license").json()
        assert h3_notice["redistribution_permission"] == "allow"
        assert h3_notice["base_model_license"]["license_id"] == "LicenseRef-MiniMax-H3-Community"
        assert h3_notice["automatic_download_allowed"] is False
        for payload in ({}, {"accepted_license": True}):
            denial = client.post("/api/v1/models/flame-2020/download", json=payload)
            assert denial.status_code == 403 and denial.json()["error"]["code"] == "MODEL_BLOCKED"
            assert denial.json()["error"]["request_id"]
        assert client.get("/api/v1/models/absent/license").status_code == 404
        assert client.get("/api/v1/system/capabilities").json()["automatic_model_download_state"] == "GATED"
        assert client.post("/api/v1/models/wav2lip/download").json()["error"]["code"] == "LICENSE_UNVERIFIED"


def test_api_local_first_reuse_does_not_download_unknown_license_weights(tmp_path):
    root = tmp_path / "roop_3d/.runtime/local"
    root.mkdir(parents=True)
    path = root / "fixture.onnx"
    path.write_bytes(b"local fixture weights")
    record = reviewed_artifact(locator={"scope": "fixture", "path": path.name}, commercial_use="deny")
    registry = registry_for(record)
    catalog = ModelCatalog(cache_root=tmp_path, registry=registry)
    catalog.provider = LocalWeightProvider(registry, root=tmp_path, scopes={"fixture": root})
    app = create_app(Settings(), repo_root=tmp_path)
    app.state.models = catalog
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.post("/api/v1/models/fixture/download")
        assert response.status_code == 200
        assert response.json() == {"model_id": "fixture", "state": "VERIFIED_LOCAL",
                                   "downloaded": False, "bytes_verified": True, "resumed": False}
        assert not (tmp_path / "roop_3d/models/cache").exists()
        assert client.post("/api/v1/models/fixture/download", json={"commercial": True}).status_code == 403


def test_api_download_gate_and_real_cached_adapter_resolution(tmp_path):
    record = remote(manual_acceptance_required=True)
    registry = registry_for(record)
    catalog = ModelCatalog(cache_root=tmp_path, registry=registry)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=DATA)
    catalog.downloader.transport = httpx.MockTransport(handler)
    app = create_app(Settings(), repo_root=tmp_path)
    app.state.models = catalog
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        denial = client.post("/api/v1/models/fixture/download")
        assert denial.status_code == 403 and denial.json()["error"]["code"] == "LICENSE_ACCEPTANCE_REQUIRED"
        assert client.post("/api/v1/models/fixture/download", json={"accepted_license": "true"}).status_code == 422
        response = client.post("/api/v1/models/fixture/download", json={"accepted_license": True})
        assert response.status_code == 200 and response.json()["downloaded"]
        retry = client.post("/api/v1/models/fixture/download", json={"accepted_license": True})
        assert not retry.json()["downloaded"] and len(calls) == 1
        with catalog.provider.open(record, accepted_license=True) as stream:
            assert stream.read() == DATA
        assert client.post("/api/v1/models/fixture/download", json={"url": "https://evil.example"}).status_code == 422
