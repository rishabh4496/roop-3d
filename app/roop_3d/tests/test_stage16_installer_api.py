"""Integration tests for Stage 16 installer REST API endpoints."""
from __future__ import annotations

from pathlib import Path
from starlette.testclient import TestClient

from roop_3d.server.app import create_app


def test_installer_api_endpoints(tmp_path: Path) -> None:
    app = create_app(repo_root=tmp_path, worker_enabled=False)
    client = TestClient(app, base_url="http://127.0.0.1:8030")

    # 1. Preflight endpoint
    res_preflight = client.get("/api/v1/installer/preflight")
    assert res_preflight.status_code == 200
    data_pf = res_preflight.json()
    assert "status" in data_pf
    assert "checks" in data_pf
    assert "resolved_hardware_profile" in data_pf

    # 2. Offline health endpoint
    res_offline = client.get("/api/v1/installer/offline-health")
    assert res_offline.status_code == 200
    data_off = res_offline.json()
    assert "offline_ready" in data_off
    assert "core_models" in data_off

    # 3. Model verify endpoint
    res_verify = client.post("/api/v1/installer/models/verify")
    assert res_verify.status_code == 200
    data_ver = res_verify.json()
    assert "total_registered" in data_ver
    assert "verified_count" in data_ver

    # 4. Updates check endpoint
    res_updates = client.get("/api/v1/installer/updates/check")
    assert res_updates.status_code == 200
    data_up = res_updates.json()
    assert "status" in data_up

    # 5. Backup endpoint
    res_backup = client.post("/api/v1/installer/updates/backup")
    assert res_backup.status_code == 200
    data_bk = res_backup.json()
    assert data_bk["status"] == "PASS"

    # 6. Recover endpoint
    res_recover = client.post("/api/v1/installer/recover")
    assert res_recover.status_code == 200
    data_rec = res_recover.json()
    assert "status" in data_rec
