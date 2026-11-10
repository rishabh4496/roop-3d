"""Integration tests for model downloader endpoints, middleware alias, and preflight check action."""
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from roop_3d.core.model_downloader import REQUIRED_MODELS, find_model_path, get_model_search_paths
from roop_3d.core.settings import Settings
from roop_3d.server.app import create_app
from roop_3d.server.diagnostics import run_preflight_lint, PreFlightLintRequest


def test_required_models_specs_validity():
    """All 9 required models must have non-empty URLs, sha256s, and reasonable expected sizes."""
    assert len(REQUIRED_MODELS) == 9
    for name, spec in REQUIRED_MODELS.items():
        assert spec.primary_url.startswith("https://")
        assert spec.fallback_url.startswith("https://")
        assert len(spec.expected_sha256) == 64
        assert spec.expected_size_bytes > 1_000_000
        assert spec.expected_sha256 in spec.acceptable_sha256s


def test_model_search_paths_includes_app_models(tmp_path: Path):
    paths = get_model_search_paths(tmp_path)
    assert any("models" in str(p) for p in paths)


def test_download_missing_models_endpoint_routes(tmp_path: Path):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    client = TestClient(app, base_url="http://127.0.0.1:8030")

    # Call with empty list so nothing actually downloads over network in this test
    payload = {"models": []}

    # 1. Official endpoint /api/v1/models/download-missing
    res1 = client.post("/api/v1/models/download-missing", json=payload)
    assert res1.status_code == 200
    assert res1.json()["status"] == "ok"

    # 2. Underscore alias /api/v1/models/download_missing
    res2 = client.post("/api/v1/models/download_missing", json=payload)
    assert res2.status_code == 200
    assert res2.json()["status"] == "ok"

    # 3. Duplicate prefix handled by ApiAliasMiddleware: /api/v1/api/v1/models/download-missing
    res3 = client.post("/api/v1/api/v1/models/download-missing", json=payload)
    assert res3.status_code == 200
    assert res3.json()["status"] == "ok"


def test_preflight_lint_produces_download_models_action(tmp_path: Path):
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    target_video = tmp_path / "test.mp4"
    target_video.write_bytes(b"\x00" * 1024)

    payload = PreFlightLintRequest(target_path=str(target_video))
    report = run_preflight_lint(tmp_path, app.state, payload)

    checks = {c["id"]: c for c in report["checks"]}
    assert "required_models_on_disk" in checks
    model_check = checks["required_models_on_disk"]

    # Since tmp_path has no models, it should offer the download action
    assert "action" in model_check
    action = model_check["action"]
    assert action["id"] == "download_models"
    assert action["endpoint"] == "/models/download-missing"
    assert len(action["models"]) > 0
