import pytest
from fastapi.testclient import TestClient

from roop_3d.tests.test_stage14_workstation import project_fixture


def test_preflight_render_linter_checks(tmp_path):
    app, payload, model = project_fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        resp = client.post("/api/v1/system/preflight-lint", json={
            "target_path": payload["target_path"],
            "inspection_id": payload["inspection_id"],
            "face_mappings": payload["face_mappings"],
            "workstation_parameters": {"restoration_blend": 0.5, "pose_mode": "retarget"},
            "mask_base64": None,
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "status" in data
        assert "ready_to_queue" in data
        assert "checks" in data
        check_ids = [c["id"] for c in data["checks"]]
        assert "3d_pose_coverage" in check_ids
        assert "vram_headroom" in check_ids
        assert "out_of_frame_seams" in check_ids
        assert "output_encoder_storage" in check_ids


def test_synthetic_thread_benchmark_runner(tmp_path):
    app, payload, model = project_fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        resp = client.post("/api/v1/system/benchmark", json={
            "frames": 100,
            "thread_counts": [1, 2, 4],
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["tested_frames"] == 100
        assert len(data["results"]) >= 3
        assert "recommended_provider" in data
        assert "recommended_threads" in data
        assert "recommended_fps" in data


def test_14_section_structured_runtime_report(tmp_path):
    app, payload, model = project_fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        resp = client.get("/api/v1/system/diagnostics-report")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "report_id" in data
        assert "overall_status" in data
        assert "sections" in data
        assert len(data["sections"]) == 14
        assert "markdown" in data
        assert "Section 1: Python & Environment Integrity" in data["markdown"]
        assert "Section 14: Strict Offline Verification & Boundary Isolation" in data["markdown"]
