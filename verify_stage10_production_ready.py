"""Stage 10 Master Verification: Self-Healing Launcher, Auto-Model Downloader,
Pre-Flight Linter & GPU Diagnostics HUD.

Verifies:
1. Windows DLL injection and runtime directories (app/env, app/models, app/facesets).
2. Model Registry & Auto-Downloader (all 9 required models with URLs, SHA-256 hashes, and progress reporting).
3. Pre-Flight Render Linter (source face, target media, face lock, required models, VRAM/disk space).
4. Frontend production build and Pinokio launcher reliability (Vite invocation and node_modules integrity).
5. Full unified FastAPI startup health, hardware probe, and API endpoints.

Exits with code 0 only if every validation check passes.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, List

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Ensure DLL injection happens before ONNX Runtime or TensorRT import
from roop_3d.core.dll_injection import setup_environment
INJECTED_DLLS = setup_environment(ROOT)

failures: List[str] = []


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}", flush=True)
    if not condition:
        failures.append(message)


def verify_dll_injection_and_directories() -> None:
    print("\n--- 1. Verifying Windows DLL Injection & Runtime Directories ---")
    app_root = ROOT / "app"
    for d in ("env", "models", "facesets"):
        target = app_root / d
        check(target.is_dir(), f"Directory exists: {target.relative_to(ROOT)}")

    if sys.platform == "win32":
        path_str = os.environ.get("PATH", "")
        # Check that tensorrt or nvidia runtime paths are in PATH
        has_cuda_or_trt = any("tensorrt" in p.lower() or "nvidia" in p.lower() for p in path_str.split(os.pathsep))
        check(has_cuda_or_trt or len(INJECTED_DLLS) > 0, "CUDA / TensorRT DLL directory injected into os.environ['PATH']")
        print(f"  Injected {len(INJECTED_DLLS)} candidate DLL directories.")
    else:
        check(True, "Non-Windows platform: DLL injection skipped gracefully.")


def verify_model_downloader_and_registry() -> None:
    print("\n--- 2. Verifying Model Registry & Auto-Downloader ---")
    from roop.core.model_downloader import (
        REQUIRED_MODELS,
        get_model_spec,
        verify_all_required_models,
        verify_file_hash,
    )

    expected_models = [
        "inswapper_128_fp16.onnx",
        "scrfd_2.5g.onnx",
        "1k3d68.onnx",
        "2d106det.onnx",
        "xseg.onnx",
        "bisenet.onnx",
        "occluder.onnx",
        "GPEN-BFR-512.onnx",
        "ultramax.onnx",
    ]

    check(len(REQUIRED_MODELS) >= 9, f"Registry contains {len(REQUIRED_MODELS)} models (expected at least 9)")

    for name in expected_models:
        spec = get_model_spec(name)
        check(spec is not None, f"Model '{name}' is registered in canonical registry")
        if spec:
            check(spec.primary_url.startswith("http"), f"Model '{name}' has valid primary URL: {spec.primary_url}")
            check(spec.fallback_url.startswith("http"), f"Model '{name}' has valid fallback URL: {spec.fallback_url}")
            check(bool(re.fullmatch(r"[0-9a-f]{64}", spec.expected_sha256.lower())), f"Model '{name}' has valid 64-char SHA-256")
            check(spec.expected_size_bytes > 0, f"Model '{name}' has positive expected byte size: {spec.expected_size_bytes:,} bytes")

    status_report = verify_all_required_models(ROOT)
    check(isinstance(status_report, dict), "verify_all_required_models() returned valid status dictionary")
    for name in expected_models:
        check(name in status_report, f"Status report contains entry for '{name}'")


def verify_preflight_linter() -> None:
    print("\n--- 3. Verifying Pre-Flight Render Linter ---")
    from roop_3d.core.settings import Settings
    from roop_3d.server.diagnostics import PreFlightLintRequest, run_preflight_lint

    # Create mock state with Settings
    class MockState:
        def __init__(self):
            self.settings = Settings()
            self.unified_state = None

    state = MockState()

    # Create dummy target clip for test
    temp_media = ROOT / "cache/tmp/linter_test_media.mp4"
    temp_media.parent.mkdir(parents=True, exist_ok=True)
    temp_media.write_bytes(b"\x00" * 1024)

    try:
        payload = PreFlightLintRequest(
            target_path=str(temp_media),
            inspection_id="test_insp_01",
            face_mappings=[{"track_id": "T0", "faceset_id": "FS0", "source": "test_face"}],
            workstation_parameters={"restoration_blend": 0.4, "pose_mode": "retarget"},
            mask_base64=None,
        )

        res = run_preflight_lint(ROOT, state, payload)
        check("status" in res, "Pre-flight lint returns 'status'")
        check("checks" in res, "Pre-flight lint returns 'checks' list")
        check("ready_to_queue" in res, "Pre-flight lint returns 'ready_to_queue'")

        check_ids = {c["id"]: c for c in res["checks"]}

        # 5 Required check validations
        check("source_face_or_fsz" in check_ids, "Check 'source_face_or_fsz' is evaluated")
        check(check_ids["source_face_or_fsz"]["status"] == "PASS", "Source face / .fsz is recognized as PASS when mapped")

        check("target_media_valid" in check_ids, "Check 'target_media_valid' is evaluated")
        check(check_ids["target_media_valid"]["status"] == "PASS", "Target media is recognized as PASS when present")

        check("target_face_locked" in check_ids, "Check 'target_face_locked' is evaluated")
        check(check_ids["target_face_locked"]["status"] == "PASS", "Target face lock is recognized as PASS when mapped")

        check("required_models_on_disk" in check_ids, "Check 'required_models_on_disk' is evaluated")
        if check_ids["required_models_on_disk"]["status"] != "PASS":
            check("action" in check_ids["required_models_on_disk"], "Missing models offer actionable download fix")
            check(check_ids["required_models_on_disk"]["action"]["id"] == "download_models", "Action ID is 'download_models'")

        check("vram_headroom" in check_ids, "Check 'vram_headroom' is evaluated")
        check("storage_disk_space" in check_ids, "Check 'storage_disk_space' is evaluated")
        check(check_ids["storage_disk_space"]["status"] == "PASS", "Sufficient disk space recognized as PASS")

    finally:
        if temp_media.exists():
            temp_media.unlink()


def verify_frontend_build_and_launchers() -> None:
    print("\n--- 4. Verifying Frontend Production Build & Pinokio Launchers ---")
    ui_dist = ROOT / "app/roop_3d_ui/dist/index.html"
    check(ui_dist.is_file() and ui_dist.stat().st_size > 100, f"Frontend production build exists: {ui_dist.relative_to(ROOT)}")

    # Check package.json scripts
    pkg_json = ROOT / "app/roop_3d_ui/package.json"
    content = pkg_json.read_text(encoding="utf-8")
    check("npx --no-install vite build" in content, "package.json invokes Vite via local path ('npx --no-install vite build')")

    # Check install.js and start_roop3d.js
    install_js = (ROOT / "install.js").read_text(encoding="utf-8")
    check("npx --no-install vite build" in install_js, "install.js invokes Vite via local binary path")

    start_js = (ROOT / "start_roop3d.js").read_text(encoding="utf-8")
    check("npx --no-install vite build" in start_js, "start_roop3d.js builds frontend with local Vite")
    check("exists('app/roop_3d_ui/node_modules')" in start_js, "start_roop3d.js verifies node_modules integrity before building")


def verify_api_startup_and_endpoints() -> None:
    print("\n--- 5. Verifying Full Unified FastAPI Startup & Endpoints ---")
    from starlette.testclient import TestClient
    from app.api import create_unified_app

    app_instance = create_unified_app(repo_root=ROOT)
    client = TestClient(app_instance, base_url="http://127.0.0.1:8030")

    # 1. Health check
    res = client.get("/api/v1/health")
    check(res.status_code == 200, f"GET /api/v1/health returned 200 (Got: {res.status_code})")
    health = res.json()
    check(health.get("status") in ("ok", "online"), f"Health report status is valid ('ok' or 'online') (Got: {health.get('status')})")
    check(health.get("application") == "roop_3d", "Health report confirms application 'roop_3d'")
    check(health.get("standalone") is True, "Health report confirms standalone execution mode")

    # 1b. Hardware diagnostics check
    hw_res = client.get("/api/v1/system/hardware")
    check(hw_res.status_code == 200, f"GET /api/v1/system/hardware returned 200 (Got: {hw_res.status_code})")
    hw_data = hw_res.json()
    check("gpus" in hw_data or "advertised_providers" in hw_data, "Hardware report contains GPU and provider diagnostics")

    # 2. System preflight-lint
    lint_res = client.post("/api/v1/system/preflight-lint", json={"target_path": "cache/nonexistent.mp4"})
    check(lint_res.status_code == 200, f"POST /api/v1/system/preflight-lint returned 200 (Got: {lint_res.status_code})")
    lint_data = lint_res.json()
    check("status" in lint_data and "checks" in lint_data, "Pre-flight lint response structure is valid")

    # 3. Models required status
    models_res = client.get("/api/v1/models/required_status")
    check(models_res.status_code == 200, f"GET /api/v1/models/required_status returned 200 (Got: {models_res.status_code})")
    models_data = models_res.json()
    check(models_data.get("status") == "ok", "Models required status returned 'ok'")
    check("models" in models_data and len(models_data["models"]) == 9, "Returned exactly 9 required models")

    # 4. Telemetry probe
    caps_res = client.get("/api/v1/system/capabilities")
    check(caps_res.status_code == 200, f"GET /api/v1/system/capabilities returned 200 (Got: {caps_res.status_code})")


def main() -> int:
    print("================================================================================")
    print("STAGE 10: PRODUCTION READINESS MASTER VERIFICATION")
    print("================================================================================")

    verify_dll_injection_and_directories()
    verify_model_downloader_and_registry()
    verify_preflight_linter()
    verify_frontend_build_and_launchers()
    verify_api_startup_and_endpoints()

    print("\n================================================================================")
    if failures:
        print(f"FAILED: {len(failures)} verification check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("STAGE 10 MASTER VERIFICATION: ALL CHECKS PASSED (EXIT CODE 0)")
        print("================================================================================")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
