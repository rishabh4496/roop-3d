#!/usr/bin/env python3
"""Stage 1 Verification Script: Unification & Removal of Legacy File Protections.

Verifies:
1. Zero read-only/legacy file protection blocks remain in the codebase.
2. Direct writes and imports across app/, roop_3d/, and frontend are allowed.
3. All 3D reconstruction, faceset inspection, timeline frame streaming,
   enhanced processor, and state store routes are cleanly registered on FastAPI.
4. Unified state store correctly manages loaded source faces, .fsz facesets,
   target media, active render modes, and active processors.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add app and root to sys.path
ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
for p in (str(APP_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 1. Import app.api and roop_3d modules
print("[1/5] Importing app.api and core roop_3d modules...", flush=True)
from app.api import app, state, UnifiedStateStore, RenderMode
import roop_3d
import roop_3d.server.app
import roop_3d.server.routes
import roop_3d.server.workstation
import roop_3d.server.workstation_preview
import roop_3d.server.live_preview
import roop_3d.server.face_extract
import roop_3d.server.face_bank
import roop_3d.server.identity_assets
import roop_3d.server.orchestrator
import roop_3d.audit.isolation as isolation
from roop_3d.audit.isolation import is_authorized, safe_path, verify, AuditError
import roop_3d.tools.git_guard as git_guard
from roop_3d.tools.git_guard import allowed_path
import roop_3d.audit.stage00 as stage00
print("      Imports successful.", flush=True)

# 2. Confirm zero read-only/legacy protection blocks remain
print("[2/5] Confirming zero read-only / legacy protection blocks remain...", flush=True)

# Test is_authorized
for path_str in ("app/api.py", "app/roop/relighting_3d.py", "app/roop/__init__.py", "react-ui/src/App.tsx", ".gitignore", "start_roop3d.js"):
    p = Path(path_str)
    assert is_authorized(p) is True, f"Path {path_str} should be authorized in unified architecture"

# Test safe_path write permissions on legacy paths
for path_str in ("app/api.py", "app/roop/relighting_3d.py", "react-ui/src/App.jsx", ".gitignore", "start_roop3d.js", "roop_3d/server/app.py"):
    resolved = safe_path(ROOT, path_str, write=True)
    expected = ROOT / Path(path_str)
    assert resolved == expected, f"safe_path write failed for {path_str}: got {resolved}, expected {expected}"

# Test allowed_path in git_guard
for path_str in ("app/api.py", "app/roop/relighting_3d.py", "react-ui/src/App.tsx", ".gitignore", "start_roop3d.js", "verify_stage1_unification.py"):
    assert allowed_path(path_str) is True, f"allowed_path rejected {path_str}"

# Test that generated binary/cache artifacts are still guarded
for gen_path in ("app/roop_3d/.venv/python.exe", "app/models/model.engine", "cache/weights.onnx", "app/__pycache__/cache.pyc"):
    assert allowed_path(gen_path) is False, f"allowed_path should reject generated artifact {gen_path}"

# Test isolation.verify returns clean PASS with 0 violations
verification = verify(ROOT, {"files": [{"path": "app/api.py", "size_bytes": 100, "sha256": "0" * 64}]})
assert verification["ok"] is True, f"verify() should report ok: True, got {verification}"
assert len(verification["violations"]) == 0, f"verify() should have 0 violations, got {verification['violations']}"

print("      All legacy protection checks removed and boundary guards updated.", flush=True)

# 3. Verify FastAPI route registrations
print("[3/5] Verifying FastAPI routes (/api/v1/...) are registered cleanly...", flush=True)

def collect_routes(fastapi_app) -> set[str]:
    paths = set()
    for route in fastapi_app.routes:
        if hasattr(route, "path"):
            paths.add(route.path)
        if "IncludedRouter" in type(route).__name__:
            prefix = getattr(route.original_router, "prefix", "") or ""
            for sub_r in route.original_router.routes:
                if hasattr(sub_r, "path"):
                    sub_p = sub_r.path
                    if not sub_p.startswith(prefix):
                        full = (prefix.rstrip("/") + "/" + sub_p.lstrip("/")).rstrip("/")
                    else:
                        full = sub_p
                    paths.add(full)
    return paths

registered_routes = collect_routes(app)
print(f"      Total registered routes: {len(registered_routes)}")

# Expected key route categories
expected_3d_routes = [
    "/api/v1/reconstructions",
    "/api/v1/reconstructions/{reconstruction_id}",
    "/api/v1/facesets/{faceset_id}/identity-asset",
    "/api/v1/facesets/{faceset_id}/identity-asset/mesh",
    "/api/v1/facesets/{faceset_id}/views/{index}/image",
]

expected_faceset_routes = [
    "/api/v1/facesets",
    "/api/v1/facesets/{faceset_id}/coverage",
    "/api/v1/facesets/{faceset_id}/views",
    "/api/v1/facesets/build",
    "/api/v1/facesets/{faceset_id}/thumb",
    "/api/v1/facesets/{faceset_id}/rename",
]

expected_timeline_routes = [
    "/api/v1/media/preview-frame",
    "/api/v1/media/detect-frame-faces",
    "/api/v1/media/split-boundaries",
    "/api/v1/media/concat-segments",
    "/api/v1/preview",
]

expected_processor_routes = [
    "/api/v1/jobs",
    "/api/v1/jobs/{job_id}",
    "/api/v1/faces/bank",
    "/api/v1/faces/extract_from_frame",
    "/api/v1/orchestrator/plan",
    "/api/v1/orchestrator/engines",
]

expected_state_routes = [
    "/api/v1/state",
    "/api/v1/render-mode",
    "/api/v1/source/faces",
    "/api/v1/source/clear",
    "/api/v1/target/media",
    "/api/v1/processors",
    "/api/v1/health",
]

missing_routes = []
for category_name, route_list in [
    ("3D Reconstruction", expected_3d_routes),
    ("Faceset Inspection", expected_faceset_routes),
    ("Timeline Frame Streaming", expected_timeline_routes),
    ("Enhanced Processor", expected_processor_routes),
    ("Unified State Store", expected_state_routes),
]:
    for r in route_list:
        if r not in registered_routes:
            missing_routes.append((category_name, r))

assert not missing_routes, f"Missing required routes: {missing_routes}"
print("      All required 3D, faceset, timeline, processor, and state routes verified.", flush=True)

# 4. Verify Unified State Store
print("[4/5] Testing Unified State Store...", flush=True)

# Test Render Modes
assert state.set_render_mode("standard_2d") == "standard_2d"
assert state.get_render_mode() == "standard_2d"
assert state.get_engine_render_mode() == "standard"

assert state.set_render_mode("estimated_2_5d") == "estimated_2_5d"
assert state.get_render_mode() == "estimated_2_5d"
assert state.get_engine_render_mode() == "pose_3d"

assert state.set_render_mode("full_3d_reconstructed") == "full_3d_reconstructed"
assert state.get_render_mode() == "full_3d_reconstructed"
assert state.get_engine_render_mode() == "reenactment_3d"

# Reset to standard_2d
state.set_render_mode("standard_2d")

# Test Source Faces
state.clear_source_faces()
assert len(state.get_source_faces()) == 0
state.add_source_face({"face_id": "face_1", "path": "/path/to/face1.png"})
state.add_source_face({"face_id": "face_2", "path": "/path/to/face2.png"})
assert len(state.get_source_faces()) == 2
assert state.get_source_faces()[0]["face_id"] == "face_1"

# Test Facesets
state.set_facesets([{"faceset_id": "fsz_akansha", "name": "Akansha", "face_count": 12}])
assert len(state.get_facesets()) == 1
assert state.get_facesets()[0]["faceset_id"] == "fsz_akansha"

# Test Target Media
state.set_target_media({"path": "target.mp4", "media_type": "video", "fps": 29.97, "frame_count": 300, "duration": 10.01})
target = state.get_target_media()
assert target is not None and target["path"] == "target.mp4"
assert target["fps"] == 29.97

# Test Active Processors
state.set_active_processors(["face_swapper", "face_enhancer", "occlusion_masker"])
processors = state.get_active_processors()
assert "face_swapper" in processors and "face_enhancer" in processors and "occlusion_masker" in processors

# Test Snapshot to_dict
snap = state.to_dict()
assert snap["render_mode"] == "standard_2d"
assert snap["engine_render_mode"] == "standard"
assert len(snap["source_faces"]) == 2
assert len(snap["facesets"]) == 1
assert snap["target_media"]["path"] == "target.mp4"
assert len(snap["active_processors"]) == 3
print("      Unified State Store verified.", flush=True)

# 5. Test Live HTTP In-Memory Client
print("[5/5] Testing FastAPI endpoints via TestClient...", flush=True)
from starlette.testclient import TestClient
client = TestClient(app, base_url="http://127.0.0.1:8030", raise_server_exceptions=True)

# Health endpoint
health_resp = client.get("/api/v1/health")
assert health_resp.status_code == 200, f"Health status: {health_resp.status_code}"
assert health_resp.json()["status"] == "ok"

# Unified State GET endpoint
state_resp = client.get("/api/v1/state")
assert state_resp.status_code == 200
assert state_resp.json()["ok"] is True
assert state_resp.json()["state"]["render_mode"] == "standard_2d"

# Render Mode POST and GET endpoints
post_mode_resp = client.post("/api/v1/render-mode", json={"render_mode": "estimated_2_5d"})
assert post_mode_resp.status_code == 200
assert post_mode_resp.json()["render_mode"] == "estimated_2_5d"
assert post_mode_resp.json()["engine_mode"] == "pose_3d"

get_mode_resp = client.get("/api/v1/render-mode")
assert get_mode_resp.status_code == 200
assert get_mode_resp.json()["render_mode"] == "estimated_2_5d"

post_mode_3d = client.post("/api/v1/render-mode", json={"render_mode": "full_3d_reconstructed"})
assert post_mode_3d.status_code == 200
assert post_mode_3d.json()["render_mode"] == "full_3d_reconstructed"
assert post_mode_3d.json()["engine_mode"] == "reenactment_3d"

# Aliased endpoint GET /api/state
alias_state_resp = client.get("/api/state")
assert alias_state_resp.status_code == 200
assert alias_state_resp.json()["ok"] is True

# Reset render mode back to standard_2d
client.post("/api/v1/render-mode", json={"render_mode": "standard_2d"})

print("      All HTTP endpoint tests passed.", flush=True)

print("\n" + "=" * 70)
print("STAGE 1 VERIFICATION SUCCESS: ZERO LEGACY PROTECTION BLOCKS REMAIN,")
print("FASTAPI ROUTES DIRECTLY MOUNTED, UNIFIED STATE STORE OPERATIONAL.")
print("=" * 70)
sys.exit(0)
