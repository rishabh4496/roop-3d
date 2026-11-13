"""Verification Gate & Regression Suite for Stage 15 Heterogeneous Scheduler.

Verifies:
(1) Stage 1/2 pre-scan executing on OpenVINO allows concurrent POST /api/preview
    execution on TensorRT with 0ms GPU lock contention (zero HTTP 503 busy waits).
(2) The 3D Self-Occlusion Visibility Map accurately masks out the far cheek on a
    60° yaw head rotation in <0.8 ms per frame.
(3) Intel QuickSync Video (QSV) hardware detection and 4K decode offloading.
(4) Asynchronous lookahead auxiliary model offloading (Frame N+2 lookahead).
"""
from __future__ import annotations

from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from roop_3d.core.hardware import probe_hardware
from roop_3d.core.settings import Settings
from roop_3d.heterogeneous_scheduler import (
    HeterogeneousScheduler,
    VideoCodecMultiplexer,
)
from roop_3d.openvino_coprocessor import (
    AuxiliaryLookaheadWorker,
    OpenVINOCoprocessor,
    get_openvino_device_info,
)
from roop_3d.reconstruction.tps_fallback import build_canonical_head_mesh
from roop_3d.server.app import create_app
from roop_3d.server.engine_gate import EngineBusy, EngineGate
from roop_3d.tests.test_workstation_preview import preview_fixture


def live_payload(fixture_payload: dict, **extra) -> dict:
    request = fixture_payload["request"]
    return {
        "target_path": request["target_path"],
        "inspection_id": request["inspection_id"],
        "face_mappings": request["face_mappings"],
        "frame_index": 2,
        **extra,
    }


# ==============================================================================
# Gate Verification 1: Concurrent Preview + Prepass with 0ms GPU Lock Contention
# ==============================================================================

def test_stage1_2_prescan_on_openvino_allows_concurrent_preview_with_zero_gpu_contention(tmp_path: Path, monkeypatch):
    """VERIFICATION GATE (1):

    Stage 1/2 pre-scan executing on OpenVINO allows concurrent POST /api/preview
    execution on TensorRT with 0ms GPU lock contention (zero HTTP 503 busy waits).
    """
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    app.state.settings = app.state.settings.model_copy(update={'heterogeneous_split_mode': True})
    gate: EngineGate = app.state.engine_gate
    scheduler = HeterogeneousScheduler(settings=app.state.settings, engine_gate=gate)
    payload = live_payload(fixture)

    prescan_started = threading.Event()
    prescan_hold = threading.Event()
    prescan_finished = threading.Event()
    prescan_error: list[Exception] = []

    def prescan_worker():
        try:
            # Simulate Stage 1/2 pre-scan running on OpenVINO
            # In split mode, acquires domain='openvino' (never blocks GPU)
            with gate.owner("STAGE_1_SPARSE_PRESCAN", domain="openvino"):
                prescan_started.set()
                # Hold pre-scan active while the concurrent preview is tested
                prescan_hold.wait(timeout=5.0)
            prescan_finished.set()
        except Exception as e:
            prescan_error.append(e)

    # Launch pre-scan on OpenVINO in background thread
    t = threading.Thread(target=prescan_worker, daemon=True)
    t.start()
    assert prescan_started.wait(timeout=3.0), "Pre-scan failed to start"

    # Verify OpenVINO owner is held, while GPU has 0 owners
    assert gate.owner_count_for_domain("openvino") == 1
    assert gate.owner_count_for_domain("gpu") == 0

    # Measure GPU preview lock acquisition latency / contention
    t_start = time.perf_counter()
    with gate.preview_section(domain="gpu"):
        contention_ms = (time.perf_counter() - t_start) * 1000.0

    # Contention must be 0ms (< 1ms threshold for thread scheduling)
    assert contention_ms < 1.0, f"GPU lock contention was {contention_ms:.3f} ms (expected ~0ms)"

    # Test concurrent preview via TestClient
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        # Request workstation preview during active OpenVINO pre-scan
        resp = client.post("/api/preview", json=payload)
        # Crucial: Must NEVER receive HTTP 503 ENGINE_BUSY during OpenVINO pre-scan!
        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 100

    # Release pre-scan
    prescan_hold.set()
    t.join(timeout=3.0)
    assert prescan_finished.is_set()
    assert not prescan_error


def test_monolithic_mode_rejects_preview_when_gpu_is_locked(tmp_path: Path, monkeypatch):
    """Negative contrast verification:

    When heterogeneous_split_mode is False, pre-scan holding the GPU owner
    causes concurrent preview attempts to fail with HTTP 503 ENGINE_BUSY.
    """
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    app.state.settings = app.state.settings.model_copy(update={'heterogeneous_split_mode': False})  # Legacy monolithic mode
    gate: EngineGate = app.state.engine_gate
    payload = live_payload(fixture)

    # In monolithic mode, prepass acquires domain='gpu'
    app.state.pipeline.render_event("job-x", {"stage": "STAGE_1_SPARSE_PRESCAN", "state": "RUNNING"})
    token = gate.acquire_owner("RENDER_JOB", domain="gpu")
    try:
        assert gate.owner_count_for_domain("gpu") == 1
        # Previews targeting GPU must be refused immediately with EngineBusy
        assert not gate.try_enter_preview(domain="gpu")
        with pytest.raises(EngineBusy):
            with gate.preview_section(domain="gpu"):
                pass

        # FastAPI endpoint returns 503
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            resp = client.post("/api/preview", json=payload)
            assert resp.status_code == 503
            assert resp.json()["busy"] is True
    finally:
        gate.release_owner(token)


# Gate Verification 2 (3D self-occlusion) lives in test_w2_3d_visibility.py.


# ==============================================================================
# Dual-Hardware Video Decode/Encode Multiplexer & Hardware Probe
# ==============================================================================

def test_hardware_probe_and_api_system_hardware_detects_qsv_and_openvino(tmp_path: Path):
    """Verify GET /api/system/hardware exposes QSV and OpenVINO hardware capabilities."""
    settings = Settings(heterogeneous_split_mode=True)
    app = create_app(settings=settings, repo_root=tmp_path)

    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        resp = client.get("/api/v1/system/hardware")
        assert resp.status_code == 200
        hw = resp.json()

        # Check required QSV fields
        assert "qsv_available" in hw
        assert "ffmpeg_qsv_encoders" in hw
        assert "ffmpeg_qsv_decoders" in hw

        # Check OpenVINO fields
        assert "openvino_available" in hw
        assert "openvino_devices" in hw
        assert "heterogeneous_split_mode" in hw
        assert hw["heterogeneous_split_mode"] is True

        # Check unversioned endpoint parity
        resp_unversioned = client.get("/api/system/hardware")
        assert resp_unversioned.status_code == 200
        assert resp_unversioned.json()["heterogeneous_split_mode"] is True


def test_video_codec_multiplexer_routes_4k_to_qsv():
    """Verify multiplexer offloads 4K decode to QSV when available or saturated."""
    multiplexer = VideoCodecMultiplexer()
    # Mock QSV decoder presence for test
    multiplexer.qsv_available = True
    multiplexer.qsv_decoders = ["h264_qsv", "hevc_qsv"]

    settings = Settings(heterogeneous_split_mode=True)

    # 4K resolution -> routes to QSV
    assert multiplexer.should_offload_to_qsv(3840, 2160, face_count=1, settings=settings) is True
    assert multiplexer.get_decoder_codec_name("h264", offload_qsv=True) == "h264_qsv"

    # Multi-face heavy saturation -> routes to QSV
    assert multiplexer.should_offload_to_qsv(1920, 1080, face_count=4, settings=settings) is True

    # Low-res single face normal load -> stays on NVDEC / reference
    assert multiplexer.should_offload_to_qsv(1280, 720, face_count=1, nvdec_saturated=False) is False


# ==============================================================================
# OpenVINO Asynchronous Lookahead Worker & Co-Processor
# ==============================================================================

def test_auxiliary_lookahead_worker_processes_frame_n_plus_2():
    """Verify asynchronous lookahead worker offloads auxiliary models in parallel."""
    worker = AuxiliaryLookaheadWorker(target_device="CPU", precision="INT8")
    worker.start()
    assert worker.is_running()

    try:
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Submit lookahead for Frame 2 (N+2) while Frame 0 is swapped
        submitted = worker.submit_lookahead(2, dummy_img)
        assert submitted is True

        # Retrieve auxiliary result for Frame 2
        result = worker.get_auxiliary_result(2, timeout=1.0)
        assert result is not None
        assert result["frame_index"] == 2
        assert "parsing_mask" in result
        assert result["face_parsing_classes"] == 19
        assert "gaze_vector" in result
        assert len(result["gaze_vector"]) == 3
        assert worker.frames_processed >= 1
    finally:
        worker.stop()
        assert not worker.is_running()


def test_heterogeneous_scheduler_end_to_end_frame_execution():
    """Verify HeterogeneousScheduler orchestrates TensorRT, OpenVINO, and SIMD raycaster."""
    settings = Settings(heterogeneous_split_mode=True)
    gate = EngineGate()
    scheduler = HeterogeneousScheduler(settings=settings, engine_gate=gate)
    mesh = build_canonical_head_mesh(68)

    dummy_frame = np.zeros((512, 512, 3), dtype=np.uint8)

    # Frame 0 at 60° yaw head rotation
    output = scheduler.execute_stage3_frame(
        frame_index=0,
        frame_image=dummy_frame,
        mesh=mesh,
        head_yaw=60.0,
        swap_func=lambda img: img + 10,
    )

    assert output["frame_index"] == 0
    # canonical head at +60 deg yaw: the far cheek and part of the face are self-occluded
    assert output["self_occluded_fraction"] > 0.2
    assert output["visibility_result"].vertex_visible.shape == (len(mesh.vertices),)
    assert output["executed_devices"]["swap"] == "nvidia_tensorrt"
    assert output["executed_devices"]["auxiliary"] == "intel_openvino"
    assert output["executed_devices"]["raycast"] == "cpu_zbuffer"
    assert scheduler.total_swaps == 1

    telemetry = scheduler.get_hardware_telemetry()
    assert telemetry["heterogeneous_split_mode"] is True
    assert telemetry["self_occlusion"]["last_execution_ms"] > 0.0
