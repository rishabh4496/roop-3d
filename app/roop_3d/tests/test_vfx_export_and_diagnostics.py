"""Comprehensive unit and integration test suite for Stage 12:
VFX matte pass exports, NLE timeline interchange, cryptographic provenance,
and 14-section diagnostics report.
"""
from fractions import Fraction
import json
from pathlib import Path
import socket
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient
import numpy as np
import pytest

from roop_3d.tests.test_stage14_workstation import project_fixture
from roop_3d.vfx_export import (
    combine_alpha_matte,
    export_alpha_matte,
    extract_foreground_pass,
    VfxPassCollector,
    build_timeline_from_prescan,
    export_fcpxml,
    export_xmeml,
    export_edl,
    create_provenance_manifest,
    save_provenance_sidecar,
    verify_provenance_sidecar,
)
from roop_3d.server.diagnostics import (
    generate_14_section_report,
    get_environment_health_card,
    purge_idle_vram,
    check_offline_compatibility,
)


def test_combine_alpha_matte_and_precision():
    # 64x64 geometric mask with center 1.0, background 0.0
    geom = np.zeros((64, 64), dtype=np.float32)
    geom[16:48, 16:48] = 1.0

    # Occlusion over top half of face (should subtract)
    occlusion = np.zeros((64, 64), dtype=np.float32)
    occlusion[16:32, 16:48] = 1.0

    # Brush override adding back a small portion in top-left
    brush = np.zeros((64, 64), dtype=np.float32)
    brush[20:25, 20:25] = 1.0

    combined = combine_alpha_matte(geom, occlusion_matte=occlusion, brush_override=brush, feather_radius=1)
    assert combined.shape == (64, 64)
    assert combined.dtype == np.float32
    assert 0.0 <= combined.min() <= combined.max() <= 1.0

    # Occluded region should have lower alpha than unoccluded bottom half
    assert combined[24, 32] < combined[40, 32]
    # Brush region should have restored alpha
    assert combined[22, 22] > combined[24, 32]

    # Test 16-bit and 8-bit quantization
    matte_16 = export_alpha_matte(combined, bit_depth=16)
    assert matte_16.dtype == np.uint16
    assert matte_16.max() > 255

    matte_8 = export_alpha_matte(combined, bit_depth=8)
    assert matte_8.dtype == np.uint8
    assert matte_8.max() <= 255


def test_extract_foreground_pass():
    swap_rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[8:24, 8:24] = 1.0

    # RGBA 8-bit mode
    rgba = extract_foreground_pass(swap_rgb, alpha, output_format="RGBA", bit_depth=8)
    assert rgba.shape == (32, 32, 4)
    assert rgba.dtype == np.uint8
    assert (rgba[16, 16, :3] == [200, 200, 200]).all()
    assert rgba[16, 16, 3] == 255
    assert rgba[0, 0, 3] == 0

    # RGB + Matte 16-bit mode
    rgb_16, matte_16 = extract_foreground_pass(swap_rgb, alpha, output_format="RGB_MATTE", bit_depth=16)
    assert rgb_16.dtype == np.uint16
    assert matte_16.dtype == np.uint16
    assert matte_16[16, 16] == 65535
    assert matte_16[0, 0] == 0


def test_vfx_pass_collector(tmp_path):
    collector = VfxPassCollector(shape=(48, 64), selected_passes=("alpha", "foreground", "occluder"))
    dummy_alpha = np.ones((48, 64), dtype=np.float32)
    dummy_rgb = np.full((48, 64, 3), 150, dtype=np.uint8)

    collector.observe("alpha", dummy_alpha)
    collector.observe("foreground", dummy_rgb, alpha=dummy_alpha)
    collector.observe("occluder", np.zeros((48, 64), dtype=np.float32))

    checkpoints = collector.save_checkpoint(tmp_path, "job_test_123", frame_index=0)
    assert "alpha" in checkpoints
    assert "foreground" in checkpoints
    assert (tmp_path / checkpoints["alpha"]).is_file()
    assert (tmp_path / checkpoints["foreground"]).is_file()


def test_nle_timeline_interchange_formats():
    prescan_data = {
        "name": "VFX Scene 01",
        "frame_rate": "24000/1001",
        "frame_count": 240,
        "width": 1920,
        "height": 1080,
        "swapped_ranges": [[24, 96], [144, 200]],
        "scene_cuts": [0, 120, 240],
        "low_confidence_frames": [{"frame": 50, "confidence": 0.42}],
        "occlusion_frames": [160],
    }

    plan = build_timeline_from_prescan(
        prescan_data,
        source_url="file:///media/source.mov",
        swapped_url="file:///media/swapped.mov",
        alpha_url="file:///media/alpha.mkv",
        confidence_threshold=0.5,
    )

    assert plan.frame_count == 240
    assert len(plan.markers) == 2

    # 1. Test FCPXML
    fcpxml_str = export_fcpxml(plan)
    assert "<?xml version=" in fcpxml_str
    assert "<!DOCTYPE fcpxml>" in fcpxml_str
    fcpxml_doc = ET.fromstring(fcpxml_str.split("\n", 2)[2])
    assert fcpxml_doc.tag == "fcpxml"
    assert fcpxml_doc.attrib.get("version") == "1.10"
    # Verify spine contains asset-clips with chapter-markers
    chapter_markers = fcpxml_doc.findall(".//chapter-marker")
    assert len(chapter_markers) >= 2
    lanes = [clip.attrib.get("lane") for clip in fcpxml_doc.findall(".//asset-clip")]
    assert "1" in lanes  # Track 2 swapped clips
    assert "2" in lanes  # Track 3 alpha clips

    # 2. Test XMEML (FCP 7 / Premiere XML)
    xmeml_str = export_xmeml(plan)
    assert "<?xml version=" in xmeml_str
    assert "<!DOCTYPE xmeml>" in xmeml_str
    xmeml_doc = ET.fromstring(xmeml_str.split("\n", 2)[2])
    assert xmeml_doc.tag == "xmeml"
    tracks = xmeml_doc.findall(".//track")
    assert len(tracks) >= 2
    markers = xmeml_doc.findall(".//marker")
    assert len(markers) >= 2

    # 3. Test CMX 3600 EDL
    edls = export_edl(plan)
    assert "V1.edl" in edls
    assert "V2.edl" in edls
    assert "V3.edl" in edls
    assert "TITLE: VFX Scene 01 V1 Original Source" in edls["V1.edl"]
    assert "001  SWAP     V     C" in edls["V2.edl"]
    assert "MARKER" in edls["V2.edl"]


def test_cryptographic_provenance_manifest_and_verification(tmp_path):
    media_file = tmp_path / "swapped_output.mov"
    media_file.write_bytes(b"SYNTHETIC_MEDIA_TEST_BYTES_FOR_PROVENANCE")

    import hashlib
    real_hash = hashlib.sha256(media_file.read_bytes()).hexdigest()

    recipe = {
        "target": "source.mov",
        "faceset": "alice",
        "workstation_parameters": {"restoration_blend": 0.4},
    }
    model_hashes = {
        "inswapper_128.onnx": "a" * 64,
        "yoloface_8n.onnx": "b" * 64,
        "gpen_bfr_512.onnx": "c" * 64,
    }

    # Generate manifest with real hash
    manifest = create_provenance_manifest(
        recipe=recipe,
        model_hashes=model_hashes,
        output_sha256=real_hash,
        signing_key="secret-key-at-least-32-bytes-long-123456",
    )
    assert manifest["c2pa_compatible"] is True
    assert "pipeline_stage_versions" in manifest
    assert "model_hashes" in manifest
    assert manifest["signature"]["algorithm"] == "hmac-sha256"

    # Save sidecar
    sidecar_path = save_provenance_sidecar(media_file, manifest)
    assert sidecar_path.is_file()

    # Verify sidecar
    res = verify_provenance_sidecar(
        media_file,
        sidecar_path,
        signing_key="secret-key-at-least-32-bytes-long-123456",
    )
    assert res["media_hash_matches"] is True
    assert res["signature_valid"] is True
    assert res["valid"] is True

    # Tampering test: tamper with media content
    tampered_media = tmp_path / "tampered.mov"
    tampered_media.write_bytes(b"TAMPERED_MEDIA_BYTES")
    tampered_res = verify_provenance_sidecar(
        tampered_media,
        sidecar_path,
        signing_key="secret-key-at-least-32-bytes-long-123456",
    )
    assert tampered_res["media_hash_matches"] is False
    assert tampered_res["valid"] is False


def test_diagnostics_report_14_sections_and_zero_missing_keys(tmp_path):
    app, payload, model = project_fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        # Test GET /api/system/diagnostics_report (aliased by ApiAliasMiddleware)
        resp = client.get("/api/system/diagnostics_report")
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "sections" in data
        assert len(data["sections"]) == 14

        # Verify all 14 sections and their required keys
        sec_map = {s["section"]: s for s in data["sections"]}
        for idx in range(1, 15):
            assert idx in sec_map, f"Missing section {idx}"
            sec = sec_map[idx]
            assert "title" in sec
            assert "status" in sec

        # Section 1: OS, Python, Node versions, environment integrity
        s1 = sec_map[1]
        assert "python_version" in s1
        assert "os" in s1
        assert "platform" in s1
        assert "node_version" in s1
        assert "env_path_status" in s1

        # Section 2: GPU topology and memory pressure gauge
        s2 = sec_map[2]
        assert "cpu_name" in s2
        assert "primary_gpu" in s2
        assert "gpu_topology" in s2
        assert "memory_pressure_gauge" in s2

        # Section 3: ORT Execution Providers and DLL alignment
        s3 = sec_map[3]
        assert "advertised_providers" in s3
        assert "preferred_provider" in s3
        assert "dll_alignment" in s3

        # Section 4: TensorRT Engine Cache validity
        s4 = sec_map[4]
        assert "cached_engines" in s4
        assert "engine_cache_validity" in s4

        # Section 5: FFmpeg codecs
        s5 = sec_map[5]
        assert "nvenc_encoders" in s5
        assert "nvenc_active" in s5

        # Section 6: Model Checksums
        s6 = sec_map[6]
        assert "registered_models_count" in s6
        assert "checksums" in s6
        assert "models_dir_status" in s6

        # Section 7: Prepass Cache hits
        s7 = sec_map[7]
        assert "prepass_cache_hits" in s7

        # Section 8: Stage 0 Matrix status
        s8 = sec_map[8]
        assert "stage0_matrix_status" in s8
        assert "populated_matrices_count" in s8

        # Section 10: Watchdog Seam logs
        s10 = sec_map[10]
        assert "watchdog_seam_logs" in s10

        # Section 12: VRAM Governor & memory pressure
        s12 = sec_map[12]
        assert "memory_pressure_gauge" in s12
        assert "idle_vram_purge_available" in s12

        # Section 13: Per-Node Latency Breakdown
        s13 = sec_map[13]
        assert "per_node_latency_breakdown" in s13
        pnl = s13["per_node_latency_breakdown"]
        assert "decode_ms" in pnl
        assert "detection_ms" in pnl
        assert "inswapper_ms" in pnl

        # Section 14: Strict Offline Verification & Boundary Isolation
        s14 = sec_map[14]
        assert s14["zero_external_calls"] is True
        assert s14["offline_compatibility_status"] == "UNVERIFIED"
        assert s14["app_env_status"] in ("VERIFIED_DIR", "MISSING", "SYMLINK_VALID")

        # Test Standing Environment-Health Card endpoint
        eh_resp = client.get("/api/system/environment_health")
        assert eh_resp.status_code == 200, eh_resp.text
        eh_data = eh_resp.json()
        assert "app_env" in eh_data
        assert "app_models" in eh_data
        assert "app_facesets" in eh_data
        assert "ffmpeg" in eh_data
        assert "dll_alignment" in eh_data
        assert "engine_cache" in eh_data
        assert "gpu_hud" in eh_data

        # Test Purge Idle VRAM endpoint
        purge_resp = client.post("/api/system/purge_vram")
        assert purge_resp.status_code == 200, purge_resp.text
        purge_data = purge_resp.json()
        assert purge_data["status"] == "ok"
        assert "freed_vram_mib" in purge_data

        # Test Offline-Safe Compatibility Check endpoint
        compat_resp = client.post("/api/system/check_compatibility", json={"force_remote": False})
        assert compat_resp.status_code == 200, compat_resp.text
        compat_data = compat_resp.json()
        assert compat_data["status"] == "UNVERIFIED"
        assert compat_data["offline_compliant"] is True


def test_zero_external_network_requests_during_boot_diagnostics_and_offline_vfx(tmp_path, monkeypatch):
    """Verify that during full app boot, diagnostics collection, and offline VFX
    rendering, strictly zero external (non-loopback) network connections are made.
    """
    external_calls = []
    orig_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) and len(address) > 0 else str(address)
        if host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0"):
            external_calls.append(address)
            raise ConnectionRefusedError(f"Prohibited external network call to {address}")
        return orig_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # 1. Full app boot
    app, payload, model = project_fixture(tmp_path)

    # 2. Diagnostics report and environment health collection
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        resp = client.get("/api/system/diagnostics_report")
        assert resp.status_code == 200
        eh = client.get("/api/system/environment_health")
        assert eh.status_code == 200
        compat = client.post("/api/system/check_compatibility", json={"force_remote": False})
        assert compat.status_code == 200
        assert compat.json()["status"] == "UNVERIFIED"

    # 3. Offline VFX Matte and foreground pass execution
    geom = np.ones((32, 32), dtype=np.float32)
    alpha = combine_alpha_matte(geom)
    rgba = extract_foreground_pass(np.zeros((32, 32, 3), dtype=np.uint8), alpha)
    assert rgba.shape == (32, 32, 4)

    # 4. Generate report direct module call
    report = generate_14_section_report(root=tmp_path, state=app.state)
    assert len(report["sections"]) == 14

    # Assert zero external socket connections
    assert len(external_calls) == 0, f"Detected external network connection attempts: {external_calls}"
