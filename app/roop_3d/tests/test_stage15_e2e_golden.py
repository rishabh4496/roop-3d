"""Stage 15: Golden-Clip End-to-End Video Swap Smoke Test.

Executes real, unmocked local model inference through worker.py and pipeline_runner.py,
verifying end-to-end face replacement, audio preservation, background preservation,
and output manifest model hashes.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import zipfile

import av
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.audit.project_paths import legacy_root
from roop_3d.audit.isolation import sha256_file
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.media.probe import probe
from roop_3d.server.app import create_app
from roop_3d.tests.model_governance_support import require_reference_models


def create_golden_video(path: Path, face_image: np.ndarray, *, count: int = 10, fps: int = 25) -> Path:
    """Create a short synthetic video clip containing a consented face fixture and synthesized audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 512, 512
    # Ensure face image is sized nicely inside the center canvas
    resized_face = cv2.resize(face_image, (384, 384), interpolation=cv2.INTER_AREA)

    with av.open(str(path), "w") as output:
        # Video stream
        vstream = output.add_stream("libx264", rate=fps)
        vstream.width, vstream.height, vstream.pix_fmt = width, height, "yuv420p"
        vstream.time_base = vstream.codec_context.time_base = Fraction(1, fps)
        vstream.codec_context.max_b_frames = 0

        # Audio stream: 48000 Hz, stereo AAC
        astream = output.add_stream("aac", rate=48000)
        astream.time_base = Fraction(1, 48000)
        astream.layout = "stereo"
        astream.format = "fltp"

        # Generate video frames: solid green border, centered face image
        for i in range(count):
            canvas = np.zeros((height, width, 3), np.uint8)
            canvas[:, :] = [0, 200, 0]  # distinct solid background
            canvas[64:448, 64:448] = resized_face
            vframe = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            vframe.pts, vframe.time_base = i, Fraction(1, fps)
            for packet in vstream.encode(vframe):
                output.mux(packet)

        # Generate audio frames matching the duration (count / fps seconds)
        total_samples = int(48000 * (count / fps))
        t = np.linspace(0, count / fps, total_samples, endpoint=False)
        # 440 Hz sine wave in float32 within [-0.5, 0.5]
        sine_wave = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        stereo_audio = np.stack([sine_wave, sine_wave], axis=0)  # shape (2, total_samples) for fltp

        aframe = av.AudioFrame.from_ndarray(stereo_audio, format="fltp", layout="stereo")
        aframe.sample_rate = 48000
        aframe.pts, aframe.time_base = 0, Fraction(1, 48000)
        for packet in astream.encode(aframe):
            output.mux(packet)

        # Flush streams
        for packet in vstream.encode(None):
            output.mux(packet)
        for packet in astream.encode(None):
            output.mux(packet)

    return path


def _wait_state(client: TestClient, job_id: str, state: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/api/v1/jobs/{job_id}").json()
        if record["state"] == state:
            return record
        if record["state"] in {"failed", "cancelled"}:
            raise AssertionError(f"Job entered {record['state']}: {record.get('failure')}")
        time.sleep(0.05)
    raise AssertionError(f"Job did not reach {state} within {timeout}s; last state={record.get('state')}")


def test_stage15_golden_clip_end_to_end_swap(tmp_path: Path):
    """Run real unmocked video face swap on a short golden clip and assert all Tier 2 contracts."""
    require_reference_models()
    # 1. Read consented face fixtures from verified local storage
    reader = FaceSetReader()
    source_fsz = legacy_root() / "app/facesets/akansha.fsz"
    target_fsz = legacy_root() / "app/facesets/anushree.fsz"
    assert source_fsz.is_file(), f"Source FaceSet {source_fsz} not found"
    assert target_fsz.is_file(), f"Target FaceSet {target_fsz} not found"

    source_doc = reader.read(source_fsz)
    target_doc = reader.read(target_fsz)

    # Get canonical face crop from target
    target_view = min(target_doc.views, key=lambda v: abs(v.yaw))
    target_face = cv2.imdecode(np.frombuffer(target_view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert target_face is not None, "Failed to decode target face image"

    # 2. Create the golden test video with video frames and audio
    target_relative = "roop_3d/.runtime/media/golden_target.mp4"
    target_path = tmp_path / target_relative
    num_frames = 4
    create_golden_video(target_path, target_face, count=num_frames, fps=25)
    assert target_path.is_file()

    # Verify input audio exists
    input_info = probe(target_path)
    assert len(input_info.audio) >= 1, "Input video must contain at least 1 audio track"
    assert len(input_info.videos) == 1
    assert input_info.videos[0].frame_count == num_frames

    # 3. Setup app and register the source FaceSet
    settings = Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider",
                        audio_policy="copy")
    app = create_app(settings, repo_root=tmp_path, worker_enabled=True)

    # Register akansha FaceSet in the app's runtime store
    fsz_dest = tmp_path / "roop_3d/outputs/facesets/akansha.fsz"
    fsz_dest.parent.mkdir(parents=True, exist_ok=True)
    fsz_dest.write_bytes(source_fsz.read_bytes())
    atomic_json(tmp_path, "roop_3d/.runtime/facesets/akansha.json", {
        "faceset_id": "akansha",
        "name": "Akansha",
        "item_count": len(source_doc.views),
        "schema_version": 1,
    })

    # 4. Inspect the target media using the API
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        inspect_resp = client.post("/api/v1/media/inspect-faces", json={"path": target_relative})
        assert inspect_resp.status_code == 200, f"Inspect failed: {inspect_resp.text}"
        inspection_data = inspect_resp.json()
        assert len(inspection_data["tracks"]) >= 1, "Face detector must find at least 1 face track in golden target"
        target_track_id = inspection_data["tracks"][0]["track_id"]
        inspection_id = inspection_data["inspection_id"]

        # 5. Submit real swap job via API
        payload = {
            "target_path": target_relative,
            "inspection_id": inspection_id,
            "face_mappings": [{"track_id": target_track_id, "faceset_id": "akansha"}],
        }
        response = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "stage15-golden-e2e"})
        assert response.status_code == 201, f"Job creation failed: {response.text}"
        job_id = response.json()["job_id"]

        # Wait for the worker to complete the job
        completed_record = _wait_state(client, job_id, "succeeded", timeout=180.0)
        assert completed_record["state"] == "succeeded"

    # 6. Output Assertions
    # (a) Output .mp4 exists in roop_3d/outputs/
    output_rel = completed_record["output"]["output_path"]
    output_file = tmp_path / output_rel
    assert output_file.is_file(), f"Output file {output_file} does not exist"

    manifest_rel = completed_record["output"]["manifest_path"]
    manifest_file = tmp_path / manifest_rel
    assert manifest_file.is_file(), f"Manifest file {manifest_file} does not exist"

    # (b) Decodes cleanly via PyAV, matches source frame count and duration
    with av.open(str(output_file)) as out_container:
        assert len(out_container.streams.video) == 1
        out_vstream = out_container.streams.video[0]
        out_frames = list(out_container.decode(out_vstream))
        assert len(out_frames) == num_frames, f"Expected {num_frames} frames, got {len(out_frames)}"

        # Verify rational duration matches
        expected_duration = Fraction(num_frames, 25)
        observed_duration = out_vstream.duration * out_vstream.time_base
        assert abs(float(observed_duration - expected_duration)) < 0.1

        # Verify audio stream is preserved
        assert len(out_container.streams.audio) >= 1, "Output video missing audio track"

    # Reopen container to decode audio (PyAV demux state was consumed decoding video)
    with av.open(str(output_file)) as audio_container:
        assert len(audio_container.streams.audio) >= 1
        out_astream = audio_container.streams.audio[0]
        out_audio_frames = list(audio_container.decode(out_astream))
        assert len(out_audio_frames) > 0, "Output audio stream contains no decodable audio frames"

    # Read original input frames for pixel comparison
    with av.open(str(target_path)) as in_container:
        in_frames = list(in_container.decode(in_container.streams.video[0]))

    # Verify background pixels unchanged & face region changed
    for frame_idx, (in_f, out_f) in enumerate(zip(in_frames, out_frames)):
        in_img = in_f.to_ndarray(format="bgr24")
        out_img = out_f.to_ndarray(format="bgr24")
        assert in_img.shape == out_img.shape == (512, 512, 3)

        # Background check: top-left corner (0:32, 0:32) was solid green [0, 200, 0]
        bg_diff = np.abs(out_img[0:32, 0:32].astype(int) - in_img[0:32, 0:32].astype(int))
        # Video compression (H.264 YUV420P) may introduce slight rounding, max diff should be small
        assert np.mean(bg_diff) < 3.0, f"Background pixels modified in frame {frame_idx}: mean diff={np.mean(bg_diff)}"

        # Face region check: center face crop (160:350, 160:350)
        face_diff = np.abs(out_img[160:350, 160:350].astype(int) - in_img[160:350, 160:350].astype(int))
        assert np.max(face_diff) > 20, f"Face region not changed in frame {frame_idx}: max diff={np.max(face_diff)}"
        assert np.mean(face_diff) > 5.0, f"Face region mean diff too low in frame {frame_idx}: mean diff={np.mean(face_diff)}"

    # (c) Output manifest assertions
    manifest = json.loads(manifest_file.read_text("utf-8"))
    assert manifest["output_sha256"] == sha256_file(output_file)[1]
    assert manifest["verified_frames"] == num_frames
    assert manifest["audio_policy"] == "copy"
    assert len(manifest["audio_tracks"]) >= 1

    # Check model hashes in manifest
    model_hashes = manifest.get("model_hashes") or manifest.get("stage12", {}).get("model_hashes", {})
    assert isinstance(model_hashes, dict), "Manifest must contain model_hashes dictionary"
    assert "identity_swapper" in model_hashes, f"identity_swapper missing from model_hashes: {model_hashes}"
    swapper_hash = model_hashes["identity_swapper"]
    assert len(swapper_hash) == 64 and all(c in "0123456789abcdef" for c in swapper_hash)
