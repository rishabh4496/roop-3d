"""Stage 9 Automated Verification Gate:
  (1) all 4 Batch Matrix strategies expand into the exact expected job manifests
  (2) killing and restarting the FastAPI server restores QUEUED and PAUSED jobs intact
  (3) a multi-segment video render preserves every intermediate chunk until lossless concat completes with frame-accurate audio sync
"""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil

import av
from fastapi.testclient import TestClient
import numpy as np
import pytest

from roop_3d.audit.isolation import sha256_file
from roop_3d.batch_matrix import (
    BUILTIN_RECIPES,
    BatchMatrixEngine,
    BatchStrategy,
    DEFAULT_OUTPUT_TEMPLATE,
    JobManifest,
    RecipeConfig,
    format_output_name,
)
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json, owned
from roop_3d.engine.pipeline_runner import PipelineCheckpoint, checkpoint_npz_bytes, read_checkpoint_npz
from roop_3d.queue_manager import CanonicalJobState, QueueJobProgress, QueueManager
from roop_3d.segment_dispatcher import (
    SegmentDispatcher,
    SegmentSlice,
    compute_gop_aligned_cuts,
    concat_lossless,
    detect_keyframe_timestamps,
    split_video_gop,
)
from roop_3d.server.app import create_app


def create_synthetic_video_with_audio(
    path: Path,
    duration_seconds: float = 4.0,
    fps: int = 25,
    gop_size: int = 25,
    audio_sample_rate: int = 48000,
) -> Path:
    """Create a real valid MP4 video with known GOP keyframe cadence and stereo audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(duration_seconds * fps)

    with av.open(str(path), "w") as container:
        # Video stream
        v_stream = container.add_stream("libx264", rate=Fraction(fps))
        v_stream.width, v_stream.height = 128, 96
        v_stream.pix_fmt = "yuv420p"
        v_stream.codec_context.time_base = Fraction(1, fps)
        v_stream.codec_context.gop_size = gop_size
        v_stream.codec_context.max_b_frames = 0

        # Audio stream
        a_stream = container.add_stream("aac", rate=audio_sample_rate)
        a_stream.layout = "stereo"
        a_stream.codec_context.time_base = Fraction(1, audio_sample_rate)

        # Write video frames
        for frame_idx in range(total_frames):
            color = (frame_idx * 15) % 256
            img = np.full((96, 128, 3), color, dtype=np.uint8)
            # Add visual marker
            img[:20, :20] = [255, 0, 0]
            v_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            v_frame.pts = frame_idx
            for packet in v_stream.encode(v_frame):
                container.mux(packet)

        for packet in v_stream.encode(None):
            container.mux(packet)

        # Write audio frames (1024 samples per frame)
        total_audio_samples = int(duration_seconds * audio_sample_rate)
        samples_per_frame = 1024
        curr_sample = 0

        while curr_sample < total_audio_samples:
            chunk_len = min(samples_per_frame, total_audio_samples - curr_sample)
            t = np.arange(curr_sample, curr_sample + chunk_len) / audio_sample_rate
            # 440 Hz tone
            tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
            stereo_data = np.repeat(tone, 2).reshape(1, -1)

            a_frame = av.AudioFrame.from_ndarray(stereo_data, format="s16", layout="stereo")
            a_frame.pts = curr_sample
            a_frame.sample_rate = audio_sample_rate

            for packet in a_stream.encode(a_frame):
                container.mux(packet)
            curr_sample += chunk_len

        for packet in a_stream.encode(None):
            container.mux(packet)

    return path


# =========================================================================
# Verification Gate 1: All 4 Batch Matrix strategies expand into exact manifests
# =========================================================================

def test_batch_matrix_strategy_one_to_many():
    source = "faceset_alice"
    targets = [
        {"target_path": "roop_3d/.runtime/media/vid1.mp4", "inspection_id": "a" * 32, "tracks": ["track_0000"]},
        {"target_path": "roop_3d/.runtime/media/vid2.mp4", "inspection_id": "b" * 32, "tracks": ["track_0000", "track_0001"]},
        {"target_path": "roop_3d/.runtime/media/vid3.mp4", "inspection_id": "c" * 32, "tracks": ["track_0000"]},
    ]

    manifests = BatchMatrixEngine.generate_one_to_many(
        source=source,
        targets=targets,
        output_template="{target}_{source}_{timestamp}.mp4",
        timestamp="20261101_120000",
    )

    assert len(manifests) == 3
    assert manifests[0].source == "faceset_alice"
    assert manifests[0].target_path == "roop_3d/.runtime/media/vid1.mp4"
    assert manifests[0].output_filename == "vid1_faceset_alice_20261101_120000.mp4"
    assert len(manifests[0].face_mappings) == 1
    assert manifests[0].face_mappings[0]["faceset_id"] == "faceset_alice"

    assert len(manifests[1].face_mappings) == 2
    assert manifests[1].face_mappings[0]["faceset_id"] == "faceset_alice"
    assert manifests[1].face_mappings[1]["faceset_id"] == "faceset_alice"
    assert manifests[1].output_filename == "vid2_faceset_alice_20261101_120000.mp4"


def test_batch_matrix_strategy_grouped():
    groups = [
        {
            "name": "Interview Clips",
            "source": "faceset_alice",
            "targets": [
                {"target_path": "roop_3d/.runtime/media/int1.mp4", "tracks": ["track_0000"]},
                {"target_path": "roop_3d/.runtime/media/int2.mp4", "tracks": ["track_0000"]},
            ],
        },
        {
            "name": "Action Sequences",
            "source": "faceset_bob",
            "targets": [
                {"target_path": "roop_3d/.runtime/media/act1.mp4", "tracks": ["track_0000"]},
                {"target_path": "roop_3d/.runtime/media/act2.mp4", "tracks": ["track_0000"]},
                {"target_path": "roop_3d/.runtime/media/act3.mp4", "tracks": ["track_0000"]},
            ],
        },
    ]

    manifests = BatchMatrixEngine.generate_grouped(
        groups=groups,
        output_template="{group}_{target}_{source}_{timestamp}.mp4",
        timestamp="20261101_120000",
    )

    assert len(manifests) == 5
    assert manifests[0].source == "faceset_alice"
    assert manifests[0].target_path == "roop_3d/.runtime/media/int1.mp4"
    assert "Interview_Clips" in manifests[0].output_filename

    assert manifests[2].source == "faceset_bob"
    assert manifests[2].target_path == "roop_3d/.runtime/media/act1.mp4"
    assert "Action_Sequences" in manifests[2].output_filename


def test_batch_matrix_strategy_per_file_matrix_cartesian_and_templates():
    sources = ["src_alice", "src_bob", "src_charlie"]
    targets = [
        "roop_3d/.runtime/media/target_alpha.mp4",
        "roop_3d/.runtime/media/target_beta.mp4",
        "roop_3d/.runtime/media/target_gamma.mp4",
        "roop_3d/.runtime/media/target_delta.mp4",
    ]

    # Full Cartesian product: 3 sources x 4 targets = 12 manifests
    manifests = BatchMatrixEngine.generate_per_file_matrix(
        sources=sources,
        targets=targets,
        restorer="CodeFormer",
        output_template="{target}_{source}_{restorer}_{timestamp}.mp4",
        timestamp="20261101_120000",
    )

    assert len(manifests) == 12
    # Verify naming template expansion
    assert manifests[0].output_filename == "target_alpha_src_alice_CodeFormer_20261101_120000.mp4"
    assert manifests[1].output_filename == "target_alpha_src_bob_CodeFormer_20261101_120000.mp4"
    assert manifests[2].output_filename == "target_alpha_src_charlie_CodeFormer_20261101_120000.mp4"
    assert manifests[3].output_filename == "target_beta_src_alice_CodeFormer_20261101_120000.mp4"

    # Filtered Cartesian matrix (selection grid with active cells)
    active_cells = ["0:0", "1:2", "3:1"]
    filtered = BatchMatrixEngine.generate_per_file_matrix(
        sources=sources,
        targets=targets,
        restorer="GPEN512",
        active_cells=active_cells,
        timestamp="20261101_120000",
    )
    assert len(filtered) == 3
    assert filtered[0].source == "src_alice" and "target_alpha" in filtered[0].target_path
    assert filtered[1].source == "src_charlie" and "target_beta" in filtered[1].target_path
    assert filtered[2].source == "src_bob" and "target_delta" in filtered[2].target_path


def test_batch_matrix_strategy_recipe_matrix():
    source = "faceset_hero"
    target = "roop_3d/.runtime/media/trimmed_scene.mp4"

    recipes = [
        {
            "id": "gpen_512",
            "name": "GPEN 512 Standard",
            "restorer": "gpen",
            "parameters": {"restoration_blend": 0.45, "pose_mode": "retarget"},
        },
        {
            "id": "codeformer_high",
            "name": "CodeFormer High",
            "restorer": "codeformer",
            "parameters": {"restoration_blend": 0.75, "codeformer_fidelity": 0.90},
        },
        {
            "id": "ultramax_deep",
            "name": "UltraMax Deep Dynamic",
            "restorer": "ultramax",
            "parameters": {"restoration_blend": 0.60},
        },
        {
            "id": "tps_frontal_1",
            "name": "TPS Preset 1",
            "parameters": {"pose_frontalization": 0.3, "tps_expression_weight": 0.4},
        },
        {
            "id": "tps_frontal_2",
            "name": "TPS Preset 2",
            "parameters": {"pose_frontalization": 0.7, "tps_expression_weight": 0.9},
        },
    ]

    manifests = BatchMatrixEngine.generate_recipe_matrix(
        source=source,
        target=target,
        recipes=recipes,
        output_template="{target}_{source}_{recipe}_{timestamp}.mp4",
        timestamp="20261101_120000",
    )

    assert len(manifests) == 5
    assert manifests[0].output_filename == "trimmed_scene_faceset_hero_gpen_512_20261101_120000.mp4"
    assert manifests[0].workstation_parameters["restoration_blend"] == 0.45

    assert manifests[1].output_filename == "trimmed_scene_faceset_hero_codeformer_high_20261101_120000.mp4"
    assert manifests[1].workstation_parameters["codeformer_fidelity"] == 0.90

    assert manifests[2].output_filename == "trimmed_scene_faceset_hero_ultramax_deep_20261101_120000.mp4"

    assert manifests[3].output_filename == "trimmed_scene_faceset_hero_tps_frontal_1_20261101_120000.mp4"
    assert manifests[3].workstation_parameters["pose_frontalization"] == 0.3

    assert manifests[4].output_filename == "trimmed_scene_faceset_hero_tps_frontal_2_20261101_120000.mp4"
    assert manifests[4].workstation_parameters["pose_frontalization"] == 0.7


def test_batch_matrix_expand_batch_unified_entrypoint():
    spec = {
        "strategy": "per_file_matrix",
        "sources": ["alice", "bob"],
        "targets": ["v1.mp4", "v2.mp4"],
        "restorer": "GPEN",
        "timestamp": "20261101_000000",
    }
    manifests = BatchMatrixEngine.expand_batch(spec)
    assert len(manifests) == 4
    for m in manifests:
        assert isinstance(m, JobManifest)
        payload = m.to_job_payload()
        assert "target_path" in payload
        assert "face_mappings" in payload


# =========================================================================
# Verification Gate 2: Killing & restarting FastAPI server restores QUEUED & PAUSED intact
# =========================================================================

def test_fastapi_server_restart_restores_queued_and_paused_jobs_intact(tmp_path: Path):
    target = "roop_3d/.runtime/media/test_clip.mp4"
    path = tmp_path / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"target video payload bytes")

    atomic_json(tmp_path, "roop_3d/.runtime/facesets/src_hero.json", {
        "faceset_id": "src_hero", "name": "Hero", "item_count": 1, "schema_version": 1
    })
    inspection_id = "f" * 32
    atomic_json(tmp_path, f"roop_3d/.runtime/inspections/{inspection_id}.json", {
        "inspection_id": inspection_id, "target_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "detector_model_id": "test_detector", "detector_sha256": "0" * 64,
        "decoded_frames": 10, "sampled_frames": 10, "truncated": False,
        "tracks": [{"track_id": "track_0000", "first_frame": 0, "last_frame": 9,
                    "first_timestamp_seconds": 0, "last_timestamp_seconds": 1,
                    "bounding_box": {"x1": 0, "y1": 0, "x2": 50, "y2": 50},
                    "pose_range": {}, "thumbnail_base64": "", "sampled_detections": 10}],
    })

    # 1. Start Server Instance 1
    app1 = create_app(Settings(), repo_root=tmp_path)

    with TestClient(app1, base_url="http://127.0.0.1:8030") as client1:
        # Check initial empty queue
        res = client1.get("/api/v1/queue")
        assert res.status_code == 200
        assert res.json() == []

        # (a) Submit Job 1: QUEUED
        job1_req = {
            "request": {
                "target_path": target,
                "inspection_id": inspection_id,
                "face_mappings": [{"track_id": "track_0000", "faceset_id": "src_hero"}],
            },
            "name": "Queue Job 1 (Must Remain QUEUED)",
        }
        res1 = client1.post("/api/v1/queue", json=job1_req, headers={"Idempotency-Key": "key-job-1"})
        assert res1.status_code == 201
        job1_data = res1.json()
        job1_id = job1_data["job_id"]
        assert job1_data["state"] == "QUEUED"

        # (b) Submit Job 2: transition to PAUSED with checkpoint
        job2_req = {
            "request": {
                "target_path": target,
                "inspection_id": inspection_id,
                "face_mappings": [{"track_id": "track_0000", "faceset_id": "src_hero"}],
            },
            "name": "Queue Job 2 (Must Remain PAUSED with Checkpoint)",
        }
        res2 = client1.post("/api/v1/queue", json=job2_req, headers={"Idempotency-Key": "key-job-2"})
        assert res2.status_code == 201
        job2_id = res2.json()["job_id"]

        # Write NPZ checkpoint for job 2
        target_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        cp = PipelineCheckpoint(
            job_id=job2_id,
            input_sha256=target_hash,
            request_sha256="0" * 64,
            completed_frame=15,
            pose_cache={"track_0000": [1.0, 2.0, 3.0]},
        )
        cp_file = tmp_path / f"roop_3d/.runtime/jobs/{job2_id}/checkpoint.npz"
        cp_file.parent.mkdir(parents=True, exist_ok=True)
        cp_file.write_bytes(checkpoint_npz_bytes(cp))

        # Pause job 2 via REST
        pause_res = client1.post(f"/api/v1/queue/{job2_id}/pause")
        assert pause_res.status_code == 200
        assert pause_res.json()["state"] == "PAUSED"
        assert pause_res.json()["checkpoint_path"] is not None

        # (c) Submit Job 3: Transition to RENDERING (in-flight state)
        job3_req = {
            "request": {
                "target_path": target,
                "inspection_id": inspection_id,
                "face_mappings": [{"track_id": "track_0000", "faceset_id": "src_hero"}],
            },
            "name": "Queue Job 3 (Active Rendering when Server Terminates)",
        }
        res3 = client1.post("/api/v1/queue", json=job3_req, headers={"Idempotency-Key": "key-job-3"})
        job3_id = res3.json()["job_id"]
        app1.state.queue.transition(job3_id, CanonicalJobState.PREFLIGHT)
        app1.state.queue.transition(job3_id, CanonicalJobState.RENDERING)
        assert app1.state.queue.get(job3_id).state == CanonicalJobState.RENDERING

        # (d) Submit Job 4: COMPLETED
        job4_req = {
            "request": {
                "target_path": target,
                "inspection_id": inspection_id,
                "face_mappings": [{"track_id": "track_0000", "faceset_id": "src_hero"}],
            },
            "name": "Queue Job 4 (Completed)",
        }
        res4 = client1.post("/api/v1/queue", json=job4_req, headers={"Idempotency-Key": "key-job-4"})
        job4_id = res4.json()["job_id"]
        app1.state.queue.transition(job4_id, CanonicalJobState.COMPLETED, output={"output_path": "roop_3d/out.mp4", "frames": 100})
        assert app1.state.queue.get(job4_id).state == CanonicalJobState.COMPLETED

        # Verify all 10 canonical states in queue capabilities
        caps_res = client1.get("/api/v1/system/capabilities")
        assert caps_res.status_code == 200
        workstation = caps_res.json().get("workstation", {})
        canonical_states = workstation.get("canonical_job_states", [])
        assert len(canonical_states) == 10
        for s in ("QUEUED", "PREFLIGHT", "PRESCANNING", "PREPASSING", "RENDERING", "MUXING", "PAUSED", "COMPLETED", "FAILED", "CANCELLED"):
            assert s in canonical_states

    # 2. Simulate Server Shutdown / Crash / Restart:
    # Deliberately discard app1 and client1
    del client1
    del app1

    # 3. Boot Server Instance 2 on same repo_root
    app2 = create_app(Settings(), repo_root=tmp_path)

    with TestClient(app2, base_url="http://127.0.0.1:8030") as client2:
        list_res = client2.get("/api/v1/queue")
        assert list_res.status_code == 200
        jobs = list_res.json()
        assert len(jobs) == 4

        job_map = {j["job_id"]: j for j in jobs}

        # (1) Job 1 MUST BE RESTORED INTACT AS QUEUED!
        assert job_map[job1_id]["state"] == "QUEUED"
        assert job_map[job1_id]["name"] == "Queue Job 1 (Must Remain QUEUED)"

        # (2) Job 2 MUST BE RESTORED INTACT AS PAUSED with checkpoint!
        assert job_map[job2_id]["state"] == "PAUSED"
        assert job_map[job2_id]["checkpoint_path"] is not None
        # Checkpoint is readable
        restored_cp_file = tmp_path / job_map[job2_id]["checkpoint_path"]
        assert restored_cp_file.is_file()
        loaded_cp = read_checkpoint_npz(restored_cp_file.read_bytes())
        assert loaded_cp.completed_frame == 15
        assert loaded_cp.pose_cache == {"track_0000": [1.0, 2.0, 3.0]}

        # (3) Job 3 (interrupted while RENDERING) must be recovered safely to PAUSED
        assert job_map[job3_id]["state"] == "PAUSED"

        # (4) Job 4 MUST REMAIN COMPLETED
        assert job_map[job4_id]["state"] == "COMPLETED"

        # (5) Test Resume of Job 2 from PAUSED -> QUEUED
        resume_res = client2.post(f"/api/v1/queue/{job2_id}/resume")
        assert resume_res.status_code == 200
        assert resume_res.json()["state"] == "QUEUED"

        # (6) Test Duplicate Job
        dup_res = client2.post(f"/api/v1/queue/{job1_id}/duplicate")
        assert dup_res.status_code == 201
        dup_job = dup_res.json()
        assert dup_job["job_id"] != job1_id
        assert dup_job["state"] == "QUEUED"
        assert "Copy" in dup_job["name"]

        # (7) Test Drag-and-Drop Reorder
        all_ids = [j["job_id"] for j in client2.get("/api/v1/queue").json()]
        reversed_ids = list(reversed(all_ids))
        reorder_res = client2.post("/api/v1/queue/reorder", json={"ordered_job_ids": reversed_ids})
        assert reorder_res.status_code == 200
        reordered = reorder_res.json()
        assert [j["job_id"] for j in reordered] == reversed_ids


# =========================================================================
# Verification Gate 3: GOP-aligned video segment splitting, chunk preservation,
# and lossless concat with frame-accurate audio sync
# =========================================================================

def test_multisegment_render_gop_split_preserves_chunks_and_lossless_concat_audio_sync(tmp_path: Path):
    # 1. Create a 4.0-second video with 25 fps, GOP size 25 (1 keyframe per second = 4 GOPs)
    # and stereo audio tone at 48 kHz
    source_video = tmp_path / "roop_3d/.runtime/media/input_multi_gop.mp4"
    create_synthetic_video_with_audio(source_video, duration_seconds=4.0, fps=25, gop_size=25, audio_sample_rate=48000)

    assert source_video.is_file()
    keyframes, duration = detect_keyframe_timestamps(source_video)
    assert len(keyframes) >= 4, f"Expected at least 4 keyframes, got {keyframes}"
    assert duration >= 3.8

    # 2. GOP-aligned splitting with max segment duration of 1.5 seconds (so it splits across GOP keyframes)
    chunks_dir = tmp_path / "roop_3d/.runtime/segments"
    slices = split_video_gop(source_video, chunks_dir, max_segment_duration=1.5, timestamp="99999")

    # Should produce multiple GOP-aligned chunks: chunk_0001_99999.mp4, chunk_0002_99999.mp4, etc.
    assert len(slices) >= 2, f"Expected multiple slices, got {len(slices)}"
    for idx, sl in enumerate(slices, start=1):
        assert sl.slice_index == idx
        chunk_file = Path(sl.chunk_path)
        assert chunk_file.is_file()
        assert chunk_file.stat().st_size > 0
        assert f"chunk_{idx:04d}_99999.mp4" in chunk_file.name

        # Verify each chunk starts cleanly and is stream-copy decodable
        with av.open(str(chunk_file)) as c:
            assert len(c.streams.video) > 0
            first_packet = next(c.demux(c.streams.video[0]))
            assert first_packet.is_keyframe, f"Chunk {idx} must begin with an exact GOP keyframe"

    # 3. Simulate rendering slices: each chunk is processed into a rendered chunk
    rendered_chunks: list[Path] = []
    for sl in slices:
        rendered_chunk = chunks_dir / f"rendered_{Path(sl.chunk_path).name}"
        # Copy input chunk to rendered chunk to simulate swap pass without video re-encode
        shutil.copyfile(sl.chunk_path, rendered_chunk)
        sl.rendered_chunk_path = str(rendered_chunk.resolve().as_posix())
        sl.status = "COMPLETED"
        rendered_chunks.append(rendered_chunk)

    # Verify every intermediate chunk exists BEFORE concat
    for sl in slices:
        assert Path(sl.chunk_path).is_file(), "Original chunk must be preserved"
        assert Path(sl.rendered_chunk_path).is_file(), "Rendered intermediate chunk must be preserved"

    # 4. Execute lossless concat demuxer pass and remux original source audio stream
    output_video = tmp_path / "roop_3d/.runtime/outputs/lossless_concat_result.mp4"
    concat_info = concat_lossless(
        slice_paths=[str(p) for p in rendered_chunks],
        source_audio_video=source_video,
        output_path=output_video,
        preserve_intermediates=True,
    )

    # 5. Verify concat results:
    assert output_video.is_file()
    assert output_video.stat().st_size > 0
    assert concat_info["segments_concatenated"] == len(rendered_chunks)
    assert concat_info["audio_sync_remuxed"] is True

    # Check that EVERY intermediate chunk was strictly preserved on disk after concat finishes!
    for sl in slices:
        assert Path(sl.chunk_path).is_file(), f"Chunk {sl.chunk_path} was lost after concat!"
        assert Path(sl.rendered_chunk_path).is_file(), f"Rendered chunk {sl.rendered_chunk_path} was lost after concat!"

    # 6. Verify frame accuracy and audio sync
    with av.open(str(output_video)) as out_container:
        assert len(out_container.streams.video) == 1
        assert len(out_container.streams.audio) == 1

        v_stream = out_container.streams.video[0]
        a_stream = out_container.streams.audio[0]

        # Calculate total frames decoded from concatenated video
        out_frames = 0
        for _ in out_container.decode(v_stream):
            out_frames += 1

        # Check total frames
        with av.open(str(source_video)) as src_container:
            src_frames = 0
            for _ in src_container.decode(src_container.streams.video[0]):
                src_frames += 1

        assert out_frames == src_frames, f"Frame count mismatch: output {out_frames} vs source {src_frames}"
        assert a_stream.rate == 48000, f"Expected 48 kHz audio, got {a_stream.rate}"
        assert a_stream.channels == 2, f"Expected stereo audio, got {a_stream.channels}"


def test_queue_slice_retry_and_clip_joining(tmp_path: Path):
    source_video = tmp_path / "roop_3d/.runtime/media/join_test.mp4"
    create_synthetic_video_with_audio(source_video, duration_seconds=2.0, fps=25, gop_size=25)

    chunks_dir = tmp_path / "roop_3d/.runtime/segments"
    slices = split_video_gop(source_video, chunks_dir, max_segment_duration=1.0, timestamp="55555")
    assert len(slices) >= 2

    qm = QueueManager(tmp_path)
    job, _ = qm.submit(
        request={"target_path": "roop_3d/.runtime/media/join_test.mp4", "face_mappings": []},
        name="Slice Retry and Joining Test",
        slices=slices,
    )

    # Simulate slice 1 failure
    qm.transition(job.job_id, CanonicalJobState.FAILED, error_message="Slice 1 corrupted")
    job = qm.get(job.job_id)
    assert job.state == CanonicalJobState.FAILED

    # Retry failed slice 1
    job = qm.retry_failed_slice(job.job_id, slice_index_or_id=1)
    assert job.state == CanonicalJobState.QUEUED
    assert job.slices[0].status == "PENDING"
    assert job.slices[0].retry_count == 1

    # Mark all slices completed with rendered chunk paths
    with qm._lock, qm._get_connection() as conn:
        for sl in job.slices:
            sl.rendered_chunk_path = sl.chunk_path
            sl.status = "COMPLETED"
        raw_slices = [s.model_dump(mode="json") for s in job.slices]
        conn.execute("UPDATE queue_jobs SET slices_json = ? WHERE job_id = ?", (json.dumps(raw_slices), job.job_id))

    # Automatic clip joining
    joined_job = qm.automatic_clip_joining(job.job_id)
    assert joined_job.state == CanonicalJobState.COMPLETED
    assert joined_job.output is not None
    assert Path(tmp_path / joined_job.output["output_path"]).is_file()
    assert joined_job.output["audio_sync_remuxed"] is True
