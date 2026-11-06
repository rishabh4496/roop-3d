"""Stage 9: the streaming render path resumes from the last verified finished segment (job_checkpoint.json)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
from roop_3d.media.video_reader import VideoReader
from roop_3d.optimized_processor import StreamingProcessor
from roop_3d.render_policy import RenderPolicy
from roop_3d.tools.verify_stream_integration import create_clip
from roop_3d.vectorized_pipeline import CpuBatchPipeline


class Stop(Exception):
    pass


POINTS = ((ARCFACE_REFERENCE_5PTS + [8, 0]) * 0.4).astype(np.float32)


def _processor(root, source, calls, *, stop_after_segments=None, identity=None):
    def inference(value, _):
        calls.append(1)
        return value

    def check():
        path = root / "roop_3d/.runtime/stream" / ("a" * 32) / "job_checkpoint.json"
        if stop_after_segments and path.is_file() and len(json.loads(path.read_text())["segments"]) >= stop_after_segments:
            raise Stop()

    return StreamingProcessor(root, "a" * 32, source, detect=lambda _: [1],
                              track=lambda _: [(0, np.array([0, 0, 64, 64]), POINTS)],
                              pipeline=CpuBatchPipeline(inference),
                              policy=RenderPolicy(hardware_profile="cpu", segment_frames=3), check=check,
                              identity=identity or {"request": "r1"}, execution_provider="CPUExecutionProvider",
                              settings_3d={"render_mode": "standard"})


def _checkpoint(root):
    return json.loads((root / "roop_3d/.runtime/stream" / ("a" * 32) / "job_checkpoint.json").read_text())


def test_interrupted_stream_render_resumes_after_last_finished_segment(tmp_path):
    source = tmp_path / "input.mp4"
    create_clip(source)  # 10 frames
    first_calls: list[int] = []
    with pytest.raises(Stop):
        _processor(tmp_path, source, first_calls, stop_after_segments=2).run(tmp_path / "first.mp4", audio=False)
    checkpoint = _checkpoint(tmp_path)
    done = [s for s in checkpoint["segments"] if s["status"] == "COMPLETED"]
    assert done and checkpoint["execution_provider"] == "CPUExecutionProvider"
    assert checkpoint["settings_3d"] == {"render_mode": "standard"}
    resume_frame = sum(s["frames"] for s in done)
    assert resume_frame == 6
    first_segment = tmp_path / done[0]["path"]
    digest = hashlib.sha256(first_segment.read_bytes()).hexdigest()

    second_calls: list[int] = []
    result = _processor(tmp_path, source, second_calls).run(tmp_path / "second.mp4", audio=False)
    assert result["resumed_from_frame"] == resume_frame
    assert result["frames"] == 10
    assert len(second_calls) <= 10 - resume_frame  # frames of finished segments were not rendered again
    assert hashlib.sha256(first_segment.read_bytes()).hexdigest() == digest  # untouched
    assert sum(1 for _ in VideoReader(tmp_path / "second.mp4", hardware="software").frames()) == 10
    assert _checkpoint(tmp_path)["state"] == "COMPLETED"
    assert all((tmp_path / s["path"]).is_file() for s in _checkpoint(tmp_path)["segments"])  # kept after the join


def test_corrupt_finished_segment_is_not_trusted(tmp_path):
    source = tmp_path / "input.mp4"
    create_clip(source)
    with pytest.raises(Stop):
        _processor(tmp_path, source, [], stop_after_segments=2).run(tmp_path / "first.mp4", audio=False)
    victim = tmp_path / _checkpoint(tmp_path)["segments"][1]["path"]
    victim.write_bytes(victim.read_bytes()[:-200])  # damaged after it was recorded
    result = _processor(tmp_path, source, []).run(tmp_path / "second.mp4", audio=False)
    assert result["resumed_from_frame"] == 3  # only the intact leading segment is kept
    assert result["frames"] == 10
    assert sum(1 for _ in VideoReader(tmp_path / "second.mp4", hardware="software").frames()) == 10


def test_checkpoint_endpoint_reports_the_resume_point(tmp_path):
    from fastapi.testclient import TestClient
    from roop_3d.server.app import create_app

    source = tmp_path / "input.mp4"
    create_clip(source)
    with pytest.raises(Stop):
        _processor(tmp_path, source, [], stop_after_segments=2).run(tmp_path / "first.mp4", audio=False)
    app = create_app(repo_root=tmp_path, worker_enabled=False)
    app.state.jobs.get = lambda job_id: object()  # the job store itself is not under test
    client = TestClient(app, base_url="http://127.0.0.1:8030")
    body = client.get(f"/api/v1/jobs/{'a' * 32}/checkpoint").json()
    assert body["available"] and body["resume_frame"] == 6 and body["completed_segments"] == 2
    assert body["execution_provider"] == "CPUExecutionProvider"
    assert client.get(f"/api/v1/jobs/{'b' * 32}/checkpoint").json() == {"available": False}
