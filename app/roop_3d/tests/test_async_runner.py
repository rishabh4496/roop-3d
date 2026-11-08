"""Defect #3: the job runner drives the four render stages through the asynchronous pipeline, correctly."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from roop_3d.tests.test_stage12_worker import _Session, _fixture, _wait_state


class _StagedSession(_Session):
    """A stub that, like ReferencePipeline, splits tracking from the swap and takes explicit composite indices."""

    def __init__(self, config, *, track_ms=0.0, swap_ms=0.0, composite_ms=0.0, fail_swap_at=None):
        super().__init__(config)
        self.track_ms, self.swap_ms, self.composite_ms, self.fail_swap_at = track_ms, swap_ms, composite_ms, fail_swap_at
        self.frames_processed = 0
        self.tracked: list[int] = []
        self.swapped: list[int] = []
        self.composited: list[tuple[int | None, float | None]] = []
        self.threads: dict[str, set[str]] = {"track": set(), "swap": set(), "composite": set()}
        self.timestamps: list[tuple[float, bool]] = []

    def set_frame_timestamp(self, seconds, *, compositor=True):
        self.timestamps.append((seconds, compositor))

    def track_frame(self, image, *, pose_blend=None, render_mode=None):
        self.threads["track"].add(threading.current_thread().name)
        time.sleep(self.track_ms)
        self.tracked.append(len(self.tracked))
        return SimpleNamespace(scene_cut=False, token=len(self.tracked) - 1)

    def infer_frame(self, image, tracked=None):
        self.threads["swap"].add(threading.current_thread().name)
        # Stage 1 must have produced this exact frame's snapshot, and in order.
        assert tracked is not None and tracked.token == self.frames_processed
        if self.fail_swap_at is not None and self.frames_processed == self.fail_swap_at:
            raise RuntimeError("injected swap failure")
        time.sleep(self.swap_ms)
        self.frames_processed += 1
        self.swapped.append(self.frames_processed - 1)
        return SimpleNamespace(replacements=(SimpleNamespace(track_id=0),), scene_cut=False)

    def composite_frame(self, image, result, frame_index=None, timestamp_seconds=None, **kwargs):
        self.threads["composite"].add(threading.current_thread().name)
        time.sleep(self.composite_ms)
        self.composited.append((frame_index, timestamp_seconds))
        return 255 - image


def _run(tmp_path, session, *, count=8):
    app, worker, payload = _fixture(tmp_path, count=count, session_factory=lambda catalog, config: session)
    worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            created = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": f"async-{time.time_ns()}"}).json()
            return worker, _wait_state(client, created["job_id"], "succeeded", timeout=30)
    finally:
        worker.stop()


def test_stages_run_on_separate_threads_in_order_with_explicit_frame_indices(tmp_path):
    session = _StagedSession(SimpleNamespace(provider="CPUExecutionProvider"))
    worker, done = _run(tmp_path, session)
    assert done["output"]["frames"] == 8
    assert session.swapped == list(range(8))
    # The composite stage was told the real frame index and clock; it never inferred them from the (ahead) session.
    assert [index for index, _ in session.composited] == list(range(8))
    assert [round(stamp, 3) for _, stamp in session.composited] == [round(i / 25, 3) for i in range(8)]
    # Tracking only touched the tracker's clock, never the compositor's.
    assert session.timestamps and all(compositor is False for _, compositor in session.timestamps)
    names = [next(iter(session.threads[stage])) for stage in ("track", "swap", "composite")]
    assert len(set(names)) == 3 and all(name.startswith("r3d-") for name in names), names
    snapshot = worker.runner.last_pipeline_snapshot
    assert [stage["name"] for stage in snapshot["stages"]] == ["decode", "track", "swap", "composite", "write"]
    assert snapshot["frames_out"] == 8 and all(stage["queue_out"]["capacity"] == 8 for stage in snapshot["stages"])
    assert snapshot["pools"][0]["backend"] in ("cuda_pinned", "numpy") and snapshot["pools"][0]["in_flight"] == 0


def test_stage_latencies_overlap_instead_of_adding(tmp_path):
    delay, count = 0.03, 24
    session = _StagedSession(SimpleNamespace(provider="CPUExecutionProvider"), track_ms=delay, swap_ms=delay, composite_ms=delay)
    began = time.perf_counter()
    worker, done = _run(tmp_path, session, count=count)
    wall = time.perf_counter() - began
    assert done["output"]["frames"] == count
    assert worker.runner.last_pipeline_snapshot["overlap_wall"]["ge3"] > 0.3
    serial_floor = count * 3 * delay
    assert wall < serial_floor + 6, "job overhead dominates this tiny fixture; overlap is asserted via the snapshot above"


def test_kill_switch_runs_the_same_stages_inline(tmp_path, monkeypatch):
    monkeypatch.setenv("ROOP3D_ASYNC_PIPELINE", "0")
    session = _StagedSession(SimpleNamespace(provider="CPUExecutionProvider"))
    worker, done = _run(tmp_path, session)
    assert done["output"]["frames"] == 8 and session.swapped == list(range(8))
    assert worker.runner.last_pipeline_snapshot is None
    assert len({name for names in session.threads.values() for name in names}) == 1, "inline mode is single threaded"


def test_stage_failure_fails_the_job_and_leaves_no_worker_threads(tmp_path):
    session = _StagedSession(SimpleNamespace(provider="CPUExecutionProvider"), fail_swap_at=3)
    app, worker, payload = _fixture(tmp_path, count=12, session_factory=lambda catalog, config: session)
    worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            job_id = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "async-fail"}).json()["job_id"]
            failed = _wait_state(client, job_id, "failed", timeout=30)
            assert failed["state"] == "failed"
    finally:
        worker.stop()
    stage_threads = {f"r3d-{name}" for name in ("decode", "track", "swap", "composite", "write")}
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(t.name in stage_threads and t.is_alive() for t in threading.enumerate()):
        time.sleep(0.05)
    assert not [t.name for t in threading.enumerate() if t.name in stage_threads and t.is_alive()]


def test_pause_and_resume_with_frames_in_flight(tmp_path):
    # A resumed job builds a fresh session (tracker state is replayed from frame 0), exactly like production.
    app, worker, payload = _fixture(tmp_path, count=24, session_factory=lambda catalog, config: _StagedSession(config, swap_ms=0.03))
    worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            job_id = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "async-pause"}).json()["job_id"]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                state = client.get(f"/api/v1/jobs/{job_id}").json()
                if state["state"] == "running" and state["progress"]["frame"] >= 3:
                    break
                time.sleep(0.01)
            else:
                pytest.fail("Worker did not begin rendering")
            assert client.post(f"/api/v1/jobs/{job_id}/pause").json()["state"] == "pausing"
            _wait_state(client, job_id, "paused", timeout=30)
            assert client.post(f"/api/v1/jobs/{job_id}/resume").json()["state"] == "queued"
            done = _wait_state(client, job_id, "succeeded", timeout=60)
            assert done["output"]["frames"] == 24
    finally:
        worker.stop()
