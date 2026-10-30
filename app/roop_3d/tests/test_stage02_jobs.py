from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

from pydantic import ValidationError
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.jobs.schema import JobProgress, JobRequest, JobState, TRANSITIONS, require_transition
from roop_3d.jobs.store import JobStore


def request(**changes) -> JobRequest:
    return JobRequest(target_path="roop_3d/.runtime/media/test.mp4", inspection_id="0" * 32,
                      face_mappings=[{"track_id": "track_0000", "faceset_id": "alice"}], **changes)


def test_complete_transition_matrix() -> None:
    assert len(TRANSITIONS) == 10
    for before in JobState:
        for after in JobState:
            if after in TRANSITIONS[before]:
                require_transition(before, after)
            else:
                with pytest.raises(DomainError, match="Cannot transition"):
                    require_transition(before, after)


def test_durable_idempotency_conflict_and_secret_key_hashing(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job, created = store.create(request(), "SECRET_IDEMPOTENCY_KEY")
    assert created
    duplicate, created = JobStore(tmp_path).create(request(), "SECRET_IDEMPOTENCY_KEY")
    assert not created and duplicate == job
    assert "SECRET_IDEMPOTENCY_KEY" not in (tmp_path / store.relative).read_text()
    changed = request().model_copy(update={"target_path": "roop_3d/.runtime/media/different.mp4"})
    with pytest.raises(DomainError) as error:
        store.create(changed, "SECRET_IDEMPOTENCY_KEY")
    assert error.value.code == "IDEMPOTENCY_CONFLICT"


def test_concurrent_creates_serialize_to_one_job(tmp_path: Path) -> None:
    def create(_):
        return JobStore(tmp_path).create(request(), "one-key")
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(create, range(16)))
    assert sum(created for _, created in results) == 1
    assert len({job.job_id for job, _ in results}) == 1
    assert len(JobStore(tmp_path).list()) == 1


def test_controls_active_acknowledgment_restart_and_progress(tmp_path: Path) -> None:
    events = []
    store = JobStore(tmp_path, events.append)
    job, _ = store.create(request(), "key")
    assert store.control(job.job_id, "pause").state == JobState.paused
    assert store.control(job.job_id, "resume").state == JobState.queued
    store.transition(job.job_id, JobState.preparing)
    store.transition(job.job_id, JobState.running)
    store.progress(job.job_id, JobProgress(frame=3, total_frames=10, fps=24, stage="detect", stage_latency_ms=12))
    with pytest.raises(DomainError):
        store.progress(job.job_id, JobProgress(frame=2))
    assert store.control(job.job_id, "pause").state == JobState.pausing
    assert JobStore(tmp_path).recover_interrupted()[0].state == JobState.interrupted
    assert store.control(job.job_id, "cancel").state == JobState.cancelled
    with pytest.raises(DomainError):
        store.control(job.job_id, "resume")
    assert any(event["type"] == "job.progress" and event["fps"] == 24 for event in events)


def test_revision_conflicts_and_nonfinite_progress(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    job, _ = store.create(request(), "key")
    with pytest.raises(DomainError) as error:
        store.transition(job.job_id, JobState.preparing, expected_revision=2)
    assert error.value.code == "JOB_REVISION_CONFLICT"
    for changes in ({"fps": float("inf")}, {"stage_latency_ms": float("nan")}, {"frame": 3, "total_frames": 2}):
        with pytest.raises(ValidationError):
            JobProgress(**changes)
    with pytest.raises(ValidationError):
        request(processing="remote")


def test_cross_process_submission_uses_disk_lock(tmp_path: Path) -> None:
    code = ("import sys; from pathlib import Path; from roop_3d.jobs.store import JobStore; "
            "from roop_3d.jobs.schema import JobRequest; "
            "print(int(JobStore(Path(sys.argv[1])).create(JobRequest.model_validate_json(sys.argv[2]), 'shared')[1]))")
    processes = [subprocess.Popen([sys.executable, "-B", "-c", code, str(tmp_path), request().model_dump_json()],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(4)]
    outputs = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        outputs.append(int(stdout))
    assert sum(outputs) == 1
    assert len(JobStore(tmp_path).list()) == 1


def test_corrupt_job_state_fails_closed_without_overwrite(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    path = tmp_path / store.relative
    path.parent.mkdir(parents=True)
    path.write_text('{"jobs": "invalid", "private": "DO_NOT_EXPOSE"}')
    original = path.read_bytes()
    with pytest.raises(DomainError) as error:
        store.create(request(), "key")
    assert error.value.code == "JOB_STORE_INVALID"
    assert "DO_NOT_EXPOSE" not in str(error.value)
    assert path.read_bytes() == original
