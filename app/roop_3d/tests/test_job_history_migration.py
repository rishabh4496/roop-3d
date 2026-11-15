"""Historical jobs survive the removal of legacy render options without becoming runnable."""
import json

import pytest

from roop_3d.core.errors import DomainError
from roop_3d.jobs.schema import JobRequest, JobState
from roop_3d.jobs.store import JobStore


def history(root, **changes):
    store = JobStore(root)
    job, _ = store.create(JobRequest(target_path="roop_3d/.runtime/media/test.mp4",
        inspection_id="0" * 32, face_mappings=[{"track_id": "track_0000", "faceset_id": "alice"}]), "history")
    path = root / store.relative
    data = json.loads(path.read_text())
    data["jobs"][job.job_id]["request"]["workstation_parameters"] = changes
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, job.job_id


def test_legacy_history_is_backed_up_and_cannot_resume(tmp_path):
    path, job_id = history(tmp_path, pose_mode="tps", restoration_blend=0.8)
    original = path.read_bytes()
    store = JobStore(tmp_path)
    job = store.get(job_id)
    assert job.state == JobState.failed
    assert job.failure.code == "LEGACY_JOB_RETIRED"
    assert next(path.parent.glob("state.pre-true3d-*.json")).read_bytes() == original
    assert JobStore(tmp_path).get(job_id) == job
    with pytest.raises(DomainError):
        store.control(job_id, "resume")


def test_unknown_corruption_is_not_migrated(tmp_path):
    path, job_id = history(tmp_path, unrelated_unknown_option=True)
    original = path.read_bytes()
    with pytest.raises(DomainError, match="unreadable"):
        JobStore(tmp_path).get(job_id)
    assert path.read_bytes() == original
    assert not list(path.parent.glob("state.pre-true3d-*.json"))
