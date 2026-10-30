"""Transactional on-disk state, persistent idempotency and worker acknowledgments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.core.storage import atomic_json, exclusive, owned
from roop_3d.jobs.schema import JobProgress, JobRecord, JobRequest, JobState, now, require_transition
from roop_3d.models.diagnostics import JobDiagnostics


class IdempotencyEntry(Contract):
    job_id: str
    request_sha256: str


class JobDatabase(Contract):
    schema_version: int = Field(default=1, ge=1, le=1)
    jobs: dict[str, JobRecord] = Field(default_factory=dict)
    idempotency: dict[str, IdempotencyEntry] = Field(default_factory=dict)


class JobStore:
    relative = "roop_3d/.runtime/jobs/state.json"

    def __init__(self, root: Path, notify: Callable[[dict], None] | None = None):
        self.root, self.notify = root, notify or (lambda event: None)

    def _read(self) -> JobDatabase:
        path = owned(self.root, self.relative)
        try:
            return JobDatabase.model_validate_json(path.read_text("utf-8")) if path.exists() else JobDatabase()
        except (ValueError, OSError):
            raise DomainError("JOB_STORE_INVALID", "Persisted job data is unreadable; it has been preserved") from None

    def _save(self, data: JobDatabase) -> None:
        atomic_json(self.root, self.relative, data.model_dump(mode="json"))

    def list(self) -> list[JobRecord]:
        with exclusive(self.root, self.relative + ".lock"):
            return sorted(self._read().jobs.values(), key=lambda job: (job.created_at, job.job_id))

    def get(self, job_id: str) -> JobRecord:
        with exclusive(self.root, self.relative + ".lock"):
            record = self._read().jobs.get(job_id)
            if record is None:
                raise DomainError("JOB_NOT_FOUND", "Job does not exist", 404)
            return record

    def create(self, request: JobRequest, idempotency_key: str,
               validate: Callable[[], None] | None = None) -> tuple[JobRecord, bool]:
        request = JobRequest.model_validate(request.model_dump())
        if not 1 <= len(idempotency_key) <= 128 or not idempotency_key.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in idempotency_key):
            raise DomainError("IDEMPOTENCY_KEY_INVALID", "Idempotency-Key must contain 1 to 128 printable ASCII characters", 422)
        key = hashlib.sha256(idempotency_key.encode()).hexdigest()
        fingerprint = hashlib.sha256(json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with exclusive(self.root, self.relative + ".lock"):
            data = self._read()
            prior = data.idempotency.get(key)
            if prior:
                if prior.request_sha256 != fingerprint:
                    raise DomainError("IDEMPOTENCY_CONFLICT", "Idempotency key was already used for a different request")
                return data.jobs[prior.job_id], False
            if validate:
                validate()
            record = JobRecord(request=request)
            data.jobs[record.job_id] = record
            data.idempotency[key] = IdempotencyEntry(job_id=record.job_id, request_sha256=fingerprint)
            self._save(data)
        self.notify({"type": "job.state", "job_id": record.job_id, "previous_state": None,
                     "state": record.state.value, "revision": record.revision})
        return record, True

    def transition(self, job_id: str, state: JobState, *, expected_revision: int | None = None) -> JobRecord:
        state = JobState(state)
        with exclusive(self.root, self.relative + ".lock"):
            data = self._read()
            before = data.jobs.get(job_id)
            if before is None:
                raise DomainError("JOB_NOT_FOUND", "Job does not exist", 404)
            if expected_revision is not None and before.revision != expected_revision:
                raise DomainError("JOB_REVISION_CONFLICT", "Job changed since it was read")
            require_transition(before.state, state)
            record = before.model_copy(update={"state": state, "revision": before.revision + 1, "updated_at": now()})
            data.jobs[job_id] = record
            self._save(data)
        self.notify({"type": "job.state", "job_id": job_id, "previous_state": before.state.value,
                     "state": state.value, "revision": record.revision})
        return record

    def progress(self, job_id: str, value: JobProgress) -> JobRecord:
        # Revalidate even if a caller passed a model_copy constructed without validation.
        value = JobProgress.model_validate(value.model_dump())
        with exclusive(self.root, self.relative + ".lock"):
            data = self._read()
            before = data.jobs.get(job_id)
            if before is None:
                raise DomainError("JOB_NOT_FOUND", "Job does not exist", 404)
            if before.state != JobState.running:
                raise DomainError("JOB_NOT_RUNNING", "Only running jobs may report frame progress")
            if value.frame < before.progress.frame:
                raise DomainError("PROGRESS_REGRESSION", "Frame progress cannot move backwards")
            record = before.model_copy(update={"progress": value, "revision": before.revision + 1, "updated_at": now()})
            data.jobs[job_id] = record
            self._save(data)
        self.notify({"type": "job.progress", "job_id": job_id, "revision": record.revision, **value.model_dump(mode="json")})
        return record

    def record_diagnostics(self, job_id: str, value: JobDiagnostics) -> JobRecord:
        """Worker hook: persist active providers for the existing job GET API."""
        value = JobDiagnostics.model_validate(value.model_dump())
        with exclusive(self.root, self.relative + ".lock"):
            data = self._read()
            before = data.jobs.get(job_id)
            if before is None:
                raise DomainError("JOB_NOT_FOUND", "Job does not exist", 404)
            if before.state not in {JobState.preparing, JobState.running}:
                raise DomainError("JOB_NOT_ACTIVE", "Only an active job may update inference diagnostics", 409)
            record = before.model_copy(update={"diagnostics": value, "revision": before.revision + 1, "updated_at": now()})
            data.jobs[job_id] = record
            self._save(data)
        return record

    def control(self, job_id: str, action: str) -> JobRecord:
        before = self.get(job_id)
        if action == "resume":
            if before.state not in {JobState.paused, JobState.interrupted}:
                raise DomainError("ILLEGAL_JOB_TRANSITION", "Only paused or interrupted jobs may resume")
            return self.transition(job_id, JobState.queued, expected_revision=before.revision)
        if action not in {"pause", "cancel"}:
            raise DomainError("CONTROL_UNKNOWN", "Unknown job control", 422)
        target = JobState.pausing if action == "pause" else JobState.cancelling
        record = self.transition(job_id, target, expected_revision=before.revision)
        # No worker owns a queued/paused/interrupted job, so control can finish now.
        # Active workers must release resources then acknowledge pausing/cancelling.
        if before.state in {JobState.queued, JobState.paused, JobState.interrupted}:
            return self.transition(job_id, JobState.paused if action == "pause" else JobState.cancelled,
                                   expected_revision=record.revision)
        return record

    def recover_interrupted(self) -> list[JobRecord]:
        recovered = []
        for record in self.list():
            if record.state in {JobState.preparing, JobState.running, JobState.pausing, JobState.cancelling}:
                recovered.append(self.transition(record.job_id, JobState.interrupted, expected_revision=record.revision))
        return recovered
