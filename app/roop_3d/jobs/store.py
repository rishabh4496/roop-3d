"""Transactional on-disk state, persistent idempotency and worker acknowledgments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract, Settings
from roop_3d.core.storage import atomic_json, exclusive, owned
from roop_3d.jobs.schema import (JobFailure, JobOutput, JobProgress, JobRecord, JobRequest,
                                 JobState, now, require_transition)
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
        # (stat fingerprint of state.json, parsed database). Records are frozen, so sharing them is safe.
        self._cache: tuple[tuple[int, int, int], JobDatabase] | None = None

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size, getattr(stat, "st_ino", 0)

    def _load(self) -> JobDatabase:
        """The parsed database, re-parsed only when state.json changed on disk (this or any other process).

        Every operation used to parse and validate *every job ever recorded* - 25 ms per ``get()`` and 70 ms per
        ``progress()`` at 800 jobs, several times per rendered frame. The worker polls this at frame rate, so job
        history alone capped throughput. Writes replace the file atomically (new mtime/size/inode), which invalidates
        the fingerprint, so a second process writing the same file is still seen immediately. Never mutate the result.
        """
        path = owned(self.root, self.relative)
        try:
            fingerprint = self._fingerprint(path)
        except FileNotFoundError:
            self._cache = None
            return JobDatabase()
        cached = self._cache
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        try:
            database = JobDatabase.model_validate_json(path.read_text("utf-8"))
        except (ValueError, OSError):
            raise DomainError("JOB_STORE_INVALID", "Persisted job data is unreadable; it has been preserved") from None
        self._cache = (fingerprint, database)
        return database

    def _read(self) -> JobDatabase:
        """A private, mutable copy for read-modify-write (the records inside are shared and immutable)."""
        database = self._load()
        return JobDatabase.model_construct(schema_version=database.schema_version, jobs=dict(database.jobs),
                                           idempotency=dict(database.idempotency))

    def _save(self, data: JobDatabase) -> None:
        # Callers hold the cross-process lock, so nobody else can replace the file between this write and the stat.
        atomic_json(self.root, self.relative, data.model_dump(mode="json"), indent=None)
        self._cache = (self._fingerprint(owned(self.root, self.relative)), data)

    def list(self) -> list[JobRecord]:
        with exclusive(self.root, self.relative + ".lock"):
            return sorted(self._load().jobs.values(), key=lambda job: (job.created_at, job.job_id))

    def get(self, job_id: str) -> JobRecord:
        with exclusive(self.root, self.relative + ".lock"):
            record = self._load().jobs.get(job_id)
            if record is None:
                raise DomainError("JOB_NOT_FOUND", "Job does not exist", 404)
            return record

    def create(self, request: JobRequest, idempotency_key: str,
               validate: Callable[[], None] | None = None, *,
               project_id: str | None = None, effective_settings: Settings | None = None) -> tuple[JobRecord, bool]:
        request = JobRequest.model_validate(request.model_dump())
        from roop_3d.security.synthetic_media import SafetyGuard
        SafetyGuard.inspect_request(
            request_metadata=request.model_dump(),
            face_labels=tuple(m.faceset_id for m in request.face_mappings),
        )
        if not 1 <= len(idempotency_key) <= 128 or not idempotency_key.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in idempotency_key):
            raise DomainError("IDEMPOTENCY_KEY_INVALID", "Idempotency-Key must contain 1 to 128 printable ASCII characters", 422)
        key = hashlib.sha256(idempotency_key.encode()).hexdigest()
        fingerprint = hashlib.sha256(json.dumps(request.identity_payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if project_id is not None:
            fingerprint = hashlib.sha256(f"{project_id}:{fingerprint}".encode()).hexdigest()
        with exclusive(self.root, self.relative + ".lock"):
            data = self._read()
            prior = data.idempotency.get(key)
            if prior:
                if prior.request_sha256 != fingerprint:
                    raise DomainError("IDEMPOTENCY_CONFLICT", "Idempotency key was already used for a different request")
                return data.jobs[prior.job_id], False
            if validate:
                validate()
            record = JobRecord(request=request, project_id=project_id, effective_settings=effective_settings)
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
            update = {"state": state, "revision": before.revision + 1, "updated_at": now()}
            if state in {JobState.preparing, JobState.running}:
                update["failure"] = None  # a resumed job no longer reports the interruption that paused it
            if state == JobState.preparing:
                # Every attempt starts counting again from its checkpoint. Progress is persisted per frame but the
                # checkpoint only every N frames, so after a crash the resumed attempt legitimately reports a lower
                # frame than the stale value left on disk; without this rewind `progress()` rejected it as
                # PROGRESS_REGRESSION and failed (terminally) every job that had been hard-killed.
                update["progress"] = before.progress.model_copy(update={
                    "frame": 0, "fps": 0.0, "stage_latency_ms": 0.0, "stage_timings_ms": {},
                    "stage": "resuming" if before.progress.frame else before.progress.stage})
            record = before.model_copy(update=update)
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

    def succeed(self, job_id: str, output: JobOutput) -> JobRecord:
        output = JobOutput.model_validate(output.model_dump())
        with exclusive(self.root, self.relative + ".lock"):
            data = self._read()
            before = data.jobs.get(job_id)
            if before is None:
                raise DomainError("JOB_NOT_FOUND", "Job does not exist", 404)
            require_transition(before.state, JobState.succeeded)
            record = before.model_copy(update={"state": JobState.succeeded, "output": output,
                                               "failure": None, "revision": before.revision + 1,
                                               "updated_at": now()})
            data.jobs[job_id] = record
            self._save(data)
        self.notify({"type": "job.state", "job_id": job_id, "previous_state": before.state.value,
                     "state": record.state.value, "revision": record.revision})
        return record

    def fail(self, job_id: str, failure: JobFailure) -> JobRecord:
        failure = JobFailure.model_validate(failure.model_dump())
        with exclusive(self.root, self.relative + ".lock"):
            data = self._read()
            before = data.jobs.get(job_id)
            if before is None:
                raise DomainError("JOB_NOT_FOUND", "Job does not exist", 404)
            require_transition(before.state, JobState.failed)
            record = before.model_copy(update={"state": JobState.failed, "failure": failure,
                                               "revision": before.revision + 1, "updated_at": now()})
            data.jobs[job_id] = record
            self._save(data)
        self.notify({"type": "job.state", "job_id": job_id, "previous_state": before.state.value,
                     "state": record.state.value, "revision": record.revision})
        return record

    def interrupt(self, job_id: str, failure: JobFailure) -> JobRecord:
        """Stage 18: stop a job in a *resumable* state and say why (disk full, host out of memory, locked file ...).

        ``failed`` is terminal, so a transient environmental problem must never use it: the checkpoint is intact and
        the user can fix the cause and resume. The reason stays visible on the record until the job runs again.
        """
        failure = JobFailure.model_validate({**failure.model_dump(), "recoverable": True})
        with exclusive(self.root, self.relative + ".lock"):
            data = self._read()
            before = data.jobs.get(job_id)
            if before is None:
                raise DomainError("JOB_NOT_FOUND", "Job does not exist", 404)
            require_transition(before.state, JobState.interrupted)
            record = before.model_copy(update={"state": JobState.interrupted, "failure": failure,
                                               "revision": before.revision + 1, "updated_at": now()})
            data.jobs[job_id] = record
            self._save(data)
        self.notify({"type": "job.state", "job_id": job_id, "previous_state": before.state.value,
                     "state": record.state.value, "revision": record.revision})
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
