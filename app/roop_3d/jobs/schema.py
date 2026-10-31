"""Persistent job contracts and the complete legal transition graph."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Literal
import uuid

from pydantic import Field, field_validator, model_validator

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.media.schema import TrackToFaceSetMap
from roop_3d.models.diagnostics import JobDiagnostics


def now() -> datetime:
    return datetime.now(timezone.utc)


class JobState(str, Enum):
    queued = "queued"
    preparing = "preparing"
    running = "running"
    pausing = "pausing"
    paused = "paused"
    cancelling = "cancelling"
    cancelled = "cancelled"
    succeeded = "succeeded"
    failed = "failed"
    interrupted = "interrupted"


TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.queued: frozenset({JobState.preparing, JobState.pausing, JobState.cancelling, JobState.failed}),
    JobState.preparing: frozenset({JobState.running, JobState.pausing, JobState.cancelling, JobState.failed, JobState.interrupted}),
    JobState.running: frozenset({JobState.pausing, JobState.cancelling, JobState.succeeded, JobState.failed, JobState.interrupted}),
    JobState.pausing: frozenset({JobState.paused, JobState.cancelling, JobState.failed, JobState.interrupted}),
    JobState.paused: frozenset({JobState.queued, JobState.cancelling}),
    JobState.cancelling: frozenset({JobState.cancelled, JobState.failed, JobState.interrupted}),
    JobState.interrupted: frozenset({JobState.queued, JobState.cancelling}),
    JobState.cancelled: frozenset(), JobState.succeeded: frozenset(), JobState.failed: frozenset(),
}


def require_transition(before: JobState, after: JobState) -> None:
    if after not in TRANSITIONS[before]:
        raise DomainError("ILLEGAL_JOB_TRANSITION", f"Cannot transition from {before.value} to {after.value}")


class JobRequest(Contract):
    target_path: str = Field(min_length=1, max_length=1024)
    inspection_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    face_mappings: tuple[TrackToFaceSetMap, ...] = Field(min_length=1, max_length=64)
    processing: Literal["local", "remote"] = "local"
    remote_consent: bool = False

    @model_validator(mode="after")
    def mapping_contract(self) -> "JobRequest":
        if len({item.track_id for item in self.face_mappings}) != len(self.face_mappings):
            raise ValueError("Each target track may have only one FaceSet mapping")
        if self.processing == "remote" and not self.remote_consent:
            raise ValueError("Remote processing requires explicit per-job consent")
        return self


class JobProgress(Contract):
    frame: int = Field(default=0, ge=0, strict=True)
    total_frames: int | None = Field(default=None, ge=1, strict=True)
    fps: float = Field(default=0, ge=0)
    stage: str = Field(default="queued", min_length=1, max_length=64)
    stage_latency_ms: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def frame_bounds(self) -> "JobProgress":
        if self.total_frames is not None and self.frame > self.total_frames:
            raise ValueError("Completed frames cannot exceed total frames")
        return self


class JobOutput(Contract):
    output_path: str = Field(min_length=1, max_length=1024)
    manifest_path: str = Field(min_length=1, max_length=1024)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frames: int = Field(ge=1, strict=True)
    audio_policy: Literal["copy", "transcode", "drop"]

    @field_validator("output_path", "manifest_path")
    @classmethod
    def owned_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or not value.startswith("roop_3d/.runtime/"):
            raise ValueError("Job output paths must remain inside the configured runtime")
        return path.as_posix()


class JobFailure(Contract):
    code: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z0-9_]+$")
    message: str = Field(min_length=1, max_length=512)
    recoverable: bool = False


class JobRecord(Contract):
    schema_version: Literal[1] = 1
    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[0-9a-f]{32}$")
    request: JobRequest
    state: JobState = JobState.queued
    revision: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime = Field(default_factory=now)
    progress: JobProgress = Field(default_factory=JobProgress)
    diagnostics: JobDiagnostics = Field(default_factory=JobDiagnostics)
    output: JobOutput | None = None
    failure: JobFailure | None = None


class FaceSetSummary(Contract):
    faceset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    name: str = Field(min_length=1, max_length=128)
    item_count: int = Field(ge=1)
    schema_version: Literal[1] = 1
