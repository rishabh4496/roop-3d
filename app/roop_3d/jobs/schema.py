"""Persistent job contracts and the complete legal transition graph."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Literal
import uuid

from pydantic import Field, field_validator, model_validator

from roop_3d.core.errors import DomainError
from roop_3d.core.stage_timing import validate_timings
from roop_3d.core.settings import Contract
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.core.animation import ParameterCurves, FaceSegment
from roop_3d.media.schema import TrackToFaceSetMap
from roop_3d.faceset.identity_blend import IdentityBlendSpec
from roop_3d.models.diagnostics import JobDiagnostics
from roop_3d.media.finishing import ExportOptions


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
    engine_id: str = Field(default="reference_onnx", pattern=r"^[a-z0-9][a-z0-9_]{0,95}$")
    target_path: str = Field(min_length=1, max_length=1024)
    inspection_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    face_mappings: tuple[TrackToFaceSetMap, ...] = Field(min_length=1, max_length=64)
    processing: Literal["local", "remote"] = "local"
    remote_consent: bool = False
    trim_start_seconds: float | None = Field(default=None, ge=0, le=86400)
    trim_end_seconds: float | None = Field(default=None, gt=0, le=86400)
    mask_base64: str | None = Field(default=None)
    workstation_parameters: WorkstationParameters | None = None
    identity_blend: IdentityBlendSpec | None = None
    parameter_curves: ParameterCurves | None = None
    face_segments: tuple[FaceSegment, ...] | None = Field(default=None, max_length=256)
    export: ExportOptions | None = None
    # Binds the render to an explicit, hash-verified target->FaceSet identity lock (roop_3d.identity.lock).
    identity_lock_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")

    def identity_payload(self) -> dict:
        value = self.model_dump(mode="json")
        # Preserve existing idempotency/checkpoint fingerprints for the default engine.
        if self.engine_id == "reference_onnx":
            value.pop("engine_id", None)
        if self.workstation_parameters:
            # Preserve fingerprints of requests saved before these opt-in controls existed.
            for field in ('swap_weight', 'erosion_px', 'pose_frontalization', 'tps_expression_weight'):
                if field not in self.workstation_parameters.model_fields_set:
                    value['workstation_parameters'].pop(field, None)
        for name in ("identity_blend", "trim_start_seconds", "trim_end_seconds", "mask_base64", "workstation_parameters", "parameter_curves", "face_segments", "export", "identity_lock_id"):
            if value.get(name) is None:
                value.pop(name, None)
        return value

    @model_validator(mode="after")
    def mapping_contract(self) -> "JobRequest":
        if self.face_segments is not None:
            if len({item.faceset_id for item in self.face_segments if item.faceset_id} | {item.faceset_id for item in self.face_mappings}) > 64:
                raise ValueError('At most 64 FaceSets per job')
            grouped = {}
            if len({item.cluster_id for item in self.face_segments}) != len(self.face_segments):
                raise ValueError('Duplicate segment identity')
            for item in self.face_segments:
                grouped.setdefault(item.track_id, []).append(item)
            for segments in grouped.values():
                ordered = sorted(segments, key=lambda item: item.start_frame)
                for left, right in zip(ordered, ordered[1:]):
                    if left.end_frame is None or left.end_frame >= right.start_frame:
                        raise ValueError('Face role intervals must not overlap')
        if self.trim_end_seconds is not None and self.trim_end_seconds <= (self.trim_start_seconds or 0):
            raise ValueError("Trim end must exceed trim start")
        if len({item.track_id for item in self.face_mappings}) != len(self.face_mappings):
            raise ValueError("Each target track may have only one FaceSet mapping")
        if self.processing == "remote" and not self.remote_consent:
            raise ValueError("Remote processing requires explicit per-job consent")
        return self

    def parameters_at(self, frame: int):
        base = self.workstation_parameters or WorkstationParameters()
        return self.parameter_curves.evaluate(frame, base) if self.parameter_curves else base


class JobProgress(Contract):
    frame: int = Field(default=0, ge=0, strict=True)
    total_frames: int | None = Field(default=None, ge=1, strict=True)
    fps: float = Field(default=0, ge=0)
    stage: str = Field(default="queued", min_length=1, max_length=64)
    stage_latency_ms: float = Field(default=0, ge=0)
    # Measured mean milliseconds per stage for the latest reporting window; unmeasured stages are absent.
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)

    @field_validator("stage_timings_ms", mode="before")
    @classmethod
    def known_stage_timings(cls, value):
        if not isinstance(value, dict):
            raise ValueError("stage_timings_ms must be an object")
        return validate_timings(value)

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
    passes: dict[str, str] = Field(default_factory=dict)
    pass_manifest_path: str | None = None
    baked_video: bool = True

    @field_validator("output_path", "manifest_path")
    @classmethod
    def owned_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or not value.startswith("roop_3d/.runtime/"):
            raise ValueError("Job output paths must remain inside the configured runtime")
        return path.as_posix()

    @field_validator('pass_manifest_path')
    @classmethod
    def optional_manifest(cls, value):
        return cls.owned_relative_path(value) if value is not None else None

    @field_validator('passes')
    @classmethod
    def pass_paths(cls, value):
        if set(value) - {'alpha', 'occluder', 'uv', 'normal', 'depth'}:
            raise ValueError('Unsupported output pass')
        return {name: cls.owned_relative_path(path) for name, path in value.items()}


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
