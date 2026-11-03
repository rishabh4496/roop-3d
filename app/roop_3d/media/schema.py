"""Finite, bounded contracts for local target video inspection and previews."""
from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from roop_3d.core.settings import Contract


class MediaRequest(Contract):
    path: str = Field(min_length=1, max_length=1024)


class InspectRequest(MediaRequest):
    # Opt-in complete traversal: legacy calls retain their frame/keyframe budgets.
    # Time, image-size, track-count and report-storage bounds still apply.
    full_video: bool = False
    max_samples: int = Field(default=32, ge=1, le=128, strict=True)
    max_decoded_frames: int = Field(default=10000, ge=1, le=10000, strict=True)
    detection_threshold: float = Field(default=0.5, ge=0.1, le=0.99)
    # Opt in to the identity-lock tracker (track states, confidence, strict reacquisition) and the
    # identity report. Off keeps the original inspection behaviour and response bit-for-bit.
    identity_lock: bool = False
    time_budget_seconds: float = Field(default=30.0, ge=1.0, le=900.0)


class AudioTrack(Contract):
    stream_index: int
    codec: str
    sample_rate: int | None
    channels: int
    language: str | None = None
    time_base: str | None = None
    channel_layout: str | None = None


class MediaProbe(Contract):
    container: str
    duration_seconds: float | None
    fps: float | None
    fps_rational: str | None
    width: int
    height: int
    video_stream_index: int
    video_codec: str
    declared_frame_count: int | None
    vfr: bool | None
    vfr_state: Literal["VERIFIED_LOCAL", "UNKNOWN"]
    timing_sample_frames: int
    timing_sample_complete: bool
    audio_tracks: tuple[AudioTrack, ...]
    time_base: str | None = None
    nominal_frame_rate: str | None = None
    pixel_format: str | None = None
    color_space: str = "UNKNOWN"
    color_range: int | None = None
    rotation_degrees: int = 0


class BoundingBox(Contract):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> "BoundingBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive area")
        return self


class PoseRange(Contract):
    yaw_degrees: tuple[float, float] | None = None
    pitch_degrees: tuple[float, float] | None = None
    roll_degrees: tuple[float, float] | None = None
    method: Literal["eye_line_roll_only", "five_point_geometric_estimate"] = "eye_line_roll_only"
    yaw_pitch_state: Literal["UNKNOWN", "CANDIDATE"] = "UNKNOWN"


class FaceTrack(Contract):
    track_id: str = Field(pattern=r"^track_[0-9]{4}$")
    persistent_id: int | None = Field(default=None, ge=0)
    scene_id: int = Field(default=0, ge=0)
    landmark_count: int = Field(default=5, ge=5, le=203)
    first_frame: int = Field(ge=0)
    last_frame: int = Field(ge=0)
    first_timestamp_seconds: float = Field(ge=0)
    last_timestamp_seconds: float = Field(ge=0)
    bounding_box: BoundingBox
    pose_range: PoseRange
    thumbnail_base64: str
    thumbnail_mime: Literal["image/jpeg"] = "image/jpeg"
    sampled_detections: int = Field(ge=1)


class FaceInspection(Contract):
    inspection_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    detector_model_id: str
    detector_sha256: str
    tracks: tuple[FaceTrack, ...]
    decoded_frames: int
    sampled_frames: int
    truncated: bool
    tracking_method: Literal["sampled_spatial_iou", "iou_arcface_hungarian", "identity_lock_v1"] = "sampled_spatial_iou"
    identity_continuity_verified: Literal[False] = False
    warnings: tuple[str, ...] = ()
    scene_cut_frames: tuple[int, ...] = ()
    recognizer_model_id: str | None = None
    recognizer_sha256: str | None = None
    # Present only for tracking_method == "identity_lock_v1".
    tracking_policy_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    detection_threshold: float | None = Field(default=None, ge=0.1, le=0.99)
    identity_report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class TrackToFaceSetMap(Contract):
    track_id: str = Field(pattern=r"^track_[0-9]{4}$")
    faceset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
