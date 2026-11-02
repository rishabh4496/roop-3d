"""Typed immutable FaceSet schemas, V2 manifest contracts, and query result models."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import ConfigDict, Field

from roop_3d.core.settings import Contract


class QualityMetrics(Contract):
    sharpness: float = Field(default=1.0, ge=0.0)
    detector_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    laplacian_variance: float = Field(default=0.0, ge=0.0)
    blur: float = Field(default=0.0, ge=0.0)
    face_pixels: float = Field(default=0.0, ge=0.0)


class FaceView(Contract):
    index: int = Field(ge=0)
    image_bytes: bytes
    embedding: tuple[float, ...] = Field(min_length=512, max_length=512)
    landmarks: tuple[tuple[float, float], ...]
    bbox: tuple[float, float, float, float]
    yaw: float = Field(ge=-90.0, le=90.0)
    pitch: float = Field(ge=-60.0, le=60.0)
    roll: float = Field(ge=-180.0, le=180.0)
    expression: dict[str, float] = Field(default_factory=dict)
    quality: QualityMetrics
    pose_bin: str = Field(min_length=1, max_length=32)
    is_synthetic: Literal[False] = False
    uv_map_path: str | None = None
    # Stage 3: optional dense (68-point) landmarks and an honesty flag. `landmarks` (5 points) stays the
    # contract every existing consumer uses; `landmarks_trusted=False` marks fabricated stand-in landmarks
    # (no detector result / archive without geometry) that a 3D reconstruction must never treat as observations.
    landmarks_68: tuple[tuple[float, float], ...] | None = Field(default=None, min_length=68, max_length=68)
    landmarks_trusted: bool = True


class PoseBinSummary(Contract):
    yaw_bin: str
    pitch_bin: str
    members: tuple[int, ...]
    support: int = Field(ge=0)
    representative_index: int = Field(ge=0)
    embedding: tuple[float, ...] = Field(min_length=512, max_length=512)


class FaceSetDocument(Contract):
    faceset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    name: str = Field(min_length=1, max_length=128)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    archive_format: Literal["legacy_v1", "legacy_v2", "v2_standard", "image_directory"]
    canonical_embedding: tuple[float, ...] = Field(min_length=512, max_length=512)
    medoid_embedding: tuple[float, ...] = Field(min_length=512, max_length=512)
    views: tuple[FaceView, ...]
    pose_bins: dict[str, PoseBinSummary]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConsentMetadata(Contract):
    consent_granted: bool = Field(default=False)
    consent_timestamp: str | None = None
    commercial_use_allowed: bool = Field(default=False)
    subject_id: str | None = None
    notes: str = Field(default="", max_length=1024)


class FaceSetV2Manifest(Contract):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True, allow_inf_nan=False, validate_default=True, hide_input_in_errors=True)
    schema_version: Literal[2] = 2
    schema_type: Literal["roop.fsz.v2", "roop.fsz"] = Field(alias="schema", default="roop.fsz.v2")
    faceset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    source_name: str = Field(min_length=1, max_length=128)
    created_at: str
    consent: ConsentMetadata
    identity: dict[str, Any]
    pose_bins: dict[str, dict[str, Any]]
    sources: list[dict[str, Any]]
    provenance: dict[str, Any]
    # Stage 0 index of the 3x3 (yaw x pitch) pose matrix: cell name -> member view indices. Informational:
    # readers always recompute the frozen vectors, so a stale or missing section cannot change a render.
    pose_matrix: dict[str, Any] | None = None


class ReferenceMatch(Contract):
    view_index: int = Field(ge=0)
    crop_bytes: bytes
    landmarks: tuple[tuple[float, float], ...]
    confidence: float = Field(ge=0.0, le=1.0)
    yaw: float = Field(ge=-90.0, le=90.0)
    pitch: float = Field(ge=-60.0, le=60.0)
    roll: float = Field(ge=-180.0, le=180.0)
    expression: dict[str, float] = Field(default_factory=dict)
    quality_score: float = Field(ge=0.0, le=1.0)
    pose_bin: str
    is_synthetic: Literal[False] = False
