"""Closed vocabularies and frozen contracts for the Stage 14 capability registry.

Everything here is declarative data. Selection logic lives in ``selector.py``; nothing in this
module loads a model, probes hardware, or touches the runtime.
"""
from __future__ import annotations

from enum import Enum
import hashlib
import json
from typing import Any, Literal

from pydantic import Field, model_validator

from roop_3d.core.settings import Contract

SCHEMA_VERSION = 1
BASELINE_PROFILE = "baseline"
ADAPTIVE_PROFILE = "adaptive"

# Categories in pipeline order. The order is also the deterministic iteration order everywhere.
Category = Literal["video_decoding", "face_detection", "face_recognition", "dense_landmarks",
                   "identity_swapping", "reconstruction_3d", "expression_transfer",
                   "restoration", "occlusion_segmentation", "video_encoding"]
CATEGORY_ORDER: tuple[str, ...] = ("video_decoding", "face_detection", "face_recognition", "dense_landmarks",
                                   "identity_swapping", "reconstruction_3d", "expression_transfer",
                                   "restoration", "occlusion_segmentation", "video_encoding")
NON_IMAGE_CATEGORIES = frozenset({"video_decoding", "video_encoding"})

DataKind = Literal["video_file", "identity_source", "frame", "face_detections", "face_embedding",
                   "landmarks_106", "landmarks_203", "swapped_crop", "identity_asset_3d",
                   "visible_face_map", "restored_crop", "composited_frame", "encoded_video"]
# Kinds the surrounding pipeline provides without an Engine: the source media, the FaceSet, and the
# built-in compositor that turns a (possibly restored) crop plus the frame into a composited frame.
SOURCE_KINDS = frozenset({"video_file", "identity_source"})
PIPELINE_DERIVATIONS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"swapped_crop", "frame"}), "composited_frame"),
    (frozenset({"restored_crop", "frame"}), "composited_frame"),
)

Tier = Literal["cpu", "6gb", "12gb"]
TIERS: tuple[str, ...] = ("cpu", "6gb", "12gb")
Mode = Literal["manual", "balanced", "quality"]
Precision = Literal["FP32", "FP16"]
Provider = Literal["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider",
                   "SoftwareCodec", "HardwareCodec", "CPUNative"]
CPU_PROVIDERS = frozenset({"CPUExecutionProvider", "SoftwareCodec", "CPUNative"})
Residency = Literal["heavy", "light", "none"]

ENGINE_ID_PATTERN = r"^[a-z0-9][a-z0-9_]{0,95}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ReasonCode(str, Enum):
    """Closed rejection/notice vocabulary used by the decision report."""
    ADAPTER_NOT_REGISTERED = "ADAPTER_NOT_REGISTERED"
    NOT_GOVERNANCE_READY = "NOT_GOVERNANCE_READY"
    TIER_UNSUPPORTED = "TIER_UNSUPPORTED"
    PROVIDER_UNSUPPORTED = "PROVIDER_UNSUPPORTED"
    PRECISION_FORBIDDEN = "PRECISION_FORBIDDEN"
    RESOLUTION_UNSUPPORTED = "RESOLUTION_UNSUPPORTED"
    CONTRACT_MISMATCH = "CONTRACT_MISMATCH"
    RUNNER_CANNOT_EXPRESS = "RUNNER_CANNOT_EXPRESS"
    EXCLUSIVE_CONFLICT = "EXCLUSIVE_CONFLICT"
    EXCEEDS_VRAM_BUDGET = "EXCEEDS_VRAM_BUDGET"
    PEAK_VRAM_UNKNOWN = "PEAK_VRAM_UNKNOWN"
    REQUIRED_CATEGORY_MISSING = "REQUIRED_CATEGORY_MISSING"
    NOT_IN_PROFILE = "NOT_IN_PROFILE"
    NO_ELIGIBLE_ENGINE = "NO_ELIGIBLE_ENGINE"
    NO_EVIDENCE_FOR_PROFILE = "NO_EVIDENCE_FOR_PROFILE"
    EVIDENCE_STALE = "EVIDENCE_STALE"
    EVIDENCE_NOT_REAL = "EVIDENCE_NOT_REAL"
    SAFETY_VETO = "SAFETY_VETO"
    BELOW_BASELINE = "BELOW_BASELINE"
    DOMINATED_BY_SELECTED = "DOMINATED_BY_SELECTED"
    NOT_SELECTED_MANUAL = "NOT_SELECTED_MANUAL"


class ModelRef(Contract):
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,95}$")
    role: str = Field(min_length=1, max_length=64)


class Resolution(Contract):
    min: int | None = Field(default=None, ge=1, le=1_000_000)
    max: int | None = Field(default=None, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def ordered(self) -> "Resolution":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Resolution min exceeds max")
        return self


class Fallback(Contract):
    """Declared, in-engine degradation. ``degrade_to`` hands a frame to another Engine's proven path."""
    mode: Literal["none", "withhold", "degrade_to"]
    target_engine_id: str | None = Field(default=None, pattern=ENGINE_ID_PATTERN)
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def coherent(self) -> "Fallback":
        if (self.mode == "degrade_to") != (self.target_engine_id is not None):
            raise ValueError("degrade_to requires exactly a target engine")
        if self.mode == "degrade_to" and not self.reason_codes:
            raise ValueError("degrade_to must list its reason codes")
        return self


class EngineEntry(Contract):
    engine_id: str = Field(pattern=ENGINE_ID_PATTERN)
    category: Category
    adapter_id: str = Field(pattern=ENGINE_ID_PATTERN)
    models: tuple[ModelRef, ...] = ()
    inputs: tuple[DataKind, ...] = Field(min_length=1)
    outputs: tuple[DataKind, ...] = Field(min_length=1)
    resolution: Resolution
    precisions: tuple[Precision, ...] = Field(min_length=1)
    providers: tuple[Provider, ...] = Field(min_length=1)
    hardware_tiers: tuple[Tier, ...] = Field(min_length=1)
    declared_peak_vram_mib: int | None = Field(default=None, ge=0, le=1_000_000)
    residency: Residency
    exclusive_with: tuple[str, ...] = ()
    description: str = Field(max_length=240)
    fallback: Fallback

    @model_validator(mode="after")
    def coherent(self) -> "EngineEntry":
        if self.category not in NON_IMAGE_CATEGORIES and self.resolution.min is None and self.resolution.max is None:
            raise ValueError("Image-processing engines must bound their resolution")
        if self.residency == "none" and self.models:
            raise ValueError("An engine with models cannot have residency 'none'")
        if self.residency != "none" and not self.models and self.declared_peak_vram_mib:
            raise ValueError("A model-free engine cannot declare VRAM")
        if len(set(self.exclusive_with)) != len(self.exclusive_with) or self.engine_id in self.exclusive_with:
            raise ValueError("exclusive_with must be unique and exclude the engine itself")
        return self

    def digest(self) -> str:
        return digest_of(self.model_dump(mode="json"))


class EngineManifest(Contract):
    schema_version: Literal[1] = 1
    content_version: str = Field(min_length=1, max_length=64)
    engines: tuple[EngineEntry, ...] = Field(min_length=1)


class CategoryRule(Contract):
    category: Category
    allowed_engines: tuple[str, ...] = Field(min_length=1)


class RenderProfile(Contract):
    profile_id: str = Field(pattern=ENGINE_ID_PATTERN)
    description: str = Field(max_length=240)
    required: tuple[CategoryRule, ...] = Field(min_length=1)
    optional: tuple[CategoryRule, ...] = ()

    @model_validator(mode="after")
    def disjoint(self) -> "RenderProfile":
        names = [rule.category for rule in self.required + self.optional]
        if len(names) != len(set(names)):
            raise ValueError("A category may appear only once per profile")
        return self


class RenderProfiles(Contract):
    schema_version: Literal[1] = 1
    content_version: str = Field(min_length=1, max_length=64)
    profiles: tuple[RenderProfile, ...] = Field(min_length=1)


class TraitMap(Contract):
    trait: Literal["frontal", "large_yaw"]
    scene_categories: tuple[str, ...] = Field(min_length=1)


class SceneRules(Contract):
    """Fixed, versioned trait thresholds. v1 consumes only signals media inspection actually produces."""
    schema_version: Literal[1] = 1
    content_version: str = Field(min_length=1, max_length=64)
    large_yaw_degrees: float = Field(gt=0, lt=90)
    trait_map: tuple[TraitMap, ...] = Field(min_length=2)
    low_confidence: Literal["withhold"] = "withhold"
    occlusion: Literal["not_assessed"] = "not_assessed"

    @model_validator(mode="after")
    def complete(self) -> "SceneRules":
        if {item.trait for item in self.trait_map} != {"frontal", "large_yaw"}:
            raise ValueError("trait_map must cover exactly frontal and large_yaw")
        return self


class EvidenceEngineRef(Contract):
    engine_id: str = Field(pattern=ENGINE_ID_PATTERN)
    entry_digest: str = Field(pattern=SHA256_PATTERN)
    model_sha256s: tuple[str, ...] = ()
    precision: Precision
    provider: Provider

    @model_validator(mode="after")
    def valid_hashes(self) -> "EvidenceEngineRef":
        import re
        if any(not re.fullmatch(SHA256_PATTERN, value) for value in self.model_sha256s):
            raise ValueError("Every model hash must be a 64-hex sha256")
        return self


class SceneScore(Contract):
    category: str
    score: float = Field(ge=0, le=100)


class EvidenceRecord(Contract):
    """A measurement of one concrete plan on one tier. Only ``measured_with == real_models`` is evidence."""
    evidence_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,95}$")
    tier: Tier
    plan_digest: str = Field(pattern=SHA256_PATTERN)
    engines: tuple[EvidenceEngineRef, ...] = Field(min_length=1)
    measured_with: Literal["real_models", "simulated"]
    composite_overall: float = Field(ge=0, le=100)
    scene_scores: tuple[SceneScore, ...] = ()
    latency_ms_p50: float = Field(gt=0)
    peak_vram_mib: int | None = Field(default=None, ge=0)
    safety_vetoes_passed: bool
    dataset_digest: str = Field(pattern=SHA256_PATTERN)
    recorded_at: str = Field(min_length=1, max_length=40)


class EvidenceFile(Contract):
    schema_version: Literal[1] = 1
    records: tuple[EvidenceRecord, ...] = ()

    @model_validator(mode="after")
    def unique(self) -> "EvidenceFile":
        ids = [item.evidence_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence ids")
        return self


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def digest_of(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
