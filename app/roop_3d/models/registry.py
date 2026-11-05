"""Authoritative artifact identities; byte, graph and rights evidence stay separate."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import Field, field_validator, model_validator

from roop_3d.audit.isolation import relative_path, safe_path
from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract

Role = Literal["detector", "recognizer", "identity_swapper", "liveportrait_appearance",
               "liveportrait_motion", "liveportrait_warp", "liveportrait_stitch", "face_parser",
               "occlusion_mask", "face_restorer", "flame_geometry", "lip_sync", "diffusion_base",
               "diffusion_adapter", "embedding_converter", "age_modifier", "frame_upscaler",
               "flame_encoder", "dense_mesh"]
RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
            *(f"lpt{i}" for i in range(1, 10))}


def filename(value: str) -> str:
    path = relative_path(value)
    if len(path.parts) != 1 or path.stem.casefold() in RESERVED or any(c in value for c in '<>"|?*'):
        raise ValueError("A portable, non-device basename is required")
    return value


class Locator(Contract):
    scope: str = Field(min_length=1, max_length=64)
    path: str

    @field_validator("path")
    @classmethod
    def canonical(cls, value: str) -> str:
        return relative_path(value).as_posix()


class ModelRecord(Contract):
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,95}$")
    name: str = Field(min_length=1, max_length=256)
    # Null explicitly preserves ancillary artifacts outside the requested role vocabulary.
    role: Role | None
    role_evidence: str
    upstream_url: str | None = None
    filenames: tuple[str, ...] = Field(default=(), max_length=1)
    weight_filenames: tuple[str, ...] = Field(default=(), max_length=1)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    byte_size: int | None = Field(default=None, gt=0, le=2**40, strict=True)
    input_output_contract: dict[str, Any]
    supported_providers: tuple[str, ...] = ()
    provider_evidence: str = "UNKNOWN: provider execution has not been verified"
    license_id: str = "UNKNOWN"
    commercial_use: Literal["allow", "deny", "unknown"] = "unknown"
    status: Literal["verified_local", "verified_remote", "candidate", "blocked"]
    locator: Locator | None = None
    evidence: str
    verified_at: str | None = None
    provenance_state: Literal["VERIFIED_LOCAL", "VERIFIED_REMOTE", "CANDIDATE", "UNKNOWN", "BLOCKED"] = "UNKNOWN"
    license_notice: str
    license_url: str | None = None
    related_license_urls: tuple[str, ...] = ()
    manual_acceptance_required: bool = False
    # True when the exact bytes are verified at a public source that publishes no license
    # for them. Download needs explicit acknowledgement and never grants commercial use or
    # inference activation; those gates still require a reviewed license.
    rights_unpublished: bool = False
    family_hint: str = "other"
    version: str = "UNKNOWN"
    model_family: str = "UNKNOWN"
    engine_type: str = "UNKNOWN"
    purpose: str = "UNKNOWN"
    source_repository: str | None = None
    # Governance fields are mandatory in the runtime contract. UNKNOWN is an
    # explicit evidence state, never an inferred permission or capability.
    original_repository: str = "UNKNOWN"
    source_revision: str = "UNKNOWN"
    version_family: str = "UNKNOWN"
    architecture: str = "UNKNOWN"
    required_base_model: str = "UNKNOWN"
    base_model_license_id: str = "UNKNOWN"
    base_model_license_url: str | None = None
    base_model_commercial_use: Literal["allow", "deny", "unknown"] = "unknown"
    base_model_redistribution_permission: Literal["allow", "deny", "unknown"] = "unknown"
    base_model_attribution_requirements: str = "UNKNOWN"
    redistribution_permission: Literal["allow", "deny", "unknown"] = "unknown"
    attribution_requirements: str = "UNKNOWN"
    hardware_requirements: dict[str, Any] = Field(default_factory=lambda: {"state": "UNKNOWN"})
    precision_requirements: tuple[str, ...] = ()
    dependency_compatibility: dict[str, Any] = Field(default_factory=lambda: {"state": "UNKNOWN"})
    serialization_format: str = "UNKNOWN"
    security_status: Literal["UNKNOWN", "REVIEW_REQUIRED", "VERIFIED", "BLOCKED"] = "UNKNOWN"
    minimum_vram_gib: float | None = Field(default=None, gt=0, le=1024)
    recommended_vram_gib: float | None = Field(default=None, gt=0, le=1024)
    dependencies: tuple[str, ...] = ()
    supported_precision: tuple[str, ...] = ()
    benchmark_status: str = "NOT_RUN"
    inference_availability: Literal["UNKNOWN", "UNAVAILABLE", "AVAILABLE", "EXPERIMENTAL"] = "UNKNOWN"

    @field_validator("model_id")
    @classmethod
    def portable_id(cls, value: str) -> str:
        return filename(value)

    @field_validator("filenames", "weight_filenames")
    @classmethod
    def portable_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(filename(value) for value in values)

    @model_validator(mode="after")
    def evidence_required(self) -> "ModelRecord":
        if self.weight_filenames and self.filenames and self.weight_filenames != self.filenames:
            raise ValueError("weight_filenames and legacy filenames must identify the same immutable artifacts")
        if not self.weight_filenames and self.filenames:
            object.__setattr__(self, "weight_filenames", self.filenames)
        if self.weight_filenames and not self.filenames:
            object.__setattr__(self, "filenames", self.weight_filenames)
        if self.status.startswith("verified_") and (not self.expected_sha256 or not self.byte_size or not self.filenames):
            raise ValueError("Verified identities require observed hash, size and filename")
        if self.status == "verified_remote" and not self.upstream_url:
            raise ValueError("Remote verification requires an upstream URL")
        # Preserve prior manifests while materializing governance fields from
        # facts already recorded there. Unknowns remain explicit.
        if self.original_repository == "UNKNOWN" and self.source_repository:
            object.__setattr__(self, "original_repository", self.source_repository)
        if self.source_revision == "UNKNOWN" and self.upstream_url:
            parts = unquote(urlsplit(self.upstream_url).path).split("/")
            if "resolve" in parts:
                index = parts.index("resolve")
                if index + 1 < len(parts) and re.fullmatch(r"[0-9a-fA-F]{40}", parts[index + 1]):
                    object.__setattr__(self, "source_revision", parts[index + 1])
        if self.serialization_format == "UNKNOWN" and self.filenames:
            object.__setattr__(self, "serialization_format", Path(self.filenames[0]).suffix.lstrip(".").upper() or "UNKNOWN")
        return self


class Manifest(Contract):
    schema_version: Literal[1] = 1
    evidence_scope: str
    source_documents: dict[str, str] = Field(default_factory=dict)
    models: tuple[ModelRecord, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> "Manifest":
        ids = [item.model_id for item in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate model identities")
        return self


def _apply_verified_sources(manifest: Manifest, document: dict[str, Any]) -> Manifest:
    """Attach reviewed public sources without rewriting the preserved local evidence.

    Each entry must name the same SHA-256 and size as the registered artifact; a mismatch
    makes the registry invalid rather than silently trusting the overlay.
    """
    sources = document["sources"]
    models = []
    for record in manifest.models:
        source = sources.get(record.model_id)
        if source is None or record.status == "blocked":
            models.append(record)
            continue
        if source["sha256"] != record.expected_sha256 or source["byte_size"] != record.byte_size:
            raise ValueError("Verified source does not match the registered artifact identity")
        update = {"upstream_url": source["upstream_url"], "original_repository": source["original_repository"],
                  "source_revision": source["source_revision"], "source_repository": source["original_repository"],
                  "provenance_state": "VERIFIED_REMOTE", "evidence": record.evidence + " Public source verified by identical SHA-256 and size: "
                  + source["upstream_url"]}
        if record.license_id == "UNKNOWN":
            update.update(rights_unpublished=True, manual_acceptance_required=True,
                          license_notice=("The public source publishes no license for these weights. Downloading requires an explicit "
                                          "acknowledgement that usage rights are unverified; commercial use and inference activation "
                                          "stay blocked until a license is reviewed."))
        models.append(ModelRecord.model_validate({**record.model_dump(), **update}))
    return Manifest(evidence_scope=manifest.evidence_scope + " " + document["evidence_scope"],
                    source_documents=manifest.source_documents, models=tuple(models))


class ModelRegistry:
    def __init__(self, root: Path | None = None, *, manifest: Manifest | None = None):
        self.root = root or application_root()
        try:
            self.manifest = manifest or Manifest.model_validate_json(
                safe_path(self.root, "roop_3d/models/model_manifest.json").read_text("utf-8"))
            for relative, digest in self.manifest.source_documents.items():
                if hashlib.sha256(safe_path(self.root, relative).read_bytes()).hexdigest() != digest:
                    raise ValueError("Preserved discovery evidence changed")
            # Supplement the preserved discovery manifest; never rewrite its local evidence.
            if manifest is None:
                for relative in ("roop_3d/models/stage16_models.json", "roop_3d/models/stage6_models.json"):
                    supplemental = safe_path(self.root, relative)
                    if supplemental.is_file():
                        extension = Manifest.model_validate_json(supplemental.read_text("utf-8"))
                        self.manifest = Manifest(
                            evidence_scope=self.manifest.evidence_scope + " " + extension.evidence_scope,
                            source_documents=self.manifest.source_documents,
                            models=self.manifest.models + extension.models)
                overlay = safe_path(self.root, "roop_3d/models/verified_sources.json")
                if overlay.is_file():
                    self.manifest = _apply_verified_sources(self.manifest, json.loads(overlay.read_text("utf-8")))
        except (OSError, ValueError):
            raise DomainError("MODEL_REGISTRY_INVALID", "Model manifest or its discovery evidence is invalid", 503) from None
        self._records = {item.model_id: item for item in self.manifest.models}

    def get(self, model_id: str) -> ModelRecord:
        try:
            return self._records[model_id]
        except KeyError:
            raise DomainError("MODEL_NOT_FOUND", "Model is not in the authoritative registry", 404) from None

    def records(self) -> tuple[ModelRecord, ...]:
        return self.manifest.models


def authorize(record: ModelRecord, *, action: Literal["resolve", "activate", "download"],
              accepted_license: bool = False, commercial: bool = False) -> None:
    """Acknowledgement never overrides a blocked status or grants commercial rights."""
    if record.status == "blocked":
        raise DomainError("MODEL_BLOCKED", record.license_notice, 403)
    if commercial and record.commercial_use != "allow":
        raise DomainError("COMMERCIAL_USE_NOT_VERIFIED", record.license_notice, 403)
    if commercial and record.required_base_model not in {"NONE", "UNKNOWN"} and record.base_model_commercial_use != "allow":
        raise DomainError("BASE_MODEL_COMMERCIAL_USE_NOT_VERIFIED", record.license_notice, 403)
    # Unpublished rights allow a bytes-only download behind an explicit acknowledgement;
    # activation (inference) keeps the full license review requirement.
    unpublished_download = action == "download" and record.rights_unpublished
    if action in {"activate", "download"} and not unpublished_download and (
            record.license_id == "UNKNOWN" or not record.license_url
            or record.commercial_use == "unknown" or record.redistribution_permission == "unknown"
            or record.attribution_requirements == "UNKNOWN"):
        raise DomainError("LICENSE_UNVERIFIED", "License, commercial use, redistribution, and attribution terms must be reviewed before activation", 403)
    if action in {"activate", "download"} and record.manual_acceptance_required and not accepted_license:
        raise DomainError("LICENSE_ACCEPTANCE_REQUIRED", record.license_notice, 403)
    if action == "download":
        if (record.status not in {"verified_local", "verified_remote"}
                or record.provenance_state != "VERIFIED_REMOTE" or not record.upstream_url
                or not record.expected_sha256 or not record.byte_size or not record.filenames):
            raise DomainError("DOWNLOAD_EVIDENCE_REQUIRED", "Reviewed upstream URL, digest, size and provenance are required", 409)


def license_info(record: ModelRecord) -> dict[str, Any]:
    try:
        authorize(record, action="download")
        allowed, reason = True, None
    except DomainError as error:
        allowed, reason = False, error.code
    return {"model_id": record.model_id, "license_id": record.license_id,
            "commercial_use": record.commercial_use, "notice": record.license_notice,
            "license_url": record.license_url, "manual_acceptance_required": record.manual_acceptance_required,
            "redistribution_permission": record.redistribution_permission,
            "attribution_requirements": record.attribution_requirements,
            "required_base_model": record.required_base_model,
            "related_license_urls": list(record.related_license_urls),
            "base_model_license": {"model": record.required_base_model, "license_id": record.base_model_license_id,
                                   "license_url": record.base_model_license_url,
                                   "commercial_use": record.base_model_commercial_use,
                                   "redistribution_permission": record.base_model_redistribution_permission,
                                   "attribution_requirements": record.base_model_attribution_requirements},
            "automatic_download_allowed": allowed, "download_gate": reason}
