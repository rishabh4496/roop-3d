"""Authoritative artifact identities; byte, graph and rights evidence stay separate."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from roop_3d.audit.isolation import relative_path, safe_path
from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract

Role = Literal["detector", "recognizer", "identity_swapper", "liveportrait_appearance",
               "liveportrait_motion", "liveportrait_warp", "liveportrait_stitch", "face_parser",
               "occlusion_mask", "face_restorer", "flame_geometry", "lip_sync", "diffusion_base",
               "diffusion_adapter"]
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
    manual_acceptance_required: bool = False
    family_hint: str = "other"

    @field_validator("model_id")
    @classmethod
    def portable_id(cls, value: str) -> str:
        return filename(value)

    @field_validator("filenames")
    @classmethod
    def portable_names(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(filename(value) for value in values)

    @model_validator(mode="after")
    def evidence_required(self) -> "ModelRecord":
        if self.status.startswith("verified_") and (not self.expected_sha256 or not self.byte_size or not self.filenames):
            raise ValueError("Verified identities require observed hash, size and filename")
        if self.status == "verified_remote" and not self.upstream_url:
            raise ValueError("Remote verification requires an upstream URL")
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


class ModelRegistry:
    def __init__(self, root: Path | None = None, *, manifest: Manifest | None = None):
        self.root = root or application_root()
        try:
            self.manifest = manifest or Manifest.model_validate_json(
                safe_path(self.root, "roop_3d/models/model_manifest.json").read_text("utf-8"))
            for relative, digest in self.manifest.source_documents.items():
                if hashlib.sha256(safe_path(self.root, relative).read_bytes()).hexdigest() != digest:
                    raise ValueError("Preserved discovery evidence changed")
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
    if action in {"activate", "download"} and record.manual_acceptance_required and not accepted_license:
        raise DomainError("LICENSE_ACCEPTANCE_REQUIRED", record.license_notice, 403)
    if action == "download":
        if record.license_id == "UNKNOWN":
            raise DomainError("LICENSE_UNVERIFIED", record.license_notice, 403)
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
            "automatic_download_allowed": allowed, "download_gate": reason}
