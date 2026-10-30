"""REST response contracts exported in the local OpenAPI document."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import Field
from roop_3d.core.settings import Contract
from roop_3d.models.registry import Role

Evidence = Literal["VERIFIED_LOCAL", "VERIFIED_REMOTE", "CANDIDATE", "UNKNOWN", "BLOCKED", "UNSUPPORTED"]


class ErrorDetail(Contract):
    code: str
    message: str
    request_id: str | None


class ErrorEnvelope(Contract):
    error: ErrorDetail


class IsolationStatus(Contract):
    status: Literal["passed", "failed", "unknown"]
    checked_at: str | None
    scope: str
    checked_files: int | None
    report: str | None


class Health(Contract):
    status: Literal["ok"] = "ok"
    application: Literal["roop_3d"] = "roop_3d"
    version: str
    pinokio_app_id: Literal["roop-3d"] = "roop-3d"
    standalone: Literal[True] = True
    uptime_seconds: float
    isolation: IsolationStatus


class ModelSummary(Contract):
    model_id: str
    name: str
    sha256: str
    size_bytes: int
    state: Evidence
    family_hint: str
    verified_at: str
    license_state: Evidence
    provenance_state: Evidence
    inference_state: Evidence
    commercial_permission: Evidence
    role: Role | None = None
    license_id: str = "UNKNOWN"
    commercial_use: Literal["allow", "deny", "unknown"] = "unknown"
    license_notice: str = ""
    manual_acceptance_required: bool = False


class Capabilities(Contract):
    schema_version: Literal[1] = 1
    enabled_modes: tuple[str, ...]
    rendering_worker_available: Literal[False] = False
    remote_adapter_state: Literal["UNSUPPORTED"] = "UNSUPPORTED"
    automatic_model_download_state: Literal["GATED"] = "GATED"
    models: tuple[ModelSummary, ...]
    model_evidence_scope: str = "Recorded artifact hashes; exact bytes reverified before detector inference"
    legal_parameter_bounds: dict[str, Any]
    inspection_parameter_bounds: dict[str, Any]
    job_parameter_bounds: dict[str, Any]
    event_schema: dict[str, Any]
    websocket_delivery: str = "Process-local, bounded, no replay; stream.ready/gap requires GET jobs refresh"
    model_search_policy: str = "Custom search roots are runtime-only; external models use exact read-only registry locators"
    model_registry_endpoint: str = "/api/v1/models/registry"
    model_download_policy: str = "Reviewed HTTPS identity and license required; installed verified local bytes are reused first"


class ModelAvailability(Contract):
    model_id: str
    state: Literal["VERIFIED_LOCAL"]
    downloaded: bool = False
    bytes_verified: Literal[True] = True
    resumed: bool = False


class ModelDownloadRequest(Contract):
    accepted_license: bool = Field(default=False, strict=True)
    commercial: bool = Field(default=False, strict=True)


class ModelLicenseNotice(Contract):
    model_id: str
    license_id: str
    commercial_use: Literal["allow", "deny", "unknown"]
    notice: str
    license_url: str | None
    manual_acceptance_required: bool
    automatic_download_allowed: bool
    download_gate: str | None
