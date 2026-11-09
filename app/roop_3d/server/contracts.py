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
    id: str | None = None
    model_id: str
    name: str
    category: str = "other"
    native_resolution: int | None = None
    installed: bool = False
    selectable: bool = False
    vram_loaded: bool = False
    trt_cached: bool = False
    download_url: str | None = None
    supports_3d_hybrid: bool = False
    local_format: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    state: Evidence
    family_hint: str
    verified_at: str | None
    license_state: Evidence
    provenance_state: Evidence
    inference_state: Evidence
    commercial_permission: Evidence
    role: Role | None = None
    license_id: str = "UNKNOWN"
    commercial_use: Literal["allow", "deny", "unknown"] = "unknown"
    license_notice: str = ""
    manual_acceptance_required: bool = False
    # A weight physically present in app/models is verified, governance-approved and never blocked
    # (models/local_trust.py). ``governance_approved`` is None where the governance gate was not evaluated.
    verified: bool = False
    governance_approved: bool | None = None
    blocked: bool = False
    local_trust: bool = False


class Capabilities(Contract):
    schema_version: Literal[1] = 1
    enabled_modes: tuple[str, ...]
    workstation: dict[str, Any] = Field(default_factory=dict)
    rendering_worker_available: bool = False
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
    redistribution_permission: Literal["allow", "deny", "unknown"] = "unknown"
    attribution_requirements: str = "UNKNOWN"
    required_base_model: str = "UNKNOWN"
    related_license_urls: list[str] = Field(default_factory=list)
    base_model_license: dict[str, Any] = Field(default_factory=dict)
