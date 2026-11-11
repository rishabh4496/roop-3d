"""REST surface of the universal model download manager (`roop_3d.audit.local_models`).

    GET    /api/v1/models/manifest             every model + 3D asset: licence badge, install and hash state, disk usage
    POST   /api/v1/models/download             fetch all missing (or the selected) manifest entries
    POST   /api/v1/models/verify-and-repair    re-hash everything, set corrupt files aside, re-fetch what is missing or bad
    GET    /api/v1/models/batches/{id}         poll a batch that download / verify-and-repair started
    DELETE /api/v1/models/batches/{id}         cancel it (verified partial bytes are kept for resume)

Progress is pushed as `model_download_progress` frames on `/ws/v1/pipeline/status`.
Licences are not bypassed: an entry whose licence is not CC0/MIT/Apache/CC-BY, or is unverified, is only fetched when the
request carries `accepted_license: true`; otherwise it is reported under `skipped` with the reason.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Query, Request
from pydantic import Field

from roop_3d.audit.isolation import AuditError
from roop_3d.audit.local_models import UniversalModelDownloadManager
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract

router = APIRouter(prefix="/api/v1")

_STATUS = {"MODEL_UNKNOWN": 404, "BATCH_UNKNOWN": 404, "DOWNLOADS_CLOSED": 503, "MANIFEST_INVALID": 500}


class ManifestDownloadRequest(Contract):
    """All fields optional: an empty body means "every missing model that needs no licence acknowledgement"."""
    model_id: str | None = None
    models: list[str] = Field(default_factory=list, max_length=128)
    accepted_license: bool = Field(default=False, strict=True)
    commercial: bool = Field(default=False, strict=True)
    force: bool = Field(default=False, strict=True)


class VerifyRepairRequest(Contract):
    model_id: str | None = None
    models: list[str] = Field(default_factory=list, max_length=128)
    repair: bool = Field(default=True, strict=True)
    accepted_license: bool = Field(default=False, strict=True)
    commercial: bool = Field(default=False, strict=True)


def manager_of(request: Request) -> UniversalModelDownloadManager:
    manager = getattr(request.app.state, "universal_models", None)
    if manager is None:
        raise DomainError("MODEL_MANIFEST_UNAVAILABLE", "The universal model manifest could not be loaded; see the server log", 503)
    return manager


def as_domain(error: AuditError) -> DomainError:
    return DomainError(error.code, error.detail, _STATUS.get(error.code, 409))


def start_manifest_download(manager: UniversalModelDownloadManager, ids: list[str], *, accepted_license: bool,
                            commercial: bool, force: bool = False) -> dict[str, Any]:
    try:
        return manager.start_batch(ids or None, accepted_license=accepted_license, commercial=commercial, force=force)
    except AuditError as error:
        raise as_domain(error) from None


@router.get("/models/manifest")
def models_manifest(request: Request, refresh: Annotated[bool, Query(description="Re-hash every file instead of trusting the cache")] = False) -> dict[str, Any]:
    return manager_of(request).manifest(force=refresh)


@router.post("/models/verify-and-repair")
def verify_and_repair(request: Request, value: Annotated[VerifyRepairRequest | None, Body()] = None) -> dict[str, Any]:
    req = value or VerifyRepairRequest()
    ids = list(dict.fromkeys(([req.model_id] if req.model_id else []) + req.models))
    manager = manager_of(request)
    unknown = [i for i in ids if i not in manager.entries]
    if unknown:
        raise DomainError("MODEL_UNKNOWN", f"Not in the manifest: {', '.join(unknown)}", 404)
    try:
        return manager.verify_and_repair(ids or None, repair=req.repair, accepted_license=req.accepted_license,
                                         commercial=req.commercial)
    except AuditError as error:
        raise as_domain(error) from None


@router.get("/models/batches/{batch_id}")
def batch_status(batch_id: str, request: Request) -> dict[str, Any]:
    try:
        return manager_of(request).batch_status(batch_id)
    except AuditError as error:
        raise as_domain(error) from None


@router.delete("/models/batches/{batch_id}")
def cancel_batch(batch_id: str, request: Request) -> dict[str, Any]:
    try:
        return manager_of(request).cancel(batch_id)
    except AuditError as error:
        raise as_domain(error) from None
