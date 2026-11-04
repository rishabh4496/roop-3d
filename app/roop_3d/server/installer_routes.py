"""REST API endpoints for environment preflight, offline models, updates, and recovery."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from roop_3d.installer.model_manager import ModelManager
from roop_3d.installer.offline_cache import check_offline_readiness
from roop_3d.installer.preflight import run_environment_preflight
from roop_3d.installer.update_manager import (
    check_update_status,
    create_configuration_backup,
    rollback_to_backup,
)
from roop_3d.installer.verification import recover_interrupted_installation

router = APIRouter(prefix="/api/v1/installer", tags=["installer"])


class ScanRequest(BaseModel):
    dir_path: str = Field(min_length=1, max_length=1024)


class ImportBundleRequest(BaseModel):
    bundle_path: str = Field(min_length=1, max_length=1024)


class RollbackRequest(BaseModel):
    backup_id: str | None = None


@router.get("/preflight")
def get_preflight(request: Request) -> dict[str, Any]:
    root: Path = request.app.state.root
    return run_environment_preflight(root)


@router.get("/offline-health")
def get_offline_health(request: Request) -> dict[str, Any]:
    root: Path = request.app.state.root
    registry = getattr(request.app.state.models, "registry", None)
    return check_offline_readiness(root, registry)


@router.post("/models/scan")
def scan_local_directory(request: Request, payload: ScanRequest) -> dict[str, Any]:
    root: Path = request.app.state.root
    registry = getattr(request.app.state.models, "registry", None)
    manager = ModelManager(root, registry)
    return manager.scan_directory(Path(payload.dir_path))


@router.post("/models/import")
def import_model_bundle(request: Request, payload: ImportBundleRequest) -> dict[str, Any]:
    root: Path = request.app.state.root
    registry = getattr(request.app.state.models, "registry", None)
    manager = ModelManager(root, registry)
    return manager.import_bundle(Path(payload.bundle_path))


@router.post("/models/verify")
def verify_models(request: Request) -> dict[str, Any]:
    root: Path = request.app.state.root
    registry = getattr(request.app.state.models, "registry", None)
    manager = ModelManager(root, registry)
    return manager.verify_all_models()


@router.get("/updates/check")
def check_updates(request: Request) -> dict[str, Any]:
    root: Path = request.app.state.root
    return check_update_status(root)


@router.post("/updates/backup")
def create_backup(request: Request) -> dict[str, Any]:
    root: Path = request.app.state.root
    return create_configuration_backup(root)


@router.post("/updates/rollback")
def perform_rollback(request: Request, payload: RollbackRequest) -> dict[str, Any]:
    root: Path = request.app.state.root
    return rollback_to_backup(root, payload.backup_id)


@router.post("/recover")
def recover_installation(request: Request) -> dict[str, Any]:
    root: Path = request.app.state.root
    return recover_interrupted_installation(root)
