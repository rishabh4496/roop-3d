"""Target-identity API: contact sheet from an identity inspection, and hash-bound locks.

Nothing here runs inference or loads a model. Biometric signatures stay in the local
sidecar; responses carry thumbnails, time ranges, confidence and assignment status only.
"""
from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

from roop_3d.core.errors import DomainError
from roop_3d.identity.lock import LockCreate, LockEdit, LockEnvironment, LockStore, file_sha256
from roop_3d.identity.report import public_tracks, read_report
from roop_3d.media.schema import FaceInspection

router = APIRouter(prefix="/api/v1/identity")


def store(request: Request) -> LockStore:
    state = request.app.state
    return LockStore(LockEnvironment(state.root, state.media.path, lambda: state.settings, state.media.detector.catalog))


@router.get("/inspections/{inspection_id}")
def contact_sheet(inspection_id: str, request: Request, path: str = Query(min_length=1, max_length=1024)):
    """Representative frames, visible ranges and confidence for every detected identity."""
    locks = store(request)
    inspection_path = locks._inspection_path(inspection_id)
    if len(inspection_id) != 32 or not inspection_path.is_file():
        raise DomainError("INSPECTION_NOT_FOUND", "Inspect the target first", 404)
    inspection = FaceInspection.model_validate_json(inspection_path.read_text("utf-8"))
    if inspection.target_sha256 != file_sha256(request.app.state.media.path(path)):
        raise DomainError("INSPECTION_TARGET_MISMATCH", "Inspection does not belong to this target video", 409)
    if inspection.tracking_method != "identity_lock_v1":
        raise DomainError("IDENTITY_INSPECTION_REQUIRED", "Re-run the target inspection with the identity lock enabled", 422)
    report = read_report(request.app.state.root, inspection_id, expected_sha256=inspection.identity_report_sha256)
    return {"inspection_id": inspection_id, "target_sha256": inspection.target_sha256,
            "tracking_policy_fingerprint": inspection.tracking_policy_fingerprint,
            "detection_threshold": inspection.detection_threshold, "decoded_frames": report["decoded_frames"],
            "truncated": report["truncated"], "track_limit_reached": report["track_limit_reached"],
            "scene_cut_frames": report["scene_cut_frames"], "event_count": len(report["events"]),
            "events_dropped": report["events_dropped"], "warnings": list(inspection.warnings),
            "tracks": public_tracks(report)}


@router.post("/locks", status_code=201)
def create_lock(value: LockCreate, request: Request):
    locks = store(request)
    return locks.public(locks.create(value), fresh=True)


@router.get("/locks")
def list_locks(request: Request, target_path: str | None = Query(default=None, max_length=1024)):
    locks = store(request)
    return [locks.public(item) for item in locks.list(target_path)]


@router.get("/locks/{lock_id}")
def read_lock(lock_id: str, request: Request, verify: bool = False):
    """``verify=true`` re-hashes the video and FaceSets from disk instead of trusting the size/mtime cache."""
    locks = store(request)
    return locks.public(locks.get(lock_id), fresh=verify)


@router.post("/locks/{lock_id}/edit")
def edit_lock(lock_id: str, value: LockEdit, request: Request):
    locks = store(request)
    return locks.public(locks.edit(lock_id, value), fresh=True)


@router.delete("/locks/{lock_id}", status_code=204)
def delete_lock(lock_id: str, request: Request) -> Response:
    store(request).delete(lock_id)
    return Response(status_code=204)
