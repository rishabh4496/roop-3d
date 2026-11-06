"""Automatic in-memory 3D reconstruction and exact sphere coverage API."""
from __future__ import annotations

from pathlib import Path
import re
import threading
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import Field, model_validator
from starlette.concurrency import run_in_threadpool

from roop_3d.audit.isolation import safe_path
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.reconstruction.faceset_3d import reconstruct_faceset

router = APIRouter(prefix="/api/v1")
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class Reconstruct3DRequest(Contract):
    faceset_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    faceset_path: str | None = Field(default=None, min_length=1, max_length=1024)
    source_image_paths: list[str] = Field(default_factory=list, max_length=32)
    live_pose: dict[str, float] | None = None

    @model_validator(mode="after")
    def source_required(self):
        if not self.faceset_id and not self.faceset_path and not self.source_image_paths:
            raise ValueError("Provide faceset_id, faceset_path, or source_image_paths")
        if self.faceset_id and (self.faceset_path or self.source_image_paths):
            raise ValueError("Choose one FaceSet source")
        if self.faceset_path and self.source_image_paths:
            raise ValueError("Choose one FaceSet source")
        if self.live_pose and not {"yaw", "pitch"}.issubset(self.live_pose):
            raise ValueError("live_pose requires yaw and pitch")
        return self


def _read_archive(request: Request, faceset_id: str | None = None, faceset_path: str | None = None):
    root = request.app.state.root
    if faceset_id:
        if not _ID.fullmatch(faceset_id):
            raise DomainError("FACESET_ID_INVALID", "FaceSet id is invalid", 422)
        if not safe_path(root, f"roop_3d/.runtime/facesets/{faceset_id}.json").is_file():
            raise DomainError("FACESET_NOT_FOUND", "FaceSet is not loaded", 404)
        choices = [safe_path(root, f"roop_3d/{base}/facesets/{faceset_id}.fsz")
                   for base in ("outputs", ".runtime")]
        archive = next((path for path in choices if path.is_file()), None)
    elif faceset_path:
        relative = faceset_path.replace("\\", "/")
        if not relative.startswith(("facesets/", "roop_3d/outputs/facesets/", "roop_3d/.runtime/facesets/")):
            raise DomainError("MEDIA_PATH_DENIED", "FaceSet path must be in the local FaceSets directory", 422)
        archive = safe_path(root, relative)
        if archive.suffix.casefold() != ".fsz" or not archive.is_file():
            raise DomainError("FACESET_NOT_FOUND", "FaceSet archive is unavailable", 404)
        faceset_id = archive.stem
        if not _ID.fullmatch(faceset_id):
            raise DomainError("FACESET_ID_INVALID", "FaceSet archive name is invalid", 422)
    else:
        return None
    if archive is None:
        raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "FaceSet archive is missing", 404)
    return FaceSetReader(request.app.state.models, cache_root=root / "roop_3d/.runtime/cache").read(
        archive, faceset_id=faceset_id)


def _from_images(request: Request, paths: list[str]):
    service = request.app.state.face_extract
    faceset_id = None
    for index, value in enumerate(paths):
        relative = value.replace("\\", "/")
        path_suffix = Path(relative).suffix.casefold()
        if path_suffix not in _IMAGE_SUFFIXES:
            raise DomainError("SOURCE_IMAGE_INVALID", "A source image is missing or has an unsupported format", 422)
        path, is_image = service._media_file(relative)
        if not is_image:
            raise DomainError("SOURCE_IMAGE_INVALID", "Source path must identify a supported image", 422)
        from roop_3d.server.face_extract import ExtractRequest
        extract = ExtractRequest(target_media_path=relative, frame_index=0, face_index=0,
                                 destination="source_gallery" if faceset_id is None else "active_faceset",
                                 faceset_name=path.stem[:128] if faceset_id is None else None,
                                 faceset_id=faceset_id)
        result = service.extract(extract)
        faceset_id = result.get("faceset", {}).get("faceset_id") or result.get("active_source", {}).get("faceset_id")
        if not faceset_id:
            raise DomainError("FACESET_CREATE_FAILED", "Source image did not create a local FaceSet", 500)
    return _read_archive(request, faceset_id=faceset_id)


def _reconstruct(request: Request, value: Reconstruct3DRequest):
    if value.source_image_paths:
        document = _from_images(request, value.source_image_paths)
    else:
        document = _read_archive(request, value.faceset_id, value.faceset_path)
    state: dict[str, Any] = getattr(request.app.state, "faceset_3d_state", None)
    if state is None:
        state = request.app.state.faceset_3d_state = {"lock": threading.RLock(), "values": {}, "latest": None}
    with state["lock"]:
        cached = state["values"].get(document.faceset_id)
        if cached and cached.get("source_sha256") == document.source_sha256:
            result = cached["payload"]
        else:
            result = reconstruct_faceset(document, live_pose=value.live_pose)
            result["source_sha256"] = document.source_sha256
            state["values"][document.faceset_id] = {"source_sha256": document.source_sha256, "payload": result}
        if value.live_pose:
            result["live_pose"] = {"yaw": float(value.live_pose["yaw"]), "pitch": float(value.live_pose["pitch"]),
                                   "roll": float(value.live_pose.get("roll", 0.0))}
        state["latest"] = document.faceset_id
        return result


@router.post("/faceset/reconstruct-3d")
async def reconstruct_3d(value: Reconstruct3DRequest, request: Request):
    return await run_in_threadpool(_reconstruct, request, value)


@router.get("/faceset/3d-state")
def faceset_3d_state(request: Request, faceset_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$"),
                     yaw: float | None = Query(default=None, ge=-180, le=180),
                     pitch: float | None = Query(default=None, ge=-90, le=90),
                     roll: float | None = Query(default=None, ge=-180, le=180)):
    state = getattr(request.app.state, "faceset_3d_state", None)
    if state is None:
        raise DomainError("RECONSTRUCTION_NOT_FOUND", "No FaceSet has been reconstructed in this process", 404)
    with state["lock"]:
        key = faceset_id or state.get("latest")
        value = state["values"].get(key) if key else None
        if value is None:
            raise DomainError("RECONSTRUCTION_NOT_FOUND", "FaceSet reconstruction is not cached", 404)
        payload = value["payload"]
        if yaw is not None and pitch is not None:
            payload = {**payload, "live_pose": {"yaw": yaw, "pitch": pitch, "roll": roll or 0.0}}
        return payload
