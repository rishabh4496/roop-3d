"""Local workstation assets, projects, and honest reconstruction inspection."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import uuid
import zipfile

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from roop_3d.audit.isolation import sha256_file
from roop_3d.bridges.provider_protocol import safe_filename
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.core.storage import atomic_json, exclusive, owned
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.jobs.project_store import ProjectDraft, ProjectManifest
from roop_3d.media.service import VIDEO_SUFFIXES

router = APIRouter(prefix="/api/v1")
CHUNK_BYTES = 1024 * 1024
MAX_ASSET_BYTES = 8 * 1024**3


class UploadRequest(Contract):
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0, le=MAX_ASSET_BYTES, strict=True)


def upload_path(root: Path, upload_id: str, suffix: str) -> Path:
    if len(upload_id) != 32 or any(c not in "0123456789abcdef" for c in upload_id):
        raise DomainError("UPLOAD_ID_INVALID", "Upload identifier is invalid", 422)
    return owned(root, f"roop_3d/.runtime/uploads/{upload_id}{suffix}")


@router.post("/assets/uploads", status_code=201)
def create_upload(value: UploadRequest, request: Request):
    try:
        safe_filename(value.filename)
    except ValueError:
        raise DomainError("UPLOAD_NAME_INVALID", "Use a portable media or FaceSet filename", 422) from None
    suffix = Path(value.filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES | {".fsz"} or suffix == ".fsz" and value.size_bytes > 256 * 1024**2:
        raise DomainError("UPLOAD_FORMAT_INVALID", "Unsupported format or oversized FaceSet archive", 422)
    root = request.app.state.root
    upload_id = uuid.uuid4().hex
    relative = upload_path(root, upload_id, ".json").relative_to(root).as_posix()
    atomic_json(root, relative, value.model_dump())
    return {"upload_id": upload_id, "chunk_bytes": min(CHUNK_BYTES, request.app.state.settings.max_body_bytes), "offset": 0}


def append_upload(root: Path, upload_id: str, offset: int, data: bytes):
    receipt = upload_path(root, upload_id, ".json")
    with exclusive(root, receipt.relative_to(root).as_posix() + ".lock"):
        if not receipt.is_file():
            raise DomainError("UPLOAD_NOT_FOUND", "Upload session is missing", 404)
        info = UploadRequest.model_validate_json(receipt.read_text("utf-8"))
        path = upload_path(root, upload_id, ".part")
        current = path.stat().st_size if path.exists() else 0
        if offset != current or not data or len(data) > CHUNK_BYTES or current + len(data) > info.size_bytes:
            raise DomainError("UPLOAD_OFFSET_INVALID", "Upload offset or chunk size is invalid", 409)
        if path.exists() and path.stat().st_nlink != 1:
            raise DomainError("UPLOAD_LINK_DENIED", "Linked upload files are forbidden", 409)
        with path.open("ab") as stream:
            stream.write(data)
        return {"offset": current + len(data)}


@router.put("/assets/uploads/{upload_id}")
async def upload_chunk(upload_id: str, request: Request, offset: int = Query(ge=0)):
    return await run_in_threadpool(append_upload, request.app.state.root, upload_id, offset, await request.body())


def coverage(document):
    return {"faceset_id": document.faceset_id, "name": document.name, "source_sha256": document.source_sha256,
            "views": [{"yaw": view.yaw, "pitch": view.pitch, "roll": view.roll, "pose_bin": view.pose_bin} for view in document.views],
            "bins": {key: value.support for key, value in document.pose_bins.items()}, "pose_state": "ESTIMATED_2_5D"}


def finish_upload(request: Request, upload_id: str):
    state, root = request.app.state, request.app.state.root
    receipt = upload_path(root, upload_id, ".json")
    with exclusive(root, receipt.relative_to(root).as_posix() + ".lock"):
        if not receipt.is_file():
            raise DomainError("UPLOAD_NOT_FOUND", "Upload session is missing", 404)
        info = UploadRequest.model_validate_json(receipt.read_text("utf-8"))
        path = upload_path(root, upload_id, ".part")
        if not path.is_file() or path.stat().st_size != info.size_bytes:
            raise DomainError("UPLOAD_INCOMPLETE", "Upload bytes are incomplete", 409)
        suffix = Path(info.filename).suffix.lower()
        digest = sha256_file(path)[1]
        if suffix == ".fsz":
            document = FaceSetReader(state.models, cache_root=root / "roop_3d/.runtime/cache").read(path, faceset_id=upload_id)
            relative = f"roop_3d/outputs/facesets/{upload_id}.fsz"
            destination = owned(root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination)
            atomic_json(root, f"roop_3d/.runtime/facesets/{upload_id}.json", {"faceset_id": upload_id,
                "name": Path(info.filename).stem[:128], "item_count": len(document.views), "schema_version": 1})
            atomic_json(root, f"roop_3d/.runtime/coverage/{upload_id}.json", coverage(document))
            result = {"kind": "faceset", "faceset_id": upload_id, "sha256": digest}
        else:
            from roop_3d.media.probe import probe
            probe(path)
            relative = f"roop_3d/.runtime/media/{upload_id}{suffix}"
            destination = owned(root, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, destination)
            result = {"kind": "video", "path": relative, "sha256": digest}
        receipt.unlink()
        return result


@router.post("/assets/uploads/{upload_id}/complete")
def complete_upload(upload_id: str, request: Request):
    try:
        return finish_upload(request, upload_id)
    except (ValueError, OSError, zipfile.BadZipFile):
        raise DomainError("ASSET_INVALID", "Uploaded file is not a valid supported asset", 422) from None


@router.delete("/assets/uploads/{upload_id}")
def discard_upload(upload_id: str, request: Request):
    root = request.app.state.root
    relative = upload_path(root, upload_id, ".json").relative_to(root).as_posix()
    with exclusive(root, relative + ".lock"):
        for suffix in (".part", ".json"):
            upload_path(root, upload_id, suffix).unlink(missing_ok=True)
    return {"discarded": True}


@router.get("/media/content")
def media_content(request: Request, path: str = Query(max_length=1024)):
    return FileResponse(request.app.state.media.path(path), headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@router.get("/media/keyframes")
def keyframes(request: Request, path: str = Query(max_length=1024)):
    from roop_3d.media.probe import open_media
    timestamps = []
    with open_media(request.app.state.media.path(path)) as source:
        for index, packet in enumerate(source.demux(source.streams.video[0])):
            if packet.is_keyframe and packet.pts is not None:
                timestamps.append(float(packet.pts * packet.time_base))
            if len(timestamps) >= 256 or index >= 50000:
                break
    return {"timestamps": timestamps, "bounded_sample": True}


@router.get("/media/inspections/{inspection_id}")
def saved_inspection(inspection_id: str, request: Request, path: str = Query(max_length=1024)):
    import re
    from roop_3d.media.schema import FaceInspection
    if not re.fullmatch(r"[a-f0-9]{32}", inspection_id):
        raise DomainError("INSPECTION_ID_INVALID", "Inspection identifier is invalid", 422)
    receipt = owned(request.app.state.root, f"roop_3d/.runtime/inspections/{inspection_id}.json")
    if not receipt.is_file():
        raise DomainError("INSPECTION_NOT_FOUND", "Saved inspection is missing", 404)
    value = FaceInspection.model_validate_json(receipt.read_text("utf-8"))
    if value.inspection_id != inspection_id or value.target_sha256 != sha256_file(request.app.state.media.path(path))[1]:
        raise DomainError("INSPECTION_TARGET_MISMATCH", "Inspection does not match the current target", 409)
    return value


@router.get("/facesets/{faceset_id}/coverage")
def faceset_coverage(faceset_id: str, request: Request):
    from roop_3d.jobs.schema import FaceSetSummary
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", faceset_id):
        raise DomainError("FACESET_ID_INVALID", "FaceSet ID is invalid", 422)
    root = request.app.state.root
    summary = owned(root, f"roop_3d/.runtime/facesets/{faceset_id}.json")
    if not summary.is_file():
        raise DomainError("FACESET_NOT_FOUND", "FaceSet is not loaded", 404)
    FaceSetSummary.model_validate_json(summary.read_text("utf-8"))
    paths = [owned(root, f"roop_3d/{base}/facesets/{faceset_id}.fsz") for base in ("outputs", ".runtime")]
    archive = next((path for path in paths if path.is_file()), None)
    if archive is None:
        raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "FaceSet archive is missing", 404)
    cached = owned(root, f"roop_3d/.runtime/coverage/{faceset_id}.json")
    if cached.is_file():
        value = json.loads(cached.read_text("utf-8"))
        if value.get("source_sha256") == sha256_file(archive)[1]:
            return value
    return coverage(FaceSetReader(request.app.state.models, cache_root=root / "roop_3d/.runtime/cache").read(archive, faceset_id=faceset_id))


@router.get("/projects")
def list_projects(request: Request):
    return request.app.state.projects.list()


@router.post("/projects", response_model=ProjectManifest, status_code=201)
def save_project(value: ProjectDraft, request: Request):
    from roop_3d.server.routes import validate_job
    validate_job(request, value.request)
    return request.app.state.projects.save(value, request.app.state.settings)


@router.post("/projects/import", response_model=ProjectManifest, status_code=201)
def import_project(value: ProjectManifest, request: Request):
    store = request.app.state.projects
    relative = store.relative(value.project_id)
    with exclusive(store.root, relative + ".lock"):
        path = owned(store.root, relative)
        if path.exists():
            raise DomainError("PROJECT_EXISTS", "Project identifier already exists; open the saved project", 409)
        atomic_json(store.root, relative, value.model_dump(mode="json"))
        try:
            return store.load(value.project_id)
        except Exception:
            path.unlink(missing_ok=True)
            raise


@router.get("/projects/{project_id}", response_model=ProjectManifest)
def open_project(project_id: str, request: Request):
    return request.app.state.projects.load(project_id)


@router.get("/projects/{project_id}/download")
def download_project(project_id: str, request: Request):
    request.app.state.projects.load(project_id)
    path = owned(request.app.state.root, request.app.state.projects.relative(project_id))
    return FileResponse(path, filename=f"{project_id}.r3dp", media_type="application/json")


@router.post("/projects/{project_id}/resume")
def resume_project(project_id: str, request: Request):
    record = request.app.state.projects.resume(project_id, request.app.state.settings)
    if request.app.state.worker is not None:
        request.app.state.worker.wake()
    return record


@router.get("/reconstructions")
def reconstructions(request: Request):
    base = owned(request.app.state.root, "roop_3d/.runtime/reconstructions")
    if not base.is_dir():
        return []
    return [{"reconstruction_id": path.stem} for path in sorted(base.glob("*.json")) if len(path.stem) == 32]


@router.get("/reconstructions/{reconstruction_id}")
def reconstruction(reconstruction_id: str, request: Request, target_path: str = Query(max_length=1024)):
    root = request.app.state.root
    request.app.state.projects.relative(reconstruction_id)
    receipt = owned(root, f"roop_3d/.runtime/reconstructions/{reconstruction_id}.json")
    if not receipt.is_file() or receipt.stat().st_size > 8192:
        raise DomainError("RECONSTRUCTION_NOT_FOUND", "A bounded reconstruction receipt is required", 404)
    try:
        data = json.loads(receipt.read_text("utf-8"))
        if data["method"] not in {"flame", "tps_2_5d"} or data["coordinate_system"] not in {"opencv", "opengl"}:
            raise ValueError()
        if sha256_file(request.app.state.media.path(target_path))[1] != data["source_sha256"]:
            raise DomainError("RECONSTRUCTION_SOURCE_CHANGED", "Reconstruction does not match this target", 409)
        def artifact(path: str, digest: str, limit: int):
            if not path.startswith("roop_3d/.runtime/reconstructions/"):
                raise ValueError()
            resolved = owned(root, path)
            if resolved.stat().st_size > limit or sha256_file(resolved)[1] != digest:
                raise ValueError()
            return resolved
        from roop_3d.reconstruction.mesh_io import load_mesh_obj
        mesh = load_mesh_obj(artifact(data["mesh_path"], data["mesh_sha256"], 8 * 1024**2))
        if len(mesh.vertices) > 100000 or len(mesh.faces) > 200000:
            raise ValueError()
        texture = None
        if data.get("uv_path"):
            path = artifact(data["uv_path"], data["uv_sha256"], 4 * 1024**2)
            from roop_3d.bridges.comfyui_bridge import BridgeConfig, decode_image
            raw = path.read_bytes()
            mime = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
            decode_image(raw, mime, BridgeConfig(max_image_bytes=4 * 1024**2))
            texture = f"data:{mime};base64," + base64.b64encode(raw).decode()
        return {"method": data["method"], "coordinate_system": data["coordinate_system"],
                "provenance": "USER_SUPPLIED_ARTIFACT", "vertices": mesh.vertices.tolist(), "faces": mesh.faces.tolist(),
                "uvs": mesh.uvs.tolist() if mesh.uvs is not None else None,
                "face_uvs": mesh.face_uvs.tolist() if mesh.face_uvs is not None else None, "texture": texture}
    except (OSError, ValueError, KeyError, TypeError):
        raise DomainError("RECONSTRUCTION_INVALID", "Reconstruction artifact or receipt failed validation", 422) from None
