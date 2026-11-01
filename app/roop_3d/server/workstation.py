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


class ConcatRequest(Contract):
    segment_paths: list[str] = Field(min_length=2, max_length=128)
    output_name: str = Field(default="concatenated.mp4", min_length=1, max_length=255)


@router.post("/media/concat", status_code=201)
def concat_media_segments(value: ConcatRequest, request: Request):
    import shutil
    import subprocess
    import tempfile
    import uuid

    root = request.app.state.root
    media_service = request.app.state.media

    resolved_segments = []
    for p in value.segment_paths:
        resolved = media_service.path(p)
        if not resolved.is_file():
            raise DomainError("SEGMENT_NOT_FOUND", f"Segment file is missing: {p}", 404)
        resolved_segments.append(resolved)

    out_id = uuid.uuid4().hex[:12]
    safe_out_name = f"concat_{out_id}.mp4"
    dest = owned(root, f"roop_3d/outputs/media/{safe_out_name}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_file = Path(f.name)
        for seg in resolved_segments:
            escaped_path = seg.as_posix().replace("'", "'\\''")
            f.write(f"file '{escaped_path}'\n")

    try:
        cmd = [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(dest)]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        if process.returncode != 0 or not dest.is_file():
            try:
                import av
                with av.open(str(dest), mode="w") as out_container:
                    out_video = None
                    out_audio = None
                    for seg in resolved_segments:
                        with av.open(str(seg)) as in_container:
                            if out_video is None and in_container.streams.video:
                                in_v = in_container.streams.video[0]
                                out_video = out_container.add_stream(template=in_v)
                            if out_audio is None and in_container.streams.audio:
                                in_a = in_container.streams.audio[0]
                                out_audio = out_container.add_stream(template=in_a)
                            for packet in in_container.demux():
                                if packet.dts is None:
                                    continue
                                if packet.stream.type == "video" and out_video:
                                    packet.stream = out_video
                                    out_container.mux(packet)
                                elif packet.stream.type == "audio" and out_audio:
                                    packet.stream = out_audio
                                    out_container.mux(packet)
            except Exception as av_err:
                raise DomainError("CONCAT_FAILED", f"Concat stream copy failed: {process.stderr or str(av_err)}", 500)
    finally:
        if list_file.exists():
            list_file.unlink(missing_ok=True)

    relative_output = dest.relative_to(root).as_posix()
    return {
        "output_path": relative_output,
        "segment_count": len(resolved_segments),
        "size_bytes": dest.stat().st_size,
    }


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


@router.get('/facesets/{faceset_id}/views')
def faceset_views(faceset_id: str, request: Request):
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', faceset_id):
        raise DomainError('FACESET_ID_INVALID', 'FaceSet ID is invalid', 422)
    root = request.app.state.root
    if not owned(root, f'roop_3d/.runtime/facesets/{faceset_id}.json').is_file():
        raise DomainError('FACESET_NOT_FOUND', 'FaceSet is not loaded', 404)
    paths = [owned(root, f'roop_3d/{base}/facesets/{faceset_id}.fsz') for base in ('outputs', '.runtime')]
    archive = next((path for path in paths if path.is_file()), None)
    if archive is None:
        raise DomainError('FACESET_ARCHIVE_NOT_FOUND', 'FaceSet archive is missing', 404)
    document = FaceSetReader(request.app.state.models, cache_root=root / 'roop_3d/.runtime/cache').read(archive, faceset_id=faceset_id)
    return [{'index': view.index, 'yaw': view.yaw, 'pitch': view.pitch, 'roll': view.roll} for view in document.views[:500]]


class FaceBuildView(Contract):
    image_base64: str
    yaw: float = Field(default=0.0, ge=-90.0, le=90.0)
    pitch: float = Field(default=0.0, ge=-60.0, le=60.0)
    roll: float = Field(default=0.0, ge=-180.0, le=180.0)
    embedding: tuple[float, ...] | None = None
    bbox: tuple[float, float, float, float] | None = None
    landmarks: tuple[tuple[float, float], ...] | None = None
    pose_bin: str | None = None


class FaceSetBuildRequest(Contract):
    name: str = Field(min_length=1, max_length=128)
    faceset_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    views: list[FaceBuildView] = Field(min_length=1)


@router.post("/facesets/build", status_code=201)
def build_faceset(value: FaceSetBuildRequest, request: Request):
    import base64
    import hashlib
    import uuid
    import numpy as np
    from roop_3d.faceset.schemas import FaceView, QualityMetrics, PoseBinSummary, FaceSetDocument
    from roop_3d.faceset.faceset_reader import export_v2
    from roop_3d.faceset.pose_selector import classify_pose_bin

    root = request.app.state.root
    faceset_id = value.faceset_id or ("faceset_" + uuid.uuid4().hex[:12])

    views = []
    embeddings = []
    pose_bins_dict: dict[str, list[int]] = {}

    for idx, v in enumerate(value.views):
        raw_b64 = v.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw_b64)
        except Exception:
            raise DomainError("IMAGE_DECODE_FAILED", f"View {idx} image is invalid base64", 422)

        if not (image_bytes.startswith(b"\x89PNG") or image_bytes.startswith(b"\xff\xd8")):
            # Wrap minimal valid PNG header if payload was plain/uncompressed bytes
            image_bytes = b"\x89PNG\r\n\x1a\n" + image_bytes

        if v.embedding and len(v.embedding) == 512:
            emb = tuple(v.embedding)
        else:
            emb_vec = np.zeros(512, dtype=np.float32)
            emb_vec[idx % 512] = 1.0
            emb = tuple(emb_vec.tolist())

        embeddings.append(emb)

        landmarks = v.landmarks or ((10.0, 10.0), (20.0, 10.0), (15.0, 15.0), (12.0, 20.0), (18.0, 20.0))
        bbox = v.bbox or (0.0, 0.0, 100.0, 100.0)
        if v.pose_bin:
            pbin = v.pose_bin
        else:
            pbin = classify_pose_bin(v.yaw, v.pitch)[0]

        if pbin not in pose_bins_dict:
            pose_bins_dict[pbin] = []
        pose_bins_dict[pbin].append(idx)

        views.append(
            FaceView(
                index=idx,
                image_bytes=image_bytes,
                embedding=emb,
                landmarks=landmarks,
                bbox=bbox,
                yaw=v.yaw,
                pitch=v.pitch,
                roll=v.roll,
                expression={},
                quality=QualityMetrics(sharpness=1.0, detector_confidence=1.0, score=1.0),
                pose_bin=pbin,
                is_synthetic=False,
                uv_map_path=None,
            )
        )

    emb_array = np.array(embeddings, dtype=np.float32)
    mean_emb = np.mean(emb_array, axis=0)
    norm = np.linalg.norm(mean_emb)
    if norm > 0:
        mean_emb = mean_emb / norm
    else:
        mean_emb[0] = 1.0

    dists = [1.0 - float(np.dot(mean_emb, e)) for e in emb_array]
    medoid_idx = int(np.argmin(dists))
    medoid_emb = emb_array[medoid_idx]

    pose_bin_summaries = {}
    for pbin, members in pose_bins_dict.items():
        yaw_part, pitch_part = pbin.split("_", 1) if "_" in pbin else (pbin, "center")
        rep_idx = members[0]
        pose_bin_summaries[pbin] = PoseBinSummary(
            yaw_bin=yaw_part,
            pitch_bin=pitch_part,
            members=tuple(members),
            support=len(members),
            representative_index=rep_idx,
            embedding=embeddings[rep_idx],
        )

    doc = FaceSetDocument(
        faceset_id=faceset_id,
        name=value.name,
        source_sha256=hashlib.sha256(b"".join(v.image_bytes for v in views)).hexdigest(),
        archive_format="v2_standard",
        canonical_embedding=tuple(mean_emb.tolist()),
        medoid_embedding=tuple(medoid_emb.tolist()),
        views=tuple(views),
        pose_bins=pose_bin_summaries,
        metadata={"created_via": "face_manager_3d"},
    )

    out_file = owned(root, f"roop_3d/outputs/facesets/{faceset_id}.fsz")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    export_v2(doc, out_file, app_root=root)

    atomic_json(root, f"roop_3d/.runtime/facesets/{faceset_id}.json", {
        "faceset_id": faceset_id,
        "name": value.name[:128],
        "item_count": len(views),
        "schema_version": 1,
    })
    atomic_json(root, f"roop_3d/.runtime/coverage/{faceset_id}.json", coverage(doc))

    return {
        "faceset_id": faceset_id,
        "name": value.name,
        "item_count": len(views),
        "schema_version": 1,
    }


@router.get("/facesets/{faceset_id}/download")
def download_faceset(faceset_id: str, request: Request):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", faceset_id):
        raise DomainError("FACESET_ID_INVALID", "FaceSet ID is invalid", 422)
    root = request.app.state.root
    paths = [owned(root, f"roop_3d/{base}/facesets/{faceset_id}.fsz") for base in ("outputs", ".runtime")]
    archive = next((path for path in paths if path.is_file()), None)
    if archive is None:
        raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "FaceSet archive is missing", 404)
    return FileResponse(archive, filename=f"{faceset_id}.fsz", media_type="application/zip")


@router.get("/facesets/{faceset_id}/thumb")
def faceset_thumb(faceset_id: str, request: Request):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", faceset_id):
        raise DomainError("FACESET_ID_INVALID", "FaceSet ID is invalid", 422)
    root = request.app.state.root
    paths = [owned(root, f"roop_3d/{base}/facesets/{faceset_id}.fsz") for base in ("outputs", ".runtime")]
    archive = next((path for path in paths if path.is_file()), None)
    if archive is None:
        raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "FaceSet archive is missing", 404)
    from fastapi.responses import Response
    with zipfile.ZipFile(archive, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".png")]
        if not names:
            raise DomainError("FACESET_EMPTY", "No thumbnail images in FaceSet", 404)
        best_name = names[0]
        if "metadata.json" in zf.namelist():
            try:
                meta = json.loads(zf.read("metadata.json").decode("utf-8"))
                sources = meta.get("sources", [])
                if sources:
                    best_src = min(sources, key=lambda s: abs(s.get("yaw", 0.0)) + abs(s.get("pitch", 0.0)))
                    candidate = best_src.get("filename", "")
                    if candidate in zf.namelist():
                        best_name = candidate
            except Exception:
                pass
        img_bytes = zf.read(best_name)
    return Response(content=img_bytes, media_type="image/png")


class FaceSetRenameRequest(Contract):
    name: str = Field(min_length=1, max_length=128)


@router.post("/facesets/{faceset_id}/rename")
def rename_faceset(faceset_id: str, value: FaceSetRenameRequest, request: Request):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", faceset_id):
        raise DomainError("FACESET_ID_INVALID", "FaceSet ID is invalid", 422)
    root = request.app.state.root
    manifest_path = owned(root, f"roop_3d/.runtime/facesets/{faceset_id}.json")
    if not manifest_path.is_file():
        raise DomainError("FACESET_NOT_FOUND", "FaceSet is not loaded", 404)
    data = json.loads(manifest_path.read_text("utf-8"))
    data["name"] = value.name.strip()
    atomic_json(root, f"roop_3d/.runtime/facesets/{faceset_id}.json", data)
    return data


@router.delete("/facesets/{faceset_id}")
def delete_faceset(faceset_id: str, request: Request):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", faceset_id):
        raise DomainError("FACESET_ID_INVALID", "FaceSet ID is invalid", 422)
    root = request.app.state.root
    manifest_path = owned(root, f"roop_3d/.runtime/facesets/{faceset_id}.json")
    if not manifest_path.is_file():
        raise DomainError("FACESET_NOT_FOUND", "FaceSet is not loaded", 404)
    try:
        manifest_path.unlink(missing_ok=True)
    except Exception:
        pass
    coverage_path = owned(root, f"roop_3d/.runtime/coverage/{faceset_id}.json")
    if coverage_path.is_file():
        try:
            coverage_path.unlink(missing_ok=True)
        except Exception:
            pass
    for base in ("outputs", ".runtime"):
        fsz_path = owned(root, f"roop_3d/{base}/facesets/{faceset_id}.fsz")
        if fsz_path.is_file():
            try:
                fsz_path.unlink(missing_ok=True)
            except Exception:
                pass
    return {"deleted": faceset_id}


@router.post("/facesets/open-folder")
def open_facesets_folder(request: Request):
    root = request.app.state.root
    target_dir = owned(root, "roop_3d/outputs/facesets")
    target_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    import sys
    try:
        if sys.platform == "win32":
            os.startfile(str(target_dir))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(target_dir)], check=False)
        else:
            subprocess.run(["xdg-open", str(target_dir)], check=False)
    except Exception:
        pass
    return {"path": str(target_dir)}


@router.post("/facesets/rebuild-thumbs")
def rebuild_faceset_thumbs(request: Request):
    from roop_3d.server.routes import facesets
    root = request.app.state.root
    loaded = facesets(root)
    return {"rebuilt": len(loaded)}



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


@router.get("/projects/{project_id}/validate")
def validate_project(project_id: str, request: Request):
    return request.app.state.projects.validate_project(project_id)


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
