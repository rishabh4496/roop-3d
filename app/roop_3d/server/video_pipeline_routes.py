"""Stage 17 routes: one-shot upload / import and the verified job download.

Path mapping (the stage prompt names three routes; here is what each really is):

``POST /api/v1/media/upload``      one-call upload of a target video or image. A thin alias of the existing, validated chunked
                                   protocol (``/assets/uploads``): same name / container / size checks, same destination.
``POST /api/v1/facesets/import``   one-call import of a ``.fsz`` archive (``FaceSetReader`` validates it; it lands in
                                   ``roop_3d/outputs/facesets`` with its catalog + coverage receipts), same as the chunked path.
``GET  /api/v1/jobs/{id}/download`` the finished MP4 of a succeeded job, integrity-verified against the job's recorded SHA-256,
                                   served with HTTP Range support so a ``<video>`` element can seek.

Both uploads read the WHOLE body in one request, so they are bounded by ``ROOP3D_MAX_BODY_BYTES`` (2 MiB by default, 64 MiB at most:
the middleware rejects larger bodies with 413 before this code runs, and that cap is deliberately not loosened here). Larger files use
the chunked ``/assets/uploads`` protocol, which is what the web UI does.
"""
from __future__ import annotations

import os
import zipfile

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.storage import owned
from roop_3d.server.workstation import (CHUNK_BYTES, UploadRequest, append_upload, create_upload, discard_upload, finish_upload)

router = APIRouter(prefix="/api/v1")
_VIDEO_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".webm": "video/webm"}


def _store(request: Request, filename: str, body: bytes, *, expect: str) -> dict:
    """Run the chunked protocol (create -> append -> complete) over an in-memory body; nothing is left behind on failure."""
    if not body:
        raise DomainError("UPLOAD_EMPTY", "The request body is empty", 422)
    created = create_upload(UploadRequest(filename=filename, size_bytes=len(body)), request)
    upload_id = created["upload_id"]
    try:
        for offset in range(0, len(body), CHUNK_BYTES):
            append_upload(request.app.state.root, upload_id, offset, body[offset:offset + CHUNK_BYTES])
        result = finish_upload(request, upload_id)
    except (ValueError, OSError, zipfile.BadZipFile):
        discard_upload(upload_id, request)
        raise DomainError("ASSET_INVALID", "Uploaded file is not a valid supported asset", 422) from None
    except DomainError:
        discard_upload(upload_id, request)
        raise
    kind = result["kind"]
    if (expect == "faceset") != (kind == "faceset"):
        # Wrong endpoint for this file type: undo what finish_upload published so a mistake leaves no asset behind.
        _unpublish(request, result)
        raise DomainError("UPLOAD_KIND_MISMATCH", f"This endpoint takes {'a .fsz FaceSet' if expect == 'faceset' else 'a video or image'}; "
                          f"the file is {'a FaceSet' if kind == 'faceset' else kind}", 422)
    return result


def _unpublish(request: Request, result: dict) -> None:
    root = request.app.state.root
    if result["kind"] == "faceset":
        fid = result["faceset_id"]
        for relative in (f"roop_3d/outputs/facesets/{fid}.fsz", f"roop_3d/.runtime/facesets/{fid}.json",
                         f"roop_3d/.runtime/coverage/{fid}.json"):
            owned(root, relative).unlink(missing_ok=True)
    else:
        owned(root, result["path"]).unlink(missing_ok=True)


@router.post("/media/upload", status_code=201)
async def upload_media(request: Request, filename: str = Query(min_length=1, max_length=255)):
    """``POST /api/v1/media/upload?filename=clip.mp4`` with the file as the raw body -> ``{kind, path, sha256}``."""
    body = await request.body()
    return await run_in_threadpool(_store, request, filename, body, expect="media")


@router.post("/facesets/import", status_code=201)
async def import_faceset(request: Request, filename: str = Query(min_length=1, max_length=255)):
    """``POST /api/v1/facesets/import?filename=me.fsz`` with the archive as the raw body -> ``{kind, faceset_id, sha256}``."""
    body = await request.body()
    return await run_in_threadpool(_store, request, filename, body, expect="faceset")


@router.get("/jobs/{job_id}/download")
def download_job_video(job_id: str, request: Request):
    """The finished video of a succeeded job, verified against the SHA-256 recorded when it was published."""
    record = request.app.state.jobs.get(job_id)
    if record.state.value != "succeeded" or record.output is None:
        raise DomainError("JOB_OUTPUT_NOT_READY", "A verified completed render is required", 409)
    if not record.output.baked_video:
        raise DomainError("JOB_VIDEO_NOT_RENDERED", "This render exported passes only", 409)
    path = owned(request.app.state.root, record.output.output_path)
    if path.is_dir():
        raise DomainError("JOB_OUTPUT_IS_SEQUENCE", "This render is an image sequence; fetch the 'sequence' artifact", 409)
    if not path.is_file() or sha256_file(path)[1] != record.output.output_sha256:
        raise DomainError("JOB_OUTPUT_CHANGED", "Published output is missing or its integrity changed", 409)
    return FileResponse(path, filename=f"roop3d_{job_id[:8]}{path.suffix.lower()}", media_type=_VIDEO_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
