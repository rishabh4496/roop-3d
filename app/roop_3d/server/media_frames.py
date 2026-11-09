"""Timeline scrubbing endpoints: cheap media info, one frame (optionally with its faces), and a filmstrip.

    GET /api/v1/media/info?path=...                       width, height, fps, total_frames, duration_sec
    GET /api/v1/media/frame?path=...&frame_index=N        a JPEG; its metadata (and faces) in the X-Frame-Meta header
        &detect_faces=true                                 faces: native-pixel bbox, 5-point landmarks, pose, confidence
        &format=json                                       the same, as JSON with the JPEG as a data URL
    GET /api/v1/media/filmstrip?path=...&count=16         evenly spaced thumbnails in one request

All three go through `MediaService.frames` (`media.frame_service.VideoFrameService`): persistent decoders and caches,
so scrubbing never reopens the file. Nothing here writes to disk.
"""
from __future__ import annotations

import base64
import json
from typing import Annotated

from fastapi import APIRouter, Query, Request, Response

router = APIRouter(prefix="/api/v1")

# Fields the scrubbing prompt listed that the engine cannot honestly provide, reported instead of invented.
UNSUPPORTED = {
    "embedding_norm": "ArcFace embeddings are L2-normalised before they leave the recognizer, so their norm is always 1.0; "
                      "locking a face computes and stores its real embedding.",
}


def _meta(result, info, size, faces) -> dict:
    meta = {"frame_index": result.frame_index, "requested_index": result.requested_index, "timestamp_sec": round(result.timestamp_sec, 4),
            "width": int(result.image.shape[1]), "height": int(result.image.shape[0]),
            "image_width": size[0], "image_height": size[1], "total_frames": info.total_frames, "fps": info.fps,
            "cache": result.cache, "decode_ms": round(result.decode_ms, 2)}
    if faces is not None:
        meta["faces"] = faces
        meta["unsupported"] = sorted(UNSUPPORTED)
    return meta


@router.get("/media/info")
def media_info(request: Request, path: Annotated[str, Query(min_length=1, max_length=1024)]) -> dict:
    return request.app.state.media.frames.info(path).as_dict()


@router.get("/media/frame", response_class=Response,
            responses={200: {"content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}}}})
def media_frame(request: Request, path: Annotated[str, Query(min_length=1, max_length=1024)],
                frame_index: Annotated[int | None, Query(ge=0, le=100_000_000)] = None,
                timestamp: Annotated[float | None, Query(ge=0, le=86400)] = None,
                detect_faces: bool = False,
                threshold: Annotated[float, Query(ge=0.1, le=0.99)] = 0.5,
                max_size: Annotated[int, Query(ge=64, le=4096)] = 1280,
                quality: Annotated[int, Query(ge=30, le=95)] = 82,
                format: Annotated[str, Query(pattern="^(jpeg|json)$")] = "jpeg") -> Response:
    frames = request.app.state.media.frames
    result = frames.read(path, frame_index=frame_index, timestamp=timestamp)
    data, out_w, out_h = frames.jpeg(path, result, max_size=max_size, quality=quality)
    faces = None
    if detect_faces:
        # JPEG responses carry the metadata in a header, so they stay light; format=json returns the full per-face analysis.
        from roop_3d.server.target_lock import annotate
        analyzed = frames.analyzed_faces(path, result, threshold)
        faces = [dict(face) for face in frames.faces(path, result, threshold, detail="full" if format == "json" else "basic")]
        if analyzed:
            annotate(request, path, result.frame_index, analyzed, faces, result.image)
    meta = _meta(result, frames.info(path), (out_w, out_h), faces)
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    if format == "json":
        return Response(json.dumps({**meta, "image": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}),
                        media_type="application/json", headers=headers)
    # ensure_ascii keeps the header latin-1 safe; separators drop the padding.
    headers["X-Frame-Meta"] = json.dumps(meta, separators=(",", ":"), ensure_ascii=True)
    headers["Access-Control-Expose-Headers"] = "X-Frame-Meta"
    return Response(data, media_type="image/jpeg", headers=headers)


@router.get("/media/filmstrip")
def media_filmstrip(request: Request, path: Annotated[str, Query(min_length=1, max_length=1024)],
                    count: Annotated[int, Query(ge=1, le=64)] = 16,
                    thumb_height: Annotated[int, Query(ge=24, le=360)] = 90) -> dict:
    return request.app.state.media.frames.filmstrip(path, count, thumb_height)
