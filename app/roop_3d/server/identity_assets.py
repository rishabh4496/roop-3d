"""Stage 7 workstation API for the identity panel: per-view FaceSet thumbnails and the Stage 3 identity asset.

Additive and local-only. Every route reads the same FaceSet archive the existing ``/facesets/{id}/views`` and
``/coverage`` routes read, and the identity asset uses the exact ``IdentityAssetStore`` / default settings that the
Stage 4 render session uses (``engine/volumetric/session.py``), so an asset built from the panel is the one a
``pose_3d`` / ``reenactment_3d`` render reuses, and vice versa.

* ``GET  /facesets/{id}/views/{index}/image``  bounded JPEG of one stored view (no embeddings).
* ``GET  /facesets/{id}/identity-asset``       coverage readiness + the cached asset manifest, or ``not_built``.
                                               Never builds: reading the panel costs no reconstruction.
* ``POST /facesets/{id}/identity-asset``       explicit user action: build (or reuse) the asset on the CPU.
* ``GET  /facesets/{id}/identity-asset/mesh``  the cached asset as a viewer mesh (glTF frame, observed atlas).

All three accept ``?template=canonical|makehuman`` (default ``canonical``, the template ``pose_3d`` / ``reenactment_3d`` render
sessions reuse). ``makehuman`` is the open CC0 head+neck template (5-point landmarks; it must be built first, see
``roop_3d.reconstruction.makehuman_build``); such assets are for inspection / export and are refused by the expression rig.
"""
from __future__ import annotations

import base64
import re
import threading
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from roop_3d.core.errors import DomainError
from roop_3d.core.storage import owned
from roop_3d.faceset.faceset_reader import FaceSetReader

router = APIRouter(prefix="/api/v1")
_FACESET_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
TemplateName = Literal["canonical", "makehuman"]
_BUILD_LOCKS: dict[str, threading.Lock] = {}
_BUILD_LOCKS_GUARD = threading.Lock()
MESH_VERTEX_LIMIT = 100_000
MESH_FACE_LIMIT = 200_000


def _archive(request: Request, faceset_id: str):
    if not _FACESET_ID.fullmatch(faceset_id):
        raise DomainError("FACESET_ID_INVALID", "FaceSet ID is invalid", 422)
    root = request.app.state.root
    if not owned(root, f"roop_3d/.runtime/facesets/{faceset_id}.json").is_file():
        raise DomainError("FACESET_NOT_FOUND", "FaceSet is not loaded", 404)
    paths = [owned(root, f"roop_3d/{base}/facesets/{faceset_id}.fsz") for base in ("outputs", ".runtime")]
    archive = next((path for path in paths if path.is_file()), None)
    if archive is None:
        raise DomainError("FACESET_ARCHIVE_NOT_FOUND", "FaceSet archive is missing", 404)
    return archive


def load_document(request: Request, faceset_id: str):
    """The validated FaceSetDocument, read exactly as the existing views/coverage routes read it."""
    archive = _archive(request, faceset_id)
    root = request.app.state.root
    return FaceSetReader(request.app.state.models, cache_root=root / "roop_3d/.runtime/cache").read(archive, faceset_id=faceset_id)


def _settings(template: str = "canonical"):
    from roop_3d.reconstruction.identity_asset import IdentityAssetSettings
    return IdentityAssetSettings(template=template)


def _store():
    # Same store and root as the render session (IdentityAssetStore() with no arguments).
    from roop_3d.reconstruction.identity_asset import IdentityAssetStore
    return IdentityAssetStore()


def _coverage_summary(report) -> dict[str, Any]:
    return {
        "readiness": report.readiness,
        "view_count": report.view_count,
        "usable_count": report.usable_count,
        "yaw_range_deg": list(report.yaw_range_deg) if report.yaw_range_deg else None,
        "yaw_span_deg": round(float(report.yaw_span_deg), 2),
        "pitch_span_deg": round(float(report.pitch_span_deg), 2),
        "yaw_coverage": {name: list(members) for name, members in report.yaw_coverage.items()},
        "pitch_coverage": {name: list(members) for name, members in report.pitch_coverage.items()},
        "missing": list(report.missing),
        "requests": list(report.requests)[:8],
        "duplicate_groups": [list(group) for group in report.duplicate_groups],
        "capabilities": dict(report.capabilities),
    }


def _manifest_summary(asset) -> dict[str, Any]:
    m = asset.manifest
    return {
        "build_key": m.build_key,
        "created_at": m.created_at,
        "algorithm_version": m.algorithm_version,
        "reconstruction_level": m.reconstruction_level,
        "landmark_layout": m.landmark_layout,
        "claims": dict(m.claims),
        "missing_information": list(m.missing_information),
        "limitations": list(m.limitations),
        "statistics": dict(m.statistics),
        "fit": {key: value for key, value in m.fit.items() if isinstance(value, (int, float, str, bool)) or value is None},
        "flame": dict(m.flame),
        "sources_used": sum(1 for source in m.sources if source.used),
        "sources_total": len(m.sources),
        "vertex_count": int(len(asset.vertices)),
        "face_count": int(len(asset.faces)),
        "texture_size": [int(asset.texture.shape[1]), int(asset.texture.shape[0])],
    }


def _status(document, store, template: str = "canonical") -> dict[str, Any]:
    from roop_3d.reconstruction.faceset_quality import analyze_faceset
    from roop_3d.reconstruction.identity_asset import compute_build_key, flame_status
    report = analyze_faceset(document)
    key = compute_build_key(document, _settings(template))
    asset = store.load(document.faceset_id, key)
    return {
        "faceset_id": document.faceset_id,
        "name": document.name,
        "state": "built" if asset is not None else "not_built",
        "template": template,
        "build_key": key,
        "coverage": _coverage_summary(report),
        "asset": _manifest_summary(asset) if asset is not None else None,
        "flame": flame_status(),
        "note": "Built only on explicit request. pose_3d / reenactment_3d renders reuse this exact asset.",
    }


@router.get("/facesets/{faceset_id}/views/{index}/image")
def faceset_view_image(faceset_id: str, index: int, request: Request, size: int = Query(default=160, ge=32, le=512)):
    document = load_document(request, faceset_id)
    view = next((item for item in document.views if item.index == index), None)
    if view is None:
        raise DomainError("FACESET_VIEW_NOT_FOUND", "FaceSet view does not exist", 404)
    image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DomainError("FACESET_VIEW_UNREADABLE", "FaceSet view image cannot be decoded", 422)
    scale = size / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise DomainError("FACESET_VIEW_ENCODE_FAILED", "FaceSet view image cannot be encoded", 500)
    return Response(content=encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=60"})


@router.get("/facesets/{faceset_id}/identity-asset")
async def identity_asset_status(faceset_id: str, request: Request, template: TemplateName = Query(default="canonical")):
    document = await run_in_threadpool(load_document, request, faceset_id)
    return await run_in_threadpool(_status, document, _store(), template)


def _build(document, template: str = "canonical") -> dict[str, Any]:
    with _BUILD_LOCKS_GUARD:
        lock = _BUILD_LOCKS.setdefault(document.faceset_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise DomainError("IDENTITY_ASSET_BUILD_IN_PROGRESS", "This FaceSet's identity asset is already being built", 409)
    try:
        store = _store()
        asset, reused = store.get_or_build(document, _settings(template))
        value = _status(document, store, template)
        value["reused"] = reused
        if value["asset"] is None:  # the store refused to persist or re-read it: report, never pretend
            raise DomainError("IDENTITY_ASSET_NOT_PERSISTED", "The identity asset was built but could not be re-read from the store", 500)
        return value
    finally:
        lock.release()


@router.post("/facesets/{faceset_id}/identity-asset")
async def build_identity_asset(faceset_id: str, request: Request, template: TemplateName = Query(default="canonical")):
    document = await run_in_threadpool(load_document, request, faceset_id)
    return await run_in_threadpool(_build, document, template)


def _mesh(document, template: str = "canonical") -> dict[str, Any]:
    from roop_3d.reconstruction.identity_asset import compute_build_key, to_gltf_frame
    from roop_3d.reconstruction.uv_synthesizer import ConfidenceState
    key = compute_build_key(document, _settings(template))
    asset = _store().load(document.faceset_id, key)
    if asset is None:
        raise DomainError("IDENTITY_ASSET_NOT_BUILT", "Build the identity asset first; it is never built implicitly", 404)
    if len(asset.vertices) > MESH_VERTEX_LIMIT or len(asset.faces) > MESH_FACE_LIMIT:
        raise DomainError("IDENTITY_ASSET_TOO_LARGE", "The identity mesh exceeds the viewer bounds", 413)
    mesh = to_gltf_frame(asset.mesh())
    texture = asset.texture.copy()
    # Unseen texels are shown as neutral grey, never as invented skin.
    texture[asset.texel_state == int(ConfidenceState.UNSEEN)] = (96, 96, 96)
    ok, png = cv2.imencode(".png", texture)
    if not ok:
        raise DomainError("TEXTURE_ENCODE_FAILED", "Failed to encode the identity atlas", 500)
    return {
        "method": "identity_asset",
        "coordinate_system": "opengl",
        "vertices": np.round(mesh.vertices.astype(np.float64), 6).tolist(),
        "faces": mesh.faces.astype(np.int64).tolist(),
        "uvs": np.round(mesh.uvs.astype(np.float64), 6).tolist() if mesh.uvs is not None else None,
        "texture": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii"),
        "provenance": f"Stage 3 identity asset {asset.manifest.build_key[:12]} · {asset.manifest.reconstruction_level}"
                      f" · {asset.manifest.landmark_layout} landmarks · unseen texels grey",
    }


@router.get("/facesets/{faceset_id}/identity-asset/mesh")
async def identity_asset_mesh(faceset_id: str, request: Request, template: TemplateName = Query(default="canonical")):
    document = await run_in_threadpool(load_document, request, faceset_id)
    return await run_in_threadpool(_mesh, document, template)
