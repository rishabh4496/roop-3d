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

``GET .../identity-asset/mesh`` additionally takes, for ``template=makehuman`` assets only:

* ``topology=makehuman_cc0|flame2023_open``  re-express the mesh on another registered topology through the barycentric bridge
  (vertices, normals and the UV albedo atlas). ``flame2023_open`` is CC-BY-4.0; its attribution is returned with the mesh.
* ``jaw_width``, ``cheekbone_prominence``, ``chin_projection``, ``nose_bridge``, ``brow_arch``, ``face_fullness``  semantic shape
  sliders (physical units, see ``/morphable-topologies``) applied through the output topology's own shape basis.
* ``albedo_prior=true``  fill texels no photo observed with the data-free surface albedo prior (``reconstruction.albedo_prior``),
  through the existing symmetry + harmonic completion. Opt-in; default is the grey "unseen" display.

``GET /morphable-topologies`` lists the registered topologies (licence, attribution, availability) and the slider definitions.
"""
from __future__ import annotations

import base64
import re
import threading
from dataclasses import dataclass
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
_CONTROLLERS: dict[tuple, Any] = {}
_CONTROLLERS_GUARD = threading.Lock()
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


def _semantic_controller(registry, topology_id: str):
    """Cached per topology: building one costs a bridge registration the first time (seconds for FLAME), milliseconds after."""
    from roop_3d.reconstruction.semantic_shape import SemanticShapeController
    key = (topology_id, registry.fingerprint(registry.default_id), registry.fingerprint(topology_id))
    with _CONTROLLERS_GUARD:
        if key not in _CONTROLLERS:
            _CONTROLLERS.clear()
            _CONTROLLERS[key] = SemanticShapeController.for_topology(registry, topology_id)
        return _CONTROLLERS[key]


def _topology_view(asset, template: str, topology: str | None, sliders: dict[str, float]):
    """Mesh / atlas the viewer gets, plus the ``topology`` and ``semantic`` evidence blocks (``None`` when untouched)."""
    from roop_3d.reconstruction.geometry_types import Mesh3D
    from roop_3d.reconstruction.topology import default_registry
    asked = {k: v for k, v in sliders.items() if v != 0.0}
    if template != "makehuman":
        if topology is not None or asked:
            raise DomainError("TOPOLOGY_REQUIRES_MAKEHUMAN_TEMPLATE",
                              "topology and semantic sliders apply to template=makehuman assets only", 422)
        return asset.mesh(), asset.texture, asset.texel_state, None, None
    registry = default_registry()
    topology_id = topology or registry.default_id
    spec = registry.spec(topology_id)
    base = registry.prepared(registry.default_id)
    if len(asset.vertices) != base.vertex_count:
        raise DomainError("TOPOLOGY_ASSET_MISMATCH", "The cached asset was built on a different MakeHuman template version; rebuild it", 409)
    vertices, faces, uvs = np.asarray(asset.vertices, np.float64), base.faces, base.uvs
    texture, texel_state, bridge_report = asset.texture, asset.texel_state, None
    is_default = topology_id == registry.default_id
    if not is_default:
        bridge = registry.bridge(registry.default_id, topology_id)
        vertices = bridge.convert_positions(vertices, "target")
        faces, uvs = bridge.target.faces, bridge.target.uvs
        moved = bridge.convert_texture(asset.texture, "target", out_size=asset.texture.shape[0], texel_state=asset.texel_state)
        texture, texel_state, bridge_report = moved["texture"], moved["texel_state"], bridge.report()
    semantic = None
    if asked:
        vertices, result = _semantic_controller(registry, topology_id).apply(vertices, sliders)
        semantic = result.public()
    mesh = asset.mesh() if is_default and not asked else Mesh3D(vertices, faces, uvs=uvs)
    block = {**spec.public(), "default": is_default, "vertex_count": int(len(vertices)),
             "uv_provenance": base.uv_provenance if is_default else registry.prepared(topology_id).uv_provenance,
             "bridge": bridge_report}
    return mesh, texture, texel_state, block, semantic


def _complete_with_prior(mesh, texture: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Hole-free atlas for display: symmetry + harmonic completion whose far field is the data-free surface albedo prior."""
    from roop_3d.reconstruction.albedo_prior import prior_field
    from roop_3d.reconstruction.mesh_bridge import rasterize_uv
    from roop_3d.reconstruction.uv_completion import complete_texture
    from roop_3d.reconstruction.uv_synthesizer import ConfidenceState
    if mesh.uvs is None:
        raise DomainError("ALBEDO_PRIOR_NEEDS_UVS", "The mesh has no UV layout to place the prior on", 422)
    size = texture.shape[0]
    tri, bary = rasterize_uv(np.asarray(mesh.uvs, np.float64), mesh.faces, size)
    footprint = tri >= 0
    positions = np.zeros((size, size, 3))
    positions[footprint] = np.einsum("nc,ncd->nd", bary[footprint], np.asarray(mesh.vertices, np.float64)[mesh.faces[tri[footprint]]])
    field, prior = prior_field(texture[..., :3], state, footprint, positions)
    known = footprint & (state >= int(ConfidenceState.INTERPOLATED))
    done = complete_texture(texture[..., :3], known, footprint, positions, mirror_x=float(np.asarray(mesh.vertices)[:, 0].mean()), far_field=field)
    total = max(int(footprint.sum()), 1)
    out = np.array(texture, copy=True)
    out[..., :3] = np.where(footprint[..., None], done.texture[..., :3], out[..., :3])
    report = {"method": "data_free_surface_prior + symmetry/harmonic completion", "third_party_data": False, "colour_space": "CIELAB",
              "observed_ratio": round(done.stats["observed_ratio"], 4), "symmetry_ratio": round(done.stats["symmetry_ratio"], 4),
              "harmonic_ratio": round(done.stats["harmonic_ratio"], 4), "skin_tone": prior.tone.public(), "fit": prior.report,
              "limits": "Completed texels are priors, not observations: a smooth skin tone with no lip, brow, eyelid or identity detail. "
                        "Texels within a few pixels of completed areas are softened to hide the seam."}
    return out, report


@dataclass
class _MeshView:
    """Everything both mesh routes (JSON and GLB) serve, prepared once: the geometry in the identity-asset (y down) frame, the
    display atlas (BGR, unseen texels grey or prior-filled) and the evidence blocks."""
    asset: Any
    source_mesh: Any
    texture: np.ndarray
    provenance: str
    topology_block: dict[str, Any] | None
    semantic: dict[str, Any] | None
    prior_block: dict[str, Any] | None


def _prepare_mesh(document, template: str, topology: str | None, sliders: dict[str, float], albedo_prior: bool) -> _MeshView:
    from roop_3d.reconstruction.identity_asset import compute_build_key
    from roop_3d.reconstruction.uv_synthesizer import ConfidenceState
    key = compute_build_key(document, _settings(template))
    asset = _store().load(document.faceset_id, key)
    if asset is None:
        raise DomainError("IDENTITY_ASSET_NOT_BUILT", "Build the identity asset first; it is never built implicitly", 404)
    source_mesh, source_texture, source_state, topology_block, semantic = _topology_view(asset, template, topology, sliders)
    if len(source_mesh.vertices) > MESH_VERTEX_LIMIT or len(source_mesh.faces) > MESH_FACE_LIMIT:
        raise DomainError("IDENTITY_ASSET_TOO_LARGE", "The identity mesh exceeds the viewer bounds", 413)
    texture = np.array(source_texture, copy=True)
    prior_block = None
    if albedo_prior:
        texture, prior_block = _complete_with_prior(source_mesh, texture, np.asarray(source_state))
    else:
        # Unseen texels are shown as neutral grey, never as invented skin.
        texture[source_state == int(ConfidenceState.UNSEEN)] = (96, 96, 96)
    provenance = (f"Stage 3 identity asset {asset.manifest.build_key[:12]} · {asset.manifest.reconstruction_level}"
                  f" · {asset.manifest.landmark_layout} landmarks · "
                  + ("unseen texels filled by the data-free albedo prior" if albedo_prior else "unseen texels grey"))
    if topology_block is not None:
        if not topology_block["default"]:
            provenance += f" · topology {topology_block['id']} ({topology_block['license_id']}; UV {topology_block['uv_provenance']})"
        provenance += " · semantic sliders applied" if semantic else ""
    return _MeshView(asset, source_mesh, texture, provenance, topology_block, semantic, prior_block)


def _mesh(document, template: str = "canonical", topology: str | None = None, sliders: dict[str, float] | None = None,
          albedo_prior: bool = False) -> dict[str, Any]:
    from roop_3d.reconstruction.identity_asset import to_gltf_frame
    view = _prepare_mesh(document, template, topology, sliders or {}, albedo_prior)
    mesh = to_gltf_frame(view.source_mesh)
    ok, png = cv2.imencode(".png", view.texture)
    if not ok:
        raise DomainError("TEXTURE_ENCODE_FAILED", "Failed to encode the identity atlas", 500)
    out = {
        "method": "identity_asset",
        "coordinate_system": "opengl",
        "vertices": np.round(mesh.vertices.astype(np.float64), 6).tolist(),
        "faces": mesh.faces.astype(np.int64).tolist(),
        "uvs": np.round(mesh.uvs.astype(np.float64), 6).tolist() if mesh.uvs is not None else None,
        "texture": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii"),
        "provenance": view.provenance,
    }
    if view.topology_block is not None:
        out["topology"] = view.topology_block
        out["semantic"] = view.semantic
    if view.prior_block is not None:
        out["albedo_prior"] = view.prior_block
    return out


_GLB_SERIALIZER = None


def _serializer():
    global _GLB_SERIALIZER
    if _GLB_SERIALIZER is None:
        from roop_3d.io.glb_export import GLBBinarySerializer
        _GLB_SERIALIZER = GLBBinarySerializer()
    return _GLB_SERIALIZER


def _shape_keys(topology_id: str, sliders: dict[str, float]):
    """The six semantic sliders as glTF morph targets (delta of +1 slider unit, mesh frame) and the weights that reproduce
    ``sliders``. Exact: the controller is linear in the slider values, apart from one whole-move limit scale applied to the weights."""
    from roop_3d.io.glb_export import MorphTarget
    from roop_3d.reconstruction.semantic_shape import SLIDER_KEYS, SLIDERS
    from roop_3d.reconstruction.topology import default_registry
    controller = _semantic_controller(default_registry(), topology_id)
    targets = []
    for slider in SLIDERS:
        unit = controller.solve({slider.key: 1.0})
        targets.append(MorphTarget(slider.key, controller.displacement(unit.delta_beta / unit.limit_scale)))
    asked = {k: v for k, v in sliders.items() if v != 0.0}
    result = controller.solve(asked) if asked else None             # also validates ranges
    scale = result.limit_scale if result else 1.0
    return targets, [float(sliders.get(key, 0.0)) * scale for key in SLIDER_KEYS], result.public() if result else None


def _normal_map(view: _MeshView, template: str, topology_block: dict[str, Any] | None):
    """Tangent-space normal map from the asset's UV displacement (metres), or ``None`` with the reason it is absent."""
    from roop_3d.io.glb_export import metres_per_uv_unit, normal_map_from_height
    height = getattr(view.asset, "displacement_uv_m", None)
    if height is None or np.asarray(height).ndim != 2 or np.asarray(height).size == 0:
        return None, "asset carries no UV displacement"
    if topology_block is not None and not topology_block["default"]:
        return None, "displacement is stored in the default topology's UV layout"
    height = np.asarray(height, np.float32)
    if float(np.ptp(height)) <= 1e-9:
        return None, "UV displacement is flat (no measured fine relief)"
    mesh = view.source_mesh
    if mesh.uvs is None:
        return None, "mesh has no UVs"
    per_texel = metres_per_uv_unit(mesh.vertices, mesh.faces, mesh.uvs) / height.shape[0]
    return normal_map_from_height(height, per_texel), "baked from the asset's UV displacement"


def _glb(document, template: str = "canonical", topology: str | None = None, sliders: dict[str, float] | None = None,
         albedo_prior: bool = False, shape_keys: bool = False, normal_map: bool = True) -> bytes:
    """Binary glTF of the identity mesh: albedo atlas, optional normal map, optional shape keys. Same inputs as ``_mesh``."""
    sliders = sliders or {}
    if shape_keys and template != "makehuman":
        raise DomainError("SHAPE_KEYS_REQUIRE_MAKEHUMAN_TEMPLATE", "Shape keys are the semantic sliders and need template=makehuman", 422)
    # With shape keys the base mesh is the un-slid one and the sliders travel as morph weights, so a viewer / Blender user can
    # keep editing them; without, the sliders are baked into the vertices exactly as the JSON route does.
    view = _prepare_mesh(document, template, topology, {} if shape_keys else sliders, albedo_prior)
    morphs, weights, semantic = [], None, view.semantic
    if shape_keys:
        topology_id = view.topology_block["id"]
        morphs, weights, semantic = _shape_keys(topology_id, sliders)
    normal_image, normal_note = _normal_map(view, template, view.topology_block) if normal_map else (None, "not requested")
    asset, mesh = view.asset, view.source_mesh
    attributes = {}
    confidence, state = getattr(asset, "vertex_confidence", None), getattr(asset, "vertex_state", None)
    if confidence is not None and state is not None and len(confidence) == len(mesh.vertices) == len(state):
        attributes = {"_CONFIDENCE": confidence, "_STATE": np.asarray(state, np.float32)}
    manifest = asset.manifest
    extras = {"roop3d_identity_asset": {
        "build_key": manifest.build_key, "reconstruction_level": manifest.reconstruction_level,
        "landmark_layout": manifest.landmark_layout, "provenance": view.provenance, "normal_map": normal_note,
        "shape_keys": [t.name for t in morphs], "semantic": semantic, "topology": view.topology_block,
        "albedo_prior": view.prior_block,
        "frame": "converted from the y-down identity-asset frame to glTF (y up, right-handed); metres",
    }}
    return _serializer().serialize(mesh, frame="mesh", albedo=view.texture, normal_map=normal_image, morph_targets=morphs,
                                   morph_weights=weights, vertex_attributes=attributes, extras=extras, name="roop3d_head")


@router.get("/facesets/{faceset_id}/identity-asset/mesh")
async def identity_asset_mesh(
    faceset_id: str, request: Request, template: TemplateName = Query(default="canonical"),
    topology: str | None = Query(default=None, max_length=64),
    jaw_width: float = Query(default=0.0), cheekbone_prominence: float = Query(default=0.0),
    chin_projection: float = Query(default=0.0), nose_bridge: float = Query(default=0.0),
    brow_arch: float = Query(default=0.0), face_fullness: float = Query(default=0.0),
    albedo_prior: bool = Query(default=False),
):
    sliders = {"jaw_width": jaw_width, "cheekbone_prominence": cheekbone_prominence, "chin_projection": chin_projection,
               "nose_bridge": nose_bridge, "brow_arch": brow_arch, "face_fullness": face_fullness}
    document = await run_in_threadpool(load_document, request, faceset_id)
    return await run_in_threadpool(_mesh, document, template, topology, sliders, albedo_prior)


@router.get("/facesets/{faceset_id}/identity-asset/mesh.glb")
async def identity_asset_mesh_glb(
    faceset_id: str, request: Request, template: TemplateName = Query(default="canonical"),
    topology: str | None = Query(default=None, max_length=64),
    jaw_width: float = Query(default=0.0), cheekbone_prominence: float = Query(default=0.0),
    chin_projection: float = Query(default=0.0), nose_bridge: float = Query(default=0.0),
    brow_arch: float = Query(default=0.0), face_fullness: float = Query(default=0.0),
    albedo_prior: bool = Query(default=False),
    shape_keys: bool = Query(default=False, description="template=makehuman only: emit the six sliders as morph targets"),
    normal_map: bool = Query(default=True, description="attach a baked normal map when the asset has measured relief"),
    download: bool = Query(default=False),
):
    """The ``/mesh`` payload as a binary glTF 2.0 stream: load it with three.js ``GLTFLoader`` (or import it into Blender)."""
    sliders = {"jaw_width": jaw_width, "cheekbone_prominence": cheekbone_prominence, "chin_projection": chin_projection,
               "nose_bridge": nose_bridge, "brow_arch": brow_arch, "face_fullness": face_fullness}
    document = await run_in_threadpool(load_document, request, faceset_id)
    data = await run_in_threadpool(_glb, document, template, topology, sliders, albedo_prior, shape_keys, normal_map)
    disposition = "attachment" if download else "inline"
    return Response(content=data, media_type="model/gltf-binary",
                    headers={"Content-Disposition": f'{disposition}; filename="{faceset_id}_head.glb"', "Cache-Control": "private, no-cache"})


@router.get("/morphable-topologies")
async def morphable_topologies():
    """Registered head topologies (licence, attribution, availability) and the semantic slider definitions."""
    from roop_3d.reconstruction.semantic_shape import slider_definitions
    from roop_3d.reconstruction.topology import default_registry
    registry = default_registry()
    return {"default": registry.default_id, "topologies": await run_in_threadpool(registry.describe),
            "semantic_sliders": slider_definitions()}
