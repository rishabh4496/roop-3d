"""Geometry-locked CodeFormer detail for a MakeHuman canonical UV albedo.

A full-head atlas is not an aligned portrait. Restore a real perspective-rendered
canonical frontal head instead, then depth-test and reproject only skin detail
onto supported UV texels. No target frame or target face crop enters this module.
Low-frequency albedo, facial features, unseen surface and UV-island gaps stay intact.
"""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.compositor.masks import BisenetFaceParser
from roop_3d.compositor.stage4 import anatomical_alpha, semantic_face_masks
from roop_3d.core.errors import DomainError
from roop_3d.enhancers.uv_detailer import TILE, UVTextureDetailer
from roop_3d.reconstruction.dense_correspondence import ReferenceSurfaceRasterizer
from roop_3d.reconstruction.makehuman_template import makehuman_template
from roop_3d.reconstruction.uv_synthesizer import remap_points

# The model's 512px FFHQ alignment, fitted through a perspective camera to the
# actual five MakeHuman landmark vertices (not an affine image transform).
_CANONICAL_LANDMARKS = np.array(((192.98138, 239.94708), (318.90277, 240.19360),
                               (256.63416, 314.01935), (201.26117, 371.41043),
                               (313.08905, 371.15118)), np.float64)


def _canonical_camera(mesh, landmark_indices):
    vertices = mesh.vertices.astype(np.float64) * (1.0, 1.0, -1.0)
    indices = np.asarray(landmark_indices, np.int32)
    intrinsics = np.array(((850.0, 0, 255.5), (0, 850.0, 255.5), (0, 0, 1)), np.float64)
    try:
        ok, rotation, translation = cv2.solvePnP(vertices[indices], _CANONICAL_LANDMARKS,
                                                intrinsics, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            raise DomainError("UV_RESTORATION_CAMERA_INVALID", "Canonical perspective head fitting failed", 422)
        rotation, translation = cv2.solvePnPRefineLM(vertices[indices], _CANONICAL_LANDMARKS,
                                                    intrinsics, None, rotation, translation)
        matrix = cv2.Rodrigues(rotation)[0]
    except cv2.error as error:
        raise DomainError("UV_RESTORATION_CAMERA_INVALID", f"Canonical camera fit failed: {error}", 422) from error
    camera_vertices = vertices @ matrix.T + translation.reshape(3)
    if not np.isfinite(camera_vertices).all() or np.any(camera_vertices[:, 2] <= 0.001):
        raise DomainError("UV_RESTORATION_CAMERA_INVALID", "Canonical head intersects the perspective camera near plane", 422)
    projected = camera_vertices[indices] @ intrinsics.T
    projected = projected[:, :2] / projected[:, 2:3]
    error = float(np.max(np.linalg.norm(projected - _CANONICAL_LANDMARKS, axis=1)))
    if error > 24:
        raise DomainError("UV_RESTORATION_CAMERA_INVALID", f"Canonical alignment residual is {error:.2f}px", 422)
    camera_normals = (mesh.normals.astype(np.float64) * (1.0, 1.0, -1.0)) @ matrix.T
    return camera_vertices, camera_normals, intrinsics


def _reproject_detail(texture, valid_mask, mesh, camera_vertices, camera_normals, intrinsics, surface, detail, skin, weight):
    """Bake image detail using UV barycentrics and the canonical camera Z-buffer."""
    size = texture.shape[0]
    uv_pixels = mesh.uvs.astype(np.float64) * (size - 1)
    result = texture.astype(np.float32)
    # Finite depth is sampled with the skin mask; missing silhouette pixels can
    # never become valid detail support through bilinear interpolation.
    depth = np.where(surface.valid, surface.depth, 0.0).astype(np.float32)
    packed = np.concatenate((detail * skin[..., None], skin[..., None], depth[..., None]), axis=2)
    changed = 0
    for triangle in mesh.faces:
        corners = uv_pixels[triangle]
        lo = np.maximum(np.ceil(corners.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.floor(corners.max(axis=0)).astype(int), size - 1)
        if np.any(lo > hi):
            continue
        basis = np.column_stack((corners[1] - corners[0], corners[2] - corners[0]))
        if abs(np.linalg.det(basis)) < 1e-9:
            continue
        yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        pair = (np.stack((xx, yy), axis=2) - corners[0]) @ np.linalg.inv(basis).T
        bary = np.concatenate((1.0 - pair.sum(axis=2, keepdims=True), pair), axis=2)
        inside = bary.min(axis=2) >= -1e-6
        if not inside.any():
            continue
        ys, xs = yy[inside], xx[inside]
        bary = bary[inside]
        positions = bary @ camera_vertices[triangle]
        normals = bary @ camera_normals[triangle]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        directions = -positions / np.linalg.norm(positions, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", normals, directions)
        points = positions @ intrinsics.T
        points = points[:, :2] / points[:, 2:3]
        samples = remap_points(packed, points[:, 0], points[:, 1])
        visible = ((facing > 0.25) & (samples[:, 3] > 0.05)
                   & (np.abs(samples[:, 4] - positions[:, 2]) < 0.0015)
                   & (points[:, 0] >= 1) & (points[:, 0] < TILE - 2)
                   & (points[:, 1] >= 1) & (points[:, 1] < TILE - 2)
                   & (valid_mask[ys, xs] > 0))
        if not visible.any():
            continue
        ys, xs, samples, facing = ys[visible], xs[visible], samples[visible], facing[visible]
        confidence = samples[:, 3] * np.clip((facing - 0.25) / 0.35, 0, 1) * valid_mask[ys, xs]
        delta = samples[:, :3] / np.maximum(samples[:, 3:4], 1e-6)
        result[ys, xs] = texture[ys, xs].astype(np.float32) + (weight * confidence)[:, None] * delta
        changed += len(xs)
    if not changed:
        raise DomainError("UV_RESTORATION_SUPPORT_EMPTY", "No depth-tested canonical skin texels support neural detail", 422)
    return np.clip(result, 0, 255).round().astype(np.uint8)


def restore_uv_texture(texture: np.ndarray, catalog, config, weight: float = 0.35, *,
                       mesh=None, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Return a same-size uint8 BGR MakeHuman UV atlas with skin-only neural detail.

    ``texture`` uses the MakeHuman template's repacked, top-left UV layout. An
    optional fitted ``mesh`` must retain that exact topology/UV layout. Supply
    ``valid_mask`` (HxW confidence in 0..1) to protect unobserved texels as well as
    geometrically unsupported surface. Weight 0 explicitly disables restoration;
    otherwise missing parser/enhancer weights and invalid tensors raise DomainError.

    The registered ``ultramax_uv`` artifact is the installed CodeFormer x/w graph.
    Its existing adapter supplies RGB NCHW [-1,1], the real scalar fidelity input,
    and validates the neural output. It is never invoked over whole-atlas tiles.
    """
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("restoration weight must be within 0..1")
    if (not isinstance(texture, np.ndarray) or texture.dtype != np.uint8 or texture.ndim != 3
            or texture.shape[2] != 3 or texture.shape[0] != texture.shape[1]
            or not 64 <= texture.shape[0] <= 4096):
        raise DomainError("UV_RESTORATION_INPUT_INVALID", "Expected a square 64..4096 uint8 BGR canonical UV atlas", 422)
    if valid_mask is None:
        validity = np.ones(texture.shape[:2], np.float32)
    else:
        validity = np.asarray(valid_mask, np.float32)
        if (validity.shape != texture.shape[:2] or not np.isfinite(validity).all()
                or np.any((validity < 0) | (validity > 1))):
            raise DomainError("UV_RESTORATION_INPUT_INVALID", "UV support must be a finite matching plane in 0..1", 422)
    if weight == 0:
        return texture.copy()
    template = makehuman_template()
    if mesh is None:
        mesh = template
    elif (mesh.vertices.shape != template.vertices.shape or not np.array_equal(mesh.faces, template.faces)
          or mesh.uvs is None or mesh.uvs.shape != template.uvs.shape
          or not np.allclose(mesh.uvs, template.uvs, atol=1e-7)):
        raise DomainError("UV_RESTORATION_LAYOUT_INVALID", "Restoration requires the repacked MakeHuman canonical UV topology", 422)
    camera_vertices, camera_normals, intrinsics = _canonical_camera(mesh, template.landmark_vertices["5pt"])
    attributes = np.column_stack((mesh.uvs, np.zeros(len(mesh.vertices), np.float32)))
    surface = ReferenceSurfaceRasterizer().rasterize(camera_vertices, mesh.faces, intrinsics, attributes, (TILE, TILE))
    if not surface.valid.any():
        raise DomainError("UV_RESTORATION_SUPPORT_EMPTY", "Canonical perspective head has no visible UV surface", 422)
    coordinates = surface.attributes[..., :2] * (texture.shape[0] - 1)
    canonical = cv2.remap(texture, coordinates[..., 0], coordinates[..., 1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
    support = cv2.remap(validity, coordinates[..., 0], coordinates[..., 1], cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT) * surface.valid
    canonical[support <= 0] = 127
    parser = BisenetFaceParser(catalog, config, normalization="imagenet")
    detailer = None
    try:
        probabilities = parser.predict(canonical)
        _, forbidden, skin, _ = semantic_face_masks(probabilities)
        allowed = (support > 0.5) & (skin > 0) & (forbidden == 0)
        skin = anatomical_alpha(skin * support, allowed, feather_px=1.5, erosion_px=0, dilation_px=0)
        detailer = UVTextureDetailer.from_catalog(catalog, config, model_name="ultramax_uv", fidelity_weight=1.0 - weight)
        if len(detailer._inputs) != 2:
            raise DomainError("ENHANCER_GRAPH_UNSUPPORTED", "Canonical skin restoration requires the registered CodeFormer image/fidelity graph", 503)
        restored = detailer._infer_tile(canonical)
    finally:
        parser.release()
        if detailer is not None:
            detailer.session.model.release()
    # The GAN cannot change face shape, eye/lip layout, diffuse albedo, or broad
    # illumination: only bounded high-frequency skin deltas return to the atlas.
    residual = restored.astype(np.float32) - canonical.astype(np.float32)
    detail = (cv2.GaussianBlur(residual, (0, 0), 0.45)
              - cv2.GaussianBlur(residual, (0, 0), 1.8))
    detail = np.clip(detail, -24.0, 24.0)
    return _reproject_detail(texture, validity, mesh, camera_vertices, camera_normals, intrinsics,
                             surface, detail, skin, float(weight))


__all__ = ["restore_uv_texture"]
