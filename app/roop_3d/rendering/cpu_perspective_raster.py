"""Bounded CPU perspective triangle rasterizer (Z-buffer, back-face rejection, perspective-correct UV sampling).

Extracted from the removed 2.5D pose-blend module. It requires true perspective camera-space vertices (+Z away from
the camera); the affine / orthographic mode is not available.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class PerspectiveRasterResult:
    image: np.ndarray
    coverage: np.ndarray
    faces: np.ndarray
    depth: np.ndarray
    normals: np.ndarray | None
    face_index: np.ndarray | None = None   # int32 HxW index into the rasterized triangle array; -1 outside the surface

def _rasterize(texture: np.ndarray, texture_coverage: np.ndarray, pixels: np.ndarray,
               camera_vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray,
               shape: tuple[int, int], *, camera_normals: np.ndarray | None,
               perspective: bool, sampling_device: str,
               roi: tuple[int, int, int, int] | None) -> PerspectiveRasterResult:
    if not perspective:
        raise DomainError("AFFINE_RASTER_REJECTED", "Only perspective Z-buffered rasterization is supported", 422)
    h, w = map(int, shape)
    atlas = np.asarray(texture)
    atlas_coverage = np.asarray(texture_coverage)
    projected = np.asarray(pixels, np.float64)
    camera = np.asarray(camera_vertices, np.float64)
    texcoords = np.asarray(uv, np.float64)
    triangles = np.asarray(faces, np.int64)
    if (h <= 0 or w <= 0 or atlas.ndim != 3 or atlas.shape[2] != 3
            or atlas_coverage.shape != atlas.shape[:2] or projected.shape != (len(camera), 2)
            or camera.ndim != 2 or camera.shape[1] != 3 or texcoords.shape != (len(camera), 2)
            or triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.isfinite(atlas_coverage).all() or not np.isfinite(projected).all()
            or not np.isfinite(camera).all() or not np.isfinite(texcoords).all()
            or (triangles.size and (triangles.min() < 0 or triangles.max() >= len(camera)))):
        raise DomainError("PERSPECTIVE_RASTER_INPUT_INVALID", "Perspective raster buffers have inconsistent shapes or values", 422)
    vertex_normals = None if camera_normals is None else np.asarray(camera_normals, np.float64)
    if vertex_normals is not None and (vertex_normals.shape != camera.shape or not np.isfinite(vertex_normals).all()):
        raise DomainError("PERSPECTIVE_RASTER_NORMALS_INVALID", "Camera normals must be finite Nx3 values", 422)

    if roi is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        if len(roi) != 4:
            raise DomainError("PERSPECTIVE_RASTER_ROI_INVALID", "ROI must be (x0, y0, x1, y1)", 422)
        rx0, ry0, rx1, ry1 = map(int, roi)
        x0, y0, x1, y1 = max(0, rx0), max(0, ry0), min(w, rx1), min(h, ry1)
    if x1 <= x0 or y1 <= y0:
        return PerspectiveRasterResult(
            np.zeros((h, w, 3), atlas.dtype), np.zeros((h, w), np.float32), np.empty((0, 3), np.int32),
            np.full((h, w), np.inf, np.float32),
            np.zeros((h, w, 3), np.float32) if vertex_normals is not None else None,
            np.full((h, w), -1, np.int32))

    rh, rw = y1 - y0, x1 - x0
    depth_roi = np.full((rh, rw), np.inf, np.float64)
    map_x = np.full((rh, rw), -1, np.float32)
    map_y = np.full((rh, rw), -1, np.float32)
    owner = np.full((rh, rw), -1, np.int32)
    normal_roi = np.zeros((rh, rw, 3), np.float32) if vertex_normals is not None else None
    origin = np.array([x0, y0], np.float64)
    for face_index, face in enumerate(triangles):
        points = projected[face] - origin
        vertices = camera[face]
        if not np.isfinite(points).all() or np.any(vertices[:, 2] <= 1e-4):
            continue
        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        if np.dot(normal, -vertices.mean(0)) <= 1e-12:
            continue
        bx0, bx1 = max(0, int(np.floor(points[:, 0].min()))), min(rw - 1, int(np.ceil(points[:, 0].max())))
        by0, by1 = max(0, int(np.floor(points[:, 1].min()))), min(rh - 1, int(np.ceil(points[:, 1].max())))
        if bx1 < bx0 or by1 < by0:
            continue
        a, b, c = points
        determinant = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(determinant) < 1e-8:
            continue
        ys, xs = np.mgrid[by0:by1 + 1, bx0:bx1 + 1].astype(np.float64)
        p = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / determinant
        q = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / determinant
        barycentric = np.stack((p, q, 1 - p - q), axis=-1)
        inside = np.all(barycentric >= -1e-5, axis=-1)
        weighted = barycentric / vertices[:, 2]
        denominator = weighted.sum(-1)
        z = 1 / np.maximum(denominator, 1e-12)
        region = depth_roi[by0:by1 + 1, bx0:bx1 + 1]
        valid = inside & (denominator > 1e-12) & (z < region)
        if not valid.any():
            continue
        sampled = (weighted @ texcoords[face]) / np.maximum(denominator[..., None], 1e-12)
        region[valid] = z[valid]
        owner[by0:by1 + 1, bx0:bx1 + 1][valid] = face_index
        map_x[by0:by1 + 1, bx0:bx1 + 1][valid] = sampled[..., 0][valid]
        map_y[by0:by1 + 1, bx0:bx1 + 1][valid] = sampled[..., 1][valid]
        if normal_roi is not None:
            interpolated = (weighted @ vertex_normals[face]) / np.maximum(denominator[..., None], 1e-12)
            length = np.linalg.norm(interpolated, axis=-1, keepdims=True)
            interpolated /= np.maximum(length, 1e-12)
            normal_region = normal_roi[by0:by1 + 1, bx0:bx1 + 1]
            normal_region[valid] = interpolated[valid].astype(np.float32)

    from roop_3d.reconstruction.uv_synthesizer import remap_points
    sampled_image = remap_points(atlas, map_x.reshape(-1), map_y.reshape(-1), cv2.INTER_LINEAR,
                                 device=sampling_device).reshape(rh, rw, 3)
    sampled_coverage = remap_points(atlas_coverage, map_x.reshape(-1), map_y.reshape(-1), cv2.INTER_LINEAR,
                                    device=sampling_device).reshape(rh, rw)
    sampled_coverage = np.clip(sampled_coverage, 0, 1).astype(np.float32)
    sampled_coverage *= np.isfinite(depth_roi)
    image_out = np.zeros((h, w, 3), dtype=atlas.dtype)
    coverage_out = np.zeros((h, w), np.float32)
    depth_out = np.full((h, w), np.inf, np.float32)
    normals_out = np.zeros((h, w, 3), np.float32) if normal_roi is not None else None
    image_out[y0:y1, x0:x1] = sampled_image
    coverage_out[y0:y1, x0:x1] = sampled_coverage
    depth_out[y0:y1, x0:x1] = depth_roi.astype(np.float32)
    if normals_out is not None:
        normals_out[y0:y1, x0:x1] = normal_roi
    owner_out = np.full((h, w), -1, np.int32)
    owner_out[y0:y1, x0:x1] = owner
    visible = np.unique(owner[owner >= 0])
    return PerspectiveRasterResult(image_out, coverage_out, triangles[visible].astype(np.int32),
                                  depth_out, normals_out, owner_out)


def rasterize(texture: np.ndarray, texture_coverage: np.ndarray, pixels: np.ndarray,
              camera_vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray,
              shape: tuple[int, int], *, perspective: bool = True,
              sampling_device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perspective z-buffer rasterization; UV coordinates are image-origin pixels."""
    result = _rasterize(texture, texture_coverage, pixels, camera_vertices, uv, faces, shape,
                        camera_normals=None, perspective=perspective, sampling_device=sampling_device, roi=None)
    return result.image, result.coverage, result.faces


def rasterize_layer(texture: np.ndarray, texture_coverage: np.ndarray, pixels: np.ndarray,
                    camera_vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray,
                    camera_normals: np.ndarray, shape: tuple[int, int], *,
                    roi: tuple[int, int, int, int], perspective: bool = True,
                    sampling_device: str = "cpu") -> PerspectiveRasterResult:
    """Rasterize a full-frame head layer and expose its depth and unit camera-space normals."""
    return _rasterize(texture, texture_coverage, pixels, camera_vertices, uv, faces, shape,
                      camera_normals=camera_normals, perspective=perspective,
                      sampling_device=sampling_device, roi=roi)
