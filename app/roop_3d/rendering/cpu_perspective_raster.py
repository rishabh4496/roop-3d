"""Bounded CPU perspective triangle rasterizer (Z-buffer, back-face rejection, perspective-correct UV sampling).

Extracted from the removed 2.5D pose-blend module. It requires true perspective camera-space vertices (+Z away from
the camera); the affine / orthographic mode is not available.
"""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


def rasterize(texture: np.ndarray, texture_coverage: np.ndarray, pixels: np.ndarray,
              camera_vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray,
              shape: tuple[int, int], *, perspective: bool = True,
              sampling_device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded triangle rasterizer. UVs use image origin (top-left), in pixels."""
    if not perspective:
        raise DomainError("AFFINE_RASTER_REJECTED", "Only perspective Z-buffered rasterization is supported", 422)
    h, w = shape
    depth = np.full((h, w), np.inf, np.float64)
    map_x = np.full((h, w), -1, np.float32)
    map_y = np.full((h, w), -1, np.float32)
    owner = np.full((h, w), -1, np.int32)
    for face_index, face in enumerate(faces):
        points = pixels[face].astype(np.float64)
        vertices = camera_vertices[face].astype(np.float64)
        if not np.isfinite(points).all() or not np.isfinite(vertices).all() or np.any(vertices[:, 2] <= 1e-4):
            continue
        if perspective:
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            if np.dot(normal, -vertices.mean(0)) <= 1e-12:
                continue
        x0, x1 = max(0, int(np.floor(points[:, 0].min()))), min(w - 1, int(np.ceil(points[:, 0].max())))
        y0, y1 = max(0, int(np.floor(points[:, 1].min()))), min(h - 1, int(np.ceil(points[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = points
        determinant = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(determinant) < 1e-8:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
        p = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / determinant
        q = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / determinant
        barycentric = np.stack((p, q, 1 - p - q), axis=-1)
        inside = np.all(barycentric >= -1e-5, axis=-1)
        weighted = barycentric / vertices[:, 2] if perspective else barycentric
        denominator = weighted.sum(-1)
        z = 1 / np.maximum(denominator, 1e-12) if perspective else np.ones_like(p)
        region = depth[y0:y1 + 1, x0:x1 + 1]
        valid = inside & (denominator > 1e-12) & (z < region)
        if not valid.any():
            continue
        sampled = (weighted @ uv[face]) / np.maximum(denominator[..., None], 1e-12)
        region[valid] = z[valid]
        owner[y0:y1 + 1, x0:x1 + 1][valid] = face_index
        map_x[y0:y1 + 1, x0:x1 + 1][valid] = sampled[..., 0][valid]
        map_y[y0:y1 + 1, x0:x1 + 1][valid] = sampled[..., 1][valid]
    from roop_3d.reconstruction.uv_synthesizer import remap_points
    image = remap_points(texture, map_x.reshape(-1), map_y.reshape(-1), cv2.INTER_LINEAR,
                         device=sampling_device).reshape(h, w, *texture.shape[2:])
    coverage = remap_points(texture_coverage, map_x.reshape(-1), map_y.reshape(-1), cv2.INTER_LINEAR,
                            device=sampling_device).reshape(h, w)
    coverage *= np.isfinite(depth)
    visible = np.unique(owner[owner >= 0])
    return image, np.clip(coverage, 0, 1).astype(np.float32), faces[visible]
