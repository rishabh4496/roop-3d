# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Exact z-buffered rasterisation of per-vertex attributes (depth, normals, swap-mask weights).

``PhysicalBufferRenderer`` is the production GPU path for appearance. This small CPU rasteriser exists for the one thing it
does not give: pixel-aligned *vertex attributes* (the 3D swap-mask weights, region ids) together with the depth and normal
buffers they must be depth-tested against. It is deterministic, has no GPU dependency and is exact (per-pixel nearest
triangle, barycentric interpolation), so it is also the reference the compositor tests use.

Conventions follow the reenactment engine: ``pixels`` are (V, 2) canvas coordinates, ``depth`` is camera depth where a
SMALLER value is CLOSER (``FAR_CAMERA_M - z`` for a +z-toward-the-viewer mesh). Triangles are not culled by winding; the
z-test alone decides visibility, so a back face only shows where no nearer surface covers the pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class RasterBuffers:
    depth: np.ndarray            # (H, W) float32, +inf where no triangle covers the pixel
    face_id: np.ndarray          # (H, W) int32, -1 where uncovered
    attributes: np.ndarray       # (H, W, K) float32, barycentric interpolation of the vertex attributes (0 where uncovered)

    @property
    def coverage(self) -> np.ndarray:
        return (self.face_id >= 0).astype(np.float32)


def rasterize_vertex_attributes(pixels: np.ndarray, depth: np.ndarray, faces: np.ndarray, attributes: np.ndarray,
                                shape: tuple[int, int]) -> RasterBuffers:
    """Nearest-triangle z-buffer with barycentric attribute interpolation (pixel centres at +0.5)."""
    xy = np.asarray(pixels, np.float64)
    z = np.asarray(depth, np.float64).reshape(-1)
    tri = np.asarray(faces, np.int64)
    attr = np.asarray(attributes, np.float64)
    if attr.ndim == 1:
        attr = attr[:, None]
    height, width = int(shape[0]), int(shape[1])
    if (xy.ndim != 2 or xy.shape[1] != 2 or z.shape[0] != len(xy) or attr.shape[0] != len(xy) or tri.ndim != 2
            or tri.shape[1] != 3 or height <= 0 or width <= 0 or not np.isfinite(xy).all() or not np.isfinite(z).all()
            or not np.isfinite(attr).all()):
        raise DomainError("RASTER_INPUT_INVALID", "Pixels, depth, faces and attributes must be finite and consistent", 422)
    if tri.size and (tri.min() < 0 or tri.max() >= len(xy)):
        raise DomainError("RASTER_INPUT_INVALID", "Face indices are outside the vertex range", 422)
    zbuf = np.full((height, width), np.inf, np.float64)
    ids = np.full((height, width), -1, np.int32)
    out = np.zeros((height, width, attr.shape[1]), np.float64)
    a, b, c = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
    det = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    x0 = np.maximum(np.floor(np.minimum(np.minimum(a[:, 0], b[:, 0]), c[:, 0]) - 0.5), 0).astype(np.int64)
    x1 = np.minimum(np.ceil(np.maximum(np.maximum(a[:, 0], b[:, 0]), c[:, 0]) - 0.5), width - 1).astype(np.int64)
    y0 = np.maximum(np.floor(np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1]) - 0.5), 0).astype(np.int64)
    y1 = np.minimum(np.ceil(np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1]) - 0.5), height - 1).astype(np.int64)
    eps = 1e-9
    for f in np.nonzero((np.abs(det) > 1e-12) & (x1 >= x0) & (y1 >= y0))[0]:
        gx, gy = np.meshgrid(np.arange(x0[f], x1[f] + 1) + 0.5, np.arange(y0[f], y1[f] + 1) + 0.5)
        l0 = ((b[f, 1] - c[f, 1]) * (gx - c[f, 0]) + (c[f, 0] - b[f, 0]) * (gy - c[f, 1])) / det[f]
        l1 = ((c[f, 1] - a[f, 1]) * (gx - c[f, 0]) + (a[f, 0] - c[f, 0]) * (gy - c[f, 1])) / det[f]
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        if not inside.any():
            continue
        i0, i1, i2 = tri[f]
        pz = l0 * z[i0] + l1 * z[i1] + l2 * z[i2]
        window = (slice(y0[f], y1[f] + 1), slice(x0[f], x1[f] + 1))
        nearer = inside & (pz < zbuf[window])
        if not nearer.any():
            continue
        zbuf[window][nearer] = pz[nearer]
        ids[window][nearer] = f
        out[window][nearer] = (l0[nearer, None] * attr[i0] + l1[nearer, None] * attr[i1] + l2[nearer, None] * attr[i2])
    return RasterBuffers(zbuf.astype(np.float32), ids, out.astype(np.float32))


def orthographic_pixels(vertices: np.ndarray, *, scale: float, center_px: tuple[float, float],
                        rotation: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weak-perspective / orthographic projection of a +x right, +y down, +z toward-the-viewer mesh.

    Returns ((V, 2) canvas pixels, (V,) camera depth with smaller = closer).
    """
    v = np.asarray(vertices, np.float64)
    if rotation is not None:
        v = v @ np.asarray(rotation, np.float64).T
    pixels = np.column_stack((v[:, 0] * scale + center_px[0], v[:, 1] * scale + center_px[1]))
    return pixels, -v[:, 2]
