"""Image warping driven by thin-plate-spline control points (shared by frontalization and expression)."""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.tps_fallback import ThinPlateSplineWarper

# Interpolating (not smoothing) spline: control points must land exactly where they are told to.
EXACT = 1e-9


def border_anchors(width: int, height: int, per_side: int | None = None) -> np.ndarray:
    """Pinned points around the image border (corners included), one every ~8 px by default.

    A thin-plate spline has long-range influence, so a sparse border would sway between its anchors.
    """
    if per_side is None:
        per_side = int(np.clip(max(width, height) // 8, 6, 32))
    xs = np.linspace(0.0, width - 1.0, per_side + 1)
    ys = np.linspace(0.0, height - 1.0, per_side + 1)
    top = [(x, 0.0) for x in xs]
    bottom = [(x, height - 1.0) for x in xs]
    left = [(0.0, y) for y in ys[1:-1]]
    right = [(width - 1.0, y) for y in ys[1:-1]]
    return np.array(top + bottom + left + right, np.float64)


def _grid_map(warper: ThinPlateSplineWarper, width: int, height: int, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    map_x, map_y = np.empty((height, width), np.float32), np.empty((height, width), np.float32)
    for start in range(0, height, 32):
        yy, xx = np.mgrid[start:min(height, start + 32), 0:width]
        sampled = warper.transform_points(np.column_stack((xx.ravel() / scale[0], yy.ravel() / scale[1]))) * scale
        map_x[start:start + len(yy)] = sampled[:, 0].reshape(yy.shape)
        map_y[start:start + len(yy)] = sampled[:, 1].reshape(yy.shape)
    return map_x, map_y


def warp_points_to(image: np.ndarray, from_points: np.ndarray, to_points: np.ndarray, *,
                   regularization: float = EXACT) -> np.ndarray:
    """Return `image` deformed so the content at `from_points` ends up at `to_points`.

    Points are (N, 2) pixel coordinates (x, y); pairs that are equal stay fixed. Implemented as a backward
    TPS (output pixel -> source pixel) so every output pixel is defined; border replication fills the edge.
    """
    if image.ndim < 2:
        raise DomainError("TPS_IMAGE_INVALID", "Expected an image array", 422)
    height, width = image.shape[:2]
    source, target = np.asarray(from_points, np.float64), np.asarray(to_points, np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or not (np.isfinite(source).all() and np.isfinite(target).all()):
        raise DomainError("TPS_POINT_MISMATCH", "Warp needs matching finite (N, 2) point sets", 422)
    scale = np.array((max(width, 1), max(height, 1)), np.float64)
    warper = ThinPlateSplineWarper(target / scale, source / scale, regularization=regularization)
    map_x, map_y = _grid_map(warper, width, height, scale)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def transform_points_with(from_points: np.ndarray, to_points: np.ndarray, query: np.ndarray, *,
                          size: tuple[int, int], regularization: float = EXACT) -> np.ndarray:
    """Where do `query` pixels move when content at `from_points` is carried to `to_points`? (forward TPS)"""
    scale = np.array(size, np.float64)
    warper = ThinPlateSplineWarper(np.asarray(from_points, np.float64) / scale, np.asarray(to_points, np.float64) / scale,
                                   regularization=regularization)
    return warper.transform_points(np.asarray(query, np.float64) / scale) * scale
