"""Replacement coverage of the dense swap: rasterised facial surface, landmark hull, extent-proportional feather.

The recipe is Roop Ultimate's matte (``procmgr_masking.blur_area``), computed in the pixels of the final swap crop:

    blend_px = max(1, int(sqrt(h * w) * BLEND / 200))      h, w: bounding box of the matte's own extent
    alpha    = GaussianBlur(erode(matte, ellipse(2 * max(1, blend_px // 4) + 1)), k = 2 * blend_px + 1)

so the transition width follows the face's size (about 12 % of it) instead of a fixed pixel ramp, and a mesh rasterised
at any resolution gives the same shape. The matte is first intersected with the convex hull of the landmarks (plus a
forehead extension along the head's own up axis) so a mesh overshoot can never paint background.
"""
from __future__ import annotations

import cv2
import numpy as np

BLEND = 25.0
# A hull smaller than this share of the rasterised surface means the landmark detector failed (a collapsed contour on a
# profile), not that the mesh overshoots; the surface alone then bounds the matte.
HULL_MIN_SHARE = 0.5
FOREHEAD_HEIGHT = 0.6      # forehead extension, as a share of the brow-to-chin distance


def hull_points(landmarks106: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
    """Landmarks plus three forehead points above the brows, along the eye-to-mouth axis (crop pixels)."""
    points = np.asarray(landmarks106, np.float64).reshape(-1, 2)
    five = np.asarray(landmarks5, np.float64).reshape(-1, 2)
    up = five[:2].mean(0) - five[3:5].mean(0)
    length = float(np.linalg.norm(up))
    if length < 1e-6 or not np.isfinite(points).all():
        return points
    up /= length
    across = np.array((-up[1], up[0]))
    along, lateral = points @ up, points @ across
    top, face_height = along.max(), along.max() - along.min()
    band = lateral[along >= top - 0.15 * face_height]
    reach = top + FOREHEAD_HEIGHT * face_height
    centre = 0.5 * (band.min() + band.max())
    extra = np.array([up * reach + across * lateral_value for lateral_value in (band.min(), centre, band.max())])
    return np.vstack((points, extra))


def hull_mask(shape: tuple[int, int], points: np.ndarray, blend: float = BLEND) -> np.ndarray | None:
    """Filled convex hull dilated by ``sqrt(h * w) * blend / 400`` (Ultimate's ``create_landmark_mask``), or ``None``."""
    pixels = np.rint(np.asarray(points, np.float64)).astype(np.int32)
    if len(pixels) < 3:
        return None
    hull = cv2.convexHull(pixels)
    if hull is None or len(hull) < 3:
        return None
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    extent = np.ptp(pixels, axis=0) + 1
    grow = max(1, int(np.sqrt(float(extent[0] * extent[1])) * blend / 400.0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    return cv2.dilate(mask, kernel).astype(bool)


def feathered_coverage(surface: np.ndarray, *, landmarks106: np.ndarray | None = None,
                       landmarks5: np.ndarray | None = None, blend: float = BLEND) -> np.ndarray:
    """Float32 0..1 coverage from a boolean/0..1 rasterised ``surface`` (see module docstring).

    ``landmarks106`` / ``landmarks5`` are in the pixels of ``surface``'s crop; without them the surface alone bounds it.
    """
    matte = np.asarray(surface) > 0.5
    if landmarks106 is not None and landmarks5 is not None:
        points = np.asarray(landmarks106, np.float64)
        if points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all() and len(points) >= 3:
            limit = hull_mask(matte.shape, hull_points(points, landmarks5), blend)
            if limit is not None and (matte & limit).sum() >= HULL_MIN_SHARE * matte.sum():
                matte = matte & limit
    ys, xs = np.nonzero(matte)
    if not len(ys):
        return np.zeros(matte.shape, np.float32)
    height, width = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    blend_px = max(1, int(np.sqrt(float(height * width)) * blend / 200.0))
    erosion = max(1, blend_px // 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1))
    eroded = cv2.erode(matte.astype(np.float32), kernel)
    size = 2 * blend_px + 1
    return np.clip(cv2.GaussianBlur(eroded, (size, size), 0), 0.0, 1.0).astype(np.float32)
