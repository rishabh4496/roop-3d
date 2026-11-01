"""Thin-plate-spline expression warping on 106-point landmarks with a strictly pinned silhouette.

Moves only expression-bearing inner landmarks (mouth opening and corners, eyebrows, eye openness) toward the
target frame's positions, scaled by `tps_expression_weight`. The face contour, the nose, and the image border
are control points that map to themselves, and the spline interpolates (it does not smooth), so the outer
boundary cannot drift however far the inner points move.

Layout: the 106-point block structure the codebase already relies on in `landmark_crop` (eyes at 33-42 and 87-96,
mouth 52-71): contour 0-32, eye 33-42, brow 43-51, mouth 52-71, nose 72-86, eye 87-96, brow 97-105 (the
standard 2d106 layout; block sizes sum to exactly 106). Because the project deliberately avoids trusting a
detector's ordering blindly, `layout_is_plausible` checks the geometry (brows above eyes above nose above mouth,
in the face's own rotated frame) and the warp refuses to run when it fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from roop_3d.reconstruction.tps_warp import border_anchors, transform_points_with, warp_points_to

BLOCKS: dict[str, range] = {"contour": range(0, 33), "eye_a": range(33, 43), "brow_a": range(43, 52), "mouth": range(52, 72),
                            "nose": range(72, 87), "eye_b": range(87, 97), "brow_b": range(97, 106)}
EXPRESSION_BLOCKS = ("mouth", "eye_a", "eye_b", "brow_a", "brow_b")
PINNED_BLOCKS = ("contour", "nose")
MAX_SHIFT_RATIO = 0.30  # per-point cap as a fraction of face width (spline stays fold-free)


def group(points: np.ndarray, name: str) -> np.ndarray:
    return np.asarray(points, np.float64)[list(BLOCKS[name])]


def layout_is_plausible(points106: np.ndarray) -> bool:
    """Geometric check that the points follow the assumed block layout (in the face's own rotated frame)."""
    points = np.asarray(points106, np.float64)
    if points.shape != (106, 2) or not np.isfinite(points).all():
        return False
    centroids = {name: group(points, name).mean(axis=0) for name in BLOCKS}
    eye_mid = (centroids["eye_a"] + centroids["eye_b"]) / 2
    across = centroids["eye_b"] - centroids["eye_a"]
    spread = float(np.linalg.norm(across))
    if spread < 1e-6:
        return False
    across = across / spread
    down = np.array((-across[1], across[0]))
    if np.dot(centroids["mouth"] - eye_mid, down) <= 0:
        down, across = -down, -across  # eye_a/eye_b may be either side of the face
    height = {name: float(np.dot(c - eye_mid, down)) for name, c in centroids.items()}
    brows = (height["brow_a"] + height["brow_b"]) / 2
    contour = group(points, "contour")
    width = float(np.ptp(contour @ across))
    mouth_width = float(np.ptp(group(points, "mouth") @ across))
    return bool(brows < -0.05 * spread < 0.05 * spread < height["nose"] < height["mouth"]
                and 0.2 * width < spread < 0.9 * width and 0.1 * width < mouth_width < 0.85 * width
                and float(np.ptp(contour @ down)) > 0.3 * width)


@dataclass(frozen=True)
class ExpressionWarp:
    image: np.ndarray
    applied: bool
    reason: str
    warped_landmarks: np.ndarray | None = None   # where the source landmarks ended up
    boundary_drift_px: float = 0.0               # max movement of contour/nose/border control points


def warp_expression(image: np.ndarray, src106: np.ndarray, dst106: np.ndarray, weight: float) -> ExpressionWarp:
    """Shift the inner expression landmarks of `src106` toward `dst106` by `weight`, pinning everything else."""
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("tps_expression_weight must be within 0..1")
    source, target = np.asarray(src106, np.float64), np.asarray(dst106, np.float64)
    if weight <= 0:
        return ExpressionWarp(image, False, "WEIGHT_ZERO")
    if source.shape != (106, 2) or target.shape != (106, 2) or not (np.isfinite(source).all() and np.isfinite(target).all()):
        return ExpressionWarp(image, False, "LANDMARKS_INVALID")
    if not layout_is_plausible(source) or not layout_is_plausible(target):
        return ExpressionWarp(image, False, "LAYOUT_UNVERIFIED")
    height, width = image.shape[:2]
    face_width = float(np.ptp(group(source, "contour")[:, 0]))
    limit = MAX_SHIFT_RATIO * max(face_width, 1.0)
    moved = source.copy()
    for name in EXPRESSION_BLOCKS:
        indices = list(BLOCKS[name])
        shift = weight * (target[indices] - source[indices])
        length = np.linalg.norm(shift, axis=1, keepdims=True)
        shift = np.where(length > limit, shift * (limit / np.maximum(length, 1e-12)), shift)
        moved[indices] = source[indices] + shift
    anchors = border_anchors(width, height)
    control_from, control_to = np.vstack((source, anchors)), np.vstack((moved, anchors))
    pinned = [i for name in PINNED_BLOCKS for i in BLOCKS[name]]
    # Duplicate control points make the spline singular; nothing moves for those, so drop the repeats.
    _, unique = np.unique(np.round(control_from, 6), axis=0, return_index=True)
    keep = np.sort(unique)
    control_from, control_to = control_from[keep], control_to[keep]
    warped = warp_points_to(image, control_from, control_to)
    landmarks = transform_points_with(control_from, control_to, source, size=(width, height))
    drift_probe = np.vstack((source[pinned], anchors))
    drift = transform_points_with(control_from, control_to, drift_probe, size=(width, height)) - drift_probe
    return ExpressionWarp(warped, True, "OK", landmarks, float(np.linalg.norm(drift, axis=1).max()))


def apply_expression_to_crop(generated: np.ndarray, inverse: np.ndarray, target106_frame: np.ndarray, weight: float,
                             detect106: Callable[[np.ndarray], np.ndarray | None], *, detect_size: int = 256) -> ExpressionWarp:
    """Pipeline hook: warp a generated crop toward the target frame's expression.

    `inverse` is the crop->frame affine; `detect106` finds 106 landmarks on an image (the generated face
    itself, so source and destination share the detector's ordering). Any detection failure leaves the
    crop untouched rather than failing the render.
    """
    if weight <= 0:
        return ExpressionWarp(generated, False, "WEIGHT_ZERO")
    height, width = generated.shape[:2]
    to_crop = cv2.invertAffineTransform(np.asarray(inverse, np.float64))
    target = np.asarray(target106_frame, np.float64) @ to_crop[:, :2].T + to_crop[:, 2]
    factor = detect_size / max(height, width) if max(height, width) < detect_size else 1.0
    probe = cv2.resize(generated, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_CUBIC) if factor != 1.0 else generated
    try:
        found = detect106(probe)
    except Exception:  # noqa: BLE001 - a landmark failure must never abort a frame
        return ExpressionWarp(generated, False, "DETECTION_FAILED")
    if found is None:
        return ExpressionWarp(generated, False, "DETECTION_FAILED")
    return warp_expression(generated, np.asarray(found, np.float64) / factor, target, weight)
