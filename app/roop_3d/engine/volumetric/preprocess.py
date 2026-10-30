"""Deterministic similarity alignment and exact paired affine transforms."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import image_bgr
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS


@dataclass(frozen=True)
class AlignedCrop:
    image: np.ndarray
    matrix: np.ndarray
    inverse: np.ndarray


def similarity(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source, destination = np.asarray(source, np.float64), np.asarray(destination, np.float64)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 2 or not np.isfinite(source).all():
        raise DomainError("LANDMARKS_INVALID", "Alignment requires matching finite point sets", 422)
    # Solve all points simultaneously for a proper rotation, uniform scale and translation.
    rows, targets = [], []
    for (x, y), (u, v) in zip(source, destination):
        rows.extend(((x, -y, 1, 0), (y, x, 0, 1)))
        targets.extend((u, v))
    solution, _, rank, _ = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)
    a, b, tx, ty = solution
    if rank < 4 or a * a + b * b < 1e-12:
        raise DomainError("ALIGNMENT_DEGENERATE", "Face landmarks cannot define an invertible alignment", 422)
    return np.array(((a, -b, tx), (b, a, ty)), np.float64)


def aligned_crop(image: np.ndarray, landmarks5: np.ndarray, size: int = 128) -> AlignedCrop:
    image_bgr(image)
    if size not in (112, 128, 256, 512) or np.asarray(landmarks5).shape != (5, 2):
        raise DomainError("ALIGNMENT_CONTRACT", "Expected five landmarks and a supported canonical crop size", 422)
    reference = ARCFACE_REFERENCE_5PTS.astype(np.float64).copy()
    if size == 112:
        pass
    else:
        reference[:, 0] += 8
        reference *= size / 128
    matrix = similarity(landmarks5, reference)
    crop = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return AlignedCrop(crop, matrix, cv2.invertAffineTransform(matrix))


def landmark_crop(image: np.ndarray, points: np.ndarray, size: int, expansion: float = 1.5,
                  vertical_shift: float = -0.1) -> AlignedCrop:
    """LivePortrait crop: eye-to-mouth axis, rotated point extents, forehead margin."""
    image_bgr(image)
    points = np.asarray(points, np.float64)
    if points.shape == (106, 2):
        eye = (points[[33, 35, 40, 39]].mean(0) + points[[87, 89, 94, 93]].mean(0)) / 2
        mouth = points[[52, 61]].mean(0)
    elif points.shape == (203, 2):
        eye = points[[0, 6, 12, 18, 24, 30, 36, 42]].mean(0)
        mouth = points[[48, 66]].mean(0)
    else:
        raise DomainError("LANDMARKS_INVALID", "Portrait crop requires observed 106 or 203 landmarks", 422)
    down = mouth - eye
    norm = np.linalg.norm(down)
    if not np.isfinite(points).all() or norm < 1e-6:
        raise DomainError("LANDMARKS_INVALID", "Portrait landmarks are degenerate", 422)
    down /= norm
    axes = np.stack(((down[1], -down[0]), down))
    rotated = points @ axes.T
    lower, upper = rotated.min(0), rotated.max(0)
    side = max(upper - lower) * expansion
    center = (lower + upper) / 2 + np.array((0, vertical_shift * side))
    linear = axes * (size / side)
    translation = np.full(2, size / 2) - center * size / side
    matrix = np.column_stack((linear, translation))
    return AlignedCrop(cv2.warpAffine(image, matrix, (size, size)), matrix, cv2.invertAffineTransform(matrix))


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ matrix[:, :2].T + matrix[:, 2]
