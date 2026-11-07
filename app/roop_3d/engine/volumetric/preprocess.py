"""Deterministic similarity alignment and exact paired affine transforms."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import image_bgr
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS

FFHQ_REFERENCE_5PTS_512 = np.array(((192.98138, 239.94708), (318.90277, 240.19360),
                                    (256.63416, 314.01935), (201.26117, 371.41043),
                                    (313.08905, 371.15118)), np.float64)


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


def aligned_crop(image: np.ndarray, landmarks5: np.ndarray, size: int = 128, *,
                 template: str = "arcface_112") -> AlignedCrop:
    image_bgr(image)
    if size not in (112, 128, 256, 512) or template not in ("arcface_112", "ffhq_512") or np.asarray(landmarks5).shape != (5, 2):
        raise DomainError("ALIGNMENT_CONTRACT", "Expected five landmarks and a supported canonical crop size", 422)
    if template == "ffhq_512":
        reference = FFHQ_REFERENCE_5PTS_512 * (size / 512.0)
    else:
        reference = ARCFACE_REFERENCE_5PTS.astype(np.float64).copy()
        if size != 112:
            reference[:, 0] += 8
            reference *= size / 128
    matrix = similarity(landmarks5, reference)
    crop = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return AlignedCrop(crop, matrix, cv2.invertAffineTransform(matrix))


def roll_compensated_aligned_crop(image: np.ndarray, landmarks5: np.ndarray, roll: float, size: int = 128,
                                 *, compensation: np.ndarray | None = None,
                                 template: str = "arcface_112") -> AlignedCrop:
    """Counter-rotate a target face around its eye midpoint before canonical alignment.

    The returned transforms still map exactly between the original full frame and the
    canonical crop, so the compositor and 3D projector need no special roll handling.
    ``compensation`` may be a precomputed Stage 13 affine for this roll and center.
    """
    points = np.asarray(landmarks5, np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all() or not np.isfinite(roll):
        raise DomainError("LANDMARKS_INVALID", "Roll compensation requires five finite landmarks and a finite roll", 422)
    image_bgr(image)
    center = tuple(float(v) for v in points[:2].mean(axis=0))
    rotation = (np.asarray(compensation, np.float64) if compensation is not None
                else cv2.getRotationMatrix2D(center, -float(np.clip(roll, -180.0, 180.0)), 1.0))
    if rotation.shape != (2, 3) or not np.isfinite(rotation).all():
        raise DomainError("ALIGNMENT_INVALID", "Roll compensation matrix must be finite 2x3 affine", 422)
    height, width = image.shape[:2]
    upright = cv2.warpAffine(image, rotation, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    upright_points = transform_points(points, rotation)
    aligned = aligned_crop(upright, upright_points, size, template=template)
    composed = (np.vstack((aligned.matrix, (0.0, 0.0, 1.0)))
                @ np.vstack((rotation, (0.0, 0.0, 1.0))))[:2]
    return AlignedCrop(aligned.image, composed, cv2.invertAffineTransform(composed))


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
