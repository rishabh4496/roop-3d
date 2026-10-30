"""Lower-face masks and mouth-only fusion for optional audio-driven edits."""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


LIP_CLASS_IDS = (11, 12, 13)  # Candidate BiSeNet mouth/upper-lip/lower-lip indices.


def lower_face_soft_mask(shape: tuple[int, ...], probabilities: np.ndarray | None = None, *,
                         feather_px: float = 3.0) -> np.ndarray:
    """Return a soft lip region; use parser lip classes or a bounded lower-face ellipse."""
    if len(shape) < 2 or min(int(shape[0]), int(shape[1])) <= 0:
        raise DomainError("MOUTH_MASK_SHAPE_INVALID", "Mouth mask dimensions must be positive", 422)
    height, width = int(shape[0]), int(shape[1])
    if not np.isfinite(feather_px) or not 0 <= feather_px <= 32:
        raise ValueError("feather_px must be finite and within 0..32")
    mask = None
    if probabilities is not None:
        scores = np.asarray(probabilities, dtype=np.float32)
        if (scores.ndim == 3 and scores.shape[:2] == (height, width) and scores.shape[2] > max(LIP_CLASS_IDS)
                and np.isfinite(scores).all() and np.all(scores >= -1e-4)):
            candidate = np.clip(np.sum(scores[..., LIP_CLASS_IDS], axis=-1), 0, 1)
            if np.count_nonzero(candidate > 0.05) >= max(4, (height * width) // 4096):
                mask = candidate
    if mask is None:
        mask = np.zeros((height, width), np.float32)
        center = (int(round(width * 0.5)), int(round(height * 0.72)))
        axes = (max(2, int(round(width * 0.22))), max(2, int(round(height * 0.105))))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1, lineType=cv2.LINE_AA)
    if feather_px > 0:
        radius = max(1, int(np.ceil(feather_px * 3)))
        mask = cv2.GaussianBlur(mask.astype(np.float32), (2 * radius + 1, 2 * radius + 1), feather_px)
    return np.clip(np.nan_to_num(mask), 0, 1).astype(np.float32)


def composite_mouth_region(base: np.ndarray, synthesized: np.ndarray, mouth_mask: np.ndarray,
                           blend: float = 1.0) -> np.ndarray:
    """Apply only the synthesized mouth pixels, preserving all pixels outside the soft mask."""
    original, edited = np.asarray(base), np.asarray(synthesized)
    alpha = np.asarray(mouth_mask, dtype=np.float32)
    if (original.ndim != 3 or original.shape[2] != 3 or edited.shape != original.shape
            or alpha.shape != original.shape[:2] or not np.isfinite(original).all()
            or not np.isfinite(edited).all() or not np.isfinite(alpha).all()):
        raise DomainError("MOUTH_COMPOSITE_INPUT_INVALID", "Mouth images and mask have incompatible shapes", 422)
    if not np.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError("Mouth blend must be finite and within 0..1")
    if blend == 0:
        return original.copy()
    weight = np.clip(alpha, 0, 1) * blend
    active = weight > 0
    result = original.astype(np.float32, copy=True)
    result[active] = (original.astype(np.float32)[active] * (1 - weight[active, None])
                      + edited.astype(np.float32)[active] * weight[active, None])
    if np.issubdtype(original.dtype, np.integer):
        limits = np.iinfo(original.dtype)
        return np.clip(result, limits.min, limits.max).round().astype(original.dtype)
    return result.astype(original.dtype)
