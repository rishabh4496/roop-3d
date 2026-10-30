"""Foreground protection helpers for glasses, hair, hands and other occluders."""
from __future__ import annotations

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.compositor.masks import OCCLUDER_CLASS_IDS


def semantic_occluder_mask(probabilities: np.ndarray, class_ids: tuple[int, ...] = OCCLUDER_CLASS_IDS) -> np.ndarray:
    """Return a soft occlusion probability map from class logits/probabilities."""
    scores = np.asarray(probabilities, dtype=np.float32)
    if scores.ndim != 3 or not np.isfinite(scores).all() or max(class_ids, default=-1) >= scores.shape[2]:
        raise DomainError("OCCLUSION_PARSER_INVALID", "Semantic occlusion tensor is invalid", 422)
    return np.clip(np.sum(scores[..., class_ids], axis=-1), 0.0, 1.0).astype(np.float32)


def subtract_occluders(face_alpha: np.ndarray, occluder: np.ndarray | None, *, threshold: float = 0.0) -> np.ndarray:
    """Remove foreground pixels from a face alpha; zero occluder alpha stays exact."""
    face = np.asarray(face_alpha, dtype=np.float32)
    if face.ndim != 2 or not np.isfinite(face).all():
        raise DomainError("OCCLUSION_MASK_INVALID", "Face alpha must be a finite 2-D mask", 422)
    face = np.clip(face, 0.0, 1.0)
    if occluder is None:
        return face
    front = np.asarray(occluder, dtype=np.float32)
    if front.shape != face.shape or not np.isfinite(front).all():
        raise DomainError("OCCLUSION_MASK_INVALID", "Occluder mask must match face alpha", 422)
    if not 0 <= threshold < 1:
        raise ValueError("threshold must be within 0..1")
    front = np.clip(front, 0.0, 1.0)
    if threshold:
        binary = (front >= threshold).astype(np.float32)
        front = cv2.GaussianBlur(binary, (3, 3), 0)
    return np.clip(face * (1.0 - front), 0.0, 1.0)


def combine_occluders(*masks: np.ndarray | None) -> np.ndarray | None:
    present = [np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0) for mask in masks if mask is not None]
    if not present:
        return None
    if any(mask.ndim != 2 or not np.isfinite(mask).all() or mask.shape != present[0].shape for mask in present):
        raise DomainError("OCCLUSION_MASK_INVALID", "All foreground masks must be finite and have matching dimensions", 422)
    result = present[0].copy()
    for mask in present[1:]:
        result = 1.0 - (1.0 - result) * (1.0 - mask)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
