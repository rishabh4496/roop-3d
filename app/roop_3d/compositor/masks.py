"""Semantic face masks with a deterministic landmark-hull fallback.

The local BiSeNet file is byte-verified in the Stage 03 registry. Its class
labels and preprocessing were not recorded there, so the adapter validates the
runtime tensor layout and exposes the conventional 19-class mapping as a
candidate that can be replaced by explicit indices when model evidence differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


FACE_CLASS_IDS = (1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13)
OCCLUDER_CLASS_IDS = (6, 17, 18)  # eyeglasses, hair, hat in the candidate 19-class map
SKIN_CLASS_IDS = (1,)


@dataclass(frozen=True)
class FaceMask:
    alpha: np.ndarray
    confidence: float
    source: str
    skin: np.ndarray
    occluder: np.ndarray


class FaceParser(Protocol):
    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return HWC class probabilities, or raise a domain error."""


def _shape2d(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        raise DomainError("COMPOSITOR_SHAPE_INVALID", "Mask dimensions must be positive", 422)
    return int(shape[0]), int(shape[1])


def _validate_landmarks(landmarks: np.ndarray | None) -> np.ndarray:
    if landmarks is None:
        return np.empty((0, 2), dtype=np.float32)
    points = np.asarray(landmarks, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise DomainError("COMPOSITOR_LANDMARKS_INVALID", "Landmarks must be finite Nx2 points", 422)
    return points


def landmark_hull_mask(
    shape: tuple[int, ...],
    landmarks: np.ndarray | None,
    *,
    feather_px: float = 5.0,
) -> np.ndarray:
    """Build a softly feathered convex hull, returning zero when geometry is unusable."""
    height, width = _shape2d(shape)
    if not np.isfinite(feather_px) or feather_px < 0 or feather_px > 64:
        raise ValueError("feather_px must be finite and within 0..64")
    points = _validate_landmarks(landmarks)
    mask = np.zeros((height, width), dtype=np.float32)
    if len(points) < 3:
        return mask
    points = np.rint(points).astype(np.int32)
    hull = cv2.convexHull(points)
    if hull is None or len(hull) < 3:
        return mask
    cv2.fillConvexPoly(mask, hull, 1.0, lineType=cv2.LINE_AA)
    if feather_px > 0:
        # A finite Gaussian kernel gives the boundary fusion a finite support ROI.
        radius = max(1, int(np.ceil(feather_px * 3)))
        kernel = radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=feather_px)
    return np.clip(np.nan_to_num(mask), 0.0, 1.0).astype(np.float32)


def build_face_mask(
    shape: tuple[int, ...],
    landmarks: np.ndarray | None,
    *,
    probabilities: np.ndarray | None = None,
    confidence_threshold: float = 0.55,
    feather_px: float = 5.0,
    face_class_ids: tuple[int, ...] = FACE_CLASS_IDS,
    occluder_class_ids: tuple[int, ...] = OCCLUDER_CLASS_IDS,
    skin_class_ids: tuple[int, ...] = SKIN_CLASS_IDS,
) -> FaceMask:
    """Build a semantic intersection mask; fall back when parsing is uncertain."""
    height, width = _shape2d(shape)
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be within 0..1")
    points = _validate_landmarks(landmarks)
    fallback = landmark_hull_mask((height, width), points, feather_px=feather_px)
    empty = np.zeros((height, width), dtype=np.float32)

    if probabilities is None:
        return FaceMask(fallback, 0.0, "landmark_hull", empty, empty)
    scores = np.asarray(probabilities, dtype=np.float32)
    if (scores.ndim != 3 or scores.shape[:2] != (height, width)
            or not np.isfinite(scores).all() or np.any(scores < -1e-4)):
        return FaceMask(fallback, 0.0, "landmark_hull", empty, empty)
    max_id = max((*face_class_ids, *occluder_class_ids, *skin_class_ids), default=-1)
    if scores.shape[2] <= max_id:
        return FaceMask(fallback, 0.0, "landmark_hull", empty, empty)

    support = fallback > 0.05
    if not support.any():
        # Parser-only callers may have no landmarks; confidence is measured on its own foreground.
        support = np.sum(scores[..., face_class_ids], axis=-1) > 0.25
    if not support.any():
        return FaceMask(fallback, 0.0, "landmark_hull", empty, empty)
    confidence = float(np.mean(np.max(scores[support], axis=-1)))
    if not np.isfinite(confidence) or confidence < confidence_threshold:
        return FaceMask(fallback, max(0.0, min(confidence, 1.0)), "landmark_hull", empty, empty)

    face = np.clip(np.sum(scores[..., face_class_ids], axis=-1), 0.0, 1.0)
    occluder = np.clip(np.sum(scores[..., occluder_class_ids], axis=-1), 0.0, 1.0)
    skin = np.clip(np.sum(scores[..., skin_class_ids], axis=-1), 0.0, 1.0)
    # Restrict semantic islands to a small dilation around the observed face hull.
    if fallback.any():
        radius = max(1, int(round(min(height, width) * 0.035)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        support_mask = cv2.dilate((fallback > 0.05).astype(np.uint8), kernel).astype(np.float32)
        face *= support_mask
        occluder *= support_mask
        skin *= support_mask
    alpha = face * (1.0 - occluder)
    if feather_px > 0:
        radius = max(1, int(np.ceil(feather_px * 3)))
        alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), sigmaX=feather_px)
    alpha = np.clip(np.nan_to_num(alpha), 0.0, 1.0).astype(np.float32)
    return FaceMask(alpha, confidence, "bisenet_candidate_19_class", skin.astype(np.float32), occluder.astype(np.float32))


class BisenetFaceParser:
    """Lazy, hash-checked adapter for the observed local BiSeNet ONNX artifact.

    The normalization mode is configurable because Stage 03 records tensor shape
    and dtype, but not color order or value range. The default RGB [0,1] path is
    therefore a candidate preprocessing contract, not a verified upstream fact.
    """

    def __init__(self, catalog=None, config=None, *, normalization: str = "zero_one"):
        from roop_3d.engine.runtime import RuntimeConfig
        from roop_3d.models.catalog import ModelCatalog

        if normalization not in {"zero_one", "minus_one_one", "imagenet"}:
            raise ValueError("Unsupported parser normalization")
        self.catalog = catalog or ModelCatalog()
        self.config = config or RuntimeConfig()
        self.normalization = normalization
        self.model = None
        self.inference_verified = False

    def _load(self):
        if self.model is None:
            from roop_3d.engine.runtime import OrtModel
            self.model = OrtModel(self.catalog, "bisenet_resnet_34.onnx", self.config)
        return self.model

    @staticmethod
    def _class_probabilities(outputs: dict[str, np.ndarray]) -> np.ndarray:
        candidates = sorted(outputs.items(), key=lambda item: (item[0] != "output", item[0]))
        for _, value in candidates:
            array = np.asarray(value, dtype=np.float32)
            if array.ndim != 4 or array.shape[0] != 1 or not np.isfinite(array).all():
                continue
            channel_axes = [axis for axis in (1, 3) if array.shape[axis] == 19]
            if not channel_axes:
                continue
            axis = channel_axes[0]
            if axis == 1:
                logits = array[0].transpose(1, 2, 0)
            else:
                logits = array[0]
            # The 19-class candidate mapping is intentionally explicit and shape checked.
            logits = logits[..., :19]
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            exp = np.exp(np.clip(shifted, -80, 0))
            return (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)).astype(np.float32)
        raise DomainError("PARSER_OUTPUT_UNSUPPORTED", "Local parser returned no supported 19-class tensor", 503)

    def predict(self, image: np.ndarray) -> np.ndarray:
        from roop_3d.engine.runtime import image_bgr as validate_image

        validate_image(image)
        source = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        rgb = source[:, :, ::-1].astype(np.float32) / 255.0
        if self.normalization == "minus_one_one":
            rgb = rgb * 2.0 - 1.0
        elif self.normalization == "imagenet":
            rgb = (rgb - np.array((0.485, 0.456, 0.406), np.float32)) / np.array((0.229, 0.224, 0.225), np.float32)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        output = self._load().run({"input": tensor})
        probabilities = self._class_probabilities(output)
        if probabilities.shape[:2] != (512, 512):
            probabilities = cv2.resize(probabilities, (512, 512), interpolation=cv2.INTER_LINEAR)
        self.inference_verified = True
        return cv2.resize(probabilities, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    def capability(self) -> dict[str, str]:
        return {"weights": "VERIFIED_LOCAL", "provenance": "UNKNOWN", "license": "UNKNOWN",
                "inference": "VERIFIED_LOCAL" if self.inference_verified else "UNKNOWN",
                "preprocessing": "CANDIDATE", "class_mapping": "CANDIDATE"}

    def release(self) -> None:
        if self.model is not None:
            self.model.release()
        self.model = None
