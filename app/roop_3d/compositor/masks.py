"""Semantic face masks with a deterministic landmark-hull fallback.

The local BiSeNet file is byte-verified in the Stage 03 registry. Its class
labels and preprocessing were not recorded there, so the adapter validates the
runtime tensor layout and exposes the conventional 19-class mapping as a
candidate that can be replaced by explicit indices when model evidence differs.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
import threading
from typing import Protocol

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


# Explicit semantic class partitioning per VFX Compositing specification (BiSeNet 19-Class)
INCLUDED_SKIN_FACE_CLASSES: tuple[int, ...] = (1, 2, 3, 4, 5, 10, 11, 12, 13)
STRICT_FORBIDDEN_CLASSES: tuple[int, ...] = (6, 7, 8, 9, 14, 15, 16, 17, 18)
FACE_CLASS_IDS = INCLUDED_SKIN_FACE_CLASSES
OCCLUDER_CLASS_IDS = (6, 17, 18)  # eyeglasses, hair, hat in standard 2D occlusion mask
SKIN_CLASS_IDS = (1,)

INCLUDED_SKIN_FACE_CLASSES_DICT: dict[int, str] = {
    1: "skin", 2: "l_brow", 3: "r_brow", 4: "l_eye", 5: "r_eye",
    10: "nose", 11: "mouth", 12: "u_lip", 13: "l_lip",
}
STRICT_FORBIDDEN_CLASSES_DICT: dict[int, str] = {
    14: "neck", 15: "neck_l", 16: "cloth", 17: "hair", 18: "hat",
    7: "l_ear", 8: "r_ear", 9: "ear_r", 6: "eye_g",
}


def extract_forbidden_mask(
    probabilities: np.ndarray,
    forbidden_classes: tuple[int, ...] = STRICT_FORBIDDEN_CLASSES,
) -> np.ndarray:
    """Extract hard-forbidden hair, neck, ear, cloth, hat and glasses mask from BiSeNet probabilities."""
    scores = np.asarray(probabilities, dtype=np.float32)
    if scores.ndim != 3 or not np.isfinite(scores).all():
        raise DomainError("OCCLUSION_PARSER_INVALID", "Semantic probability tensor is invalid", 422)
    valid_ids = [cid for cid in forbidden_classes if cid < scores.shape[2]]
    if not valid_ids:
        return np.zeros(scores.shape[:2], dtype=np.float32)
    return np.clip(np.sum(scores[..., valid_ids], axis=-1), 0.0, 1.0).astype(np.float32)


def compute_dual_bisenet_forbidden_mask(
    parser: FaceParser | None,
    target_crop: np.ndarray,
    swapped_crop: np.ndarray | None = None,
    *,
    forbidden_classes: tuple[int, ...] = STRICT_FORBIDDEN_CLASSES,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Run BiSeNet (resnet34 / resnet18) on both the target ROI and the 3D-swapped face crop.

    Returns:
        (target_forbidden, swap_forbidden, combined_forbidden_mask)
    """
    height, width = target_crop.shape[:2]
    empty = np.zeros((height, width), dtype=np.float32)
    if parser is None:
        return empty, None, empty
    try:
        target_prob = parser.predict(target_crop)
        target_forbidden = extract_forbidden_mask(target_prob, forbidden_classes)
    except Exception:
        target_forbidden = empty

    swap_forbidden = None
    if swapped_crop is not None:
        try:
            swap_prob = parser.predict(swapped_crop)
            if swap_prob.shape[:2] != (height, width):
                swap_prob = cv2.resize(swap_prob, (width, height), interpolation=cv2.INTER_LINEAR)
            swap_forbidden = extract_forbidden_mask(swap_prob, forbidden_classes)
        except Exception:
            swap_forbidden = None

    if swap_forbidden is not None:
        combined = np.maximum(target_forbidden, swap_forbidden)
    else:
        combined = target_forbidden
    return target_forbidden, swap_forbidden, np.clip(combined, 0.0, 1.0).astype(np.float32)


def turn_ratio(landmarks5) -> float | None:
    """Inter-ocular distance over eye-midpoint-to-mouth-midpoint distance from the five face points.

    About 0.85-1.0 for a frontal face; it falls roughly with cos(yaw) as the far eye disappears behind the nose
    (about 0.65 at 45 degrees, about 0.1 in profile).  None when the points are unusable.
    """
    try:
        points = np.asarray(landmarks5, np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if points.shape[0] < 5 or not np.isfinite(points[:5]).all():
        return None
    eyes_gap = float(np.linalg.norm(points[0] - points[1]))
    eye_mouth = float(np.linalg.norm(points[:2].mean(axis=0) - points[3:5].mean(axis=0)))
    return eyes_gap / eye_mouth if eye_mouth > 1e-6 else None


def chin_line_clamp(mask: np.ndarray, landmarks5, *, fraction: float = 0.78, feather: float = 0.10) -> np.ndarray:
    """Cut a matte below the chin using only the five face points (robust at every yaw).

    The line is perpendicular to the face's up-vector (eye midpoint -> mouth midpoint), ``fraction`` of the eye-to-mouth
    distance below the mouth centre (about where the chin ends), and fades over ``feather`` of that distance.  It replaces
    the 106-point submental curve on turned faces, where that contour collapses, so the neck and the collar of the source
    photograph that the mesh also covers never reach the frame.
    """
    values = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    try:
        points = np.asarray(landmarks5, np.float64).reshape(-1, 2)
    except (TypeError, ValueError):
        return values
    if points.shape[0] < 5 or not np.isfinite(points[:5]).all():
        return values
    eyes, mouth = points[:2].mean(axis=0), points[3:5].mean(axis=0)
    axis = mouth - eyes
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return values
    axis /= length
    height, width = values.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    below = ((xs - mouth[0]) * axis[0] + (ys - mouth[1]) * axis[1]) / length       # 0 at the mouth, 1 = one eye-mouth distance
    keep = np.clip((fraction + feather - below) / max(feather, 1e-3), 0.0, 1.0)
    return (values * keep).astype(np.float32)


#: Below this turn ratio (about 40 degrees of yaw) the jaw contour that feeds the submental cutoff is unreliable.
JAW_CUTOFF_MIN_TURN_RATIO = 0.7


def fill_mask_holes(mask: np.ndarray, *, max_hole_fraction: float = 0.04) -> np.ndarray:
    """Close enclosed holes of a soft 0..1 silhouette (eye sockets, a missed mouth corner).

    A face silhouette is one solid region; a hole inside it can only be a mesh gap, and nothing downstream can refill
    it, so the original eye or lip shows through the swap.  Only holes not touching the border and smaller than
    ``max_hole_fraction`` of the image are filled (a real gap such as a hand or a microphone stays open).
    """
    values = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    binary = (values > 0.5).astype(np.uint8)
    if not binary.any():
        return values
    inverse = 1 - binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=4)
    height, width = binary.shape
    limit = max_hole_fraction * height * width
    filled = values.copy()
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if touches_border or area > limit:
            continue
        region = labels == label
        # Feather the patch inwards so the filled eye does not start at a hard edge.
        filled[region] = 1.0
    return np.maximum(filled, cv2.GaussianBlur(filled, (0, 0), 1.2) * (filled > 0.5))


def confident_forbidden(forbidden: np.ndarray, *, low: float = 0.5, high: float = 0.85) -> np.ndarray:
    """Keep confident hair/neck/ear evidence from a parser run on a SYNTHESISED crop and drop its weak fringe.

    BiSeNet on a generated face over a flat canvas reports a diffuse 0.2-0.5 non-face probability across the mouth,
    chin and cheek; subtracting that softly erodes the swap exactly where the identity lives.  Values below ``low``
    become 0, values above ``high`` stay 1, with a smooth ramp between.
    """
    values = np.clip(np.asarray(forbidden, np.float32), 0.0, 1.0)
    ramp = np.clip((values - low) / max(high - low, 1e-6), 0.0, 1.0)
    return (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)


def compose_active_mask(
    mesh_face_only_mask: np.ndarray,
    xseg_face_mask: np.ndarray | None,
    forbidden_hair_neck_ear_mask: np.ndarray,
) -> np.ndarray:
    """Hard-subtract STRICT_FORBIDDEN_CLASSES from 3D mesh silhouette mask before blending:
    active_mask = mesh_face_only_mask * xseg_face_mask * (1.0 - forbidden_hair_neck_ear_mask)
    """
    mesh = np.clip(np.asarray(mesh_face_only_mask, dtype=np.float32), 0.0, 1.0)
    if xseg_face_mask is not None:
        xseg = np.clip(np.asarray(xseg_face_mask, dtype=np.float32), 0.0, 1.0)
        if xseg.shape != mesh.shape:
            xseg = cv2.resize(xseg, (mesh.shape[1], mesh.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        xseg = np.ones_like(mesh, dtype=np.float32)
    forbidden = np.clip(np.asarray(forbidden_hair_neck_ear_mask, dtype=np.float32), 0.0, 1.0)
    if forbidden.shape != mesh.shape:
        forbidden = cv2.resize(forbidden, (mesh.shape[1], mesh.shape[0]), interpolation=cv2.INTER_LINEAR)
    active_mask = mesh * xseg * (1.0 - forbidden)
    return np.clip(active_mask, 0.0, 1.0).astype(np.float32)


def apply_submental_jaw_cutoff(
    mask: np.ndarray,
    landmarks: np.ndarray | None,
    *,
    falloff_px: float = 6.0,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Automatic Submental Chin-to-Neck Gradient Cutoff.

    Clamps the lower boundary of the mask strictly along the 3D mandibular jaw ridge
    (gonion-to-menton curve) with a tight 4px–8px inward SDF falloff so zero swap pixels
    ever spill onto the throat or neck.
    """
    m = np.asarray(mask, dtype=np.float32)
    h, w = m.shape[:2] if shape is None else shape
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    if not 2.0 <= falloff_px <= 32.0:
        raise ValueError("falloff_px must be within 2.0..32.0")
    if landmarks is None or len(landmarks) < 3:
        return np.clip(m, 0.0, 1.0)

    points = np.asarray(landmarks, dtype=np.float32)
    if len(points) == 68:
        jaw_pts = points[0:17]
    elif len(points) == 106:
        jaw_pts = points[0:33]
    elif len(points) >= 17:
        jaw_pts = points[0:17]
    elif len(points) == 5:
        eye_l, eye_r = points[0], points[1]
        mouth_l, mouth_r = points[3], points[4]
        mouth_center = (mouth_l + mouth_r) * 0.5
        eye_center = (eye_l + eye_r) * 0.5
        face_axis_y = mouth_center - eye_center
        face_axis_x = np.array([-face_axis_y[1], face_axis_y[0]], dtype=np.float32)
        norm_x = float(np.linalg.norm(face_axis_x))
        if norm_x > 1e-6:
            face_axis_x /= norm_x
        mouth_width = float(np.linalg.norm(mouth_r - mouth_l))
        menton = mouth_center + face_axis_y * 0.45
        gonion_r = mouth_center - face_axis_x * (mouth_width * 1.2) - face_axis_y * 0.1
        gonion_l = mouth_center + face_axis_x * (mouth_width * 1.2) - face_axis_y * 0.1
        t = np.linspace(0, 1, 9, dtype=np.float32)[:, None]
        right_half = (1 - t) * gonion_r + t * menton
        left_half = (1 - t) * menton + t * gonion_l
        jaw_pts = np.vstack((right_half[:-1], left_half))
    else:
        return np.clip(m, 0.0, 1.0)

    p_start = jaw_pts[0]
    p_end = jaw_pts[-1]
    margin = max(h, w) + 100
    poly_pts = [
        *jaw_pts,
        [p_end[0] + (50 if p_end[0] >= w / 2 else -50), p_end[1] - 50],
        [w + margin, -margin],
        [-margin, -margin],
        [p_start[0] + (-50 if p_start[0] < w / 2 else 50), p_start[1] - 50],
    ]
    face_half = np.zeros((h, w), dtype=np.uint8)
    int_pts = np.rint(np.array(poly_pts, dtype=np.float32)).astype(np.int32)
    cv2.fillPoly(face_half, [int_pts], 255)

    dist = cv2.distanceTransform(face_half, cv2.DIST_L2, 5)
    falloff = np.clip(dist / float(falloff_px), 0.0, 1.0)
    falloff = falloff * falloff * (3.0 - 2.0 * falloff)
    return np.clip(m * falloff, 0.0, 1.0).astype(np.float32)


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
    face_class_ids: tuple[int, ...] = INCLUDED_SKIN_FACE_CLASSES,
    occluder_class_ids: tuple[int, ...] = STRICT_FORBIDDEN_CLASSES,
    skin_class_ids: tuple[int, ...] = SKIN_CLASS_IDS,
    guide_image: np.ndarray | None = None,
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
        if guide_image is not None and guide_image.shape[:2] == (height, width):
            from roop_3d.compositor.hairline import refine_hairline_matte
            alpha = refine_hairline_matte(alpha, guide_image, radius=8, eps=1e-4, smooth_feather_px=feather_px)
        else:
            radius = max(1, int(np.ceil(feather_px * 3)))
            alpha = cv2.GaussianBlur(alpha, (2 * radius + 1, 2 * radius + 1), sigmaX=feather_px)
    alpha = np.clip(np.nan_to_num(alpha), 0.0, 1.0).astype(np.float32)
    return FaceMask(alpha, confidence, "bisenet_candidate_19_class", skin.astype(np.float32), occluder.astype(np.float32))


_SOFTMAX_POOL: ThreadPoolExecutor | None = None
_SOFTMAX_POOL_LOCK = threading.Lock()


def _softmax_pool() -> ThreadPoolExecutor:
    global _SOFTMAX_POOL
    with _SOFTMAX_POOL_LOCK:
        if _SOFTMAX_POOL is None:
            _SOFTMAX_POOL = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) // 2)), thread_name_prefix="parser-softmax")
        return _SOFTMAX_POOL


def softmax_last_axis(logits: np.ndarray, *, chunks: int = 8) -> np.ndarray:
    """Softmax over the last axis, split into row bands that numpy evaluates in parallel (it releases the GIL).

    Element-for-element the same arithmetic as the single-thread version (so bit-identical); on a 512x512x19 map the
    19-class softmax was a visible slice of every composite frame.
    """
    logits = np.asarray(logits, np.float32)
    out = np.empty(logits.shape, np.float32)
    rows = logits.shape[0]
    if rows < 64 or chunks <= 1:
        bands = [slice(0, rows)]
    else:
        edges = np.linspace(0, rows, chunks + 1, dtype=int)
        bands = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def work(band: slice) -> None:
        part = logits[band]
        exp = np.exp(np.clip(part - np.max(part, axis=-1, keepdims=True), -80, 0))
        out[band] = exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), 1e-12)

    if len(bands) == 1:
        work(bands[0])
    else:
        list(_softmax_pool().map(work, bands))
    return out


class BisenetFaceParser:
    """Lazy, hash-checked adapter for the observed local BiSeNet ONNX artifact (resnet34 / resnet18).

    The normalization mode is configurable because Stage 03 records tensor shape
    and dtype, but not color order or value range. The default RGB [0,1] path is
    therefore a candidate preprocessing contract, not a verified upstream fact.
    """

    def __init__(self, catalog=None, config=None, *, normalization: str = "zero_one", model_name: str | None = None):
        from roop_3d.engine.runtime import RuntimeConfig
        from roop_3d.models.catalog import ModelCatalog

        if normalization not in {"zero_one", "minus_one_one", "imagenet"}:
            raise ValueError("Unsupported parser normalization")
        self.catalog = catalog or ModelCatalog()
        self.config = config or RuntimeConfig()
        self.normalization = normalization
        self.model_name = model_name
        self.model = None
        self.inference_verified = False
        # The parser is stateless and deterministic, and the compositor asks for the same crop twice per frame (face
        # mask and dual-forbidden mask).  A few results are kept by image content, which also lets a spare thread compute
        # a frame's parses ahead of the composite stage (``prefetch``).  ``ROOP3D_PARSER_CACHE=0`` disables it.
        self.cache_enabled = os.environ.get("ROOP3D_PARSER_CACHE", "1") != "0"
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self._inflight: dict[tuple, threading.Event] = {}
        self._unconsumed: set[tuple] = set()          # prefetched results the composite stage has not read yet
        self._cache_lock = threading.Lock()

    def _load(self):
        if self.model is None:
            from roop_3d.engine.runtime import OrtModel
            name = self.model_name or "bisenet_resnet_34.onnx"
            if self.catalog is not None and hasattr(self.catalog, "has") and not self.catalog.has(name):
                for candidate in ("bisenet_resnet_34.onnx", "bisenet_resnet18.onnx", "bisenet_resnet_18.onnx", "bisenet.onnx"):
                    if self.catalog.has(candidate):
                        name = candidate
                        break
            self.model = OrtModel(self.catalog, name, self.config)
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
            return softmax_last_axis(logits)
        raise DomainError("PARSER_OUTPUT_UNSUPPORTED", "Local parser returned no supported 19-class tensor", 503)

    #: Frames the swap stage may parse ahead of the composite stage (2 crops each); bounds the cache and its memory.
    PREFETCH_LEAD_CROPS = 6

    def prefetch_pressure(self) -> int:
        with self._cache_lock:
            return len(self._unconsumed)

    def predict(self, image: np.ndarray, *, prefetch: bool = False) -> np.ndarray:
        if not self.cache_enabled:
            return self._predict(image)
        array = np.ascontiguousarray(image)
        key = (array.shape, hashlib.blake2b(array.view(np.uint8).reshape(-1), digest_size=16).digest())
        while True:
            with self._cache_lock:
                hit = self._cache.get(key)
                if hit is not None:
                    self._cache.move_to_end(key)
                    if not prefetch:
                        self._unconsumed.discard(key)
                    return hit
                pending = self._inflight.get(key)
                if pending is None:
                    self._inflight[key] = threading.Event()
                    break
            pending.wait(30.0)
        try:
            result = self._predict(array)
            result.setflags(write=False)          # shared between callers: nobody may edit it in place
            with self._cache_lock:
                self._cache[key] = result
                if prefetch:
                    self._unconsumed.add(key)
                while len(self._cache) > self.PREFETCH_LEAD_CROPS * 2 + 8:
                    evicted, _ = self._cache.popitem(last=False)
                    self._unconsumed.discard(evicted)
            return result
        finally:
            with self._cache_lock:
                self._inflight.pop(key).set()

    def _predict(self, image: np.ndarray) -> np.ndarray:
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
        with self._cache_lock:
            self._cache.clear()
            self._unconsumed.clear()
        if self.model is not None:
            self.model.release()
        self.model = None
