"""Pose- and scale-resilient face detection around the single-pass SCRFD/RetinaFace/YOLO ``Detector``.

Why this exists (measured on ``roop-keep/single``, see ``outputs/test 1/detect_eval``): the detector letterboxes every
frame into one 640x640 input (aspect ratio preserved, top-left anchored; that part is correct) and so has one fixed
operating scale.  Faces that fill or overflow the frame (close-ups: s1/s2 lost ~55 % of their frames) are far outside the
scale range it was trained on, and rolled faces / faces tiny in a 4K frame are equally missed.

Staged recovery, entered ONLY when the primary pass has no confident face (score >= ``primary``), so frames that already
work cost nothing extra:

1. ``zoom_out`` - the frame is shrunk inside a black border (1.5x, 2x, 3x) so an oversized face lands in range.
2. ``tile``     - large frames are searched in overlapping native-resolution tiles (small faces).
3. ``rotate``   - the frame is rotated +-30/60/90 deg so rolled faces are upright for the detector.

Each stage runs at the fallback threshold (``fallback`` = 0.30 for score-aware callers), maps boxes and landmarks back to
native pixels, and every candidate must pass geometry sanity in the frame it was detected in (where the face is upright)
and, when a ``validator`` is attached, a dense-landmark check (2d106det agreement).  The search stops at the first stage
that yields a validated face; results are merged with the primary pass by NMS.  No score is ever inflated: a recovered
face keeps the score the detector gave it in the transformed view.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Callable

import cv2
import numpy as np

from roop_3d.media.detector import iou

log = logging.getLogger("roop3d.robust_detector")

PRIMARY = 0.50
FALLBACK = 0.30
MAX_FACES = 32

Detection = tuple[np.ndarray, np.ndarray, float]
Validator = Callable[[np.ndarray, np.ndarray, np.ndarray], bool]


# ----------------------------------------------------------------------------------------------- geometry
def geometry_ok(box: np.ndarray, points: np.ndarray, min_px: float = 16.0) -> bool:
    """Roll-invariant sanity of a box + SCRFD five points (eye, eye, nose, mouth, mouth).

    Unlike a "nose below eyes in image-y" test this works on the face's own axes, so it is applied in the view where
    the face is upright; profile faces (narrow box, collapsed eye line) are tolerated up to a box aspect of 0.30."""
    box = np.asarray(box, np.float64)
    points = np.asarray(points, np.float64).reshape(5, 2)
    if not np.isfinite(box).all() or not np.isfinite(points).all():
        return False
    w, h = float(box[2] - box[0]), float(box[3] - box[1])
    if min(w, h) < min_px or not 0.30 <= w / max(h, 1e-6) <= 2.0:
        return False
    pad = 0.25
    if not ((points[:, 0] >= box[0] - pad * w) & (points[:, 0] <= box[2] + pad * w)
            & (points[:, 1] >= box[1] - pad * h) & (points[:, 1] <= box[3] + pad * h)).all():
        return False
    eyes, mouth = points[:2].mean(axis=0), points[3:].mean(axis=0)
    axis = mouth - eyes
    length = float(np.linalg.norm(axis))
    if length < 0.18 * max(w, h):                      # eyes and mouth collapsed onto each other
        return False
    t = float(np.dot(points[2] - eyes, axis) / (length * length))
    # the nose lies between the eyes and the mouth, and the eyes->mouth axis points down or sideways (roll <= ~100 deg)
    return 0.05 <= t <= 0.95 and float(axis[1]) / length >= -0.17


def _nms(items: list[Detection], limit: int = MAX_FACES, overlap: float = 0.4) -> list[Detection]:
    kept: list[Detection] = []
    for item in sorted(items, key=lambda d: -float(d[2])):
        if all(iou(item[0], other[0]) < overlap for other in kept):
            kept.append(item)
        if len(kept) >= limit:
            break
    return kept


def _tile_origins(extent: int, tile: int) -> list[int]:
    if extent <= tile:
        return [0]
    count = int(math.ceil((extent - tile) / (tile * 0.75))) + 1          # >= 25 % overlap
    return sorted({int(round(v)) for v in np.linspace(0, extent - tile, count)})


@dataclass
class _View:
    """One transformed rendering of the frame plus the map from its coordinates back to native pixels."""
    image: np.ndarray
    to_native: Callable[[np.ndarray], np.ndarray]                  # (n, 2) view xy -> native xy
    kind: str
    detail: str
    angle: float = 0.0                                             # view rotation (deg) for box reconstruction


class RobustDetector:
    """Drop-in for ``Detector`` (``detect`` / ``detect_scored`` / attribute access are forwarded to the wrapped one)."""

    def __init__(self, base: Any, *, validator: Validator | None = None, primary: float = PRIMARY, fallback: float = FALLBACK,
                 zoom_out: tuple[float, ...] = (1.5, 2.0, 3.0), tile_sizes: tuple[int, ...] = (1280, 800),
                 max_tiles: int = 12, angles: tuple[float, ...] = (30, -30, 60, -60, 90, -90),
                 rotate_max_side: int = 1280):
        self.base = base
        self.validator = validator
        self.primary, self.fallback = primary, fallback
        self.zoom_out, self.tile_sizes, self.max_tiles = zoom_out, tile_sizes, max_tiles
        self.angles, self.rotate_max_side = angles, rotate_max_side
        self.stats: dict[str, int] = {"frames": 0, "recovery_frames": 0, "recovered": 0, "passes": 0}

    def __getattr__(self, name: str):                              # catalog, _model_id, release_for_build, ...
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    # ------------------------------------------------------------------------------------------- public
    def detect(self, image: np.ndarray, threshold: float, **kwargs):
        scored, record = self.detect_scored(image, threshold, **kwargs)
        return [(box, points) for box, points, _ in scored], record

    def detect_scored(self, image: np.ndarray, threshold: float, *, recover: bool = True,
                      max_passes: int | None = None) -> tuple[list[Detection], dict]:
        """Primary pass at ``threshold``; staged recovery (see module doc) when it holds no confident face.

        The recovery floor is ``max(threshold, fallback)``: a caller that asks for 0.5 never receives sub-0.5 faces, a
        score-aware caller that asks for 0.30 receives validated 0.30-0.50 faces from the recovery stages too."""
        detections, record = self.base.detect_scored(image, threshold)
        detections = list(detections)
        self.stats["frames"] += 1
        confident = max(self.primary, threshold)
        if not recover or any(float(s) >= confident for _, _, s in detections):
            return detections, record
        floor = max(float(threshold), self.fallback)
        budget = [max_passes if max_passes is not None else 10 ** 6]
        found, stage = self._recover(image, floor, budget)
        if not found:
            return detections, record
        self.stats["recovery_frames"] += 1
        self.stats["recovered"] += len(found)
        merged = _nms([*detections, *found])
        record = dict(record)
        record["recovery"] = stage
        return merged, record

    # ------------------------------------------------------------------------------------------ stages
    def _recover(self, image: np.ndarray, floor: float, budget: list[int]) -> tuple[list[Detection], dict | None]:
        for stage in (self._zoom_views, self._tile_views, self._rotation_views):
            for view in stage(image):
                if budget[0] <= 0:
                    return [], None
                budget[0] -= 1
                self.stats["passes"] += 1
                found = self._detect_view(view, floor)
                if found:
                    return found, {"stage": view.kind, "detail": view.detail}
        return [], None

    def _detect_view(self, view: _View, floor: float) -> list[Detection]:
        raw, _ = self.base.detect_scored(view.image, floor)
        out: list[Detection] = []
        for box, points, score in raw:
            if not geometry_ok(box, points):
                continue
            if self.validator is not None:
                try:
                    if not self.validator(view.image, np.asarray(box), np.asarray(points)):
                        continue
                except Exception as error:                        # a failing optional model never takes detection down
                    log.warning("dense-landmark validation failed: %s", error)
            out.append(self._to_native(view, np.asarray(box, np.float64), np.asarray(points, np.float64), float(score)))
        return out

    @staticmethod
    def _to_native(view: _View, box: np.ndarray, points: np.ndarray, score: float) -> Detection:
        pts = view.to_native(points.reshape(5, 2))
        centre = view.to_native(np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]]))[0]
        w, h = box[2] - box[0], box[3] - box[1]
        theta = math.radians(view.angle)
        aabb_w = abs(math.cos(theta)) * w + abs(math.sin(theta)) * h
        aabb_h = abs(math.sin(theta)) * w + abs(math.cos(theta)) * h
        # Area-preserving box of the rotated rectangle's axis-aligned extent (exact at 0 and 90 deg).
        shrink = math.sqrt(w * h / max(aabb_w * aabb_h, 1e-9)) if view.angle else 1.0
        scale_native = _view_scale(view)
        half = np.array([aabb_w, aabb_h]) * shrink * scale_native / 2.0
        native_box = np.concatenate([centre - half, centre + half])
        return native_box.astype(np.float32), pts.astype(np.float32), score

    # ------------------------------------------------------------------------------------------- views
    def _zoom_views(self, image: np.ndarray):
        height, width = image.shape[:2]
        for k in self.zoom_out:
            pad_y, pad_x = int(round(height * (k - 1) / 2)), int(round(width * (k - 1) / 2))
            padded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            offset = np.array([pad_x, pad_y], np.float64)
            yield _View(padded, (lambda p, o=offset: p - o), "zoom_out", f"{k:g}x")

    def _tile_views(self, image: np.ndarray):
        height, width = image.shape[:2]
        if max(height, width) < 1100:
            return
        for tile in self.tile_sizes:
            if tile >= max(height, width) * 0.9:
                continue
            xs, ys = _tile_origins(width, tile), _tile_origins(height, tile)
            if len(xs) * len(ys) > self.max_tiles:
                continue
            for y in ys:
                for x in xs:
                    crop = image[y:y + tile, x:x + tile]
                    offset = np.array([x, y], np.float64)
                    yield _View(crop, (lambda p, o=offset: p + o), "tile", f"{tile}px@{x},{y}")

    def _rotation_views(self, image: np.ndarray):
        height, width = image.shape[:2]
        shrink = min(1.0, self.rotate_max_side / max(height, width))
        base = cv2.resize(image, (max(1, int(round(width * shrink))), max(1, int(round(height * shrink)))),
                          interpolation=cv2.INTER_AREA) if shrink < 1.0 else image
        bh, bw = base.shape[:2]
        for angle in self.angles:
            # cv2 angle > 0 rotates the image counter-clockwise; the canvas grows so nothing is cut off.
            matrix = cv2.getRotationMatrix2D((bw / 2, bh / 2), angle, 1.0)
            cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
            new_w, new_h = int(round(bh * sin + bw * cos)), int(round(bh * cos + bw * sin))
            matrix[0, 2] += new_w / 2 - bw / 2
            matrix[1, 2] += new_h / 2 - bh / 2
            rotated = cv2.warpAffine(base, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            inverse = cv2.invertAffineTransform(matrix)

            def back(p: np.ndarray, inv=inverse, s=shrink) -> np.ndarray:
                return (p @ inv[:, :2].T + inv[:, 2]) / s

            yield _View(rotated, back, "rotate", f"{angle:+g}deg", angle=angle)


def _view_scale(view: _View) -> float:
    """Native pixels per view pixel (all views are similarity maps)."""
    a, b = view.to_native(np.array([[0.0, 0.0], [100.0, 0.0]]))
    return float(np.linalg.norm(b - a) / 100.0)


def dense_validator(source: Any, *, min_agreement: float = 0.6) -> Validator:
    """Validator from the independent 2d106det landmarks: the five points must agree with them.

    ``source`` is a ``LandmarkEngine`` (``has_106`` + ``landmarks106``) or a plain ``(image, box) -> (106, 2)`` callable
    (``DenseLandmarks.infer106``).  Without the 106 model the validator accepts (geometry alone decides) rather than
    inventing a result."""
    from roop_3d.media.face_analysis import agrees_with_106

    def landmarks(image: np.ndarray, box: np.ndarray):
        if source is None:
            return None
        if hasattr(source, "landmarks106"):
            return source.landmarks106(image, box) if getattr(source, "has_106", False) else None
        try:
            return source(image, box)
        except Exception as error:                              # e.g. MODEL_NOT_INSTALLED
            log.debug("106-point landmarks unavailable: %s", error)
            return None

    def check(image: np.ndarray, box: np.ndarray, points: np.ndarray) -> bool:
        dense = landmarks(image, box)
        if dense is None or not np.isfinite(dense).all():
            return True
        _, fraction = agrees_with_106(box, points, dense)
        return bool(fraction >= min_agreement)

    return check
