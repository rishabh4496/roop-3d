"""Pose-aware frame face analysis (Stage 12): adaptive detection + embedding + dense landmarks for every face of a frame.

One SCRFD pass is run at the tolerant threshold (default 0.30). Detections at or above the primary threshold (0.50) are
kept as they are (``detection_tier="primary"``). Those in [0.30, 0.50) are *profile-tolerant* candidates: SCRFD scores
steep profiles low, so they are accepted only when their landmarks are valid - the five points must be geometrically
plausible AND agree with the independent 106-point landmark model (>= 4 of 5 within 12 % of the box side of a 106 point).
Nothing below the primary threshold survives on score alone.

Every kept face gets: a 512-D L2-normalised ArcFace embedding, the 5-point landmarks, 106-point landmarks and 68-point 3D
landmarks when those local ONNX models exist (reported as null, with a reason, otherwise), head pose, and a small JPEG
crop. Boxes and points are in exact native-frame pixel coordinates (floats; nothing is rounded to whole pixels).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import logging
from typing import Any

import cv2
import numpy as np

from roop_3d.core.pose import as_pose3d, pose_cell
from roop_3d.engine.detection.dense_landmarks import geometric_pose
from roop_3d.faceset_loader import LandmarkEngine, _consistent_with_5pt

log = logging.getLogger("roop3d.face_analysis")

PRIMARY_THRESHOLD = 0.50
TOLERANT_THRESHOLD = 0.30
MAX_FACES = 32
MIN_FACE_PX = 24.0


@dataclass
class AnalyzedFace:
    box: np.ndarray                      # (4,) native px x1, y1, x2, y2
    kps5: np.ndarray                     # (5, 2)
    score: float
    tier: str                            # primary | profile_tolerant
    pose_legacy: tuple[float, float, float]
    embedding: np.ndarray | None
    kps106: np.ndarray | None
    kps68_3d: np.ndarray | None
    kps68_status: str
    validity: dict[str, Any] = field(default_factory=dict)
    thumbnail_b64: str | None = None


def five_point_plausible(box: np.ndarray, points: np.ndarray) -> tuple[bool, str]:
    """Cheap geometric sanity of SCRFD's five landmarks (order: eye, eye, nose, mouth corner, mouth corner)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    if min(w, h) < MIN_FACE_PX:
        return False, "face_too_small"
    if not 0.45 <= w / max(h, 1e-6) <= 1.7:
        return False, "box_aspect"
    pad = 0.15
    inside = ((points[:, 0] >= x1 - pad * w) & (points[:, 0] <= x2 + pad * w)
              & (points[:, 1] >= y1 - pad * h) & (points[:, 1] <= y2 + pad * h))
    if not inside.all():
        return False, "landmark_outside_box"
    eyes_y, nose_y, mouth_y = points[:2, 1].mean(), points[2, 1], points[3:, 1].mean()
    if not (eyes_y < nose_y < mouth_y):
        return False, "vertical_order"
    return True, "ok"


def agrees_with_106(box: np.ndarray, kps5: np.ndarray, kps106: np.ndarray | None) -> tuple[bool, float]:
    """Fraction of the five points lying within 12 % of the box side of some 106-point landmark (>= 0.8 passes)."""
    if kps106 is None:
        return True, float("nan")                       # no independent model: the geometric check alone decides
    side = max(float(box[2] - box[0]), float(box[3] - box[1]), 1.0)
    dist = np.linalg.norm(kps5[:, None, :] - kps106[None, :, :], axis=2).min(axis=1)
    fraction = float((dist <= 0.12 * side).mean())
    return fraction >= 0.8, fraction


def thumbnail_b64(image: np.ndarray, box: np.ndarray, side: int = 96) -> str | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 0.6
    left, top = int(max(0, np.floor(cx - half))), int(max(0, np.floor(cy - half)))
    right, bottom = int(min(w, np.ceil(cx + half))), int(min(h, np.ceil(cy + half)))
    if right - left < 4 or bottom - top < 4:
        return None
    crop = image[top:bottom, left:right]
    scale = side / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
                      interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None


class FaceAnalyzer:
    def __init__(self, detector: Any, recognizer: Any, engine: LandmarkEngine | None = None, *,
                 primary: float = PRIMARY_THRESHOLD, tolerant: float = TOLERANT_THRESHOLD):
        self.detector, self.recognizer, self.engine = detector, recognizer, engine
        self.primary, self.tolerant = primary, tolerant

    def signature(self) -> str:
        selection = getattr(self.detector, '_model_id', 'scrfd')
        selection = selection() if callable(selection) else selection
        return f"{selection}:{self.primary}:{self.tolerant}:{self.engine.signature() if self.engine else 'none'}"

    def analyze(self, image: np.ndarray, threshold: float | None = None, *, thumbnails: bool = True,
                detector: Any = None) -> list[AnalyzedFace]:
        detector = detector if detector is not None else self.detector
        primary = self.primary if threshold is None else float(threshold)
        tolerant = min(self.tolerant, primary)
        scored_detect = getattr(detector, "detect_scored", None)
        if scored_detect is not None:
            detections, _ = scored_detect(image, tolerant)
        else:
            detections = [(box, points, None) for box, points in detector.detect(image, tolerant)[0]]
        faces: list[AnalyzedFace] = []
        for box, points, score in detections[:MAX_FACES]:
            box = np.asarray(box, np.float64)
            kps5 = np.asarray(points, np.float64).reshape(5, 2)
            score = 1.0 if score is None else float(score)
            kps106 = None
            if self.engine is not None and self.engine.has_106:
                try:
                    kps106 = self.engine.landmarks106(image, box)
                    if kps106 is not None and not np.isfinite(kps106).all():
                        kps106 = None
                except Exception as error:               # a failing optional model never takes detection down
                    log.warning("106-point landmarks failed: %s", error)
            plausible, why = five_point_plausible(box, kps5)
            agrees, fraction = agrees_with_106(box, kps5, kps106)
            tier = "primary" if score >= primary else "profile_tolerant"
            validity = {"geometry": why, "agrees_with_106": None if np.isnan(fraction) else round(fraction, 3)}
            if tier == "profile_tolerant" and not (plausible and agrees):
                continue
            if tier == "primary" and not plausible and why in ("face_too_small",):
                continue
            validity["valid"] = bool(plausible and agrees)
            kps68, status = None, "unavailable"
            if self.engine is not None and self.engine.has_68:
                try:
                    candidate = self.engine.landmarks68(image, box)
                    if candidate is not None and np.isfinite(candidate).all():
                        kps68, status = (candidate, "ok") if _consistent_with_5pt(candidate, kps5) else (None, "rejected_inconsistent_with_5pt")
                except Exception as error:
                    log.warning("68-point landmarks failed: %s", error)
                    status = "error"
            embedding = None
            if self.recognizer is not None:
                try:
                    embedding = np.asarray(self.recognizer.embed(image, kps5.astype(np.float32)), np.float32)
                    norm = float(np.linalg.norm(embedding))
                    embedding = embedding / norm if norm > 0 else None
                except Exception as error:
                    log.warning("embedding failed: %s", error)
            faces.append(AnalyzedFace(
                box=box, kps5=kps5, score=score, tier=tier, pose_legacy=tuple(float(v) for v in geometric_pose(kps5)),
                embedding=embedding, kps106=kps106, kps68_3d=kps68, kps68_status=status, validity=validity,
                thumbnail_b64=thumbnail_b64(image, box) if thumbnails else None))
        faces.sort(key=lambda f: float(f.box[0]))                  # left to right: the rank face extraction uses
        return faces


def face_to_dict(face: AnalyzedFace, rank: int, frame_index: int, width: int, height: int, *, detail: str = "full") -> dict:
    """JSON-ready face. ``detail='basic'`` leaves out the heavy fields (embedding, thumbnail, dense landmarks)."""
    yaw, pitch, roll = face.pose_legacy
    pose3d = as_pose3d((yaw, pitch, roll))
    x1, y1, x2, y2 = (float(v) for v in face.box)
    out: dict[str, Any] = {
        "face_id": f"f{frame_index}_{rank}", "face_index": rank, "track_id": None, "cluster_id": None,
        "bbox": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "bbox_px": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "bbox_norm": [round(x1 / width, 6), round(y1 / height, 6), round(x2 / width, 6), round(y2 / height, 6)],
        "kps_5": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in face.kps5],
        "pose": {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)},          # legacy: pitch + = down
        "pose3d": {"yaw": round(float(pose3d.yaw), 2), "pitch": round(float(pose3d.pitch), 2), "roll": round(float(pose3d.roll), 2)},
        "pose_cell": pose_cell(pose3d).name, "pose_source": "scrfd_5pt",
        "confidence": round(face.score, 4), "detection_tier": face.tier, "landmark_validity": face.validity,
        "kps_68_3d_status": face.kps68_status, "has_embedding": face.embedding is not None,
    }
    if detail == "full":
        out["embedding"] = [round(float(v), 6) for v in face.embedding] if face.embedding is not None else None
        out["crop_thumbnail_base64"] = face.thumbnail_b64
        out["kps_68_3d"] = [[round(float(v), 3) for v in p] for p in face.kps68_3d] if face.kps68_3d is not None else None
        out["kps_106"] = [[round(float(v), 3) for v in p] for p in face.kps106] if face.kps106 is not None else None
    return out
