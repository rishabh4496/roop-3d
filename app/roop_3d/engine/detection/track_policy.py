"""Identity-lock association policy for :class:`FaceTracker` (opt-in, ``identity_lock_v1``).

The legacy tracker links a detection to the best earlier track and silently starts a
new identity when the link fails. For a replacement pipeline that is the dangerous
direction: a wrong link swaps the wrong person. This policy therefore never *guesses*:

* a link made on weak appearance evidence is flagged ``UNCERTAIN`` and the swap is withheld;
* two plausible identities for one face (similar people, crossing faces) are ``AMBIGUOUS``
  and withheld until the evidence separates them again;
* a track that disappears (occlusion, exit, scene cut) becomes ``LOST`` and is only
  ``REACQUIRED`` by a strict, margin-checked appearance match followed by a short probation.

Every threshold is a documented heuristic on ArcFace cosine similarity. They are
recorded in the policy fingerprint so any change invalidates saved identity locks, but
they have not been calibrated against a labelled benchmark. See docs/IDENTITY_LOCK_STAGE1.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
from typing import Any

import cv2
import numpy as np

POLICY_VERSION = "identity_lock_v1"

TRACKED = "TRACKED"
UNCERTAIN = "UNCERTAIN"
LOST = "LOST"
REACQUIRED = "REACQUIRED"

# Reasons carried on TrackedFace.reason / track events.
AMBIGUOUS = "AMBIGUOUS_IDENTITY"
APPEARANCE_LOW = "APPEARANCE_LOW"
RECOVERING = "RECOVERING"
REACQUIRING = "REACQUIRING"
UNCERTAIN_TIMEOUT = "UNCERTAIN_TIMEOUT"
SCENE_CUT = "SCENE_CUT"
NOT_DETECTED = "NOT_DETECTED"
EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class TrackPolicy:
    strong_cosine: float = 0.55        # gallery may adapt only above this
    match_cosine: float = 0.35         # appearance alone supports the link
    continuity_cosine: float = 0.25    # with strong spatial continuity the link is kept
    weak_cosine: float = 0.15          # below this only spatial continuity can keep a link (UNCERTAIN)
    continuity_iou: float = 0.5
    reacquire_cosine: float = 0.5
    reacquire_margin: float = 0.08
    ambiguity_margin: float = 0.06
    reacquire_frames: int = 3          # frames withheld after (re)acquisition
    recovery_frames: int = 2           # clean frames required after an uncertain run
    uncertain_limit: int = 45          # uncertain frames before a track is declared LOST
    lost_limit: int = 64
    lost_retention_frames: int = 1800
    min_confidence: float = 0.35
    anchor_limit: int = 6
    gallery_rate: float = 0.1

    def __post_init__(self) -> None:
        cosines = (self.strong_cosine, self.match_cosine, self.continuity_cosine, self.weak_cosine)
        if not all(-1 < value < 1 for value in cosines + (self.reacquire_cosine,)):
            raise ValueError("Cosine thresholds must lie in (-1, 1)")
        if not self.strong_cosine >= self.match_cosine >= self.continuity_cosine >= self.weak_cosine:
            raise ValueError("Cosine thresholds must be ordered strong >= match >= continuity >= weak")
        if not 0 < self.continuity_iou <= 1 or not 0 <= self.min_confidence <= 1:
            raise ValueError("Invalid IoU/confidence threshold")
        if not 0 < self.reacquire_margin < 1 or not 0 < self.ambiguity_margin < 1:
            raise ValueError("Margins must lie in (0, 1)")
        if not 1 <= self.reacquire_frames <= 60 or not 0 <= self.recovery_frames <= 60:
            raise ValueError("Probation frame counts must lie in 1..60 / 0..60")
        if not 1 <= self.uncertain_limit <= 3600 or not 1 <= self.lost_limit <= 256:
            raise ValueError("Uncertain/lost limits out of bounds")
        if not 1 <= self.lost_retention_frames <= 100000 or not 1 <= self.anchor_limit <= 16:
            raise ValueError("Retention/anchor limits out of bounds")
        if not 0 < self.gallery_rate < 1:
            raise ValueError("Gallery rate must lie in (0, 1)")

    def as_dict(self) -> dict[str, Any]:
        return {"version": POLICY_VERSION, **{item.name: getattr(self, item.name) for item in fields(self)}}

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class TrackEvent:
    frame: int
    track_id: int
    state: str
    reason: str
    scene_id: int


@dataclass
class TrackMemory:
    track_id: int
    observation: Any                      # last FaceObservation (kept untyped to avoid an import cycle)
    gallery: np.ndarray
    anchors: list[np.ndarray]
    predicted_box: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    missed: int = 0
    age: int = 1
    state: str = TRACKED
    hold: int = 0                         # clean frames still required before a swap is allowed
    uncertain_run: int = 0
    # Set by an ambiguity episode; only appearance evidence that separates this track from every
    # rival clears it, because spatial continuity through a crossing proves nothing between look-alikes.
    needs_discrimination: bool = False
    last_frame: int = 0
    scene_id: int = 0
    parked_by_cut: bool = False
    lost_at: int | None = None


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


def appearance(memory: TrackMemory, embedding: np.ndarray) -> float:
    """Best cosine against the stable gallery and its diverse anchors (never the last frame)."""
    best = float(np.dot(memory.gallery, embedding))
    for anchor in memory.anchors:
        best = max(best, float(np.dot(anchor, embedding)))
    return best


def normalized_landmarks(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float32).reshape(-1, 2)
    center = np.array(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), np.float32)
    size = max(float(box[2] - box[0]), float(box[3] - box[1]), 1e-6)
    return (points - center) / size


def landmark_similarity(previous_points: np.ndarray, previous_box: np.ndarray,
                        points: np.ndarray, box: np.ndarray) -> float:
    """1.0 for identical face-relative geometry, falling to 0 at 0.25 face-sizes mean error."""
    left, right = normalized_landmarks(previous_points, previous_box), normalized_landmarks(points, box)
    if left.shape != right.shape:
        return 0.0
    error = float(np.linalg.norm(left - right, axis=1).mean())
    return float(np.clip(1.0 - error / 0.25, 0.0, 1.0))


def confidence(cosine: float, overlap: float, policy: TrackPolicy) -> float:
    """0..1 evidence that a link is the same person: 70% appearance, 30% spatial continuity."""
    span = max(policy.strong_cosine - policy.weak_cosine, 1e-6)
    appearance_term = float(np.clip((cosine - policy.weak_cosine) / span, 0.0, 1.0))
    spatial_term = float(np.clip(overlap / 0.5, 0.0, 1.0))
    return 0.7 * appearance_term + 0.3 * spatial_term


def propagate_box(previous_gray: np.ndarray, current_gray: np.ndarray, box: np.ndarray, *,
                  max_points: int = 80) -> tuple[np.ndarray, float] | None:
    """Move ``box`` from ``previous_gray`` to ``current_gray`` with forward/backward-checked LK flow.

    Returns ``(box, trusted_fraction)`` or None when too few features survive the
    forward-backward check (the caller then keeps its constant-velocity prediction).
    """
    if previous_gray.shape != current_gray.shape or previous_gray.ndim != 2:
        return None
    height, width = previous_gray.shape
    x1, y1 = int(max(0, np.floor(box[0]))), int(max(0, np.floor(box[1])))
    x2, y2 = int(min(width, np.ceil(box[2]))), int(min(height, np.ceil(box[3])))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    mask = np.zeros_like(previous_gray)
    mask[y1:y2, x1:x2] = 255
    points = cv2.goodFeaturesToTrack(previous_gray, max_points, 0.01, 3, mask=mask)
    if points is None or len(points) < 6:
        return None
    window = dict(winSize=(21, 21), maxLevel=3)
    forward, status_a, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None, **window)
    backward, status_b, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, forward, None, **window)
    if forward is None or backward is None:
        return None
    error = np.linalg.norm((points - backward).reshape(-1, 2), axis=1)
    good = (status_a.reshape(-1) == 1) & (status_b.reshape(-1) == 1) & (error < 1.5)
    if int(good.sum()) < 6:
        return None
    before, after = points.reshape(-1, 2)[good], forward.reshape(-1, 2)[good]
    shift = np.median(after - before, axis=0)
    spread_before = float(np.median(np.linalg.norm(before - np.median(before, axis=0), axis=1)))
    spread_after = float(np.median(np.linalg.norm(after - np.median(after, axis=0), axis=1)))
    scale = float(np.clip(spread_after / spread_before, 0.8, 1.25)) if spread_before > 1.0 else 1.0
    center = np.array(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)) + shift
    half = np.array(((box[2] - box[0]) / 2, (box[3] - box[1]) / 2)) * scale
    moved = np.array((center[0] - half[0], center[1] - half[1], center[0] + half[0], center[1] + half[1]), np.float32)
    return moved, float(good.sum()) / len(points)
