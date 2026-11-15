"""Identity-lock association policy for :class:`FaceTracker` (opt-in, ``identity_lock_v1``).

The legacy tracker links a detection to the best earlier track and silently starts a
new identity when the link fails. For a replacement pipeline that is the dangerous
direction: a wrong link swaps the wrong person. This policy therefore never *guesses*:

* a link made on weak appearance evidence is flagged ``UNCERTAIN`` and the swap is withheld;
* two plausible identities for one face (similar people, crossing faces) are ``AMBIGUOUS``
  and withheld until the evidence separates them again;
* a track that disappears (occlusion, exit, scene cut) becomes ``LOST`` and is only
  ``REACQUIRED`` by a strict, margin-checked appearance match followed by a short probation.

Association is two-tier (ByteTrack style) and motion-aware: Kalman-predicted boxes (``motion.py``), IoU/GIoU spatial
gating, pose-aware ArcFace cosine against an EMA gallery plus per-yaw-bin prototypes, solved by Hungarian assignment. A
head turning into profile lowers the embedding similarity before the box moves, so while the yaw trajectory is steep or
changing the continuity cosine floor drops by ``turn_relief`` and the profile is learned as its own prototype. After the
whole video has been tracked, ``identity/consolidation.py`` merges the tracker IDs that belong to one person (the ``merge_*``
/ ``veto_*`` / ``artifact_*`` fields below) and numbers them canonically.

Every threshold is a documented heuristic on ArcFace cosine similarity. They are
recorded in the policy fingerprint so any change invalidates saved identity locks. The ``merge_*`` / ``veto_*`` values were
calibrated on the single-subject clips (``tools/verify_stage3 --calibrate``); the online association thresholds were not
calibrated against a labelled benchmark. See docs/IDENTITY_LOCK_STAGE1.md.
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
    gallery_rate: float = 0.1          # EMA rate of the identity gallery (strong, unambiguous matches only)
    bin_rate: float = 0.2              # EMA rate of the pose-binned prototypes
    # Head turns: a steep or sharply changing yaw lowers embedding similarity before it recovers, so for such frames the
    # continuity cosine floor drops by ``turn_relief`` as long as the box is spatially continuous with the track.
    turn_yaw: float = 40.0
    turn_step: float = 15.0
    turn_relief: float = 0.15
    # A track that is not detected stays spatially matchable (Kalman-coasted) for this many frames; afterwards it needs appearance.
    spatial_bridge_frames: int = 8
    # Frames a missed track stays alive in the active set before it is parked for appearance-only reacquisition.
    bridge_frames: int = 90
    # Global tracklet consolidation (identity/consolidation.py): which tracker IDs are one person. Calibrated with
    # ``tools.verify_stage3 --calibrate`` on the single-subject clips: pose-matched prototype cosine between *different*
    # people (different videos, pairs that are visibly the same person excluded) has p50 0.03 and p95 0.15-0.22, p99 <= 0.25
    # in every yaw bin; the same person scores 0.96 frontal and still 0.45 (median) against the frontal prototype in profile.
    merge_cosine: float = 0.45         # pose-matched prototype cosine that merges two tracklets on appearance alone (> 1.8x impostor p99)
    merge_weak_cosine: float = 0.20    # with strict spatial/temporal continuity and nobody else around (impostor p95)
    merge_turn_cosine: float = 0.05    # same, when one side meets the other in a steep turn (embedding collapses in profile)
    veto_cosine: float = 0.15          # two well supported frontal prototypes below this are two people, whatever the position
    continuity_gap_frames: int = 60    # largest hand-over gap bridged by continuity
    merge_gap_frames: int = 3600       # largest gap bridged by appearance alone
    min_identity_frames: int = 5       # fewer detections than this is noise, not a person
    artifact_containment: float = 0.5  # share of a box inside a larger face's box ...
    artifact_area_ratio: float = 0.35  # ... and no more than this fraction of its area: a part of that face
    seamless_gap_frames: int = 3       # two IDs meeting within this many frames of one shot ...
    seamless_iou: float = 0.6          # ... at a box overlapping at least this much are one object whatever the embedding says

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
        if not 0 < self.gallery_rate < 1 or not 0 < self.bin_rate < 1:
            raise ValueError("Gallery/bin rates must lie in (0, 1)")
        if not 0 < self.turn_yaw <= 90 or not 0 < self.turn_step <= 90 or not 0 <= self.turn_relief < self.continuity_cosine:
            raise ValueError("Turn thresholds out of bounds")
        if not 0 <= self.spatial_bridge_frames <= 600 or not 1 <= self.bridge_frames <= 120:
            raise ValueError("Bridge windows out of bounds")
        if not -1 < self.merge_turn_cosine <= self.merge_weak_cosine <= self.merge_cosine < 1 or not 0 < self.veto_cosine < 1:
            raise ValueError("Merge cosines must be ordered turn <= weak <= merge, veto within (0, 1)")
        if not 0 <= self.continuity_gap_frames <= self.merge_gap_frames <= 100000 or not 1 <= self.min_identity_frames <= 1000:
            raise ValueError("Merge windows out of bounds")
        if not 0 < self.artifact_containment <= 1 or not 0 < self.artifact_area_ratio <= 1:
            raise ValueError("Artifact thresholds must lie in (0, 1]")
        if not 0 <= self.seamless_gap_frames <= self.continuity_gap_frames or not 0.3 <= self.seamless_iou <= 1:
            raise ValueError("Seamless hand-over bounds out of range")

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
    # Pose-binned appearance prototypes: how this person looks at 30-degree yaw steps.
    bins: dict[int, np.ndarray] = field(default_factory=dict)
    # Kalman predictors (frame-number time), created by the tracker; ``clock`` is the frame number of the last update.
    motion: Any = None
    center: Any = None
    trajectory: Any = None
    clock: float = 0.0


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


YAW_BIN_DEGREES = 30.0


def yaw_bin(yaw: float) -> int:
    """Pose bin of a yaw angle: ..., -1 = about -30 deg, 0 = frontal, +1 = about +30 deg, +/-3 = profile."""
    return int(np.clip(round(float(yaw) / YAW_BIN_DEGREES), -3, 3))


def appearance(memory: TrackMemory, embedding: np.ndarray, yaw: float | None = None) -> float:
    """Best cosine against the stable gallery, its diverse anchors and the pose-binned prototypes (never the last frame).

    A pose-binned prototype only speaks for observations at (or next to) its own yaw: embeddings of *different* people
    resemble each other far more in profile than in frontal view, so a profile prototype must not vouch for a frontal face.
    ``yaw=None`` (no pose information) consults every bin."""
    best = float(np.dot(memory.gallery, embedding))
    for anchor in memory.anchors:
        best = max(best, float(np.dot(anchor, embedding)))
    wanted = None if yaw is None else yaw_bin(yaw)
    for key, prototype in memory.bins.items():
        if wanted is None or abs(key - wanted) <= 1:
            best = max(best, float(np.dot(prototype, embedding)))
    return best


def learn_bin(memory: TrackMemory, embedding: np.ndarray, yaw: float, rate: float) -> None:
    """Fold an observation into the prototype of its yaw bin (EMA; the first sample seeds the bin)."""
    key = yaw_bin(yaw)
    current = memory.bins.get(key)
    memory.bins[key] = unit(embedding.astype(np.float32)) if current is None else unit((1 - rate) * current + rate * embedding)


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
