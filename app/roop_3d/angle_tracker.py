"""Three-stage offline multi-angle trajectory scan for video renders.

Stage A accepts detector observations from the full target video. Stage B links faces
forwards and backwards, uses constant-velocity box prediction plus appearance/landmark
continuity, then bridges short detector gaps. Stage C applies bidirectional One-Euro
smoothing to the complete selected trajectory before a renderer starts at frame 0.
The module owns no detector/model session; callers inject analysis so preview, export,
and the verification harness share the same deterministic tracking contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from roop_3d.core.pose import as_pose3d
from roop_3d.identity.target_lock import TargetIdentityLock
from roop_3d.temporal import OneEuroFilter

MAX_BRIDGE_GAP = 8
TARGET_MATCH_SAME_ANGLE = 0.40
TARGET_MATCH_PROFILE_CONTINUITY = 0.22
RENDER_MODE_TO_ENGINE = {
    "standard_2d": "standard",
    "estimated_2_5d": "pose_3d",
    "full_3d_reconstructed": "reenactment_3d",
}


def normalize_render_mode(mode: str) -> str:
    """Translate the workstation's Stage 13 labels to the established engine names."""
    if mode in RENDER_MODE_TO_ENGINE:
        return RENDER_MODE_TO_ENGINE[mode]
    if mode in RENDER_MODE_TO_ENGINE.values():
        return mode
    raise ValueError(f"Unsupported render mode: {mode}")


def pose_match_threshold(pose_delta: float, *, yaw: float, spatial_iou: float) -> float:
    """Angle-adaptive identity gate; only strong spatial continuity relaxes profile matching."""
    if spatial_iou <= 0.55 or abs(yaw) < 65.0:
        return TARGET_MATCH_SAME_ANGLE
    fraction = float(np.clip(pose_delta / 90.0, 0.0, 1.0))
    return max(TARGET_MATCH_PROFILE_CONTINUITY,
               TARGET_MATCH_SAME_ANGLE - (TARGET_MATCH_SAME_ANGLE - TARGET_MATCH_PROFILE_CONTINUITY) * fraction)


def _unit(vector: Any) -> np.ndarray | None:
    if vector is None:
        return None
    value = np.asarray(vector, np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if value.shape != (512,) or not np.isfinite(value).all() or norm < 1e-8:
        return None
    return value / norm


def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, np.float32)
    if shape is not None and result.shape != shape:
        return None
    if result.size == 0 or not np.isfinite(result).all():
        return None
    return result.copy()


@dataclass(frozen=True)
class AngleDetection:
    bbox: np.ndarray
    landmarks5: np.ndarray
    pose: tuple[float, float, float]  # standard frame: pitch positive = looking up
    embedding: np.ndarray | None = None
    landmarks68_3d: np.ndarray | None = None
    landmarks106: np.ndarray | None = None
    confidence: float = 1.0
    frame_index: int = 0

    def __post_init__(self):
        box = _array(self.bbox, (4,))
        five = _array(self.landmarks5, (5, 2))
        if box is None or five is None or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Angle detections require a positive bbox and five finite landmarks")
        if not 0.0 <= float(self.confidence) <= 1.0 or self.frame_index < 0:
            raise ValueError("Invalid detection confidence or frame index")
        pose = as_pose3d(self.pose, frame="standard")
        object.__setattr__(self, "bbox", box)
        object.__setattr__(self, "landmarks5", five)
        object.__setattr__(self, "pose", pose.as_tuple())
        object.__setattr__(self, "embedding", _unit(self.embedding))
        object.__setattr__(self, "landmarks68_3d", _landmarks(self.landmarks68_3d, (68, 3)))
        object.__setattr__(self, "landmarks106", _landmarks(self.landmarks106, (106, 2)))


@dataclass(frozen=True)
class AngleTrackFrame:
    frame_index: int
    track_id: int
    bbox: np.ndarray
    landmarks5: np.ndarray
    pose: tuple[float, float, float]
    embedding: np.ndarray | None
    landmarks68_3d: np.ndarray | None
    landmarks106: np.ndarray | None
    alignment_matrix: np.ndarray
    detected: bool
    confidence: float
    identity_similarity: float | None = None
    identity_threshold: float | None = None


@dataclass(frozen=True)
class AngleScanResult:
    frames_seen: int
    detection_count: int
    track_count: int
    target_track_id: int | None
    target_frames: tuple[AngleTrackFrame | None, ...]
    forward_tracks: int
    backward_tracks: int
    bridged_frames: int
    render_modes: tuple[str, ...] = ("standard_2d", "estimated_2_5d", "full_3d_reconstructed")

    @property
    def locked_frames(self) -> int:
        return sum(item is not None for item in self.target_frames)


def _landmarks(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    result = _array(value, shape)
    return result


def optical_flow_landmarks_3d(previous_bgr: np.ndarray, current_bgr: np.ndarray,
                              landmarks68_3d: np.ndarray) -> np.ndarray:
    """Propagate observed 3D landmarks by 2D pyramidal flow while preserving depth.

    Inputs may be BGR or grayscale uint8 frames; the z coordinate is never changed.
    """
    previous, current = np.asarray(previous_bgr), np.asarray(current_bgr)
    points = _landmarks(landmarks68_3d, (68, 3))
    if (previous.dtype != np.uint8 or current.dtype != np.uint8 or previous.shape != current.shape
            or previous.ndim not in (2, 3) or previous.ndim == 3 and previous.shape[2] != 3 or points is None):
        raise ValueError("Optical flow requires equal grayscale/BGR frames and 68 finite 3D landmarks")
    before = previous if previous.ndim == 2 else cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    after = current if current.ndim == 2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    moved, status, _error = cv2.calcOpticalFlowPyrLK(
        before, after, points[:, :2].astype(np.float32).reshape(-1, 1, 2), None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if moved is None or status is None:
        raise ValueError("Optical flow could not track the 68 landmarks")
    valid = status.reshape(-1).astype(bool)
    result = points.copy()
    result[valid, :2] = moved.reshape(-1, 2)[valid]
    return result


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def _landmark_similarity(a: AngleDetection, b: AngleDetection) -> float:
    aw, ah = max(float(a.bbox[2] - a.bbox[0]), 1.0), max(float(a.bbox[3] - a.bbox[1]), 1.0)
    bw, bh = max(float(b.bbox[2] - b.bbox[0]), 1.0), max(float(b.bbox[3] - b.bbox[1]), 1.0)
    pa = (a.landmarks5 - a.bbox[:2]) / np.array((aw, ah), np.float32)
    pb = (b.landmarks5 - b.bbox[:2]) / np.array((bw, bh), np.float32)
    distance = float(np.linalg.norm(pa - pb, axis=1).mean())
    return float(np.exp(-3.0 * distance))


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    return float(np.clip(a @ b, -1.0, 1.0)) if a is not None and b is not None else 0.0


def _max_assignment(scores: np.ndarray) -> list[tuple[int, int]]:
    # Bounded to 32 faces per frame. Deterministic greedy matching preserves the strongest links.
    candidates = sorted(((float(scores[row, col]), row, col) for row in range(scores.shape[0])
                         for col in range(scores.shape[1]) if scores[row, col] > 0), reverse=True)
    rows: set[int] = set()
    columns: set[int] = set()
    pairs = []
    for score, row, column in candidates:
        if row not in rows and column not in columns:
            rows.add(row)
            columns.add(column)
            pairs.append((row, column))
    return pairs


@dataclass
class _Link:
    keys: list[tuple[int, int]]
    last: AngleDetection
    last_frame: int
    kalman: "_KalmanBox"


class _KalmanBox:
    """Constant-velocity Kalman state over bbox corners, used in each scan direction."""

    def __init__(self, bbox: np.ndarray, frame_index: int):
        self.state = np.r_[np.asarray(bbox, np.float64), np.zeros(4, np.float64)]
        self.covariance = np.diag((4.0, 4.0, 4.0, 4.0, 25.0, 25.0, 25.0, 25.0))
        self.frame_index = int(frame_index)

    def predict_to(self, frame_index: int) -> np.ndarray:
        dt = float(frame_index - self.frame_index)
        if dt == 0:
            return self.state[:4].astype(np.float32).copy()
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = np.eye(4) * dt
        q = max(abs(dt), 1.0)
        process = np.diag((q, q, q, q, 2 * q, 2 * q, 2 * q, 2 * q))
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        self.frame_index = int(frame_index)
        return self.state[:4].astype(np.float32).copy()

    def correct(self, bbox: np.ndarray) -> None:
        observation = np.asarray(bbox, np.float64)
        measurement = np.zeros((4, 8), np.float64)
        measurement[:, :4] = np.eye(4)
        noise = np.eye(4, dtype=np.float64) * 4.0
        innovation = observation - measurement @ self.state
        covariance = measurement @ self.covariance @ measurement.T + noise
        gain = self.covariance @ measurement.T @ np.linalg.inv(covariance)
        self.state += gain @ innovation
        self.covariance = (np.eye(8) - gain @ measurement) @ self.covariance


class AngleTrajectoryTracer:
    """Analyze, bidirectionally link, target-lock, bridge and smooth a video trajectory."""

    def __init__(self, *, max_gap: int = MAX_BRIDGE_GAP, min_cutoff: float = 0.15,
                 beta: float = 0.04, derivative_cutoff: float = 1.0):
        if not 0 <= max_gap <= 8:
            raise ValueError("max_gap must be within 0..8 frames")
        self.max_gap = max_gap
        self.filter_config = dict(min_cutoff=min_cutoff, beta=beta, derivative_cutoff=derivative_cutoff)

    def scan_video(self, frames: Iterable[Any], analyzer: Callable[[Any, int], Sequence[Any]], *,
                   target_lock: TargetIdentityLock | None = None, registry: Any = None,
                   reference_bank: Sequence[Any] | None = None, fps: float = 30.0,
                   gap_flow: Callable[[int, int, np.ndarray, np.ndarray], Mapping[int, np.ndarray]] | None = None) -> AngleScanResult:
        """Stage A adapter for decoded video frames; `analyzer` returns real per-face observations."""
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        scanned = []
        for index, frame in enumerate(frames):
            scanned.append([_coerce_detection(item, index) for item in analyzer(frame, index)])
        return self.scan(scanned, target_lock=target_lock, registry=registry, reference_bank=reference_bank,
                         fps=fps, gap_flow=gap_flow)

    def scan(self, detections_by_frame: Sequence[Sequence[AngleDetection]], *,
             target_lock: TargetIdentityLock | None = None, registry: Any = None,
             reference_bank: Sequence[Any] | None = None, fps: float = 30.0,
             gap_flow: Callable[[int, int, np.ndarray, np.ndarray], Mapping[int, np.ndarray]] | None = None) -> AngleScanResult:
        """Run all three stages on already extracted face observations."""
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        frames = [tuple(_coerce_detection(face, frame_index) for face in faces)
                  for frame_index, faces in enumerate(detections_by_frame)]
        forward = self._link_direction(frames, reverse=False)
        backward = self._link_direction(frames, reverse=True)
        tracks = self._merge_directions(forward, backward, frames)
        bank = list(reference_bank) if reference_bank is not None else self._bank_from_lock(target_lock, registry)
        target_id, chosen = self._choose_target(tracks, frames, target_lock, bank)
        target_frames = self._target_frames(chosen, frames, target_lock, bank, fps, target_id, gap_flow)
        bridged = sum(item is not None and not item.detected for item in target_frames)
        return AngleScanResult(len(frames), sum(map(len, frames)), len(tracks), target_id, tuple(target_frames),
                               len(forward), len(backward), bridged)

    @staticmethod
    def _bank_from_lock(lock: TargetIdentityLock | None, registry: Any) -> list[Any]:
        if lock is None or registry is None:
            return []
        if hasattr(registry, "registry"):
            registry = registry.registry(lock.media_path)
        cluster = getattr(registry, "clusters", {}).get(lock.cluster_id)
        return list(getattr(cluster, "bank", ())) if cluster is not None else []

    def _link_direction(self, frames: Sequence[Sequence[AngleDetection]], *, reverse: bool) -> list[list[tuple[int, int]]]:
        order = range(len(frames) - 1, -1, -1) if reverse else range(len(frames))
        links: dict[int, _Link] = {}
        next_id = 0
        for frame_index in order:
            detections = frames[frame_index]
            active_ids = [identity for identity, state in links.items()
                          if abs(frame_index - state.last_frame) <= self.max_gap + 1]
            predicted_boxes = {identity: links[identity].kalman.predict_to(frame_index) for identity in active_ids}
            scores = np.zeros((len(detections), len(active_ids)), np.float64)
            for row, face in enumerate(detections):
                for col, identity in enumerate(active_ids):
                    state = links[identity]
                    predicted = predicted_boxes[identity]
                    spatial = _iou(predicted, face.bbox)
                    appearance = _cosine(state.last.embedding, face.embedding)
                    geometry = _landmark_similarity(state.last, face)
                    if appearance >= 0.18 or spatial >= 0.08 or geometry >= 0.78:
                        # High-confidence detections get the first association pass, as in ByteTrack;
                        # the lower-confidence pass can only consume tracks left unmatched above.
                        scores[row, col] = 0.45 * spatial + 0.35 * max(0.0, appearance) + 0.20 * geometry
            high_rows = [row for row, face in enumerate(detections) if face.confidence >= 0.5]
            low_rows = [row for row, face in enumerate(detections) if face.confidence < 0.5]
            matches: dict[int, int] = {}
            used_columns: set[int] = set()
            if high_rows and active_ids:
                for row, column in _max_assignment(scores[high_rows]):
                    matches[high_rows[row]] = column
                    used_columns.add(column)
            remaining_columns = [column for column in range(len(active_ids)) if column not in used_columns]
            if low_rows and remaining_columns:
                for row, column in _max_assignment(scores[np.ix_(low_rows, remaining_columns)]):
                    matches[low_rows[row]] = remaining_columns[column]
            for row, face in enumerate(detections):
                if row in matches:
                    identity = active_ids[matches[row]]
                    state = links[identity]
                    state.kalman.correct(face.bbox)
                    state.keys.append((frame_index, row))
                    state.last, state.last_frame = face, frame_index
                else:
                    identity = next_id
                    next_id += 1
                    links[identity] = _Link([(frame_index, row)], face, frame_index,
                                            _KalmanBox(face.bbox, frame_index))
        return [state.keys for state in links.values()]

    @staticmethod
    def _merge_directions(forward, backward, frames) -> dict[int, dict[int, tuple[int, AngleDetection]]]:
        memberships = [set(keys) for keys in (*forward, *backward)]
        parents = list(range(len(memberships)))

        def find(value):
            while parents[value] != value:
                parents[value] = parents[parents[value]]
                value = parents[value]
            return value

        owner: dict[tuple[int, int], int] = {}
        for index, keys in enumerate(memberships):
            for key in keys:
                if key in owner:
                    a, b = find(index), find(owner[key])
                    parents[a] = b
                else:
                    owner[key] = index
        merged: dict[int, dict[int, tuple[int, AngleDetection]]] = {}
        for index, keys in enumerate((*forward, *backward)):
            root = find(index)
            result = merged.setdefault(root, {})
            for frame, detection_index in keys:
                detection = frames[frame][detection_index]
                previous = result.get(frame)
                if previous is None or detection.confidence > previous[1].confidence:
                    result[frame] = (detection_index, detection)
        # IDs are deterministic by the first observed frame and face index.
        return {new_id: merged[key] for new_id, key in enumerate(sorted(merged, key=lambda k: min(merged[k])))}

    def _choose_target(self, tracks, frames, lock, bank):
        if not tracks:
            return None, {}
        if lock is None:
            selected = max(tracks.items(), key=lambda item: (len(item[1]), item[0]))
            return selected[0], selected[1]
        anchor_frame = min(max(int(lock.anchor_frame_index), 0), max(len(frames) - 1, 0))
        anchor_box = np.asarray(lock.anchor_bbox, np.float32)
        best = None
        for track_id, points in tracks.items():
            if anchor_frame in points:
                detection = points[anchor_frame][1]
                candidate = (_iou(anchor_box, detection.bbox), track_id, points)
                if best is None or candidate[0] > best[0]:
                    best = candidate
        if best is None:
            # Small edit offsets are common in a lock made from a nearby keyframe.
            for track_id, points in tracks.items():
                frame, detection = min(points.items(), key=lambda item: abs(item[0] - anchor_frame))
                if abs(frame - anchor_frame) <= self.max_gap + 1:
                    candidate = (_iou(anchor_box, detection.bbox), track_id, points)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
        if best is None:
            return None, {}
        return best[1], best[2]

    def _target_frames(self, points, frames, lock, bank, fps, track_id, gap_flow):
        if not points:
            return [None] * len(frames)
        observed: dict[int, AngleTrackFrame] = {}
        previous_box = np.asarray(lock.anchor_bbox, np.float32) if lock is not None else None
        for frame_index, (detection_index, face) in sorted(points.items()):
            spatial = _iou(previous_box, face.bbox) if previous_box is not None else 0.0
            similarity, threshold = _target_score(face, bank, spatial)
            is_anchor = lock is not None and frame_index == lock.anchor_frame_index
            if bank and similarity is not None and similarity < threshold and not is_anchor:
                # A linked profile may continue through a pose change only with the configured profile/IoU gate.
                if not (spatial > 0.55 and abs(face.pose[0]) > 65 and similarity >= TARGET_MATCH_PROFILE_CONTINUITY):
                    continue
            observed[frame_index] = _make_frame(frame_index, track_id, face, detected=True,
                                                similarity=similarity, threshold=threshold)
            previous_box = face.bbox
        output: list[AngleTrackFrame | None] = [None] * len(frames)
        for index, frame in observed.items():
            output[index] = frame
        for left, right in zip(sorted(observed), sorted(observed)[1:]):
            gap = right - left - 1
            if gap <= 0 or gap > self.max_gap:
                continue
            a, b = observed[left], observed[right]
            refinements = {}
            if gap_flow is not None and a.landmarks68_3d is not None and b.landmarks68_3d is not None:
                try:
                    refinements = gap_flow(left, right, a.landmarks68_3d, b.landmarks68_3d)
                except (ValueError, cv2.error, TypeError):
                    refinements = {}
            for step in range(1, gap + 1):
                amount = step / (gap + 1)
                frame = _interpolate_frame(a, b, left + step, amount)
                if gap_flow is not None and a.landmarks68_3d is not None and b.landmarks68_3d is not None:
                    points68 = refinements.get(left + step) if isinstance(refinements, Mapping) else None
                    points68 = _landmarks(points68, (68, 3))
                    if points68 is not None:
                        detection = AngleDetection(frame.bbox, frame.landmarks5, frame.pose, frame.embedding,
                            points68, frame.landmarks106, frame.confidence, frame.frame_index)
                        frame = _make_frame(frame.frame_index, frame.track_id, detection, detected=False,
                            similarity=frame.identity_similarity, threshold=frame.identity_threshold)
                output[left + step] = frame
        # Stage C: smooth each contiguous target segment in both time directions.
        start = 0
        while start < len(output):
            while start < len(output) and output[start] is None:
                start += 1
            end = start
            while end < len(output) and output[end] is not None:
                end += 1
            if end > start:
                _smooth_segment(output, start, end, fps, self.filter_config)
            start = max(end, start + 1)
        return output


def face_tracker_analyzer(tracker: Any, landmark_engine: Any = None, *, include203: bool = False):
    """Adapt the repository's detector/recognizer tracker for Stage A extraction.

    The tracker supplies native-frame boxes, five points, 106 points, pose and 512-D
    embeddings. An optional local ``LandmarkEngine`` adds observed 68-point 3D values.
    No model is created here; pass the session-owned objects to keep model lifetime bounded.
    """
    def analyze(image: np.ndarray, frame_index: int) -> list[AngleDetection]:
        result = []
        for face in tracker.observe(image, include203=include203):
            dense68 = landmark_engine.landmarks68(image, face.box) if landmark_engine is not None else None
            pose = as_pose3d(face.pose)  # FaceTracker stores its legacy pitch tuple.
            result.append(AngleDetection(face.box, face.landmarks5, pose.as_tuple(), face.embedding,
                                         dense68, face.landmarks106, 1.0, frame_index))
        return result
    return analyze


def _target_score(face: AngleDetection, bank: Sequence[Any], spatial_iou: float) -> tuple[float | None, float]:
    if not bank or face.embedding is None:
        return None, TARGET_MATCH_SAME_ANGLE
    pose = face.pose
    nearest = min(bank, key=lambda item: _pose_delta(pose, item))
    delta = _pose_delta(pose, nearest)
    similarity = _cosine(face.embedding, _unit(getattr(nearest, "embedding", None)
                                                if not isinstance(nearest, Mapping)
                                                else nearest.get("embedding")))
    threshold = pose_match_threshold(delta, yaw=pose[0], spatial_iou=spatial_iou)
    return similarity, threshold


def _pose_delta(pose, reference) -> float:
    if isinstance(reference, Mapping):
        yaw, pitch = float(reference.get("yaw", 0)), float(reference.get("pitch", 0))
    else:
        yaw, pitch = float(getattr(reference, "yaw", 0)), float(getattr(reference, "pitch", 0))
    return float(np.hypot(float(pose[0]) - yaw, float(pose[1]) - pitch))


def _alignment(landmarks5: np.ndarray) -> np.ndarray:
    from roop_3d.engine.volumetric.preprocess import similarity
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    reference = ARCFACE_REFERENCE_5PTS.astype(np.float64).copy()
    reference[:, 0] += 8.0
    return similarity(landmarks5, reference)


def _make_frame(frame_index, track_id, detection, *, detected, similarity=None, threshold=None):
    return AngleTrackFrame(frame_index, track_id, detection.bbox.copy(), detection.landmarks5.copy(), detection.pose,
        None if detection.embedding is None else detection.embedding.copy(),
        None if detection.landmarks68_3d is None else detection.landmarks68_3d.copy(),
        None if detection.landmarks106 is None else detection.landmarks106.copy(),
        _alignment(detection.landmarks5), detected, float(detection.confidence), similarity, threshold)


def _interpolate_frame(a: AngleTrackFrame, b: AngleTrackFrame, frame_index: int, amount: float) -> AngleTrackFrame:
    interpolate = lambda x, y: (1.0 - amount) * x + amount * y
    pose_a, pose_b = np.asarray(a.pose), np.asarray(b.pose)
    pose_b = pose_a + (pose_b - pose_a + 180.0) % 360.0 - 180.0
    pose = tuple(float(v) for v in interpolate(pose_a, pose_b))
    if a.embedding is not None and b.embedding is not None:
        embedding = _spherical_lerp(a.embedding, b.embedding, amount)
    else:
        embedding = a.embedding if a.embedding is not None else b.embedding
    def optional(left, right):
        if left is None:
            return None if right is None else right.copy()
        if right is None or left.shape != right.shape:
            return left.copy()
        return interpolate(left, right).astype(np.float32)
    five = optional(a.landmarks5, b.landmarks5)
    bbox = optional(a.bbox, b.bbox)
    detection = AngleDetection(bbox, five, pose, embedding, optional(a.landmarks68_3d, b.landmarks68_3d),
                               optional(a.landmarks106, b.landmarks106), min(a.confidence, b.confidence) * 0.8,
                               frame_index)
    return _make_frame(frame_index, a.track_id, detection, detected=False,
                       similarity=min(v for v in (a.identity_similarity, b.identity_similarity) if v is not None)
                       if a.identity_similarity is not None or b.identity_similarity is not None else None,
                       threshold=max(v for v in (a.identity_threshold, b.identity_threshold) if v is not None)
                       if a.identity_threshold is not None or b.identity_threshold is not None else None)


def _spherical_lerp(a, b, amount):
    a, b = _unit(a), _unit(b)
    cosine = float(np.clip(a @ b, -1, 1))
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        result = (1 - amount) * a + amount * b
    else:
        result = (np.sin((1 - amount) * angle) * a + np.sin(amount * angle) * b) / np.sin(angle)
    return _unit(result)


def _smooth_values(values: np.ndarray, fps: float, config: dict[str, float]) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if len(values) < 3:
        return values.copy()
    dt = 1.0 / fps
    forward_filter = OneEuroFilter(**config)
    forward = np.stack([forward_filter.update(value, index * dt) for index, value in enumerate(values)])
    reverse_filter = OneEuroFilter(**config)
    backward = np.stack([reverse_filter.update(value, index * dt) for index, value in enumerate(values[::-1])])[::-1]
    return (forward + backward) * 0.5


def _smooth_segment(output: list[AngleTrackFrame | None], start: int, end: int, fps: float,
                    config: dict[str, float]) -> None:
    items = [output[index] for index in range(start, end)]
    boxes = _smooth_values(np.stack([item.bbox for item in items]), fps, config)
    five = _smooth_values(np.stack([item.landmarks5 for item in items]), fps, config)
    poses = np.asarray([item.pose for item in items], np.float64)
    poses[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(poses[:, 2])))
    poses = _smooth_values(poses, fps, config)
    dense68 = _smooth_optional([item.landmarks68_3d for item in items], fps, config)
    dense106 = _smooth_optional([item.landmarks106 for item in items], fps, config)
    for offset, item in enumerate(items):
        detection = AngleDetection(boxes[offset], five[offset], tuple(poses[offset]), item.embedding,
            None if dense68 is None else dense68[offset], None if dense106 is None else dense106[offset],
            item.confidence, item.frame_index)
        output[start + offset] = _make_frame(item.frame_index, item.track_id, detection, detected=item.detected,
            similarity=item.identity_similarity, threshold=item.identity_threshold)


def _smooth_optional(values, fps, config):
    if any(value is None for value in values):
        return None
    shapes = {value.shape for value in values}
    if len(shapes) != 1:
        return None
    stacked = np.stack(values)
    shape = stacked.shape
    return _smooth_values(stacked.reshape(len(stacked), -1), fps, config).reshape(shape).astype(np.float32)


def _coerce_detection(value: Any, frame_index: int) -> AngleDetection:
    if isinstance(value, AngleDetection):
        if value.frame_index == frame_index:
            return value
        return AngleDetection(value.bbox, value.landmarks5, value.pose, value.embedding, value.landmarks68_3d,
                              value.landmarks106, value.confidence, frame_index)
    bbox = getattr(value, "bbox", getattr(value, "box", None))
    points5 = getattr(value, "landmarks5", getattr(value, "kps_5", None))
    pose = getattr(value, "pose", (0, 0, 0))
    pose = as_pose3d(pose)  # FaceTracker observations use the legacy tuple frame.
    return AngleDetection(bbox, points5, pose.as_tuple(), getattr(value, "embedding", None),
                          getattr(value, "landmarks68_3d", getattr(value, "landmarks68", None)),
                          getattr(value, "landmarks106", None), float(getattr(value, "confidence", 1.0)), frame_index)
