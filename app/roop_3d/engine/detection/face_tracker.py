"""Bounded multi-face observations and deterministic IoU/identity assignment."""
from __future__ import annotations

from dataclasses import dataclass, replace
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.detection.dense_landmarks import DenseLandmarks, geometric_pose
from roop_3d.engine.runtime import RuntimeConfig, image_bgr, tensor, session_factory
from roop_3d.faceset.recognizer import Recognizer
from roop_3d.media.detector import Detector, iou
from roop_3d.models.catalog import ModelCatalog


@dataclass(frozen=True)
class FaceObservation:
    box: np.ndarray
    landmarks5: np.ndarray
    landmarks106: np.ndarray
    embedding: np.ndarray
    pose: tuple[float, float, float]
    landmarks203: np.ndarray | None = None


@dataclass(frozen=True)
class TrackedFace:
    track_id: int
    observation: FaceObservation
    age: int
    scene_id: int


def optimal_assignment(scores: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-weight assignment with zero-weight dummy rows/columns (Hungarian)."""
    rows, columns = scores.shape
    # Give every real row and column its own unmatched option, even for square
    # matrices. Otherwise a low-confidence match can displace a stronger one.
    size = rows + columns
    if not size:
        return []
    costs = np.zeros((size, size), np.float64)
    costs[:rows, :columns] = -scores
    row_potential = np.zeros(size + 1)
    col_potential = np.zeros(size + 1)
    owner = np.zeros(size + 1, np.int32)
    predecessor = np.zeros(size + 1, np.int32)
    for row in range(1, size + 1):
        owner[0], column = row, 0
        minimum = np.full(size + 1, np.inf)
        visited = np.zeros(size + 1, bool)
        while True:
            visited[column] = True
            current_row = owner[column]
            delta, next_column = np.inf, 0
            for candidate in range(1, size + 1):
                if visited[candidate]:
                    continue
                reduced = costs[current_row - 1, candidate - 1] - row_potential[current_row] - col_potential[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate], predecessor[candidate] = reduced, column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(size + 1):
                if visited[candidate]:
                    row_potential[owner[candidate]] += delta
                    col_potential[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            previous = predecessor[column]
            owner[column] = owner[previous]
            column = previous
    return [(int(owner[j] - 1), j - 1) for j in range(1, size + 1)
            if 0 < owner[j] <= rows and j <= columns and scores[owner[j] - 1, j - 1] > 0]


class FaceTracker:
    def __init__(self, catalog: ModelCatalog | None = None, *, max_tracks: int = 32, max_missed: int = 15,
                 threshold: float = 0.5, config: RuntimeConfig = RuntimeConfig(), detector: Detector | None = None, stabilize: bool = True):
        if not 1 <= max_tracks <= 32 or not 0 <= max_missed <= 120 or not 0.1 <= threshold <= 0.99:
            raise ValueError("Invalid tracker bounds")
        self.catalog = catalog or ModelCatalog()
        factory = session_factory(self.catalog, config)
        self.detector, self.recognizer = detector or Detector(self.catalog, factory), Recognizer(self.catalog, factory)
        self._owns_detector = detector is None
        self.dense = DenseLandmarks(self.catalog, config)
        self.max_tracks, self.max_missed, self.threshold = max_tracks, max_missed, threshold
        self._tracks: dict[int, tuple[FaceObservation, int, int]] = {}
        from roop_3d.temporal_stabilizer import TemporalStabilizer
        self.stabilizer = TemporalStabilizer()
        self.stabilize = stabilize
        self.frame_index = 0
        self.next_id = self.scene_id = 0
        self.previous_thumbnail: np.ndarray | None = None
        self.scene_cut = False
        self.released = False

    def observe(self, image: np.ndarray, *, include203: bool = False) -> list[FaceObservation]:
        image_bgr(image)
        detections, _ = self.detector.detect(image, self.threshold)
        observations = []
        for box, points in detections[:self.max_tracks]:
            points106 = self.dense.infer106(image, box)
            embedding = self.recognizer.embed(image, points)
            tensor(embedding, (512,))
            if abs(float(np.linalg.norm(embedding)) - 1) > 1e-4:
                raise DomainError("IDENTITY_EMBEDDING_INVALID", "Recognizer did not produce a unit embedding", 422)
            points203 = self.dense.infer203(image, points106) if include203 else None
            observations.append(FaceObservation(box.astype(np.float32), points.astype(np.float32), points106,
                                                embedding, geometric_pose(points), points203))
        return observations

    def process(self, image: np.ndarray, *, include203: bool = False) -> tuple[TrackedFace, ...]:
        if self.released:
            raise DomainError("SESSION_RELEASED", "Tracker has been released", 409)
        image_bgr(image)
        thumb = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
        self.scene_cut = self.previous_thumbnail is not None and self._cut(self.previous_thumbnail, thumb)
        if self.scene_cut:
            self._tracks.clear()
            self.stabilizer.reset()
            self.scene_id += 1
        self.previous_thumbnail = thumb
        return self.assign(self.observe(image, include203=include203))

    @staticmethod
    def _cut(previous: np.ndarray, current: np.ndarray) -> bool:
        difference = np.abs(previous.astype(np.float32) - current.astype(np.float32)).mean() / 255
        histograms = [cv2.calcHist([cv2.cvtColor(img, cv2.COLOR_BGR2HSV)], [0, 2], None, [16, 16], [0, 180, 0, 256])
                      for img in (previous, current)]
        distance = cv2.compareHist(*histograms, cv2.HISTCMP_BHATTACHARYYA)
        return bool(difference > 0.45 or difference > 0.18 and distance > 0.7)

    def assign(self, observations: list[FaceObservation]) -> tuple[TrackedFace, ...]:
        """Association is independently testable on real or analytic observations."""
        if self.released:
            raise DomainError("SESSION_RELEASED", "Tracker has been released", 409)
        if len(observations) > self.max_tracks:
            raise DomainError("TRACK_BUDGET", "Observation count exceeds track budget", 422)
        ids = sorted(self._tracks)
        scores = np.full((len(observations), len(ids)), -1.0)
        for row, face in enumerate(observations):
            tensor(face.embedding, (512,))
            for column, identity in enumerate(ids):
                previous, missed, _ = self._tracks[identity]
                cosine = float(np.dot(face.embedding, previous.embedding))
                overlap = iou(face.box, previous.box)
                if cosine >= 0.35 or cosine >= 0.15 and overlap >= 0.4:
                    scores[row, column] = 0.8 * cosine + 0.2 * overlap - missed * 0.002
        matches = dict(optimal_assignment(scores))
        updated: dict[int, tuple[FaceObservation, int, int]] = {}
        output = []
        for row, face in enumerate(observations):
            if row in matches:
                identity = ids[matches[row]]
                age = self._tracks[identity][2] + 1
            else:
                identity, age = self.next_id, 1
                self.next_id += 1
            updated[identity] = (face, 0, age)
            output.append(TrackedFace(identity, face, age, self.scene_id))
        for identity in ids:
            previous, missed, age = self._tracks[identity]
            if identity not in updated and missed < self.max_missed and len(updated) < self.max_tracks:
                updated[identity] = (previous, missed + 1, age)
        self._tracks = updated
        if not self.stabilize:
            return tuple(output)
        self.stabilizer.retain(updated)
        self.frame_index += 1
        timestamp = getattr(self, "timestamp_seconds", self.frame_index / 30.0)
        filtered = []
        for item in output:
            face = item.observation
            points = self.stabilizer.filter(item.track_id, 'landmarks5', face.landmarks5, timestamp)
            # Unwrap Euler angles against the last raw sample before filtering.
            state = self.stabilizer.tracks.get((item.track_id, 'pose'))
            angles = np.asarray(face.pose)
            if state is not None:
                angles = state.raw + (angles - state.raw + 180) % 360 - 180
            pose = self.stabilizer.filter(item.track_id, 'pose', angles, timestamp)
            filtered.append(replace(item, observation=replace(face, landmarks5=points.astype(np.float32), pose=tuple(pose))))
        return tuple(filtered)

    @property
    def active_ids(self) -> frozenset[int]:
        return frozenset(self._tracks)

    def release(self) -> None:
        self._tracks.clear()
        self.stabilizer.reset()
        self.previous_thumbnail = None
        if self._owns_detector:
            self.detector._session = None
        self.recognizer._session = None
        self.dense.release()
        self.released = True
