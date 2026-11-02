"""Bounded multi-face observations and deterministic IoU/identity assignment."""
from __future__ import annotations

from dataclasses import dataclass, replace
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.detection import track_policy as tp
from roop_3d.engine.detection.dense_landmarks import DenseLandmarks, geometric_pose
from roop_3d.engine.detection.track_policy import TrackPolicy
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
    # Identity-lock fields. The legacy tracker leaves the defaults (always swappable).
    state: str = tp.TRACKED
    confidence: float = 1.0
    swap_allowed: bool = True
    reason: str | None = None


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
                 threshold: float = 0.5, config: RuntimeConfig = RuntimeConfig(), detector: Detector | None = None, stabilize: bool = True,
                 policy: TrackPolicy | None = None):
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
        # Opt-in identity-lock policy. None keeps the legacy association byte-for-byte.
        self.policy = policy
        self._mem: dict[int, tp.TrackMemory] = {}
        self._lost: dict[int, tp.TrackMemory] = {}
        self.events: list[tp.TrackEvent] = []
        self.events_dropped = 0
        self._policy_frame = -1
        self._small_gray: np.ndarray | None = None
        self._flow_frames: tuple[np.ndarray, np.ndarray, float] | None = None

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
        cut = self.previous_thumbnail is not None and self._cut(self.previous_thumbnail, thumb)
        self.scene_cut = cut
        if cut:
            self.mark_scene_cut()
        self.previous_thumbnail = thumb
        if self.policy is not None:
            self._update_flow_frames(image, cut)
        return self.assign(self.observe(image, include203=include203))

    def mark_scene_cut(self) -> None:
        """Explicit shot boundary. Legacy: forget every track. Policy: park them for strict reacquisition."""
        self.scene_cut = True
        self.stabilizer.reset()
        self.scene_id += 1
        if self.policy is None:
            self._tracks.clear()
            return
        self._flow_frames = None
        for identity in sorted(self._mem):
            memory = self._mem.pop(identity)
            memory.parked_by_cut = True
            self._park(memory, tp.SCENE_CUT)

    def _update_flow_frames(self, image: np.ndarray, cut: bool) -> None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 480.0 / max(gray.shape))
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
        previous = self._small_gray
        self._flow_frames = ((previous, small, scale)
                             if previous is not None and not cut and previous.shape == small.shape else None)
        self._small_gray = small

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
        if self.policy is not None:
            return self._assign_policy(observations)
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
        return self._stabilize(output, updated)

    def _stabilize(self, output: list[TrackedFace], retained) -> tuple[TrackedFace, ...]:
        if not self.stabilize:
            return tuple(output)
        self.stabilizer.retain(retained)
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

    # ------------------------------------------------------------------ identity-lock policy
    def _event(self, memory: tp.TrackMemory, state: str, reason: str) -> None:
        if len(self.events) >= 20000:
            self.events_dropped += 1
            return
        self.events.append(tp.TrackEvent(self._policy_frame, memory.track_id, state, reason, self.scene_id))

    def _park(self, memory: tp.TrackMemory, reason: str) -> None:
        """Retire a track to the bounded LOST store; it can only come back through reacquisition."""
        policy = self.policy
        memory.state, memory.lost_at, memory.hold = tp.LOST, self._policy_frame, 0
        self._lost[memory.track_id] = memory
        if reason != tp.NOT_DETECTED:       # the first missed frame already logged NOT_DETECTED
            self._event(memory, tp.LOST, reason)
        for identity in [key for key, value in self._lost.items()
                         if self._policy_frame - (value.lost_at or 0) > policy.lost_retention_frames]:
            del self._lost[identity]
        while len(self._lost) > policy.lost_limit:
            del self._lost[min(self._lost, key=lambda key: (self._lost[key].lost_at or 0, key))]

    def _propagate_missed(self) -> None:
        """Carry the box of tracks that were not detected this frame forward with optical flow."""
        for memory in self._mem.values():
            if memory.missed == 0:
                continue
            moved = None
            if self._flow_frames is not None:
                previous, current, scale = self._flow_frames
                # On the first miss the last seen box is the previous-frame position.
                base = memory.observation.box if memory.missed == 1 else memory.predicted_box
                moved = tp.propagate_box(previous, current, np.asarray(base, np.float32) * scale)
            memory.predicted_box = (moved[0] / scale).astype(np.float32) if moved is not None \
                else (memory.predicted_box + memory.velocity * 0.5).astype(np.float32)

    def track_states(self) -> dict[int, str]:
        """State of every track the policy still remembers (active or LOST)."""
        return {**{identity: memory.state for identity, memory in self._lost.items()},
                **{identity: memory.state for identity, memory in self._mem.items()}}

    def _ambiguous(self, row: int, column: int, observations, ids, cosine, overlap) -> bool:
        """True when another face or another track is a plausible, near-equal identity for this link."""
        policy, base = self.policy, cosine[row, column]
        for other in range(len(observations)):
            if other == row or cosine[other, column] < policy.weak_cosine:
                continue
            plausible = (overlap[other, column] >= 0.1
                         or iou(observations[other].box, observations[row].box) >= 0.1)
            if plausible and base - cosine[other, column] < policy.ambiguity_margin:
                return True
        for other in range(len(ids)):
            if other == column or cosine[row, other] < policy.weak_cosine:
                continue
            if overlap[row, other] >= 0.1 and base - cosine[row, other] < policy.ambiguity_margin:
                return True
        return False

    def _discriminated(self, row: int, column: int, cosine: np.ndarray) -> bool:
        """This face resembles its track clearly more than any other face/track (regardless of position)."""
        rivals = [cosine[other, column] for other in range(cosine.shape[0]) if other != row]
        rivals += [cosine[row, other] for other in range(cosine.shape[1]) if other != column]
        return not rivals or cosine[row, column] - max(rivals) >= self.policy.ambiguity_margin

    def _reacquire(self, rows: list[int], observations,
                   candidates: list[tp.TrackMemory]) -> dict[int, tp.TrackMemory]:
        """Strict appearance-only recovery with margins against every other face and track."""
        policy = self.policy
        if not rows or not candidates:
            return {}
        similarity = np.array([[tp.appearance(track, observations[row].embedding) for track in candidates]
                               for row in rows])
        gate = np.where(similarity >= policy.reacquire_cosine, similarity, -1.0)
        accepted: dict[int, tp.TrackMemory] = {}
        for local_row, column in optimal_assignment(gate):
            row, track = rows[local_row], candidates[column]
            best = float(similarity[local_row, column])
            rivals = [float(similarity[local_row, other]) for other in range(len(candidates)) if other != column]
            rivals += [float(tp.appearance(track, observations[other].embedding))
                       for other in range(len(observations)) if other != row]
            # A currently visible track that looks equally like this face also makes the link unsafe.
            rivals += [float(tp.appearance(active, observations[row].embedding))
                       for active in self._mem.values() if active.track_id != track.track_id and active.missed == 0]
            if not rivals or best - max(rivals) >= policy.reacquire_margin:
                accepted[row] = track
        return accepted

    def _assign_policy(self, observations: list[FaceObservation]) -> tuple[TrackedFace, ...]:
        policy = self.policy
        frame = self._policy_frame = self._policy_frame + 1
        ids = sorted(self._mem)
        rows, columns = len(observations), len(ids)
        cosine = np.zeros((rows, columns))
        overlap = np.zeros((rows, columns))
        scores = np.full((rows, columns), -1.0)
        for row, face in enumerate(observations):
            tensor(face.embedding, (512,))
            for column, identity in enumerate(ids):
                memory = self._mem[identity]
                cosine[row, column] = similarity = tp.appearance(memory, face.embedding)
                overlap[row, column] = shared = iou(face.box, memory.predicted_box)
                if (similarity >= policy.match_cosine or similarity >= policy.weak_cosine and shared >= 0.4
                        or shared >= policy.continuity_iou and memory.missed <= 2):
                    layout = tp.landmark_similarity(memory.observation.landmarks5, memory.observation.box,
                                                    face.landmarks5, face.box)
                    scores[row, column] = max(0.6 * similarity + 0.25 * shared + 0.15 * layout
                                              - 0.002 * memory.missed, 0.01)
        matches = dict(optimal_assignment(scores))
        results: dict[int, tuple[int, str, float, bool, str | None, tp.TrackMemory]] = {}
        for row, column in matches.items():
            face, memory = observations[row], self._mem[ids[column]]
            similarity, shared = float(cosine[row, column]), float(overlap[row, column])
            score = tp.confidence(similarity, shared, policy)
            previous_state = memory.state
            ambiguous = self._ambiguous(row, column, observations, ids, cosine, overlap)
            if not ambiguous and memory.needs_discrimination:
                ambiguous = not self._discriminated(row, column, cosine)
            if ambiguous:
                memory.needs_discrimination = True
            elif memory.needs_discrimination:
                memory.needs_discrimination = False
            clean = False
            if ambiguous:
                state, reason = tp.UNCERTAIN, tp.AMBIGUOUS
            elif similarity >= policy.match_cosine:
                clean = True
            elif (similarity >= policy.continuity_cosine and shared >= policy.continuity_iou
                    and previous_state == tp.TRACKED and memory.hold == 0):
                clean = True        # profile/angle change: spatial continuity keeps an established lock
            else:
                state, reason = tp.UNCERTAIN, tp.APPEARANCE_LOW
            allowed = False
            if clean:
                if previous_state == tp.LOST:
                    memory.hold, memory.state = max(memory.hold, policy.recovery_frames), tp.REACQUIRED
                if memory.hold > 0:
                    memory.hold -= 1
                    state, reason = ((tp.REACQUIRED, tp.REACQUIRING) if memory.state == tp.REACQUIRED
                                     else (tp.UNCERTAIN, tp.RECOVERING))
                elif score < policy.min_confidence:
                    state, reason = tp.UNCERTAIN, "LOW_CONFIDENCE"
                    memory.uncertain_run += 1
                else:
                    state, reason, allowed = tp.TRACKED, None, True
                    memory.uncertain_run = 0
            else:
                memory.hold, memory.uncertain_run = policy.recovery_frames, memory.uncertain_run + 1
            if allowed and similarity >= policy.strong_cosine and not ambiguous:
                memory.gallery = tp.unit((1 - policy.gallery_rate) * memory.gallery + policy.gallery_rate * face.embedding)
                if (len(memory.anchors) < policy.anchor_limit
                        and max(float(np.dot(anchor, face.embedding)) for anchor in memory.anchors) < 0.85):
                    memory.anchors.append(face.embedding.astype(np.float32))
            results[row] = (ids[column], state, score, allowed, reason, memory)
        matched_columns = {ids[column] for column in matches.values()}
        pool = [self._mem[identity] for identity in ids if identity not in matched_columns] + list(self._lost.values())
        accepted = self._reacquire([row for row in range(rows) if row not in matches], observations, pool)
        output: list[TrackedFace] = []
        new_mem: dict[int, tp.TrackMemory] = {}
        for row, face in enumerate(observations):
            if row in results:
                identity, state, score, allowed, reason, memory = results[row]
                delta = (face.box - memory.observation.box) / (memory.missed + 1)
                memory.velocity = (0.5 * memory.velocity + 0.5 * delta).astype(np.float32)
                memory.age += 1
            elif row in accepted:
                memory = accepted[row]
                identity, state, allowed, reason = memory.track_id, tp.REACQUIRED, False, tp.REACQUIRING
                score = tp.confidence(tp.appearance(memory, face.embedding), 0.0, policy)
                memory.hold, memory.uncertain_run, memory.parked_by_cut = policy.reacquire_frames - 1, 0, False
                memory.velocity, memory.lost_at, memory.age = np.zeros(4, np.float32), None, memory.age + 1
                self._lost.pop(identity, None)
                self.stabilizer.tracks = {key: value for key, value in self.stabilizer.tracks.items() if key[0] != identity}
            else:
                identity = self.next_id
                self.next_id += 1
                memory = tp.TrackMemory(identity, face, tp.unit(face.embedding.astype(np.float32)),
                                        [face.embedding.astype(np.float32)], face.box.copy())
                state, score, allowed, reason = tp.TRACKED, 1.0, True, None
            if memory.state != state:
                self._event(memory, state, reason or "")
            memory.state, memory.observation, memory.missed = state, face, 0
            memory.predicted_box = (face.box + memory.velocity).astype(np.float32)
            memory.last_frame, memory.scene_id = frame, self.scene_id
            new_mem[identity] = memory
            output.append(TrackedFace(identity, face, memory.age, self.scene_id, state, float(score), allowed, reason))
        for identity in ids:
            if identity in new_mem:
                continue
            memory = self._mem[identity]
            memory.missed += 1
            if memory.state != tp.LOST:
                self._event(memory, tp.LOST, tp.NOT_DETECTED)
            memory.state, memory.hold = tp.LOST, 0
            new_mem[identity] = memory
        for identity in sorted(new_mem):
            memory = new_mem[identity]
            expired = memory.missed > self.max_missed or memory.uncertain_run > policy.uncertain_limit
            if expired:
                del new_mem[identity]
                self._park(memory, tp.UNCERTAIN_TIMEOUT if memory.uncertain_run > policy.uncertain_limit else tp.NOT_DETECTED)
        while len(new_mem) > self.max_tracks:
            victim = max(new_mem, key=lambda key: (new_mem[key].missed, -new_mem[key].last_frame, key))
            self._park(new_mem.pop(victim), tp.EXPIRED)
        self._mem = new_mem
        self._propagate_missed()
        return self._stabilize(output, self._mem)

    @property
    def active_ids(self) -> frozenset[int]:
        return frozenset(self._mem if self.policy is not None else self._tracks)

    def release(self) -> None:
        self._tracks.clear()
        self._mem.clear()
        self._lost.clear()
        self._small_gray = self._flow_frames = None
        self.stabilizer.reset()
        self.previous_thumbnail = None
        if self._owns_detector:
            self.detector._session = None
        self.recognizer._session = None
        self.dense.release()
        self.released = True
