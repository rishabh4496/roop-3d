"""Global tracklet consolidation: one person, one canonical target ID for the whole video.

The online tracker (``FaceTracker`` with a :class:`TrackPolicy`) can still end a track and start another when a head
turns away, is occluded for longer than its bridge window, or the shot changes. This pass runs once over the finished
tracklets and decides, with the whole timeline in view, which of them are the same person.

Steps
-----
1. **Artifacts.** A tracklet whose boxes sit inside a much larger co-occurring face (an ear, a jaw, a shoulder) is a part of
   that face, not a person; a tracklet with fewer than ``min_identity_frames`` detections is noise. Neither gets an ID.
2. **Links.** Every pair of surviving tracklets is scored. Two tracklets that are ever seen *in the same frame* are different
   people (cannot-link). Two tracklets whose frontal prototypes disagree below ``veto_cosine`` are different people. Otherwise
   a link needs one of

   * appearance: pose-matched ArcFace prototype cosine >= ``merge_cosine`` within ``merge_gap_frames`` ;
   * continuity: they meet within ``continuity_gap_frames`` at a plausible position/size, nobody else is there, and the
     cosine is >= ``merge_weak_cosine`` (or >= ``merge_turn_cosine`` when either side meets the other in a steep turn, where the
     embedding legitimately collapses).

3. **Clustering.** Links are applied strongest first (union-find), skipping any union that would put two cannot-link tracklets
   (or two clusters that co-occur) together.
4. **Canonical IDs.** Clusters are numbered 1, 2, ... by descending trusted-observation count, then first frame: the main
   subject is ``Face 1``. The result is deterministic for a given input.

Nothing here reads pixels; the input is the tracker's own output. Every decision is recorded for the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from roop_3d.engine.detection import track_policy as tp
from roop_3d.engine.detection.track_policy import TrackPolicy

YAW_BIN = tp.YAW_BIN_DEGREES
ARTIFACT_MIN_COOCCUR = 0.6          # share of a tracklet's frames that must be inside a larger face to call it a part of it
BOUNDARY_MARGIN_FRAMES = 5
NEIGHBOUR_FACE_SIZES = 3.0


# ------------------------------------------------------------------------------------------------ tracklets
@dataclass
class Tracklet:
    """Everything one raw tracker ID observed, as arrays (frames ascending)."""
    track_id: int
    frames: np.ndarray                  # (n,) int frame numbers
    boxes: np.ndarray                   # (n, 4) float
    embeddings: np.ndarray              # (n, 512) float32, unit
    yaws: np.ndarray                    # (n,)
    trusted: np.ndarray                 # (n,) bool: the tracker allowed this observation (embedding is real evidence)
    real: np.ndarray                    # (n,) bool: a detection of its own (not a low-score continuation)
    scenes: np.ndarray | None = None    # (n,) int shot number of each observation
    bins: dict[int, tuple[np.ndarray, int]] = field(default_factory=dict)
    centroid: np.ndarray | None = None

    @property
    def first(self) -> int:
        return int(self.frames[0])

    @property
    def last(self) -> int:
        return int(self.frames[-1])

    @property
    def support(self) -> int:
        return int(self.real.sum())

    @property
    def trusted_count(self) -> int:
        return int(self.trusted.sum())

    def finalize(self) -> "Tracklet":
        evidence = self.trusted if self.trusted.any() else self.real if self.real.any() else np.ones(len(self.frames), bool)
        bins: dict[int, list[np.ndarray]] = {}
        for embedding, yaw in zip(self.embeddings[evidence], self.yaws[evidence]):
            bins.setdefault(tp.yaw_bin(float(yaw)), []).append(embedding)
        self.bins = {key: (tp.unit(np.mean(rows, axis=0).astype(np.float32)), len(rows)) for key, rows in bins.items()}
        frontal = [self.bins[key][0] * self.bins[key][1] for key in self.bins if abs(key) <= 1]
        pool = frontal or [vector * count for vector, count in self.bins.values()]
        self.centroid = tp.unit(np.sum(pool, axis=0).astype(np.float32)) if pool else None
        return self


class TrackletRecorder:
    """Collects the tracker's per-frame output; ``tracklets()`` hands the arrays to :func:`consolidate`."""

    def __init__(self, max_tracks: int = 256):
        self._rows: dict[int, dict[str, list]] = {}
        self.max_tracks = max_tracks
        self.overflow = False

    def observe(self, frame: int, faces: Iterable) -> None:
        for face in faces:
            rows = self._rows.get(face.track_id)
            if rows is None:
                if len(self._rows) >= self.max_tracks:
                    self.overflow = True
                    continue
                rows = self._rows[face.track_id] = {"frame": [], "box": [], "emb": [], "yaw": [], "trusted": [], "real": [],
                                                    "scene": []}
            observation = face.observation
            rows["frame"].append(int(frame))
            rows["box"].append(np.asarray(observation.box, np.float64))
            rows["emb"].append(np.asarray(observation.embedding, np.float32))
            yaw = float(observation.pose[0])
            rows["yaw"].append(yaw if np.isfinite(yaw) else 0.0)
            low = face.reason == "LOW_SCORE_CONTINUITY"
            rows["trusted"].append(bool(face.swap_allowed) and not low)
            rows["real"].append(not low)
            rows["scene"].append(int(face.scene_id))

    def tracklets(self) -> list[Tracklet]:
        result = []
        for track_id in sorted(self._rows):
            rows = self._rows[track_id]
            order = np.argsort(rows["frame"], kind="stable")
            result.append(Tracklet(
                track_id, np.asarray(rows["frame"], np.int64)[order], np.asarray(rows["box"], np.float64)[order],
                np.asarray(rows["emb"], np.float32)[order], np.asarray(rows["yaw"], np.float64)[order],
                np.asarray(rows["trusted"], bool)[order], np.asarray(rows["real"], bool)[order],
                np.asarray(rows["scene"], np.int64)[order]).finalize())
        return result


# ------------------------------------------------------------------------------------------------ results
@dataclass(frozen=True)
class Decision:
    """Why two tracklets were (not) merged."""
    left: int
    right: int
    merged: bool
    reason: str
    cosine: float | None = None
    gap_frames: int | None = None


@dataclass
class Identity:
    canonical_id: int
    raw_ids: tuple[int, ...]
    first_frame: int
    last_frame: int
    observations: int
    trusted: int
    centroid: np.ndarray | None
    bins: dict[int, np.ndarray]
    anchors: list[np.ndarray]


@dataclass
class Consolidation:
    canonical_of: dict[int, int | None]          # raw tracker id -> canonical id (None: artifact / noise, no ID)
    identities: list[Identity]
    suppressed: dict[int, str]                    # raw id -> reason
    decisions: list[Decision]

    def canonical(self, raw_id: int) -> int | None:
        return self.canonical_of.get(raw_id)

    @property
    def unique_ids(self) -> int:
        return len(self.identities)


# ------------------------------------------------------------------------------------------------ geometry helpers
def _area(box) -> float:
    return max(float(box[2] - box[0]), 0.0) * max(float(box[3] - box[1]), 0.0)


def _contained(inner, outer) -> float:
    """Share of ``inner``'s area that lies inside ``outer``."""
    width = min(inner[2], outer[2]) - max(inner[0], outer[0])
    height = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = _area(inner)
    return 0.0 if width <= 0 or height <= 0 or area <= 0 else float(width * height / area)


def _center(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def _size(box) -> float:
    return max(float(np.sqrt(_area(box))), 1e-6)


def _shared_frames(left: Tracklet, right: Tracklet) -> np.ndarray:
    return np.intersect1d(left.frames, right.frames, assume_unique=False)


def _is_artifact(tracklet: Tracklet, others: list[Tracklet], policy: TrackPolicy) -> bool:
    """True when this tracklet's boxes mostly lie inside a much larger face that is present on the same frames."""
    for other in others:
        if other.track_id == tracklet.track_id:
            continue
        shared = _shared_frames(tracklet, other)
        if len(shared) < max(1, ARTIFACT_MIN_COOCCUR * len(tracklet.frames)):
            continue
        mine = dict(zip(tracklet.frames.tolist(), tracklet.boxes))
        theirs = dict(zip(other.frames.tolist(), other.boxes))
        inside = sum(1 for frame in shared.tolist()
                     if _contained(mine[frame], theirs[frame]) >= policy.artifact_containment
                     and _area(mine[frame]) <= policy.artifact_area_ratio * _area(theirs[frame]))
        if inside >= ARTIFACT_MIN_COOCCUR * len(shared):
            return True
    return False


def _boundary(left: Tracklet, right: Tracklet) -> tuple[int, int, int]:
    """(gap in frames, index in ``left``, index in ``right``) of the closest pair of observations in time."""
    positions = np.searchsorted(left.frames, right.frames)
    best = None
    for j, position in enumerate(positions):
        for i in (position - 1, position):
            if 0 <= i < len(left.frames):
                gap = abs(int(right.frames[j]) - int(left.frames[i]))
                if best is None or gap < best[0]:
                    best = (gap, i, j)
    return best


def _prototype_cosine(left: Tracklet, right: Tracklet) -> tuple[float, int]:
    """Best cosine between prototypes of the same (or adjacent) yaw bin, and the number of samples that backed it."""
    best, support = -1.0, 0
    for key_left, (vector_left, count_left) in left.bins.items():
        for key_right, (vector_right, count_right) in right.bins.items():
            if abs(key_left - key_right) <= 1:
                cosine = float(np.dot(vector_left, vector_right))
                if cosine > best:
                    best, support = cosine, min(count_left, count_right)
    return best, support


def _seamless(left: Tracklet, right: Tracklet, i: int, j: int, gap: int, policy: TrackPolicy) -> bool:
    """Hand-over within a few frames of the same shot at (nearly) the same box."""
    if gap > policy.seamless_gap_frames:
        return False
    if left.scenes is not None and right.scenes is not None and left.scenes[i] != right.scenes[j]:
        return False
    a, b = left.boxes[i], right.boxes[j]
    width, height = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return False
    inter = width * height
    return inter / (_area(a) + _area(b) - inter) >= policy.seamless_iou


def _frontal_conflict(left: Tracklet, right: Tracklet, policy: TrackPolicy) -> float | None:
    """Cosine of the two frontal prototypes when both are well supported and disagree (certainly two people)."""
    a, b = left.bins.get(0), right.bins.get(0)
    if a is None or b is None or min(a[1], b[1]) < 5:
        return None
    cosine = float(np.dot(a[0], b[0]))
    return cosine if cosine < policy.veto_cosine else None


def _others_near(boundary_boxes, window: tuple[int, int], others: list[Tracklet]) -> bool:
    """Another tracklet is on the scene near either boundary box during the window (the link would be ambiguous)."""
    low, high = window[0] - BOUNDARY_MARGIN_FRAMES, window[1] + BOUNDARY_MARGIN_FRAMES
    for other in others:
        inside = (other.frames >= low) & (other.frames <= high)
        if not inside.any():
            continue
        for box in other.boxes[inside]:
            for reference in boundary_boxes:
                if np.linalg.norm(_center(box) - _center(reference)) <= NEIGHBOUR_FACE_SIZES * max(_size(box), _size(reference)):
                    return True
    return False


def _continuity(left: Tracklet, right: Tracklet, policy: TrackPolicy, i: int, j: int, gap: int) -> bool:
    """Same place, same scale, soon after: a plausible hand-over between two tracker IDs."""
    if gap > policy.continuity_gap_frames:
        return False
    a, b = left.boxes[i], right.boxes[j]
    scale = np.sqrt(_size(a) * _size(b))
    shift = float(np.linalg.norm(_center(a) - _center(b)) / scale)
    ratio = _size(b) / _size(a)
    allowed_shift = min(0.8 + 0.12 * gap, 4.0)
    allowed_ratio = min(1.6 + 0.04 * gap, 3.0)
    return shift <= allowed_shift and 1.0 / allowed_ratio <= ratio <= allowed_ratio


def _steep(left: Tracklet, right: Tracklet, i: int, j: int, policy: TrackPolicy) -> bool:
    return abs(float(left.yaws[i])) >= policy.turn_yaw or abs(float(right.yaws[j])) >= policy.turn_yaw


# ------------------------------------------------------------------------------------------------ the pass
def consolidate(tracklets: Iterable[Tracklet], policy: TrackPolicy | None = None) -> Consolidation:
    policy = policy or TrackPolicy()
    tracklets = [item for item in tracklets if len(item.frames)]
    decisions: list[Decision] = []
    suppressed: dict[int, str] = {}

    # 1. artifacts and noise ---------------------------------------------------------------------------------------
    survivors = []
    for item in tracklets:
        if item.support < policy.min_identity_frames:
            suppressed[item.track_id] = f"fewer than {policy.min_identity_frames} detections ({item.support})"
        elif _is_artifact(item, tracklets, policy):
            suppressed[item.track_id] = "box lies inside a larger face on the same frames (part of that face)"
        else:
            survivors.append(item)

    # 2. candidate links -------------------------------------------------------------------------------------------
    parent = {item.track_id: item.track_id for item in survivors}
    by_id = {item.track_id: item for item in survivors}

    def find(key: int) -> int:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    members: dict[int, list[int]] = {item.track_id: [item.track_id] for item in survivors}
    cannot: set[frozenset[int]] = set()
    links: list[tuple[float, int, int, str, float, int]] = []
    for index, left in enumerate(survivors):
        for right in survivors[index + 1:]:
            pair = frozenset((left.track_id, right.track_id))
            if len(_shared_frames(left, right)):
                cannot.add(pair)
                decisions.append(Decision(left.track_id, right.track_id, False, "seen together on the same frame"))
                continue
            gap, i, j = _boundary(left, right)
            window = (min(int(left.frames[i]), int(right.frames[j])), max(int(left.frames[i]), int(right.frames[j])))
            others = [item for item in survivors if item.track_id not in pair]
            if _seamless(left, right, i, j, gap, policy) and not _others_near((left.boxes[i], right.boxes[j]), window, others):
                # The same box on the next frame of the same shot, nobody else around: one object, whatever the embedding says
                # (a rendered, rolled or heavily degraded face can change its ArcFace mode without moving).
                cosine, _ = _prototype_cosine(left, right)
                links.append((2.0 - 0.001 * gap, left.track_id, right.track_id, "seamless", float(cosine), int(gap)))
                continue
            conflict = _frontal_conflict(left, right, policy)
            if conflict is not None:
                cannot.add(pair)
                decisions.append(Decision(left.track_id, right.track_id, False,
                                          "frontal appearance disagrees", float(conflict)))
                continue
            cosine, _ = _prototype_cosine(left, right)
            if cosine == -1.0 and left.centroid is not None and right.centroid is not None:
                cosine = float(np.dot(left.centroid, right.centroid))
            reason = None
            if cosine >= policy.merge_cosine and gap <= policy.merge_gap_frames:
                reason = "appearance"
            elif _continuity(left, right, policy, i, j, gap):
                if not _others_near((left.boxes[i], right.boxes[j]), window, others):
                    floor = policy.merge_turn_cosine if _steep(left, right, i, j, policy) else policy.merge_weak_cosine
                    if cosine >= floor:
                        reason = "continuity" if floor == policy.merge_weak_cosine else "continuity (turn)"
            if reason is None:
                decisions.append(Decision(left.track_id, right.track_id, False, "no sufficient evidence",
                                          float(cosine), int(gap)))
                continue
            # Strongest first: appearance evidence, then closeness in time.
            links.append((float(cosine) - 0.001 * min(gap, 500) + (0.5 if reason == "appearance" else 0.0),
                          left.track_id, right.track_id, reason, float(cosine), int(gap)))

    # 3. union-find under cannot-link / co-occurrence ---------------------------------------------------------------
    frames_of = {key: set(by_id[key].frames.tolist()) for key in by_id}
    cluster_frames = dict(frames_of)
    for _, a, b, reason, cosine, gap in sorted(links, key=lambda item: (-item[0], item[1], item[2])):
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            continue
        blocked = (cluster_frames[root_a] & cluster_frames[root_b]
                   or any(frozenset((x, y)) in cannot for x in members[root_a] for y in members[root_b]))
        if blocked:
            decisions.append(Decision(a, b, False, "would merge people seen together", cosine, gap))
            continue
        parent[root_b] = root_a
        members[root_a] += members.pop(root_b)
        cluster_frames[root_a] |= cluster_frames.pop(root_b)
        decisions.append(Decision(a, b, True, reason, cosine, gap))

    # 4. canonical numbering ---------------------------------------------------------------------------------------
    clusters = []
    for root, raw_ids in members.items():
        group = [by_id[key] for key in raw_ids]
        trusted = sum(item.trusted_count for item in group)
        clusters.append((-trusted, min(item.first for item in group), sorted(raw_ids), group))
    clusters.sort(key=lambda item: (item[0], item[1], item[2]))
    canonical_of: dict[int, int | None] = {item.track_id: None for item in tracklets}
    identities = []
    for number, (neg_trusted, first, raw_ids, group) in enumerate(clusters, start=1):
        for key in raw_ids:
            canonical_of[key] = number
        identities.append(_identity(number, raw_ids, group))
    return Consolidation(canonical_of, identities, suppressed, decisions)


def _identity(number: int, raw_ids: list[int], group: list[Tracklet], anchor_limit: int = 8) -> Identity:
    pooled: dict[int, list[tuple[np.ndarray, int]]] = {}
    for item in group:
        for key, (vector, count) in item.bins.items():
            pooled.setdefault(key, []).append((vector, count))
    bins = {key: tp.unit(np.sum([vector * count for vector, count in rows], axis=0).astype(np.float32))
            for key, rows in pooled.items()}
    frontal = [bins[key] * sum(count for _, count in pooled[key]) for key in bins if abs(key) <= 1]
    pool = frontal or [bins[key] * sum(count for _, count in pooled[key]) for key in bins]
    centroid = tp.unit(np.sum(pool, axis=0).astype(np.float32)) if pool else None
    anchors: list[np.ndarray] = []

    def offer(vector: np.ndarray) -> None:
        if len(anchors) < anchor_limit and all(float(np.dot(anchor, vector)) < 0.85 for anchor in anchors):
            anchors.append(vector)

    for key in sorted(bins, key=lambda k: (abs(k), k)):          # frontal first, then each turned view
        offer(bins[key])
    # Tracker IDs of one person can carry different embedding modes (a rolled or rendered head): keep each mode's prototypes.
    for item in sorted(group, key=lambda t: -t.trusted_count):
        for key in sorted(item.bins, key=lambda k: (abs(k), k)):
            offer(item.bins[key][0])
    return Identity(number, tuple(raw_ids), min(item.first for item in group), max(item.last for item in group),
                    sum(item.support for item in group), sum(item.trusted_count for item in group), centroid, bins, anchors)
