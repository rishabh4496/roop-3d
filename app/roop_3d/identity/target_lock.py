"""Persistent per-video identity slots and quality-ranked 5 x 3 pose banks."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
import uuid
from typing import Any

import numpy as np

STRONG = 0.55
IMPOSTOR_FLOOR = 0.30  # diagnostic margin baseline; assignment uses MIN_COSINE
MIN_COSINE = 0.42
BUCKET_QUALITY_THRESHOLD = 100.0


def _unit(v) -> np.ndarray:
    a = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


def pose_bucket(yaw: float, pitch: float) -> tuple[int, int] | None:
    """Half-open bins; outer endpoints included. Out-of-grid poses do not claim coverage."""
    if not np.isfinite([yaw, pitch]).all() or not (-75 <= yaw <= 75 and -45 <= pitch <= 45):
        return None
    return min(4, int((yaw + 75) // 30)), min(2, int((pitch + 45) // 30))


class KalmanBox:
    """Constant velocity Kalman filter in frame coordinates, safe for backward seeks."""
    def __init__(self, bbox, frame):
        self.x = np.r_[np.asarray(bbox, float), np.zeros(4)]
        self.p = np.eye(8) * 10
        self.frame = frame

    def predict(self, frame):
        dt = frame - self.frame
        # Distant seeks/re-entry carry no useful motion evidence.
        if dt <= 0 or dt > 30:
            return self.x[:4].copy()
        f = np.eye(8)
        f[:4, 4:] = np.eye(4) * dt
        return (f @ self.x)[:4]

    def update(self, bbox, frame):
        if frame <= self.frame:
            return
        dt = frame - self.frame
        if dt > 30:
            self.__init__(bbox, frame)
            return
        f = np.eye(8)
        f[:4, 4:] = np.eye(4) * dt
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + np.eye(8)
        k = np.linalg.solve(self.p[:4, :4] + np.eye(4) * 4, self.p[:4, :]).T
        self.x += k @ (np.asarray(bbox) - self.x[:4])
        self.p = (np.eye(8) - k @ np.eye(8)[:4]) @ self.p
        self.frame = frame


@dataclass
class BankEntry:
    embedding: np.ndarray
    yaw: float                     # standard frame, degrees
    pitch: float
    roll: float
    frame_index: int
    bbox: list[float]
    source: str                    # anchor | auto | user
    score: float = 1.0

    quality: float = 0.0
    crop: np.ndarray | None = field(default=None, repr=False)
    landmarks3d: np.ndarray | None = field(default=None, repr=False)
    occlusion_ratio: float | None = None
    pose6dof: dict | None = None

    def public(self) -> dict:
        return {"bucket": pose_bucket(self.yaw, self.pitch), "quality": round(self.quality, 3),
                "quality_locked": self.quality >= BUCKET_QUALITY_THRESHOLD,
                "has_crop": self.crop is not None, "has_landmarks3d": self.landmarks3d is not None,
                "pose6dof": self.pose6dof, "occlusion_ratio": self.occlusion_ratio, "yaw": round(self.yaw, 2), "pitch": round(self.pitch, 2), "roll": round(self.roll, 2),
                "frame_index": self.frame_index, "bbox": [round(v, 2) for v in self.bbox], "source": self.source,
                "match_score": round(self.score, 4)}


@dataclass
class PersonCluster:
    cluster_id: int
    bank: list[BankEntry] = field(default_factory=list)
    locked: bool = False
    track_id: str | None = None
    seen: dict[int, list[float]] = field(default_factory=dict)       # frame_index -> bbox (trajectory samples)

    motion: KalmanBox | None = field(default=None, repr=False)
    anchor_embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def face_id(self) -> int:
        return self.cluster_id - 1

    @property
    def centroid(self) -> np.ndarray:
        return _unit(np.mean([e.embedding for e in self.bank], axis=0))

    def score(self, embedding: np.ndarray) -> tuple[float, BankEntry | None]:
        if not self.bank:
            return -1.0, None
        embedding = _unit(embedding)
        sims = np.array([float(e.embedding @ embedding) for e in self.bank])
        k = int(np.argmax(sims))
        return max(float(sims[k]), float(self.centroid @ embedding),
                   float(self.anchor_embedding @ embedding) if self.anchor_embedding is not None else -1.), self.bank[k]

    def consider(self, entry: BankEntry, *, force: bool = False) -> bool:
        """Keep the best measurement per angular bucket; retain one out-of-grid anchor."""
        entry.embedding = _unit(entry.embedding)
        bucket = pose_bucket(entry.yaw, entry.pitch)
        near = [e for e in self.bank if pose_bucket(e.yaw, e.pitch) == bucket]
        if near:
            previous = near[0]
            if not force and entry.quality <= previous.quality:
                return False
            self.bank = [e for e in self.bank if e is not previous]
        self.bank.append(entry)
        return True

    def reference_for_pose(self, yaw: float, pitch: float) -> BankEntry | None:
        """Exact bucket when observed, otherwise nearest measured pose (never invent coverage)."""
        bucket = pose_bucket(yaw, pitch)
        candidates = [e for e in self.bank if bucket is not None and pose_bucket(e.yaw, e.pitch) == bucket]
        return min(candidates or self.bank,
                   key=lambda e: ((e.yaw-yaw)**2 + (e.pitch-pitch)**2, -e.quality, e.frame_index), default=None)

    def coverage(self) -> dict:
        entries = [e for e in self.bank if pose_bucket(e.yaw, e.pitch) is not None]
        locked = sum(e.quality >= BUCKET_QUALITY_THRESHOLD for e in entries)
        return {"captured": len(entries), "quality_locked": locked, "total": 15,
                "buckets": [e.public() for e in entries]}


@dataclass
class FaceRef:
    """What the clustering needs of one detected face."""
    embedding: np.ndarray | None
    yaw: float
    pitch: float
    roll: float
    bbox: list[float]
    quality: float = 0.0
    crop: np.ndarray | None = None
    landmarks3d: np.ndarray | None = None
    occlusion_ratio: float | None = None
    pose6dof: dict | None = None

    def entry(self, frame_index: int, source="auto", score=1.0) -> BankEntry:
        return BankEntry(_unit(self.embedding), self.yaw, self.pitch, self.roll, frame_index,
                         list(self.bbox), source, score, self.quality, self.crop, self.landmarks3d, self.occlusion_ratio, self.pose6dof)


@dataclass
class Association:
    cluster_id: int | None
    score: float
    margin: float
    reason: str                    # strong | pose_bridge | new_cluster | no_embedding
    entry: BankEntry | None = None


class FaceTrackRegistry:
    """All person clusters of one video."""

    def __init__(self) -> None:
        self.clusters: dict[int, PersonCluster] = {}
        self.next_id = 1
        self.lock = threading.RLock()
        self.scan_lock = threading.Lock()

    def _new(self, ref: FaceRef, frame_index: int) -> PersonCluster:
        cluster = PersonCluster(self.next_id)
        self.next_id += 1
        self.clusters[cluster.cluster_id] = cluster
        cluster.bank.append(ref.entry(frame_index))
        cluster.motion = KalmanBox(ref.bbox, frame_index)
        cluster.track_id = f"track_{cluster.face_id:04d}"
        cluster.seen[frame_index] = list(ref.bbox)
        return cluster

    def associate(self, refs: list[FaceRef], frame_index: int, *, learn: bool = True,
                  only: set[int] | None = None) -> list[Association]:
        """Assign every face of one frame to a cluster (creating clusters for unmatched faces)."""
        with self.lock:
            out: list[Association | None] = [None] * len(refs)
            valid = [i for i, r in enumerate(refs) if r.embedding is not None
                     and np.isfinite(r.embedding).all() and np.linalg.norm(r.embedding) > 0]
            for i, r in enumerate(refs):
                if i not in valid:
                    out[i] = Association(None, 0.0, 0.0, "no_embedding")
            emb = {i: _unit(refs[i].embedding) for i in valid}
            ids = [c for c in self.clusters if only is None or c in only]
            sims = np.full((len(refs), len(ids)), -1.0)
            entries: dict[tuple[int, int], BankEntry | None] = {}
            for i in valid:
                for j, cid in enumerate(ids):
                    sims[i, j], entries[(i, j)] = self.clusters[cid].score(emb[i])
            from scipy.optimize import linear_sum_assignment
            from roop_3d.media.detector import iou
            # Dummy columns allow unmatched observations without displacing valid matches.
            costs = np.full((len(valid), len(ids) + len(valid)), 2.0)
            for row, i in enumerate(valid):
                for j, cid in enumerate(ids):
                    if sims[i, j] < MIN_COSINE:
                        continue
                    cluster = self.clusters[cid]
                    predicted = cluster.motion.predict(frame_index) if cluster.motion else refs[i].bbox
                    costs[row, j] = .70 * (1 - sims[i, j]) + .30 * (1 - iou(np.asarray(refs[i].bbox), np.asarray(predicted)))
            if valid:
                rows, columns = linear_sum_assignment(costs)
                for row, j in zip(rows, columns):
                    if j >= len(ids) or costs[row, j] >= 2:
                        continue
                    i = valid[row]
                    score = float(sims[i, j])
                    alternatives = [sims[i, k] for k in range(len(ids)) if k != j]
                    margin = score - max(alternatives + [IMPOSTOR_FLOOR])
                    out[i] = Association(ids[j], score, margin, "strong" if score >= STRONG else "pose_bridge", entries[(i,j)])
            for i in valid:
                if out[i] is None:
                    cluster = self._new(refs[i], frame_index)
                    out[i] = Association(cluster.cluster_id, 1.0, 1.0, "new_cluster", cluster.bank[0])
                    continue
                cluster = self.clusters[out[i].cluster_id]
                cluster.seen[frame_index] = list(refs[i].bbox)
                if learn and out[i].reason != "new_cluster":
                    cluster.consider(refs[i].entry(frame_index, score=out[i].score))
                if cluster.motion is not None:
                    cluster.motion.update(refs[i].bbox, frame_index)
            return [a for a in out if a is not None]


TargetFaceRegistry = FaceTrackRegistry
ClusterRegistry = FaceTrackRegistry  # compatibility for existing clients


@dataclass
class TargetIdentityLock:
    lock_id: str
    target_number: int
    media_path: str
    cluster_id: int
    anchor_frame_index: int
    anchor_bbox: list[float]
    anchor_pose: tuple[float, float, float]
    created_at: float = field(default_factory=time.time)
    scan: dict[str, Any] = field(default_factory=lambda: {"state": "idle"})

    def public(self, registry: ClusterRegistry) -> dict:
        with registry.lock:
            cluster = registry.clusters[self.cluster_id]
            return {"lock_id": self.lock_id, "target_number": self.target_number, "label": f"face {cluster.face_id}", "face_id": cluster.face_id, "coverage": cluster.coverage(),
                    'track_id': cluster.track_id,
                    "media_path": self.media_path, "cluster_id": self.cluster_id, "anchor_frame_index": self.anchor_frame_index,
                    "anchor_bbox": [round(v, 3) for v in self.anchor_bbox],
                    "anchor_pose": {"yaw": round(self.anchor_pose[0], 2), "pitch": round(self.anchor_pose[1], 2),
                                    "roll": round(self.anchor_pose[2], 2)},
                    "bank_size": len(cluster.bank), "reference_bank": [e.public() for e in cluster.bank],
                    "observed_frames": sorted(cluster.seen), "scan": dict(self.scan)}


class TargetLockStore:
    """Registries per media path and the locks defined on them (in memory; one process)."""

    def __init__(self) -> None:
        self._registries: dict[str, ClusterRegistry] = {}
        self._locks: dict[str, TargetIdentityLock] = {}
        self._lock = threading.RLock()

    def registry(self, media_path: str) -> ClusterRegistry:
        with self._lock:
            return self._registries.setdefault(media_path, ClusterRegistry())

    def locks_for(self, media_path: str) -> list[TargetIdentityLock]:
        with self._lock:
            return sorted((l for l in self._locks.values() if l.media_path == media_path), key=lambda l: l.target_number)

    def get(self, lock_id: str) -> TargetIdentityLock | None:
        with self._lock:
            return self._locks.get(lock_id)

    def remove(self, lock_id: str) -> TargetIdentityLock | None:
        """Unlock a target. Its person cluster (and so its immutable ``face N`` slot and reference bank) is kept: the face
        simply stops being a locked target, and locking it again attaches a new lock to the same cluster."""
        registry_lock = None
        with self._lock:
            lock = self._locks.get(lock_id)
            if lock is None:
                return None
            registry = self._registries.get(lock.media_path)
            registry_lock = registry.lock if registry is not None else None
        if registry_lock is not None:
            with registry_lock, self._lock:
                self._locks.pop(lock_id, None)
                cluster = registry.clusters.get(lock.cluster_id)
                if cluster is not None and not any(l.cluster_id == lock.cluster_id and l.media_path == lock.media_path
                                                   for l in self._locks.values()):
                    cluster.locked = False
        else:
            with self._lock:
                self._locks.pop(lock_id, None)
        lock.scan = {**lock.scan, "state": "cancelled"}
        return lock

    def create(self, media_path: str, frame_index: int, ref: FaceRef, *, cluster_id: int | None = None) -> TargetIdentityLock:
        """Lock the clicked face: it joins the cluster it already belongs to (or starts one) and that cluster is flagged."""
        registry = self.registry(media_path)
        with registry.lock, self._lock:
            if ref.embedding is None or not np.isfinite(ref.embedding).all() or np.linalg.norm(ref.embedding) == 0:
                raise ValueError("The clicked face has no valid embedding")
            if cluster_id is None:
                cluster_id = registry.associate([ref], frame_index, learn=False)[0].cluster_id
            if cluster_id is None or cluster_id not in registry.clusters:
                raise ValueError("The clicked face has no identity assignment")
            cluster = registry.clusters[cluster_id]
            cluster.locked = True
            if cluster.anchor_embedding is None:
                cluster.anchor_embedding = _unit(ref.embedding).copy()
            # the anchor replaces the identical auto entry created when this frame was first inspected
            cluster.bank = [e for e in cluster.bank if not (e.frame_index == frame_index and float(e.embedding @ _unit(ref.embedding)) > 0.97)]
            cluster.consider(ref.entry(frame_index, "anchor"), force=True)
            existing = [l for l in self._locks.values() if l.media_path == media_path and l.cluster_id == cluster.cluster_id]
            if existing:
                return existing[0]
            number = 1 + max([l.target_number for l in self._locks.values() if l.media_path == media_path], default=0)
            lock = TargetIdentityLock(uuid.uuid4().hex, number, media_path, cluster.cluster_id, frame_index, list(ref.bbox),
                                      (ref.yaw, ref.pitch, ref.roll))
            self._locks[lock.lock_id] = lock
            return lock
