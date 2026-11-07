"""Rank-preserved person clusters and the Target Person Reference Bank (Stage 12).

A single 2-D frame embedding stops matching when the target turns from 0 deg to 65 deg yaw (measured on real FaceSets:
same-person cosine falls from 0.92 at < 20 deg to 0.49-0.61 at 50-90 deg, while different people reach 0.48). So a locked
target is a *bank* of embeddings at different poses, and a face matches the best bank entry, not one anchor.

"Rank-preserved": a person keeps ONE cluster id for the whole video whatever its left-to-right order in a frame. Ids are
never derived from the x-order. The left-to-right rank is reported separately (``face_index``).

Association rule (per frame, greedy by descending score, one face per cluster and one cluster per face):

* accept when cosine >= ``STRONG`` (0.55), OR
* cosine >= ``WEAK`` (0.40) and it beats the next-best alternative for that face AND for that cluster by ``MARGIN`` (0.10),
  where the alternative is never taken below ``IMPOSTOR_FLOOR`` (0.30) - the pose-bridging case, where a profile face only
  resembles the bank weakly but nobody else in the frame does.
* otherwise the face starts a new cluster. Fragmenting a person is preferred to merging two people.

The bank grows by chain linking: keyframes are visited outward from the anchor frame in time, and an accepted face joins the
bank when it adds pose coverage, so 0 -> 30 -> 65 deg links transitively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
import time
import uuid
from typing import Any

import numpy as np

STRONG, WEAK, MARGIN = 0.55, 0.40, 0.10
IMPOSTOR_FLOOR = 0.30          # measured: different people average 0.13 (p99 0.37); a lone weak match must clear this by MARGIN
MAX_BANK = 12
MIN_POSE_GAP_DEG = 12.0


def _unit(v) -> np.ndarray:
    a = np.asarray(v, np.float32).reshape(-1)
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else a


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

    def public(self) -> dict:
        return {"yaw": round(self.yaw, 2), "pitch": round(self.pitch, 2), "roll": round(self.roll, 2),
                "frame_index": self.frame_index, "bbox": [round(v, 2) for v in self.bbox], "source": self.source,
                "match_score": round(self.score, 4)}


@dataclass
class PersonCluster:
    cluster_id: int
    bank: list[BankEntry] = field(default_factory=list)
    locked: bool = False
    seen: dict[int, list[float]] = field(default_factory=dict)       # frame_index -> bbox (trajectory samples)

    def score(self, embedding: np.ndarray) -> tuple[float, BankEntry | None]:
        if not self.bank:
            return -1.0, None
        sims = np.array([float(e.embedding @ embedding) for e in self.bank])
        k = int(np.argmax(sims))
        return float(sims[k]), self.bank[k]

    def consider(self, entry: BankEntry, *, force: bool = False) -> bool:
        """Add to the bank when it widens pose coverage (or force). Returns whether it was added."""
        near = [e for e in self.bank if abs(e.yaw - entry.yaw) < MIN_POSE_GAP_DEG and abs(e.pitch - entry.pitch) < MIN_POSE_GAP_DEG]
        if near and not force:
            return False
        if len(self.bank) >= MAX_BANK:
            if not force:
                return False
            auto = [e for e in self.bank if e.source == "auto"]
            if auto:
                self.bank.remove(auto[0])
        self.bank.append(entry)
        return True


@dataclass
class FaceRef:
    """What the clustering needs of one detected face."""
    embedding: np.ndarray | None
    yaw: float
    pitch: float
    roll: float
    bbox: list[float]


@dataclass
class Association:
    cluster_id: int | None
    score: float
    margin: float
    reason: str                    # strong | pose_bridge | new_cluster | no_embedding
    entry: BankEntry | None = None


class ClusterRegistry:
    """All person clusters of one video."""

    def __init__(self) -> None:
        self.clusters: dict[int, PersonCluster] = {}
        self.next_id = 1
        self.lock = threading.RLock()

    def _new(self, ref: FaceRef, frame_index: int) -> PersonCluster:
        cluster = PersonCluster(self.next_id)
        self.next_id += 1
        self.clusters[cluster.cluster_id] = cluster
        cluster.bank.append(BankEntry(_unit(ref.embedding), ref.yaw, ref.pitch, ref.roll, frame_index, ref.bbox, "auto"))
        return cluster

    def associate(self, refs: list[FaceRef], frame_index: int, *, learn: bool = True,
                  only: set[int] | None = None) -> list[Association]:
        """Assign every face of one frame to a cluster (creating clusters for unmatched faces)."""
        with self.lock:
            out: list[Association | None] = [None] * len(refs)
            valid = [i for i, r in enumerate(refs) if r.embedding is not None]
            for i, r in enumerate(refs):
                if r.embedding is None:
                    out[i] = Association(None, 0.0, 0.0, "no_embedding")
            emb = {i: _unit(refs[i].embedding) for i in valid}
            ids = [c for c in self.clusters if only is None or c in only]
            sims = np.full((len(refs), len(ids)), -1.0)
            entries: dict[tuple[int, int], BankEntry | None] = {}
            for i in valid:
                for j, cid in enumerate(ids):
                    sims[i, j], entries[(i, j)] = self.clusters[cid].score(emb[i])
            pairs = sorted(((sims[i, j], i, j) for i in valid for j in range(len(ids))), reverse=True)
            used_f: set[int] = set()
            used_c: set[int] = set()
            for s, i, j in pairs:
                if i in used_f or j in used_c or s < WEAK:
                    continue
                other_c = [sims[i, k] for k in range(len(ids)) if k != j]
                other_f = [sims[k, j] for k in valid if k != i]
                margin = float(s - max(other_c + other_f + [IMPOSTOR_FLOOR]))
                if s >= STRONG:
                    reason = "strong"
                elif margin >= MARGIN:
                    reason = "pose_bridge"
                else:
                    continue
                used_f.add(i)
                used_c.add(j)
                out[i] = Association(ids[j], float(s), margin, reason, entries[(i, j)])
            for i in valid:
                if out[i] is None:
                    cluster = self._new(refs[i], frame_index)
                    out[i] = Association(cluster.cluster_id, 1.0, 1.0, "new_cluster", cluster.bank[0])
                    continue
                cluster = self.clusters[out[i].cluster_id]
                cluster.seen[frame_index] = list(refs[i].bbox)
                if learn and out[i].reason != "new_cluster":
                    cluster.consider(BankEntry(_unit(refs[i].embedding), refs[i].yaw, refs[i].pitch, refs[i].roll, frame_index,
                                               refs[i].bbox, "auto", out[i].score))
            return [a for a in out if a is not None]


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
        cluster = registry.clusters[self.cluster_id]
        return {"lock_id": self.lock_id, "target_number": self.target_number, "label": f"Target #{self.target_number}",
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

    def create(self, media_path: str, frame_index: int, ref: FaceRef) -> TargetIdentityLock:
        """Lock the clicked face: it joins the cluster it already belongs to (or starts one) and that cluster is flagged."""
        registry = self.registry(media_path)
        with registry.lock, self._lock:
            assoc = registry.associate([ref], frame_index, learn=False)[0]
            if assoc.cluster_id is None:
                raise ValueError("The clicked face has no embedding")
            cluster = registry.clusters[assoc.cluster_id]
            cluster.locked = True
            # the anchor replaces the identical auto entry created when this frame was first inspected
            cluster.bank = [e for e in cluster.bank if not (e.frame_index == frame_index and float(e.embedding @ _unit(ref.embedding)) > 0.97)]
            cluster.consider(BankEntry(_unit(ref.embedding), ref.yaw, ref.pitch, ref.roll, frame_index, ref.bbox, "anchor"), force=True)
            existing = [l for l in self._locks.values() if l.media_path == media_path and l.cluster_id == cluster.cluster_id]
            if existing:
                return existing[0]
            number = 1 + max([l.target_number for l in self._locks.values() if l.media_path == media_path], default=0)
            lock = TargetIdentityLock(uuid.uuid4().hex, number, media_path, cluster.cluster_id, frame_index, list(ref.bbox),
                                      (ref.yaw, ref.pitch, ref.roll))
            self._locks[lock.lock_id] = lock
            return lock
